"""Focused tests for WHAM-primary, MediaPipe-assisted temporal fusion."""

import warnings

import numpy as np
import pytest

from pose_pipeline.camera import CameraModel
from pose_pipeline.fusion import (
    FusionConfig,
    compute_mediapipe_weights,
    compute_temporal_agreement,
    fuse_wham_mediapipe,
)


def test_visibility_presence_and_score_gate_visible_joints_above_occluded_joints():
    score = np.array([[0.95, 0.15]], dtype=np.float64)
    visibility = np.array([[0.95, 0.10]], dtype=np.float64)
    presence = np.array([[0.90, 0.20]], dtype=np.float64)
    statuses = np.array([["observed", "occluded"]], dtype=object)

    weights = compute_mediapipe_weights(
        score,
        visibility,
        presence,
        statuses=statuses,
        smooth=False,
    )

    assert weights.shape == (1, 2)
    assert weights[0, 0] > weights[0, 1]
    assert weights[0, 1] < 0.1


def test_temporal_agreement_downweights_a_single_frame_outlier():
    points = np.array(
        [
            [[0.50, 0.50]],
            [[0.51, 0.50]],
            [[0.95, 0.95]],
            [[0.53, 0.50]],
            [[0.54, 0.50]],
        ],
        dtype=np.float64,
    )

    agreement = compute_temporal_agreement(points, timestamps=np.arange(5, dtype=np.float64))

    assert agreement.shape == (5, 1)
    assert agreement[2, 0] < agreement[1, 0]
    assert agreement[2, 0] < agreement[3, 0]


def test_temporal_agreement_handles_all_missing_frames_without_runtime_warning():
    """Dropped detector frames must be a clean zero-confidence signal."""
    points = np.array(
        [
            [[0.40, 0.45], [0.60, 0.45]],
            [[np.nan, np.nan], [np.nan, np.nan]],
            [[0.42, 0.45], [0.58, 0.45]],
        ],
        dtype=np.float64,
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        agreement = compute_temporal_agreement(points)

    assert agreement.shape == (3, 2)
    assert np.all(agreement[1] == 0.0)
    assert not any(issubclass(item.category, RuntimeWarning) for item in caught)


def test_low_confidence_mediapipe_cannot_overwrite_valid_wham_depth():
    wham = np.array([[[0.0, 0.8, 0.20]]], dtype=np.float64)
    mediapipe = np.array([[[0.0, 0.8, 2.5]]], dtype=np.float64)

    result = fuse_wham_mediapipe(
        wham,
        mediapipe_3d=mediapipe,
        score=np.array([[0.1]]),
        visibility=np.array([[0.1]]),
        presence=np.array([[0.1]]),
        config=FusionConfig(mediapipe_3d_weight=1.0),
    )

    assert result.joints.shape == (1, 1, 3)
    assert result.joints[0, 0, 2] == pytest.approx(0.20, abs=0.02)
    assert result.diagnostics["mediapipeWeights"][0][0] < 0.01
    assert result.provenance[0][0]["primary"] == "wham"


def test_visible_mediapipe_can_correct_wham_lateral_position_but_not_replace_primary():
    wham = np.array([[[0.0, 0.8, 0.20], [0.0, 0.8, 0.20]]], dtype=np.float64)
    mediapipe = np.array([[[0.7, 0.8, 0.20], [0.7, 0.8, 2.5]]], dtype=np.float64)
    score = np.array([[0.95, 0.10]])

    result = fuse_wham_mediapipe(
        wham,
        mediapipe_3d=mediapipe,
        score=score,
        visibility=score,
        presence=score,
        config=FusionConfig(mediapipe_3d_weight=1.0),
    )

    assert result.joints[0, 0, 0] > 0.0
    assert result.joints[0, 0, 2] == pytest.approx(0.20, abs=0.10)
    assert result.diagnostics["mediapipeWeights"][0][0] > result.diagnostics["mediapipeWeights"][0][1]
    assert result.provenance[0][0]["source"] == "fused"


def test_optional_2d_reprojection_corrects_image_position_while_retaining_wham_depth():
    wham = np.array([[[0.0, 0.0, 0.50]]], dtype=np.float64)
    camera = CameraModel.orthographic(scale=0.5, principal_point=(0.5, 0.5))
    observed_2d = np.array([[[0.75, 0.5]]], dtype=np.float64)

    result = fuse_wham_mediapipe(
        wham,
        mediapipe_2d=observed_2d,
        score=np.array([[0.95]]),
        visibility=np.array([[0.95]]),
        presence=np.array([[0.95]]),
        camera=camera,
        config=FusionConfig(reprojection_weight=1.0, mediapipe_3d_weight=0.0),
    )

    assert result.joints[0, 0, 0] > 0.0
    assert result.joints[0, 0, 2] == pytest.approx(0.50, abs=1e-6)
    assert result.diagnostics["meanReprojectionError"] < 0.25


def test_default_camera_uses_orthographic_for_root_relative_wham_depth():
    """Root-relative WHAM depth must not trigger a singular perspective fit."""
    wham = np.array(
        [
            [[-0.30, 0.40, -0.30], [0.30, 0.40, 0.30], [-0.20, -0.40, -0.20], [0.20, -0.40, 0.20]],
            [[-0.28, 0.42, -0.28], [0.32, 0.42, 0.32], [-0.18, -0.38, -0.18], [0.22, -0.38, 0.22]],
        ],
        dtype=np.float64,
    )
    reference_camera = CameraModel.orthographic(scale=0.4, principal_point=(0.5, 0.5))
    observed_2d = np.stack([reference_camera.project(frame) for frame in wham], axis=0)
    scores = np.full((len(wham), wham.shape[1]), 0.95, dtype=np.float64)

    result = fuse_wham_mediapipe(
        wham,
        mediapipe_2d=observed_2d,
        score=scores,
        visibility=scores,
        presence=scores,
        config=FusionConfig(mediapipe_3d_weight=0.0, reprojection_weight=0.5),
    )

    assert result.metadata["fusion"]["cameraModel"] == "orthographic"
    assert result.diagnostics["meanReprojectionError"] < 0.05
    assert result.diagnostics["meanWhamCorrection"] < 0.05


def test_json_point_records_are_accepted_as_fusion_inputs():
    result = fuse_wham_mediapipe(
        [{"x": 0.0, "y": 0.0, "z": 0.5}],
        mediapipe_3d=[{"x": 0.1, "y": 0.0, "z": 0.5}],
        score=[0.9],
        visibility=[0.9],
        presence=[0.9],
    )

    assert result.joints.shape == (1, 3)
    assert result.joints[0, 0] > 0.0


def test_valid_2d_anchor_survives_missing_mediapipe_3d_joint():
    wham = np.array([[[0.0, 0.0, 0.5]]], dtype=np.float64)
    observed_2d = np.array([[[0.75, 0.5]]], dtype=np.float64)
    missing_3d = np.array([[[np.nan, np.nan, np.nan]]], dtype=np.float64)
    camera = CameraModel.orthographic(scale=0.5, principal_point=(0.5, 0.5))

    result = fuse_wham_mediapipe(
        wham,
        mediapipe_3d=missing_3d,
        mediapipe_2d=observed_2d,
        score=np.array([[0.95]]),
        visibility=np.array([[0.95]]),
        presence=np.array([[0.95]]),
        camera=camera,
        config=FusionConfig(mediapipe_3d_weight=1.0, reprojection_weight=1.0),
    )

    assert result.joints[0, 0, 0] > 0.0
    assert result.joints[0, 0, 2] == pytest.approx(0.5, abs=1e-6)


def test_result_has_explicit_canonical_metadata_and_json_safe_diagnostics():
    result = fuse_wham_mediapipe(
        np.array([[[0.0, 0.0, 0.1]]], dtype=np.float32),
        config=FusionConfig(world_motion_available=False),
    )

    assert result.metadata["coordinateSystem"] == "smplx"
    assert result.metadata["backendActual"] == "wham"
    assert result.metadata["fusionMode"] == "wham_mediapipe"
    payload = result.to_dict()
    assert payload["backendMetadata"]["landmarkAxisPolicy"]["canonical"] == "smplx_y_up_z_forward"
    assert isinstance(payload["diagnostics"]["meanWhamCorrection"], float)
