"""Integration coverage for the explicit WHAM + MediaPipe extraction mode."""

import json
import os
import subprocess
import sys

import numpy as np

from video_pose_extractor import (
    FUSION_MODE_WHAM_MEDIAPIPE,
    fusion_frame_overlay_fallback,
    fuse_wham_mediapipe_sequence,
    mediapipe_world_to_canonical_landmarks,
)
from pose_pipeline.fusion import FusionConfig, fuse_wham_mediapipe


def test_synthetic_cli_accepts_explicit_wham_mediapipe_mode_and_exports_contract_metadata(tmp_path):
    """The explicit mode must be accepted without changing legacy synthetic output."""
    output_json = str(tmp_path / "fusion_motion.json")
    script_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "video_pose_extractor.py")
    )

    result = subprocess.run(
        [
            sys.executable,
            script_path,
            "--synthetic",
            "--backend",
            "wham",
            "--fusion-mode",
            "wham_mediapipe",
            "--output",
            output_json,
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    assert result.returncode == 0, result.stderr
    with open(output_json, "r", encoding="utf-8") as stream:
        payload = json.load(stream)

    assert payload["fusionMode"] == "wham_mediapipe"
    assert payload["backendRequested"] == "wham"
    assert payload["backendActual"] == "synthetic"
    assert payload["backendFallback"] is False
    assert payload["overlayBackend"] == "synthetic"
    assert payload["overlaySource"] == "synthetic"
    assert payload["backendMetadata"]["fusionMode"] == "wham_mediapipe"
    assert payload["uncertaintyIntervals"] == []


def test_fusion_integration_aligns_timestamps_and_converts_mediapipe_depth_once():
    """WHAM input remains canonical while MediaPipe world depth crosses one seam."""
    wham = np.zeros((2, 33, 3), dtype=np.float64)
    wham[0, :, 1] = 1.0
    wham[1, :, 1] = 1.1
    wham[:, :, 2] = 0.25
    media_pipe_world = np.zeros((2, 33, 3), dtype=np.float64)
    media_pipe_world[:, :, 1] = 1.0
    media_pipe_world[:, :, 2] = -0.25
    media_pipe_norm = np.zeros((2, 33, 3), dtype=np.float64)
    media_pipe_norm[:, :, 0] = 0.5
    media_pipe_norm[:, :, 1] = 0.25
    media_pipe_norm[:, :, 2] = 0.95

    result = fuse_wham_mediapipe_sequence(
        [
            {"landmarks3d": wham[0]},
            {"landmarks3d": wham[1]},
        ],
        [(media_pipe_world[0], media_pipe_norm[0], 0.95), (media_pipe_world[1], media_pipe_norm[1], 0.95)],
        timestamps=[0.0, 0.041],
        frame_indices=[10, 11],
    )

    assert result.metadata["fusionMode"] == FUSION_MODE_WHAM_MEDIAPIPE
    assert result.metadata["observationAlignment"]["frameIndices"] == [10, 11]
    assert result.metadata["observationAlignment"]["timestampsSec"] == [0.0, 0.041]
    assert result.joints[0, 0, 2] == 0.25
    np.testing.assert_allclose(
        mediapipe_world_to_canonical_landmarks(media_pipe_world)[0, 0],
        [0.0, -1.0, 0.25],
    )


def test_fusion_sequence_reorders_adapter_rows_by_frame_identity_and_preserves_alignment():
    """Temporal adapters may return rows out of order; fusion must not retime them."""
    first = np.zeros((33, 3), dtype=np.float64)
    first[:, 0] = 1.0
    second = np.zeros((33, 3), dtype=np.float64)
    second[:, 0] = 2.0

    # Deliberately return frame 11 before frame 10.  The frame identity is
    # part of the official adapter contract and is stronger than list order.
    result = fuse_wham_mediapipe_sequence(
        [
            {"frameIndex": 11, "timestamp": 1.0 / 30.0, "landmarks3d": second},
            {"frameIndex": 10, "timestamp": 0.0, "landmarks3d": first},
        ],
        [None, None],
        timestamps=[0.0, 1.0 / 30.0],
        frame_indices=[10, 11],
    )

    np.testing.assert_allclose(result.joints[:, :, 0], [[1.0] * 33, [2.0] * 33])
    alignment = result.metadata["observationAlignment"]
    assert alignment["frameIndices"] == [10, 11]
    assert alignment["qualitySourceFrameIndices"] == [10, 11]


def test_partial_joint_fusion_selects_mediapipe_only_for_invalid_wham_and_records_reason():
    """A missing WHAM joint may use MediaPipe while valid WHAM joints remain primary."""
    wham = np.zeros((1, 2, 3), dtype=np.float64)
    wham[0, 0] = [0.1, 0.2, 0.3]
    wham[0, 1] = [np.nan, np.nan, np.nan]
    mediapipe = np.zeros((1, 2, 3), dtype=np.float64)
    mediapipe[0, 0] = [9.0, 9.0, 9.0]
    mediapipe[0, 1] = [0.7, 0.8, 0.9]

    result = fuse_wham_mediapipe(
        wham,
        mediapipe_3d=mediapipe,
        score=np.ones((1, 2), dtype=np.float64),
        visibility=np.ones((1, 2), dtype=np.float64),
        presence=np.ones((1, 2), dtype=np.float64),
        config=FusionConfig(mediapipe_3d_weight=0.0, reprojection_weight=0.0),
    )

    np.testing.assert_allclose(result.joints[0, 0], wham[0, 0])
    np.testing.assert_allclose(result.joints[0, 1], mediapipe[0, 1])
    assert result.provenance[0][0]["source"] == "wham"
    assert result.provenance[0][1]["source"] == "mediapipe"
    assert result.provenance[0][1]["reason"] == "wham_invalid_mediapipe_selected"


def test_sequence_fusion_keeps_valid_wham_joint_primary_when_auxiliary_disagrees():
    """The extraction seam uses strict stage-aware selection, not post-hoc replacement."""
    wham = np.zeros((33, 3), dtype=np.float64)
    wham[0] = [0.1, 0.2, 0.3]
    media_pipe_world = np.zeros((33, 3), dtype=np.float64)
    media_pipe_world[0] = [9.0, 9.0, -9.0]
    result = fuse_wham_mediapipe_sequence(
        [{"frameIndex": 2, "landmarks3d": wham}],
        [(media_pipe_world, None, 1.0)],
        timestamps=[0.0],
        frame_indices=[2],
    )

    np.testing.assert_allclose(result.joints[0, 0], wham[0])
    assert result.provenance[0][0]["source"] == "wham"


def test_fusion_sequence_reports_adapter_errors_without_losing_frame_provenance():
    """Malformed/missing adapter rows become explicit frame diagnostics."""
    valid = np.zeros((33, 3), dtype=np.float64)
    result = fuse_wham_mediapipe_sequence(
        [
            {"frameIndex": 4, "landmarks3d": valid},
            {"frameIndex": 6, "landmarks3d": None, "error": "decoder timeout"},
        ],
        [None, None],
        timestamps=[0.0, 1.0 / 30.0],
        frame_indices=[4, 6],
    )

    assert result.metadata["frameProvenance"][1]["reason"] == "wham_missing"
    assert "decoder timeout" in result.metadata["frameErrors"][0]["message"]


def test_overlay_provenance_marks_frames_using_mediapipe_fallback_joints():
    """A fused frame with WHAM-invalid joints must not be labelled WHAM-only in overlays."""
    class Result:
        provenance = [
            [{"source": "wham", "usedFallback": False, "reason": "wham_valid"}],
            [{
                "source": "mediapipe",
                "usedFallback": True,
                "reason": "wham_invalid_mediapipe_selected",
            }],
        ]

    used, reason = fusion_frame_overlay_fallback(Result(), 1)

    assert used is True
    assert reason == "wham_invalid_mediapipe_selected"
    assert fusion_frame_overlay_fallback(Result(), 0) == (False, "")


def test_full_adapter_camera_trajectory_and_temporal_metadata_survive_fusion():
    """Optional full-WHAM stages remain available to downstream retargeting/UI."""
    wham = np.zeros((33, 3), dtype=np.float64)
    result = fuse_wham_mediapipe_sequence(
        {
            "frames": [{"frameIndex": 8, "landmarks3d": wham}],
            "metadata": {
                "camera": {"projectionType": "orthographic", "scale": 0.4},
                "trajectory": {"worldMotionAvailable": True, "root": [[0.0, 0.0, 0.0]]},
                "temporalConstraints": [1.0],
            },
        },
        [None],
        timestamps=[0.0],
        frame_indices=[8],
    )

    assert result.metadata["camera"]["cameraModel"] == "orthographic"
    assert result.metadata["trajectory"]["worldMotionAvailable"] is True
    assert result.metadata["temporalConstraintsApplied"] is True
