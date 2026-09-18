"""Focused tests for the dependency-free fusion camera/coordinate API."""

import numpy as np
import pytest

from pose_pipeline.camera import (
    CANONICAL_COORDINATE_SYSTEM,
    CameraModel,
    canonical_coordinate_metadata,
    estimate_camera_from_anchors,
    project_points,
    wham_to_canonical,
)


def test_wham_raw_coordinates_convert_once_and_preserve_metadata_columns():
    raw = np.array(
        [
            [0.2, -0.5, 0.3, 0.8],
            [-0.1, 0.25, -0.4, 0.2],
        ],
        dtype=np.float32,
    )

    canonical = wham_to_canonical(raw)

    np.testing.assert_allclose(canonical[:, :3], [[0.2, 0.5, -0.3], [-0.1, -0.25, 0.4]])
    np.testing.assert_allclose(canonical[:, 3], raw[:, 3])
    metadata = canonical_coordinate_metadata(world_motion_available=False)
    assert metadata["coordinateSystem"] == CANONICAL_COORDINATE_SYSTEM
    assert metadata["axisY"] == "up"
    assert metadata["axisZ"] == "forward"
    assert metadata["worldMotionAvailable"] is False


def test_orthographic_projection_uses_canonical_up_axis_as_image_down():
    camera = CameraModel.orthographic(scale=0.1, principal_point=(0.5, 0.5))
    points = np.array([[1.0, 2.0, 0.0], [-1.0, -2.0, 0.0]])

    projected = project_points(points, camera)

    np.testing.assert_allclose(projected, [[0.6, 0.3], [0.4, 0.7]])


def test_camera_anchor_estimation_recovers_orthographic_scale_and_translation():
    source = np.array(
        [[-0.5, 0.8, 0.0], [0.5, 0.8, 0.0], [-0.4, 0.0, 0.0], [0.4, 0.0, 0.0]],
        dtype=np.float64,
    )
    expected = CameraModel.orthographic(scale=0.2, principal_point=(0.35, 0.6))
    observed = expected.project(source)

    estimated = estimate_camera_from_anchors(source, observed, mode="orthographic")

    assert estimated.projection_type == "orthographic"
    assert estimated.scale == pytest.approx(0.2, abs=1e-6)
    assert estimated.principal_point == pytest.approx((0.35, 0.6), abs=1e-6)
    np.testing.assert_allclose(estimated.project(source), observed, atol=1e-6)
