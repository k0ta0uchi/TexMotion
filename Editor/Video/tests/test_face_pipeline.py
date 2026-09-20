"""
test_face_pipeline.py - Unit tests for FacePipeline and faceTrack payload format.
"""

import math
import numpy as np
import pytest
from pose_pipeline.face_pipeline import (
    FacePipeline,
    ARKIT_52_BLENDSHAPES,
    build_face_track_payload
)
from pose_pipeline.kinematics import matrix_to_quaternion


def test_arkit_52_blendshapes_count():
    """Verify exact 52 blendshape names are present."""
    assert len(ARKIT_52_BLENDSHAPES) == 52
    assert "eyeBlinkLeft" in ARKIT_52_BLENDSHAPES
    assert "eyeBlinkRight" in ARKIT_52_BLENDSHAPES
    assert "jawOpen" in ARKIT_52_BLENDSHAPES
    assert "mouthSmileLeft" in ARKIT_52_BLENDSHAPES


def test_build_face_track_payload_structure():
    """Verify build_face_track_payload generates exact VideoMotionData faceTrack schema."""
    timestamps = [0.0, 0.033, 0.066]
    n_frames = len(timestamps)

    shape_data = {name: [0.0] * n_frames for name in ARKIT_52_BLENDSHAPES}
    shape_data["eyeBlinkLeft"] = [0.0, 0.1, 0.0]
    shape_data["jawOpen"] = [0.0, 0.0, 0.2]

    # Flat quaternion array [frame * 4 + i] -> x, y, z, w
    head_rotations_flat = [
        0.0, 0.0, 0.0, 1.0,
        0.01, 0.02, 0.0, 0.999,
        0.0, 0.0, 0.0, 1.0
    ]
    confidences = [0.95, 0.96, 0.94]

    payload = build_face_track_payload(timestamps, shape_data, head_rotations_flat, confidences)

    assert payload["version"] == 1
    assert payload["timestamps"] == [0.0, 0.033, 0.066]
    assert len(payload["headRotationsFlat"]) == n_frames * 4
    assert payload["confidences"] == confidences

    shapes = payload["shapes"]
    assert len(shapes) == 52
    shape_map = {s["shapeName"]: s["weights"] for s in shapes}
    assert shape_map["eyeBlinkLeft"] == [0.0, 0.1, 0.0]
    assert shape_map["jawOpen"] == [0.0, 0.0, 0.2]
    assert shape_map["mouthSmileLeft"] == [0.0, 0.0, 0.0]


def test_face_pipeline_fallback_when_model_missing():
    """Verify FacePipeline initializes cleanly with is_available=False if model is not found."""
    pipeline = FacePipeline(model_path="non_existent_model_file.task", allow_download=False)
    assert not pipeline.is_available

    # Should process frames safely with neutral/identity fallback
    dummy_frames = [np.zeros((100, 100, 3), dtype=np.uint8) for _ in range(3)]
    timestamps = [0.0, 0.033, 0.066]

    result = pipeline.process_frames(dummy_frames, timestamps)
    assert result["version"] == 1
    assert len(result["timestamps"]) == 3
    assert len(result["shapes"]) == 52
    # All blendshapes are neutral 0.0
    for shape in result["shapes"]:
        assert all(w == 0.0 for w in shape["weights"])
    # All head rotations are identity [0, 0, 0, 1]
    assert result["headRotationsFlat"] == [0.0, 0.0, 0.0, 1.0] * 3
    # Confidences are 0.0
    assert result["confidences"] == [0.0, 0.0, 0.0]

    pipeline.close()


def test_face_head_rotation_matrix_to_quaternion():
    """Verify 3x3 rotation matrix to quaternion conversion for head rigid rotation."""
    # Identity matrix -> [0, 0, 0, 1]
    R_id = np.eye(3)
    q_id = matrix_to_quaternion(R_id)
    assert np.allclose(q_id, [0.0, 0.0, 0.0, 1.0], atol=1e-5)

    # 90-degree yaw around Y axis
    # R_y(90 deg) = [[0, 0, 1], [0, 1, 0], [-1, 0, 0]]
    R_y = np.array([
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
        [-1.0, 0.0, 0.0]
    ])
    q_y = matrix_to_quaternion(R_y)
    # Expected: [0, sin(45 deg), 0, cos(45 deg)] = [0, 0.7071, 0, 0.7071]
    expected_y = np.array([0.0, math.sin(math.pi / 4), 0.0, math.cos(math.pi / 4)])
    assert np.allclose(q_y, expected_y, atol=1e-4)
