"""
test_temporal_repair.py - Unit tests for short-gap SLERP and provenance generation.
"""

import math
import numpy as np
import pytest
from pose_pipeline.temporal_repair import (
    repair_short_gaps_slerp,
    repair_short_gaps_positions,
    repair_long_gaps_kinematic_hermite,
    quaternion_log,
    quaternion_exp,
    quaternion_squad,
    cubic_hermite_interpolate,
)
from pose_pipeline.observations import ObservationStatus
from pose_pipeline.kinematics import normalize_quaternion


def test_repair_short_gap_slerp():
    """Verify short gap <= 0.2s is smoothly repaired using quaternion SLERP."""
    fps = 30.0
    frames = 20
    n_joints = 3

    # Base rotations: all identity
    rotations = np.zeros((frames, n_joints, 4), dtype=np.float64)
    rotations[:, :, 3] = 1.0  # [0, 0, 0, 1]

    # At joint 1, rotate around Y: from 0 deg at frame 4 to 40 deg at frame 9
    # Gap exists at frames 5, 6, 7, 8 (4 frames = 0.133s <= 0.2s)
    angle_start = 0.0
    angle_end = math.radians(40.0)

    q_start = np.array([0.0, math.sin(angle_start / 2), 0.0, math.cos(angle_start / 2)])
    q_end = np.array([0.0, math.sin(angle_end / 2), 0.0, math.cos(angle_end / 2)])

    rotations[4, 1] = q_start
    rotations[9, 1] = q_end

    confidences = np.ones((frames, n_joints), dtype=np.float64)
    # Drop confidence in gap [5, 8]
    confidences[5:9, 1] = 0.1

    repaired_rot, repaired_conf, provenance = repair_short_gaps_slerp(
        rotations,
        confidences,
        fps=fps,
        max_gap_seconds=0.2,
        confidence_threshold=0.35
    )

    # Verify provenance
    assert len(provenance) == 1
    rec = provenance[0]
    assert rec["jointIndex"] == 1
    assert rec["startFrame"] == 5
    assert rec["endFrame"] == 8
    assert rec["gapLength"] == 4
    assert rec["durationSeconds"] == pytest.approx(4 / 30.0, abs=1e-3)
    assert rec["method"] == "quaternion_slerp"

    # Verify midpoint interpolation: at frame 6 or 7, rotation should be intermediate
    # u for frame 6.5 (between frame 4 and frame 9, span=5): (6.5 - 4) / 5 = 0.5 -> 20 deg
    q_mid = repaired_rot[6, 1]
    # Length should be unit
    assert math.isclose(np.linalg.norm(q_mid), 1.0, abs_tol=1e-5)
    # Y angle should be between start and end
    assert 0.0 < q_mid[1] < q_end[1]

    # Verify confidences were updated
    assert all(repaired_conf[t, 1] > 0.5 for t in range(5, 9))


def test_skip_long_gap_exceeding_threshold():
    """Verify gap > 0.2s is not repaired (preventing hallucination)."""
    fps = 30.0
    frames = 30
    rotations = np.zeros((frames, 4), dtype=np.float64)
    rotations[:, 3] = 1.0

    confidences = np.ones(frames, dtype=np.float64)
    # Gap of 10 frames = 0.333s > 0.2s
    confidences[10:20] = 0.0

    repaired_rot, repaired_conf, provenance = repair_short_gaps_slerp(
        rotations,
        confidences,
        fps=fps,
        max_gap_seconds=0.2
    )

    assert len(provenance) == 0
    assert all(repaired_conf[10:20] == 0.0)


def test_repair_with_observation_status():
    """Verify ObservationStatus (OCCLUDED/PREDICTED) triggers repair and sets priorStatus."""
    fps = 30.0
    frames = 15
    rotations = np.zeros((frames, 4), dtype=np.float64)
    rotations[:, 3] = 1.0
    confidences = np.ones(frames, dtype=np.float64)

    statuses = [ObservationStatus.OBSERVED] * frames
    statuses[5] = ObservationStatus.OCCLUDED
    statuses[6] = ObservationStatus.OCCLUDED

    repaired_rot, repaired_conf, provenance = repair_short_gaps_slerp(
        rotations,
        confidences,
        statuses=statuses,
        fps=fps,
        max_gap_seconds=0.2
    )

    assert len(provenance) == 1
    assert provenance[0]["priorStatus"] == "OCCLUDED"
    assert provenance[0]["startFrame"] == 5
    assert provenance[0]["endFrame"] == 6


def test_boundary_gap_safety():
    """Verify missing frames at beginning or end do not cause crashes."""
    fps = 30.0
    frames = 10
    rotations = np.zeros((frames, 4), dtype=np.float64)
    rotations[:, 3] = 1.0
    confidences = np.ones(frames, dtype=np.float64)

    # Missing at start (frame 0) and end (frame 9)
    confidences[0] = 0.0
    confidences[9] = 0.0

    repaired_rot, repaired_conf, provenance = repair_short_gaps_slerp(
        rotations,
        confidences,
        fps=fps,
        max_gap_seconds=0.2
    )

    # Cannot interpolate without valid boundary frame on one side -> skipped safely
    assert len(provenance) == 0


def test_repair_short_gaps_positions():
    """Verify linear position repair on 3D coordinates."""
    fps = 30.0
    frames = 10
    pos = np.zeros((frames, 3))
    pos[2] = [0.0, 1.0, 0.0]
    pos[5] = [0.0, 4.0, 0.0]

    conf = np.ones(frames)
    conf[3:5] = 0.1  # 2 frames gap = 0.067s <= 0.2s

    repaired_pos, provenance = repair_short_gaps_positions(pos, conf, fps=fps, max_gap_seconds=0.2)
    assert len(provenance) == 1
    # Frame 3 (u = 1/3): 1.0 + 1/3 * 3.0 = 2.0
    assert repaired_pos[3, 1] == pytest.approx(2.0, abs=1e-3)
    # Frame 4 (u = 2/3): 1.0 + 2/3 * 3.0 = 3.0
    assert repaired_pos[4, 1] == pytest.approx(3.0, abs=1e-3)


def test_quaternion_log_exp_roundtrip():
    """Verify log and exp mapping on SO(3) roundtrips accurately."""
    # Rotation of 60 degrees around [1, 1, 0] axis
    axis = np.array([1.0, 1.0, 0.0]) / math.sqrt(2.0)
    angle = math.radians(60.0)
    q = np.array([axis[0] * math.sin(angle / 2), axis[1] * math.sin(angle / 2), 0.0, math.cos(angle / 2)])

    omega = quaternion_log(q)
    q_reconstructed = quaternion_exp(omega)

    assert np.allclose(q, q_reconstructed, atol=1e-5)


def test_quaternion_squad_interpolation():
    """Verify Squad interpolation produces valid unit quaternions with smooth progression."""
    q0 = np.array([0.0, 0.0, 0.0, 1.0])
    q1 = np.array([0.0, math.sin(math.radians(45) / 2), 0.0, math.cos(math.radians(45) / 2)])
    s0 = np.array([0.0, math.sin(math.radians(10) / 2), 0.0, math.cos(math.radians(10) / 2)])
    s1 = np.array([0.0, math.sin(math.radians(35) / 2), 0.0, math.cos(math.radians(35) / 2)])

    for u in [0.0, 0.25, 0.5, 0.75, 1.0]:
        q_interp = quaternion_squad(q0, q1, s0, s1, u)
        assert math.isclose(np.linalg.norm(q_interp), 1.0, abs_tol=1e-5)
    
    # Boundary endpoints match
    assert np.allclose(quaternion_squad(q0, q1, s0, s1, 0.0), q0, atol=1e-5)
    assert np.allclose(quaternion_squad(q0, q1, s0, s1, 1.0), q1, atol=1e-5)


def test_cubic_hermite_interpolate():
    """Verify Cubic Hermite Spline position interpolation matches boundary velocity direction."""
    p0 = np.array([0.0, 0.0, 0.0])
    p1 = np.array([1.0, 0.0, 0.0])
    v0 = np.array([2.0, 0.0, 0.0])  # moving right fast
    v1 = np.array([0.0, 0.0, 0.0])  # stopping at p1
    duration = 1.0

    p_start = cubic_hermite_interpolate(p0, p1, v0, v1, duration, 0.0)
    p_mid = cubic_hermite_interpolate(p0, p1, v0, v1, duration, 0.5)
    p_end = cubic_hermite_interpolate(p0, p1, v0, v1, duration, 1.0)

    assert np.allclose(p_start, p0, atol=1e-5)
    assert np.allclose(p_end, p1, atol=1e-5)
    # Due to initial velocity v0=2, midpoint x should be > 0.5 (front-loaded displacement)
    assert p_mid[0] > 0.5


def test_repair_long_gaps_kinematic_hermite():
    """Verify medium-to-long gap (0.5s = 15 frames) is repaired with Hermite Squad and Position inpainting."""
    fps = 30.0
    frames = 40
    n_joints = 2

    rotations = np.zeros((frames, n_joints, 4), dtype=np.float64)
    rotations[:, :, 3] = 1.0

    root_pos = np.zeros((frames, 3), dtype=np.float64)
    # Root position moving steadily forward along Z: 0.1m per frame (3.0 m/s)
    for t in range(frames):
        root_pos[t, 2] = t * 0.1

    # Joint 1 rotates from 0 to 60 deg around X
    for t in range(frames):
        ang = math.radians(t / float(frames) * 60.0)
        rotations[t, 1] = [math.sin(ang / 2), 0.0, 0.0, math.cos(ang / 2)]

    confidences = np.ones((frames, n_joints), dtype=np.float64)
    # Create 15-frame gap (0.5s) from frame 12 to 26
    confidences[12:27, :] = 0.05

    repaired_rot, repaired_pos, repaired_conf, provenance = repair_long_gaps_kinematic_hermite(
        rotations,
        confidences,
        root_positions=root_pos,
        fps=fps,
        min_gap_seconds=0.2,
        max_gap_seconds=2.0,
        confidence_threshold=0.35,
        damping=0.2
    )

    # Verify provenance was created for both rotation and root position
    assert len(provenance) >= 2
    methods = [p["method"] for p in provenance]
    assert "kinematic_hermite_squad" in methods
    assert "cubic_hermite_position" in methods

    # Verify inpainting occurred in gap
    for t in range(12, 27):
        # Rotations should be unit quaternions
        assert math.isclose(np.linalg.norm(repaired_rot[t, 1]), 1.0, abs_tol=1e-5)
        # Root Z positions should advance forward monotonically
        assert repaired_pos[t - 1, 2] < repaired_pos[t, 2]


def test_repair_long_gaps_exceeding_max_threshold():
    """Verify gap exceeding max_gap_seconds (> 2.5s) is safely skipped to avoid wild extrapolation."""
    fps = 30.0
    frames = 120  # 4 seconds
    rotations = np.zeros((frames, 4), dtype=np.float64)
    rotations[:, 3] = 1.0
    confidences = np.ones(frames, dtype=np.float64)

    # 90-frame gap = 3.0s > 2.5s
    confidences[15:105] = 0.0

    repaired_rot, _, _, provenance = repair_long_gaps_kinematic_hermite(
        rotations,
        confidences,
        fps=fps,
        min_gap_seconds=0.2,
        max_gap_seconds=2.5
    )

    # Should be skipped
    assert len(provenance) == 0
    assert all(confidences[15:105] == 0.0)

