"""
test_contact_track.py - Unit tests for Contact Track extraction and schema.
"""

import numpy as np
import pytest
from pose_pipeline.sequence_fit import SequenceOptimizer
from video_pose_extractor import apply_foot_locking


def test_sequence_optimizer_build_contact_track():
    """Verify build_contact_track produces valid VideoMotionData contactTrack schema."""
    fps = 30.0
    optimizer = SequenceOptimizer(fps=fps)

    frames = 30
    l_contact = np.zeros(frames, dtype=bool)
    r_contact = np.zeros(frames, dtype=bool)

    # Left foot contact from frame 5 to 15 (0.167s to 0.500s)
    l_contact[5:16] = True
    # Right foot contact from frame 18 to 28 (0.600s to 0.933s)
    r_contact[18:29] = True

    l_foot_pos = np.zeros((frames, 3))
    l_foot_pos[:, 0] = 0.15
    r_foot_pos = np.zeros((frames, 3))
    r_foot_pos[:, 0] = -0.15

    contact_track = optimizer.build_contact_track(
        l_contact, r_contact,
        l_foot_pos=l_foot_pos,
        r_foot_pos=r_foot_pos
    )

    assert contact_track["version"] == 1
    assert "intervals" in contact_track
    intervals = contact_track["intervals"]
    assert len(intervals) == 2

    # Verify left interval
    left_int = intervals[0]
    assert left_int["foot"] == "left"
    assert left_int["start"] == pytest.approx(5 / fps, abs=0.01)
    assert left_int["end"] == pytest.approx(15 / fps, abs=0.01)
    assert left_int["mode"] == "flat"
    assert left_int["confidence"] >= 0.8
    assert len(left_int["anchor"]) == 3
    assert left_int["anchor"][0] == pytest.approx(0.15, abs=0.02)

    # Verify right interval
    right_int = intervals[1]
    assert right_int["foot"] == "right"
    assert right_int["start"] == pytest.approx(18 / fps, abs=0.01)
    assert right_int["end"] == pytest.approx(28 / fps, abs=0.01)
    assert right_int["anchor"][0] == pytest.approx(-0.15, abs=0.02)


def test_apply_foot_locking_returns_contact_track():
    """Verify apply_foot_locking with return_contact_track=True extracts valid intervals."""
    fps = 30.0
    frames = 40
    joints = np.zeros((frames, 22, 3), dtype=np.float64)

    # Upright spine
    joints[:, 0] = [0.0, 0.9, 0.0]   # Pelvis
    joints[:, 12] = [0.0, 1.4, 0.0]  # Neck
    joints[:, 15] = [0.0, 1.6, 0.0]  # Head

    # Left leg: hip(1), knee(4), ankle(7), foot(10)
    joints[:, 1] = [0.15, 0.85, 0.0]
    joints[:, 4] = [0.15, 0.45, 0.0]
    joints[:, 7] = [0.15, 0.06, 0.0]  # On ground
    joints[:, 10] = [0.15, 0.02, 0.1] # Toe on ground

    # Right leg: hip(2), knee(5), ankle(8), foot(11)
    joints[:, 2] = [-0.15, 0.85, 0.0]
    joints[:, 5] = [-0.15, 0.45, 0.0]
    joints[:, 8] = [-0.15, 0.06, 0.0]
    joints[:, 11] = [-0.15, 0.02, 0.1]

    # Right leg lifts up in middle
    joints[15:25, 8, 1] = 0.30
    joints[15:25, 11, 1] = 0.25

    locked_joints, contact_track = apply_foot_locking(joints, fps, return_contact_track=True)

    assert locked_joints.shape == joints.shape
    assert isinstance(contact_track, dict)
    assert contact_track["version"] == 1
    assert "intervals" in contact_track
    intervals = contact_track["intervals"]

    # Left foot remained grounded throughout
    left_intervals = [it for it in intervals if it["foot"] == "left"]
    assert len(left_intervals) >= 1
    assert left_intervals[0]["mode"] in ("flat", "toe", "heel")
    assert len(left_intervals[0]["anchor"]) == 3

    # Backward compatibility check: default return_contact_track=False returns ndarray only
    locked_only = apply_foot_locking(joints, fps)
    assert isinstance(locked_only, np.ndarray)


def test_contact_track_empty_on_short_or_inverted():
    """Verify empty intervals on degenerate input."""
    short_joints = np.zeros((2, 22, 3))
    _, contact_track = apply_foot_locking(short_joints, 30.0, return_contact_track=True)
    assert contact_track == {"version": 1, "intervals": []}
