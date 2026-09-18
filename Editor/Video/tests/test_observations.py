"""
test_observations.py - Tests for observations data models, status flags, and uncertainty intervals.
"""

import pytest
import numpy as np
from dataclasses import FrozenInstanceError
from pose_pipeline.observations import (
    Keypoint2D,
    Keypoint3D,
    FrameObservations,
    ObservationStatus,
    UncertaintyInterval
)


def test_keypoint2d_creation_and_immutability():
    kp = Keypoint2D(x=0.45, y=0.65, score=0.92, visibility=0.95, presence=0.90, status=ObservationStatus.OBSERVED)
    assert kp.x == 0.45
    assert kp.y == 0.65
    assert kp.score == 0.92
    assert kp.visibility == 0.95
    assert kp.presence == 0.90
    assert kp.status == ObservationStatus.OBSERVED
    arr = kp.to_array()
    assert np.allclose(arr, [0.45, 0.65])

    # Immutability check: modifying frozen dataclass must raise FrozenInstanceError
    with pytest.raises(FrozenInstanceError):
        kp.x = 0.99  # type: ignore


def test_keypoint3d_creation_and_immutability():
    kp = Keypoint3D(x=0.1, y=-0.42, z=0.05, score=0.88, status=ObservationStatus.OBSERVED)
    assert kp.x == 0.1
    assert kp.y == -0.42
    assert kp.z == 0.05
    assert kp.score == 0.88
    assert kp.status == ObservationStatus.OBSERVED
    arr = kp.to_array()
    assert np.allclose(arr, [0.1, -0.42, 0.05])

    with pytest.raises(FrozenInstanceError):
        kp.z = 1.0  # type: ignore


def test_observation_status_flags_coverage():
    """Verifies all observation provenance status flags can be assigned and checked."""
    statuses = [
        ObservationStatus.MISSING,
        ObservationStatus.OBSERVED,
        ObservationStatus.OCCLUDED,
        ObservationStatus.PREDICTED,
        ObservationStatus.USER_CONSTRAINED,
    ]
    for st in statuses:
        kp2d = Keypoint2D(x=0.5, y=0.5, status=st)
        kp3d = Keypoint3D(x=0.0, y=0.0, z=0.0, status=st)
        assert kp2d.status == st
        assert kp3d.status == st


def test_frame_observations_populated(sample_keypoints_2d, sample_landmarks_3d):
    lms_3d_objs = [
        Keypoint3D(x=float(p[0]), y=float(p[1]), z=float(p[2]), score=0.85)
        for p in sample_landmarks_3d
    ]
    frame_obs = FrameObservations(
        frame_index=12,
        timestamp=0.40,
        keypoints_2d=sample_keypoints_2d,
        landmarks_3d=lms_3d_objs,
        raw_confidence=0.85,
        detector_name="test_detector"
    )

    assert frame_obs.frame_index == 12
    assert frame_obs.timestamp == 0.40
    assert frame_obs.num_keypoints == 33

    arr_2d = frame_obs.get_2d_array()
    assert arr_2d.shape == (33, 2)

    arr_3d = frame_obs.get_3d_array()
    assert arr_3d.shape == (33, 3)

    vis = frame_obs.get_visibilities()
    assert vis.shape == (33,)
    assert np.all(vis > 0.0)

    confs = frame_obs.get_confidences()
    assert confs.shape == (33,)
    assert np.all(confs > 0.0)


def test_frame_observations_empty():
    """Verifies that empty FrameObservations safely returns (0, ...) numpy arrays without exceptions."""
    empty_obs = FrameObservations(frame_index=0, timestamp=0.0)
    assert empty_obs.num_keypoints == 0
    assert empty_obs.get_2d_array().shape == (0, 2)
    assert empty_obs.get_3d_array().shape == (0, 3)
    assert empty_obs.get_visibilities().shape == (0,)
    assert empty_obs.get_confidences().shape == (0,)


def test_frame_observations_with_occlusions(occluded_keypoints_2d):
    """Verifies FrameObservations behavior with occluded and missing keypoints."""
    obs = FrameObservations(
        frame_index=1,
        timestamp=0.033,
        keypoints_2d=occluded_keypoints_2d
    )
    assert obs.num_keypoints == 33
    vis = obs.get_visibilities()
    assert vis[15] == pytest.approx(0.10, abs=1e-4)  # Occluded wrist
    assert vis[17] == pytest.approx(0.0, abs=1e-4)   # Missing finger
    assert obs.keypoints_2d[15].status == ObservationStatus.OCCLUDED
    assert obs.keypoints_2d[17].status == ObservationStatus.MISSING


def test_uncertainty_interval_serialization():
    interval = UncertaintyInterval(
        start_frame=15,
        end_frame=25,
        reason="leg_crossing_ambiguity",
        confidence=0.45,
        recommended_action="swap_crossing"
    )
    d = interval.to_dict()
    assert d["startFrame"] == 15
    assert d["endFrame"] == 25
    assert d["reason"] == "leg_crossing_ambiguity"
    assert d["confidence"] == 0.45
    assert d["recommendedAction"] == "swap_crossing"

    restored = UncertaintyInterval.from_dict(d)
    assert restored.start_frame == 15
    assert restored.end_frame == 25
    assert restored.confidence == 0.45
    assert restored.reason == "leg_crossing_ambiguity"
    assert restored.recommended_action == "swap_crossing"


def test_uncertainty_interval_from_dict_defaults():
    """Verifies graceful defaults when reconstructing UncertaintyInterval from empty or partial dicts."""
    empty_dict = {}
    restored = UncertaintyInterval.from_dict(empty_dict)
    assert restored.start_frame == 0
    assert restored.end_frame == 0
    assert restored.reason == ""
    assert restored.confidence == 0.0
    assert restored.recommended_action == ""
