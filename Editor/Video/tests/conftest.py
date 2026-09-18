"""
conftest.py - Pytest fixtures and mock generators for pose_pipeline tests.
"""

import sys
import os
import pytest
import numpy as np

# Ensure Editor/Video is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pose_pipeline.observations import Keypoint2D, Keypoint3D, FrameObservations, ObservationStatus


@pytest.fixture
def sample_keypoints_2d():
    """Returns a list of 33 normalized 2D keypoints simulating a standing person."""
    kps = []
    for i in range(33):
        # Default standing coordinates in normalized screen space [0, 1]
        x = 0.5 + (0.05 if i % 2 == 1 else -0.05)
        y = 0.1 + (i / 33.0) * 0.8
        score = 0.85
        kps.append(Keypoint2D(x=x, y=y, score=score, visibility=score, presence=score, status=ObservationStatus.OBSERVED))
    return kps


@pytest.fixture
def occluded_keypoints_2d():
    """Returns a list of 33 keypoints simulating a person with occluded / missing limbs."""
    kps = []
    for i in range(33):
        if i in [15, 16, 27, 28]:  # Wrists and ankles occluded
            kps.append(Keypoint2D(x=0.5, y=0.5, score=0.15, visibility=0.10, presence=0.20, status=ObservationStatus.OCCLUDED))
        elif i in [17, 18, 19, 20, 21, 22, 29, 30, 31, 32]:  # Fingers & toes missing
            kps.append(Keypoint2D(x=0.0, y=0.0, score=0.0, visibility=0.0, presence=0.0, status=ObservationStatus.MISSING))
        else:
            kps.append(Keypoint2D(x=0.5, y=0.1 + (i / 33.0) * 0.8, score=0.9, visibility=0.9, presence=0.9, status=ObservationStatus.OBSERVED))
    return kps


@pytest.fixture
def sample_landmarks_3d():
    """Returns a (33, 3) numpy array simulating standing 3D landmarks (SMPL-X / MediaPipe meters)."""
    lms = np.zeros((33, 3), dtype=np.float32)
    # Hips (23, 24)
    lms[23] = np.array([-0.10, 0.0, 0.0], dtype=np.float32)
    lms[24] = np.array([0.10, 0.0, 0.0], dtype=np.float32)
    # Knees (25, 26) - 0.42m below hips
    lms[25] = np.array([-0.10, -0.42, 0.02], dtype=np.float32)
    lms[26] = np.array([0.10, -0.42, 0.02], dtype=np.float32)
    # Ankles (27, 28) - 0.42m below knees
    lms[27] = np.array([-0.10, -0.84, 0.0], dtype=np.float32)
    lms[28] = np.array([0.10, -0.84, 0.0], dtype=np.float32)
    # Shoulders (11, 12) - 0.45m above hips
    lms[11] = np.array([-0.18, 0.45, 0.0], dtype=np.float32)
    lms[12] = np.array([0.18, 0.45, 0.0], dtype=np.float32)
    # Elbows (13, 14)
    lms[13] = np.array([-0.28, 0.20, 0.0], dtype=np.float32)
    lms[14] = np.array([0.28, 0.20, 0.0], dtype=np.float32)
    # Wrists (15, 16)
    lms[15] = np.array([-0.30, -0.05, 0.0], dtype=np.float32)
    lms[16] = np.array([0.30, -0.05, 0.0], dtype=np.float32)
    # Head / Nose (0)
    lms[0] = np.array([0.0, 0.65, -0.05], dtype=np.float32)
    return lms


@pytest.fixture
def sample_smplx_22_joints():
    """Returns a (22, 3) numpy array simulating standing 3D joints in SMPL-X 22 hierarchy."""
    joints = np.zeros((22, 3), dtype=np.float64)
    # 0: Pelvis
    joints[0] = np.array([0.0, 0.0, 0.0])
    # 1: L_Hip, 2: R_Hip
    joints[1] = np.array([0.08, -0.06, 0.0])
    joints[2] = np.array([-0.08, -0.06, 0.0])
    # 3: Spine1, 6: Spine2, 9: Spine3, 12: Neck, 15: Head
    joints[3] = np.array([0.0, 0.11, -0.01])
    joints[6] = np.array([0.0, 0.23, -0.02])
    joints[9] = np.array([0.0, 0.38, -0.03])
    joints[12] = np.array([0.0, 0.50, -0.03])
    joints[15] = np.array([0.0, 0.65, -0.02])
    # 4: L_Knee, 5: R_Knee (-0.38m from hips)
    joints[4] = np.array([0.08, -0.44, 0.0])
    joints[5] = np.array([-0.08, -0.44, 0.0])
    # 7: L_Ankle, 8: R_Ankle (-0.39m from knees)
    joints[7] = np.array([0.08, -0.83, 0.0])
    joints[8] = np.array([-0.08, -0.83, 0.0])
    # 10: L_Foot, 11: R_Foot
    joints[10] = np.array([0.08, -0.87, 0.12])
    joints[11] = np.array([-0.08, -0.87, 0.12])
    # 13: L_Collar, 14: R_Collar, 16: L_Shoulder, 17: R_Shoulder
    joints[13] = np.array([0.06, 0.46, -0.03])
    joints[14] = np.array([-0.06, 0.46, -0.03])
    joints[16] = np.array([0.18, 0.46, -0.03])
    joints[17] = np.array([-0.18, 0.46, -0.03])
    # 18: L_Elbow, 19: R_Elbow
    joints[18] = np.array([0.44, 0.46, -0.03])
    joints[19] = np.array([-0.44, 0.46, -0.03])
    # 20: L_Wrist, 21: R_Wrist
    joints[20] = np.array([0.70, 0.46, -0.03])
    joints[21] = np.array([-0.70, 0.46, -0.03])
    return joints


@pytest.fixture
def mock_frame_bgr():
    """Generates a dummy 640x480 BGR image."""
    return np.zeros((480, 640, 3), dtype=np.uint8)


@pytest.fixture
def synthetic_video_sample(tmp_path):
    """
    Creates a temporary valid 30-frame 640x360 MP4 test video with animated content.
    Properly releases resources to prevent file locking on Windows.
    """
    import cv2
    video_path = str(tmp_path / "fixture_synthetic_video.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(video_path, fourcc, 30.0, (640, 360))

    try:
        for t in range(30):
            frame = np.full((360, 640, 3), (25, 25, 30), dtype=np.uint8)
            cv2.circle(frame, (320, 180), 10 + t * 2, (80, 200, 255), -1)
            writer.write(frame)
    finally:
        writer.release()

    yield video_path

    # Windows cleanup
    if os.path.exists(video_path):
        try:
            os.remove(video_path)
        except OSError:
            pass
