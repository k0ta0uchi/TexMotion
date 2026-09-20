"""
test_rtmpose_backend.py - Tests for RTMPose ONNX detector backend, SimCC decoding,
lifecycle management, fallback mechanics, preflight check, and manifest validation.
"""

import os
import gc
import pytest
import numpy as np

from pose_pipeline.backends.rtmpose_backend import (
    RTMPoseBackend,
    COCO_TO_MEDIAPIPE_MAP,
    check_backend_available,
    verify_model_manifest
)
from pose_pipeline.backends.base import BackendCapabilities
from pose_pipeline.observations import ObservationStatus, FrameObservations


def test_rtmpose_capabilities():
    backend = RTMPoseBackend()
    caps = backend.get_capabilities()
    assert isinstance(caps, BackendCapabilities)
    assert caps.name == "rtmpose"
    assert caps.has_2d is True
    assert caps.has_3d is False
    assert caps.keypoint_count == 33


def test_rtmpose_lifecycle_missing_model(mock_frame_bgr):
    """
    Lifecycle test:
    - is_available returns False when weights are absent.
    - initialize returns False gracefully without raising exceptions.
    - detect returns None when uninitialized.
    - close properly cleans up and is idempotent.
    """
    backend = RTMPoseBackend(model_path="non_existent_weights_xyz.onnx", allow_download=False)
    assert backend.is_available() is False
    assert backend.initialize() is False
    assert backend.is_initialized is False

    # Detect on uninitialized backend
    result = backend.detect(mock_frame_bgr, timestamp_sec=0.0)
    assert result is None

    # Idempotent close
    backend.close()
    assert backend.is_initialized is False
    backend.close()


def test_rtmpose_context_manager(mock_frame_bgr):
    """Verifies that RTMPoseBackend works as a context manager and safely exits."""
    with RTMPoseBackend(model_path="non_existent.onnx", allow_download=False) as backend:
        assert backend.is_initialized is False
        res = backend.detect(mock_frame_bgr, 0.0)
        assert res is None
    assert backend.is_initialized is False


def test_rtmpose_simcc_decoding_standard():
    """Verifies SimCC 1D heatmap peak decoding under normal conditions."""
    backend = RTMPoseBackend()
    K = 17
    W_sub = 384
    H_sub = 512
    simcc_x = np.zeros((1, K, W_sub), dtype=np.float32)
    simcc_y = np.zeros((1, K, H_sub), dtype=np.float32)

    # Set peaks at known coordinates
    for k in range(K):
        simcc_x[0, k, 192] = 6.0  # Peak in middle
        simcc_y[0, k, 256] = 6.0

    coords, scores = backend._decode_simcc(
        [simcc_x, simcc_y],
        scale_x=1.0, scale_y=1.0,
        pad_x=0, pad_y=0,
        orig_w=192, orig_h=256
    )

    assert coords.shape == (17, 2)
    assert scores.shape == (17,)
    assert np.all(scores > 0.5)
    assert np.isclose(coords[0, 0], 0.5, atol=0.05)
    assert np.isclose(coords[0, 1], 0.5, atol=0.05)


def test_rtmpose_simcc_decoding_boundaries_and_clipping():
    """Verifies SimCC decoding at corners (0,0) and (W-1, H-1), ensuring coordinates are clipped in [0, 1]."""
    backend = RTMPoseBackend()
    K = 17
    W_sub = 384
    H_sub = 512

    # Corner 1: top-left (0, 0)
    simcc_x = np.zeros((1, K, W_sub), dtype=np.float32)
    simcc_y = np.zeros((1, K, H_sub), dtype=np.float32)
    simcc_x[0, :, 0] = 8.0
    simcc_y[0, :, 0] = 8.0

    coords_tl, scores_tl = backend._decode_simcc(
        [simcc_x, simcc_y],
        scale_x=1.0, scale_y=1.0,
        pad_x=0, pad_y=0,
        orig_w=192, orig_h=256
    )
    assert np.all(coords_tl >= 0.0)
    assert np.isclose(coords_tl[0, 0], 0.0, atol=0.02)
    assert np.isclose(coords_tl[0, 1], 0.0, atol=0.02)

    # Corner 2: bottom-right
    simcc_x.fill(0.0)
    simcc_y.fill(0.0)
    simcc_x[0, :, -1] = 8.0
    simcc_y[0, :, -1] = 8.0

    coords_br, scores_br = backend._decode_simcc(
        [simcc_x, simcc_y],
        scale_x=1.0, scale_y=1.0,
        pad_x=0, pad_y=0,
        orig_w=192, orig_h=256
    )
    assert np.all(coords_br <= 1.0)
    assert np.all(coords_br >= 0.0)


def test_rtmpose_simcc_decoding_flat_uniform_zero():
    """Verifies that all-zero heatmaps (no peak) do not produce NaNs or infs."""
    backend = RTMPoseBackend()
    K, W_sub, H_sub = 17, 384, 512
    simcc_x = np.zeros((1, K, W_sub), dtype=np.float32)
    simcc_y = np.zeros((1, K, H_sub), dtype=np.float32)

    coords, scores = backend._decode_simcc(
        [simcc_x, simcc_y],
        scale_x=1.0, scale_y=1.0,
        pad_x=0, pad_y=0,
        orig_w=192, orig_h=256
    )
    assert coords.shape == (17, 2)
    assert not np.any(np.isnan(coords))
    assert not np.any(np.isinf(coords))


def test_rtmpose_direct_keypoint_fallback():
    """Verifies fallback when model output is directly (1, 17, 3) keypoints array instead of SimCC heatmaps."""
    backend = RTMPoseBackend()
    kps_out = np.zeros((1, 17, 3), dtype=np.float32)
    kps_out[0, :, 0] = 96.0   # pixel X
    kps_out[0, :, 1] = 128.0  # pixel Y
    kps_out[0, :, 2] = 0.85   # score

    coords, scores = backend._decode_simcc(
        [kps_out],
        scale_x=1.0, scale_y=1.0,
        pad_x=0, pad_y=0,
        orig_w=192, orig_h=256
    )
    assert coords.shape == (17, 2)
    assert np.isclose(coords[0, 0], 0.5, atol=0.01)
    assert np.isclose(coords[0, 1], 0.5, atol=0.01)
    assert np.allclose(scores, 0.85)


def test_rtmpose_coco_to_mediapipe_conversion():
    """
    Verifies that RTMPose's 17 COCO detections are mapped into the 33 MediaPipe slot layout,
    and unmapped keypoints (e.g. feet indices, hands) are marked with ObservationStatus.MISSING.
    """
    backend = RTMPoseBackend()
    backend.is_initialized = True

    # Deterministic mock decoding
    coords_17 = np.zeros((17, 2), dtype=np.float32)
    for k in range(17):
        coords_17[k] = [0.4 + k * 0.01, 0.3 + k * 0.02]
    scores_17 = np.full((17,), 0.88, dtype=np.float32)

    backend._decode_simcc = lambda outputs, sx, sy, px, py, ow, oh: (coords_17, scores_17)
    class MockSession:
        def run(self, output_names, feeds):
            return [None, None]

    backend._session = MockSession()
    backend._input_name = "input"
    backend._output_names = ["simcc_x", "simcc_y"]
    backend._preprocess = lambda img: (np.zeros((1, 3, 256, 192), dtype=np.float32), 1.0, 1.0, 0, 0)

    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    obs = backend.detect(dummy_frame, timestamp_sec=0.5, frame_index=15)

    assert obs is not None
    assert isinstance(obs, FrameObservations)
    assert obs.num_keypoints == 33
    assert obs.frame_index == 15
    assert obs.timestamp == 0.5
    assert obs.detector_name == "rtmpose"

    mapped_mp_indices = set(COCO_TO_MEDIAPIPE_MAP.values())
    for mp_idx in range(33):
        kp2d = obs.keypoints_2d[mp_idx]
        lm3d = obs.landmarks_3d[mp_idx]
        if mp_idx in mapped_mp_indices:
            assert kp2d.status == ObservationStatus.OBSERVED
            assert kp2d.score == pytest.approx(0.88, abs=1e-3)
            assert lm3d.status in (ObservationStatus.PREDICTED, ObservationStatus.OBSERVED)
        else:
            assert kp2d.status == ObservationStatus.MISSING
            assert kp2d.score == 0.0
            assert lm3d.status == ObservationStatus.MISSING


def test_preflight_check_backend_available():
    """Verifies preflight check_backend_available function without throwing unhandled exceptions."""
    # When model path is non-existent
    is_avail, msg = check_backend_available("definitely_missing_model.onnx")
    assert is_avail is False
    assert "not found" in msg.lower() or "not installed" in msg.lower()

    # Default path resolution check
    is_avail_def, _ = check_backend_available(None)
    assert isinstance(is_avail_def, bool)


def test_verify_model_manifest(tmp_path):
    """Verifies manifest validation on missing, empty, and non-empty fake ONNX files."""
    # 1. Non-existent file
    valid, info = verify_model_manifest(str(tmp_path / "non_existent.onnx"))
    assert valid is False
    assert "error" in info

    # 2. Corrupt/empty file (0 bytes)
    empty_file = tmp_path / "empty_model.onnx"
    empty_file.write_bytes(b"")
    valid_empty, info_empty = verify_model_manifest(str(empty_file))
    assert info_empty["isNonEmpty"] is False

    # 3. Non-empty file (> 1024 bytes)
    dummy_onnx = tmp_path / "valid_size.onnx"
    dummy_onnx.write_bytes(b"x" * 2048)
    valid_non_empty, info_non_empty = verify_model_manifest(str(dummy_onnx))
    assert info_non_empty["isNonEmpty"] is True
    assert info_non_empty["sizeBytes"] == 2048


def test_rtmpose_upright_coordinate_frame():
    """
    Verifies that RTMPose generates 3D landmarks consistent with MediaPipe's Down-is-+Y convention,
    so that downstream SMPL-X conversion (lm[:, 1] = -mp[:, 1]) produces an upright avatar
    (head above pelvis) rather than an inverted/handstand posture (Roll Z ≈ 180°).
    """
    from video_pose_extractor import convert_mediapipe_landmarks_to_smplx_3d

    backend = RTMPoseBackend()
    backend.is_initialized = True

    # Person standing upright:
    # Nose (0): y=0.2 (top of image)
    # Shoulders (5, 6): y=0.35
    # Hips (11, 12): y=0.55
    # Knees (13, 14): y=0.75
    # Ankles (15, 16): y=0.90 (bottom of image)
    coords_17 = np.zeros((17, 2), dtype=np.float32)
    coords_17[0] = [0.5, 0.20]   # nose
    coords_17[1] = [0.48, 0.18]  # l_eye
    coords_17[2] = [0.52, 0.18]  # r_eye
    coords_17[3] = [0.45, 0.20]  # l_ear
    coords_17[4] = [0.55, 0.20]  # r_ear
    coords_17[5] = [0.42, 0.35]  # l_shoulder
    coords_17[6] = [0.58, 0.35]  # r_shoulder
    coords_17[7] = [0.38, 0.45]  # l_elbow
    coords_17[8] = [0.62, 0.45]  # r_elbow
    coords_17[9] = [0.36, 0.55]  # l_wrist
    coords_17[10] = [0.64, 0.55] # r_wrist
    coords_17[11] = [0.45, 0.55] # l_hip
    coords_17[12] = [0.55, 0.55] # r_hip
    coords_17[13] = [0.45, 0.75] # l_knee
    coords_17[14] = [0.55, 0.75] # r_knee
    coords_17[15] = [0.45, 0.90] # l_ankle
    coords_17[16] = [0.55, 0.90] # r_ankle
    scores_17 = np.full((17,), 0.90, dtype=np.float32)

    backend._decode_simcc = lambda outputs, sx, sy, px, py, ow, oh: (coords_17, scores_17)
    class MockSession:
        def run(self, output_names, feeds):
            return [None, None]
    backend._session = MockSession()
    backend._input_name = "input"
    backend._output_names = ["simcc_x", "simcc_y"]
    backend._preprocess = lambda img: (np.zeros((1, 3, 256, 192), dtype=np.float32), 1.0, 1.0, 0, 0)

    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    obs = backend.detect(dummy_frame, timestamp_sec=0.0, frame_index=0)
    assert obs is not None

    # In MediaPipe World convention, origin is at pelvis (0,0,0) and Down is +Y
    # Therefore, head landmarks must have negative Y (upward), and ankle landmarks must have positive Y (downward)
    nose_3d = obs.landmarks_3d[0]
    l_ankle_3d = obs.landmarks_3d[27]
    assert nose_3d.y < 0.0, f"Head 3D Y should be negative (upward in MediaPipe convention), got: {nose_3d.y}"
    assert l_ankle_3d.y > 0.0, f"Ankle 3D Y should be positive (downward in MediaPipe convention), got: {l_ankle_3d.y}"

    # Convert to SMPL-X joints (33 -> 22)
    mp_landmarks = np.zeros((33, 3), dtype=np.float64)
    for i in range(33):
        mp_landmarks[i] = [obs.landmarks_3d[i].x, obs.landmarks_3d[i].y, obs.landmarks_3d[i].z]
    
    joints = convert_mediapipe_landmarks_to_smplx_3d(mp_landmarks)
    pelvis = joints[0]
    head = joints[15]
    l_ankle = joints[7]

    # In TexMotion SMPL-X space, Up is +Y: Head must be distinctly above Pelvis, and Ankles below Pelvis!
    assert head[1] > pelvis[1] + 0.30, f"Avatar head must be above pelvis! head_y={head[1]}, pelvis_y={pelvis[1]}"
    assert l_ankle[1] < pelvis[1] - 0.40, f"Avatar ankles must be below pelvis! ankle_y={l_ankle[1]}, pelvis_y={pelvis[1]}"

