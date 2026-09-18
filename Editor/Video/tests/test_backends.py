"""
test_backends.py - Tests for detector backends, lifecycle, and fallback mechanics.
"""

import pytest
import numpy as np
from pose_pipeline.backends.base import PoseBackend, BackendCapabilities
from pose_pipeline.backends.mediapipe_backend import MediaPipeBackend, check_mediapipe_available
from pose_pipeline.backends.rtmpose_backend import RTMPoseBackend, check_backend_available
from pose_pipeline.backends import create_pose_backend


def test_mediapipe_backend_capabilities():
    backend = MediaPipeBackend()
    caps = backend.get_capabilities()
    assert caps.name == "mediapipe"
    assert caps.has_2d is True
    assert caps.has_3d is True
    assert caps.keypoint_count == 33


def test_mediapipe_backend_lifecycle(mock_frame_bgr):
    """Verifies MediaPipe backend lifecycle: is_available, close, and detect on uninitialized."""
    backend = MediaPipeBackend()
    avail = backend.is_available()
    assert isinstance(avail, bool)

    # Detect on uninitialized backend returns None safely
    assert backend.detect(mock_frame_bgr, timestamp_sec=0.0) is None

    # Idempotent close
    backend.close()
    backend.close()
    assert backend.is_initialized is False


def test_rtmpose_backend_capabilities():
    backend = RTMPoseBackend()
    caps = backend.get_capabilities()
    assert caps.name == "rtmpose"
    assert caps.has_2d is True
    assert caps.has_3d is False
    assert caps.keypoint_count == 33


def test_rtmpose_fallback_when_model_missing(mock_frame_bgr):
    # Pass a non-existent model path
    backend = RTMPoseBackend(model_path="non_existent_model_file.onnx")
    # is_available should be False
    assert backend.is_available() is False
    # initialize should gracefully return False without raising an unhandled exception
    assert backend.initialize() is False
    # detect should return None safely
    res = backend.detect(mock_frame_bgr, timestamp_sec=0.0)
    assert res is None
    backend.close()


def test_create_pose_backend_factory_fallback():
    """
    Verifies that create_pose_backend gracefully falls back to MediaPipe
    when RTMPose weights or dependencies are absent.
    """
    # preferred = "rtmpose" with non-existent model -> fallback to MediaPipe
    backend_rtm = create_pose_backend(preferred="rtmpose", model_path="missing_path_xyz.onnx")
    assert isinstance(backend_rtm, MediaPipeBackend)
    backend_rtm.close()

    # preferred = "auto" with non-existent model -> fallback to MediaPipe
    backend_auto = create_pose_backend(preferred="auto", model_path="missing_path_xyz.onnx")
    assert isinstance(backend_auto, MediaPipeBackend)
    backend_auto.close()

    # preferred = "mediapipe" -> MediaPipe
    backend_mp = create_pose_backend(preferred="mediapipe")
    assert isinstance(backend_mp, MediaPipeBackend)
    backend_mp.close()


def test_preflight_checks():
    """Verifies preflight checks for both backends."""
    is_mp, _ = check_mediapipe_available()
    assert isinstance(is_mp, bool)

    is_rtm, reason = check_backend_available("non_existent.onnx")
    assert is_rtm is False
    assert isinstance(reason, str)
