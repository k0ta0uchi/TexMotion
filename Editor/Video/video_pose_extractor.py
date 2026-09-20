#!/usr/bin/env python3
"""
video_pose_extractor.py - TexMotion Video-to-Motion 3D Pose Extraction Pipeline

Extracts 3D humanoid motion from 2D video using a detector-specific backend
(MediaPipe/RTMPose or the bundled/native WHAM quality core, with optional
HMR2/HybrIK adapters) and analytical
2-bone inverse kinematics (IK) matching TexMotion's SMPL-X 22 joint hierarchy
(defined in SmplxJointDefinitions.cs).

Features:
- Extracts 33 3D landmarks per video frame.
- Solves 2-bone analytical IK for arms and legs, computing 22 SMPL-X local rotations.
- Detects foot contacts and applies floor snapping (foot locking) to eliminate sliding.
- In-Place motion processing (zeroes horizontal XZ root motion, preserves vertical bounce).
- Temporal smoothing via Savitzky-Golay / continuous quaternion filtering.
- Streaming progress reporting via JSON Lines to stdout.
- Exports motion data matching the VideoMotionData schema.
- Resilient MediaPipe fallback with requested/actual backend provenance.
"""

import argparse
import json
import math
import os
import sys
import tempfile
import time
import shutil
import subprocess
import numpy as np

import urllib.request
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any, Union


# MediaPipe Hybrid API Detection
MEDIAPIPE_MODE = None  # 'tasks', 'solutions', or None

try:
    import mediapipe as mp
    try:
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions, RunningMode
        MEDIAPIPE_MODE = 'tasks'
    except (ImportError, AttributeError):
        if hasattr(mp, "solutions") and hasattr(mp.solutions, "pose"):
            MEDIAPIPE_MODE = 'solutions'
except ImportError:
    MEDIAPIPE_MODE = None

MEDIAPIPE_AVAILABLE = MEDIAPIPE_MODE is not None

try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False

try:
    from scipy.signal import savgol_filter
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

# Pose Pipeline Modular Hybrid Architecture (Phases 1-3)
POSE_PIPELINE_IMPORT_ERROR = None
try:
    from pose_pipeline.observations import UncertaintyInterval
    from pose_pipeline.kinematics import BoneLengthModel, constrain_knee_flexion
    from pose_pipeline.sequence_fit import SequenceOptimizer
    from pose_pipeline.face_pipeline import FacePipeline
    from pose_pipeline.temporal_repair import repair_short_gaps_slerp, repair_long_gaps_kinematic_hermite
    from pose_pipeline.fusion import FusionConfig, fuse_wham_mediapipe, align_observation_sequences, _sequence_values
    from pose_pipeline.backends.rtmpose_backend import RTMPoseBackend
    from pose_pipeline.backends.mediapipe_backend import MediaPipeBackend
    from pose_pipeline.backends import create_pose_backend, PyTorchPoseBackend
    POSE_PIPELINE_AVAILABLE = True
except Exception as exc:
    POSE_PIPELINE_IMPORT_ERROR = str(exc)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    try:
        from pose_pipeline.observations import UncertaintyInterval
        from pose_pipeline.kinematics import BoneLengthModel, constrain_knee_flexion
        from pose_pipeline.sequence_fit import SequenceOptimizer
        from pose_pipeline.face_pipeline import FacePipeline
        from pose_pipeline.temporal_repair import repair_short_gaps_slerp, repair_long_gaps_kinematic_hermite
        from pose_pipeline.fusion import FusionConfig, fuse_wham_mediapipe, align_observation_sequences, _sequence_values
        from pose_pipeline.backends.rtmpose_backend import RTMPoseBackend
        from pose_pipeline.backends.mediapipe_backend import MediaPipeBackend
        from pose_pipeline.backends import create_pose_backend, PyTorchPoseBackend
        POSE_PIPELINE_AVAILABLE = True
    except Exception as retry_exc:
        POSE_PIPELINE_IMPORT_ERROR = str(retry_exc) or POSE_PIPELINE_IMPORT_ERROR
        POSE_PIPELINE_AVAILABLE = False
        FusionConfig = None
        fuse_wham_mediapipe = None
        FacePipeline = None
        repair_short_gaps_slerp = None
        repair_long_gaps_kinematic_hermite = None

# ============================================================================
# SMPL-X 22 Joint Hierarchy (Matching SmplxJointDefinitions.cs)
# ============================================================================

SMPLX_JOINT_NAMES = [
    "Pelvis",      # 0
    "L_Hip",       # 1
    "R_Hip",       # 2
    "Spine1",      # 3
    "L_Knee",      # 4
    "R_Knee",      # 5
    "Spine2",      # 6
    "L_Ankle",     # 7
    "R_Ankle",     # 8
    "Spine3",      # 9
    "L_Foot",      # 10
    "R_Foot",      # 11
    "Neck",        # 12
    "L_Collar",    # 13
    "R_Collar",    # 14
    "Head",        # 15
    "L_Shoulder",  # 16
    "R_Shoulder",  # 17
    "L_Elbow",     # 18
    "R_Elbow",     # 19
    "L_Wrist",     # 20
    "R_Wrist",     # 21
]

SMPLX_PARENTS = [
    -1,  # 0: Pelvis
     0,  # 1: L_Hip
     0,  # 2: R_Hip
     0,  # 3: Spine1
     1,  # 4: L_Knee
     2,  # 5: R_Knee
     3,  # 6: Spine2
     4,  # 7: L_Ankle
     5,  # 8: R_Ankle
     6,  # 9: Spine3
     7,  # 10: L_Foot
     8,  # 11: R_Foot
     9,  # 12: Neck
     9,  # 13: L_Collar
     9,  # 14: R_Collar
    12,  # 15: Head
    13,  # 16: L_Shoulder
    14,  # 17: R_Shoulder
    16,  # 18: L_Elbow
    17,  # 19: R_Elbow
    18,  # 20: L_Wrist
    19,  # 21: R_Wrist
]

SMPLX_JOINT_COUNT = 22

# MediaPipe 33 Landmark Indices
MP_NOSE = 0
MP_LEFT_EYE_INNER = 1
MP_LEFT_EYE = 2
MP_LEFT_EYE_OUTER = 3
MP_RIGHT_EYE_INNER = 4
MP_RIGHT_EYE = 5
MP_RIGHT_EYE_OUTER = 6
MP_LEFT_EAR = 7
MP_RIGHT_EAR = 8
MP_MOUTH_LEFT = 9
MP_MOUTH_RIGHT = 10
MP_LEFT_SHOULDER = 11
MP_RIGHT_SHOULDER = 12
MP_LEFT_ELBOW = 13
MP_RIGHT_ELBOW = 14
MP_LEFT_WRIST = 15
MP_RIGHT_WRIST = 16
MP_LEFT_PINKY = 17
MP_RIGHT_PINKY = 18
MP_LEFT_INDEX = 19
MP_RIGHT_INDEX = 20
MP_LEFT_THUMB = 21
MP_RIGHT_THUMB = 22
MP_LEFT_HIP = 23
MP_RIGHT_HIP = 24
MP_LEFT_KNEE = 25
MP_RIGHT_KNEE = 26
MP_LEFT_ANKLE = 27
MP_RIGHT_ANKLE = 28
MP_LEFT_HEEL = 29
MP_RIGHT_HEEL = 30
MP_LEFT_FOOT_INDEX = 31
MP_RIGHT_FOOT_INDEX = 32

# ============================================================================
# MediaPipe Tracker Backend & Default Pose
# ============================================================================

def get_default_tpose_landmarks(invert_y: bool = True) -> np.ndarray:
    """
    Returns valid default standing humanoid landmarks in the legacy camera
    stream.  ``invert_y=True`` keeps the raw image/world convention (+Y down)
    used by MediaPipe and RTMPose; ``invert_y=False`` returns canonical SMPL-X
    (+Y up) coordinates for a quality-backend placeholder.
    """
    lms = np.zeros((33, 3), dtype=np.float64)
    lms[MP_NOSE] = [0.0, -0.85, -0.05]
    lms[MP_LEFT_EYE] = [0.03, -0.87, -0.04]
    lms[MP_RIGHT_EYE] = [-0.03, -0.87, -0.04]
    lms[MP_LEFT_EAR] = [0.09, -0.85, 0.0]
    lms[MP_RIGHT_EAR] = [-0.09, -0.85, 0.0]
    lms[MP_LEFT_SHOULDER] = [0.20, -0.65, 0.0]
    lms[MP_RIGHT_SHOULDER] = [-0.20, -0.65, 0.0]
    lms[MP_LEFT_ELBOW] = [0.45, -0.65, 0.0]
    lms[MP_RIGHT_ELBOW] = [-0.45, -0.65, 0.0]
    lms[MP_LEFT_WRIST] = [0.70, -0.65, 0.0]
    lms[MP_RIGHT_WRIST] = [-0.70, -0.65, 0.0]
    lms[MP_LEFT_HIP] = [0.10, 0.0, 0.0]
    lms[MP_RIGHT_HIP] = [-0.10, 0.0, 0.0]
    lms[MP_LEFT_KNEE] = [0.10, 0.45, 0.0]
    lms[MP_RIGHT_KNEE] = [-0.10, 0.45, 0.0]
    lms[MP_LEFT_ANKLE] = [0.10, 0.85, 0.0]
    lms[MP_RIGHT_ANKLE] = [-0.10, 0.85, 0.0]
    lms[MP_LEFT_FOOT_INDEX] = [0.10, 0.90, -0.15]
    lms[MP_RIGHT_FOOT_INDEX] = [-0.10, 0.90, -0.15]
    if not invert_y:
        lms[:, 1] *= -1.0
    return lms


class MediaPipePoseTracker:
    """
    Hybrid MediaPipe Pose backend supporting both Tasks API (mediapipe >= 0.10.15 / 1.0+)
    and legacy Solutions API (mediapipe <= 0.10.14).
    Uses only locally available .task model bundles when using Tasks API.  The
    editor's model manager is responsible for any explicit downloads; runtime
    extraction must not block on an implicit network request.
    """
    MODEL_URLS = {
        0: "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
        1: "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task",
        2: "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task",
    }
    MODEL_NAMES = {
        0: "pose_landmarker_lite.task",
        1: "pose_landmarker_full.task",
        2: "pose_landmarker_heavy.task",
    }

    def __init__(
        self,
        model_complexity: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        model_directory: str = None,
        allow_download: bool = False,
    ):
        self.mode = MEDIAPIPE_MODE
        self.model_complexity = model_complexity
        self.model_directory = model_directory
        self.allow_download = bool(allow_download)
        self.landmarker = None
        self.solution_tracker = None
        # MediaPipe Tasks VIDEO graphs reject non-increasing timestamps.  A
        # WHAM run probes one seed frame before its aligned fusion pass, so
        # clamp timestamps here instead of destroying/recreating the native
        # graph (which can crash on Windows with an XNNPACK access violation).
        self._last_timestamp_ms = -1

        eff_det_conf = min(0.38, float(min_detection_confidence))
        eff_trk_conf = min(0.38, float(min_tracking_confidence))

        if self.mode == 'tasks':
            model_path = self._ensure_model(model_complexity)
            options = PoseLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=model_path),
                running_mode=RunningMode.VIDEO,
                min_pose_detection_confidence=eff_det_conf,
                min_tracking_confidence=eff_trk_conf
            )
            self.landmarker = PoseLandmarker.create_from_options(options)
            sys.stderr.write(f"[TexMotion] MediaPipe Tasks API initialized with {os.path.basename(model_path)}\n")
            sys.stderr.flush()
        elif self.mode == 'solutions':
            self.solution_tracker = mp.solutions.pose.Pose(
                static_image_mode=False,
                model_complexity=model_complexity,
                smooth_landmarks=True,
                min_detection_confidence=eff_det_conf,
                min_tracking_confidence=eff_trk_conf
            )
            sys.stderr.write("[TexMotion] MediaPipe Solutions API initialized\n")
            sys.stderr.flush()
        else:
            raise RuntimeError("MediaPipe is not installed. Please install it using: pip install mediapipe opencv-python")

    def _ensure_model(self, complexity: int) -> str:
        complexity = min(2, max(0, complexity))
        model_name = self.MODEL_NAMES[complexity]
        url = self.MODEL_URLS[complexity]

        # Search standard cache locations
        configured_dir = self.model_directory or os.environ.get("TEXMOTION_VIDEO_MODEL_DIR", "")
        app_data = os.environ.get("APPDATA", "")
        default_video_cache = (
            os.path.join(app_data, "TexMotion", "Models", "Video") if app_data else ""
        )
        candidate_dirs = [
            configured_dir,
            default_video_cache,
            os.path.join(os.environ.get("APPDATA", ""), "TexMotion", "Models"),
            os.path.join(os.path.expanduser("~"), ".cache", "texmotion", "models"),
            os.path.join(tempfile.gettempdir(), "texmotion_models"),
            os.path.dirname(os.path.abspath(__file__)),
        ]

        for c_dir in candidate_dirs:
            if not c_dir:
                continue
            path = os.path.join(c_dir, model_name)
            if os.path.exists(path) and os.path.getsize(path) > 1000000:
                return path

        # Runtime extraction is offline by default.  The editor's model
        # manager can place a task bundle in one of the directories above
        # before starting a job; do not silently download from a moving URL.
        if not self.allow_download:
            raise FileNotFoundError(
                f"MediaPipe task model {model_name} was not found locally. "
                "Install it from the Video Models panel before extraction."
            )

        # If explicitly enabled by a legacy caller, download to the primary
        # cache directory.
        target_dir = configured_dir or default_video_cache or (
            candidate_dirs[2] if candidate_dirs[2] else candidate_dirs[4]
        )
        os.makedirs(target_dir, exist_ok=True)
        target_path = os.path.join(target_dir, model_name)

        sys.stderr.write(f"[TexMotion] Downloading {model_name} (~9MB)...\n")
        sys.stderr.flush()
        try:
            urllib.request.urlretrieve(url, target_path)
            sys.stderr.write(f"[TexMotion] Downloaded successfully: {target_path}\n")
            sys.stderr.flush()
            return target_path
        except Exception as e:
            # Fallback to temp dir
            fallback_path = os.path.join(tempfile.gettempdir(), model_name)
            if not os.path.exists(fallback_path):
                urllib.request.urlretrieve(url, fallback_path)
            return fallback_path

    def detect(self, frame_bgr: np.ndarray, timestamp_ms: int) -> tuple:
        """
        Detects 33 3D world landmarks and 2D normalized image landmarks from frame_bgr.
        Returns: (world_landmarks, norm_landmarks, confidence) or (None, None, 0.0)
        """
        rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        def _get_conf(lm):
            v = getattr(lm, 'visibility', None)
            p = getattr(lm, 'presence', None)
            if v is not None and p is not None:
                return float(max(v, p))
            elif v is not None:
                return float(v)
            elif p is not None:
                return float(p)
            return 1.0

        if self.mode == 'tasks' and self.landmarker is not None:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            requested_timestamp_ms = int(timestamp_ms)
            timestamp_ms = max(self._last_timestamp_ms + 1, requested_timestamp_ms)
            self._last_timestamp_ms = timestamp_ms
            results = self.landmarker.detect_for_video(mp_image, int(timestamp_ms))
            if results.pose_world_landmarks and len(results.pose_world_landmarks) > 0:
                lms_world = results.pose_world_landmarks[0]
                world_landmarks = np.zeros((33, 3), dtype=np.float64)
                vis_sum = 0.0
                for idx, lm in enumerate(lms_world):
                    world_landmarks[idx] = [lm.x, lm.y, lm.z]
                    vis_sum += _get_conf(lm)
                conf = vis_sum / 33.0

                norm_landmarks = None
                if results.pose_landmarks and len(results.pose_landmarks) > 0:
                    lms_norm = results.pose_landmarks[0]
                    norm_landmarks = np.zeros((33, 3), dtype=np.float64)
                    for idx, lm in enumerate(lms_norm):
                        norm_landmarks[idx] = [lm.x, lm.y, _get_conf(lm)]

                return world_landmarks, norm_landmarks, conf
            return None, None, 0.0

        elif self.mode == 'solutions' and self.solution_tracker is not None:
            results = self.solution_tracker.process(rgb_frame)
            if results.pose_world_landmarks:
                world_landmarks = np.zeros((33, 3), dtype=np.float64)
                vis_sum = 0.0
                for idx, lm in enumerate(results.pose_world_landmarks.landmark):
                    world_landmarks[idx] = [lm.x, lm.y, lm.z]
                    vis_sum += _get_conf(lm)
                conf = vis_sum / 33.0

                norm_landmarks = None
                if results.pose_landmarks:
                    norm_landmarks = np.zeros((33, 3), dtype=np.float64)
                    for idx, lm in enumerate(results.pose_landmarks.landmark):
                        norm_landmarks[idx] = [lm.x, lm.y, _get_conf(lm)]

                return world_landmarks, norm_landmarks, conf
            return None, None, 0.0

        return None, None, 0.0

    def close(self):
        try:
            if self.landmarker is not None:
                self.landmarker.close()
        except Exception:
            pass
        self.landmarker = None
        self._last_timestamp_ms = -1
        try:
            if self.solution_tracker is not None:
                self.solution_tracker.close()
        except Exception:
            pass
        self.solution_tracker = None


class _SharedMediaPipeAuxiliary:
    """Legacy tuple facade over a quality backend's MediaPipe detector.

    Native WHAM already creates a ``MediaPipeBackend`` for its image-side
    keypoints.  The fusion code historically created a second legacy tracker
    for world landmarks, which means multiple TFLite graphs (and a close/
    recreate cycle after the seed probe) on Windows.  Reusing the native
    detector avoids that unsafe lifecycle while keeping the extractor's tuple
    contract unchanged.
    """

    shared_backend = True

    def __init__(self, backend):
        self._backend = backend

    def detect(self, frame_bgr: np.ndarray, timestamp_ms: int) -> tuple:
        observation = self._backend.detect(
            frame_bgr,
            timestamp_sec=float(timestamp_ms) / 1000.0,
            frame_index=0,
        )
        if observation is None:
            return None, None, 0.0

        world = None
        landmarks_3d = getattr(observation, "landmarks_3d", None) or []
        if landmarks_3d:
            world = np.asarray(
                [[float(point.x), float(point.y), float(point.z)] for point in landmarks_3d],
                dtype=np.float64,
            )

        norm = None
        keypoints_2d = getattr(observation, "keypoints_2d", None) or []
        if keypoints_2d:
            norm = np.asarray(
                [[float(point.x), float(point.y), float(point.score)] for point in keypoints_2d],
                dtype=np.float64,
            )

        confidence = float(getattr(observation, "raw_confidence", 0.0) or 0.0)
        return world, norm, confidence

    def close(self):
        # The PyTorch/WHAM owner closes the shared detector.  This facade must
        # never close it independently or the quality backend will crash when
        # it tries to consume its detector later in the frame loop.
        return None


# ============================================================================
# Symmetry Pairs & Temporal Landmark Stabilization
# ============================================================================

SYMMETRY_PAIRS = [
    (MP_LEFT_SHOULDER, MP_RIGHT_SHOULDER),
    (MP_LEFT_ELBOW, MP_RIGHT_ELBOW),
    (MP_LEFT_WRIST, MP_RIGHT_WRIST),
    (MP_LEFT_PINKY, MP_RIGHT_PINKY),
    (MP_LEFT_INDEX, MP_RIGHT_INDEX),
    (MP_LEFT_THUMB, MP_RIGHT_THUMB),
    (MP_LEFT_HIP, MP_RIGHT_HIP),
    (MP_LEFT_KNEE, MP_RIGHT_KNEE),
    (MP_LEFT_ANKLE, MP_RIGHT_ANKLE),
    (MP_LEFT_HEEL, MP_RIGHT_HEEL),
    (MP_LEFT_FOOT_INDEX, MP_RIGHT_FOOT_INDEX),
    (MP_LEFT_EYE_INNER, MP_RIGHT_EYE_INNER),
    (MP_LEFT_EYE, MP_RIGHT_EYE),
    (MP_LEFT_EYE_OUTER, MP_RIGHT_EYE_OUTER),
    (MP_LEFT_EAR, MP_RIGHT_EAR),
    (MP_MOUTH_LEFT, MP_MOUTH_RIGHT),
]


def swap_landmarks(lms: np.ndarray) -> np.ndarray:
    """Swaps left and right anatomical landmarks to correct MediaPipe label inversion."""
    if lms is None:
        return None
    out = np.copy(lms)
    for l_idx, r_idx in SYMMETRY_PAIRS:
        out[l_idx] = lms[r_idx]
        out[r_idx] = lms[l_idx]
    return out


class RobustPoseTracker:
    """
    Prevents MediaPipe's sudden left/right and front/back label inversion bug.
    Uses face landmark visibility and temporal shoulder continuity to detect
    and automatically correct label swaps across video frames.
    """
    def __init__(self):
        self.prev_norm = None
        self.swaps_fixed = 0

    def process(self, world_lms: np.ndarray, norm_lms: np.ndarray) -> tuple:
        if world_lms is None or norm_lms is None:
            return world_lms, norm_lms

        w = np.copy(world_lms)
        n = np.copy(norm_lms)

        ls_x = n[MP_LEFT_SHOULDER, 0]
        rs_x = n[MP_RIGHT_SHOULDER, 0]
        face_vis = (n[MP_NOSE, 2] + n[MP_LEFT_EYE, 2] + n[MP_RIGHT_EYE, 2]) / 3.0
        shoulder_width = abs(ls_x - rs_x)

        needs_swap = False
        if face_vis > 0.35:
            # When face is clearly visible, subject is facing camera:
            # Subject's left shoulder must be on image right (ls_x > rs_x).
            if ls_x < rs_x:
                needs_swap = True
        elif self.prev_norm is not None:
            prev_ls_x = self.prev_norm[MP_LEFT_SHOULDER, 0]
            prev_rs_x = self.prev_norm[MP_RIGHT_SHOULDER, 0]
            # Detect sudden jump/swap without physically rotating through narrow profile
            if (ls_x < rs_x) and (prev_ls_x > prev_rs_x) and shoulder_width > 0.04:
                needs_swap = True

        if needs_swap:
            w = swap_landmarks(w)
            n = swap_landmarks(n)
            self.swaps_fixed += 1

        self.prev_norm = np.copy(n)
        return w, n


class OneEuroFilter:
    """
    1-Euro Filter for low-latency, adaptive filtering of noisy 3D landmark streams.
    Reference: Casiez et al., CHI 2012.
    """
    def __init__(self, freq: float = 30.0, mincutoff: float = 1.2, beta: float = 0.02, dcutoff: float = 1.0):
        self.freq = float(freq)
        self.mincutoff = float(mincutoff)
        self.beta = float(beta)
        self.dcutoff = float(dcutoff)
        self.x_prev = None
        self.dx_prev = None

    def _alpha(self, cutoff: float) -> float:
        te = 1.0 / max(1e-4, self.freq)
        tau = 1.0 / (2.0 * np.pi * cutoff)
        return 1.0 / (1.0 + tau / te)

    def filter(self, x: np.ndarray) -> np.ndarray:
        if self.x_prev is None:
            self.x_prev = np.copy(x)
            self.dx_prev = np.zeros_like(x)
            return np.copy(x)

        dx = (x - self.x_prev) * self.freq
        edx = self._alpha(self.dcutoff) * dx + (1.0 - self._alpha(self.dcutoff)) * self.dx_prev
        self.dx_prev = edx

        cutoff = self.mincutoff + self.beta * np.abs(edx)
        alpha = self._alpha(cutoff)

        x_hat = alpha * x + (1.0 - alpha) * self.x_prev
        self.x_prev = x_hat
        return x_hat


class OneEuroFilter3D:
    """
    Applies adaptive 1-Euro filtering to all 33 3D landmarks independently,
    effectively eliminating depth jitter (Z-flutter) while preserving responsive motion.
    """
    def __init__(self, num_points: int = 33, freq: float = 30.0, mincutoff: float = 1.2, beta: float = 0.02):
        self.filters = [OneEuroFilter(freq=freq, mincutoff=mincutoff, beta=beta) for _ in range(num_points)]

    def filter(self, landmarks: np.ndarray) -> np.ndarray:
        if landmarks is None:
            return None
        out = np.zeros_like(landmarks)
        for i in range(len(landmarks)):
            out[i] = self.filters[i].filter(landmarks[i])
        return out


# MediaPipe 2D / 3D Visualization Connections
POSE_CONNECTIONS_TORSO = [
    (MP_LEFT_SHOULDER, MP_RIGHT_SHOULDER),
    (MP_LEFT_SHOULDER, MP_LEFT_HIP),
    (MP_RIGHT_SHOULDER, MP_RIGHT_HIP),
    (MP_LEFT_HIP, MP_RIGHT_HIP),
]

POSE_CONNECTIONS_FACE = [
    (MP_NOSE, MP_LEFT_EYE_INNER), (MP_LEFT_EYE_INNER, MP_LEFT_EYE), (MP_LEFT_EYE, MP_LEFT_EYE_OUTER), (MP_LEFT_EYE_OUTER, MP_LEFT_EAR),
    (MP_NOSE, MP_RIGHT_EYE_INNER), (MP_RIGHT_EYE_INNER, MP_RIGHT_EYE), (MP_RIGHT_EYE, MP_RIGHT_EYE_OUTER), (MP_RIGHT_EYE_OUTER, MP_RIGHT_EAR),
    (MP_MOUTH_LEFT, MP_MOUTH_RIGHT),
]

POSE_CONNECTIONS_LEFT_ARM = [
    (MP_LEFT_SHOULDER, MP_LEFT_ELBOW),
    (MP_LEFT_ELBOW, MP_LEFT_WRIST),
    (MP_LEFT_WRIST, MP_LEFT_PINKY),
    (MP_LEFT_WRIST, MP_LEFT_INDEX),
    (MP_LEFT_WRIST, MP_LEFT_THUMB),
    (MP_LEFT_PINKY, MP_LEFT_INDEX),
]

POSE_CONNECTIONS_RIGHT_ARM = [
    (MP_RIGHT_SHOULDER, MP_RIGHT_ELBOW),
    (MP_RIGHT_ELBOW, MP_RIGHT_WRIST),
    (MP_RIGHT_WRIST, MP_RIGHT_PINKY),
    (MP_RIGHT_WRIST, MP_RIGHT_INDEX),
    (MP_RIGHT_WRIST, MP_RIGHT_THUMB),
    (MP_RIGHT_PINKY, MP_RIGHT_INDEX),
]

POSE_CONNECTIONS_LEFT_LEG = [
    (MP_LEFT_HIP, MP_LEFT_KNEE),
    (MP_LEFT_KNEE, MP_LEFT_ANKLE),
    (MP_LEFT_ANKLE, MP_LEFT_HEEL),
    (MP_LEFT_HEEL, MP_LEFT_FOOT_INDEX),
    (MP_LEFT_ANKLE, MP_LEFT_FOOT_INDEX),
]

POSE_CONNECTIONS_RIGHT_LEG = [
    (MP_RIGHT_HIP, MP_RIGHT_KNEE),
    (MP_RIGHT_KNEE, MP_RIGHT_ANKLE),
    (MP_RIGHT_ANKLE, MP_RIGHT_HEEL),
    (MP_RIGHT_HEEL, MP_RIGHT_FOOT_INDEX),
    (MP_RIGHT_ANKLE, MP_RIGHT_FOOT_INDEX),
]


def draw_pose_overlay(
    frame_bgr: np.ndarray,
    norm_landmarks: np.ndarray,
    frame_idx: int,
    total_frames: int,
    conf: float,
    backend_name: str = "MediaPipe",
    *,
    secondary_landmarks: np.ndarray = None,
    overlay_mode: str = "dual",
) -> np.ndarray:
    """
    Renders high-aesthetic pose skeleton lines and joint points onto frame_bgr.
    Supports:
    - Dual mode: 2D detector guide (subtle cyan/white) + 3D pose projection (vibrant neon).
    - 3D mode: 3D model hypothesis projection only.
    - 2D mode: Direct 2D keypoint tracking only.
    Uses vibrant color-coded body segments:
    - Torso: Neon Magenta
    - Left Arm: Neon Cyan
    - Right Arm: Neon Coral / Orange
    - Left Leg: Neon Sky Blue
    - Right Leg: Neon Red / Orange
    - Face: Neon Lime Green
    Also renders a sleek semi-transparent HUD badge with frame index, backend name, mode, and confidence score.
    """
    out = frame_bgr.copy()
    h, w, _ = out.shape

    mode_key = str(overlay_mode or "dual").strip().lower()
    if mode_key not in ("dual", "3d", "2d"):
        mode_key = "dual"

    # Connections and styles: (connections, BGR color, thickness)
    connection_groups = [
        (POSE_CONNECTIONS_TORSO, (235, 65, 235), 3),      # Torso: Magenta
        (POSE_CONNECTIONS_LEFT_ARM, (255, 215, 0), 3),    # Left arm: Cyan
        (POSE_CONNECTIONS_RIGHT_ARM, (0, 140, 255), 3),   # Right arm: Coral
        (POSE_CONNECTIONS_LEFT_LEG, (255, 170, 50), 3),   # Left leg: Sky Blue
        (POSE_CONNECTIONS_RIGHT_LEG, (40, 80, 255), 3),   # Right leg: Bright Orange
        (POSE_CONNECTIONS_FACE, (80, 235, 80), 2),        # Face: Lime
    ]

    def _extract_pixel_pts(landmarks):
        if landmarks is None or len(landmarks) < 33:
            return None, None
        pts = []
        vis = []
        for i in range(33):
            lx, ly = landmarks[i][0], landmarks[i][1]
            v = landmarks[i][2] if len(landmarks[i]) > 2 else 1.0
            px = int(np.clip(lx * w, 0, w - 1))
            py = int(np.clip(ly * h, 0, h - 1))
            pts.append((px, py))
            vis.append(v)

        if len(pts) > 28:
            dx_ank = pts[MP_LEFT_ANKLE][0] - pts[MP_RIGHT_ANKLE][0]
            if dx_ank < int(w * 0.035):
                if pts[MP_RIGHT_ANKLE][0] >= pts[MP_LEFT_ANKLE][0] - int(w * 0.015):
                    pts[MP_RIGHT_ANKLE] = (max(pts[MP_RIGHT_ANKLE][0], pts[MP_LEFT_ANKLE][0] + int(w * 0.025)), pts[MP_RIGHT_ANKLE][1])
                    pts[MP_RIGHT_KNEE] = (max(pts[MP_RIGHT_KNEE][0], pts[MP_LEFT_KNEE][0] - int(w * 0.010)), pts[MP_RIGHT_KNEE][1])
        return pts, vis

    pts_main, vis_main = _extract_pixel_pts(norm_landmarks)
    pts_sec, vis_sec = _extract_pixel_pts(secondary_landmarks)

    # 1. Dual Mode: Render 2D detector guide skeleton in subtle semi-transparent layer
    if mode_key == "dual" and pts_sec is not None:
        guide_layer = out.copy()
        guide_color = (255, 230, 160)  # Soft cyan-white guide line
        for connections, _, _ in connection_groups:
            for j1, j2 in connections:
                if vis_sec[j1] > 0.15 and vis_sec[j2] > 0.15:
                    cv2.line(guide_layer, pts_sec[j1], pts_sec[j2], guide_color, 2, cv2.LINE_AA)
        for i in range(33):
            if vis_sec[i] > 0.15:
                cv2.circle(guide_layer, pts_sec[i], 3, (240, 240, 240), -1, cv2.LINE_AA)
        # Blend guide skeleton into background (45% guide, 55% original)
        cv2.addWeighted(guide_layer, 0.45, out, 0.55, 0, out)

    # 2. Main Pose Skeleton: Vibrant neon rendering
    if pts_main is not None:
        for connections, color, thickness in connection_groups:
            for j1, j2 in connections:
                if vis_main[j1] > 0.15 and vis_main[j2] > 0.15:
                    cv2.line(out, pts_main[j1], pts_main[j2], color, thickness, cv2.LINE_AA)

        for i in range(33):
            if vis_main[i] > 0.15:
                pt = pts_main[i]
                # Outer white glow
                cv2.circle(out, pt, 5, (255, 255, 255), -1, cv2.LINE_AA)
                # Inner colored dot
                is_left = i in [MP_LEFT_SHOULDER, MP_LEFT_ELBOW, MP_LEFT_WRIST, MP_LEFT_HIP, MP_LEFT_KNEE, MP_LEFT_ANKLE]
                dot_color = (255, 200, 0) if is_left else (0, 140, 255)
                cv2.circle(out, pt, 3, dot_color, -1, cv2.LINE_AA)

    # 3. Semi-transparent HUD overlay on top-left
    hud_w, hud_h = 320, 58
    if w >= hud_w + 20 and h >= hud_h + 20:
        hud_roi = out[12:12 + hud_h, 12:12 + hud_w]
        dark_overlay = np.full_like(hud_roi, (24, 26, 32))
        cv2.addWeighted(dark_overlay, 0.82, hud_roi, 0.18, 0, hud_roi)
        out[12:12 + hud_h, 12:12 + hud_w] = hud_roi
        cv2.rectangle(out, (12, 12), (12 + hud_w, 12 + hud_h), (60, 160, 240), 1, cv2.LINE_AA)

        tag = " [DUAL 2D+3D]" if (mode_key == "dual" and pts_sec is not None) else (
            " [3D POSE]" if mode_key == "3d" else (" [2D TRACK]" if mode_key == "2d" else "")
        )
        display_title = f"TexMotion {backend_name.upper()} Overlay{tag}"
        cv2.putText(out, display_title, (22, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        conf_pct = int(round(conf * 100))
        conf_color = (80, 245, 120) if conf_pct >= 70 else ((50, 210, 255) if conf_pct >= 45 else (80, 100, 255))
        sub_desc = "2D Guide + 3D Pose" if (mode_key == "dual" and pts_sec is not None) else (
            "3D Pose Projection" if mode_key == "3d" else "2D Keypoints"
        )
        cv2.putText(out, f"Frame: {frame_idx + 1}/{total_frames} | Conf: {conf_pct}% | {sub_desc}", (22, 53), cv2.FONT_HERSHEY_SIMPLEX, 0.38, conf_color, 1, cv2.LINE_AA)

    return out


def project_quality_3d_to_image(
    landmarks_3d: np.ndarray,
    anchor_norm: np.ndarray = None,
    coordinate_system: str = "legacy_overlay",
) -> np.ndarray:
    """Project a quality backend's 3D hypothesis into normalized image space.

    WHAM/HMR2/HybrIK adapters are allowed to return only body-model joints.
    Passing an empty 2D result to :func:`draw_pose_overlay` used to produce a
    raw video frame, or encouraged the caller to substitute MediaPipe points.
    This lightweight projection keeps the *quality model's* limb geometry in
    the overlay while adapting translation and scale to observed 2D anchors.
    Depth is intentionally not collapsed into 2D; it remains in the exported
    motion data.
    """
    if landmarks_3d is None:
        return None
    arr = np.asarray(landmarks_3d, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] < 33 or arr.shape[1] < 3:
        return None
    arr = arr[:33, :3]
    finite = np.isfinite(arr).all(axis=1)
    # Missing quality slots are represented by zero-valued canonical joints;
    # retain pelvis at the origin but avoid drawing absent limbs at (0, 0).
    nonzero = np.linalg.norm(arr, axis=1) > 1e-7
    valid = finite & (nonzero | (np.arange(33) == 0))
    if int(np.count_nonzero(valid)) < 3:
        return None

    xy = arr[:, :2].copy()
    # ``draw_pose_overlay`` consumes image-space coordinates where Y increases
    # downwards.  Canonical SMPL-X uses +Y up, so convert only the projected
    # image axis here.  Legacy overlay streams already use image axes.
    coordinate_key = str(coordinate_system or "legacy_overlay").strip().lower()
    if coordinate_key in ("smplx", "canonical", "canonical_smplx", "smplx_y_up_z_forward"):
        xy[:, 1] *= -1.0
    valid_xy = xy[valid]
    body_center = np.median(valid_xy, axis=0)
    projected_xy = np.zeros((33, 2), dtype=np.float64)

    norm = None
    norm_pts_valid = np.zeros(33, dtype=bool)
    if anchor_norm is not None:
        cand_norm = np.asarray(anchor_norm, dtype=np.float64)
        if cand_norm.ndim == 2 and cand_norm.shape[0] >= 33 and cand_norm.shape[1] >= 2:
            norm = cand_norm
            norm_pts_valid = np.isfinite(norm[:, :2]).all(axis=1) & (
                (norm[:, 2] > 0.15) if norm.shape[1] > 2 else True
            )

    # Use torso anchors from an independent detector only to place the body in
    # the camera image.  Every non-anchor joint still comes from the quality
    # 3D hypothesis, so crossed/occluded limbs cannot be replaced by MediaPipe.
    anchor_indices = (MP_LEFT_SHOULDER, MP_RIGHT_SHOULDER, MP_LEFT_HIP, MP_RIGHT_HIP, 0)
    target_anchor = []
    source_anchor = []
    if norm is not None:
        for idx in anchor_indices:
            if not valid[idx] or not norm_pts_valid[idx]:
                continue
            point = norm[idx, :2]
            if np.all((point >= -0.05) & (point <= 1.05)):
                source_anchor.append(xy[idx])
                target_anchor.append(point)

    if len(source_anchor) >= 2:
        source_anchor = np.asarray(source_anchor, dtype=np.float64)
        target_anchor = np.asarray(target_anchor, dtype=np.float64)
        source_center = np.mean(source_anchor, axis=0)
        target_center = np.mean(target_anchor, axis=0)

        target_by_index = {
            idx: point for idx, point in zip(anchor_indices, target_anchor)
        }

        def _span(a_idx, b_idx, values):
            a = values.get(a_idx) if isinstance(values, dict) else None
            b = values.get(b_idx) if isinstance(values, dict) else None
            if a is None or b is None:
                return 0.0
            return float(np.linalg.norm(np.asarray(a)[:2] - np.asarray(b)[:2]))

        source_by_index = {idx: xy[idx] for idx in anchor_indices if valid[idx]}
        source_shoulder_span = _span(MP_LEFT_SHOULDER, MP_RIGHT_SHOULDER, source_by_index)
        target_shoulder_span = _span(MP_LEFT_SHOULDER, MP_RIGHT_SHOULDER, target_by_index)
        source_hip_span = _span(MP_LEFT_HIP, MP_RIGHT_HIP, source_by_index)
        target_hip_span = _span(MP_LEFT_HIP, MP_RIGHT_HIP, target_by_index)

        source_torso_span = 0.0
        target_torso_span = 0.0
        if all(idx in source_by_index for idx in (MP_LEFT_SHOULDER, MP_RIGHT_SHOULDER, MP_LEFT_HIP, MP_RIGHT_HIP)):
            source_torso_span = float(np.linalg.norm(
                (source_by_index[MP_LEFT_SHOULDER] + source_by_index[MP_RIGHT_SHOULDER]) * 0.5
                - (source_by_index[MP_LEFT_HIP] + source_by_index[MP_RIGHT_HIP]) * 0.5
            ))
        if all(idx in target_by_index for idx in (MP_LEFT_SHOULDER, MP_RIGHT_SHOULDER, MP_LEFT_HIP, MP_RIGHT_HIP)):
            target_torso_span = float(np.linalg.norm(
                (target_by_index[MP_LEFT_SHOULDER] + target_by_index[MP_RIGHT_SHOULDER]) * 0.5
                - (target_by_index[MP_LEFT_HIP] + target_by_index[MP_RIGHT_HIP]) * 0.5
            ))

        scale_candidates = []
        if source_shoulder_span >= 0.05 and target_shoulder_span >= 0.02:
            scale_candidates.append(target_shoulder_span / source_shoulder_span)
        if source_hip_span >= 0.05 and target_hip_span >= 0.02:
            scale_candidates.append(target_hip_span / source_hip_span)
        if source_torso_span >= 0.10 and target_torso_span >= 0.05:
            scale_candidates.append(target_torso_span / source_torso_span)

        # Include leg vertical extent when hips and ankles are healthy and anatomically plausible
        if (
            valid[MP_LEFT_HIP] and valid[MP_RIGHT_HIP] and valid[MP_LEFT_ANKLE] and valid[MP_RIGHT_ANKLE]
            and norm_pts_valid[MP_LEFT_HIP] and norm_pts_valid[MP_RIGHT_HIP]
            and norm_pts_valid[MP_LEFT_ANKLE] and norm_pts_valid[MP_RIGHT_ANKLE]
        ):
            s_hip_y = (xy[MP_LEFT_HIP, 1] + xy[MP_RIGHT_HIP, 1]) * 0.5
            s_ank_y = (xy[MP_LEFT_ANKLE, 1] + xy[MP_RIGHT_ANKLE, 1]) * 0.5
            t_hip_y = (norm[MP_LEFT_HIP, 1] + norm[MP_RIGHT_HIP, 1]) * 0.5
            t_ank_y = (norm[MP_LEFT_ANKLE, 1] + norm[MP_RIGHT_ANKLE, 1]) * 0.5
            s_leg_span = abs(s_ank_y - s_hip_y)
            t_leg_span = abs(t_ank_y - t_hip_y)
            if s_leg_span >= 0.25 and t_leg_span >= 0.10:
                scale_candidates.append(t_leg_span / s_leg_span)

        if scale_candidates:
            scale = float(np.median(np.asarray(scale_candidates, dtype=np.float64)))
        else:
            source_extent = float(np.ptp(source_anchor[:, 0]))
            target_extent = float(np.ptp(target_anchor[:, 0]))
            scale = target_extent / max(source_extent, 0.10)
        scale = float(np.clip(scale, 0.15, 2.2))
        projected_xy = (xy - source_center) * scale + target_center
    else:
        # No image observations were returned by the adapter.  Fit a stable
        # centered view so the WHAM skeleton remains visible in the overlay.
        extent = np.ptp(valid_xy, axis=0)
        extent = float(max(np.max(extent), 1e-3))
        scale = 0.72 / extent
        projected_xy = (xy - body_center) * scale + np.array([0.5, 0.5], dtype=np.float64)

    # Missing quality slots (detailed face, hand digits) must not be drawn at the
    # chest/pelvis origin. Suppress unmodeled face details or snap hand/foot digits.
    face_detail_slots = (1, 3, 4, 6, 9, 10)
    for s_idx in face_detail_slots:
        if np.linalg.norm(arr[s_idx, :3]) < 1e-5 or (valid[0] and np.linalg.norm(arr[s_idx, :3] - arr[0, :3]) > 0.40):
            valid[s_idx] = False

    # Hand digits (pinky, index, thumb) should snap to wrists when 3D coords are absent
    for digits, wrist_idx in (((17, 19, 21), MP_LEFT_WRIST), ((18, 20, 22), MP_RIGHT_WRIST)):
        for d_idx in digits:
            if np.linalg.norm(arr[d_idx, :3]) < 1e-5 or (valid[wrist_idx] and np.linalg.norm(arr[d_idx, :3] - arr[wrist_idx, :3]) > 0.35):
                if valid[wrist_idx]:
                    projected_xy[d_idx] = projected_xy[wrist_idx]
                    valid[d_idx] = True
                else:
                    valid[d_idx] = False

    # Alias foot tips from valid ankles so feet stay at ankle level rather than zero origin
    for foot_pts, ankle_idx in (((29, 31), MP_LEFT_ANKLE), ((30, 32), MP_RIGHT_ANKLE)):
        if valid[ankle_idx]:
            for f_idx in foot_pts:
                if np.linalg.norm(arr[f_idx, :3]) < 1e-5:
                    projected_xy[f_idx] = projected_xy[ankle_idx]
                    valid[f_idx] = True

    projected = np.zeros((33, 3), dtype=np.float64)
    projected[:, :2] = np.clip(projected_xy, 0.01, 0.99)
    projected[:, 2] = np.where(valid, 1.0, 0.0)
    return projected


def assess_quality_pose_alignment(
    landmarks_3d: np.ndarray,
    anchor_norm: np.ndarray,
    coordinate_system: str = "smplx",
    *,
    median_error_threshold: float = 0.14,
    p90_error_threshold: float = 0.30,
    min_extent_ratio: float = 0.42,
    max_extent_ratio: float = 2.30,
) -> tuple:
    """Check whether a quality pose is safe to show/use for retargeting.

    Temporal quality models can return finite values while the decoded body
    collapses (or sends one limb far outside the observed silhouette).  A
    simple finite-value check cannot catch this failure.  Compare the
    quality-model projection against the independent MediaPipe image
    observation and return ``(needs_fallback, reason, projected)``.  The
    caller can retain WHAM depth diagnostics while using MediaPipe for the
    affected frame's retargeting and overlay.
    """
    projected = project_quality_3d_to_image(
        landmarks_3d,
        anchor_norm,
        coordinate_system=coordinate_system,
    )
    if projected is None or anchor_norm is None:
        return False, "", projected
    quality = np.asarray(projected, dtype=np.float64)
    observed = np.asarray(anchor_norm, dtype=np.float64)
    if quality.ndim != 2 or observed.ndim != 2 or quality.shape[0] < 33 or observed.shape[0] < 33:
        return False, "", projected

    # Face slots are intentionally sparse for COCO/WHAM-17 adapters.  Body
    # slots carry the silhouette evidence needed to identify a collapsed limb.
    body_indices = np.asarray(
        [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32],
        dtype=np.int64,
    )
    quality_valid = (
        np.isfinite(quality[body_indices, :2]).all(axis=1)
        & (quality[body_indices, 2] > 0.15)
    )
    observed_valid = np.isfinite(observed[body_indices, :2]).all(axis=1)
    if observed.shape[1] > 2:
        observed_valid &= observed[body_indices, 2] > 0.15
    valid = quality_valid & observed_valid
    if int(np.count_nonzero(valid)) < 4:
        return False, "", projected

    errors = np.linalg.norm(
        quality[body_indices[valid], :2] - observed[body_indices[valid], :2],
        axis=1,
    )
    median_error = float(np.median(errors)) if len(errors) else 0.0
    p90_error = float(np.percentile(errors, 90.0)) if len(errors) else 0.0

    q_points = quality[body_indices[valid], :2]
    o_points = observed[body_indices[valid], :2]
    quality_extent = float(np.max(np.ptp(q_points, axis=0)))
    observed_extent = float(np.max(np.ptp(o_points, axis=0)))
    extent_ratio = quality_extent / max(observed_extent, 1e-6)

    reasons = []
    if median_error > float(median_error_threshold):
        reasons.append(f"median_2d_error={median_error:.3f}")
    if p90_error > float(p90_error_threshold):
        reasons.append(f"p90_2d_error={p90_error:.3f}")
    if extent_ratio < float(min_extent_ratio):
        reasons.append(f"body_extent_collapsed={extent_ratio:.2f}")
    elif extent_ratio > float(max_extent_ratio):
        reasons.append(f"body_extent_exploded={extent_ratio:.2f}")
    if not reasons:
        return False, "", projected
    return True, "WHAM geometry disagrees with MediaPipe (" + ", ".join(reasons) + ")", projected


def canonical_smplx_to_legacy_landmarks(landmarks_3d: np.ndarray) -> np.ndarray:
    """Encode canonical SMPL-X (+Y up, +Z forward) for the legacy stream."""
    arr = np.asarray(landmarks_3d, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] < 3:
        raise ValueError("canonical landmarks must have shape (N, >=3)")
    out = np.array(arr[:, :3], dtype=np.float64, copy=True)
    out[:, 1] *= -1.0
    out[:, 2] *= -1.0
    return out


FUSION_MODE_OFF = "off"
FUSION_MODE_WHAM_MEDIAPIPE = "wham_mediapipe"


def normalize_fusion_mode(value: str = None) -> str:
    """Normalize the persisted/CLI fusion mode without changing legacy backends.

    ``wham_mediapipe`` is intentionally a mode separate from ``backend``.  This
    lets older ``mediapipe``/``rtmpose``/``auto`` jobs continue to use their
    existing path while a WHAM job can explicitly opt into the fused contract.
    """
    mode = str(value or FUSION_MODE_OFF).strip().lower().replace("-", "_")
    aliases = {
        "": FUSION_MODE_OFF,
        "none": FUSION_MODE_OFF,
        "disabled": FUSION_MODE_OFF,
        "fusion": FUSION_MODE_WHAM_MEDIAPIPE,
        "wham_mediapipe_fusion": FUSION_MODE_WHAM_MEDIAPIPE,
        "whammediapipe": FUSION_MODE_WHAM_MEDIAPIPE,
    }
    mode = aliases.get(mode, mode)
    if mode not in (FUSION_MODE_OFF, FUSION_MODE_WHAM_MEDIAPIPE):
        raise ValueError(
            f"Unsupported fusion mode {value!r}; expected '{FUSION_MODE_OFF}' "
            f"or '{FUSION_MODE_WHAM_MEDIAPIPE}'."
        )
    return mode


def _provenance_text(value):
    """Return a non-empty, lower-case provenance token or ``None``."""
    if value is None:
        return None
    text = str(value).strip().lower()
    return text or None


def _provenance_value(metadata: dict, *keys):
    """Read the first non-empty value from a metadata mapping."""
    if not isinstance(metadata, dict):
        return None
    for key in keys:
        value = metadata.get(key)
        if value is not None and str(value).strip():
            return value
    return None


def _optional_stage_diagnostics(metadata: dict):
    """Flatten backend optional-stage evidence into Unity-friendly records.

    Unity's ``JsonUtility`` cannot deserialize arbitrary dictionaries.  Keep
    the source backend metadata untouched, but provide a small list of stable
    records for the result card.  Every record deliberately has the same
    ``stage``, ``status``, and ``reason`` fields so newer backend-specific
    details remain optional rather than breaking old readers.
    """
    if not isinstance(metadata, dict):
        return []

    records = []
    seen = set()

    def append(stage, status, reason):
        stage_text = str(stage or "").strip()
        if not stage_text:
            return
        status_text = _provenance_text(status) or "unknown"
        reason_text = str(reason or "").strip()
        key = (stage_text, status_text, reason_text)
        if key in seen:
            return
        seen.add(key)
        records.append({
            "stage": stage_text,
            "status": status_text,
            "reason": reason_text,
        })

    # Prefer an explicitly flattened contract when a custom adapter already
    # supplies it.  Accept a mapping as well for integrations that naturally
    # represent diagnostics by stage name.
    explicit = _provenance_value(metadata, "optionalStageDiagnostics", "optional_stage_diagnostics")
    if isinstance(explicit, dict):
        explicit = [dict(value, stage=key) if isinstance(value, dict) else {
            "stage": key,
            "reason": value,
        } for key, value in explicit.items()]
    if isinstance(explicit, (list, tuple)):
        for item in explicit:
            if isinstance(item, dict):
                stage = item.get("stage", item.get("name", item.get("id")))
                status = item.get("status", item.get("state"))
                reason = item.get("reason", item.get("error", item.get("detail", "")))
                append(stage, status, reason)
            elif item is not None:
                append("optional_stage", "unknown", item)

    # ``whamStageDiagnostics`` is the resolver's structured dictionary.  Its
    # ``available`` flag is file/capability evidence, not proof of execution,
    # but it is still useful when paired with the runtime missing/error lists.
    stage_mapping = _provenance_value(
        metadata,
        "whamStageDiagnostics",
        "stageDiagnostics",
        "assetDiagnostics",
        "optional_stage_diagnostics_by_name",
    )
    if isinstance(stage_mapping, dict):
        for stage, detail in stage_mapping.items():
            if isinstance(detail, dict):
                available = detail.get("available")
                status = detail.get("status", detail.get("state"))
                if status is None and available is not None:
                    status = "available" if bool(available) else "missing"
                reason = _provenance_value(detail, "reason", "error", "diagnostic", "expected") or ""
            else:
                status = "unknown"
                reason = detail
            append(stage, status, reason)

    missing = _provenance_value(metadata, "missingOptionalAssets", "missing_optional_assets")
    if isinstance(missing, (list, tuple, set)):
        for stage in missing:
            append(stage, "missing", "Optional stage is unavailable at runtime.")

    errors = _provenance_value(metadata, "optionalErrors", "optional_errors")
    if isinstance(errors, (list, tuple, set)):
        for error in errors:
            append("optional_error", "error", error)

    return records


def build_runtime_provenance(
    requested_backend: str,
    actual_backend: str,
    backend_metadata: dict = None,
    backend_fallback: bool = False,
    fallback_reason: str = None,
) -> dict:
    """Build the stable, result-card-facing runtime provenance contract.

    The values are intentionally conservative.  A checkpoint path or static
    asset entry only proves that a file exists; it never upgrades a runner or
    detector to ``active``.  Custom/official adapters can provide explicit
    status fields, while the bundled path derives status only from evidence
    produced by the initialized runtime.
    """
    metadata = dict(backend_metadata) if isinstance(backend_metadata, dict) else {}
    requested = _provenance_text(requested_backend) or "auto"
    actual = _provenance_text(actual_backend) or "unknown"
    fallback = str(fallback_reason).strip() if fallback_reason is not None else None
    if not fallback:
        fallback = _provenance_value(metadata, "fallbackReason", "backendFallbackReason")
        fallback = str(fallback).strip() if fallback is not None and str(fallback).strip() else None

    explicit_official = _provenance_value(metadata, "officialRunnerStatus", "official_runner_status")
    if explicit_official is not None:
        official_status = str(explicit_official).strip().lower()
    elif bool(backend_fallback):
        official_status = "fallback"
    elif actual in ("wham", "hmr2", "hybrik", "pytorch"):
        # ``nativeWhamCore`` and a live initialized backend are execution
        # evidence.  Static asset resolution alone does not reach this branch.
        runtime_active = bool(
            metadata.get("officialRunnerActive") or
            metadata.get("nativeWhamCore") or
            metadata.get("initialized") or
            metadata.get("backendReady")
        )
        official_status = "active" if runtime_active else "not_verified"
    elif requested in ("wham", "hmr2", "hybrik", "pytorch"):
        official_status = "fallback"
    else:
        official_status = "not_requested"

    explicit_hmr2 = _provenance_value(
        metadata,
        "hmr2ImageFeaturesStatus",
        "hmr2_image_features_status",
    )
    if explicit_hmr2 is not None:
        hmr2_status = str(explicit_hmr2).strip().lower()
    elif bool(backend_fallback) and requested in ("wham", "hmr2", "hybrik", "pytorch"):
        hmr2_status = "fallback"
    else:
        runner_status = _provenance_text(
            _provenance_value(metadata, "imageFeatureRunnerStatus", "image_feature_runner_status")
        )
        runner_name = " ".join(
            str(_provenance_value(metadata, key) or "").strip().lower()
            for key in (
                "imageFeatureRunner",
                "imageFeatureRunnerRuntime",
                "imageFeatureRunnerModule",
                "imageFeatureRuntime",
            )
        )
        if runner_status == "active" and ("hmr2" in runner_name or actual in ("wham", "hmr2")):
            hmr2_status = "active"
        elif runner_status in ("fallback", "error", "failed", "rejected") or _provenance_value(
            metadata, "imageFeatureRunnerError", "imageFeatureFallbackReason"
        ):
            hmr2_status = "fallback"
        elif metadata.get("imageFeatureVerified") is True and metadata.get("imageFeaturesUsed", True):
            hmr2_status = "active"
        elif any(
            token in str(item).lower()
            for item in (metadata.get("missingOptionalAssets") or [])
            for token in ("image_feature", "hmr2")
        ):
            hmr2_status = "missing"
        else:
            hmr2_status = "not_configured"

    explicit_vitpose = _provenance_value(
        metadata,
        "vitpose2DStatus",
        "vitpose_2d_status",
    )
    if explicit_vitpose is not None:
        vitpose_status = str(explicit_vitpose).strip().lower()
    elif bool(backend_fallback) and requested in ("wham", "hmr2", "hybrik", "pytorch"):
        vitpose_status = "fallback"
    else:
        detector = _provenance_text(
            _provenance_value(metadata, "vitpose2DDetector", "inputDetector", "detectorName")
        ) or ""
        if metadata.get("vitpose2DActive") is True or "vitpose" in detector:
            vitpose_status = "active"
        elif metadata.get("vitpose2DError") or metadata.get("vitpose2DFallbackReason"):
            vitpose_status = "fallback"
        elif _provenance_value(metadata, "imageFeatureBackbonePath", "vitposeCheckpointPath"):
            # The checkpoint is present, but no detector execution evidence was
            # supplied.  Keep the distinction visible to the result card.
            vitpose_status = "not_verified"
        else:
            vitpose_status = "not_configured"

    return {
        "backendRequested": requested,
        "backendActual": actual,
        "officialRunnerStatus": official_status,
        "hmr2ImageFeaturesStatus": hmr2_status,
        "vitpose2DStatus": vitpose_status,
        "fallbackReason": fallback,
        "optionalStageDiagnostics": _optional_stage_diagnostics(metadata),
    }


def mediapipe_world_to_canonical_landmarks(landmarks_3d: np.ndarray) -> np.ndarray:
    """Convert the existing MediaPipe world contract into canonical SMPL-X.

    MediaPipe's ``pose_world_landmarks`` uses a camera/image-oriented frame:
    +Y points down (the head is negative relative to the hip) and +Z points
    toward/away from the camera according to the legacy overlay convention.
    Canonical SMPL-X uses +Y up and +Z forward, so both axes are negated at
    this one boundary.  Keeping the seam separate from the WHAM adapter
    prevents the same Y/Z conversion from being applied twice when both
    streams are fused.
    """
    arr = np.asarray(landmarks_3d, dtype=np.float64)
    if arr.ndim == 0 or arr.shape[-1] < 3:
        raise ValueError("MediaPipe landmarks must have shape (..., >=3)")
    result = np.array(arr, dtype=np.float64, copy=True)
    result[..., 1] *= -1.0
    result[..., 2] *= -1.0
    return result


def _quality_observation_arrays(observations, frame_count: int = None):
    """Extract canonical 33-slot WHAM arrays and scores from observations."""
    if observations is None:
        observations = []
    values = list(observations)
    count = int(frame_count if frame_count is not None else len(values))
    joints = np.full((count, 33, 3), np.nan, dtype=np.float64)
    scores = np.zeros((count, 33), dtype=np.float64)
    for frame_index, observation in enumerate(values[:count]):
        if observation is None:
            continue
        points = getattr(observation, "landmarks_3d", None)
        if points is None and isinstance(observation, dict):
            points = observation.get(
                "landmarks3d",
                observation.get(
                    "landmarks_3d",
                    observation.get("keypoints3d", observation.get("keypoints_3d", observation.get("joints3d"))),
                ),
            )
        if points is None and isinstance(observation, (list, tuple, np.ndarray)):
            points = observation
        if points is None:
            continue
        for joint_index, point in enumerate(list(points)[:33]):
            try:
                xyz = np.asarray([
                    float(getattr(point, "x", point[0] if isinstance(point, (list, tuple, np.ndarray)) else 0.0)),
                    float(getattr(point, "y", point[1] if isinstance(point, (list, tuple, np.ndarray)) else 0.0)),
                    float(getattr(point, "z", point[2] if isinstance(point, (list, tuple, np.ndarray)) else 0.0)),
                ], dtype=np.float64)
            except (TypeError, ValueError, IndexError):
                continue
            status = getattr(point, "status", None)
            if isinstance(point, dict):
                xyz = np.asarray([
                    point.get("x", 0.0), point.get("y", 0.0), point.get("z", 0.0)
                ], dtype=np.float64)
                status = point.get("status", status)
            if not np.isfinite(xyz).all():
                continue
            # Plain array/dict adapter rows do not carry an explicit status;
            # absence of that field means observed, not missing.  Treating
            # ``None`` as the string ``"none"`` silently discarded every
            # valid full-WHAM row before fusion.
            if status is not None and str(status).strip().lower() in ("missing", "0", "none"):
                continue
            try:
                numeric_status = int(status) if status is not None else 1
            except (TypeError, ValueError):
                numeric_status = 1
            if numeric_status == 0:
                continue
            joints[frame_index, joint_index] = xyz
            score = getattr(point, "score", 1.0)
            if isinstance(point, dict):
                score = point.get("score", point.get("confidence", 1.0))
            try:
                scores[frame_index, joint_index] = float(np.clip(float(score), 0.0, 1.0))
            except (TypeError, ValueError):
                scores[frame_index, joint_index] = 1.0
    return joints, scores


def _auxiliary_observation_arrays(observations, frame_count: int):
    """Extract canonical MediaPipe 3D + normalized 2D arrays and gate fields."""
    values = list(observations or [])
    points_3d = np.full((frame_count, 33, 3), np.nan, dtype=np.float64)
    points_2d = np.full((frame_count, 33, 2), np.nan, dtype=np.float64)
    score = np.zeros((frame_count, 33), dtype=np.float64)
    visibility = np.zeros((frame_count, 33), dtype=np.float64)
    presence = np.zeros((frame_count, 33), dtype=np.float64)
    statuses = np.full((frame_count, 33), "missing", dtype=object)
    for frame_index, observation in enumerate(values[:frame_count]):
        if observation is None:
            continue
        world = norm = None
        if isinstance(observation, (tuple, list)) and not hasattr(observation, "landmarks_3d"):
            # A raw ``(J, 3)`` array/list is a 3D row, while the legacy
            # detector tuple is ``(world, normalized, confidence)``.
            if isinstance(observation, np.ndarray) and observation.ndim == 2 and observation.shape[-1] >= 3:
                world = observation
                norm = None
                confidence = 1.0
            elif observation and isinstance(observation[0], (list, tuple, np.ndarray, dict)) and np.asarray(observation[0]).ndim == 1 and np.asarray(observation[0]).shape[-1] >= 3:
                world = observation
                norm = None
                confidence = 1.0
            else:
                world = observation[0] if len(observation) > 0 else None
                norm = observation[1] if len(observation) > 1 else None
                confidence = observation[2] if len(observation) > 2 else 0.0
        elif isinstance(observation, dict):
            world = observation.get(
                "landmarks3d",
                observation.get(
                    "landmarks_3d",
                    observation.get("keypoints3d", observation.get("keypoints_3d")),
                ),
            )
            norm = observation.get(
                "keypoints2d",
                observation.get("keypoints_2d", observation.get("landmarks2d", observation.get("landmarks_2d"))),
            )
            confidence = observation.get("confidence", observation.get("score", 0.0))
        else:
            world = getattr(observation, "landmarks_3d", None)
            norm = getattr(observation, "keypoints_2d", None)
            confidence = getattr(observation, "raw_confidence", 0.0)
        try:
            frame_confidence = float(np.clip(float(confidence), 0.0, 1.0))
        except (TypeError, ValueError):
            frame_confidence = 0.0
        world_points = world if world is not None else []
        norm_points = norm if norm is not None else []
        for joint_index, point in enumerate(list(world_points)[:33]):
            try:
                if isinstance(point, dict):
                    xyz = np.asarray([point.get("x", 0.0), point.get("y", 0.0), point.get("z", 0.0)], dtype=np.float64)
                    point_score = point.get("score", frame_confidence)
                    point_visibility = point.get("visibility", point_score)
                    point_presence = point.get("presence", point_score)
                    point_status = point.get("status", "observed")
                elif isinstance(point, (list, tuple, np.ndarray)):
                    xyz = np.asarray(point[:3], dtype=np.float64)
                    point_score = frame_confidence
                    point_visibility = point_score
                    point_presence = point_score
                    point_status = "observed"
                else:
                    xyz = np.asarray([float(point.x), float(point.y), float(point.z)], dtype=np.float64)
                    point_score = getattr(point, "score", frame_confidence)
                    point_visibility = getattr(point, "visibility", point_score)
                    point_presence = getattr(point, "presence", point_score)
                    point_status = getattr(point, "status", "observed")
                if np.isfinite(xyz).all():
                    points_3d[frame_index, joint_index] = xyz
                    score[frame_index, joint_index] = float(np.clip(float(point_score), 0.0, 1.0))
                    visibility[frame_index, joint_index] = float(np.clip(float(point_visibility), 0.0, 1.0))
                    presence[frame_index, joint_index] = float(np.clip(float(point_presence), 0.0, 1.0))
                    statuses[frame_index, joint_index] = point_status
            except (TypeError, ValueError, AttributeError, IndexError):
                continue
        for joint_index, point in enumerate(list(norm_points)[:33]):
            try:
                if isinstance(point, dict):
                    xy = np.asarray([point.get("x", 0.0), point.get("y", 0.0)], dtype=np.float64)
                    point_score = point.get("score", frame_confidence)
                elif isinstance(point, (list, tuple, np.ndarray)):
                    xy = np.asarray(point[:2], dtype=np.float64)
                    point_score = point[2] if len(point) > 2 else frame_confidence
                else:
                    xy = np.asarray([float(point.x), float(point.y)], dtype=np.float64)
                    point_score = getattr(point, "score", frame_confidence)
                if np.isfinite(xy).all():
                    points_2d[frame_index, joint_index] = xy
                    if score[frame_index, joint_index] <= 0.0:
                        value = float(np.clip(float(point_score), 0.0, 1.0))
                        score[frame_index, joint_index] = value
                        visibility[frame_index, joint_index] = value
                        presence[frame_index, joint_index] = value
                    if statuses[frame_index, joint_index] == "missing":
                        statuses[frame_index, joint_index] = "observed"
            except (TypeError, ValueError, AttributeError, IndexError):
                continue
    return points_3d, points_2d, score, visibility, presence, statuses


def fuse_wham_mediapipe_sequence(
    wham_observations,
    mediapipe_observations=None,
    timestamps=None,
    frame_indices=None,
    config=None,
    backend_metadata=None,
    rtmpose_observations=None,
    camera=None,
    trajectory_metadata=None,
    temporal_constraints=None,
):
    """Run the fusion API on aligned observation sequences.

    The adapter already returns WHAM in canonical SMPL-X coordinates.  This
    integration seam therefore passes that array to ``fuse_wham_mediapipe`` as
    canonical data and converts MediaPipe world Y/Z exactly once before the call.
    The returned joints are canonical as well; callers must not apply a
    WHAM-specific Unity rotation or flip.
    """
    def envelope_metadata(value):
        if not isinstance(value, dict):
            return {}
        raw = value.get("metadata", value.get("backendMetadata", {}))
        result = dict(raw) if isinstance(raw, dict) else {}
        # Official/full adapters commonly put camera and trajectory records
        # beside ``frames`` rather than nesting them under ``metadata``.
        # Preserve those fields without copying a potentially large frame
        # tensor into exported backend metadata.
        for key in (
            "camera", "cameraModel", "cameraMetadata",
            "trajectory", "trajectoryMetadata", "cameraTrajectory",
            "temporalConstraints", "temporal_constraints", "frameErrors",
        ):
            if key in value:
                result[key] = value[key]
        return result

    # Keep sequence normalisation/alignment in the dependency-light fusion
    # helper.  Full WHAM adapters may return an envelope with ``frames`` or
    # records out of order; list position is only a last-resort legacy path.
    primary_envelope_metadata = envelope_metadata(wham_observations)
    auxiliary_envelope_metadata = envelope_metadata(mediapipe_observations)
    primary_values = _sequence_values(wham_observations)
    auxiliary_values = _sequence_values(mediapipe_observations)
    rtmpose_values = _sequence_values(rtmpose_observations)
    def sequence_length(value):
        return len(value) if value is not None else 0

    requested_count = max(
        len(primary_values),
        len(auxiliary_values),
        len(rtmpose_values),
        sequence_length(timestamps),
        sequence_length(frame_indices),
    )
    if requested_count <= 0:
        return None
    aligned_wham, aligned_mp, alignment = align_observation_sequences(
        primary_values,
        auxiliary_values,
        frame_indices=frame_indices,
        timestamps=timestamps,
        frame_count=requested_count,
    )
    aligned_rtmpose, _unused_rtmpose, rtmpose_alignment = align_observation_sequences(
        rtmpose_values,
        None,
        frame_indices=alignment["frameIndices"],
        timestamps=alignment["timestampsSec"],
        frame_count=requested_count,
    )
    frame_count = requested_count
    wham_3d, wham_scores = _quality_observation_arrays(aligned_wham, frame_count)
    aux_3d, aux_2d, score, visibility, presence, statuses = _auxiliary_observation_arrays(
        aligned_mp, frame_count
    )
    # RTMPose contributes image-space anchors independently of MediaPipe's
    # metric world stream.  Prefer it where it has a stronger score and use
    # it for slots MediaPipe did not observe; neither detector can overwrite a
    # valid WHAM 3D joint at this seam.
    rtm_3d, rtm_2d, rtm_score, rtm_visibility, rtm_presence, rtm_statuses = _auxiliary_observation_arrays(
        aligned_rtmpose, frame_count
    )
    rtm_finite = np.isfinite(rtm_2d).all(axis=-1)
    mp_finite = np.isfinite(aux_2d).all(axis=-1)
    use_rtm = rtm_finite & (~mp_finite | (rtm_score > score))
    combined_2d = np.array(aux_2d, dtype=np.float64, copy=True)
    combined_score = np.array(score, dtype=np.float64, copy=True)
    combined_visibility = np.array(visibility, dtype=np.float64, copy=True)
    combined_presence = np.array(presence, dtype=np.float64, copy=True)
    combined_statuses = np.array(statuses, dtype=object, copy=True)
    combined_2d[use_rtm] = rtm_2d[use_rtm]
    combined_score[use_rtm] = rtm_score[use_rtm]
    combined_visibility[use_rtm] = rtm_visibility[use_rtm]
    combined_presence[use_rtm] = rtm_presence[use_rtm]
    combined_statuses[use_rtm] = rtm_statuses[use_rtm]
    aux_3d = mediapipe_world_to_canonical_landmarks(aux_3d)
    if not np.isfinite(aux_3d).any():
        aux_3d_input = None
    else:
        aux_3d_input = aux_3d
    if not np.isfinite(combined_2d).any():
        aux_2d_input = None
    else:
        aux_2d_input = combined_2d
    timestamps = alignment["timestampsSec"]
    frame_indices = alignment["frameIndices"]
    fusion_config = config if isinstance(config, FusionConfig) else FusionConfig.from_dict(config)
    metadata = dict(backend_metadata or {})
    for envelope in (primary_envelope_metadata, auxiliary_envelope_metadata):
        if isinstance(envelope, dict):
            metadata.update(envelope)
    # Adapter metadata may arrive either at the sequence envelope or on an
    # individual frame.  Preserve camera/trajectory/constraint records for
    # export while passing a concrete camera mapping to the numerical fusion
    # core when one is available.
    frame_metadata = [
        envelope for envelope in (primary_envelope_metadata, auxiliary_envelope_metadata)
        if isinstance(envelope, dict)
    ]
    frame_errors = []
    for frame_number, (quality_row, media_row, rtm_row) in enumerate(zip(aligned_wham, aligned_mp, aligned_rtmpose)):
        for label, row in (("wham", quality_row), ("mediapipe", media_row), ("rtmpose", rtm_row)):
            if row is None:
                continue
            row_meta = {}
            if isinstance(row, dict):
                raw_meta = row.get("metadata", row.get("backendMetadata", {}))
                if isinstance(raw_meta, dict):
                    row_meta = raw_meta
                error = row.get("error", row.get("errorMessage", row.get("failure")))
            else:
                row_meta = getattr(row, "backend_metadata", {}) or {}
                error = getattr(row, "error", None)
            if isinstance(row_meta, dict):
                frame_metadata.append(row_meta)
            if error:
                frame_errors.append({
                    "frame": int(frame_indices[frame_number]),
                    "stage": label,
                    "message": str(error),
                })
    if camera is None:
        for row_meta in frame_metadata:
            candidate = row_meta.get("camera", row_meta.get("cameraModel", row_meta.get("cameraMetadata")))
            if candidate is not None:
                camera = candidate
                break
    if trajectory_metadata is None:
        for row_meta in frame_metadata:
            candidate = row_meta.get("trajectory", row_meta.get("trajectoryMetadata", row_meta.get("cameraTrajectory")))
            if candidate is not None:
                trajectory_metadata = candidate
                break
    if temporal_constraints is None:
        for row_meta in frame_metadata:
            candidate = row_meta.get("temporalConstraints", row_meta.get("temporal_constraints"))
            if isinstance(candidate, dict):
                candidate = candidate.get("weights", candidate.get("mask", candidate.get("values")))
            if candidate is not None:
                temporal_constraints = candidate
                break
    metadata.update({
        "primaryBackend": "wham",
        "auxiliaryBackends": ["mediapipe_3d", "mediapipe_2d"] + (["rtmpose_2d"] if np.any(use_rtm) else []),
        "observationAlignment": {
            **alignment,
            "timestampSource": "sample_times",
            "rtmposeSourceFrameIndices": rtmpose_alignment.get("qualitySourceFrameIndices", []),
        },
        "frameErrors": frame_errors,
    })
    if trajectory_metadata is not None:
        metadata["trajectory"] = trajectory_metadata
    if temporal_constraints is not None:
        metadata["temporalConstraints"] = temporal_constraints
    if isinstance(trajectory_metadata, dict):
        if "worldMotionAvailable" in trajectory_metadata:
            fusion_config.world_motion_available = bool(trajectory_metadata["worldMotionAvailable"])
        elif "world_motion_available" in trajectory_metadata:
            fusion_config.world_motion_available = bool(trajectory_metadata["world_motion_available"])
        if "scaleIsRelative" in trajectory_metadata:
            fusion_config.scale_is_relative = bool(trajectory_metadata["scaleIsRelative"])
        elif "scale_is_relative" in trajectory_metadata:
            fusion_config.scale_is_relative = bool(trajectory_metadata["scale_is_relative"])
    result = fuse_wham_mediapipe(
        wham_3d,
        mediapipe_3d=aux_3d_input,
        mediapipe_2d=aux_2d_input,
        score=combined_score if aux_3d_input is not None or aux_2d_input is not None else None,
        visibility=combined_visibility if aux_3d_input is not None or aux_2d_input is not None else None,
        presence=combined_presence if aux_3d_input is not None or aux_2d_input is not None else None,
        statuses=combined_statuses if aux_3d_input is not None or aux_2d_input is not None else None,
        timestamps=timestamps,
        frame_indices=frame_indices,
        config=fusion_config,
        wham_score=wham_scores,
        wham_coordinate_system="smplx",
        mediapipe_coordinate_system="smplx",
        backend_metadata=metadata,
        camera=camera,
        trajectory_metadata=trajectory_metadata,
        temporal_constraints=temporal_constraints,
        selection_mode="fallback_only",
    )
    result.metadata["observationAlignment"] = metadata["observationAlignment"]
    result.metadata["frameErrors"] = frame_errors
    result.metadata["rtmpose2dUsed"] = bool(np.any(use_rtm))
    return result


# WHAM tracks COCO-17 body joints. MediaPipe indices outside this set
# (face details 1, 3, 4, 6, 9, 10 and hand fingers 17-22) are naturally
# completed by MediaPipe and must not be treated as a WHAM failure/fallback.
WHAM_CORE_BODY_JOINTS = {11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28}


def fusion_frame_overlay_fallback(fusion_result, frame_index: int) -> tuple:
    """Return whether a fused frame contains an explicit MediaPipe fallback.

    ``FusionResult`` keeps joint-level provenance because a valid WHAM joint
    must remain primary even when a different joint falls back.  Overlay
    metadata is frame-level, so summarize only joints explicitly marked as a
    fallback here; normal WHAM+MediaPipe reprojection remains a fused overlay.
    """
    if fusion_result is None:
        return False, ""
    provenance = getattr(fusion_result, "provenance", None)
    if provenance is None and isinstance(fusion_result, dict):
        provenance = fusion_result.get("jointProvenance", fusion_result.get("provenance"))
    if not isinstance(provenance, (list, tuple)):
        return False, ""
    try:
        row = provenance[int(frame_index)]
    except (IndexError, TypeError, ValueError):
        return False, ""
    if not isinstance(row, (list, tuple)):
        return False, ""
    has_wham_joints = any(
        isinstance(item, dict) and item.get("whamValid") is True
        for item in row
    )
    reasons = []
    for item in row:
        if not isinstance(item, dict):
            continue
        joint = item.get("joint")
        # If WHAM successfully predicted core body joints for this frame,
        # natural auxiliary topology completions (such as fingers or facial slots)
        # must not falsely trigger a full-frame WHAM geometry fallback.
        if has_wham_joints and joint is not None:
            try:
                if int(joint) not in WHAM_CORE_BODY_JOINTS:
                    continue
            except (TypeError, ValueError):
                pass
        source = str(item.get("source", "")).strip().lower()
        used_fallback = bool(item.get("usedFallback", False))
        # ``usedFallback`` is the canonical flag.  The source/whamValid
        # fallback catches provenance records produced by older fusion cores.
        if not used_fallback and not (
            source == "mediapipe" and item.get("whamValid") is False
        ):
            continue
        reason = str(item.get("reason", "mediapipe_fallback")).strip()
        if reason and reason not in reasons:
            reasons.append(reason)
    return bool(reasons), "; ".join(reasons)


# ============================================================================
# Progress Reporting (Streaming JSON Lines on stdout)
# ============================================================================

def emit_progress(progress: float, stage: str, frame: int, total_frames: int, extra: dict = None):
    """
    Emits a single JSON Line to standard output for Unity Editor progress tracking.
    All diagnostic text or warnings are kept strictly on sys.stderr.
    """
    msg = {
        "progress": round(float(np.clip(progress, 0.0, 1.0)), 4),
        "stage": str(stage),
        "frame": int(frame),
        "total_frames": int(total_frames),
    }
    if extra:
        msg.update(extra)
    print(json.dumps(msg), flush=True)


def transcode_to_h264_if_needed(video_path: str) -> bool:
    """
    Transcodes an MP4 video to Unity/Windows Media Foundation native H.264 (yuv420p)
    using ffmpeg if available. Fixes 'Preparing Pose Overlay Video...' infinite hangs.
    Includes Windows file-lock retry loop for rock-solid reliability.
    """
    if not video_path or not os.path.exists(video_path):
        return False

    ffmpeg_bin = shutil.which("ffmpeg")
    if not ffmpeg_bin:
        if os.path.exists(r"C:\user\bin\ffmpeg.exe"):
            ffmpeg_bin = r"C:\user\bin\ffmpeg.exe"
        else:
            return False

    temp_h264 = f"{os.path.splitext(video_path)[0]}_h264_temp.mp4"
    try:
        cmd = [
            ffmpeg_bin, "-y",
            "-i", video_path,
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-bsf:v", "h264_metadata=colour_primaries=1:transfer_characteristics=1:matrix_coefficients=1",
            "-preset", "ultrafast",
            "-tune", "fastdecode",
            "-g", "15",
            "-keyint_min", "15",
            "-movflags", "+faststart",
            temp_h264
        ]
        result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=180)
        if result.returncode == 0 and os.path.exists(temp_h264) and os.path.getsize(temp_h264) > 0:
            # Atomic replace with Windows file sharing retry loop
            replaced = False
            for attempt in range(5):
                try:
                    os.replace(temp_h264, video_path)
                    replaced = True
                    break
                except PermissionError:
                    time.sleep(0.2)
            if replaced:
                sys.stderr.write(f"[TexMotion] Successfully transcoded overlay video to Unity H.264: {video_path}\n")
                sys.stderr.flush()
                return True
    except Exception as e:
        sys.stderr.write(f"[TexMotion] Warning: H.264 transcode attempt failed: {e}\n")
        sys.stderr.flush()
    finally:
        if os.path.exists(temp_h264):
            try:
                os.remove(temp_h264)
            except Exception:
                pass
    return False


# ============================================================================
# 3D Math & Quaternion Operations
# ============================================================================

def normalize_vector(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    norm = np.linalg.norm(v)
    if norm < eps:
        return np.zeros_like(v)
    return v / norm


def normalize_quaternion(q: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Safely normalizes a quaternion, falling back to identity [0, 0, 0, 1] if degenerate."""
    norm = np.linalg.norm(q)
    if norm < eps or np.isnan(norm):
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    q_norm = q / norm
    return q_norm if q_norm[3] >= 0.0 else -q_norm


def matrix_to_quaternion(R: np.ndarray) -> np.ndarray:
    """
    Converts a 3x3 rotation matrix to a normalized unit quaternion [x, y, z, w].
    Implements Shepperd's algorithm for numerical stability.
    """
    trace = np.trace(R)
    if trace > 0.0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(max(1e-8, 1.0 + R[0, 0] - R[1, 1] - R[2, 2]))
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(max(1e-8, 1.0 + R[1, 1] - R[0, 0] - R[2, 2]))
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(max(1e-8, 1.0 + R[2, 2] - R[0, 0] - R[1, 1]))
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s

    q = np.array([x, y, z, w], dtype=np.float64)
    norm = np.linalg.norm(q)
    if norm < 1e-8:
        return np.array([0.0, 0.0, 0.0, 1.0])
    return q / norm


def quaternion_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Multiplies two quaternions q = q1 * q2."""
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return np.array([
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    ], dtype=np.float64)


def quaternion_inverse(q: np.ndarray) -> np.ndarray:
    """Computes the conjugate/inverse of a unit quaternion."""
    return np.array([-q[0], -q[1], -q[2], q[3]], dtype=np.float64)


def quaternion_slerp(q1: np.ndarray, q2: np.ndarray, t: float) -> np.ndarray:
    """Spherical linear interpolation between two unit quaternions."""
    cos_theta = np.dot(q1, q2)
    q2_adj = np.copy(q2)
    if cos_theta < 0.0:
        cos_theta = -cos_theta
        q2_adj = -q2_adj

    if cos_theta > 0.9995:
        # Linear interpolation for very close orientations
        result = (1.0 - t) * q1 + t * q2_adj
        return normalize_vector(result)

    theta = np.arccos(np.clip(cos_theta, -1.0, 1.0))
    sin_theta = np.sin(theta)
    w1 = np.sin((1.0 - t) * theta) / sin_theta
    w2 = np.sin(t * theta) / sin_theta
    return w1 * q1 + w2 * q2_adj


def rotation_vector_to_quaternion(v_from: np.ndarray, v_to: np.ndarray) -> np.ndarray:
    """Computes the shortest arc unit quaternion rotating unit vector v_from to v_to."""
    u = normalize_vector(v_from)
    v = normalize_vector(v_to)
    dot = np.dot(u, v)
    if dot > 0.999999:
        return np.array([0.0, 0.0, 0.0, 1.0])
    if dot < -0.999999:
        ortho = np.array([0.0, 1.0, 0.0]) if abs(u[1]) < 0.9 else np.array([0.0, 0.0, 1.0])
        axis = normalize_vector(np.cross(u, ortho))
        return np.array([axis[0], axis[1], axis[2], 0.0])

    axis = np.cross(u, v)
    w = 1.0 + dot
    q = np.array([axis[0], axis[1], axis[2], w], dtype=np.float64)
    return normalize_vector(q)


def quat_rotate_vector(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotates a 3D vector v by unit quaternion q."""
    qv = np.array([v[0], v[1], v[2], 0.0])
    q_inv = quaternion_inverse(q)
    return quaternion_multiply(quaternion_multiply(q, qv), q_inv)[:3]


def stabilize_hinge_quaternion(
    rest_dir: np.ndarray,
    v_target_local: np.ndarray,
    default_hinge_axis: np.ndarray,
    min_bend_angle_rad: float = 0.08
) -> np.ndarray:
    """
    Computes a smooth, stabilized unit quaternion for anatomical hinge joints (elbow/knee).
    Eliminates cross-product axis singularity when the limb is near full extension,
    smoothly blending toward identity without sudden axis flips.
    """
    u = normalize_vector(rest_dir)
    v = normalize_vector(v_target_local)
    dot = float(np.clip(np.dot(u, v), -1.0, 1.0))
    angle = np.arccos(dot)

    if angle < min_bend_angle_rad:
        t = angle / min_bend_angle_rad
        axis = normalize_vector(default_hinge_axis)
        half_ang = (angle * t) * 0.5
        s = np.sin(half_ang)
        c = np.cos(half_ang)
        return np.array([axis[0] * s, axis[1] * s, axis[2] * s, c], dtype=np.float64)

    cross = np.cross(u, v)
    cross_norm = np.linalg.norm(cross)
    if cross_norm < 1e-4:
        axis = normalize_vector(default_hinge_axis)
    else:
        axis = cross / cross_norm

    half_ang = angle * 0.5
    s = np.sin(half_ang)
    c = np.cos(half_ang)
    q = np.array([axis[0] * s, axis[1] * s, axis[2] * s, c], dtype=np.float64)
    return normalize_vector(q)


def make_orthonormal_frame(primary: np.ndarray, secondary: np.ndarray, primary_axis: str = 'y', secondary_axis: str = 'x') -> np.ndarray:
    """
    Constructs a 3x3 orthonormal basis matrix [x | y | z] where primary vector
    aligns with primary_axis, and secondary vector defines the secondary_axis plane.
    """
    p = normalize_vector(primary)
    s = normalize_vector(secondary)

    if abs(np.dot(p, s)) > 0.999:
        alt = np.array([1.0, 0.0, 0.0]) if abs(p[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        s = normalize_vector(np.cross(p, alt))

    if primary_axis == 'y' and secondary_axis == 'x':
        z = normalize_vector(np.cross(s, p))
        x = normalize_vector(np.cross(p, z))
        y = p
        return np.column_stack([x, y, z])
    elif primary_axis == 'x' and secondary_axis == 'z':
        y = normalize_vector(np.cross(s, p))
        z = normalize_vector(np.cross(p, y))
        x = p
        return np.column_stack([x, y, z])
    elif primary_axis == 'x' and secondary_axis == 'y':
        z = normalize_vector(np.cross(p, s))
        y = normalize_vector(np.cross(z, p))
        x = p
        return np.column_stack([x, y, z])
    else:
        t = normalize_vector(np.cross(p, s))
        s_ortho = normalize_vector(np.cross(t, p))
        return np.column_stack([s_ortho, p, t])


def quaternion_to_matrix(q: np.ndarray) -> np.ndarray:
    """Converts a unit quaternion [x, y, z, w] to a 3x3 rotation matrix."""
    x, y, z, w = q
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm > 1e-8:
        x, y, z, w = x / norm, y / norm, z / norm, w / norm
    else:
        return np.eye(3, dtype=np.float64)
    return np.array([
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y)],
        [2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x)],
        [2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y)]
    ], dtype=np.float64)


def quaternion_to_euler(q: np.ndarray) -> np.ndarray:
    """
    Converts unit quaternion [x, y, z, w] to intrinsic XYZ Euler angles [rx, ry, rz] in radians.
    Convention: R = Rx(rx) * Ry(ry) * Rz(rz).
    """
    R = quaternion_to_matrix(q)
    sy = np.clip(R[0, 2], -1.0, 1.0)
    ry = math.asin(sy)
    if abs(sy) < 0.999999:
        rx = math.atan2(-R[1, 2], R[2, 2])
        rz = math.atan2(-R[0, 1], R[0, 0])
    else:
        rx = math.atan2(R[2, 1], R[1, 1])
        rz = 0.0
    return np.array([rx, ry, rz], dtype=np.float64)


def euler_to_quaternion(euler: np.ndarray) -> np.ndarray:
    """
    Converts intrinsic XYZ Euler angles [rx, ry, rz] in radians to unit quaternion [x, y, z, w].
    """
    rx, ry, rz = euler
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    Rx = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]], dtype=np.float64)
    Ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=np.float64)
    Rz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    R = Rx @ Ry @ Rz
    return matrix_to_quaternion(R)


# ============================================================================
# Anatomical Joint Limit (ROM) Constraints
# ============================================================================

def swing_twist_decompose(q: np.ndarray, twist_axis: np.ndarray) -> tuple:
    """
    Decomposes a unit quaternion q into Swing and Twist components:
        q = q_swing * q_twist
    where q_twist represents rotation exclusively around twist_axis,
    and q_swing represents rotation around an axis perpendicular to twist_axis.
    Mathematically immune to gimbal lock and euler angle wrap-around singularities.
    Returns: (q_swing, q_twist)
    """
    axis = normalize_vector(twist_axis)
    q_vec = q[:3]
    dot = float(np.dot(q_vec, axis))
    p = dot * axis

    q_twist = np.array([p[0], p[1], p[2], q[3]], dtype=np.float64)
    q_twist = normalize_quaternion(q_twist)

    q_twist_inv = np.array([-q_twist[0], -q_twist[1], -q_twist[2], q_twist[3]], dtype=np.float64)
    q_swing = quaternion_multiply(q, q_twist_inv)
    q_swing = normalize_quaternion(q_swing)

    return q_swing, q_twist


def _clamp_swing_twist(
    q: np.ndarray,
    twist_axis: np.ndarray,
    cone_max_rad: float,
    twist_min_rad: float,
    twist_max_rad: float
) -> np.ndarray:
    """
    Constrains 3-DOF spherical joint using anatomical Swing-Twist decomposition.
    Eliminates gimbal lock and euler singularities.
    """
    if q[3] < 0.0:
        q = -q

    q_swing, q_twist = swing_twist_decompose(q, twist_axis)

    # 1. Clamp Swing (Spherical Cone Limit around rest direction)
    w_s = float(np.clip(abs(q_swing[3]), 0.0, 1.0))
    swing_angle = 2.0 * math.acos(w_s)
    if swing_angle > cone_max_rad and swing_angle > 1e-4:
        half_max = cone_max_rad * 0.5
        scale = math.sin(half_max) / math.sin(swing_angle * 0.5)
        w_sign = 1.0 if q_swing[3] >= 0.0 else -1.0
        q_swing = np.array([
            q_swing[0] * scale,
            q_swing[1] * scale,
            q_swing[2] * scale,
            math.cos(half_max) * w_sign
        ], dtype=np.float64)
        q_swing = normalize_quaternion(q_swing)

    # 2. Clamp Twist (Rotation angle around twist_axis)
    axis = normalize_vector(twist_axis)
    twist_dot = float(np.dot(q_twist[:3], axis))
    twist_angle = 2.0 * math.atan2(twist_dot, q_twist[3])
    twist_angle = (twist_angle + math.pi) % (2.0 * math.pi) - math.pi
    twist_clamped = float(np.clip(twist_angle, twist_min_rad, twist_max_rad))
    half_t = twist_clamped * 0.5
    st = math.sin(half_t)
    ct = math.cos(half_t)
    q_twist = np.array([axis[0] * st, axis[1] * st, axis[2] * st, ct], dtype=np.float64)

    # 3. Recombine: q_res = q_swing * q_twist
    q_res = quaternion_multiply(q_swing, q_twist)
    return normalize_quaternion(q_res)


def _clamp_hinge_x(q: np.ndarray, angle_min_rad: float, angle_max_rad: float) -> np.ndarray:
    """Clamps a 1-DOF hinge joint strictly around local X axis (Knees)."""
    if q[3] < 0.0:
        q = -q
    angle = 2.0 * math.atan2(q[0], q[3])
    angle_clamped = float(np.clip(angle, angle_min_rad, angle_max_rad))
    half = angle_clamped * 0.5
    return np.array([math.sin(half), 0.0, 0.0, math.cos(half)], dtype=np.float64)


def _clamp_hinge_y_neg(q: np.ndarray, angle_min_rad: float, angle_max_rad: float) -> np.ndarray:
    """Clamps a 1-DOF hinge joint strictly around local -Y axis (Left Elbow)."""
    if q[3] < 0.0:
        q = -q
    angle = 2.0 * math.atan2(-q[1], q[3])
    angle_clamped = float(np.clip(angle, angle_min_rad, angle_max_rad))
    half = angle_clamped * 0.5
    return np.array([0.0, -math.sin(half), 0.0, math.cos(half)], dtype=np.float64)


def _clamp_hinge_y_pos(q: np.ndarray, angle_min_rad: float, angle_max_rad: float) -> np.ndarray:
    """Clamps a 1-DOF hinge joint strictly around local +Y axis (Right Elbow)."""
    if q[3] < 0.0:
        q = -q
    angle = 2.0 * math.atan2(q[1], q[3])
    angle_clamped = float(np.clip(angle, angle_min_rad, angle_max_rad))
    half = angle_clamped * 0.5
    return np.array([0.0, math.sin(half), 0.0, math.cos(half)], dtype=np.float64)


JOINT_LIMIT_CONFIGS = {
    # 0: Pelvis -> Free 360-degree rotation (no artificial restriction)
    0: {"type": "free"},
    # 1: L_Hip, 2: R_Hip -> Thigh extends along -Y in SMPL-X T-Pose
    1: {
        "type": "swing_twist",
        "twist_axis": np.array([0.0, -1.0, 0.0]),
        "cone_max": math.radians(115.0),
        "twist_min": math.radians(-45.0),
        "twist_max": math.radians(45.0),
    },
    2: {
        "type": "swing_twist",
        "twist_axis": np.array([0.0, -1.0, 0.0]),
        "cone_max": math.radians(115.0),
        "twist_min": math.radians(-45.0),
        "twist_max": math.radians(45.0),
    },
    # 3: Spine1, 6: Spine2 -> Intervertebral flexibility along +Y
    3: {
        "type": "swing_twist",
        "twist_axis": np.array([0.0, 1.0, 0.0]),
        "cone_max": math.radians(25.0),
        "twist_min": math.radians(-20.0),
        "twist_max": math.radians(20.0),
    },
    6: {
        "type": "swing_twist",
        "twist_axis": np.array([0.0, 1.0, 0.0]),
        "cone_max": math.radians(25.0),
        "twist_min": math.radians(-20.0),
        "twist_max": math.radians(20.0),
    },
    # 4: L_Knee, 5: R_Knee -> Strict 1-DOF Hinges around X (Hyper-extension < 0 strictly prevented)
    4: {
        "type": "hinge_x",
        "angle_min": 0.0,
        "angle_max": math.radians(150.0),
    },
    5: {
        "type": "hinge_x",
        "angle_min": 0.0,
        "angle_max": math.radians(150.0),
    },
    # 7: L_Ankle, 8: R_Ankle -> Along -Y
    7: {
        "type": "swing_twist",
        "twist_axis": np.array([0.0, -1.0, 0.0]),
        "cone_max": math.radians(50.0),
        "twist_min": math.radians(-30.0),
        "twist_max": math.radians(30.0),
    },
    8: {
        "type": "swing_twist",
        "twist_axis": np.array([0.0, -1.0, 0.0]),
        "cone_max": math.radians(50.0),
        "twist_min": math.radians(-30.0),
        "twist_max": math.radians(30.0),
    },
    # 9: Spine3 -> Upper Thoracic Spine along +Y
    9: {
        "type": "swing_twist",
        "twist_axis": np.array([0.0, 1.0, 0.0]),
        "cone_max": math.radians(30.0),
        "twist_min": math.radians(-25.0),
        "twist_max": math.radians(25.0),
    },
    # 10: L_Foot, 11: R_Foot -> Feet point along +Z
    10: {
        "type": "swing_twist",
        "twist_axis": np.array([0.0, 0.0, 1.0]),
        "cone_max": math.radians(40.0),
        "twist_min": math.radians(-20.0),
        "twist_max": math.radians(20.0),
    },
    11: {
        "type": "swing_twist",
        "twist_axis": np.array([0.0, 0.0, 1.0]),
        "cone_max": math.radians(40.0),
        "twist_min": math.radians(-20.0),
        "twist_max": math.radians(20.0),
    },
    # 12: Neck -> Cervical rotation limit (prevents extreme twist)
    12: {
        "type": "swing_twist",
        "twist_axis": np.array([0.0, 1.0, 0.0]),
        "cone_max": math.radians(45.0),
        "twist_min": math.radians(-45.0),
        "twist_max": math.radians(45.0),
    },
    # 13: L_Collar, 14: R_Collar -> Clavicle
    13: {
        "type": "swing_twist",
        "twist_axis": np.array([1.0, 0.0, 0.0]),
        "cone_max": math.radians(25.0),
        "twist_min": math.radians(-15.0),
        "twist_max": math.radians(15.0),
    },
    14: {
        "type": "swing_twist",
        "twist_axis": np.array([-1.0, 0.0, 0.0]),
        "cone_max": math.radians(25.0),
        "twist_min": math.radians(-15.0),
        "twist_max": math.radians(15.0),
    },
    # 15: Head -> Prevents 180-degree neck twist (owl illusion)
    15: {
        "type": "swing_twist",
        "twist_axis": np.array([0.0, 1.0, 0.0]),
        "cone_max": math.radians(55.0),
        "twist_min": math.radians(-60.0),
        "twist_max": math.radians(60.0),
    },
    # 16: L_Shoulder, 17: R_Shoulder -> Upper arm along +X (L) / -X (R)
    16: {
        "type": "swing_twist",
        "twist_axis": np.array([1.0, 0.0, 0.0]),
        "cone_max": math.radians(130.0),
        "twist_min": math.radians(-90.0),
        "twist_max": math.radians(90.0),
    },
    17: {
        "type": "swing_twist",
        "twist_axis": np.array([-1.0, 0.0, 0.0]),
        "cone_max": math.radians(130.0),
        "twist_min": math.radians(-90.0),
        "twist_max": math.radians(90.0),
    },
    # 18: L_Elbow -> Strict 1-DOF Hinge around -Y (0 to 150 deg)
    18: {
        "type": "hinge_y_neg",
        "angle_min": 0.0,
        "angle_max": math.radians(150.0),
    },
    # 19: R_Elbow -> Strict 1-DOF Hinge around +Y (0 to 150 deg)
    19: {
        "type": "hinge_y_pos",
        "angle_min": 0.0,
        "angle_max": math.radians(150.0),
    },
    # 20: L_Wrist, 21: R_Wrist -> Wrist along +X (L) / -X (R)
    20: {
        "type": "swing_twist",
        "twist_axis": np.array([1.0, 0.0, 0.0]),
        "cone_max": math.radians(75.0),
        "twist_min": math.radians(-85.0),
        "twist_max": math.radians(85.0),
    },
    21: {
        "type": "swing_twist",
        "twist_axis": np.array([-1.0, 0.0, 0.0]),
        "cone_max": math.radians(75.0),
        "twist_min": math.radians(-85.0),
        "twist_max": math.radians(85.0),
    },
}


def _apply_joint_limits_single_frame(local_rot: np.ndarray) -> np.ndarray:
    out = np.copy(local_rot)
    for j in range(min(SMPLX_JOINT_COUNT, local_rot.shape[0])):
        cfg = JOINT_LIMIT_CONFIGS.get(j, None)
        if cfg is None or cfg["type"] == "free":
            continue

        q = out[j]
        j_type = cfg["type"]
        if j_type == "swing_twist":
            out[j] = _clamp_swing_twist(
                q,
                cfg["twist_axis"],
                cfg["cone_max"],
                cfg["twist_min"],
                cfg["twist_max"]
            )
        elif j_type == "hinge_x":
            out[j] = _clamp_hinge_x(q, cfg["angle_min"], cfg["angle_max"])
        elif j_type == "hinge_y_neg":
            out[j] = _clamp_hinge_y_neg(q, cfg["angle_min"], cfg["angle_max"])
        elif j_type == "hinge_y_pos":
            out[j] = _clamp_hinge_y_pos(q, cfg["angle_min"], cfg["angle_max"])
    return out


def apply_anatomical_joint_limits(local_rotations: np.ndarray) -> np.ndarray:
    """
    Post-processing filter that clamps SMPL-X local joint rotations within
    medically accurate anatomical range-of-motion (ROM) limits.
    Prevents joint inversion, hyper-extension, neck 180-degree rotation,
    shoulder dislocation, and spinal breakage caused by MediaPipe tracking noise.

    Accepts:
        local_rotations: np.ndarray of shape (22, 4) or (T, 22, 4)
    Returns:
        np.ndarray of same shape with anatomically constrained unit quaternions.
    """
    if local_rotations.ndim == 2:
        return _apply_joint_limits_single_frame(local_rotations)
    elif local_rotations.ndim == 3:
        out = np.copy(local_rotations)
        for t in range(local_rotations.shape[0]):
            out[t] = _apply_joint_limits_single_frame(local_rotations[t])
        return out
    else:
        raise ValueError(f"apply_anatomical_joint_limits expects 2D or 3D array, got shape {local_rotations.shape}")
# ============================================================================

def convert_mediapipe_landmarks_to_smplx_3d(
    mp_landmarks: np.ndarray,
    norm_landmarks: np.ndarray = None,
    invert_y: bool = True,
    invert_z: bool = True,
    resolve_occlusion: bool = True,
) -> np.ndarray:
    """
    Converts 33 MediaPipe 3D world landmarks into 22 SMPL-X joint 3D positions.
    Coordinate frame conversion:
    MediaPipe World Landmarks use a camera-aligned metric frame.  ``invert_y``
    and ``invert_z`` are explicit because the lightweight backends do not all
    emit the same axis convention: MediaPipe world and RTMPose pseudo-world
    outputs use +Y down/+Z away.  The legacy defaults retain the original
    MediaPipe-to-SMPL-X conversion for callers that pass raw camera-oriented
    data; the extraction pipeline selects the policy from the active detector
    instead of applying one global flip.
    TexMotion SMPL-X Frame:
    +X: Character's Left (+X)
    -X: Character's Right (-X)
    +Y: Up (+Y)
    +Z: Forward (+Z)
    """
    lm = np.zeros_like(mp_landmarks)
    lm[:, 0] = mp_landmarks[:, 0]    # Left is +X, Right is -X
    lm[:, 1] = -mp_landmarks[:, 1] if invert_y else mp_landmarks[:, 1]
    lm[:, 2] = -mp_landmarks[:, 2] if invert_z else mp_landmarks[:, 2]

    joints = np.zeros((SMPLX_JOINT_COUNT, 3), dtype=np.float64)

    # 0: Pelvis (midpoint between hips)
    l_hip = lm[MP_LEFT_HIP]
    r_hip = lm[MP_RIGHT_HIP]
    pelvis = (l_hip + r_hip) * 0.5
    joints[0] = pelvis
    joints[1] = l_hip
    joints[2] = r_hip

    # Shoulders and Torso
    l_shoulder = lm[MP_LEFT_SHOULDER]
    r_shoulder = lm[MP_RIGHT_SHOULDER]
    shoulder_mid = (l_shoulder + r_shoulder) * 0.5

    # Spine positions smoothly placed along the torso
    joints[3] = pelvis + (shoulder_mid - pelvis) * 0.333  # Spine1 (lower lumbar)
    joints[6] = pelvis + (shoulder_mid - pelvis) * 0.667  # Spine2 (mid spine)
    joints[9] = pelvis + (shoulder_mid - pelvis) * 0.900  # Spine3 (upper chest)

    # Head and Neck: robust anchor using nose, eyes, and ears
    nose = lm[MP_NOSE]
    ear_mid = (lm[MP_LEFT_EAR] + lm[MP_RIGHT_EAR]) * 0.5
    if np.linalg.norm(ear_mid) > 0.01:
        head = (nose + ear_mid) * 0.5
    else:
        eye_mid = (lm[MP_LEFT_EYE] + lm[MP_RIGHT_EYE]) * 0.5
        head = (nose + eye_mid) * 0.5 if np.linalg.norm(eye_mid) > 0.01 else nose + np.array([0.0, 0.08, -0.02])

    joints[12] = shoulder_mid + (head - shoulder_mid) * 0.35  # Neck
    joints[15] = head                                        # Head

    # 13: L_Collar & 14: R_Collar
    joints[13] = joints[9] + (l_shoulder - joints[9]) * 0.45
    joints[14] = joints[9] + (r_shoulder - joints[9]) * 0.45

    # Arms
    # 16: L_Shoulder, 18: L_Elbow, 20: L_Wrist
    joints[16] = l_shoulder
    joints[18] = lm[MP_LEFT_ELBOW]
    joints[20] = lm[MP_LEFT_WRIST]

    # 17: R_Shoulder, 19: R_Elbow, 21: R_Wrist
    joints[17] = r_shoulder
    joints[19] = lm[MP_RIGHT_ELBOW]
    joints[21] = lm[MP_RIGHT_WRIST]

    # Legs
    # 4: L_Knee, 7: L_Ankle, 10: L_Foot
    joints[4] = lm[MP_LEFT_KNEE]
    joints[7] = lm[MP_LEFT_ANKLE]
    joints[10] = lm[MP_LEFT_FOOT_INDEX]

    # 5: R_Knee, 8: R_Ankle, 11: R_Foot
    joints[5] = lm[MP_RIGHT_KNEE]
    joints[8] = lm[MP_RIGHT_ANKLE]
    joints[11] = lm[MP_RIGHT_FOOT_INDEX]

    # Resolve behind-the-head and behind-the-back arm occlusions only for the
    # legacy MediaPipe-derived stream.  Canonical WHAM/fused joints already
    # represent the learned 3D hypothesis; applying this legacy heuristic to
    # them can silently rewrite an occluded limb and invalidate provenance.
    if resolve_occlusion:
        joints = resolve_arm_occlusion_and_behind_head_pose(joints, norm_landmarks)

    return joints


def resolve_arm_occlusion_and_behind_head_pose(
    joints: np.ndarray,
    norm_landmarks: np.ndarray = None
) -> np.ndarray:
    """
    Solves behind-the-head and behind-the-back arm occlusion failure cases in MediaPipe.
    When a subject places their hands behind their head (e.g. resting head in hands, sit-ups, stretches):
    - Frontal camera loses line-of-sight to the wrists and hands (self-occlusion).
    - MediaPipe often hallucinates and collapses the wrists onto the nose/mouth in front of the face,
      while pulling elbows inward toward the ears/cheeks, causing the avatar's arms to shrivel into the chest.

    This function detects this specific geometric collapse pattern:
    1. Upper arm is raised / elevated near or above shoulder height.
    2. Wrist has low visibility OR wrist is collapsed into the head volume (X near center, Z in front of face).
    3. Reconstructs anatomically sound 3D elbow and wrist positions behind the head (-Z in SMPL-X frame).
    """
    out = np.copy(joints)
    head = out[15]
    neck = out[12]
    pelvis = out[0]
    l_sh = out[16]
    r_sh = out[17]

    head_radius = 0.18

    # --- Left Arm Behind-the-Head Occlusion Check ---
    l_el = out[18]
    l_wr = out[20]

    vis_l_wr = norm_landmarks[MP_LEFT_WRIST, 2] if (norm_landmarks is not None and len(norm_landmarks) > MP_LEFT_WRIST) else 1.0
    vis_l_el = norm_landmarks[MP_LEFT_ELBOW, 2] if (norm_landmarks is not None and len(norm_landmarks) > MP_LEFT_ELBOW) else 1.0

    # Is the upper arm raised? (Elbow Y near or above shoulder height in SMPL-X +Y up)
    l_arm_raised = (l_el[1] >= l_sh[1] - 0.14)
    # Is the wrist collapsed into the head volume (X near center, Y near head/neck)?
    l_wr_in_head_region = (abs(l_wr[0] - head[0]) < head_radius and abs(l_wr[1] - head[1]) < 0.22)
    # Is the wrist in front of the head (+Z in SMPL-X)?
    l_wr_in_front_of_head = (l_wr[2] >= head[2] - 0.05)

    if l_arm_raised and l_wr_in_head_region and (vis_l_wr < 0.75 or l_wr_in_front_of_head):
        # Target wrist: placed behind the occipital / neck region (-Z in SMPL-X)
        out[20, 0] = head[0] + 0.05   # slightly left of midline (+X)
        out[20, 1] = head[1] - 0.04   # back of skull / upper neck
        out[20, 2] = head[2] - 0.14   # BEHIND head (-Z in SMPL-X)

        # Check if elbow also collapsed inward toward ears / cheeks
        l_upper_arm_len = 0.28
        if l_el[0] < l_sh[0] + 0.08:
            # Reconstruct natural flared outward-upward elbow position
            flair_dir = normalize_vector(np.array([0.85, 0.45, -0.28]))
            out[18] = l_sh + flair_dir * l_upper_arm_len
        else:
            out[18, 2] = min(out[18, 2], head[2] - 0.05)

    # --- Right Arm Behind-the-Head Occlusion Check ---
    r_el = out[19]
    r_wr = out[21]

    vis_r_wr = norm_landmarks[MP_RIGHT_WRIST, 2] if (norm_landmarks is not None and len(norm_landmarks) > MP_RIGHT_WRIST) else 1.0
    vis_r_el = norm_landmarks[MP_RIGHT_ELBOW, 2] if (norm_landmarks is not None and len(norm_landmarks) > MP_RIGHT_ELBOW) else 1.0

    r_arm_raised = (r_el[1] >= r_sh[1] - 0.14)
    r_wr_in_head_region = (abs(r_wr[0] - head[0]) < head_radius and abs(r_wr[1] - head[1]) < 0.22)
    r_wr_in_front_of_head = (r_wr[2] >= head[2] - 0.05)

    if r_arm_raised and r_wr_in_head_region and (vis_r_wr < 0.75 or r_wr_in_front_of_head):
        out[21, 0] = head[0] - 0.05   # slightly right of midline (-X)
        out[21, 1] = head[1] - 0.04   # back of skull / upper neck
        out[21, 2] = head[2] - 0.14   # BEHIND head (-Z in SMPL-X)

        r_upper_arm_len = 0.28
        if r_el[0] > r_sh[0] - 0.08:
            flair_dir_r = normalize_vector(np.array([-0.85, 0.45, -0.28]))
            out[19] = r_sh + flair_dir_r * r_upper_arm_len
        else:
            out[19, 2] = min(out[19, 2], head[2] - 0.05)

    # --- Behind-the-Back Occlusion Check (Hands Behind Lower Back / Hips) ---
    l_wr_near_pelvis = (abs(l_wr[0] - pelvis[0]) < 0.22 and abs(l_wr[1] - pelvis[1]) < 0.20)
    if l_wr_near_pelvis and vis_l_wr < 0.65 and l_wr[2] >= pelvis[2]:
        out[20, 2] = pelvis[2] - 0.12

    r_wr_near_pelvis = (abs(r_wr[0] - pelvis[0]) < 0.22 and abs(r_wr[1] - pelvis[1]) < 0.20)
    if r_wr_near_pelvis and vis_r_wr < 0.65 and r_wr[2] >= pelvis[2]:
        out[21, 2] = pelvis[2] - 0.12

    return out


def resolve_crossing_legs_occlusion(
    joints_seq: np.ndarray,
    norm_landmarks_seq: list = None
) -> np.ndarray:
    """
    Detects crossing or tightly overlapping legs in 2D image coordinates and 3D landmark streams,
    and reconstructs anatomically accurate lateral leg crossing and anterior-posterior depth separation.
    
    MediaPipe Pose fundamentally collapses crossed legs into narrow parallel straight legs
    due to single-camera depth ambiguity and lack of self-occlusion reasoning for limbs.
    This function analyzes the lateral displacement, horizontal disparity, foot index orientation,
    and relative depth to robustly reconstruct full crossing motions.
    """
    T = joints_seq.shape[0]
    if T == 0:
        return joints_seq

    out = np.copy(joints_seq)

    # 1. First pass: detect crossing intent and direction per frame
    crossing_weights = np.zeros(T, dtype=np.float64)
    crossing_side = np.zeros(T, dtype=np.int32)  # +1: Right in front, -1: Left in front

    for t in range(T):
        p_hip_l = out[t, 1]
        p_hip_r = out[t, 2]
        p_kn_l  = out[t, 4]
        p_kn_r  = out[t, 5]
        p_ank_l = out[t, 7]
        p_ank_r = out[t, 8]
        pelvis  = out[t, 0]

        norm = norm_landmarks_seq[t] if (norm_landmarks_seq and t < len(norm_landmarks_seq)) else None

        if norm is not None and len(norm) > 32:
            # MediaPipe 2D image coordinates: X in [0, 1] (0 is left, 1 is right of image)
            # Subject's Left Ankle is landmark 27 (typically on image right ~0.52)
            # Subject's Right Ankle is landmark 28 (typically on image left ~0.48)
            dx_2d = float(norm[27, 0] - norm[28, 0])
            dist_3d = float(np.linalg.norm(p_ank_l - p_ank_r))

            # When legs are crossing, dx_2d drops below ~0.038 or becomes negative, and dist_3d < 0.16m
            if dx_2d < 0.038 or dist_3d < 0.16:
                # Intensity of crossing [0.0, 1.0]
                w_cross = float(np.clip((0.040 - dx_2d) / 0.045, 0.0, 1.0))
                if dist_3d < 0.12 and w_cross < 0.4:
                    w_cross = float(np.clip((0.12 - dist_3d) / 0.08, 0.4, 1.0))

                crossing_weights[t] = w_cross

                # Which leg is crossing in front?
                # In TexMotion SMPL-X coordinates: +Z is forward (towards camera / front of body)
                dz_3d = out[t, 8, 2] - out[t, 7, 2]
                r_toe_cross = norm[32, 0] > norm[24, 0] + 0.01
                l_toe_cross = norm[31, 0] < norm[23, 0] - 0.01

                if (dz_3d >= -0.02) or (norm[28, 0] >= norm[27, 0] - 0.015) or r_toe_cross:
                    crossing_side[t] = 1   # Right crosses in front
                else:
                    crossing_side[t] = -1  # Left crosses in front

    # Temporal smoothing of crossing weights to eliminate single-frame fluttering
    if T >= 5:
        kernel = np.array([0.15, 0.70, 0.15], dtype=np.float64)
        smoothed_weights = np.convolve(crossing_weights, kernel, mode='same')
    else:
        smoothed_weights = crossing_weights

    # 2. Second pass: apply lateral crossing and depth offset
    for t in range(T):
        w = smoothed_weights[t]
        if w < 0.08:
            continue

        side = crossing_side[t]
        if side == 0:
            side = 1

        p_hip_l = out[t, 1]
        p_hip_r = out[t, 2]
        p_kn_l  = out[t, 4]
        p_kn_r  = out[t, 5]
        p_ank_l = out[t, 7]
        p_ank_r = out[t, 8]
        pelvis  = out[t, 0]

        if side == 1:
            # Right leg crosses in front of Left leg:
            # In TexMotion SMPL-X frame: +X is character's Left, -X is character's Right
            target_ank_x = max(p_ank_r[0], p_ank_l[0] + 0.055 * w)
            target_ank_z = max(p_ank_r[2], p_ank_l[2] + 0.075 * w)  # Forward in front
            target_kn_x  = max(p_kn_r[0], pelvis[0] + 0.025 * w)
            target_kn_z  = max(p_kn_r[2], pelvis[2] + 0.040 * w)

            out[t, 8, 0] = (1.0 - w) * p_ank_r[0] + w * target_ank_x
            out[t, 8, 2] = (1.0 - w) * p_ank_r[2] + w * target_ank_z
            out[t, 5, 0] = (1.0 - w) * p_kn_r[0] + w * target_kn_x
            out[t, 5, 2] = (1.0 - w) * p_kn_r[2] + w * target_kn_z

            out[t, 11, 0] = out[t, 8, 0] + 0.02 * w
            out[t, 11, 2] = out[t, 8, 2] + 0.14

            # Left leg (support / back leg) stays stable near body center
            out[t, 7, 0] = min(out[t, 7, 0], max(0.01, p_hip_l[0] * 0.35))
            out[t, 7, 2] = min(out[t, 7, 2], out[t, 8, 2] - 0.05 * w)
        else:
            # Left leg crosses in front of Right leg:
            target_ank_x = min(p_ank_l[0], p_ank_r[0] - 0.055 * w)
            target_ank_z = max(p_ank_l[2], p_ank_r[2] + 0.075 * w)
            target_kn_x  = min(p_kn_l[0], pelvis[0] - 0.025 * w)
            target_kn_z  = max(p_kn_l[2], pelvis[2] + 0.040 * w)

            out[t, 7, 0] = (1.0 - w) * p_ank_l[0] + w * target_ank_x
            out[t, 7, 2] = (1.0 - w) * p_ank_l[2] + w * target_ank_z
            out[t, 4, 0] = (1.0 - w) * p_kn_l[0] + w * target_kn_x
            out[t, 4, 2] = (1.0 - w) * p_kn_l[2] + w * target_kn_z

            out[t, 10, 0] = out[t, 7, 0] - 0.02 * w
            out[t, 10, 2] = out[t, 7, 2] + 0.14

            out[t, 8, 0] = max(out[t, 8, 0], min(-0.01, p_hip_r[0] * 0.35))
            out[t, 8, 2] = min(out[t, 8, 2], out[t, 7, 2] - 0.05 * w)

    return out


def inpaint_and_constrain_leg_kinematics(
    joints_seq: np.ndarray,
    norm_landmarks_seq: list = None
) -> np.ndarray:
    """
    Guarantees anatomically valid, stable legs even when MediaPipe completely loses tracking
    (e.g., crossing legs, self-occlusion, floor blending, or cropped ROI).
    
    1. Detects invalid / lost legs per frame using visibility, bone lengths, and spatial sanity.
    2. Interpolates (inpaints) missing leg poses smoothly from neighboring valid frames.
    3. If an entire segment or video has missing legs, synthesizes stable anatomical legs from pelvis.
    4. Strictly enforces constant bone lengths (thigh and shin lengths) while fully preserving
       lateral adduction and crossing angles.
    """
    T = joints_seq.shape[0]
    if T == 0:
        return joints_seq

    out = np.copy(joints_seq)

    valid_l = np.ones(T, dtype=bool)
    valid_r = np.ones(T, dtype=bool)

    # Standard human anatomical proportions (relative to Pelvis)
    STD_THIGH_LEN = 0.42
    STD_SHIN_LEN = 0.42

    for t in range(T):
        norm = norm_landmarks_seq[t] if (norm_landmarks_seq and t < len(norm_landmarks_seq)) else None

        # --- Left Leg Inspection ---
        p_hip_l = out[t, 1]
        p_kn_l = out[t, 4]
        p_ank_l = out[t, 7]
        thigh_l = float(np.linalg.norm(p_kn_l - p_hip_l))
        shin_l = float(np.linalg.norm(p_ank_l - p_kn_l))

        vis_l = 1.0
        if norm is not None and len(norm) > MP_LEFT_FOOT_INDEX:
            vis_l = float(norm[MP_LEFT_KNEE, 2] + norm[MP_LEFT_ANKLE, 2] + norm[MP_LEFT_FOOT_INDEX, 2]) / 3.0

        is_valid_len_l = (0.20 <= thigh_l <= 0.65) and (0.20 <= shin_l <= 0.65)
        dist_from_pelvis_l = float(np.linalg.norm(p_ank_l - out[t, 0]))
        is_reasonable_reach_l = (dist_from_pelvis_l <= 1.35) and (float(np.linalg.norm(p_ank_l - p_hip_l)) <= (thigh_l + shin_l + 0.12))
        geom_valid_l = is_valid_len_l and is_reasonable_reach_l

        if not geom_valid_l or vis_l < 0.08:
            valid_l[t] = False

        # --- Right Leg Inspection ---
        p_hip_r = out[t, 2]
        p_kn_r = out[t, 5]
        p_ank_r = out[t, 8]
        thigh_r = float(np.linalg.norm(p_kn_r - p_hip_r))
        shin_r = float(np.linalg.norm(p_ank_r - p_kn_r))

        vis_r = 1.0
        if norm is not None and len(norm) > MP_RIGHT_FOOT_INDEX:
            vis_r = float(norm[MP_RIGHT_KNEE, 2] + norm[MP_RIGHT_ANKLE, 2] + norm[MP_RIGHT_FOOT_INDEX, 2]) / 3.0

        is_valid_len_r = (0.20 <= thigh_r <= 0.65) and (0.20 <= shin_r <= 0.65)
        dist_from_pelvis_r = float(np.linalg.norm(p_ank_r - out[t, 0]))
        is_reasonable_reach_r = (dist_from_pelvis_r <= 1.35) and (float(np.linalg.norm(p_ank_r - p_hip_r)) <= (thigh_r + shin_r + 0.12))
        geom_valid_r = is_valid_len_r and is_reasonable_reach_r

        if not geom_valid_r or vis_r < 0.08:
            valid_r[t] = False

    # Reference lengths computed from valid frames or standard default
    val_thighs_l = [np.linalg.norm(out[t, 4] - out[t, 1]) for t in range(T) if valid_l[t]]
    val_shins_l = [np.linalg.norm(out[t, 7] - out[t, 4]) for t in range(T) if valid_l[t]]
    ref_thigh_l = float(np.clip(np.median(val_thighs_l), 0.35, 0.48)) if len(val_thighs_l) > 0 else STD_THIGH_LEN
    ref_shin_l = float(np.clip(np.median(val_shins_l), 0.35, 0.48)) if len(val_shins_l) > 0 else STD_SHIN_LEN

    val_thighs_r = [np.linalg.norm(out[t, 5] - out[t, 2]) for t in range(T) if valid_r[t]]
    val_shins_r = [np.linalg.norm(out[t, 8] - out[t, 5]) for t in range(T) if valid_r[t]]
    ref_thigh_r = float(np.clip(np.median(val_thighs_r), 0.35, 0.48)) if len(val_thighs_r) > 0 else STD_THIGH_LEN
    ref_shin_r = float(np.clip(np.median(val_shins_r), 0.35, 0.48)) if len(val_shins_r) > 0 else STD_SHIN_LEN

    # --- Temporal Inpainting for Left Leg ---
    valid_indices_l = np.where(valid_l)[0]
    if len(valid_indices_l) > 0:
        for t in range(T):
            if not valid_l[t]:
                before = valid_indices_l[valid_indices_l < t]
                after = valid_indices_l[valid_indices_l > t]

                if len(before) > 0 and len(after) > 0:
                    t0 = before[-1]
                    t1 = after[0]
                    alpha = (t - t0) / float(t1 - t0)
                    v_kn_0 = out[t0, 4] - out[t0, 1]
                    v_kn_1 = out[t1, 4] - out[t1, 1]
                    v_kn = (1.0 - alpha) * v_kn_0 + alpha * v_kn_1

                    v_ank_0 = out[t0, 7] - out[t0, 4]
                    v_ank_1 = out[t1, 7] - out[t1, 4]
                    v_ank = (1.0 - alpha) * v_ank_0 + alpha * v_ank_1

                    out[t, 4] = out[t, 1] + normalize_vector(v_kn) * ref_thigh_l
                    out[t, 7] = out[t, 4] + normalize_vector(v_ank) * ref_shin_l
                elif len(before) > 0:
                    t0 = before[-1]
                    v_kn = out[t0, 4] - out[t0, 1]
                    v_ank = out[t0, 7] - out[t0, 4]
                    out[t, 4] = out[t, 1] + normalize_vector(v_kn) * ref_thigh_l
                    out[t, 7] = out[t, 4] + normalize_vector(v_ank) * ref_shin_l
                else:
                    t1 = after[0]
                    v_kn = out[t1, 4] - out[t1, 1]
                    v_ank = out[t1, 7] - out[t1, 4]
                    out[t, 4] = out[t, 1] + normalize_vector(v_kn) * ref_thigh_l
                    out[t, 7] = out[t, 4] + normalize_vector(v_ank) * ref_shin_l
                out[t, 10] = out[t, 7] + np.array([0.0, -0.05, 0.14])
    else:
        for t in range(T):
            out[t, 4] = out[t, 1] + np.array([-0.02, -ref_thigh_l, 0.02])
            out[t, 7] = out[t, 4] + np.array([0.02, -ref_shin_l, -0.01])
            out[t, 10] = out[t, 7] + np.array([0.0, -0.05, 0.14])

    # --- Temporal Inpainting for Right Leg ---
    valid_indices_r = np.where(valid_r)[0]
    if len(valid_indices_r) > 0:
        for t in range(T):
            if not valid_r[t]:
                before = valid_indices_r[valid_indices_r < t]
                after = valid_indices_r[valid_indices_r > t]

                if len(before) > 0 and len(after) > 0:
                    t0 = before[-1]
                    t1 = after[0]
                    alpha = (t - t0) / float(t1 - t0)
                    v_kn_0 = out[t0, 5] - out[t0, 2]
                    v_kn_1 = out[t1, 5] - out[t1, 2]
                    v_kn = (1.0 - alpha) * v_kn_0 + alpha * v_kn_1

                    v_ank_0 = out[t0, 8] - out[t0, 5]
                    v_ank_1 = out[t1, 8] - out[t1, 5]
                    v_ank = (1.0 - alpha) * v_ank_0 + alpha * v_ank_1

                    out[t, 5] = out[t, 2] + normalize_vector(v_kn) * ref_thigh_r
                    out[t, 8] = out[t, 5] + normalize_vector(v_ank) * ref_shin_r
                elif len(before) > 0:
                    t0 = before[-1]
                    v_kn = out[t0, 5] - out[t0, 2]
                    v_ank = out[t0, 8] - out[t0, 5]
                    out[t, 5] = out[t, 2] + normalize_vector(v_kn) * ref_thigh_r
                    out[t, 8] = out[t, 5] + normalize_vector(v_ank) * ref_shin_r
                else:
                    t1 = after[0]
                    v_kn = out[t1, 5] - out[t1, 2]
                    v_ank = out[t1, 8] - out[t1, 5]
                    out[t, 5] = out[t, 2] + normalize_vector(v_kn) * ref_thigh_r
                    out[t, 8] = out[t, 5] + normalize_vector(v_ank) * ref_shin_r
                out[t, 11] = out[t, 8] + np.array([0.0, -0.05, 0.14])
    else:
        for t in range(T):
            out[t, 5] = out[t, 2] + np.array([0.02, -ref_thigh_r, 0.02])
            out[t, 8] = out[t, 5] + np.array([-0.02, -ref_shin_r, -0.01])
            out[t, 11] = out[t, 8] + np.array([0.0, -0.05, 0.14])

    # --- Strict Constant Bone Length Enforcement Preserving Crossing Tilt ---
    for t in range(T):
        # Left Leg
        p_hip_l = out[t, 1]
        v_thigh_l = out[t, 4] - p_hip_l
        dir_thigh_l = normalize_vector(v_thigh_l) if np.linalg.norm(v_thigh_l) > 1e-4 else np.array([0.0, -1.0, 0.0])
        out[t, 4] = p_hip_l + dir_thigh_l * ref_thigh_l

        v_shin_l = out[t, 7] - out[t, 4]
        dir_shin_l = normalize_vector(v_shin_l) if np.linalg.norm(v_shin_l) > 1e-4 else np.array([0.0, -1.0, 0.0])
        out[t, 7] = out[t, 4] + dir_shin_l * ref_shin_l

        # Prevent knee hyperextension
        if POSE_PIPELINE_AVAILABLE:
            out[t, 4] = constrain_knee_flexion(p_hip_l, out[t, 4], out[t, 7], is_left=True)

        v_toe_l = out[t, 10] - out[t, 7]
        if np.linalg.norm(v_toe_l) < 0.05 or np.linalg.norm(v_toe_l) > 0.25:
            v_toe_l = np.array([0.0, -0.05, 0.14])
        out[t, 10] = out[t, 7] + normalize_vector(v_toe_l) * 0.14

        # Right Leg
        p_hip_r = out[t, 2]
        v_thigh_r = out[t, 5] - p_hip_r
        dir_thigh_r = normalize_vector(v_thigh_r) if np.linalg.norm(v_thigh_r) > 1e-4 else np.array([0.0, -1.0, 0.0])
        out[t, 5] = p_hip_r + dir_thigh_r * ref_thigh_r

        v_shin_r = out[t, 8] - out[t, 5]
        dir_shin_r = normalize_vector(v_shin_r) if np.linalg.norm(v_shin_r) > 1e-4 else np.array([0.0, -1.0, 0.0])
        out[t, 8] = out[t, 5] + dir_shin_r * ref_shin_r

        # Prevent knee hyperextension
        if POSE_PIPELINE_AVAILABLE:
            out[t, 5] = constrain_knee_flexion(p_hip_r, out[t, 5], out[t, 8], is_left=False)

        v_toe_r = out[t, 11] - out[t, 8]
        if np.linalg.norm(v_toe_r) < 0.05 or np.linalg.norm(v_toe_r) > 0.25:
            v_toe_r = np.array([0.0, -0.05, 0.14])
        out[t, 11] = out[t, 8] + normalize_vector(v_toe_r) * 0.14

    return out


# ============================================================================
# Analytical 2-Bone Inverse Kinematics & SMPL-X Local Rotations
# ============================================================================

def solve_two_bone_ik(
    p_root: np.ndarray,
    p_mid: np.ndarray,
    p_target: np.ndarray,
    l1: float,
    l2: float,
    default_bend_dir: np.ndarray
) -> np.ndarray:
    """
    Solves analytical 2-bone IK: given p_root, target end-effector p_target,
    and bone lengths l1, l2, analytically computes the middle joint position p_mid.
    Preserves the knee/elbow bend direction.
    """
    d_vec = p_target - p_root
    d = np.linalg.norm(d_vec)
    d_clamped = np.clip(d, 1e-4, (l1 + l2) * 0.9999)
    u_d = d_vec / d if d > 1e-6 else np.array([0.0, -1.0, 0.0])

    cos_alpha = np.clip((l1**2 + d_clamped**2 - l2**2) / (2.0 * l1 * d_clamped), -1.0, 1.0)
    sin_alpha = np.sqrt(max(0.0, 1.0 - cos_alpha**2))

    k_orig = p_mid - p_root
    k_proj = k_orig - np.dot(k_orig, u_d) * u_d
    if np.linalg.norm(k_proj) < 1e-4:
        k_proj = default_bend_dir - np.dot(default_bend_dir, u_d) * u_d
    if np.linalg.norm(k_proj) < 1e-4:
        k_proj = np.array([0.0, 0.0, 1.0])

    u_k = normalize_vector(k_proj)
    p_mid_solved = p_root + l1 * (cos_alpha * u_d + sin_alpha * u_k)
    return p_mid_solved


def solve_smplx_local_rotations(
    joints_3d: np.ndarray,
    prev_q_pelvis: np.ndarray = None,
    hand_landmarks: dict = None,
    prev_wrist_twist: tuple = None
) -> tuple:
    """
    Computes all 22 SMPL-X local joint rotations as unit quaternions [x, y, z, w]
    from 3D joint positions using Anatomical Pole-Vector 2-Bone IK.
    Supports full 360-degree pelvis orientations (including handstands, flips, and inverted poses),
    anatomical wrist flexion/pronation estimation from hand landmarks, and enforces strict
    anatomical range-of-motion limits to eliminate reverse joints and dislocations.
    Output: (local_rotations [22, 4], q_pelvis [4], (twist_l, twist_r))
    """
    world_rotations = [np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64) for _ in range(SMPLX_JOINT_COUNT)]

    # 1. Pelvis Orientation with 360-Degree Inversion/Handstand Support
    l_hip = joints_3d[1]
    r_hip = joints_3d[2]
    pelvis = joints_3d[0]
    spine1 = joints_3d[3]
    l_sh = joints_3d[16]
    r_sh = joints_3d[17]
    shoulder_mid = (l_sh + r_sh) * 0.5

    x_hips = normalize_vector(l_hip - r_hip)
    x_sh = normalize_vector(l_sh - r_sh)
    d_hips = float(np.linalg.norm(l_hip - r_hip))

    w_sh = 0.85 if d_hips < 0.12 else 0.35
    x_pelvis = normalize_vector(w_sh * x_sh + (1.0 - w_sh) * x_hips)
    if np.linalg.norm(x_pelvis) < 1e-4:
        x_pelvis = np.array([1.0, 0.0, 0.0])

    # Spine vector (shoulder_mid - pelvis) - faithfully captures full 360-degree tilt and inversion
    spine_vec = shoulder_mid - pelvis
    if np.linalg.norm(spine_vec) < 1e-4:
        spine_vec = spine1 - pelvis
    if np.linalg.norm(spine_vec) < 1e-4:
        spine_vec = np.array([0.0, 1.0, 0.0])
    u_raw = normalize_vector(spine_vec)

    # Dynamically clamp single-camera perspective Z-axis depth illusion without killing real dives/bows
    # Allows deep forward bows and crouching dives (up to ~60 deg) while avoiding extreme depth noise
    u_pelvis_z = np.clip(u_raw[2], -0.85, 0.85)
    u_pelvis = normalize_vector(np.array([u_raw[0], u_raw[1], u_pelvis_z]))

    # Forward vector (Z) computed from lateral and spine axes
    z_raw = np.cross(x_pelvis, u_pelvis)
    if np.linalg.norm(z_raw) < 1e-3:
        if prev_q_pelvis is not None:
            z_raw = quat_rotate_vector(prev_q_pelvis, np.array([0.0, 0.0, 1.0]))
        else:
            z_raw = np.array([0.0, 0.0, 1.0])
    z_pelvis = normalize_vector(z_raw)

    # Strict orthonormal basis construction
    x_pelvis_ortho = normalize_vector(np.cross(u_pelvis, z_pelvis))
    y_pelvis_ortho = normalize_vector(np.cross(z_pelvis, x_pelvis_ortho))
    R_pelvis = np.column_stack([x_pelvis_ortho, y_pelvis_ortho, z_pelvis])
    q_pelvis = matrix_to_quaternion(R_pelvis)

    if prev_q_pelvis is not None:
        if np.dot(prev_q_pelvis, q_pelvis) < 0.0:
            q_pelvis = -q_pelvis
    world_rotations[0] = q_pelvis

    # 2. Chest (Spine3) Orientation
    neck = joints_3d[12]
    spine3 = joints_3d[9]
    u_raw_chest = normalize_vector(neck - spine3)
    if np.linalg.norm(neck - spine3) < 1e-4:
        u_raw_chest = u_pelvis
    u_chest_z = np.clip(u_raw_chest[2], -0.85, 0.85)
    u_chest = normalize_vector(np.array([u_raw_chest[0], u_raw_chest[1], u_chest_z]))

    z_chest_raw = np.cross(x_sh, u_chest)
    if np.linalg.norm(z_chest_raw) < 1e-4:
        z_chest_raw = z_pelvis
    z_chest = normalize_vector(z_chest_raw)
    x_chest = normalize_vector(np.cross(u_chest, z_chest))
    y_chest = normalize_vector(np.cross(z_chest, x_chest))
    R_chest = np.column_stack([x_chest, y_chest, z_chest])
    q_spine3 = matrix_to_quaternion(R_chest)
    world_rotations[9] = q_spine3

    # Spine1 & Spine2 interpolated smoothly between pelvis and chest
    world_rotations[3] = quaternion_slerp(q_pelvis, q_spine3, 0.333)
    world_rotations[6] = quaternion_slerp(q_pelvis, q_spine3, 0.667)

    # 3. Neck & Head
    head = joints_3d[15]
    u_head = normalize_vector(head - neck)
    # Align head with chest direction; fallback if degenerate or inverted relative to chest
    if np.linalg.norm(head - neck) < 1e-4 or np.dot(u_head, y_chest) < 0.2:
        u_head = y_chest
    z_head_raw = np.cross(x_chest, u_head)
    if np.linalg.norm(z_head_raw) < 1e-4:
        z_head_raw = z_chest
    z_head = normalize_vector(z_head_raw)
    x_head = normalize_vector(np.cross(u_head, z_head))
    y_head = normalize_vector(np.cross(z_head, x_head))
    world_rotations[15] = matrix_to_quaternion(np.column_stack([x_head, y_head, z_head]))
    world_rotations[12] = quaternion_slerp(q_spine3, world_rotations[15], 0.5)

    # 4. Collars
    world_rotations[13] = q_spine3  # L_Collar
    world_rotations[14] = q_spine3  # R_Collar

    # 5. Anatomical Pole-Vector 2-Bone IK for Arms
    # Standard SMPL-X T-Pose:
    # Left Arm extends along +X ([1, 0, 0]), bend direction is +Z (Forward), hinge axis is [0, -1, 0]
    # Right Arm extends along -X ([-1, 0, 0]), bend direction is +Z (Forward), hinge axis is [0, 1, 0]

    # --- Left Arm ---
    p_sh_l = joints_3d[16]
    p_el_l = joints_3d[18]
    p_wr_l = joints_3d[20]

    v_up_l = normalize_vector(p_el_l - p_sh_l)
    v_fo_l = normalize_vector(p_wr_l - p_el_l)

    # Elbow pole normal (triangle normal of sh-el-wr)
    # Left Arm SMPL-X rest has local hinge = -Y, bone = +X. To bend forward (+Z), Y column must align with -n.
    n_pole_l = -np.cross(v_up_l, v_fo_l)
    if np.linalg.norm(n_pole_l) < 0.06:
        # Arm is nearly straight: fallback to chest forward to define bend plane
        n_pole_l = -np.cross(v_up_l, z_chest)
        if np.linalg.norm(n_pole_l) < 0.02:
            n_pole_l = np.array([0.0, -1.0, 0.0])
    n_pole_l = normalize_vector(n_pole_l)

    # Construct orthonormal orientation matrix for Left Upper Arm:
    # X: upper arm bone axis (+X = v_up_l)
    # Y: hinge axis
    # Z: bend direction (forward)
    x_arm_l = v_up_l
    y_arm_l = n_pole_l
    z_arm_l = normalize_vector(np.cross(x_arm_l, y_arm_l))
    y_arm_l = normalize_vector(np.cross(z_arm_l, x_arm_l))
    R_arm_l = np.column_stack([x_arm_l, y_arm_l, z_arm_l])
    world_rotations[16] = matrix_to_quaternion(R_arm_l)

    # Left Elbow: strictly 1-DOF hinge around [0, -1, 0] (local Y axis)
    dot_el_l = float(np.clip(np.dot(v_up_l, v_fo_l), -1.0, 1.0))
    angle_el_l = float(np.arccos(dot_el_l))
    angle_el_l = float(np.clip(angle_el_l, 0.0, np.radians(150.0)))
    half_el_l = angle_el_l * 0.5
    s_l = np.sin(half_el_l)
    c_l = np.cos(half_el_l)
    q_el_local_l = np.array([0.0, -s_l, 0.0, c_l], dtype=np.float64)
    world_rotations[18] = quaternion_multiply(world_rotations[16], q_el_local_l)

    # --- Left Wrist (Joint 20) Orientation Estimation ---
    p_idx_l = hand_landmarks.get("l_index", None) if hand_landmarks else None
    p_pnk_l = hand_landmarks.get("l_pinky", None) if hand_landmarks else None
    prev_twist_l = prev_wrist_twist[0] if (prev_wrist_twist and len(prev_wrist_twist) > 0) else None
    prev_twist_r = prev_wrist_twist[1] if (prev_wrist_twist and len(prev_wrist_twist) > 1) else None
    twist_l_out = 0.0
    twist_r_out = 0.0

    if p_idx_l is not None and p_pnk_l is not None:
        p_hand_center_l = 0.5 * (p_idx_l + p_pnk_l)
        v_hand_raw_l = p_hand_center_l - p_wr_l
        if np.linalg.norm(v_hand_raw_l) > 0.02:
            v_hand_l = normalize_vector(v_hand_raw_l)
        else:
            v_hand_l = v_fo_l
    else:
        v_hand_l = v_fo_l

    # Wrist flexion / extension / deviation: orientation change from forearm to hand
    q_bend_l = rotation_vector_to_quaternion(v_fo_l, v_hand_l)
    q_wr_base_l = quaternion_multiply(q_bend_l, world_rotations[18])

    # Hand pronation / supination (twist around hand axis) with jitter suppression
    if p_idx_l is not None and p_pnk_l is not None and np.linalg.norm(p_pnk_l - p_idx_l) > 0.035:
        v_lat_l = normalize_vector(p_pnk_l - p_idx_l)
        n_hand_l = np.cross(v_hand_l, v_lat_l)
        if np.linalg.norm(n_hand_l) > 0.02:
            n_hand_l = normalize_vector(n_hand_l)
            # In SMPL-X T-Pose (palms down), dorsal normal points up (+Y). Use +Y as canonical rest reference.
            v_ref_dorsal = quat_rotate_vector(q_wr_base_l, np.array([0.0, 1.0, 0.0]))
            v_ref_proj = v_ref_dorsal - np.dot(v_ref_dorsal, v_hand_l) * v_hand_l
            n_hand_proj = n_hand_l - np.dot(n_hand_l, v_hand_l) * v_hand_l
            if np.linalg.norm(v_ref_proj) > 1e-4 and np.linalg.norm(n_hand_proj) > 1e-4:
                v_ref_proj = normalize_vector(v_ref_proj)
                n_hand_proj = normalize_vector(n_hand_proj)
                dot_twist = float(np.clip(np.dot(v_ref_proj, n_hand_proj), -1.0, 1.0))
                cross_twist = float(np.dot(np.cross(v_ref_proj, n_hand_proj), v_hand_l))
                raw_twist = math.atan2(cross_twist, dot_twist)
                raw_twist = float(np.clip(raw_twist, -np.radians(85.0), np.radians(85.0)))
                # Temporal exponential moving average to eliminate fluttering jitter
                twist_l_out = (0.60 * raw_twist + 0.40 * prev_twist_l) if prev_twist_l is not None else raw_twist
                half_twist = twist_l_out * 0.5
                st = math.sin(half_twist)
                ct = math.cos(half_twist)
                q_twist_l = np.array([v_hand_l[0] * st, v_hand_l[1] * st, v_hand_l[2] * st, ct], dtype=np.float64)
                world_rotations[20] = quaternion_multiply(q_twist_l, q_wr_base_l)
            else:
                twist_l_out = prev_twist_l if prev_twist_l is not None else 0.0
                world_rotations[20] = q_wr_base_l
        else:
            twist_l_out = prev_twist_l if prev_twist_l is not None else 0.0
            world_rotations[20] = q_wr_base_l
    else:
        twist_l_out = prev_twist_l if prev_twist_l is not None else 0.0
        world_rotations[20] = q_wr_base_l

    # --- Right Arm ---
    p_sh_r = joints_3d[17]
    p_el_r = joints_3d[19]
    p_wr_r = joints_3d[21]

    v_up_r = normalize_vector(p_el_r - p_sh_r)
    v_fo_r = normalize_vector(p_wr_r - p_el_r)

    # Right Arm SMPL-X rest has local hinge = +Y, bone = -X. To bend forward (+Z), Y column must align with +n.
    n_pole_r = np.cross(v_up_r, v_fo_r)
    if np.linalg.norm(n_pole_r) < 0.06:
        n_pole_r = np.cross(v_up_r, z_chest)
        if np.linalg.norm(n_pole_r) < 0.02:
            n_pole_r = np.array([0.0, 1.0, 0.0])
    n_pole_r = normalize_vector(n_pole_r)

    # Right Upper Arm: rests at [-1, 0, 0]
    x_arm_r = -v_up_r
    y_arm_r = n_pole_r
    z_arm_r = normalize_vector(np.cross(x_arm_r, y_arm_r))
    y_arm_r = normalize_vector(np.cross(z_arm_r, x_arm_r))
    R_arm_r = np.column_stack([x_arm_r, y_arm_r, z_arm_r])
    world_rotations[17] = matrix_to_quaternion(R_arm_r)

    # Right Elbow: strictly 1-DOF hinge around [0, 1, 0] (local Y axis)
    dot_el_r = float(np.clip(np.dot(v_up_r, v_fo_r), -1.0, 1.0))
    angle_el_r = float(np.arccos(dot_el_r))
    angle_el_r = float(np.clip(angle_el_r, 0.0, np.radians(150.0)))
    half_el_r = angle_el_r * 0.5
    s_r = np.sin(half_el_r)
    c_r = np.cos(half_el_r)
    q_el_local_r = np.array([0.0, s_r, 0.0, c_r], dtype=np.float64)
    world_rotations[19] = quaternion_multiply(world_rotations[17], q_el_local_r)

    # --- Right Wrist (Joint 21) Orientation Estimation ---
    p_idx_r = hand_landmarks.get("r_index", None) if hand_landmarks else None
    p_pnk_r = hand_landmarks.get("r_pinky", None) if hand_landmarks else None

    if p_idx_r is not None and p_pnk_r is not None:
        p_hand_center_r = 0.5 * (p_idx_r + p_pnk_r)
        v_hand_raw_r = p_hand_center_r - p_wr_r
        if np.linalg.norm(v_hand_raw_r) > 0.02:
            v_hand_r = normalize_vector(v_hand_raw_r)
        else:
            v_hand_r = v_fo_r
    else:
        v_hand_r = v_fo_r

    q_bend_r = rotation_vector_to_quaternion(v_fo_r, v_hand_r)
    q_wr_base_r = quaternion_multiply(q_bend_r, world_rotations[19])

    if p_idx_r is not None and p_pnk_r is not None and np.linalg.norm(p_pnk_r - p_idx_r) > 0.035:
        v_lat_r = normalize_vector(p_pnk_r - p_idx_r)
        n_hand_r = np.cross(v_lat_r, v_hand_r)
        if np.linalg.norm(n_hand_r) > 0.02:
            n_hand_r = normalize_vector(n_hand_r)
            # In SMPL-X T-Pose (palms down), dorsal normal points up (+Y). Use +Y as canonical rest reference.
            v_ref_dorsal = quat_rotate_vector(q_wr_base_r, np.array([0.0, 1.0, 0.0]))
            v_ref_proj = v_ref_dorsal - np.dot(v_ref_dorsal, v_hand_r) * v_hand_r
            n_hand_proj = n_hand_r - np.dot(n_hand_r, v_hand_r) * v_hand_r
            if np.linalg.norm(v_ref_proj) > 1e-4 and np.linalg.norm(n_hand_proj) > 1e-4:
                v_ref_proj = normalize_vector(v_ref_proj)
                n_hand_proj = normalize_vector(n_hand_proj)
                dot_twist = float(np.clip(np.dot(v_ref_proj, n_hand_proj), -1.0, 1.0))
                cross_twist = float(np.dot(np.cross(v_ref_proj, n_hand_proj), v_hand_r))
                raw_twist = math.atan2(cross_twist, dot_twist)
                raw_twist = float(np.clip(raw_twist, -np.radians(85.0), np.radians(85.0)))
                twist_r_out = (0.60 * raw_twist + 0.40 * prev_twist_r) if prev_twist_r is not None else raw_twist
                half_twist = twist_r_out * 0.5
                st = math.sin(half_twist)
                ct = math.cos(half_twist)
                q_twist_r = np.array([v_hand_r[0] * st, v_hand_r[1] * st, v_hand_r[2] * st, ct], dtype=np.float64)
                world_rotations[21] = quaternion_multiply(q_twist_r, q_wr_base_r)
            else:
                twist_r_out = prev_twist_r if prev_twist_r is not None else 0.0
                world_rotations[21] = q_wr_base_r
        else:
            twist_r_out = prev_twist_r if prev_twist_r is not None else 0.0
            world_rotations[21] = q_wr_base_r
    else:
        twist_r_out = prev_twist_r if prev_twist_r is not None else 0.0
        world_rotations[21] = q_wr_base_r

    # 6. Anatomical Pole-Vector 2-Bone IK for Legs
    # Standard SMPL-X T-Pose:
    # Legs extend along -Y ([0, -1, 0]), bend direction is -Z (Backward), hinge axis is [1, 0, 0] (local X)

    # --- Left Leg ---
    p_hip_l = joints_3d[1]
    p_knee_l = joints_3d[4]
    p_ank_l = joints_3d[7]

    v_up_ll = normalize_vector(p_knee_l - p_hip_l)
    v_fo_ll = normalize_vector(p_ank_l - p_knee_l)

    # Knee bend plane: flexion moves backward (-Z)
    n_pole_ll = np.cross(v_up_ll, v_fo_ll)
    if np.linalg.norm(n_pole_ll) < 0.06:
        n_pole_ll = np.array([1.0, 0.0, 0.0])
    n_pole_ll = normalize_vector(n_pole_ll)

    y_leg_l = -v_up_ll
    x_leg_l = n_pole_ll
    z_leg_l = normalize_vector(np.cross(x_leg_l, y_leg_l))
    x_leg_l = normalize_vector(np.cross(y_leg_l, z_leg_l))
    R_leg_l = np.column_stack([x_leg_l, y_leg_l, z_leg_l])
    world_rotations[1] = matrix_to_quaternion(R_leg_l)

    # Left Knee: 1-DOF hinge around [1, 0, 0] (local X axis)
    dot_kn_l = float(np.clip(np.dot(v_up_ll, v_fo_ll), -1.0, 1.0))
    angle_kn_l = float(np.arccos(dot_kn_l))
    angle_kn_l = float(np.clip(angle_kn_l, 0.0, np.radians(150.0)))
    half_kn_l = angle_kn_l * 0.5
    s_kn_l = np.sin(half_kn_l)
    c_kn_l = np.cos(half_kn_l)
    q_knee_local_l = np.array([s_kn_l, 0.0, 0.0, c_kn_l], dtype=np.float64)
    world_rotations[4] = quaternion_multiply(world_rotations[1], q_knee_local_l)
    world_rotations[7] = world_rotations[4]
    world_rotations[10] = world_rotations[7]

    # --- Right Leg ---
    p_hip_r = joints_3d[2]
    p_knee_r = joints_3d[5]
    p_ank_r = joints_3d[8]

    v_up_rl = normalize_vector(p_knee_r - p_hip_r)
    v_fo_rl = normalize_vector(p_ank_r - p_knee_r)

    n_pole_rl = np.cross(v_up_rl, v_fo_rl)
    if np.linalg.norm(n_pole_rl) < 0.06:
        n_pole_rl = np.array([1.0, 0.0, 0.0])
    n_pole_rl = normalize_vector(n_pole_rl)

    y_leg_r = -v_up_rl
    x_leg_r = n_pole_rl
    z_leg_r = normalize_vector(np.cross(x_leg_r, y_leg_r))
    x_leg_r = normalize_vector(np.cross(y_leg_r, z_leg_r))
    R_leg_r = np.column_stack([x_leg_r, y_leg_r, z_leg_r])
    world_rotations[2] = matrix_to_quaternion(R_leg_r)

    # Right Knee: 1-DOF hinge around [1, 0, 0] (local X axis)
    dot_kn_r = float(np.clip(np.dot(v_up_rl, v_fo_rl), -1.0, 1.0))
    angle_kn_r = float(np.arccos(dot_kn_r))
    angle_kn_r = float(np.clip(angle_kn_r, 0.0, np.radians(150.0)))
    half_kn_r = angle_kn_r * 0.5
    s_kn_r = np.sin(half_kn_r)
    c_kn_r = np.cos(half_kn_r)
    q_knee_local_r = np.array([s_kn_r, 0.0, 0.0, c_kn_r], dtype=np.float64)
    world_rotations[5] = quaternion_multiply(world_rotations[2], q_knee_local_r)
    world_rotations[8] = world_rotations[5]
    world_rotations[11] = world_rotations[8]

    # Convert World Rotations to Local Rotations via Parent Hierarchy with Canonical Hemisphere Alignment
    local_rotations = np.zeros((SMPLX_JOINT_COUNT, 4), dtype=np.float64)
    for j in range(SMPLX_JOINT_COUNT):
        p = SMPLX_PARENTS[j]
        if p == -1:
            local_rotations[j] = world_rotations[j]
        else:
            q_parent = world_rotations[p]
            q_child = world_rotations[j]
            # Shortest arc / canonical hemisphere alignment (q ~ -q)
            if np.dot(q_parent, q_child) < 0.0:
                q_child = -q_child
                world_rotations[j] = q_child
            q_parent_inv = quaternion_inverse(q_parent)
            local_rotations[j] = quaternion_multiply(q_parent_inv, q_child)

        # Ensure w >= 0 canonical representation for local quaternion
        if local_rotations[j, 3] < 0.0:
            local_rotations[j] = -local_rotations[j]

    # Enforce anatomical joint limits to prevent unnatural poses or dislocations
    local_rotations = apply_anatomical_joint_limits(local_rotations)

    return local_rotations, q_pelvis, (twist_l_out, twist_r_out)



# ============================================================================
# Foot Contact Detection and Floor Snapping (Foot Locking)
# ============================================================================

def apply_foot_locking(
    joints_seq: np.ndarray,
    fps: float,
    height_thresh: float = 0.08,
    vel_thresh: float = 0.45,
    return_contact_track: bool = False
) -> Union[np.ndarray, Tuple[np.ndarray, Dict[str, Any]]]:
    """
    Detects stance/contact phases for left and right feet and snaps the feet to the floor
    during the contact phase to eliminate foot sliding.
    Then analytically adjusts the knee positions via 2-bone IK.
    If return_contact_track is True, returns (joints, contact_track_dict).
    """
    empty_contact_track = {"version": 1, "intervals": []}
    T = joints_seq.shape[0]
    if T < 4:
        return (joints_seq, empty_contact_track) if return_contact_track else joints_seq

    result_joints = np.copy(joints_seq)

    # Detect per-frame inverted poses (head below pelvis by > 0.05m)
    # Head=15, Neck=12, Pelvis=0
    is_inverted_mask = np.zeros(T, dtype=bool)
    for t in range(T):
        head_y = joints_seq[t, 15, 1]
        neck_y = joints_seq[t, 12, 1]
        pelvis_y = joints_seq[t, 0, 1]
        if max(head_y, neck_y) < pelvis_y - 0.05:
            is_inverted_mask[t] = True

    # 1. Estimate floor level from feet vertical positions ONLY during upright frames
    upright_indices = np.where(~is_inverted_mask)[0]
    if len(upright_indices) == 0:
        # Video is entirely inverted (handstand/freeze) - bypass foot locking completely
        return (joints_seq, empty_contact_track) if return_contact_track else joints_seq

    upright_feet_y = np.concatenate([
        result_joints[upright_indices, 7, 1],   # L_Ankle Y
        result_joints[upright_indices, 8, 1],   # R_Ankle Y
        result_joints[upright_indices, 10, 1],  # L_Foot Y
        result_joints[upright_indices, 11, 1],  # R_Foot Y
    ])
    floor_y = float(np.percentile(upright_feet_y, 5.0))
    ankle_offset = 0.06  # typical nominal ankle height above floor

    all_contact_intervals = []

    # 2. Compute velocities
    for leg_idx, (hip_i, knee_i, ankle_i, foot_i) in enumerate([(1, 4, 7, 10), (2, 5, 8, 11)]):
        foot_name = "left" if leg_idx == 0 else "right"
        # Make explicit copies of original positions to avoid view-overwrite bugs
        orig_ankles = np.copy(result_joints[:, ankle_i, :])
        orig_toes = np.copy(result_joints[:, foot_i, :])

        vel = np.zeros(T)
        for t in range(1, T):
            vel[t] = np.linalg.norm(orig_ankles[t] - orig_ankles[t - 1]) * fps
        vel[0] = vel[1]

        # Contact detection with guards:
        # Foot cannot contact floor if body is inverted, or if ankle is airborne/above hip
        heights = orig_ankles[:, 1] - floor_y
        is_above_pelvis = orig_ankles[:, 1] > (result_joints[:, 0, 1] - 0.15)
        contact_mask = (heights < height_thresh) & (vel < vel_thresh) & (~is_inverted_mask) & (~is_above_pelvis)

        # Find continuous contact segments
        segments = []
        in_segment = False
        start_t = 0
        for t in range(T):
            if contact_mask[t] and not in_segment:
                in_segment = True
                start_t = t
            elif not contact_mask[t] and in_segment:
                in_segment = False
                if t - start_t >= 3:
                    segments.append((start_t, t - 1))
        if in_segment and T - start_t >= 3:
            segments.append((start_t, T - 1))

        # Collect contact track intervals
        for seg_start, seg_end in segments:
            anchor_pos = np.median(orig_ankles[seg_start:seg_end + 1], axis=0)
            anchor_pos[1] = floor_y + ankle_offset
            seg_vel = vel[seg_start:seg_end + 1]
            conf = float(np.clip(1.0 - (np.mean(seg_vel) / max(vel_thresh, 1e-5)) * 0.2, 0.5, 0.99))

            # Mode detection
            toe_y_mean = np.mean(orig_toes[seg_start:seg_end + 1, 1])
            ankle_y_mean = np.mean(orig_ankles[seg_start:seg_end + 1, 1])
            if toe_y_mean - ankle_y_mean < -0.04:
                mode = "toe"
            elif ankle_y_mean - toe_y_mean < -0.04:
                mode = "heel"
            else:
                mode = "flat"

            all_contact_intervals.append({
                "foot": foot_name,
                "start": round(float(seg_start / fps), 3),
                "end": round(float(seg_end / fps), 3),
                "mode": mode,
                "confidence": round(conf, 2),
                "anchor": [round(float(anchor_pos[0]), 3), round(float(anchor_pos[1]), 3), round(float(anchor_pos[2]), 3)]
            })

        # Apply locking and 2-bone IK adjustment
        for seg_start, seg_end in segments:
            # Anchor position
            anchor_pos = np.median(orig_ankles[seg_start:seg_end + 1], axis=0)
            anchor_pos[1] = floor_y + ankle_offset

            seg_len = seg_end - seg_start + 1
            for t in range(seg_start, seg_end + 1):
                # Do not snap crossing feet to straight standing anchor
                dist_between_ankles = float(np.linalg.norm(result_joints[t, 7] - result_joints[t, 8]))
                if dist_between_ankles < 0.15:
                    continue

                # Blend factor at boundaries to prevent popping
                blend = 1.0
                if t == seg_start and seg_len > 3:
                    blend = 0.5
                elif t == seg_end and seg_len > 3:
                    blend = 0.5

                locked_ankle = (1.0 - blend) * orig_ankles[t] + blend * anchor_pos
                result_joints[t, ankle_i] = locked_ankle

                # Adjust foot toe position using preserved original offset
                toe_offset = orig_toes[t] - orig_ankles[t]
                result_joints[t, foot_i] = locked_ankle + toe_offset
                result_joints[t, foot_i, 1] = max(result_joints[t, foot_i, 1], floor_y)

                # Solve 2-bone IK for knee preserving original bone lengths
                p_hip = result_joints[t, hip_i]
                p_knee = result_joints[t, knee_i]
                l1 = float(np.linalg.norm(p_knee - p_hip))
                l2 = float(np.linalg.norm(orig_ankles[t] - p_knee))
                if l1 > 0.05 and l2 > 0.05:
                    bend_dir = np.array([0.0, 0.0, 1.0])
                    solved_knee = solve_two_bone_ik(p_hip, p_knee, locked_ankle, l1, l2, bend_dir)
                    result_joints[t, knee_i] = solved_knee

    all_contact_intervals.sort(key=lambda x: (x["start"], x["foot"]))
    contact_track = {
        "version": 1,
        "intervals": all_contact_intervals
    }

    if return_contact_track:
        return result_joints, contact_track
    return result_joints


# ============================================================================
# In-Place Filtering & Temporal Smoothing
# ============================================================================

def apply_in_place_filtering(root_positions: np.ndarray) -> np.ndarray:
    """
    Zeroes horizontal X and Z root motion while preserving natural vertical Y bounce.
    """
    filtered = np.copy(root_positions)
    filtered[:, 0] = 0.0  # Zero X
    filtered[:, 2] = 0.0  # Zero Z
    return filtered


def clamp_quaternion_angular_speed(
    quats: np.ndarray,
    fps: float,
    max_deg_per_sec: float = 720.0
) -> np.ndarray:
    """
    Clamps the angular speed between consecutive quaternion frames
    to prevent physically impossible instantaneous popping or snapping.
    Uses Slerp interpolation when angular delta exceeds max_deg_per_sec.
    """
    T = quats.shape[0]
    out = np.copy(quats)
    max_angle_per_frame_rad = (max_deg_per_sec / max(1.0, fps)) * (np.pi / 180.0)

    for t in range(1, T):
        q_prev = out[t - 1]
        q_curr = out[t]

        dot = np.dot(q_prev, q_curr)
        if dot < 0.0:
            q_curr = -q_curr
            out[t] = q_curr
            dot = -dot

        dot = np.clip(dot, -1.0, 1.0)
        angle = 2.0 * np.arccos(dot)

        if angle > max_angle_per_frame_rad and angle > 1e-4:
            slerp_t = max_angle_per_frame_rad / angle
            sin_ang = np.sin(angle * 0.5)
            if sin_ang > 1e-4:
                w1 = np.sin((1.0 - slerp_t) * angle * 0.5) / sin_ang
                w2 = np.sin(slerp_t * angle * 0.5) / sin_ang
                q_clamped = w1 * q_prev + w2 * q_curr
                out[t] = q_clamped / np.linalg.norm(q_clamped)
            else:
                out[t] = q_prev

    return out


def apply_temporal_smoothing(
    root_positions: np.ndarray,
    local_rotations: np.ndarray,
    fps: float = 30.0,
    window_length: int = 7,
    polyorder: int = 2
) -> tuple:
    """
    Applies Savitzky-Golay filtering to root positions and quaternion rotations
    with quaternion continuity enforcement and biomechanical angular speed clamping.
    """
    T = root_positions.shape[0]
    if T <= window_length:
        window_length = T if T % 2 == 1 else max(3, T - 1)
    if window_length <= polyorder:
        return root_positions, local_rotations

    # 1. Smooth root positions
    smooth_pos = np.copy(root_positions)
    if SCIPY_AVAILABLE:
        smooth_pos = savgol_filter(root_positions, window_length, polyorder, axis=0)
    else:
        # Simple moving average fallback
        kernel = np.ones(window_length) / window_length
        for c in range(3):
            smooth_pos[:, c] = np.convolve(root_positions[:, c], kernel, mode='same')

    # 2. Smooth local rotations
    smooth_rot = np.copy(local_rotations)
    for j in range(SMPLX_JOINT_COUNT):
        quats = np.copy(local_rotations[:, j, :])
        # Enforce quaternion continuity along time
        for t in range(1, T):
            if np.dot(quats[t], quats[t - 1]) < 0.0:
                quats[t] = -quats[t]

        if SCIPY_AVAILABLE:
            q_filtered = savgol_filter(quats, window_length, polyorder, axis=0)
        else:
            kernel = np.ones(window_length) / window_length
            q_filtered = np.zeros_like(quats)
            for c in range(4):
                q_filtered[:, c] = np.convolve(quats[:, c], kernel, mode='same')

        # Renormalize quaternions
        norms = np.linalg.norm(q_filtered, axis=1, keepdims=True)
        norms = np.where(norms < 1e-8, 1.0, norms)
        renorm_quats = q_filtered / norms

        # Apply biomechanical angular speed clamping (eliminates remaining sudden jumps)
        smooth_rot[:, j, :] = clamp_quaternion_angular_speed(renorm_quats, fps=fps, max_deg_per_sec=720.0)

    return smooth_pos, smooth_rot


# ============================================================================
# Synthetic Kinematic Fallback Generator
# ============================================================================

def generate_synthetic_motion(
    frames: int = 60,
    fps: float = 30.0,
    in_place: bool = True
) -> tuple:
    """
    Generates a natural walking kinematic pose sequence for testing and standalone validation.
    Returns: (root_positions, local_rotations)
    """
    root_positions = np.zeros((frames, 3), dtype=np.float64)
    local_rotations = np.zeros((frames, SMPLX_JOINT_COUNT, 4), dtype=np.float64)

    for t in range(frames):
        phase = 2.0 * math.pi * (t / fps) * 1.0  # 1.0 Hz gait
        # Root bounce
        root_y = 0.95 + 0.03 * math.sin(phase * 2.0)
        root_x = 0.0 if in_place else 0.02 * math.sin(phase)
        root_z = 0.0 if in_place else (t / fps) * 1.2
        root_positions[t] = np.array([root_x, root_y, root_z])

        # All joints default to identity
        for j in range(SMPLX_JOINT_COUNT):
            local_rotations[t, j] = np.array([0.0, 0.0, 0.0, 1.0])

        # Leg swings
        l_hip_pitch = 0.35 * math.sin(phase)
        r_hip_pitch = -0.35 * math.sin(phase)
        l_knee_bend = max(0.0, 0.5 * math.sin(phase + math.pi * 0.3))
        r_knee_bend = max(0.0, 0.5 * math.sin(phase + math.pi * 1.3))

        # L_Hip (1) & R_Hip (2)
        local_rotations[t, 1] = np.array([math.sin(l_hip_pitch / 2), 0.0, 0.0, math.cos(l_hip_pitch / 2)])
        local_rotations[t, 2] = np.array([math.sin(r_hip_pitch / 2), 0.0, 0.0, math.cos(r_hip_pitch / 2)])

        # L_Knee (4) & R_Knee (5)
        local_rotations[t, 4] = np.array([math.sin(l_knee_bend / 2), 0.0, 0.0, math.cos(l_knee_bend / 2)])
        local_rotations[t, 5] = np.array([math.sin(r_knee_bend / 2), 0.0, 0.0, math.cos(r_knee_bend / 2)])

        # Arm swings (opposite to legs)
        l_arm_swing = -0.25 * math.sin(phase)
        r_arm_swing = 0.25 * math.sin(phase)
        local_rotations[t, 16] = np.array([0.0, math.sin(l_arm_swing / 2), 0.0, math.cos(l_arm_swing / 2)])
        local_rotations[t, 17] = np.array([0.0, math.sin(r_arm_swing / 2), 0.0, math.cos(r_arm_swing / 2)])

        # Wrist swings (natural subtle wrist flexion / extension matching arm swing)
        l_wrist_pitch = 0.12 * math.sin(phase)
        r_wrist_pitch = -0.12 * math.sin(phase)
        local_rotations[t, 20] = np.array([math.sin(l_wrist_pitch / 2), 0.0, 0.0, math.cos(l_wrist_pitch / 2)])
        local_rotations[t, 21] = np.array([math.sin(r_wrist_pitch / 2), 0.0, 0.0, math.cos(r_wrist_pitch / 2)])

    local_rotations = apply_anatomical_joint_limits(local_rotations)

    return root_positions, local_rotations


# ============================================================================
# Main Extraction Pipeline
# ============================================================================

def process_video(
    video_path: str,
    output_path: str,
    target_fps: float = None,
    trim_start: float = 0.0,
    trim_end: float = 0.0,
    in_place: bool = False,
    smooth: bool = False,
    foot_lock: bool = True,
    model_complexity: int = 1,
    min_detection_confidence: float = 0.5,
    min_tracking_confidence: float = 0.5,
    overlay_video_path: str = None,
    backend: str = "auto",
    pytorch_model_path: str = None,
    pytorch_adapter_module: str = None,
    pytorch_device: str = "auto",
    max_sequence_frames: int = 0,
    video_model_directory: str = None,
    hmr2_runtime_path: str = None,
    hmr2_body_model_path: str = None,
    rtmpose_model_path: str = None,
    wham_asset_manifest_path: str = None,
    wham_body_model_path: str = None,
    wham_image_feature_backbone_path: str = None,
    wham_image_feature_model_definition_path: str = None,
    wham_image_feature_config_path: str = None,
    wham_image_feature_path: str = None,
    wham_camera_model_path: str = None,
    wham_dpvo_model_path: str = None,
    wham_preprocess_directory: str = None,
    fusion_mode: str = FUSION_MODE_OFF,
    fusion_config: dict = None,
    overlay_mode: str = "dual",
) -> dict:
    """
    Full pipeline: reads video, extracts landmarks, computes 2-bone IK rotations,
    generates a detector-specific pose overlay video, applies foot locking and smoothing,
    and exports to output_path and overlay_video_path.
    """
    emit_progress(0.02, "init", 0, 100)

    requested_backend = str(backend or "auto").strip().lower() or "auto"
    fusion_mode = normalize_fusion_mode(fusion_mode)
    # Selecting the explicit fusion mode implies WHAM as the primary when no
    # legacy backend was selected.  Explicit MediaPipe/RTMPose requests remain
    # untouched for backwards compatibility.
    if fusion_mode == FUSION_MODE_WHAM_MEDIAPIPE and requested_backend == "auto":
        requested_backend = "wham"
    # WHAM quality extraction is inherently a multi-observation path.  Keep
    # direct Python/CLI callers consistent with the editor (which passes the
    # explicit fusion mode) and do not silently retarget a raw WHAM stream.
    if requested_backend == "wham" and fusion_mode == FUSION_MODE_OFF:
        fusion_mode = FUSION_MODE_WHAM_MEDIAPIPE
    quality_backend_requested = requested_backend in ("pytorch", "wham", "hmr2", "hybrik")

    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Input video file not found: {video_path}")

    if not OPENCV_AVAILABLE:
        raise RuntimeError("OpenCV (cv2) is required for video reading.")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video file: {video_path}")

    orig_fps = cap.get(cv2.CAP_PROP_FPS)
    if orig_fps <= 0.0 or math.isnan(orig_fps):
        orig_fps = 30.0

    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_video_frames / orig_fps

    start_sec = max(0.0, trim_start)
    end_sec = duration if trim_end <= 0.0 or trim_end > duration else trim_end
    if end_sec <= start_sec:
        end_sec = duration

    fps_out = float(target_fps) if target_fps and target_fps > 0.0 else float(orig_fps)

    # Compute target sample timestamps from an integer sample count.  Repeated
    # floating-point ``curr_t += dt`` accumulation can step one frame past the
    # clip duration (for example, a 164-frame/30-FPS clip became 165 samples),
    # which then asks the decoder for a non-existent frame and retimes the
    # final pose.  Subtract a tiny epsilon so an exact N/FPS duration remains
    # exactly N samples while genuinely partial durations still round up.
    sample_duration = max(0.0, float(end_sec - start_sec))
    sample_count = max(
        1,
        int(math.ceil(sample_duration * float(fps_out) - 1.0e-9)),
    )
    sample_times = [
        float(start_sec + (index / float(fps_out)))
        for index in range(sample_count)
    ]

    total_sampled_frames = len(sample_times)
    if total_sampled_frames <= 0:
        raise ValueError(f"No frames to extract in time range [{start_sec:.2f}s, {end_sec:.2f}s]")

    cap_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cap_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    landmarks_seq = []
    norm_landmarks_seq = []
    landmark_coordinate_systems = []
    frame_confidences = []
    last_valid_landmarks = None
    last_valid_norm_landmarks = None
    last_landmark_coordinate_system = "legacy_overlay"
    last_quality_overlay_landmarks = None
    quality_overlay_coordinate_system = "smplx"
    fusion_result = None
    fusion_observation_rows = []
    fusion_rtmpose_rows = []
    fusion_intervals = []
    last_video_frame = None
    backend_metadata_snapshot = {}
    quality_frame_fallbacks = []
    quality_frame_fallback_reasons = []
    quality_alignment_warnings = []

    # Resolve overlay video path
    if not overlay_video_path:
        out_base = os.path.splitext(output_path)[0]
        overlay_video_path = f"{out_base}_overlay.mp4"

    out_video_dir = os.path.dirname(os.path.abspath(overlay_video_path))
    if out_video_dir and not os.path.exists(out_video_dir):
        os.makedirs(out_video_dir, exist_ok=True)

    # Initialize VideoWriter for overlay video (mp4v codec)
    overlay_writer = None
    try:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        overlay_writer = cv2.VideoWriter(overlay_video_path, fourcc, fps_out, (cap_width, cap_height))
    except Exception as e:
        sys.stderr.write(f"[TexMotion] Warning: Could not initialize overlay VideoWriter: {e}\n")
        overlay_writer = None

    # Initialize Pose Tracker.  Explicit quality requests may fall back to the
    # lightweight MediaPipe path, but the factory records the requested backend
    # and concrete failure reason so the exported result and editor stay
    # auditable.
    pose_tracker = None
    use_pipeline_backend = False
    backend_initialization_error = None
    if POSE_PIPELINE_AVAILABLE:
        try:
            pose_tracker = create_pose_backend(
                preferred=backend,
                model_path=rtmpose_model_path,
                model_directory=video_model_directory,
                model_complexity=model_complexity,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
                pytorch_model_path=pytorch_model_path,
                pytorch_adapter_module=pytorch_adapter_module,
                pytorch_device=pytorch_device,
                max_sequence_frames=max_sequence_frames,
                runtime_path=hmr2_runtime_path,
                body_model_path=hmr2_body_model_path or wham_body_model_path,
                asset_manifest_path=wham_asset_manifest_path,
                image_feature_backbone_path=wham_image_feature_backbone_path,
                image_feature_model_definition=wham_image_feature_model_definition_path,
                image_feature_config_path=wham_image_feature_config_path,
                image_feature_path=wham_image_feature_path,
                camera_model_path=wham_camera_model_path,
                dpvo_model_path=wham_dpvo_model_path,
                wham_preprocess_directory=(
                    wham_preprocess_directory
                    or os.path.join(
                        os.path.dirname(os.path.abspath(output_path)),
                        "." + Path(output_path).stem + "_wham_preprocess",
                    )
                ),
                # Keep the legacy automatic fallback, but retain provenance
                # on the returned MediaPipe backend for diagnostics.
                strict_quality=False,
            )
            use_pipeline_backend = True
            sys.stderr.write(f"[TexMotion] Initialized modular pose backend: {pose_tracker.name}\n")
            sys.stderr.flush()
        except Exception as e:
            backend_initialization_error = str(e)
            sys.stderr.write(f"[TexMotion Warning] Modular backend initialization failed: {e}. Falling back to MediaPipePoseTracker.\n")
            pose_tracker = None

    quality_backend_active = POSE_PIPELINE_AVAILABLE and isinstance(pose_tracker, PyTorchPoseBackend)
    # Keep the original quality owner for cleanup even if a later safety
    # fallback promotes an auxiliary detector to the primary tracker.
    quality_backend_owner = pose_tracker if quality_backend_active else None

    if pose_tracker is None:
        if not MEDIAPIPE_AVAILABLE:
            raise RuntimeError("MediaPipe is not installed or supported in this Python environment. Please install it using: pip install mediapipe opencv-python")

        pose_tracker = MediaPipePoseTracker(
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            model_directory=video_model_directory,
            allow_download=False,
        )
        if quality_backend_requested:
            setattr(pose_tracker, "fallback_from", requested_backend)
            setattr(
                pose_tracker,
                "fallback_reason",
                (
                    f"Modular {requested_backend.upper()} backend initialization failed: "
                    f"{backend_initialization_error}"
                    if backend_initialization_error
                    else (
                        "The modular quality backend could not be imported in this Python "
                        f"environment: {POSE_PIPELINE_IMPORT_ERROR}"
                        if POSE_PIPELINE_IMPORT_ERROR
                        else "The modular quality backend could not be imported in this Python environment."
                    )
                ),
            )

    # Auxiliary 3D tracker when using 2D backend (RTMPose) for Hybrid Observation Fusion
    aux_mediapipe = None
    if use_pipeline_backend and pose_tracker and pose_tracker.name == "rtmpose" and MEDIAPIPE_AVAILABLE:
        try:
            aux_mediapipe = MediaPipePoseTracker(
                model_complexity=model_complexity,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
                model_directory=video_model_directory,
                allow_download=False,
            )
            sys.stderr.write("[TexMotion] Enabled Hybrid Fusion: RTMPose 2D Precision + MediaPipe 3D Spatial Auxiliary.\n")
            sys.stderr.flush()
        except Exception as e:
            sys.stderr.write(f"[TexMotion Warning] Could not initialize auxiliary MediaPipe: {e}\n")
            aux_mediapipe = None

    # A quality adapter can combine its temporal learned 3D hypothesis with
    # MediaPipe's cheap 3D observation.  Keep this auxiliary path explicit and
    # never let it overwrite a valid learned body-model result.  RTMPose is
    # similarly available as an optional independent 2D observation when its
    # local ONNX weights are already installed (no download is triggered).
    aux_rtmpose = None
    if quality_backend_active and MEDIAPIPE_AVAILABLE:
        quality_input_detector = ""
        try:
            shared_detector = None
            get_shared_detector = getattr(pose_tracker, "get_mediapipe_auxiliary", None)
            if callable(get_shared_detector):
                shared_detector = get_shared_detector()
            if shared_detector is not None:
                aux_mediapipe = _SharedMediaPipeAuxiliary(shared_detector)
                backend_metadata_snapshot["mediapipeAuxiliaryStatus"] = "shared"
                backend_metadata_snapshot["mediapipeAuxiliaryOwner"] = "quality_backend"
                sys.stderr.write(
                    "[TexMotion] Quality backend auxiliary: reusing WHAM's MediaPipe 3D detector.\n"
                )
            else:
                # A native WHAM adapter commonly owns a MediaPipe graph for
                # its 2D input stream.  If an old/custom bridge does not
                # expose that graph through get_mediapipe_auxiliary(),
                # constructing another Tasks/ XNNPACK graph is unsafe on
                # Windows (0xC0000005) and was the source of intermittent
                # extraction process crashes.  Only allow a standalone
                # auxiliary when the quality backend explicitly reports a
                # non-MediaPipe input detector (for example RTMPose).
                try:
                    quality_metadata = pose_tracker.get_metadata()
                    if isinstance(quality_metadata, dict):
                        quality_input_detector = str(
                            quality_metadata.get("inputDetector", "")
                        ).strip().lower()
                except Exception:
                    quality_input_detector = ""
                if quality_input_detector in ("rtmpose", "rtmpose_2d", "onnx"):
                    aux_mediapipe = MediaPipePoseTracker(
                        model_complexity=model_complexity,
                        min_detection_confidence=min_detection_confidence,
                        min_tracking_confidence=min_tracking_confidence,
                        model_directory=video_model_directory,
                        allow_download=False,
                    )
                    backend_metadata_snapshot["mediapipeAuxiliaryStatus"] = "standalone"
                    backend_metadata_snapshot["mediapipeAuxiliaryOwner"] = "extractor"
                    sys.stderr.write(
                        "[TexMotion] Quality backend auxiliary: MediaPipe 3D observations enabled "
                        "(WHAM input detector is RTMPose).\n"
                    )
                else:
                    # Keep WHAM's one native graph alive and let the temporal
                    # hypothesis run without optional MediaPipe fusion.  The
                    # exported metadata makes this decision explicit so the
                    # editor can explain why the fusion stream was disabled.
                    aux_mediapipe = None
                    backend_metadata_snapshot["mediapipeAuxiliaryStatus"] = (
                        "disabled_duplicate_graph"
                    )
                    backend_metadata_snapshot["mediapipeAuxiliaryOwner"] = "quality_backend"
                    backend_metadata_snapshot["mediapipeAuxiliaryReason"] = (
                        "WHAM owns a MediaPipe graph but the adapter did not expose it; "
                        "a second native graph was disabled to prevent a Windows TFLite/XNNPACK crash."
                    )
                    sys.stderr.write(
                        "[TexMotion Warning] Quality backend auxiliary disabled: WHAM's "
                        "MediaPipe detector was not shareable; a second native graph is unsafe.\n"
                    )
            sys.stderr.flush()
        except Exception as e:
            backend_metadata_snapshot["mediapipeAuxiliaryStatus"] = "unavailable"
            backend_metadata_snapshot["mediapipeAuxiliaryReason"] = str(e)
            sys.stderr.write(f"[TexMotion Warning] Could not initialize quality MediaPipe auxiliary: {e}\n")
            # If the bridge failed while exposing its detector, still inspect
            # the declared owner before considering a standalone RTMPose
            # stream below.  This keeps a MediaPipe-owned WHAM graph in the
            # one-graph safety mode even when the sharing hook itself throws.
            try:
                quality_metadata = pose_tracker.get_metadata()
                if isinstance(quality_metadata, dict):
                    quality_input_detector = str(
                        quality_metadata.get("inputDetector", quality_input_detector)
                    ).strip().lower()
            except Exception:
                pass
        # Avoid a second ONNX Runtime session when WHAM already owns the
        # RTMPose detector for its image-side input.  Like MediaPipe/TFLite,
        # duplicate native sessions can terminate the Windows subprocess
        # without a Python traceback.  WHAM's detector remains the primary
        # 2D observation in this case; the result metadata records why the
        # optional duplicate stream was skipped.
        if quality_input_detector in ("rtmpose", "rtmpose_2d", "onnx"):
            backend_metadata_snapshot["rtmposeAuxiliaryStatus"] = (
                "disabled_duplicate_session"
            )
            backend_metadata_snapshot["rtmposeAuxiliaryReason"] = (
                "WHAM already owns the RTMPose input detector; a second ONNX Runtime session was disabled."
            )
            sys.stderr.write(
                "[TexMotion] RTMPose auxiliary disabled: WHAM already owns the RTMPose input detector.\n"
            )
        elif quality_input_detector in (
            "mediapipe", "mediapipe_pose", "mediapipe_3d"
        ):
            try:
                candidate_rtm = RTMPoseBackend(model_path=rtmpose_model_path, model_directory=video_model_directory)
                if candidate_rtm.is_available() and candidate_rtm.initialize():
                    aux_rtmpose = candidate_rtm
                    backend_metadata_snapshot["rtmposeAuxiliaryStatus"] = "standalone"
                    sys.stderr.write("[TexMotion] Quality backend auxiliary: RTMPose 2D observations enabled.\n")
                    sys.stderr.flush()
                else:
                    candidate_rtm.close()
                    backend_metadata_snapshot["rtmposeAuxiliaryStatus"] = "unavailable"
            except Exception as e:
                backend_metadata_snapshot["rtmposeAuxiliaryStatus"] = "unavailable"
                backend_metadata_snapshot["rtmposeAuxiliaryReason"] = str(e)
                sys.stderr.write(f"[TexMotion] RTMPose auxiliary unavailable for quality backend: {e}\n")
        else:
            backend_metadata_snapshot["rtmposeAuxiliaryStatus"] = (
                "disabled_unknown_owner"
            )
            backend_metadata_snapshot["rtmposeAuxiliaryReason"] = (
                "WHAM input detector ownership was not declared; optional RTMPose was disabled for native-session safety."
            )
            sys.stderr.write(
                "[TexMotion] RTMPose auxiliary disabled: WHAM input detector ownership is unknown.\n"
            )

    actual_backend_name = getattr(pose_tracker, "name", "mediapipe") if pose_tracker is not None else "mediapipe"
    fallback_from = getattr(pose_tracker, "fallback_from", None) if pose_tracker is not None else None
    fallback_reason = getattr(pose_tracker, "fallback_reason", None) if pose_tracker is not None else None
    backend_fallback = bool(fallback_reason)
    if quality_backend_active:
        # The overlay is always rendered from this backend's 3D hypothesis.
        # Auxiliary MediaPipe/RTMPose observations are used only as optional
        # camera anchors and are never reported as the overlay source.
        overlay_backend_name = actual_backend_name
        overlay_source = "quality_3d_projection"
    elif aux_mediapipe is not None:
        overlay_backend_name = "rtmpose_hybrid"
        overlay_source = "mediapipe_3d_auxiliary"
    else:
        overlay_backend_name = actual_backend_name
        overlay_source = "backend_2d" if actual_backend_name in ("rtmpose", "mediapipe") else "backend_output"
    if backend_fallback:
        # The concrete overlay is MediaPipe after a fallback.  Keep the badge
        # honest and put the requested backend/reason into the exported
        # metadata and editor diagnostics rather than calling it WHAM.
        overlay_display_name = "MediaPipe Fallback"
    elif quality_backend_active and actual_backend_name == "wham":
        overlay_display_name = "WHAM Temporal 3D"
    elif overlay_backend_name == "rtmpose_hybrid":
        overlay_display_name = "RTMPose Hybrid"
    else:
        overlay_display_name = overlay_backend_name

    # Temporal research adapters (WHAM/HMR2/HybrIK) need frames on both sides
    # of an occlusion.  Decode the sampled sequence once and run their
    # sequence hook before entering the existing overlay/export loop.  The
    # regular MediaPipe/RTMPose paths never allocate this cache.
    sequence_observations = None
    sequence_frames = None
    sequence_inference_error = None
    sequence_decode_missing_frames = []
    wham_initialization_observation = None
    wham_initialization_frame = None
    wham_seed_probe_consumed = False
    if use_pipeline_backend and getattr(pose_tracker, "supports_sequence", False):
        try:
            sequence_frames = []
            for t_sec in sample_times:
                target_frame_num = int(round(t_sec * orig_fps))
                cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame_num)
                ret, sequence_frame = cap.read()
                sequence_frames.append(sequence_frame if ret and sequence_frame is not None else None)

            # Some Windows codecs intermittently fail a random seek even though
            # the same frame can be decoded when the stream is read forward.
            # A missing cached frame used to make the temporal pass return a
            # valid WHAM row while the export loop silently skipped that video
            # frame, producing a shorter overlay and a large pose jump.  Retry
            # only the gaps with one sequential decoder so normal clips retain
            # the inexpensive seek path.
            sequence_decode_missing_frames = [
                index for index, value in enumerate(sequence_frames)
                if value is None
            ]
            if sequence_decode_missing_frames:
                retry_cap = cv2.VideoCapture(video_path)
                try:
                    if retry_cap.isOpened():
                        wanted = {
                            int(round(sample_times[index] * orig_fps)): index
                            for index in sequence_decode_missing_frames
                        }
                        max_target = max(wanted) if wanted else -1
                        source_index = -1
                        while source_index < max_target:
                            ret, retry_frame = retry_cap.read()
                            source_index += 1
                            if not ret or retry_frame is None:
                                break
                            target_index = wanted.get(source_index)
                            if target_index is not None:
                                sequence_frames[target_index] = retry_frame
                        sequence_decode_missing_frames = [
                            index for index, value in enumerate(sequence_frames)
                            if value is None
                        ]
                finally:
                    retry_cap.release()
            if sequence_decode_missing_frames:
                backend_metadata_snapshot["sequenceDecodeMissingFrames"] = list(
                    sequence_decode_missing_frames
                )

            # The released WHAM checkpoint expects the first 3D seed in its
            # camera/image convention.  A neutral embedded SMPL seed has the
            # right tensor shape but no relation to this subject, which can
            # make the recurrent rollout collapse or flip vertically.  Probe
            # the already-initialized MediaPipe auxiliary once before the
            # temporal pass and let the native adapter consume that observation
            # as its image-conditioned initialization.  A short leading
            # detector gap is tolerated by trying subsequent cached frames.
            if (
                quality_backend_active
                and aux_mediapipe is not None
                and hasattr(pose_tracker, "set_initialization_from_observation")
            ):
                for seed_index, seed_frame in enumerate(sequence_frames):
                    if seed_frame is None:
                        continue
                    try:
                        # MediaPipePoseTracker exposes the legacy tuple contract
                        # ``(world, normalized, confidence)`` (the modular
                        # quality backends use the richer FrameObservations
                        # contract).  Keep the seed probe on the MediaPipe
                        # contract and pass milliseconds so Tasks VIDEO mode
                        # receives a monotonic timestamp.
                        wham_seed_probe_consumed = True
                        candidate = aux_mediapipe.detect(
                            seed_frame,
                            int(round(sample_times[seed_index] * 1000.0)),
                        )
                    except Exception as exc:
                        sys.stderr.write(
                            f"[TexMotion] WHAM initialization observation failed at frame "
                            f"{seed_index}: {exc}\n"
                        )
                        candidate = None
                    if candidate is None:
                        continue
                    try:
                        seeded = bool(
                            pose_tracker.set_initialization_from_observation(candidate)
                        )
                    except Exception as exc:
                        seeded = False
                        sys.stderr.write(
                            f"[TexMotion] WHAM initialization seed rejected at frame "
                            f"{seed_index}: {exc}\n"
                        )
                    if seeded:
                        wham_initialization_observation = candidate
                        wham_initialization_frame = int(seed_index)
                        backend_metadata_snapshot["smplInitializationSource"] = (
                            "mediapipe_observation"
                        )
                        backend_metadata_snapshot["smplInitializationFrame"] = int(seed_index)
                        sys.stderr.write(
                            "[TexMotion] WHAM seeded from MediaPipe camera-space 3D "
                            f"observation at frame {seed_index}.\n"
                        )
                        sys.stderr.flush()
                        break
            # MediaPipePoseTracker and MediaPipeBackend both clamp VIDEO
            # timestamps monotonically.  Recreating a native graph after the
            # seed probe is unsafe on Windows (the old close/recreate cycle
            # could terminate the process with 0xC0000005), and is no longer
            # needed for either the shared or standalone auxiliary facade.
            if wham_seed_probe_consumed:
                backend_metadata_snapshot["mediapipeSeedTrackerReset"] = False
            emit_progress(0.04, "wham_sequence", 0, total_sampled_frames)
            sequence_observations = pose_tracker.detect_sequence(
                sequence_frames, sample_times, list(range(total_sampled_frames))
            )
            if not isinstance(sequence_observations, (list, tuple)):
                sequence_observations = None
            else:
                emit_progress(0.08, "wham_sequence", total_sampled_frames, total_sampled_frames)
                sys.stderr.write(
                    f"[TexMotion] Temporal backend returned {len(sequence_observations)} sequence observations.\n"
                )
                sys.stderr.flush()
        except Exception as e:
            sequence_inference_error = str(e)
            sys.stderr.write(f"[TexMotion Warning] Temporal backend sequence inference failed: {e}. Falling back to frame inference.\n")
            sequence_observations = None

    if sequence_inference_error:
        # A frame-wise retry may still succeed, so this is not automatically a
        # backend fallback.  It is nevertheless material provenance: explicit
        # fusion cannot claim to have run after a failed temporal prepass.
        backend_metadata_snapshot["sequenceInferenceError"] = sequence_inference_error

    # A temporal adapter can initialize successfully but fail while loading a
    # checkpoint or executing its sequence method.  When every quality frame
    # is unavailable and the auxiliary MediaPipe stream is ready, switch to
    # that stream explicitly and retain the concrete runtime error for the
    # result card/Timeline Editor.  Partial WHAM drops still use the previous
    # quality hypothesis frame-by-frame and do not trigger a global fallback.
    if quality_backend_active and aux_mediapipe is not None and (
        sequence_observations is not None or sequence_inference_error
    ):
        sequence_failed = (
            sequence_observations is None or
            all(observation is None for observation in sequence_observations)
        )
        runtime_error = (
            getattr(pose_tracker, "_last_error", None) or
            sequence_inference_error or
            "temporal backend returned no valid observations"
        )
        if sequence_failed:
            fallback_from = requested_backend
            fallback_reason = (
                f"{requested_backend.upper()} sequence inference failed: {runtime_error}"
            )
            backend_fallback = True
            # Capture the quality backend diagnostics before releasing it.
            # The fallback tracker below exposes MediaPipe metadata, so a
            # post-close query would otherwise erase the checkpoint/preflight
            # state and the concrete sequence failure from the result.
            try:
                quality_metadata = pose_tracker.get_metadata()
                if isinstance(quality_metadata, dict):
                    # Keep the requested quality report nested so the concrete
                    # MediaPipe fallback remains the selected backend in the
                    # top-level metadata contract.
                    backend_metadata_snapshot["sequenceBackendMetadata"] = dict(
                        quality_metadata
                    )
                    backend_metadata_snapshot["fallbackBackendMetadata"] = dict(
                        quality_metadata
                    )
                    for key in (
                        "preflightStatus",
                        "preflightPhase",
                        "preflightFrames",
                        "preflightError",
                        "missingAssets",
                        "incompatibleAssets",
                        "assetDiagnostics",
                    ):
                        if key in quality_metadata:
                            backend_metadata_snapshot[key] = quality_metadata[key]
            except Exception as metadata_error:
                backend_metadata_snapshot["sequenceMetadataError"] = str(metadata_error)
            backend_metadata_snapshot["sequenceInferenceError"] = (
                sequence_inference_error or runtime_error
            )
            backend_metadata_snapshot["sequenceFallbackFrom"] = fallback_from
            backend_metadata_snapshot["sequenceFallbackReason"] = fallback_reason
            sys.stderr.write(
                f"[TexMotion Warning] {fallback_reason}. Falling back to MediaPipe auxiliary.\n"
            )
            sys.stderr.flush()
            try:
                pose_tracker.close()
            except Exception:
                pass
            pose_tracker = aux_mediapipe
            aux_mediapipe = None
            quality_backend_active = False
            use_pipeline_backend = False
            sequence_observations = None
            # Keep the already-decoded frames so the fallback tracker does not
            # depend on a second VideoCapture seek (which can fail for short
            # or variable-frame-rate clips).
            actual_backend_name = "mediapipe"
            overlay_backend_name = "mediapipe"
            overlay_source = "backend_2d"
            overlay_display_name = "MediaPipe Fallback"

    # Explicit WHAM + MediaPipe mode: align both streams by the sampled
    # timestamp/frame index and invoke the canonical fusion API once for the
    # complete sequence.  The WHAM adapter already emits SMPL-X coordinates;
    # both MediaPipe world axes are converted at the helper
    # boundary.  This keeps the fused result canonical all the way into the
    # existing SMPL-X22 retargeter.
    fusion_active = bool(
        fusion_mode == FUSION_MODE_WHAM_MEDIAPIPE and quality_backend_active
    )
    if fusion_active and sequence_observations is not None:
        if aux_mediapipe is not None and sequence_frames is not None:
            fusion_observation_rows = []
            total_fusion_frames = len(sequence_frames)
            for fusion_index, fusion_frame in enumerate(sequence_frames):
                if fusion_frame is None:
                    fusion_observation_rows.append(None)
                    continue
                try:
                    world, norm, confidence = aux_mediapipe.detect(
                        fusion_frame,
                        int(round(sample_times[fusion_index] * 1000.0)),
                    )
                    fusion_observation_rows.append((world, norm, confidence))
                except Exception as exc:
                    sys.stderr.write(
                        f"[TexMotion Warning] MediaPipe fusion observation failed at frame "
                        f"{fusion_index}: {exc}\n"
                    )
                    fusion_observation_rows.append(None)
                if fusion_index % max(1, total_fusion_frames // 40) == 0 or fusion_index == total_fusion_frames - 1:
                    frac = (fusion_index + 1) / float(max(1, total_fusion_frames))
                    emit_progress(
                        0.08 + 0.30 * frac,
                        "fusion_observing",
                        fusion_index + 1,
                        total_fusion_frames,
                    )
        else:
            fusion_observation_rows = [None] * len(sequence_observations)

        if aux_rtmpose is not None and sequence_frames is not None:
            fusion_rtmpose_rows = []
            for fusion_index, fusion_frame in enumerate(sequence_frames):
                if fusion_frame is None:
                    fusion_rtmpose_rows.append(None)
                    continue
                try:
                    fusion_rtmpose_rows.append(
                        aux_rtmpose.detect(
                            fusion_frame,
                            timestamp_sec=float(sample_times[fusion_index]),
                            frame_index=int(fusion_index),
                        )
                    )
                except Exception as exc:
                    sys.stderr.write(
                        f"[TexMotion Warning] RTMPose fusion observation failed at frame "
                        f"{fusion_index}: {exc}\n"
                    )
                    fusion_rtmpose_rows.append(None)
        else:
            fusion_rtmpose_rows = [None] * len(sequence_observations)

        if pose_tracker is not None and hasattr(pose_tracker, "get_metadata"):
            # Preserve extractor-owned auxiliary provenance when replacing the
            # temporary snapshot with the backend's richer metadata payload.
            auxiliary_metadata = {
                key: value
                for key, value in backend_metadata_snapshot.items()
                if key.startswith("mediapipeAuxiliary")
                or key.startswith("rtmposeAuxiliary")
            }
            try:
                backend_metadata_snapshot = dict(pose_tracker.get_metadata() or {})
            except Exception as exc:
                backend_metadata_snapshot = {"metadataError": str(exc)}
            backend_metadata_snapshot.update(auxiliary_metadata)
            if sequence_decode_missing_frames:
                backend_metadata_snapshot["sequenceDecodeMissingFrames"] = list(
                    sequence_decode_missing_frames
                )
            # ``get_metadata`` reflects the adapter's current state but does
            # not know which extractor frame supplied an automatic seed (or
            # whether the auxiliary graph was reset after probing it).  Keep
            # those run-level diagnostics alongside the adapter metadata so
            # the Unity result card can explain the exact initialization path.
            if wham_seed_probe_consumed:
                backend_metadata_snapshot.setdefault("mediapipeSeedTrackerReset", False)
            if wham_initialization_observation is not None:
                backend_metadata_snapshot["smplInitializationSource"] = (
                    "mediapipe_observation"
                )
                if wham_initialization_frame is not None:
                    backend_metadata_snapshot["smplInitializationFrame"] = int(
                        wham_initialization_frame
                    )

        try:
            emit_progress(0.39, "fusion_optimizing", 0, total_sampled_frames)
            fusion_result = fuse_wham_mediapipe_sequence(
                sequence_observations,
                fusion_observation_rows,
                timestamps=sample_times,
                frame_indices=list(range(total_sampled_frames)),
                config=fusion_config,
                backend_metadata=backend_metadata_snapshot,
                rtmpose_observations=fusion_rtmpose_rows,
            )
            emit_progress(0.42, "fusion_optimizing", total_sampled_frames, total_sampled_frames)
            if fusion_result is not None:
                # The overlay is rendered from the fused canonical 3D stream.
                # Keep this identity separate from ``backendActual=wham`` so
                # the editor never presents fused output as WHAM-only.
                overlay_backend_name = "wham_mediapipe"
                overlay_source = "fused_3d_projection"
                overlay_display_name = "WHAM + MediaPipe Fusion"
                fusion_intervals = [
                    UncertaintyInterval(
                        int(interval.get("startFrame", 0)),
                        int(interval.get("endFrame", 0)),
                        str(interval.get("reason", "")),
                        float(interval.get("confidence", 0.0)),
                        str(interval.get("recommendedAction", "")),
                    )
                    for interval in (fusion_result.uncertainty_intervals or [])
                    if isinstance(interval, dict)
                ]
                sys.stderr.write(
                    "[TexMotion] WHAM + MediaPipe fusion completed on "
                    f"{len(sequence_observations)} aligned frames.\n"
                )
                sys.stderr.flush()
        except Exception as exc:
            # Keep the learned WHAM stream usable if the optional optimizer
            # cannot run.  The concrete reason is exported for review rather
            # than relabelling the result as a successful fusion.
            fusion_result = None
            fusion_active = False
            backend_metadata_snapshot["fusionError"] = str(exc)
            sys.stderr.write(f"[TexMotion Warning] WHAM + MediaPipe fusion failed: {exc}\n")
            sys.stderr.flush()

    if pose_tracker is not None and hasattr(pose_tracker, "get_metadata"):
        try:
            runtime_metadata = dict(pose_tracker.get_metadata() or {})
            runtime_metadata.update(backend_metadata_snapshot)
            backend_metadata_snapshot = runtime_metadata
        except Exception as exc:
            backend_metadata_snapshot.setdefault("metadataError", str(exc))

    # The selected stream is canonical for a successful quality adapter (and
    # for fused output).  Legacy MediaPipe/RTMPose streams stay image-oriented
    # until the original conversion seam below.
    canonical_quality_stream = bool(quality_backend_active)
    if fusion_result is not None:
        canonical_quality_stream = True

    # Keep axis conversion detector-specific.  MediaPipe world coordinates and
    # RTMPose pseudo-world coordinates both use the camera/image convention
    # (+Y down, +Z away), so both need the vertical flip before retargeting.
    # Quality adapters (including WHAM) and successful fusion already emit
    # canonical SMPL-X coordinates; keep them intact through retargeting.
    actual_backend_key = str(actual_backend_name or "").strip().lower()
    invert_landmark_y = bool(actual_backend_key in ("mediapipe", "rtmpose", "rtmpose_hybrid"))
    invert_landmark_z = not canonical_quality_stream
    sys.stderr.write(
        "[TexMotion] Landmark axis policy: "
        f"Y={'inverted' if invert_landmark_y else 'native'} "
        f"(backend={actual_backend_name}); Z={'inverted' if invert_landmark_z else 'native'}\n"
    )
    sys.stderr.flush()

    sym_tracker = RobustPoseTracker()
    euro_filter = OneEuroFilter3D(freq=fps_out, mincutoff=1.2, beta=0.02)

    # MediaPipe can miss an isolated frame even when the surrounding VIDEO
    # graph observations are valid. A quality pose that fails the silhouette
    # gate on such a gap must not be rendered as an unconstrained WHAM
    # projection (that is the source of screen-spanning limbs in the overlay).
    # Keep the fallback local and temporal: interpolate the nearest valid
    # MediaPipe world/2D rows for retargeting and visualization, while leaving
    # the WHAM hypothesis and its provenance untouched.
    def _interpolated_auxiliary_row(frame_index):
        rows = fusion_observation_rows
        if not rows:
            return None

        def _valid(row):
            if row is None or len(row) < 2:
                return False
            try:
                world = np.asarray(row[0], dtype=np.float64)
                norm = np.asarray(row[1], dtype=np.float64)
            except (TypeError, ValueError):
                return False
            return (
                world.ndim == 2 and world.shape[0] >= 33 and world.shape[1] >= 3
                and norm.ndim == 2 and norm.shape[0] >= 33 and norm.shape[1] >= 2
                and np.isfinite(world[:33, :3]).all()
                and np.isfinite(norm[:33, :2]).all()
            )

        previous = None
        following = None
        for candidate in range(min(int(frame_index) - 1, len(rows) - 1), -1, -1):
            if _valid(rows[candidate]):
                previous = candidate
                break
        for candidate in range(max(int(frame_index) + 1, 0), len(rows)):
            if _valid(rows[candidate]):
                following = candidate
                break
        if previous is None and following is None:
            return None
        if previous is None:
            return rows[following], True
        if following is None:
            return rows[previous], True
        if previous == following:
            return rows[previous], True

        before = rows[previous]
        after = rows[following]
        alpha = float(np.clip(
            (float(frame_index) - float(previous)) / max(float(following - previous), 1.0),
            0.0,
            1.0,
        ))
        world_before = np.asarray(before[0], dtype=np.float64)
        world_after = np.asarray(after[0], dtype=np.float64)
        norm_before = np.asarray(before[1], dtype=np.float64)
        norm_after = np.asarray(after[1], dtype=np.float64)
        world = (1.0 - alpha) * world_before + alpha * world_after
        norm = (1.0 - alpha) * norm_before + alpha * norm_after
        conf_before = float(before[2]) if len(before) > 2 else 0.0
        conf_after = float(after[2]) if len(after) > 2 else 0.0
        return (world, norm, (1.0 - alpha) * conf_before + alpha * conf_after), True

    try:
        # Stage 1: Landmark Extraction, Stabilization & Overlay Video Generation
        for idx, t_sec in enumerate(sample_times):
            quality_overlay_landmarks = None
            quality_overlay_projection = None
            quality_overlay_fallback = False
            quality_overlay_fallback_reason = ""
            quality_overlay_suppressed = False
            frame_decode_failed = False
            frame_coordinate_system = last_landmark_coordinate_system
            if sequence_frames is not None and idx < len(sequence_frames):
                frame = sequence_frames[idx]
                ret = frame is not None
            else:
                target_frame_num = int(round(t_sec * orig_fps))
                cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame_num)
                ret, frame = cap.read()

            if not ret or frame is None:
                # Keep the sequence shape and overlay stream aligned even when
                # a Windows codec rejects an isolated random-access read.  The
                # cached temporal pose is safer than dropping the frame (which
                # shifts all subsequent overlay timestamps).  A black frame is
                # used only when the first source frame itself cannot be read.
                frame_decode_failed = True
                if last_video_frame is not None:
                    frame = np.asarray(last_video_frame).copy()
                else:
                    frame = np.zeros(
                        (max(int(cap_height), 1), max(int(cap_width), 1), 3),
                        dtype=np.uint8,
                    )
                ret = True
            else:
                # Keep one bounded copy for a potential decode-gap recovery;
                # sequence extraction already owns the full temporal cache when
                # a quality backend is active.
                last_video_frame = np.asarray(frame).copy()

            timestamp_ms = int(round(t_sec * 1000.0))
            if use_pipeline_backend and pose_tracker:
                obs = None if frame_decode_failed else (
                    sequence_observations[idx]
                    if sequence_observations is not None and idx < len(sequence_observations)
                    else pose_tracker.detect(frame, timestamp_sec=t_sec, frame_index=idx)
                )

                # Run auxiliary MediaPipe 3D tracker if enabled
                aux_world, aux_norm, aux_conf = None, None, 0.0
                aux_fallback_world, aux_fallback_norm, aux_fallback_conf = None, None, 0.0
                aux_row_interpolated = False
                if aux_mediapipe is not None:
                    if fusion_result is not None and idx < len(fusion_observation_rows):
                        # The prepass already sampled this exact frame/time
                        # pair for fusion; reusing it avoids a second detector
                        # call that could advance a stateful VIDEO graph.
                        row = fusion_observation_rows[idx]
                        if row is not None:
                            aux_world, aux_norm, aux_conf = row
                    else:
                        aux_world, aux_norm, aux_conf = aux_mediapipe.detect(frame, timestamp_ms)

                # A short MediaPipe dropout must not expose an unconstrained
                # WHAM projection.  Reuse/interpolate the nearest valid
                # auxiliary row for the safety gate and per-frame fallback;
                # this does not alter the numerical fusion result for frames
                # that already have a valid observation.
                if fusion_result is not None and (aux_world is None or aux_norm is None):
                    interpolated = _interpolated_auxiliary_row(idx)
                    if interpolated is not None:
                        fallback_row, aux_row_interpolated = interpolated
                        if fallback_row is not None and len(fallback_row) >= 2:
                            aux_fallback_world = fallback_row[0]
                            aux_fallback_norm = fallback_row[1]
                            aux_fallback_conf = float(fallback_row[2]) if len(fallback_row) > 2 else 0.0

                # Fill missing image observations from an independent 2D
                # detector, while preserving temporal quality 3D coordinates.
                if quality_backend_active and aux_rtmpose is not None and obs is not None:
                    try:
                        # Reuse the prepass result in explicit fusion mode so
                        # the stateful quality/2D observations retain the same
                        # frame identity and timestamp used by the numerical
                        # fusion seam.
                        if fusion_result is not None and idx < len(fusion_rtmpose_rows):
                            aux_2d_obs = fusion_rtmpose_rows[idx]
                        else:
                            aux_2d_obs = aux_rtmpose.detect(frame, timestamp_sec=t_sec, frame_index=idx)
                        if aux_2d_obs is not None and aux_2d_obs.keypoints_2d:
                            if not obs.keypoints_2d:
                                obs.keypoints_2d = list(aux_2d_obs.keypoints_2d)
                            else:
                                # Keep model-provided points where present;
                                # use RTMPose only for missing slots.
                                obs.keypoints_2d = [
                                    primary if primary.status != 0 else auxiliary
                                    for primary, auxiliary in zip(obs.keypoints_2d, aux_2d_obs.keypoints_2d)
                                ]
                    except Exception:
                        pass

                # A temporal quality adapter may intentionally return only
                # 3D body-model joints (the image observations are internal to
                # WHAM/HMR2/HybrIK).  Accept that result instead of treating
                # it as a dropped frame and copying the previous MediaPipe
                # pose.
                if obs is not None and (obs.keypoints_2d or obs.landmarks_3d):
                    conf = float(obs.raw_confidence)
                    # ``PyTorchPoseBackend`` preserves a canonical 33-slot
                    # list even when a quality model emits only 3D joints.
                    # Do not turn those missing slots into zero-valued image
                    # observations: Stage B uses absent 2D evidence to avoid
                    # inventing a crossing hazard for every frame.
                    has_2d_observation = any(
                        getattr(kp, "status", 0) != 0 and float(getattr(kp, "score", 0.0)) > 0.0
                        for kp in (obs.keypoints_2d[:33] if obs.keypoints_2d else [])
                    )
                    norm_landmarks = np.zeros((33, 3), dtype=np.float64) if has_2d_observation else None
                    if norm_landmarks is not None:
                        for k_idx, kp in enumerate(obs.keypoints_2d[:33]):
                            norm_landmarks[k_idx] = [kp.x, kp.y, kp.score]

                    has_valid_quality_3d = bool(obs.landmarks_3d) and any(
                        getattr(kp, "status", 0) != 0 and
                        np.all(np.isfinite([float(kp.x), float(kp.y), float(kp.z)]))
                        for kp in obs.landmarks_3d[:33]
                    )
                    if fusion_result is not None and idx < len(fusion_result.joints):
                        # FusionResult.joints are already canonical SMPL-X
                        # coordinates.  Do not route them through the legacy
                        # Y/Z inversion path; doing so would double-convert
                        # the WHAM hypothesis before retargeting.
                        frame_landmarks = np.asarray(fusion_result.joints[idx], dtype=np.float64).copy()
                        frame_coordinate_system = "smplx"
                        quality_overlay_coordinate_system = "smplx"
                        quality_overlay_landmarks = frame_landmarks.copy()
                        if idx < len(fusion_observation_rows):
                            row = fusion_observation_rows[idx]
                            if row is not None and len(row) > 1 and row[1] is not None:
                                norm_landmarks = row[1]
                            elif aux_fallback_norm is not None:
                                norm_landmarks = aux_fallback_norm
                            if row is not None and len(row) > 2:
                                conf = max(conf, float(row[2]))
                            elif aux_fallback_conf > 0.0:
                                conf = max(conf, aux_fallback_conf)
                        elif aux_fallback_norm is not None:
                            norm_landmarks = aux_fallback_norm
                    elif quality_backend_active and has_valid_quality_3d and len(obs.landmarks_3d) >= 33:
                        # Learned temporal 3D is the primary solution.  The
                        # MediaPipe stream remains an auxiliary observation
                        # for adapters that consume it, but must not collapse
                        # occluded limbs by replacing this result.
                        # PyTorch quality adapters (including the bundled
                        # native WHAM bridge) expose canonical SMPL-X
                        # coordinates (+Y up, +Z forward).  Keep that contract
                        # intact through the extractor; legacy callers are
                        # converted only when they enter this pipeline.
                        # RTMPose's optional pseudo-3D stream is built from
                        # image-space keypoints (+Y down, +Z away), not a
                        # learned SMPL-X coordinate.  MediaPipe/RTMPose can
                        # run without the auxiliary graph, so infer the
                        # default from the concrete detector instead of
                        # treating every 33-point observation as canonical.
                        metadata_coord = (obs.backend_metadata or {}).get("coordinateSystem")
                        if metadata_coord is None:
                            detector_key = str(getattr(obs, "detector_name", actual_backend_name) or actual_backend_name).lower()
                            metadata_coord = "legacy_overlay" if detector_key in ("rtmpose", "rtmpose_hybrid", "mediapipe") else "smplx"
                        quality_coord = str(metadata_coord).lower()
                        quality_is_smplx = quality_coord not in (
                            "mediapipe", "mediapipe_world", "mp_world",
                            "legacy_overlay", "image_y_down", "image_y_down_z_away",
                            "rtmpose", "rtmpose_hybrid",
                        )
                        quality_xyz = np.array([[kp.x, kp.y, kp.z] for kp in obs.landmarks_3d[:33]], dtype=np.float64)
                        frame_landmarks = quality_xyz
                        frame_coordinate_system = "smplx" if quality_is_smplx else "legacy_overlay"
                        quality_overlay_coordinate_system = frame_coordinate_system
                        # Preserve the pure learned hypothesis for the
                        # overlay before optional auxiliary fill-in.  Missing
                        # WHAM slots remain hidden rather than being drawn as
                        # MediaPipe geometry.
                        quality_overlay_landmarks = np.copy(frame_landmarks)
                        if aux_world is not None and len(aux_world) >= 33:
                            # Use only reliable auxiliary points to fill a
                            # missing quality slot; do not average competing
                            # depth hypotheses across an occlusion.
                            aux_world_canonical = mediapipe_world_to_canonical_landmarks(aux_world)
                            for k_idx, kp in enumerate(obs.landmarks_3d[:33]):
                                if kp.status == 0 and np.isfinite(aux_world_canonical[k_idx]).all():
                                    frame_landmarks[k_idx] = aux_world_canonical[k_idx]
                            conf = max(conf, float(aux_conf) * 0.5)
                    elif aux_world is not None and len(aux_world) >= 33:
                        # HYBRID FUSION: MediaPipe provides metric 3D depth Z and full 33-joint topology
                        frame_landmarks = np.copy(aux_world)

                        # RTMPose high-precision 2D keypoints anchor the (X, Y) positions
                        if norm_landmarks is not None and len(norm_landmarks) >= 33:
                            mp_hip_mid = (aux_world[MP_LEFT_HIP] + aux_world[MP_RIGHT_HIP]) * 0.5
                            rtm_l_hip = norm_landmarks[MP_LEFT_HIP]
                            rtm_r_hip = norm_landmarks[MP_RIGHT_HIP]
                            rtm_pelvis_2d = (rtm_l_hip[:2] + rtm_r_hip[:2]) * 0.5

                            rtm_sh_mid_2d = (norm_landmarks[MP_LEFT_SHOULDER][:2] + norm_landmarks[MP_RIGHT_SHOULDER][:2]) * 0.5
                            rtm_torso_2d = max(0.08, float(np.linalg.norm(rtm_sh_mid_2d - rtm_pelvis_2d)))
                            mp_torso_3d = max(0.20, float(np.linalg.norm(
                                (aux_world[MP_LEFT_SHOULDER] + aux_world[MP_RIGHT_SHOULDER]) * 0.5 - mp_hip_mid
                            )))
                            scale_2d_to_3d = mp_torso_3d / rtm_torso_2d

                            # Anchor X, Y coordinates of detected limbs using RTMPose
                            for k_idx, kp in enumerate(obs.keypoints_2d[:33]):
                                if kp.score >= 0.3:
                                    delta_x = (kp.x - rtm_pelvis_2d[0]) * scale_2d_to_3d
                                    delta_y = (kp.y - rtm_pelvis_2d[1]) * scale_2d_to_3d
                                    blend_w = float(np.clip(kp.score, 0.4, 0.85))
                                    frame_landmarks[k_idx, 0] = (1.0 - blend_w) * frame_landmarks[k_idx, 0] + blend_w * delta_x
                                    frame_landmarks[k_idx, 1] = (1.0 - blend_w) * frame_landmarks[k_idx, 1] + blend_w * delta_y

                        conf = max(conf, float(aux_conf))
                    elif has_valid_quality_3d and len(obs.landmarks_3d) >= 33:
                        # Use temporal quality output directly when no
                        # MediaPipe auxiliary is requested.  This preserves
                        # learned depth through self-occlusion.
                        frame_landmarks = np.zeros((33, 3), dtype=np.float64)
                        # RTMPose's pseudo-3D values are image-derived and
                        # must retain the legacy Y-down/Z-away convention
                        # when it is running without MediaPipe.  Only the
                        # temporal quality adapters are canonical by default.
                        metadata_coord = (obs.backend_metadata or {}).get("coordinateSystem")
                        if metadata_coord is None:
                            detector_key = str(getattr(obs, "detector_name", actual_backend_name) or actual_backend_name).lower()
                            metadata_coord = "legacy_overlay" if detector_key in ("rtmpose", "rtmpose_hybrid", "mediapipe") else "smplx"
                        quality_coord = str(metadata_coord).lower()
                        quality_is_smplx = quality_coord not in (
                            "mediapipe", "mediapipe_world", "mp_world",
                            "legacy_overlay", "image_y_down", "image_y_down_z_away",
                            "rtmpose", "rtmpose_hybrid",
                        )
                        quality_xyz = np.array([[kp.x, kp.y, kp.z] for kp in obs.landmarks_3d[:33]], dtype=np.float64)
                        frame_landmarks = quality_xyz
                        frame_coordinate_system = "smplx" if quality_is_smplx else "legacy_overlay"
                        quality_overlay_coordinate_system = frame_coordinate_system
                        quality_overlay_landmarks = np.copy(frame_landmarks)
                    else:
                        frame_landmarks = get_default_tpose_landmarks(invert_y=invert_landmark_y)
                        frame_coordinate_system = "smplx" if canonical_quality_stream else "legacy_overlay"
                elif aux_world is not None and not quality_backend_active:
                    # RTMPose frame dropped, fallback to MediaPipe
                    frame_landmarks = aux_world
                    norm_landmarks = aux_norm
                    conf = float(aux_conf)
                    frame_coordinate_system = "legacy_overlay"
                elif fusion_result is not None and idx < len(fusion_result.joints):
                    # A temporal WHAM row may be missing while the aligned
                    # MediaPipe row is still usable.  Consume the already
                    # fused, stage-aware frame instead of copying the previous
                    # quality pose (which would discard per-joint fallback
                    # provenance and retime the observation).
                    frame_landmarks = np.asarray(fusion_result.joints[idx], dtype=np.float64).copy()
                    frame_coordinate_system = "smplx"
                    quality_overlay_coordinate_system = "smplx"
                    quality_overlay_landmarks = frame_landmarks.copy()
                    row = (
                        fusion_observation_rows[idx]
                        if idx < len(fusion_observation_rows)
                        else None
                    )
                    if row is not None and len(row) > 1 and row[1] is not None:
                        norm_landmarks = row[1]
                    elif aux_fallback_norm is not None:
                        norm_landmarks = aux_fallback_norm
                        conf = max(float(conf), float(aux_fallback_conf))
                    if row is not None and len(row) > 2:
                        conf = float(np.clip(row[2], 0.0, 1.0))
                elif quality_backend_active:
                    # A quality model that drops a frame must not be replaced
                    # by the auxiliary MediaPipe stream.  The outer recovery
                    # path reuses the previous quality hypothesis instead.
                    frame_landmarks, norm_landmarks, conf = None, None, 0.0
                else:
                    frame_landmarks, norm_landmarks, conf = None, None, 0.0
            else:
                if frame_decode_failed:
                    frame_landmarks, norm_landmarks, conf = None, None, 0.0
                else:
                    frame_landmarks, norm_landmarks, conf = pose_tracker.detect(frame, timestamp_ms)
                frame_coordinate_system = "legacy_overlay"

            # Fusion may select MediaPipe for only a subset of joints when the
            # aligned WHAM row is invalid.  Keep that decision visible in the
            # frame-level overlay provenance instead of labelling the whole
            # video as an uninterrupted WHAM projection.
            if (
                quality_backend_active
                and fusion_result is not None
                and not frame_decode_failed
            ):
                fusion_fallback, fusion_fallback_reason = fusion_frame_overlay_fallback(
                    fusion_result, idx
                )
                if fusion_fallback:
                    quality_overlay_fallback = True
                    quality_overlay_fallback_reason = fusion_fallback_reason or (
                        "WHAM joint invalid; MediaPipe fallback selected"
                    )

            if frame_decode_failed:
                # Never render a finite but unverified WHAM projection for a
                # frame whose source pixels were not decoded.  Keep the last
                # aligned pose (or a neutral first-frame placeholder) and make
                # the decision visible in per-frame fallback diagnostics.
                if last_valid_landmarks is not None:
                    frame_landmarks = np.asarray(last_valid_landmarks, dtype=np.float64).copy()
                    frame_coordinate_system = last_landmark_coordinate_system
                    norm_landmarks = (
                        np.asarray(last_valid_norm_landmarks, dtype=np.float64).copy()
                        if last_valid_norm_landmarks is not None
                        else None
                    )
                    conf = max(float(conf), 0.5)
                else:
                    frame_landmarks = get_default_tpose_landmarks(
                        invert_y=not canonical_quality_stream
                    )
                    frame_coordinate_system = (
                        "smplx" if canonical_quality_stream else "legacy_overlay"
                    )
                    norm_landmarks = None
                    conf = 0.0
                if quality_backend_active and fusion_result is not None:
                    quality_overlay_fallback = True
                    quality_overlay_fallback_reason = (
                        "source video frame decode failed; held previous aligned pose"
                        if last_valid_landmarks is not None
                        else "source video frame decode failed; suppressed unsafe quality overlay"
                    )
                    quality_overlay_suppressed = last_valid_landmarks is None

            # Validate the learned projection for diagnostics/overlay only.
            # Selection has already happened per joint in the pre-retargeting
            # fusion stage.  Replacing every finite WHAM joint here based on a
            # frame-level silhouette check would erase learned occlusion depth
            # and make provenance claim a fallback that did not happen in the
            # fusion solver.
            if (
                quality_backend_active
                and fusion_result is not None
                and quality_overlay_landmarks is not None
                and norm_landmarks is not None
            ):
                try:
                    needs_fallback, quality_alignment_reason, quality_overlay_projection = assess_quality_pose_alignment(
                        quality_overlay_landmarks,
                        norm_landmarks,
                        coordinate_system=quality_overlay_coordinate_system,
                    )
                except Exception as exc:
                    needs_fallback, quality_alignment_reason, quality_overlay_projection = False, "", None
                    sys.stderr.write(
                        f"[TexMotion Warning] Quality alignment check failed at frame {idx}: {exc}\n"
                    )
                if needs_fallback:
                    quality_alignment_warnings.append({
                        "frame": int(idx),
                        "reason": str(quality_alignment_reason),
                        "selection": "stage_aware_joint_fusion",
                    })
                    # A finite WHAM hypothesis can still be geometrically
                    # unusable when the optional image-feature stage is not
                    # provisioned.  Keep its provenance/diagnostic projection
                    # above, but switch this frame's retargeting and overlay
                    # to the already aligned MediaPipe world observation.  The
                    # conversion flips MediaPipe's raw Y/Z camera axes exactly
                    # once, so this path cannot reintroduce the historical
                    # upside-down avatar.  If the current auxiliary row is a
                    # short dropout, use the temporally interpolated row made
                    # above; never render a known-bad WHAM projection merely
                    # because the detector missed one frame.
                    fallback_candidates = []
                    if aux_world is not None:
                        fallback_candidates.append((aux_world, aux_conf, "current"))
                    if aux_fallback_world is not None and aux_fallback_world is not aux_world:
                        fallback_candidates.append((aux_fallback_world, aux_fallback_conf, "temporal"))
                    fallback_applied = False
                    for raw_world, raw_conf, source_label in fallback_candidates:
                        try:
                            fallback_world = mediapipe_world_to_canonical_landmarks(
                                np.asarray(raw_world, dtype=np.float64)
                            )
                            if (
                                fallback_world.ndim == 2
                                and fallback_world.shape[0] >= 33
                                and fallback_world.shape[1] >= 3
                                and np.isfinite(fallback_world[:33, :3]).all()
                            ):
                                frame_landmarks = fallback_world[:33, :3].copy()
                                frame_coordinate_system = "smplx"
                                quality_overlay_fallback = True
                                quality_overlay_fallback_reason = str(quality_alignment_reason)
                                if source_label == "temporal":
                                    quality_overlay_fallback_reason += "; MediaPipe gap filled by temporal interpolation"
                                conf = max(float(conf), float(raw_conf))
                                fallback_applied = True
                                sys.stderr.write(
                                    f"[TexMotion Warning] Frame {idx}: {quality_alignment_reason}; "
                                    f"using {source_label} MediaPipe fallback for retargeting/overlay.\n"
                                )
                                break
                        except Exception as exc:
                            sys.stderr.write(
                                f"[TexMotion Warning] Frame {idx}: {quality_alignment_reason}; "
                                f"MediaPipe {source_label} fallback conversion failed ({exc}).\n"
                            )
                    if not fallback_applied:
                        # No finite auxiliary pose exists for this frame.  Hold
                        # the last retargeted pose when possible; this is a
                        # safer, auditable temporal fallback than drawing
                        # screen-spanning WHAM limbs.  If this is the first
                        # frame, suppress the quality skeleton overlay.
                        if last_valid_landmarks is not None:
                            frame_landmarks = np.asarray(last_valid_landmarks, dtype=np.float64).copy()
                            frame_coordinate_system = last_landmark_coordinate_system
                            if last_valid_norm_landmarks is not None:
                                norm_landmarks = np.asarray(last_valid_norm_landmarks, dtype=np.float64).copy()
                            quality_overlay_fallback = True
                            quality_overlay_fallback_reason = (
                                str(quality_alignment_reason) +
                                "; MediaPipe observation unavailable, held previous aligned pose"
                            )
                            conf = max(float(conf), 0.5)
                            sys.stderr.write(
                                f"[TexMotion Warning] Frame {idx}: {quality_alignment_reason}; "
                                "MediaPipe observation unavailable, holding previous aligned pose.\n"
                            )
                        else:
                            quality_overlay_suppressed = True
                            frame_landmarks = get_default_tpose_landmarks(
                                invert_y=not canonical_quality_stream
                            )
                            frame_coordinate_system = (
                                "smplx" if canonical_quality_stream else "legacy_overlay"
                            )
                            quality_overlay_fallback = True
                            quality_overlay_fallback_reason = (
                                str(quality_alignment_reason) +
                                "; MediaPipe observation unavailable on frame 0, defaulted to safe T-pose"
                            )
                            conf = 0.0
                            sys.stderr.write(
                                f"[TexMotion Warning] Frame {idx}: {quality_alignment_reason}; "
                                "MediaPipe observation unavailable, suppressing unsafe WHAM overlay and defaulting to safe pose.\n"
                            )
                    sys.stderr.flush()

            if frame_landmarks is not None:
                # 1. Correct MediaPipe label swaps only on the legacy image
                # stream.  WHAM and fused output are already canonical SMPL-X
                # coordinates; running the MediaPipe swap heuristic there can
                # exchange learned left/right joints a second time.
                frame_is_canonical = frame_coordinate_system in (
                    "smplx", "canonical", "canonical_smplx", "smplx_y_up_z_forward"
                )
                if not frame_is_canonical:
                    frame_landmarks, norm_landmarks = sym_tracker.process(frame_landmarks, norm_landmarks)
                # 2. Apply 1-Euro Filter to eliminate single-frame depth flutter (Z-jitter)
                frame_landmarks = euro_filter.filter(frame_landmarks)

                last_valid_landmarks = frame_landmarks
                last_valid_norm_landmarks = norm_landmarks
                last_landmark_coordinate_system = frame_coordinate_system
                if (
                    quality_backend_active
                    and quality_overlay_landmarks is not None
                    and not quality_overlay_fallback
                ):
                    last_quality_overlay_landmarks = quality_overlay_landmarks
            else:
                if last_valid_landmarks is not None:
                    frame_landmarks = last_valid_landmarks
                    norm_landmarks = last_valid_norm_landmarks
                    conf = 0.5
                else:
                    # Keep the sequence shape valid if the first quality
                    # inference drops.  This is a neutral pose placeholder,
                    # never a MediaPipe substitution; later frames can still
                    # carry the learned WHAM hypothesis.
                    frame_landmarks = get_default_tpose_landmarks(invert_y=invert_landmark_y)
                    last_valid_landmarks = frame_landmarks
                    norm_landmarks = None
                    conf = 0.0 if quality_backend_active else 0.5
                    frame_coordinate_system = "smplx" if canonical_quality_stream else "legacy_overlay"

            if quality_backend_active and quality_overlay_landmarks is None and not quality_overlay_fallback:
                quality_overlay_landmarks = last_quality_overlay_landmarks

            quality_frame_fallbacks.append(bool(quality_overlay_fallback))
            quality_frame_fallback_reasons.append(str(quality_overlay_fallback_reason or ""))

            landmarks_seq.append(frame_landmarks)
            norm_landmarks_seq.append(norm_landmarks)
            landmark_coordinate_systems.append(frame_coordinate_system)
            frame_confidences.append(float(np.clip(conf, 0.0, 1.0)))

            # Render and write overlay frame according to overlay_mode (dual, 3d, 2d).
            if overlay_writer is not None:
                backend_disp = overlay_display_name
                lms_2d = norm_landmarks
                lms_3d = None
                if quality_backend_active and not quality_overlay_fallback:
                    lms_3d = quality_overlay_projection
                    if lms_3d is None:
                        lms_3d = project_quality_3d_to_image(
                            quality_overlay_landmarks if quality_overlay_landmarks is not None else frame_landmarks,
                            norm_landmarks,
                            coordinate_system=quality_overlay_coordinate_system,
                        )
                elif quality_overlay_fallback:
                    backend_disp = "WHAM -> MediaPipe Fallback"

                if quality_overlay_suppressed:
                    lms_3d = None

                norm_mode = str(overlay_mode or "dual").strip().lower()
                if norm_mode == "2d":
                    primary_lms = lms_2d
                    sec_lms = None
                    disp_name = "MediaPipe"
                elif norm_mode == "3d":
                    primary_lms = lms_3d if lms_3d is not None else lms_2d
                    sec_lms = None
                    disp_name = backend_disp
                else:  # "dual" (default)
                    if lms_3d is not None and lms_2d is not None:
                        primary_lms = lms_3d
                        sec_lms = lms_2d
                        disp_name = backend_disp
                    else:
                        primary_lms = lms_3d if lms_3d is not None else lms_2d
                        sec_lms = None
                        disp_name = backend_disp

                overlay_frame = draw_pose_overlay(
                    frame,
                    primary_lms,
                    idx,
                    total_sampled_frames,
                    conf,
                    backend_name=disp_name,
                    secondary_landmarks=sec_lms,
                    overlay_mode=norm_mode,
                )
                overlay_writer.write(overlay_frame)

            # Progress stream during extraction (0.43 to 0.60 in fusion mode, 0.05 to 0.60 in standard)
            base_prog = 0.43 if fusion_active else 0.05
            span_prog = 0.17 if fusion_active else 0.55
            progress = base_prog + span_prog * ((idx + 1) / total_sampled_frames)
            if idx % max(1, total_sampled_frames // 20) == 0 or idx == total_sampled_frames - 1:
                emit_progress(progress, "extracting", idx + 1, total_sampled_frames)

    finally:
        cap.release()
        if overlay_writer is not None:
            overlay_writer.release()
            time.sleep(0.15)
            transcode_to_h264_if_needed(overlay_video_path)
        # Close each native owner at most once.  The shared MediaPipe facade
        # deliberately has a no-op close, but custom adapters may return the
        # owner itself and MediaPipe Tasks/XNNPACK is not guaranteed to be
        # safe when close() is called twice during an exception path.
        closed_owners = set()
        for owner in (pose_tracker, quality_backend_owner, aux_mediapipe, aux_rtmpose):
            if owner is None or id(owner) in closed_owners:
                continue
            closed_owners.add(id(owner))
            close = getattr(owner, "close", None)
            if callable(close):
                try:
                    close()
                except Exception as close_error:
                    sys.stderr.write(
                        f"[TexMotion Warning] Pose backend cleanup failed: {close_error}\n"
                    )

    sys.stderr.write(f"[TexMotion] Landmark stabilization: Corrected {sym_tracker.swaps_fixed} MediaPipe label inversions.\n")
    sys.stderr.flush()

    total_frames = len(landmarks_seq)
    if total_frames == 0:
        raise RuntimeError("No frames could be extracted from the video.")

    # Convert the canonical 33-landmark stream (MediaPipe-compatible topology)
    # to 22 SMPL-X joints with occlusion reasoning.  For WHAM/HMR2/HybrIK the
    # values in this stream remain the learned quality backend's 3D solution.
    joints_seq = np.zeros((total_frames, SMPLX_JOINT_COUNT, 3), dtype=np.float64)
    for t in range(total_frames):
        frame_coordinate_system = (
            landmark_coordinate_systems[t]
            if t < len(landmark_coordinate_systems)
            else "legacy_overlay"
        )
        frame_is_canonical = frame_coordinate_system in (
            "smplx", "canonical", "canonical_smplx", "smplx_y_up_z_forward"
        )
        joints_seq[t] = convert_mediapipe_landmarks_to_smplx_3d(
            landmarks_seq[t],
            norm_landmarks_seq[t],
            invert_y=False if frame_is_canonical else invert_landmark_y,
            invert_z=False if frame_is_canonical else invert_landmark_z,
            resolve_occlusion=not frame_is_canonical,
        )

    # Preserve uncertainty emitted by a quality backend and merge it with the
    # deterministic Stage B/C optimizer diagnostics below.  Backend intervals
    # are evidence from the model; they are never converted into confident
    # landmarks or silently discarded by the legacy fit.
    uncertainty_intervals = list(fusion_intervals or [])
    uncertainty_intervals.extend(list(getattr(pose_tracker, "last_uncertainty_intervals", []) or []))
    # Surface frame-level quality fallbacks as first-class uncertainty spans
    # so the Unity result card/timeline can explain why MediaPipe was used.
    if quality_frame_fallbacks:
        fallback_start = None
        fallback_reasons = []
        for frame_index, used_fallback in enumerate(quality_frame_fallbacks):
            if used_fallback and fallback_start is None:
                fallback_start = frame_index
                fallback_reasons = []
            if used_fallback:
                reason = quality_frame_fallback_reasons[frame_index] if frame_index < len(quality_frame_fallback_reasons) else ""
                if reason and reason not in fallback_reasons:
                    fallback_reasons.append(reason)
            if fallback_start is not None and (not used_fallback or frame_index == len(quality_frame_fallbacks) - 1):
                end_frame = frame_index if used_fallback and frame_index == len(quality_frame_fallbacks) - 1 else frame_index - 1
                uncertainty_intervals.append(
                    UncertaintyInterval(
                        int(fallback_start),
                        int(max(fallback_start, end_frame)),
                        "wham_geometry_fallback",
                        0.95,
                        "use_mediapipe_fallback",
                    )
                )
                fallback_start = None
                fallback_reasons = []
    if POSE_PIPELINE_AVAILABLE:
        try:
            bone_model = BoneLengthModel()
            bone_model.calibrate_from_observations(landmarks_seq, frame_confidences)
            seq_opt = SequenceOptimizer(fps=fps_out)
            joints_seq, optimizer_intervals, (l_contact, r_contact) = seq_opt.optimize_sequence(
                joints_seq,
                norm_landmarks_seq,
                frame_confidences,
                bone_model=bone_model
            )
            uncertainty_intervals.extend(optimizer_intervals or [])
        except Exception as e:
            sys.stderr.write(f"[TexMotion Warning] Sequence optimization error: {e}\n")
    else:
        # Fallback to legacy heuristics when pose_pipeline is not installed
        joints_seq = resolve_crossing_legs_occlusion(joints_seq, norm_landmarks_seq)
        joints_seq = inpaint_and_constrain_leg_kinematics(joints_seq, norm_landmarks_seq)

    # Stage 2: Foot Contact Locking (0.60 to 0.75)
    emit_progress(0.62, "foot_locking", 0, total_frames)
    contact_track = {"version": 1, "intervals": []}
    if foot_lock:
        joints_seq, contact_track = apply_foot_locking(joints_seq, fps_out, return_contact_track=True)
    elif POSE_PIPELINE_AVAILABLE and 'seq_opt' in locals() and hasattr(seq_opt, "last_contact_track"):
        contact_track = getattr(seq_opt, "last_contact_track", contact_track)
    emit_progress(0.75, "foot_locking", total_frames, total_frames)

    # Stage 3: Analytical 2-Bone IK & Local Rotations (0.75 to 0.88)
    emit_progress(0.76, "solving_ik", 0, total_frames)
    local_rotations = np.zeros((total_frames, SMPLX_JOINT_COUNT, 4), dtype=np.float64)
    root_positions = np.zeros((total_frames, 3), dtype=np.float64)

    # Reconstruct true vertical pelvis height from ground contact / foot & hand relative positions
    # (MediaPipe world landmarks are centered at the hips where pelvis is (0,0,0))
    nominal_leg_length = 0.88
    if total_frames > 0:
        leg_lengths = []
        for t in range(total_frames):
            head_above_pelvis = joints_seq[t, 15, 1] - joints_seq[t, 0, 1]
            if head_above_pelvis > 0.10:
                fd = min(joints_seq[t, 10, 1], joints_seq[t, 11, 1], joints_seq[t, 7, 1] - 0.05, joints_seq[t, 8, 1] - 0.05)
                leg_lengths.append(-fd)
        if leg_lengths:
            nominal_leg_length = float(np.percentile(leg_lengths, 90.0))
    nominal_leg_length = float(np.clip(nominal_leg_length, 0.70, 0.98))

    # Compute raw continuous pelvis Y per frame with smooth hysteresis transition
    pelvis_y_raw = np.zeros(total_frames, dtype=np.float64)
    for t in range(total_frames):
        # Continuous inversion weight based on head-to-pelvis vertical vector
        # upright: head > pelvis + 0.10 -> w_inv = 0.0
        # inverted: head < pelvis - 0.10 -> w_inv = 1.0
        delta_head_pelvis = joints_seq[t, 15, 1] - joints_seq[t, 0, 1]
        w_inv = float(np.clip((0.10 - delta_head_pelvis) / 0.20, 0.0, 1.0))

        # Upright ground contact estimation (feet)
        fd_up = min(joints_seq[t, 10, 1], joints_seq[t, 11, 1], joints_seq[t, 7, 1] - 0.05, joints_seq[t, 8, 1] - 0.05)
        y_upright = float(np.clip(-fd_up, nominal_leg_length * 0.40, nominal_leg_length * 1.35))

        # Inverted ground contact estimation (hands and head)
        contact_inv = min(joints_seq[t, 20, 1], joints_seq[t, 21, 1], joints_seq[t, 15, 1])
        y_inverted = float(np.clip(-contact_inv, 0.35, nominal_leg_length * 1.50))

        # Continuous blend prevents single-frame popping during transitions
        pelvis_y_raw[t] = (1.0 - w_inv) * y_upright + w_inv * y_inverted

    # Temporal Gaussian smoothing on pelvis Y to eliminate single-frame jitter
    if total_frames >= 5:
        kernel = np.array([0.061, 0.242, 0.383, 0.242, 0.061], dtype=np.float64)
        pelvis_y_smooth = np.convolve(pelvis_y_raw, kernel, mode='same')
        pelvis_y_smooth[0] = pelvis_y_raw[0]
        pelvis_y_smooth[-1] = pelvis_y_raw[-1]
    else:
        pelvis_y_smooth = pelvis_y_raw

    prev_q_pelvis = None
    prev_wrist_twist = (None, None)
    for t in range(total_frames):
        root_positions[t] = np.array([joints_seq[t, 0, 0], pelvis_y_smooth[t], joints_seq[t, 0, 2]], dtype=np.float64)

        # Extract MediaPipe hand landmarks in TexMotion SMPL-X coordinate frame
        raw_lm = landmarks_seq[t]
        frame_coordinate_system = (
            landmark_coordinate_systems[t]
            if t < len(landmark_coordinate_systems)
            else "legacy_overlay"
        )
        frame_is_canonical = frame_coordinate_system in (
            "smplx", "canonical", "canonical_smplx", "smplx_y_up_z_forward"
        )
        if frame_is_canonical:
            # Fused/quality streams are already canonical; copying avoids a
            # WHAM-only Y/Z flip before solving the existing SMPL-X rotations.
            lm_t = np.asarray(raw_lm, dtype=np.float64).copy()
        else:
            lm_t = np.zeros_like(raw_lm)
            lm_t[:, 0] = raw_lm[:, 0]
            lm_t[:, 1] = -raw_lm[:, 1] if invert_landmark_y else raw_lm[:, 1]
            lm_t[:, 2] = -raw_lm[:, 2] if invert_landmark_z else raw_lm[:, 2]
        hand_data = {
            "l_wrist": lm_t[MP_LEFT_WRIST],
            "l_index": lm_t[MP_LEFT_INDEX],
            "l_pinky": lm_t[MP_LEFT_PINKY],
            "l_thumb": lm_t[MP_LEFT_THUMB],
            "r_wrist": lm_t[MP_RIGHT_WRIST],
            "r_index": lm_t[MP_RIGHT_INDEX],
            "r_pinky": lm_t[MP_RIGHT_PINKY],
            "r_thumb": lm_t[MP_RIGHT_THUMB],
        }

        local_rotations[t], prev_q_pelvis, prev_wrist_twist = solve_smplx_local_rotations(
            joints_seq[t], prev_q_pelvis, hand_landmarks=hand_data, prev_wrist_twist=prev_wrist_twist
        )

        if t % max(1, total_frames // 10) == 0 or t == total_frames - 1:
            prog = 0.76 + 0.12 * ((t + 1) / total_frames)
            emit_progress(prog, "solving_ik", t + 1, total_frames)

    # Enforce anatomical joint limits across the entire sequence
    local_rotations = apply_anatomical_joint_limits(local_rotations)

    # Stage 4: In-Place Filtering
    if in_place:
        root_positions = apply_in_place_filtering(root_positions)

    # Stage 5: Temporal Smoothing (0.88 to 0.95)
    emit_progress(0.90, "smoothing", 0, total_frames)
    if smooth:
        root_positions, local_rotations = apply_temporal_smoothing(root_positions, local_rotations, fps=fps_out)
        if POSE_PIPELINE_AVAILABLE:
            try:
                seq_opt = SequenceOptimizer(fps=fps_out)
                local_rotations = seq_opt.smooth_quaternions_so3(local_rotations)
            except Exception as e:
                sys.stderr.write(f"[TexMotion Warning] SO3 smoothing error: {e}\n")
        local_rotations = apply_anatomical_joint_limits(local_rotations)
    emit_progress(0.95, "smoothing", total_frames, total_frames)

    # Stage 5b: Short Dropout Temporal Repair (0.2s SLERP) & Long Gap Kinematic Inpainting (0.2s-2.5s Hermite)
    repair_provenance = []
    if POSE_PIPELINE_AVAILABLE and total_frames > 2:
        try:
            if repair_short_gaps_slerp is not None:
                local_rotations, _, short_prov = repair_short_gaps_slerp(
                    local_rotations,
                    frame_confidences,
                    fps=fps_out,
                    max_gap_seconds=0.2,
                    confidence_threshold=0.35
                )
                if short_prov:
                    repair_provenance.extend(short_prov)

            if repair_long_gaps_kinematic_hermite is not None:
                local_rotations, root_positions, _, long_prov = repair_long_gaps_kinematic_hermite(
                    local_rotations,
                    frame_confidences,
                    root_positions=root_positions,
                    contact_track=contact_track,
                    fps=fps_out,
                    min_gap_seconds=0.2,
                    max_gap_seconds=2.5,
                    confidence_threshold=0.35,
                    damping=0.35
                )
                if long_prov:
                    repair_provenance.extend(long_prov)

            local_rotations = apply_anatomical_joint_limits(local_rotations)
        except Exception as repair_err:
            sys.stderr.write(f"[TexMotion Warning] Temporal gap repair error: {repair_err}\n")

    # Stage 5c: Facial Blendshapes & Head Rotation (MediaPipe FaceLandmarker)
    face_track = None
    if POSE_PIPELINE_AVAILABLE and FacePipeline is not None:
        try:
            with FacePipeline(
                model_directory=video_model_directory,
                min_detection_confidence=0.3,
                allow_download=True
            ) as face_pipe:
                if face_pipe.is_available:
                    face_track = face_pipe.process_video(video_path, sample_times)
        except Exception as face_err:
            sys.stderr.write(f"[TexMotion Warning] Face pipeline error: {face_err}\n")
            face_track = None

    # Stage 6: Export final JSON matching VideoMotionData schema
    emit_progress(0.96, "exporting", 0, total_frames)

    # Format JSON payload
    root_pos_json = [
        {"x": round(float(p[0]), 5), "y": round(float(p[1]), 5), "z": round(float(p[2]), 5)}
        for p in root_positions
    ]

    local_rot_json = [
        [
            {"x": round(float(q[0]), 5), "y": round(float(q[1]), 5), "z": round(float(q[2]), 5), "w": round(float(q[3]), 5)}
            for q in local_rotations[t]
        ]
        for t in range(total_frames)
    ]

    flat_rot_json = [
        {"x": round(float(q[0]), 5), "y": round(float(q[1]), 5), "z": round(float(q[2]), 5), "w": round(float(q[3]), 5)}
        for t in range(total_frames)
        for q in local_rotations[t]
    ]

    # Keep interval records stable and remove exact duplicates when a quality
    # adapter and Stage B/C flag the same span.
    unique_intervals = []
    seen_intervals = set()
    for interval in uncertainty_intervals:
        if hasattr(interval, "to_dict"):
            interval_key = (
                int(interval.start_frame), int(interval.end_frame),
                str(interval.reason), str(interval.recommended_action)
            )
            if interval_key in seen_intervals:
                continue
            seen_intervals.add(interval_key)
            unique_intervals.append(interval)
    uncertainty_intervals = unique_intervals
    # Use the metadata captured while the backend was initialized.  Quality
    # adapters release their model in ``close()``, so querying them after the
    # extraction would lose checkpoint and implementation provenance.
    backend_metadata = dict(backend_metadata_snapshot or {})
    if not backend_metadata and pose_tracker is not None and hasattr(pose_tracker, "get_metadata"):
        try:
            backend_metadata = dict(pose_tracker.get_metadata() or {})
        except Exception as e:
            backend_metadata = {"name": getattr(pose_tracker, "name", "unknown"), "metadataError": str(e)}
    if fusion_result is not None:
        fusion_metadata = dict(fusion_result.metadata or {})
        backend_metadata.update(fusion_metadata)
        backend_metadata["fusionDiagnostics"] = fusion_result.diagnostics.to_dict()
        backend_metadata["jointProvenance"] = fusion_result.provenance

    backend_metadata["schemaVersion"] = 2
    backend_metadata["backendRequested"] = requested_backend
    backend_metadata["backendActual"] = actual_backend_name
    backend_metadata["fusionMode"] = fusion_mode
    backend_metadata["backendFallback"] = backend_fallback
    if backend_fallback and fallback_from:
        # A temporal quality failure can promote an already initialized
        # MediaPipe auxiliary tracker.  Keep ``selectedBackend``/readiness
        # truthful for the concrete tracker while retaining the failed quality
        # report under ``fallbackBackendMetadata``.
        backend_metadata["selectedBackend"] = actual_backend_name
        backend_metadata["backendReady"] = bool(
            getattr(pose_tracker, "is_initialized", True)
        )
        backend_metadata.setdefault(
            "fallbackBackendMetadata",
            backend_metadata.get("sequenceBackendMetadata", {}),
        )
    backend_metadata["landmarkAxisPolicy"] = {
        "invertY": bool(invert_landmark_y),
        "invertZ": bool(invert_landmark_z),
        "whamRaw": "camera_y_down_z_away",
        "canonical": "smplx_y_up_z_forward",
        "overlay": "image_y_down",
        "reason": (
            "WHAM native output is canonicalized at the adapter boundary; the "
            "legacy stream is Y-down until final SMPL-X conversion. RTMPose "
            "pseudo-world and MediaPipe world coordinates are Y-down; both "
            "receive the legacy vertical flip. Canonical quality/fused streams receive no additional "
            "Y/Z conversion."
        ),
    }
    if fallback_from:
        backend_metadata["fallbackFrom"] = fallback_from
    if fallback_reason:
        backend_metadata["fallbackReason"] = fallback_reason
    backend_metadata["overlayBackend"] = overlay_backend_name
    backend_metadata["overlaySource"] = overlay_source
    fallback_records = [
        {
            "frame": int(index),
            "reason": str(quality_frame_fallback_reasons[index] or "wham_geometry_fallback"),
        }
        for index, used in enumerate(quality_frame_fallbacks)
        if used
    ]
    backend_metadata["qualityFrameFallbackCount"] = len(fallback_records)
    backend_metadata["qualityFrameFallbacks"] = fallback_records
    backend_metadata["qualityAlignmentWarnings"] = quality_alignment_warnings
    if fallback_records and fusion_result is not None:
        overlay_source = "fused_3d_projection_with_mediapipe_fallback"
        backend_metadata["overlaySource"] = overlay_source
        fallback_reason_values = list(dict.fromkeys(
            str(record["reason"]).strip()
            for record in fallback_records
            if str(record.get("reason", "")).strip()
        ))
        # Keep the concrete gate measurement visible in the result card and
        # Timeline tooltip.  The first reason is representative for the
        # common case; any additional distinct reasons remain available in the
        # per-frame ``qualityFrameFallbacks`` records.
        reason_detail = fallback_reason_values[0] if fallback_reason_values else (
            "WHAM geometry failed silhouette alignment with MediaPipe."
        )
        backend_metadata["qualityFallbackReason"] = (
            "One or more WHAM frames failed silhouette alignment; MediaPipe 3D "
            f"was used for those frames while WHAM diagnostics were retained. {reason_detail}"
        )
    backend_metadata.setdefault("primaryBackend", actual_backend_name)
    if fusion_result is not None:
        # MediaPipe is the auxiliary observation in explicit fusion mode;
        # RTMPose, when available, contributes an independent 2D anchor.
        auxiliary_backends = ["mediapipe_3d", "mediapipe_2d"]
        if aux_rtmpose is not None:
            auxiliary_backends.append("rtmpose_2d")
        backend_metadata["auxiliaryBackends"] = auxiliary_backends
    elif actual_backend_key == "rtmpose" and aux_mediapipe is not None:
        backend_metadata["auxiliaryBackends"] = ["mediapipe_3d"]
    else:
        backend_metadata.setdefault("auxiliaryBackends", [])

    # Keep per-backend provenance useful even when an adapter only supplies
    # flat metadata.  The full per-joint source/weight/status table remains in
    # ``jointProvenance`` when fusion is active.
    if actual_backend_key == "wham" or fusion_result is not None:
        backend_metadata.setdefault("wham", {
            "implementation": backend_metadata.get("adapterImplementation", backend_metadata.get("adapter", "wham")),
            "checkpoint": backend_metadata.get("checkpoint", backend_metadata.get("modelPath")),
            "coordinateSystem": "smplx",
        })
    if fusion_result is not None:
        backend_metadata.setdefault("mediapipe", {
            "implementation": "mediapipe_pose",
            "observationCount": 33,
            "coordinateSystem": "smplx",
        })
        backend_metadata["fusionMode"] = FUSION_MODE_WHAM_MEDIAPIPE
    if fusion_result is not None:
        backend_metadata["fusionAvailable"] = True
        backend_metadata["fusionIntervals"] = [ui.to_dict() for ui in fusion_intervals]
    elif fusion_mode == FUSION_MODE_WHAM_MEDIAPIPE:
        backend_metadata["fusionAvailable"] = False
        backend_metadata.setdefault(
            "fusionError",
            sequence_inference_error or backend_initialization_error or
            "WHAM + MediaPipe fusion was not executed",
        )

    exported_feature_adopted_indices = backend_metadata.get(
        "imageFeatureAdoptedFrameIndices",
        backend_metadata.get("imageFeatureAdoptedFrames", []),
    )
    if not isinstance(exported_feature_adopted_indices, (list, tuple)):
        exported_feature_adopted_indices = []
    exported_feature_adopted_indices = [
        int(index) for index in exported_feature_adopted_indices
        if isinstance(index, (int, float, np.integer, np.floating))
    ]
    exported_feature_input_indices = backend_metadata.get(
        "imageFeatureInputFrameIndices", []
    )
    if not isinstance(exported_feature_input_indices, (list, tuple)):
        exported_feature_input_indices = []
    exported_feature_input_indices = [
        int(index) for index in exported_feature_input_indices
        if isinstance(index, (int, float, np.integer, np.floating))
    ]

    # Publish one small, stable provenance seam for the Unity result card.
    # Keep the detailed backend metadata alongside it for diagnostics, but do
    # not make the card understand every adapter-specific field or dictionary.
    runtime_provenance = build_runtime_provenance(
        requested_backend,
        actual_backend_name,
        backend_metadata,
        backend_fallback=backend_fallback,
        fallback_reason=fallback_reason,
    )
    backend_metadata.update(runtime_provenance)

    motion_data = {
        "frames": total_frames,
        "frameRate": float(fps_out),
        "jointCount": SMPLX_JOINT_COUNT,
        "jointNames": SMPLX_JOINT_NAMES,
        "rootPositions": root_pos_json,
        "localRotations": local_rot_json,
        "flatLocalRotations": flat_rot_json,
        "timestamps": [round(float(sample_times[t]), 4) for t in range(total_frames)],
        "confidences": [round(float(frame_confidences[t]), 4) for t in range(total_frames)],
        "sourceVideoPath": os.path.abspath(video_path),
        "videoWidth": cap_width,
        "videoHeight": cap_height,
        "videoFps": float(orig_fps),
        "overlayVideoPath": os.path.abspath(overlay_video_path) if overlay_video_path and os.path.exists(overlay_video_path) else None,
        "inPlace": bool(in_place),
        "smoothed": bool(smooth),
        "footLocking": bool(foot_lock),
        "detectorName": "rtmpose_hybrid" if (aux_mediapipe is not None and not quality_backend_active) else actual_backend_name,
        "backendRequested": runtime_provenance["backendRequested"],
        "backendActual": runtime_provenance["backendActual"],
        "fusionMode": fusion_mode,
        "backendFallback": backend_fallback,
        "fallbackFrom": fallback_from,
        "fallbackReason": runtime_provenance["fallbackReason"],
        "officialRunnerStatus": runtime_provenance["officialRunnerStatus"],
        "hmr2ImageFeaturesStatus": runtime_provenance["hmr2ImageFeaturesStatus"],
        "vitpose2DStatus": runtime_provenance["vitpose2DStatus"],
        "optionalStageDiagnostics": runtime_provenance["optionalStageDiagnostics"],
        "overlayBackend": overlay_backend_name,
        "overlaySource": overlay_source,
        "backendMetadata": backend_metadata,
        # Keep image-feature execution provenance easy to consume without
        # requiring editor clients to know the nested backend schema.  These
        # fields are copied from the native adapter; no local descriptor is
        # promoted to a learned feature source here.
        "imageFeatureRunner": backend_metadata.get("imageFeatureRunner"),
        "imageFeatureRunnerStatus": backend_metadata.get("imageFeatureRunnerStatus"),
        "imageFeatureRunnerMetadata": backend_metadata.get("imageFeatureRunnerMetadata", {}),
        "imageFeatureInputFrameIndices": exported_feature_input_indices,
        "imageFeatureAdoptedFrameIndices": exported_feature_adopted_indices,
        "imageFeatureAdoptedFrameCount": len(exported_feature_adopted_indices),
        "imageFeatureFallback": backend_metadata.get("imageFeatureFallback"),
        "jointProvenance": backend_metadata.get("jointProvenance", []),
        "fusionDiagnostics": backend_metadata.get("fusionDiagnostics", {}),
        "uncertaintyIntervals": [ui.to_dict() for ui in uncertainty_intervals] if uncertainty_intervals else [],
        "contactTrack": contact_track,
        "faceTrack": face_track,
        "repairProvenance": repair_provenance,
    }

    out_dir = os.path.dirname(os.path.abspath(output_path))
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(motion_data, f, indent=2)

    emit_progress(1.0, "completed", total_frames, total_frames, {
        "output": output_path,
        "overlay": overlay_video_path if overlay_video_path and os.path.exists(overlay_video_path) else None
    })
    return motion_data


# ============================================================================
# CLI Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="TexMotion 3D Video Pose Extractor CLI (MediaPipe Pose + 2-Bone IK to SMPL-X 22)"
    )
    parser.add_argument("--video", type=str, default="", help="Path to input video file")
    parser.add_argument("--output", type=str, required=True, help="Path to output motion JSON file")
    parser.add_argument("--overlay-video", type=str, default=None, help="Path to output pose overlay MP4 video")
    parser.add_argument("--fps", type=float, default=None, help="Target extraction frame rate (default: video fps or 30.0)")
    parser.add_argument("--trim-start", type=float, default=0.0, help="Trim start time in seconds (default: 0.0)")
    parser.add_argument("--trim-end", type=float, default=0.0, help="Trim end time in seconds (default: 0.0 for full video)")
    parser.add_argument("--in-place", action="store_true", default=False, help="Zero horizontal XZ root motion")
    parser.add_argument("--smooth", action="store_true", default=False, help="Apply temporal smoothing")
    parser.add_argument("--no-foot-lock", action="store_true", default=False, help="Disable foot contact locking")
    parser.add_argument("--model-complexity", type=int, default=1, choices=[0, 1, 2], help="MediaPipe model complexity (default: 1)")
    parser.add_argument("--min-detection-confidence", type=float, default=0.5, help="Min detection confidence")
    parser.add_argument("--min-tracking-confidence", type=float, default=0.5, help="Min tracking confidence")
    parser.add_argument(
        "--backend", type=str, default="auto",
        choices=["auto", "mediapipe", "rtmpose", "pytorch", "wham", "hmr2", "hybrik"],
        help="Pose backend: lightweight MediaPipe/RTMPose or optional PyTorch quality adapter (default: auto)"
    )
    parser.add_argument("--pytorch-model", dest="pytorch_model_path", type=str, default=None, help="Optional TorchScript quality model path")
    parser.add_argument("--pytorch-adapter", dest="pytorch_adapter_module", type=str, default=None, help="Optional Python module or .py adapter for WHAM/HMR2/HybrIK")
    parser.add_argument("--pytorch-device", type=str, default="auto", choices=["auto", "cpu", "cuda"], help="PyTorch quality backend device (default: auto)")
    parser.add_argument("--max-sequence-frames", type=int, default=0, help="Optional cap for quality backend sequence length; 0 means unlimited")
    parser.add_argument("--video-model-dir", dest="video_model_directory", type=str, default=None, help="Directory containing downloaded Video2Motion model assets")
    parser.add_argument("--hmr2-runtime", dest="hmr2_runtime_path", type=str, default=None, help="Official 4D-Humans/HMR2 source checkout or installed runtime path")
    parser.add_argument("--hmr2-body-model", dest="hmr2_body_model_path", type=str, default=None, help="Licensed neutral SMPL body model used by official HMR2")
    parser.add_argument("--rtmpose-model", dest="rtmpose_model_path", type=str, default=None, help="Optional explicit RTMPose/DWPose ONNX model path")
    parser.add_argument("--wham-asset-manifest", dest="wham_asset_manifest_path", type=str, default=None, help="Optional WHAM offline asset manifest path")
    parser.add_argument("--wham-body-model", dest="wham_body_model_path", type=str, default=None, help="Optional SMPL/SMPL-X body model path")
    parser.add_argument("--wham-image-feature-backbone", dest="wham_image_feature_backbone_path", type=str, default=None, help="Optional ViTPose image-feature backbone checkpoint")
    parser.add_argument("--wham-image-feature-model-definition", dest="wham_image_feature_model_definition_path", type=str, default=None, help="Optional ViTPose model-definition module or .py file")
    parser.add_argument("--wham-image-feature-config", dest="wham_image_feature_config_path", type=str, default=None, help="Optional ViTPose runner/model config path")
    parser.add_argument("--wham-image-features", dest="wham_image_feature_path", type=str, default=None, help="Optional precomputed WHAM image feature archive (.npy/.npz/.pt)")
    parser.add_argument("--wham-camera", dest="wham_camera_model_path", type=str, default=None, help="Optional camera calibration or camera-motion archive")
    parser.add_argument("--wham-dpvo", dest="wham_dpvo_model_path", type=str, default=None, help="Optional DPVO checkpoint or exported camera-motion archive")
    parser.add_argument("--wham-preprocess-dir", dest="wham_preprocess_directory", type=str, default=None, help="Optional per-video WHAM feature/camera preprocessing cache directory")
    parser.add_argument(
        "--export-vitpose-features",
        action="store_true",
        default=False,
        help="Extract aligned ViTPose image features from the video and write them to --output (.npz)",
    )
    parser.add_argument(
        "--vitpose-feature-chunk-size",
        type=int,
        default=32,
        help="Number of video frames sent to the ViTPose feature runner per batch",
    )
    parser.add_argument(
        "--fusion-mode", dest="fusion_mode", type=str, default=FUSION_MODE_OFF,
        choices=[FUSION_MODE_OFF, FUSION_MODE_WHAM_MEDIAPIPE],
        help="Explicit quality fusion mode; wham_mediapipe keeps WHAM primary and MediaPipe auxiliary observations",
    )
    parser.add_argument(
        "--overlay-mode", dest="overlay_mode", type=str, default="dual",
        choices=["dual", "3d", "2d"],
        help="Overlay visualization mode: dual (2D guide + 3D pose), 3d (3D pose only), or 2d (2D tracking only)",
    )
    parser.add_argument("--synthetic", action="store_true", default=False, help="Generate synthetic test motion (for testing)")

    args = parser.parse_args()

    try:
        if args.export_vitpose_features:
            if not args.video:
                raise ValueError("--video is required with --export-vitpose-features")
            from pose_pipeline.wham_preprocess import export_vitpose_features_from_video

            def report_feature_progress(fraction, frame, total):
                emit_progress(
                    0.05 + (0.90 * float(fraction)),
                    "vitpose_features",
                    int(frame),
                    int(total),
                )

            emit_progress(0.01, "vitpose_features_init", 0, 0)
            result = export_vitpose_features_from_video(
                args.video,
                args.output,
                checkpoint_path=args.wham_image_feature_backbone_path,
                model_definition=args.wham_image_feature_model_definition_path,
                config_path=args.wham_image_feature_config_path,
                target_fps=args.fps,
                trim_start=args.trim_start,
                trim_end=args.trim_end,
                device=args.pytorch_device,
                chunk_size=args.vitpose_feature_chunk_size,
                progress=report_feature_progress,
            )
            emit_progress(
                1.0,
                "completed",
                int(result.get("frames", 0)),
                int(result.get("frames", 0)),
                {"output": result.get("path"), **result},
            )
            return

        if args.synthetic or (not args.video and not os.path.exists(args.video)):
            # Synthetic motion generation
            emit_progress(0.1, "synthetic_generation", 0, 60)
            fps_val = args.fps if args.fps else 30.0
            root_positions, local_rotations = generate_synthetic_motion(
                frames=60,
                fps=fps_val,
                in_place=args.in_place
            )
            if args.smooth:
                root_positions, local_rotations = apply_temporal_smoothing(root_positions, local_rotations, fps=fps_val)

            root_pos_json = [
                {"x": round(float(p[0]), 5), "y": round(float(p[1]), 5), "z": round(float(p[2]), 5)}
                for p in root_positions
            ]
            local_rot_json = [
                [
                    {"x": round(float(q[0]), 5), "y": round(float(q[1]), 5), "z": round(float(q[2]), 5), "w": round(float(q[3]), 5)}
                    for q in local_rotations[t]
                ]
                for t in range(len(root_positions))
            ]
            flat_rot_json = [
                {"x": round(float(q[0]), 5), "y": round(float(q[1]), 5), "z": round(float(q[2]), 5), "w": round(float(q[3]), 5)}
                for t in range(len(root_positions))
                for q in local_rotations[t]
            ]
            # Create synthetic overlay video if requested or default
            overlay_vid = args.overlay_video if args.overlay_video else f"{os.path.splitext(args.output)[0]}_overlay.mp4"
            try:
                ov_writer = cv2.VideoWriter(overlay_vid, cv2.VideoWriter_fourcc(*'mp4v'), fps_val, (640, 360))
                for t in range(len(root_positions)):
                    syn_frame = np.full((360, 640, 3), (22, 24, 30), dtype=np.uint8)
                    # Draw grid
                    for gx in range(0, 640, 40):
                        cv2.line(syn_frame, (gx, 0), (gx, 360), (35, 38, 48), 1)
                    for gy in range(0, 360, 40):
                        cv2.line(syn_frame, (0, gy), (640, gy), (35, 38, 48), 1)
                    # Fake synthetic 2D pose from phase
                    p_phase = 2.0 * math.pi * (t / fps_val)
                    cx, cy = 320, 160
                    # Draw simple synthetic skeleton
                    cv2.circle(syn_frame, (cx, cy - 60), 16, (255, 255, 255), 2)  # Head
                    cv2.line(syn_frame, (cx, cy - 44), (cx, cy + 40), (235, 65, 235), 3)  # Spine
                    # Arms
                    cv2.line(syn_frame, (cx, cy - 30), (int(cx - 50 + 20*math.sin(p_phase)), int(cy + 20)), (0, 140, 255), 3)
                    cv2.line(syn_frame, (cx, cy - 30), (int(cx + 50 - 20*math.sin(p_phase)), int(cy + 20)), (255, 215, 0), 3)
                    # Legs
                    cv2.line(syn_frame, (cx, cy + 40), (int(cx - 30 - 30*math.sin(p_phase)), int(cy + 120)), (40, 80, 255), 3)
                    cv2.line(syn_frame, (cx, cy + 40), (int(cx + 30 + 30*math.sin(p_phase)), int(cy + 120)), (255, 170, 50), 3)
                    # HUD
                    cv2.putText(syn_frame, "TexMotion Synthetic Overlay", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    cv2.putText(syn_frame, f"Frame: {t + 1}/{len(root_positions)} | Synthetic Test", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (80, 245, 120), 1)
                    ov_writer.write(syn_frame)
                ov_writer.release()
                if overlay_vid and os.path.exists(overlay_vid):
                    transcode_to_h264_if_needed(overlay_vid)
            except Exception:
                overlay_vid = None

            synthetic_provenance = {
                "backendRequested": args.backend,
                "backendActual": "synthetic",
                "officialRunnerStatus": "not_executed",
                "hmr2ImageFeaturesStatus": "not_executed",
                "vitpose2DStatus": "not_executed",
                "fallbackReason": None,
                "optionalStageDiagnostics": [],
            }
            synthetic_backend_metadata = {
                "schemaVersion": 2,
                **synthetic_provenance,
                "fusionMode": args.fusion_mode,
                "backendFallback": False,
                "overlayBackend": "synthetic",
                "overlaySource": "synthetic",
                "primaryBackend": "synthetic",
                "auxiliaryBackends": [],
            }
            motion_data = {
                "frames": len(root_positions),
                "frameRate": float(fps_val),
                "jointCount": SMPLX_JOINT_COUNT,
                "jointNames": SMPLX_JOINT_NAMES,
                "rootPositions": root_pos_json,
                "localRotations": local_rot_json,
                "flatLocalRotations": flat_rot_json,
                "timestamps": [round(float(t / fps_val), 4) for t in range(len(root_positions))],
                "confidences": [1.0 for _ in range(len(root_positions))],
                "sourceVideoPath": "synthetic",
                "videoWidth": 640,
                "videoHeight": 360,
                "videoFps": float(fps_val),
                "overlayVideoPath": os.path.abspath(overlay_vid) if overlay_vid and os.path.exists(overlay_vid) else None,
                "inPlace": bool(args.in_place),
                "smoothed": bool(args.smooth),
                "footLocking": not args.no_foot_lock,
                "detectorName": "synthetic",
                "backendRequested": synthetic_provenance["backendRequested"],
                "backendActual": synthetic_provenance["backendActual"],
                "fusionMode": args.fusion_mode,
                "backendFallback": False,
                "fallbackFrom": None,
                "fallbackReason": synthetic_provenance["fallbackReason"],
                "officialRunnerStatus": synthetic_provenance["officialRunnerStatus"],
                "hmr2ImageFeaturesStatus": synthetic_provenance["hmr2ImageFeaturesStatus"],
                "vitpose2DStatus": synthetic_provenance["vitpose2DStatus"],
                "optionalStageDiagnostics": synthetic_provenance["optionalStageDiagnostics"],
                "overlayBackend": "synthetic",
                "overlaySource": "synthetic",
                "backendMetadata": synthetic_backend_metadata,
                "jointProvenance": [],
                "fusionDiagnostics": {},
                "uncertaintyIntervals": [],
                "contactTrack": {"version": 1, "intervals": []},
                "faceTrack": None,
                "repairProvenance": [],
            }
            out_dir = os.path.dirname(os.path.abspath(args.output))
            if out_dir and not os.path.exists(out_dir):
                os.makedirs(out_dir, exist_ok=True)
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(motion_data, f, indent=2)
            emit_progress(1.0, "completed", 60, 60, {
                "output": args.output,
                "overlay": overlay_vid if overlay_vid and os.path.exists(overlay_vid) else None
            })
            sys.exit(0)

        process_video(
            video_path=args.video,
            output_path=args.output,
            target_fps=args.fps,
            trim_start=args.trim_start,
            trim_end=args.trim_end,
            in_place=args.in_place,
            smooth=args.smooth,
            foot_lock=not args.no_foot_lock,
            model_complexity=args.model_complexity,
            min_detection_confidence=args.min_detection_confidence,
            min_tracking_confidence=args.min_tracking_confidence,
            overlay_video_path=args.overlay_video,
            backend=args.backend,
            pytorch_model_path=args.pytorch_model_path,
            pytorch_adapter_module=args.pytorch_adapter_module,
            pytorch_device=args.pytorch_device,
            max_sequence_frames=args.max_sequence_frames,
            video_model_directory=args.video_model_directory,
            hmr2_runtime_path=args.hmr2_runtime_path,
            hmr2_body_model_path=args.hmr2_body_model_path,
            rtmpose_model_path=args.rtmpose_model_path,
            wham_asset_manifest_path=args.wham_asset_manifest_path,
            wham_body_model_path=args.wham_body_model_path,
            wham_image_feature_backbone_path=args.wham_image_feature_backbone_path,
            wham_image_feature_model_definition_path=args.wham_image_feature_model_definition_path,
            wham_image_feature_config_path=args.wham_image_feature_config_path,
            wham_image_feature_path=args.wham_image_feature_path,
            wham_camera_model_path=args.wham_camera_model_path,
            wham_dpvo_model_path=args.wham_dpvo_model_path,
            wham_preprocess_directory=args.wham_preprocess_directory,
            fusion_mode=args.fusion_mode,
            overlay_mode=args.overlay_mode,
        )

    except Exception as e:
        emit_progress(1.0, "error", 0, 0, {"error": str(e)})
        sys.stderr.write(f"[TexMotion Error] {str(e)}\n")
        sys.stderr.flush()
        sys.exit(1)


if __name__ == "__main__":
    main()
