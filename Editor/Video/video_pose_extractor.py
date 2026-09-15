#!/usr/bin/env python3
"""
video_pose_extractor.py - TexMotion Video-to-Motion 3D Pose Extraction Pipeline

Extracts 3D humanoid motion from 2D video using MediaPipe Pose Landmarker
and analytical 2-bone inverse kinematics (IK) matching TexMotion's SMPL-X 22
joint hierarchy (defined in SmplxJointDefinitions.cs).

Features:
- Extracts 33 3D landmarks per video frame.
- Solves 2-bone analytical IK for arms and legs, computing 22 SMPL-X local rotations.
- Detects foot contacts and applies floor snapping (foot locking) to eliminate sliding.
- In-Place motion processing (zeroes horizontal XZ root motion, preserves vertical bounce).
- Temporal smoothing via Savitzky-Golay / continuous quaternion filtering.
- Streaming progress reporting via JSON Lines to stdout.
- Exports motion data matching the VideoMotionData schema.
- Resilient fallback pose extractor when MediaPipe is not installed.
"""

import argparse
import json
import math
import os
import sys
import time
import shutil
import subprocess
import numpy as np

import urllib.request
import tempfile

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

def get_default_tpose_landmarks() -> np.ndarray:
    """
    Returns valid default standing humanoid landmarks in MediaPipe world coordinates
    (-Y is Up, +X is Left, -X is Right, -Z is Forward).
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
    return lms


class MediaPipePoseTracker:
    """
    Hybrid MediaPipe Pose backend supporting both Tasks API (mediapipe >= 0.10.15 / 1.0+)
    and legacy Solutions API (mediapipe <= 0.10.14).
    Automatically downloads the .task model bundle if using Tasks API.
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

    def __init__(self, model_complexity: int = 1, min_detection_confidence: float = 0.5, min_tracking_confidence: float = 0.5):
        self.mode = MEDIAPIPE_MODE
        self.model_complexity = model_complexity
        self.landmarker = None
        self.solution_tracker = None

        if self.mode == 'tasks':
            model_path = self._ensure_model(model_complexity)
            options = PoseLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=model_path),
                running_mode=RunningMode.VIDEO,
                min_pose_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence
            )
            self.landmarker = PoseLandmarker.create_from_options(options)
            sys.stderr.write(f"[TexMotion] MediaPipe Tasks API initialized with {os.path.basename(model_path)}\n")
            sys.stderr.flush()
        elif self.mode == 'solutions':
            self.solution_tracker = mp.solutions.pose.Pose(
                static_image_mode=False,
                model_complexity=model_complexity,
                smooth_landmarks=True,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence
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
        candidate_dirs = [
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

        # If not found, download to primary cache directory
        target_dir = candidate_dirs[0] if candidate_dirs[0] else candidate_dirs[2]
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

        if self.mode == 'tasks' and self.landmarker is not None:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            results = self.landmarker.detect_for_video(mp_image, int(timestamp_ms))
            if results.pose_world_landmarks and len(results.pose_world_landmarks) > 0:
                lms_world = results.pose_world_landmarks[0]
                world_landmarks = np.zeros((33, 3), dtype=np.float64)
                vis_sum = 0.0
                for idx, lm in enumerate(lms_world):
                    world_landmarks[idx] = [lm.x, lm.y, lm.z]
                    vis_sum += getattr(lm, 'visibility', 1.0)
                conf = vis_sum / 33.0

                norm_landmarks = None
                if results.pose_landmarks and len(results.pose_landmarks) > 0:
                    lms_norm = results.pose_landmarks[0]
                    norm_landmarks = np.zeros((33, 3), dtype=np.float64)
                    for idx, lm in enumerate(lms_norm):
                        norm_landmarks[idx] = [lm.x, lm.y, getattr(lm, 'visibility', 1.0)]

                return world_landmarks, norm_landmarks, conf
            return None, None, 0.0

        elif self.mode == 'solutions' and self.solution_tracker is not None:
            results = self.solution_tracker.process(rgb_frame)
            if results.pose_world_landmarks:
                world_landmarks = np.zeros((33, 3), dtype=np.float64)
                vis_sum = 0.0
                for idx, lm in enumerate(results.pose_world_landmarks.landmark):
                    world_landmarks[idx] = [lm.x, lm.y, lm.z]
                    vis_sum += getattr(lm, 'visibility', 1.0)
                conf = vis_sum / 33.0

                norm_landmarks = None
                if results.pose_landmarks:
                    norm_landmarks = np.zeros((33, 3), dtype=np.float64)
                    for idx, lm in enumerate(results.pose_landmarks.landmark):
                        norm_landmarks[idx] = [lm.x, lm.y, getattr(lm, 'visibility', 1.0)]

                return world_landmarks, norm_landmarks, conf
            return None, None, 0.0

        return None, None, 0.0

    def close(self):
        try:
            if self.landmarker is not None:
                self.landmarker.close()
        except Exception:
            pass
        try:
            if self.solution_tracker is not None:
                self.solution_tracker.close()
        except Exception:
            pass


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
    conf: float
) -> np.ndarray:
    """
    Renders high-aesthetic MediaPipe pose skeleton lines and joint points onto frame_bgr.
    Uses vibrant color-coded body segments:
    - Torso: Neon Magenta
    - Left Arm: Neon Cyan
    - Right Arm: Neon Coral / Orange
    - Left Leg: Neon Sky Blue
    - Right Leg: Neon Red / Orange
    - Face: Neon Lime Green
    Also renders a sleek semi-transparent HUD badge with frame index and confidence score.
    """
    out = frame_bgr.copy()
    h, w, _ = out.shape

    if norm_landmarks is None or len(norm_landmarks) < 33:
        return out

    # Convert norm landmarks to pixel coordinates
    pts = []
    vis = []
    for i in range(33):
        lx, ly = norm_landmarks[i][0], norm_landmarks[i][1]
        v = norm_landmarks[i][2] if len(norm_landmarks[i]) > 2 else 1.0
        px = int(np.clip(lx * w, 0, w - 1))
        py = int(np.clip(ly * h, 0, h - 1))
        pts.append((px, py))
        vis.append(v)

    # Connections and styles: (connections, BGR color, thickness)
    connection_groups = [
        (POSE_CONNECTIONS_TORSO, (235, 65, 235), 3),      # Torso: Magenta
        (POSE_CONNECTIONS_LEFT_ARM, (255, 215, 0), 3),    # Left arm: Cyan
        (POSE_CONNECTIONS_RIGHT_ARM, (0, 140, 255), 3),   # Right arm: Coral
        (POSE_CONNECTIONS_LEFT_LEG, (255, 170, 50), 3),   # Left leg: Sky Blue
        (POSE_CONNECTIONS_RIGHT_LEG, (40, 80, 255), 3),   # Right leg: Bright Orange
        (POSE_CONNECTIONS_FACE, (80, 235, 80), 2),        # Face: Lime
    ]

    # Draw skeleton lines
    for connections, color, thickness in connection_groups:
        for j1, j2 in connections:
            if vis[j1] > 0.25 and vis[j2] > 0.25:
                cv2.line(out, pts[j1], pts[j2], color, thickness, cv2.LINE_AA)

    # Draw joint dots (circles with white glow border)
    for i in range(33):
        if vis[i] > 0.25:
            pt = pts[i]
            # Outer white glow
            cv2.circle(out, pt, 5, (255, 255, 255), -1, cv2.LINE_AA)
            # Inner colored dot
            is_left = i in [MP_LEFT_SHOULDER, MP_LEFT_ELBOW, MP_LEFT_WRIST, MP_LEFT_HIP, MP_LEFT_KNEE, MP_LEFT_ANKLE]
            dot_color = (255, 200, 0) if is_left else (0, 140, 255)
            cv2.circle(out, pt, 3, dot_color, -1, cv2.LINE_AA)

    # Semi-transparent HUD overlay on top-left
    hud_w, hud_h = 290, 56
    if w >= hud_w + 20 and h >= hud_h + 20:
        hud_roi = out[12:12 + hud_h, 12:12 + hud_w]
        dark_overlay = np.full_like(hud_roi, (24, 26, 32))
        cv2.addWeighted(dark_overlay, 0.82, hud_roi, 0.18, 0, hud_roi)
        out[12:12 + hud_h, 12:12 + hud_w] = hud_roi
        cv2.rectangle(out, (12, 12), (12 + hud_w, 12 + hud_h), (60, 160, 240), 1, cv2.LINE_AA)

        cv2.putText(out, "TexMotion MediaPipe Overlay", (22, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        conf_pct = int(round(conf * 100))
        conf_color = (80, 245, 120) if conf_pct >= 70 else ((50, 210, 255) if conf_pct >= 45 else (80, 100, 255))
        cv2.putText(out, f"Frame: {frame_idx + 1}/{total_frames} | Confidence: {conf_pct}%", (22, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.42, conf_color, 1, cv2.LINE_AA)

    return out


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
        result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=60)
        if result.returncode == 0 and os.path.exists(temp_h264) and os.path.getsize(temp_h264) > 0:
            os.replace(temp_h264, video_path)
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

def convert_mediapipe_landmarks_to_smplx_3d(mp_landmarks: np.ndarray, norm_landmarks: np.ndarray = None) -> np.ndarray:
    """
    Converts 33 MediaPipe 3D world landmarks into 22 SMPL-X joint 3D positions.
    Coordinate frame conversion:
    MediaPipe World Landmarks: X=right, Y=down, Z=away from camera.
    TexMotion SMPL-X Frame:
    +X: Character's Left (+X)
    -X: Character's Right (-X)
    +Y: Up (+Y)
    +Z: Forward (+Z)
    """
    lm = np.zeros_like(mp_landmarks)
    lm[:, 0] = mp_landmarks[:, 0]    # Left is +X, Right is -X
    lm[:, 1] = -mp_landmarks[:, 1]   # Invert Y so Up is +Y
    lm[:, 2] = -mp_landmarks[:, 2]   # Invert Z so Forward is +Z

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

    # Resolve behind-the-head and behind-the-back arm occlusions
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
    vel_thresh: float = 0.45
) -> np.ndarray:
    """
    Detects stance/contact phases for left and right feet and snaps the feet to the floor
    during the contact phase to eliminate foot sliding.
    Then analytically adjusts the knee positions via 2-bone IK.
    """
    T = joints_seq.shape[0]
    if T < 4:
        return joints_seq

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
        return joints_seq

    upright_feet_y = np.concatenate([
        result_joints[upright_indices, 7, 1],   # L_Ankle Y
        result_joints[upright_indices, 8, 1],   # R_Ankle Y
        result_joints[upright_indices, 10, 1],  # L_Foot Y
        result_joints[upright_indices, 11, 1],  # R_Foot Y
    ])
    floor_y = float(np.percentile(upright_feet_y, 5.0))
    ankle_offset = 0.06  # typical nominal ankle height above floor

    # 2. Compute velocities
    for leg_idx, (hip_i, knee_i, ankle_i, foot_i) in enumerate([(1, 4, 7, 10), (2, 5, 8, 11)]):
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

        # Apply locking and 2-bone IK adjustment
        for seg_start, seg_end in segments:
            # Anchor position
            anchor_pos = np.median(orig_ankles[seg_start:seg_end + 1], axis=0)
            anchor_pos[1] = floor_y + ankle_offset

            seg_len = seg_end - seg_start + 1
            for t in range(seg_start, seg_end + 1):
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
    overlay_video_path: str = None
) -> dict:
    """
    Full pipeline: reads video, extracts landmarks, computes 2-bone IK rotations,
    generates MediaPipe pose overlay video, applies foot locking and smoothing,
    and exports to output_path and overlay_video_path.
    """
    emit_progress(0.02, "init", 0, 100)

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

    # Compute target sample timestamps
    sample_times = []
    curr_t = start_sec
    dt = 1.0 / fps_out
    while curr_t < end_sec:
        sample_times.append(curr_t)
        curr_t += dt

    total_sampled_frames = len(sample_times)
    if total_sampled_frames <= 0:
        raise ValueError(f"No frames to extract in time range [{start_sec:.2f}s, {end_sec:.2f}s]")

    cap_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cap_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    landmarks_seq = []
    norm_landmarks_seq = []
    frame_confidences = []
    last_valid_landmarks = None
    last_valid_norm_landmarks = None

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

    # Ensure MediaPipe is available
    if not MEDIAPIPE_AVAILABLE:
        raise RuntimeError("MediaPipe is not installed or supported in this Python environment. Please install it using: pip install mediapipe opencv-python")

    pose_tracker = MediaPipePoseTracker(
        model_complexity=model_complexity,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence
    )
    sym_tracker = RobustPoseTracker()
    euro_filter = OneEuroFilter3D(freq=fps_out, mincutoff=1.2, beta=0.02)

    try:
        # Stage 1: Landmark Extraction, Stabilization & Overlay Video Generation
        for idx, t_sec in enumerate(sample_times):
            target_frame_num = int(round(t_sec * orig_fps))
            cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame_num)
            ret, frame = cap.read()

            if not ret or frame is None:
                if last_valid_landmarks is not None:
                    landmarks_seq.append(last_valid_landmarks)
                    norm_landmarks_seq.append(last_valid_norm_landmarks)
                    frame_confidences.append(0.5)
                continue

            timestamp_ms = int(round(t_sec * 1000.0))
            frame_landmarks, norm_landmarks, conf = pose_tracker.detect(frame, timestamp_ms)

            if frame_landmarks is not None:
                # 1. Correct MediaPipe label swaps (left/right and front/back inversion)
                frame_landmarks, norm_landmarks = sym_tracker.process(frame_landmarks, norm_landmarks)
                # 2. Apply 1-Euro Filter to eliminate single-frame depth flutter (Z-jitter)
                frame_landmarks = euro_filter.filter(frame_landmarks)

                last_valid_landmarks = frame_landmarks
                last_valid_norm_landmarks = norm_landmarks
            else:
                if last_valid_landmarks is not None:
                    frame_landmarks = last_valid_landmarks
                    norm_landmarks = last_valid_norm_landmarks
                    conf = 0.5
                else:
                    frame_landmarks = get_default_tpose_landmarks()
                    last_valid_landmarks = frame_landmarks
                    norm_landmarks = None
                    conf = 0.5

            landmarks_seq.append(frame_landmarks)
            norm_landmarks_seq.append(norm_landmarks)
            frame_confidences.append(float(np.clip(conf, 0.0, 1.0)))

            # Render and write overlay frame
            if overlay_writer is not None:
                overlay_frame = draw_pose_overlay(frame, norm_landmarks, idx, total_sampled_frames, conf)
                overlay_writer.write(overlay_frame)

            # Progress stream during extraction (0.05 to 0.60)
            progress = 0.05 + 0.55 * ((idx + 1) / total_sampled_frames)
            if idx % max(1, total_sampled_frames // 20) == 0 or idx == total_sampled_frames - 1:
                emit_progress(progress, "extracting", idx + 1, total_sampled_frames)

    finally:
        cap.release()
        if overlay_writer is not None:
            overlay_writer.release()
            transcode_to_h264_if_needed(overlay_video_path)
        pose_tracker.close()

    sys.stderr.write(f"[TexMotion] Landmark stabilization: Corrected {sym_tracker.swaps_fixed} MediaPipe label inversions.\n")
    sys.stderr.flush()

    total_frames = len(landmarks_seq)
    if total_frames == 0:
        raise RuntimeError("No frames could be extracted from the video.")

    # Convert 33 MediaPipe landmarks to 22 SMPL-X 3D Joint positions with occlusion reasoning
    joints_seq = np.zeros((total_frames, SMPLX_JOINT_COUNT, 3), dtype=np.float64)
    for t in range(total_frames):
        joints_seq[t] = convert_mediapipe_landmarks_to_smplx_3d(landmarks_seq[t], norm_landmarks_seq[t])

    # Stage 2: Foot Contact Locking (0.60 to 0.75)
    emit_progress(0.62, "foot_locking", 0, total_frames)
    if foot_lock:
        joints_seq = apply_foot_locking(joints_seq, fps_out)
    emit_progress(0.75, "foot_locking", total_frames, total_frames)

    # Stage 3: Analytical 2-Bone IK & Local Rotations (0.75 to 0.88)
    emit_progress(0.76, "solving_ik", 0, total_frames)
    local_rotations = np.zeros((total_frames, SMPLX_JOINT_COUNT, 4), dtype=np.float64)
    root_positions = np.zeros((total_frames, 3), dtype=np.float64)

    # Reconstruct true vertical pelvis height from ground contact / foot & hand relative positions
    # (MediaPipe world landmarks are centered at the hips where pelvis is (0,0,0))
    nominal_leg_length = 0.90
    if total_frames > 0:
        leg_lengths = []
        for t in range(total_frames):
            head_above_pelvis = joints_seq[t, 15, 1] - joints_seq[t, 0, 1]
            if head_above_pelvis > 0.10:
                fd = min(joints_seq[t, 10, 1], joints_seq[t, 11, 1], joints_seq[t, 7, 1] - 0.05, joints_seq[t, 8, 1] - 0.05)
                leg_lengths.append(-fd)
        if leg_lengths:
            nominal_leg_length = float(np.percentile(leg_lengths, 90.0))
    nominal_leg_length = max(0.50, nominal_leg_length)

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
        lm_t = np.zeros_like(raw_lm)
        lm_t[:, 0] = raw_lm[:, 0]
        lm_t[:, 1] = -raw_lm[:, 1]
        lm_t[:, 2] = -raw_lm[:, 2]
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
        local_rotations = apply_anatomical_joint_limits(local_rotations)
    emit_progress(0.95, "smoothing", total_frames, total_frames)

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
    parser.add_argument("--synthetic", action="store_true", default=False, help="Generate synthetic test motion (for testing)")

    args = parser.parse_args()

    try:
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
        )

    except Exception as e:
        emit_progress(1.0, "error", 0, 0, {"error": str(e)})
        sys.stderr.write(f"[TexMotion Error] {str(e)}\n")
        sys.stderr.flush()
        sys.exit(1)


if __name__ == "__main__":
    main()
