"""
mediapipe_backend.py - MediaPipe Pose Landmarker backend implementation.
"""

import sys
from typing import Optional, List, Tuple
import numpy as np
from .base import PoseBackend, BackendCapabilities
from ..observations import FrameObservations, Keypoint2D, Keypoint3D, ObservationStatus

try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    mp = None


def check_mediapipe_available() -> Tuple[bool, str]:
    """Preflight check for MediaPipe readiness."""
    if not OPENCV_AVAILABLE:
        return False, "OpenCV (cv2) is not installed."
    if not MEDIAPIPE_AVAILABLE or mp is None:
        return False, "MediaPipe is not installed."
    return True, "MediaPipe backend available"


class MediaPipeBackend(PoseBackend):
    """Wraps MediaPipe Pose Landmarker producing 33 landmarks."""

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

    def _ensure_model(self) -> Optional[str]:
        import os, tempfile, urllib.request, sys
        complexity = min(2, max(0, self.model_complexity))
        target_name = self.MODEL_NAMES.get(complexity, "pose_landmarker_full.task")

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
            os.path.join(os.path.dirname(__file__), "..", ".."),
            os.path.dirname(os.path.abspath(__file__)),
        ]

        # 1. Exact match in candidate directories
        for d in candidate_dirs:
            if not d:
                continue
            p = os.path.join(d, target_name)
            if os.path.exists(p) and os.path.getsize(p) > 1024 * 1024:
                return p

        # 2. Any valid task model in candidate directories
        for d in candidate_dirs:
            if not d or not os.path.exists(d):
                continue
            for alt in ["pose_landmarker_heavy.task", "pose_landmarker_full.task", "pose_landmarker_lite.task", "pose_landmarker.task"]:
                p = os.path.join(d, alt)
                if os.path.exists(p) and os.path.getsize(p) > 1024 * 1024:
                    return p

        # 3. Download if not found.  Quality adapters can explicitly request
        # an offline-only auxiliary detector so a missing task asset fails
        # quickly (and the caller can surface the exact fallback reason)
        # instead of blocking extraction on a network timeout.
        if not self.allow_download:
            return None
        url = self.MODEL_URLS.get(complexity, self.MODEL_URLS[1])
        target_dir = configured_dir or default_video_cache or (
            candidate_dirs[2] if candidate_dirs[2] else candidate_dirs[4]
        )
        try:
            os.makedirs(target_dir, exist_ok=True)
            download_path = os.path.join(target_dir, target_name)
            tmp_path = download_path + ".tmp"
            sys.stderr.write(f"[TexMotion] Downloading MediaPipe task model: {target_name}...\n")
            sys.stderr.flush()
            urllib.request.urlretrieve(url, tmp_path)
            if os.path.exists(tmp_path) and os.path.getsize(tmp_path) > 1024 * 1024:
                if os.path.exists(download_path):
                    os.remove(download_path)
                os.rename(tmp_path, download_path)
                return download_path
        except Exception as e:
            sys.stderr.write(f"[TexMotion Warning] Failed to download {target_name}: {e}\n")
            sys.stderr.flush()

        return None

    def __init__(
        self,
        model_complexity: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        model_directory: Optional[str] = None,
        allow_download: bool = True,
    ):
        super().__init__("mediapipe")
        self.model_complexity = model_complexity
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.model_directory = model_directory
        self.allow_download = bool(allow_download)
        self._landmarker = None
        self._mode = None
        self._last_timestamp_ms = -1

    def is_available(self) -> bool:
        return MEDIAPIPE_AVAILABLE and OPENCV_AVAILABLE

    def get_capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            name="mediapipe",
            has_2d=True,
            has_3d=True,
            has_visibility=True,
            has_presence=True,
            is_temporal=True,
            keypoint_count=33
        )

    def initialize(self) -> bool:
        if not self.is_available():
            return False

        self._last_timestamp_ms = -1

        # 1. Try Tasks API first if task model exists or can be downloaded
        try:
            from mediapipe.tasks.python import BaseOptions
            from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions, RunningMode

            model_path = self._ensure_model()
            if model_path:
                options = PoseLandmarkerOptions(
                    base_options=BaseOptions(model_asset_path=model_path),
                    running_mode=RunningMode.VIDEO,
                    min_pose_detection_confidence=self.min_detection_confidence,
                    min_pose_presence_confidence=self.min_detection_confidence,
                    min_tracking_confidence=self.min_tracking_confidence,
                    output_segmentation_masks=False
                )
                self._landmarker = PoseLandmarker.create_from_options(options)
                self._mode = "tasks"
                self.is_initialized = True
                return True
        except Exception as e:
            import sys
            sys.stderr.write(f"[TexMotion Warning] MediaPipe Tasks API initialization failed: {e}\n")

        # 2. Fallback to Solutions API
        try:
            if hasattr(mp, "solutions") and hasattr(mp.solutions, "pose"):
                self._landmarker = mp.solutions.pose.Pose(
                    static_image_mode=False,
                    model_complexity=self.model_complexity,
                    smooth_landmarks=True,
                    min_detection_confidence=self.min_detection_confidence,
                    min_tracking_confidence=self.min_tracking_confidence
                )
                self._mode = "solutions"
                self.is_initialized = True
                return True
        except Exception:
            pass

        self.is_initialized = False
        return False

    def detect(self, frame_bgr: np.ndarray, timestamp_sec: float, frame_index: int = 0) -> Optional[FrameObservations]:
        if not self.is_initialized or self._landmarker is None:
            return None
        if frame_bgr is None or frame_bgr.size == 0:
            return None

        try:
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            rgb = np.ascontiguousarray(rgb)
            kps_2d: List[Keypoint2D] = []
            lms_3d: List[Keypoint3D] = []

            if self._mode == "tasks":
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                # Enforce strictly monotonic timestamps for MediaPipe Tasks VIDEO mode
                desired_ms = int(round(timestamp_sec * 1000.0))
                timestamp_ms = max(self._last_timestamp_ms + 1, desired_ms)
                self._last_timestamp_ms = timestamp_ms

                result = self._landmarker.detect_for_video(mp_image, timestamp_ms)
                if not result.pose_landmarks or len(result.pose_landmarks) == 0:
                    return None

                raw_2d = result.pose_landmarks[0]
                raw_3d = result.pose_world_landmarks[0] if result.pose_world_landmarks else None

                for i, lm in enumerate(raw_2d):
                    vis = float(getattr(lm, "visibility", 1.0))
                    pres = float(getattr(lm, "presence", 1.0))
                    score = min(vis, pres)
                    status = ObservationStatus.OBSERVED if score >= 0.4 else ObservationStatus.OCCLUDED

                    kps_2d.append(Keypoint2D(
                        x=float(lm.x),
                        y=float(lm.y),
                        score=float(score),
                        visibility=vis,
                        presence=pres,
                        status=status
                    ))

                    if raw_3d and i < len(raw_3d):
                        wlm = raw_3d[i]
                        lms_3d.append(Keypoint3D(
                            x=float(wlm.x),
                            y=float(wlm.y),
                            z=float(wlm.z),
                            score=float(score),
                            visibility=vis,
                            presence=pres,
                            status=status
                        ))
                    else:
                        lms_3d.append(Keypoint3D(
                            x=float(lm.x),
                            y=float(lm.y),
                            z=0.0,
                            score=float(score),
                            visibility=vis,
                            presence=pres,
                            status=status
                        ))

            elif self._mode == "solutions":
                result = self._landmarker.process(rgb)
                if not result or not result.pose_landmarks:
                    return None

                raw_2d = result.pose_landmarks.landmark
                raw_3d = result.pose_world_landmarks.landmark if result.pose_world_landmarks else None

                for i, lm in enumerate(raw_2d):
                    vis = float(getattr(lm, "visibility", 1.0))
                    pres = float(getattr(lm, "presence", 1.0))
                    score = min(vis, pres)
                    status = ObservationStatus.OBSERVED if score >= 0.4 else ObservationStatus.OCCLUDED

                    kps_2d.append(Keypoint2D(
                        x=float(lm.x),
                        y=float(lm.y),
                        score=float(score),
                        visibility=vis,
                        presence=pres,
                        status=status
                    ))

                    if raw_3d and i < len(raw_3d):
                        wlm = raw_3d[i]
                        lms_3d.append(Keypoint3D(
                            x=float(wlm.x),
                            y=float(wlm.y),
                            z=float(wlm.z),
                            score=float(score),
                            visibility=vis,
                            presence=pres,
                            status=status
                        ))
                    else:
                        lms_3d.append(Keypoint3D(
                            x=float(lm.x),
                            y=float(lm.y),
                            z=0.0,
                            score=float(score),
                            visibility=vis,
                            presence=pres,
                            status=status
                        ))

            scores = [kp.score for kp in kps_2d]
            mean_conf = float(np.mean(scores)) if scores else 0.0

            return FrameObservations(
                frame_index=frame_index,
                timestamp=timestamp_sec,
                keypoints_2d=kps_2d,
                landmarks_3d=lms_3d,
                raw_confidence=mean_conf,
                detector_name="mediapipe"
            )

        except Exception as e:
            sys.stderr.write(f"[TexMotion Warning] MediaPipe detection error: {e}\n")
            return None

    def close(self):
        if self._landmarker is not None:
            try:
                self._landmarker.close()
            except Exception:
                pass
            self._landmarker = None
        self._mode = None
        self._last_timestamp_ms = -1
        self.is_initialized = False
