"""
face_pipeline.py - MediaPipe Tasks FaceLandmarker pipeline for blendshapes and head rotation.

Extracts:
- 52 ARKit-compatible facial blendshape weights over time.
- Head rigid rotation quaternion [x, y, z, w] (flattened to headRotationsFlat).
- Detection confidences synchronized with body pose timestamps.
Safe fallback when MediaPipe FaceLandmarker is unavailable or face is not detected.
"""

import os
import sys
import tempfile
import urllib.request
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

from .kinematics import matrix_to_quaternion, normalize_quaternion

try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False

try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_tasks
    from mediapipe.tasks.python import vision as mp_vision
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    mp = None
    mp_tasks = None
    mp_vision = None


ARKIT_52_BLENDSHAPES = [
    "browDownLeft", "browDownRight", "browInnerUp", "browOuterUpLeft", "browOuterUpRight",
    "cheekPuff", "cheekSquintLeft", "cheekSquintRight",
    "eyeBlinkLeft", "eyeBlinkRight", "eyeLookDownLeft", "eyeLookDownRight",
    "eyeLookInLeft", "eyeLookInRight", "eyeLookOutLeft", "eyeLookOutRight",
    "eyeLookUpLeft", "eyeLookUpRight", "eyeSquintLeft", "eyeSquintRight",
    "eyeWideLeft", "eyeWideRight",
    "jawForward", "jawLeft", "jawOpen", "jawRight",
    "mouthClose", "mouthDimpleLeft", "mouthDimpleRight", "mouthFrownLeft", "mouthFrownRight",
    "mouthFunnel", "mouthLeft", "mouthLowerDownLeft", "mouthLowerDownRight",
    "mouthPressLeft", "mouthPressRight", "mouthPucker", "mouthRight",
    "mouthRollLower", "mouthRollUpper", "mouthShrugLower", "mouthShrugUpper",
    "mouthSmileLeft", "mouthSmileRight", "mouthStretchLeft", "mouthStretchRight",
    "mouthUpperUpLeft", "mouthUpperUpRight",
    "noseSneerLeft", "noseSneerRight", "tongueOut"
]

DEFAULT_FACE_TASK_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/latest/face_landmarker.task"
)
DEFAULT_FACE_TASK_NAME = "face_landmarker.task"


def locate_or_download_face_model(
    model_directory: Optional[str] = None,
    allow_download: bool = True
) -> Optional[str]:
    """Finds existing face_landmarker.task or downloads it if allowed."""
    configured_dir = model_directory or os.environ.get("TEXMOTION_VIDEO_MODEL_DIR", "")
    app_data = os.environ.get("APPDATA", "")
    default_video_cache = (
        os.path.join(app_data, "TexMotion", "Models", "Video") if app_data else ""
    )
    candidate_dirs = [
        configured_dir,
        default_video_cache,
        os.path.join(os.environ.get("APPDATA", ""), "TexMotion", "Models") if app_data else "",
        os.path.join(os.path.expanduser("~"), ".cache", "texmotion", "models"),
        os.path.join(tempfile.gettempdir(), "texmotion_models"),
        os.path.join(os.path.dirname(__file__), "..", "models"),
        os.path.dirname(os.path.abspath(__file__)),
    ]

    for d in candidate_dirs:
        if not d:
            continue
        p = os.path.join(d, DEFAULT_FACE_TASK_NAME)
        if os.path.exists(p) and os.path.getsize(p) > 1024 * 1024:
            return p

    if not allow_download:
        return None

    target_dir = configured_dir or default_video_cache or (
        candidate_dirs[4] if os.path.exists(candidate_dirs[4]) else tempfile.gettempdir()
    )
    try:
        os.makedirs(target_dir, exist_ok=True)
        target_path = os.path.join(target_dir, DEFAULT_FACE_TASK_NAME)
        tmp_path = target_path + ".tmp"
        sys.stderr.write(f"[TexMotion] Downloading MediaPipe FaceLandmarker model: {DEFAULT_FACE_TASK_NAME}...\n")
        sys.stderr.flush()
        urllib.request.urlretrieve(DEFAULT_FACE_TASK_URL, tmp_path)
        if os.path.exists(tmp_path) and os.path.getsize(tmp_path) > 1024 * 1024:
            if os.path.exists(target_path):
                os.remove(target_path)
            os.rename(tmp_path, target_path)
            return target_path
    except Exception as e:
        sys.stderr.write(f"[TexMotion Warning] Face model download failed: {e}\n")
        sys.stderr.flush()
    return None


class FacePipeline:
    """
    Extracts facial blendshapes and rigid head rotation synchronized with video timestamps.
    Safe against missing models, uninstalled dependencies, and face absence/occlusion.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        model_directory: Optional[str] = None,
        min_detection_confidence: float = 0.3,
        allow_download: bool = False
    ):
        self.min_confidence = float(min_detection_confidence)
        self.is_available = False
        self.detector = None
        self._init_error = None

        if not OPENCV_AVAILABLE:
            self._init_error = "OpenCV (cv2) not available"
            return
        if not MEDIAPIPE_AVAILABLE or mp_vision is None:
            self._init_error = "MediaPipe Tasks vision API not available"
            return

        if model_path is not None:
            resolved_model = model_path
        else:
            resolved_model = locate_or_download_face_model(
                model_directory=model_directory,
                allow_download=allow_download
            )

        if not resolved_model or not os.path.exists(resolved_model):
            self._init_error = "FaceLandmarker model file not found"
            return

        try:
            base_options = mp_tasks.BaseOptions(model_asset_path=resolved_model)
            options = mp_vision.FaceLandmarkerOptions(
                base_options=base_options,
                running_mode=mp_vision.RunningMode.VIDEO,
                output_face_blendshapes=True,
                output_facial_transformation_matrixes=True,
                num_faces=1,
                min_face_detection_confidence=self.min_confidence,
                min_face_presence_confidence=self.min_confidence,
                min_tracking_confidence=self.min_confidence,
            )
            self.detector = mp_vision.FaceLandmarker.create_from_options(options)
            self.is_available = True
        except Exception as e:
            self._init_error = str(e)
            self.detector = None
            self.is_available = False

    def close(self):
        """Release detector resources."""
        if self.detector is not None:
            try:
                self.detector.close()
            except Exception:
                pass
            self.detector = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def process_frames(
        self,
        frames_bgr: List[np.ndarray],
        timestamps: List[float]
    ) -> Dict[str, Any]:
        """
        Extracts face blendshapes and head rotation from a sequence of BGR frames.
        Synchronized with given timestamps (in seconds).
        """
        n_frames = len(timestamps)
        shape_data: Dict[str, List[float]] = {name: [0.0] * n_frames for name in ARKIT_52_BLENDSHAPES}
        head_rotations_flat: List[float] = []
        confidences: List[float] = [0.0] * n_frames

        identity_quat = [0.0, 0.0, 0.0, 1.0]

        if not self.is_available or self.detector is None:
            # Safe neutral fallback
            for _ in range(n_frames):
                head_rotations_flat.extend(identity_quat)
            return build_face_track_payload(timestamps, shape_data, head_rotations_flat, confidences)

        for i in range(n_frames):
            t_sec = timestamps[i]
            timestamp_ms = int(round(t_sec * 1000.0))
            frame_bgr = frames_bgr[i] if i < len(frames_bgr) else None

            if frame_bgr is None or frame_bgr.size == 0:
                head_rotations_flat.extend(identity_quat)
                continue

            try:
                rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                result = self.detector.detect_for_video(mp_image, timestamp_ms)

                if (
                    result is not None
                    and result.face_blendshapes
                    and len(result.face_blendshapes) > 0
                ):
                    face_bs = result.face_blendshapes[0]
                    # Map categories to weights
                    bs_dict = {cat.category_name: float(cat.score) for cat in face_bs}

                    # Confidence estimation: presence or score of key landmarks
                    face_conf = 0.95
                    if hasattr(result, "face_landmarks") and result.face_landmarks:
                        lms = result.face_landmarks[0]
                        vis_scores = [float(getattr(lm, "visibility", 1.0) or 1.0) for lm in lms[:10]]
                        face_conf = float(np.clip(np.mean(vis_scores), 0.0, 1.0)) if vis_scores else 0.95

                    if face_conf >= self.min_confidence:
                        confidences[i] = round(face_conf, 4)
                        for name in ARKIT_52_BLENDSHAPES:
                            w = bs_dict.get(name, 0.0)
                            shape_data[name][i] = round(float(np.clip(w, 0.0, 1.0)), 4)

                        # Head rotation from facial transformation matrix
                        if (
                            hasattr(result, "facial_transformation_matrixes")
                            and result.facial_transformation_matrixes
                            and len(result.facial_transformation_matrixes) > 0
                        ):
                            mat = np.array(result.facial_transformation_matrixes[0], dtype=np.float64)
                            R = mat[:3, :3]
                            quat = matrix_to_quaternion(R)
                            head_rotations_flat.extend([
                                round(float(quat[0]), 5),
                                round(float(quat[1]), 5),
                                round(float(quat[2]), 5),
                                round(float(quat[3]), 5)
                            ])
                        else:
                            head_rotations_flat.extend(identity_quat)
                    else:
                        # Below minimum confidence -> neutral
                        confidences[i] = 0.0
                        head_rotations_flat.extend(identity_quat)
                else:
                    confidences[i] = 0.0
                    head_rotations_flat.extend(identity_quat)
            except Exception as e:
                # Per-frame exception safety
                confidences[i] = 0.0
                head_rotations_flat.extend(identity_quat)

        return build_face_track_payload(timestamps, shape_data, head_rotations_flat, confidences)

    def process_video(
        self,
        video_path: str,
        sample_times: List[float]
    ) -> Optional[Dict[str, Any]]:
        """
        Reads video at sample_times and extracts face track payload.
        Returns None if video cannot be opened or pipeline is unavailable.
        """
        if not OPENCV_AVAILABLE or not os.path.exists(video_path):
            return None

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None

        frames_bgr = []
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

        try:
            for t_sec in sample_times:
                target_frame_idx = int(round(t_sec * fps))
                cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame_idx)
                ret, frame = cap.read()
                if ret and frame is not None:
                    frames_bgr.append(frame)
                else:
                    frames_bgr.append(None)
        finally:
            cap.release()

        return self.process_frames(frames_bgr, sample_times)


def build_face_track_payload(
    timestamps: List[float],
    shape_data: Dict[str, List[float]],
    head_rotations_flat: List[float],
    confidences: List[float]
) -> Dict[str, Any]:
    """Formats face extraction result into VideoMotionData compatible faceTrack dict."""
    shapes_list = [
        {
            "shapeName": name,
            "weights": shape_data.get(name, [0.0] * len(timestamps))
        }
        for name in ARKIT_52_BLENDSHAPES
    ]
    return {
        "version": 1,
        "timestamps": [round(float(t), 4) for t in timestamps],
        "shapes": shapes_list,
        "headRotationsFlat": head_rotations_flat,
        "confidences": [round(float(c), 4) for c in confidences]
    }
