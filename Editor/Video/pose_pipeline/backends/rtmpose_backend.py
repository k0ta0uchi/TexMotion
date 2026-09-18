"""
rtmpose_backend.py - High-precision 2D keypoint backend using ONNX Runtime.

Supports RTMPose and DWPose ONNX models (17-joint COCO and WholeBody).
Provides robust 2D limb overlap and crossed-leg detection to overcome
MediaPipe's single-view limb collapse.
Includes offline preflight checks, manifest verification, and CPU/GPU fallback.
"""

import os
import gc
from typing import Optional, List, Tuple, Dict, Any
import numpy as np
from .base import PoseBackend, BackendCapabilities
from ..observations import FrameObservations, Keypoint2D, Keypoint3D, ObservationStatus

try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False

try:
    import onnxruntime as ort
    ORT_AVAILABLE = True
except ImportError:
    ORT_AVAILABLE = False
    ort = None

# COCO-17 to MediaPipe-33 joint index mapping
COCO_TO_MEDIAPIPE_MAP = {
    0: 0,    # nose -> nose
    1: 2,    # l_eye -> left_eye
    2: 5,    # r_eye -> right_eye
    3: 7,    # l_ear -> left_ear
    4: 8,    # r_ear -> right_ear
    5: 11,   # l_shoulder -> left_shoulder
    6: 12,   # r_shoulder -> right_shoulder
    7: 13,   # l_elbow -> left_elbow
    8: 14,   # r_elbow -> right_elbow
    9: 15,   # l_wrist -> left_wrist
    10: 16,  # r_wrist -> right_wrist
    11: 23,  # l_hip -> left_hip
    12: 24,  # r_hip -> right_hip
    13: 25,  # l_knee -> left_knee
    14: 26,  # r_knee -> right_knee
    15: 27,  # l_ankle -> left_ankle
    16: 28,  # r_ankle -> right_ankle
}


def check_backend_available(
    model_path: Optional[str] = None,
    model_directory: Optional[str] = None,
) -> Tuple[bool, str]:
    """
    Preflight validation function verifying dependencies and model readiness
    for completely offline operation.
    """
    if not OPENCV_AVAILABLE:
        return False, "OpenCV (cv2) is not installed."
    if not ORT_AVAILABLE:
        return False, "ONNX Runtime (onnxruntime) is not installed."

    backend = RTMPoseBackend(model_path=model_path, model_directory=model_directory)
    resolved = backend._resolve_model_path()
    if not resolved or not os.path.exists(resolved):
        return False, f"RTMPose ONNX model weights not found at: {model_path or 'default paths'}"

    # Verify model file is non-empty
    try:
        size = os.path.getsize(resolved)
        if size < 1024:
            return False, f"Model file is corrupted or empty ({size} bytes)."
    except Exception as e:
        return False, f"Cannot access model file: {e}"

    return True, f"RTMPose backend available with weights at {resolved}"


def verify_model_manifest(model_path: str) -> Tuple[bool, Dict[str, Any]]:
    """
    Validates model manifest: verifies file size, accessibility, and session loadability.
    """
    if not os.path.exists(model_path):
        return False, {"error": "Model file does not exist", "path": model_path}

    size = os.path.getsize(model_path)
    manifest = {
        "path": os.path.abspath(model_path),
        "sizeBytes": size,
        "isNonEmpty": size > 1024,
    }

    if not ORT_AVAILABLE:
        manifest["onnxruntimeAvailable"] = False
        return False, manifest

    manifest["onnxruntimeAvailable"] = True
    manifest["availableProviders"] = ort.get_available_providers()
    return True, manifest


class RTMPoseBackend(PoseBackend):
    """
    RTMPose/DWPose 2D keypoint detection backend powered by ONNX Runtime.
    Runs on CPU, DirectML, or CUDA depending on local availability.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        input_size: Tuple[int, int] = (256, 192),
        model_directory: Optional[str] = None,
        allow_download: bool = True,
    ):
        super().__init__("rtmpose")
        self.model_path = model_path
        self.input_size = input_size  # (height, width)
        self.model_directory = model_directory
        # Quality adapters (for example the bundled WHAM temporal core) must
        # remain offline.  Keep the historical auto-download default for the
        # standalone RTMPose path, while allowing callers to opt out.
        self.allow_download = bool(allow_download)
        self._session = None
        self._input_name = None
        self._output_names = None
        self._active_provider = None

    def is_available(self) -> bool:
        if not (ORT_AVAILABLE and OPENCV_AVAILABLE):
            return False
        path = self._resolve_model_path()
        return path is not None and os.path.exists(path) and os.path.getsize(path) > 1024

    MODEL_URL = "https://huggingface.co/bukuroo/RTMPose-ONNX/resolve/main/rtmpose-m.onnx"
    MODEL_NAME = "rtmpose-m.onnx"

    def _ensure_model(self) -> Optional[str]:
        """
        Attempts to find or automatically download the RTMPose-m ONNX model
        if not already present locally.
        """
        import urllib.request
        import sys

        resolved = self._resolve_model_path()
        if resolved and os.path.exists(resolved) and os.path.getsize(resolved) > 1024 * 1024:
            return resolved

        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        app_data = os.environ.get("APPDATA", "")
        default_video_cache = (
            os.path.join(app_data, "TexMotion", "Models", "Video") if app_data else ""
        )
        target_dir = self.model_directory or os.environ.get("TEXMOTION_VIDEO_MODEL_DIR")
        if not target_dir:
            # Match the Unity Settings default while retaining the package-local
            # cache as a fallback for source checkouts and older installations.
            target_dir = default_video_cache or os.path.join(base_dir, "models")
        os.makedirs(target_dir, exist_ok=True)
        target_file = os.path.join(target_dir, self.MODEL_NAME)
        tmp_file = target_file + ".tmp"

        try:
            sys.stderr.write(f"[TexMotion] RTMPose weights not found. Auto-downloading {self.MODEL_NAME} (~52MB)...\n")
            sys.stderr.flush()

            req = urllib.request.Request(
                self.MODEL_URL,
                headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) TexMotion/1.0"}
            )
            with urllib.request.urlopen(req, timeout=120) as resp, open(tmp_file, "wb") as out_f:
                total_len = int(resp.headers.get("Content-Length", 0))
                downloaded = 0
                chunk_size = 1024 * 64
                while True:
                    chunk = resp.read(chunk_size)
                    if not chunk:
                        break
                    out_f.write(chunk)
                    downloaded += len(chunk)
                    if total_len > 0 and downloaded % (1024 * 1024 * 5) < chunk_size:
                        pct = (downloaded / total_len) * 100.0
                        sys.stderr.write(f"[TexMotion] Downloaded {downloaded // (1024*1024)}MB / {total_len // (1024*1024)}MB ({pct:.1f}%)\n")
                        sys.stderr.flush()

            if os.path.exists(tmp_file) and os.path.getsize(tmp_file) > 1024 * 1024:
                if os.path.exists(target_file):
                    os.remove(target_file)
                os.replace(tmp_file, target_file)
                sys.stderr.write(f"[TexMotion] Successfully downloaded RTMPose model to {target_file}\n")
                sys.stderr.flush()
                return target_file
        except Exception as e:
            sys.stderr.write(f"[TexMotion Warning] RTMPose auto-download failed: {e}\n")
            sys.stderr.write(f"[TexMotion Warning] You can manually download {self.MODEL_URL} and place it at {target_file}\n")
            sys.stderr.flush()
        finally:
            if os.path.exists(tmp_file):
                try:
                    os.remove(tmp_file)
                except Exception:
                    pass
        return None

    def _resolve_model_path(self) -> Optional[str]:
        if self.model_path:
            # When an explicit model_path is passed, strictly resolve only that path.
            if os.path.exists(self.model_path) and os.path.getsize(self.model_path) > 1024:
                return os.path.abspath(self.model_path)
            return None

        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        configured_dir = self.model_directory or os.environ.get("TEXMOTION_VIDEO_MODEL_DIR", "")
        app_data = os.environ.get("APPDATA", "")
        default_video_cache = (
            os.path.join(app_data, "TexMotion", "Models", "Video") if app_data else ""
        )
        candidates = [
            os.path.join(configured_dir, "rtmpose-m.onnx") if configured_dir else "",
            os.path.join(configured_dir, "dwpose-l.onnx") if configured_dir else "",
            os.path.join(default_video_cache, "rtmpose-m.onnx") if default_video_cache else "",
            os.path.join(default_video_cache, "dwpose-l.onnx") if default_video_cache else "",
            os.path.join(base_dir, "models", "rtmpose-m.onnx"),
            os.path.join(base_dir, "models", "dwpose-l.onnx"),
            os.path.join(os.environ.get("APPDATA", ""), "TexMotion", "Models", "rtmpose-m.onnx"),
            os.path.join(os.path.expanduser("~"), ".cache", "texmotion", "models", "rtmpose-m.onnx"),
            os.path.join(base_dir, "rtmpose-m.onnx"),
            os.path.join(base_dir, "dwpose-l.onnx"),
            os.environ.get("RTMPOSE_MODEL_PATH", "")
        ]
        for c in candidates:
            if c and os.path.exists(c) and os.path.getsize(c) > 1024:
                return os.path.abspath(c)
        return None

    def get_capabilities(self) -> BackendCapabilities:
        has_dml = False
        has_cuda = False
        if ORT_AVAILABLE:
            providers = ort.get_available_providers()
            has_dml = "DmlExecutionProvider" in providers
            has_cuda = "CUDAExecutionProvider" in providers

        return BackendCapabilities(
            name="rtmpose",
            has_2d=True,
            has_3d=False,
            has_visibility=True,
            has_presence=False,
            is_temporal=False,
            keypoint_count=33,
            supports_directml=has_dml,
            supports_cuda=has_cuda
        )

    def initialize(self) -> bool:
        if not (ORT_AVAILABLE and OPENCV_AVAILABLE):
            return False

        path = self._resolve_model_path()
        if not path:
            # Only auto-download if no custom model path was explicitly
            # specified and the caller permits network access.
            if not self.model_path and self.allow_download:
                path = self._ensure_model()
        if not path:
            return False

        available_providers = ort.get_available_providers()
        sess_options = ort.SessionOptions()
        sess_options.enable_mem_pattern = False
        sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

        # Try GPU providers first, then fallback to CPU
        candidate_provider_groups = []
        gpu_providers = []
        if "DmlExecutionProvider" in available_providers:
            gpu_providers.append("DmlExecutionProvider")
        if "CUDAExecutionProvider" in available_providers:
            gpu_providers.append("CUDAExecutionProvider")

        if gpu_providers:
            candidate_provider_groups.append(gpu_providers + ["CPUExecutionProvider"])
        candidate_provider_groups.append(["CPUExecutionProvider"])

        for prov_list in candidate_provider_groups:
            try:
                self._session = ort.InferenceSession(path, sess_options=sess_options, providers=prov_list)
                self._input_name = self._session.get_inputs()[0].name
                self._output_names = [out.name for out in self._session.get_outputs()]
                self.model_path = path
                self._active_provider = self._session.get_providers()[0]
                self.is_initialized = True
                return True
            except Exception:
                self._session = None

        self.is_initialized = False
        return False

    def _preprocess(self, img_bgr: np.ndarray) -> Tuple[np.ndarray, float, float, int, int]:
        h, w = img_bgr.shape[:2]
        target_h, target_w = self.input_size

        scale = min(float(target_w) / max(1, w), float(target_h) / max(1, h))
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        padded = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        pad_x = (target_w - new_w) // 2
        pad_y = (target_h - new_h) // 2
        padded[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized

        rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB).astype(np.float32)
        mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
        std = np.array([58.395, 57.12, 57.375], dtype=np.float32)
        norm = (rgb - mean) / std

        # NCHW format
        tensor = np.transpose(norm, (2, 0, 1))[np.newaxis, :, :, :].astype(np.float32)
        return tensor, scale, scale, pad_x, pad_y

    def detect(self, frame_bgr: np.ndarray, timestamp_sec: float, frame_index: int = 0) -> Optional[FrameObservations]:
        if not self.is_initialized or self._session is None:
            return None
        if frame_bgr is None or frame_bgr.size == 0:
            return None

        try:
            orig_h, orig_w = frame_bgr.shape[:2]
            tensor, scale_x, scale_y, pad_x, pad_y = self._preprocess(frame_bgr)

            outputs = self._session.run(self._output_names, {self._input_name: tensor})
            coords_17, scores_17 = self._decode_simcc(outputs, scale_x, scale_y, pad_x, pad_y, orig_w, orig_h)

            # Map 17 COCO keypoints into 33 MediaPipe slot layout
            kps_2d: List[Keypoint2D] = [
                Keypoint2D(x=0.0, y=0.0, score=0.0, visibility=0.0, presence=0.0, status=ObservationStatus.MISSING)
                for _ in range(33)
            ]
            lms_3d: List[Keypoint3D] = [
                Keypoint3D(x=0.0, y=0.0, z=0.0, score=0.0, visibility=0.0, presence=0.0, status=ObservationStatus.MISSING)
                for _ in range(33)
            ]

            # Compute pelvis reference and scale for MediaPipe World Landmark compatibility (Y=down, origin at pelvis)
            hip_l_x, hip_l_y = coords_17[11]
            hip_r_x, hip_r_y = coords_17[12]
            pelvis_x = float((hip_l_x + hip_r_x) * 0.5)
            pelvis_y = float((hip_l_y + hip_r_y) * 0.5)

            sh_l_x, sh_l_y = coords_17[5]
            sh_r_x, sh_r_y = coords_17[6]
            sh_mid_y = float((sh_l_y + sh_r_y) * 0.5)
            torso_span = max(0.08, abs(pelvis_y - sh_mid_y))
            # Standard adult torso length from pelvis to mid-shoulder is ~0.48m
            scale_m = 0.48 / torso_span

            for coco_idx, mp_idx in COCO_TO_MEDIAPIPE_MAP.items():
                if coco_idx < len(coords_17):
                    x_norm, y_norm = coords_17[coco_idx]
                    score = float(scores_17[coco_idx])
                    status = ObservationStatus.OBSERVED if score >= 0.3 else ObservationStatus.OCCLUDED

                    kps_2d[mp_idx] = Keypoint2D(
                        x=float(x_norm),
                        y=float(y_norm),
                        score=score,
                        visibility=score,
                        presence=score,
                        status=status
                    )
                    # For 3D representation, adhere to MediaPipe World Landmark standard:
                    # Origin at pelvis (0,0,0), X=right, Y=down (positive downwards!), Z=away
                    # This ensures convert_mediapipe_landmarks_to_smplx_3d (which does lm[:,1] = -mp[:,1])
                    # maps head to +Y (up) and feet to -Y (down), eliminating avatar inversion!
                    lms_3d[mp_idx] = Keypoint3D(
                        x=float((x_norm - pelvis_x) * scale_m),
                        y=float((y_norm - pelvis_y) * scale_m),
                        z=0.0,
                        score=score,
                        visibility=score,
                        presence=score,
                        status=ObservationStatus.PREDICTED if score >= 0.3 else ObservationStatus.OCCLUDED
                    )

            mean_conf = float(np.mean(scores_17)) if len(scores_17) > 0 else 0.0

            return FrameObservations(
                frame_index=frame_index,
                timestamp=timestamp_sec,
                keypoints_2d=kps_2d,
                landmarks_3d=lms_3d,
                raw_confidence=mean_conf,
                detector_name="rtmpose"
            )

        except Exception:
            return None

    def _decode_simcc(
        self,
        outputs: List[np.ndarray],
        scale_x: float,
        scale_y: float,
        pad_x: int,
        pad_y: int,
        orig_w: int,
        orig_h: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Decodes SimCC 1D heatmaps into normalized image coordinates [0, 1]
        with subpixel refinement.
        """
        if len(outputs) >= 2 and outputs[0].ndim == 3 and outputs[1].ndim == 3:
            # Output 0 is SimCC-X (1, K, W_sub) or SimCC-Y
            # Match outputs by shape if names are ambiguous
            target_h, target_w = self.input_size
            simcc_x = outputs[0][0]
            simcc_y = outputs[1][0]

            num_kps = simcc_x.shape[0]
            w_sub = simcc_x.shape[-1]
            h_sub = simcc_y.shape[-1]

            split_ratio_x = float(w_sub) / max(1.0, float(target_w))
            split_ratio_y = float(h_sub) / max(1.0, float(target_h))

            loc_x = np.argmax(simcc_x, axis=-1).astype(np.float32) / max(1e-4, split_ratio_x)
            loc_y = np.argmax(simcc_y, axis=-1).astype(np.float32) / max(1e-4, split_ratio_y)

            max_x = np.max(simcc_x, axis=-1)
            max_y = np.max(simcc_y, axis=-1)
            scores = np.minimum(max_x, max_y)

            # Sigmoid if values are raw logits
            if np.any(scores > 1.0) or np.any(scores < 0.0):
                scores = 1.0 / (1.0 + np.exp(-np.clip(scores, -10.0, 10.0)))

            coords = np.zeros((num_kps, 2), dtype=np.float32)
            for k in range(num_kps):
                px = (loc_x[k] - pad_x) / max(1e-5, scale_x)
                py = (loc_y[k] - pad_y) / max(1e-5, scale_y)
                coords[k, 0] = np.clip(px / max(1.0, float(orig_w)), 0.0, 1.0)
                coords[k, 1] = np.clip(py / max(1.0, float(orig_h)), 0.0, 1.0)

            return coords, scores

        # Fallback: direct keypoints format (1, K, 2 or 3)
        kps = outputs[0][0]
        num_kps = kps.shape[0]
        target_h, target_w = self.input_size
        px = (kps[:, 0] - pad_x) / max(1e-5, scale_x)
        py = (kps[:, 1] - pad_y) / max(1e-5, scale_y)
        coords = np.zeros((num_kps, 2), dtype=np.float32)
        coords[:, 0] = np.clip(px / max(1.0, float(orig_w)), 0.0, 1.0)
        coords[:, 1] = np.clip(py / max(1.0, float(orig_h)), 0.0, 1.0)
        scores = kps[:, 2] if kps.shape[1] > 2 else np.ones(num_kps, dtype=np.float32)
        return coords, scores

    def close(self):
        self._session = None
        self.is_initialized = False
        gc.collect()
