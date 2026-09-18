"""
Detector backends for 2D and 3D pose extraction with automatic fallback.
"""

import sys
from typing import Any, Optional, Tuple
from .base import PoseBackend, BackendCapabilities
from .mediapipe_backend import MediaPipeBackend, check_mediapipe_available
from .rtmpose_backend import RTMPoseBackend, check_backend_available, verify_model_manifest
from .pytorch_backend import (
    PyTorchBackendConfig,
    PyTorchPoseBackend,
    TorchPoseBackend,
    check_pytorch_backend_available,
)


def create_pose_backend(
    preferred: str = "auto",
    model_path: Optional[str] = None,
    model_complexity: int = 1,
    min_detection_confidence: float = 0.5,
    min_tracking_confidence: float = 0.5,
    pytorch_model_path: Optional[str] = None,
    pytorch_adapter_module: Optional[str] = None,
    pytorch_device: str = "auto",
    max_sequence_frames: int = 0,
    model_directory: Optional[str] = None,
    strict_quality: bool = False,
    asset_manifest_path: Optional[str] = None,
    wham_asset_paths: Optional[dict] = None,
    image_feature_backbone_path: Optional[str] = None,
    image_feature_path: Optional[str] = None,
    camera_model_path: Optional[str] = None,
    camera_path: Optional[str] = None,
    dpvo_model_path: Optional[str] = None,
    dpvo_path: Optional[str] = None,
    wham_preprocess_directory: Optional[str] = None,
    image_feature_extractor: Any = None,
    image_feature_runner: Any = None,
    image_feature_runner_config: Any = None,
    image_feature_runner_module: Optional[str] = None,
    image_feature_runner_path: Optional[str] = None,
    image_feature_checkpoint_path: Optional[str] = None,
    image_feature_model_definition: Any = None,
    image_feature_config_path: Any = None,
    image_feature_model_factory: Any = None,
    vitpose_runner: Any = None,
    vitpose_runner_config: Any = None,
    vitpose_runner_module: Optional[str] = None,
    vitpose_checkpoint_path: Optional[str] = None,
    vitpose_model_definition: Any = None,
    vitpose_config_path: Any = None,
    vitpose_model_factory: Any = None,
    runtime_path: Optional[str] = None,
    body_model_path: Optional[str] = None,
) -> PoseBackend:
    """
    Factory function for pose estimation backends.
    Gracefully falls back to MediaPipe if RTMPose weights or ONNX Runtime are unavailable.
    Explicit quality requests can opt into strict mode so a failed WHAM/HMR2/
    HybrIK initialization is surfaced to the caller instead of being hidden by
    the lightweight fallback.
    """
    pref = preferred.lower().strip()
    fallback_reason = None
    fallback_from = None

    # Quality backends are intentionally opt-in.  They are imported lazily.
    # Callers that explicitly request one from the video UI can set
    # ``strict_quality`` so a missing adapter cannot silently produce a
    # MediaPipe result labelled as WHAM/HMR2/HybrIK.
    quality_kind = pref if pref in ("pytorch", "wham", "hmr2", "hybrik") else None
    if quality_kind:
        quality_display_name = "PyTorch quality" if quality_kind == "pytorch" else quality_kind.upper()
        quality = PyTorchPoseBackend(
            kind=quality_kind,
            model_path=pytorch_model_path,
            adapter_module=pytorch_adapter_module,
            device=pytorch_device,
            max_sequence_frames=max_sequence_frames,
            model_directory=model_directory,
            runtime_path=runtime_path,
            body_model_path=body_model_path,
            asset_manifest_path=asset_manifest_path,
            wham_asset_paths=wham_asset_paths,
            image_feature_backbone_path=image_feature_backbone_path,
            image_feature_path=image_feature_path,
            camera_model_path=camera_model_path,
            camera_path=camera_path,
            dpvo_model_path=dpvo_model_path,
            dpvo_path=dpvo_path,
            wham_preprocess_directory=wham_preprocess_directory,
            image_feature_extractor=image_feature_extractor,
            image_feature_runner=image_feature_runner,
            image_feature_runner_config=image_feature_runner_config,
            image_feature_runner_module=image_feature_runner_module,
            image_feature_runner_path=image_feature_runner_path,
            image_feature_checkpoint_path=image_feature_checkpoint_path,
            image_feature_model_definition=image_feature_model_definition,
            image_feature_config_path=image_feature_config_path,
            image_feature_model_factory=image_feature_model_factory,
            vitpose_runner=vitpose_runner,
            vitpose_runner_config=vitpose_runner_config,
            vitpose_runner_module=vitpose_runner_module,
            vitpose_checkpoint_path=vitpose_checkpoint_path,
            vitpose_model_definition=vitpose_model_definition,
            vitpose_config_path=vitpose_config_path,
            vitpose_model_factory=vitpose_model_factory,
        )
        available, reason = check_pytorch_backend_available(
            quality_kind,
            pytorch_model_path,
            pytorch_adapter_module,
            model_directory=model_directory,
            asset_manifest_path=asset_manifest_path,
            wham_asset_paths=wham_asset_paths,
        )
        if not available:
            # The factory rejects missing dependencies/assets before calling
            # initialize().  Mark that gate explicitly so the fallback object
            # can export a failed quality preflight instead of ``not_run``.
            record_failure = getattr(quality, "record_preflight_failure", None)
            if callable(record_failure):
                record_failure(reason, phase="availability")
        if available and quality.initialize():
            return quality
        init_reason = getattr(quality, "_last_error", None)
        detail = init_reason or reason
        # Preserve the failed quality backend's complete preflight/asset
        # report on the concrete fallback object.  Closing and discarding the
        # quality instance without this snapshot makes the exported result
        # look like an unexplained MediaPipe extraction.
        try:
            quality_failure_metadata = dict(quality.get_metadata() or {})
        except Exception as metadata_error:
            quality_failure_metadata = {
                "selectedBackend": quality_kind,
                "preflightStatus": "failed",
                "preflightError": str(metadata_error),
            }
        if strict_quality:
            quality.close()
            raise RuntimeError(
                f"Requested {quality_display_name} backend could not initialize: {detail}"
            )
        fallback_from = quality_kind
        fallback_reason = (
            f"Requested {quality_display_name} backend could not initialize: {detail}"
        )
        sys.stderr.write(
            f"[TexMotion Warning] {quality_display_name} backend not available ({detail}). "
            "Falling back to MediaPipe.\n"
        )
        quality.close()

    if pref == "rtmpose":
        is_avail, reason = check_backend_available(model_path, model_directory=model_directory)
        if is_avail:
            backend = RTMPoseBackend(
                model_path=model_path,
                model_directory=model_directory,
                allow_download=False,
            )
            if backend.initialize():
                return backend
            fallback_from = "rtmpose"
            fallback_reason = "Requested RTMPose backend failed to initialize."
            sys.stderr.write(f"[TexMotion Warning] {fallback_reason} Falling back to MediaPipe.\n")
        else:
            fallback_from = "rtmpose"
            fallback_reason = f"Requested RTMPose backend is unavailable: {reason}"
            sys.stderr.write(f"[TexMotion Warning] {fallback_reason} Falling back to MediaPipe.\n")

    elif pref == "auto":
        is_avail, _ = check_backend_available(model_path, model_directory=model_directory)
        if is_avail:
            backend = RTMPoseBackend(
                model_path=model_path,
                model_directory=model_directory,
                allow_download=False,
            )
            if backend.initialize():
                return backend

    # MediaPipe fallback / default
    mp_backend = MediaPipeBackend(
        model_complexity=model_complexity,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
        model_directory=model_directory,
        allow_download=False,
    )
    if not mp_backend.initialize():
        raise RuntimeError("MediaPipe backend failed to initialize.")
    if fallback_reason:
        # Keep provenance on the concrete fallback object.  The extractor
        # serializes these fields so the UI can explain why the selected
        # quality backend was not used instead of showing an ambiguous
        # MediaPipe result with no context.
        setattr(mp_backend, "fallback_from", fallback_from)
        setattr(mp_backend, "fallback_reason", fallback_reason)
        if quality_kind:
            setattr(mp_backend, "fallback_backend_metadata", quality_failure_metadata)
    return mp_backend


__all__ = [
    "PoseBackend",
    "BackendCapabilities",
    "MediaPipeBackend",
    "RTMPoseBackend",
    "PyTorchBackendConfig",
    "PyTorchPoseBackend",
    "TorchPoseBackend",
    "create_pose_backend",
    "check_backend_available",
    "check_pytorch_backend_available",
    "check_mediapipe_available",
    "verify_model_manifest",
]
