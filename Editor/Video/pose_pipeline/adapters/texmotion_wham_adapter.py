"""Bundled WHAM adapter bridge for TexMotion's temporal quality backend.

The WHAM checkpoint is distributed as a research PyTorch state-dict archive,
so it cannot be passed directly to the generic TorchScript loader.  This bridge
keeps the model-specific integration lazy and gives projects a stable adapter
path in the Settings catalog.  When a local checkpoint is present it first
tries TexMotion's built-in native WHAM stages (the checkpoint-compatible
MotionEncoder, trajectory/SMPL decoders, with local RTMPose 2D input and an
offline MediaPipe fallback).
Set ``TEXMOTION_WHAM_ADAPTER_IMPL`` to an installed full WHAM integration
module when the official repository and its optional ViTPose, DPVO, and SMPL
assets are available.  Such a module should expose ``create_backend(config)``
and return an object with ``infer_sequence(frames_bgr, timestamps_sec,
frame_indices)``.

Keeping this bridge in the downloaded model cache makes a WHAM download
self-describing and prevents a missing adapter path from silently selecting a
different backend.  The default lightweight TexMotion runtime never imports
torch or this module.
"""

from __future__ import annotations

import importlib
import os
from typing import Any, Dict, Optional, Sequence


def _implementation_name(config: Dict[str, Any]) -> Optional[str]:
    explicit = config.get("adapterImplementation") or config.get("adapter_impl")
    # The package ships native checkpoint-compatible WHAM stages. Use them by
    # default so downloading the checkpoint + bundled bridge is a runnable
    # local path; an external full implementation can still override this with
    # TEXMOTION_WHAM_ADAPTER_IMPL or adapter_impl.
    return explicit or os.environ.get("TEXMOTION_WHAM_ADAPTER_IMPL") or "pose_pipeline.adapters.wham_native"


def _load_implementation(name: str, config: Dict[str, Any]) -> Any:
    module = importlib.import_module(name)
    factory = getattr(module, "create_backend", None) or getattr(module, "create_adapter", None)
    if factory is not None:
        return factory(config)
    cls = getattr(module, "WHAMAdapter", None) or getattr(module, "Adapter", None)
    if cls is not None:
        return cls(config)
    return module


class WHAMAdapter:
    """Thin contract adapter around a project-provided WHAM implementation."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = dict(config or {})
        self.model_path = self.config.get("modelPath") or self.config.get("model_path")
        self.device = self.config.get("device", "cpu")
        self._implementation: Any = None
        self._error: Optional[str] = None
        self._implementation_name: Optional[str] = None

    def initialize(self) -> bool:
        name = _implementation_name(self.config)
        if not name:
            self._error = (
                "WHAM checkpoint is not available for the bundled native temporal core. "
                "Configure a local checkpoint path, or set TEXMOTION_WHAM_ADAPTER_IMPL to "
                "explicitly use a full local WHAM adapter exposing create_backend/create_adapter "
                "and infer_sequence (or infer_frame)."
            )
            return False

        try:
            self._implementation = _load_implementation(name, self.config)
            self._implementation_name = name
            initialize = getattr(self._implementation, "initialize", None)
            if initialize is not None and initialize() is False:
                implementation_error = getattr(self._implementation, "error", None) or getattr(
                    self._implementation, "_error", None
                )
                self._error = implementation_error or "Configured WHAM implementation failed to initialize."
                self._implementation = None
                self._implementation_name = None
                return False
            if not any(
                callable(getattr(self._implementation, method, None))
                for method in ("infer_sequence", "predict_sequence", "infer_frame", "predict")
            ):
                self._error = (
                    "Configured WHAM implementation must expose infer_sequence, predict_sequence, "
                    "infer_frame, or predict."
                )
                self._implementation = None
                self._implementation_name = None
                return False
            return True
        except Exception as exc:
            self._error = str(exc)
            self._implementation = None
            self._implementation_name = None
            return False

    def infer_sequence(
        self,
        frames_bgr: Sequence[Any],
        timestamps_sec: Sequence[float],
        frame_indices: Sequence[int],
    ) -> Any:
        if self._implementation is None:
            raise RuntimeError(self._error or "WHAM adapter is not initialized")
        method = getattr(self._implementation, "infer_sequence", None) or getattr(
            self._implementation, "predict_sequence", None
        )
        if method is None:
            method = getattr(self._implementation, "infer_frame", None) or getattr(
                self._implementation, "predict", None
            )
            if method is None:
                raise RuntimeError("WHAM implementation exposes no inference method")
            return [method(frame, timestamp, index) for frame, timestamp, index in zip(
                frames_bgr, timestamps_sec, frame_indices
            )]
        return method(frames_bgr, timestamps_sec, frame_indices)

    def set_initialization_from_observation(self, observation: Any) -> bool:
        """Forward the optional first-frame seed to the concrete WHAM core.

        The downloaded bridge is the adapter object seen by
        ``PyTorchPoseBackend``; without this forwarding seam the native
        implementation would always keep its neutral embedded SMPL seed even
        though the extractor had a valid MediaPipe camera-space observation.
        External/full WHAM implementations that do not expose the seam simply
        retain their own initialization contract.
        """
        if self._implementation is None:
            return False
        setter = getattr(self._implementation, "set_initialization_from_observation", None)
        if not callable(setter):
            return False
        try:
            return bool(setter(observation))
        except Exception as exc:
            self._error = str(exc)
            return False

    def get_mediapipe_auxiliary(self) -> Any:
        """Forward the native WHAM MediaPipe detector ownership seam."""
        if self._implementation is None:
            return None
        getter = getattr(self._implementation, "get_mediapipe_auxiliary", None)
        if callable(getter):
            candidate = getter()
            if candidate is not None:
                return candidate
        # Keep old cached/native implementations safe as well.  The adapter
        # may predate this forwarding method while still owning a live
        # MediaPipe detector on its implementation object.
        candidate = getattr(self._implementation, "_detector", None)
        name = str(getattr(candidate, "name", "") or "").strip().lower()
        detect = getattr(candidate, "detect", None)
        if name in ("mediapipe", "mediapipe_pose", "mediapipe_3d") and callable(detect):
            return candidate
        return None

    def close(self) -> None:
        if self._implementation is not None:
            close = getattr(self._implementation, "close", None)
            if close is not None:
                close()
        self._implementation = None
        self._implementation_name = None

    def get_metadata(self) -> Dict[str, Any]:
        metadata = {
            "adapter": "texmotion_wham_adapter",
            "adapterImplementation": self._implementation_name or _implementation_name(self.config),
            "modelPath": self.model_path,
            "device": self.device,
            "contract": "infer_sequence",
            "error": self._error,
        }
        # Surface native-core provenance (checkpoint, input detector, and
        # intentionally omitted official stages) through the bridge itself as
        # well as through per-frame metadata in PyTorchPoseBackend.
        if self._implementation is not None:
            getter = getattr(self._implementation, "get_metadata", None)
            if callable(getter):
                try:
                    details = getter()
                    if isinstance(details, dict):
                        metadata.update(details)
                except Exception:
                    pass
        return metadata


def create_backend(config: Optional[Dict[str, Any]] = None) -> WHAMAdapter:
    return WHAMAdapter(config)


__all__ = ["WHAMAdapter", "create_backend"]
