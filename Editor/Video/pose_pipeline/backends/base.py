"""
base.py - Abstract base class and capabilities for pose estimation backends.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Sequence, List, Dict, Any
import numpy as np
from ..observations import FrameObservations


@dataclass
class BackendCapabilities:
    """Describes features supported by a specific pose backend."""
    name: str
    has_2d: bool = True
    has_3d: bool = False
    has_visibility: bool = True
    has_presence: bool = False
    is_temporal: bool = False
    keypoint_count: int = 33
    supports_directml: bool = False
    supports_cuda: bool = False


class PoseBackend(ABC):
    """Abstract interface for video frame pose estimation backends."""

    def __init__(self, name: str):
        self.name = name
        self.is_initialized = False
        # Sequence backends (WHAM/HMR2/HybrIK) need context on both sides of an
        # occlusion.  Frame backends leave this false and keep the low-latency
        # path unchanged.
        self.supports_sequence = False

    @abstractmethod
    def is_available(self) -> bool:
        """Returns True if the backend dependencies and runtime are available."""
        pass

    @abstractmethod
    def initialize(self) -> bool:
        """Initializes model weights, sessions, and resources. Returns True on success."""
        pass

    @abstractmethod
    def detect(self, frame_bgr: np.ndarray, timestamp_sec: float, frame_index: int = 0) -> Optional[FrameObservations]:
        """Runs pose estimation on a single BGR image frame."""
        pass

    @abstractmethod
    def get_capabilities(self) -> BackendCapabilities:
        """Returns capabilities descriptor."""
        pass

    def close(self):
        """Releases all resources and model sessions."""
        self.is_initialized = False

    def detect_sequence(
        self,
        frames_bgr: Sequence[np.ndarray],
        timestamps_sec: Sequence[float],
        frame_indices: Optional[Sequence[int]] = None,
    ) -> List[Optional[FrameObservations]]:
        """Runs the backend over a sequence, using frame inference by default.

        Quality backends override this method to run a temporal model.  The
        default implementation intentionally keeps the interface useful for
        adapters that only expose per-frame inference.
        """
        indices = frame_indices if frame_indices is not None else range(len(frames_bgr))
        return [
            self.detect(frame, float(ts), int(index))
            if frame is not None else None
            for frame, ts, index in zip(frames_bgr, timestamps_sec, indices)
        ]

    def get_metadata(self) -> Dict[str, Any]:
        """Returns serializable backend provenance for motion export."""
        metadata = {
            "name": self.name,
            "supportsSequence": bool(self.supports_sequence),
            "initialized": bool(self.is_initialized),
            "selectedBackend": self.name,
            "backendReady": bool(self.is_initialized),
        }
        fallback_from = getattr(self, "fallback_from", None)
        fallback_reason = getattr(self, "fallback_reason", None)
        if fallback_from:
            metadata["fallbackFrom"] = fallback_from
        if fallback_reason:
            metadata["fallbackReason"] = fallback_reason
        fallback_backend_metadata = getattr(self, "fallback_backend_metadata", None)
        if isinstance(fallback_backend_metadata, dict):
            # Retain the failed quality backend's diagnostics alongside the
            # concrete fallback.  Promote the high-value asset/preflight
            # fields for consumers that do not inspect nested metadata, while
            # leaving ``selectedBackend``/``backendReady`` unambiguous for the
            # actual fallback object.
            metadata["fallbackBackendMetadata"] = dict(fallback_backend_metadata)
            for key in (
                "preflightStatus",
                "preflightPhase",
                "preflightFrames",
                "preflightError",
                "missingAssets",
                "incompatibleAssets",
                "assetDiagnostics",
            ):
                if key in fallback_backend_metadata:
                    metadata[key] = fallback_backend_metadata[key]
        return metadata

    def __enter__(self):
        if not self.is_initialized:
            self.initialize()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
