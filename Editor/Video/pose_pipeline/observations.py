"""
observations.py - Immutable observation data structures for TexMotion pose extraction.

Provides immutable frame and keypoint representations that separate raw detector
measurements from downstream kinematics, inpainting, and temporal optimizations.
"""

from dataclasses import dataclass, field
from enum import IntEnum
from typing import List, Tuple, Optional, Dict, Any
import numpy as np


class ObservationStatus(IntEnum):
    """Observation provenance state for an individual keypoint."""
    MISSING = 0          # Not detected / decode failed
    OBSERVED = 1         # Confidently detected from image pixels
    OCCLUDED = 2         # Present in scene but occluded by body / objects
    PREDICTED = 3        # Inpainted or temporally extrapolated
    USER_CONSTRAINED = 4 # Manually pinned or overridden by user


@dataclass(frozen=True)
class Keypoint2D:
    """Immutable 2D keypoint in normalized coordinates [0, 1]."""
    x: float
    y: float
    score: float = 1.0
    visibility: float = 1.0
    presence: float = 1.0
    status: ObservationStatus = ObservationStatus.OBSERVED

    def to_array(self) -> np.ndarray:
        return np.array([self.x, self.y], dtype=np.float32)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "x": float(self.x),
            "y": float(self.y),
            "score": float(self.score),
            "visibility": float(self.visibility),
            "presence": float(self.presence),
            "status": int(self.status)
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Keypoint2D":
        return cls(
            x=float(data.get("x", 0.0)),
            y=float(data.get("y", 0.0)),
            score=float(data.get("score", 1.0)),
            visibility=float(data.get("visibility", 1.0)),
            presence=float(data.get("presence", 1.0)),
            status=ObservationStatus(data.get("status", int(ObservationStatus.OBSERVED)))
        )


@dataclass(frozen=True)
class Keypoint3D:
    """Immutable 3D keypoint in meters relative to mid-hip or camera root."""
    x: float
    y: float
    z: float
    score: float = 1.0
    visibility: float = 1.0
    presence: float = 1.0
    status: ObservationStatus = ObservationStatus.OBSERVED
    uncertainty_radius: float = 0.0

    def to_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z], dtype=np.float32)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "x": float(self.x),
            "y": float(self.y),
            "z": float(self.z),
            "score": float(self.score),
            "visibility": float(self.visibility),
            "presence": float(self.presence),
            "status": int(self.status),
            "uncertaintyRadius": float(self.uncertainty_radius)
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Keypoint3D":
        return cls(
            x=float(data.get("x", 0.0)),
            y=float(data.get("y", 0.0)),
            z=float(data.get("z", 0.0)),
            score=float(data.get("score", 1.0)),
            visibility=float(data.get("visibility", 1.0)),
            presence=float(data.get("presence", 1.0)),
            status=ObservationStatus(data.get("status", int(ObservationStatus.OBSERVED))),
            uncertainty_radius=float(data.get("uncertaintyRadius", 0.0))
        )


@dataclass(frozen=True)
class UncertaintyInterval:
    """Represents a time interval with ambiguous or occluded pose inference."""
    start_frame: int
    end_frame: int
    reason: str  # e.g., "leg_crossing_ambiguity", "arm_self_occlusion", "low_confidence"
    confidence: float
    recommended_action: str = ""  # e.g., "swap_crossing", "fix_arm_behind"

    def contains_frame(self, frame: int) -> bool:
        return self.start_frame <= frame <= self.end_frame

    def to_dict(self) -> Dict[str, Any]:
        return {
            "startFrame": int(self.start_frame),
            "endFrame": int(self.end_frame),
            "reason": str(self.reason),
            "confidence": float(self.confidence),
            "recommendedAction": str(self.recommended_action)
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UncertaintyInterval":
        return cls(
            start_frame=int(data.get("startFrame", 0)),
            end_frame=int(data.get("endFrame", 0)),
            reason=str(data.get("reason", "")),
            confidence=float(data.get("confidence", 0.0)),
            recommended_action=str(data.get("recommendedAction", ""))
        )


@dataclass
class FrameObservations:
    """
    Observation container for a single video sample frame.
    Separates 2D raw detector output from estimated 3D landmarks.
    """
    frame_index: int
    timestamp: float
    keypoints_2d: List[Keypoint2D] = field(default_factory=list)
    landmarks_3d: List[Keypoint3D] = field(default_factory=list)
    raw_confidence: float = 0.0
    detector_name: str = "unknown"
    is_synthetic: bool = False
    has_crossing_hazard: bool = False
    has_arm_occlusion_hazard: bool = False
    # Optional provenance emitted by a temporal/research backend.  Keeping it
    # on the observation prevents uncertainty from being lost before the
    # sequence fit and JSON export stages.  Existing callers can omit it.
    backend_metadata: Dict[str, Any] = field(default_factory=dict)
    uncertainty_intervals: List[UncertaintyInterval] = field(default_factory=list)

    @property
    def num_keypoints(self) -> int:
        return len(self.keypoints_2d)

    def get_2d_array(self) -> np.ndarray:
        """Returns (N, 2) numpy array of 2D coordinates."""
        if not self.keypoints_2d:
            return np.zeros((0, 2), dtype=np.float32)
        return np.array([[kp.x, kp.y] for kp in self.keypoints_2d], dtype=np.float32)

    def get_3d_array(self) -> np.ndarray:
        """Returns (N, 3) numpy array of 3D coordinates."""
        if not self.landmarks_3d:
            return np.zeros((0, 3), dtype=np.float32)
        return np.array([[lm.x, lm.y, lm.z] for lm in self.landmarks_3d], dtype=np.float32)

    def get_visibilities(self) -> np.ndarray:
        """Returns (N,) numpy array of visibility scores."""
        if not self.keypoints_2d:
            return np.zeros((0,), dtype=np.float32)
        return np.array([kp.visibility for kp in self.keypoints_2d], dtype=np.float32)

    def get_confidences(self) -> np.ndarray:
        """Returns (N,) numpy array of keypoint detector scores."""
        if not self.keypoints_2d:
            return np.zeros((0,), dtype=np.float32)
        return np.array([kp.score for kp in self.keypoints_2d], dtype=np.float32)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "frameIndex": self.frame_index,
            "timestamp": float(self.timestamp),
            "keypoints2d": [kp.to_dict() for kp in self.keypoints_2d],
            "landmarks3d": [lm.to_dict() for lm in self.landmarks_3d],
            "rawConfidence": float(self.raw_confidence),
            "detectorName": self.detector_name,
            "isSynthetic": bool(self.is_synthetic),
            "hasCrossingHazard": bool(self.has_crossing_hazard),
            "hasArmOcclusionHazard": bool(self.has_arm_occlusion_hazard),
            "backendMetadata": dict(self.backend_metadata),
            "uncertaintyIntervals": [u.to_dict() for u in self.uncertainty_intervals]
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FrameObservations":
        return cls(
            frame_index=int(data.get("frameIndex", 0)),
            timestamp=float(data.get("timestamp", 0.0)),
            keypoints_2d=[Keypoint2D.from_dict(k) for k in data.get("keypoints2d", [])],
            landmarks_3d=[Keypoint3D.from_dict(l) for l in data.get("landmarks3d", [])],
            raw_confidence=float(data.get("rawConfidence", 0.0)),
            detector_name=str(data.get("detectorName", "unknown")),
            is_synthetic=bool(data.get("isSynthetic", False)),
            has_crossing_hazard=bool(data.get("hasCrossingHazard", False)),
            has_arm_occlusion_hazard=bool(data.get("hasArmOcclusionHazard", False)),
            backend_metadata=dict(data.get("backendMetadata", {})),
            uncertainty_intervals=[UncertaintyInterval.from_dict(u) for u in data.get("uncertaintyIntervals", [])]
        )
