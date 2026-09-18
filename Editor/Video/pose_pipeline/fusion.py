"""WHAM-primary temporal fusion with gated MediaPipe observations.

This module is intentionally isolated from the video extractor and sequence
optimizer.  It provides a small NumPy-only core that can be used by either
the Python quality backend or focused offline tests without importing
MediaPipe, OpenCV, PyTorch, or SciPy.

The important policy is explicit in the implementation: WHAM 3D is the
initial state and remains the strong prior.  MediaPipe contributes only when
its score/visibility/presence and temporal agreement support the observation;
2D reprojection changes image-aligned lateral coordinates while retaining the
WHAM depth hypothesis.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from .camera import (
    CANONICAL_COORDINATE_SYSTEM,
    CameraModel,
    canonical_coordinate_metadata,
    estimate_camera_from_anchors,
)

try:  # The local observation classes are lightweight, but keep this optional.
    from .observations import FrameObservations, ObservationStatus
except Exception:  # pragma: no cover - only relevant to standalone imports
    FrameObservations = None  # type: ignore[assignment,misc]
    ObservationStatus = None  # type: ignore[assignment,misc]


ArrayLike = Union[np.ndarray, Sequence[float], Sequence[Sequence[float]]]


@dataclass
class FusionConfig:
    """Tunable, serializable weights for the isolated fusion core."""

    mediapipe_3d_weight: float = 0.25
    # ``mediapipe_observation_weight`` mirrors the product setting name and,
    # when supplied, takes precedence over mediapipe_3d_weight.
    mediapipe_observation_weight: Optional[float] = None
    reprojection_weight: float = 0.5
    wham_weight: float = 1.0
    temporal_smoothing: float = 0.65
    occluded_gate: float = 0.08
    predicted_gate: float = 0.20
    robust_delta_3d: float = 0.15
    robust_delta_2d: float = 0.05
    max_iterations: int = 4
    step_size: float = 1.0
    min_wham_score: float = 0.10
    observed_weight_threshold: float = 0.20
    disagreement_threshold_meters: float = 0.20
    world_motion_available: bool = False
    scale_is_relative: bool = True
    record_provenance: bool = True
    selection_mode: str = "weighted"

    @classmethod
    def from_dict(cls, data: Optional[Mapping[str, Any]]) -> "FusionConfig":
        """Build config from snake_case or job-style camelCase keys."""

        if not data:
            return cls()
        aliases = {
            "mediapipe3dWeight": "mediapipe_3d_weight",
            "mediaPipe3dWeight": "mediapipe_3d_weight",
            "mediaPipeObservationWeight": "mediapipe_observation_weight",
            "mediapipeObservationWeight": "mediapipe_observation_weight",
            "reprojectionWeight": "reprojection_weight",
            "whamWeight": "wham_weight",
            "temporalSmoothing": "temporal_smoothing",
            "occludedGate": "occluded_gate",
            "predictedGate": "predicted_gate",
            "robustDelta3d": "robust_delta_3d",
            "robustDelta2d": "robust_delta_2d",
            "maxIterations": "max_iterations",
            "stepSize": "step_size",
            "minWhamScore": "min_wham_score",
            "observedWeightThreshold": "observed_weight_threshold",
            "disagreementThresholdMeters": "disagreement_threshold_meters",
            "worldMotionAvailable": "world_motion_available",
            "scaleIsRelative": "scale_is_relative",
            "recordProvenance": "record_provenance",
            "selectionMode": "selection_mode",
        }
        values: Dict[str, Any] = {}
        valid_names = set(cls.__dataclass_fields__)  # type: ignore[attr-defined]
        for key, value in data.items():
            target = aliases.get(key, key)
            if target in valid_names:
                values[target] = value
        return cls(**values)

    def __post_init__(self) -> None:
        self.mediapipe_3d_weight = max(0.0, float(self.mediapipe_3d_weight))
        if self.mediapipe_observation_weight is not None:
            self.mediapipe_observation_weight = max(0.0, float(self.mediapipe_observation_weight))
        self.reprojection_weight = max(0.0, float(self.reprojection_weight))
        self.wham_weight = max(0.0, float(self.wham_weight))
        self.temporal_smoothing = float(np.clip(self.temporal_smoothing, 0.0, 1.0))
        self.occluded_gate = float(np.clip(self.occluded_gate, 0.0, 1.0))
        self.predicted_gate = float(np.clip(self.predicted_gate, 0.0, 1.0))
        self.robust_delta_3d = max(1e-6, float(self.robust_delta_3d))
        self.robust_delta_2d = max(1e-6, float(self.robust_delta_2d))
        self.max_iterations = max(1, int(self.max_iterations))
        self.step_size = float(np.clip(self.step_size, 0.0, 1.0))
        self.min_wham_score = float(np.clip(self.min_wham_score, 0.0, 1.0))
        self.observed_weight_threshold = max(0.0, float(self.observed_weight_threshold))
        self.disagreement_threshold_meters = max(1e-6, float(self.disagreement_threshold_meters))
        self.selection_mode = str(self.selection_mode or "weighted").strip().lower().replace("-", "_")
        if self.selection_mode not in ("weighted", "fallback_only", "stage_aware"):
            raise ValueError("selection_mode must be 'weighted' or 'fallback_only'")

    @property
    def effective_mediapipe_3d_weight(self) -> float:
        return (
            self.mediapipe_observation_weight
            if self.mediapipe_observation_weight is not None
            else self.mediapipe_3d_weight
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mediapipe3dWeight": float(self.effective_mediapipe_3d_weight),
            "reprojectionWeight": float(self.reprojection_weight),
            "whamWeight": float(self.wham_weight),
            "temporalSmoothing": float(self.temporal_smoothing),
            "occludedGate": float(self.occluded_gate),
            "predictedGate": float(self.predicted_gate),
            "robustDelta3d": float(self.robust_delta_3d),
            "robustDelta2d": float(self.robust_delta_2d),
            "maxIterations": int(self.max_iterations),
            "stepSize": float(self.step_size),
            "minWhamScore": float(self.min_wham_score),
            "observedWeightThreshold": float(self.observed_weight_threshold),
            "disagreementThresholdMeters": float(self.disagreement_threshold_meters),
            "worldMotionAvailable": bool(self.world_motion_available),
            "scaleIsRelative": bool(self.scale_is_relative),
            "recordProvenance": bool(self.record_provenance),
            "selectionMode": "fallback_only" if self.selection_mode in ("fallback_only", "stage_aware") else "weighted",
        }


def _as_array(value: Any, *, dimensions: int, name: str) -> Optional[np.ndarray]:
    """Extract an array from common observation/container forms."""

    if value is None:
        return None
    if FrameObservations is not None and isinstance(value, FrameObservations):
        if dimensions == 2:
            return value.get_2d_array().astype(np.float64, copy=False)
        if dimensions == 3:
            return value.get_3d_array().astype(np.float64, copy=False)
    if isinstance(value, Mapping):
        coordinate_names = ("x", "y", "z")[:dimensions]
        if all(name_key in value for name_key in coordinate_names):
            try:
                return np.asarray(
                    [float(value.get(name_key)) for name_key in coordinate_names],
                    dtype=np.float64,
                )
            except (TypeError, ValueError):
                return np.full((dimensions,), np.nan, dtype=np.float64)
        keys = {
            2: ("keypoints2d", "keypoints_2d", "landmarks2d", "landmarks_2d", "points2d", "points_2d"),
            3: ("landmarks3d", "landmarks_3d", "keypoints3d", "keypoints_3d", "points3d", "points_3d", "joints"),
        }[dimensions]
        nested = value.get("observation")
        if nested is not None:
            extracted = _as_array(nested, dimensions=dimensions, name=name)
            if extracted is not None:
                return extracted
        for key in keys:
            if key in value:
                return _as_array(value[key], dimensions=dimensions, name=name)
        return None
    if isinstance(value, (list, tuple)) and value and FrameObservations is not None:
        if all(isinstance(item, FrameObservations) for item in value):
            frames = [_as_array(item, dimensions=dimensions, name=name) for item in value]
            if all(frame is not None for frame in frames):
                return np.stack(frames, axis=0)  # type: ignore[arg-type]
    # JSON adapters commonly represent one frame as a list of ``{x, y, z}``
    # records. ``np.asarray`` cannot convert that form directly; preserve
    # malformed coordinates as NaN so finite gates can mark only that point
    # missing instead of aborting an otherwise valid sequence.
    if isinstance(value, (list, tuple)) and value:
        if all(isinstance(item, Mapping) for item in value):
            rows = [_as_array(item, dimensions=dimensions, name=name) for item in value]
            if all(row is not None for row in rows):
                return np.stack(rows, axis=0)  # type: ignore[arg-type]
        if all(isinstance(item, (list, tuple)) for item in value):
            nested_rows = [_as_array(item, dimensions=dimensions, name=name) for item in value]
            if all(row is not None for row in nested_rows):
                try:
                    return np.stack(nested_rows, axis=0)  # type: ignore[arg-type]
                except ValueError:
                    pass
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} could not be converted to a numeric array") from exc
    if array.ndim == 0 or array.shape[-1] < dimensions:
        raise ValueError(f"{name} must have shape (..., >={dimensions})")
    return array[..., :dimensions]


def _extract_keypoint_field(value: Any, field_name: str, *, dimensions: int) -> Optional[np.ndarray]:
    """Read score/visibility/presence/status from FrameObservations if given."""

    if value is None:
        return None
    if FrameObservations is not None and isinstance(value, FrameObservations):
        keypoints = value.keypoints_2d if dimensions == 2 else value.landmarks_3d
        return np.asarray([getattr(point, field_name, 0.0) for point in keypoints], dtype=np.float64)
    if isinstance(value, (list, tuple)) and value and FrameObservations is not None and all(isinstance(item, FrameObservations) for item in value):
        rows = [_extract_keypoint_field(item, field_name, dimensions=dimensions) for item in value]
        return np.stack(rows, axis=0) if all(row is not None for row in rows) else None  # type: ignore[arg-type]
    return None


def _sequence_values(value: Any) -> List[Any]:
    """Normalise the common sequence-shaped adapter return values.

    Full WHAM integrations do not all return the same wrapper: some return a
    list of frame records, while others return ``{"frames": [...]}`` or a
    batched NumPy tensor.  Keeping this tolerant parsing at the fusion seam
    means the extractor can preserve frame identity without teaching the
    retargeter about each adapter's JSON dialect.
    """

    if value is None:
        return []
    if isinstance(value, Mapping):
        for key in ("frames", "observations", "results", "sequence"):
            nested = value.get(key)
            if isinstance(nested, (list, tuple, np.ndarray)):
                if isinstance(nested, np.ndarray) and nested.ndim >= 3:
                    return [item for item in nested]
                return list(nested)
        # Some adapters return batched fields directly, e.g.
        # ``{"landmarks3d": (T, J, 3), "confidence": (T,)}``.
        for key in (
            "landmarks3d", "landmarks_3d", "keypoints3d", "keypoints_3d",
            "keypoints2d", "keypoints_2d", "joints3d", "joints_3d",
        ):
            nested = value.get(key)
            if isinstance(nested, (list, tuple, np.ndarray)):
                try:
                    nested_array = np.asarray(nested)
                except (TypeError, ValueError):
                    nested_array = None
                if nested_array is not None and nested_array.ndim >= 3:
                    rows = []
                    for index in range(nested_array.shape[0]):
                        row = dict(value)
                        row[key] = nested_array[index]
                        for confidence_key in ("confidence", "score"):
                            confidence = value.get(confidence_key)
                            if isinstance(confidence, (list, tuple, np.ndarray)):
                                try:
                                    confidence_array = np.asarray(confidence)
                                except (TypeError, ValueError):
                                    confidence_array = None
                                if confidence_array is not None and confidence_array.ndim > 0 and len(confidence_array) == nested_array.shape[0]:
                                    row[confidence_key] = confidence_array[index]
                        rows.append(row)
                    return rows
        # A single frame mapping is still a valid sequence input.
        return [value]
    if isinstance(value, np.ndarray):
        if value.ndim >= 3:
            return [item for item in value]
        return [value]
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _observation_metadata(value: Any) -> Dict[str, Any]:
    """Read serialisable metadata from a frame record or observation object."""

    if isinstance(value, Mapping):
        raw = value.get("metadata", value.get("backendMetadata", {}))
        return dict(raw) if isinstance(raw, Mapping) else {}
    raw = getattr(value, "backend_metadata", {})
    return dict(raw) if isinstance(raw, Mapping) else {}


def _observation_identity(value: Any) -> Tuple[Optional[int], Optional[float]]:
    """Return ``(frameIndex, timestampSec)`` from adapter naming variants."""

    if isinstance(value, Mapping):
        index = next((value.get(key) for key in ("frameIndex", "frame_index", "frame", "index") if key in value), None)
        timestamp = next((value.get(key) for key in ("timestampSec", "timestamp_sec", "timestamp", "time") if key in value), None)
    else:
        index = next((getattr(value, key, None) for key in ("frame_index", "frameIndex", "frame", "index") if hasattr(value, key)), None)
        timestamp = next((getattr(value, key, None) for key in ("timestamp_sec", "timestampSec", "timestamp", "time") if hasattr(value, key)), None)
    try:
        frame_index = int(index) if index is not None else None
    except (TypeError, ValueError, OverflowError):
        frame_index = None
    try:
        timestamp_sec = float(timestamp) if timestamp is not None else None
        if timestamp_sec is not None and not np.isfinite(timestamp_sec):
            timestamp_sec = None
    except (TypeError, ValueError, OverflowError):
        timestamp_sec = None
    return frame_index, timestamp_sec


def align_observation_sequences(
    primary: Any,
    auxiliary: Any = None,
    *,
    frame_indices: Optional[Sequence[int]] = None,
    timestamps: Optional[Sequence[float]] = None,
    frame_count: Optional[int] = None,
    timestamp_tolerance: Optional[float] = None,
) -> Tuple[List[Any], List[Any], Dict[str, Any]]:
    """Align adapter records by frame identity, then timestamp, then position.

    The quality adapter is allowed to drop a decoded frame or return records
    out of order.  The old integration used list position for both streams,
    which silently paired a WHAM result from one frame with MediaPipe pixels
    from another.  Frame index is authoritative; timestamps are the fallback
    for adapters that do not expose source indices.  Positional matching is
    retained only for anonymous records so legacy adapters remain usable.

    The returned diagnostics are intentionally JSON-compatible and are kept
    on the fusion result for the Timeline Editor.  Missing records remain
    explicit ``None`` slots instead of being replaced by a neighbouring pose.
    """

    primary_values = _sequence_values(primary)
    auxiliary_values = _sequence_values(auxiliary)
    requested_indices = list(frame_indices) if frame_indices is not None else []
    requested_times = list(timestamps) if timestamps is not None else []
    inferred_count = max(len(primary_values), len(auxiliary_values), len(requested_indices), len(requested_times), 0)
    count = int(frame_count if frame_count is not None else inferred_count)
    if count < 0:
        raise ValueError("frame_count must be non-negative")
    if len(requested_indices) == 0:
        requested_indices = list(range(count))
    if len(requested_times) == 0:
        requested_times = [float(index) / 30.0 for index in range(count)]
    if len(requested_indices) != count or len(requested_times) != count:
        raise ValueError("frame_indices and timestamps must contain one value per aligned frame")
    requested_indices = [int(index) for index in requested_indices]
    requested_times = [float(value) for value in requested_times]
    if timestamp_tolerance is None:
        deltas = np.diff(np.asarray(requested_times, dtype=np.float64)) if count > 1 else np.asarray([], dtype=np.float64)
        positive_deltas = np.abs(deltas[np.isfinite(deltas) & (np.abs(deltas) > 1e-9)])
        timestamp_tolerance = float(np.median(positive_deltas) * 0.51) if len(positive_deltas) else 1.0 / 60.0
    timestamp_tolerance = max(1e-6, float(timestamp_tolerance))

    def align(values: List[Any]) -> Tuple[List[Any], List[Dict[str, Any]], List[int]]:
        aligned: List[Any] = [None] * count
        matches: List[Dict[str, Any]] = []
        used: set = set()
        identities = [_observation_identity(item) for item in values]
        # First pass: exact source frame index.  Duplicate identities are
        # retained as unmatched diagnostics rather than nondeterministically
        # replacing a previously selected frame.
        for target_pos, target_index in enumerate(requested_indices):
            candidate = next(
                (source_pos for source_pos, (source_index, _timestamp) in enumerate(identities)
                 if source_pos not in used and source_index == target_index),
                None,
            )
            if candidate is not None:
                used.add(candidate)
                aligned[target_pos] = values[candidate]
                source_index, source_timestamp = identities[candidate]
                matches.append({
                    "targetPosition": int(target_pos),
                    "targetFrameIndex": int(target_index),
                    "sourcePosition": int(candidate),
                    "sourceFrameIndex": int(source_index) if source_index is not None else None,
                    "sourceTimestampSec": source_timestamp,
                    "match": "frame_index",
                })
        # Second pass: nearest timestamp, only when the source record exposes
        # one.  A conservative tolerance prevents a delayed frame from being
        # silently attached to a different sample.
        for target_pos, target_time in enumerate(requested_times):
            if aligned[target_pos] is not None:
                continue
            candidates = [
                (abs(float(source_timestamp) - target_time), source_pos)
                for source_pos, (_source_index, source_timestamp) in enumerate(identities)
                if source_pos not in used and source_timestamp is not None
            ]
            if candidates:
                distance, candidate = min(candidates)
                if distance <= timestamp_tolerance:
                    used.add(candidate)
                    aligned[target_pos] = values[candidate]
                    source_index, source_timestamp = identities[candidate]
                    matches.append({
                        "targetPosition": int(target_pos),
                        "targetFrameIndex": int(requested_indices[target_pos]),
                        "sourcePosition": int(candidate),
                        "sourceFrameIndex": int(source_index) if source_index is not None else None,
                        "sourceTimestampSec": source_timestamp,
                        "match": "timestamp",
                        "timestampErrorSec": float(distance),
                    })
        # Last pass: anonymous records use their position, but only if that
        # target has not already been matched by a stronger identity.
        for target_pos in range(count):
            if aligned[target_pos] is not None:
                continue
            candidate = next(
                (source_pos for source_pos, (source_index, source_timestamp) in enumerate(identities)
                 if source_pos not in used and source_index is None and source_timestamp is None),
                None,
            )
            if candidate is None:
                continue
            used.add(candidate)
            aligned[target_pos] = values[candidate]
            matches.append({
                "targetPosition": int(target_pos),
                "targetFrameIndex": int(requested_indices[target_pos]),
                "sourcePosition": int(candidate),
                "sourceFrameIndex": None,
                "sourceTimestampSec": None,
                "match": "position_anonymous",
            })
        unmatched = [int(index) for index in range(len(values)) if index not in used]
        return aligned, matches, unmatched

    aligned_primary, primary_matches, primary_unmatched = align(primary_values)
    aligned_auxiliary, auxiliary_matches, auxiliary_unmatched = align(auxiliary_values)

    def source_indices(matches: List[Dict[str, Any]]) -> List[Optional[int]]:
        by_target = {entry["targetPosition"]: entry.get("sourceFrameIndex") for entry in matches}
        return [by_target.get(index) for index in range(count)]

    diagnostics = {
        "frameIndices": requested_indices,
        "timestampsSec": requested_times,
        "timestampToleranceSec": timestamp_tolerance,
        "qualitySourceFrameIndices": source_indices(primary_matches),
        "mediapipeSourceFrameIndices": source_indices(auxiliary_matches),
        "qualityMatches": primary_matches,
        "mediapipeMatches": auxiliary_matches,
        "missingQualityFrames": [requested_indices[index] for index, value in enumerate(aligned_primary) if value is None],
        "missingMediaPipeFrames": [requested_indices[index] for index, value in enumerate(aligned_auxiliary) if value is None],
        "unmatchedQualitySourcePositions": primary_unmatched,
        "unmatchedMediaPipeSourcePositions": auxiliary_unmatched,
    }
    return aligned_primary, aligned_auxiliary, diagnostics


def _ensure_sequence(array: np.ndarray, *, dimensions: int, name: str) -> Tuple[np.ndarray, bool]:
    if array.ndim == 1 and dimensions == 1:
        array = array[None, :]
    if array.ndim == 2 and array.shape[-1] == dimensions:
        return array[None, ...], True
    if array.ndim == 3 and array.shape[-1] == dimensions:
        return array, False
    raise ValueError(f"{name} must have shape (J,{dimensions}) or (T,J,{dimensions})")


def _broadcast_numeric(value: Any, shape: Tuple[int, int], *, default: float, name: str) -> np.ndarray:
    if value is None:
        return np.full(shape, float(default), dtype=np.float64)
    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 0:
        return np.full(shape, float(array), dtype=np.float64)
    if array.ndim == 1:
        if array.shape[0] == shape[1]:
            array = array[None, :]
        elif array.shape[0] == shape[0]:
            array = array[:, None]
    try:
        return np.broadcast_to(array, shape).astype(np.float64, copy=True)
    except ValueError as exc:
        raise ValueError(f"{name} cannot broadcast to shape {shape}") from exc


def _broadcast_statuses(value: Any, shape: Tuple[int, int]) -> Optional[np.ndarray]:
    if value is None:
        return None
    array = np.asarray(value, dtype=object)
    if array.ndim == 0:
        return np.full(shape, array.item(), dtype=object)
    if array.ndim == 1:
        if array.shape[0] == shape[1]:
            array = array[None, :]
        elif array.shape[0] == shape[0]:
            array = array[:, None]
    try:
        return np.broadcast_to(array, shape).astype(object, copy=True)
    except ValueError as exc:
        raise ValueError(f"statuses cannot broadcast to shape {shape}") from exc


def _status_gate(status: Any, *, occluded_gate: float, predicted_gate: float) -> float:
    if status is None:
        return 1.0
    if isinstance(status, str):
        name = status.strip().lower().replace("-", "_")
        if name in ("missing", "none", "invalid"):
            return 0.0
        if name in ("occluded", "occlusion"):
            return occluded_gate
        if name in ("predicted", "inpainted"):
            return predicted_gate
        return 1.0
    try:
        numeric = int(status)
    except (TypeError, ValueError):
        return 1.0
    # ObservationStatus values: MISSING=0, OBSERVED=1, OCCLUDED=2,
    # PREDICTED=3, USER_CONSTRAINED=4.
    if numeric == 0:
        return 0.0
    if numeric == 2:
        return occluded_gate
    if numeric == 3:
        return predicted_gate
    return 1.0


def compute_temporal_agreement(
    points_2d: ArrayLike,
    timestamps: Optional[Sequence[float]] = None,
    *,
    body_scale: Optional[Union[float, Sequence[float], np.ndarray]] = None,
    minimum_scale: float = 0.05,
) -> np.ndarray:
    """Compute a per-frame/joint agreement gate from neighboring 2D motion.

    A point that moves much farther than the local body scale in one frame is
    treated as an outlier.  Real timestamps are used when available, so a
    variable-FPS clip is not accidentally penalized by frame-number spacing.
    The returned values are in ``[0, 1]`` and retain the input's single-frame
    versus sequence shape.
    """

    points = np.asarray(points_2d, dtype=np.float64)
    if points.ndim == 2 and points.shape[-1] >= 2:
        single = True
        points = points[None, :, :2]
    elif points.ndim == 3 and points.shape[-1] >= 2:
        single = False
        points = points[..., :2]
    else:
        raise ValueError("points_2d must have shape (J,>=2) or (T,J,>=2)")
    frames, joints = points.shape[:2]
    finite = np.isfinite(points).all(axis=-1)
    if frames <= 1:
        result = finite.astype(np.float64)
        return result[0] if single else result

    if timestamps is None:
        time = np.arange(frames, dtype=np.float64)
    else:
        time = np.asarray(timestamps, dtype=np.float64).reshape(-1)
        if len(time) != frames:
            raise ValueError("timestamps must contain one value per frame")
        time = np.nan_to_num(time, nan=0.0)
    dt = np.maximum(np.abs(np.diff(time)), 1e-6)

    if body_scale is None:
        # Avoid ``nanmax`` here.  A detector may return an all-missing frame
        # (or a slot that is missing in every frame), and ``nanmax`` emits a
        # RuntimeWarning for those all-NaN reductions even though the result
        # is intentionally replaced by zero below.  Finite sentinels keep the
        # reduction deterministic and warning-free while preserving the same
        # body-scale semantics for valid points.
        minima = np.min(
            np.where(finite[..., None], points, np.inf),
            axis=1,
        )
        maxima = np.max(
            np.where(finite[..., None], points, -np.inf),
            axis=1,
        )
        has_points = np.any(finite, axis=1)
        spans = np.where(has_points[..., None], maxima - minima, 0.0)
        scales = np.max(spans, axis=-1)
        # Single-joint clips have no spatial span; a conservative normalized
        # image scale still distinguishes a genuine detector jump.
        scale = np.maximum(scales, float(minimum_scale))
        scale = np.median(scale) if np.all(scale > 0.0) else float(minimum_scale)
    else:
        scale_array = np.asarray(body_scale, dtype=np.float64)
        if scale_array.ndim == 0:
            scale = max(float(scale_array), float(minimum_scale))
        else:
            scale = np.maximum(np.nan_to_num(scale_array, nan=minimum_scale), minimum_scale)
            if scale.shape != (frames,):
                scale = float(np.median(scale))

    agreement = np.ones((frames, joints), dtype=np.float64)
    for frame in range(frames):
        if 0 < frame < frames - 1:
            # Compare the sample to the linearly interpolated neighboring
            # samples.  This distinguishes an isolated detector spike (the
            # middle sample is far from its expected position) from the two
            # otherwise-valid neighbors that merely lead to that spike.
            expected = 0.5 * (points[frame - 1] + points[frame + 1])
            motion = np.linalg.norm(points[frame] - expected, axis=-1)
        elif frame > 0:
            motion = np.linalg.norm(points[frame] - points[frame - 1], axis=-1) / dt[frame - 1]
        elif frame < frames - 1:
            motion = np.linalg.norm(points[frame + 1] - points[frame], axis=-1) / dt[frame]
        else:
            motion = np.zeros((joints,), dtype=np.float64)
        agreement[frame] = np.exp(-np.clip(motion / np.asarray(scale), 0.0, 60.0))
    agreement[~finite] = 0.0
    return agreement[0] if single else agreement


def _smooth_weights(weights: np.ndarray, smoothing: float) -> np.ndarray:
    if weights.ndim != 2 or weights.shape[0] <= 1 or smoothing <= 0.0:
        return weights
    alpha = float(np.clip(smoothing, 0.0, 1.0))
    result = weights.copy()
    # Smooth in log space to avoid a confident frame instantly reviving a
    # missing/occluded landmark.  Very low raw confidence is kept untouched;
    # this is the safety property that protects WHAM depth from detector noise.
    for frame in range(1, len(result)):
        previous = result[frame - 1]
        current = weights[frame]
        candidate = np.exp(alpha * np.log(np.maximum(current, 1e-8)) + (1.0 - alpha) * np.log(np.maximum(previous, 1e-8)))
        result[frame] = np.where(current < 0.05, current, candidate)
    return np.clip(result, 0.0, 1.0)


def compute_mediapipe_weights(
    score: ArrayLike,
    visibility: Optional[ArrayLike] = None,
    presence: Optional[ArrayLike] = None,
    *,
    temporal_agreement: Optional[ArrayLike] = None,
    statuses: Optional[Any] = None,
    occlusion_gate: float = 0.08,
    predicted_gate: float = 0.20,
    smooth: bool = True,
    smoothing: float = 0.65,
) -> np.ndarray:
    """Compute the gated MediaPipe observation weight ``w_mp(j,t)``.

    The base formula is ``visibility * presence * detectorScore * temporal``.
    Status gates and optional temporal smoothing are applied afterwards.  All
    values are clipped to ``[0, 1]`` and missing/non-finite observations have
    zero weight.
    """

    score_array = np.asarray(score, dtype=np.float64)
    candidates = [score_array]
    for value in (visibility, presence, temporal_agreement):
        if value is not None:
            candidates.append(np.asarray(value, dtype=np.float64))
    shapes = [array.shape for array in candidates if array.ndim > 0]
    if not shapes:
        shape: Tuple[int, ...] = ()
    else:
        try:
            shape = np.broadcast_shapes(*shapes)
        except ValueError as exc:
            raise ValueError("MediaPipe gate inputs have incompatible shapes") from exc
    if len(shape) == 1:
        shape = (1, shape[0])
        single = True
    elif len(shape) == 2:
        single = False
    else:
        raise ValueError("gate inputs must have shape (J,) or (T,J)")

    def broadcast(value: Optional[ArrayLike], default: float) -> np.ndarray:
        if value is None:
            return np.full(shape, default, dtype=np.float64)
        array = np.asarray(value, dtype=np.float64)
        if array.ndim == 1:
            array = array[None, :]
        return np.broadcast_to(array, shape).astype(np.float64, copy=True)

    score_values = broadcast(score, 0.0)
    vis_values = broadcast(visibility, 1.0)
    pres_values = broadcast(presence, 1.0)
    temporal_values = broadcast(temporal_agreement, 1.0)
    raw = (
        np.clip(np.nan_to_num(score_values, nan=0.0), 0.0, 1.0)
        * np.clip(np.nan_to_num(vis_values, nan=0.0), 0.0, 1.0)
        * np.clip(np.nan_to_num(pres_values, nan=0.0), 0.0, 1.0)
        * np.clip(np.nan_to_num(temporal_values, nan=0.0), 0.0, 1.0)
    )

    status_values = _broadcast_statuses(statuses, shape)
    if status_values is not None:
        vectorized_gate = np.ones(shape, dtype=np.float64)
        for index in np.ndindex(shape):
            vectorized_gate[index] = _status_gate(
                status_values[index],
                occluded_gate=float(np.clip(occlusion_gate, 0.0, 1.0)),
                predicted_gate=float(np.clip(predicted_gate, 0.0, 1.0)),
            )
        raw *= vectorized_gate

    finite = np.isfinite(score_values) & np.isfinite(vis_values) & np.isfinite(pres_values) & np.isfinite(temporal_values)
    raw[~finite] = 0.0
    result = _smooth_weights(raw, smoothing) if smooth else raw
    result = np.clip(result, 0.0, 1.0)
    return result[0] if single else result


# Descriptive aliases used by callers that treat the gate as a generic
# observation/visibility gate.
compute_observation_weights = compute_mediapipe_weights
compute_visibility_weights = compute_mediapipe_weights
temporal_visibility_gate = compute_mediapipe_weights


@dataclass
class FusionDiagnostics(Mapping[str, Any]):
    """Mapping-like diagnostic payload retained on every fusion result."""

    data: Dict[str, Any] = field(default_factory=dict)

    def __getitem__(self, key: str) -> Any:
        return self.data[key]

    def __iter__(self):
        return iter(self.data)

    def __len__(self) -> int:
        return len(self.data)

    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)

    def to_dict(self) -> Dict[str, Any]:
        return _json_safe(self.data)

    @property
    def mean_reprojection_error(self) -> float:
        return float(self.data.get("meanReprojectionError", 0.0))

    @property
    def mean_wham_correction(self) -> float:
        return float(self.data.get("meanWhamCorrection", 0.0))


@dataclass
class FusionResult:
    """Fused 3D sequence, provenance, canonical metadata, and diagnostics."""

    joints: np.ndarray
    provenance: Any
    diagnostics: FusionDiagnostics
    metadata: Dict[str, Any]
    uncertainty_intervals: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def fused_3d(self) -> np.ndarray:
        return self.joints

    @property
    def landmarks_3d(self) -> np.ndarray:
        return self.joints

    @property
    def points(self) -> np.ndarray:
        return self.joints

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.joints.shape

    def __array__(self, dtype: Optional[np.dtype] = None) -> np.ndarray:
        return np.asarray(self.joints, dtype=dtype)

    def __getitem__(self, item: Any) -> Any:
        return self.joints[item]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "landmarks3d": _json_safe(self.joints),
            "backendMetadata": _json_safe(self.metadata),
            "uncertaintyIntervals": _json_safe(self.uncertainty_intervals),
            "jointProvenance": _json_safe(self.provenance),
            "diagnostics": self.diagnostics.to_dict(),
        }


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "to_dict") and callable(getattr(value, "to_dict")):
        try:
            return _json_safe(value.to_dict())
        except Exception:
            pass
    return value


def _huber_factor(residual: np.ndarray, delta: float) -> np.ndarray:
    residual = np.asarray(residual, dtype=np.float64)
    return np.where(residual <= delta, 1.0, delta / np.maximum(residual, 1e-12))


def _contiguous_intervals(
    mask: np.ndarray,
    *,
    frame_indices: np.ndarray,
    reason: str,
    confidence: np.ndarray,
    recommended_action: str,
) -> List[Dict[str, Any]]:
    """Compress a frame mask into export-style inclusive intervals."""

    if mask.ndim != 1:
        raise ValueError("interval mask must be one-dimensional")
    intervals: List[Dict[str, Any]] = []
    start: Optional[int] = None
    for index, active in enumerate(mask):
        if bool(active) and start is None:
            start = index
        closing = start is not None and (not bool(active) or index == len(mask) - 1)
        if closing:
            end = index if bool(active) and index == len(mask) - 1 else index - 1
            if end >= start:
                local_confidence = float(np.clip(np.mean(confidence[start : end + 1]), 0.0, 1.0))
                intervals.append({
                    "startFrame": int(frame_indices[start]),
                    "endFrame": int(frame_indices[end]),
                    "reason": reason,
                    "confidence": local_confidence,
                    "recommendedAction": recommended_action,
                })
            start = None
    return intervals


def _frame_observation_metadata(value: Any) -> Dict[str, Any]:
    if FrameObservations is not None and isinstance(value, FrameObservations):
        return dict(value.backend_metadata or {})
    if isinstance(value, Mapping):
        metadata = value.get("metadata", value.get("backendMetadata", {}))
        return dict(metadata) if isinstance(metadata, Mapping) else {}
    return {}


def _normalise_camera(camera: Optional[Union[CameraModel, Mapping[str, Any], str]]) -> Optional[CameraModel]:
    if camera is None:
        return None
    if isinstance(camera, CameraModel):
        return camera
    if isinstance(camera, Mapping):
        return CameraModel.from_dict(camera)
    if isinstance(camera, str):
        return CameraModel(projection_type=camera)
    raise TypeError("camera must be CameraModel, mapping, string, or None")


def _estimate_default_camera(
    wham: np.ndarray,
    points_2d: np.ndarray,
    weights: np.ndarray,
) -> CameraModel:
    """Estimate a stable camera from all valid frames.

    WHAM's temporal output is root-relative, so its Z origin is the subject
    pelvis rather than the physical camera plane.  Feeding those signed,
    near-zero depths into an automatic perspective fit makes ``focal_length``
    collapse and causes the 2D unprojection term to create very large 3D
    corrections.  The default monocular fusion camera therefore uses an
    orthographic fit; callers with calibrated/world camera geometry can still
    pass an explicit ``CameraModel`` to opt into perspective projection.
    """

    valid = np.isfinite(wham).all(axis=-1) & np.isfinite(points_2d).all(axis=-1) & (weights > 0.0)
    if int(np.count_nonzero(valid)) < 2:
        return CameraModel.orthographic(scale=1.0, principal_point=(0.5, 0.5))
    return estimate_camera_from_anchors(
        wham[valid],
        points_2d[valid],
        mode="orthographic",
        weights=weights[valid],
    )


def _field_from_observations(value: Any, field_name: str, dimensions: int) -> Optional[np.ndarray]:
    extracted = _extract_keypoint_field(value, field_name, dimensions=dimensions)
    if extracted is not None:
        return extracted
    if isinstance(value, Mapping):
        raw = value.get(field_name, value.get({"score": "scores", "visibility": "visibilities", "presence": "presences", "status": "statuses"}.get(field_name, field_name)))
        if raw is not None:
            return np.asarray(raw, dtype=object if field_name == "status" else np.float64)
    return None


def fuse_wham_mediapipe(
    wham_3d: Any,
    mediapipe_3d: Any = None,
    mediapipe_2d: Any = None,
    *,
    score: Any = None,
    visibility: Any = None,
    presence: Any = None,
    statuses: Any = None,
    temporal_agreement: Any = None,
    timestamps: Optional[Sequence[float]] = None,
    camera: Optional[Union[CameraModel, Mapping[str, Any], str]] = None,
    config: Optional[Union[FusionConfig, Mapping[str, Any]]] = None,
    wham_score: Any = None,
    mediapipe_score: Any = None,
    wham_coordinate_system: str = CANONICAL_COORDINATE_SYSTEM,
    mediapipe_coordinate_system: str = CANONICAL_COORDINATE_SYSTEM,
    coordinate_system: Optional[str] = None,
    frame_indices: Optional[Sequence[int]] = None,
    backend_metadata: Optional[Mapping[str, Any]] = None,
    temporal_constraints: Any = None,
    trajectory_metadata: Optional[Mapping[str, Any]] = None,
    provenance_reasons: Optional[Any] = None,
    selection_mode: Optional[str] = None,
) -> FusionResult:
    """Fuse a WHAM 3D sequence with gated MediaPipe 3D/2D observations.

    Parameters accept either NumPy arrays or ``FrameObservations``/mapping
    containers.  The result always uses canonical SMPL-X coordinates.  Input
    arrays may be one frame ``(J,3)`` or a sequence ``(T,J,3)``; the result
    preserves that dimensionality.  If WHAM is invalid for a point, a valid
    MediaPipe 3D point is used as an explicit fallback and marked in
    provenance/diagnostics.
    """

    fusion_config = config if isinstance(config, FusionConfig) else FusionConfig.from_dict(config)
    selection_mode = str(selection_mode or fusion_config.selection_mode or "weighted").strip().lower().replace("-", "_")
    if selection_mode not in ("weighted", "fallback_only", "stage_aware"):
        raise ValueError("selection_mode must be 'weighted' or 'fallback_only'")
    fallback_only = selection_mode in ("fallback_only", "stage_aware")
    if coordinate_system is not None:
        wham_coordinate_system = coordinate_system

    wham_source = _as_array(wham_3d, dimensions=3, name="wham_3d")
    if wham_source is None:
        raise ValueError("wham_3d is required")
    wham_sequence, input_single = _ensure_sequence(wham_source, dimensions=3, name="wham_3d")
    wham_sequence = _convert_if_needed(wham_sequence, wham_coordinate_system)
    frames, joints = wham_sequence.shape[:2]
    if joints == 0:
        raise ValueError("wham_3d must contain at least one joint")

    # Pull fields from observation containers before normalizing optional 3D
    # and 2D arrays.  Explicit keyword arguments always win.
    if mediapipe_score is not None and score is None:
        score = mediapipe_score
    if score is None:
        score = _field_from_observations(mediapipe_2d if mediapipe_2d is not None else mediapipe_3d, "score", 2 if mediapipe_2d is not None else 3)
    if visibility is None:
        visibility = _field_from_observations(mediapipe_2d if mediapipe_2d is not None else mediapipe_3d, "visibility", 2 if mediapipe_2d is not None else 3)
    if presence is None:
        presence = _field_from_observations(mediapipe_2d if mediapipe_2d is not None else mediapipe_3d, "presence", 2 if mediapipe_2d is not None else 3)
    if statuses is None:
        statuses = _field_from_observations(mediapipe_2d if mediapipe_2d is not None else mediapipe_3d, "status", 2 if mediapipe_2d is not None else 3)

    mp3d_raw = _as_array(mediapipe_3d, dimensions=3, name="mediapipe_3d")
    mp2d_raw = _as_array(mediapipe_2d, dimensions=2, name="mediapipe_2d")
    mp3d: Optional[np.ndarray] = None
    mp2d: Optional[np.ndarray] = None
    if mp3d_raw is not None:
        mp3d, _ = _ensure_sequence(mp3d_raw, dimensions=3, name="mediapipe_3d")
        if mp3d.shape[0] != frames or mp3d.shape[1] != joints:
            try:
                mp3d = np.broadcast_to(mp3d, (frames, joints, 3)).astype(np.float64, copy=True)
            except ValueError as exc:
                raise ValueError("mediapipe_3d must align with wham_3d") from exc
        mp3d = _convert_if_needed(mp3d, mediapipe_coordinate_system)
    if mp2d_raw is not None:
        mp2d, _ = _ensure_sequence(mp2d_raw, dimensions=2, name="mediapipe_2d")
        if mp2d.shape[0] != frames or mp2d.shape[1] != joints:
            try:
                mp2d = np.broadcast_to(mp2d, (frames, joints, 2)).astype(np.float64, copy=True)
            except ValueError as exc:
                raise ValueError("mediapipe_2d must align with wham_3d") from exc

    score_values = _broadcast_numeric(score, (frames, joints), default=1.0 if (mp3d is not None or mp2d is not None) else 0.0, name="score")
    visibility_values = _broadcast_numeric(visibility, (frames, joints), default=1.0, name="visibility")
    presence_values = _broadcast_numeric(presence, (frames, joints), default=1.0, name="presence")
    status_values = _broadcast_statuses(statuses, (frames, joints))

    if temporal_agreement is None and mp2d is not None:
        temporal_agreement_values = compute_temporal_agreement(mp2d, timestamps=timestamps)
        if temporal_agreement_values.ndim == 1:
            temporal_agreement_values = temporal_agreement_values[None, :]
    else:
        temporal_agreement_values = _broadcast_numeric(temporal_agreement, (frames, joints), default=1.0, name="temporal_agreement")

    mp_weights = compute_mediapipe_weights(
        score_values,
        visibility_values,
        presence_values,
        temporal_agreement=temporal_agreement_values,
        statuses=status_values,
        occlusion_gate=fusion_config.occluded_gate,
        predicted_gate=fusion_config.predicted_gate,
        smooth=True,
        smoothing=fusion_config.temporal_smoothing,
    )
    if mp_weights.ndim == 1:
        mp_weights = mp_weights[None, :]

    # A full WHAM adapter may expose a per-frame/per-joint temporal validity
    # mask (for example a forward/backward consistency or kinematic limit
    # check).  It gates only auxiliary observations; WHAM remains the primary
    # hypothesis and is never discarded merely because MediaPipe is noisy.
    temporal_constraints_input = temporal_constraints
    if isinstance(temporal_constraints, Mapping):
        temporal_constraints = next(
            (
                temporal_constraints.get(key)
                for key in ("weights", "mask", "values", "validity")
                if temporal_constraints.get(key) is not None
            ),
            None,
        )
    if temporal_constraints is not None:
        constraint_values = _broadcast_numeric(
            temporal_constraints,
            (frames, joints),
            default=1.0,
            name="temporal_constraints",
        )
        constraint_values = np.clip(np.nan_to_num(constraint_values, nan=0.0), 0.0, 1.0)
        mp_weights *= constraint_values
    else:
        constraint_values = np.ones((frames, joints), dtype=np.float64)

    wham_values = _array_from_optional_score(wham_sequence, wham_score, (frames, joints), default=1.0)
    wham_finite = np.isfinite(wham_sequence).all(axis=-1)
    wham_weights = np.clip(np.nan_to_num(wham_values, nan=0.0), 0.0, 1.0) * wham_finite
    wham_weights = np.where(wham_weights >= fusion_config.min_wham_score, wham_weights, 0.0)
    mp3d_finite = np.isfinite(mp3d).all(axis=-1) if mp3d is not None else np.zeros((frames, joints), dtype=bool)
    mp2d_finite = np.isfinite(mp2d).all(axis=-1) if mp2d is not None else np.zeros((frames, joints), dtype=bool)
    # Keep independent gates for the independent observation modalities. A
    # missing/NaN MP-3D joint must not disable a valid MP-2D reprojection
    # anchor (and vice versa); the shared mask used previously dropped the 2D
    # residual whenever a 3D value was absent.
    mp3d_weights = mp_weights * mp3d_finite if mp3d is not None else np.zeros((frames, joints), dtype=np.float64)
    mp2d_weights = mp_weights * mp2d_finite if mp2d is not None else np.zeros((frames, joints), dtype=np.float64)
    # Camera fitting is an observation-alignment operation, not a joint
    # selection operation.  In the stage-aware/fallback-only policy we later
    # suppress MediaPipe terms for valid WHAM joints, but suppressing them
    # before estimating the 2D camera leaves no anchors and silently falls
    # back to scale=1/principal=(0.5,0.5).  That default projects a perfectly
    # valid root-relative WHAM body into the wrong image location and makes
    # the UI report a false geometry fallback on nearly every frame.  Keep a
    # private copy for camera fitting while preserving the strict WHAM-primary
    # numerical selection below.
    camera_fit_weights = np.array(mp2d_weights, dtype=np.float64, copy=True)
    if fallback_only:
        # Extraction uses a strict stage-aware policy: a valid migrated WHAM
        # joint remains the primary signal.  MediaPipe/RTMPose can fill only
        # a missing or invalid WHAM slot; direct callers can still request the
        # historical weighted correction mode explicitly.
        valid_wham_mask = wham_weights > 0.0
        mp3d_weights = np.where(valid_wham_mask, 0.0, mp3d_weights)
        mp2d_weights = np.where(valid_wham_mask, 0.0, mp2d_weights)
    effective_mp_weights = np.maximum(mp3d_weights, mp2d_weights)

    camera_model = _normalise_camera(camera)
    if mp2d is not None and camera_model is None:
        camera_model = _estimate_default_camera(wham_sequence, mp2d, camera_fit_weights)
    if camera_model is None:
        camera_model = CameraModel.orthographic(scale=1.0, principal_point=(0.5, 0.5))

    fused = np.array(wham_sequence, dtype=np.float64, copy=True)
    # Invalid WHAM values are initialized from valid MP 3D.  If only 2D is
    # available, zero is retained and unprojection uses that known depth; no
    # fabricated depth is introduced by the camera term.
    invalid_wham = wham_weights <= 0.0
    if mp3d is not None:
        fused[invalid_wham & mp3d_finite] = mp3d[invalid_wham & mp3d_finite]
    fused[~np.isfinite(fused)] = 0.0

    mp3d_weight = float(fusion_config.effective_mediapipe_3d_weight)
    reprojection_errors = np.zeros((frames, joints), dtype=np.float64)
    robust_3d = np.ones((frames, joints), dtype=np.float64)
    robust_2d = np.ones((frames, joints), dtype=np.float64)
    for _ in range(fusion_config.max_iterations):
        finite_current = np.isfinite(fused).all(axis=-1)
        if mp3d is not None:
            residual_3d = np.linalg.norm(
                np.nan_to_num(fused - mp3d, nan=0.0, posinf=0.0, neginf=0.0),
                axis=-1,
            )
            residual_3d[~mp3d_finite] = 0.0
            robust_3d = _huber_factor(residual_3d, fusion_config.robust_delta_3d)
        else:
            residual_3d = np.zeros((frames, joints), dtype=np.float64)
            robust_3d = np.zeros((frames, joints), dtype=np.float64)
        if mp2d is not None:
            projected = camera_model.project(fused)
            reprojection_errors = np.linalg.norm(
                np.nan_to_num(projected - mp2d, nan=0.0, posinf=0.0, neginf=0.0),
                axis=-1,
            )
            reprojection_errors[~mp2d_finite] = 0.0
            robust_2d = _huber_factor(reprojection_errors, fusion_config.robust_delta_2d)
        else:
            reprojection_errors.fill(0.0)
            robust_2d.fill(0.0)

        # Fully vectorized term calculation across all (frames, joints)
        w_wham = np.where(wham_weights > 0.0, fusion_config.wham_weight * wham_weights, 0.0)[..., None]
        safe_p_wham = np.nan_to_num(wham_sequence, nan=0.0)

        if mp3d is not None:
            w_mp3d_mask = mp3d_finite & (mp3d_weights > 0.0)
            w_mp3d = np.where(w_mp3d_mask, mp3d_weight * mp3d_weights * robust_3d, 0.0)[..., None]
            safe_p_mp3d = np.nan_to_num(mp3d, nan=0.0)
        else:
            w_mp3d = np.zeros((frames, joints, 1), dtype=np.float64)
            safe_p_mp3d = np.zeros((frames, joints, 3), dtype=np.float64)

        if mp2d is not None:
            w_mp2d_mask = mp2d_finite & (mp2d_weights > 0.0)
            w_mp2d = np.where(w_mp2d_mask, fusion_config.reprojection_weight * mp2d_weights * robust_2d, 0.0)[..., None]
            safe_mp2d = np.nan_to_num(mp2d, nan=0.0)
            safe_fused_z = np.nan_to_num(fused[..., 2], nan=0.0)
            safe_p_mp2d = np.nan_to_num(camera_model.unproject(safe_mp2d, safe_fused_z), nan=0.0)
        else:
            w_mp2d = np.zeros((frames, joints, 1), dtype=np.float64)
            safe_p_mp2d = np.zeros((frames, joints, 3), dtype=np.float64)

        denom = w_wham + w_mp3d + w_mp2d
        denom_valid = (denom > 1e-12).squeeze(-1)
        safe_denom = np.where(denom > 1e-12, denom, 1.0)
        target = (w_wham * safe_p_wham + w_mp3d * safe_p_mp3d + w_mp2d * safe_p_mp2d) / safe_denom

        step_update = (1.0 - fusion_config.step_size) * fused + fusion_config.step_size * target
        fused = np.where(
            denom_valid[..., None],
            np.where(finite_current[..., None], step_update, target),
            fused,
        )

    fused = np.nan_to_num(fused, nan=0.0, posinf=0.0, neginf=0.0)
    final_projected = camera_model.project(fused)
    if mp2d is not None:
        reprojection_errors = np.linalg.norm(np.nan_to_num(final_projected - mp2d, nan=0.0), axis=-1)
        reprojection_errors[~mp2d_finite] = 0.0
    else:
        reprojection_errors.fill(0.0)
    wham_corrections = np.linalg.norm(np.nan_to_num(fused - wham_sequence, nan=0.0), axis=-1)
    wham_corrections[~wham_finite] = 0.0
    disagreement = np.linalg.norm(np.nan_to_num(fused - mp3d, nan=0.0), axis=-1) if mp3d is not None else np.zeros((frames, joints), dtype=np.float64)
    disagreement[~mp3d_finite] = 0.0
    wham_mp_disagreement = np.linalg.norm(np.nan_to_num(wham_sequence - mp3d, nan=0.0), axis=-1) if mp3d is not None else np.zeros((frames, joints), dtype=np.float64)
    wham_mp_disagreement[~(wham_finite & mp3d_finite)] = 0.0

    if frame_indices is None:
        frame_numbers = np.arange(frames, dtype=np.int64)
    else:
        frame_numbers = np.asarray(frame_indices, dtype=np.int64).reshape(-1)
        if len(frame_numbers) != frames:
            raise ValueError("frame_indices must contain one value per frame")

    intervals: List[Dict[str, Any]] = []
    low_confidence_frame = np.any((effective_mp_weights < fusion_config.observed_weight_threshold) & (mp3d_finite | mp2d_finite), axis=1)
    low_confidence_score = np.clip(1.0 - np.mean(effective_mp_weights, axis=1), 0.0, 1.0)
    intervals.extend(_contiguous_intervals(
        low_confidence_frame,
        frame_indices=frame_numbers,
        reason="low_confidence",
        confidence=low_confidence_score,
        recommended_action="review_observation_gate",
    ))
    disagreement_frame = np.any(wham_mp_disagreement > fusion_config.disagreement_threshold_meters, axis=1)
    disagreement_confidence = np.clip(
        1.0 - np.max(wham_mp_disagreement, axis=1) / max(fusion_config.disagreement_threshold_meters, 1e-6),
        0.0,
        1.0,
    )
    intervals.extend(_contiguous_intervals(
        disagreement_frame,
        frame_indices=frame_numbers,
        reason="wham_mediapipe_disagreement",
        confidence=disagreement_confidence,
        recommended_action="review_depth_hypothesis",
    ))

    metadata = canonical_coordinate_metadata(
        world_motion_available=fusion_config.world_motion_available,
        scale_is_relative=fusion_config.scale_is_relative,
    )
    metadata.update({
        "schemaVersion": 2,
        "backendRequested": "wham",
        "backendActual": "wham",
        "fusionMode": "wham_mediapipe",
        "backendFallback": False,
        "fallbackFrom": None,
        "fallbackReason": None,
        "overlayBackend": "wham_mediapipe",
        "overlaySource": "fused_3d_projection",
        "fusion": {
            "cameraModel": camera_model.projection_type,
            "weightsVersion": 1,
            "meanReprojectionError": _weighted_scalar_mean(reprojection_errors, mp2d_finite),
            "meanWhamCorrection": _weighted_scalar_mean(wham_corrections, wham_finite),
            "configuration": fusion_config.to_dict(),
        },
        "camera": camera_model.to_dict(),
        "temporalConstraintsApplied": bool(temporal_constraints is not None),
        "selectionMode": "fallback_only" if fallback_only else "weighted",
    })
    if trajectory_metadata is not None:
        metadata["trajectory"] = _json_safe(trajectory_metadata)
    if temporal_constraints is not None:
        metadata["temporalConstraints"] = _json_safe(
            temporal_constraints_input if isinstance(temporal_constraints_input, Mapping) else constraint_values
        )
    if backend_metadata:
        # Caller metadata is useful for checkpoint hashes, but the canonical
        # and fusion identity fields above remain authoritative.
        authoritative_keys = {
            "schemaVersion",
            "coordinateSystem",
            "fusionMode",
            "backendActual",
            "backendRequested",
            "backendFallback",
            "fallbackFrom",
            "fallbackReason",
            "overlayBackend",
            "overlaySource",
            "fusion",
            "camera",
            "trajectory",
            "temporalConstraints",
            "temporalConstraintsApplied",
            "observationAlignment",
            "frameErrors",
        }
        for key, value in backend_metadata.items():
            if key not in authoritative_keys:
                metadata[key] = _json_safe(value)

    provenance = _build_provenance(
        fused,
        wham_weights,
        effective_mp_weights,
        mp3d_finite,
        mp2d_finite,
        wham_finite,
        wham_mp_disagreement,
        fusion_config,
        statuses=status_values,
        frame_indices=frame_numbers,
        reasons=provenance_reasons,
    ) if fusion_config.record_provenance else []
    frame_provenance = _build_frame_provenance(provenance) if provenance else []
    metadata["frameProvenance"] = frame_provenance
    diagnostics_data: Dict[str, Any] = {
        "schemaVersion": 2,
        "backendRequested": "wham",
        "backendActual": "wham",
        "fusionMode": "wham_mediapipe",
        "selectionMode": "fallback_only" if fallback_only else "weighted",
        "coordinateSystem": CANONICAL_COORDINATE_SYSTEM,
        "mediapipeWeights": effective_mp_weights,
        "whamWeights": wham_weights,
        "temporalAgreement": temporal_agreement_values,
        "temporalConstraints": constraint_values,
        "reprojectionErrors": reprojection_errors,
        "whamCorrections": wham_corrections,
        "disagreementMeters": wham_mp_disagreement,
        "robust3dFactors": robust_3d,
        "robust2dFactors": robust_2d,
        "meanReprojectionError": _weighted_scalar_mean(reprojection_errors, mp2d_finite),
        "meanWhamCorrection": _weighted_scalar_mean(wham_corrections, wham_finite),
        "uncertaintyIntervals": intervals,
        "warnings": [],
    }
    if mp2d is not None and camera_model is not None and camera is None:
        diagnostics_data["warnings"].append("camera_estimated_from_image_anchors")
    if not np.any(wham_finite):
        diagnostics_data["warnings"].append("wham_missing_media_pipe_fallback_used")

    output_joints = fused[0] if input_single else fused
    output_provenance = (provenance[0] if provenance else []) if input_single else provenance
    return FusionResult(
        joints=output_joints,
        provenance=output_provenance,
        diagnostics=FusionDiagnostics(diagnostics_data),
        metadata=metadata,
        uncertainty_intervals=intervals,
    )


def _convert_if_needed(points: np.ndarray, coordinate_system: str) -> np.ndarray:
    if str(coordinate_system or CANONICAL_COORDINATE_SYSTEM).strip().lower() in (
        "smplx",
        "canonical",
        "canonical_smplx",
        "smplx_y_up_z_forward",
    ):
        return np.asarray(points, dtype=np.float64)
    from .camera import transform_coordinates
    return transform_coordinates(points, coordinate_system, CANONICAL_COORDINATE_SYSTEM).astype(np.float64, copy=False)


def _array_from_optional_score(points: np.ndarray, value: Any, shape: Tuple[int, int], *, default: float) -> np.ndarray:
    if value is None:
        return np.full(shape, default, dtype=np.float64)
    return _broadcast_numeric(value, shape, default=default, name="wham_score")


def _weighted_scalar_mean(values: np.ndarray, mask: np.ndarray) -> float:
    selected = np.asarray(values, dtype=np.float64)[np.asarray(mask, dtype=bool)]
    selected = selected[np.isfinite(selected)]
    return float(np.mean(selected)) if len(selected) else 0.0


def _build_provenance(
    fused: np.ndarray,
    wham_weights: np.ndarray,
    mp_weights: np.ndarray,
    mp3d_finite: np.ndarray,
    mp2d_finite: np.ndarray,
    wham_finite: np.ndarray,
    disagreement: np.ndarray,
    config: FusionConfig,
    *,
    statuses: Optional[np.ndarray],
    frame_indices: Optional[Sequence[int]] = None,
    reasons: Optional[Any] = None,
) -> List[List[Dict[str, Any]]]:
    frames, joints = fused.shape[:2]
    if frame_indices is None:
        frame_numbers = list(range(frames))
    else:
        frame_numbers = [int(value) for value in frame_indices]
        if len(frame_numbers) != frames:
            frame_numbers = list(range(frames))

    def explicit_reason(frame: int, joint: int) -> Optional[str]:
        if reasons is None:
            return None
        try:
            value = reasons[frame][joint]
        except (IndexError, KeyError, TypeError):
            return None
        if isinstance(value, Mapping):
            value = value.get("reason")
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    result: List[List[Dict[str, Any]]] = []
    mp_can_contribute = bool(
        config.effective_mediapipe_3d_weight > 0.0 or config.reprojection_weight > 0.0
    )
    for frame in range(frames):
        row: List[Dict[str, Any]] = []
        for joint in range(joints):
            has_wham = bool(wham_finite[frame, joint] and wham_weights[frame, joint] > 0.0)
            has_mp3d = bool(mp3d_finite[frame, joint])
            has_mp2d = bool(mp2d_finite[frame, joint])
            has_mp_support = bool(mp_weights[frame, joint] > 0.0 and (has_mp3d or has_mp2d))
            observed = bool(mp_weights[frame, joint] >= config.observed_weight_threshold and (has_mp3d or has_mp2d))
            if has_wham and has_mp_support and observed and mp_can_contribute:
                source = "fused" if disagreement[frame, joint] > 1e-6 or has_mp2d else "wham"
                primary = "wham"
            elif has_wham:
                source = "wham"
                primary = "wham"
            elif has_mp3d or has_mp2d:
                source = "mediapipe"
                primary = "mediapipe"
            else:
                source = "missing"
                primary = "none"
            if observed:
                status = "observed"
            elif has_wham and (has_mp3d or has_mp2d):
                status = "occluded" if mp_weights[frame, joint] > 0.0 else "predicted"
            elif has_wham:
                status = "predicted"
            else:
                status = "missing"
            auxiliary: List[str] = []
            if has_mp3d:
                auxiliary.append("mediapipe_3d")
            if has_mp2d:
                auxiliary.append("mediapipe_2d")
            confidence = float(np.clip(max(wham_weights[frame, joint], mp_weights[frame, joint]), 0.0, 1.0))
            reason = explicit_reason(frame, joint)
            if reason is None:
                if source == "mediapipe" and not has_wham:
                    reason = "wham_invalid_mediapipe_selected"
                elif source == "fused":
                    reason = "mediapipe_observation_fused"
                elif source == "wham" and has_mp_support:
                    reason = "wham_primary_mediapipe_rejected"
                elif source == "wham":
                    reason = "wham_valid"
                elif source == "missing":
                    reason = "wham_and_mediapipe_missing"
                else:
                    reason = "mediapipe_low_confidence"
            row.append({
                "frame": int(frame_numbers[frame]),
                "joint": int(joint),
                "source": source,
                "primary": primary,
                "auxiliary": auxiliary,
                "confidence": confidence,
                "observationWeight": float(np.clip(mp_weights[frame, joint], 0.0, 1.0)),
                "status": status,
                "reason": reason,
                "whamValid": has_wham,
                "mediapipe3dValid": has_mp3d,
                "mediapipe2dValid": has_mp2d,
                "usedFallback": bool(source == "mediapipe" and not has_wham),
                "coordinateSystem": CANONICAL_COORDINATE_SYSTEM,
            })
        result.append(row)
    return result


def _build_frame_provenance(
    provenance: List[List[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    """Summarise per-joint provenance into timeline-friendly frame records."""

    output: List[Dict[str, Any]] = []
    for row in provenance:
        counts: Dict[str, int] = {}
        reasons: List[str] = []
        fallback_joints: List[int] = []
        for item in row:
            source = str(item.get("source", "missing"))
            counts[source] = counts.get(source, 0) + 1
            reason = str(item.get("reason", ""))
            if reason and reason not in reasons:
                reasons.append(reason)
            if bool(item.get("usedFallback", False)):
                fallback_joints.append(int(item.get("joint", len(fallback_joints))))
        frame = int(row[0].get("frame", len(output))) if row else len(output)
        if fallback_joints:
            frame_reason = "mediapipe_fallback"
        elif not counts.get("wham", 0) and not counts.get("fused", 0):
            frame_reason = "wham_missing"
        else:
            frame_reason = reasons[0] if reasons else ""
        output.append({
            "frame": frame,
            "reason": frame_reason,
            "sourceCounts": counts,
            "reasons": reasons,
            "fallbackJoints": fallback_joints,
            "usedWham": bool(counts.get("wham", 0) or counts.get("fused", 0)),
            "usedMediaPipe": bool(counts.get("mediapipe", 0) or counts.get("fused", 0)),
        })
    return output


# Friendly names for callers that use a class/function distinction.
fuse_pose_sequence = fuse_wham_mediapipe
fuse_3d = fuse_wham_mediapipe
fuse = fuse_wham_mediapipe
compute_temporal_weights = compute_mediapipe_weights


__all__ = [
    "FusionConfig",
    "FusionDiagnostics",
    "FusionResult",
    "align_observation_sequences",
    "compute_temporal_agreement",
    "compute_mediapipe_weights",
    "compute_observation_weights",
    "compute_visibility_weights",
    "temporal_visibility_gate",
    "compute_temporal_weights",
    "fuse_wham_mediapipe",
    "fuse_pose_sequence",
    "fuse_3d",
    "fuse",
]
