"""Small, dependency-free camera and coordinate helpers for pose fusion.

The quality pose pipeline has one internal coordinate contract: canonical
SMPL-X coordinates (``+X`` character-left, ``+Y`` up, ``+Z`` forward).  WHAM
native output and the legacy image stream use camera/image axes instead.  The
conversion functions in this module are deliberately kept at the adapter
boundary so a score, visibility value, or provenance column can never be
mistaken for a coordinate axis.

Only NumPy is required.  In particular this module does not import OpenCV,
SciPy, MediaPipe, PyTorch, or any other optional inference dependency.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple, Union

import numpy as np


CANONICAL_COORDINATE_SYSTEM = "smplx"
WHAM_RAW_COORDINATE_SYSTEM = "wham_camera_y_down_z_away"
LEGACY_OVERLAY_COORDINATE_SYSTEM = "image_y_down_z_away"

_CANONICAL_AXIS_POLICY = {
    "whamRaw": WHAM_RAW_COORDINATE_SYSTEM,
    "canonical": "smplx_y_up_z_forward",
    "overlay": "image_y_down",
}


def canonical_coordinate_metadata(
    *,
    world_motion_available: bool = False,
    scale_is_relative: bool = True,
    source: Optional[str] = None,
) -> Dict[str, Any]:
    """Return serializable metadata for the canonical SMPL-X contract.

    ``scale_is_relative`` is true for the normal monocular case: the pose is
    in meter-like subject-relative units, not a world-calibrated trajectory.
    The returned dictionary is a fresh object so callers can safely augment it
    for export.
    """

    metadata: Dict[str, Any] = {
        "coordinateSystem": CANONICAL_COORDINATE_SYSTEM,
        "axisX": "left",
        "axisY": "up",
        "axisZ": "forward",
        "unit": "meter",
        "scaleIsRelative": bool(scale_is_relative),
        "worldMotionAvailable": bool(world_motion_available),
        "landmarkAxisPolicy": dict(_CANONICAL_AXIS_POLICY),
    }
    if source:
        metadata["sourceCoordinateSystem"] = str(source)
    return metadata


# Both names are useful to callers that prefer a noun or a constant.  Keep a
# copy rather than exposing a mutable module-level dictionary.
CANONICAL_COORDINATE_METADATA = canonical_coordinate_metadata()
CANONICAL_METADATA = CANONICAL_COORDINATE_METADATA


def _normalise_coordinate_name(name: str) -> str:
    value = str(name or "").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "canonical": CANONICAL_COORDINATE_SYSTEM,
        "smplx": CANONICAL_COORDINATE_SYSTEM,
        "smplx_y_up_z_forward": CANONICAL_COORDINATE_SYSTEM,
        "canonical_smplx": CANONICAL_COORDINATE_SYSTEM,
        "wham": WHAM_RAW_COORDINATE_SYSTEM,
        "wham_raw": WHAM_RAW_COORDINATE_SYSTEM,
        "wham_camera": WHAM_RAW_COORDINATE_SYSTEM,
        "wham_camera_y_down_z_away": WHAM_RAW_COORDINATE_SYSTEM,
        "camera_y_down_z_away": WHAM_RAW_COORDINATE_SYSTEM,
        "legacy": LEGACY_OVERLAY_COORDINATE_SYSTEM,
        "overlay": LEGACY_OVERLAY_COORDINATE_SYSTEM,
        "legacy_overlay": LEGACY_OVERLAY_COORDINATE_SYSTEM,
        "image_y_down": LEGACY_OVERLAY_COORDINATE_SYSTEM,
        "image_y_down_z_away": LEGACY_OVERLAY_COORDINATE_SYSTEM,
    }
    if value not in aliases:
        raise ValueError(f"Unsupported coordinate system: {name!r}")
    return aliases[value]


def transform_coordinates(
    points: np.ndarray,
    source: str,
    target: str = CANONICAL_COORDINATE_SYSTEM,
) -> np.ndarray:
    """Convert point coordinates between supported camera and SMPL-X frames.

    The input may have shape ``(..., 3)`` or ``(..., >=4)``.  Any columns
    after XYZ (for example score, visibility, or a provenance index) are
    copied unchanged.  A copy is always returned, including for an identity
    conversion.
    """

    array = np.asarray(points)
    if array.ndim == 0 or array.shape[-1] < 3:
        raise ValueError("points must have shape (..., >=3)")

    source_name = _normalise_coordinate_name(source)
    target_name = _normalise_coordinate_name(target)
    dtype = np.result_type(array.dtype, np.float32)
    result = np.array(array, dtype=dtype, copy=True)

    # Both WHAM camera coordinates and the legacy image stream have +Y down
    # and +Z away.  Canonical SMPL-X uses +Y up and +Z forward.
    if source_name != target_name:
        source_is_camera = source_name != CANONICAL_COORDINATE_SYSTEM
        target_is_camera = target_name != CANONICAL_COORDINATE_SYSTEM
        if source_is_camera != target_is_camera:
            result[..., 1] *= -1.0
            result[..., 2] *= -1.0
        # The two camera conventions currently share the same XYZ axis
        # direction.  Keeping this branch explicit makes adding a future
        # calibrated camera frame less error-prone.
    return result


def wham_to_canonical(points: np.ndarray) -> np.ndarray:
    """Convert raw WHAM camera ``(+Y down, +Z away)`` points to SMPL-X."""

    return transform_coordinates(points, WHAM_RAW_COORDINATE_SYSTEM, CANONICAL_COORDINATE_SYSTEM)


def canonical_to_legacy_overlay(points: np.ndarray) -> np.ndarray:
    """Encode canonical SMPL-X points for the legacy image-oriented stream."""

    return transform_coordinates(points, CANONICAL_COORDINATE_SYSTEM, LEGACY_OVERLAY_COORDINATE_SYSTEM)


def legacy_overlay_to_canonical(points: np.ndarray) -> np.ndarray:
    """Decode legacy image-oriented points into canonical SMPL-X coordinates."""

    return transform_coordinates(points, LEGACY_OVERLAY_COORDINATE_SYSTEM, CANONICAL_COORDINATE_SYSTEM)


# Adapters in older code use these spellings; aliases keep the conversion
# contract discoverable without forcing callers to know the implementation
# module's preferred terminology.
canonicalize_wham_coordinates = wham_to_canonical
canonical_smplx_to_legacy = canonical_to_legacy_overlay
legacy_to_canonical = legacy_overlay_to_canonical


@dataclass(frozen=True, init=False)
class CameraModel:
    """Normalized-image orthographic or perspective camera.

    Coordinates passed to :meth:`project` are canonical SMPL-X coordinates.
    Image coordinates are normalized to ``[0, 1]`` conventionally, with image
    Y increasing downwards.  ``scale`` is normalized image units per meter for
    an orthographic camera; ``focal_length`` is normalized focal length for a
    perspective camera.

    ``model=`` is accepted as a compatibility alias for
    ``projection_type=``.  The custom initializer keeps this lightweight API
    pleasant for callers that construct a camera from a JSON job dictionary.
    """

    projection_type: str
    scale: float
    principal_point: Tuple[float, float]
    focal_length: float
    min_depth: float
    world_motion_available: bool

    def __init__(
        self,
        projection_type: str = "orthographic",
        scale: float = 1.0,
        principal_point: Sequence[float] = (0.5, 0.5),
        focal_length: float = 1.0,
        min_depth: float = 1e-4,
        world_motion_available: bool = False,
        **kwargs: Any,
    ) -> None:
        if "model" in kwargs:
            projection_type = kwargs.pop("model")
        if "camera_model" in kwargs:
            projection_type = kwargs.pop("camera_model")
        if "translation" in kwargs:
            principal_point = kwargs.pop("translation")
        if kwargs:
            unknown = ", ".join(sorted(kwargs))
            raise TypeError(f"Unexpected camera parameter(s): {unknown}")

        projection = str(projection_type).strip().lower()
        if projection in ("ortho", "orthographic"):
            projection = "orthographic"
        elif projection in ("perspective", "persp"):
            projection = "perspective"
        elif projection == "auto":
            # A concrete camera is required for projection.  Auto is resolved
            # by estimate_camera_from_anchors; treating it as orthographic is
            # a safe deterministic default for direct construction.
            projection = "orthographic"
        else:
            raise ValueError(f"Unsupported camera model: {projection_type!r}")

        pp = tuple(float(x) for x in principal_point)
        if len(pp) != 2 or not np.isfinite(pp).all():
            raise ValueError("principal_point must contain two finite values")
        if not np.isfinite(scale) or float(scale) <= 0.0:
            raise ValueError("scale must be positive and finite")
        if not np.isfinite(focal_length) or float(focal_length) <= 0.0:
            raise ValueError("focal_length must be positive and finite")
        if not np.isfinite(min_depth) or float(min_depth) <= 0.0:
            raise ValueError("min_depth must be positive and finite")

        object.__setattr__(self, "projection_type", projection)
        object.__setattr__(self, "scale", float(scale))
        object.__setattr__(self, "principal_point", (pp[0], pp[1]))
        object.__setattr__(self, "focal_length", float(focal_length))
        object.__setattr__(self, "min_depth", float(min_depth))
        object.__setattr__(self, "world_motion_available", bool(world_motion_available))

    @property
    def model(self) -> str:
        """Compatibility alias for :attr:`projection_type`."""

        return self.projection_type

    @property
    def mode(self) -> str:
        """Compatibility alias used by job settings."""

        return self.projection_type

    @classmethod
    def orthographic(
        cls,
        scale: float = 1.0,
        principal_point: Sequence[float] = (0.5, 0.5),
        **kwargs: Any,
    ) -> "CameraModel":
        return cls(
            projection_type="orthographic",
            scale=scale,
            principal_point=principal_point,
            **kwargs,
        )

    @classmethod
    def perspective(
        cls,
        focal_length: float = 1.0,
        principal_point: Sequence[float] = (0.5, 0.5),
        **kwargs: Any,
    ) -> "CameraModel":
        return cls(
            projection_type="perspective",
            focal_length=focal_length,
            principal_point=principal_point,
            **kwargs,
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CameraModel":
        """Construct a camera from snake_case or export-style JSON fields."""

        projection = data.get("projection_type", data.get("projectionType", data.get("cameraModel", data.get("model", "orthographic"))))
        principal = data.get("principal_point", data.get("principalPoint", data.get("translation", (0.5, 0.5))))
        return cls(
            projection_type=str(projection),
            scale=float(data.get("scale", 1.0)),
            principal_point=principal,
            focal_length=float(data.get("focal_length", data.get("focalLength", 1.0))),
            min_depth=float(data.get("min_depth", data.get("minDepth", 1e-4))),
            world_motion_available=bool(data.get("world_motion_available", data.get("worldMotionAvailable", False))),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Return JSON-compatible camera parameters."""

        return {
            "cameraModel": self.projection_type,
            "projectionType": self.projection_type,
            "scale": float(self.scale),
            "principalPoint": [float(self.principal_point[0]), float(self.principal_point[1])],
            "focalLength": float(self.focal_length),
            "worldMotionAvailable": bool(self.world_motion_available),
        }

    def project(self, points: np.ndarray) -> np.ndarray:
        """Project canonical XYZ points to normalized image UV coordinates."""

        array = np.asarray(points, dtype=np.float64)
        if array.ndim == 0 or array.shape[-1] < 3:
            raise ValueError("points must have shape (..., >=3)")
        xyz = array[..., :3]
        x = xyz[..., 0]
        # Canonical +Y is up; image +Y is down.
        image_y_axis = -xyz[..., 1]
        cx, cy = self.principal_point
        if self.projection_type == "orthographic":
            u = cx + self.scale * x
            v = cy + self.scale * image_y_axis
        else:
            z = xyz[..., 2]
            sign = np.where(z < 0.0, -1.0, 1.0)
            safe_z = np.where(np.abs(z) >= self.min_depth, z, sign * self.min_depth)
            u = cx + self.focal_length * x / safe_z
            v = cy + self.focal_length * image_y_axis / safe_z
        return np.stack((u, v), axis=-1)

    def unproject(self, points_2d: np.ndarray, depth: Union[float, np.ndarray]) -> np.ndarray:
        """Unproject normalized UV points using supplied canonical Z depth."""

        uv = np.asarray(points_2d, dtype=np.float64)
        if uv.ndim == 0 or uv.shape[-1] < 2:
            raise ValueError("points_2d must have shape (..., >=2)")
        z = np.asarray(depth, dtype=np.float64)
        z = np.broadcast_to(z, uv[..., 0].shape)
        cx, cy = self.principal_point
        if self.projection_type == "orthographic":
            x = (uv[..., 0] - cx) / self.scale
            y = -(uv[..., 1] - cy) / self.scale
        else:
            sign = np.where(z < 0.0, -1.0, 1.0)
            safe_z = np.where(np.abs(z) >= self.min_depth, z, sign * self.min_depth)
            x = (uv[..., 0] - cx) * safe_z / self.focal_length
            y = -(uv[..., 1] - cy) * safe_z / self.focal_length
        return np.stack((x, y, z), axis=-1)

    def reprojection_error(
        self,
        points_3d: np.ndarray,
        observed_2d: np.ndarray,
        weights: Optional[np.ndarray] = None,
    ) -> Union[float, np.ndarray]:
        """Return per-point UV error, or a weighted mean for scalar output."""

        predicted = self.project(points_3d)
        observed = np.asarray(observed_2d, dtype=np.float64)[..., :2]
        error = np.linalg.norm(predicted - observed, axis=-1)
        if weights is None:
            return error
        w = np.asarray(weights, dtype=np.float64)
        if w.shape != error.shape:
            w = np.broadcast_to(w, error.shape)
        denom = float(np.sum(np.clip(w, 0.0, None)))
        return float(np.sum(error * np.clip(w, 0.0, None)) / denom) if denom > 1e-12 else 0.0


def project_points(
    points: np.ndarray,
    camera: Optional[Union[CameraModel, str]] = None,
    *,
    mode: str = "orthographic",
    scale: float = 1.0,
    principal_point: Sequence[float] = (0.5, 0.5),
    focal_length: float = 1.0,
) -> np.ndarray:
    """Project points using a :class:`CameraModel` or a simple mode string."""

    if camera is None:
        camera = CameraModel(
            projection_type=mode,
            scale=scale,
            principal_point=principal_point,
            focal_length=focal_length,
        )
    elif isinstance(camera, str):
        camera = CameraModel(
            projection_type=camera,
            scale=scale,
            principal_point=principal_point,
            focal_length=focal_length,
        )
    if not isinstance(camera, CameraModel):
        raise TypeError("camera must be CameraModel, a model string, or None")
    return camera.project(points)


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    total = float(np.sum(weights))
    return np.sum(values * weights[..., None], axis=0) / total if total > 1e-12 else np.mean(values, axis=0)


def estimate_camera_from_anchors(
    points_3d: np.ndarray,
    points_2d: np.ndarray,
    *,
    mode: str = "auto",
    weights: Optional[np.ndarray] = None,
    min_depth: float = 1e-4,
) -> CameraModel:
    """Fit a normalized-image camera from visible 3D/2D anchor pairs.

    The fit is intentionally modest and deterministic: it estimates one
    shared scale/focal length plus a 2D principal-point translation.  This is
    enough to separate the person's image anchor from body-depth fusion while
    remaining usable in an offline lightweight profile.
    """

    xyz = np.asarray(points_3d, dtype=np.float64)
    uv = np.asarray(points_2d, dtype=np.float64)
    if xyz.shape[:-1] != uv.shape[:-1] or xyz.ndim == 0 or xyz.shape[-1] < 3 or uv.shape[-1] < 2:
        raise ValueError("points_3d and points_2d must share shape (..., >=3)/(.., >=2)")
    xyz = xyz[..., :3].reshape(-1, 3)
    uv = uv[..., :2].reshape(-1, 2)
    valid = np.isfinite(xyz).all(axis=1) & np.isfinite(uv).all(axis=1)
    if weights is None:
        w = np.ones((len(xyz),), dtype=np.float64)
    else:
        w = np.asarray(weights, dtype=np.float64).reshape(-1)
        if len(w) != len(xyz):
            raise ValueError("weights must match the number of anchors")
        valid &= np.isfinite(w)
    w = np.clip(np.nan_to_num(w, nan=0.0), 0.0, None)
    valid &= w > 0.0
    if int(np.count_nonzero(valid)) == 0:
        return CameraModel(projection_type="orthographic", scale=1.0, min_depth=min_depth)
    xyz = xyz[valid]
    uv = uv[valid]
    w = w[valid]

    requested = str(mode or "auto").strip().lower()
    if requested not in ("auto", "orthographic", "ortho", "perspective", "persp"):
        raise ValueError(f"Unsupported camera mode: {mode!r}")
    if requested == "auto":
        finite_depth = xyz[:, 2]
        depth_span = float(np.ptp(finite_depth)) if len(finite_depth) else 0.0
        depth_magnitude = float(np.median(np.abs(finite_depth))) if len(finite_depth) else 0.0
        requested = "perspective" if depth_span > max(0.15, depth_magnitude * 0.35) else "orthographic"
    projection = "perspective" if requested in ("perspective", "persp") else "orthographic"

    if projection == "orthographic":
        features = np.stack((xyz[:, 0], -xyz[:, 1]), axis=1)
    else:
        safe_z = np.where(np.abs(xyz[:, 2]) >= min_depth, xyz[:, 2], np.where(xyz[:, 2] < 0, -min_depth, min_depth))
        features = np.stack((xyz[:, 0] / safe_z, -xyz[:, 1] / safe_z), axis=1)

    feature_center = _weighted_mean(features, w)
    observed_center = _weighted_mean(uv, w)
    centered_features = features - feature_center
    centered_observed = uv - observed_center
    denominator = float(np.sum(w[:, None] * centered_features * centered_features))
    numerator = float(np.sum(w[:, None] * centered_features * centered_observed))
    fitted_scale = numerator / denominator if denominator > 1e-12 else 1.0
    if not np.isfinite(fitted_scale) or fitted_scale <= 1e-8:
        # A positive scale is required by the camera contract.  If anchors
        # are nearly collinear, estimate a robust ratio from their span.
        feature_span = float(np.ptp(features, axis=0).max())
        observed_span = float(np.ptp(uv, axis=0).max())
        fitted_scale = observed_span / feature_span if feature_span > 1e-8 else 1.0
    fitted_scale = float(np.clip(fitted_scale, 1e-5, 1e5))
    principal = _weighted_mean(uv - fitted_scale * features, w)

    if projection == "orthographic":
        return CameraModel(
            projection_type=projection,
            scale=fitted_scale,
            principal_point=(float(principal[0]), float(principal[1])),
            min_depth=min_depth,
        )
    return CameraModel(
        projection_type=projection,
        focal_length=fitted_scale,
        principal_point=(float(principal[0]), float(principal[1])),
        min_depth=min_depth,
    )


def fit_camera(*args: Any, **kwargs: Any) -> CameraModel:
    """Compatibility alias for :func:`estimate_camera_from_anchors`."""

    return estimate_camera_from_anchors(*args, **kwargs)


def estimate_image_anchor(*args: Any, **kwargs: Any) -> CameraModel:
    """Compatibility alias emphasizing the camera/body separation use case."""

    return estimate_camera_from_anchors(*args, **kwargs)


__all__ = [
    "CANONICAL_COORDINATE_SYSTEM",
    "WHAM_RAW_COORDINATE_SYSTEM",
    "LEGACY_OVERLAY_COORDINATE_SYSTEM",
    "CANONICAL_COORDINATE_METADATA",
    "CANONICAL_METADATA",
    "canonical_coordinate_metadata",
    "transform_coordinates",
    "wham_to_canonical",
    "canonicalize_wham_coordinates",
    "canonical_to_legacy_overlay",
    "canonical_smplx_to_legacy",
    "legacy_overlay_to_canonical",
    "legacy_to_canonical",
    "CameraModel",
    "project_points",
    "estimate_camera_from_anchors",
    "fit_camera",
    "estimate_image_anchor",
]
