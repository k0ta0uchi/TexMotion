"""Offline per-video preprocessing contract for the native WHAM adapter.

The research WHAM pipeline normally produces ViTPose frame features and DPVO
camera poses before temporal inference.  This small runner keeps those stages
explicit and deterministic: callers provide local extractors, while this
module validates frame alignment and writes archives consumable by
``NativeWHAMAdapter``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

import numpy as np


def _invoke(stage: Any, frames: Sequence[np.ndarray], timestamps: np.ndarray) -> Any:
    if hasattr(stage, "extract_sequence"):
        stage = stage.extract_sequence
    if not callable(stage):
        raise TypeError("preprocess stage must be callable or expose extract_sequence")
    try:
        return stage(frames, timestamps.tolist(), list(range(len(frames))))
    except TypeError:
        return stage(frames)


def extract_video_image_features(
    frames: Sequence[np.ndarray],
    extractor: Any,
    timestamps: Optional[Sequence[float]] = None,
) -> np.ndarray:
    """Run a local frame feature extractor and enforce ``(frames, dimension)``."""
    count = len(frames)
    times = np.asarray(timestamps if timestamps is not None else np.arange(count) / 30.0, dtype=np.float32)
    if times.shape != (count,):
        raise ValueError(f"timestamps must have shape ({count},)")
    values = np.asarray(_invoke(extractor, frames, times), dtype=np.float32)
    if values.ndim != 2 or values.shape[0] != count:
        raise ValueError(f"image features must have shape ({count}, D), got {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError("image features contain non-finite values")
    return values


def export_camera_poses(camera_poses: Any, output_path: str | Path, frame_count: int) -> Path:
    """Write aligned DPVO poses as ``cameraPoses`` rows ``[t, q]`` (N x 7)."""
    values = np.asarray(camera_poses, dtype=np.float32)
    if values.shape != (frame_count, 7):
        raise ValueError(f"camera poses must have shape ({frame_count}, 7), got {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError("camera poses contain non-finite values")
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(target, cameraPoses=values)
    return target


def run_wham_preprocess(
    frames: Sequence[np.ndarray],
    output_dir: str | Path,
    *,
    image_feature_extractor: Any = None,
    camera_pose_extractor: Any = None,
    timestamps: Optional[Sequence[float]] = None,
    metadata: Optional[dict] = None,
) -> dict:
    """Create the per-video archives required by WHAM optional stages."""
    if image_feature_extractor is None and camera_pose_extractor is None:
        raise ValueError("configure an image feature extractor and/or camera pose extractor")
    count = len(frames)
    times = np.asarray(timestamps if timestamps is not None else np.arange(count) / 30.0, dtype=np.float32)
    if times.shape != (count,):
        raise ValueError(f"timestamps must have shape ({count},)")
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    result = {"frames": count, "timestamps": times.tolist()}
    if image_feature_extractor is not None:
        features = extract_video_image_features(frames, image_feature_extractor, times)
        path = root / "image_features.npz"
        np.savez_compressed(path, features=features, timestamps=times)
        result.update({"imageFeaturePath": str(path), "featureShape": list(features.shape)})
    if camera_pose_extractor is not None:
        poses = _invoke(camera_pose_extractor, frames, times)
        path = export_camera_poses(poses, root / "camera_poses.npz", count)
        result["cameraPosePath"] = str(path)
    manifest = root / "manifest.json"
    manifest.write_text(json.dumps({**(metadata or {}), **result}, indent=2), encoding="utf-8")
    result["manifestPath"] = str(manifest)
    return result


__all__ = ["extract_video_image_features", "export_camera_poses", "run_wham_preprocess"]
