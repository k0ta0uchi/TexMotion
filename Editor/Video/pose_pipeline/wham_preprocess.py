"""Offline per-video preprocessing contract for the native WHAM adapter.

The research WHAM pipeline normally produces ViTPose frame features and DPVO
camera poses before temporal inference.  This small runner keeps those stages
explicit and deterministic: callers provide local extractors, while this
module validates frame alignment and writes archives consumable by
``NativeWHAMAdapter``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import math
from typing import Any, Callable, Mapping, Optional, Sequence

import numpy as np

try:  # OpenCV is part of the normal Video2Motion profile.
    import cv2
except Exception:  # pragma: no cover - lightweight unit-test profile
    cv2 = None


def _invoke(
    stage: Any,
    frames: Sequence[np.ndarray],
    timestamps: np.ndarray,
    frame_indices: Optional[Sequence[int]] = None,
    frame_bboxes: Optional[Sequence[Any]] = None,
) -> Any:
    """Invoke a preprocessing stage while preserving source frame identity.

    Most extractors only need decoded frames and timestamps, but WHAM's
    diagnostics and sparse/trimmed video paths also pass the original frame
    numbers.  Keep the historical three-argument call shape for custom stages
    and fall back to the legacy one-argument form when a stage does not accept
    metadata.
    """
    if hasattr(stage, "extract_sequence"):
        stage = stage.extract_sequence
    if not callable(stage):
        raise TypeError("preprocess stage must be callable or expose extract_sequence")
    indices = list(frame_indices) if frame_indices is not None else list(range(len(frames)))
    if frame_bboxes is not None:
        try:
            return stage(
                frames,
                timestamps.tolist(),
                indices,
                frame_bboxes=list(frame_bboxes),
            )
        except TypeError:
            # Existing custom extractors use the historical three-argument
            # contract.  They remain valid; only HMR2-compatible runners opt
            # into the bbox-aware keyword.
            pass
    try:
        return stage(frames, timestamps.tolist(), indices)
    except TypeError:
        return stage(frames)


def extract_video_image_features(
    frames: Sequence[np.ndarray],
    extractor: Any,
    timestamps: Optional[Sequence[float]] = None,
    frame_indices: Optional[Sequence[int]] = None,
    frame_bboxes: Optional[Sequence[Any]] = None,
) -> np.ndarray:
    """Run a local frame feature extractor and enforce ``(frames, dimension)``."""
    count = len(frames)
    times = np.asarray(timestamps if timestamps is not None else np.arange(count) / 30.0, dtype=np.float32)
    if times.shape != (count,):
        raise ValueError(f"timestamps must have shape ({count},)")
    indices = list(frame_indices) if frame_indices is not None else list(range(count))
    if len(indices) != count:
        raise ValueError(f"frame_indices must have length {count}")
    raw = _invoke(extractor, frames, times, indices, frame_bboxes)
    if isinstance(raw, Mapping):
        raw = raw.get("features", raw.get("image_features", raw.get("imageFeatures")))
    raw_values = np.asarray(raw)
    if not np.issubdtype(raw_values.dtype, np.floating):
        raise ValueError(
            "image features must use a floating-point dtype (float32/float64), "
            f"got {raw_values.dtype}"
        )
    values = np.asarray(raw_values, dtype=np.float32)
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
    frame_indices: Optional[Sequence[int]] = None,
    frame_bboxes: Optional[Sequence[Any]] = None,
    metadata: Optional[dict] = None,
) -> dict:
    """Create the per-video archives required by WHAM optional stages."""
    if image_feature_extractor is None and camera_pose_extractor is None:
        raise ValueError("configure an image feature extractor and/or camera pose extractor")
    count = len(frames)
    times = np.asarray(timestamps if timestamps is not None else np.arange(count) / 30.0, dtype=np.float32)
    if times.shape != (count,):
        raise ValueError(f"timestamps must have shape ({count},)")
    indices = list(frame_indices) if frame_indices is not None else list(range(count))
    if len(indices) != count:
        raise ValueError(f"frame_indices must have length {count}")
    if frame_bboxes is not None:
        frame_bboxes = list(frame_bboxes)
        if len(frame_bboxes) != count:
            raise ValueError(f"frame_bboxes must have length {count}")
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    result = {"frames": count, "timestamps": times.tolist()}
    if image_feature_extractor is not None:
        features = extract_video_image_features(
            frames, image_feature_extractor, times, indices, frame_bboxes
        )
        path = root / "image_features.npz"
        np.savez_compressed(
            path,
            features=features,
            timestamps=times,
            frame_indices=np.asarray(indices, dtype=np.int64),
        )
        result.update({
            "imageFeaturePath": str(path),
            "featureShape": list(features.shape),
            "featureFrameIndices": [int(index) for index in indices],
            "featureFrameCount": int(count),
        })
    if camera_pose_extractor is not None:
        poses = _invoke(camera_pose_extractor, frames, times, indices)
        path = export_camera_poses(poses, root / "camera_poses.npz", count)
        result["cameraPosePath"] = str(path)
    manifest = root / "manifest.json"
    manifest.write_text(json.dumps({**(metadata or {}), **result}, indent=2), encoding="utf-8")
    result["manifestPath"] = str(manifest)
    return result


def export_vitpose_features_from_video(
    video_path: str | Path,
    output_path: str | Path,
    *,
    checkpoint_path: str | Path,
    model_definition: Any = None,
    config_path: Any = None,
    target_fps: Optional[float] = None,
    trim_start: float = 0.0,
    trim_end: float = 0.0,
    device: str = "auto",
    chunk_size: int = 32,
    runner: Any = None,
    progress: Optional[Callable[[float, int, int], None]] = None,
) -> dict:
    """Extract and persist an aligned ``vitpose_features.npz`` archive.

    The exporter is deliberately independent from the full pose extraction
    pipeline.  It samples the same timestamp sequence as ``process_video``
    (FPS and trim range included), invokes the configured ViTPose/HMR2
    feature runner in bounded chunks, and writes a self-describing archive
    atomically.  ``runner`` is injectable for tests or an embedding caller;
    production callers provide a checkpoint plus a compatible model
    definition/configuration.

    ``progress`` receives ``(fraction, processed_frames, total_frames)``.
    ``frame_indices`` are source-video frame numbers, while ``timestamps``
    are seconds relative to the source video, matching WHAM's preprocessor.
    """
    if cv2 is None:
        raise RuntimeError("OpenCV (cv2) is required for ViTPose feature extraction")
    source = Path(video_path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Input video file not found: {source}")
    if not checkpoint_path and runner is None:
        raise ValueError(
            "ViTPose checkpoint_path is required when no feature runner is supplied"
        )
    try:
        chunk = int(chunk_size)
    except (TypeError, ValueError):
        raise ValueError("chunk_size must be a positive integer") from None
    if chunk <= 0:
        raise ValueError("chunk_size must be a positive integer")

    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video file: {source}")
    try:
        original_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if not math.isfinite(original_fps) or original_fps <= 0.0:
            original_fps = 30.0
        total_source_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        duration = (
            float(total_source_frames) / original_fps
            if total_source_frames > 0 else 0.0
        )
        start = max(0.0, float(trim_start or 0.0))
        end = duration if float(trim_end or 0.0) <= 0.0 else min(
            duration, float(trim_end)
        )
        if duration <= 0.0:
            raise RuntimeError("Video has no readable frames")
        if end <= start:
            end = duration
        output_fps = float(target_fps) if target_fps and target_fps > 0.0 else original_fps
        sample_duration = max(0.0, end - start)
        sample_count = max(1, int(math.ceil(sample_duration * output_fps - 1.0e-9)))
        timestamps = np.asarray(
            [start + (index / output_fps) for index in range(sample_count)],
            dtype=np.float32,
        )
        source_frame_indices = np.rint(timestamps.astype(np.float64) * original_fps).astype(np.int64)
        source_frame_indices = np.clip(source_frame_indices, 0, max(0, total_source_frames - 1))
        if len(set(source_frame_indices.tolist())) == len(source_frame_indices.tolist()):
            frame_indices = source_frame_indices
        else:
            # When sampling rate exceeds source FPS or edge clipping produces
            # duplicate frame lookups, use strict sequential sample frame indices
            # so that every row has a unique identifier for feature extraction and WHAM integration.
            frame_indices = np.arange(sample_count, dtype=np.int64)

        if runner is None:
            # Import lazily so the lightweight MediaPipe/RTMPose profile does
            # not import torch merely because this module is loaded.
            from .vitpose_runner import create_vitpose_runner

            runner = create_vitpose_runner(
                checkpoint_path=checkpoint_path,
                model_definition=model_definition,
                config_path=config_path,
                device=device,
            )

        features_chunks = []
        total = len(frame_indices)
        for begin in range(0, total, chunk):
            end_index = min(total, begin + chunk)
            frames = []
            for position in range(begin, end_index):
                source_index = int(source_frame_indices[position])
                # Frame-index seeking is deterministic for CFR video and is
                # less fragile than repeated millisecond seeks on Windows.
                cap.set(cv2.CAP_PROP_POS_FRAMES, source_index)
                ok, frame = cap.read()
                if not ok or frame is None:
                    cap.set(cv2.CAP_PROP_POS_MSEC, float(timestamps[position]) * 1000.0)
                    ok, frame = cap.read()
                if not ok or frame is None:
                    raise RuntimeError(
                        f"Unable to decode source frame {source_index} "
                        f"(sample {position + 1}/{total})"
                    )
                frames.append(frame)
            values = runner.extract_sequence(
                frames,
                timestamps[begin:end_index].tolist(),
                frame_indices[begin:end_index].tolist(),
            )
            raw_values = np.asarray(values)
            if not np.issubdtype(raw_values.dtype, np.floating):
                raise ValueError(
                    "ViTPose runner output must use a floating-point dtype "
                    f"(float32/float64), got {raw_values.dtype}"
                )
            values = np.asarray(raw_values, dtype=np.float32)
            expected = end_index - begin
            if values.ndim != 2 or values.shape[0] != expected or values.shape[1] <= 0:
                raise ValueError(
                    f"ViTPose runner returned {values.shape}; expected ({expected}, D)"
                )
            if not np.isfinite(values).all():
                raise ValueError("ViTPose feature output contains non-finite values")
            features_chunks.append(values)
            if progress is not None:
                progress(float(end_index) / float(total), end_index, total)

        features = np.concatenate(features_chunks, axis=0).astype(np.float32, copy=False)
        if features.shape[0] != total:
            raise ValueError(
                f"ViTPose feature rows are not aligned: expected {total}, got {features.shape[0]}"
            )
        destination = Path(output_path).expanduser()
        if destination.suffix.lower() != ".npz":
            destination = destination.with_suffix(destination.suffix + ".npz" if destination.suffix else ".npz")
        destination = destination.resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(destination.name + ".tmp")
        runner_metadata = {}
        get_metadata = getattr(runner, "get_metadata", None)
        if callable(get_metadata):
            try:
                candidate = get_metadata()
                if isinstance(candidate, Mapping):
                    runner_metadata = dict(candidate)
            except Exception:
                runner_metadata = {}
        np.savez_compressed(
            str(temporary),
            features=features,
            timestamps=timestamps,
            frame_indices=frame_indices,
            source_frame_indices=source_frame_indices,
            fps=np.asarray(output_fps, dtype=np.float32),
            source_fps=np.asarray(original_fps, dtype=np.float32),
            source_video=np.asarray(str(source)),
            metadata=np.asarray(json.dumps(runner_metadata, ensure_ascii=False)),
        )
        # numpy appends .npz when the name has no .npz suffix.  The temporary
        # name intentionally ends in .tmp, so account for that behavior.
        temporary_npz = Path(str(temporary) + ".npz")
        if temporary_npz.is_file():
            temporary = temporary_npz
        os.replace(str(temporary), str(destination))
        if progress is not None:
            progress(1.0, total, total)
        return {
            "path": str(destination),
            "frames": int(total),
            "featureShape": [int(value) for value in features.shape],
            "fps": float(output_fps),
            "sourceFps": float(original_fps),
            "frameIndices": [int(value) for value in frame_indices.tolist()],
            "sourceFrameIndices": [int(value) for value in source_frame_indices.tolist()],
            "runnerMetadata": runner_metadata,
        }
    finally:
        cap.release()
        close = getattr(runner, "close", None) if runner is not None else None
        if callable(close):
            try:
                close()
            except Exception:
                pass


def _frame_descriptor(frame: np.ndarray) -> np.ndarray:
    """Build a compact descriptor for explicit offline tooling.

    This helper is intentionally dependency-light, but its output is not a
    learned WHAM feature. Native WHAM never calls it automatically; callers
    that use it must treat the resulting archive as an explicitly verified
    external input rather than substituting it for HMR2/ViTPose tokens.
    """
    value = np.asarray(frame)
    if value.ndim != 3 or value.shape[2] < 3:
        raise ValueError("WHAM image feature frames must be HxWx3 BGR images")
    value = value[..., :3].astype(np.float32) / 255.0
    if cv2 is not None:
        small = cv2.resize(value, (16, 16), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(value, cv2.COLOR_BGR2GRAY)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    else:
        # Keep the fallback pure NumPy for tests and minimal offline installs.
        y_idx = np.linspace(0, value.shape[0] - 1, 16).astype(np.int32)
        x_idx = np.linspace(0, value.shape[1] - 1, 16).astype(np.int32)
        small = value[np.ix_(y_idx, x_idx)]
        gray = np.mean(value, axis=2)
        gx = np.diff(gray, axis=1, prepend=gray[:, :1])
        gy = np.diff(gray, axis=0, prepend=gray[:1, :])
    gray_small = np.mean(small, axis=2)
    hist, _ = np.histogram(gray, bins=16, range=(0.0, 1.0), density=True)
    stats = np.concatenate(
        (
            small.reshape(-1),
            gray_small.reshape(-1),
            np.asarray(hist, dtype=np.float32),
            np.asarray(
                [
                    float(np.mean(value)),
                    float(np.std(value)),
                    float(np.mean(np.abs(gx))),
                    float(np.mean(np.abs(gy))),
                    float(np.percentile(gray, 10.0)),
                    float(np.percentile(gray, 90.0)),
                ],
                dtype=np.float32,
            ),
        )
    ).astype(np.float32)
    return np.nan_to_num(stats, nan=0.0, posinf=0.0, neginf=0.0)


def build_local_image_features(
    frames: Sequence[np.ndarray],
    feature_dim: int,
    timestamps: Optional[Sequence[float]] = None,
) -> np.ndarray:
    """Create an explicit diagnostic descriptor archive.

    The random projection is fixed across runs and machines for reproducible
    diagnostics. It is intentionally *not* used as an automatic WHAM learned
    feature fallback; production callers should pass a compatible HMR2/
    ViTPose extractor or a verified archive to ``run_wham_preprocess``.
    """
    count = len(frames)
    width = int(feature_dim)
    if width <= 0:
        raise ValueError("feature_dim must be positive")
    times = np.asarray(
        timestamps if timestamps is not None else np.arange(count) / 30.0,
        dtype=np.float32,
    )
    if times.shape != (count,):
        raise ValueError(f"timestamps must have shape ({count},)")
    descriptors = np.stack([_frame_descriptor(frame) for frame in frames], axis=0)
    # Include timing as a small periodic signal so repeated still frames do
    # not become indistinguishable to a temporal integrator.
    phase = np.column_stack((np.sin(times * 2.0 * math.pi), np.cos(times * 2.0 * math.pi)))
    descriptors = np.concatenate((descriptors, phase.astype(np.float32)), axis=1)
    # A fixed Gaussian projection is cheaper and more portable than loading a
    # second neural network, and avoids a hidden dependency on SciPy/sklearn.
    seed = np.uint64(0x5445584D4F54494F ^ (width * 0x9E3779B1))
    rng = np.random.default_rng(seed)
    projection = rng.standard_normal((descriptors.shape[1], width), dtype=np.float32)
    projection /= np.sqrt(float(descriptors.shape[1]))
    features = descriptors @ projection
    # Normalise each frame but preserve all-zero handling expected by WHAM's
    # Integrator (a genuinely missing row is still represented by zeros).
    scale = np.linalg.norm(features, axis=1, keepdims=True)
    features = features / np.maximum(scale, 1.0e-6)
    return np.nan_to_num(features.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)


def estimate_local_camera_poses(
    frames: Sequence[np.ndarray],
    timestamps: Optional[Sequence[float]] = None,
    camera_model: Optional[Mapping[str, Any]] = None,
) -> np.ndarray:
    """Estimate DPVO-compatible camera rows with a local CPU motion runner.

    When the official DPVO Python runtime is installed, callers can replace
    this function through ``camera_pose_extractor``.  The bundled fallback
    estimates a robust 2D affine motion from background features and emits
    ``[tx, ty, tz, qx, qy, qz, qw]`` rows.  It is deliberately conservative:
    an untrackable pair produces the previous pose, never NaNs or a large jump.
    """
    count = len(frames)
    if count == 0:
        return np.zeros((0, 7), dtype=np.float32)
    del timestamps  # reserved for future velocity scaling; pose rows are absolute.
    poses = np.zeros((count, 7), dtype=np.float32)
    poses[:, 6] = 1.0
    if count == 1 or cv2 is None:
        return poses
    first = np.asarray(frames[0])
    if first.ndim != 3 or first.shape[2] < 3:
        raise ValueError("camera pose frames must be HxWx3 BGR images")
    height, width = first.shape[:2]
    focal = float((camera_model or {}).get("fx", max(width, height)))
    focal = max(focal, 1.0)
    prev_gray = cv2.cvtColor(first[..., :3], cv2.COLOR_BGR2GRAY)
    previous_points = cv2.goodFeaturesToTrack(
        prev_gray, maxCorners=160, qualityLevel=0.01, minDistance=7, blockSize=7
    )
    cumulative = np.zeros(3, dtype=np.float32)
    cumulative_yaw = 0.0
    for index in range(1, count):
        current = np.asarray(frames[index])
        if current.ndim != 3 or current.shape[2] < 3:
            poses[index] = poses[index - 1]
            continue
        gray = cv2.cvtColor(current[..., :3], cv2.COLOR_BGR2GRAY)
        dx = dy = scale_delta = rotation_delta = 0.0
        if previous_points is not None and len(previous_points) >= 4:
            tracked, status, _ = cv2.calcOpticalFlowPyrLK(
                prev_gray, gray, previous_points, None,
                winSize=(21, 21), maxLevel=3,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03),
            )
            if tracked is not None and status is not None:
                good_old = previous_points[status.reshape(-1) == 1]
                good_new = tracked[status.reshape(-1) == 1]
                if len(good_old) >= 4:
                    matrix, inliers = cv2.estimateAffinePartial2D(
                        good_old, good_new, method=cv2.RANSAC,
                        ransacReprojThreshold=2.5, maxIters=100
                    )
                    if matrix is not None and np.isfinite(matrix).all():
                        dx = float(matrix[0, 2])
                        dy = float(matrix[1, 2])
                        scale_delta = float(np.hypot(matrix[0, 0], matrix[1, 0]) - 1.0)
                        rotation_delta = float(np.arctan2(matrix[1, 0], matrix[0, 0]))
        # Camera translation is inverse image motion.  Keep the estimate
        # bounded because WHAM consumes angular velocity, not metric SLAM.
        cumulative += np.asarray(
            [-dx / focal, -dy / focal, -scale_delta], dtype=np.float32
        )
        cumulative = np.clip(cumulative, -10.0, 10.0)
        cumulative_yaw = float(np.clip(cumulative_yaw + rotation_delta, -math.pi, math.pi))
        poses[index, :3] = cumulative
        poses[index, 2] = cumulative[2]
        poses[index, 3:7] = np.asarray(
            [0.0, 0.0, math.sin(cumulative_yaw * 0.5), math.cos(cumulative_yaw * 0.5)],
            dtype=np.float32,
        )
        previous_points = cv2.goodFeaturesToTrack(
            gray, maxCorners=160, qualityLevel=0.01, minDistance=7, blockSize=7
        )
        prev_gray = gray
    return poses


__all__ = [
    "extract_video_image_features",
    "export_vitpose_features_from_video",
    "export_camera_poses",
    "run_wham_preprocess",
    "build_local_image_features",
    "estimate_local_camera_poses",
]
