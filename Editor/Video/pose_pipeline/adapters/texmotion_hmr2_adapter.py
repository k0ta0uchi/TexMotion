"""Bundled HMR2 / 4D-Humans bridge for the official implementation.

TexMotion owns the adapter contract and the frame batching here.  The HMR2
source tree, checkpoint archive, and neutral SMPL file remain user supplied
because they are third-party/research assets.  This module is intentionally
lazy: importing the normal MediaPipe extractor never imports PyTorch or the
4D-Humans package.

The bridge follows the official 4D-Humans flow (``ViTDetDataset`` followed by
``HMR2``) and returns the common TexMotion quality-adapter schema.  A full
frame box is used because the VideoMotion pipeline has already selected the
single subject; this avoids requiring Detectron2 just to run one person.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import os
import shutil
import sys
import tarfile
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np


# OpenPose-25 slots used by the official SMPL wrapper -> COCO-17 ordering used
# by TexMotion's generic observation normalizer.
OPENPOSE25_TO_COCO17: Tuple[int, ...] = (
    0, 15, 16, 17, 18, 5, 2, 6, 3, 7, 4, 12, 9, 13, 10, 14, 11
)


def _as_path(value: Any) -> Optional[Path]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return Path(text).expanduser().resolve()
    except (OSError, RuntimeError):
        return Path(text).expanduser()


def _existing_file(value: Any) -> Optional[Path]:
    path = _as_path(value)
    return path if path is not None and path.is_file() else None


def _safe_extract_archive(archive_path: Path, destination: Path) -> Path:
    """Extract a checkpoint archive without allowing path traversal."""

    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()

    def safe_target(member_name: str) -> Path:
        target = (root / member_name).resolve()
        try:
            target.relative_to(root)
        except ValueError as exc:
            raise RuntimeError(
                f"Refusing unsafe HMR2 archive member outside extraction directory: {member_name}"
            ) from exc
        return target

    if archive_path.suffix.lower() == ".zip":
        with zipfile.ZipFile(str(archive_path)) as archive:
            for info in archive.infolist():
                target = safe_target(info.filename)
                if info.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(info, "r") as source, target.open("wb") as output:
                    shutil.copyfileobj(source, output)
        return destination

    with tarfile.open(str(archive_path), mode="r:*") as archive:
        for member in archive.getmembers():
            target = safe_target(member.name)
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            if not member.isfile():
                # Symlinks and devices are deliberately not materialized.
                continue
            source = archive.extractfile(member)
            if source is None:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with source, target.open("wb") as output:
                shutil.copyfileobj(source, output)
    return destination


def _find_checkpoint(root: Path) -> Optional[Path]:
    if root.is_file() and root.suffix.lower() in (".ckpt", ".pt", ".pth"):
        return root
    if not root.is_dir():
        return None
    candidates = sorted(
        (path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in (".ckpt", ".pt", ".pth")),
        key=lambda path: ("hmr2" not in path.name.lower(), len(path.parts), str(path).lower()),
    )
    return candidates[0] if candidates else None


def _resolve_checkpoint(config: Mapping[str, Any]) -> Tuple[Optional[Path], Optional[Path], Optional[str]]:
    """Resolve a checkpoint file, extracting the official archive if needed."""

    raw = (
        config.get("checkpointPath")
        or config.get("hmr2ModelPath")
        or config.get("modelPath")
        or config.get("model_path")
    )
    candidates: List[Path] = []
    explicit = _as_path(raw)
    if explicit is not None:
        candidates.append(explicit)
    model_directory = _as_path(config.get("modelDirectory"))
    if model_directory is not None:
        candidates.extend(
            [
                model_directory / "hmr2a_model.tar.gz",
                # The upstream 4D-Humans downloader names the same bundled
                # data archive ``hmr2_data.tar.gz`` in current revisions.
                model_directory / "hmr2_data.tar.gz",
                model_directory / "hmr2a.ckpt",
                model_directory / "hmr2b.ckpt",
                model_directory / "hmr2_model.tar.gz",
            ]
        )
    for candidate in candidates:
        if not candidate.exists():
            continue
        if candidate.is_file() and candidate.suffix.lower() in (".ckpt", ".pt", ".pth"):
            return candidate, candidate.parent, None
        if candidate.is_dir():
            found = _find_checkpoint(candidate)
            if found is not None:
                return found, candidate, None
        if candidate.is_file() and (
            candidate.name.lower().endswith((".tar.gz", ".tgz", ".tar", ".zip"))
        ):
            digest = hashlib.sha256(
                f"{candidate}:{candidate.stat().st_size}:{candidate.stat().st_mtime_ns}".encode("utf-8")
            ).hexdigest()[:16]
            base = model_directory or candidate.parent
            extract_root = base / ".texmotion_hmr2" / digest
            try:
                _safe_extract_archive(candidate, extract_root)
                found = _find_checkpoint(extract_root)
            except Exception as exc:
                return None, extract_root, f"Failed to extract HMR2 checkpoint archive '{candidate}': {exc}"
            if found is not None:
                return found, extract_root, None
            return None, extract_root, f"No .ckpt file was found after extracting HMR2 archive '{candidate}'."
    return None, None, (
        "Official HMR2 checkpoint was not found. Select hmr2a_model.tar.gz or a .ckpt "
        "file in TexMotion Settings (HMR2 Model Path)."
    )


def _resolve_runtime(config: Mapping[str, Any]) -> Tuple[Optional[Path], Optional[str]]:
    explicit = _as_path(config.get("runtimePath") or config.get("hmr2RuntimePath"))
    candidates: List[Path] = []
    if explicit is not None:
        candidates.append(explicit)
    env = _as_path(os.environ.get("TEXMOTION_HMR2_RUNTIME"))
    if env is not None:
        candidates.append(env)
    model_directory = _as_path(config.get("modelDirectory"))
    if model_directory is not None:
        candidates.extend([model_directory / "hmr2-runtime", model_directory / "4D-Humans"])
    for candidate in candidates:
        if candidate.is_dir() and (
            (candidate / "hmr2").is_dir() or (candidate / "pyproject.toml").is_file()
            or (candidate / "setup.py").is_file()
        ):
            return candidate, None
    # An explicit path is authoritative.  Falling through to an unrelated
    # globally installed package would make a typo in Settings appear to work
    # while loading a different HMR2 revision than the selected checkpoint.
    if explicit is not None:
        return explicit, (
            "Configured 4D-Humans/HMR2 runtime directory does not contain the official "
            f"'hmr2' package: {explicit}"
        )
    try:
        if importlib.util.find_spec("hmr2") is not None:
            return None, None
    except (ImportError, ValueError):
        pass
    return None, (
        "Official 4D-Humans/HMR2 runtime is not installed. Clone/install the official "
        "4D-Humans repository, then set HMR2 Runtime Path in TexMotion Settings."
    )


def _activate_runtime(runtime_root: Optional[Path]) -> None:
    if runtime_root is None:
        return
    candidates = [runtime_root, runtime_root.parent]
    for candidate in candidates:
        text = str(candidate)
        if text not in sys.path:
            sys.path.insert(0, text)


def _find_body_model(config: Mapping[str, Any], checkpoint_root: Optional[Path]) -> Optional[Path]:
    raw = config.get("bodyModelPath") or config.get("hmr2BodyModelPath")
    explicit = _as_path(raw)
    if explicit is not None:
        if explicit.is_file():
            return explicit
        if explicit.is_dir():
            for name in ("SMPL_NEUTRAL.pkl", "basicModel_neutral_lbs_10_207_0_v1.0.0.pkl"):
                found = next(iter(explicit.rglob(name)), None)
                if found is not None:
                    return found
    roots = [checkpoint_root, _as_path(config.get("modelDirectory"))]
    for root in roots:
        if root is None or not root.exists():
            continue
        for name in ("SMPL_NEUTRAL.pkl", "basicModel_neutral_lbs_10_207_0_v1.0.0.pkl"):
            found = next(iter(root.rglob(name)), None)
            if found is not None:
                return found
    return None


def _stage_body_model(body_model: Optional[Path], checkpoint_root: Optional[Path]) -> Optional[Path]:
    if body_model is None:
        return None
    if not body_model.is_file():
        raise FileNotFoundError(f"Configured HMR2 neutral SMPL body model was not found: {body_model}")
    root = (checkpoint_root or body_model.parent) / ".texmotion_smpl"
    root.mkdir(parents=True, exist_ok=True)
    target = root / "SMPL_NEUTRAL.pkl"
    if body_model.resolve() != target.resolve():
        shutil.copy2(str(body_model), str(target))
    return root


def _load_official_config(config_path: Path) -> Any:
    module = importlib.import_module("hmr2.configs")
    get_config = getattr(module, "get_config", None)
    if get_config is None:
        raise ImportError("The official HMR2 runtime does not expose hmr2.configs.get_config")
    attempts = (
        lambda: get_config(str(config_path), update_cachedir=True),
        lambda: get_config(config_file=str(config_path), update_cachedir=True),
        lambda: get_config(config_path=str(config_path), update_cachedir=True),
    )
    last_error: Optional[Exception] = None
    for attempt in attempts:
        try:
            return attempt()
        except (TypeError, ValueError, OSError) as exc:
            last_error = exc
    raise RuntimeError(f"Unable to load official HMR2 model_config.yaml: {last_error}")


def _find_config_file(checkpoint: Path, extraction_root: Optional[Path]) -> Optional[Path]:
    candidates = [
        checkpoint.parent.parent / "model_config.yaml",
        checkpoint.parent / "model_config.yaml",
    ]
    if extraction_root is not None:
        candidates.append(extraction_root / "model_config.yaml")
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    for root in (checkpoint.parent, extraction_root):
        if root is None or not root.exists():
            continue
        found = next(iter(root.rglob("model_config.yaml")), None)
        if found is not None:
            return found
    return None


def _to_numpy(value: Any) -> Optional[np.ndarray]:
    if value is None:
        return None
    try:
        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "cpu"):
            value = value.cpu()
        if hasattr(value, "numpy"):
            return np.asarray(value.numpy())
        return np.asarray(value)
    except Exception:
        return None


def _coerce_official_points(value: Any, dimensions: int) -> Optional[np.ndarray]:
    """Return official SMPL/OpenPose points in COCO-17 ordering."""
    array = _to_numpy(value)
    if array is None:
        return None
    while array.ndim > 2 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 2 or array.shape[1] < dimensions:
        return None
    array = np.asarray(array[:, :dimensions], dtype=np.float32)
    if array.shape[0] >= max(OPENPOSE25_TO_COCO17) + 1:
        array = array[list(OPENPOSE25_TO_COCO17)]
    elif array.shape[0] not in (17, 22):
        return None
    return np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)


def _coerce_official_joints(value: Any) -> Optional[np.ndarray]:
    """Return one HMR2 3D pose in COCO-17 joint order."""
    return _coerce_official_points(value, 3)


def _project_crop_points(
    points: Any,
    box_center: Any,
    box_size: Any,
    image_size: Any,
    crop_size: float = 256.0,
) -> Optional[np.ndarray]:
    """Map HMR2 crop coordinates back to normalized full-frame coordinates."""
    array = _to_numpy(points)
    center = _to_numpy(box_center)
    size = _to_numpy(box_size)
    image = _to_numpy(image_size)
    if array is None or center is None or size is None or image is None:
        return None
    array = np.asarray(array, dtype=np.float32)
    center = np.asarray(center, dtype=np.float32).reshape(-1)
    size = np.asarray(size, dtype=np.float32).reshape(-1)
    image = np.asarray(image, dtype=np.float32).reshape(-1)
    if size.size == 1:
        size = np.repeat(size, 2)
    if array.ndim != 2 or array.shape[1] < 2 or center.size < 2 or size.size < 2 or image.size < 2:
        return None
    xy = array[:, :2].copy()
    finite = np.isfinite(xy).all(axis=1)
    if not finite.any():
        return np.zeros((len(xy), 2), dtype=np.float32)
    finite_xy = xy[finite]
    maximum = float(np.max(np.abs(finite_xy))) if len(finite_xy) else 0.0
    # Official HMR2 calls ``perspective_projection`` with focal_length divided
    # by MODEL.IMAGE_SIZE and a zero camera centre.  Its output is therefore
    # crop-pixel coordinates expressed as ``pixels / crop_size`` around the
    # crop centre: x=-0.5 is the left edge, x=+0.5 the right edge.  Convert
    # that exact convention back to pixels before applying the inverse crop.
    # A few local wrappers expose actual crop pixels instead; retain a narrow
    # range check so those values are still accepted without changing the
    # official path.
    if maximum <= 4.0:
        crop_xy = xy * float(crop_size) + float(crop_size) * 0.5
    else:
        crop_xy = xy
    top_left = center[:2] - size[:2] * 0.5
    full_xy = top_left + (crop_xy / max(1.0, float(crop_size))) * size[:2]
    normalized = full_xy / np.maximum(image[:2], 1.0)
    return np.clip(np.nan_to_num(normalized, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)


class HMR2Adapter:
    """Official HMR2 runner implementing TexMotion's quality adapter contract."""

    def __init__(self, config: Optional[Mapping[str, Any]] = None):
        self.config = dict(config or {})
        self._torch = self.config.get("torch")
        self._device = str(self.config.get("device") or "cpu")
        try:
            self._batch_size = max(1, int(self.config.get("batchSize", 8)))
        except (TypeError, ValueError):
            self._batch_size = 8
        self._model: Any = None
        self._cfg: Any = None
        self._dataset_type: Any = None
        self._checkpoint: Optional[Path] = None
        self._runtime: Optional[Path] = None
        self._body_model: Optional[Path] = None
        self._error: Optional[str] = None
        self._preflight_status = "not_run"
        self._preflight_error: Optional[str] = None
        self._metadata: Dict[str, Any] = {
            "adapter": "texmotion_hmr2_adapter",
            "adapterImplementation": "texmotion_official_hmr2",
            "selectedBackend": "hmr2",
            "officialRunnerStatus": "not_initialized",
            "coordinateSystem": "smplx",
            "jointTopology": "coco17",
            "inferenceMode": "official_hmr2",
        }

    def initialize(self) -> bool:
        try:
            if self._torch is None:
                self._torch = importlib.import_module("torch")
            runtime, runtime_error = _resolve_runtime(self.config)
            if runtime_error:
                raise RuntimeError(runtime_error)
            self._runtime = runtime
            _activate_runtime(runtime)
            # The upstream config module builds its cache path from HOME;
            # Windows Python environments do not always define it.
            os.environ.setdefault("HOME", str(Path.home()))

            checkpoint, extraction_root, checkpoint_error = _resolve_checkpoint(self.config)
            if checkpoint_error:
                raise RuntimeError(checkpoint_error)
            if checkpoint is None:
                raise FileNotFoundError("Official HMR2 checkpoint resolution returned no file")
            self._checkpoint = checkpoint

            body_model = _find_body_model(self.config, extraction_root)
            self._body_model = body_model
            staged_body = _stage_body_model(body_model, extraction_root)
            config_path = _find_config_file(checkpoint, extraction_root)
            if config_path is None:
                raise FileNotFoundError(
                    f"HMR2 model_config.yaml was not found near checkpoint '{checkpoint}'. "
                    "Use the official 4D-Humans checkpoint archive, not a raw state-dict export."
                )
            cfg = _load_official_config(config_path)
            # ``get_config`` freezes its YACS node. Match the upstream
            # ``load_hmr2`` crop override and only defrost while applying our
            # explicit neutral-SMPL staging path.
            needs_bbox_shape = False
            try:
                needs_bbox_shape = (
                    str(cfg.MODEL.BACKBONE.TYPE).lower() == "vit"
                    and "BBOX_SHAPE" not in cfg.MODEL
                )
            except (AttributeError, KeyError):
                pass
            if needs_bbox_shape or (staged_body is not None and hasattr(cfg, "SMPL")):
                cfg.defrost()
                if needs_bbox_shape:
                    if int(cfg.MODEL.IMAGE_SIZE) != 256:
                        raise ValueError(
                            f"Official HMR2 ViT checkpoint expects MODEL.IMAGE_SIZE=256, got {cfg.MODEL.IMAGE_SIZE}"
                        )
                    cfg.MODEL.BBOX_SHAPE = [192, 256]
                if staged_body is not None and hasattr(cfg, "SMPL"):
                    cfg.SMPL.MODEL_PATH = str(staged_body)
                cfg.freeze()
            if hasattr(cfg, "SMPL"):
                configured_smpl_root = _as_path(getattr(cfg.SMPL, "MODEL_PATH", None))
                smpl_file = (
                    configured_smpl_root
                    if configured_smpl_root is not None and configured_smpl_root.is_file()
                    else (configured_smpl_root / "SMPL_NEUTRAL.pkl" if configured_smpl_root is not None else None)
                )
                if smpl_file is None or not smpl_file.is_file():
                    raise FileNotFoundError(
                        "Official HMR2 neutral SMPL body model is missing. Obtain "
                        "basicModel_neutral_lbs_10_207_0_v1.0.0.pkl from the SMPL provider "
                        "and set HMR2 Body Model Path in TexMotion Settings."
                    )
                if self._body_model is None:
                    self._body_model = smpl_file

            model_module = importlib.import_module("hmr2.models.hmr2")
            model_type = getattr(model_module, "HMR2", None)
            if model_type is None:
                raise ImportError("The official HMR2 runtime does not expose hmr2.models.hmr2.HMR2")
            load_kwargs = {"strict": False, "cfg": cfg}
            try:
                model = model_type.load_from_checkpoint(str(checkpoint), init_renderer=False, **load_kwargs)
            except TypeError:
                model = model_type.load_from_checkpoint(str(checkpoint), **load_kwargs)
            requested = self._device.lower().strip()
            if requested.startswith("cuda") and not bool(self._torch.cuda.is_available()):
                requested = "cpu"
            self._device = requested or "cpu"
            self._model = model.to(self._device)
            self._model.eval()
            dataset_module = importlib.import_module("hmr2.datasets.vitdet_dataset")
            self._dataset_type = getattr(dataset_module, "ViTDetDataset", None)
            if self._dataset_type is None:
                raise ImportError("The official HMR2 runtime does not expose ViTDetDataset")
            self._cfg = cfg
            self._preflight_status = "loaded"
            self._metadata.update(
                {
                    "officialRunnerStatus": "active",
                    "backendReady": True,
                    "checkpoint": str(checkpoint),
                    "modelPath": str(checkpoint),
                    "runtimePath": str(runtime) if runtime is not None else "installed-python-package",
                    "bodyModelPath": str(body_model) if body_model is not None else None,
                    "hmr2ImageFeaturesStatus": "official_hmr2_internal",
                    "batchSize": self._batch_size,
                }
            )
            return True
        except Exception as exc:
            self._error = f"Official HMR2 initialization failed: {exc}"
            self._preflight_status = "failed"
            self._preflight_error = self._error
            self._metadata.update(
                {
                    "officialRunnerStatus": "error",
                    "backendReady": False,
                    "error": self._error,
                    "preflightStatus": self._preflight_status,
                    "preflightError": self._preflight_error,
                }
            )
            return False

    def _image_size(self) -> float:
        try:
            value = self._cfg.MODEL.IMAGE_SIZE
            if isinstance(value, (list, tuple)):
                return float(value[0])
            return float(value)
        except Exception:
            return 256.0

    def _make_item(self, frame: np.ndarray) -> Dict[str, Any]:
        if frame is None:
            raise ValueError("HMR2 cannot infer a null frame")
        height, width = frame.shape[:2]
        boxes = np.asarray([[0.0, 0.0, float(max(1, width - 1)), float(max(1, height - 1))]], dtype=np.float32)
        dataset = self._dataset_type(self._cfg, frame, boxes)
        return dataset[0]

    def _collate(self, items: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        for key in items[0].keys():
            values = [item[key] for item in items]
            first = values[0]
            if hasattr(first, "dim") and hasattr(first, "unsqueeze"):
                result[key] = self._torch.stack(values, dim=0)
            elif isinstance(first, np.ndarray):
                result[key] = self._torch.from_numpy(np.stack(values, axis=0))
            elif np.isscalar(first):
                result[key] = self._torch.as_tensor(values)
            else:
                try:
                    result[key] = self._torch.as_tensor(np.stack(values, axis=0))
                except Exception:
                    result[key] = values
        return result

    def _move_batch(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        moved: Dict[str, Any] = {}
        for key, value in batch.items():
            moved[key] = value.to(self._device) if hasattr(value, "to") else value
        return moved

    def _result_rows(self, outputs: Mapping[str, Any], items: Sequence[Mapping[str, Any]], frames: Sequence[np.ndarray], indices: Sequence[int]) -> List[Dict[str, Any]]:
        joints = _to_numpy(outputs.get("pred_keypoints_3d"))
        projected = _to_numpy(outputs.get("pred_keypoints_2d"))
        confidence_value = outputs.get("pred_confidence")
        if confidence_value is None:
            confidence_value = outputs.get("pred_scores")
        confidences = _to_numpy(confidence_value)
        if joints is None:
            raise RuntimeError("Official HMR2 output did not contain pred_keypoints_3d")
        if joints.ndim == 2:
            joints = joints[None, ...]
        if projected is not None and projected.ndim == 2:
            projected = projected[None, ...]
        rows: List[Dict[str, Any]] = []
        for row_index, (item, frame, source_index) in enumerate(zip(items, frames, indices)):
            pose3d = _coerce_official_joints(joints[row_index])
            if pose3d is None:
                raise RuntimeError(f"Official HMR2 returned an invalid 3D joint tensor at frame {source_index}")
            pose2d = None
            if projected is not None and row_index < len(projected):
                projected_points = _coerce_official_points(projected[row_index], 2)
                pose2d = _project_crop_points(
                    projected_points,
                    item.get("box_center"),
                    item.get("box_size"),
                    item.get("img_size")
                    if item.get("img_size") is not None
                    else np.asarray([frame.shape[1], frame.shape[0]], dtype=np.float32),
                    crop_size=self._image_size(),
                )
            confidence = 1.0
            if confidences is not None:
                values = np.asarray(confidences[row_index] if confidences.ndim > 1 else confidences).reshape(-1)
                if values.size:
                    confidence = float(np.clip(np.nanmean(values), 0.0, 1.0))
            rows.append(
                {
                    "frameIndex": int(source_index),
                    "keypoints2d": pose2d,
                    "landmarks3d": pose3d,
                    "confidence": confidence,
                    "metadata": dict(self._metadata),
                }
            )
        return rows

    def infer_sequence(
        self,
        frames_bgr: Sequence[np.ndarray],
        timestamps_sec: Optional[Sequence[float]] = None,
        frame_indices: Optional[Sequence[int]] = None,
    ) -> List[Optional[Dict[str, Any]]]:
        if self._model is None or self._dataset_type is None:
            raise RuntimeError(self._error or "Official HMR2 adapter is not initialized")
        frames = list(frames_bgr or [])
        indices = list(frame_indices) if frame_indices is not None else list(range(len(frames)))
        if len(indices) != len(frames):
            raise ValueError("HMR2 frames and frame_indices have different lengths")
        if not frames:
            return []
        rows: List[Optional[Dict[str, Any]]] = []
        inference_context = getattr(self._torch, "inference_mode", None)
        if inference_context is None:
            inference_context = self._torch.no_grad
        for start in range(0, len(frames), self._batch_size):
            batch_frames = frames[start:start + self._batch_size]
            batch_indices = indices[start:start + self._batch_size]
            items = [self._make_item(frame) for frame in batch_frames]
            batch = self._move_batch(self._collate(items))
            with inference_context():
                outputs = self._model(batch)
            if not isinstance(outputs, Mapping):
                raise RuntimeError("Official HMR2 model returned a non-mapping output")
            rows.extend(self._result_rows(outputs, items, batch_frames, batch_indices))
        return rows

    def infer_frame(self, frame: np.ndarray, timestamp: float = 0.0, index: int = 0) -> Dict[str, Any]:
        return self.infer_sequence([frame], [timestamp], [index])[0]

    def get_metadata(self) -> Dict[str, Any]:
        result = dict(self._metadata)
        if self._error:
            result["error"] = self._error
        result["preflightStatus"] = self._preflight_status
        if self._preflight_error:
            result["preflightError"] = self._preflight_error
        return result

    def close(self) -> None:
        self._model = None
        self._dataset_type = None
        self._cfg = None
        if self._torch is not None and hasattr(self._torch, "cuda"):
            try:
                self._torch.cuda.empty_cache()
            except Exception:
                pass


def create_backend(config: Optional[Mapping[str, Any]] = None) -> HMR2Adapter:
    """Factory used by :class:`PyTorchPoseBackend`."""
    return HMR2Adapter(config)


__all__ = [
    "HMR2Adapter",
    "OPENPOSE25_TO_COCO17",
    "_coerce_official_joints",
    "_project_crop_points",
    "_resolve_checkpoint",
    "_safe_extract_archive",
    "create_backend",
]
