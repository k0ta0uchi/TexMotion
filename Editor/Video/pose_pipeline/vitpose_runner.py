"""Small, dependency-light ViTPose image-feature runner for WHAM.

The WHAM adapter consumes an image-feature extractor through a deliberately
small contract::

    runner.extract_sequence(frames, timestamps=None, frame_indices=None)
    runner(frames, timestamps=None, frame_indices=None)

Both methods return a finite ``numpy.ndarray`` with shape ``(N, D)``.  Frames
are expected to be BGR ``H x W x 3`` arrays, as produced by OpenCV.  The
default deterministic preprocessing converts BGR to RGB, resizes the whole
frame to ``(256, 192)`` (height, width), scales pixels to ``[0, 1]``, and
applies ImageNet mean/std normalization.  ``input_size``, ``mean``, ``std``,
and ``color_order`` can be overridden in the constructor/config.

This module intentionally does not import PyTorch at module import time.  A
runtime PyTorch installation is loaded lazily on the first ``initialize`` or
``extract_sequence`` call.  Tests and embedded callers can inject a compatible
``torch_module`` object instead.

Configuration/API expectations
------------------------------

The minimal configuration is a local ``checkpoint_path`` plus one model
definition hook:

* ``model_factory``/``modelFactory``: callable receiving the resolved model
  config mapping and returning a model object.  Official WHAM-style factories
  whose parameter is named ``checkpoint_path``/``checkpoint_pth`` receive the
  resolved checkpoint path instead.  A factory that loads that checkpoint
  itself must set ``factory_loads_checkpoint=True``; or
* ``model_definition``/``modelDefinition``: a Python module name, ``.py``
  file, callable, or model instance.  A module is searched for
  ``create_model``, ``build_model``, ``get_model``, ``make_model``, ``Model``,
  and ``ViTPose`` in that order.

``config_path``/``configPath`` may point to JSON, YAML (when PyYAML is
installed), or a Python config module exposing ``CONFIG``, ``MODEL_CONFIG``,
or ``config``.  ``model_config`` may instead be supplied as a mapping.  The
loaded mapping is passed unchanged to the model-definition hook, so project
specific architecture keys remain available.  Common runner aliases are
accepted in both snake_case and the camelCase names used by TexMotion, for
example ``imageFeatureBackbonePath`` for the checkpoint and
``modelDefinitionPath`` for the definition file.

The checkpoint is loaded with ``torch.load(path, map_location=device)`` and
the standard ``state_dict``/``model_state_dict``/``weights`` wrappers are
recognized.  A strict state-dict load is used by default.  If model code or
weights are incompatible, :class:`ViTPoseRunnerError` identifies the local
path, failing stage, and the configuration action needed to correct it.

The runner is intentionally an adapter, not a bundled ViTPose model
definition: ViTPose releases differ in architecture/config and their
checkpoint alone is not sufficient to reconstruct a model.  Supplying the
definition/config hook makes that boundary explicit while keeping WHAM's
offline installation free of a hard mmpose dependency.
"""

from __future__ import annotations

import importlib
import importlib.util
import inspect
import hashlib
import json
import math
import os
import sys
import threading
from contextlib import nullcontext
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Mapping, Optional, Sequence

import numpy as np


class ViTPoseRunnerError(RuntimeError):
    """Actionable error raised for runner setup, inference, or output faults.

    ``error_code`` is intentionally stable because the editor and exported
    diagnostics consume it without parsing the human-readable message.
    """

    def __init__(
        self,
        message: str,
        *,
        error_code: str = "feature_contract_failed",
        next_action: Optional[str] = None,
    ) -> None:
        super().__init__(str(message))
        self.error_code = str(error_code)
        self.next_action = next_action


# Friendly spelling aliases are kept because existing integrations use both
# ``ViTPose`` and ``VitPose``.  The aliases are assigned again at the bottom
# after the main class definition.
VitPoseRunnerError = ViTPoseRunnerError


_MISSING = object()
_DEFAULT_INPUT_SIZE = (256, 192)
_DEFAULT_MEAN = (0.485, 0.456, 0.406)
_DEFAULT_STD = (0.229, 0.224, 0.225)

_CHECKPOINT_KEYS = (
    "checkpoint_path",
    "checkpointPath",
    "HMR2CheckpointPath",
    "hmr2CheckpointPath",
    "hmr2_checkpoint_path",
    "checkpoint",
    "model_path",
    "modelPath",
    "image_feature_backbone_path",
    "imageFeatureBackbonePath",
    "vitpose_checkpoint",
    "vitposeCheckpoint",
)
_MODEL_DEFINITION_KEYS = (
    "model_definition",
    "modelDefinition",
    "model_definition_path",
    "modelDefinitionPath",
    "model_definition_module",
    "modelDefinitionModule",
    "model_def",
    "modelDef",
    "model_code",
    "modelCode",
    "model_module",
    "modelModule",
)
_MODEL_FACTORY_KEYS = (
    "model_factory",
    "modelFactory",
    "build_model",
    "buildModel",
    "factory",
)
_CONFIG_PATH_KEYS = (
    "config_path",
    "configPath",
    "model_config_path",
    "modelConfigPath",
)
_MODEL_CONFIG_KEYS = (
    "model_config",
    "modelConfig",
    "architecture_config",
    "architectureConfig",
)
_TORCH_KEYS = ("torch_module", "torchModule", "torch")
_FEATURE_METHOD_KEYS = ("feature_method", "featureMethod", "forward_method", "forwardMethod")
_OUTPUT_KEY_KEYS = ("output_key", "outputKey", "feature_key", "featureKey")
_OUTPUT_INDEX_KEYS = ("output_index", "outputIndex", "feature_index", "featureIndex")
_RUNNER_KIND_KEYS = (
    "runtime",
    "runner_kind",
    "runnerKind",
    "feature_runner_kind",
    "featureRunnerKind",
    "feature_runtime",
    "featureRuntime",
    "model_runtime",
    "modelRuntime",
)
_CROP_MODE_KEYS = (
    "crop_mode",
    "cropMode",
    "bbox_mode",
    "bboxMode",
    "person_crop",
    "personCrop",
    "use_person_crop",
    "usePersonCrop",
)
_ENCODE_KEYS = ("encode", "encode_features", "encodeFeatures", "feature_encode", "featureEncode")
_BBOX_REQUIRED_KEYS = (
    "require_bboxes",
    "requireBboxes",
    "require_person_bboxes",
    "requirePersonBboxes",
)
_FACTORY_LOADS_CHECKPOINT_KEYS = (
    "factory_loads_checkpoint",
    "factoryLoadsCheckpoint",
    "model_factory_loads_checkpoint",
    "modelFactoryLoadsCheckpoint",
)

_FEATURE_OUTPUT_KEYS = (
    "features",
    "feature",
    "embeddings",
    "embedding",
    "image_features",
    "imageFeatures",
    "backbone",
    "last_hidden_state",
    "output",
)
_NON_IMAGE_OUTPUT_KEYS = {
    "pred_keypoints_3d",
    "pred_keypoints_2d",
    "pred_smpl_params",
    "pred_smpl",
    "pred_cam",
    "pred_camera",
    "camera",
    "camera_params",
    "smpl_params",
    "smpl_pose",
    "poses",
    "pose",
    "heatmaps",
    "heatmap",
    "logits",
    "preds",
}


def _first(mapping: Mapping[str, Any], names: Sequence[str], default: Any = None) -> Any:
    """Return the first configured, non-``None`` value for a list of aliases."""

    for name in names:
        if name in mapping and mapping[name] is not None:
            return mapping[name]
    return default


def _as_path(value: Any) -> Optional[Path]:
    if value is None:
        return None
    if isinstance(value, Path):
        return value.expanduser()
    if isinstance(value, (str, bytes)):
        return Path(value).expanduser()
    return None


_VITPOSE_SHA256_CACHE: Dict[str, Tuple[float, int, str]] = {}
_VITPOSE_SHA256_LOCK = threading.Lock()


def _sha256_file(path: Optional[Path]) -> Optional[str]:
    """Return a checkpoint hash without keeping the whole file in memory."""

    if path is None:
        return None
    try:
        resolved = path.expanduser().resolve() if isinstance(path, Path) else Path(path).expanduser().resolve()
        if not resolved.is_file():
            return None
        stat = resolved.stat()
        cache_key = os.path.normcase(str(resolved))
        with _VITPOSE_SHA256_LOCK:
            cached = _VITPOSE_SHA256_CACHE.get(cache_key)
            if cached is not None and cached[0] == stat.st_mtime and cached[1] == stat.st_size:
                return cached[2]
        digest = hashlib.sha256()
        with resolved.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        hexdigest = digest.hexdigest()
        with _VITPOSE_SHA256_LOCK:
            _VITPOSE_SHA256_CACHE[cache_key] = (stat.st_mtime, stat.st_size, hexdigest)
        return hexdigest
    except OSError:
        return None


def _stable_mapping_hash(value: Mapping[str, Any]) -> str:
    """Hash a config mapping for provenance without requiring JSON-native values."""

    try:
        encoded = json.dumps(value, sort_keys=True, default=str, separators=(",", ":")).encode("utf-8")
    except Exception:
        encoded = repr(sorted((str(key), repr(item)) for key, item in value.items())).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _normalise_input_size(value: Any) -> tuple[int, int]:
    if value is None:
        return _DEFAULT_INPUT_SIZE
    if isinstance(value, (int, np.integer)):
        value = (int(value), int(value))
    try:
        values = tuple(int(part) for part in value)
    except (TypeError, ValueError):
        raise ValueError("input_size must be an integer or a (height, width) pair") from None
    if len(values) != 2 or any(part <= 0 for part in values):
        raise ValueError("input_size must contain two positive values: (height, width)")
    return values


def _normalise_channel_values(value: Any, name: str) -> tuple[float, float, float]:
    try:
        values = tuple(float(part) for part in value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must contain three numeric channel values") from None
    if len(values) != 3 or not np.isfinite(np.asarray(values, dtype=np.float64)).all():
        raise ValueError(f"{name} must contain three finite channel values")
    return values


def _configured_bool(mapping: Mapping[str, Any], names: Sequence[str], default: bool) -> bool:
    """Read a boolean option without treating non-empty strings as truthy by accident."""

    value = _first(mapping, names, _MISSING)
    if value is _MISSING:
        return bool(default)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"", "0", "false", "no", "off", "none", "null"}:
            return False
        if normalized in {"1", "true", "yes", "on"}:
            return True
    return bool(value)


def _runner_kind_is_hmr2(value: Any) -> bool:
    normalized = str(value or "").strip().lower().replace("-", "_")
    return normalized in {
        "hmr2",
        "hmr_2",
        "4dhumans",
        "4d_humans",
        "hmr2_vitpose",
        "hmr2_vit",
        "hmr2_feature",
    }


def _coerce_frame_bbox(bbox: Any, frame_shape: Sequence[int]) -> Optional[tuple[float, float, float]]:
    """Normalize common person-box forms to ``(center_x, center_y, side_px)``.

    WHAM's HMR2 preprocessing stores a center/scale triplet where ``scale``
    is the crop side divided by 200.  Native WHAM passes that exact form in a
    mapping so a small/portrait frame cannot be mistaken for normalized
    coordinates.  Array callers may use either ``[cx, cy, scale]`` or
    ``[x1, y1, x2, y2]``; values in ``[0, 1]`` are treated as normalized only
    when every coordinate is in that range.
    """

    if bbox is None:
        return None
    try:
        height, width = int(frame_shape[0]), int(frame_shape[1])
    except (IndexError, TypeError, ValueError):
        return None
    if height <= 0 or width <= 0:
        return None

    normalized = False
    format_name = ""
    value = bbox
    if isinstance(value, Mapping):
        # Preserve flags on the outer mapping when ``bbox`` itself is a list;
        # this is the common detector shape ``{bbox: [...], normalized: true}``.
        outer = value
        normalized = _configured_bool(outer, ("normalized", "normalised"), False)
        format_name = str(outer.get("format", outer.get("bboxFormat", ""))).strip().lower()
        nested = outer.get("bbox", outer.get("box", outer.get("bounding_box", _MISSING)))
        if nested is not _MISSING:
            value = nested
            if isinstance(value, Mapping):
                normalized = _configured_bool(
                    value, ("normalized", "normalised"), normalized
                )
                format_name = str(
                    value.get("format", value.get("bboxFormat", format_name))
                ).strip().lower()
        if isinstance(value, Mapping):
            if all(key in value for key in ("cx", "cy")):
                scale = value.get("scale", value.get("size", value.get("side", value.get("bbox_size"))))
                value = [value.get("cx"), value.get("cy"), scale]
            elif all(key in value for key in ("center_x", "center_y")):
                scale = value.get("scale", value.get("size", value.get("side", value.get("bbox_size"))))
                value = [value.get("center_x"), value.get("center_y"), scale]
            elif all(key in value for key in ("x1", "y1", "x2", "y2")):
                value = [value.get("x1"), value.get("y1"), value.get("x2"), value.get("y2")]
            elif all(key in value for key in ("left", "top", "right", "bottom")):
                value = [value.get("left"), value.get("top"), value.get("right", value.get("x2")), value.get("bottom", value.get("y2"))]
            else:
                return None
    try:
        numbers = np.asarray(value, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        return None
    if numbers.size not in (3, 4) or not np.isfinite(numbers).all():
        return None

    if numbers.size == 4 or format_name in {"xyxy", "x1y1x2y2", "corners"}:
        x1, y1, x2, y2 = (float(part) for part in numbers[:4])
        if not normalized and numbers.size == 4 and np.all(np.abs(numbers[:4]) <= 1.0):
            normalized = True
        if normalized:
            x1, x2 = x1 * width, x2 * width
            y1, y2 = y1 * height, y2 * height
        center_x = (x1 + x2) * 0.5
        center_y = (y1 + y2) * 0.5
        side = max(abs(x2 - x1), abs(y2 - y1))
    else:
        center_x, center_y, scale = (float(part) for part in numbers[:3])
        if scale <= 0.0:
            return None
        if normalized or (np.all(np.abs(numbers[:3]) <= 1.0) and format_name in {"normalized", "cxcys_normalized"}):
            center_x *= width
            center_y *= height
            side = scale * max(width, height)
        else:
            # Official HMR2/WHAM center-scale convention: crop side = 200*s.
            side = scale * 200.0

    if not np.isfinite([center_x, center_y, side]).all() or side <= 1.0e-6:
        return None
    return float(center_x), float(center_y), float(side)


def _crop_frame_to_bbox(frame: Any, bbox: Any, input_size: Sequence[int]) -> Optional[np.ndarray]:
    """Crop one BGR/RGB frame using WHAM's square HMR2 person crop contract."""

    value = np.asarray(frame)
    if value.ndim != 3 or value.shape[2] < 3:
        return None
    parsed = _coerce_frame_bbox(bbox, value.shape)
    if parsed is None:
        return None
    center_x, center_y, side = parsed
    # Keep the zero padding behavior of WHAM's ``crop`` helper for a person
    # partly outside the image; this is important for source-frame alignment.
    x0 = int(math.floor(center_x - side * 0.5))
    y0 = int(math.floor(center_y - side * 0.5))
    x1 = int(math.ceil(center_x + side * 0.5))
    y1 = int(math.ceil(center_y + side * 0.5))
    crop_width = max(1, x1 - x0)
    crop_height = max(1, y1 - y0)
    crop = np.zeros((crop_height, crop_width, 3), dtype=np.float32)
    source_x0 = max(0, x0)
    source_y0 = max(0, y0)
    source_x1 = min(value.shape[1], x1)
    source_y1 = min(value.shape[0], y1)
    if source_x1 > source_x0 and source_y1 > source_y0:
        crop_y0 = source_y0 - y0
        crop_x0 = source_x0 - x0
        crop[crop_y0:crop_y0 + (source_y1 - source_y0), crop_x0:crop_x0 + (source_x1 - source_x0)] = value[
            source_y0:source_y1, source_x0:source_x1, :3
        ]
    return _resize_bilinear(crop, int(input_size[0]), int(input_size[1]))


def _load_torch(injected: Any = None) -> Any:
    """Resolve the optional torch dependency only when runtime work starts."""

    if injected is not None:
        return injected
    try:
        return importlib.import_module("torch")
    except Exception as exc:  # pragma: no cover - depends on the host profile
        raise ViTPoseRunnerError(
            "ViTPose feature extraction requires PyTorch at runtime, but importing "
            f"torch failed: {exc}. Install the optional requirements-pytorch.txt "
            "profile or pass a compatible torch_module test seam."
        ) from exc


def _load_module_reference(reference: Any, purpose: str) -> Any:
    """Load a module/file reference while retaining useful source diagnostics."""

    if isinstance(reference, ModuleType):
        return reference
    if not isinstance(reference, (str, Path)):
        return reference

    raw_reference = str(reference)
    # ``Path`` objects and existing files are unambiguously file references.
    # Do not split a Windows drive letter (``C:\\...``) as a
    # module:attribute separator.  A string may still use ``file.py:hook``;
    # parse that only when the portion before the colon is file-like.  This
    # leaves ``package.module:hook`` available for normal Python imports.
    path_candidate = Path(raw_reference).expanduser()
    candidate: Optional[Path] = None
    separator = ""
    attribute = ""
    if isinstance(reference, Path) or path_candidate.is_file() or path_candidate.suffix.lower() == ".py":
        candidate = path_candidate
    elif path_candidate.is_dir() and ":" not in raw_reference:
        raise ViTPoseRunnerError(
            f"ViTPose {purpose} '{path_candidate}' is not a file. Configure a Python "
            "model_definition/modelDefinition module or file."
        )
    else:
        possible_path, possible_separator, possible_attribute = raw_reference.rpartition(":")
        possible_candidate = Path(possible_path).expanduser()
        if possible_separator and (
            possible_candidate.is_file() or possible_candidate.suffix.lower() == ".py"
        ):
            candidate = possible_candidate
            separator = possible_separator
            attribute = possible_attribute
    if candidate is not None:
        if not candidate.is_file():
            raise ViTPoseRunnerError(
                f"ViTPose {purpose} '{candidate}' is not a file. Configure a Python "
                "model_definition/modelDefinition module or file."
            )
        module_name = "_texmotion_vitpose_" + str(abs(hash(str(candidate.resolve()))))
        try:
            spec = importlib.util.spec_from_file_location(module_name, str(candidate))
            if spec is None or spec.loader is None:
                raise ImportError("Python import spec could not be created")
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
        except Exception as exc:
            sys.modules.pop(module_name, None)
            raise ViTPoseRunnerError(
                f"Unable to import ViTPose {purpose} '{candidate}': {exc}. "
                "Check the model code imports and model-definition/config paths."
            ) from exc
        if separator and attribute:
            try:
                return getattr(module, attribute)
            except AttributeError as exc:
                raise ViTPoseRunnerError(
                    f"ViTPose {purpose} '{candidate}' has no '{attribute}' hook. "
                    "Expose create_model/build_model or use module.py:hook."
                ) from exc
        return module

    # A dotted module name or a module:attribute reference is useful when the
    # official ViTPose/mmpose stack is installed in a project environment.
    module_name, separator, attribute = raw_reference.partition(":")
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:
        raise ViTPoseRunnerError(
            f"Unable to import ViTPose {purpose} '{raw_reference}': {exc}. "
            "Set model_definition/modelDefinition to an importable module or .py file."
        ) from exc
    if separator and attribute:
        try:
            return getattr(module, attribute)
        except AttributeError as exc:
            raise ViTPoseRunnerError(
                f"ViTPose {purpose} module '{module_name}' has no '{attribute}' hook."
            ) from exc
    return module


def _load_config_reference(reference: Any) -> dict[str, Any]:
    """Load JSON/YAML/Python config values without making YAML mandatory."""

    if reference is None:
        return {}
    if isinstance(reference, Mapping):
        return dict(reference)
    if callable(reference) and not isinstance(reference, (str, Path)):
        try:
            loaded = reference()
        except TypeError:
            loaded = reference
        if isinstance(loaded, Mapping):
            return dict(loaded)
        raise ViTPoseRunnerError("ViTPose model config hook must return a mapping")

    source = _as_path(reference)
    if source is None:
        raise ViTPoseRunnerError("ViTPose model config must be a mapping or local config path")
    if not source.is_file():
        # A dotted Python module is allowed as a config hook too.
        module = _load_module_reference(reference, "config")
        for name in ("MODEL_CONFIG", "CONFIG", "model_config", "config"):
            value = getattr(module, name, _MISSING)
            if isinstance(value, Mapping):
                return dict(value)
        getter = getattr(module, "get_config", None)
        if callable(getter):
            value = getter()
            if isinstance(value, Mapping):
                return dict(value)
        raise ViTPoseRunnerError(
            f"ViTPose config '{reference}' exposes no CONFIG/MODEL_CONFIG mapping"
        )

    suffix = source.suffix.lower()
    try:
        if suffix == ".json":
            value = json.loads(source.read_text(encoding="utf-8"))
        elif suffix in (".yaml", ".yml"):
            try:
                import yaml  # type: ignore
            except Exception as exc:  # pragma: no cover - optional dependency
                raise ViTPoseRunnerError(
                    f"ViTPose YAML config '{source}' needs PyYAML; install it or use JSON/Python config: {exc}"
                ) from exc
            value = yaml.safe_load(source.read_text(encoding="utf-8"))
        elif suffix == ".py":
            module = _load_module_reference(source, "config")
            value = _MISSING
            for name in ("MODEL_CONFIG", "CONFIG", "model_config", "config"):
                candidate = getattr(module, name, _MISSING)
                if isinstance(candidate, Mapping):
                    value = candidate
                    break
            if value is _MISSING:
                getter = getattr(module, "get_config", None)
                value = getter() if callable(getter) else _MISSING
        else:
            raise ViTPoseRunnerError(
                f"Unsupported ViTPose config format '{source.suffix}' for '{source}'. "
                "Use .json, .yaml/.yml, or .py."
            )
    except ViTPoseRunnerError:
        raise
    except Exception as exc:
        raise ViTPoseRunnerError(
            f"Unable to read ViTPose model config '{source}': {exc}"
        ) from exc
    if not isinstance(value, Mapping):
        raise ViTPoseRunnerError(
            f"ViTPose model config '{source}' must contain a mapping at its root "
            "or expose CONFIG/MODEL_CONFIG."
        )
    return dict(value)


def _is_bundled_placeholder_definition(reference: Any, device: Optional[str] = None) -> bool:
    """Detect TexMotion's contract template before loading a large checkpoint."""
    source = _as_path(reference)
    if source is None or not source.is_file():
        return False
    if source.name.lower() != "wham_vitpose_model_definition.py":
        return False
    try:
        text = source.read_text(encoding="utf-8").lower()
    except OSError:
        return False
    if "pure pytorch vitpose implementation" in text:
        if device is not None and str(device).startswith("cuda"):
            return False
        return True
    return "not an architecture" in text and "contract template" in text


def _call_hook(
    hook: Callable[..., Any],
    config: Mapping[str, Any],
    payload: Any = _MISSING,
    checkpoint_path: Any = None,
) -> Any:
    """Invoke model hooks with a small, signature-aware compatibility seam."""

    try:
        signature = inspect.signature(hook)
    except (TypeError, ValueError):
        signature = None
    if signature is None:
        return hook(config)

    positional = [
        parameter
        for parameter in signature.parameters.values()
        if parameter.kind
        in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    has_varargs = any(
        parameter.kind == inspect.Parameter.VAR_POSITIONAL
        for parameter in signature.parameters.values()
    )
    keyword_only = {
        parameter.name: parameter
        for parameter in signature.parameters.values()
        if parameter.kind == inspect.Parameter.KEYWORD_ONLY
    }

    def argument_for(name: str, fallback_index: Optional[int] = None) -> Any:
        normalized = str(name).replace("-", "_").lower()
        if normalized in {
            "checkpoint_path",
            "checkpointpath",
            "checkpoint_pth",
            "checkpointpth",
            "weights_path",
            "weightspath",
            "model_path",
            "modelpath",
        }:
            return checkpoint_path
        if normalized in {
            "checkpoint_payload",
            "checkpointpayload",
            "checkpoint",
            "payload",
            "state_dict",
            "statedict",
            "weights",
        }:
            return payload if payload is not _MISSING else None
        if normalized in {"config", "model_config", "modelconfig", "cfg"}:
            return config
        if fallback_index == 0:
            return config
        if fallback_index == 1:
            return payload if payload is not _MISSING else None
        return None

    if keyword_only and not positional and not has_varargs:
        kwargs: dict[str, Any] = {}
        for name in keyword_only:
            value = argument_for(name)
            if value is not None or name in {
                "config",
                "model_config",
                "cfg",
                "checkpoint_path",
                "checkpointPath",
                "checkpoint_pth",
                "checkpointPth",
            }:
                kwargs[name] = value
        return hook(**kwargs)

    if has_varargs:
        args = [config, payload if payload is not _MISSING else None, None]
    else:
        args = [argument_for(parameter.name, index) for index, parameter in enumerate(positional)]
    return hook(*args)


def _is_model_instance(value: Any) -> bool:
    """Best-effort distinction between a model object and a factory callable."""

    if isinstance(value, ModuleType):
        return False
    return any(
        callable(getattr(value, name, None))
        for name in ("load_state_dict", "eval", "forward")
    ) and callable(value)


def _find_model_factory(definition: Any) -> Any:
    if _is_model_instance(definition):
        return definition
    if callable(definition) and not isinstance(definition, ModuleType):
        return definition
    if isinstance(definition, ModuleType):
        for name in (
            "create_model",
            "build_model",
            "get_model",
            "make_model",
            "ViTPose",
            "ViTPoseModel",
            "Model",
        ):
            candidate = getattr(definition, name, None)
            if callable(candidate):
                return candidate
        for name in ("MODEL", "model", "NETWORK", "network"):
            candidate = getattr(definition, name, None)
            if _is_model_instance(candidate):
                return candidate
    return None


def _is_mapping_state_dict(value: Any) -> bool:
    if not isinstance(value, Mapping) or not value:
        return False
    if not all(isinstance(key, str) for key in value):
        return False
    # Tensor/array values distinguish a raw state dict from metadata such as
    # {epoch: 10, config: {...}}.  Scalar tensor values are allowed as well.
    return any(hasattr(item, "shape") or hasattr(item, "detach") for item in value.values()) or all(
        np.isscalar(item) for item in value.values()
    )


def _mapping_path(mapping: Mapping[str, Any], path: str) -> Any:
    value: Any = mapping
    for part in path.split("."):
        if not isinstance(value, Mapping) or part not in value:
            return _MISSING
        value = value[part]
    return value


def _extract_checkpoint_model(payload: Any) -> Any:
    if _is_model_instance(payload):
        return payload
    if isinstance(payload, Mapping):
        for key in ("model", "network", "module"):
            candidate = payload.get(key, _MISSING)
            if _is_model_instance(candidate):
                return candidate
    return None


def _extract_state_dict(payload: Any, configured_key: Any = None) -> Optional[Mapping[str, Any]]:
    if configured_key:
        if isinstance(payload, Mapping):
            selected = _mapping_path(payload, str(configured_key))
            if _is_mapping_state_dict(selected):
                return selected
    if _is_mapping_state_dict(payload):
        return payload
    if isinstance(payload, Mapping):
        for key in (
            "state_dict",
            "model_state_dict",
            "modelStateDict",
            "weights",
            "params",
            "model",
            "network",
            "backbone",
        ):
            candidate = payload.get(key, _MISSING)
            if _is_mapping_state_dict(candidate):
                return candidate
    return None


def _strip_prefix(state: Mapping[str, Any], prefix: str) -> Mapping[str, Any]:
    if state and all(str(key).startswith(prefix) for key in state):
        return {str(key)[len(prefix):]: value for key, value in state.items()}
    return state


def _load_checkpoint(torch_module: Any, path: Path, device: str) -> Any:
    loader = getattr(torch_module, "load", None)
    if not callable(loader):
        raise ViTPoseRunnerError(
            "The configured torch_module has no callable load(path, map_location=...) "
            "hook; provide real PyTorch or a compatible test seam."
        )
    try:
        # ``weights_only`` is intentionally attempted only as a compatibility
        # improvement; older/fake torch APIs may not accept the keyword.
        try:
            return loader(str(path), map_location=device, weights_only=False)
        except TypeError:
            return loader(str(path), map_location=device)
    except Exception as exc:
        raise ViTPoseRunnerError(
            f"Unable to load ViTPose checkpoint '{path}' on device '{device}': {exc}. "
            "Verify that it is a readable torch .pth file and matches the model definition."
        ) from exc


def _resize_bilinear(image: np.ndarray, height: int, width: int) -> np.ndarray:
    """Resize an image with deterministic NumPy bilinear interpolation."""

    source_height, source_width = image.shape[:2]
    if (source_height, source_width) == (height, width):
        return image.astype(np.float32, copy=True)
    if source_height <= 0 or source_width <= 0:
        raise ValueError("frame dimensions must be positive")

    y = np.linspace(0.0, float(source_height - 1), height, dtype=np.float32)
    x = np.linspace(0.0, float(source_width - 1), width, dtype=np.float32)
    y0 = np.floor(y).astype(np.int32)
    x0 = np.floor(x).astype(np.int32)
    y1 = np.minimum(y0 + 1, source_height - 1)
    x1 = np.minimum(x0 + 1, source_width - 1)
    wy = (y - y0).reshape(-1, 1, 1)
    wx = (x - x0).reshape(1, -1, 1)

    top_left = image[y0[:, None], x0[None, :]]
    top_right = image[y0[:, None], x1[None, :]]
    bottom_left = image[y1[:, None], x0[None, :]]
    bottom_right = image[y1[:, None], x1[None, :]]
    top = top_left * (1.0 - wx) + top_right * wx
    bottom = bottom_left * (1.0 - wx) + bottom_right * wx
    return (top * (1.0 - wy) + bottom * wy).astype(np.float32, copy=False)


def _to_numpy(value: Any) -> Optional[np.ndarray]:
    """Convert torch/numpy-like values without importing torch."""

    if value is None:
        return None
    candidate = value
    detach = getattr(candidate, "detach", None)
    if callable(detach):
        candidate = detach()
    cpu = getattr(candidate, "cpu", None)
    if callable(cpu):
        candidate = cpu()
    numpy_method = getattr(candidate, "numpy", None)
    if callable(numpy_method):
        candidate = numpy_method()
    try:
        return np.asarray(candidate)
    except (TypeError, ValueError):
        return None


def _feature_array(value: Any, frame_count: int, output_key: Any = None, output_index: Any = None) -> Optional[np.ndarray]:
    """Find and flatten a per-frame model output to ``(N, D)``."""

    if value is None:
        return None

    if output_key is not None and isinstance(value, Mapping):
        selected = _mapping_path(value, str(output_key))
        if selected is not _MISSING:
            selected_key = str(output_key).split(".")[-1].strip().lower()
            if selected_key in _NON_IMAGE_OUTPUT_KEYS:
                return None
            found = _feature_array(selected, frame_count)
            if found is not None:
                return found

    if output_index is not None and isinstance(value, (tuple, list)):
        try:
            selected = value[int(output_index)]
        except (IndexError, TypeError, ValueError):
            selected = _MISSING
        if selected is not _MISSING:
            found = _feature_array(selected, frame_count)
            if found is not None:
                return found

    if isinstance(value, Mapping):
        # Only known feature fields are safe to auto-select.  Recursing into
        # arbitrary model output keys used to turn HMR2's ``pred_keypoints_3d``
        # or camera/SMPL tensors into a convincing-looking WHAM image token.
        # Unknown fields remain available through an explicit output_key, while
        # pose/SMPL/camera outputs are never accepted as image features.
        for key in _FEATURE_OUTPUT_KEYS:
            if key not in value:
                continue
            if str(key).strip().lower() in _NON_IMAGE_OUTPUT_KEYS:
                continue
            found = _feature_array(value[key], frame_count, output_key, output_index)
            if found is not None:
                return found
        return None

    if isinstance(value, (tuple, list)):
        # Uniform per-frame lists become a useful array first; heterogeneous
        # model return tuples are then searched recursively.
        array = _to_numpy(value)
        if array is not None and array.dtype != object:
            found = _feature_array(array, frame_count, output_key, output_index)
            if found is not None:
                return found
        for item in value:
            found = _feature_array(item, frame_count, output_key, output_index)
            if found is not None:
                return found
        return None

    array = _to_numpy(value)
    if array is None:
        return None
    if array.ndim == 0:
        if frame_count != 1:
            return None
        array = array.reshape(1, 1)
    elif array.ndim == 1:
        if array.shape[0] == frame_count:
            array = array.reshape(frame_count, 1)
        elif frame_count == 1:
            array = array.reshape(1, -1)
        else:
            return None
    elif array.shape[0] == frame_count:
        array = array.reshape(frame_count, -1)
    elif array.ndim >= 2 and array.shape[0] == 1 and array.shape[1] == frame_count:
        array = array[0].reshape(frame_count, -1)
    else:
        return None
    return array


class ViTPoseFeatureRunner:
    """Load a local ViTPose checkpoint and expose WHAM's feature contract.

    ``checkpoint_path`` may be supplied directly or through a config mapping.
    ``model_factory`` and ``model_definition`` are intentionally injectable so
    the runner can work with an official ViTPose/mmpose model definition,
    project-local code, or a tiny CPU fake in tests.
    """

    def __init__(
        self,
        config: Optional[Mapping[str, Any] | str | Path] = None,
        checkpoint_path: Any = None,
        model_definition: Any = None,
        config_path: Any = None,
        model_factory: Any = None,
        *,
        model_config: Any = None,
        torch_module: Any = None,
        device: Any = None,
        input_size: Any = None,
        mean: Any = None,
        std: Any = None,
        color_order: Any = None,
        bgr_to_rgb: Any = None,
        feature_method: Any = None,
        output_key: Any = None,
        output_index: Any = None,
        batch_size: Any = None,
        feature_dim: Any = None,
        strict_checkpoint: Any = None,
        runtime: Any = None,
        runner_kind: Any = None,
        crop_mode: Any = None,
        encode: Any = None,
        require_bboxes: Any = None,
        **kwargs: Any,
    ) -> None:
        raw: dict[str, Any] = {}
        if isinstance(config, Mapping):
            raw.update(config)
        elif config is not None:
            # A path ending in a checkpoint suffix is a convenient positional
            # shorthand; all other paths are interpreted as model config.
            config_path_candidate = _as_path(config)
            candidate_name = (
                config_path_candidate.name.lower()
                if config_path_candidate is not None
                else ""
            )
            if config_path_candidate is not None and candidate_name.endswith(
                (".pth", ".pth.tar", ".pt", ".bin", ".ckpt", ".tar", ".safetensors")
            ):
                if checkpoint_path is None:
                    checkpoint_path = config
            elif config_path is None:
                config_path = config
        raw.update({key: value for key, value in kwargs.items() if value is not None})

        self.config = dict(raw)
        self._config_path = config_path if config_path is not None else _first(raw, _CONFIG_PATH_KEYS)
        inline_model_config = model_config if model_config is not None else _first(raw, _MODEL_CONFIG_KEYS)
        # ``config={...}`` is a common spelling when the outer constructor
        # receives a runner mapping and the nested value is the architecture
        # mapping.  Keep it as a compatibility alias without changing the
        # outer config dictionary passed to callers.
        if inline_model_config is None and isinstance(raw.get("config"), Mapping):
            inline_model_config = raw["config"]
        self._config_defaults: dict[str, Any] = {}
        self._config_defaults_error: Optional[ViTPoseRunnerError] = None
        if self._config_path is not None:
            try:
                self._config_defaults.update(_load_config_reference(self._config_path))
            except ViTPoseRunnerError as exc:
                # Preserve lazy setup semantics: malformed model config is
                # reported by initialize() with the checkpoint context rather
                # than making construction/import fail unexpectedly.
                self._config_defaults_error = exc
        if isinstance(inline_model_config, Mapping):
            self._config_defaults.update(dict(inline_model_config))
        elif inline_model_config is not None:
            try:
                self._config_defaults.update(_load_config_reference(inline_model_config))
            except ViTPoseRunnerError as exc:
                self._config_defaults_error = exc
        resolved_config = dict(self._config_defaults)
        resolved_config.update(raw)
        self._checkpoint_path = checkpoint_path if checkpoint_path is not None else _first(resolved_config, _CHECKPOINT_KEYS)
        self._model_definition = (
            model_definition if model_definition is not None else _first(resolved_config, _MODEL_DEFINITION_KEYS)
        )
        self._model_factory = model_factory if model_factory is not None else _first(resolved_config, _MODEL_FACTORY_KEYS)
        self._torch_module = torch_module if torch_module is not None else _first(resolved_config, _TORCH_KEYS)
        self.device = str(device if device is not None else _first(resolved_config, ("device",), "cpu"))
        if self.device.lower() == "auto":
            # Device auto-selection is delayed until torch is loaded.  CPU is
            # retained when the injected/fake module has no CUDA seam.
            self.device = "auto"

        preprocess_config = _first(resolved_config, ("preprocess", "preprocess_config", "preprocessConfig"), {})
        if not isinstance(preprocess_config, Mapping):
            preprocess_config = {}
        self._input_size_explicit = bool(
            input_size is not None
            or any(
                key in resolved_config
                for key in ("input_size", "inputSize", "image_size", "imageSize")
            )
            or any(
                key in preprocess_config
                for key in ("input_size", "inputSize", "image_size", "imageSize")
            )
        )
        self._input_size_value = (
            input_size
            if input_size is not None
            else _first(resolved_config, ("input_size", "inputSize", "image_size", "imageSize"), _MISSING)
        )
        if self._input_size_value is _MISSING:
            self._input_size_value = _first(preprocess_config, ("input_size", "inputSize", "image_size", "imageSize"), None)
        self.input_size = _normalise_input_size(self._input_size_value)
        self._mean_value = mean if mean is not None else _first(resolved_config, ("mean", "image_mean", "imageMean"), _MISSING)
        if self._mean_value is _MISSING:
            self._mean_value = _first(preprocess_config, ("mean", "image_mean", "imageMean"), _DEFAULT_MEAN)
        self.mean = _normalise_channel_values(self._mean_value, "mean")
        self._std_value = std if std is not None else _first(resolved_config, ("std", "image_std", "imageStd"), _MISSING)
        if self._std_value is _MISSING:
            self._std_value = _first(preprocess_config, ("std", "image_std", "imageStd"), _DEFAULT_STD)
        self.std = _normalise_channel_values(self._std_value, "std")
        if any(value == 0.0 for value in self.std):
            raise ValueError("std must contain non-zero channel values")
        self.color_order = str(
            color_order
            if color_order is not None
            else _first(resolved_config, ("color_order", "colorOrder", "input_color", "inputColor"), _first(preprocess_config, ("color_order", "colorOrder", "input_color", "inputColor"), "rgb"))
        ).lower()
        if self.color_order not in ("rgb", "bgr"):
            raise ValueError("color_order must be 'rgb' or 'bgr'")
        if bgr_to_rgb is not None:
            self.color_order = "rgb" if bool(bgr_to_rgb) else "bgr"
        self.feature_method = (
            feature_method if feature_method is not None else _first(resolved_config, _FEATURE_METHOD_KEYS)
        )
        self.output_key = output_key if output_key is not None else _first(resolved_config, _OUTPUT_KEY_KEYS)
        self.output_index = output_index if output_index is not None else _first(resolved_config, _OUTPUT_INDEX_KEYS)
        self.batch_size = batch_size if batch_size is not None else _first(resolved_config, ("batch_size", "batchSize"))
        if self.batch_size is not None:
            try:
                self.batch_size = int(self.batch_size)
            except (TypeError, ValueError):
                raise ValueError("batch_size must be a positive integer") from None
            if self.batch_size <= 0:
                raise ValueError("batch_size must be a positive integer")
        configured_feature_dim = feature_dim if feature_dim is not None else _first(resolved_config, ("feature_dim", "featureDim"))
        self.feature_dim = int(configured_feature_dim) if configured_feature_dim is not None else None
        if self.feature_dim is not None and self.feature_dim <= 0:
            raise ValueError("feature_dim must be positive")
        configured_strict = strict_checkpoint if strict_checkpoint is not None else _first(
            resolved_config, ("strict_checkpoint", "strictCheckpoint", "strict_state_dict", "strictStateDict"), True
        )
        self.strict_checkpoint = bool(configured_strict)
        self._state_dict_key = _first(resolved_config, ("state_dict_key", "stateDictKey", "checkpoint_state_key"))
        self._factory_loads_checkpoint = _configured_bool(
            resolved_config, _FACTORY_LOADS_CHECKPOINT_KEYS, False
        )
        self._model_config_inline = inline_model_config
        self.model_config: dict[str, Any] = {}
        self._model: Any = None
        self._torch: Any = None
        self._initialized = False
        self._last_error: Optional[str] = None
        self._model_definition_label: Optional[str] = None
        self._checkpoint_sha256: Optional[str] = None
        self._config_sha256: Optional[str] = None
        self._factory_version: Optional[str] = None
        self._load_diagnostics: dict[str, Any] = {
            "strict": bool(self.strict_checkpoint),
            "factoryLoadsCheckpoint": bool(self._factory_loads_checkpoint),
            "missingKeys": [],
            "unexpectedKeys": [],
        }
        self._last_error_code: Optional[str] = None
        self._last_preflight: dict[str, Any] = {}
        self._last_extraction_succeeded = False
        configured_kind = (
            runtime
            if runtime is not None
            else runner_kind
            if runner_kind is not None
            else _first(resolved_config, _RUNNER_KIND_KEYS, "vitpose")
        )
        self.runner_kind = str(configured_kind or "vitpose").strip().lower().replace("-", "_")
        if self.runner_kind in ("", "default", "generic"):
            self.runner_kind = "vitpose"
        # WHAM's official FeatureExtractor calls HMR2 on a 256x256 square
        # crop.  Keep generic ViTPose's conventional 256x192 default, but do
        # not make an HMR2-compatible definition silently receive a different
        # aspect ratio when no input size was declared.
        if _runner_kind_is_hmr2(self.runner_kind) and not self._input_size_explicit:
            self.input_size = (256, 256)
        if _runner_kind_is_hmr2(self.runner_kind) and self.output_key is None:
            # HMR2 implementations commonly return auxiliary SMPL values
            # alongside the encoded image token.  Prefer the official
            # ``img_feat`` field when present while retaining tensor/tuple
            # compatibility for model definitions that return the feature
            # directly.
            self.output_key = "img_feat"
        configured_crop_mode = (
            crop_mode if crop_mode is not None else _first(resolved_config, _CROP_MODE_KEYS, None)
        )
        if isinstance(configured_crop_mode, str):
            normalized_crop_mode = configured_crop_mode.strip().lower().replace("-", "_")
            crop_enabled = normalized_crop_mode in {
                "bbox", "person", "person_bbox", "person_crop", "hmr2", "top_down", "true", "1", "yes"
            }
        elif configured_crop_mode is None:
            crop_enabled = _runner_kind_is_hmr2(self.runner_kind)
        else:
            crop_enabled = bool(configured_crop_mode)
        self.person_crop = bool(crop_enabled)
        configured_encode = encode if encode is not None else _first(resolved_config, _ENCODE_KEYS, _MISSING)
        self.encode = (
            _runner_kind_is_hmr2(self.runner_kind)
            if configured_encode is _MISSING
            else _configured_bool({"value": configured_encode}, ("value",), False)
        )
        configured_bbox_required = require_bboxes if require_bboxes is not None else _first(
            resolved_config, _BBOX_REQUIRED_KEYS, _MISSING
        )
        self.require_bboxes = (
            bool(self.person_crop)
            if configured_bbox_required is _MISSING
            else _configured_bool({"value": configured_bbox_required}, ("value",), False)
        )
        self._last_adopted_frame_indices: list[int] = []
        self._last_input_frame_count = 0

    @property
    def model(self) -> Any:
        """The initialized model object, or ``None`` before initialization."""

        return self._model

    @property
    def checkpoint_path(self) -> Optional[str]:
        return str(self._checkpoint_path) if self._checkpoint_path is not None else None

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    @property
    def last_error(self) -> Optional[str]:
        return self._last_error

    def _remember_error(self, error: ViTPoseRunnerError) -> ViTPoseRunnerError:
        """Keep the stable failure code visible after a rejected extraction."""

        self._last_error = str(error)
        self._last_error_code = error.error_code
        self._last_extraction_succeeded = False
        return error

    def _resolve_device(self, torch_module: Any) -> str:
        if self.device != "auto":
            return self.device
        cuda = getattr(torch_module, "cuda", None)
        available = getattr(cuda, "is_available", None) if cuda is not None else None
        try:
            return "cuda" if callable(available) and bool(available()) else "cpu"
        except Exception:
            return "cpu"

    def _build_model(self, payload: Any) -> Any:
        definition = self._model_definition
        if definition is not None:
            self._model_definition_label = str(definition)
            try:
                definition = _load_module_reference(definition, "model definition")
            except ViTPoseRunnerError:
                raise
            except Exception as exc:
                raise ViTPoseRunnerError(
                    f"Unable to load ViTPose model definition '{self._model_definition}': {exc}. "
                    "Configure model_definition/modelDefinition with compatible model code."
                ) from exc
            # A model-definition module may describe its expected runtime
            # without forcing callers to duplicate that declaration in the
            # runner config.  This is useful for HMR2 modules whose forward
            # hook requires ``encode=True`` and person crops.
            declared_kind = None
            if isinstance(definition, ModuleType):
                declared_kind = next(
                    (
                        getattr(definition, name, None)
                        for name in ("RUNNER_KIND", "runner_kind", "RUNTIME", "runtime", "MODEL_RUNTIME")
                        if getattr(definition, name, None) is not None
                    ),
                    None,
                )
            if declared_kind is not None and self.runner_kind == "vitpose":
                self.runner_kind = str(declared_kind).strip().lower().replace("-", "_")
                if _runner_kind_is_hmr2(self.runner_kind):
                    self.person_crop = True
                    self.encode = True
                    self.require_bboxes = True
                    if not self._input_size_explicit:
                        self.input_size = (256, 256)
                    if self.output_key is None:
                        self.output_key = "img_feat"

        factory = self._model_factory
        if factory is not None:
            if isinstance(factory, (str, Path)):
                factory = _load_module_reference(factory, "model factory")
            if not callable(factory):
                raise ViTPoseRunnerError(
                    "ViTPose model_factory/modelFactory must be callable or a module:hook reference.",
                    error_code="model_factory_missing",
                )
            self._model_definition_label = self._model_definition_label or getattr(factory, "__name__", str(factory))
        elif definition is not None:
            factory = _find_model_factory(definition)
            if factory is None:
                raise ViTPoseRunnerError(
                    f"ViTPose model definition '{self._model_definition}' exposes no supported factory. "
                    "Add create_model(config), build_model(config), Model(config), or ViTPose(config).",
                    error_code="model_factory_missing",
                )
            self._model_definition_label = self._model_definition_label or str(self._model_definition)
        else:
            checkpoint_model = _extract_checkpoint_model(payload)
            if checkpoint_model is not None:
                return checkpoint_model
            raise ViTPoseRunnerError(
                "ViTPose model-definition hook is required for a state-dict checkpoint. "
                "Configure model_definition/modelDefinition (module or .py path) or "
                "model_factory/modelFactory, then retry checkpoint loading.",
                error_code="model_factory_missing",
            )

        try:
            model = (
                factory
                if _is_model_instance(factory)
                else _call_hook(
                    factory,
                    self.model_config,
                    payload,
                    checkpoint_path=self.checkpoint_path,
                )
            )
        except Exception as exc:
            raise ViTPoseRunnerError(
                f"ViTPose model-definition hook '{self._model_definition_label}' could not build a model: {exc}. "
                "Check the architecture config and model code imports.",
                error_code="checkpoint_architecture_mismatch",
            ) from exc
        if isinstance(model, (tuple, list)) and model:
            # ``hmr2.models.load_hmr2`` returns ``(model, model_cfg)`` while
            # WHAM's local ``hmr2`` helper returns only the model.  Accept the
            # former without making callers write a wrapper solely to discard
            # the architecture config.
            candidate_model = model[0]
            candidate_config = model[1] if len(model) > 1 else None
            if isinstance(candidate_config, Mapping):
                self.model_config.update(dict(candidate_config))
            model = candidate_model
        if model is None or not callable(model):
            raise ViTPoseRunnerError(
                f"ViTPose model-definition hook '{self._model_definition_label}' returned "
                f"{type(model).__name__}, not a callable model. Return an nn.Module-like object.",
                error_code="model_factory_missing",
            )
        if self.runner_kind == "vitpose":
            declared_model_kind = next(
                (
                    getattr(model, name, None)
                for name in ("RUNNER_KIND", "runner_kind", "RUNTIME", "runtime", "MODEL_RUNTIME")
                    if getattr(model, name, None) is not None
                ),
                None,
            )
            if declared_model_kind is not None:
                self.runner_kind = str(declared_model_kind).strip().lower().replace("-", "_")
                if _runner_kind_is_hmr2(self.runner_kind):
                    self.person_crop = True
                    self.encode = True
                    self.require_bboxes = True
                    if not self._input_size_explicit:
                        self.input_size = (256, 256)
                    if self.output_key is None:
                        self.output_key = "img_feat"
        return model

    def _prepare_model_config(self) -> None:
        if self._config_defaults_error is not None:
            raise self._config_defaults_error
        loaded: dict[str, Any] = dict(self._config_defaults)
        if self._config_path is not None:
            # Reload on initialize so a config edited between construction
            # and use is reflected, while retaining the constructor-time
            # defaults needed for preprocessing settings.
            loaded.update(_load_config_reference(self._config_path))
        if isinstance(self._model_config_inline, Mapping):
            loaded.update(dict(self._model_config_inline))
        elif self._model_config_inline is not None:
            loaded.update(_load_config_reference(self._model_config_inline))
        # Keep top-level non-runtime options available to a factory.  This is
        # important for the common ``ViTPoseFeatureRunner(config_dict)`` API.
        loaded.update(self.config)
        checkpoint = _as_path(self._checkpoint_path)
        if checkpoint is not None:
            # Factories that mirror WHAM's official ``hmr2(checkpoint_pth)``
            # hook need the path in their resolved config even when the
            # runner received it as a constructor argument.
            checkpoint_value = str(checkpoint.expanduser())
            loaded.setdefault("checkpoint_path", checkpoint_value)
            loaded.setdefault("checkpointPath", checkpoint_value)
        self.model_config = loaded

    def _load_weights(self, model: Any, payload: Any, checkpoint: Path) -> None:
        if self._factory_loads_checkpoint:
            # A factory explicitly opted into owning checkpoint loading.  This
            # is the official WHAM/HMR2 shape: ``hmr2(checkpoint_pth)`` builds
            # and loads the model itself, so requiring a second strict state
            # dict load would reject otherwise compatible models.
            self._load_diagnostics = {
                "strict": bool(self.strict_checkpoint),
                "factoryLoadsCheckpoint": True,
                "missingKeys": [],
                "unexpectedKeys": [],
            }
            return
        if _is_model_instance(payload) and payload is model:
            return
        state = _extract_state_dict(payload, self._state_dict_key)
        if state is None:
            # A factory is allowed to build a model from the full checkpoint
            # payload itself.  It must opt into that behavior explicitly so a
            # random, uninitialized model cannot be reported as successful.
            if bool(self.model_config.get("factory_loads_checkpoint", self.model_config.get("factoryLoadsCheckpoint", False))):
                return
            raise ViTPoseRunnerError(
                f"ViTPose checkpoint '{checkpoint}' contains no recognized state_dict. "
                "Expected state_dict/model_state_dict/weights or configure state_dict_key. "
                "Verify that this is a model checkpoint, not a feature archive.",
                error_code="checkpoint_not_executable",
            )
        loader = getattr(model, "load_state_dict", None)
        if not callable(loader):
            raise ViTPoseRunnerError(
                f"ViTPose model from '{self._model_definition_label}' has no load_state_dict hook "
                f"for checkpoint '{checkpoint}'. Return a torch.nn.Module-like model.",
                error_code="checkpoint_architecture_mismatch",
            )
        load_error: Optional[Exception] = None
        result: Any = None
        candidate_states = [state]
        for prefix in ("module.", "model."):
            stripped = _strip_prefix(state, prefix)
            if stripped is not state:
                candidate_states.append(stripped)
        for candidate in candidate_states:
            try:
                result = loader(candidate, strict=self.strict_checkpoint)
                state = candidate
                load_error = None
                break
            except Exception as exc:
                load_error = exc
        if load_error is not None:
            raise ViTPoseRunnerError(
                f"ViTPose checkpoint '{checkpoint}' is incompatible with model definition "
                f"'{self._model_definition_label}': {load_error}. "
                "Check model_definition/modelDefinition, model config dimensions, and checkpoint variant.",
                error_code="checkpoint_architecture_mismatch",
            ) from load_error

        missing: Any = []
        unexpected: Any = []
        if isinstance(result, Mapping):
            missing = result.get("missing_keys", result.get("missingKeys", [])) or []
            unexpected = result.get("unexpected_keys", result.get("unexpectedKeys", [])) or []
        else:
            missing = getattr(result, "missing_keys", getattr(result, "missingKeys", [])) or []
            unexpected = getattr(result, "unexpected_keys", getattr(result, "unexpectedKeys", [])) or []
        self._load_diagnostics = {
            "strict": bool(self.strict_checkpoint),
            "factoryLoadsCheckpoint": False,
            "missingKeys": [str(key) for key in missing],
            "unexpectedKeys": [str(key) for key in unexpected],
        }
        if self.strict_checkpoint and (missing or unexpected):
            raise ViTPoseRunnerError(
                f"ViTPose checkpoint '{checkpoint}' is incompatible with model definition "
                f"'{self._model_definition_label}': missing_keys={list(missing)}, "
                f"unexpected_keys={list(unexpected)}. Check model config and checkpoint variant.",
                error_code="checkpoint_architecture_mismatch",
            )

    def initialize(self) -> bool:
        """Load torch, model-definition/config, and checkpoint exactly once.

        The public runner seam deliberately returns a boolean.  Callers that
        need the model use the ``model`` property after a successful result;
        this prevents a truthy partially initialized object from being treated
        as a successful preflight.
        """

        if self._initialized:
            return True
        self._last_error = None
        self._last_error_code = None
        self._last_extraction_succeeded = False
        checkpoint = _as_path(self._checkpoint_path)
        if checkpoint is None:
            error = (
                "ViTPose checkpoint_path/imageFeatureBackbonePath is required. "
                "Provide a local .pth checkpoint; a precomputed feature archive is not a model checkpoint."
            )
            self._last_error = error
            self._last_error_code = "checkpoint_not_executable"
            raise ViTPoseRunnerError(error, error_code=self._last_error_code)
        checkpoint = checkpoint.resolve()
        if not checkpoint.is_file():
            error = (
                f"ViTPose checkpoint '{checkpoint}' was not found. Configure a readable local "
                "checkpoint_path/imageFeatureBackbonePath."
            )
            self._last_error = error
            self._last_error_code = "runner_runtime_missing"
            raise ViTPoseRunnerError(error, error_code=self._last_error_code)
        try:
            self._checkpoint_path = checkpoint
            self._prepare_model_config()
            self._checkpoint_sha256 = _sha256_file(checkpoint)
            self._config_sha256 = _stable_mapping_hash(self.model_config)
            # A downloaded ViTPose backbone is normally a multi-gigabyte raw
            # state-dict archive.  Loading it before checking the architecture
            # hook can exhaust Windows virtual memory and terminate the host
            # with an access violation instead of producing a Python error.
            # Refuse that ambiguous configuration up front; the native WHAM
            # adapter will retain its explicit local-descriptor fallback until
            # compatible model code/configuration is supplied.
            if self._model_definition is None and self._model_factory is None:
                raise ViTPoseRunnerError(
                    "ViTPose model-definition hook is required before loading a state-dict "
                    "checkpoint. Configure model_definition/modelDefinition (module or .py "
                    "path) or model_factory/modelFactory; the checkpoint was not loaded.",
                    error_code="model_factory_missing",
                )
            self._torch = _load_torch(self._torch_module)
            self.device = self._resolve_device(self._torch)
            if (
                _is_bundled_placeholder_definition(self._model_definition, device=self.device)
                and self._model_factory is None
            ):
                raise ViTPoseRunnerError(
                    "The bundled ViTPose model-definition template is not an architecture; "
                    "the checkpoint was not loaded. Configure a compatible HMR2/ViTPose "
                    "model factory or replace the template with declared model code.",
                    error_code="checkpoint_not_executable",
                )
            payload = (
                _MISSING
                if self._factory_loads_checkpoint
                else _load_checkpoint(self._torch, checkpoint, self.device)
            )
            model = self._build_model(payload)
            self._load_weights(model, payload, checkpoint)
            to_device = getattr(model, "to", None)
            if callable(to_device):
                moved = to_device(self.device)
                if moved is not None:
                    model = moved
            evaluate = getattr(model, "eval", None)
            if callable(evaluate):
                evaluated = evaluate()
                if evaluated is not None and callable(evaluated):
                    model = evaluated
            self._model = model
            factory_module = inspect.getmodule(model)
            self._factory_version = str(
                getattr(factory_module, "__version__", None)
                or getattr(factory_module, "VERSION", None)
                or getattr(factory_module, "__name__", None)
                or self._model_definition_label
                or "unknown"
            )
            if self.feature_dim is None:
                for name in ("feature_dim", "embed_dim", "num_features", "out_channels"):
                    candidate = getattr(model, name, None)
                    if isinstance(candidate, (int, np.integer)) and int(candidate) > 0:
                        self.feature_dim = int(candidate)
                        break
            self._checkpoint_path = checkpoint
            self._initialized = True
            return True
        except ViTPoseRunnerError as exc:
            self._last_error = str(exc)
            self._last_error_code = exc.error_code
            self._model = None
            raise
        except Exception as exc:
            error = f"ViTPose runner initialization failed for checkpoint '{checkpoint}': {exc}"
            self._last_error = error
            text = str(exc).lower()
            self._last_error_code = (
                "checkpoint_architecture_mismatch"
                if "load_state_dict" in text or "incompatible" in text or "unexpected key" in text
                else "official_runner_inference_failed"
            )
            self._model = None
            raise ViTPoseRunnerError(error, error_code=self._last_error_code) from exc

    def _preprocess_frame(self, frame: Any) -> np.ndarray:
        value = np.asarray(frame)
        if value.ndim != 3 or value.shape[2] < 3:
            raise ViTPoseRunnerError(
                "ViTPose frames must be HxWx3 BGR/RGB arrays; received "
                f"shape {value.shape}."
            )
        value = value[..., :3]
        if not np.issubdtype(value.dtype, np.floating):
            value = value.astype(np.float32)
            value /= 255.0
        else:
            value = value.astype(np.float32, copy=False)
            if not np.isfinite(value).all():
                raise ViTPoseRunnerError("ViTPose frame contains non-finite pixel values")
            # Float video readers vary between [0, 1] and [0, 255].  Preserve
            # normalized input and deterministically scale the latter.
            if value.size and float(np.nanmax(value)) > 1.0:
                value = value / 255.0
        if not np.isfinite(value).all():
            raise ViTPoseRunnerError("ViTPose frame contains non-finite pixel values")
        if self.color_order == "rgb":
            value = value[..., ::-1]
        value = _resize_bilinear(value, self.input_size[0], self.input_size[1])
        mean = np.asarray(self.mean, dtype=np.float32).reshape(1, 1, 3)
        std = np.asarray(self.std, dtype=np.float32).reshape(1, 1, 3)
        value = (value - mean) / std
        return np.transpose(value, (2, 0, 1)).astype(np.float32, copy=False)

    def preprocess(self, frames: Sequence[Any]) -> np.ndarray:
        """Return the deterministic float32 ``(N, 3, H, W)`` model input."""

        values = list(frames)
        if not values:
            return np.empty((0, 3, self.input_size[0], self.input_size[1]), dtype=np.float32)
        try:
            return np.stack([self._preprocess_frame(frame) for frame in values], axis=0)
        except ViTPoseRunnerError:
            raise
        except Exception as exc:
            raise ViTPoseRunnerError(f"ViTPose frame preprocessing failed: {exc}") from exc

    def _invoke_model(self, batch: Any) -> Any:
        model = self._model
        if model is None:
            raise ViTPoseRunnerError("ViTPose model is not initialized")

        def invoke(target: Callable[..., Any]) -> Any:
            if self.encode:
                try:
                    # HMR2's feature path is distinct from its SMPL decode
                    # path; calling the latter would return pose parameters,
                    # not WHAM image tokens.
                    return target(batch, encode=True)
                except TypeError as exc:
                    raise ViTPoseRunnerError(
                        "Configured HMR2/ViTPose model does not accept encode=True; "
                        "provide an HMR2-compatible feature definition."
                    ) from exc
            return target(batch)

        method = self.feature_method
        if method is not None:
            if callable(method):
                try:
                    if self.encode:
                        return method(model, batch, encode=True)
                    return method(model, batch)
                except TypeError as first_error:
                    if self.encode:
                        try:
                            # A bound feature hook often accepts
                            # ``(batch, encode=True)`` and does not need the
                            # model object supplied by an unbound callback.
                            return method(batch, encode=True)
                        except TypeError as bound_error:
                            raise ViTPoseRunnerError(
                                "Configured HMR2/ViTPose feature_method does not accept encode=True; "
                                "provide an HMR2-compatible feature definition."
                            ) from bound_error
                    try:
                        return method(batch)
                    except TypeError:
                        raise first_error
            target = model
            for part in str(method).split("."):
                target = getattr(target, part)
            if not callable(target):
                raise ViTPoseRunnerError(f"Configured ViTPose feature_method '{method}' is not callable")
            return invoke(target)
        for name in ("extract_features", "extract_feat", "forward_features"):
            target = getattr(model, name, None)
            if callable(target):
                return invoke(target)
        if callable(model):
            return invoke(model)
        forward = getattr(model, "forward", None)
        if callable(forward):
            return invoke(forward)
        raise ViTPoseRunnerError("ViTPose model exposes no callable forward/extract_features hook")

    def _to_tensor(self, batch: np.ndarray) -> Any:
        torch_module = self._torch
        creator = getattr(torch_module, "from_numpy", None)
        if not callable(creator):
            creator = getattr(torch_module, "as_tensor", None)
        tensor = creator(batch) if callable(creator) else batch
        to_device = getattr(tensor, "to", None)
        if callable(to_device):
            moved = to_device(self.device)
            if moved is not None:
                tensor = moved
        return tensor

    def _infer_batch(
        self,
        batch: np.ndarray,
        frame_count: int,
        frame_indices: Optional[Sequence[int]] = None,
    ) -> np.ndarray:
        tensor = self._to_tensor(batch)
        no_grad = getattr(self._torch, "no_grad", None)
        context = no_grad() if callable(no_grad) else nullcontext()
        if context is None:
            context = nullcontext()
        try:
            with context:
                output = self._invoke_model(tensor)
        except ViTPoseRunnerError:
            raise
        except Exception as exc:
            raise ViTPoseRunnerError(
                f"ViTPose model inference failed for {frame_count} frame(s): {exc}. "
                "Check input_size/color_order and the model feature_method hook."
            ) from exc
        if isinstance(output, Mapping):
            output_indices = _first(
                output,
                ("frame_indices", "frameIndices", "source_frame_indices", "sourceFrameIndices"),
                None,
            )
            if output_indices is not None and frame_indices is not None:
                try:
                    returned_indices = []
                    for value in np.asarray(output_indices).reshape(-1).tolist():
                        if isinstance(value, (bool, np.bool_)):
                            raise ValueError("boolean values are not frame indices")
                        numeric = float(value)
                        if not np.isfinite(numeric) or numeric != math.trunc(numeric):
                            raise ValueError(f"{value!r} is not an integer-like frame index")
                        returned_indices.append(int(numeric))
                except (TypeError, ValueError, OverflowError) as exc:
                    raise ViTPoseRunnerError(
                        f"ViTPose output frame indices are not integer-like: {exc}",
                        error_code="feature_frame_count_mismatch",
                    ) from exc
                expected_indices = [int(value) for value in frame_indices]
                if returned_indices != expected_indices:
                    raise ViTPoseRunnerError(
                        "ViTPose output frame indices are not aligned with input frame_indices",
                        error_code="feature_frame_count_mismatch",
                    )
        features = _feature_array(output, frame_count, self.output_key, self.output_index)
        if features is None:
            raise ViTPoseRunnerError(
                "ViTPose model output is incompatible with the feature contract: "
                f"expected a tensor/array with {frame_count} frame rows, got {type(output).__name__}. "
                "Configure output_key/outputKey for an image-encoder feature field; "
                "HMR2 3D/SMPL/camera outputs are not image features.",
                error_code="feature_contract_failed",
            )
        raw_features = np.asarray(features)
        if not np.issubdtype(raw_features.dtype, np.floating):
            raise ViTPoseRunnerError(
                "ViTPose image feature output must use a floating-point dtype "
                f"(float32/float64), got {raw_features.dtype}",
                error_code="feature_contract_failed",
            )
        try:
            features = np.asarray(raw_features, dtype=np.float32)
        except (TypeError, ValueError) as exc:
            raise ViTPoseRunnerError(f"ViTPose feature output cannot be converted to float32: {exc}") from exc
        if features.ndim != 2 or features.shape[0] != frame_count or features.shape[1] <= 0:
            raise ViTPoseRunnerError(
                f"ViTPose image feature output must have shape ({frame_count}, D), got {features.shape}",
                error_code=(
                    "feature_frame_count_mismatch"
                    if features.ndim != 2 or features.shape[0] != frame_count
                    else "feature_contract_failed"
                ),
            )
        if not np.isfinite(features).all():
            raise ViTPoseRunnerError(
                "ViTPose image feature output contains non-finite values",
                error_code="feature_non_finite",
            )
        return np.ascontiguousarray(features, dtype=np.float32)

    def extract_sequence(
        self,
        frames: Sequence[Any],
        timestamps: Optional[Sequence[float]] = None,
        frame_indices: Optional[Sequence[int]] = None,
        *,
        indices: Optional[Sequence[int]] = None,
        frame_bboxes: Optional[Sequence[Any]] = None,
        bboxes: Optional[Sequence[Any]] = None,
        person_bboxes: Optional[Sequence[Any]] = None,
    ) -> np.ndarray:
        """Extract finite frame features with shape ``(N, D)`` for WHAM."""

        values = list(frames)
        count = len(values)
        bbox_options = [
            value for value in (frame_bboxes, bboxes, person_bboxes) if value is not None
        ]
        if len(bbox_options) > 1:
            raise ViTPoseRunnerError(
                "Provide only one of frame_bboxes, bboxes, or person_bboxes"
            )
        supplied_bboxes = bbox_options[0] if bbox_options else None
        if frame_indices is not None and indices is not None:
            raise ViTPoseRunnerError("Provide only one of frame_indices or indices; both refer to the same rows")
        if frame_indices is None:
            frame_indices = indices
        if timestamps is not None:
            try:
                times = np.asarray(timestamps)
            except Exception as exc:
                raise ViTPoseRunnerError(f"ViTPose timestamps are not readable: {exc}") from exc
            if times.shape != (count,):
                raise ViTPoseRunnerError(
                    f"timestamps are not aligned with frames: expected shape ({count},), got {times.shape}"
                )
            if not np.isfinite(times.astype(np.float64, copy=False)).all():
                raise ViTPoseRunnerError("ViTPose timestamps contain non-finite values")
        if frame_indices is not None:
            parsed_indices = np.asarray(frame_indices)
            if parsed_indices.shape != (count,):
                raise ViTPoseRunnerError(f"frame_indices must have shape ({count},), got {parsed_indices.shape}")
            try:
                raw_indices = parsed_indices.tolist()
                source_indices = []
                for value in raw_indices:
                    if isinstance(value, (bool, np.bool_)):
                        raise ValueError("boolean values are not frame indices")
                    numeric = float(value)
                    if not np.isfinite(numeric) or numeric != math.trunc(numeric):
                        raise ValueError(f"{value!r} is not an integer-like frame index")
                    source_indices.append(int(numeric))
            except (TypeError, ValueError, OverflowError) as exc:
                raise ViTPoseRunnerError(f"frame_indices are not integer-like: {exc}") from exc
            if len(set(source_indices)) != len(source_indices):
                raise self._remember_error(ViTPoseRunnerError(
                    "frame_indices contain duplicate source frame IDs; each feature row must map to one frame",
                    error_code="feature_frame_count_mismatch",
                ))
        else:
            source_indices = list(range(count))

        if supplied_bboxes is not None:
            if isinstance(supplied_bboxes, Mapping):
                supplied_bboxes = [
                    supplied_bboxes.get(index, supplied_bboxes.get(str(index)))
                    for index in source_indices
                ]
            else:
                supplied_bboxes = list(supplied_bboxes)
            if len(supplied_bboxes) != count:
                raise ViTPoseRunnerError(
                    f"frame_bboxes must have length {count}, got {len(supplied_bboxes)}"
                )
        elif self.person_crop and self.require_bboxes and count:
            raise ViTPoseRunnerError(
                "HMR2/ViTPose person-crop inference requires one bounding box per frame"
            )

        self._last_input_frame_count = count
        self._last_adopted_frame_indices = []
        try:
            self.initialize()
        except ViTPoseRunnerError as exc:
            self._last_error_code = exc.error_code
            raise
        if count == 0:
            dimension = int(self.feature_dim or 0)
            if dimension <= 0:
                # The model cannot be called to infer a dimension from an
                # empty sequence.  (0, 0) is still a valid finite N-by-D
                # archive and callers can configure feature_dim when needed.
                return np.empty((0, 0), dtype=np.float32)
            return np.empty((0, dimension), dtype=np.float32)

        adopted_positions: list[int] = []
        if self.person_crop:
            prepared_rows = []
            for position, frame in enumerate(values):
                bbox = supplied_bboxes[position] if supplied_bboxes is not None else None
                cropped = _crop_frame_to_bbox(frame, bbox, self.input_size)
                if cropped is None:
                    continue
                prepared_rows.append(self._preprocess_frame(cropped))
                adopted_positions.append(position)
            self._last_adopted_frame_indices = [source_indices[position] for position in adopted_positions]
            if prepared_rows:
                prepared = np.stack(prepared_rows, axis=0)
            else:
                dimension = int(self.feature_dim or 0)
                if dimension <= 0:
                    raise ViTPoseRunnerError(
                        "HMR2/ViTPose received no valid person bounding boxes and feature_dim is unknown"
                    )
                return np.zeros((count, dimension), dtype=np.float32)
        else:
            prepared = self.preprocess(values)
            adopted_positions = list(range(count))
            self._last_adopted_frame_indices = list(source_indices)

        adopted_count = len(adopted_positions)
        size = int(self.batch_size or adopted_count)
        try:
            outputs = [
                self._infer_batch(
                    prepared[start : start + size],
                    min(size, adopted_count - start),
                    [source_indices[position] for position in adopted_positions[start : start + size]],
                )
                for start in range(0, adopted_count, size)
            ]
        except ViTPoseRunnerError as exc:
            raise self._remember_error(exc)
        try:
            inferred = np.concatenate(outputs, axis=0).astype(np.float32, copy=False)
        except Exception as exc:
            raise ViTPoseRunnerError(f"ViTPose feature batches could not be aligned: {exc}") from exc
        if inferred.shape[0] != len(adopted_positions) or inferred.ndim != 2 or not np.isfinite(inferred).all():
            raise ViTPoseRunnerError(
                "ViTPose feature contract violated: expected finite "
                f"({len(adopted_positions)}, D), got {inferred.shape}"
            )
        if self.feature_dim is None:
            self.feature_dim = int(inferred.shape[1])
        elif int(inferred.shape[1]) != int(self.feature_dim):
            raise ViTPoseRunnerError(
                f"ViTPose image feature dimension changed: expected D={self.feature_dim}, got {inferred.shape[1]}",
                error_code="feature_dim_mismatch",
            )
        if len(adopted_positions) == count:
            features = inferred
        else:
            features = np.zeros((count, int(inferred.shape[1])), dtype=np.float32)
            features[np.asarray(adopted_positions, dtype=np.int64)] = inferred
        self._last_extraction_succeeded = True
        return np.ascontiguousarray(features, dtype=np.float32)

    def preflight(
        self,
        frames: Optional[Sequence[Any]] = None,
        timestamps_sec: Optional[Sequence[float]] = None,
        frame_indices: Optional[Sequence[int]] = None,
        person_bboxes: Optional[Sequence[Any]] = None,
    ) -> dict[str, Any]:
        """Build the model and verify one finite ``(1, D)`` feature row.

        Preflight is deliberately local and deterministic.  It never marks a
        checkpoint ready merely because the file exists, and it returns a
        serializable report so callers can show an actionable fallback reason.
        """

        probe_frames = list(frames) if frames is not None else [
            np.zeros((self.input_size[0], self.input_size[1], 3), dtype=np.uint8)
        ]
        probe_times = list(timestamps_sec) if timestamps_sec is not None else [0.0] * len(probe_frames)
        probe_indices = list(frame_indices) if frame_indices is not None else list(range(len(probe_frames)))
        probe_boxes = person_bboxes
        if probe_boxes is None and self.person_crop:
            probe_boxes = []
            for frame in probe_frames:
                array = np.asarray(frame)
                height = max(1, int(array.shape[0])) if array.ndim >= 2 else 1
                width = max(1, int(array.shape[1])) if array.ndim >= 2 else 1
                probe_boxes.append(
                    {"cx": width * 0.5, "cy": height * 0.5, "scale": max(width, height) / 200.0}
                )
        report: dict[str, Any] = {
            "status": "feature_contract_failed",
            "runner": "official_hmr2" if _runner_kind_is_hmr2(self.runner_kind) else "compatible_vitpose",
            "runtimeKind": self.runner_kind,
            "featureDim": self.feature_dim,
            "frameCount": len(probe_frames),
            "acceptedFrameCount": 0,
            "errorCode": None,
            "error": None,
        }
        try:
            self.initialize()
            features = self.extract_sequence(
                probe_frames,
                probe_times,
                probe_indices,
                person_bboxes=probe_boxes,
            )
            if features.shape != (len(probe_frames), features.shape[1]) or not np.isfinite(features).all():
                raise ViTPoseRunnerError(
                    f"Preflight feature output is not finite/aligned: got {features.shape}",
                    error_code="feature_contract_failed",
                )
            report.update(
                {
                    "status": "ready",
                    "featureDim": int(features.shape[1]),
                    "acceptedFrameCount": len(self._last_adopted_frame_indices),
                    "errorCode": None,
                    "error": None,
                }
            )
        except ViTPoseRunnerError as exc:
            self._last_error_code = exc.error_code
            report.update(
                {
                    "status": exc.error_code,
                    "errorCode": exc.error_code,
                    "error": str(exc),
                }
            )
        except Exception as exc:
            self._last_error_code = "official_runner_inference_failed"
            report.update(
                {
                    "status": self._last_error_code,
                    "errorCode": self._last_error_code,
                    "error": str(exc),
                }
            )
        self._last_preflight = dict(report)
        return report

    def __call__(
        self,
        frames: Sequence[Any],
        timestamps: Optional[Sequence[float]] = None,
        frame_indices: Optional[Sequence[int]] = None,
        *,
        indices: Optional[Sequence[int]] = None,
        frame_bboxes: Optional[Sequence[Any]] = None,
        bboxes: Optional[Sequence[Any]] = None,
        person_bboxes: Optional[Sequence[Any]] = None,
    ) -> np.ndarray:
        return self.extract_sequence(
            frames,
            timestamps,
            frame_indices,
            indices=indices,
            frame_bboxes=frame_bboxes,
            bboxes=bboxes,
            person_bboxes=person_bboxes,
        )

    def get_metadata(self) -> dict[str, Any]:
        """Return serializable runner provenance and the latest diagnostic."""

        is_hmr2 = _runner_kind_is_hmr2(self.runner_kind)
        if self._last_error_code:
            stage_status = "rejected"
        elif self._last_extraction_succeeded:
            stage_status = "used"
        elif self._initialized:
            stage_status = "available"
        else:
            stage_status = "unavailable"
        metadata: dict[str, Any] = {
            "runner": "vitpose",
            "runtime": self.runner_kind,
            "runnerKind": self.runner_kind,
            "runtimeKind": "hmr2" if is_hmr2 else "vitpose",
            "imageFeatureStage": "hmr2_image_features" if is_hmr2 else "vitpose_image_features",
            "status": stage_status,
            "checkpointPath": self.checkpoint_path,
            "checkpointSha256": self._checkpoint_sha256,
            "configSha256": self._config_sha256,
            "modelFactoryVersion": self._factory_version,
            "checkpointLoad": dict(self._load_diagnostics),
            "modelDefinition": self._model_definition_label or (str(self._model_definition) if self._model_definition is not None else None),
            "device": self.device,
            "inputSize": list(self.input_size),
            "colorOrder": self.color_order,
            "encode": bool(self.encode),
            "personCrop": bool(self.person_crop),
            "adoptedFrameIndices": list(self._last_adopted_frame_indices),
            "adoptedFrameCount": len(self._last_adopted_frame_indices),
            "inputFrameCount": int(self._last_input_frame_count),
            "featureDim": self.feature_dim,
            "initialized": self._initialized,
            "preflight": dict(self._last_preflight),
        }
        if self._last_error:
            metadata["error"] = self._last_error
            metadata["errorCode"] = self._last_error_code or "feature_contract_failed"
        return metadata

    def close(self) -> None:
        """Release the model reference; a later call can initialize again."""

        self._model = None
        self._initialized = False


class OfficialHMR2FeatureRunner(ViTPoseFeatureRunner):
    """Lazy bridge to the official 4D-Humans HMR2 image encoder.

    The generic :class:`ViTPoseFeatureRunner` remains the compatible
    MMPose/Transformers seam.  This class is selected for ``runtime=hmr2``
    when no local factory is supplied and uses the official runtime's
    ``ViTDetDataset`` crop contract.  Third-party code and checkpoints remain
    external assets; an absent runtime is a deterministic preflight failure,
    never a generic ViTPose success.
    """

    def __init__(
        self,
        config: Optional[Mapping[str, Any] | str | Path] = None,
        **kwargs: Any,
    ) -> None:
        values = dict(config) if isinstance(config, Mapping) else config
        if isinstance(values, dict):
            values.setdefault("runtime", "hmr2")
            values.setdefault("runner_kind", "hmr2")
        else:
            kwargs.setdefault("runtime", "hmr2")
            kwargs.setdefault("runner_kind", "hmr2")
        super().__init__(values, **kwargs)
        self._official_runtime_path = _first(
            self.config,
            (
                "runtime_path",
                "runtimePath",
                "HMR2RuntimePath",
                "hmr2RuntimePath",
            ),
            None,
        )
        self._official_model: Any = None
        self._official_config: Any = None
        self._official_dataset_type: Any = None
        self._official_runtime_label: Optional[str] = None
        self._official_checkpoint_root: Optional[Path] = None
        self._official_body_model: Optional[Path] = None
        self._official_body_model_status = "not_required"
        self._official_config_path: Optional[Path] = None
        self._official_require_body_model = _configured_bool(
            self.config,
            (
                "require_body_model",
                "requireBodyModel",
                "require_smpl",
                "requireSMPL",
            ),
            False,
        )

    @property
    def _uses_local_factory(self) -> bool:
        return self._model_factory is not None or self._model_definition is not None

    def _official_error(self, message: str, code: str) -> ViTPoseRunnerError:
        self._last_error = str(message)
        self._last_error_code = code
        return ViTPoseRunnerError(message, error_code=code)

    def _activate_official_runtime(self) -> None:
        if self._official_runtime_path is not None:
            runtime = Path(str(self._official_runtime_path)).expanduser()
            if not runtime.is_dir():
                raise self._official_error(
                    f"Configured HMR2 runtime directory was not found: {runtime}",
                    "runner_runtime_missing",
                )
            for candidate in (runtime, runtime.parent):
                text = str(candidate.resolve())
                if text not in sys.path:
                    sys.path.insert(0, text)
            self._official_runtime_label = str(runtime.resolve())
        else:
            self._official_runtime_label = "installed-python-package"
        try:
            importlib.import_module("hmr2")
        except Exception as exc:
            raise self._official_error(
                "Official 4D-Humans/HMR2 runtime could not be imported: "
                f"{exc}. Configure HMR2RuntimePath or install the selected official runtime.",
                "runner_runtime_missing",
            ) from exc
        try:
            # Keep the official branch lazy, but make its torch/device state
            # explicit before dataset collation or model construction.  An
            # explicitly requested unavailable device is an error; silently
            # moving an HMR2 checkpoint to another backend invalidates the
            # provenance shown to the caller.
            self._torch = self._torch or _load_torch(self._torch_module)
            if self.device == "auto":
                self.device = self._resolve_device(self._torch)
            if self.device.lower().startswith("cuda"):
                cuda = getattr(self._torch, "cuda", None)
                available = getattr(cuda, "is_available", None) if cuda is not None else None
                if not callable(available) or not bool(available()):
                    raise self._official_error(
                        f"Configured HMR2 device '{self.device}' is unavailable in the selected PyTorch environment.",
                        "runner_runtime_missing",
                    )
        except ViTPoseRunnerError:
            raise
        except Exception as exc:
            raise self._official_error(
                f"Official HMR2 PyTorch runtime could not be prepared: {exc}",
                "runner_runtime_missing",
            ) from exc

    @staticmethod
    def _find_body_model(root: Optional[Path]) -> Optional[Path]:
        if root is None or not root.exists():
            return None
        names = (
            "SMPL_NEUTRAL.pkl",
            "basicModel_neutral_lbs_10_207_0_v1.0.0.pkl",
            "SMPLX_NEUTRAL.npz",
        )
        if root.is_file() and root.name in names:
            return root
        for name in names:
            found = next(iter(root.rglob(name)), None)
            if found is not None:
                return found
        return None

    def _validate_body_model(self, checkpoint_root: Optional[Path], config: Any) -> None:
        configured = _first(
            self.config,
            (
                "body_model_path",
                "bodyModelPath",
                "hmr2_body_model_path",
                "hmr2BodyModelPath",
            ),
            None,
        )
        explicit_requested = configured is not None
        if configured is None and config is not None:
            smpl_config = None
            if isinstance(config, Mapping):
                smpl_config = config.get("SMPL", config.get("smpl"))
            else:
                smpl_config = getattr(config, "SMPL", getattr(config, "smpl", None))
            if smpl_config is not None:
                if isinstance(smpl_config, Mapping):
                    configured = smpl_config.get("MODEL_PATH", smpl_config.get("model_path"))
                else:
                    configured = getattr(smpl_config, "MODEL_PATH", getattr(smpl_config, "model_path", None))
        candidate = _as_path(configured)
        if candidate is not None and candidate.is_dir():
            candidate = self._find_body_model(candidate)
        if candidate is None:
            candidate = self._find_body_model(checkpoint_root)
        if candidate is None:
            model_directory = _as_path(
                _first(self.config, ("model_directory", "modelDirectory"), None)
            )
            candidate = self._find_body_model(model_directory)
        if candidate is not None and candidate.is_file():
            self._official_body_model = candidate.resolve()
            self._official_body_model_status = "available"
            return
        if self._official_require_body_model or explicit_requested:
            self._official_body_model_status = "missing"
            raise self._official_error(
                "Official HMR2 neutral SMPL/SMPL-X body model is required but was not found. "
                "Configure bodyModelPath/HMR2BodyModelPath with the licensed neutral body model.",
                "body_model_missing",
            )
        self._official_body_model_status = "not_required"

    def _configure_body_model(self, config: Any) -> Any:
        """Point the official config at an explicitly licensed body model."""

        if config is None or self._official_body_model is None:
            return config
        smpl_config = None
        if isinstance(config, Mapping):
            smpl_config = config.get("SMPL", config.get("smpl"))
        else:
            smpl_config = getattr(config, "SMPL", getattr(config, "smpl", None))
        if smpl_config is None:
            return config
        model_root = str(
            self._official_body_model
            if self._official_body_model.is_dir()
            else self._official_body_model.parent
        )
        if isinstance(smpl_config, Mapping):
            smpl_config["MODEL_PATH"] = model_root
            return config
        defrost = getattr(config, "defrost", None)
        freeze = getattr(config, "freeze", None)
        try:
            if callable(defrost):
                defrost()
            setattr(smpl_config, "MODEL_PATH", model_root)
        finally:
            if callable(freeze):
                freeze()
        return config

    def _load_official_config(self, checkpoint: Path) -> Any:
        config_ref = _first(
            self.config,
            ("config_path", "configPath", "HMR2ConfigPath", "hmr2ConfigPath"),
            None,
        )
        if config_ref is None:
            # Most official archives include this file beside the checkpoint.
            # Let ``load_hmr2`` own config resolution only when the archive
            # genuinely does not provide one.
            candidates = [checkpoint.parent / "model_config.yaml"]
            if checkpoint.parent.parent != checkpoint.parent:
                candidates.append(checkpoint.parent.parent / "model_config.yaml")
            for parent in (checkpoint.parent, checkpoint.parent.parent):
                if parent is not None and parent.exists():
                    found = next(iter(parent.rglob("model_config.yaml")), None)
                    if found is not None:
                        candidates.append(found)
            config_ref = next((path for path in candidates if path.is_file()), None)
            if config_ref is None:
                return None
        config_path = Path(str(config_ref)).expanduser()
        if not config_path.is_file():
            raise self._official_error(
                f"Official HMR2 config was not found: {config_path}",
                "missing_model",
            )
        self._official_config_path = config_path.resolve()
        try:
            configs = importlib.import_module("hmr2.configs")
            getter = getattr(configs, "get_config", None)
            if not callable(getter):
                raise ImportError("hmr2.configs.get_config is unavailable")
            for attempt in (
                lambda: getter(str(config_path), update_cachedir=True),
                lambda: getter(config_file=str(config_path), update_cachedir=True),
                lambda: getter(config_path=str(config_path), update_cachedir=True),
            ):
                try:
                    return attempt()
                except (TypeError, ValueError, OSError):
                    continue
            raise RuntimeError("all official HMR2 config loader signatures failed")
        except ViTPoseRunnerError:
            raise
        except Exception as exc:
            raise self._official_error(
                f"Official HMR2 model config could not be loaded: {exc}",
                "missing_model",
            ) from exc

    def _initialize_official(self) -> bool:
        checkpoint = _as_path(self._checkpoint_path)
        if checkpoint is None or not checkpoint.is_file():
            error = self._official_error(
                "Official HMR2 checkpoint is required and must be a readable .ckpt/.pth file.",
                "missing_model",
            )
            raise error
        checkpoint = checkpoint.resolve()
        self._checkpoint_path = checkpoint
        self._official_checkpoint_root = checkpoint.parent
        self._checkpoint_sha256 = _sha256_file(checkpoint)
        self._prepare_model_config()
        self._config_sha256 = _stable_mapping_hash(self.model_config)
        self._activate_official_runtime()
        try:
            models = importlib.import_module("hmr2.models")
            loader = getattr(models, "load_hmr2", None)
            cfg = self._load_official_config(checkpoint)
            # ``hmr2.models.load_hmr2`` performs a global ``check_smpl_exists``
            # before honoring a config.  When TexMotion has an explicitly
            # configured licensed body model, use the official HMR2 class
            # loader instead so that the resolved config can point at it.
            self._validate_body_model(self._official_checkpoint_root, cfg)
            direct_model_type = None
            if self._official_body_model is not None:
                model_module = importlib.import_module("hmr2.models.hmr2")
                direct_model_type = getattr(model_module, "HMR2", None)
                if direct_model_type is None or not callable(
                    getattr(direct_model_type, "load_from_checkpoint", None)
                ):
                    raise self._official_error(
                        "Official HMR2 body-model override requires HMR2.load_from_checkpoint.",
                        "model_factory_missing",
                    )
                cfg = self._configure_body_model(cfg)
            if direct_model_type is not None:
                self._official_model = direct_model_type.load_from_checkpoint(
                    str(checkpoint), strict=False, **({"cfg": cfg} if cfg is not None else {})
                )
                loader = None
            elif callable(loader):
                try:
                    result = loader(str(checkpoint)) if cfg is None else loader(str(checkpoint), cfg)
                except TypeError as positional_error:
                    # Some official revisions expose a keyword-only config or
                    # checkpoint argument.  Keep the call explicit rather than
                    # falling back to the generic ViTPose factory.
                    try:
                        result = (
                            loader(checkpoint_path=str(checkpoint))
                            if cfg is None
                            else loader(checkpoint_path=str(checkpoint), cfg=cfg)
                        )
                    except TypeError:
                        raise positional_error
                if isinstance(result, (tuple, list)):
                    self._official_model = result[0]
                    if len(result) > 1:
                        cfg = result[1]
                else:
                    self._official_model = result
            else:
                model_module = importlib.import_module("hmr2.models.hmr2")
                model_type = getattr(model_module, "HMR2", None)
                if model_type is None or not callable(getattr(model_type, "load_from_checkpoint", None)):
                    raise ImportError("the official runtime exposes neither load_hmr2 nor HMR2.load_from_checkpoint")
                load_kwargs: dict[str, Any] = {}
                if cfg is not None:
                    load_kwargs["cfg"] = cfg
                self._official_model = model_type.load_from_checkpoint(str(checkpoint), **load_kwargs)
            if self._official_model is None:
                raise RuntimeError("official HMR2 model factory returned no model")
            self._official_config = cfg
            dataset_module = importlib.import_module("hmr2.datasets.vitdet_dataset")
            self._official_dataset_type = getattr(dataset_module, "ViTDetDataset", None)
            if self._official_dataset_type is None:
                raise ImportError("the official HMR2 runtime does not expose ViTDetDataset")
            to_device = getattr(self._official_model, "to", None)
            if callable(to_device):
                moved = to_device(self.device)
                if moved is not None:
                    self._official_model = moved
            evaluate = getattr(self._official_model, "eval", None)
            if callable(evaluate):
                evaluated = evaluate()
                if evaluated is not None and callable(evaluated):
                    self._official_model = evaluated
            self._model = self._official_model
            self._initialized = True
            self._factory_version = (
                "hmr2.models.load_hmr2" if callable(loader) else "hmr2.models.hmr2.HMR2.load_from_checkpoint"
            )
            return True
        except ViTPoseRunnerError:
            raise
        except Exception as exc:
            text = str(exc)
            lowered = text.lower()
            if "smpl" in lowered or "body model" in lowered or "smpl-x" in lowered:
                code = "body_model_missing"
            elif "checkpoint" in lowered or "state_dict" in lowered or "load_from_checkpoint" in lowered:
                code = "checkpoint_architecture_mismatch"
            elif "config" in lowered or "model_config" in lowered:
                code = "missing_model"
            else:
                code = "model_factory_missing"
            raise self._official_error(
                f"Official HMR2 model construction failed for '{checkpoint}': {exc}",
                code,
            ) from exc

    @staticmethod
    def _move_nested(value: Any, device: str) -> Any:
        if isinstance(value, Mapping):
            return {key: OfficialHMR2FeatureRunner._move_nested(item, device) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            converted = [OfficialHMR2FeatureRunner._move_nested(item, device) for item in value]
            return type(value)(converted)
        mover = getattr(value, "to", None)
        if callable(mover):
            moved = mover(device)
            return value if moved is None else moved
        return value

    def _official_bbox(self, frame: Any, bbox: Any) -> np.ndarray:
        array = np.asarray(frame)
        if array.ndim != 3 or array.shape[2] < 3:
            raise self._official_error(
                "Official HMR2 received a frame that is not HxWx3.",
                "feature_contract_failed",
            )
        parsed = _coerce_frame_bbox(bbox, array.shape)
        if bbox is not None and parsed is None:
            raise self._official_error(
                "Official HMR2 person bounding box is invalid for the input frame.",
                "feature_contract_failed",
            )
        if parsed is None:
            height, width = array.shape[:2]
            parsed = (width * 0.5, height * 0.5, float(max(width, height)))
        cx, cy, side = parsed
        return np.asarray(
            [[cx - side * 0.5, cy - side * 0.5, cx + side * 0.5, cy + side * 0.5]],
            dtype=np.float32,
        )

    def _official_collate(self, items: Sequence[Mapping[str, Any]]) -> Any:
        try:
            data = importlib.import_module("torch.utils.data") if self._torch is None else getattr(self._torch, "utils", None)
            if data is not None and not hasattr(data, "default_collate"):
                data = getattr(data, "data", None)
            collate = getattr(data, "default_collate", None)
            if callable(collate):
                return collate(list(items))
        except Exception:
            pass
        keys = list(items[0])
        result: dict[str, Any] = {}
        for key in keys:
            values = [item[key] for item in items]
            first = values[0]
            if hasattr(first, "shape"):
                stack = getattr(self._torch, "stack", None)
                if callable(stack):
                    result[key] = stack(values, dim=0)
                    continue
            result[key] = values
        return result

    def _official_infer_batch(self, frames: Sequence[Any], boxes: Sequence[Any], indices: Sequence[int]) -> np.ndarray:
        items = []
        for frame, bbox in zip(frames, boxes):
            try:
                dataset = self._official_dataset_type(
                    self._official_config,
                    np.asarray(frame),
                    self._official_bbox(frame, bbox),
                )
                items.append(dataset[0])
            except ViTPoseRunnerError:
                raise
            except Exception as exc:
                raise self._official_error(
                    f"Official HMR2 ViTDetDataset crop failed at frame {indices[len(items)]}: {exc}",
                    "official_runner_inference_failed",
                ) from exc
        batch = self._move_nested(self._official_collate(items), self.device)
        context_factory = getattr(self._torch, "inference_mode", None) or getattr(self._torch, "no_grad", None)
        context = context_factory() if callable(context_factory) else nullcontext()
        try:
            with context:
                try:
                    output = self._official_model(batch, encode=True)
                except TypeError as exc:
                    signature = None
                    try:
                        signature = inspect.signature(self._official_model)
                    except (TypeError, ValueError):
                        pass
                    if signature is not None and "encode" not in signature.parameters:
                        output = self._official_model(batch)
                    else:
                        raise exc
        except ViTPoseRunnerError:
            raise
        except Exception as exc:
            raise self._official_error(
                f"Official HMR2 image-feature inference failed: {exc}",
                "official_runner_inference_failed",
            ) from exc
        features = _feature_array(output, len(items), self.output_key or "img_feat", self.output_index)
        if features is None:
            raise self._official_error(
                "Official HMR2 output did not expose an image-encoder feature field "
                "(expected img_feat/image_features); 3D joints and SMPL parameters are not features.",
                "feature_contract_failed",
            )
        raw = np.asarray(features)
        if not np.issubdtype(raw.dtype, np.floating):
            raise self._official_error(
                f"Official HMR2 image features must be floating-point, got {raw.dtype}",
                "feature_contract_failed",
            )
        values = np.asarray(raw, dtype=np.float32)
        if values.ndim != 2 or values.shape[0] != len(items) or values.shape[1] <= 0:
            raise self._official_error(
                f"Official HMR2 image features must have shape ({len(items)}, D), got {values.shape}",
                "feature_frame_count_mismatch" if values.ndim != 2 or values.shape[0] != len(items) else "feature_contract_failed",
            )
        if not np.isfinite(values).all():
            raise self._official_error(
                "Official HMR2 image features contain non-finite values",
                "feature_non_finite",
            )
        return np.ascontiguousarray(values, dtype=np.float32)

    def initialize(self) -> bool:
        if self._uses_local_factory:
            return super().initialize()
        if self._initialized:
            return True
        self._last_error = None
        self._last_error_code = None
        try:
            return bool(self._initialize_official())
        except ViTPoseRunnerError as exc:
            self._last_error = str(exc)
            self._last_error_code = exc.error_code
            self._model = None
            self._initialized = False
            return False
        except Exception as exc:
            self._last_error = str(exc)
            self._last_error_code = "official_runner_inference_failed"
            self._model = None
            self._initialized = False
            return False

    def extract_sequence(
        self,
        frames: Sequence[Any],
        timestamps: Optional[Sequence[float]] = None,
        frame_indices: Optional[Sequence[int]] = None,
        *,
        indices: Optional[Sequence[int]] = None,
        frame_bboxes: Optional[Sequence[Any]] = None,
        bboxes: Optional[Sequence[Any]] = None,
        person_bboxes: Optional[Sequence[Any]] = None,
    ) -> np.ndarray:
        if self._uses_local_factory:
            return super().extract_sequence(
                frames,
                timestamps,
                frame_indices,
                indices=indices,
                frame_bboxes=frame_bboxes,
                bboxes=bboxes,
                person_bboxes=person_bboxes,
            )
        values = list(frames)
        count = len(values)
        if frame_indices is not None and indices is not None:
            raise self._official_error("Provide only one of frame_indices or indices", "feature_frame_count_mismatch")
        source_indices = list(frame_indices if frame_indices is not None else indices if indices is not None else range(count))
        if len(source_indices) != count or len(set(source_indices)) != count:
            raise self._official_error("frame_indices must be aligned and unique", "feature_frame_count_mismatch")
        try:
            parsed_indices: list[int] = []
            for value in source_indices:
                if isinstance(value, (bool, np.bool_)):
                    raise ValueError("boolean values are not frame indices")
                numeric = float(value)
                if not np.isfinite(numeric) or numeric != math.trunc(numeric):
                    raise ValueError(f"{value!r} is not an integer-like frame index")
                parsed_indices.append(int(numeric))
            source_indices = parsed_indices
        except (TypeError, ValueError, OverflowError) as exc:
            raise self._official_error(f"frame_indices are not integer-like: {exc}", "feature_frame_count_mismatch") from exc
        times = list(timestamps) if timestamps is not None else [index / 30.0 for index in range(count)]
        if len(times) != count:
            raise self._official_error("timestamps are not aligned with frames", "feature_frame_count_mismatch")
        try:
            if not np.isfinite(np.asarray(times, dtype=np.float64)).all():
                raise ValueError("timestamps contain non-finite values")
        except (TypeError, ValueError, OverflowError) as exc:
            raise self._official_error(f"timestamps are not valid: {exc}", "feature_frame_count_mismatch") from exc
        bbox_options = [value for value in (frame_bboxes, bboxes, person_bboxes) if value is not None]
        if len(bbox_options) > 1:
            raise self._official_error("Provide only one of frame_bboxes, bboxes, or person_bboxes", "feature_frame_count_mismatch")
        supplied = list(bbox_options[0]) if bbox_options else [None] * count
        if len(supplied) != count:
            raise self._official_error("person_bboxes must have one entry per frame", "feature_frame_count_mismatch")
        if count == 0:
            return np.empty((0, int(self.feature_dim or 0)), dtype=np.float32)
        if not self.initialize():
            raise self._official_error(self._last_error or "Official HMR2 runner is not initialized", self._last_error_code or "runner_runtime_missing")
        outputs = []
        batch_size = int(self.batch_size or count)
        for start in range(0, count, batch_size):
            outputs.append(
                self._official_infer_batch(
                    values[start : start + batch_size],
                    supplied[start : start + batch_size],
                    source_indices[start : start + batch_size],
                )
            )
        features = np.concatenate(outputs, axis=0).astype(np.float32, copy=False)
        if self.feature_dim is None:
            self.feature_dim = int(features.shape[1])
        elif self.feature_dim != int(features.shape[1]):
            raise self._official_error(
                f"Official HMR2 image feature dimension changed: expected D={self.feature_dim}, got {features.shape[1]}",
                "feature_dim_mismatch",
            )
        self._last_input_frame_count = count
        self._last_adopted_frame_indices = list(source_indices)
        self._last_extraction_succeeded = True
        return np.ascontiguousarray(features)

    def get_metadata(self) -> dict[str, Any]:
        metadata = super().get_metadata()
        metadata.update(
            {
                "runner": "official_hmr2",
                "runtime": "hmr2",
                "runnerKind": "hmr2",
                "runtimeKind": "hmr2",
                "imageFeatureStage": "hmr2_image_features",
                "runtimePath": self._official_runtime_label,
                "modelFactoryVersion": self._factory_version or "hmr2.models.load_hmr2",
                "modelFactory": self._factory_version or "hmr2.models.load_hmr2",
                "configPath": str(self._official_config_path) if self._official_config_path else None,
                "bodyModelPath": str(self._official_body_model) if self._official_body_model else None,
                "bodyModelStatus": self._official_body_model_status,
                "bodyModelRequired": bool(self._official_require_body_model),
            }
        )
        return metadata

    def close(self) -> None:
        self._official_model = None
        self._official_dataset_type = None
        self._official_config = None
        super().close()


# Public aliases cover the naming used by WHAM/TexMotion callers.
ViTPoseRunner = ViTPoseFeatureRunner
VitPoseFeatureRunner = ViTPoseFeatureRunner
VitPoseRunner = ViTPoseFeatureRunner
ViTPoseFeatureExtractor = ViTPoseFeatureRunner
HMR2FeatureRunner = OfficialHMR2FeatureRunner
HMR2Runner = OfficialHMR2FeatureRunner


def create_vitpose_runner(config: Optional[Mapping[str, Any] | str | Path] = None, **kwargs: Any) -> ViTPoseFeatureRunner:
    """Construct the configured runner without importing PyTorch."""

    return ViTPoseFeatureRunner(config, **kwargs)


def create_vitpose_feature_runner(
    config: Optional[Mapping[str, Any] | str | Path] = None, **kwargs: Any
) -> ViTPoseFeatureRunner:
    return create_vitpose_runner(config, **kwargs)


def create_vitpose_extractor(
    config: Optional[Mapping[str, Any] | str | Path] = None, **kwargs: Any
) -> ViTPoseFeatureRunner:
    return create_vitpose_runner(config, **kwargs)


def create_hmr2_runner(
    config: Optional[Mapping[str, Any] | str | Path] = None, **kwargs: Any
) -> OfficialHMR2FeatureRunner:
    """Construct the official HMR2 image-feature runner seam."""
    return OfficialHMR2FeatureRunner(config, **kwargs)


__all__ = [
    "ViTPoseRunnerError",
    "VitPoseRunnerError",
    "ViTPoseFeatureRunner",
    "OfficialHMR2FeatureRunner",
    "ViTPoseRunner",
    "VitPoseFeatureRunner",
    "VitPoseRunner",
    "ViTPoseFeatureExtractor",
    "HMR2FeatureRunner",
    "HMR2Runner",
    "create_vitpose_runner",
    "create_vitpose_feature_runner",
    "create_vitpose_extractor",
    "create_hmr2_runner",
]
