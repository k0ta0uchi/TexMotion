"""Optional PyTorch quality backends for temporal 3D pose recovery.

This module deliberately does *not* import torch at module import time.  The
normal TexMotion installation remains MediaPipe/ONNX-only; users who install a
research stack can expose it through a small adapter module or a TorchScript
file.  The bundled WHAM bridge additionally provides a native temporal
motion-encoder core when its checkpoint is present.  WHAM, HMR2 and HybrIK
still have different preprocessing and output schemas, so pretending they are
interchangeable here would create silent pose errors.  The adapter contract
keeps that model-specific code isolated while sharing device selection,
lifecycle, result normalization and fallback behavior.

Adapter contract (the adapter module is selected by ``TEXMOTION_*_ADAPTER``):

    def create_backend(config: dict) -> object
    backend.infer_sequence(frames_bgr, timestamps_sec, frame_indices) -> result

``infer_frame``/``predict`` may be supplied instead for a frame-only adapter.
The result can be a list of FrameObservations, a list of dictionaries, or a
dictionary containing ``keypoints2d``, ``landmarks3d`` and optional
``uncertaintyIntervals``.  A TorchScript model can be used without an adapter
when ``--backend pytorch --pytorch-model path/to/model.pt`` is supplied.
"""

from __future__ import annotations

import importlib
import importlib.util
import inspect
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .base import BackendCapabilities, PoseBackend
from ..observations import (
    FrameObservations,
    Keypoint2D,
    Keypoint3D,
    ObservationStatus,
    UncertaintyInterval,
)


# SMPL-X 22 -> MediaPipe 33 indices.  Quality adapters may return either
# topology; retaining a canonical 33-slot representation keeps the existing
# extractor and retargeter backward compatible.
SMPLX_TO_MEDIAPIPE = {
    0: 0,    # pelvis -> synthetic midpoint slot
    1: 23, 2: 24, 3: 11,
    4: 25, 5: 26, 6: 12,
    7: 27, 8: 28, 9: 13,
    10: 31, 11: 32, 12: 10,
    13: 11, 14: 12, 15: 0,
    16: 11, 17: 12, 18: 13, 19: 14, 20: 15, 21: 16,
}

# Keep the quality adapter contract tolerant of image-conditioned models that
# emit the common COCO-17 projection rather than SMPL-X joints.  The primary
# 3D path still prefers the 22-joint SMPL-X mapping above.
COCO17_TO_MEDIAPIPE = {
    0: 0, 1: 2, 2: 5, 3: 7, 4: 8,
    5: 11, 6: 12, 7: 13, 8: 14, 9: 15, 10: 16,
    11: 23, 12: 24, 13: 25, 14: 26, 15: 27, 16: 28,
}

# External WHAM bridges historically emitted a H36M/J17 ordering:
# rankle, rknee, rhip, lhip, lknee, lankle, rwrist, relbow, rshoulder,
# lshoulder, lelbow, lwrist, neck, headtop, hip, spine, head.  Keep this
# mapping explicit and opt in through ``jointTopology=wham_j17`` metadata;
# the bundled native motion encoder now advertises its actual COCO-17 output,
# while generic 17-joint adapters remain COCO by default.
WHAM_J17_TO_MEDIAPIPE = {
    0: 28,  # right ankle
    1: 26,  # right knee
    2: 24,  # right hip
    3: 23,  # left hip
    4: 25,  # left knee
    5: 27,  # left ankle
    6: 16,  # right wrist
    7: 14,  # right elbow
    8: 12,  # right shoulder
    9: 11,  # left shoulder
    10: 13, # left elbow
    11: 15, # left wrist
    16: 0,  # head -> nose anchor
}

_WHAM_J17_TOPOLOGIES = {
    "wham_j17",
    "wham-j17",
    "wham17",
    "h36m_j17",
    "j17_h36m",
}

_ADAPTER_ENV = "TEXMOTION_PYTORCH_ADAPTER"
_MODEL_ENV = "TEXMOTION_PYTORCH_MODEL"
_ADAPTER_ENV_BY_KIND = {
    "wham": "TEXMOTION_WHAM_ADAPTER",
    "hmr2": "TEXMOTION_HMR2_ADAPTER",
    "hybrik": "TEXMOTION_HYBRIK_ADAPTER",
}
_DEFAULT_ADAPTER_MODULES = {
    # The package ships a native WHAM motion-encoder bridge.  Keep the fully
    # qualified module name so a downloaded checkpoint works without requiring
    # an external WHAM checkout; TEXMOTION_WHAM_ADAPTER remains an opt-in
    # override for projects that need the complete official stack.
    "wham": "pose_pipeline.adapters.texmotion_wham_adapter",
    # Official 4D-Humans/HMR2 bridge shipped with TexMotion.  Runtime source,
    # checkpoint and licensed SMPL assets are still resolved by the bridge.
    "hmr2": "pose_pipeline.adapters.texmotion_hmr2_adapter",
    "hybrik": "texmotion_hybrik_adapter",
}


# WHAM is a multi-stage research pipeline rather than a single checkpoint.
# Keep the contract in this backend so both the native motion-encoder bridge
# and a user-provided full implementation receive the same deterministic
# paths.  The native bridge only needs ``checkpoint`` and ``adapter``; the
# other stages are intentionally optional there and are reported instead of
# being silently inferred or downloaded.
WHAM_ASSET_CONTRACT_VERSION = 1
WHAM_ASSET_ROLES = (
    "checkpoint",
    "body_model",
    "image_feature_backbone",
    "camera",
    "dpvo",
    "adapter",
)
WHAM_NATIVE_REQUIRED_ROLES = ("checkpoint", "adapter")
WHAM_FULL_REQUIRED_ROLES = WHAM_ASSET_ROLES
# Missing optional WHAM stages must never make extraction appear to have been
# skipped.  The native bridge records the exact fallback reason; unverified
# local image descriptors are rejected and never fed to WHAM's integrator.
WHAM_INFERENCE_POLICY = "continue_with_explicit_fallback"
WHAM_MISSING_ASSET_BEHAVIOR = (
    "Extraction continues when an optional WHAM asset or runner is missing; "
    "TexMotion never silently skips inference and records the exact fallback reason."
)

_WHAM_ASSET_ROLE_ALIASES = {
    "checkpoint": {
        "checkpoint", "model", "model_path", "modelpath", "wham", "wham_checkpoint",
        "whamcheckpoint", "weights", "weight", "network", "network_checkpoint",
    },
    "body_model": {
        "body", "body_model", "bodymodel", "smpl", "smpl_model", "smplmodel",
        "smplx", "smpl_x", "smplx_model", "smplxmodel", "smpl_body_model",
        "smplx_body_model", "body_asset",
    },
    "image_feature_backbone": {
        "image", "image_feature", "image_features", "image_feature_backbone",
        "imagefeature", "imagefeaturebackbone", "backbone", "feature_backbone",
        "vitpose", "vitpose_checkpoint", "image_encoder", "image_encoder_checkpoint",
    },
    "camera": {
        "camera", "camera_model", "cameramodel", "camera_config", "cameraconfig",
        "camera_calibration", "calibration", "slam", "slam_config",
    },
    "dpvo": {
        "dpvo", "dpvo_model", "dpvomodel", "dpvo_checkpoint", "dpvoweights",
        "camera_motion", "camera_motion_checkpoint", "trajectory",
    },
    "adapter": {
        "adapter", "adapter_path", "adapterpath", "adapter_module", "adaptermodule",
        "wham_adapter", "whamadapter", "bridge", "implementation",
    },
}

_WHAM_DEFAULT_ASSET_NAMES = {
    # Preserve the catalogued checkpoint name first; the aliases make manual
    # provisioning predictable without searching the entire filesystem.
    "checkpoint": (
        "wham_vit_w_3dpw.pth.tar",
        "wham.pt",
        "wham.pth.tar",
    ),
    "body_model": (
        "SMPLX_NEUTRAL.npz",
        "smplx/SMPLX_NEUTRAL.npz",
        "smplx_neutral.npz",
        "SMPL_NEUTRAL.pkl",
        "smpl/SMPL_NEUTRAL.pkl",
        "smpl_neutral.pkl",
    ),
    "image_feature_backbone": (
        "vitpose-huge.pth",
        "vitpose_huge.pth",
        "vitpose-huge.pth.tar",
        "vitpose.pth",
        "image_feature_backbone.pth",
    ),
    "camera": (
        "camera.yaml",
        "camera.yml",
        "camera_config.yaml",
        "camera_config.json",
        "camera_model.json",
    ),
    "dpvo": (
        "dpvo.pth",
        "dpvo.pth.tar",
        "dpvo_model.pth",
        "dpvo.ckpt",
    ),
    "adapter": (
        "texmotion_wham_adapter.py",
        "wham_adapter.py",
    ),
}

_WHAM_ASSET_ENV = {
    "checkpoint": (
        "TEXMOTION_WHAM_CHECKPOINT",
        "TEXMOTION_PYTORCH_MODEL",
    ),
    "body_model": (
        "TEXMOTION_WHAM_BODY_MODEL",
        "TEXMOTION_WHAM_SMPL_MODEL",
        "TEXMOTION_WHAM_SMPLX_MODEL",
        "TEXMOTION_SMPL_MODEL_PATH",
        "TEXMOTION_SMPLX_MODEL_PATH",
    ),
    "image_feature_backbone": (
        "TEXMOTION_WHAM_IMAGE_FEATURE_BACKBONE",
        "TEXMOTION_WHAM_IMAGE_FEATURE_PATH",
        "TEXMOTION_WHAM_VITPOSE_MODEL",
        "TEXMOTION_VITPOSE_MODEL_PATH",
    ),
    "camera": (
        "TEXMOTION_WHAM_CAMERA_MODEL",
        "TEXMOTION_WHAM_CAMERA_PATH",
        "TEXMOTION_WHAM_CAMERA_CONFIG",
    ),
    "dpvo": (
        "TEXMOTION_WHAM_DPVO_MODEL",
        "TEXMOTION_WHAM_DPVO_PATH",
        "TEXMOTION_DPVO_MODEL_PATH",
    ),
    "adapter": (
        "TEXMOTION_WHAM_ADAPTER",
        "TEXMOTION_PYTORCH_ADAPTER",
    ),
}

_WHAM_MANIFEST_ENV = "TEXMOTION_WHAM_ASSET_MANIFEST"
_WHAM_MANIFEST_NAMES = (
    "wham_asset_manifest.json",
    "wham-assets.json",
    "wham_assets.json",
)

_WHAM_STAGE_LABELS = {
    "checkpoint": "WHAM checkpoint",
    "body_model": "SMPL/SMPL-X body model",
    "image_feature_backbone": "image-feature backbone",
    "camera": "camera calibration/configuration",
    "dpvo": "DPVO camera-motion assets",
    "adapter": "WHAM adapter",
}


def _canonical_wham_asset_role(role: Any) -> Optional[str]:
    """Normalize manifest/config role names while keeping the public keys stable."""
    if role is None:
        return None
    normalized = str(role).strip().lower().replace("-", "_").replace(" ", "_")
    for canonical, aliases in _WHAM_ASSET_ROLE_ALIASES.items():
        if normalized == canonical or normalized in aliases:
            return canonical
    return None


def _path_value(value: Any) -> Optional[str]:
    """Extract a path-like value from common manifest entry shapes."""
    if value is None:
        return None
    if isinstance(value, Mapping):
        for key in (
            "path", "absolutePath", "absolute_path", "file", "fileName", "filename",
            "relativePath", "relative_path", "location",
        ):
            if key in value and value[key] is not None:
                return _path_value(value[key])
        return None
    if isinstance(value, (list, tuple)):
        for item in value:
            candidate = _path_value(item)
            if candidate:
                return candidate
        return None
    text = str(value).strip()
    return text or None


def _manifest_asset_values(payload: Any) -> Dict[str, Any]:
    """Read flat or list-based WHAM asset manifests without requiring a schema package."""
    if not isinstance(payload, Mapping):
        return {}
    container = payload.get(
        "assets",
        payload.get("whamAssets", payload.get("stages", payload)),
    )
    result: Dict[str, Any] = {}
    if isinstance(container, Mapping):
        for key, value in container.items():
            normalized_key = str(key).strip().lower().replace("-", "_").replace(" ", "_")
            role = (
                "image_feature_archive"
                if normalized_key in {
                    "image_feature_path", "imagefeaturepath", "image_features_path",
                    "imagefeaturespath", "image_feature_archive", "imagefeaturearchive",
                    "image_feature_archive_path", "imagefeaturearchivepath", "feature_archive",
                    "featurearchive", "image_features", "imagefeatures", "image_feature",
                    "imagefeature",
                }
                else _canonical_wham_asset_role(key)
            )
            path: Any = _path_value(value)
            if isinstance(value, Mapping):
                aliases = value.get("aliases", value.get("alternatePaths", value.get("alternate_paths", [])))
                choices = [path] if path else []
                if isinstance(aliases, (list, tuple)):
                    choices.extend(str(item).strip() for item in aliases if str(item).strip())
                elif aliases:
                    choices.append(str(aliases).strip())
                path = choices or None
            if role and path:
                result[role] = path
        return result
    if isinstance(container, (list, tuple)):
        for value in container:
            if not isinstance(value, Mapping):
                continue
            role_value = value.get("role", value.get("id", value.get("name")))
            normalized_role = str(role_value).strip().lower().replace("-", "_").replace(" ", "_")
            role = (
                "image_feature_archive"
                if normalized_role in {
                    "image_feature_path", "imagefeaturepath", "image_features_path",
                    "imagefeaturespath", "image_feature_archive", "imagefeaturearchive",
                    "image_feature_archive_path", "imagefeaturearchivepath", "feature_archive",
                    "featurearchive",
                }
                else _canonical_wham_asset_role(role_value)
            )
            path: Any = _path_value(value)
            aliases = value.get("aliases", value.get("alternatePaths", value.get("alternate_paths", [])))
            choices = [path] if path else []
            if isinstance(aliases, (list, tuple)):
                choices.extend(str(item).strip() for item in aliases if str(item).strip())
            elif aliases:
                choices.append(str(aliases).strip())
            path = choices or None
            if role and path:
                result[role] = path
    return result


def _absolute_candidate(value: Optional[str], base_directory: Optional[Path] = None) -> Optional[str]:
    """Resolve one candidate without globbing or probing network locations."""
    if not value:
        return None
    try:
        expanded = os.path.expandvars(os.path.expanduser(str(value).strip()))
        if not expanded:
            return None
        path = Path(expanded)
        if not path.is_absolute() and base_directory is not None:
            path = base_directory / path
        return str(path.resolve())
    except (OSError, RuntimeError, TypeError, ValueError):
        return None


def _existing_asset_path(value: Any, base_directory: Optional[Path] = None) -> Optional[str]:
    if isinstance(value, (list, tuple)):
        for item in value:
            candidate = _existing_asset_path(item, base_directory)
            if candidate:
                return candidate
        return None
    if isinstance(value, Mapping):
        return _existing_asset_path(_path_value(value), base_directory)
    path = _absolute_candidate(value, base_directory)
    if not path:
        return None
    try:
        return path if Path(path).is_file() and Path(path).stat().st_size > 0 else None
    except OSError:
        return None


def _manifest_candidates(
    model_directory: Optional[str],
    asset_manifest_path: Optional[str],
) -> Tuple[List[Tuple[str, Path]], Optional[str]]:
    """Return manifest candidates in deterministic precedence order."""
    candidates: List[Tuple[str, Path]] = []
    seen = set()

    def add(source: str, value: Optional[str], base: Optional[Path] = None) -> None:
        path = _absolute_candidate(value, base)
        if not path or path in seen:
            return
        seen.add(path)
        candidates.append((source, Path(path)))

    # An explicit manifest is authoritative, so it is the only candidate read
    # when provided.  This avoids accidentally selecting a stale package
    # manifest after a user has chosen a project-local one.
    if asset_manifest_path:
        add("explicit", asset_manifest_path)
        return candidates, None
    env_manifest = os.environ.get(_WHAM_MANIFEST_ENV)
    if env_manifest:
        add("environment", env_manifest)
        return candidates, None
    base = _absolute_candidate(model_directory)
    model_root = Path(base) if base else None
    if model_root:
        for name in _WHAM_MANIFEST_NAMES:
            add("model_directory", name, model_root)
    package_models = Path(__file__).resolve().parents[2] / "models"
    for name in _WHAM_MANIFEST_NAMES:
        add("package", name, package_models)
    return candidates, None


def _default_wham_adapter_path() -> Optional[str]:
    path = Path(__file__).resolve().parents[1] / "adapters" / "texmotion_wham_adapter.py"
    return _existing_asset_path(str(path))


@dataclass(frozen=True)
class WhamAssetResolution:
    """Resolved local paths and diagnostics for the offline WHAM asset contract."""

    contract_version: int = WHAM_ASSET_CONTRACT_VERSION
    manifest_path: Optional[str] = None
    paths: Dict[str, Optional[str]] = field(default_factory=dict)
    stage_diagnostics: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    missing_stages: List[str] = field(default_factory=list)
    native_available: bool = False
    full_parity_available: bool = False
    manifest_error: Optional[str] = None
    # Optional precomputed (N,D) frame-feature archive. It is deliberately
    # outside ``paths`` because a ViTPose backbone and an archive are separate
    # stages and must never be conflated by the resolver.
    image_feature_archive_path: Optional[str] = None

    @property
    def missing_required_stages(self) -> List[str]:
        """Alias used by callers that call the full contract requirements required stages."""
        return list(self.missing_stages)

    @property
    def missing(self) -> List[str]:
        """Short alias for integrations that treat the result as a preflight report."""
        return list(self.missing_stages)

    @property
    def complete(self) -> bool:
        return bool(self.full_parity_available)

    @property
    def resolved_paths(self) -> Dict[str, Optional[str]]:
        return dict(self.paths)

    @property
    def diagnostic_text(self) -> str:
        if self.full_parity_available:
            return (
                "WHAM offline asset contract is complete; declared full-parity files are ready. "
                "Runtime still reports any runner incompatibility explicitly and continues with "
                "a labeled local fallback instead of silently skipping inference."
            )
        details = []
        for role in self.missing_stages:
            stage = self.stage_diagnostics.get(role, {})
            label = _WHAM_STAGE_LABELS.get(role, role)
            details.append(f"{role}: {label} ({stage.get('expected', 'local asset')})")
        missing = ", ".join(details) if details else "unknown stage"
        prefix = "WHAM offline asset contract is incomplete; missing stages: " + missing + "."
        if self.manifest_error:
            prefix += " Manifest warning: " + self.manifest_error + "."
        return (
            prefix +
            " Full WHAM parity is unavailable until these local assets are provisioned; "
            "the Settings catalog can download public ViTPose/DPVO files, while licensed body data "
            "and calibrated camera/pose exports remain user-provided. Extraction still continues "
            "with the configured safe fallback and records the exact missing-stage reason; it is "
            "never silently skipped."
        )

    def __getitem__(self, role: str) -> Optional[str]:
        canonical = _canonical_wham_asset_role(role) or role
        return self.paths[canonical]

    def get(self, role: str, default: Optional[str] = None) -> Optional[str]:
        canonical = _canonical_wham_asset_role(role) or role
        return self.paths.get(canonical, default)

    def __contains__(self, role: object) -> bool:
        canonical = _canonical_wham_asset_role(role) or role
        return canonical in self.paths

    def items(self):
        return self.paths.items()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "contractVersion": int(self.contract_version),
            "manifestPath": self.manifest_path,
            "paths": dict(self.paths),
            "stages": {key: dict(value) for key, value in self.stage_diagnostics.items()},
            "missingStages": list(self.missing_stages),
            "nativeAvailable": bool(self.native_available),
            "fullParityAvailable": bool(self.full_parity_available),
            "diagnostic": self.diagnostic_text,
            "imageFeaturePath": self.image_feature_archive_path,
            "imageFeatureArchive": self.image_feature_archive_path,
        }

    def to_metadata(self) -> Dict[str, Any]:
        """Return stable JSON-safe metadata for Python output and Unity diagnostics."""
        paths = dict(self.paths)
        return {
            "assetContractVersion": int(self.contract_version),
            "assetManifestPath": self.manifest_path,
            "whamAssets": paths,
            "whamAssetPaths": paths,
            "checkpointPath": paths.get("checkpoint"),
            "bodyModelPath": paths.get("body_model"),
            "smplModelPath": paths.get("body_model"),
            "smplxModelPath": paths.get("body_model"),
            "imageFeatureBackbonePath": paths.get("image_feature_backbone"),
            "imageFeaturePath": self.image_feature_archive_path,
            "imageFeatureArchive": self.image_feature_archive_path,
            "cameraModelPath": paths.get("camera"),
            "dpvoModelPath": paths.get("dpvo"),
            "adapterPath": paths.get("adapter"),
            "whamStageDiagnostics": {
                key: dict(value) for key, value in self.stage_diagnostics.items()
            },
            "whamMissingStages": list(self.missing_stages),
            "whamNativeAvailable": bool(self.native_available),
            "whamFullParity": bool(self.full_parity_available),
            "whamAssetDiagnostic": self.diagnostic_text,
            "whamInferencePolicy": WHAM_INFERENCE_POLICY,
            "whamMissingAssetBehavior": WHAM_MISSING_ASSET_BEHAVIOR,
        }

    def to_adapter_dict(self) -> Dict[str, Any]:
        """Expose both canonical and legacy-friendly camelCase path keys."""
        body = self.paths.get("body_model")
        image = self.paths.get("image_feature_backbone")
        image_archive = self.image_feature_archive_path
        camera = self.paths.get("camera")
        dpvo = self.paths.get("dpvo")
        return {
            "assetManifestPath": self.manifest_path,
            "asset_manifest_path": self.manifest_path,
            "whamAssetManifest": self.to_dict(),
            "whamAssets": dict(self.paths),
            "whamAssetPaths": dict(self.paths),
            "wham_asset_paths": dict(self.paths),
            "whamStageDiagnostics": {
                key: dict(value) for key, value in self.stage_diagnostics.items()
            },
            "assetPaths": dict(self.paths),
            "asset_paths": dict(self.paths),
            "checkpointPath": self.paths.get("checkpoint"),
            "checkpoint_path": self.paths.get("checkpoint"),
            "checkpoint": self.paths.get("checkpoint"),
            "bodyModelPath": body,
            "body_model_path": body,
            "body_model": body,
            "smplModelPath": body,
            "smpl_model_path": body,
            "smplxModelPath": body,
            "smplx_model_path": body,
            # Keep a ViTPose backbone checkpoint distinct from a precomputed
            # (N,D) feature archive.  Treating the same .pth path as both
            # makes the native adapter try to coerce model weights into frame
            # features and hides the actionable "extractor required" state.
            "imageFeatureBackbonePath": image,
            "image_feature_backbone_path": image,
            "image_feature_backbone": image,
            "imageFeaturePath": image_archive,
            "imageFeatureArchive": image_archive,
            "image_feature_archive": image_archive,
            "imageFeaturesPath": image_archive,
            "cameraModelPath": camera,
            "camera_model_path": camera,
            "camera": camera,
            "cameraPath": camera,
            "dpvoModelPath": dpvo,
            "dpvo_model_path": dpvo,
            "dpvo": dpvo,
            "dpvoPath": dpvo,
            "adapterPath": self.paths.get("adapter"),
            "adapter_path": self.paths.get("adapter"),
            "adapter": self.paths.get("adapter"),
            "assetContractVersion": int(self.contract_version),
            "asset_contract_version": int(self.contract_version),
            "whamMissingStages": list(self.missing_stages),
            "missing_stages": list(self.missing_stages),
            "whamFullParity": bool(self.full_parity_available),
            "full_parity_available": bool(self.full_parity_available),
            "whamAssetDiagnostic": self.diagnostic_text,
            "whamInferencePolicy": WHAM_INFERENCE_POLICY,
            "whamMissingAssetBehavior": WHAM_MISSING_ASSET_BEHAVIOR,
        }


def resolve_wham_assets(
    model_directory: Optional[str] = None,
    model_path: Optional[str] = None,
    adapter_module: Optional[str] = None,
    asset_manifest_path: Optional[str] = None,
    wham_asset_paths: Optional[Mapping[str, Any]] = None,
    *,
    asset_paths: Optional[Mapping[str, Any]] = None,
    wham_assets: Optional[Mapping[str, Any]] = None,
    manifest_path: Optional[str] = None,
    asset_manifest: Optional[str] = None,
    body_model_path: Optional[str] = None,
    smpl_model_path: Optional[str] = None,
    smplx_model_path: Optional[str] = None,
    body_model: Optional[str] = None,
    smpl_path: Optional[str] = None,
    smplx_path: Optional[str] = None,
    image_feature_backbone_path: Optional[str] = None,
    image_feature_path: Optional[str] = None,
    camera_model_path: Optional[str] = None,
    camera_path: Optional[str] = None,
    dpvo_model_path: Optional[str] = None,
    dpvo_path: Optional[str] = None,
    checkpoint_path: Optional[str] = None,
    adapter_path: Optional[str] = None,
) -> WhamAssetResolution:
    """Resolve WHAM stages from explicit values, a manifest, env, then cache names.

    Resolution is local-only and deterministic: no globbing, subprocesses, or
    network requests are used.  Explicit values win over manifest values;
    manifest values win over environment variables; then the configured model
    directory and finally the package ``models`` directory are checked.
    """
    if asset_manifest_path is None:
        asset_manifest_path = manifest_path or asset_manifest
    model_root_value = _absolute_candidate(model_directory)
    model_root = Path(model_root_value) if model_root_value else None

    merged_explicit: Dict[str, str] = {}
    # Optional precomputed (N,D) archive is kept outside the contract stages.
    # ``image_feature_path`` is intentionally never treated as ViTPose weights.
    image_feature_archive_path = _absolute_candidate(image_feature_path, model_root)
    for mapping in (asset_paths, wham_asset_paths, wham_assets):
        if not isinstance(mapping, Mapping):
            continue
        for key, value in mapping.items():
            normalized_key = str(key).strip().lower().replace("-", "_").replace(" ", "_")
            if normalized_key in {
                "manifest", "asset_manifest", "asset_manifest_path", "manifest_path",
                "wham_asset_manifest", "wham_asset_manifest_path",
            }:
                manifest_value = _path_value(value)
                if asset_manifest_path is None and manifest_value:
                    asset_manifest_path = manifest_value
                continue
            if normalized_key in {
                "image_feature_path", "image_features_path", "image_feature_archive",
                "imagefeaturepath", "imagefeaturespath", "imagefeaturearchive",
                "image_feature_archive_path", "imagefeaturearchivepath", "image_features_archive", "feature_archive",
                "featurearchive", "image_features", "imagefeatures", "image_feature", "imagefeature",
                "precomputed_image_features", "features_archive",
            }:
                archive_value = _path_value(value)
                if image_feature_archive_path is None and archive_value:
                    image_feature_archive_path = _absolute_candidate(archive_value, model_root)
                continue
            role = _canonical_wham_asset_role(key)
            path = _path_value(value)
            if role and path:
                merged_explicit[role] = path
    named_values = {
        "body_model": smplx_model_path or smplx_path or smpl_model_path or smpl_path or body_model_path or body_model,
        "image_feature_backbone": image_feature_backbone_path,
        "camera": camera_model_path or camera_path,
        "dpvo": dpvo_model_path or dpvo_path,
        "adapter": adapter_path,
    }
    for role, value in named_values.items():
        if value:
            merged_explicit[role] = value
    if checkpoint_path or model_path:
        merged_explicit["checkpoint"] = checkpoint_path or model_path
    if adapter_module:
        merged_explicit["adapter"] = adapter_module

    manifest_values: Dict[str, Any] = {}
    resolved_manifest_path: Optional[str] = None
    manifest_error: Optional[str] = None
    manifest_candidates, _ = _manifest_candidates(model_directory, asset_manifest_path)
    for source, candidate in manifest_candidates:
        try:
            if not candidate.is_file():
                if asset_manifest_path or os.environ.get(_WHAM_MANIFEST_ENV):
                    manifest_error = f"manifest not found: {candidate}"
                continue
            with candidate.open("r", encoding="utf-8") as handle:
                manifest_payload = json.load(handle)
            manifest_values = _manifest_asset_values(manifest_payload)
            if image_feature_archive_path is None:
                manifest_archive = _path_value(manifest_values.get("image_feature_archive"))
                if manifest_archive:
                    manifest_base = candidate.parent
                    image_feature_archive_path = _absolute_candidate(manifest_archive, manifest_base)
            resolved_manifest_path = str(candidate)
            break
        except (OSError, ValueError, TypeError) as exc:
            manifest_error = f"could not read manifest {candidate}: {exc}"
            if asset_manifest_path or os.environ.get(_WHAM_MANIFEST_ENV):
                break

    package_models = Path(__file__).resolve().parents[2] / "models"
    paths: Dict[str, Optional[str]] = {}
    sources: Dict[str, str] = {}
    for role in WHAM_ASSET_ROLES:
        candidates: List[Tuple[str, Optional[str], Optional[Path]]] = []
        explicit_is_authoritative = role in merged_explicit
        if explicit_is_authoritative:
            candidates.append(("explicit", merged_explicit[role], None))
        # An explicitly configured adapter is authoritative.  Falling back to
        # the bundled bridge after a user selected a missing adapter would hide
        # a configuration error and make the resulting provenance ambiguous.
        if role in manifest_values and not explicit_is_authoritative:
            manifest_base = Path(resolved_manifest_path).parent if resolved_manifest_path else model_root
            candidates.append(("manifest", manifest_values[role], manifest_base))
        if not explicit_is_authoritative:
            for env_name in _WHAM_ASSET_ENV.get(role, ()):
                env_value = os.environ.get(env_name)
                if env_value:
                    candidates.append(("environment", env_value, None))
            for name in _WHAM_DEFAULT_ASSET_NAMES.get(role, ()):
                if model_root:
                    candidates.append(("model_directory", name, model_root))
                candidates.append(("package", name, package_models))
        if role == "adapter" and not explicit_is_authoritative:
            # The bundled bridge lives beside this backend, not in the model
            # cache.  It is still a local contract asset and is copied to the
            # cache by the explicit Unity downloader action when requested.
            bundled_adapter = Path(__file__).resolve().parents[1] / "adapters" / "texmotion_wham_adapter.py"
            candidates.append(("package", str(bundled_adapter), None))

        found: Optional[str] = None
        found_source = "missing"
        for source, value, base in candidates:
            # Adapter module names are valid contract values even though they
            # are not filesystem paths.  File paths remain absolute.
            candidate_path = _existing_asset_path(value, base)
            if candidate_path:
                found = candidate_path
                found_source = source
                break
            if role == "adapter" and value and not os.path.isfile(str(value)) and _module_spec_available(str(value)):
                found = str(value)
                found_source = source
                break
        paths[role] = found
        if found:
            sources[role] = found_source

    stage_diagnostics: Dict[str, Dict[str, Any]] = {}
    for role in WHAM_ASSET_ROLES:
        path = paths.get(role)
        if role in merged_explicit:
            expected = _path_value(merged_explicit[role]) or "an explicit local path"
        else:
            expected = ", ".join(_WHAM_DEFAULT_ASSET_NAMES.get(role, ()))
        stage_diagnostics[role] = {
            "path": path,
            "source": sources.get(role, "missing"),
            "present": bool(path),
            "requiredForNative": role in WHAM_NATIVE_REQUIRED_ROLES,
            "requiredForFullParity": role in WHAM_FULL_REQUIRED_ROLES,
            "expected": expected or "an explicit local path",
        }

    missing = [role for role in WHAM_FULL_REQUIRED_ROLES if not paths.get(role)]
    native_available = all(paths.get(role) for role in WHAM_NATIVE_REQUIRED_ROLES)
    return WhamAssetResolution(
        contract_version=WHAM_ASSET_CONTRACT_VERSION,
        manifest_path=resolved_manifest_path,
        paths=paths,
        stage_diagnostics=stage_diagnostics,
        missing_stages=missing,
        native_available=bool(native_available),
        full_parity_available=not missing,
        manifest_error=manifest_error,
        image_feature_archive_path=image_feature_archive_path,
    )


# Private spelling retained for integrations that use the existing resolver
# naming convention in this module.
_resolve_wham_assets = resolve_wham_assets


@dataclass(frozen=True)
class PyTorchBackendConfig:
    """Serializable settings passed to a research adapter."""

    kind: str = "pytorch"
    model_path: Optional[str] = None
    adapter_module: Optional[str] = None
    device: str = "auto"
    max_sequence_frames: int = 0
    # Directory containing local Video2Motion assets (MediaPipe task model,
    # RTMPose weights, and optional quality checkpoints).  Quality adapters
    # receive this value so a native implementation can resolve its 2D
    # auxiliary detector without relying on global environment variables.
    model_directory: Optional[str] = None
    # Optional official runtime source checkout used by HMR2/4D-Humans.
    runtime_path: Optional[str] = None
    # Optional WHAM full-pipeline assets.  They remain unset for legacy
    # MediaPipe/RTMPose and generic PyTorch callers.
    asset_manifest_path: Optional[str] = None
    wham_asset_paths: Optional[Dict[str, Any]] = None
    body_model_path: Optional[str] = None
    smpl_model_path: Optional[str] = None
    smplx_model_path: Optional[str] = None
    image_feature_backbone_path: Optional[str] = None
    image_feature_path: Optional[str] = None
    # Optional in-process ViTPose preprocessing.  These values are kept in
    # the serializable adapter payload so the WHAM adapter can resolve the
    # runner lazily after it knows the checkpoint feature width.
    image_feature_extractor: Any = None
    image_feature_runner: Any = None
    image_feature_runner_config: Any = None
    image_feature_runner_module: Optional[str] = None
    image_feature_checkpoint_path: Optional[str] = None
    image_feature_model_definition: Any = None
    image_feature_config_path: Any = None
    image_feature_model_factory: Any = None
    # Direct ViTPose spellings are accepted for callers that construct this
    # dataclass instead of going through ``PyTorchPoseBackend``.
    vitpose_runner: Any = None
    vitpose_runner_config: Any = None
    vitpose_runner_module: Optional[str] = None
    vitpose_checkpoint_path: Optional[str] = None
    vitpose_model_definition: Any = None
    vitpose_config_path: Any = None
    vitpose_model_factory: Any = None
    camera_model_path: Optional[str] = None
    dpvo_model_path: Optional[str] = None
    wham_preprocess_directory: Optional[str] = None
    adapter_implementation: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        # Selecting WHAM in the editor and having a ViTPose checkpoint in the
        # configured model directory opts into the bundled inline runner.
        # Custom runner modules/objects still take precedence.
        runner_module = (
            self.image_feature_runner_module
            or self.vitpose_runner_module
            or (
                "pose_pipeline.vitpose_runner"
                if (self.image_feature_backbone_path or self.vitpose_checkpoint_path)
                else None
            )
        )
        return {
            "kind": self.kind,
            "modelPath": self.model_path,
            "adapterModule": self.adapter_module,
            "device": self.device,
            "maxSequenceFrames": self.max_sequence_frames,
            "modelDirectory": self.model_directory,
            "runtimePath": self.runtime_path,
            "assetManifestPath": self.asset_manifest_path,
            "whamAssetPaths": dict(self.wham_asset_paths or {}),
            "bodyModelPath": self.body_model_path,
            "smplModelPath": self.smpl_model_path,
            "smplxModelPath": self.smplx_model_path,
            "imageFeatureBackbonePath": self.image_feature_backbone_path,
            "imageFeaturePath": self.image_feature_path,
            "imageFeatureExtractor": self.image_feature_extractor,
            "imageFeatureRunner": self.image_feature_runner or self.vitpose_runner,
            "imageFeatureRunnerConfig": self.image_feature_runner_config or self.vitpose_runner_config,
            "imageFeatureRunnerModule": runner_module,
            "imageFeatureCheckpointPath": self.image_feature_checkpoint_path or self.vitpose_checkpoint_path,
            "imageFeatureModelDefinition": self.image_feature_model_definition or self.vitpose_model_definition,
            "imageFeatureConfigPath": self.image_feature_config_path or self.vitpose_config_path,
            "imageFeatureModelFactory": self.image_feature_model_factory or self.vitpose_model_factory,
            # ViTPose aliases are included for integrations that use the model
            # name rather than the generic image-feature terminology.
            "vitposeCheckpointPath": self.image_feature_checkpoint_path or self.vitpose_checkpoint_path,
            "vitposeModelDefinition": self.image_feature_model_definition or self.vitpose_model_definition,
            "vitposeConfigPath": self.image_feature_config_path or self.vitpose_config_path,
            "vitposeModelFactory": self.image_feature_model_factory or self.vitpose_model_factory,
            "vitposeRunner": self.image_feature_runner or self.vitpose_runner,
            "vitposeRunnerConfig": self.image_feature_runner_config or self.vitpose_runner_config,
            "vitposeRunnerModule": runner_module,
            "cameraModelPath": self.camera_model_path,
            "dpvoModelPath": self.dpvo_model_path,
            "whamPreprocessDirectory": self.wham_preprocess_directory,
            "adapterImplementation": self.adapter_implementation,
        }


def _torch_available() -> Tuple[bool, str]:
    """Check torch without importing it (important for the default runtime)."""
    if importlib.util.find_spec("torch") is None:
        return False, "PyTorch is not installed in the selected Python environment."
    return True, "PyTorch is installed."


def _module_spec_available(module_name: Optional[str]) -> bool:
    if not module_name:
        return False
    try:
        if os.path.isfile(module_name):
            return True
        return importlib.util.find_spec(module_name) is not None
    except (ImportError, ValueError):
        return False


def _resolve_model_path(model_path: Optional[str]) -> Optional[str]:
    candidates: List[str] = []
    if model_path:
        candidates.append(model_path)
    env_path = os.environ.get(_MODEL_ENV)
    if env_path:
        candidates.append(env_path)
    base = Path(__file__).resolve().parents[2]
    candidates.extend([
        str(base / "models" / "texmotion_quality.pt"),
        str(base / "models" / "wham.pt"),
        str(base / "models" / "hmr2.pt"),
        str(base / "models" / "hybrik.pt"),
    ])
    for candidate in candidates:
        if candidate and os.path.isfile(candidate):
            try:
                if os.path.getsize(candidate) > 1024:
                    return os.path.abspath(candidate)
            except OSError:
                continue
    return None


def _candidate_adapter(kind: str, explicit: Optional[str]) -> Optional[str]:
    if explicit:
        return explicit
    kind_key = kind.lower().strip()
    return os.environ.get(_ADAPTER_ENV_BY_KIND.get(kind_key, "")) or os.environ.get(_ADAPTER_ENV)


def check_pytorch_backend_available(
    kind: str = "pytorch",
    model_path: Optional[str] = None,
    adapter_module: Optional[str] = None,
    model_directory: Optional[str] = None,
    asset_manifest_path: Optional[str] = None,
    wham_asset_paths: Optional[Mapping[str, Any]] = None,
) -> Tuple[bool, str]:
    """Return a user-facing preflight result without importing torch.

    A quality backend is considered configured when either a valid TorchScript
    artifact or an adapter module is present.  WHAM/HMR2/HybrIK adapters remain
    optional and are never imported by the default MediaPipe/RTMPose path.
    """
    ok, reason = _torch_available()
    if not ok:
        return False, reason
    kind_key = kind.lower().strip()
    if kind_key == "wham":
        assets = resolve_wham_assets(
            model_directory=model_directory,
            model_path=model_path,
            adapter_module=adapter_module,
            asset_manifest_path=asset_manifest_path,
            wham_asset_paths=wham_asset_paths,
        )
        if not assets.native_available:
            return False, assets.diagnostic_text
        full_note = ""
        if not assets.full_parity_available:
            full_note = " " + assets.diagnostic_text
        return True, (
            "PyTorch WHAM native/full adapter contract is available locally."
            + full_note
        )

    resolved_adapter = _candidate_adapter(kind, adapter_module)
    if resolved_adapter and _module_spec_available(resolved_adapter):
        return True, f"PyTorch {kind} adapter is available ({resolved_adapter})."
    resolved_model = _resolve_model_path(model_path)
    if resolved_model:
        return True, f"PyTorch {kind} TorchScript model is available at {resolved_model}."
    if kind_key in _DEFAULT_ADAPTER_MODULES:
        default_name = _DEFAULT_ADAPTER_MODULES[kind_key]
        if _module_spec_available(default_name):
            return True, f"PyTorch {kind} adapter is available ({default_name})."
    return False, (
        f"No {kind} PyTorch model or adapter was found. Set {_MODEL_ENV} or "
        f"{_ADAPTER_ENV} after installing the optional quality environment."
    )


def _to_numpy(value: Any) -> Optional[np.ndarray]:
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        return value
    try:
        # Works for torch.Tensor while keeping torch optional.
        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "cpu"):
            value = value.cpu()
        if hasattr(value, "numpy"):
            return np.asarray(value.numpy())
    except Exception:
        return None
    try:
        return np.asarray(value)
    except Exception:
        return None


def _first_value(mapping: Dict[str, Any], names: Iterable[str]) -> Any:
    for name in names:
        if name in mapping:
            return mapping[name]
    return None


def _intervals_from(value: Any) -> List[UncertaintyInterval]:
    if value is None:
        return []
    if isinstance(value, dict):
        value = value.get("uncertaintyIntervals", value.get("intervals", []))
    if not isinstance(value, (list, tuple)):
        return []
    result: List[UncertaintyInterval] = []
    for item in value:
        if isinstance(item, UncertaintyInterval):
            result.append(item)
        elif isinstance(item, dict):
            try:
                result.append(UncertaintyInterval.from_dict(item))
            except (TypeError, ValueError):
                continue
    return result


class PyTorchPoseBackend(PoseBackend):
    """Adapter-backed temporal body model with optional TorchScript support."""

    def __init__(
        self,
        kind: str = "pytorch",
        model_path: Optional[str] = None,
        adapter_module: Optional[str] = None,
        device: str = "auto",
        max_sequence_frames: int = 0,
        model_directory: Optional[str] = None,
        asset_manifest_path: Optional[str] = None,
        wham_asset_paths: Optional[Mapping[str, Any]] = None,
        asset_paths: Optional[Mapping[str, Any]] = None,
        body_model_path: Optional[str] = None,
        smpl_model_path: Optional[str] = None,
        smplx_model_path: Optional[str] = None,
        body_model: Optional[str] = None,
        smpl_path: Optional[str] = None,
        smplx_path: Optional[str] = None,
        image_feature_backbone_path: Optional[str] = None,
        camera_model_path: Optional[str] = None,
        dpvo_model_path: Optional[str] = None,
        adapter_implementation: Optional[str] = None,
        wham_assets: Optional[Mapping[str, Any]] = None,
        wham_asset_manifest_path: Optional[str] = None,
        manifest_path: Optional[str] = None,
        asset_manifest: Optional[str] = None,
        image_feature_path: Optional[str] = None,
        camera_path: Optional[str] = None,
        dpvo_path: Optional[str] = None,
        adapter_path: Optional[str] = None,
        wham_preprocess_directory: Optional[str] = None,
        # Inline ViTPose runner hooks.  Keep these optional and late in the
        # signature so existing positional callers retain their behavior.
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
    ):
        self.kind = kind.lower().strip() or "pytorch"
        merged_asset_paths: Dict[str, Any] = {}
        if isinstance(asset_paths, Mapping):
            merged_asset_paths.update(asset_paths)
        if isinstance(wham_asset_paths, Mapping):
            merged_asset_paths.update(wham_asset_paths)
        if isinstance(wham_assets, Mapping):
            merged_asset_paths.update(wham_assets)
        resolved_manifest_path = asset_manifest_path or wham_asset_manifest_path or manifest_path or asset_manifest
        resolved_body_path = body_model_path or smplx_model_path or smplx_path or smpl_model_path or smpl_path or body_model
        resolved_image_backbone_path = image_feature_backbone_path
        resolved_image_archive_path = image_feature_path
        resolved_image_feature_runner = image_feature_runner or vitpose_runner
        resolved_image_feature_runner_config = (
            image_feature_runner_config or vitpose_runner_config
        )
        resolved_image_feature_runner_module = (
            image_feature_runner_module
            or image_feature_runner_path
            or vitpose_runner_module
        )
        resolved_image_feature_checkpoint = (
            image_feature_checkpoint_path
            or vitpose_checkpoint_path
            or resolved_image_backbone_path
        )
        resolved_image_feature_model_definition = (
            image_feature_model_definition or vitpose_model_definition
        )
        resolved_image_feature_config_path = image_feature_config_path or vitpose_config_path
        resolved_image_feature_model_factory = (
            image_feature_model_factory or vitpose_model_factory
        )
        resolved_camera_path = camera_model_path or camera_path
        resolved_dpvo_path = dpvo_model_path or dpvo_path
        if adapter_path:
            merged_asset_paths["adapter"] = adapter_path
        self.config = PyTorchBackendConfig(
            kind=self.kind,
            model_path=model_path,
            adapter_module=adapter_module,
            device=device,
            max_sequence_frames=max(0, int(max_sequence_frames)),
            model_directory=model_directory,
            runtime_path=runtime_path,
            asset_manifest_path=resolved_manifest_path,
            wham_asset_paths=merged_asset_paths,
            body_model_path=resolved_body_path,
            smpl_model_path=smpl_model_path or smpl_path,
            smplx_model_path=smplx_model_path or smplx_path,
            image_feature_backbone_path=resolved_image_backbone_path,
            image_feature_path=resolved_image_archive_path,
            image_feature_extractor=image_feature_extractor,
            image_feature_runner=resolved_image_feature_runner,
            image_feature_runner_config=resolved_image_feature_runner_config,
            image_feature_runner_module=resolved_image_feature_runner_module,
            image_feature_checkpoint_path=resolved_image_feature_checkpoint,
            image_feature_model_definition=resolved_image_feature_model_definition,
            image_feature_config_path=resolved_image_feature_config_path,
            image_feature_model_factory=resolved_image_feature_model_factory,
            camera_model_path=resolved_camera_path,
            dpvo_model_path=resolved_dpvo_path,
            wham_preprocess_directory=wham_preprocess_directory,
            adapter_implementation=adapter_implementation,
        )
        super().__init__(self.kind if self.kind != "pytorch" else "pytorch_quality")
        self.supports_sequence = True
        self._torch = None
        self._model = None
        self._adapter = None
        self._adapter_name: Optional[str] = None
        self._device = "cpu"
        self._resolved_model_path: Optional[str] = None
        self._last_error: Optional[str] = None
        # A quality backend is not ready merely because its module/checkpoint
        # loaded.  Keep the load and smoke-inference gate explicit so callers
        # can distinguish a usable backend from a configured-but-incompatible
        # asset and preserve the concrete reason when the factory falls back.
        self._preflight_status = "not_run"
        self._preflight_error: Optional[str] = None
        self._preflight_phase: Optional[str] = None
        self._preflight_frames = 0
        self._backend_ready = False
        self._incompatible_assets: List[str] = []
        self._sequence_error: Optional[str] = None
        self._asset_resolution: WhamAssetResolution = resolve_wham_assets(
            model_directory=model_directory,
            model_path=model_path,
            adapter_module=adapter_module,
            asset_manifest_path=resolved_manifest_path,
            wham_asset_paths=merged_asset_paths,
            body_model_path=resolved_body_path,
            smpl_model_path=smpl_model_path or smpl_path,
            smplx_model_path=smplx_model_path or smplx_path,
            image_feature_backbone_path=resolved_image_backbone_path,
            image_feature_path=resolved_image_archive_path,
            camera_model_path=resolved_camera_path,
            dpvo_model_path=resolved_dpvo_path,
            adapter_path=adapter_path,
        ) if self.kind == "wham" else resolve_wham_assets()
        # Public read-only-ish convenience for diagnostics and integrations.
        self.wham_asset_paths = dict(self._asset_resolution.paths) if self.kind == "wham" else {}
        self.last_uncertainty_intervals: List[UncertaintyInterval] = []
        self.last_result_metadata: Dict[str, Any] = {}
        self._adapter_metadata_snapshot: Dict[str, Any] = {}

    @property
    def model_path(self) -> Optional[str]:
        return self._resolved_model_path or self.config.model_path

    def is_available(self) -> bool:
        if self._preflight_status == "failed":
            return False
        ok, _ = check_pytorch_backend_available(
            self.kind,
            self.config.model_path,
            self.config.adapter_module,
            model_directory=self.config.model_directory,
            asset_manifest_path=self.config.asset_manifest_path,
            wham_asset_paths=self.wham_asset_paths,
        )
        return ok

    @staticmethod
    def _preflight_frame_result(result: Any) -> Any:
        """Extract one observation from common adapter smoke-test results."""
        if isinstance(result, Mapping):
            values = result.get("frames", result.get("observations"))
            if isinstance(values, (list, tuple)):
                return values[0] if values else None
            if isinstance(values, np.ndarray):
                return values[0] if values.ndim > 0 and len(values) else None
        if isinstance(result, (list, tuple)):
            return result[0] if result else None
        if isinstance(result, np.ndarray) and result.ndim > 0 and result.shape[0] == 1:
            return result[0]
        return result

    @staticmethod
    def _invoke_sequence_method(
        method: Any,
        frames: Sequence[np.ndarray],
        timestamps: Sequence[float],
        indices: Sequence[int],
    ) -> Any:
        """Call a temporal adapter without masking errors raised by inference.

        A few legacy adapters only accept ``frames`` (or ``frames`` and
        ``timestamps``).  Inspect the callable to select that compatibility
        signature before invoking it.  Retrying after a caught ``TypeError``
        is unsafe: a model can legitimately raise ``TypeError`` while
        executing, and the retry would replace the actionable model error with
        an unrelated Python argument error.
        """
        candidates = (
            (frames, timestamps, indices),
            (frames, timestamps),
            (frames,),
        )
        try:
            signature = inspect.signature(method)
        except (TypeError, ValueError):
            signature = None
        if signature is not None:
            for args in candidates:
                try:
                    signature.bind(*args)
                except TypeError:
                    continue
                return method(*args)
            # Preserve the adapter's normal Python error for an unsupported
            # contract instead of inventing a fallback invocation.
            return method(*candidates[0])
        return method(*candidates[0])

    @staticmethod
    def _result_error(result: Any) -> Optional[str]:
        if not isinstance(result, Mapping):
            return None
        for key in ("error", "errorMessage", "failure", "exception"):
            value = result.get(key)
            if value:
                return str(value)
        return None

    def _run_preflight(self) -> None:
        """Validate one short inference before exposing a quality backend as ready.

        Adapters may expose either a sequence method or a frame method.  The
        probe uses a deterministic black frame and validates that the result
        can cross the same normalization boundary used by extraction.  This
        catches corrupt checkpoints, incompatible model factories, and
        adapters that only construct successfully but cannot execute.
        """
        self._preflight_status = "running"
        self._preflight_phase = "short_inference"
        self._preflight_frames = 1
        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        timestamp = 0.0
        frame_index = 0
        if self._adapter is not None:
            sequence_method = getattr(self._adapter, "infer_sequence", None)
            sequence_method = sequence_method or getattr(self._adapter, "predict_sequence", None)
            if callable(sequence_method):
                result = self._invoke_sequence_method(
                    sequence_method, [frame], [timestamp], [frame_index]
                )
            else:
                result = self._invoke(frame, timestamp, frame_index)
        else:
            result = self._invoke(frame, timestamp, frame_index)

        result = self._preflight_frame_result(result)
        observation = self._frame_observation(
            result, frame, timestamp, frame_index, []
        )
        if observation is None:
            raise RuntimeError(
                self._result_error(result) or "short inference returned no observation"
            )
        observed_2d = any(
            getattr(point, "status", ObservationStatus.MISSING) != ObservationStatus.MISSING
            for point in observation.keypoints_2d
        )
        observed_3d = any(
            getattr(point, "status", ObservationStatus.MISSING) != ObservationStatus.MISSING
            for point in observation.landmarks_3d
        )
        if not (observed_2d or observed_3d):
            raise RuntimeError("short inference returned no finite pose observations")
        self._preflight_status = "passed"
        self._preflight_phase = "ready"
        self._preflight_error = None
        self._backend_ready = True
        if self._adapter is not None:
            setattr(self._adapter, "_preflight_status", "passed")
            setattr(self._adapter, "_preflight_error", None)

    def record_preflight_failure(
        self,
        error: Any,
        phase: str = "availability",
    ) -> None:
        """Record a failed dependency/asset gate before model initialization.

        The factory performs a cheap availability check before calling
        ``initialize``.  Preserve that rejection in the same metadata schema
        as a load or short-inference failure so a MediaPipe fallback cannot
        look like an unattempted quality backend.
        """
        message = str(error or "quality backend availability check failed")
        self._preflight_status = "failed"
        self._preflight_phase = str(phase or "availability")
        self._preflight_error = message
        self._backend_ready = False
        self._last_error = message
        self._incompatible_assets = ["backend_availability"]
        if self.kind == "wham" and "checkpoint" in self._asset_resolution.missing_stages:
            self._incompatible_assets.append("checkpoint")

    def get_capabilities(self) -> BackendCapabilities:
        capabilities = BackendCapabilities(
            name=self.name,
            has_2d=True,
            has_3d=True,
            has_visibility=True,
            has_presence=False,
            is_temporal=True,
            keypoint_count=33,
            supports_directml=False,
            supports_cuda=self._device.startswith("cuda"),
        )
        # BackendCapabilities predates the optional WHAM stages, so attach the
        # richer flags dynamically for consumers that understand them while
        # retaining the original dataclass/API for MediaPipe and RTMPose.
        setattr(capabilities, "supports_cpu", True)
        setattr(capabilities, "full_wham_parity", bool(self._asset_resolution.full_parity_available))
        setattr(capabilities, "native_wham_core", bool(self._asset_resolution.native_available))
        setattr(capabilities, "world_motion", bool(self._asset_resolution.full_parity_available))
        setattr(capabilities, "missing_asset_stages", tuple(self._asset_resolution.missing_stages))
        setattr(capabilities, "asset_contract_version", WHAM_ASSET_CONTRACT_VERSION)
        # CamelCase aliases mirror the serialized metadata contract.
        setattr(capabilities, "supportsCPU", True)
        setattr(capabilities, "fullWhamParity", bool(self._asset_resolution.full_parity_available))
        setattr(capabilities, "nativeWhamCore", bool(self._asset_resolution.native_available))
        setattr(capabilities, "worldMotion", bool(self._asset_resolution.full_parity_available))
        setattr(capabilities, "missingAssetStages", tuple(self._asset_resolution.missing_stages))
        return capabilities

    def _resolve_assets(self) -> WhamAssetResolution:
        """Refresh the local WHAM contract immediately before adapter loading."""
        if self.kind != "wham":
            return self._asset_resolution
        merged_paths = dict(self.config.wham_asset_paths or {})
        # Resolved paths provide discovered package/cache locations, while the
        # original config retains explicit entries (including an optional
        # feature archive that is intentionally outside the contract roles).
        merged_paths.update(self.wham_asset_paths or {})
        self._asset_resolution = resolve_wham_assets(
            model_directory=self.config.model_directory,
            model_path=self.config.model_path,
            adapter_module=self.config.adapter_module,
            asset_manifest_path=self.config.asset_manifest_path,
            wham_asset_paths=merged_paths,
            body_model_path=self.config.body_model_path,
            smpl_model_path=self.config.smpl_model_path,
            smplx_model_path=self.config.smplx_model_path,
            image_feature_backbone_path=self.config.image_feature_backbone_path,
            image_feature_path=self.config.image_feature_path,
            camera_model_path=self.config.camera_model_path,
            dpvo_model_path=self.config.dpvo_model_path,
        )
        self.wham_asset_paths = dict(self._asset_resolution.paths)
        return self._asset_resolution

    def _load_adapter_module(self, module_ref: str):
        if os.path.isfile(module_ref):
            # A pre-native TexMotion install may still point at the cached
            # placeholder bridge (the file was copied once during download and
            # therefore survives package upgrades).  Prefer the package bridge
            # when the requested file has the canonical WHAM filename but does
            # not contain the native implementation marker.  The editor also
            # refreshes the cache on extraction; this guard keeps direct CLI
            # invocations self-healing when the editor has not been reopened.
            try:
                requested_path = Path(module_ref).resolve()
                if requested_path.name.lower() == "texmotion_wham_adapter.py":
                    bundled_path = (Path(__file__).resolve().parents[1] /
                                     "adapters" / "texmotion_wham_adapter.py")
                    if bundled_path.is_file() and requested_path != bundled_path:
                        requested_text = requested_path.read_text(encoding="utf-8", errors="ignore")
                        bundled_text = bundled_path.read_text(encoding="utf-8", errors="ignore")
                        # Refresh pre-native cached bridges as well as older
                        # bridges that lack the first-frame initialization
                        # forwarding seam.  The latter otherwise keeps WHAM
                        # on its neutral embedded seed even when the native
                        # core now supports MediaPipe-conditioned seeding.
                        if ("pose_pipeline.adapters.wham_native" in bundled_text and
                                ("pose_pipeline.adapters.wham_native" not in requested_text or
                                 "set_initialization_from_observation" not in requested_text or
                                 "get_mediapipe_auxiliary" not in requested_text)):
                            sys.stderr.write(
                                "[TexMotion] Legacy WHAM adapter bridge detected; using the bundled native bridge.\n"
                            )
                            sys.stderr.flush()
                            module_ref = str(bundled_path)
            except Exception:
                # Loading the explicitly configured adapter below remains the
                # fallback for custom deployments and unusual filesystem
                # encodings.
                pass
            module_name = f"texmotion_external_{self.kind}_adapter"
            spec = importlib.util.spec_from_file_location(module_name, module_ref)
            if spec is None or spec.loader is None:
                raise ImportError(f"Unable to load adapter module: {module_ref}")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
        return importlib.import_module(module_ref)

    def initialize(self) -> bool:
        self._last_error = None
        self._preflight_status = "loading"
        self._preflight_phase = "model_load"
        self._preflight_error = None
        self._preflight_frames = 0
        self._backend_ready = False
        self._incompatible_assets = []
        self._adapter_metadata_snapshot = {}
        try:
            assets = self._resolve_assets()
            if self.kind == "wham" and not assets.native_available:
                # A quality request must never turn an absent WHAM checkpoint
                # or adapter into an apparently successful MediaPipe result.
                # The caller may still choose its normal fallback, but it gets
                # an actionable, stage-specific error first.
                raise FileNotFoundError(assets.diagnostic_text)
            if self.kind == "wham":
                self._resolved_model_path = assets.paths.get("checkpoint")
            self._torch = importlib.import_module("torch")
            requested = self.config.device.lower().strip()
            if requested in ("", "auto"):
                requested = "cuda" if bool(self._torch.cuda.is_available()) else "cpu"
            if requested.startswith("cuda") and not bool(self._torch.cuda.is_available()):
                requested = "cpu"
            self._device = requested

            adapter_ref = _candidate_adapter(self.kind, self.config.adapter_module)
            if not adapter_ref and self.kind == "wham":
                resolved_adapter = assets.paths.get("adapter")
                if resolved_adapter:
                    adapter_ref = resolved_adapter
            if not adapter_ref and self.kind in _DEFAULT_ADAPTER_MODULES:
                candidate = _DEFAULT_ADAPTER_MODULES[self.kind]
                if _module_spec_available(candidate):
                    adapter_ref = candidate
            if adapter_ref and _module_spec_available(adapter_ref):
                module = self._load_adapter_module(adapter_ref)
                adapter_config = self.config.to_dict()
                if self.kind == "wham":
                    adapter_config.update(assets.to_adapter_dict())
                    # The resolver may discover a ViTPose checkpoint from the
                    # model directory/manifest rather than from an explicit
                    # constructor argument.  Keep the inline runner's
                    # checkpoint aliases populated for external adapters that
                    # do not inspect ``imageFeatureBackbonePath``.
                    feature_checkpoint = (
                        self.config.image_feature_checkpoint_path
                        or assets.paths.get("image_feature_backbone")
                    )
                    adapter_config["imageFeatureCheckpointPath"] = feature_checkpoint
                    adapter_config["vitposeCheckpointPath"] = feature_checkpoint
                    # The editor may provide only a model directory; in that
                    # case the asset resolver discovers ``vitpose*.pth`` after
                    # ``PyTorchBackendConfig.to_dict`` has already been
                    # serialized.  Carry the bundled runner module through to
                    # external adapters as well, so selecting WHAM in the UI
                    # always attempts the same inline ViTPose stage.
                    if feature_checkpoint and not (
                        adapter_config.get("imageFeatureRunnerModule")
                        or adapter_config.get("vitposeRunnerModule")
                    ):
                        adapter_config["imageFeatureRunnerModule"] = (
                            "pose_pipeline.vitpose_runner"
                        )
                        adapter_config["vitposeRunnerModule"] = (
                            "pose_pipeline.vitpose_runner"
                        )
                    # Keep the old modelPath key authoritative for bundled
                    # and external bridges while exposing the explicit alias
                    # expected by newer full-stage implementations.
                    adapter_config["modelPath"] = assets.paths.get("checkpoint") or self.config.model_path
                    adapter_config["adapterModule"] = adapter_ref
                    adapter_config["adapterImplementation"] = (
                        self.config.adapter_implementation
                        or os.environ.get("TEXMOTION_WHAM_ADAPTER_IMPL")
                    )
                adapter_config.update({"torch": self._torch, "device": self._device})
                factory = getattr(module, "create_backend", None) or getattr(module, "create_adapter", None)
                if factory is not None:
                    self._adapter = factory(adapter_config)
                else:
                    cls = getattr(module, "Backend", None) or getattr(module, "Adapter", None)
                    self._adapter = cls(adapter_config) if cls else module
                self._adapter_name = adapter_ref
                init = getattr(self._adapter, "initialize", None)
                if init is not None and init() is False:
                    # Adapter bridges (notably WHAM) retain a concrete
                    # configuration error such as a missing implementation
                    # module.  Preserve that detail for the UI instead of
                    # replacing it with an opaque generic failure.
                    adapter_error = getattr(self._adapter, "_error", None)
                    if not adapter_error:
                        try:
                            adapter_error = (self._adapter.get_metadata() or {}).get("error")
                        except Exception:
                            adapter_error = None
                    raise RuntimeError(adapter_error or "PyTorch adapter initialization returned false")
            else:
                self._resolved_model_path = _resolve_model_path(self.config.model_path)
                if not self._resolved_model_path:
                    raise FileNotFoundError("No TorchScript model or quality adapter was found")
                self._model = self._torch.jit.load(self._resolved_model_path, map_location=self._device)
                self._model.eval()

            # Construction/loading is only the first gate.  A one-frame
            # smoke inference must cross the same result normalization path
            # before the backend is advertised as ready to the extractor.
            self._run_preflight()
            self.is_initialized = True
            return True
        except Exception as exc:
            self._last_error = str(exc)
            if self._preflight_phase == "short_inference" or self._preflight_status == "running":
                self._preflight_status = "failed"
                self._preflight_phase = "short_inference"
                self._incompatible_assets = ["backend_short_inference"]
                if self._adapter_name:
                    self._incompatible_assets.append("adapter")
            else:
                self._preflight_status = "failed"
                self._preflight_phase = self._preflight_phase or "model_load"
                self._incompatible_assets = ["backend_model_load"]
                if self.kind == "wham":
                    self._incompatible_assets.append("checkpoint")
            self._preflight_error = str(exc)
            self._backend_ready = False
            if self._adapter is not None:
                setattr(self._adapter, "_preflight_status", "failed")
                setattr(self._adapter, "_preflight_error", str(exc))
                try:
                    adapter_metadata = getattr(self._adapter, "get_metadata", None)
                    if callable(adapter_metadata):
                        snapshot = adapter_metadata()
                        if isinstance(snapshot, dict):
                            self._adapter_metadata_snapshot = dict(snapshot)
                except Exception:
                    self._adapter_metadata_snapshot = {}
            self._model = None
            self._adapter = None
            self.is_initialized = False
            sys.stderr.write(f"[TexMotion Warning] PyTorch {self.kind} backend unavailable: {exc}. Falling back.\n")
            return False

    def _invoke(self, frame: np.ndarray, timestamp: float, index: int) -> Any:
        if self._adapter is not None:
            method = getattr(self._adapter, "infer_frame", None) or getattr(self._adapter, "predict", None)
            if method is None and callable(self._adapter):
                method = self._adapter
            if method is None:
                raise RuntimeError("Quality adapter must expose infer_frame, predict, or __call__")
            return method(frame, timestamp, index)
        if self._model is None:
            return None
        # Generic TorchScript convention: BCHW RGB float in [0, 1].  Concrete
        # research models should use an adapter for crop/camera preprocessing.
        rgb = frame[:, :, ::-1].copy()
        tensor = self._torch.from_numpy(np.transpose(rgb, (2, 0, 1))).float().div(255.0).unsqueeze(0)
        return self._model(tensor.to(self._device))

    def _normalize_points(self, value: Any, dims: int, width: int, height: int) -> Tuple[np.ndarray, np.ndarray]:
        arr = _to_numpy(value)
        if arr is None:
            return np.zeros((0, dims), dtype=np.float32), np.zeros((0,), dtype=np.float32)
        arr = np.asarray(arr)
        while arr.ndim > 2 and arr.shape[0] == 1:
            arr = arr[0]
        if arr.ndim != 2 or arr.shape[1] < 2:
            return np.zeros((0, dims), dtype=np.float32), np.zeros((0,), dtype=np.float32)
        coords = arr[:, :dims].astype(np.float32, copy=False)
        score = arr[:, dims] if arr.shape[1] > dims else np.ones((len(coords),), dtype=np.float32)
        score = np.clip(np.nan_to_num(score, nan=0.0), 0.0, 1.0)
        finite_rows = np.isfinite(coords).all(axis=1)
        # Do not let ``nan_to_num`` turn a corrupt model row into a seemingly
        # valid origin.  The row remains in the canonical slot for alignment,
        # but a zero score lets callers classify it as missing.
        score = np.where(finite_rows, score, 0.0)
        # Accept either normalized coordinates or pixel coordinates.
        if dims == 2:
            if len(coords) and float(np.nanmax(np.abs(coords[:, :2]))) > 1.5:
                coords[:, 0] /= max(1.0, float(width))
                coords[:, 1] /= max(1.0, float(height))
            coords[:, :2] = np.clip(coords[:, :2], 0.0, 1.0)
        else:
            # Body-model coordinates are camera/root-relative meters and may
            # legitimately be negative.  Only sanitize non-finite values.
            coords = np.nan_to_num(coords, nan=0.0, posinf=0.0, neginf=0.0)
        return coords, score.astype(np.float32)

    def _frame_observation(
        self,
        output: Any,
        frame: np.ndarray,
        timestamp: float,
        frame_index: int,
        inherited_intervals: Optional[List[UncertaintyInterval]] = None,
    ) -> Optional[FrameObservations]:
        if isinstance(output, FrameObservations):
            output.detector_name = self.name
            adapter_metadata = dict(output.backend_metadata or {})
            output.backend_metadata.update(self.get_metadata())
            output.backend_metadata.update(adapter_metadata)
            output.backend_metadata.setdefault("coordinateSystem", "smplx")
            if inherited_intervals:
                existing = list(output.uncertainty_intervals or [])
                existing_keys = {
                    (u.start_frame, u.end_frame, u.reason, u.recommended_action)
                    for u in existing
                }
                existing.extend(
                    u for u in inherited_intervals
                    if u.contains_frame(frame_index) and
                    (u.start_frame, u.end_frame, u.reason, u.recommended_action) not in existing_keys
                )
                output.uncertainty_intervals = existing
            return output
        mapping = output if isinstance(output, dict) else {}
        if isinstance(output, (tuple, list)) and not mapping:
            # A common adapter return is (keypoints2d, landmarks3d, confidence).
            values = list(output)
            mapping = {
                "keypoints2d": values[0] if values else None,
                "landmarks3d": values[1] if len(values) > 1 else None,
                "confidence": values[2] if len(values) > 2 else None,
            }
        nested = mapping.get("observation") if mapping else None
        if isinstance(nested, (dict, FrameObservations)):
            return self._frame_observation(nested, frame, timestamp, frame_index, inherited_intervals)

        width = int(frame.shape[1]) if frame is not None and frame.ndim >= 2 else 1
        height = int(frame.shape[0]) if frame is not None and frame.ndim >= 2 else 1
        points2d = _first_value(mapping, ("keypoints2d", "keypoints_2d", "joints2d", "joints_2d", "keypoints"))
        points3d = _first_value(mapping, ("landmarks3d", "landmarks_3d", "keypoints3d", "keypoints_3d", "joints3d", "joints_3d"))
        if points3d is None and points2d is None and not mapping:
            points3d = output
        p2, s2 = self._normalize_points(points2d, 2, width, height)
        p3, s3 = self._normalize_points(points3d, 3, width, height)
        # If a generic model returned Kx3, it is interpreted as 3D.  A model
        # returning only 2D should use an explicit keypoints2d field.
        if len(p3) == 0 and len(p2) == 0:
            return None
        count = max(len(p2), len(p3))
        kps = [Keypoint2D(0.0, 0.0, score=0.0, visibility=0.0, presence=0.0, status=ObservationStatus.MISSING) for _ in range(33)]
        lms = [Keypoint3D(0.0, 0.0, 0.0, score=0.0, visibility=0.0, presence=0.0, status=ObservationStatus.MISSING) for _ in range(33)]
        adapter_metadata = mapping.get("metadata", {}) if isinstance(mapping, dict) else {}
        if not isinstance(adapter_metadata, dict):
            adapter_metadata = {}
        topology = str(
            adapter_metadata.get("jointTopology", adapter_metadata.get("joint_topology", ""))
            or ""
        ).strip().lower().replace(" ", "_")
        map_2d = (
            SMPLX_TO_MEDIAPIPE if len(p2) == 22 else
            COCO17_TO_MEDIAPIPE if len(p2) == 17 else
            {i: i for i in range(min(33, len(p2)))}
        )
        map_3d = (
            SMPLX_TO_MEDIAPIPE if len(p3) == 22 else
            WHAM_J17_TO_MEDIAPIPE if len(p3) == 17 and topology in _WHAM_J17_TOPOLOGIES else
            COCO17_TO_MEDIAPIPE if len(p3) == 17 else
            {i: i for i in range(min(33, len(p3)))}
        )
        for source_idx, target_idx in map_2d.items():
            score2 = float(s2[source_idx]) if source_idx < len(s2) else 0.0
            if source_idx < len(p2):
                x, y = p2[source_idx, :2]
                status = (
                    ObservationStatus.MISSING if score2 <= 0.0 else
                    (ObservationStatus.OBSERVED if score2 >= 0.35 else ObservationStatus.OCCLUDED)
                )
                kps[target_idx] = Keypoint2D(float(x), float(y), score2, score2, score2, status)
        # COCO/WHAM-17 both expose ankles but no explicit heel/foot landmarks.
        # Alias each mapped ankle into the complete MediaPipe foot chain so
        # downstream SMPL-X conversion and overlay rendering never substitute
        # an unrelated MediaPipe heel for a valid temporal-model ankle.
        if len(p2) == 17:
            for ankle_idx, heel_idx, foot_idx in ((27, 29, 31), (28, 30, 32)):
                ankle = kps[ankle_idx]
                if ankle.status != ObservationStatus.MISSING:
                    alias = Keypoint2D(
                        ankle.x, ankle.y, ankle.score, ankle.visibility,
                        ankle.presence, ankle.status,
                    )
                    kps[heel_idx] = alias
                    kps[foot_idx] = alias
        for source_idx, target_idx in map_3d.items():
            score2 = float(s2[source_idx]) if source_idx < len(s2) else 0.0
            score3 = float(s3[source_idx]) if source_idx < len(s3) else score2
            if source_idx < len(p3):
                x, y, z = p3[source_idx, :3]
                status = (
                    ObservationStatus.MISSING if score3 <= 0.0 else
                    (ObservationStatus.OBSERVED if score3 >= 0.35 else ObservationStatus.PREDICTED)
                )
                lms[target_idx] = Keypoint3D(float(x), float(y), float(z), score3, score3, score3, status)
        if len(p3) == 17:
            for ankle_idx, heel_idx, foot_idx in ((27, 29, 31), (28, 30, 32)):
                ankle = lms[ankle_idx]
                if ankle.status != ObservationStatus.MISSING:
                    alias = Keypoint3D(
                        ankle.x, ankle.y, ankle.z, ankle.score, ankle.visibility,
                        ankle.presence, ankle.status,
                    )
                    lms[heel_idx] = alias
                    lms[foot_idx] = alias
            # WHAM's J17 includes head-top and head but has no eye/ear slots.
            # Reusing head-top for both ears gives the canonical conversion a
            # stable head anchor without inventing a second learned point.
            if topology in _WHAM_J17_TOPOLOGIES and 13 < len(p3):
                source_idx = 13
                score3 = float(s3[source_idx]) if source_idx < len(s3) else 0.0
                if np.isfinite(p3[source_idx, :3]).all():
                    status = ObservationStatus.OBSERVED if score3 >= 0.35 else ObservationStatus.PREDICTED
                    head_top = Keypoint3D(
                        float(p3[source_idx, 0]), float(p3[source_idx, 1]),
                        float(p3[source_idx, 2]), score3, score3, score3, status,
                    )
                    lms[7] = Keypoint3D(
                        head_top.x, head_top.y, head_top.z, head_top.score,
                        head_top.visibility, head_top.presence, head_top.status,
                    )
                    lms[8] = Keypoint3D(
                        head_top.x, head_top.y, head_top.z, head_top.score,
                        head_top.visibility, head_top.presence, head_top.status,
                    )
        confidence = mapping.get("confidence", mapping.get("score", None))
        if confidence is None:
            vals = [x for x in (s2 if len(s2) else s3) if np.isfinite(x)]
            confidence = float(np.mean(vals)) if vals else 0.0
        confidence = float(np.clip(np.asarray(confidence).reshape(-1)[0], 0.0, 1.0))
        all_intervals = _intervals_from(mapping.get("uncertaintyIntervals", mapping.get("uncertainties")))
        if inherited_intervals:
            all_intervals = inherited_intervals
        matching = [u for u in all_intervals if u.contains_frame(frame_index)]
        # Quality adapters conventionally emit SMPL-X coordinates (+Y up,
        # +Z forward), while the legacy extractor consumes MediaPipe world
        # coordinates (+Y down, +Z away).  Carry the convention explicitly so
        # the integration layer can convert it once instead of silently
        # inverting a learned 3D hypothesis.  An adapter may override this
        # when it deliberately returns MediaPipe coordinates.
        metadata = self.get_metadata()
        adapter_metadata = mapping.get("metadata", {})
        if isinstance(adapter_metadata, dict):
            metadata.update(adapter_metadata)
        metadata.setdefault("coordinateSystem", "smplx")
        return FrameObservations(
            frame_index=frame_index,
            timestamp=timestamp,
            keypoints_2d=kps,
            landmarks_3d=lms,
            raw_confidence=confidence,
            detector_name=self.name,
            backend_metadata=metadata,
            uncertainty_intervals=matching,
        )

    def detect(self, frame_bgr: np.ndarray, timestamp_sec: float, frame_index: int = 0) -> Optional[FrameObservations]:
        if not self.is_initialized or frame_bgr is None or frame_bgr.size == 0:
            return None
        try:
            result = self._invoke(frame_bgr, timestamp_sec, frame_index)
            return self._frame_observation(result, frame_bgr, timestamp_sec, frame_index, self.last_uncertainty_intervals)
        except Exception as exc:
            self._last_error = str(exc)
            return None

    def set_initialization_from_observation(self, observation: Any) -> bool:
        """Forward a first-frame auxiliary pose seed to the quality adapter.

        WHAM's bundled adapter exposes this optional seam so the extractor can
        initialize the recurrent camera-space state from MediaPipe before the
        temporal pass.  Other adapters simply return ``False`` and retain
        their own initialization contract.
        """
        if self._adapter is None:
            return False
        setter = getattr(self._adapter, "set_initialization_from_observation", None)
        if not callable(setter):
            return False
        try:
            return bool(setter(observation))
        except Exception as exc:
            self._last_error = str(exc)
            return False

    def get_mediapipe_auxiliary(self) -> Any:
        """Return the WHAM adapter's MediaPipe detector when it owns one.

        Native WHAM initializes a detector for its 2D input stream.  The
        extractor also needs a MediaPipe 3D observation for WHAM fusion, but
        constructing a second Tasks graph in the same Windows process can
        trigger a native access violation (especially with the XNNPACK
        delegate).  Expose the already initialized detector so the extractor
        can share it instead of allocating another graph.  ``None`` is
        returned for RTMPose/custom detectors and for adapters that do not
        expose a detector.
        """
        adapter = self._adapter
        if adapter is None:
            return None
        provider = getattr(adapter, "get_mediapipe_auxiliary", None)
        if callable(provider):
            try:
                candidate = provider()
            except Exception:
                candidate = None
            if candidate is not None:
                return candidate
        # Older cached WHAM bridge modules may not implement the forwarding
        # method above, while still retaining the native implementation and
        # its detector as ``_implementation._detector``.  Inspect both
        # ownership layers so an extraction never falls back to constructing
        # a second MediaPipe Tasks graph merely because the bridge is stale.
        owners = [adapter, getattr(adapter, "_implementation", None)]
        for owner in owners:
            candidate = getattr(owner, "_detector", None) if owner is not None else None
            if candidate is None:
                continue
            name = str(getattr(candidate, "name", "") or "").strip().lower()
            if name in ("mediapipe", "mediapipe_pose", "mediapipe_3d"):
                detect = getattr(candidate, "detect", None)
                if callable(detect) and bool(getattr(candidate, "is_initialized", True)):
                    return candidate
        return None

    def detect_sequence(
        self,
        frames_bgr: Sequence[np.ndarray],
        timestamps_sec: Sequence[float],
        frame_indices: Optional[Sequence[int]] = None,
    ) -> List[Optional[FrameObservations]]:
        if not self.is_initialized:
            return [None for _ in frames_bgr]
        self._sequence_error = None
        self.last_result_metadata = {}
        self._last_error = None
        try:
            frame_count = len(frames_bgr)
            timestamp_count = len(timestamps_sec)
        except TypeError as exc:
            self._sequence_error = f"invalid sequence inputs: {exc}"
            self._last_error = self._sequence_error
            self.last_result_metadata = {"sequenceInferenceError": self._sequence_error}
            return [None for _ in (frames_bgr or [])]
        if frame_count != timestamp_count:
            self._sequence_error = (
                "sequence inputs have different lengths: "
                f"frames={frame_count}, timestamps={timestamp_count}"
            )
            self._last_error = self._sequence_error
            self.last_result_metadata = {"sequenceInferenceError": self._sequence_error}
            return [None for _ in frames_bgr]
        indices = list(frame_indices if frame_indices is not None else range(len(frames_bgr)))
        if len(indices) != frame_count:
            self._sequence_error = (
                "sequence inputs have different lengths: "
                f"frames={frame_count}, frame_indices={len(indices)}"
            )
            self._last_error = self._sequence_error
            self.last_result_metadata = {"sequenceInferenceError": self._sequence_error}
            return [None for _ in frames_bgr]
        valid = [(frame, float(ts), int(idx)) for frame, ts, idx in zip(frames_bgr, timestamps_sec, indices) if frame is not None]
        if self.config.max_sequence_frames > 0:
            valid = valid[: self.config.max_sequence_frames]
        if not valid:
            return [None for _ in frames_bgr]
        try:
            if self._adapter is not None:
                method = getattr(self._adapter, "infer_sequence", None) or getattr(self._adapter, "predict_sequence", None)
                if method is not None:
                    result = self._invoke_sequence_method(
                        method,
                        [x[0] for x in valid],
                        [x[1] for x in valid],
                        [x[2] for x in valid],
                    )
                else:
                    result = [self._invoke(x[0], x[1], x[2]) for x in valid]
            else:
                result = [self._invoke(x[0], x[1], x[2]) for x in valid]
            self.last_uncertainty_intervals = _intervals_from(result.get("uncertaintyIntervals", result.get("uncertainties"))) if isinstance(result, dict) else []
            sequence_values = result.get("frames", result.get("observations")) if isinstance(result, dict) else result
            # Adapters often return one dictionary of batched tensors instead
            # of ``frames=[...]``.  Split only tensors whose leading dimension
            # is the sequence length; metadata and interval lists stay global.
            if isinstance(result, dict) and sequence_values is None:
                batched_keys = []
                for key, value in result.items():
                    array = _to_numpy(value)
                    if array is not None and array.ndim >= 1 and array.shape[0] == len(valid):
                        batched_keys.append(key)
                if batched_keys:
                    sequence_values = [
                        {key: (value[index] if key in batched_keys else value) for key, value in result.items()}
                        for index in range(len(valid))
                    ]
            if isinstance(sequence_values, np.ndarray):
                sequence_values = list(sequence_values)
            if not isinstance(sequence_values, (list, tuple)):
                sequence_values = [sequence_values]
            result_error = self._result_error(result)
            if not sequence_values or all(value is None for value in sequence_values):
                self._sequence_error = result_error or "sequence inference returned no observations"
            elif len(sequence_values) < len(valid):
                self._sequence_error = result_error or (
                    "sequence inference returned fewer observations than requested: "
                    f"returned={len(sequence_values)}, requested={len(valid)}"
                )
            # Prefer the frame identity emitted by the adapter over list
            # position.  Temporal adapters may reorder/pad their output while
            # batching; pairing by position silently retimes a pose to the
            # wrong source frame and can turn a short occlusion into a full
            # sequence collapse.  Keep positional assignment only as a
            # compatibility fallback for legacy adapters that do not emit an
            # identity field.
            expected_indices = {idx for _frame, _ts, idx in valid}

            def _value_identity(value: Any) -> Optional[int]:
                candidate = value
                if isinstance(candidate, FrameObservations):
                    candidate = getattr(candidate, "frame_index", None)
                elif isinstance(candidate, dict):
                    for key in (
                        "frameIndex", "frame_index", "sourceFrameIndex",
                        "source_frame_index", "index",
                    ):
                        if candidate.get(key) is not None:
                            candidate = candidate.get(key)
                            break
                    else:
                        metadata = candidate.get("metadata", candidate.get("backendMetadata", {}))
                        if isinstance(metadata, dict):
                            for key in (
                                "frameIndex", "frame_index", "sourceFrameIndex",
                                "source_frame_index", "index",
                            ):
                                if metadata.get(key) is not None:
                                    candidate = metadata.get(key)
                                    break
                            else:
                                candidate = None
                        else:
                            candidate = None
                else:
                    candidate = None
                try:
                    parsed = int(candidate)
                except (TypeError, ValueError, OverflowError):
                    return None
                return parsed if parsed in expected_indices else None

            by_index: Dict[int, Any] = {}
            unassigned_values: List[Any] = []
            for value in sequence_values:
                identity = _value_identity(value)
                if identity is None or identity in by_index:
                    unassigned_values.append(value)
                else:
                    by_index[identity] = value

            remaining_indices = [idx for _frame, _ts, idx in valid if idx not in by_index]
            for idx, value in zip(remaining_indices, unassigned_values):
                by_index[idx] = value
            out: List[Optional[FrameObservations]] = []
            for frame, ts, idx in zip(frames_bgr, timestamps_sec, indices):
                value = by_index.get(idx)
                out.append(self._frame_observation(value, frame, float(ts), idx, self.last_uncertainty_intervals) if frame is not None and value is not None else None)
            self.last_result_metadata = dict(result.get("metadata", {})) if isinstance(result, dict) and isinstance(result.get("metadata", {}), dict) else {}
            if self._sequence_error:
                self._last_error = self._sequence_error
                self.last_result_metadata["sequenceInferenceError"] = self._sequence_error
            return out
        except Exception as exc:
            self._sequence_error = str(exc)
            self._last_error = self._sequence_error
            self.last_result_metadata = {
                "sequenceInferenceError": self._sequence_error,
            }
            return [None for _ in frames_bgr]

    def get_metadata(self) -> Dict[str, Any]:
        metadata = super().get_metadata()
        native_runtime_details: Optional[Dict[str, Any]] = None
        metadata.update({
            "backendKind": self.kind,
            "runtime": "pytorch",
            "device": self._device,
            "modelPath": self.model_path,
            "adapter": self._adapter_name,
            "coordinateSystem": "smplx",
            "uncertaintyIntervals": [u.to_dict() for u in self.last_uncertainty_intervals],
        })
        if self.kind == "wham":
            # Include the contract before adapter provenance so an external
            # adapter cannot hide missing local stages.  The same fields are
            # emitted before and after initialize for useful preflight UI.
            metadata.update(self._asset_resolution.to_metadata())
            metadata["capabilities"] = {
                "nativeWhamCore": bool(self._asset_resolution.native_available),
                "fullWhamParity": bool(self._asset_resolution.full_parity_available),
                "worldMotion": bool(self._asset_resolution.full_parity_available),
                "supportsCPU": True,
                "supportsCUDA": bool(self._device.startswith("cuda")),
                "missingStages": list(self._asset_resolution.missing_stages),
            }
            metadata["capabilityFlags"] = dict(metadata["capabilities"])
        if self._last_error:
            metadata["lastError"] = self._last_error
        if self._adapter is not None:
            adapter_metadata = getattr(self._adapter, "get_metadata", None)
            if callable(adapter_metadata):
                try:
                    details = adapter_metadata()
                    if isinstance(details, dict):
                        # Keep the generic backend provenance while exposing
                        # model-specific fields such as native WHAM core,
                        # checkpoint path, and omitted optional stages.
                        metadata.update(details)
                        if (
                            self.kind == "wham"
                            and str(details.get("adapterImplementation", "")).strip().lower()
                            == "texmotion_native_wham"
                        ):
                            # The static contract only proves that files exist.
                            # Preserve the native adapter's post-initialization
                            # capability report so raw ViTPose/DPVO weights do
                            # not masquerade as executable image/camera stages.
                            native_runtime_details = dict(details)
                except Exception:
                    pass
        elif self._adapter_metadata_snapshot:
            metadata.update(self._adapter_metadata_snapshot)
        metadata.update(self.last_result_metadata)
        if self.kind == "wham":
            # The resolved contract remains authoritative for static paths and
            # stage diagnostics.  A native adapter may additionally report
            # runtime-only requirements (for example a frame-feature archive
            # or exported DPVO poses); those must be merged after the static
            # contract or the UI would claim full parity merely because the
            # checkpoint files are present.
            static_metadata = self._asset_resolution.to_metadata()
            metadata.update(static_metadata)
            static_capabilities = {
                "nativeWhamCore": bool(self._asset_resolution.native_available),
                "fullWhamParity": bool(self._asset_resolution.full_parity_available),
                "worldMotion": bool(self._asset_resolution.full_parity_available),
                "supportsCPU": True,
                "supportsCUDA": bool(self._device.startswith("cuda")),
                "missingStages": list(self._asset_resolution.missing_stages),
            }
            metadata["capabilities"] = static_capabilities
            metadata["capabilityFlags"] = dict(static_capabilities)

            if native_runtime_details is not None:
                runtime_capabilities = native_runtime_details.get("capabilities", {})
                if not isinstance(runtime_capabilities, dict):
                    runtime_capabilities = {}
                runtime_missing = native_runtime_details.get("missingOptionalAssets", [])
                if not isinstance(runtime_missing, (list, tuple)):
                    runtime_missing = []
                runtime_missing = [str(item) for item in runtime_missing if str(item).strip()]

                merged_capabilities = dict(static_capabilities)
                # Keep every detailed native capability while retaining the
                # stable summary keys consumed by Unity and external clients.
                merged_capabilities.update(runtime_capabilities)
                native_core = bool(
                    runtime_capabilities.get(
                        "nativeMotionEncoder", static_capabilities["nativeWhamCore"]
                    )
                )
                runtime_full = bool(native_core and not runtime_missing)
                runtime_world = bool(
                    runtime_capabilities.get(
                        "worldTrajectory",
                        runtime_capabilities.get("cameraMotion", False),
                    )
                )
                merged_capabilities.update({
                    "nativeWhamCore": native_core,
                    "fullWhamParity": runtime_full,
                    "worldMotion": runtime_world,
                    "supportsCPU": True,
                    "supportsCUDA": bool(self._device.startswith("cuda")),
                    "missingStages": list(runtime_missing),
                })
                metadata["capabilities"] = merged_capabilities
                metadata["capabilityFlags"] = dict(merged_capabilities)
                metadata["whamFullParity"] = runtime_full
                metadata["whamMissingStages"] = list(runtime_missing)
                metadata["missingOptionalAssets"] = list(runtime_missing)

                runtime_errors = native_runtime_details.get("optionalErrors", [])
                if not isinstance(runtime_errors, (list, tuple)):
                    runtime_errors = []
                runtime_errors = [str(item) for item in runtime_errors if str(item).strip()]
                metadata["optionalErrors"] = list(runtime_errors)
                if runtime_missing:
                    detail = ", ".join(runtime_missing)
                    diagnostic = (
                        "WHAM checkpoint assets are present, but runtime stages are unavailable: "
                        + detail
                        + ". Downloaded weights require an offline feature extractor and/or "
                        "exported camera poses before full-parity inference can run."
                    )
                    if runtime_errors:
                        diagnostic += " " + " ".join(runtime_errors)
                    metadata["whamAssetDiagnostic"] = diagnostic
        # These fields are deliberately written last.  Adapter metadata is
        # useful provenance, but an adapter must not spoof the selected
        # backend, readiness gate, or missing/incompatible asset report.
        missing_assets: List[str] = []
        if self.kind == "wham":
            missing_assets.extend(str(item) for item in self._asset_resolution.missing_stages)
            reported_missing = metadata.get("missingAssets", [])
            if isinstance(reported_missing, (list, tuple)):
                missing_assets.extend(str(item) for item in reported_missing)
            runtime_missing = metadata.get("missingOptionalAssets", [])
            if isinstance(runtime_missing, (list, tuple)):
                missing_assets.extend(str(item) for item in runtime_missing)
        missing_assets = list(dict.fromkeys(item for item in missing_assets if item.strip()))
        incompatible_assets = list(self._incompatible_assets)
        reported_incompatible = metadata.get("incompatibleAssets", [])
        if isinstance(reported_incompatible, (list, tuple)):
            incompatible_assets.extend(str(item) for item in reported_incompatible)
        runtime_errors = metadata.get("optionalErrors", [])
        if isinstance(runtime_errors, (list, tuple)):
            for error in runtime_errors:
                text = str(error).strip()
                if text and any(token in text.lower() for token in ("incompat", "invalid", "requires", "unavailable")):
                    incompatible_assets.append(text)
        incompatible_assets = list(dict.fromkeys(str(item) for item in incompatible_assets if str(item).strip()))
        metadata.update({
            "selectedBackend": self.kind,
            "backendReady": bool(self._backend_ready and self.is_initialized),
            "preflightStatus": self._preflight_status,
            "preflightPhase": self._preflight_phase,
            "preflightFrames": int(self._preflight_frames),
            "preflightError": self._preflight_error,
            "sequenceInferenceError": self._sequence_error,
            "missingAssets": missing_assets,
            "incompatibleAssets": incompatible_assets,
            "assetDiagnostics": (
                {key: dict(value) for key, value in self._asset_resolution.stage_diagnostics.items()}
                if self.kind == "wham" else {}
            ),
        })
        return metadata

    def close(self):
        if self._adapter is not None:
            close = getattr(self._adapter, "close", None)
            if close:
                try:
                    close()
                except Exception:
                    pass
        self._adapter = None
        self._model = None
        self._torch = None
        self.is_initialized = False


# Friendly alias for callers that use the generic naming convention.
TorchPoseBackend = PyTorchPoseBackend


__all__ = [
    "PyTorchBackendConfig",
    "PyTorchPoseBackend",
    "TorchPoseBackend",
    "check_pytorch_backend_available",
    "WhamAssetResolution",
    "resolve_wham_assets",
    "_resolve_wham_assets",
    "WHAM_ASSET_CONTRACT_VERSION",
    "WHAM_ASSET_ROLES",
    "SMPLX_TO_MEDIAPIPE",
]
