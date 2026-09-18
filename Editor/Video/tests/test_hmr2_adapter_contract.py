"""Contracts for the bundled official 4D-Humans/HMR2 bridge."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


VIDEO_DIR = Path(__file__).resolve().parents[1]
ROOT = VIDEO_DIR.parents[1]


def test_hmr2_uses_the_bundled_texmotion_bridge() -> None:
    """Selecting HMR2 must not require the user to author a Python bridge."""
    backend = (VIDEO_DIR / "pose_pipeline" / "backends" / "pytorch_backend.py").read_text(
        encoding="utf-8"
    )
    assert '"hmr2": "pose_pipeline.adapters.texmotion_hmr2_adapter"' in backend
    adapter = VIDEO_DIR / "pose_pipeline" / "adapters" / "texmotion_hmr2_adapter.py"
    assert adapter.is_file()
    assert importlib.util.spec_from_file_location("texmotion_hmr2_adapter", adapter) is not None


def test_hmr2_bridge_exposes_the_quality_adapter_contract() -> None:
    adapter = (VIDEO_DIR / "pose_pipeline" / "adapters" / "texmotion_hmr2_adapter.py").read_text(
        encoding="utf-8"
    )
    assert "def create_backend" in adapter
    assert "def infer_sequence" in adapter
    assert "def infer_frame" in adapter
    assert "HMR2 / 4D-Humans" in adapter


def test_hmr2_settings_resolve_bundled_adapter_before_manual_override() -> None:
    downloader = (VIDEO_DIR / "VideoModelDownloader.cs").read_text(encoding="utf-8-sig")
    assert "GetBundledHmr2AdapterPath" in downloader
    assert "VideoAssetRole.Hmr2Adapter" in downloader


def _load_adapter_module():
    path = VIDEO_DIR / "pose_pipeline" / "adapters" / "texmotion_hmr2_adapter.py"
    spec = importlib.util.spec_from_file_location("texmotion_hmr2_contract_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_hmr2_bridge_maps_official_openpose25_to_coco17() -> None:
    module = _load_adapter_module()
    source = np.arange(25 * 3, dtype=np.float32).reshape(25, 3)
    result = module._coerce_official_joints(source)
    assert result.shape == (17, 3)
    np.testing.assert_array_equal(result[1], source[15])
    np.testing.assert_array_equal(result[-1], source[11])


def test_hmr2_bridge_projects_crop_coordinates_to_full_frame() -> None:
    module = _load_adapter_module()
    result = module._project_crop_points(
        np.asarray([[-0.5, -0.5], [0.5, 0.5]], dtype=np.float32),
        np.asarray([50.0, 50.0], dtype=np.float32),
        np.asarray(100.0, dtype=np.float32),
        np.asarray([100.0, 100.0], dtype=np.float32),
    )
    np.testing.assert_allclose(result, np.asarray([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32))


def test_hmr2_options_forward_runtime_and_body_model_paths() -> None:
    runner = (VIDEO_DIR / "VideoMotionJobRunner.cs").read_text(encoding="utf-8-sig")
    extractor = (VIDEO_DIR / "video_pose_extractor.py").read_text(encoding="utf-8")
    assert "--hmr2-runtime" in runner
    assert "--hmr2-body-model" in runner
    assert "hmr2_runtime_path" in extractor
    assert "hmr2_body_model_path" in extractor


def test_hmr2_bridge_reports_missing_official_runtime_without_silent_success(tmp_path) -> None:
    module = _load_adapter_module()
    adapter = module.HMR2Adapter(
        {
            "runtimePath": str(tmp_path / "missing-hmr2-runtime"),
            "modelPath": str(tmp_path / "missing-hmr2a_model.tar.gz"),
        }
    )
    assert adapter.initialize() is False
    metadata = adapter.get_metadata()
    assert metadata["officialRunnerStatus"] == "error"
    assert "4D-Humans" in metadata["error"]


def test_hmr2_sequence_runner_uses_bounded_batches() -> None:
    module = _load_adapter_module()
    adapter = module.HMR2Adapter({"batchSize": 3})
    assert adapter._batch_size == 3
    source = (VIDEO_DIR / "pose_pipeline" / "adapters" / "texmotion_hmr2_adapter.py").read_text(
        encoding="utf-8"
    )
    assert "for start in range(0, len(frames), self._batch_size)" in source


def test_hmr2_bridge_accepts_the_upstream_data_archive_name(tmp_path) -> None:
    module = _load_adapter_module()
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    archive = model_dir / "hmr2_data.tar.gz"
    archive.write_bytes(b"placeholder")
    # The resolver must consider the official archive spelling even when the
    # archive is not extractable in this lightweight contract test.
    checkpoint, _root, error = module._resolve_checkpoint({"modelDirectory": str(model_dir)})
    assert checkpoint is None
    assert error is not None
    assert "hmr2_data.tar.gz" in error or "No .ckpt" in error
