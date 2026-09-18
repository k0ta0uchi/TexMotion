"""Focused tests for the optional PyTorch quality backend contract."""

import os
import textwrap

import numpy as np
import pytest

from pose_pipeline.backends import create_pose_backend
from pose_pipeline.backends import pytorch_backend
from pose_pipeline.backends.pytorch_backend import PyTorchPoseBackend, check_pytorch_backend_available
from pose_pipeline.observations import ObservationStatus


def test_pytorch_preflight_is_safe_when_quality_stack_is_missing(monkeypatch):
    monkeypatch.delenv("TEXMOTION_PYTORCH_ADAPTER", raising=False)
    monkeypatch.delenv("TEXMOTION_PYTORCH_MODEL", raising=False)
    available, reason = check_pytorch_backend_available("wham", model_path="missing.pt")
    assert isinstance(available, bool)
    assert isinstance(reason, str) and reason


def test_quality_factory_falls_back_to_mediapipe_without_model():
    backend = create_pose_backend(preferred="wham", model_path="missing-quality.pt")
    try:
        assert backend.name == "mediapipe"
    finally:
        backend.close()


def test_quality_factory_marks_unavailable_preflight_failed_and_preserves_reason():
    """An availability rejection is a failed quality preflight, not an unexplained fallback."""
    backend = create_pose_backend(
        preferred="wham",
        pytorch_model_path="missing-quality-preflight.pt",
        strict_quality=False,
    )
    try:
        metadata = backend.get_metadata()
        failed = metadata["fallbackBackendMetadata"]
        assert failed["preflightStatus"] == "failed"
        assert failed["preflightPhase"] == "availability"
        assert failed["preflightError"]
        assert failed["missingAssets"]
        assert "Requested WHAM backend could not initialize" in metadata["fallbackReason"]
    finally:
        backend.close()


def test_pytorch_adapter_sequence_normalizes_22_joints(tmp_path, mock_frame_bgr):
    adapter_file = tmp_path / "quality_adapter.py"
    adapter_file.write_text(textwrap.dedent("""
        import numpy as np
        class Adapter:
            def infer_sequence(self, frames, timestamps, indices):
                t = len(frames)
                k = np.zeros((t, 22, 3), dtype=np.float32)
                k[:, 0, 1] = 0.1
                k[:, 20, 0] = 0.35
                return {
                    'landmarks3d': k,
                    'confidence': np.full((t,), 0.91, dtype=np.float32),
                    'uncertaintyIntervals': [
                        {'startFrame': 1, 'endFrame': 2,
                         'reason': 'leg_crossing_ambiguity',
                         'confidence': 0.42, 'recommendedAction': 'swap_crossing'}
                    ],
                    'metadata': {'checkpoint': 'fixture-quality'}
                }
        def create_backend(config):
            return Adapter()
    """), encoding="utf-8")

    backend = PyTorchPoseBackend(kind="wham", adapter_module=str(adapter_file), device="cpu")
    if not backend.initialize():
        # Environments without optional PyTorch should still exercise the
        # no-import/fallback tests above; this test is meaningful when torch is
        # installed (the supported quality profile).
        return
    try:
        frames = [mock_frame_bgr, mock_frame_bgr, mock_frame_bgr]
        observations = backend.detect_sequence(frames, [0.0, 1 / 30, 2 / 30], [0, 1, 2])
        assert len(observations) == 3
        assert observations[1].detector_name == "wham"
        assert len(observations[1].landmarks_3d) == 33
        # SMPL-X wrist 20 maps to the canonical MediaPipe left-wrist slot 15.
        assert observations[1].landmarks_3d[15].x == pytest.approx(0.35, abs=1e-6)
        assert observations[1].uncertainty_intervals[0].reason == "leg_crossing_ambiguity"
        assert observations[0].landmarks_3d[15].status == ObservationStatus.OBSERVED
        assert backend.get_metadata()["backendKind"] == "wham"
        assert backend.get_metadata()["device"] == "cpu"
    finally:
        backend.close()


def test_coco17_quality_observation_aliases_ankles_to_heels_and_feet(mock_frame_bgr):
    """COCO-17 has no heels; aliases must keep the 33-slot leg chain connected."""
    backend = PyTorchPoseBackend(kind="wham", device="cpu")
    points = np.zeros((17, 4), dtype=np.float32)
    points[15, :3] = [0.11, -0.70, 0.02]
    points[16, :3] = [-0.12, -0.71, -0.01]
    points[:, 3] = 0.9

    observation = backend._frame_observation(
        {"landmarks3d": points, "confidence": 0.9},
        mock_frame_bgr,
        timestamp=0.0,
        frame_index=0,
    )

    assert observation is not None
    left_ankle = observation.landmarks_3d[27]
    right_ankle = observation.landmarks_3d[28]
    for index in (29, 31):
        assert observation.landmarks_3d[index].status == ObservationStatus.OBSERVED
        assert observation.landmarks_3d[index].to_array().tolist() == pytest.approx(left_ankle.to_array().tolist())
    for index in (30, 32):
        assert observation.landmarks_3d[index].status == ObservationStatus.OBSERVED
        assert observation.landmarks_3d[index].to_array().tolist() == pytest.approx(right_ankle.to_array().tolist())


def test_wham_j17_quality_observation_uses_h36m_order_for_3d(mock_frame_bgr):
    """WHAM's learned J17 output must not be interpreted as COCO order."""
    backend = PyTorchPoseBackend(kind="wham", device="cpu")
    points = np.zeros((17, 4), dtype=np.float32)
    # WHAM J17: rankle, rknee, rhip, lhip, lknee, lankle, rwrist,
    # relbow, rshoulder, lshoulder, lelbow, lwrist, neck, headtop, hip,
    # spine, head.  Distinct values expose an accidental COCO mapping.
    for index in range(17):
        points[index, :3] = [float(index), float(index) + 0.1, float(index) + 0.2]
    points[:, 3] = 0.9

    observation = backend._frame_observation(
        {
            "landmarks3d": points,
            "confidence": 0.9,
            "metadata": {"jointTopology": "wham_j17"},
        },
        mock_frame_bgr,
        timestamp=0.0,
        frame_index=0,
    )

    assert observation is not None
    assert observation.landmarks_3d[28].to_array().tolist() == pytest.approx(points[0, :3].tolist())
    assert observation.landmarks_3d[26].to_array().tolist() == pytest.approx(points[1, :3].tolist())
    assert observation.landmarks_3d[24].to_array().tolist() == pytest.approx(points[2, :3].tolist())
    assert observation.landmarks_3d[23].to_array().tolist() == pytest.approx(points[3, :3].tolist())
    assert observation.landmarks_3d[16].to_array().tolist() == pytest.approx(points[6, :3].tolist())
    assert observation.landmarks_3d[14].to_array().tolist() == pytest.approx(points[7, :3].tolist())
    assert observation.landmarks_3d[12].to_array().tolist() == pytest.approx(points[8, :3].tolist())
    assert observation.landmarks_3d[11].to_array().tolist() == pytest.approx(points[9, :3].tolist())
    assert observation.landmarks_3d[0].to_array().tolist() == pytest.approx(points[16, :3].tolist())
    # Head-top aliases to both ears; ankle aliases fill the foot chain.
    assert observation.landmarks_3d[7].to_array().tolist() == pytest.approx(points[13, :3].tolist())
    assert observation.landmarks_3d[8].to_array().tolist() == pytest.approx(points[13, :3].tolist())
    assert observation.landmarks_3d[29].to_array().tolist() == pytest.approx(points[5, :3].tolist())
    assert observation.landmarks_3d[30].to_array().tolist() == pytest.approx(points[0, :3].tolist())


def test_pytorch_backend_has_no_runtime_state_before_initialize():
    backend = PyTorchPoseBackend(kind="hybrik", model_path="missing.pt")
    assert backend.is_initialized is False
    assert backend.detect(np.zeros((4, 4, 3), dtype=np.uint8), 0.0) is None
    backend.close()


def test_wham_asset_resolution_is_deterministic_and_explicit_paths_win(tmp_path):
    """The offline WHAM contract must resolve every stage without network access."""
    model_directory = tmp_path / "models"
    model_directory.mkdir()
    manifest = model_directory / "wham_asset_manifest.json"
    manifest.write_text(
        textwrap.dedent(
            """
            {
              "contractVersion": 1,
              "assets": {
                "checkpoint": "manifest-checkpoint.pth.tar",
                "bodyModel": "manifest-smplx.npz",
                "imageFeatureBackbone": "manifest-vitpose.pth",
                "camera": "manifest-camera.yaml",
                "dpvo": "manifest-dpvo.pth",
                "adapter": "manifest-adapter.py"
              }
            }
            """
        ),
        encoding="utf-8",
    )
    for file_name in (
        "manifest-checkpoint.pth.tar",
        "manifest-smplx.npz",
        "manifest-vitpose.pth",
        "manifest-camera.yaml",
        "manifest-dpvo.pth",
        "manifest-adapter.py",
    ):
        (model_directory / file_name).write_bytes(b"fixture")

    explicit_checkpoint = tmp_path / "explicit-checkpoint.pth.tar"
    explicit_adapter = tmp_path / "explicit-adapter.py"
    explicit_checkpoint.write_bytes(b"fixture")
    explicit_adapter.write_bytes(b"fixture")

    resolver = getattr(pytorch_backend, "resolve_wham_assets", None)
    assert callable(resolver)
    resolved = resolver(
        model_directory=str(model_directory),
        model_path=str(explicit_checkpoint),
        adapter_module=str(explicit_adapter),
        asset_manifest_path=str(manifest),
        wham_asset_paths={"body_model": str(model_directory / "manifest-smplx.npz")},
    )

    assert resolved.paths["checkpoint"] == str(explicit_checkpoint.resolve())
    assert resolved.paths["adapter"] == str(explicit_adapter.resolve())
    assert resolved.paths["body_model"] == str((model_directory / "manifest-smplx.npz").resolve())
    assert resolved.full_parity_available is True
    assert resolved.missing_stages == []
    assert resolved.to_dict()["contractVersion"] == 1

    missing_checkpoint = model_directory / "does-not-exist.pth.tar"
    blocked = resolver(
        model_directory=str(model_directory),
        checkpoint_path=str(missing_checkpoint),
        asset_manifest_path=str(manifest),
    )
    assert blocked.paths["checkpoint"] is None
    assert str(missing_checkpoint) in blocked.diagnostic_text


def test_wham_asset_metadata_exposes_missing_full_stages(tmp_path):
    checkpoint = tmp_path / "wham.pth.tar"
    adapter = tmp_path / "adapter.py"
    checkpoint.write_bytes(b"fixture")
    adapter.write_bytes(b"fixture")

    backend = PyTorchPoseBackend(
        kind="wham",
        model_path=str(checkpoint),
        adapter_module=str(adapter),
        model_directory=str(tmp_path),
        device="cpu",
    )
    metadata = backend.get_metadata()
    capabilities = backend.get_capabilities()

    assert "assetContractVersion" in metadata
    assert metadata["whamFullParity"] is False
    assert metadata["capabilityFlags"]["fullWhamParity"] is False
    assert "body_model" in metadata["whamMissingStages"]
    assert "image_feature_backbone" in metadata["whamMissingStages"]
    assert "dpvo" in metadata["whamMissingStages"]
    assert "offline" in metadata["whamAssetDiagnostic"].lower()
    assert getattr(capabilities, "full_wham_parity") is False
    assert getattr(capabilities, "missing_asset_stages")


def test_wham_metadata_declares_non_silent_missing_asset_policy(tmp_path):
    """A missing optional stage must be explicit while extraction may continue."""
    checkpoint = tmp_path / "wham.pth.tar"
    adapter = tmp_path / "adapter.py"
    checkpoint.write_bytes(b"fixture")
    adapter.write_bytes(b"fixture")

    backend = PyTorchPoseBackend(
        kind="wham",
        model_path=str(checkpoint),
        adapter_module=str(adapter),
        model_directory=str(tmp_path),
        device="cpu",
    )
    metadata = backend.get_metadata()

    assert metadata["whamInferencePolicy"] == "continue_with_explicit_fallback"
    assert metadata["whamMissingAssetBehavior"]
    assert "never silently" in metadata["whamMissingAssetBehavior"].lower()


def test_wham_adapter_receives_resolved_optional_asset_paths(tmp_path):
    """Full/native adapters receive paths even when they own stage loading."""
    adapter_file = tmp_path / "fixture_adapter.py"
    adapter_file.write_text(
        textwrap.dedent(
            """
            class Adapter:
                def __init__(self, config):
                    self.config = config
                def initialize(self):
                    return True
                def infer_frame(self, frame, timestamp, index):
                    return {'landmarks3d': [[0.0, 0.0, 0.0, 1.0]]}
            def create_backend(config):
                return Adapter(config)
            """
        ),
        encoding="utf-8",
    )
    assets = {}
    for role, file_name in {
        "checkpoint": "wham.pth.tar",
        "body_model": "smplx.npz",
        "image_feature_backbone": "vitpose.pth",
        "camera": "camera.yaml",
        "dpvo": "dpvo.pth",
    }.items():
        path = tmp_path / file_name
        path.write_bytes(b"fixture")
        assets[role] = str(path)

    backend = PyTorchPoseBackend(
        kind="wham",
        model_path=assets["checkpoint"],
        adapter_module=str(adapter_file),
        model_directory=str(tmp_path),
        wham_asset_paths=assets,
        device="cpu",
    )
    if not backend.initialize():
        pytest.skip("PyTorch is not installed in the test environment")
    try:
        config = backend._adapter.config
        assert config["bodyModelPath"] == str((tmp_path / "smplx.npz").resolve())
        assert config["imageFeatureBackbonePath"] == str((tmp_path / "vitpose.pth").resolve())
        assert config["cameraModelPath"] == str((tmp_path / "camera.yaml").resolve())
        assert config["dpvoModelPath"] == str((tmp_path / "dpvo.pth").resolve())
        assert config["whamAssets"]["checkpoint"] == str((tmp_path / "wham.pth.tar").resolve())
    finally:
        backend.close()


def test_wham_adapter_payload_keeps_vitpose_backbone_distinct_from_feature_archive(tmp_path):
    model_directory = tmp_path / "models"
    model_directory.mkdir()
    backbone = model_directory / "vitpose-huge.pth"
    backbone.write_bytes(b"fixture")
    resolved = pytorch_backend.resolve_wham_assets(
        model_directory=str(model_directory),
        image_feature_backbone_path=str(backbone),
    )

    payload = resolved.to_adapter_dict()

    assert payload["imageFeatureBackbonePath"] == str(backbone.resolve())
    assert payload.get("imageFeaturePath") is None
    assert payload.get("imageFeatureArchive") is None


def test_wham_config_auto_connects_bundled_vitpose_runner(tmp_path):
    """A WHAM selection with a ViTPose checkpoint opts into the inline runner."""
    backbone = tmp_path / "vitpose-huge.pth"
    backbone.write_bytes(b"fixture")

    config = pytorch_backend.PyTorchBackendConfig(
        kind="wham",
        image_feature_backbone_path=str(backbone),
    )
    payload = config.to_dict()

    assert payload["imageFeatureRunnerModule"] == "pose_pipeline.vitpose_runner"
    assert payload["vitposeRunnerModule"] == "pose_pipeline.vitpose_runner"


def test_wham_config_forwards_downloaded_runner_definition_and_config(tmp_path):
    definition = tmp_path / "wham_vitpose_model_definition.py"
    config_path = tmp_path / "wham_vitpose_runner_config.json"
    definition.write_text("# fixture", encoding="utf-8")
    config_path.write_text("{}", encoding="utf-8")

    payload = pytorch_backend.PyTorchBackendConfig(
        kind="wham",
        image_feature_backbone_path=str(tmp_path / "vitpose-huge.pth"),
        image_feature_model_definition=str(definition),
        image_feature_config_path=str(config_path),
    ).to_dict()

    assert payload["imageFeatureModelDefinition"] == str(definition)
    assert payload["imageFeatureConfigPath"] == str(config_path)
    assert payload["vitposeModelDefinition"] == str(definition)
    assert payload["vitposeConfigPath"] == str(config_path)


def test_wham_discovered_vitpose_runner_is_forwarded_to_adapter(tmp_path):
    """Model-directory discovery keeps automatic runner wiring for adapters."""
    adapter_file = tmp_path / "fixture_adapter.py"
    adapter_file.write_text(
        textwrap.dedent(
            """
            class Adapter:
                def __init__(self, config):
                    self.config = config
                def initialize(self):
                    return True
                def infer_frame(self, frame, timestamp, index):
                    return {'landmarks3d': [[0.0, 0.0, 0.0, 1.0]]}
            def create_backend(config):
                return Adapter(config)
            """
        ),
        encoding="utf-8",
    )
    for file_name in ("wham.pth.tar", "vitpose-huge.pth"):
        (tmp_path / file_name).write_bytes(b"fixture")

    backend = PyTorchPoseBackend(
        kind="wham",
        model_path=str(tmp_path / "wham.pth.tar"),
        adapter_module=str(adapter_file),
        model_directory=str(tmp_path),
        device="cpu",
    )
    if not backend.initialize():
        pytest.skip("PyTorch is not installed in the test environment")
    try:
        config = backend._adapter.config
        assert config["imageFeatureBackbonePath"] == str((tmp_path / "vitpose-huge.pth").resolve())
        assert config["imageFeatureRunnerModule"] == "pose_pipeline.vitpose_runner"
        assert config["vitposeRunnerModule"] == "pose_pipeline.vitpose_runner"
    finally:
        backend.close()


def test_wham_initialize_requires_short_inference_preflight(tmp_path):
    """Loading an adapter is not enough to mark a requested WHAM backend ready."""
    checkpoint = tmp_path / "wham.pth.tar"
    checkpoint.write_bytes(b"fixture")
    adapter_file = tmp_path / "preflight_failure_adapter.py"
    adapter_file.write_text(
        textwrap.dedent(
            """
            class Adapter:
                def initialize(self):
                    return True
                def infer_frame(self, frame, timestamp, index):
                    raise RuntimeError('fixture short inference incompatibility')
            def create_backend(config):
                return Adapter()
            """
        ),
        encoding="utf-8",
    )

    backend = PyTorchPoseBackend(
        kind="wham",
        model_path=str(checkpoint),
        adapter_module=str(adapter_file),
        device="cpu",
    )
    try:
        assert backend.initialize() is False
        assert backend.is_initialized is False
        metadata = backend.get_metadata()
        assert metadata["preflightStatus"] == "failed"
        assert "short inference incompatibility" in metadata["preflightError"]
        assert metadata["backendReady"] is False
    finally:
        backend.close()


def test_wham_preflight_preserves_sequence_execution_typeerror(tmp_path):
    """A TypeError raised by model execution must not be replaced by a retry signature error."""
    checkpoint = tmp_path / "wham.pth.tar"
    checkpoint.write_bytes(b"fixture")
    adapter_file = tmp_path / "preflight_sequence_typeerror_adapter.py"
    adapter_file.write_text(
        textwrap.dedent(
            """
            class Adapter:
                def initialize(self):
                    return True
                def infer_sequence(self, frames, timestamps, indices):
                    del frames, timestamps, indices
                    raise TypeError('fixture sequence model execution type mismatch')
            def create_backend(config):
                return Adapter()
            """
        ),
        encoding="utf-8",
    )

    backend = PyTorchPoseBackend(
        kind="wham",
        model_path=str(checkpoint),
        adapter_module=str(adapter_file),
        device="cpu",
    )
    try:
        assert backend.initialize() is False
        metadata = backend.get_metadata()
        assert metadata["preflightPhase"] == "short_inference"
        assert "sequence model execution type mismatch" in metadata["preflightError"]
    finally:
        backend.close()


def test_wham_initialize_reports_ready_only_after_short_inference(tmp_path):
    """A loadable adapter with a valid smoke result becomes ready."""
    checkpoint = tmp_path / "wham.pth.tar"
    checkpoint.write_bytes(b"fixture")
    adapter_file = tmp_path / "preflight_ready_adapter.py"
    adapter_file.write_text(
        textwrap.dedent(
            """
            class Adapter:
                def initialize(self):
                    return True
                def infer_frame(self, frame, timestamp, index):
                    del frame, timestamp
                    return {
                        'frameIndex': index,
                        'landmarks3d': [[0.0, 0.0, 0.0, 1.0]],
                        'confidence': 1.0,
                    }
                def close(self):
                    pass
            def create_backend(config):
                return Adapter()
            """
        ),
        encoding="utf-8",
    )

    backend = PyTorchPoseBackend(
        kind="wham",
        model_path=str(checkpoint),
        adapter_module=str(adapter_file),
        device="cpu",
    )
    try:
        assert backend.initialize() is True
        metadata = backend.get_metadata()
        assert metadata["preflightStatus"] == "passed"
        assert metadata["preflightFrames"] == 1
        assert metadata["backendReady"] is True
    finally:
        backend.close()


def test_pytorch_sequence_error_is_explicit_in_backend_metadata():
    """Temporal failures remain inspectable after the backend returns empty rows."""
    class FailingSequenceAdapter:
        def infer_sequence(self, frames, timestamps, indices):
            del frames, timestamps, indices
            raise RuntimeError("fixture temporal decoder failure")

    backend = PyTorchPoseBackend(kind="wham", device="cpu")
    backend._adapter = FailingSequenceAdapter()
    backend._adapter_name = "failing_sequence_fixture"
    backend._device = "cpu"
    backend.is_initialized = True
    try:
        observations = backend.detect_sequence(
            [np.zeros((8, 8, 3), dtype=np.uint8)],
            [0.0],
            [17],
        )
        assert observations == [None]
        metadata = backend.get_metadata()
        assert "fixture temporal decoder failure" in metadata["sequenceInferenceError"]
        assert metadata["lastError"] == metadata["sequenceInferenceError"]
    finally:
        backend.close()


def test_backend_metadata_keeps_incompatible_assets_and_selected_backend_visible(tmp_path):
    checkpoint = tmp_path / "wham.pth.tar"
    checkpoint.write_bytes(b"fixture")
    adapter = tmp_path / "adapter.py"
    adapter.write_text("def create_backend(config): return object()", encoding="utf-8")
    backend = PyTorchPoseBackend(
        kind="wham",
        model_path=str(checkpoint),
        adapter_module=str(adapter),
        device="cpu",
    )
    try:
        metadata = backend.get_metadata()
        assert metadata["selectedBackend"] == "wham"
        assert isinstance(metadata["missingAssets"], list)
        assert isinstance(metadata["incompatibleAssets"], list)
        assert metadata["assetDiagnostics"]
    finally:
        backend.close()


def test_quality_fallback_keeps_failed_preflight_asset_diagnostics_visible(tmp_path):
    checkpoint = tmp_path / "wham.pth.tar"
    checkpoint.write_bytes(b"fixture")
    adapter = tmp_path / "adapter.py"
    adapter.write_text(
        textwrap.dedent(
            """
            class Adapter:
                def initialize(self):
                    return True
                def infer_frame(self, frame, timestamp, index):
                    raise RuntimeError('fixture fallback smoke failure')
            def create_backend(config):
                return Adapter()
            """
        ),
        encoding="utf-8",
    )

    backend = create_pose_backend(
        preferred="wham",
        pytorch_model_path=str(checkpoint),
        pytorch_adapter_module=str(adapter),
        pytorch_device="cpu",
        strict_quality=False,
    )
    try:
        assert backend.name == "mediapipe"
        metadata = backend.get_metadata()
        assert metadata["preflightStatus"] == "failed"
        assert "backend_short_inference" in metadata["incompatibleAssets"]
        assert metadata["fallbackReason"]
        assert metadata["fallbackBackendMetadata"]["selectedBackend"] == "wham"
    finally:
        backend.close()


def test_quality_factory_preflight_uses_model_directory_asset_resolution(tmp_path):
    """Factory readiness must inspect the same local asset directory as initialize."""
    (tmp_path / "wham_vit_w_3dpw.pth.tar").write_bytes(b"fixture")
    adapter = tmp_path / "adapter.py"
    adapter.write_text(
        textwrap.dedent(
            """
            class Adapter:
                def initialize(self):
                    return True
                def infer_frame(self, frame, timestamp, index):
                    del frame, timestamp
                    return {'frameIndex': index, 'landmarks3d': [[0.0, 0.0, 0.0, 1.0]]}
            def create_backend(config):
                return Adapter()
            """
        ),
        encoding="utf-8",
    )
    backend = create_pose_backend(
        preferred="wham",
        pytorch_adapter_module=str(adapter),
        model_directory=str(tmp_path),
        pytorch_device="cpu",
        strict_quality=True,
    )
    try:
        assert backend.name == "wham"
        assert backend.get_metadata()["backendReady"] is True
    finally:
        backend.close()
