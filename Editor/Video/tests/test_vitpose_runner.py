"""Focused contract tests for the optional inline ViTPose feature runner."""

import builtins
import importlib
import json
import sys
from contextlib import nullcontext
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest


class _FakeTorch:
    """Small torch-shaped seam used to keep this test CPU/profile portable."""

    float32 = np.float32

    def __init__(self):
        self.load_calls = []

    def load(self, path, map_location=None):
        assert map_location == "cpu"
        self.load_calls.append((str(path), map_location))
        return {"state_dict": {"scale": np.asarray(1.0, dtype=np.float32)}}

    @staticmethod
    def from_numpy(value):
        return value

    @staticmethod
    def no_grad():
        return nullcontext()


class _FakeModel:
    def __init__(self, config):
        self.config = dict(config)
        self.loaded = None
        self.eval_called = False
        self.device = None

    def to(self, device):
        self.device = device
        return self

    def eval(self):
        self.eval_called = True
        return self

    def load_state_dict(self, state, strict=True):
        assert strict is True
        if set(state) != {"scale"}:
            raise RuntimeError("missing scale")
        self.loaded = state
        return None

    def __call__(self, batch):
        # The runner must convert BGR input to RGB before this reduction.
        return batch.mean(axis=(2, 3))


def _write_definition(path):
    path.write_text(
        """
import numpy as np

class Model:
    def __init__(self, config):
        self.config = dict(config)
        self.loaded = None
        self.eval_called = False
        self.device = None

    def to(self, device):
        self.device = device
        return self

    def eval(self):
        self.eval_called = True
        return self

    def load_state_dict(self, state, strict=True):
        if strict and set(state) != {'scale'}:
            raise RuntimeError('missing scale')
        self.loaded = state

    def __call__(self, batch):
        return batch.mean(axis=(2, 3))

def create_model(config):
    return Model(config)
""",
        encoding="utf-8",
    )


def test_runner_import_does_not_import_torch(monkeypatch):
    module_name = "pose_pipeline.vitpose_runner"
    sys.modules.pop(module_name, None)
    original_import = builtins.__import__

    def reject_torch(name, *args, **kwargs):
        if name == "torch" or name.startswith("torch."):
            raise AssertionError("vitpose_runner imported torch at module import time")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", reject_torch)
    module = importlib.import_module(module_name)
    assert callable(module.create_vitpose_runner)


def test_runner_loads_hooked_checkpoint_and_returns_deterministic_features(tmp_path):
    from pose_pipeline.vitpose_runner import ViTPoseFeatureRunner

    definition = tmp_path / "vitpose_definition.py"
    _write_definition(definition)
    config_path = tmp_path / "vitpose_config.json"
    config_path.write_text(json.dumps({"feature_dim": 3, "name": "fixture"}), encoding="utf-8")
    checkpoint = tmp_path / "vitpose.pth"
    checkpoint.write_bytes(b"fixture checkpoint")

    runner = ViTPoseFeatureRunner(
        checkpoint_path=checkpoint,
        model_definition=definition,
        config_path=config_path,
        torch_module=_FakeTorch(),
        device="cpu",
        input_size=(2, 2),
        mean=(0.0, 0.0, 0.0),
        std=(1.0, 1.0, 1.0),
    )
    frames = [
        np.asarray([[[0, 0, 255], [0, 0, 255]]], dtype=np.uint8),
        np.asarray([[[0, 255, 0], [255, 0, 0]]], dtype=np.uint8),
    ]

    first = runner.extract_sequence(frames, [0.0, 0.5], [4, 9])
    second = runner(frames, [0.0, 0.5], [4, 9])

    assert first.shape == (2, 3)
    assert first.dtype == np.float32
    assert np.isfinite(first).all()
    np.testing.assert_array_equal(first, second)
    # BGR red becomes RGB [1, 0, 0] after deterministic channel conversion.
    np.testing.assert_allclose(first[0], [1.0, 0.0, 0.0], atol=1e-6)
    assert runner.model.config["feature_dim"] == 3
    assert runner.model.loaded is not None
    assert runner.model.eval_called is True
    assert runner.model.device == "cpu"


def test_runner_accepts_hmr2_checkpoint_as_positional_config(tmp_path):
    """A positional HMR2 ``.ckpt``/``.pth.tar`` is a checkpoint, not config."""
    from pose_pipeline.vitpose_runner import ViTPoseFeatureRunner

    checkpoint = tmp_path / "hmr2.ckpt"
    checkpoint.write_bytes(b"fixture checkpoint")
    runner = ViTPoseFeatureRunner(
        checkpoint,
        model_factory=lambda config: _FakeModel(config),
        torch_module=_FakeTorch(),
    )

    assert runner.checkpoint_path == str(checkpoint)


def test_runner_preserves_frame_order_and_requires_aligned_timestamps(tmp_path):
    """Temporal rows stay attached to their source frame, even for sparse IDs."""
    from pose_pipeline.vitpose_runner import ViTPoseFeatureRunner, ViTPoseRunnerError

    definition = tmp_path / "vitpose_definition.py"
    _write_definition(definition)
    checkpoint = tmp_path / "vitpose.pth"
    checkpoint.write_bytes(b"fixture checkpoint")
    fake_torch = _FakeTorch()
    runner = ViTPoseFeatureRunner(
        checkpoint_path=checkpoint,
        model_definition=definition,
        torch_module=fake_torch,
        device="cpu",
        input_size=(1, 1),
        mean=(0.0, 0.0, 0.0),
        std=(1.0, 1.0, 1.0),
    )
    # The order deliberately differs from the numeric frame IDs.  The model
    # receives rows in decode order; IDs/timestamps must not cause retiming.
    frames = [
        np.asarray([[[0, 0, 255]]], dtype=np.uint8),
        np.asarray([[[255, 0, 0]]], dtype=np.uint8),
        np.asarray([[[0, 255, 0]]], dtype=np.uint8),
    ]
    features = runner.extract_sequence(frames, [2.0, 2.5, 3.0], [101, 107, 110])

    assert features.shape == (3, 3)
    np.testing.assert_allclose(features[0], [1.0, 0.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(features[1], [0.0, 0.0, 1.0], atol=1e-6)
    np.testing.assert_allclose(features[2], [0.0, 1.0, 0.0], atol=1e-6)
    assert fake_torch.load_calls == [(str(checkpoint), "cpu")]

    with pytest.raises(ViTPoseRunnerError, match="aligned"):
        runner.extract_sequence(frames, [2.0, 2.5], [101, 107, 110])


def test_runner_rejects_duplicate_or_non_integer_frame_indices(tmp_path):
    """Source-frame identity must be unique and integer-like before inference."""
    from pose_pipeline.vitpose_runner import ViTPoseFeatureRunner, ViTPoseRunnerError

    checkpoint = tmp_path / "vitpose.pth"
    checkpoint.write_bytes(b"fixture checkpoint")
    runner = ViTPoseFeatureRunner(
        checkpoint_path=checkpoint,
        model_factory=lambda config: _FakeModel(config),
        torch_module=_FakeTorch(),
        input_size=(1, 1),
        mean=(0.0, 0.0, 0.0),
        std=(1.0, 1.0, 1.0),
    )
    frame = np.zeros((1, 1, 3), dtype=np.uint8)

    with pytest.raises(ViTPoseRunnerError, match="duplicate"):
        runner.extract_sequence([frame, frame], [0.0, 0.1], [7, 7])
    assert runner.get_metadata()["errorCode"] == "feature_frame_count_mismatch"
    with pytest.raises(ViTPoseRunnerError, match="integer"):
        runner.extract_sequence([frame], [0.0], [7.5])


def test_runner_rejects_hmr2_pose_outputs_as_image_features(tmp_path):
    """HMR2 3D joints are not interchangeable with WHAM image tokens."""
    from pose_pipeline.vitpose_runner import ViTPoseFeatureRunner, ViTPoseRunnerError

    checkpoint = tmp_path / "hmr2.ckpt"
    checkpoint.write_bytes(b"fixture checkpoint")

    class PoseOnlyModel(_FakeModel):
        def __call__(self, batch, encode=False):
            del encode
            return {"pred_keypoints_3d": np.zeros((len(batch), 17, 3), dtype=np.float32)}

    runner = ViTPoseFeatureRunner(
        {"runtime": "hmr2", "input_size": [1, 1]},
        checkpoint_path=checkpoint,
        model_factory=lambda config: PoseOnlyModel(config),
        torch_module=_FakeTorch(),
        input_size=(1, 1),
        mean=(0.0, 0.0, 0.0),
        std=(1.0, 1.0, 1.0),
    )

    with pytest.raises(ViTPoseRunnerError, match="image feature"):
        runner.extract_sequence(
            [np.zeros((1, 1, 3), dtype=np.uint8)], [0.0], [11],
            frame_bboxes=[[0.5, 0.5, 0.08]],
        )


def test_runner_preflight_and_metadata_separate_hmr2_image_features(tmp_path):
    """Preflight proves one feature row and records the HMR2 runtime identity."""
    from pose_pipeline.vitpose_runner import ViTPoseFeatureRunner

    checkpoint = tmp_path / "hmr2.ckpt"
    checkpoint.write_bytes(b"fixture checkpoint")

    class FeatureModel(_FakeModel):
        feature_dim = 2

        def __call__(self, batch, encode=False):
            assert encode is True
            return np.ones((len(batch), 2), dtype=np.float32)

    runner = ViTPoseFeatureRunner(
        {"runtime": "hmr2", "input_size": [1, 1]},
        checkpoint_path=checkpoint,
        model_factory=lambda config: FeatureModel(config),
        torch_module=_FakeTorch(),
        input_size=(1, 1),
        mean=(0.0, 0.0, 0.0),
        std=(1.0, 1.0, 1.0),
    )

    report = runner.preflight(
        [np.zeros((1, 1, 3), dtype=np.uint8)], [0.0], [101]
    )

    assert report["status"] == "ready"
    assert report["acceptedFrameCount"] == 1
    assert report["featureDim"] == 2
    metadata = runner.get_metadata()
    assert metadata["runtimeKind"] == "hmr2"
    assert metadata["imageFeatureStage"] == "hmr2_image_features"
    assert metadata["checkpointSha256"]


def test_hmr2_factory_requires_the_official_runtime_when_no_local_factory_exists(tmp_path):
    """HMR2 selection must not silently degrade to a generic ViTPose hook."""
    from pose_pipeline.vitpose_runner import create_hmr2_runner

    checkpoint = tmp_path / "hmr2.ckpt"
    checkpoint.write_bytes(b"fixture checkpoint")
    runner = create_hmr2_runner(
        {"runtime": "hmr2"},
        checkpoint_path=checkpoint,
        torch_module=_FakeTorch(),
        device="cpu",
    )

    report = runner.preflight(
        [np.zeros((4, 4, 3), dtype=np.uint8)], [0.0], [1],
        person_bboxes=[[2.0, 2.0, 0.08]],
    )

    assert report["status"] != "ready"
    assert report["errorCode"] in {"runner_runtime_missing", "model_factory_missing"}
    metadata = runner.get_metadata()
    assert metadata["runtimeKind"] == "hmr2"
    assert metadata["runner"] == "official_hmr2"


def test_official_hmr2_runner_uses_vitdet_dataset_and_image_encoder(monkeypatch, tmp_path):
    """The official branch must use ViTDetDataset and ``encode=True`` rows."""
    from pose_pipeline.vitpose_runner import create_hmr2_runner

    checkpoint = tmp_path / "hmr2.ckpt"
    checkpoint.write_bytes(b"fixture checkpoint")
    hmr2 = ModuleType("hmr2")
    models = ModuleType("hmr2.models")
    datasets = ModuleType("hmr2.datasets")
    dataset_module = ModuleType("hmr2.datasets.vitdet_dataset")

    class Model:
        def to(self, device):
            self.device = device
            return self

        def eval(self):
            self.evaluated = True
            return self

        def __call__(self, batch, encode=False):
            assert encode is True
            return {
                "img_feat": np.asarray(
                    [[float(index), float(index) + 0.5] for index, _ in enumerate(batch)],
                    dtype=np.float32,
                )
            }

    class ViTDetDataset:
        def __init__(self, config, frame, boxes):
            del config, boxes
            self.frame = frame

        def __getitem__(self, index):
            assert index == 0
            return {"image": self.frame}

    models.load_hmr2 = lambda checkpoint_path: (Model(), {"official": True})
    dataset_module.ViTDetDataset = ViTDetDataset
    hmr2.models = models
    hmr2.datasets = datasets
    datasets.vitdet_dataset = dataset_module
    for name, module in {
        "hmr2": hmr2,
        "hmr2.models": models,
        "hmr2.datasets": datasets,
        "hmr2.datasets.vitdet_dataset": dataset_module,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)

    runner = create_hmr2_runner(
        {"runtime": "hmr2", "batch_size": 1},
        checkpoint_path=checkpoint,
        torch_module=_FakeTorch(),
    )
    frames = [
        np.zeros((6, 8, 3), dtype=np.uint8),
        np.ones((6, 8, 3), dtype=np.uint8),
    ]
    features = runner.extract_sequence(
        frames,
        [0.0, 0.25],
        [101, 107],
        person_bboxes=[[4.0, 3.0, 0.08], [4.0, 3.0, 0.08]],
    )

    assert features.shape == (2, 2)
    assert np.isfinite(features).all()
    assert runner.get_metadata()["runner"] == "official_hmr2"
    assert runner.get_metadata()["modelFactory"] == "hmr2.models.load_hmr2"


def test_runner_rejects_wrong_feature_dimension_without_truncating(tmp_path):
    """A WHAM feature width mismatch is an actionable error, not silent padding."""
    from pose_pipeline.vitpose_runner import ViTPoseFeatureRunner, ViTPoseRunnerError

    checkpoint = tmp_path / "vitpose.pth"
    checkpoint.write_bytes(b"fixture checkpoint")

    class WrongWidthModel(_FakeModel):
        def __call__(self, batch):
            return np.zeros((len(batch), 2), dtype=np.float32)

    fake_torch = _FakeTorch()
    runner = ViTPoseFeatureRunner(
        checkpoint_path=checkpoint,
        model_factory=lambda config: WrongWidthModel(config),
        torch_module=fake_torch,
        feature_dim=3,
        device="cpu",
    )

    with pytest.raises(ViTPoseRunnerError, match="feature dimension") as error:
        runner.extract_sequence([np.zeros((2, 2, 3), dtype=np.uint8)], [0.0], [5])
    message = str(error.value).lower()
    assert "expected" in message and "3" in message
    assert "2" in message


def test_runner_wraps_forward_failures_with_frame_diagnostics(tmp_path):
    """Model failures identify the inference phase and never look like empty output."""
    from pose_pipeline.vitpose_runner import ViTPoseFeatureRunner, ViTPoseRunnerError

    checkpoint = tmp_path / "vitpose.pth"
    checkpoint.write_bytes(b"fixture checkpoint")

    class ExplodingModel(_FakeModel):
        def __call__(self, batch):
            del batch
            raise RuntimeError("fixture forward exploded")

    runner = ViTPoseFeatureRunner(
        checkpoint_path=checkpoint,
        model_factory=lambda config: ExplodingModel(config),
        torch_module=_FakeTorch(),
        device="cpu",
    )

    with pytest.raises(ViTPoseRunnerError, match="inference") as error:
        runner.extract_sequence(
            [np.zeros((2, 2, 3), dtype=np.uint8)],
            [1.25],
            [73],
        )
    message = str(error.value)
    assert "fixture forward exploded" in message
    assert "73" in message or "frame" in message.lower()


def test_configured_runner_wins_over_local_descriptor_and_preserves_ids(tmp_path):
    """The explicit ViTPose runner is used before the deterministic fallback."""
    from pose_pipeline.adapters.wham_native import NativeWHAMAdapter

    backbone = tmp_path / "vitpose-huge.pth"
    backbone.write_bytes(b"fixture backbone")
    output_dir = tmp_path / "prepared"
    frames = [
        np.full((4, 4, 3), value, dtype=np.uint8)
        for value in (12, 34, 56)
    ]

    class ConfiguredRunner:
        def __init__(self):
            self.calls = []

        def extract_sequence(self, images, timestamps, indices):
            self.calls.append((list(images), list(timestamps), list(indices)))
            return np.asarray(
                [[float(index), float(index) + 0.5, 7.0] for index in indices],
                dtype=np.float32,
            )

    configured = ConfiguredRunner()
    adapter = NativeWHAMAdapter(
        {
            "modelDirectory": str(tmp_path),
            "imageFeatureBackbonePath": str(backbone),
            "imageFeatureRunner": configured,
            "whamPreprocessDirectory": str(output_dir),
        }
    )
    adapter._architecture = {"d_feat": 3}
    adapter._loaded_components["imageFeatureIntegrator"] = True
    frame_indices = [101, 107, 110]
    timestamps = [1.0, 1.2, 1.4]
    adapter._prepare_local_runtime(frames, timestamps, frame_indices)

    assert len(configured.calls) == 1
    np.testing.assert_allclose(configured.calls[0][1], timestamps)
    assert configured.calls[0][2] == frame_indices
    with np.load(adapter._feature_source) as archive:
        np.testing.assert_array_equal(
            archive["features"],
            np.asarray(
                [[101.0, 101.5, 7.0], [107.0, 107.5, 7.0], [110.0, 110.5, 7.0]],
                dtype=np.float32,
            ),
        )
    adapter._image_features_for_sequence(len(frames), frames, timestamps, frame_indices)
    metadata = adapter.get_metadata()
    assert metadata["imageFeatureRunner"] == "vitpose_inline"
    assert metadata["imageFeatureRunnerActive"] is True
    assert metadata["imageFeatureRunnerError"] is None
    assert not any("local descriptor" in item.lower() for item in metadata["runtimeWarnings"])


def test_failed_configured_runner_falls_back_with_provenance(tmp_path):
    """A broken optional runner keeps extraction usable and records why it fell back."""
    from pose_pipeline.adapters.wham_native import NativeWHAMAdapter

    backbone = tmp_path / "vitpose-huge.pth"
    backbone.write_bytes(b"fixture backbone")
    frames = [
        np.full((8, 8, 3), value, dtype=np.uint8)
        for value in (0, 64, 128)
    ]

    adapter = NativeWHAMAdapter(
        {
            "modelDirectory": str(tmp_path),
            "imageFeatureBackbonePath": str(backbone),
            "imageFeatureRunnerModule": "pose_pipeline.missing_vitpose_runner",
            "whamPreprocessDirectory": str(tmp_path / "fallback"),
        }
    )
    adapter._architecture = {"d_feat": 5}
    adapter._loaded_components["imageFeatureIntegrator"] = True
    adapter._prepare_local_runtime(frames, [0.0, 0.1, 0.2], [7, 9, 11])

    adapter._image_features_for_sequence(3, frames, [0.0, 0.1, 0.2], [7, 9, 11])
    metadata = adapter.get_metadata()
    diagnostics = " ".join(metadata["optionalErrors"] + metadata["runtimeWarnings"])
    assert "pose_pipeline.missing_vitpose_runner" in diagnostics
    assert "falling back" in diagnostics.lower() or "fallback" in diagnostics.lower()
    assert metadata["imageFeatureRunner"] is None
    assert metadata["imageFeatureRunnerActive"] is False
    assert "No module named" in metadata["imageFeatureRunnerError"]
    manifest = json.loads(
        (tmp_path / "fallback" / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["featureRunner"] == "local_frame_descriptor"
    assert manifest["featureRunnerStatus"] == "rejected"
    assert manifest["featureRunnerConsumed"] is False
    assert manifest["imageFeatureRunnerModule"] == "pose_pipeline.missing_vitpose_runner"
    assert adapter._feature_source is None
    assert metadata["imageFeatureFallback"] == "local_frame_descriptor"
    assert metadata["imageFeatureFallbackConsumed"] is False


def test_runner_reports_model_definition_and_checkpoint_incompatibility(tmp_path):
    from pose_pipeline.vitpose_runner import ViTPoseFeatureRunner, ViTPoseRunnerError

    checkpoint = tmp_path / "bad.pth"
    checkpoint.write_bytes(b"fixture checkpoint")
    with pytest.raises(ViTPoseRunnerError, match="model_definition"):
        ViTPoseFeatureRunner(
            checkpoint_path=checkpoint,
            model_definition=tmp_path / "missing_definition.py",
            torch_module=_FakeTorch(),
        ).initialize()


def test_runner_refuses_ambiguous_checkpoint_before_loading_weights(tmp_path):
    """A raw ViTPose backbone must not be loaded without model code/config."""
    from pose_pipeline.vitpose_runner import ViTPoseFeatureRunner, ViTPoseRunnerError

    checkpoint = tmp_path / "vitpose-huge.pth"
    checkpoint.write_bytes(b"fixture checkpoint")
    fake_torch = _FakeTorch()

    with pytest.raises(ViTPoseRunnerError, match="model-definition hook"):
        ViTPoseFeatureRunner(
            checkpoint_path=checkpoint,
            torch_module=fake_torch,
        ).initialize()
    assert fake_torch.load_calls == []

    class IncompatibleModel(_FakeModel):
        def load_state_dict(self, state, strict=True):
            raise RuntimeError("unexpected key from incompatible checkpoint")

    with pytest.raises(ViTPoseRunnerError, match="checkpoint"):
        ViTPoseFeatureRunner(
            checkpoint_path=checkpoint,
            model_factory=lambda config: IncompatibleModel(config),
            torch_module=_FakeTorch(),
        ).initialize()


def test_bundled_model_definition_is_rejected_as_a_placeholder(tmp_path):
    """The shipped template must not be mistaken for an HMR2 architecture."""
    from pose_pipeline.vitpose_runner import ViTPoseFeatureRunner, ViTPoseRunnerError

    checkpoint = tmp_path / "vitpose.pth"
    checkpoint.write_bytes(b"fixture checkpoint")
    placeholder = (
        Path(__file__).resolve().parents[1]
        / "models"
        / "wham_vitpose_model_definition.py"
    )

    fake_torch = _FakeTorch()
    with pytest.raises(ViTPoseRunnerError, match="not an architecture"):
        ViTPoseFeatureRunner(
            checkpoint_path=checkpoint,
            model_definition=placeholder,
            torch_module=fake_torch,
        ).initialize()
    assert fake_torch.load_calls == []


def test_hmr2_runner_uses_person_crop_encode_and_reports_adopted_frames(tmp_path):
    """The WHAM feature contract must call an HMR2 model on bbox crops."""
    from pose_pipeline.vitpose_runner import ViTPoseFeatureRunner

    checkpoint = tmp_path / "hmr2.ckpt"
    checkpoint.write_bytes(b"fixture checkpoint")

    class HMR2Model(_FakeModel):
        feature_dim = 3

        def __init__(self, config):
            super().__init__(config)
            self.calls = []

        def __call__(self, batch, encode=False):
            self.calls.append((np.asarray(batch).shape, encode))
            assert encode is True
            return np.full((len(batch), 3), 0.75, dtype=np.float32)

    model = HMR2Model({})
    runner = ViTPoseFeatureRunner(
        {
            "runtime": "hmr2",
            "feature_dim": 3,
            "input_size": [4, 4],
            "mean": [0.0, 0.0, 0.0],
            "std": [1.0, 1.0, 1.0],
        },
        checkpoint_path=checkpoint,
        model_factory=lambda config: model,
        torch_module=_FakeTorch(),
        device="cpu",
    )
    frames = [np.full((20, 12, 3), 80, dtype=np.uint8), np.full((20, 12, 3), 120, dtype=np.uint8)]
    boxes = [[6.0, 10.0, 0.08], None]

    features = runner.extract_sequence(
        frames,
        timestamps=[0.0, 0.1],
        frame_indices=[101, 107],
        frame_bboxes=boxes,
    )

    assert features.shape == (2, 3)
    np.testing.assert_allclose(features[0], [0.75, 0.75, 0.75])
    np.testing.assert_allclose(features[1], [0.0, 0.0, 0.0])
    assert model.calls == [((1, 3, 4, 4), True)]
    metadata = runner.get_metadata()
    assert metadata["runtime"] == "hmr2"
    assert metadata["adoptedFrameIndices"] == [101]
    assert metadata["adoptedFrameCount"] == 1
    assert metadata["featureDim"] == 3


def test_hmr2_factory_can_own_checkpoint_loading_and_receives_checkpoint_path(tmp_path):
    """The official WHAM ``hmr2(checkpoint_path)`` seam must stay usable.

    WHAM's reference feature model loads its own checkpoint in the factory and
    exposes ``model(norm_img, encode=True)``.  The adapter must not force a
    second raw checkpoint load or pass the runner config where the path is
    expected.
    """
    from pose_pipeline.vitpose_runner import ViTPoseFeatureRunner

    checkpoint = tmp_path / "hmr2.ckpt"
    checkpoint.write_bytes(b"factory-owned checkpoint")
    calls = []

    class FactoryOwnedModel:
        runner_kind = "hmr2"
        feature_dim = 2

        def to(self, device):
            assert device == "cpu"
            return self

        def eval(self):
            return self

        def __call__(self, batch, encode=False):
            assert encode is True
            assert np.asarray(batch).shape == (1, 3, 256, 256)
            return np.ones((1, 2), dtype=np.float32)

    def factory(checkpoint_pth):
        calls.append(checkpoint_pth)
        return FactoryOwnedModel()

    fake_torch = _FakeTorch()
    runner = ViTPoseFeatureRunner(
        {"runtime": "hmr2", "factory_loads_checkpoint": True},
        checkpoint_path=checkpoint,
        model_factory=factory,
        torch_module=fake_torch,
        device="cpu",
        mean=(0.0, 0.0, 0.0),
        std=(1.0, 1.0, 1.0),
    )

    features = runner.extract_sequence(
        [np.zeros((12, 16, 3), dtype=np.uint8)], [0.0], [42],
        frame_bboxes=[[8.0, 6.0, 0.08]],
    )

    assert calls == [str(checkpoint.resolve())]
    assert fake_torch.load_calls == []
    np.testing.assert_array_equal(features, np.ones((1, 2), dtype=np.float32))


def test_hmr2_factory_tuple_return_keeps_official_model_and_config(tmp_path):
    """Factories modeled after ``load_hmr2`` may return ``(model, cfg)``."""
    from pose_pipeline.vitpose_runner import ViTPoseFeatureRunner

    checkpoint = tmp_path / "hmr2.ckpt"
    checkpoint.write_bytes(b"factory-owned checkpoint")

    class TupleModel:
        runner_kind = "hmr2"
        feature_dim = 1

        def to(self, device):
            return self

        def eval(self):
            return self

        def __call__(self, batch, encode=False):
            assert encode is True
            return np.ones((len(batch), 1), dtype=np.float32)

    def factory(checkpoint_pth):
        assert checkpoint_pth == str(checkpoint.resolve())
        return TupleModel(), {"official_cfg": True}

    runner = ViTPoseFeatureRunner(
        {"runtime": "hmr2", "factory_loads_checkpoint": True},
        checkpoint_path=checkpoint,
        model_factory=factory,
        torch_module=_FakeTorch(),
    )
    features = runner.extract_sequence(
        [np.zeros((8, 8, 3), dtype=np.uint8)], [0.0], [1],
        frame_bboxes=[[4.0, 4.0, 0.08]],
    )

    np.testing.assert_array_equal(features, np.ones((1, 1), dtype=np.float32))
    assert runner.model_config["official_cfg"] is True


def test_missing_inline_runner_never_feeds_local_descriptors_to_integrator(tmp_path):
    """A local descriptor is an explicit fallback, never a learned feature."""
    from pose_pipeline.adapters.wham_native import NativeWHAMAdapter

    backbone = tmp_path / "vitpose-huge.pth"
    backbone.write_bytes(b"fixture backbone")
    adapter = NativeWHAMAdapter(
        {
            "modelDirectory": str(tmp_path),
            "imageFeatureBackbonePath": str(backbone),
            "imageFeatureRunnerModule": "pose_pipeline.no_such_runner",
            "whamPreprocessDirectory": str(tmp_path / "fallback"),
        }
    )
    adapter._architecture = {"d_feat": 3, "d_embed": 2, "n_joints": 1}
    adapter._loaded_components["imageFeatureIntegrator"] = True
    adapter._model = SimpleNamespace(
        integrator=SimpleNamespace(layer1=SimpleNamespace(in_features=8))
    )
    frames = [np.full((8, 8, 3), value, dtype=np.uint8) for value in (16, 64)]

    adapter._prepare_local_runtime(frames, [0.0, 0.1], [101, 107])

    assert adapter._feature_fallback_active is True
    assert adapter._feature_fallback_kind == "local_frame_descriptor"
    assert adapter._image_features_for_sequence(2, frames, [0.0, 0.1], [101, 107]) is None
    metadata = adapter.get_metadata()
    assert metadata["imageFeatureFallback"] == "local_frame_descriptor"
    assert metadata["imageFeatureFallbackConsumed"] is False


def test_incompatible_configured_runner_is_rejected_without_archive_injection(tmp_path):
    """A runner output-width mismatch must not become a local learned archive."""
    from pose_pipeline.adapters.wham_native import NativeWHAMAdapter

    backbone = tmp_path / "vitpose-huge.pth"
    backbone.write_bytes(b"fixture backbone")

    class WrongWidthRunner:
        def extract_sequence(self, images, timestamps, indices, **kwargs):
            del timestamps, indices, kwargs
            return np.zeros((len(images), 2), dtype=np.float32)

    adapter = NativeWHAMAdapter(
        {
            "modelDirectory": str(tmp_path),
            "imageFeatureBackbonePath": str(backbone),
            "imageFeatureRunner": WrongWidthRunner(),
            "whamPreprocessDirectory": str(tmp_path / "fallback"),
        }
    )
    adapter._architecture = {"d_feat": 3}
    adapter._loaded_components["imageFeatureIntegrator"] = True
    frames = [np.zeros((8, 8, 3), dtype=np.uint8) for _ in range(2)]

    adapter._prepare_local_runtime(frames, [0.0, 0.1], [11, 17])

    assert adapter._feature_source is None
    assert adapter._feature_fallback_active is True
    assert adapter._image_features_for_sequence(2, frames, [0.0, 0.1], [11, 17]) is None
    metadata = adapter.get_metadata()
    assert metadata["imageFeatureRunnerStatus"] == "fallback"
    assert metadata["imageFeatureFallbackConsumed"] is False
    assert "incompatible" in metadata["imageFeatureFallbackReason"].lower()


def test_native_adapter_passes_hmr2_person_crops_and_runner_metadata(tmp_path):
    """Native preprocessing forwards tracked boxes instead of encoding full frames."""
    from pose_pipeline.adapters.wham_native import NativeWHAMAdapter

    backbone = tmp_path / "vitpose-huge.pth"
    backbone.write_bytes(b"fixture backbone")
    frames = [np.full((12, 16, 3), value, dtype=np.uint8) for value in (32, 64)]
    frame_indices = [101, 107]
    timestamps = [0.0, 0.1]

    class HMR2Runner:
        person_crop = True
        runner_kind = "hmr2"

        def __init__(self):
            self.calls = []

        def extract_sequence(self, images, times, indices, *, frame_bboxes=None):
            self.calls.append((list(images), list(times), list(indices), list(frame_bboxes)))
            assert frame_bboxes[0]["format"] == "cxcys"
            assert frame_bboxes[1] is None
            return np.ones((len(images), 3), dtype=np.float32)

        def get_metadata(self):
            return {
                "runtime": "hmr2",
                "runnerKind": "hmr2",
                "personCrop": True,
                "encode": True,
                "featureDim": 3,
                "adoptedFrameIndices": [frame_indices[0]],
            }

    runner = HMR2Runner()
    adapter = NativeWHAMAdapter(
        {
            "modelDirectory": str(tmp_path),
            "imageFeatureBackbonePath": str(backbone),
            "imageFeatureRunner": runner,
            "whamPreprocessDirectory": str(tmp_path / "prepared"),
        }
    )
    adapter._architecture = {"d_feat": 3}
    adapter._loaded_components["imageFeatureIntegrator"] = True
    boxes = [
        {"cx": 8.0, "cy": 6.0, "scale": 0.08, "format": "cxcys"},
        None,
    ]

    adapter._prepare_local_runtime(
        frames, timestamps, frame_indices, frame_bboxes=boxes
    )

    assert len(runner.calls) == 1
    assert runner.calls[0][2] == frame_indices
    assert runner.calls[0][3] == boxes
    metadata = adapter.get_metadata()
    assert metadata["imageFeatureRunnerMetadata"]["runtime"] == "hmr2"
    assert metadata["imageFeatureAdoptedFrameIndices"] == [frame_indices[0]]
    assert metadata["imageFeatureAdoptedFrameCount"] == 1
    manifest = json.loads(
        (tmp_path / "prepared" / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["featureRunnerMetadata"]["runtime"] == "hmr2"
    assert manifest["featureAdoptedFrameIndices"] == [frame_indices[0]]
