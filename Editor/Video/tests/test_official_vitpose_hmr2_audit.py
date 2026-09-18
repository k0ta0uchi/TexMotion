"""Adversarial contract probes for the official ViTPose/HMR2 runner spec.

These tests deliberately stay at the runner/preprocess/metadata seams.  They
do not download research weights and they do not modify the production
pipeline; a real official-runtime acceptance run remains an explicit
unverified item in the audit report.
"""

from __future__ import annotations

import textwrap

import numpy as np
import pytest

from pose_pipeline.adapters.wham_native import (
    NativeWHAMAdapter,
    _UnverifiedFeatureSourceError,
    _coerce_image_features,
)
from pose_pipeline.backends.pytorch_backend import PyTorchPoseBackend
from pose_pipeline.wham_preprocess import (
    export_vitpose_features_from_video,
    extract_video_image_features,
)
from video_pose_extractor import build_runtime_provenance


def test_raw_checkpoint_mapping_is_rejected_as_feature_archive() -> None:
    """FR-03/FR-05: state dictionaries cannot masquerade as image features."""

    raw_checkpoint = {
        "state_dict": {
            "encoder.weight": np.zeros((2, 2), dtype=np.float32),
        }
    }

    with pytest.raises(_UnverifiedFeatureSourceError, match="raw model checkpoint"):
        _coerce_image_features(raw_checkpoint, frame_count=1, expected_dim=2)


def test_feature_frame_count_contract_rejects_missing_rows() -> None:
    """FR-05: every requested input frame needs one feature row."""

    frames = [np.zeros((4, 4, 3), dtype=np.uint8) for _ in range(2)]

    with pytest.raises(ValueError, match=r"shape \(2, D\)"):
        extract_video_image_features(
            frames,
            lambda images, timestamps, indices: np.zeros((len(images) - 1, 3), dtype=np.float32),
            timestamps=[0.0, 0.1],
            frame_indices=[101, 107],
        )


def test_feature_dimension_and_finite_contracts_are_checked() -> None:
    """FR-05: wrong width and NaN/Inf outputs are rejected before WHAM."""

    finite_wrong_width = np.zeros((2, 2), dtype=np.float32)
    with pytest.raises(ValueError, match="require dimension 3"):
        _coerce_image_features(finite_wrong_width, frame_count=2, expected_dim=3)

    non_finite = np.zeros((2, 3), dtype=np.float32)
    non_finite[1, 2] = np.inf
    with pytest.raises(ValueError, match="non-finite"):
        extract_video_image_features(
            [np.zeros((4, 4, 3), dtype=np.uint8) for _ in range(2)],
            lambda images, timestamps, indices: non_finite,
            timestamps=[0.0, 0.1],
            frame_indices=[101, 107],
        )


def test_quality_preflight_metadata_never_reports_ready_after_model_load_failure(tmp_path) -> None:
    """FR-07: an unavailable WHAM asset is a failed preflight, not readiness."""

    backend = PyTorchPoseBackend(
        kind="wham",
        model_path=str(tmp_path / "missing-wham.pth.tar"),
        device="cpu",
    )
    try:
        assert backend.initialize() is False
        metadata = backend.get_metadata()
        assert metadata["preflightStatus"] == "failed"
        assert metadata["backendReady"] is False
        assert metadata["preflightError"]
        assert metadata["preflightPhase"] in {"availability", "model_load"}
    finally:
        backend.close()


def test_quality_preflight_ready_requires_one_frame_smoke_inference(tmp_path) -> None:
    """FR-07: model construction alone cannot produce a ready backend."""

    pytest.importorskip("torch")
    checkpoint = tmp_path / "wham.pth.tar"
    checkpoint.write_bytes(b"fixture")
    adapter_file = tmp_path / "preflight_adapter.py"
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
                del config
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
        assert metadata["preflightPhase"] == "ready"
        assert metadata["preflightFrames"] == 1
        assert metadata["backendReady"] is True
    finally:
        backend.close()


def test_runner_failure_keeps_an_actionable_fallback_reason_and_rejects_descriptor(tmp_path) -> None:
    """FR-09/FR-11: runner failure is visible and never becomes learned input."""

    backbone = tmp_path / "vitpose-huge.pth"
    backbone.write_bytes(b"raw checkpoint")
    adapter = NativeWHAMAdapter(
        {
            "modelDirectory": str(tmp_path),
            "imageFeatureBackbonePath": str(backbone),
            "imageFeatureRunnerModule": "pose_pipeline.no_such_official_runner",
            "whamPreprocessDirectory": str(tmp_path / "prepared"),
        }
    )
    adapter._architecture = {"d_feat": 3}
    adapter._loaded_components["imageFeatureIntegrator"] = True
    frames = [np.zeros((8, 8, 3), dtype=np.uint8) for _ in range(2)]

    adapter._prepare_local_runtime(frames, [0.0, 0.1], [101, 107])
    assert adapter._image_features_for_sequence(
        2, frames, [0.0, 0.1], [101, 107]
    ) is None

    metadata = adapter.get_metadata()
    assert metadata["imageFeatureFallback"] == "local_frame_descriptor"
    assert metadata["imageFeatureFallbackConsumed"] is False
    assert metadata["imageFeatureFallbackReason"]
    assert "not fed" in metadata["imageFeatureFallbackReason"].lower()
    assert metadata["imageFeatureRunnerStatus"] == "fallback"


def test_hmr2_vitpose_and_mediapipe_provenance_stay_separate() -> None:
    """FR-06/FR-10: HMR2 image tokens do not imply ViTPose-2D or MediaPipe use."""

    provenance = build_runtime_provenance(
        "wham",
        "wham",
        {
            "nativeWhamCore": True,
            "backendReady": True,
            "inputDetector": "mediapipe",
            "imageFeatureRunner": "official_hmr2",
            "imageFeatureRunnerRuntime": "hmr2",
            "imageFeatureRunnerStatus": "active",
            "imageFeatureVerified": True,
            "imageFeaturesUsed": True,
        },
    )

    assert provenance["officialRunnerStatus"] == "active"
    assert provenance["hmr2ImageFeaturesStatus"] == "active"
    assert provenance["vitpose2DStatus"] == "not_configured"
    assert provenance["fallbackReason"] is None


def test_feature_metadata_distinguishes_available_archive_from_integrator_use(tmp_path) -> None:
    """FR-10: feature availability is not evidence that WHAM consumed it."""

    feature_path = tmp_path / "features.npy"
    np.save(feature_path, np.ones((2, 3), dtype=np.float32))
    adapter = NativeWHAMAdapter({"imageFeaturesPath": str(feature_path)})
    adapter._architecture = {"d_feat": 3, "d_embed": 2, "n_joints": 1}
    adapter._loaded_components["imageFeatureIntegrator"] = True
    adapter._model = type(
        "Model",
        (),
        {"integrator": type("Integrator", (), {"layer1": type("Layer", (), {"in_features": 8})()})()},
    )()
    frames = [np.zeros((8, 8, 3), dtype=np.uint8) for _ in range(2)]

    values = adapter._image_features_for_sequence(
        2, frames, [0.0, 0.1], [101, 107]
    )
    assert values.shape == (2, 3)
    available = adapter.get_metadata()
    assert available["imageFeatureSource"] == "configured_not_used"
    assert available["imageFeatureSourceProvenance"] == "verified_archive"
    assert available["imageFeatureVerified"] is False
    assert available["officialImageFeatureIntegrator"] is False

    # This flag is set by the native temporal path only after calling the
    # WHAM integrator.  The metadata should then expose actual use evidence.
    adapter._image_features_used = True
    consumed = adapter.get_metadata()
    assert consumed["imageFeatureSource"] == "archive"
    assert consumed["imageFeatureVerified"] is True
    assert consumed["officialImageFeatureIntegrator"] is True


def test_exporter_closes_runner_when_feature_contract_fails(tmp_path) -> None:
    """FR-01: extraction cleanup runs even when a feature batch is rejected."""

    cv2 = pytest.importorskip("cv2")
    video_path = tmp_path / "source.avi"
    writer = cv2.VideoWriter(
        str(video_path), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (8, 8)
    )
    if not writer.isOpened():
        pytest.skip("OpenCV test video codec is unavailable")
    try:
        writer.write(np.zeros((8, 8, 3), dtype=np.uint8))
    finally:
        writer.release()

    class FailingRunner:
        def __init__(self):
            self.closed = False

        def extract_sequence(self, frames, timestamps, indices):
            del timestamps, indices
            return np.zeros((len(frames) - 1, 2), dtype=np.float32)

        def close(self):
            self.closed = True

    runner = FailingRunner()
    with pytest.raises(ValueError, match=r"expected \(1, D\)"):
        export_vitpose_features_from_video(
            video_path,
            tmp_path / "features.npz",
            checkpoint_path="fixture.pth",
            runner=runner,
        )
    assert runner.closed is True
