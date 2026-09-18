import json

import numpy as np
import pytest

from pose_pipeline.wham_preprocess import (
    build_local_image_features,
    estimate_local_camera_poses,
    extract_video_image_features,
    export_vitpose_features_from_video,
    export_camera_poses,
    run_wham_preprocess,
)
from pose_pipeline.adapters.wham_native import NativeWHAMAdapter


def test_one_click_vitpose_export_writes_aligned_archive(tmp_path):
    cv2 = pytest.importorskip("cv2")
    video_path = tmp_path / "source.avi"
    writer = cv2.VideoWriter(
        str(video_path), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (8, 8)
    )
    if not writer.isOpened():
        pytest.skip("OpenCV test video codec is unavailable")
    try:
        for value in range(6):
            writer.write(np.full((8, 8, 3), value * 20, dtype=np.uint8))
    finally:
        writer.release()

    class FakeRunner:
        def __init__(self):
            self.calls = []

        def extract_sequence(self, frames, timestamps, indices):
            self.calls.append((len(frames), list(timestamps), list(indices)))
            return np.asarray(
                [[float(index), float(index) + 0.5] for index in indices],
                dtype=np.float32,
            )

        def get_metadata(self):
            return {"runner": "fixture"}

    runner = FakeRunner()
    result = export_vitpose_features_from_video(
        video_path,
        tmp_path / "vitpose_features.npz",
        checkpoint_path="fixture.pth",
        target_fps=5.0,
        runner=runner,
        chunk_size=2,
    )

    assert result["frames"] == 3
    assert result["featureShape"] == [3, 2]
    assert [call[0] for call in runner.calls] == [2, 1]
    with np.load(result["path"]) as archive:
        assert archive["features"].shape == (3, 2)
        assert archive["timestamps"].shape == (3,)
        assert archive["frame_indices"].tolist() == [0, 2, 4]
        assert archive["source_video"].item() == str(video_path.resolve())


def test_per_video_runner_writes_aligned_feature_and_camera_archives(tmp_path):
    frames = [np.zeros((4, 5, 3), dtype=np.uint8) for _ in range(3)]

    def features(images, timestamps, indices):
        return np.asarray([[float(i), float(timestamps[i])] for i in indices])

    def camera(images, timestamps, indices):
        return np.tile(np.array([[0, 0, 0, 0, 0, 0, 1]], dtype=np.float32), (len(images), 1))

    result = run_wham_preprocess(
        frames,
        tmp_path,
        image_feature_extractor=features,
        camera_pose_extractor=camera,
        timestamps=[0.0, 0.5, 1.0],
    )
    with np.load(result["imageFeaturePath"]) as archive:
        assert archive["features"].shape == (3, 2)
        assert archive["timestamps"].tolist() == [0.0, 0.5, 1.0]
    with np.load(result["cameraPosePath"]) as archive:
        assert archive["cameraPoses"].shape == (3, 7)
    assert json.loads((tmp_path / "manifest.json").read_text())["frames"] == 3


def test_runner_rejects_misaligned_camera_rows(tmp_path):
    with pytest.raises(ValueError, match="camera poses"):
        export_camera_poses(np.zeros((2, 7)), tmp_path / "camera.npz", 3)


def test_feature_preprocess_rejects_integer_rows_instead_of_coercing_them():
    frames = [np.zeros((4, 5, 3), dtype=np.uint8)]

    with pytest.raises(ValueError, match="floating-point"):
        extract_video_image_features(
            frames,
            lambda images, timestamps, indices: np.ones((len(images), 2), dtype=np.int32),
        )


def test_native_adapter_auto_prepares_downloaded_backbone_and_dpvo_assets(tmp_path):
    frames = [
        np.full((32, 32, 3), value, dtype=np.uint8)
        for value in (0, 40, 80, 120)
    ]
    backbone = tmp_path / "vitpose-huge.pth"
    dpvo = tmp_path / "dpvo.pth"
    camera = tmp_path / "camera.yaml"
    backbone.write_bytes(b"backbone")
    dpvo.write_bytes(b"dpvo")
    camera.write_text("fx: 32\nfy: 32\ncx: 16\ncy: 16\n", encoding="utf-8")

    adapter = NativeWHAMAdapter(
        {
            "modelDirectory": str(tmp_path),
            "imageFeatureBackbonePath": str(backbone),
            "cameraModelPath": str(camera),
            "dpvoModelPath": str(dpvo),
            "whamPreprocessDirectory": str(tmp_path / "prepared"),
        }
    )
    adapter._architecture = {"d_feat": 24}
    adapter._loaded_components["imageFeatureIntegrator"] = True
    adapter._loaded_components["trajectoryDecoder"] = True
    adapter._prepare_local_runtime(frames, [0.0, 0.1, 0.2, 0.3], [0, 1, 2, 3])

    # The raw ViTPose checkpoint has no verified extractor, so the native
    # adapter must not synthesize descriptors for WHAM's learned integrator.
    assert adapter._auto_feature_preprocess_used is False
    assert adapter._auto_camera_preprocess_used is True
    assert adapter._feature_source is None
    assert adapter._camera_source is not None
    with np.load(adapter._camera_source) as archive:
        assert archive["cameraPoses"].shape == (4, 7)
    # The generated paths must be consumed by the same loaders used for
    # caller-supplied archives; otherwise a successful preprocessing pass
    # would still leave WHAM on zero camera motion or an empty feature input.
    camera_motion = adapter._camera_for_sequence(4)
    assert camera_motion.shape == (4, 6)
    assert adapter._camera_motion_source == "local_optical_flow"
    assert adapter._image_features_for_sequence(
        4, frames, [0.0, 0.1, 0.2, 0.3], [0, 1, 2, 3]
    ) is None
    metadata = adapter.get_metadata()
    assert metadata["whamPreprocessManifestPath"]
    assert metadata["runtimeWarnings"]
    assert metadata["imageFeatureFallback"] == "local_frame_descriptor"
    assert metadata["imageFeatureFallbackConsumed"] is False


def test_native_adapter_labels_local_optical_flow_as_non_dpvo(tmp_path):
    """Downloaded DPVO weights alone must not be reported as true DPVO motion."""
    frames = [
        np.full((32, 32, 3), value, dtype=np.uint8)
        for value in (0, 40, 80)
    ]
    backbone = tmp_path / "vitpose-huge.pth"
    dpvo = tmp_path / "dpvo.pth"
    backbone.write_bytes(b"backbone")
    dpvo.write_bytes(b"dpvo")
    adapter = NativeWHAMAdapter(
        {
            "modelDirectory": str(tmp_path),
            "imageFeatureBackbonePath": str(backbone),
            "dpvoModelPath": str(dpvo),
            "whamPreprocessDirectory": str(tmp_path / "prepared"),
        }
    )
    adapter._architecture = {"d_feat": 8}
    adapter._loaded_components["imageFeatureIntegrator"] = True
    adapter._loaded_components["trajectoryDecoder"] = True

    adapter._prepare_local_runtime(frames, [0.0, 0.1, 0.2], [0, 1, 2])
    adapter._camera_for_sequence(3)

    metadata = adapter.get_metadata()
    assert metadata["cameraMotionSource"] == "local_optical_flow"
    assert metadata["cameraMotionProvenance"]["source"] == "local_optical_flow"
    assert metadata["cameraMotionProvenance"]["verified"] is False
    assert metadata["dpvoVerified"] is False
    assert any("optical-flow" in item.lower() for item in metadata["runtimeWarnings"])
