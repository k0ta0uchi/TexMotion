import json

import numpy as np
import pytest

from pose_pipeline.wham_preprocess import (
    extract_video_image_features,
    export_camera_poses,
    run_wham_preprocess,
)


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
