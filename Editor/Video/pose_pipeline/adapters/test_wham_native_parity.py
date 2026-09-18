"""Deterministic coverage for the offline WHAM parity pieces.

These tests intentionally stay inside the adapter package so the optional
research stages can be exercised without a detector, SMPL installation, or a
CUDA runtime.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pose_pipeline.adapters.wham_native import (  # noqa: E402
    EmbeddedSMPL,
    NativeWHAMAdapter,
    NativeWhamNetwork,
    _camera_angular_velocity_from_poses,
    _coerce_smpl_initialization,
)
from pose_pipeline.observations import FrameObservations, Keypoint2D  # noqa: E402


def test_official_decoder_modules_keep_wham_checkpoint_names():
    network = NativeWhamNetwork(
        d_embed=8,
        d_feat=4,
        n_layers=2,
        enable_decoders=True,
    )

    state_keys = set(network.state_dict())
    assert "trajectory_decoder.regressor.declayer1.weight" in state_keys
    assert "integrator.layer3.bias" in state_keys
    assert "motion_decoder.neural_init.linear3.weight" in state_keys
    assert "motion_decoder.regressor.declayer3.bias" in state_keys
    assert "trajectory_refiner.refiner.declayer1.weight" in state_keys


def test_smpl_initialization_root_centers_joints_and_preserves_pose():
    joints = np.zeros((17, 3), dtype=np.float32)
    joints[11] = [-0.2, 1.0, 0.3]
    joints[12] = [0.2, 1.0, 0.3]
    full_pose = np.tile(np.eye(3, dtype=np.float32), (24, 1, 1))

    init_kp3d, init_smpl6d, source = _coerce_smpl_initialization(
        {"joints": joints, "full_pose": full_pose}
    )

    assert source == "configured"
    assert init_kp3d.shape == (17, 3)
    assert init_smpl6d.shape == (24, 6)
    assert init_kp3d[11].tolist() == pytest.approx([-0.2, 0.0, 0.0])
    assert init_kp3d[12].tolist() == pytest.approx([0.2, 0.0, 0.0])
    assert init_smpl6d[0].tolist() == pytest.approx([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])


def test_camera_angular_velocity_matches_official_shape_and_padding():
    # DPVO stores [tx, ty, tz, qx, qy, qz, qw].  Two identity poses are a
    # deterministic local-only camera stream and must produce zero motion.
    poses = np.zeros((3, 7), dtype=np.float32)
    poses[:, 6] = 1.0

    angular_velocity = _camera_angular_velocity_from_poses(poses, fps=30.0)

    assert angular_velocity.shape == (3, 6)
    assert np.isfinite(angular_velocity).all()
    np.testing.assert_allclose(angular_velocity, np.zeros((3, 6), dtype=np.float32))


def test_metadata_exposes_optional_stage_capabilities_before_checkpoint_load():
    adapter = NativeWHAMAdapter({"torch": torch, "device": "cpu"})

    metadata = adapter.get_metadata()

    assert "capabilities" in metadata
    assert metadata["capabilities"]["nativeMotionEncoder"] is False
    assert metadata["capabilities"]["smplDerivedInitialization"] is False
    assert "missingOptionalAssets" in metadata
    assert "camera_motion_or_slam" in metadata["missingOptionalAssets"]


def test_mediapipe_tuple_observation_can_seed_native_wham():
    """The extractor's lightweight MediaPipe tuple must reach WHAM's seed."""
    adapter = NativeWHAMAdapter({"torch": torch, "device": "cpu"})
    world = np.full((33, 3), np.nan, dtype=np.float32)
    # COCO/MediaPipe hip, shoulder, and limb slots used by the seed seam.
    for index, value in {
        23: (-0.2, 0.0, 0.0),
        24: (0.2, 0.0, 0.0),
        11: (-0.2, -0.45, 0.0),
        12: (0.2, -0.45, 0.0),
        25: (-0.2, 0.35, 0.0),
        26: (0.2, 0.35, 0.0),
    }.items():
        world[index] = value

    assert adapter.set_initialization_from_observation((world, None, 0.9)) is True
    assert adapter._smpl_seed is not None
    assert adapter._smpl_seed[2] == "mediapipe_observation"
    np.testing.assert_allclose(adapter._smpl_seed[0][11], [-0.2, 0.0, 0.0])
    np.testing.assert_allclose(adapter._smpl_seed[0][12], [0.2, 0.0, 0.0])


class _Detector:
    is_initialized = True

    def detect(self, frame, timestamp_sec, frame_index=0):
        keypoints = [Keypoint2D(0.0, 0.0, 0.0) for _ in range(33)]
        for index in (11, 12, 23, 24, 27, 28):
            keypoints[index] = Keypoint2D(0.4 + index / 1000.0, 0.5, 0.9)
        return FrameObservations(frame_index, timestamp_sec, keypoints, [])

    def close(self):
        pass


def test_full_checkpoint_named_decoder_rolls_out_optional_stages(tmp_path):
    checkpoint = tmp_path / "full_wham_fixture.pth.tar"
    network = NativeWhamNetwork(
        d_embed=8,
        d_feat=4,
        n_layers=2,
        enable_decoders=True,
    )
    torch.save({"state_dict": network.state_dict()}, checkpoint)
    adapter = NativeWHAMAdapter(
        {
            "modelPath": str(checkpoint),
            "device": "cpu",
            "torch": torch,
            "detector": _Detector(),
            "d_feat": 4,
            "imageFeatures": np.ones((2, 4), dtype=np.float32),
            "cameraPoses": np.tile(np.array([[0, 0, 0, 0, 0, 0, 1]], dtype=np.float32), (2, 1)),
        }
    )

    assert adapter.initialize() is True, adapter.error
    try:
        result = adapter.infer_sequence(
            [np.zeros((32, 32, 3), dtype=np.uint8) for _ in range(2)],
            [0.0, 1 / 30],
            [0, 1],
        )
        assert len(result["frames"]) == 2
        assert "trajectory" in result
        assert adapter.get_metadata()["capabilities"]["imageFeatureIntegrator"] is True
        assert adapter.get_metadata()["capabilities"]["trajectoryDecoder"] is True
        assert adapter.get_metadata()["capabilities"]["smplDecoder"] is False
    finally:
        adapter.close()


def test_embedded_smpl_buffers_provide_neutral_seed_without_smplx():
    vertices = 24
    regressor = torch.zeros((24, vertices), dtype=torch.float32)
    regressor[torch.arange(24), torch.arange(24)] = 1.0
    wham_regressor = torch.zeros((17, vertices), dtype=torch.float32)
    wham_regressor[torch.arange(17), torch.arange(17)] = 1.0
    feet_regressor = torch.zeros((4, vertices), dtype=torch.float32)
    feet_regressor[:, :4] = torch.eye(4)
    lbs_weights = torch.eye(24, dtype=torch.float32)

    model = EmbeddedSMPL(
        {
            "shapedirs": torch.zeros((vertices, 3, 10)),
            "v_template": torch.arange(vertices * 3, dtype=torch.float32).reshape(vertices, 3),
            "J_regressor": regressor,
            "posedirs": torch.zeros((vertices, 3, 207)),
            "lbs_weights": lbs_weights,
            "parents": torch.tensor([-1] + list(range(23))),
            "J_regressor_wham": wham_regressor,
            "J_regressor_feet": feet_regressor,
        }
    )

    joints, pose = model.neutral_initialization(torch.device("cpu"))

    assert joints.shape == (17, 3)
    assert pose.shape == (24, 6)
    assert torch.isfinite(joints).all()
