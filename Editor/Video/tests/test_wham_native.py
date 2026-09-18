"""Smoke tests for the bundled WHAM temporal core and checkpoint loader."""

import numpy as np
import pytest
from types import SimpleNamespace

torch = pytest.importorskip("torch")

from pose_pipeline.adapters.wham_native import (  # noqa: E402
    NativeWHAMAdapter,
    NativeWhamNetwork,
    _as_keypoints2d,
    _canonicalize_wham_coordinates,
    _coerce_camera_angular_velocity,
    _coerce_image_features,
    _infer_architecture,
)
from pose_pipeline.adapters.texmotion_wham_adapter import _implementation_name  # noqa: E402
from pose_pipeline.observations import FrameObservations, Keypoint2D  # noqa: E402
from video_pose_extractor import (  # noqa: E402
    canonical_smplx_to_legacy_landmarks,
    assess_quality_pose_alignment,
    project_quality_3d_to_image,
    draw_pose_overlay,
)


class _Detector:
    is_initialized = True

    def detect(self, frame, timestamp_sec, frame_index=0):
        keypoints = [Keypoint2D(0.0, 0.0, 0.0) for _ in range(33)]
        # Distinct values catch reversed COCO/MediaPipe indexing and overflow.
        keypoints[11] = Keypoint2D(0.11, 0.21, 0.81)  # left shoulder
        keypoints[23] = Keypoint2D(0.23, 0.43, 0.83)  # left hip
        keypoints[27] = Keypoint2D(0.27, 0.47, 0.87)  # left ankle
        return FrameObservations(frame_index, timestamp_sec, keypoints, [])

    def close(self):
        pass


def test_bundled_wham_bridge_defaults_to_native_core(monkeypatch):
    monkeypatch.delenv("TEXMOTION_WHAM_ADAPTER_IMPL", raising=False)
    assert _implementation_name({}) == "pose_pipeline.adapters.wham_native"


def test_wham_local_feature_and_dpvo_archives_are_loaded(tmp_path):
    features = np.ones((2, 4), dtype=np.float32)
    feature_path = tmp_path / "vitpose_features.npy"
    np.save(feature_path, features)
    assert np.array_equal(_coerce_image_features(str(feature_path), 2, 4), features)

    poses = np.zeros((2, 7), dtype=np.float32)
    poses[:, 6] = 1.0
    dpvo_path = tmp_path / "dpvo_poses.npy"
    np.save(dpvo_path, poses)
    motion = _coerce_camera_angular_velocity(str(dpvo_path), 2, 30.0)
    assert motion.shape == (2, 6)


def test_wham_feature_archive_single_frame_probe_slices_first_frame(tmp_path):
    """A 1-frame preflight probe must safely slice multi-frame archive without rejecting."""
    features = np.arange(258 * 4, dtype=np.float32).reshape(258, 4)
    archive = {
        "features": features,
        "frame_indices": list(range(258)),
        "fps": 30.0,
        "timestamps": [i * (1.0 / 30.0) for i in range(258)],
        "trim_start": 0.0,
        "trim_end": 8.6,
    }
    # A 1-frame probe request (such as backend preflight)
    probed = _coerce_image_features(archive, 1, 4, frame_indices=[0])
    assert probed.shape == (1, 4)
    assert np.array_equal(probed[0], features[0])

    # Full sequence request matches perfectly
    full = _coerce_image_features(
        archive,
        258,
        4,
        frame_indices=list(range(258)),
        expected_fps=30.0,
        expected_trim_start=0.0,
        expected_trim_end=8.6,
        expected_timestamps=[i * (1.0 / 30.0) for i in range(258)],
    )
    assert full.shape == (258, 4)
    assert np.array_equal(full, features)


def test_wham_feature_archive_must_match_source_frame_indices_and_float_dtype():
    """Archive metadata cannot silently retime or coerce non-feature values."""
    with pytest.raises(ValueError, match="frame_indices"):
        _coerce_image_features(
            {
                "features": np.ones((2, 4), dtype=np.float32),
                "frame_indices": [101, 110],
            },
            2,
            4,
            frame_indices=[101, 107],
        )
    with pytest.raises(ValueError, match="floating-point"):
        _coerce_image_features(
            {"features": np.ones((2, 4), dtype=np.int32)},
            2,
            4,
            frame_indices=[101, 107],
        )


def test_wham_feature_archive_must_match_fps_trim_and_timestamps():
    """A valid tensor from another trim/FPS is still the wrong video input."""
    archive = {
        "features": np.ones((2, 4), dtype=np.float32),
        "frame_indices": [101, 107],
        "fps": 24.0,
        "trim_start": 1.0,
        "trim_end": 1.5,
        "timestamps": [1.0, 1.25],
    }
    with pytest.raises(ValueError, match="fps"):
        _coerce_image_features(
            archive,
            2,
            4,
            frame_indices=[101, 107],
            expected_fps=30.0,
        )
    with pytest.raises(ValueError, match="trim_start"):
        _coerce_image_features(
            archive,
            2,
            4,
            frame_indices=[101, 107],
            expected_fps=24.0,
            expected_trim_start=0.0,
        )
    with pytest.raises(ValueError, match="timestamps"):
        _coerce_image_features(
            archive,
            2,
            4,
            frame_indices=[101, 107],
            expected_fps=24.0,
            expected_trim_start=1.0,
            expected_trim_end=1.5,
            expected_timestamps=[0.0, 0.25],
        )


def test_wham_integrator_dimension_is_inferred_not_overridden_by_config():
    """A stale d_feat setting must fail before constructing a wrong Integrator."""
    state = {
        "motion_encoder.embed_layer.weight": np.zeros((2, 5), dtype=np.float32),
        "integrator.layer1.weight": np.zeros((4, 7), dtype=np.float32),
    }

    with pytest.raises(ValueError, match="Integrator.*dimension"):
        _infer_architecture(state, {"d_feat": 4})


def test_wham_settings_aliases_select_local_feature_and_camera_paths(tmp_path):
    feature_path = tmp_path / "features.npy"
    np.save(feature_path, np.zeros((1, 4), dtype=np.float32))
    camera_path = tmp_path / "camera.npy"
    np.save(camera_path, np.zeros((1, 6), dtype=np.float32))
    adapter = NativeWHAMAdapter({
        "imageFeaturePath": str(feature_path),
        "cameraModelPath": str(camera_path),
    })
    assert adapter._feature_source == str(feature_path)
    assert adapter._camera_source == str(camera_path)


def test_wham_vitpose_backbone_is_not_misread_as_feature_archive(tmp_path):
    """A downloaded ViTPose checkpoint needs an extractor, not archive coercion."""
    backbone_path = tmp_path / "vitpose-huge.pth"
    backbone_path.write_bytes(b"checkpoint")
    adapter = NativeWHAMAdapter({"imageFeatureBackbonePath": str(backbone_path)})

    assert adapter._feature_backbone_path == str(backbone_path)
    assert adapter._feature_source is None


def test_wham_rejects_raw_torch_checkpoint_configured_as_image_features(tmp_path):
    """A model state archive must never be consumed as WHAM image tokens."""
    backbone_path = tmp_path / "raw_vitpose_checkpoint.pth"
    torch.save(
        {"state_dict": {"encoder.weight": torch.zeros((1, 1), dtype=torch.float32)}},
        backbone_path,
    )
    adapter = NativeWHAMAdapter(
        {
            "modelDirectory": str(tmp_path),
            "imageFeaturesPath": str(backbone_path),
            "whamPreprocessDirectory": str(tmp_path / "prepared"),
        }
    )
    adapter._architecture = {"d_feat": 1, "d_embed": 2, "n_joints": 1}
    adapter._loaded_components["imageFeatureIntegrator"] = True
    adapter._model = SimpleNamespace(
        integrator=SimpleNamespace(layer1=SimpleNamespace(in_features=8))
    )

    frames = [np.zeros((8, 8, 3), dtype=np.uint8) for _ in range(2)]
    features = adapter._image_features_for_sequence(
        2, frames, [0.0, 0.1], [3, 7]
    )

    assert features is None
    assert adapter._feature_source is None
    metadata = adapter.get_metadata()
    assert metadata["imageFeatureSource"] == "unverified_local_descriptor_rejected"
    assert metadata["imageFeatureVerified"] is False
    assert "checkpoint" in metadata["imageFeatureFallbackReason"].lower()
    assert metadata["imageFeatureFallbackConsumed"] is False


def test_wham_body_model_path_alias_is_not_treated_as_callable_decoder(tmp_path):
    """A path-valued bodyModel alias must not be invoked as a model object.

    Unity supplies the resolved licensed body-model file through both the
    canonical ``bodyModelPath`` key and the legacy ``bodyModel`` alias.  The
    latter is a path string, not an already-instantiated SMPL module.  Before
    the regression fix the adapter called that string while creating the
    neutral seed and recorded ``'str' object is not callable``.
    """

    body_path = tmp_path / "SMPLX_NEUTRAL.npz"
    body_path.write_bytes(b"licensed-body-model-placeholder")
    adapter = NativeWHAMAdapter({
        "bodyModel": str(body_path),
        "allowNeutralSMPLInitialization": True,
    })

    adapter._initialize_smpl_assets({})
    adapter._smpl_initialization()

    assert not isinstance(adapter._smpl_model, str)
    assert adapter._smpl_seed is not None
    assert "str' object is not callable" not in str(adapter._smpl_error or "")


def test_wham_camera_calibration_yaml_is_safe_without_motion_poses(tmp_path):
    """Intrinsics configure the camera but must not be treated as DPVO poses."""
    camera_path = tmp_path / "camera.yaml"
    camera_path.write_text("fx: 900\nfy: 900\ncx: 320\ncy: 240\n", encoding="utf-8")
    adapter = NativeWHAMAdapter({"cameraModelPath": str(camera_path)})

    motion = adapter._camera_for_sequence(3)

    assert motion.shape == (3, 6)
    assert np.allclose(motion, 0.0)
    assert adapter.get_metadata()["cameraCalibrationAvailable"] is True
    assert adapter.get_metadata()["cameraMotionSource"] in ("calibration_only", "none")


def test_wham_dpvo_weights_are_not_misread_as_camera_poses(tmp_path):
    """A DPVO weight checkpoint is metadata until a pose exporter is configured."""
    dpvo_path = tmp_path / "dpvo.pth"
    torch.save({"state_dict": {"encoder.weight": torch.zeros(1)}}, dpvo_path)
    adapter = NativeWHAMAdapter({"dpvoModelPath": str(dpvo_path)})

    motion = adapter._camera_for_sequence(2)

    assert motion.shape == (2, 6)
    assert np.allclose(motion, 0.0)
    metadata = adapter.get_metadata()
    assert metadata["dpvoAssetPath"] == str(dpvo_path)
    assert metadata["cameraMotionSource"] in ("dpvo_weights_only", "none")
    assert any("pose" in str(error).lower() for error in metadata["optionalErrors"])


def test_native_wham_rejects_unverified_local_descriptors_for_learned_features(tmp_path):
    """A local descriptor may explain fallback, but must not enter WHAM's integrator."""
    backbone_path = tmp_path / "vitpose-huge.pth"
    backbone_path.write_bytes(b"backbone")
    adapter = NativeWHAMAdapter(
        {
            "modelDirectory": str(tmp_path),
            "imageFeatureBackbonePath": str(backbone_path),
            "imageFeatureRunnerModule": "pose_pipeline.missing_vitpose_runner",
            "whamPreprocessDirectory": str(tmp_path / "prepared"),
        }
    )
    adapter._architecture = {"d_feat": 8}
    adapter._loaded_components["imageFeatureIntegrator"] = True
    frames = [np.zeros((8, 8, 3), dtype=np.uint8) for _ in range(2)]

    adapter._prepare_local_runtime(frames, [0.0, 0.1], [3, 7])

    assert adapter._feature_source is None
    metadata = adapter.get_metadata()
    assert metadata["imageFeatureSource"] == "unverified_local_descriptor_rejected"
    assert metadata["imageFeatureVerified"] is False
    assert metadata["imageFeatureFallbackReason"]
    assert "not fed" in metadata["imageFeatureFallbackReason"].lower()
    assert any("image" in item for item in metadata["missingOptionalAssets"])


def test_native_wham_distinguishes_dpvo_runner_from_local_optical_flow(tmp_path):
    """Camera provenance must identify real DPVO output instead of generic motion."""
    dpvo_path = tmp_path / "dpvo.pth"
    dpvo_path.write_bytes(b"dpvo")
    frames = [np.zeros((16, 16, 3), dtype=np.uint8) for _ in range(2)]

    def dpvo_runner(images, timestamps, indices):
        del images, timestamps, indices
        rows = np.zeros((2, 7), dtype=np.float32)
        rows[:, 6] = 1.0
        return rows

    adapter = NativeWHAMAdapter(
        {
            "modelDirectory": str(tmp_path),
            "dpvoModelPath": str(dpvo_path),
            "dpvoRunner": dpvo_runner,
            "whamPreprocessDirectory": str(tmp_path / "prepared"),
        }
    )
    adapter._architecture = {"d_feat": 8}
    adapter._loaded_components["trajectoryDecoder"] = True
    adapter._prepare_local_runtime(frames, [0.0, 0.1], [0, 1])
    adapter._camera_for_sequence(2)

    metadata = adapter.get_metadata()
    assert metadata["cameraMotionSource"] == "dpvo"
    assert metadata["cameraMotionProvenance"]["source"] == "dpvo"
    assert metadata["cameraMotionProvenance"]["verified"] is True
    assert metadata["dpvoVerified"] is True


def test_wham_refreshes_one_frame_preflight_camera_cache_for_full_sequence(tmp_path):
    """The smoke pass must not leave a one-row camera cache for the clip."""
    dpvo_path = tmp_path / "dpvo.pth"
    dpvo_path.write_bytes(b"dpvo")
    adapter = NativeWHAMAdapter(
        {
            "modelDirectory": str(tmp_path),
            "dpvoModelPath": str(dpvo_path),
            "whamPreprocessDirectory": str(tmp_path / "prepared"),
        }
    )
    adapter._architecture = {"d_feat": 8}
    adapter._loaded_components["trajectoryDecoder"] = True
    one_frame = [np.zeros((16, 16, 3), dtype=np.uint8)]
    adapter._prepare_local_runtime(one_frame, [0.0], [0])
    assert adapter._camera_for_sequence(1).shape == (1, 6)

    full_sequence = [np.zeros((16, 16, 3), dtype=np.uint8) for _ in range(3)]
    adapter._prepare_local_runtime(full_sequence, [0.0, 0.1, 0.2], [0, 1, 2])
    motion = adapter._camera_for_sequence(3)

    assert motion.shape == (3, 6)


def test_wham_refreshes_same_length_camera_cache_when_source_frames_change(tmp_path):
    """A one-frame smoke cache must not be reused for a different one-frame clip."""
    dpvo_path = tmp_path / "dpvo.pth"
    dpvo_path.write_bytes(b"dpvo")

    def camera_runner(images, timestamps, indices):
        del timestamps, indices
        marker = float(np.mean(images[0]))
        rows = np.zeros((len(images), 7), dtype=np.float32)
        rows[:, 3] = 0.0
        rows[:, 4] = 0.0
        rows[:, 5] = 0.0
        rows[:, 6] = 1.0
        if marker:
            rows[-1, 5] = np.sin(0.25)
            rows[-1, 6] = np.cos(0.25)
        return rows

    adapter = NativeWHAMAdapter(
        {
            "modelDirectory": str(tmp_path),
            "dpvoModelPath": str(dpvo_path),
            "cameraPoseExtractor": camera_runner,
            "whamPreprocessDirectory": str(tmp_path / "prepared"),
        }
    )
    adapter._architecture = {"d_feat": 8}
    adapter._loaded_components["trajectoryDecoder"] = True

    first_frames = [
        np.zeros((16, 16, 3), dtype=np.uint8),
        np.zeros((16, 16, 3), dtype=np.uint8),
    ]
    adapter._prepare_local_runtime(first_frames, [0.0, 0.1], [0, 1])
    first_motion = adapter._camera_for_sequence(2)
    assert np.allclose(first_motion, 0.0)

    second_frames = [
        np.full((16, 16, 3), 255, dtype=np.uint8),
        np.full((16, 16, 3), 255, dtype=np.uint8),
    ]
    adapter._prepare_local_runtime(second_frames, [0.0, 0.1], [0, 1])
    second_motion = adapter._camera_for_sequence(2)

    assert second_motion.shape == (2, 6)
    assert not np.allclose(second_motion, first_motion)


def test_wham_mediapipe_to_coco17_mapping():
    keypoints = [Keypoint2D(0.0, 0.0, 0.0) for _ in range(33)]
    keypoints[11] = Keypoint2D(0.11, 0.21, 0.81)
    keypoints[23] = Keypoint2D(0.23, 0.43, 0.83)
    keypoints[27] = Keypoint2D(0.27, 0.47, 0.87)
    observation = FrameObservations(0, 0.0, keypoints, [])
    points, scores = _as_keypoints2d(observation)
    assert points.shape == (17, 2)
    assert scores.shape == (17,)
    assert points[5].tolist() == pytest.approx([0.11, 0.21])
    assert points[11].tolist() == pytest.approx([0.23, 0.43])
    assert points[15].tolist() == pytest.approx([0.27, 0.47])


def test_wham_camera_coordinates_are_canonicalized_at_adapter_boundary():
    """Raw WHAM camera Y-down/Z-away output becomes canonical SMPL-X once."""
    raw = np.array(
        [
            [0.0, -0.55, -0.12, 0.9],  # head: above image pelvis in raw camera Y
            [0.0, 0.00, 0.00, 0.9],   # pelvis
            [0.0, 0.60, 0.23, 0.9],   # ankle: below image pelvis in raw camera Y
        ],
        dtype=np.float32,
    )
    canonical = _canonicalize_wham_coordinates(raw)
    assert canonical[:, 0].tolist() == pytest.approx(raw[:, 0].tolist())
    assert canonical[:, 1].tolist() == pytest.approx([0.55, 0.0, -0.60])
    assert canonical[:, 2].tolist() == pytest.approx([0.12, 0.0, -0.23])
    # The score/provenance column is not an axis and must survive unchanged.
    assert canonical[:, 3].tolist() == pytest.approx(raw[:, 3].tolist())


def test_canonical_quality_stream_is_encoded_once_for_legacy_and_overlay():
    canonical = np.zeros((33, 3), dtype=np.float64)
    canonical[0] = [0.0, 0.0, 0.0]       # pelvis
    canonical[15] = [0.0, 0.70, 0.10]    # head: above pelvis, forward
    canonical[27] = [0.0, -0.80, -0.20]  # ankle: below pelvis, backward

    legacy = canonical_smplx_to_legacy_landmarks(canonical)
    assert legacy[15].tolist() == pytest.approx([0.0, -0.70, -0.10])
    assert legacy[27].tolist() == pytest.approx([0.0, 0.80, 0.20])

    # The overlay consumes the legacy camera-oriented stream: head remains
    # above the ankle in image coordinates (smaller image Y).
    projected = project_quality_3d_to_image(legacy)
    assert projected[15, 1] < projected[27, 1]


def test_quality_overlay_does_not_explode_when_shoulders_are_collapsed():
    """A self-occluded WHAM shoulder pair must not set an unbounded 2D scale."""
    canonical = np.zeros((33, 3), dtype=np.float64)
    canonical[0] = [0.0, 0.0, 0.0]
    # The temporal model can briefly collapse the two shoulder slots while
    # both arms are behind the head.  Keep a stable torso/hip extent so the
    # overlay can use that as its scale reference.
    canonical[11] = [-0.0005, 0.55, 0.0]
    canonical[12] = [0.0005, 0.55, 0.0]
    canonical[23] = [-0.08, 0.0, 0.0]
    canonical[24] = [0.08, 0.0, 0.0]
    canonical[15] = [0.0, 0.75, 0.0]
    canonical[27] = [-0.08, -0.85, 0.0]
    canonical[28] = [0.08, -0.85, 0.0]
    norm = np.zeros((33, 3), dtype=np.float64)
    norm[[0, 11, 12, 23, 24, 15, 27, 28], 2] = 1.0
    norm[0, :2] = [0.50, 0.50]
    norm[11, :2] = [0.45, 0.30]
    norm[12, :2] = [0.55, 0.30]
    norm[23, :2] = [0.47, 0.58]
    norm[24, :2] = [0.53, 0.58]
    norm[15, :2] = [0.50, 0.18]
    norm[27, :2] = [0.44, 0.90]
    norm[28, :2] = [0.56, 0.90]

    projected = project_quality_3d_to_image(canonical, norm, coordinate_system="smplx")

    assert projected is not None
    # No valid quality joint should be pinned to the clipping boundary just
    # because one occlusion-corrupted anchor pair became coincident.
    assert np.all(projected[[0, 11, 12, 15, 23, 24, 27, 28], :2] > 0.03)
    assert np.all(projected[[0, 11, 12, 15, 23, 24, 27, 28], :2] < 0.97)
    assert float(np.ptp(projected[:, 0])) < 0.75
    assert float(np.ptp(projected[:, 1])) < 0.85


def test_quality_alignment_gate_flags_collapsed_wham_geometry():
    """A finite but tiny WHAM body must be routed to the safe auxiliary pose."""
    quality = np.zeros((33, 3), dtype=np.float64)
    observed = np.zeros((33, 3), dtype=np.float64)
    body_indices = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]
    for index in body_indices:
        quality[index] = [0.001 * (index % 3), 0.02 * (index % 5), 0.0]
        observed[index] = [0.35 + 0.02 * (index % 4), 0.20 + 0.04 * (index % 6), 1.0]
    observed[0, :2] = [0.5, 0.5]

    needs_fallback, reason, projected = assess_quality_pose_alignment(
        quality, observed, coordinate_system="smplx"
    )

    assert needs_fallback is True
    assert "body_extent_collapsed" in reason
    assert projected is not None


def test_native_wham_state_dict_fixture_loads_and_rolls_out(tmp_path):
    checkpoint = tmp_path / "wham_fixture.pth.tar"
    torch.save({"state_dict": NativeWhamNetwork().state_dict()}, checkpoint)
    adapter = NativeWHAMAdapter({
        "modelPath": str(checkpoint),
        "device": "cpu",
        "torch": torch,
        "detector": _Detector(),
    })
    assert adapter.is_available() is True
    assert adapter.initialize() is True, adapter.error
    try:
        frames = [np.zeros((48, 64, 3), dtype=np.uint8) for _ in range(2)]
        result = adapter.infer_sequence(frames, [0.0, 1 / 30], [4, 5])
        assert len(result["frames"]) == 2
        assert result["frames"][0]["landmarks3d"].shape == (17, 4)
        assert result["frames"][0]["keypoints2d"].shape == (17, 3)
        assert result["metadata"]["loadedParameterCount"] > 0
    finally:
        adapter.close()


def test_sparse_face_and_digit_slots_not_bunched_at_origin():
    """Unmodeled COCO-17 slots (detailed face/digits) must be suppressed from overlay."""
    canonical = np.zeros((33, 3), dtype=np.float64)
    # Give canonical main joints
    canonical[0] = [0.0, 0.0, 0.0]        # pelvis
    canonical[11] = [-0.20, 0.50, 0.0]    # l shoulder
    canonical[12] = [0.20, 0.50, 0.0]     # r shoulder
    canonical[15] = [-0.35, 0.20, 0.0]    # l wrist
    canonical[16] = [0.35, 0.20, 0.0]     # r wrist
    canonical[23] = [-0.15, 0.0, 0.0]     # l hip
    canonical[24] = [0.15, 0.0, 0.0]      # r hip
    canonical[27] = [-0.15, -0.75, 0.0]   # l ankle
    canonical[28] = [0.15, -0.75, 0.0]    # r ankle

    # Detailed face slots (1, 3, 4, 6, 9, 10) are zeros
    norm = np.zeros((33, 3), dtype=np.float64)
    norm[[0, 11, 12, 23, 24, 27, 28], 2] = 1.0
    norm[0, :2] = [0.5, 0.2]
    norm[11, :2] = [0.4, 0.35]
    norm[12, :2] = [0.6, 0.35]
    norm[23, :2] = [0.45, 0.55]
    norm[24, :2] = [0.55, 0.55]
    norm[27, :2] = [0.45, 0.85]
    norm[28, :2] = [0.55, 0.85]

    projected = project_quality_3d_to_image(canonical, norm, coordinate_system="smplx")
    assert projected is not None
    # Face detail slots (1, 3, 4, 6, 9, 10) must have valid == 0.0 (suppressed)
    for f_idx in (1, 3, 4, 6, 9, 10):
        assert projected[f_idx, 2] == 0.0, f"Slot {f_idx} was not suppressed"


def test_draw_pose_overlay_modes():
    """draw_pose_overlay renders successfully across dual, 3d, and 2d modes."""
    frame = np.full((360, 640, 3), 30, dtype=np.uint8)
    main_pts = np.zeros((33, 3), dtype=np.float64)
    main_pts[:, :2] = 0.5
    main_pts[:, 2] = 1.0
    sec_pts = np.zeros((33, 3), dtype=np.float64)
    sec_pts[:, :2] = 0.48
    sec_pts[:, 2] = 1.0

    # Dual mode with secondary
    out_dual = draw_pose_overlay(
        frame, main_pts, 0, 10, 0.95, backend_name="WHAM",
        secondary_landmarks=sec_pts, overlay_mode="dual",
    )
    assert out_dual.shape == frame.shape

    # 3D mode
    out_3d = draw_pose_overlay(
        frame, main_pts, 0, 10, 0.95, backend_name="WHAM",
        secondary_landmarks=sec_pts, overlay_mode="3d",
    )
    assert out_3d.shape == frame.shape

    # 2D mode
    out_2d = draw_pose_overlay(
        frame, sec_pts, 0, 10, 0.95, backend_name="MediaPipe",
        secondary_landmarks=None, overlay_mode="2d",
    )
    assert out_2d.shape == frame.shape
