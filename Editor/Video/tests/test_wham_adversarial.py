"""Adversarial regression coverage for the WHAM/fusion collapse failure.

These tests deliberately use small deterministic fixtures instead of a model
download.  They exercise the seams where a valid temporal pose can otherwise
collapse into face/limb aliases, acquire a second axis flip, lose sparse
frames, or hide a fallback behind an apparently successful backend name.
"""

import warnings

import numpy as np
import pytest

from pose_pipeline.backends import create_pose_backend
from pose_pipeline.backends.pytorch_backend import PyTorchPoseBackend
from pose_pipeline.camera import (
    CameraModel,
    canonical_to_legacy_overlay,
    legacy_overlay_to_canonical,
    transform_coordinates,
    wham_to_canonical,
)
from pose_pipeline.fusion import (
    FusionConfig,
    align_observation_sequences,
    fuse_wham_mediapipe,
)
from video_pose_extractor import (
    assess_quality_pose_alignment,
    draw_pose_overlay,
    project_quality_3d_to_image,
)


def _canonical_body_pose(mode="normal"):
    """Return a sparse but anatomically ordered canonical 33-slot pose."""

    pose = np.zeros((33, 3), dtype=np.float64)
    pose[0] = [0.0, 0.0, 0.1]  # pelvis
    for index, (x, y) in {
        11: (-0.20, 0.55),  # left shoulder
        12: (0.20, 0.55),   # right shoulder
        13: (-0.35, 0.30),  # left elbow
        14: (0.35, 0.30),   # right elbow
        15: (-0.50, 0.10),  # left wrist
        16: (0.50, 0.10),   # right wrist
        23: (-0.15, 0.00),  # left hip
        24: (0.15, 0.00),   # right hip
        25: (-0.15, -0.40), # left knee
        26: (0.15, -0.40),  # right knee
        27: (-0.15, -0.80), # left ankle
        28: (0.15, -0.80),  # right ankle
        29: (-0.15, -0.80), # left heel
        30: (0.15, -0.80),  # right heel
        31: (-0.15, -0.90), # left foot
        32: (0.15, -0.90),  # right foot
    }.items():
        pose[index] = [x, y, 0.0]

    if mode == "collapsed":
        for index in (11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32):
            pose[index] = [0.0, 0.03 * (index % 2), 0.0]
    elif mode == "edge_spanning":
        # Three points on one arm create the same screen-spanning condition as
        # a topology/decoder error, while leaving independent torso anchors
        # intact for the alignment gate.
        for index in (13, 15, 16):
            pose[index] = [-100.0, 0.10, 0.0]
    elif mode != "normal":
        raise ValueError(f"unknown pose fixture mode: {mode}")
    return pose


def _image_anchors_for_pose(pose):
    """Encode a deterministic affine image observation for a canonical pose."""

    camera_xy = np.asarray(pose, dtype=np.float64).copy()
    camera_xy[:, 1] *= -1.0  # canonical +Y up -> image +Y down
    anchors = np.zeros((33, 3), dtype=np.float64)
    body_indices = (0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32)
    for index in body_indices:
        anchors[index, :2] = 0.5 + 0.30 * camera_xy[index, :2]
        anchors[index, 2] = 1.0
    return anchors


def test_wham_j17_ordering_maps_all_semantic_slots_without_coco_aliases(mock_frame_bgr):
    """Every WHAM J17 slot must land on its H36M/WHAM semantic destination."""

    backend = PyTorchPoseBackend(kind="wham", device="cpu")
    points = np.zeros((17, 4), dtype=np.float64)
    for source_index in range(17):
        points[source_index, :3] = [source_index + 0.11, source_index + 0.22, source_index + 0.33]
        points[source_index, 3] = 0.9

    observation = backend._frame_observation(
        {
            "landmarks3d": points,
            "confidence": 0.9,
            "metadata": {"jointTopology": "wham_j17"},
        },
        mock_frame_bgr,
        timestamp=0.25,
        frame_index=17,
    )

    assert observation is not None
    source_to_target = {
        0: (28, 30, 32),   # right ankle -> right ankle/heel/foot
        1: (26,),          # right knee
        2: (24,),          # right hip
        3: (23,),          # left hip
        4: (25,),          # left knee
        5: (27, 29, 31),   # left ankle -> left ankle/heel/foot
        6: (16,),          # right wrist
        7: (14,),          # right elbow
        8: (12,),          # right shoulder
        9: (11,),          # left shoulder
        10: (13,),         # left elbow
        11: (15,),         # left wrist
        13: (7, 8),        # head-top -> both ears as a stable head anchor
        16: (0,),          # head -> nose anchor
    }
    for source_index, target_indices in source_to_target.items():
        expected = points[source_index, :3]
        for target_index in target_indices:
            np.testing.assert_allclose(
                observation.landmarks_3d[target_index].to_array(), expected, atol=1e-6
            )

    # WHAM's missing spine slot must not accidentally populate an unrelated
    # face joint; all targets above are explicit topology aliases.
    assert observation.landmarks_3d[1].status.name == "MISSING"
    assert observation.landmarks_3d[6].status.name == "MISSING"


def test_wham_coordinate_roundtrip_flips_only_axes_and_never_metadata_columns():
    """Raw WHAM -> canonical -> overlay must be exactly one Y/Z conversion."""

    raw = np.array(
        [
            [[0.2, -0.5, 0.3, 0.8, 11.0], [-0.1, 0.25, -0.4, 0.2, 12.0]],
            [[0.4, -0.7, 0.5, 0.6, 21.0], [-0.3, 0.45, -0.2, 0.4, 22.0]],
        ],
        dtype=np.float64,
    )
    original = raw.copy()

    canonical = wham_to_canonical(raw)
    overlay = canonical_to_legacy_overlay(canonical)
    roundtrip = legacy_overlay_to_canonical(overlay)

    np.testing.assert_allclose(canonical[..., :3], raw[..., [0, 1, 2]] * [1.0, -1.0, -1.0])
    # The legacy overlay is camera-oriented (and therefore matches raw WHAM
    # axes), while decoding it returns the canonical representation.
    np.testing.assert_allclose(overlay, raw)
    np.testing.assert_allclose(roundtrip, canonical)
    np.testing.assert_allclose(raw, original)
    np.testing.assert_allclose(canonical[..., 3:], raw[..., 3:])

    # MediaPipe's raw world landmarks are camera-oriented (+Y down, +Z away).
    # The extraction integration helper flips both axes at that seam; this
    # separate transform check keeps the canonical camera conversion explicit.
    mediapipe_world = raw[..., :3].copy()
    converted = transform_coordinates(
        mediapipe_world, "image_y_down_z_away", "smplx"
    )
    np.testing.assert_allclose(converted[..., 1], -mediapipe_world[..., 1])
    np.testing.assert_allclose(converted[..., 2], -mediapipe_world[..., 2])


def test_degenerate_geometry_and_missing_observations_stay_finite_without_runtime_warnings():
    """Zero/all-missing clips must not emit NaNs or reduction warnings."""

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert project_quality_3d_to_image(np.zeros((33, 3), dtype=np.float64)) is None
        needs_fallback, reason, projected = assess_quality_pose_alignment(
            np.zeros((33, 3), dtype=np.float64),
            np.zeros((33, 3), dtype=np.float64),
            coordinate_system="smplx",
        )
        result = fuse_wham_mediapipe(
            np.full((2, 3, 3), np.nan, dtype=np.float64),
            mediapipe_2d=np.full((2, 3, 2), np.nan, dtype=np.float64),
            score=np.ones((2, 3), dtype=np.float64),
            visibility=np.ones((2, 3), dtype=np.float64),
            presence=np.ones((2, 3), dtype=np.float64),
        )
        perspective = CameraModel(projection_type="perspective", min_depth=1e-3)
        perspective_projection = perspective.project(np.zeros((2, 3), dtype=np.float64))

    assert needs_fallback is False
    assert reason == ""
    assert projected is None
    assert np.isfinite(result.joints).all()
    assert np.isfinite(perspective_projection).all()
    assert not any(issubclass(item.category, RuntimeWarning) for item in caught)


def test_partial_joint_fusion_keeps_independent_2d_3d_modalities_and_provenance():
    """A missing joint in one modality must not erase valid data in another."""

    wham = np.array([[[0.0, 0.0, 0.5], [np.nan, np.nan, np.nan], [0.8, 0.1, 0.2]]])
    mediapipe_3d = np.array([[[np.nan, np.nan, np.nan], [0.3, 0.2, 0.7], [np.nan, np.nan, np.nan]]])
    mediapipe_2d = np.array([[[0.8, 0.5], [np.nan, np.nan], [np.nan, np.nan]]])
    score = np.full((1, 3), 0.95, dtype=np.float64)

    result = fuse_wham_mediapipe(
        wham,
        mediapipe_3d=mediapipe_3d,
        mediapipe_2d=mediapipe_2d,
        score=score,
        visibility=score,
        presence=score,
        camera=CameraModel.orthographic(scale=0.5, principal_point=(0.5, 0.5)),
        config=FusionConfig(
            mediapipe_3d_weight=1.0,
            reprojection_weight=1.0,
            temporal_smoothing=0.0,
            max_iterations=3,
        ),
    )

    assert np.isfinite(result.joints).all()
    # Joint 0 has no MP-3D but its valid 2D observation corrects image X while
    # retaining WHAM depth; joint 1 has no WHAM but valid MP-3D is used.
    assert result.joints[0, 0, 0] > 0.0
    assert result.joints[0, 0, 2] == pytest.approx(0.5, abs=1e-6)
    np.testing.assert_allclose(result.joints[0, 1], mediapipe_3d[0, 1], atol=1e-6)
    np.testing.assert_allclose(result.joints[0, 2], wham[0, 2], atol=1e-6)

    provenance = result.provenance[0]
    assert provenance[0]["source"] == "fused"
    assert provenance[0]["auxiliary"] == ["mediapipe_2d"]
    assert provenance[1]["source"] == "mediapipe"
    assert provenance[1]["primary"] == "mediapipe"
    assert provenance[2]["source"] == "wham"
    assert provenance[2]["status"] == "predicted"


def test_sparse_frame_indices_drive_disagreement_intervals_and_backend_outputs():
    """Sparse source frame IDs must survive both temporal and fusion seams."""

    wham = np.zeros((3, 1, 3), dtype=np.float64)
    mediapipe = np.zeros((3, 1, 3), dtype=np.float64)
    mediapipe[1, 0, 0] = 1.0
    result = fuse_wham_mediapipe(
        wham,
        mediapipe_3d=mediapipe,
        score=np.ones((3, 1), dtype=np.float64),
        visibility=np.ones((3, 1), dtype=np.float64),
        presence=np.ones((3, 1), dtype=np.float64),
        frame_indices=[101, 205, 309],
        config=FusionConfig(mediapipe_3d_weight=0.0, reprojection_weight=0.0),
    )
    disagreement = [
        interval for interval in result.uncertainty_intervals
        if interval["reason"] == "wham_mediapipe_disagreement"
    ]
    assert disagreement == [{
        "startFrame": 205,
        "endFrame": 205,
        "reason": "wham_mediapipe_disagreement",
        "confidence": 0.0,
        "recommendedAction": "review_depth_hypothesis",
    }]

    class _IndexRecordingAdapter:
        def __init__(self):
            self.calls = []

        def infer_sequence(self, frames, timestamps, indices):
            self.calls.append((len(frames), list(timestamps), list(indices)))
            return {
                "frames": [
                    {
                        "landmarks3d": np.tile(
                            [[float(index), float(index) + 0.1, float(index) + 0.2]],
                            (33, 1),
                        ),
                        "metadata": {"coordinateSystem": "smplx"},
                    }
                    for index in indices
                ],
                "uncertaintyIntervals": [
                    {
                        "startFrame": 110,
                        "endFrame": 110,
                        "reason": "frame_alignment_fixture",
                        "confidence": 0.5,
                        "recommendedAction": "review_frame",
                    }
                ],
            }

    backend = PyTorchPoseBackend(kind="wham", device="cpu")
    adapter = _IndexRecordingAdapter()
    backend._adapter = adapter
    backend._adapter_name = "frame_alignment_fixture"
    backend._device = "cpu"
    backend.is_initialized = True
    frames = [
        np.zeros((8, 8, 3), dtype=np.uint8),
        None,
        np.ones((8, 8, 3), dtype=np.uint8),
    ]
    observations = backend.detect_sequence(
        frames,
        timestamps_sec=[1.01, 1.07, 1.13],
        frame_indices=[101, 107, 110],
    )
    try:
        assert adapter.calls == [(2, [1.01, 1.13], [101, 110])]
        assert observations[1] is None
        assert observations[0].frame_index == 101
        assert observations[2].frame_index == 110
        assert observations[0].landmarks_3d[0].x == pytest.approx(101.0)
        assert observations[2].landmarks_3d[0].x == pytest.approx(110.0)
        assert observations[2].uncertainty_intervals[0].start_frame == 110
    finally:
        backend.close()


def test_alignment_prefers_frame_identity_over_out_of_order_positions_and_keeps_gaps():
    """Out-of-order adapter records must not be paired by list position."""

    primary, auxiliary, diagnostics = align_observation_sequences(
        [
            {"frameIndex": 30, "timestampSec": 1.0, "value": "quality-30"},
            {"frameIndex": 10, "timestampSec": 0.3, "value": "quality-10"},
        ],
        [
            {"frameIndex": 10, "timestampSec": 0.3, "value": "mp-10"},
            {"frameIndex": 30, "timestampSec": 1.0, "value": "mp-30"},
        ],
        frame_indices=[10, 20, 30],
        timestamps=[0.3, 0.6, 1.0],
        frame_count=3,
    )

    assert [item["value"] if item is not None else None for item in primary] == [
        "quality-10", None, "quality-30"
    ]
    assert [item["value"] if item is not None else None for item in auxiliary] == [
        "mp-10", None, "mp-30"
    ]
    assert diagnostics["qualitySourceFrameIndices"] == [10, None, 30]
    assert diagnostics["mediapipeSourceFrameIndices"] == [10, None, 30]
    assert diagnostics["missingQualityFrames"] == [20]
    assert diagnostics["missingMediaPipeFrames"] == [20]
    assert all(match["match"] == "frame_index" for match in diagnostics["qualityMatches"])


def test_pytorch_sequence_preserves_explicit_adapter_frame_identity():
    """Backend batching must not retime explicitly indexed WHAM rows."""

    class _OutOfOrderAdapter:
        def infer_sequence(self, frames, timestamps, indices):
            del frames, timestamps
            rows = []
            for index in reversed(indices):
                rows.append({
                    "frameIndex": index,
                    "landmarks3d": np.tile(
                        [[float(index), float(index) + 0.1, float(index) + 0.2]],
                        (17, 1),
                    ),
                    "metadata": {
                        "coordinateSystem": "smplx",
                        "jointTopology": "wham_j17",
                    },
                })
            return {"frames": rows}

    backend = PyTorchPoseBackend(kind="wham", device="cpu")
    backend._adapter = _OutOfOrderAdapter()
    backend._adapter_name = "out_of_order_fixture"
    backend._device = "cpu"
    backend.is_initialized = True
    frames = [
        np.zeros((8, 8, 3), dtype=np.uint8),
        np.ones((8, 8, 3), dtype=np.uint8),
    ]
    try:
        observations = backend.detect_sequence(
            frames,
            timestamps_sec=[0.0, 1.0 / 30.0],
            frame_indices=[101, 110],
        )
        assert [item.frame_index for item in observations] == [101, 110]
        assert observations[0].landmarks_3d[0].x == pytest.approx(101.0)
        assert observations[1].landmarks_3d[0].x == pytest.approx(110.0)
    finally:
        backend.close()


def test_pytorch_sequence_preserves_identity_for_full_810_frame_clip():
    """The production-sized clip must not be retimed when WHAM emits reverse order."""

    class _ReverseAdapter:
        def infer_sequence(self, frames, timestamps, indices):
            del frames, timestamps
            return {
                "frames": [
                    {
                        "frameIndex": index,
                        "landmarks3d": np.tile([[float(index), 0.0, 0.0]], (17, 1)),
                        "metadata": {"jointTopology": "wham_j17", "coordinateSystem": "smplx"},
                    }
                    for index in reversed(indices)
                ]
            }

    backend = PyTorchPoseBackend(kind="wham", device="cpu")
    backend._adapter = _ReverseAdapter()
    backend._adapter_name = "reverse_810_fixture"
    backend._device = "cpu"
    backend.is_initialized = True
    frames = [np.zeros((2, 2, 3), dtype=np.uint8) for _ in range(810)]
    indices = list(range(1, 811))
    try:
        observations = backend.detect_sequence(
            frames, [index / 30.0 for index in indices], indices
        )
        assert len(observations) == 810
        assert [item.frame_index for item in observations] == indices
        assert [item.landmarks_3d[0].x for item in observations] == pytest.approx(indices)
    finally:
        backend.close()


def test_wham_contract_metadata_overrides_spoofed_adapter_stage_claims():
    """An adapter cannot claim unavailable WHAM stages are complete."""

    class _SpoofingAdapter:
        def get_metadata(self):
            return {
                "whamFullParity": True,
                "whamMissingStages": [],
                "capabilityFlags": {"fullWhamParity": True, "worldMotion": True},
            }

    backend = PyTorchPoseBackend(kind="wham", device="cpu")
    backend._adapter = _SpoofingAdapter()
    try:
        metadata = backend.get_metadata()
        assert metadata["whamFullParity"] is False
        assert metadata["whamMissingStages"]
        assert metadata["capabilityFlags"]["fullWhamParity"] is False
        assert metadata["capabilities"]["worldMotion"] is False
        assert metadata["whamStageDiagnostics"]
    finally:
        backend.close()


def test_native_wham_runtime_stage_diagnostics_override_static_file_presence(tmp_path):
    """Installed weights must not be reported as executable full parity.

    ViTPose/DPVO checkpoints and a camera template are static assets.  The
    native adapter can still report a missing frame-feature extractor or
    camera-pose export after those files are present; the runtime capability
    contract must remain authoritative for the result card.
    """

    for name in (
        "wham_vit_w_3dpw.pth.tar",
        "SMPLX_NEUTRAL.npz",
        "vitpose-huge.pth",
        "camera.yaml",
        "dpvo.pth",
        "texmotion_wham_adapter.py",
    ):
        (tmp_path / name).write_bytes(b"asset")

    backend = PyTorchPoseBackend(kind="wham", model_directory=str(tmp_path))

    class _NativeRuntimeMetadata:
        def get_metadata(self):
            return {
                "adapterImplementation": "texmotion_native_wham",
                "capabilities": {
                    "nativeMotionEncoder": True,
                    "imageFeatureIntegrator": False,
                    "trajectoryDecoder": True,
                    "smplDecoder": True,
                    "worldTrajectory": False,
                },
                "missingOptionalAssets": ["image_feature_extractor_for_backbone", "camera_motion_or_slam"],
                "optionalErrors": ["ViTPose backbone requires a frame-feature extractor"],
                "cameraMotionSource": "dpvo_weights_only",
                "worldMotionAvailable": False,
            }

    backend._adapter = _NativeRuntimeMetadata()
    try:
        metadata = backend.get_metadata()
        assert metadata["whamFullParity"] is False
        assert metadata["capabilityFlags"]["fullWhamParity"] is False
        assert metadata["capabilities"]["worldMotion"] is False
        assert metadata["missingOptionalAssets"] == [
            "image_feature_extractor_for_backbone",
            "camera_motion_or_slam",
        ]
        assert metadata["optionalErrors"]
    finally:
        backend.close()


def test_quality_fallback_metadata_preserves_requested_backend_and_concrete_reason(tmp_path):
    """A failed WHAM initialization must not masquerade as MediaPipe success."""

    missing_checkpoint = tmp_path / "missing-wham-checkpoint.pth.tar"
    backend = create_pose_backend(
        preferred="wham",
        pytorch_model_path=str(missing_checkpoint),
        strict_quality=False,
    )
    try:
        metadata = backend.get_metadata()
        assert backend.name == "mediapipe"
        assert metadata["fallbackFrom"] == "wham"
        assert "Requested WHAM backend could not initialize" in metadata["fallbackReason"]
        assert metadata["fallbackReason"].strip()
    finally:
        backend.close()


def test_alignment_gate_measures_direct_and_fallback_frames_and_bounds_overlay_edges():
    """Collapsed/edge-spanning hypotheses must select a bounded fallback overlay."""

    normal = _canonical_body_pose()
    anchors = _image_anchors_for_pose(normal)
    decisions = []
    reasons = []
    projections = []
    for pose in (
        normal,
        _canonical_body_pose("collapsed"),
        _canonical_body_pose("edge_spanning"),
    ):
        needs_fallback, reason, projection = assess_quality_pose_alignment(
            pose,
            anchors,
            coordinate_system="smplx",
        )
        decisions.append(needs_fallback)
        reasons.append(reason)
        projections.append(projection)

    # This is also the deterministic direct-vs-fallback measurement used in
    # the review report: one direct frame and two guarded frames.
    assert decisions == [False, True, True]
    assert "body_extent_collapsed" in reasons[1]
    assert "p90_2d_error" in reasons[2]

    cv2 = pytest.importorskip("cv2")
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    bad_overlay = draw_pose_overlay(
        frame,
        projections[2],
        frame_idx=0,
        total_frames=1,
        conf=0.9,
        backend_name="WHAM",
    )
    selected_overlay = draw_pose_overlay(
        frame,
        anchors if decisions[2] else projections[2],
        frame_idx=0,
        total_frames=1,
        conf=0.9,
        backend_name="MediaPipe Fallback",
    )

    # The bad quality projection visibly reaches the image edge; the selected
    # fallback overlay keeps both arm colors inside the body silhouette.
    line_colors = np.asarray([(255, 215, 0), (0, 140, 255)], dtype=np.uint8)

    def edge_line_count(image):
        color_mask = np.zeros(image.shape[:2], dtype=bool)
        for color in line_colors:
            color_mask |= np.all(image == color, axis=2)
        y, x = np.where(color_mask)
        edge = ((x < int(image.shape[1] * 0.04)) | (x >= int(image.shape[1] * 0.96))) & (y >= 60)
        return int(np.count_nonzero(edge))

    assert edge_line_count(bad_overlay) > 0
    assert edge_line_count(selected_overlay) == 0


def test_sha256_cache_concurrency_and_case_normalization(tmp_path):
    """Ensure _sha256_file handles case-insensitivity, non-existent files, and threads safely."""
    from concurrent.futures import ThreadPoolExecutor
    from pose_pipeline.adapters.wham_native import _sha256_file, _SHA256_CACHE

    test_file = tmp_path / "test_weight.bin"
    content = b"TexMotion adversarial test payload" * 1024
    test_file.write_bytes(content)

    import hashlib
    expected_hash = hashlib.sha256(content).hexdigest()

    # Case normalization check (Windows file system case-insensitivity)
    hash_normal = _sha256_file(str(test_file))
    hash_upper = _sha256_file(str(test_file).upper())
    assert hash_normal == expected_hash
    assert hash_upper == expected_hash

    # Concurrency stress test
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(_sha256_file, str(test_file)) for _ in range(32)]
        results = [f.result() for f in futures]
    assert all(r == expected_hash for r in results)

    # Missing file and invalid types
    assert _sha256_file(None) is None
    assert _sha256_file("") is None
    assert _sha256_file(tmp_path / "non_existent.pth") is None


def test_metadata_snapshot_isolation_and_aliasing_safety():
    """Frame-level metadata must be defensively isolated so mutating one frame cannot corrupt others."""
    from pose_pipeline.adapters.wham_native import NativeWHAMAdapter

    adapter = NativeWHAMAdapter()
    snapshot = adapter.get_metadata()
    assert isinstance(snapshot, dict)

    # Simulate adapter frame output construction
    frame_a = {"frameIndex": 0, "metadata": dict(snapshot or {})}
    frame_b = {"frameIndex": 1, "metadata": dict(snapshot or {})}

    # Mutating frame_a must NOT affect frame_b or the original snapshot
    frame_a["metadata"]["customPerFrameTag"] = "frame_0_mutated"
    assert "customPerFrameTag" not in frame_b["metadata"]
    assert "customPerFrameTag" not in snapshot


def test_fusion_nan_and_degenerate_safety():
    """fuse_wham_mediapipe must strictly return finite floats even on all-NaN or all-zero inputs."""
    frames, joints = 4, 33
    all_nan_wham = np.full((frames, joints, 3), np.nan, dtype=np.float64)
    all_nan_mp3d = np.full((frames, joints, 3), np.nan, dtype=np.float64)
    all_nan_mp2d = np.full((frames, joints, 2), np.nan, dtype=np.float64)
    zero_scores = np.zeros((frames, joints), dtype=np.float64)

    # Must complete without unhandled exceptions
    result = fuse_wham_mediapipe(
        all_nan_wham,
        mediapipe_3d=all_nan_mp3d,
        mediapipe_2d=all_nan_mp2d,
        score=zero_scores,
        wham_score=zero_scores,
        selection_mode="weighted",
    )

    assert result is not None
    assert np.all(np.isfinite(result.fused_3d))
    assert result.fused_3d.shape == (frames, joints, 3)
    assert np.all(np.isfinite(result.diagnostics["reprojectionErrors"]))
    assert np.all(np.isfinite(result.diagnostics["whamCorrections"]))
    assert np.all(np.isfinite(result.diagnostics["disagreementMeters"]))
    assert np.isfinite(result.diagnostics.mean_reprojection_error)
    assert np.isfinite(result.diagnostics.mean_wham_correction)

