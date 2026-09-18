"""
test_extractor_integration.py - Integration tests for video_pose_extractor.py, CLI, and leg kinematics.
"""

import os
import json
import subprocess
import sys
import math
from types import SimpleNamespace
import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import video_pose_extractor as extractor
from video_pose_extractor import (
    _SharedMediaPipeAuxiliary,
    inpaint_and_constrain_leg_kinematics,
    SMPLX_JOINT_COUNT,
)


def test_shared_mediapipe_auxiliary_facade_reuses_quality_detector_contract():
    """WHAM fusion can consume the native detector without a second graph."""
    class FakeDetector:
        name = "mediapipe"
        is_initialized = True

        def __init__(self):
            self.calls = []

        def detect(self, frame, timestamp_sec, frame_index=0):
            self.calls.append((frame, timestamp_sec, frame_index))
            return SimpleNamespace(
                landmarks_3d=[SimpleNamespace(x=1.0, y=2.0, z=3.0)],
                keypoints_2d=[SimpleNamespace(x=0.25, y=0.5, score=0.8)],
                raw_confidence=0.8,
            )

    detector = FakeDetector()
    facade = _SharedMediaPipeAuxiliary(detector)
    world, norm, confidence = facade.detect(np.zeros((2, 2, 3), dtype=np.uint8), 1250)

    assert detector.calls[0][1:] == (1.25, 0)
    np.testing.assert_allclose(world, [[1.0, 2.0, 3.0]])
    np.testing.assert_allclose(norm, [[0.25, 0.5, 0.8]])
    assert confidence == 0.8
    facade.close()  # shared facade must not close the owner
    assert detector.is_initialized is True


def test_synthetic_extraction_cli(tmp_path):
    output_json = str(tmp_path / "test_motion.json")
    script_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "video_pose_extractor.py"))

    # Run extraction in synthetic mode with explicit UTF-8 on Windows
    cmd = [
        sys.executable,
        script_path,
        "--synthetic",
        "--output", output_json,
        "--fps", "30.0"
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    assert res.returncode == 0, f"Extraction failed with stderr: {res.stderr}"
    assert os.path.exists(output_json)

    # Validate JSON Schema
    with open(output_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["frames"] == 60
    assert data["jointCount"] == 22
    assert "uncertaintyIntervals" in data
    assert isinstance(data["uncertaintyIntervals"], list)
    assert len(data["rootPositions"]) == 60
    assert len(data["flatLocalRotations"]) == 60 * 22
    assert len(data["timestamps"]) == 60
    assert len(data["confidences"]) == 60


def test_synthetic_extraction_stdout_jsonlines_protocol(tmp_path):
    """Verifies that CLI streams structured JSON Lines progress events to stdout for Unity Editor."""
    output_json = str(tmp_path / "protocol_motion.json")
    script_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "video_pose_extractor.py"))

    cmd = [
        sys.executable,
        script_path,
        "--synthetic",
        "--output", output_json
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    assert res.returncode == 0, f"CLI exited with error: {res.stderr}"

    # Verify stdout lines are valid JSON progress messages
    lines = [line.strip() for line in res.stdout.splitlines() if line.strip()]
    assert len(lines) >= 2, "No progress events streamed to stdout"

    stages = []
    for line in lines:
        try:
            payload = json.loads(line)
            assert "progress" in payload
            assert "stage" in payload
            assert 0.0 <= payload["progress"] <= 1.0
            stages.append(payload["stage"])
        except json.JSONDecodeError:
            pytest.fail(f"Non-JSON line found on stdout: {line}")

    assert "completed" in stages


def test_raised_kick_preservation_left_leg():
    """
    Verifies that high kicks or raised legs (knee above horizontal)
    are NOT forcibly snapped downward to -0.4.
    """
    frames = 5
    joints_seq = np.zeros((frames, SMPLX_JOINT_COUNT, 3), dtype=np.float64)

    # Pelvis at (0, 0.95, 0)
    for t in range(frames):
        joints_seq[t, 0] = np.array([0.0, 0.95, 0.0])
        joints_seq[t, 1] = np.array([-0.10, 0.95, 0.0])  # L_Hip
        joints_seq[t, 2] = np.array([0.10, 0.95, 0.0])   # R_Hip

        # Left leg raised in a forward kick:
        # Knee is at Y = 1.05 (ABOVE hip 0.95), Z = 0.40 (forward)
        joints_seq[t, 4] = np.array([-0.10, 1.05, 0.40])  # L_Knee
        joints_seq[t, 7] = np.array([-0.10, 1.15, 0.80])  # L_Ankle

        # Right leg standing: Knee at Y = 0.53, Ankle at Y = 0.11
        joints_seq[t, 5] = np.array([0.10, 0.53, 0.0])
        joints_seq[t, 8] = np.array([0.10, 0.11, 0.0])

    constrained = inpaint_and_constrain_leg_kinematics(joints_seq)

    assert constrained[2, 4, 1] >= 0.95, f"Raised kick knee was forced downward: Y={constrained[2, 4, 1]}"
    assert constrained[2, 4, 2] > 0.25, f"Raised kick forward extension lost: Z={constrained[2, 4, 2]}"


def test_raised_kick_preservation_right_leg():
    """Verifies that right leg forward high kicks are also preserved without downward snapping."""
    frames = 5
    joints_seq = np.zeros((frames, SMPLX_JOINT_COUNT, 3), dtype=np.float64)

    for t in range(frames):
        joints_seq[t, 0] = np.array([0.0, 0.95, 0.0])
        joints_seq[t, 1] = np.array([-0.10, 0.95, 0.0])
        joints_seq[t, 2] = np.array([0.10, 0.95, 0.0])

        # Standing Left leg
        joints_seq[t, 4] = np.array([-0.10, 0.53, 0.0])
        joints_seq[t, 7] = np.array([-0.10, 0.11, 0.0])

        # Right leg raised forward
        joints_seq[t, 5] = np.array([0.10, 1.05, 0.40])  # R_Knee
        joints_seq[t, 8] = np.array([0.10, 1.15, 0.80])  # R_Ankle

    constrained = inpaint_and_constrain_leg_kinematics(joints_seq)

    assert constrained[2, 5, 1] >= 0.95, f"Right kick knee was forced downward: Y={constrained[2, 5, 1]}"
    assert constrained[2, 5, 2] > 0.25, f"Right kick forward extension lost: Z={constrained[2, 5, 2]}"


def test_synthetic_extraction_with_uncertainty_and_sequence_fit(tmp_path):
    """
    Verifies that the end-to-end extractor pipeline produces valid JSON
    with uncertainty intervals and normalized quaternions using the unified sequence optimizer.
    """
    script_path = os.path.join(os.path.dirname(__file__), "..", "video_pose_extractor.py")
    output_json = str(tmp_path / "integrated_motion.json")

    cmd = [
        sys.executable,
        script_path,
        "--synthetic",
        "--output", output_json,
        "--fps", "30"
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    assert res.returncode == 0, f"Extraction failed: {res.stderr}"

    assert os.path.exists(output_json)
    with open(output_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["frames"] > 0
    assert data["jointCount"] == 22
    assert "uncertaintyIntervals" in data
    assert isinstance(data["uncertaintyIntervals"], list)
    assert len(data["localRotations"]) == data["frames"]

    # Verify quaternion normalization on all frames
    for t in range(data["frames"]):
        for j in range(22):
            q = data["localRotations"][t][j]
            norm = math.sqrt(q["x"]**2 + q["y"]**2 + q["z"]**2 + q["w"]**2)
            assert 0.98 <= norm <= 1.02, f"Unnormalized quaternion at frame {t}, joint {j}: norm={norm}"


def test_backend_cli_and_detector_name_export(tmp_path):
    """
    Verifies that the CLI accepts --backend argument and exports detectorName
    and backendRequested fields into the output JSON.
    """
    script_path = os.path.join(os.path.dirname(__file__), "..", "video_pose_extractor.py")
    output_json = str(tmp_path / "backend_motion.json")

    cmd = [
        sys.executable,
        script_path,
        "--synthetic",
        "--backend", "rtmpose",
        "--output", output_json
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    assert res.returncode == 0, f"Extraction failed: {res.stderr}"

    assert os.path.exists(output_json)
    with open(output_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert "detectorName" in data
    assert "backendRequested" in data
    assert data["backendRequested"] == "rtmpose"
    assert data["detectorName"] == "synthetic"


def test_runtime_provenance_contract_distinguishes_optional_stage_truth():
    """Result-card fields must expose execution state, not file presence."""
    metadata = {
        "adapterImplementation": "texmotion_native_wham",
        "nativeWhamCore": True,
        "imageFeatureBackbonePath": "vitpose-huge.pth",
        "imageFeatureRunnerStatus": "fallback",
        "imageFeatureRunnerError": "No verified HMR2 runner",
        "inputDetector": "rtmpose",
        "whamStageDiagnostics": {
            "image_feature_extractor_for_backbone": {
                "available": False,
                "expected": "verified HMR2/ViTPose runner",
            },
        },
        "missingOptionalAssets": ["camera_motion_or_slam"],
        "optionalErrors": ["DPVO runner is not configured"],
    }

    assert hasattr(extractor, "build_runtime_provenance")
    provenance = extractor.build_runtime_provenance(
        "wham",
        "wham",
        metadata,
        backend_fallback=False,
        fallback_reason=None,
    )

    assert provenance["backendRequested"] == "wham"
    assert provenance["backendActual"] == "wham"
    assert provenance["officialRunnerStatus"] == "active"
    assert provenance["hmr2ImageFeaturesStatus"] == "fallback"
    assert provenance["vitpose2DStatus"] == "not_verified"
    assert provenance["fallbackReason"] is None
    diagnostics = provenance["optionalStageDiagnostics"]
    assert {item["stage"] for item in diagnostics} == {
        "image_feature_extractor_for_backbone",
        "camera_motion_or_slam",
        "optional_error",
    }
    assert all(set(("stage", "status", "reason")) <= set(item) for item in diagnostics)


def test_synthetic_output_contains_backward_compatible_runtime_provenance(tmp_path):
    """Legacy synthetic extraction emits the same stable result-card keys."""
    output_json = str(tmp_path / "provenance_motion.json")
    script_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "video_pose_extractor.py"))
    result = subprocess.run(
        [sys.executable, script_path, "--synthetic", "--backend", "wham", "--output", output_json],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    assert result.returncode == 0, result.stderr
    with open(output_json, "r", encoding="utf-8") as stream:
        payload = json.load(stream)

    for key in (
        "backendRequested",
        "backendActual",
        "officialRunnerStatus",
        "hmr2ImageFeaturesStatus",
        "vitpose2DStatus",
        "fallbackReason",
        "optionalStageDiagnostics",
    ):
        assert key in payload
        assert key in payload["backendMetadata"]
    assert payload["backendRequested"] == "wham"
    assert payload["backendActual"] == "synthetic"
    assert payload["fallbackReason"] is None
    assert isinstance(payload["optionalStageDiagnostics"], list)


def test_runtime_provenance_marks_quality_stages_as_fallback_after_backend_switch():
    provenance = extractor.build_runtime_provenance(
        "wham",
        "mediapipe",
        {},
        backend_fallback=True,
        fallback_reason="WHAM initialization failed",
    )

    assert provenance["officialRunnerStatus"] == "fallback"
    assert provenance["hmr2ImageFeaturesStatus"] == "fallback"
    assert provenance["vitpose2DStatus"] == "fallback"
    assert provenance["fallbackReason"] == "WHAM initialization failed"
