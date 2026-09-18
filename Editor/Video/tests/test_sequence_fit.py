"""
test_sequence_fit.py - Tests for temporal fitting, multi-hypothesis crossing resolution, and contact hysteresis.
"""

import numpy as np
import pytest
from pose_pipeline.sequence_fit import SequenceOptimizer
from pose_pipeline.hypotheses import HypothesisScore, score_hypotheses


def test_multi_hypothesis_leg_crossing_left_front(sample_landmarks_3d):
    """Verifies Hypothesis A: Left leg enters in front and wins crossing resolution."""
    optimizer = SequenceOptimizer(fps=30.0)
    num_frames = 15

    frames_3d = []
    frames_2d = []
    confidences = [0.85] * num_frames

    # Construct a motion where left and right ankles cross at frame 7
    for t in range(num_frames):
        lms3d = sample_landmarks_3d.copy()
        lms2d = np.zeros((33, 2), dtype=np.float32)

        # Left ankle enters in front (Z = -0.15) and moves rightward (+X)
        # Right ankle stays behind (Z = 0.05) and moves leftward (-X)
        alpha = t / (num_frames - 1)
        x_left = 0.40 + alpha * 0.20
        x_right = 0.60 - alpha * 0.20

        # At t=7, both ankles have almost same X (~0.50) -> 2D overlap!
        lms2d[27] = np.array([x_left, 0.85])
        lms2d[28] = np.array([x_right, 0.85])
        lms2d[23] = np.array([0.45, 0.50])
        lms2d[24] = np.array([0.55, 0.50])

        lms3d[27] = np.array([x_left - 0.5, -0.84, -0.15])
        lms3d[28] = np.array([x_right - 0.5, -0.84, 0.05])

        frames_3d.append(lms3d)
        frames_2d.append(lms2d)

    resolved_3d, uncertainties = optimizer.resolve_leg_crossings_multi_hypothesis(
        frames_3d, frames_2d, confidences
    )

    assert len(resolved_3d) == num_frames
    assert len(uncertainties) >= 1
    crossing = uncertainties[0]
    assert crossing.reason == "leg_crossing_ambiguity"
    assert crossing.recommended_action == "swap_crossing"
    # Overlap occurs around frame 7 (0-indexed)
    assert crossing.start_frame <= 8 and crossing.end_frame >= 6
    # Left ankle remains in front of Right ankle at crossing point
    assert resolved_3d[7][27][2] < resolved_3d[7][28][2]


def test_multi_hypothesis_leg_crossing_right_front(sample_landmarks_3d):
    """Verifies Hypothesis B: Right leg enters in front and wins crossing resolution."""
    optimizer = SequenceOptimizer(fps=30.0)
    num_frames = 15

    frames_3d = []
    frames_2d = []
    confidences = [0.85] * num_frames

    for t in range(num_frames):
        lms3d = sample_landmarks_3d.copy()
        lms2d = np.zeros((33, 2), dtype=np.float32)

        alpha = t / (num_frames - 1)
        x_left = 0.40 + alpha * 0.20
        x_right = 0.60 - alpha * 0.20

        lms2d[27] = np.array([x_left, 0.85])
        lms2d[28] = np.array([x_right, 0.85])
        lms2d[23] = np.array([0.45, 0.50])
        lms2d[24] = np.array([0.55, 0.50])

        # Right ankle enters in front (Z = -0.20), Left stays behind (Z = 0.10)
        lms3d[27] = np.array([x_left - 0.5, -0.84, 0.10])
        lms3d[28] = np.array([x_right - 0.5, -0.84, -0.20])

        frames_3d.append(lms3d)
        frames_2d.append(lms2d)

    resolved_3d, uncertainties = optimizer.resolve_leg_crossings_multi_hypothesis(
        frames_3d, frames_2d, confidences
    )

    assert len(resolved_3d) == num_frames
    assert len(uncertainties) >= 1
    # Right ankle remains in front of Left ankle at crossing point
    assert resolved_3d[7][28][2] < resolved_3d[7][27][2]


def test_sequence_optimizer_short_sequence():
    """Verifies that SequenceOptimizer safely handles short sequences (< 3 frames) without errors."""
    optimizer = SequenceOptimizer(fps=30.0)
    f3d = [np.zeros((33, 3))]
    f2d = [np.zeros((33, 2))]
    res, unc = optimizer.resolve_leg_crossings_multi_hypothesis(f3d, f2d, [0.9])
    assert len(res) == 1
    assert len(unc) == 0


def test_sequence_optimizer_no_crossing(sample_landmarks_3d):
    """Verifies that normal parallel walking with clear leg separation detects 0 crossing hazards."""
    optimizer = SequenceOptimizer(fps=30.0)
    num_frames = 10
    f3d = []
    f2d = []
    for _ in range(num_frames):
        lms3d = sample_landmarks_3d.copy()
        lms2d = np.zeros((33, 2), dtype=np.float32)
        # Feet well separated (X = 0.35 and 0.65, distance = 0.30 >> 0.06 threshold)
        lms2d[27] = np.array([0.35, 0.85])
        lms2d[28] = np.array([0.65, 0.85])
        lms2d[23] = np.array([0.45, 0.50])
        lms2d[24] = np.array([0.55, 0.50])
        f3d.append(lms3d)
        f2d.append(lms2d)

    res, uncertainties = optimizer.resolve_leg_crossings_multi_hypothesis(f3d, f2d, [0.9] * num_frames)
    assert len(res) == num_frames
    assert len(uncertainties) == 0


def test_quaternion_so3_smoothing():
    optimizer = SequenceOptimizer(fps=30.0)
    frames = 10
    joints = 22

    # Create rotations with an artificial hemisphere flip (q and -q)
    rotations = np.zeros((frames, joints, 4), dtype=np.float32)
    for t in range(frames):
        for j in range(joints):
            # Identity quaternion [0, 0, 0, 1]
            rotations[t, j] = np.array([0.0, 0.0, 0.0, 1.0])

    # Flip sign at frame 4
    rotations[4, 1] = np.array([0.0, 0.0, 0.0, -1.0])

    smoothed = optimizer.smooth_quaternions_so3(rotations)

    # After smoothing, all dot products between consecutive frames should be positive
    for t in range(1, frames):
        dot = np.dot(smoothed[t - 1, 1], smoothed[t, 1])
        assert dot >= 0.0, f"Discontinuous quaternion flip detected at frame {t}"


def test_foot_contact_hysteresis():
    optimizer = SequenceOptimizer(fps=30.0)
    frames = 10
    root_pos = np.zeros((frames, 3))
    l_foot = np.zeros((frames, 3))
    r_foot = np.zeros((frames, 3))

    # Left foot stationary for first 5 frames, then moving fast (> 0.5 m/s)
    for t in range(frames):
        if t >= 5:
            l_foot[t] = np.array([0.0, 0.0, (t - 4) * 0.1])  # moving fast
        else:
            l_foot[t] = np.array([0.0, 0.0, 0.0])  # stationary

    l_contact, r_contact = optimizer.apply_foot_contact_hysteresis(root_pos, l_foot, r_foot, vel_threshold=0.15)

    assert bool(l_contact[1]) is True   # stationary frame
    assert bool(l_contact[8]) is False  # moving fast


def test_foot_contact_hysteresis_chattering_prevention():
    """Verifies that foot speed within the hysteresis band (0.15 to 0.225 m/s) does not chatter."""
    optimizer = SequenceOptimizer(fps=30.0)
    frames = 12
    root = np.zeros((frames, 3))
    l_foot = np.zeros((frames, 3))
    r_foot = np.zeros((frames, 3))

    # Speed at 0.18 m/s (between 0.15 and 0.225 m/s)
    speed = 0.18
    dt = 1.0 / 30.0
    for t in range(1, frames):
        l_foot[t] = l_foot[t - 1] + np.array([0.0, 0.0, speed * dt])

    l_contact, _ = optimizer.apply_foot_contact_hysteresis(root, l_foot, r_foot, vel_threshold=0.15)
    # Once transitioning to swing (False), it should not oscillate back to contact
    assert bool(l_contact[0]) is False
    assert not np.any(l_contact[2:])


def test_resolve_arm_occlusion_front_of_face(sample_landmarks_3d):
    """Verifies that wrist passing in front of the face is resolved as front hypothesis with uncertainty."""
    optimizer = SequenceOptimizer(fps=30.0)
    num_frames = 12
    frames_3d = []
    frames_2d = []
    confidences = [0.9] * num_frames

    for t in range(num_frames):
        lms3d = sample_landmarks_3d.copy()
        # Left wrist (15 in MP33) moves close to head (0 in MP33)
        # Entry (t=0..3): wrist is in front of head (Z is more negative in MP33)
        # Danger zone (t=4..7): wrist overlaps head in XY
        lms3d[0] = np.array([0.0, 0.40, 0.0]) # Head
        if t < 4:
            lms3d[15] = np.array([0.0, 0.38, -0.15]) # Front of head
        elif t <= 7:
            lms3d[15] = np.array([0.02, 0.39, 0.05]) # Ambiguous/occluded Z
        else:
            lms3d[15] = np.array([0.0, 0.38, -0.14]) # Exit to front

        frames_3d.append(lms3d)
        frames_2d.append(np.zeros((33, 2)))

    resolved_3d, uncertainties = optimizer.resolve_arm_occlusion_multi_hypothesis(
        frames_3d, frames_2d, confidences
    )

    assert len(resolved_3d) == num_frames
    assert len(uncertainties) >= 1
    assert "arm_left_head_occlusion" in uncertainties[0].reason
    # At crossing center (t=5), left wrist Z should be in front of head (smaller Z in MP33)
    assert resolved_3d[5][15][2] < resolved_3d[5][0][2]


def test_resolve_arm_occlusion_behind_head(sample_landmarks_3d):
    """Verifies that wrist going behind the head is resolved as behind hypothesis with uncertainty."""
    optimizer = SequenceOptimizer(fps=30.0)
    num_frames = 12
    frames_3d = []
    frames_2d = []
    confidences = [0.9] * num_frames

    for t in range(num_frames):
        lms3d = sample_landmarks_3d.copy()
        lms3d[0] = np.array([0.0, 0.40, 0.0]) # Head
        # Right wrist (16 in MP33) goes behind head
        # Entry (t=0..3): wrist is behind head (Z is positive in MP33)
        if t < 4:
            lms3d[16] = np.array([0.0, 0.38, 0.18]) # Behind head
        elif t <= 7:
            lms3d[16] = np.array([0.01, 0.39, -0.02]) # Ambiguous
        else:
            lms3d[16] = np.array([0.0, 0.38, 0.16]) # Exit to behind

        frames_3d.append(lms3d)
        frames_2d.append(np.zeros((33, 2)))

    resolved_3d, uncertainties = optimizer.resolve_arm_occlusion_multi_hypothesis(
        frames_3d, frames_2d, confidences
    )

    assert len(resolved_3d) == num_frames
    assert len(uncertainties) >= 1
    assert "arm_right_head_occlusion" in uncertainties[0].reason
    # At crossing center (t=5), right wrist Z should be behind head (larger Z in MP33)
    assert resolved_3d[5][16][2] > resolved_3d[5][0][2]


def test_unified_sequence_optimizer_pipeline(sample_smplx_22_joints):
    """Verifies that SequenceOptimizer.optimize_sequence runs the full Phase 4 pipeline correctly."""
    from pose_pipeline.kinematics import BoneLengthModel
    optimizer = SequenceOptimizer(fps=30.0)
    num_frames = 15

    joints_seq = np.zeros((num_frames, 22, 3), dtype=np.float64)
    norm_lms_seq = [np.zeros((33, 3), dtype=np.float64) for _ in range(num_frames)]
    confidences = [0.9] * num_frames

    for t in range(num_frames):
        joints_seq[t] = sample_smplx_22_joints.copy()
        # Introduce an intentional forward knee hyperextension (Z < 0 in SMPL-X is backward bending, Z > 0 is forward bend)
        # Set knee in front of hip-ankle line
        joints_seq[t, 4] = np.array([0.1, -0.4, 0.15]) # Overextended knee

    bone_model = BoneLengthModel()
    # Calibrate bone model with initial poses
    bone_model.calibrate_from_observations([j for j in joints_seq], confidences)

    opt_joints, uncertainties, (l_cont, r_cont) = optimizer.optimize_sequence(
        joints_seq, norm_lms_seq, confidences, bone_model=bone_model
    )

    assert opt_joints.shape == (num_frames, 22, 3)
    assert len(l_cont) == num_frames
    assert len(r_cont) == num_frames
    # Overextended knee should be constrained
    assert opt_joints[0, 4, 2] <= joints_seq[0, 4, 2]


def test_unified_sequence_optimizer_empty():
    """Verifies that empty sequences return cleanly without errors."""
    optimizer = SequenceOptimizer(fps=30.0)
    opt_joints, unc, (l_c, r_c) = optimizer.optimize_sequence(
        np.zeros((0, 22, 3)), [], []
    )
    assert len(opt_joints) == 0
    assert len(unc) == 0
    assert len(l_c) == 0
    assert len(r_c) == 0


def test_hypothesis_scoring_keeps_explicit_provenance_and_alternatives():
    """Scoring exposes the evidence used instead of returning only a depth guess."""
    candidates = [
        {"name": "left_leg_front", "pose": np.zeros((2, 3)), "evidence": {"wham": 0.50, "reprojection": 0.25}},
        {"name": "right_leg_front", "pose": np.ones((2, 3)), "evidence": {"wham": 0.49, "reprojection": 0.26}},
    ]

    result = score_hypotheses(candidates, ambiguity_margin=0.05)

    assert isinstance(result.scores[0], HypothesisScore)
    assert result.selected.name == "left_leg_front"
    assert result.alternatives[0].name == "right_leg_front"
    assert result.ambiguous is True
    assert result.selected.provenance["evidence"] == {"wham": 0.50, "reprojection": 0.25}


def test_crossing_uncertainty_serializes_alternatives_and_scores(sample_landmarks_3d):
    """An unresolved crossing carries ranked hypotheses and their provenance."""
    optimizer = SequenceOptimizer(fps=30.0)
    frames_3d = []
    frames_2d = []
    for _ in range(6):
        frame = sample_landmarks_3d.copy()
        frame[27, 2] = 0.0
        frame[28, 2] = 0.0
        frames_3d.append(frame)
        landmarks = np.zeros((33, 2), dtype=np.float64)
        landmarks[27] = [0.5, 0.8]
        landmarks[28] = [0.5, 0.8]
        frames_2d.append(landmarks)

    _, uncertainties = optimizer.resolve_leg_crossings_multi_hypothesis(
        frames_3d, frames_2d, [0.9] * len(frames_3d)
    )

    assert uncertainties
    interval = uncertainties[0]
    assert interval.primary_hypothesis in {"left_leg_front", "right_leg_front"}
    assert interval.alternative_hypotheses
    payload = interval.to_dict()
    assert payload["primaryHypothesis"] == interval.primary_hypothesis
    assert payload["alternativeHypotheses"] == list(interval.alternative_hypotheses)
    assert len(payload["hypothesisScores"]) >= 2
    assert payload["provenance"]["type"] == "leg_crossing"


def test_temporal_penalties_use_real_dt_and_report_contact_sliding(sample_smplx_22_joints):
    """Biomechanical diagnostics include bone, ROM, velocity, acceleration and contact terms."""
    optimizer = SequenceOptimizer(fps=10.0)
    sequence = np.stack([sample_smplx_22_joints.copy() for _ in range(4)])
    sequence[2, 7, 0] += 0.2  # bone-length outlier and a velocity/acceleration spike
    sequence[2, 10, 0] += 0.2  # the foot itself slides while contact is held
    contacts = (np.array([False, True, True, False]), np.zeros(4, dtype=bool))

    penalties = optimizer.compute_temporal_penalties(sequence, contact_flags=contacts)

    assert set(("bone_length", "joint_limits", "velocity", "acceleration", "contact", "sliding")) <= set(penalties)
    assert penalties["bone_length"] > 0.0
    assert penalties["velocity"] > 0.0
    assert penalties["acceleration"] > 0.0
    assert penalties["sliding"] > 0.0
    assert penalties["dt"] == pytest.approx(0.1)


def test_inverse_pose_is_not_forced_upright(sample_smplx_22_joints):
    """Position fitting must preserve an inverted body instead of applying an upright prior."""
    optimizer = SequenceOptimizer(fps=30.0)
    pose = sample_smplx_22_joints.copy()
    pose[:, 1] *= -1.0
    sequence = np.stack([pose, pose.copy(), pose.copy()])

    optimized, _, _ = optimizer.optimize_sequence(sequence, [None] * 3, [0.9] * 3)

    assert optimized[1, 15, 1] < optimized[1, 10, 1]


def test_sideways_pose_is_not_forced_to_world_up(sample_smplx_22_joints):
    """The optimizer does not rotate a valid sideways sequence toward a standing template."""
    optimizer = SequenceOptimizer(fps=30.0)
    pose = sample_smplx_22_joints.copy()
    # Rotate the whole pose around Z so the torso lies mostly along X.
    angle = np.pi / 2.0
    rot_z = np.array([[np.cos(angle), -np.sin(angle), 0.0], [np.sin(angle), np.cos(angle), 0.0], [0.0, 0.0, 1.0]])
    pose = pose @ rot_z.T
    sequence = np.stack([pose, pose.copy(), pose.copy()])

    optimized, _, _ = optimizer.optimize_sequence(sequence, [None] * 3, [0.9] * 3)

    # Head-to-pelvis direction remains horizontal after fitting.
    direction = optimized[1, 15] - optimized[1, 0]
    assert abs(direction[0]) > abs(direction[1])
