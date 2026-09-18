"""
test_kinematics.py - Comprehensive tests for anatomical limits, bone lengths,
SO(3) operations, and SMPL-X forward kinematics.
"""

import numpy as np
import pytest

from pose_pipeline.kinematics import (
    BoneLengthModel,
    constrain_knee_flexion,
    quaternion_slerp,
    quaternion_multiply,
    normalize_quaternion,
    quaternion_to_matrix,
    matrix_to_quaternion,
    euler_to_quaternion,
    quaternion_to_euler,
    matrix_to_euler,
    forward_kinematics_smplx,
    apply_anatomical_joint_limits,
    SMPLX_JOINT_COUNT,
    SMPLX_PARENTS,
    SMPLX_DEFAULT_OFFSETS
)


def test_bone_length_calibration(sample_landmarks_3d):
    model = BoneLengthModel()
    # Provide 10 identical standing frames
    seqs = [sample_landmarks_3d.copy() for _ in range(10)]
    confs = [0.9 for _ in range(10)]

    model.calibrate_from_observations(seqs, confs)
    assert model.is_calibrated is True

    # Check that calibrated lengths match expected human proportions
    assert np.isclose(model.bone_lengths["thigh_l"], model.bone_lengths["thigh_r"], atol=1e-5)


def test_bone_length_calibration_insufficient_frames_defaults():
    """Verifies fallback to default anatomical bone lengths when fewer than 5 confident frames exist."""
    model = BoneLengthModel()
    seqs = [np.zeros((33, 3), dtype=np.float32) for _ in range(3)]
    confs = [0.9, 0.9, 0.9]  # Only 3 frames (< 5 required)

    model.calibrate_from_observations(seqs, confs)
    assert model.is_calibrated is True
    assert np.isclose(model.bone_lengths["thigh_l"], 0.42, atol=1e-4)
    assert np.isclose(model.bone_lengths["shin_l"], 0.42, atol=1e-4)
    assert np.isclose(model.bone_lengths["upperarm_l"], 0.28, atol=1e-4)


def test_enforce_limb_lengths_mediapipe(sample_landmarks_3d):
    model = BoneLengthModel()
    seqs = [sample_landmarks_3d.copy() for _ in range(10)]
    confs = [0.9 for _ in range(10)]
    model.calibrate_from_observations(seqs, confs)

    # Artificially shrink knee distance from hip (leg)
    noisy_lms = sample_landmarks_3d.copy()
    noisy_lms[25] = noisy_lms[23] + (noisy_lms[25] - noisy_lms[23]) * 0.5  # Shrunk to 50%
    shrunk_dist = np.linalg.norm(noisy_lms[25] - noisy_lms[23])
    assert shrunk_dist < 0.30

    # Artificially stretch left elbow distance from shoulder (arm)
    noisy_lms[13] = noisy_lms[11] + (noisy_lms[13] - noisy_lms[11]) * 2.0  # Stretched 200%

    # Apply bone length projection
    fixed_lms = model.enforce_limb_lengths(noisy_lms)
    restored_leg = np.linalg.norm(fixed_lms[25] - fixed_lms[23])
    restored_arm = np.linalg.norm(fixed_lms[13] - fixed_lms[11])

    assert np.isclose(restored_leg, model.bone_lengths["thigh_l"], atol=1e-4)
    assert np.isclose(restored_arm, model.bone_lengths["upperarm_l"], atol=1e-4)


def test_enforce_limb_lengths_smplx_22():
    model = BoneLengthModel()
    # Calibrate with standard human lengths
    model.bone_lengths["thigh_l"] = 0.42
    model.bone_lengths["shin_l"] = 0.40
    model.bone_lengths["upperarm_l"] = 0.28
    model.bone_lengths["forearm_l"] = 0.25
    model.bone_lengths["thigh_r"] = 0.42
    model.bone_lengths["shin_r"] = 0.40
    model.bone_lengths["upperarm_r"] = 0.28
    model.bone_lengths["forearm_r"] = 0.25
    model.is_calibrated = True

    # 22 SMPL-X joints
    joints = np.zeros((22, 3), dtype=np.float64)
    # L_Hip (1), L_Knee (4), L_Ankle (7)
    joints[1] = np.array([-0.10, 0.95, 0.0])
    joints[4] = np.array([-0.10, 0.70, 0.0])  # length = 0.25 (distorted)
    joints[7] = np.array([-0.10, 0.50, 0.0])  # length = 0.20 (distorted)

    # L_Shoulder (16), L_Elbow (18), L_Wrist (20)
    joints[16] = np.array([-0.20, 1.40, 0.0])
    joints[18] = np.array([-0.35, 1.40, 0.0])  # length = 0.15 (distorted)
    joints[20] = np.array([-0.50, 1.40, 0.0])  # length = 0.15 (distorted)

    fixed = model.enforce_limb_lengths(joints)

    thigh_len = np.linalg.norm(fixed[4] - fixed[1])
    shin_len = np.linalg.norm(fixed[7] - fixed[4])
    uarm_len = np.linalg.norm(fixed[18] - fixed[16])
    farm_len = np.linalg.norm(fixed[20] - fixed[18])

    assert np.isclose(thigh_len, 0.42, atol=1e-4)
    assert np.isclose(shin_len, 0.40, atol=1e-4)
    assert np.isclose(uarm_len, 0.28, atol=1e-4)
    assert np.isclose(farm_len, 0.25, atol=1e-4)


def test_constrain_knee_flexion():
    hip = np.array([0.0, 0.0, 0.0])
    # Knee bent backwards anatomically (Z > 0)
    knee_valid = np.array([0.0, -0.4, 0.1])
    ankle_valid = np.array([0.0, -0.8, 0.0])

    res = constrain_knee_flexion(hip, knee_valid, ankle_valid, is_left=True)
    assert np.allclose(res, knee_valid)

    # Hyperextended knee bent forward (Z < 0)
    knee_hyperextended = np.array([0.0, -0.4, -0.15])
    res_fixed = constrain_knee_flexion(hip, knee_hyperextended, ankle_valid, is_left=True)
    # The corrected knee must have eliminated the forward hyperextension (Z >= 0)
    assert res_fixed[2] >= 0.0


def test_quaternion_slerp():
    q0 = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    q1 = np.array([0.0, float(np.sin(np.pi / 4)), 0.0, float(np.cos(np.pi / 4))], dtype=np.float32)

    q_mid = quaternion_slerp(q0, q1, 0.5)
    expected_y = float(np.sin(np.pi / 8))
    expected_w = float(np.cos(np.pi / 8))

    assert np.isclose(q_mid[1], expected_y, atol=1e-3)
    assert np.isclose(q_mid[3], expected_w, atol=1e-3)


def test_quaternion_multiply():
    q0 = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    q1 = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    prod = quaternion_multiply(q0, q1)
    assert np.allclose(prod, q1)


def test_normalize_quaternion_safe():
    zero_q = np.array([0.0, 0.0, 0.0, 0.0])
    norm_q = normalize_quaternion(zero_q)
    assert np.allclose(norm_q, [0.0, 0.0, 0.0, 1.0])
    assert not np.any(np.isnan(norm_q))


def test_so3_converters_roundtrip():
    # Test Euler -> Quaternion -> Matrix -> Euler round-trip
    original_euler = np.array([20.0, -35.0, 45.0], dtype=np.float64)

    quat = euler_to_quaternion(original_euler)
    # Quaternion should be unit length
    assert np.isclose(np.linalg.norm(quat), 1.0, atol=1e-5)

    R = quaternion_to_matrix(quat)
    # Matrix should be orthogonal (det=1, R*R^T = I)
    assert np.isclose(np.linalg.det(R), 1.0, atol=1e-5)
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-5)

    recovered_quat = matrix_to_quaternion(R)
    # Sign ambiguity in quaternion (q and -q represent the same SO(3) element)
    dot = abs(float(np.dot(quat, recovered_quat)))
    assert np.isclose(dot, 1.0, atol=1e-4)

    recovered_euler = matrix_to_euler(R)
    assert np.allclose(original_euler, recovered_euler, atol=1e-3)


def test_forward_kinematics_smplx_identity():
    # 22 Identity quaternions [0, 0, 0, 1]
    rotations = np.zeros((SMPLX_JOINT_COUNT, 4), dtype=np.float64)
    rotations[:, 3] = 1.0

    root_pos = np.array([0.0, 1.0, 0.0])
    global_pos, global_rot = forward_kinematics_smplx(rotations, root_position=root_pos)

    assert global_pos.shape == (22, 3)
    assert global_rot.shape == (22, 4)
    # Pelvis at root_position
    assert np.allclose(global_pos[0], root_pos)

    # Spine1 is child of Pelvis
    expected_spine1 = root_pos + SMPLX_DEFAULT_OFFSETS[3]
    assert np.allclose(global_pos[3], expected_spine1, atol=1e-4)


def test_forward_kinematics_smplx_rotation():
    rotations = np.zeros((SMPLX_JOINT_COUNT, 4), dtype=np.float64)
    rotations[:, 3] = 1.0

    # Rotate Pelvis by 90 degrees around Y (turns body right)
    rotations[0] = euler_to_quaternion(np.array([0.0, 90.0, 0.0]))

    root_pos = np.array([0.0, 1.0, 0.0])
    global_pos, global_rot = forward_kinematics_smplx(rotations, root_position=root_pos)

    # Spine1 offset in local frame is [0, 0.15, 0]. Rotated 90 deg around Y is still [0, 0.15, 0]
    # L_Hip offset is [+0.08, -0.06, 0]. Rotated 90 deg around Y:
    # new offset has Y = -0.06
    l_hip_pos = global_pos[1]
    assert np.isclose(l_hip_pos[1], 1.0 - 0.06, atol=1e-3)
    assert np.isclose(abs(l_hip_pos[0]) + abs(l_hip_pos[2]), 0.08, atol=1e-3)


def test_apply_anatomical_joint_limits():
    # Construct rotations with extreme values
    rotations = np.zeros((3, SMPLX_JOINT_COUNT, 4), dtype=np.float64)
    rotations[:, :, 3] = 1.0

    # L_Knee (joint 4): set forward hyperextension Euler angle of -60 deg pitch
    rotations[1, 4] = euler_to_quaternion(np.array([-60.0, 0.0, 0.0]))

    clamped = apply_anatomical_joint_limits(rotations)

    # Clamped Euler pitch should be >= 0 (no hyperextension)
    clamped_euler = quaternion_to_euler(clamped[1, 4])
    assert clamped_euler[0] >= -1.0  # Margin of 1 deg
