"""
kinematics.py - Anatomical joint limits, bone length preservation, and forward kinematics.

Ensures extracted poses respect human biomechanical constraints:
- Bone length invariance across frames (limbs cannot stretch or shrink).
- Anatomical angular limits (knees do not invert forward, elbows do not hyper-extend).
- SO(3) rotation converters (Euler, quaternion, rotation matrix).
- SMPL-X 22 Forward Kinematics.
"""

from typing import Dict, List, Tuple, Optional
import math
import numpy as np

# SMPL-X 22 joint hierarchy and names
SMPLX_JOINT_COUNT = 22

SMPLX_PARENTS = [
    -1,  # 0: Pelvis
     0,  # 1: L_Hip
     0,  # 2: R_Hip
     0,  # 3: Spine1
     1,  # 4: L_Knee
     2,  # 5: R_Knee
     3,  # 6: Spine2
     4,  # 7: L_Ankle
     5,  # 8: R_Ankle
     6,  # 9: Spine3
     7,  # 10: L_Foot
     8,  # 11: R_Foot
     9,  # 12: Neck
     9,  # 13: L_Collar
     9,  # 14: R_Collar
    12,  # 15: Head
    13,  # 16: L_Shoulder
    14,  # 17: R_Shoulder
    16,  # 18: L_Elbow
    17,  # 19: R_Elbow
    18,  # 20: L_Wrist
    19,  # 21: R_Wrist
]

SMPLX_JOINT_NAMES = [
    "Pelvis", "L_Hip", "R_Hip", "Spine1", "L_Knee", "R_Knee", "Spine2",
    "L_Ankle", "R_Ankle", "Spine3", "L_Foot", "R_Foot", "Neck", "L_Collar",
    "R_Collar", "Head", "L_Shoulder", "R_Shoulder", "L_Elbow", "R_Elbow",
    "L_Wrist", "R_Wrist"
]

# Standard neutral T-pose offsets from parent (meters in SMPL-X frame)
# +X: Left, -X: Right, +Y: Up, +Z: Forward
SMPLX_DEFAULT_OFFSETS = [
    np.array([ 0.00,  0.00,  0.00], dtype=np.float64),  # 0: Pelvis
    np.array([ 0.08, -0.06,  0.00], dtype=np.float64),  # 1: L_Hip
    np.array([-0.08, -0.06,  0.00], dtype=np.float64),  # 2: R_Hip
    np.array([ 0.00,  0.11, -0.01], dtype=np.float64),  # 3: Spine1
    np.array([ 0.00, -0.38,  0.00], dtype=np.float64),  # 4: L_Knee
    np.array([ 0.00, -0.38,  0.00], dtype=np.float64),  # 5: R_Knee
    np.array([ 0.00,  0.12, -0.01], dtype=np.float64),  # 6: Spine2
    np.array([ 0.00, -0.39,  0.00], dtype=np.float64),  # 7: L_Ankle
    np.array([ 0.00, -0.39,  0.00], dtype=np.float64),  # 8: R_Ankle
    np.array([ 0.00,  0.12,  0.01], dtype=np.float64),  # 9: Spine3
    np.array([ 0.00, -0.07,  0.10], dtype=np.float64),  # 10: L_Foot
    np.array([ 0.00, -0.07,  0.10], dtype=np.float64),  # 11: R_Foot
    np.array([ 0.00,  0.18, -0.01], dtype=np.float64),  # 12: Neck
    np.array([ 0.08,  0.10, -0.02], dtype=np.float64),  # 13: L_Collar
    np.array([-0.08,  0.10, -0.02], dtype=np.float64),  # 14: R_Collar
    np.array([ 0.00,  0.09,  0.02], dtype=np.float64),  # 15: Head
    np.array([ 0.11,  0.00,  0.00], dtype=np.float64),  # 16: L_Shoulder
    np.array([-0.11,  0.00,  0.00], dtype=np.float64),  # 17: R_Shoulder
    np.array([ 0.25,  0.00,  0.00], dtype=np.float64),  # 18: L_Elbow
    np.array([-0.25,  0.00,  0.00], dtype=np.float64),  # 19: R_Elbow
    np.array([ 0.23,  0.00,  0.00], dtype=np.float64),  # 20: L_Wrist
    np.array([-0.23,  0.00,  0.00], dtype=np.float64),  # 21: R_Wrist
]

# Biomechanical joint limit ranges (degrees)
JOINT_LIMITS_DEG = {
    "Knee_Flexion": (0.0, 155.0),     # strictly flexes backward (0 = straight, 155 = max bend)
    "Knee_Twist": (-10.0, 10.0),
    "Elbow_Flexion": (0.0, 150.0),    # strictly flexes forward
    "Elbow_Twist": (-30.0, 30.0),
    "Hip_Flexion": (-30.0, 130.0),    # extension to high kick
    "Hip_Adduction": (-45.0, 55.0),   # leg crossing up to 55 deg
    "Hip_Twist": (-45.0, 45.0),
    "Shoulder_Flexion": (-60.0, 180.0),
    "Shoulder_Abduction": (-30.0, 180.0),
    "Shoulder_Twist": (-90.0, 90.0),
}


# ============================================================================
# SO(3) Rotation Converters ([x, y, z, w] quaternion convention)
# ============================================================================

def normalize_quaternion(q: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Safely normalizes quaternion [x, y, z, w]. Canonical form: w >= 0."""
    q_arr = np.asarray(q, dtype=np.float64)
    norm = np.linalg.norm(q_arr)
    if norm < eps:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    q_norm = q_arr / norm
    if q_norm[3] < 0.0:
        q_norm = -q_norm
    return q_norm


def quaternion_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Multiplies two quaternions q = q1 * q2 in [x, y, z, w] order."""
    x1, y1, z1, w1 = np.asarray(q1, dtype=np.float64)
    x2, y2, z2, w2 = np.asarray(q2, dtype=np.float64)
    return np.array([
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
        w1*w2 - x1*x2 - y1*y2 - z1*z2
    ], dtype=np.float64)


def quaternion_conjugate(q: np.ndarray) -> np.ndarray:
    """Returns conjugate [-x, -y, -z, w] of quaternion."""
    x, y, z, w = np.asarray(q, dtype=np.float64)
    return np.array([-x, -y, -z, w], dtype=np.float64)


def quaternion_slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    """Spherical linear interpolation between two quaternions on SO(3)."""
    q0 = normalize_quaternion(q0)
    q1 = normalize_quaternion(q1)

    dot = np.dot(q0, q1)
    if dot < 0.0:
        q1 = -q1
        dot = -dot

    dot = np.clip(dot, -1.0, 1.0)
    if dot > 0.9995:
        res = q0 + t * (q1 - q0)
        return normalize_quaternion(res)

    theta_0 = math.acos(dot)
    theta = theta_0 * t
    sin_theta = math.sin(theta)
    sin_theta_0 = math.sin(theta_0)

    s0 = math.cos(theta) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0
    res = (s0 * q0) + (s1 * q1)
    return normalize_quaternion(res)


def quaternion_to_matrix(q: np.ndarray) -> np.ndarray:
    """Converts unit quaternion [x, y, z, w] to 3x3 rotation matrix."""
    q_norm = normalize_quaternion(q)
    x, y, z, w = q_norm

    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    wx = w * x
    wy = w * y
    wz = w * z

    return np.array([
        [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz),       2.0 * (xz + wy)],
        [2.0 * (xy + wz),       1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
        [2.0 * (xz - wy),       2.0 * (yz + wx),       1.0 - 2.0 * (xx + yy)]
    ], dtype=np.float64)


def matrix_to_quaternion(R: np.ndarray) -> np.ndarray:
    """Converts 3x3 rotation matrix to unit quaternion [x, y, z, w] with w >= 0."""
    R = np.asarray(R, dtype=np.float64)
    trace = R[0, 0] + R[1, 1] + R[2, 2]

    if trace > 0.0:
        s = 0.5 / math.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
        s = 2.0 * math.sqrt(max(0.0, 1.0 + R[0, 0] - R[1, 1] - R[2, 2]))
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * math.sqrt(max(0.0, 1.0 + R[1, 1] - R[0, 0] - R[2, 2]))
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * math.sqrt(max(0.0, 1.0 + R[2, 2] - R[0, 0] - R[1, 1]))
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s

    return normalize_quaternion(np.array([x, y, z, w], dtype=np.float64))


def euler_to_quaternion(euler_deg: np.ndarray, order: str = "xyz") -> np.ndarray:
    """
    Converts intrinsic Euler angles in degrees to unit quaternion [x, y, z, w].
    Default order: XYZ (roll around X, pitch around Y, yaw around Z).
    """
    rx, ry, rz = np.radians(np.asarray(euler_deg, dtype=np.float64))

    cx = math.cos(rx * 0.5)
    sx = math.sin(rx * 0.5)
    cy = math.cos(ry * 0.5)
    sy = math.sin(ry * 0.5)
    cz = math.cos(rz * 0.5)
    sz = math.sin(rz * 0.5)

    if order.lower() == "xyz":
        x = sx * cy * cz + cx * sy * sz
        y = cx * sy * cz - sx * cy * sz
        z = cx * cy * sz + sx * sy * cz
        w = cx * cy * cz - sx * sy * sz
    else:
        # Generic composition via rotation matrices
        Rx = np.array([[1, 0, 0], [0, math.cos(rx), -math.sin(rx)], [0, math.sin(rx), math.cos(rx)]])
        Ry = np.array([[math.cos(ry), 0, math.sin(ry)], [0, 1, 0], [-math.sin(ry), 0, math.cos(ry)]])
        Rz = np.array([[math.cos(rz), -math.sin(rz), 0], [math.sin(rz), math.cos(rz), 0], [0, 0, 1]])
        R = Rx @ Ry @ Rz
        return matrix_to_quaternion(R)

    return normalize_quaternion(np.array([x, y, z, w], dtype=np.float64))


def quaternion_to_euler(q: np.ndarray) -> np.ndarray:
    """
    Converts unit quaternion [x, y, z, w] to XYZ Euler angles in degrees [-180, 180].
    """
    R = quaternion_to_matrix(q)
    return matrix_to_euler(R)


def matrix_to_euler(R: np.ndarray) -> np.ndarray:
    """
    Extracts XYZ Euler angles in degrees from 3x3 rotation matrix.
    Guarded against gimbal lock at pitch = +-90 deg.
    """
    R = np.asarray(R, dtype=np.float64)
    # R[0, 2] = sin(pitch)
    sin_pitch = np.clip(R[0, 2], -1.0, 1.0)
    pitch = math.asin(sin_pitch)

    if abs(sin_pitch) < 0.9999:
        roll = math.atan2(-R[1, 2], R[2, 2])
        yaw = math.atan2(-R[0, 1], R[0, 0])
    else:
        # Gimbal lock
        roll = math.atan2(R[2, 1], R[1, 1])
        yaw = 0.0

    return np.degrees(np.array([roll, pitch, yaw], dtype=np.float64))


# ============================================================================
# Forward Kinematics for SMPL-X 22 Hierarchy
# ============================================================================

def forward_kinematics_smplx(
    local_rotations: np.ndarray,
    bone_offsets: Optional[List[np.ndarray]] = None,
    root_position: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculates 3D global joint positions and orientations for SMPL-X 22 skeleton.
    local_rotations: (22, 4) local quaternions [x, y, z, w].
    bone_offsets: Optional list of 22 rest offset vectors from parent. Defaults to SMPLX_DEFAULT_OFFSETS.
    root_position: Optional (3,) world position of Pelvis (joint 0).
    Returns:
        global_positions: (22, 3) world coordinates of each joint in meters.
        global_rotations: (22, 4) world orientation quaternions.
    """
    if bone_offsets is None:
        bone_offsets = SMPLX_DEFAULT_OFFSETS
    if root_position is None:
        root_position = np.zeros(3, dtype=np.float64)

    global_positions = np.zeros((SMPLX_JOINT_COUNT, 3), dtype=np.float64)
    global_rotations = np.zeros((SMPLX_JOINT_COUNT, 4), dtype=np.float64)

    # Joint 0: Pelvis
    q_root = normalize_quaternion(local_rotations[0])
    global_rotations[0] = q_root
    global_positions[0] = np.asarray(root_position, dtype=np.float64)

    for j in range(1, SMPLX_JOINT_COUNT):
        p = SMPLX_PARENTS[j]
        q_parent_global = global_rotations[p]
        R_parent = quaternion_to_matrix(q_parent_global)

        # Global offset = R_parent * offset_local
        offset_world = R_parent @ bone_offsets[j]
        global_positions[j] = global_positions[p] + offset_world

        # Global rotation = q_parent_global * q_local
        q_local = normalize_quaternion(local_rotations[j])
        global_rotations[j] = normalize_quaternion(quaternion_multiply(q_parent_global, q_local))

    return global_positions, global_rotations


# ============================================================================
# Anatomical Joint Constraints & Biomechanical Limiting
# ============================================================================

def constrain_knee_flexion(
    hip: np.ndarray,
    knee: np.ndarray,
    ankle: np.ndarray,
    is_left: bool,
    flexion_axis_normal: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Biomechanically constrains knee joint:
    Prevents knee hyperextension (forward backward bending / bird legs).
    In human anatomy, knee only flexes backwards (heel towards gluteus).
    In TexMotion coordinate system (+Y up, +Z forward):
    When knee flexes, the knee joint projects forward (+Z) and foot moves backward (-Z).
    If the knee bends backwards relative to thigh-shin plane, pulls it to anatomical limit.
    """
    hip = np.asarray(hip, dtype=np.float64)
    knee = np.asarray(knee, dtype=np.float64)
    ankle = np.asarray(ankle, dtype=np.float64)

    thigh = knee - hip
    shin = ankle - knee
    len_thigh = np.linalg.norm(thigh)
    len_shin = np.linalg.norm(shin)

    if len_thigh < 1e-4 or len_shin < 1e-4:
        return knee

    leg_line = ankle - hip
    len_leg = np.linalg.norm(leg_line)
    if len_leg < 1e-4:
        return knee

    u_leg = leg_line / len_leg

    # Vector from hip-ankle line to knee
    knee_proj = hip + np.dot(knee - hip, u_leg) * u_leg
    lateral_disp = knee - knee_proj
    disp_norm = np.linalg.norm(lateral_disp)

    # In SMPL-X (+Z forward), knee bend must have positive forward displacement (Z >= -0.02)
    # If lateral_disp has negative forward component (knee bending backward into hyperextension),
    # clamp it to anatomical zero (straight leg or slight forward bend).
    if lateral_disp[2] < -0.02:
        # Hyperextension detected! Project knee forward
        corrected_disp = lateral_disp.copy()
        corrected_disp[2] = 0.01  # Snap slightly forward to anatomical hinge direction
        if np.linalg.norm(corrected_disp) > 1e-4:
            scale = min(disp_norm, 0.25)
            knee_corrected = knee_proj + (corrected_disp / np.linalg.norm(corrected_disp)) * scale
        else:
            knee_corrected = knee_proj + np.array([0.0, 0.0, 0.01])
        return knee_corrected

    return knee


def swing_twist_decompose(q: np.ndarray, twist_axis: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Decomposes quaternion q into q_swing * q_twist around given twist_axis."""
    q = normalize_quaternion(q)
    axis = twist_axis / (np.linalg.norm(twist_axis) + 1e-8)

    # Projection of vector part onto twist axis
    p = np.dot(q[:3], axis) * axis
    q_twist = np.array([p[0], p[1], p[2], q[3]], dtype=np.float64)
    q_twist = normalize_quaternion(q_twist)

    # q_swing = q * conj(q_twist)
    q_swing = normalize_quaternion(quaternion_multiply(q, quaternion_conjugate(q_twist)))
    return q_swing, q_twist


def apply_anatomical_joint_limits(local_rotations: np.ndarray) -> np.ndarray:
    """
    Enforces biomechanical anatomical limits (ROM) on local rotations.
    Prevents hyperextension, spinal dislocation, and unnatural inversions.
    Accepts: (22, 4) or (T, 22, 4) in [x, y, z, w].
    """
    is_seq = (local_rotations.ndim == 3)
    rot_seq = local_rotations if is_seq else local_rotations[np.newaxis, ...]
    out = np.copy(rot_seq)

    T, J, _ = out.shape

    for t in range(T):
        for j in range(min(J, SMPLX_JOINT_COUNT)):
            q = normalize_quaternion(out[t, j])

            if j in (4, 5):  # L_Knee, R_Knee (Hinge around X axis, flexion 0 to 155 deg)
                euler = quaternion_to_euler(q)
                # Flexion around X axis: [0, 155]
                euler[0] = np.clip(euler[0], 0.0, 155.0)
                euler[1] = np.clip(euler[1], -10.0, 10.0)  # twist limit
                euler[2] = np.clip(euler[2], -8.0, 8.0)    # lateral tilt
                out[t, j] = euler_to_quaternion(euler)

            elif j in (18, 19):  # L_Elbow, R_Elbow (Hinge flexion 0 to 150 deg)
                euler = quaternion_to_euler(q)
                euler[0] = np.clip(euler[0], 0.0, 150.0)
                euler[1] = np.clip(euler[1], -25.0, 25.0)
                euler[2] = np.clip(euler[2], -15.0, 15.0)
                out[t, j] = euler_to_quaternion(euler)

            elif j in (1, 2):  # L_Hip, R_Hip
                euler = quaternion_to_euler(q)
                euler[0] = np.clip(euler[0], -30.0, 130.0)
                euler[1] = np.clip(euler[1], -45.0, 45.0)
                euler[2] = np.clip(euler[2], -45.0, 55.0)
                out[t, j] = euler_to_quaternion(euler)

            elif j in (16, 17):  # L_Shoulder, R_Shoulder
                euler = quaternion_to_euler(q)
                euler[0] = np.clip(euler[0], -60.0, 180.0)
                euler[1] = np.clip(euler[1], -90.0, 90.0)
                euler[2] = np.clip(euler[2], -40.0, 180.0)
                out[t, j] = euler_to_quaternion(euler)

    return out if is_seq else out[0]


# ============================================================================
# Constant Bone Length Model
# ============================================================================

class BoneLengthModel:
    """
    Estimates subject-specific bone lengths from confident frames
    and holds them constant across the entire motion sequence,
    preventing limbs from stretching, shrinking, or collapsing during occlusions.
    """

    def __init__(self):
        self.bone_lengths: Dict[str, float] = {}
        self.is_calibrated: bool = False

    def calibrate_from_observations(
        self,
        landmark_sequences: List[Optional[np.ndarray]],
        confidence_sequences: List[float]
    ):
        """
        Calculates median bone lengths from frames with confidence >= 0.5.
        landmark_sequences: List of (33, 3) or (22, 3) arrays.
        """
        valid_lengths: Dict[str, List[float]] = {
            "thigh_l": [], "shin_l": [],
            "thigh_r": [], "shin_r": [],
            "upperarm_l": [], "forearm_l": [],
            "upperarm_r": [], "forearm_r": [],
            "torso": []
        }

        defaults = {
            "thigh_l": 0.42, "shin_l": 0.42,
            "thigh_r": 0.42, "shin_r": 0.42,
            "upperarm_l": 0.28, "forearm_l": 0.26,
            "upperarm_r": 0.28, "forearm_r": 0.26,
            "torso": 0.50
        }

        for lms, conf in zip(landmark_sequences, confidence_sequences):
            if conf < 0.5 or lms is None:
                continue

            if len(lms) >= 33:
                # MediaPipe 33 layout
                # Left leg: Hip (23), Knee (25), Ankle (27)
                thigh_l = float(np.linalg.norm(lms[25] - lms[23]))
                shin_l = float(np.linalg.norm(lms[27] - lms[25]))
                if 0.2 < thigh_l < 0.8: valid_lengths["thigh_l"].append(thigh_l)
                if 0.2 < shin_l < 0.8: valid_lengths["shin_l"].append(shin_l)

                # Right leg: Hip (24), Knee (26), Ankle (28)
                thigh_r = float(np.linalg.norm(lms[26] - lms[24]))
                shin_r = float(np.linalg.norm(lms[28] - lms[26]))
                if 0.2 < thigh_r < 0.8: valid_lengths["thigh_r"].append(thigh_r)
                if 0.2 < shin_r < 0.8: valid_lengths["shin_r"].append(shin_r)

                # Left arm: Shoulder (11), Elbow (13), Wrist (15)
                uarm_l = float(np.linalg.norm(lms[13] - lms[11]))
                farm_l = float(np.linalg.norm(lms[15] - lms[13]))
                if 0.15 < uarm_l < 0.6: valid_lengths["upperarm_l"].append(uarm_l)
                if 0.15 < farm_l < 0.6: valid_lengths["forearm_l"].append(farm_l)

                # Right arm: Shoulder (12), Elbow (14), Wrist (16)
                uarm_r = float(np.linalg.norm(lms[14] - lms[12]))
                farm_r = float(np.linalg.norm(lms[16] - lms[14]))
                if 0.15 < uarm_r < 0.6: valid_lengths["upperarm_r"].append(uarm_r)
                if 0.15 < farm_r < 0.6: valid_lengths["forearm_r"].append(farm_r)

            elif len(lms) >= 22:
                # SMPL-X 22 layout
                thigh_l = float(np.linalg.norm(lms[4] - lms[1]))
                shin_l = float(np.linalg.norm(lms[7] - lms[4]))
                if 0.2 < thigh_l < 0.8: valid_lengths["thigh_l"].append(thigh_l)
                if 0.2 < shin_l < 0.8: valid_lengths["shin_l"].append(shin_l)

                thigh_r = float(np.linalg.norm(lms[5] - lms[2]))
                shin_r = float(np.linalg.norm(lms[8] - lms[5]))
                if 0.2 < thigh_r < 0.8: valid_lengths["thigh_r"].append(thigh_r)
                if 0.2 < shin_r < 0.8: valid_lengths["shin_r"].append(shin_r)

                uarm_l = float(np.linalg.norm(lms[18] - lms[16]))
                farm_l = float(np.linalg.norm(lms[20] - lms[18]))
                if 0.15 < uarm_l < 0.6: valid_lengths["upperarm_l"].append(uarm_l)
                if 0.15 < farm_l < 0.6: valid_lengths["forearm_l"].append(farm_l)

                uarm_r = float(np.linalg.norm(lms[19] - lms[17]))
                farm_r = float(np.linalg.norm(lms[21] - lms[19]))
                if 0.15 < uarm_r < 0.6: valid_lengths["upperarm_r"].append(uarm_r)
                if 0.15 < farm_r < 0.6: valid_lengths["forearm_r"].append(farm_r)

        for bone, vals in valid_lengths.items():
            if len(vals) >= 5:
                self.bone_lengths[bone] = float(np.median(vals))
            else:
                self.bone_lengths[bone] = defaults[bone]

        # Symmetrize left and right
        leg_thigh = (self.bone_lengths["thigh_l"] + self.bone_lengths["thigh_r"]) * 0.5
        leg_shin = (self.bone_lengths["shin_l"] + self.bone_lengths["shin_r"]) * 0.5
        arm_upper = (self.bone_lengths["upperarm_l"] + self.bone_lengths["upperarm_r"]) * 0.5
        arm_fore = (self.bone_lengths["forearm_l"] + self.bone_lengths["forearm_r"]) * 0.5

        self.bone_lengths["thigh_l"] = self.bone_lengths["thigh_r"] = leg_thigh
        self.bone_lengths["shin_l"] = self.bone_lengths["shin_r"] = leg_shin
        self.bone_lengths["upperarm_l"] = self.bone_lengths["upperarm_r"] = arm_upper
        self.bone_lengths["forearm_l"] = self.bone_lengths["forearm_r"] = arm_fore
        self.is_calibrated = True

    def enforce_limb_lengths(self, landmarks_3d: np.ndarray) -> np.ndarray:
        """
        Projects joints onto constant bone lengths while preserving anatomical direction.
        Supports both MediaPipe 33 and SMPL-X 22 formats.
        """
        if not self.is_calibrated or landmarks_3d is None or len(landmarks_3d) < 22:
            return landmarks_3d

        lms = np.copy(landmarks_3d)

        if len(lms) >= 33:
            # MediaPipe 33 indices
            # Left leg (23 -> 25 -> 27)
            dir_tl = lms[25] - lms[23]
            d_tl = np.linalg.norm(dir_tl)
            if d_tl > 1e-4:
                lms[25] = lms[23] + (dir_tl / d_tl) * self.bone_lengths["thigh_l"]
            dir_sl = lms[27] - lms[25]
            d_sl = np.linalg.norm(dir_sl)
            if d_sl > 1e-4:
                lms[27] = lms[25] + (dir_sl / d_sl) * self.bone_lengths["shin_l"]

            # Right leg (24 -> 26 -> 28)
            dir_tr = lms[26] - lms[24]
            d_tr = np.linalg.norm(dir_tr)
            if d_tr > 1e-4:
                lms[26] = lms[24] + (dir_tr / d_tr) * self.bone_lengths["thigh_r"]
            dir_sr = lms[28] - lms[26]
            d_sr = np.linalg.norm(dir_sr)
            if d_sr > 1e-4:
                lms[28] = lms[26] + (dir_sr / d_sr) * self.bone_lengths["shin_r"]

            # Left arm (11 -> 13 -> 15)
            dir_ual = lms[13] - lms[11]
            d_ual = np.linalg.norm(dir_ual)
            if d_ual > 1e-4:
                lms[13] = lms[11] + (dir_ual / d_ual) * self.bone_lengths["upperarm_l"]
            dir_fal = lms[15] - lms[13]
            d_fal = np.linalg.norm(dir_fal)
            if d_fal > 1e-4:
                lms[15] = lms[13] + (dir_fal / d_fal) * self.bone_lengths["forearm_l"]

            # Right arm (12 -> 14 -> 16)
            dir_uar = lms[14] - lms[12]
            d_uar = np.linalg.norm(dir_uar)
            if d_uar > 1e-4:
                lms[14] = lms[12] + (dir_uar / d_uar) * self.bone_lengths["upperarm_r"]
            dir_far = lms[16] - lms[14]
            d_far = np.linalg.norm(dir_far)
            if d_far > 1e-4:
                lms[16] = lms[14] + (dir_far / d_far) * self.bone_lengths["forearm_r"]

        elif len(lms) >= 22:
            # SMPL-X 22 indices
            # Left leg (1 -> 4 -> 7)
            dir_tl = lms[4] - lms[1]
            d_tl = np.linalg.norm(dir_tl)
            if d_tl > 1e-4:
                lms[4] = lms[1] + (dir_tl / d_tl) * self.bone_lengths["thigh_l"]
            dir_sl = lms[7] - lms[4]
            d_sl = np.linalg.norm(dir_sl)
            if d_sl > 1e-4:
                lms[7] = lms[4] + (dir_sl / d_sl) * self.bone_lengths["shin_l"]

            # Right leg (2 -> 5 -> 8)
            dir_tr = lms[5] - lms[2]
            d_tr = np.linalg.norm(dir_tr)
            if d_tr > 1e-4:
                lms[5] = lms[2] + (dir_tr / d_tr) * self.bone_lengths["thigh_r"]
            dir_sr = lms[8] - lms[5]
            d_sr = np.linalg.norm(dir_sr)
            if d_sr > 1e-4:
                lms[8] = lms[5] + (dir_sr / d_sr) * self.bone_lengths["shin_r"]

            # Left arm (16 -> 18 -> 20)
            dir_ual = lms[18] - lms[16]
            d_ual = np.linalg.norm(dir_ual)
            if d_ual > 1e-4:
                lms[18] = lms[16] + (dir_ual / d_ual) * self.bone_lengths["upperarm_l"]
            dir_fal = lms[20] - lms[18]
            d_fal = np.linalg.norm(dir_fal)
            if d_fal > 1e-4:
                lms[20] = lms[18] + (dir_fal / d_fal) * self.bone_lengths["forearm_l"]

            # Right arm (17 -> 19 -> 21)
            dir_uar = lms[19] - lms[17]
            d_uar = np.linalg.norm(dir_uar)
            if d_uar > 1e-4:
                lms[19] = lms[17] + (dir_uar / d_uar) * self.bone_lengths["upperarm_r"]
            dir_far = lms[21] - lms[19]
            d_far = np.linalg.norm(dir_far)
            if d_far > 1e-4:
                lms[21] = lms[19] + (dir_far / d_far) * self.bone_lengths["forearm_r"]

        return lms
