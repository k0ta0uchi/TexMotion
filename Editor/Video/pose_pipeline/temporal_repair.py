"""
temporal_repair.py - Short-duration gap repair using Quaternion SLERP and linear interpolation.

Repairs tracking dropouts, occlusions, and predicted gaps of duration <= 0.2s
using confidence-weighted Quaternion Spherical Linear Interpolation (SLERP) on SO(3).
Outputs a structured repairProvenance log for transparent downstream reporting.
"""

from typing import List, Dict, Any, Optional, Tuple, Union
import math
import numpy as np
from .kinematics import (
    quaternion_slerp,
    normalize_quaternion,
    quaternion_multiply,
    quaternion_conjugate,
)
from .observations import ObservationStatus


def repair_short_gaps_slerp(
    rotations: np.ndarray,
    confidences: np.ndarray,
    statuses: Optional[Union[List[List[Any]], np.ndarray]] = None,
    fps: float = 30.0,
    max_gap_seconds: float = 0.2,
    confidence_threshold: float = 0.35
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
    """
    Interpolates short gaps (<= 0.2 seconds) in rotation trajectories using quaternion SLERP.

    Parameters:
        rotations: (Frames, Joints, 4) or (Frames, 4) numpy array of unit quaternions [x, y, z, w].
        confidences: (Frames,) or (Frames, Joints) array of detection confidences [0, 1].
        statuses: Optional matrix of ObservationStatus enums/ints indicating observation provenance.
        fps: Playback frame rate used to calculate max frame gap threshold.
        max_gap_seconds: Maximum dropout duration to interpolate (default 0.2s).
        confidence_threshold: Threshold below which a frame is considered invalid/missing.

    Returns:
        repaired_rotations: Array of identical shape with repaired rotations.
        repaired_confidences: Array of identical shape with updated confidences.
        repair_provenance: List of records describing each repaired interval.
    """
    if len(rotations) < 3:
        return np.copy(rotations), np.copy(confidences), []

    out_rot = np.copy(rotations)
    out_conf = np.copy(confidences)
    provenance: List[Dict[str, Any]] = []

    fps = max(1.0, float(fps))
    max_gap_frames = max(1, int(round(fps * max_gap_seconds)))

    # Handle shape (Frames, 4) vs (Frames, Joints, 4)
    is_multi_joint = (out_rot.ndim == 3 and out_rot.shape[-1] == 4)
    n_frames = out_rot.shape[0]
    n_joints = out_rot.shape[1] if is_multi_joint else 1

    # Normalize confidences to (Frames, Joints)
    if out_conf.ndim == 1:
        conf_matrix = np.tile(out_conf[:, np.newaxis], (1, n_joints))
    else:
        conf_matrix = np.copy(out_conf)

    for j in range(n_joints):
        # Identify missing or low confidence frames
        is_gap = np.zeros(n_frames, dtype=bool)
        for t in range(n_frames):
            c = float(conf_matrix[t, j])
            stat = None
            if statuses is not None:
                try:
                    stat = statuses[t][j] if is_multi_joint else statuses[t]
                except (IndexError, TypeError):
                    stat = None

            status_missing = False
            if stat is not None:
                if isinstance(stat, ObservationStatus):
                    status_missing = (stat in (
                        ObservationStatus.MISSING,
                        ObservationStatus.OCCLUDED,
                        ObservationStatus.PREDICTED
                    ))
                elif isinstance(stat, int):
                    status_missing = (stat in (
                        int(ObservationStatus.MISSING),
                        int(ObservationStatus.OCCLUDED),
                        int(ObservationStatus.PREDICTED)
                    ))

            if c < confidence_threshold or status_missing:
                is_gap[t] = True

        # Find contiguous gap segments
        t = 0
        while t < n_frames:
            if is_gap[t]:
                start_t = t
                while t < n_frames and is_gap[t]:
                    t += 1
                end_t = t - 1
                gap_len = end_t - start_t + 1

                # Check if gap is within max_gap_frames and bounded by valid frames
                prev_t = start_t - 1
                next_t = end_t + 1
                if 1 <= gap_len <= max_gap_frames and prev_t >= 0 and next_t < n_frames:
                    c_prev = float(conf_matrix[prev_t, j])
                    c_next = float(conf_matrix[next_t, j])

                    # Ensure boundaries are confident enough
                    if c_prev >= confidence_threshold and c_next >= confidence_threshold:
                        q_prev = out_rot[prev_t, j] if is_multi_joint else out_rot[prev_t]
                        q_next = out_rot[next_t, j] if is_multi_joint else out_rot[next_t]

                        for cur_t in range(start_t, end_t + 1):
                            u = (cur_t - prev_t) / float(next_t - prev_t)
                            q_interp = quaternion_slerp(q_prev, q_next, u)
                            if is_multi_joint:
                                out_rot[cur_t, j] = q_interp
                            else:
                                out_rot[cur_t] = q_interp

                            # Interpolate confidence with a slight discount for interpolation
                            c_interp = ((1.0 - u) * c_prev + u * c_next) * 0.90
                            conf_matrix[cur_t, j] = max(conf_matrix[cur_t, j], c_interp)

                        # Determine prior status for provenance
                        prior_status_name = "OCCLUDED"
                        if statuses is not None:
                            try:
                                sample_stat = statuses[start_t][j] if is_multi_joint else statuses[start_t]
                                if isinstance(sample_stat, ObservationStatus):
                                    prior_status_name = sample_stat.name
                            except Exception:
                                pass

                        record = {
                            "jointIndex": int(j),
                            "startFrame": int(start_t),
                            "endFrame": int(end_t),
                            "gapLength": int(gap_len),
                            "durationSeconds": round(float(gap_len / fps), 4),
                            "method": "quaternion_slerp",
                            "priorStatus": prior_status_name,
                            "confidence": round(float(np.mean(conf_matrix[start_t:end_t + 1, j])), 4),
                            "prevConfidence": round(float(c_prev), 4),
                            "nextConfidence": round(float(c_next), 4)
                        }
                        provenance.append(record)
            else:
                t += 1

    if out_conf.ndim == 1:
        out_conf = np.mean(conf_matrix, axis=1)
    else:
        out_conf = conf_matrix

    return out_rot, out_conf, provenance


def repair_short_gaps_positions(
    positions: np.ndarray,
    confidences: np.ndarray,
    fps: float = 30.0,
    max_gap_seconds: float = 0.2,
    confidence_threshold: float = 0.35
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """
    Interpolates short gaps in 3D joint positions using linear interpolation.
    """
    if len(positions) < 3:
        return np.copy(positions), []

    out_pos = np.copy(positions)
    provenance: List[Dict[str, Any]] = []

    fps = max(1.0, float(fps))
    max_gap_frames = max(1, int(round(fps * max_gap_seconds)))
    n_frames = out_pos.shape[0]
    n_joints = out_pos.shape[1] if out_pos.ndim == 3 else 1

    if confidences.ndim == 1:
        conf_matrix = np.tile(confidences[:, np.newaxis], (1, n_joints))
    else:
        conf_matrix = confidences

    for j in range(n_joints):
        is_gap = [conf_matrix[t, j] < confidence_threshold for t in range(n_frames)]
        t = 0
        while t < n_frames:
            if is_gap[t]:
                start_t = t
                while t < n_frames and is_gap[t]:
                    t += 1
                end_t = t - 1
                gap_len = end_t - start_t + 1
                prev_t = start_t - 1
                next_t = end_t + 1

                if 1 <= gap_len <= max_gap_frames and prev_t >= 0 and next_t < n_frames:
                    if conf_matrix[prev_t, j] >= confidence_threshold and conf_matrix[next_t, j] >= confidence_threshold:
                        p_prev = out_pos[prev_t, j] if out_pos.ndim == 3 else out_pos[prev_t]
                        p_next = out_pos[next_t, j] if out_pos.ndim == 3 else out_pos[next_t]

                        for cur_t in range(start_t, end_t + 1):
                            u = (cur_t - prev_t) / float(next_t - prev_t)
                            p_interp = (1.0 - u) * p_prev + u * p_next
                            if out_pos.ndim == 3:
                                out_pos[cur_t, j] = p_interp
                            else:
                                out_pos[cur_t] = p_interp

                        provenance.append({
                            "jointIndex": int(j),
                            "startFrame": int(start_t),
                            "endFrame": int(end_t),
                            "gapLength": int(gap_len),
                            "durationSeconds": round(float(gap_len / fps), 4),
                            "method": "linear_lerp"
                        })
            else:
                t += 1

    return out_pos, provenance


# ============================================================================
# Kinematic Hermite & Squad Long-Gap Inpainting (Approach A)
# ============================================================================

def quaternion_log(q: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Computes the quaternion logarithm log(q) mapping SO(3) -> so(3) Lie algebra."""
    q_norm = normalize_quaternion(q)
    v = q_norm[:3]
    w = float(np.clip(q_norm[3], -1.0, 1.0))
    v_norm = float(np.linalg.norm(v))
    if v_norm < eps:
        return np.zeros(3, dtype=np.float64)
    theta = math.acos(w)
    return (theta / v_norm) * v


def quaternion_exp(omega: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Computes the quaternion exponential exp(omega) mapping so(3) Lie algebra -> SO(3)."""
    theta = float(np.linalg.norm(omega))
    if theta < eps:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    axis = omega / theta
    sin_half = math.sin(theta)
    cos_half = math.cos(theta)
    return np.array([
        axis[0] * sin_half,
        axis[1] * sin_half,
        axis[2] * sin_half,
        cos_half
    ], dtype=np.float64)


def quaternion_squad(
    q0: np.ndarray,
    q1: np.ndarray,
    s0: np.ndarray,
    s1: np.ndarray,
    u: float
) -> np.ndarray:
    """
    Spherical and Quadrangle Interpolation (Squad) for C1-continuous orientation curves.
    squad(q0, q1, s0, s1, u) = slerp(slerp(q0, q1, u), slerp(s0, s1, u), 2*u*(1-u))
    """
    u = float(np.clip(u, 0.0, 1.0))
    q_slerp = quaternion_slerp(q0, q1, u)
    s_slerp = quaternion_slerp(s0, s1, u)
    h = 2.0 * u * (1.0 - u)
    return quaternion_slerp(q_slerp, s_slerp, h)


def compute_squad_control_point(
    q_prev: np.ndarray,
    q_curr: np.ndarray,
    q_next: np.ndarray
) -> np.ndarray:
    """Computes intermediate Squad tangent control point s_i for C1 continuity."""
    q_inv = quaternion_conjugate(q_curr)
    q_next_rel = quaternion_multiply(q_inv, q_next)
    q_prev_rel = quaternion_multiply(q_inv, q_prev)

    log1 = quaternion_log(q_next_rel)
    log2 = quaternion_log(q_prev_rel)
    exponent = -0.25 * (log1 + log2)
    delta_q = quaternion_exp(exponent)
    return normalize_quaternion(quaternion_multiply(q_curr, delta_q))


def cubic_hermite_interpolate(
    p0: np.ndarray,
    p1: np.ndarray,
    v0: np.ndarray,
    v1: np.ndarray,
    duration: float,
    u: float,
    damping: float = 0.0
) -> np.ndarray:
    """
    Cubic Hermite Spline position interpolation with optional velocity damping.
    Ensures C1 continuity with boundary velocities v0 and v1.
    """
    u = float(np.clip(u, 0.0, 1.0))
    u2 = u * u
    u3 = u2 * u

    # Hermite basis functions
    h00 = 2.0 * u3 - 3.0 * u2 + 1.0
    h10 = u3 - 2.0 * u2 + u
    h01 = -2.0 * u3 + 3.0 * u2
    h11 = u3 - u2

    # Apply velocity damping for long-duration dropouts to prevent overshoot
    v0_eff = v0 * math.exp(-damping * u)
    v1_eff = v1 * math.exp(-damping * (1.0 - u))

    T = max(1e-4, float(duration))
    return h00 * p0 + h10 * (v0_eff * T) + h01 * p1 + h11 * (v1_eff * T)


def repair_long_gaps_kinematic_hermite(
    rotations: np.ndarray,
    confidences: np.ndarray,
    root_positions: Optional[np.ndarray] = None,
    contact_track: Optional[Dict[str, Any]] = None,
    statuses: Optional[Union[List[List[Any]], np.ndarray]] = None,
    fps: float = 30.0,
    min_gap_seconds: float = 0.2,
    max_gap_seconds: float = 2.5,
    confidence_threshold: float = 0.35,
    damping: float = 0.35
) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray, List[Dict[str, Any]]]:
    """
    Inpaints medium-to-long tracking dropouts (0.2s to 2.5s) using Kinematic Hermite
    Splines and SO(3) Squad interpolation with velocity damping and contact preservation.

    Parameters:
        rotations: (Frames, Joints, 4) or (Frames, 4) array of unit quaternions.
        confidences: (Frames,) or (Frames, Joints) array of detection confidences.
        root_positions: Optional (Frames, 3) root translation trajectory.
        contact_track: Optional dict containing foot contact intervals to preserve floor anchoring.
        statuses: Optional matrix of ObservationStatus flags.
        fps: Sampling rate (frames per second).
        min_gap_seconds: Minimum gap length to consider for long-duration inpainting (default 0.2s).
        max_gap_seconds: Maximum gap duration allowed for kinematic inpainting (default 2.5s).
        confidence_threshold: Below this, a frame is considered missing/occluded.
        damping: Velocity decay rate across long gaps to prevent ballistic divergence.

    Returns:
        (repaired_rotations, repaired_root_positions, repaired_confidences, provenance_records)
    """
    if len(rotations) < 4:
        return np.copy(rotations), np.copy(root_positions) if root_positions is not None else None, np.copy(confidences), []

    out_rot = np.copy(rotations)
    out_pos = np.copy(root_positions) if root_positions is not None else None
    out_conf = np.copy(confidences)
    provenance: List[Dict[str, Any]] = []

    fps = max(1.0, float(fps))
    dt = 1.0 / fps
    min_gap_frames = max(1, int(round(fps * min_gap_seconds)))
    max_gap_frames = max(min_gap_frames, int(round(fps * max_gap_seconds)))

    is_multi_joint = (out_rot.ndim == 3 and out_rot.shape[-1] == 4)
    n_frames = out_rot.shape[0]
    n_joints = out_rot.shape[1] if is_multi_joint else 1

    if out_conf.ndim == 1:
        conf_matrix = np.tile(out_conf[:, np.newaxis], (1, n_joints))
    else:
        conf_matrix = np.copy(out_conf)

    # Preserve original confidence for independent position gap detection
    orig_root_conf = np.copy(conf_matrix[:, 0])

    # 1. Inpaint rotations joint-by-joint
    for j in range(n_joints):
        is_gap = np.zeros(n_frames, dtype=bool)
        for t in range(n_frames):
            c = float(conf_matrix[t, j])
            stat = None
            if statuses is not None:
                try:
                    stat = statuses[t][j] if is_multi_joint else statuses[t]
                except (IndexError, TypeError):
                    stat = None

            status_missing = False
            if stat is not None:
                if isinstance(stat, ObservationStatus):
                    status_missing = (stat in (
                        ObservationStatus.MISSING,
                        ObservationStatus.OCCLUDED,
                        ObservationStatus.PREDICTED
                    ))
                elif isinstance(stat, int):
                    status_missing = (stat in (
                        int(ObservationStatus.MISSING),
                        int(ObservationStatus.OCCLUDED),
                        int(ObservationStatus.PREDICTED)
                    ))

            if c < confidence_threshold or status_missing:
                is_gap[t] = True

        t = 0
        while t < n_frames:
            if is_gap[t]:
                start_t = t
                while t < n_frames and is_gap[t]:
                    t += 1
                end_t = t - 1
                gap_len = end_t - start_t + 1

                prev_t = start_t - 1
                next_t = end_t + 1

                if min_gap_frames <= gap_len <= max_gap_frames and prev_t >= 0 and next_t < n_frames:
                    c_prev = float(conf_matrix[prev_t, j])
                    c_next = float(conf_matrix[next_t, j])

                    if c_prev >= confidence_threshold and c_next >= confidence_threshold:
                        q_prev = out_rot[prev_t, j] if is_multi_joint else out_rot[prev_t]
                        q_next = out_rot[next_t, j] if is_multi_joint else out_rot[next_t]

                        # Pre-boundary anchor for Squad tangent (fallback to boundary if clamped)
                        prev_anchor_t = max(0, prev_t - 2)
                        q_prev_anchor = out_rot[prev_anchor_t, j] if is_multi_joint else out_rot[prev_anchor_t]
                        s0 = compute_squad_control_point(q_prev_anchor, q_prev, q_next)

                        # Post-boundary anchor for Squad tangent
                        next_anchor_t = min(n_frames - 1, next_t + 2)
                        q_next_anchor = out_rot[next_anchor_t, j] if is_multi_joint else out_rot[next_anchor_t]
                        s1 = compute_squad_control_point(q_prev, q_next, q_next_anchor)

                        gap_duration = gap_len * dt

                        for cur_t in range(start_t, end_t + 1):
                            u = (cur_t - prev_t) / float(next_t - prev_t)
                            q_interp = quaternion_squad(q_prev, q_next, s0, s1, u)

                            if is_multi_joint:
                                out_rot[cur_t, j] = q_interp
                            else:
                                out_rot[cur_t] = q_interp

                            # Confidence decays gracefully towards middle of long gap
                            decay_weight = 1.0 - 0.3 * math.sin(u * math.pi)
                            c_interp = ((1.0 - u) * c_prev + u * c_next) * decay_weight
                            conf_matrix[cur_t, j] = max(conf_matrix[cur_t, j], c_interp)

                        provenance.append({
                            "jointIndex": int(j),
                            "startFrame": int(start_t),
                            "endFrame": int(end_t),
                            "gapLength": int(gap_len),
                            "durationSeconds": round(float(gap_duration), 4),
                            "method": "kinematic_hermite_squad",
                            "confidence": round(float(np.mean(conf_matrix[start_t:end_t + 1, j])), 4),
                        })
            else:
                t += 1

    # 2. Inpaint root translation trajectory using Cubic Hermite Spline
    if out_pos is not None:
        # Detect gaps in root position using original root confidence (joint 0)
        is_pos_gap = orig_root_conf < confidence_threshold

        t = 0
        while t < n_frames:
            if is_pos_gap[t]:
                start_t = t
                while t < n_frames and is_pos_gap[t]:
                    t += 1
                end_t = t - 1
                gap_len = end_t - start_t + 1
                prev_t = start_t - 1
                next_t = end_t + 1

                if min_gap_frames <= gap_len <= max_gap_frames and prev_t >= 0 and next_t < n_frames:
                    p0 = out_pos[prev_t]
                    p1 = out_pos[next_t]

                    # Estimate boundary velocities from surrounding frames
                    k_back = max(1, min(3, prev_t))
                    v0 = (p0 - out_pos[prev_t - k_back]) / (k_back * dt)

                    k_fwd = max(1, min(3, n_frames - 1 - next_t))
                    v1 = (out_pos[next_t + k_fwd] - p1) / (k_fwd * dt)

                    gap_dur = (next_t - prev_t) * dt

                    for cur_t in range(start_t, end_t + 1):
                        u = (cur_t - prev_t) / float(next_t - prev_t)
                        out_pos[cur_t] = cubic_hermite_interpolate(p0, p1, v0, v1, gap_dur, u, damping=damping)

                    provenance.append({
                        "jointIndex": 0,
                        "type": "root_position",
                        "startFrame": int(start_t),
                        "endFrame": int(end_t),
                        "gapLength": int(gap_len),
                        "durationSeconds": round(float(gap_len * dt), 4),
                        "method": "cubic_hermite_position",
                    })
            else:
                t += 1

    if out_conf.ndim == 1:
        out_conf = np.mean(conf_matrix, axis=1)
    else:
        out_conf = conf_matrix

    return out_rot, out_pos, out_conf, provenance

