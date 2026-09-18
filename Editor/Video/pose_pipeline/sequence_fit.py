"""
sequence_fit.py - Temporal sequence fitting, multi-hypothesis crossing resolution, and contact optimization.

Implements:
- Multi-hypothesis leg crossing resolution evaluating entry/exit trajectories.
- Competing hypotheses ranking and uncertainty interval export.
- Temporal rotation smoothing on SO(3) with quaternion sign continuity.
- Foot contact hysteresis using world root motion.
- Floor snapping for grounded feet.
"""

from typing import List, Tuple, Optional, Dict, Any
import math
import numpy as np
from .observations import UncertaintyInterval, FrameObservations
from .hypotheses import (
    HypothesisResult,
    HypothesisScore,
    HypothesisUncertaintyInterval,
    PoseHypothesis,
    score_hypotheses,
)
from .kinematics import (
    quaternion_slerp,
    quaternion_multiply,
    normalize_quaternion,
    quaternion_to_matrix,
    matrix_to_quaternion,
    constrain_knee_flexion
)


class SequenceOptimizer:
    """
    Optimizes multi-frame motion sequences:
    - Resolves self-occlusions and crossings using entry/exit temporal context.
    - Eliminates high-frequency jitter on SO(3).
    - Detects ambiguous intervals and generates user-facing uncertainty reports.
    - Accurately identifies foot contacts using world root displacement.
    """

    def __init__(self, fps: float = 30.0):
        self.fps = max(1.0, float(fps))
        self.dt = 1.0 / self.fps
        # Rich diagnostics are retained on the optimizer so the historical
        # three-item optimize_sequence return value remains compatible.
        self.last_hypothesis_results: List[HypothesisResult] = []
        self.last_temporal_penalties: Dict[str, Any] = {}
        self.last_diagnostics: Dict[str, Any] = {}
        self.hypothesis_provenance: List[Dict[str, Any]] = []
        self.ambiguity_margin = 0.05

    def resolve_leg_crossings_multi_hypothesis(
        self,
        frames_3d: List[Optional[np.ndarray]],
        frames_2d: List[Optional[np.ndarray]],
        confidences: List[float]
    ) -> Tuple[List[np.ndarray], List[UncertaintyInterval]]:
        """
        Detects leg crossing intervals and resolves them using competing hypotheses
        evaluated across both entry and exit trajectories.
        Supports both SMPL-X 22 and MediaPipe 33 coordinate arrays.
        """
        n_frames = len(frames_3d)
        if n_frames < 3:
            return [f.copy() if f is not None else np.zeros((22, 3)) for f in frames_3d], []

        resolved_3d = [f.copy() if f is not None else None for f in frames_3d]
        uncertainty_intervals: List[UncertaintyInterval] = []

        # Find first valid frame to determine joint layout
        first_valid = next((f for f in frames_3d if f is not None), None)
        if first_valid is None:
            return frames_3d, []

        is_mp33 = (len(first_valid) >= 33)
        # Joint indices
        l_hip_idx = 23 if is_mp33 else 1
        r_hip_idx = 24 if is_mp33 else 2
        l_knee_idx = 25 if is_mp33 else 4
        r_knee_idx = 26 if is_mp33 else 5
        l_ank_idx = 27 if is_mp33 else 7
        r_ank_idx = 28 if is_mp33 else 8

        # 1. Identify crossing hazard intervals: where 2D ankle distance is small or X order flips
        in_hazard = False
        hazard_start = 0
        crossing_windows: List[Tuple[int, int]] = []

        for t in range(n_frames):
            f2d = frames_2d[t] if t < len(frames_2d) else None
            if f2d is None or len(f2d) <= max(l_ank_idx, r_ank_idx, 28):
                # Fallback to 3D X distance
                if resolved_3d[t] is not None:
                    dist_x = abs(resolved_3d[t][l_ank_idx][0] - resolved_3d[t][r_ank_idx][0])
                    is_overlap = dist_x < 0.08
                else:
                    is_overlap = False
            else:
                # 2D screen distance (MediaPipe 2D is always 33 points).
                l_ank_2d = f2d[27] if len(f2d) >= 33 else f2d[min(7, len(f2d)-1)]
                r_ank_2d = f2d[28] if len(f2d) >= 33 else f2d[min(8, len(f2d)-1)]
                dist_2d = float(np.linalg.norm(l_ank_2d - r_ank_2d))
                is_overlap = dist_2d < 0.065

            if is_overlap:
                if not in_hazard:
                    in_hazard = True
                    hazard_start = t
            else:
                if in_hazard:
                    in_hazard = False
                    if t - hazard_start >= 1:
                        crossing_windows.append((hazard_start, t - 1))

        if in_hazard:
            crossing_windows.append((hazard_start, n_frames - 1))

        # 2. Evaluate each crossing interval with Competing Hypotheses
        for (start_t, end_t) in crossing_windows:
            ctx_start = max(0, start_t - 4)
            ctx_end = min(n_frames - 1, end_t + 4)

            # In SMPL-X (+Z is forward): larger Z is forward
            # In MediaPipe (-Z is forward): smaller Z is forward
            z_mult = -1.0 if is_mp33 else 1.0

            # Entry evidence
            entry_f = resolved_3d[ctx_start]
            z_left_enter = entry_f[l_ank_idx][2] * z_mult if entry_f is not None else 0.0
            z_right_enter = entry_f[r_ank_idx][2] * z_mult if entry_f is not None else 0.0

            # Exit evidence
            exit_f = resolved_3d[ctx_end]
            z_left_exit = exit_f[l_ank_idx][2] * z_mult if exit_f is not None else 0.0
            z_right_exit = exit_f[r_ank_idx][2] * z_mult if exit_f is not None else 0.0

            # Score hypotheses (Left-in-front vs Right-in-front)
            score_left_front = 0.0
            score_right_front = 0.0

            if z_left_enter > z_right_enter:
                score_left_front += 1.5
            else:
                score_right_front += 1.5

            if z_left_exit > z_right_exit:
                score_left_front += 1.5
            else:
                score_right_front += 1.5

            # Keep the ranking and its evidence explicit.  The old branch
            # below still performs the same depth edit, while this result is
            # retained for diagnostics and uncertainty export.
            leg_result = score_hypotheses(
                [
                    {
                        "name": "left_leg_front",
                        "score": score_left_front,
                        "components": {
                            "entry_depth": 1.5 if z_left_enter > z_right_enter else 0.0,
                            "exit_depth": 1.5 if z_left_exit > z_right_exit else 0.0,
                        },
                        "provenance": {
                            "type": "leg_crossing",
                            "source": "wham_temporal_prior",
                            "frontSign": z_mult,
                        },
                    },
                    {
                        "name": "right_leg_front",
                        "score": score_right_front,
                        "components": {
                            "entry_depth": 1.5 if z_left_enter <= z_right_enter else 0.0,
                            "exit_depth": 1.5 if z_left_exit <= z_right_exit else 0.0,
                        },
                        "provenance": {
                            "type": "leg_crossing",
                            "source": "wham_temporal_prior",
                            "frontSign": z_mult,
                        },
                    },
                ],
                ambiguity_margin=self.ambiguity_margin,
                provenance={
                    "type": "leg_crossing",
                    "contextFrames": [ctx_start, ctx_end],
                    "coordinateLayout": "mediapipe33" if is_mp33 else "smplx22",
                },
            )

            # Apply winning hypothesis across crossing interval
            for t in range(start_t, end_t + 1):
                cur_3d = resolved_3d[t]
                if cur_3d is None:
                    continue

                l_z = cur_3d[l_ank_idx][2]
                r_z = cur_3d[r_ank_idx][2]

                if score_left_front >= score_right_front:
                    # Left in front
                    if is_mp33:
                        # Smaller Z in MP is front
                        if l_z >= r_z - 0.08:
                            cur_3d[l_ank_idx][2] = r_z - 0.10
                            cur_3d[l_knee_idx][2] = cur_3d[r_knee_idx][2] - 0.08
                    else:
                        # Larger Z in SMPL-X is front
                        if l_z <= r_z + 0.08:
                            cur_3d[l_ank_idx][2] = r_z + 0.10
                            cur_3d[l_knee_idx][2] = cur_3d[r_knee_idx][2] + 0.08
                else:
                    # Right in front
                    if is_mp33:
                        if r_z >= l_z - 0.08:
                            cur_3d[r_ank_idx][2] = l_z - 0.10
                            cur_3d[r_knee_idx][2] = cur_3d[l_knee_idx][2] - 0.08
                    else:
                        if r_z <= l_z + 0.08:
                            cur_3d[r_ank_idx][2] = l_z + 0.10
                            cur_3d[r_knee_idx][2] = cur_3d[l_knee_idx][2] + 0.08

            leg_interval = leg_result.to_uncertainty_interval(
                start_t,
                end_t,
                "leg_crossing_ambiguity",
                "swap_crossing",
                {
                    "type": "leg_crossing",
                    "selectedBy": "entry_exit_depth_score",
                    "frontSign": z_mult,
                },
            )
            uncertainty_intervals.append(leg_interval)
            self.last_hypothesis_results.append(leg_result)
            self.hypothesis_provenance.append(leg_interval.to_dict())

        return resolved_3d, uncertainty_intervals

    def compute_temporal_penalties(
        self,
        joints_seq: np.ndarray,
        timestamps: Optional[List[float]] = None,
        bone_model: Optional[Any] = None,
        root_positions: Optional[np.ndarray] = None,
        contact_flags: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        floor_y: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Compute explainable temporal biomechanical penalty energies.

        The returned dictionary intentionally exposes both scalar terms and
        per-frame arrays.  Consumers that only need a quality score can use
        ``bone_length``, ``joint_limits``, ``velocity``, ``acceleration``,
        ``contact`` and ``sliding``; diagnostic UIs can use the corresponding
        ``*_per_frame`` arrays.  Derivatives use timestamp deltas when given,
        rather than assuming a nominal frame rate.
        """
        if isinstance(joints_seq, np.ndarray) and joints_seq.ndim == 3:
            sequence = [np.asarray(joints_seq[t], dtype=np.float64).copy() for t in range(len(joints_seq))]
        else:
            sequence = [None if frame is None else np.asarray(frame, dtype=np.float64).copy() for frame in joints_seq]
        frames = len(sequence)
        if frames == 0:
            empty = np.zeros(0, dtype=np.float64)
            return {
                "bone_length": 0.0,
                "joint_limits": 0.0,
                "joint_limit": 0.0,
                "velocity": 0.0,
                "acceleration": 0.0,
                "contact": 0.0,
                "sliding": 0.0,
                "foot_sliding": 0.0,
                "total": 0.0,
                "dt": self.dt,
                "dt_per_frame": empty,
                "bone_length_per_frame": empty,
                "joint_limits_per_frame": empty,
                "velocity_per_frame": empty,
                "acceleration_per_frame": empty,
                "contact_per_frame": empty,
                "sliding_per_frame": empty,
                "contact_flags": (np.zeros(0, dtype=bool), np.zeros(0, dtype=bool)),
            }

        first = next((frame for frame in sequence if frame is not None), None)
        if first is None or first.ndim != 2 or first.shape[1] < 3:
            empty = np.zeros(frames, dtype=np.float64)
            return {
                "bone_length": 0.0,
                "joint_limits": 0.0,
                "joint_limit": 0.0,
                "velocity": 0.0,
                "acceleration": 0.0,
                "contact": 0.0,
                "sliding": 0.0,
                "foot_sliding": 0.0,
                "total": 0.0,
                "dt": self.dt,
                "dt_per_frame": np.zeros(max(0, frames - 1), dtype=np.float64),
                "bone_length_per_frame": empty,
                "joint_limits_per_frame": empty,
                "velocity_per_frame": empty,
                "acceleration_per_frame": empty,
                "contact_per_frame": empty,
                "sliding_per_frame": empty,
                "contact_flags": (np.zeros(frames, dtype=bool), np.zeros(frames, dtype=bool)),
            }

        joint_count = len(first)
        if joint_count >= 33:
            segments = [
                ("thigh_l", 23, 25), ("shin_l", 25, 27),
                ("thigh_r", 24, 26), ("shin_r", 26, 28),
                ("upperarm_l", 11, 13), ("forearm_l", 13, 15),
                ("upperarm_r", 12, 14), ("forearm_r", 14, 16),
            ]
            knees = [(23, 25, 27), (24, 26, 28)]
            elbows = [(11, 13, 15), (12, 14, 16)]
            left_foot_index, right_foot_index = 31, 32
        else:
            segments = [
                ("thigh_l", 1, 4), ("shin_l", 4, 7),
                ("thigh_r", 2, 5), ("shin_r", 5, 8),
                ("upperarm_l", 16, 18), ("forearm_l", 18, 20),
                ("upperarm_r", 17, 19), ("forearm_r", 19, 21),
            ] if joint_count >= 22 else []
            knees = [(1, 4, 7), (2, 5, 8)] if joint_count >= 22 else []
            elbows = [(16, 18, 20), (17, 19, 21)] if joint_count >= 22 else []
            left_foot_index, right_foot_index = 10, 11

        # Robust, subject-specific bone targets are used as a soft energy. A
        # supplied calibrated model takes precedence over sequence medians.
        observed_lengths: Dict[str, List[float]] = {name: [] for name, _, _ in segments}
        for frame in sequence:
            if frame is None or len(frame) <= max((child for _, _, child in segments), default=-1):
                continue
            for name, parent, child in segments:
                length = float(np.linalg.norm(frame[child, :3] - frame[parent, :3]))
                if np.isfinite(length) and length > 1e-5:
                    observed_lengths[name].append(length)
        model_lengths = getattr(bone_model, "bone_lengths", {}) if bone_model is not None else {}
        targets: Dict[str, float] = {}
        for name, values in observed_lengths.items():
            model_value = model_lengths.get(name) if isinstance(model_lengths, dict) else None
            if model_value is not None and float(model_value) > 1e-5:
                targets[name] = float(model_value)
            elif values:
                targets[name] = float(np.median(values))
        bone_per_frame = np.zeros(frames, dtype=np.float64)
        bone_counts = np.zeros(frames, dtype=np.float64)
        for index, frame in enumerate(sequence):
            if frame is None:
                continue
            for name, parent, child in segments:
                target = targets.get(name)
                if target is None or max(parent, child) >= len(frame):
                    continue
                length = float(np.linalg.norm(frame[child, :3] - frame[parent, :3]))
                if np.isfinite(length):
                    bone_per_frame[index] += ((length - target) / max(target, 1e-5)) ** 2
                    bone_counts[index] += 1.0
        valid_bone = bone_counts > 0
        bone_per_frame[valid_bone] /= bone_counts[valid_bone]
        bone_length_penalty = float(np.mean(bone_per_frame[valid_bone])) if np.any(valid_bone) else 0.0

        # Position-derived ROM penalty. It is deliberately soft; the actual
        # legacy knee guard below remains available for the final pose, while
        # this term lets candidate ranking account for joint-limit violations.
        limits_per_frame = np.zeros(frames, dtype=np.float64)
        limits_counts = np.zeros(frames, dtype=np.float64)
        for index, frame in enumerate(sequence):
            if frame is None:
                continue
            for parent, joint, child in knees + elbows:
                if max(parent, joint, child) >= len(frame):
                    continue
                incoming = frame[parent, :3] - frame[joint, :3]
                outgoing = frame[child, :3] - frame[joint, :3]
                incoming_norm = float(np.linalg.norm(incoming))
                outgoing_norm = float(np.linalg.norm(outgoing))
                if incoming_norm < 1e-5 or outgoing_norm < 1e-5:
                    continue
                angle = math.degrees(math.acos(float(np.clip(np.dot(incoming, outgoing) / (incoming_norm * outgoing_norm), -1.0, 1.0))))
                flexion = max(0.0, 180.0 - angle)
                maximum = 155.0 if (parent, joint, child) in knees else 150.0
                limits_per_frame[index] += (max(0.0, flexion - maximum) / 30.0) ** 2
                limits_counts[index] += 1.0
            # Preserve the established directional knee check as a penalty,
            # not an uprightness rule.
            for parent, joint, child in knees:
                if max(parent, joint, child) >= len(frame):
                    continue
                corrected = constrain_knee_flexion(frame[parent], frame[joint], frame[child], is_left=(parent == knees[0][0]))
                scale = max(float(np.linalg.norm(frame[joint] - frame[parent])), 0.1)
                limits_per_frame[index] += 0.25 * (float(np.linalg.norm(corrected - frame[joint])) / scale) ** 2
                limits_counts[index] += 1.0
        valid_limits = limits_counts > 0
        limits_per_frame[valid_limits] /= limits_counts[valid_limits]
        joint_limit_penalty = float(np.mean(limits_per_frame[valid_limits])) if np.any(valid_limits) else 0.0

        if timestamps is not None and len(timestamps) >= frames:
            try:
                dt_values = np.maximum(np.diff(np.asarray(timestamps[:frames], dtype=np.float64)), 1e-6)
                dt_values[~np.isfinite(dt_values)] = self.dt
            except (TypeError, ValueError):
                dt_values = np.full(max(0, frames - 1), self.dt, dtype=np.float64)
        else:
            dt_values = np.full(max(0, frames - 1), self.dt, dtype=np.float64)

        velocity_per_frame = np.zeros(frames, dtype=np.float64)
        velocity_values: List[float] = []
        for index in range(1, frames):
            previous, current = sequence[index - 1], sequence[index]
            if previous is None or current is None or previous.shape != current.shape:
                continue
            velocity = np.linalg.norm(current[:, :3] - previous[:, :3], axis=1) / dt_values[index - 1]
            value = float(np.mean((velocity / 4.0) ** 2))
            velocity_per_frame[index] = value
            velocity_values.append(value)
        velocity_penalty = float(np.mean(velocity_values)) if velocity_values else 0.0

        acceleration_per_frame = np.zeros(frames, dtype=np.float64)
        acceleration_values: List[float] = []
        for index in range(2, frames):
            first_frame, middle_frame, last_frame = sequence[index - 2:index + 1]
            if first_frame is None or middle_frame is None or last_frame is None:
                continue
            if not (first_frame.shape == middle_frame.shape == last_frame.shape):
                continue
            velocity_before = (middle_frame[:, :3] - first_frame[:, :3]) / dt_values[index - 2]
            velocity_after = (last_frame[:, :3] - middle_frame[:, :3]) / dt_values[index - 1]
            acceleration = (velocity_after - velocity_before) / max((dt_values[index - 2] + dt_values[index - 1]) * 0.5, 1e-6)
            value = float(np.mean((np.linalg.norm(acceleration, axis=1) / 25.0) ** 2))
            acceleration_per_frame[index] = value
            acceleration_values.append(value)
        acceleration_penalty = float(np.mean(acceleration_values)) if acceleration_values else 0.0

        # Contact/sliding terms use world root displacement where available.
        left_feet = np.full((frames, 3), np.nan, dtype=np.float64)
        right_feet = np.full((frames, 3), np.nan, dtype=np.float64)
        for index, frame in enumerate(sequence):
            if frame is not None and max(left_foot_index, right_foot_index) < len(frame):
                left_feet[index] = frame[left_foot_index, :3]
                right_feet[index] = frame[right_foot_index, :3]
        roots = None
        if root_positions is not None:
            try:
                candidate_roots = np.asarray(root_positions, dtype=np.float64)
                if candidate_roots.ndim == 2 and len(candidate_roots) == frames and candidate_roots.shape[1] >= 3:
                    roots = candidate_roots[:, :3]
            except (TypeError, ValueError):
                roots = None
        world_left = left_feet.copy()
        world_right = right_feet.copy()
        if roots is not None:
            valid_left = np.isfinite(world_left).all(axis=1)
            valid_right = np.isfinite(world_right).all(axis=1)
            world_left[valid_left] += roots[valid_left]
            world_right[valid_right] += roots[valid_right]
        if contact_flags is None:
            contacts = self.apply_foot_contact_hysteresis(
                roots,
                np.nan_to_num(left_feet, nan=0.0),
                np.nan_to_num(right_feet, nan=0.0),
            )
        else:
            contacts = (
                np.asarray(contact_flags[0], dtype=bool)[:frames],
                np.asarray(contact_flags[1], dtype=bool)[:frames],
            )
        left_flags = np.pad(contacts[0], (0, max(0, frames - len(contacts[0]))))[:frames]
        right_flags = np.pad(contacts[1], (0, max(0, frames - len(contacts[1]))))[:frames]
        contacts = (left_flags, right_flags)
        valid_floor = np.concatenate([
            world_left[np.isfinite(world_left).all(axis=1), 1],
            world_right[np.isfinite(world_right).all(axis=1), 1],
        ])
        actual_floor = float(floor_y) if floor_y is not None else (float(np.percentile(valid_floor, 5)) if len(valid_floor) else 0.0)
        contact_values: List[float] = []
        contact_per_frame = np.zeros(frames, dtype=np.float64)
        sliding_values: List[float] = []
        sliding_per_frame = np.zeros(frames, dtype=np.float64)
        for index in range(frames):
            for position, active in ((world_left[index], left_flags[index]), (world_right[index], right_flags[index])):
                if active and np.isfinite(position).all():
                    penetration = max(actual_floor - float(position[1]), 0.0)
                    value = (penetration / 0.05) ** 2
                    contact_per_frame[index] += value
                    contact_values.append(value)
            if index == 0:
                continue
            for before, current, was_active, is_active in (
                (world_left[index - 1], world_left[index], left_flags[index - 1], left_flags[index]),
                (world_right[index - 1], world_right[index], right_flags[index - 1], right_flags[index]),
            ):
                if was_active and is_active and np.isfinite(before).all() and np.isfinite(current).all():
                    speed = float(np.linalg.norm((current - before)[[0, 2]])) / dt_values[index - 1]
                    value = (speed / 0.15) ** 2
                    sliding_per_frame[index] += value
                    sliding_values.append(value)
        contact_penalty = float(np.mean(contact_values)) if contact_values else 0.0
        sliding_penalty = float(np.mean(sliding_values)) if sliding_values else 0.0
        total = bone_length_penalty + joint_limit_penalty + velocity_penalty + acceleration_penalty + contact_penalty + sliding_penalty
        return {
            "bone_length": bone_length_penalty,
            "joint_limits": joint_limit_penalty,
            "joint_limit": joint_limit_penalty,
            "velocity": velocity_penalty,
            "acceleration": acceleration_penalty,
            "contact": contact_penalty,
            "sliding": sliding_penalty,
            "foot_sliding": sliding_penalty,
            "total": float(total),
            "dt": float(np.median(dt_values)) if len(dt_values) else self.dt,
            "dt_per_frame": dt_values,
            "bone_length_per_frame": bone_per_frame,
            "joint_limits_per_frame": limits_per_frame,
            "velocity_per_frame": velocity_per_frame,
            "acceleration_per_frame": acceleration_per_frame,
            "contact_per_frame": contact_per_frame,
            "sliding_per_frame": sliding_per_frame,
            "bone_targets": targets,
            "contact_flags": contacts,
        }

    def temporal_penalties(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        """Alias for :meth:`compute_temporal_penalties`."""
        return self.compute_temporal_penalties(*args, **kwargs)

    def bone_length_penalty(self, joints_seq: np.ndarray, bone_model: Optional[Any] = None) -> float:
        return float(self.compute_temporal_penalties(joints_seq, bone_model=bone_model)["bone_length"])

    def joint_limit_penalty(self, joints_seq: np.ndarray) -> float:
        return float(self.compute_temporal_penalties(joints_seq)["joint_limits"])

    def velocity_penalty(self, joints_seq: np.ndarray, timestamps: Optional[List[float]] = None) -> float:
        return float(self.compute_temporal_penalties(joints_seq, timestamps=timestamps)["velocity"])

    def acceleration_penalty(self, joints_seq: np.ndarray, timestamps: Optional[List[float]] = None) -> float:
        return float(self.compute_temporal_penalties(joints_seq, timestamps=timestamps)["acceleration"])

    def contact_penalty(self, joints_seq: np.ndarray, root_positions: Optional[np.ndarray] = None, contact_flags: Optional[Tuple[np.ndarray, np.ndarray]] = None) -> float:
        return float(self.compute_temporal_penalties(joints_seq, root_positions=root_positions, contact_flags=contact_flags)["contact"])

    def sliding_penalty(self, joints_seq: np.ndarray, root_positions: Optional[np.ndarray] = None, contact_flags: Optional[Tuple[np.ndarray, np.ndarray]] = None) -> float:
        return float(self.compute_temporal_penalties(joints_seq, root_positions=root_positions, contact_flags=contact_flags)["sliding"])

    def smooth_quaternions_so3(self, local_rotations: np.ndarray) -> np.ndarray:
        """
        Enforces quaternion sign continuity and applies geodesic SO(3) smoothing
        with angular acceleration regularization.
        local_rotations: (Frames, 22, 4) in [x, y, z, w].
        """
        frames, joints, _ = local_rotations.shape
        smoothed = np.copy(local_rotations)

        if frames < 2:
            return smoothed

        # Step 1: Enforce antipodal sign continuity (dot(q[t], q[t-1]) >= 0)
        for j in range(joints):
            for t in range(1, frames):
                q_prev = smoothed[t - 1, j]
                q_curr = smoothed[t, j]
                if np.dot(q_prev, q_curr) < 0.0:
                    smoothed[t, j] = -q_curr

        # Step 2: Temporal geodesic smoothing with angular speed damping
        if frames >= 3:
            temp = np.copy(smoothed)
            for j in range(joints):
                for t in range(1, frames - 1):
                    q_prev = temp[t - 1, j]
                    q_mid = temp[t, j]
                    q_next = temp[t + 1, j]

                    # Spherical midpoint between neighbors
                    q_ambient = quaternion_slerp(q_prev, q_next, 0.5)

                    # Geodesic distance (angle) between current and ambient
                    dot = abs(float(np.dot(q_mid, q_ambient)))
                    angle = 2.0 * math.acos(min(1.0, dot))

                    # If large velocity peak, preserve it (lower weight to ambient)
                    # If high-frequency flutter, blend more toward ambient
                    blend = 0.70 if angle < 0.25 else 0.85
                    smoothed[t, j] = quaternion_slerp(q_ambient, q_mid, blend)

        return smoothed

    def apply_foot_contact_hysteresis(
        self,
        root_positions: Optional[np.ndarray],
        left_foot_positions: np.ndarray,
        right_foot_positions: np.ndarray,
        vel_threshold: float = 0.15
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes foot contact states with velocity hysteresis using world root motion.
        vel_threshold: speed in meters/sec below which foot is considered in contact.
        Returns: (left_contact_flags, right_contact_flags) of shape (Frames,)
        """
        frames = len(left_foot_positions)
        l_contact = np.zeros(frames, dtype=bool)
        r_contact = np.zeros(frames, dtype=bool)

        if frames < 2:
            return l_contact, r_contact

        use_root = (root_positions is not None and len(root_positions) == frames)

        # Compute world-space velocities
        for t in range(1, frames):
            if use_root:
                p_l_curr = left_foot_positions[t] + root_positions[t]
                p_l_prev = left_foot_positions[t - 1] + root_positions[t - 1]
                p_r_curr = right_foot_positions[t] + root_positions[t]
                p_r_prev = right_foot_positions[t - 1] + root_positions[t - 1]
            else:
                p_l_curr = left_foot_positions[t]
                p_l_prev = left_foot_positions[t - 1]
                p_r_curr = right_foot_positions[t]
                p_r_prev = right_foot_positions[t - 1]

            l_speed = float(np.linalg.norm(p_l_curr - p_l_prev)) / self.dt
            r_speed = float(np.linalg.norm(p_r_curr - p_r_prev)) / self.dt

            # Hysteresis update
            # Enter contact: speed < vel_threshold
            # Exit contact: speed > vel_threshold * 1.5
            if l_contact[t - 1]:
                l_contact[t] = (l_speed < (vel_threshold * 1.5))
            else:
                l_contact[t] = (l_speed < vel_threshold)

            if r_contact[t - 1]:
                r_contact[t] = (r_speed < (vel_threshold * 1.5))
            else:
                r_contact[t] = (r_speed < vel_threshold)

        return l_contact, r_contact

    def apply_floor_snapping(
        self,
        foot_positions: np.ndarray,
        contact_flags: np.ndarray,
        floor_y: float = 0.0
    ) -> np.ndarray:
        """
        Snaps grounded feet to floor plane during confirmed contact phases
        to eliminate foot sliding and ground penetration.
        """
        out_pos = np.copy(foot_positions)
        for t in range(len(out_pos)):
            if contact_flags[t]:
                out_pos[t, 1] = max(floor_y, out_pos[t, 1])
        return out_pos

    def resolve_arm_occlusion_multi_hypothesis(
        self,
        frames_3d: List[Optional[np.ndarray]],
        frames_2d: List[Optional[np.ndarray]],
        confidences: List[float]
    ) -> Tuple[List[np.ndarray], List[UncertaintyInterval]]:
        """
        Detects arm self-occlusion intervals (wrists near head or torso) and resolves them
        using competing hypotheses (Front-of-Face vs Behind-Head) evaluated across temporal context.
        Supports both SMPL-X 22 and MediaPipe 33 coordinate arrays.
        """
        n_frames = len(frames_3d)
        if n_frames < 3:
            return [f.copy() if f is not None else np.zeros((22, 3)) for f in frames_3d], []

        resolved_3d = [f.copy() if f is not None else None for f in frames_3d]
        uncertainty_intervals: List[UncertaintyInterval] = []

        first_valid = next((f for f in frames_3d if f is not None), None)
        if first_valid is None:
            return frames_3d, []

        is_mp33 = (len(first_valid) >= 33)
        l_wrist_idx = 15 if is_mp33 else 20
        r_wrist_idx = 16 if is_mp33 else 21
        l_elbow_idx = 13 if is_mp33 else 18
        r_elbow_idx = 14 if is_mp33 else 19
        head_idx = 0 if is_mp33 else 15

        z_mult = -1.0 if is_mp33 else 1.0

        for arm_side in ["left", "right"]:
            wrist_idx = l_wrist_idx if arm_side == "left" else r_wrist_idx
            elbow_idx = l_elbow_idx if arm_side == "left" else r_elbow_idx

            in_hazard = False
            hazard_start = 0
            hazard_windows: List[Tuple[int, int]] = []

            for t in range(n_frames):
                cur = resolved_3d[t]
                if cur is None:
                    continue

                dist_to_head_xy = float(np.linalg.norm(cur[wrist_idx][:2] - cur[head_idx][:2]))
                if dist_to_head_xy < 0.25:
                    if not in_hazard:
                        in_hazard = True
                        hazard_start = t
                else:
                    if in_hazard:
                        in_hazard = False
                        if t - hazard_start >= 1:
                            hazard_windows.append((hazard_start, t - 1))

            if in_hazard:
                hazard_windows.append((hazard_start, n_frames - 1))

            for (start_t, end_t) in hazard_windows:
                ctx_start = max(0, start_t - 4)
                ctx_end = min(n_frames - 1, end_t + 4)

                entry_f = resolved_3d[ctx_start]
                z_enter = (entry_f[wrist_idx][2] - entry_f[head_idx][2]) * z_mult if entry_f is not None else 0.0

                exit_f = resolved_3d[ctx_end]
                z_exit = (exit_f[wrist_idx][2] - exit_f[head_idx][2]) * z_mult if exit_f is not None else 0.0

                score_front = 0.0
                score_behind = 0.0

                if z_enter > 0.01:
                    score_front += 1.5
                elif z_enter < -0.01:
                    score_behind += 1.5

                if z_exit > 0.01:
                    score_front += 1.5
                elif z_exit < -0.01:
                    score_behind += 1.5

                arm_result = score_hypotheses(
                    [
                        {
                            "name": f"{arm_side}_arm_front",
                            "score": score_front,
                            "components": {
                                "entry_depth": 1.5 if z_enter > 0.01 else 0.0,
                                "exit_depth": 1.5 if z_exit > 0.01 else 0.0,
                            },
                            "provenance": {
                                "type": "arm_depth",
                                "side": arm_side,
                                "source": "wham_temporal_prior",
                                "frontSign": z_mult,
                            },
                        },
                        {
                            "name": f"{arm_side}_arm_behind",
                            "score": score_behind,
                            "components": {
                                "entry_depth": 1.5 if z_enter < -0.01 else 0.0,
                                "exit_depth": 1.5 if z_exit < -0.01 else 0.0,
                            },
                            "provenance": {
                                "type": "arm_depth",
                                "side": arm_side,
                                "source": "wham_temporal_prior",
                                "frontSign": z_mult,
                            },
                        },
                    ],
                    ambiguity_margin=self.ambiguity_margin,
                    provenance={
                        "type": "arm_depth",
                        "side": arm_side,
                        "contextFrames": [ctx_start, ctx_end],
                        "coordinateLayout": "mediapipe33" if is_mp33 else "smplx22",
                    },
                )

                for t in range(start_t, end_t + 1):
                    cur = resolved_3d[t]
                    if cur is None:
                        continue

                    h_z = cur[head_idx][2]
                    w_z = cur[wrist_idx][2]

                    if score_front >= score_behind:
                        target_z = h_z - 0.12 if is_mp33 else h_z + 0.12
                        if is_mp33:
                            if w_z > target_z:
                                cur[wrist_idx][2] = target_z
                                cur[elbow_idx][2] = min(cur[elbow_idx][2], target_z + 0.05)
                        else:
                            if w_z < target_z:
                                cur[wrist_idx][2] = target_z
                                cur[elbow_idx][2] = max(cur[elbow_idx][2], target_z - 0.05)
                    else:
                        target_z = h_z + 0.15 if is_mp33 else h_z - 0.15
                        if is_mp33:
                            if w_z < target_z:
                                cur[wrist_idx][2] = target_z
                                cur[elbow_idx][2] = max(cur[elbow_idx][2], target_z - 0.05)
                        else:
                            if w_z > target_z:
                                cur[wrist_idx][2] = target_z
                                cur[elbow_idx][2] = min(cur[elbow_idx][2], target_z + 0.05)

                arm_interval = arm_result.to_uncertainty_interval(
                    start_t,
                    end_t,
                    f"arm_{arm_side}_head_occlusion",
                    "swap_depth_front_behind",
                    {
                        "type": "arm_depth",
                        "side": arm_side,
                        "selectedBy": "entry_exit_depth_score",
                        "frontSign": z_mult,
                    },
                )
                uncertainty_intervals.append(arm_interval)
                self.last_hypothesis_results.append(arm_result)
                self.hypothesis_provenance.append(arm_interval.to_dict())

        return resolved_3d, uncertainty_intervals

    def optimize_sequence(
        self,
        joints_seq: np.ndarray,
        norm_landmarks_seq: List[Optional[np.ndarray]],
        confidences: List[float],
        bone_model: Optional[Any] = None,
        root_positions: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, List[UncertaintyInterval], Tuple[np.ndarray, np.ndarray]]:
        """
        Unified Sequence Fitting (Phase 4):
        Performs holistic optimization of the motion sequence:
        1. Bone length invariant preservation (BoneLengthModel)
        2. Multi-hypothesis leg crossing resolution
        3. Multi-hypothesis arm/head self-occlusion resolution
        4. Anatomical joint limits enforcement (knee flexion overextension prevention)
        5. Foot contact hysteresis detection & ground stabilization
        Returns:
            optimized_joints: (Frames, 22, 3) array of 3D joint positions
            uncertainty_intervals: List of all detected ambiguous intervals
            (left_contact, right_contact): Bool arrays indicating foot contact states
        """
        frames = len(joints_seq)
        if frames == 0:
            return np.zeros((0, 22, 3)), [], (np.zeros(0, dtype=bool), np.zeros(0, dtype=bool))

        out_joints = np.copy(joints_seq)
        all_uncertainties: List[UncertaintyInterval] = []

        # 1. Enforce subject-specific bone lengths if bone_model is provided
        if bone_model is not None and getattr(bone_model, "is_calibrated", False):
            for t in range(frames):
                if out_joints[t] is not None:
                    out_joints[t] = bone_model.enforce_limb_lengths(out_joints[t])

        # 2. Multi-hypothesis leg crossing resolution
        joints_list = [out_joints[t] for t in range(frames)]
        resolved_legs, leg_uncertainties = self.resolve_leg_crossings_multi_hypothesis(
            joints_list, norm_landmarks_seq, confidences
        )
        for t in range(frames):
            if resolved_legs[t] is not None:
                out_joints[t] = resolved_legs[t]
        all_uncertainties.extend(leg_uncertainties)

        # 3. Multi-hypothesis arm self-occlusion resolution
        joints_list = [out_joints[t] for t in range(frames)]
        resolved_arms, arm_uncertainties = self.resolve_arm_occlusion_multi_hypothesis(
            joints_list, norm_landmarks_seq, confidences
        )
        for t in range(frames):
            if resolved_arms[t] is not None:
                out_joints[t] = resolved_arms[t]
        all_uncertainties.extend(arm_uncertainties)

        # 4. Enforce anatomical joint limits (knees do not bend forward)
        for t in range(frames):
            if out_joints[t] is not None and len(out_joints[t]) >= 22:
                # Left leg: Hip=1, Knee=4, Ankle=7
                # Right leg: Hip=2, Knee=5, Ankle=8
                out_joints[t, 4] = constrain_knee_flexion(out_joints[t, 1], out_joints[t, 4], out_joints[t, 7], is_left=True)
                out_joints[t, 5] = constrain_knee_flexion(out_joints[t, 2], out_joints[t, 5], out_joints[t, 8], is_left=False)

        # 5. Foot contact hysteresis detection
        l_foot_pos = out_joints[:, 10] if out_joints.shape[1] > 10 else out_joints[:, 7]
        r_foot_pos = out_joints[:, 11] if out_joints.shape[1] > 11 else out_joints[:, 8]
        l_contact, r_contact = self.apply_foot_contact_hysteresis(
            root_positions, l_foot_pos, r_foot_pos, vel_threshold=0.15
        )

        all_uncertainties.sort(key=lambda u: u.start_frame)
        return out_joints, all_uncertainties, (l_contact, r_contact)
