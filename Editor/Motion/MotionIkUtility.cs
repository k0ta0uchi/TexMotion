using System;
using System.Collections.Generic;
using TexMotion.Runtime.Motion;
using UnityEngine;

namespace TexMotion.Editor.Motion
{
    /// <summary>
    /// Utility for analytical Two-Bone IK, Forward Kinematics evaluation, and Foot Grounding.
    /// </summary>
    public static class MotionIkUtility
    {
        // Reference SMPL-X T-Pose bone lengths / relative offsets in standard humanoid scale (meters)
        private static readonly Dictionary<SmplxJoint, Vector3> DefaultJointOffsets = new Dictionary<SmplxJoint, Vector3>
        {
            { SmplxJoint.Pelvis,     new Vector3(0.00f, 0.95f, 0.00f) },
            { SmplxJoint.L_Hip,      new Vector3(-0.09f, -0.06f, 0.00f) },
            { SmplxJoint.R_Hip,      new Vector3( 0.09f, -0.06f, 0.00f) },
            { SmplxJoint.Spine1,     new Vector3(0.00f, 0.12f, 0.00f) },
            { SmplxJoint.L_Knee,     new Vector3(0.00f, -0.42f, 0.00f) },
            { SmplxJoint.R_Knee,     new Vector3(0.00f, -0.42f, 0.00f) },
            { SmplxJoint.Spine2,     new Vector3(0.00f, 0.14f, 0.00f) },
            { SmplxJoint.L_Ankle,    new Vector3(0.00f, -0.41f, 0.00f) },
            { SmplxJoint.R_Ankle,    new Vector3(0.00f, -0.41f, 0.00f) },
            { SmplxJoint.Spine3,     new Vector3(0.00f, 0.13f, 0.00f) },
            { SmplxJoint.L_Foot,     new Vector3(0.00f, -0.08f, 0.14f) },
            { SmplxJoint.R_Foot,     new Vector3(0.00f, -0.08f, 0.14f) },
            { SmplxJoint.Neck,       new Vector3(0.00f, 0.09f, 0.00f) },
            { SmplxJoint.L_Collar,   new Vector3(-0.06f, 0.05f, 0.00f) },
            { SmplxJoint.R_Collar,   new Vector3( 0.06f, 0.05f, 0.00f) },
            { SmplxJoint.Head,       new Vector3(0.00f, 0.11f, 0.00f) },
            { SmplxJoint.L_Shoulder, new Vector3(-0.13f, 0.00f, 0.00f) },
            { SmplxJoint.R_Shoulder, new Vector3( 0.13f, 0.00f, 0.00f) },
            { SmplxJoint.L_Elbow,    new Vector3(-0.27f, 0.00f, 0.00f) },
            { SmplxJoint.R_Elbow,    new Vector3( 0.27f, 0.00f, 0.00f) },
            { SmplxJoint.L_Wrist,    new Vector3(-0.26f, 0.00f, 0.00f) },
            { SmplxJoint.R_Wrist,    new Vector3( 0.26f, 0.00f, 0.00f) },
        };

        public static Vector3 GetDefaultOffset(SmplxJoint joint)
        {
            return DefaultJointOffsets.TryGetValue(joint, out var v) ? v : Vector3.zero;
        }

        /// <summary>
        /// Computes global 3D positions of all 22 SMPL-X joints for a given frame using Forward Kinematics.
        /// </summary>
        public static Vector3[] ComputeForwardKinematics(Vector3 rootPosition, Quaternion[] localRotations)
        {
            ComputeForwardKinematics(rootPosition, localRotations, out var worldPositions, out _);
            return worldPositions;
        }

        /// <summary>
        /// Computes global 3D positions and orientations of all 22 SMPL-X joints for a given frame using Forward Kinematics.
        /// </summary>
        public static void ComputeForwardKinematics(
            Vector3 rootPosition,
            Quaternion[] localRotations,
            out Vector3[] worldPositions,
            out Quaternion[] worldRotations)
        {
            int jointCount = SmplxJointDefinitions.JointCount;
            worldPositions = new Vector3[jointCount];
            worldRotations = new Quaternion[jointCount];

            worldPositions[0] = rootPosition + DefaultJointOffsets[SmplxJoint.Pelvis];
            worldRotations[0] = localRotations != null && localRotations.Length > 0 ? localRotations[0] : Quaternion.identity;

            for (int i = 1; i < jointCount; i++)
            {
                int parent = SmplxJointDefinitions.Parents[i];
                Quaternion parentRot = worldRotations[parent];
                Vector3 parentPos = worldPositions[parent];

                SmplxJoint joint = (SmplxJoint)i;
                Vector3 localOffset = DefaultJointOffsets.TryGetValue(joint, out var off) ? off : Vector3.zero;
                Vector3 worldPos = parentPos + (parentRot * localOffset);

                Quaternion localRot = localRotations != null && localRotations.Length > i ? localRotations[i] : Quaternion.identity;
                Quaternion worldRot = parentRot * localRot;

                worldPositions[i] = worldPos;
                worldRotations[i] = worldRot;
            }
        }

        /// <summary>
        /// Analytical Two-Bone Inverse Kinematics solver.
        /// Rotates rootJoint and midJoint so that endJoint reaches targetPosition.
        /// </summary>
        public static bool SolveTwoBoneIK(
            Vector3 rootPos,
            Vector3 midPos,
            Vector3 endPos,
            Vector3 targetPos,
            Vector3 poleVector,
            out Quaternion rootDeltaRot,
            out Quaternion midDeltaRot)
        {
            rootDeltaRot = Quaternion.identity;
            midDeltaRot = Quaternion.identity;

            float l1 = Vector3.Distance(rootPos, midPos);
            float l2 = Vector3.Distance(midPos, endPos);
            if (l1 < 0.001f || l2 < 0.001f) return false;

            Vector3 toTarget = targetPos - rootPos;
            float targetDist = toTarget.magnitude;
            if (targetDist < 0.0001f) return false;

            // Clamp target distance within reachable range
            float maxReach = (l1 + l2) * 0.9999f;
            float minReach = Mathf.Abs(l1 - l2) * 1.0001f;
            float clampedDist = Mathf.Clamp(targetDist, minReach, maxReach);

            // Law of Cosines for angles
            float cosMid = (l1 * l1 + l2 * l2 - clampedDist * clampedDist) / (2f * l1 * l2);
            cosMid = Mathf.Clamp(cosMid, -1f, 1f);
            float angleMidRad = Mathf.Acos(cosMid);

            float cosRoot = (l1 * l1 + clampedDist * clampedDist - l2 * l2) / (2f * l1 * clampedDist);
            cosRoot = Mathf.Clamp(cosRoot, -1f, 1f);
            float angleRootRad = Mathf.Acos(cosRoot);

            // Current limb geometry
            Vector3 curL1 = midPos - rootPos;
            Vector3 curL2 = endPos - midPos;

            // Determine bend normal using pole vector
            Vector3 targetDir = toTarget / targetDist;
            Vector3 poleDir = poleVector - rootPos;
            Vector3 normal = Vector3.Cross(targetDir, poleDir);
            if (normal.sqrMagnitude < 0.0001f)
            {
                normal = Vector3.Cross(targetDir, Vector3.up);
                if (normal.sqrMagnitude < 0.0001f)
                {
                    normal = Vector3.Cross(targetDir, Vector3.right);
                }
            }
            normal.Normalize();

            // Calculate bend direction perpendicular to target line towards pole
            Vector3 bendDir = Vector3.Cross(normal, targetDir).normalized;

            // Desired mid position
            Vector3 desiredMid = rootPos + (targetDir * (l1 * Mathf.Cos(angleRootRad))) + (bendDir * (l1 * Mathf.Sin(angleRootRad)));

            // Compute rotations
            Quaternion rotToMid = Quaternion.FromToRotation(curL1, desiredMid - rootPos);
            Vector3 rotatedMid = rootPos + (rotToMid * curL1);
            Vector3 rotatedEnd = rotatedMid + (rotToMid * curL2);

            Quaternion rotToEnd = Quaternion.FromToRotation(rotatedEnd - rotatedMid, targetPos - rotatedMid);

            rootDeltaRot = rotToMid;
            midDeltaRot = rotToEnd;

            return true;
        }

        /// <summary>
        /// Solves Two-Bone IK for limb joints and applies the delta rotations
        /// properly transformed into parent local coordinate spaces.
        /// </summary>
        public static bool ApplyLimbIK(
            EditableMotionData data,
            int frame,
            SmplxJoint upperJoint,
            SmplxJoint midJoint,
            SmplxJoint endJoint,
            Vector3 targetWorldPos,
            Vector3 poleWorldPos)
        {
            return ApplyLimbIK(data, frame, upperJoint, midJoint, endJoint, targetWorldPos, (Vector3?)poleWorldPos, true);
        }

        /// <summary>
        /// Solves Two-Bone IK for limb joints and applies the delta rotations
        /// properly transformed into parent local coordinate spaces.
        /// If poleWorldPos is null or zero, preserves the natural bend direction of midJoint.
        /// </summary>
        public static bool ApplyLimbIK(
            EditableMotionData data,
            int frame,
            SmplxJoint upperJoint,
            SmplxJoint midJoint,
            SmplxJoint endJoint,
            Vector3 targetWorldPos,
            Vector3? poleWorldPos = null,
            bool recordUndo = true)
        {
            if (data == null || frame < 0 || frame >= data.Frames) return false;

            // Compute current forward kinematics (positions and orientations)
            Vector3 rootPos = data.GetRootPosition(frame);
            int jointCount = SmplxJointDefinitions.JointCount;
            Quaternion[] locals = new Quaternion[jointCount];
            for (int j = 0; j < jointCount; j++) locals[j] = data.GetJointRotation(frame, (SmplxJoint)j);

            ComputeForwardKinematics(rootPos, locals, out var worldPositions, out var worldRotations);

            int uIdx = (int)upperJoint;
            int mIdx = (int)midJoint;
            int eIdx = (int)endJoint;

            Vector3 pUpper = worldPositions[uIdx];
            Vector3 pMid = worldPositions[mIdx];
            Vector3 pEnd = worldPositions[eIdx];

            // Default pole to current mid joint position to preserve natural bend plane
            Vector3 effectivePole = poleWorldPos.HasValue && poleWorldPos.Value != Vector3.zero
                ? poleWorldPos.Value
                : pMid;

            if (!SolveTwoBoneIK(pUpper, pMid, pEnd, targetWorldPos, effectivePole, out Quaternion rootDeltaRot, out Quaternion midDeltaRot))
            {
                return false;
            }

            int upperParent = SmplxJointDefinitions.Parents[uIdx];
            Quaternion parentWorld = upperParent >= 0 ? worldRotations[upperParent] : Quaternion.identity;
            Quaternion curUpperWorld = worldRotations[uIdx];
            Quaternion curMidWorld = worldRotations[mIdx];

            // 1. Calculate new Upper world rotation and convert to local space
            Quaternion newUpperWorld = rootDeltaRot * curUpperWorld;
            Quaternion newUpperLocal = Quaternion.Inverse(parentWorld) * newUpperWorld;

            // 2. Calculate new Mid world rotation and convert to local space (relative to new Upper)
            Quaternion newMidWorld = midDeltaRot * (rootDeltaRot * curMidWorld);
            Quaternion newMidLocal = Quaternion.Inverse(newUpperWorld) * newMidWorld;

            if (recordUndo)
            {
                data.RecordUndo($"IK on {endJoint} (Frame {frame})");
            }

            data.SetJointRotation(frame, upperJoint, newUpperLocal);
            data.SetJointRotation(frame, midJoint, newMidLocal);

            return true;
        }
    }
}
