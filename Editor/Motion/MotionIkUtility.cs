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

            worldPositions[0] = rootPosition.y > 0.5f ? rootPosition : rootPosition + DefaultJointOffsets[SmplxJoint.Pelvis];
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

            // Compute rotations using pure managed FromToRotation for maximum portability and speed
            Quaternion rotToMid = SafeFromToRotation(curL1, desiredMid - rootPos);
            Vector3 rotatedMid = rootPos + (rotToMid * curL1);
            Vector3 rotatedEnd = rotatedMid + (rotToMid * curL2);

            Quaternion rotToEnd = SafeFromToRotation(rotatedEnd - rotatedMid, targetPos - rotatedMid);

            rootDeltaRot = rotToMid;
            midDeltaRot = rotToEnd;

            return true;
        }

        /// <summary>
        /// Pure managed calculation of FromToRotation without relying on native engine icalls.
        /// </summary>
        public static Quaternion SafeFromToRotation(Vector3 from, Vector3 to)
        {
            Vector3 v0 = from.normalized;
            Vector3 v1 = to.normalized;
            float d = Vector3.Dot(v0, v1);
            if (d >= 0.999999f)
            {
                return Quaternion.identity;
            }
            if (d <= -0.999999f)
            {
                Vector3 axis = Vector3.Cross(Vector3.right, v0);
                if (axis.sqrMagnitude < 0.0001f)
                    axis = Vector3.Cross(Vector3.up, v0);
                axis.Normalize();
                return new Quaternion(axis.x, axis.y, axis.z, 0f);
            }

            Vector3 a = Vector3.Cross(v0, v1);
            float w = 1f + d;
            float mag = Mathf.Sqrt(a.x * a.x + a.y * a.y + a.z * a.z + w * w);
            if (mag > 0.00001f)
            {
                return new Quaternion(a.x / mag, a.y / mag, a.z / mag, w / mag);
            }
            return Quaternion.identity;
        }

        /// <summary>
        /// Pure managed quaternion inverse calculation.
        /// </summary>
        public static Quaternion SafeInverse(Quaternion q)
        {
            float lengthSq = q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w;
            if (lengthSq > 0.00001f)
            {
                float inv = 1f / lengthSq;
                return new Quaternion(-q.x * inv, -q.y * inv, -q.z * inv, q.w * inv);
            }
            return Quaternion.identity;
        }

        /// <summary>
        /// Pure managed quaternion spherical linear interpolation.
        /// </summary>
        public static Quaternion SafeSlerp(Quaternion a, Quaternion b, float t)
        {
            float dot = a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
            if (dot < 0.0f)
            {
                b = new Quaternion(-b.x, -b.y, -b.z, -b.w);
                dot = -dot;
            }

            if (dot > 0.9995f)
            {
                float rx = a.x + (b.x - a.x) * t;
                float ry = a.y + (b.y - a.y) * t;
                float rz = a.z + (b.z - a.z) * t;
                float rw = a.w + (b.w - a.w) * t;
                float mag = Mathf.Sqrt(rx * rx + ry * ry + rz * rz + rw * rw);
                return mag > 0.00001f ? new Quaternion(rx / mag, ry / mag, rz / mag, rw / mag) : Quaternion.identity;
            }

            float theta = Mathf.Acos(Mathf.Clamp(dot, -1f, 1f));
            float sinTheta = Mathf.Sin(theta);
            if (Mathf.Abs(sinTheta) < 0.0001f) return a;

            float wa = Mathf.Sin((1f - t) * theta) / sinTheta;
            float wb = Mathf.Sin(t * theta) / sinTheta;

            return new Quaternion(
                wa * a.x + wb * b.x,
                wa * a.y + wb * b.y,
                wa * a.z + wb * b.z,
                wa * a.w + wb * b.w
            );
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
            Quaternion newUpperLocal = SafeInverse(parentWorld) * newUpperWorld;

            // 2. Calculate new Mid world rotation and convert to local space (relative to new Upper)
            Quaternion newMidWorld = midDeltaRot * (rootDeltaRot * curMidWorld);
            Quaternion newMidLocal = SafeInverse(newUpperWorld) * newMidWorld;

            if (recordUndo)
            {
                data.RecordUndo($"IK on {endJoint} (Frame {frame})");
            }

            data.SetJointRotation(frame, upperJoint, newUpperLocal);
            data.SetJointRotation(frame, midJoint, newMidLocal);

            return true;
        }

        /// <summary>
        /// Pure managed calculation of angular difference between two Quaternions in degrees.
        /// Does not rely on Unity native InternalCalls, safe for tests and batch execution.
        /// </summary>
        public static float CalcAngle(Quaternion a, Quaternion b)
        {
            float dot = Mathf.Clamp(Mathf.Abs(a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w), -1f, 1f);
            return Mathf.Acos(dot) * 2f * Mathf.Rad2Deg;
        }

        /// <summary>
        /// Pure managed extraction of Angle and Axis from a Quaternion.
        /// </summary>
        public static void ToAngleAxisManaged(Quaternion q, out float angle, out Vector3 axis)
        {
            float qw = Mathf.Clamp(q.w, -1f, 1f);
            angle = 2f * Mathf.Acos(Mathf.Abs(qw)) * Mathf.Rad2Deg;
            float sinHalf = Mathf.Sqrt(Mathf.Max(0f, 1f - qw * qw));
            if (sinHalf < 0.0001f)
            {
                axis = Vector3.up;
            }
            else
            {
                float sign = qw < 0f ? -1f : 1f;
                axis = new Vector3(q.x * sign / sinHalf, q.y * sign / sinHalf, q.z * sign / sinHalf);
            }
        }

        /// <summary>
        /// Pure managed creation of a Quaternion from an axis and angle in degrees.
        /// </summary>
        public static Quaternion AngleAxisManaged(float angleDeg, Vector3 axis)
        {
            float halfRad = (angleDeg * 0.5f) * Mathf.Deg2Rad;
            float s = Mathf.Sin(halfRad);
            Vector3 normAxis = axis.sqrMagnitude > 0.00001f ? axis.normalized : Vector3.up;
            return new Quaternion(normAxis.x * s, normAxis.y * s, normAxis.z * s, Mathf.Cos(halfRad));
        }

        /// <summary>
        /// Pure managed spherical linear interpolation (Slerp) between two Quaternions.
        /// Does not rely on Unity native InternalCalls, safe for standalone CLI and tests.
        /// </summary>
        public static Quaternion SlerpManaged(Quaternion a, Quaternion b, float t)
        {
            t = Mathf.Clamp01(t);
            float dot = a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;

            if (dot < 0.0f)
            {
                b = new Quaternion(-b.x, -b.y, -b.z, -b.w);
                dot = -dot;
            }

            if (dot > 0.9995f)
            {
                Quaternion result = new Quaternion(
                    a.x + t * (b.x - a.x),
                    a.y + t * (b.y - a.y),
                    a.z + t * (b.z - a.z),
                    a.w + t * (b.w - a.w)
                );
                return NormalizeManaged(result);
            }

            float theta0 = Mathf.Acos(Mathf.Clamp(dot, -1f, 1f));
            float theta = theta0 * t;
            float sinTheta = Mathf.Sin(theta);
            float sinTheta0 = Mathf.Sin(theta0);

            float s0 = Mathf.Cos(theta) - dot * sinTheta / sinTheta0;
            float s1 = sinTheta / sinTheta0;

            return new Quaternion(
                s0 * a.x + s1 * b.x,
                s0 * a.y + s1 * b.y,
                s0 * a.z + s1 * b.z,
                s0 * a.w + s1 * b.w
            );
        }

        /// <summary>
        /// Pure managed normalization of a Quaternion.
        /// </summary>
        public static Quaternion NormalizeManaged(Quaternion q)
        {
            float mag = Mathf.Sqrt(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w);
            if (mag < 0.00001f) return Quaternion.identity;
            return new Quaternion(q.x / mag, q.y / mag, q.z / mag, q.w / mag);
        }

        /// <summary>
        /// Pure managed Euler-to-Quaternion conversion (Z-X-Y intrinsic / Unity Euler convention).
        /// Safe for standalone tests and CLI tools without Unity internal calls.
        /// </summary>
        public static Quaternion EulerManaged(float x, float y, float z)
        {
            float rx = x * 0.5f * Mathf.Deg2Rad;
            float ry = y * 0.5f * Mathf.Deg2Rad;
            float rz = z * 0.5f * Mathf.Deg2Rad;
            float sinX = Mathf.Sin(rx), cosX = Mathf.Cos(rx);
            float sinY = Mathf.Sin(ry), cosY = Mathf.Cos(ry);
            float sinZ = Mathf.Sin(rz), cosZ = Mathf.Cos(rz);
            return new Quaternion(
                sinX * cosY * cosZ - cosX * sinY * sinZ,
                cosX * sinY * cosZ + sinX * cosY * sinZ,
                cosX * cosY * sinZ - sinX * sinY * cosZ,
                cosX * cosY * cosZ + sinX * sinY * sinZ
            );
        }

        public static Quaternion EulerManaged(Vector3 euler) => EulerManaged(euler.x, euler.y, euler.z);
    }
}
