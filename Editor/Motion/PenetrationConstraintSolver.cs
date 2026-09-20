using System;
using System.Collections.Generic;
using TexMotion.Runtime.Motion;
using UnityEngine;

namespace TexMotion.Editor.Motion
{
    /// <summary>
    /// Configuration parameters for self-penetration avoidance between limbs and torso/thighs.
    /// </summary>
    [Serializable]
    public class PenetrationOptions
    {
        /// <summary>Clearance margin offset added to torso capsule radius (meters).</summary>
        public float TorsoRadiusOffset = 0.0f;

        /// <summary>Clearance margin offset added to thigh capsule radius (meters).</summary>
        public float ThighRadiusOffset = 0.0f;

        /// <summary>Safety padding added beyond the combined capsule radii (meters).</summary>
        public float Margin = 0.015f;

        /// <summary>Relaxation factor for intentional contact poses (0.0 = full push, 1.0 = full preservation).</summary>
        public float RelaxationThreshold = 0.5f;

        /// <summary>Whether to detect crossed-arm postures and relax push-out forces.</summary>
        public bool ProtectCrossedArms = true;

        /// <summary>Whether to detect hands placed on chest/torso and relax push-out forces.</summary>
        public bool ProtectHandOnChest = true;

        /// <summary>Interpolation blend strength for penetration correction [0.0, 1.0].</summary>
        public float PushBlendWeight = 1.0f;
    }

    /// <summary>
    /// Geometric capsule definition for limb and torso collision volumes.
    /// </summary>
    public struct SegmentCapsule
    {
        public Vector3 PointA;
        public Vector3 PointB;
        public float Radius;

        public SegmentCapsule(Vector3 a, Vector3 b, float radius)
        {
            PointA = a;
            PointB = b;
            Radius = Mathf.Max(0.01f, radius);
        }

        public Vector3 GetClosestPoint(Vector3 point)
        {
            Vector3 segment = PointB - PointA;
            float lengthSq = segment.sqrMagnitude;
            if (lengthSq < 0.00001f) return PointA;

            float t = Vector3.Dot(point - PointA, segment) / lengthSq;
            t = Mathf.Clamp01(t);
            return PointA + segment * t;
        }

        public float DistanceToPoint(Vector3 point, out Vector3 closestOnAxis)
        {
            closestOnAxis = GetClosestPoint(point);
            return Vector3.Distance(point, closestOnAxis);
        }
    }

    /// <summary>
    /// Realtime analytical penetration constraint solver for humanoids.
    /// Defines capsule geometry for arms, torso (pelvis to chest), and thighs,
    /// detecting mesh/joint penetrations and pushing arms outward using analytical Two-Bone IK.
    /// Includes heuristic relaxation for intentional contact gestures (crossed arms, hands on chest).
    /// </summary>
    public static class PenetrationConstraintSolver
    {
        // Default anatomical radii in meters (SMPL-X / standard humanoid scale)
        public const float DefaultLowerTorsoRadius = 0.14f;  // Pelvis -> Spine1
        public const float DefaultUpperTorsoRadius = 0.16f;  // Spine1 -> Spine3 / Chest
        public const float DefaultThighRadius = 0.09f;       // Hip -> Knee
        public const float DefaultUpperArmRadius = 0.05f;    // Shoulder -> Elbow
        public const float DefaultForearmRadius = 0.04f;     // Elbow -> Wrist
        public const float DefaultHandRadius = 0.045f;       // Wrist / Hand sphere

        /// <summary>
        /// Solves self-penetration constraints across a range of frames.
        /// </summary>
        public static bool SolvePenetration(
            EditableMotionData data,
            int startFrame,
            int endFrame,
            PenetrationOptions options = null)
        {
            if (data == null || data.Frames == 0) return false;
            options = options ?? new PenetrationOptions();

            int start = Mathf.Clamp(Mathf.Min(startFrame, endFrame), 0, data.Frames - 1);
            int end = Mathf.Clamp(Mathf.Max(startFrame, endFrame), 0, data.Frames - 1);

            bool anyModified = false;
            for (int t = start; t <= end; t++)
            {
                if (SolveFramePenetration(data, t, options))
                {
                    anyModified = true;
                }
            }

            return anyModified;
        }

        /// <summary>
        /// Solves self-penetration constraints for a single frame.
        /// </summary>
        public static bool SolveFramePenetration(
            EditableMotionData data,
            int frame,
            PenetrationOptions options = null)
        {
            if (data == null || frame < 0 || frame >= data.Frames) return false;
            options = options ?? new PenetrationOptions();

            // 1. Evaluate Forward Kinematics for current pose
            int jointCount = SmplxJointDefinitions.JointCount;
            Quaternion[] frameRots = new Quaternion[jointCount];
            for (int j = 0; j < jointCount; j++) frameRots[j] = data.GetJointRotation(frame, (SmplxJoint)j);

            Vector3 rootPos = data.GetRootPosition(frame);
            MotionIkUtility.ComputeForwardKinematics(rootPos, frameRots, out Vector3[] worldPos, out Quaternion[] worldRots);

            // 2. Build obstacle capsules (Torso & Thighs)
            float torsoRadOffset = options.TorsoRadiusOffset;
            float thighRadOffset = options.ThighRadiusOffset;

            // Lower Torso: Pelvis to Spine1
            SegmentCapsule lowerTorso = new SegmentCapsule(
                worldPos[(int)SmplxJoint.Pelvis],
                worldPos[(int)SmplxJoint.Spine1],
                DefaultLowerTorsoRadius + torsoRadOffset);

            // Upper Torso / Chest: Spine1 to Spine3
            SegmentCapsule upperTorso = new SegmentCapsule(
                worldPos[(int)SmplxJoint.Spine1],
                worldPos[(int)SmplxJoint.Spine3],
                DefaultUpperTorsoRadius + torsoRadOffset);

            // Left Thigh: L_Hip to L_Knee
            SegmentCapsule leftThigh = new SegmentCapsule(
                worldPos[(int)SmplxJoint.L_Hip],
                worldPos[(int)SmplxJoint.L_Knee],
                DefaultThighRadius + thighRadOffset);

            // Right Thigh: R_Hip to R_Knee
            SegmentCapsule rightThigh = new SegmentCapsule(
                worldPos[(int)SmplxJoint.R_Hip],
                worldPos[(int)SmplxJoint.R_Knee],
                DefaultThighRadius + thighRadOffset);

            var obstacles = new SegmentCapsule[] { lowerTorso, upperTorso, leftThigh, rightThigh };

            // 3. Detect intentional contact poses (Crossed arms, Hand on chest)
            float leftRelaxation = 0f;
            float rightRelaxation = 0f;

            Vector3 spine3Pos = worldPos[(int)SmplxJoint.Spine3];
            Quaternion spine3Rot = worldRots[(int)SmplxJoint.Spine3];
            Vector3 chestForward = spine3Rot * Vector3.forward;
            Vector3 chestRight = spine3Rot * Vector3.right;

            Vector3 lWrist = worldPos[(int)SmplxJoint.L_Wrist];
            Vector3 rWrist = worldPos[(int)SmplxJoint.R_Wrist];
            Vector3 lElbow = worldPos[(int)SmplxJoint.L_Elbow];
            Vector3 rElbow = worldPos[(int)SmplxJoint.R_Elbow];

            // Heuristic A: Crossed Arms
            if (options.ProtectCrossedArms)
            {
                float wristDist = Vector3.Distance(lWrist, rWrist);
                float toChestDist = (Vector3.Distance(lWrist, spine3Pos) + Vector3.Distance(rWrist, spine3Pos)) * 0.5f;

                // Both wrists are close to each other and positioned in front of chest
                bool armsInFront = Vector3.Dot(lWrist - spine3Pos, chestForward) > -0.05f &&
                                   Vector3.Dot(rWrist - spine3Pos, chestForward) > -0.05f;

                if (wristDist < 0.28f && toChestDist < 0.38f && armsInFront)
                {
                    float crossingWeight = Mathf.Clamp01(1.0f - (wristDist / 0.28f)) * options.RelaxationThreshold;
                    leftRelaxation = Mathf.Max(leftRelaxation, crossingWeight);
                    rightRelaxation = Mathf.Max(rightRelaxation, crossingWeight);
                }
            }

            // Heuristic B: Hand on Chest / Torso
            if (options.ProtectHandOnChest)
            {
                // Left hand near chest surface
                float lChestDot = Vector3.Dot(lWrist - spine3Pos, chestForward);
                float lChestDist = Vector3.Distance(lWrist, spine3Pos);
                if (lChestDot > 0.02f && lChestDist < 0.26f)
                {
                    leftRelaxation = Mathf.Max(leftRelaxation, options.RelaxationThreshold * 0.75f);
                }

                // Right hand near chest surface
                float rChestDot = Vector3.Dot(rWrist - spine3Pos, chestForward);
                float rChestDist = Vector3.Distance(rWrist, spine3Pos);
                if (rChestDot > 0.02f && rChestDist < 0.26f)
                {
                    rightRelaxation = Mathf.Max(rightRelaxation, options.RelaxationThreshold * 0.75f);
                }
            }

            bool modified = false;

            // 4. Solve Left Arm
            if (SolveLimbPenetration(data, frame, true, obstacles, worldPos, worldRots, options, leftRelaxation))
            {
                modified = true;
            }

            // 5. Solve Right Arm
            if (SolveLimbPenetration(data, frame, false, obstacles, worldPos, worldRots, options, rightRelaxation))
            {
                modified = true;
            }

            return modified;
        }

        private static bool SolveLimbPenetration(
            EditableMotionData data,
            int frame,
            bool isLeft,
            SegmentCapsule[] obstacles,
            Vector3[] worldPos,
            Quaternion[] worldRots,
            PenetrationOptions options,
            float relaxation)
        {
            SmplxJoint shoulderJoint = isLeft ? SmplxJoint.L_Shoulder : SmplxJoint.R_Shoulder;
            SmplxJoint elbowJoint = isLeft ? SmplxJoint.L_Elbow : SmplxJoint.R_Elbow;
            SmplxJoint wristJoint = isLeft ? SmplxJoint.L_Wrist : SmplxJoint.R_Wrist;

            Vector3 shoulderPos = worldPos[(int)shoulderJoint];
            Vector3 elbowPos = worldPos[(int)elbowJoint];
            Vector3 wristPos = worldPos[(int)wristJoint];

            // Sample points along the arm: Elbow, mid-forearm, wrist
            Vector3 midForearm = (elbowPos + wristPos) * 0.5f;

            Vector3 wristCorrection = Vector3.zero;
            Vector3 elbowCorrection = Vector3.zero;

            float effectivePushWeight = options.PushBlendWeight * (1.0f - relaxation);
            if (effectivePushWeight <= 0.01f) return false;

            // Check penetration against all torso & thigh capsules
            foreach (var obstacle in obstacles)
            {
                // Test Wrist
                float requiredDistWrist = obstacle.Radius + DefaultHandRadius + options.Margin;
                float distWrist = obstacle.DistanceToPoint(wristPos, out Vector3 closestAxisWrist);
                if (distWrist < requiredDistWrist)
                {
                    float pen = requiredDistWrist - distWrist;
                    Vector3 pushDir = distWrist > 0.001f ? (wristPos - closestAxisWrist).normalized : (isLeft ? -Vector3.right : Vector3.right);
                    wristCorrection += pushDir * pen;
                }

                // Test Mid-forearm
                float requiredDistMid = obstacle.Radius + DefaultForearmRadius + options.Margin;
                float distMid = obstacle.DistanceToPoint(midForearm, out Vector3 closestAxisMid);
                if (distMid < requiredDistMid)
                {
                    float pen = requiredDistMid - distMid;
                    Vector3 pushDir = distMid > 0.001f ? (midForearm - closestAxisMid).normalized : (isLeft ? -Vector3.right : Vector3.right);
                    wristCorrection += pushDir * (pen * 0.7f);
                }

                // Test Elbow
                float requiredDistElbow = obstacle.Radius + DefaultUpperArmRadius + options.Margin;
                float distElbow = obstacle.DistanceToPoint(elbowPos, out Vector3 closestAxisElbow);
                if (distElbow < requiredDistElbow)
                {
                    float pen = requiredDistElbow - distElbow;
                    Vector3 pushDir = distElbow > 0.001f ? (elbowPos - closestAxisElbow).normalized : (isLeft ? -Vector3.right : Vector3.right);
                    elbowCorrection += pushDir * pen;
                }
            }

            if (wristCorrection.sqrMagnitude < 0.0001f && elbowCorrection.sqrMagnitude < 0.0001f)
            {
                return false;
            }

            // Apply push-out to Wrist via Two-Bone IK
            Vector3 targetWristPos = wristPos + (wristCorrection * effectivePushWeight);

            // Optional elbow pole vector adjustment if elbow penetrated
            Vector3 naturalElbow = elbowPos + (elbowCorrection * effectivePushWeight);

            // Use Two-Bone IK to reposition wrist and naturally rotate shoulder & elbow
            bool ikSuccess = MotionIkUtility.ApplyLimbIK(
                data,
                frame,
                shoulderJoint,
                elbowJoint,
                wristJoint,
                targetWristPos,
                naturalElbow,
                recordUndo: false);

            return ikSuccess;
        }

        /// <summary>
        /// Diagnoses number of limb-to-torso penetrations on a given frame.
        /// </summary>
        public static int CountPenetrations(
            EditableMotionData data,
            int frame,
            PenetrationOptions options = null)
        {
            if (data == null || frame < 0 || frame >= data.Frames) return 0;
            options = options ?? new PenetrationOptions();

            int jointCount = SmplxJointDefinitions.JointCount;
            Quaternion[] frameRots = new Quaternion[jointCount];
            for (int j = 0; j < jointCount; j++) frameRots[j] = data.GetJointRotation(frame, (SmplxJoint)j);

            Vector3 rootPos = data.GetRootPosition(frame);
            MotionIkUtility.ComputeForwardKinematics(rootPos, frameRots, out Vector3[] worldPos, out _);

            SegmentCapsule lowerTorso = new SegmentCapsule(
                worldPos[(int)SmplxJoint.Pelvis],
                worldPos[(int)SmplxJoint.Spine1],
                DefaultLowerTorsoRadius + options.TorsoRadiusOffset);

            SegmentCapsule upperTorso = new SegmentCapsule(
                worldPos[(int)SmplxJoint.Spine1],
                worldPos[(int)SmplxJoint.Spine3],
                DefaultUpperTorsoRadius + options.TorsoRadiusOffset);

            SegmentCapsule leftThigh = new SegmentCapsule(
                worldPos[(int)SmplxJoint.L_Hip],
                worldPos[(int)SmplxJoint.L_Knee],
                DefaultThighRadius + options.ThighRadiusOffset);

            SegmentCapsule rightThigh = new SegmentCapsule(
                worldPos[(int)SmplxJoint.R_Hip],
                worldPos[(int)SmplxJoint.R_Knee],
                DefaultThighRadius + options.ThighRadiusOffset);

            var obstacles = new SegmentCapsule[] { lowerTorso, upperTorso, leftThigh, rightThigh };
            var armPoints = new Vector3[]
            {
                worldPos[(int)SmplxJoint.L_Elbow],
                worldPos[(int)SmplxJoint.L_Wrist],
                worldPos[(int)SmplxJoint.R_Elbow],
                worldPos[(int)SmplxJoint.R_Wrist],
            };

            int count = 0;
            foreach (var pt in armPoints)
            {
                foreach (var obs in obstacles)
                {
                    if (obs.DistanceToPoint(pt, out _) < obs.Radius + DefaultForearmRadius)
                    {
                        count++;
                        break;
                    }
                }
            }

            return count;
        }
    }
}
