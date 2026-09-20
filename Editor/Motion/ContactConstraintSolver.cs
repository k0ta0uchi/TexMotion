using System;
using System.Collections.Generic;
using TexMotion.Runtime.Motion;
using UnityEngine;

namespace TexMotion.Editor.Motion
{
    /// <summary>
    /// Measured or fallback leg dimensions and joint offsets for an avatar.
    /// Used by ContactConstraintSolver to ensure exact ground contact regardless of avatar proportions.
    /// </summary>
    public class AvatarLegDimensions
    {
        public float UpperLegLengthL = 0.42f;
        public float LowerLegLengthL = 0.41f;
        public float FootLengthL = 0.15f;
        public float SoleOffsetL = 0.08f;

        public float UpperLegLengthR = 0.42f;
        public float LowerLegLengthR = 0.41f;
        public float FootLengthR = 0.15f;
        public float SoleOffsetR = 0.08f;

        public Vector3 PelvisToHipL = new Vector3(-0.09f, -0.06f, 0.0f);
        public Vector3 PelvisToHipR = new Vector3(0.09f, -0.06f, 0.0f);

        public bool IsCustomAvatar = false;
        public string AvatarName = "Standard Humanoid Fallback";

        /// <summary>
        /// Extracts accurate bone lengths and sole offset from an avatar Animator.
        /// Falls back to standard SMPL-X / humanoid anatomical proportions if avatar is null or invalid.
        /// </summary>
        public static AvatarLegDimensions ExtractFromAvatar(Animator avatar)
        {
            var dims = new AvatarLegDimensions();
            if (avatar == null || !avatar.isHuman)
            {
                return dims;
            }

            Transform hips = avatar.GetBoneTransform(HumanBodyBones.Hips);
            Transform lUpper = avatar.GetBoneTransform(HumanBodyBones.LeftUpperLeg);
            Transform lLower = avatar.GetBoneTransform(HumanBodyBones.LeftLowerLeg);
            Transform lFoot = avatar.GetBoneTransform(HumanBodyBones.LeftFoot);
            Transform lToes = avatar.GetBoneTransform(HumanBodyBones.LeftToes);

            Transform rUpper = avatar.GetBoneTransform(HumanBodyBones.RightUpperLeg);
            Transform rLower = avatar.GetBoneTransform(HumanBodyBones.RightLowerLeg);
            Transform rFoot = avatar.GetBoneTransform(HumanBodyBones.RightFoot);
            Transform rToes = avatar.GetBoneTransform(HumanBodyBones.RightToes);

            if (hips == null || lUpper == null || lLower == null || lFoot == null)
            {
                return dims;
            }

            dims.IsCustomAvatar = true;
            dims.AvatarName = avatar.gameObject.name;

            // Measure left leg lengths
            float lUpperLen = Vector3.Distance(lUpper.position, lLower.position);
            float lLowerLen = Vector3.Distance(lLower.position, lFoot.position);
            float lFootLen = lToes != null ? Vector3.Distance(lFoot.position, lToes.position) : 0.15f;

            if (lUpperLen > 0.05f && lUpperLen < 2.0f) dims.UpperLegLengthL = lUpperLen;
            if (lLowerLen > 0.05f && lLowerLen < 2.0f) dims.LowerLegLengthL = lLowerLen;
            if (lFootLen > 0.02f && lFootLen < 0.8f) dims.FootLengthL = lFootLen;

            // Measure right leg lengths
            if (rUpper != null && rLower != null && rFoot != null)
            {
                float rUpperLen = Vector3.Distance(rUpper.position, rLower.position);
                float rLowerLen = Vector3.Distance(rLower.position, rFoot.position);
                float rFootLen = rToes != null ? Vector3.Distance(rFoot.position, rToes.position) : dims.FootLengthL;

                if (rUpperLen > 0.05f && rUpperLen < 2.0f) dims.UpperLegLengthR = rUpperLen;
                if (rLowerLen > 0.05f && rLowerLen < 2.0f) dims.LowerLegLengthR = rLowerLen;
                if (rFootLen > 0.02f && rFootLen < 0.8f) dims.FootLengthR = rFootLen;
            }
            else
            {
                dims.UpperLegLengthR = dims.UpperLegLengthL;
                dims.LowerLegLengthR = dims.LowerLegLengthL;
                dims.FootLengthR = dims.FootLengthL;
            }

            // Measure Pelvis to Hip relative offsets
            Vector3 hipLocalL = hips.InverseTransformPoint(lUpper.position);
            if (hipLocalL.sqrMagnitude > 0.0001f && hipLocalL.sqrMagnitude < 4.0f)
            {
                dims.PelvisToHipL = hipLocalL;
            }

            if (rUpper != null)
            {
                Vector3 hipLocalR = hips.InverseTransformPoint(rUpper.position);
                if (hipLocalR.sqrMagnitude > 0.0001f && hipLocalR.sqrMagnitude < 4.0f)
                {
                    dims.PelvisToHipR = hipLocalR;
                }
            }
            else
            {
                dims.PelvisToHipR = new Vector3(-dims.PelvisToHipL.x, dims.PelvisToHipL.y, dims.PelvisToHipL.z);
            }

            // Measure Sole Offsets (distance from foot bone to ground / sole)
            dims.SoleOffsetL = ComputeSoleOffset(avatar, lFoot, lToes, dims.LowerLegLengthL);
            dims.SoleOffsetR = rFoot != null
                ? ComputeSoleOffset(avatar, rFoot, rToes, dims.LowerLegLengthR)
                : dims.SoleOffsetL;

            return dims;
        }

        private static float ComputeSoleOffset(Animator avatar, Transform foot, Transform toes, float lowerLegLength)
        {
            if (foot == null) return lowerLegLength * 0.18f;

            // Strategy 1: Check avatar mesh bounds
            var renderers = avatar.GetComponentsInChildren<Renderer>();
            float minMeshY = float.MaxValue;
            bool foundMesh = false;
            foreach (var r in renderers)
            {
                if (r != null && r.enabled)
                {
                    Bounds b = r.bounds;
                    if (b.size.sqrMagnitude > 0.001f)
                    {
                        if (b.min.y < minMeshY)
                        {
                            minMeshY = b.min.y;
                            foundMesh = true;
                        }
                    }
                }
            }

            if (foundMesh)
            {
                float diff = foot.position.y - minMeshY;
                if (diff >= 0.02f && diff <= 0.35f)
                {
                    return diff;
                }
            }

            // Strategy 2: If toes exist, foot to toes height difference + toe bottom margin
            if (toes != null)
            {
                float toeDiff = foot.position.y - toes.position.y;
                if (toeDiff > 0.01f && toeDiff < 0.25f)
                {
                    return toeDiff + 0.025f;
                }
            }

            // Strategy 3: Avatar root position to foot height
            if (avatar != null)
            {
                float rootDiff = foot.position.y - avatar.transform.position.y;
                if (rootDiff >= 0.03f && rootDiff <= 0.25f)
                {
                    return rootDiff;
                }
            }

            // Fallback: anatomical proportion ~18% of lower leg length
            return Mathf.Clamp(lowerLegLength * 0.18f, 0.05f, 0.15f);
        }
    }

    /// <summary>
    /// Avatar bone-length linked contact constraint solver.
    /// Locks foot contact positions to prevent foot sliding, accurately places foot soles on the ground,
    /// and preserves jumping / airborne dynamics without unnatural grounding.
    /// </summary>
    public static class ContactConstraintSolver
    {
        private const int BLEND_FRAME_WINDOW = 3;

        /// <summary>
        /// Solves grounding constraints on the motion data using the avatar's real bone dimensions.
        /// </summary>
        public static bool ApplyAvatarGrounding(EditableMotionData motionData, Animator avatar, float blendWeight = 1.0f)
        {
            return Solve(motionData, avatar, blendWeight, 0.0f);
        }

        /// <summary>
        /// Solves grounding constraints on the motion data using the avatar's real bone dimensions.
        /// </summary>
        /// <param name="motionData">Target editable motion data.</param>
        /// <param name="avatar">Target avatar Animator (or null for default human proportions).</param>
        /// <param name="blendWeight">Overall constraint strength [0.0..1.0].</param>
        /// <param name="groundY">Floor plane height in world space.</param>
        /// <returns>True if constraint was successfully applied.</returns>
        public static bool Solve(
            EditableMotionData motionData,
            Animator avatar = null,
            float blendWeight = 1.0f,
            float groundY = 0.0f)
        {
            if (motionData == null || motionData.Frames <= 0) return false;
            blendWeight = Mathf.Clamp01(blendWeight);
            if (blendWeight <= 0.0001f) return true;

            // 1. Measure avatar leg dimensions
            AvatarLegDimensions dims = AvatarLegDimensions.ExtractFromAvatar(avatar);

            // 2. Ensure contact track exists
            ContactTrackData contactTrack = motionData.ContactTrack;
            if (contactTrack == null || contactTrack.intervals == null || contactTrack.intervals.Length == 0)
            {
                // Auto-detect intervals if none present
                DetectContactIntervals(motionData, dims: dims);
                contactTrack = motionData.ContactTrack;
                if (contactTrack == null || contactTrack.intervals == null || contactTrack.intervals.Length == 0)
                {
                    return false;
                }
            }

            // 3. Prepare anchor positions for each contact interval
            PrepareAnchors(motionData, contactTrack, dims, groundY);

            // 4. Solve frame by frame
            int frames = motionData.Frames;
            float fps = motionData.FrameRate > 0f ? motionData.FrameRate : 30.0f;

            for (int t = 0; t < frames; t++)
            {
                float time = motionData.GetTimestamp(t);

                // Check active contact intervals for left and right feet
                ContactIntervalData leftInterval = FindActiveInterval(contactTrack, "left", time);
                ContactIntervalData rightInterval = FindActiveInterval(contactTrack, "right", time);

                // Compute Pelvis FK
                Vector3 rootPos = motionData.GetRootPosition(t);
                Quaternion pelvisRot = motionData.GetJointRotation(t, SmplxJoint.Pelvis);

                // Adjust pelvis Y if feet are contacting and avatar cannot reach ground anchor
                if (leftInterval != null || rightInterval != null)
                {
                    rootPos = AdjustPelvisHeightForContact(
                        motionData, t, rootPos, pelvisRot, dims, groundY, leftInterval, rightInterval, blendWeight);
                }

                // Solve Left Foot
                if (leftInterval != null)
                {
                    SolveFootGrounding(
                        motionData, t, rootPos, pelvisRot, dims,
                        isLeft: true,
                        interval: leftInterval,
                        groundY: groundY,
                        blendWeight: blendWeight);
                }

                // Solve Right Foot
                if (rightInterval != null)
                {
                    SolveFootGrounding(
                        motionData, t, rootPos, pelvisRot, dims,
                        isLeft: false,
                        interval: rightInterval,
                        groundY: groundY,
                        blendWeight: blendWeight);
                }
            }

            return true;
        }

        /// <summary>
        /// Prepares or validates 3D anchor positions for all intervals in the contact track.
        /// </summary>
        private static void PrepareAnchors(
            EditableMotionData motionData,
            ContactTrackData contactTrack,
            AvatarLegDimensions dims,
            float groundY)
        {
            float fps = motionData.FrameRate > 0f ? motionData.FrameRate : 30.0f;

            foreach (var interval in contactTrack.intervals)
            {
                if (interval == null) continue;

                bool isLeft = !string.Equals(interval.foot, "right", StringComparison.OrdinalIgnoreCase);

                // If anchor is already specified and valid, keep it
                if (interval.anchor != null && interval.anchor.Length >= 3)
                {
                    continue;
                }

                // Calculate average or median FK position during the contact interval
                int startFrame = Mathf.Clamp(Mathf.RoundToInt(interval.start * fps), 0, motionData.Frames - 1);
                int endFrame = Mathf.Clamp(Mathf.RoundToInt(interval.end * fps), 0, motionData.Frames - 1);
                int sampleCount = 0;
                Vector3 sumPos = Vector3.zero;

                for (int f = startFrame; f <= endFrame; f++)
                {
                    ComputeLegFK(motionData, f, dims, isLeft, out _, out _, out Vector3 anklePos, out _);
                    sumPos += anklePos;
                    sampleCount++;
                }

                Vector3 anchorPos = sampleCount > 0 ? (sumPos / sampleCount) : Vector3.zero;

                float soleOffset = isLeft ? dims.SoleOffsetL : dims.SoleOffsetR;
                anchorPos.y = groundY + soleOffset;

                interval.anchor = new float[] { anchorPos.x, anchorPos.y, anchorPos.z };
            }
        }

        /// <summary>
        /// Solves Two-Bone IK and ankle alignment for a contacting foot at frame t.
        /// </summary>
        private static void SolveFootGrounding(
            EditableMotionData motionData,
            int frame,
            Vector3 rootPos,
            Quaternion pelvisRot,
            AvatarLegDimensions dims,
            bool isLeft,
            ContactIntervalData interval,
            float groundY,
            float blendWeight)
        {
            SmplxJoint hipJoint = isLeft ? SmplxJoint.L_Hip : SmplxJoint.R_Hip;
            SmplxJoint kneeJoint = isLeft ? SmplxJoint.L_Knee : SmplxJoint.R_Knee;
            SmplxJoint ankleJoint = isLeft ? SmplxJoint.L_Ankle : SmplxJoint.R_Ankle;

            float upperLen = isLeft ? dims.UpperLegLengthL : dims.UpperLegLengthR;
            float lowerLen = isLeft ? dims.LowerLegLengthL : dims.LowerLegLengthR;
            float footLen = isLeft ? dims.FootLengthL : dims.FootLengthR;
            float soleOffset = isLeft ? dims.SoleOffsetL : dims.SoleOffsetR;
            Vector3 pelvisToHip = isLeft ? dims.PelvisToHipL : dims.PelvisToHipR;

            Quaternion origHipLocal = motionData.GetJointRotation(frame, hipJoint);
            Quaternion origKneeLocal = motionData.GetJointRotation(frame, kneeJoint);
            Quaternion origAnkleLocal = motionData.GetJointRotation(frame, ankleJoint);

            // 1. Compute current leg forward kinematics
            Quaternion curHipWorld = pelvisRot * origHipLocal;
            Vector3 curHipPos = rootPos + (pelvisRot * pelvisToHip);

            Quaternion curKneeWorld = curHipWorld * origKneeLocal;
            Vector3 curKneePos = curHipPos + (curHipWorld * new Vector3(0, -upperLen, 0));

            Quaternion curAnkleWorld = curKneeWorld * origAnkleLocal;
            Vector3 curAnklePos = curKneePos + (curKneeWorld * new Vector3(0, -lowerLen, 0));

            // 2. Determine target ankle position based on anchor and contact mode
            Vector3 anchor = (interval.anchor != null && interval.anchor.Length >= 3)
                ? new Vector3(interval.anchor[0], interval.anchor[1], interval.anchor[2])
                : curAnklePos;

            float targetAnkleY;
            string mode = interval.mode?.ToLowerInvariant() ?? "flat";
            if (mode == "toe")
            {
                // Toe contact: toe on floor, heel lifted
                targetAnkleY = groundY + soleOffset + (footLen * 0.4f);
            }
            else if (mode == "heel")
            {
                // Heel contact: heel on floor, toe lifted
                targetAnkleY = groundY + (soleOffset * 0.75f);
            }
            else
            {
                // Flat contact: full foot sole parallel on floor
                targetAnkleY = groundY + soleOffset;
            }

            Vector3 targetAnklePos = new Vector3(anchor.x, targetAnkleY, anchor.z);

            // 3. Compute contact boundary easing (blend in / blend out at interval edges)
            float fps = motionData.FrameRate > 0f ? motionData.FrameRate : 30.0f;
            float time = motionData.GetTimestamp(frame);
            float edgeWeight = ComputeIntervalEdgeWeight(time, interval.start, interval.end, fps);

            float effectiveWeight = edgeWeight * blendWeight * Mathf.Clamp01(interval.confidence);
            if (effectiveWeight <= 0.001f) return;

            Vector3 blendedTargetPos = Vector3.Lerp(curAnklePos, targetAnklePos, effectiveWeight);

            // 4. Pole vector: preserve the original knee bend direction
            Vector3 limbDir = (curAnklePos - curHipPos).normalized;
            Vector3 kneeOffset = curKneePos - curHipPos;
            Vector3 kneeProj = Vector3.Project(kneeOffset, limbDir);
            Vector3 bendNormal = (kneeOffset - kneeProj).normalized;

            if (bendNormal.sqrMagnitude < 0.001f)
            {
                bendNormal = curHipWorld * Vector3.forward;
            }

            Vector3 polePos = curKneePos + (bendNormal * 0.5f);

            // 5. Solve Two-Bone IK
            if (!MotionIkUtility.SolveTwoBoneIK(
                curHipPos, curKneePos, curAnklePos, blendedTargetPos, polePos,
                out Quaternion hipDeltaRot, out Quaternion kneeDeltaRot))
            {
                return;
            }

            // 6. Apply rotation deltas
            Quaternion newHipWorld = hipDeltaRot * curHipWorld;
            Quaternion newHipLocal = MotionIkUtility.SafeInverse(pelvisRot) * newHipWorld;

            Quaternion newKneeWorld = kneeDeltaRot * (hipDeltaRot * curKneeWorld);
            Quaternion newKneeLocal = MotionIkUtility.SafeInverse(newHipWorld) * newKneeWorld;

            // 7. Adjust ankle rotation to keep foot sole properly aligned with the ground
            Quaternion finalAnkleLocal = origAnkleLocal;
            Vector3 curForward = curAnkleWorld * Vector3.forward;
            curForward.y = 0f;
            if (curForward.sqrMagnitude > 0.001f)
            {
                curForward.Normalize();
                float yawHalfRad = Mathf.Atan2(curForward.x, curForward.z) * 0.5f;
                Quaternion yawOnlyWorld = new Quaternion(0f, Mathf.Sin(yawHalfRad), 0f, Mathf.Cos(yawHalfRad));

                if (mode == "flat")
                {
                    // Foot flat: sole parallel with floor UP
                    Quaternion targetAnkleLocal = MotionIkUtility.SafeInverse(newKneeWorld) * yawOnlyWorld;
                    finalAnkleLocal = MotionIkUtility.SafeSlerp(origAnkleLocal, targetAnkleLocal, effectiveWeight);
                }
                else if (mode == "toe")
                {
                    // Toe contact: pitch forward ~30 deg
                    // Quaternion for pitch 30 deg: (sin(15 deg), 0, 0, cos(15 deg)) = (0.258819f, 0, 0, 0.965926f)
                    Quaternion pitch30 = new Quaternion(0.258819f, 0f, 0f, 0.965926f);
                    Quaternion toeAnkleWorld = yawOnlyWorld * pitch30;
                    Quaternion targetAnkleLocal = MotionIkUtility.SafeInverse(newKneeWorld) * toeAnkleWorld;
                    finalAnkleLocal = MotionIkUtility.SafeSlerp(origAnkleLocal, targetAnkleLocal, effectiveWeight);
                }
            }

            // Slerp with original rotations based on effective weight
            Quaternion finalHipLocal = MotionIkUtility.SafeSlerp(origHipLocal, newHipLocal, effectiveWeight);
            Quaternion finalKneeLocal = MotionIkUtility.SafeSlerp(origKneeLocal, newKneeLocal, effectiveWeight);

            motionData.SetJointRotation(frame, hipJoint, finalHipLocal);
            motionData.SetJointRotation(frame, kneeJoint, finalKneeLocal);
            motionData.SetJointRotation(frame, ankleJoint, finalAnkleLocal);
        }

        /// <summary>
        /// Computes forward kinematics for a single leg.
        /// </summary>
        public static void ComputeLegFK(
            EditableMotionData motionData,
            int frame,
            AvatarLegDimensions dims,
            bool isLeft,
            out Vector3 hipPos,
            out Vector3 kneePos,
            out Vector3 anklePos,
            out Vector3 footPos)
        {
            Vector3 rootPos = motionData.GetRootPosition(frame);
            Quaternion pelvisRot = motionData.GetJointRotation(frame, SmplxJoint.Pelvis);

            SmplxJoint hipJoint = isLeft ? SmplxJoint.L_Hip : SmplxJoint.R_Hip;
            SmplxJoint kneeJoint = isLeft ? SmplxJoint.L_Knee : SmplxJoint.R_Knee;
            SmplxJoint ankleJoint = isLeft ? SmplxJoint.L_Ankle : SmplxJoint.R_Ankle;

            float upperLen = isLeft ? dims.UpperLegLengthL : dims.UpperLegLengthR;
            float lowerLen = isLeft ? dims.LowerLegLengthL : dims.LowerLegLengthR;
            float footLen = isLeft ? dims.FootLengthL : dims.FootLengthR;
            float soleOffset = isLeft ? dims.SoleOffsetL : dims.SoleOffsetR;
            Vector3 pelvisToHip = isLeft ? dims.PelvisToHipL : dims.PelvisToHipR;

            Quaternion hipRot = pelvisRot * motionData.GetJointRotation(frame, hipJoint);
            hipPos = rootPos + (pelvisRot * pelvisToHip);

            Quaternion kneeRot = hipRot * motionData.GetJointRotation(frame, kneeJoint);
            kneePos = hipPos + (hipRot * new Vector3(0, -upperLen, 0));

            Quaternion ankleRot = kneeRot * motionData.GetJointRotation(frame, ankleJoint);
            anklePos = kneePos + (kneeRot * new Vector3(0, -lowerLen, 0));

            footPos = anklePos + (ankleRot * new Vector3(0, -soleOffset, footLen));
        }

        /// <summary>
        /// Adjusts Pelvis Y position if feet are contacting and avatar leg length requires it.
        /// Prevents overstretched or hyperextended knees.
        /// </summary>
        private static Vector3 AdjustPelvisHeightForContact(
            EditableMotionData motionData,
            int frame,
            Vector3 rootPos,
            Quaternion pelvisRot,
            AvatarLegDimensions dims,
            float groundY,
            ContactIntervalData leftInterval,
            ContactIntervalData rightInterval,
            float blendWeight)
        {
            float shiftDown = 0f;

            if (leftInterval != null)
            {
                Vector3 hipPosL = rootPos + (pelvisRot * dims.PelvisToHipL);
                float targetAnkleYL = groundY + dims.SoleOffsetL;
                Vector3 anchorL = (leftInterval.anchor != null && leftInterval.anchor.Length >= 3)
                    ? new Vector3(leftInterval.anchor[0], targetAnkleYL, leftInterval.anchor[2])
                    : new Vector3(hipPosL.x, targetAnkleYL, hipPosL.z);

                float distL = Vector3.Distance(hipPosL, anchorL);
                float maxReachL = (dims.UpperLegLengthL + dims.LowerLegLengthL) * 0.98f;
                if (distL > maxReachL)
                {
                    shiftDown = Mathf.Max(shiftDown, distL - maxReachL);
                }
            }

            if (rightInterval != null)
            {
                Vector3 hipPosR = rootPos + (pelvisRot * dims.PelvisToHipR);
                float targetAnkleYR = groundY + dims.SoleOffsetR;
                Vector3 anchorR = (rightInterval.anchor != null && rightInterval.anchor.Length >= 3)
                    ? new Vector3(rightInterval.anchor[0], targetAnkleYR, rightInterval.anchor[2])
                    : new Vector3(hipPosR.x, targetAnkleYR, hipPosR.z);

                float distR = Vector3.Distance(hipPosR, anchorR);
                float maxReachR = (dims.UpperLegLengthR + dims.LowerLegLengthR) * 0.98f;
                if (distR > maxReachR)
                {
                    shiftDown = Mathf.Max(shiftDown, distR - maxReachR);
                }
            }

            if (shiftDown > 0.001f)
            {
                float appliedShift = Mathf.Min(shiftDown, 0.35f) * blendWeight;
                rootPos.y -= appliedShift;
                motionData.SetRootPosition(frame, rootPos);
            }

            return rootPos;
        }

        /// <summary>
        /// Computes smooth-step easing weight near interval start and end to avoid popping.
        /// </summary>
        private static float ComputeIntervalEdgeWeight(float time, float start, float end, float fps)
        {
            if (time < start || time > end) return 0f;

            float blendDuration = Mathf.Max(0.04f, BLEND_FRAME_WINDOW / fps);
            float intervalDuration = end - start;

            if (intervalDuration < blendDuration * 2f)
            {
                blendDuration = intervalDuration * 0.5f;
            }

            if (blendDuration <= 0.001f) return 1f;

            float distFromStart = time - start;
            float distFromEnd = end - time;
            float minDist = Mathf.Min(distFromStart, distFromEnd);

            if (minDist < blendDuration)
            {
                float t = minDist / blendDuration;
                return t * t * (3f - 2f * t); // SmoothStep
            }

            return 1f;
        }

        /// <summary>
        /// Finds the active contact interval for the specified foot at a given time.
        /// </summary>
        public static ContactIntervalData FindActiveInterval(ContactTrackData track, string foot, float time)
        {
            if (track?.intervals == null) return null;
            for (int i = 0; i < track.intervals.Length; i++)
            {
                var interval = track.intervals[i];
                if (interval == null) continue;
                if (string.Equals(interval.foot, foot, StringComparison.OrdinalIgnoreCase))
                {
                    if (time >= interval.start && time <= interval.end)
                    {
                        return interval;
                    }
                }
            }
            return null;
        }

        /// <summary>
        /// Auto-detects foot contact intervals from foot speed and height profiles.
        /// </summary>
        public static void DetectContactIntervals(
            EditableMotionData motionData,
            float velocityThreshold = 0.20f,
            float heightThreshold = 0.12f,
            AvatarLegDimensions dims = null)
        {
            if (motionData == null || motionData.Frames < 2) return;

            if (dims == null)
            {
                dims = AvatarLegDimensions.ExtractFromAvatar(motionData.TargetAvatar);
            }

            int frames = motionData.Frames;
            float fps = motionData.FrameRate > 0f ? motionData.FrameRate : 30.0f;
            float dt = 1.0f / fps;

            var detectedIntervals = new List<ContactIntervalData>();

            // Detect Left and Right separately
            DetectLimbIntervals(motionData, dims, isLeft: true, velocityThreshold, heightThreshold, dt, detectedIntervals);
            DetectLimbIntervals(motionData, dims, isLeft: false, velocityThreshold, heightThreshold, dt, detectedIntervals);

            if (motionData.ContactTrack == null)
            {
                motionData.ContactTrack = new ContactTrackData();
            }

            motionData.ContactTrack.intervals = detectedIntervals.ToArray();
        }

        private static void DetectLimbIntervals(
            EditableMotionData motionData,
            AvatarLegDimensions dims,
            bool isLeft,
            float velThreshold,
            float heightThreshold,
            float dt,
            List<ContactIntervalData> results)
        {
            int frames = motionData.Frames;
            var anklePositions = new Vector3[frames];

            // 1. Calculate ankle world positions for all frames
            for (int t = 0; t < frames; t++)
            {
                ComputeLegFK(motionData, t, dims, isLeft, out _, out _, out anklePositions[t], out _);
            }

            // Find lowest foot height across entire motion
            float minAnkleY = float.MaxValue;
            for (int t = 0; t < frames; t++)
            {
                if (anklePositions[t].y < minAnkleY) minAnkleY = anklePositions[t].y;
            }

            // 2. Identify contact states
            bool[] isContact = new bool[frames];
            for (int t = 0; t < frames; t++)
            {
                Vector3 vel;
                if (t == 0) vel = (anklePositions[1] - anklePositions[0]) / dt;
                else if (t == frames - 1) vel = (anklePositions[t] - anklePositions[t - 1]) / dt;
                else vel = (anklePositions[t + 1] - anklePositions[t - 1]) / (2f * dt);

                float speed = vel.magnitude;
                float heightAboveMin = anklePositions[t].y - minAnkleY;

                isContact[t] = (speed < velThreshold) && (heightAboveMin < heightThreshold);
            }

            // 3. Group contiguous frames into intervals (minimum 3 frames duration)
            int minFrames = 3;
            int spanStart = -1;
            string footStr = isLeft ? "left" : "right";

            for (int t = 0; t < frames; t++)
            {
                if (isContact[t])
                {
                    if (spanStart < 0) spanStart = t;
                }
                else
                {
                    if (spanStart >= 0)
                    {
                        int spanEnd = t - 1;
                        if (spanEnd - spanStart + 1 >= minFrames)
                        {
                            AddDetectedInterval(motionData, spanStart, spanEnd, footStr, anklePositions, results);
                        }
                        spanStart = -1;
                    }
                }
            }

            // Check trailing interval
            if (spanStart >= 0 && (frames - spanStart) >= minFrames)
            {
                AddDetectedInterval(motionData, spanStart, frames - 1, footStr, anklePositions, results);
            }
        }

        private static void AddDetectedInterval(
            EditableMotionData motionData,
            int startFrame,
            int endFrame,
            string foot,
            Vector3[] anklePositions,
            List<ContactIntervalData> results)
        {
            float startTime = motionData.GetTimestamp(startFrame);
            float endTime = motionData.GetTimestamp(endFrame);

            // Compute anchor as average position during contact
            Vector3 sumPos = Vector3.zero;
            int count = endFrame - startFrame + 1;
            for (int f = startFrame; f <= endFrame; f++)
            {
                sumPos += anklePositions[f];
            }
            Vector3 anchor = sumPos / count;

            results.Add(new ContactIntervalData
            {
                foot = foot,
                start = startTime,
                end = endTime,
                mode = "flat",
                confidence = 0.95f,
                anchor = new float[] { anchor.x, anchor.y, anchor.z }
            });
        }
    }
}
