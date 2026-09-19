using System;
using System.Collections.Generic;
using TexMotion.Runtime.Motion;
using UnityEngine;

namespace TexMotion.Editor.Motion
{
    /// <summary>
    /// Frame stepping modes for stylized Japanese limited animation (anime 2s / 3s).
    /// </summary>
    public enum AnimeStepMode
    {
        Off,
        DynamicAnime,   // Dynamically switches between 1s (fast), 2s (medium), and 3s (slow) based on velocity
        Strict2s,       // Enforces 2-frame hold (15 fps at 30 fps base)
        Strict3s        // Enforces 3-frame hold (10 fps at 30 fps base)
    }

    /// <summary>
    /// Preset configurations for the stylized hand-keyed motion polisher.
    /// </summary>
    public enum StylizedPolishPreset
    {
        Custom,
        SnappyAction,       // Strong snap & ease, anime 2s-3s stepping, dynamic push
        RealisticWeight,    // Heavy landing cushion, contrapposto, organic kinematic drag
        SubtlePolish        // Gentle exaggeration, smooth arcs, micro-cushioning
    }

    /// <summary>
    /// Configurable options for transforming dense AI-extracted mocap into stylized, hand-keyed quality animation.
    /// </summary>
    [System.Serializable]
    public class StylizedPolishOptions
    {
        // 1. Timing & Spacing
        public bool EnableSnapAndEase = true;
        [Range(0.1f, 1.0f)] public float SnapIntensity = 0.50f;
        [Range(5f, 40f)] public float HoldThresholdDegPerSec = 16.0f;

        public AnimeStepMode StepMode = AnimeStepMode.Off;

        public bool EnableKeyframeDecimator = false;
        [Range(0.5f, 10f)] public float DecimatorToleranceDeg = 2.5f;

        // 2. Weight & Physics
        public bool EnableLandingCushion = true;
        [Range(0.01f, 0.08f)] public float CushionDepth = 0.035f;
        [Range(2, 8)] public int CushionRecoveryFrames = 4;

        public bool EnableContrapposto = true;
        [Range(0.2f, 2.5f)] public float ContrappostoWeight = 1.20f;

        // 3. Overlapping & Drag
        public bool EnableKinematicChainDelay = true;
        [Range(0.2f, 2.0f)] public float DragDelayFrames = 0.85f;

        public bool EnableOvershoot = true;
        [Range(0.1f, 0.8f)] public float OvershootAmount = 0.35f;
        [Range(2, 6)] public int OvershootFrames = 3;

        // 4. Pose & Silhouette
        public bool EnablePoseExaggeration = true;
        [Range(1.0f, 1.5f)] public float ExaggerationScale = 1.18f;

        public bool EnableTrajectoryArcSmoothing = true;
        [Range(3, 9)] public int ArcSmoothWindow = 5;
        [Range(0.2f, 1.0f)] public float ArcBlendWeight = 0.75f;

        /// <summary>
        /// Applies preset parameter values.
        /// </summary>
        public void ApplyPreset(StylizedPolishPreset preset)
        {
            switch (preset)
            {
                case StylizedPolishPreset.SnappyAction:
                    EnableSnapAndEase = true;
                    SnapIntensity = 0.70f;
                    HoldThresholdDegPerSec = 22.0f;
                    StepMode = AnimeStepMode.DynamicAnime;
                    EnableKeyframeDecimator = false;
                    EnableLandingCushion = true;
                    CushionDepth = 0.045f;
                    CushionRecoveryFrames = 3;
                    EnableContrapposto = true;
                    ContrappostoWeight = 1.40f;
                    EnableKinematicChainDelay = true;
                    DragDelayFrames = 1.10f;
                    EnableOvershoot = true;
                    OvershootAmount = 0.50f;
                    OvershootFrames = 3;
                    EnablePoseExaggeration = true;
                    ExaggerationScale = 1.28f;
                    EnableTrajectoryArcSmoothing = true;
                    ArcSmoothWindow = 5;
                    ArcBlendWeight = 0.85f;
                    break;

                case StylizedPolishPreset.RealisticWeight:
                    EnableSnapAndEase = true;
                    SnapIntensity = 0.35f;
                    HoldThresholdDegPerSec = 14.0f;
                    StepMode = AnimeStepMode.Off;
                    EnableKeyframeDecimator = false;
                    EnableLandingCushion = true;
                    CushionDepth = 0.040f;
                    CushionRecoveryFrames = 5;
                    EnableContrapposto = true;
                    ContrappostoWeight = 1.30f;
                    EnableKinematicChainDelay = true;
                    DragDelayFrames = 0.90f;
                    EnableOvershoot = true;
                    OvershootAmount = 0.30f;
                    OvershootFrames = 4;
                    EnablePoseExaggeration = true;
                    ExaggerationScale = 1.12f;
                    EnableTrajectoryArcSmoothing = true;
                    ArcSmoothWindow = 7;
                    ArcBlendWeight = 0.70f;
                    break;

                case StylizedPolishPreset.SubtlePolish:
                    EnableSnapAndEase = true;
                    SnapIntensity = 0.25f;
                    HoldThresholdDegPerSec = 12.0f;
                    StepMode = AnimeStepMode.Off;
                    EnableKeyframeDecimator = false;
                    EnableLandingCushion = true;
                    CushionDepth = 0.020f;
                    CushionRecoveryFrames = 4;
                    EnableContrapposto = true;
                    ContrappostoWeight = 0.80f;
                    EnableKinematicChainDelay = true;
                    DragDelayFrames = 0.50f;
                    EnableOvershoot = true;
                    OvershootAmount = 0.20f;
                    OvershootFrames = 2;
                    EnablePoseExaggeration = true;
                    ExaggerationScale = 1.08f;
                    EnableTrajectoryArcSmoothing = true;
                    ArcSmoothWindow = 5;
                    ArcBlendWeight = 0.50f;
                    break;
            }
        }
    }

    /// <summary>
    /// Implements 9 specialized mathematical and kinematic filters that transform raw, uniform-velocity
    /// monocular mocap into stylized, punchy, hand-keyed character animation.
    /// </summary>
    public static class StylizedMotionPolisher
    {
        private static readonly int[] UpperLimbJoints = new int[]
        {
            (int)SmplxJoint.L_Shoulder, (int)SmplxJoint.L_Elbow, (int)SmplxJoint.L_Wrist,
            (int)SmplxJoint.R_Shoulder, (int)SmplxJoint.R_Elbow, (int)SmplxJoint.R_Wrist
        };

        private static readonly int[] LowerLimbJoints = new int[]
        {
            (int)SmplxJoint.L_Hip, (int)SmplxJoint.L_Knee, (int)SmplxJoint.L_Ankle, (int)SmplxJoint.L_Foot,
            (int)SmplxJoint.R_Hip, (int)SmplxJoint.R_Knee, (int)SmplxJoint.R_Ankle, (int)SmplxJoint.R_Foot
        };

        private static readonly int[] SpineNeckHeadJoints = new int[]
        {
            (int)SmplxJoint.Spine1, (int)SmplxJoint.Spine2, (int)SmplxJoint.Spine3,
            (int)SmplxJoint.Neck, (int)SmplxJoint.Head
        };

        /// <summary>
        /// Applies the complete suite of enabled stylized filters sequentially across the specified frame range.
        /// </summary>
        public static bool PolishMotion(
            EditableMotionData data,
            int startFrame,
            int endFrame,
            StylizedPolishOptions options,
            BodyPartMask mask = BodyPartMask.All)
        {
            if (data == null || data.Frames < 2) return false;
            if (startFrame < 0) startFrame = 0;
            if (endFrame >= data.Frames) endFrame = data.Frames - 1;
            if (startFrame >= endFrame) return false;

            // 1. Pose Exaggeration (Push silhouettes first while raw orientations are clean)
            if (options.EnablePoseExaggeration)
            {
                ApplyPoseExaggeration(data, startFrame, endFrame, options.ExaggerationScale, mask);
            }

            // 2. Contrapposto & Pelvis Tilt (Harmonize weight shift between pelvis and spine)
            if (options.EnableContrapposto)
            {
                ApplyContrapposto(data, startFrame, endFrame, options.ContrappostoWeight);
            }

            // 3. 3D Trajectory Arc Smoothing (Clean spatial path of wrists/ankles via 2-Bone IK)
            if (options.EnableTrajectoryArcSmoothing)
            {
                ApplyTrajectoryArcSmoothing(data, startFrame, endFrame, options.ArcSmoothWindow, options.ArcBlendWeight, mask);
            }

            // 4. Kinematic Chain Delay / Drag (Inject follow-through lag from spine to extremities)
            if (options.EnableKinematicChainDelay)
            {
                ApplyKinematicChainDelay(data, startFrame, endFrame, options.DragDelayFrames, mask);
            }

            // 5. Landing Cushion & Bounce (Add gravity & ground reaction force to pelvis Y)
            if (options.EnableLandingCushion)
            {
                ApplyLandingCushion(data, startFrame, endFrame, options.CushionDepth, options.CushionRecoveryFrames);
            }

            // 6. Overshoot & Settling (Add momentum follow-through on abrupt decelerations)
            if (options.EnableOvershoot)
            {
                ApplyOvershootAndSettling(data, startFrame, endFrame, options.OvershootAmount, options.OvershootFrames, mask);
            }

            // 7. Snap & Ease / Moving Hold (Transform linear velocity into punchy S-curves & holds)
            if (options.EnableSnapAndEase)
            {
                ApplySnapAndEase(data, startFrame, endFrame, options.SnapIntensity, options.HoldThresholdDegPerSec, mask);
            }

            // 8. Keyframe Decimator (Optional curve decimation to extreme keyframes)
            if (options.EnableKeyframeDecimator)
            {
                ApplyKeyframeDecimation(data, startFrame, endFrame, options.DecimatorToleranceDeg, mask);
            }

            // 9. Anime Stepped Interpolation (Optional 2s / 3s cell-look stepping)
            if (options.StepMode != AnimeStepMode.Off)
            {
                ApplyAnimeStepped(data, startFrame, endFrame, options.StepMode);
            }

            return true;
        }

        #region 1. Pose Exaggeration (Dynamic Push)

        /// <summary>
        /// Multiplies angular displacement from neutral pose by a scale factor (>1.0),
        /// pushing limbs outward and deepening poses for greater silhouette clarity.
        /// </summary>
        public static void ApplyPoseExaggeration(
            EditableMotionData data,
            int startFrame,
            int endFrame,
            float scale,
            BodyPartMask mask)
        {
            if (Mathf.Approximately(scale, 1.0f) || scale <= 0f) return;

            int jointCount = SmplxJointDefinitions.JointCount;

            for (int t = startFrame; t <= endFrame; t++)
            {
                for (int j = 1; j < jointCount; j++) // Skip root pelvis translation/rotation
                {
                    if (!BodyPartMaskUtility.IsJointInMask((SmplxJoint)j, mask)) continue;

                    Quaternion rot = data.LocalRotations[t, j];
                    rot.ToAngleAxis(out float angle, out Vector3 axis);

                    if (axis.sqrMagnitude < 0.001f || float.IsNaN(angle) || float.IsInfinity(angle)) continue;

                    if (angle > 180f) angle -= 360f;

                    // Clamped push to prevent unnatural joint hyperextension
                    float effectiveScale = scale;
                    SmplxJoint joint = (SmplxJoint)j;

                    // Moderate head and neck exaggeration to avoid uncanny twists
                    if (joint == SmplxJoint.Neck || joint == SmplxJoint.Head)
                    {
                        effectiveScale = Mathf.Lerp(1.0f, scale, 0.45f);
                    }

                    float scaledAngle = angle * effectiveScale;

                    // Anatomical safety clamps
                    if (joint == SmplxJoint.L_Knee || joint == SmplxJoint.R_Knee)
                    {
                        // Knees only flex backward (positive X); clamp against hyperextension
                        scaledAngle = Mathf.Clamp(scaledAngle, -5f, 150f);
                    }
                    else if (joint == SmplxJoint.L_Elbow || joint == SmplxJoint.R_Elbow)
                    {
                        // Elbows only flex forward/inward; clamp against hyperextension
                        scaledAngle = Mathf.Clamp(scaledAngle, -5f, 155f);
                    }
                    else if (joint == SmplxJoint.Neck || joint == SmplxJoint.Head)
                    {
                        scaledAngle = Mathf.Clamp(scaledAngle, -55f, 55f);
                    }
                    else
                    {
                        scaledAngle = Mathf.Clamp(scaledAngle, -165f, 165f);
                    }

                    data.LocalRotations[t, j] = Quaternion.AngleAxis(scaledAngle, axis.normalized);
                }
            }
        }

        #endregion

        #region 2. Weight & Contrapposto (Pelvis & Spine Harmony)

        /// <summary>
        /// Analyzes ground contact support foot, tilts pelvis toward the load-bearing side,
        /// and counter-rotates the upper torso to enforce natural classical contrapposto posture.
        /// </summary>
        public static void ApplyContrapposto(
            EditableMotionData data,
            int startFrame,
            int endFrame,
            float weight)
        {
            if (weight <= 0.01f) return;

            int lAnkle = (int)SmplxJoint.L_Ankle;
            int rAnkle = (int)SmplxJoint.R_Ankle;
            int pelvis = (int)SmplxJoint.Pelvis;
            int spine1 = (int)SmplxJoint.Spine1;
            int spine2 = (int)SmplxJoint.Spine2;

            for (int t = startFrame; t <= endFrame; t++)
            {
                // Evaluate FK for current frame to find feet heights and ground proximity
                Quaternion[] frameRots = new Quaternion[SmplxJointDefinitions.JointCount];
                for (int j = 0; j < frameRots.Length; j++) frameRots[j] = data.LocalRotations[t, j];

                Vector3[] fkPos = MotionIkUtility.ComputeForwardKinematics(data.RootPositions[t], frameRots);

                float lFootY = fkPos[lAnkle].y;
                float rFootY = fkPos[rAnkle].y;

                // Height delta indicates which leg is supporting body weight
                float heightDiff = rFootY - lFootY; // Positive = Left foot lower (more weighted)
                float tiltAngleDeg = Mathf.Clamp(heightDiff * 35.0f * weight, -6.0f, 6.0f);

                if (Mathf.Abs(tiltAngleDeg) > 0.3f)
                {
                    // Tilt Pelvis lateral roll (Z-axis in local space)
                    Quaternion pelvisTilt = Quaternion.Euler(0f, 0f, tiltAngleDeg);
                    data.LocalRotations[t, pelvis] = data.LocalRotations[t, pelvis] * pelvisTilt;

                    // Counter-tilt lower and middle spine to keep head balanced
                    Quaternion spineCounterTilt = Quaternion.Euler(0f, 0f, -tiltAngleDeg * 0.55f);
                    data.LocalRotations[t, spine1] = data.LocalRotations[t, spine1] * spineCounterTilt;
                    data.LocalRotations[t, spine2] = data.LocalRotations[t, spine2] * spineCounterTilt;
                }
            }
        }

        #endregion

        #region 3. 3D Trajectory Arc Smoothing (Wrists & Ankles)

        /// <summary>
        /// Smooths the 3D world-space trajectory of limb extremities (wrists and ankles)
        /// into graceful circular arcs, eliminating jittery camera-estimation noise via 2-Bone IK.
        /// </summary>
        public static void ApplyTrajectoryArcSmoothing(
            EditableMotionData data,
            int startFrame,
            int endFrame,
            int windowSize,
            float blendWeight,
            BodyPartMask mask)
        {
            if (windowSize < 3 || blendWeight <= 0.01f) return;
            int halfWin = windowSize / 2;

            int jointCount = SmplxJointDefinitions.JointCount;
            int totalFrames = data.Frames;

            // Extract FK trajectories for wrists and ankles
            Vector3[] lWristPath = new Vector3[totalFrames];
            Vector3[] rWristPath = new Vector3[totalFrames];
            Vector3[] lAnklePath = new Vector3[totalFrames];
            Vector3[] rAnklePath = new Vector3[totalFrames];

            Quaternion[] curRots = new Quaternion[jointCount];

            for (int t = 0; t < totalFrames; t++)
            {
                for (int j = 0; j < jointCount; j++) curRots[j] = data.LocalRotations[t, j];
                Vector3[] fk = MotionIkUtility.ComputeForwardKinematics(data.RootPositions[t], curRots);
                lWristPath[t] = fk[(int)SmplxJoint.L_Wrist];
                rWristPath[t] = fk[(int)SmplxJoint.R_Wrist];
                lAnklePath[t] = fk[(int)SmplxJoint.L_Ankle];
                rAnklePath[t] = fk[(int)SmplxJoint.R_Ankle];
            }

            // Smooth trajectories using weighted Gaussian-like kernel
            Vector3[] smoothLW = SmoothVectorPath(lWristPath, startFrame, endFrame, halfWin);
            Vector3[] smoothRW = SmoothVectorPath(rWristPath, startFrame, endFrame, halfWin);
            Vector3[] smoothLA = SmoothVectorPath(lAnklePath, startFrame, endFrame, halfWin);
            Vector3[] smoothRA = SmoothVectorPath(rAnklePath, startFrame, endFrame, halfWin);

            // Apply 2-Bone IK toward smoothed targets
            for (int t = startFrame; t <= endFrame; t++)
            {
                // Left Arm
                if (BodyPartMaskUtility.IsJointInMask(SmplxJoint.L_Wrist, mask))
                {
                    Vector3 target = Vector3.Lerp(lWristPath[t], smoothLW[t], blendWeight);
                    MotionIkUtility.ApplyLimbIK(data, t, SmplxJoint.L_Shoulder, SmplxJoint.L_Elbow, SmplxJoint.L_Wrist, target, null, false);
                }

                // Right Arm
                if (BodyPartMaskUtility.IsJointInMask(SmplxJoint.R_Wrist, mask))
                {
                    Vector3 target = Vector3.Lerp(rWristPath[t], smoothRW[t], blendWeight);
                    MotionIkUtility.ApplyLimbIK(data, t, SmplxJoint.R_Shoulder, SmplxJoint.R_Elbow, SmplxJoint.R_Wrist, target, null, false);
                }

                // Left Leg
                if (BodyPartMaskUtility.IsJointInMask(SmplxJoint.L_Ankle, mask))
                {
                    Vector3 target = Vector3.Lerp(lAnklePath[t], smoothLA[t], blendWeight);
                    MotionIkUtility.ApplyLimbIK(data, t, SmplxJoint.L_Hip, SmplxJoint.L_Knee, SmplxJoint.L_Ankle, target, null, false);
                }

                // Right Leg
                if (BodyPartMaskUtility.IsJointInMask(SmplxJoint.R_Ankle, mask))
                {
                    Vector3 target = Vector3.Lerp(rAnklePath[t], smoothRA[t], blendWeight);
                    MotionIkUtility.ApplyLimbIK(data, t, SmplxJoint.R_Hip, SmplxJoint.R_Knee, SmplxJoint.R_Ankle, target, null, false);
                }
            }
        }

        private static Vector3[] SmoothVectorPath(Vector3[] original, int start, int end, int halfWin)
        {
            Vector3[] result = (Vector3[])original.Clone();
            for (int i = start; i <= end; i++)
            {
                Vector3 sum = Vector3.zero;
                float totalWeight = 0f;
                for (int w = -halfWin; w <= halfWin; w++)
                {
                    int idx = Mathf.Clamp(i + w, 0, original.Length - 1);
                    float weight = 1.0f - (Mathf.Abs(w) / (halfWin + 1f));
                    sum += original[idx] * weight;
                    totalWeight += weight;
                }
                result[i] = sum / Mathf.Max(0.001f, totalWeight);
            }
            return result;
        }

        #endregion

        #region 4. Kinematic Chain Delay & Follow-Through (Drag)

        /// <summary>
        /// Injects organic lag and whiplash follow-through by delaying downstream joint rotations
        /// (head, forearms, wrists) by sub-frame intervals relative to the core spine and pelvis.
        /// </summary>
        public static void ApplyKinematicChainDelay(
            EditableMotionData data,
            int startFrame,
            int endFrame,
            float baseDelayFrames,
            BodyPartMask mask)
        {
            if (baseDelayFrames <= 0.05f) return;

            // Define joint delay hierarchy
            var jointDelays = new Dictionary<SmplxJoint, float>
            {
                { SmplxJoint.Spine2,    baseDelayFrames * 0.30f },
                { SmplxJoint.Spine3,    baseDelayFrames * 0.50f },
                { SmplxJoint.Neck,      baseDelayFrames * 0.70f },
                { SmplxJoint.Head,      baseDelayFrames * 1.00f },
                { SmplxJoint.L_Shoulder,baseDelayFrames * 0.40f },
                { SmplxJoint.R_Shoulder,baseDelayFrames * 0.40f },
                { SmplxJoint.L_Elbow,   baseDelayFrames * 0.90f },
                { SmplxJoint.R_Elbow,   baseDelayFrames * 0.90f },
                { SmplxJoint.L_Wrist,   baseDelayFrames * 1.40f },
                { SmplxJoint.R_Wrist,   baseDelayFrames * 1.40f },
            };

            int totalFrames = data.Frames;

            foreach (var kvp in jointDelays)
            {
                SmplxJoint joint = kvp.Key;
                float delay = kvp.Value;
                int jIdx = (int)joint;

                if (!BodyPartMaskUtility.IsJointInMask(joint, mask)) continue;

                // Cache original rotations for this joint
                Quaternion[] orig = new Quaternion[totalFrames];
                for (int t = 0; t < totalFrames; t++) orig[t] = data.LocalRotations[t, jIdx];

                for (int t = startFrame; t <= endFrame; t++)
                {
                    float delayedT = t - delay;
                    if (delayedT < 0f) delayedT = 0f;

                    int f0 = Mathf.FloorToInt(delayedT);
                    int f1 = Mathf.Min(f0 + 1, totalFrames - 1);
                    float frac = delayedT - f0;

                    Quaternion q0 = orig[f0];
                    Quaternion q1 = orig[f1];
                    if (Quaternion.Dot(q0, q1) < 0f) q1 = new Quaternion(-q1.x, -q1.y, -q1.z, -q1.w);

                    data.LocalRotations[t, jIdx] = Quaternion.Slerp(q0, q1, frac);
                }
            }
        }

        #endregion

        #region 5. Landing Cushion & Pelvis Bounce

        /// <summary>
        /// Detects foot touch-down impact events, applies a responsive downward compression (squash)
        /// to pelvis Y, and settles back via a damped harmonic oscillation.
        /// </summary>
        public static void ApplyLandingCushion(
            EditableMotionData data,
            int startFrame,
            int endFrame,
            float depthMeters,
            int recoveryFrames)
        {
            if (depthMeters <= 0.001f || recoveryFrames < 2) return;

            int totalFrames = data.Frames;
            float[] footMinY = new float[totalFrames];
            Quaternion[] curRots = new Quaternion[SmplxJointDefinitions.JointCount];

            // Measure lowest foot altitude per frame
            for (int t = 0; t < totalFrames; t++)
            {
                for (int j = 0; j < curRots.Length; j++) curRots[j] = data.LocalRotations[t, j];
                Vector3[] fk = MotionIkUtility.ComputeForwardKinematics(data.RootPositions[t], curRots);
                float lY = Mathf.Min(fk[(int)SmplxJoint.L_Ankle].y, fk[(int)SmplxJoint.L_Foot].y);
                float rY = Mathf.Min(fk[(int)SmplxJoint.R_Ankle].y, fk[(int)SmplxJoint.R_Foot].y);
                footMinY[t] = Mathf.Min(lY, rY);
            }

            // Identify impact points where downward velocity arrests or reverses near floor
            float[] cushionOffsets = new float[totalFrames];

            for (int t = Mathf.Max(1, startFrame); t <= Mathf.Min(endFrame - 1, totalFrames - 2); t++)
            {
                float vPrev = footMinY[t] - footMinY[t - 1];
                float vNext = footMinY[t + 1] - footMinY[t];

                // Downward motion stopping abruptly near lowest baseline (floor proximity check)
                if (footMinY[t] < 0.20f && vPrev < -0.006f && vNext >= -0.002f)
                {
                    // Trigger cushion impulse for the next recoveryFrames
                    for (int k = 0; k < recoveryFrames; k++)
                    {
                        int targetT = t + k;
                        if (targetT >= totalFrames) break;

                        float progress = (float)k / recoveryFrames;
                        // Damped sine curve: peak at ~35% progress, decaying to zero
                        float impulse = Mathf.Sin(progress * Mathf.PI) * Mathf.Exp(-2.0f * progress);
                        float currentOffset = -depthMeters * impulse;

                        if (currentOffset < cushionOffsets[targetT])
                        {
                            cushionOffsets[targetT] = currentOffset;
                        }
                    }
                }
            }

            // Apply offsets to RootPositions
            for (int t = startFrame; t <= endFrame; t++)
            {
                if (Mathf.Abs(cushionOffsets[t]) > 0.0005f)
                {
                    data.RootPositions[t] += new Vector3(0f, cushionOffsets[t], 0f);
                }
            }
        }

        #endregion

        #region 6. Overshoot & Settling

        /// <summary>
        /// Adds a dynamic momentum overshoot and decaying settling vibration on rapid decelerations.
        /// </summary>
        public static void ApplyOvershootAndSettling(
            EditableMotionData data,
            int startFrame,
            int endFrame,
            float overshootFactor,
            int settleFrames,
            BodyPartMask mask)
        {
            if (overshootFactor <= 0.01f || settleFrames < 2) return;

            int totalFrames = data.Frames;

            // Target expressive extremity and head joints
            int[] reactiveJoints = new int[]
            {
                (int)SmplxJoint.Head,
                (int)SmplxJoint.L_Wrist, (int)SmplxJoint.R_Wrist,
                (int)SmplxJoint.Spine3
            };

            foreach (int j in reactiveJoints)
            {
                if (!BodyPartMaskUtility.IsJointInMask((SmplxJoint)j, mask)) continue;

                // Compute angular velocities
                float[] angularVel = new float[totalFrames];
                for (int t = 1; t < totalFrames; t++)
                {
                    angularVel[t] = Quaternion.Angle(data.LocalRotations[t - 1, j], data.LocalRotations[t, j]);
                }

                for (int t = Mathf.Max(2, startFrame); t < Mathf.Min(endFrame - settleFrames, totalFrames - settleFrames); t++)
                {
                    float accel = angularVel[t] - angularVel[t - 1];

                    // Strong deceleration event
                    if (angularVel[t - 1] > 6.0f && accel < -4.0f)
                    {
                        Quaternion incomingDelta = data.LocalRotations[t, j] * Quaternion.Inverse(data.LocalRotations[t - 1, j]);
                        incomingDelta.ToAngleAxis(out float dAngle, out Vector3 dAxis);
                        if (dAxis.sqrMagnitude < 0.001f) continue;
                        if (dAngle > 180f) dAngle -= 360f;

                        for (int k = 1; k <= settleFrames; k++)
                        {
                            int targetT = t + k;
                            float progress = (float)k / settleFrames;
                            // Damped oscillation
                            float impulse = Mathf.Sin(progress * Mathf.PI * 1.5f) * Mathf.Exp(-3.0f * progress) * overshootFactor;
                            Quaternion deltaRot = Quaternion.AngleAxis(dAngle * impulse, dAxis);

                            data.LocalRotations[targetT, j] = data.LocalRotations[targetT, j] * deltaRot;
                        }

                        t += settleFrames; // Skip ahead past current settling duration
                    }
                }
            }
        }

        #endregion

        #region 7. Snap & Ease / Moving Hold

        /// <summary>
        /// Accentuates the contrast between moving holds and fast transitions,
        /// reshaping uniform linear mocap spacing into punchy S-curves.
        /// </summary>
        public static void ApplySnapAndEase(
            EditableMotionData data,
            int startFrame,
            int endFrame,
            float snapIntensity,
            float holdThresholdDegPerSec,
            BodyPartMask mask)
        {
            if (snapIntensity <= 0.01f) return;

            int jointCount = SmplxJointDefinitions.JointCount;
            float dt = 1.0f / Mathf.Max(1.0f, data.FrameRate);
            float holdThresholdPerFrame = holdThresholdDegPerSec * dt;

            for (int j = 1; j < jointCount; j++)
            {
                if (!BodyPartMaskUtility.IsJointInMask((SmplxJoint)j, mask)) continue;

                // Identify continuous moving segments between holds
                int segStart = -1;

                for (int t = startFrame; t <= endFrame; t++)
                {
                    float speed = (t > 0) ? Quaternion.Angle(data.LocalRotations[t - 1, j], data.LocalRotations[t, j]) : 0f;
                    bool isMoving = speed > holdThresholdPerFrame;

                    if (isMoving && segStart < 0)
                    {
                        segStart = Mathf.Max(startFrame, t - 1);
                    }
                    else if (!isMoving && segStart >= 0)
                    {
                        int segEnd = t;
                        int segLength = segEnd - segStart;

                        if (segLength >= 3 && segLength <= 14)
                        {
                            // Respace transition with Hermite S-curve
                            Quaternion qStart = data.LocalRotations[segStart, j];
                            Quaternion qEnd = data.LocalRotations[segEnd, j];

                            for (int k = 1; k < segLength; k++)
                            {
                                int curT = segStart + k;
                                float linearU = (float)k / segLength;
                                float easedU = BodyPartMaskUtility.EvaluateEasing(EasingType.SmoothStep, linearU);
                                float blendedU = Mathf.Lerp(linearU, easedU, snapIntensity);

                                data.LocalRotations[curT, j] = Quaternion.Slerp(qStart, qEnd, blendedU);
                            }
                        }

                        segStart = -1;
                    }
                }

                // Flush trailing segment if still open at endFrame
                if (segStart >= 0)
                {
                    int segEnd = endFrame;
                    int segLength = segEnd - segStart;
                    if (segLength >= 3 && segLength <= 14)
                    {
                        Quaternion qStart = data.LocalRotations[segStart, j];
                        Quaternion qEnd = data.LocalRotations[segEnd, j];

                        for (int k = 1; k < segLength; k++)
                        {
                            int curT = segStart + k;
                            float linearU = (float)k / segLength;
                            float easedU = BodyPartMaskUtility.EvaluateEasing(EasingType.SmoothStep, linearU);
                            float blendedU = Mathf.Lerp(linearU, easedU, snapIntensity);

                            data.LocalRotations[curT, j] = Quaternion.Slerp(qStart, qEnd, blendedU);
                        }
                    }
                }
            }
        }

        #endregion

        #region 8. Keyframe Decimator

        /// <summary>
        /// Reduces redundant intermediate keyframes using spherical tolerance decimation,
        /// distilling raw dense mocap into clean hand-keyed keyframes and breakdowns.
        /// </summary>
        public static void ApplyKeyframeDecimation(
            EditableMotionData data,
            int startFrame,
            int endFrame,
            float toleranceDegrees,
            BodyPartMask mask)
        {
            if (toleranceDegrees <= 0.1f) return;

            int jointCount = SmplxJointDefinitions.JointCount;

            for (int j = 1; j < jointCount; j++)
            {
                if (!BodyPartMaskUtility.IsJointInMask((SmplxJoint)j, mask)) continue;

                // Extract rotational segment
                var keptIndices = new HashSet<int> { startFrame, endFrame };
                DecimateRDP(data.LocalRotations, j, startFrame, endFrame, toleranceDegrees, keptIndices);

                // Slerp reconstruct intermediate discarded frames
                var sortedKept = new List<int>(keptIndices);
                sortedKept.Sort();

                for (int i = 0; i < sortedKept.Count - 1; i++)
                {
                    int fA = sortedKept[i];
                    int fB = sortedKept[i + 1];
                    int span = fB - fA;
                    if (span <= 1) continue;

                    Quaternion qA = data.LocalRotations[fA, j];
                    Quaternion qB = data.LocalRotations[fB, j];

                    for (int k = 1; k < span; k++)
                    {
                        float u = (float)k / span;
                        data.LocalRotations[fA + k, j] = Quaternion.Slerp(qA, qB, u);
                    }
                }
            }
        }

        private static void DecimateRDP(
            Quaternion[,] rots,
            int joint,
            int start,
            int end,
            float tolDeg,
            HashSet<int> kept)
        {
            if (end - start <= 1) return;

            Quaternion qStart = rots[start, joint];
            Quaternion qEnd = rots[end, joint];

            float maxError = 0f;
            int maxIdx = start;

            for (int i = start + 1; i < end; i++)
            {
                float u = (float)(i - start) / (end - start);
                Quaternion interp = Quaternion.Slerp(qStart, qEnd, u);
                float err = Quaternion.Angle(rots[i, joint], interp);

                if (err > maxError)
                {
                    maxError = err;
                    maxIdx = i;
                }
            }

            if (maxError > tolDeg)
            {
                kept.Add(maxIdx);
                DecimateRDP(rots, joint, start, maxIdx, tolDeg, kept);
                DecimateRDP(rots, joint, maxIdx, end, tolDeg, kept);
            }
        }

        #endregion

        #region 9. Anime Stepped Interpolation (Limited Animation)

        /// <summary>
        /// Converts continuous floating mocap into crisp stepped intervals (anime 2s or 3s)
        /// for Japanese limited animation aesthetics.
        /// </summary>
        public static void ApplyAnimeStepped(
            EditableMotionData data,
            int startFrame,
            int endFrame,
            AnimeStepMode mode)
        {
            if (mode == AnimeStepMode.Off) return;

            int jointCount = SmplxJointDefinitions.JointCount;

            int holdCounter = 0;
            int currentStepSize = 2; // Default 2s (15fps)

            int anchorFrame = startFrame;

            for (int t = startFrame; t <= endFrame; t++)
            {
                if (mode == AnimeStepMode.Strict2s)
                {
                    currentStepSize = 2;
                }
                else if (mode == AnimeStepMode.Strict3s)
                {
                    currentStepSize = 3;
                }
                else if (mode == AnimeStepMode.DynamicAnime)
                {
                    // Compute instantaneous velocity across extremities
                    float speed = 0f;
                    if (t > 0)
                    {
                        speed = (data.RootPositions[t] - data.RootPositions[t - 1]).magnitude * 30.0f;
                        speed += Quaternion.Angle(data.LocalRotations[t - 1, (int)SmplxJoint.L_Wrist], data.LocalRotations[t, (int)SmplxJoint.L_Wrist]) * 0.05f;
                    }

                    if (speed > 1.8f) currentStepSize = 1;      // Full 1s for fast action
                    else if (speed > 0.4f) currentStepSize = 2; // Standard 2s for normal movement
                    else currentStepSize = 3;                   // 3s for slow idle/tame holds
                }

                if (t == anchorFrame)
                {
                    holdCounter = 1;
                }
                else
                {
                    if (holdCounter < currentStepSize)
                    {
                        // Copy anchor pose to current frame (Stepped hold)
                        data.RootPositions[t] = data.RootPositions[anchorFrame];
                        for (int j = 0; j < jointCount; j++)
                        {
                            data.LocalRotations[t, j] = data.LocalRotations[anchorFrame, j];
                        }
                        holdCounter++;
                    }
                    else
                    {
                        // Step boundary reached, update anchor
                        anchorFrame = t;
                        holdCounter = 1;
                    }
                }
            }
        }

        #endregion
    }
}
