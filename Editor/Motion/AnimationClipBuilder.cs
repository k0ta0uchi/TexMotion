using System;
using System.Collections.Generic;
using TexMotion.Runtime.Motion;
using TexMotion.Runtime.Native;
using UnityEditor;
using UnityEngine;

namespace TexMotion.Editor.Motion
{
    public struct AnimationBuildOptions
    {
        public string ClipName;
        public bool IsLoop;
        public bool InPlace; // Fix XZ root movement
        public float Speed;
        public Animator TargetAvatar;
        public HandPoseType HandPose;
        public FaceEmotionType FaceEmotion;
        public float EmotionIntensity;
        public float MinConfidenceThreshold; // Optional confidence filter for VideoMotionData

        public static AnimationBuildOptions CreateDefault(
            string clipName = "TexMotion_Anim",
            bool isLoop = false,
            bool inPlace = true,
            HandPoseType handPose = HandPoseType.NaturalRelaxed,
            FaceEmotionType faceEmotion = FaceEmotionType.AutoDetect)
        {
            return new AnimationBuildOptions
            {
                ClipName = clipName,
                IsLoop = isLoop,
                InPlace = inPlace,
                Speed = 1.0f,
                TargetAvatar = null,
                HandPose = handPose,
                FaceEmotion = faceEmotion,
                EmotionIntensity = 1.0f,
                MinConfidenceThreshold = 0f
            };
        }
    }

    /// <summary>
    /// Converts generated SMPL-X22 motion data or video-extracted motion data into native Unity Humanoid Muscle AnimationClip assets
    /// for 100% reliable VRChat Action Layer playback and universal avatar compatibility.
    /// </summary>
    public static class AnimationClipBuilder
    {
        /// <summary>
        /// Builds an AnimationClip directly from video-extracted motion data.
        /// </summary>
        public static AnimationClip BuildAnimationClip(VideoMotionData videoMotionData, AnimationBuildOptions options)
        {
            return BuildAnimationClip((GeneratedMotionData)videoMotionData, options);
        }

        /// <summary>
        /// Builds an AnimationClip from Kimodo text diffusion or video motion data.
        /// </summary>
        public static AnimationClip BuildAnimationClip(GeneratedMotionData motionData, AnimationBuildOptions options)
        {
            if (motionData == null || motionData.Frames <= 0)
            {
                throw new ArgumentException("Invalid motion data.");
            }

            int frames = motionData.Frames;
            int jointCount = motionData.JointCount > 0 ? motionData.JointCount : SmplxJointDefinitions.JointCount;

            // Enforce quaternion hemisphere continuity on input rotations
            EnforceHemisphereContinuity(motionData.LocalRotations, frames, jointCount);

            float effectiveSpeed = options.Speed > 0f ? options.Speed : 1.0f;
            float baseFrameRate = motionData.FrameRate > 0f ? motionData.FrameRate : 30.0f;

            var clip = new AnimationClip
            {
                name = string.IsNullOrEmpty(options.ClipName) ? "TexMotion_Generated" : options.ClipName,
                frameRate = baseFrameRate * effectiveSpeed
            };

            float dt = 1.0f / clip.frameRate;

            if (options.TargetAvatar != null && options.TargetAvatar.isHuman && options.TargetAvatar.avatar != null)
            {
                BuildHumanoidMuscleCurves(clip, motionData, options, dt, frames);
                BuildFaceCurves(clip, motionData, options, dt, frames);
            }
            else
            {
                BuildGenericCurves(clip, motionData, options, dt, frames);
            }

#if UNITY_EDITOR
            var clipSettings = AnimationUtility.GetAnimationClipSettings(clip);
            clipSettings.loopTime = options.IsLoop;
            clipSettings.loopBlend = options.IsLoop;
            clipSettings.loopBlendOrientation = options.IsLoop;
            clipSettings.loopBlendPositionY = options.IsLoop;
            clipSettings.loopBlendPositionXZ = options.IsLoop;
            clipSettings.keepOriginalOrientation = true;
            clipSettings.keepOriginalPositionY = true;
            clipSettings.keepOriginalPositionXZ = true;
            AnimationUtility.SetAnimationClipSettings(clip, clipSettings);
#endif

            return clip;
        }

        private static void BuildHumanoidMuscleCurves(
            AnimationClip clip,
            GeneratedMotionData motionData,
            AnimationBuildOptions options,
            float dt,
            int frames)
        {
#if UNITY_EDITOR
            Animator animator = options.TargetAvatar;
            GameObject tempClone = UnityEngine.Object.Instantiate(animator.gameObject, Vector3.zero, Quaternion.identity);
            tempClone.hideFlags = HideFlags.HideAndDontSave;

            HumanPoseHandler poseHandler = null;
            try
            {
                var cloneAnim = tempClone.GetComponent<Animator>();
                cloneAnim.enabled = false;

                // 1. Cache body bone transforms & initial rotations
                var boneMap = new Dictionary<SmplxJoint, Transform>();
                var initialRotations = new Dictionary<SmplxJoint, Quaternion>();

                foreach (var kvp in SmplxJointDefinitions.SmplxToHumanBodyBones)
                {
                    Transform bone = cloneAnim.GetBoneTransform(kvp.Value);
                    if (bone != null)
                    {
                        boneMap[kvp.Key] = bone;
                        initialRotations[kvp.Key] = bone.localRotation;
                    }
                }

                // 2. Cache finger bone transforms & initial rest rotations to PREVENT compounding accumulation
                var fingerBoneMap = new Dictionary<HumanBodyBones, Transform>();
                var initialFingerRotations = new Dictionary<HumanBodyBones, Quaternion>();

                CacheFingerBones(cloneAnim, HandPosePresets.LeftFingerBones, fingerBoneMap, initialFingerRotations);
                CacheFingerBones(cloneAnim, HandPosePresets.RightFingerBones, fingerBoneMap, initialFingerRotations);

                Transform hips = cloneAnim.GetBoneTransform(HumanBodyBones.Hips);
                Vector3 initialHipsPos = hips != null ? hips.localPosition : Vector3.zero;

                poseHandler = new HumanPoseHandler(cloneAnim.avatar, tempClone.transform);
                var humanPose = new HumanPose();

                int muscleCount = HumanTrait.MuscleCount;
                var muscleCurves = new AnimationCurve[muscleCount];
                for (int m = 0; m < muscleCount; m++) muscleCurves[m] = new AnimationCurve();

                var rootPosX = new AnimationCurve();
                var rootPosY = new AnimationCurve();
                var rootPosZ = new AnimationCurve();
                var rootRotX = new AnimationCurve();
                var rootRotY = new AnimationCurve();
                var rootRotZ = new AnimationCurve();
                var rootRotW = new AnimationCurve();

                Vector3 firstFramePos = (motionData.RootPositions != null && motionData.RootPositions.Length > 0)
                    ? motionData.RootPositions[0]
                    : Vector3.zero;

                // Track previous rotations for quaternion hemisphere continuity
                var prevBoneRotations = new Dictionary<SmplxJoint, Quaternion>();
                Quaternion prevBodyRot = Quaternion.identity;
                bool hasPrevBodyRot = false;

                VideoMotionData videoData = motionData as VideoMotionData;
                float effectiveSpeed = options.Speed > 0f ? options.Speed : 1.0f;

                for (int t = 0; t < frames; t++)
                {
                    // Confidence filtering for video-extracted motion
                    if (videoData != null && options.MinConfidenceThreshold > 0f)
                    {
                        // Always keep first and last frame for curve duration boundaries
                        if (t > 0 && t < frames - 1 && videoData.GetFrameConfidence(t) < options.MinConfidenceThreshold)
                        {
                            continue; // Skip low-confidence frame; curve interpolation smoothly bridges the gap
                        }
                    }

                    float time = GetFrameTime(motionData, t, dt, effectiveSpeed);

                    // 1. Apply Body Joint Rotations with Quaternion Hemisphere Continuity
                    foreach (var kvp in boneMap)
                    {
                        SmplxJoint joint = kvp.Key;
                        Transform bone = kvp.Value;
                        Quaternion smplRot = motionData.LocalRotations[t, (int)joint];
                        Quaternion rest = initialRotations[joint];
                        Quaternion targetRot = ConvertSmplRotationToUnity(smplRot, joint, rest);

                        if (prevBoneRotations.TryGetValue(joint, out Quaternion prevRot))
                        {
                            if (Quaternion.Dot(prevRot, targetRot) < 0f)
                            {
                                targetRot = new Quaternion(-targetRot.x, -targetRot.y, -targetRot.z, -targetRot.w);
                            }
                        }
                        prevBoneRotations[joint] = targetRot;
                        bone.localRotation = targetRot;
                    }

                    // 2. Apply Finger Poses (always relative to CACHED initial rest rotations, preventing accumulation)
                    if (options.HandPose != HandPoseType.KeepFree)
                    {
                        ApplyFingerPoseToClone(fingerBoneMap, initialFingerRotations, HandPosePresets.LeftFingerBones, options.HandPose, true);
                        ApplyFingerPoseToClone(fingerBoneMap, initialFingerRotations, HandPosePresets.RightFingerBones, options.HandPose, false);
                    }
                    else
                    {
                        ResetFingerPoses(fingerBoneMap, initialFingerRotations);
                    }

                    // 3. Apply Root Position
                    if (hips != null && motionData.RootPositions != null && motionData.RootPositions.Length > t)
                    {
                        Vector3 rawPos = motionData.RootPositions[t];
                        Vector3 delta = rawPos - firstFramePos;
                        if (options.InPlace)
                        {
                            hips.localPosition = new Vector3(initialHipsPos.x, initialHipsPos.y + delta.y, initialHipsPos.z);
                        }
                        else
                        {
                            hips.localPosition = initialHipsPos + delta;
                        }
                    }

                    // 4. Sample HumanPose
                    poseHandler.GetHumanPose(ref humanPose);

                    // Ensure root rotation quaternion hemisphere continuity
                    Quaternion bodyRot = humanPose.bodyRotation;
                    if (hasPrevBodyRot)
                    {
                        if (Quaternion.Dot(prevBodyRot, bodyRot) < 0f)
                        {
                            bodyRot = new Quaternion(-bodyRot.x, -bodyRot.y, -bodyRot.z, -bodyRot.w);
                        }
                    }
                    prevBodyRot = bodyRot;
                    hasPrevBodyRot = true;

                    // Record Root Motion curves
                    rootPosX.AddKey(time, humanPose.bodyPosition.x);
                    rootPosY.AddKey(time, humanPose.bodyPosition.y);
                    rootPosZ.AddKey(time, humanPose.bodyPosition.z);

                    rootRotX.AddKey(time, bodyRot.x);
                    rootRotY.AddKey(time, bodyRot.y);
                    rootRotZ.AddKey(time, bodyRot.z);
                    rootRotW.AddKey(time, bodyRot.w);

                    // Record 95 Muscle Curves
                    for (int m = 0; m < muscleCount; m++)
                    {
                        muscleCurves[m].AddKey(time, humanPose.muscles[m]);
                    }
                }

                // Bind Muscle Curves to AnimationClip
                for (int m = 0; m < muscleCount; m++)
                {
                    string muscleName = HumanTrait.MuscleName[m];
                    clip.SetCurve("", typeof(Animator), muscleName, muscleCurves[m]);
                }

                // Bind Root Motion Curves
                clip.SetCurve("", typeof(Animator), "RootT.x", rootPosX);
                clip.SetCurve("", typeof(Animator), "RootT.y", rootPosY);
                clip.SetCurve("", typeof(Animator), "RootT.z", rootPosZ);
                clip.SetCurve("", typeof(Animator), "RootQ.x", rootRotX);
                clip.SetCurve("", typeof(Animator), "RootQ.y", rootRotY);
                clip.SetCurve("", typeof(Animator), "RootQ.z", rootRotZ);
                clip.SetCurve("", typeof(Animator), "RootQ.w", rootRotW);
            }
            finally
            {
                // Ensure proper disposal of HumanPoseHandler native memory
                poseHandler?.Dispose();

                if (tempClone != null)
                {
                    UnityEngine.Object.DestroyImmediate(tempClone);
                }
            }
#endif
        }

        private static void CacheFingerBones(
            Animator animator,
            HumanBodyBones[] fingerBones,
            Dictionary<HumanBodyBones, Transform> boneMap,
            Dictionary<HumanBodyBones, Quaternion> initialRotations)
        {
            foreach (var boneType in fingerBones)
            {
                if (boneMap.ContainsKey(boneType)) continue;
                Transform bone = animator.GetBoneTransform(boneType);
                if (bone != null)
                {
                    boneMap[boneType] = bone;
                    initialRotations[boneType] = bone.localRotation;
                }
            }
        }

        private static void ApplyFingerPoseToClone(
            Dictionary<HumanBodyBones, Transform> boneMap,
            Dictionary<HumanBodyBones, Quaternion> initialRotations,
            HumanBodyBones[] fingerBones,
            HandPoseType pose,
            bool isLeft)
        {
            foreach (var boneType in fingerBones)
            {
                if (boneMap.TryGetValue(boneType, out Transform bone) && bone != null)
                {
                    if (initialRotations.TryGetValue(boneType, out Quaternion rest))
                    {
                        Quaternion offset = HandPosePresets.GetFingerLocalRotation(pose, boneType, isLeft);
                        bone.localRotation = rest * offset;
                    }
                }
            }
        }

        private static void ResetFingerPoses(
            Dictionary<HumanBodyBones, Transform> boneMap,
            Dictionary<HumanBodyBones, Quaternion> initialRotations)
        {
            foreach (var kvp in boneMap)
            {
                if (kvp.Value != null && initialRotations.TryGetValue(kvp.Key, out Quaternion rest))
                {
                    kvp.Value.localRotation = rest;
                }
            }
        }

        private static void BuildFaceCurves(
            AnimationClip clip,
            GeneratedMotionData motionData,
            AnimationBuildOptions options,
            float dt,
            int frames)
        {
#if UNITY_EDITOR
            if (options.FaceEmotion == FaceEmotionType.None) return;

            var faceRenderer = FaceEmotionHelper.FindFaceRenderer(options.TargetAvatar.gameObject);
            if (faceRenderer == null) return;

            string facePath = AnimationUtility.CalculateTransformPath(faceRenderer.transform, options.TargetAvatar.transform);
            var weights = FaceEmotionHelper.GetBlendShapeWeightsForEmotion(faceRenderer, options.FaceEmotion, options.EmotionIntensity);

            float effectiveSpeed = options.Speed > 0f ? options.Speed : 1.0f;
            float totalDuration = GetFrameTime(motionData, frames - 1, dt, effectiveSpeed);

            foreach (var kvp in weights)
            {
                int shapeIndex = kvp.Key;
                float targetWeight = kvp.Value;
                string shapeName = faceRenderer.sharedMesh.GetBlendShapeName(shapeIndex);
                string propertyName = $"blendShape.{shapeName}";

                var curve = new AnimationCurve();
                curve.AddKey(0f, targetWeight);
                curve.AddKey(totalDuration, targetWeight);

                clip.SetCurve(facePath, typeof(SkinnedMeshRenderer), propertyName, curve);
            }
#endif
        }

        private static void BuildGenericCurves(
            AnimationClip clip,
            GeneratedMotionData motionData,
            AnimationBuildOptions options,
            float dt,
            int frames)
        {
#if UNITY_EDITOR
            float effectiveSpeed = options.Speed > 0f ? options.Speed : 1.0f;
            VideoMotionData videoData = motionData as VideoMotionData;

            for (int j = 0; j < SmplxJointDefinitions.JointCount; j++)
            {
                string jointName = SmplxJointDefinitions.JointNames[j];
                string path = j == 0 ? jointName : $"Pelvis/{jointName}";

                var rotXCurve = new AnimationCurve();
                var rotYCurve = new AnimationCurve();
                var rotZCurve = new AnimationCurve();
                var rotWCurve = new AnimationCurve();

                Quaternion prevQ = Quaternion.identity;
                bool hasPrevQ = false;

                for (int t = 0; t < frames; t++)
                {
                    if (videoData != null && options.MinConfidenceThreshold > 0f)
                    {
                        if (t > 0 && t < frames - 1 && videoData.GetFrameConfidence(t) < options.MinConfidenceThreshold)
                        {
                            continue;
                        }
                    }

                    float time = GetFrameTime(motionData, t, dt, effectiveSpeed);
                    Quaternion q = motionData.LocalRotations[t, j];

                    if (hasPrevQ)
                    {
                        if (Quaternion.Dot(prevQ, q) < 0f)
                        {
                            q = new Quaternion(-q.x, -q.y, -q.z, -q.w);
                        }
                    }
                    prevQ = q;
                    hasPrevQ = true;

                    rotXCurve.AddKey(time, q.x);
                    rotYCurve.AddKey(time, q.y);
                    rotZCurve.AddKey(time, q.z);
                    rotWCurve.AddKey(time, q.w);
                }

                clip.SetCurve(path, typeof(Transform), "m_LocalRotation.x", rotXCurve);
                clip.SetCurve(path, typeof(Transform), "m_LocalRotation.y", rotYCurve);
                clip.SetCurve(path, typeof(Transform), "m_LocalRotation.z", rotZCurve);
                clip.SetCurve(path, typeof(Transform), "m_LocalRotation.w", rotWCurve);
            }
#endif
        }

        private static float GetFrameTime(GeneratedMotionData motionData, int frameIndex, float defaultDt, float speed)
        {
            float time = motionData.GetTimestamp(frameIndex);
            if (speed > 0f && Math.Abs(speed - 1.0f) > 0.0001f)
            {
                time /= speed;
            }
            return time;
        }

        /// <summary>
        /// Enforces quaternion hemisphere continuity (dot(q[t], q[t-1]) >= 0) across all joints and frames.
        /// </summary>
        public static void EnforceHemisphereContinuity(Quaternion[,] rotations, int frames, int jointCount)
        {
            if (rotations == null || frames <= 1 || jointCount <= 0) return;

            for (int j = 0; j < jointCount; j++)
            {
                for (int t = 1; t < frames; t++)
                {
                    Quaternion prev = rotations[t - 1, j];
                    Quaternion curr = rotations[t, j];

                    if (prev.x == 0 && prev.y == 0 && prev.z == 0 && prev.w == 0) continue;
                    if (curr.x == 0 && curr.y == 0 && curr.z == 0 && curr.w == 0) continue;

                    if (Quaternion.Dot(prev, curr) < 0f)
                    {
                        rotations[t, j] = new Quaternion(-curr.x, -curr.y, -curr.z, -curr.w);
                    }
                }
            }
        }

        public static Quaternion ConvertSmplRotationToUnity(Quaternion smplRot, SmplxJoint joint, Quaternion restPoseRot)
        {
            if (smplRot.x == 0 && smplRot.y == 0 && smplRot.z == 0 && smplRot.w == 0)
            {
                return restPoseRot;
            }

            Quaternion converted = new Quaternion(smplRot.x, -smplRot.y, -smplRot.z, smplRot.w);
            return restPoseRot * converted;
        }

        public static Quaternion ConvertUnityRotationToSmpl(Quaternion unityRot, SmplxJoint joint, Quaternion restPoseRot)
        {
            Quaternion converted = Quaternion.Inverse(restPoseRot) * unityRot;
            return new Quaternion(converted.x, -converted.y, -converted.z, converted.w);
        }
    }
}
