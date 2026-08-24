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
                EmotionIntensity = 1.0f
            };
        }
    }

    /// <summary>
    /// Converts generated SMPL-X22 motion data into native Unity Humanoid Muscle AnimationClip assets
    /// for 100% reliable VRChat Action Layer playback and universal avatar compatibility.
    /// </summary>
    public static class AnimationClipBuilder
    {
        public static AnimationClip BuildAnimationClip(GeneratedMotionData motionData, AnimationBuildOptions options)
        {
            if (motionData == null || motionData.Frames <= 0)
            {
                throw new ArgumentException("Invalid motion data.");
            }

            var clip = new AnimationClip
            {
                name = string.IsNullOrEmpty(options.ClipName) ? "TexMotion_Generated" : options.ClipName,
                frameRate = motionData.FrameRate * options.Speed
            };

            float dt = 1.0f / clip.frameRate;
            int frames = motionData.Frames;

            if (options.TargetAvatar != null && options.TargetAvatar.isHuman && options.TargetAvatar.avatar != null)
            {
                BuildHumanoidMuscleCurves(clip, motionData, options, dt, frames);
                BuildFaceCurves(clip, options, dt, frames);
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

            try
            {
                var cloneAnim = tempClone.GetComponent<Animator>();
                cloneAnim.enabled = false;

                // Cache bone transforms & initial rotations
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

                Transform hips = cloneAnim.GetBoneTransform(HumanBodyBones.Hips);
                Vector3 initialHipsPos = hips != null ? hips.localPosition : Vector3.zero;

                var poseHandler = new HumanPoseHandler(cloneAnim.avatar, tempClone.transform);
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

                Vector3 firstFramePos = motionData.RootPositions.Length > 0 ? motionData.RootPositions[0] : Vector3.zero;

                for (int t = 0; t < frames; t++)
                {
                    float time = t * dt;

                    // 1. Apply Body Joint Rotations
                    foreach (var kvp in boneMap)
                    {
                        SmplxJoint joint = kvp.Key;
                        Transform bone = kvp.Value;
                        Quaternion smplRot = motionData.LocalRotations[t, (int)joint];
                        Quaternion rest = initialRotations[joint];
                        bone.localRotation = ConvertSmplRotationToUnity(smplRot, joint, rest);
                    }

                    // 2. Apply Finger Poses
                    if (options.HandPose != HandPoseType.KeepFree)
                    {
                        ApplyFingerPoseToClone(cloneAnim, HandPosePresets.LeftFingerBones, options.HandPose, true);
                        ApplyFingerPoseToClone(cloneAnim, HandPosePresets.RightFingerBones, options.HandPose, false);
                    }

                    // 3. Apply Root Position
                    if (hips != null && motionData.RootPositions.Length > 0)
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

                    // Record Root Motion curves
                    rootPosX.AddKey(time, humanPose.bodyPosition.x);
                    rootPosY.AddKey(time, humanPose.bodyPosition.y);
                    rootPosZ.AddKey(time, humanPose.bodyPosition.z);

                    rootRotX.AddKey(time, humanPose.bodyRotation.x);
                    rootRotY.AddKey(time, humanPose.bodyRotation.y);
                    rootRotZ.AddKey(time, humanPose.bodyRotation.z);
                    rootRotW.AddKey(time, humanPose.bodyRotation.w);

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
                UnityEngine.Object.DestroyImmediate(tempClone);
            }
#endif
        }

        private static void ApplyFingerPoseToClone(Animator animator, HumanBodyBones[] fingerBones, HandPoseType pose, bool isLeft)
        {
            foreach (var boneType in fingerBones)
            {
                Transform bone = animator.GetBoneTransform(boneType);
                if (bone == null) continue;

                Quaternion rest = bone.localRotation;
                Quaternion offset = HandPosePresets.GetFingerLocalRotation(pose, boneType, isLeft);
                bone.localRotation = rest * offset;
            }
        }

        private static void BuildFaceCurves(AnimationClip clip, AnimationBuildOptions options, float dt, int frames)
        {
#if UNITY_EDITOR
            if (options.FaceEmotion == FaceEmotionType.None) return;

            var faceRenderer = FaceEmotionHelper.FindFaceRenderer(options.TargetAvatar.gameObject);
            if (faceRenderer == null) return;

            string facePath = AnimationUtility.CalculateTransformPath(faceRenderer.transform, options.TargetAvatar.transform);
            var weights = FaceEmotionHelper.GetBlendShapeWeightsForEmotion(faceRenderer, options.FaceEmotion, options.EmotionIntensity);

            float totalDuration = (frames - 1) * dt;

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

        private static void BuildGenericCurves(AnimationClip clip, GeneratedMotionData motionData, AnimationBuildOptions options, float dt, int frames)
        {
#if UNITY_EDITOR
            for (int j = 0; j < SmplxJointDefinitions.JointCount; j++)
            {
                string jointName = SmplxJointDefinitions.JointNames[j];
                string path = j == 0 ? jointName : $"Pelvis/{jointName}";

                var rotXCurve = new AnimationCurve();
                var rotYCurve = new AnimationCurve();
                var rotZCurve = new AnimationCurve();
                var rotWCurve = new AnimationCurve();

                for (int t = 0; t < frames; t++)
                {
                    float time = t * dt;
                    Quaternion q = motionData.LocalRotations[t, j];
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

        private static Quaternion ConvertSmplRotationToUnity(Quaternion smplRot, SmplxJoint joint, Quaternion restPoseRot)
        {
            if (smplRot.x == 0 && smplRot.y == 0 && smplRot.z == 0 && smplRot.w == 0)
            {
                return restPoseRot;
            }

            Quaternion converted = new Quaternion(smplRot.x, -smplRot.y, -smplRot.z, smplRot.w);
            return restPoseRot * converted;
        }
    }
}
