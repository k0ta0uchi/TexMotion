using System;
using System.Collections.Generic;
using UnityEngine;

namespace TexMotion.Runtime.Motion
{
    public enum FaceEmotionType
    {
        AutoDetect, // プロンプトから自動推論
        None,       // 表情を変更しない
        Smile,      // 笑顔 (Joy / Smile)
        Wink,       // ウインク
        Surprise,   // 驚き (Surprise / Open mouth)
        Angry,      // 怒り (Angry)
        Smug        // ドヤ顔・いたずら笑顔
    }

    public static class FaceEmotionHelper
    {
        /// <summary>
        /// Infers emotion from natural language prompt keywords.
        /// </summary>
        public static FaceEmotionType InferEmotionFromPrompt(string prompt)
        {
            if (string.IsNullOrEmpty(prompt)) return FaceEmotionType.None;

            string lower = prompt.ToLowerInvariant();

            if (lower.Contains("wink") || lower.Contains("ウインク")) return FaceEmotionType.Wink;
            if (lower.Contains("angry") || lower.Contains("mad") || lower.Contains("fight") || lower.Contains("punch") || lower.Contains("怒")) return FaceEmotionType.Angry;
            if (lower.Contains("surpris") || lower.Contains("shock") || lower.Contains("scared") || lower.Contains("gasp") || lower.Contains("驚")) return FaceEmotionType.Surprise;
            if (lower.Contains("smug") || lower.Contains("proud") || lower.Contains("doya")) return FaceEmotionType.Smug;
            if (lower.Contains("smile") || lower.Contains("happy") || lower.Contains("laugh") || lower.Contains("wave") || lower.Contains("dance") || lower.Contains("cheer") || lower.Contains("笑") || lower.Contains("嬉")) return FaceEmotionType.Smile;

            return FaceEmotionType.None;
        }

        /// <summary>
        /// Finds the primary face SkinnedMeshRenderer on the avatar.
        /// </summary>
        public static SkinnedMeshRenderer FindFaceRenderer(GameObject avatar)
        {
            if (avatar == null) return null;

            var smrs = avatar.GetComponentsInChildren<SkinnedMeshRenderer>();
            
            // Priority 1: Named "Body", "Face", "Head"
            foreach (var smr in smrs)
            {
                string name = smr.name.ToLowerInvariant();
                if (name.Contains("face") || name.Contains("head") || name == "body")
                {
                    if (smr.sharedMesh != null && smr.sharedMesh.blendShapeCount > 0)
                    {
                        return smr;
                    }
                }
            }

            // Priority 2: Any mesh with blendshapes
            foreach (var smr in smrs)
            {
                if (smr.sharedMesh != null && smr.sharedMesh.blendShapeCount > 0)
                {
                    return smr;
                }
            }

            return null;
        }

        /// <summary>
        /// Resolves blendshape targets and weights (0 to 100) for a given emotion.
        /// </summary>
        public static Dictionary<int, float> GetBlendShapeWeightsForEmotion(SkinnedMeshRenderer faceRenderer, FaceEmotionType emotion, float intensity = 1.0f)
        {
            var result = new Dictionary<int, float>();
            if (faceRenderer == null || faceRenderer.sharedMesh == null || emotion == FaceEmotionType.None)
            {
                return result;
            }

            var mesh = faceRenderer.sharedMesh;
            int count = mesh.blendShapeCount;

            string[] keywords = GetKeywordsForEmotion(emotion);
            float targetWeight = Mathf.Clamp01(intensity) * 100f;

            for (int i = 0; i < count; i++)
            {
                string shapeName = mesh.GetBlendShapeName(i).ToLowerInvariant();

                foreach (var kw in keywords)
                {
                    if (shapeName.Contains(kw))
                    {
                        result[i] = targetWeight;
                        break;
                    }
                }
            }

            return result;
        }

        private static string[] GetKeywordsForEmotion(FaceEmotionType emotion)
        {
            switch (emotion)
            {
                case FaceEmotionType.Smile:
                    // Cover common VRChat/VRM, ARKit and Blender naming schemes.
                    // VRM avatars frequently expose the preset as Fcl_MTH_S or
                    // Fcl_ALL_Joy rather than a literal "smile" key.
                    return new string[]
                    {
                        "smile", "joy", "happy", "笑", "にこり", "喜",
                        "mouth_smile", "mouthsmile", "mouthsmileleft", "mouthsmileright",
                        "lip_corner", "lipcorner", "fcl_mth_s", "fcl_mth_smile", "fcl_all_joy"
                    };

                case FaceEmotionType.Wink:
                    return new string[] { "wink", "ウィンク", "eye_blink_l", "blink_l" };

                case FaceEmotionType.Surprise:
                    return new string[] { "surpris", "びっくり", "驚", "eye_wide", "mouth_open" };

                case FaceEmotionType.Angry:
                    return new string[] { "angry", "怒", "brow_down" };

                case FaceEmotionType.Smug:
                    return new string[] { "smug", "doya", "得意", "にやり" };

                default:
                    return Array.Empty<string>();
            }
        }
    }
}
