using System;
using System.Collections.Generic;
using UnityEngine;

namespace TexMotion.Runtime.Motion
{
    /// <summary>
    /// Configuration for an individual blendshape mapping entry between MediaPipe 52 blendshapes
    /// and avatar-specific blendshapes.
    /// </summary>
    [Serializable]
    public class FaceBlendShapeMapping
    {
        [Tooltip("Source blend shape name from MediaPipe 52 (e.g. eyeBlinkLeft, jawOpen).")]
        public string SourceShapeName;

        [Tooltip("Target blend shape name on the avatar SkinnedMeshRenderer (e.g. vrc.blink_l, Fcl_MTH_Fun).")]
        public string TargetShapeName;

        [Tooltip("Gain multiplier applied to the normalized source weight (default 1.0).")]
        [Range(0f, 5f)]
        public float Multiplier = 1.0f;

        [Tooltip("Deadzone threshold [0.0, 1.0]. Weights below this value are suppressed to 0.")]
        [Range(0f, 0.5f)]
        public float DeadZone = 0.0f;

        [Tooltip("If true, swaps left and right source data.")]
        public bool InvertLeftRight = false;

        [Tooltip("Whether this blendshape mapping is active.")]
        public bool Enabled = true;

        public FaceBlendShapeMapping() { }

        public FaceBlendShapeMapping(string source, string target, float multiplier = 1.0f, float deadZone = 0.0f, bool invertLeftRight = false)
        {
            SourceShapeName = source;
            TargetShapeName = target;
            Multiplier = multiplier;
            DeadZone = deadZone;
            InvertLeftRight = invertLeftRight;
            Enabled = true;
        }
    }

    /// <summary>
    /// ScriptableObject holding blendshape retargeting definitions between MediaPipe 52 Face Landmarker
    /// outputs and avatar mesh blendshapes (ARKit / Perfect Sync, VRChat standard, VRoid/VRM, MMD).
    /// </summary>
    [CreateAssetMenu(fileName = "NewFaceMappingProfile", menuName = "TexMotion/Face Mapping Profile", order = 200)]
    public class FaceMappingProfile : ScriptableObject
    {
        [Header("Target Mesh Configuration")]
        [Tooltip("Root-relative path to the facial SkinnedMeshRenderer (e.g. 'Body' or 'Face'). If empty, auto-detected from avatar.")]
        public string TargetMeshPath = "";

        [Header("Global Adjustments")]
        [Tooltip("Global multiplier applied to all mapped blendshapes.")]
        [Range(0f, 5f)]
        public float GlobalMultiplier = 1.0f;

        [Tooltip("Whether to apply head rotation extracted from video face tracking to the avatar's head bone.")]
        public bool ApplyHeadRotation = true;

        [Tooltip("Blend weight for head rotation [0.0, 1.0].")]
        [Range(0f, 1f)]
        public float HeadRotationWeight = 1.0f;

        [Header("BlendShape Mappings")]
        [SerializeField]
        public List<FaceBlendShapeMapping> Mappings = new List<FaceBlendShapeMapping>();

        /// <summary>
        /// Standard 52 MediaPipe Face Landmarker / ARKit blendshape names.
        /// </summary>
        public static readonly string[] MediaPipe52BlendShapes = new string[]
        {
            // Brows
            "browDownLeft", "browDownRight", "browInnerUp", "browOuterUpLeft", "browOuterUpRight",
            // Eyes
            "eyeBlinkLeft", "eyeBlinkRight", "eyeLookDownLeft", "eyeLookDownRight",
            "eyeLookInLeft", "eyeLookInRight", "eyeLookOutLeft", "eyeLookOutRight",
            "eyeLookUpLeft", "eyeLookUpRight", "eyeSquintLeft", "eyeSquintRight",
            "eyeWideLeft", "eyeWideRight",
            // Cheeks & Nose
            "cheekPuff", "cheekSquintLeft", "cheekSquintRight",
            "noseSneerLeft", "noseSneerRight",
            // Jaw
            "jawForward", "jawLeft", "jawRight", "jawOpen",
            // Mouth
            "mouthClose", "mouthFunnel", "mouthPucker", "mouthLeft", "mouthRight",
            "mouthSmileLeft", "mouthSmileRight", "mouthFrownLeft", "mouthFrownRight",
            "mouthDimpleLeft", "mouthDimpleRight", "mouthStretchLeft", "mouthStretchRight",
            "mouthRollLower", "mouthRollUpper", "mouthShrugLower", "mouthShrugUpper",
            "mouthPressLeft", "mouthPressRight", "mouthLowerDownLeft", "mouthLowerDownRight",
            "mouthUpperUpLeft", "mouthUpperUpRight",
            // Tongue
            "tongueOut"
        };

        /// <summary>
        /// Evaluates the final Unity BlendShape weight (0.0 to 100.0) from a raw normalized weight (0.0 to 1.0).
        /// </summary>
        public float EvaluateWeight(FaceBlendShapeMapping mapping, float rawWeight01)
        {
            if (mapping == null || !mapping.Enabled) return 0f;
            if (rawWeight01 <= mapping.DeadZone) return 0f;

            float range = 1.0f - mapping.DeadZone;
            float normalized = range > 0.0001f ? (rawWeight01 - mapping.DeadZone) / range : 0f;
            float finalWeight = normalized * mapping.Multiplier * GlobalMultiplier * 100.0f;
            return Mathf.Clamp(finalWeight, 0f, 150f);
        }

        /// <summary>
        /// Gets the effective source shape name considering the InvertLeftRight setting.
        /// </summary>
        public string GetEffectiveSourceShape(FaceBlendShapeMapping mapping)
        {
            if (mapping == null) return string.Empty;
            if (!mapping.InvertLeftRight) return mapping.SourceShapeName;
            return GetOppositeShapeName(mapping.SourceShapeName);
        }

        /// <summary>
        /// Finds an existing mapping for the specified source shape name.
        /// </summary>
        public FaceBlendShapeMapping FindMapping(string sourceShapeName)
        {
            if (Mappings == null || string.IsNullOrEmpty(sourceShapeName)) return null;
            return Mappings.Find(m => string.Equals(m.SourceShapeName, sourceShapeName, StringComparison.OrdinalIgnoreCase));
        }

        /// <summary>
        /// Returns the opposite left/right shape name if applicable, or the original name if symmetric.
        /// </summary>
        public static string GetOppositeShapeName(string shapeName)
        {
            if (string.IsNullOrEmpty(shapeName)) return string.Empty;

            if (shapeName.EndsWith("Left", StringComparison.OrdinalIgnoreCase))
            {
                return shapeName.Substring(0, shapeName.Length - 4) + (char.IsUpper(shapeName[shapeName.Length - 4]) ? "Right" : "right");
            }
            if (shapeName.EndsWith("Right", StringComparison.OrdinalIgnoreCase))
            {
                return shapeName.Substring(0, shapeName.Length - 5) + (char.IsUpper(shapeName[shapeName.Length - 5]) ? "Left" : "left");
            }
            if (shapeName.EndsWith("_L", StringComparison.OrdinalIgnoreCase))
            {
                return shapeName.Substring(0, shapeName.Length - 2) + "_R";
            }
            if (shapeName.EndsWith("_R", StringComparison.OrdinalIgnoreCase))
            {
                return shapeName.Substring(0, shapeName.Length - 2) + "_L";
            }
            if (shapeName.EndsWith(".l", StringComparison.OrdinalIgnoreCase))
            {
                return shapeName.Substring(0, shapeName.Length - 2) + ".r";
            }
            if (shapeName.EndsWith(".r", StringComparison.OrdinalIgnoreCase))
            {
                return shapeName.Substring(0, shapeName.Length - 2) + ".l";
            }

            return shapeName;
        }

        /// <summary>
        /// Automatically generates or updates mappings by scanning the blendshapes on the provided SkinnedMeshRenderer.
        /// Supports ARKit 52 (Perfect Sync), VRChat standards, VRoid/VRM (Fcl_*), and MMD naming conventions.
        /// </summary>
        /// <param name="renderer">Target avatar SkinnedMeshRenderer</param>
        /// <returns>Number of successfully mapped blendshapes</returns>
        public int AutoGenerateMapping(SkinnedMeshRenderer renderer)
        {
            if (renderer == null || renderer.sharedMesh == null)
            {
                return 0;
            }

            var mesh = renderer.sharedMesh;
            int shapeCount = mesh.blendShapeCount;
            if (shapeCount == 0) return 0;

            var avatarShapes = new List<string>(shapeCount);
            for (int i = 0; i < shapeCount; i++)
            {
                avatarShapes.Add(mesh.GetBlendShapeName(i));
            }

            if (Mappings == null)
            {
                Mappings = new List<FaceBlendShapeMapping>();
            }

            int mappedCount = 0;

            foreach (var mpShape in MediaPipe52BlendShapes)
            {
                string bestTarget = FindBestMatchForShape(mpShape, avatarShapes);
                if (!string.IsNullOrEmpty(bestTarget))
                {
                    var existing = FindMapping(mpShape);
                    if (existing != null)
                    {
                        existing.TargetShapeName = bestTarget;
                    }
                    else
                    {
                        Mappings.Add(new FaceBlendShapeMapping(mpShape, bestTarget));
                    }
                    mappedCount++;
                }
            }

            return mappedCount;
        }

        /// <summary>
        /// Resets the mapping list to standard 1:1 ARKit 52 naming.
        /// </summary>
        public void ResetToStandard52()
        {
            if (Mappings == null) Mappings = new List<FaceBlendShapeMapping>();
            Mappings.Clear();

            foreach (var shape in MediaPipe52BlendShapes)
            {
                Mappings.Add(new FaceBlendShapeMapping(shape, shape));
            }
        }

        /// <summary>
        /// Creates an in-memory or default profile with ARKit 52 defaults.
        /// </summary>
        public static FaceMappingProfile CreateDefault()
        {
            var profile = ScriptableObject.CreateInstance<FaceMappingProfile>();
            profile.ResetToStandard52();
            return profile;
        }

        private static string FindBestMatchForShape(string mpShape, List<string> availableShapes)
        {
            // 1. Exact case-insensitive match
            foreach (var avShape in availableShapes)
            {
                if (string.Equals(mpShape, avShape, StringComparison.OrdinalIgnoreCase))
                {
                    return avShape;
                }
            }

            // 2. Normalized match (ignoring underscores, dots, and hyphens)
            string normalizedMp = NormalizeShapeName(mpShape);
            foreach (var avShape in availableShapes)
            {
                if (string.Equals(normalizedMp, NormalizeShapeName(avShape), StringComparison.OrdinalIgnoreCase))
                {
                    return avShape;
                }
            }

            // 3. Known aliases and naming conventions (VRChat, VRoid/VRM, MMD)
            if (_shapeAliases.TryGetValue(mpShape, out string[] aliases))
            {
                foreach (var alias in aliases)
                {
                    string normAlias = NormalizeShapeName(alias);
                    foreach (var avShape in availableShapes)
                    {
                        string normAv = NormalizeShapeName(avShape);
                        if (string.Equals(alias, avShape, StringComparison.OrdinalIgnoreCase) ||
                            string.Equals(normAlias, normAv, StringComparison.OrdinalIgnoreCase) ||
                            normAv.EndsWith(normAlias, StringComparison.OrdinalIgnoreCase))
                        {
                            return avShape;
                        }
                    }
                }
            }

            return null;
        }

        private static string NormalizeShapeName(string name)
        {
            if (string.IsNullOrEmpty(name)) return string.Empty;
            return name.Replace("_", "").Replace(".", "").Replace("-", "").Replace(" ", "").ToLowerInvariant();
        }

        // Comprehensive dictionary of common avatar blendshape aliases for MediaPipe 52 shapes
        private static readonly Dictionary<string, string[]> _shapeAliases = new Dictionary<string, string[]>(StringComparer.OrdinalIgnoreCase)
        {
            // Eyes
            ["eyeBlinkLeft"] = new string[] {
                "vrc.blink_l", "blink_l", "blink_left", "eyeblink_l", "eye_blink_l",
                "fcl_eye_close_l", "fcl_eye_close", "eye_close_l", "blink", "まばたき", "ウィンク", "ウィンク左"
            },
            ["eyeBlinkRight"] = new string[] {
                "vrc.blink_r", "blink_r", "blink_right", "eyeblink_r", "eye_blink_r",
                "fcl_eye_close_r", "fcl_eye_close", "eye_close_r", "blink", "まばたき", "ウィンク右"
            },
            ["eyeWideLeft"] = new string[] {
                "eye_wide_l", "eyewide_l", "fcl_eye_surprised", "surprised", "びっくり", "目見開き", "見開き"
            },
            ["eyeWideRight"] = new string[] {
                "eye_wide_r", "eyewide_r", "fcl_eye_surprised", "surprised", "びっくり", "目見開き", "見開き"
            },
            ["eyeSquintLeft"] = new string[] {
                "eye_squint_l", "squint_l", "fcl_eye_joy", "笑い", "なごみ", "じと目"
            },
            ["eyeSquintRight"] = new string[] {
                "eye_squint_r", "squint_r", "fcl_eye_joy", "笑い", "なごみ", "じと目"
            },

            // Jaw & Mouth open
            ["jawOpen"] = new string[] {
                "jaw_open", "mouth_open", "vrc.v_aa", "v_aa", "aa", "fcl_mth_a", "mth_a", "fcl_mth_o", "a", "あ", "あ２", "開口", "口開"
            },
            ["mouthClose"] = new string[] {
                "mouth_close", "vrc.v_sil", "v_sil", "sil", "口閉", "ん"
            },

            // Smiles & Frowns
            ["mouthSmileLeft"] = new string[] {
                "mouth_smile_l", "smile_l", "smileleft", "fcl_mth_fun", "fcl_mth_joy", "fcl_mth_s", "fcl_all_joy", "joy", "fun", "smile", "笑い", "口角上げ", "にやり"
            },
            ["mouthSmileRight"] = new string[] {
                "mouth_smile_r", "smile_r", "smileright", "fcl_mth_fun", "fcl_mth_joy", "fcl_mth_s", "fcl_all_joy", "joy", "fun", "smile", "笑い", "口角上げ", "にやり"
            },
            ["mouthFrownLeft"] = new string[] {
                "mouth_frown_l", "frown_l", "fcl_mth_sorrow", "sorrow", "口角下げ", "へ", "困る"
            },
            ["mouthFrownRight"] = new string[] {
                "mouth_frown_r", "frown_r", "fcl_mth_sorrow", "sorrow", "口角下げ", "へ", "困る"
            },

            // Mouth Shapes & Visemes
            ["mouthFunnel"] = new string[] {
                "mouth_funnel", "funnel", "vrc.v_oh", "v_oh", "fcl_mth_o", "mth_o", "o", "お"
            },
            ["mouthPucker"] = new string[] {
                "mouth_pucker", "pucker", "vrc.v_ou", "v_ou", "fcl_mth_u", "mth_u", "u", "う"
            },
            ["mouthDimpleLeft"] = new string[] { "mouth_dimple_l", "dimple_l", "えくぼ_l" },
            ["mouthDimpleRight"] = new string[] { "mouth_dimple_r", "dimple_r", "えくぼ_r" },
            ["mouthStretchLeft"] = new string[] { "mouth_stretch_l", "stretch_l", "fcl_mth_i", "i", "い" },
            ["mouthStretchRight"] = new string[] { "mouth_stretch_r", "stretch_r", "fcl_mth_i", "i", "い" },

            // Brows
            ["browInnerUp"] = new string[] {
                "brow_inner_up", "brow_up", "fcl_brw_joy", "fcl_brw_surprised", "眉上", "上"
            },
            ["browDownLeft"] = new string[] {
                "brow_down_l", "fcl_brw_angry", "fcl_brw_sorrow", "怒り", "下", "困る", "眉下"
            },
            ["browDownRight"] = new string[] {
                "brow_down_r", "fcl_brw_angry", "fcl_brw_sorrow", "怒り", "下", "困る", "眉下"
            },
            ["browOuterUpLeft"] = new string[] { "brow_outer_up_l", "brow_up_l" },
            ["browOuterUpRight"] = new string[] { "brow_outer_up_r", "brow_up_r" },

            // Cheeks & Nose & Tongue
            ["cheekPuff"] = new string[] { "cheek_puff", "puff", "ぷくー", "頬膨らまし" },
            ["cheekSquintLeft"] = new string[] { "cheek_squint_l" },
            ["cheekSquintRight"] = new string[] { "cheek_squint_r" },
            ["noseSneerLeft"] = new string[] { "nose_sneer_l" },
            ["noseSneerRight"] = new string[] { "nose_sneer_r" },
            ["tongueOut"] = new string[] { "tongue_out", "tongue", "べー", "舌出し", "舌" }
        };
    }
}
