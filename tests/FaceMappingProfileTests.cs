using System;
using System.Collections.Generic;
using TexMotion.Editor.Motion;
using TexMotion.Editor.VRChat;
using TexMotion.Runtime.Motion;
using TexMotion.Runtime.Native;
using TexMotion.Runtime.VRChat;
using UnityEngine;

namespace TexMotion.Tests
{
    public static class FaceMappingProfileTests
    {
        public static void Main(string[] args)
        {
            Console.WriteLine("=================================================");
            Console.WriteLine("=== FaceMappingProfile & VRChat Face Sync Tests ===");
            Console.WriteLine("=================================================");

            int totalTests = 0;
            int passedTests = 0;

            void Assert(bool condition, string testName, string details = "")
            {
                totalTests++;
                if (condition)
                {
                    passedTests++;
                    Console.WriteLine($"[PASS] {testName}");
                }
                else
                {
                    Console.ForegroundColor = ConsoleColor.Red;
                    Console.WriteLine($"[FAIL] {testName} - {details}");
                    Console.ResetColor();
                }
            }

            // 1. Standard 52 Shape Names Integrity
            {
                var shapes = FaceMappingProfile.MediaPipe52BlendShapes;
                Assert(shapes != null && shapes.Length == 52, "MediaPipe 52 Blendshapes Count", $"Expected 52, got {shapes?.Length ?? 0}");

                var unique = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
                bool allUnique = true;
                foreach (var s in shapes)
                {
                    if (!unique.Add(s)) { allUnique = false; break; }
                }
                Assert(allUnique, "MediaPipe 52 Unique Names");
            }

            // 2. Opposite Shape Name (Left / Right Symmetry)
            {
                Assert(FaceMappingProfile.GetOppositeShapeName("eyeBlinkLeft") == "eyeBlinkRight", "Opposite Left->Right CamelCase");
                Assert(FaceMappingProfile.GetOppositeShapeName("eyeBlinkRight") == "eyeBlinkLeft", "Opposite Right->Left CamelCase");
                Assert(FaceMappingProfile.GetOppositeShapeName("mouthSmile_L") == "mouthSmile_R", "Opposite _L->_R");
                Assert(FaceMappingProfile.GetOppositeShapeName("mouthSmile_R") == "mouthSmile_L", "Opposite _R->_L");
                Assert(FaceMappingProfile.GetOppositeShapeName("vrc.blink.l") == "vrc.blink.r", "Opposite .l->.r");
                Assert(FaceMappingProfile.GetOppositeShapeName("jawOpen") == "jawOpen", "Symmetric shape remains unchanged");
            }

            // 3. Weight Evaluation (Gain, DeadZone, Global Multiplier)
            {
                var profile = ScriptableObject.CreateInstance<FaceMappingProfile>();
                profile.GlobalMultiplier = 1.0f;

                var mapping = new FaceBlendShapeMapping("jawOpen", "vrc.v_aa", 1.0f, 0.0f, false);
                // 0.5 raw -> 50.0 Unity blendshape weight
                float w1 = profile.EvaluateWeight(mapping, 0.5f);
                Assert(Mathf.Approximately(w1, 50.0f), "EvaluateWeight Linear 0.5 -> 50.0", $"Got {w1}");

                // Deadzone test (0.2 deadzone, 0.1 raw -> 0.0)
                mapping.DeadZone = 0.2f;
                float w2 = profile.EvaluateWeight(mapping, 0.1f);
                Assert(Mathf.Approximately(w2, 0.0f), "EvaluateWeight Below DeadZone -> 0.0", $"Got {w2}");

                // Deadzone remapped test (0.2 deadzone, 0.6 raw -> (0.6 - 0.2)/(0.8) = 0.5 -> 50.0)
                float w3 = profile.EvaluateWeight(mapping, 0.6f);
                Assert(Mathf.Approximately(w3, 50.0f), "EvaluateWeight Above DeadZone Remapped", $"Got {w3}");

                // Multiplier test (Multiplier = 1.5, raw 0.5, no deadzone -> 75.0)
                mapping.DeadZone = 0.0f;
                mapping.Multiplier = 1.5f;
                float w4 = profile.EvaluateWeight(mapping, 0.5f);
                Assert(Mathf.Approximately(w4, 75.0f), "EvaluateWeight Multiplier 1.5 -> 75.0", $"Got {w4}");

                // Disabled mapping -> 0.0
                mapping.Enabled = false;
                float w5 = profile.EvaluateWeight(mapping, 0.8f);
                Assert(Mathf.Approximately(w5, 0.0f), "EvaluateWeight Disabled -> 0.0", $"Got {w5}");
            }

            // 4. FaceTrackData Helper Methods
            {
                var faceTrack = new FaceTrackData();
                faceTrack.shapes = new FaceShapeTrackData[]
                {
                    new FaceShapeTrackData { shapeName = "eyeBlinkLeft", weights = new float[] { 0.0f, 0.5f, 1.0f } },
                    new FaceShapeTrackData { shapeName = "jawOpen", weights = new float[] { 0.1f, 0.2f, 0.3f } }
                };
                faceTrack.headRotationsFlat = new float[]
                {
                    0f, 0f, 0f, 1f, // Frame 0
                    0f, 0.1f, 0f, 0.995f // Frame 1
                };

                Assert(faceTrack.HasHeadRotations, "FaceTrack HasHeadRotations True");
                Assert(faceTrack.FindShape("eyeBlinkLeft") != null, "FaceTrack FindShape Match");
                Assert(faceTrack.FindShape("EYEBLINKLEFT") != null, "FaceTrack FindShape Case Insensitive");
                Assert(faceTrack.FindShape("nonExistent") == null, "FaceTrack FindShape Missing");

                Assert(Mathf.Approximately(faceTrack.GetWeight("eyeBlinkLeft", 1), 0.5f), "FaceTrack GetWeight Match", $"Got {faceTrack.GetWeight("eyeBlinkLeft", 1)}");
                Assert(Mathf.Approximately(faceTrack.GetWeight("eyeBlinkLeft", 99), 0.0f), "FaceTrack GetWeight OutOfRange -> 0.0");

                Quaternion q0 = faceTrack.GetHeadRotation(0);
                Assert(q0 == Quaternion.identity, "FaceTrack GetHeadRotation Frame 0 Identity");

                Quaternion q1 = faceTrack.GetHeadRotation(1);
                Assert(Mathf.Approximately(q1.y, 0.1f), "FaceTrack GetHeadRotation Frame 1 Y match", $"Got {q1.y}");
            }

            // 5. Invert Left/Right Effective Source
            {
                var profile = ScriptableObject.CreateInstance<FaceMappingProfile>();
                var mappingNormal = new FaceBlendShapeMapping("eyeBlinkLeft", "vrc.blink_l", 1.0f, 0f, false);
                var mappingInverted = new FaceBlendShapeMapping("eyeBlinkLeft", "vrc.blink_l", 1.0f, 0f, true);

                Assert(profile.GetEffectiveSourceShape(mappingNormal) == "eyeBlinkLeft", "Effective Source Normal");
                Assert(profile.GetEffectiveSourceShape(mappingInverted) == "eyeBlinkRight", "Effective Source Inverted");
            }

            // 6. ResetToStandard52 and FindMapping
            {
                var profile = ScriptableObject.CreateInstance<FaceMappingProfile>();
                profile.ResetToStandard52();

                Assert(profile.Mappings.Count == 52, "ResetToStandard52 Count 52", $"Got {profile.Mappings.Count}");
                var blinkL = profile.FindMapping("eyeBlinkLeft");
                Assert(blinkL != null && blinkL.TargetShapeName == "eyeBlinkLeft", "FindMapping eyeBlinkLeft");
            }

            // 7. VideoMotionData with FaceTrack & Backward Compatibility
            {
                var vmd = VideoMotionData.CreateEmpty(10, 30f);
                Assert(vmd.FaceTrack == null, "VideoMotionData Default FaceTrack is null");

                vmd.FaceTrack = new FaceTrackData
                {
                    shapes = new FaceShapeTrackData[]
                    {
                        new FaceShapeTrackData { shapeName = "jawOpen", weights = new float[10] }
                    }
                };
                Assert(vmd.FaceTrack != null, "VideoMotionData FaceTrack Assigned");
            }

            // 8. AnimationClipBuilder Options and FaceFX Mode
            {
                var options = AnimationBuildOptions.CreateDefault("TestClip");
                Assert(!options.ExportFaceOnly, "Default ExportFaceOnly is false");
                Assert(!options.ApplyHeadRotationFromFace, "Default ApplyHeadRotationFromFace is false");
                Assert(options.HeadRotationBlendWeight == 1.0f, "Default HeadRotationBlendWeight is 1.0");

                var vmd = VideoMotionData.CreateEmpty(5, 30f);
                vmd.FaceTrack = new FaceTrackData
                {
                    shapes = new FaceShapeTrackData[]
                    {
                        new FaceShapeTrackData { shapeName = "jawOpen", weights = new float[] { 0.0f, 0.2f, 0.5f, 0.8f, 1.0f } },
                        new FaceShapeTrackData { shapeName = "eyeBlinkLeft", weights = new float[] { 0.0f, 0.0f, 1.0f, 1.0f, 0.0f } }
                    }
                };

                // Test building face-only clip (VRChat FX Layer)
                var faceProfile = ScriptableObject.CreateInstance<FaceMappingProfile>();
                faceProfile.TargetMeshPath = "Body";
                faceProfile.ResetToStandard52();

                var faceClip = AnimationClipBuilder.BuildFaceAnimationClip(vmd, options, faceProfile);
                Assert(faceClip != null, "BuildFaceAnimationClip Returns Valid Clip");
                Assert(faceClip.name.Contains("FaceFX") || faceClip.name.Contains("TestClip_FX"), "FaceClip Name Formatting", $"Name: {faceClip?.name}");

                // Test synchronized clips generation
                AnimationClipBuilder.BuildSynchronizedClips(vmd, options, faceProfile, out AnimationClip bodyClip, out AnimationClip syncedFaceClip);
                Assert(bodyClip != null, "BuildSynchronizedClips BodyClip generated");
                Assert(syncedFaceClip != null, "BuildSynchronizedClips SyncedFaceClip generated");
            }

            // 9. VrcMotionConfig Facial Sync Fields
            {
                var config = new VrcMotionConfig();
                Assert(!config.SyncFaceToFxLayer, "VrcMotionConfig SyncFaceToFxLayer default false");
                config.SyncFaceToFxLayer = true;
                Assert(config.SyncFaceToFxLayer, "VrcMotionConfig SyncFaceToFxLayer set true");
            }

            Console.WriteLine("=================================================");
            Console.WriteLine($"=== RESULTS: {passedTests} / {totalTests} Tests Passed ===");
            Console.WriteLine("=================================================");

            if (passedTests != totalTests)
            {
                Environment.Exit(1);
            }
        }
    }
}
