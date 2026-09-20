using System;
using UnityEngine;
using TexMotion.Runtime.Motion;
using TexMotion.Editor.Motion;

namespace TexMotion.Tests
{
    public static class InpaintVerificationTests
    {
        public static int Main(string[] args)
        {
            int passCount = 0;
            int totalTests = 4;
            Console.WriteLine("=== Running Kinematic Hermite Inpainting Verification Tests ===");

            try
            {
                // Test 1: VideoMotionData InpaintGapsKinematicHermite basic detection and repair
                int frames = 60;
                var rootPositions = new Vector3[frames];
                var localRotations = new Quaternion[frames, SmplxJointDefinitions.JointCount];
                var confidences = new float[frames];
                var timestamps = new float[frames];

                for (int t = 0; t < frames; t++)
                {
                    timestamps[t] = t / 30f;
                    confidences[t] = 1.0f;
                    rootPositions[t] = new Vector3(0f, 0.95f, t * 0.05f); // moving forward 1.5 m/s
                    for (int j = 0; j < SmplxJointDefinitions.JointCount; j++)
                    {
                        localRotations[t, j] = Quaternion.identity;
                    }
                }

                // Introduce a 15-frame dropout (0.5s) from frame 20 to 34
                for (int t = 20; t <= 34; t++)
                {
                    confidences[t] = 0.1f;
                }

                var motionData = new VideoMotionData(
                    frames,
                    SmplxJointDefinitions.JointCount,
                    rootPositions,
                    localRotations,
                    frameRate: 30f,
                    timestamps: timestamps,
                    confidences: confidences
                );

                int inpainted = motionData.InpaintGapsKinematicHermite(
                    minConfidenceThreshold: 0.35f,
                    minGapSeconds: 0.2f,
                    maxGapSeconds: 2.0f,
                    damping: 0.2f
                );

                if (inpainted > 0 && motionData.Confidences[25] >= 0.6f)
                {
                    Console.WriteLine($"[PASS] Test 1: InpaintGapsKinematicHermite repaired {inpainted} frame intervals and restored confidence.");
                    passCount++;
                }
                else
                {
                    Console.WriteLine($"[FAIL] Test 1: Inpainted count was {inpainted}, conf was {motionData.Confidences[25]}");
                }

                // Test 2: Root trajectory progression is smooth and monotonic
                bool monotonicForward = true;
                for (int t = 19; t <= 35; t++)
                {
                    if (motionData.RootPositions[t].z <= motionData.RootPositions[t - 1].z)
                    {
                        monotonicForward = false;
                        break;
                    }
                }

                if (monotonicForward)
                {
                    Console.WriteLine("[PASS] Test 2: Root position Z advanced monotonically across the gap without backward jump.");
                    passCount++;
                }
                else
                {
                    Console.WriteLine("[FAIL] Test 2: Root position Z did not advance monotonically.");
                }

                // Test 3: Rotations remain valid unit quaternions
                bool allUnit = true;
                for (int t = 20; t <= 34; t++)
                {
                    for (int j = 0; j < SmplxJointDefinitions.JointCount; j++)
                    {
                        var q = motionData.LocalRotations[t, j];
                        float mag = Mathf.Sqrt(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w);
                        if (Mathf.Abs(mag - 1.0f) > 1e-4f)
                        {
                            allUnit = false;
                            break;
                        }
                    }
                    if (!allUnit) break;
                }

                if (allUnit)
                {
                    Console.WriteLine("[PASS] Test 3: Inpainted local rotations are all valid unit quaternions.");
                    passCount++;
                }
                else
                {
                    Console.WriteLine("[FAIL] Test 3: Inpainted rotations contained non-unit quaternions.");
                }

                // Test 4: EditableMotionData Undo/Redo integration
                var editable = new EditableMotionData(motionData);
                // Modify a frame to record an initial state
                Vector3 origRoot = editable.RootPositions[25];
                int undoInpainted = editable.InpaintGapsKinematicHermite(0.35f, 0.2f, 2.0f, 0.2f);
                editable.Undo();

                if (Vector3.Distance(editable.RootPositions[25], origRoot) < 1e-5f)
                {
                    Console.WriteLine("[PASS] Test 4: EditableMotionData.InpaintGapsKinematicHermite supports complete Undo restoration.");
                    passCount++;
                }
                else
                {
                    Console.WriteLine($"[FAIL] Test 4: Undo did not restore root position (Diff: {Vector3.Distance(editable.RootPositions[25], origRoot)})");
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[FAIL] Exception thrown: {ex}");
            }

            Console.WriteLine($"=== Test Summary: {passCount}/{totalTests} PASSED ===");
            return passCount == totalTests ? 0 : 1;
        }
    }
}
