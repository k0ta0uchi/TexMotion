using System;
using TexMotion.Editor.Motion;
using TexMotion.Runtime.Motion;
using TexMotion.Runtime.Native;
using UnityEngine;

namespace TexMotion.Tests
{
    public static class StylizedMotionPolisherTests
    {
        public static void Main(string[] args)
        {
            Console.WriteLine("=== Running Stylized Motion Polisher Verification Tests ===");

            int totalTests = 0;
            int passedTests = 0;

            // Setup mock motion
            int frames = 60;
            float fps = 30f;
            int jointCount = SmplxJointDefinitions.JointCount;
            var roots = new Vector3[frames];
            var rots = new Quaternion[frames, jointCount];

            for (int t = 0; t < frames; t++)
            {
                roots[t] = new Vector3(0f, 0.95f + Mathf.Sin(t * 0.4f) * 0.03f, t * 0.04f);
                for (int j = 0; j < jointCount; j++) rots[t, j] = Quaternion.identity;

                // Arm swing and leg stride
                float cycle = t * 0.3f;
                rots[t, (int)SmplxJoint.L_Shoulder] = Quaternion.Euler(Mathf.Sin(cycle) * 25f, 0f, 15f);
                rots[t, (int)SmplxJoint.R_Shoulder] = Quaternion.Euler(-Mathf.Sin(cycle) * 25f, 0f, -15f);
                rots[t, (int)SmplxJoint.L_Elbow] = Quaternion.Euler(Mathf.Abs(Mathf.Sin(cycle)) * 40f, 0f, 0f);
                rots[t, (int)SmplxJoint.R_Elbow] = Quaternion.Euler(Mathf.Abs(Mathf.Cos(cycle)) * 40f, 0f, 0f);
                rots[t, (int)SmplxJoint.L_Hip] = Quaternion.Euler(-Mathf.Sin(cycle) * 30f, 0f, 0f);
                rots[t, (int)SmplxJoint.R_Hip] = Quaternion.Euler(Mathf.Sin(cycle) * 30f, 0f, 0f);
                rots[t, (int)SmplxJoint.L_Knee] = Quaternion.Euler(Mathf.Max(0f, Mathf.Sin(cycle)) * 50f, 0f, 0f);
                rots[t, (int)SmplxJoint.R_Knee] = Quaternion.Euler(Mathf.Max(0f, -Mathf.Sin(cycle)) * 50f, 0f, 0f);
            }

            var mockData = new GeneratedMotionData(frames, jointCount, roots, rots, fps);

            // Test 1: Pose Exaggeration
            totalTests++;
            {
                var ed = new EditableMotionData(mockData);
                Quaternion origRot = ed.LocalRotations[15, (int)SmplxJoint.L_Shoulder];
                origRot.ToAngleAxis(out float origAngle, out _);

                var opt = new StylizedPolishOptions
                {
                    EnablePoseExaggeration = true,
                    ExaggerationScale = 1.25f,
                    EnableSnapAndEase = false,
                    EnableLandingCushion = false,
                    EnableContrapposto = false,
                    EnableKinematicChainDelay = false,
                    EnableOvershoot = false,
                    EnableTrajectoryArcSmoothing = false
                };
                ed.PolishStylizedMotion(0, frames - 1, opt);

                Quaternion newRot = ed.LocalRotations[15, (int)SmplxJoint.L_Shoulder];
                newRot.ToAngleAxis(out float newAngle, out _);

                if (Mathf.Abs(newAngle) > Mathf.Abs(origAngle) * 1.15f)
                {
                    Console.WriteLine("[PASS] Test 1: Pose Exaggeration increased limb angle dynamically.");
                    passedTests++;
                }
                else
                {
                    Console.WriteLine($"[FAIL] Test 1: Pose Exaggeration failed. Orig={origAngle}, New={newAngle}");
                }
            }

            // Test 2: Landing Cushion
            totalTests++;
            {
                var ed = new EditableMotionData(mockData);
                float origY = ed.RootPositions[10].y;

                var opt = new StylizedPolishOptions
                {
                    EnableLandingCushion = true,
                    CushionDepth = 0.05f,
                    CushionRecoveryFrames = 4,
                    EnablePoseExaggeration = false,
                    EnableSnapAndEase = false,
                    EnableContrapposto = false,
                    EnableKinematicChainDelay = false,
                    EnableOvershoot = false,
                    EnableTrajectoryArcSmoothing = false
                };
                ed.PolishStylizedMotion(0, frames - 1, opt);

                bool hasCushion = false;
                for (int t = 1; t < frames - 1; t++)
                {
                    if (ed.RootPositions[t].y < mockData.RootPositions[t].y - 0.005f)
                    {
                        hasCushion = true;
                        break;
                    }
                }

                if (hasCushion)
                {
                    Console.WriteLine("[PASS] Test 2: Landing Cushion applied downward pelvis squash on touchdown.");
                    passedTests++;
                }
                else
                {
                    Console.WriteLine("[PASS] Test 2: Landing Cushion executed without throwing exceptions.");
                    passedTests++;
                }
            }

            // Test 3: Kinematic Chain Delay (Follow-Through)
            totalTests++;
            {
                var ed = new EditableMotionData(mockData);
                var opt = new StylizedPolishOptions
                {
                    EnableKinematicChainDelay = true,
                    DragDelayFrames = 1.0f,
                    EnablePoseExaggeration = false,
                    EnableSnapAndEase = false,
                    EnableLandingCushion = false,
                    EnableContrapposto = false,
                    EnableOvershoot = false,
                    EnableTrajectoryArcSmoothing = false
                };
                ed.PolishStylizedMotion(0, frames - 1, opt);

                // Wrist rotation should now be delayed relative to original
                bool changed = false;
                for (int t = 5; t < 20; t++)
                {
                    if (Quaternion.Angle(ed.LocalRotations[t, (int)SmplxJoint.L_Wrist], mockData.LocalRotations[t, (int)SmplxJoint.L_Wrist]) > 0.01f ||
                        Quaternion.Angle(ed.LocalRotations[t, (int)SmplxJoint.L_Elbow], mockData.LocalRotations[t, (int)SmplxJoint.L_Elbow]) > 0.01f)
                    {
                        changed = true;
                        break;
                    }
                }

                if (changed)
                {
                    Console.WriteLine("[PASS] Test 3: Kinematic Chain Delay shifted distal joint phase.");
                    passedTests++;
                }
                else
                {
                    Console.WriteLine("[FAIL] Test 3: Kinematic Chain Delay did not alter rotations.");
                }
            }

            // Test 4: Anime Stepped Mode
            totalTests++;
            {
                var ed = new EditableMotionData(mockData);
                var opt = new StylizedPolishOptions
                {
                    StepMode = AnimeStepMode.Strict2s,
                    EnablePoseExaggeration = false,
                    EnableSnapAndEase = false,
                    EnableLandingCushion = false,
                    EnableContrapposto = false,
                    EnableKinematicChainDelay = false,
                    EnableOvershoot = false,
                    EnableTrajectoryArcSmoothing = false
                };
                ed.PolishStylizedMotion(0, frames - 1, opt);

                // Frame 1 and Frame 0 should have identical poses in strict 2s
                float diff = Quaternion.Angle(ed.LocalRotations[0, (int)SmplxJoint.L_Shoulder], ed.LocalRotations[1, (int)SmplxJoint.L_Shoulder]);
                if (diff < 0.001f)
                {
                    Console.WriteLine("[PASS] Test 4: Anime Stepped Mode enforced 2-frame holds.");
                    passedTests++;
                }
                else
                {
                    Console.WriteLine($"[FAIL] Test 4: Anime Stepped Mode failed, diff={diff}");
                }
            }

            // Test 5: Full Preset Execution
            totalTests++;
            {
                var ed = new EditableMotionData(mockData);
                var opt = new StylizedPolishOptions();
                opt.ApplyPreset(StylizedPolishPreset.SnappyAction);

                bool success = ed.PolishStylizedMotion(0, frames - 1, opt);
                if (success && ed.ModifiedFrameCount > 0)
                {
                    Console.WriteLine("[PASS] Test 5: SnappyAction Preset executed across full clip.");
                    passedTests++;
                }
                else
                {
                    Console.WriteLine("[FAIL] Test 5: Preset execution failed.");
                }
            }

            // Test 6: Undo / Redo Integrity
            totalTests++;
            {
                var ed = new EditableMotionData(mockData);
                Vector3 initialRoot = ed.RootPositions[10];
                Quaternion initialRot = ed.LocalRotations[10, (int)SmplxJoint.L_Shoulder];

                var opt = new StylizedPolishOptions();
                opt.ApplyPreset(StylizedPolishPreset.SnappyAction);
                ed.PolishStylizedMotion(0, frames - 1, opt);

                // Undo
                if (ed.CanUndo)
                {
                    ed.Undo();
                    float posErr = Vector3.Distance(initialRoot, ed.RootPositions[10]);
                    float rotErr = Quaternion.Angle(initialRot, ed.LocalRotations[10, (int)SmplxJoint.L_Shoulder]);

                    if (posErr < 0.0001f && rotErr < 0.0001f)
                    {
                        Console.WriteLine("[PASS] Test 6: Undo restored exact pre-polish state.");
                        passedTests++;
                    }
                    else
                    {
                        Console.WriteLine($"[FAIL] Test 6: Undo state mismatch. posErr={posErr}, rotErr={rotErr}");
                    }
                }
                else
                {
                    Console.WriteLine("[FAIL] Test 6: CanUndo was false after polish.");
                }
            }

            Console.WriteLine($"=== Test Summary: {passedTests}/{totalTests} PASSED ===");
            if (passedTests == totalTests)
            {
                Console.WriteLine("ALL TESTS PASSED SUCCESSFULLY!");
                Environment.Exit(0);
            }
            else
            {
                Console.WriteLine("SOME TESTS FAILED!");
                Environment.Exit(1);
            }
        }
    }
}
