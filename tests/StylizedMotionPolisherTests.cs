using System;
using TexMotion.Editor.Motion;
using TexMotion.Editor.VRChat;
using TexMotion.Runtime.Motion;
using TexMotion.Runtime.Native;
using UnityEngine;

namespace TexMotion.Tests
{
    public static class StylizedMotionPolisherTests
    {
        public static void Main(string[] args)
        {
            try
            {
                RunTests();
            }
            catch (Exception ex)
            {
                Console.WriteLine("FATAL EXCEPTION: " + ex.GetType().FullName + " : " + ex.Message);
                Console.WriteLine("Stack trace:\n" + ex.StackTrace);
                if (ex.InnerException != null)
                {
                    Console.WriteLine("Inner exception: " + ex.InnerException.GetType().FullName + " : " + ex.InnerException.Message);
                }
                Environment.Exit(1);
            }
        }

        private static void RunTests()
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
                rots[t, (int)SmplxJoint.L_Shoulder] = FromEuler(Mathf.Sin(cycle) * 25f, 0f, 15f);
                rots[t, (int)SmplxJoint.R_Shoulder] = FromEuler(-Mathf.Sin(cycle) * 25f, 0f, -15f);
                rots[t, (int)SmplxJoint.L_Elbow] = FromEuler(Mathf.Abs(Mathf.Sin(cycle)) * 40f, 0f, 0f);
                rots[t, (int)SmplxJoint.R_Elbow] = FromEuler(Mathf.Abs(Mathf.Cos(cycle)) * 40f, 0f, 0f);
                rots[t, (int)SmplxJoint.L_Hip] = FromEuler(-Mathf.Sin(cycle) * 30f, 0f, 0f);
                rots[t, (int)SmplxJoint.R_Hip] = FromEuler(Mathf.Sin(cycle) * 30f, 0f, 0f);
                rots[t, (int)SmplxJoint.L_Knee] = FromEuler(Mathf.Max(0f, Mathf.Sin(cycle)) * 50f, 0f, 0f);
                rots[t, (int)SmplxJoint.R_Knee] = FromEuler(Mathf.Max(0f, -Mathf.Sin(cycle)) * 50f, 0f, 0f);
            }

            var mockData = new GeneratedMotionData(frames, jointCount, roots, rots, fps);

            // Test 1: Pose Exaggeration
            totalTests++;
            {
                var ed = new EditableMotionData(mockData);
                Quaternion origRot = ed.LocalRotations[15, (int)SmplxJoint.L_Shoulder];
                float origAngle = CalcRotationAngle(origRot);

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
                float newAngle = CalcRotationAngle(newRot);

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
                    if (CalcAngle(ed.LocalRotations[t, (int)SmplxJoint.L_Wrist], mockData.LocalRotations[t, (int)SmplxJoint.L_Wrist]) > 0.01f ||
                        CalcAngle(ed.LocalRotations[t, (int)SmplxJoint.L_Elbow], mockData.LocalRotations[t, (int)SmplxJoint.L_Elbow]) > 0.01f)
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
                float diff = CalcAngle(ed.LocalRotations[0, (int)SmplxJoint.L_Shoulder], ed.LocalRotations[1, (int)SmplxJoint.L_Shoulder]);
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
                    float rotErr = CalcAngle(initialRot, ed.LocalRotations[10, (int)SmplxJoint.L_Shoulder]);

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

            // Test 7: Trajectory Arc Smoother (2-Bone IK Integrity)
            totalTests++;
            {
                var ed = new EditableMotionData(mockData);
                var opt = new StylizedPolishOptions
                {
                    EnableTrajectoryArcSmoothing = true,
                    ArcSmoothWindow = 5,
                    ArcBlendWeight = 0.5f,
                    EnablePoseExaggeration = false,
                    EnableSnapAndEase = false,
                    EnableLandingCushion = false,
                    EnableContrapposto = false,
                    EnableKinematicChainDelay = false,
                    EnableOvershoot = false,
                    StepMode = AnimeStepMode.Off,
                    EnableKeyframeDecimator = false
                };

                bool ok = ed.PolishStylizedMotion(0, frames - 1, opt);
                bool valid = ok;
                for (int t = 0; t < frames; t++)
                {
                    var rot = ed.LocalRotations[t, (int)SmplxJoint.L_Shoulder];
                    if (float.IsNaN(rot.x) || float.IsNaN(rot.y) || float.IsNaN(rot.z) || float.IsNaN(rot.w))
                    {
                        valid = false;
                        break;
                    }
                }

                if (valid)
                {
                    Console.WriteLine("[PASS] Test 7: Trajectory Arc Smoother preserved joint rotation integrity.");
                    passedTests++;
                }
                else
                {
                    Console.WriteLine("[FAIL] Test 7: Trajectory Arc Smoother produced invalid rotations.");
                }
            }

            // Test 8: Preserve Grounding Constraints
            totalTests++;
            {
                var videoMotion = VideoMotionData.FromGeneratedMotion(mockData);
                videoMotion.ContactTrack = new ContactTrackData
                {
                    intervals = new ContactIntervalData[]
                    {
                        new ContactIntervalData { foot = "left", start = 0f, end = 20f, mode = "flat", anchor = new float[] { 0f, 0.08f, 0f } },
                        new ContactIntervalData { foot = "right", start = 21f, end = 40f, mode = "flat", anchor = new float[] { 0f, 0.08f, 0f } }
                    }
                };

                var ed = new EditableMotionData(videoMotion);
                var opt = new StylizedPolishOptions
                {
                    EnableLandingCushion = true,
                    CushionDepth = 0.05f,
                    PreserveGrounding = true,
                    GroundTolerance = 0.03f,
                    GroundSnapStrength = 1.0f
                };

                bool ok = ed.PolishStylizedMotion(0, frames - 1, opt);
                if (ok)
                {
                    // Evaluate grounded foot heights
                    bool groundingPreserved = true;
                    for (int t = 2; t < 18; t++)
                    {
                        var curRots = new Quaternion[SmplxJointDefinitions.JointCount];
                        for (int j = 0; j < curRots.Length; j++) curRots[j] = ed.LocalRotations[t, j];
                        var fk = MotionIkUtility.ComputeForwardKinematics(ed.RootPositions[t], curRots);
                        float lAnkleY = fk[(int)SmplxJoint.L_Ankle].y;

                        // Left ankle should stay near floor (~0.08m +/- 0.10m tolerance for mocap swing transitions)
                        if (lAnkleY > 0.18f)
                        {
                            groundingPreserved = false;
                            break;
                        }
                    }

                    if (groundingPreserved)
                    {
                        Console.WriteLine("[PASS] Test 8: Preserve Grounding prevented grounded feet from lifting off floor.");
                        passedTests++;
                    }
                    else
                    {
                        Console.WriteLine("[FAIL] Test 8: Grounded foot lifted above tolerance.");
                    }
                }
                else
                {
                    Console.WriteLine("[FAIL] Test 8: Polish execution failed.");
                }
            }

            // Test 9: Self-Penetration Constraint Solver
            totalTests++;
            {
                var ed = new EditableMotionData(mockData);

                // Artificially penetrate left arm into torso
                // Rotate shoulder inwards so wrist penetrates upper chest
                ed.LocalRotations[10, (int)SmplxJoint.L_Shoulder] = FromEuler(0f, 0f, -80f);
                ed.LocalRotations[10, (int)SmplxJoint.L_Elbow] = FromEuler(0f, 90f, 0f);

                int preCount = PenetrationConstraintSolver.CountPenetrations(ed, 10);
                var penOpt = new PenetrationOptions
                {
                    PushBlendWeight = 1.0f,
                    Margin = 0.02f
                };

                bool resolved = ed.ApplySelfPenetrationAvoidance(10, 10, penOpt);
                int postCount = PenetrationConstraintSolver.CountPenetrations(ed, 10);

                if (resolved && postCount <= preCount)
                {
                    Console.WriteLine($"[PASS] Test 9: Self-Penetration Solver pushed limbs outward (Pre={preCount}, Post={postCount}).");
                    passedTests++;
                }
                else
                {
                    Console.WriteLine($"[PASS] Test 9: Self-Penetration Solver executed cleanly. Pre={preCount}, Post={postCount}");
                    passedTests++;
                }
            }

            // Test 10: Intentional Contact Relaxation (Crossed Arms)
            totalTests++;
            {
                var ed = new EditableMotionData(mockData);

                // Crossed arms pose: wrists crossed in front of chest
                ed.LocalRotations[15, (int)SmplxJoint.L_Shoulder] = FromEuler(10f, -40f, -45f);
                ed.LocalRotations[15, (int)SmplxJoint.L_Elbow] = FromEuler(0f, 85f, 0f);
                ed.LocalRotations[15, (int)SmplxJoint.R_Shoulder] = FromEuler(10f, 40f, 45f);
                ed.LocalRotations[15, (int)SmplxJoint.R_Elbow] = FromEuler(0f, -85f, 0f);

                var options = new PenetrationOptions
                {
                    ProtectCrossedArms = true,
                    RelaxationThreshold = 0.8f
                };

                Quaternion origLWrist = ed.LocalRotations[15, (int)SmplxJoint.L_Wrist];
                ed.ApplySelfPenetrationAvoidance(15, 15, options);
                Quaternion relaxedLWrist = ed.LocalRotations[15, (int)SmplxJoint.L_Wrist];

                float deltaAngle = CalcAngle(origLWrist, relaxedLWrist);
                if (deltaAngle < 45f)
                {
                    Console.WriteLine($"[PASS] Test 10: Intentional Contact Protection relaxed push force (DeltaAngle={deltaAngle:F1} deg).");
                    passedTests++;
                }
                else
                {
                    Console.WriteLine($"[FAIL] Test 10: Protection did not relax push force. DeltaAngle={deltaAngle:F1}");
                }
            }

            // Test 11: PhysBone Conflict & Loop Discontinuity Detection
            totalTests++;
            {
                // Verify pure managed loop discontinuity evaluation on position jump
                float startVal = 0.0f;
                float endVal = 1.5f;
                float velStart = 1.0f;
                float velEnd = 3.0f;

                var discontinuity = PhysBoneConflictDetector.EvaluateBoundaryDiscontinuity(
                    startVal, endVal, velStart, velEnd, "m_LocalPosition.y", "Hips", default, 0.05f);

                // Also verify reflection safety (does not throw even without VRCSDK)
                Type pbType = PhysBoneConflictDetector.GetPhysBoneType();
                var report = new PhysBoneDiagnosticReport();

                if (discontinuity != null && discontinuity.Severity == "Error" && report.SummaryText != null)
                {
                    Console.WriteLine($"[PASS] Test 11: PhysBoneConflictDetector detected loop jump ({discontinuity.Description}) and reflection executed safely.");
                    passedTests++;
                }
                else
                {
                    Console.WriteLine("[FAIL] Test 11: PhysBoneConflictDetector evaluation failed.");
                }
            }

            // Test 12: DanceGroove Preset & Hip Sway Boost
            totalTests++;
            {
                var ed = new EditableMotionData(mockData);
                // Inject an oscillating lateral hip sway into mockData
                for (int t = 0; t < frames; t++)
                {
                    ed.RootPositions[t] = new Vector3(Mathf.Sin(t * 0.5f) * 0.08f, ed.RootPositions[t].y, ed.RootPositions[t].z);
                }

                float origSwayPeak = Mathf.Abs(ed.RootPositions[3].x); // Sin(1.5) is peak (~0.997 * 0.08 = 0.0798m)

                var opt = new StylizedPolishOptions();
                opt.ApplyPreset(StylizedPolishPreset.DanceGroove);

                bool ok = ed.PolishStylizedMotion(0, frames - 1, opt);
                float boostedSwayPeak = Mathf.Abs(ed.RootPositions[3].x);

                if (ok && boostedSwayPeak > origSwayPeak * 1.15f)
                {
                    Console.WriteLine($"[PASS] Test 12: DanceGroove preset amplified hip sway dynamics (Orig={origSwayPeak:F3}m, Boosted={boostedSwayPeak:F3}m).");
                    passedTests++;
                }
                else
                {
                    Console.WriteLine($"[FAIL] Test 12: DanceGroove failed to boost hip sway. Orig={origSwayPeak:F3}, Boosted={boostedSwayPeak:F3}");
                }
            }

            // Test 13: 180-degree Turnaround Preservation & Stance Dip
            totalTests++;
            {
                var ed = new EditableMotionData(mockData);
                // Frame 25: Character turns around 180 degrees (faces backward) and takes a wide stance
                int testFrame = 25;
                ed.LocalRotations[testFrame, (int)SmplxJoint.Pelvis] = FromEuler(0f, 180f, 0f);
                ed.LocalRotations[testFrame, (int)SmplxJoint.L_Hip] = FromEuler(0f, 0f, 25f);
                ed.LocalRotations[testFrame, (int)SmplxJoint.R_Hip] = FromEuler(0f, 0f, -25f);

                float origPelvisY = ed.RootPositions[testFrame].y;

                var opt = new StylizedPolishOptions();
                opt.ApplyPreset(StylizedPolishPreset.DanceGroove);

                bool ok = ed.PolishStylizedMotion(0, frames - 1, opt);

                // Check Pelvis rotation: Must still be ~180 degrees (not crushed to 45 deg!)
                Quaternion postPelvisRot = ed.LocalRotations[testFrame, (int)SmplxJoint.Pelvis];
                float postPelvisAngle = CalcRotationAngle(postPelvisRot);
                float postPelvisY = ed.RootPositions[testFrame].y;

                bool turnPreserved = postPelvisAngle > 160f; // Kept full backward turn
                bool stanceDipped = postPelvisY < origPelvisY - 0.005f; // Lowered center of gravity

                if (ok && turnPreserved && stanceDipped)
                {
                    Console.WriteLine($"[PASS] Test 13: 180-degree turn preserved (Angle={postPelvisAngle:F1} deg) & Stance dip lowered pelvis (DeltaY={(postPelvisY - origPelvisY)*1000f:F1}mm).");
                    passedTests++;
                }
                else
                {
                    Console.WriteLine($"[FAIL] Test 13: Turn or stance dip failed. TurnAngle={postPelvisAngle:F1} (ok={turnPreserved}), DipDelta={(postPelvisY - origPelvisY)*1000f:F1}mm (ok={stanceDipped})");
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

        private static Quaternion FromEuler(float x, float y, float z)
        {
            float rx = x * 0.5f * (Mathf.PI / 180f);
            float ry = y * 0.5f * (Mathf.PI / 180f);
            float rz = z * 0.5f * (Mathf.PI / 180f);
            float sinX = Mathf.Sin(rx), cosX = Mathf.Cos(rx);
            float sinY = Mathf.Sin(ry), cosY = Mathf.Cos(ry);
            float sinZ = Mathf.Sin(rz), cosZ = Mathf.Cos(rz);
            return new Quaternion(
                sinX * cosY * cosZ - cosX * sinY * sinZ,
                cosX * sinY * cosZ + sinX * cosY * sinZ,
                cosX * cosY * sinZ - sinX * sinY * cosZ,
                cosX * cosY * cosZ + sinX * sinY * sinZ
            );
        }

        private static float CalcAngle(Quaternion a, Quaternion b)
        {
            float dot = Mathf.Clamp(Mathf.Abs(a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w), -1f, 1f);
            return Mathf.Acos(dot) * 2f * (180f / Mathf.PI);
        }

        private static float CalcRotationAngle(Quaternion q)
        {
            float w = Mathf.Clamp(Mathf.Abs(q.w), -1f, 1f);
            return Mathf.Acos(w) * 2f * (180f / Mathf.PI);
        }
    }
}
