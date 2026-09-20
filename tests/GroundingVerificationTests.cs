using System;
using System.Collections.Generic;
using TexMotion.Editor.Motion;
using TexMotion.Runtime.Motion;
using TexMotion.Runtime.Native;
using UnityEngine;

namespace TexMotion.Tests
{
    public static class GroundingVerificationTests
    {
        public static void Main(string[] args)
        {
            Console.WriteLine("=== Running Unity Grounding & Contact Constraint Solver Verification Tests ===");

            int totalTests = 0;
            int passedTests = 0;

            // Setup mock walking and jumping motion (60 frames, 30fps)
            // Frames 0..20: Left foot contact, Right swinging
            // Frames 21..35: Jump / airborne phase (both feet off ground)
            // Frames 36..59: Right foot contact, Left swinging
            int frames = 60;
            float fps = 30f;
            int jointCount = SmplxJointDefinitions.JointCount;
            var roots = new Vector3[frames];
            var rots = new Quaternion[frames, jointCount];

            for (int t = 0; t < frames; t++)
            {
                float time = (float)t / fps;
                float height = 0.95f;
                if (t >= 21 && t <= 35)
                {
                    // Airborne jump peak
                    float jumpPhase = (float)(t - 21) / 14f;
                    height += Mathf.Sin(jumpPhase * Mathf.PI) * 0.30f;
                }

                roots[t] = new Vector3(0f, height, time * 0.5f); // Moving forward along Z
                for (int j = 0; j < jointCount; j++) rots[t, j] = Quaternion.identity;

                // Leg pose using pure managed euler conversion
                rots[t, (int)SmplxJoint.L_Hip] = FromEuler(15f, 0f, 0f);
                rots[t, (int)SmplxJoint.L_Knee] = FromEuler(20f, 0f, 0f);
                rots[t, (int)SmplxJoint.R_Hip] = FromEuler(-15f, 0f, 0f);
                rots[t, (int)SmplxJoint.R_Knee] = FromEuler(10f, 0f, 0f);
            }

            var mockMotion = new GeneratedMotionData(frames, jointCount, roots, rots, fps);

            // ==========================================
            // Test 1: AvatarLegDimensions Fallback Check
            // ==========================================
            totalTests++;
            {
                var dims = AvatarLegDimensions.ExtractFromAvatar(null);
                bool valid = dims != null &&
                             Mathf.Abs(dims.UpperLegLengthL - 0.42f) < 0.01f &&
                             Mathf.Abs(dims.LowerLegLengthL - 0.41f) < 0.01f &&
                             Mathf.Abs(dims.SoleOffsetL - 0.08f) < 0.01f &&
                             dims.PelvisToHipL.x < 0f &&
                             dims.PelvisToHipR.x > 0f &&
                             !dims.IsCustomAvatar;

                if (valid)
                {
                    Console.WriteLine("[PASS] Test 1: AvatarLegDimensions correctly falls back to standard humanoid anatomical proportions.");
                    passedTests++;
                }
                else
                {
                    Console.WriteLine($"[FAIL] Test 1: AvatarLegDimensions fallback failed: Upper={dims?.UpperLegLengthL}, Lower={dims?.LowerLegLengthL}, Sole={dims?.SoleOffsetL}");
                }
            }

            // ==========================================
            // Test 2: ContactTrack Operations & Queries
            // ==========================================
            totalTests++;
            {
                var ed = new EditableMotionData(mockMotion);
                ed.AddContactInterval("left", 0.0f, 0.66f, "flat", 1.0f);
                ed.AddContactInterval("right", 1.20f, 1.96f, "flat", 1.0f);

                bool hasLeft0 = ed.IsFootInContact("left", 0.2f);
                bool hasRight0 = ed.IsFootInContact("right", 0.2f);
                bool hasLeftAir = ed.IsFootInContact("left", 0.9f); // Jump phase
                bool hasRightAir = ed.IsFootInContact("right", 0.9f); // Jump phase
                bool hasRightEnd = ed.IsFootInContact("right", 1.5f);

                var leftActive = ed.GetActiveContactInterval("left", 0.2f);
                var intervals = ed.GetContactIntervals();

                bool ok = hasLeft0 && !hasRight0 && !hasLeftAir && !hasRightAir && hasRightEnd &&
                          leftActive != null && intervals.Count == 2;

                if (ok)
                {
                    Console.WriteLine("[PASS] Test 2: ContactTrack intervals addition and time queries succeeded.");
                    passedTests++;
                }
                else
                {
                    Console.WriteLine($"[FAIL] Test 2: Contact queries failed. left0={hasLeft0}, right0={hasRight0}, leftAir={hasLeftAir}, rightAir={hasRightAir}, rightEnd={hasRightEnd}");
                }
            }

            // ==========================================
            // Test 3: ContactTrack Undo / Redo
            // ==========================================
            totalTests++;
            {
                var ed = new EditableMotionData(mockMotion);
                int initialCount = ed.GetContactIntervals().Count; // 0

                ed.AddContactInterval("left", 0f, 0.5f);
                int afterAdd = ed.GetContactIntervals().Count; // 1

                ed.AddContactInterval("right", 1.0f, 1.5f);
                int afterAdd2 = ed.GetContactIntervals().Count; // 2

                bool undoOk = ed.Undo();
                int afterUndo1 = ed.GetContactIntervals().Count; // 1

                ed.Undo();
                int afterUndo2 = ed.GetContactIntervals().Count; // 0

                ed.Redo();
                int afterRedo = ed.GetContactIntervals().Count; // 1

                if (initialCount == 0 && afterAdd == 1 && afterAdd2 == 2 && afterUndo1 == 1 && afterUndo2 == 0 && afterRedo == 1)
                {
                    Console.WriteLine("[PASS] Test 3: ContactTrack editing supports full Undo / Redo tracking.");
                    passedTests++;
                }
                else
                {
                    Console.WriteLine($"[FAIL] Test 3: Undo/Redo failed. init={initialCount}, add1={afterAdd}, add2={afterAdd2}, undo1={afterUndo1}, undo2={afterUndo2}, redo={afterRedo}");
                }
            }

            // ==========================================
            // Test 4: Contact Constraint Solver (Foot Sliding Suppression & Grounding IK)
            // ==========================================
            totalTests++;
            {
                var ed = new EditableMotionData(mockMotion);
                ed.AddContactInterval("left", 0.0f, 0.60f, "flat", 1.0f); // Frames 0..18
                ed.AddContactInterval("right", 1.20f, 1.90f, "flat", 1.0f); // Frames 36..57

                var dims = AvatarLegDimensions.ExtractFromAvatar(null);

                // Compute pre-solver FK for Left foot at frames 5 and 10 (notice that root is moving along Z)
                ContactConstraintSolver.ComputeLegFK(ed, 5, dims, isLeft: true, out _, out _, out Vector3 preAnkleF5, out Vector3 preFootF5);
                ContactConstraintSolver.ComputeLegFK(ed, 10, dims, isLeft: true, out _, out _, out Vector3 preAnkleF10, out Vector3 preFootF10);

                float preSlideZ = Mathf.Abs(preAnkleF10.z - preAnkleF5.z); // Sliding because root moved forward

                // Apply Grounding Constraint
                ed.ApplyAvatarGroundingConstraint(null, blendWeight: 1.0f);

                // Compute post-solver FK for Left foot at frames 5 and 10
                ContactConstraintSolver.ComputeLegFK(ed, 5, dims, isLeft: true, out _, out _, out Vector3 postAnkleF5, out Vector3 postFootF5);
                ContactConstraintSolver.ComputeLegFK(ed, 10, dims, isLeft: true, out _, out _, out Vector3 postAnkleF10, out Vector3 postFootF10);

                float postSlideZ = Mathf.Abs(postAnkleF10.z - postAnkleF5.z); // Should be near zero because anchor is locked

                // Foot sole height should be at floor level (Y ~ 0)
                float footHeightF5 = postFootF5.y;
                float footHeightF10 = postFootF10.y;

                bool slideEliminated = postSlideZ < preSlideZ * 0.25f && postSlideZ < 0.03f;
                bool grounded = Mathf.Abs(footHeightF5) < 0.06f && Mathf.Abs(footHeightF10) < 0.06f;

                if (slideEliminated && grounded)
                {
                    Console.WriteLine($"[PASS] Test 4: Foot slide eliminated (Pre={preSlideZ:F3}m, Post={postSlideZ:F3}m) and foot grounded (Height={footHeightF5:F3}m).");
                    passedTests++;
                }
                else
                {
                    Console.WriteLine($"[FAIL] Test 4: Grounding IK failed. preSlide={preSlideZ}, postSlide={postSlideZ}, footH5={footHeightF5}, footH10={footHeightF10}");
                }
            }

            // ==========================================
            // Test 5: Jumping / Airborne Non-Grounding Preservation
            // ==========================================
            totalTests++;
            {
                var ed = new EditableMotionData(mockMotion);
                ed.AddContactInterval("left", 0.0f, 0.60f, "flat", 1.0f);
                ed.AddContactInterval("right", 1.20f, 1.90f, "flat", 1.0f);

                // Check Jump phase at Frame 28 (mid-air)
                Vector3 preRootJump = ed.GetRootPosition(28);
                Quaternion preLHip = ed.GetJointRotation(28, SmplxJoint.L_Hip);
                Quaternion preRHip = ed.GetJointRotation(28, SmplxJoint.R_Hip);

                ed.ApplyAvatarGroundingConstraint(null, blendWeight: 1.0f);

                Vector3 postRootJump = ed.GetRootPosition(28);
                Quaternion postLHip = ed.GetJointRotation(28, SmplxJoint.L_Hip);
                Quaternion postRHip = ed.GetJointRotation(28, SmplxJoint.R_Hip);

                // Airborne frame must retain exact height and rotation without forced grounding
                float rootDiff = Vector3.Distance(preRootJump, postRootJump);
                float lHipDiff = CalcAngle(preLHip, postLHip);
                float rHipDiff = CalcAngle(preRHip, postRHip);

                bool airbornePreserved = rootDiff < 0.001f && lHipDiff < 0.001f && rHipDiff < 0.001f;

                if (airbornePreserved)
                {
                    Console.WriteLine("[PASS] Test 5: Airborne jump phase preserved height and dynamics without forced grounding.");
                    passedTests++;
                }
                else
                {
                    Console.WriteLine($"[FAIL] Test 5: Airborne frames were modified! rootDiff={rootDiff}, lHipDiff={lHipDiff}, rHipDiff={rHipDiff}");
                }
            }

            // ==========================================
            // Test 6: Auto-Detection of Contact Intervals
            // ==========================================
            totalTests++;
            {
                var ed = new EditableMotionData(mockMotion);
                ed.ClearContactIntervals();
                int countBefore = ed.GetContactIntervals().Count;

                ed.AutoDetectContactTrack(velocityThreshold: 0.55f, heightThreshold: 0.20f);
                int countAfter = ed.GetContactIntervals().Count;

                if (countBefore == 0 && countAfter >= 1)
                {
                    Console.WriteLine($"[PASS] Test 6: Auto-detection detected {countAfter} contact intervals from motion dynamics.");
                    passedTests++;
                }
                else
                {
                    Console.WriteLine($"[FAIL] Test 6: Auto-detection failed to find contact intervals. Before={countBefore}, After={countAfter}");
                }
            }

            // ==========================================
            // Test 7: Coexistence with Existing Grounding & Foot Locking
            // ==========================================
            totalTests++;
            {
                var ed = new EditableMotionData(mockMotion);

                // Test existing LockFootPosition
                bool lockOk = ed.LockFootPosition(0, 5, lockLeft: true, lockRight: false);

                // Test existing ApplyFootGrounding
                bool groundOk = ed.ApplyFootGrounding(0, 5, groundY: 0f);

                // Test existing GroundEntireClip
                bool entireOk = ed.GroundEntireClip(groundY: 0f);

                // Test new ApplyAvatarGroundingConstraint
                ed.AddContactInterval("left", 0.2f, 0.5f);
                ed.ApplyAvatarGroundingConstraint(null, blendWeight: 0.8f);

                if (lockOk && groundOk && entireOk)
                {
                    Console.WriteLine("[PASS] Test 7: Existing LockFootPosition, ApplyFootGrounding, and GroundEntireClip coexist cleanly.");
                    passedTests++;
                }
                else
                {
                    Console.WriteLine($"[FAIL] Test 7: Existing grounding methods threw or failed. lockOk={lockOk}, groundOk={groundOk}, entireOk={entireOk}");
                }
            }

            // ==========================================
            // Summary
            // ==========================================
            Console.WriteLine($"\n=== Test Summary: {passedTests}/{totalTests} PASSED ===");
            if (passedTests == totalTests)
            {
                Console.WriteLine("ALL UNITY GROUNDING TESTS PASSED SUCCESSFULLY!");
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
    }
}
