using System;
using System.Collections.Generic;
using TexMotion.Runtime.Motion;
using UnityEngine;

namespace TexMotion.Editor.Motion
{
    public struct GlitchInfo
    {
        public int Frame;
        public SmplxJoint Joint;
        public float AngularVelocityDegPerSec;
        public string Description;

        public int FrameIndex => Frame;
        public string Reason => Description;
    }

    /// <summary>
    /// Analyzes motion frames to detect sudden glitches, flips, and high-frequency jitter,
    /// and provides automated Slerp repair operations.
    /// </summary>
    public static class GlitchDetector
    {
        public const float DefaultAngularThresholdDegPerSec = 720.0f; // Rapid 720 deg/s jerk
        public const float ReversalAngleThresholdDeg = 110.0f;        // Sharp back-and-forth flip

        /// <summary>
        /// Scans all frames in the motion data and returns a list of detected glitches.
        /// </summary>
        public static List<GlitchInfo> DetectGlitches(
            EditableMotionData data,
            float thresholdDegPerSec = DefaultAngularThresholdDegPerSec,
            BodyPartMask mask = BodyPartMask.All)
        {
            var glitches = new List<GlitchInfo>();
            if (data == null || data.Frames < 3) return glitches;

            float dt = data.FrameRate > 0f ? 1.0f / data.FrameRate : 1.0f / 30.0f;
            int jointCount = SmplxJointDefinitions.JointCount;

            for (int t = 1; t < data.Frames - 1; t++)
            {
                for (int j = 0; j < jointCount; j++)
                {
                    SmplxJoint joint = (SmplxJoint)j;
                    if (!BodyPartMaskUtility.ContainsJoint(mask, joint)) continue;

                    Quaternion qPrev = data.GetJointRotation(t - 1, joint);
                    Quaternion qCurr = data.GetJointRotation(t, joint);
                    Quaternion qNext = data.GetJointRotation(t + 1, joint);

                    float anglePrev = Quaternion.Angle(qPrev, qCurr);
                    float angleNext = Quaternion.Angle(qCurr, qNext);
                    float angleSpan = Quaternion.Angle(qPrev, qNext);

                    float speed = anglePrev / dt;

                    // Condition 1: Exceeds excessive angular velocity
                    if (speed > thresholdDegPerSec)
                    {
                        glitches.Add(new GlitchInfo
                        {
                            Frame = t,
                            Joint = joint,
                            AngularVelocityDegPerSec = speed,
                            Description = $"{joint}: Extreme angular speed ({speed:F0}°/s)"
                        });
                        continue;
                    }

                    // Condition 2: Sudden reversal / flick (jumps out and immediately snaps back)
                    if (anglePrev > ReversalAngleThresholdDeg && angleNext > ReversalAngleThresholdDeg && angleSpan < 45.0f)
                    {
                        glitches.Add(new GlitchInfo
                        {
                            Frame = t,
                            Joint = joint,
                            AngularVelocityDegPerSec = speed,
                            Description = $"{joint}: Sudden isolated flip ({anglePrev:F0}° spike)"
                        });
                    }
                }
            }

            return glitches;
        }

        /// <summary>
        /// Fixes a specific glitch frame by Slerping between (frame - 1) and (frame + 1).
        /// </summary>
        public static bool FixGlitch(EditableMotionData data, int frame, BodyPartMask mask = BodyPartMask.All)
        {
            if (data == null || frame <= 0 || frame >= data.Frames - 1) return false;

            return data.SmoothFrame(frame, mask, 0.5f);
        }

        public static bool FixGlitch(EditableMotionData data, GlitchInfo glitch, BodyPartMask mask = BodyPartMask.All)
        {
            return FixGlitch(data, glitch.Frame, mask);
        }

        /// <summary>
        /// Fixes all detected glitch frames in one atomic undo step.
        /// </summary>
        public static int FixAllGlitches(EditableMotionData data, List<GlitchInfo> glitches, BodyPartMask mask = BodyPartMask.All)
        {
            if (data == null || glitches == null || glitches.Count == 0) return 0;

            data.RecordUndo($"Auto-Fix {glitches.Count} Glitches");

            var uniqueFrames = new HashSet<int>();
            foreach (var g in glitches)
            {
                if (g.Frame > 0 && g.Frame < data.Frames - 1)
                {
                    uniqueFrames.Add(g.Frame);
                }
            }

            int fixedCount = 0;
            foreach (int f in uniqueFrames)
            {
                // Perform direct Slerp without individual undo records
                int prev = f - 1;
                int next = f + 1;

                Vector3 targetRoot = Vector3.Lerp(data.GetRootPosition(prev), data.GetRootPosition(next), 0.5f);
                data.SetRootPosition(f, Vector3.Lerp(data.GetRootPosition(f), targetRoot, 0.7f));

                int jointCount = SmplxJointDefinitions.JointCount;
                for (int j = 0; j < jointCount; j++)
                {
                    SmplxJoint joint = (SmplxJoint)j;
                    if (!BodyPartMaskUtility.ContainsJoint(mask, joint)) continue;

                    Quaternion qPrev = data.GetJointRotation(prev, joint);
                    Quaternion qNext = data.GetJointRotation(next, joint);
                    if (Quaternion.Dot(qPrev, qNext) < 0f)
                    {
                        qNext = new Quaternion(-qNext.x, -qNext.y, -qNext.z, -qNext.w);
                    }
                    Quaternion smoothed = Quaternion.Slerp(qPrev, qNext, 0.5f);
                    data.SetJointRotation(f, joint, smoothed);
                }

                fixedCount++;
            }

            return fixedCount;
        }
    }
}
