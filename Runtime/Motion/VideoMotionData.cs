using System;
using UnityEngine;
using TexMotion.Runtime.Native;

namespace TexMotion.Runtime.Motion
{
    /// <summary>
    /// Represents motion extracted from video (SMPL-X22 joint hierarchy compatible),
    /// supporting timestamps, root positions, 22 joint local rotations, and frame/joint confidence metrics.
    /// Inherits from GeneratedMotionData for universal compatibility with AnimationClipBuilder and TexMotion pipelines.
    /// </summary>
    public class VideoMotionData : GeneratedMotionData
    {
        /// <summary>
        /// Exact timestamp in seconds for each frame.
        /// </summary>
        public float[] Timestamps { get; }

        /// <summary>
        /// Frame-level pose estimation confidence score [0.0, 1.0].
        /// </summary>
        public float[] Confidences { get; }

        /// <summary>
        /// Optional per-joint confidence score [frame, joint] in [0.0, 1.0].
        /// </summary>
        public float[,] JointConfidences { get; }

        /// <summary>
        /// Source video file path or identifier if available.
        /// </summary>
        /// <summary>
        /// Source video file path or identifier if available.
        /// </summary>
        public string SourceVideoPath { get; set; }

        /// <summary>
        /// Path to the preview overlay video (with MediaPipe skeletons drawn) if generated.
        /// </summary>
        public string OverlayVideoPath { get; set; }

        /// <summary>
        /// Original video width in pixels.
        /// </summary>
        public int VideoWidth { get; set; }

        /// <summary>
        /// Original video height in pixels.
        /// </summary>
        public int VideoHeight { get; set; }

        /// <summary>
        /// Original video frame rate.
        /// </summary>
        public float VideoFps { get; set; }

        /// <summary>
        /// True if frame timestamps have variable spacing (VFR or dropped frames).
        /// </summary>
        public bool HasVariableTimestamps { get; }

        /// <summary>
        /// Mean confidence score across all frames.
        /// </summary>
        public float AverageConfidence { get; }

        /// <summary>
        /// Minimum confidence score across all frames.
        /// </summary>
        public float MinConfidence { get; }

        /// <summary>
        /// Maximum confidence score across all frames.
        /// </summary>
        public float MaxConfidence { get; }

        /// <summary>
        /// Total motion duration in seconds.
        /// </summary>
        public float Duration
        {
            get
            {
                if (Frames <= 0) return 0f;
                if (Timestamps != null && Timestamps.Length > 0)
                {
                    return Timestamps[Frames - 1] - Timestamps[0];
                }
                return FrameRate > 0f ? Frames / FrameRate : 0f;
            }
        }

        public VideoMotionData(
            int frames,
            int jointCount,
            Vector3[] rootPositions,
            Quaternion[,] localRotations,
            float frameRate = 30.0f,
            float[] timestamps = null,
            float[] confidences = null,
            float[,] jointConfidences = null,
            string sourceVideoPath = null,
            int videoWidth = 0,
            int videoHeight = 0,
            float videoFps = 0f,
            string overlayVideoPath = null)
            : base(frames, jointCount, rootPositions, localRotations, frameRate)
        {
            SourceVideoPath = sourceVideoPath;
            OverlayVideoPath = overlayVideoPath;
            VideoWidth = videoWidth;
            VideoHeight = videoHeight;
            VideoFps = videoFps > 0f ? videoFps : frameRate;

            // Initialize or assign timestamps
            if (timestamps != null && timestamps.Length == frames)
            {
                Timestamps = timestamps;
                HasVariableTimestamps = CheckVariableTimestamps(timestamps, frameRate);
            }
            else
            {
                Timestamps = new float[frames];
                float dt = frameRate > 0f ? 1.0f / frameRate : 1.0f / 30.0f;
                for (int i = 0; i < frames; i++)
                {
                    Timestamps[i] = i * dt;
                }
                HasVariableTimestamps = false;
            }

            // Initialize or assign confidences
            if (confidences != null && confidences.Length == frames)
            {
                Confidences = confidences;
            }
            else
            {
                Confidences = new float[frames];
                for (int i = 0; i < frames; i++)
                {
                    Confidences[i] = 1.0f;
                }
            }

            JointConfidences = jointConfidences;

            // Compute confidence statistics
            float sum = 0f;
            float min = float.MaxValue;
            float max = float.MinValue;
            for (int i = 0; i < frames; i++)
            {
                float c = Confidences[i];
                sum += c;
                if (c < min) min = c;
                if (c > max) max = c;
            }
            AverageConfidence = frames > 0 ? sum / frames : 0f;
            MinConfidence = frames > 0 ? min : 0f;
            MaxConfidence = frames > 0 ? max : 0f;

            // Ensure quaternion hemisphere continuity across all frames and joints
            EnforceQuaternionContinuity();
        }

        public VideoMotionData(
            int frames,
            Vector3[] rootPositions,
            Quaternion[,] localRotations,
            float frameRate = 30.0f,
            float[] timestamps = null,
            float[] confidences = null,
            float[,] jointConfidences = null,
            string sourceVideoPath = null)
            : this(frames, SmplxJointDefinitions.JointCount, rootPositions, localRotations, frameRate, timestamps, confidences, jointConfidences, sourceVideoPath)
        {
        }

        /// <summary>
        /// Gets exact timestamp in seconds for a given frame index.
        /// </summary>
        public override float GetTimestamp(int frameIndex)
        {
            if (Timestamps != null && frameIndex >= 0 && frameIndex < Timestamps.Length)
            {
                return Timestamps[frameIndex];
            }
            return base.GetTimestamp(frameIndex);
        }

        /// <summary>
        /// Gets frame-level confidence score in [0.0, 1.0].
        /// </summary>
        public float GetFrameConfidence(int frameIndex)
        {
            if (Confidences != null && frameIndex >= 0 && frameIndex < Confidences.Length)
            {
                return Confidences[frameIndex];
            }
            return 1.0f;
        }

        /// <summary>
        /// Gets joint-level confidence score in [0.0, 1.0], falling back to frame confidence if joint confidences not present.
        /// </summary>
        public float GetJointConfidence(int frameIndex, int jointIndex)
        {
            if (JointConfidences != null &&
                frameIndex >= 0 && frameIndex < JointConfidences.GetLength(0) &&
                jointIndex >= 0 && jointIndex < JointConfidences.GetLength(1))
            {
                return JointConfidences[frameIndex, jointIndex];
            }
            return GetFrameConfidence(frameIndex);
        }

        /// <summary>
        /// Checks whether a frame meets a given confidence threshold.
        /// </summary>
        public bool IsFrameReliable(int frameIndex, float minConfidence = 0.5f)
        {
            return GetFrameConfidence(frameIndex) >= minConfidence;
        }

        /// <summary>
        /// Smooths out low-confidence outlier frames by interpolating from adjacent reliable frames.
        /// </summary>
        public void InterpolateOutliers(float minConfidenceThreshold = 0.3f)
        {
            if (Frames <= 2 || LocalRotations == null) return;

            for (int t = 0; t < Frames; t++)
            {
                if (Confidences[t] >= minConfidenceThreshold) continue;

                // Find previous valid frame
                int prevIdx = -1;
                for (int p = t - 1; p >= 0; p--)
                {
                    if (Confidences[p] >= minConfidenceThreshold)
                    {
                        prevIdx = p;
                        break;
                    }
                }

                // Find next valid frame
                int nextIdx = -1;
                for (int n = t + 1; n < Frames; n++)
                {
                    if (Confidences[n] >= minConfidenceThreshold)
                    {
                        nextIdx = n;
                        break;
                    }
                }

                if (prevIdx >= 0 && nextIdx >= 0)
                {
                    float alpha = (float)(t - prevIdx) / (nextIdx - prevIdx);
                    if (RootPositions != null && RootPositions.Length == Frames)
                    {
                        RootPositions[t] = LerpVector3(RootPositions[prevIdx], RootPositions[nextIdx], alpha);
                    }
                    for (int j = 0; j < JointCount; j++)
                    {
                        LocalRotations[t, j] = SlerpQuaternion(LocalRotations[prevIdx, j], LocalRotations[nextIdx, j], alpha);
                    }
                }
                else if (prevIdx >= 0)
                {
                    if (RootPositions != null && RootPositions.Length == Frames)
                    {
                        RootPositions[t] = RootPositions[prevIdx];
                    }
                    for (int j = 0; j < JointCount; j++)
                    {
                        LocalRotations[t, j] = LocalRotations[prevIdx, j];
                    }
                }
                else if (nextIdx >= 0)
                {
                    if (RootPositions != null && RootPositions.Length == Frames)
                    {
                        RootPositions[t] = RootPositions[nextIdx];
                    }
                    for (int j = 0; j < JointCount; j++)
                    {
                        LocalRotations[t, j] = LocalRotations[nextIdx, j];
                    }
                }
            }
        }

        /// <summary>
        /// Validates structural integrity of motion data arrays against frame and joint dimensions.
        /// </summary>
        public bool Validate(out string errorMessage)
        {
            if (Frames <= 0)
            {
                errorMessage = "Frame count must be greater than zero.";
                return false;
            }
            if (JointCount != SmplxJointDefinitions.JointCount)
            {
                errorMessage = $"Joint count must be {SmplxJointDefinitions.JointCount}, but was {JointCount}.";
                return false;
            }
            if (RootPositions == null || RootPositions.Length != Frames)
            {
                errorMessage = $"Root positions array length ({RootPositions?.Length ?? 0}) does not match Frames ({Frames}).";
                return false;
            }
            if (LocalRotations == null || LocalRotations.GetLength(0) != Frames || LocalRotations.GetLength(1) != JointCount)
            {
                errorMessage = "Local rotations dimensions do not match [Frames, JointCount].";
                return false;
            }
            if (Timestamps != null && Timestamps.Length != Frames)
            {
                errorMessage = "Timestamps array length does not match Frames.";
                return false;
            }
            if (Confidences != null && Confidences.Length != Frames)
            {
                errorMessage = "Confidences array length does not match Frames.";
                return false;
            }

            errorMessage = null;
            return true;
        }

        /// <summary>
        /// Creates an empty VideoMotionData instance with default allocations.
        /// </summary>
        public static VideoMotionData CreateEmpty(int frames, float frameRate = 30.0f)
        {
            int jointCount = SmplxJointDefinitions.JointCount;
            var rootPositions = new Vector3[frames];
            var localRotations = new Quaternion[frames, jointCount];
            for (int t = 0; t < frames; t++)
            {
                for (int j = 0; j < jointCount; j++)
                {
                    localRotations[t, j] = Quaternion.identity;
                }
            }
            return new VideoMotionData(frames, jointCount, rootPositions, localRotations, frameRate);
        }

        /// <summary>
        /// Wraps or converts an existing GeneratedMotionData into VideoMotionData.
        /// </summary>
        public static VideoMotionData FromGeneratedMotion(
            GeneratedMotionData source,
            float[] confidences = null,
            string sourceVideoPath = null,
            string overlayVideoPath = null)
        {
            if (source == null) return null;

            var vmd = new VideoMotionData(
                source.Frames,
                source.JointCount,
                source.RootPositions,
                source.LocalRotations,
                source.FrameRate,
                null,
                confidences,
                null,
                sourceVideoPath);
            vmd.OverlayVideoPath = overlayVideoPath;
            return vmd;
        }

        private static bool CheckVariableTimestamps(float[] timestamps, float frameRate)
        {
            if (timestamps == null || timestamps.Length <= 1) return false;
            float expectedDt = frameRate > 0f ? 1.0f / frameRate : 0.03333f;
            for (int i = 1; i < timestamps.Length; i++)
            {
                float dt = timestamps[i] - timestamps[i - 1];
                if (Math.Abs(dt - expectedDt) > 0.005f)
                {
                    return true;
                }
            }
            return false;
        }

        private static Vector3 LerpVector3(Vector3 a, Vector3 b, float t)
        {
            return new Vector3(
                a.x + (b.x - a.x) * t,
                a.y + (b.y - a.y) * t,
                a.z + (b.z - a.z) * t
            );
        }

        private static Quaternion SlerpQuaternion(Quaternion a, Quaternion b, float t)
        {
            float dot = a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
            if (dot < 0.0f)
            {
                b = new Quaternion(-b.x, -b.y, -b.z, -b.w);
                dot = -dot;
            }

            if (dot > 0.9995f)
            {
                return NormalizeQuaternion(new Quaternion(
                    a.x + (b.x - a.x) * t,
                    a.y + (b.y - a.y) * t,
                    a.z + (b.z - a.z) * t,
                    a.w + (b.w - a.w) * t
                ));
            }

            float theta = (float)Math.Acos(Math.Max(-1f, Math.Min(1f, dot)));
            float sinTheta = (float)Math.Sin(theta);
            if (Math.Abs(sinTheta) < 0.0001f)
            {
                return a;
            }

            float wa = (float)Math.Sin((1f - t) * theta) / sinTheta;
            float wb = (float)Math.Sin(t * theta) / sinTheta;

            return new Quaternion(
                wa * a.x + wb * b.x,
                wa * a.y + wb * b.y,
                wa * a.z + wb * b.z,
                wa * a.w + wb * b.w
            );
        }

        private static Quaternion NormalizeQuaternion(Quaternion q)
        {
            float mag = (float)Math.Sqrt(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w);
            if (mag > 0.00001f)
            {
                return new Quaternion(q.x / mag, q.y / mag, q.z / mag, q.w / mag);
            }
            return Quaternion.identity;
        }
    }
}
