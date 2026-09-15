using System;
using System.Collections.Generic;
using TexMotion.Runtime.Motion;
using TexMotion.Runtime.Native;
using UnityEngine;

namespace TexMotion.Editor.Motion
{
    /// <summary>
    /// Represents an editable, frame-by-frame representation of a generated or video-extracted motion.
    /// Supports individual joint rotation adjustments, root position offsets, copy/paste, pose mirroring,
    /// temporal smoothing, per-frame reset, and full Undo/Redo history.
    /// </summary>
    public class EditableMotionData
    {
        public string ClipName { get; set; } = "TexMotion_Edited";
        public int Frames { get; private set; }
        public float FrameRate { get; set; } = 30.0f;
        public float Duration => Frames > 0 && FrameRate > 0 ? (float)(Frames - 1) / FrameRate : 0f;

        public Vector3[] RootPositions { get; private set; }
        public Quaternion[,] LocalRotations { get; private set; } // [Frames, 22]
        public float[] Timestamps { get; private set; }

        public GeneratedMotionData OriginalMotionData { get; private set; }
        public VideoMotionData SourceVideoData { get; private set; }
        public Animator TargetAvatar { get; set; }

        // Generation options carried over
        public bool InPlace { get; set; } = true;
        public HandPoseType HandPose { get; set; } = HandPoseType.NaturalRelaxed;
        public FaceEmotionType FaceEmotion { get; set; } = FaceEmotionType.None;
        public float EmotionIntensity { get; set; } = 0.8f;

        // Tracking modified frames
        private readonly HashSet<int> _modifiedFrames = new HashSet<int>();

        // Original backup
        private Vector3[] _originalRootPositions;
        private Quaternion[,] _originalLocalRotations;

        // Clipboard for copy & paste
        private static Vector3? _clipboardRootPos;
        private static Quaternion[] _clipboardRotations;
        public static bool HasClipboardData => _clipboardRotations != null;

        // Undo / Redo structures
        private struct FrameStateSnapshot
        {
            public string Description;
            public Vector3[] RootPositions;
            public Quaternion[,] LocalRotations;
            public HashSet<int> ModifiedFrames;
        }

        private readonly Stack<FrameStateSnapshot> _undoStack = new Stack<FrameStateSnapshot>();
        private readonly Stack<FrameStateSnapshot> _redoStack = new Stack<FrameStateSnapshot>();
        private const int MAX_UNDO_LEVELS = 30;

        public bool CanUndo => _undoStack.Count > 0;
        public bool CanRedo => _redoStack.Count > 0;
        public int ModifiedFrameCount => _modifiedFrames.Count;

        public EditableMotionData(GeneratedMotionData sourceData, Animator targetAvatar = null, string clipName = "EditedMotion")
        {
            if (sourceData == null || sourceData.Frames <= 0)
                throw new ArgumentException("Source motion data is null or empty.");

            OriginalMotionData = sourceData;
            SourceVideoData = sourceData as VideoMotionData;
            TargetAvatar = targetAvatar;
            ClipName = string.IsNullOrEmpty(clipName) ? "TexMotion_Edited" : clipName;

            Frames = sourceData.Frames;
            FrameRate = sourceData.FrameRate > 0f ? sourceData.FrameRate : 30.0f;

            int jointCount = SmplxJointDefinitions.JointCount;

            RootPositions = new Vector3[Frames];
            LocalRotations = new Quaternion[Frames, jointCount];
            Timestamps = new float[Frames];

            _originalRootPositions = new Vector3[Frames];
            _originalLocalRotations = new Quaternion[Frames, jointCount];

            for (int t = 0; t < Frames; t++)
            {
                Timestamps[t] = sourceData.GetTimestamp(t);

                Vector3 pos = (sourceData.RootPositions != null && sourceData.RootPositions.Length > t)
                    ? sourceData.RootPositions[t]
                    : Vector3.zero;
                RootPositions[t] = pos;
                _originalRootPositions[t] = pos;

                for (int j = 0; j < jointCount; j++)
                {
                    Quaternion rot = Quaternion.identity;
                    if (sourceData.LocalRotations != null &&
                        t < sourceData.LocalRotations.GetLength(0) &&
                        j < sourceData.LocalRotations.GetLength(1))
                    {
                        rot = sourceData.LocalRotations[t, j];
                        if (rot.x == 0 && rot.y == 0 && rot.z == 0 && rot.w == 0)
                        {
                            rot = Quaternion.identity;
                        }
                    }
                    LocalRotations[t, j] = rot;
                    _originalLocalRotations[t, j] = rot;
                }
            }
        }

        /// <summary>
        /// Creates an EditableMotionData instance by sampling an existing AnimationClip on a Humanoid avatar.
        /// </summary>
        public static EditableMotionData FromAnimationClip(AnimationClip clip, Animator targetAvatar)
        {
            if (clip == null || targetAvatar == null || !targetAvatar.isHuman) return null;

            float fps = clip.frameRate > 0 ? clip.frameRate : 30.0f;
            float duration = clip.length;
            int frames = Mathf.Max(2, Mathf.RoundToInt(duration * fps));
            int jointCount = SmplxJointDefinitions.JointCount;

            var rootPositions = new Vector3[frames];
            var localRotations = new Quaternion[frames, jointCount];

            GameObject tempAvatar = UnityEngine.Object.Instantiate(targetAvatar.gameObject, Vector3.zero, Quaternion.identity);
            tempAvatar.hideFlags = HideFlags.HideAndDontSave;

            try
            {
                var anim = tempAvatar.GetComponent<Animator>();
                anim.enabled = false;

                var boneMap = new Dictionary<SmplxJoint, Transform>();
                var initialRotations = new Dictionary<SmplxJoint, Quaternion>();

                foreach (var kvp in SmplxJointDefinitions.SmplxToHumanBodyBones)
                {
                    Transform bone = anim.GetBoneTransform(kvp.Value);
                    if (bone != null)
                    {
                        boneMap[kvp.Key] = bone;
                        initialRotations[kvp.Key] = bone.localRotation;
                    }
                }

                Transform hips = anim.GetBoneTransform(HumanBodyBones.Hips);

                for (int t = 0; t < frames; t++)
                {
                    float time = (float)t / fps;
                    clip.SampleAnimation(tempAvatar, time);

                    if (hips != null)
                    {
                        rootPositions[t] = hips.localPosition;
                    }

                    foreach (var kvp in boneMap)
                    {
                        SmplxJoint joint = kvp.Key;
                        Transform bone = kvp.Value;
                        Quaternion rest = initialRotations[joint];
                        localRotations[t, (int)joint] = AnimationClipBuilder.ConvertUnityRotationToSmpl(bone.localRotation, joint, rest);
                    }
                }
            }
            finally
            {
                UnityEngine.Object.DestroyImmediate(tempAvatar);
            }

            var motionData = new GeneratedMotionData(frames, jointCount, rootPositions, localRotations, fps);
            return new EditableMotionData(motionData, targetAvatar, clip.name);
        }

        #region Undo / Redo

        public void RecordUndo(string description = "Edit Frame")
        {
            _redoStack.Clear();

            var snapshot = new FrameStateSnapshot
            {
                Description = description,
                RootPositions = (Vector3[])RootPositions.Clone(),
                LocalRotations = (Quaternion[,])LocalRotations.Clone(),
                ModifiedFrames = new HashSet<int>(_modifiedFrames)
            };

            _undoStack.Push(snapshot);
            if (_undoStack.Count > MAX_UNDO_LEVELS)
            {
                // Trim oldest
                var array = _undoStack.ToArray();
                _undoStack.Clear();
                for (int i = MAX_UNDO_LEVELS - 1; i >= 0; i--)
                {
                    _undoStack.Push(array[i]);
                }
            }
        }

        public bool Undo()
        {
            if (!CanUndo) return false;

            var currentState = new FrameStateSnapshot
            {
                Description = "Current",
                RootPositions = (Vector3[])RootPositions.Clone(),
                LocalRotations = (Quaternion[,])LocalRotations.Clone(),
                ModifiedFrames = new HashSet<int>(_modifiedFrames)
            };
            _redoStack.Push(currentState);

            var previous = _undoStack.Pop();
            RestoreFromSnapshot(previous);
            return true;
        }

        public bool Redo()
        {
            if (!CanRedo) return false;

            var currentState = new FrameStateSnapshot
            {
                Description = "Current",
                RootPositions = (Vector3[])RootPositions.Clone(),
                LocalRotations = (Quaternion[,])LocalRotations.Clone(),
                ModifiedFrames = new HashSet<int>(_modifiedFrames)
            };
            _undoStack.Push(currentState);

            var next = _redoStack.Pop();
            RestoreFromSnapshot(next);
            return true;
        }

        private void RestoreFromSnapshot(FrameStateSnapshot snapshot)
        {
            RootPositions = (Vector3[])snapshot.RootPositions.Clone();
            LocalRotations = (Quaternion[,])snapshot.LocalRotations.Clone();
            _modifiedFrames.Clear();
            foreach (var f in snapshot.ModifiedFrames)
            {
                _modifiedFrames.Add(f);
            }
        }

        #endregion

        #region Frame Pose Query & Modification

        public bool IsFrameModified(int frame) => _modifiedFrames.Contains(frame);

        public Vector3 GetRootPosition(int frame)
        {
            if (frame < 0 || frame >= Frames) return Vector3.zero;
            return RootPositions[frame];
        }

        public void SetRootPosition(int frame, Vector3 position)
        {
            if (frame < 0 || frame >= Frames) return;
            if (RootPositions[frame] != position)
            {
                RootPositions[frame] = position;
                _modifiedFrames.Add(frame);
            }
        }

        public Quaternion GetJointRotation(int frame, SmplxJoint joint)
        {
            if (frame < 0 || frame >= Frames) return Quaternion.identity;
            int j = (int)joint;
            if (j < 0 || j >= SmplxJointDefinitions.JointCount) return Quaternion.identity;
            return LocalRotations[frame, j];
        }

        public Vector3 GetJointEuler(int frame, SmplxJoint joint)
        {
            Quaternion rot = GetJointRotation(frame, joint);
            Vector3 euler = rot.eulerAngles;
            // Normalize to -180 .. +180 range
            return new Vector3(
                NormalizeAngle(euler.x),
                NormalizeAngle(euler.y),
                NormalizeAngle(euler.z)
            );
        }

        public void SetJointRotation(int frame, SmplxJoint joint, Quaternion rotation)
        {
            if (frame < 0 || frame >= Frames) return;
            int j = (int)joint;
            if (j < 0 || j >= SmplxJointDefinitions.JointCount) return;

            if (LocalRotations[frame, j] != rotation)
            {
                LocalRotations[frame, j] = rotation;
                _modifiedFrames.Add(frame);
            }
        }

        public void SetJointEuler(int frame, SmplxJoint joint, Vector3 euler)
        {
            Quaternion rot = Quaternion.Euler(euler.x, euler.y, euler.z);
            SetJointRotation(frame, joint, rot);
        }

        private static float NormalizeAngle(float angle)
        {
            while (angle > 180f) angle -= 360f;
            while (angle < -180f) angle += 360f;
            return angle;
        }

        #endregion

        #region Pose Editing Operations

        /// <summary>
        /// Copies the entire pose (root position + all 22 joint rotations) of the specified frame to clipboard.
        /// </summary>
        public void CopyFramePose(int frame)
        {
            if (frame < 0 || frame >= Frames) return;

            _clipboardRootPos = RootPositions[frame];
            int jointCount = SmplxJointDefinitions.JointCount;
            _clipboardRotations = new Quaternion[jointCount];

            for (int j = 0; j < jointCount; j++)
            {
                _clipboardRotations[j] = LocalRotations[frame, j];
            }
        }

        /// <summary>
        /// Pastes clipboard pose to the specified frame.
        /// </summary>
        public bool PasteFramePose(int frame, bool includeRootPosition = true)
        {
            if (!HasClipboardData || frame < 0 || frame >= Frames) return false;

            RecordUndo($"Paste Pose to Frame {frame}");

            if (includeRootPosition && _clipboardRootPos.HasValue)
            {
                RootPositions[frame] = _clipboardRootPos.Value;
            }

            int jointCount = SmplxJointDefinitions.JointCount;
            for (int j = 0; j < jointCount; j++)
            {
                LocalRotations[frame, j] = _clipboardRotations[j];
            }

            _modifiedFrames.Add(frame);
            return true;
        }

        /// <summary>
        /// Resets the specified frame to its original state before any edits.
        /// </summary>
        public void ResetFrameToOriginal(int frame)
        {
            if (frame < 0 || frame >= Frames) return;

            RecordUndo($"Reset Frame {frame} to Original");

            RootPositions[frame] = _originalRootPositions[frame];
            int jointCount = SmplxJointDefinitions.JointCount;
            for (int j = 0; j < jointCount; j++)
            {
                LocalRotations[frame, j] = _originalLocalRotations[frame, j];
            }

            _modifiedFrames.Remove(frame);
        }

        /// <summary>
        /// Resets a specific joint in a frame to its original rotation.
        /// </summary>
        public void ResetJointToOriginal(int frame, SmplxJoint joint)
        {
            if (frame < 0 || frame >= Frames) return;
            int j = (int)joint;
            if (j < 0 || j >= SmplxJointDefinitions.JointCount) return;

            RecordUndo($"Reset {joint} on Frame {frame}");
            LocalRotations[frame, j] = _originalLocalRotations[frame, j];
        }

        /// <summary>
        /// Resets all frames back to original generation output.
        /// </summary>
        public void ResetAllFramesToOriginal()
        {
            RecordUndo("Reset All Frames to Original");

            int jointCount = SmplxJointDefinitions.JointCount;
            for (int t = 0; t < Frames; t++)
            {
                RootPositions[t] = _originalRootPositions[t];
                for (int j = 0; j < jointCount; j++)
                {
                    LocalRotations[t, j] = _originalLocalRotations[t, j];
                }
            }
            _modifiedFrames.Clear();
        }

        /// <summary>
        /// Smooths the specified frame by interpolating between (frame - 1) and (frame + 1).
        /// Useful for fixing outlier poses or jitter in video-extracted frames.
        /// </summary>
        public bool SmoothFrame(int frame, float blendWeight = 0.5f)
        {
            if (frame <= 0 || frame >= Frames - 1) return false;

            RecordUndo($"Smooth Frame {frame}");

            int prev = frame - 1;
            int next = frame + 1;

            // Blend Root Position
            Vector3 targetRoot = Vector3.Lerp(RootPositions[prev], RootPositions[next], blendWeight);
            RootPositions[frame] = Vector3.Lerp(RootPositions[frame], targetRoot, 0.7f);

            // Blend 22 Joints
            int jointCount = SmplxJointDefinitions.JointCount;
            for (int j = 0; j < jointCount; j++)
            {
                Quaternion qPrev = LocalRotations[prev, j];
                Quaternion qNext = LocalRotations[next, j];
                if (Quaternion.Dot(qPrev, qNext) < 0f)
                {
                    qNext = new Quaternion(-qNext.x, -qNext.y, -qNext.z, -qNext.w);
                }
                Quaternion interpolated = Quaternion.Slerp(qPrev, qNext, blendWeight);

                Quaternion curr = LocalRotations[frame, j];
                if (Quaternion.Dot(curr, interpolated) < 0f)
                {
                    interpolated = new Quaternion(-interpolated.x, -interpolated.y, -interpolated.z, -interpolated.w);
                }
                LocalRotations[frame, j] = Quaternion.Slerp(curr, interpolated, 0.75f);
            }

            _modifiedFrames.Add(frame);
            return true;
        }

        /// <summary>
        /// Mirrors the pose of the specified frame across the sagittal (X) plane.
        /// Swaps left and right limbs and mirrors central spine rotations.
        /// </summary>
        public void MirrorFrame(int frame)
        {
            if (frame < 0 || frame >= Frames) return;

            RecordUndo($"Mirror Frame {frame}");

            // Mirror Root Position X
            Vector3 root = RootPositions[frame];
            RootPositions[frame] = new Vector3(-root.x, root.y, root.z);

            // Joint mapping pairs for left/right swap
            var swapPairs = new (SmplxJoint Left, SmplxJoint Right)[]
            {
                (SmplxJoint.L_Hip, SmplxJoint.R_Hip),
                (SmplxJoint.L_Knee, SmplxJoint.R_Knee),
                (SmplxJoint.L_Ankle, SmplxJoint.R_Ankle),
                (SmplxJoint.L_Foot, SmplxJoint.R_Foot),
                (SmplxJoint.L_Collar, SmplxJoint.R_Collar),
                (SmplxJoint.L_Shoulder, SmplxJoint.R_Shoulder),
                (SmplxJoint.L_Elbow, SmplxJoint.R_Elbow),
                (SmplxJoint.L_Wrist, SmplxJoint.R_Wrist)
            };

            // Temporary buffer for current frame rotations
            int jointCount = SmplxJointDefinitions.JointCount;
            Quaternion[] temp = new Quaternion[jointCount];
            for (int j = 0; j < jointCount; j++)
            {
                temp[j] = LocalRotations[frame, j];
            }

            // Swap and mirror limb pairs
            foreach (var (left, right) in swapPairs)
            {
                int lIdx = (int)left;
                int rIdx = (int)right;

                Quaternion lRot = temp[lIdx];
                Quaternion rRot = temp[rIdx];

                // Mirror quaternion: (x, -y, -z, w)
                LocalRotations[frame, lIdx] = new Quaternion(rRot.x, -rRot.y, -rRot.z, rRot.w);
                LocalRotations[frame, rIdx] = new Quaternion(lRot.x, -lRot.y, -lRot.z, lRot.w);
            }

            // Central joints: Pelvis, Spine1, Spine2, Spine3, Neck, Head
            SmplxJoint[] centralJoints = new SmplxJoint[]
            {
                SmplxJoint.Pelvis,
                SmplxJoint.Spine1,
                SmplxJoint.Spine2,
                SmplxJoint.Spine3,
                SmplxJoint.Neck,
                SmplxJoint.Head
            };

            foreach (var c in centralJoints)
            {
                int idx = (int)c;
                Quaternion q = temp[idx];
                LocalRotations[frame, idx] = new Quaternion(q.x, -q.y, -q.z, q.w);
            }

            _modifiedFrames.Add(frame);
        }

        /// <summary>
        /// Sets the frame pose to a neutral T-Pose.
        /// </summary>
        public void SetFrameToTPose(int frame)
        {
            if (frame < 0 || frame >= Frames) return;

            RecordUndo($"Set Frame {frame} to T-Pose");

            int jointCount = SmplxJointDefinitions.JointCount;
            for (int j = 0; j < jointCount; j++)
            {
                LocalRotations[frame, j] = Quaternion.identity;
            }

            _modifiedFrames.Add(frame);
        }

        #endregion

        #region Export & Build

        /// <summary>
        /// Creates an updated GeneratedMotionData or VideoMotionData containing all modifications.
        /// </summary>
        public GeneratedMotionData ToGeneratedMotionData()
        {
            if (SourceVideoData != null)
            {
                return new VideoMotionData(
                    Frames,
                    SmplxJointDefinitions.JointCount,
                    (Vector3[])RootPositions.Clone(),
                    (Quaternion[,])LocalRotations.Clone(),
                    FrameRate,
                    SourceVideoData.Timestamps != null ? (float[])SourceVideoData.Timestamps.Clone() : (float[])Timestamps.Clone(),
                    SourceVideoData.Confidences != null ? (float[])SourceVideoData.Confidences.Clone() : null,
                    SourceVideoData.JointConfidences != null ? (float[,])SourceVideoData.JointConfidences.Clone() : null,
                    SourceVideoData.SourceVideoPath,
                    SourceVideoData.VideoWidth,
                    SourceVideoData.VideoHeight,
                    SourceVideoData.VideoFps,
                    SourceVideoData.OverlayVideoPath
                );
            }

            return new GeneratedMotionData(
                Frames,
                SmplxJointDefinitions.JointCount,
                (Vector3[])RootPositions.Clone(),
                (Quaternion[,])LocalRotations.Clone(),
                FrameRate
            );
        }

        /// <summary>
        /// Builds an AnimationClip directly from this edited motion data using AnimationClipBuilder.
        /// </summary>
        public AnimationClip BuildAnimationClip(AnimationBuildOptions options)
        {
            GeneratedMotionData data = ToGeneratedMotionData();
            return AnimationClipBuilder.BuildAnimationClip(data, options);
        }

        #endregion
    }
}
