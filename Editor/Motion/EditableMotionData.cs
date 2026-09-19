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
        /// Pastes clipboard pose to the specified frame with body part masking.
        /// </summary>
        public bool PasteFramePose(int frame, bool includeRootPosition = true)
        {
            return PasteFramePose(frame, BodyPartMask.All, includeRootPosition);
        }

        public bool PasteFramePose(int frame, BodyPartMask mask, bool includeRootPosition = true)
        {
            if (!HasClipboardData || frame < 0 || frame >= Frames) return false;

            RecordUndo($"Paste Pose to Frame {frame} ({mask})");

            if (includeRootPosition && _clipboardRootPos.HasValue && (mask & BodyPartMask.Pelvis) != 0)
            {
                RootPositions[frame] = _clipboardRootPos.Value;
            }

            int jointCount = SmplxJointDefinitions.JointCount;
            for (int j = 0; j < jointCount; j++)
            {
                SmplxJoint joint = (SmplxJoint)j;
                if (!BodyPartMaskUtility.ContainsJoint(mask, joint)) continue;

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
            ResetFrameToOriginal(frame, BodyPartMask.All);
        }

        public void ResetFrameToOriginal(int frame, BodyPartMask mask)
        {
            if (frame < 0 || frame >= Frames) return;

            RecordUndo($"Reset Frame {frame} ({mask}) to Original");

            if ((mask & BodyPartMask.Pelvis) != 0)
            {
                RootPositions[frame] = _originalRootPositions[frame];
            }

            int jointCount = SmplxJointDefinitions.JointCount;
            for (int j = 0; j < jointCount; j++)
            {
                SmplxJoint joint = (SmplxJoint)j;
                if (!BodyPartMaskUtility.ContainsJoint(mask, joint)) continue;

                LocalRotations[frame, j] = _originalLocalRotations[frame, j];
            }

            if (mask == BodyPartMask.All)
            {
                _modifiedFrames.Remove(frame);
            }
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
            return SmoothFrame(frame, BodyPartMask.All, blendWeight);
        }

        public bool SmoothFrame(int frame, BodyPartMask mask, float blendWeight = 0.5f)
        {
            if (frame <= 0 || frame >= Frames - 1) return false;

            RecordUndo($"Smooth Frame {frame} ({mask})");

            int prev = frame - 1;
            int next = frame + 1;

            // Blend Root Position
            if ((mask & BodyPartMask.Pelvis) != 0)
            {
                Vector3 targetRoot = Vector3.Lerp(RootPositions[prev], RootPositions[next], blendWeight);
                RootPositions[frame] = Vector3.Lerp(RootPositions[frame], targetRoot, 0.7f);
            }

            // Blend 22 Joints
            int jointCount = SmplxJointDefinitions.JointCount;
            for (int j = 0; j < jointCount; j++)
            {
                SmplxJoint joint = (SmplxJoint)j;
                if (!BodyPartMaskUtility.ContainsJoint(mask, joint)) continue;

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

        #region Advanced Pose Operations

        /// <summary>
        /// Interpolates / tweens a range of frames between startFrame and endFrame using the specified easing curve and body mask.
        /// </summary>
        public bool TweenRange(
            int startFrame,
            int endFrame,
            EasingType easing = EasingType.EaseInOut,
            BodyPartMask mask = BodyPartMask.All,
            bool includeRoot = true)
        {
            if (startFrame < 0 || endFrame >= Frames || endFrame - startFrame < 2)
                return false;

            RecordUndo($"Tween Range [{startFrame}..{endFrame}] ({easing}, {mask})");

            int span = endFrame - startFrame;
            int jointCount = SmplxJointDefinitions.JointCount;

            Vector3 startRoot = RootPositions[startFrame];
            Vector3 endRoot = RootPositions[endFrame];

            for (int t = startFrame + 1; t < endFrame; t++)
            {
                float linearT = (float)(t - startFrame) / span;
                float easedT = BodyPartMaskUtility.EvaluateEasing(easing, linearT);

                if (includeRoot && (mask & BodyPartMask.Pelvis) != 0)
                {
                    RootPositions[t] = Vector3.Lerp(startRoot, endRoot, easedT);
                }

                for (int j = 0; j < jointCount; j++)
                {
                    SmplxJoint joint = (SmplxJoint)j;
                    if (!BodyPartMaskUtility.ContainsJoint(mask, joint)) continue;

                    Quaternion qStart = LocalRotations[startFrame, j];
                    Quaternion qEnd = LocalRotations[endFrame, j];

                    if (Quaternion.Dot(qStart, qEnd) < 0f)
                    {
                        qEnd = new Quaternion(-qEnd.x, -qEnd.y, -qEnd.z, -qEnd.w);
                    }

                    LocalRotations[t, j] = Quaternion.Slerp(qStart, qEnd, easedT);
                }

                _modifiedFrames.Add(t);
            }

            return true;
        }

        /// <summary>
        /// Smoothly transitions the trailing frames back to frame 0 for seamless loop playback.
        /// </summary>
        public bool BlendLoopBoundary(
            int blendFrames = 10,
            BodyPartMask mask = BodyPartMask.All,
            bool matchRoot = true)
        {
            if (Frames < 4) return false;

            blendFrames = Mathf.Clamp(blendFrames, 2, Frames / 2);
            RecordUndo($"Loop Boundary Blend ({blendFrames} frames, {mask})");

            int startBlend = Frames - blendFrames;
            int jointCount = SmplxJointDefinitions.JointCount;

            Vector3 root0 = RootPositions[0];

            for (int t = startBlend; t < Frames; t++)
            {
                int k = t - startBlend;
                float linearT = (float)(k + 1) / (blendFrames + 1);
                float weight = BodyPartMaskUtility.EvaluateEasing(EasingType.SmoothStep, linearT);

                if (matchRoot && (mask & BodyPartMask.Pelvis) != 0)
                {
                    // If InPlace, blend X and Z, and maintain continuous height Y
                    Vector3 currentRoot = RootPositions[t];
                    Vector3 targetRoot = new Vector3(root0.x, root0.y, InPlace ? root0.z : currentRoot.z);
                    RootPositions[t] = Vector3.Lerp(currentRoot, targetRoot, weight);
                }

                for (int j = 0; j < jointCount; j++)
                {
                    SmplxJoint joint = (SmplxJoint)j;
                    if (!BodyPartMaskUtility.ContainsJoint(mask, joint)) continue;

                    Quaternion qCurr = LocalRotations[t, j];
                    Quaternion q0 = LocalRotations[0, j];

                    if (Quaternion.Dot(qCurr, q0) < 0f)
                    {
                        q0 = new Quaternion(-q0.x, -q0.y, -q0.z, -q0.w);
                    }

                    LocalRotations[t, j] = Quaternion.Slerp(qCurr, q0, weight);
                }

                _modifiedFrames.Add(t);
            }

            return true;
        }

        /// <summary>
        /// Applies relative additive offsets to root position and joint euler rotations across a frame range,
        /// with optional edge falloff fading.
        /// </summary>
        public bool ApplyRangeOffset(
            int startFrame,
            int endFrame,
            Vector3 rootOffset,
            Dictionary<SmplxJoint, Vector3> eulerOffsets,
            bool fadeEdges = true)
        {
            if (startFrame < 0 || endFrame >= Frames || startFrame > endFrame) return false;

            RecordUndo($"Apply Range Offset [{startFrame}..{endFrame}]");

            int span = Mathf.Max(1, endFrame - startFrame);
            float fadeLength = fadeEdges ? Mathf.Max(1f, span * 0.15f) : 0f;

            for (int t = startFrame; t <= endFrame; t++)
            {
                float weight = 1.0f;
                if (fadeEdges && span > 2)
                {
                    float distFromStart = t - startFrame;
                    float distFromEnd = endFrame - t;
                    float minDist = Mathf.Min(distFromStart, distFromEnd);
                    if (minDist < fadeLength)
                    {
                        weight = BodyPartMaskUtility.EvaluateEasing(EasingType.SmoothStep, minDist / fadeLength);
                    }
                }

                if (rootOffset != Vector3.zero)
                {
                    RootPositions[t] += rootOffset * weight;
                }

                if (eulerOffsets != null && eulerOffsets.Count > 0)
                {
                    foreach (var kvp in eulerOffsets)
                    {
                        SmplxJoint joint = kvp.Key;
                        Vector3 deltaEuler = kvp.Value * weight;
                        Quaternion curRot = LocalRotations[t, (int)joint];
                        Quaternion deltaRot = Quaternion.Euler(deltaEuler.x, deltaEuler.y, deltaEuler.z);
                        LocalRotations[t, (int)joint] = curRot * deltaRot;
                    }
                }

                _modifiedFrames.Add(t);
            }

            return true;
        }

        public bool ApplyRangeOffset(
            int startFrame,
            int endFrame,
            Vector3 rootOffset,
            Vector3 armEulerOffset,
            bool fadeEdges = true)
        {
            var eulers = new Dictionary<SmplxJoint, Vector3>();
            if (armEulerOffset != Vector3.zero)
            {
                eulers[SmplxJoint.L_Shoulder] = new Vector3(armEulerOffset.x, armEulerOffset.y, armEulerOffset.z);
                eulers[SmplxJoint.R_Shoulder] = new Vector3(armEulerOffset.x, armEulerOffset.y, -armEulerOffset.z);
            }
            return ApplyRangeOffset(startFrame, endFrame, rootOffset, eulers, fadeEdges);
        }

        /// <summary>
        /// Retimes (stretches or compresses) the specified frame range [startFrame, endFrame]
        /// into a new duration (newFrameCount), resampling poses via Slerp.
        /// </summary>
        public bool RetimeRange(int startFrame, int endFrame, int newFrameCount)
        {
            if (startFrame < 0 || endFrame >= Frames || startFrame >= endFrame || newFrameCount < 2)
                return false;

            int oldRangeCount = endFrame - startFrame + 1;
            int newTotalFrames = Frames - oldRangeCount + newFrameCount;
            if (newTotalFrames < 3) return false;

            RecordUndo($"Retime Range [{startFrame}..{endFrame}] ({oldRangeCount} -> {newFrameCount} frames)");

            int jointCount = SmplxJointDefinitions.JointCount;
            var newRoots = new Vector3[newTotalFrames];
            var newRotations = new Quaternion[newTotalFrames, jointCount];
            var newTimestamps = new float[newTotalFrames];

            // 1. Copy frames before startFrame
            for (int i = 0; i < startFrame; i++)
            {
                newRoots[i] = RootPositions[i];
                for (int j = 0; j < jointCount; j++) newRotations[i, j] = LocalRotations[i, j];
                newTimestamps[i] = (float)i / FrameRate;
            }

            // 2. Resample the retimed range
            for (int k = 0; k < newFrameCount; k++)
            {
                int targetIdx = startFrame + k;
                float progress = (float)k / (newFrameCount - 1);
                float sampleFrameExact = startFrame + (progress * (oldRangeCount - 1));

                int f0 = Mathf.FloorToInt(sampleFrameExact);
                int f1 = Mathf.Min(f0 + 1, endFrame);
                float frac = sampleFrameExact - f0;

                newRoots[targetIdx] = Vector3.Lerp(RootPositions[f0], RootPositions[f1], frac);

                for (int j = 0; j < jointCount; j++)
                {
                    Quaternion q0 = LocalRotations[f0, j];
                    Quaternion q1 = LocalRotations[f1, j];
                    if (Quaternion.Dot(q0, q1) < 0f)
                    {
                        q1 = new Quaternion(-q1.x, -q1.y, -q1.z, -q1.w);
                    }
                    newRotations[targetIdx, j] = Quaternion.Slerp(q0, q1, frac);
                }

                newTimestamps[targetIdx] = (float)targetIdx / FrameRate;
            }

            // 3. Copy frames after endFrame
            int tailOffset = newTotalFrames - (Frames - 1 - endFrame);
            for (int i = endFrame + 1; i < Frames; i++)
            {
                int targetIdx = startFrame + newFrameCount + (i - (endFrame + 1));
                newRoots[targetIdx] = RootPositions[i];
                for (int j = 0; j < jointCount; j++) newRotations[targetIdx, j] = LocalRotations[i, j];
                newTimestamps[targetIdx] = (float)targetIdx / FrameRate;
            }

            // Apply new arrays
            Frames = newTotalFrames;
            RootPositions = newRoots;
            LocalRotations = newRotations;
            Timestamps = newTimestamps;

            _modifiedFrames.Clear();
            for (int i = startFrame; i < startFrame + newFrameCount; i++)
            {
                _modifiedFrames.Add(i);
            }

            return true;
        }

        /// <summary>
        /// Prevents upper arms from penetrating the avatar's chest/torso by clamping the minimum armpit opening angle.
        /// </summary>
        public bool ApplyArmpitPenetrationLimiter(int startFrame, int endFrame, float minArmpitAngleDeg = 20.0f)
        {
            if (startFrame < 0 || endFrame >= Frames || startFrame > endFrame) return false;

            RecordUndo($"Armpit Penetration Limiter ({minArmpitAngleDeg}°)");

            for (int t = startFrame; t <= endFrame; t++)
            {
                // Left Shoulder: clamp roll/adduction away from negative Y / positive X
                Vector3 lEuler = GetJointEuler(t, SmplxJoint.L_Shoulder);
                if (Mathf.Abs(lEuler.z) < minArmpitAngleDeg)
                {
                    lEuler.z = Mathf.Sign(lEuler.z == 0 ? -1f : lEuler.z) * minArmpitAngleDeg;
                    SetJointEuler(t, SmplxJoint.L_Shoulder, lEuler);
                }

                // Right Shoulder
                Vector3 rEuler = GetJointEuler(t, SmplxJoint.R_Shoulder);
                if (Mathf.Abs(rEuler.z) < minArmpitAngleDeg)
                {
                    rEuler.z = Mathf.Sign(rEuler.z == 0 ? 1f : rEuler.z) * minArmpitAngleDeg;
                    SetJointEuler(t, SmplxJoint.R_Shoulder, rEuler);
                }

                _modifiedFrames.Add(t);
            }

            return true;
        }

        /// <summary>
        /// <summary>
        /// Clamps or snaps foot heights to the floor plane (groundY).
        /// If snapFloating is true, feet above the ground are lowered to contact groundY.
        /// </summary>
        public bool ApplyFootGrounding(int startFrame, int endFrame, float groundY = 0f, float ankleGroundOffset = 0.05f, bool snapFloating = true)
        {
            if (startFrame < 0 || endFrame >= Frames || startFrame > endFrame) return false;

            RecordUndo($"Foot Grounding [F{startFrame}..F{endFrame}] (Ground={groundY:F2}m)");

            int jointCount = SmplxJointDefinitions.JointCount;
            Quaternion[] locals = new Quaternion[jointCount];

            for (int t = startFrame; t <= endFrame; t++)
            {
                Vector3 rootPos = RootPositions[t];
                for (int j = 0; j < jointCount; j++) locals[j] = LocalRotations[t, j];

                Vector3[] fkPos = MotionIkUtility.ComputeForwardKinematics(rootPos, locals);

                float lFootY = fkPos[(int)SmplxJoint.L_Foot].y;
                float rFootY = fkPos[(int)SmplxJoint.R_Foot].y;
                float lAnkY = fkPos[(int)SmplxJoint.L_Ankle].y - ankleGroundOffset;
                float rAnkY = fkPos[(int)SmplxJoint.R_Ankle].y - ankleGroundOffset;

                float lowestPoint = Mathf.Min(Mathf.Min(lFootY, rFootY), Mathf.Min(lAnkY, rAnkY));
                float diff = groundY - lowestPoint;

                if (diff > 0.0005f || (snapFloating && Mathf.Abs(diff) > 0.0005f))
                {
                    RootPositions[t] = new Vector3(rootPos.x, rootPos.y + diff, rootPos.z);
                    _modifiedFrames.Add(t);
                }
            }

            return true;
        }

        public bool ApplyFootGrounding(int frame, float groundY = 0f, float ankleGroundOffset = 0.05f, bool snapFloating = true)
        {
            return ApplyFootGrounding(frame, frame, groundY, ankleGroundOffset, snapFloating);
        }

        /// <summary>
        /// Offsets the entire motion clip vertically so the lowest foot contact across all frames lands exactly on groundY.
        /// Preserves all relative jumping and running dynamics while anchoring the lowest point to the ground.
        /// </summary>
        public bool GroundEntireClip(float groundY = 0f, float ankleGroundOffset = 0.05f)
        {
            if (Frames <= 0) return false;

            RecordUndo($"Ground Entire Clip (Ground={groundY:F2}m)");

            int jointCount = SmplxJointDefinitions.JointCount;
            Quaternion[] locals = new Quaternion[jointCount];
            float globalLowest = float.MaxValue;

            for (int t = 0; t < Frames; t++)
            {
                Vector3 rootPos = RootPositions[t];
                for (int j = 0; j < jointCount; j++) locals[j] = LocalRotations[t, j];

                Vector3[] fkPos = MotionIkUtility.ComputeForwardKinematics(rootPos, locals);
                float lFootY = fkPos[(int)SmplxJoint.L_Foot].y;
                float rFootY = fkPos[(int)SmplxJoint.R_Foot].y;
                float lAnkY = fkPos[(int)SmplxJoint.L_Ankle].y - ankleGroundOffset;
                float rAnkY = fkPos[(int)SmplxJoint.R_Ankle].y - ankleGroundOffset;

                float frameLowest = Mathf.Min(Mathf.Min(lFootY, rFootY), Mathf.Min(lAnkY, rAnkY));
                if (frameLowest < globalLowest) globalLowest = frameLowest;
            }

            if (globalLowest != float.MaxValue)
            {
                float verticalShift = groundY - globalLowest;
                if (Mathf.Abs(verticalShift) > 0.0005f)
                {
                    for (int t = 0; t < Frames; t++)
                    {
                        RootPositions[t] = new Vector3(RootPositions[t].x, RootPositions[t].y + verticalShift, RootPositions[t].z);
                        _modifiedFrames.Add(t);
                    }
                }
            }

            return true;
        }

        /// <summary>
        /// Locks foot positions to prevent sliding during a contact range by solving inverse kinematics.
        /// </summary>
        public bool LockFootPosition(int startFrame, int endFrame, bool lockLeft = true, bool lockRight = false)
        {
            if (startFrame < 0 || endFrame >= Frames || startFrame >= endFrame) return false;

            RecordUndo($"Lock Foot Position [F{startFrame}..F{endFrame}]");

            int jointCount = SmplxJointDefinitions.JointCount;
            Quaternion[] baseLocals = new Quaternion[jointCount];
            for (int j = 0; j < jointCount; j++) baseLocals[j] = LocalRotations[startFrame, j];

            Vector3[] baseFk = MotionIkUtility.ComputeForwardKinematics(RootPositions[startFrame], baseLocals);
            Vector3 targetLAnkle = baseFk[(int)SmplxJoint.L_Ankle];
            Vector3 targetRAnkle = baseFk[(int)SmplxJoint.R_Ankle];

            for (int t = startFrame + 1; t <= endFrame; t++)
            {
                if (lockLeft)
                {
                    MotionIkUtility.ApplyLimbIK(this, t, SmplxJoint.L_Hip, SmplxJoint.L_Knee, SmplxJoint.L_Ankle, targetLAnkle, targetLAnkle + Vector3.forward);
                }

                if (lockRight)
                {
                    MotionIkUtility.ApplyLimbIK(this, t, SmplxJoint.R_Hip, SmplxJoint.R_Knee, SmplxJoint.R_Ankle, targetRAnkle, targetRAnkle + Vector3.forward);
                }

                _modifiedFrames.Add(t);
            }

            return true;
        }

        /// <summary>
        /// Solves Two-Bone IK on the active frame for the specified end joint (L/R Wrist, L/R Ankle).
        /// </summary>
        public bool ApplyTwoBoneIK(int frame, SmplxJoint endJoint, Vector3 targetWorldPos, Vector3 poleWorldPos)
        {
            if (frame < 0 || frame >= Frames) return false;

            SmplxJoint upperJoint;
            SmplxJoint midJoint;

            switch (endJoint)
            {
                case SmplxJoint.L_Wrist:
                    upperJoint = SmplxJoint.L_Shoulder;
                    midJoint = SmplxJoint.L_Elbow;
                    break;
                case SmplxJoint.R_Wrist:
                    upperJoint = SmplxJoint.R_Shoulder;
                    midJoint = SmplxJoint.R_Elbow;
                    break;
                case SmplxJoint.L_Ankle:
                case SmplxJoint.L_Foot:
                    upperJoint = SmplxJoint.L_Hip;
                    midJoint = SmplxJoint.L_Knee;
                    endJoint = SmplxJoint.L_Ankle;
                    break;
                case SmplxJoint.R_Ankle:
                case SmplxJoint.R_Foot:
                    upperJoint = SmplxJoint.R_Hip;
                    midJoint = SmplxJoint.R_Knee;
                    endJoint = SmplxJoint.R_Ankle;
                    break;
                default:
                    return false;
            }

            return MotionIkUtility.ApplyLimbIK(this, frame, upperJoint, midJoint, endJoint, targetWorldPos, poleWorldPos);
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
                    SourceVideoData.OverlayVideoPath,
                    SourceVideoData.UncertaintyIntervals != null ? (UncertaintyInterval[])SourceVideoData.UncertaintyIntervals.Clone() : null,
                    detectorName: SourceVideoData.DetectorName,
                    backendRequested: SourceVideoData.BackendRequested,
                    backendFallbackReason: SourceVideoData.BackendFallbackReason,
                    backendActual: SourceVideoData.BackendActual,
                    fusionMode: SourceVideoData.FusionMode,
                    backendFallback: SourceVideoData.BackendFallback,
                    fallbackFrom: SourceVideoData.FallbackFrom,
                    overlayBackend: SourceVideoData.OverlayBackend,
                    overlaySource: SourceVideoData.OverlaySource,
                    backendMetadata: SourceVideoData.BackendMetadata
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

        #region Occlusion & Ambiguity Operations (Phases 3-4)

        public UncertaintyInterval[] UncertaintyIntervals => SourceVideoData?.UncertaintyIntervals ?? Array.Empty<UncertaintyInterval>();

        /// <summary>
        /// Returns true if the frame is part of an uncertainty or occlusion interval.
        /// </summary>
        public bool HasUncertainty(int frame, out UncertaintyInterval interval)
        {
            interval = null;
            if (SourceVideoData == null) return false;
            return SourceVideoData.HasUncertainty(frame, out interval);
        }

        /// <summary>
        /// Returns true if the frame is part of an uncertainty or occlusion interval, returning the reason string.
        /// </summary>
        public bool HasUncertainty(int frame, out string reason)
        {
            if (HasUncertainty(frame, out UncertaintyInterval interval) && interval != null)
            {
                reason = interval.Reason ?? string.Empty;
                return true;
            }
            reason = string.Empty;
            return false;
        }

        /// <summary>
        /// Reverses the anterior-posterior (crossing) order between left and right legs across a range of frames.
        /// If left leg was in front, moves right leg in front, and vice versa.
        /// </summary>
        public void SwapLegCrossing(int startFrame, int endFrame)
        {
            if (startFrame < 0) startFrame = 0;
            if (endFrame >= Frames) endFrame = Frames - 1;
            if (startFrame > endFrame) return;

            RecordUndo($"Swap Leg Crossing (Frames {startFrame}-{endFrame})");

            int lHip = (int)SmplxJoint.L_Hip;
            int rHip = (int)SmplxJoint.R_Hip;
            int lKnee = (int)SmplxJoint.L_Knee;
            int rKnee = (int)SmplxJoint.R_Knee;
            int lAnkle = (int)SmplxJoint.L_Ankle;
            int rAnkle = (int)SmplxJoint.R_Ankle;

            for (int t = startFrame; t <= endFrame; t++)
            {
                // Invert the sagittal flexion (pitch) and adduction (roll) of hips
                Quaternion qLHip = LocalRotations[t, lHip];
                Quaternion qRHip = LocalRotations[t, rHip];

                Vector3 eulerL = qLHip.eulerAngles;
                Vector3 eulerR = qRHip.eulerAngles;

                // Adjust pitch angle (X)
                float pitchL = NormalizeAngle(eulerL.x);
                float pitchR = NormalizeAngle(eulerR.x);
                float avgPitch = (pitchL + pitchR) * 0.5f;
                float diffPitch = (pitchL - pitchR) * 0.5f;

                eulerL.x = avgPitch - diffPitch;
                eulerR.x = avgPitch + diffPitch;

                // Also swap lateral adduction/abduction delta
                float rollL = NormalizeAngle(eulerL.z);
                float rollR = NormalizeAngle(eulerR.z);
                float avgRoll = (rollL + rollR) * 0.5f;
                float diffRoll = (rollL - rollR) * 0.5f;
                eulerL.z = avgRoll - diffRoll;
                eulerR.z = avgRoll + diffRoll;

                LocalRotations[t, lHip] = Quaternion.Euler(eulerL);
                LocalRotations[t, rHip] = Quaternion.Euler(eulerR);

                // Knee flexion inversion
                Quaternion qLKnee = LocalRotations[t, lKnee];
                Quaternion qRKnee = LocalRotations[t, rKnee];
                Vector3 eulerLKnee = qLKnee.eulerAngles;
                Vector3 eulerRKnee = qRKnee.eulerAngles;

                float kneeL = NormalizeAngle(eulerLKnee.x);
                float kneeR = NormalizeAngle(eulerRKnee.x);
                float avgKnee = (kneeL + kneeR) * 0.5f;
                float diffKnee = (kneeL - kneeR) * 0.5f;

                eulerLKnee.x = avgKnee - diffKnee;
                eulerRKnee.x = avgKnee + diffKnee;

                LocalRotations[t, lKnee] = Quaternion.Euler(eulerLKnee);
                LocalRotations[t, rKnee] = Quaternion.Euler(eulerRKnee);

                // Ankle compensation
                Quaternion qLAnk = LocalRotations[t, lAnkle];
                Quaternion qRAnk = LocalRotations[t, rAnkle];
                Vector3 eulerLAnk = qLAnk.eulerAngles;
                Vector3 eulerRAnk = qRAnk.eulerAngles;

                float ankL = NormalizeAngle(eulerLAnk.x);
                float ankR = NormalizeAngle(eulerRAnk.x);
                float avgAnk = (ankL + ankR) * 0.5f;
                float diffAnk = (ankL - ankR) * 0.5f;
                eulerLAnk.x = avgAnk - diffAnk;
                eulerRAnk.x = avgAnk + diffAnk;

                LocalRotations[t, lAnkle] = Quaternion.Euler(eulerLAnk);
                LocalRotations[t, rAnkle] = Quaternion.Euler(eulerRAnk);

                _modifiedFrames.Add(t);
            }
        }

        /// <summary>
        /// Adjusts shoulder and elbow rotations to position arm behind head or in front of chest.
        /// </summary>
        public void FixArmPose(int startFrame, int endFrame, bool isLeftArm, bool behindHead)
        {
            if (startFrame < 0) startFrame = 0;
            if (endFrame >= Frames) endFrame = Frames - 1;
            if (startFrame > endFrame) return;

            string label = behindHead ? "Fix Arm Behind Head" : "Fix Arm Front Chest";
            RecordUndo($"{label} (Frames {startFrame}-{endFrame})");

            int shJoint = (int)(isLeftArm ? SmplxJoint.L_Shoulder : SmplxJoint.R_Shoulder);
            int elJoint = (int)(isLeftArm ? SmplxJoint.L_Elbow : SmplxJoint.R_Elbow);

            for (int t = startFrame; t <= endFrame; t++)
            {
                if (behindHead)
                {
                    // Arm flared backwards/outwards behind head
                    float shY = isLeftArm ? -40f : 40f;
                    float shZ = isLeftArm ? 25f : -25f;
                    LocalRotations[t, shJoint] = Quaternion.Euler(15f, shY, shZ);
                    LocalRotations[t, elJoint] = Quaternion.Euler(95f, 0f, 0f);
                }
                else
                {
                    // Arm positioned in front of chest
                    float shY = isLeftArm ? 35f : -35f;
                    float shZ = isLeftArm ? -20f : 20f;
                    LocalRotations[t, shJoint] = Quaternion.Euler(30f, shY, shZ);
                    LocalRotations[t, elJoint] = Quaternion.Euler(75f, 0f, 0f);
                }

                _modifiedFrames.Add(t);
            }
        }

        /// <summary>
        /// Backward-compatible wrapper for FixArmPose(..., behindHead: true).
        /// </summary>
        public void FixArmBehindHead(int startFrame, int endFrame, bool isLeftArm)
        {
            FixArmPose(startFrame, endFrame, isLeftArm, behindHead: true);
        }

        /// <summary>
        /// Applies the stylized hand-keyed motion enhancement suite across [startFrame..endFrame]
        /// with full Undo/Redo tracking.
        /// </summary>
        public bool PolishStylizedMotion(int startFrame, int endFrame, StylizedPolishOptions options, BodyPartMask mask = BodyPartMask.All)
        {
            if (options == null || startFrame < 0 || endFrame >= Frames || startFrame > endFrame)
                return false;

            RecordUndo($"Stylized Hand-Keyed Polish [{startFrame}..{endFrame}]");

            bool success = StylizedMotionPolisher.PolishMotion(this, startFrame, endFrame, options, mask);
            if (success)
            {
                for (int t = startFrame; t <= endFrame; t++)
                {
                    _modifiedFrames.Add(t);
                }
            }
            return success;
        }

        #endregion
    }
}
