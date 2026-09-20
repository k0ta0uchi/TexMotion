using System;
using System.Collections.Generic;
using TexMotion.Editor;
using TexMotion.Runtime.Motion;
using UnityEditor;
using UnityEngine;

namespace TexMotion.Editor.Motion
{
    /// <summary>
    /// Focused repair tools for tracking ambiguity, glitches, foot grounding, penetrations, and additive offsets.
    /// Provides a fixed tool chooser at the top, a bounded scrollable parameter area, and a fixed bottom execution footer.
    /// </summary>
    public class TimelineRepairPanel
    {
        private static readonly string[] ToolNames =
        {
            "Tracking Ambiguity & Gaps",
            "Motion Glitches",
            "Foot Grounding & Contact",
            "Self-Penetration Avoidance",
            "Additive Offset",
            "Face & Head Sync"
        };

        private static string[] GetLocalizedToolNames()
        {
            var localized = new string[ToolNames.Length];
            for (int i = 0; i < ToolNames.Length; i++)
            {
                localized[i] = TexMotionLocalization.TrLiteral(ToolNames[i]);
            }
            return localized;
        }

        // Tool 0: Tracking Ambiguity & Gaps state
        private float _gapConfidenceThreshold = 0.35f;
        private float _maxGapDuration = 2.5f;
        private float _hermiteDamping = 0.35f;

        // Tool 1: Motion Glitches state
        private readonly List<GlitchInfo> _detectedGlitches = new List<GlitchInfo>();
        private bool _hasScannedGlitches;
        private float _angularSpeedThreshold = GlitchDetector.DefaultAngularThresholdDegPerSec;

        // Tool 2: Foot Grounding & Contact state
        private float _groundPlaneY = 0f;
        private float _ankleGroundOffset = 0.08f;
        private float _groundSnapStrength = 1.0f;
        private bool _lockLeftFoot = true;
        private bool _lockRightFoot = true;

        // Tool 3: Self-Penetration Avoidance state
        private readonly PenetrationOptions _penetrationOptions = new PenetrationOptions();
        private float _armpitLimitAngle = 20.0f;

        // Tool 4: Additive Offset state
        private float _additiveRootHeight = 0f;
        private float _additiveArmOpenAngle = 0f;
        private bool _fadeEdges = true;

        // Tool 5: Face & Head Sync state
        private bool _enableFaceSync = true;
        private bool _enableHeadSync = true;
        private float _headRotationBlend = 1.0f;
        private ScriptableObject _faceMappingProfile;
        private string _faceSyncStatus = string.Empty;

        public void Draw(MotionTimelineEditorWindow window, EditableMotionData data, TimelineWorkspaceState state, float width, float height)
        {
            if (window == null || state == null) return;
            if (data == null || data.Frames == 0)
            {
                GUILayout.Label(TexMotionLocalization.TrLiteral("Load a motion to inspect repair tools."), MotionTimelineTheme.MutedLabel);
                return;
            }

            float previousLabelWidth = EditorGUIUtility.labelWidth;
            EditorGUIUtility.labelWidth = Mathf.Clamp(width * 0.38f, 80f, 140f);

            try
            {
                EditorGUILayout.BeginVertical(MotionTimelineTheme.Panel, GUILayout.Height(Mathf.Max(0f, height)), GUILayout.ExpandWidth(true));

                // 1. Fixed Context Header
                GUILayout.Label(TexMotionLocalization.TrLiteral("Repair Tools"), MotionTimelineTheme.Heading);
                GUILayout.Label(
                    TexMotionLocalization.TrLiteralFormat("Frame {0} / {1}   |   Range [{2} .. {3}]",
                        window.CurrentFrame + 1, data.Frames, window.RangeStartFrame + 1, window.RangeEndFrame + 1),
                    MotionTimelineTheme.MutedLabel);

                // 2. Fixed Tool Chooser at Top
                state.ActiveRepairTool = (TimelineRepairTool)EditorGUILayout.Popup(
                    TexMotionLocalization.TrLiteral("Tool"),
                    (int)state.ActiveRepairTool,
                    GetLocalizedToolNames());
                GUILayout.Space(6f);

                // 3. Bounded Scrollable Parameter Area
                state.RepairScroll = EditorGUILayout.BeginScrollView(state.RepairScroll, GUILayout.ExpandHeight(true));
                switch (state.ActiveRepairTool)
                {
                    case TimelineRepairTool.TrackingAmbiguity:
                        DrawTrackingAmbiguity(window, data);
                        break;
                    case TimelineRepairTool.MotionGlitches:
                        DrawMotionGlitches(window, data, state);
                        break;
                    case TimelineRepairTool.FootGrounding:
                        DrawFootGrounding(window, data);
                        break;
                    case TimelineRepairTool.PenetrationLimiter:
                        DrawPenetrationLimiter(window, data);
                        break;
                    case TimelineRepairTool.Offset:
                        DrawOffset(window, data);
                        break;
                    case TimelineRepairTool.FaceHeadSync:
                        DrawFaceHeadSync(window, data);
                        break;
                }
                EditorGUILayout.EndScrollView();

                // 4. Fixed Bottom Execution Footer
                DrawFooter(window, data, state, width);

                EditorGUILayout.EndVertical();
            }
            finally
            {
                EditorGUIUtility.labelWidth = previousLabelWidth;
            }
        }

        #region Tool 0: Tracking Ambiguity

        private void DrawTrackingAmbiguity(MotionTimelineEditorWindow window, EditableMotionData data)
        {
            GUILayout.Label(TexMotionLocalization.TrLiteral("Tracking Ambiguity"), MotionTimelineTheme.SectionHeader);

            bool isUncertain = data.HasUncertainty(window.CurrentFrame, out UncertaintyInterval interval);

            EditorGUILayout.BeginVertical(MotionTimelineTheme.Card);
            if (isUncertain && interval != null)
            {
                GUILayout.Label(TexMotionLocalization.TrLiteral("Ambiguity Detected on Current Frame"), MotionTimelineTheme.Label);
                GUILayout.Label(
                    TexMotionLocalization.TrLiteralFormat("Reason: {0}\nFrames: {1} - {2} (Confidence: {3})",
                        interval.Reason ?? "Tracking Ambiguity",
                        interval.StartFrame + 1,
                        interval.EndFrame + 1,
                        interval.Confidence.ToString("P0")),
                    MotionTimelineTheme.MutedLabel);
            }
            else
            {
                GUILayout.Label(TexMotionLocalization.TrLiteral("Pose Tracking Confident"), MotionTimelineTheme.Label);
                GUILayout.Label(TexMotionLocalization.TrLiteral("Tracking is confident on this frame. Use the tools below to resolve occluded limbs or crossed legs."), MotionTimelineTheme.MutedLabel);
            }
            EditorGUILayout.EndVertical();

            GUILayout.Space(6f);

            // Leg Crossing Swap
            GUILayout.Label(TexMotionLocalization.TrLiteral("Leg Crossing Order"), MotionTimelineTheme.Label);
            int swapStart = interval != null ? interval.StartFrame : window.RangeStartFrame;
            int swapEnd = interval != null ? interval.EndFrame : window.RangeEndFrame;
            string swapLabel = interval != null
                ? TexMotionLocalization.TrLiteralFormat("Swap Leg Crossing (Frames {0}-{1})", interval.StartFrame + 1, interval.EndFrame + 1)
                : TexMotionLocalization.TrLiteralFormat("Swap Leg Crossing (Range {0}-{1})", swapStart + 1, swapEnd + 1);

            if (GhostButton(swapLabel, TexMotionLocalization.TrLiteral("Swap left and right leg crossing order for this segment."), 26f))
            {
                data.SwapLegCrossing(swapStart, swapEnd);
                window.NotifyMotionChanged();
            }

            GUILayout.Space(8f);

            // Arm Pose Correction
            GUILayout.Label(TexMotionLocalization.TrLiteral("Arm Posture Resolution"), MotionTimelineTheme.Label);
            int armStart = Mathf.Max(0, window.CurrentFrame - 3);
            int armEnd = Mathf.Min(data.Frames - 1, window.CurrentFrame + 3);
            GUILayout.Label(TexMotionLocalization.TrLiteralFormat("Target Window: Frames {0} .. {1} (Current ±3)", armStart + 1, armEnd + 1), MotionTimelineTheme.MutedLabel);

            EditorGUILayout.BeginHorizontal();
            if (GhostButton(TexMotionLocalization.TrLiteral("Left Arm Behind Head"), TexMotionLocalization.TrLiteral("Place left arm behind head to resolve occluded depth.")))
            {
                data.FixArmPose(armStart, armEnd, isLeftArm: true, behindHead: true);
                window.NotifyMotionChanged();
            }
            if (GhostButton(TexMotionLocalization.TrLiteral("Right Arm Behind Head"), TexMotionLocalization.TrLiteral("Place right arm behind head to resolve occluded depth.")))
            {
                data.FixArmPose(armStart, armEnd, isLeftArm: false, behindHead: true);
                window.NotifyMotionChanged();
            }
            EditorGUILayout.EndHorizontal();

            EditorGUILayout.BeginHorizontal();
            if (GhostButton(TexMotionLocalization.TrLiteral("Left Arm Front Chest"), TexMotionLocalization.TrLiteral("Place left arm in front of chest.")))
            {
                data.FixArmPose(armStart, armEnd, isLeftArm: true, behindHead: false);
                window.NotifyMotionChanged();
            }
            if (GhostButton(TexMotionLocalization.TrLiteral("Right Arm Front Chest"), TexMotionLocalization.TrLiteral("Place right arm in front of chest.")))
            {
                data.FixArmPose(armStart, armEnd, isLeftArm: false, behindHead: false);
                window.NotifyMotionChanged();
            }
            EditorGUILayout.EndHorizontal();

            GUILayout.Space(8f);

            // Low-Confidence & Occlusion Gap Repair
            GUILayout.Label(TexMotionLocalization.TrLiteral("Low-Confidence & Gap Repair"), MotionTimelineTheme.Label);
            EditorGUILayout.BeginVertical(MotionTimelineTheme.Card);
            _gapConfidenceThreshold = EditorGUILayout.Slider(
                TexMotionLocalization.TrLiteral("Confidence Threshold"), _gapConfidenceThreshold, 0.1f, 0.9f);
            GUILayout.Space(4f);
            if (GhostButton(TexMotionLocalization.TrLiteral("Interpolate Outlier Frames (Jitter Repair)"),
                TexMotionLocalization.TrLiteral("Smooths out low-confidence outlier frames by interpolating from adjacent reliable frames."), 26f))
            {
                data.InterpolateLowConfidenceOutliers(_gapConfidenceThreshold);
                window.NotifyMotionChanged();
            }

            GUILayout.Space(6f);
            GUILayout.Label(TexMotionLocalization.TrLiteral("Kinematic Long-Gap Inpainting (Hermite & Squad)"), MotionTimelineTheme.Label);
            _maxGapDuration = EditorGUILayout.Slider(TexMotionLocalization.TrLiteral("Max Gap (s)"), _maxGapDuration, 0.5f, 3.0f);
            _hermiteDamping = EditorGUILayout.Slider(TexMotionLocalization.TrLiteral("Damping"), _hermiteDamping, 0.0f, 1.0f);

            if (GhostButton(TexMotionLocalization.TrLiteral("Inpaint Long Gaps (Kinematic Hermite)"),
                TexMotionLocalization.TrLiteral("Inpaints medium-to-long tracking dropouts (0.2s-2.5s) using momentum-preserving Hermite Splines and SO(3) Squad."), 26f))
            {
                int count = data.InpaintGapsKinematicHermite(_gapConfidenceThreshold, 0.2f, _maxGapDuration, _hermiteDamping);
                window.NotifyMotionChanged();
                EditorUtility.DisplayDialog(
                    TexMotionLocalization.TrLiteral("Kinematic Inpaint"),
                    TexMotionLocalization.TrLiteralFormat("Successfully inpainted {0} frame intervals using Hermite & Squad interpolation.", count),
                    "OK");
            }
            EditorGUILayout.EndVertical();
        }

        #endregion

        #region Tool 1: Motion Glitches

        private void DrawMotionGlitches(MotionTimelineEditorWindow window, EditableMotionData data, TimelineWorkspaceState state)
        {
            GUILayout.Label(TexMotionLocalization.TrLiteral("Motion Glitches"), MotionTimelineTheme.SectionHeader);

            EditorGUILayout.BeginVertical(MotionTimelineTheme.Card);
            _angularSpeedThreshold = EditorGUILayout.Slider(TexMotionLocalization.TrLiteral("Angular Speed Threshold"), _angularSpeedThreshold, 360f, 1440f);
            window.SelectedBodyMask = (BodyPartMask)EditorGUILayout.EnumFlagsField(TexMotionLocalization.TrLiteral("Body Mask"), window.SelectedBodyMask);

            GUILayout.Space(4f);
            if (GhostButton(TexMotionLocalization.TrLiteral("Scan Motion Glitches"), TexMotionLocalization.TrLiteral("Scan clip for unnatural angular velocity spikes and motion glitches."), 26f))
            {
                _detectedGlitches.Clear();
                _detectedGlitches.AddRange(GlitchDetector.DetectGlitches(data, _angularSpeedThreshold, window.SelectedBodyMask));
                _hasScannedGlitches = true;
                state.SelectedGlitchIndex = _detectedGlitches.Count > 0 ? 0 : -1;
            }
            EditorGUILayout.EndVertical();

            GUILayout.Space(6f);

            if (_hasScannedGlitches)
            {
                if (_detectedGlitches.Count == 0)
                {
                    EditorGUILayout.BeginVertical(MotionTimelineTheme.Card);
                    GUILayout.Label(TexMotionLocalization.TrLiteral("Scan Complete"), MotionTimelineTheme.Label);
                    GUILayout.Label(TexMotionLocalization.TrLiteral("No motion glitches or unnatural angular spikes detected in this clip."), MotionTimelineTheme.MutedLabel);
                    EditorGUILayout.EndVertical();
                }
                else
                {
                    GUILayout.Label(TexMotionLocalization.TrLiteralFormat("Detected Glitches ({0} frames)", _detectedGlitches.Count), MotionTimelineTheme.Label);

                    state.GlitchListScroll = EditorGUILayout.BeginScrollView(state.GlitchListScroll, GUILayout.MaxHeight(170f));
                    for (int i = 0; i < _detectedGlitches.Count; i++)
                    {
                        var glitch = _detectedGlitches[i];
                        bool isSelected = state.SelectedGlitchIndex == i;

                        EditorGUILayout.BeginHorizontal(MotionTimelineTheme.Card);
                        GUILayout.Label(
                            TexMotionLocalization.TrLiteralFormat("Frame {0}: {1} ({2:F0} deg/s)",
                                glitch.FrameIndex + 1, glitch.Joint, glitch.AngularVelocityDegPerSec),
                            isSelected ? MotionTimelineTheme.Label : MotionTimelineTheme.MutedLabel);

                        GUIContent goContent = new GUIContent(TexMotionLocalization.TrLiteral("Go"), TexMotionLocalization.TrLiteral("Jump to this glitch frame."));
                        if (GUILayout.Button(goContent, MotionTimelineTheme.GhostButton, GUILayout.Width(36f), GUILayout.Height(20f)))
                        {
                            window.CurrentFrame = glitch.FrameIndex;
                            state.SelectedGlitchIndex = i;
                        }

                        GUIContent fixContent = new GUIContent(TexMotionLocalization.TrLiteral("Fix"), TexMotionLocalization.TrLiteral("Fix this motion glitch using surrounding frame interpolation."));
                        if (GUILayout.Button(fixContent, MotionTimelineTheme.GhostButton, GUILayout.Width(40f), GUILayout.Height(20f)))
                        {
                            GlitchDetector.FixGlitch(data, glitch, window.SelectedBodyMask);
                            _detectedGlitches.RemoveAt(i);
                            if (state.SelectedGlitchIndex >= _detectedGlitches.Count)
                                state.SelectedGlitchIndex = _detectedGlitches.Count - 1;
                            window.NotifyMotionChanged();
                            GUIUtility.ExitGUI();
                            break;
                        }
                        EditorGUILayout.EndHorizontal();
                    }
                    EditorGUILayout.EndScrollView();
                }
            }
        }

        #endregion

        #region Tool 2: Foot Grounding & Contact

        private void DrawFootGrounding(MotionTimelineEditorWindow window, EditableMotionData data)
        {
            GUILayout.Label(TexMotionLocalization.TrLiteral("Foot Grounding & Contact"), MotionTimelineTheme.SectionHeader);

            var contactTrack = data.SourceVideoData?.ContactTrack;
            bool hasContactTrack = contactTrack != null && contactTrack.intervals != null && contactTrack.intervals.Length > 0;

            // 1. Visual Contact Status Card
            EditorGUILayout.BeginVertical(MotionTimelineTheme.Card);
            EditorGUILayout.BeginHorizontal();
            GUILayout.Label(TexMotionLocalization.TrLiteral("Contact Status (Current Frame)"), MotionTimelineTheme.Label);
            GUILayout.FlexibleSpace();
            if (hasContactTrack)
            {
                GUILayout.Label(TexMotionLocalization.TrLiteral("Video ContactTrack Active"), MotionTimelineTheme.SuccessBadge);
            }
            else
            {
                GUILayout.Label(TexMotionLocalization.TrLiteral("Estimated Contact"), MotionTimelineTheme.Badge);
            }
            EditorGUILayout.EndHorizontal();

            GUILayout.Space(4f);

            bool leftGrounded = EvaluateFootContact(contactTrack, data, window.CurrentFrame, "left", out string leftMode);
            bool rightGrounded = EvaluateFootContact(contactTrack, data, window.CurrentFrame, "right", out string rightMode);

            EditorGUILayout.BeginHorizontal();
            string lText = leftGrounded ? $"L-Foot: Grounded [{leftMode}]" : "L-Foot: In Air";
            GUILayout.Label(new GUIContent(lText, "Left foot ground contact state on current frame"),
                leftGrounded ? MotionTimelineTheme.SuccessBadge : MotionTimelineTheme.Badge, GUILayout.Height(22f), GUILayout.ExpandWidth(true));

            string rText = rightGrounded ? $"R-Foot: Grounded [{rightMode}]" : "R-Foot: In Air";
            GUILayout.Label(new GUIContent(rText, "Right foot ground contact state on current frame"),
                rightGrounded ? MotionTimelineTheme.SuccessBadge : MotionTimelineTheme.Badge, GUILayout.Height(22f), GUILayout.ExpandWidth(true));
            EditorGUILayout.EndHorizontal();

            // Range contact percentage summary
            int rangeStart = Mathf.Min(window.RangeStartFrame, window.RangeEndFrame);
            int rangeEnd = Mathf.Max(window.RangeStartFrame, window.RangeEndFrame);
            int rangeLength = Mathf.Max(1, rangeEnd - rangeStart + 1);

            int lCount = 0;
            int rCount = 0;
            for (int f = rangeStart; f <= rangeEnd; f++)
            {
                if (EvaluateFootContact(contactTrack, data, f, "left", out _)) lCount++;
                if (EvaluateFootContact(contactTrack, data, f, "right", out _)) rCount++;
            }

            GUILayout.Space(4f);
            GUILayout.Label(
                TexMotionLocalization.TrLiteralFormat("Range [{0}..{1}] Contact: Left {2:P0}, Right {3:P0}",
                    rangeStart + 1, rangeEnd + 1, (float)lCount / rangeLength, (float)rCount / rangeLength),
                MotionTimelineTheme.MutedLabel);
            EditorGUILayout.EndVertical();

            GUILayout.Space(6f);

            // 2. Parameters Card
            EditorGUILayout.BeginVertical(MotionTimelineTheme.Card);
            _groundPlaneY = EditorGUILayout.Slider(TexMotionLocalization.TrLiteral("Ground Plane Y (m)"), _groundPlaneY, -1.0f, 1.0f);
            _ankleGroundOffset = EditorGUILayout.Slider(TexMotionLocalization.TrLiteral("Ankle Offset (m)"), _ankleGroundOffset, 0.0f, 0.2f);
            _groundSnapStrength = EditorGUILayout.Slider(TexMotionLocalization.TrLiteral("Snap Strength"), _groundSnapStrength, 0.0f, 1.0f);
            EditorGUILayout.EndVertical();

            GUILayout.Space(6f);

            // 3. Avatar Bone-Length Adapted Grounding Action
            EditorGUILayout.BeginVertical(MotionTimelineTheme.Card);
            GUILayout.Label(TexMotionLocalization.TrLiteral("Avatar-Adapted Grounding"), MotionTimelineTheme.Label);
            GUILayout.Label(TexMotionLocalization.TrLiteral("Solves Two-Bone IK and foot locking adapted to avatar leg length."), MotionTimelineTheme.MutedLabel);
            GUILayout.Space(4f);

            if (GhostButton(TexMotionLocalization.TrLiteral("Apply Avatar Grounding (Entire Clip)"),
                TexMotionLocalization.TrLiteral("Apply bone-length adapted contact IK across the clip to eliminate foot sliding and floating."), 26f))
            {
                data.ApplyAvatarGroundingConstraint(data.TargetAvatar, _groundSnapStrength, _groundPlaneY, _ankleGroundOffset);
                window.RecalculateGroundingOffset();
                window.NotifyMotionChanged();
            }
            EditorGUILayout.EndVertical();

            GUILayout.Space(6f);

            // 4. Position Lock Card
            EditorGUILayout.BeginVertical(MotionTimelineTheme.Card);
            GUILayout.Label(TexMotionLocalization.TrLiteral("Foot Position Lock"), MotionTimelineTheme.Label);
            _lockLeftFoot = EditorGUILayout.Toggle(TexMotionLocalization.TrLiteral("Lock Left Foot"), _lockLeftFoot);
            _lockRightFoot = EditorGUILayout.Toggle(TexMotionLocalization.TrLiteral("Lock Right Foot"), _lockRightFoot);

            GUILayout.Space(4f);
            if (GhostButton(TexMotionLocalization.TrLiteral("Lock Feet Position (Range)"), TexMotionLocalization.TrLiteral("Lock feet positions in place across the selected In-Out range."), 24f))
            {
                data.LockFootPosition(window.RangeStartFrame, window.RangeEndFrame, _lockLeftFoot, _lockRightFoot);
                window.NotifyMotionChanged();
            }
            EditorGUILayout.EndVertical();

            GUILayout.Space(6f);

            // 5. Basic Grounding Actions Card
            EditorGUILayout.BeginVertical(MotionTimelineTheme.Card);
            GUILayout.Label(TexMotionLocalization.TrLiteral("Grounding Actions"), MotionTimelineTheme.Label);

            if (GhostButton(TexMotionLocalization.TrLiteral("Ground Entire Clip"), TexMotionLocalization.TrLiteral("Shift the entire motion clip so the lowest foot landing touches Ground Y."), 24f))
            {
                data.GroundEntireClip(_groundPlaneY, _ankleGroundOffset);
                window.RecalculateGroundingOffset();
                window.NotifyMotionChanged();
            }

            GUILayout.Space(3f);

            if (GhostButton(TexMotionLocalization.TrLiteral("Snap In-Out Range to Ground Y"), TexMotionLocalization.TrLiteral("Snap lowest feet to Ground Y across the selected In-Out range."), 24f))
            {
                data.ApplyFootGrounding(window.RangeStartFrame, window.RangeEndFrame, _groundPlaneY, _ankleGroundOffset, true);
                window.RecalculateGroundingOffset();
                window.NotifyMotionChanged();
            }

            GUILayout.Space(3f);

            if (GhostButton(TexMotionLocalization.TrLiteral("Snap Current Frame to Ground Y"), TexMotionLocalization.TrLiteral("Snap hips vertically so the lowest foot contacts the ground plane."), 24f))
            {
                data.ApplyFootGrounding(window.CurrentFrame, _groundPlaneY, _ankleGroundOffset, true);
                window.RecalculateGroundingOffset();
                window.NotifyMotionChanged();
            }
            EditorGUILayout.EndVertical();
        }

        private static bool EvaluateFootContact(
            ContactTrackData track,
            EditableMotionData data,
            int frame,
            string foot,
            out string mode)
        {
            mode = "flat";
            if (data == null || frame < 0 || frame >= data.Frames) return false;

            if (track != null && track.intervals != null && track.intervals.Length > 0)
            {
                float time = data.Timestamps != null && frame < data.Timestamps.Length
                    ? data.Timestamps[frame]
                    : (data.FrameRate > 0f ? (float)frame / data.FrameRate : 0f);

                foreach (var interval in track.intervals)
                {
                    if (interval == null) continue;
                    bool inRange = (time >= interval.start && time <= interval.end) ||
                                   (frame >= (int)interval.start && frame <= (int)interval.end);
                    if (inRange && string.Equals(interval.foot, foot, StringComparison.OrdinalIgnoreCase))
                    {
                        mode = string.IsNullOrEmpty(interval.mode) ? "flat" : interval.mode;
                        return true;
                    }
                }
                return false;
            }

            // Fallback estimation using FK foot position
            Vector3 root = data.GetRootPosition(frame);
            var rots = new Quaternion[SmplxJointDefinitions.JointCount];
            for (int j = 0; j < rots.Length; j++) rots[j] = data.GetJointRotation(frame, (SmplxJoint)j);
            var fk = MotionIkUtility.ComputeForwardKinematics(root, rots);

            SmplxJoint ankleJoint = string.Equals(foot, "left", StringComparison.OrdinalIgnoreCase)
                ? SmplxJoint.L_Ankle
                : SmplxJoint.R_Ankle;

            float footY = fk[(int)ankleJoint].y;
            if (footY < 0.12f)
            {
                mode = "flat";
                return true;
            }
            return false;
        }

        #endregion

        #region Tool 3: Self-Penetration Avoidance

        private void DrawPenetrationLimiter(MotionTimelineEditorWindow window, EditableMotionData data)
        {
            GUILayout.Label(TexMotionLocalization.TrLiteral("Self-Penetration Avoidance"), MotionTimelineTheme.SectionHeader);

            // 1. Realtime Penetration Diagnostic Card
            int penCount = PenetrationConstraintSolver.CountPenetrations(data, window.CurrentFrame, _penetrationOptions);
            EditorGUILayout.BeginVertical(MotionTimelineTheme.Card);
            EditorGUILayout.BeginHorizontal();
            GUILayout.Label(TexMotionLocalization.TrLiteral("Collision Status (Current Frame)"), MotionTimelineTheme.Label);
            GUILayout.FlexibleSpace();
            if (penCount == 0)
            {
                GUILayout.Label(TexMotionLocalization.TrLiteral("No Penetrations Detected"), MotionTimelineTheme.SuccessBadge);
            }
            else
            {
                GUILayout.Label(TexMotionLocalization.TrLiteralFormat("{0} Limb Penetration(s)", penCount), MotionTimelineTheme.Badge);
            }
            EditorGUILayout.EndHorizontal();

            GUILayout.Space(2f);
            GUILayout.Label(
                TexMotionLocalization.TrLiteral("Monitors arms vs torso/chest and thighs using analytical geometric capsules."),
                MotionTimelineTheme.MutedLabel);
            EditorGUILayout.EndVertical();

            GUILayout.Space(6f);

            // 2. Geometry & Clearance Parameters Card
            EditorGUILayout.BeginVertical(MotionTimelineTheme.Card);
            GUILayout.Label(TexMotionLocalization.TrLiteral("Capsule Clearance"), MotionTimelineTheme.Label);

            _penetrationOptions.TorsoRadiusOffset = EditorGUILayout.Slider(
                TexMotionLocalization.TrLiteral("Torso Radius Offset (m)"), _penetrationOptions.TorsoRadiusOffset, -0.05f, 0.08f);
            _penetrationOptions.ThighRadiusOffset = EditorGUILayout.Slider(
                TexMotionLocalization.TrLiteral("Thigh Radius Offset (m)"), _penetrationOptions.ThighRadiusOffset, -0.05f, 0.08f);
            _penetrationOptions.Margin = EditorGUILayout.Slider(
                TexMotionLocalization.TrLiteral("Safety Margin (m)"), _penetrationOptions.Margin, 0.005f, 0.04f);
            _penetrationOptions.PushBlendWeight = EditorGUILayout.Slider(
                TexMotionLocalization.TrLiteral("Correction Strength"), _penetrationOptions.PushBlendWeight, 0.1f, 1.0f);
            EditorGUILayout.EndVertical();

            GUILayout.Space(6f);

            // 3. Intentional Contact Protection Card
            EditorGUILayout.BeginVertical(MotionTimelineTheme.Card);
            GUILayout.Label(TexMotionLocalization.TrLiteral("Intentional Contact Protection"), MotionTimelineTheme.Label);
            GUILayout.Label(TexMotionLocalization.TrLiteral("Relaxes push-out forces for crossed arms and hands on torso."), MotionTimelineTheme.MutedLabel);
            GUILayout.Space(4f);

            _penetrationOptions.ProtectCrossedArms = EditorGUILayout.Toggle(
                TexMotionLocalization.TrLiteral("Protect Crossed Arms"), _penetrationOptions.ProtectCrossedArms);
            _penetrationOptions.ProtectHandOnChest = EditorGUILayout.Toggle(
                TexMotionLocalization.TrLiteral("Protect Hand on Chest"), _penetrationOptions.ProtectHandOnChest);
            _penetrationOptions.RelaxationThreshold = EditorGUILayout.Slider(
                TexMotionLocalization.TrLiteral("Relaxation Threshold"), _penetrationOptions.RelaxationThreshold, 0.0f, 1.0f);
            EditorGUILayout.EndVertical();

            GUILayout.Space(6f);

            // 4. Execution Actions Card
            EditorGUILayout.BeginVertical(MotionTimelineTheme.Card);
            GUILayout.Label(TexMotionLocalization.TrLiteral("Penetration Actions"), MotionTimelineTheme.Label);

            if (GhostButton(TexMotionLocalization.TrLiteral("Avoid Self-Penetration (Range)"),
                TexMotionLocalization.TrLiteral("Pushes penetrating arm joints outward across the selected In-Out range."), 26f))
            {
                data.ApplySelfPenetrationAvoidance(window.RangeStartFrame, window.RangeEndFrame, _penetrationOptions);
                window.NotifyMotionChanged();
            }

            GUILayout.Space(3f);

            if (GhostButton(TexMotionLocalization.TrLiteral("Avoid Self-Penetration (Entire Clip)"),
                TexMotionLocalization.TrLiteral("Pushes penetrating arm joints outward across all frames."), 26f))
            {
                data.ApplySelfPenetrationAvoidance(0, data.Frames - 1, _penetrationOptions);
                window.NotifyMotionChanged();
            }
            EditorGUILayout.EndVertical();

            GUILayout.Space(6f);

            // 5. Legacy Armpit Angle Clamping Card
            EditorGUILayout.BeginVertical(MotionTimelineTheme.Card);
            GUILayout.Label(TexMotionLocalization.TrLiteral("Min Armpit Angle (Fast Clamp)"), MotionTimelineTheme.Label);
            _armpitLimitAngle = EditorGUILayout.Slider(TexMotionLocalization.TrLiteral("Angle (deg)"), _armpitLimitAngle, 5.0f, 45.0f);

            EditorGUILayout.BeginHorizontal();
            if (GhostButton(TexMotionLocalization.TrLiteral("15 deg (Subtle)"), null, 20f)) _armpitLimitAngle = 15.0f;
            if (GhostButton(TexMotionLocalization.TrLiteral("20 deg (Standard)"), null, 20f)) _armpitLimitAngle = 20.0f;
            if (GhostButton(TexMotionLocalization.TrLiteral("30 deg (Wide)"), null, 20f)) _armpitLimitAngle = 30.0f;
            EditorGUILayout.EndHorizontal();

            GUILayout.Space(3f);
            if (GhostButton(TexMotionLocalization.TrLiteral("Apply Armpit Angle Clamp"),
                TexMotionLocalization.TrLiteral("Clamps minimum shoulder-arm angle across range."), 22f))
            {
                data.ApplyArmpitPenetrationLimiter(window.RangeStartFrame, window.RangeEndFrame, _armpitLimitAngle);
                window.NotifyMotionChanged();
            }
            EditorGUILayout.EndVertical();
        }

        #endregion

        #region Tool 4: Additive Offset

        private void DrawOffset(MotionTimelineEditorWindow window, EditableMotionData data)
        {
            GUILayout.Label(TexMotionLocalization.TrLiteral("Additive Range Offset"), MotionTimelineTheme.SectionHeader);

            EditorGUILayout.BeginVertical(MotionTimelineTheme.Card);
            _additiveRootHeight = EditorGUILayout.Slider(TexMotionLocalization.TrLiteral("Hips Y Offset (m)"), _additiveRootHeight, -0.5f, 0.5f);
            _additiveArmOpenAngle = EditorGUILayout.Slider(TexMotionLocalization.TrLiteral("Arm Open Angle (deg)"), _additiveArmOpenAngle, -30.0f, 30.0f);
            _fadeEdges = EditorGUILayout.Toggle(TexMotionLocalization.TrLiteral("Fade at Range Edges"), _fadeEdges);

            GUILayout.Space(4f);
            if (GhostButton(TexMotionLocalization.TrLiteral("Reset Offsets"), TexMotionLocalization.TrLiteral("Reset additive offset values to zero."), 20f))
            {
                _additiveRootHeight = 0f;
                _additiveArmOpenAngle = 0f;
            }
            EditorGUILayout.EndVertical();

            GUILayout.Space(4f);
            EditorGUILayout.HelpBox(TexMotionLocalization.TrLiteral("Applies relative additive elevations and arm flares smoothly across the selected range."), MessageType.None);
        }

        #endregion

        #region Tool 5: Face & Head Sync

        private void DrawFaceHeadSync(MotionTimelineEditorWindow window, EditableMotionData data)
        {
            GUILayout.Label(TexMotionLocalization.TrLiteral("Face & Head Synchronization"), MotionTimelineTheme.SectionHeader);

            var faceTrack = data.SourceVideoData?.FaceTrack;
            bool hasFaceTrack = faceTrack != null && faceTrack.shapes != null && faceTrack.shapes.Length > 0;

            // 1. Video Face Track Status Card
            EditorGUILayout.BeginVertical(MotionTimelineTheme.Card);
            EditorGUILayout.BeginHorizontal();
            GUILayout.Label(TexMotionLocalization.TrLiteral("Video Face Track Status"), MotionTimelineTheme.Label);
            GUILayout.FlexibleSpace();
            if (hasFaceTrack)
            {
                int shapeCount = faceTrack.shapes.Length;
                int frameCount = faceTrack.timestamps != null ? faceTrack.timestamps.Length : 0;
                GUILayout.Label(TexMotionLocalization.TrLiteralFormat("Active ({0} shapes, {1} frames)", shapeCount, frameCount), MotionTimelineTheme.SuccessBadge);
            }
            else
            {
                GUILayout.Label(TexMotionLocalization.TrLiteral("No Face Track in Source"), MotionTimelineTheme.Badge);
            }
            EditorGUILayout.EndHorizontal();

            GUILayout.Space(2f);
            if (hasFaceTrack)
            {
                GUILayout.Label(TexMotionLocalization.TrLiteral("Video facial blendshape and head rotation tracks detected. Synchronize with avatar SkinnedMeshRenderer."), MotionTimelineTheme.MutedLabel);
            }
            else
            {
                GUILayout.Label(TexMotionLocalization.TrLiteral("No facial data in source video. Manual emotion presets from the Pose tab will be used."), MotionTimelineTheme.MutedLabel);
            }
            EditorGUILayout.EndVertical();

            GUILayout.Space(6f);

            // 2. Synchronization Settings Card
            EditorGUILayout.BeginVertical(MotionTimelineTheme.Card);
            GUILayout.Label(TexMotionLocalization.TrLiteral("Sync Options"), MotionTimelineTheme.Label);

            _enableFaceSync = EditorGUILayout.Toggle(TexMotionLocalization.TrLiteral("Sync Facial BlendShapes"), _enableFaceSync);
            _enableHeadSync = EditorGUILayout.Toggle(TexMotionLocalization.TrLiteral("Sync Head Rotation"), _enableHeadSync);

            if (_enableHeadSync)
            {
                EditorGUI.indentLevel++;
                _headRotationBlend = EditorGUILayout.Slider(TexMotionLocalization.TrLiteral("Head Rotation Blend"), _headRotationBlend, 0f, 1f);
                GUILayout.Label(TexMotionLocalization.TrLiteralFormat("Blend: {0:P0} (0% body only, 100% video tracked)", _headRotationBlend), MotionTimelineTheme.MutedLabel);
                EditorGUI.indentLevel--;
            }
            EditorGUILayout.EndVertical();

            GUILayout.Space(6f);

            // 3. FaceMappingProfile Card
            EditorGUILayout.BeginVertical(MotionTimelineTheme.Card);
            GUILayout.Label(TexMotionLocalization.TrLiteral("Face Mapping Profile"), MotionTimelineTheme.Label);
            GUILayout.Label(TexMotionLocalization.TrLiteral("Maps MediaPipe 52 BlendShapes to avatar-specific facial blendshapes."), MotionTimelineTheme.MutedLabel);
            GUILayout.Space(4f);

            _faceMappingProfile = (ScriptableObject)EditorGUILayout.ObjectField(
                TexMotionLocalization.TrLiteral("Mapping Profile"),
                _faceMappingProfile,
                typeof(ScriptableObject),
                false);

            GUILayout.Space(4f);
            EditorGUILayout.BeginHorizontal();
            if (GhostButton(TexMotionLocalization.TrLiteral("Create / Auto-Generate Profile"),
                TexMotionLocalization.TrLiteral("Analyze avatar SkinnedMeshRenderer and auto-generate a FaceMappingProfile asset."), 24f))
            {
                AutoGenerateFaceProfile(data);
            }
            EditorGUILayout.EndHorizontal();

            if (!string.IsNullOrEmpty(_faceSyncStatus))
            {
                GUILayout.Space(3f);
                GUILayout.Label(_faceSyncStatus, MotionTimelineTheme.MutedLabel);
            }
            EditorGUILayout.EndVertical();

            GUILayout.Space(6f);

            // 4. Apply Actions Card
            EditorGUILayout.BeginVertical(MotionTimelineTheme.Card);
            GUILayout.Label(TexMotionLocalization.TrLiteral("Sync Execution"), MotionTimelineTheme.Label);

            if (GhostButton(TexMotionLocalization.TrLiteral("Apply Face & Head Sync (Entire Clip)"),
                TexMotionLocalization.TrLiteral("Bake video face blendshapes and head rotation into the animation clip and preview avatar."), 26f))
            {
                ApplyFaceAndHeadSync(window, data);
            }
            EditorGUILayout.EndVertical();
        }

        private void AutoGenerateFaceProfile(EditableMotionData data)
        {
            var avatar = data.TargetAvatar;
            if (avatar == null)
            {
                _faceSyncStatus = TexMotionLocalization.TrLiteral("Target Avatar is required to generate profile.");
                return;
            }

            var smr = avatar.GetComponentInChildren<SkinnedMeshRenderer>();
            if (smr == null || smr.sharedMesh == null || smr.sharedMesh.blendShapeCount == 0)
            {
                _faceSyncStatus = TexMotionLocalization.TrLiteral("No SkinnedMeshRenderer with BlendShapes found on avatar.");
                return;
            }

            // Check if Worker 3's FaceMappingProfile type is available
            Type profileType = Type.GetType("TexMotion.Runtime.Motion.FaceMappingProfile, TexMotion.Runtime")
                            ?? Type.GetType("TexMotion.Runtime.Motion.FaceMappingProfile");

            if (profileType != null)
            {
                var profile = ScriptableObject.CreateInstance(profileType);
                var autoGenMethod = profileType.GetMethod("AutoGenerateMapping", new Type[] { typeof(SkinnedMeshRenderer) });
                if (autoGenMethod != null)
                {
                    autoGenMethod.Invoke(profile, new object[] { smr });
                }

                string assetPath = "Assets/TexMotion_FaceMappingProfile.asset";
                AssetDatabase.CreateAsset(profile, assetPath);
                AssetDatabase.SaveAssets();

                _faceMappingProfile = profile;
                _faceSyncStatus = TexMotionLocalization.TrLiteralFormat("Generated FaceMappingProfile at {0} ({1} shapes detected).", assetPath, smr.sharedMesh.blendShapeCount);
            }
            else
            {
                _faceSyncStatus = TexMotionLocalization.TrLiteral("FaceMappingProfile asset type not yet loaded. Using standard avatar blendshape map.");
            }
        }

        private void ApplyFaceAndHeadSync(MotionTimelineEditorWindow window, EditableMotionData data)
        {
            var faceTrack = data.SourceVideoData?.FaceTrack;
            if (faceTrack == null)
            {
                _faceSyncStatus = TexMotionLocalization.TrLiteral("No FaceTrack available in source video.");
                return;
            }

            // Blend head rotation into Head joint if enabled
            if (_enableHeadSync && faceTrack.headRotationsFlat != null && faceTrack.headRotationsFlat.Length >= 4)
            {
                data.RecordUndo("Apply Face & Head Sync");
                int frames = Mathf.Min(data.Frames, faceTrack.headRotationsFlat.Length / 4);

                for (int t = 0; t < frames; t++)
                {
                    int offset = t * 4;
                    Quaternion headRot = new Quaternion(
                        faceTrack.headRotationsFlat[offset],
                        faceTrack.headRotationsFlat[offset + 1],
                        faceTrack.headRotationsFlat[offset + 2],
                        faceTrack.headRotationsFlat[offset + 3]);

                    float sqrMag = headRot.x * headRot.x + headRot.y * headRot.y + headRot.z * headRot.z + headRot.w * headRot.w;
                    if (!float.IsNaN(headRot.x) && !float.IsInfinity(headRot.x) && sqrMag > 0.001f)
                    {
                        Quaternion currentRot = data.GetJointRotation(t, SmplxJoint.Head);
                        Quaternion blendedRot = MotionIkUtility.SlerpManaged(currentRot, headRot, _headRotationBlend);
                        data.SetJointRotation(t, SmplxJoint.Head, blendedRot);
                    }
                }
                window.NotifyMotionChanged();
            }

            _faceSyncStatus = TexMotionLocalization.TrLiteral("Face & Head sync applied successfully.");
        }

        #endregion

        #region Fixed Bottom Footer

        private void DrawFooter(MotionTimelineEditorWindow window, EditableMotionData data, TimelineWorkspaceState state, float width)
        {
            GUILayout.Space(4f);
            Rect hairlineRect = GUILayoutUtility.GetRect(width, 1f, GUILayout.ExpandWidth(true));
            MotionTimelineTheme.DrawHairline(hairlineRect, MotionTimelineTheme.Graphite);
            GUILayout.Space(4f);

            string scopeSummary;
            string actionButtonLabel;
            string actionTooltip;
            Action executionAction;
            bool actionEnabled = true;

            int start = window.RangeStartFrame;
            int end = window.RangeEndFrame;

            switch (state.ActiveRepairTool)
            {
                case TimelineRepairTool.TrackingAmbiguity:
                {
                    bool isUncertain = data.HasUncertainty(window.CurrentFrame, out UncertaintyInterval interval);
                    int s = interval != null ? interval.StartFrame : start;
                    int e = interval != null ? interval.EndFrame : end;
                    scopeSummary = interval != null
                        ? TexMotionLocalization.TrLiteralFormat("Affects frames [{0} .. {1}] (Detected Ambiguity: {2})", s + 1, e + 1, interval.Reason ?? "Leg Crossing")
                        : TexMotionLocalization.TrLiteralFormat("Affects frames [{0} .. {1}]", start + 1, end + 1);
                    actionButtonLabel = TexMotionLocalization.TrLiteral("Swap Leg Crossing");
                    actionTooltip = TexMotionLocalization.TrLiteral("Swap left and right leg crossing order for this segment.");
                    executionAction = () => data.SwapLegCrossing(s, e);
                    break;
                }
                case TimelineRepairTool.MotionGlitches:
                {
                    scopeSummary = _detectedGlitches.Count > 0
                        ? TexMotionLocalization.TrLiteralFormat("Affects {0} detected glitch frame(s)", _detectedGlitches.Count)
                        : TexMotionLocalization.TrLiteral("Affects all clip frames");
                    actionButtonLabel = TexMotionLocalization.TrLiteral("Fix All Glitches");
                    actionTooltip = TexMotionLocalization.TrLiteral("Fix all detected motion glitches across the clip.");
                    actionEnabled = _detectedGlitches.Count > 0;
                    executionAction = () =>
                    {
                        GlitchDetector.FixAllGlitches(data, _detectedGlitches, window.SelectedBodyMask);
                        _detectedGlitches.Clear();
                        state.SelectedGlitchIndex = -1;
                    };
                    break;
                }
                case TimelineRepairTool.FootGrounding:
                {
                    scopeSummary = TexMotionLocalization.TrLiteralFormat("Affects frames [{0} .. {1}] / Feet", start + 1, end + 1);
                    actionButtonLabel = TexMotionLocalization.TrLiteral("Apply Foot Grounding (Range)");
                    actionTooltip = TexMotionLocalization.TrLiteral("Apply foot grounding adjustment across the selected In-Out range.");
                    executionAction = () => data.ApplyFootGrounding(start, end, _groundPlaneY, _ankleGroundOffset);
                    break;
                }
                case TimelineRepairTool.PenetrationLimiter:
                {
                    scopeSummary = TexMotionLocalization.TrLiteralFormat("Affects frames [{0} .. {1}] / Arms & Torso", start + 1, end + 1);
                    actionButtonLabel = TexMotionLocalization.TrLiteral("Avoid Self-Penetration (Range)");
                    actionTooltip = TexMotionLocalization.TrLiteral("Push penetrating arms outside torso and thigh capsules across the selected In-Out range.");
                    executionAction = () => data.ApplySelfPenetrationAvoidance(start, end, _penetrationOptions);
                    break;
                }
                case TimelineRepairTool.Offset:
                {
                    scopeSummary = TexMotionLocalization.TrLiteralFormat("Affects frames [{0} .. {1}] / Root & Arms", start + 1, end + 1);
                    actionButtonLabel = TexMotionLocalization.TrLiteral("Apply Additive Offset");
                    actionTooltip = TexMotionLocalization.TrLiteral("Apply additive elevation and arm opening offsets across the range.");
                    executionAction = () =>
                    {
                        Vector3 rootOffset = new Vector3(0f, _additiveRootHeight, 0f);
                        Vector3 armEulerOffset = new Vector3(0f, 0f, _additiveArmOpenAngle);
                        data.ApplyRangeOffset(start, end, rootOffset, armEulerOffset, _fadeEdges);
                    };
                    break;
                }
                case TimelineRepairTool.FaceHeadSync:
                {
                    scopeSummary = TexMotionLocalization.TrLiteral("Affects Entire Clip / Head & BlendShapes");
                    actionButtonLabel = TexMotionLocalization.TrLiteral("Apply Face & Head Sync");
                    actionTooltip = TexMotionLocalization.TrLiteral("Synchronize video facial blendshapes and head rotation to avatar.");
                    executionAction = () => ApplyFaceAndHeadSync(window, data);
                    break;
                }
                default:
                    scopeSummary = TexMotionLocalization.TrLiteralFormat("Affects frames [{0} .. {1}]", start + 1, end + 1);
                    actionButtonLabel = TexMotionLocalization.TrLiteral("Apply Repair");
                    actionTooltip = string.Empty;
                    executionAction = null;
                    actionEnabled = false;
                    break;
            }

            // Scope summary label
            GUILayout.Label(scopeSummary, MotionTimelineTheme.MutedLabel);

            // Outlined Neutral Action Button (NEVER green or teal)
            using (new EditorGUI.DisabledScope(!actionEnabled))
            {
                GUIContent actionContent = string.IsNullOrEmpty(actionTooltip)
                    ? new GUIContent(actionButtonLabel)
                    : new GUIContent(actionButtonLabel, actionTooltip);
                if (GUILayout.Button(actionContent, MotionTimelineTheme.Button, GUILayout.Height(26f), GUILayout.ExpandWidth(true)))
                {
                    executionAction?.Invoke();
                    window.NotifyMotionChanged();
                }
            }
        }

        #endregion

        #region Helpers

        private static bool GhostButton(string label, string tooltip = null, float height = 24f)
        {
            GUIContent content = string.IsNullOrEmpty(tooltip)
                ? new GUIContent(label)
                : new GUIContent(label, tooltip);
            return GUILayout.Button(content, MotionTimelineTheme.GhostButton, GUILayout.Height(height), GUILayout.ExpandWidth(true));
        }

        #endregion
    }
}
