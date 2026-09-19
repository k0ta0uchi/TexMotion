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
            "Tracking Ambiguity",
            "Motion Glitches",
            "Foot Grounding & Contact",
            "Penetration Limiter",
            "Additive Offset"
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

        // Tool 1: Motion Glitches state
        private readonly List<GlitchInfo> _detectedGlitches = new List<GlitchInfo>();
        private bool _hasScannedGlitches;
        private float _angularSpeedThreshold = GlitchDetector.DefaultAngularThresholdDegPerSec;

        // Tool 2: Foot Grounding & Contact state
        private float _groundPlaneY = 0f;
        private float _ankleGroundOffset = 0.08f;
        private bool _lockLeftFoot = true;
        private bool _lockRightFoot = true;

        // Tool 3: Penetration Limiter state
        private float _armpitLimitAngle = 20.0f;

        // Tool 4: Additive Offset state
        private float _additiveRootHeight = 0f;
        private float _additiveArmOpenAngle = 0f;
        private bool _fadeEdges = true;

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

            EditorGUILayout.BeginVertical(MotionTimelineTheme.Card);
            _groundPlaneY = EditorGUILayout.Slider(TexMotionLocalization.TrLiteral("Ground Plane Y (m)"), _groundPlaneY, -1.0f, 1.0f);
            _ankleGroundOffset = EditorGUILayout.Slider(TexMotionLocalization.TrLiteral("Ankle Offset (m)"), _ankleGroundOffset, 0.0f, 0.2f);
            EditorGUILayout.EndVertical();

            GUILayout.Space(6f);

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

        #endregion

        #region Tool 3: Penetration Limiter

        private void DrawPenetrationLimiter(MotionTimelineEditorWindow window, EditableMotionData data)
        {
            GUILayout.Label(TexMotionLocalization.TrLiteral("Armpit Penetration Limiter"), MotionTimelineTheme.SectionHeader);

            EditorGUILayout.BeginVertical(MotionTimelineTheme.Card);
            GUILayout.Label(TexMotionLocalization.TrLiteral("Torso Clearance"), MotionTimelineTheme.Label);
            GUILayout.Label(TexMotionLocalization.TrLiteral("Clamps minimum arm opening angle to prevent shoulders and upper arms from penetrating the chest."), MotionTimelineTheme.MutedLabel);
            GUILayout.Space(4f);

            _armpitLimitAngle = EditorGUILayout.Slider(TexMotionLocalization.TrLiteral("Min Armpit Angle (deg)"), _armpitLimitAngle, 5.0f, 45.0f);

            EditorGUILayout.BeginHorizontal();
            if (GhostButton(TexMotionLocalization.TrLiteral("15 deg (Subtle)"), null, 20f)) _armpitLimitAngle = 15.0f;
            if (GhostButton(TexMotionLocalization.TrLiteral("20 deg (Standard)"), null, 20f)) _armpitLimitAngle = 20.0f;
            if (GhostButton(TexMotionLocalization.TrLiteral("30 deg (Wide)"), null, 20f)) _armpitLimitAngle = 30.0f;
            EditorGUILayout.EndHorizontal();
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
                    scopeSummary = TexMotionLocalization.TrLiteralFormat("Affects frames [{0} .. {1}] / Upper Arms", start + 1, end + 1);
                    actionButtonLabel = TexMotionLocalization.TrLiteral("Apply Armpit Limiter");
                    actionTooltip = TexMotionLocalization.TrLiteral("Apply armpit penetration limiter to prevent arm-torso clipping.");
                    executionAction = () => data.ApplyArmpitPenetrationLimiter(start, end, _armpitLimitAngle);
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
