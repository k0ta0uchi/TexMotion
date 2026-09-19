using System;
using TexMotion.Editor;
using TexMotion.Runtime.Motion;
using UnityEditor;
using UnityEngine;

namespace TexMotion.Editor.Motion
{
    /// <summary>
    /// Timing and temporal editing tools: Range Tweening, Loop Blending, and Range Retiming.
    /// Provides a fixed tool chooser at the top, a bounded scrollable parameter area, and a fixed bottom execution footer.
    /// </summary>
    public class TimelineTimingPanel
    {
        private static readonly string[] TimingToolNames =
        {
            "Range Tweening",
            "Loop Blending",
            "Range Retiming"
        };

        private static readonly string[] InterpolationNames =
        {
            "Linear",
            "SmoothStep",
            "EaseIn",
            "EaseOut",
            "Hermite"
        };

        private static string[] GetLocalizedTimingToolNames()
        {
            var localized = new string[TimingToolNames.Length];
            for (int i = 0; i < TimingToolNames.Length; i++)
            {
                localized[i] = TexMotionLocalization.TrLiteral(TimingToolNames[i]);
            }
            return localized;
        }

        private static string[] GetLocalizedInterpolationNames()
        {
            var localized = new string[InterpolationNames.Length];
            for (int i = 0; i < InterpolationNames.Length; i++)
            {
                localized[i] = TexMotionLocalization.TrLiteral(InterpolationNames[i]);
            }
            return localized;
        }

        // Tool 0: Range Tweening state
        private int _interpolationIndex = 1; // 0: Linear, 1: SmoothStep, 2: EaseIn, 3: EaseOut, 4: Hermite
        private bool _includeRoot = true;

        // Tool 1: Loop Blending state
        private int _loopBlendFrames = 10;
        private bool _matchRootPosition = true;

        // Tool 2: Range Retiming state
        private int _retimeNewFrameCount = 30;
        private int _lastObservedSpan = -1;

        public void Draw(MotionTimelineEditorWindow window, EditableMotionData data, TimelineWorkspaceState state, float width, float height)
        {
            if (window == null || state == null) return;
            if (data == null || data.Frames == 0)
            {
                GUILayout.Label(TexMotionLocalization.TrLiteral("Load a motion to inspect timing tools."), MotionTimelineTheme.MutedLabel);
                return;
            }

            float previousLabelWidth = EditorGUIUtility.labelWidth;
            EditorGUIUtility.labelWidth = Mathf.Clamp(width * 0.38f, 80f, 140f);

            try
            {
                EditorGUILayout.BeginVertical(MotionTimelineTheme.Panel, GUILayout.Height(Mathf.Max(0f, height)), GUILayout.ExpandWidth(true));

                // 1. Fixed Context Header
                GUILayout.Label(TexMotionLocalization.TrLiteral("Timing & Temporal Tools"), MotionTimelineTheme.Heading);
                GUILayout.Label(
                    TexMotionLocalization.TrLiteralFormat("Frame {0} / {1}   |   Range [{2} .. {3}]",
                        window.CurrentFrame + 1, data.Frames, window.RangeStartFrame + 1, window.RangeEndFrame + 1),
                    MotionTimelineTheme.MutedLabel);

                // 2. Fixed Tool Chooser at Top
                state.ActiveTimingTool = (TimelineTimingTool)EditorGUILayout.Popup(
                    TexMotionLocalization.TrLiteral("Tool"),
                    (int)state.ActiveTimingTool,
                    GetLocalizedTimingToolNames());
                GUILayout.Space(6f);

                // 3. Bounded Scrollable Parameter Area
                state.TimingScroll = EditorGUILayout.BeginScrollView(state.TimingScroll, GUILayout.ExpandHeight(true));
                switch (state.ActiveTimingTool)
                {
                    case TimelineTimingTool.Tween:
                        DrawTween(window, data);
                        break;
                    case TimelineTimingTool.Loop:
                        DrawLoop(window, data);
                        break;
                    case TimelineTimingTool.Retime:
                        DrawRetime(window, data);
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

        #region Tool 0: Range Tweening

        private void DrawTween(MotionTimelineEditorWindow window, EditableMotionData data)
        {
            GUILayout.Label(TexMotionLocalization.TrLiteral("Range Keyframe Tweening"), MotionTimelineTheme.SectionHeader);

            EditorGUILayout.BeginVertical(MotionTimelineTheme.Card);
            _interpolationIndex = EditorGUILayout.Popup(
                TexMotionLocalization.TrLiteral("Interpolation"),
                _interpolationIndex,
                GetLocalizedInterpolationNames());
            _includeRoot = EditorGUILayout.Toggle(TexMotionLocalization.TrLiteral("Include Root Pos"), _includeRoot);
            EditorGUILayout.EndVertical();

            GUILayout.Space(6f);

            DrawMaskSelector(window);

            GUILayout.Space(6f);

            int span = window.RangeEndFrame - window.RangeStartFrame;
            if (span < 2)
            {
                EditorGUILayout.HelpBox(TexMotionLocalization.TrLiteral("Range Tweening requires at least 3 frames (In-point, interior frames, Out-point). Extend the range in the timeline below."), MessageType.Warning);
            }
            else
            {
                EditorGUILayout.BeginVertical(MotionTimelineTheme.Card);
                GUILayout.Label(TexMotionLocalization.TrLiteral("Interpolation Scope"), MotionTimelineTheme.Label);
                GUILayout.Label(
                    TexMotionLocalization.TrLiteralFormat("Keyframe Start: Frame {0}\nKeyframe End: Frame {1}\nInterior Resampled: {2} frames",
                        window.RangeStartFrame + 1, window.RangeEndFrame + 1, span - 1),
                    MotionTimelineTheme.MutedLabel);
                EditorGUILayout.EndVertical();
            }
        }

        private EasingType GetSelectedEasing()
        {
            switch (_interpolationIndex)
            {
                case 0: return EasingType.Linear;
                case 1: return EasingType.SmoothStep;
                case 2: return EasingType.EaseIn;
                case 3: return EasingType.EaseOut;
                case 4: return EasingType.SmoothStep; // Hermite cubic spline interpolation (3t^2 - 2t^3)
                default: return EasingType.SmoothStep;
            }
        }

        #endregion

        #region Tool 1: Loop Blending

        private void DrawLoop(MotionTimelineEditorWindow window, EditableMotionData data)
        {
            GUILayout.Label(TexMotionLocalization.TrLiteral("Loop Boundary Blending"), MotionTimelineTheme.SectionHeader);

            EditorGUILayout.BeginVertical(MotionTimelineTheme.Card);
            int maxFrames = Mathf.Max(2, data.Frames / 2);
            _loopBlendFrames = EditorGUILayout.IntSlider(TexMotionLocalization.TrLiteral("Transition Frames"), _loopBlendFrames, 2, maxFrames);
            _matchRootPosition = EditorGUILayout.Toggle(TexMotionLocalization.TrLiteral("Blend Root Pos"), _matchRootPosition);
            EditorGUILayout.EndVertical();

            GUILayout.Space(6f);

            DrawMaskSelector(window);

            GUILayout.Space(6f);

            EditorGUILayout.BeginVertical(MotionTimelineTheme.Card);
            GUILayout.Label(TexMotionLocalization.TrLiteral("Boundary Details"), MotionTimelineTheme.Label);
            int startBlend = Mathf.Max(0, data.Frames - _loopBlendFrames);
            GUILayout.Label(
                TexMotionLocalization.TrLiteralFormat("Cross-blends frames [{0} .. {1}] into Frame 1 for continuous, seamless cyclic playback.",
                    startBlend + 1, data.Frames),
                MotionTimelineTheme.MutedLabel);
            EditorGUILayout.EndVertical();
        }

        #endregion

        #region Tool 2: Range Retiming

        private void DrawRetime(MotionTimelineEditorWindow window, EditableMotionData data)
        {
            GUILayout.Label(TexMotionLocalization.TrLiteral("Range Retiming (Time Warp)"), MotionTimelineTheme.SectionHeader);

            int currentRangeCount = Mathf.Max(1, window.RangeEndFrame - window.RangeStartFrame + 1);

            // Auto-initialize new frame count if the range span changed or uninitialized
            if (_lastObservedSpan != currentRangeCount && _retimeNewFrameCount < 2)
            {
                _retimeNewFrameCount = currentRangeCount;
                _lastObservedSpan = currentRangeCount;
            }

            float fps = data.FrameRate > 0f ? data.FrameRate : 30f;
            float oldDurationSec = (float)currentRangeCount / fps;

            EditorGUILayout.BeginVertical(MotionTimelineTheme.Card);
            _retimeNewFrameCount = EditorGUILayout.IntSlider(TexMotionLocalization.TrLiteral("New Frame Count"), _retimeNewFrameCount, 2, Mathf.Max(currentRangeCount * 3, 120));
            float newDurationSec = (float)_retimeNewFrameCount / fps;
            float speedRatio = (float)currentRangeCount / Mathf.Max(1, _retimeNewFrameCount);

            GUILayout.Space(4f);

            // Quick speed presets
            EditorGUILayout.BeginHorizontal();
            if (GhostButton(TexMotionLocalization.TrLiteral("0.5x Slow"), TexMotionLocalization.TrLiteral("Slow down range to half speed (2x frames)."), 20f)) _retimeNewFrameCount = Mathf.Max(2, Mathf.RoundToInt(currentRangeCount * 2.0f));
            if (GhostButton(TexMotionLocalization.TrLiteral("1.0x Reset"), TexMotionLocalization.TrLiteral("Reset retime count to match original range span."), 20f)) _retimeNewFrameCount = currentRangeCount;
            if (GhostButton(TexMotionLocalization.TrLiteral("1.5x Fast"), TexMotionLocalization.TrLiteral("Speed up range to 1.5x speed."), 20f)) _retimeNewFrameCount = Mathf.Max(2, Mathf.RoundToInt(currentRangeCount / 1.5f));
            if (GhostButton(TexMotionLocalization.TrLiteral("2.0x Double"), TexMotionLocalization.TrLiteral("Double speed of range (half frames)."), 20f)) _retimeNewFrameCount = Mathf.Max(2, Mathf.RoundToInt(currentRangeCount / 2.0f));
            EditorGUILayout.EndHorizontal();
            EditorGUILayout.EndVertical();

            GUILayout.Space(6f);

            // Duration Preview Card
            EditorGUILayout.BeginVertical(MotionTimelineTheme.Card);
            GUILayout.Label(TexMotionLocalization.TrLiteral("Duration Preview"), MotionTimelineTheme.Label);
            GUILayout.Label(
                TexMotionLocalization.TrLiteralFormat("Original: {0} frames ({1:F2}s)\nRetimed:  {2} frames ({3:F2}s)\nSpeed Factor: {4:F2}x\nNew Total Clip: {5} frames",
                    currentRangeCount, oldDurationSec, _retimeNewFrameCount, newDurationSec, speedRatio, data.Frames - currentRangeCount + _retimeNewFrameCount),
                MotionTimelineTheme.MutedLabel);
            EditorGUILayout.EndVertical();

            if (currentRangeCount < 2)
            {
                GUILayout.Space(4f);
                EditorGUILayout.HelpBox(TexMotionLocalization.TrLiteral("Retiming requires at least 2 frames in the selected range."), MessageType.Warning);
            }
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
            int span = end - start;

            switch (state.ActiveTimingTool)
            {
                case TimelineTimingTool.Tween:
                {
                    scopeSummary = TexMotionLocalization.TrLiteralFormat("Affects frames [{0} .. {1}] ({2} frames) / {3}",
                        start + 1, end + 1, span + 1, window.SelectedBodyMask);
                    actionButtonLabel = TexMotionLocalization.TrLiteral("Apply Range Tween");
                    actionTooltip = TexMotionLocalization.TrLiteral("Interpolate interior frames between range start and end keyframes.");
                    actionEnabled = span >= 2 && window.SelectedBodyMask != BodyPartMask.None;
                    executionAction = () =>
                    {
                        EasingType easing = GetSelectedEasing();
                        data.TweenRange(start, end, easing, window.SelectedBodyMask, _includeRoot);
                    };
                    break;
                }
                case TimelineTimingTool.Loop:
                {
                    scopeSummary = TexMotionLocalization.TrLiteralFormat("Affects trailing {0} frames (Entire clip boundary margin) / {1}",
                        _loopBlendFrames, window.SelectedBodyMask);
                    actionButtonLabel = TexMotionLocalization.TrLiteral("Apply Loop Boundary Blend");
                    actionTooltip = TexMotionLocalization.TrLiteral("Blend clip boundaries smoothly for seamless cyclic looping.");
                    actionEnabled = data.Frames >= 4 && window.SelectedBodyMask != BodyPartMask.None;
                    executionAction = () =>
                    {
                        data.BlendLoopBoundary(_loopBlendFrames, window.SelectedBodyMask, _matchRootPosition);
                    };
                    break;
                }
                case TimelineTimingTool.Retime:
                {
                    int currentCount = Mathf.Max(1, span + 1);
                    scopeSummary = TexMotionLocalization.TrLiteralFormat("Resamples frames [{0} .. {1}] ({2} -> {3} frames)",
                        start + 1, end + 1, currentCount, _retimeNewFrameCount);
                    actionButtonLabel = TexMotionLocalization.TrLiteral("Apply Retime Range");
                    actionTooltip = TexMotionLocalization.TrLiteral("Time-warp or stretch the selected frame range to a new duration.");
                    actionEnabled = currentCount >= 2 && _retimeNewFrameCount >= 2;
                    executionAction = () =>
                    {
                        int oldStart = start;
                        data.RetimeRange(oldStart, end, _retimeNewFrameCount);
                        window.RangeEndFrame = oldStart + _retimeNewFrameCount - 1;
                    };
                    break;
                }
                default:
                    scopeSummary = TexMotionLocalization.TrLiteralFormat("Affects frames [{0} .. {1}]", start + 1, end + 1);
                    actionButtonLabel = TexMotionLocalization.TrLiteral("Apply Timing Action");
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

        private static void DrawMaskSelector(MotionTimelineEditorWindow window)
        {
            EditorGUILayout.BeginVertical(MotionTimelineTheme.Card);
            window.SelectedBodyMask = (BodyPartMask)EditorGUILayout.EnumFlagsField(TexMotionLocalization.TrLiteral("Body Parts"), window.SelectedBodyMask);

            EditorGUILayout.BeginHorizontal();
            if (GUILayout.Button(new GUIContent(TexMotionLocalization.TrLiteral("All"), TexMotionLocalization.TrLiteral("Body part mask: All")), MotionTimelineTheme.GhostButton, GUILayout.Height(18f))) window.SelectedBodyMask = BodyPartMask.All;
            if (GUILayout.Button(new GUIContent(TexMotionLocalization.TrLiteral("Upper"), TexMotionLocalization.TrLiteral("Body part mask: Upper body")), MotionTimelineTheme.GhostButton, GUILayout.Height(18f))) window.SelectedBodyMask = BodyPartMask.UpperBody;
            if (GUILayout.Button(new GUIContent(TexMotionLocalization.TrLiteral("Lower"), TexMotionLocalization.TrLiteral("Body part mask: Lower body")), MotionTimelineTheme.GhostButton, GUILayout.Height(18f))) window.SelectedBodyMask = BodyPartMask.LowerBody;
            if (GUILayout.Button(new GUIContent(TexMotionLocalization.TrLiteral("Arms"), TexMotionLocalization.TrLiteral("Body part mask: Arms")), MotionTimelineTheme.GhostButton, GUILayout.Height(18f))) window.SelectedBodyMask = BodyPartMask.Arms;
            if (GUILayout.Button(new GUIContent(TexMotionLocalization.TrLiteral("Legs"), TexMotionLocalization.TrLiteral("Body part mask: Legs")), MotionTimelineTheme.GhostButton, GUILayout.Height(18f))) window.SelectedBodyMask = BodyPartMask.Legs;
            EditorGUILayout.EndHorizontal();
            EditorGUILayout.EndVertical();
        }

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
