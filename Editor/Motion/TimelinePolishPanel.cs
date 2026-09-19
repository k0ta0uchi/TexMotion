using System;
using System.Collections.Generic;
using TexMotion.Editor;
using TexMotion.Runtime.Motion;
using UnityEditor;
using UnityEngine;

namespace TexMotion.Editor.Motion
{
    /// <summary>
    /// Stylized motion polish panel with presets, active filter badges, and progressive filter disclosure.
    /// Follows TIMELINE_UI_REDESIGN.md and DESIGN.md contracts (neutral surfaces, bounded layout, no emojis).
    /// </summary>
    public class TimelinePolishPanel
    {
        private readonly StylizedPolishOptions _fallbackOptions = new StylizedPolishOptions();
        private readonly List<string> _activeFilterNames = new List<string>(9);
        private bool _initialized;

        /// <summary>
        /// Draws the stylized polish workspace inside the inspector area.
        /// Fixed preset and active-filter summary at top, bounded central parameters scroll,
        /// and fixed scope summary with neutral Apply action at bottom.
        /// </summary>
        public void Draw(MotionTimelineEditorWindow window, EditableMotionData data, TimelineWorkspaceState state, float width, float height)
        {
            if (window == null || state == null) return;
            if (data == null || data.Frames == 0)
            {
                GUILayout.Label(TexMotionLocalization.TrLiteral("Load a motion clip to apply stylized polish."), MotionTimelineTheme.MutedLabel);
                return;
            }

            var options = (window.PolishOptions != null) ? window.PolishOptions : _fallbackOptions;

            if (!_initialized)
            {
                _initialized = true;
                SyncPresetToOptions(state.ActivePolishPreset, options);
            }

            float prevLabelWidth = EditorGUIUtility.labelWidth;
            EditorGUIUtility.labelWidth = Mathf.Clamp(width * 0.44f, 110f, 160f);

            try
            {
                EditorGUILayout.BeginVertical(MotionTimelineTheme.Panel, GUILayout.Height(Mathf.Max(0f, height)), GUILayout.ExpandWidth(true));

                // 1. Top Fixed Section: Heading, Preset Bar, and Active Filters Summary
                DrawHeader();
                DrawPresetBar(window, state, options);
                GUILayout.Space(6f);
                DrawActiveFiltersSummary(options, width);
                GUILayout.Space(6f);

                // 2. Central Bounded Parameter Area
                state.PolishScroll = EditorGUILayout.BeginScrollView(state.PolishScroll, GUILayout.ExpandHeight(true));
                DrawFilterDetails(state, options);
                EditorGUILayout.EndScrollView();

                // 3. Fixed Bottom Footer: Scope Summary & Apply Polish Action
                DrawFooter(window, data, options);

                EditorGUILayout.EndVertical();
            }
            finally
            {
                EditorGUIUtility.labelWidth = prevLabelWidth;
            }
        }

        private static void DrawHeader()
        {
            GUILayout.Label(TexMotionLocalization.TrLiteral("Stylized Motion Polish"), MotionTimelineTheme.Heading);
            GUILayout.Label(TexMotionLocalization.TrLiteral("Keyframe-level physics, weight, and timing enhancements."), MotionTimelineTheme.MutedLabel);
            GUILayout.Space(4f);
        }

        private static void DrawPresetBar(MotionTimelineEditorWindow window, TimelineWorkspaceState state, StylizedPolishOptions options)
        {
            EditorGUILayout.BeginHorizontal();
            GUILayout.Label(TexMotionLocalization.TrLiteral("Preset"), MotionTimelineTheme.MutedLabel, GUILayout.Width(46f));

            DrawPresetButton(window, state, options, TimelinePolishPreset.Action, StylizedPolishPreset.SnappyAction, TexMotionLocalization.TrLiteral("Action"), TexMotionLocalization.TrLiteral("Snappy, high-contrast action anime preset."));
            DrawPresetButton(window, state, options, TimelinePolishPreset.Weight, StylizedPolishPreset.RealisticWeight, TexMotionLocalization.TrLiteral("Weight"), TexMotionLocalization.TrLiteral("Realistic weight, landing cushions, and natural settling."));
            DrawPresetButton(window, state, options, TimelinePolishPreset.Subtle, StylizedPolishPreset.SubtlePolish, TexMotionLocalization.TrLiteral("Subtle"), TexMotionLocalization.TrLiteral("Subtle natural polish without changing overall timing."));

            // Custom Preset Button
            bool isCustom = state.ActivePolishPreset == TimelinePolishPreset.Custom;
            GUIStyle customStyle = isCustom ? MotionTimelineTheme.TabActive : MotionTimelineTheme.GhostButton;
            GUIContent customContent = new GUIContent(TexMotionLocalization.TrLiteral("Custom"), TexMotionLocalization.TrLiteral("Custom manual configuration of polish filters."));
            if (GUILayout.Button(customContent, customStyle, GUILayout.Height(22f)))
            {
                state.ActivePolishPreset = TimelinePolishPreset.Custom;
                window.Repaint();
            }

            EditorGUILayout.EndHorizontal();
        }

        private static void DrawPresetButton(
            MotionTimelineEditorWindow window,
            TimelineWorkspaceState state,
            StylizedPolishOptions options,
            TimelinePolishPreset preset,
            StylizedPolishPreset enginePreset,
            string label,
            string tooltip = null)
        {
            bool isSelected = state.ActivePolishPreset == preset;
            GUIStyle style = isSelected ? MotionTimelineTheme.TabActive : MotionTimelineTheme.GhostButton;
            GUIContent content = string.IsNullOrEmpty(tooltip) ? new GUIContent(label) : new GUIContent(label, tooltip);
            if (GUILayout.Button(content, style, GUILayout.Height(22f)))
            {
                state.ActivePolishPreset = preset;
                options.ApplyPreset(enginePreset);
                window.Repaint();
            }
        }

        private void DrawActiveFiltersSummary(StylizedPolishOptions options, float availableWidth)
        {
            UpdateActiveFilterNames(options);

            EditorGUILayout.BeginVertical(MotionTimelineTheme.Card);

            EditorGUILayout.BeginHorizontal();
            GUILayout.Label(TexMotionLocalization.TrLiteral("Active Filters"), MotionTimelineTheme.MutedLabel);
            GUILayout.FlexibleSpace();
            GUILayout.Label(TexMotionLocalization.TrLiteralFormat("{0} of 9 enabled", _activeFilterNames.Count), MotionTimelineTheme.MutedLabel);
            EditorGUILayout.EndHorizontal();

            GUILayout.Space(2f);

            if (_activeFilterNames.Count == 0)
            {
                EditorGUILayout.BeginHorizontal();
                GUILayout.Label(TexMotionLocalization.TrLiteral("No filters enabled"), MotionTimelineTheme.Badge);
                GUILayout.FlexibleSpace();
                EditorGUILayout.EndHorizontal();
            }
            else
            {
                float maxLineWidth = Mathf.Max(120f, availableWidth - 28f);
                float currentLineWidth = 0f;
                const float badgeGap = 4f;

                EditorGUILayout.BeginHorizontal();
                for (int i = 0; i < _activeFilterNames.Count; i++)
                {
                    string name = TexMotionLocalization.TrLiteral(_activeFilterNames[i]);
                    GUIContent content = new GUIContent(name);
                    float badgeWidth = MotionTimelineTheme.SuccessBadge.CalcSize(content).x + 4f;

                    if (currentLineWidth > 0f && (currentLineWidth + badgeWidth > maxLineWidth))
                    {
                        EditorGUILayout.EndHorizontal();
                        EditorGUILayout.BeginHorizontal();
                        currentLineWidth = 0f;
                    }

                    GUILayout.Label(content, MotionTimelineTheme.SuccessBadge);
                    currentLineWidth += badgeWidth + badgeGap;
                }
                GUILayout.FlexibleSpace();
                EditorGUILayout.EndHorizontal();
            }

            EditorGUILayout.EndVertical();
        }

        private void UpdateActiveFilterNames(StylizedPolishOptions options)
        {
            _activeFilterNames.Clear();
            if (options.EnableSnapAndEase) _activeFilterNames.Add("Snap & Ease");
            if (options.StepMode != AnimeStepMode.Off) _activeFilterNames.Add("Frame Stepping");
            if (options.EnableKeyframeDecimator) _activeFilterNames.Add("Decimator");
            if (options.EnableLandingCushion) _activeFilterNames.Add("Landing Cushion");
            if (options.EnableContrapposto) _activeFilterNames.Add("Contrapposto");
            if (options.EnableKinematicChainDelay) _activeFilterNames.Add("Kinematic Drag");
            if (options.EnableOvershoot) _activeFilterNames.Add("Overshoot");
            if (options.EnablePoseExaggeration) _activeFilterNames.Add("Pose Exaggeration");
            if (options.EnableTrajectoryArcSmoothing) _activeFilterNames.Add("Arc Smoother");
        }

        private static void DrawFilterDetails(TimelineWorkspaceState state, StylizedPolishOptions options)
        {
            EditorGUILayout.BeginVertical(MotionTimelineTheme.Card);

            state.PolishShowCustomDetails = EditorGUILayout.Foldout(
                state.PolishShowCustomDetails,
                TexMotionLocalization.TrLiteral("Customize Filters"),
                true,
                EditorStyles.foldoutHeader);

            if (state.PolishShowCustomDetails)
            {
                GUILayout.Space(6f);

                // 4 Category Tabs (2x2 grid for clean wrapping without label clipping)
                EditorGUILayout.BeginHorizontal();
                DrawCategoryTab(state, TimelinePolishCategory.TimingSpacing, TexMotionLocalization.TrLiteral("Timing & Spacing"));
                DrawCategoryTab(state, TimelinePolishCategory.WeightBalance, TexMotionLocalization.TrLiteral("Weight & Balance"));
                EditorGUILayout.EndHorizontal();

                EditorGUILayout.BeginHorizontal();
                DrawCategoryTab(state, TimelinePolishCategory.OverlapDrag, TexMotionLocalization.TrLiteral("Overlap & Drag"));
                DrawCategoryTab(state, TimelinePolishCategory.PoseSilhouette, TexMotionLocalization.TrLiteral("Pose & Silhouette"));
                EditorGUILayout.EndHorizontal();

                GUILayout.Space(8f);
                MotionTimelineTheme.DrawHairline(EditorGUILayout.GetControlRect(false, 1f), MotionTimelineTheme.Graphite);
                GUILayout.Space(8f);

                // Filter parameters with automatic switch to Custom preset on any edit
                EditorGUI.BeginChangeCheck();
                switch (state.ActivePolishCategory)
                {
                    case TimelinePolishCategory.TimingSpacing:
                        DrawTimingSpacingFilters(options);
                        break;
                    case TimelinePolishCategory.WeightBalance:
                        DrawWeightBalanceFilters(options);
                        break;
                    case TimelinePolishCategory.OverlapDrag:
                        DrawOverlapDragFilters(options);
                        break;
                    case TimelinePolishCategory.PoseSilhouette:
                        DrawPoseSilhouetteFilters(options);
                        break;
                }
                if (EditorGUI.EndChangeCheck())
                {
                    state.ActivePolishPreset = TimelinePolishPreset.Custom;
                }
            }

            EditorGUILayout.EndVertical();
        }

        private static void DrawCategoryTab(TimelineWorkspaceState state, TimelinePolishCategory category, string label)
        {
            bool isSelected = state.ActivePolishCategory == category;
            GUIStyle style = isSelected ? MotionTimelineTheme.TabActive : MotionTimelineTheme.TabInactive;
            if (GUILayout.Button(label, style, GUILayout.Height(22f)))
            {
                state.ActivePolishCategory = category;
            }
        }

        private static void DrawTimingSpacingFilters(StylizedPolishOptions options)
        {
            // 1. Snap & Ease
            GUIContent snapContent = new GUIContent(TexMotionLocalization.TrLiteral("Snap & Ease"), TexMotionLocalization.TrLiteral("Snap fast poses with sharp ease-ins and ease-outs."));
            options.EnableSnapAndEase = EditorGUILayout.ToggleLeft(snapContent, options.EnableSnapAndEase, EditorStyles.boldLabel);
            if (options.EnableSnapAndEase)
            {
                EditorGUI.indentLevel++;
                options.SnapIntensity = EditorGUILayout.Slider(TexMotionLocalization.TrLiteral("Snap Intensity"), options.SnapIntensity, 0.1f, 1.0f);
                options.HoldThresholdDegPerSec = EditorGUILayout.Slider(TexMotionLocalization.TrLiteral("Hold Threshold (deg/s)"), options.HoldThresholdDegPerSec, 5.0f, 40.0f);
                EditorGUI.indentLevel--;
            }

            GUILayout.Space(6f);

            // 2. Anime Frame Stepping
            bool stepEnabled = options.StepMode != AnimeStepMode.Off;
            GUIContent stepContent = new GUIContent(TexMotionLocalization.TrLiteral("Anime Frame Stepping"), TexMotionLocalization.TrLiteral("Hold poses across 2 or 3 frames for a stylized anime feel."));
            bool newStepEnabled = EditorGUILayout.ToggleLeft(stepContent, stepEnabled, EditorStyles.boldLabel);
            if (newStepEnabled != stepEnabled)
            {
                options.StepMode = newStepEnabled ? AnimeStepMode.DynamicAnime : AnimeStepMode.Off;
            }
            if (options.StepMode != AnimeStepMode.Off)
            {
                EditorGUI.indentLevel++;
                options.StepMode = (AnimeStepMode)EditorGUILayout.EnumPopup(TexMotionLocalization.TrLiteral("Step Mode"), options.StepMode);
                EditorGUI.indentLevel--;
            }

            GUILayout.Space(6f);

            // 3. Keyframe Decimator
            GUIContent decimatorContent = new GUIContent(TexMotionLocalization.TrLiteral("Keyframe Decimator"), TexMotionLocalization.TrLiteral("Remove redundant dense keyframes while preserving extremes."));
            options.EnableKeyframeDecimator = EditorGUILayout.ToggleLeft(decimatorContent, options.EnableKeyframeDecimator, EditorStyles.boldLabel);
            if (options.EnableKeyframeDecimator)
            {
                EditorGUI.indentLevel++;
                options.DecimatorToleranceDeg = EditorGUILayout.Slider(TexMotionLocalization.TrLiteral("Tolerance (deg)"), options.DecimatorToleranceDeg, 0.5f, 10.0f);
                EditorGUI.indentLevel--;
            }
        }

        private static void DrawWeightBalanceFilters(StylizedPolishOptions options)
        {
            // 4. Landing Cushion
            GUIContent cushionContent = new GUIContent(TexMotionLocalization.TrLiteral("Landing Cushion & Bounce"), TexMotionLocalization.TrLiteral("Simulate downward knee and hip compression upon landing."));
            options.EnableLandingCushion = EditorGUILayout.ToggleLeft(cushionContent, options.EnableLandingCushion, EditorStyles.boldLabel);
            if (options.EnableLandingCushion)
            {
                EditorGUI.indentLevel++;
                options.CushionDepth = EditorGUILayout.Slider(TexMotionLocalization.TrLiteral("Cushion Depth (m)"), options.CushionDepth, 0.01f, 0.08f);
                options.CushionRecoveryFrames = EditorGUILayout.IntSlider(TexMotionLocalization.TrLiteral("Recovery Frames"), options.CushionRecoveryFrames, 2, 8);
                EditorGUI.indentLevel--;
            }

            GUILayout.Space(6f);

            // 5. Contrapposto
            GUIContent contrappostoContent = new GUIContent(TexMotionLocalization.TrLiteral("Contrapposto Booster"), TexMotionLocalization.TrLiteral("Tilt hips and spine to emphasize physical weight shifting."));
            options.EnableContrapposto = EditorGUILayout.ToggleLeft(contrappostoContent, options.EnableContrapposto, EditorStyles.boldLabel);
            if (options.EnableContrapposto)
            {
                EditorGUI.indentLevel++;
                options.ContrappostoWeight = EditorGUILayout.Slider(TexMotionLocalization.TrLiteral("Boost Weight"), options.ContrappostoWeight, 0.2f, 2.5f);
                EditorGUI.indentLevel--;
            }
        }

        private static void DrawOverlapDragFilters(StylizedPolishOptions options)
        {
            // 6. Kinematic Drag
            GUIContent dragContent = new GUIContent(TexMotionLocalization.TrLiteral("Kinematic Drag"), TexMotionLocalization.TrLiteral("Add trailing delay down kinematic limb chains."));
            options.EnableKinematicChainDelay = EditorGUILayout.ToggleLeft(dragContent, options.EnableKinematicChainDelay, EditorStyles.boldLabel);
            if (options.EnableKinematicChainDelay)
            {
                EditorGUI.indentLevel++;
                options.DragDelayFrames = EditorGUILayout.Slider(TexMotionLocalization.TrLiteral("Drag Delay (frames)"), options.DragDelayFrames, 0.2f, 2.0f);
                EditorGUI.indentLevel--;
            }

            GUILayout.Space(6f);

            // 7. Overshoot & Settling
            GUIContent overshootContent = new GUIContent(TexMotionLocalization.TrLiteral("Overshoot & Settling"), TexMotionLocalization.TrLiteral("Add spring-like bounce and settling to sudden stops."));
            options.EnableOvershoot = EditorGUILayout.ToggleLeft(overshootContent, options.EnableOvershoot, EditorStyles.boldLabel);
            if (options.EnableOvershoot)
            {
                EditorGUI.indentLevel++;
                options.OvershootAmount = EditorGUILayout.Slider(TexMotionLocalization.TrLiteral("Overshoot Amount"), options.OvershootAmount, 0.1f, 0.8f);
                options.OvershootFrames = EditorGUILayout.IntSlider(TexMotionLocalization.TrLiteral("Settle Frames"), options.OvershootFrames, 2, 6);
                EditorGUI.indentLevel--;
            }
        }

        private static void DrawPoseSilhouetteFilters(StylizedPolishOptions options)
        {
            // 8. Pose Exaggeration
            GUIContent exagContent = new GUIContent(TexMotionLocalization.TrLiteral("Pose Exaggeration"), TexMotionLocalization.TrLiteral("Scale up peak poses for stronger, dynamic silhouettes."));
            options.EnablePoseExaggeration = EditorGUILayout.ToggleLeft(exagContent, options.EnablePoseExaggeration, EditorStyles.boldLabel);
            if (options.EnablePoseExaggeration)
            {
                EditorGUI.indentLevel++;
                options.ExaggerationScale = EditorGUILayout.Slider(TexMotionLocalization.TrLiteral("Exaggeration Scale"), options.ExaggerationScale, 1.0f, 1.5f);
                EditorGUI.indentLevel--;
            }

            GUILayout.Space(6f);

            // 9. Trajectory Arc Smoother
            GUIContent arcContent = new GUIContent(TexMotionLocalization.TrLiteral("Trajectory Arc Smoother"), TexMotionLocalization.TrLiteral("Smooth hand and foot paths into clean, beautiful spatial arcs."));
            options.EnableTrajectoryArcSmoothing = EditorGUILayout.ToggleLeft(arcContent, options.EnableTrajectoryArcSmoothing, EditorStyles.boldLabel);
            if (options.EnableTrajectoryArcSmoothing)
            {
                EditorGUI.indentLevel++;
                options.ArcSmoothWindow = EditorGUILayout.IntSlider(TexMotionLocalization.TrLiteral("Window Size"), options.ArcSmoothWindow, 3, 9);
                options.ArcBlendWeight = EditorGUILayout.Slider(TexMotionLocalization.TrLiteral("Blend Weight"), options.ArcBlendWeight, 0.2f, 1.0f);
                EditorGUI.indentLevel--;
            }
        }

        private static void DrawFooter(MotionTimelineEditorWindow window, EditableMotionData data, StylizedPolishOptions options)
        {
            GUILayout.Space(4f);
            MotionTimelineTheme.DrawHairline(EditorGUILayout.GetControlRect(false, 1f), MotionTimelineTheme.Graphite);
            GUILayout.Space(4f);

            int start1Based = Mathf.Min(window.RangeStartFrame, window.RangeEndFrame) + 1;
            int end1Based = Mathf.Max(window.RangeStartFrame, window.RangeEndFrame) + 1;

            // Scope summary
            GUILayout.Label(
                TexMotionLocalization.TrLiteralFormat("Affects frames [{0} .. {1}] | Mask: {2}", start1Based, end1Based, window.SelectedBodyMask),
                MotionTimelineTheme.MutedLabel);
            GUILayout.Space(4f);

            // Neutral outlined execution button
            using (new EditorGUI.DisabledScope(data == null || data.Frames == 0))
            {
                GUIContent applyContent = new GUIContent(TexMotionLocalization.TrLiteral("Apply Polish"), TexMotionLocalization.TrLiteral("Apply selected stylized polish filters to the In-Out range."));
                if (GUILayout.Button(applyContent, MotionTimelineTheme.Button, GUILayout.Height(28f)))
                {
                    int start = Mathf.Min(window.RangeStartFrame, window.RangeEndFrame);
                    int end = Mathf.Max(window.RangeStartFrame, window.RangeEndFrame);
                    data.PolishStylizedMotion(start, end, options, window.SelectedBodyMask);
                    window.RefreshAfterEdit();
                }
            }
        }

        private static void SyncPresetToOptions(TimelinePolishPreset preset, StylizedPolishOptions options)
        {
            switch (preset)
            {
                case TimelinePolishPreset.Action:
                    options.ApplyPreset(StylizedPolishPreset.SnappyAction);
                    break;
                case TimelinePolishPreset.Weight:
                    options.ApplyPreset(StylizedPolishPreset.RealisticWeight);
                    break;
                case TimelinePolishPreset.Subtle:
                    options.ApplyPreset(StylizedPolishPreset.SubtlePolish);
                    break;
            }
        }
    }
}
