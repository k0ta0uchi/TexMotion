using System;
using System.Collections.Generic;
using TexMotion.Runtime.Motion;
using UnityEditor;
using UnityEngine;

namespace TexMotion.Editor.Motion
{
    /// <summary>
    /// Focused Single-Joint Inspector and Pose sub-tools for the redesigned Motion Timeline.
    /// Replaces the 20+ vertically stacked joint sliders with single-target inspection,
    /// fast searchable joint picker, and bounded sub-tools (Pose Actions, Palette, Hand & Face, IK Pins).
    /// </summary>
    public class TimelinePosePanel
    {
        private static readonly string[] Categories = { "All", "Spine & Head", "Arms", "Legs" };
        private static readonly string[] SubToolNames = { "Joint", "Actions", "Palette", "Hand & Face", "IK Pins" };
        private static readonly string[] Axes = { "X", "Y", "Z" };

        private static readonly (HandPoseType Type, string Label)[] HandPresets =
        {
            (HandPoseType.NaturalRelaxed, "Relaxed"),
            (HandPoseType.Fist, "Fist"),
            (HandPoseType.OpenPalm, "Open"),
            (HandPoseType.Point, "Point"),
            (HandPoseType.Peace, "Peace"),
            (HandPoseType.KeepFree, "Keep Free")
        };

        private static readonly (FaceEmotionType Type, string Label)[] FacePresets =
        {
            (FaceEmotionType.None, "None"),
            (FaceEmotionType.Smile, "Smile"),
            (FaceEmotionType.Wink, "Wink"),
            (FaceEmotionType.Surprise, "Surprise"),
            (FaceEmotionType.Angry, "Angry"),
            (FaceEmotionType.Smug, "Smug"),
            (FaceEmotionType.AutoDetect, "Auto")
        };

        private static readonly (SmplxJoint Joint, string LimbName)[] IkLimbs =
        {
            (SmplxJoint.L_Wrist, "Left Wrist (Arm)"),
            (SmplxJoint.R_Wrist, "Right Wrist (Arm)"),
            (SmplxJoint.L_Ankle, "Left Ankle (Leg)"),
            (SmplxJoint.R_Ankle, "Right Ankle (Leg)")
        };

        private readonly PosePalette _palette = new PosePalette();
        private int _paletteSlot;
        private float _blend = 1.0f;

        public void Draw(MotionTimelineEditorWindow window, EditableMotionData data, TimelineWorkspaceState state, float width, float height)
        {
            if (window == null || state == null) return;
            if (data == null || data.Frames == 0)
            {
                GUILayout.Label(TexMotionLocalization.TrLiteral("Load a motion to inspect its pose."), MotionTimelineTheme.MutedLabel);
                return;
            }

            int frame = Mathf.Clamp(window.CurrentFrame, 0, Mathf.Max(0, data.Frames - 1));
            float previousLabelWidth = EditorGUIUtility.labelWidth;
            EditorGUIUtility.labelWidth = Mathf.Clamp(width * 0.32f, 70f, 110f);

            try
            {
                EditorGUILayout.BeginVertical(MotionTimelineTheme.Panel, GUILayout.Height(Mathf.Max(0f, height)), GUILayout.ExpandWidth(true));

                // 1. Target Header (Fixed)
                DrawTargetHeader(window, data, frame);
                GUILayout.Space(4f);

                // 2. Sub-tools Bar (Fixed)
                DrawSubtoolsBar(state);
                GUILayout.Space(6f);

                // 3. Bounded Scrollable Area (wraps manipulation controls and tool content)
                state.PoseScroll = EditorGUILayout.BeginScrollView(state.PoseScroll, GUILayout.ExpandHeight(true));
                switch (state.ActivePoseTool)
                {
                    case TimelinePoseTool.Joint:
                        DrawJointTool(window, data, state, frame);
                        break;
                    case TimelinePoseTool.PoseActions:
                        DrawPoseActionsTool(window, data, frame);
                        break;
                    case TimelinePoseTool.Palette:
                        DrawPaletteTool(window, data, frame);
                        break;
                    case TimelinePoseTool.HandFace:
                        DrawHandFaceTool(window, data, state);
                        break;
                    case TimelinePoseTool.IkPins:
                        DrawIkPinsTool(window, data, frame);
                        break;
                }
                EditorGUILayout.EndScrollView();

                EditorGUILayout.EndVertical();
            }
            finally
            {
                EditorGUIUtility.labelWidth = previousLabelWidth;
            }
        }

        #region Target Header & Sub-tools Bar

        private static void DrawTargetHeader(MotionTimelineEditorWindow window, EditableMotionData data, int frame)
        {
            string targetName = window.SelectedJoint.HasValue
                ? MotionTimelineEditorWindow.GetJointFriendlyName(window.SelectedJoint.Value)
                : TexMotionLocalization.TrLiteral("Root Position");
            bool isModified = data.IsFrameModified(frame);

            EditorGUILayout.BeginVertical(MotionTimelineTheme.Card);
            EditorGUILayout.BeginHorizontal();
            GUILayout.Label(TexMotionLocalization.TrLiteralFormat("Target: {0}", targetName), MotionTimelineTheme.Heading);
            GUILayout.FlexibleSpace();
            GUILayout.Label(
                TexMotionLocalization.TrLiteral(isModified ? "Modified" : "Original"),
                isModified ? MotionTimelineTheme.SuccessBadge : MotionTimelineTheme.Badge,
                GUILayout.Height(18f));
            EditorGUILayout.EndHorizontal();

            GUILayout.Space(2f);
            GUILayout.Label(
                TexMotionLocalization.TrLiteralFormat("Frame {0} / {1}  (Frame {2} of {3})", frame, Mathf.Max(0, data.Frames - 1), frame + 1, data.Frames),
                MotionTimelineTheme.MutedLabel);
            EditorGUILayout.EndVertical();
        }

        private static void DrawSubtoolsBar(TimelineWorkspaceState state)
        {
            EditorGUILayout.BeginHorizontal();
            for (int i = 0; i < SubToolNames.Length; i++)
            {
                var tool = (TimelinePoseTool)i;
                bool isSelected = state.ActivePoseTool == tool;
                GUIContent content = new GUIContent(TexMotionLocalization.TrLiteral(SubToolNames[i]), GetSubtoolTooltip(tool));
                if (GUILayout.Button(content, isSelected ? MotionTimelineTheme.TabActive : MotionTimelineTheme.TabInactive, GUILayout.Height(24f), GUILayout.ExpandWidth(true)))
                {
                    state.ActivePoseTool = tool;
                    GUI.FocusControl(null);
                }
            }
            EditorGUILayout.EndHorizontal();
        }

        private static string GetSubtoolTooltip(TimelinePoseTool tool)
        {
            switch (tool)
            {
                case TimelinePoseTool.Joint:
                    return TexMotionLocalization.TrLiteral("Single-joint selection and Euler angle editing.");
                case TimelinePoseTool.PoseActions:
                    return TexMotionLocalization.TrLiteral("Clipboard operations, pose mirroring, smoothing, and reset.");
                case TimelinePoseTool.Palette:
                    return TexMotionLocalization.TrLiteral("Quick-save and blend up to 8 pose snapshots.");
                case TimelinePoseTool.HandFace:
                    return TexMotionLocalization.TrLiteral("Expressive hand gesture and facial emotion presets.");
                case TimelinePoseTool.IkPins:
                    return TexMotionLocalization.TrLiteral("Pin hands and feet in world space while editing other joints.");
                default:
                    return string.Empty;
            }
        }

        #endregion

        #region Tool 1: Joint Manipulation & Picker

        private static void DrawJointTool(MotionTimelineEditorWindow window, EditableMotionData data, TimelineWorkspaceState state, int frame)
        {
            DrawJointSelector(window, state);
            GUILayout.Space(6f);

            EditorGUILayout.BeginVertical(MotionTimelineTheme.Card);
            if (window.SelectedJoint.HasValue)
            {
                DrawJointRotationControls(window, data, frame, window.SelectedJoint.Value);
            }
            else
            {
                DrawRootPositionControls(window, data, frame);
            }
            EditorGUILayout.EndVertical();
        }

        private static void DrawJointSelector(MotionTimelineEditorWindow window, TimelineWorkspaceState state)
        {
            EditorGUILayout.BeginVertical(MotionTimelineTheme.Card);
            GUILayout.Label(TexMotionLocalization.TrLiteral("Joint Selector"), MotionTimelineTheme.SectionHeader);
            GUILayout.Space(3f);

            // Category filter row
            EditorGUILayout.BeginHorizontal();
            for (int c = 0; c < Categories.Length; c++)
            {
                bool isSel = state.JointCategoryFilter == c;
                string catTooltip = GetCategoryTooltip(c);
                GUIContent catContent = new GUIContent(TexMotionLocalization.TrLiteral(Categories[c]), catTooltip);
                if (GUILayout.Button(catContent, isSel ? MotionTimelineTheme.TabActive : MotionTimelineTheme.GhostButton, GUILayout.Height(20f), GUILayout.ExpandWidth(true)))
                {
                    state.JointCategoryFilter = c;
                }
            }
            EditorGUILayout.EndHorizontal();
            GUILayout.Space(3f);

            // Search filter row
            EditorGUILayout.BeginHorizontal();
            GUILayout.Label(new GUIContent(TexMotionLocalization.TrLiteral("Search"), TexMotionLocalization.TrLiteral("Filter joints by name")), MotionTimelineTheme.Label, GUILayout.Width(46f));
            state.JointSearchFilter = EditorGUILayout.TextField(state.JointSearchFilter ?? string.Empty);
            if (!string.IsNullOrEmpty(state.JointSearchFilter))
            {
                if (GUILayout.Button("X", MotionTimelineTheme.GhostButton, GUILayout.Width(20f), GUILayout.Height(18f)))
                {
                    state.JointSearchFilter = string.Empty;
                    GUI.FocusControl(null);
                }
            }
            EditorGUILayout.EndHorizontal();
            GUILayout.Space(3f);

            // Target dropdown & quick Root button
            EditorGUILayout.BeginHorizontal();
            GUILayout.Label(TexMotionLocalization.TrLiteral("Target"), MotionTimelineTheme.Label, GUILayout.Width(46f));
            DrawTargetDropdown(window, state);

            bool isRoot = !window.SelectedJoint.HasValue;
            GUIContent rootContent = new GUIContent(TexMotionLocalization.TrLiteral("Root"), TexMotionLocalization.TrLiteral("Select root joint"));
            if (GUILayout.Button(rootContent, isRoot ? MotionTimelineTheme.TabActive : MotionTimelineTheme.GhostButton, GUILayout.Width(44f), GUILayout.Height(19f)))
            {
                window.SelectedJoint = null;
                window.RefreshPreview();
            }
            EditorGUILayout.EndHorizontal();

            EditorGUILayout.EndVertical();
        }

        private static string GetCategoryTooltip(int categoryIndex)
        {
            switch (categoryIndex)
            {
                case 0: return TexMotionLocalization.TrLiteral("Filter joints: All");
                case 1: return TexMotionLocalization.TrLiteral("Filter joints: Spine & Head");
                case 2: return TexMotionLocalization.TrLiteral("Filter joints: Arms");
                case 3: return TexMotionLocalization.TrLiteral("Filter joints: Legs");
                default: return string.Empty;
            }
        }

        private static void DrawTargetDropdown(MotionTimelineEditorWindow window, TimelineWorkspaceState state)
        {
            var targets = new List<SmplxJoint?>();
            var names = new List<string>();

            string search = (state.JointSearchFilter ?? string.Empty).Trim();

            if ((state.JointCategoryFilter == 0 || state.JointCategoryFilter == 1) && Matches("Root Position", search))
            {
                targets.Add(null);
                names.Add(TexMotionLocalization.TrLiteral("Root Position"));
            }

            for (int i = 0; i < SmplxJointDefinitions.JointCount; i++)
            {
                var joint = (SmplxJoint)i;
                string name = MotionTimelineEditorWindow.GetJointFriendlyName(joint);
                if (!InCategory(joint, state.JointCategoryFilter)) continue;
                if (!string.IsNullOrEmpty(search) && !Matches(name, search) && !Matches(joint.ToString(), search)) continue;

                targets.Add(joint);
                names.Add(name);
            }

            int current = targets.IndexOf(window.SelectedJoint);
            int popupIndex;
            string[] displayNames;

            if (current >= 0)
            {
                popupIndex = current;
                displayNames = names.ToArray();
            }
            else
            {
                // If current selection is filtered out by search/category, insert it as option 0
                string currentName = window.SelectedJoint.HasValue
                    ? MotionTimelineEditorWindow.GetJointFriendlyName(window.SelectedJoint.Value)
                    : TexMotionLocalization.TrLiteral("Root Position");
                targets.Insert(0, window.SelectedJoint);
                names.Insert(0, $"{currentName} (current)");
                popupIndex = 0;
                displayNames = names.ToArray();
            }

            if (displayNames.Length == 0)
            {
                EditorGUILayout.LabelField(TexMotionLocalization.TrLiteral("No matching joints"), MotionTimelineTheme.MutedLabel);
                return;
            }

            EditorGUI.BeginChangeCheck();
            int selected = EditorGUILayout.Popup(popupIndex, displayNames);
            if (EditorGUI.EndChangeCheck() && selected >= 0 && selected < targets.Count)
            {
                window.SelectedJoint = targets[selected];
                state.ActivePoseTool = TimelinePoseTool.Joint;
                window.RefreshPreview();
            }
        }

        private static bool Matches(string value, string search)
        {
            if (string.IsNullOrEmpty(search)) return true;
            return value.IndexOf(search, StringComparison.OrdinalIgnoreCase) >= 0;
        }

        private static bool InCategory(SmplxJoint joint, int category)
        {
            if (category == 0) return true;
            switch (joint)
            {
                case SmplxJoint.Pelvis:
                case SmplxJoint.Spine1:
                case SmplxJoint.Spine2:
                case SmplxJoint.Spine3:
                case SmplxJoint.Neck:
                case SmplxJoint.Head:
                    return category == 1;

                case SmplxJoint.L_Collar:
                case SmplxJoint.R_Collar:
                case SmplxJoint.L_Shoulder:
                case SmplxJoint.R_Shoulder:
                case SmplxJoint.L_Elbow:
                case SmplxJoint.R_Elbow:
                case SmplxJoint.L_Wrist:
                case SmplxJoint.R_Wrist:
                    return category == 2;

                case SmplxJoint.L_Hip:
                case SmplxJoint.R_Hip:
                case SmplxJoint.L_Knee:
                case SmplxJoint.R_Knee:
                case SmplxJoint.L_Ankle:
                case SmplxJoint.R_Ankle:
                case SmplxJoint.L_Foot:
                case SmplxJoint.R_Foot:
                    return category == 3;

                default:
                    return false;
            }
        }

        private static void DrawRootPositionControls(MotionTimelineEditorWindow window, EditableMotionData data, int frame)
        {
            GUILayout.Label(TexMotionLocalization.TrLiteral("Root Position (metres)"), MotionTimelineTheme.SectionHeader);
            GUILayout.Space(4f);

            Vector3 pos = data.GetRootPosition(frame);
            EditorGUI.BeginChangeCheck();

            for (int axis = 0; axis < 3; axis++)
            {
                EditorGUILayout.BeginHorizontal();
                GUILayout.Label(Axes[axis], MotionTimelineTheme.Label, GUILayout.Width(16f));
                pos[axis] = EditorGUILayout.Slider(pos[axis], -5f, 5f);
                if (GUILayout.Button(new GUIContent("-0.1", "Move -0.1m"), MotionTimelineTheme.GhostButton, GUILayout.Width(34f), GUILayout.Height(18f)))
                {
                    pos[axis] -= 0.1f;
                    GUI.changed = true;
                }
                if (GUILayout.Button(new GUIContent("+0.1", "Move +0.1m"), MotionTimelineTheme.GhostButton, GUILayout.Width(34f), GUILayout.Height(18f)))
                {
                    pos[axis] += 0.1f;
                    GUI.changed = true;
                }
                EditorGUILayout.EndHorizontal();
            }

            if (EditorGUI.EndChangeCheck())
            {
                data.RecordUndo($"Move Root on Frame {frame}");
                data.SetRootPosition(frame, pos);
                window.RefreshPreview();
            }

            GUILayout.Space(6f);
            EditorGUILayout.BeginHorizontal();
            GUIContent snapGroundContent = new GUIContent(TexMotionLocalization.TrLiteral("Snap Ground (Y=0)"), TexMotionLocalization.TrLiteral("Snap root position vertically to ground level (Y=0)."));
            if (GUILayout.Button(snapGroundContent, MotionTimelineTheme.GhostButton, GUILayout.Height(24f), GUILayout.ExpandWidth(true)))
            {
                data.RecordUndo($"Snap Root Ground on Frame {frame}");
                pos.y = 0f;
                data.SetRootPosition(frame, pos);
                window.RefreshPreview();
            }
            GUIContent zeroPosContent = new GUIContent(TexMotionLocalization.TrLiteral("Zero Position"), TexMotionLocalization.TrLiteral("Reset root position to origin (0, 0, 0)."));
            if (GUILayout.Button(zeroPosContent, MotionTimelineTheme.GhostButton, GUILayout.Height(24f), GUILayout.ExpandWidth(true)))
            {
                data.RecordUndo($"Zero Root Position on Frame {frame}");
                data.SetRootPosition(frame, Vector3.zero);
                window.RefreshPreview();
            }
            GUIContent resetPosContent = new GUIContent(TexMotionLocalization.TrLiteral("Reset to Original"), TexMotionLocalization.TrLiteral("Revert root position to original motion value."));
            if (GUILayout.Button(resetPosContent, MotionTimelineTheme.GhostButton, GUILayout.Height(24f), GUILayout.ExpandWidth(true)))
            {
                data.ResetFrameToOriginal(frame, BodyPartMask.Pelvis);
                window.RefreshPreview();
            }
            EditorGUILayout.EndHorizontal();
        }

        private static void DrawJointRotationControls(MotionTimelineEditorWindow window, EditableMotionData data, int frame, SmplxJoint joint)
        {
            string jointName = MotionTimelineEditorWindow.GetJointFriendlyName(joint);
            GUILayout.Label($"{jointName} ({TexMotionLocalization.TrLiteral("Joint Rotation (Euler Degrees)")})", MotionTimelineTheme.SectionHeader);
            GUILayout.Space(4f);

            Vector3 euler = data.GetJointEuler(frame, joint);
            EditorGUI.BeginChangeCheck();

            for (int axis = 0; axis < 3; axis++)
            {
                EditorGUILayout.BeginHorizontal();
                GUILayout.Label(Axes[axis], MotionTimelineTheme.Label, GUILayout.Width(16f));
                euler[axis] = EditorGUILayout.Slider(euler[axis], -180f, 180f);
                if (GUILayout.Button(new GUIContent("-5", "Nudge -5 degrees"), MotionTimelineTheme.GhostButton, GUILayout.Width(30f), GUILayout.Height(18f)))
                {
                    euler[axis] = Mathf.DeltaAngle(0f, euler[axis] - 5f);
                    GUI.changed = true;
                }
                if (GUILayout.Button(new GUIContent("+5", "Nudge +5 degrees"), MotionTimelineTheme.GhostButton, GUILayout.Width(30f), GUILayout.Height(18f)))
                {
                    euler[axis] = Mathf.DeltaAngle(0f, euler[axis] + 5f);
                    GUI.changed = true;
                }
                EditorGUILayout.EndHorizontal();
            }

            if (EditorGUI.EndChangeCheck())
            {
                data.RecordUndo($"Rotate {jointName} on Frame {frame}");
                data.SetJointEuler(frame, joint, euler);
                window.RefreshPreview();
            }

            GUILayout.Space(6f);
            EditorGUILayout.BeginHorizontal();
            GUIContent zeroRotContent = new GUIContent(TexMotionLocalization.TrLiteral("Zero Rotation"), TexMotionLocalization.TrLiteral("Reset Euler angles of this joint to (0, 0, 0)."));
            if (GUILayout.Button(zeroRotContent, MotionTimelineTheme.GhostButton, GUILayout.Height(24f), GUILayout.ExpandWidth(true)))
            {
                data.RecordUndo($"Zero Rotation on {jointName} Frame {frame}");
                data.SetJointEuler(frame, joint, Vector3.zero);
                window.RefreshPreview();
            }
            GUIContent resetRotContent = new GUIContent(TexMotionLocalization.TrLiteral("Reset to Original"), TexMotionLocalization.TrLiteral("Revert this joint rotation to original motion value."));
            if (GUILayout.Button(resetRotContent, MotionTimelineTheme.GhostButton, GUILayout.Height(24f), GUILayout.ExpandWidth(true)))
            {
                data.ResetJointToOriginal(frame, joint);
                window.RefreshPreview();
            }
            EditorGUILayout.EndHorizontal();
        }

        #endregion

        #region Tool 2: Pose Actions

        private static void DrawPoseActionsTool(MotionTimelineEditorWindow window, EditableMotionData data, int frame)
        {
            EditorGUILayout.BeginVertical(MotionTimelineTheme.Card);
            GUILayout.Label(TexMotionLocalization.TrLiteral("Pose Actions"), MotionTimelineTheme.SectionHeader);
            GUILayout.Space(4f);

            DrawBodyMaskSelector(window);
            GUILayout.Space(6f);

            BodyPartMask mask = window.SelectedBodyMask;
            bool hasClipboard = EditableMotionData.HasClipboardData;
            bool canSmooth = frame > 0 && frame < data.Frames - 1;
            bool hasMask = mask != BodyPartMask.None;

            EditorGUILayout.BeginHorizontal();
            GUIContent copyContent = new GUIContent(TexMotionLocalization.TrLiteral("Copy Pose"), TexMotionLocalization.TrLiteral("Copy current frame pose to clipboard."));
            if (GUILayout.Button(copyContent, MotionTimelineTheme.GhostButton, GUILayout.Height(26f), GUILayout.ExpandWidth(true)))
            {
                data.CopyFramePose(frame);
                window.Repaint();
            }
            using (new EditorGUI.DisabledScope(!hasClipboard || !hasMask))
            {
                GUIContent pasteContent = new GUIContent(TexMotionLocalization.TrLiteral("Paste Pose"), TexMotionLocalization.TrLiteral("Paste pose from clipboard to current frame."));
                if (GUILayout.Button(pasteContent, MotionTimelineTheme.GhostButton, GUILayout.Height(26f), GUILayout.ExpandWidth(true)))
                {
                    data.PasteFramePose(frame, mask);
                    window.RefreshPreview();
                }
            }
            EditorGUILayout.EndHorizontal();

            if (hasClipboard)
            {
                GUILayout.Space(2f);
                GUILayout.Label(TexMotionLocalization.TrLiteral("Paste clipboard pose with selected body mask"), MotionTimelineTheme.MutedLabel);
            }

            GUILayout.Space(6f);

            EditorGUILayout.BeginHorizontal();
            using (new EditorGUI.DisabledScope(!hasMask))
            {
                GUIContent mirrorContent = new GUIContent(TexMotionLocalization.TrLiteral("Mirror Pose"), TexMotionLocalization.TrLiteral("Mirror left and right limbs across the sagittal plane."));
                if (GUILayout.Button(mirrorContent, MotionTimelineTheme.GhostButton, GUILayout.Height(26f), GUILayout.ExpandWidth(true)))
                {
                    ApplyMasked(data, frame, mask, () => data.MirrorFrame(frame));
                    window.RefreshPreview();
                }
            }
            using (new EditorGUI.DisabledScope(!canSmooth || !hasMask))
            {
                GUIContent smoothContent = new GUIContent(TexMotionLocalization.TrLiteral("Smooth Frame"), TexMotionLocalization.TrLiteral("Smooth current frame with adjacent frames."));
                if (GUILayout.Button(smoothContent, MotionTimelineTheme.GhostButton, GUILayout.Height(26f), GUILayout.ExpandWidth(true)))
                {
                    data.SmoothFrame(frame, mask);
                    window.RefreshPreview();
                }
            }
            EditorGUILayout.EndHorizontal();

            GUILayout.Space(4f);

            EditorGUILayout.BeginHorizontal();
            using (new EditorGUI.DisabledScope(!hasMask))
            {
                GUIContent resetFrameContent = new GUIContent(TexMotionLocalization.TrLiteral("Reset Frame"), TexMotionLocalization.TrLiteral("Revert current frame to original motion data."));
                if (GUILayout.Button(resetFrameContent, MotionTimelineTheme.GhostButton, GUILayout.Height(26f), GUILayout.ExpandWidth(true)))
                {
                    data.ResetFrameToOriginal(frame, mask);
                    window.RefreshPreview();
                }
                GUIContent tPoseContent = new GUIContent(TexMotionLocalization.TrLiteral("T-Pose"), TexMotionLocalization.TrLiteral("Reset all joints to canonical humanoid T-Pose."));
                if (GUILayout.Button(tPoseContent, MotionTimelineTheme.GhostButton, GUILayout.Height(26f), GUILayout.ExpandWidth(true)))
                {
                    ApplyMasked(data, frame, mask, () => data.SetFrameToTPose(frame));
                    window.RefreshPreview();
                }
            }
            EditorGUILayout.EndHorizontal();

            GUILayout.Space(8f);
            EditorGUILayout.HelpBox(
                TexMotionLocalization.TrLiteral("Smooth current frame with adjacent frames"),
                MessageType.None);

            EditorGUILayout.EndVertical();
        }

        private static void DrawBodyMaskSelector(MotionTimelineEditorWindow window)
        {
            window.SelectedBodyMask = (BodyPartMask)EditorGUILayout.EnumFlagsField(TexMotionLocalization.TrLiteral("Body Mask"), window.SelectedBodyMask);
            GUILayout.Space(2f);

            EditorGUILayout.BeginHorizontal();
            if (GUILayout.Button(new GUIContent(TexMotionLocalization.TrLiteral("All"), TexMotionLocalization.TrLiteral("Body part mask: All")), window.SelectedBodyMask == BodyPartMask.All ? MotionTimelineTheme.TabActive : MotionTimelineTheme.GhostButton, GUILayout.Height(18f)))
                window.SelectedBodyMask = BodyPartMask.All;
            if (GUILayout.Button(new GUIContent(TexMotionLocalization.TrLiteral("Upper"), TexMotionLocalization.TrLiteral("Body part mask: Upper body")), window.SelectedBodyMask == BodyPartMask.UpperBody ? MotionTimelineTheme.TabActive : MotionTimelineTheme.GhostButton, GUILayout.Height(18f)))
                window.SelectedBodyMask = BodyPartMask.UpperBody;
            if (GUILayout.Button(new GUIContent(TexMotionLocalization.TrLiteral("Lower"), TexMotionLocalization.TrLiteral("Body part mask: Lower body")), window.SelectedBodyMask == BodyPartMask.LowerBody ? MotionTimelineTheme.TabActive : MotionTimelineTheme.GhostButton, GUILayout.Height(18f)))
                window.SelectedBodyMask = BodyPartMask.LowerBody;
            if (GUILayout.Button(new GUIContent(TexMotionLocalization.TrLiteral("Arms"), TexMotionLocalization.TrLiteral("Body part mask: Arms")), window.SelectedBodyMask == BodyPartMask.Arms ? MotionTimelineTheme.TabActive : MotionTimelineTheme.GhostButton, GUILayout.Height(18f)))
                window.SelectedBodyMask = BodyPartMask.Arms;
            if (GUILayout.Button(new GUIContent(TexMotionLocalization.TrLiteral("Legs"), TexMotionLocalization.TrLiteral("Body part mask: Legs")), window.SelectedBodyMask == BodyPartMask.Legs ? MotionTimelineTheme.TabActive : MotionTimelineTheme.GhostButton, GUILayout.Height(18f)))
                window.SelectedBodyMask = BodyPartMask.Legs;
            if (GUILayout.Button(TexMotionLocalization.TrLiteral("None"), window.SelectedBodyMask == BodyPartMask.None ? MotionTimelineTheme.TabActive : MotionTimelineTheme.GhostButton, GUILayout.Height(18f)))
                window.SelectedBodyMask = BodyPartMask.None;
            EditorGUILayout.EndHorizontal();
        }

        private static void ApplyMasked(EditableMotionData data, int frame, BodyPartMask mask, Action operation)
        {
            if (mask == BodyPartMask.All)
            {
                operation();
                return;
            }

            Vector3 root = data.GetRootPosition(frame);
            var rotations = new Quaternion[SmplxJointDefinitions.JointCount];
            for (int i = 0; i < rotations.Length; i++)
            {
                rotations[i] = data.GetJointRotation(frame, (SmplxJoint)i);
            }

            operation();

            if ((mask & BodyPartMask.Pelvis) == 0)
            {
                data.SetRootPosition(frame, root);
            }

            for (int i = 0; i < rotations.Length; i++)
            {
                var joint = (SmplxJoint)i;
                if (!BodyPartMaskUtility.ContainsJoint(mask, joint))
                {
                    data.SetJointRotation(frame, joint, rotations[i]);
                }
            }
        }

        #endregion

        #region Tool 3: Pose Palette

        private void DrawPaletteTool(MotionTimelineEditorWindow window, EditableMotionData data, int frame)
        {
            EditorGUILayout.BeginVertical(MotionTimelineTheme.Card);
            GUILayout.Label(TexMotionLocalization.TrLiteral("Pose Palette (8 Slots)"), MotionTimelineTheme.SectionHeader);
            GUILayout.Space(4f);

            DrawBodyMaskSelector(window);
            GUILayout.Space(6f);

            for (int row = 0; row < 2; row++)
            {
                EditorGUILayout.BeginHorizontal();
                for (int col = 0; col < 4; col++)
                {
                    int slot = row * 4 + col;
                    bool isOccupied = _palette.HasPose(slot);
                    bool isSelected = _paletteSlot == slot;
                    string status = TexMotionLocalization.TrLiteral(isOccupied ? "Saved" : "Empty");
                    string slotLabel = $"{TexMotionLocalization.TrLiteralFormat("Slot {0}", slot + 1)}\n{status}";

                    if (GUILayout.Toggle(isSelected, slotLabel, isSelected ? MotionTimelineTheme.TabActive : MotionTimelineTheme.GhostButton, GUILayout.Height(34f), GUILayout.ExpandWidth(true)))
                    {
                        _paletteSlot = slot;
                    }
                }
                EditorGUILayout.EndHorizontal();
                GUILayout.Space(3f);
            }

            var currentSlot = _palette.GetSlot(_paletteSlot);
            string slotStatus = currentSlot != null && currentSlot.IsOccupied ? currentSlot.Name : TexMotionLocalization.TrLiteral("Empty");
            GUILayout.Label($"{TexMotionLocalization.TrLiteralFormat("Slot {0}", _paletteSlot + 1)}: {slotStatus}", MotionTimelineTheme.MutedLabel);
            GUILayout.Space(4f);

            string captureText = _palette.HasPose(_paletteSlot) ? TexMotionLocalization.TrLiteral("Capture Current Pose") : TexMotionLocalization.TrLiteral("Capture Current Pose");
            GUIContent captureContent = new GUIContent(captureText, TexMotionLocalization.TrLiteral("Capture current frame pose into this slot."));
            if (GUILayout.Button(captureContent, MotionTimelineTheme.GhostButton, GUILayout.Height(24f), GUILayout.ExpandWidth(true)))
            {
                _palette.Capture(_paletteSlot, data, frame);
                window.Repaint();
            }

            GUILayout.Space(6f);
            _blend = EditorGUILayout.Slider(TexMotionLocalization.TrLiteral("Blend Weight"), _blend, 0f, 1f);
            GUILayout.Space(4f);

            bool hasPose = _palette.HasPose(_paletteSlot);
            bool canApply = hasPose && window.SelectedBodyMask != BodyPartMask.None && _blend > 0f;

            EditorGUILayout.BeginHorizontal();
            using (new EditorGUI.DisabledScope(!canApply))
            {
                GUIContent applyCurrentContent = new GUIContent(TexMotionLocalization.TrLiteral("Apply to Current Frame"), TexMotionLocalization.TrLiteral("Apply or blend this saved pose into the current frame."));
                if (GUILayout.Button(applyCurrentContent, MotionTimelineTheme.GhostButton, GUILayout.Height(24f), GUILayout.ExpandWidth(true)))
                {
                    _palette.Apply(_paletteSlot, data, frame, _blend, window.SelectedBodyMask);
                    window.RefreshPreview();
                }
                GUIContent applyRangeContent = new GUIContent(TexMotionLocalization.TrLiteral("Apply to Range"), TexMotionLocalization.TrLiteral("Apply or blend this saved pose across the selected In-Out range."));
                if (GUILayout.Button(applyRangeContent, MotionTimelineTheme.GhostButton, GUILayout.Height(24f), GUILayout.ExpandWidth(true)))
                {
                    int start = Mathf.Min(window.RangeStartFrame, window.RangeEndFrame);
                    int end = Mathf.Max(window.RangeStartFrame, window.RangeEndFrame);
                    for (int f = start; f <= end; f++)
                    {
                        _palette.Apply(_paletteSlot, data, f, _blend, window.SelectedBodyMask);
                    }
                    window.RefreshPreview();
                }
            }
            using (new EditorGUI.DisabledScope(!hasPose))
            {
                GUIContent clearContent = new GUIContent(TexMotionLocalization.TrLiteral("Clear Slot"), TexMotionLocalization.TrLiteral("Clear saved pose from this slot."));
                if (GUILayout.Button(clearContent, MotionTimelineTheme.GhostButton, GUILayout.Width(70f), GUILayout.Height(24f)))
                {
                    _palette.ClearSlot(_paletteSlot);
                    window.Repaint();
                }
            }
            EditorGUILayout.EndHorizontal();

            EditorGUILayout.EndVertical();
        }

        #endregion

        #region Tool 4: Hand & Face Assistance

        private static void DrawHandFaceTool(MotionTimelineEditorWindow window, EditableMotionData data, TimelineWorkspaceState state)
        {
            EditorGUILayout.BeginVertical(MotionTimelineTheme.Card);
            GUILayout.Label(TexMotionLocalization.TrLiteral("Hand Gestures"), MotionTimelineTheme.SectionHeader);
            GUILayout.Space(4f);

            EditorGUI.BeginChangeCheck();
            var hand = (HandPoseType)EditorGUILayout.EnumPopup(TexMotionLocalization.TrLiteral("Hand Pose"), data.HandPose);
            if (EditorGUI.EndChangeCheck())
            {
                data.HandPose = hand;
                window.RefreshPreview();
            }

            GUILayout.Space(2f);
            EditorGUILayout.BeginHorizontal();
            for (int i = 0; i < 3; i++)
            {
                var preset = HandPresets[i];
                bool isSel = data.HandPose == preset.Type;
                GUIContent content = new GUIContent(TexMotionLocalization.TrLiteral(preset.Label), TexMotionLocalization.TrLiteral("Apply selected hand preset to current frame."));
                if (GUILayout.Button(content, isSel ? MotionTimelineTheme.TabActive : MotionTimelineTheme.GhostButton, GUILayout.Height(20f), GUILayout.ExpandWidth(true)))
                {
                    data.HandPose = preset.Type;
                    window.RefreshPreview();
                }
            }
            EditorGUILayout.EndHorizontal();
            EditorGUILayout.BeginHorizontal();
            for (int i = 3; i < HandPresets.Length; i++)
            {
                var preset = HandPresets[i];
                bool isSel = data.HandPose == preset.Type;
                GUIContent content = new GUIContent(TexMotionLocalization.TrLiteral(preset.Label), TexMotionLocalization.TrLiteral("Apply selected hand preset to current frame."));
                if (GUILayout.Button(content, isSel ? MotionTimelineTheme.TabActive : MotionTimelineTheme.GhostButton, GUILayout.Height(20f), GUILayout.ExpandWidth(true)))
                {
                    data.HandPose = preset.Type;
                    window.RefreshPreview();
                }
            }
            EditorGUILayout.EndHorizontal();

            GUILayout.Space(10f);
            GUILayout.Label(TexMotionLocalization.TrLiteral("Facial Expressions"), MotionTimelineTheme.SectionHeader);
            GUILayout.Space(4f);

            EditorGUI.BeginChangeCheck();
            var face = (FaceEmotionType)EditorGUILayout.EnumPopup(TexMotionLocalization.TrLiteral("Face Emotion"), data.FaceEmotion);
            if (EditorGUI.EndChangeCheck())
            {
                data.FaceEmotion = face;
                window.RefreshPreview();
            }

            GUILayout.Space(2f);
            EditorGUILayout.BeginHorizontal();
            for (int i = 0; i < 4; i++)
            {
                var preset = FacePresets[i];
                bool isSel = data.FaceEmotion == preset.Type;
                GUIContent content = new GUIContent(TexMotionLocalization.TrLiteral(preset.Label), TexMotionLocalization.TrLiteral("Apply selected facial emotion preset to current frame."));
                if (GUILayout.Button(content, isSel ? MotionTimelineTheme.TabActive : MotionTimelineTheme.GhostButton, GUILayout.Height(20f), GUILayout.ExpandWidth(true)))
                {
                    data.FaceEmotion = preset.Type;
                    window.RefreshPreview();
                }
            }
            EditorGUILayout.EndHorizontal();
            EditorGUILayout.BeginHorizontal();
            for (int i = 4; i < FacePresets.Length; i++)
            {
                var preset = FacePresets[i];
                bool isSel = data.FaceEmotion == preset.Type;
                GUIContent content = new GUIContent(TexMotionLocalization.TrLiteral(preset.Label), TexMotionLocalization.TrLiteral("Apply selected facial emotion preset to current frame."));
                if (GUILayout.Button(content, isSel ? MotionTimelineTheme.TabActive : MotionTimelineTheme.GhostButton, GUILayout.Height(20f), GUILayout.ExpandWidth(true)))
                {
                    data.FaceEmotion = preset.Type;
                    window.RefreshPreview();
                }
            }
            EditorGUILayout.EndHorizontal();

            if (face != FaceEmotionType.None && face != FaceEmotionType.AutoDetect)
            {
                GUILayout.Space(4f);
                EditorGUI.BeginChangeCheck();
                float intensity = EditorGUILayout.Slider(TexMotionLocalization.TrLiteral("Emotion Intensity"), data.EmotionIntensity, 0.1f, 1f);
                if (EditorGUI.EndChangeCheck())
                {
                    data.EmotionIntensity = intensity;
                    window.RefreshPreview();
                }
            }

            if (data.SourceVideoData?.FaceTrack != null && data.SourceVideoData.FaceTrack.shapes != null && data.SourceVideoData.FaceTrack.shapes.Length > 0)
            {
                GUILayout.Space(6f);
                EditorGUILayout.BeginHorizontal();
                GUILayout.Label(TexMotionLocalization.TrLiteral("Video Face Track Present"), MotionTimelineTheme.SuccessBadge);
                GUILayout.FlexibleSpace();
                if (GUILayout.Button(TexMotionLocalization.TrLiteral("Sync in Repair Tab"), MotionTimelineTheme.GhostButton, GUILayout.Height(18f)))
                {
                    if (state != null)
                    {
                        state.Mode = TimelineInspectorMode.Repair;
                        state.ActiveRepairTool = TimelineRepairTool.FaceHeadSync;
                    }
                }
                EditorGUILayout.EndHorizontal();
            }

            EditorGUILayout.EndVertical();
        }

        #endregion

        #region Tool 5: IK Pins

        private static void DrawIkPinsTool(MotionTimelineEditorWindow window, EditableMotionData data, int frame)
        {
            EditorGUILayout.BeginVertical(MotionTimelineTheme.Card);
            GUILayout.Label(TexMotionLocalization.TrLiteral("IK Pins Solver"), MotionTimelineTheme.SectionHeader);
            GUILayout.Space(4f);

            EditorGUI.BeginChangeCheck();
            GUIContent ikPinsToggleContent = new GUIContent(TexMotionLocalization.TrLiteral("Enable IK Pins Solving"), TexMotionLocalization.TrLiteral("Pin hands and feet in world space while editing other joints."));
            bool enabled = EditorGUILayout.Toggle(ikPinsToggleContent, window.EnableIkPins);
            if (EditorGUI.EndChangeCheck())
            {
                window.EnableIkPins = enabled;
                window.RefreshPreview();
            }

            GUILayout.Space(8f);
            GUILayout.Label(TexMotionLocalization.TrLiteral("Target Limb"), MotionTimelineTheme.SectionHeader);
            GUILayout.Space(4f);

            for (int i = 0; i < IkLimbs.Length; i++)
            {
                var limb = IkLimbs[i];
                bool isSelected = window.SelectedJoint == limb.Joint;

                EditorGUILayout.BeginHorizontal();
                GUILayout.Label(TexMotionLocalization.TrLiteral(limb.LimbName), MotionTimelineTheme.Label, GUILayout.Width(130f));
                if (GUILayout.Button(isSelected ? TexMotionLocalization.TrLiteral("Active") : TexMotionLocalization.TrLiteral("Select"), isSelected ? MotionTimelineTheme.TabActive : MotionTimelineTheme.GhostButton, GUILayout.Height(20f), GUILayout.ExpandWidth(true)))
                {
                    window.SelectedJoint = limb.Joint;
                    window.RefreshPreview();
                }
                GUIContent resetLimbContent = new GUIContent(TexMotionLocalization.TrLiteral("Reset Limb"), TexMotionLocalization.TrLiteral("Revert this limb to original motion pose."));
                if (GUILayout.Button(resetLimbContent, MotionTimelineTheme.GhostButton, GUILayout.Width(80f), GUILayout.Height(20f)))
                {
                    data.ResetJointToOriginal(frame, limb.Joint);
                    window.RefreshPreview();
                }
                EditorGUILayout.EndHorizontal();
                GUILayout.Space(2f);
            }

            GUILayout.Space(6f);
            GUIContent resetAllContent = new GUIContent(TexMotionLocalization.TrLiteral("Reset All Limbs to Original"), TexMotionLocalization.TrLiteral("Revert all limbs to original motion pose."));
            if (GUILayout.Button(resetAllContent, MotionTimelineTheme.GhostButton, GUILayout.Height(24f), GUILayout.ExpandWidth(true)))
            {
                for (int i = 0; i < IkLimbs.Length; i++)
                {
                    data.ResetJointToOriginal(frame, IkLimbs[i].Joint);
                }
                window.RefreshPreview();
            }

            EditorGUILayout.EndVertical();
        }

        #endregion
    }
}
