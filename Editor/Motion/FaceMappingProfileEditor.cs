using System;
using System.Collections.Generic;
using TexMotion.Runtime.Motion;
using UnityEditor;
using UnityEngine;

namespace TexMotion.Editor.Motion
{
    /// <summary>
    /// Custom Inspector for FaceMappingProfile providing intuitive blendshape mapping management,
    /// avatar mesh scanning, keyword-based auto-detection, and per-shape gain/deadzone tuning.
    /// </summary>
    [CustomEditor(typeof(FaceMappingProfile))]
    public class FaceMappingProfileEditor : UnityEditor.Editor
    {
        private FaceMappingProfile _profile;
        private SkinnedMeshRenderer _targetRenderer;
        private GameObject _targetAvatar;
        private string _searchQuery = "";
        private bool _showGlobalSettings = true;
        private bool _showMappings = true;
        private bool _filterMappedOnly = false;
        private bool _filterUnmappedOnly = false;

        // Cached blend shape names from _targetRenderer for dropdown selection
        private string[] _meshBlendShapeNames = Array.Empty<string>();

        private void OnEnable()
        {
            _profile = (FaceMappingProfile)target;
            RefreshMeshBlendShapes();
        }

        public override void OnInspectorGUI()
        {
            serializedObject.Update();

            DrawProfileHeader();
            DrawTargetMeshSection();
            DrawGlobalSettingsSection();
            DrawQuickActionsSection();
            DrawMappingsSection();

            serializedObject.ApplyModifiedProperties();
        }

        private void DrawProfileHeader()
        {
            EditorGUILayout.Space(6);
            using (new EditorGUILayout.VerticalScope(EditorStyles.helpBox))
            {
                EditorGUILayout.LabelField("TexMotion Face Mapping Profile", EditorStyles.boldLabel);
                EditorGUILayout.LabelField(
                    "Maps MediaPipe 52 facial blendshapes to avatar mesh blendshapes (ARKit, VRChat, VRoid, MMD).",
                    EditorStyles.miniLabel);

                int totalShapes = FaceMappingProfile.MediaPipe52BlendShapes.Length;
                int mappedShapes = 0;
                if (_profile.Mappings != null)
                {
                    foreach (var m in _profile.Mappings)
                    {
                        if (m != null && m.Enabled && !string.IsNullOrEmpty(m.TargetShapeName))
                        {
                            mappedShapes++;
                        }
                    }
                }

                float ratio = totalShapes > 0 ? (float)mappedShapes / totalShapes : 0f;
                Rect rect = GUILayoutUtility.GetRect(18, 18, "TextField");
                EditorGUI.ProgressBar(rect, ratio, $"{mappedShapes} / {totalShapes} Mapped ({Mathf.RoundToInt(ratio * 100f)}%)");
            }
            EditorGUILayout.Space(4);
        }

        private void DrawTargetMeshSection()
        {
            using (new EditorGUILayout.VerticalScope(EditorStyles.helpBox))
            {
                EditorGUILayout.LabelField("Target Avatar & Mesh", EditorStyles.boldLabel);

                EditorGUI.BeginChangeCheck();
                _targetAvatar = (GameObject)EditorGUILayout.ObjectField(
                    new GUIContent("Avatar Root", "Drop avatar root GameObject here to auto-detect face mesh."),
                    _targetAvatar, typeof(GameObject), true);
                if (EditorGUI.EndChangeCheck() && _targetAvatar != null)
                {
                    var foundRenderer = FaceEmotionHelper.FindFaceRenderer(_targetAvatar);
                    if (foundRenderer != null)
                    {
                        _targetRenderer = foundRenderer;
                        _profile.TargetMeshPath = AnimationUtility.CalculateTransformPath(foundRenderer.transform, _targetAvatar.transform);
                        EditorUtility.SetDirty(_profile);
                        RefreshMeshBlendShapes();
                    }
                }

                EditorGUI.BeginChangeCheck();
                _targetRenderer = (SkinnedMeshRenderer)EditorGUILayout.ObjectField(
                    new GUIContent("Face SkinnedMesh", "Target facial SkinnedMeshRenderer containing blendshapes."),
                    _targetRenderer, typeof(SkinnedMeshRenderer), true);
                if (EditorGUI.EndChangeCheck())
                {
                    if (_targetRenderer != null && _targetAvatar != null)
                    {
                        _profile.TargetMeshPath = AnimationUtility.CalculateTransformPath(_targetRenderer.transform, _targetAvatar.transform);
                        EditorUtility.SetDirty(_profile);
                    }
                    RefreshMeshBlendShapes();
                }

                SerializedProperty targetMeshPathProp = serializedObject.FindProperty("TargetMeshPath");
                EditorGUILayout.PropertyField(targetMeshPathProp, new GUIContent("Mesh Path", "Root-relative path to the facial renderer in the avatar hierarchy (e.g. 'Body')."));
            }
            EditorGUILayout.Space(4);
        }

        private void DrawGlobalSettingsSection()
        {
            _showGlobalSettings = EditorGUILayout.Foldout(_showGlobalSettings, "Global Adjustments", true, EditorStyles.foldoutHeader);
            if (!_showGlobalSettings) return;

            using (new EditorGUILayout.VerticalScope(EditorStyles.helpBox))
            {
                SerializedProperty globalMultProp = serializedObject.FindProperty("GlobalMultiplier");
                SerializedProperty applyHeadProp = serializedObject.FindProperty("ApplyHeadRotation");
                SerializedProperty headWeightProp = serializedObject.FindProperty("HeadRotationWeight");

                EditorGUILayout.PropertyField(globalMultProp, new GUIContent("Global Gain Multiplier", "Overall scaling factor for all mapped blendshapes."));
                EditorGUILayout.PropertyField(applyHeadProp, new GUIContent("Apply Head Rotation", "Apply video-tracked head rotation to the avatar's Head bone."));
                if (applyHeadProp.boolValue)
                {
                    EditorGUI.indentLevel++;
                    EditorGUILayout.PropertyField(headWeightProp, new GUIContent("Head Rotation Blend", "Blend weight for head rotation [0.0 - 1.0]."));
                    EditorGUI.indentLevel--;
                }
            }
            EditorGUILayout.Space(4);
        }

        private void DrawQuickActionsSection()
        {
            using (new EditorGUILayout.VerticalScope(EditorStyles.helpBox))
            {
                EditorGUILayout.LabelField("Quick Setup Actions", EditorStyles.boldLabel);

                using (new EditorGUILayout.HorizontalScope())
                {
                    GUI.enabled = _targetRenderer != null;
                    if (GUILayout.Button(new GUIContent("⚡ Auto-Map from Mesh", "Scans the assigned SkinnedMeshRenderer and auto-generates matching mappings."), GUILayout.Height(24)))
                    {
                        Undo.RecordObject(_profile, "Auto-Generate BlendShape Mapping");
                        int count = _profile.AutoGenerateMapping(_targetRenderer);
                        EditorUtility.SetDirty(_profile);
                        serializedObject.Update();
                        EditorUtility.DisplayDialog("Auto-Map Complete", $"Successfully mapped {count} blendshapes from {_targetRenderer.name}.", "OK");
                    }
                    GUI.enabled = true;

                    if (GUILayout.Button(new GUIContent("📋 Reset to ARKit 52", "Sets up standard 1:1 ARKit 52 blendshape names."), GUILayout.Height(24)))
                    {
                        if (EditorUtility.DisplayDialog("Reset Mappings", "Reset all mappings to standard ARKit 52 names?", "Yes", "No"))
                        {
                            Undo.RecordObject(_profile, "Reset to ARKit 52");
                            _profile.ResetToStandard52();
                            EditorUtility.SetDirty(_profile);
                            serializedObject.Update();
                        }
                    }

                    if (GUILayout.Button(new GUIContent("🗑 Clear All", "Removes all mappings."), GUILayout.Height(24)))
                    {
                        if (EditorUtility.DisplayDialog("Clear Mappings", "Remove all blendshape mappings?", "Yes", "No"))
                        {
                            Undo.RecordObject(_profile, "Clear Face Mappings");
                            _profile.Mappings.Clear();
                            EditorUtility.SetDirty(_profile);
                            serializedObject.Update();
                        }
                    }
                }
            }
            EditorGUILayout.Space(4);
        }

        private void DrawMappingsSection()
        {
            _showMappings = EditorGUILayout.Foldout(_showMappings, $"BlendShape Mappings ({_profile.Mappings.Count})", true, EditorStyles.foldoutHeader);
            if (!_showMappings) return;

            using (new EditorGUILayout.VerticalScope(EditorStyles.helpBox))
            {
                // Filter bar
                using (new EditorGUILayout.HorizontalScope())
                {
                    EditorGUILayout.LabelField("Search:", GUILayout.Width(50));
                    _searchQuery = EditorGUILayout.TextField(_searchQuery);
                    if (!string.IsNullOrEmpty(_searchQuery) && GUILayout.Button("×", GUILayout.Width(20)))
                    {
                        _searchQuery = "";
                        GUI.FocusControl(null);
                    }
                }

                using (new EditorGUILayout.HorizontalScope())
                {
                    EditorGUI.BeginChangeCheck();
                    bool showAll = GUILayout.Toggle(!_filterMappedOnly && !_filterUnmappedOnly, "All", EditorStyles.miniButtonLeft);
                    if (showAll && (_filterMappedOnly || _filterUnmappedOnly))
                    {
                        _filterMappedOnly = false;
                        _filterUnmappedOnly = false;
                    }

                    _filterMappedOnly = GUILayout.Toggle(_filterMappedOnly, "Mapped Only", EditorStyles.miniButtonMid);
                    if (_filterMappedOnly) _filterUnmappedOnly = false;

                    _filterUnmappedOnly = GUILayout.Toggle(_filterUnmappedOnly, "Unmapped Only", EditorStyles.miniButtonRight);
                    if (_filterUnmappedOnly) _filterMappedOnly = false;
                }

                EditorGUILayout.Space(4);

                // Table Header
                using (new EditorGUILayout.HorizontalScope(EditorStyles.toolbar))
                {
                    GUILayout.Label("En", EditorStyles.miniBoldLabel, GUILayout.Width(22));
                    GUILayout.Label("MediaPipe Source", EditorStyles.miniBoldLabel, GUILayout.Width(130));
                    GUILayout.Label("Avatar Target Shape", EditorStyles.miniBoldLabel, GUILayout.MinWidth(110));
                    GUILayout.Label("Gain", EditorStyles.miniBoldLabel, GUILayout.Width(45));
                    GUILayout.Label("DeadZ", EditorStyles.miniBoldLabel, GUILayout.Width(45));
                    GUILayout.Label("L/R", EditorStyles.miniBoldLabel, GUILayout.Width(30));
                    GUILayout.Label("", GUILayout.Width(22));
                }

                SerializedProperty mappingsProp = serializedObject.FindProperty("Mappings");

                int displayedCount = 0;
                int toDeleteIndex = -1;

                for (int i = 0; i < mappingsProp.arraySize; i++)
                {
                    SerializedProperty element = mappingsProp.GetArrayElementAtIndex(i);
                    SerializedProperty srcProp = element.FindPropertyRelative("SourceShapeName");
                    SerializedProperty tgtProp = element.FindPropertyRelative("TargetShapeName");
                    SerializedProperty multProp = element.FindPropertyRelative("Multiplier");
                    SerializedProperty deadProp = element.FindPropertyRelative("DeadZone");
                    SerializedProperty invProp = element.FindPropertyRelative("InvertLeftRight");
                    SerializedProperty enProp = element.FindPropertyRelative("Enabled");

                    string srcName = srcProp.stringValue ?? "";
                    string tgtName = tgtProp.stringValue ?? "";

                    // Filter logic
                    bool isMapped = !string.IsNullOrEmpty(tgtName);
                    if (_filterMappedOnly && !isMapped) continue;
                    if (_filterUnmappedOnly && isMapped) continue;

                    if (!string.IsNullOrEmpty(_searchQuery))
                    {
                        if (!srcName.IndexOf(_searchQuery, StringComparison.OrdinalIgnoreCase).Equals(-1) == false &&
                            !tgtName.IndexOf(_searchQuery, StringComparison.OrdinalIgnoreCase).Equals(-1) == false)
                        {
                            continue;
                        }
                    }

                    displayedCount++;

                    using (new EditorGUILayout.HorizontalScope())
                    {
                        enProp.boolValue = EditorGUILayout.Toggle(enProp.boolValue, GUILayout.Width(22));

                        // Source Shape (Label or Popup)
                        EditorGUILayout.LabelField(new GUIContent(srcName, srcName), GUILayout.Width(130));

                        // Target Shape: Dropdown if mesh available, otherwise TextField
                        if (_meshBlendShapeNames.Length > 0)
                        {
                            int currentIndex = Array.IndexOf(_meshBlendShapeNames, tgtName);
                            int selectedIndex = EditorGUILayout.Popup(currentIndex, _meshBlendShapeNames, GUILayout.MinWidth(110));
                            if (selectedIndex >= 0 && selectedIndex < _meshBlendShapeNames.Length)
                            {
                                tgtProp.stringValue = _meshBlendShapeNames[selectedIndex];
                            }
                        }
                        else
                        {
                            tgtProp.stringValue = EditorGUILayout.TextField(tgtProp.stringValue, GUILayout.MinWidth(110));
                        }

                        // Multiplier
                        multProp.floatValue = EditorGUILayout.FloatField(multProp.floatValue, GUILayout.Width(45));

                        // DeadZone
                        deadProp.floatValue = EditorGUILayout.FloatField(deadProp.floatValue, GUILayout.Width(45));

                        // Invert Left/Right
                        invProp.boolValue = EditorGUILayout.Toggle(invProp.boolValue, GUILayout.Width(30));

                        // Delete button
                        if (GUILayout.Button("×", EditorStyles.miniButton, GUILayout.Width(22)))
                        {
                            toDeleteIndex = i;
                        }
                    }
                }

                if (toDeleteIndex >= 0)
                {
                    mappingsProp.DeleteArrayElementAtIndex(toDeleteIndex);
                }

                if (displayedCount == 0)
                {
                    EditorGUILayout.HelpBox("No blendshape mappings match the current filter.", MessageType.Info);
                }

                EditorGUILayout.Space(6);

                // Add button
                if (GUILayout.Button("+ Add Custom Mapping Entry", EditorStyles.miniButton))
                {
                    int newIndex = mappingsProp.arraySize;
                    mappingsProp.InsertArrayElementAtIndex(newIndex);
                    var newElement = mappingsProp.GetArrayElementAtIndex(newIndex);
                    newElement.FindPropertyRelative("SourceShapeName").stringValue = "customShape";
                    newElement.FindPropertyRelative("TargetShapeName").stringValue = "";
                    newElement.FindPropertyRelative("Multiplier").floatValue = 1.0f;
                    newElement.FindPropertyRelative("DeadZone").floatValue = 0.0f;
                    newElement.FindPropertyRelative("InvertLeftRight").boolValue = false;
                    newElement.FindPropertyRelative("Enabled").boolValue = true;
                }
            }
        }

        private void RefreshMeshBlendShapes()
        {
            if (_targetRenderer != null && _targetRenderer.sharedMesh != null)
            {
                var mesh = _targetRenderer.sharedMesh;
                int count = mesh.blendShapeCount;
                var list = new List<string>(count + 1);
                list.Add("(None)");
                for (int i = 0; i < count; i++)
                {
                    list.Add(mesh.GetBlendShapeName(i));
                }
                _meshBlendShapeNames = list.ToArray();
            }
            else
            {
                _meshBlendShapeNames = Array.Empty<string>();
            }
        }
    }
}
