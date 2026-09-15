using System;
using System.Collections.Generic;
using System.IO;
using TexMotion.Editor.VRChat;
using TexMotion.Runtime.Motion;
using TexMotion.Runtime.Native;
using TexMotion.Runtime.VRChat;
using UnityEditor;
using UnityEngine;
using UnityEngine.Video;

namespace TexMotion.Editor.Motion
{
    /// <summary>
    /// Dedicated motion timeline editor window.
    /// Allows inspecting motions on a visual timeline, scrubbing frame-by-frame,
    /// and directly tweaking bone rotations, root positions, mirroring, smoothing, and copying poses per frame.
    /// </summary>
    public class MotionTimelineEditorWindow : EditorWindow
    {
        // Active editable motion data
        private EditableMotionData _data;

        // Playback state
        private int _currentFrame = 0;
        private bool _isPlaying = false;
        private bool _isLoop = true;
        private float _playbackSpeed = 1.0f;
        private double _lastUpdateTime = 0;
        private float _accumulatedTime = 0f;

        // 3D Viewport & Avatar Preview
        private PreviewRenderUtility _previewUtility;
        private GameObject _previewInstance;
        private readonly Dictionary<SmplxJoint, Transform> _previewBoneMap = new Dictionary<SmplxJoint, Transform>();
        private readonly Dictionary<SmplxJoint, Quaternion> _previewInitialRotations = new Dictionary<SmplxJoint, Quaternion>();
        private Transform _previewHipsTransform;
        private Vector3 _previewInitialHipsPos;
        private Vector2 _previewDir = new Vector2(180f, 10f);
        private float _previewDistance = 2.8f;
        private Vector3 _previewPivot = new Vector3(0, 1.0f, 0);

        // Bone Skeleton & Interactive Gizmo Manipulation (Blender-style)
        private bool _showSkeleton = true;
        private bool _showGizmos = true;
        private SmplxJoint? _selectedJoint = null;
        private SmplxJoint? _hoveredJoint = null;

        private enum GizmoAxis
        {
            None,
            X,      // Red - Pitch
            Y,      // Green - Yaw
            Z,      // Blue - Roll
            Screen  // White - View axis ring
        }
        private GizmoAxis _activeGizmoAxis = GizmoAxis.None;
        private GizmoAxis _hoveredGizmoAxis = GizmoAxis.None;
        private Vector2 _dragStartMousePos;
        private Vector3 _dragStartEuler;
        private Quaternion _dragStartLocalRot;

        // Bone Hierarchy Connections for Skeleton Line Drawing
        private static readonly (SmplxJoint Parent, SmplxJoint Child)[] BoneConnections = new[]
        {
            (SmplxJoint.Pelvis, SmplxJoint.Spine1),
            (SmplxJoint.Spine1, SmplxJoint.Spine2),
            (SmplxJoint.Spine2, SmplxJoint.Spine3),
            (SmplxJoint.Spine3, SmplxJoint.Neck),
            (SmplxJoint.Neck, SmplxJoint.Head),

            (SmplxJoint.Spine3, SmplxJoint.L_Collar),
            (SmplxJoint.L_Collar, SmplxJoint.L_Shoulder),
            (SmplxJoint.L_Shoulder, SmplxJoint.L_Elbow),
            (SmplxJoint.L_Elbow, SmplxJoint.L_Wrist),

            (SmplxJoint.Spine3, SmplxJoint.R_Collar),
            (SmplxJoint.R_Collar, SmplxJoint.R_Shoulder),
            (SmplxJoint.R_Shoulder, SmplxJoint.R_Elbow),
            (SmplxJoint.R_Elbow, SmplxJoint.R_Wrist),

            (SmplxJoint.Pelvis, SmplxJoint.L_Hip),
            (SmplxJoint.L_Hip, SmplxJoint.L_Knee),
            (SmplxJoint.L_Knee, SmplxJoint.L_Ankle),
            (SmplxJoint.L_Ankle, SmplxJoint.L_Foot),

            (SmplxJoint.Pelvis, SmplxJoint.R_Hip),
            (SmplxJoint.R_Hip, SmplxJoint.R_Knee),
            (SmplxJoint.R_Knee, SmplxJoint.R_Ankle),
            (SmplxJoint.R_Ankle, SmplxJoint.R_Foot)
        };

        // Video synchronization (when editing VideoMotionData)
        private GameObject _videoPlayerGo;
        private VideoPlayer _videoPlayer;
        private RenderTexture _videoTexture;
        private string _loadedVideoPath = "";

        private enum ViewportLayout
        {
            SideBySide,
            AvatarOnly,
            VideoOnly
        }
        private ViewportLayout _viewportLayout = ViewportLayout.SideBySide;

        // Timeline UI state
        private Vector2 _timelineScrollPos;
        private float _pixelsPerFrame = 28.0f;
        private bool _isDraggingScrubber = false;
        private const float TIMELINE_HEIGHT = 140f;

        // Inspector foldouts
        private Vector2 _inspectorScrollPos;
        private bool _foldoutRoot = true;
        private bool _foldoutTorso = true;
        private bool _foldoutLeftArm = false;
        private bool _foldoutRightArm = false;
        private bool _foldoutLeftLeg = false;
        private bool _foldoutRightLeg = false;
        private bool _foldoutTools = true;

        // GUI Styles
        private GUIStyle _timelineBgStyle;
        private GUIStyle _frameBoxStyle;
        private GUIStyle _keyframeMarkerStyle;
        private GUIStyle _rulerTextStyle;

        [MenuItem("Tools/TexMotion/Motion Timeline Editor", false, 1)]
        public static MotionTimelineEditorWindow ShowEditor()
        {
            var window = GetWindow<MotionTimelineEditorWindow>("Timeline Editor");
            window.minSize = new Vector2(850, 640);
            window.Show();
            return window;
        }

        /// <summary>
        /// Opens the Timeline Editor with newly generated or extracted motion data.
        /// </summary>
        public static MotionTimelineEditorWindow OpenWithMotion(
            GeneratedMotionData motionData,
            Animator targetAvatar,
            string motionName,
            HandPoseType handPose = HandPoseType.NaturalRelaxed,
            FaceEmotionType faceEmotion = FaceEmotionType.None,
            float emotionIntensity = 0.8f,
            bool inPlace = true)
        {
            var window = ShowEditor();
            var editable = new EditableMotionData(motionData, targetAvatar, motionName)
            {
                HandPose = handPose,
                FaceEmotion = faceEmotion,
                EmotionIntensity = emotionIntensity,
                InPlace = inPlace
            };
            window.LoadMotion(editable);
            return window;
        }

        public static MotionTimelineEditorWindow OpenWithEditableData(EditableMotionData editableData)
        {
            var window = ShowEditor();
            window.LoadMotion(editableData);
            return window;
        }

        public void LoadMotion(EditableMotionData data)
        {
            _data = data;
            _currentFrame = 0;
            _isPlaying = false;
            _accumulatedTime = 0f;

            CleanupPreviewInstance();
            CleanupVideoPlayer();

            SetupPreviewInstance();
            SetupVideoPlayerIfApplicable();

            ApplyCurrentFrameToPreview();
            Repaint();
        }

        private void OnEnable()
        {
            InitPreviewUtility();
            EditorApplication.update += OnEditorUpdate;
            AssemblyReloadEvents.beforeAssemblyReload += OnBeforeAssemblyReload;
        }

        private void OnDisable()
        {
            AssemblyReloadEvents.beforeAssemblyReload -= OnBeforeAssemblyReload;
            EditorApplication.update -= OnEditorUpdate;
            CleanupVideoPlayer();
            CleanupPreviewInstance();
            CleanupPreviewUtility();
        }

        private void OnDestroy()
        {
            CleanupVideoPlayer();
            CleanupPreviewInstance();
            CleanupPreviewUtility();
        }

        private void OnBeforeAssemblyReload()
        {
            CleanupVideoPlayer();
            CleanupPreviewInstance();
        }

        private void OnEditorUpdate()
        {
            if (_isPlaying && _data != null && _data.Frames > 0)
            {
                double now = EditorApplication.timeSinceStartup;
                double delta = now - _lastUpdateTime;
                _lastUpdateTime = now;

                if (delta > 0.2) delta = 0.033; // Clamp large hitch

                _accumulatedTime += (float)delta * _playbackSpeed;
                float frameDuration = 1.0f / Mathf.Max(1.0f, _data.FrameRate);

                if (_accumulatedTime >= frameDuration)
                {
                    int framesToAdvance = Mathf.FloorToInt(_accumulatedTime / frameDuration);
                    _accumulatedTime %= frameDuration;

                    int nextFrame = _currentFrame + framesToAdvance;
                    if (nextFrame >= _data.Frames)
                    {
                        if (_isLoop)
                        {
                            nextFrame = 0;
                        }
                        else
                        {
                            nextFrame = _data.Frames - 1;
                            _isPlaying = false;
                        }
                    }

                    if (nextFrame != _currentFrame)
                    {
                        _currentFrame = nextFrame;
                        ApplyCurrentFrameToPreview();
                        SyncVideoPlayerToCurrentFrame();
                        Repaint();
                    }
                }
            }
            else
            {
                _lastUpdateTime = EditorApplication.timeSinceStartup;
            }
        }

        private void OnGUI()
        {
            InitStyles();
            HandleKeyboardShortcuts();

            if (_data == null || _data.Frames <= 0)
            {
                DrawEmptyState();
                return;
            }

            DrawTopHeaderToolbar();

            // Main central split: Left Viewport (3D + optional video), Right Pose Inspector
            EditorGUILayout.BeginHorizontal();

            // Left Workspace: 3D Viewport
            float leftWidth = Mathf.Max(350f, position.width - 380f);
            float centralHeight = position.height - 45f - TIMELINE_HEIGHT;

            EditorGUILayout.BeginVertical(GUILayout.Width(leftWidth), GUILayout.Height(centralHeight));
            DrawViewportWorkspace(leftWidth, centralHeight);
            EditorGUILayout.EndVertical();

            // Right Workspace: Pose & Joint Inspector
            EditorGUILayout.BeginVertical(EditorStyles.helpBox, GUILayout.Width(370f), GUILayout.Height(centralHeight));
            DrawPoseInspector(centralHeight);
            EditorGUILayout.EndVertical();

            EditorGUILayout.EndHorizontal();

            // Bottom Workspace: Full Timeline with Ruler and Scrubber
            DrawBottomTimelinePanel();
        }

        #region Empty State

        private void DrawEmptyState()
        {
            EditorGUILayout.BeginVertical(EditorStyles.helpBox);
            GUILayout.FlexibleSpace();

            EditorGUILayout.BeginHorizontal();
            GUILayout.FlexibleSpace();
            EditorGUILayout.BeginVertical();

            var labelStyle = new GUIStyle(EditorStyles.boldLabel)
            {
                fontSize = 18,
                alignment = TextAnchor.MiddleCenter,
                normal = { textColor = new Color(0.9f, 0.9f, 0.95f) }
            };
            EditorGUILayout.LabelField("🎬 TexMotion Timeline & Frame Pose Editor", labelStyle, GUILayout.Height(30));

            var subStyle = new GUIStyle(EditorStyles.label)
            {
                fontSize = 12,
                alignment = TextAnchor.MiddleCenter,
                wordWrap = true,
                normal = { textColor = new Color(0.7f, 0.7f, 0.75f) }
            };
            EditorGUILayout.LabelField("No motion is currently loaded.\nGenerate motion from Text or extract from Video in TexMotion Studio to automatically start editing poses here.", subStyle, GUILayout.Width(500), GUILayout.Height(45));

            EditorGUILayout.Space(15);
            EditorGUILayout.BeginHorizontal();
            GUILayout.FlexibleSpace();
            if (GUILayout.Button("Open TexMotion Studio", GUILayout.Width(220), GUILayout.Height(38)))
            {
                TexMotionWindow.ShowWindow();
            }
            GUILayout.FlexibleSpace();
            EditorGUILayout.EndHorizontal();

            EditorGUILayout.EndVertical();
            GUILayout.FlexibleSpace();
            EditorGUILayout.EndHorizontal();

            GUILayout.FlexibleSpace();
            EditorGUILayout.EndVertical();
        }

        #endregion

        #region Top Header Toolbar

        private void DrawTopHeaderToolbar()
        {
            EditorGUILayout.BeginHorizontal(EditorStyles.toolbar, GUILayout.Height(30));

            // Clip Name
            EditorGUILayout.LabelField("🎬 Clip:", EditorStyles.miniBoldLabel, GUILayout.Width(45));
            _data.ClipName = EditorGUILayout.TextField(_data.ClipName, GUILayout.Width(150));

            // Info Badges
            string info = $"Frames: {_data.Frames} | {_data.Duration:F2}s ({_data.FrameRate:F0} FPS)";
            EditorGUILayout.LabelField(info, EditorStyles.miniLabel, GUILayout.Width(170));

            // Target Avatar selector
            EditorGUILayout.LabelField("Avatar:", EditorStyles.miniBoldLabel, GUILayout.Width(45));
            var prevAvatar = _data.TargetAvatar;
            _data.TargetAvatar = (Animator)EditorGUILayout.ObjectField(_data.TargetAvatar, typeof(Animator), true, GUILayout.Width(140));
            if (prevAvatar != _data.TargetAvatar)
            {
                SetupPreviewInstance();
                ApplyCurrentFrameToPreview();
                Repaint();
            }

            // Modification Status
            int modCount = _data.ModifiedFrameCount;
            if (modCount > 0)
            {
                var modStyle = new GUIStyle(EditorStyles.miniBoldLabel)
                {
                    normal = { textColor = new Color(1.0f, 0.75f, 0.25f) }
                };
                EditorGUILayout.LabelField($"● {modCount} mod", modStyle, GUILayout.Width(80));
            }
            else
            {
                EditorGUILayout.LabelField("✓ Clean", EditorStyles.miniLabel, GUILayout.Width(60));
            }

            GUILayout.FlexibleSpace();

            // Undo / Redo
            GUI.enabled = _data.CanUndo;
            if (GUILayout.Button("↩ Undo", EditorStyles.toolbarButton, GUILayout.Width(60)))
            {
                _data.Undo();
                ApplyCurrentFrameToPreview();
                Repaint();
            }
            GUI.enabled = _data.CanRedo;
            if (GUILayout.Button("↪ Redo", EditorStyles.toolbarButton, GUILayout.Width(60)))
            {
                _data.Redo();
                ApplyCurrentFrameToPreview();
                Repaint();
            }
            GUI.enabled = true;

            // Revert All
            if (modCount > 0)
            {
                GUI.backgroundColor = new Color(0.9f, 0.4f, 0.4f);
                if (GUILayout.Button("↩ Revert All", EditorStyles.toolbarButton, GUILayout.Width(80)))
                {
                    if (EditorUtility.DisplayDialog("Revert All Frames", "Discard all frame modifications and revert back to original generated motion?", "Revert", "Cancel"))
                    {
                        _data.ResetAllFramesToOriginal();
                        ApplyCurrentFrameToPreview();
                        Repaint();
                    }
                }
                GUI.backgroundColor = Color.white;
            }

            // Save As Clip
            GUI.backgroundColor = new Color(0.3f, 0.7f, 1.0f);
            if (GUILayout.Button("💾 Save .anim", EditorStyles.toolbarButton, GUILayout.Width(90)))
            {
                SaveEditedMotionAsAsset();
            }

            // Apply to Avatar
            GUI.backgroundColor = new Color(0.2f, 0.85f, 0.45f);
            if (GUILayout.Button("✨ Apply to Avatar", EditorStyles.toolbarButton, GUILayout.Width(110)))
            {
                ApplyEditedMotionToAvatar();
            }
            GUI.backgroundColor = Color.white;

            EditorGUILayout.EndHorizontal();
        }

        #endregion

        #region Left Viewport Workspace

        private void DrawViewportWorkspace(float width, float height)
        {
            bool hasVideo = _data.SourceVideoData != null && (!string.IsNullOrEmpty(_data.SourceVideoData.OverlayVideoPath) || !string.IsNullOrEmpty(_data.SourceVideoData.SourceVideoPath));

            if (hasVideo)
            {
                // Layout Toolbar for Video + Avatar
                EditorGUILayout.BeginHorizontal(EditorStyles.toolbar);
                EditorGUILayout.LabelField("3D Viewport & Video", EditorStyles.miniBoldLabel);
                GUILayout.FlexibleSpace();
                _viewportLayout = (ViewportLayout)GUILayout.Toolbar((int)_viewportLayout,
                    new string[] { "Side-by-Side", "Avatar Only", "Video Only" },
                    EditorStyles.miniButton, GUILayout.Width(250));
                EditorGUILayout.EndHorizontal();
            }

            Rect viewportRect = GUILayoutUtility.GetRect(width - 8f, height - (hasVideo ? 30f : 10f), GUILayout.ExpandWidth(true), GUILayout.ExpandHeight(true));

            if (hasVideo && _viewportLayout == ViewportLayout.SideBySide)
            {
                float gap = 6f;
                float halfW = Mathf.Max(20f, (viewportRect.width - gap) * 0.5f);
                Rect leftVideoRect = new Rect(viewportRect.x, viewportRect.y, halfW, viewportRect.height);
                Rect rightAvatarRect = new Rect(viewportRect.x + halfW + gap, viewportRect.y, halfW, viewportRect.height);

                DrawVideoTexture(leftVideoRect);
                DrawAvatarPreview(rightAvatarRect);
            }
            else if (hasVideo && _viewportLayout == ViewportLayout.VideoOnly)
            {
                DrawVideoTexture(viewportRect);
            }
            else
            {
                DrawAvatarPreview(viewportRect);
            }
        }

        private void DrawAvatarPreview(Rect rect)
        {
            if (rect.width <= 4f || rect.height <= 4f) return;

            InitPreviewUtility();

            if (_previewInstance == null && _data != null && _data.TargetAvatar != null)
            {
                SetupPreviewInstance();
            }

            Event evt = Event.current;

            // 1. Process Gizmo & Joint Selection first (Blender-style direct manipulation)
            bool gizmoHandled = HandleGizmoAndSelectionEvents(rect, evt);
            if (gizmoHandled)
            {
                return;
            }

            // Handle Camera interaction (orbit, zoom, pan)
            int controlID = GUIUtility.GetControlID(FocusType.Passive);

            switch (evt.GetTypeForControl(controlID))
            {
                case EventType.MouseDown:
                    if (rect.Contains(evt.mousePosition))
                    {
                        GUIUtility.hotControl = controlID;
                        evt.Use();
                    }
                    break;
                case EventType.MouseDrag:
                    if (GUIUtility.hotControl == controlID)
                    {
                        if (evt.button == 0) // Left click orbit
                        {
                            _previewDir.x -= evt.delta.x * 0.8f;
                            _previewDir.y += evt.delta.y * 0.8f;
                            _previewDir.y = Mathf.Clamp(_previewDir.y, -80f, 80f);
                        }
                        else if (evt.button == 1 || evt.button == 2) // Right / Middle click pan
                        {
                            _previewPivot.y += evt.delta.y * 0.005f * _previewDistance;
                            _previewPivot.x -= evt.delta.x * 0.005f * _previewDistance;
                        }
                        evt.Use();
                        Repaint();
                    }
                    break;
                case EventType.MouseUp:
                    if (GUIUtility.hotControl == controlID)
                    {
                        GUIUtility.hotControl = 0;
                        evt.Use();
                    }
                    break;
                case EventType.ScrollWheel:
                    if (rect.Contains(evt.mousePosition))
                    {
                        _previewDistance = Mathf.Clamp(_previewDistance + evt.delta.y * 0.15f, 0.8f, 10f);
                        evt.Use();
                        Repaint();
                    }
                    break;
            }

            if (evt.type == EventType.Repaint)
            {
                if (_previewUtility != null && _previewInstance != null)
                {
                    _previewUtility.BeginPreview(rect, GUIStyle.none);

                    Quaternion camRot = Quaternion.Euler(_previewDir.y, _previewDir.x, 0);
                    Vector3 camPos = _previewPivot + camRot * (Vector3.forward * _previewDistance);

                    _previewUtility.camera.transform.position = camPos;
                    _previewUtility.camera.transform.LookAt(_previewPivot);

                    _previewUtility.lights[0].transform.rotation = Quaternion.Euler(40f, 40f, 0);
                    _previewUtility.lights[1].transform.rotation = Quaternion.Euler(140f, -40f, 0);

                    ApplyCurrentFrameToPreview();

                    _previewUtility.Render(true);
                    Texture resultTex = _previewUtility.EndPreview();
                    GUI.DrawTexture(rect, resultTex, ScaleMode.StretchToFill, false);

                    // Overlay Skeleton and 3D Rotation Gizmos (Blender Pose Mode)
                    DrawSkeletonOverlay(rect, _previewUtility.camera);
                    DrawBoneGizmo(rect, _previewUtility.camera);
                }
                else
                {
                    EditorGUI.DrawRect(rect, new Color(0.10f, 0.12f, 0.16f));
                    var missingStyle = new GUIStyle(EditorStyles.boldLabel)
                    {
                        alignment = TextAnchor.MiddleCenter,
                        normal = { textColor = new Color(0.95f, 0.65f, 0.2f) }
                    };
                    GUI.Label(rect, "⚠️ No Avatar Assigned\nPlease select a Humanoid Avatar in the toolbar above.", missingStyle);
                }

                // Overlay Viewport HUD (Toggles, Camera Reset, Bone badge, hint)
                DrawViewportHUD(rect);
            }
        }

        #region Bone Skeleton & Gizmo Interaction

        private bool TryWorldToScreen(Camera cam, Rect viewportRect, Vector3 worldPos, out Vector2 screenPos)
        {
            screenPos = Vector2.zero;
            if (cam == null) return false;

            Vector3 v = cam.WorldToViewportPoint(worldPos);
            if (v.z <= 0.01f) return false; // Behind camera

            screenPos = new Vector2(
                viewportRect.x + v.x * viewportRect.width,
                viewportRect.y + (1.0f - v.y) * viewportRect.height
            );
            return true;
        }

        private void DrawSkeletonOverlay(Rect rect, Camera cam)
        {
            if (!_showSkeleton || _previewInstance == null || cam == null) return;

            Handles.BeginGUI();

            // 1. Draw Bone Connection Lines
            foreach (var (parentJoint, childJoint) in BoneConnections)
            {
                if (_previewBoneMap.TryGetValue(parentJoint, out Transform parent) && parent != null &&
                    _previewBoneMap.TryGetValue(childJoint, out Transform child) && child != null)
                {
                    if (TryWorldToScreen(cam, rect, parent.position, out Vector2 p1) &&
                        TryWorldToScreen(cam, rect, child.position, out Vector2 p2))
                    {
                        bool isHighlighted = (_selectedJoint == parentJoint || _selectedJoint == childJoint);
                        Color boneColor = isHighlighted
                            ? new Color(1.0f, 0.85f, 0.25f, 0.95f)
                            : GetBoneLineColor(parentJoint, childJoint);

                        float width = isHighlighted ? 3.5f : 2.0f;
                        Handles.color = boneColor;
                        Handles.DrawAAPolyLine(width, new Vector3(p1.x, p1.y, 0), new Vector3(p2.x, p2.y, 0));
                    }
                }
            }

            // 2. Draw Joint Nodes (Spheres)
            foreach (var kvp in _previewBoneMap)
            {
                SmplxJoint joint = kvp.Key;
                Transform bone = kvp.Value;
                if (bone == null) continue;

                if (TryWorldToScreen(cam, rect, bone.position, out Vector2 pt))
                {
                    bool isSelected = (_selectedJoint == joint);
                    bool isHovered = (_hoveredJoint == joint);

                    float radius = isSelected ? 8f : (isHovered ? 7f : 5f);
                    Color fillCol = GetJointColor(joint, isHovered, isSelected);

                    // Outer dark ring for contrast
                    Handles.color = isSelected ? Color.white : new Color(0.08f, 0.10f, 0.15f, 0.95f);
                    Handles.DrawSolidDisc(new Vector3(pt.x, pt.y, 0), Vector3.forward, radius + 1.5f);

                    // Core colored disc
                    Handles.color = fillCol;
                    Handles.DrawSolidDisc(new Vector3(pt.x, pt.y, 0), Vector3.forward, radius);

                    if (isSelected)
                    {
                        // Glowing accent ring
                        Handles.color = new Color(1.0f, 0.88f, 0.2f, 0.75f);
                        Handles.DrawWireDisc(new Vector3(pt.x, pt.y, 0), Vector3.forward, radius + 3.5f);
                    }
                }
            }

            Handles.EndGUI();
        }

        private void DrawBoneGizmo(Rect rect, Camera cam)
        {
            if (!_showGizmos || !_selectedJoint.HasValue || _previewInstance == null || cam == null)
                return;

            SmplxJoint joint = _selectedJoint.Value;
            if (!_previewBoneMap.TryGetValue(joint, out Transform bone) || bone == null)
                return;

            Vector3 center = bone.position;
            if (!TryWorldToScreen(cam, rect, center, out Vector2 centerScreen))
                return;

            Handles.BeginGUI();

            float gizmoRadius = 0.16f * _previewDistance;
            int segments = 32;

            Vector3 right = bone.right;
            Vector3 up = bone.up;
            Vector3 forward = bone.forward;

            // X Ring (Pitch / Red)
            DrawGizmoRing(rect, cam, center, up, forward, gizmoRadius, segments,
                _activeGizmoAxis == GizmoAxis.X ? Color.yellow : (_hoveredGizmoAxis == GizmoAxis.X ? Color.white : new Color(0.95f, 0.25f, 0.25f, 0.95f)),
                _activeGizmoAxis == GizmoAxis.X || _hoveredGizmoAxis == GizmoAxis.X ? 4.0f : 2.5f);

            // Y Ring (Yaw / Green)
            DrawGizmoRing(rect, cam, center, forward, right, gizmoRadius, segments,
                _activeGizmoAxis == GizmoAxis.Y ? Color.yellow : (_hoveredGizmoAxis == GizmoAxis.Y ? Color.white : new Color(0.30f, 0.90f, 0.35f, 0.95f)),
                _activeGizmoAxis == GizmoAxis.Y || _hoveredGizmoAxis == GizmoAxis.Y ? 4.0f : 2.5f);

            // Z Ring (Roll / Blue)
            DrawGizmoRing(rect, cam, center, right, up, gizmoRadius, segments,
                _activeGizmoAxis == GizmoAxis.Z ? Color.yellow : (_hoveredGizmoAxis == GizmoAxis.Z ? Color.white : new Color(0.25f, 0.60f, 1.00f, 0.95f)),
                _activeGizmoAxis == GizmoAxis.Z || _hoveredGizmoAxis == GizmoAxis.Z ? 4.0f : 2.5f);

            // Screen View Ring (White)
            Vector3 camRight = cam.transform.right;
            Vector3 camUp = cam.transform.up;
            DrawGizmoRing(rect, cam, center, camRight, camUp, gizmoRadius * 1.18f, segments,
                _activeGizmoAxis == GizmoAxis.Screen ? Color.yellow : (_hoveredGizmoAxis == GizmoAxis.Screen ? Color.white : new Color(0.85f, 0.88f, 0.95f, 0.5f)),
                _activeGizmoAxis == GizmoAxis.Screen || _hoveredGizmoAxis == GizmoAxis.Screen ? 3.0f : 1.5f);

            Handles.EndGUI();
        }

        private void DrawGizmoRing(Rect rect, Camera cam, Vector3 center, Vector3 u, Vector3 v, float radius, int segments, Color color, float width)
        {
            var points = new List<Vector3>();
            for (int i = 0; i <= segments; i++)
            {
                float angle = (i / (float)segments) * Mathf.PI * 2f;
                Vector3 worldPt = center + (Mathf.Cos(angle) * u + Mathf.Sin(angle) * v) * radius;
                if (TryWorldToScreen(cam, rect, worldPt, out Vector2 screenPt))
                {
                    points.Add(new Vector3(screenPt.x, screenPt.y, 0));
                }
            }

            if (points.Count > 1)
            {
                Handles.color = color;
                Handles.DrawAAPolyLine(width, points.ToArray());
            }
        }

        private bool HandleGizmoAndSelectionEvents(Rect rect, Event evt)
        {
            if (_previewUtility == null || _previewInstance == null) return false;
            Camera cam = _previewUtility.camera;
            if (cam == null) return false;

            Vector2 mousePos = evt.mousePosition;
            bool containsMouse = rect.Contains(mousePos);

            // 1. Mouse Move: Hover Detection
            if (evt.type == EventType.MouseMove && containsMouse)
            {
                SmplxJoint? prevHoverJoint = _hoveredJoint;
                GizmoAxis prevHoverAxis = _hoveredGizmoAxis;

                _hoveredGizmoAxis = HitTestGizmoAxis(rect, cam, mousePos);
                if (_hoveredGizmoAxis == GizmoAxis.None)
                {
                    _hoveredJoint = HitTestJoint(rect, cam, mousePos);
                }
                else
                {
                    _hoveredJoint = null;
                }

                if (prevHoverJoint != _hoveredJoint || prevHoverAxis != _hoveredGizmoAxis)
                {
                    Repaint();
                }
            }

            // 2. Mouse Down: Click Selection or Start Gizmo Drag
            if (evt.type == EventType.MouseDown && evt.button == 0 && containsMouse)
            {
                // Test Gizmo Axis first if a joint is already selected
                if (_selectedJoint.HasValue && _showGizmos)
                {
                    GizmoAxis hitAxis = HitTestGizmoAxis(rect, cam, mousePos);
                    if (hitAxis != GizmoAxis.None)
                    {
                        _activeGizmoAxis = hitAxis;
                        _dragStartMousePos = mousePos;
                        _dragStartEuler = _data.GetJointEuler(_currentFrame, _selectedJoint.Value);
                        _dragStartLocalRot = _previewBoneMap[_selectedJoint.Value].localRotation;
                        _data.RecordUndo($"Rotate {_selectedJoint.Value} Frame {_currentFrame}");
                        evt.Use();
                        return true;
                    }
                }

                // Test Joint Node click
                if (_showSkeleton)
                {
                    SmplxJoint? hitJoint = HitTestJoint(rect, cam, mousePos);
                    if (hitJoint.HasValue)
                    {
                        _selectedJoint = hitJoint.Value;
                        FocusJointInInspector(hitJoint.Value);
                        _activeGizmoAxis = GizmoAxis.None;
                        evt.Use();
                        Repaint();
                        return true;
                    }
                }

                // Clicked on empty space with Gizmo active: check distance to clear selection
                if (_selectedJoint.HasValue)
                {
                    if (_previewBoneMap.TryGetValue(_selectedJoint.Value, out Transform curBone) && curBone != null)
                    {
                        if (TryWorldToScreen(cam, rect, curBone.position, out Vector2 curCenter))
                        {
                            if (Vector2.Distance(mousePos, curCenter) > 80f && !evt.shift)
                            {
                                _selectedJoint = null;
                                Repaint();
                            }
                        }
                    }
                }
            }

            // 3. Mouse Drag: Rotate Bone via Gizmo
            if (evt.type == EventType.MouseDrag && _activeGizmoAxis != GizmoAxis.None && _selectedJoint.HasValue)
            {
                SmplxJoint joint = _selectedJoint.Value;
                if (_previewBoneMap.TryGetValue(joint, out Transform bone) && bone != null)
                {
                    Vector2 delta = mousePos - _dragStartMousePos;

                    if (_activeGizmoAxis == GizmoAxis.Screen)
                    {
                        if (TryWorldToScreen(cam, rect, bone.position, out Vector2 center))
                        {
                            float a1 = Mathf.Atan2(_dragStartMousePos.y - center.y, _dragStartMousePos.x - center.x) * Mathf.Rad2Deg;
                            float a2 = Mathf.Atan2(mousePos.y - center.y, mousePos.x - center.x) * Mathf.Rad2Deg;
                            float deltaAngle = Mathf.DeltaAngle(a1, a2);

                            Vector3 newEuler = _dragStartEuler;
                            newEuler.z += deltaAngle;
                            _data.SetJointEuler(_currentFrame, joint, newEuler);
                            ApplyCurrentFrameToPreview();
                        }
                    }
                    else
                    {
                        Vector3 newEuler = _dragStartEuler;
                        float sensitivity = 0.65f;

                        if (_activeGizmoAxis == GizmoAxis.X)
                        {
                            newEuler.x += -delta.y * sensitivity;
                        }
                        else if (_activeGizmoAxis == GizmoAxis.Y)
                        {
                            newEuler.y += delta.x * sensitivity;
                        }
                        else if (_activeGizmoAxis == GizmoAxis.Z)
                        {
                            newEuler.z += (delta.x - delta.y) * 0.5f * sensitivity;
                        }

                        _data.SetJointEuler(_currentFrame, joint, newEuler);
                        ApplyCurrentFrameToPreview();
                    }

                    evt.Use();
                    Repaint();
                    return true;
                }
            }

            // 4. Mouse Up: Release Gizmo Drag
            if (evt.type == EventType.MouseUp && _activeGizmoAxis != GizmoAxis.None)
            {
                _activeGizmoAxis = GizmoAxis.None;
                evt.Use();
                Repaint();
                return true;
            }

            return false;
        }

        private SmplxJoint? HitTestJoint(Rect rect, Camera cam, Vector2 mousePos)
        {
            SmplxJoint? nearestJoint = null;
            float nearestDist = 14f; // 14px threshold

            foreach (var kvp in _previewBoneMap)
            {
                Transform bone = kvp.Value;
                if (bone == null) continue;

                if (TryWorldToScreen(cam, rect, bone.position, out Vector2 screenPt))
                {
                    float d = Vector2.Distance(mousePos, screenPt);
                    if (d < nearestDist)
                    {
                        nearestDist = d;
                        nearestJoint = kvp.Key;
                    }
                }
            }

            return nearestJoint;
        }

        private GizmoAxis HitTestGizmoAxis(Rect rect, Camera cam, Vector2 mousePos)
        {
            if (!_selectedJoint.HasValue || !_showGizmos) return GizmoAxis.None;
            if (!_previewBoneMap.TryGetValue(_selectedJoint.Value, out Transform bone) || bone == null) return GizmoAxis.None;

            Vector3 center = bone.position;
            if (!TryWorldToScreen(cam, rect, center, out Vector2 centerScreen)) return GizmoAxis.None;

            float gizmoRadius = 0.16f * _previewDistance;
            int segments = 24;
            float threshold = 9f; // 9px hit threshold

            // 1. Screen View Ring
            Vector3 camRight = cam.transform.right;
            Vector3 camUp = cam.transform.up;
            if (DistanceToRing(rect, cam, center, camRight, camUp, gizmoRadius * 1.18f, segments, mousePos) < threshold)
            {
                return GizmoAxis.Screen;
            }

            // 2. X Ring (Pitch / Red)
            if (DistanceToRing(rect, cam, center, bone.up, bone.forward, gizmoRadius, segments, mousePos) < threshold)
            {
                return GizmoAxis.X;
            }

            // 3. Y Ring (Yaw / Green)
            if (DistanceToRing(rect, cam, center, bone.forward, bone.right, gizmoRadius, segments, mousePos) < threshold)
            {
                return GizmoAxis.Y;
            }

            // 4. Z Ring (Roll / Blue)
            if (DistanceToRing(rect, cam, center, bone.right, bone.up, gizmoRadius, segments, mousePos) < threshold)
            {
                return GizmoAxis.Z;
            }

            return GizmoAxis.None;
        }

        private float DistanceToRing(Rect rect, Camera cam, Vector3 center, Vector3 u, Vector3 v, float radius, int segments, Vector2 mousePos)
        {
            float minDist = float.MaxValue;
            Vector2 prevPt = Vector2.zero;
            bool hasPrev = false;

            for (int i = 0; i <= segments; i++)
            {
                float angle = (i / (float)segments) * Mathf.PI * 2f;
                Vector3 worldPt = center + (Mathf.Cos(angle) * u + Mathf.Sin(angle) * v) * radius;
                if (TryWorldToScreen(cam, rect, worldPt, out Vector2 currPt))
                {
                    if (hasPrev)
                    {
                        float d = DistanceToLineSegment(mousePos, prevPt, currPt);
                        if (d < minDist) minDist = d;
                    }
                    prevPt = currPt;
                    hasPrev = true;
                }
                else
                {
                    hasPrev = false;
                }
            }

            return minDist;
        }

        private static float DistanceToLineSegment(Vector2 p, Vector2 a, Vector2 b)
        {
            Vector2 ab = b - a;
            float sqrLen = ab.sqrMagnitude;
            if (sqrLen < 0.0001f) return Vector2.Distance(p, a);

            float t = Mathf.Clamp01(Vector2.Dot(p - a, ab) / sqrLen);
            Vector2 proj = a + t * ab;
            return Vector2.Distance(p, proj);
        }

        private void DrawViewportHUD(Rect rect)
        {
            // Top-Left Toolbar: Skeleton & Gizmos Toggles
            Rect toolBarRect = new Rect(rect.x + 8, rect.y + 8, 160, 24);
            GUILayout.BeginArea(toolBarRect);
            EditorGUILayout.BeginHorizontal();

            GUI.backgroundColor = _showSkeleton ? new Color(0.3f, 0.85f, 0.45f) : new Color(0.35f, 0.35f, 0.35f);
            _showSkeleton = GUILayout.Toggle(_showSkeleton, "🦴 Skeleton", EditorStyles.miniButtonLeft, GUILayout.Width(78), GUILayout.Height(22));

            GUI.backgroundColor = _showGizmos ? new Color(0.3f, 0.7f, 1.0f) : new Color(0.35f, 0.35f, 0.35f);
            _showGizmos = GUILayout.Toggle(_showGizmos, "🎯 Gizmos", EditorStyles.miniButtonRight, GUILayout.Width(68), GUILayout.Height(22));

            GUI.backgroundColor = Color.white;
            EditorGUILayout.EndHorizontal();
            GUILayout.EndArea();

            // Top-Right: Camera Reset & Active Bone Badge
            float rightWidth = _selectedJoint.HasValue ? 280 : 95;
            Rect rightBarRect = new Rect(rect.xMax - rightWidth - 8, rect.y + 8, rightWidth, 24);
            GUILayout.BeginArea(rightBarRect);
            EditorGUILayout.BeginHorizontal();

            if (_selectedJoint.HasValue)
            {
                var badgeStyle = new GUIStyle(EditorStyles.miniBoldLabel)
                {
                    normal = { textColor = new Color(1.0f, 0.85f, 0.25f) }
                };
                string friendlyName = GetJointFriendlyName(_selectedJoint.Value);
                EditorGUILayout.LabelField($"🎯 {friendlyName}", badgeStyle, GUILayout.Width(130));

                if (GUILayout.Button("Clear", EditorStyles.miniButton, GUILayout.Width(42), GUILayout.Height(20)))
                {
                    _selectedJoint = null;
                }

                if (GUILayout.Button("Reset", EditorStyles.miniButton, GUILayout.Width(45), GUILayout.Height(20)))
                {
                    _data.ResetJointToOriginal(_currentFrame, _selectedJoint.Value);
                    ApplyCurrentFrameToPreview();
                }
            }

            if (GUILayout.Button("🔄 Reset Cam", EditorStyles.miniButton, GUILayout.Width(88), GUILayout.Height(20)))
            {
                _previewDir = new Vector2(180f, 10f);
                _previewDistance = 2.8f;
                _previewPivot = _previewHipsTransform != null
                    ? new Vector3(0, _previewHipsTransform.position.y + 0.2f, 0)
                    : new Vector3(0, 1.0f, 0);
            }

            EditorGUILayout.EndHorizontal();
            GUILayout.EndArea();

            // Bottom-Left Hint overlay
            Rect hintRect = new Rect(rect.x + 8, rect.y + rect.height - 22, 290, 18);
            EditorGUI.DrawRect(hintRect, new Color(0.06f, 0.08f, 0.12f, 0.75f));
            var hintStyle = new GUIStyle(EditorStyles.miniLabel)
            {
                normal = { textColor = new Color(0.7f, 0.75f, 0.85f) },
                alignment = TextAnchor.MiddleLeft,
                padding = new RectOffset(6, 0, 1, 0)
            };
            string hint = _selectedJoint.HasValue
                ? $"Drag Ring to Rotate ({GetJointFriendlyName(_selectedJoint.Value)})"
                : "Click Bone to Select | Drag: Orbit | Scroll: Zoom";
            GUI.Label(hintRect, hint, hintStyle);
        }

        private void FocusJointInInspector(SmplxJoint joint)
        {
            string name = joint.ToString();
            if (name.StartsWith("L_Collar") || name.StartsWith("L_Shoulder") || name.StartsWith("L_Elbow") || name.StartsWith("L_Wrist"))
            {
                _foldoutLeftArm = true;
            }
            else if (name.StartsWith("R_Collar") || name.StartsWith("R_Shoulder") || name.StartsWith("R_Elbow") || name.StartsWith("R_Wrist"))
            {
                _foldoutRightArm = true;
            }
            else if (name.StartsWith("L_Hip") || name.StartsWith("L_Knee") || name.StartsWith("L_Ankle") || name.StartsWith("L_Foot"))
            {
                _foldoutLeftLeg = true;
            }
            else if (name.StartsWith("R_Hip") || name.StartsWith("R_Knee") || name.StartsWith("R_Ankle") || name.StartsWith("R_Foot"))
            {
                _foldoutRightLeg = true;
            }
            else if (joint == SmplxJoint.Pelvis)
            {
                _foldoutRoot = true;
            }
            else
            {
                _foldoutTorso = true;
            }
        }

        public static string GetJointFriendlyName(SmplxJoint joint)
        {
            switch (joint)
            {
                case SmplxJoint.Pelvis: return "Hips / Root";
                case SmplxJoint.L_Hip: return "Left Hip";
                case SmplxJoint.R_Hip: return "Right Hip";
                case SmplxJoint.Spine1: return "Spine (Lower)";
                case SmplxJoint.L_Knee: return "Left Knee";
                case SmplxJoint.R_Knee: return "Right Knee";
                case SmplxJoint.Spine2: return "Chest (Middle)";
                case SmplxJoint.L_Ankle: return "Left Ankle";
                case SmplxJoint.R_Ankle: return "Right Ankle";
                case SmplxJoint.Spine3: return "Upper Chest";
                case SmplxJoint.L_Foot: return "Left Toes";
                case SmplxJoint.R_Foot: return "Right Toes";
                case SmplxJoint.Neck: return "Neck";
                case SmplxJoint.L_Collar: return "Left Collar (Shoulder)";
                case SmplxJoint.R_Collar: return "Right Collar (Shoulder)";
                case SmplxJoint.Head: return "Head";
                case SmplxJoint.L_Shoulder: return "Left Upper Arm";
                case SmplxJoint.R_Shoulder: return "Right Upper Arm";
                case SmplxJoint.L_Elbow: return "Left Elbow";
                case SmplxJoint.R_Elbow: return "Right Elbow";
                case SmplxJoint.L_Wrist: return "Left Wrist (Hand)";
                case SmplxJoint.R_Wrist: return "Right Wrist (Hand)";
                default: return joint.ToString();
            }
        }

        private static Color GetBoneLineColor(SmplxJoint parent, SmplxJoint child)
        {
            string childName = child.ToString();
            if (childName.StartsWith("L_")) return new Color(0.2f, 0.9f, 0.55f, 0.75f); // Left: Emerald Green
            if (childName.StartsWith("R_")) return new Color(0.95f, 0.35f, 0.45f, 0.75f); // Right: Coral Red
            return new Color(0.55f, 0.75f, 1.0f, 0.75f); // Spine / Torso: Soft Sky Blue
        }

        private static Color GetJointColor(SmplxJoint joint, bool isHovered, bool isSelected)
        {
            if (isSelected) return new Color(1.0f, 0.85f, 0.20f, 1.0f); // Gold
            if (isHovered) return new Color(1.0f, 1.0f, 1.0f, 1.0f);     // Pure White

            string name = joint.ToString();
            if (name.StartsWith("L_")) return new Color(0.2f, 0.9f, 0.55f, 0.9f);
            if (name.StartsWith("R_")) return new Color(0.95f, 0.35f, 0.45f, 0.9f);
            return new Color(0.45f, 0.75f, 1.0f, 0.9f);
        }

        #endregion

        private void DrawVideoTexture(Rect rect)
        {
            if (rect.width <= 4f || rect.height <= 4f) return;

            // Background
            EditorGUI.DrawRect(rect, new Color(0.08f, 0.09f, 0.11f));

            if (_videoTexture != null && _videoPlayer != null && _videoPlayer.isPrepared)
            {
                GUI.DrawTexture(rect, _videoTexture, ScaleMode.ScaleToFit);
            }
            else
            {
                var labelStyle = new GUIStyle(EditorStyles.centeredGreyMiniLabel)
                {
                    alignment = TextAnchor.MiddleCenter
                };
                EditorGUI.LabelField(rect, "Loading Video Stream...", labelStyle);
            }
        }

        #endregion

        #region Right Pose Inspector

        private void DrawPoseInspector(float height)
        {
            _inspectorScrollPos = EditorGUILayout.BeginScrollView(_inspectorScrollPos);

            // Frame Header & Navigation
            EditorGUILayout.BeginVertical(EditorStyles.helpBox);
            EditorGUILayout.BeginHorizontal();

            EditorGUILayout.LabelField($"📍 Frame {_currentFrame + 1} / {_data.Frames}", EditorStyles.boldLabel);
            float curTime = _data.Timestamps != null && _data.Timestamps.Length > _currentFrame
                ? _data.Timestamps[_currentFrame]
                : (float)_currentFrame / _data.FrameRate;
            EditorGUILayout.LabelField($"Time: {curTime:F2}s", EditorStyles.miniLabel, GUILayout.Width(75));

            bool isMod = _data.IsFrameModified(_currentFrame);
            Color badgeCol = isMod ? new Color(1.0f, 0.75f, 0.2f) : new Color(0.4f, 0.8f, 0.4f);
            string badgeText = isMod ? "● Modified" : "✓ Original";
            var badgeStyle = new GUIStyle(EditorStyles.miniBoldLabel) { normal = { textColor = badgeCol } };
            EditorGUILayout.LabelField(badgeText, badgeStyle, GUILayout.Width(75));

            EditorGUILayout.EndHorizontal();

            // Quick Frame Step Buttons
            EditorGUILayout.BeginHorizontal();
            if (GUILayout.Button("◀ Prev Frame", EditorStyles.miniButton, GUILayout.Height(22)))
            {
                SetCurrentFrame(_currentFrame - 1);
            }
            int newFrame = EditorGUILayout.IntSlider(_currentFrame + 1, 1, _data.Frames) - 1;
            if (newFrame != _currentFrame)
            {
                SetCurrentFrame(newFrame);
            }
            if (GUILayout.Button("Next Frame ▶", EditorStyles.miniButton, GUILayout.Height(22)))
            {
                SetCurrentFrame(_currentFrame + 1);
            }
            EditorGUILayout.EndHorizontal();

            EditorGUILayout.EndVertical();

            EditorGUILayout.Space(6);

            // Frame Operation Tools
            _foldoutTools = EditorGUILayout.Foldout(_foldoutTools, "🛠️ Frame Pose Tools", true, EditorStyles.foldoutHeader);
            if (_foldoutTools)
            {
                EditorGUILayout.BeginVertical(EditorStyles.helpBox);

                EditorGUILayout.BeginHorizontal();
                if (GUILayout.Button(new GUIContent("📋 Copy", "Copy current frame pose to clipboard"), EditorStyles.miniButtonLeft, GUILayout.Height(26)))
                {
                    _data.CopyFramePose(_currentFrame);
                }
                GUI.enabled = EditableMotionData.HasClipboardData;
                if (GUILayout.Button(new GUIContent("📄 Paste", "Paste clipboard pose onto this frame"), EditorStyles.miniButtonRight, GUILayout.Height(26)))
                {
                    _data.PasteFramePose(_currentFrame);
                    ApplyCurrentFrameToPreview();
                }
                GUI.enabled = true;

                if (GUILayout.Button(new GUIContent("🔄 Mirror", "Mirror pose across left and right limbs"), EditorStyles.miniButton, GUILayout.Height(26)))
                {
                    _data.MirrorFrame(_currentFrame);
                    ApplyCurrentFrameToPreview();
                }

                if (GUILayout.Button(new GUIContent("⚖️ Smooth", "Interpolate with neighbor frames (smoothing)"), EditorStyles.miniButton, GUILayout.Height(26)))
                {
                    _data.SmoothFrame(_currentFrame);
                    ApplyCurrentFrameToPreview();
                }
                EditorGUILayout.EndHorizontal();

                EditorGUILayout.BeginHorizontal();
                if (GUILayout.Button(new GUIContent("↩️ Reset Frame", "Revert this frame to original generated pose"), EditorStyles.miniButton, GUILayout.Height(22)))
                {
                    _data.ResetFrameToOriginal(_currentFrame);
                    ApplyCurrentFrameToPreview();
                }
                if (GUILayout.Button(new GUIContent("🧘 T-Pose", "Set this frame to neutral T-Pose"), EditorStyles.miniButton, GUILayout.Height(22)))
                {
                    _data.SetFrameToTPose(_currentFrame);
                    ApplyCurrentFrameToPreview();
                }
                EditorGUILayout.EndHorizontal();

                EditorGUILayout.EndVertical();
            }

            EditorGUILayout.Space(6);

            // Bone & Joint Groups
            DrawBoneSection("👤 Root & Hips", ref _foldoutRoot, () =>
            {
                DrawRootPositionInspector();
                DrawJointRotationSlider(SmplxJoint.Pelvis, "Pelvis / Hips Rotation");
            });

            DrawBoneSection("🦴 Torso & Head", ref _foldoutTorso, () =>
            {
                DrawJointRotationSlider(SmplxJoint.Spine1, "Spine (Lower)");
                DrawJointRotationSlider(SmplxJoint.Spine2, "Chest (Middle)");
                DrawJointRotationSlider(SmplxJoint.Spine3, "Upper Chest");
                DrawJointRotationSlider(SmplxJoint.Neck, "Neck");
                DrawJointRotationSlider(SmplxJoint.Head, "Head");
            });

            DrawBoneSection("💪 Left Arm", ref _foldoutLeftArm, () =>
            {
                DrawJointRotationSlider(SmplxJoint.L_Collar, "Left Shoulder (Collar)");
                DrawJointRotationSlider(SmplxJoint.L_Shoulder, "Left Upper Arm");
                DrawJointRotationSlider(SmplxJoint.L_Elbow, "Left Elbow (Forearm)");
                DrawJointRotationSlider(SmplxJoint.L_Wrist, "Left Wrist (Hand)");
            });

            DrawBoneSection("💪 Right Arm", ref _foldoutRightArm, () =>
            {
                DrawJointRotationSlider(SmplxJoint.R_Collar, "Right Shoulder (Collar)");
                DrawJointRotationSlider(SmplxJoint.R_Shoulder, "Right Upper Arm");
                DrawJointRotationSlider(SmplxJoint.R_Elbow, "Right Elbow (Forearm)");
                DrawJointRotationSlider(SmplxJoint.R_Wrist, "Right Wrist (Hand)");
            });

            DrawBoneSection("🦵 Left Leg", ref _foldoutLeftLeg, () =>
            {
                DrawJointRotationSlider(SmplxJoint.L_Hip, "Left Hip (Upper Leg)");
                DrawJointRotationSlider(SmplxJoint.L_Knee, "Left Knee (Lower Leg)");
                DrawJointRotationSlider(SmplxJoint.L_Ankle, "Left Ankle (Foot)");
                DrawJointRotationSlider(SmplxJoint.L_Foot, "Left Toes");
            });

            DrawBoneSection("🦵 Right Leg", ref _foldoutRightLeg, () =>
            {
                DrawJointRotationSlider(SmplxJoint.R_Hip, "Right Hip (Upper Leg)");
                DrawJointRotationSlider(SmplxJoint.R_Knee, "Right Knee (Lower Leg)");
                DrawJointRotationSlider(SmplxJoint.R_Ankle, "Right Ankle (Foot)");
                DrawJointRotationSlider(SmplxJoint.R_Foot, "Right Toes");
            });

            EditorGUILayout.EndScrollView();
        }

        private void DrawBoneSection(string title, ref bool foldout, Action drawContents)
        {
            foldout = EditorGUILayout.Foldout(foldout, title, true, EditorStyles.foldoutHeader);
            if (foldout)
            {
                EditorGUILayout.BeginVertical(EditorStyles.helpBox);
                drawContents();
                EditorGUILayout.EndVertical();
                EditorGUILayout.Space(4);
            }
        }

        private void DrawRootPositionInspector()
        {
            bool isSelected = (_selectedJoint == SmplxJoint.Pelvis);
            if (isSelected)
            {
                GUI.backgroundColor = new Color(1.0f, 0.9f, 0.4f, 0.35f);
                EditorGUILayout.BeginVertical(EditorStyles.helpBox);
                GUI.backgroundColor = Color.white;
            }

            EditorGUILayout.BeginHorizontal();
            EditorGUILayout.LabelField("Root Position (Offset):", EditorStyles.miniBoldLabel);
            GUI.backgroundColor = isSelected ? new Color(1.0f, 0.85f, 0.2f) : Color.white;
            if (GUILayout.Button(isSelected ? "🎯 Active" : "🎯 Select", EditorStyles.miniButton, GUILayout.Width(isSelected ? 65 : 60)))
            {
                _selectedJoint = isSelected ? (SmplxJoint?)null : SmplxJoint.Pelvis;
                Repaint();
            }
            GUI.backgroundColor = Color.white;
            EditorGUILayout.EndHorizontal();

            Vector3 rootPos = _data.GetRootPosition(_currentFrame);

            EditorGUI.BeginChangeCheck();
            float x = EditorGUILayout.Slider("X (Left/Right)", rootPos.x, -2.0f, 2.0f);
            float y = EditorGUILayout.Slider("Y (Height)", rootPos.y, -2.0f, 2.0f);
            float z = EditorGUILayout.Slider("Z (Fwd/Back)", rootPos.z, -2.0f, 2.0f);

            if (EditorGUI.EndChangeCheck())
            {
                _data.RecordUndo($"Adjust Root Position Frame {_currentFrame}");
                _data.SetRootPosition(_currentFrame, new Vector3(x, y, z));
                ApplyCurrentFrameToPreview();
            }

            if (isSelected)
            {
                EditorGUILayout.EndVertical();
            }
        }

        private void DrawJointRotationSlider(SmplxJoint joint, string label)
        {
            bool isSelected = (_selectedJoint == joint);
            Vector3 euler = _data.GetJointEuler(_currentFrame, joint);

            if (isSelected)
            {
                GUI.backgroundColor = new Color(1.0f, 0.9f, 0.4f, 0.35f);
                EditorGUILayout.BeginVertical(EditorStyles.helpBox);
                GUI.backgroundColor = Color.white;
            }

            EditorGUILayout.BeginHorizontal();

            // Bone select toggle button
            GUI.backgroundColor = isSelected ? new Color(1.0f, 0.85f, 0.2f) : Color.white;
            if (GUILayout.Button(isSelected ? "🎯 Active" : "🎯", EditorStyles.miniButton, GUILayout.Width(isSelected ? 62 : 28)))
            {
                _selectedJoint = isSelected ? (SmplxJoint?)null : joint;
                Repaint();
            }
            GUI.backgroundColor = Color.white;

            var titleStyle = isSelected ? EditorStyles.boldLabel : EditorStyles.miniBoldLabel;
            EditorGUILayout.LabelField(label, titleStyle);

            if (GUILayout.Button("Reset", EditorStyles.miniButton, GUILayout.Width(45)))
            {
                _data.ResetJointToOriginal(_currentFrame, joint);
                ApplyCurrentFrameToPreview();
                if (isSelected) EditorGUILayout.EndVertical();
                return;
            }
            EditorGUILayout.EndHorizontal();

            EditorGUI.BeginChangeCheck();
            float rx = EditorGUILayout.Slider("Pitch (X)", euler.x, -180f, 180f);
            float ry = EditorGUILayout.Slider("Yaw (Y)", euler.y, -180f, 180f);
            float rz = EditorGUILayout.Slider("Roll (Z)", euler.z, -180f, 180f);

            if (EditorGUI.EndChangeCheck())
            {
                _data.RecordUndo($"Rotate {joint} on Frame {_currentFrame}");
                _data.SetJointEuler(_currentFrame, joint, new Vector3(rx, ry, rz));
                ApplyCurrentFrameToPreview();
            }

            if (isSelected)
            {
                EditorGUILayout.EndVertical();
            }

            EditorGUILayout.Space(2);
        }

        #endregion

        #region Bottom Timeline Panel

        private void DrawBottomTimelinePanel()
        {
            Rect panelRect = GUILayoutUtility.GetRect(position.width, TIMELINE_HEIGHT, GUILayout.ExpandWidth(true));
            GUI.Box(panelRect, GUIContent.none, _timelineBgStyle);

            // Sub-bar 1: Playback Controls (height 36px)
            Rect ctrlRect = new Rect(panelRect.x + 8, panelRect.y + 4, panelRect.width - 16, 32);
            DrawPlaybackControls(ctrlRect);

            // Sub-bar 2: Timeline Ruler and Frames Track (height 90px)
            Rect trackRect = new Rect(panelRect.x + 8, panelRect.y + 40, panelRect.width - 16, panelRect.height - 46);
            DrawTimelineTrack(trackRect);
        }

        private void DrawPlaybackControls(Rect rect)
        {
            GUILayout.BeginArea(rect);
            EditorGUILayout.BeginHorizontal();

            // First Frame
            if (GUILayout.Button(new GUIContent("|◀", "First Frame (Home)"), EditorStyles.miniButtonLeft, GUILayout.Width(36), GUILayout.Height(28)))
            {
                SetCurrentFrame(0);
            }

            // Prev Keyframe (modified frame)
            if (GUILayout.Button(new GUIContent("◆◀", "Previous Modified Frame"), EditorStyles.miniButtonMid, GUILayout.Width(36), GUILayout.Height(28)))
            {
                JumpToPreviousModifiedFrame();
            }

            // Step Back 1 Frame
            if (GUILayout.Button(new GUIContent("◀", "Previous Frame (Left Arrow)"), EditorStyles.miniButtonMid, GUILayout.Width(36), GUILayout.Height(28)))
            {
                SetCurrentFrame(_currentFrame - 1);
            }

            // Play / Pause
            GUI.backgroundColor = _isPlaying ? new Color(0.95f, 0.65f, 0.2f) : new Color(0.3f, 0.85f, 0.45f);
            string playIcon = _isPlaying ? "⏸ Pause" : "▶ Play";
            if (GUILayout.Button(new GUIContent(playIcon, "Toggle Playback (Space)"), EditorStyles.miniButtonMid, GUILayout.Width(75), GUILayout.Height(28)))
            {
                TogglePlay();
            }
            GUI.backgroundColor = Color.white;

            // Step Forward 1 Frame
            if (GUILayout.Button(new GUIContent("▶", "Next Frame (Right Arrow)"), EditorStyles.miniButtonMid, GUILayout.Width(36), GUILayout.Height(28)))
            {
                SetCurrentFrame(_currentFrame + 1);
            }

            // Next Keyframe
            if (GUILayout.Button(new GUIContent("▶◆", "Next Modified Frame"), EditorStyles.miniButtonMid, GUILayout.Width(36), GUILayout.Height(28)))
            {
                JumpToNextModifiedFrame();
            }

            // Last Frame
            if (GUILayout.Button(new GUIContent("▶|", "Last Frame (End)"), EditorStyles.miniButtonRight, GUILayout.Width(36), GUILayout.Height(28)))
            {
                SetCurrentFrame(_data.Frames - 1);
            }

            EditorGUILayout.Space(12);

            // Loop Toggle
            _isLoop = GUILayout.Toggle(_isLoop, "🔁 Loop", EditorStyles.miniButton, GUILayout.Width(65), GUILayout.Height(28));

            EditorGUILayout.Space(8);

            // Playback Speed
            EditorGUILayout.LabelField("Speed:", EditorStyles.miniLabel, GUILayout.Width(40));
            string[] speeds = new string[] { "0.25x", "0.5x", "1.0x", "2.0x" };
            float[] speedVals = new float[] { 0.25f, 0.5f, 1.0f, 2.0f };
            int curSpeedIdx = 2;
            for (int s = 0; s < speedVals.Length; s++)
            {
                if (Mathf.Abs(_playbackSpeed - speedVals[s]) < 0.05f) curSpeedIdx = s;
            }
            int newSpeedIdx = EditorGUILayout.Popup(curSpeedIdx, speeds, GUILayout.Width(65), GUILayout.Height(26));
            _playbackSpeed = speedVals[newSpeedIdx];

            GUILayout.FlexibleSpace();

            // Timeline Zoom Slider
            EditorGUILayout.LabelField("Zoom:", EditorStyles.miniLabel, GUILayout.Width(38));
            _pixelsPerFrame = EditorGUILayout.Slider(_pixelsPerFrame, 8.0f, 60.0f, GUILayout.Width(110));

            EditorGUILayout.EndHorizontal();
            GUILayout.EndArea();
        }

        private void DrawTimelineTrack(Rect rect)
        {
            int totalFrames = _data.Frames;
            float totalTrackWidth = totalFrames * _pixelsPerFrame;

            // Handle Mouse Wheel Zoom & Pan within track
            Event evt = Event.current;
            if (rect.Contains(evt.mousePosition))
            {
                if (evt.type == EventType.ScrollWheel)
                {
                    if (evt.control || evt.command) // Zoom
                    {
                        _pixelsPerFrame = Mathf.Clamp(_pixelsPerFrame - evt.delta.y * 1.5f, 8.0f, 60.0f);
                        evt.Use();
                        Repaint();
                    }
                }
            }

            // Horizontal ScrollView for timeline
            Rect viewRect = new Rect(0, 0, Mathf.Max(rect.width, totalTrackWidth + 40f), rect.height - 18);
            _timelineScrollPos = GUI.BeginScrollView(rect, _timelineScrollPos, viewRect, true, false);

            // Draw Background Grid & Tick Marks
            float rulerHeight = 22f;
            Rect rulerRect = new Rect(0, 0, viewRect.width, rulerHeight);
            EditorGUI.DrawRect(rulerRect, new Color(0.14f, 0.16f, 0.20f));

            float frameTrackY = rulerHeight;
            float frameTrackHeight = viewRect.height - rulerHeight;

            // Draw Frame Bars & Ticks
            int step = _pixelsPerFrame < 15f ? 10 : (_pixelsPerFrame < 30f ? 5 : 1);

            for (int f = 0; f < totalFrames; f++)
            {
                float x = f * _pixelsPerFrame;
                bool isModified = _data.IsFrameModified(f);
                bool isCurrent = (f == _currentFrame);

                // Ruler Ticks
                if (f % step == 0)
                {
                    float time = (float)f / _data.FrameRate;
                    string label = $"{f}\n{time:F2}s";
                    Rect tickLabelRect = new Rect(x - 15, 2, 40, rulerHeight);
                    GUI.Label(tickLabelRect, label, _rulerTextStyle);

                    // Grid line
                    EditorGUI.DrawRect(new Rect(x, rulerHeight, 1, frameTrackHeight), new Color(0.25f, 0.28f, 0.35f, 0.4f));
                }

                // Frame Cell Box
                Rect frameBox = new Rect(x + 1, frameTrackY + 4, _pixelsPerFrame - 2, frameTrackHeight - 8);

                Color boxCol = isCurrent
                    ? new Color(0.2f, 0.55f, 0.95f, 0.45f)
                    : (isModified ? new Color(0.9f, 0.7f, 0.2f, 0.25f) : new Color(0.18f, 0.20f, 0.25f, 0.6f));

                EditorGUI.DrawRect(frameBox, boxCol);

                // Modified Keyframe Marker (Diamond Icon)
                if (isModified)
                {
                    Rect diamondRect = new Rect(x + (_pixelsPerFrame * 0.5f) - 6, frameTrackY + 12, 12, 12);
                    GUI.Label(diamondRect, "◆", _keyframeMarkerStyle);
                }
            }

            // Current Frame Scrubber Line (Red Cursor)
            float scrubberX = _currentFrame * _pixelsPerFrame + (_pixelsPerFrame * 0.5f);
            Rect scrubberLine = new Rect(scrubberX - 1.5f, 0, 3f, viewRect.height);
            EditorGUI.DrawRect(scrubberLine, new Color(1.0f, 0.3f, 0.3f, 0.95f));

            // Scrubber Top Triangle
            Rect scrubberHead = new Rect(scrubberX - 6f, 0, 12f, 10f);
            EditorGUI.DrawRect(scrubberHead, new Color(1.0f, 0.35f, 0.35f, 1.0f));

            // Mouse Click & Drag Scrubber Interaction
            int trackControlID = GUIUtility.GetControlID(FocusType.Passive);
            switch (evt.GetTypeForControl(trackControlID))
            {
                case EventType.MouseDown:
                    if (evt.button == 0 && viewRect.Contains(evt.mousePosition))
                    {
                        GUIUtility.hotControl = trackControlID;
                        _isDraggingScrubber = true;
                        int clickedFrame = Mathf.Clamp(Mathf.FloorToInt(evt.mousePosition.x / _pixelsPerFrame), 0, totalFrames - 1);
                        SetCurrentFrame(clickedFrame);
                        evt.Use();
                    }
                    break;
                case EventType.MouseDrag:
                    if (GUIUtility.hotControl == trackControlID && _isDraggingScrubber)
                    {
                        int draggedFrame = Mathf.Clamp(Mathf.FloorToInt(evt.mousePosition.x / _pixelsPerFrame), 0, totalFrames - 1);
                        if (draggedFrame != _currentFrame)
                        {
                            SetCurrentFrame(draggedFrame);
                        }
                        evt.Use();
                    }
                    break;
                case EventType.MouseUp:
                    if (GUIUtility.hotControl == trackControlID)
                    {
                        GUIUtility.hotControl = 0;
                        _isDraggingScrubber = false;
                        evt.Use();
                    }
                    break;
            }

            GUI.EndScrollView();
        }

        #endregion

        #region Navigation & Frame Management

        public void SetCurrentFrame(int frame)
        {
            if (_data == null || _data.Frames <= 0) return;
            _currentFrame = Mathf.Clamp(frame, 0, _data.Frames - 1);
            ApplyCurrentFrameToPreview();
            SyncVideoPlayerToCurrentFrame();
            Repaint();
        }

        private void TogglePlay()
        {
            _isPlaying = !_isPlaying;
            _lastUpdateTime = EditorApplication.timeSinceStartup;
            _accumulatedTime = 0f;

            if (_isPlaying && _currentFrame >= _data.Frames - 1)
            {
                SetCurrentFrame(0);
            }
            Repaint();
        }

        private void JumpToPreviousModifiedFrame()
        {
            if (_data == null) return;
            for (int f = _currentFrame - 1; f >= 0; f--)
            {
                if (_data.IsFrameModified(f))
                {
                    SetCurrentFrame(f);
                    return;
                }
            }
        }

        private void JumpToNextModifiedFrame()
        {
            if (_data == null) return;
            for (int f = _currentFrame + 1; f < _data.Frames; f++)
            {
                if (_data.IsFrameModified(f))
                {
                    SetCurrentFrame(f);
                    return;
                }
            }
        }

        private void HandleKeyboardShortcuts()
        {
            Event evt = Event.current;
            if (evt.type == EventType.KeyDown)
            {
                // Ignore shortcuts if focusing a textfield
                if (GUIUtility.keyboardControl != 0 && EditorGUIUtility.editingTextField)
                    return;

                switch (evt.keyCode)
                {
                    case KeyCode.Space:
                        TogglePlay();
                        evt.Use();
                        break;
                    case KeyCode.LeftArrow:
                        SetCurrentFrame(_currentFrame - 1);
                        evt.Use();
                        break;
                    case KeyCode.RightArrow:
                        SetCurrentFrame(_currentFrame + 1);
                        evt.Use();
                        break;
                    case KeyCode.Home:
                        SetCurrentFrame(0);
                        evt.Use();
                        break;
                    case KeyCode.End:
                        if (_data != null) SetCurrentFrame(_data.Frames - 1);
                        evt.Use();
                        break;
                    case KeyCode.Z:
                        if (evt.control || evt.command)
                        {
                            if (evt.shift) _data.Redo();
                            else _data.Undo();
                            ApplyCurrentFrameToPreview();
                            evt.Use();
                            Repaint();
                        }
                        break;
                    case KeyCode.Y:
                        if (evt.control || evt.command)
                        {
                            _data.Redo();
                            ApplyCurrentFrameToPreview();
                            evt.Use();
                            Repaint();
                        }
                        break;
                    case KeyCode.C:
                        if (evt.control || evt.command)
                        {
                            _data.CopyFramePose(_currentFrame);
                            evt.Use();
                        }
                        break;
                    case KeyCode.V:
                        if (evt.control || evt.command)
                        {
                            _data.PasteFramePose(_currentFrame);
                            ApplyCurrentFrameToPreview();
                            evt.Use();
                            Repaint();
                        }
                        break;
                }
            }
        }

        #endregion

        #region Preview & Model Binding

        private void InitPreviewUtility()
        {
            if (_previewUtility == null)
            {
                _previewUtility = new PreviewRenderUtility();
                _previewUtility.cameraFieldOfView = 30f;
                _previewUtility.camera.nearClipPlane = 0.1f;
                _previewUtility.camera.farClipPlane = 100f;
                _previewUtility.camera.clearFlags = CameraClearFlags.SolidColor;
                _previewUtility.camera.backgroundColor = new Color(0.12f, 0.14f, 0.18f);
                _previewUtility.lights[0].intensity = 1.3f;
                _previewUtility.lights[1].intensity = 0.9f;
            }
        }

        private void CleanupPreviewUtility()
        {
            if (_previewUtility != null)
            {
                _previewUtility.Cleanup();
                _previewUtility = null;
            }
        }

        private void SetupPreviewInstance()
        {
            if (_previewUtility == null) InitPreviewUtility();

            if (_data == null) return;

            if (_data.TargetAvatar == null)
            {
                // Auto-detect avatar if not set
                var anims = FindObjectsOfType<Animator>();
                foreach (var a in anims)
                {
                    if (a.isHuman)
                    {
                        _data.TargetAvatar = a;
                        break;
                    }
                }
            }

            if (_data.TargetAvatar == null)
            {
                CleanupPreviewInstance();
                return;
            }

            CleanupPreviewInstance();

            _previewInstance = Instantiate(_data.TargetAvatar.gameObject, Vector3.zero, Quaternion.identity);
            _previewInstance.hideFlags = HideFlags.HideAndDontSave;

            var animators = _previewInstance.GetComponentsInChildren<Animator>();
            foreach (var a in animators)
            {
                a.enabled = false;
            }

            var colliders = _previewInstance.GetComponentsInChildren<Collider>();
            foreach (var c in colliders) DestroyImmediate(c);

            var rbs = _previewInstance.GetComponentsInChildren<Rigidbody>();
            foreach (var r in rbs) DestroyImmediate(r);

            // Disable all MonoBehaviour components on clone
            foreach (var comp in _previewInstance.GetComponentsInChildren<MonoBehaviour>())
            {
                comp.enabled = false;
            }

            // Cache SMPL-X to Avatar bone mappings using exact transform hierarchy paths
            _previewBoneMap.Clear();
            _previewInitialRotations.Clear();

            var origAnim = _data.TargetAvatar;
            if (origAnim != null && origAnim.isHuman)
            {
                Transform origHips = origAnim.GetBoneTransform(HumanBodyBones.Hips);
                if (origHips != null)
                {
                    string hipsPath = AnimationUtility.CalculateTransformPath(origHips, _data.TargetAvatar.transform);
                    _previewHipsTransform = _previewInstance.transform.Find(hipsPath);
                    if (_previewHipsTransform != null)
                    {
                        _previewInitialHipsPos = _previewHipsTransform.localPosition;
                        _previewPivot = new Vector3(0, _previewHipsTransform.position.y + 0.2f, 0);
                    }
                }

                foreach (var kvp in SmplxJointDefinitions.SmplxToHumanBodyBones)
                {
                    Transform origBone = origAnim.GetBoneTransform(kvp.Value);
                    if (origBone != null)
                    {
                        string bonePath = AnimationUtility.CalculateTransformPath(origBone, _data.TargetAvatar.transform);
                        Transform cloneBone = _previewInstance.transform.Find(bonePath);
                        if (cloneBone != null)
                        {
                            _previewBoneMap[kvp.Key] = cloneBone;
                            _previewInitialRotations[kvp.Key] = cloneBone.localRotation;
                        }
                    }
                }
            }

            // CRITICAL: Register the clone GameObject with PreviewRenderUtility!
            _previewUtility.AddSingleGO(_previewInstance);
            ApplyCurrentFrameToPreview();
        }

        private void CleanupPreviewInstance()
        {
            if (_previewInstance != null)
            {
                DestroyImmediate(_previewInstance);
                _previewInstance = null;
            }
            _previewBoneMap.Clear();
            _previewInitialRotations.Clear();
            _previewHipsTransform = null;
        }

        private void ApplyCurrentFrameToPreview()
        {
            if (_previewInstance == null || _data == null || _currentFrame < 0 || _currentFrame >= _data.Frames)
                return;

            // Apply Root Position
            if (_previewHipsTransform != null)
            {
                Vector3 curPos = _data.GetRootPosition(_currentFrame);
                Vector3 firstPos = _data.GetRootPosition(0);
                Vector3 delta = curPos - firstPos;

                if (_data.InPlace)
                {
                    _previewHipsTransform.localPosition = new Vector3(_previewInitialHipsPos.x, _previewInitialHipsPos.y + delta.y, _previewInitialHipsPos.z);
                }
                else
                {
                    _previewHipsTransform.localPosition = _previewInitialHipsPos + delta;
                }
            }

            // Apply 22 Joints
            foreach (var kvp in _previewBoneMap)
            {
                SmplxJoint joint = kvp.Key;
                Transform bone = kvp.Value;
                if (bone == null) continue;

                Quaternion smplRot = _data.GetJointRotation(_currentFrame, joint);
                Quaternion rest = _previewInitialRotations[joint];
                bone.localRotation = AnimationClipBuilder.ConvertSmplRotationToUnity(smplRot, joint, rest);
            }
        }

        #endregion

        #region Video Player Synchronization

        private void SetupVideoPlayerIfApplicable()
        {
            if (_data == null || _data.SourceVideoData == null) return;

            string videoPath = !string.IsNullOrEmpty(_data.SourceVideoData.OverlayVideoPath) && File.Exists(_data.SourceVideoData.OverlayVideoPath)
                ? _data.SourceVideoData.OverlayVideoPath
                : (!string.IsNullOrEmpty(_data.SourceVideoData.SourceVideoPath) && File.Exists(_data.SourceVideoData.SourceVideoPath) ? _data.SourceVideoData.SourceVideoPath : null);

            if (string.IsNullOrEmpty(videoPath)) return;

            CleanupVideoPlayer();

            _videoPlayerGo = new GameObject("TexMotion_Timeline_VideoPlayer")
            {
                hideFlags = HideFlags.HideAndDontSave
            };

            _videoPlayer = _videoPlayerGo.AddComponent<VideoPlayer>();
            _videoPlayer.playOnAwake = false;
            _videoPlayer.isLooping = false;
            _videoPlayer.renderMode = VideoRenderMode.RenderTexture;
            _videoPlayer.url = videoPath;
            _loadedVideoPath = videoPath;

            _videoTexture = new RenderTexture(640, 360, 0, RenderTextureFormat.ARGB32);
            _videoTexture.Create();
            _videoPlayer.targetTexture = _videoTexture;

            _videoPlayer.prepareCompleted += (vp) =>
            {
                SyncVideoPlayerToCurrentFrame();
                Repaint();
            };
            _videoPlayer.Prepare();
        }

        private void SyncVideoPlayerToCurrentFrame()
        {
            if (_videoPlayer == null || !_videoPlayer.isPrepared || _data == null) return;

            float time = _data.Timestamps != null && _data.Timestamps.Length > _currentFrame
                ? _data.Timestamps[_currentFrame]
                : (float)_currentFrame / _data.FrameRate;

            float duration = Mathf.Max(0.01f, (float)_videoPlayer.length);
            _videoPlayer.time = Mathf.Clamp(time, 0f, duration);
            _videoPlayer.Play();
            _videoPlayer.Pause();
            EditorApplication.QueuePlayerLoopUpdate();
        }

        private void CleanupVideoPlayer()
        {
            if (_videoPlayer != null)
            {
                _videoPlayer.Stop();
                _videoPlayer = null;
            }
            if (_videoPlayerGo != null)
            {
                DestroyImmediate(_videoPlayerGo);
                _videoPlayerGo = null;
            }
            if (_videoTexture != null)
            {
                _videoTexture.Release();
                DestroyImmediate(_videoTexture);
                _videoTexture = null;
            }
            _loadedVideoPath = "";
        }

        #endregion

        #region Save & Apply Operations

        private void SaveEditedMotionAsAsset()
        {
            if (_data == null || _data.TargetAvatar == null)
            {
                EditorUtility.DisplayDialog("Error", "Target Avatar is not configured. Please assign an avatar.", "OK");
                return;
            }

            string saveDir = "Assets/GeneratedMotions";
            if (!Directory.Exists(saveDir))
            {
                Directory.CreateDirectory(saveDir);
                AssetDatabase.Refresh();
            }

            string cleanName = string.IsNullOrEmpty(_data.ClipName) ? "TexMotion_Edited" : _data.ClipName;
            string assetPath = $"{saveDir}/{cleanName}_{DateTime.Now:yyyyMMdd_HHmmss}.anim";

            var buildOptions = AnimationBuildOptions.CreateDefault(
                cleanName,
                _isLoop,
                _data.InPlace,
                _data.HandPose,
                _data.FaceEmotion
            );
            buildOptions.TargetAvatar = _data.TargetAvatar;
            buildOptions.EmotionIntensity = _data.EmotionIntensity;

            try
            {
                var clip = _data.BuildAnimationClip(buildOptions);
                AssetDatabase.CreateAsset(clip, assetPath);
                AssetDatabase.SaveAssets();
                AssetDatabase.Refresh();

                EditorGUIUtility.PingObject(clip);
                Selection.activeObject = clip;

                EditorUtility.DisplayDialog("Motion Saved", $"Successfully exported edited AnimationClip to:\n{assetPath}", "OK");
            }
            catch (Exception ex)
            {
                Debug.LogError($"[TexMotion Timeline] Failed to save clip: {ex}");
                EditorUtility.DisplayDialog("Save Error", $"Failed to save clip:\n{ex.Message}", "OK");
            }
        }

        private void ApplyEditedMotionToAvatar()
        {
            if (_data == null || _data.TargetAvatar == null)
            {
                EditorUtility.DisplayDialog("Error", "Target Avatar is required to apply motion.", "OK");
                return;
            }

            string saveDir = "Assets/GeneratedMotions";
            if (!Directory.Exists(saveDir))
            {
                Directory.CreateDirectory(saveDir);
                AssetDatabase.Refresh();
            }

            string cleanName = string.IsNullOrEmpty(_data.ClipName) ? "TexMotion_Edited" : _data.ClipName;
            string assetPath = $"{saveDir}/{cleanName}_{DateTime.Now:yyyyMMdd_HHmmss}.anim";

            var buildOptions = AnimationBuildOptions.CreateDefault(
                cleanName,
                _isLoop,
                _data.InPlace,
                _data.HandPose,
                _data.FaceEmotion
            );
            buildOptions.TargetAvatar = _data.TargetAvatar;
            buildOptions.EmotionIntensity = _data.EmotionIntensity;

            try
            {
                var clip = _data.BuildAnimationClip(buildOptions);
                AssetDatabase.CreateAsset(clip, assetPath);
                AssetDatabase.SaveAssets();
                AssetDatabase.Refresh();

                // Setup Avatar Motion (Modular Avatar or Direct VRCSDK)
                var vrcConfig = new VrcMotionConfig
                {
                    MotionName = cleanName,
                    SetupMode = ModularAvatarSetup.IsModularAvatarInstalled() ? VrcSetupMode.ModularAvatar : VrcSetupMode.DirectVRCSDK,
                    MotionType = _isLoop ? VrcMotionType.ToggleLoopPose : VrcMotionType.OneShotEmote,
                    TargetLayer = VrcTargetLayer.ActionLayer,
                    InPlace = _data.InPlace
                };

                if (vrcConfig.SetupMode == VrcSetupMode.ModularAvatar)
                {
                    ModularAvatarSetup.SetupAvatarMotion(_data.TargetAvatar.gameObject, clip, vrcConfig, saveDir);
                }
                else
                {
                    VrcDirectSetup.SetupDirectAvatarMotion(_data.TargetAvatar.gameObject, clip, vrcConfig, null, saveDir);
                }

                EditorUtility.DisplayDialog("Applied to Avatar", $"Successfully built and applied '{cleanName}' to {_data.TargetAvatar.name}!", "OK");
            }
            catch (Exception ex)
            {
                Debug.LogError($"[TexMotion Timeline] Failed to apply to avatar: {ex}");
                EditorUtility.DisplayDialog("Apply Error", $"Failed to apply motion:\n{ex.Message}", "OK");
            }
        }

        #endregion

        #region Styles Initialization

        private void InitStyles()
        {
            if (_timelineBgStyle == null)
            {
                _timelineBgStyle = new GUIStyle(GUI.skin.box)
                {
                    normal = { background = MakeColorTexture(new Color(0.10f, 0.11f, 0.14f)) }
                };
            }

            if (_rulerTextStyle == null)
            {
                _rulerTextStyle = new GUIStyle(EditorStyles.miniLabel)
                {
                    fontSize = 9,
                    alignment = TextAnchor.UpperCenter,
                    normal = { textColor = new Color(0.65f, 0.68f, 0.75f) }
                };
            }

            if (_keyframeMarkerStyle == null)
            {
                _keyframeMarkerStyle = new GUIStyle(EditorStyles.boldLabel)
                {
                    fontSize = 12,
                    alignment = TextAnchor.MiddleCenter,
                    normal = { textColor = new Color(1.0f, 0.78f, 0.2f) }
                };
            }
        }

        private static Texture2D MakeColorTexture(Color col)
        {
            var tex = new Texture2D(1, 1);
            tex.SetPixel(0, 0, col);
            tex.Apply();
            return tex;
        }

        #endregion
    }
}
