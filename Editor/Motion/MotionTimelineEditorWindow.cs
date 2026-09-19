using System;
using System.Collections.Generic;
using System.IO;
using TexMotion.Editor;
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
        private readonly Dictionary<HumanBodyBones, Transform> _previewFingerBoneMap = new Dictionary<HumanBodyBones, Transform>();
        private readonly Dictionary<HumanBodyBones, Quaternion> _previewInitialFingerRotations = new Dictionary<HumanBodyBones, Quaternion>();
        private readonly Dictionary<int, float> _previewInitialFaceWeights = new Dictionary<int, float>();
        private SkinnedMeshRenderer _previewFaceRenderer;
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
        private double _lastVideoSeekTime = 0;

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
        private const float TOP_HEADER_HEIGHT = 84f;
        private const float PANEL_GAP = 8f;

        // Inspector foldouts
        private Vector2 _inspectorScrollPos;
        private bool _foldoutRoot = true;
        private bool _foldoutTorso = true;
        private bool _foldoutLeftArm = false;
        private bool _foldoutRightArm = false;
        private bool _foldoutLeftLeg = false;
        private bool _foldoutRightLeg = false;
        // Secondary pose operations are available on demand so the joint inspector stays compact.
        private bool _foldoutTools = false;

        // Advanced Pose Correction State
        private bool _foldoutAdvancedTools = true;
        private BodyPartMask _selectedBodyMask = BodyPartMask.All;
        private int _rangeStartFrame = 0;
        private int _rangeEndFrame = 10;
        private EasingType _selectedEasing = EasingType.EaseInOut;
        private int _loopBlendFrames = 10;
        private float _armpitLimitAngle = 20.0f;
        private float _groundPlaneY = 0.0f;
        private Vector3 _additiveRootOffset = Vector3.zero;
        private Vector3 _additiveArmEulerOffset = Vector3.zero;
        private bool _offsetFadeEdges = true;
        private int _retimeNewFrameCount = 15;

        // Visual Features
        private bool _enableOnionSkin = false;
        private bool _enableIkPins = false;
        private SmplxJoint? _activeIkJoint = null;
        private bool _isDraggingIk = false;
        private Vector2 _ikDragStartMouse;
        private Vector3 _ikDragStartPos;

        // Pose Palette & Glitch Detector
        private readonly PosePalette _posePalette = new PosePalette();
        private int _selectedPaletteSlot = 0;
        private float _paletteBlendWeight = 1.0f;
        private List<GlitchInfo> _detectedGlitches = new List<GlitchInfo>();

        // GUI Styles
        private GUIStyle _topBarStyle;
        private GUIStyle _panelStyle;
        private GUIStyle _panelHeaderStyle;
        private GUIStyle _cardStyle;
        private GUIStyle _sectionStyle;
        private GUIStyle _ghostButtonStyle;
        private GUIStyle _primaryButtonStyle;
        private GUIStyle _dangerButtonStyle;
        private GUIStyle _accentButtonStyle;
        private GUIStyle _emptyCardStyle;
        private GUIStyle _emptyTitleStyle;
        private GUIStyle _emptyDescriptionStyle;
        private GUIStyle _emptyGuideStyle;
        private GUIStyle _emptyGuideHeaderStyle;
        private GUIStyle _emptyGuideStepStyle;
        private GUIStyle _centeredHintStyle;
        private GUIStyle _badgeStyle;
        private GUIStyle _overlayBadgeStyle;
        private GUIStyle _hintOverlayStyle;
        private GUIStyle _videoBadgeStyle;
        private GUIStyle _warningBadgeStyle;
        private GUIStyle _frameStatusStyle;
        private GUIStyle _sectionLabelStyle;
        private GUIStyle _metaLabelStyle;
        private GUIStyle _modifiedStyle;
        private GUIStyle _cleanStyle;
        private GUIStyle _timelineBgStyle;
        private GUIStyle _keyframeMarkerStyle;
        private GUIStyle _rulerTextStyle;
        private bool _stylesInitialized;
        // Increment when the shared IMGUI theme changes. EditorWindow instances can
        // survive an assembly reload while their native GUIStyle textures do not.
        private const int TIMELINE_STYLE_VERSION = 2;
        private int _stylesVersion;

        // Unity menu command paths are static technical identifiers; the window title is localized below.
        [MenuItem("Tools/TexMotion/Motion Timeline Editor", false, 1)]
        public static MotionTimelineEditorWindow ShowEditor()
        {
            var window = GetWindow<MotionTimelineEditorWindow>(TexMotionLocalization.TrLiteral("Timeline Editor"));
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
            // Rebuild all non-serialized GUIStyle references after a domain/assembly
            // reload. MotionTimelineTheme releases generated textures during reload,
            // so retaining the old style would make buttons fall back to a grey bar.
            _stylesInitialized = false;
            _stylesVersion = 0;
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
                    bool looped = false;
                    if (nextFrame >= _data.Frames)
                    {
                        if (_isLoop)
                        {
                            nextFrame = 0;
                            looped = true;
                        }
                        else
                        {
                            nextFrame = _data.Frames - 1;
                            _isPlaying = false;
                            if (_videoPlayer != null && _videoPlayer.isPlaying)
                            {
                                _videoPlayer.Pause();
                            }
                        }
                    }

                    if (nextFrame != _currentFrame || looped)
                    {
                        _currentFrame = nextFrame;
                        ApplyCurrentFrameToPreview();
                        SyncVideoPlaybackDuringPlay(looped, now);
                        EditorApplication.QueuePlayerLoopUpdate();
                        Repaint();
                    }
                }
            }
            else
            {
                _lastUpdateTime = EditorApplication.timeSinceStartup;
                if (!_isPlaying && _videoPlayer != null && _videoPlayer.isPlaying)
                {
                    _videoPlayer.Pause();
                }
            }
        }

        private void OnGUI()
        {
            InitStyles();
            HandleKeyboardShortcuts();

            // IMGUI state is process-global and another editor window can leave a
            // disabled/content-tinted state behind. Normalize it before drawing so
            // labels never disappear while a surface still renders its background.
            GUI.enabled = true;
            GUI.color = Color.white;
            GUI.contentColor = Color.white;
            GUI.backgroundColor = Color.white;

            // Keep the editor surface anchored to the DESIGN.md midnight canvas.
            // Drawing this before any layout groups prevents Unity's current skin
            // from leaking a light grey background into empty gutters.
            if (Event.current.type == EventType.Repaint)
                EditorGUI.DrawRect(new Rect(0f, 0f, position.width, position.height), MotionTimelineTheme.Void);

            if (_data == null || _data.Frames <= 0)
            {
                DrawEmptyState();
                return;
            }

            GUILayout.Space(4f);
            DrawTopHeaderToolbar();
            GUILayout.Space(4f);

            // Main central split: Left Viewport (3D + optional video), Right Pose Inspector
            float inspectorWidth = Mathf.Clamp(position.width * 0.29f, 310f, 420f);
            float leftWidth = Mathf.Max(360f, position.width - inspectorWidth - PANEL_GAP - 8f);
            float centralHeight = Mathf.Max(190f, position.height - TOP_HEADER_HEIGHT - TIMELINE_HEIGHT - 14f);
            EditorGUILayout.BeginHorizontal(GUILayout.Height(centralHeight));

            // Left Workspace: 3D Viewport
            EditorGUILayout.BeginVertical(_panelStyle, GUILayout.Width(leftWidth), GUILayout.Height(centralHeight));
            DrawViewportWorkspace(leftWidth, centralHeight);
            EditorGUILayout.EndVertical();

            GUILayout.Space(PANEL_GAP);

            // Right Workspace: Pose & Joint Inspector
            EditorGUILayout.BeginVertical(_panelStyle, GUILayout.Width(inspectorWidth), GUILayout.Height(centralHeight));
            DrawPoseInspector(centralHeight);
            EditorGUILayout.EndVertical();

            EditorGUILayout.EndHorizontal();

            GUILayout.Space(4f);
            // Bottom Workspace: Full Timeline with Ruler and Scrubber
            DrawBottomTimelinePanel();
        }

        #region Empty State

        private void DrawEmptyState()
        {
            EditorGUILayout.BeginVertical();
            GUILayout.FlexibleSpace();

            EditorGUILayout.BeginHorizontal();
            GUILayout.FlexibleSpace();

            // Centered Card Container (Width: 540)
            EditorGUILayout.BeginVertical(_emptyCardStyle, GUILayout.Width(540));
            GUILayout.Space(24);

            // Title
            GUILayout.Label(TexMotionLocalization.Tr(TexMotionLocalization.TimelinePoseEditor), _emptyTitleStyle);

            GUILayout.Space(10);

            // Subtitle / Description
            GUILayout.Label(TexMotionLocalization.Tr(TexMotionLocalization.TimelineEmptyState), _emptyDescriptionStyle);

            GUILayout.Space(20);

            // Quick Guide Box
            EditorGUILayout.BeginVertical(_emptyGuideStyle);
            GUILayout.Space(10);

            GUILayout.Label(TexMotionLocalization.Tr(TexMotionLocalization.TimelineGettingStarted), _emptyGuideHeaderStyle);
            GUILayout.Space(6);

            EditorGUILayout.BeginHorizontal();
            GUILayout.Space(24);
            EditorGUILayout.BeginVertical();
            GUILayout.Label(TexMotionLocalization.Tr(TexMotionLocalization.TimelineStepOne), _emptyGuideStepStyle);
            GUILayout.Space(3);
            GUILayout.Label(TexMotionLocalization.Tr(TexMotionLocalization.TimelineStepTwo), _emptyGuideStepStyle);
            GUILayout.Space(3);
            GUILayout.Label(TexMotionLocalization.Tr(TexMotionLocalization.TimelineStepThree), _emptyGuideStepStyle);
            EditorGUILayout.EndVertical();
            GUILayout.Space(24);
            EditorGUILayout.EndHorizontal();

            GUILayout.Space(10);
            EditorGUILayout.EndVertical();

            GUILayout.Space(22);

            // Action Button: Open TexMotion Studio
            EditorGUILayout.BeginHorizontal();
            GUILayout.FlexibleSpace();

            // The empty state can be entered from another editor window that left
            // IMGUI disabled. Force this primary action into an enabled, untinted
            // state immediately before layout so it cannot collapse into Unity's
            // grey disabled-button fallback.
            GUI.enabled = true;
            GUI.color = Color.white;
            GUI.contentColor = Color.white;
            GUI.backgroundColor = Color.white;
            string openStudioLabel = TexMotionLocalization.Tr(TexMotionLocalization.OpenTexMotionStudio);
            if (GUILayout.Button(openStudioLabel, _primaryButtonStyle, GUILayout.Width(240), GUILayout.Height(38)))
            {
                TexMotionWindow.ShowWindow();
            }

            GUILayout.FlexibleSpace();
            EditorGUILayout.EndHorizontal();

            GUILayout.Space(24);
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
            EditorGUILayout.BeginVertical(_topBarStyle, GUILayout.Height(TOP_HEADER_HEIGHT));

            // Identity row: clip, metadata and the two document-level actions.
            EditorGUILayout.BeginHorizontal(GUILayout.Height(32f));
            EditorGUILayout.LabelField(TexMotionLocalization.Tr(TexMotionLocalization.Clip), _sectionLabelStyle, GUILayout.Width(58f));
            _data.ClipName = EditorGUILayout.TextField(_data.ClipName, GUILayout.MinWidth(120f), GUILayout.MaxWidth(260f));

            string info = TexMotionLocalization.TrFormat("Frames: {0} | {1:F2}s ({2:F0} FPS)", _data.Frames, _data.Duration, _data.FrameRate);
            EditorGUILayout.LabelField(info, _metaLabelStyle, GUILayout.Width(155f));

            EditorGUILayout.LabelField(TexMotionLocalization.Tr(TexMotionLocalization.Avatar), _sectionLabelStyle, GUILayout.Width(58f));
            var prevAvatar = _data.TargetAvatar;
            _data.TargetAvatar = (Animator)EditorGUILayout.ObjectField(_data.TargetAvatar, typeof(Animator), true, GUILayout.MinWidth(90f), GUILayout.MaxWidth(170f));
            if (prevAvatar != _data.TargetAvatar)
            {
                SetupPreviewInstance();
                ApplyCurrentFrameToPreview();
                Repaint();
            }

            GUILayout.FlexibleSpace();
            int modCount = _data.ModifiedFrameCount;
            EditorGUILayout.LabelField(
                modCount > 0
                    ? TexMotionLocalization.TrFormat(TexMotionLocalization.ModifiedCount, modCount)
                    : TexMotionLocalization.Tr(TexMotionLocalization.Clean),
                modCount > 0 ? _modifiedStyle : _cleanStyle,
                GUILayout.Width(modCount > 0 ? 82f : 68f));

            if (GUILayout.Button(TexMotionLocalization.Tr(TexMotionLocalization.SaveAnim), _ghostButtonStyle, GUILayout.Width(96f), GUILayout.Height(26f)))
                SaveEditedMotionAsAsset();

            var prevBg = GUI.backgroundColor;
            GUI.backgroundColor = new Color(0.2f, 0.85f, 0.45f);
            if (GUILayout.Button(TexMotionLocalization.Tr(TexMotionLocalization.ApplyToAvatar), GUILayout.Width(130f), GUILayout.Height(26f)))
                ApplyEditedMotionToAvatar();
            GUI.backgroundColor = prevBg;
            EditorGUILayout.EndHorizontal();

            GUILayout.Space(4f);

            // Playback row: the same compact controls are mirrored in the timeline,
            // but remain available at the top while the inspector is scrolled.
            EditorGUILayout.BeginHorizontal(GUILayout.Height(28f));
            EditorGUILayout.LabelField(TexMotionLocalization.Tr(TexMotionLocalization.Play), _sectionLabelStyle, GUILayout.Width(58f));
            if (GUILayout.Button(new GUIContent("|◀", TexMotionLocalization.Tr(TexMotionLocalization.FirstFrameHome)), _ghostButtonStyle, GUILayout.Width(30f), GUILayout.Height(26f)))
                SetCurrentFrame(0);
            if (GUILayout.Button(new GUIContent("◀", TexMotionLocalization.Tr(TexMotionLocalization.PreviousFrameLeft)), _ghostButtonStyle, GUILayout.Width(26f), GUILayout.Height(26f)))
                SetCurrentFrame(_currentFrame - 1);

            string headerPlayText = _isPlaying
                ? TexMotionLocalization.Tr(TexMotionLocalization.Pause)
                : TexMotionLocalization.Tr(TexMotionLocalization.Play);
            if (GUILayout.Button(new GUIContent(headerPlayText, TexMotionLocalization.Tr(TexMotionLocalization.TogglePlaybackSpace)), _accentButtonStyle, GUILayout.Width(76f), GUILayout.Height(26f)))
                TogglePlay();
            if (GUILayout.Button(new GUIContent("▶", TexMotionLocalization.Tr(TexMotionLocalization.NextFrameRight)), _ghostButtonStyle, GUILayout.Width(26f), GUILayout.Height(26f)))
                SetCurrentFrame(_currentFrame + 1);
            if (GUILayout.Button(new GUIContent("▶|", TexMotionLocalization.Tr(TexMotionLocalization.LastFrameEnd)), _ghostButtonStyle, GUILayout.Width(30f), GUILayout.Height(26f)))
                SetCurrentFrame(_data.Frames - 1);

            _isLoop = GUILayout.Toggle(_isLoop, new GUIContent(TexMotionLocalization.Tr(TexMotionLocalization.LoopPlayback)), _ghostButtonStyle, GUILayout.Width(74f), GUILayout.Height(26f));
            EditorGUILayout.LabelField(TexMotionLocalization.Tr(TexMotionLocalization.Speed), _metaLabelStyle, GUILayout.Width(42f));
            if (GUILayout.Button(new GUIContent("−", TexMotionLocalization.Tr(TexMotionLocalization.DecreaseSpeed)), _ghostButtonStyle, GUILayout.Width(24f), GUILayout.Height(26f)))
                _playbackSpeed = Mathf.Max(0.1f, Mathf.Round((_playbackSpeed - 0.1f) * 10f) / 10f);
            if (GUILayout.Button(new GUIContent($"{_playbackSpeed:F1}x", TexMotionLocalization.Tr(TexMotionLocalization.ResetSpeed)), _ghostButtonStyle, GUILayout.Width(46f), GUILayout.Height(26f)))
                _playbackSpeed = 1.0f;
            if (GUILayout.Button(new GUIContent("+", TexMotionLocalization.Tr(TexMotionLocalization.IncreaseSpeed)), _ghostButtonStyle, GUILayout.Width(24f), GUILayout.Height(26f)))
                _playbackSpeed = Mathf.Min(3.0f, Mathf.Round((_playbackSpeed + 0.1f) * 10f) / 10f);
            GUILayout.FlexibleSpace();

            GUI.enabled = _data.CanUndo;
            if (GUILayout.Button(TexMotionLocalization.Tr(TexMotionLocalization.Undo), _ghostButtonStyle, GUILayout.Width(64f), GUILayout.Height(26f)))
            {
                _data.Undo();
                ApplyCurrentFrameToPreview();
                Repaint();
            }
            GUI.enabled = _data.CanRedo;
            if (GUILayout.Button(TexMotionLocalization.Tr(TexMotionLocalization.Redo), _ghostButtonStyle, GUILayout.Width(64f), GUILayout.Height(26f)))
            {
                _data.Redo();
                ApplyCurrentFrameToPreview();
                Repaint();
            }
            GUI.enabled = true;

            if (modCount > 0 && GUILayout.Button(TexMotionLocalization.Tr(TexMotionLocalization.RevertAll), _dangerButtonStyle, GUILayout.Width(94f), GUILayout.Height(26f)))
            {
                if (EditorUtility.DisplayDialog(
                    TexMotionLocalization.Tr(TexMotionLocalization.RevertAllTitle),
                    TexMotionLocalization.Tr(TexMotionLocalization.RevertAllMessage),
                    TexMotionLocalization.Tr(TexMotionLocalization.Revert),
                    TexMotionLocalization.Tr(TexMotionLocalization.Cancel)))
                {
                    _data.ResetAllFramesToOriginal();
                    ApplyCurrentFrameToPreview();
                    Repaint();
                }
            }
            EditorGUILayout.EndHorizontal();
            EditorGUILayout.EndVertical();
        }

        #endregion

        #region Left Viewport Workspace

        private void DrawViewportWorkspace(float width, float height)
        {
            bool hasVideo = _data.SourceVideoData != null && (!string.IsNullOrEmpty(_data.SourceVideoData.OverlayVideoPath) || !string.IsNullOrEmpty(_data.SourceVideoData.SourceVideoPath));

            EditorGUILayout.BeginVertical(_cardStyle);

            if (hasVideo)
            {
                // Layout Toolbar for Video + Avatar
                EditorGUILayout.BeginHorizontal(_panelHeaderStyle, GUILayout.Height(30f));
                EditorGUILayout.LabelField(TexMotionLocalization.Tr(TexMotionLocalization.ViewportVideo), _sectionLabelStyle);
                GUILayout.FlexibleSpace();
                _viewportLayout = (ViewportLayout)GUILayout.Toolbar((int)_viewportLayout,
                    new string[]
                    {
                        TexMotionLocalization.TrLiteral("Side-by-Side"),
                        TexMotionLocalization.TrLiteral("Avatar Only"),
                        TexMotionLocalization.TrLiteral("Video Only")
                    },
                    _ghostButtonStyle, GUILayout.Width(250), GUILayout.Height(24f));
                EditorGUILayout.EndHorizontal();
            }
            else
            {
                EditorGUILayout.BeginHorizontal(_panelHeaderStyle, GUILayout.Height(30f));
                EditorGUILayout.LabelField(TexMotionLocalization.Tr(TexMotionLocalization.ViewportVideo), _sectionLabelStyle);
                GUILayout.FlexibleSpace();
                EditorGUILayout.EndHorizontal();
            }

            GUILayout.Space(4f);

            Rect viewportRect = GUILayoutUtility.GetRect(
                Mathf.Max(40f, width - 24f),
                Mathf.Max(40f, height - 42f),
                GUILayout.ExpandWidth(true),
                GUILayout.ExpandHeight(true));

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

            EditorGUILayout.EndVertical();
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

                    if (_enableOnionSkin)
                    {
                        DrawOnionSkinOverlay(rect, _previewUtility.camera);
                    }

                    if (_enableIkPins)
                    {
                        DrawIkPinsOverlay(rect, _previewUtility.camera);
                    }
                }
                else
                {
                    EditorGUI.DrawRect(rect, MotionTimelineTheme.Void);
                    float stateWidth = Mathf.Min(360f, Mathf.Max(220f, rect.width - 32f));
                    Rect stateRect = new Rect(
                        rect.center.x - stateWidth * 0.5f,
                        rect.center.y - 28f,
                        stateWidth,
                        56f);
                    EditorGUI.DrawRect(stateRect, MotionTimelineTheme.Obsidian);
                    EditorGUI.DrawRect(new Rect(stateRect.x, stateRect.y, stateRect.width, 1f), MotionTimelineTheme.Graphite);
                    EditorGUI.DrawRect(new Rect(stateRect.x, stateRect.yMax - 1f, stateRect.width, 1f), MotionTimelineTheme.Graphite);
                    GUI.Label(stateRect, TexMotionLocalization.Tr(TexMotionLocalization.MissingAvatar), _centeredHintStyle);
                }
            }

            // Overlay Viewport HUD (Toggles, Camera Reset, Bone badge, hint)
            DrawViewportHUD(rect);
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

        private void DrawOnionSkinOverlay(Rect rect, Camera cam)
        {
            if (_data == null || cam == null) return;

            Handles.BeginGUI();

            int jointCount = SmplxJointDefinitions.JointCount;
            Quaternion[] locals = new Quaternion[jointCount];

            // 1. Draw Previous Frame (Cyan)
            if (_currentFrame > 0)
            {
                int prevFrame = _currentFrame - 1;
                Vector3 prevRoot = _data.GetRootPosition(prevFrame);
                for (int j = 0; j < jointCount; j++) locals[j] = _data.GetJointRotation(prevFrame, (SmplxJoint)j);
                Vector3[] prevFk = MotionIkUtility.ComputeForwardKinematics(prevRoot, locals);

                Color prevColor = new Color(0.0f, 0.85f, 1.0f, 0.45f);
                DrawGhostBones(rect, cam, prevFk, prevColor, 2.0f);
            }

            // 2. Draw Next Frame (Magenta)
            if (_currentFrame < _data.Frames - 1)
            {
                int nextFrame = _currentFrame + 1;
                Vector3 nextRoot = _data.GetRootPosition(nextFrame);
                for (int j = 0; j < jointCount; j++) locals[j] = _data.GetJointRotation(nextFrame, (SmplxJoint)j);
                Vector3[] nextFk = MotionIkUtility.ComputeForwardKinematics(nextRoot, locals);

                Color nextColor = new Color(1.0f, 0.2f, 0.75f, 0.45f);
                DrawGhostBones(rect, cam, nextFk, nextColor, 2.0f);
            }

            Handles.EndGUI();
        }

        private void DrawGhostBones(Rect rect, Camera cam, Vector3[] fkPositions, Color color, float lineWidth)
        {
            Handles.color = color;
            foreach (var (parentJoint, childJoint) in BoneConnections)
            {
                Vector3 p1World = fkPositions[(int)parentJoint];
                Vector3 p2World = fkPositions[(int)childJoint];
                if (TryWorldToScreen(cam, rect, p1World, out Vector2 p1) &&
                    TryWorldToScreen(cam, rect, p2World, out Vector2 p2))
                {
                    Handles.DrawAAPolyLine(lineWidth, new Vector3(p1.x, p1.y, 0), new Vector3(p2.x, p2.y, 0));
                }
            }

            // Ghost key joint discs
            SmplxJoint[] markerJoints = new[] { SmplxJoint.L_Wrist, SmplxJoint.R_Wrist, SmplxJoint.L_Ankle, SmplxJoint.R_Ankle, SmplxJoint.Head };
            foreach (var joint in markerJoints)
            {
                Vector3 pos = fkPositions[(int)joint];
                if (TryWorldToScreen(cam, rect, pos, out Vector2 pt))
                {
                    Handles.DrawSolidDisc(new Vector3(pt.x, pt.y, 0), Vector3.forward, 3.5f);
                }
            }
        }

        private static readonly SmplxJoint[] IkPinJoints = new[]
        {
            SmplxJoint.L_Wrist,
            SmplxJoint.R_Wrist,
            SmplxJoint.L_Ankle,
            SmplxJoint.R_Ankle
        };

        private void DrawIkPinsOverlay(Rect rect, Camera cam)
        {
            if (_data == null || cam == null || _previewInstance == null) return;

            Handles.BeginGUI();
            Event evt = Event.current;

            foreach (var joint in IkPinJoints)
            {
                if (_previewBoneMap.TryGetValue(joint, out Transform bone) && bone != null)
                {
                    if (TryWorldToScreen(cam, rect, bone.position, out Vector2 screenPt))
                    {
                        Rect pinRect = new Rect(screenPt.x - 10, screenPt.y - 10, 20, 20);
                        bool isHovered = pinRect.Contains(evt.mousePosition);
                        bool isActive = (_activeIkJoint == joint);

                        Color pinCol = isActive
                            ? MotionTimelineTheme.AcidLime
                            : (isHovered ? MotionTimelineTheme.SignalTeal : new Color(0.2f, 0.85f, 0.45f, 0.95f));

                        // Outer ring
                        Handles.color = new Color(0.05f, 0.08f, 0.12f, 0.95f);
                        Handles.DrawSolidDisc(new Vector3(screenPt.x, screenPt.y, 0), Vector3.forward, 9f);

                        // Inner icon disc
                        Handles.color = pinCol;
                        Handles.DrawSolidDisc(new Vector3(screenPt.x, screenPt.y, 0), Vector3.forward, 7.5f);
                        Handles.color = Color.white;
                        Handles.DrawWireDisc(new Vector3(screenPt.x, screenPt.y, 0), Vector3.forward, 3.5f);

                        // Start Drag
                        if (evt.type == EventType.MouseDown && evt.button == 0 && pinRect.Contains(evt.mousePosition))
                        {
                            _activeIkJoint = joint;
                            _isDraggingIk = true;
                            _ikDragStartMouse = evt.mousePosition;
                            _ikDragStartPos = bone.position;
                            evt.Use();
                            GUIUtility.hotControl = GUIUtility.GetControlID(FocusType.Passive);
                        }
                    }
                }
            }

            // Process drag
            if (_isDraggingIk && _activeIkJoint.HasValue)
            {
                if (evt.type == EventType.MouseDrag)
                {
                    Vector2 deltaMouse = evt.mousePosition - _ikDragStartMouse;
                    Vector3 camRight = cam.transform.right;
                    Vector3 camUp = cam.transform.up;
                    float factor = 0.003f * _previewDistance;
                    Vector3 worldDelta = (camRight * deltaMouse.x * factor) - (camUp * deltaMouse.y * factor);

                    Vector3 targetPos = _ikDragStartPos + worldDelta;
                    Vector3 polePos = targetPos + cam.transform.forward * 0.25f;

                    _data.ApplyTwoBoneIK(_currentFrame, _activeIkJoint.Value, targetPos, polePos);
                    ApplyCurrentFrameToPreview();
                    evt.Use();
                    Repaint();
                }
                else if (evt.type == EventType.MouseUp)
                {
                    _isDraggingIk = false;
                    _activeIkJoint = null;
                    GUIUtility.hotControl = 0;
                    evt.Use();
                    Repaint();
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
                // Don't intercept clicks inside HUD toolbar areas
                if (mousePos.y < rect.y + 36f || mousePos.y > rect.yMax - 26f)
                {
                    return false;
                }

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
                        _data.RecordUndo(
                            TexMotionLocalization.TrFormat(
                                "Rotate {0} Frame {1}",
                                GetJointFriendlyName(_selectedJoint.Value),
                                _currentFrame));
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
            // 1. Top-Left Toolbar: Skeleton, Gizmos, Onion Skin, IK Pins & Playback
            Rect skelRect = new Rect(rect.x + 8, rect.y + 8, 80, 22);
            Rect gizmoRect = new Rect(rect.x + 90, rect.y + 8, 70, 22);
            Rect onionRect = new Rect(rect.x + 162, rect.y + 8, 66, 22);
            Rect ikRect = new Rect(rect.x + 230, rect.y + 8, 68, 22);

            bool newShowSkel = GUI.Toggle(skelRect, _showSkeleton, TexMotionLocalization.TrLiteral("🦴 Skeleton"), _showSkeleton ? _accentButtonStyle : _ghostButtonStyle);
            if (newShowSkel != _showSkeleton)
            {
                _showSkeleton = newShowSkel;
                Repaint();
            }

            bool newShowGizmos = GUI.Toggle(gizmoRect, _showGizmos, TexMotionLocalization.TrLiteral("🎯 Gizmos"), _showGizmos ? _accentButtonStyle : _ghostButtonStyle);
            if (newShowGizmos != _showGizmos)
            {
                _showGizmos = newShowGizmos;
                Repaint();
            }

            bool newShowOnion = GUI.Toggle(onionRect, _enableOnionSkin, TexMotionLocalization.TrLiteral("🧅 Onion"), _enableOnionSkin ? _accentButtonStyle : _ghostButtonStyle);
            if (newShowOnion != _enableOnionSkin)
            {
                _enableOnionSkin = newShowOnion;
                Repaint();
            }

            bool newShowIk = GUI.Toggle(ikRect, _enableIkPins, TexMotionLocalization.TrLiteral("🦾 IK Pins"), _enableIkPins ? _accentButtonStyle : _ghostButtonStyle);
            if (newShowIk != _enableIkPins)
            {
                _enableIkPins = newShowIk;
                Repaint();
            }

            // Step Back
            Rect hudPrevRect = new Rect(rect.x + 302, rect.y + 8, 24, 22);
            if (GUI.Button(hudPrevRect, new GUIContent("◀", TexMotionLocalization.Tr(TexMotionLocalization.PreviousFrameLeft)), _ghostButtonStyle))
            {
                SetCurrentFrame(_currentFrame - 1);
            }

            // Play / Pause Toggle in HUD
            Rect hudPlayRect = new Rect(rect.x + 326, rect.y + 8, 68, 22);
            string hudPlayIcon = _isPlaying
                ? TexMotionLocalization.Tr(TexMotionLocalization.Pause)
                : TexMotionLocalization.Tr(TexMotionLocalization.Play);
            if (GUI.Button(hudPlayRect, new GUIContent(hudPlayIcon, TexMotionLocalization.Tr(TexMotionLocalization.TogglePlaybackSpace)), _accentButtonStyle))
            {
                TogglePlay();
            }
            // Step Forward
            Rect hudNextRect = new Rect(rect.x + 394, rect.y + 8, 24, 22);
            if (GUI.Button(hudNextRect, new GUIContent("▶", TexMotionLocalization.Tr(TexMotionLocalization.NextFrameRight)), _ghostButtonStyle))
            {
                SetCurrentFrame(_currentFrame + 1);
            }

            // 2. Top-Right: Camera Reset, Active Bone Badge, Clear, Reset
            float curX = rect.xMax - 8;

            // 🔄 Reset Cam
            curX -= 88;
            Rect resetCamRect = new Rect(curX, rect.y + 8, 88, 22);
            if (GUI.Button(resetCamRect, TexMotionLocalization.TrLiteral("🔄 Reset Cam"), _ghostButtonStyle))
            {
                _previewDir = new Vector2(180f, 10f);
                _previewDistance = 2.8f;
                _previewPivot = _previewHipsTransform != null
                    ? new Vector3(0, _previewHipsTransform.position.y + 0.2f, 0)
                    : new Vector3(0, 1.0f, 0);
                Repaint();
            }

            if (_selectedJoint.HasValue)
            {
                curX -= 6; // gap

                // Reset Bone Rotation button
                curX -= 46;
                Rect resetJointRect = new Rect(curX, rect.y + 8, 46, 22);
                if (GUI.Button(resetJointRect, TexMotionLocalization.TrLiteral("Reset"), _ghostButtonStyle))
                {
                    _data.ResetJointToOriginal(_currentFrame, _selectedJoint.Value);
                    ApplyCurrentFrameToPreview();
                    Repaint();
                }

                curX -= 3; // gap

                // Clear selection button
                curX -= 44;
                Rect clearRect = new Rect(curX, rect.y + 8, 44, 22);
                if (GUI.Button(clearRect, TexMotionLocalization.TrLiteral("Clear"), _ghostButtonStyle))
                {
                    _selectedJoint = null;
                    Repaint();
                }

                curX -= 6; // gap

                // Active Bone Badge
                string friendlyName = GetJointFriendlyName(_selectedJoint.Value);
                string badgeText = $"🎯 {friendlyName}";
                _overlayBadgeStyle.normal.textColor = MotionTimelineTheme.AcidLime;
                _overlayBadgeStyle.alignment = TextAnchor.MiddleCenter;
                var badgeStyle = _overlayBadgeStyle;
                Vector2 textSize = badgeStyle.CalcSize(new GUIContent(badgeText));
                float badgeWidth = Mathf.Max(textSize.x + 14f, 115f);
                curX -= badgeWidth;

                Rect badgeRect = new Rect(curX, rect.y + 8, badgeWidth, 22);
                EditorGUI.DrawRect(badgeRect, MotionTimelineTheme.WithAlpha(MotionTimelineTheme.Obsidian, 0.94f));
                GUI.Label(badgeRect, badgeText, badgeStyle);
            }

            // 3. Bottom-Left Hint overlay
            Rect hintRect = new Rect(rect.x + 8, rect.y + rect.height - 24, 310, 18);
            EditorGUI.DrawRect(hintRect, MotionTimelineTheme.WithAlpha(MotionTimelineTheme.Carbon, 0.94f));
            _hintOverlayStyle.alignment = TextAnchor.MiddleLeft;
            _hintOverlayStyle.padding = new RectOffset(6, 0, 1, 0);
            var hintStyle = _hintOverlayStyle;
            string hint = _selectedJoint.HasValue
                ? TexMotionLocalization.TrFormat("Drag Ring to Rotate ({0})", GetJointFriendlyName(_selectedJoint.Value))
                : TexMotionLocalization.TrLiteral("Click Bone to Select | Drag: Orbit | Scroll: Zoom");
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
                case SmplxJoint.Pelvis: return TexMotionLocalization.TrLiteral("Hips / Root");
                case SmplxJoint.L_Hip: return TexMotionLocalization.TrLiteral("Left Hip");
                case SmplxJoint.R_Hip: return TexMotionLocalization.TrLiteral("Right Hip");
                case SmplxJoint.Spine1: return TexMotionLocalization.TrLiteral("Spine (Lower)");
                case SmplxJoint.L_Knee: return TexMotionLocalization.TrLiteral("Left Knee");
                case SmplxJoint.R_Knee: return TexMotionLocalization.TrLiteral("Right Knee");
                case SmplxJoint.Spine2: return TexMotionLocalization.TrLiteral("Chest (Middle)");
                case SmplxJoint.L_Ankle: return TexMotionLocalization.TrLiteral("Left Ankle");
                case SmplxJoint.R_Ankle: return TexMotionLocalization.TrLiteral("Right Ankle");
                case SmplxJoint.Spine3: return TexMotionLocalization.TrLiteral("Upper Chest");
                case SmplxJoint.L_Foot: return TexMotionLocalization.TrLiteral("Left Toes");
                case SmplxJoint.R_Foot: return TexMotionLocalization.TrLiteral("Right Toes");
                case SmplxJoint.Neck: return TexMotionLocalization.TrLiteral("Neck");
                case SmplxJoint.L_Collar: return TexMotionLocalization.TrLiteral("Left Collar (Shoulder)");
                case SmplxJoint.R_Collar: return TexMotionLocalization.TrLiteral("Right Collar (Shoulder)");
                case SmplxJoint.Head: return TexMotionLocalization.TrLiteral("Head");
                case SmplxJoint.L_Shoulder: return TexMotionLocalization.TrLiteral("Left Upper Arm");
                case SmplxJoint.R_Shoulder: return TexMotionLocalization.TrLiteral("Right Upper Arm");
                case SmplxJoint.L_Elbow: return TexMotionLocalization.TrLiteral("Left Elbow");
                case SmplxJoint.R_Elbow: return TexMotionLocalization.TrLiteral("Right Elbow");
                case SmplxJoint.L_Wrist: return TexMotionLocalization.TrLiteral("Left Wrist (Hand)");
                case SmplxJoint.R_Wrist: return TexMotionLocalization.TrLiteral("Right Wrist (Hand)");
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
            EditorGUI.DrawRect(rect, MotionTimelineTheme.Void);

            if (_videoTexture != null && _videoPlayer != null && _videoPlayer.isPrepared)
            {
                GUI.DrawTexture(rect, _videoTexture, ScaleMode.ScaleToFit);
            }
            else
            {
                float stateWidth = Mathf.Min(320f, Mathf.Max(180f, rect.width - 32f));
                Rect stateRect = new Rect(
                    rect.center.x - stateWidth * 0.5f,
                    rect.center.y - 24f,
                    stateWidth,
                    48f);
                EditorGUI.DrawRect(stateRect, MotionTimelineTheme.Obsidian);
                EditorGUI.DrawRect(new Rect(stateRect.x, stateRect.y, stateRect.width, 1f), MotionTimelineTheme.Graphite);
                EditorGUI.DrawRect(new Rect(stateRect.x, stateRect.yMax - 1f, stateRect.width, 1f), MotionTimelineTheme.Graphite);
                EditorGUI.LabelField(stateRect, TexMotionLocalization.Tr(TexMotionLocalization.LoadingVideo), _centeredHintStyle);
            }

            DrawVideoBackendBadge(rect);
        }

        private void DrawVideoBackendBadge(Rect rect)
        {
            if (Event.current.type != EventType.Repaint || _data == null || _data.SourceVideoData == null)
                return;
            if (rect.width < 120f || rect.height < 36f)
                return;

            VideoMotionData videoData = _data.SourceVideoData;
            bool hasOverlay = !string.IsNullOrEmpty(videoData.OverlayVideoPath) && File.Exists(videoData.OverlayVideoPath);
            string detectorLabel = GetVideoDetectorDisplayName(videoData);
            string badgeText = hasOverlay
                ? TexMotionLocalization.TrFormat("{0} {1}", detectorLabel, TexMotionLocalization.TrLiteral("Overlay"))
                : detectorLabel;
            string overlaySource = string.IsNullOrEmpty(videoData.OverlaySource)
                ? (hasOverlay ? videoData.DetectorName : TexMotionLocalization.TrLiteral("source video"))
                : videoData.OverlaySource;

            _videoBadgeStyle.fontStyle = FontStyle.Bold;
            _videoBadgeStyle.normal.textColor = GetVideoDetectorColor(videoData.DetectorName);
            var badgeStyle = _videoBadgeStyle;
            float badgeWidth = Mathf.Min(Mathf.Max(185f, badgeStyle.CalcSize(new GUIContent(badgeText)).x + 8f), rect.width - 16f);
            Rect badgeRect = new Rect(rect.x + 8f, rect.y + 8f, badgeWidth, 20f);
            EditorGUI.DrawRect(badgeRect, MotionTimelineTheme.WithAlpha(MotionTimelineTheme.Obsidian, 0.94f));
            GUI.Label(
                badgeRect,
                new GUIContent(badgeText, TexMotionLocalization.TrFormat("Overlay source: {0}", overlaySource)),
                badgeStyle);

            if (videoData.UsedBackendFallback && rect.height > 62f)
            {
                string requested = string.IsNullOrEmpty(videoData.BackendRequested)
                    ? TexMotionLocalization.TrLiteral("selected backend")
                    : videoData.BackendRequested.ToUpperInvariant();
                string actual = string.IsNullOrEmpty(videoData.BackendActual)
                    ? videoData.DetectorName
                    : videoData.BackendActual;
                string fallbackText = TexMotionLocalization.TrFormat(
                    "⚠ {0} → {1} {2}",
                    requested,
                    (actual ?? TexMotionLocalization.TrLiteral("unknown")).ToUpperInvariant(),
                    TexMotionLocalization.TrLiteral("fallback"));
                _warningBadgeStyle.fontStyle = FontStyle.Bold;
                _warningBadgeStyle.normal.textColor = MotionTimelineTheme.CoralRed;
                var warningStyle = _warningBadgeStyle;
                Rect warningRect = new Rect(rect.x + 8f, rect.y + 30f, badgeWidth, 18f);
                EditorGUI.DrawRect(warningRect, MotionTimelineTheme.WithAlpha(MotionTimelineTheme.CoralRed, 0.15f));
                GUI.Label(
                    warningRect,
                    new GUIContent(
                        fallbackText,
                        videoData.BackendFallbackReason ?? TexMotionLocalization.TrLiteral("Backend fallback")),
                    warningStyle);
            }

            // Keep per-frame quality fallbacks visible even when the requested
            // backend initialized successfully.  This is different from the
            // global BackendFallback flag: WHAM may be available while a
            // subset of frames fails the silhouette-alignment safety gate.
            int qualityFallbackCount = videoData.BackendMetadata != null
                ? videoData.BackendMetadata.QualityFrameFallbackCount
                : 0;
            if (qualityFallbackCount > 0 && rect.height > 62f)
            {
                string qualityReason = string.IsNullOrWhiteSpace(videoData.BackendMetadata.QualityFallbackReason)
                    ? TexMotionLocalization.TrLiteral("Quality pose failed auxiliary silhouette alignment.")
                    : videoData.BackendMetadata.QualityFallbackReason;
                _warningBadgeStyle.fontStyle = FontStyle.Bold;
                _warningBadgeStyle.normal.textColor = MotionTimelineTheme.CoralRed;
                var qualityWarningStyle = _warningBadgeStyle;
                float warningY = rect.y + (videoData.UsedBackendFallback ? 50f : 30f);
                Rect qualityWarningRect = new Rect(rect.x + 8f, warningY, badgeWidth, 18f);
                EditorGUI.DrawRect(qualityWarningRect, MotionTimelineTheme.WithAlpha(MotionTimelineTheme.CoralRed, 0.15f));
                GUI.Label(
                    qualityWarningRect,
                    new GUIContent(
                        TexMotionLocalization.TrFormat(
                            "⚠ WHAM geometry → MediaPipe ({0} frame(s))",
                            qualityFallbackCount),
                        qualityReason),
                    qualityWarningStyle);
            }
        }

        private static string GetVideoDetectorDisplayName(string detectorName)
        {
            if (string.Equals(detectorName, "rtmpose_hybrid", StringComparison.OrdinalIgnoreCase))
                return TexMotionLocalization.TrLiteral("● RTMPose Hybrid (2D + 3D Auxiliary)");
            if (string.Equals(detectorName, "rtmpose", StringComparison.OrdinalIgnoreCase))
                return TexMotionLocalization.TrLiteral("● RTMPose 2D");
            if (string.Equals(detectorName, "wham", StringComparison.OrdinalIgnoreCase))
                return TexMotionLocalization.TrLiteral("● WHAM Temporal 3D");
            if (string.Equals(detectorName, "hmr2", StringComparison.OrdinalIgnoreCase))
                return TexMotionLocalization.TrLiteral("● HMR2 / 4D-Humans 3D");
            if (string.Equals(detectorName, "hybrik", StringComparison.OrdinalIgnoreCase))
                return TexMotionLocalization.TrLiteral("● HybrIK 3D + IK");
            if (string.Equals(detectorName, "pytorch", StringComparison.OrdinalIgnoreCase) ||
                string.Equals(detectorName, "pytorch_quality", StringComparison.OrdinalIgnoreCase))
                return TexMotionLocalization.TrLiteral("● PyTorch Temporal 3D");
            return TexMotionLocalization.TrLiteral("● MediaPipe Pose");
        }

        private static string GetVideoDetectorDisplayName(VideoMotionData motionData)
        {
            if (motionData != null && motionData.IsActualWhamMediaPipeFusion)
                return TexMotionLocalization.TrLiteral("● WHAM + MediaPipe Fusion");
            return GetVideoDetectorDisplayName(motionData != null ? motionData.DetectorName : null);
        }

        private static Color GetVideoDetectorColor(string detectorName)
        {
            if (string.Equals(detectorName, "wham", StringComparison.OrdinalIgnoreCase) ||
                string.Equals(detectorName, "hmr2", StringComparison.OrdinalIgnoreCase) ||
                string.Equals(detectorName, "hybrik", StringComparison.OrdinalIgnoreCase) ||
                string.Equals(detectorName, "pytorch", StringComparison.OrdinalIgnoreCase) ||
                string.Equals(detectorName, "pytorch_quality", StringComparison.OrdinalIgnoreCase))
                return MotionTimelineTheme.SignalTeal;
            if (string.Equals(detectorName, "rtmpose_hybrid", StringComparison.OrdinalIgnoreCase))
                return MotionTimelineTheme.PulseGreen;
            return MotionTimelineTheme.SignalTeal;
        }

        #endregion

        #region Right Pose Inspector

        private void DrawPoseInspector(float height)
        {
            _inspectorScrollPos = EditorGUILayout.BeginScrollView(_inspectorScrollPos);

            EditorGUILayout.BeginHorizontal(_panelHeaderStyle, GUILayout.Height(30f));
            EditorGUILayout.LabelField(TexMotionLocalization.Tr(TexMotionLocalization.Frame), _sectionLabelStyle);
            GUILayout.FlexibleSpace();
            EditorGUILayout.LabelField(
                TexMotionLocalization.Tr(TexMotionLocalization.TimelineEditor),
                _metaLabelStyle,
                GUILayout.Width(100f));
            EditorGUILayout.EndHorizontal();
            GUILayout.Space(4f);

            // Frame Header & Navigation
            EditorGUILayout.BeginVertical(_cardStyle);
            EditorGUILayout.BeginHorizontal();

            EditorGUILayout.LabelField(
                TexMotionLocalization.TrFormat("📍 Frame {0} / {1}", _currentFrame + 1, _data.Frames),
                _sectionLabelStyle);
            float curTime = _data.Timestamps != null && _data.Timestamps.Length > _currentFrame
                ? _data.Timestamps[_currentFrame]
                : (float)_currentFrame / _data.FrameRate;
            EditorGUILayout.LabelField(
                TexMotionLocalization.TrFormat("Time: {0:F2}s", curTime),
                _metaLabelStyle,
                GUILayout.Width(75));

            bool isMod = _data.IsFrameModified(_currentFrame);
            _frameStatusStyle.normal.textColor = isMod ? MotionTimelineTheme.AcidLime : MotionTimelineTheme.PulseGreen;
            string badgeText = isMod
                ? TexMotionLocalization.TrLiteral("● Modified")
                : TexMotionLocalization.TrLiteral("✓ Original");
            EditorGUILayout.LabelField(badgeText, _frameStatusStyle, GUILayout.Width(75));

            EditorGUILayout.EndHorizontal();

            // Quick Frame Step Buttons
            EditorGUILayout.BeginHorizontal();
            if (GUILayout.Button(TexMotionLocalization.TrLiteral("◀ Prev Frame"), _ghostButtonStyle, GUILayout.Height(22)))
            {
                SetCurrentFrame(_currentFrame - 1);
            }
            int newFrame = EditorGUILayout.IntSlider(_currentFrame + 1, 1, _data.Frames) - 1;
            if (newFrame != _currentFrame)
            {
                SetCurrentFrame(newFrame);
            }
            if (GUILayout.Button(TexMotionLocalization.TrLiteral("Next Frame ▶"), _ghostButtonStyle, GUILayout.Height(22)))
            {
                SetCurrentFrame(_currentFrame + 1);
            }
            EditorGUILayout.EndHorizontal();

            EditorGUILayout.EndVertical();

            EditorGUILayout.Space(6);

            // Frame Operation Tools
            _foldoutTools = EditorGUILayout.Foldout(
                _foldoutTools,
                TexMotionLocalization.TrLiteral("🛠️ Secondary Pose Tools"),
                true,
                EditorStyles.foldoutHeader);
            if (_foldoutTools)
            {
                EditorGUILayout.BeginVertical(_sectionStyle);

                EditorGUILayout.BeginHorizontal();
                if (GUILayout.Button(
                    new GUIContent(
                        TexMotionLocalization.TrLiteral("📋 Copy"),
                        TexMotionLocalization.TrLiteral("Copy current frame pose to clipboard")),
                    _ghostButtonStyle,
                    GUILayout.Height(26)))
                {
                    _data.CopyFramePose(_currentFrame);
                }
                GUI.enabled = EditableMotionData.HasClipboardData;
                if (GUILayout.Button(
                    new GUIContent(
                        TexMotionLocalization.TrLiteral("📄 Paste"),
                        TexMotionLocalization.TrLiteral("Paste clipboard pose onto this frame")),
                    _ghostButtonStyle,
                    GUILayout.Height(26)))
                {
                    _data.PasteFramePose(_currentFrame);
                    ApplyCurrentFrameToPreview();
                }
                GUI.enabled = true;

                if (GUILayout.Button(
                    new GUIContent(
                        TexMotionLocalization.TrLiteral("🔄 Mirror"),
                        TexMotionLocalization.TrLiteral("Mirror pose across left and right limbs")),
                    _ghostButtonStyle,
                    GUILayout.Height(26)))
                {
                    _data.MirrorFrame(_currentFrame);
                    ApplyCurrentFrameToPreview();
                }

                if (GUILayout.Button(
                    new GUIContent(
                        TexMotionLocalization.TrLiteral("⚖️ Smooth"),
                        TexMotionLocalization.TrLiteral("Interpolate with neighbor frames (smoothing)")),
                    _ghostButtonStyle,
                    GUILayout.Height(26)))
                {
                    _data.SmoothFrame(_currentFrame);
                    ApplyCurrentFrameToPreview();
                }
                EditorGUILayout.EndHorizontal();

                EditorGUILayout.BeginHorizontal();
                if (GUILayout.Button(
                    new GUIContent(
                        TexMotionLocalization.TrLiteral("↩️ Reset Frame"),
                        TexMotionLocalization.TrLiteral("Revert this frame to original generated pose")),
                    _ghostButtonStyle,
                    GUILayout.Height(22)))
                {
                    _data.ResetFrameToOriginal(_currentFrame);
                    ApplyCurrentFrameToPreview();
                }
                if (GUILayout.Button(
                    new GUIContent(
                        TexMotionLocalization.TrLiteral("🧘 T-Pose"),
                        TexMotionLocalization.TrLiteral("Set this frame to neutral T-Pose")),
                    _ghostButtonStyle,
                    GUILayout.Height(22)))
                {
                    _data.SetFrameToTPose(_currentFrame);
                    ApplyCurrentFrameToPreview();
                }
                EditorGUILayout.EndHorizontal();

                EditorGUILayout.EndVertical();
            }

            EditorGUILayout.Space(6);

            // Hand and face assistance is part of the editable preview state. Keep
            // it beside the pose tools so changing a dropdown immediately updates
            // the avatar clone without rebuilding the extracted motion data.
            DrawHandFaceAssistanceControls();

            EditorGUILayout.Space(6);

            // Occlusion & Ambiguity Tools (Phases 3-4)
            DrawOcclusionAndAmbiguityTools();

            EditorGUILayout.Space(6);

            // Advanced Pose Correction (Tweens, Masking, IK, Palette, Glitches, etc.)
            DrawAdvancedPoseCorrectionTools();

            EditorGUILayout.Space(6);

            // Bone & Joint Groups
            DrawBoneSection(TexMotionLocalization.TrLiteral("👤 Root & Hips"), ref _foldoutRoot, () =>
            {
                DrawRootPositionInspector();
                DrawJointRotationSlider(SmplxJoint.Pelvis, TexMotionLocalization.TrLiteral("Pelvis / Hips Rotation"));
            });

            DrawBoneSection(TexMotionLocalization.TrLiteral("🦴 Torso & Head"), ref _foldoutTorso, () =>
            {
                DrawJointRotationSlider(SmplxJoint.Spine1, TexMotionLocalization.TrLiteral("Spine (Lower)"));
                DrawJointRotationSlider(SmplxJoint.Spine2, TexMotionLocalization.TrLiteral("Chest (Middle)"));
                DrawJointRotationSlider(SmplxJoint.Spine3, TexMotionLocalization.TrLiteral("Upper Chest"));
                DrawJointRotationSlider(SmplxJoint.Neck, TexMotionLocalization.TrLiteral("Neck"));
                DrawJointRotationSlider(SmplxJoint.Head, TexMotionLocalization.TrLiteral("Head"));
            });

            DrawBoneSection(TexMotionLocalization.TrLiteral("💪 Left Arm"), ref _foldoutLeftArm, () =>
            {
                DrawJointRotationSlider(SmplxJoint.L_Collar, TexMotionLocalization.TrLiteral("Left Shoulder (Collar)"));
                DrawJointRotationSlider(SmplxJoint.L_Shoulder, TexMotionLocalization.TrLiteral("Left Upper Arm"));
                DrawJointRotationSlider(SmplxJoint.L_Elbow, TexMotionLocalization.TrLiteral("Left Elbow (Forearm)"));
                DrawJointRotationSlider(SmplxJoint.L_Wrist, TexMotionLocalization.TrLiteral("Left Wrist (Hand)"));
            });

            DrawBoneSection(TexMotionLocalization.TrLiteral("💪 Right Arm"), ref _foldoutRightArm, () =>
            {
                DrawJointRotationSlider(SmplxJoint.R_Collar, TexMotionLocalization.TrLiteral("Right Shoulder (Collar)"));
                DrawJointRotationSlider(SmplxJoint.R_Shoulder, TexMotionLocalization.TrLiteral("Right Upper Arm"));
                DrawJointRotationSlider(SmplxJoint.R_Elbow, TexMotionLocalization.TrLiteral("Right Elbow (Forearm)"));
                DrawJointRotationSlider(SmplxJoint.R_Wrist, TexMotionLocalization.TrLiteral("Right Wrist (Hand)"));
            });

            DrawBoneSection(TexMotionLocalization.TrLiteral("🦵 Left Leg"), ref _foldoutLeftLeg, () =>
            {
                DrawJointRotationSlider(SmplxJoint.L_Hip, TexMotionLocalization.TrLiteral("Left Hip (Upper Leg)"));
                DrawJointRotationSlider(SmplxJoint.L_Knee, TexMotionLocalization.TrLiteral("Left Knee (Lower Leg)"));
                DrawJointRotationSlider(SmplxJoint.L_Ankle, TexMotionLocalization.TrLiteral("Left Ankle (Foot)"));
                DrawJointRotationSlider(SmplxJoint.L_Foot, TexMotionLocalization.TrLiteral("Left Toes"));
            });

            DrawBoneSection(TexMotionLocalization.TrLiteral("🦵 Right Leg"), ref _foldoutRightLeg, () =>
            {
                DrawJointRotationSlider(SmplxJoint.R_Hip, TexMotionLocalization.TrLiteral("Right Hip (Upper Leg)"));
                DrawJointRotationSlider(SmplxJoint.R_Knee, TexMotionLocalization.TrLiteral("Right Knee (Lower Leg)"));
                DrawJointRotationSlider(SmplxJoint.R_Ankle, TexMotionLocalization.TrLiteral("Right Ankle (Foot)"));
                DrawJointRotationSlider(SmplxJoint.R_Foot, TexMotionLocalization.TrLiteral("Right Toes"));
            });

            EditorGUILayout.EndScrollView();
        }

        private void DrawHandFaceAssistanceControls()
        {
            if (_data == null) return;

            EditorGUILayout.BeginVertical(_sectionStyle);
            EditorGUILayout.LabelField(
                TexMotionLocalization.Tr(TexMotionLocalization.LiveHandFaceAssistance),
                _sectionLabelStyle);

            EditorGUI.BeginChangeCheck();
            HandPoseType handPose = (HandPoseType)EditorGUILayout.EnumPopup(
                TexMotionLocalization.Tr(TexMotionLocalization.HandPose),
                _data.HandPose);
            FaceEmotionType faceEmotion = (FaceEmotionType)EditorGUILayout.EnumPopup(
                TexMotionLocalization.Tr(TexMotionLocalization.FaceEmotion),
                _data.FaceEmotion);
            float intensity = _data.EmotionIntensity;
            if (faceEmotion != FaceEmotionType.None && faceEmotion != FaceEmotionType.AutoDetect)
            {
                intensity = EditorGUILayout.Slider(
                    TexMotionLocalization.Tr(TexMotionLocalization.EmotionIntensity),
                    intensity,
                    0.1f,
                    1.0f);
            }

            if (EditorGUI.EndChangeCheck())
            {
                _data.HandPose = handPose;
                _data.FaceEmotion = faceEmotion;
                _data.EmotionIntensity = Mathf.Clamp(intensity, 0.1f, 1.0f);
                ApplyCurrentFrameToPreview();
                Repaint();
            }

            EditorGUILayout.LabelField(
                TexMotionLocalization.Tr(TexMotionLocalization.LiveHandFaceAssistanceHint),
                _metaLabelStyle);
            EditorGUILayout.EndVertical();
        }

        private void DrawBoneSection(string title, ref bool foldout, Action drawContents)
        {
            foldout = EditorGUILayout.Foldout(foldout, title, true, EditorStyles.foldoutHeader);
            if (foldout)
            {
                EditorGUILayout.BeginVertical(_sectionStyle);
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
                GUI.backgroundColor = MotionTimelineTheme.WithAlpha(MotionTimelineTheme.AcidLime, 0.18f);
                EditorGUILayout.BeginVertical(_sectionStyle);
                GUI.backgroundColor = Color.white;
            }

            EditorGUILayout.BeginHorizontal();
            EditorGUILayout.LabelField(TexMotionLocalization.TrLiteral("Root Position (Offset):"), EditorStyles.miniBoldLabel);
            if (GUILayout.Button(
                isSelected
                    ? TexMotionLocalization.TrLiteral("🎯 Active")
                    : TexMotionLocalization.TrLiteral("🎯 Select"),
                isSelected ? _accentButtonStyle : _ghostButtonStyle,
                GUILayout.Width(isSelected ? 65 : 60)))
            {
                _selectedJoint = isSelected ? (SmplxJoint?)null : SmplxJoint.Pelvis;
                Repaint();
            }
            EditorGUILayout.EndHorizontal();

            Vector3 rootPos = _data.GetRootPosition(_currentFrame);

            EditorGUI.BeginChangeCheck();
            float x = EditorGUILayout.Slider(TexMotionLocalization.TrLiteral("X (Left/Right)"), rootPos.x, -2.0f, 2.0f);
            float y = EditorGUILayout.Slider(TexMotionLocalization.TrLiteral("Y (Height)"), rootPos.y, -2.0f, 2.0f);
            float z = EditorGUILayout.Slider(TexMotionLocalization.TrLiteral("Z (Fwd/Back)"), rootPos.z, -2.0f, 2.0f);

            if (EditorGUI.EndChangeCheck())
            {
                _data.RecordUndo(TexMotionLocalization.TrFormat("Adjust Root Position Frame {0}", _currentFrame));
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
                GUI.backgroundColor = MotionTimelineTheme.WithAlpha(MotionTimelineTheme.AcidLime, 0.18f);
                EditorGUILayout.BeginVertical(_sectionStyle);
                GUI.backgroundColor = Color.white;
            }

            EditorGUILayout.BeginHorizontal();

            // Bone select toggle button
            if (GUILayout.Button(
                isSelected ? TexMotionLocalization.TrLiteral("🎯 Active") : "🎯",
                isSelected ? _accentButtonStyle : _ghostButtonStyle,
                GUILayout.Width(isSelected ? 62 : 28)))
            {
                _selectedJoint = isSelected ? (SmplxJoint?)null : joint;
                Repaint();
            }

            var titleStyle = isSelected ? EditorStyles.boldLabel : EditorStyles.miniBoldLabel;
            EditorGUILayout.LabelField(label, titleStyle);

            if (GUILayout.Button(TexMotionLocalization.TrLiteral("Reset"), _ghostButtonStyle, GUILayout.Width(45)))
            {
                _data.ResetJointToOriginal(_currentFrame, joint);
                ApplyCurrentFrameToPreview();
                EditorGUILayout.EndHorizontal();
                if (isSelected) EditorGUILayout.EndVertical();
                return;
            }
            EditorGUILayout.EndHorizontal();

            EditorGUI.BeginChangeCheck();
            float rx = EditorGUILayout.Slider(TexMotionLocalization.TrLiteral("Pitch (X)"), euler.x, -180f, 180f);
            float ry = EditorGUILayout.Slider(TexMotionLocalization.TrLiteral("Yaw (Y)"), euler.y, -180f, 180f);
            float rz = EditorGUILayout.Slider(TexMotionLocalization.TrLiteral("Roll (Z)"), euler.z, -180f, 180f);

            if (EditorGUI.EndChangeCheck())
            {
                _data.RecordUndo(TexMotionLocalization.TrFormat("Rotate {0} on Frame {1}", GetJointFriendlyName(joint), _currentFrame));
                _data.SetJointEuler(_currentFrame, joint, new Vector3(rx, ry, rz));
                ApplyCurrentFrameToPreview();
            }

            if (isSelected)
            {
                EditorGUILayout.EndVertical();
            }

            EditorGUILayout.Space(2);
        }

        private bool _foldoutOcclusion = true;

        private void DrawOcclusionAndAmbiguityTools()
        {
            _foldoutOcclusion = EditorGUILayout.Foldout(
                _foldoutOcclusion,
                TexMotionLocalization.TrLiteral("🔮 Occlusion & Ambiguity Tools"),
                true,
                EditorStyles.foldoutHeader);
            if (!_foldoutOcclusion) return;

            EditorGUILayout.BeginVertical(_sectionStyle);

            bool isUncertain = _data.HasUncertainty(_currentFrame, out UncertaintyInterval interval);
            if (isUncertain && interval != null)
            {
                EditorGUILayout.HelpBox(
                    TexMotionLocalization.TrFormat(
                        "⚠ Ambiguity: {0}\nFrames: {1} - {2} (Confidence: {3:P0})",
                        interval.Reason ?? TexMotionLocalization.TrLiteral("Tracking Ambiguity"),
                        interval.StartFrame + 1,
                        interval.EndFrame + 1,
                        interval.Confidence),
                    MessageType.Warning);
            }
            else
            {
                EditorGUILayout.LabelField(
                    TexMotionLocalization.TrLiteral("Pose tracking confident on this frame. Use tools below to manually resolve occlusion orders."),
                    EditorStyles.miniLabel);
            }

            EditorGUILayout.BeginHorizontal();

            // Swap Leg Crossing button
            string swapText = isUncertain && interval != null && !string.IsNullOrEmpty(interval.Reason) && interval.Reason.Contains("crossing")
                ? TexMotionLocalization.TrFormat("🔄 Swap Leg Crossing ({0}-{1})", interval.StartFrame + 1, interval.EndFrame + 1)
                : TexMotionLocalization.TrLiteral("🔄 Swap Leg Crossing");

            if (GUILayout.Button(
                new GUIContent(
                    swapText,
                    TexMotionLocalization.TrLiteral("Reverses anterior/posterior leg crossing order (Swaps which leg is in front)")),
                _accentButtonStyle,
                GUILayout.Height(26)))
            {
                int s = interval != null ? interval.StartFrame : Mathf.Max(0, _currentFrame - 4);
                int e = interval != null ? interval.EndFrame : Mathf.Min(_data.Frames - 1, _currentFrame + 4);
                _data.SwapLegCrossing(s, e);
                ApplyCurrentFrameToPreview();
                Repaint();
            }
            EditorGUILayout.EndHorizontal();

            EditorGUILayout.Space(3);

            // Arm Pose correction buttons: Behind Head vs Front Chest
            EditorGUILayout.BeginHorizontal();
            if (GUILayout.Button(
                new GUIContent(
                    TexMotionLocalization.TrLiteral("✋ L-Arm Behind"),
                    TexMotionLocalization.TrLiteral("Fixes left arm posture to behind head")),
                _ghostButtonStyle,
                GUILayout.Height(24)))
            {
                int s = Mathf.Max(0, _currentFrame - 3);
                int e = Mathf.Min(_data.Frames - 1, _currentFrame + 3);
                _data.FixArmPose(s, e, isLeftArm: true, behindHead: true);
                ApplyCurrentFrameToPreview();
                Repaint();
            }
            if (GUILayout.Button(
                new GUIContent(
                    TexMotionLocalization.TrLiteral("✋ R-Arm Behind"),
                    TexMotionLocalization.TrLiteral("Fixes right arm posture to behind head")),
                _ghostButtonStyle,
                GUILayout.Height(24)))
            {
                int s = Mathf.Max(0, _currentFrame - 3);
                int e = Mathf.Min(_data.Frames - 1, _currentFrame + 3);
                _data.FixArmPose(s, e, isLeftArm: false, behindHead: true);
                ApplyCurrentFrameToPreview();
                Repaint();
            }
            EditorGUILayout.EndHorizontal();

            EditorGUILayout.BeginHorizontal();
            if (GUILayout.Button(
                new GUIContent(
                    TexMotionLocalization.TrLiteral("✋ L-Arm Front"),
                    TexMotionLocalization.TrLiteral("Fixes left arm posture to front of chest")),
                _ghostButtonStyle,
                GUILayout.Height(24)))
            {
                int s = Mathf.Max(0, _currentFrame - 3);
                int e = Mathf.Min(_data.Frames - 1, _currentFrame + 3);
                _data.FixArmPose(s, e, isLeftArm: true, behindHead: false);
                ApplyCurrentFrameToPreview();
                Repaint();
            }
            if (GUILayout.Button(
                new GUIContent(
                    TexMotionLocalization.TrLiteral("✋ R-Arm Front"),
                    TexMotionLocalization.TrLiteral("Fixes right arm posture to front of chest")),
                _ghostButtonStyle,
                GUILayout.Height(24)))
            {
                int s = Mathf.Max(0, _currentFrame - 3);
                int e = Mathf.Min(_data.Frames - 1, _currentFrame + 3);
                _data.FixArmPose(s, e, isLeftArm: false, behindHead: false);
                ApplyCurrentFrameToPreview();
                Repaint();
            }
            EditorGUILayout.EndHorizontal();

            EditorGUILayout.EndVertical();
        }

        private void DrawAdvancedPoseCorrectionTools()
        {
            _foldoutAdvancedTools = EditorGUILayout.Foldout(
                _foldoutAdvancedTools,
                TexMotionLocalization.TrLiteral("🛠️ Advanced Pose Correction"),
                true,
                EditorStyles.foldoutHeader);
            if (!_foldoutAdvancedTools) return;

            EditorGUILayout.BeginVertical(_sectionStyle);

            // 1. Target Body Mask & In-Out Selection
            EditorGUILayout.LabelField(TexMotionLocalization.TrLiteral("🎯 Scope & Masking"), EditorStyles.boldLabel);
            EditorGUILayout.BeginVertical(_cardStyle);

            EditorGUILayout.BeginHorizontal();
            EditorGUILayout.LabelField(TexMotionLocalization.TrLiteral("Body Mask:"), GUILayout.Width(75));
            _selectedBodyMask = (BodyPartMask)EditorGUILayout.EnumPopup(_selectedBodyMask);
            EditorGUILayout.EndHorizontal();

            EditorGUILayout.BeginHorizontal();
            if (GUILayout.Button(TexMotionLocalization.TrLiteral("All"), _ghostButtonStyle, GUILayout.Height(18))) _selectedBodyMask = BodyPartMask.All;
            if (GUILayout.Button(TexMotionLocalization.TrLiteral("Upper"), _ghostButtonStyle, GUILayout.Height(18))) _selectedBodyMask = BodyPartMask.UpperBody;
            if (GUILayout.Button(TexMotionLocalization.TrLiteral("Lower"), _ghostButtonStyle, GUILayout.Height(18))) _selectedBodyMask = BodyPartMask.LowerBody;
            if (GUILayout.Button(TexMotionLocalization.TrLiteral("Arms"), _ghostButtonStyle, GUILayout.Height(18))) _selectedBodyMask = BodyPartMask.Arms;
            if (GUILayout.Button(TexMotionLocalization.TrLiteral("Legs"), _ghostButtonStyle, GUILayout.Height(18))) _selectedBodyMask = BodyPartMask.Legs;
            EditorGUILayout.EndHorizontal();

            // Masked actions on current frame
            EditorGUILayout.BeginHorizontal();
            GUI.enabled = EditableMotionData.HasClipboardData;
            if (GUILayout.Button(new GUIContent(TexMotionLocalization.TrLiteral("📋 Paste Masked"), TexMotionLocalization.TrLiteral("Paste clipboard pose to masked parts only")), _ghostButtonStyle, GUILayout.Height(20)))
            {
                _data.PasteFramePose(_currentFrame, _selectedBodyMask);
                ApplyCurrentFrameToPreview();
                Repaint();
            }
            GUI.enabled = true;
            if (GUILayout.Button(new GUIContent(TexMotionLocalization.TrLiteral("↩️ Reset Masked"), TexMotionLocalization.TrLiteral("Reset masked parts to original generated pose")), _ghostButtonStyle, GUILayout.Height(20)))
            {
                _data.ResetFrameToOriginal(_currentFrame, _selectedBodyMask);
                ApplyCurrentFrameToPreview();
                Repaint();
            }
            if (GUILayout.Button(new GUIContent(TexMotionLocalization.TrLiteral("⚖️ Smooth Masked"), TexMotionLocalization.TrLiteral("Smooth masked parts with neighbor frames")), _ghostButtonStyle, GUILayout.Height(20)))
            {
                _data.SmoothFrame(_currentFrame, _selectedBodyMask);
                ApplyCurrentFrameToPreview();
                Repaint();
            }
            EditorGUILayout.EndHorizontal();

            EditorGUILayout.Space(4);

            // In / Out Range
            EditorGUILayout.BeginHorizontal();
            EditorGUILayout.LabelField(TexMotionLocalization.TrLiteral("Range In:"), GUILayout.Width(60));
            _rangeStartFrame = EditorGUILayout.IntSlider(_rangeStartFrame + 1, 1, _data.Frames) - 1;
            if (GUILayout.Button(TexMotionLocalization.TrLiteral("Set In"), _ghostButtonStyle, GUILayout.Width(50), GUILayout.Height(18)))
            {
                _rangeStartFrame = _currentFrame;
                if (_rangeEndFrame < _rangeStartFrame) _rangeEndFrame = _rangeStartFrame;
                Repaint();
            }
            EditorGUILayout.EndHorizontal();

            EditorGUILayout.BeginHorizontal();
            EditorGUILayout.LabelField(TexMotionLocalization.TrLiteral("Range Out:"), GUILayout.Width(60));
            _rangeEndFrame = EditorGUILayout.IntSlider(_rangeEndFrame + 1, 1, _data.Frames) - 1;
            if (_rangeEndFrame < _rangeStartFrame) _rangeEndFrame = _rangeStartFrame;
            if (GUILayout.Button(TexMotionLocalization.TrLiteral("Set Out"), _ghostButtonStyle, GUILayout.Width(50), GUILayout.Height(18)))
            {
                _rangeEndFrame = _currentFrame;
                if (_rangeStartFrame > _rangeEndFrame) _rangeStartFrame = _rangeEndFrame;
                Repaint();
            }
            EditorGUILayout.EndHorizontal();

            EditorGUILayout.BeginHorizontal();
            if (GUILayout.Button(TexMotionLocalization.TrLiteral("Select All Frames"), _ghostButtonStyle, GUILayout.Height(18)))
            {
                _rangeStartFrame = 0;
                _rangeEndFrame = Mathf.Max(0, _data.Frames - 1);
                Repaint();
            }
            EditorGUILayout.LabelField(TexMotionLocalization.TrFormat("Frames: {0} ({1}..{2})", _rangeEndFrame - _rangeStartFrame + 1, _rangeStartFrame + 1, _rangeEndFrame + 1), _metaLabelStyle);
            EditorGUILayout.EndHorizontal();

            EditorGUILayout.EndVertical();

            EditorGUILayout.Space(6);

            // 2. Tweening & Interpolation
            EditorGUILayout.LabelField(TexMotionLocalization.TrLiteral("🎬 Range Tween (Keyframing)"), EditorStyles.boldLabel);
            EditorGUILayout.BeginVertical(_cardStyle);
            EditorGUILayout.BeginHorizontal();
            EditorGUILayout.LabelField(TexMotionLocalization.TrLiteral("Easing:"), GUILayout.Width(60));
            _selectedEasing = (EasingType)EditorGUILayout.EnumPopup(_selectedEasing);
            EditorGUILayout.EndHorizontal();
            if (GUILayout.Button(
                new GUIContent(TexMotionLocalization.TrLiteral("Interpolate In ➔ Out (Tween)"), TexMotionLocalization.TrLiteral("Smoothly interpolates poses from Range In to Range Out using selected easing and body mask")),
                _accentButtonStyle,
                GUILayout.Height(24)))
            {
                _data.TweenRange(_rangeStartFrame, _rangeEndFrame, _selectedEasing, _selectedBodyMask);
                ApplyCurrentFrameToPreview();
                Repaint();
            }
            EditorGUILayout.EndVertical();

            EditorGUILayout.Space(6);

            // 3. Loop Boundary Blender
            EditorGUILayout.LabelField(TexMotionLocalization.TrLiteral("🔁 Loop Blender"), EditorStyles.boldLabel);
            EditorGUILayout.BeginVertical(_cardStyle);
            int maxLoop = Mathf.Max(2, _data.Frames / 2);
            _loopBlendFrames = EditorGUILayout.IntSlider(TexMotionLocalization.TrLiteral("Margin Frames:"), _loopBlendFrames, 2, maxLoop);
            if (GUILayout.Button(
                new GUIContent(TexMotionLocalization.TrLiteral("Blend Loop Boundary (Start ↔ End)"), TexMotionLocalization.TrLiteral("Cross-blends motion start and end frames for seamless looping")),
                _ghostButtonStyle,
                GUILayout.Height(24)))
            {
                _data.BlendLoopBoundary(_loopBlendFrames, _selectedBodyMask);
                ApplyCurrentFrameToPreview();
                Repaint();
            }
            EditorGUILayout.EndVertical();

            EditorGUILayout.Space(6);

            // 4. Additive Range Offset
            EditorGUILayout.LabelField(TexMotionLocalization.TrLiteral("➕ Additive Range Offset"), EditorStyles.boldLabel);
            EditorGUILayout.BeginVertical(_cardStyle);
            _additiveRootOffset.y = EditorGUILayout.Slider(TexMotionLocalization.TrLiteral("Hips Y Offset (m):"), _additiveRootOffset.y, -0.5f, 0.5f);
            _additiveArmEulerOffset.z = EditorGUILayout.Slider(TexMotionLocalization.TrLiteral("Arm Open Angle (deg):"), _additiveArmEulerOffset.z, -30f, 30f);
            _offsetFadeEdges = EditorGUILayout.Toggle(TexMotionLocalization.TrLiteral("Fade at Range Edges"), _offsetFadeEdges);
            if (GUILayout.Button(
                new GUIContent(TexMotionLocalization.TrLiteral("Apply Additive Offset"), TexMotionLocalization.TrLiteral("Adds root elevation and arm angle across the selected frame range")),
                _ghostButtonStyle,
                GUILayout.Height(24)))
            {
                _data.ApplyRangeOffset(_rangeStartFrame, _rangeEndFrame, _additiveRootOffset, _additiveArmEulerOffset, _offsetFadeEdges);
                ApplyCurrentFrameToPreview();
                Repaint();
            }
            EditorGUILayout.EndVertical();

            EditorGUILayout.Space(6);

            // 5. Penetration Limiter & Foot Grounding
            EditorGUILayout.LabelField(TexMotionLocalization.TrLiteral("🛡️ Safety & Grounding"), EditorStyles.boldLabel);
            EditorGUILayout.BeginVertical(_cardStyle);
            _armpitLimitAngle = EditorGUILayout.Slider(TexMotionLocalization.TrLiteral("Min Armpit Angle:"), _armpitLimitAngle, 5f, 40f);
            if (GUILayout.Button(
                new GUIContent(TexMotionLocalization.TrLiteral("Apply Armpit Limiter (Range)"), TexMotionLocalization.TrLiteral("Prevents arms from penetrating chest/torso across range")),
                _ghostButtonStyle,
                GUILayout.Height(22)))
            {
                _data.ApplyArmpitPenetrationLimiter(_rangeStartFrame, _rangeEndFrame, _armpitLimitAngle);
                ApplyCurrentFrameToPreview();
                Repaint();
            }

            EditorGUILayout.Space(3);
            _groundPlaneY = EditorGUILayout.FloatField(TexMotionLocalization.TrLiteral("Ground Plane Y:"), _groundPlaneY);
            EditorGUILayout.BeginHorizontal();
            if (GUILayout.Button(
                new GUIContent(TexMotionLocalization.TrLiteral("🦶 Ground (Cur)"), TexMotionLocalization.TrLiteral("Snaps feet to ground plane on current frame")),
                _ghostButtonStyle,
                GUILayout.Height(22)))
            {
                _data.ApplyFootGrounding(_currentFrame, _groundPlaneY);
                ApplyCurrentFrameToPreview();
                Repaint();
            }
            if (GUILayout.Button(
                new GUIContent(TexMotionLocalization.TrLiteral("🦶 Ground (Range)"), TexMotionLocalization.TrLiteral("Snaps feet to ground plane across In-Out range")),
                _ghostButtonStyle,
                GUILayout.Height(22)))
            {
                for (int f = _rangeStartFrame; f <= _rangeEndFrame; f++)
                {
                    _data.ApplyFootGrounding(f, _groundPlaneY);
                }
                ApplyCurrentFrameToPreview();
                Repaint();
            }
            if (GUILayout.Button(
                new GUIContent(TexMotionLocalization.TrLiteral("⚓ Lock Feet"), TexMotionLocalization.TrLiteral("Locks feet positions to prevent sliding across In-Out range")),
                _ghostButtonStyle,
                GUILayout.Height(22)))
            {
                _data.LockFootPosition(_rangeStartFrame, _rangeEndFrame, isLeftFoot: true);
                _data.LockFootPosition(_rangeStartFrame, _rangeEndFrame, isLeftFoot: false);
                ApplyCurrentFrameToPreview();
                Repaint();
            }
            EditorGUILayout.EndHorizontal();
            EditorGUILayout.EndVertical();

            EditorGUILayout.Space(6);

            // 6. Glitch Highlighter & Auto-Fixer
            EditorGUILayout.LabelField(TexMotionLocalization.TrLiteral("⚡ Glitch Highlighter & Fixer"), EditorStyles.boldLabel);
            EditorGUILayout.BeginVertical(_cardStyle);
            EditorGUILayout.BeginHorizontal();
            if (GUILayout.Button(
                new GUIContent(TexMotionLocalization.TrLiteral("🔍 Scan Glitches"), TexMotionLocalization.TrLiteral("Scans for unnatural spikes, high angular velocity, or sudden flips")),
                _accentButtonStyle,
                GUILayout.Height(24)))
            {
                _detectedGlitches = GlitchDetector.DetectGlitches(_data);
            }
            if (_detectedGlitches != null && _detectedGlitches.Count > 0)
            {
                if (GUILayout.Button(
                    new GUIContent(TexMotionLocalization.TrLiteral("⚡ Fix All Glitches"), TexMotionLocalization.TrLiteral("Automatically repairs all detected glitches via neighboring frame Slerp")),
                    _ghostButtonStyle,
                    GUILayout.Height(24)))
                {
                    GlitchDetector.FixAllGlitches(_data, _detectedGlitches);
                    _detectedGlitches.Clear();
                    ApplyCurrentFrameToPreview();
                    Repaint();
                }
            }
            EditorGUILayout.EndHorizontal();

            if (_detectedGlitches != null && _detectedGlitches.Count > 0)
            {
                EditorGUILayout.HelpBox(TexMotionLocalization.TrFormat("⚠️ {0} glitch frame(s) detected!", _detectedGlitches.Count), MessageType.Warning);
                for (int i = 0; i < Mathf.Min(5, _detectedGlitches.Count); i++)
                {
                    var g = _detectedGlitches[i];
                    EditorGUILayout.BeginHorizontal();
                    EditorGUILayout.LabelField(TexMotionLocalization.TrFormat("Frame {0}: {1}", g.FrameIndex + 1, g.Reason), EditorStyles.miniLabel);
                    if (GUILayout.Button(TexMotionLocalization.TrLiteral("Go"), _ghostButtonStyle, GUILayout.Width(35), GUILayout.Height(16)))
                    {
                        SetCurrentFrame(g.FrameIndex);
                    }
                    if (GUILayout.Button(TexMotionLocalization.TrLiteral("Fix"), _ghostButtonStyle, GUILayout.Width(35), GUILayout.Height(16)))
                    {
                        GlitchDetector.FixGlitch(_data, g);
                        _detectedGlitches.RemoveAt(i);
                        ApplyCurrentFrameToPreview();
                        Repaint();
                        break;
                    }
                    EditorGUILayout.EndHorizontal();
                }
                if (_detectedGlitches.Count > 5)
                {
                    EditorGUILayout.LabelField(TexMotionLocalization.TrFormat("... and {0} more.", _detectedGlitches.Count - 5), _metaLabelStyle);
                }
            }
            EditorGUILayout.EndVertical();

            EditorGUILayout.Space(6);

            // 7. Pose Palette & Blending
            EditorGUILayout.LabelField(TexMotionLocalization.TrLiteral("🎨 Pose Palette"), EditorStyles.boldLabel);
            EditorGUILayout.BeginVertical(_cardStyle);

            EditorGUILayout.BeginHorizontal();
            for (int s = 0; s < 8; s++)
            {
                bool has = _posePalette.HasPose(s);
                bool isSel = (_selectedPaletteSlot == s);
                string slotLabel = string.Format("{0}{1}", s + 1, has ? "●" : "");
                Color origCol = GUI.backgroundColor;
                if (isSel) GUI.backgroundColor = MotionTimelineTheme.ElectricCyan;
                else if (has) GUI.backgroundColor = MotionTimelineTheme.PulseGreen;
                if (GUILayout.Button(slotLabel, GUILayout.Width(26), GUILayout.Height(22)))
                {
                    _selectedPaletteSlot = s;
                }
                GUI.backgroundColor = origCol;
            }
            EditorGUILayout.EndHorizontal();

            EditorGUILayout.BeginHorizontal();
            if (GUILayout.Button(
                new GUIContent(TexMotionLocalization.TrFormat("💾 Store to Slot {0}", _selectedPaletteSlot + 1), TexMotionLocalization.TrLiteral("Stores current frame pose into the active slot")),
                _ghostButtonStyle,
                GUILayout.Height(22)))
            {
                _posePalette.Capture(_selectedPaletteSlot, _data, _currentFrame);
            }
            if (GUILayout.Button(
                new GUIContent(TexMotionLocalization.TrLiteral("🗑️ Clear"), TexMotionLocalization.TrLiteral("Clears the active slot")),
                _ghostButtonStyle,
                GUILayout.Width(50),
                GUILayout.Height(22)))
            {
                _posePalette.Clear(_selectedPaletteSlot);
            }
            EditorGUILayout.EndHorizontal();

            GUI.enabled = _posePalette.HasPose(_selectedPaletteSlot);
            _paletteBlendWeight = EditorGUILayout.Slider(TexMotionLocalization.TrLiteral("Blend Weight:"), _paletteBlendWeight, 0f, 1f);
            EditorGUILayout.BeginHorizontal();
            if (GUILayout.Button(
                new GUIContent(TexMotionLocalization.TrLiteral("Apply Blend (Cur)"), TexMotionLocalization.TrLiteral("Blends active palette pose into current frame with selected mask")),
                _accentButtonStyle,
                GUILayout.Height(22)))
            {
                _posePalette.Apply(_selectedPaletteSlot, _data, _currentFrame, _paletteBlendWeight, _selectedBodyMask);
                ApplyCurrentFrameToPreview();
                Repaint();
            }
            if (GUILayout.Button(
                new GUIContent(TexMotionLocalization.TrLiteral("Apply Blend (Range)"), TexMotionLocalization.TrLiteral("Blends active palette pose across In-Out range with selected mask")),
                _ghostButtonStyle,
                GUILayout.Height(22)))
            {
                for (int f = _rangeStartFrame; f <= _rangeEndFrame; f++)
                {
                    _posePalette.Apply(_selectedPaletteSlot, _data, f, _paletteBlendWeight, _selectedBodyMask);
                }
                ApplyCurrentFrameToPreview();
                Repaint();
            }
            EditorGUILayout.EndHorizontal();
            GUI.enabled = true;

            EditorGUILayout.EndVertical();

            EditorGUILayout.Space(6);

            // 8. Retiming (Time Warp)
            EditorGUILayout.LabelField(TexMotionLocalization.TrLiteral("⏳ Range Retiming"), EditorStyles.boldLabel);
            EditorGUILayout.BeginVertical(_cardStyle);
            int curDuration = Mathf.Max(1, _rangeEndFrame - _rangeStartFrame + 1);
            EditorGUILayout.LabelField(TexMotionLocalization.TrFormat("Current Range: {0} frames", curDuration), _metaLabelStyle);
            _retimeNewFrameCount = EditorGUILayout.IntSlider(TexMotionLocalization.TrLiteral("New Frames:"), _retimeNewFrameCount, 2, Mathf.Max(curDuration * 3, 120));
            if (GUILayout.Button(
                new GUIContent(TexMotionLocalization.TrLiteral("⏳ Retime Range (In-Out)"), TexMotionLocalization.TrLiteral("Resamples and rescales the In-Out range frames")),
                _ghostButtonStyle,
                GUILayout.Height(24)))
            {
                _data.RetimeRange(_rangeStartFrame, _rangeEndFrame, _retimeNewFrameCount);
                _currentFrame = Mathf.Clamp(_currentFrame, 0, _data.Frames - 1);
                ApplyCurrentFrameToPreview();
                Repaint();
            }
            EditorGUILayout.EndVertical();

            EditorGUILayout.EndVertical();
        }

        #endregion

        #region Bottom Timeline Panel

        private void DrawBottomTimelinePanel()
        {
            Rect panelRect = GUILayoutUtility.GetRect(position.width, TIMELINE_HEIGHT, GUILayout.ExpandWidth(true));
            GUI.Box(panelRect, GUIContent.none, _timelineBgStyle);

            if (Event.current.type == EventType.Repaint)
            {
                EditorGUI.DrawRect(new Rect(panelRect.x, panelRect.y, panelRect.width, 1f), MotionTimelineTheme.Graphite);
                EditorGUI.DrawRect(new Rect(panelRect.x, panelRect.yMax - 1f, panelRect.width, 1f), MotionTimelineTheme.Graphite);
            }

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
            if (GUILayout.Button(new GUIContent("|◀", TexMotionLocalization.Tr(TexMotionLocalization.FirstFrameHome)), _ghostButtonStyle, GUILayout.Width(36), GUILayout.Height(28)))
            {
                SetCurrentFrame(0);
            }

            // Prev Keyframe (modified frame)
            if (GUILayout.Button(new GUIContent("◆◀", TexMotionLocalization.Tr(TexMotionLocalization.PreviousModifiedFrame)), _ghostButtonStyle, GUILayout.Width(36), GUILayout.Height(28)))
            {
                JumpToPreviousModifiedFrame();
            }

            // Step Back 1 Frame
            if (GUILayout.Button(new GUIContent("◀", TexMotionLocalization.Tr(TexMotionLocalization.PreviousFrameLeft)), _ghostButtonStyle, GUILayout.Width(36), GUILayout.Height(28)))
            {
                SetCurrentFrame(_currentFrame - 1);
            }

            // Play / Pause
            string playIcon = _isPlaying
                ? TexMotionLocalization.Tr(TexMotionLocalization.Pause)
                : TexMotionLocalization.Tr(TexMotionLocalization.Play);
            if (GUILayout.Button(new GUIContent(playIcon, TexMotionLocalization.Tr(TexMotionLocalization.TogglePlaybackSpace)), _accentButtonStyle, GUILayout.Width(75), GUILayout.Height(28)))
            {
                TogglePlay();
            }

            // Step Forward 1 Frame
            if (GUILayout.Button(new GUIContent("▶", TexMotionLocalization.Tr(TexMotionLocalization.NextFrameRight)), _ghostButtonStyle, GUILayout.Width(36), GUILayout.Height(28)))
            {
                SetCurrentFrame(_currentFrame + 1);
            }

            // Next Keyframe
            if (GUILayout.Button(new GUIContent("▶◆", TexMotionLocalization.Tr(TexMotionLocalization.NextModifiedFrame)), _ghostButtonStyle, GUILayout.Width(36), GUILayout.Height(28)))
            {
                JumpToNextModifiedFrame();
            }

            // Last Frame
            if (GUILayout.Button(new GUIContent("▶|", TexMotionLocalization.Tr(TexMotionLocalization.LastFrameEnd)), _ghostButtonStyle, GUILayout.Width(36), GUILayout.Height(28)))
            {
                SetCurrentFrame(_data.Frames - 1);
            }

            EditorGUILayout.Space(12);

            // Loop Toggle
            _isLoop = GUILayout.Toggle(_isLoop, TexMotionLocalization.Tr(TexMotionLocalization.LoopPlayback), _ghostButtonStyle, GUILayout.Width(65), GUILayout.Height(28));

            EditorGUILayout.Space(8);

            // Playback Speed (0.1x step)
            EditorGUILayout.LabelField(TexMotionLocalization.Tr(TexMotionLocalization.Speed), _metaLabelStyle, GUILayout.Width(44));

            // Decrease by 0.1x
            if (GUILayout.Button(new GUIContent("-0.1", TexMotionLocalization.Tr(TexMotionLocalization.DecreaseSpeed)), _ghostButtonStyle, GUILayout.Width(36), GUILayout.Height(28)))
            {
                _playbackSpeed = Mathf.Max(0.1f, Mathf.Round((_playbackSpeed - 0.1f) * 10f) / 10f);
            }

            // Current speed indicator & click-to-reset button
            if (GUILayout.Button(new GUIContent($"{_playbackSpeed:F1}x", TexMotionLocalization.Tr(TexMotionLocalization.ResetSpeed)), _ghostButtonStyle, GUILayout.Width(46), GUILayout.Height(28)))
            {
                _playbackSpeed = 1.0f;
            }

            // Increase by 0.1x
            if (GUILayout.Button(new GUIContent("+0.1", TexMotionLocalization.Tr(TexMotionLocalization.IncreaseSpeed)), _ghostButtonStyle, GUILayout.Width(36), GUILayout.Height(28)))
            {
                _playbackSpeed = Mathf.Min(3.0f, Mathf.Round((_playbackSpeed + 0.1f) * 10f) / 10f);
            }

            // 0.1x step slider (0.1 to 3.0)
            float newSpeedVal = EditorGUILayout.Slider(_playbackSpeed, 0.1f, 3.0f, GUILayout.Width(110));
            _playbackSpeed = Mathf.Round(newSpeedVal * 10f) / 10f;

            GUILayout.FlexibleSpace();

            // Timeline Zoom Slider
            EditorGUILayout.LabelField(TexMotionLocalization.Tr(TexMotionLocalization.Zoom), _metaLabelStyle, GUILayout.Width(38));
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
            EditorGUI.DrawRect(rulerRect, MotionTimelineTheme.Obsidian);

            float frameTrackY = rulerHeight;
            float frameTrackHeight = viewRect.height - rulerHeight;

            // Draw In / Out Range Selection Highlight
            if (_rangeEndFrame > _rangeStartFrame)
            {
                float startX = _rangeStartFrame * _pixelsPerFrame;
                float endX = (_rangeEndFrame + 1) * _pixelsPerFrame;
                Rect rangeHighlight = new Rect(startX, rulerHeight, endX - startX, frameTrackHeight);
                EditorGUI.DrawRect(rangeHighlight, new Color(0.0f, 0.85f, 1.0f, 0.12f));

                // In and Out Boundary Lines
                EditorGUI.DrawRect(new Rect(startX, 0, 2f, viewRect.height), MotionTimelineTheme.SignalTeal);
                EditorGUI.DrawRect(new Rect(endX - 2f, 0, 2f, viewRect.height), MotionTimelineTheme.SignalTeal);

                GUI.Label(new Rect(startX + 3, 2, 20, rulerHeight), "IN", _rulerTextStyle);
                GUI.Label(new Rect(endX - 24, 2, 22, rulerHeight), "OUT", _rulerTextStyle);
            }

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
                    string label = TexMotionLocalization.TrFormat("{0}\n{1:F2}s", f, time);
                    Rect tickLabelRect = new Rect(x - 15, 2, 40, rulerHeight);
                    GUI.Label(tickLabelRect, label, _rulerTextStyle);

                    // Grid line
                    EditorGUI.DrawRect(new Rect(x, rulerHeight, 1, frameTrackHeight), MotionTimelineTheme.WithAlpha(MotionTimelineTheme.Graphite, 0.72f));
                }

                // Frame Cell Box
                Rect frameBox = new Rect(x + 1, frameTrackY + 4, _pixelsPerFrame - 2, frameTrackHeight - 8);

                bool isUncertain = _data.HasUncertainty(f, out UncertaintyInterval uncertaintyInfo);

                Color boxCol = isCurrent
                    ? MotionTimelineTheme.WithAlpha(MotionTimelineTheme.AcidLime, 0.36f)
                    : (isUncertain
                        ? MotionTimelineTheme.WithAlpha(MotionTimelineTheme.CoralRed, 0.30f)
                        : (isModified ? MotionTimelineTheme.WithAlpha(MotionTimelineTheme.PulseGreen, 0.30f) : MotionTimelineTheme.WithAlpha(MotionTimelineTheme.Graphite, 0.72f)));

                EditorGUI.DrawRect(frameBox, boxCol);

                // Top accent bar for uncertainty
                if (isUncertain)
                {
                    Rect warnBar = new Rect(frameBox.x, frameBox.y, frameBox.width, 3);
                    EditorGUI.DrawRect(warnBar, MotionTimelineTheme.CoralRed);

                    // Tooltip label for tracking ambiguity / uncertainty
                    string tip = uncertaintyInfo != null && !string.IsNullOrEmpty(uncertaintyInfo.Reason)
                        ? TexMotionLocalization.TrFormat(
                            "Frame {0}: {1} (Confidence: {2:P0})",
                            f + 1,
                            uncertaintyInfo.Reason,
                            uncertaintyInfo.Confidence)
                        : TexMotionLocalization.TrFormat(
                            "Frame {0}: Occlusion / Ambiguity Warning",
                            f + 1);
                    GUI.Label(frameBox, new GUIContent(string.Empty, tip));
                }

                // Glitch Warning Marker
                if (_detectedGlitches != null && _detectedGlitches.Count > 0)
                {
                    foreach (var g in _detectedGlitches)
                    {
                        if (g.Frame == f)
                        {
                            Rect glitchBar = new Rect(frameBox.x, frameBox.yMax - 4, frameBox.width, 4);
                            EditorGUI.DrawRect(glitchBar, MotionTimelineTheme.CoralRed);
                            GUI.Label(frameBox, new GUIContent(string.Empty, $"⚡ {g.Description}"));
                            break;
                        }
                    }
                }

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
            EditorGUI.DrawRect(scrubberLine, MotionTimelineTheme.AcidLime);

            // Scrubber Top Triangle
            Rect scrubberHead = new Rect(scrubberX - 6f, 0, 12f, 10f);
            EditorGUI.DrawRect(scrubberHead, MotionTimelineTheme.AcidLime);

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
            if (_isPlaying)
            {
                if (_videoPlayer != null && _videoPlayer.isPrepared)
                {
                    float targetTime = GetCurrentFrameTime();
                    float vidLen = Mathf.Max(0.01f, (float)_videoPlayer.length);
                    _videoPlayer.time = Mathf.Clamp(targetTime, 0f, vidLen);
                    _videoPlayer.Play();
                    _lastVideoSeekTime = EditorApplication.timeSinceStartup;
                    EditorApplication.QueuePlayerLoopUpdate();
                }
            }
            else
            {
                SyncVideoPlayerToCurrentFrame();
            }
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

            if (_videoPlayer != null && _videoPlayer.isPrepared)
            {
                if (_isPlaying)
                {
                    _videoPlayer.playbackSpeed = _playbackSpeed;
                    float targetTime = GetCurrentFrameTime();
                    float vidLen = Mathf.Max(0.01f, (float)_videoPlayer.length);
                    _videoPlayer.time = Mathf.Clamp(targetTime, 0f, vidLen);
                    _videoPlayer.Play();
                    _lastVideoSeekTime = EditorApplication.timeSinceStartup;
                    EditorApplication.QueuePlayerLoopUpdate();
                }
                else
                {
                    _videoPlayer.Pause();
                }
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
                _previewUtility.camera.backgroundColor = MotionTimelineTheme.Void;
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

                CachePreviewFingerBones(origAnim);
            }

            _previewFaceRenderer = FaceEmotionHelper.FindFaceRenderer(_previewInstance);
            _previewInitialFaceWeights.Clear();
            if (_previewFaceRenderer != null && _previewFaceRenderer.sharedMesh != null)
            {
                for (int i = 0; i < _previewFaceRenderer.sharedMesh.blendShapeCount; i++)
                {
                    _previewInitialFaceWeights[i] = _previewFaceRenderer.GetBlendShapeWeight(i);
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
            _previewFingerBoneMap.Clear();
            _previewInitialFingerRotations.Clear();
            _previewInitialFaceWeights.Clear();
            _previewFaceRenderer = null;
            _previewHipsTransform = null;
        }

        private void CachePreviewFingerBones(Animator sourceAnimator)
        {
            _previewFingerBoneMap.Clear();
            _previewInitialFingerRotations.Clear();
            if (sourceAnimator == null || _previewInstance == null)
                return;

            CachePreviewFingerBoneSet(sourceAnimator, HandPosePresets.LeftFingerBones);
            CachePreviewFingerBoneSet(sourceAnimator, HandPosePresets.RightFingerBones);
        }

        private void CachePreviewFingerBoneSet(Animator sourceAnimator, HumanBodyBones[] bones)
        {
            foreach (var boneType in bones)
            {
                Transform sourceBone = sourceAnimator.GetBoneTransform(boneType);
                if (sourceBone == null) continue;

                string path = AnimationUtility.CalculateTransformPath(sourceBone, sourceAnimator.transform);
                Transform cloneBone = _previewInstance.transform.Find(path);
                if (cloneBone == null) continue;

                _previewFingerBoneMap[boneType] = cloneBone;
                _previewInitialFingerRotations[boneType] = cloneBone.localRotation;
            }
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

            ApplyHandPoseAndFaceToPreview();
        }

        private void ApplyHandPoseAndFaceToPreview()
        {
            if (_data == null)
                return;

            HandPoseType pose = _data.HandPose;
            foreach (var kvp in _previewFingerBoneMap)
            {
                if (kvp.Value == null || !_previewInitialFingerRotations.TryGetValue(kvp.Key, out Quaternion rest))
                    continue;

                Quaternion offset = pose == HandPoseType.KeepFree
                    ? Quaternion.identity
                    : HandPosePresets.GetFingerLocalRotation(pose, kvp.Key, kvp.Key.ToString().StartsWith("Left", StringComparison.Ordinal));
                kvp.Value.localRotation = rest * offset;
            }

            if (_previewFaceRenderer == null || _previewFaceRenderer.sharedMesh == null)
                return;

            // Restore the avatar's original weights first so switching from Smile
            // to None (or another emotion) is reversible in the live preview.
            foreach (var kvp in _previewInitialFaceWeights)
            {
                _previewFaceRenderer.SetBlendShapeWeight(kvp.Key, kvp.Value);
            }

            FaceEmotionType emotion = _data.FaceEmotion == FaceEmotionType.AutoDetect
                ? FaceEmotionType.None
                : _data.FaceEmotion;
            if (emotion == FaceEmotionType.None)
                return;

            var weights = FaceEmotionHelper.GetBlendShapeWeightsForEmotion(
                _previewFaceRenderer,
                emotion,
                _data.EmotionIntensity);
            foreach (var kvp in weights)
            {
                _previewFaceRenderer.SetBlendShapeWeight(kvp.Key, kvp.Value);
            }
        }

        #endregion

        #region Video Player Synchronization

        private static (int width, int height) CalculateTargetTextureSize(int srcWidth, int srcHeight, int maxDimension = 1280)
        {
            if (srcWidth <= 0 || srcHeight <= 0)
                return (640, 360);

            float aspect = (float)srcWidth / srcHeight;
            int targetW = srcWidth;
            int targetH = srcHeight;

            if (targetW > maxDimension || targetH > maxDimension)
            {
                if (targetW >= targetH)
                {
                    targetW = maxDimension;
                    targetH = Mathf.Max(2, Mathf.RoundToInt(maxDimension / aspect));
                }
                else
                {
                    targetH = maxDimension;
                    targetW = Mathf.Max(2, Mathf.RoundToInt(maxDimension * aspect));
                }
            }

            targetW = Mathf.Max(2, (targetW / 2) * 2);
            targetH = Mathf.Max(2, (targetH / 2) * 2);

            return (targetW, targetH);
        }

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
            _videoPlayer.aspectRatio = VideoAspectRatio.FitInside;
            string rfcUri = new System.Uri(videoPath).AbsoluteUri;
            _videoPlayer.url = rfcUri;
            _loadedVideoPath = videoPath;

            int initW = _data.SourceVideoData.VideoWidth;
            int initH = _data.SourceVideoData.VideoHeight;
            var (texW, texH) = CalculateTargetTextureSize(initW, initH);

            _videoTexture = new RenderTexture(texW, texH, 0, RenderTextureFormat.ARGB32)
            {
                name = "TexMotion_Timeline_VideoRT",
                wrapMode = TextureWrapMode.Clamp,
                filterMode = FilterMode.Bilinear
            };
            _videoTexture.Create();
            _videoPlayer.targetTexture = _videoTexture;

            _videoPlayer.prepareCompleted += (vp) =>
            {
                int vpW = (int)vp.width;
                int vpH = (int)vp.height;
                if (vpW > 0 && vpH > 0)
                {
                    var (expectedW, expectedH) = CalculateTargetTextureSize(vpW, vpH);
                    if (_videoTexture == null || _videoTexture.width != expectedW || _videoTexture.height != expectedH)
                    {
                        vp.targetTexture = null;
                        if (_videoTexture != null)
                        {
                            _videoTexture.Release();
                            DestroyImmediate(_videoTexture);
                        }
                        _videoTexture = new RenderTexture(expectedW, expectedH, 0, RenderTextureFormat.ARGB32)
                        {
                            name = "TexMotion_Timeline_VideoRT",
                            wrapMode = TextureWrapMode.Clamp,
                            filterMode = FilterMode.Bilinear
                        };
                        _videoTexture.Create();
                        vp.targetTexture = _videoTexture;
                    }
                }

                if (_isPlaying)
                {
                    _videoPlayer.playbackSpeed = _playbackSpeed;
                    float curTime = GetCurrentFrameTime();
                    float vidLen = Mathf.Max(0.01f, (float)_videoPlayer.length);
                    _videoPlayer.time = Mathf.Clamp(curTime, 0f, vidLen);
                    _videoPlayer.Play();
                }
                else
                {
                    SyncVideoPlayerToCurrentFrame();
                }
                Repaint();
            };
            _videoPlayer.Prepare();
        }

        private float GetCurrentFrameTime()
        {
            if (_data == null) return 0f;
            return _data.Timestamps != null && _data.Timestamps.Length > _currentFrame
                ? _data.Timestamps[_currentFrame]
                : (float)_currentFrame / Mathf.Max(1.0f, _data.FrameRate);
        }

        private void SyncVideoPlaybackDuringPlay(bool looped, double currentTime)
        {
            if (_videoPlayer == null || !_videoPlayer.isPrepared || _data == null) return;

            _videoPlayer.playbackSpeed = _playbackSpeed;
            float targetTime = GetCurrentFrameTime();
            float vidLen = Mathf.Max(0.01f, (float)_videoPlayer.length);

            if (looped)
            {
                _videoPlayer.time = 0.0;
                _videoPlayer.Play();
                _lastVideoSeekTime = currentTime;
            }
            else if (!_videoPlayer.isPlaying && (currentTime - _lastVideoSeekTime) > 0.35)
            {
                _videoPlayer.time = Mathf.Clamp(targetTime, 0f, vidLen);
                _videoPlayer.Play();
                _lastVideoSeekTime = currentTime;
            }
            else if (_videoPlayer.isPlaying && (currentTime - _lastVideoSeekTime) > 0.5)
            {
                float drift = Mathf.Abs((float)_videoPlayer.time - targetTime);
                if (drift > 0.35f)
                {
                    _videoPlayer.time = Mathf.Clamp(targetTime, 0f, vidLen);
                    _videoPlayer.Play();
                    _lastVideoSeekTime = currentTime;
                }
            }
        }

        private void SyncVideoPlayerToCurrentFrame()
        {
            if (_videoPlayer == null || !_videoPlayer.isPrepared || _data == null) return;

            float time = GetCurrentFrameTime();
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
            if (_data == null) return;

            if (_data.TargetAvatar == null)
            {
                if (TexMotionWindow.Instance != null && TexMotionWindow.Instance.TargetAvatarObject != null)
                {
                    _data.TargetAvatar = TexMotionWindow.Instance.TargetAvatarObject.GetComponent<Animator>();
                }
                if (_data.TargetAvatar == null)
                {
                    var anims = UnityEngine.Object.FindObjectsOfType<Animator>();
                    foreach (var a in anims)
                    {
                        if (a.isHuman) { _data.TargetAvatar = a; break; }
                    }
                }
            }

            if (_data.TargetAvatar == null)
            {
                EditorUtility.DisplayDialog(
                    TexMotionLocalization.TrLiteral("Error"),
                    TexMotionLocalization.TrLiteral("Target Avatar is not configured. Please assign an avatar."),
                    TexMotionLocalization.TrLiteral("OK"));
                return;
            }

            string saveDir = MotionLibraryManager.GeneratedDirectory;
            if (!Directory.Exists(saveDir))
            {
                Directory.CreateDirectory(saveDir);
                AssetDatabase.Refresh();
            }

            string motionName = string.IsNullOrEmpty(_data.ClipName) ? "EditedMotion" : _data.ClipName;
            string clipFileName = motionName.StartsWith("Anim_") ? $"{motionName}.anim" : $"Anim_{motionName}.anim";
            string assetPath = $"{saveDir}/{clipFileName}";

            var buildOptions = new AnimationBuildOptions
            {
                ClipName = motionName.StartsWith("Anim_") ? motionName : $"Anim_{motionName}",
                IsLoop = _isLoop,
                InPlace = _data.InPlace,
                Speed = 1.0f,
                TargetAvatar = _data.TargetAvatar,
                HandPose = _data.HandPose,
                FaceEmotion = _data.FaceEmotion,
                EmotionIntensity = _data.EmotionIntensity
            };

            try
            {
                var clip = _data.BuildAnimationClip(buildOptions);
                AssetDatabase.CreateAsset(clip, assetPath);
                AssetDatabase.SaveAssets();
                AssetDatabase.Refresh();

                EditorGUIUtility.PingObject(clip);
                Selection.activeObject = clip;

                if (TexMotionWindow.Instance != null)
                {
                    TexMotionWindow.Instance.RefreshLibraryExternal();
                }
                else
                {
                    MotionLibraryManager.ScanLibrary();
                }

                EditorUtility.DisplayDialog(
                    TexMotionLocalization.TrLiteral("Motion Saved"),
                    TexMotionLocalization.TrFormat("Successfully exported edited AnimationClip to:\n{0}", assetPath),
                    TexMotionLocalization.TrLiteral("OK"));
            }
            catch (Exception ex)
            {
                Debug.LogError($"[TexMotion Timeline] Failed to save clip: {ex}");
                EditorUtility.DisplayDialog(
                    TexMotionLocalization.TrLiteral("Save Error"),
                    TexMotionLocalization.TrFormat("Failed to save clip:\n{0}", ex.Message),
                    TexMotionLocalization.TrLiteral("OK"));
            }
        }

        private void ApplyEditedMotionToAvatar()
        {
            if (_data == null) return;

            if (_data.TargetAvatar == null)
            {
                if (TexMotionWindow.Instance != null && TexMotionWindow.Instance.TargetAvatarObject != null)
                {
                    _data.TargetAvatar = TexMotionWindow.Instance.TargetAvatarObject.GetComponent<Animator>();
                }
                if (_data.TargetAvatar == null)
                {
                    var anims = UnityEngine.Object.FindObjectsOfType<Animator>();
                    foreach (var a in anims)
                    {
                        if (a.isHuman) { _data.TargetAvatar = a; break; }
                    }
                }
            }

            if (_data.TargetAvatar == null)
            {
                EditorUtility.DisplayDialog(
                    TexMotionLocalization.TrLiteral("Error"),
                    TexMotionLocalization.TrLiteral("Target Avatar is required to apply motion."),
                    TexMotionLocalization.TrLiteral("OK"));
                return;
            }

            var targetAvatarObj = _data.TargetAvatar.gameObject;

            VrcSetupMode setupMode = TexMotionWindow.Instance != null
                ? TexMotionWindow.Instance.SetupMode
                : (ModularAvatarSetup.IsModularAvatarInstalled() ? VrcSetupMode.ModularAvatar : VrcSetupMode.DirectVRCSDK);

            if (setupMode == VrcSetupMode.ModularAvatar && !ModularAvatarSetup.IsModularAvatarInstalled())
            {
                bool shouldInstall = EditorUtility.DisplayDialog(
                    "Modular Avatar が見つかりません",
                    "Modular Avatar (非破壊モード) でのセットアップには「Modular Avatar」が必要です。\n\nModular Avatar を自動的に導入しますか？\n（※キャンセルした場合は「Setup Target」を「DirectVRCSDK」に変更して直接登録することも可能です）",
                    "はい (自動導入する)",
                    "キャンセル"
                );

                if (shouldInstall)
                {
                    ModularAvatarSetup.InstallModularAvatar(() =>
                    {
                        ApplyEditedMotionToAvatar();
                    });
                }
                return;
            }

            try
            {
                string saveDir = MotionLibraryManager.GeneratedDirectory;
                if (!Directory.Exists(saveDir)) Directory.CreateDirectory(saveDir);

                string motionName = string.IsNullOrEmpty(_data.ClipName) ? "EditedMotion" : _data.ClipName;
                string clipFileName = motionName.StartsWith("Anim_") ? $"{motionName}.anim" : $"Anim_{motionName}.anim";
                string clipPath = $"{saveDir}/{clipFileName}";

                VrcMotionType motionType = _isLoop
                    ? VrcMotionType.ToggleLoopPose
                    : (TexMotionWindow.Instance != null ? TexMotionWindow.Instance.MotionType : VrcMotionType.OneShotEmote);

                VrcTargetLayer targetLayer = TexMotionWindow.Instance != null
                    ? TexMotionWindow.Instance.TargetLayer
                    : VrcTargetLayer.ActionLayer;

                ScriptableObject customMenu = TexMotionWindow.Instance != null
                    ? TexMotionWindow.Instance.CustomTargetMenu
                    : null;

                var buildOptions = new AnimationBuildOptions
                {
                    ClipName = motionName.StartsWith("Anim_") ? motionName : $"Anim_{motionName}",
                    IsLoop = motionType == VrcMotionType.ToggleLoopPose,
                    InPlace = _data.InPlace,
                    Speed = 1.0f,
                    TargetAvatar = _data.TargetAvatar,
                    HandPose = _data.HandPose,
                    FaceEmotion = _data.FaceEmotion,
                    EmotionIntensity = _data.EmotionIntensity
                };

                var clip = _data.BuildAnimationClip(buildOptions);
                AssetDatabase.CreateAsset(clip, clipPath);
                AssetDatabase.SaveAssets();
                AssetDatabase.Refresh();

                var vrcConfig = new VrcMotionConfig
                {
                    MotionName = motionName,
                    SetupMode = setupMode,
                    MotionType = motionType,
                    TargetLayer = targetLayer,
                    InPlace = _data.InPlace
                };

                if (setupMode == VrcSetupMode.ModularAvatar)
                {
                    GameObject setupObj = ModularAvatarSetup.SetupAvatarMotion(targetAvatarObj, clip, vrcConfig, saveDir);
                    if (setupObj != null)
                    {
                        EditorGUIUtility.PingObject(setupObj);
                        Selection.activeGameObject = setupObj;
                    }
                    EditorUtility.DisplayDialog(
                        TexMotionLocalization.TrLiteral("Setup Complete"),
                        TexMotionLocalization.TrFormat(
                            "Motion '{0}' was successfully applied to {1} via Modular Avatar!",
                            motionName,
                            targetAvatarObj.name),
                        TexMotionLocalization.TrLiteral("Great!"));
                }
                else
                {
                    VrcDirectSetup.SetupDirectAvatarMotion(targetAvatarObj, clip, vrcConfig, customMenu, saveDir);
                    string menuName = customMenu != null
                        ? customMenu.name
                        : TexMotionLocalization.TrLiteral("VRCExpressionsMenu");
                    EditorUtility.DisplayDialog(
                        TexMotionLocalization.TrLiteral("Setup Complete"),
                        TexMotionLocalization.TrFormat(
                            "Motion '{0}' was successfully added directly into '{1}', ExpressionParameters, and {2}!",
                            motionName,
                            menuName,
                            targetLayer),
                        TexMotionLocalization.TrLiteral("Great!"));
                }

                if (TexMotionWindow.Instance != null)
                {
                    TexMotionWindow.Instance.RefreshLibraryExternal();
                }
                else
                {
                    MotionLibraryManager.ScanLibrary();
                }
            }
            catch (Exception ex)
            {
                Debug.LogError($"[TexMotion Timeline] Failed to apply to avatar: {ex}");
                EditorUtility.DisplayDialog(
                    TexMotionLocalization.TrLiteral("Apply Error"),
                    TexMotionLocalization.TrFormat("Failed to apply motion:\n{0}", ex.Message),
                    TexMotionLocalization.TrLiteral("OK"));
            }
        }

        #endregion

        #region Styles Initialization

        private void InitStyles()
        {
            if (_stylesInitialized &&
                _stylesVersion == TIMELINE_STYLE_VERSION &&
                HasDrawableBackground(_primaryButtonStyle) &&
                HasDrawableBackground(_ghostButtonStyle))
                return;

            _stylesInitialized = true;
            _stylesVersion = TIMELINE_STYLE_VERSION;
            _topBarStyle = MotionTimelineTheme.Toolbar;
            _panelStyle = MotionTimelineTheme.Panel;
            _panelHeaderStyle = MotionTimelineTheme.Toolbar;
            _cardStyle = MotionTimelineTheme.Card;
            _sectionStyle = MotionTimelineTheme.CreateStyle(
                EditorStyles.helpBox,
                MotionTimelineTheme.Carbon,
                MotionTimelineTheme.Mist,
                6,
                8,
                TextAnchor.UpperLeft,
                0);
            _ghostButtonStyle = MotionTimelineTheme.GhostButton;
            _primaryButtonStyle = MotionTimelineTheme.PrimaryButton;
            _dangerButtonStyle = MotionTimelineTheme.CreateStyle(
                EditorStyles.label,
                MotionTimelineTheme.WithAlpha(MotionTimelineTheme.CoralRed, 0.18f),
                MotionTimelineTheme.CoralRed,
                6,
                4,
                TextAnchor.MiddleCenter,
                11,
                1);
            _accentButtonStyle = MotionTimelineTheme.CreateStyle(
                EditorStyles.label,
                MotionTimelineTheme.WithAlpha(MotionTimelineTheme.PulseGreen, 0.28f),
                MotionTimelineTheme.Bone,
                6,
                6,
                TextAnchor.MiddleCenter,
                11,
                1);
            _badgeStyle = MotionTimelineTheme.Badge;
            _modifiedStyle = MotionTimelineTheme.WarningBadge;
            _cleanStyle = MotionTimelineTheme.SuccessBadge;
            _overlayBadgeStyle = MotionTimelineTheme.CreateStyle(
                MotionTimelineTheme.Badge,
                MotionTimelineTheme.Obsidian,
                MotionTimelineTheme.AcidLime,
                4,
                5,
                TextAnchor.MiddleCenter,
                10,
                1);
            _hintOverlayStyle = MotionTimelineTheme.CreateStyle(
                MotionTimelineTheme.MutedLabel,
                Color.clear,
                MotionTimelineTheme.Mist,
                0,
                6,
                TextAnchor.MiddleLeft,
                0);
            _videoBadgeStyle = MotionTimelineTheme.CreateStyle(
                MotionTimelineTheme.Badge,
                MotionTimelineTheme.Obsidian,
                MotionTimelineTheme.SignalTeal,
                4,
                5,
                TextAnchor.MiddleLeft,
                10,
                1);
            _warningBadgeStyle = MotionTimelineTheme.CreateStyle(
                MotionTimelineTheme.WarningBadge,
                MotionTimelineTheme.WithAlpha(MotionTimelineTheme.CoralRed, 0.15f),
                MotionTimelineTheme.CoralRed,
                4,
                5,
                TextAnchor.MiddleLeft,
                10,
                1);
            _frameStatusStyle = MotionTimelineTheme.CreateStyle(
                MotionTimelineTheme.SuccessBadge,
                Color.clear,
                MotionTimelineTheme.PulseGreen,
                0,
                0,
                TextAnchor.MiddleRight,
                10);
            _sectionLabelStyle = MotionTimelineTheme.Heading;
            _metaLabelStyle = MotionTimelineTheme.MutedLabel;

            _emptyCardStyle = MotionTimelineTheme.CreateStyle(
                EditorStyles.helpBox,
                MotionTimelineTheme.Carbon,
                MotionTimelineTheme.Mist,
                12,
                24,
                TextAnchor.UpperCenter,
                0);
            _emptyGuideStyle = MotionTimelineTheme.CreateStyle(
                EditorStyles.helpBox,
                MotionTimelineTheme.Obsidian,
                MotionTimelineTheme.Mist,
                6,
                8,
                TextAnchor.UpperLeft,
                0);
            _emptyTitleStyle = MotionTimelineTheme.CreateStyle(
                EditorStyles.label,
                Color.clear,
                MotionTimelineTheme.Paper,
                0,
                0,
                TextAnchor.MiddleCenter,
                19);
            _emptyTitleStyle.wordWrap = true;
            _emptyDescriptionStyle = MotionTimelineTheme.CreateStyle(
                EditorStyles.label,
                Color.clear,
                MotionTimelineTheme.Mist,
                0,
                0,
                TextAnchor.MiddleCenter,
                12);
            _emptyDescriptionStyle.wordWrap = true;
            _emptyGuideHeaderStyle = MotionTimelineTheme.CreateStyle(
                EditorStyles.miniBoldLabel,
                Color.clear,
                MotionTimelineTheme.SignalTeal,
                0,
                0,
                TextAnchor.MiddleCenter,
                11);
            _emptyGuideStepStyle = MotionTimelineTheme.CreateStyle(
                EditorStyles.miniLabel,
                Color.clear,
                MotionTimelineTheme.Mist,
                0,
                0,
                TextAnchor.MiddleLeft,
                11);
            _emptyGuideStepStyle.wordWrap = true;
            _centeredHintStyle = MotionTimelineTheme.CreateStyle(
                EditorStyles.centeredGreyMiniLabel,
                Color.clear,
                MotionTimelineTheme.Fog,
                0,
                0,
                TextAnchor.MiddleCenter,
                0);
            _centeredHintStyle.wordWrap = true;

            if (_timelineBgStyle == null)
            {
                _timelineBgStyle = MotionTimelineTheme.CreateStyle(
                    GUI.skin.box,
                    MotionTimelineTheme.Carbon,
                    MotionTimelineTheme.Mist,
                    6,
                    8,
                    TextAnchor.UpperLeft,
                    0);
            }

            if (_rulerTextStyle == null)
            {
                _rulerTextStyle = MotionTimelineTheme.CreateStyle(
                    EditorStyles.miniLabel,
                    Color.clear,
                    MotionTimelineTheme.Fog,
                    0,
                    0,
                    TextAnchor.UpperCenter,
                    9);
            }

            if (_keyframeMarkerStyle == null)
            {
                _keyframeMarkerStyle = MotionTimelineTheme.CreateStyle(
                    EditorStyles.boldLabel,
                    Color.clear,
                    MotionTimelineTheme.AcidLime,
                    0,
                    0,
                    TextAnchor.MiddleCenter,
                    12);
            }
        }

        private static bool HasDrawableBackground(GUIStyle style)
        {
            if (style == null || style.normal == null)
                return false;

            // UnityEngine.Object overloads == so this also catches textures destroyed
            // by MotionTimelineTheme.ClearCaches during assembly reload.
            return style.normal.background != null;
        }

        #endregion
    }
}
