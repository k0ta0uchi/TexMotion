using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using TexMotion.Editor.Motion;
using TexMotion.Editor.Video;
using TexMotion.Editor.VRChat;
using TexMotion.Runtime.Motion;
using TexMotion.Runtime.Native;
using TexMotion.Runtime.VRChat;
using UnityEditor;
using UnityEngine;
using UnityEngine.Video;
using Debug = UnityEngine.Debug;

namespace TexMotion.Editor
{
    public class TexMotionWindow : EditorWindow
    {
        private enum Tab
        {
            Generator,
            VideoMotion,
            Library,
            Settings
        }

        private Tab _currentTab = Tab.Generator;

        private enum SettingsCategory
        {
            General,
            MotionGenerator,
            VideoMotion
        }

        private SettingsCategory _settingsCategory = SettingsCategory.General;

        // Generator settings
        private string _prompt = "a person throws a sharp right punch forward with energy";
        private uint _frames = 50; // 2.5 sec at 20fps
        private uint _steps = 20;
        private ulong _seed = 0;
        private bool _randomizeSeed = true;
        private float _textCfg = 2.0f;
        private bool _inPlace = true;
        public static TexMotionWindow Instance { get; private set; }
        public VrcSetupMode SetupMode => _setupMode;
        public VrcTargetLayer TargetLayer => _targetLayer;
        public ScriptableObject CustomTargetMenu => _customTargetMenu;
        public VrcMotionType MotionType => _motionType;
        public GameObject TargetAvatarObject => _targetAvatar;

        private VrcSetupMode _setupMode = VrcSetupMode.DirectVRCSDK;
        private VrcMotionType _motionType = VrcMotionType.OneShotEmote;
        private VrcTargetLayer _targetLayer = VrcTargetLayer.ActionLayer;
        private string _motionName = "Punch";
        private GameObject _targetAvatar;
        private ScriptableObject _customTargetMenu;

        // Hand & Face Assistance Settings
        private HandPoseType _handPose = HandPoseType.Fist;
        private FaceEmotionType _faceEmotion = FaceEmotionType.Angry;
        private float _emotionIntensity = 0.9f;

        // Video Motion State
        private string _videoPath = "";
        private UnityEngine.Object _videoAsset;
        private float _videoTargetFps = 30.0f;
        private bool _videoInPlace = true;
        private bool _videoSmoothing = true;
        private bool _videoFootLocking = true;
        private VideoOverlayMode _videoOverlayMode = VideoOverlayMode.Dual;
        private float _videoTrimStart = 0.0f;
        private float _videoTrimEnd = 0.0f;
        private float _videoMinConfidenceThreshold = 0.3f;
        private string _videoMotionName = "VideoMotion";
        private HandPoseType _videoHandPose = HandPoseType.NaturalRelaxed;
        private FaceEmotionType _videoFaceEmotion = FaceEmotionType.None;
        private float _videoEmotionIntensity = 0.8f;
        private bool _isVideoExtracting = false;
        private float _videoExtractionProgress = 0.0f;
        private string _videoExtractionStatus = "";
        private CancellationTokenSource _videoCts;
        private double _videoExtractionStartTime = 0;
        private string _videoExtractionLog = "";
        private string _lastExtractionLogLine = "";
        private bool _showVideoExtractionLog = false;
        private Vector2 _videoExtractionLogScroll = Vector2.zero;
        private readonly object _videoExtractionLogLock = new object();
        private volatile bool _videoExtractionLogDirty;
        private bool _uiLayoutInitialized = false;
        private bool _uiIsVideoExtracting = false;
        private float _uiVideoExtractionProgress = 0.0f;
        private string _uiVideoExtractionStatus = "";
        private string _uiLastExtractionLogLine = "";
        private bool _uiShowGpuHelpBox = false;
        private bool _uiShowExtractionLogLine = false;
        private bool _uiHasVideoMotionData = false;
        private VideoMotionData _currentVideoMotionData;
        private float _detectedVideoDuration = 0.0f;
        private int _detectedVideoWidth = 0;
        private int _detectedVideoHeight = 0;
        private float _detectedVideoFps = 0.0f;
        private PythonRuntimeInfo _detectedPythonInfo;
        private bool _isCheckingPython = false;
        private bool _isInstallingDependencies = false;
        private bool _isBuildingVideoVenv = false;
        private float _videoVenvProgress = 0.0f;
        private string _videoVenvStatus = "";
        private string _videoVenvError = "";
        private string _videoVenvLog = "";
        private volatile bool _videoVenvLogDirty;
        private bool _showVideoVenvLog = false;
        private readonly object _videoVenvLogLock = new object();
        private CancellationTokenSource _videoVenvCts;
        private bool _isVitPoseFeatureExporting = false;
        private float _vitPoseFeatureExportProgress = 0.0f;
        private string _vitPoseFeatureExportStatus = "";
        private string _vitPoseFeatureExportError = "";
        private CancellationTokenSource _vitPoseFeatureExportCts;
        private bool _autoVitPoseExportTriggeredForVideo = false;
        private string _lastAutoVitPoseVideoPath = "";
        private UvRuntimeInfo _detectedUvInfo;
        private bool _isCheckingUv = false;
        private bool _isInstallingUv = false;
        private float _uvInstallProgress = 0.0f;
        private string _uvInstallStatus = "";
        private string _uvInstallError = "";
        private CancellationTokenSource _uvInstallCts;
        private bool _isVideoModelDownloading = false;
        private float _videoModelDownloadProgress = 0.0f;
        private string _videoModelDownloadStatus = "";
        private string _videoModelDownloadError = "";
        private CancellationTokenSource _videoModelDownloadCts;
        private VideoPreflightReport _videoPreflightReport;
        private bool _videoPreflightRequested;
        // UI-only foldout state. Keep this separate from TexMotionSettings so
        // collapsing a panel never changes the extraction backend configuration.
        private const string FoldoutVideoExtractionAdvanced = "video.extraction.advanced";
        private const string FoldoutVideoAssistance = "video.assistance";
        private const string FoldoutSettingsPythonEnvironment = "settings.python.environment";
        private const string FoldoutSettingsModelRepository = "settings.model.repository";
        private const string FoldoutSettingsLocalStorage = "settings.model.storage";
        private const string FoldoutSettingsVideoModelCatalog = "settings.video.model-catalog";
        private const string FoldoutSettingsWham = "settings.video.wham";
        private const string FoldoutSettingsWhamOptional = "settings.video.wham.optional";
        private const string FoldoutSettingsVideoAdvancedAssets = "settings.video.advanced-assets";
        private const string FoldoutSettingsVideoInference = "settings.video.inference";
        private const string FoldoutSettingsWorkflow = "settings.workflow";
        private const string FoldoutPrefsPrefix = "TexMotionWindow.Foldout.";

        private bool _showVideoExtractionAdvanced;
        private bool _showVideoAssistance;
        private bool _showSettingsPythonEnvironment;
        private bool _showSettingsModelRepository;
        private bool _showSettingsLocalStorage;
        private bool _showVideoModelCatalog;
        private bool _showSettingsWham;
        private bool _showSettingsWhamOptional;
        private bool _showSettingsVideoAdvancedAssets;
        private bool _showSettingsVideoInference;
        private bool _showSettingsWorkflow;

        // Motion Library State
        private List<MotionLibraryItem> _libraryItems = new List<MotionLibraryItem>();
        private MotionLibraryItem _selectedLibraryItem;

        // 3D Preview Utility & Direct Transform Engine
        private PreviewRenderUtility _previewUtility;
        private GameObject _previewInstance;
        private GeneratedMotionData _currentMotionData;
        private AnimationClip _activePreviewClip;
        private bool _isPlayingPreview = false;
        private float _previewTime = 0f;
        private double _lastUpdateTime = 0;
        private Vector2 _previewDir = new Vector2(180f, 10f); // Camera orbit angles
        private float _previewDistance = 2.8f;
        private Vector3 _previewPivot = new Vector3(0, 1.0f, 0);

        // Cached preview bone transforms
        private readonly Dictionary<SmplxJoint, Transform> _previewBoneMap = new Dictionary<SmplxJoint, Transform>();
        private readonly Dictionary<SmplxJoint, Quaternion> _previewInitialRotations = new Dictionary<SmplxJoint, Quaternion>();
        private readonly Dictionary<int, float> _previewInitialFaceWeights = new Dictionary<int, float>();
        private Transform _previewHipsTransform;
        private Vector3 _previewInitialHipsPos;
        private SkinnedMeshRenderer _previewFaceRenderer;

        // Side-by-Side Video & Avatar Preview
        private enum PreviewDisplayMode
        {
            SideBySide,
            AvatarOnly,
            VideoOnly
        }
        private PreviewDisplayMode _previewDisplayMode = PreviewDisplayMode.SideBySide;
        private GameObject _videoPlayerGo;
        private VideoPlayer _videoPlayer;
        private RenderTexture _previewVideoTexture;
        private string _loadedVideoPath = "";
        private string _failedVideoPath = "";
        private bool _videoPlayerFailed = false;
        private double _videoPrepareStartTime = 0;
        private const double VIDEO_PREPARE_TIMEOUT_SEC = 20.0;
        private double _lastVideoSeekTime = 0;

        // Runtime state
        private bool _isGenerating = false;
        private bool _isDownloading = false;
        private float _overallProgress = 0f;
        private float _fileProgress = 0f;
        private string _statusMessage = "";
        private CancellationTokenSource _cts;

        // Inference Engine instance
        private KimodoInferenceEngine _engine;
        private Vector2 _scrollPos;

        // Linear-inspired IMGUI surfaces. These are cached because OnGUI is called
        // every editor frame; keeping the textures and styles here also lets the
        // existing controls retain their behavior while sharing one visual system.
        private bool _uiStylesInitialized;
        private Texture2D _uiCarbonTexture;
        private Texture2D _uiObsidianTexture;
        private Texture2D _uiGraphiteTexture;
        private Texture2D _uiSelectedTexture;
        private Texture2D _uiAcidTexture;

        private GUIStyle _cardStyle;
        private GUIStyle _nestedCardStyle;
        private GUIStyle _inlineCardStyle;
        private GUIStyle _selectedCardStyle;
        private GUIStyle _tabBarStyle;
        private GUIStyle _tabStyle;
        private GUIStyle _activeTabStyle;
        private GUIStyle _primaryActionStyle;
        private GUIStyle _headerTitleStyle;
        private GUIStyle _headerSubtitleStyle;
        private GUIStyle _sectionTitleStyle;
        private GUIStyle _dropZoneLabelStyle;
        private GUIStyle _badgeStyle;
        private GUIStyle _chipStyle;

        private static readonly Color UiVoid = new Color(8f / 255f, 9f / 255f, 10f / 255f);
        private static readonly Color UiCarbon = new Color(15f / 255f, 16f / 255f, 17f / 255f);
        private static readonly Color UiObsidian = new Color(22f / 255f, 23f / 255f, 24f / 255f);
        private static readonly Color UiGraphite = new Color(35f / 255f, 37f / 255f, 42f / 255f);
        private static readonly Color UiSmoke = new Color(56f / 255f, 59f / 255f, 63f / 255f);
        private static readonly Color UiFog = new Color(138f / 255f, 143f / 255f, 152f / 255f);
        private static readonly Color UiMist = new Color(208f / 255f, 214f / 255f, 224f / 255f);
        private static readonly Color UiPaper = Color.white;
        private static readonly Color UiAcid = new Color(228f / 255f, 242f / 255f, 34f / 255f);

        private static Texture2D CreateUiTexture(Color fill, Color border)
        {
            var texture = new Texture2D(3, 3, TextureFormat.RGBA32, false)
            {
                hideFlags = HideFlags.HideAndDontSave,
                filterMode = FilterMode.Point,
                wrapMode = TextureWrapMode.Clamp
            };

            for (int y = 0; y < 3; y++)
            {
                for (int x = 0; x < 3; x++)
                {
                    texture.SetPixel(x, y, x == 0 || x == 2 || y == 0 || y == 2 ? border : fill);
                }
            }

            texture.Apply();
            return texture;
        }

        private static void SetStyleBackground(GUIStyle style, Texture2D texture)
        {
            style.normal.background = texture;
            style.hover.background = texture;
            style.active.background = texture;
            style.focused.background = texture;
            style.onNormal.background = texture;
            style.onHover.background = texture;
            style.onActive.background = texture;
            style.onFocused.background = texture;
        }

        private static void SetStyleTextColor(GUIStyle style, Color color)
        {
            style.normal.textColor = color;
            style.hover.textColor = color;
            style.active.textColor = color;
            style.focused.textColor = color;
            style.onNormal.textColor = color;
            style.onHover.textColor = color;
            style.onActive.textColor = color;
            style.onFocused.textColor = color;
        }

        private void EnsureUiStyles()
        {
            if (_uiStylesInitialized) return;

            _uiCarbonTexture = CreateUiTexture(UiCarbon, UiGraphite);
            _uiObsidianTexture = CreateUiTexture(UiObsidian, UiGraphite);
            _uiGraphiteTexture = CreateUiTexture(UiGraphite, UiSmoke);
            _uiSelectedTexture = CreateUiTexture(new Color(28f / 255f, 31f / 255f, 37f / 255f), UiAcid);
            _uiAcidTexture = CreateUiTexture(UiAcid, UiAcid);

            _cardStyle = new GUIStyle(EditorStyles.helpBox)
            {
                padding = new RectOffset(12, 12, 10, 10),
                margin = new RectOffset(0, 0, 0, 0),
                border = new RectOffset(1, 1, 1, 1)
            };
            SetStyleBackground(_cardStyle, _uiCarbonTexture);
            SetStyleTextColor(_cardStyle, UiMist);

            _nestedCardStyle = new GUIStyle(EditorStyles.textArea)
            {
                padding = new RectOffset(8, 8, 7, 7),
                margin = new RectOffset(0, 0, 4, 4),
                border = new RectOffset(1, 1, 1, 1)
            };
            SetStyleBackground(_nestedCardStyle, _uiObsidianTexture);
            SetStyleTextColor(_nestedCardStyle, UiMist);

            _inlineCardStyle = new GUIStyle(EditorStyles.helpBox)
            {
                padding = new RectOffset(8, 8, 6, 6),
                margin = new RectOffset(0, 0, 0, 0),
                border = new RectOffset(1, 1, 1, 1)
            };
            SetStyleBackground(_inlineCardStyle, _uiCarbonTexture);
            SetStyleTextColor(_inlineCardStyle, UiMist);

            _selectedCardStyle = new GUIStyle(_cardStyle);
            SetStyleBackground(_selectedCardStyle, _uiSelectedTexture);

            _tabBarStyle = new GUIStyle(EditorStyles.toolbar)
            {
                fixedHeight = 38f,
                padding = new RectOffset(6, 6, 5, 5),
                margin = new RectOffset(0, 0, 0, 0),
                border = new RectOffset(1, 1, 1, 1)
            };
            SetStyleBackground(_tabBarStyle, _uiCarbonTexture);

            _tabStyle = new GUIStyle(EditorStyles.miniButton)
            {
                fixedHeight = 28f,
                margin = new RectOffset(2, 2, 0, 0),
                border = new RectOffset(1, 1, 1, 1),
                alignment = TextAnchor.MiddleCenter,
                fontStyle = FontStyle.Normal
            };
            SetStyleBackground(_tabStyle, _uiObsidianTexture);
            SetStyleTextColor(_tabStyle, UiMist);

            _activeTabStyle = new GUIStyle(_tabStyle);
            SetStyleBackground(_activeTabStyle, _uiAcidTexture);
            SetStyleTextColor(_activeTabStyle, UiVoid);

            _primaryActionStyle = new GUIStyle(EditorStyles.miniButton)
            {
                fixedHeight = 40f,
                margin = new RectOffset(0, 0, 2, 2),
                border = new RectOffset(1, 1, 1, 1),
                alignment = TextAnchor.MiddleCenter,
                fontStyle = FontStyle.Bold
            };
            SetStyleBackground(_primaryActionStyle, _uiAcidTexture);
            SetStyleTextColor(_primaryActionStyle, UiVoid);

            _headerTitleStyle = new GUIStyle(EditorStyles.boldLabel)
            {
                fontSize = 18,
                alignment = TextAnchor.MiddleLeft
            };
            SetStyleTextColor(_headerTitleStyle, UiPaper);

            _headerSubtitleStyle = new GUIStyle(EditorStyles.miniLabel)
            {
                alignment = TextAnchor.MiddleLeft
            };
            SetStyleTextColor(_headerSubtitleStyle, UiFog);

            _sectionTitleStyle = new GUIStyle(EditorStyles.boldLabel)
            {
                fontSize = 12,
                alignment = TextAnchor.MiddleLeft
            };
            SetStyleTextColor(_sectionTitleStyle, UiPaper);

            _dropZoneLabelStyle = new GUIStyle(EditorStyles.centeredGreyMiniLabel)
            {
                fontSize = 12,
                fontStyle = FontStyle.Bold,
                alignment = TextAnchor.MiddleCenter
            };
            SetStyleTextColor(_dropZoneLabelStyle, UiMist);

            _badgeStyle = new GUIStyle(EditorStyles.miniButton)
            {
                alignment = TextAnchor.MiddleCenter,
                fontStyle = FontStyle.Bold,
                border = new RectOffset(1, 1, 1, 1)
            };
            SetStyleBackground(_badgeStyle, _uiGraphiteTexture);
            SetStyleTextColor(_badgeStyle, UiMist);

            _chipStyle = new GUIStyle(_badgeStyle)
            {
                fixedHeight = 21f
            };
            _uiStylesInitialized = true;
        }

        private void CleanupUiStyles()
        {
            DestroyUiTexture(ref _uiCarbonTexture);
            DestroyUiTexture(ref _uiObsidianTexture);
            DestroyUiTexture(ref _uiGraphiteTexture);
            DestroyUiTexture(ref _uiSelectedTexture);
            DestroyUiTexture(ref _uiAcidTexture);

            _cardStyle = null;
            _nestedCardStyle = null;
            _inlineCardStyle = null;
            _selectedCardStyle = null;
            _tabBarStyle = null;
            _tabStyle = null;
            _activeTabStyle = null;
            _primaryActionStyle = null;
            _headerTitleStyle = null;
            _headerSubtitleStyle = null;
            _sectionTitleStyle = null;
            _dropZoneLabelStyle = null;
            _badgeStyle = null;
            _chipStyle = null;
            _uiStylesInitialized = false;
        }

        private static void DestroyUiTexture(ref Texture2D texture)
        {
            if (texture == null) return;
            UnityEngine.Object.DestroyImmediate(texture);
            texture = null;
        }

        private void BeginCard()
        {
            EnsureUiStyles();
            EditorGUILayout.BeginVertical(_cardStyle);
        }

        private void BeginNestedCard()
        {
            EnsureUiStyles();
            EditorGUILayout.BeginVertical(_nestedCardStyle);
        }

        private void BeginInlineCard()
        {
            EnsureUiStyles();
            EditorGUILayout.BeginHorizontal(_inlineCardStyle);
        }

        private void BeginNestedRow()
        {
            EnsureUiStyles();
            EditorGUILayout.BeginHorizontal(_nestedCardStyle);
        }

        private static bool GetFoldoutState(string key, bool defaultValue)
        {
            try
            {
                return EditorPrefs.GetBool(FoldoutPrefsPrefix + key, defaultValue);
            }
            catch
            {
                return defaultValue;
            }
        }

        private static void SetFoldoutState(string key, bool expanded)
        {
            try
            {
                EditorPrefs.SetBool(FoldoutPrefsPrefix + key, expanded);
            }
            catch
            {
                // Foldout persistence is a convenience and must never interrupt
                // drawing the window if the editor preferences backend is unavailable.
            }
        }

        private void LoadFoldoutStates()
        {
            _showVideoExtractionAdvanced = GetFoldoutState(FoldoutVideoExtractionAdvanced, false);
            _showVideoAssistance = GetFoldoutState(FoldoutVideoAssistance, false);
            // Keep the two provisioning surfaces discoverable on first open;
            // users can collapse them and the choice is persisted in EditorPrefs.
            _showSettingsPythonEnvironment = GetFoldoutState(FoldoutSettingsPythonEnvironment, true);
            _showSettingsModelRepository = GetFoldoutState(FoldoutSettingsModelRepository, false);
            _showSettingsLocalStorage = GetFoldoutState(FoldoutSettingsLocalStorage, false);
            _showVideoModelCatalog = GetFoldoutState(FoldoutSettingsVideoModelCatalog, true);
            _showSettingsWham = GetFoldoutState(FoldoutSettingsWham, true);
            _showSettingsWhamOptional = GetFoldoutState(FoldoutSettingsWhamOptional, false);
            _showSettingsVideoAdvancedAssets = GetFoldoutState(FoldoutSettingsVideoAdvancedAssets, false);
            _showSettingsVideoInference = GetFoldoutState(FoldoutSettingsVideoInference, false);
            _showSettingsWorkflow = GetFoldoutState(FoldoutSettingsWorkflow, false);
        }

        private bool DrawPersistedFoldoutHeader(string key, bool expanded, string title, string summary)
        {
            EnsureUiStyles();
            EditorGUILayout.BeginHorizontal(_nestedCardStyle);
            bool nextExpanded = EditorGUILayout.Foldout(expanded, title, true, EditorStyles.foldoutHeader);
            if (!string.IsNullOrEmpty(summary))
            {
                GUILayout.FlexibleSpace();
                EditorGUILayout.LabelField(summary, _headerSubtitleStyle, GUILayout.ExpandWidth(false));
            }
            EditorGUILayout.EndHorizontal();

            if (nextExpanded != expanded)
            {
                SetFoldoutState(key, nextExpanded);
            }
            return nextExpanded;
        }

        [MenuItem("Tools/TexMotion/Motion Studio", false, 0)]
        public static void ShowWindow()
        {
            var window = GetWindow<TexMotionWindow>("TexMotion");
            window.minSize = new Vector2(520, 780);
            window.Show();
        }

        /// <summary>Opens the shared Settings tab from an asset setup guide.</summary>
        public static void ShowSettingsWindow()
        {
            var window = GetWindow<TexMotionWindow>("TexMotion");
            window.minSize = new Vector2(520, 780);
            window._currentTab = Tab.Settings;
            window.Show();
            window.Focus();
        }

        private void OnEnable()
        {
            Instance = this;
            LoadFoldoutStates();
            if (TexMotionSettings.instance != null)
            {
                _videoOverlayMode = TexMotionSettings.instance.VideoOverlayMode;
            }
            AutoDetectAvatar();
            EnsureEngineLoaded();
            InitPreviewUtility();
            RefreshLibrary();
            CheckPythonEnvironmentAsync();
            CheckUvAsync();
            EditorApplication.update += OnEditorUpdate;
            AssemblyReloadEvents.beforeAssemblyReload += OnBeforeAssemblyReload;
        }

        private void OnDisable()
        {
            if (Instance == this) Instance = null;
            AssemblyReloadEvents.beforeAssemblyReload -= OnBeforeAssemblyReload;
            EditorApplication.update -= OnEditorUpdate;
            CleanupVideoPlayer();
            CleanupPreviewUtility();
            _cts?.Cancel();
            _videoCts?.Cancel();
            _videoVenvCts?.Cancel();
            _vitPoseFeatureExportCts?.Cancel();
            _uvInstallCts?.Cancel();
            _videoModelDownloadCts?.Cancel();
            _engine?.Dispose();
            _engine = null;
            CleanupUiStyles();
        }

        public void RefreshLibraryExternal()
        {
            RefreshLibrary();
        }

        private void OnDestroy()
        {
            CleanupVideoPlayer();
            CleanupPreviewUtility();
            CleanupUiStyles();
        }

        private void OnBeforeAssemblyReload()
        {
            CleanupVideoPlayer();
            CleanupPreviewUtility();
        }

        private void RefreshLibrary()
        {
            _libraryItems = MotionLibraryManager.ScanGeneratedMotions(_targetAvatar);
        }

        private void InitPreviewUtility()
        {
            if (_previewUtility == null)
            {
                _previewUtility = new PreviewRenderUtility();
                _previewUtility.cameraFieldOfView = 30f;
                _previewUtility.camera.nearClipPlane = 0.1f;
                _previewUtility.camera.farClipPlane = 100f;
                _previewUtility.camera.clearFlags = CameraClearFlags.SolidColor;
                _previewUtility.camera.backgroundColor = new Color(0.13f, 0.15f, 0.20f);
                _previewUtility.lights[0].intensity = 1.3f;
                _previewUtility.lights[0].transform.rotation = Quaternion.Euler(40f, 40f, 0);
                _previewUtility.lights[1].intensity = 0.9f;
                _previewUtility.lights[1].transform.rotation = Quaternion.Euler(140f, -40f, 0);
            }
        }

        private void CleanupPreviewUtility()
        {
            CleanupPreviewInstance();

            if (_previewUtility != null)
            {
                _previewUtility.Cleanup();
                _previewUtility = null;
            }
        }

        private void CleanupPreviewInstance(bool preservePlayState = false)
        {
            if (!preservePlayState)
            {
                _isPlayingPreview = false;
            }
            _previewBoneMap.Clear();
            _previewInitialRotations.Clear();
            _previewInitialFaceWeights.Clear();
            _previewHipsTransform = null;
            _previewFaceRenderer = null;
            _activePreviewClip = null;
            CleanupVideoPlayer();

            if (_previewInstance != null)
            {
                DestroyImmediate(_previewInstance);
                _previewInstance = null;
            }
        }

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

        private void SetupVideoPlayer(string videoPath)
        {
            if (string.IsNullOrEmpty(videoPath) || !File.Exists(videoPath))
            {
                return;
            }

            if (_videoPlayer != null && _loadedVideoPath == videoPath && _previewVideoTexture != null)
            {
                return;
            }

            CleanupVideoPlayer();

            try
            {
                _loadedVideoPath = videoPath;
                _videoPlayerFailed = false;
                _videoPrepareStartTime = EditorApplication.timeSinceStartup;

                var (texW, texH) = CalculateTargetTextureSize(_detectedVideoWidth, _detectedVideoHeight);

                _previewVideoTexture = new RenderTexture(texW, texH, 0, RenderTextureFormat.ARGB32)
                {
                    name = "TexMotion_PreviewVideoRT",
                    wrapMode = TextureWrapMode.Clamp,
                    filterMode = FilterMode.Bilinear
                };
                _previewVideoTexture.Create();

                _videoPlayerGo = new GameObject("TexMotion_VideoPreviewPlayer")
                {
                    hideFlags = HideFlags.HideAndDontSave
                };

                _videoPlayer = _videoPlayerGo.AddComponent<VideoPlayer>();
                _videoPlayer.playOnAwake = false;
                _videoPlayer.renderMode = VideoRenderMode.RenderTexture;
                _videoPlayer.aspectRatio = VideoAspectRatio.FitInside;
                _videoPlayer.targetTexture = _previewVideoTexture;
                _videoPlayer.source = VideoSource.Url;

                string fullPath = Path.GetFullPath(videoPath);
                // Directly pass local file path to avoid Windows Media Foundation URI scheme timeouts
                _videoPlayer.url = fullPath;

                _videoPlayer.audioOutputMode = VideoAudioOutputMode.None;
                _videoPlayer.isLooping = false; // Disable native looping to prevent Windows Media Foundation EOF lockups; handled deterministically via RestartVideoLoop
                _videoPlayer.skipOnDrop = true;

                _videoPlayer.errorReceived += (source, message) =>
                {
                    Debug.LogWarning($"[TexMotion] VideoPlayer playback error: {message}. Falling back to 3D avatar preview.");
                    _videoPlayerFailed = true;
                    _failedVideoPath = _loadedVideoPath;
                    Repaint();
                };

                _videoPlayer.prepareCompleted += (source) =>
                {
                    _videoPlayerFailed = false;
                    int vpW = (int)source.width;
                    int vpH = (int)source.height;
                    if (vpW > 0 && vpH > 0)
                    {
                        var (expectedW, expectedH) = CalculateTargetTextureSize(vpW, vpH);
                        if (_previewVideoTexture == null || _previewVideoTexture.width != expectedW || _previewVideoTexture.height != expectedH)
                        {
                            source.targetTexture = null;
                            if (_previewVideoTexture != null)
                            {
                                _previewVideoTexture.Release();
                                DestroyImmediate(_previewVideoTexture);
                            }
                            _previewVideoTexture = new RenderTexture(expectedW, expectedH, 0, RenderTextureFormat.ARGB32)
                            {
                                name = "TexMotion_PreviewVideoRT",
                                wrapMode = TextureWrapMode.Clamp,
                                filterMode = FilterMode.Bilinear
                            };
                            _previewVideoTexture.Create();
                            source.targetTexture = _previewVideoTexture;
                        }
                    }

                    float vidLen = Mathf.Max(0.01f, (float)source.length);
                    float initialTime = Mathf.Clamp(_previewTime, 0f, vidLen);
                    source.time = initialTime;

                    if (_isPlayingPreview)
                    {
                        source.Play();
                        _lastUpdateTime = EditorApplication.timeSinceStartup;
                    }
                    else
                    {
                        // Render initial frame then pause
                        source.Play();
                        source.Pause();
                    }
                    Repaint();
                };
                _videoPlayer.loopPointReached += (source) =>
                {
                    if (_isPlayingPreview)
                    {
                        RestartVideoLoop();
                    }
                    Repaint();
                };
                _videoPlayer.Prepare();
                EditorApplication.QueuePlayerLoopUpdate();
            }
            catch (Exception ex)
            {
                Debug.LogWarning($"[TexMotion] Could not initialize preview VideoPlayer: {ex.Message}");
                _videoPlayerFailed = true;
                CleanupVideoPlayer();
            }
        }

        private void CleanupVideoPlayer()
        {
            _videoPlayerFailed = false;
            _videoPrepareStartTime = 0;

            if (_videoPlayer != null)
            {
                try
                {
                    _videoPlayer.targetTexture = null;
                    if (_videoPlayer.isPlaying)
                    {
                        _videoPlayer.Stop();
                    }
                }
                catch {}
            }

            if (_videoPlayerGo != null)
            {
                try
                {
                    DestroyImmediate(_videoPlayerGo);
                }
                catch {}
                _videoPlayerGo = null;
            }
            _videoPlayer = null;

            if (_previewVideoTexture != null)
            {
                _previewVideoTexture.Release();
                DestroyImmediate(_previewVideoTexture);
                _previewVideoTexture = null;
            }

            _loadedVideoPath = "";
        }

        private void RestartVideoLoop()
        {
            if (_videoPlayer == null || !_videoPlayer.isPrepared || _videoPlayerFailed) return;

            double now = EditorApplication.timeSinceStartup;
            if (now - _lastVideoSeekTime < 0.20) return; // Prevent rapid-fire seek deadlocks
            _lastVideoSeekTime = now;

            try
            {
                _videoPlayer.time = 0.0;
                _videoPlayer.Play();
                EditorApplication.QueuePlayerLoopUpdate();
            }
            catch (Exception ex)
            {
                Debug.LogWarning($"[TexMotion] Video loop restart warning: {ex.Message}");
            }
        }

        private void SyncVideoPlayer(float time, bool isPlaying)
        {
            if (_videoPlayer == null || !_videoPlayer.isPrepared || _videoPlayerFailed) return;

            double now = EditorApplication.timeSinceStartup;
            float vidLen = Mathf.Max(0.01f, (float)_videoPlayer.length);
            float targetTime = Mathf.Clamp(time, 0f, vidLen);

            if (isPlaying)
            {
                // Only seek if time differs by more than 50ms, so Play() is never delayed or cancelled by a redundant seek
                if (Mathf.Abs((float)_videoPlayer.time - targetTime) > 0.05f)
                {
                    _videoPlayer.time = targetTime;
                }
                _videoPlayer.Play();
                _lastVideoSeekTime = now;
                EditorApplication.QueuePlayerLoopUpdate();
            }
            else
            {
                if (_videoPlayer.isPlaying)
                {
                    _videoPlayer.Pause();
                }
                if (Mathf.Abs((float)_videoPlayer.time - targetTime) > 0.02f)
                {
                    _videoPlayer.time = targetTime;
                    _videoPlayer.Play();
                    _videoPlayer.Pause();
                }
                _lastVideoSeekTime = now;
                EditorApplication.QueuePlayerLoopUpdate();
            }
        }

        private void SetupPreviewInstance(bool autoPlay = false)
        {
            if (_previewUtility == null) InitPreviewUtility();
            bool shouldPlay = autoPlay || _isPlayingPreview;
            CleanupPreviewInstance(preservePlayState: true);
            _isPlayingPreview = shouldPlay;

            // Initialize VideoPlayer if previewing VideoMotionData
            if (_currentMotionData is VideoMotionData vmd)
            {
                string videoToPlay = !string.IsNullOrEmpty(vmd.OverlayVideoPath) && File.Exists(vmd.OverlayVideoPath)
                    ? vmd.OverlayVideoPath
                    : (!string.IsNullOrEmpty(vmd.SourceVideoPath) && File.Exists(vmd.SourceVideoPath) ? vmd.SourceVideoPath : null);

                if (!string.IsNullOrEmpty(videoToPlay))
                {
                    SetupVideoPlayer(videoToPlay);
                }
            }

            if (_targetAvatar == null) return;

            _previewInstance = Instantiate(_targetAvatar, Vector3.zero, Quaternion.identity);
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

            var origAnim = _targetAvatar.GetComponent<Animator>();
            if (origAnim == null)
            {
                origAnim = _targetAvatar.GetComponentInChildren<Animator>(true);
            }
            if (origAnim != null && origAnim.isHuman)
            {
                Transform origHips = origAnim.GetBoneTransform(HumanBodyBones.Hips);
                if (origHips != null)
                {
                    string hipsPath = AnimationUtility.CalculateTransformPath(origHips, _targetAvatar.transform);
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
                        string bonePath = AnimationUtility.CalculateTransformPath(origBone, _targetAvatar.transform);
                        Transform cloneBone = _previewInstance.transform.Find(bonePath);
                        if (cloneBone != null)
                        {
                            _previewBoneMap[kvp.Key] = cloneBone;
                            _previewInitialRotations[kvp.Key] = cloneBone.localRotation;
                        }
                    }
                }
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
            ApplyStaticFingersAndFaceToPreview();

            _previewUtility.AddSingleGO(_previewInstance);
            ApplyCurrentPoseToPreview(0f);
        }

        private void ApplyStaticFingersAndFaceToPreview()
        {
            if (_previewInstance == null || _targetAvatar == null) return;

            // Avatars commonly keep the humanoid Animator on a child object rather
            // than on the root GameObject. Resolve the same animator used for body
            // preview so finger and face assistance cannot silently no-op.
            var origAnim = _targetAvatar.GetComponent<Animator>();
            if (origAnim == null)
            {
                origAnim = _targetAvatar.GetComponentInChildren<Animator>(true);
            }
            if (origAnim == null) return;

            HandPoseType activeHandPose = (_currentTab == Tab.VideoMotion) ? _videoHandPose : _handPose;
            FaceEmotionType activeFaceEmotion = (_currentTab == Tab.VideoMotion) ? _videoFaceEmotion : _faceEmotion;
            float activeIntensity = (_currentTab == Tab.VideoMotion) ? _videoEmotionIntensity : _emotionIntensity;

            ApplyFingersToPreviewInstance(origAnim, HandPosePresets.LeftFingerBones, activeHandPose, true);
            ApplyFingersToPreviewInstance(origAnim, HandPosePresets.RightFingerBones, activeHandPose, false);

            if (activeFaceEmotion == FaceEmotionType.AutoDetect)
            {
                activeFaceEmotion = (_currentTab == Tab.VideoMotion)
                    ? FaceEmotionType.None
                    : FaceEmotionHelper.InferEmotionFromPrompt(_prompt);
            }

            if (_previewFaceRenderer != null && _previewFaceRenderer.sharedMesh != null)
            {
                // Restore the avatar's original expression before applying the
                // selected preset. This makes switching to None reversible.
                foreach (var kvp in _previewInitialFaceWeights)
                {
                    _previewFaceRenderer.SetBlendShapeWeight(kvp.Key, kvp.Value);
                }

                if (activeFaceEmotion != FaceEmotionType.None)
                {
                    var weights = FaceEmotionHelper.GetBlendShapeWeightsForEmotion(_previewFaceRenderer, activeFaceEmotion, activeIntensity);
                    foreach (var kvp in weights)
                    {
                        _previewFaceRenderer.SetBlendShapeWeight(kvp.Key, kvp.Value);
                    }
                }
            }
        }

        private void ApplyFingersToPreviewInstance(Animator origAnim, HumanBodyBones[] fingerBones, HandPoseType pose, bool isLeft)
        {
            foreach (var boneType in fingerBones)
            {
                Transform origBone = origAnim.GetBoneTransform(boneType);
                if (origBone == null) continue;

                string path = AnimationUtility.CalculateTransformPath(origBone, _targetAvatar.transform);
                Transform cloneBone = _previewInstance.transform.Find(path);
                if (cloneBone != null)
                {
                    Quaternion rest = origBone.localRotation;
                    Quaternion offset = pose == HandPoseType.KeepFree
                        ? Quaternion.identity
                        : HandPosePresets.GetFingerLocalRotation(pose, boneType, isLeft);
                    cloneBone.localRotation = rest * offset;
                }
            }
        }

        private void OnEditorUpdate()
        {
            // Process output arrives on asynchronous stream threads. Defer
            // repainting to the editor update loop so the live setup log stays
            // responsive without calling Unity UI APIs off the main thread.
            if (_videoVenvLogDirty || _videoExtractionLogDirty)
            {
                _videoVenvLogDirty = false;
                _videoExtractionLogDirty = false;
                Repaint();
            }

            if (_isVideoExtracting)
            {
                Repaint();
            }

            double currentTime = EditorApplication.timeSinceStartup;
            double dt = currentTime - _lastUpdateTime;
            _lastUpdateTime = currentTime;

            // Clamp dt to avoid huge jumps on frame hiccups
            if (dt > 0.1) dt = 0.1;
            if (dt < 0) dt = 0;

            // Pump PlayerLoop in Edit Mode so VideoPlayer can complete Prepare() immediately
            if (_videoPlayer != null && !_videoPlayer.isPrepared && !_videoPlayerFailed)
            {
                EditorApplication.QueuePlayerLoopUpdate();
            }

            if (_isPlayingPreview && _previewInstance != null)
            {
                float duration = GetActivePreviewDuration();
                if (duration > 0)
                {
                    // If video is still preparing/buffering, hold at t=0 with timeout fallback
                    if (_currentMotionData is VideoMotionData && _videoPlayer != null && !_videoPlayer.isPrepared && !_videoPlayerFailed)
                    {
                        // Extraction owns the video path until it has produced a
                        // VideoMotionData result.  A previous preview player can
                        // still be observed for one editor tick while that async
                        // job is being torn down; never turn that transient state
                        // into a playback failure or report it as an extraction
                        // timeout.
                        if (_isVideoExtracting)
                        {
                            _previewTime = 0f;
                            ApplyCurrentPoseToPreview(0f);
                            EditorApplication.QueuePlayerLoopUpdate();
                            Repaint();
                            return;
                        }

                        double elapsed = currentTime - _videoPrepareStartTime;
                        if (elapsed > VIDEO_PREPARE_TIMEOUT_SEC)
                        {
                            Debug.LogWarning($"[TexMotion] Video preview preparation timed out ({elapsed:F1}s). Continuing with the 3D avatar preview; motion extraction is unaffected.");
                            _videoPlayerFailed = true;
                            _failedVideoPath = _loadedVideoPath;
                            if (_videoPlayer != null)
                            {
                                try { _videoPlayer.Stop(); } catch {}
                            }
                        }
                        else
                        {
                            _previewTime = 0f;
                            ApplyCurrentPoseToPreview(0f);
                            EditorApplication.QueuePlayerLoopUpdate();
                            Repaint();
                            return;
                        }
                    }

                    // Autonomous timeline advancement ensures playback never stops when window loses focus
                    _previewTime += (float)dt;
                    if (_previewTime >= duration)
                    {
                        _previewTime = 0f;
                        RestartVideoLoop();
                    }

                    // Unified loop recovery and gentle drift correction
                    if (_currentMotionData is VideoMotionData && _videoPlayer != null && _videoPlayer.isPrepared && !_videoPlayerFailed)
                    {
                        float vidLen = Mathf.Max(0.01f, (float)_videoPlayer.length);
                        float vTime = (float)_videoPlayer.time;

                        // Case 1: Video reached end or timeline wrapped to beginning -> restart cleanly via RestartVideoLoop
                        if (vTime >= vidLen - 0.08f || (vTime >= vidLen - 0.25f && _previewTime < 0.35f))
                        {
                            RestartVideoLoop();
                        }
                        // Case 2: Video stopped unexpectedly while timeline is active -> resume playback
                        else if (!_videoPlayer.isPlaying && (currentTime - _lastVideoSeekTime) > 0.35)
                        {
                            _videoPlayer.Play();
                            _lastVideoSeekTime = currentTime;
                        }
                        // Case 3: Gentle drift correction (rate-limited to max once per 0.6s)
                        else if (_videoPlayer.isPlaying && (currentTime - _lastVideoSeekTime) > 0.6)
                        {
                            float drift = Mathf.Abs(vTime - _previewTime);
                            if (drift > 0.35f)
                            {
                                _videoPlayer.time = Mathf.Clamp(_previewTime, 0f, vidLen);
                                _videoPlayer.Play();
                                _lastVideoSeekTime = currentTime;
                            }
                        }
                    }

                    ApplyCurrentPoseToPreview(_previewTime);
                    EditorApplication.QueuePlayerLoopUpdate(); // Keep PlayerLoop and VideoPlayer decoding even when unfocused!
                    Repaint();
                }
            }
            else if (!_isPlayingPreview && _videoPlayer != null && _videoPlayer.isPlaying)
            {
                _videoPlayer.Pause();
            }
        }

        private float GetActivePreviewDuration()
        {
            var motion = _currentMotionData ?? _currentVideoMotionData;
            if (motion != null && motion.Frames > 0)
            {
                if (motion is VideoMotionData vmd && vmd.Duration > 0f)
                {
                    return vmd.Duration;
                }
                return motion.Frames / Mathf.Max(1.0f, motion.FrameRate);
            }
            if (_activePreviewClip != null)
            {
                return _activePreviewClip.length;
            }
            return 0f;
        }

        private void ApplyCurrentPoseToPreview(float time)
        {
            if (_previewInstance == null) return;

            if (_currentMotionData == null && _currentVideoMotionData != null)
            {
                _currentMotionData = _currentVideoMotionData;
            }

            if (_currentMotionData != null && _currentMotionData.Frames > 0)
            {
                ApplyMotionPoseToPreview(time);
            }
            else if (_activePreviewClip != null)
            {
                AnimationMode.SampleAnimationClip(_previewInstance, _activePreviewClip, time);
            }
        }

        private void ApplyMotionPoseToPreview(float time)
        {
            if (_currentMotionData == null && _currentVideoMotionData != null)
            {
                _currentMotionData = _currentVideoMotionData;
            }
            if (_previewInstance == null || _currentMotionData == null || _currentMotionData.Frames == 0) return;

            int totalFrames = _currentMotionData.Frames;
            float totalDuration = GetActivePreviewDuration();
            if (totalDuration <= 0) return;

            float normTime = (time % totalDuration);
            int frameIdx = 0;
            int nextFrameIdx = 0;
            float t = 0f;

            if (_currentMotionData is VideoMotionData vmd && vmd.Timestamps != null && vmd.Timestamps.Length == totalFrames)
            {
                frameIdx = 0;
                while (frameIdx < totalFrames - 1 && vmd.Timestamps[frameIdx + 1] <= normTime)
                {
                    frameIdx++;
                }
                nextFrameIdx = Mathf.Min(frameIdx + 1, totalFrames - 1);
                float dt = vmd.Timestamps[nextFrameIdx] - vmd.Timestamps[frameIdx];
                t = dt > 0.0001f ? (normTime - vmd.Timestamps[frameIdx]) / dt : 0f;
            }
            else
            {
                float framePos = (normTime / totalDuration) * totalFrames;
                frameIdx = Mathf.Clamp((int)framePos, 0, totalFrames - 1);
                nextFrameIdx = Mathf.Min(frameIdx + 1, totalFrames - 1);
                t = framePos - frameIdx;
            }
            t = Mathf.Clamp01(t);

            bool effectiveInPlace = (_currentTab == Tab.VideoMotion) ? _videoInPlace : _inPlace;

            if (_previewHipsTransform != null && _currentMotionData.RootPositions != null && _currentMotionData.RootPositions.Length > 0)
            {
                Vector3 p0 = _currentMotionData.RootPositions[frameIdx];
                Vector3 p1 = _currentMotionData.RootPositions[nextFrameIdx];
                Vector3 p = Vector3.Lerp(p0, p1, t);

                Vector3 firstP = _currentMotionData.RootPositions[0];
                Vector3 delta = p - firstP;

                if (effectiveInPlace)
                {
                    _previewHipsTransform.localPosition = new Vector3(
                        _previewInitialHipsPos.x,
                        _previewInitialHipsPos.y + delta.y,
                        _previewInitialHipsPos.z
                    );
                }
                else
                {
                    _previewHipsTransform.localPosition = _previewInitialHipsPos + delta;
                }
            }

            foreach (var kvp in _previewBoneMap)
            {
                SmplxJoint joint = kvp.Key;
                Transform bone = kvp.Value;
                if (bone == null) continue;

                Quaternion q0 = _currentMotionData.LocalRotations[frameIdx, (int)joint];
                Quaternion q1 = _currentMotionData.LocalRotations[nextFrameIdx, (int)joint];
                Quaternion q = Quaternion.Slerp(q0, q1, t);

                Quaternion rest = _previewInitialRotations[joint];
                bone.localRotation = ConvertSmplRotationToUnity(q, joint, rest);
            }
        }

        private static Quaternion ConvertSmplRotationToUnity(Quaternion smplRot, SmplxJoint joint, Quaternion restPoseRot)
        {
            if (smplRot.w == 0 && smplRot.x == 0 && smplRot.y == 0 && smplRot.z == 0)
            {
                return restPoseRot;
            }
            Quaternion converted = new Quaternion(smplRot.x, -smplRot.y, -smplRot.z, smplRot.w);
            return restPoseRot * converted;
        }

        private void AutoDetectAvatar()
        {
            if (_targetAvatar != null) return;

            var animators = FindObjectsOfType<Animator>();
            foreach (var anim in animators)
            {
                if (anim.isHuman)
                {
                    _targetAvatar = anim.gameObject;
                    break;
                }
            }
        }

        private bool EnsureEngineLoaded()
        {
            if (_engine != null && _engine.IsModelLoaded) return true;

            var settings = TexMotionSettings.instance;
            if (!settings.AreModelsPresent()) return false;

            if (_engine == null) _engine = new KimodoInferenceEngine();

            string motionPath = settings.GetMotionModelPath();
            string textBundleDir = settings.GetTextBundleDirectory();

            bool success = _engine.LoadModel(motionPath, textBundleDir, null, KimodoDevice.Auto, out string error);
            if (!success)
            {
                _statusMessage = TexMotionLocalization.TrFormat("Failed to load model: {0}", error);
                return false;
            }

            _statusMessage = TexMotionLocalization.TrLiteral("Kimodo engine loaded & ready.");
            return true;
        }

        private void OnGUI()
        {
            if (Event.current.type == EventType.Layout || !_uiLayoutInitialized)
            {
                _uiIsVideoExtracting = _isVideoExtracting;
                _uiVideoExtractionProgress = _videoExtractionProgress;
                _uiVideoExtractionStatus = _videoExtractionStatus;
                lock (_videoExtractionLogLock)
                {
                    _uiLastExtractionLogLine = _lastExtractionLogLine;
                }
                _uiShowGpuHelpBox = _uiVideoExtractionProgress <= 0.06f;
                _uiShowExtractionLogLine = !string.IsNullOrEmpty(_uiLastExtractionLogLine);
                _uiHasVideoMotionData = _currentVideoMotionData != null;
                _uiLayoutInitialized = true;
            }

            EnsureUiStyles();
            EditorGUI.DrawRect(new Rect(0f, 0f, position.width, position.height), UiVoid);

            Color previousGuiColor = GUI.color;
            Color previousBackgroundColor = GUI.backgroundColor;
            Color previousContentColor = GUI.contentColor;
            GUI.color = Color.white;
            GUI.backgroundColor = Color.white;
            GUI.contentColor = UiMist;

            try
            {
                DrawHeader();
                DrawTabBar();

                if (_currentTab == Tab.Settings)
                {
                    DrawSettingsCategoryBar();
                }

                _scrollPos = EditorGUILayout.BeginScrollView(_scrollPos);
                EditorGUILayout.Space(6);

                if (_currentTab == Tab.Generator)
                {
                    DrawGeneratorTab();
                }
                else if (_currentTab == Tab.VideoMotion)
                {
                    DrawVideoTab();
                }
                else if (_currentTab == Tab.Library)
                {
                    DrawLibraryTab();
                }
                else
                {
                    DrawSettingsTab();
                }

                EditorGUILayout.EndScrollView();
                DrawFooterStatus();
            }
            finally
            {
                GUI.color = previousGuiColor;
                GUI.backgroundColor = previousBackgroundColor;
                GUI.contentColor = previousContentColor;
            }
        }

        private void DrawHeader()
        {
            EnsureUiStyles();
            var headerRect = EditorGUILayout.GetControlRect(false, 62);
            EditorGUI.DrawRect(headerRect, UiCarbon);
            EditorGUI.DrawRect(new Rect(headerRect.x, headerRect.yMax - 1f, headerRect.width, 1f), UiGraphite);
            EditorGUI.DrawRect(new Rect(headerRect.x + 12f, headerRect.y + 13f, 3f, 34f), UiAcid);

            GUI.Label(new Rect(headerRect.x + 24f, headerRect.y + 8f, headerRect.width - 36f, 24f), TexMotionLocalization.TrLiteral("✨ TexMotion Studio"), _headerTitleStyle);
            GUI.Label(new Rect(headerRect.x + 24f, headerRect.y + 32f, headerRect.width - 36f, 18f), TexMotionLocalization.TrLiteral("AI-Powered Text & Video-to-Motion for VRChat Avatars"), _headerSubtitleStyle);
        }

        private void DrawTabBar()
        {
            EnsureUiStyles();
            EditorGUILayout.Space(4);
            EditorGUILayout.BeginHorizontal(_tabBarStyle, GUILayout.Height(38f));

            // Keep all four destinations visible at the minimum editor width.  The
            // previous natural-width layout let the longer localized labels push the
            // last button outside the window, which made the tab bar look clipped and
            // also introduced a horizontal scroll bar in the content area.
            bool compact = position.width < 680f;
            string generatorLabel = compact
                ? TexMotionLocalization.TrLiteral("Generator")
                : TexMotionLocalization.TrLiteral("Motion Generator");
            string settingsLabel = compact
                ? TexMotionLocalization.TrLiteral("Settings")
                : TexMotionLocalization.Tr(TexMotionLocalization.Settings);
            // Account for the tab-bar padding and each button's horizontal margin
            // before dividing the available width between the four destinations.
            float tabWidth = Mathf.Max(64f, (position.width - 36f) / 4f);
            GUILayoutOption[] tabOptions =
            {
                GUILayout.Width(tabWidth),
                GUILayout.Height(28f)
            };

            if (GUILayout.Button("🎬 " + generatorLabel, _currentTab == Tab.Generator ? _activeTabStyle : _tabStyle, tabOptions))
            {
                _currentTab = Tab.Generator;
            }

            if (GUILayout.Button("🎥 " + TexMotionLocalization.Tr(TexMotionLocalization.VideoMotion), _currentTab == Tab.VideoMotion ? _activeTabStyle : _tabStyle, tabOptions))
            {
                _currentTab = Tab.VideoMotion;
            }

            if (GUILayout.Button("📚 " + TexMotionLocalization.TrLiteral("Library") + $" ({_libraryItems.Count})", _currentTab == Tab.Library ? _activeTabStyle : _tabStyle, tabOptions))
            {
                _currentTab = Tab.Library;
                RefreshLibrary();
            }

            if (GUILayout.Button("⚙️ " + settingsLabel, _currentTab == Tab.Settings ? _activeTabStyle : _tabStyle, tabOptions))
            {
                _currentTab = Tab.Settings;
            }

            EditorGUILayout.EndHorizontal();

            // Clear separator line between fixed tab bar and scrolling content area
            var sepRect = EditorGUILayout.GetControlRect(false, 1f);
            EditorGUI.DrawRect(sepRect, UiGraphite);
            EditorGUILayout.Space(2);
        }

        private void DrawSettingsCategoryBar()
        {
            EnsureUiStyles();
            EditorGUILayout.BeginHorizontal(_tabBarStyle, GUILayout.Height(34f));

            float tabWidth = Mathf.Max(64f, (position.width - 36f) / 3f);
            GUILayoutOption[] catOptions =
            {
                GUILayout.Width(tabWidth),
                GUILayout.Height(24f)
            };

            if (GUILayout.Button("🌐 " + TexMotionLocalization.TrLiteral("General"),
                _settingsCategory == SettingsCategory.General ? _activeTabStyle : _tabStyle, catOptions))
            {
                if (_settingsCategory != SettingsCategory.General)
                {
                    _settingsCategory = SettingsCategory.General;
                    _scrollPos = Vector2.zero;
                }
            }

            if (GUILayout.Button("🎬 " + TexMotionLocalization.TrLiteral("Motion Generation"),
                _settingsCategory == SettingsCategory.MotionGenerator ? _activeTabStyle : _tabStyle, catOptions))
            {
                if (_settingsCategory != SettingsCategory.MotionGenerator)
                {
                    _settingsCategory = SettingsCategory.MotionGenerator;
                    _scrollPos = Vector2.zero;
                }
            }

            if (GUILayout.Button("🎥 " + TexMotionLocalization.Tr(TexMotionLocalization.VideoMotion),
                _settingsCategory == SettingsCategory.VideoMotion ? _activeTabStyle : _tabStyle, catOptions))
            {
                if (_settingsCategory != SettingsCategory.VideoMotion)
                {
                    _settingsCategory = SettingsCategory.VideoMotion;
                    _scrollPos = Vector2.zero;
                }
            }

            EditorGUILayout.EndHorizontal();

            var sepRect = EditorGUILayout.GetControlRect(false, 1f);
            EditorGUI.DrawRect(sepRect, UiGraphite);
            EditorGUILayout.Space(2);
        }

        private void DrawGeneratorTab()
        {
            var settings = TexMotionSettings.instance;
            bool modelsReady = settings.AreModelsPresent();

            DrawModelStatusBanner(modelsReady);
            EditorGUILayout.Space(10);

            // 1. Avatar Selection
            BeginCard();
            EditorGUILayout.LabelField("1. " + TexMotionLocalization.TrLiteral("Target Avatar"), EditorStyles.boldLabel);
            var prevAvatar = _targetAvatar;
            _targetAvatar = (GameObject)EditorGUILayout.ObjectField(TexMotionLocalization.TrLiteral("Avatar GameObject"), _targetAvatar, typeof(GameObject), true);
            if (prevAvatar != _targetAvatar && _currentMotionData != null)
            {
                SetupPreviewInstance();
            }

            if (_targetAvatar == null)
            {
                EditorGUILayout.HelpBox(TexMotionLocalization.TrLiteral("Please select your VRChat Avatar from the Hierarchy."), MessageType.Info);
            }
            EditorGUILayout.EndVertical();

            EditorGUILayout.Space(10);

            // 2. Prompt Input
            BeginCard();
            EditorGUILayout.LabelField("2. " + TexMotionLocalization.TrLiteral("Motion Prompt"), EditorStyles.boldLabel);
            EditorGUILayout.LabelField(TexMotionLocalization.TrLiteral("Describe the animation in natural language:"), EditorStyles.miniLabel);
            _prompt = EditorGUILayout.TextArea(_prompt, GUILayout.Height(48));

            EditorGUILayout.BeginHorizontal();
            if (GUILayout.Button(TexMotionLocalization.TrLiteral("🥊 Punch"), EditorStyles.miniButton)) { _prompt = "a person throws a sharp right punch forward with energy"; _motionName = "Punch"; _handPose = HandPoseType.Fist; _faceEmotion = FaceEmotionType.Angry; }
            if (GUILayout.Button(TexMotionLocalization.TrLiteral("👋 Wave"), EditorStyles.miniButton)) { _prompt = "a person waves both hands warmly and smiles"; _motionName = "WaveHands"; _handPose = HandPoseType.NaturalRelaxed; _faceEmotion = FaceEmotionType.Smile; }
            if (GUILayout.Button(TexMotionLocalization.TrLiteral("🕺 Dance"), EditorStyles.miniButton)) { _prompt = "a dynamic hip hop dance routine"; _motionName = "Dance"; _handPose = HandPoseType.OpenPalm; }
            if (GUILayout.Button(TexMotionLocalization.TrLiteral("🧘 Stretch"), EditorStyles.miniButton)) { _prompt = "stretching arms overhead happily"; _motionName = "Stretch"; }
            EditorGUILayout.EndHorizontal();
            EditorGUILayout.EndVertical();

            EditorGUILayout.Space(10);

            // 3. Setup Target Configuration
            DrawSetupTargetSection(false);

            EditorGUILayout.Space(10);

            // 4. Assistance & Quality
            BeginCard();
            EditorGUILayout.LabelField("4. " + TexMotionLocalization.TrLiteral("Hand & Face Assistance"), EditorStyles.boldLabel);
            _handPose = (HandPoseType)EditorGUILayout.EnumPopup(TexMotionLocalization.TrLiteral("Hand Pose"), _handPose);
            _faceEmotion = (FaceEmotionType)EditorGUILayout.EnumPopup(TexMotionLocalization.TrLiteral("Face Emotion"), _faceEmotion);
            if (_faceEmotion != FaceEmotionType.None)
            {
                _emotionIntensity = EditorGUILayout.Slider(TexMotionLocalization.TrLiteral("Emotion Intensity"), _emotionIntensity, 0.1f, 1.0f);
            }

            EditorGUILayout.Space(5);
            EditorGUILayout.LabelField(TexMotionLocalization.TrLiteral("Generation Quality"), EditorStyles.boldLabel);
            _frames = (uint)EditorGUILayout.IntSlider(TexMotionLocalization.TrLiteral("Duration (Frames)"), (int)_frames, 20, 180);
            EditorGUILayout.LabelField(TexMotionLocalization.TrFormat("Approx. Duration: {0:F1} seconds", _frames / 20.0f), EditorStyles.miniLabel);

            _steps = (uint)EditorGUILayout.IntSlider(TexMotionLocalization.TrLiteral("Diffusion Steps"), (int)_steps, 5, 50);
            _textCfg = EditorGUILayout.Slider(TexMotionLocalization.TrLiteral("Text CFG Scale"), _textCfg, 1.0f, 5.0f);

            _randomizeSeed = EditorGUILayout.Toggle(TexMotionLocalization.TrLiteral("Randomize Seed"), _randomizeSeed);
            if (!_randomizeSeed)
            {
                _seed = (ulong)EditorGUILayout.LongField(TexMotionLocalization.TrLiteral("Seed"), (long)_seed);
            }
            EditorGUILayout.EndVertical();

            EditorGUILayout.Space(12);

            // 5. Generate Button
            GUI.enabled = !_isGenerating && !_isDownloading && modelsReady && _targetAvatar != null && !string.IsNullOrWhiteSpace(_prompt);
            GUI.backgroundColor = Color.white;

            if (GUILayout.Button(
                _isGenerating
                    ? "⏳ " + TexMotionLocalization.TrLiteral("Generating Motion...")
                    : "🚀 1. " + TexMotionLocalization.TrLiteral("Generate Motion Preview"),
                _primaryActionStyle,
                GUILayout.Height(40)))
            {
                StartGeneration();
            }

            GUI.backgroundColor = Color.white;
            GUI.enabled = true;

            // 6. Interactive 3D Preview Section
            if (_currentMotionData != null && _currentVideoMotionData == null)
            {
                EditorGUILayout.Space(15);
                DrawInteractive3DPreviewSection();
            }
        }

        private async void CheckPythonEnvironmentAsync()
        {
            if (_isCheckingPython) return;
            _isCheckingPython = true;
            try
            {
                string customPath = null;
                try
                {
                    customPath = TexMotionSettings.instance?.GetEffectiveVideoPythonExecutablePath();
                }
                catch {}

                _detectedPythonInfo = await VideoMotionJobRunner.DetectPythonRuntimeAsync(customPath);
            }
            catch (Exception ex)
            {
                Debug.LogWarning($"[TexMotion] Python detection warning: {ex.Message}");
            }
            finally
            {
                _isCheckingPython = false;
                TryTriggerAutoViTPoseFeatureExport();
                Repaint();
            }
        }

        private async void CheckUvAsync(string explicitPath = null)
        {
            if (_isCheckingUv || _isInstallingUv) return;
            _isCheckingUv = true;
            try
            {
                string configuredPath = explicitPath;
                if (string.IsNullOrWhiteSpace(configuredPath))
                {
                    try { configuredPath = TexMotionSettings.instance?.UvExecutablePath; } catch { }
                }

                _detectedUvInfo = await Task.Run(() => PythonEnvironmentManager.DetectUv(configuredPath));
            }
            catch (Exception ex)
            {
                _detectedUvInfo = new UvRuntimeInfo
                {
                    IsAvailable = false,
                    ErrorMessage = ex.Message
                };
            }
            finally
            {
                _isCheckingUv = false;
                Repaint();
            }
        }

        private async void InstallUvAsync(TexMotionSettings settings)
        {
            if (_isInstallingUv || settings == null) return;

            _isInstallingUv = true;
            _uvInstallProgress = 0.0f;
            _uvInstallStatus = TexMotionLocalization.TrLiteral("Starting the official uv installer...");
            _uvInstallError = "";
            _uvInstallCts = new CancellationTokenSource();
            _videoVenvLog = "";
            _showVideoVenvLog = true;
            _videoVenvLogDirty = true;
            CancellationToken cancellationToken = _uvInstallCts.Token;

            try
            {
                var progress = new Progress<PythonEnvironmentProgress>(p =>
                {
                    _uvInstallProgress = p.Progress;
                    _uvInstallStatus = TexMotionLocalization.TrLiteral(p.Message);
                    Repaint();
                });
                Action<string> log = line =>
                {
                    AppendVideoVenvLog(line + Environment.NewLine);
                };

                UvInstallResult result = await PythonEnvironmentManager.InstallUvAsync(
                    progress, log, cancellationToken);
                if (result == null || !result.IsUsable)
                {
                    string message = result?.ErrorMessage ?? TexMotionLocalization.TrLiteral("uv installation did not complete.");
                    string diagnostics = GetLastVideoVenvLogLines(result?.Diagnostics);
                    if (!string.IsNullOrEmpty(diagnostics)) message += "\n" + TexMotionLocalization.TrLiteral("Latest uv output:") + "\n" + diagnostics;
                    throw new InvalidOperationException(message);
                }

                _detectedUvInfo = result.Runtime;
                if (!string.IsNullOrWhiteSpace(result.ExecutablePath))
                {
                    settings.UvExecutablePath = result.ExecutablePath;
                    settings.Save();
                }
                _uvInstallProgress = 1.0f;
                _uvInstallStatus = TexMotionLocalization.TrFormat("uv is installed and ready: {0}", result.ExecutablePath);
                _statusMessage = TexMotionLocalization.TrLiteral("uv installed. You can now create the Video Motion venv.");
            }
            catch (OperationCanceledException)
            {
                _uvInstallStatus = TexMotionLocalization.TrLiteral("uv installation cancelled.");
                _uvInstallError = "";
                _statusMessage = _uvInstallStatus;
                AppendVideoVenvLog("[TexMotion] uv installation cancelled." + Environment.NewLine);
            }
            catch (Exception ex)
            {
                _uvInstallStatus = TexMotionLocalization.TrLiteral("uv installation failed.");
                _uvInstallError = ex.Message;
                _statusMessage = _uvInstallStatus;
                AppendVideoVenvLog("[TexMotion] uv installation failed: " + ex.Message + Environment.NewLine);
                Debug.LogError("[TexMotion] " + ex);
            }
            finally
            {
                _isInstallingUv = false;
                _uvInstallCts?.Dispose();
                _uvInstallCts = null;
                Repaint();
            }
        }

        private void CancelUvInstall()
        {
            if (!_isInstallingUv) return;
            _uvInstallStatus = TexMotionLocalization.TrLiteral("Cancelling uv installation...");
            _uvInstallCts?.Cancel();
            Repaint();
        }

        private async void InstallPythonDependenciesAsync()
        {
            if (_isInstallingDependencies || _isBuildingVideoVenv) return;

            var settings = TexMotionSettings.instance;
            if (settings != null && (settings.UseVideoVenv || string.IsNullOrEmpty(settings.CustomPythonExecutablePath)))
            {
                BuildVideoVenvAsync(settings);
                return;
            }

            _isInstallingDependencies = true;
            _statusMessage = TexMotionLocalization.TrLiteral("Installing Python dependencies (mediapipe, opencv-python, numpy, scipy)...");
            Repaint();

            try
            {
                string pyExe = _detectedPythonInfo?.ExecutablePath;
                if (string.IsNullOrEmpty(pyExe) || !File.Exists(pyExe))
                {
                    pyExe = "python";
                }

                bool installIntoVenv = _detectedPythonInfo != null && _detectedPythonInfo.IsVirtualEnv;
                string installArguments = "-m pip install mediapipe opencv-python numpy scipy onnxruntime" +
                    (installIntoVenv ? "" : " --user");

                await Task.Run(() =>
                {
                    var psi = new ProcessStartInfo
                    {
                        FileName = pyExe,
                        Arguments = installArguments,
                        UseShellExecute = false,
                        RedirectStandardOutput = true,
                        RedirectStandardError = true,
                        CreateNoWindow = true
                    };
                    using var proc = Process.Start(psi);
                    if (proc == null) throw new InvalidOperationException("Failed to start pip for " + pyExe + ".");
                    Task<string> stdoutTask = proc.StandardOutput.ReadToEndAsync();
                    Task<string> stderrTask = proc.StandardError.ReadToEndAsync();
                    proc.WaitForExit();
                    string stdout = stdoutTask.GetAwaiter().GetResult();
                    string stderr = stderrTask.GetAwaiter().GetResult();
                    if (proc.ExitCode != 0)
                    {
                        string diagnostics = string.IsNullOrWhiteSpace(stderr) ? stdout : stderr;
                        throw new InvalidOperationException(
                            "pip dependency installation failed (exit code " + proc.ExitCode + "): " + diagnostics.Trim());
                    }
                });

                _statusMessage = TexMotionLocalization.TrLiteral("Dependencies installed successfully. Rechecking Python...");
                CheckPythonEnvironmentAsync();
            }
            catch (Exception ex)
            {
                _statusMessage = TexMotionLocalization.TrFormat("Installation failed: {0}", ex.Message);
                Debug.LogError($"[TexMotion] {ex}");
            }
            finally
            {
                _isInstallingDependencies = false;
                Repaint();
            }
        }

        private void AppendVideoVenvLog(string text)
        {
            if (string.IsNullOrEmpty(text)) return;
            const int maxLogLength = 12000;
            lock (_videoVenvLogLock)
            {
                _videoVenvLog += text;
                if (_videoVenvLog.Length > maxLogLength)
                {
                    _videoVenvLog = _videoVenvLog.Substring(_videoVenvLog.Length - maxLogLength);
                }
            }
            _videoVenvLogDirty = true;
        }

        private string GetVideoVenvLogSnapshot()
        {
            lock (_videoVenvLogLock) return _videoVenvLog;
        }

        private static string GetVideoVenvLogPreview(string log)
        {
            const int maxPreviewLength = 2400;
            if (string.IsNullOrEmpty(log) || log.Length <= maxPreviewLength) return log;

            int start = log.Length - maxPreviewLength;
            int firstLineBreak = log.IndexOf('\n', start);
            if (firstLineBreak >= 0 && firstLineBreak + 1 < log.Length) start = firstLineBreak + 1;
            return "[... showing the end of the log; Copy contains the full output ...]" + Environment.NewLine +
                log.Substring(start);
        }

        private static string GetLastVideoVenvLogLines(string log)
        {
            if (string.IsNullOrWhiteSpace(log)) return null;
            string[] lines = log.Split(new[] { '\r', '\n' }, StringSplitOptions.RemoveEmptyEntries);
            var selected = new List<string>(3);
            for (int i = lines.Length - 1; i >= 0 && selected.Count < 3; i--)
            {
                string line = lines[i].Trim();
                if (!string.IsNullOrEmpty(line) && IsVideoVenvDiagnosticLine(line)) selected.Add(line);
            }
            if (selected.Count == 0)
            {
                for (int i = lines.Length - 1; i >= 0 && selected.Count < 3; i--)
                {
                    string line = lines[i].Trim();
                    if (!string.IsNullOrEmpty(line)) selected.Add(line);
                }
            }
            if (selected.Count == 0) return null;
            selected.Reverse();

            var summary = new StringBuilder();
            for (int i = 0; i < selected.Count; i++)
            {
                if (i > 0) summary.AppendLine();
                summary.Append(selected[i]);
            }
            const int maxSummaryLength = 900;
            if (summary.Length > maxSummaryLength)
            {
                summary.Length = maxSummaryLength;
                summary.Append("...");
            }
            return summary.ToString();
        }

        private static bool IsVideoVenvDiagnosticLine(string line)
        {
            string lower = line.ToLowerInvariant();
            return lower.Contains("error") ||
                lower.Contains("failed") ||
                lower.Contains("denied") ||
                lower.Contains("permission") ||
                lower.Contains("not found") ||
                lower.Contains("no matching") ||
                lower.Contains("unable") ||
                lower.Contains("could not") ||
                lower.Contains("caused by") ||
                lower.Contains("timeout") ||
                lower.Contains("forbidden") ||
                lower.Contains("connect");
        }

        private async void BuildVideoVenvAsync(TexMotionSettings settings)
        {
            if (_isBuildingVideoVenv || settings == null) return;

            _isBuildingVideoVenv = true;
            _videoVenvProgress = 0.0f;
            _videoVenvError = "";
            _videoVenvLog = "";
            _showVideoVenvLog = true;
            _videoVenvLogDirty = true;
            _videoVenvCts = new CancellationTokenSource();
            CancellationToken cancellationToken = _videoVenvCts.Token;

            try
            {
                PythonDependencyProfile profile = settings.VideoPythonEnvironmentProfile == PythonDependencyProfile.PyTorch
                    ? PythonDependencyProfile.PyTorch
                    : PythonDependencyProfile.Lightweight;
                var options = PythonEnvironmentOptions.CreateDefault(profile);
                options.ProjectRoot = settings.GetProjectRootDirectory();
                options.EnvironmentDirectory = settings.GetEffectiveVideoVenvPath();
                options.RequirementsPath = PythonEnvironmentManager.FindRequirementsPath(options.ProjectRoot);
                options.PyTorchRequirementsPath = PythonEnvironmentManager.FindPyTorchRequirementsPath(options.ProjectRoot);
                options.AllowMissingOptionalProfile = false;

                var progress = new Progress<PythonEnvironmentProgress>(p =>
                {
                    _videoVenvProgress = p.Progress;
                    _videoVenvStatus = TexMotionLocalization.TrLiteral(p.Message);
                    Repaint();
                });
                Action<string> log = line =>
                {
                    AppendVideoVenvLog(line + Environment.NewLine);
                };
                PythonEnvironmentSetupResult result = await PythonEnvironmentManager.EnsureEnvironmentAsync(
                    options, progress, log, cancellationToken, settings.UvExecutablePath);
                if (result == null || !result.IsUsable)
                {
                    string message = result?.ErrorMessage ?? TexMotionLocalization.TrLiteral("Video Motion environment setup did not complete.");
                    if (result != null && result.ExitCode != 0) message += " (uv exit code " + result.ExitCode + ").";
                    string diagnostics = GetLastVideoVenvLogLines(result?.Diagnostics ?? GetVideoVenvLogSnapshot());
                    if (!string.IsNullOrEmpty(diagnostics)) message += "\n" + TexMotionLocalization.TrLiteral("Latest uv output:") + "\n" + diagnostics;
                    throw new InvalidOperationException(message);
                }

                settings.CustomPythonExecutablePath = result.PythonExecutable;
                settings.UseVideoVenv = true;
                settings.Save();
                _videoVenvProgress = 1.0f;
                _videoVenvStatus = TexMotionLocalization.TrFormat("Video Motion environment ready ({0}).",
                    profile == PythonDependencyProfile.PyTorch ? TexMotionLocalization.TrLiteral("PyTorch Quality") : TexMotionLocalization.TrLiteral("Lightweight ONNX"));
                _statusMessage = TexMotionLocalization.TrLiteral("Video Motion venv ready. Re-probing Python capabilities...");
                CheckPythonEnvironmentAsync();
            }
            catch (OperationCanceledException)
            {
                _videoVenvStatus = TexMotionLocalization.TrLiteral("Video Motion environment setup cancelled.");
                _videoVenvError = "";
                _statusMessage = _videoVenvStatus;
                AppendVideoVenvLog("[TexMotion] Video Motion environment setup cancelled." + Environment.NewLine);
            }
            catch (Exception ex)
            {
                _videoVenvStatus = TexMotionLocalization.TrLiteral("Video Motion environment setup failed.");
                _videoVenvError = ex.Message;
                _statusMessage = _videoVenvStatus;
                AppendVideoVenvLog("[TexMotion] Video Motion environment setup failed: " + ex.Message + Environment.NewLine);
                Debug.LogError("[TexMotion] " + ex);
            }
            finally
            {
                _isBuildingVideoVenv = false;
                _videoVenvCts?.Dispose();
                _videoVenvCts = null;
                Repaint();
            }
        }

        private void CancelVideoVenvSetup()
        {
            if (!_isBuildingVideoVenv) return;
            _videoVenvStatus = TexMotionLocalization.TrLiteral("Cancelling Video Motion environment setup...");
            _videoVenvCts?.Cancel();
            Repaint();
        }

        private bool CheckRtmposeModelExists()
        {
            try
            {
                var settings = TexMotionSettings.instance;
                if (settings != null)
                {
                    if (IsUsableVideoModelPath(settings.GetRTMPoseModelPathIfPresent())) return true;
                    // An invalid explicit selection should not hide a valid
                    // default model downloaded into the configured directory.
                    if (IsUsableVideoModelPath(settings.GetVideoModelPath("rtmpose-m.onnx"))) return true;
                }

                // Preserve compatibility with package-local and legacy caches.
                string packageModel = Path.GetFullPath(Path.Combine(Application.dataPath, "..", "Editor", "Video", "models", "rtmpose-m.onnx"));
                if (IsUsableVideoModelPath(packageModel)) return true;
                string appData = Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData);
                if (IsUsableVideoModelPath(Path.Combine(appData, "TexMotion", "Models", "rtmpose-m.onnx"))) return true;
                string userHome = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);
                if (IsUsableVideoModelPath(Path.Combine(userHome, ".cache", "texmotion", "models", "rtmpose-m.onnx"))) return true;
                string cwdModel = Path.Combine(Directory.GetCurrentDirectory(), "Editor", "Video", "models", "rtmpose-m.onnx");
                if (IsUsableVideoModelPath(cwdModel)) return true;
                string packageCacheModel = Path.Combine(Directory.GetCurrentDirectory(), "Packages", "com.k0ta0uchi.texmotion", "Editor", "Video", "models", "rtmpose-m.onnx");
                if (IsUsableVideoModelPath(packageCacheModel)) return true;
            }
            catch {}
            return false;
        }

        private static bool IsUsableVideoModelPath(string path)
        {
            if (string.IsNullOrWhiteSpace(path)) return false;
            try { return File.Exists(path) && new FileInfo(path).Length > 1024 * 1024; }
            catch { return false; }
        }

        private static bool IsExistingAdapterFile(string path)
        {
            if (string.IsNullOrWhiteSpace(path)) return false;
            try { return File.Exists(path) && new FileInfo(path).Length > 0; }
            catch { return false; }
        }

        private static bool LooksLikeAdapterFileReference(string value)
        {
            if (string.IsNullOrWhiteSpace(value)) return false;
            string trimmed = value.Trim();
            return Path.IsPathRooted(trimmed) ||
                   trimmed.EndsWith(".py", StringComparison.OrdinalIgnoreCase) ||
                   trimmed.IndexOf(Path.DirectorySeparatorChar) >= 0 ||
                   trimmed.IndexOf(Path.AltDirectorySeparatorChar) >= 0;
        }

        private static bool IsVideoQualityBackend(VideoPoseBackend backend)
        {
            return backend == VideoPoseBackend.PyTorch || IsWhamBackend(backend) ||
                   backend == VideoPoseBackend.HMR2 || backend == VideoPoseBackend.HybrIK;
        }

        private static bool IsWhamBackend(VideoPoseBackend backend)
        {
            return backend == VideoPoseBackend.WHAM || backend == VideoPoseBackend.WHAMMediaPipe;
        }

        private static string GetVideoQualityBackendDescription(VideoPoseBackend backend)
        {
            switch (backend)
            {
                case VideoPoseBackend.WHAM: return TexMotionLocalization.TrLiteral("Official WHAM temporal 3D runner.");
                case VideoPoseBackend.WHAMMediaPipe: return TexMotionLocalization.TrLiteral("WHAM + MediaPipe (WHAM primary, MediaPipe auxiliary).");
                case VideoPoseBackend.HMR2: return TexMotionLocalization.TrLiteral("HMR2 / 4D-Humans image-conditioned SMPL backend.");
                case VideoPoseBackend.HybrIK: return TexMotionLocalization.TrLiteral("HybrIK analytical/neural IK backend.");
                default: return TexMotionLocalization.TrLiteral("Generic TorchScript temporal 3D backend.");
            }
        }

        private void StartVideoModelDownload(string modelId, TexMotionSettings settings)
        {
            if (_isVideoModelDownloading || settings == null) return;
            _isVideoModelDownloading = true;
            _videoModelDownloadProgress = 0.0f;
            _videoModelDownloadStatus = TexMotionLocalization.TrLiteral("Preparing Video2Motion model download...");
            _videoModelDownloadError = "";
            _videoModelDownloadCts = new CancellationTokenSource();
            _statusMessage = TexMotionLocalization.TrLiteral("Downloading Video2Motion model...");
            Repaint();

            _ = DownloadVideoModelsAsync(settings, modelId, false, _videoModelDownloadCts.Token);
        }

        private void StartWhamSetupDownload(TexMotionSettings settings)
        {
            if (_isVideoModelDownloading || settings == null) return;
            _isVideoModelDownloading = true;
            _videoModelDownloadProgress = 0.0f;
            _videoModelDownloadStatus = TexMotionLocalization.TrLiteral("Preparing required WHAM downloads...");
            _videoModelDownloadError = "";
            _videoModelDownloadCts = new CancellationTokenSource();
            _statusMessage = TexMotionLocalization.TrLiteral("Preparing the official WHAM setup...");
            Repaint();
            _ = DownloadWhamSetupAsync(settings, _videoModelDownloadCts.Token);
        }

        private async Task DownloadWhamSetupAsync(
            TexMotionSettings settings,
            CancellationToken cancellationToken)
        {
            try
            {
                VideoWhamSetupSummary summary = VideoModelCatalog.GetWhamSetupSummary(
                    settings,
                    settings.VideoBackend);
                var downloadable = new List<VideoModelDefinition>();
                if (summary.RequiredAssets != null)
                {
                    for (int i = 0; i < summary.RequiredAssets.Count; i++)
                    {
                        VideoAssetStatusInfo status = summary.RequiredAssets[i];
                        VideoModelDefinition asset = status == null ? null : VideoModelCatalog.Find(status.Id);
                        if (status == null || status.IsReady || asset == null || !asset.CanDownload) continue;
                        bool duplicate = false;
                        for (int j = 0; j < downloadable.Count; j++)
                        {
                            if (string.Equals(downloadable[j].Id, asset.Id, StringComparison.OrdinalIgnoreCase))
                            {
                                duplicate = true;
                                break;
                            }
                        }
                        if (!duplicate) downloadable.Add(asset);
                    }
                }

                if (downloadable.Count == 0)
                {
                    _videoModelDownloadProgress = 1.0f;
                    _videoModelDownloadStatus = TexMotionLocalization.TrLiteral(
                        "No automatic WHAM download is available. Open the manual setup guide for the remaining items.");
                    _videoModelDownloadError = "";
                    return;
                }

                for (int i = 0; i < downloadable.Count; i++)
                {
                    cancellationToken.ThrowIfCancellationRequested();
                    VideoModelDefinition asset = downloadable[i];
                    var itemProgress = new Progress<DownloadProgress>(p =>
                    {
                        float baseProgress = (float)i / downloadable.Count;
                        _videoModelDownloadProgress = Mathf.Clamp01(baseProgress + p.OverallProgress / downloadable.Count);
                        _videoModelDownloadStatus = TexMotionLocalization.TrFormat(
                            "WHAM setup: {0} ({1}/{2})",
                            p.StatusText,
                            i + 1,
                            downloadable.Count);
                        Repaint();
                    });
                    await VideoModelDownloader.DownloadAsync(settings, asset.Id, itemProgress, cancellationToken);
                }

                settings.Save();
                _videoModelDownloadProgress = 1.0f;
                _videoModelDownloadStatus = TexMotionLocalization.TrLiteral(
                    "Downloadable WHAM setup assets are ready. Complete any manual items, then run Preflight.");
                _videoModelDownloadError = "";
                _statusMessage = _videoModelDownloadStatus;
            }
            catch (OperationCanceledException)
            {
                _videoModelDownloadStatus = TexMotionLocalization.TrLiteral("WHAM setup download cancelled.");
                _videoModelDownloadError = "";
            }
            catch (Exception ex)
            {
                _videoModelDownloadStatus = TexMotionLocalization.TrLiteral("WHAM setup download failed.");
                _videoModelDownloadError = ex.Message;
                _statusMessage = _videoModelDownloadStatus;
            }
            finally
            {
                _isVideoModelDownloading = false;
                _videoModelDownloadCts?.Dispose();
                _videoModelDownloadCts = null;
                Repaint();
            }
        }

        private void StartAllVideoModelDownload(TexMotionSettings settings)
        {
            if (_isVideoModelDownloading || settings == null) return;
            _isVideoModelDownloading = true;
            _videoModelDownloadProgress = 0.0f;
            _videoModelDownloadStatus = TexMotionLocalization.TrLiteral("Preparing all Video2Motion model downloads...");
            _videoModelDownloadError = "";
            _videoModelDownloadCts = new CancellationTokenSource();
            _statusMessage = TexMotionLocalization.TrLiteral("Downloading all Video2Motion models...");
            Repaint();

            _ = DownloadVideoModelsAsync(settings, null, true, _videoModelDownloadCts.Token);
        }

        private async Task DownloadVideoModelsAsync(
            TexMotionSettings settings,
            string modelId,
            bool all,
            CancellationToken cancellationToken)
        {
            try
            {
                var progress = new Progress<DownloadProgress>(p =>
                {
                    _videoModelDownloadProgress = p.OverallProgress;
                    _videoModelDownloadStatus = TexMotionLocalization.TrLiteral(p.StatusText);
                    Repaint();
                });
                if (all)
                {
                    await VideoModelDownloader.DownloadAllAsync(settings, progress, cancellationToken);
                }
                else
                {
                    await VideoModelDownloader.DownloadAsync(settings, modelId, progress, cancellationToken);
                }

                settings.Save();
                _videoModelDownloadProgress = 1.0f;
                VideoModelDefinition downloadedModel = all ? null : VideoModelCatalog.Find(modelId);
                _videoModelDownloadStatus = all
                    ? TexMotionLocalization.TrLiteral("All Video2Motion models and downloadable WHAM companions are ready.")
                    : downloadedModel != null && downloadedModel.Kind == VideoModelKind.WHAM
                        ? downloadedModel.IsWhamCompanionAsset
                            ? TexMotionLocalization.TrFormat("WHAM companion is ready: {0}", downloadedModel.DisplayName)
                            : TexMotionLocalization.TrFormat("Video2Motion model and WHAM adapter are ready: {0}", downloadedModel.DisplayName)
                        : TexMotionLocalization.TrFormat("Video2Motion model is ready: {0}", downloadedModel?.DisplayName ?? modelId);
                _statusMessage = _videoModelDownloadStatus;
            }
            catch (OperationCanceledException)
            {
                _videoModelDownloadStatus = TexMotionLocalization.TrLiteral("Video2Motion model download cancelled.");
                _videoModelDownloadError = "";
                _statusMessage = _videoModelDownloadStatus;
            }
            catch (Exception ex)
            {
                _videoModelDownloadStatus = TexMotionLocalization.TrLiteral("Video2Motion model download failed.");
                _videoModelDownloadError = ex.Message;
                _statusMessage = _videoModelDownloadStatus;
                Debug.LogError("[TexMotion] " + ex);
            }
            finally
            {
                _isVideoModelDownloading = false;
                _videoModelDownloadCts?.Dispose();
                _videoModelDownloadCts = null;
                Repaint();
            }
        }

        private void CancelVideoModelDownload()
        {
            if (!_isVideoModelDownloading) return;
            _videoModelDownloadStatus = TexMotionLocalization.TrLiteral("Cancelling Video2Motion model download...");
            _videoModelDownloadCts?.Cancel();
            Repaint();
        }

        private void DrawPythonEnvironmentBanner()
        {
            BeginInlineCard();

            if (_isCheckingPython)
            {
                GUI.color = new Color(0.7f, 0.7f, 0.7f);
                EditorGUILayout.LabelField("⏳ " + TexMotionLocalization.TrLiteral("Checking Python environment..."), EditorStyles.miniLabel);
                GUI.color = Color.white;
            }
            else if (_isInstallingDependencies)
            {
                GUI.color = new Color(0.35f, 0.75f, 0.95f);
                EditorGUILayout.LabelField("📦 " + TexMotionLocalization.TrLiteral("Installing Python dependencies via pip..."), EditorStyles.boldLabel);
                GUI.color = Color.white;
            }
            else if (_detectedPythonInfo != null && _detectedPythonInfo.IsFullyConfigured)
            {
                GUI.color = new Color(0.3f, 0.95f, 0.4f);
                EditorGUILayout.LabelField(
                    "● " + TexMotionLocalization.TrFormat("Video Pose Dependencies Ready ({0})", _detectedPythonInfo.Version),
                    EditorStyles.boldLabel,
                    GUILayout.Width(280));
                GUI.color = Color.white;
                EditorGUILayout.LabelField(_detectedPythonInfo.ExecutablePath, EditorStyles.miniLabel);
            }
            else
            {
                GUI.color = new Color(1.0f, 0.45f, 0.35f);
                EditorGUILayout.LabelField("⚠️ " + TexMotionLocalization.TrLiteral("Python Dependencies Missing"), EditorStyles.boldLabel, GUILayout.Width(220));
                GUI.color = Color.white;

                if (GUILayout.Button("📦 " + TexMotionLocalization.TrLiteral("Install Dependencies"), GUILayout.Height(22)))
                {
                    InstallPythonDependenciesAsync();
                }

                if (GUILayout.Button("🔄 " + TexMotionLocalization.TrLiteral("Re-check"), GUILayout.Width(75), GUILayout.Height(22)))
                {
                    CheckPythonEnvironmentAsync();
                }
            }

            EditorGUILayout.EndHorizontal();
        }

        private void DrawVideoTab()
        {
            // 0. Python Environment Banner
            DrawPythonEnvironmentBanner();
            DrawHybridPipelineStatusCard();
            EditorGUILayout.Space(6);

            // 1. Target Avatar Selector
            BeginCard();
            EditorGUILayout.LabelField("1. " + TexMotionLocalization.TrLiteral("Target Avatar"), EditorStyles.boldLabel);
            var prevAvatar = _targetAvatar;
            _targetAvatar = (GameObject)EditorGUILayout.ObjectField(TexMotionLocalization.TrLiteral("Avatar GameObject"), _targetAvatar, typeof(GameObject), true);
            if (prevAvatar != _targetAvatar && _currentMotionData != null)
            {
                SetupPreviewInstance();
            }

            if (_targetAvatar == null)
            {
                EditorGUILayout.HelpBox(TexMotionLocalization.TrLiteral("Please select your VRChat Avatar from the Hierarchy."), MessageType.Info);
            }
            EditorGUILayout.EndVertical();

            EditorGUILayout.Space(10);

            // 2. Video File Selector (Browse + Drag & Drop)
            BeginCard();
            EditorGUILayout.LabelField("2. " + TexMotionLocalization.TrLiteral("Video Source File"), EditorStyles.boldLabel);

            // Drag & Drop Box Area
            Rect dropRect = GUILayoutUtility.GetRect(0f, 65f, GUILayout.ExpandWidth(true));
            GUI.Box(dropRect, GUIContent.none, _nestedCardStyle);

            string dropText = string.IsNullOrEmpty(_videoPath)
                ? "📂 " + TexMotionLocalization.TrLiteral("Drag & Drop Video File Here (.mp4, .mov, .webm, .avi)\n— or click 'Browse Video' below —")
                : TexMotionLocalization.TrFormat("Selected: {0}\n(Drag & drop another video file to replace)", Path.GetFileName(_videoPath));

            GUI.Label(dropRect, dropText, _dropZoneLabelStyle);

            // Handle Drag & Drop Events
            Event evt = Event.current;
            switch (evt.type)
            {
                case EventType.DragUpdated:
                case EventType.DragPerform:
                    if (!dropRect.Contains(evt.mousePosition)) break;

                    DragAndDrop.visualMode = DragAndDropVisualMode.Copy;

                    if (evt.type == EventType.DragPerform)
                    {
                        DragAndDrop.AcceptDrag();
                        if (DragAndDrop.paths != null && DragAndDrop.paths.Length > 0)
                        {
                            SetVideoSourcePath(DragAndDrop.paths[0]);
                        }
                        else if (DragAndDrop.objectReferences != null && DragAndDrop.objectReferences.Length > 0)
                        {
                            var obj = DragAndDrop.objectReferences[0];
                            string assetPath = AssetDatabase.GetAssetPath(obj);
                            if (!string.IsNullOrEmpty(assetPath))
                            {
                                SetVideoSourcePath(Path.GetFullPath(assetPath));
                            }
                        }
                        evt.Use();
                        Repaint();
                    }
                    break;
            }

            EditorGUILayout.Space(6);

            // Video Object Field and Browse Button
            EditorGUILayout.BeginHorizontal();
            var prevAsset = _videoAsset;
            _videoAsset = EditorGUILayout.ObjectField(TexMotionLocalization.TrLiteral("Video Asset / Clip"), _videoAsset, typeof(UnityEngine.Object), false);
            if (_videoAsset != prevAsset && _videoAsset != null)
            {
                string assetPath = AssetDatabase.GetAssetPath(_videoAsset);
                if (!string.IsNullOrEmpty(assetPath))
                {
                    SetVideoSourcePath(Path.GetFullPath(assetPath));
                }
            }

            if (GUILayout.Button("📂 " + TexMotionLocalization.TrLiteral("Browse Video..."), GUILayout.Width(130), GUILayout.Height(20)))
            {
                string picked = EditorUtility.OpenFilePanel(TexMotionLocalization.TrLiteral("Select Video for Motion Extraction"), "", "mp4,mov,webm,avi,mkv");
                if (!string.IsNullOrEmpty(picked))
                {
                    SetVideoSourcePath(picked);
                }
            }
            EditorGUILayout.EndHorizontal();

            // Display File Path
            if (!string.IsNullOrEmpty(_videoPath))
            {
                EditorGUILayout.BeginHorizontal();
                EditorGUILayout.LabelField(TexMotionLocalization.TrLiteral("File Path:"), GUILayout.Width(65));
                EditorGUILayout.SelectableLabel(_videoPath, EditorStyles.textField, GUILayout.Height(18));
                if (GUILayout.Button("✕", GUILayout.Width(24), GUILayout.Height(18)))
                {
                    _videoPath = "";
                    _videoAsset = null;
                    _detectedVideoDuration = 0f;
                }
                EditorGUILayout.EndHorizontal();
            }
            else
            {
                EditorGUILayout.BeginHorizontal();
                EditorGUILayout.HelpBox(TexMotionLocalization.TrLiteral("No video file selected. You can select an MP4 video or test with a synthetic walking motion."), MessageType.Info);
                if (GUILayout.Button("🧪 " + TexMotionLocalization.TrLiteral("Use Test Motion"), GUILayout.Width(135), GUILayout.Height(30)))
                {
                    _videoPath = "synthetic_motion_test";
                    _videoMotionName = "SyntheticWalk";
                    _detectedVideoDuration = 2.0f;
                    _detectedVideoFps = 30.0f;
                    _videoTrimEnd = 2.0f;
                }
                EditorGUILayout.EndHorizontal();
            }

            EditorGUILayout.EndVertical();

            EditorGUILayout.Space(10);

            // 3. Video Info & Extraction Settings Card
            BeginCard();
            EditorGUILayout.LabelField("3. " + TexMotionLocalization.TrLiteral("Extraction Settings & Video Info"), EditorStyles.boldLabel);

            if (!string.IsNullOrEmpty(_videoPath))
            {
                BeginNestedCard();
                bool isSynthetic = _videoPath == "synthetic_motion_test";
                string sourceName = isSynthetic
                    ? TexMotionLocalization.TrLiteral("Kinematic Synthetic Generator")
                    : Path.GetFileName(_videoPath);
                EditorGUILayout.LabelField("📹 " + TexMotionLocalization.TrFormat("Source: {0}", sourceName), EditorStyles.boldLabel);

                if (!isSynthetic && File.Exists(_videoPath))
                {
                    var fileInfo = new FileInfo(_videoPath);
                    float sizeMb = fileInfo.Length / (1024f * 1024f);
                    string metaStr = TexMotionLocalization.TrFormat("Size: {0:F1} MB", sizeMb);
                    if (_detectedVideoDuration > 0f)
                    {
                        metaStr += TexMotionLocalization.TrFormat(
                            " | Duration: {0:F2}s | Resolution: {1}x{2} | FPS: {3:F1}",
                            _detectedVideoDuration,
                            _detectedVideoWidth,
                            _detectedVideoHeight,
                            _detectedVideoFps);
                    }
                    EditorGUILayout.LabelField(metaStr, EditorStyles.miniLabel);
                }
                EditorGUILayout.EndVertical();
                EditorGUILayout.Space(6);
            }

            _videoTargetFps = EditorGUILayout.Slider(TexMotionLocalization.TrLiteral("Target FPS"), _videoTargetFps, 15f, 60f);
            _videoTargetFps = Mathf.Round(_videoTargetFps);

            // Keep the same input sanitation active while the advanced panel is
            // collapsed so extraction receives the same values as before.
            _videoTrimStart = Mathf.Max(0f, _videoTrimStart);
            _videoTrimEnd = Mathf.Max(0f, _videoTrimEnd);
            if (_detectedVideoDuration > 0f && _videoTrimEnd > _detectedVideoDuration)
            {
                _videoTrimEnd = _detectedVideoDuration;
            }

            _showVideoExtractionAdvanced = DrawPersistedFoldoutHeader(
                FoldoutVideoExtractionAdvanced,
                _showVideoExtractionAdvanced,
                "⚙️ " + TexMotionLocalization.TrLiteral("Advanced Extraction"),
                TexMotionLocalization.TrLiteral("Trim • quality • backend"));
            if (_showVideoExtractionAdvanced)
            {
                EditorGUILayout.Space(4);
                EditorGUILayout.LabelField("✂️ " + TexMotionLocalization.TrLiteral("Video Trimming (Seconds)"), _sectionTitleStyle);
                EditorGUILayout.BeginHorizontal();
                _videoTrimStart = EditorGUILayout.FloatField(TexMotionLocalization.TrLiteral("Start Time (s)"), _videoTrimStart);
                _videoTrimStart = Mathf.Max(0f, _videoTrimStart);

                _videoTrimEnd = EditorGUILayout.FloatField(TexMotionLocalization.TrLiteral("End Time (0=full)"), _videoTrimEnd);
                _videoTrimEnd = Mathf.Max(0f, _videoTrimEnd);
                EditorGUILayout.EndHorizontal();

                if (_videoTrimEnd > 0f && _videoTrimEnd > _videoTrimStart)
                {
                    float trimmedDuration = _videoTrimEnd - _videoTrimStart;
                    int estFrames = Mathf.RoundToInt(trimmedDuration * _videoTargetFps);
                    EditorGUILayout.LabelField(
                        TexMotionLocalization.TrFormat(
                            "Trimmed Segment: {0:F2}s to {1:F2}s ({2:F2}s, ~{3} frames)",
                            _videoTrimStart,
                            _videoTrimEnd,
                            trimmedDuration,
                            estFrames),
                        EditorStyles.miniLabel);
                }
                else
                {
                    EditorGUILayout.LabelField(TexMotionLocalization.TrLiteral("Trimming: Full video duration will be processed."), EditorStyles.miniLabel);
                }

                EditorGUILayout.Space(6);
                EditorGUILayout.LabelField("⚙️ " + TexMotionLocalization.TrLiteral("Extraction Quality & Processing"), _sectionTitleStyle);
                _videoInPlace = EditorGUILayout.Toggle(TexMotionLocalization.TrLiteral("In-Place Root (Keep hips centered)"), _videoInPlace);
                _videoSmoothing = EditorGUILayout.Toggle(TexMotionLocalization.TrLiteral("Temporal Smoothing (Savitzky-Golay)"), _videoSmoothing);
                _videoFootLocking = EditorGUILayout.Toggle(TexMotionLocalization.TrLiteral("Foot Locking & Floor Snapping"), _videoFootLocking);
                _videoMinConfidenceThreshold = EditorGUILayout.Slider(TexMotionLocalization.TrLiteral("Min Confidence Cutoff"), _videoMinConfidenceThreshold, 0.0f, 0.8f);

                EditorGUI.BeginChangeCheck();
                _videoOverlayMode = (VideoOverlayMode)EditorGUILayout.EnumPopup(
                    new GUIContent(
                        TexMotionLocalization.TrLiteral("Overlay Mode"),
                        TexMotionLocalization.TrLiteral("Dual: 2D detector guide (subtle) + 3D pose projection (vibrant neon). Recommended.\nPose3D: WHAM 3D pose projection only.\nTracking2D: 2D keypoint tracking only.")
                    ),
                    _videoOverlayMode
                );
                if (EditorGUI.EndChangeCheck() && TexMotionSettings.instance != null)
                {
                    TexMotionSettings.instance.VideoOverlayMode = _videoOverlayMode;
                    TexMotionSettings.instance.Save();
                }

            }

            // Backend selection is a primary extraction control and must stay
            // visible even when the optional trim/quality panel is collapsed.
            EditorGUILayout.Space(6);
            DrawVideoBackendSummaryCard();
            EditorGUILayout.Space(6);
            DrawVideoViTPoseFeatureSection();

            EditorGUILayout.EndVertical();

            EditorGUILayout.Space(10);

            // 4. Hand & Face Assistance
            BeginCard();
            _showVideoAssistance = DrawPersistedFoldoutHeader(
                FoldoutVideoAssistance,
                _showVideoAssistance,
                "4. " + TexMotionLocalization.TrLiteral("Hand & Face Assistance"),
                TexMotionLocalization.TrLiteral("Hand pose • emotion"));
            if (_showVideoAssistance)
            {
                EditorGUI.BeginChangeCheck();
                _videoHandPose = (HandPoseType)EditorGUILayout.EnumPopup(TexMotionLocalization.TrLiteral("Hand Pose"), _videoHandPose);
                _videoFaceEmotion = (FaceEmotionType)EditorGUILayout.EnumPopup(TexMotionLocalization.TrLiteral("Face Emotion"), _videoFaceEmotion);
                if (_videoFaceEmotion != FaceEmotionType.None)
                {
                    _videoEmotionIntensity = EditorGUILayout.Slider(TexMotionLocalization.TrLiteral("Emotion Intensity"), _videoEmotionIntensity, 0.1f, 1.0f);
                }
                if (EditorGUI.EndChangeCheck())
                {
                    // Keep the Studio preview in sync before extraction as well as
                    // after it. The clone is deliberately static, so changing the
                    // dropdown must reapply finger rotations/blendshapes directly.
                    ApplyStaticFingersAndFaceToPreview();
                    Repaint();
                }
            }
            EditorGUILayout.EndVertical();

            EditorGUILayout.Space(10);

            // 5. Setup Target Section
            DrawSetupTargetSection(true);

            EditorGUILayout.Space(12);

            // 6. Extraction Action Button & Real-time Progress Bar
            bool pythonCanStart = _detectedPythonInfo != null && _detectedPythonInfo.IsAvailable && _detectedPythonInfo.HasNumPy;
            if (!pythonCanStart && _detectedPythonInfo != null && _detectedPythonInfo.IsAvailable)
            {
                EditorGUILayout.HelpBox(
                    TexMotionLocalization.TrFormat(
                        "Video extraction is disabled until the selected Python runtime has NumPy. {0}",
                        _detectedPythonInfo.GetSummary()),
                    MessageType.Warning);
            }
            bool canExtract = !_uiIsVideoExtracting && !_isGenerating && !_isDownloading && !_isVideoModelDownloading &&
                !_isVitPoseFeatureExporting &&
                pythonCanStart &&
                _targetAvatar != null &&
                (!string.IsNullOrEmpty(_videoPath) && (File.Exists(_videoPath) || _videoPath == "synthetic_motion_test"));

            GUI.enabled = canExtract;
            GUI.backgroundColor = Color.white;

            string extractButtonLabel = _uiIsVideoExtracting
                ? "⏳ " + TexMotionLocalization.TrLiteral("Extracting 3D Humanoid Motion...")
                : _isVitPoseFeatureExporting
                    ? "⏳ " + TexMotionLocalization.TrLiteral("Extracting ViTPose features...")
                    : "🎥 1. " + TexMotionLocalization.TrLiteral("Extract Motion from Video");

            if (GUILayout.Button(
                extractButtonLabel,
                _primaryActionStyle,
                GUILayout.Height(42)))
            {
                StartVideoExtraction();
            }

            GUI.backgroundColor = Color.white;
            GUI.enabled = true;

            if (_uiIsVideoExtracting)
            {
                EditorGUILayout.Space(8);
                BeginCard();

                EditorGUILayout.BeginHorizontal();
                double elapsed = _videoExtractionStartTime > 0 ? (EditorApplication.timeSinceStartup - _videoExtractionStartTime) : 0;
                int elapsedMin = (int)(elapsed / 60);
                int elapsedSec = (int)(elapsed % 60);
                string[] spinners = new[] { "⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏" };
                int spinnerIndex = (int)(EditorApplication.timeSinceStartup * 8.0) % spinners.Length;
                string sp = spinners[spinnerIndex];

                Color prevColor = GUI.color;
                GUI.color = new Color(0.35f, 0.85f, 1.0f);
                EditorGUILayout.LabelField(
                    $"{sp} " + TexMotionLocalization.TrLiteral("Extraction in Progress"),
                    EditorStyles.boldLabel);
                GUI.color = new Color(0.8f, 0.95f, 1.0f);
                EditorGUILayout.LabelField(
                    $"⏱ {elapsedMin:D2}:{elapsedSec:D2}",
                    EditorStyles.boldLabel,
                    GUILayout.Width(75));
                GUI.color = prevColor;
                EditorGUILayout.EndHorizontal();

                EditorGUILayout.Space(4);
                EditorGUI.ProgressBar(EditorGUILayout.GetControlRect(false, 22), _uiVideoExtractionProgress, _uiVideoExtractionStatus);

                if (_uiShowGpuHelpBox)
                {
                    EditorGUILayout.Space(4);
                    EditorGUILayout.HelpBox(
                        "⚡ " + TexMotionLocalization.TrLiteral("GPU is performing deep temporal sequence optimization across all video frames. This calculation is actively running on your GPU and takes several minutes for high-accuracy motion. Please keep this window open."),
                        MessageType.Info);
                }

                if (_uiShowExtractionLogLine)
                {
                    EditorGUILayout.Space(2);
                    EditorGUILayout.LabelField("💬 " + _uiLastExtractionLogLine, EditorStyles.miniLabel);
                }

                EditorGUILayout.Space(6);
                EditorGUILayout.BeginHorizontal();
                _showVideoExtractionLog = EditorGUILayout.ToggleLeft(
                    TexMotionLocalization.TrLiteral("Show Live Process Log"),
                    _showVideoExtractionLog,
                    GUILayout.Width(170));

                GUILayout.FlexibleSpace();

                if (GUILayout.Button("🛑 " + TexMotionLocalization.TrLiteral("Cancel Extraction"), GUILayout.Width(150), GUILayout.Height(24)))
                {
                    _videoCts?.Cancel();
                }
                EditorGUILayout.EndHorizontal();

                if (_showVideoExtractionLog)
                {
                    EditorGUILayout.Space(4);
                    BeginNestedCard();
                    EditorGUILayout.LabelField(TexMotionLocalization.TrLiteral("Live Process Output:"), EditorStyles.miniLabel);
                    string logSnapshot;
                    lock (_videoExtractionLogLock)
                    {
                        logSnapshot = _videoExtractionLog;
                    }
                    _videoExtractionLogScroll = EditorGUILayout.BeginScrollView(_videoExtractionLogScroll, GUILayout.Height(110));
                    EditorGUILayout.TextArea(string.IsNullOrEmpty(logSnapshot) ? TexMotionLocalization.TrLiteral("Waiting for process output...") : logSnapshot, EditorStyles.miniLabel);
                    EditorGUILayout.EndScrollView();
                    EditorGUILayout.EndVertical();
                }

                EditorGUILayout.EndVertical();
            }

            // 7. Video Motion Result Card & Interactive 3D Preview Viewport
            if (_uiHasVideoMotionData)
            {
                EditorGUILayout.Space(15);
                DrawVideoMotionResultCard();
                EditorGUILayout.Space(8);
                DrawInteractive3DPreviewSection();
            }
        }

        /// <summary>
        /// Shows and selects the persisted backend while keeping model paths, adapters,
        /// and device details on the Settings tab.
        /// </summary>
        private void DrawVideoBackendSummaryCard()
        {
            var settings = TexMotionSettings.instance;
            VideoPoseBackend backend = settings != null ? settings.VideoBackend : VideoPoseBackend.Auto;
            string backendLabel = GetVideoBackendDisplayName(backend);
            bool compactLayout = position.width < 680f;

            BeginCard();
            EditorGUI.BeginChangeCheck();
            VideoPoseBackend selectedBackend;
            if (compactLayout)
            {
                // Stack the title and popup on narrow windows so the enum control
                // cannot force the whole Video Motion tab wider than the viewport.
                EditorGUILayout.LabelField("🤖 " + TexMotionLocalization.TrLiteral("Pose Estimation Backend"), EditorStyles.boldLabel);
                GUI.color = new Color(0.55f, 0.85f, 1.0f);
                selectedBackend = (VideoPoseBackend)EditorGUILayout.EnumPopup(
                    new GUIContent(backendLabel, TexMotionLocalization.TrLiteral("Choose the pose backend for this extraction. Detailed model paths remain in Settings.")),
                    backend);
                GUI.color = Color.white;
            }
            else
            {
                EditorGUILayout.BeginHorizontal();
                EditorGUILayout.LabelField("🤖 " + TexMotionLocalization.TrLiteral("Pose Estimation Backend"), EditorStyles.boldLabel);
                GUI.color = new Color(0.55f, 0.85f, 1.0f);
                selectedBackend = (VideoPoseBackend)EditorGUILayout.EnumPopup(
                    new GUIContent(backendLabel, TexMotionLocalization.TrLiteral("Choose the pose backend for this extraction. Detailed model paths remain in Settings.")),
                    backend);
                GUI.color = Color.white;
                EditorGUILayout.EndHorizontal();
            }
            bool backendChanged = EditorGUI.EndChangeCheck();
            if (settings != null && backendChanged && selectedBackend != backend)
            {
                settings.VideoBackend = selectedBackend;
                settings.Save();
                backend = selectedBackend;
                _statusMessage = TexMotionLocalization.TrFormat("Video Motion backend set to {0}.", GetVideoBackendDisplayName(selectedBackend));
            }

            if (IsWhamBackend(backend))
            {
                VideoWhamSetupSummary summary = settings == null
                    ? null
                    : VideoModelCatalog.GetWhamSetupSummary(settings, backend);
                EditorGUILayout.LabelField(
                    TexMotionLocalization.TrFormat(
                        "{0}  •  {1}",
                        GetVideoQualityBackendDescription(backend),
                        summary == null ? TexMotionLocalization.TrLiteral("Setup unavailable") : GetWhamSetupSummaryLabel(summary)),
                    EditorStyles.miniLabel);
            }
            else if (backend == VideoPoseBackend.RTMPose)
            {
                EditorGUILayout.LabelField(
                    TexMotionLocalization.TrLiteral("RTMPose 2D ONNX. Configure the model in Settings."),
                    EditorStyles.miniLabel);
            }
            else if (backend == VideoPoseBackend.MediaPipe)
            {
                EditorGUILayout.LabelField(
                    TexMotionLocalization.TrLiteral("MediaPipe Pose Landmarker. Configure model quality in Settings."),
                    EditorStyles.miniLabel);
            }
            else
            {
                EditorGUILayout.LabelField(TexMotionLocalization.TrLiteral("Automatic selection uses local RTMPose/MediaPipe assets when available."), EditorStyles.miniLabel);
            }

            EditorGUILayout.LabelField(
                TexMotionLocalization.TrLiteral("Choose the backend here; configure model details, adapters, and device in Settings → Video2Motion."),
                EditorStyles.miniLabel);
            if (GUILayout.Button(TexMotionLocalization.TrLiteral("Open Settings"), EditorStyles.miniButton, GUILayout.Height(22f)))
            {
                _currentTab = Tab.Settings;
            }
            EditorGUILayout.EndVertical();
        }

        private static string GetVideoBackendDisplayName(VideoPoseBackend backend)
        {
            switch (backend)
            {
                case VideoPoseBackend.MediaPipe: return "MediaPipe";
                case VideoPoseBackend.RTMPose: return "RTMPose";
                case VideoPoseBackend.PyTorch: return "PyTorch Quality";
                case VideoPoseBackend.WHAM: return "Official WHAM";
                case VideoPoseBackend.WHAMMediaPipe: return "WHAM + MediaPipe";
                case VideoPoseBackend.HMR2: return "HMR2 / 4D-Humans";
                case VideoPoseBackend.HybrIK: return "HybrIK";
                default: return "Auto";
            }
        }

        /// <summary>
        /// Presents the supported local hybrid inference path in one place so that users
        /// can distinguish the lightweight Windows pipeline from optional research stacks.
        /// The card intentionally remains visible while the one-time local setup is pending:
        /// model provisioning may require a download, but extraction itself is offline.
        /// </summary>
        private void DrawHybridPipelineStatusCard()
        {
            var settings = TexMotionSettings.instance;
            VideoPoseBackend selectedBackend = settings != null ? settings.VideoBackend : VideoPoseBackend.Auto;
            bool pythonReady = _detectedPythonInfo != null && _detectedPythonInfo.IsAvailable;
            bool mediaPipeReady = _detectedPythonInfo != null && _detectedPythonInfo.HasMediaPipe;
            bool onnxReady = _detectedPythonInfo != null && _detectedPythonInfo.HasOnnxRuntime;
            bool rtmposeModelReady = CheckRtmposeModelExists();
            bool mediaPipeFallbackSelected = selectedBackend == VideoPoseBackend.MediaPipe;
            bool qualityBackendSelected = IsVideoQualityBackend(selectedBackend);
            VideoPreflightReport qualityPreflight = qualityBackendSelected && settings != null
                ? VideoModelCatalog.GetPreflightReport(settings, selectedBackend)
                : null;
            // Backend readiness comes from the same role-aware report shown in
            // Settings. In particular, a WHAM adapter must not satisfy HMR2 or
            // HybrIK, and an HMR2 checkpoint alone is not a runnable runtime.
            bool qualityAssetReady = qualityBackendSelected && qualityPreflight != null && qualityPreflight.IsReady;
            bool localAssetsReady = qualityBackendSelected
                ? pythonReady && qualityAssetReady
                : pythonReady && mediaPipeReady && onnxReady && rtmposeModelReady;

            BeginCard();

            EditorGUILayout.BeginHorizontal();
            EditorGUILayout.LabelField("⚡ " + TexMotionLocalization.TrLiteral("Hybrid Video-to-Motion"), _sectionTitleStyle);
            Color previousBadgeColor = GUI.color;
            GUI.color = new Color(0.45f, 1.0f, 0.65f);
            GUILayout.Label("● " + TexMotionLocalization.TrLiteral("100% Offline"), _badgeStyle, GUILayout.Width(105), GUILayout.Height(22));
            GUI.color = previousBadgeColor;
            EditorGUILayout.EndHorizontal();

            EditorGUILayout.LabelField(
                qualityBackendSelected
                    ? TexMotionLocalization.TrLiteral("Temporal PyTorch 3D  +  RTMPose 2D  +  MediaPipe 3D Auxiliary  +  Multi-Hypothesis Fit (Stage B/C)")
                    : TexMotionLocalization.TrLiteral("RTMPose 2D  +  MediaPipe 3D  +  Multi-Hypothesis Fit (Stage B/C)"),
                EditorStyles.boldLabel);
            EditorGUILayout.LabelField(
                qualityBackendSelected
                    ? TexMotionLocalization.TrLiteral("PyTorch: Required for the selected quality backend  •  CUDA: Optional  •  CPU fallback remains available")
                    : TexMotionLocalization.TrLiteral("PyTorch / CUDA: Not required  •  Execution: CPU or DirectML via ONNX Runtime"),
                EditorStyles.miniLabel);

            EditorGUILayout.Space(3);
            EditorGUILayout.BeginHorizontal();
            DrawHybridPipelineChip(TexMotionLocalization.TrLiteral(qualityBackendSelected ? "Temporal 3D" : "RTMPose 2D"), (qualityBackendSelected ? qualityAssetReady : (onnxReady && rtmposeModelReady))
                ? new Color(0.35f, 0.95f, 0.55f)
                : new Color(0.95f, 0.75f, 0.25f));
            DrawHybridPipelineChip(TexMotionLocalization.TrLiteral(qualityBackendSelected ? "RTMPose 2D Aux" : "MediaPipe 3D"), (qualityBackendSelected ? onnxReady && rtmposeModelReady : mediaPipeReady)
                ? new Color(0.35f, 0.95f, 0.55f)
                : new Color(0.95f, 0.75f, 0.25f));
            DrawHybridPipelineChip(TexMotionLocalization.TrLiteral(qualityBackendSelected ? "MediaPipe 3D Aux" : "Stage B/C Fit"),
                qualityBackendSelected
                    ? (mediaPipeReady ? new Color(0.35f, 0.95f, 0.55f) : new Color(0.95f, 0.75f, 0.25f))
                    : new Color(0.35f, 0.82f, 1.0f));
            if (qualityBackendSelected)
            {
                DrawHybridPipelineChip(TexMotionLocalization.TrLiteral("Stage B/C Fit"), new Color(0.35f, 0.82f, 1.0f));
            }
            EditorGUILayout.EndHorizontal();

            if (localAssetsReady)
            {
                GUI.color = new Color(0.35f, 0.95f, 0.55f);
                EditorGUILayout.LabelField(
                    "● " + TexMotionLocalization.TrLiteral("Local runtime ready — no network access is used while extracting."),
                    EditorStyles.miniLabel);
                GUI.color = Color.white;
            }
            else
            {
                GUI.color = new Color(0.95f, 0.75f, 0.25f);
                EditorGUILayout.LabelField(
                    "○ " + TexMotionLocalization.TrLiteral("One-time local setup pending — cache the model/runtime once, then extract offline."),
                    EditorStyles.miniLabel);
                GUI.color = Color.white;
            }

            if (qualityBackendSelected && qualityPreflight != null && qualityPreflight.NonReadyAssets.Count > 0)
            {
                EditorGUILayout.HelpBox(
                    qualityPreflight.Describe() +
                    "\nOpen Settings > Video2Motion to use the row-specific Browse, Guide, or download action.",
                    qualityPreflight.IsReady ? MessageType.Info : MessageType.Warning);
            }

            if (mediaPipeFallbackSelected)
            {
                EditorGUILayout.LabelField(
                    TexMotionLocalization.TrLiteral("Compatibility mode selected: MediaPipe only. Choose Auto or RTMPose to enable the hybrid 2D auxiliary pass."),
                    EditorStyles.miniLabel);
            }
            else if (qualityBackendSelected)
            {
                EditorGUILayout.LabelField(
                    TexMotionLocalization.TrLiteral("Selected path: temporal PyTorch quality inference keeps competing 3D hypotheses through occlusion; RTMPose and MediaPipe remain auxiliary evidence for the Stage B/C fit."),
                    EditorStyles.miniLabel);
            }
            else
            {
                EditorGUILayout.LabelField(
                    TexMotionLocalization.TrLiteral("Selected path: hybrid observations with temporal multi-hypothesis optimization; heavy PyTorch/CUDA backends are not part of this path."),
                    EditorStyles.miniLabel);
            }

            EditorGUILayout.EndVertical();
        }

        private void DrawHybridPipelineChip(string label, Color color)
        {
            EnsureUiStyles();
            SetStyleTextColor(_chipStyle, color);
            GUILayout.Label("● " + label, _chipStyle, GUILayout.Height(21), GUILayout.ExpandWidth(true));
        }

        private void SetVideoSourcePath(string path)
        {
            if (string.IsNullOrEmpty(path)) return;
            _videoPath = path.Replace('\\', '/');
            string baseName = Path.GetFileNameWithoutExtension(_videoPath);
            if (!string.IsNullOrEmpty(baseName))
            {
                var sb = new StringBuilder();
                foreach (char c in baseName)
                {
                    if (char.IsLetterOrDigit(c) || c == '_') sb.Append(c);
                    else sb.Append('_');
                }
                string clean = sb.ToString().Trim('_');
                _videoMotionName = string.IsNullOrEmpty(clean) ? "VideoMotion" : clean;
            }

            _detectedVideoDuration = 0f;
            _detectedVideoWidth = 0;
            _detectedVideoHeight = 0;
            _detectedVideoFps = 0f;
            _videoTrimStart = 0f;
            _videoTrimEnd = 0f;

            string projectRoot = Path.GetFullPath(Application.dataPath + "/..").Replace('\\', '/');
            if (_videoPath.StartsWith(projectRoot, StringComparison.OrdinalIgnoreCase))
            {
                string relPath = _videoPath.Substring(projectRoot.Length).TrimStart('/');
                var clip = AssetDatabase.LoadAssetAtPath<UnityEngine.Video.VideoClip>(relPath);
                if (clip != null)
                {
                    _videoAsset = clip;
                    _detectedVideoDuration = (float)clip.length;
                    _detectedVideoWidth = (int)clip.width;
                    _detectedVideoHeight = (int)clip.height;
                    _detectedVideoFps = (float)clip.frameRate;
                    _videoTrimEnd = _detectedVideoDuration;
                    OnVideoSourceLoaded();
                    return;
                }
            }

            string assetPath = FileUtil.GetProjectRelativePath(_videoPath);
            if (!string.IsNullOrEmpty(assetPath))
            {
                var clip = AssetDatabase.LoadAssetAtPath<UnityEngine.Video.VideoClip>(assetPath);
                if (clip != null)
                {
                    _videoAsset = clip;
                    _detectedVideoDuration = (float)clip.length;
                    _detectedVideoWidth = (int)clip.width;
                    _detectedVideoHeight = (int)clip.height;
                    _detectedVideoFps = (float)clip.frameRate;
                    _videoTrimEnd = _detectedVideoDuration;
                    OnVideoSourceLoaded();
                    return;
                }
            }

            // External file: parse MP4/MOV container headers for dimensions and duration
            if (TryDetectExternalVideoMetadata(_videoPath, out int extW, out int extH, out float extDuration, out float extFps))
            {
                _detectedVideoWidth = extW;
                _detectedVideoHeight = extH;
                _detectedVideoDuration = extDuration;
                _detectedVideoFps = extFps;
                _videoTrimEnd = extDuration;
            }

            OnVideoSourceLoaded();
        }

        private void OnVideoSourceLoaded()
        {
            if (string.IsNullOrEmpty(_videoPath) || _videoPath == "synthetic_motion_test") return;
            string generatedCameraPath;
            VideoModelCatalog.AutoConfigureWhamCameraFromVideo(
                TexMotionSettings.instance,
                _videoPath,
                _detectedVideoWidth,
                _detectedVideoHeight,
                out generatedCameraPath);
            _autoVitPoseExportTriggeredForVideo = false;
            TryTriggerAutoViTPoseFeatureExport();
        }

        private static bool TryDetectExternalVideoMetadata(
            string filePath,
            out int width,
            out int height,
            out float duration,
            out float fps)
        {
            width = 0;
            height = 0;
            duration = 0f;
            fps = 0f;

            if (string.IsNullOrEmpty(filePath) || !File.Exists(filePath)) return false;

            try
            {
                using (var fs = new FileStream(filePath, FileMode.Open, FileAccess.Read, FileShare.ReadWrite))
                using (var reader = new BinaryReader(fs))
                {
                    long length = fs.Length;
                    uint timeScale = 0;
                    ulong durationUnits = 0;

                    while (fs.Position < length - 8)
                    {
                        long boxStart = fs.Position;
                        uint boxSize = ReadBigEndianUInt32(reader);
                        if (boxSize == 0 && boxStart == 0) break;
                        byte[] typeBytes = reader.ReadBytes(4);
                        if (typeBytes.Length < 4) break;
                        string boxType = Encoding.ASCII.GetString(typeBytes);

                        long actualSize = boxSize;
                        if (boxSize == 1)
                        {
                            if (fs.Position > length - 8) break;
                            actualSize = (long)ReadBigEndianUInt64(reader);
                        }
                        else if (boxSize == 0)
                        {
                            actualSize = length - boxStart;
                        }

                        if (actualSize < 8) break;

                        if (boxType == "moov" || boxType == "trak" || boxType == "mdia")
                        {
                            // Container box: descend into children
                            continue;
                        }
                        else if (boxType == "mvhd")
                        {
                            byte version = reader.ReadByte();
                            fs.Seek(3, SeekOrigin.Current);
                            if (version == 1)
                            {
                                fs.Seek(16, SeekOrigin.Current);
                                timeScale = ReadBigEndianUInt32(reader);
                                durationUnits = ReadBigEndianUInt64(reader);
                            }
                            else
                            {
                                fs.Seek(8, SeekOrigin.Current);
                                timeScale = ReadBigEndianUInt32(reader);
                                durationUnits = ReadBigEndianUInt32(reader);
                            }
                            if (timeScale > 0 && durationUnits > 0)
                            {
                                duration = (float)((double)durationUnits / timeScale);
                            }
                        }
                        else if (boxType == "tkhd")
                        {
                            byte version = reader.ReadByte();
                            fs.Seek(3, SeekOrigin.Current);
                            if (version == 1)
                            {
                                fs.Seek(8 + 8 + 4 + 4 + 8, SeekOrigin.Current);
                            }
                            else
                            {
                                fs.Seek(4 + 4 + 4 + 4 + 4, SeekOrigin.Current);
                            }
                            fs.Seek(52, SeekOrigin.Current);
                            uint wFixed = ReadBigEndianUInt32(reader);
                            uint hFixed = ReadBigEndianUInt32(reader);
                            int w = (int)(wFixed >> 16);
                            int h = (int)(hFixed >> 16);
                            if (w > 0 && h > 0 && width == 0)
                            {
                                width = w;
                                height = h;
                            }
                        }

                        fs.Seek(boxStart + actualSize, SeekOrigin.Begin);
                    }

                    if (width > 0 && height > 0)
                    {
                        if (fps <= 0f) fps = 30f;
                        return true;
                    }
                }
            }
            catch { }

            return false;
        }

        private static uint ReadBigEndianUInt32(BinaryReader reader)
        {
            byte[] b = reader.ReadBytes(4);
            if (b.Length < 4) return 0;
            return ((uint)b[0] << 24) | ((uint)b[1] << 16) | ((uint)b[2] << 8) | b[3];
        }

        private static ulong ReadBigEndianUInt64(BinaryReader reader)
        {
            byte[] b = reader.ReadBytes(8);
            if (b.Length < 8) return 0;
            return ((ulong)b[0] << 56) | ((ulong)b[1] << 48) | ((ulong)b[2] << 40) | ((ulong)b[3] << 32)
                 | ((ulong)b[4] << 24) | ((ulong)b[5] << 16) | ((ulong)b[6] << 8) | b[7];
        }

        private void DrawVideoMotionResultCard()
        {
            if (_currentVideoMotionData == null) return;

            BeginCard();
            EditorGUILayout.BeginHorizontal();
            EditorGUILayout.LabelField("✅ " + TexMotionLocalization.TrLiteral("Extracted Motion Metrics"), EditorStyles.boldLabel);

            float avgConf = _currentVideoMotionData.AverageConfidence;
            if (avgConf >= 0.70f)
            {
                GUI.color = new Color(0.3f, 0.95f, 0.4f);
                EditorGUILayout.LabelField(
                    "● " + TexMotionLocalization.TrFormat("High Confidence ({0:F1}%)", avgConf * 100f),
                    EditorStyles.boldLabel,
                    GUILayout.Width(180));
            }
            else if (avgConf >= 0.45f)
            {
                GUI.color = new Color(0.95f, 0.75f, 0.25f);
                EditorGUILayout.LabelField(
                    "● " + TexMotionLocalization.TrFormat("Moderate Confidence ({0:F1}%)", avgConf * 100f),
                    EditorStyles.boldLabel,
                    GUILayout.Width(180));
            }
            else
            {
                GUI.color = new Color(1.0f, 0.5f, 0.3f);
                EditorGUILayout.LabelField(
                    "● " + TexMotionLocalization.TrFormat("Low Confidence ({0:F1}%)", avgConf * 100f),
                    EditorStyles.boldLabel,
                    GUILayout.Width(180));
            }
            GUI.color = Color.white;
            EditorGUILayout.EndHorizontal();

            string detectorLabel = GetVideoDetectorDisplayName(_currentVideoMotionData);
            bool isHybridDetector = string.Equals(
                _currentVideoMotionData.DetectorName,
                "rtmpose_hybrid",
                StringComparison.OrdinalIgnoreCase);
            bool isWhamFusion = _currentVideoMotionData.IsActualWhamMediaPipeFusion;
            bool isWhamRequested = isWhamFusion ||
                string.Equals(_currentVideoMotionData.BackendRequested, "wham", StringComparison.OrdinalIgnoreCase) ||
                string.Equals(_currentVideoMotionData.BackendRequested, "wham_mediapipe", StringComparison.OrdinalIgnoreCase) ||
                string.Equals(_currentVideoMotionData.DetectorName, "wham", StringComparison.OrdinalIgnoreCase) ||
                string.Equals(_currentVideoMotionData.DetectorName, "wham_mediapipe", StringComparison.OrdinalIgnoreCase);
            VideoBackendMetadata backendMetadata = _currentVideoMotionData.BackendMetadata;
            SetStyleTextColor(_badgeStyle, isHybridDetector
                ? new Color(0.35f, 0.95f, 0.55f)
                : new Color(0.55f, 0.85f, 1.0f));
            _badgeStyle.alignment = TextAnchor.MiddleLeft;
            GUILayout.Label(detectorLabel, _badgeStyle, GUILayout.Height(22), GUILayout.ExpandWidth(false));

            EditorGUILayout.LabelField(
                TexMotionLocalization.TrFormat(
                    "Detector: {0} | Frames: {1} | Length: {2:F2}s | FPS: {3:F0} | Confidence Range: [{4:F0}% - {5:F0}%]",
                    detectorLabel,
                    _currentVideoMotionData.Frames,
                    _currentVideoMotionData.Duration,
                    _currentVideoMotionData.FrameRate,
                    _currentVideoMotionData.MinConfidence * 100f,
                    _currentVideoMotionData.MaxConfidence * 100f),
                EditorStyles.miniLabel);

            // Keep the requested and actual execution paths explicit.  A
            // checkpoint on disk is not proof that WHAM ran, so this line is
            // always derived from the extraction result's actual detector.
            EditorGUILayout.LabelField(
                TexMotionLocalization.TrFormat("Actual backend: {0}", detectorLabel),
                EditorStyles.miniLabel);

            // Keep the official WHAM stage contract visible in the result card.
            // These values come from runtime evidence, rather than from the
            // presence of a checkpoint path in Settings.
            if (isWhamRequested)
            {
                string officialRunnerStatus = _currentVideoMotionData.OfficialRunnerStatus;
                string hmr2ImageFeaturesStatus = _currentVideoMotionData.Hmr2ImageFeaturesStatus;
                string vitPose2DStatus = _currentVideoMotionData.VitPose2DStatus;
                if (backendMetadata != null)
                {
                    if (string.IsNullOrWhiteSpace(officialRunnerStatus))
                        officialRunnerStatus = backendMetadata.OfficialRunnerStatus;
                    if (string.IsNullOrWhiteSpace(hmr2ImageFeaturesStatus))
                        hmr2ImageFeaturesStatus = backendMetadata.Hmr2ImageFeaturesStatus;
                    if (string.IsNullOrWhiteSpace(vitPose2DStatus))
                        vitPose2DStatus = backendMetadata.VitPose2DStatus;
                }
                officialRunnerStatus = string.IsNullOrWhiteSpace(officialRunnerStatus)
                    ? TexMotionLocalization.TrLiteral("unknown") : officialRunnerStatus;
                hmr2ImageFeaturesStatus = string.IsNullOrWhiteSpace(hmr2ImageFeaturesStatus)
                    ? TexMotionLocalization.TrLiteral("unknown") : hmr2ImageFeaturesStatus;
                vitPose2DStatus = string.IsNullOrWhiteSpace(vitPose2DStatus)
                    ? TexMotionLocalization.TrLiteral("unknown") : vitPose2DStatus;
                EditorGUILayout.LabelField(
                    TexMotionLocalization.TrFormat("Official WHAM runner: {0}", officialRunnerStatus),
                    EditorStyles.miniLabel);
                EditorGUILayout.LabelField(
                    TexMotionLocalization.TrFormat("HMR2 image features: {0}", hmr2ImageFeaturesStatus),
                    EditorStyles.miniLabel);
                EditorGUILayout.LabelField(
                    TexMotionLocalization.TrFormat("ViTPose 2D detector: {0}", vitPose2DStatus),
                    EditorStyles.miniLabel);
            }

            if (_currentVideoMotionData.UsedBackendFallback)
            {
                string requested = string.IsNullOrEmpty(_currentVideoMotionData.BackendRequested)
                    ? TexMotionLocalization.TrLiteral("selected backend")
                    : _currentVideoMotionData.BackendRequested.ToUpperInvariant();
                string reason = string.IsNullOrEmpty(_currentVideoMotionData.BackendFallbackReason)
                    ? TexMotionLocalization.TrLiteral("The selected backend was unavailable.")
                    : _currentVideoMotionData.BackendFallbackReason;
                EditorGUILayout.HelpBox(
                    TexMotionLocalization.TrFormat(
                        "{0} was unavailable; the extraction fell back to {1}.\n{2} {3}",
                        requested,
                        detectorLabel,
                        TexMotionLocalization.TrLiteral("Reason:"),
                        reason),
                    MessageType.Warning);
                EditorGUILayout.LabelField(
                    TexMotionLocalization.TrFormat("Fallback reason: {0}", reason),
                    EditorStyles.wordWrappedMiniLabel);
            }

            if (isHybridDetector)
            {
                EditorGUILayout.LabelField(
                    TexMotionLocalization.TrLiteral("RTMPose 2D observations were fused with MediaPipe 3D auxiliaries and Stage B/C multi-hypothesis fitting."),
                    EditorStyles.miniLabel);
            }
            else if (isWhamFusion)
            {
                EditorGUILayout.LabelField(
                    TexMotionLocalization.TrLiteral("WHAM temporal 3D is primary; MediaPipe visibility-gated observations refine the canonical fused result."),
                    EditorStyles.miniLabel);
                if (_currentVideoMotionData.BackendMetadata != null && _currentVideoMotionData.BackendMetadata.Fusion != null)
                {
                    var fusion = _currentVideoMotionData.BackendMetadata.Fusion;
                    EditorGUILayout.LabelField(
                        TexMotionLocalization.TrFormat(
                            "Fusion: {0} camera | Reprojection error {1:F3} | WHAM correction {2:F3}m",
                            fusion.CameraModel,
                            fusion.MeanReprojectionError,
                            fusion.MeanWhamCorrection),
                        EditorStyles.miniLabel);
                }
                if (backendMetadata != null &&
                    !string.IsNullOrWhiteSpace(backendMetadata.SmplInitializationSource))
                {
                    string seedFrame = backendMetadata.SmplInitializationFrame >= 0
                        ? TexMotionLocalization.TrFormat(" (frame {0})", backendMetadata.SmplInitializationFrame + 1)
                        : string.Empty;
                    EditorGUILayout.LabelField(
                        TexMotionLocalization.TrFormat("WHAM initialization: {0}{1}", backendMetadata.SmplInitializationSource, seedFrame),
                        EditorStyles.miniLabel);
                }
            }
            else if (IsQualityDetectorName(_currentVideoMotionData.DetectorName))
            {
                EditorGUILayout.LabelField(
                    TexMotionLocalization.TrLiteral("Temporal PyTorch 3D hypotheses were fused with available RTMPose/MediaPipe auxiliary observations and Stage B/C fitting."),
                    EditorStyles.miniLabel);
            }

            // A quality backend can initialize successfully and still emit a
            // finite-but-collapsed body for individual frames.  The extractor
            // then keeps the WHAM diagnostics while using MediaPipe 3D for
            // that frame.  Surface this separately from BackendFallback so a
            // user can see the exact safety-gate reason in the result card.
            if (backendMetadata != null && backendMetadata.QualityFrameFallbackCount > 0)
            {
                int fallbackCount = backendMetadata.QualityFrameFallbackCount;
                int totalFrames = Mathf.Max(1, _currentVideoMotionData.Frames);
                string fallbackReason = string.IsNullOrWhiteSpace(backendMetadata.QualityFallbackReason)
                    ? TexMotionLocalization.TrLiteral("The quality pose failed silhouette alignment with the auxiliary observation.")
                    : backendMetadata.QualityFallbackReason;
                EditorGUILayout.HelpBox(
                    TexMotionLocalization.TrFormat(
                        "WHAM geometry fallback: {0}/{1} frame(s) used MediaPipe 3D for safe retargeting and overlay.\n{2} {3}\n{4}",
                        fallbackCount,
                        totalFrames,
                        TexMotionLocalization.TrLiteral("Reason:"),
                        fallbackReason,
                        TexMotionLocalization.TrLiteral("The affected spans are listed below and highlighted in the Timeline Editor.")),
                    MessageType.Warning);
            }

            // A WHAM checkpoint and companion files can be present while the
            // runtime still lacks executable image-feature or camera-motion
            // stages. Keep this diagnostic separate from the global backend
            // fallback and per-frame geometry gate so the user can fix the
            // actual missing stage instead of guessing from the avatar pose.
            if (isWhamRequested && backendMetadata != null)
            {
                string runtimeDiagnostic = backendMetadata.WhamAssetDiagnostic;
                if (string.IsNullOrWhiteSpace(runtimeDiagnostic) &&
                    backendMetadata.MissingOptionalAssets != null &&
                    backendMetadata.MissingOptionalAssets.Length > 0)
                {
                    runtimeDiagnostic =
                        TexMotionLocalization.TrLiteral("WHAM runtime stages unavailable: ") +
                        string.Join(", ", backendMetadata.MissingOptionalAssets);
                }
                if (!string.IsNullOrWhiteSpace(runtimeDiagnostic))
                {
                    var detail = runtimeDiagnostic.Trim();
                    if (backendMetadata.OptionalErrors != null && backendMetadata.OptionalErrors.Length > 0)
                    {
                        detail += "\n" + string.Join("\n", backendMetadata.OptionalErrors);
                    }
                    EditorGUILayout.HelpBox(
                        TexMotionLocalization.TrLiteral("WHAM asset/runtime diagnostic:") + "\n" + detail,
                        (!backendMetadata.WhamFullParity ||
                         (backendMetadata.MissingOptionalAssets != null && backendMetadata.MissingOptionalAssets.Length > 0))
                            ? MessageType.Warning
                            : MessageType.Info);
                }

                // Surface the Python preflight gate separately from the
                // extraction fallback. A native checkpoint can produce a
                // pose while an optional stage is still unverified; the
                // phase/error and asset names make that state actionable.
                if (!backendMetadata.BackendReady ||
                    !string.IsNullOrWhiteSpace(backendMetadata.PreflightStatus) ||
                    !string.IsNullOrWhiteSpace(backendMetadata.PreflightError) ||
                    (backendMetadata.MissingAssets != null && backendMetadata.MissingAssets.Length > 0) ||
                    (backendMetadata.IncompatibleAssets != null && backendMetadata.IncompatibleAssets.Length > 0))
                {
                    string preflight = string.IsNullOrWhiteSpace(backendMetadata.PreflightStatus)
                        ? TexMotionLocalization.TrLiteral("not reported")
                        : backendMetadata.PreflightStatus;
                    string phase = string.IsNullOrWhiteSpace(backendMetadata.PreflightPhase)
                        ? string.Empty
                        : TexMotionLocalization.TrFormat(" ({0})", backendMetadata.PreflightPhase);
                    string line = TexMotionLocalization.TrFormat(
                        "WHAM preflight: {0}{1} | ready={2} | smoke frames={3}",
                        preflight,
                        phase,
                        backendMetadata.BackendReady ? "yes" : "no",
                        Mathf.Max(0, backendMetadata.PreflightFrames));
                    if (!string.IsNullOrWhiteSpace(backendMetadata.PreflightError))
                        line += "\n" + backendMetadata.PreflightError;
                    if (backendMetadata.MissingAssets != null && backendMetadata.MissingAssets.Length > 0)
                        line += "\nMissing: " + string.Join(", ", backendMetadata.MissingAssets);
                    if (backendMetadata.IncompatibleAssets != null && backendMetadata.IncompatibleAssets.Length > 0)
                        line += "\nIncompatible: " + string.Join(", ", backendMetadata.IncompatibleAssets);
                    EditorGUILayout.HelpBox(line, backendMetadata.BackendReady ? MessageType.Info : MessageType.Warning);
                }

                if (!string.IsNullOrWhiteSpace(backendMetadata.ImageFeatureFallback) ||
                    !string.IsNullOrWhiteSpace(backendMetadata.ImageFeatureFallbackReason))
                {
                    string fallback = string.IsNullOrWhiteSpace(backendMetadata.ImageFeatureFallback)
                        ? TexMotionLocalization.TrLiteral("image-feature stage")
                        : backendMetadata.ImageFeatureFallback;
                    string detail = TexMotionLocalization.TrFormat(
                        "Image-feature fallback: {0} (consumed by WHAM integrator: {1}).",
                        fallback,
                        backendMetadata.ImageFeatureFallbackConsumed ? "yes" : "no");
                    if (!string.IsNullOrWhiteSpace(backendMetadata.ImageFeatureFallbackReason))
                        detail += "\n" + backendMetadata.ImageFeatureFallbackReason;
                    EditorGUILayout.HelpBox(detail, MessageType.Warning);
                }

                if (backendMetadata.CameraMotionProvenance != null ||
                    !string.IsNullOrWhiteSpace(backendMetadata.CameraMotionRunner))
                {
                    string source = backendMetadata.CameraMotionProvenance != null &&
                        !string.IsNullOrWhiteSpace(backendMetadata.CameraMotionProvenance.Source)
                        ? backendMetadata.CameraMotionProvenance.Source
                        : (string.IsNullOrWhiteSpace(backendMetadata.CameraMotionRunner)
                            ? "not reported"
                            : backendMetadata.CameraMotionRunner);
                    string runner = backendMetadata.CameraMotionProvenance != null &&
                        !string.IsNullOrWhiteSpace(backendMetadata.CameraMotionProvenance.Runner)
                        ? backendMetadata.CameraMotionProvenance.Runner
                        : backendMetadata.CameraMotionRunner;
                    EditorGUILayout.HelpBox(
                        TexMotionLocalization.TrFormat(
                            "Camera-motion provenance: source={0}, runner={1}, DPVO verified={2}",
                            source,
                            string.IsNullOrWhiteSpace(runner) ? "unknown" : runner,
                            backendMetadata.DpvoVerified ? "yes" : "no"),
                        backendMetadata.DpvoVerified ? MessageType.Info : MessageType.Warning);
                }

                // A checkpoint being present does not prove that the optional
                // image-feature runner was executable.  Keep this state
                // explicit so a fallback never looks like a skipped stage.
                bool runnerFallback = !backendMetadata.ImageFeatureRunnerActive &&
                    (string.Equals(backendMetadata.ImageFeaturePreprocess, "local_frame_descriptor", StringComparison.OrdinalIgnoreCase) ||
                     !string.IsNullOrWhiteSpace(backendMetadata.ImageFeatureRunnerError) ||
                     string.Equals(backendMetadata.ImageFeatureRunnerStatus, "fallback", StringComparison.OrdinalIgnoreCase));
                if (runnerFallback || !string.IsNullOrWhiteSpace(backendMetadata.WhamInferencePolicy))
                {
                    string runnerStage = string.IsNullOrWhiteSpace(backendMetadata.ImageFeaturePreprocess)
                        ? TexMotionLocalization.TrLiteral("not available")
                        : backendMetadata.ImageFeaturePreprocess;
                    string policy = string.IsNullOrWhiteSpace(backendMetadata.WhamMissingAssetBehavior)
                        ? TexMotionLocalization.TrLiteral("Missing optional files are reported and extraction continues with an explicit fallback; inference is never silently skipped.")
                        : backendMetadata.WhamMissingAssetBehavior;
                    string runnerDetail = runnerFallback
                        ? TexMotionLocalization.TrFormat("WHAM image-feature runner inactive; extraction continued with {0}.", runnerStage)
                        : TexMotionLocalization.TrLiteral("WHAM image-feature runner active.");
                    if (!string.IsNullOrWhiteSpace(backendMetadata.ImageFeatureRunnerError))
                    {
                        runnerDetail += "\n" + TexMotionLocalization.TrLiteral("Runner error:") + " " + backendMetadata.ImageFeatureRunnerError;
                    }
                    EditorGUILayout.HelpBox(runnerDetail + "\n" + policy, runnerFallback ? MessageType.Warning : MessageType.Info);
                }

                // A downloaded backbone/DPVO checkpoint is now prepared per
                // video by the local Python runner. Surface the concrete
                // provenance so users can distinguish a successful local
                // preparation from an older cache that still needs setup.
                if (!string.IsNullOrWhiteSpace(backendMetadata.ImageFeaturePreprocess) ||
                    !string.IsNullOrWhiteSpace(backendMetadata.CameraMotionRunner))
                {
                    string featureStage = string.IsNullOrWhiteSpace(backendMetadata.ImageFeaturePreprocess)
                        ? "not used"
                        : backendMetadata.ImageFeaturePreprocess;
                    string cameraStage = string.IsNullOrWhiteSpace(backendMetadata.CameraMotionRunner)
                        ? "not used"
                        : backendMetadata.CameraMotionRunner;
                    string manifest = string.IsNullOrWhiteSpace(backendMetadata.WhamPreprocessManifestPath)
                        ? string.Empty
                        : "\nManifest: " + backendMetadata.WhamPreprocessManifestPath;
                    EditorGUILayout.HelpBox(
                        TexMotionLocalization.TrLiteral("WHAM per-video preprocessing: image features = ") +
                        featureStage + ", camera motion = " + cameraStage + manifest,
                        MessageType.Info);
                }
                if (backendMetadata.RuntimeWarnings != null && backendMetadata.RuntimeWarnings.Length > 0)
                {
                    EditorGUILayout.HelpBox(
                        TexMotionLocalization.TrLiteral("WHAM local runner notes:") + "\n" +
                        string.Join("\n", backendMetadata.RuntimeWarnings),
                        MessageType.Info);
                }
            }

            if (_currentVideoMotionData.HasVariableTimestamps)
            {
                EditorGUILayout.LabelField("ℹ️ " + TexMotionLocalization.TrLiteral("Variable frame timestamps detected — synchronized with precision time interpolation."), EditorStyles.miniLabel);
            }

            DrawVideoUncertaintySummary();

            EditorGUILayout.BeginHorizontal();
            if (GUILayout.Button("✨ " + TexMotionLocalization.TrLiteral("Interpolate Low-Confidence Outliers"), EditorStyles.miniButton))
            {
                _currentVideoMotionData.InterpolateOutliers(_videoMinConfidenceThreshold > 0f ? _videoMinConfidenceThreshold : 0.4f);
                _statusMessage = TexMotionLocalization.TrLiteral("Interpolated low-confidence outlier frames.");
                SetupPreviewInstance();
                Repaint();
            }
            EditorGUILayout.EndHorizontal();

            EditorGUILayout.EndVertical();
        }

        /// <summary>
        /// Shows the uncertainty intervals emitted by the Stage B/C optimizer and provides
        /// a direct route to the timeline editor, where the highlighted spans can be inspected.
        /// </summary>
        private void DrawVideoUncertaintySummary()
        {
            UncertaintyInterval[] intervals = _currentVideoMotionData.UncertaintyIntervals;
            bool hasIntervals = intervals != null && intervals.Length > 0;

            EditorGUILayout.Space(6);
            BeginNestedCard();
            EditorGUILayout.BeginHorizontal();

            if (hasIntervals)
            {
                GUI.color = new Color(1.0f, 0.76f, 0.25f);
                EditorGUILayout.LabelField(
                    "⚠ " + TexMotionLocalization.TrFormat("Uncertainty Intervals  •  {0} detected", intervals.Length),
                    EditorStyles.boldLabel);
                GUI.color = Color.white;
            }
            else
            {
                GUI.color = new Color(0.35f, 0.95f, 0.55f);
                EditorGUILayout.LabelField("✓ " + TexMotionLocalization.TrLiteral("Uncertainty Intervals  •  None detected"), EditorStyles.boldLabel);
                GUI.color = Color.white;
            }

            string intervalSource = _currentVideoMotionData.IsActualWhamMediaPipeFusion
                ? TexMotionLocalization.TrLiteral("WHAM + MediaPipe Fusion")
                : TexMotionLocalization.TrLiteral("Stage B/C");
            EditorGUILayout.LabelField(intervalSource, EditorStyles.miniLabel, GUILayout.ExpandWidth(false));
            EditorGUILayout.EndHorizontal();

            if (hasIntervals)
            {
                const int maxVisibleIntervals = 4;
                int visibleCount = Mathf.Min(intervals.Length, maxVisibleIntervals);
                for (int i = 0; i < visibleCount; i++)
                {
                    UncertaintyInterval interval = intervals[i];
                    if (interval == null) continue;

                    int startFrame = Mathf.Max(0, interval.StartFrame);
                    int endFrame = Mathf.Max(startFrame, interval.EndFrame);
                    float confidence = Mathf.Clamp01(interval.Confidence);
                    string reason = string.IsNullOrWhiteSpace(interval.Reason)
                        ? TexMotionLocalization.TrLiteral("Ambiguous pose hypothesis")
                        : interval.Reason;

                    EditorGUILayout.LabelField(
                        TexMotionLocalization.TrFormat(
                            "• Frames {0}–{1}  |  {2}  |  Confidence {3:F0}%",
                            startFrame + 1,
                            endFrame + 1,
                            reason,
                            confidence * 100f),
                        EditorStyles.miniLabel);

                    if (!string.IsNullOrWhiteSpace(interval.RecommendedAction))
                    {
                        EditorGUILayout.LabelField(
                            TexMotionLocalization.TrFormat("  Suggested action: {0}", interval.RecommendedAction),
                            EditorStyles.miniLabel);
                    }
                }

                if (intervals.Length > maxVisibleIntervals)
                {
                    EditorGUILayout.LabelField(
                        TexMotionLocalization.TrFormat(
                            "  + {0} additional interval(s) highlighted in the timeline.",
                            intervals.Length - maxVisibleIntervals),
                        EditorStyles.miniLabel);
                }

                EditorGUILayout.HelpBox(
                    TexMotionLocalization.TrLiteral("Review the highlighted spans in the Timeline Editor. Scrub the interval and add a correction anchor when the two pose hypotheses are visually indistinguishable."),
                    MessageType.Warning);
                if (GUILayout.Button("✏️ " + TexMotionLocalization.TrLiteral("Review Uncertainty Intervals in Timeline Editor"), GUILayout.Height(26)))
                {
                    OpenInTimelineEditor();
                }
            }
            else
            {
                EditorGUILayout.LabelField(
                    TexMotionLocalization.TrLiteral("The optimizer found no ambiguous spans. You can still inspect and edit every frame in the Timeline Editor."),
                    EditorStyles.miniLabel);
            }

            EditorGUILayout.EndVertical();
        }

        private static string GetVideoDetectorDisplayName(string detectorName)
        {
            if (string.Equals(detectorName, "rtmpose_hybrid", StringComparison.OrdinalIgnoreCase))
            {
                return "● " + TexMotionLocalization.TrLiteral("RTMPose Hybrid (2D + 3D Auxiliary)");
            }

            if (string.Equals(detectorName, "rtmpose", StringComparison.OrdinalIgnoreCase))
            {
                return "● " + TexMotionLocalization.TrLiteral("RTMPose 2D Overlay");
            }

            if (string.Equals(detectorName, "synthetic", StringComparison.OrdinalIgnoreCase))
            {
                return "● " + TexMotionLocalization.TrLiteral("Synthetic Pose Overlay");
            }

            if (string.Equals(detectorName, "wham", StringComparison.OrdinalIgnoreCase))
            {
                return "● " + TexMotionLocalization.TrLiteral("Official WHAM Temporal 3D");
            }

            if (string.Equals(detectorName, "hmr2", StringComparison.OrdinalIgnoreCase))
            {
                return "● " + TexMotionLocalization.TrLiteral("HMR2 / 4D-Humans 3D");
            }

            if (string.Equals(detectorName, "hybrik", StringComparison.OrdinalIgnoreCase))
            {
                return "● " + TexMotionLocalization.TrLiteral("HybrIK 3D + IK");
            }

            if (string.Equals(detectorName, "pytorch_quality", StringComparison.OrdinalIgnoreCase) ||
                string.Equals(detectorName, "pytorch", StringComparison.OrdinalIgnoreCase))
            {
                return "● " + TexMotionLocalization.TrLiteral("PyTorch Temporal 3D");
            }

            return "● " + TexMotionLocalization.TrLiteral("MediaPipe Pose Overlay");
        }

        private static string GetVideoDetectorDisplayName(VideoMotionData motionData)
        {
            if (motionData != null && motionData.IsActualWhamMediaPipeFusion)
            {
                return "● " + TexMotionLocalization.TrLiteral("WHAM + MediaPipe");
            }
            return GetVideoDetectorDisplayName(motionData != null ? motionData.DetectorName : null);
        }

        private static bool IsQualityDetectorName(string detectorName)
        {
            return string.Equals(detectorName, "wham", StringComparison.OrdinalIgnoreCase) ||
                   string.Equals(detectorName, "hmr2", StringComparison.OrdinalIgnoreCase) ||
                   string.Equals(detectorName, "hybrik", StringComparison.OrdinalIgnoreCase) ||
                   string.Equals(detectorName, "pytorch", StringComparison.OrdinalIgnoreCase) ||
                   string.Equals(detectorName, "pytorch_quality", StringComparison.OrdinalIgnoreCase);
        }

        private void DrawSetupTargetSection(bool isVideo = false)
        {
            BeginCard();
            EditorGUILayout.LabelField(
                isVideo
                    ? "5. " + TexMotionLocalization.TrLiteral("VRChat Setup Target & Menu")
                    : "3. " + TexMotionLocalization.TrLiteral("VRChat Setup Target & Destination Menu"),
                EditorStyles.boldLabel);

            if (isVideo)
            {
                _videoMotionName = EditorGUILayout.TextField(TexMotionLocalization.TrLiteral("Motion Name"), _videoMotionName);
            }
            else
            {
                _motionName = EditorGUILayout.TextField(TexMotionLocalization.TrLiteral("Motion Name"), _motionName);
            }

            _setupMode = (VrcSetupMode)EditorGUILayout.EnumPopup(TexMotionLocalization.TrLiteral("Setup Target"), _setupMode);
            _motionType = (VrcMotionType)EditorGUILayout.EnumPopup(TexMotionLocalization.TrLiteral("Motion Playback"), _motionType);
            _targetLayer = (VrcTargetLayer)EditorGUILayout.EnumPopup(TexMotionLocalization.TrLiteral("Target Layer"), _targetLayer);

            if (!isVideo)
            {
                _inPlace = EditorGUILayout.Toggle(TexMotionLocalization.TrLiteral("In-Place (Stay on origin)"), _inPlace);
            }

            if (_setupMode == VrcSetupMode.DirectVRCSDK)
            {
                EditorGUILayout.Space(6);
                EditorGUILayout.LabelField("🎯 " + TexMotionLocalization.TrLiteral("Destination Expressions Menu:"), EditorStyles.boldLabel);

                _customTargetMenu = (ScriptableObject)EditorGUILayout.ObjectField(TexMotionLocalization.TrLiteral("Target Menu Asset"), _customTargetMenu, typeof(ScriptableObject), false);

                if (_targetAvatar != null)
                {
                    var availableMenus = VrcDirectSetup.FindAllAvatarMenus(_targetAvatar);
                    if (availableMenus.Count > 0)
                    {
                        EditorGUILayout.LabelField(TexMotionLocalization.TrLiteral("Quick Select from Avatar's Menus:"), EditorStyles.miniLabel);
                        EditorGUILayout.BeginHorizontal();
                        foreach (var m in availableMenus)
                        {
                            string mName = m.name;
                            bool isCurrent = _customTargetMenu == m || (_customTargetMenu == null && m == availableMenus[0]);
                            GUI.backgroundColor = isCurrent ? new Color(0.35f, 0.75f, 0.95f) : new Color(0.25f, 0.25f, 0.25f);
                            if (GUILayout.Button($"📁 {mName}", EditorStyles.miniButton, GUILayout.Height(24)))
                            {
                                _customTargetMenu = m;
                            }
                            GUI.backgroundColor = Color.white;
                        }
                        EditorGUILayout.EndHorizontal();
                    }
                }

                string currentSelectedName = _customTargetMenu != null
                    ? _customTargetMenu.name
                    : TexMotionLocalization.TrLiteral("Root Menu");
                EditorGUILayout.HelpBox(
                    TexMotionLocalization.TrFormat("Motion button will be added directly into '{0}'.", currentSelectedName),
                    MessageType.Info);
            }
            else
            {
                EditorGUILayout.HelpBox(TexMotionLocalization.TrLiteral("Modular Avatar Mode: Will create a non-destructive child object under your avatar."), MessageType.Info);
            }

            EditorGUILayout.EndVertical();
        }

        private void DrawLibraryTab()
        {
            EditorGUILayout.BeginHorizontal();
            EditorGUILayout.LabelField("📚 " + TexMotionLocalization.TrLiteral("Saved Motion Library"), EditorStyles.boldLabel);
            if (GUILayout.Button("🔄 " + TexMotionLocalization.TrLiteral("Refresh"), GUILayout.Width(80)))
            {
                RefreshLibrary();
            }
            EditorGUILayout.EndHorizontal();

            EditorGUILayout.LabelField(TexMotionLocalization.TrLiteral("Manage, preview, and add/remove generated motions on your avatar."), EditorStyles.miniLabel);
            EditorGUILayout.Space(8);

            // Target Avatar selector
            BeginInlineCard();
            EditorGUILayout.LabelField(TexMotionLocalization.TrLiteral("Target Avatar:"), GUILayout.Width(90));
            _targetAvatar = (GameObject)EditorGUILayout.ObjectField(_targetAvatar, typeof(GameObject), true);
            EditorGUILayout.EndHorizontal();
            EditorGUILayout.Space(5);

            DrawSetupTargetSection(false);
            EditorGUILayout.Space(8);

            if (_libraryItems.Count == 0)
            {
                EditorGUILayout.HelpBox(TexMotionLocalization.TrLiteral("No generated motions found in Assets/TexMotion/Generated. Generate a new motion in 'Motion Generator' or extract one in 'Video Motion'!"), MessageType.Info);
                return;
            }

            for (int i = 0; i < _libraryItems.Count; i++)
            {
                var item = _libraryItems[i];
                bool isSelected = _selectedLibraryItem == item;

                EnsureUiStyles();
                EditorGUILayout.BeginVertical(isSelected ? _selectedCardStyle : _cardStyle);

                EditorGUILayout.BeginHorizontal();

                EditorGUILayout.BeginVertical();
                EditorGUILayout.LabelField("🎬 " + item.Name, EditorStyles.boldLabel);
                EditorGUILayout.LabelField(
                    TexMotionLocalization.TrFormat(
                        "{0:F2}s ({1:F0}fps) | {2}",
                        item.Duration,
                        item.FrameRate,
                        item.IsLoop ? TexMotionLocalization.TrLiteral("Loop") : TexMotionLocalization.TrLiteral("One-Shot")),
                    EditorStyles.miniLabel);
                EditorGUILayout.EndVertical();

                if (item.IsAppliedToAvatar)
                {
                    GUI.color = new Color(0.3f, 0.9f, 0.4f);
                    EditorGUILayout.LabelField("● " + TexMotionLocalization.TrLiteral("Applied to Avatar"), EditorStyles.boldLabel, GUILayout.Width(130));
                    GUI.color = Color.white;
                }
                else
                {
                    GUI.color = new Color(0.6f, 0.6f, 0.6f);
                    EditorGUILayout.LabelField("○ " + TexMotionLocalization.TrLiteral("Asset Only"), EditorStyles.miniLabel, GUILayout.Width(90));
                    GUI.color = Color.white;
                }

                GUI.backgroundColor = isSelected ? new Color(0.95f, 0.65f, 0.2f) : Color.white;
                if (GUILayout.Button("👁 " + TexMotionLocalization.TrLiteral(isSelected ? "Viewing" : "Preview"), GUILayout.Width(75), GUILayout.Height(28)))
                {
                    _selectedLibraryItem = item;
                    _activePreviewClip = item.Clip;
                    _currentMotionData = null;
                    _currentVideoMotionData = null;
                    _previewTime = 0f;
                    _isPlayingPreview = true;
                    _lastUpdateTime = EditorApplication.timeSinceStartup;
                    SetupPreviewInstance();
                }
                GUI.backgroundColor = Color.white;

                GUI.backgroundColor = new Color(0.95f, 0.7f, 0.2f);
                if (GUILayout.Button("✏️ " + TexMotionLocalization.TrLiteral("Edit"), GUILayout.Width(60), GUILayout.Height(28)))
                {
                    Animator targetAnim = _targetAvatar != null ? _targetAvatar.GetComponent<Animator>() : null;
                    if (targetAnim != null && item.Clip != null)
                    {
                        var editData = EditableMotionData.FromAnimationClip(item.Clip, targetAnim);
                        if (editData != null)
                        {
                            MotionTimelineEditorWindow.OpenWithEditableData(editData);
                        }
                    }
                    else
                    {
                        EditorUtility.DisplayDialog(
                            TexMotionLocalization.TrLiteral("Target Avatar Required"),
                            TexMotionLocalization.TrLiteral("Please assign a humanoid Target Avatar above to edit this clip."),
                            TexMotionLocalization.TrLiteral("OK"));
                    }
                }
                GUI.backgroundColor = Color.white;

                if (item.IsAppliedToAvatar)
                {
                    GUI.backgroundColor = new Color(0.9f, 0.4f, 0.4f);
                    if (GUILayout.Button("❌ " + TexMotionLocalization.TrLiteral("Remove"), GUILayout.Width(75), GUILayout.Height(28)))
                    {
                        MotionLibraryManager.RemoveFromAvatar(_targetAvatar, item);
                        RefreshLibrary();
                    }
                    GUI.backgroundColor = Color.white;
                }
                else
                {
                    GUI.backgroundColor = new Color(0.3f, 0.85f, 0.45f);
                    if (GUILayout.Button("✨ " + TexMotionLocalization.TrLiteral("Apply"), GUILayout.Width(75), GUILayout.Height(28)))
                    {
                        ApplyLibraryItemToAvatar(item);
                        RefreshLibrary();
                    }
                    GUI.backgroundColor = Color.white;
                }

                GUI.backgroundColor = new Color(0.5f, 0.5f, 0.5f);
                if (GUILayout.Button("🗑️", GUILayout.Width(30), GUILayout.Height(28)))
                {
                    if (EditorUtility.DisplayDialog(
                        TexMotionLocalization.TrLiteral("Delete Motion"),
                        TexMotionLocalization.TrFormat("Are you sure you want to completely delete '{0}' from project?", item.Name),
                        TexMotionLocalization.TrLiteral("Delete"),
                        TexMotionLocalization.TrLiteral("Cancel")))
                    {
                        MotionLibraryManager.DeleteMotionFiles(item, _targetAvatar);
                        RefreshLibrary();
                    }
                }
                GUI.backgroundColor = Color.white;

                EditorGUILayout.EndHorizontal();
                EditorGUILayout.EndVertical();
                EditorGUILayout.Space(2);
            }

            if (_selectedLibraryItem != null && _activePreviewClip != null)
            {
                EditorGUILayout.Space(12);
                DrawLibraryPreviewSection();
            }
        }

        private void DrawLibraryPreviewSection()
        {
            BeginCard();

            EditorGUILayout.LabelField(
                "🎬 " + TexMotionLocalization.TrFormat("Previewing Library Motion: '{0}'", _selectedLibraryItem.Name),
                EditorStyles.boldLabel);
            EditorGUILayout.LabelField(
                TexMotionLocalization.TrFormat("Duration: {0:F2}s | FPS: {1:F0}", _activePreviewClip.length, _activePreviewClip.frameRate),
                EditorStyles.miniLabel);

            EditorGUILayout.Space(5);

            Rect previewRect = GUILayoutUtility.GetRect(400, 240, GUILayout.ExpandWidth(true));
            Event evt = Event.current;

            if (previewRect.Contains(evt.mousePosition))
            {
                if (evt.type == EventType.MouseDrag && evt.button == 0)
                {
                    _previewDir.x -= evt.delta.x * 0.8f;
                    _previewDir.y += evt.delta.y * 0.8f;
                    _previewDir.y = Mathf.Clamp(_previewDir.y, -80f, 80f);
                    evt.Use();
                    Repaint();
                }
                else if (evt.type == EventType.ScrollWheel)
                {
                    _previewDistance += evt.delta.y * 0.15f;
                    _previewDistance = Mathf.Clamp(_previewDistance, 1.0f, 8.0f);
                    evt.Use();
                    Repaint();
                }
            }

            if (_previewUtility != null && _previewInstance != null)
            {
                _previewUtility.BeginPreview(previewRect, GUIStyle.none);

                Quaternion camRot = Quaternion.Euler(_previewDir.y, _previewDir.x, 0);
                Vector3 camPos = _previewPivot + camRot * (Vector3.forward * _previewDistance);

                _previewUtility.camera.transform.position = camPos;
                _previewUtility.camera.transform.LookAt(_previewPivot);

                _activePreviewClip.SampleAnimation(_previewInstance, _previewTime);

                _previewUtility.Render(true);
                Texture previewTex = _previewUtility.EndPreview();
                GUI.DrawTexture(previewRect, previewTex, ScaleMode.StretchToFill, false);
            }

            EditorGUILayout.Space(8);

            EditorGUILayout.BeginHorizontal();
            EditorGUI.BeginChangeCheck();
            _previewTime = EditorGUILayout.Slider(TexMotionLocalization.TrLiteral("Timeline"), _previewTime, 0f, _activePreviewClip.length);
            if (EditorGUI.EndChangeCheck())
            {
                _isPlayingPreview = false;
                if (_previewInstance != null)
                {
                    _activePreviewClip.SampleAnimation(_previewInstance, _previewTime);
                }
            }
            EditorGUILayout.LabelField(
                TexMotionLocalization.TrFormat("{0:F2}s / {1:F2}s", _previewTime, _activePreviewClip.length),
                GUILayout.Width(80));
            EditorGUILayout.EndHorizontal();

            EditorGUILayout.BeginHorizontal();
            GUI.backgroundColor = _isPlayingPreview ? new Color(0.95f, 0.65f, 0.2f) : new Color(0.3f, 0.85f, 0.45f);
            if (GUILayout.Button(TexMotionLocalization.TrLiteral(_isPlayingPreview ? "⏸ Pause" : "▶ Play"), GUILayout.Height(28)))
            {
                _isPlayingPreview = !_isPlayingPreview;
                if (_isPlayingPreview) _lastUpdateTime = EditorApplication.timeSinceStartup;
            }
            GUI.backgroundColor = Color.white;

            if (GUILayout.Button(TexMotionLocalization.TrLiteral("⏹ Reset"), GUILayout.Height(28), GUILayout.Width(75)))
            {
                _isPlayingPreview = false;
                _previewTime = 0f;
                if (_previewInstance != null) _activePreviewClip.SampleAnimation(_previewInstance, 0f);
            }

            if (GUILayout.Button(TexMotionLocalization.TrLiteral("🔄 Reset View"), GUILayout.Height(28), GUILayout.Width(95)))
            {
                _previewDir = new Vector2(180f, 10f);
                _previewDistance = 2.8f;
            }
            EditorGUILayout.EndHorizontal();

            EditorGUILayout.EndVertical();
        }

        private void ApplyLibraryItemToAvatar(MotionLibraryItem item)
        {
            if (_targetAvatar == null || item.Clip == null) return;

            var vrcConfig = new VrcMotionConfig
            {
                MotionName = item.Name,
                SetupMode = _setupMode,
                MotionType = item.IsLoop ? VrcMotionType.ToggleLoopPose : VrcMotionType.OneShotEmote,
                TargetLayer = _targetLayer,
                InPlace = true
            };

            if (_setupMode == VrcSetupMode.ModularAvatar)
            {
                if (!ModularAvatarSetup.IsModularAvatarInstalled())
                {
                    ModularAvatarSetup.InstallModularAvatar(() => ApplyLibraryItemToAvatar(item));
                    return;
                }

                GameObject setupObj = ModularAvatarSetup.SetupAvatarMotion(_targetAvatar, item.Clip, vrcConfig, MotionLibraryManager.GeneratedDirectory);
                _statusMessage = TexMotionLocalization.TrFormat("Applied '{0}' to {1} via Modular Avatar!", item.Name, _targetAvatar.name);
                EditorGUIUtility.PingObject(setupObj);
            }
            else
            {
                VrcDirectSetup.SetupDirectAvatarMotion(_targetAvatar, item.Clip, vrcConfig, _customTargetMenu, MotionLibraryManager.GeneratedDirectory);
                string menuName = _customTargetMenu != null
                    ? _customTargetMenu.name
                    : TexMotionLocalization.TrLiteral("VRCExpressionsMenu");
                _statusMessage = TexMotionLocalization.TrFormat("Applied '{0}' to {1}!", item.Name, menuName);
            }

            RefreshLibrary();
        }

        private void DrawInteractive3DPreviewSection()
        {
            GeneratedMotionData motion = (_currentTab == Tab.VideoMotion)
                ? (_currentVideoMotionData ?? _currentMotionData)
                : (_currentMotionData ?? _currentVideoMotionData);

            if (motion == null) return;

            if (_currentMotionData == null)
            {
                _currentMotionData = motion;
            }

            BeginCard();

            try
            {
                float duration = GetActivePreviewDuration();
                VideoMotionData vmd = motion as VideoMotionData;

                if (vmd != null)
                {
                    string videoToPlay = !string.IsNullOrEmpty(vmd.OverlayVideoPath) && File.Exists(vmd.OverlayVideoPath)
                        ? vmd.OverlayVideoPath
                        : (!string.IsNullOrEmpty(vmd.SourceVideoPath) && File.Exists(vmd.SourceVideoPath) ? vmd.SourceVideoPath : null);

                    if (!string.IsNullOrEmpty(videoToPlay) && (_videoPlayer == null || _loadedVideoPath != videoToPlay))
                    {
                        if (!_videoPlayerFailed || _failedVideoPath != videoToPlay)
                        {
                            SetupVideoPlayer(videoToPlay);
                        }
                    }

                    // Viewport Toolbar with Display Modes
                    EditorGUILayout.BeginHorizontal();
                    EditorGUILayout.LabelField("🎥 " + TexMotionLocalization.TrLiteral("Synchronized Side-by-Side Motion Preview"), EditorStyles.boldLabel);
                    GUILayout.FlexibleSpace();
                    _previewDisplayMode = (PreviewDisplayMode)GUILayout.Toolbar((int)_previewDisplayMode,
                        new string[]
                        {
                            TexMotionLocalization.TrLiteral("Side-by-Side"),
                            TexMotionLocalization.TrLiteral("Avatar Only"),
                            TexMotionLocalization.TrLiteral("Video Only")
                        },
                        EditorStyles.miniButton, GUILayout.Width(270));
                    EditorGUILayout.EndHorizontal();

                    EditorGUILayout.LabelField(
                        TexMotionLocalization.TrFormat(
                            "Frames: {0} | Length: {1:F2}s | FPS: {2:F0} | Confidence: {3:F1}%",
                            vmd.Frames,
                            duration,
                            vmd.FrameRate,
                            vmd.AverageConfidence * 100f),
                        EditorStyles.miniLabel);
                }
                else
                {
                    EditorGUILayout.LabelField(
                        "🎬 2. " + TexMotionLocalization.TrLiteral("3D Motion Preview (Drag to Rotate, Scroll to Zoom)"),
                        EditorStyles.boldLabel);
                    EditorGUILayout.LabelField(
                        TexMotionLocalization.TrFormat(
                            "Frames: {0} | Length: {1:F2}s | Hand: {2} | Face: {3}",
                            motion.Frames,
                            duration,
                            _handPose,
                            _faceEmotion),
                        EditorStyles.miniLabel);
                }

                EditorGUILayout.Space(5);

                // Render viewports according to display mode
                if (vmd != null)
                {
                    if (_previewDisplayMode == PreviewDisplayMode.SideBySide)
                    {
                        Rect containerRect = GUILayoutUtility.GetRect(400, 290, GUILayout.ExpandWidth(true));
                        if (containerRect.width < 50f) containerRect.width = position.width > 100f ? position.width - 40f : 400f;
                        if (containerRect.height < 50f) containerRect.height = 290f;
                        float gap = 6f;
                        float halfW = Mathf.Max(20f, (containerRect.width - gap) * 0.5f);

                        Rect leftRect = new Rect(containerRect.x, containerRect.y, halfW, containerRect.height);
                        Rect rightRect = new Rect(containerRect.x + halfW + gap, containerRect.y, halfW, containerRect.height);

                        DrawVideoViewport(leftRect, vmd);
                        DrawAvatarViewport(rightRect, duration);
                    }
                    else if (_previewDisplayMode == PreviewDisplayMode.VideoOnly)
                    {
                        Rect videoRect = GUILayoutUtility.GetRect(400, 320, GUILayout.ExpandWidth(true));
                        if (videoRect.width < 50f) videoRect.width = position.width > 100f ? position.width - 40f : 400f;
                        if (videoRect.height < 50f) videoRect.height = 320f;
                        DrawVideoViewport(videoRect, vmd);
                    }
                    else // AvatarOnly
                    {
                        Rect avatarRect = GUILayoutUtility.GetRect(400, 320, GUILayout.ExpandWidth(true));
                        if (avatarRect.width < 50f) avatarRect.width = position.width > 100f ? position.width - 40f : 400f;
                        if (avatarRect.height < 50f) avatarRect.height = 320f;
                        DrawAvatarViewport(avatarRect, duration);
                    }
                }
                else
                {
                    // Standard Generator preview (Avatar Only)
                    Rect previewRect = GUILayoutUtility.GetRect(400, 270, GUILayout.ExpandWidth(true));
                    if (previewRect.width < 50f) previewRect.width = position.width > 100f ? position.width - 40f : 400f;
                    if (previewRect.height < 50f) previewRect.height = 270f;
                    DrawAvatarViewport(previewRect, duration);
                }

                EditorGUILayout.Space(8);

                // Synchronized Timeline Slider
                EditorGUILayout.BeginHorizontal();
                EditorGUI.BeginChangeCheck();
                _previewTime = EditorGUILayout.Slider(TexMotionLocalization.TrLiteral("Timeline"), _previewTime, 0f, duration);
                if (EditorGUI.EndChangeCheck())
                {
                    _isPlayingPreview = false;
                    ApplyMotionPoseToPreview(_previewTime);
                    SyncVideoPlayer(_previewTime, false);
                }
                EditorGUILayout.LabelField(
                    TexMotionLocalization.TrFormat("{0:F2}s / {1:F2}s", _previewTime, duration),
                    GUILayout.Width(85));
                EditorGUILayout.EndHorizontal();

                // Playback Control Buttons
                EditorGUILayout.BeginHorizontal();
                GUI.backgroundColor = _isPlayingPreview ? new Color(0.95f, 0.65f, 0.2f) : new Color(0.3f, 0.85f, 0.45f);
                if (GUILayout.Button(TexMotionLocalization.TrLiteral(_isPlayingPreview ? "⏸ Pause" : "▶ Play"), GUILayout.Height(30)))
                {
                    _isPlayingPreview = !_isPlayingPreview;
                    if (_isPlayingPreview)
                    {
                        _videoPlayerFailed = false;
                        _lastUpdateTime = EditorApplication.timeSinceStartup;
                        if (_previewTime >= duration - 0.08f || _previewTime == 0f)
                        {
                            _previewTime = 0f;
                            RestartVideoLoop();
                        }
                        else
                        {
                            if (_videoPlayer != null && _videoPlayer.isPrepared)
                            {
                                float vidLen = Mathf.Max(0.01f, (float)_videoPlayer.length);
                                _videoPlayer.time = Mathf.Clamp(_previewTime, 0f, vidLen);
                                _videoPlayer.Play();
                                EditorApplication.QueuePlayerLoopUpdate();
                            }
                        }
                    }
                    else
                    {
                        if (_videoPlayer != null && _videoPlayer.isPlaying)
                        {
                            _videoPlayer.Pause();
                        }
                    }
                    Repaint();
                }
                GUI.backgroundColor = Color.white;

                if (GUILayout.Button(TexMotionLocalization.TrLiteral("⏹ Reset"), GUILayout.Height(30), GUILayout.Width(75)))
                {
                    _isPlayingPreview = false;
                    _previewTime = 0f;
                    _videoPlayerFailed = false;
                    ApplyMotionPoseToPreview(0f);
                    if (_videoPlayer != null && _videoPlayer.isPrepared)
                    {
                        _videoPlayer.time = 0.0;
                        _videoPlayer.Play();
                        _videoPlayer.Pause();
                        EditorApplication.QueuePlayerLoopUpdate();
                    }
                    Repaint();
                }

                if (GUILayout.Button(TexMotionLocalization.TrLiteral("🔄 Reset View"), GUILayout.Height(30), GUILayout.Width(95)))
                {
                    _previewDir = new Vector2(180f, 10f);
                    _previewDistance = 2.8f;
                }
                EditorGUILayout.EndHorizontal();

                EditorGUILayout.Space(14);
                EditorGUILayout.LabelField(TexMotionLocalization.TrLiteral("Apply to Avatar:"), EditorStyles.boldLabel);

                EditorGUILayout.BeginHorizontal();

                GUI.backgroundColor = new Color(0.95f, 0.7f, 0.2f);
                if (GUILayout.Button(TexMotionLocalization.TrLiteral("✏️ Edit in Timeline"), GUILayout.Height(40)))
                {
                    OpenInTimelineEditor();
                }

                GUI.backgroundColor = new Color(0.2f, 0.85f, 0.45f);
                if (GUILayout.Button(TexMotionLocalization.TrLiteral("✨ Apply to Avatar"), GUILayout.Height(40)))
                {
                    ApplyMotionToAvatar();
                }

                GUI.backgroundColor = new Color(0.4f, 0.65f, 0.95f);
                if (GUILayout.Button(TexMotionLocalization.TrLiteral("💾 Save .anim Only"), GUILayout.Height(40)))
                {
                    SaveClipOnly();
                }

                GUI.backgroundColor = new Color(0.85f, 0.35f, 0.35f);
                if (GUILayout.Button(TexMotionLocalization.TrLiteral("🗑️ Discard"), GUILayout.Height(40), GUILayout.Width(80)))
                {
                    _currentMotionData = null;
                    _currentVideoMotionData = null;
                    CleanupPreviewInstance();
                    _statusMessage = TexMotionLocalization.TrLiteral("Preview discarded.");
                }

                GUI.backgroundColor = Color.white;
                EditorGUILayout.EndHorizontal();
            }
            finally
            {
                EditorGUILayout.EndVertical();
            }
        }

        private void DrawVideoViewport(Rect rect, VideoMotionData vmd)
        {
            if (rect.width <= 4f || rect.height <= 4f)
                return;

            // Dark viewport background
            if (Event.current.type == EventType.Repaint)
            {
                EditorGUI.DrawRect(rect, new Color(0.08f, 0.10f, 0.14f));
            }

            // Subtle border
            DrawViewportBorder(rect, new Color(0.25f, 0.35f, 0.50f, 0.6f));

            // Render video texture or overlay preparation gauge
            if (_previewVideoTexture != null && _videoPlayer != null && _videoPlayer.isPrepared)
            {
                if (Event.current.type == EventType.Repaint)
                {
                    GUI.DrawTexture(rect, _previewVideoTexture, ScaleMode.ScaleToFit);
                }
            }
            else if (_videoPlayer != null && !_videoPlayer.isPrepared && !_videoPlayerFailed)
            {
                DrawVideoPreparationOverlay(rect);
            }
            else if (_videoPlayerFailed)
            {
                DrawVideoFallbackOverlay(rect);
            }
            else
            {
                GUI.Label(rect, TexMotionLocalization.TrLiteral("Video Stream Not Available"), EditorStyles.centeredGreyMiniLabel);
            }

            // Draw HUD Badges strictly during Repaint to eliminate flickering / blinking
            if (Event.current.type == EventType.Repaint)
            {
                // Top-Left HUD Badge: Stream Title
                if (rect.width > 180f && rect.height > 40f)
                {
                    var badgeStyle = new GUIStyle(EditorStyles.miniLabel)
                    {
                        fontStyle = FontStyle.Bold,
                        normal = { textColor = vmd != null &&
                            string.Equals(vmd.DetectorName, "rtmpose_hybrid", StringComparison.OrdinalIgnoreCase)
                            ? new Color(0.35f, 0.95f, 0.55f)
                            : new Color(0.35f, 0.85f, 1.0f) },
                        padding = new RectOffset(6, 0, 2, 0),
                        clipping = TextClipping.Clip
                    };
                    string detectorTitle = GetVideoDetectorDisplayName(vmd != null ? vmd.DetectorName : null);
                    float measuredWidth = badgeStyle.CalcSize(new GUIContent(detectorTitle)).x + 12f;
                    float badgeWidth = Mathf.Min(Mathf.Max(185f, measuredWidth), rect.width - 16f);
                    Rect titleBadge = new Rect(rect.x + 8, rect.y + 8, badgeWidth, 20);
                    EditorGUI.DrawRect(titleBadge, new Color(0.05f, 0.07f, 0.11f, 0.75f));
                    GUI.Label(titleBadge, detectorTitle, badgeStyle);
                }

                // Current Frame calculation
                int frameIdx = 0;
                if (vmd != null && vmd.Frames > 0)
                {
                    float dur = Mathf.Max(0.001f, vmd.Duration);
                    frameIdx = Mathf.Clamp(Mathf.RoundToInt((_previewTime / dur) * (vmd.Frames - 1)), 0, vmd.Frames - 1);
                }

                // Top-Right HUD Badge: Confidence Score
                if (vmd != null && vmd.Frames > 0 && rect.width > 240f && rect.height > 40f)
                {
                    float conf = vmd.GetFrameConfidence(frameIdx);
                    Rect confBadge = new Rect(rect.x + rect.width - 110, rect.y + 8, 102, 20);
                    EditorGUI.DrawRect(confBadge, new Color(0.05f, 0.07f, 0.11f, 0.75f));

                    Color confColor = conf >= 0.70f
                        ? new Color(0.3f, 0.95f, 0.4f)
                        : (conf >= 0.45f ? new Color(0.95f, 0.75f, 0.25f) : new Color(1.0f, 0.45f, 0.35f));

                    var confStyle = new GUIStyle(EditorStyles.miniLabel)
                    {
                        fontStyle = FontStyle.Bold,
                        normal = { textColor = confColor },
                        alignment = TextAnchor.MiddleCenter
                    };
                    GUI.Label(confBadge, TexMotionLocalization.TrFormat("Conf: {0:F0}%", conf * 100f), confStyle);
                }

                // Bottom-Left HUD Badge: Frame Number
                if (vmd != null && vmd.Frames > 0 && rect.width > 140f && rect.height > 50f)
                {
                    Rect frameBadge = new Rect(rect.x + 8, rect.y + rect.height - 24, 110, 18);
                    EditorGUI.DrawRect(frameBadge, new Color(0.05f, 0.07f, 0.11f, 0.70f));
                    var frameStyle = new GUIStyle(EditorStyles.miniLabel)
                    {
                        normal = { textColor = new Color(0.80f, 0.85f, 0.95f) },
                        padding = new RectOffset(6, 0, 1, 0)
                    };
                    GUI.Label(frameBadge, TexMotionLocalization.TrFormat("Frame: {0} / {1}", frameIdx + 1, vmd.Frames), frameStyle);
                }
            }
        }

        private void DrawVideoPreparationOverlay(Rect viewportRect)
        {
            if (Event.current.type != EventType.Repaint)
                return;

            // Continually trigger Repaint to keep shimmer and pulsation smooth at 60fps
            Repaint();

            float time = (float)EditorApplication.timeSinceStartup;

            // Center card inside viewport
            float cardWidth = Mathf.Min(340f, viewportRect.width - 24f);
            float cardHeight = Mathf.Min(118f, viewportRect.height - 20f);
            if (cardWidth < 80f || cardHeight < 40f) return;

            float cardX = viewportRect.x + (viewportRect.width - cardWidth) * 0.5f;
            float cardY = viewportRect.y + (viewportRect.height - cardHeight) * 0.5f;
            Rect cardRect = new Rect(cardX, cardY, cardWidth, cardHeight);

            // 1. Subtle Outer Glow / Shadow
            Rect shadowRect = new Rect(cardRect.x - 2, cardRect.y - 2, cardRect.width + 4, cardRect.height + 4);
            EditorGUI.DrawRect(shadowRect, new Color(0.02f, 0.04f, 0.08f, 0.60f));

            // 2. Card Background (Deep Translucent Dark Glass)
            EditorGUI.DrawRect(cardRect, new Color(0.07f, 0.10f, 0.16f, 0.94f));

            // 3. Card Border with cyan accent
            DrawViewportBorder(cardRect, new Color(0.20f, 0.65f, 0.95f, 0.70f));

            // 4. Header with Pulsing Status Indicator
            float pulse = 0.65f + 0.35f * Mathf.Sin(time * 4.0f);
            Color indicatorColor = new Color(0.35f, 0.85f, 1.0f, pulse);

            var titleStyle = new GUIStyle(EditorStyles.boldLabel)
            {
                alignment = TextAnchor.MiddleCenter,
                fontSize = 12,
                normal = { textColor = new Color(0.92f, 0.96f, 1.0f) }
            };
            Rect titleRect = new Rect(cardRect.x + 10, cardRect.y + 12, cardRect.width - 20, 20);
            GUI.Label(titleRect, "⏳ " + TexMotionLocalization.TrLiteral("Preparing Pose Overlay Video..."), titleStyle);

            // 5. Subtitle Status
            var subStyle = new GUIStyle(EditorStyles.miniLabel)
            {
                alignment = TextAnchor.MiddleCenter,
                fontSize = 10,
                normal = { textColor = new Color(0.65f, 0.78f, 0.92f, 0.85f) }
            };
            Rect subRect = new Rect(cardRect.x + 10, cardRect.y + 34, cardRect.width - 20, 16);
            GUI.Label(subRect, TexMotionLocalization.TrLiteral("Loading frames & buffering stream..."), subStyle);

            // 6. Modern Animated Progress Gauge (Indeterminate Shimmer Bar)
            float barPaddingX = 22f;
            float barWidth = cardRect.width - (barPaddingX * 2);
            float barHeight = 6f;
            float barY = cardRect.y + 60f;
            Rect barBgRect = new Rect(cardRect.x + barPaddingX, barY, barWidth, barHeight);

            // Bar background slot
            EditorGUI.DrawRect(barBgRect, new Color(0.04f, 0.06f, 0.10f, 0.95f));
            DrawViewportBorder(barBgRect, new Color(0.18f, 0.28f, 0.40f, 0.60f));

            // Shimmer / Wave bar indicator moving smoothly
            float segmentWidth = barWidth * 0.35f;
            float cycle = Mathf.Repeat(time * 0.75f, 1.0f);
            float travelDistance = barWidth + segmentWidth;
            float segmentX = barBgRect.x - segmentWidth + (cycle * travelDistance);

            float visibleLeft = Mathf.Max(barBgRect.x, segmentX);
            float visibleRight = Mathf.Min(barBgRect.x + barWidth, segmentX + segmentWidth);
            if (visibleRight > visibleLeft)
            {
                Rect fillRect = new Rect(visibleLeft, barBgRect.y, visibleRight - visibleLeft, barHeight);
                EditorGUI.DrawRect(fillRect, new Color(0.15f, 0.80f, 0.95f, 0.92f));
            }

            // 7. Dynamic status dots at bottom
            int dotCount = ((int)(time * 2.5f) % 4);
            string dots = new string('.', dotCount);
            var dotStyle = new GUIStyle(EditorStyles.miniLabel)
            {
                alignment = TextAnchor.MiddleCenter,
                normal = { textColor = indicatorColor },
                fontSize = 10,
                fontStyle = FontStyle.Bold
            };
            Rect dotRect = new Rect(cardRect.x, cardRect.y + 74f, cardRect.width, 18);
            GUI.Label(dotRect, TexMotionLocalization.TrFormat("Buffering video stream{0}", dots), dotStyle);
        }

        private void DrawVideoFallbackOverlay(Rect viewportRect)
        {
            float cardWidth = Mathf.Min(340f, viewportRect.width - 24f);
            float cardHeight = Mathf.Min(135f, viewportRect.height - 20f);
            if (cardWidth < 80f || cardHeight < 40f) return;

            float cardX = viewportRect.x + (viewportRect.width - cardWidth) * 0.5f;
            float cardY = viewportRect.y + (viewportRect.height - cardHeight) * 0.5f;
            Rect cardRect = new Rect(cardX, cardY, cardWidth, cardHeight);

            if (Event.current.type == EventType.Repaint)
            {
                // Card Background & Border with amber warning accent
                EditorGUI.DrawRect(cardRect, new Color(0.08f, 0.09f, 0.12f, 0.94f));
                DrawViewportBorder(cardRect, new Color(0.95f, 0.65f, 0.20f, 0.70f));

                var titleStyle = new GUIStyle(EditorStyles.boldLabel)
                {
                    alignment = TextAnchor.MiddleCenter,
                    fontSize = 12,
                    normal = { textColor = new Color(1.0f, 0.85f, 0.45f) }
                };
                Rect titleRect = new Rect(cardRect.x + 10, cardRect.y + 12, cardRect.width - 20, 20);
                GUI.Label(titleRect, "⚠️ " + TexMotionLocalization.TrLiteral("Video Overlay Playback Skipped"), titleStyle);

                var subStyle = new GUIStyle(EditorStyles.miniLabel)
                {
                    alignment = TextAnchor.MiddleCenter,
                    fontSize = 10,
                    normal = { textColor = new Color(0.80f, 0.85f, 0.92f, 0.85f) }
                };
                Rect subRect = new Rect(cardRect.x + 10, cardRect.y + 34, cardRect.width - 20, 16);
                GUI.Label(subRect, TexMotionLocalization.TrLiteral("OS codec decode unavailable or timed out."), subStyle);

                var infoStyle = new GUIStyle(EditorStyles.miniLabel)
                {
                    alignment = TextAnchor.MiddleCenter,
                    normal = { textColor = new Color(0.35f, 0.90f, 0.55f) },
                    fontSize = 10,
                    fontStyle = FontStyle.Bold
                };
                Rect infoRect = new Rect(cardRect.x + 10, cardRect.y + 54, cardRect.width - 20, 18);
                GUI.Label(infoRect, "✓ " + TexMotionLocalization.TrLiteral("3D Avatar Motion Playing Smoothly"), infoStyle);
            }

            // Interactive Retry Button
            Rect retryRect = new Rect(cardRect.x + 40, cardRect.y + 84, cardRect.width - 80, 24);
            if (GUI.Button(retryRect, "🔄 " + TexMotionLocalization.TrLiteral("Retry Video Playback"), EditorStyles.miniButton))
            {
                _videoPlayerFailed = false;
                _failedVideoPath = "";
                if (!string.IsNullOrEmpty(_loadedVideoPath) && File.Exists(_loadedVideoPath))
                {
                    SetupVideoPlayer(_loadedVideoPath);
                }
                else if (_currentMotionData is VideoMotionData vmd)
                {
                    string targetVid = !string.IsNullOrEmpty(vmd.OverlayVideoPath) && File.Exists(vmd.OverlayVideoPath)
                        ? vmd.OverlayVideoPath
                        : vmd.SourceVideoPath;
                    if (!string.IsNullOrEmpty(targetVid) && File.Exists(targetVid))
                    {
                        SetupVideoPlayer(targetVid);
                    }
                }
                Repaint();
            }
        }

        private void DrawAvatarViewport(Rect rect, float duration)
        {
            if (rect.width <= 4f || rect.height <= 4f)
                return;

            // Handle 3D Orbit Camera Drag & Zoom Events
            Event evt = Event.current;
            if (rect.Contains(evt.mousePosition))
            {
                if (evt.type == EventType.MouseDrag && evt.button == 0)
                {
                    _previewDir.x -= evt.delta.x * 0.8f;
                    _previewDir.y += evt.delta.y * 0.8f;
                    _previewDir.y = Mathf.Clamp(_previewDir.y, -80f, 80f);
                    evt.Use();
                    Repaint();
                }
                else if (evt.type == EventType.ScrollWheel)
                {
                    _previewDistance += evt.delta.y * 0.15f;
                    _previewDistance = Mathf.Clamp(_previewDistance, 1.0f, 8.0f);
                    evt.Use();
                    Repaint();
                }
            }

            // Render PreviewUtility 3D Scene (Only during Repaint event to prevent RenderTexture size errors)
            if (_previewUtility != null && _previewInstance != null && rect.width > 4f && rect.height > 4f)
            {
                if (Event.current.type == EventType.Repaint)
                {
                    _previewUtility.BeginPreview(rect, GUIStyle.none);

                    Quaternion camRot = Quaternion.Euler(_previewDir.y, _previewDir.x, 0);
                    Vector3 camPos = _previewPivot + camRot * (Vector3.forward * _previewDistance);

                    _previewUtility.camera.transform.position = camPos;
                    _previewUtility.camera.transform.LookAt(_previewPivot);

                    ApplyMotionPoseToPreview(_previewTime);

                    _previewUtility.Render(true);
                    Texture previewTex = _previewUtility.EndPreview();
                    GUI.DrawTexture(rect, previewTex, ScaleMode.StretchToFill, false);
                }
            }
            else
            {
                if (Event.current.type == EventType.Repaint)
                {
                    EditorGUI.DrawRect(rect, new Color(0.1f, 0.1f, 0.12f));
                    GUI.Label(rect, TexMotionLocalization.TrLiteral("Loading 3D Preview..."), EditorStyles.centeredGreyMiniLabel);
                }
            }

            // Subtle border
            DrawViewportBorder(rect, new Color(0.25f, 0.35f, 0.50f, 0.6f));

            // Top-Left HUD Badge: Avatar Stream Title
            Rect titleBadge = new Rect(rect.x + 8, rect.y + 8, 160, 20);
            EditorGUI.DrawRect(titleBadge, new Color(0.05f, 0.07f, 0.11f, 0.75f));
            var badgeStyle = new GUIStyle(EditorStyles.miniLabel)
            {
                fontStyle = FontStyle.Bold,
                normal = { textColor = new Color(0.4f, 0.95f, 0.6f) },
                padding = new RectOffset(6, 0, 2, 0)
            };
            GUI.Label(titleBadge, "● " + TexMotionLocalization.TrLiteral("3D Avatar (SMPL-X 22)"), badgeStyle);

            // Bottom-Right Hint: Orbit / Zoom
            Rect hintBadge = new Rect(rect.x + rect.width - 150, rect.y + rect.height - 24, 142, 18);
            EditorGUI.DrawRect(hintBadge, new Color(0.05f, 0.07f, 0.11f, 0.70f));
            var hintStyle = new GUIStyle(EditorStyles.miniLabel)
            {
                normal = { textColor = new Color(0.70f, 0.75f, 0.85f) },
                alignment = TextAnchor.MiddleCenter
            };
            GUI.Label(hintBadge, TexMotionLocalization.TrLiteral("Drag: Orbit | Scroll: Zoom"), hintStyle);
        }

        private static void DrawViewportBorder(Rect rect, Color borderColor)
        {
            EditorGUI.DrawRect(new Rect(rect.x, rect.y, rect.width, 1), borderColor);
            EditorGUI.DrawRect(new Rect(rect.x, rect.y + rect.height - 1, rect.width, 1), borderColor);
            EditorGUI.DrawRect(new Rect(rect.x, rect.y, 1, rect.height), borderColor);
            EditorGUI.DrawRect(new Rect(rect.x + rect.width - 1, rect.y, 1, rect.height), borderColor);
        }

        private void OpenInTimelineEditor()
        {
            if (_currentMotionData == null) return;

            Animator targetAnim = null;
            if (_targetAvatar != null)
            {
                targetAnim = _targetAvatar.GetComponent<Animator>();
                if (targetAnim == null)
                {
                    targetAnim = _targetAvatar.GetComponentInChildren<Animator>();
                }
            }
            if (targetAnim == null)
            {
                var anims = FindObjectsOfType<Animator>();
                foreach (var a in anims)
                {
                    if (a.isHuman) { targetAnim = a; break; }
                }
            }

            bool isVideo = (_currentMotionData is VideoMotionData);
            string mName = isVideo ? _videoMotionName : _motionName;
            var hPose = isVideo ? _videoHandPose : _handPose;
            var fEmo = isVideo ? _videoFaceEmotion : _faceEmotion;
            var eInt = isVideo ? _videoEmotionIntensity : _emotionIntensity;
            bool inPl = isVideo ? _videoInPlace : _inPlace;

            MotionTimelineEditorWindow.OpenWithMotion(
                _currentMotionData,
                targetAnim,
                mName,
                hPose,
                fEmo,
                eInt,
                inPl
            );
        }

        private void ApplyMotionToAvatar()
        {
            if (_currentMotionData == null || _targetAvatar == null) return;

            if (_setupMode == VrcSetupMode.ModularAvatar && !ModularAvatarSetup.IsModularAvatarInstalled())
            {
                bool shouldInstall = EditorUtility.DisplayDialog(
                    "Modular Avatar が見つかりません",
                    "Modular Avatar (非破壊モード) でのセットアップには「Modular Avatar」が必要です。\n\nModular Avatar を自動的に導入しますか？\n（※キャンセルした場合は「Setup Target」を「DirectVRCSDK」に変更して直接登録することも可能です）",
                    "はい (自動導入する)",
                    "キャンセル"
                );

                if (shouldInstall)
                {
                    _statusMessage = "Modular Avatar を自動インストール中...";
                    ModularAvatarSetup.InstallModularAvatar(() =>
                    {
                        ApplyMotionToAvatar();
                    });
                }
                return;
            }

            try
            {
                string saveDir = MotionLibraryManager.GeneratedDirectory;
                if (!Directory.Exists(saveDir)) Directory.CreateDirectory(saveDir);

                var animator = _targetAvatar.GetComponent<Animator>();

                bool isVideo = (_currentTab == Tab.VideoMotion) || (_currentMotionData is VideoMotionData);
                string motionName = isVideo && !string.IsNullOrEmpty(_videoMotionName) ? _videoMotionName : _motionName;
                bool inPlace = isVideo ? _videoInPlace : _inPlace;
                HandPoseType activeHandPose = isVideo ? _videoHandPose : _handPose;
                FaceEmotionType activeEmotion = isVideo ? _videoFaceEmotion : _faceEmotion;
                float activeIntensity = isVideo ? _videoEmotionIntensity : _emotionIntensity;
                float minConf = isVideo ? _videoMinConfidenceThreshold : 0f;

                if (activeEmotion == FaceEmotionType.AutoDetect)
                {
                    activeEmotion = isVideo ? FaceEmotionType.None : FaceEmotionHelper.InferEmotionFromPrompt(_prompt);
                }

                var buildOptions = new AnimationBuildOptions
                {
                    ClipName = $"Anim_{motionName}",
                    IsLoop = _motionType == VrcMotionType.ToggleLoopPose,
                    InPlace = inPlace,
                    Speed = 1.0f,
                    TargetAvatar = animator,
                    HandPose = activeHandPose,
                    FaceEmotion = activeEmotion,
                    EmotionIntensity = activeIntensity,
                    MinConfidenceThreshold = minConf
                };

                var clip = AnimationClipBuilder.BuildAnimationClip(_currentMotionData, buildOptions);

                string clipPath = $"{saveDir}/Anim_{motionName}.anim";
                AssetDatabase.CreateAsset(clip, clipPath);
                AssetDatabase.SaveAssets();

                var vrcConfig = new VrcMotionConfig
                {
                    MotionName = motionName,
                    SetupMode = _setupMode,
                    MotionType = _motionType,
                    TargetLayer = _targetLayer,
                    InPlace = inPlace
                };

                if (_setupMode == VrcSetupMode.ModularAvatar)
                {
                    GameObject setupObj = ModularAvatarSetup.SetupAvatarMotion(_targetAvatar, clip, vrcConfig, saveDir);
                    _statusMessage = TexMotionLocalization.TrFormat(
                        "Successfully applied '{0}' to {1} via Modular Avatar!",
                        motionName,
                        _targetAvatar.name);
                    EditorGUIUtility.PingObject(setupObj);
                    Selection.activeGameObject = setupObj;
                    EditorUtility.DisplayDialog(
                        TexMotionLocalization.TrLiteral("Setup Complete"),
                        TexMotionLocalization.TrFormat(
                            "Motion '{0}' was successfully applied to {1} via Modular Avatar!",
                            motionName,
                            _targetAvatar.name),
                        TexMotionLocalization.TrLiteral("Great!"));
                }
                else
                {
                    VrcDirectSetup.SetupDirectAvatarMotion(_targetAvatar, clip, vrcConfig, _customTargetMenu, saveDir);
                    string menuName = _customTargetMenu != null
                        ? _customTargetMenu.name
                        : TexMotionLocalization.TrLiteral("VRCExpressionsMenu");
                    _statusMessage = TexMotionLocalization.TrFormat(
                        "Successfully added '{0}' directly to {1} & Animator!",
                        motionName,
                        menuName);
                    EditorUtility.DisplayDialog(
                        TexMotionLocalization.TrLiteral("Setup Complete"),
                        TexMotionLocalization.TrFormat(
                            "Motion '{0}' was successfully added directly into '{1}', ExpressionParameters, and {2}!",
                            motionName,
                            menuName,
                            _targetLayer),
                        TexMotionLocalization.TrLiteral("Great!"));
                }
                _currentMotionData = null;
                _currentVideoMotionData = null;
                CleanupPreviewInstance();
                RefreshLibrary();
            }
            catch (Exception ex)
            {
                _statusMessage = TexMotionLocalization.TrFormat("Failed to apply motion: {0}", ex.Message);
                Debug.LogError($"[TexMotion] {ex}");
                EditorUtility.DisplayDialog(TexMotionLocalization.TrLiteral("Error"), ex.Message, TexMotionLocalization.TrLiteral("OK"));
            }
        }

        private void SaveClipOnly()
        {
            if (_currentMotionData == null || _targetAvatar == null) return;

            try
            {
                string saveDir = MotionLibraryManager.GeneratedDirectory;
                if (!Directory.Exists(saveDir)) Directory.CreateDirectory(saveDir);

                var animator = _targetAvatar.GetComponent<Animator>();

                bool isVideo = (_currentTab == Tab.VideoMotion) || (_currentMotionData is VideoMotionData);
                string motionName = isVideo && !string.IsNullOrEmpty(_videoMotionName) ? _videoMotionName : _motionName;
                bool inPlace = isVideo ? _videoInPlace : _inPlace;
                HandPoseType activeHandPose = isVideo ? _videoHandPose : _handPose;
                FaceEmotionType activeEmotion = isVideo ? _videoFaceEmotion : _faceEmotion;
                float activeIntensity = isVideo ? _videoEmotionIntensity : _emotionIntensity;
                float minConf = isVideo ? _videoMinConfidenceThreshold : 0f;

                if (activeEmotion == FaceEmotionType.AutoDetect)
                {
                    activeEmotion = isVideo ? FaceEmotionType.None : FaceEmotionHelper.InferEmotionFromPrompt(_prompt);
                }

                var buildOptions = new AnimationBuildOptions
                {
                    ClipName = $"Anim_{motionName}",
                    IsLoop = _motionType == VrcMotionType.ToggleLoopPose,
                    InPlace = inPlace,
                    Speed = 1.0f,
                    TargetAvatar = animator,
                    HandPose = activeHandPose,
                    FaceEmotion = activeEmotion,
                    EmotionIntensity = activeIntensity,
                    MinConfidenceThreshold = minConf
                };

                var clip = AnimationClipBuilder.BuildAnimationClip(_currentMotionData, buildOptions);

                string clipPath = $"{saveDir}/Anim_{motionName}.anim";
                AssetDatabase.CreateAsset(clip, clipPath);
                AssetDatabase.SaveAssets();

                _statusMessage = TexMotionLocalization.TrFormat("Saved AnimationClip to {0}", clipPath);
                EditorGUIUtility.PingObject(clip);
                Selection.activeObject = clip;

                EditorUtility.DisplayDialog(
                    TexMotionLocalization.TrLiteral("Saved"),
                    TexMotionLocalization.TrFormat("AnimationClip saved to {0}", clipPath),
                    TexMotionLocalization.TrLiteral("OK"));

                _currentMotionData = null;
                _currentVideoMotionData = null;
                CleanupPreviewInstance();
                RefreshLibrary();
            }
            catch (Exception ex)
            {
                _statusMessage = TexMotionLocalization.TrFormat("Failed to save clip: {0}", ex.Message);
                Debug.LogError($"[TexMotion] {ex}");
            }
        }

        private void DrawModelStatusBanner(bool ready)
        {
            BeginInlineCard();
            if (ready)
            {
                GUI.color = new Color(0.4f, 0.9f, 0.5f);
                EditorGUILayout.LabelField("● " + TexMotionLocalization.TrLiteral("Models Ready"), EditorStyles.boldLabel, GUILayout.Width(110));
                GUI.color = Color.white;
                EditorGUILayout.LabelField(TexMotionLocalization.TrLiteral("Kimodo + LLM2Vec Text Bundle Loaded"), EditorStyles.miniLabel);
            }
            else
            {
                GUI.color = new Color(1.0f, 0.4f, 0.4f);
                EditorGUILayout.LabelField("● " + TexMotionLocalization.TrLiteral("Models Missing"), EditorStyles.boldLabel, GUILayout.Width(110));
                GUI.color = Color.white;
                if (GUILayout.Button(TexMotionLocalization.TrLiteral("Go to Settings to Download"), GUILayout.Height(20)))
                {
                    _currentTab = Tab.Settings;
                }
            }
            EditorGUILayout.EndHorizontal();
        }

        private void DrawSettingsTab()
        {
            var settings = TexMotionSettings.instance;

            switch (_settingsCategory)
            {
                case SettingsCategory.General:
                    DrawGeneralSettings(settings);
                    break;
                case SettingsCategory.MotionGenerator:
                    DrawMotionGeneratorSettings(settings);
                    break;
                case SettingsCategory.VideoMotion:
                    DrawVideoMotionSettings(settings);
                    break;
            }

            if (GUI.changed)
            {
                settings.Save();
            }
        }

        private void DrawGeneralSettings(TexMotionSettings settings)
        {
            // 1. Language Setting
            BeginCard();
            EditorGUILayout.LabelField("🌐 " + TexMotionLocalization.Tr(TexMotionLocalization.Language), EditorStyles.boldLabel);
            var selectedLanguage = (TexMotionLanguage)EditorGUILayout.EnumPopup(
                TexMotionLocalization.Tr(TexMotionLocalization.Language), settings.Language);
            if (selectedLanguage != settings.Language)
            {
                settings.Language = selectedLanguage;
                settings.Save();
                Repaint();
            }
            EditorGUILayout.EndVertical();
            EditorGUILayout.Space(10);

            // 2. Motion Timeline Editor Integration
            BeginCard();
            _showSettingsWorkflow = DrawPersistedFoldoutHeader(
                FoldoutSettingsWorkflow,
                _showSettingsWorkflow,
                "⏱️ " + TexMotionLocalization.TrLiteral("Motion Timeline Editor"),
                TexMotionLocalization.TrLiteral("Workflow and launch"));
            if (_showSettingsWorkflow)
            {
                settings.AutoOpenTimelineEditor = EditorGUILayout.Toggle(
                    TexMotionLocalization.TrLiteral("Auto Open Timeline Editor"),
                    settings.AutoOpenTimelineEditor);
                EditorGUILayout.HelpBox(
                    TexMotionLocalization.TrLiteral("Automatically opens the dedicated Motion Timeline Editor upon completing text motion generation or video pose extraction."),
                    MessageType.None);
                if (GUILayout.Button(TexMotionLocalization.TrLiteral("Launch Motion Timeline Editor"), GUILayout.Height(26)))
                {
                    MotionTimelineEditorWindow.ShowEditor();
                }
            }
            EditorGUILayout.EndVertical();
            EditorGUILayout.Space(10);

            // 3. Local Storage & Cache Configuration
            BeginCard();
            _showSettingsLocalStorage = DrawPersistedFoldoutHeader(
                FoldoutSettingsLocalStorage,
                _showSettingsLocalStorage,
                "📁 " + TexMotionLocalization.TrLiteral("Local Storage Configuration"),
                TexMotionLocalization.TrLiteral("Cache and resolved paths"));
            if (_showSettingsLocalStorage)
            {
                settings.UseCustomLocalPath = EditorGUILayout.Toggle(TexMotionLocalization.TrLiteral("Use Custom Local Path"), settings.UseCustomLocalPath);

                if (settings.UseCustomLocalPath)
                {
                    EditorGUILayout.BeginHorizontal();
                    settings.CustomLocalModelDirectory = EditorGUILayout.TextField(TexMotionLocalization.TrLiteral("Local Directory"), settings.CustomLocalModelDirectory);
                    if (GUILayout.Button(TexMotionLocalization.TrLiteral("Browse..."), GUILayout.Width(75)))
                    {
                        string dir = EditorUtility.OpenFolderPanel(TexMotionLocalization.TrLiteral("Select Model Directory"), settings.CustomLocalModelDirectory, "");
                        if (!string.IsNullOrEmpty(dir))
                        {
                            settings.CustomLocalModelDirectory = dir;
                        }
                    }
                    EditorGUILayout.EndHorizontal();
                }
                else
                {
                    EditorGUILayout.LabelField(TexMotionLocalization.TrLiteral("Default Cache Directory (AppData):"), EditorStyles.miniLabel);
                    EditorGUILayout.SelectableLabel(settings.GetDefaultCacheDirectory(), EditorStyles.textField, GUILayout.Height(20));
                }

                EditorGUILayout.Space(5);
                EditorGUILayout.LabelField(TexMotionLocalization.TrLiteral("Resolved Paths:"), _sectionTitleStyle);

                string motionPath = settings.GetMotionModelPath();
                bool motionExists = File.Exists(motionPath);
                EditorGUILayout.LabelField(
                    TexMotionLocalization.TrFormat(
                        "Motion Model: {0}",
                        motionExists ? TexMotionLocalization.TrLiteral("✅ Found") : TexMotionLocalization.TrLiteral("❌ Not Found")),
                    EditorStyles.miniLabel);
                EditorGUILayout.SelectableLabel(motionPath, EditorStyles.textField, GUILayout.Height(18));

                string bundleDir = settings.GetTextBundleDirectory();
                bool bundleExists = settings.IsValidTextBundle(bundleDir);
                EditorGUILayout.LabelField(
                    TexMotionLocalization.TrFormat(
                        "Text Bundle: {0}",
                        bundleExists ? TexMotionLocalization.TrLiteral("✅ Found") : TexMotionLocalization.TrLiteral("❌ Not Found")),
                    EditorStyles.miniLabel);
                EditorGUILayout.SelectableLabel(bundleDir, EditorStyles.textField, GUILayout.Height(18));

                if (GUILayout.Button("🔄 " + TexMotionLocalization.TrLiteral("Reload Engine"), GUILayout.Height(24)))
                {
                    _engine?.UnloadModel();
                    EnsureEngineLoaded();
                }
            }
            EditorGUILayout.EndVertical();
        }

        private void DrawMotionGeneratorSettings(TexMotionSettings settings)
        {
            // 1. Model Status & Download
            BeginCard();
            EditorGUILayout.LabelField("📥 " + TexMotionLocalization.Tr(TexMotionLocalization.ModelDownloader), EditorStyles.boldLabel);
            EditorGUILayout.LabelField(
                TexMotionLocalization.TrFormat("Destination: {0}", settings.GetEffectiveModelDirectory()),
                EditorStyles.miniLabel);

            string motionPath = settings.GetMotionModelPath();
            bool motionExists = File.Exists(motionPath);
            string bundleDir = settings.GetTextBundleDirectory();
            bool bundleExists = settings.IsValidTextBundle(bundleDir);

            BeginNestedRow();
            EditorGUILayout.LabelField("Motion Model (GGUF):", GUILayout.Width(170));
            GUI.color = motionExists ? new Color(0.35f, 0.95f, 0.50f) : new Color(1.0f, 0.40f, 0.40f);
            EditorGUILayout.LabelField(motionExists ? "● " + TexMotionLocalization.TrLiteral("Ready") : "○ " + TexMotionLocalization.TrLiteral("Missing"), EditorStyles.boldLabel);
            GUI.color = Color.white;
            EditorGUILayout.EndHorizontal();

            BeginNestedRow();
            EditorGUILayout.LabelField("Text Bundle Folder:", GUILayout.Width(170));
            GUI.color = bundleExists ? new Color(0.35f, 0.95f, 0.50f) : new Color(1.0f, 0.40f, 0.40f);
            EditorGUILayout.LabelField(bundleExists ? "● " + TexMotionLocalization.TrLiteral("Ready") : "○ " + TexMotionLocalization.TrLiteral("Missing"), EditorStyles.boldLabel);
            GUI.color = Color.white;
            EditorGUILayout.EndHorizontal();

            EditorGUILayout.Space(6);

            GUI.enabled = !_isDownloading && !_isVideoModelDownloading && !_isGenerating && !_isVideoExtracting &&
                          !_isInstallingDependencies && !_isBuildingVideoVenv && !_isInstallingUv;
            if (GUILayout.Button("📥 " + TexMotionLocalization.TrLiteral("Download Models from Hugging Face"), GUILayout.Height(35)))
            {
                StartDownload();
            }
            GUI.enabled = true;

            if (_isDownloading)
            {
                if (GUILayout.Button(TexMotionLocalization.TrLiteral("Cancel Download"), GUILayout.Height(25)))
                {
                    _cts?.Cancel();
                }
            }
            EditorGUILayout.EndVertical();
            EditorGUILayout.Space(10);

            // 2. Model Repository & Files
            BeginCard();
            _showSettingsModelRepository = DrawPersistedFoldoutHeader(
                FoldoutSettingsModelRepository,
                _showSettingsModelRepository,
                "📦 " + TexMotionLocalization.TrLiteral("Model Repository & Files"),
                TexMotionLocalization.TrLiteral("Hugging Face repository"));
            if (_showSettingsModelRepository)
            {
                settings.HuggingFaceRepo = EditorGUILayout.TextField(TexMotionLocalization.TrLiteral("HF Repo (User/Repo)"), settings.HuggingFaceRepo);
                settings.MotionModelFileName = EditorGUILayout.TextField(TexMotionLocalization.TrLiteral("Motion GGUF File"), settings.MotionModelFileName);
                settings.TextBundleDirName = EditorGUILayout.TextField(TexMotionLocalization.TrLiteral("Text Bundle Folder"), settings.TextBundleDirName);

                EditorGUILayout.Space(5);
                EditorGUILayout.HelpBox(
                    TexMotionLocalization.TrFormat(
                        "Target Repo: {0}",
                        "https://huggingface.co/" + settings.HuggingFaceRepo),
                    MessageType.None);

                if (GUILayout.Button("🔄 " + TexMotionLocalization.TrLiteral("Reload Engine"), GUILayout.Height(24)))
                {
                    _engine?.UnloadModel();
                    EnsureEngineLoaded();
                }
            }
            EditorGUILayout.EndVertical();
        }

        private void DrawVideoMotionSettings(TexMotionSettings settings)
        {
            // Step 1: Python Runtime & Dependencies
            DrawVideoPythonEnvironmentSettings(settings);
            EditorGUILayout.Space(10);

            // Step 2: Pose & Body Models (WHAM / HMR2 / ViTPose)
            DrawVideo2MotionModelSettings(settings);
            EditorGUILayout.Space(10);

            // Step 3: Video Inference Parameters & Default Backend
            DrawVideoInferenceSettings(settings);
        }

        private void DrawVideo2MotionModelSettings(TexMotionSettings settings)
        {
            // The compact setup surface is the only normal entry point.  The
            // legacy catalog code below is retained temporarily for migration
            // of existing EditorPrefs and explicit advanced paths, but is not
            // drawn by the simplified workflow.
            DrawSimplifiedWhamSetupCard(settings);
            // A valid TexMotionSettings instance always uses the compact card.
            // Keep the legacy implementation below available for migration
            // diagnostics without leaving compiler-reported unreachable code.
            if (settings != null) return;

            // The Video2Motion catalog is an independent cache. Keep it interactive
            // even while the optional uv/venv setup card is probing or installing;
            // otherwise a stale setup flag makes every model Download button look
            // disabled although the catalog operation itself is safe to start.
            bool outerGuiEnabled = GUI.enabled;
            GUI.enabled = true;
            BeginCard();
            int readyCount = 0;
            for (int i = 0; i < VideoModelCatalog.All.Count; i++)
            {
                if (VideoModelCatalog.IsInstalled(settings, VideoModelCatalog.All[i])) readyCount++;
            }
            _showVideoModelCatalog = DrawPersistedFoldoutHeader(
                FoldoutSettingsVideoModelCatalog,
                _showVideoModelCatalog,
                "🎥 " + TexMotionLocalization.TrLiteral("Video2Motion Model Assets"),
                TexMotionLocalization.TrFormat(
                    "{0}/{1} {2}",
                    readyCount,
                    VideoModelCatalog.All.Count,
                    TexMotionLocalization.TrLiteral("ready")));

            // Keep an active download and its cancel/progress controls visible even
            // if the user collapses the catalog while the operation is running.
            if (!_showVideoModelCatalog && !_isVideoModelDownloading)
            {
                EditorGUILayout.EndVertical();
                GUI.enabled = outerGuiEnabled;
                return;
            }

            bool canDownload = !_isVideoModelDownloading && !_isGenerating && !_isVideoExtracting;
            bool previousGuiEnabled = GUI.enabled;
            EditorGUILayout.HelpBox(
                TexMotionLocalization.TrLiteral("WHAM is the focused setup path: choose the checkpoint, provision companion assets, run the preflight, then create a ViTPose feature archive when the optional image-feature stage is needed. Every fallback or missing file is reported in the result card; nothing is skipped silently."),
                MessageType.None);
            DrawWhamWorkflowCard(settings, previousGuiEnabled, canDownload);

            _showSettingsVideoAdvancedAssets = DrawPersistedFoldoutHeader(
                FoldoutSettingsVideoAdvancedAssets,
                _showSettingsVideoAdvancedAssets,
                "⚙ " + TexMotionLocalization.TrLiteral("Other Video2Motion Assets"),
                TexMotionLocalization.TrLiteral("RTMPose / MediaPipe / HMR2 / HybrIK"));
            if (_showSettingsVideoAdvancedAssets)
            {
                BeginNestedCard();
                EditorGUILayout.LabelField(
                    TexMotionLocalization.TrLiteral("Optional backends and shared cache"),
                    EditorStyles.boldLabel);
                EditorGUILayout.LabelField(
                    TexMotionLocalization.TrLiteral("These assets are kept out of the WHAM workflow until you need another backend."),
                    EditorStyles.miniLabel);
                EditorGUILayout.BeginHorizontal();
                settings.VideoModelDirectory = EditorGUILayout.TextField(
                    new GUIContent(
                        TexMotionLocalization.TrLiteral("Download Directory"),
                        TexMotionLocalization.TrLiteral("Separate cache for Video2Motion model assets.")),
                    settings.VideoModelDirectory);
                if (GUILayout.Button(TexMotionLocalization.TrLiteral("Browse..."), GUILayout.Width(75)))
                {
                    string selected = EditorUtility.OpenFolderPanel(
                        TexMotionLocalization.TrLiteral("Select Video2Motion model directory"),
                        settings.GetEffectiveVideoModelDirectory(),
                        "");
                    if (!string.IsNullOrEmpty(selected)) settings.VideoModelDirectory = selected;
                }
                EditorGUILayout.EndHorizontal();
                EditorGUILayout.LabelField(
                    TexMotionLocalization.TrFormat(
                        "Resolved directory: {0}",
                        settings.GetEffectiveVideoModelDirectory()),
                    EditorStyles.miniLabel);

                string selectedRtmposePath = settings.GetEffectiveRTMPoseModelPath();
                EditorGUILayout.BeginHorizontal();
                EditorGUILayout.LabelField(TexMotionLocalization.TrLiteral("Active RTMPose/DWPose file"), GUILayout.Width(175));
                EditorGUILayout.SelectableLabel(selectedRtmposePath, EditorStyles.textField, GUILayout.Height(18));
                if (GUILayout.Button(TexMotionLocalization.TrLiteral("Browse..."), GUILayout.Width(75)))
                {
                    string selected = EditorUtility.OpenFilePanel(TexMotionLocalization.TrLiteral("Select RTMPose/DWPose ONNX model"), "", "onnx");
                    if (!string.IsNullOrEmpty(selected))
                    {
                        settings.RTMPoseModelPath = selected;
                        settings.Save();
                    }
                }
                EditorGUILayout.EndHorizontal();

                EditorGUILayout.Space(4);
            for (int i = 0; i < VideoModelCatalog.All.Count; i++)
            {
                VideoModelDefinition model = VideoModelCatalog.All[i];
                if (model.Kind == VideoModelKind.WHAM) continue;
                bool installed = VideoModelCatalog.IsInstalled(settings, model);
                string sizeLabel = VideoModelCatalog.FormatBytes(model.EstimatedBytes);

                BeginNestedRow();
                EditorGUILayout.LabelField(model.DisplayName, GUILayout.Width(205));
                EditorGUILayout.LabelField(sizeLabel, EditorStyles.miniLabel, GUILayout.Width(58));
                GUI.color = installed ? new Color(0.35f, 0.95f, 0.50f) : new Color(1.0f, 0.72f, 0.25f);
                EditorGUILayout.LabelField(
                    installed ? "● " + TexMotionLocalization.TrLiteral("Ready") :
                        model.IsManualProvisioning ? "○ " + TexMotionLocalization.TrLiteral("Manual") :
                        "○ " + TexMotionLocalization.TrLiteral("Missing"),
                    EditorStyles.miniLabel,
                    GUILayout.Width(65));
                GUI.color = Color.white;

                GUI.enabled = previousGuiEnabled && canDownload && model.CanDownload;
                if (GUILayout.Button(TexMotionLocalization.TrLiteral(installed ? "Redownload" : "Download"), EditorStyles.miniButton, GUILayout.Width(78)))
                {
                    StartVideoModelDownload(model.Id, settings);
                }

                if (model.CanSelectForRuntime)
                {
                    GUI.enabled = previousGuiEnabled && installed && canDownload;
                    if (GUILayout.Button(TexMotionLocalization.TrLiteral("Use"), EditorStyles.miniButton, GUILayout.Width(38)))
                    {
                        ApplyVideoModelSelection(settings, model);
                        _statusMessage = TexMotionLocalization.TrFormat("Selected {0} for Video Motion inference.", model.DisplayName);
                    }
                }

                GUI.enabled = previousGuiEnabled;
                if (GUILayout.Button(TexMotionLocalization.TrLiteral("Source"), EditorStyles.miniButton, GUILayout.Width(52)))
                {
                    Application.OpenURL(string.IsNullOrWhiteSpace(model.ReferenceUrl) ? model.RemoteUrl : model.ReferenceUrl);
                }
                VideoAssetGuide modelGuide = VideoModelCatalog.GetAssetGuide(model);
                if (modelGuide != null && GUILayout.Button(TexMotionLocalization.TrLiteral("Guide"), EditorStyles.miniButton, GUILayout.Width(52)))
                {
                    VideoAssetGuideWindow.Show(model, settings);
                }
                EditorGUILayout.EndHorizontal();

                if (!string.IsNullOrWhiteSpace(model.Description))
                {
                    string prefix = model.IsExperimental ? TexMotionLocalization.TrLiteral("Experimental: ") : "";
                    EditorGUILayout.LabelField("  " + prefix + TexMotionLocalization.TrLiteral(model.Description), EditorStyles.miniLabel);
                }
            }
            DrawQualityRuntimeAssetCatalog(settings, previousGuiEnabled, canDownload);
            GUI.enabled = previousGuiEnabled && canDownload;
            string totalSize = VideoModelCatalog.FormatBytes(VideoModelCatalog.EstimatedTotalBytes);
            if (GUILayout.Button(
                "📥 " + TexMotionLocalization.TrFormat("Download All Video2Motion Models (estimated {0})", totalSize),
                GUILayout.Height(30)))
            {
                StartAllVideoModelDownload(settings);
            }
            GUI.enabled = previousGuiEnabled;
                EditorGUILayout.EndVertical();
            }

            // Keep status visible after a collapsed section so a running or
            // failed download never leaves the user guessing what happened.

            if (_isVideoModelDownloading)
            {
                EditorGUI.ProgressBar(
                    EditorGUILayout.GetControlRect(false, 18),
                    _videoModelDownloadProgress,
                    _videoModelDownloadStatus);
                if (GUILayout.Button(TexMotionLocalization.TrLiteral("Cancel Video2Motion model download"), EditorStyles.miniButton, GUILayout.Height(22)))
                {
                    CancelVideoModelDownload();
                }
            }
            else if (!string.IsNullOrEmpty(_videoModelDownloadStatus))
            {
                MessageType statusType = string.IsNullOrEmpty(_videoModelDownloadError) ? MessageType.Info : MessageType.Error;
                EditorGUILayout.HelpBox(
                    _videoModelDownloadStatus +
                    (string.IsNullOrEmpty(_videoModelDownloadError) ? "" : "\n" + _videoModelDownloadError),
                    statusType);
            }

            EditorGUILayout.EndVertical();
            GUI.enabled = outerGuiEnabled;
        }

        /// <summary>
        /// Compact Video2Motion setup surface.  WHAM is presented as one
        /// workflow with required, optional, and diagnostic sections; detailed
        /// model-definition paths stay behind the advanced foldout.
        /// </summary>
        private void DrawSimplifiedWhamSetupCard(TexMotionSettings settings)
        {
            bool previousGuiEnabled = GUI.enabled;
            GUI.enabled = true;
            BeginCard();

            VideoPoseBackend backend = settings != null ? settings.VideoBackend : VideoPoseBackend.Auto;
            VideoWhamSetupSummary whamSummary = IsWhamBackend(backend)
                ? VideoModelCatalog.GetWhamSetupSummary(settings, backend)
                : null;
            string summary = whamSummary != null
                ? GetWhamSetupSummaryLabel(whamSummary)
                : TexMotionLocalization.TrLiteral("Choose a backend to see its local assets");
            _showVideoModelCatalog = DrawPersistedFoldoutHeader(
                FoldoutSettingsVideoModelCatalog,
                _showVideoModelCatalog,
                "🎥 " + TexMotionLocalization.TrLiteral("Video2Motion Setup"),
                summary);

            if (_showVideoModelCatalog || _isVideoModelDownloading)
            {
                bool canDownload = !_isVideoModelDownloading && !_isGenerating && !_isVideoExtracting;
                EditorGUILayout.LabelField(
                    TexMotionLocalization.TrLiteral("Choose the backend in Video Motion. Prepare its local assets here, then run extraction offline."),
                    EditorStyles.miniLabel);
                if (whamSummary != null)
                    DrawSimplifiedWhamSetupContent(settings, whamSummary, previousGuiEnabled, canDownload);
                else
                    DrawCompactBackendSetupContent(settings, backend, previousGuiEnabled, canDownload);
                DrawVideoModelDownloadStatus();
            }

            EditorGUILayout.EndVertical();
            GUI.enabled = previousGuiEnabled;
        }

        private void DrawSimplifiedWhamSetupContent(
            TexMotionSettings settings,
            VideoWhamSetupSummary summary,
            bool previousGuiEnabled,
            bool canDownload)
        {
            BeginNestedCard();
            DrawWhamSetupStatus(summary);

            bool primaryEnabled = previousGuiEnabled && canDownload;
            GUI.enabled = primaryEnabled;
            string primaryLabel = summary.IsReady
                ? TexMotionLocalization.TrLiteral("Run WHAM preflight")
                : summary.State == VideoWhamSetupState.Manual || summary.State == VideoWhamSetupState.Incompatible
                    ? TexMotionLocalization.TrLiteral("Open manual setup guide")
                    : TexMotionLocalization.TrLiteral("Download missing WHAM assets");
            if (GUILayout.Button(primaryLabel, _primaryActionStyle, GUILayout.Height(32f)))
            {
                if (summary.IsReady)
                    RunVideoRuntimePreflight(settings);
                else if (summary.State == VideoWhamSetupState.Manual || summary.State == VideoWhamSetupState.Incompatible)
                    OpenFirstWhamGuide(summary, settings);
                else
                    StartWhamSetupDownload(settings);
            }
            GUI.enabled = primaryEnabled;
            if (GUILayout.Button(TexMotionLocalization.TrLiteral("Recheck"), EditorStyles.miniButton, GUILayout.Height(24f)))
                RunVideoRuntimePreflight(settings);
            GUI.enabled = previousGuiEnabled;

            EditorGUILayout.Space(5f);
            EditorGUILayout.LabelField(TexMotionLocalization.TrLiteral("Required"), EditorStyles.boldLabel);
            DrawWhamSetupGroup(
                summary,
                TexMotionLocalization.TrLiteral("Python environment"),
                TexMotionLocalization.TrLiteral("The local runner is started only during extraction; CUDA is optional."),
                previousGuiEnabled,
                canDownload,
                "python-environment");
            DrawWhamSetupGroup(
                summary,
                TexMotionLocalization.TrLiteral("WHAM temporal model"),
                TexMotionLocalization.TrLiteral("Official WHAM temporal 3D checkpoint and adapter."),
                previousGuiEnabled,
                canDownload,
                "wham",
                "wham-adapter");
            DrawWhamSetupGroup(
                summary,
                TexMotionLocalization.TrLiteral("HMR2 image feature runner"),
                TexMotionLocalization.TrLiteral("HMR2 ViT image features and the initial SMPL estimate used by official WHAM."),
                previousGuiEnabled,
                canDownload,
                "hmr2",
                "hmr2-runtime",
                "hmr2-adapter",
                "hmr2-smpl-body");
            DrawWhamSetupGroup(
                summary,
                TexMotionLocalization.TrLiteral("ViTPose 2D detector"),
                TexMotionLocalization.TrLiteral("2D keypoint detection for WHAM preprocessing. This is separate from HMR2 image features."),
                previousGuiEnabled,
                canDownload,
                "wham-image-feature-backbone");
            if (summary.Backend == VideoPoseBackend.WHAMMediaPipe)
            {
                DrawWhamSetupGroup(
                    summary,
                    TexMotionLocalization.TrLiteral("MediaPipe auxiliary task"),
                    TexMotionLocalization.TrLiteral("Visibility and 3D observations used only by WHAM + MediaPipe fusion."),
                    previousGuiEnabled,
                    canDownload,
                    GetMediaPipeCatalogId(settings));
            }

            _showSettingsWhamOptional = DrawPersistedFoldoutHeader(
                FoldoutSettingsWhamOptional,
                _showSettingsWhamOptional,
                TexMotionLocalization.TrLiteral("Optional"),
                TexMotionLocalization.TrLiteral("Camera and research stages"));
            if (_showSettingsWhamOptional)
            {
                EditorGUILayout.LabelField(
                    TexMotionLocalization.TrLiteral("These assets improve camera/world motion or preserve an offline cache. Basic WHAM extraction can start without them."),
                    EditorStyles.miniLabel);
                DrawWhamSetupGroup(summary, TexMotionLocalization.TrLiteral("DPVO camera motion"),
                    TexMotionLocalization.TrLiteral("World camera motion; without it the result stays camera-relative."),
                    previousGuiEnabled, canDownload, "wham-dpvo");
                DrawWhamSetupGroup(summary, TexMotionLocalization.TrLiteral("Camera calibration"),
                    TexMotionLocalization.TrLiteral("Video-specific intrinsics for projection."),
                    previousGuiEnabled, canDownload, "wham-camera");
                DrawWhamSetupGroup(summary, TexMotionLocalization.TrLiteral("Precomputed image feature archive"),
                    TexMotionLocalization.TrLiteral("Optional per-video cache for an offline feature stage."),
                    previousGuiEnabled, canDownload, "wham-image-feature-archive");
                DrawWhamSetupGroup(summary, TexMotionLocalization.TrLiteral("SMPL/SMPL-X body model"),
                    TexMotionLocalization.TrLiteral("Licensed body data can be supplied when a local runtime requires it."),
                    previousGuiEnabled, canDownload, "wham-smplx-body");
            }

            _showSettingsVideoAdvancedAssets = DrawPersistedFoldoutHeader(
                FoldoutSettingsVideoAdvancedAssets,
                _showSettingsVideoAdvancedAssets,
                TexMotionLocalization.TrLiteral("Advanced"),
                TexMotionLocalization.TrLiteral("Custom runtimes only"));
            if (_showSettingsVideoAdvancedAssets)
            {
                EditorGUILayout.HelpBox(
                    TexMotionLocalization.TrLiteral("Most users do not need these fields. Use them only when a licensed runtime or external cache is outside the default model directory."),
                    MessageType.Info);
                DrawWhamCompanionPathSettings(settings);
            }

            DrawWhamSetupDiagnostics(summary);
            EditorGUILayout.EndVertical();
        }

        private void DrawWhamSetupStatus(VideoWhamSetupSummary summary)
        {
            if (summary == null) return;
            string state = GetWhamSetupStateLabel(summary.State);
            Color stateColor = GetWhamSetupStateColor(summary.State);
            EditorGUILayout.BeginHorizontal();
            EditorGUILayout.LabelField(TexMotionLocalization.TrLiteral("Official WHAM"), EditorStyles.boldLabel);
            Color previous = GUI.color;
            GUI.color = stateColor;
            GUILayout.Label("● " + state, _badgeStyle, GUILayout.Height(22f), GUILayout.ExpandWidth(false));
            GUI.color = previous;
            EditorGUILayout.EndHorizontal();

            string detail = summary.IsReady
                ? TexMotionLocalization.TrLiteral("Ready to extract offline.")
                : TexMotionLocalization.TrFormat(
                    "{0} required item(s) need attention before official inference can run.",
                    summary.MissingRequiredCount);
            EditorGUILayout.HelpBox(detail, summary.IsReady ? MessageType.Info : MessageType.Warning);
        }

        private void DrawWhamSetupGroup(
            VideoWhamSetupSummary summary,
            string label,
            string description,
            bool previousGuiEnabled,
            bool canDownload,
            params string[] ids)
        {
            var statuses = new List<VideoAssetStatusInfo>();
            if (summary != null && summary.RequiredAssets != null)
            {
                for (int i = 0; i < ids.Length; i++)
                {
                    if (string.Equals(ids[i], "python-environment", StringComparison.OrdinalIgnoreCase))
                    {
                        for (int j = 0; j < summary.RequiredAssets.Count; j++)
                        {
                            VideoAssetStatusInfo runtimeStatus = summary.RequiredAssets[j];
                            if (runtimeStatus != null &&
                                (runtimeStatus.Role == VideoAssetRole.PythonRequirements ||
                                 runtimeStatus.Role == VideoAssetRole.PythonRuntime))
                                statuses.Add(runtimeStatus);
                        }
                        continue;
                    }
                    VideoAssetStatusInfo status = FindWhamSetupStatus(summary.RequiredAssets, ids[i]);
                    if (status != null) statuses.Add(status);
                }
            }
            if (summary != null && summary.OptionalAssets != null && statuses.Count == 0)
            {
                for (int i = 0; i < ids.Length; i++)
                {
                    VideoAssetStatusInfo status = FindWhamSetupStatus(summary.OptionalAssets, ids[i]);
                    if (status != null) statuses.Add(status);
                }
            }
            if (statuses.Count == 0) return;

            bool ready = true;
            VideoAssetStatusInfo actionStatus = null;
            VideoAssetStatus worst = VideoAssetStatus.Ready;
            for (int i = 0; i < statuses.Count; i++)
            {
                VideoAssetStatusInfo status = statuses[i];
                if (status == null || status.IsReady) continue;
                ready = false;
                if (actionStatus == null) actionStatus = status;
                if (status.Status == VideoAssetStatus.Incompatible) worst = VideoAssetStatus.Incompatible;
                else if (worst != VideoAssetStatus.Incompatible && status.Status == VideoAssetStatus.Manual) worst = VideoAssetStatus.Manual;
                else if (worst == VideoAssetStatus.Ready) worst = status.Status;
            }

            BeginNestedCard();
            EditorGUILayout.BeginHorizontal();
            EditorGUILayout.LabelField(label, EditorStyles.boldLabel);
            Color previous = GUI.color;
            GUI.color = GetWhamSetupStateColor(ready ? VideoWhamSetupState.Ready :
                worst == VideoAssetStatus.Incompatible ? VideoWhamSetupState.Incompatible :
                worst == VideoAssetStatus.Manual ? VideoWhamSetupState.Manual : VideoWhamSetupState.Missing);
            GUILayout.Label(
                "● " + GetWhamSetupStateLabel(ready ? VideoWhamSetupState.Ready :
                    worst == VideoAssetStatus.Incompatible ? VideoWhamSetupState.Incompatible :
                    worst == VideoAssetStatus.Manual ? VideoWhamSetupState.Manual : VideoWhamSetupState.Missing),
                _badgeStyle,
                GUILayout.Height(20f),
                GUILayout.ExpandWidth(false));
            GUI.color = previous;
            EditorGUILayout.EndHorizontal();
            EditorGUILayout.LabelField(description, EditorStyles.miniLabel);

            if (!ready && actionStatus != null)
            {
                bool incompatible = actionStatus.Status == VideoAssetStatus.Incompatible || worst == VideoAssetStatus.Incompatible;
                EditorGUILayout.LabelField(
                    string.IsNullOrWhiteSpace(actionStatus.Reason) ? TexMotionLocalization.TrLiteral("Setup is required.") : actionStatus.Reason,
                    EditorStyles.wordWrappedMiniLabel);
                EditorGUILayout.BeginVertical();
                GUI.enabled = previousGuiEnabled && canDownload;
                VideoModelDefinition asset = VideoModelCatalog.Find(actionStatus.Id);
                if (asset != null && incompatible)
                {
                    if (asset.AssetRole == VideoAssetRole.WhamCamera &&
                        GUILayout.Button(TexMotionLocalization.TrLiteral("Auto-configure from Video / Standard FOV"), EditorStyles.miniButton, GUILayout.Height(22f)))
                    {
                        string outPath;
                        VideoModelCatalog.AutoConfigureWhamCameraFromVideo(
                            TexMotionSettings.instance,
                            _videoPath,
                            _detectedVideoWidth,
                            _detectedVideoHeight,
                            out outPath);
                        RunVideoRuntimePreflight(TexMotionSettings.instance);
                        Repaint();
                    }
                    if (GUILayout.Button(TexMotionLocalization.TrLiteral("Choose compatible file"), EditorStyles.miniButton, GUILayout.Height(22f)))
                        HandleIncompatibleWhamAssetAction(TexMotionSettings.instance, asset, actionStatus);
                    if (GUILayout.Button(TexMotionLocalization.TrLiteral("Guide"), EditorStyles.miniButton, GUILayout.Height(22f)))
                        VideoAssetGuideWindow.Show(asset, TexMotionSettings.instance);
                    EditorGUILayout.HelpBox(
                        TexMotionLocalization.TrLiteral("Incompatible templates cannot be fixed by Download. Choose a compatible local file, then run Recheck."),
                        MessageType.Warning);
                }
                else if (asset != null)
                {
                    if (asset.CanDownload && GUILayout.Button(TexMotionLocalization.TrLiteral("Download"), EditorStyles.miniButton, GUILayout.Height(22f)))
                        StartVideoModelDownload(asset.Id, TexMotionSettings.instance);
                    else if (GUILayout.Button(TexMotionLocalization.TrLiteral("Guide"), EditorStyles.miniButton, GUILayout.Height(22f)))
                        VideoAssetGuideWindow.Show(asset, TexMotionSettings.instance);
                }
                if (actionStatus.Id == "python-runtime" || actionStatus.Id.IndexOf("python", StringComparison.OrdinalIgnoreCase) >= 0)
                {
                    if (GUILayout.Button(TexMotionLocalization.TrLiteral("Install environment"), EditorStyles.miniButton, GUILayout.Height(22f)))
                    {
                        TexMotionSettings.instance.VideoPythonEnvironmentProfile = PythonDependencyProfile.PyTorch;
                        BuildVideoVenvAsync(TexMotionSettings.instance);
                    }
                }
                GUI.enabled = previousGuiEnabled;
                EditorGUILayout.EndVertical();
            }
            EditorGUILayout.EndVertical();
        }

        private void DrawWhamSetupDiagnostics(VideoWhamSetupSummary summary)
        {
            if (summary == null) return;
            EditorGUILayout.Space(5f);
            EditorGUILayout.LabelField(TexMotionLocalization.TrLiteral("Diagnostic"), EditorStyles.boldLabel);
            EditorGUILayout.LabelField(
                TexMotionLocalization.TrFormat("Official WHAM: {0}", summary.OfficialRunnerStatus),
                EditorStyles.miniLabel);
            EditorGUILayout.LabelField(
                TexMotionLocalization.TrFormat("HMR2 image feature runner: {0}", summary.Hmr2ImageFeaturesStatus),
                EditorStyles.miniLabel);
            EditorGUILayout.LabelField(
                TexMotionLocalization.TrFormat("ViTPose 2D detector: {0}", summary.VitPose2DStatus),
                EditorStyles.miniLabel);
            if (summary.OptionalUnavailableCount > 0)
            {
                EditorGUILayout.LabelField(
                    TexMotionLocalization.TrFormat(
                        "{0} optional stage(s) are unavailable; basic extraction remains possible.",
                        summary.OptionalUnavailableCount),
                    EditorStyles.miniLabel);
            }
        }

        private void DrawCompactBackendSetupContent(
            TexMotionSettings settings,
            VideoPoseBackend backend,
            bool previousGuiEnabled,
            bool canDownload)
        {
            BeginNestedCard();
            EditorGUILayout.LabelField(GetVideoBackendDisplayName(backend), EditorStyles.boldLabel);
            EditorGUILayout.LabelField(
                TexMotionLocalization.TrLiteral("Download or select the model for the backend chosen in Video Motion."),
                EditorStyles.miniLabel);
            IReadOnlyList<VideoAssetStatusInfo> statuses = VideoModelCatalog.GetRuntimeAssetStatuses(settings, backend);
            for (int i = 0; i < statuses.Count; i++)
            {
                VideoAssetStatusInfo status = statuses[i];
                if (status == null || status.Role == VideoAssetRole.PythonRequirements || status.Role == VideoAssetRole.PythonRuntime) continue;
                DrawCompactBackendAssetRow(settings, status, previousGuiEnabled, canDownload);
            }
            EditorGUILayout.EndVertical();
        }

        private void DrawCompactBackendAssetRow(
            TexMotionSettings settings,
            VideoAssetStatusInfo status,
            bool previousGuiEnabled,
            bool canDownload)
        {
            BeginNestedCard();
            EditorGUILayout.BeginHorizontal();
            EditorGUILayout.LabelField(status.DisplayName, EditorStyles.boldLabel);
            EditorGUILayout.LabelField(status.Status.ToString(), EditorStyles.miniLabel, GUILayout.ExpandWidth(false));
            EditorGUILayout.EndHorizontal();
            if (!status.IsReady)
                EditorGUILayout.LabelField(status.Reason ?? string.Empty, EditorStyles.wordWrappedMiniLabel);

            EditorGUILayout.BeginVertical();
            GUI.enabled = previousGuiEnabled && canDownload;
            VideoModelDefinition asset = VideoModelCatalog.Find(status.Id);
            if (asset != null && asset.CanDownload && GUILayout.Button(TexMotionLocalization.TrLiteral("Download"), EditorStyles.miniButton, GUILayout.Height(22f)))
                StartVideoModelDownload(asset.Id, settings);
            if (asset != null && asset.HasSetupGuide && GUILayout.Button(TexMotionLocalization.TrLiteral("Guide"), EditorStyles.miniButton, GUILayout.Height(22f)))
                VideoAssetGuideWindow.Show(asset, settings);
            GUI.enabled = previousGuiEnabled;
            EditorGUILayout.EndVertical();
            EditorGUILayout.EndVertical();
        }

        private void DrawVideoModelDownloadStatus()
        {
            if (_isVideoModelDownloading)
            {
                EditorGUI.ProgressBar(EditorGUILayout.GetControlRect(false, 18f), _videoModelDownloadProgress, _videoModelDownloadStatus);
                if (GUILayout.Button(TexMotionLocalization.TrLiteral("Cancel download"), EditorStyles.miniButton, GUILayout.Height(22f)))
                    CancelVideoModelDownload();
            }
            else if (!string.IsNullOrWhiteSpace(_videoModelDownloadStatus))
            {
                MessageType type = string.IsNullOrWhiteSpace(_videoModelDownloadError) ? MessageType.Info : MessageType.Error;
                EditorGUILayout.HelpBox(
                    _videoModelDownloadStatus +
                    (string.IsNullOrWhiteSpace(_videoModelDownloadError) ? string.Empty : "\n" + _videoModelDownloadError),
                    type);
            }
        }

        private void OpenFirstWhamGuide(VideoWhamSetupSummary summary, TexMotionSettings settings)
        {
            if (summary == null) return;
            IReadOnlyList<VideoAssetStatusInfo> assets = summary.RequiredAssets;
            for (int i = 0; i < assets.Count; i++)
            {
                VideoAssetStatusInfo status = assets[i];
                if (status == null || status.IsReady) continue;
                VideoModelDefinition asset = VideoModelCatalog.Find(status.Id);
                if (asset != null && asset.HasSetupGuide)
                {
                    VideoAssetGuideWindow.Show(asset, settings);
                    return;
                }
            }
            _statusMessage = TexMotionLocalization.TrLiteral("No manual guide is available for the selected WHAM state.");
        }

        private static VideoAssetStatusInfo FindWhamSetupStatus(
            IReadOnlyList<VideoAssetStatusInfo> statuses,
            string id)
        {
            if (statuses == null || string.IsNullOrWhiteSpace(id)) return null;
            for (int i = 0; i < statuses.Count; i++)
            {
                if (statuses[i] != null && string.Equals(statuses[i].Id, id, StringComparison.OrdinalIgnoreCase))
                    return statuses[i];
            }
            return null;
        }

        private static string GetWhamSetupSummaryLabel(VideoWhamSetupSummary summary)
        {
            if (summary == null) return string.Empty;
            if (summary.IsReady) return TexMotionLocalization.TrLiteral("Ready");
            if (summary.State == VideoWhamSetupState.Manual) return TexMotionLocalization.TrLiteral("Manual setup needed");
            if (summary.State == VideoWhamSetupState.Incompatible) return TexMotionLocalization.TrLiteral("Compatibility error");
            return TexMotionLocalization.TrFormat("{0} required item(s) missing", summary.MissingRequiredCount);
        }

        private static string GetWhamSetupStateLabel(VideoWhamSetupState state)
        {
            switch (state)
            {
                case VideoWhamSetupState.Ready: return TexMotionLocalization.TrLiteral("Ready");
                case VideoWhamSetupState.Manual: return TexMotionLocalization.TrLiteral("Manual setup needed");
                case VideoWhamSetupState.Incompatible: return TexMotionLocalization.TrLiteral("Compatibility error");
                case VideoWhamSetupState.Checking: return TexMotionLocalization.TrLiteral("Checking");
                default: return TexMotionLocalization.TrLiteral("Missing");
            }
        }

        private static Color GetWhamSetupStateColor(VideoWhamSetupState state)
        {
            switch (state)
            {
                case VideoWhamSetupState.Ready: return new Color(0.35f, 0.95f, 0.55f);
                case VideoWhamSetupState.Incompatible: return new Color(1.0f, 0.45f, 0.35f);
                case VideoWhamSetupState.Manual: return new Color(1.0f, 0.72f, 0.25f);
                default: return new Color(0.95f, 0.75f, 0.25f);
            }
        }

        private static string GetMediaPipeCatalogId(TexMotionSettings settings)
        {
            int complexity = settings == null ? 2 : Mathf.Clamp(settings.VideoModelComplexity, 0, 2);
            return complexity <= 0 ? "mediapipe-lite" : complexity == 1 ? "mediapipe-full" : "mediapipe-heavy";
        }

        private void DrawWhamWorkflowCard(TexMotionSettings settings, bool previousGuiEnabled, bool canDownload)
        {
            if (settings == null) return;
            BeginNestedCard();
            VideoModelDefinition whamModel = VideoModelCatalog.Find("wham");
            bool checkpointReady = whamModel != null && VideoModelCatalog.IsInstalled(settings, whamModel);
            int missingCount = 0;
            try { missingCount = VideoModelCatalog.GetMissingWhamAssetIds(settings).Count; }
            catch { }
            string summary = !checkpointReady
                ? TexMotionLocalization.TrLiteral("Checkpoint required")
                : missingCount > 0
                    ? TexMotionLocalization.TrFormat("{0} companion assets need attention", missingCount)
                    : TexMotionLocalization.TrLiteral("WHAM ready");
            _showSettingsWham = DrawPersistedFoldoutHeader(
                FoldoutSettingsWham,
                _showSettingsWham,
                "🧍 " + TexMotionLocalization.TrLiteral("WHAM Setup"),
                summary);
            if (_showSettingsWham)
            {
                EditorGUILayout.HelpBox(
                    TexMotionLocalization.TrLiteral("1. Install or browse the WHAM checkpoint and adapter.  2. Download or provide the companion assets.  3. Run Runtime Preflight.  4. Create a ViTPose feature archive only when the image-feature stage needs it."),
                    MessageType.Info);
                DrawWhamCoreModelRow(settings, whamModel, previousGuiEnabled, canDownload);
                DrawWhamCompanionAssetRows(settings, previousGuiEnabled, canDownload);
                DrawWhamCompanionPathSettings(settings);
                DrawVideoRuntimePreflight(settings);
            }
            EditorGUILayout.EndVertical();
        }

        private void DrawWhamCoreModelRow(
            TexMotionSettings settings,
            VideoModelDefinition model,
            bool previousGuiEnabled,
            bool canDownload)
        {
            if (settings == null || model == null) return;
            bool installed = VideoModelCatalog.IsInstalled(settings, model);
            long estimatedBytes = model.EstimatedBytes + VideoModelCatalog.WhamAdapterEstimatedBytes;
            BeginNestedRow();
            EditorGUILayout.LabelField(
                TexMotionLocalization.TrLiteral("WHAM checkpoint + adapter"),
                GUILayout.Width(205));
            EditorGUILayout.LabelField(VideoModelCatalog.FormatBytes(estimatedBytes), EditorStyles.miniLabel, GUILayout.Width(62));
            GUI.color = installed ? new Color(0.35f, 0.95f, 0.50f) : new Color(1.0f, 0.72f, 0.25f);
            EditorGUILayout.LabelField(
                installed ? "● " + TexMotionLocalization.TrLiteral("Ready") : "○ " + TexMotionLocalization.TrLiteral("Missing"),
                EditorStyles.miniLabel,
                GUILayout.Width(70));
            GUI.color = Color.white;
            GUI.enabled = previousGuiEnabled && canDownload && model.CanDownload;
            if (GUILayout.Button(
                TexMotionLocalization.TrLiteral(installed ? "Redownload" : "Download"),
                EditorStyles.miniButton,
                GUILayout.Width(82)))
            {
                StartVideoModelDownload(model.Id, settings);
            }
            GUI.enabled = previousGuiEnabled && installed && canDownload;
            if (GUILayout.Button(TexMotionLocalization.TrLiteral("Use"), EditorStyles.miniButton, GUILayout.Width(42)))
            {
                ApplyVideoModelSelection(settings, model);
                _statusMessage = TexMotionLocalization.TrFormat("Selected {0} for Video Motion inference.", model.DisplayName);
            }
            GUI.enabled = previousGuiEnabled;
            if (GUILayout.Button(TexMotionLocalization.TrLiteral("Source"), EditorStyles.miniButton, GUILayout.Width(52)))
                Application.OpenURL(string.IsNullOrWhiteSpace(model.ReferenceUrl) ? model.RemoteUrl : model.ReferenceUrl);
            VideoAssetGuide guide = VideoModelCatalog.GetAssetGuide(model);
            if (guide != null && GUILayout.Button(TexMotionLocalization.TrLiteral("Guide"), EditorStyles.miniButton, GUILayout.Width(52)))
                VideoAssetGuideWindow.Show(model, settings);
            EditorGUILayout.EndHorizontal();
            EditorGUILayout.LabelField(
                "  " + TexMotionLocalization.TrLiteral(model.Description),
                EditorStyles.miniLabel);
            EditorGUILayout.LabelField(
                "  " + TexMotionLocalization.TrLiteral("The adapter is installed together with this checkpoint."),
                EditorStyles.miniLabel);
        }

        private void DrawWhamCompanionAssetRows(TexMotionSettings settings, bool previousGuiEnabled, bool canDownload)
        {
            if (settings == null) return;
            EditorGUILayout.Space(3);
            EditorGUILayout.LabelField(
                TexMotionLocalization.TrLiteral("WHAM companion assets"),
                EditorStyles.boldLabel);
            EditorGUILayout.LabelField(
                TexMotionLocalization.TrLiteral("Downloadable ViTPose/DPVO files and camera templates are listed here. Licensed body data and precomputed archives use Browse."),
                EditorStyles.miniLabel);
            for (int i = 0; i < VideoModelCatalog.WhamAssets.Count; i++)
            {
                VideoModelDefinition asset = VideoModelCatalog.WhamAssets[i];
                if (asset.AssetRole == VideoAssetRole.WhamAdapter || asset.AssetRole == VideoAssetRole.WhamManifest)
                    continue;
                VideoAssetStatusInfo assetStatus = VideoModelCatalog.GetAssetStatus(settings, asset, asset.RequiredForFullWhamParity);
                bool installed = assetStatus != null && assetStatus.Status == VideoAssetStatus.Ready;
                bool incompatible = assetStatus != null && assetStatus.Status == VideoAssetStatus.Incompatible;
                bool isRunnerTemplate = asset.AssetRole == VideoAssetRole.WhamImageFeatureModelDefinition ||
                    asset.AssetRole == VideoAssetRole.WhamImageFeatureConfig;
                BeginNestedRow();
                EditorGUILayout.LabelField(asset.DisplayName, GUILayout.Width(230));
                GUI.color = installed ? new Color(0.35f, 0.95f, 0.50f) : new Color(1.0f, 0.72f, 0.25f);
                string companionStatus = installed
                    ? "● " + TexMotionLocalization.TrLiteral(isRunnerTemplate ? "Template" : "Ready")
                    : assetStatus != null && assetStatus.Status == VideoAssetStatus.Incompatible
                        ? "! " + TexMotionLocalization.TrLiteral("Incompatible")
                        : (asset.IsManualProvisioning
                            ? "○ " + TexMotionLocalization.TrLiteral("Manual")
                            : "○ " + TexMotionLocalization.TrLiteral("Missing"));
                EditorGUILayout.LabelField(companionStatus, EditorStyles.miniLabel, GUILayout.Width(78));
                GUI.color = Color.white;
                GUI.enabled = previousGuiEnabled && canDownload;
                string actionLabel = incompatible
                    ? TexMotionLocalization.TrLiteral("Choose compatible file")
                    : asset.IsManualProvisioning
                    ? TexMotionLocalization.TrLiteral("Browse")
                    : asset.AssetRole == VideoAssetRole.WhamCamera
                        ? TexMotionLocalization.TrLiteral(installed ? "Reinstall" : "Install Template")
                        : TexMotionLocalization.TrLiteral(installed ? "Redownload" : "Download");
                GUILayoutOption actionWidth = incompatible
                    ? GUILayout.MinWidth(132f)
                    : GUILayout.Width(92f);
                if (GUILayout.Button(actionLabel, EditorStyles.miniButton, actionWidth))
                {
                    if (incompatible)
                    {
                        HandleIncompatibleWhamAssetAction(settings, asset, assetStatus);
                    }
                    else if (!asset.IsManualProvisioning)
                    {
                        StartVideoModelDownload(asset.Id, settings);
                    }
                    else
                    {
                        string selected = EditorUtility.OpenFilePanel(
                            TexMotionLocalization.TrFormat("Provision {0}", asset.DisplayName),
                            settings.GetEffectiveVideoModelDirectory(),
                            "");
                        if (!string.IsNullOrEmpty(selected))
                        {
                            string destination = VideoModelCatalog.GetWhamAssetPath(settings, asset);
                            string directory = Path.GetDirectoryName(destination);
                            if (!string.IsNullOrEmpty(directory)) Directory.CreateDirectory(directory);
                            try
                            {
                                if (!string.Equals(Path.GetFullPath(selected), Path.GetFullPath(destination), StringComparison.OrdinalIgnoreCase))
                                    File.Copy(selected, destination, true);
                            }
                            catch (Exception ex)
                            {
                                Debug.LogWarning($"[TexMotion] Asset copy failed: {ex.Message}");
                            }
                            settings.Save();
                        }
                    }
                }
                GUI.enabled = previousGuiEnabled;
                if (GUILayout.Button(TexMotionLocalization.TrLiteral("Source"), EditorStyles.miniButton, GUILayout.Width(52)))
                    Application.OpenURL(asset.ReferenceUrl);
                VideoAssetGuide companionGuide = VideoModelCatalog.GetAssetGuide(asset);
                if (companionGuide != null && GUILayout.Button(TexMotionLocalization.TrLiteral("Guide"), EditorStyles.miniButton, GUILayout.Width(52)))
                    VideoAssetGuideWindow.Show(asset, settings);
                EditorGUILayout.EndHorizontal();
                EditorGUILayout.LabelField("  " + TexMotionLocalization.TrLiteral(asset.Description), EditorStyles.miniLabel);
                if (incompatible)
                {
                    EditorGUILayout.HelpBox(
                        TexMotionLocalization.TrLiteral("Incompatible templates cannot be fixed by Download. Choose a compatible local file, then run Recheck."),
                        MessageType.Warning);
                }
                if (assetStatus != null && !installed &&
                    (assetStatus.Status == VideoAssetStatus.Incompatible || assetStatus.Status == VideoAssetStatus.Missing || assetStatus.Status == VideoAssetStatus.Manual))
                {
                    EditorGUILayout.LabelField("  " + assetStatus.ToString(), EditorStyles.miniLabel);
                }
            }
            EditorGUILayout.LabelField(
                TexMotionLocalization.TrFormat(
                    "Contract: {0}",
                    VideoModelCatalog.GetWhamAssetContractSummary(settings)),
                EditorStyles.miniLabel);
        }

        private void DrawVideoRuntimePreflight(TexMotionSettings settings)
        {
            if (settings == null) return;
            BeginNestedCard();
            EditorGUILayout.LabelField("Runtime Preflight", EditorStyles.boldLabel);
            EditorGUILayout.LabelField(
                "Checks the selected backend, exact local paths, role-specific file formats, WHAM hooks, and the Python profile before extraction.",
                EditorStyles.miniLabel);

            VideoPreflightReport report = _videoPreflightReport;
            if (report == null || !_videoPreflightRequested || report.Backend != settings.VideoBackend)
                report = VideoModelCatalog.GetPreflightReport(settings, settings.VideoBackend);
            _videoPreflightReport = report;

            MessageType reportType = report.IsReady ? MessageType.Info : MessageType.Warning;
            EditorGUILayout.HelpBox(
                report.IsReady
                    ? "Preflight ready: all required files are present and compatible."
                    : "Preflight incomplete: the exact missing or incompatible files are listed below.",
                reportType);
            IReadOnlyList<VideoAssetStatusInfo> actionable = report.NonReadyAssets;
            for (int i = 0; i < actionable.Count; i++)
            {
                VideoAssetStatusInfo status = actionable[i];
                if (status == null)
                {
                    EditorGUILayout.HelpBox("Unknown runtime asset status.", MessageType.Warning);
                    continue;
                }
                EditorGUILayout.BeginHorizontal();
                EditorGUILayout.LabelField(
                    (status.IsBlocking ? "Required" : "Optional/full parity") + ": " + status.ToString(),
                    EditorStyles.miniLabel);
                VideoModelDefinition asset = VideoModelCatalog.Find(status.Id);
                bool incompatible = status.Status == VideoAssetStatus.Incompatible;
                if (asset != null && incompatible &&
                    GUILayout.Button(TexMotionLocalization.TrLiteral("Choose compatible file"), EditorStyles.miniButton, GUILayout.Width(132)))
                {
                    HandleIncompatibleWhamAssetAction(settings, asset, status);
                }
                else if (asset != null && asset.CanDownload && status.CanInstall &&
                    GUILayout.Button("Install", EditorStyles.miniButton, GUILayout.Width(58)))
                {
                    StartVideoModelDownload(asset.Id, settings);
                }
                if (asset != null && VideoModelCatalog.GetAssetGuide(asset) != null &&
                    GUILayout.Button("Guide", EditorStyles.miniButton, GUILayout.Width(52)))
                {
                    VideoAssetGuideWindow.Show(asset, settings);
                }
                EditorGUILayout.EndHorizontal();
                if (status.Status == VideoAssetStatus.Incompatible)
                {
                    EditorGUILayout.HelpBox(status.Reason ?? "The selected file is incompatible with this role.", MessageType.Error);
                    EditorGUILayout.HelpBox(
                        TexMotionLocalization.TrLiteral("Incompatible templates cannot be fixed by Download. Choose a compatible local file, then run Recheck."),
                        MessageType.Warning);
                }
            }

            EditorGUILayout.BeginHorizontal();
            GUI.enabled = !_isVideoModelDownloading && !_isBuildingVideoVenv && !_isInstallingUv && !_isVideoExtracting;
            if (GUILayout.Button("Run Preflight", EditorStyles.miniButton, GUILayout.Height(24)))
            {
                RunVideoRuntimePreflight(settings);
            }
            bool qualityBackend = IsVideoQualityBackend(settings.VideoBackend);
            if (GUILayout.Button("Install Python Environment", EditorStyles.miniButton, GUILayout.Height(24)))
            {
                settings.VideoPythonEnvironmentProfile = qualityBackend
                    ? PythonDependencyProfile.PyTorch
                    : PythonDependencyProfile.Lightweight;
                BuildVideoVenvAsync(settings);
            }
            GUI.enabled = true;
            EditorGUILayout.EndHorizontal();
            EditorGUILayout.EndVertical();
        }

        private void DrawQualityRuntimeAssetCatalog(TexMotionSettings settings, bool previousGuiEnabled, bool canDownload)
        {
            if (settings == null || VideoModelCatalog.RuntimeAdapters.Count == 0) return;
            EditorGUILayout.Space(4);
            EditorGUILayout.LabelField("Quality Backend Runtime Assets", EditorStyles.boldLabel);
            EditorGUILayout.LabelField(
                "HMR2/4D-Humans and HybrIK source/runtime bridges are not redistributed. Browse a compatible local bridge and preflight verifies its entry point before extraction.",
                EditorStyles.miniLabel);
            for (int i = 0; i < VideoModelCatalog.RuntimeAdapters.Count; i++)
            {
                VideoModelDefinition asset = VideoModelCatalog.RuntimeAdapters[i];
                VideoAssetStatusInfo status = VideoModelCatalog.GetAssetStatus(settings, asset, asset.RequiredForFullWhamParity);
                BeginNestedRow();
                EditorGUILayout.LabelField(asset.DisplayName, GUILayout.Width(235));
                string state = status == null ? "?" : status.Status.ToString();
                EditorGUILayout.LabelField(state, EditorStyles.miniLabel, GUILayout.Width(82));
                GUI.enabled = previousGuiEnabled && canDownload;
                if (GUILayout.Button(asset.IsDirectoryAsset ? "Browse Folder" : "Browse", EditorStyles.miniButton, GUILayout.Width(86)))
                {
                    if (asset.IsDirectoryAsset)
                    {
                        string selected = EditorUtility.OpenFolderPanel("Select " + asset.DisplayName, settings.GetEffectiveVideoModelDirectory(), "");
                        if (!string.IsNullOrEmpty(selected))
                        {
                            if (asset.AssetRole == VideoAssetRole.Hmr2Runtime)
                                settings.HMR2RuntimePath = selected;
                            else
                                settings.VideoModelDirectory = selected;
                            settings.Save();
                        }
                    }
                    else
                    {
                        string selected = EditorUtility.OpenFilePanel("Select " + asset.DisplayName, settings.GetEffectiveVideoModelDirectory(), "py,pkl,npz");
                        if (!string.IsNullOrEmpty(selected))
                        {
                            if (asset.AssetRole == VideoAssetRole.WhamBodyModel)
                                settings.WHAMBodyModelPath = selected;
                            else if (asset.AssetRole == VideoAssetRole.Hmr2BodyModel)
                                settings.HMR2BodyModelPath = selected;
                            else if (asset.AssetRole == VideoAssetRole.Hmr2Adapter)
                                settings.HMR2AdapterPath = selected;
                            else if (asset.AssetRole == VideoAssetRole.HybrIKAdapter)
                                settings.HybrIKAdapterPath = selected;
                            else
                                settings.VideoQualityAdapterPath = selected;
                            settings.Save();
                        }
                    }
                }
                GUI.enabled = previousGuiEnabled;
                if (GUILayout.Button("Source", EditorStyles.miniButton, GUILayout.Width(52)))
                    Application.OpenURL(asset.ReferenceUrl);
                VideoAssetGuide runtimeGuide = VideoModelCatalog.GetAssetGuide(asset);
                if (runtimeGuide != null && GUILayout.Button("Guide", EditorStyles.miniButton, GUILayout.Width(52)))
                {
                    VideoAssetGuideWindow.Show(asset, settings);
                }
                EditorGUILayout.EndHorizontal();
                if (status != null && !status.IsReady)
                    EditorGUILayout.LabelField("  " + status.ToString(), EditorStyles.miniLabel);
            }
        }

        private void RunVideoRuntimePreflight(TexMotionSettings settings)
        {
            if (settings == null) return;
            _videoPreflightReport = VideoModelCatalog.GetPreflightReport(settings, settings.VideoBackend);
            _videoPreflightRequested = true;
            _statusMessage = _videoPreflightReport.IsReady
                ? "Video2Motion preflight passed."
                : _videoPreflightReport.Describe();
            Repaint();
        }

        /// <summary>
        /// Replaces an incompatible local companion asset with a user-selected
        /// file/folder.  Downloading an incompatible WHAM row only restores the
        /// bundled contract template (or zero-intrinsics camera template), so
        /// routing the action here gives the user the operation that can
        /// actually make the row Ready.
        /// </summary>
        private void HandleIncompatibleWhamAssetAction(
            TexMotionSettings settings,
            VideoModelDefinition asset,
            VideoAssetStatusInfo status)
        {
            if (settings == null || asset == null) return;

            string selected;
            if (asset.IsDirectoryAsset)
            {
                selected = EditorUtility.OpenFolderPanel(
                    TexMotionLocalization.TrFormat("Choose compatible {0}", asset.DisplayName),
                    settings.GetEffectiveVideoModelDirectory(),
                    string.Empty);
            }
            else
            {
                string filter = string.Empty;
                if (asset.SupportedExtensions != null && asset.SupportedExtensions.Length > 0)
                {
                    var extensions = new List<string>();
                    for (int i = 0; i < asset.SupportedExtensions.Length; i++)
                    {
                        string extension = asset.SupportedExtensions[i];
                        if (string.IsNullOrWhiteSpace(extension)) continue;
                        extension = extension.Trim();
                        if (extension.StartsWith(".", StringComparison.Ordinal)) extension = extension.Substring(1);
                        extensions.Add(extension);
                    }
                    filter = string.Join(",", extensions.ToArray());
                }
                selected = EditorUtility.OpenFilePanel(
                    TexMotionLocalization.TrFormat("Choose compatible {0}", asset.DisplayName),
                    settings.GetEffectiveVideoModelDirectory(),
                    filter);
            }

            if (string.IsNullOrWhiteSpace(selected))
            {
                _statusMessage = TexMotionLocalization.TrLiteral("No compatible file was selected. Open Guide for the required format.");
                Repaint();
                return;
            }

            AssignVideoAssetPath(settings, asset, selected);
            settings.Save();
            _videoPreflightRequested = false;
            _videoPreflightReport = null;
            string reason = status == null || string.IsNullOrWhiteSpace(status.Reason)
                ? TexMotionLocalization.TrLiteral("Run Recheck to validate the selected file.")
                : TexMotionLocalization.TrFormat("Previous status: {0}. Run Recheck to validate the selected file.", status.Reason);
            _statusMessage = TexMotionLocalization.TrFormat(
                "Selected {0}. {1}",
                Path.GetFileName(selected),
                reason);
            Repaint();
        }

        private static void AssignVideoAssetPath(
            TexMotionSettings settings,
            VideoModelDefinition asset,
            string selected)
        {
            if (settings == null || asset == null) return;
            switch (asset.AssetRole)
            {
                case VideoAssetRole.WhamBodyModel:
                    settings.WHAMBodyModelPath = selected;
                    break;
                case VideoAssetRole.Hmr2BodyModel:
                    settings.HMR2BodyModelPath = selected;
                    break;
                case VideoAssetRole.WhamImageFeatureBackbone:
                    settings.WHAMImageFeatureBackbonePath = selected;
                    break;
                case VideoAssetRole.WhamImageFeatureArchive:
                    settings.WHAMImageFeaturePath = selected;
                    break;
                case VideoAssetRole.WhamImageFeatureModelDefinition:
                    settings.WHAMImageFeatureModelDefinitionPath = selected;
                    break;
                case VideoAssetRole.WhamImageFeatureConfig:
                    settings.WHAMImageFeatureConfigPath = selected;
                    break;
                case VideoAssetRole.WhamCamera:
                    settings.WHAMCameraModelPath = selected;
                    break;
                case VideoAssetRole.WhamDpvo:
                    settings.WHAMDpvoModelPath = selected;
                    break;
                case VideoAssetRole.Hmr2Adapter:
                    settings.HMR2AdapterPath = selected;
                    break;
                case VideoAssetRole.HybrIKAdapter:
                    settings.HybrIKAdapterPath = selected;
                    break;
                case VideoAssetRole.Hmr2Runtime:
                    settings.HMR2RuntimePath = selected;
                    break;
                case VideoAssetRole.WhamAdapter:
                    settings.VideoQualityAdapterPath = selected;
                    break;
                case VideoAssetRole.WhamManifest:
                    settings.WHAMAssetManifestPath = selected;
                    break;
                case VideoAssetRole.WhamCheckpoint:
                    if (asset.Kind == VideoModelKind.HMR2) settings.HMR2ModelPath = selected;
                    else settings.WHAMModelPath = selected;
                    break;
                case VideoAssetRole.InferenceModel:
                    settings.RTMPoseModelPath = selected;
                    break;
            }
        }

        private void DrawWhamCompanionPathSettings(TexMotionSettings settings)
        {
            if (settings == null) return;
            BeginNestedCard();
            EditorGUILayout.LabelField(TexMotionLocalization.TrLiteral("WHAM Stage Paths (optional explicit overrides)"), EditorStyles.boldLabel);
            EditorGUILayout.LabelField(
                TexMotionLocalization.TrLiteral("The cache is scanned automatically. Explicit paths let you keep licensed assets, ViTPose features, or DPVO exports outside the cache."),
                EditorStyles.miniLabel);

            DrawWhamCompanionPathField(
                settings,
                TexMotionLocalization.TrLiteral("ViTPose 2D Detector"),
                TexMotionLocalization.TrLiteral("ViTPose checkpoint used for official WHAM 2D preprocessing. It is separate from the HMR2 image-feature runner."),
                settings.WHAMImageFeatureBackbonePath,
                value => settings.WHAMImageFeatureBackbonePath = value,
                TexMotionLocalization.TrLiteral("Select ViTPose image-feature backbone"), "pth");
            DrawWhamCompanionPathField(
                settings,
                TexMotionLocalization.TrLiteral("ViTPose Model Definition"),
                TexMotionLocalization.TrLiteral("Python model-definition hook required to construct a raw ViTPose state-dict checkpoint. The catalog installs a contract template; replace it with a compatible MMPose/Transformers factory."),
                settings.WHAMImageFeatureModelDefinitionPath,
                value => settings.WHAMImageFeatureModelDefinitionPath = value,
                TexMotionLocalization.TrLiteral("Select ViTPose model-definition module"), "py");
            DrawWhamCompanionPathField(
                settings,
                TexMotionLocalization.TrLiteral("ViTPose Runner Config"),
                TexMotionLocalization.TrLiteral("Optional JSON/YAML/Python runner config. A config does not replace the model-definition implementation."),
                settings.WHAMImageFeatureConfigPath,
                value => settings.WHAMImageFeatureConfigPath = value,
                TexMotionLocalization.TrLiteral("Select ViTPose runner config"), "json,yaml,yml,py");
            DrawWhamCompanionPathField(
                settings,
                TexMotionLocalization.TrLiteral("Feature Archive"),
                TexMotionLocalization.TrLiteral("Precomputed WHAM frame features (.npy/.npz/.pt), one row per sampled frame."),
                settings.WHAMImageFeaturePath,
                value => settings.WHAMImageFeaturePath = value,
                TexMotionLocalization.TrLiteral("Select WHAM image feature archive"), "npy,npz,pt,pth");
            DrawViTPoseFeatureExportControl(settings);
            DrawWhamCompanionPathField(
                settings,
                TexMotionLocalization.TrLiteral("Camera Config / Poses"),
                TexMotionLocalization.TrLiteral("Intrinsics configure projection; exported DPVO/SLAM poses provide world motion. If no pose archive is supplied, a local per-video optical-flow cache is generated."),
                settings.WHAMCameraModelPath,
                value => settings.WHAMCameraModelPath = value,
                TexMotionLocalization.TrLiteral("Select WHAM camera calibration or pose archive"), "yaml,yml,json,npy,npz,pt,pth");
            DrawWhamCompanionPathField(
                settings,
                TexMotionLocalization.TrLiteral("DPVO Weights / Poses"),
                TexMotionLocalization.TrLiteral("DPVO weights can be used by a configured runner; pose exports can be consumed directly. Without a runner, TexMotion generates a conservative local optical-flow cache and labels it."),
                settings.WHAMDpvoModelPath,
                value => settings.WHAMDpvoModelPath = value,
                TexMotionLocalization.TrLiteral("Select DPVO checkpoint or pose archive"), "pth,pt,tar,npy,npz,json");

            EditorGUILayout.EndVertical();
        }

        private void DrawWhamCompanionPathField(
            TexMotionSettings settings,
            string label,
            string tooltip,
            string current,
            Action<string> assign,
            string dialogTitle,
            string extension)
        {
            string value = EditorGUILayout.TextField(
                new GUIContent(TexMotionLocalization.TrLiteral(label), TexMotionLocalization.TrLiteral(tooltip)),
                current ?? string.Empty);
            // Keep the path field readable when the Settings window is narrow.
            // The actions get their own row instead of competing with a long
            // absolute path for the same horizontal pixels.
            EditorGUILayout.BeginHorizontal();
            if (GUILayout.Button(TexMotionLocalization.TrLiteral("Browse..."), EditorStyles.miniButton, GUILayout.Height(22f)))
            {
                string selected = EditorUtility.OpenFilePanel(
                    TexMotionLocalization.TrLiteral(dialogTitle),
                    settings.GetEffectiveVideoModelDirectory(),
                    extension);
                if (!string.IsNullOrEmpty(selected)) value = selected;
            }
            if (GUILayout.Button(TexMotionLocalization.TrLiteral("Clear"), EditorStyles.miniButton, GUILayout.Height(22f))) value = string.Empty;
            EditorGUILayout.EndHorizontal();
            if (!string.Equals(current ?? string.Empty, value, StringComparison.Ordinal))
            {
                assign(value);
                settings.Save();
            }
            string resolved = string.IsNullOrWhiteSpace(value) ? TexMotionLocalization.TrLiteral("auto") : value;
            bool exists = false;
            try { exists = !string.IsNullOrWhiteSpace(value) && File.Exists(value); } catch { }
            EditorGUILayout.LabelField(
                "  " + (exists ? "● " : "○ ") + TexMotionLocalization.TrFormat("Resolved: {0}", resolved),
                EditorStyles.miniLabel);
        }

        private void DrawViTPoseFeatureExportControl(TexMotionSettings settings)
        {
            if (settings == null) return;
            string sourceVideo = _videoPath;
            bool sourceReady = !string.IsNullOrWhiteSpace(sourceVideo) &&
                !string.Equals(sourceVideo, "synthetic_motion_test", StringComparison.OrdinalIgnoreCase) &&
                File.Exists(sourceVideo);
            string configuredOutput = settings.WHAMImageFeaturePath;
            string outputPath = configuredOutput;
            if (string.IsNullOrWhiteSpace(outputPath))
            {
                outputPath = Path.Combine(
                    settings.GetEffectiveVideoModelDirectory(),
                    "vitpose_features.npz");
            }

            EditorGUILayout.BeginVertical(EditorStyles.helpBox);
            EditorGUILayout.LabelField(
                TexMotionLocalization.TrLiteral("Create ViTPose Feature Archive"),
                EditorStyles.boldLabel);
            EditorGUILayout.LabelField(
                TexMotionLocalization.TrLiteral(
                    "One button samples the selected video with the same FPS/trim settings, runs the configured local ViTPose runner, and registers vitpose_features.npz for WHAM."),
                EditorStyles.miniLabel);
            EditorGUILayout.LabelField(
                TexMotionLocalization.TrFormat(
                    "Source video: {0}",
                    sourceReady ? sourceVideo : TexMotionLocalization.TrLiteral("not selected")),
                EditorStyles.miniLabel);
            GUI.enabled = !_isVitPoseFeatureExporting;
            if (GUILayout.Button(TexMotionLocalization.TrLiteral("Choose Video..."), EditorStyles.miniButton, GUILayout.Height(22f)))
            {
                string selected = EditorUtility.OpenFilePanel(
                    TexMotionLocalization.TrLiteral("Select source video for ViTPose features"),
                    sourceReady ? Path.GetDirectoryName(sourceVideo) : "",
                    "mp4,mov,avi,mkv,webm");
                if (!string.IsNullOrWhiteSpace(selected))
                {
                    _videoPath = selected.Replace('\\', '/');
                    sourceVideo = _videoPath;
                    sourceReady = File.Exists(sourceVideo);
                }
            }
            GUI.enabled = true;
            EditorGUILayout.LabelField(
                TexMotionLocalization.TrFormat("Output archive: {0}", outputPath),
                EditorStyles.miniLabel);
            EditorGUILayout.LabelField(
                TexMotionLocalization.TrFormat(
                    "Sampling: {0:F0} FPS, trim {1:F2}s–{2}",
                    _videoTargetFps,
                    _videoTrimStart,
                    _videoTrimEnd > 0f ? _videoTrimEnd.ToString("F2") + "s" : TexMotionLocalization.TrLiteral("end")),
                EditorStyles.miniLabel);

            string checkpoint = settings.GetWHAMImageFeatureBackbonePathIfPresent();
            string definition = settings.GetWHAMImageFeatureModelDefinitionPathIfPresent();
            bool definitionIsPlaceholder = IsBundledViTPoseDefinitionPlaceholder(definition);
            bool runnerInputsReady = !string.IsNullOrWhiteSpace(checkpoint) &&
                !string.IsNullOrWhiteSpace(definition) && !definitionIsPlaceholder;
            if (!runnerInputsReady)
            {
                EditorGUILayout.HelpBox(
                    TexMotionLocalization.TrLiteral(
                        definitionIsPlaceholder
                            ? "The bundled ViTPose model-definition file is a contract template, not an architecture. Replace it with a compatible create_model/model factory before extracting."
                            : "Select a ViTPose checkpoint and compatible model-definition hook above. A raw .pth file alone cannot run inference."),
                    MessageType.Warning);
            }

            bool previousEnabled = GUI.enabled;
            GUI.enabled = previousEnabled && sourceReady && runnerInputsReady &&
                !_isVitPoseFeatureExporting && !_isVideoExtracting && !_isVideoModelDownloading;
            if (GUILayout.Button(
                _isVitPoseFeatureExporting
                    ? TexMotionLocalization.TrLiteral("Extracting ViTPose features...")
                    : TexMotionLocalization.TrLiteral("Extract Features (one click)"),
                GUILayout.Height(24)))
            {
                StartViTPoseFeatureExportAsync(settings, sourceVideo, outputPath, checkpoint, definition);
            }
            GUI.enabled = previousEnabled && _isVitPoseFeatureExporting;
            if (GUILayout.Button(TexMotionLocalization.TrLiteral("Cancel"), EditorStyles.miniButton, GUILayout.Height(22f)))
                CancelViTPoseFeatureExport();
            GUI.enabled = previousEnabled;

            if (_isVitPoseFeatureExporting)
            {
                EditorGUI.ProgressBar(
                    EditorGUILayout.GetControlRect(false, 18),
                    _vitPoseFeatureExportProgress,
                    _vitPoseFeatureExportStatus);
            }
            else if (!string.IsNullOrWhiteSpace(_vitPoseFeatureExportError))
            {
                EditorGUILayout.HelpBox(_vitPoseFeatureExportError, MessageType.Error);
            }
            else if (!string.IsNullOrWhiteSpace(_vitPoseFeatureExportStatus))
            {
                EditorGUILayout.HelpBox(_vitPoseFeatureExportStatus, MessageType.Info);
            }
            EditorGUILayout.EndVertical();
        }

        private static bool IsBundledViTPoseDefinitionPlaceholder(string path)
        {
            if (string.IsNullOrWhiteSpace(path) || !File.Exists(path)) return false;
            try
            {
                string name = Path.GetFileName(path);
                if (!string.Equals(name, "wham_vitpose_model_definition.py", StringComparison.OrdinalIgnoreCase))
                    return false;
                string text = File.ReadAllText(path);
                if (text.IndexOf("pure pytorch", StringComparison.OrdinalIgnoreCase) >= 0)
                    return false;
                return text.IndexOf("placeholder", StringComparison.OrdinalIgnoreCase) >= 0 ||
                    text.IndexOf("not an architecture", StringComparison.OrdinalIgnoreCase) >= 0;
            }
            catch
            {
                return false;
            }
        }

        private async void StartViTPoseFeatureExportAsync(
            TexMotionSettings settings,
            string sourceVideo,
            string outputPath,
            string checkpoint,
            string definition)
        {
            if (_isVitPoseFeatureExporting || settings == null) return;
            _isVitPoseFeatureExporting = true;
            _vitPoseFeatureExportProgress = 0.0f;
            _vitPoseFeatureExportStatus = TexMotionLocalization.TrLiteral("Preparing ViTPose feature extraction...");
            _vitPoseFeatureExportError = string.Empty;
            _vitPoseFeatureExportCts = new CancellationTokenSource();
            try
            {
                var options = new ViTPoseFeatureExportOptions
                {
                    VideoPath = sourceVideo,
                    OutputPath = outputPath,
                    TargetFps = _videoTargetFps,
                    TrimStart = _videoTrimStart,
                    TrimEnd = _videoTrimEnd,
                    CheckpointPath = checkpoint,
                    ModelDefinitionPath = definition,
                    ConfigPath = settings.GetWHAMImageFeatureConfigPathIfPresent(),
                    Device = "auto",
                    PythonExecutable = settings.GetEffectiveVideoPythonExecutablePath(),
                    ScriptPath = VideoMotionJobRunner.FindScriptPath(),
                    ChunkSize = 32
                };
                var progress = new Progress<VideoJobProgress>(p =>
                {
                    _vitPoseFeatureExportProgress = Mathf.Clamp01(p.Progress);
                    _vitPoseFeatureExportStatus = TexMotionLocalization.TrLiteral(p.Stage ?? "processing");
                    Repaint();
                });
                string resultPath = await VideoMotionJobRunner.RunViTPoseFeatureExportAsync(
                    options,
                    progress,
                    _vitPoseFeatureExportCts.Token,
                    line => AppendVideoVenvLog(line + Environment.NewLine));
                settings.WHAMImageFeaturePath = resultPath;
                settings.Save();
                _vitPoseFeatureExportProgress = 1.0f;
                _vitPoseFeatureExportStatus = TexMotionLocalization.TrFormat(
                    "ViTPose feature archive ready: {0}", resultPath);
                _statusMessage = _vitPoseFeatureExportStatus;
            }
            catch (OperationCanceledException)
            {
                _vitPoseFeatureExportStatus = TexMotionLocalization.TrLiteral("ViTPose feature extraction cancelled.");
                _vitPoseFeatureExportError = string.Empty;
            }
            catch (Exception ex)
            {
                _vitPoseFeatureExportStatus = TexMotionLocalization.TrLiteral("ViTPose feature extraction failed.");
                _vitPoseFeatureExportError = ex.Message;
                _statusMessage = _vitPoseFeatureExportStatus;
                Debug.LogError("[TexMotion] ViTPose feature export failed: " + ex);
            }
            finally
            {
                _isVitPoseFeatureExporting = false;
                _vitPoseFeatureExportCts?.Dispose();
                _vitPoseFeatureExportCts = null;
                Repaint();
            }
        }

        private void CancelViTPoseFeatureExport()
        {
            if (!_isVitPoseFeatureExporting) return;
            _vitPoseFeatureExportStatus = TexMotionLocalization.TrLiteral("Cancelling ViTPose feature extraction...");
            _vitPoseFeatureExportCts?.Cancel();
            Repaint();
        }

        private string GetEffectiveViTPoseFeatureOutputPath(string sourceVideo)
        {
            if (string.IsNullOrEmpty(sourceVideo)) return null;
            var settings = TexMotionSettings.instance;
            string cacheDir = settings != null ? settings.GetWHAMPreprocessDirectoryIfPresent() : null;
            if (string.IsNullOrEmpty(cacheDir) && settings != null)
            {
                cacheDir = settings.GetEffectiveVideoModelDirectory();
            }
            if (string.IsNullOrEmpty(cacheDir))
            {
                cacheDir = Path.Combine(Path.GetTempPath(), "TexMotion", "VideoPreprocess");
            }
            if (!Directory.Exists(cacheDir))
            {
                try { Directory.CreateDirectory(cacheDir); } catch { }
            }
            string cleanName = Path.GetFileNameWithoutExtension(sourceVideo);
            foreach (char c in Path.GetInvalidFileNameChars())
            {
                cleanName = cleanName.Replace(c, '_');
            }
            return Path.Combine(cacheDir, $"{cleanName}_vitpose_features.npz").Replace('\\', '/');
        }

        private void TryTriggerAutoViTPoseFeatureExport()
        {
            if (_isVitPoseFeatureExporting || _isVideoExtracting) return;
            if (string.IsNullOrEmpty(_videoPath) || _videoPath == "synthetic_motion_test" || !File.Exists(_videoPath)) return;

            var settings = TexMotionSettings.instance;
            if (settings == null) return;

            string checkpoint = settings.GetWHAMImageFeatureBackbonePathIfPresent();
            string definition = settings.GetWHAMImageFeatureModelDefinitionPathIfPresent();
            bool definitionIsPlaceholder = IsBundledViTPoseDefinitionPlaceholder(definition);
            bool runnerInputsReady = !string.IsNullOrWhiteSpace(checkpoint) &&
                !string.IsNullOrWhiteSpace(definition) && !definitionIsPlaceholder;
            if (!runnerInputsReady) return;

            string targetCachePath = GetEffectiveViTPoseFeatureOutputPath(_videoPath);
            if (string.IsNullOrEmpty(targetCachePath)) return;

            if (File.Exists(targetCachePath))
            {
                if (string.IsNullOrEmpty(settings.WHAMImageFeaturePath) || settings.WHAMImageFeaturePath != targetCachePath)
                {
                    settings.WHAMImageFeaturePath = targetCachePath;
                    settings.Save();
                }
                return;
            }

            if (_autoVitPoseExportTriggeredForVideo && string.Equals(_lastAutoVitPoseVideoPath, _videoPath, StringComparison.OrdinalIgnoreCase))
            {
                return;
            }

            bool hasCuda = _detectedPythonInfo != null && _detectedPythonInfo.IsAvailable && _detectedPythonInfo.HasCuda;
            _autoVitPoseExportTriggeredForVideo = true;
            _lastAutoVitPoseVideoPath = _videoPath;

            if (hasCuda)
            {
                _statusMessage = TexMotionLocalization.TrLiteral("GPU (CUDA) detected. Starting automatic background feature extraction...");
                StartViTPoseFeatureExportAsync(settings, _videoPath, targetCachePath, checkpoint, definition);
            }
        }

        private void DrawVideoViTPoseFeatureSection()
        {
            if (string.IsNullOrEmpty(_videoPath) || _videoPath == "synthetic_motion_test" || !File.Exists(_videoPath))
                return;

            var settings = TexMotionSettings.instance;
            if (settings == null) return;

            string checkpoint = settings.GetWHAMImageFeatureBackbonePathIfPresent();
            string definition = settings.GetWHAMImageFeatureModelDefinitionPathIfPresent();
            bool definitionIsPlaceholder = IsBundledViTPoseDefinitionPlaceholder(definition);
            bool runnerInputsReady = !string.IsNullOrWhiteSpace(checkpoint) &&
                !string.IsNullOrWhiteSpace(definition) && !definitionIsPlaceholder;

            string cachePath = GetEffectiveViTPoseFeatureOutputPath(_videoPath);
            bool cacheExists = !string.IsNullOrEmpty(cachePath) && File.Exists(cachePath);
            bool hasCuda = _detectedPythonInfo != null && _detectedPythonInfo.IsAvailable && _detectedPythonInfo.HasCuda;

            BeginCard();

            EditorGUILayout.BeginHorizontal();
            EditorGUILayout.LabelField(
                TexMotionLocalization.TrLiteral("ViTPose Features (WHAM Quality)"),
                EditorStyles.boldLabel);

            if (_isVitPoseFeatureExporting)
            {
                Color prevColor = GUI.color;
                GUI.color = new Color(0.35f, 0.85f, 1.0f);
                GUILayout.Label("⏳ " + TexMotionLocalization.TrLiteral("Processing"), _badgeStyle, GUILayout.Width(90), GUILayout.Height(20));
                GUI.color = prevColor;
            }
            else if (cacheExists)
            {
                Color prevColor = GUI.color;
                GUI.color = new Color(0.35f, 0.95f, 0.55f);
                GUILayout.Label("✓ " + TexMotionLocalization.TrLiteral("ViTPose Features Cached"), _badgeStyle, GUILayout.Width(170), GUILayout.Height(20));
                GUI.color = prevColor;
            }
            else if (hasCuda)
            {
                Color prevColor = GUI.color;
                GUI.color = new Color(0.45f, 0.95f, 0.65f);
                GUILayout.Label("⚡ CUDA Ready", _badgeStyle, GUILayout.Width(100), GUILayout.Height(20));
                GUI.color = prevColor;
            }
            else
            {
                Color prevColor = GUI.color;
                GUI.color = new Color(0.95f, 0.75f, 0.25f);
                GUILayout.Label("⚠️ CPU Only", _badgeStyle, GUILayout.Width(90), GUILayout.Height(20));
                GUI.color = prevColor;
            }
            EditorGUILayout.EndHorizontal();

            if (_isVitPoseFeatureExporting)
            {
                EditorGUILayout.Space(4);
                EditorGUILayout.LabelField(
                    TexMotionLocalization.TrLiteral("Background extraction in progress. You can continue other work."),
                    EditorStyles.miniLabel);

                EditorGUI.ProgressBar(
                    EditorGUILayout.GetControlRect(false, 18),
                    _vitPoseFeatureExportProgress,
                    _vitPoseFeatureExportStatus);

                EditorGUILayout.Space(2);
                if (GUILayout.Button(TexMotionLocalization.TrLiteral("Cancel"), EditorStyles.miniButton, GUILayout.Width(90), GUILayout.Height(22)))
                {
                    CancelViTPoseFeatureExport();
                }
            }
            else if (cacheExists)
            {
                EditorGUILayout.Space(2);
                EditorGUILayout.LabelField(
                    TexMotionLocalization.TrFormat("Cached archive: {0}", Path.GetFileName(cachePath)),
                    EditorStyles.miniLabel);

                if (runnerInputsReady && !_isVideoExtracting)
                {
                    if (GUILayout.Button(TexMotionLocalization.TrLiteral("Re-extract Features"), EditorStyles.miniButton, GUILayout.Width(160), GUILayout.Height(22)))
                    {
                        StartViTPoseFeatureExportAsync(settings, _videoPath, cachePath, checkpoint, definition);
                    }
                }
            }
            else if (!runnerInputsReady)
            {
                EditorGUILayout.Space(2);
                EditorGUILayout.HelpBox(
                    TexMotionLocalization.TrLiteral("ViTPose checkpoint or model definition is missing. Configure it in Settings."),
                    MessageType.Info);
            }
            else if (!hasCuda)
            {
                EditorGUILayout.Space(2);
                EditorGUILayout.HelpBox(
                    TexMotionLocalization.TrLiteral("CPU environment detected. Auto-extraction of ViTPose features (650M params) is skipped to avoid high load."),
                    MessageType.Warning);

                if (!_isVideoExtracting && GUILayout.Button(TexMotionLocalization.TrLiteral("Extract Features Manually (CPU)"), GUILayout.Height(26)))
                {
                    StartViTPoseFeatureExportAsync(settings, _videoPath, cachePath, checkpoint, definition);
                }
            }
            else
            {
                EditorGUILayout.Space(2);
                if (!_isVideoExtracting && GUILayout.Button(TexMotionLocalization.TrLiteral("Extract Features in Background (GPU)"), GUILayout.Height(26)))
                {
                    StartViTPoseFeatureExportAsync(settings, _videoPath, cachePath, checkpoint, definition);
                }
            }

            if (!string.IsNullOrWhiteSpace(_vitPoseFeatureExportError))
            {
                EditorGUILayout.HelpBox(_vitPoseFeatureExportError, MessageType.Error);
            }

            EditorGUILayout.EndVertical();
        }

        /// <summary>
        /// Central settings surface for every Video Motion inference choice. Keeping
        /// these values on the project singleton means extraction is reproducible and
        /// the Video Motion tab does not carry a second, unsaved configuration.
        /// </summary>
        private void DrawVideoInferenceSettings(TexMotionSettings settings)
        {
            if (settings == null) return;

            BeginCard();
            _showSettingsVideoInference = DrawPersistedFoldoutHeader(
                FoldoutSettingsVideoInference,
                _showSettingsVideoInference,
                "🧠 " + TexMotionLocalization.TrLiteral("Video2Motion Inference"),
                GetVideoBackendDisplayName(settings.VideoBackend));
            if (_showSettingsVideoInference)
            {
                EditorGUILayout.HelpBox(
                    TexMotionLocalization.TrLiteral(
                        "Model checkpoint, adapter, device, and temporal settings are stored here and used by the Video Motion tab. " +
                        "The backend can also be changed from the Video Motion dropdown; both tabs persist the same choice. " +
                        "The model catalog above handles one-click provisioning for public RTMPose, DWPose, MediaPipe, WHAM, and HybrIK assets; HMR2 and licensed runtime stages remain manual with Guide actions."),
                    MessageType.None);

                VideoPoseBackend previousBackend = settings.VideoBackend;
                settings.VideoBackend = (VideoPoseBackend)EditorGUILayout.EnumPopup(
                    new GUIContent(
                        TexMotionLocalization.TrLiteral("Detector Backend"),
                        TexMotionLocalization.TrLiteral("Select the inference path used for the next extraction.")),
                    settings.VideoBackend);
                if (previousBackend != settings.VideoBackend)
                {
                    settings.Save();
                    _statusMessage = TexMotionLocalization.TrFormat("Video Motion backend set to {0}.", GetVideoBackendDisplayName(settings.VideoBackend));
                }

                VideoPoseBackend backend = settings.VideoBackend;
                if (backend == VideoPoseBackend.RTMPose)
                {
                    bool hasOrt = _detectedPythonInfo != null && _detectedPythonInfo.HasOnnxRuntime;
                    bool modelReady = CheckRtmposeModelExists();
                    BeginNestedCard();
                    EditorGUILayout.LabelField(TexMotionLocalization.TrLiteral("RTMPose 2D ONNX"), EditorStyles.boldLabel);
                    EditorGUILayout.LabelField(
                        (hasOrt ? "● " + TexMotionLocalization.TrLiteral("ONNX Runtime available") : "○ " + TexMotionLocalization.TrLiteral("ONNX Runtime not detected")) +
                        "  •  " + (modelReady ? "● " + TexMotionLocalization.TrLiteral("local model ready") : "○ " + TexMotionLocalization.TrLiteral("download a model from the catalog above")),
                        EditorStyles.miniLabel);
                    EditorGUILayout.LabelField(
                        TexMotionLocalization.TrFormat("Active file: {0}", Path.GetFileName(settings.GetEffectiveRTMPoseModelPath())),
                        EditorStyles.miniLabel);
                    EditorGUILayout.EndVertical();
                }
                else if (backend == VideoPoseBackend.MediaPipe)
                {
                    settings.VideoModelComplexity = EditorGUILayout.IntPopup(
                        TexMotionLocalization.TrLiteral("MediaPipe Complexity"),
                        Mathf.Clamp(settings.VideoModelComplexity, 0, 2),
                        new[]
                        {
                            TexMotionLocalization.TrLiteral("0 - Fast / Lightweight"),
                            TexMotionLocalization.TrLiteral("1 - Balanced"),
                            TexMotionLocalization.TrLiteral("2 - High Precision")
                        },
                        new[] { 0, 1, 2 });
                    EditorGUILayout.LabelField(
                        TexMotionLocalization.TrLiteral("Choose the matching MediaPipe task asset in the catalog above (lite/full/heavy)."),
                        EditorStyles.miniLabel);
                }
                else if (IsVideoQualityBackend(backend))
                {
                string qualityPath = settings.GetQualityModelPath(backend) ?? string.Empty;
                EditorGUILayout.BeginHorizontal();
                qualityPath = EditorGUILayout.TextField(
                    new GUIContent(
                        TexMotionLocalization.TrLiteral("Checkpoint"),
                        TexMotionLocalization.TrLiteral("Optional .pt/.pth/.ckpt/.tar checkpoint. Adapters may load their own weights.")),
                    qualityPath);
                if (GUILayout.Button(TexMotionLocalization.TrLiteral("Browse..."), GUILayout.Width(75)))
                {
                    string selected = EditorUtility.OpenFilePanel(
                        TexMotionLocalization.TrFormat("Select {0} checkpoint", GetVideoBackendDisplayName(backend)), "", "pt,pth,ckpt,tar,gz");
                    if (!string.IsNullOrEmpty(selected)) qualityPath = selected;
                }
                settings.SetQualityModelPath(backend, qualityPath);
                EditorGUILayout.EndHorizontal();

                string resolvedQualityPath = settings.GetQualityModelPathIfPresent(backend);
                EditorGUILayout.LabelField(
                    resolvedQualityPath == null
                        ? "○ " + TexMotionLocalization.TrLiteral("Checkpoint not found; install it from the catalog or select a local file.")
                        : "● " + TexMotionLocalization.TrFormat("Checkpoint ready: {0}", Path.GetFileName(resolvedQualityPath)),
                    EditorStyles.miniLabel);

                // A WHAM catalog download copies the companion adapter beside
                // the checkpoint. Surface that path automatically so selecting
                // WHAM in Settings is enough to run the downloaded pair.
                if (IsWhamBackend(backend))
                {
                    // Also refresh an adapter copied by an older package so the
                    // Settings card cannot report a stale placeholder as ready.
                    string adapterPathBeforeRefresh = settings.VideoQualityAdapterPath;
                    try { VideoModelDownloader.EnsureBundledWhamAdapter(settings); }
                    catch (Exception ex) { Debug.LogWarning("[TexMotion] Could not refresh bundled WHAM adapter: " + ex.Message); }
                    if (!string.Equals(adapterPathBeforeRefresh, settings.VideoQualityAdapterPath, StringComparison.Ordinal))
                    {
                        settings.Save();
                    }
                }
                if (IsWhamBackend(backend) && string.IsNullOrWhiteSpace(settings.VideoQualityAdapterPath))
                {
                    string bundledAdapter = VideoModelCatalog.GetWhamAdapterPath(settings);
                    try
                    {
                        if (File.Exists(bundledAdapter))
                        {
                            settings.VideoQualityAdapterPath = bundledAdapter;
                            settings.Save();
                        }
                    }
                    catch { }
                }

                string adapterPath = settings.GetVideoQualityAdapterPath(backend);
                string bundledHmr2AdapterPath = null;
                bool showingBundledHmr2Adapter = false;
                if (backend == VideoPoseBackend.HMR2 && string.IsNullOrWhiteSpace(adapterPath))
                {
                    bundledHmr2AdapterPath = VideoModelCatalog.GetBundledHmr2AdapterPath();
                    adapterPath = bundledHmr2AdapterPath ?? string.Empty;
                    showingBundledHmr2Adapter = !string.IsNullOrWhiteSpace(bundledHmr2AdapterPath);
                }
                EditorGUILayout.BeginHorizontal();
                adapterPath = EditorGUILayout.TextField(
                    new GUIContent(
                        TexMotionLocalization.TrLiteral("Adapter"),
                        TexMotionLocalization.TrLiteral("Python module name or .py adapter implementing infer_sequence.")),
                    adapterPath ?? string.Empty);
                if (GUILayout.Button(TexMotionLocalization.TrLiteral("Browse..."), GUILayout.Width(75)))
                {
                    string selected = EditorUtility.OpenFilePanel(TexMotionLocalization.TrLiteral("Select quality adapter"), "", "py");
                    if (!string.IsNullOrEmpty(selected)) adapterPath = selected;
                }
                EditorGUILayout.EndHorizontal();
                if (!(showingBundledHmr2Adapter &&
                      string.Equals(adapterPath, bundledHmr2AdapterPath, StringComparison.OrdinalIgnoreCase)))
                {
                    settings.SetVideoQualityAdapterPath(backend, adapterPath);
                }

                if (IsWhamBackend(backend))
                {
                    bool whamAdapterReady = false;
                    try { whamAdapterReady = !string.IsNullOrWhiteSpace(adapterPath) && File.Exists(adapterPath); }
                    catch { }
                    bool whamCheckpointReady = false;
                    try { whamCheckpointReady = settings.GetQualityModelPathIfPresent(VideoPoseBackend.WHAM) != null; }
                    catch { }
                    string whamImplementation = null;
                    try { whamImplementation = Environment.GetEnvironmentVariable("TEXMOTION_WHAM_ADAPTER_IMPL"); }
                    catch { }
                    bool isBundledBridge = whamAdapterReady && string.Equals(
                        Path.GetFileName(adapterPath),
                        VideoModelCatalog.WhamAdapterFileName,
                        StringComparison.OrdinalIgnoreCase);
                    if (isBundledBridge && string.IsNullOrWhiteSpace(whamImplementation))
                    {
                        EditorGUILayout.HelpBox(
                            (whamCheckpointReady
                                ? TexMotionLocalization.TrLiteral("WHAM checkpoint and bundled adapter bridge are installed. TexMotion's native temporal WHAM core is available for the local path; ")
                                : TexMotionLocalization.TrLiteral("The bundled WHAM adapter bridge is installed. Download a compatible WHAM checkpoint to enable TexMotion's native temporal core. ")) +
                            TexMotionLocalization.TrLiteral(
                                "It requires the PyTorch Quality environment, a compatible checkpoint, and the local MediaPipe auxiliary asset. " +
                                "An official/external WHAM implementation may be selected with TEXMOTION_WHAM_ADAPTER_IMPL when the full research " +
                                "preprocessing or camera-motion stages are needed. Extraction reports the exact fallback reason if the native core cannot initialize."),
                            MessageType.Info);
                    }
                    else
                    {
                        EditorGUILayout.LabelField(
                            whamAdapterReady
                                ? "● " + TexMotionLocalization.TrFormat("WHAM adapter configured: {0}", Path.GetFileName(adapterPath))
                                : "○ " + TexMotionLocalization.TrLiteral("WHAM adapter missing — download WHAM from the catalog to install the bundled adapter."),
                            EditorStyles.miniLabel);
                    }
                }

                string[] devices = { "auto", "cpu", "cuda" };
                string configuredDevice = string.IsNullOrWhiteSpace(settings.VideoQualityDevice)
                    ? "auto"
                    : settings.VideoQualityDevice.Trim().ToLowerInvariant();
                int deviceIndex = Mathf.Max(0, Array.IndexOf(devices, configuredDevice));
                deviceIndex = EditorGUILayout.Popup(
                    new GUIContent(
                        TexMotionLocalization.TrLiteral("PyTorch Device"),
                        TexMotionLocalization.TrLiteral("CUDA is optional; Auto selects CUDA when available and otherwise CPU.")),
                    deviceIndex,
                    new[]
                    {
                        TexMotionLocalization.TrLiteral("Auto"),
                        TexMotionLocalization.TrLiteral("CPU"),
                        TexMotionLocalization.TrLiteral("CUDA")
                    });
                settings.VideoQualityDevice = devices[Mathf.Clamp(deviceIndex, 0, devices.Length - 1)];
                settings.VideoMaxSequenceFrames = Mathf.Max(0, EditorGUILayout.IntField(
                    new GUIContent(
                        TexMotionLocalization.TrLiteral("Max Sequence Frames"),
                        TexMotionLocalization.TrLiteral("0 keeps the full sampled sequence; cap this for constrained memory.")),
                    Mathf.Max(0, settings.VideoMaxSequenceFrames)));

                EditorGUILayout.HelpBox(
                    TexMotionLocalization.TrFormat(
                        "{0} Requires the PyTorch Quality environment profile. If unavailable, extraction reports the reason and falls back safely.",
                        GetVideoQualityBackendDescription(backend)),
                    MessageType.Warning);

                if (IsWhamBackend(backend))
                {
                    EditorGUILayout.LabelField(
                        TexMotionLocalization.TrLiteral("WHAM asset details are grouped in the WHAM Setup card above."),
                        EditorStyles.miniLabel);
                }
            }
            else
            {
                EditorGUILayout.LabelField(
                    TexMotionLocalization.TrLiteral("Auto selects RTMPose + MediaPipe from the local catalog when dependencies and weights are ready."),
                    EditorStyles.miniLabel);
            }
            }

            EditorGUILayout.EndVertical();
        }

        private static void ApplyVideoModelSelection(TexMotionSettings settings, VideoModelDefinition model)
        {
            if (settings == null || model == null) return;
            string path = VideoModelCatalog.GetInstalledPath(settings, model) ??
                          VideoModelCatalog.GetDestinationPath(settings, model);
            if (string.IsNullOrEmpty(path)) return;

            switch (model.Kind)
            {
                case VideoModelKind.RTMPose:
                case VideoModelKind.DWPose:
                    settings.RTMPoseModelPath = path;
                    settings.VideoBackend = VideoPoseBackend.RTMPose;
                    break;
                case VideoModelKind.MediaPipe:
                    settings.VideoBackend = VideoPoseBackend.MediaPipe;
                    if (model.Id.IndexOf("lite", StringComparison.OrdinalIgnoreCase) >= 0)
                        settings.VideoModelComplexity = 0;
                    else if (model.Id.IndexOf("full", StringComparison.OrdinalIgnoreCase) >= 0)
                        settings.VideoModelComplexity = 1;
                    else
                        settings.VideoModelComplexity = 2;
                    break;
                case VideoModelKind.WHAM:
                    settings.SetQualityModelPath(VideoPoseBackend.WHAM, path);
                    settings.VideoBackend = VideoPoseBackend.WHAM;
                    string whamAdapterPath = VideoModelCatalog.GetWhamAdapterPath(settings);
                    if (File.Exists(whamAdapterPath))
                    {
                        settings.VideoQualityAdapterPath = whamAdapterPath;
                    }
                    break;
                case VideoModelKind.HMR2:
                    settings.SetQualityModelPath(VideoPoseBackend.HMR2, path);
                    settings.VideoBackend = VideoPoseBackend.HMR2;
                    break;
                case VideoModelKind.HybrIK:
                    settings.SetQualityModelPath(VideoPoseBackend.HybrIK, path);
                    settings.VideoBackend = VideoPoseBackend.HybrIK;
                    break;
            }
            settings.Save();
        }

        private void DrawVideoPythonEnvironmentSettings(TexMotionSettings settings)
        {
            BeginCard();
            string environmentSummary = _isBuildingVideoVenv
                ? TexMotionLocalization.TrLiteral("Building")
                : (_isInstallingUv
                    ? TexMotionLocalization.TrLiteral("Installing uv")
                    : (settings.IsVideoVenvCreated()
                        ? TexMotionLocalization.TrLiteral("Ready")
                        : TexMotionLocalization.TrLiteral("Setup required")));
            _showSettingsPythonEnvironment = DrawPersistedFoldoutHeader(
                FoldoutSettingsPythonEnvironment,
                _showSettingsPythonEnvironment,
                "🧪 " + TexMotionLocalization.TrLiteral("Video Motion Python Environment"),
                environmentSummary);
            // Keep setup controls and cancellation available while an operation is
            // active, even if the user collapses the section mid-run.
            if (_showSettingsPythonEnvironment || _isBuildingVideoVenv || _isInstallingUv)
            {
                EditorGUILayout.HelpBox(
                    TexMotionLocalization.TrLiteral("Create an isolated project-local venv with uv. The lightweight profile runs MediaPipe + RTMPose/ONNX on CPU or DirectML; the optional quality profile adds the larger PyTorch lifting/refinement stack (WHAM/HybrIK/HMR2 adapters when installed). Live uv output opens below while setup runs and can be copied for diagnostics."),
                    MessageType.None);

            PythonDependencyProfile previousProfile = settings.VideoPythonEnvironmentProfile;
            settings.VideoPythonEnvironmentProfile = (PythonDependencyProfile)EditorGUILayout.EnumPopup(
                TexMotionLocalization.TrLiteral("Inference Profile"), settings.VideoPythonEnvironmentProfile);
            if (previousProfile != settings.VideoPythonEnvironmentProfile && !_isBuildingVideoVenv)
            {
                _videoVenvStatus = TexMotionLocalization.TrLiteral("Profile changed. Build or update the venv to apply it.");
                _videoVenvError = "";
            }

            string profileDescription = settings.VideoPythonEnvironmentProfile == PythonDependencyProfile.PyTorch
                ? TexMotionLocalization.TrLiteral("PyTorch Quality — larger install; enables optional temporal 3D lifting/refinement experiments.")
                : TexMotionLocalization.TrLiteral("Lightweight ONNX — recommended default; no PyTorch or CUDA dependency required.");
            EditorGUILayout.LabelField(profileDescription, EditorStyles.miniLabel);

            settings.UseVideoVenv = EditorGUILayout.Toggle(
                new GUIContent(
                    TexMotionLocalization.TrLiteral("Use project venv for Video Motion"),
                    TexMotionLocalization.TrLiteral("When enabled, extraction uses the venv Python after setup.")),
                settings.UseVideoVenv);

            EditorGUILayout.BeginHorizontal();
            settings.VideoVenvPath = EditorGUILayout.TextField(
                new GUIContent(
                    TexMotionLocalization.TrLiteral("Venv Path"),
                    TexMotionLocalization.TrLiteral("Leave empty for <project>/.texmotion-venv.")),
                settings.VideoVenvPath);
            if (GUILayout.Button(TexMotionLocalization.TrLiteral("Browse..."), GUILayout.Width(75)))
            {
                string selected = EditorUtility.OpenFolderPanel(
                    TexMotionLocalization.TrLiteral("Select Video Motion venv directory"),
                    settings.GetEffectiveVideoVenvPath(), "");
                if (!string.IsNullOrEmpty(selected)) settings.VideoVenvPath = selected;
            }
            EditorGUILayout.EndHorizontal();

            EditorGUILayout.BeginHorizontal();
            settings.UvExecutablePath = EditorGUILayout.TextField(
                new GUIContent(
                    TexMotionLocalization.TrLiteral("uv Executable"),
                    TexMotionLocalization.TrLiteral("Use 'uv' for PATH lookup or select uv.exe.")),
                settings.UvExecutablePath);
            if (GUILayout.Button(TexMotionLocalization.TrLiteral("Browse..."), GUILayout.Width(75)))
            {
                string selected = EditorUtility.OpenFilePanel(TexMotionLocalization.TrLiteral("Select uv executable"), "", Application.platform == RuntimePlatform.WindowsEditor ? "exe" : "");
                if (!string.IsNullOrEmpty(selected)) settings.UvExecutablePath = selected;
            }
            EditorGUILayout.EndHorizontal();

            BeginNestedCard();
            if (_detectedUvInfo != null && _detectedUvInfo.IsAvailable)
            {
                GUI.color = new Color(0.35f, 0.95f, 0.45f);
                EditorGUILayout.LabelField("✓ " + TexMotionLocalization.TrLiteral("uv detected"), EditorStyles.boldLabel);
                GUI.color = Color.white;
                EditorGUILayout.LabelField(_detectedUvInfo.GetSummary(), EditorStyles.miniLabel);

                bool selectedPathIsDifferent = !string.Equals(
                    settings.UvExecutablePath ?? "",
                    _detectedUvInfo.ExecutablePath ?? "",
                    StringComparison.OrdinalIgnoreCase);
                if (selectedPathIsDifferent && GUILayout.Button(TexMotionLocalization.TrLiteral("Use detected uv path"), EditorStyles.miniButton, GUILayout.Height(22)))
                {
                    settings.UvExecutablePath = _detectedUvInfo.ExecutablePath;
                    settings.Save();
                }
            }
            else
            {
                GUI.color = new Color(1.0f, 0.72f, 0.25f);
                EditorGUILayout.LabelField("⚠ " + TexMotionLocalization.TrLiteral("uv not detected"), EditorStyles.boldLabel);
                GUI.color = Color.white;
                EditorGUILayout.LabelField(
                    TexMotionLocalization.TrFormat(
                        "Official Windows installer default: {0}",
                        PythonEnvironmentManager.GetDefaultUvExecutablePath()),
                    EditorStyles.miniLabel);
                if (_detectedUvInfo != null && !string.IsNullOrEmpty(_detectedUvInfo.ErrorMessage))
                {
                    EditorGUILayout.LabelField(_detectedUvInfo.ErrorMessage, EditorStyles.miniLabel);
                }
            }

            EditorGUILayout.BeginHorizontal();
            GUI.enabled = !_isCheckingUv && !_isInstallingUv && !_isBuildingVideoVenv;
            if (GUILayout.Button(
                TexMotionLocalization.TrLiteral(_isCheckingUv ? "Detecting uv..." : "🔍 Detect uv"),
                EditorStyles.miniButton,
                GUILayout.Height(23)))
            {
                CheckUvAsync(settings.UvExecutablePath);
            }
            GUI.enabled = !_isCheckingUv && !_isInstallingUv && !_isBuildingVideoVenv && !_isVideoExtracting;
            if (GUILayout.Button(TexMotionLocalization.TrLiteral("⬇ Install uv"), EditorStyles.miniButton, GUILayout.Height(23)))
            {
                InstallUvAsync(settings);
            }
            GUI.enabled = true;
            if (GUILayout.Button(TexMotionLocalization.TrLiteral("Open uv docs"), EditorStyles.miniButton, GUILayout.Height(23)))
            {
                Application.OpenURL("https://docs.astral.sh/uv/getting-started/installation/");
            }
            EditorGUILayout.EndHorizontal();

            if (_isInstallingUv)
            {
                EditorGUI.ProgressBar(EditorGUILayout.GetControlRect(false, 18), _uvInstallProgress, _uvInstallStatus);
                if (GUILayout.Button(TexMotionLocalization.TrLiteral("Cancel uv installation"), EditorStyles.miniButton, GUILayout.Height(22)))
                {
                    CancelUvInstall();
                }
            }
            else if (!string.IsNullOrEmpty(_uvInstallStatus))
            {
                MessageType installStatusType = string.IsNullOrEmpty(_uvInstallError) ? MessageType.Info : MessageType.Error;
                EditorGUILayout.HelpBox(_uvInstallStatus + (string.IsNullOrEmpty(_uvInstallError) ? "" : "\n" + _uvInstallError), installStatusType);
            }
            EditorGUILayout.EndVertical();

            string effectiveVenvPath = settings.GetEffectiveVideoVenvPath();
            string effectivePythonPath = settings.GetVideoVenvPythonPath();
            EditorGUILayout.LabelField(TexMotionLocalization.TrLiteral("Resolved venv"), effectiveVenvPath, EditorStyles.miniLabel);
            EditorGUILayout.LabelField(TexMotionLocalization.TrLiteral("Python"), effectivePythonPath, EditorStyles.miniLabel);

            bool venvReady = settings.IsVideoVenvCreated();
            bool canBuild = !_isBuildingVideoVenv && !_isInstallingUv && !_isGenerating && !_isDownloading && !_isVideoExtracting;
            GUI.enabled = canBuild;
            string buildLabel = venvReady
                ? TexMotionLocalization.TrLiteral("🔄 Update Video Motion venv")
                : TexMotionLocalization.TrLiteral("⚙️ Create Video Motion venv");
            if (GUILayout.Button(buildLabel + " " + TexMotionLocalization.TrLiteral("via uv"), GUILayout.Height(32)))
            {
                BuildVideoVenvAsync(settings);
            }
            GUI.enabled = true;

            if (_isBuildingVideoVenv)
            {
                EditorGUI.ProgressBar(EditorGUILayout.GetControlRect(false, 18), _videoVenvProgress, _videoVenvStatus);
                if (GUILayout.Button(TexMotionLocalization.TrLiteral("Cancel setup"), GUILayout.Height(22))) CancelVideoVenvSetup();
            }

            if (!string.IsNullOrEmpty(_videoVenvStatus) && !_isBuildingVideoVenv)
            {
                MessageType statusType = string.IsNullOrEmpty(_videoVenvError) ? MessageType.Info : MessageType.Error;
                EditorGUILayout.HelpBox(_videoVenvStatus + (string.IsNullOrEmpty(_videoVenvError) ? "" : "\n" + _videoVenvError), statusType);
            }

            if (_detectedPythonInfo != null)
            {
                bool usingVenv = false;
                if (settings.UseVideoVenv && settings.IsVideoVenvCreated() &&
                    !string.IsNullOrWhiteSpace(_detectedPythonInfo.ExecutablePath) &&
                    !string.IsNullOrWhiteSpace(effectivePythonPath))
                {
                    try
                    {
                        usingVenv = string.Equals(
                            Path.GetFullPath(_detectedPythonInfo.ExecutablePath.Trim()),
                            Path.GetFullPath(effectivePythonPath.Trim()),
                            StringComparison.OrdinalIgnoreCase);
                    }
                    catch
                    {
                        usingVenv = false;
                    }
                }

                EditorGUILayout.LabelField(
                    usingVenv
                        ? "✓ " + TexMotionLocalization.TrLiteral("Probe: project venv is the active Video Motion runtime.")
                        : TexMotionLocalization.TrFormat("Probe: {0}", _detectedPythonInfo.GetSummary()),
                    EditorStyles.miniLabel);
            }

            EditorGUILayout.BeginHorizontal();
            GUI.enabled = !_isBuildingVideoVenv && !_isCheckingPython;
            if (GUILayout.Button(TexMotionLocalization.TrLiteral("🔍 Re-probe Python capabilities"), GUILayout.Height(23))) CheckPythonEnvironmentAsync();
            GUI.enabled = true;
            _showVideoVenvLog = EditorGUILayout.ToggleLeft(TexMotionLocalization.TrLiteral("Show live setup log"), _showVideoVenvLog, GUILayout.Width(135));
            EditorGUILayout.EndHorizontal();

            string videoVenvLog = GetVideoVenvLogSnapshot();
            string videoVenvLogPreview = GetVideoVenvLogPreview(videoVenvLog);
            bool showSetupLog = _showVideoVenvLog || _isBuildingVideoVenv || _isInstallingUv;
            if (showSetupLog)
            {
                BeginNestedCard();
                EditorGUILayout.BeginHorizontal();
                EditorGUILayout.LabelField(
                    TexMotionLocalization.TrLiteral((_isBuildingVideoVenv || _isInstallingUv) ? "Live uv output" : "Last uv output"),
                    EditorStyles.miniLabel);
                EditorGUILayout.LabelField(
                    TexMotionLocalization.TrFormat("{0} chars", videoVenvLog.Length),
                    EditorStyles.miniLabel,
                    GUILayout.Width(65));
                GUI.enabled = !string.IsNullOrEmpty(videoVenvLog);
                if (GUILayout.Button(TexMotionLocalization.TrLiteral("Copy"), EditorStyles.miniButton, GUILayout.Width(52)))
                {
                    EditorGUIUtility.systemCopyBuffer = videoVenvLog;
                }
                GUI.enabled = !_isBuildingVideoVenv && !_isInstallingUv;
                if (GUILayout.Button(TexMotionLocalization.TrLiteral("Clear"), EditorStyles.miniButton, GUILayout.Width(52)))
                {
                    lock (_videoVenvLogLock) _videoVenvLog = "";
                    _videoVenvLogDirty = true;
                }
                GUI.enabled = true;
                EditorGUILayout.EndHorizontal();
                EditorGUILayout.TextArea(
                    string.IsNullOrEmpty(videoVenvLog)
                        ? TexMotionLocalization.TrLiteral("Waiting for uv output...")
                        : videoVenvLogPreview,
                    GUILayout.MinHeight(100), GUILayout.MaxHeight(240));
                EditorGUILayout.EndVertical();
            }
            }

            EditorGUILayout.EndVertical();
        }

        private void DrawFooterStatus()
        {
            if (_isGenerating || _isDownloading || _uiIsVideoExtracting)
            {
                EditorGUILayout.Space(5);
                float prog = _uiIsVideoExtracting ? _uiVideoExtractionProgress : _overallProgress;
                string msg = _uiIsVideoExtracting ? _uiVideoExtractionStatus : _statusMessage;
                EditorGUI.ProgressBar(EditorGUILayout.GetControlRect(false, 18), prog, msg);
            }
            else if (!string.IsNullOrEmpty(_statusMessage))
            {
                EditorGUILayout.Space(5);
                EditorGUILayout.HelpBox(_statusMessage, MessageType.Info);
            }
        }

        private async void StartDownload()
        {
            _isDownloading = true;
            _overallProgress = 0f;
            _fileProgress = 0f;
            _statusMessage = TexMotionLocalization.TrLiteral("Starting download from Hugging Face...");
            _cts = new CancellationTokenSource();

            var settings = TexMotionSettings.instance;

            var progressReporter = new Progress<DownloadProgress>(p =>
            {
                _overallProgress = p.OverallProgress;
                _fileProgress = p.FileProgress;
                _statusMessage = p.StatusText;
                Repaint();
            });

            try
            {
                await ModelDownloader.DownloadAllModelsAsync(settings, progressReporter, _cts.Token);
                _statusMessage = TexMotionLocalization.TrLiteral("Models downloaded successfully!");
                EnsureEngineLoaded();
            }
            catch (Exception ex)
            {
                _statusMessage = TexMotionLocalization.TrFormat("Download failed: {0}", ex.Message);
                Debug.LogError($"[TexMotion] {ex}");
            }
            finally
            {
                _isDownloading = false;
                Repaint();
            }
        }

        private async void StartGeneration()
        {
            if (!EnsureEngineLoaded())
            {
                EditorUtility.DisplayDialog(
                    TexMotionLocalization.TrLiteral("Error"),
                    TexMotionLocalization.TrLiteral("Model is not loaded. Please check settings."),
                    TexMotionLocalization.TrLiteral("OK"));
                return;
            }

            _isGenerating = true;
            _overallProgress = 0.3f;
            _statusMessage = TexMotionLocalization.TrLiteral("Running motion diffusion...");
            Repaint();

            ulong seed = _randomizeSeed ? (ulong)UnityEngine.Random.Range(1, int.MaxValue) : _seed;

            try
            {
                _currentMotionData = await _engine.GenerateAsync(
                    _prompt,
                    _frames,
                    _steps,
                    seed,
                    _textCfg);
                _currentVideoMotionData = null;

                _overallProgress = 0.8f;
                _statusMessage = TexMotionLocalization.TrLiteral("Setting up 3D preview viewport...");
                Repaint();

                _activePreviewClip = null;
                _previewTime = 0f;
                _isPlayingPreview = true;
                _lastUpdateTime = EditorApplication.timeSinceStartup;

                SetupPreviewInstance(autoPlay: true);
                _isPlayingPreview = true;

                _overallProgress = 1.0f;
                _statusMessage = TexMotionLocalization.TrFormat("Motion ready! Previewing on {0}.", _targetAvatar.name);

                if (TexMotionSettings.instance.AutoOpenTimelineEditor)
                {
                    OpenInTimelineEditor();
                }
            }
            catch (Exception ex)
            {
                _statusMessage = TexMotionLocalization.TrFormat("Generation error: {0}", ex.Message);
                Debug.LogError($"[TexMotion] Generation failed: {ex}");
                EditorUtility.DisplayDialog(
                    TexMotionLocalization.TrLiteral("Generation Error"),
                    ex.Message,
                    TexMotionLocalization.TrLiteral("OK"));
            }
            finally
            {
                _isGenerating = false;
                Repaint();
            }
        }

        private async void StartVideoExtraction()
        {
            if (_targetAvatar == null)
            {
                EditorUtility.DisplayDialog(
                    TexMotionLocalization.TrLiteral("Avatar Missing"),
                    TexMotionLocalization.TrLiteral("Please select your target Avatar in the scene first."),
                    TexMotionLocalization.TrLiteral("OK"));
                return;
            }

            _isVideoExtracting = true;
            _videoExtractionStartTime = EditorApplication.timeSinceStartup;
            lock (_videoExtractionLogLock)
            {
                _videoExtractionLog = "";
                _lastExtractionLogLine = "";
            }
            _isPlayingPreview = false;
            _previewTime = 0f;
            CleanupVideoPlayer();
            CleanupPreviewInstance();
            _currentMotionData = null;
            _currentVideoMotionData = null;

            _videoExtractionProgress = 0.05f;
            _videoExtractionStatus = TexMotionLocalization.TrLiteral("Initializing pose extraction pipeline...");
            _statusMessage = TexMotionLocalization.TrLiteral("Starting video pose extractor...");
            _videoCts = new CancellationTokenSource();
            Repaint();

            bool isSynthetic = _videoPath == "synthetic_motion_test";
            var videoSettings = TexMotionSettings.instance;
            VideoPoseBackend configuredBackend = videoSettings != null ? videoSettings.VideoBackend : VideoPoseBackend.Auto;
            // Refresh the paired bridge before resolving the adapter path. A
            // previous TexMotion version may have left the old placeholder
            // bridge in the user cache; using the package copy here upgrades
            // it transparently so the native WHAM implementation is actually
            // loaded on the next extraction.
            if (videoSettings != null && IsWhamBackend(configuredBackend))
            {
                try
                {
                    VideoModelDownloader.EnsureBundledWhamAdapter(videoSettings);
                }
                catch (Exception ex)
                {
                    Debug.LogWarning("[TexMotion] Could not refresh bundled WHAM adapter: " + ex.Message);
                }
            }
            string configuredQualityModel = videoSettings != null
                ? videoSettings.GetQualityModelPathIfPresent(configuredBackend)
                : null;
            string configuredAdapter = videoSettings != null && !string.IsNullOrWhiteSpace(videoSettings.GetVideoQualityAdapterPath(configuredBackend))
                ? videoSettings.GetVideoQualityAdapterPath(configuredBackend).Trim()
                : null;
            // A persisted .py path can outlive a cache reset or a package
            // upgrade. Passing that stale path prevents the Python bridge
            // from loading the bundled native WHAM adapter, even though the
            // catalog row is installed. Module names (for example
            // ``my_adapter``) are intentionally preserved because they are
            // resolved by Python's import path rather than the filesystem.
            if (LooksLikeAdapterFileReference(configuredAdapter) &&
                !IsExistingAdapterFile(configuredAdapter))
            {
                configuredAdapter = null;
            }
            // WHAM downloads include a bundled adapter bridge. Resolve it
            // automatically when the user selected WHAM but has not manually
            // populated the shared adapter field in Settings yet.
            if (string.IsNullOrWhiteSpace(configuredAdapter) &&
                IsWhamBackend(configuredBackend) && videoSettings != null)
            {
                string bundledAdapter = null;
                try
                {
                    VideoModelDefinition adapterDefinition = VideoModelCatalog.FindWhamAsset("wham-adapter");
                    bundledAdapter = VideoModelCatalog.GetInstalledWhamAssetPath(videoSettings, adapterDefinition) ??
                        VideoModelCatalog.GetWhamAdapterPath(videoSettings);
                    if (IsExistingAdapterFile(bundledAdapter)) configuredAdapter = bundledAdapter;
                }
                catch { }
            }
            string configuredBodyModel = null;
            if (videoSettings != null)
            {
                configuredBodyModel = configuredBackend == VideoPoseBackend.HMR2
                    ? videoSettings.GetHMR2BodyModelPathIfPresent()
                    : videoSettings.GetWHAMBodyModelPathIfPresent();
            }
            string configuredDevice = videoSettings != null && !string.IsNullOrWhiteSpace(videoSettings.VideoQualityDevice)
                ? videoSettings.VideoQualityDevice.Trim().ToLowerInvariant()
                : "auto";
            if (configuredDevice != "auto" && configuredDevice != "cpu" && configuredDevice != "cuda") configuredDevice = "auto";

            string configuredFeatureArchive = videoSettings?.GetWHAMImageFeaturePathIfPresent();
            if (string.IsNullOrEmpty(configuredFeatureArchive) && !string.IsNullOrEmpty(_videoPath))
            {
                string targetCachePath = GetEffectiveViTPoseFeatureOutputPath(_videoPath);
                if (!string.IsNullOrEmpty(targetCachePath) && File.Exists(targetCachePath))
                {
                    configuredFeatureArchive = targetCachePath;
                    if (videoSettings != null)
                    {
                        videoSettings.WHAMImageFeaturePath = targetCachePath;
                        videoSettings.Save();
                    }
                }
            }

            var options = new VideoExtractionOptions
            {
                VideoPath = isSynthetic ? "" : _videoPath,
                TargetFps = _videoTargetFps,
                TrimStartTime = _videoTrimStart,
                TrimEndTime = _videoTrimEnd,
                InPlace = _videoInPlace,
                TemporalSmoothing = _videoSmoothing,
                FootLocking = _videoFootLocking,
                ModelComplexity = videoSettings != null ? Mathf.Clamp(videoSettings.VideoModelComplexity, 0, 2) : 2,
                MinDetectionConfidence = 0.5f,
                MinTrackingConfidence = 0.5f,
                Backend = configuredBackend,
                FusionMode = IsWhamBackend(configuredBackend)
                    ? VideoFusionMode.WhamMediaPipe
                    : VideoFusionMode.Off,
                PyTorchModelPath = configuredQualityModel,
                PyTorchAdapterModule = configuredAdapter,
                PyTorchDevice = configuredDevice,
                MaxSequenceFrames = videoSettings != null ? Mathf.Max(0, videoSettings.VideoMaxSequenceFrames) : 0,
                VideoModelDirectory = videoSettings?.GetEffectiveVideoModelDirectory(),
                RTMPoseModelPath = videoSettings?.GetRTMPoseModelPathIfPresent(),
                WhamAssetManifestPath = videoSettings?.GetWHAMAssetManifestPathIfPresent(),
                WhamBodyModelPath = configuredBodyModel,
                WhamImageFeatureBackbonePath = videoSettings?.GetWHAMImageFeatureBackbonePathIfPresent(),
                WhamImageFeatureModelDefinitionPath = videoSettings?.GetWHAMImageFeatureModelDefinitionPathIfPresent(),
                WhamImageFeatureConfigPath = videoSettings?.GetWHAMImageFeatureConfigPathIfPresent(),
                WhamImageFeaturePath = configuredFeatureArchive,
                WhamCameraModelPath = videoSettings?.GetWHAMCameraModelPathIfPresent(),
                WhamDpvoModelPath = videoSettings?.GetWHAMDpvoModelPathIfPresent(),
                OverlayMode = _videoOverlayMode,
                Synthetic = isSynthetic
            };

            var progress = new Progress<VideoJobProgress>(p =>
            {
                _videoExtractionProgress = p.Progress;
                _videoExtractionStatus = p.Message;
                _statusMessage = TexMotionLocalization.TrFormat("[Video] {0}", TexMotionLocalization.TrLiteral(p.Message));
                Repaint();
            });

            try
            {
                var result = await VideoMotionJobRunner.RunExtractionAsync(
                    options,
                    progress,
                    _videoCts.Token,
                    onStderrLine: line =>
                    {
                        if (string.IsNullOrWhiteSpace(line)) return;
                        lock (_videoExtractionLogLock)
                        {
                            string trimmed = line.Trim();
                            _lastExtractionLogLine = trimmed;
                            if (_videoExtractionLog.Length > 16000)
                            {
                                _videoExtractionLog = _videoExtractionLog.Substring(_videoExtractionLog.Length - 8000);
                            }
                            _videoExtractionLog += trimmed + "\n";
                            _videoExtractionLogDirty = true;
                        }
                    });

                _currentVideoMotionData = result;
                _currentMotionData = result;

                _videoExtractionProgress = 1.0f;
                _videoExtractionStatus = result.UsedBackendFallback
                    ? TexMotionLocalization.TrFormat("Extraction complete with fallback ({0}).", result.DetectorName)
                    : TexMotionLocalization.TrLiteral("Extraction complete! Initializing 3D viewport...");
                string resultRequestedBackend = string.IsNullOrEmpty(result.BackendRequested)
                    ? TexMotionLocalization.TrLiteral("selected backend")
                    : result.BackendRequested.ToUpperInvariant();
                _statusMessage = result.UsedBackendFallback
                    ? TexMotionLocalization.TrFormat(
                        "Video motion extracted with fallback: {0} → {1}. See the result card for the reason.",
                        resultRequestedBackend,
                        result.DetectorName)
                    : TexMotionLocalization.TrFormat(
                        "Video motion extracted ({0} frames, {1:F2}s). Ready to preview!",
                        result.Frames,
                        result.Duration);
                Repaint();

                _activePreviewClip = null;
                _previewTime = 0f;
                _isPlayingPreview = true;
                _lastUpdateTime = EditorApplication.timeSinceStartup;

                SetupPreviewInstance(autoPlay: true);
                _isPlayingPreview = true;

                if (!result.UsedBackendFallback)
                {
                    _videoExtractionStatus = TexMotionLocalization.TrFormat("Extracted {0} frames successfully!", result.Frames);
                }

                if (TexMotionSettings.instance.AutoOpenTimelineEditor)
                {
                    OpenInTimelineEditor();
                }
            }
            catch (OperationCanceledException)
            {
                _statusMessage = TexMotionLocalization.TrLiteral("Video motion extraction cancelled.");
                _videoExtractionStatus = TexMotionLocalization.TrLiteral("Extraction cancelled.");
            }
            catch (Exception ex)
            {
                _statusMessage = TexMotionLocalization.TrFormat("Video extraction failed: {0}", ex.Message);
                _videoExtractionStatus = TexMotionLocalization.TrLiteral("Extraction failed.");
                Debug.LogError($"[TexMotion Video] {ex}");
                EditorUtility.DisplayDialog(
                    TexMotionLocalization.TrLiteral("Extraction Error"),
                    TexMotionLocalization.TrFormat("Video extraction failed:\n{0}", ex.Message),
                    TexMotionLocalization.TrLiteral("OK"));
            }
            finally
            {
                _isVideoExtracting = false;
                Repaint();
            }
        }
    }
}
