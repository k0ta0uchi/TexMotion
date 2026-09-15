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

        // Generator settings
        private string _prompt = "a person throws a sharp right punch forward with energy";
        private uint _frames = 50; // 2.5 sec at 20fps
        private uint _steps = 20;
        private ulong _seed = 0;
        private bool _randomizeSeed = true;
        private float _textCfg = 2.0f;
        private bool _inPlace = true;
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
        private float _videoTrimStart = 0.0f;
        private float _videoTrimEnd = 0.0f;
        private int _videoModelComplexity = 1;
        private float _videoMinConfidenceThreshold = 0.3f;
        private string _videoMotionName = "VideoMotion";
        private HandPoseType _videoHandPose = HandPoseType.NaturalRelaxed;
        private FaceEmotionType _videoFaceEmotion = FaceEmotionType.None;
        private float _videoEmotionIntensity = 0.8f;
        private bool _isVideoExtracting = false;
        private float _videoExtractionProgress = 0.0f;
        private string _videoExtractionStatus = "";
        private CancellationTokenSource _videoCts;
        private VideoMotionData _currentVideoMotionData;
        private float _detectedVideoDuration = 0.0f;
        private int _detectedVideoWidth = 0;
        private int _detectedVideoHeight = 0;
        private float _detectedVideoFps = 0.0f;
        private PythonRuntimeInfo _detectedPythonInfo;
        private bool _isCheckingPython = false;
        private bool _isInstallingDependencies = false;

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
        private bool _videoPlayerFailed = false;
        private double _videoPrepareStartTime = 0;
        private const double VIDEO_PREPARE_TIMEOUT_SEC = 8.0;
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

        [MenuItem("Tools/TexMotion/Motion Studio", false, 0)]
        public static void ShowWindow()
        {
            var window = GetWindow<TexMotionWindow>("TexMotion");
            window.minSize = new Vector2(520, 780);
            window.Show();
        }

        private void OnEnable()
        {
            AutoDetectAvatar();
            EnsureEngineLoaded();
            InitPreviewUtility();
            RefreshLibrary();
            CheckPythonEnvironmentAsync();
            EditorApplication.update += OnEditorUpdate;
            AssemblyReloadEvents.beforeAssemblyReload += OnBeforeAssemblyReload;
        }

        private void OnDisable()
        {
            AssemblyReloadEvents.beforeAssemblyReload -= OnBeforeAssemblyReload;
            EditorApplication.update -= OnEditorUpdate;
            CleanupVideoPlayer();
            CleanupPreviewUtility();
            _cts?.Cancel();
            _videoCts?.Cancel();
            _engine?.Dispose();
            _engine = null;
        }

        private void OnDestroy()
        {
            CleanupVideoPlayer();
            CleanupPreviewUtility();
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

                int texW = 640;
                int texH = 360;
                if (_detectedVideoWidth > 0 && _detectedVideoHeight > 0)
                {
                    texW = Mathf.Clamp(_detectedVideoWidth, 320, 1280);
                    texH = Mathf.Clamp(_detectedVideoHeight, 180, 720);
                }

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
                _videoPlayer.targetTexture = _previewVideoTexture;
                _videoPlayer.source = VideoSource.Url;

                // Use native Windows absolute path for VideoPlayer.url with normalized slashes
                string fullPath = Path.GetFullPath(videoPath).Replace('\\', '/');
                _videoPlayer.url = fullPath;

                _videoPlayer.audioOutputMode = VideoAudioOutputMode.None;
                _videoPlayer.isLooping = false; // Disable native looping to prevent Windows Media Foundation EOF lockups; handled deterministically via RestartVideoLoop
                _videoPlayer.skipOnDrop = true;

                _videoPlayer.errorReceived += (source, message) =>
                {
                    Debug.LogWarning($"[TexMotion] VideoPlayer playback error: {message}. Falling back to 3D avatar preview.");
                    _videoPlayerFailed = true;
                    Repaint();
                };

                _videoPlayer.prepareCompleted += (source) =>
                {
                    _videoPlayerFailed = false;
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
            ApplyStaticFingersAndFaceToPreview();

            _previewUtility.AddSingleGO(_previewInstance);
            ApplyCurrentPoseToPreview(0f);
        }

        private void ApplyStaticFingersAndFaceToPreview()
        {
            if (_previewInstance == null || _targetAvatar == null) return;

            var origAnim = _targetAvatar.GetComponent<Animator>();
            if (origAnim == null) return;

            HandPoseType activeHandPose = (_currentTab == Tab.VideoMotion) ? _videoHandPose : _handPose;
            FaceEmotionType activeFaceEmotion = (_currentTab == Tab.VideoMotion) ? _videoFaceEmotion : _faceEmotion;
            float activeIntensity = (_currentTab == Tab.VideoMotion) ? _videoEmotionIntensity : _emotionIntensity;

            if (activeHandPose != HandPoseType.KeepFree)
            {
                ApplyFingersToPreviewInstance(origAnim, HandPosePresets.LeftFingerBones, activeHandPose, true);
                ApplyFingersToPreviewInstance(origAnim, HandPosePresets.RightFingerBones, activeHandPose, false);
            }

            if (activeFaceEmotion == FaceEmotionType.AutoDetect)
            {
                activeFaceEmotion = (_currentTab == Tab.VideoMotion)
                    ? FaceEmotionType.None
                    : FaceEmotionHelper.InferEmotionFromPrompt(_prompt);
            }

            if (_previewFaceRenderer != null && activeFaceEmotion != FaceEmotionType.None)
            {
                var weights = FaceEmotionHelper.GetBlendShapeWeightsForEmotion(_previewFaceRenderer, activeFaceEmotion, activeIntensity);
                foreach (var kvp in weights)
                {
                    _previewFaceRenderer.SetBlendShapeWeight(kvp.Key, kvp.Value);
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
                    Quaternion offset = HandPosePresets.GetFingerLocalRotation(pose, boneType, isLeft);
                    cloneBone.localRotation = rest * offset;
                }
            }
        }

        private void OnEditorUpdate()
        {
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
                        double elapsed = currentTime - _videoPrepareStartTime;
                        if (elapsed > VIDEO_PREPARE_TIMEOUT_SEC)
                        {
                            Debug.LogWarning($"[TexMotion] Video preparation timed out ({elapsed:F1}s). Proceeding with 3D avatar preview.");
                            _videoPlayerFailed = true;
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
            if (_currentMotionData != null && _currentMotionData.Frames > 0)
            {
                if (_currentMotionData is VideoMotionData vmd && vmd.Duration > 0f)
                {
                    return vmd.Duration;
                }
                return _currentMotionData.Frames / _currentMotionData.FrameRate;
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
                _statusMessage = $"Failed to load model: {error}";
                return false;
            }

            _statusMessage = "Kimodo engine loaded & ready.";
            return true;
        }

        private void OnGUI()
        {
            DrawHeader();
            DrawTabBar();

            _scrollPos = EditorGUILayout.BeginScrollView(_scrollPos);
            EditorGUILayout.Space(10);

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

        private void DrawHeader()
        {
            var headerRect = EditorGUILayout.GetControlRect(false, 55);
            EditorGUI.DrawRect(headerRect, new Color(0.12f, 0.14f, 0.20f));

            var labelStyle = new GUIStyle(EditorStyles.boldLabel)
            {
                fontSize = 18,
                normal = { textColor = new Color(0.4f, 0.8f, 1.0f) },
                alignment = TextAnchor.MiddleLeft
            };

            var subStyle = new GUIStyle(EditorStyles.miniLabel)
            {
                normal = { textColor = new Color(0.7f, 0.75f, 0.85f) },
                alignment = TextAnchor.MiddleLeft
            };

            GUI.Label(new Rect(headerRect.x + 15, headerRect.y + 8, headerRect.width - 30, 24), "✨ TexMotion Studio", labelStyle);
            GUI.Label(new Rect(headerRect.x + 15, headerRect.y + 30, headerRect.width - 30, 18), "AI-Powered Text & Video-to-Motion for VRChat Avatars", subStyle);
        }

        private void DrawTabBar()
        {
            EditorGUILayout.Space(5);
            EditorGUILayout.BeginHorizontal();
            
            GUI.backgroundColor = _currentTab == Tab.Generator ? new Color(0.35f, 0.65f, 0.95f) : new Color(0.25f, 0.25f, 0.25f);
            if (GUILayout.Button("🎬 Motion Generator", GUILayout.Height(30)))
            {
                _currentTab = Tab.Generator;
            }

            GUI.backgroundColor = _currentTab == Tab.VideoMotion ? new Color(0.35f, 0.65f, 0.95f) : new Color(0.25f, 0.25f, 0.25f);
            if (GUILayout.Button("🎥 Video Motion", GUILayout.Height(30)))
            {
                _currentTab = Tab.VideoMotion;
            }

            GUI.backgroundColor = _currentTab == Tab.Library ? new Color(0.35f, 0.65f, 0.95f) : new Color(0.25f, 0.25f, 0.25f);
            if (GUILayout.Button($"📚 Library ({_libraryItems.Count})", GUILayout.Height(30)))
            {
                _currentTab = Tab.Library;
                RefreshLibrary();
            }

            GUI.backgroundColor = _currentTab == Tab.Settings ? new Color(0.35f, 0.65f, 0.95f) : new Color(0.25f, 0.25f, 0.25f);
            if (GUILayout.Button("⚙️ Settings & Models", GUILayout.Height(30)))
            {
                _currentTab = Tab.Settings;
            }

            GUI.backgroundColor = Color.white;
            EditorGUILayout.EndHorizontal();
        }

        private void DrawGeneratorTab()
        {
            var settings = TexMotionSettings.instance;
            bool modelsReady = settings.AreModelsPresent();

            DrawModelStatusBanner(modelsReady);
            EditorGUILayout.Space(10);

            // 1. Avatar Selection
            EditorGUILayout.BeginVertical(EditorStyles.helpBox);
            EditorGUILayout.LabelField("1. Target Avatar", EditorStyles.boldLabel);
            var prevAvatar = _targetAvatar;
            _targetAvatar = (GameObject)EditorGUILayout.ObjectField("Avatar GameObject", _targetAvatar, typeof(GameObject), true);
            if (prevAvatar != _targetAvatar && _currentMotionData != null)
            {
                SetupPreviewInstance();
            }

            if (_targetAvatar == null)
            {
                EditorGUILayout.HelpBox("Please select your VRChat Avatar from the Hierarchy.", MessageType.Info);
            }
            EditorGUILayout.EndVertical();

            EditorGUILayout.Space(10);

            // 2. Prompt Input
            EditorGUILayout.BeginVertical(EditorStyles.helpBox);
            EditorGUILayout.LabelField("2. Motion Prompt", EditorStyles.boldLabel);
            EditorGUILayout.LabelField("Describe the animation in natural language:", EditorStyles.miniLabel);
            _prompt = EditorGUILayout.TextArea(_prompt, GUILayout.Height(48));

            EditorGUILayout.BeginHorizontal();
            if (GUILayout.Button("🥊 Punch", EditorStyles.miniButton)) { _prompt = "a person throws a sharp right punch forward with energy"; _motionName = "Punch"; _handPose = HandPoseType.Fist; _faceEmotion = FaceEmotionType.Angry; }
            if (GUILayout.Button("👋 Wave", EditorStyles.miniButton)) { _prompt = "a person waves both hands warmly and smiles"; _motionName = "WaveHands"; _handPose = HandPoseType.NaturalRelaxed; _faceEmotion = FaceEmotionType.Smile; }
            if (GUILayout.Button("🕺 Dance", EditorStyles.miniButton)) { _prompt = "a dynamic hip hop dance routine"; _motionName = "Dance"; _handPose = HandPoseType.OpenPalm; }
            if (GUILayout.Button("🧘 Stretch", EditorStyles.miniButton)) { _prompt = "stretching arms overhead happily"; _motionName = "Stretch"; }
            EditorGUILayout.EndHorizontal();
            EditorGUILayout.EndVertical();

            EditorGUILayout.Space(10);

            // 3. Setup Target Configuration
            DrawSetupTargetSection(false);

            EditorGUILayout.Space(10);

            // 4. Assistance & Quality
            EditorGUILayout.BeginVertical(EditorStyles.helpBox);
            EditorGUILayout.LabelField("4. Hand & Face Assistance", EditorStyles.boldLabel);
            _handPose = (HandPoseType)EditorGUILayout.EnumPopup("Hand Pose", _handPose);
            _faceEmotion = (FaceEmotionType)EditorGUILayout.EnumPopup("Face Emotion", _faceEmotion);
            if (_faceEmotion != FaceEmotionType.None)
            {
                _emotionIntensity = EditorGUILayout.Slider("Emotion Intensity", _emotionIntensity, 0.1f, 1.0f);
            }

            EditorGUILayout.Space(5);
            EditorGUILayout.LabelField("Generation Quality", EditorStyles.boldLabel);
            _frames = (uint)EditorGUILayout.IntSlider("Duration (Frames)", (int)_frames, 20, 180);
            EditorGUILayout.LabelField($"Approx. Duration: {(_frames / 20.0f):F1} seconds", EditorStyles.miniLabel);

            _steps = (uint)EditorGUILayout.IntSlider("Diffusion Steps", (int)_steps, 5, 50);
            _textCfg = EditorGUILayout.Slider("Text CFG Scale", _textCfg, 1.0f, 5.0f);

            _randomizeSeed = EditorGUILayout.Toggle("Randomize Seed", _randomizeSeed);
            if (!_randomizeSeed)
            {
                _seed = (ulong)EditorGUILayout.LongField("Seed", (long)_seed);
            }
            EditorGUILayout.EndVertical();

            EditorGUILayout.Space(12);

            // 5. Generate Button
            GUI.enabled = !_isGenerating && !_isDownloading && modelsReady && _targetAvatar != null && !string.IsNullOrWhiteSpace(_prompt);
            GUI.backgroundColor = new Color(0.2f, 0.70f, 0.95f);

            if (GUILayout.Button(_isGenerating ? "⏳ Generating Motion..." : "🚀 1. Generate Motion Preview", GUILayout.Height(40)))
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
                    customPath = TexMotionSettings.instance?.CustomPythonExecutablePath;
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
                Repaint();
            }
        }

        private async void InstallPythonDependenciesAsync()
        {
            if (_isInstallingDependencies) return;
            _isInstallingDependencies = true;
            _statusMessage = "Installing Python dependencies (mediapipe, opencv-python, numpy, scipy)...";
            Repaint();

            try
            {
                string pyExe = _detectedPythonInfo?.ExecutablePath;
                if (string.IsNullOrEmpty(pyExe) || !File.Exists(pyExe))
                {
                    pyExe = "python";
                }

                await Task.Run(() =>
                {
                    var psi = new ProcessStartInfo
                    {
                        FileName = pyExe,
                        Arguments = "-m pip install mediapipe opencv-python numpy scipy --user",
                        UseShellExecute = false,
                        RedirectStandardOutput = true,
                        RedirectStandardError = true,
                        CreateNoWindow = true
                    };
                    using var proc = Process.Start(psi);
                    proc?.WaitForExit();
                });

                _statusMessage = "Dependencies installed successfully!";
                CheckPythonEnvironmentAsync();
            }
            catch (Exception ex)
            {
                _statusMessage = $"Installation failed: {ex.Message}";
                Debug.LogError($"[TexMotion] {ex}");
            }
            finally
            {
                _isInstallingDependencies = false;
                Repaint();
            }
        }

        private void DrawPythonEnvironmentBanner()
        {
            EditorGUILayout.BeginHorizontal(EditorStyles.helpBox);

            if (_isCheckingPython)
            {
                GUI.color = new Color(0.7f, 0.7f, 0.7f);
                EditorGUILayout.LabelField("⏳ Checking Python environment...", EditorStyles.miniLabel);
                GUI.color = Color.white;
            }
            else if (_isInstallingDependencies)
            {
                GUI.color = new Color(0.35f, 0.75f, 0.95f);
                EditorGUILayout.LabelField("📦 Installing Python dependencies via pip...", EditorStyles.boldLabel);
                GUI.color = Color.white;
            }
            else if (_detectedPythonInfo != null && _detectedPythonInfo.IsFullyConfigured)
            {
                GUI.color = new Color(0.3f, 0.95f, 0.4f);
                EditorGUILayout.LabelField($"● MediaPipe & OpenCV Ready ({_detectedPythonInfo.Version})", EditorStyles.boldLabel, GUILayout.Width(250));
                GUI.color = Color.white;
                EditorGUILayout.LabelField(_detectedPythonInfo.ExecutablePath, EditorStyles.miniLabel);
            }
            else
            {
                GUI.color = new Color(1.0f, 0.45f, 0.35f);
                EditorGUILayout.LabelField("⚠️ Python Dependencies Missing", EditorStyles.boldLabel, GUILayout.Width(220));
                GUI.color = Color.white;

                if (GUILayout.Button("📦 Install Dependencies", GUILayout.Height(22)))
                {
                    InstallPythonDependenciesAsync();
                }

                if (GUILayout.Button("🔄 Re-check", GUILayout.Width(75), GUILayout.Height(22)))
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
            EditorGUILayout.Space(6);

            // 1. Target Avatar Selector
            EditorGUILayout.BeginVertical(EditorStyles.helpBox);
            EditorGUILayout.LabelField("1. Target Avatar", EditorStyles.boldLabel);
            var prevAvatar = _targetAvatar;
            _targetAvatar = (GameObject)EditorGUILayout.ObjectField("Avatar GameObject", _targetAvatar, typeof(GameObject), true);
            if (prevAvatar != _targetAvatar && _currentMotionData != null)
            {
                SetupPreviewInstance();
            }

            if (_targetAvatar == null)
            {
                EditorGUILayout.HelpBox("Please select your VRChat Avatar from the Hierarchy.", MessageType.Info);
            }
            EditorGUILayout.EndVertical();

            EditorGUILayout.Space(10);

            // 2. Video File Selector (Browse + Drag & Drop)
            EditorGUILayout.BeginVertical(EditorStyles.helpBox);
            EditorGUILayout.LabelField("2. Video Source File", EditorStyles.boldLabel);

            // Drag & Drop Box Area
            Rect dropRect = GUILayoutUtility.GetRect(0f, 65f, GUILayout.ExpandWidth(true));
            GUI.Box(dropRect, GUIContent.none, EditorStyles.helpBox);

            var dropStyle = new GUIStyle(EditorStyles.centeredGreyMiniLabel)
            {
                fontSize = 12,
                fontStyle = FontStyle.Bold,
                normal = { textColor = new Color(0.4f, 0.75f, 1.0f) },
                alignment = TextAnchor.MiddleCenter
            };

            string dropText = string.IsNullOrEmpty(_videoPath)
                ? "📂 Drag & Drop Video File Here (.mp4, .mov, .webm, .avi)\n— or click 'Browse Video' below —"
                : $"Selected: {Path.GetFileName(_videoPath)}\n(Drag & drop another video file to replace)";

            GUI.Label(dropRect, dropText, dropStyle);

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
            _videoAsset = EditorGUILayout.ObjectField("Video Asset / Clip", _videoAsset, typeof(UnityEngine.Object), false);
            if (_videoAsset != prevAsset && _videoAsset != null)
            {
                string assetPath = AssetDatabase.GetAssetPath(_videoAsset);
                if (!string.IsNullOrEmpty(assetPath))
                {
                    SetVideoSourcePath(Path.GetFullPath(assetPath));
                }
            }

            if (GUILayout.Button("📂 Browse Video...", GUILayout.Width(130), GUILayout.Height(20)))
            {
                string picked = EditorUtility.OpenFilePanel("Select Video for Motion Extraction", "", "mp4,mov,webm,avi,mkv");
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
                EditorGUILayout.LabelField("File Path:", GUILayout.Width(65));
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
                EditorGUILayout.HelpBox("No video file selected. You can select an MP4 video or test with a synthetic walking motion.", MessageType.Info);
                if (GUILayout.Button("🧪 Use Test Motion", GUILayout.Width(135), GUILayout.Height(30)))
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
            EditorGUILayout.BeginVertical(EditorStyles.helpBox);
            EditorGUILayout.LabelField("3. Extraction Settings & Video Info", EditorStyles.boldLabel);

            if (!string.IsNullOrEmpty(_videoPath))
            {
                EditorGUILayout.BeginVertical(EditorStyles.textArea);
                bool isSynthetic = _videoPath == "synthetic_motion_test";
                EditorGUILayout.LabelField($"📹 Source: {(isSynthetic ? "Kinematic Synthetic Generator" : Path.GetFileName(_videoPath))}", EditorStyles.boldLabel);

                if (!isSynthetic && File.Exists(_videoPath))
                {
                    var fileInfo = new FileInfo(_videoPath);
                    float sizeMb = fileInfo.Length / (1024f * 1024f);
                    string metaStr = $"Size: {sizeMb:F1} MB";
                    if (_detectedVideoDuration > 0f)
                    {
                        metaStr += $" | Duration: {_detectedVideoDuration:F2}s | Resolution: {_detectedVideoWidth}x{_detectedVideoHeight} | FPS: {_detectedVideoFps:F1}";
                    }
                    EditorGUILayout.LabelField(metaStr, EditorStyles.miniLabel);
                }
                EditorGUILayout.EndVertical();
                EditorGUILayout.Space(6);
            }

            _videoTargetFps = EditorGUILayout.Slider("Target FPS", _videoTargetFps, 15f, 60f);
            _videoTargetFps = Mathf.Round(_videoTargetFps);

            EditorGUILayout.Space(4);
            EditorGUILayout.LabelField("✂️ Video Trimming (Seconds)", EditorStyles.boldLabel);
            EditorGUILayout.BeginHorizontal();
            _videoTrimStart = EditorGUILayout.FloatField("Start Time (s)", _videoTrimStart);
            _videoTrimStart = Mathf.Max(0f, _videoTrimStart);

            _videoTrimEnd = EditorGUILayout.FloatField("End Time (0=full)", _videoTrimEnd);
            _videoTrimEnd = Mathf.Max(0f, _videoTrimEnd);
            EditorGUILayout.EndHorizontal();

            if (_detectedVideoDuration > 0f && _videoTrimEnd > _detectedVideoDuration)
            {
                _videoTrimEnd = _detectedVideoDuration;
            }

            if (_videoTrimEnd > 0f && _videoTrimEnd > _videoTrimStart)
            {
                float trimmedDuration = _videoTrimEnd - _videoTrimStart;
                int estFrames = Mathf.RoundToInt(trimmedDuration * _videoTargetFps);
                EditorGUILayout.LabelField($"Trimmed Segment: {_videoTrimStart:F2}s to {_videoTrimEnd:F2}s ({trimmedDuration:F2}s, ~{estFrames} frames)", EditorStyles.miniLabel);
            }
            else
            {
                EditorGUILayout.LabelField("Trimming: Full video duration will be processed.", EditorStyles.miniLabel);
            }

            EditorGUILayout.Space(6);
            EditorGUILayout.LabelField("⚙️ Extraction Quality & Processing", EditorStyles.boldLabel);
            _videoInPlace = EditorGUILayout.Toggle("In-Place Root (Keep hips centered)", _videoInPlace);
            _videoSmoothing = EditorGUILayout.Toggle("Temporal Smoothing (Savitzky-Golay)", _videoSmoothing);
            _videoFootLocking = EditorGUILayout.Toggle("Foot Locking & Floor Snapping", _videoFootLocking);
            _videoModelComplexity = EditorGUILayout.IntPopup("MediaPipe Complexity", _videoModelComplexity,
                new string[] { "0 - Fast / Lightweight", "1 - Balanced (Recommended)", "2 - High Precision" },
                new int[] { 0, 1, 2 });
            _videoMinConfidenceThreshold = EditorGUILayout.Slider("Min Confidence Cutoff", _videoMinConfidenceThreshold, 0.0f, 0.8f);

            EditorGUILayout.EndVertical();

            EditorGUILayout.Space(10);

            // 4. Hand & Face Assistance
            EditorGUILayout.BeginVertical(EditorStyles.helpBox);
            EditorGUILayout.LabelField("4. Hand & Face Assistance", EditorStyles.boldLabel);
            _videoHandPose = (HandPoseType)EditorGUILayout.EnumPopup("Hand Pose", _videoHandPose);
            _videoFaceEmotion = (FaceEmotionType)EditorGUILayout.EnumPopup("Face Emotion", _videoFaceEmotion);
            if (_videoFaceEmotion != FaceEmotionType.None)
            {
                _videoEmotionIntensity = EditorGUILayout.Slider("Emotion Intensity", _videoEmotionIntensity, 0.1f, 1.0f);
            }
            EditorGUILayout.EndVertical();

            EditorGUILayout.Space(10);

            // 5. Setup Target Section
            DrawSetupTargetSection(true);

            EditorGUILayout.Space(12);

            // 6. Extraction Action Button & Real-time Progress Bar
            bool canExtract = !_isVideoExtracting && !_isGenerating && !_isDownloading &&
                _targetAvatar != null &&
                (!string.IsNullOrEmpty(_videoPath) && (File.Exists(_videoPath) || _videoPath == "synthetic_motion_test"));

            GUI.enabled = canExtract;
            GUI.backgroundColor = new Color(0.2f, 0.70f, 0.95f);

            if (GUILayout.Button(_isVideoExtracting ? "⏳ Extracting 3D Humanoid Motion..." : "🎥 1. Extract Motion from Video", GUILayout.Height(42)))
            {
                StartVideoExtraction();
            }

            GUI.backgroundColor = Color.white;
            GUI.enabled = true;

            if (_isVideoExtracting)
            {
                EditorGUILayout.Space(8);
                EditorGUILayout.BeginVertical(EditorStyles.helpBox);
                EditorGUILayout.LabelField("Extraction in Progress", EditorStyles.boldLabel);
                EditorGUI.ProgressBar(EditorGUILayout.GetControlRect(false, 20), _videoExtractionProgress, _videoExtractionStatus);
                EditorGUILayout.Space(4);
                if (GUILayout.Button("🛑 Cancel Extraction", GUILayout.Height(26)))
                {
                    _videoCts?.Cancel();
                }
                EditorGUILayout.EndVertical();
            }

            // 7. Video Motion Result Card & Interactive 3D Preview Viewport
            if (_currentVideoMotionData != null)
            {
                EditorGUILayout.Space(15);
                DrawVideoMotionResultCard();
                EditorGUILayout.Space(8);
                DrawInteractive3DPreviewSection();
            }
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
                }
            }
        }

        private void DrawVideoMotionResultCard()
        {
            if (_currentVideoMotionData == null) return;

            EditorGUILayout.BeginVertical(EditorStyles.helpBox);
            EditorGUILayout.BeginHorizontal();
            EditorGUILayout.LabelField("✅ Extracted Motion Metrics", EditorStyles.boldLabel);

            float avgConf = _currentVideoMotionData.AverageConfidence;
            if (avgConf >= 0.70f)
            {
                GUI.color = new Color(0.3f, 0.95f, 0.4f);
                EditorGUILayout.LabelField($"● High Confidence ({(avgConf * 100f):F1}%)", EditorStyles.boldLabel, GUILayout.Width(180));
            }
            else if (avgConf >= 0.45f)
            {
                GUI.color = new Color(0.95f, 0.75f, 0.25f);
                EditorGUILayout.LabelField($"● Moderate Confidence ({(avgConf * 100f):F1}%)", EditorStyles.boldLabel, GUILayout.Width(180));
            }
            else
            {
                GUI.color = new Color(1.0f, 0.5f, 0.3f);
                EditorGUILayout.LabelField($"● Low Confidence ({(avgConf * 100f):F1}%)", EditorStyles.boldLabel, GUILayout.Width(180));
            }
            GUI.color = Color.white;
            EditorGUILayout.EndHorizontal();

            EditorGUILayout.LabelField(
                $"Frames: {_currentVideoMotionData.Frames} | Length: {_currentVideoMotionData.Duration:F2}s | FPS: {_currentVideoMotionData.FrameRate:F0} | Confidence Range: [{(_currentVideoMotionData.MinConfidence * 100f):F0}% - {(_currentVideoMotionData.MaxConfidence * 100f):F0}%]",
                EditorStyles.miniLabel);

            if (_currentVideoMotionData.HasVariableTimestamps)
            {
                EditorGUILayout.LabelField("ℹ️ Variable frame timestamps detected — synchronized with precision time interpolation.", EditorStyles.miniLabel);
            }

            EditorGUILayout.BeginHorizontal();
            if (GUILayout.Button("✨ Interpolate Low-Confidence Outliers", EditorStyles.miniButton))
            {
                _currentVideoMotionData.InterpolateOutliers(_videoMinConfidenceThreshold > 0f ? _videoMinConfidenceThreshold : 0.4f);
                _statusMessage = "Interpolated low-confidence outlier frames.";
                SetupPreviewInstance();
                Repaint();
            }
            EditorGUILayout.EndHorizontal();

            EditorGUILayout.EndVertical();
        }

        private void DrawSetupTargetSection(bool isVideo = false)
        {
            EditorGUILayout.BeginVertical(EditorStyles.helpBox);
            EditorGUILayout.LabelField(isVideo ? "5. VRChat Setup Target & Menu" : "3. VRChat Setup Target & Destination Menu", EditorStyles.boldLabel);

            if (isVideo)
            {
                _videoMotionName = EditorGUILayout.TextField("Motion Name", _videoMotionName);
            }
            else
            {
                _motionName = EditorGUILayout.TextField("Motion Name", _motionName);
            }

            _setupMode = (VrcSetupMode)EditorGUILayout.EnumPopup("Setup Target", _setupMode);
            _motionType = (VrcMotionType)EditorGUILayout.EnumPopup("Motion Playback", _motionType);
            _targetLayer = (VrcTargetLayer)EditorGUILayout.EnumPopup("Target Layer", _targetLayer);

            if (!isVideo)
            {
                _inPlace = EditorGUILayout.Toggle("In-Place (Stay on origin)", _inPlace);
            }

            if (_setupMode == VrcSetupMode.DirectVRCSDK)
            {
                EditorGUILayout.Space(6);
                EditorGUILayout.LabelField("🎯 Destination Expressions Menu:", EditorStyles.boldLabel);

                _customTargetMenu = (ScriptableObject)EditorGUILayout.ObjectField("Target Menu Asset", _customTargetMenu, typeof(ScriptableObject), false);

                if (_targetAvatar != null)
                {
                    var availableMenus = VrcDirectSetup.FindAllAvatarMenus(_targetAvatar);
                    if (availableMenus.Count > 0)
                    {
                        EditorGUILayout.LabelField("Quick Select from Avatar's Menus:", EditorStyles.miniLabel);
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

                string currentSelectedName = _customTargetMenu != null ? _customTargetMenu.name : "Root Menu";
                EditorGUILayout.HelpBox($"Motion button will be added directly into '{currentSelectedName}'.", MessageType.Info);
            }
            else
            {
                EditorGUILayout.HelpBox("Modular Avatar Mode: Will create a non-destructive child object under your avatar.", MessageType.Info);
            }

            EditorGUILayout.EndVertical();
        }

        private void DrawLibraryTab()
        {
            EditorGUILayout.BeginHorizontal();
            EditorGUILayout.LabelField("📚 Saved Motion Library", EditorStyles.boldLabel);
            if (GUILayout.Button("🔄 Refresh", GUILayout.Width(80)))
            {
                RefreshLibrary();
            }
            EditorGUILayout.EndHorizontal();

            EditorGUILayout.LabelField("Manage, preview, and add/remove generated motions on your avatar.", EditorStyles.miniLabel);
            EditorGUILayout.Space(8);

            // Target Avatar selector
            EditorGUILayout.BeginHorizontal(EditorStyles.helpBox);
            EditorGUILayout.LabelField("Target Avatar:", GUILayout.Width(90));
            _targetAvatar = (GameObject)EditorGUILayout.ObjectField(_targetAvatar, typeof(GameObject), true);
            EditorGUILayout.EndHorizontal();
            EditorGUILayout.Space(5);

            DrawSetupTargetSection(false);
            EditorGUILayout.Space(8);

            if (_libraryItems.Count == 0)
            {
                EditorGUILayout.HelpBox("No generated motions found in Assets/TexMotion/Generated. Generate a new motion in 'Motion Generator' or extract one in 'Video Motion'!", MessageType.Info);
                return;
            }

            for (int i = 0; i < _libraryItems.Count; i++)
            {
                var item = _libraryItems[i];
                bool isSelected = _selectedLibraryItem == item;

                EditorGUILayout.BeginVertical(isSelected ? EditorStyles.selectionRect : EditorStyles.helpBox);

                EditorGUILayout.BeginHorizontal();
                
                EditorGUILayout.BeginVertical();
                EditorGUILayout.LabelField($"🎬 {item.Name}", EditorStyles.boldLabel);
                EditorGUILayout.LabelField($"{item.Duration:F2}s ({item.FrameRate:F0}fps) | {(item.IsLoop ? "Loop" : "One-Shot")}", EditorStyles.miniLabel);
                EditorGUILayout.EndVertical();

                if (item.IsAppliedToAvatar)
                {
                    GUI.color = new Color(0.3f, 0.9f, 0.4f);
                    EditorGUILayout.LabelField("● Applied to Avatar", EditorStyles.boldLabel, GUILayout.Width(130));
                    GUI.color = Color.white;
                }
                else
                {
                    GUI.color = new Color(0.6f, 0.6f, 0.6f);
                    EditorGUILayout.LabelField("○ Asset Only", EditorStyles.miniLabel, GUILayout.Width(90));
                    GUI.color = Color.white;
                }

                GUI.backgroundColor = isSelected ? new Color(0.95f, 0.65f, 0.2f) : Color.white;
                if (GUILayout.Button(isSelected ? "👁 Viewing" : "👁 Preview", GUILayout.Width(75), GUILayout.Height(28)))
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
                if (GUILayout.Button("✏️ Edit", GUILayout.Width(60), GUILayout.Height(28)))
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
                        EditorUtility.DisplayDialog("Target Avatar Required", "Please assign a humanoid Target Avatar above to edit this clip.", "OK");
                    }
                }
                GUI.backgroundColor = Color.white;

                if (item.IsAppliedToAvatar)
                {
                    GUI.backgroundColor = new Color(0.9f, 0.4f, 0.4f);
                    if (GUILayout.Button("❌ Remove", GUILayout.Width(75), GUILayout.Height(28)))
                    {
                        MotionLibraryManager.RemoveFromAvatar(_targetAvatar, item);
                        RefreshLibrary();
                    }
                    GUI.backgroundColor = Color.white;
                }
                else
                {
                    GUI.backgroundColor = new Color(0.3f, 0.85f, 0.45f);
                    if (GUILayout.Button("✨ Apply", GUILayout.Width(75), GUILayout.Height(28)))
                    {
                        ApplyLibraryItemToAvatar(item);
                        RefreshLibrary();
                    }
                    GUI.backgroundColor = Color.white;
                }

                GUI.backgroundColor = new Color(0.5f, 0.5f, 0.5f);
                if (GUILayout.Button("🗑️", GUILayout.Width(30), GUILayout.Height(28)))
                {
                    if (EditorUtility.DisplayDialog("Delete Motion", $"Are you sure you want to completely delete '{item.Name}' from project?", "Delete", "Cancel"))
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
            EditorGUILayout.BeginVertical(EditorStyles.helpBox);

            EditorGUILayout.LabelField($"🎬 Previewing Library Motion: '{_selectedLibraryItem.Name}'", EditorStyles.boldLabel);
            EditorGUILayout.LabelField($"Duration: {_activePreviewClip.length:F2}s | FPS: {_activePreviewClip.frameRate:F0}", EditorStyles.miniLabel);

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
            _previewTime = EditorGUILayout.Slider("Timeline", _previewTime, 0f, _activePreviewClip.length);
            if (EditorGUI.EndChangeCheck())
            {
                _isPlayingPreview = false;
                if (_previewInstance != null)
                {
                    _activePreviewClip.SampleAnimation(_previewInstance, _previewTime);
                }
            }
            EditorGUILayout.LabelField($"{_previewTime:F2}s / {_activePreviewClip.length:F2}s", GUILayout.Width(80));
            EditorGUILayout.EndHorizontal();

            EditorGUILayout.BeginHorizontal();
            GUI.backgroundColor = _isPlayingPreview ? new Color(0.95f, 0.65f, 0.2f) : new Color(0.3f, 0.85f, 0.45f);
            if (GUILayout.Button(_isPlayingPreview ? "⏸ Pause" : "▶ Play", GUILayout.Height(28)))
            {
                _isPlayingPreview = !_isPlayingPreview;
                if (_isPlayingPreview) _lastUpdateTime = EditorApplication.timeSinceStartup;
            }
            GUI.backgroundColor = Color.white;

            if (GUILayout.Button("⏹ Reset", GUILayout.Height(28), GUILayout.Width(75)))
            {
                _isPlayingPreview = false;
                _previewTime = 0f;
                if (_previewInstance != null) _activePreviewClip.SampleAnimation(_previewInstance, 0f);
            }

            if (GUILayout.Button("🔄 Reset View", GUILayout.Height(28), GUILayout.Width(95)))
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
                _statusMessage = $"Applied '{item.Name}' to {_targetAvatar.name} via Modular Avatar!";
                EditorGUIUtility.PingObject(setupObj);
            }
            else
            {
                VrcDirectSetup.SetupDirectAvatarMotion(_targetAvatar, item.Clip, vrcConfig, _customTargetMenu, MotionLibraryManager.GeneratedDirectory);
                string menuName = _customTargetMenu != null ? _customTargetMenu.name : "VRCExpressionsMenu";
                _statusMessage = $"Applied '{item.Name}' to {menuName}!";
            }

            RefreshLibrary();
        }

        private void DrawInteractive3DPreviewSection()
        {
            EditorGUILayout.BeginVertical(EditorStyles.helpBox);

            float duration = GetActivePreviewDuration();
            VideoMotionData vmd = _currentMotionData as VideoMotionData;

            if (vmd != null)
            {
                string videoToPlay = !string.IsNullOrEmpty(vmd.OverlayVideoPath) && File.Exists(vmd.OverlayVideoPath)
                    ? vmd.OverlayVideoPath
                    : (!string.IsNullOrEmpty(vmd.SourceVideoPath) && File.Exists(vmd.SourceVideoPath) ? vmd.SourceVideoPath : null);

                if (!string.IsNullOrEmpty(videoToPlay) && (_videoPlayer == null || _loadedVideoPath != videoToPlay || (_videoPlayerFailed && _isPlayingPreview)))
                {
                    SetupVideoPlayer(videoToPlay);
                }

                // Viewport Toolbar with Display Modes
                EditorGUILayout.BeginHorizontal();
                EditorGUILayout.LabelField("🎥 Synchronized Side-by-Side Motion Preview", EditorStyles.boldLabel);
                GUILayout.FlexibleSpace();
                _previewDisplayMode = (PreviewDisplayMode)GUILayout.Toolbar((int)_previewDisplayMode,
                    new string[] { "Side-by-Side", "Avatar Only", "Video Only" },
                    EditorStyles.miniButton, GUILayout.Width(270));
                EditorGUILayout.EndHorizontal();

                EditorGUILayout.LabelField(
                    $"Frames: {vmd.Frames} | Length: {duration:F2}s | FPS: {vmd.FrameRate:F0} | Confidence: {(vmd.AverageConfidence * 100f):F1}%",
                    EditorStyles.miniLabel);
            }
            else
            {
                EditorGUILayout.LabelField("🎬 2. 3D Motion Preview (Drag to Rotate, Scroll to Zoom)", EditorStyles.boldLabel);
                EditorGUILayout.LabelField($"Frames: {_currentMotionData.Frames} | Length: {duration:F2}s | Hand: {_handPose} | Face: {_faceEmotion}", EditorStyles.miniLabel);
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
            _previewTime = EditorGUILayout.Slider("Timeline", _previewTime, 0f, duration);
            if (EditorGUI.EndChangeCheck())
            {
                _isPlayingPreview = false;
                ApplyMotionPoseToPreview(_previewTime);
                SyncVideoPlayer(_previewTime, false);
            }
            EditorGUILayout.LabelField($"{_previewTime:F2}s / {duration:F2}s", GUILayout.Width(85));
            EditorGUILayout.EndHorizontal();

            // Playback Control Buttons
            EditorGUILayout.BeginHorizontal();
            GUI.backgroundColor = _isPlayingPreview ? new Color(0.95f, 0.65f, 0.2f) : new Color(0.3f, 0.85f, 0.45f);
            if (GUILayout.Button(_isPlayingPreview ? "⏸ Pause" : "▶ Play", GUILayout.Height(30)))
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

            if (GUILayout.Button("⏹ Reset", GUILayout.Height(30), GUILayout.Width(75)))
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

            if (GUILayout.Button("🔄 Reset View", GUILayout.Height(30), GUILayout.Width(95)))
            {
                _previewDir = new Vector2(180f, 10f);
                _previewDistance = 2.8f;
            }
            EditorGUILayout.EndHorizontal();

            EditorGUILayout.Space(14);
            EditorGUILayout.LabelField("Apply to Avatar:", EditorStyles.boldLabel);

            EditorGUILayout.BeginHorizontal();

            GUI.backgroundColor = new Color(0.95f, 0.7f, 0.2f);
            if (GUILayout.Button("✏️ Edit in Timeline", GUILayout.Height(40)))
            {
                OpenInTimelineEditor();
            }

            GUI.backgroundColor = new Color(0.2f, 0.85f, 0.45f);
            if (GUILayout.Button("✨ Apply to Avatar", GUILayout.Height(40)))
            {
                ApplyMotionToAvatar();
            }

            GUI.backgroundColor = new Color(0.4f, 0.65f, 0.95f);
            if (GUILayout.Button("💾 Save .anim Only", GUILayout.Height(40)))
            {
                SaveClipOnly();
            }

            GUI.backgroundColor = new Color(0.85f, 0.35f, 0.35f);
            if (GUILayout.Button("🗑️ Discard", GUILayout.Height(40), GUILayout.Width(80)))
            {
                _currentMotionData = null;
                _currentVideoMotionData = null;
                CleanupPreviewInstance();
                _statusMessage = "Preview discarded.";
            }

            GUI.backgroundColor = Color.white;
            EditorGUILayout.EndHorizontal();

            EditorGUILayout.EndVertical();
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
                GUI.Label(rect, "Video Stream Not Available", EditorStyles.centeredGreyMiniLabel);
            }

            // Draw HUD Badges strictly during Repaint to eliminate flickering / blinking
            if (Event.current.type == EventType.Repaint)
            {
                // Top-Left HUD Badge: Stream Title
                if (rect.width > 180f && rect.height > 40f)
                {
                    Rect titleBadge = new Rect(rect.x + 8, rect.y + 8, 175, 20);
                    EditorGUI.DrawRect(titleBadge, new Color(0.05f, 0.07f, 0.11f, 0.75f));
                    var badgeStyle = new GUIStyle(EditorStyles.miniLabel)
                    {
                        fontStyle = FontStyle.Bold,
                        normal = { textColor = new Color(0.35f, 0.85f, 1.0f) },
                        padding = new RectOffset(6, 0, 2, 0)
                    };
                    GUI.Label(titleBadge, "● MediaPipe Pose Overlay", badgeStyle);
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
                    GUI.Label(confBadge, $"Conf: {(conf * 100f):F0}%", confStyle);
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
                    GUI.Label(frameBadge, $"Frame: {frameIdx + 1} / {vmd.Frames}", frameStyle);
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
            GUI.Label(titleRect, "⏳ Preparing Pose Overlay Video...", titleStyle);

            // 5. Subtitle Status
            var subStyle = new GUIStyle(EditorStyles.miniLabel)
            {
                alignment = TextAnchor.MiddleCenter,
                fontSize = 10,
                normal = { textColor = new Color(0.65f, 0.78f, 0.92f, 0.85f) }
            };
            Rect subRect = new Rect(cardRect.x + 10, cardRect.y + 34, cardRect.width - 20, 16);
            GUI.Label(subRect, "Loading frames & buffering stream...", subStyle);

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
            GUI.Label(dotRect, $"Buffering video stream{dots}", dotStyle);
        }

        private void DrawVideoFallbackOverlay(Rect viewportRect)
        {
            if (Event.current.type != EventType.Repaint)
                return;

            float cardWidth = Mathf.Min(340f, viewportRect.width - 24f);
            float cardHeight = Mathf.Min(118f, viewportRect.height - 20f);
            if (cardWidth < 80f || cardHeight < 40f) return;

            float cardX = viewportRect.x + (viewportRect.width - cardWidth) * 0.5f;
            float cardY = viewportRect.y + (viewportRect.height - cardHeight) * 0.5f;
            Rect cardRect = new Rect(cardX, cardY, cardWidth, cardHeight);

            // Card Background & Border with amber warning accent
            EditorGUI.DrawRect(cardRect, new Color(0.08f, 0.09f, 0.12f, 0.94f));
            DrawViewportBorder(cardRect, new Color(0.95f, 0.65f, 0.20f, 0.70f));

            var titleStyle = new GUIStyle(EditorStyles.boldLabel)
            {
                alignment = TextAnchor.MiddleCenter,
                fontSize = 12,
                normal = { textColor = new Color(1.0f, 0.85f, 0.45f) }
            };
            Rect titleRect = new Rect(cardRect.x + 10, cardRect.y + 14, cardRect.width - 20, 20);
            GUI.Label(titleRect, "⚠️ Video Overlay Playback Skipped", titleStyle);

            var subStyle = new GUIStyle(EditorStyles.miniLabel)
            {
                alignment = TextAnchor.MiddleCenter,
                fontSize = 10,
                normal = { textColor = new Color(0.80f, 0.85f, 0.92f, 0.85f) }
            };
            Rect subRect = new Rect(cardRect.x + 10, cardRect.y + 36, cardRect.width - 20, 16);
            GUI.Label(subRect, "OS codec decode unavailable or timed out.", subStyle);

            var infoStyle = new GUIStyle(EditorStyles.miniLabel)
            {
                alignment = TextAnchor.MiddleCenter,
                normal = { textColor = new Color(0.35f, 0.90f, 0.55f) },
                fontSize = 10,
                fontStyle = FontStyle.Bold
            };
            Rect infoRect = new Rect(cardRect.x + 10, cardRect.y + 66, cardRect.width - 20, 20);
            GUI.Label(infoRect, "✓ 3D Avatar Motion Playing Smoothly", infoStyle);
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
                    GUI.Label(rect, "Loading 3D Preview...", EditorStyles.centeredGreyMiniLabel);
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
            GUI.Label(titleBadge, "● 3D Avatar (SMPL-X 22)", badgeStyle);

            // Bottom-Right Hint: Orbit / Zoom
            Rect hintBadge = new Rect(rect.x + rect.width - 150, rect.y + rect.height - 24, 142, 18);
            EditorGUI.DrawRect(hintBadge, new Color(0.05f, 0.07f, 0.11f, 0.70f));
            var hintStyle = new GUIStyle(EditorStyles.miniLabel)
            {
                normal = { textColor = new Color(0.70f, 0.75f, 0.85f) },
                alignment = TextAnchor.MiddleCenter
            };
            GUI.Label(hintBadge, "Drag: Orbit | Scroll: Zoom", hintStyle);
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
                    _statusMessage = $"Successfully applied '{motionName}' to {_targetAvatar.name} via Modular Avatar!";
                    EditorGUIUtility.PingObject(setupObj);
                    Selection.activeGameObject = setupObj;
                    EditorUtility.DisplayDialog("Setup Complete", $"Motion '{motionName}' was successfully applied to {_targetAvatar.name} via Modular Avatar!", "Great!");
                }
                else
                {
                    VrcDirectSetup.SetupDirectAvatarMotion(_targetAvatar, clip, vrcConfig, _customTargetMenu, saveDir);
                    string menuName = _customTargetMenu != null ? _customTargetMenu.name : "VRCExpressionsMenu";
                    _statusMessage = $"Successfully added '{motionName}' directly to {menuName} & Animator!";
                    EditorUtility.DisplayDialog("Setup Complete", $"Motion '{motionName}' was successfully added directly into '{menuName}', ExpressionParameters, and {_targetLayer}!", "Great!");
                }
                
                _currentMotionData = null;
                _currentVideoMotionData = null;
                CleanupPreviewInstance();
                RefreshLibrary();
            }
            catch (Exception ex)
            {
                _statusMessage = $"Failed to apply motion: {ex.Message}";
                Debug.LogError($"[TexMotion] {ex}");
                EditorUtility.DisplayDialog("Error", ex.Message, "OK");
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

                _statusMessage = $"Saved AnimationClip to {clipPath}";
                EditorGUIUtility.PingObject(clip);
                Selection.activeObject = clip;

                EditorUtility.DisplayDialog("Saved", $"AnimationClip saved to {clipPath}", "OK");

                _currentMotionData = null;
                _currentVideoMotionData = null;
                CleanupPreviewInstance();
                RefreshLibrary();
            }
            catch (Exception ex)
            {
                _statusMessage = $"Failed to save clip: {ex.Message}";
                Debug.LogError($"[TexMotion] {ex}");
            }
        }

        private void DrawModelStatusBanner(bool ready)
        {
            EditorGUILayout.BeginHorizontal(EditorStyles.helpBox);
            if (ready)
            {
                GUI.color = new Color(0.4f, 0.9f, 0.5f);
                EditorGUILayout.LabelField("● Models Ready", EditorStyles.boldLabel, GUILayout.Width(110));
                GUI.color = Color.white;
                EditorGUILayout.LabelField("Kimodo + LLM2Vec Text Bundle Loaded", EditorStyles.miniLabel);
            }
            else
            {
                GUI.color = new Color(1.0f, 0.4f, 0.4f);
                EditorGUILayout.LabelField("● Models Missing", EditorStyles.boldLabel, GUILayout.Width(110));
                GUI.color = Color.white;
                if (GUILayout.Button("Go to Settings to Download", GUILayout.Height(20)))
                {
                    _currentTab = Tab.Settings;
                }
            }
            EditorGUILayout.EndHorizontal();
        }

        private void DrawSettingsTab()
        {
            var settings = TexMotionSettings.instance;

            EditorGUILayout.BeginVertical(EditorStyles.helpBox);
            EditorGUILayout.LabelField("📦 Hugging Face Repository", EditorStyles.boldLabel);
            settings.HuggingFaceRepo = EditorGUILayout.TextField("HF Repo (User/Repo)", settings.HuggingFaceRepo);
            settings.MotionModelFileName = EditorGUILayout.TextField("Motion GGUF File", settings.MotionModelFileName);
            settings.TextBundleDirName = EditorGUILayout.TextField("Text Bundle Folder", settings.TextBundleDirName);

            EditorGUILayout.Space(5);
            EditorGUILayout.HelpBox($"Target Repo: https://huggingface.co/{settings.HuggingFaceRepo}", MessageType.None);
            EditorGUILayout.EndVertical();

            EditorGUILayout.Space(10);

            EditorGUILayout.BeginVertical(EditorStyles.helpBox);
            EditorGUILayout.LabelField("📁 Local Storage Configuration", EditorStyles.boldLabel);
            settings.UseCustomLocalPath = EditorGUILayout.Toggle("Use Custom Local Path", settings.UseCustomLocalPath);

            if (settings.UseCustomLocalPath)
            {
                EditorGUILayout.BeginHorizontal();
                settings.CustomLocalModelDirectory = EditorGUILayout.TextField("Local Directory", settings.CustomLocalModelDirectory);
                if (GUILayout.Button("Browse...", GUILayout.Width(75)))
                {
                    string dir = EditorUtility.OpenFolderPanel("Select Model Directory", settings.CustomLocalModelDirectory, "");
                    if (!string.IsNullOrEmpty(dir))
                    {
                        settings.CustomLocalModelDirectory = dir;
                    }
                }
                EditorGUILayout.EndHorizontal();
            }
            else
            {
                EditorGUILayout.LabelField("Default Cache Directory (AppData):", EditorStyles.miniLabel);
                EditorGUILayout.SelectableLabel(settings.GetDefaultCacheDirectory(), EditorStyles.textField, GUILayout.Height(20));
            }

            EditorGUILayout.Space(5);
            EditorGUILayout.LabelField("Resolved Paths:", EditorStyles.boldLabel);

            string motionPath = settings.GetMotionModelPath();
            bool motionExists = File.Exists(motionPath);
            EditorGUILayout.LabelField($"Motion Model: {(motionExists ? "✅ Found" : "❌ Not Found")}", EditorStyles.miniLabel);
            EditorGUILayout.SelectableLabel(motionPath, EditorStyles.textField, GUILayout.Height(18));

            string bundleDir = settings.GetTextBundleDirectory();
            bool bundleExists = settings.IsValidTextBundle(bundleDir);
            EditorGUILayout.LabelField($"Text Bundle: {(bundleExists ? "✅ Found" : "❌ Not Found")}", EditorStyles.miniLabel);
            EditorGUILayout.SelectableLabel(bundleDir, EditorStyles.textField, GUILayout.Height(18));

            if (GUILayout.Button("🔄 Reload Engine", GUILayout.Height(24)))
            {
                _engine?.UnloadModel();
                EnsureEngineLoaded();
            }

            EditorGUILayout.EndVertical();

            EditorGUILayout.Space(15);

            // Download Section
            EditorGUILayout.BeginVertical(EditorStyles.helpBox);
            EditorGUILayout.LabelField("⬇️ Model Downloader", EditorStyles.boldLabel);
            EditorGUILayout.LabelField($"Destination: {settings.GetEffectiveModelDirectory()}", EditorStyles.miniLabel);

            GUI.enabled = !_isDownloading && !_isGenerating && !_isVideoExtracting;
            if (GUILayout.Button("📥 Download Models from Hugging Face", GUILayout.Height(35)))
            {
                StartDownload();
            }
            GUI.enabled = true;

            if (_isDownloading)
            {
                if (GUILayout.Button("Cancel Download", GUILayout.Height(25)))
                {
                    _cts?.Cancel();
                }
            }
            EditorGUILayout.EndVertical();

            EditorGUILayout.Space(15);

            // Motion Timeline Editor Settings
            EditorGUILayout.BeginVertical(EditorStyles.helpBox);
            EditorGUILayout.LabelField("⏱️ Motion Timeline Editor", EditorStyles.boldLabel);
            settings.AutoOpenTimelineEditor = EditorGUILayout.Toggle("Auto Open Timeline Editor", settings.AutoOpenTimelineEditor);
            EditorGUILayout.HelpBox("Automatically opens the dedicated Motion Timeline Editor upon completing text motion generation or video pose extraction.", MessageType.None);
            if (GUILayout.Button("Launch Motion Timeline Editor", GUILayout.Height(26)))
            {
                MotionTimelineEditorWindow.ShowEditor();
            }
            EditorGUILayout.EndVertical();

            if (GUI.changed)
            {
                settings.Save();
            }
        }

        private void DrawFooterStatus()
        {
            if (_isGenerating || _isDownloading || _isVideoExtracting)
            {
                EditorGUILayout.Space(5);
                float prog = _isVideoExtracting ? _videoExtractionProgress : _overallProgress;
                string msg = _isVideoExtracting ? _videoExtractionStatus : _statusMessage;
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
            _statusMessage = "Starting download from Hugging Face...";
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
                _statusMessage = "Models downloaded successfully!";
                EnsureEngineLoaded();
            }
            catch (Exception ex)
            {
                _statusMessage = $"Download failed: {ex.Message}";
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
                EditorUtility.DisplayDialog("Error", "Model is not loaded. Please check settings.", "OK");
                return;
            }

            _isGenerating = true;
            _overallProgress = 0.3f;
            _statusMessage = "Running motion diffusion...";
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
                _statusMessage = "Setting up 3D preview viewport...";
                Repaint();

                _activePreviewClip = null;
                _previewTime = 0f;
                _isPlayingPreview = true;
                _lastUpdateTime = EditorApplication.timeSinceStartup;

                SetupPreviewInstance(autoPlay: true);
                _isPlayingPreview = true;

                _overallProgress = 1.0f;
                _statusMessage = $"Motion ready! Previewing on {_targetAvatar.name}.";

                if (TexMotionSettings.instance.AutoOpenTimelineEditor)
                {
                    OpenInTimelineEditor();
                }
            }
            catch (Exception ex)
            {
                _statusMessage = $"Generation error: {ex.Message}";
                Debug.LogError($"[TexMotion] Generation failed: {ex}");
                EditorUtility.DisplayDialog("Generation Error", ex.Message, "OK");
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
                EditorUtility.DisplayDialog("Avatar Missing", "Please select your target Avatar in the scene first.", "OK");
                return;
            }

            _isVideoExtracting = true;
            _isPlayingPreview = false;
            _previewTime = 0f;
            CleanupVideoPlayer();
            CleanupPreviewInstance();
            _currentMotionData = null;
            _currentVideoMotionData = null;

            _videoExtractionProgress = 0.05f;
            _videoExtractionStatus = "Initializing pose extraction pipeline...";
            _statusMessage = "Starting video pose extractor...";
            _videoCts = new CancellationTokenSource();
            Repaint();

            bool isSynthetic = _videoPath == "synthetic_motion_test";

            var options = new VideoExtractionOptions
            {
                VideoPath = isSynthetic ? "" : _videoPath,
                TargetFps = _videoTargetFps,
                TrimStartTime = _videoTrimStart,
                TrimEndTime = _videoTrimEnd,
                InPlace = _videoInPlace,
                TemporalSmoothing = _videoSmoothing,
                FootLocking = _videoFootLocking,
                ModelComplexity = _videoModelComplexity,
                MinDetectionConfidence = 0.5f,
                MinTrackingConfidence = 0.5f,
                Synthetic = isSynthetic
            };

            var progress = new Progress<VideoJobProgress>(p =>
            {
                _videoExtractionProgress = p.Progress;
                _videoExtractionStatus = p.Message;
                _statusMessage = $"[Video] {p.Message}";
                Repaint();
            });

            try
            {
                var result = await VideoMotionJobRunner.RunExtractionAsync(options, progress, _videoCts.Token);

                _currentVideoMotionData = result;
                _currentMotionData = result;

                _videoExtractionProgress = 1.0f;
                _videoExtractionStatus = "Extraction complete! Initializing 3D viewport...";
                _statusMessage = $"Video motion extracted ({result.Frames} frames, {result.Duration:F2}s). Ready to preview!";
                Repaint();

                _activePreviewClip = null;
                _previewTime = 0f;
                _isPlayingPreview = true;
                _lastUpdateTime = EditorApplication.timeSinceStartup;

                SetupPreviewInstance(autoPlay: true);
                _isPlayingPreview = true;

                _videoExtractionStatus = $"Extracted {result.Frames} frames successfully!";

                if (TexMotionSettings.instance.AutoOpenTimelineEditor)
                {
                    OpenInTimelineEditor();
                }
            }
            catch (OperationCanceledException)
            {
                _statusMessage = "Video motion extraction cancelled.";
                _videoExtractionStatus = "Extraction cancelled.";
            }
            catch (Exception ex)
            {
                _statusMessage = $"Video extraction failed: {ex.Message}";
                _videoExtractionStatus = "Extraction failed.";
                Debug.LogError($"[TexMotion Video] {ex}");
                EditorUtility.DisplayDialog("Extraction Error", $"Video extraction failed:\n{ex.Message}", "OK");
            }
            finally
            {
                _isVideoExtracting = false;
                Repaint();
            }
        }
    }
}
