using System;
using System.Collections.Generic;
using System.IO;
using System.Threading;
using System.Threading.Tasks;
using TexMotion.Editor.Motion;
using TexMotion.Editor.VRChat;
using TexMotion.Runtime.Motion;
using TexMotion.Runtime.Native;
using TexMotion.Runtime.VRChat;
using UnityEditor;
using UnityEngine;

namespace TexMotion.Editor
{
    public class TexMotionWindow : EditorWindow
    {
        private enum Tab
        {
            Generator,
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

        // Motion Library State
        private List<MotionLibraryItem> _libraryItems = new List<MotionLibraryItem>();
        private MotionLibraryItem _selectedLibraryItem;
        private Vector2 _libraryScrollPos;

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
            EditorApplication.update += OnEditorUpdate;
        }

        private void OnDisable()
        {
            EditorApplication.update -= OnEditorUpdate;
            CleanupPreviewUtility();
            _cts?.Cancel();
            _engine?.Dispose();
            _engine = null;
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

        private void CleanupPreviewInstance()
        {
            _isPlayingPreview = false;
            _previewBoneMap.Clear();
            _previewInitialRotations.Clear();
            _previewHipsTransform = null;
            _previewFaceRenderer = null;
            _activePreviewClip = null;

            if (_previewInstance != null)
            {
                DestroyImmediate(_previewInstance);
                _previewInstance = null;
            }
        }

        private void SetupPreviewInstance()
        {
            if (_previewUtility == null) InitPreviewUtility();
            CleanupPreviewInstance();

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

            if (_handPose != HandPoseType.KeepFree)
            {
                ApplyFingersToPreviewInstance(origAnim, HandPosePresets.LeftFingerBones, _handPose, true);
                ApplyFingersToPreviewInstance(origAnim, HandPosePresets.RightFingerBones, _handPose, false);
            }

            FaceEmotionType activeEmotion = _faceEmotion;
            if (activeEmotion == FaceEmotionType.AutoDetect)
            {
                activeEmotion = FaceEmotionHelper.InferEmotionFromPrompt(_prompt);
            }

            if (_previewFaceRenderer != null && activeEmotion != FaceEmotionType.None)
            {
                var weights = FaceEmotionHelper.GetBlendShapeWeightsForEmotion(_previewFaceRenderer, activeEmotion, _emotionIntensity);
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

            if (_isPlayingPreview && _previewInstance != null)
            {
                float duration = GetActivePreviewDuration();
                if (duration > 0)
                {
                    _previewTime += (float)dt;
                    if (_previewTime > duration)
                    {
                        _previewTime = 0f;
                    }

                    ApplyCurrentPoseToPreview(_previewTime);
                    Repaint();
                }
            }
        }

        private float GetActivePreviewDuration()
        {
            if (_currentMotionData != null && _currentMotionData.Frames > 0)
            {
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
            float totalDuration = totalFrames / _currentMotionData.FrameRate;
            if (totalDuration <= 0) return;

            float normTime = (time % totalDuration) * _currentMotionData.FrameRate;
            int frameIdx = Mathf.Clamp((int)normTime, 0, totalFrames - 1);
            int nextFrameIdx = Mathf.Min(frameIdx + 1, totalFrames - 1);
            float t = normTime - frameIdx;

            if (_previewHipsTransform != null && _currentMotionData.RootPositions.Length > 0)
            {
                Vector3 p0 = _currentMotionData.RootPositions[frameIdx];
                Vector3 p1 = _currentMotionData.RootPositions[nextFrameIdx];
                Vector3 p = Vector3.Lerp(p0, p1, t);

                Vector3 firstP = _currentMotionData.RootPositions[0];
                Vector3 delta = p - firstP;

                if (_inPlace)
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
            GUI.Label(new Rect(headerRect.x + 15, headerRect.y + 30, headerRect.width - 30, 18), "AI-Powered Text-to-Motion for VRChat Avatars", subStyle);
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

            // 3. Setup Target Configuration (Modular Avatar vs Direct VRCSDK)
            DrawSetupTargetSection();

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
            if (_currentMotionData != null)
            {
                EditorGUILayout.Space(15);
                DrawInteractive3DPreviewSection();
            }
        }

        private void DrawSetupTargetSection()
        {
            EditorGUILayout.BeginVertical(EditorStyles.helpBox);
            EditorGUILayout.LabelField("3. VRChat Setup Target & Destination Menu", EditorStyles.boldLabel);

            _motionName = EditorGUILayout.TextField("Motion Name", _motionName);
            _setupMode = (VrcSetupMode)EditorGUILayout.EnumPopup("Setup Target", _setupMode);
            _motionType = (VrcMotionType)EditorGUILayout.EnumPopup("Motion Playback", _motionType);
            _targetLayer = (VrcTargetLayer)EditorGUILayout.EnumPopup("Target Layer", _targetLayer);
            _inPlace = EditorGUILayout.Toggle("In-Place (Stay on origin)", _inPlace);

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

            // Setup Target configuration in Library tab as well
            DrawSetupTargetSection();
            EditorGUILayout.Space(8);

            if (_libraryItems.Count == 0)
            {
                EditorGUILayout.HelpBox("No generated motions found in Assets/TexMotion/Generated. Generate a new motion in the 'Motion Generator' tab!", MessageType.Info);
                return;
            }

            // List of items
            for (int i = 0; i < _libraryItems.Count; i++)
            {
                var item = _libraryItems[i];
                bool isSelected = _selectedLibraryItem == item;

                EditorGUILayout.BeginVertical(isSelected ? EditorStyles.selectionRect : EditorStyles.helpBox);

                EditorGUILayout.BeginHorizontal();
                
                // Name & Info
                EditorGUILayout.BeginVertical();
                EditorGUILayout.LabelField($"🎬 {item.Name}", EditorStyles.boldLabel);
                EditorGUILayout.LabelField($"{item.Duration:F2}s ({item.FrameRate:F0}fps) | {(item.IsLoop ? "Loop" : "One-Shot")}", EditorStyles.miniLabel);
                EditorGUILayout.EndVertical();

                // Status Badge
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

                // Action: Preview Button
                GUI.backgroundColor = isSelected ? new Color(0.95f, 0.65f, 0.2f) : Color.white;
                if (GUILayout.Button(isSelected ? "👁 Viewing" : "👁 Preview", GUILayout.Width(75), GUILayout.Height(28)))
                {
                    _selectedLibraryItem = item;
                    _activePreviewClip = item.Clip;
                    _currentMotionData = null;
                    _previewTime = 0f;
                    _isPlayingPreview = true;
                    _lastUpdateTime = EditorApplication.timeSinceStartup;
                    SetupPreviewInstance();
                }
                GUI.backgroundColor = Color.white;

                // Action: Apply / Remove from avatar
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

                // Action: Delete Asset
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

            // Preview Section in Library
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

            // Timeline Scrub
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

            // Playback controls
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

            float duration = _currentMotionData.Frames / _currentMotionData.FrameRate;
            EditorGUILayout.LabelField("🎬 2. 3D Motion Preview (Drag to Rotate, Scroll to Zoom)", EditorStyles.boldLabel);
            EditorGUILayout.LabelField($"Frames: {_currentMotionData.Frames} | Length: {duration:F2}s | Hand: {_handPose} | Face: {_faceEmotion}", EditorStyles.miniLabel);

            EditorGUILayout.Space(5);

            Rect previewRect = GUILayoutUtility.GetRect(400, 260, GUILayout.ExpandWidth(true));
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

                ApplyMotionPoseToPreview(_previewTime);

                _previewUtility.Render(true);
                Texture previewTex = _previewUtility.EndPreview();
                GUI.DrawTexture(previewRect, previewTex, ScaleMode.StretchToFill, false);
            }
            else
            {
                EditorGUI.DrawRect(previewRect, new Color(0.1f, 0.1f, 0.12f));
                GUI.Label(previewRect, "Loading 3D Preview...", EditorStyles.centeredGreyMiniLabel);
            }

            EditorGUILayout.Space(8);

            EditorGUILayout.BeginHorizontal();
            EditorGUI.BeginChangeCheck();
            _previewTime = EditorGUILayout.Slider("Timeline", _previewTime, 0f, duration);
            if (EditorGUI.EndChangeCheck())
            {
                _isPlayingPreview = false;
                ApplyMotionPoseToPreview(_previewTime);
            }
            EditorGUILayout.LabelField($"{_previewTime:F2}s / {duration:F2}s", GUILayout.Width(80));
            EditorGUILayout.EndHorizontal();

            EditorGUILayout.BeginHorizontal();
            GUI.backgroundColor = _isPlayingPreview ? new Color(0.95f, 0.65f, 0.2f) : new Color(0.3f, 0.85f, 0.45f);
            if (GUILayout.Button(_isPlayingPreview ? "⏸ Pause" : "▶ Play", GUILayout.Height(30)))
            {
                _isPlayingPreview = !_isPlayingPreview;
                if (_isPlayingPreview)
                {
                    _lastUpdateTime = EditorApplication.timeSinceStartup;
                }
            }
            GUI.backgroundColor = Color.white;

            if (GUILayout.Button("⏹ Reset", GUILayout.Height(30), GUILayout.Width(75)))
            {
                _isPlayingPreview = false;
                _previewTime = 0f;
                ApplyMotionPoseToPreview(0f);
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
                CleanupPreviewInstance();
                _statusMessage = "Preview discarded.";
            }

            GUI.backgroundColor = Color.white;
            EditorGUILayout.EndHorizontal();

            EditorGUILayout.EndVertical();
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

                FaceEmotionType activeEmotion = _faceEmotion;
                if (activeEmotion == FaceEmotionType.AutoDetect)
                {
                    activeEmotion = FaceEmotionHelper.InferEmotionFromPrompt(_prompt);
                }

                var buildOptions = new AnimationBuildOptions
                {
                    ClipName = $"Anim_{_motionName}",
                    IsLoop = _motionType == VrcMotionType.ToggleLoopPose,
                    InPlace = _inPlace,
                    Speed = 1.0f,
                    TargetAvatar = animator,
                    HandPose = _handPose,
                    FaceEmotion = activeEmotion,
                    EmotionIntensity = _emotionIntensity
                };

                var clip = AnimationClipBuilder.BuildAnimationClip(_currentMotionData, buildOptions);

                string clipPath = $"{saveDir}/Anim_{_motionName}.anim";
                AssetDatabase.CreateAsset(clip, clipPath);
                AssetDatabase.SaveAssets();

                var vrcConfig = new VrcMotionConfig
                {
                    MotionName = _motionName,
                    SetupMode = _setupMode,
                    MotionType = _motionType,
                    TargetLayer = _targetLayer,
                    InPlace = _inPlace
                };

                if (_setupMode == VrcSetupMode.ModularAvatar)
                {
                    GameObject setupObj = ModularAvatarSetup.SetupAvatarMotion(_targetAvatar, clip, vrcConfig, saveDir);
                    _statusMessage = $"Successfully applied '{_motionName}' to {_targetAvatar.name} via Modular Avatar!";
                    EditorGUIUtility.PingObject(setupObj);
                    Selection.activeGameObject = setupObj;
                    EditorUtility.DisplayDialog("Setup Complete", $"Motion '{_motionName}' was successfully applied to {_targetAvatar.name} via Modular Avatar!", "Great!");
                }
                else
                {
                    VrcDirectSetup.SetupDirectAvatarMotion(_targetAvatar, clip, vrcConfig, _customTargetMenu, saveDir);
                    string menuName = _customTargetMenu != null ? _customTargetMenu.name : "VRCExpressionsMenu";
                    _statusMessage = $"Successfully added '{_motionName}' directly to {menuName} & Animator!";
                    EditorUtility.DisplayDialog("Setup Complete", $"Motion '{_motionName}' was successfully added directly into '{menuName}', ExpressionParameters, and {_targetLayer}!", "Great!");
                }
                
                _currentMotionData = null;
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

                FaceEmotionType activeEmotion = _faceEmotion;
                if (activeEmotion == FaceEmotionType.AutoDetect)
                {
                    activeEmotion = FaceEmotionHelper.InferEmotionFromPrompt(_prompt);
                }

                var buildOptions = new AnimationBuildOptions
                {
                    ClipName = $"Anim_{_motionName}",
                    IsLoop = _motionType == VrcMotionType.ToggleLoopPose,
                    InPlace = _inPlace,
                    Speed = 1.0f,
                    TargetAvatar = animator,
                    HandPose = _handPose,
                    FaceEmotion = activeEmotion,
                    EmotionIntensity = _emotionIntensity
                };

                var clip = AnimationClipBuilder.BuildAnimationClip(_currentMotionData, buildOptions);

                string clipPath = $"{saveDir}/Anim_{_motionName}.anim";
                AssetDatabase.CreateAsset(clip, clipPath);
                AssetDatabase.SaveAssets();

                _statusMessage = $"Saved AnimationClip to {clipPath}";
                EditorGUIUtility.PingObject(clip);
                Selection.activeObject = clip;

                EditorUtility.DisplayDialog("Saved", $"AnimationClip saved to {clipPath}", "OK");

                _currentMotionData = null;
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

            GUI.enabled = !_isDownloading && !_isGenerating;
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

            if (GUI.changed)
            {
                settings.Save();
            }
        }

        private void DrawFooterStatus()
        {
            if (_isGenerating || _isDownloading)
            {
                EditorGUILayout.Space(5);
                EditorGUI.ProgressBar(EditorGUILayout.GetControlRect(false, 18), _overallProgress, _statusMessage);
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

                _overallProgress = 0.8f;
                _statusMessage = "Setting up 3D preview viewport...";
                Repaint();

                _activePreviewClip = null;
                _previewTime = 0f;
                _isPlayingPreview = true;
                _lastUpdateTime = EditorApplication.timeSinceStartup;

                SetupPreviewInstance();

                _overallProgress = 1.0f;
                _statusMessage = $"Motion ready! Previewing on {_targetAvatar.name}.";
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
    }
}
