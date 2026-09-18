using System;
using System.Collections.Generic;
using System.IO;
using TexMotion.Editor.Video;
using UnityEngine;

#if UNITY_EDITOR
using UnityEditor;
#endif

namespace TexMotion.Editor
{
    /// <summary>
    /// Logical runner selection for the WHAM image-feature stage.  This is a
    /// string on the persisted settings object on purpose: older projects can
    /// be opened without an enum migration and the values mirror the runner
    /// names used by the offline contract.
    /// </summary>
    public static class TexMotionWhamRunnerKinds
    {
        public const string OfficialHmr2 = "official_hmr2";
        public const string CompatibleVitPose = "compatible_vitpose";
        public const string Archive = "archive";

        public static string Normalize(string value)
        {
            if (string.Equals(value, CompatibleVitPose, StringComparison.OrdinalIgnoreCase))
                return CompatibleVitPose;
            if (string.Equals(value, Archive, StringComparison.OrdinalIgnoreCase))
                return Archive;
            return OfficialHmr2;
        }
    }

    public enum TexMotionLanguage
    {
        English = 0,
        Japanese = 1
    }

    [FilePath("ProjectSettings/TexMotionSettings.asset", FilePathAttribute.Location.ProjectFolder)]
    public class TexMotionSettings : ScriptableSingleton<TexMotionSettings>
    {
        [Header("Language")]
        public TexMotionLanguage Language = TexMotionLanguage.English;

        [Header("Hugging Face Repository")]
        public string HuggingFaceRepo = "k0ta0uchi/Llama-3-Kimodo-SMPLX-RP-v1-GGUF";
        public string MotionModelFileName = "kimodo-smplx-rp-v1-f32.gguf";
        public string TextBundleDirName = "llm2vec-text-bundle";

        [Header("Model Storage Mode")]
        public bool UseCustomLocalPath = false;
        public string CustomLocalModelDirectory = "";

        [Header("Python Video Pose Extraction")]
        public string CustomPythonExecutablePath = "";

        [Tooltip("Dependency profile for the isolated Video Motion virtual environment.")]
        public PythonDependencyProfile VideoPythonEnvironmentProfile = PythonDependencyProfile.Lightweight;

        [Tooltip("Optional project-local venv path. Leave empty to use .texmotion-venv at the project root.")]
        public string VideoVenvPath = "";

        [Tooltip("uv executable or an absolute path to uv.exe.")]
        public string UvExecutablePath = "uv";

        [Tooltip("Use the project venv for Video Motion extraction after it is created.")]
        public bool UseVideoVenv = true;

        [Header("Video2Motion Model Storage")]
        [Tooltip("Directory for RTMPose, DWPose, MediaPipe task assets, and optional WHAM/HMR2/HybrIK checkpoints. Leave empty for %APPDATA%/TexMotion/Models/Video.")]
        public string VideoModelDirectory = "";

        [Tooltip("Optional RTMPose/DWPose model file. Leave empty to use rtmpose-m.onnx in the Video2Motion model directory.")]
        public string RTMPoseModelPath = "";

        [Header("WHAM Companion Assets")]
        [Tooltip("Optional explicit WHAM offline asset manifest. Leave empty to resolve wham_asset_manifest.json from the Video2Motion model directory.")]
        public string WHAMAssetManifestPath = "";

        [Tooltip("Optional SMPL/SMPL-X body model path. The native WHAM checkpoint may provide embedded neutral buffers.")]
        public string WHAMBodyModelPath = "";

        [Tooltip("WHAM image-feature runner: official_hmr2, compatible_vitpose, or archive. The official HMR2 runner is the default and does not require a user-authored Python module.")]
        public string WHAMRunnerKind = TexMotionWhamRunnerKinds.OfficialHmr2;

        [Tooltip("Official HMR2 image-feature runtime/checkpoint path. This is kept separate from the ViTPose 2D detector checkpoint.")]
        public string HMR2CheckpointPath = "";

        [Tooltip("Optional compatible ViTPose/MMPose runtime directory or package path. It is used only when WHAMRunnerKind is compatible_vitpose.")]
        public string ViTPoseRuntimePath = "";

        [Tooltip("Optional compatible ViTPose checkpoint. A raw checkpoint is not executable without its matching runtime/configuration.")]
        public string ViTPoseModelPath = "";

        [Tooltip("Optional compatible ViTPose model architecture configuration.")]
        public string ViTPoseConfigPath = "";

        [Tooltip("Optional compatible ViTPose model-definition/factory file. The normal official HMR2 path does not ask users to author this file.")]
        public string ViTPoseModelDefinitionPath = "";

        [Tooltip("Output key used by the image-feature runner when selecting a feature tensor.")]
        public string ImageFeatureOutputKey = "img_feat";

        [Tooltip("Expected image-feature dimension used by preflight. Zero lets the runner report its actual dimension.")]
        public int ImageFeatureDim = 0;

        [Tooltip("Allow an unavailable WHAM image-feature stage to fall back to MediaPipe while retaining an explicit diagnostic.")]
        public bool AllowMediaPipeFallback = true;

        [Tooltip("Require the WHAM image-feature stage. When enabled, runner/archive failure must fail extraction instead of falling back.")]
        public bool RequireImageFeatures = false;

        [Tooltip("ViTPose 2D detector checkpoint used by official WHAM preprocessing. HMR2 image features are configured by the official runner separately.")]
        public string WHAMImageFeatureBackbonePath = "";

        [Tooltip("Optional ViTPose model-definition module or .py file. A checkpoint alone cannot construct a ViTPose network.")]
        public string WHAMImageFeatureModelDefinitionPath = "";

        [Tooltip("Optional ViTPose runner/model config (.json/.yaml/.py) used with the model-definition hook.")]
        public string WHAMImageFeatureConfigPath = "";

        [Tooltip("Optional precomputed WHAM image features (.npy/.npz/.pt). Rows must be one feature vector per sampled frame.")]
        public string WHAMImageFeaturePath = "";

        [Tooltip("Optional camera calibration or exported camera-motion archive (.npy/.npz/.json/.yaml). Calibration alone does not provide world trajectory.")]
        public string WHAMCameraModelPath = "";

        [Tooltip("Optional DPVO checkpoint or exported camera-motion archive. A checkpoint alone needs a local DPVO runtime to produce poses.")]
        public string WHAMDpvoModelPath = "";

        [Tooltip("Optional directory for per-video preprocessed features and camera poses. Leave empty to use the Video2Motion model directory.")]
        public string WHAMPreprocessDirectory = "";

        [Header("Video2Motion Inference")]
        [Tooltip("Backend used by Video Motion extraction. The same choice is selectable from the Video Motion and Settings dropdowns.")]
        public VideoPoseBackend VideoBackend = VideoPoseBackend.Auto;

        [Tooltip("Visualization mode in exported _overlay.mp4. Dual: 2D detector guide + 3D pose projection (Recommended).")]
        public VideoOverlayMode VideoOverlayMode = VideoOverlayMode.Dual;

        [Tooltip("MediaPipe Pose Landmarker complexity: 0 (fast), 1 (balanced), or 2 (heavy).")]
        public int VideoModelComplexity = 2;

        [Tooltip("Optional WHAM checkpoint. Select WHAM or WHAM + MediaPipe Fusion in Video2Motion Inference to use it.")]
        public string WHAMModelPath = "";

        [Tooltip("Optional HMR2 / 4D-Humans checkpoint. Select HMR2 in Video2Motion Inference to use it.")]
        public string HMR2ModelPath = "";

        [Tooltip("Optional HybrIK/HybrIK-X checkpoint. Select HybrIK in Video2Motion Inference to use it.")]
        public string HybrIKModelPath = "";

        [Tooltip("Optional generic TorchScript checkpoint used when the PyTorch Quality backend is selected.")]
        public string VideoQualityModelPath = "";

        [Tooltip("Optional Python adapter module or .py file implementing the temporal quality backend contract.")]
        public string VideoQualityAdapterPath = "";

        [Tooltip("Optional HMR2/4D-Humans adapter module or .py file. Kept separate from WHAM and HybrIK so preflight cannot accept the wrong backend bridge.")]
        public string HMR2AdapterPath = "";

        [Tooltip("Optional HybrIK/HybrIK-X adapter module or .py file. Kept separate from WHAM and HMR2 so preflight cannot accept the wrong backend bridge.")]
        public string HybrIKAdapterPath = "";

        [Tooltip("Optional licensed neutral SMPL body model for the official HMR2 runtime. Leave empty to resolve basicModel_neutral_lbs_10_207_0_v1.0.0.pkl from the Video2Motion model directory.")]
        public string HMR2BodyModelPath = "";

        [Tooltip("Optional local 4D-Humans/HMR2 source checkout or compatible runtime directory. Leave empty to resolve hmr2-runtime from the Video2Motion model directory.")]
        public string HMR2RuntimePath = "";

        [Tooltip("Quality backend device: auto, cpu, or cuda. CUDA is optional; CPU remains supported.")]
        public string VideoQualityDevice = "auto";

        [Tooltip("Maximum sampled frames passed to a temporal quality backend. Zero keeps the complete sequence.")]
        public int VideoMaxSequenceFrames = 0;

        [Header("Workflow & Timeline Editor")]
        public bool AutoOpenTimelineEditor = true;

        public string GetDefaultCacheDirectory()
        {
            string path = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData), "TexMotion", "Models");
            if (!Directory.Exists(path))
            {
                Directory.CreateDirectory(path);
            }
            return path;
        }

        public string GetEffectiveModelDirectory()
        {
            if (UseCustomLocalPath && !string.IsNullOrEmpty(CustomLocalModelDirectory) && Directory.Exists(CustomLocalModelDirectory))
            {
                return CustomLocalModelDirectory;
            }
            return GetDefaultCacheDirectory();
        }

        /// <summary>
        /// Returns the Unity project root. Application.dataPath is the most reliable
        /// source inside the Editor; the current directory is retained for package tests.
        /// </summary>
        public string GetProjectRootDirectory()
        {
            try
            {
                if (!string.IsNullOrEmpty(Application.dataPath))
                {
                    var parent = Directory.GetParent(Application.dataPath);
                    if (parent != null && Directory.Exists(parent.FullName))
                    {
                        return parent.FullName;
                    }
                }
            }
            catch { }

            return Directory.GetCurrentDirectory();
        }

        /// <summary>
        /// Default isolated environment used by Video Motion. It is deliberately
        /// project-local so different Unity projects cannot mutate one another.
        /// </summary>
        public string GetDefaultVideoVenvPath()
        {
            return Path.Combine(GetProjectRootDirectory(), ".texmotion-venv");
        }

        public string GetEffectiveVideoVenvPath()
        {
            if (string.IsNullOrWhiteSpace(VideoVenvPath)) return GetDefaultVideoVenvPath();
            try
            {
                return Path.GetFullPath(VideoVenvPath.Trim());
            }
            catch
            {
                return GetDefaultVideoVenvPath();
            }
        }

        public string GetVideoVenvPythonPath()
        {
            string venv = GetEffectiveVideoVenvPath();
            return Path.Combine(venv, Application.platform == RuntimePlatform.WindowsEditor ? "Scripts" : "bin", Application.platform == RuntimePlatform.WindowsEditor ? "python.exe" : "python");
        }

        public bool IsVideoVenvCreated()
        {
            try
            {
                return File.Exists(GetVideoVenvPythonPath());
            }
            catch
            {
                return false;
            }
        }

        /// <summary>
        /// Returns the executable used for Video Motion probing and extraction.
        /// A configured project venv takes precedence only when the toggle is enabled
        /// and its interpreter exists; otherwise the explicit legacy path is retained.
        /// </summary>
        public string GetEffectiveVideoPythonExecutablePath()
        {
            if (UseVideoVenv && IsVideoVenvCreated()) return GetVideoVenvPythonPath();
            return CustomPythonExecutablePath;
        }

        /// <summary>
        /// Resolves the requirements file shipped next to the extractor.
        /// </summary>
        public string GetVideoRequirementsPath()
        {
            return GetVideoRequirementsPath(VideoPythonEnvironmentProfile);
        }

        /// <summary>
        /// Resolves a profile-specific requirements manifest without mutating
        /// the persisted Settings profile. This is used by catalog preflight
        /// when the selected backend differs from the currently open UI card.
        /// </summary>
        public string GetVideoRequirementsPath(PythonDependencyProfile profile)
        {
            string fileName = profile == PythonDependencyProfile.PyTorch
                ? "requirements-pytorch.txt"
                : "requirements.txt";

            string packagePath = Path.GetFullPath(Path.Combine("Packages/com.k0ta0uchi.texmotion/Editor/Video", fileName));
            if (File.Exists(packagePath)) return packagePath;

            string editorPath = Path.Combine(GetProjectRootDirectory(), "Editor", "Video", fileName);
            return editorPath;
        }

        /// <summary>
        /// Default cache directory for video-to-motion model assets. It is kept
        /// separate from the Kimodo GGUF/text-bundle cache so large ONNX/task
        /// files can be moved or backed up independently.
        /// </summary>
        public string GetDefaultVideoModelDirectory()
        {
            return Path.Combine(GetDefaultCacheDirectory(), "Video");
        }

        public string GetEffectiveVideoModelDirectory()
        {
            if (string.IsNullOrWhiteSpace(VideoModelDirectory)) return GetDefaultVideoModelDirectory();
            try
            {
                return Path.GetFullPath(VideoModelDirectory.Trim());
            }
            catch
            {
                return GetDefaultVideoModelDirectory();
            }
        }

        /// <summary>Resolves a named Video2Motion asset inside the configured directory.</summary>
        public string GetVideoModelPath(string fileName)
        {
            if (string.IsNullOrWhiteSpace(fileName)) return GetEffectiveVideoModelDirectory();
            return Path.Combine(GetEffectiveVideoModelDirectory(), Path.GetFileName(fileName));
        }

        /// <summary>
        /// Resolves the optional official 4D-Humans/HMR2 source checkout. An
        /// explicit local path always wins; otherwise the catalog cache keeps
        /// the existing offline/local-path behavior.
        /// </summary>
        public string GetHMR2RuntimePath()
        {
            if (!string.IsNullOrWhiteSpace(HMR2RuntimePath))
            {
                try { return Path.GetFullPath(HMR2RuntimePath.Trim()); }
                catch { return HMR2RuntimePath.Trim(); }
            }
            return GetVideoModelPath("hmr2-runtime");
        }

        /// <summary>Returns the configured/catalog HMR2 runtime only when it exists.</summary>
        public string GetHMR2RuntimePathIfPresent()
        {
            string path = GetHMR2RuntimePath();
            try { return Directory.Exists(path) ? path : null; }
            catch { return null; }
        }

        /// <summary>
        /// Returns the official HMR2 checkpoint while preserving the legacy
        /// HMR2ModelPath setting.  New projects use HMR2CheckpointPath; old
        /// projects continue to resolve the same cache and explicit file.
        /// </summary>
        public string GetHMR2CheckpointPath()
        {
            string configured = !string.IsNullOrWhiteSpace(HMR2CheckpointPath)
                ? HMR2CheckpointPath
                : HMR2ModelPath;
            if (!string.IsNullOrWhiteSpace(configured))
            {
                try { return Path.GetFullPath(configured.Trim()); }
                catch { return configured.Trim(); }
            }
            return GetVideoModelPath("hmr2a_model.tar.gz");
        }

        /// <summary>Returns the configured/catalog HMR2 checkpoint only when it exists.</summary>
        public string GetHMR2CheckpointPathIfPresent()
        {
            string path = GetHMR2CheckpointPath();
            try
            {
                if (File.Exists(path) && new FileInfo(path).Length > 1024) return path;
            }
            catch { }

            try
            {
                VideoModelDefinition definition = VideoModelCatalog.Find("hmr2");
                return definition == null ? null : VideoModelCatalog.GetInstalledPath(this, definition);
            }
            catch { return null; }
        }

        /// <summary>
        /// Normalizes the persisted runner value to the three values in the
        /// official WHAM/HMR2 settings contract.
        /// </summary>
        public string GetWHAMRunnerKind()
        {
            return TexMotionWhamRunnerKinds.Normalize(WHAMRunnerKind);
        }

        /// <summary>Pascal-case alias for callers that use the logical WHAM name.</summary>
        public string GetWhamRunnerKind()
        {
            return GetWHAMRunnerKind();
        }

        /// <summary>Resolves the separately licensed neutral SMPL body used by HMR2.</summary>
        public string GetHMR2BodyModelPath()
        {
            if (!string.IsNullOrWhiteSpace(HMR2BodyModelPath))
            {
                try { return Path.GetFullPath(HMR2BodyModelPath.Trim()); }
                catch { return HMR2BodyModelPath.Trim(); }
            }
            return GetVideoModelPath("basicModel_neutral_lbs_10_207_0_v1.0.0.pkl");
        }

        public string GetHMR2BodyModelPathIfPresent()
        {
            try
            {
                VideoModelDefinition definition = VideoModelCatalog.FindRuntimeAdapter("hmr2-smpl-body");
                string discovered = definition == null ? null : VideoModelCatalog.GetInstalledWhamAssetPath(this, definition);
                if (!string.IsNullOrWhiteSpace(discovered)) return discovered;
            }
            catch { }
            string path = GetHMR2BodyModelPath();
            try { return File.Exists(path) && new FileInfo(path).Length > 0 ? path : null; }
            catch { return null; }
        }

        /// <summary>
        /// Resolves the adapter path for a selected quality backend. WHAM and
        /// the generic TorchScript backend retain the legacy shared setting;
        /// HMR2 and HybrIK intentionally use dedicated fields because a valid
        /// Python file for one bridge is not evidence that another bridge is
        /// compatible.
        /// </summary>
        public string GetVideoQualityAdapterPath(VideoPoseBackend backend)
        {
            string configured;
            switch (backend)
            {
                case VideoPoseBackend.HMR2: configured = HMR2AdapterPath; break;
                case VideoPoseBackend.HybrIK: configured = HybrIKAdapterPath; break;
                default: configured = VideoQualityAdapterPath; break;
            }
            if (string.IsNullOrWhiteSpace(configured)) return string.Empty;
            string trimmed = configured.Trim();
            // Python module references (for example ``my_adapter`` or
            // ``package.adapters.hmr2``) must remain import names; only file
            // paths should be normalized to an absolute path.
            if (!Path.IsPathRooted(trimmed) &&
                trimmed.IndexOf(Path.DirectorySeparatorChar) < 0 &&
                trimmed.IndexOf(Path.AltDirectorySeparatorChar) < 0 &&
                !trimmed.EndsWith(".py", StringComparison.OrdinalIgnoreCase))
                return trimmed;
            try { return Path.GetFullPath(trimmed); }
            catch { return trimmed; }
        }

        public void SetVideoQualityAdapterPath(VideoPoseBackend backend, string path)
        {
            string value = path ?? string.Empty;
            switch (backend)
            {
                case VideoPoseBackend.HMR2: HMR2AdapterPath = value; break;
                case VideoPoseBackend.HybrIK: HybrIKAdapterPath = value; break;
                default: VideoQualityAdapterPath = value; break;
            }
        }

        public string GetEffectiveRTMPoseModelPath()
        {
            if (!string.IsNullOrWhiteSpace(RTMPoseModelPath))
            {
                try { return Path.GetFullPath(RTMPoseModelPath.Trim()); }
                catch { }
            }
            return GetVideoModelPath("rtmpose-m.onnx");
        }

        public string GetRTMPoseModelPathIfPresent()
        {
            string path = GetEffectiveRTMPoseModelPath();
            try
            {
                return File.Exists(path) && new FileInfo(path).Length > 1024 ? path : null;
            }
            catch
            {
                return null;
            }
        }

        /// <summary>Returns an explicit WHAM companion path when it exists.</summary>
        public string GetWhamCompanionPathIfPresent(string configuredPath, string defaultFileName)
        {
            string path = configuredPath;
            if (string.IsNullOrWhiteSpace(path)) path = GetVideoModelPath(defaultFileName);
            try
            {
                string fullPath = Path.GetFullPath(path.Trim());
                return File.Exists(fullPath) && new FileInfo(fullPath).Length > 0 ? fullPath : null;
            }
            catch
            {
                return null;
            }
        }

        public string GetWHAMAssetManifestPathIfPresent()
        {
            return ResolveCatalogWhamAssetPathIfPresent("wham-asset-manifest", WHAMAssetManifestPath, "wham_asset_manifest.json");
        }

        public string GetWHAMBodyModelPathIfPresent()
        {
            return ResolveCatalogWhamAssetPathIfPresent("wham-smplx-body", WHAMBodyModelPath, "SMPLX_NEUTRAL.npz");
        }

        public string GetWHAMImageFeatureBackbonePathIfPresent()
        {
            return ResolveCatalogWhamAssetPathIfPresent(
                "wham-image-feature-backbone",
                string.IsNullOrWhiteSpace(ViTPoseModelPath) ? WHAMImageFeatureBackbonePath : ViTPoseModelPath,
                "vitpose-huge.pth");
        }

        public string GetWHAMImageFeatureModelDefinitionPathIfPresent()
        {
            return ResolveCatalogWhamAssetPathIfPresent(
                "wham-vitpose-model-definition",
                string.IsNullOrWhiteSpace(ViTPoseModelDefinitionPath) ? WHAMImageFeatureModelDefinitionPath : ViTPoseModelDefinitionPath,
                "wham_vitpose_model_definition.py");
        }

        public string GetWHAMImageFeatureConfigPathIfPresent()
        {
            return ResolveCatalogWhamAssetPathIfPresent(
                "wham-vitpose-runner-config",
                string.IsNullOrWhiteSpace(ViTPoseConfigPath) ? WHAMImageFeatureConfigPath : ViTPoseConfigPath,
                "wham_vitpose_runner_config.json");
        }

        public string GetWHAMPreprocessDirectoryIfPresent()
        {
            if (string.IsNullOrWhiteSpace(WHAMPreprocessDirectory)) return null;
            try
            {
                string path = Path.GetFullPath(WHAMPreprocessDirectory.Trim());
                if (Directory.Exists(path)) return path;
            }
            catch { }
            return null;
        }

        /// <summary>
        /// Resolves the explicitly selected compatible ViTPose runtime.  It is
        /// intentionally not aliased to HMR2RuntimePath: the UI must be able
        /// to report a raw ViTPose checkpoint and an executable HMR2 runtime
        /// as different states.
        /// </summary>
        public string GetViTPoseRuntimePath()
        {
            if (string.IsNullOrWhiteSpace(ViTPoseRuntimePath)) return string.Empty;
            try { return Path.GetFullPath(ViTPoseRuntimePath.Trim()); }
            catch { return ViTPoseRuntimePath.Trim(); }
        }

        public string GetViTPoseRuntimePathIfPresent()
        {
            string path = GetViTPoseRuntimePath();
            try
            {
                if (Directory.Exists(path) || (File.Exists(path) && new FileInfo(path).Length > 0))
                    return path;
            }
            catch { }
            return null;
        }

        public string GetWHAMImageFeaturePathIfPresent()
        {
            return ResolveCatalogWhamAssetPathIfPresent("wham-image-feature-archive", WHAMImageFeaturePath, "vitpose_features.npz");
        }

        public string GetWHAMCameraModelPathIfPresent()
        {
            return ResolveCatalogWhamAssetPathIfPresent("wham-camera", WHAMCameraModelPath, "camera.yaml");
        }

        public string GetWHAMDpvoModelPathIfPresent()
        {
            return ResolveCatalogWhamAssetPathIfPresent("wham-dpvo", WHAMDpvoModelPath, "dpvo.pth");
        }

        private string ResolveCatalogWhamAssetPathIfPresent(string catalogId, string configuredPath, string defaultFileName)
        {
            try
            {
                // A generated or manually selected archive is an explicit
                // user choice and must take precedence over an older file in
                // the catalog cache. This is essential for the one-click
                // ViTPose exporter, whose rows are tied to a particular video
                // FPS/trim range.
                if (!string.IsNullOrWhiteSpace(configuredPath))
                {
                    string explicitPath = Path.GetFullPath(configuredPath.Trim());
                    if (File.Exists(explicitPath) && new FileInfo(explicitPath).Length > 0)
                        return explicitPath;
                }
                VideoModelDefinition definition = VideoModelCatalog.FindWhamAsset(catalogId);
                string discovered = definition == null ? null : VideoModelCatalog.GetInstalledWhamAssetPath(this, definition);
                if (!string.IsNullOrWhiteSpace(discovered)) return discovered;
            }
            catch { }
            return GetWhamCompanionPathIfPresent(configuredPath, defaultFileName);
        }

        /// <summary>
        /// Returns the checkpoint path associated with the selected temporal quality
        /// backend. The paths are intentionally persisted in Settings while the
        /// active backend itself can be changed from either UI tab.
        /// </summary>
        public string GetQualityModelPath(VideoPoseBackend backend)
        {
            switch (backend)
            {
                case VideoPoseBackend.WHAM:
                case VideoPoseBackend.WHAMMediaPipe:
                    return string.IsNullOrWhiteSpace(WHAMModelPath) ? GetVideoModelPath("wham_vit_w_3dpw.pth.tar") : WHAMModelPath;
                case VideoPoseBackend.HMR2:
                    return GetHMR2CheckpointPath();
                case VideoPoseBackend.HybrIK:
                    return string.IsNullOrWhiteSpace(HybrIKModelPath) ? GetVideoModelPath("hybrikx_rle_hrnet.pth") : HybrIKModelPath;
                case VideoPoseBackend.PyTorch: return VideoQualityModelPath;
                default: return string.Empty;
            }
        }

        /// <summary>Returns a usable quality checkpoint path, or null when none is configured.</summary>
        public string GetQualityModelPathIfPresent(VideoPoseBackend backend)
        {
            string path = GetQualityModelPath(backend);
            VideoModelDefinition definition = null;
            switch (backend)
            {
                case VideoPoseBackend.WHAM:
                case VideoPoseBackend.WHAMMediaPipe:
                    definition = VideoModelCatalog.Find("wham");
                    break;
                case VideoPoseBackend.HMR2:
                    definition = VideoModelCatalog.Find("hmr2");
                    break;
                case VideoPoseBackend.HybrIK:
                    definition = VideoModelCatalog.Find("hybrik");
                    break;
            }
            if (!string.IsNullOrWhiteSpace(path))
            {
                try
                {
                    string fullPath = Path.GetFullPath(path.Trim());
                    // Keep explicitly selected checkpoints usable even when
                    // they are outside the catalog cache (and therefore have
                    // no catalog minimum-size metadata).
                    if (File.Exists(fullPath) && new FileInfo(fullPath).Length > 1024)
                    {
                        if (definition == null) return fullPath;
                        VideoAssetStatusInfo status = VideoModelCatalog.GetAssetStatus(this, definition, true);
                        if (status != null && status.IsReady) return fullPath;
                    }
                }
                catch { }
            }

            // A catalog row can be shown as installed in a legacy cache while
            // Settings still contains an old/empty path. Resolve that same
            // catalog definition here so extraction receives the checkpoint
            // that the UI already reported as ready. This also handles a user
            // changing VideoModelDirectory after downloading a model.
            if (definition == null) return null;
            try { return VideoModelCatalog.GetInstalledPath(this, definition); }
            catch { return null; }
        }

        /// <summary>Updates the persisted checkpoint path for a quality backend.</summary>
        public void SetQualityModelPath(VideoPoseBackend backend, string path)
        {
            switch (backend)
            {
                case VideoPoseBackend.WHAM:
                case VideoPoseBackend.WHAMMediaPipe:
                    WHAMModelPath = path ?? string.Empty;
                    break;
                case VideoPoseBackend.HMR2:
                    HMR2CheckpointPath = path ?? string.Empty;
                    // Keep the old serialized key synchronized for projects
                    // and integrations that still read HMR2ModelPath.
                    HMR2ModelPath = HMR2CheckpointPath;
                    break;
                case VideoPoseBackend.HybrIK: HybrIKModelPath = path ?? string.Empty; break;
                case VideoPoseBackend.PyTorch: VideoQualityModelPath = path ?? string.Empty; break;
            }
        }

        /// <summary>
        /// Smartly finds the motion GGUF model path from effective directory or its subdirectories/parent directories.
        /// </summary>
        public string GetMotionModelPath()
        {
            string baseDir = GetEffectiveModelDirectory();

            // Candidate 1: Direct in base directory
            string direct = Path.Combine(baseDir, MotionModelFileName);
            if (File.Exists(direct)) return direct;

            // Candidate 2: in models/ subfolder
            string inModels = Path.Combine(baseDir, "models", MotionModelFileName);
            if (File.Exists(inModels)) return inModels;

            // Candidate 3: in parent models/ or parent directory
            try
            {
                var parent = Directory.GetParent(baseDir);
                if (parent != null)
                {
                    string inParent = Path.Combine(parent.FullName, MotionModelFileName);
                    if (File.Exists(inParent)) return inParent;

                    string inParentModels = Path.Combine(parent.FullName, "models", MotionModelFileName);
                    if (File.Exists(inParentModels)) return inParentModels;
                }
            }
            catch {}

            // Candidate 4: Any .gguf file containing "kimodo" or "smplx"
            try
            {
                var files = Directory.GetFiles(baseDir, "*.gguf", SearchOption.AllDirectories);
                foreach (var f in files)
                {
                    string lower = Path.GetFileName(f).ToLowerInvariant();
                    if (lower.Contains("kimodo") || lower.Contains("smplx") || lower.Contains("rp-v1"))
                    {
                        return f;
                    }
                }
            }
            catch {}

            return direct;
        }

        /// <summary>
        /// Smartly finds the text bundle directory containing tokenizer.gguf, embedding.gguf, etc.
        /// </summary>
        public string GetTextBundleDirectory()
        {
            string baseDir = GetEffectiveModelDirectory();

            // Candidate 1: base/llm2vec-text-bundle
            string dir1 = Path.Combine(baseDir, TextBundleDirName);
            if (IsValidTextBundle(dir1)) return dir1;

            // Candidate 2: base/generated/llm2vec-text-bundle
            string dir2 = Path.Combine(baseDir, "generated", TextBundleDirName);
            if (IsValidTextBundle(dir2)) return dir2;

            // Candidate 3: base directory directly contains tokenizer.gguf
            if (IsValidTextBundle(baseDir)) return baseDir;

            // Candidate 4: Check parent directory (e.g. if base is J:/kimodo.cpp/models -> check J:/kimodo.cpp/generated/llm2vec-text-bundle)
            try
            {
                var parent = Directory.GetParent(baseDir);
                if (parent != null)
                {
                    string parentGen = Path.Combine(parent.FullName, "generated", TextBundleDirName);
                    if (IsValidTextBundle(parentGen)) return parentGen;

                    string parentBundle = Path.Combine(parent.FullName, TextBundleDirName);
                    if (IsValidTextBundle(parentBundle)) return parentBundle;
                }
            }
            catch {}

            // Candidate 5: Recursive search for folder with tokenizer.gguf
            try
            {
                var tokenizers = Directory.GetFiles(baseDir, "tokenizer.gguf", SearchOption.AllDirectories);
                if (tokenizers.Length > 0)
                {
                    string candidate = Path.GetDirectoryName(tokenizers[0]);
                    if (IsValidTextBundle(candidate)) return candidate;
                }
            }
            catch {}

            return dir1;
        }

        public bool IsValidTextBundle(string directoryPath)
        {
            if (string.IsNullOrEmpty(directoryPath) || !Directory.Exists(directoryPath)) return false;
            return File.Exists(Path.Combine(directoryPath, "tokenizer.gguf")) &&
                   File.Exists(Path.Combine(directoryPath, "embedding.gguf"));
        }

        public List<string> GetRequiredBundleFiles()
        {
            var list = new List<string>
            {
                "tokenizer.gguf",
                "embedding.gguf",
                "final-norm.gguf"
            };

            for (int i = 0; i < 32; i++)
            {
                list.Add($"layer-{i:D2}.gguf");
            }

            return list;
        }

        public bool AreModelsPresent()
        {
            string motionPath = GetMotionModelPath();
            if (!File.Exists(motionPath)) return false;

            string bundleDir = GetTextBundleDirectory();
            return IsValidTextBundle(bundleDir);
        }

        public void Save()
        {
            Save(true);
        }
    }
}
