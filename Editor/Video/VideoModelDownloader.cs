using System;
using System.Collections.Generic;
using System.IO;
using System.Security.Cryptography;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using TexMotion.Editor.Video;
using UnityEngine;
using UnityEngine.Networking;

namespace TexMotion.Editor
{
    /// <summary>Model family used by the Video2Motion runtime.</summary>
    public enum VideoModelKind
    {
        RTMPose,
        DWPose,
        MediaPipe,
        WHAM,
        HMR2,
        HybrIK
    }

    /// <summary>
    /// Stable role names used by the WHAM offline asset contract.  These are
    /// metadata only; the existing model catalog/download workflow remains
    /// unchanged for MediaPipe and RTMPose.
    /// </summary>
    public enum VideoAssetRole
    {
        InferenceModel,
        WhamCheckpoint,
        /// <summary>HMR2 checkpoint used by the official image-feature runner.</summary>
        Hmr2Checkpoint,
        /// <summary>
        /// Explicit semantic alias for the HMR2 checkpoint/runtime seam;
        /// Hmr2Checkpoint remains the serialized contract name.
        /// </summary>
        Hmr2ImageFeatureRunner = Hmr2Checkpoint,
        WhamBodyModel,
        /// <summary>Separately licensed neutral SMPL body asset used by HMR2.</summary>
        Hmr2BodyModel,
        /// <summary>
        /// ViTPose 2D keypoint detector used by official WHAM preprocessing;
        /// this is intentionally distinct from HMR2 image features.
        /// </summary>
        WhamImageFeatureBackbone,
        /// <summary>Explicit semantic alias for the ViTPose 2D detector role.</summary>
        WhamViTPose2DDetector = WhamImageFeatureBackbone,
        WhamImageFeatureArchive,
        /// <summary>
        /// Optional precomputed HMR2 image-feature archive; this is not a
        /// ViTPose 2D keypoint archive.
        /// </summary>
        Hmr2ImageFeatureArchive = WhamImageFeatureArchive,
        WhamCamera,
        WhamDpvo,
        WhamAdapter,
        WhamManifest,
        /// <summary>Optional local model-definition hook for the ViTPose runner.</summary>
        WhamImageFeatureModelDefinition,
        /// <summary>Optional local runner/model configuration for ViTPose.</summary>
        WhamImageFeatureConfig,
        /// <summary>Official HMR2/4D-Humans adapter or project bridge.</summary>
        Hmr2Adapter,
        /// <summary>Official HMR2 source/runtime checkout or compatible bridge.</summary>
        Hmr2Runtime,
        /// <summary>Explicit compatible ViTPose/MMPose runtime selected by the user.</summary>
        ViTPoseRuntime,
        /// <summary>Official HybrIK adapter or project bridge.</summary>
        HybrIKAdapter,
        /// <summary>Requirements manifest used to provision the isolated Python environment.</summary>
        PythonRequirements,
        /// <summary>Interpreter created by the explicit Python environment install action.</summary>
        PythonRuntime
    }

    /// <summary>
    /// Actionable setup metadata for assets which cannot be provisioned safely
    /// by the package (or which still need an upstream runtime/adapter after a
    /// public checkpoint is downloaded).  Keeping this metadata beside the
    /// catalog definition prevents Settings from presenting a bare path or a
    /// Source URL without explaining the next required step.
    /// </summary>
    [Serializable]
    public sealed class VideoAssetGuide
    {
        public string OfficialUrl;
        public string RecommendedInstallDirectory;
        public string RequiredFileName;
        public string BrowseSetting;
        public string PreflightSteps;
        public string LicenseReason;

        public VideoAssetGuide(
            string officialUrl,
            string recommendedInstallDirectory,
            string requiredFileName,
            string browseSetting,
            string preflightSteps,
            string licenseReason)
        {
            OfficialUrl = officialUrl ?? string.Empty;
            RecommendedInstallDirectory = recommendedInstallDirectory ?? string.Empty;
            RequiredFileName = requiredFileName ?? string.Empty;
            BrowseSetting = browseSetting ?? string.Empty;
            PreflightSteps = preflightSteps ?? string.Empty;
            LicenseReason = licenseReason ?? string.Empty;
        }
    }

    /// <summary>Deterministic local verification result for one runtime asset.</summary>
    public enum VideoAssetStatus
    {
        Ready,
        Missing,
        Incompatible,
        Manual,
        NotRequired
    }

    /// <summary>
    /// Human-readable, path-preserving result of catalog verification.  The
    /// Settings UI uses this object directly so a missing or malformed file is
    /// never collapsed into a generic "not ready" label.
    /// </summary>
    [Serializable]
    public sealed class VideoAssetStatusInfo
    {
        public string Id;
        public string DisplayName;
        public VideoAssetRole Role;
        public VideoAssetStatus Status;
        /// <summary>Stable FR-11 code when a row is missing or incompatible.</summary>
        public string ErrorCode;
        /// <summary>Actionable next step shown beside a failed preflight row.</summary>
        public string NextAction;
        public string ExpectedPath;
        public string ResolvedPath;
        public string Reason;
        public bool Required;
        public bool CanInstall;

        public bool IsReady => Status == VideoAssetStatus.Ready;
        public bool IsBlocking => Required &&
                                   Status != VideoAssetStatus.Ready &&
                                   Status != VideoAssetStatus.NotRequired;

        public override string ToString()
        {
            string path = string.IsNullOrWhiteSpace(ResolvedPath) ? ExpectedPath : ResolvedPath;
            string detail = Reason ?? string.Empty;
            if (!string.IsNullOrWhiteSpace(ErrorCode))
                detail += " [" + ErrorCode + "]";
            return string.Format("{0}: {1} ({2}) - {3}",
                DisplayName ?? Id ?? "Video asset",
                Status,
                path ?? "<path unavailable>",
                detail);
        }
    }

    /// <summary>Stable error identifiers shared by catalog diagnostics and the UI.</summary>
    public static class VideoFeatureErrorCodes
    {
        public const string RunnerRuntimeMissing = "runner_runtime_missing";
        public const string ModelFactoryMissing = "model_factory_missing";
        public const string CheckpointArchitectureMismatch = "checkpoint_architecture_mismatch";
        public const string CheckpointNotExecutable = "checkpoint_not_executable";
        public const string BodyModelMissing = "body_model_missing";
        public const string FeatureFrameCountMismatch = "feature_frame_count_mismatch";
        public const string FeatureDimMismatch = "feature_dim_mismatch";
        public const string FeatureNonFinite = "feature_non_finite";
        public const string OfficialRunnerInferenceFailed = "official_runner_inference_failed";
        public const string FallbackMediaPipe = "fallback_mediapipe";
    }

    /// <summary>Aggregate result returned by the explicit Video2Motion preflight action.</summary>
    [Serializable]
    public sealed class VideoPreflightReport
    {
        public VideoPoseBackend Backend;
        public IReadOnlyList<VideoAssetStatusInfo> Assets;

        public bool IsReady
        {
            get
            {
                if (Assets == null) return false;
                for (int i = 0; i < Assets.Count; i++)
                {
                    if (Assets[i] != null && Assets[i].IsBlocking) return false;
                }
                return true;
            }
        }

        public IReadOnlyList<VideoAssetStatusInfo> BlockingAssets
        {
            get
            {
                var result = new List<VideoAssetStatusInfo>();
                if (Assets == null) return result;
                for (int i = 0; i < Assets.Count; i++)
                {
                    if (Assets[i] != null && Assets[i].IsBlocking) result.Add(Assets[i]);
                }
                return result;
            }
        }

        /// <summary>
        /// Every non-ready row, including optional/full-parity stages. Settings
        /// uses this list so an optional fallback is still visible and can be
        /// acted on instead of disappearing from preflight.
        /// </summary>
        public IReadOnlyList<VideoAssetStatusInfo> NonReadyAssets
        {
            get
            {
                var result = new List<VideoAssetStatusInfo>();
                if (Assets == null) return result;
                for (int i = 0; i < Assets.Count; i++)
                {
                    if (Assets[i] != null && !Assets[i].IsReady && Assets[i].Status != VideoAssetStatus.NotRequired)
                        result.Add(Assets[i]);
                }
                return result;
            }
        }

        public string Describe()
        {
            var lines = new List<string>();
            IReadOnlyList<VideoAssetStatusInfo> nonReady = NonReadyAssets;
            for (int i = 0; i < nonReady.Count; i++)
            {
                VideoAssetStatusInfo status = nonReady[i];
                string detail = (status.IsBlocking ? "- Required: " : "- Optional/full parity: ") + status;
                if (!string.IsNullOrWhiteSpace(status.ErrorCode))
                    detail += " [code=" + status.ErrorCode + "]";
                if (!string.IsNullOrWhiteSpace(status.NextAction))
                    detail += "\n  Next: " + status.NextAction;
                lines.Add(detail);
            }
            if (lines.Count == 0) return "Video2Motion preflight ready.";
            return IsReady
                ? "Video2Motion preflight ready; optional/full-parity stages are not configured:\n" + string.Join("\n", lines)
                : "Video2Motion preflight incomplete:\n" + string.Join("\n", lines);
        }
    }

    /// <summary>
    /// Compact setup state consumed by the WHAM Settings card.  The detailed
    /// catalog rows remain available for advanced diagnostics, but the normal
    /// UI uses this summary so users can see one actionable state instead of a
    /// list of internal hook files.
    /// </summary>
    public enum VideoWhamSetupState
    {
        Ready,
        Missing,
        Manual,
        Incompatible,
        Checking
    }

    [Serializable]
    public sealed class VideoWhamSetupSummary
    {
        public VideoPoseBackend Backend;
        public VideoWhamSetupState State;
        public IReadOnlyList<VideoAssetStatusInfo> RequiredAssets;
        public IReadOnlyList<VideoAssetStatusInfo> OptionalAssets;
        public IReadOnlyList<VideoAssetStatusInfo> ManualAssets;
        public IReadOnlyList<VideoAssetStatusInfo> IncompatibleAssets;
        public string OfficialRunnerStatus;
        public string Hmr2ImageFeaturesStatus;
        public string VitPose2DStatus;

        public bool IsReady => State == VideoWhamSetupState.Ready;

        public int MissingRequiredCount
        {
            get
            {
                int count = 0;
                if (RequiredAssets == null) return count;
                for (int i = 0; i < RequiredAssets.Count; i++)
                {
                    VideoAssetStatusInfo asset = RequiredAssets[i];
                    if (asset != null && asset.Status != VideoAssetStatus.Ready &&
                        asset.Status != VideoAssetStatus.NotRequired) count++;
                }
                return count;
            }
        }

        /// <summary>Required rows that need user-supplied/licensed files.</summary>
        public int ManualRequiredCount
        {
            get { return CountRequiredStatus(VideoAssetStatus.Manual); }
        }

        /// <summary>Required rows whose local file cannot satisfy its role.</summary>
        public int IncompatibleRequiredCount
        {
            get { return CountRequiredStatus(VideoAssetStatus.Incompatible); }
        }

        public int OptionalUnavailableCount
        {
            get
            {
                int count = 0;
                if (OptionalAssets == null) return count;
                for (int i = 0; i < OptionalAssets.Count; i++)
                {
                    VideoAssetStatusInfo asset = OptionalAssets[i];
                    if (asset != null && !asset.IsReady && asset.Status != VideoAssetStatus.NotRequired) count++;
                }
                return count;
            }
        }

        private int CountRequiredStatus(VideoAssetStatus status)
        {
            int count = 0;
            if (RequiredAssets == null) return count;
            for (int i = 0; i < RequiredAssets.Count; i++)
            {
                VideoAssetStatusInfo asset = RequiredAssets[i];
                if (asset != null && asset.Status == status) count++;
            }
            return count;
        }
    }

    /// <summary>Bundled requirements metadata exposed by the Settings catalog.</summary>
    [Serializable]
    public sealed class VideoDependencyDefinition
    {
        public string Id;
        public string DisplayName;
        public string FileName;
        public PythonDependencyProfile Profile;
        public string Description;
        public string ReferenceUrl;
        public string[] RequiredPackages;

        public VideoDependencyDefinition(
            string id,
            string displayName,
            string fileName,
            PythonDependencyProfile profile,
            string description,
            string referenceUrl,
            string[] requiredPackages)
        {
            Id = id;
            DisplayName = displayName;
            FileName = fileName;
            Profile = profile;
            Description = description;
            ReferenceUrl = referenceUrl;
            RequiredPackages = requiredPackages ?? new string[0];
        }
    }

    /// <summary>
    /// Describes a model asset that can be cached by the Video2Motion settings UI.
    /// URLs point to public model artifacts; extraction itself only reads the local
    /// cache after the explicit download action has completed.
    /// </summary>
    [Serializable]
    public sealed class VideoModelDefinition
    {
        public string Id;
        public string DisplayName;
        public string FileName;
        public string RemoteUrl;
        public string Description;
        public long EstimatedBytes;
        public long MinimumBytes;
        public VideoModelKind Kind;
        public bool IsExperimental;
        /// <summary>Project or documentation page for provenance and manual setup.</summary>
        public string ReferenceUrl;
        /// <summary>Optional offline-contract role for this asset.</summary>
        public VideoAssetRole AssetRole;
        /// <summary>Whether this asset is required by the full WHAM pipeline.</summary>
        public bool RequiredForFullWhamParity;
        /// <summary>True when the asset must be supplied under its upstream terms.</summary>
        public bool IsManualProvisioning;
        /// <summary>Alternate filenames accepted by the deterministic Python resolver.</summary>
        public string[] Aliases;
        /// <summary>Optional SHA-256 digest for deterministic integrity verification.</summary>
        public string ExpectedSha256;
        /// <summary>Optional extension allow-list used to reject an incompatible local file.</summary>
        public string[] SupportedExtensions;
        /// <summary>True when the asset is a source/runtime directory instead of a file.</summary>
        public bool IsDirectoryAsset;
        /// <summary>Setup guidance shown in the Settings catalog and preflight.</summary>
        public VideoAssetGuide Guide;

        public VideoModelDefinition(
            string id,
            string displayName,
            string fileName,
            string remoteUrl,
            string description,
            long estimatedBytes,
            VideoModelKind kind,
            bool isExperimental = false,
            long minimumBytes = 1024 * 1024,
            string referenceUrl = null,
            VideoAssetRole assetRole = VideoAssetRole.InferenceModel,
            bool requiredForFullWhamParity = false,
            bool isManualProvisioning = false,
            string[] aliases = null,
            string expectedSha256 = null,
            string[] supportedExtensions = null,
            bool isDirectoryAsset = false,
            VideoAssetGuide guide = null)
        {
            Id = id;
            DisplayName = displayName;
            FileName = fileName;
            RemoteUrl = remoteUrl;
            Description = description;
            EstimatedBytes = estimatedBytes;
            MinimumBytes = minimumBytes;
            Kind = kind;
            IsExperimental = isExperimental;
            ReferenceUrl = string.IsNullOrWhiteSpace(referenceUrl) ? remoteUrl : referenceUrl;
            AssetRole = assetRole;
            RequiredForFullWhamParity = requiredForFullWhamParity;
            IsManualProvisioning = isManualProvisioning;
            Aliases = aliases ?? new string[0];
            ExpectedSha256 = expectedSha256;
            SupportedExtensions = supportedExtensions ?? new string[0];
            IsDirectoryAsset = isDirectoryAsset;
            Guide = guide;
        }

        /// <summary>Whether selecting this row can configure an extraction backend.</summary>
        public bool CanSelectForRuntime => Kind == VideoModelKind.RTMPose || Kind == VideoModelKind.DWPose ||
                                           Kind == VideoModelKind.MediaPipe ||
                                           (Kind == VideoModelKind.WHAM &&
                                            (AssetRole == VideoAssetRole.InferenceModel ||
                                             AssetRole == VideoAssetRole.WhamCheckpoint)) ||
                                            Kind == VideoModelKind.HMR2 || Kind == VideoModelKind.HybrIK;

        /// <summary>
        /// True for the HMR2 checkpoint/runtime seam that supplies official
        /// WHAM image features and the initial SMPL estimate. This must not be
        /// confused with the separate ViTPose 2D detector row.
        /// </summary>
        public bool IsHmr2ImageFeatureRunnerAsset =>
            AssetRole == VideoAssetRole.Hmr2ImageFeatureRunner ||
            AssetRole == VideoAssetRole.Hmr2Runtime ||
            AssetRole == VideoAssetRole.ViTPoseRuntime ||
            AssetRole == VideoAssetRole.Hmr2Adapter;

        /// <summary>
        /// True for the separate ViTPose 2D detector used by official WHAM
        /// preprocessing. Its checkpoint is not an HMR2 feature archive.
        /// </summary>
        public bool IsWhamViTPose2DDetector =>
            AssetRole == VideoAssetRole.WhamViTPose2DDetector;

        /// <summary>
        /// True for an optional, precomputed HMR2 feature archive. It is not
        /// the ViTPose detector checkpoint and cannot satisfy that 2D role.
        /// </summary>
        public bool IsHmr2ImageFeatureArchive =>
            AssetRole == VideoAssetRole.Hmr2ImageFeatureArchive;

        /// <summary>Whether this is an auxiliary WHAM full-parity asset.</summary>
        public bool IsWhamCompanionAsset => Kind == VideoModelKind.WHAM &&
                                            AssetRole != VideoAssetRole.InferenceModel &&
                                            AssetRole != VideoAssetRole.WhamCheckpoint;

        /// <summary>
        /// True when Settings can materialize this asset without a separately
        /// licensed user file.  Camera calibration is handled as a bundled
        /// template by the downloader even though it has no remote URL.
        /// </summary>
        public bool CanDownload => !IsManualProvisioning && IsAutomaticallyDownloadable;

        /// <summary>
        /// Whether an explicit Settings download action may materialize this
        /// row. A URL can be a documentation/registration page, so manual
        /// provisioning always wins over URL presence. Bundled contract files
        /// are eligible even though their URLs are provenance links only.
        /// </summary>
        public bool IsAutomaticallyDownloadable
        {
            get
            {
                if (IsManualProvisioning) return false;
                if (AssetRole == VideoAssetRole.WhamCamera ||
                    AssetRole == VideoAssetRole.WhamAdapter ||
                    AssetRole == VideoAssetRole.WhamManifest ||
                    AssetRole == VideoAssetRole.WhamImageFeatureModelDefinition ||
                    AssetRole == VideoAssetRole.WhamImageFeatureConfig)
                    return true;
                return !string.IsNullOrWhiteSpace(RemoteUrl);
            }
        }

        /// <summary>Whether the user should read upstream setup guidance for this row.</summary>
        public bool HasSetupGuide => Guide != null || IsManualProvisioning;
    }

    /// <summary>
    /// Built-in catalog for the local, offline Video2Motion assets. The catalog
    /// intentionally includes model variants so users can choose a smaller CPU
    /// model or a larger whole-body model without editing Python files.
    /// </summary>
    public static class VideoModelCatalog
    {
        /// <summary>Bundled TexMotion adapter installed alongside the WHAM checkpoint.</summary>
        public const string WhamAdapterFileName = "texmotion_wham_adapter.py";
        public const long WhamAdapterEstimatedBytes = 8L * 1024L;
        /// <summary>Bundled manifest describing all full-pipeline WHAM stages.</summary>
        public const string WhamAssetManifestFileName = "wham_asset_manifest.json";
        /// <summary>Bundled starter file for camera intrinsics.</summary>
        public const string WhamCameraTemplateFileName = "camera_template.yaml";
        /// <summary>Bundled configuration template for the optional ViTPose runner hook.</summary>
        public const string WhamImageFeatureConfigFileName = "wham_vitpose_runner_config.json";
        /// <summary>Bundled model-definition contract template for the optional ViTPose runner hook.</summary>
        public const string WhamImageFeatureModelDefinitionFileName = "wham_vitpose_model_definition.py";

        private static readonly VideoModelDefinition[] Definitions =
        {
            new VideoModelDefinition(
                "rtmpose-m",
                "RTMPose-M (COCO 2D)",
                "rtmpose-m.onnx",
                "https://huggingface.co/bukuroo/RTMPose-ONNX/resolve/main/rtmpose-m.onnx",
                "Recommended CPU/DirectML 2D keypoint model for the hybrid pipeline.",
                54L * 1024L * 1024L,
                VideoModelKind.RTMPose),
            new VideoModelDefinition(
                "rtmpose-s",
                "RTMPose-S (COCO 2D)",
                "rtmpose-s.onnx",
                "https://huggingface.co/bukuroo/RTMPose-ONNX/resolve/main/rtmpose-s.onnx",
                "Smaller and faster RTMPose variant for low-power machines.",
                22L * 1024L * 1024L,
                VideoModelKind.RTMPose),
            new VideoModelDefinition(
                "rtmpose-l",
                "RTMPose-L (COCO 2D)",
                "rtmpose-l.onnx",
                "https://huggingface.co/bukuroo/RTMPose-ONNX/resolve/main/rtmpose-l.onnx",
                "Higher-capacity 2D detector for difficult or small subjects.",
                111L * 1024L * 1024L,
                VideoModelKind.RTMPose),
            new VideoModelDefinition(
                "rtmpose-x",
                "RTMPose-X (COCO 2D)",
                "rtmpose-x.onnx",
                "https://huggingface.co/bukuroo/RTMPose-ONNX/resolve/main/rtmpose-x.onnx",
                "Largest COCO 2D variant; highest memory and latency cost.",
                198L * 1024L * 1024L,
                VideoModelKind.RTMPose),
            new VideoModelDefinition(
                "rtmpose-m-wholebody",
                "RTMPose-M WholeBody",
                "rtmpose-m-wholebody.onnx",
                "https://huggingface.co/bukuroo/RTMPose-ONNX/resolve/main/rtmpose-m-wholebody.onnx",
                "WholeBody graph for body, hands, and face auxiliary observations.",
                72L * 1024L * 1024L,
                VideoModelKind.RTMPose,
                true),
            new VideoModelDefinition(
                "rtmpose-m-hand",
                "RTMPose-M Hand",
                "rtmpose-m-hand.onnx",
                "https://huggingface.co/bukuroo/RTMPose-ONNX/resolve/main/rtmpose-m-hand.onnx",
                "Hand-focused graph; download for custom hand-pose adapters.",
                55L * 1024L * 1024L,
                VideoModelKind.RTMPose,
                true),
            new VideoModelDefinition(
                "dwpose-l",
                "DWPose-L (WholeBody ONNX)",
                "dwpose-l.onnx",
                "https://huggingface.co/SceneWorks/dwpose-onnx/resolve/main/rtmw-dw-x-l_simcc-cocktail14_270e-384x288_20231122.onnx",
                "Optional DWPose/RTMW graph; use with a compatible ONNX adapter.",
                229L * 1024L * 1024L,
                VideoModelKind.DWPose,
                true),
            new VideoModelDefinition(
                "mediapipe-lite",
                "MediaPipe Pose Landmarker Lite",
                "pose_landmarker_lite.task",
                "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
                "Fastest 33-joint MediaPipe task model (complexity 0).",
                6L * 1024L * 1024L,
                VideoModelKind.MediaPipe),
            new VideoModelDefinition(
                "mediapipe-full",
                "MediaPipe Pose Landmarker Full",
                "pose_landmarker_full.task",
                "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task",
                "Balanced 33-joint MediaPipe task model (complexity 1).",
                10L * 1024L * 1024L,
                VideoModelKind.MediaPipe),
            new VideoModelDefinition(
                "mediapipe-heavy",
                "MediaPipe Pose Landmarker Heavy",
                "pose_landmarker_heavy.task",
                "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task",
                "Highest-quality 33-joint MediaPipe task model (complexity 2).",
                30L * 1024L * 1024L,
                VideoModelKind.MediaPipe),
            new VideoModelDefinition(
                "wham",
                "WHAM (World-grounded 3D)",
                "wham_vit_w_3dpw.pth.tar",
                "https://drive.usercontent.google.com/download?id=1i7kt9RlCCCNEW2aYaDWVr-G778JkLNcB&export=download&confirm=t",
                "Optional temporal 3D WHAM checkpoint. TexMotion includes a native temporal WHAM core and bundled adapter bridge for the local path; an official/external WHAM implementation is optional for the full research preprocessing and camera-motion stages.",
                1L * 1024L * 1024L * 1024L,
                VideoModelKind.WHAM,
                true,
                16L * 1024L * 1024L,
                "https://github.com/yohanshin/WHAM",
                VideoAssetRole.WhamCheckpoint,
                true),
            new VideoModelDefinition(
                "hmr2",
                "HMR2 / 4D-Humans",
                "hmr2a_model.tar.gz",
                "https://www.cs.utexas.edu/~pavlakos/4dhumans/hmr2a_model.tar.gz",
                "Official HMR2.0a image-conditioned body-model archive from 4D-Humans. The bundled TexMotion bridge loads it through the official runtime; the archive alone still needs the runtime and licensed neutral SMPL body.",
                1L * 1024L * 1024L * 1024L,
                VideoModelKind.HMR2,
                true,
                16L * 1024L * 1024L,
                "https://github.com/shubham-goel/4D-Humans",
                VideoAssetRole.Hmr2Checkpoint,
                true,
                false,
                new[] { "hmr2a.ckpt", "hmr2b.ckpt", "hmr2_model.tar.gz", "hmr2_data.tar.gz" },
                null,
                new[] { ".tar.gz", ".ckpt", ".pth", ".pt" },
                false,
                new VideoAssetGuide(
                    "https://www.cs.utexas.edu/~pavlakos/4dhumans/hmr2a_model.tar.gz",
                    "{VideoModelDirectory}",
                    "hmr2a_model.tar.gz or the upstream hmr2_data.tar.gz (or a declared HMR2 checkpoint alias)",
                    "HMR2CheckpointPath",
                    "Download/select the archive, configure the HMR2 runtime and licensed neutral SMPL body, then run Runtime Preflight. The TexMotion bridge is automatic.",
                    "The checkpoint is an upstream 4D-Humans artifact. TexMotion does not redistribute or assert rights to its weights; the separately licensed SMPL body and runtime are also required for the official path.")),
            new VideoModelDefinition(
                "hybrik",
                "HybrIK-X (SMPL-X)",
                "hybrikx_rle_hrnet.pth",
                "https://drive.usercontent.google.com/download?id=1R0WbySXs_vceygKg_oWeLMNAZCEoCadG&export=download&confirm=t",
                "Optional analytical/neural HybrIK-X checkpoint. Google Drive can require a browser confirmation; use Source for manual provisioning if needed.",
                1L * 1024L * 1024L * 1024L,
                VideoModelKind.HybrIK,
                true,
                16L * 1024L * 1024L,
                "https://github.com/jeffffffli/HybrIK",
                guide: new VideoAssetGuide(
                    "https://github.com/jeffffffli/HybrIK",
                    "{VideoModelDirectory}",
                    "hybrikx_rle_hrnet.pth",
                    "HybrIKModelPath",
                    "Download/select the checkpoint, install a compatible HybrIK adapter, then run Runtime Preflight before extraction.",
                    "The checkpoint is an upstream artifact and does not include a runnable bridge. Verify source, checkpoint, and body-model terms separately before use."))
        };

        // Full WHAM keeps companion stages in a separate catalog so Settings
        // can show stage-specific actions. Public ViTPose/DPVO files and the
        // camera starter template are downloadable; separately licensed body
        // data remains Browse-only.
        private static readonly VideoModelDefinition[] WhamAssetDefinitions =
        {
            new VideoModelDefinition(
                "wham-smplx-body",
                "WHAM SMPL/SMPL-X Body Model",
                "SMPLX_NEUTRAL.npz",
                "https://smpl-x.is.tue.mpg.de/",
                "Required for full WHAM parity; manually provision a licensed SMPL-X or compatible SMPL body model.",
                600L * 1024L * 1024L,
                VideoModelKind.WHAM,
                true,
                1L,
                "https://smpl-x.is.tue.mpg.de/",
                VideoAssetRole.WhamBodyModel,
                true,
                true,
                new[] { "smplx/SMPLX_NEUTRAL.npz", "smplx_neutral.npz", "SMPL_NEUTRAL.pkl" },
                null,
                new[] { ".npz", ".pkl" },
                false,
                new VideoAssetGuide(
                    "https://smpl-x.is.tue.mpg.de/",
                    "{VideoModelDirectory}",
                    "SMPLX_NEUTRAL.npz (or a compatible declared SMPL/SMPL-X alias)",
                    "WHAMBodyModelPath",
                    "Register/download the body model, choose it with Browse, then run Runtime Preflight and replace any incompatible/template warning.",
                    "SMPL/SMPL-X body data is separately licensed and must be supplied by the user. TexMotion never downloads or bundles it automatically.")),
            new VideoModelDefinition(
                "wham-image-feature-backbone",
                "ViTPose 2D Detector",
                "vitpose-huge.pth",
                "https://huggingface.co/nielsr/vitpose-original-checkpoints/resolve/main/vitpose%2B_huge.pth?download=true",
                "ViTPose checkpoint used for official WHAM 2D preprocessing. It is a detector and cannot substitute for the HMR2 ViT image-feature runner.",
                4L * 1024L * 1024L * 1024L,
                VideoModelKind.WHAM,
                true,
                16L * 1024L * 1024L,
                "https://github.com/ViTAE-Transformer/ViTPose",
                VideoAssetRole.WhamImageFeatureBackbone,
                true,
                false,
                new[] { "vitpose_huge.pth", "vitpose-huge.pth.tar", "vitpose-h-multi-coco.pth", "image_feature_backbone.pth" },
                null,
                new[] { ".pth", ".pt", ".tar" },
                false,
                new VideoAssetGuide(
                    "https://github.com/ViTAE-Transformer/ViTPose",
                    "{VideoModelDirectory}",
                    "vitpose-huge.pth",
                    "WHAMImageFeatureBackbonePath",
                    "Download the detector checkpoint, then run Runtime Preflight. HMR2 image features are configured separately.",
                    "The checkpoint is public to fetch, but its architecture/runtime terms and integration are upstream responsibilities; a raw state-dict is not executable by itself.")),
            new VideoModelDefinition(
                "wham-image-feature-archive",
                "WHAM Precomputed Image Feature Archive (HMR2 features)",
                "vitpose_features.npz",
                "https://github.com/yohanshin/WHAM",
                "Manual precomputed per-frame HMR2 image-feature archive. It is not a ViTPose 2D keypoint archive; use this when a compatible HMR2 image-feature runner is not available, and keep rows aligned with sampled frames.",
                1L,
                VideoModelKind.WHAM,
                true,
                1L,
                "https://github.com/yohanshin/WHAM",
                VideoAssetRole.WhamImageFeatureArchive,
                false,
                true,
                new[] { "image_features.npz", "vitpose_features.npy", "image_feature_archive.npz", "hmr2_image_features.npz" },
                null,
                new[] { ".npy", ".npz", ".pt", ".pth" },
                false,
                new VideoAssetGuide(
                    "https://github.com/yohanshin/WHAM",
                    "{VideoModelDirectory}",
                    "vitpose_features.npz (or a compatible HMR2 feature archive alias)",
                    "WHAMImageFeaturePath",
                    "In Settings > Video2Motion, choose a source video and press Create ViTPose Feature Archive. The one-click exporter writes aligned HMR2 feature rows and registers the archive; run Runtime Preflight afterwards.",
                    "The one-click exporter requires a compatible local HMR2/ViTPose feature model definition/factory and checkpoint. Generated rows are tied to the selected video's FPS and trim range; upstream data rights remain the user's responsibility.")),
            new VideoModelDefinition(
                "wham-vitpose-model-definition",
                "WHAM ViTPose Model Definition Hook (HMR2 image features)",
                WhamImageFeatureModelDefinitionFileName,
                "https://github.com/ViTAE-Transformer/ViTPose",
                "Installs the local image-feature model contract template. A raw HMR2/ViTPose checkpoint still needs a compatible model factory (official HMR2, MMPose/Transformers, or a project adapter); the template makes that requirement explicit and is passed to the runner when configured. It is not a ViTPose 2D detector implementation.",
                8L * 1024L,
                VideoModelKind.WHAM,
                true,
                1L,
                "https://github.com/ViTAE-Transformer/ViTPose",
                VideoAssetRole.WhamImageFeatureModelDefinition,
                true,
                false,
                null,
                null,
                new[] { ".py" },
                false,
                new VideoAssetGuide(
                    "https://github.com/ViTAE-Transformer/ViTPose",
                    "{VideoModelDirectory}",
                    WhamImageFeatureModelDefinitionFileName,
                    "WHAMImageFeatureModelDefinitionPath",
                    "Replace the bundled contract template with a compatible HMR2/ViTPose create_model factory, choose it with Browse, and run Runtime Preflight.",
                    "The package template documents the hook only; the upstream ViTPose architecture and weights are not redistributed as a runnable implementation.")),
            new VideoModelDefinition(
                "wham-vitpose-runner-config",
                "WHAM ViTPose Runner Configuration (HMR2 image features)",
                WhamImageFeatureConfigFileName,
                "https://github.com/ViTAE-Transformer/ViTPose",
                "Installs a local runner configuration template with the expected WHAM image-feature contract. It does not replace the upstream HMR2/ViTPose architecture package or the separate ViTPose 2D detector; missing or incompatible hooks remain visible in extraction diagnostics.",
                4L * 1024L,
                VideoModelKind.WHAM,
                true,
                1L,
                "https://github.com/ViTAE-Transformer/ViTPose",
                VideoAssetRole.WhamImageFeatureConfig,
                true,
                false,
                null,
                null,
                new[] { ".json", ".yaml", ".yml", ".py" },
                false,
                new VideoAssetGuide(
                    "https://github.com/ViTAE-Transformer/ViTPose",
                    "{VideoModelDirectory}",
                    WhamImageFeatureConfigFileName,
                    "WHAMImageFeatureConfigPath",
                    "Review the runner/feature contract, replace the template if needed, choose the config with Browse, and run Runtime Preflight.",
                    "A runner config is metadata, not the ViTPose implementation. The compatible upstream runner/factory remains a manual project responsibility.")),
            new VideoModelDefinition(
                "wham-camera",
                "WHAM Camera Calibration/Config",
                "camera.yaml",
                "https://github.com/yohanshin/WHAM",
                "Installs a starter camera.yaml template. Replace its intrinsics for the input video; calibration alone does not contain motion.",
                16L * 1024L,
                VideoModelKind.WHAM,
                true,
                1L,
                "https://github.com/yohanshin/WHAM",
                VideoAssetRole.WhamCamera,
                true,
                false,
                new[] { "camera.yml", "camera_config.yaml", "camera_config.json" },
                null,
                new[] { ".yaml", ".yml", ".json", ".npy", ".npz", ".pt", ".pth" },
                false,
                new VideoAssetGuide(
                    "https://github.com/yohanshin/WHAM",
                    "{VideoModelDirectory}",
                    "camera.yaml",
                    "WHAMCameraModelPath",
                    "Replace zero/template intrinsics with calibrated values for the input video, choose the file with Browse, and run Runtime Preflight.",
                    "Camera calibration is video-specific. The package may install a starter template, but it cannot claim calibrated values or recovered world motion for it.")),
            new VideoModelDefinition(
                "wham-dpvo",
                "WHAM DPVO Camera-Motion Assets",
                "dpvo.pth",
                "https://huggingface.co/pablovela5620/dpvo/resolve/main/dpvo.pth?download=true",
                "Optional downloadable DPVO checkpoint. A configured DPVO runtime can export camera poses; otherwise TexMotion creates a conservative per-video local optical-flow pose cache and labels that fallback in the result.",
                16L * 1024L * 1024L,
                VideoModelKind.WHAM,
                true,
                1024L * 1024L,
                "https://github.com/princeton-vl/DPVO",
                VideoAssetRole.WhamDpvo,
                true,
                false,
                new[] { "dpvo.pth.tar", "dpvo_model.pth", "dpvo.ckpt" },
                null,
                new[] { ".pth", ".pt", ".tar", ".ckpt", ".npy", ".npz", ".json" },
                false,
                new VideoAssetGuide(
                    "https://github.com/princeton-vl/DPVO",
                    "{VideoModelDirectory}",
                    "dpvo.pth (or a compatible DPVO pose archive)",
                    "WHAMDpvoModelPath",
                    "Install a compatible DPVO runtime when using the checkpoint, choose the checkpoint/export with Browse, and run Runtime Preflight.",
                    "The checkpoint alone does not provide a camera-motion runner. TexMotion keeps the deterministic local fallback explicit when DPVO is unavailable.")),
            new VideoModelDefinition(
                "wham-adapter",
                "WHAM Adapter Bridge",
                WhamAdapterFileName,
                "https://github.com/yohanshin/WHAM",
                "Bundled TexMotion adapter bridge; an external full implementation may be selected explicitly.",
                WhamAdapterEstimatedBytes,
                VideoModelKind.WHAM,
                false,
                1L,
                "https://github.com/yohanshin/WHAM",
                VideoAssetRole.WhamAdapter,
                true,
                false),
            new VideoModelDefinition(
                "wham-asset-manifest",
                "WHAM Offline Asset Manifest",
                WhamAssetManifestFileName,
                "https://github.com/yohanshin/WHAM",
                "Contract metadata for deterministic local WHAM stage resolution.",
                16L * 1024L,
                VideoModelKind.WHAM,
                false,
                1L,
                "https://github.com/yohanshin/WHAM",
                VideoAssetRole.WhamManifest,
                true,
                false,
                null,
                null,
                new[] { ".json" })
        };

        // HMR2 uses a TexMotion-owned bridge. The official 4D-Humans runtime,
        // checkpoint and licensed SMPL body remain user-provided, but users do
        // not need to author a Python adapter just to select HMR2.
        private static readonly VideoModelDefinition[] RuntimeAdapterDefinitions =
        {
            new VideoModelDefinition(
                "hmr2-adapter",
                "HMR2 / 4D-Humans Adapter",
                "texmotion_hmr2_adapter.py",
                "https://github.com/shubham-goel/4D-Humans",
                "TexMotion's bundled official HMR2 bridge. Configure the official 4D-Humans runtime, checkpoint and licensed neutral SMPL body; a custom adapter remains an optional override.",
                1L,
                VideoModelKind.HMR2,
                true,
                1L,
                "https://github.com/shubham-goel/4D-Humans",
                VideoAssetRole.Hmr2Adapter,
                true,
                true,
                new[] { "hmr2_adapter.py", "texmotion_hmr2_adapter.py" },
                null,
                new[] { ".py" },
                false,
                new VideoAssetGuide(
                    "https://github.com/shubham-goel/4D-Humans",
                    "{VideoModelDirectory}",
                    "texmotion_hmr2_adapter.py (or a compatible HMR2 adapter)",
                    "HMR2AdapterPath",
                    "The TexMotion bridge is selected automatically. Use Browse only to override it with a compatible local adapter, then run Runtime Preflight.",
                    "TexMotion does not redistribute the official HMR2 runtime, checkpoint, or separately licensed SMPL body model. The bridge itself is included.")),
            new VideoModelDefinition(
                "hmr2-runtime",
                "Official HMR2 / 4D-Humans Runtime",
                "hmr2-runtime",
                "https://github.com/shubham-goel/4D-Humans",
                "Manual source checkout required by the official HMR2 runtime. Follow the upstream README (Python 3.10, environment.yml or pip install -e .[all]), provide the HMR2 archive plus neutral SMPL, then configure a local adapter.",
                1L,
                VideoModelKind.HMR2,
                true,
                1L,
                "https://github.com/shubham-goel/4D-Humans",
                VideoAssetRole.Hmr2Runtime,
                true,
                true,
                new[] { "4D-Humans", "hmr2" },
                null,
                null,
                true,
                new VideoAssetGuide(
                    "https://github.com/shubham-goel/4D-Humans",
                    "{VideoModelDirectory}/hmr2-runtime",
                    "4D-Humans source checkout containing hmr2/ (or package metadata identifying HMR2)",
                    "HMR2RuntimePath",
                    "Clone/install the pinned upstream runtime outside the package, select its folder with Browse Folder, configure its Python environment, and run Runtime Preflight.",
                    "The official HMR2 source/runtime and its dependencies are third-party research software. TexMotion cannot bundle or install them under unverified terms.")),
            new VideoModelDefinition(
                "hmr2-smpl-body",
                "HMR2 SMPL Neutral Body Model",
                "basicModel_neutral_lbs_10_207_0_v1.0.0.pkl",
                "https://smplify.is.tue.mpg.de/",
                "Manual licensed SMPL neutral model required by the official 4D-Humans runtime. Register with the upstream provider and select the downloaded .pkl file with Browse.",
                1L,
                VideoModelKind.HMR2,
                true,
                1L,
                "https://smplify.is.tue.mpg.de/",
                VideoAssetRole.Hmr2BodyModel,
                true,
                true,
                new[] { "SMPL_NEUTRAL.pkl", "smpl/SMPL_NEUTRAL.pkl" },
                null,
                // Official 4D-Humans calls smplx.SMPL with the licensed
                // neutral SMPL pickle.  Accepting SMPL-X .npz files here
                // makes the row look ready and only fails later in model
                // construction, so keep the role-specific contract strict.
                new[] { ".pkl" },
                false,
                new VideoAssetGuide(
                    "https://smpl.is.tue.mpg.de/",
                    "{VideoModelDirectory}",
                    "basicModel_neutral_lbs_10_207_0_v1.0.0.pkl (or a compatible neutral SMPL alias)",
                    "HMR2BodyModelPath",
                    "Register/download the neutral SMPL body model, choose the .pkl file with Browse, then run HMR2 Runtime Preflight.",
                    "SMPL is separately licensed. The official HMR2 runtime requires the user to obtain and register this body model; TexMotion never downloads it automatically.")),
            new VideoModelDefinition(
                "vitpose-runtime",
                "Compatible ViTPose/MMPose Runtime",
                "vitpose-runtime",
                "https://github.com/ViTAE-Transformer/ViTPose",
                "Optional manual runtime for an explicitly selected compatible ViTPose/MMPose image-feature runner. The normal WHAM path uses the bundled official HMR2 runner instead.",
                1L,
                VideoModelKind.WHAM,
                true,
                1L,
                "https://github.com/ViTAE-Transformer/ViTPose",
                VideoAssetRole.ViTPoseRuntime,
                false,
                true,
                new[] { "mmpose", "vitpose" },
                null,
                null,
                true,
                new VideoAssetGuide(
                    "https://github.com/ViTAE-Transformer/ViTPose",
                    "{VideoModelDirectory}/vitpose-runtime",
                    "A compatible ViTPose/MMPose source checkout or installed runtime",
                    "ViTPoseRuntimePath",
                    "Select a compatible runtime folder with Browse Folder, configure its checkpoint/configuration, then run Runtime Preflight.",
                    "The compatible runtime and its model definitions are upstream software. TexMotion does not download or require a Python module name for the default official HMR2 route.")),
            new VideoModelDefinition(
                "hybrik-adapter",
                "HybrIK / HybrIK-X Adapter",
                "texmotion_hybrik_adapter.py",
                "https://github.com/jeffffffli/HybrIK",
                "Manual compatible adapter for the HybrIK-X checkpoint. Configure a bridge exposing create_backend/infer_sequence before selecting HybrIK.",
                1L,
                VideoModelKind.HybrIK,
                true,
                1L,
                "https://github.com/jeffffffli/HybrIK",
                VideoAssetRole.HybrIKAdapter,
                true,
                true,
                new[] { "hybrik_adapter.py", "texmotion_hybrik_adapter.py" },
                null,
                new[] { ".py" },
                false,
                new VideoAssetGuide(
                    "https://github.com/jeffffffli/HybrIK",
                    "{VideoModelDirectory}",
                    "texmotion_hybrik_adapter.py (or a compatible HybrIK adapter)",
                    "HybrIKAdapterPath",
                    "Install/configure a bridge exposing create_backend and infer_sequence, choose the .py file with Browse, then run Runtime Preflight.",
                    "The HybrIK source, runtime, and checkpoint usage terms are upstream responsibilities. TexMotion requires an explicit local adapter instead of treating a raw checkpoint as executable."))
        };

        private static readonly VideoDependencyDefinition[] DependencyDefinitions =
        {
            new VideoDependencyDefinition(
                "python-lightweight",
                "Python Lightweight Video Dependencies",
                "requirements.txt",
                PythonDependencyProfile.Lightweight,
                "MediaPipe, OpenCV, NumPy, SciPy, and ONNX Runtime for the offline 2D/3D detector path.",
                "https://github.com/k0ta0uchi/TexMotion",
                new[] { "mediapipe", "opencv-python", "numpy", "scipy", "onnxruntime" }),
            new VideoDependencyDefinition(
                "python-pytorch-quality",
                "Python PyTorch Quality / HMR2 Dependencies",
                "requirements-pytorch.txt",
                PythonDependencyProfile.PyTorch,
                "PyTorch quality dependencies plus the official 4D-Humans/HMR2 runtime contract. The upstream source uses Python 3.10, pip install -e .[all], and a separately licensed SMPL neutral model; source/runtime packages remain user-installed.",
                "https://github.com/shubham-goel/4D-Humans",
                new[]
                {
                    "torch", "torchvision", "pytorch-lightning", "smplx==0.1.28",
                    "pyrender", "opencv-python", "yacs", "scikit-image", "einops",
                    "timm", "dill", "pandas", "gdown", "webdataset", "chumpy",
                    "detectron2 (optional [all])"
                })
        };

        public static IReadOnlyList<VideoModelDefinition> All => Definitions;

        /// <summary>
        /// Auxiliary WHAM assets required for full parity. They are separate
        /// from <see cref="All"/> so Settings can show stage-specific actions;
        /// Download All includes public companion artifacts and the template.
        /// </summary>
        public static IReadOnlyList<VideoModelDefinition> WhamAssets => WhamAssetDefinitions;

        /// <summary>Manual quality-backend bridges/runtime sources.</summary>
        public static IReadOnlyList<VideoModelDefinition> RuntimeAdapters => RuntimeAdapterDefinitions;

        /// <summary>Bundled requirements manifests for explicit Python setup.</summary>
        public static IReadOnlyList<VideoDependencyDefinition> PythonDependencies => DependencyDefinitions;

        public static VideoModelDefinition Find(string id)
        {
            if (string.IsNullOrWhiteSpace(id)) return null;
            for (int i = 0; i < Definitions.Length; i++)
            {
                if (string.Equals(Definitions[i].Id, id.Trim(), StringComparison.OrdinalIgnoreCase))
                    return Definitions[i];
            }
            // Companion IDs are discoverable through the same catalog lookup,
            // while remaining absent from All so legacy UI totals/download-all
            // semantics do not change.
            for (int i = 0; i < WhamAssetDefinitions.Length; i++)
            {
                if (string.Equals(WhamAssetDefinitions[i].Id, id.Trim(), StringComparison.OrdinalIgnoreCase))
                    return WhamAssetDefinitions[i];
            }
            for (int i = 0; i < RuntimeAdapterDefinitions.Length; i++)
            {
                if (string.Equals(RuntimeAdapterDefinitions[i].Id, id.Trim(), StringComparison.OrdinalIgnoreCase))
                    return RuntimeAdapterDefinitions[i];
            }
            return null;
        }

        public static VideoModelDefinition FindWhamAsset(string id)
        {
            if (string.IsNullOrWhiteSpace(id)) return null;
            for (int i = 0; i < WhamAssetDefinitions.Length; i++)
            {
                if (string.Equals(WhamAssetDefinitions[i].Id, id.Trim(), StringComparison.OrdinalIgnoreCase))
                    return WhamAssetDefinitions[i];
            }
            return null;
        }

        public static VideoModelDefinition FindRuntimeAdapter(string id)
        {
            if (string.IsNullOrWhiteSpace(id)) return null;
            for (int i = 0; i < RuntimeAdapterDefinitions.Length; i++)
            {
                if (string.Equals(RuntimeAdapterDefinitions[i].Id, id.Trim(), StringComparison.OrdinalIgnoreCase))
                    return RuntimeAdapterDefinitions[i];
            }
            return null;
        }

        /// <summary>
        /// Returns the setup guide associated with a catalog row. Manual rows
        /// without bespoke metadata still get a useful deterministic fallback,
        /// so adding a manual asset can never leave the Settings UI without an
        /// actionable next step.
        /// </summary>
        public static VideoAssetGuide GetAssetGuide(VideoModelDefinition asset)
        {
            if (asset == null || !asset.HasSetupGuide) return null;
            if (asset.Guide != null) return asset.Guide;
            return new VideoAssetGuide(
                string.IsNullOrWhiteSpace(asset.ReferenceUrl) ? asset.RemoteUrl : asset.ReferenceUrl,
                "{VideoModelDirectory}",
                asset.FileName,
                "Video2Motion model catalog",
                "Open Settings > Video2Motion, run Runtime Preflight, and resolve this row before extraction.",
                "This asset is marked for manual/local provisioning under its upstream terms.");
        }

        public static string GetDestinationPath(TexMotionSettings settings, VideoModelDefinition model)
        {
            if (settings == null || model == null) return null;
            return settings.GetVideoModelPath(model.FileName);
        }

        /// <summary>
        /// Returns the persisted explicit model path when one exists, otherwise
        /// the catalog cache destination.  This is deliberately separate from
        /// <see cref="GetDestinationPath"/> so changing a local override never
        /// changes where a new catalog download is written.
        /// </summary>
        public static string GetConfiguredModelPath(TexMotionSettings settings, VideoModelDefinition model)
        {
            if (settings == null || model == null) return null;
            string configured = null;
            switch (model.AssetRole)
            {
                case VideoAssetRole.Hmr2Checkpoint:
                case VideoAssetRole.WhamCheckpoint:
                    configured = settings.GetQualityModelPath(model.Kind == VideoModelKind.HMR2
                        ? VideoPoseBackend.HMR2
                        : VideoPoseBackend.WHAM);
                    break;
                case VideoAssetRole.Hmr2BodyModel:
                    configured = settings.GetHMR2BodyModelPath();
                    break;
                case VideoAssetRole.Hmr2Runtime:
                case VideoAssetRole.Hmr2Adapter:
                case VideoAssetRole.ViTPoseRuntime:
                case VideoAssetRole.HybrIKAdapter:
                case VideoAssetRole.WhamAdapter:
                    configured = GetRuntimeAdapterPath(settings, model);
                    break;
                case VideoAssetRole.InferenceModel:
                    if (model.Kind == VideoModelKind.RTMPose || model.Kind == VideoModelKind.DWPose)
                        configured = settings.RTMPoseModelPath;
                    break;
            }
            if (!string.IsNullOrWhiteSpace(configured))
            {
                if (model.AssetRole == VideoAssetRole.WhamAdapter &&
                    !Path.IsPathRooted(configured.Trim()) &&
                    configured.IndexOf(Path.DirectorySeparatorChar) < 0 &&
                    configured.IndexOf(Path.AltDirectorySeparatorChar) < 0 &&
                    !configured.Trim().EndsWith(".py", StringComparison.OrdinalIgnoreCase))
                    return configured.Trim();
                try { return Path.GetFullPath(configured.Trim()); }
                catch { return configured.Trim(); }
            }
            return GetDestinationPath(settings, model);
        }

        public static string GetRuntimeAdapterPath(TexMotionSettings settings, VideoModelDefinition asset)
        {
            if (settings == null || asset == null) return null;
            if (asset.AssetRole == VideoAssetRole.Hmr2Runtime)
                return settings.GetHMR2RuntimePath();
            if (asset.AssetRole == VideoAssetRole.ViTPoseRuntime)
                return settings.GetViTPoseRuntimePath();
            if (asset.AssetRole == VideoAssetRole.WhamAdapter ||
                asset.AssetRole == VideoAssetRole.Hmr2Adapter ||
                asset.AssetRole == VideoAssetRole.HybrIKAdapter)
            {
                string configured = asset.AssetRole == VideoAssetRole.Hmr2Adapter
                    ? settings.GetVideoQualityAdapterPath(VideoPoseBackend.HMR2)
                    : asset.AssetRole == VideoAssetRole.HybrIKAdapter
                        ? settings.GetVideoQualityAdapterPath(VideoPoseBackend.HybrIK)
                        : settings.GetVideoQualityAdapterPath(VideoPoseBackend.WHAM);
                if (!string.IsNullOrWhiteSpace(configured))
                {
                    return configured;
                }
                if (asset.AssetRole == VideoAssetRole.Hmr2Adapter)
                {
                    string bundled = GetBundledHmr2AdapterPath();
                    if (!string.IsNullOrWhiteSpace(bundled)) return bundled;
                }
            }
            return GetDestinationPath(settings, asset);
        }

        public static string GetPythonRequirementsPath(TexMotionSettings settings, PythonDependencyProfile profile)
        {
            if (settings == null) return null;
            // Settings owns package/project path resolution and retains the
            // existing local/offline behavior.
            string path = settings.GetVideoRequirementsPath(profile);
            if (string.IsNullOrWhiteSpace(path))
                return profile == PythonDependencyProfile.PyTorch ? "requirements-pytorch.txt" : "requirements.txt";
            return path;
        }

        public static string GetWhamAdapterPath(TexMotionSettings settings)
        {
            return settings == null ? null : settings.GetVideoModelPath(WhamAdapterFileName);
        }

        /// <summary>
        /// Finds the HMR2 bridge shipped beside video_pose_extractor.py. This
        /// works for both package-cache and Assets installations and avoids a
        /// second copy of the adapter in the user's model directory.
        /// </summary>
        public static string GetBundledHmr2AdapterPath()
        {
            try
            {
                string scriptPath = VideoMotionJobRunner.FindScriptPath();
                string videoDirectory = Path.GetDirectoryName(scriptPath);
                if (string.IsNullOrWhiteSpace(videoDirectory)) return null;
                string[] candidates =
                {
                    Path.Combine(videoDirectory, "pose_pipeline", "adapters", "texmotion_hmr2_adapter.py"),
                    Path.Combine(videoDirectory, "adapters", "texmotion_hmr2_adapter.py")
                };
                for (int i = 0; i < candidates.Length; i++)
                {
                    if (File.Exists(candidates[i])) return Path.GetFullPath(candidates[i]);
                }
            }
            catch { }
            return null;
        }

        public static string GetWhamAssetManifestPath(TexMotionSettings settings)
        {
            if (settings == null) return null;
            VideoModelDefinition manifest = FindWhamAsset("wham-asset-manifest");
            return manifest == null ? settings.GetVideoModelPath(WhamAssetManifestFileName) : GetConfiguredWhamAssetPath(settings, manifest);
        }

        public static string GetWhamAssetPath(TexMotionSettings settings, VideoModelDefinition asset)
        {
            return settings == null || asset == null ? null : settings.GetVideoModelPath(asset.FileName);
        }

        public static string GetConfiguredWhamAssetPath(TexMotionSettings settings, VideoModelDefinition asset)
        {
            if (settings == null || asset == null) return null;
            string configured = null;
            switch (asset.AssetRole)
            {
                case VideoAssetRole.WhamBodyModel: configured = settings.WHAMBodyModelPath; break;
                case VideoAssetRole.Hmr2BodyModel: configured = settings.HMR2BodyModelPath; break;
                case VideoAssetRole.WhamImageFeatureBackbone:
                    configured = string.IsNullOrWhiteSpace(settings.ViTPoseModelPath)
                        ? settings.WHAMImageFeatureBackbonePath
                        : settings.ViTPoseModelPath;
                    break;
                case VideoAssetRole.WhamImageFeatureArchive: configured = settings.WHAMImageFeaturePath; break;
                case VideoAssetRole.WhamImageFeatureModelDefinition:
                    configured = string.IsNullOrWhiteSpace(settings.ViTPoseModelDefinitionPath)
                        ? settings.WHAMImageFeatureModelDefinitionPath
                        : settings.ViTPoseModelDefinitionPath;
                    break;
                case VideoAssetRole.WhamImageFeatureConfig:
                    configured = string.IsNullOrWhiteSpace(settings.ViTPoseConfigPath)
                        ? settings.WHAMImageFeatureConfigPath
                        : settings.ViTPoseConfigPath;
                    break;
                case VideoAssetRole.WhamCamera: configured = settings.WHAMCameraModelPath; break;
                case VideoAssetRole.WhamDpvo: configured = settings.WHAMDpvoModelPath; break;
                case VideoAssetRole.WhamAdapter: configured = settings.GetVideoQualityAdapterPath(VideoPoseBackend.WHAM); break;
                case VideoAssetRole.WhamManifest: configured = settings.WHAMAssetManifestPath; break;
            }
            if (!string.IsNullOrWhiteSpace(configured))
            {
                if (asset.AssetRole == VideoAssetRole.WhamAdapter &&
                    !Path.IsPathRooted(configured.Trim()) &&
                    configured.IndexOf(Path.DirectorySeparatorChar) < 0 &&
                    configured.IndexOf(Path.AltDirectorySeparatorChar) < 0 &&
                    !configured.Trim().EndsWith(".py", StringComparison.OrdinalIgnoreCase))
                    return configured.Trim();
                try { return Path.GetFullPath(configured.Trim()); }
                catch { return configured.Trim(); }
            }
            return GetWhamAssetPath(settings, asset);
        }

        /// <summary>
        /// Destination used by a companion download. An explicit Settings
        /// path is honored even when the file is not present yet; otherwise
        /// the configured Video2Motion cache is used.
        /// </summary>
        public static string GetWhamAssetDownloadPath(TexMotionSettings settings, VideoModelDefinition asset)
        {
            if (settings == null || asset == null) return null;
            return GetConfiguredWhamAssetPath(settings, asset);
        }

        public static IReadOnlyDictionary<string, string> GetWhamAssetPaths(TexMotionSettings settings)
        {
            var paths = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
            for (int i = 0; i < WhamAssetDefinitions.Length; i++)
            {
                VideoModelDefinition asset = WhamAssetDefinitions[i];
                string path = GetConfiguredWhamAssetPath(settings, asset);
                if (!string.IsNullOrEmpty(path))
                {
                    string role = asset.AssetRole.ToString();
                    paths[role] = path;
                    if (asset.AssetRole == VideoAssetRole.WhamBodyModel)
                    {
                        paths["body_model"] = path;
                        paths["smpl_model"] = path;
                        paths["smplx_model"] = path;
                    }
                    if (asset.IsWhamViTPose2DDetector)
                    {
                        // ``image_feature_backbone`` is the historical
                        // Python resolver key. Keep it for compatibility,
                        // but expose an unambiguous key for editor callers:
                        // this path is the ViTPose 2D detector, never the
                        // HMR2 image-feature runner/checkpoint.
                        paths["image_feature_backbone"] = path;
                        paths["vitpose_2d"] = path;
                    }
                    if (asset.AssetRole == VideoAssetRole.WhamImageFeatureArchive)
                    {
                        paths["image_feature_archive"] = path;
                        paths["imageFeaturePath"] = path;
                    }
                    if (asset.AssetRole == VideoAssetRole.WhamImageFeatureModelDefinition)
                    {
                        paths["image_feature_model_definition"] = path;
                        paths["vitpose_model_definition"] = path;
                    }
                    if (asset.AssetRole == VideoAssetRole.WhamImageFeatureConfig)
                    {
                        paths["image_feature_config"] = path;
                        paths["vitpose_config"] = path;
                    }
                    if (asset.AssetRole == VideoAssetRole.WhamCamera) paths["camera"] = path;
                    if (asset.AssetRole == VideoAssetRole.WhamDpvo) paths["dpvo"] = path;
                    if (asset.AssetRole == VideoAssetRole.WhamAdapter) paths["adapter"] = path;
                }
            }
            string checkpoint = GetConfiguredModelPath(settings, Find("wham"));
            if (!string.IsNullOrEmpty(checkpoint)) paths["checkpoint"] = checkpoint;
            string hmr2ImageFeature = GetConfiguredModelPath(settings, Find("hmr2"));
            if (!string.IsNullOrEmpty(hmr2ImageFeature))
            {
                paths["hmr2_image_feature"] = hmr2ImageFeature;
                // Keep the official HMR2 image-feature runner's three input
                // roles explicit for adapters that consume the asset map
                // directly.  The ViTPose 2D detector remains under its own
                // ``image_feature_backbone``/``vitpose_2d`` keys above.
                paths["hmr2_checkpoint"] = hmr2ImageFeature;
            }
            if (settings != null)
            {
                string hmr2Runtime = settings.GetHMR2RuntimePath();
                if (!string.IsNullOrWhiteSpace(hmr2Runtime)) paths["hmr2_runtime"] = hmr2Runtime;
                string hmr2Body = settings.GetHMR2BodyModelPath();
                if (!string.IsNullOrWhiteSpace(hmr2Body)) paths["hmr2_body_model"] = hmr2Body;
                paths["image_feature_runner"] = settings.GetWHAMRunnerKind();
            }
            if (settings != null)
            {
                string hmr2Checkpoint = settings.GetHMR2CheckpointPathIfPresent();
                if (!string.IsNullOrEmpty(hmr2Checkpoint))
                {
                    paths["hmr2_checkpoint"] = hmr2Checkpoint;
                    paths["hmr2CheckpointPath"] = hmr2Checkpoint;
                }
                string hmr2Runtime = settings.GetHMR2RuntimePathIfPresent();
                if (!string.IsNullOrEmpty(hmr2Runtime))
                {
                    paths["hmr2_runtime"] = hmr2Runtime;
                    paths["hmr2RuntimePath"] = hmr2Runtime;
                }
                string vitposeRuntime = settings.GetViTPoseRuntimePathIfPresent();
                if (!string.IsNullOrEmpty(vitposeRuntime))
                    paths["vitpose_runtime"] = vitposeRuntime;
            }
            string manifest = GetWhamAssetManifestPath(settings);
            if (!string.IsNullOrEmpty(manifest))
            {
                paths["manifest"] = manifest;
                paths["asset_manifest"] = manifest;
            }
            // Precomputed HMR2 image features are an auxiliary archive rather
            // than the ViTPose 2D detector or a WHAM checkpoint stage.
            // Preserve the explicit path for callers that build a backend
            // from this dictionary directly.
            if (settings != null && !string.IsNullOrWhiteSpace(settings.WHAMImageFeaturePath))
            {
                string archive = settings.WHAMImageFeaturePath.Trim();
                try { archive = Path.GetFullPath(archive); } catch { }
                paths["image_feature_archive"] = archive;
                paths["imageFeaturePath"] = archive;
            }
            return paths;
        }

        /// <summary>
        /// Returns one deterministic status row for every asset needed by the
        /// selected backend.  Rows stay in catalog order and preserve both the
        /// configured path and the resolved local path.
        /// </summary>
        public static IReadOnlyList<VideoAssetStatusInfo> GetRuntimeAssetStatuses(
            TexMotionSettings settings,
            VideoPoseBackend backend)
        {
            var statuses = new List<VideoAssetStatusInfo>();
            if (settings == null)
            {
                statuses.Add(new VideoAssetStatusInfo
                {
                    Id = "settings",
                    DisplayName = "TexMotion Settings",
                    Role = VideoAssetRole.PythonRuntime,
                    Status = VideoAssetStatus.Missing,
                    Reason = "TexMotionSettings is null; no local asset directory can be resolved.",
                    Required = true,
                    CanInstall = false
                });
                return statuses;
            }

            PythonDependencyProfile profile = RequiresQualityDependencies(backend)
                ? PythonDependencyProfile.PyTorch
                : PythonDependencyProfile.Lightweight;
            VideoDependencyDefinition dependency = FindDependency(profile);
            string requirementsPath = GetPythonRequirementsPath(settings, profile);
            statuses.Add(CreateDependencyStatus(settings, dependency, requirementsPath));

            string pythonPath = settings.GetEffectiveVideoPythonExecutablePath();
            string expectedPythonPath = settings.UseVideoVenv
                ? settings.GetVideoVenvPythonPath()
                : (string.IsNullOrWhiteSpace(settings.CustomPythonExecutablePath)
                    ? "<select a Python executable>"
                    : settings.CustomPythonExecutablePath);
            statuses.Add(new VideoAssetStatusInfo
            {
                Id = "python-runtime",
                DisplayName = "Python Video Runtime",
                Role = VideoAssetRole.PythonRuntime,
                Status = IsReadableFile(pythonPath, 1L)
                    ? VideoAssetStatus.Ready
                    : VideoAssetStatus.Missing,
                ExpectedPath = expectedPythonPath,
                ResolvedPath = IsReadableFile(pythonPath, 1L) ? pythonPath : null,
                Reason = IsReadableFile(pythonPath, 1L)
                    ? "Interpreter exists; use Re-probe to verify imported packages."
                    : settings.UseVideoVenv
                        ? "Project Video Motion venv is not installed. Click Install/Build Video Motion venv."
                        : "Select an existing Python executable or enable the project venv install action.",
                Required = true,
                CanInstall = true
            });

            switch (backend)
            {
                case VideoPoseBackend.Auto:
                    // Auto has a deterministic MediaPipe baseline and may add
                    // RTMPose when its optional ONNX model is installed.
                    AddModelStatus(statuses, settings, FindMediaPipeDefinition(settings.VideoModelComplexity), true);
                    AddModelStatus(statuses, settings, Find("rtmpose-m"), false);
                    break;
                case VideoPoseBackend.RTMPose:
                    AddModelStatus(statuses, settings, Find("rtmpose-m"), true);
                    break;
                case VideoPoseBackend.MediaPipe:
                    AddModelStatus(statuses, settings, FindMediaPipeDefinition(settings.VideoModelComplexity), true);
                    break;
                case VideoPoseBackend.WHAM:
                case VideoPoseBackend.WHAMMediaPipe:
                    AddModelStatus(statuses, settings, Find("wham"), true);
                    AddWhamStatuses(statuses, settings, true);
                    AddWhamImageFeatureRunnerStatuses(statuses, settings);
                    if (backend == VideoPoseBackend.WHAMMediaPipe)
                        AddModelStatus(statuses, settings, FindMediaPipeDefinition(settings.VideoModelComplexity), true);
                    break;
                case VideoPoseBackend.HMR2:
                    AddModelStatus(statuses, settings, Find("hmr2"), true);
                    AddRuntimeAdapterStatus(statuses, settings, FindRuntimeAdapter("hmr2-smpl-body"), true);
                    AddRuntimeAdapterStatus(statuses, settings, FindRuntimeAdapter("hmr2-adapter"), true);
                    AddRuntimeAdapterStatus(statuses, settings, FindRuntimeAdapter("hmr2-runtime"), true);
                    break;
                case VideoPoseBackend.HybrIK:
                    AddModelStatus(statuses, settings, Find("hybrik"), true);
                    AddModelStatus(statuses, settings, FindWhamAsset("wham-smplx-body"), true);
                    AddRuntimeAdapterStatus(statuses, settings, FindRuntimeAdapter("hybrik-adapter"), true);
                    break;
                case VideoPoseBackend.PyTorch:
                    AddConfiguredPathStatus(
                        statuses,
                        "pytorch-checkpoint",
                        "PyTorch Quality Checkpoint",
                        VideoAssetRole.InferenceModel,
                        settings.VideoQualityModelPath,
                        true,
                        false,
                        new[] { ".pt", ".pth", ".ckpt", ".tar", ".bin" });
                    AddConfiguredPathStatus(
                        statuses,
                        "pytorch-adapter",
                        "PyTorch Quality Adapter",
                        VideoAssetRole.WhamAdapter,
                        settings.VideoQualityAdapterPath,
                        true,
                        false,
                        new[] { ".py" });
                    break;
            }

            return statuses;
        }

        /// <summary>Alias kept intentionally discoverable for callers that call this a preflight.</summary>
        public static VideoPreflightReport GetPreflightReport(
            TexMotionSettings settings,
            VideoPoseBackend backend)
        {
            return new VideoPreflightReport
            {
                Backend = backend,
                Assets = GetRuntimeAssetStatuses(settings, backend)
            };
        }

        /// <summary>
        /// Builds the small, user-facing WHAM setup contract.  The official
        /// path has three separate model responsibilities: WHAM temporal
        /// inference, HMR2 image features/initial SMPL, and ViTPose 2D
        /// detection.  Keeping these roles separate prevents a ViTPose
        /// checkpoint from being presented as an HMR2 feature extractor.
        /// </summary>
        public static VideoWhamSetupSummary GetWhamSetupSummary(
            TexMotionSettings settings,
            VideoPoseBackend backend)
        {
            var required = new List<VideoAssetStatusInfo>();
            var optional = new List<VideoAssetStatusInfo>();
            var manual = new List<VideoAssetStatusInfo>();
            var incompatible = new List<VideoAssetStatusInfo>();

            if (settings == null)
            {
                required.Add(new VideoAssetStatusInfo
                {
                    Id = "settings",
                    DisplayName = "TexMotion Settings",
                    Role = VideoAssetRole.PythonRuntime,
                    Status = VideoAssetStatus.Missing,
                    Required = true,
                    Reason = "TexMotionSettings is null."
                });
            }
            else
            {
                AddWhamSetupRequired(required, settings, Find("wham"));
                AddWhamSetupRequired(required, settings, FindWhamAsset("wham-adapter"));

                // Python is part of the executable official runner contract,
                // even though its rows are intentionally hidden in the compact
                // asset list.
                IReadOnlyList<VideoAssetStatusInfo> runtime = GetRuntimeAssetStatuses(settings, backend);
                AppendRuntimeRole(required, runtime, VideoAssetRole.PythonRequirements);
                AppendRuntimeRole(required, runtime, VideoAssetRole.PythonRuntime);

                string runnerKind = settings.GetWHAMRunnerKind();
                if (runnerKind == TexMotionWhamRunnerKinds.OfficialHmr2)
                {
                    // HMR2 supplies the official WHAM image-feature encoder
                    // and initial SMPL estimate. Keep its four files grouped
                    // in the UI, while retaining individual statuses for
                    // diagnostics.
                    AddWhamSetupRequired(required, settings, Find("hmr2"));
                    AddWhamSetupRequired(required, settings, FindRuntimeAdapter("hmr2-runtime"));
                    AddWhamSetupRequired(required, settings, FindRuntimeAdapter("hmr2-adapter"));
                    AddWhamSetupRequired(required, settings, FindRuntimeAdapter("hmr2-smpl-body"));
                }
                else if (runnerKind == TexMotionWhamRunnerKinds.CompatibleVitPose)
                {
                    AddWhamSetupRequired(required, settings, FindWhamAsset("wham-image-feature-backbone"));
                    AddWhamSetupRequired(required, settings, FindRuntimeAdapter("vitpose-runtime"));
                    if (!string.IsNullOrWhiteSpace(settings.ViTPoseConfigPath) ||
                        !string.IsNullOrWhiteSpace(settings.WHAMImageFeatureConfigPath))
                        AddWhamSetupRequired(required, settings, FindWhamAsset("wham-vitpose-runner-config"));
                    if (!string.IsNullOrWhiteSpace(settings.ViTPoseModelDefinitionPath) ||
                        !string.IsNullOrWhiteSpace(settings.WHAMImageFeatureModelDefinitionPath))
                        AddWhamSetupRequired(required, settings, FindWhamAsset("wham-vitpose-model-definition"));
                }
                else
                {
                    // Archive mode intentionally has no executable runner;
                    // the archive itself is the selected feature stage.
                    AddWhamSetupRequired(required, settings, FindWhamAsset("wham-image-feature-archive"));
                }

                // Official WHAM preprocessing still has a separate ViTPose
                // 2D detector contract in every image-feature mode. It is not
                // the HMR2 runner checkpoint and is reported independently.
                if (runnerKind != TexMotionWhamRunnerKinds.CompatibleVitPose)
                    AddWhamSetupRequired(required, settings, FindWhamAsset("wham-image-feature-backbone"));

                if (backend == VideoPoseBackend.WHAMMediaPipe)
                    AddWhamSetupRequired(required, settings, FindMediaPipeDefinition(settings.VideoModelComplexity));

                AddWhamSetupOptional(optional, settings, FindWhamAsset("wham-camera"));
                AddWhamSetupOptional(optional, settings, FindWhamAsset("wham-dpvo"));
                AddWhamSetupOptional(optional, settings, FindWhamAsset("wham-image-feature-archive"));
                AddWhamSetupOptional(optional, settings, FindWhamAsset("wham-asset-manifest"));
                AddWhamSetupOptional(optional, settings, FindWhamAsset("wham-smplx-body"));
            }

            // Keep the two dimensions independent: optional licensed or
            // malformed rows remain visible in their respective diagnostics,
            // but they must not turn a runnable native WHAM path into a
            // blocked setup state. Only rows in RequiredAssets determine the
            // compact state badge.
            ClassifyWhamSetupAssets(required, manual, incompatible);
            ClassifyWhamSetupAssets(optional, manual, incompatible);

            VideoWhamSetupState state = GetWhamSetupState(required);

            return new VideoWhamSetupSummary
            {
                Backend = backend,
                State = state,
                RequiredAssets = required,
                OptionalAssets = optional,
                ManualAssets = manual,
                IncompatibleAssets = incompatible,
                OfficialRunnerStatus = GetWhamOfficialRunnerStatus(required, settings),
                // Keep the diagnostics keyed by their actual model
                // responsibility: HMR2 produces image features/initial SMPL;
                // the ViTPose row below is only the separate 2D detector.
                Hmr2ImageFeaturesStatus = GetWhamImageFeatureSummaryStatus(required, settings),
                VitPose2DStatus = GetWhamSetupRoleStatus(required, VideoAssetRole.WhamImageFeatureBackbone)
            };
        }

        private static void AddConfiguredRunnerSetupRequired(
            List<VideoAssetStatusInfo> destination,
            TexMotionSettings settings,
            string id,
            string displayName,
            VideoAssetRole role,
            string path,
            string[] extensions)
        {
            if (destination == null || settings == null) return;
            string normalized = path;
            try
            {
                if (!string.IsNullOrWhiteSpace(normalized)) normalized = Path.GetFullPath(normalized.Trim());
            }
            catch { }
            bool isDirectory = role == VideoAssetRole.ViTPoseRuntime &&
                               !string.IsNullOrWhiteSpace(normalized) &&
                               Directory.Exists(normalized);
            bool exists = isDirectory || IsReadableFile(normalized, 1L);
            bool compatible = exists && (isDirectory || HasSupportedExtension(normalized, extensions));
            VideoAssetStatus status = !exists
                ? VideoAssetStatus.Manual
                : compatible ? VideoAssetStatus.Ready : VideoAssetStatus.Incompatible;
            destination.Add(new VideoAssetStatusInfo
            {
                Id = id,
                DisplayName = displayName,
                Role = role,
                Status = status,
                Required = true,
                ErrorCode = status == VideoAssetStatus.Ready
                    ? null
                    : role == VideoAssetRole.ViTPoseRuntime
                        ? VideoFeatureErrorCodes.RunnerRuntimeMissing
                        : VideoFeatureErrorCodes.ModelFactoryMissing,
                NextAction = status == VideoAssetStatus.Ready
                    ? string.Empty
                    : "Open Guide, choose a compatible runtime with Browse, then run Runtime Preflight.",
                ExpectedPath = normalized,
                ResolvedPath = exists && compatible ? normalized : null,
                Reason = status == VideoAssetStatus.Ready
                    ? "Compatible ViTPose runtime is configured."
                    : "A compatible ViTPose/MMPose runtime must be supplied explicitly; the official HMR2 path does not use this setting."
            });
        }

        private static string GetWhamImageFeatureSummaryStatus(
            IReadOnlyList<VideoAssetStatusInfo> required,
            TexMotionSettings settings)
        {
            string runnerKind = settings == null
                ? TexMotionWhamRunnerKinds.OfficialHmr2
                : settings.GetWHAMRunnerKind();
            if (runnerKind == TexMotionWhamRunnerKinds.Archive)
                return GetWhamSetupRoleStatus(required, VideoAssetRole.WhamImageFeatureArchive);
            if (runnerKind == TexMotionWhamRunnerKinds.CompatibleVitPose)
                return GetWhamSetupRoleStatus(required, VideoAssetRole.ViTPoseRuntime,
                    VideoAssetRole.WhamImageFeatureBackbone);
            return GetWhamSetupRoleStatus(required, VideoAssetRole.Hmr2Checkpoint,
                VideoAssetRole.Hmr2Runtime,
                VideoAssetRole.Hmr2Adapter,
                VideoAssetRole.Hmr2BodyModel);
        }

        private static string GetWhamOfficialRunnerStatus(
            IReadOnlyList<VideoAssetStatusInfo> required,
            TexMotionSettings settings)
        {
            string runnerKind = settings == null
                ? TexMotionWhamRunnerKinds.OfficialHmr2
                : settings.GetWHAMRunnerKind();
            if (runnerKind == TexMotionWhamRunnerKinds.CompatibleVitPose)
            {
                return GetWhamSetupRoleStatus(
                    required,
                    VideoAssetRole.WhamCheckpoint,
                    VideoAssetRole.WhamAdapter,
                    VideoAssetRole.PythonRequirements,
                    VideoAssetRole.PythonRuntime,
                    VideoAssetRole.ViTPoseRuntime,
                    VideoAssetRole.WhamImageFeatureBackbone,
                    VideoAssetRole.WhamImageFeatureConfig,
                    VideoAssetRole.WhamImageFeatureModelDefinition);
            }
            if (runnerKind == TexMotionWhamRunnerKinds.Archive)
            {
                return GetWhamSetupRoleStatus(
                    required,
                    VideoAssetRole.WhamCheckpoint,
                    VideoAssetRole.WhamAdapter,
                    VideoAssetRole.PythonRequirements,
                    VideoAssetRole.PythonRuntime,
                    VideoAssetRole.WhamImageFeatureArchive,
                    VideoAssetRole.WhamImageFeatureBackbone);
            }
            return GetWhamSetupRoleStatus(
                required,
                VideoAssetRole.WhamCheckpoint,
                VideoAssetRole.WhamAdapter,
                VideoAssetRole.PythonRequirements,
                VideoAssetRole.PythonRuntime,
                VideoAssetRole.Hmr2Checkpoint,
                VideoAssetRole.Hmr2Runtime,
                VideoAssetRole.Hmr2Adapter,
                VideoAssetRole.Hmr2BodyModel,
                VideoAssetRole.WhamImageFeatureBackbone);
        }

        private static void ClassifyWhamSetupAssets(
            IReadOnlyList<VideoAssetStatusInfo> source,
            List<VideoAssetStatusInfo> manual,
            List<VideoAssetStatusInfo> incompatible)
        {
            if (source == null) return;
            for (int i = 0; i < source.Count; i++)
            {
                VideoAssetStatusInfo asset = source[i];
                if (asset == null) continue;
                if (asset.Status == VideoAssetStatus.Manual)
                {
                    if (manual != null && !manual.Contains(asset)) manual.Add(asset);
                }
                else if (asset.Status == VideoAssetStatus.Incompatible)
                {
                    if (incompatible != null && !incompatible.Contains(asset)) incompatible.Add(asset);
                }
            }
        }

        private static VideoWhamSetupState GetWhamSetupState(
            IReadOnlyList<VideoAssetStatusInfo> required)
        {
            bool hasManual = false;
            bool hasMissing = false;
            if (required == null) return VideoWhamSetupState.Missing;
            for (int i = 0; i < required.Count; i++)
            {
                VideoAssetStatusInfo asset = required[i];
                if (asset == null || asset.Status == VideoAssetStatus.Ready ||
                    asset.Status == VideoAssetStatus.NotRequired)
                    continue;
                if (asset.Status == VideoAssetStatus.Incompatible) return VideoWhamSetupState.Incompatible;
                if (asset.Status == VideoAssetStatus.Manual) hasManual = true;
                else if (asset.Status == VideoAssetStatus.Missing) hasMissing = true;
            }
            if (hasManual) return VideoWhamSetupState.Manual;
            if (hasMissing) return VideoWhamSetupState.Missing;
            return VideoWhamSetupState.Ready;
        }

        /// <summary>
        /// Collapses a set of role rows into the state string consumed by the
        /// compact diagnostic card. Incompatible beats manual, which beats
        /// missing, while an absent role is itself a missing contract row.
        /// </summary>
        private static string GetWhamSetupRoleStatus(
            IReadOnlyList<VideoAssetStatusInfo> assets,
            params VideoAssetRole[] roles)
        {
            if (assets == null || roles == null || roles.Length == 0) return "missing";
            bool found = false;
            bool hasManual = false;
            bool hasMissing = false;
            for (int roleIndex = 0; roleIndex < roles.Length; roleIndex++)
            {
                bool roleFound = false;
                for (int i = 0; i < assets.Count; i++)
                {
                    VideoAssetStatusInfo asset = assets[i];
                    if (asset == null || asset.Role != roles[roleIndex]) continue;
                    found = true;
                    roleFound = true;
                    if (asset.Status == VideoAssetStatus.Incompatible) return "incompatible";
                    if (asset.Status == VideoAssetStatus.Manual) hasManual = true;
                    else if (asset.Status == VideoAssetStatus.Missing ||
                             asset.Status == VideoAssetStatus.NotRequired) hasMissing = true;
                }
                if (!roleFound) hasMissing = true;
            }
            if (!found || hasMissing) return hasManual ? "manual" : "missing";
            if (hasManual) return "manual";
            return "ready";
        }

        private static void AddWhamSetupRequired(
            List<VideoAssetStatusInfo> destination,
            TexMotionSettings settings,
            VideoModelDefinition asset)
        {
            if (destination == null || settings == null || asset == null) return;
            VideoAssetStatusInfo status = GetAssetStatus(settings, asset, true);
            if (status != null) destination.Add(status);
        }

        private static void AddWhamSetupOptional(
            List<VideoAssetStatusInfo> destination,
            TexMotionSettings settings,
            VideoModelDefinition asset)
        {
            if (destination == null || settings == null || asset == null) return;
            VideoAssetStatusInfo status = GetAssetStatus(settings, asset, false);
            if (status != null) destination.Add(status);
        }

        private static void AppendRuntimeRole(
            List<VideoAssetStatusInfo> destination,
            IReadOnlyList<VideoAssetStatusInfo> runtime,
            VideoAssetRole role)
        {
            if (destination == null || runtime == null) return;
            for (int i = 0; i < runtime.Count; i++)
            {
                VideoAssetStatusInfo status = runtime[i];
                if (status != null && status.Role == role)
                {
                    destination.Add(status);
                    return;
                }
            }
        }

        public static VideoPreflightReport Preflight(TexMotionSettings settings, VideoPoseBackend backend)
        {
            return GetPreflightReport(settings, backend);
        }

        /// <summary>
        /// Returns every missing or incompatible required file in deterministic
        /// order.  The paths in this list are suitable for direct UI display.
        /// </summary>
        public static IReadOnlyList<VideoAssetStatusInfo> GetMissingAssetDetails(
            TexMotionSettings settings,
            VideoPoseBackend backend)
        {
            var result = new List<VideoAssetStatusInfo>();
            IReadOnlyList<VideoAssetStatusInfo> statuses = GetRuntimeAssetStatuses(settings, backend);
            for (int i = 0; i < statuses.Count; i++)
            {
                if (statuses[i] != null && statuses[i].IsBlocking) result.Add(statuses[i]);
            }
            return result;
        }

        public static string DescribePreflight(TexMotionSettings settings, VideoPoseBackend backend)
        {
            return GetPreflightReport(settings, backend).Describe();
        }

        private static bool RequiresQualityDependencies(VideoPoseBackend backend)
        {
            return backend == VideoPoseBackend.PyTorch ||
                   backend == VideoPoseBackend.WHAM ||
                   backend == VideoPoseBackend.WHAMMediaPipe ||
                   backend == VideoPoseBackend.HMR2 ||
                   backend == VideoPoseBackend.HybrIK;
        }

        private static VideoDependencyDefinition FindDependency(PythonDependencyProfile profile)
        {
            for (int i = 0; i < DependencyDefinitions.Length; i++)
            {
                if (DependencyDefinitions[i].Profile == profile) return DependencyDefinitions[i];
            }
            return null;
        }

        private static VideoModelDefinition FindMediaPipeDefinition(int complexity)
        {
            string id = complexity <= 0 ? "mediapipe-lite" : complexity == 1 ? "mediapipe-full" : "mediapipe-heavy";
            return Find(id);
        }

        private static VideoAssetStatusInfo CreateDependencyStatus(
            TexMotionSettings settings,
            VideoDependencyDefinition dependency,
            string requirementsPath)
        {
            bool exists = IsReadableFile(requirementsPath, 1L);
            return new VideoAssetStatusInfo
            {
                Id = dependency == null ? "python-requirements" : dependency.Id,
                DisplayName = dependency == null ? "Python Requirements" : dependency.DisplayName,
                Role = VideoAssetRole.PythonRequirements,
                Status = exists ? VideoAssetStatus.Ready : VideoAssetStatus.Missing,
                ExpectedPath = requirementsPath,
                ResolvedPath = exists ? requirementsPath : null,
                Reason = exists
                    ? "Bundled requirements manifest is available for the selected profile."
                    : "Requirements manifest is missing; reinstall the TexMotion package or choose a local project copy.",
                Required = true,
                CanInstall = false
            };
        }

        private static void AddModelStatus(
            List<VideoAssetStatusInfo> statuses,
            TexMotionSettings settings,
            VideoModelDefinition model,
            bool required)
        {
            if (model == null) return;
            string configuredPath = GetConfiguredModelPath(settings, model);
            statuses.Add(InspectAsset(settings, model, configuredPath, required, false));
        }

        private static void AddWhamStatuses(
            List<VideoAssetStatusInfo> statuses,
            TexMotionSettings settings,
            bool required)
        {
            for (int i = 0; i < WhamAssetDefinitions.Length; i++)
            {
                VideoModelDefinition asset = WhamAssetDefinitions[i];
                if (asset.AssetRole == VideoAssetRole.WhamManifest ||
                    asset.AssetRole == VideoAssetRole.WhamAdapter ||
                    asset.AssetRole == VideoAssetRole.WhamImageFeatureArchive ||
                    asset.AssetRole == VideoAssetRole.WhamImageFeatureModelDefinition ||
                    asset.AssetRole == VideoAssetRole.WhamImageFeatureConfig ||
                    asset.RequiredForFullWhamParity)
                {
                    string path = asset.AssetRole == VideoAssetRole.WhamAdapter
                        ? GetRuntimeAdapterPath(settings, asset)
                        : GetConfiguredWhamAssetPath(settings, asset);
                    string runnerKind = settings.GetWHAMRunnerKind();
                    bool selectedRunnerAsset =
                        (asset.AssetRole == VideoAssetRole.WhamImageFeatureArchive &&
                         runnerKind == TexMotionWhamRunnerKinds.Archive) ||
                        (asset.AssetRole == VideoAssetRole.WhamImageFeatureBackbone &&
                         runnerKind != TexMotionWhamRunnerKinds.Archive);
                    // The selected image-feature route is required; unrelated
                    // full-parity rows remain visible as optional stages. This
                    // keeps fallback-capable WHAM setup actionable without
                    // collapsing HMR2 and ViTPose roles into one row.
                    bool rowRequired = required &&
                        (asset.AssetRole == VideoAssetRole.WhamAdapter || selectedRunnerAsset);
                    statuses.Add(InspectAsset(settings, asset, path, rowRequired, false));
                }
            }
        }

        /// <summary>
        /// Adds the selected image-feature route to ordinary backend preflight.
        /// WHAM's temporal checkpoint, HMR2 image-feature runner, ViTPose 2D
        /// detector, and optional archive are separate contracts; a path in
        /// one row must never make another row look executable.
        /// </summary>
        private static void AddWhamImageFeatureRunnerStatuses(
            List<VideoAssetStatusInfo> statuses,
            TexMotionSettings settings)
        {
            if (statuses == null || settings == null) return;
            string runnerKind = settings.GetWHAMRunnerKind();
            if (runnerKind == TexMotionWhamRunnerKinds.Archive)
            {
                // AddWhamStatuses owns the archive row so it can preserve the
                // catalog order and mark it required for the selected route.
                return;
            }

            if (runnerKind == TexMotionWhamRunnerKinds.CompatibleVitPose)
            {
                AddRuntimeAdapterStatus(statuses, settings, FindRuntimeAdapter("vitpose-runtime"), true);
                VideoModelDefinition config = FindWhamAsset("wham-vitpose-runner-config");
                if (config != null &&
                    (!string.IsNullOrWhiteSpace(settings.ViTPoseConfigPath) ||
                     !string.IsNullOrWhiteSpace(settings.WHAMImageFeatureConfigPath)))
                    AddWhamStatusForRole(statuses, settings, config, true);
                VideoModelDefinition definition = FindWhamAsset("wham-vitpose-model-definition");
                if (definition != null &&
                    (!string.IsNullOrWhiteSpace(settings.ViTPoseModelDefinitionPath) ||
                     !string.IsNullOrWhiteSpace(settings.WHAMImageFeatureModelDefinitionPath)))
                    AddWhamStatusForRole(statuses, settings, definition, true);
                return;
            }

            // Official HMR2 is the default route. The bundled bridge is
            // resolved automatically; only the upstream runtime, checkpoint,
            // and neutral body remain user-provided.
            AddModelStatus(statuses, settings, Find("hmr2"), true);
            AddRuntimeAdapterStatus(statuses, settings, FindRuntimeAdapter("hmr2-runtime"), true);
            AddRuntimeAdapterStatus(statuses, settings, FindRuntimeAdapter("hmr2-adapter"), true);
            AddRuntimeAdapterStatus(statuses, settings, FindRuntimeAdapter("hmr2-smpl-body"), true);
        }

        private static void AddWhamStatusForRole(
            List<VideoAssetStatusInfo> statuses,
            TexMotionSettings settings,
            VideoModelDefinition asset,
            bool required)
        {
            if (asset == null) return;
            string path = asset.AssetRole == VideoAssetRole.WhamAdapter
                ? GetRuntimeAdapterPath(settings, asset)
                : GetConfiguredWhamAssetPath(settings, asset);
            statuses.Add(InspectAsset(settings, asset, path, required, false));
        }

        private static void AddRuntimeAdapterStatus(
            List<VideoAssetStatusInfo> statuses,
            TexMotionSettings settings,
            VideoModelDefinition asset,
            bool required)
        {
            if (asset == null) return;
            string configured = asset.AssetRole == VideoAssetRole.WhamBodyModel ||
                                asset.AssetRole == VideoAssetRole.Hmr2BodyModel
                ? GetConfiguredWhamAssetPath(settings, asset)
                : GetRuntimeAdapterPath(settings, asset);
            statuses.Add(InspectAsset(settings, asset, configured, required, false));
        }

        private static void AddConfiguredPathStatus(
            List<VideoAssetStatusInfo> statuses,
            string id,
            string displayName,
            VideoAssetRole role,
            string path,
            bool required,
            bool canInstall,
            string[] extensions)
        {
            string normalized = path;
            try
            {
                if (!string.IsNullOrWhiteSpace(normalized)) normalized = Path.GetFullPath(normalized.Trim());
            }
            catch { }
            bool isDirectoryRuntime = role == VideoAssetRole.ViTPoseRuntime &&
                                      !string.IsNullOrWhiteSpace(normalized) &&
                                      Directory.Exists(normalized);
            bool exists = isDirectoryRuntime || IsReadableFile(normalized, 1025L);
            bool compatible = exists && (isDirectoryRuntime || HasSupportedExtension(normalized, extensions));
            VideoAssetStatus status = !exists
                ? VideoAssetStatus.Missing
                : compatible ? VideoAssetStatus.Ready : VideoAssetStatus.Incompatible;
            statuses.Add(new VideoAssetStatusInfo
            {
                Id = id,
                DisplayName = displayName,
                Role = role,
                Status = status,
                ErrorCode = status == VideoAssetStatus.Ready
                    ? null
                    : role == VideoAssetRole.ViTPoseRuntime
                        ? VideoFeatureErrorCodes.RunnerRuntimeMissing
                        : role == VideoAssetRole.WhamImageFeatureBackbone
                            ? VideoFeatureErrorCodes.CheckpointArchitectureMismatch
                            : VideoFeatureErrorCodes.ModelFactoryMissing,
                NextAction = status == VideoAssetStatus.Ready
                    ? string.Empty
                    : "Choose a compatible asset with Browse, then run Runtime Preflight.",
                ExpectedPath = normalized,
                ResolvedPath = exists ? normalized : null,
                Reason = !exists
                    ? "Select a local file or use the catalog download action."
                    : compatible
                        ? "Local file exists and has a supported checkpoint/adapter extension."
                        : "File extension is incompatible with this runtime role.",
                Required = required,
                CanInstall = canInstall
            });
        }

        private static VideoAssetStatusInfo InspectAsset(
            TexMotionSettings settings,
            VideoModelDefinition asset,
            string configuredPath,
            bool required,
            bool canInstall)
        {
            string expectedPath = configuredPath;
            string resolvedPath = FindExistingAssetPath(settings, asset, configuredPath);
            string reason;
            VideoAssetStatus status;
            if (string.IsNullOrWhiteSpace(resolvedPath))
            {
                status = asset.IsManualProvisioning ? VideoAssetStatus.Manual : VideoAssetStatus.Missing;
                reason = asset.IsManualProvisioning
                    ? "Manual/local provisioning is required; use Browse or the Source link."
                    : "File is missing; use the explicit Install/Download action."
                      + (string.IsNullOrWhiteSpace(expectedPath) ? string.Empty : " Expected: " + expectedPath);
            }
            else if (!ValidateAssetFile(resolvedPath, asset, out reason))
            {
                status = VideoAssetStatus.Incompatible;
            }
            else
            {
                status = VideoAssetStatus.Ready;
                reason = asset.IsManualProvisioning
                    ? "Local user-provided asset is readable and matches the declared role."
                    : "Local catalog asset is readable and matches the declared role.";
            }
            string errorCode = status == VideoAssetStatus.Ready || status == VideoAssetStatus.NotRequired
                ? null
                : GetAssetErrorCode(asset, status, reason);
            return new VideoAssetStatusInfo
            {
                Id = asset.Id,
                DisplayName = asset.DisplayName,
                Role = asset.AssetRole,
                Status = status,
                ErrorCode = errorCode,
                NextAction = GetAssetNextAction(asset, status),
                ExpectedPath = expectedPath,
                ResolvedPath = resolvedPath,
                Reason = reason,
                Required = required,
                CanInstall = canInstall || asset.CanDownload
            };
        }

        private static string GetAssetErrorCode(
            VideoModelDefinition asset,
            VideoAssetStatus status,
            string reason)
        {
            if (asset == null) return VideoFeatureErrorCodes.OfficialRunnerInferenceFailed;
            if (asset.AssetRole == VideoAssetRole.Hmr2Runtime ||
                asset.AssetRole == VideoAssetRole.ViTPoseRuntime ||
                asset.AssetRole == VideoAssetRole.Hmr2Adapter)
                return VideoFeatureErrorCodes.RunnerRuntimeMissing;
            if (asset.AssetRole == VideoAssetRole.Hmr2BodyModel ||
                asset.AssetRole == VideoAssetRole.WhamBodyModel)
                return VideoFeatureErrorCodes.BodyModelMissing;
            if (asset.AssetRole == VideoAssetRole.WhamImageFeatureModelDefinition ||
                asset.AssetRole == VideoAssetRole.WhamImageFeatureConfig)
                return VideoFeatureErrorCodes.ModelFactoryMissing;
            if (asset.AssetRole == VideoAssetRole.Hmr2Checkpoint ||
                asset.AssetRole == VideoAssetRole.WhamCheckpoint ||
                asset.AssetRole == VideoAssetRole.WhamImageFeatureBackbone)
            {
                return status == VideoAssetStatus.Incompatible
                    ? VideoFeatureErrorCodes.CheckpointArchitectureMismatch
                    : VideoFeatureErrorCodes.CheckpointNotExecutable;
            }
            return VideoFeatureErrorCodes.OfficialRunnerInferenceFailed;
        }

        private static string GetAssetNextAction(VideoModelDefinition asset, VideoAssetStatus status)
        {
            if (asset == null || status == VideoAssetStatus.Ready || status == VideoAssetStatus.NotRequired)
                return string.Empty;
            if (status == VideoAssetStatus.Incompatible)
                return "Choose a compatible asset with Browse, then run Runtime Preflight.";
            if (asset.IsManualProvisioning)
                return "Open Guide, provision the upstream asset, choose it with Browse, then run Runtime Preflight.";
            return "Use Download when available, then run Runtime Preflight.";
        }

        private static string FindExistingAssetPath(
            TexMotionSettings settings,
            VideoModelDefinition asset,
            string configuredPath)
        {
            if (settings == null || asset == null) return null;
            var candidates = new List<string>();
            Action<string> add = path =>
            {
                if (string.IsNullOrWhiteSpace(path)) return;
                try { path = Path.GetFullPath(path.Trim()); } catch { }
                for (int i = 0; i < candidates.Count; i++)
                {
                    if (string.Equals(candidates[i], path, StringComparison.OrdinalIgnoreCase)) return;
                }
                candidates.Add(path);
            };

            add(configuredPath);
            bool explicitPath = HasExplicitAssetPath(settings, asset);
            if (!explicitPath) add(GetDestinationPath(settings, asset));
            for (int i = 0; !explicitPath && i < asset.Aliases.Length; i++)
            {
                if (string.IsNullOrWhiteSpace(asset.Aliases[i])) continue;
                if (asset.IsDirectoryAsset)
                    add(Path.Combine(settings.GetEffectiveVideoModelDirectory(), asset.Aliases[i]));
                else
                    // Keep catalog aliases such as smplx/SMPLX_NEUTRAL.npz
                    // relative to the cache; GetVideoModelPath intentionally
                    // strips subdirectories for ordinary user filenames.
                    add(Path.Combine(settings.GetEffectiveVideoModelDirectory(), asset.Aliases[i]));
            }

            for (int i = 0; i < candidates.Count; i++)
            {
                string path = candidates[i];
                try
                {
                    if (asset.IsDirectoryAsset ? Directory.Exists(path) : File.Exists(path)) return path;
                }
                catch { }
            }
            return null;
        }

        private static bool HasExplicitAssetPath(TexMotionSettings settings, VideoModelDefinition asset)
        {
            if (settings == null || asset == null) return false;
            switch (asset.AssetRole)
            {
                case VideoAssetRole.WhamBodyModel: return !string.IsNullOrWhiteSpace(settings.WHAMBodyModelPath);
                case VideoAssetRole.WhamImageFeatureBackbone:
                    return !string.IsNullOrWhiteSpace(settings.ViTPoseModelPath) ||
                           !string.IsNullOrWhiteSpace(settings.WHAMImageFeatureBackbonePath);
                case VideoAssetRole.WhamImageFeatureArchive: return !string.IsNullOrWhiteSpace(settings.WHAMImageFeaturePath);
                case VideoAssetRole.WhamImageFeatureModelDefinition:
                    return !string.IsNullOrWhiteSpace(settings.ViTPoseModelDefinitionPath) ||
                           !string.IsNullOrWhiteSpace(settings.WHAMImageFeatureModelDefinitionPath);
                case VideoAssetRole.WhamImageFeatureConfig:
                    return !string.IsNullOrWhiteSpace(settings.ViTPoseConfigPath) ||
                           !string.IsNullOrWhiteSpace(settings.WHAMImageFeatureConfigPath);
                case VideoAssetRole.WhamCamera: return !string.IsNullOrWhiteSpace(settings.WHAMCameraModelPath);
                case VideoAssetRole.WhamDpvo: return !string.IsNullOrWhiteSpace(settings.WHAMDpvoModelPath);
                case VideoAssetRole.WhamAdapter: return !string.IsNullOrWhiteSpace(settings.VideoQualityAdapterPath);
                case VideoAssetRole.Hmr2Adapter: return !string.IsNullOrWhiteSpace(settings.HMR2AdapterPath);
                case VideoAssetRole.HybrIKAdapter: return !string.IsNullOrWhiteSpace(settings.HybrIKAdapterPath);
                case VideoAssetRole.Hmr2BodyModel: return !string.IsNullOrWhiteSpace(settings.HMR2BodyModelPath);
                case VideoAssetRole.Hmr2Runtime: return !string.IsNullOrWhiteSpace(settings.HMR2RuntimePath);
                case VideoAssetRole.ViTPoseRuntime: return !string.IsNullOrWhiteSpace(settings.ViTPoseRuntimePath);
                case VideoAssetRole.WhamManifest: return !string.IsNullOrWhiteSpace(settings.WHAMAssetManifestPath);
                case VideoAssetRole.WhamCheckpoint:
                case VideoAssetRole.Hmr2Checkpoint:
                    return asset.Kind == VideoModelKind.HMR2
                        ? !string.IsNullOrWhiteSpace(settings.HMR2CheckpointPath) ||
                          !string.IsNullOrWhiteSpace(settings.HMR2ModelPath)
                        : !string.IsNullOrWhiteSpace(settings.WHAMModelPath);
                case VideoAssetRole.InferenceModel:
                    return (asset.Kind == VideoModelKind.RTMPose || asset.Kind == VideoModelKind.DWPose) &&
                           !string.IsNullOrWhiteSpace(settings.RTMPoseModelPath);
                default: return false;
            }
        }

        // Shared by catalog preflight and the download transaction. Keep the
        // validation policy in one place so a file accepted by Download is
        // the same file accepted by Settings/Preflight.
        internal static bool ValidateAssetFile(string path, VideoModelDefinition asset, out string reason)
        {
            reason = string.Empty;
            if (asset.IsDirectoryAsset)
            {
                if (!Directory.Exists(path))
                {
                    reason = "Expected a local runtime directory, but the directory does not exist.";
                    return false;
                }
                bool hasRuntimeSource = asset.AssetRole == VideoAssetRole.ViTPoseRuntime
                    ? Directory.Exists(Path.Combine(path, "mmpose")) ||
                      Directory.Exists(Path.Combine(path, "vitpose")) ||
                      File.Exists(Path.Combine(path, "pyproject.toml")) ||
                      File.Exists(Path.Combine(path, "setup.py"))
                    : Directory.Exists(Path.Combine(path, "hmr2")) ||
                      File.Exists(Path.Combine(path, "pyproject.toml")) ||
                      File.Exists(Path.Combine(path, "setup.py"));
                if (!hasRuntimeSource)
                {
                    reason = asset.AssetRole == VideoAssetRole.ViTPoseRuntime
                        ? "Directory exists but does not contain a ViTPose/MMPose runtime (mmpose/, vitpose/, pyproject.toml, or setup.py)."
                        : "Directory exists but does not contain an HMR2/4D-Humans source checkout (hmr2/, pyproject.toml, or setup.py).";
                    return false;
                }
                if (asset.AssetRole == VideoAssetRole.Hmr2Runtime &&
                    !HasHmr2RuntimeMarker(path))
                {
                    reason = "Runtime directory is present but its package metadata does not identify the HMR2/4D-Humans architecture.";
                    return false;
                }
                return true;
            }

            long length;
            try { length = new FileInfo(path).Length; }
            catch
            {
                reason = "File metadata could not be read.";
                return false;
            }
            long minimum = Math.Max(1L, asset.MinimumBytes);
            if (length < minimum)
            {
                reason = string.Format("File is {0}; expected at least {1}.", FormatBytes(length), FormatBytes(minimum));
                return false;
            }
            if (!HasSupportedExtension(path, asset.SupportedExtensions))
            {
                reason = "File extension is incompatible with the declared runtime role.";
                return false;
            }
            if (!string.IsNullOrWhiteSpace(asset.ExpectedSha256))
            {
                string actual = ComputeSha256(path);
                if (!string.Equals(actual, asset.ExpectedSha256.Trim(), StringComparison.OrdinalIgnoreCase))
                {
                    reason = "SHA-256 checksum does not match the catalog entry.";
                    return false;
                }
            }

            if (asset.AssetRole == VideoAssetRole.WhamAdapter ||
                asset.AssetRole == VideoAssetRole.Hmr2Adapter ||
                asset.AssetRole == VideoAssetRole.HybrIKAdapter)
            {
                string source = ReadSmallText(path);
                bool hasFactory = source != null && source.IndexOf("create_backend", StringComparison.OrdinalIgnoreCase) >= 0;
                bool hasInference = source != null &&
                                    (source.IndexOf("infer_sequence", StringComparison.OrdinalIgnoreCase) >= 0 ||
                                     source.IndexOf("infer_frame", StringComparison.OrdinalIgnoreCase) >= 0 ||
                                     source.IndexOf("predict_sequence", StringComparison.OrdinalIgnoreCase) >= 0 ||
                                     source.IndexOf("predict", StringComparison.OrdinalIgnoreCase) >= 0);
                bool requiresSequence = asset.AssetRole == VideoAssetRole.Hmr2Adapter ||
                                        asset.AssetRole == VideoAssetRole.HybrIKAdapter;
                if (!hasFactory || (requiresSequence && !hasInference))
                {
                    reason = requiresSequence
                        ? "Adapter file is readable but does not expose the required create_backend and inference hooks."
                        : "Adapter file is readable but does not expose the required create_backend entry point.";
                    return false;
                }
            }
            else if (asset.AssetRole == VideoAssetRole.WhamImageFeatureModelDefinition)
            {
                string source = ReadSmallText(path);
                if (source == null || source.IndexOf("create_model", StringComparison.OrdinalIgnoreCase) < 0 ||
                    source.IndexOf("raise RuntimeError", StringComparison.OrdinalIgnoreCase) >= 0)
                {
                    reason = "Model-definition file is a contract/placeholder or does not expose a usable create_model hook.";
                    return false;
                }
            }
            else if (asset.AssetRole == VideoAssetRole.WhamImageFeatureConfig)
            {
                string source = ReadSmallText(path);
                if (source == null || source.IndexOf("runner", StringComparison.OrdinalIgnoreCase) < 0 ||
                    source.IndexOf("feature", StringComparison.OrdinalIgnoreCase) < 0)
                {
                    reason = "ViTPose runner config is readable but does not declare a runner and feature contract.";
                    return false;
                }
            }
            else if (asset.AssetRole == VideoAssetRole.WhamCamera)
            {
                string source = ReadSmallText(path);
                if (source != null &&
                    (source.IndexOf("fx: 0", StringComparison.OrdinalIgnoreCase) >= 0 ||
                     source.IndexOf("fy: 0", StringComparison.OrdinalIgnoreCase) >= 0))
                {
                    reason = "Camera template still has zero intrinsics; replace fx/fy/cx/cy with calibrated values.";
                    return false;
                }
            }
            else if (asset.AssetRole == VideoAssetRole.WhamManifest)
            {
                string source = ReadSmallText(path);
                if (source == null || source.IndexOf("contractVersion", StringComparison.OrdinalIgnoreCase) < 0 ||
                    source.IndexOf("assets", StringComparison.OrdinalIgnoreCase) < 0)
                {
                    reason = "WHAM manifest is not a readable contract manifest.";
                    return false;
                }
            }
            return true;
        }

        private static bool HasHmr2RuntimeMarker(string path)
        {
            try
            {
                if (Directory.Exists(Path.Combine(path, "hmr2")) ||
                    Directory.Exists(Path.Combine(path, "4D-Humans")) ||
                    Directory.Exists(Path.Combine(path, "4dhumans")))
                    return true;

                string[] metadataFiles = { "pyproject.toml", "setup.py", "setup.cfg", "README.md" };
                for (int i = 0; i < metadataFiles.Length; i++)
                {
                    string source = ReadSmallText(Path.Combine(path, metadataFiles[i]));
                    if (source == null) continue;
                    if (source.IndexOf("hmr2", StringComparison.OrdinalIgnoreCase) >= 0 ||
                        source.IndexOf("4d-humans", StringComparison.OrdinalIgnoreCase) >= 0 ||
                        source.IndexOf("4dhumans", StringComparison.OrdinalIgnoreCase) >= 0)
                        return true;
                }
            }
            catch { }
            return false;
        }

        private static string ReadSmallText(string path)
        {
            try
            {
                if (new FileInfo(path).Length > 2L * 1024L * 1024L) return null;
                return File.ReadAllText(path);
            }
            catch { return null; }
        }

        private static bool IsReadableFile(string path, long minimumBytes)
        {
            if (string.IsNullOrWhiteSpace(path)) return false;
            try { return File.Exists(path) && new FileInfo(path).Length >= Math.Max(1L, minimumBytes); }
            catch { return false; }
        }

        private static bool HasSupportedExtension(string path, string[] extensions)
        {
            if (extensions == null || extensions.Length == 0) return true;
            if (string.IsNullOrWhiteSpace(path)) return false;
            string lower = path.ToLowerInvariant();
            for (int i = 0; i < extensions.Length; i++)
            {
                string extension = extensions[i];
                if (string.IsNullOrWhiteSpace(extension)) continue;
                extension = extension.Trim().ToLowerInvariant();
                if (!extension.StartsWith(".")) extension = "." + extension;
                if (lower.EndsWith(extension, StringComparison.Ordinal)) return true;
            }
            return false;
        }

        private static string ComputeSha256(string path)
        {
            try
            {
                using (var stream = File.OpenRead(path))
                using (var sha = SHA256.Create())
                {
                    byte[] digest = sha.ComputeHash(stream);
                    var builder = new StringBuilder(digest.Length * 2);
                    for (int i = 0; i < digest.Length; i++) builder.Append(digest[i].ToString("x2"));
                    return builder.ToString();
                }
            }
            catch { return string.Empty; }
        }

        public static bool IsInstalled(TexMotionSettings settings, VideoModelDefinition model)
        {
            string installedPath = GetInstalledPath(settings, model);
            if (string.IsNullOrEmpty(installedPath)) return false;
            // WHAM is a paired asset: the checkpoint alone cannot satisfy the
            // TexMotion adapter contract, so expose the row as ready only after
            // the adapter has also been copied into the configured cache.
            if (model.Kind == VideoModelKind.WHAM || model.Kind == VideoModelKind.HMR2 || model.Kind == VideoModelKind.HybrIK)
            {
                VideoModelDefinition adapter = model.Kind == VideoModelKind.WHAM
                    ? FindWhamAsset("wham-adapter")
                    : FindRuntimeAdapter(model.Kind == VideoModelKind.HMR2 ? "hmr2-adapter" : "hybrik-adapter");
                string adapterPath = model.Kind == VideoModelKind.WHAM
                    ? GetInstalledWhamAssetPath(settings, adapter)
                    : FindExistingAssetPath(settings, adapter, GetRuntimeAdapterPath(settings, adapter));
                try
                {
                    if (string.IsNullOrWhiteSpace(adapterPath) ||
                        !ValidateAssetFile(adapterPath, adapter, out _)) return false;

                    // A checkpoint is not a runnable HMR2/HybrIK backend by
                    // itself. The model row is green only when every
                    // backend-specific bridge/body/runtime role is also
                    // verified; this prevents a WHAM adapter or unrelated
                    // SMPL file from being treated as a valid HMR2 setup.
                    if (model.Kind == VideoModelKind.HMR2)
                    {
                        return IsStatusReady(settings, FindRuntimeAdapter("hmr2-smpl-body")) &&
                               IsStatusReady(settings, FindRuntimeAdapter("hmr2-adapter")) &&
                               IsStatusReady(settings, FindRuntimeAdapter("hmr2-runtime"));
                    }
                    if (model.Kind == VideoModelKind.HybrIK)
                    {
                        return IsStatusReady(settings, FindWhamAsset("wham-smplx-body")) &&
                               IsStatusReady(settings, FindRuntimeAdapter("hybrik-adapter"));
                    }
                    return true;
                }
                catch { return false; }
            }
            return true;
        }

        private static bool IsStatusReady(TexMotionSettings settings, VideoModelDefinition asset)
        {
            if (asset == null) return false;
            return GetAssetStatus(settings, asset, true)?.IsReady == true;
        }

        /// <summary>Checks one auxiliary asset without applying the checkpoint-pair rule.</summary>
        public static bool IsWhamAssetInstalled(TexMotionSettings settings, VideoModelDefinition asset)
        {
            return !string.IsNullOrEmpty(GetInstalledWhamAssetPath(settings, asset));
        }

        /// <summary>Checks presence plus role-specific compatibility for a WHAM companion.</summary>
        public static bool IsWhamAssetReady(TexMotionSettings settings, VideoModelDefinition asset)
        {
            if (settings == null || asset == null) return false;
            string configured = asset.AssetRole == VideoAssetRole.WhamAdapter ||
                                 asset.AssetRole == VideoAssetRole.Hmr2Adapter ||
                                 asset.AssetRole == VideoAssetRole.HybrIKAdapter ||
                                 asset.AssetRole == VideoAssetRole.Hmr2Runtime
                ? GetRuntimeAdapterPath(settings, asset)
                : GetConfiguredWhamAssetPath(settings, asset);
            return InspectAsset(settings, asset, configured, false, false).IsReady;
        }

        /// <summary>Returns a role-aware status for one catalog row.</summary>
        public static VideoAssetStatusInfo GetAssetStatus(
            TexMotionSettings settings,
            VideoModelDefinition asset,
            bool required = false)
        {
            if (settings == null || asset == null) return null;
            string configured = asset.AssetRole == VideoAssetRole.WhamAdapter ||
                                asset.AssetRole == VideoAssetRole.Hmr2Adapter ||
                                asset.AssetRole == VideoAssetRole.HybrIKAdapter ||
                                asset.AssetRole == VideoAssetRole.Hmr2Runtime
                ? GetRuntimeAdapterPath(settings, asset)
                : asset.Kind == VideoModelKind.WHAM && asset.IsWhamCompanionAsset
                    ? GetConfiguredWhamAssetPath(settings, asset)
                    : asset.AssetRole == VideoAssetRole.WhamBodyModel ||
                      asset.AssetRole == VideoAssetRole.Hmr2BodyModel
                    ? GetConfiguredWhamAssetPath(settings, asset)
                    : GetConfiguredModelPath(settings, asset);
            return InspectAsset(settings, asset, configured, required, false);
        }

        /// <summary>
        /// Returns a local path for an auxiliary WHAM asset, checking the
        /// configured cache first and then the declared aliases. No download
        /// or recursive filesystem search occurs here.
        /// </summary>
        public static string GetInstalledWhamAssetPath(TexMotionSettings settings, VideoModelDefinition asset)
        {
            if (settings == null || asset == null) return null;
            string configured = asset.AssetRole == VideoAssetRole.WhamAdapter
                ? GetRuntimeAdapterPath(settings, asset)
                : GetConfiguredWhamAssetPath(settings, asset);
            string path = FindExistingAssetPath(settings, asset, configured);
            if (string.IsNullOrWhiteSpace(path)) return null;
            string reason;
            return ValidateAssetFile(path, asset, out reason) ? path : null;
        }

        private static bool IsValidWhamAssetFile(string path, long minimumBytes)
        {
            if (string.IsNullOrWhiteSpace(path)) return false;
            try
            {
                return File.Exists(path) && new FileInfo(path).Length >= Math.Max(1L, minimumBytes);
            }
            catch { return false; }
        }

        /// <summary>True only when all six full WHAM stage assets are local.</summary>
        public static bool IsWhamFullParityInstalled(TexMotionSettings settings)
        {
            if (settings == null) return false;
            IReadOnlyList<VideoAssetStatusInfo> statuses = GetRuntimeAssetStatuses(settings, VideoPoseBackend.WHAM);
            for (int i = 0; i < statuses.Count; i++)
            {
                VideoAssetStatusInfo status = statuses[i];
                if (status != null && status.Role != VideoAssetRole.PythonRequirements &&
                    status.Role != VideoAssetRole.PythonRuntime && !status.IsReady) return false;
            }
            return true;
        }

        /// <summary>Returns the WHAM companion IDs that are not present locally.</summary>
        public static IReadOnlyList<string> GetMissingWhamAssetIds(TexMotionSettings settings)
        {
            var missing = new List<string>();
            IReadOnlyList<VideoAssetStatusInfo> statuses = GetRuntimeAssetStatuses(settings, VideoPoseBackend.WHAM);
            for (int i = 0; i < statuses.Count; i++)
            {
                VideoAssetStatusInfo status = statuses[i];
                if (status != null && status.Role != VideoAssetRole.PythonRequirements &&
                    status.Role != VideoAssetRole.PythonRuntime && !status.IsReady)
                    missing.Add(status.Id);
            }
            return missing;
        }

        /// <summary>
        /// Automatically configures or updates camera.yaml based on the selected video file and resolution.
        /// Replaces zero/template intrinsics with estimated intrinsics (standard ~60 deg FOV).
        /// </summary>
        public static bool AutoConfigureWhamCameraFromVideo(
            TexMotionSettings settings,
            string videoPath,
            int videoWidth,
            int videoHeight,
            out string targetFilePath)
        {
            targetFilePath = null;
            if (settings == null) settings = TexMotionSettings.instance;
            if (settings == null) return false;

            try
            {
                string configured = settings.WHAMCameraModelPath;
                if (!string.IsNullOrWhiteSpace(configured))
                {
                    try { configured = Path.GetFullPath(configured.Trim()); } catch { }
                }

                if (string.IsNullOrWhiteSpace(configured))
                {
                    VideoModelDefinition cameraDef = FindWhamAsset("wham-camera");
                    configured = GetWhamAssetDownloadPath(settings, cameraDef);
                }

                if (string.IsNullOrWhiteSpace(configured))
                {
                    configured = Path.Combine(settings.GetEffectiveVideoModelDirectory(), "Video", "camera.yaml");
                }

                string dir = Path.GetDirectoryName(configured);
                if (!string.IsNullOrEmpty(dir) && !Directory.Exists(dir))
                {
                    Directory.CreateDirectory(dir);
                }

                // If file exists, check whether it is locked or manually specified by the user
                if (File.Exists(configured))
                {
                    string existingText = ReadSmallText(configured);
                    if (existingText != null &&
                        (existingText.IndexOf("# manual", StringComparison.OrdinalIgnoreCase) >= 0 ||
                         existingText.IndexOf("# lock", StringComparison.OrdinalIgnoreCase) >= 0))
                    {
                        targetFilePath = configured;
                        return false;
                    }
                }

                int w = videoWidth > 0 ? videoWidth : 1920;
                int h = videoHeight > 0 ? videoHeight : 1080;

                // Standard FOV ~60 degrees: f = w / (2 * tan(30 deg)) = w * 0.8660254
                float focal = w * 0.8660254f;
                float cx = w * 0.5f;
                float cy = h * 0.5f;

                string videoName = string.IsNullOrEmpty(videoPath) ? "unspecified" : Path.GetFileName(videoPath);
                var sb = new StringBuilder();
                sb.AppendLine("# TexMotion WHAM camera calibration auto-generated from video");
                sb.AppendFormat(System.Globalization.CultureInfo.InvariantCulture, "# Source: {0}\n", videoName);
                sb.AppendFormat(System.Globalization.CultureInfo.InvariantCulture, "# Resolution: {0}x{1} (estimated FOV ~60deg)\n", w, h);
                sb.AppendFormat(System.Globalization.CultureInfo.InvariantCulture, "fx: {0:F2}\n", focal);
                sb.AppendFormat(System.Globalization.CultureInfo.InvariantCulture, "fy: {0:F2}\n", focal);
                sb.AppendFormat(System.Globalization.CultureInfo.InvariantCulture, "cx: {0:F2}\n", cx);
                sb.AppendFormat(System.Globalization.CultureInfo.InvariantCulture, "cy: {0:F2}\n", cy);

                File.WriteAllText(configured, sb.ToString(), Encoding.UTF8);
                targetFilePath = configured;

                if (string.IsNullOrWhiteSpace(settings.WHAMCameraModelPath) ||
                    !string.Equals(Path.GetFullPath(settings.WHAMCameraModelPath), configured, StringComparison.OrdinalIgnoreCase))
                {
                    settings.WHAMCameraModelPath = configured;
#if UNITY_EDITOR
                    UnityEditor.EditorUtility.SetDirty(settings);
#endif
                }

                Debug.Log(string.Format(
                    System.Globalization.CultureInfo.InvariantCulture,
                    "[TexMotion] WHAM camera.yaml configured automatically for video '{0}' ({1}x{2}): {3}",
                    videoName, w, h, configured));

                return true;
            }
            catch (Exception ex)
            {
                Debug.LogWarning("[TexMotion] Failed to auto-configure camera.yaml: " + ex.Message);
                return false;
            }
        }


        /// <summary>
        /// Human-readable offline diagnostic suitable for setup panels and
        /// logs. It deliberately never implies that missing assets can be
        /// downloaded automatically because several are separately licensed.
        /// </summary>
        public static string GetWhamAssetContractSummary(TexMotionSettings settings)
        {
            IReadOnlyList<VideoAssetStatusInfo> missing = GetRuntimeAssetStatuses(settings, VideoPoseBackend.WHAM);
            var details = new List<string>();
            for (int i = 0; i < missing.Count; i++)
            {
                VideoAssetStatusInfo status = missing[i];
                if (status.Role == VideoAssetRole.PythonRequirements || status.Role == VideoAssetRole.PythonRuntime) continue;
                if (status.IsReady) continue;
                details.Add((status.IsBlocking ? "Required: " : "Optional/full parity: ") + status);
            }
            if (details.Count == 0)
                return TexMotionLocalization.Tr(TexMotionLocalization.WhamContractReady);
            VideoPreflightReport report = GetPreflightReport(settings, VideoPoseBackend.WHAM);
            return report.IsReady
                ? "Native WHAM path ready; optional/full-parity stages remain unavailable:\n" + string.Join("\n", details)
                : TexMotionLocalization.TrFormat(
                    TexMotionLocalization.WhamContractIncomplete,
                    string.Join("\n", details));
        }

        /// <summary>
        /// Resolves both the configured destination and legacy package/cache
        /// locations so an existing model is not reported as missing after an
        /// upgrade. New downloads always go to the configured destination.
        /// </summary>
        public static string GetInstalledPath(TexMotionSettings settings, VideoModelDefinition model)
        {
            if (settings == null || model == null) return null;
            var candidates = new List<string>();
            var seen = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
            Action<string> add = path =>
            {
                if (string.IsNullOrWhiteSpace(path)) return;
                try { path = Path.GetFullPath(path); } catch { }
                if (seen.Add(path)) candidates.Add(path);
            };

            add(GetConfiguredModelPath(settings, model));
            add(GetDestinationPath(settings, model));
            try
            {
                string appData = Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData);
                string userHome = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);
                add(Path.Combine(appData, "TexMotion", "Models", model.FileName));
                add(Path.Combine(userHome, ".cache", "texmotion", "models", model.FileName));
                for (int i = 0; i < model.Aliases.Length; i++)
                {
                    if (string.IsNullOrWhiteSpace(model.Aliases[i])) continue;
                    add(Path.Combine(settings.GetEffectiveVideoModelDirectory(), model.Aliases[i]));
                }

                string packageModels = Path.GetFullPath(Path.Combine(Application.dataPath, "..", "Editor", "Video", "models"));
                string packageVideo = Path.GetFullPath(Path.Combine(Application.dataPath, "..", "Editor", "Video"));
                add(Path.Combine(packageModels, model.FileName));
                add(Path.Combine(packageVideo, model.FileName));

                string cwd = Directory.GetCurrentDirectory();
                add(Path.Combine(cwd, "Editor", "Video", "models", model.FileName));
                add(Path.Combine(cwd, "Packages", "com.k0ta0uchi.texmotion", "Editor", "Video", "models", model.FileName));
                for (int i = 0; i < model.Aliases.Length; i++)
                {
                    if (string.IsNullOrWhiteSpace(model.Aliases[i])) continue;
                    add(Path.Combine(appData, "TexMotion", "Models", model.Aliases[i]));
                    add(Path.Combine(userHome, ".cache", "texmotion", "models", model.Aliases[i]));
                    add(Path.Combine(packageModels, model.Aliases[i]));
                    add(Path.Combine(cwd, "Editor", "Video", "models", model.Aliases[i]));
                }
            }
            catch { }

            for (int i = 0; i < candidates.Count; i++)
            {
                try
                {
                    if (model.IsDirectoryAsset)
                    {
                        if (Directory.Exists(candidates[i])) return candidates[i];
                    }
                    else if (File.Exists(candidates[i]) &&
                             new FileInfo(candidates[i]).Length >= Math.Max(1024L, model.MinimumBytes))
                    {
                        string reason;
                        if (ValidateAssetFile(candidates[i], model, out reason)) return candidates[i];
                    }
                }
                catch { }
            }
            return null;
        }

        public static long EstimatedTotalBytes
        {
            get
            {
                long total = 0;
                for (int i = 0; i < Definitions.Length; i++)
                {
                    if (Definitions[i].CanDownload) total += Math.Max(0L, Definitions[i].EstimatedBytes);
                }
                total += WhamAdapterEstimatedBytes;
                for (int i = 0; i < WhamAssetDefinitions.Length; i++)
                {
                    VideoModelDefinition asset = WhamAssetDefinitions[i];
                    if (asset.AssetRole == VideoAssetRole.WhamAdapter || asset.AssetRole == VideoAssetRole.WhamManifest)
                        continue;
                    if (asset.CanDownload) total += Math.Max(0L, asset.EstimatedBytes);
                }
                // The manifest is a bundled file and is intentionally tiny,
                // but include it so the Download All estimate is complete.
                VideoModelDefinition manifest = FindWhamAsset("wham-asset-manifest");
                total += manifest == null ? 0L : Math.Max(0L, manifest.EstimatedBytes);
                return total;
            }
        }

        public static string FormatBytes(long bytes)
        {
            if (bytes <= 0) return "0 B";
            string[] units = { "B", "KB", "MB", "GB", "TB" };
            double value = bytes;
            int unit = 0;
            while (value >= 1024.0 && unit < units.Length - 1)
            {
                value /= 1024.0;
                unit++;
            }
            return value.ToString(value >= 10.0 || unit == 0 ? "F0" : "F1") + " " + units[unit];
        }
    }

    /// <summary>Downloads one or all catalogued Video2Motion assets to the configured cache.</summary>
    public static class VideoModelDownloader
    {
        /// <summary>
        /// Refreshes the bundled WHAM bridge in the configured model cache.
        ///
        /// Older TexMotion versions copied a small placeholder bridge which
        /// required TEXMOTION_WHAM_ADAPTER_IMPL. Keep the cache self-healing:
        /// when the package ships a newer bridge (including the native
        /// temporal implementation), replace the cached file before an
        /// extraction starts. A user supplied adapter path is left untouched;
        /// this method only owns the canonical bundled-file destination.
        /// </summary>
        public static bool EnsureBundledWhamAdapter(TexMotionSettings settings)
        {
            if (settings == null) return false;

            string sourcePath = FindBundledWhamAdapterSource();
            string destinationPath = VideoModelCatalog.GetWhamAdapterPath(settings);
            if (string.IsNullOrWhiteSpace(sourcePath) || !File.Exists(sourcePath) ||
                string.IsNullOrWhiteSpace(destinationPath))
            {
                return false;
            }

            try
            {
                string directory = Path.GetDirectoryName(destinationPath);
                if (!string.IsNullOrEmpty(directory)) Directory.CreateDirectory(directory);

                bool needsCopy = true;
                if (File.Exists(destinationPath))
                {
                    FileInfo sourceInfo = new FileInfo(sourcePath);
                    FileInfo destinationInfo = new FileInfo(destinationPath);
                    // Avoid reading the bridge on every IMGUI repaint.  A
                    // package update changes its timestamp or length; retain
                    // the byte comparison only for equal-size/equal-time files
                    // so an in-place edit is still detected.
                    needsCopy = sourceInfo.Length != destinationInfo.Length ||
                                sourceInfo.LastWriteTimeUtc > destinationInfo.LastWriteTimeUtc;
                    if (!needsCopy)
                    {
                        needsCopy = !FilesEqual(sourcePath, destinationPath);
                    }
                }
                if (needsCopy)
                {
                    string tempPath = destinationPath + ".tmp";
                    TryDelete(tempPath);
                    try
                    {
                        File.Copy(sourcePath, tempPath, true);
                        TryDelete(destinationPath);
                        File.Move(tempPath, destinationPath);
                    }
                    catch
                    {
                        TryDelete(tempPath);
                        throw;
                    }
                }

                // Keep automatic WHAM selection useful for users who have not
                // entered an explicit adapter path in Settings yet.
                if ((settings.VideoBackend == VideoPoseBackend.WHAM ||
                     settings.VideoBackend == VideoPoseBackend.WHAMMediaPipe) &&
                    string.IsNullOrWhiteSpace(settings.VideoQualityAdapterPath))
                {
                    settings.VideoQualityAdapterPath = destinationPath;
                }
                return File.Exists(destinationPath);
            }
            catch
            {
                // Extraction retains its normal fallback diagnostics if a
                // locked/read-only cache prevents refreshing the companion.
                return false;
            }
        }

        private static bool FilesEqual(string leftPath, string rightPath)
        {
            try
            {
                var left = File.ReadAllBytes(leftPath);
                var right = File.ReadAllBytes(rightPath);
                if (left.Length != right.Length) return false;
                for (int i = 0; i < left.Length; i++)
                {
                    if (left[i] != right[i]) return false;
                }
                return true;
            }
            catch
            {
                return false;
            }
        }

        public static async Task DownloadAsync(
            TexMotionSettings settings,
            string modelId,
            IProgress<DownloadProgress> progress,
            CancellationToken cancellationToken = default)
        {
            var model = VideoModelCatalog.Find(modelId);
            if (model == null) throw new ArgumentException("Unknown Video2Motion model: " + modelId, nameof(modelId));

            if (!model.CanDownload)
            {
                string expected = VideoModelCatalog.GetConfiguredModelPath(settings, model);
                throw new InvalidOperationException(TexMotionLocalization.TrFormat(
                    TexMotionLocalization.ManualWhamAsset,
                    model.DisplayName,
                    expected));
            }

            if (model.IsWhamCompanionAsset)
            {
                await DownloadWhamCompanionAsync(settings, model, progress, cancellationToken, 0, 1);
                return;
            }

            bool includesWhamAdapter = model.Kind == VideoModelKind.WHAM;
            int totalFiles = includesWhamAdapter ? 2 : 1;
            await DownloadModelAsync(settings, model, 0, totalFiles, progress, cancellationToken);
            if (includesWhamAdapter)
            {
                InstallBundledWhamAdapter(settings, 1, totalFiles, progress, cancellationToken);
            }
            progress?.Report(new DownloadProgress
            {
                OverallProgress = 1f,
                FileProgress = 1f,
                CompletedFiles = totalFiles,
                TotalFiles = totalFiles,
                CurrentFileName = includesWhamAdapter
                    ? model.FileName + " + " + VideoModelCatalog.WhamAdapterFileName
                    : model.FileName,
                StatusText = includesWhamAdapter
                    ? TexMotionLocalization.TrFormat(TexMotionLocalization.ReadyModelWithAdapter, model.DisplayName)
                    : TexMotionLocalization.TrFormat(TexMotionLocalization.ReadyModel, model.DisplayName)
            });
        }

        private static async Task DownloadWhamCompanionAsync(
            TexMotionSettings settings,
            VideoModelDefinition asset,
            IProgress<DownloadProgress> progress,
            CancellationToken cancellationToken,
            int fileIndex,
            int totalFiles)
        {
            if (settings == null) throw new ArgumentNullException(nameof(settings));
            if (asset == null) throw new ArgumentNullException(nameof(asset));
            cancellationToken.ThrowIfCancellationRequested();
            totalFiles = Math.Max(1, totalFiles);

            if (asset.AssetRole == VideoAssetRole.WhamAdapter)
            {
                InstallBundledWhamAdapter(settings, fileIndex, totalFiles, progress, cancellationToken);
                return;
            }

            if (asset.AssetRole == VideoAssetRole.WhamManifest)
            {
                InstallBundledWhamManifest(settings, fileIndex, totalFiles, progress, cancellationToken);
                return;
            }

            if (asset.AssetRole == VideoAssetRole.WhamImageFeatureModelDefinition ||
                asset.AssetRole == VideoAssetRole.WhamImageFeatureConfig)
            {
                InstallBundledWhamRunnerAsset(settings, asset, fileIndex, totalFiles, progress, cancellationToken);
                return;
            }

            // Camera calibration is not a universal downloadable artifact. A
            // zeroed starter template is useful for discovery and keeps the
            // operation offline; the UI and metadata explicitly tell users to
            // replace its intrinsics before using a world-motion runner.
            if (asset.AssetRole == VideoAssetRole.WhamCamera)
            {
                if (VideoModelCatalog.IsWhamAssetReady(settings, asset))
                {
                    ReportSkippedWhamCompanion(asset, fileIndex, totalFiles, progress);
                    return;
                }
                InstallBundledWhamCameraTemplate(settings, asset, fileIndex, totalFiles, progress, cancellationToken);
                return;
            }

            string expected = VideoModelCatalog.GetWhamAssetPath(settings, asset);
            if (VideoModelCatalog.IsWhamAssetReady(settings, asset))
            {
                ReportSkippedWhamCompanion(asset, fileIndex, totalFiles, progress);
                return;
            }

            if (asset.CanDownload)
            {
                await DownloadModelAsync(
                    settings,
                    asset,
                    fileIndex,
                    totalFiles,
                    progress,
                    cancellationToken,
                    VideoModelCatalog.GetWhamAssetDownloadPath(settings, asset));
                return;
            }
            throw new InvalidOperationException(TexMotionLocalization.TrFormat(
                TexMotionLocalization.ManualWhamAsset,
                asset.DisplayName,
                expected));
        }

        private static void ReportSkippedWhamCompanion(
            VideoModelDefinition asset,
            int fileIndex,
            int totalFiles,
            IProgress<DownloadProgress> progress)
        {
            progress?.Report(new DownloadProgress
            {
                OverallProgress = (float)(fileIndex + 1) / Math.Max(1, totalFiles),
                FileProgress = 1f,
                CompletedFiles = fileIndex + 1,
                TotalFiles = totalFiles,
                CurrentFileName = asset.FileName,
                StatusText = TexMotionLocalization.TrFormat(
                    TexMotionLocalization.SkippedExistingFile,
                    asset.DisplayName,
                    fileIndex + 1,
                    totalFiles)
            });
        }

        public static async Task DownloadAllAsync(
            TexMotionSettings settings,
            IProgress<DownloadProgress> progress,
            CancellationToken cancellationToken = default)
        {
            if (settings == null) throw new ArgumentNullException(nameof(settings));
            // Include only public/downloadable model files plus ViTPose/DPVO
            // and the camera starter template. HMR2 weights, licensed
            // SMPL/SMPL-X data, and third-party runtimes remain manual and are
            // deliberately excluded so Download All is always actionable.
            int companionDownloads = 0;
            for (int i = 0; i < VideoModelCatalog.WhamAssets.Count; i++)
            {
                VideoModelDefinition asset = VideoModelCatalog.WhamAssets[i];
                if (asset.AssetRole == VideoAssetRole.WhamAdapter ||
                    asset.AssetRole == VideoAssetRole.WhamManifest ||
                    asset.AssetRole == VideoAssetRole.WhamImageFeatureModelDefinition ||
                    asset.AssetRole == VideoAssetRole.WhamImageFeatureConfig)
                    continue;
                if (asset.CanDownload) companionDownloads++;
            }
            int bundledCompanions = 4; // adapter + manifest + runner definition/config templates
            int downloadableModels = 0;
            for (int i = 0; i < VideoModelCatalog.All.Count; i++)
            {
                if (VideoModelCatalog.All[i].CanDownload) downloadableModels++;
            }
            int total = downloadableModels + bundledCompanions + companionDownloads;
            int completed = 0;
            for (int i = 0; i < VideoModelCatalog.All.Count; i++)
            {
                cancellationToken.ThrowIfCancellationRequested();
                VideoModelDefinition model = VideoModelCatalog.All[i];
                if (!model.CanDownload) continue;
                await DownloadModelAsync(settings, model, completed, total, progress, cancellationToken);
                completed++;
                if (model.Kind == VideoModelKind.WHAM)
                {
                    InstallBundledWhamAdapter(settings, completed, total, progress, cancellationToken);
                    completed++;
                }
            }

            VideoModelDefinition runnerDefinition = VideoModelCatalog.FindWhamAsset("wham-vitpose-model-definition");
            InstallBundledWhamRunnerAsset(settings, runnerDefinition, completed, total, progress, cancellationToken);
            completed++;
            VideoModelDefinition runnerConfig = VideoModelCatalog.FindWhamAsset("wham-vitpose-runner-config");
            InstallBundledWhamRunnerAsset(settings, runnerConfig, completed, total, progress, cancellationToken);
            completed++;

            for (int i = 0; i < VideoModelCatalog.WhamAssets.Count; i++)
            {
                cancellationToken.ThrowIfCancellationRequested();
                VideoModelDefinition asset = VideoModelCatalog.WhamAssets[i];
                if (asset.AssetRole == VideoAssetRole.WhamAdapter || asset.AssetRole == VideoAssetRole.WhamManifest ||
                    asset.AssetRole == VideoAssetRole.WhamImageFeatureModelDefinition || asset.AssetRole == VideoAssetRole.WhamImageFeatureConfig ||
                    !asset.CanDownload)
                    continue;
                await DownloadWhamCompanionAsync(settings, asset, progress, cancellationToken, completed, total);
                completed++;
            }

            InstallBundledWhamManifest(settings, completed, total, progress, cancellationToken);
            completed++;

            VideoPreflightReport preflight = VideoModelCatalog.GetPreflightReport(settings, settings.VideoBackend);
            string completionStatus = preflight.IsReady
                ? TexMotionLocalization.Tr(TexMotionLocalization.AllVideoModelsReady)
                : "Catalog downloads complete; Runtime Preflight still requires " +
                  preflight.BlockingAssets.Count + " asset(s).";
            progress?.Report(new DownloadProgress
            {
                OverallProgress = 1f,
                FileProgress = 1f,
                CompletedFiles = completed,
                TotalFiles = total,
                CurrentFileName = TexMotionLocalization.Tr(TexMotionLocalization.Complete),
                StatusText = completionStatus
            });
        }

        private static void InstallBundledWhamAdapter(
            TexMotionSettings settings,
            int fileIndex,
            int totalFiles,
            IProgress<DownloadProgress> progress,
            CancellationToken cancellationToken)
        {
            if (settings == null) throw new ArgumentNullException(nameof(settings));
            cancellationToken.ThrowIfCancellationRequested();

            string sourcePath = FindBundledWhamAdapterSource();
            if (string.IsNullOrEmpty(sourcePath) || !File.Exists(sourcePath))
            {
                throw new FileNotFoundException(
                    TexMotionLocalization.Tr(TexMotionLocalization.BundledWhamAdapterMissing),
                    sourcePath);
            }

            string destinationPath = VideoModelCatalog.GetWhamAdapterPath(settings);
            if (string.IsNullOrEmpty(destinationPath))
                throw new InvalidOperationException(TexMotionLocalization.Tr(TexMotionLocalization.WhamAdapterDestinationEmpty));
            string directory = Path.GetDirectoryName(destinationPath);
            if (!string.IsNullOrEmpty(directory)) Directory.CreateDirectory(directory);

            progress?.Report(new DownloadProgress
            {
                OverallProgress = (float)fileIndex / Math.Max(1, totalFiles),
                FileProgress = 0f,
                CompletedFiles = fileIndex,
                TotalFiles = totalFiles,
                CurrentFileName = VideoModelCatalog.WhamAdapterFileName,
                StatusText = TexMotionLocalization.Tr(TexMotionLocalization.InstallingWhamAdapter)
            });

            string tempPath = destinationPath + ".tmp";
            TryDelete(tempPath);
            try
            {
                File.Copy(sourcePath, tempPath, true);
                cancellationToken.ThrowIfCancellationRequested();
                TryDelete(destinationPath);
                File.Move(tempPath, destinationPath);
                if ((settings.VideoBackend == VideoPoseBackend.WHAM ||
                     settings.VideoBackend == VideoPoseBackend.WHAMMediaPipe) &&
                    string.IsNullOrWhiteSpace(settings.VideoQualityAdapterPath))
                {
                    settings.VideoQualityAdapterPath = destinationPath;
                }
                progress?.Report(new DownloadProgress
                {
                    OverallProgress = (float)(fileIndex + 1) / Math.Max(1, totalFiles),
                    FileProgress = 1f,
                    CompletedFiles = fileIndex + 1,
                    TotalFiles = totalFiles,
                    CurrentFileName = VideoModelCatalog.WhamAdapterFileName,
                    StatusText = TexMotionLocalization.Tr(TexMotionLocalization.WhamAdapterReady)
                });
            }
            catch
            {
                TryDelete(tempPath);
                throw;
            }
        }

        private static void InstallBundledWhamManifest(
            TexMotionSettings settings,
            int fileIndex,
            int totalFiles,
            IProgress<DownloadProgress> progress,
            CancellationToken cancellationToken)
        {
            if (settings == null) throw new ArgumentNullException(nameof(settings));
            cancellationToken.ThrowIfCancellationRequested();

            string sourcePath = FindBundledWhamManifestSource();
            if (string.IsNullOrEmpty(sourcePath) || !File.Exists(sourcePath))
            {
                throw new FileNotFoundException(
                    TexMotionLocalization.Tr(TexMotionLocalization.BundledWhamManifestMissing),
                    sourcePath);
            }

            string destinationPath = VideoModelCatalog.GetWhamAssetManifestPath(settings);
            if (string.IsNullOrEmpty(destinationPath))
                throw new InvalidOperationException(TexMotionLocalization.Tr(TexMotionLocalization.WhamManifestDestinationEmpty));
            string directory = Path.GetDirectoryName(destinationPath);
            if (!string.IsNullOrEmpty(directory)) Directory.CreateDirectory(directory);

            progress?.Report(new DownloadProgress
            {
                OverallProgress = (float)fileIndex / Math.Max(1, totalFiles),
                FileProgress = 0f,
                CompletedFiles = fileIndex,
                TotalFiles = totalFiles,
                CurrentFileName = VideoModelCatalog.WhamAssetManifestFileName,
                StatusText = TexMotionLocalization.Tr(TexMotionLocalization.InstallingWhamManifest)
            });

            string tempPath = destinationPath + ".tmp";
            TryDelete(tempPath);
            try
            {
                File.Copy(sourcePath, tempPath, true);
                cancellationToken.ThrowIfCancellationRequested();
                TryDelete(destinationPath);
                File.Move(tempPath, destinationPath);
                progress?.Report(new DownloadProgress
                {
                    OverallProgress = (float)(fileIndex + 1) / Math.Max(1, totalFiles),
                    FileProgress = 1f,
                    CompletedFiles = fileIndex + 1,
                    TotalFiles = totalFiles,
                    CurrentFileName = VideoModelCatalog.WhamAssetManifestFileName,
                    StatusText = TexMotionLocalization.Tr(TexMotionLocalization.WhamManifestReady)
                });
            }
            catch
            {
                TryDelete(tempPath);
                throw;
            }
        }

        private static void InstallBundledWhamRunnerAsset(
            TexMotionSettings settings,
            VideoModelDefinition asset,
            int fileIndex,
            int totalFiles,
            IProgress<DownloadProgress> progress,
            CancellationToken cancellationToken)
        {
            if (settings == null) throw new ArgumentNullException(nameof(settings));
            if (asset == null) throw new ArgumentNullException(nameof(asset));
            cancellationToken.ThrowIfCancellationRequested();

            string sourcePath = FindBundledWhamRunnerAssetSource(asset);
            if (string.IsNullOrEmpty(sourcePath) || !File.Exists(sourcePath))
                throw new FileNotFoundException("Bundled WHAM runner asset is missing.", sourcePath);

            string destinationPath = VideoModelCatalog.GetWhamAssetDownloadPath(settings, asset);
            if (string.IsNullOrEmpty(destinationPath))
                throw new InvalidOperationException("WHAM runner asset destination is empty.");
            string directory = Path.GetDirectoryName(destinationPath);
            if (!string.IsNullOrEmpty(directory)) Directory.CreateDirectory(directory);

            progress?.Report(new DownloadProgress
            {
                OverallProgress = (float)fileIndex / Math.Max(1, totalFiles),
                FileProgress = 0f,
                CompletedFiles = fileIndex,
                TotalFiles = totalFiles,
                CurrentFileName = asset.FileName,
                StatusText = "Installing " + asset.DisplayName
            });

            string tempPath = destinationPath + ".tmp";
            TryDelete(tempPath);
            try
            {
                File.Copy(sourcePath, tempPath, true);
                cancellationToken.ThrowIfCancellationRequested();
                TryDelete(destinationPath);
                File.Move(tempPath, destinationPath);
                progress?.Report(new DownloadProgress
                {
                    OverallProgress = (float)(fileIndex + 1) / Math.Max(1, totalFiles),
                    FileProgress = 1f,
                    CompletedFiles = fileIndex + 1,
                    TotalFiles = totalFiles,
                    CurrentFileName = asset.FileName,
                    StatusText = asset.DisplayName + " ready"
                });
            }
            catch
            {
                TryDelete(tempPath);
                throw;
            }
        }

        private static void InstallBundledWhamCameraTemplate(
            TexMotionSettings settings,
            VideoModelDefinition asset,
            int fileIndex,
            int totalFiles,
            IProgress<DownloadProgress> progress,
            CancellationToken cancellationToken)
        {
            if (settings == null) throw new ArgumentNullException(nameof(settings));
            cancellationToken.ThrowIfCancellationRequested();

            string sourcePath = FindBundledWhamCameraTemplateSource();
            if (string.IsNullOrEmpty(sourcePath) || !File.Exists(sourcePath))
            {
                throw new FileNotFoundException(
                    TexMotionLocalization.Tr(TexMotionLocalization.BundledWhamCameraMissing),
                    sourcePath);
            }

            VideoModelDefinition camera = asset ?? VideoModelCatalog.FindWhamAsset("wham-camera");
            string destinationPath = VideoModelCatalog.GetWhamAssetDownloadPath(settings, camera);
            if (string.IsNullOrEmpty(destinationPath))
                throw new InvalidOperationException(TexMotionLocalization.Tr(TexMotionLocalization.WhamCameraDestinationEmpty));
            string directory = Path.GetDirectoryName(destinationPath);
            if (!string.IsNullOrEmpty(directory)) Directory.CreateDirectory(directory);

            progress?.Report(new DownloadProgress
            {
                OverallProgress = (float)fileIndex / Math.Max(1, totalFiles),
                FileProgress = 0f,
                CompletedFiles = fileIndex,
                TotalFiles = totalFiles,
                CurrentFileName = camera == null ? "camera.yaml" : camera.FileName,
                StatusText = TexMotionLocalization.Tr(TexMotionLocalization.InstallingWhamCamera)
            });

            string tempPath = destinationPath + ".tmp";
            TryDelete(tempPath);
            try
            {
                File.Copy(sourcePath, tempPath, true);
                cancellationToken.ThrowIfCancellationRequested();
                TryDelete(destinationPath);
                File.Move(tempPath, destinationPath);
                progress?.Report(new DownloadProgress
                {
                    OverallProgress = (float)(fileIndex + 1) / Math.Max(1, totalFiles),
                    FileProgress = 1f,
                    CompletedFiles = fileIndex + 1,
                    TotalFiles = totalFiles,
                    CurrentFileName = camera == null ? "camera.yaml" : camera.FileName,
                    StatusText = TexMotionLocalization.Tr(TexMotionLocalization.WhamCameraReady)
                });
            }
            catch
            {
                TryDelete(tempPath);
                throw;
            }
        }

        private static string FindBundledWhamAdapterSource()
        {
            var candidates = new List<string>();
            try
            {
                string fileName = VideoModelCatalog.WhamAdapterFileName;
                string cwd = Directory.GetCurrentDirectory();
                candidates.Add(Path.Combine(cwd, "Editor", "Video", "pose_pipeline", "adapters", fileName));
                candidates.Add(Path.Combine(cwd, "Packages", "com.k0ta0uchi.texmotion", "Editor", "Video", "pose_pipeline", "adapters", fileName));
                if (!string.IsNullOrEmpty(Application.dataPath))
                {
                    candidates.Add(Path.Combine(Application.dataPath, "Editor", "Video", "pose_pipeline", "adapters", fileName));
                    candidates.Add(Path.Combine(Application.dataPath, "TexMotion", "Editor", "Video", "pose_pipeline", "adapters", fileName));
                    candidates.Add(Path.GetFullPath(Path.Combine(Application.dataPath, "..", "Editor", "Video", "pose_pipeline", "adapters", fileName)));
                    candidates.Add(Path.GetFullPath(Path.Combine(Application.dataPath, "..", "Packages", "com.k0ta0uchi.texmotion", "Editor", "Video", "pose_pipeline", "adapters", fileName)));
                }
            }
            catch { }

            for (int i = 0; i < candidates.Count; i++)
            {
                try
                {
                    if (File.Exists(candidates[i])) return candidates[i];
                }
                catch { }
            }
            return null;
        }

        private static string FindBundledWhamManifestSource()
        {
            var candidates = new List<string>();
            try
            {
                string fileName = VideoModelCatalog.WhamAssetManifestFileName;
                string cwd = Directory.GetCurrentDirectory();
                candidates.Add(Path.Combine(cwd, "Editor", "Video", "models", fileName));
                candidates.Add(Path.Combine(cwd, "Packages", "com.k0ta0uchi.texmotion", "Editor", "Video", "models", fileName));
                if (!string.IsNullOrEmpty(Application.dataPath))
                {
                    candidates.Add(Path.Combine(Application.dataPath, "Editor", "Video", "models", fileName));
                    candidates.Add(Path.Combine(Application.dataPath, "TexMotion", "Editor", "Video", "models", fileName));
                    candidates.Add(Path.GetFullPath(Path.Combine(Application.dataPath, "..", "Editor", "Video", "models", fileName)));
                    candidates.Add(Path.GetFullPath(Path.Combine(Application.dataPath, "..", "Packages", "com.k0ta0uchi.texmotion", "Editor", "Video", "models", fileName)));
                }
            }
            catch { }

            for (int i = 0; i < candidates.Count; i++)
            {
                try
                {
                    if (File.Exists(candidates[i])) return candidates[i];
                }
                catch { }
            }
            return null;
        }

        private static string FindBundledWhamCameraTemplateSource()
        {
            var candidates = new List<string>();
            try
            {
                string fileName = VideoModelCatalog.WhamCameraTemplateFileName;
                string cwd = Directory.GetCurrentDirectory();
                candidates.Add(Path.Combine(cwd, "Editor", "Video", "models", fileName));
                candidates.Add(Path.Combine(cwd, "Packages", "com.k0ta0uchi.texmotion", "Editor", "Video", "models", fileName));
                if (!string.IsNullOrEmpty(Application.dataPath))
                {
                    candidates.Add(Path.Combine(Application.dataPath, "Editor", "Video", "models", fileName));
                    candidates.Add(Path.Combine(Application.dataPath, "TexMotion", "Editor", "Video", "models", fileName));
                    candidates.Add(Path.GetFullPath(Path.Combine(Application.dataPath, "..", "Editor", "Video", "models", fileName)));
                    candidates.Add(Path.GetFullPath(Path.Combine(Application.dataPath, "..", "Packages", "com.k0ta0uchi.texmotion", "Editor", "Video", "models", fileName)));
                }
            }
            catch { }

            for (int i = 0; i < candidates.Count; i++)
            {
                try
                {
                    if (File.Exists(candidates[i])) return candidates[i];
                }
                catch { }
            }
            return null;
        }

        private static string FindBundledWhamRunnerAssetSource(VideoModelDefinition asset)
        {
            if (asset == null) return null;
            var candidates = new List<string>();
            try
            {
                string fileName = asset.FileName;
                string cwd = Directory.GetCurrentDirectory();
                candidates.Add(Path.Combine(cwd, "Editor", "Video", "models", fileName));
                candidates.Add(Path.Combine(cwd, "Packages", "com.k0ta0uchi.texmotion", "Editor", "Video", "models", fileName));
                if (!string.IsNullOrEmpty(Application.dataPath))
                {
                    candidates.Add(Path.Combine(Application.dataPath, "Editor", "Video", "models", fileName));
                    candidates.Add(Path.Combine(Application.dataPath, "TexMotion", "Editor", "Video", "models", fileName));
                    candidates.Add(Path.GetFullPath(Path.Combine(Application.dataPath, "..", "Editor", "Video", "models", fileName)));
                    candidates.Add(Path.GetFullPath(Path.Combine(Application.dataPath, "..", "Packages", "com.k0ta0uchi.texmotion", "Editor", "Video", "models", fileName)));
                }
            }
            catch { }

            for (int i = 0; i < candidates.Count; i++)
            {
                try
                {
                    if (File.Exists(candidates[i])) return candidates[i];
                }
                catch { }
            }
            return null;
        }

        private static async Task DownloadModelAsync(
            TexMotionSettings settings,
            VideoModelDefinition model,
            int fileIndex,
            int totalFiles,
            IProgress<DownloadProgress> progress,
            CancellationToken cancellationToken,
            string destinationOverride = null)
        {
            if (settings == null) throw new ArgumentNullException(nameof(settings));
            if (model == null) throw new ArgumentNullException(nameof(model));
            if (!model.CanDownload)
            {
                string expected = VideoModelCatalog.GetConfiguredModelPath(settings, model);
                throw new InvalidOperationException(TexMotionLocalization.TrFormat(
                    TexMotionLocalization.ManualWhamAsset,
                    model.DisplayName,
                    expected));
            }
            string destinationPath = string.IsNullOrWhiteSpace(destinationOverride)
                ? VideoModelCatalog.GetDestinationPath(settings, model)
                : destinationOverride;
            if (string.IsNullOrEmpty(destinationPath))
                throw new InvalidOperationException(TexMotionLocalization.Tr(TexMotionLocalization.VideoModelDestinationEmpty));

            // Only skip when the file is already present at the currently
            // configured destination.  GetInstalledPath also recognizes
            // legacy/package caches for compatibility, but a download request
            // after changing the destination must materialize a copy there.
            string existingReason;
            if (IsValidFile(destinationPath, model.MinimumBytes) &&
                VideoModelCatalog.ValidateAssetFile(destinationPath, model, out existingReason))
            {
                progress?.Report(new DownloadProgress
                {
                    OverallProgress = (float)(fileIndex + 1) / Math.Max(1, totalFiles),
                    FileProgress = 1f,
                    CompletedFiles = fileIndex + 1,
                    TotalFiles = totalFiles,
                    CurrentFileName = model.FileName,
                    StatusText = TexMotionLocalization.TrFormat(
                        TexMotionLocalization.SkippedExistingFile,
                        model.DisplayName,
                        fileIndex + 1,
                        totalFiles)
                });
                return;
            }

            string directory = Path.GetDirectoryName(destinationPath);
            if (!string.IsNullOrEmpty(directory)) Directory.CreateDirectory(directory);
            string tempPath = destinationPath + ".tmp";
            TryDelete(tempPath);

            progress?.Report(new DownloadProgress
            {
                OverallProgress = (float)fileIndex / Math.Max(1, totalFiles),
                FileProgress = 0f,
                CompletedFiles = fileIndex,
                TotalFiles = totalFiles,
                CurrentFileName = model.FileName,
                StatusText = TexMotionLocalization.TrFormat(
                    TexMotionLocalization.DownloadingModel,
                    model.DisplayName)
            });

            try
            {
                using (var request = UnityWebRequest.Get(model.RemoteUrl))
                {
                    request.downloadHandler = new DownloadHandlerFile(tempPath);
                    var operation = request.SendWebRequest();
                    while (!operation.isDone)
                    {
                        cancellationToken.ThrowIfCancellationRequested();
                        float fileProgress = Mathf.Clamp01(request.downloadProgress);
                        progress?.Report(new DownloadProgress
                        {
                            OverallProgress = ((float)fileIndex + fileProgress) / Math.Max(1, totalFiles),
                            FileProgress = fileProgress,
                            CompletedFiles = fileIndex,
                            TotalFiles = totalFiles,
                            CurrentFileName = model.FileName,
                            StatusText = TexMotionLocalization.TrFormat(
                                TexMotionLocalization.DownloadingModelProgress,
                                model.DisplayName,
                                fileProgress * 100f)
                        });
                        // UnityWebRequest is a Unity API; keep polling on the
                        // editor synchronization context instead of resuming
                        // on an arbitrary worker thread.
                        await Task.Delay(100, cancellationToken);
                    }

                    cancellationToken.ThrowIfCancellationRequested();
                    if (request.result != UnityWebRequest.Result.Success)
                        throw new InvalidOperationException(TexMotionLocalization.TrFormat(
                            TexMotionLocalization.DownloadFailed,
                            model.DisplayName,
                            request.error,
                            model.RemoteUrl));
                }

                long downloadedSize = new FileInfo(tempPath).Length;
                cancellationToken.ThrowIfCancellationRequested();
                if (downloadedSize < Math.Max(1024L, model.MinimumBytes))
                    throw new InvalidDataException(TexMotionLocalization.TrFormat(
                        TexMotionLocalization.DownloadedFileTooSmall,
                        downloadedSize));

                string validationReason;
                if (!VideoModelCatalog.ValidateAssetFile(tempPath, model, out validationReason))
                    throw new InvalidDataException(model.DisplayName + " is incompatible: " + validationReason);

                TryDelete(destinationPath);
                File.Move(tempPath, destinationPath);
                progress?.Report(new DownloadProgress
                {
                    OverallProgress = (float)(fileIndex + 1) / Math.Max(1, totalFiles),
                    FileProgress = 1f,
                    CompletedFiles = fileIndex + 1,
                    TotalFiles = totalFiles,
                    CurrentFileName = model.FileName,
                    StatusText = TexMotionLocalization.TrFormat(
                        TexMotionLocalization.ReadyModel,
                        model.DisplayName)
                });
            }
            catch (OperationCanceledException)
            {
                TryDelete(tempPath);
                throw;
            }
            catch
            {
                TryDelete(tempPath);
                throw;
            }
        }

        private static void TryDelete(string path)
        {
            if (string.IsNullOrEmpty(path)) return;
            try
            {
                if (File.Exists(path)) File.Delete(path);
            }
            catch { }
        }

        private static bool IsValidFile(string path, long minimumBytes)
        {
            if (string.IsNullOrWhiteSpace(path)) return false;
            try
            {
                return File.Exists(path) && new FileInfo(path).Length >= Math.Max(1024L, minimumBytes);
            }
            catch { return false; }
        }
    }
}
