using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading;
using System.Threading.Tasks;
using TexMotion.Runtime.Motion;
using TexMotion.Runtime.Native;
using UnityEditor;
using UnityEngine;
using Debug = UnityEngine.Debug;

namespace TexMotion.Editor.Video
{
    /// <summary>
    /// Represents structured progress reported from the video pose extraction job.
    /// </summary>
    [Serializable]
    public struct VideoJobProgress
    {
        /// <summary>
        /// Normalized overall progress from 0.0 to 1.0.
        /// </summary>
        public float Progress;

        /// <summary>
        /// Current processing stage name: "init", "extracting", "foot_locking", "solving_ik", "smoothing", "exporting", "completed", "error".
        /// </summary>
        public string Stage;

        /// <summary>
        /// 1-based index of the frame currently being processed.
        /// </summary>
        public int CurrentFrame;

        /// <summary>
        /// Total number of sampled frames in this extraction job.
        /// </summary>
        public int TotalFrames;

        /// <summary>
        /// Path to the written motion JSON file when completed.
        /// </summary>
        public string OutputPath;

        /// <summary>
        /// Error message if the extraction failed.
        /// </summary>
        public string ErrorMessage;

        /// <summary>
        /// Raw output line from subprocess stdout for diagnostics.
        /// </summary>
        public string RawMessage;

        /// <summary>
        /// Human-readable status message for UI display.
        /// </summary>
        private string _message;
        public string Message
        {
            get
            {
                if (!string.IsNullOrEmpty(_message)) return _message;
                if (IsError)
                {
                    return TexMotionLocalization.TrFormat(
                        TexMotionLocalization.VideoError,
                        ErrorMessage ?? string.Empty);
                }
                if (IsCompleted) return TexMotionLocalization.Tr(TexMotionLocalization.ExtractionComplete);
                string stageDisplay = FormatStageLabel(Stage);
                if (TotalFrames > 0) return $"{stageDisplay} ({Percentage:F0}%) - Frame {CurrentFrame}/{TotalFrames}";
                return $"{stageDisplay} ({Percentage:F0}%)";
            }
            set => _message = value;
        }

        private static string FormatStageLabel(string stage)
        {
            if (string.IsNullOrEmpty(stage)) return "processing";
            switch (stage.ToLowerInvariant())
            {
                case "init": return TexMotionLocalization.TrLiteral("初期化中");
                case "wham_sequence": return TexMotionLocalization.TrLiteral("WHAM時系列推論 (GPU)");
                case "fusion_observing": return TexMotionLocalization.TrLiteral("空間観測フュージョン");
                case "fusion_optimizing": return TexMotionLocalization.TrLiteral("姿勢最適化");
                case "extracting": return TexMotionLocalization.TrLiteral("モーション抽出・描画");
                case "foot_locking": return TexMotionLocalization.TrLiteral("接地ロック検出");
                case "solving_ik": return TexMotionLocalization.TrLiteral("ヒューマノイドIK計算");
                case "smoothing": return TexMotionLocalization.TrLiteral("モーション平滑化");
                case "exporting": return TexMotionLocalization.TrLiteral("アニメーション書き出し");
                default: return stage;
            }
        }

        public bool IsCompleted => string.Equals(Stage, "completed", StringComparison.OrdinalIgnoreCase) || Progress >= 1.0f;
        public bool IsError => string.Equals(Stage, "error", StringComparison.OrdinalIgnoreCase) || !string.IsNullOrEmpty(ErrorMessage);
        public float Percentage => Mathf.Clamp01(Progress) * 100f;

        public VideoJobProgress(
            float progress,
            string stage,
            int currentFrame = 0,
            int totalFrames = 0,
            string outputPath = null,
            string errorMessage = null,
            string rawMessage = null,
            string message = null)
        {
            Progress = progress;
            Stage = stage ?? string.Empty;
            CurrentFrame = currentFrame;
            TotalFrames = totalFrames;
            OutputPath = outputPath;
            ErrorMessage = errorMessage;
            RawMessage = rawMessage;
            _message = message;
        }

        public override string ToString()
        {
            return Message;
        }
    }

    /// <summary>
    /// Supported pose estimation backends for video-to-motion extraction.
    /// </summary>
    public enum VideoPoseBackend
    {
        Auto,
        MediaPipe,
        RTMPose,
        /// <summary>Optional TorchScript or adapter-backed quality inference.</summary>
        PyTorch,
        /// <summary>WHAM temporal body-model adapter (requires the quality profile).</summary>
        WHAM,
        /// <summary>4D-Humans/HMR2 body-model adapter (requires the quality profile).</summary>
        HMR2,
        /// <summary>HybrIK body-model adapter (requires the quality profile).</summary>
        HybrIK,
        /// <summary>
        /// Explicit WHAM-primary, MediaPipe-assisted temporal fusion.  This is
        /// appended to preserve the serialized values of existing settings.
        /// </summary>
        WHAMMediaPipe
    }

    /// <summary>
    /// Optional temporal fusion mode.  It is intentionally independent from
    /// VideoPoseBackend so existing MediaPipe/RTMPose choices remain valid.
    /// </summary>
    public enum VideoFusionMode
    {
        Off,
        WhamMediaPipe
    }

    /// <summary>
    /// Visualization mode for pose overlay video rendering.
    /// </summary>
    public enum VideoOverlayMode
    {
        /// <summary>Dual comparison: 2D detector guide (subtle) + 3D pose projection (vibrant neon). Recommended.</summary>
        Dual = 0,
        /// <summary>WHAM 3D pose hypothesis projection only.</summary>
        Pose3D = 1,
        /// <summary>Direct 2D detector keypoints tracking only.</summary>
        Tracking2D = 2
    }

    /// <summary>
    /// Configuration options for invoking the video pose extraction pipeline.
    /// </summary>
    [Serializable]
    public class VideoJobOptions
    {
        private const string DefaultPyTorchDevice = "auto";

        /// <summary>
        /// Pose estimation backend to use (Auto, MediaPipe, RTMPose, or an optional
        /// PyTorch/WHAM/WHAMMediaPipe/HMR2/HybrIK quality adapter).
        /// </summary>
        public VideoPoseBackend Backend = VideoPoseBackend.Auto;

        /// <summary>
        /// Explicit fusion mode passed to the extractor.  WHAM selections use
        /// WHAM + MediaPipe by default; legacy backends remain unchanged.
        /// </summary>
        public VideoFusionMode FusionMode = VideoFusionMode.Off;

        /// <summary>Optional TorchScript checkpoint for the quality backend.</summary>
        public string PyTorchModelPath = null;

        /// <summary>Optional adapter module or .py file for WHAM/WHAMMediaPipe/HMR2/HybrIK.</summary>
        public string PyTorchAdapterModule = null;

        /// <summary>Quality backend device: auto, cpu, or cuda.</summary>
        public string PyTorchDevice = DefaultPyTorchDevice;

        /// <summary>Optional maximum sequence length for a temporal quality backend.</summary>
        public int MaxSequenceFrames = 0;

        /// <summary>Directory containing downloaded RTMPose/DWPose and MediaPipe assets.</summary>
        public string VideoModelDirectory = null;

        /// <summary>Optional official 4D-Humans/HMR2 source checkout or runtime path.</summary>
        public string HMR2RuntimePath = null;

        /// <summary>
        /// Explicit official HMR2 image-feature checkpoint.  The current
        /// extractor receives this through its legacy image-feature checkpoint
        /// option for the official WHAM/HMR2 route; keeping the value here
        /// lets Settings and diagnostics preserve an explicit user path
        /// without confusing it with the ViTPose 2D detector.
        /// </summary>
        public string HMR2CheckpointPath = null;

        /// <summary>Optional licensed neutral SMPL body model used by official HMR2.</summary>
        public string HMR2BodyModelPath = null;

        /// <summary>Optional explicit WHAM offline asset manifest.</summary>
        public string WhamAssetManifestPath = null;

        /// <summary>Optional SMPL/SMPL-X body model used by the WHAM decoder.</summary>
        public string WhamBodyModelPath = null;

        /// <summary>Optional ViTPose image-feature backbone checkpoint.</summary>
        public string WhamImageFeatureBackbonePath = null;

        /// <summary>Optional ViTPose model-definition module or Python file.</summary>
        public string WhamImageFeatureModelDefinitionPath = null;

        /// <summary>Optional ViTPose runner/model config path.</summary>
        public string WhamImageFeatureConfigPath = null;

        /// <summary>Logical WHAM image-feature runner selection.</summary>
        public string WHAMRunnerKind = "official_hmr2";

        /// <summary>Optional compatible ViTPose/MMPose runtime path.</summary>
        public string ViTPoseRuntimePath = null;

        /// <summary>Optional output key and expected feature dimension for diagnostics.</summary>
        public string ImageFeatureOutputKey = null;
        public int ImageFeatureDim = 0;

        /// <summary>Fallback policy retained in the editor-side job contract.</summary>
        public bool AllowMediaPipeFallback = true;
        public bool RequireImageFeatures = false;

        /// <summary>Optional precomputed WHAM image-feature archive.</summary>
        public string WhamImageFeaturePath = null;

        /// <summary>Optional camera calibration or exported camera-motion archive.</summary>
        public string WhamCameraModelPath = null;

        /// <summary>Optional DPVO checkpoint or exported camera-motion archive.</summary>
        public string WhamDpvoModelPath = null;

        /// <summary>
        /// Optional per-video cache for WHAM image features and camera poses.
        /// When empty, the Python extractor places a hidden cache beside the
        /// output motion JSON so each video gets its own aligned archives.
        /// </summary>
        public string WhamPreprocessDirectory = null;

        /// <summary>Optional explicit RTMPose/DWPose ONNX file selected in the Settings catalog.</summary>
        public string RTMPoseModelPath = null;

        /// <summary>
        /// Path to the source video file (MP4, MOV, AVI, etc.).
        /// </summary>
        public string VideoPath = string.Empty;

        /// <summary>
        /// Output path where the resulting motion JSON should be saved.
        /// If empty, a temporary path will be generated automatically.
        /// </summary>
        public string OutputPath = string.Empty;

        /// <summary>
        /// Optional path to write the skeleton overlay video.
        /// If empty, the extractor generates a default _overlay.mp4 alongside the output JSON.
        /// </summary>
        public string OverlayVideoPath = string.Empty;

        /// <summary>
        /// Visualization mode for pose overlay video rendering (Dual, Pose3D, Tracking2D).
        /// </summary>
        public VideoOverlayMode OverlayMode = VideoOverlayMode.Dual;

        /// <summary>
        /// Optional explicit path to the Python executable. If null, auto-detection is used.
        /// </summary>
        public string PythonExecutable = null;

        /// <summary>
        /// Optional explicit path to video_pose_extractor.py. If null, auto-detection is used.
        /// </summary>
        public string ScriptPath = null;

        /// <summary>
        /// Target sampling frame rate (default 30.0 fps). Set to 0 to preserve source video frame rate.
        /// </summary>
        public float TargetFps = 30.0f;

        /// <summary>
        /// Trim start time in seconds (0.0 = beginning of video).
        /// </summary>
        public float TrimStart = 0.0f;

        /// <summary>
        /// Property alias for TrimStart.
        /// </summary>
        public float TrimStartTime
        {
            get => TrimStart;
            set => TrimStart = value;
        }

        /// <summary>
        /// Trim end time in seconds (0.0 = entire video duration).
        /// </summary>
        public float TrimEnd = 0.0f;

        /// <summary>
        /// Property alias for TrimEnd.
        /// </summary>
        public float TrimEndTime
        {
            get => TrimEnd;
            set => TrimEnd = value;
        }

        /// <summary>
        /// If true, zeroes horizontal X and Z root motion while preserving natural vertical bounce.
        /// </summary>
        public bool InPlace = false;

        /// <summary>
        /// If true, applies Savitzky-Golay / continuous quaternion temporal smoothing.
        /// </summary>
        public bool TemporalSmoothing = false;

        /// <summary>
        /// If true, applies floor contact locking and 2-bone IK knee adjustments to eliminate foot sliding.
        /// </summary>
        public bool FootLocking = true;

        /// <summary>
        /// MediaPipe model complexity: 0 (Fast), 1 (Balanced, default), 2 (Accurate).
        /// </summary>
        public int ModelComplexity = 1;

        /// <summary>
        /// Minimum detection confidence threshold in [0.0, 1.0].
        /// </summary>
        public float MinDetectionConfidence = 0.5f;

        /// <summary>
        /// Minimum tracking confidence threshold in [0.0, 1.0].
        /// </summary>
        public float MinTrackingConfidence = 0.5f;

        /// <summary>
        /// If true or if video is empty, generates synthetic humanoid motion for testing and fallback.
        /// </summary>
        public bool SyntheticFallback = false;

        /// <summary>
        /// Property alias for SyntheticFallback.
        /// </summary>
        public bool Synthetic
        {
            get => SyntheticFallback;
            set => SyntheticFallback = value;
        }

        /// <summary>
        /// Builds the command-line argument string for video_pose_extractor.py.
        /// </summary>
        public string BuildCommandLineArguments(string scriptPath, string effectiveOutputPath)
        {
            var sb = new StringBuilder();
            sb.Append($"\"{scriptPath}\"");

            if (SyntheticFallback || string.IsNullOrEmpty(VideoPath))
            {
                sb.Append(" --synthetic");
            }
            else
            {
                sb.Append($" --video \"{VideoPath}\"");
            }

            sb.Append($" --output \"{effectiveOutputPath}\"");

            if (!string.IsNullOrEmpty(OverlayVideoPath))
            {
                sb.Append($" --overlay-video \"{OverlayVideoPath}\"");
            }

            string overlayModeArg = OverlayMode switch
            {
                VideoOverlayMode.Pose3D => "3d",
                VideoOverlayMode.Tracking2D => "2d",
                _ => "dual"
            };
            sb.Append($" --overlay-mode {overlayModeArg}");

            if (TargetFps > 0f)
            {
                sb.Append($" --fps {TargetFps.ToString("F2", CultureInfo.InvariantCulture)}");
            }

            if (TrimStart > 0f)
            {
                sb.Append($" --trim-start {TrimStart.ToString("F2", CultureInfo.InvariantCulture)}");
            }

            if (TrimEnd > 0f)
            {
                sb.Append($" --trim-end {TrimEnd.ToString("F2", CultureInfo.InvariantCulture)}");
            }

            if (InPlace)
            {
                sb.Append(" --in-place");
            }

            if (TemporalSmoothing)
            {
                sb.Append(" --smooth");
            }

            if (!FootLocking)
            {
                sb.Append(" --no-foot-lock");
            }

            sb.Append($" --model-complexity {Mathf.Clamp(ModelComplexity, 0, 2)}");
            sb.Append($" --min-detection-confidence {MinDetectionConfidence.ToString("F2", CultureInfo.InvariantCulture)}");
            sb.Append($" --min-tracking-confidence {MinTrackingConfidence.ToString("F2", CultureInfo.InvariantCulture)}");

            string backendArg = Backend switch
            {
                VideoPoseBackend.MediaPipe => "mediapipe",
                VideoPoseBackend.RTMPose => "rtmpose",
                VideoPoseBackend.PyTorch => "pytorch",
                VideoPoseBackend.WHAM => "wham",
                VideoPoseBackend.HMR2 => "hmr2",
                VideoPoseBackend.HybrIK => "hybrik",
                VideoPoseBackend.WHAMMediaPipe => "wham",
                _ => "auto"
            };
            sb.Append($" --backend {backendArg}");

            bool whamBackend = Backend == VideoPoseBackend.WHAM || Backend == VideoPoseBackend.WHAMMediaPipe;
            string fusionModeArg = (FusionMode == VideoFusionMode.WhamMediaPipe || Backend == VideoPoseBackend.WHAMMediaPipe)
                ? "wham_mediapipe"
                : "off";
            if (whamBackend && FusionMode == VideoFusionMode.Off)
            {
                fusionModeArg = "wham_mediapipe";
            }
            sb.Append($" --fusion-mode {fusionModeArg}");

            if (Backend == VideoPoseBackend.PyTorch || whamBackend ||
                Backend == VideoPoseBackend.HMR2 || Backend == VideoPoseBackend.HybrIK)
            {
                if (!string.IsNullOrWhiteSpace(PyTorchModelPath))
                    sb.Append($" --pytorch-model \"{PyTorchModelPath}\"");
                if (!string.IsNullOrWhiteSpace(PyTorchAdapterModule))
                    sb.Append($" --pytorch-adapter \"{PyTorchAdapterModule}\"");
                string device = string.IsNullOrWhiteSpace(PyTorchDevice) ? DefaultPyTorchDevice : PyTorchDevice.Trim().ToLowerInvariant();
                if (device == "auto" || device == "cpu" || device == "cuda")
                    sb.Append($" --pytorch-device {device}");
                if (MaxSequenceFrames > 0)
                    sb.Append($" --max-sequence-frames {MaxSequenceFrames}");
            }

            AppendQuotedOption(sb, "--video-model-dir", VideoModelDirectory);
            AppendQuotedOption(sb, "--hmr2-runtime", HMR2RuntimePath);
            AppendQuotedOption(sb, "--hmr2-body-model", HMR2BodyModelPath);
            AppendQuotedOption(sb, "--rtmpose-model", RTMPoseModelPath);
            AppendQuotedOption(sb, "--wham-asset-manifest", WhamAssetManifestPath);
            AppendQuotedOption(sb, "--wham-body-model", WhamBodyModelPath);
            // The legacy Python CLI exposes one image-feature checkpoint
            // option. For the official HMR2 route, use the separately
            // resolved HMR2 checkpoint there; compatible ViTPose/archive
            // routes retain their own configured path. The C# settings and
            // result metadata still keep these sources explicitly separate.
            string imageFeatureCheckpoint = WhamImageFeatureBackbonePath;
            if (whamBackend &&
                string.Equals(WHAMRunnerKind, TexMotionWhamRunnerKinds.OfficialHmr2, StringComparison.OrdinalIgnoreCase) &&
                !string.IsNullOrWhiteSpace(HMR2CheckpointPath))
            {
                imageFeatureCheckpoint = HMR2CheckpointPath;
            }
            AppendQuotedOption(sb, "--wham-image-feature-backbone", imageFeatureCheckpoint);
            AppendQuotedOption(sb, "--wham-image-feature-model-definition", WhamImageFeatureModelDefinitionPath);
            AppendQuotedOption(sb, "--wham-image-feature-config", WhamImageFeatureConfigPath);
            AppendQuotedOption(sb, "--wham-image-features", WhamImageFeaturePath);
            AppendQuotedOption(sb, "--wham-camera", WhamCameraModelPath);
            AppendQuotedOption(sb, "--wham-dpvo", WhamDpvoModelPath);
            AppendQuotedOption(sb, "--wham-preprocess-dir", WhamPreprocessDirectory);

            return sb.ToString();
        }

        private static void AppendQuotedOption(StringBuilder arguments, string option, string value)
        {
            if (!string.IsNullOrWhiteSpace(value))
                arguments.Append($" {option} \"{value}\"");
        }
    }

    /// <summary>
    /// Options for the Settings one-click ViTPose feature archive exporter.
    /// The exporter writes an aligned ``vitpose_features.npz`` file and does
    /// not run the full pose-to-motion conversion.
    /// </summary>
    [Serializable]
    public class ViTPoseFeatureExportOptions
    {
        public string VideoPath = string.Empty;
        public string OutputPath = string.Empty;
        public string PythonExecutable = null;
        public string ScriptPath = null;
        public float TargetFps = 30.0f;
        public float TrimStart = 0.0f;
        public float TrimEnd = 0.0f;
        public string CheckpointPath = null;
        public string ModelDefinitionPath = null;
        public string ConfigPath = null;
        public string Device = "auto";
        public int ChunkSize = 32;

        public string BuildCommandLineArguments(string scriptPath, string effectiveOutputPath)
        {
            var sb = new StringBuilder();
            sb.Append('"').Append(scriptPath).Append('"');
            AppendQuotedOption(sb, "--video", VideoPath);
            AppendQuotedOption(sb, "--output", effectiveOutputPath);
            sb.Append(" --export-vitpose-features");
            if (TargetFps > 0f)
                sb.Append(" --fps ").Append(TargetFps.ToString("F2", CultureInfo.InvariantCulture));
            if (TrimStart > 0f)
                sb.Append(" --trim-start ").Append(TrimStart.ToString("F2", CultureInfo.InvariantCulture));
            if (TrimEnd > 0f)
                sb.Append(" --trim-end ").Append(TrimEnd.ToString("F2", CultureInfo.InvariantCulture));
            AppendQuotedOption(sb, "--wham-image-feature-backbone", CheckpointPath);
            AppendQuotedOption(sb, "--wham-image-feature-model-definition", ModelDefinitionPath);
            AppendQuotedOption(sb, "--wham-image-feature-config", ConfigPath);
            string device = string.IsNullOrWhiteSpace(Device) ? "auto" : Device.Trim().ToLowerInvariant();
            if (device == "auto" || device == "cpu" || device == "cuda")
                sb.Append(" --pytorch-device ").Append(device);
            if (ChunkSize > 0)
                sb.Append(" --vitpose-feature-chunk-size ").Append(ChunkSize.ToString(CultureInfo.InvariantCulture));
            return sb.ToString();
        }

        private static void AppendQuotedOption(StringBuilder arguments, string option, string value)
        {
            if (!string.IsNullOrWhiteSpace(value))
                arguments.Append(' ').Append(option).Append(" \"").Append(value).Append('"');
        }
    }

    /// <summary>
    /// Alias for VideoJobOptions for naming consistency across editor modules.
    /// </summary>
    [Serializable]
    public class VideoExtractionOptions : VideoJobOptions
    {
    }

    /// <summary>
    /// Diagnostics and environment information about the Python runtime.
    /// </summary>
    [Serializable]
    public class PythonRuntimeInfo
    {
        public bool IsAvailable;
        public string ExecutablePath;
        public string Version;
        public string EnvironmentPath;
        public bool IsVirtualEnv;
        public bool HasMediaPipe;
        public bool HasOpenCV;
        public bool HasNumPy;
        public bool HasSciPy;
        public bool HasOnnxRuntime;
        public bool HasTorch;
        public bool HasCuda;
        public string ErrorMessage;

        /// <summary>
        /// True if Python is available and the extractor's mandatory core
        /// dependencies (MediaPipe, OpenCV, NumPy) are installed. SciPy and
        /// ONNX Runtime remain optional for the MediaPipe-only path.
        /// </summary>
        public bool IsFullyConfigured => IsAvailable && HasMediaPipe && HasOpenCV && HasNumPy;

        public string GetSummary()
        {
            if (!IsAvailable)
            {
                return TexMotionLocalization.TrFormat(
                    TexMotionLocalization.PythonUnavailableSummary,
                    ErrorMessage ?? TexMotionLocalization.Tr(TexMotionLocalization.NotDetected));
            }

            var sb = new StringBuilder();
            sb.Append(TexMotionLocalization.TrFormat(
                TexMotionLocalization.PythonSummary,
                Version ?? TexMotionLocalization.Tr(TexMotionLocalization.Unknown),
                ExecutablePath));
            if (IsVirtualEnv) sb.Append(TexMotionLocalization.Tr(TexMotionLocalization.VirtualEnvironmentSuffix));

            var missing = new List<string>();
            if (!HasMediaPipe) missing.Add("mediapipe");
            if (!HasOpenCV) missing.Add("opencv-python");
            if (!HasNumPy) missing.Add("numpy");
            if (!HasSciPy) missing.Add("scipy");
            if (!HasOnnxRuntime) missing.Add("onnxruntime (optional for RTMPose)");

            if (missing.Count == 0)
            {
                sb.Append(TexMotionLocalization.Tr(TexMotionLocalization.DependenciesReadySummary));
            }
            else
            {
                sb.Append(TexMotionLocalization.TrFormat(
                    TexMotionLocalization.MissingPackagesSummary,
                    string.Join(", ", missing)));
            }

            return sb.ToString();
        }
    }

    /// <summary>
    /// Exception thrown when the video pose extraction subprocess encounters a failure.
    /// </summary>
    public class VideoExtractionException : Exception
    {
        public int ExitCode { get; }
        public string Stderr { get; }
        public string Stage { get; }

        public VideoExtractionException(string message, int exitCode = -1, string stderr = null, string stage = null)
            : base(message)
        {
            ExitCode = exitCode;
            Stderr = stderr;
            Stage = stage;
        }

        public VideoExtractionException(string message, Exception innerException, int exitCode = -1, string stderr = null, string stage = null)
            : base(message, innerException)
        {
            ExitCode = exitCode;
            Stderr = stderr;
            Stage = stage;
        }
    }

    /// <summary>
    /// Service runner for executing video pose extraction via video_pose_extractor.py subprocess.
    /// Provides async invocation, line-by-line JSON streaming progress reporting, cancellation,
    /// diagnostics stderr capture, Python runtime probing, and output JSON deserialization into VideoMotionData.
    /// </summary>
    public static class VideoMotionJobRunner
    {
        /// <summary>
        /// Adds an actionable explanation for Windows process termination
        /// codes that are emitted by native ML runtimes before Python can
        /// write a traceback.  In particular, -1073741819 is 0xC0000005,
        /// an access violation commonly raised by TFLite/XNNPACK, ONNX
        /// Runtime, or a native video codec DLL.
        /// </summary>
        private static string DescribeNativeExitCode(int exitCode)
        {
            uint unsignedCode = unchecked((uint)exitCode);
            switch (unsignedCode)
            {
                case 0xC0000005u:
                    return "Native process access violation (0xC0000005). " +
                           "The extraction process was terminated inside a native ML/video DLL; " +
                           "the last stage above identifies the likely backend. " +
                           "For WHAM, keep one MediaPipe/TFLite graph per process and verify the local model assets.";
                case 0xC0000409u:
                    return "Native process stack-buffer overrun (0xC0000409). " +
                           "Check the selected ML backend and its native runtime installation.";
                case 0xC000001Du:
                    return "Native process illegal instruction (0xC000001D). " +
                           "The selected runtime may require CPU instructions unavailable on this machine.";
                default:
                    return null;
            }
        }

        #region Public Async Extraction API

        /// <summary>
        /// Executes video pose extraction asynchronously using the specified options.
        /// Reports progress via IProgress, action callbacks, and captures diagnostics stderr.
        /// </summary>
        public static async Task<VideoMotionData> RunExtractionAsync(
            VideoJobOptions options,
            IProgress<VideoJobProgress> progress = null,
            CancellationToken cancellationToken = default,
            Action<string> onStderrLine = null)
        {
            if (options == null)
            {
                throw new ArgumentNullException(nameof(options));
            }

            // 1. Locate video_pose_extractor.py script
            string scriptPath = options.ScriptPath;
            if (string.IsNullOrEmpty(scriptPath) || !File.Exists(scriptPath))
            {
                scriptPath = FindScriptPath();
            }

            if (string.IsNullOrEmpty(scriptPath) || !File.Exists(scriptPath))
            {
                throw new FileNotFoundException(
                    TexMotionLocalization.Tr(TexMotionLocalization.ExtractorScriptMissing));
            }

            // 2. Resolve Python executable
            string pythonExe = options.PythonExecutable;
            if (string.IsNullOrEmpty(pythonExe))
            {
                try
                {
                    pythonExe = TexMotionSettings.instance?.GetEffectiveVideoPythonExecutablePath();
                }
                catch {}
            }

            if (string.IsNullOrEmpty(pythonExe) || !File.Exists(pythonExe))
            {
                var runtime = DetectPythonRuntime();
                if (!runtime.IsAvailable)
                {
                    throw new InvalidOperationException(TexMotionLocalization.TrFormat(
                        TexMotionLocalization.NoWorkingPythonRuntime,
                        runtime.ErrorMessage));
                }
                pythonExe = runtime.ExecutablePath;
            }

            // Probe the exact interpreter that will launch the extractor.  The
            // script imports NumPy before it can emit a JSON diagnostic, so
            // allowing the process to start with a broken environment only
            // produces an opaque ModuleNotFoundError dialog in Unity.
            PythonRuntimeInfo selectedRuntime = ProbePythonExecutable(pythonExe);
            bool syntheticRun = options.SyntheticFallback || string.IsNullOrEmpty(options.VideoPath);
            var missingDependencies = new List<string>();
            if (!selectedRuntime.IsAvailable)
            {
                throw new InvalidOperationException(TexMotionLocalization.TrFormat(
                    TexMotionLocalization.SelectedPythonRuntimeFailed,
                    selectedRuntime.ErrorMessage ?? pythonExe));
            }
            if (!selectedRuntime.HasNumPy) missingDependencies.Add("numpy");
            if (!syntheticRun && !selectedRuntime.HasOpenCV) missingDependencies.Add("opencv-python");
            if (missingDependencies.Count > 0)
            {
                throw new InvalidOperationException(TexMotionLocalization.TrFormat(
                    TexMotionLocalization.MissingVideoDependencies,
                    string.Join(", ", missingDependencies),
                    selectedRuntime.ExecutablePath));
            }

            // 3. Resolve effective output path
            string outputPath = options.OutputPath;
            if (string.IsNullOrEmpty(outputPath))
            {
                string tempDir = Path.Combine(Path.GetTempPath(), "TexMotion", "VideoJobs");
                if (!Directory.Exists(tempDir))
                {
                    Directory.CreateDirectory(tempDir);
                }
                outputPath = Path.Combine(tempDir, $"motion_{Guid.NewGuid():N}.json");
            }

            string outDir = Path.GetDirectoryName(outputPath);
            if (!string.IsNullOrEmpty(outDir) && !Directory.Exists(outDir))
            {
                Directory.CreateDirectory(outDir);
            }

            // 4. Build command line arguments
            string arguments = options.BuildCommandLineArguments(scriptPath, outputPath);

            // 5. Capture main thread synchronization context for safe callback dispatch
            var syncContext = SynchronizationContext.Current;

            // 6. Diagnostics capture
            var stderrBuilder = new StringBuilder();
            var stdoutLines = new List<string>();
            string lastStage = "init";
            string stageErrorMessage = null;

            var psi = new ProcessStartInfo
            {
                FileName = pythonExe,
                Arguments = arguments,
                UseShellExecute = false,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                CreateNoWindow = true,
                StandardOutputEncoding = Encoding.UTF8,
                StandardErrorEncoding = Encoding.UTF8,
                WorkingDirectory = Path.GetDirectoryName(scriptPath)
            };

            psi.EnvironmentVariables["PYTHONUNBUFFERED"] = "1";
            psi.EnvironmentVariables["PYTHONIOENCODING"] = "utf-8";

            using var process = new Process { StartInfo = psi, EnableRaisingEvents = true };
            var tcs = new TaskCompletionSource<int>();

            process.Exited += (s, e) =>
            {
                tcs.TrySetResult(process.ExitCode);
            };

            process.OutputDataReceived += (s, e) =>
            {
                if (e.Data == null) return;
                string line = e.Data;
                lock (stdoutLines)
                {
                    stdoutLines.Add(line);
                }

                if (TryParseProgressJson(line, out var p))
                {
                    lastStage = p.Stage;
                    if (p.IsError)
                    {
                        stageErrorMessage = p.ErrorMessage;
                    }
                    DispatchProgress(syncContext, progress, null, null, p);
                }
            };

            process.ErrorDataReceived += (s, e) =>
            {
                if (e.Data == null) return;
                string errLine = e.Data;
                lock (stderrBuilder)
                {
                    stderrBuilder.AppendLine(errLine);
                }
                onStderrLine?.Invoke(errLine);
            };

            using (cancellationToken.Register(() =>
            {
                KillProcessTree(process);
                tcs.TrySetCanceled(cancellationToken);
            }))
            {
                try
                {
                    if (!process.Start())
                    {
                        throw new InvalidOperationException(TexMotionLocalization.TrFormat(
                            TexMotionLocalization.FailedToStartPythonProcess,
                            pythonExe));
                    }

                    process.BeginOutputReadLine();
                    process.BeginErrorReadLine();

                    // Initial progress notification
                    DispatchProgress(syncContext, progress, null, null,
                        new VideoJobProgress(0.01f, "init", 0, 100));

                    if (process.HasExited)
                    {
                        tcs.TrySetResult(process.ExitCode);
                    }

                    int exitCode = await tcs.Task.ConfigureAwait(false);

                    // Ensure all output buffers are flushed
                    process.WaitForExit();

                    if (cancellationToken.IsCancellationRequested)
                    {
                        throw new OperationCanceledException(
                            TexMotionLocalization.Tr(TexMotionLocalization.ExtractionCancelled),
                            cancellationToken);
                    }

                    string stderrText;
                    lock (stderrBuilder)
                    {
                        stderrText = stderrBuilder.ToString();
                    }

                    if (exitCode != 0)
                    {
                        string msg = string.IsNullOrEmpty(stageErrorMessage)
                            ? TexMotionLocalization.TrFormat(
                                TexMotionLocalization.ExtractionFailedExitCode,
                                exitCode,
                                stderrText)
                            : TexMotionLocalization.TrFormat(
                                TexMotionLocalization.ExtractionFailedStage,
                                lastStage,
                                stageErrorMessage,
                                stderrText);
                        string nativeDiagnostic = DescribeNativeExitCode(exitCode);
                        if (!string.IsNullOrEmpty(nativeDiagnostic))
                        {
                            msg += "\nDiagnostic: " + nativeDiagnostic;
                        }
                        throw new VideoExtractionException(msg, exitCode, stderrText, lastStage);
                    }

                    if (!string.IsNullOrEmpty(stageErrorMessage))
                    {
                        throw new VideoExtractionException(
                            TexMotionLocalization.TrFormat(
                                TexMotionLocalization.ExtractionReportedError,
                                stageErrorMessage,
                                stderrText),
                            exitCode,
                            stderrText,
                            lastStage);
                    }
                }
                catch (Exception) when (cancellationToken.IsCancellationRequested)
                {
                    KillProcessTree(process);
                    throw new OperationCanceledException(
                        TexMotionLocalization.Tr(TexMotionLocalization.ExtractionCancelled),
                        cancellationToken);
                }
            }

            // 7. Validate output file existence
            if (!File.Exists(outputPath))
            {
                string stderrText;
                lock (stderrBuilder)
                {
                    stderrText = stderrBuilder.ToString();
                }
                throw new FileNotFoundException(TexMotionLocalization.TrFormat(
                    TexMotionLocalization.OutputMotionFileMissing,
                    outputPath,
                    stderrText));
            }

            // 8. Deserialize into VideoMotionData
            var motionData = LoadFromJsonFile(outputPath);
            if (string.IsNullOrEmpty(motionData.SourceVideoPath))
            {
                motionData.SourceVideoPath = options.VideoPath;
            }

            // 9. Report final completed progress
            DispatchProgress(syncContext, progress, null, null,
                new VideoJobProgress(1.0f, "completed", motionData.Frames, motionData.Frames, outputPath));

            return motionData;
        }

        /// <summary>
        /// Runs the lightweight ViTPose archive exporter used by Settings. The
        /// process emits the same JSONL progress contract as normal extraction,
        /// while the returned path points to a validated ``.npz`` archive.
        /// </summary>
        public static async Task<string> RunViTPoseFeatureExportAsync(
            ViTPoseFeatureExportOptions options,
            IProgress<VideoJobProgress> progress = null,
            CancellationToken cancellationToken = default,
            Action<string> onStderrLine = null)
        {
            if (options == null) throw new ArgumentNullException(nameof(options));
            if (string.IsNullOrWhiteSpace(options.VideoPath) || !File.Exists(options.VideoPath))
                throw new FileNotFoundException("ViTPose feature export video was not found.", options.VideoPath);

            string scriptPath = options.ScriptPath;
            if (string.IsNullOrEmpty(scriptPath) || !File.Exists(scriptPath))
                scriptPath = FindScriptPath();
            if (string.IsNullOrEmpty(scriptPath) || !File.Exists(scriptPath))
                throw new FileNotFoundException(TexMotionLocalization.Tr(TexMotionLocalization.ExtractorScriptMissing));

            string pythonExe = options.PythonExecutable;
            if (string.IsNullOrEmpty(pythonExe))
            {
                try { pythonExe = TexMotionSettings.instance?.GetEffectiveVideoPythonExecutablePath(); }
                catch { }
            }
            if (string.IsNullOrEmpty(pythonExe) || !File.Exists(pythonExe))
            {
                var runtime = DetectPythonRuntime();
                if (!runtime.IsAvailable)
                    throw new InvalidOperationException(TexMotionLocalization.TrFormat(
                        TexMotionLocalization.NoWorkingPythonRuntime, runtime.ErrorMessage));
                pythonExe = runtime.ExecutablePath;
            }

            PythonRuntimeInfo selectedRuntime = ProbePythonExecutable(pythonExe);
            if (!selectedRuntime.IsAvailable)
                throw new InvalidOperationException(TexMotionLocalization.TrFormat(
                    TexMotionLocalization.SelectedPythonRuntimeFailed,
                    selectedRuntime.ErrorMessage ?? pythonExe));
            var missingDependencies = new List<string>();
            if (!selectedRuntime.HasNumPy) missingDependencies.Add("numpy");
            if (!selectedRuntime.HasOpenCV) missingDependencies.Add("opencv-python");
            if (missingDependencies.Count > 0)
                throw new InvalidOperationException(TexMotionLocalization.TrFormat(
                    TexMotionLocalization.MissingVideoDependencies,
                    string.Join(", ", missingDependencies), selectedRuntime.ExecutablePath));

            string outputPath = options.OutputPath;
            if (string.IsNullOrWhiteSpace(outputPath))
            {
                string cache = Path.Combine(Path.GetTempPath(), "TexMotion", "VideoJobs");
                Directory.CreateDirectory(cache);
                outputPath = Path.Combine(cache, "vitpose_features_" + Guid.NewGuid().ToString("N") + ".npz");
            }
            if (!outputPath.EndsWith(".npz", StringComparison.OrdinalIgnoreCase))
                outputPath += ".npz";
            string outDir = Path.GetDirectoryName(outputPath);
            if (!string.IsNullOrEmpty(outDir)) Directory.CreateDirectory(outDir);

            string arguments = options.BuildCommandLineArguments(scriptPath, outputPath);
            var syncContext = SynchronizationContext.Current;
            var stderrBuilder = new StringBuilder();
            string lastStage = "init";
            string stageErrorMessage = null;
            var psi = new ProcessStartInfo
            {
                FileName = pythonExe,
                Arguments = arguments,
                UseShellExecute = false,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                CreateNoWindow = true,
                StandardOutputEncoding = Encoding.UTF8,
                StandardErrorEncoding = Encoding.UTF8,
                WorkingDirectory = Path.GetDirectoryName(scriptPath)
            };
            psi.EnvironmentVariables["PYTHONUNBUFFERED"] = "1";
            psi.EnvironmentVariables["PYTHONIOENCODING"] = "utf-8";
            using var process = new Process { StartInfo = psi, EnableRaisingEvents = true };
            var tcs = new TaskCompletionSource<int>();
            process.Exited += (s, e) => tcs.TrySetResult(process.ExitCode);
            process.OutputDataReceived += (s, e) =>
            {
                if (e.Data == null) return;
                if (TryParseProgressJson(e.Data, out var item))
                {
                    lastStage = item.Stage;
                    if (item.IsError) stageErrorMessage = item.ErrorMessage;
                    DispatchProgress(syncContext, progress, null, null, item);
                }
            };
            process.ErrorDataReceived += (s, e) =>
            {
                if (e.Data == null) return;
                lock (stderrBuilder) stderrBuilder.AppendLine(e.Data);
                onStderrLine?.Invoke(e.Data);
            };

            using (cancellationToken.Register(() =>
            {
                KillProcessTree(process);
                tcs.TrySetCanceled(cancellationToken);
            }))
            {
                try
                {
                    if (!process.Start())
                        throw new InvalidOperationException(TexMotionLocalization.TrFormat(
                            TexMotionLocalization.FailedToStartPythonProcess, pythonExe));
                    process.BeginOutputReadLine();
                    process.BeginErrorReadLine();
                    DispatchProgress(syncContext, progress, null, null,
                        new VideoJobProgress(0.01f, "vitpose_features_init", 0, 0));
                    if (process.HasExited) tcs.TrySetResult(process.ExitCode);
                    int exitCode = await tcs.Task.ConfigureAwait(false);
                    process.WaitForExit();
                    if (cancellationToken.IsCancellationRequested)
                        throw new OperationCanceledException(TexMotionLocalization.Tr(TexMotionLocalization.ExtractionCancelled), cancellationToken);
                    string stderrText;
                    lock (stderrBuilder) stderrText = stderrBuilder.ToString();
                    if (exitCode != 0)
                    {
                        string message = string.IsNullOrEmpty(stageErrorMessage)
                            ? TexMotionLocalization.TrFormat(TexMotionLocalization.ExtractionFailedExitCode, exitCode, stderrText)
                            : TexMotionLocalization.TrFormat(TexMotionLocalization.ExtractionFailedStage, lastStage, stageErrorMessage, stderrText);
                        string nativeDiagnostic = DescribeNativeExitCode(exitCode);
                        if (!string.IsNullOrEmpty(nativeDiagnostic)) message += "\nDiagnostic: " + nativeDiagnostic;
                        throw new VideoExtractionException(message, exitCode, stderrText, lastStage);
                    }
                    if (!string.IsNullOrEmpty(stageErrorMessage))
                        throw new VideoExtractionException(
                            TexMotionLocalization.TrFormat(TexMotionLocalization.ExtractionReportedError, stageErrorMessage, stderrText),
                            exitCode, stderrText, lastStage);
                }
                catch (Exception) when (cancellationToken.IsCancellationRequested)
                {
                    KillProcessTree(process);
                    throw new OperationCanceledException(TexMotionLocalization.Tr(TexMotionLocalization.ExtractionCancelled), cancellationToken);
                }
            }

            if (!File.Exists(outputPath) || new FileInfo(outputPath).Length <= 0)
                throw new FileNotFoundException("ViTPose feature export did not produce a readable .npz archive.", outputPath);
            DispatchProgress(syncContext, progress, null, null,
                new VideoJobProgress(1.0f, "completed", 0, 0, outputPath));
            return Path.GetFullPath(outputPath);
        }

        /// <summary>
        /// Overload accepting an Action callback for progress notifications.
        /// </summary>
        public static Task<VideoMotionData> RunExtractionAsync(
            VideoJobOptions options,
            Action<VideoJobProgress> onProgress,
            CancellationToken cancellationToken = default,
            Action<string> onStderrLine = null)
        {
            var progressAdapter = onProgress != null
                ? new Progress<VideoJobProgress>(onProgress)
                : null;
            return RunExtractionAsync(options, progressAdapter, cancellationToken, onStderrLine);
        }

        /// <summary>
        /// Overload accepting an IProgress&lt;float&gt; for simple normalized progress reporting.
        /// </summary>
        public static Task<VideoMotionData> RunExtractionAsync(
            VideoJobOptions options,
            IProgress<float> progress,
            CancellationToken cancellationToken = default,
            Action<string> onStderrLine = null)
        {
            var progressAdapter = progress != null
                ? new Progress<VideoJobProgress>(p => progress.Report(p.Progress))
                : null;
            return RunExtractionAsync(options, progressAdapter, cancellationToken, onStderrLine);
        }

        /// <summary>
        /// Simplified overload with default options for a video path.
        /// </summary>
        public static Task<VideoMotionData> RunExtractionAsync(
            string videoPath,
            string outputPath = null,
            IProgress<VideoJobProgress> progress = null,
            CancellationToken cancellationToken = default)
        {
            var options = new VideoJobOptions
            {
                VideoPath = videoPath,
                OutputPath = outputPath
            };
            return RunExtractionAsync(options, progress, cancellationToken);
        }

        #endregion

        #region Progress Parsing & Thread Safety

        /// <summary>
        /// Attempts to parse a JSON line emitted by video_pose_extractor.py into a VideoJobProgress struct.
        /// Handles both standard JsonUtility parsing and robust regex fallback.
        /// </summary>
        public static bool TryParseProgressJson(string jsonLine, out VideoJobProgress progress)
        {
            progress = default;
            if (string.IsNullOrEmpty(jsonLine)) return false;

            string trimmed = jsonLine.Trim();
            if (!trimmed.StartsWith("{") || !trimmed.EndsWith("}")) return false;

            // Attempt 1: JsonUtility
            try
            {
                var dto = JsonUtility.FromJson<ProgressJsonDto>(trimmed);
                if (dto != null && (!string.IsNullOrEmpty(dto.stage) || dto.progress > 0f))
                {
                    progress = new VideoJobProgress(
                        dto.progress,
                        dto.stage ?? "processing",
                        dto.frame,
                        dto.total_frames,
                        dto.output,
                        dto.error,
                        trimmed
                    );
                    return true;
                }
            }
            catch
            {
                // Fallback to regex below
            }

            // Attempt 2: Manual regex parsing fallback
            try
            {
                if (trimmed.Contains("\"progress\"") || trimmed.Contains("\"stage\""))
                {
                    float prog = ExtractFloat(trimmed, "progress", 0f);
                    string stage = ExtractString(trimmed, "stage", "processing");
                    int frame = ExtractInt(trimmed, "frame", 0);
                    int total = ExtractInt(trimmed, "total_frames", 0);
                    string output = ExtractString(trimmed, "output", null);
                    string error = ExtractString(trimmed, "error", null);

                    progress = new VideoJobProgress(prog, stage, frame, total, output, error, trimmed);
                    return true;
                }
            }
            catch
            {
            }

            return false;
        }

        private static void DispatchProgress(
            SynchronizationContext syncContext,
            IProgress<VideoJobProgress> progress,
            Action<VideoJobProgress> onProgress,
            IProgress<float> progressFloat,
            VideoJobProgress p)
        {
            if (syncContext != null)
            {
                syncContext.Post(_ =>
                {
                    progress?.Report(p);
                    onProgress?.Invoke(p);
                    progressFloat?.Report(p.Progress);
                }, null);
            }
            else
            {
                EditorApplication.delayCall += () =>
                {
                    progress?.Report(p);
                    onProgress?.Invoke(p);
                    progressFloat?.Report(p.Progress);
                };
            }
        }

        private static float ExtractFloat(string json, string key, float defaultValue)
        {
            var m = Regex.Match(json, $"\"{key}\"\\s*:\\s*([-\\d\\.eE]+)");
            if (m.Success && float.TryParse(m.Groups[1].Value, NumberStyles.Float, CultureInfo.InvariantCulture, out float v))
            {
                return v;
            }
            return defaultValue;
        }

        private static int ExtractInt(string json, string key, int defaultValue)
        {
            var m = Regex.Match(json, $"\"{key}\"\\s*:\\s*(\\d+)");
            if (m.Success && int.TryParse(m.Groups[1].Value, NumberStyles.Integer, CultureInfo.InvariantCulture, out int v))
            {
                return v;
            }
            return defaultValue;
        }

        private static bool ExtractBool(string json, string key, bool defaultValue)
        {
            var m = Regex.Match(json, $"\"{key}\"\\s*:\\s*(true|false)", RegexOptions.IgnoreCase);
            if (m.Success && bool.TryParse(m.Groups[1].Value, out bool value))
            {
                return value;
            }
            return defaultValue;
        }

        private static string ExtractString(string json, string key, string defaultValue)
        {
            var m = Regex.Match(json, $"\"{key}\"\\s*:\\s*\"([^\"]*)\"");
            if (m.Success)
            {
                return m.Groups[1].Value;
            }
            return defaultValue;
        }

#pragma warning disable CS0649
        [Serializable]
        private class ProgressJsonDto
        {
            public float progress;
            public string stage;
            public int frame;
            public int total_frames;
            public string output;
            public string error;
        }
#pragma warning restore CS0649

        #endregion

        #region Motion JSON Deserialization

        /// <summary>
        /// Reads and deserializes a motion JSON file exported by video_pose_extractor.py into VideoMotionData.
        /// </summary>
        public static VideoMotionData LoadFromJsonFile(string filePath)
        {
            if (string.IsNullOrEmpty(filePath))
            {
                throw new ArgumentNullException(nameof(filePath));
            }

            if (!File.Exists(filePath))
            {
                throw new FileNotFoundException(TexMotionLocalization.TrFormat(
                    TexMotionLocalization.MotionJsonNotFound,
                    filePath));
            }

            string jsonText = File.ReadAllText(filePath, Encoding.UTF8);
            var motionData = DeserializeFromJson(jsonText);
            if (string.IsNullOrEmpty(motionData.SourceVideoPath))
            {
                motionData.SourceVideoPath = filePath;
            }
            return motionData;
        }

        /// <summary>
        /// Deserializes JSON text matching the VideoMotionData schema into a VideoMotionData instance.
        /// Reconstructs SMPL-X 22 local rotations, root positions, timestamps, and confidences.
        /// </summary>
        public static VideoMotionData DeserializeFromJson(string jsonText)
        {
            if (string.IsNullOrEmpty(jsonText))
            {
                throw new ArgumentNullException(
                    nameof(jsonText),
                    TexMotionLocalization.Tr(TexMotionLocalization.JsonContentEmpty));
            }

            MotionDataJsonDto dto = null;
            try
            {
                dto = JsonUtility.FromJson<MotionDataJsonDto>(jsonText);
            }
            catch
            {
                // Fallback to manual parser below
            }

            if (dto == null || dto.frames <= 0)
            {
                dto = ParseMotionJsonDtoFallback(jsonText);
            }

            if (dto == null || dto.frames <= 0)
            {
                throw new FormatException(TexMotionLocalization.Tr(
                    TexMotionLocalization.MotionJsonDeserializeFailed));
            }

            int frames = dto.frames;
            int parsedJointCount = dto.jointCount > 0 ? dto.jointCount : SmplxJointDefinitions.JointCount;
            // VideoMotionData is a SMPL-X22 runtime contract.  Older or
            // malformed payloads may advertise a different count; preserve
            // the available prefix and fill the canonical 22-joint shape so
            // downstream retargeting never receives an incompatible matrix.
            int jointCount = SmplxJointDefinitions.JointCount;
            float frameRate = dto.frameRate > 0f ? dto.frameRate : 30.0f;

            // 1. Root positions
            var rootPositions = new Vector3[frames];
            if (dto.rootPositions != null && dto.rootPositions.Length >= frames)
            {
                for (int t = 0; t < frames; t++)
                {
                    rootPositions[t] = new Vector3(
                        SanitizeFinite(dto.rootPositions[t].x),
                        SanitizeFinite(dto.rootPositions[t].y),
                        SanitizeFinite(dto.rootPositions[t].z));
                }
            }
            else
            {
                for (int t = 0; t < frames; t++)
                {
                    rootPositions[t] = Vector3.zero;
                }
            }

            // 2. 22 Local rotations
            var localRotations = new Quaternion[frames, jointCount];
            if (dto.flatLocalRotations != null && dto.flatLocalRotations.Length >= frames * parsedJointCount)
            {
                for (int t = 0; t < frames; t++)
                {
                    for (int j = 0; j < jointCount; j++)
                    {
                        int sourceIndex = t * parsedJointCount + j;
                        if (j < parsedJointCount && sourceIndex < dto.flatLocalRotations.Length)
                        {
                            var q = dto.flatLocalRotations[sourceIndex];
                            localRotations[t, j] = SanitizeQuaternion(q);
                        }
                        else
                        {
                            localRotations[t, j] = Quaternion.identity;
                        }
                    }
                }
            }
            else
            {
                // Fallback: parse from nested localRotations if flatLocalRotations wasn't present
                var fallbackRotations = ParseLocalRotationsFallback(jsonText, frames, parsedJointCount);
                if (fallbackRotations != null)
                {
                    for (int t = 0; t < frames; t++)
                    {
                        for (int j = 0; j < jointCount; j++)
                        {
                            localRotations[t, j] = j < parsedJointCount
                                ? SanitizeQuaternion(fallbackRotations[t, j])
                                : Quaternion.identity;
                        }
                    }
                }
                else
                {
                    for (int t = 0; t < frames; t++)
                    {
                        for (int j = 0; j < jointCount; j++)
                        {
                            localRotations[t, j] = Quaternion.identity;
                        }
                    }
                }
            }

            // 3. Timestamps & Confidences
            float[] timestamps = dto.timestamps;
            float[] confidences = dto.confidences;

            // 4. Uncertainty Intervals (Phases 3-4)
            UncertaintyInterval[] intervals = null;
            if (dto.uncertaintyIntervals != null && dto.uncertaintyIntervals.Length > 0)
            {
                var list = new System.Collections.Generic.List<UncertaintyInterval>();
                for (int i = 0; i < dto.uncertaintyIntervals.Length; i++)
                {
                    var u = dto.uncertaintyIntervals[i];
                    if (u != null)
                    {
                        int startFrame = Mathf.Clamp(u.startFrame, 0, frames - 1);
                        int endFrame = Mathf.Clamp(u.endFrame, startFrame, frames - 1);
                        var interval = new UncertaintyInterval(
                            startFrame,
                            endFrame,
                            u.reason,
                            Mathf.Clamp01(SanitizeFinite(u.confidence)),
                            u.recommendedAction)
                        {
                            PrimaryHypothesis = u.primaryHypothesis,
                            AlternativeHypotheses = u.alternativeHypotheses ?? Array.Empty<string>(),
                            MaxJointDisagreementMeters = u.maxJointDisagreementMeters
                        };
                        list.Add(interval);
                    }
                }
                intervals = list.Count > 0 ? list.ToArray() : null;
            }

            // A pre-provenance motion file can still contain detectorName (or
            // no backend fields at all).  Do not infer a successful fusion
            // from those legacy fields; mark the provenance explicitly as
            // legacy while retaining the original detector label for display.
            PromoteTopLevelImageFeatureMetadata(dto);
            bool hasModernProvenance = dto.backendMetadata != null ||
                HasJsonField(jsonText, "backendRequested") ||
                HasJsonField(jsonText, "backendActual") ||
                HasJsonField(jsonText, "fusionMode") ||
                HasJsonField(jsonText, "backendFallback") ||
                HasJsonField(jsonText, "overlayBackend") ||
                HasJsonField(jsonText, "overlaySource") ||
                HasJsonField(jsonText, "officialRunnerStatus") ||
                HasJsonField(jsonText, "hmr2ImageFeaturesStatus") ||
                HasJsonField(jsonText, "vitpose2DStatus") ||
                HasJsonField(jsonText, "optionalStageDiagnostics");
            string detectorName = string.IsNullOrEmpty(dto.detectorName) ? "legacy" : dto.detectorName;
            string backendRequested = dto.backendRequested ?? dto.backendMetadata?.backendRequested;
            string backendActual = dto.backendActual ?? dto.backendMetadata?.backendActual;
            string fusionMode = dto.fusionMode ?? dto.backendMetadata?.fusionMode;
            string fallbackFrom = dto.fallbackFrom ?? dto.backendMetadata?.fallbackFrom;
            string fallbackReason = dto.fallbackReason ?? dto.backendMetadata?.fallbackReason;
            string officialRunnerStatus = dto.officialRunnerStatus ?? dto.backendMetadata?.officialRunnerStatus;
            string hmr2ImageFeaturesStatus = dto.hmr2ImageFeaturesStatus ?? dto.backendMetadata?.hmr2ImageFeaturesStatus;
            string vitPose2DStatus = dto.vitpose2DStatus ?? dto.backendMetadata?.vitpose2DStatus;
            VideoOptionalStageDiagnostic[] optionalStageDiagnostics = ConvertOptionalStageDiagnostics(
                dto.optionalStageDiagnostics ?? dto.backendMetadata?.optionalStageDiagnostics);
            string overlayBackend = dto.overlayBackend ?? dto.backendMetadata?.overlayBackend;
            string overlaySource = dto.overlaySource ?? dto.backendMetadata?.overlaySource;
            bool backendFallback = dto.backendFallback || (dto.backendMetadata?.backendFallback ?? false);
            if (!hasModernProvenance)
            {
                backendRequested = "legacy";
                backendActual = "legacy";
                fusionMode = "off";
                backendFallback = false;
                fallbackFrom = null;
                fallbackReason = null;
                officialRunnerStatus = null;
                hmr2ImageFeaturesStatus = null;
                vitPose2DStatus = null;
                optionalStageDiagnostics = Array.Empty<VideoOptionalStageDiagnostic>();
                overlayBackend = "legacy";
                overlaySource = "legacy";
            }
            else
            {
                backendRequested = string.IsNullOrEmpty(backendRequested) ? detectorName : backendRequested;
                backendActual = string.IsNullOrEmpty(backendActual) ? detectorName : backendActual;
                fusionMode = string.IsNullOrEmpty(fusionMode) ? "off" : fusionMode;
                overlayBackend = string.IsNullOrEmpty(overlayBackend) ? detectorName : overlayBackend;
                overlaySource = string.IsNullOrEmpty(overlaySource) ? "backend_output" : overlaySource;
            }

            VideoBackendMetadata backendMetadata = ConvertBackendMetadata(
                dto.backendMetadata,
                backendRequested,
                backendActual,
                fusionMode,
                backendFallback,
                fallbackFrom,
                fallbackReason,
                overlayBackend,
                overlaySource,
                detectorName,
                officialRunnerStatus,
                hmr2ImageFeaturesStatus,
                vitPose2DStatus,
                optionalStageDiagnostics);

            var motionData = new VideoMotionData(
                frames,
                jointCount,
                rootPositions,
                localRotations,
                frameRate,
                timestamps,
                confidences,
                null, // jointConfidences
                dto.sourceVideoPath,
                dto.videoWidth,
                dto.videoHeight,
                dto.videoFps,
                dto.overlayVideoPath,
                intervals,
                detectorName,
                backendRequested,
                fallbackReason,
                backendActual,
                fusionMode,
                backendFallback,
                fallbackFrom,
                overlayBackend,
                overlaySource,
                backendMetadata,
                officialRunnerStatus,
                hmr2ImageFeaturesStatus,
                vitPose2DStatus,
                optionalStageDiagnostics
            );

            if (!motionData.Validate(out string validationError))
            {
                Debug.LogWarning(TexMotionLocalization.TrFormat(
                    TexMotionLocalization.VideoDataValidationWarning,
                    validationError));
            }

            return motionData;
        }

        /// <summary>
        /// Promotes the extractor's flat image-feature provenance into the
        /// editor DTO without changing the runtime VideoMotionData contract.
        /// The nested stage is retained verbatim in BackendMetadataJsonDto so
        /// status/error code/counts remain available to the result card.
        /// </summary>
        private static void PromoteTopLevelImageFeatureMetadata(MotionDataJsonDto dto)
        {
            if (dto == null) return;
            ImageFeatureRunnerMetadataJsonDto stage = dto.imageFeatureStage ??
                dto.imageFeatureRunnerMetadata ?? dto.backendMetadata?.imageFeatureStage ??
                dto.backendMetadata?.imageFeatureRunnerMetadata;
            bool hasFlat = !string.IsNullOrWhiteSpace(dto.imageFeatureRunner) ||
                           !string.IsNullOrWhiteSpace(dto.imageFeatureRunnerStatus) ||
                           stage != null ||
                           dto.imageFeatureRunnerFeatureDim > 0 ||
                           dto.imageFeatureInputFrameCount > 0 ||
                           dto.imageFeatureAdoptedFrameCount > 0 ||
                           !string.IsNullOrWhiteSpace(dto.imageFeatureErrorCode) ||
                           !string.IsNullOrWhiteSpace(dto.imageFeatureFallback);
            if (!hasFlat) return;
            if (dto.backendMetadata == null) dto.backendMetadata = new BackendMetadataJsonDto();
            if (stage != null)
            {
                dto.backendMetadata.imageFeatureStage = stage;
                dto.backendMetadata.imageFeatureRunnerMetadata = stage;
            }
            if (string.IsNullOrWhiteSpace(dto.backendMetadata.imageFeatureRunner))
                dto.backendMetadata.imageFeatureRunner = dto.imageFeatureRunner;
            if (string.IsNullOrWhiteSpace(dto.backendMetadata.imageFeatureRunnerStatus))
                dto.backendMetadata.imageFeatureRunnerStatus = dto.imageFeatureRunnerStatus;
            if (string.IsNullOrWhiteSpace(dto.backendMetadata.imageFeatureFallback))
                dto.backendMetadata.imageFeatureFallback = dto.imageFeatureFallback;
            if (dto.backendMetadata.imageFeatureRunnerFeatureDim <= 0)
                dto.backendMetadata.imageFeatureRunnerFeatureDim = dto.imageFeatureRunnerFeatureDim;
            if (dto.backendMetadata.imageFeatureInputFrameCount <= 0)
                dto.backendMetadata.imageFeatureInputFrameCount = dto.imageFeatureInputFrameCount;
            if (dto.backendMetadata.imageFeatureAdoptedFrameCount <= 0)
                dto.backendMetadata.imageFeatureAdoptedFrameCount = dto.imageFeatureAdoptedFrameCount;
            if (string.IsNullOrWhiteSpace(dto.backendMetadata.imageFeatureRequestedRunner))
                dto.backendMetadata.imageFeatureRequestedRunner = dto.imageFeatureRequestedRunner;
            if (string.IsNullOrWhiteSpace(dto.backendMetadata.imageFeatureStageStatus))
                dto.backendMetadata.imageFeatureStageStatus = dto.imageFeatureStageStatus;
            if (string.IsNullOrWhiteSpace(dto.backendMetadata.imageFeatureRuntime))
                dto.backendMetadata.imageFeatureRuntime = dto.imageFeatureRuntime;
            if (string.IsNullOrWhiteSpace(dto.backendMetadata.imageFeatureCheckpointPath))
                dto.backendMetadata.imageFeatureCheckpointPath = dto.imageFeatureCheckpointPath;
            if (string.IsNullOrWhiteSpace(dto.backendMetadata.imageFeatureCheckpointSha256))
                dto.backendMetadata.imageFeatureCheckpointSha256 = dto.imageFeatureCheckpointSha256;
            if (string.IsNullOrWhiteSpace(dto.backendMetadata.imageFeatureErrorCode))
                dto.backendMetadata.imageFeatureErrorCode = dto.imageFeatureErrorCode;
            if (string.IsNullOrWhiteSpace(dto.backendMetadata.imageFeatureNextAction))
                dto.backendMetadata.imageFeatureNextAction = dto.imageFeatureNextAction;
            if (dto.backendMetadata.imageFeatureInputFrameIndices == null)
                dto.backendMetadata.imageFeatureInputFrameIndices = dto.imageFeatureInputFrameIndices;
            if (dto.backendMetadata.imageFeatureAdoptedFrameIndices == null)
                dto.backendMetadata.imageFeatureAdoptedFrameIndices = dto.imageFeatureAdoptedFrameIndices;
            if (dto.imageFeatureAdoptedFrameCount > 0 && stage != null && stage.acceptedFrameCount <= 0)
                stage.acceptedFrameCount = dto.imageFeatureAdoptedFrameCount;
        }

        private static bool HasJsonField(string json, string fieldName)
        {
            return !string.IsNullOrEmpty(json) && !string.IsNullOrEmpty(fieldName) &&
                json.IndexOf($"\"{fieldName}\"", StringComparison.OrdinalIgnoreCase) >= 0;
        }

        private static float SanitizeFinite(float value, float fallback = 0f)
        {
            return float.IsNaN(value) || float.IsInfinity(value) ? fallback : value;
        }

        private static Quaternion SanitizeQuaternion(Vector4Dto value)
        {
            float x = SanitizeFinite(value.x);
            float y = SanitizeFinite(value.y);
            float z = SanitizeFinite(value.z);
            float w = SanitizeFinite(value.w);
            float magnitude = Mathf.Sqrt(x * x + y * y + z * z + w * w);
            if (magnitude < 0.00001f) return Quaternion.identity;
            return new Quaternion(x / magnitude, y / magnitude, z / magnitude, w / magnitude);
        }

        private static Quaternion SanitizeQuaternion(Quaternion value)
        {
            return SanitizeQuaternion(new Vector4Dto { x = value.x, y = value.y, z = value.z, w = value.w });
        }

        private static VideoBackendMetadata ConvertBackendMetadata(
            BackendMetadataJsonDto dto,
            string backendRequested,
            string backendActual,
            string fusionMode,
            bool backendFallback,
            string fallbackFrom,
            string fallbackReason,
            string overlayBackend,
            string overlaySource,
            string detectorName,
            string officialRunnerStatus,
            string hmr2ImageFeaturesStatus,
            string vitPose2DStatus,
            VideoOptionalStageDiagnostic[] optionalStageDiagnostics)
        {
            var metadata = VideoBackendMetadata.CreateLegacy(
                backendRequested,
                string.IsNullOrEmpty(backendActual) ? detectorName : backendActual,
                fusionMode,
                backendFallback,
                fallbackFrom,
                fallbackReason,
                overlayBackend,
                overlaySource);
            if (dto == null)
            {
                metadata.OfficialRunnerStatus = officialRunnerStatus;
                metadata.Hmr2ImageFeaturesStatus = hmr2ImageFeaturesStatus;
                metadata.VitPose2DStatus = vitPose2DStatus;
                metadata.OptionalStageDiagnostics = optionalStageDiagnostics ??
                    Array.Empty<VideoOptionalStageDiagnostic>();
                return metadata;
            }

            metadata.SchemaVersion = dto.schemaVersion > 0 ? dto.schemaVersion : metadata.SchemaVersion;
            metadata.BackendRequested = string.IsNullOrEmpty(dto.backendRequested) ? metadata.BackendRequested : dto.backendRequested;
            metadata.BackendActual = string.IsNullOrEmpty(dto.backendActual) ? metadata.BackendActual : dto.backendActual;
            metadata.OfficialRunnerStatus = string.IsNullOrEmpty(dto.officialRunnerStatus)
                ? officialRunnerStatus
                : dto.officialRunnerStatus;
            metadata.Hmr2ImageFeaturesStatus = string.IsNullOrEmpty(dto.hmr2ImageFeaturesStatus)
                ? hmr2ImageFeaturesStatus
                : dto.hmr2ImageFeaturesStatus;
            metadata.VitPose2DStatus = string.IsNullOrEmpty(dto.vitpose2DStatus)
                ? vitPose2DStatus
                : dto.vitpose2DStatus;
            metadata.FusionMode = string.IsNullOrEmpty(dto.fusionMode) ? metadata.FusionMode : dto.fusionMode;
            metadata.BackendFallback = dto.backendFallback || metadata.BackendFallback;
            metadata.FallbackFrom = string.IsNullOrEmpty(dto.fallbackFrom) ? metadata.FallbackFrom : dto.fallbackFrom;
            metadata.FallbackReason = string.IsNullOrEmpty(dto.fallbackReason) ? metadata.FallbackReason : dto.fallbackReason;
            metadata.OverlayBackend = string.IsNullOrEmpty(dto.overlayBackend) ? metadata.OverlayBackend : dto.overlayBackend;
            metadata.OverlaySource = string.IsNullOrEmpty(dto.overlaySource) ? metadata.OverlaySource : dto.overlaySource;
            metadata.CoordinateSystem = string.IsNullOrEmpty(dto.coordinateSystem) ? metadata.CoordinateSystem : dto.coordinateSystem;
            metadata.WorldMotionAvailable = dto.worldMotionAvailable;
            metadata.ScaleIsRelative = dto.scaleIsRelative;
            metadata.FusionAvailable = dto.fusionAvailable;
            metadata.FusionError = dto.fusionError;
            metadata.PrimaryBackend = string.IsNullOrEmpty(dto.primaryBackend) ? metadata.BackendActual : dto.primaryBackend;
            metadata.AuxiliaryBackends = dto.auxiliaryBackends ?? Array.Empty<string>();
            metadata.QualityFrameFallbackCount = Mathf.Max(0, dto.qualityFrameFallbackCount);
            metadata.QualityFallbackReason = dto.qualityFallbackReason;
            metadata.WhamAssetDiagnostic = dto.whamAssetDiagnostic;
            metadata.WhamMissingStages = dto.whamMissingStages ?? Array.Empty<string>();
            metadata.SelectedBackend = dto.selectedBackend;
            metadata.BackendReady = dto.backendReady;
            metadata.PreflightStatus = dto.preflightStatus;
            metadata.PreflightPhase = dto.preflightPhase;
            metadata.PreflightError = dto.preflightError;
            metadata.PreflightFrames = Mathf.Max(0, dto.preflightFrames);
            metadata.MissingAssets = dto.missingAssets ?? Array.Empty<string>();
            metadata.IncompatibleAssets = dto.incompatibleAssets ?? Array.Empty<string>();
            metadata.AssetDiagnostics = dto.assetDiagnostics ?? Array.Empty<string>();
            metadata.MissingOptionalAssets = dto.missingOptionalAssets ?? Array.Empty<string>();
            metadata.OptionalErrors = dto.optionalErrors ?? Array.Empty<string>();
            metadata.OptionalStageDiagnostics = dto.optionalStageDiagnostics != null
                ? ConvertOptionalStageDiagnostics(dto.optionalStageDiagnostics)
                : optionalStageDiagnostics ?? Array.Empty<VideoOptionalStageDiagnostic>();
            metadata.ImageFeaturePreprocess = dto.imageFeaturePreprocess;
            metadata.ImageFeatureRunnerActive = dto.imageFeatureRunnerActive;
            metadata.ImageFeatureRunnerStatus = dto.imageFeatureRunnerStatus;
            metadata.ImageFeatureRunner = dto.imageFeatureRunner;
            metadata.ImageFeatureRunnerModule = dto.imageFeatureRunnerModule;
            metadata.ImageFeatureRunnerError = dto.imageFeatureRunnerError;
            metadata.ImageFeatureFallback = dto.imageFeatureFallback;
            metadata.ImageFeatureFallbackReason = dto.imageFeatureFallbackReason;
            metadata.ImageFeatureFallbackConsumed = dto.imageFeatureFallbackConsumed;
            ImageFeatureRunnerMetadataJsonDto featureStage = dto.imageFeatureStage ??
                dto.imageFeatureRunnerMetadata;
            if (featureStage != null)
            {
                if (!string.IsNullOrWhiteSpace(featureStage.runner))
                    metadata.ImageFeatureRunner = featureStage.runner;
                if (!string.IsNullOrWhiteSpace(featureStage.status))
                    metadata.ImageFeatureRunnerStatus = featureStage.status;
                if (!string.IsNullOrWhiteSpace(featureStage.error))
                    metadata.ImageFeatureRunnerError = featureStage.error;
                if (!string.IsNullOrWhiteSpace(featureStage.fallbackReason) &&
                    string.IsNullOrWhiteSpace(metadata.ImageFeatureFallbackReason))
                    metadata.ImageFeatureFallbackReason = featureStage.fallbackReason;
            }
            metadata.ImageFeatureRunnerFeatureDim = dto.imageFeatureRunnerFeatureDim > 0
                ? dto.imageFeatureRunnerFeatureDim
                : featureStage != null ? Mathf.Max(0, featureStage.featureDim) : 0;
            metadata.ImageFeatureInputFrameCount = dto.imageFeatureInputFrameCount > 0
                ? dto.imageFeatureInputFrameCount
                : featureStage != null
                    ? Mathf.Max(0, featureStage.frameCount > 0 ? featureStage.frameCount : featureStage.inputFrameCount)
                    : (dto.imageFeatureInputFrameIndices == null ? 0 : dto.imageFeatureInputFrameIndices.Length);
            metadata.ImageFeatureAdoptedFrameCount = dto.imageFeatureAdoptedFrameCount > 0
                ? dto.imageFeatureAdoptedFrameCount
                : featureStage != null
                    ? Mathf.Max(0, featureStage.acceptedFrameCount > 0 ? featureStage.acceptedFrameCount : featureStage.adoptedFrameCount)
                    : (dto.imageFeatureAdoptedFrameIndices == null ? 0 : dto.imageFeatureAdoptedFrameIndices.Length);
            metadata.ImageFeatureRequestedRunner = !string.IsNullOrWhiteSpace(dto.imageFeatureRequestedRunner)
                ? dto.imageFeatureRequestedRunner
                : featureStage?.requestedRunner;
            metadata.ImageFeatureStageStatus = !string.IsNullOrWhiteSpace(dto.imageFeatureStageStatus)
                ? dto.imageFeatureStageStatus
                : featureStage?.status;
            metadata.ImageFeatureRuntime = !string.IsNullOrWhiteSpace(dto.imageFeatureRuntime)
                ? dto.imageFeatureRuntime
                : featureStage?.runtimeKind ?? featureStage?.runtime;
            metadata.ImageFeatureCheckpointPath = !string.IsNullOrWhiteSpace(dto.imageFeatureCheckpointPath)
                ? dto.imageFeatureCheckpointPath
                : featureStage?.checkpointPath;
            metadata.ImageFeatureCheckpointSha256 = !string.IsNullOrWhiteSpace(dto.imageFeatureCheckpointSha256)
                ? dto.imageFeatureCheckpointSha256
                : featureStage?.checkpointSha256;
            metadata.ImageFeatureErrorCode = !string.IsNullOrWhiteSpace(dto.imageFeatureErrorCode)
                ? dto.imageFeatureErrorCode
                : featureStage?.errorCode;
            metadata.ImageFeatureNextAction = !string.IsNullOrWhiteSpace(dto.imageFeatureNextAction)
                ? dto.imageFeatureNextAction
                : featureStage?.nextAction;
            metadata.CameraMotionRunner = dto.cameraMotionRunner;
            metadata.DpvoVerified = dto.dpvoVerified;
            if (dto.cameraMotionProvenance != null)
            {
                metadata.CameraMotionProvenance = new VideoCameraMotionProvenance
                {
                    Source = dto.cameraMotionProvenance.source,
                    Verified = dto.cameraMotionProvenance.verified,
                    Runner = dto.cameraMotionProvenance.runner,
                    Asset = dto.cameraMotionProvenance.asset
                };
            }
            metadata.WhamPreprocessManifestPath = dto.whamPreprocessManifestPath;
            List<string> runtimeWarnings = new List<string>(dto.runtimeWarnings ?? Array.Empty<string>());
            if (featureStage != null)
            {
                string stageStatus = string.IsNullOrWhiteSpace(featureStage.status) ? "unknown" : featureStage.status;
                string stageSummary = "Image feature stage: status=" + stageStatus;
                if (!string.IsNullOrWhiteSpace(featureStage.runtimePath))
                    stageSummary += ", runtime=" + featureStage.runtimePath;
                else if (!string.IsNullOrWhiteSpace(featureStage.runtime))
                    stageSummary += ", runtime=" + featureStage.runtime;
                if (!string.IsNullOrWhiteSpace(featureStage.checkpointPath))
                    stageSummary += ", checkpoint=" + featureStage.checkpointPath;
                if (featureStage.featureDim > 0)
                    stageSummary += string.Format(", featureDim={0}", featureStage.featureDim);
                int stageFrameCount = featureStage.frameCount > 0
                    ? featureStage.frameCount
                    : featureStage.inputFrameCount;
                int stageAcceptedFrameCount = featureStage.acceptedFrameCount > 0
                    ? featureStage.acceptedFrameCount
                    : featureStage.adoptedFrameCount;
                if (stageFrameCount > 0 || stageAcceptedFrameCount > 0)
                    stageSummary += string.Format(", acceptedFrames={0}/{1}",
                        Mathf.Max(0, stageAcceptedFrameCount),
                        Mathf.Max(0, stageFrameCount));
                if (!string.IsNullOrWhiteSpace(featureStage.errorCode))
                    stageSummary += ", errorCode=" + featureStage.errorCode;
                if (!string.IsNullOrWhiteSpace(featureStage.error))
                    stageSummary += ", error=" + featureStage.error;
                if (!string.IsNullOrWhiteSpace(featureStage.nextAction))
                    stageSummary += ", nextAction=" + featureStage.nextAction;
                runtimeWarnings.Add(stageSummary);
            }
            metadata.RuntimeWarnings = runtimeWarnings.ToArray();
            metadata.WhamFullParity = dto.whamFullParity;
            metadata.WhamInferencePolicy = dto.whamInferencePolicy;
            metadata.WhamMissingAssetBehavior = dto.whamMissingAssetBehavior;
            metadata.SmplInitializationSource = dto.smplInitializationSource;
            metadata.SmplInitializationFrame = dto.smplInitializationFrame;
            metadata.MediapipeSeedTrackerReset = dto.mediapipeSeedTrackerReset;

            if (dto.fusion != null)
            {
                metadata.Fusion = new VideoFusionMetadata
                {
                    CameraModel = string.IsNullOrEmpty(dto.fusion.cameraModel) ? "orthographic" : dto.fusion.cameraModel,
                    WindowSeconds = dto.fusion.windowSeconds,
                    WeightsVersion = dto.fusion.weightsVersion,
                    MeanReprojectionError = dto.fusion.meanReprojectionError,
                    MeanWhamCorrection = dto.fusion.meanWhamCorrection
                };
            }
            metadata.Wham = ConvertBackendSourceMetadata(dto.wham);
            metadata.MediaPipe = ConvertBackendSourceMetadata(dto.mediapipe);
            metadata.RTMPose = ConvertBackendSourceMetadata(dto.rtmpose);
            return metadata;
        }

        private static VideoOptionalStageDiagnostic[] ConvertOptionalStageDiagnostics(
            OptionalStageDiagnosticJsonDto[] diagnostics)
        {
            if (diagnostics == null || diagnostics.Length == 0)
                return Array.Empty<VideoOptionalStageDiagnostic>();

            var result = new List<VideoOptionalStageDiagnostic>(diagnostics.Length);
            for (int i = 0; i < diagnostics.Length; i++)
            {
                OptionalStageDiagnosticJsonDto dto = diagnostics[i];
                if (dto == null || string.IsNullOrEmpty(dto.stage)) continue;
                result.Add(new VideoOptionalStageDiagnostic
                {
                    Stage = dto.stage,
                    Status = string.IsNullOrEmpty(dto.status) ? "unknown" : dto.status,
                    Reason = dto.reason ?? string.Empty
                });
            }
            return result.ToArray();
        }

        private static VideoBackendSourceMetadata ConvertBackendSourceMetadata(BackendSourceMetadataJsonDto dto)
        {
            if (dto == null) return null;
            return new VideoBackendSourceMetadata
            {
                Implementation = dto.implementation,
                ModelPath = dto.modelPath,
                Checkpoint = dto.checkpoint,
                ModelHash = dto.modelHash,
                CoordinateSystem = dto.coordinateSystem,
                ObservationCount = dto.observationCount,
                Error = dto.error
            };
        }

        private static Quaternion[,] ParseLocalRotationsFallback(string jsonText, int frames, int jointCount)
        {
            int rotStart = jsonText.IndexOf("\"localRotations\"", StringComparison.OrdinalIgnoreCase);
            if (rotStart < 0) return null;

            var result = new Quaternion[frames, jointCount];
            var matches = Regex.Matches(
                jsonText.Substring(rotStart),
                @"\{[^{}]*""x""\s*:\s*([-\d\.eE]+)[^{}]*""y""\s*:\s*([-\d\.eE]+)[^{}]*""z""\s*:\s*([-\d\.eE]+)[^{}]*""w""\s*:\s*([-\d\.eE]+)[^{}]*\}"
            );

            int expected = frames * jointCount;
            if (matches.Count < expected)
            {
                return null;
            }

            for (int t = 0; t < frames; t++)
            {
                for (int j = 0; j < jointCount; j++)
                {
                    var m = matches[t * jointCount + j];
                    float x = float.Parse(m.Groups[1].Value, CultureInfo.InvariantCulture);
                    float y = float.Parse(m.Groups[2].Value, CultureInfo.InvariantCulture);
                    float z = float.Parse(m.Groups[3].Value, CultureInfo.InvariantCulture);
                    float w = float.Parse(m.Groups[4].Value, CultureInfo.InvariantCulture);
                    result[t, j] = new Quaternion(x, y, z, w);
                }
            }
            return result;
        }

        private static MotionDataJsonDto ParseMotionJsonDtoFallback(string jsonText)
        {
            var dto = new MotionDataJsonDto
            {
                frames = ExtractInt(jsonText, "frames", 0),
                jointCount = ExtractInt(jsonText, "jointCount", SmplxJointDefinitions.JointCount),
                frameRate = ExtractFloat(jsonText, "frameRate", 30.0f),
                sourceVideoPath = ExtractString(jsonText, "sourceVideoPath", null),
                videoWidth = ExtractInt(jsonText, "videoWidth", 0),
                videoHeight = ExtractInt(jsonText, "videoHeight", 0),
                videoFps = ExtractFloat(jsonText, "videoFps", 0f),
                overlayVideoPath = ExtractString(jsonText, "overlayVideoPath", null),
                detectorName = ExtractString(jsonText, "detectorName", "mediapipe"),
                backendRequested = ExtractString(jsonText, "backendRequested", null),
                backendActual = ExtractString(jsonText, "backendActual", null),
                fusionMode = ExtractString(jsonText, "fusionMode", null),
                backendFallback = ExtractBool(jsonText, "backendFallback", false),
                fallbackFrom = ExtractString(jsonText, "fallbackFrom", null),
                fallbackReason = ExtractString(jsonText, "fallbackReason", null),
                officialRunnerStatus = ExtractString(jsonText, "officialRunnerStatus", null),
                hmr2ImageFeaturesStatus = ExtractString(jsonText, "hmr2ImageFeaturesStatus", null),
                vitpose2DStatus = ExtractString(jsonText, "vitpose2DStatus", null),
                imageFeatureRunner = ExtractString(jsonText, "imageFeatureRunner", null),
                imageFeatureRunnerStatus = ExtractString(jsonText, "imageFeatureRunnerStatus", null),
                imageFeatureRunnerFeatureDim = ExtractInt(jsonText, "imageFeatureRunnerFeatureDim", 0),
                imageFeatureInputFrameCount = ExtractInt(jsonText, "imageFeatureInputFrameCount", 0),
                imageFeatureAdoptedFrameCount = ExtractInt(jsonText, "imageFeatureAdoptedFrameCount", 0),
                imageFeatureRequestedRunner = ExtractString(
                    jsonText,
                    "imageFeatureRequestedRunner",
                    ExtractString(jsonText, "requestedRunner", null)),
                imageFeatureStageStatus = ExtractString(
                    jsonText,
                    "imageFeatureStageStatus",
                    ExtractString(jsonText, "status", null)),
                imageFeatureRuntime = ExtractString(
                    jsonText,
                    "imageFeatureRuntime",
                    ExtractString(jsonText, "runtimeKind", null)),
                imageFeatureCheckpointPath = ExtractString(
                    jsonText,
                    "imageFeatureCheckpointPath",
                    ExtractString(jsonText, "checkpointPath", null)),
                imageFeatureCheckpointSha256 = ExtractString(
                    jsonText,
                    "imageFeatureCheckpointSha256",
                    ExtractString(jsonText, "checkpointSha256", null)),
                imageFeatureErrorCode = ExtractString(
                    jsonText,
                    "imageFeatureErrorCode",
                    ExtractString(jsonText, "errorCode", null)),
                imageFeatureNextAction = ExtractString(
                    jsonText,
                    "imageFeatureNextAction",
                    ExtractString(jsonText, "nextAction", null)),
                imageFeatureFallback = ExtractString(jsonText, "imageFeatureFallback", null),
                overlayBackend = ExtractString(jsonText, "overlayBackend", null),
                overlaySource = ExtractString(jsonText, "overlaySource", null)
            };

            if (dto.frames <= 0) return null;

            // Parse rootPositions
            int rootStart = jsonText.IndexOf("\"rootPositions\"", StringComparison.OrdinalIgnoreCase);
            if (rootStart >= 0)
            {
                var posMatches = Regex.Matches(
                    jsonText.Substring(rootStart),
                    @"\{[^{}]*""x""\s*:\s*([-\d\.eE]+)[^{}]*""y""\s*:\s*([-\d\.eE]+)[^{}]*""z""\s*:\s*([-\d\.eE]+)[^{}]*\}"
                );
                int count = Math.Min(dto.frames, posMatches.Count);
                dto.rootPositions = new Vector3Dto[count];
                for (int i = 0; i < count; i++)
                {
                    var m = posMatches[i];
                    dto.rootPositions[i] = new Vector3Dto
                    {
                        x = float.Parse(m.Groups[1].Value, CultureInfo.InvariantCulture),
                        y = float.Parse(m.Groups[2].Value, CultureInfo.InvariantCulture),
                        z = float.Parse(m.Groups[3].Value, CultureInfo.InvariantCulture)
                    };
                }
            }

            // Parse flatLocalRotations
            int rotStart = jsonText.IndexOf("\"flatLocalRotations\"", StringComparison.OrdinalIgnoreCase);
            if (rotStart >= 0)
            {
                var rotMatches = Regex.Matches(
                    jsonText.Substring(rotStart),
                    @"\{[^{}]*""x""\s*:\s*([-\d\.eE]+)[^{}]*""y""\s*:\s*([-\d\.eE]+)[^{}]*""z""\s*:\s*([-\d\.eE]+)[^{}]*""w""\s*:\s*([-\d\.eE]+)[^{}]*\}"
                );
                int count = Math.Min(dto.frames * dto.jointCount, rotMatches.Count);
                dto.flatLocalRotations = new Vector4Dto[count];
                for (int i = 0; i < count; i++)
                {
                    var m = rotMatches[i];
                    dto.flatLocalRotations[i] = new Vector4Dto
                    {
                        x = float.Parse(m.Groups[1].Value, CultureInfo.InvariantCulture),
                        y = float.Parse(m.Groups[2].Value, CultureInfo.InvariantCulture),
                        z = float.Parse(m.Groups[3].Value, CultureInfo.InvariantCulture),
                        w = float.Parse(m.Groups[4].Value, CultureInfo.InvariantCulture)
                    };
                }
            }

            // Parse timestamps
            dto.timestamps = ExtractFloatArray(jsonText, "timestamps", dto.frames);
            dto.confidences = ExtractFloatArray(jsonText, "confidences", dto.frames);
            dto.optionalStageDiagnostics = ParseOptionalStageDiagnosticsFallback(jsonText);

            // Parse uncertaintyIntervals
            int uncStart = jsonText.IndexOf("\"uncertaintyIntervals\"", StringComparison.OrdinalIgnoreCase);
            if (uncStart >= 0)
            {
                var uncMatches = Regex.Matches(
                    jsonText.Substring(uncStart),
                    @"\{[^{}]*""startFrame""\s*:\s*(\d+)[^{}]*""endFrame""\s*:\s*(\d+)[^{}]*""reason""\s*:\s*""([^""]*)""[^{}]*""confidence""\s*:\s*([-\d\.eE]+)[^{}]*\}"
                );
                if (uncMatches.Count > 0)
                {
                    dto.uncertaintyIntervals = new UncertaintyIntervalJsonDto[uncMatches.Count];
                    for (int i = 0; i < uncMatches.Count; i++)
                    {
                        var m = uncMatches[i];
                        var actMatch = Regex.Match(m.Value, @"""recommendedAction""\s*:\s*""([^""]*)""");
                        dto.uncertaintyIntervals[i] = new UncertaintyIntervalJsonDto
                        {
                            startFrame = int.Parse(m.Groups[1].Value),
                            endFrame = int.Parse(m.Groups[2].Value),
                            reason = m.Groups[3].Value,
                            confidence = float.Parse(m.Groups[4].Value, CultureInfo.InvariantCulture),
                            recommendedAction = actMatch.Success ? actMatch.Groups[1].Value : ""
                        };
                    }
                }
            }

            return dto;
        }

        private static OptionalStageDiagnosticJsonDto[] ParseOptionalStageDiagnosticsFallback(string jsonText)
        {
            if (string.IsNullOrEmpty(jsonText)) return null;
            int diagnosticsStart = jsonText.IndexOf("\"optionalStageDiagnostics\"", StringComparison.OrdinalIgnoreCase);
            if (diagnosticsStart < 0) return null;
            int arrayStart = jsonText.IndexOf('[', diagnosticsStart);
            if (arrayStart < 0) return null;
            int arrayEnd = jsonText.IndexOf(']', arrayStart);
            if (arrayEnd < 0) return null;

            string payload = jsonText.Substring(arrayStart, arrayEnd - arrayStart + 1);
            var matches = Regex.Matches(payload, @"\{[^{}]*\}");
            if (matches.Count == 0) return null;
            var result = new List<OptionalStageDiagnosticJsonDto>();
            for (int i = 0; i < matches.Count; i++)
            {
                string item = matches[i].Value;
                string stage = ExtractString(item, "stage", null);
                if (string.IsNullOrEmpty(stage)) continue;
                result.Add(new OptionalStageDiagnosticJsonDto
                {
                    stage = stage,
                    status = ExtractString(item, "status", "unknown"),
                    reason = ExtractString(item, "reason", string.Empty)
                });
            }
            return result.Count > 0 ? result.ToArray() : null;
        }

        private static float[] ExtractFloatArray(string json, string key, int maxCount)
        {
            int idx = json.IndexOf($"\"{key}\"", StringComparison.OrdinalIgnoreCase);
            if (idx < 0) return null;

            int arrStart = json.IndexOf('[', idx);
            if (arrStart < 0) return null;

            int arrEnd = json.IndexOf(']', arrStart);
            if (arrEnd < 0) return null;

            string sub = json.Substring(arrStart + 1, arrEnd - arrStart - 1);
            var matches = Regex.Matches(sub, @"[-\d\.eE]+");
            if (matches.Count == 0) return null;

            int count = Math.Min(maxCount, matches.Count);
            float[] result = new float[count];
            for (int i = 0; i < count; i++)
            {
                result[i] = float.Parse(matches[i].Value, CultureInfo.InvariantCulture);
            }
            return result;
        }

        private static string SafeGetDataPath()
        {
            try
            {
                return Application.dataPath;
            }
            catch
            {
                return null;
            }
        }

        private static bool IsWindowsPlatform
        {
            get
            {
                try
                {
                    return Application.platform == RuntimePlatform.WindowsEditor || Application.platform == RuntimePlatform.WindowsPlayer;
                }
                catch
                {
                    return Environment.OSVersion.Platform == PlatformID.Win32NT;
                }
            }
        }

#pragma warning disable CS0649
        [Serializable]
        private class MotionDataJsonDto
        {
            public int frames;
            public float frameRate;
            public int jointCount;
            public string[] jointNames;
            public Vector3Dto[] rootPositions;
            public Vector4Dto[] flatLocalRotations;
            public float[] timestamps;
            public float[] confidences;
            public string sourceVideoPath;
            public int videoWidth;
            public int videoHeight;
            public float videoFps;
            public string overlayVideoPath;
            public string detectorName;
            public string backendRequested;
            public string backendActual;
            public string fusionMode;
            public bool backendFallback;
            public string fallbackFrom;
            public string fallbackReason;
            public string officialRunnerStatus;
            public string hmr2ImageFeaturesStatus;
            public string vitpose2DStatus;
            // Flat image-feature provenance is emitted alongside the nested
            // backend metadata so old result consumers can still display the
            // selected runner without knowing adapter-specific payloads.
            public string imageFeatureRunner;
            public string imageFeatureRunnerStatus;
            public int imageFeatureRunnerFeatureDim;
            public int imageFeatureInputFrameCount;
            public int imageFeatureAdoptedFrameCount;
            public string imageFeatureRequestedRunner;
            public string imageFeatureStageStatus;
            public string imageFeatureRuntime;
            public string imageFeatureCheckpointPath;
            public string imageFeatureCheckpointSha256;
            public string imageFeatureErrorCode;
            public string imageFeatureNextAction;
            public ImageFeatureRunnerMetadataJsonDto imageFeatureStage;
            public ImageFeatureRunnerMetadataJsonDto imageFeatureRunnerMetadata;
            public int[] imageFeatureInputFrameIndices;
            public int[] imageFeatureAdoptedFrameIndices;
            public string imageFeatureFallback;
            public string overlayBackend;
            public string overlaySource;
            public bool inPlace;
            public bool smoothed;
            public bool footLocking;
            public UncertaintyIntervalJsonDto[] uncertaintyIntervals;
            public OptionalStageDiagnosticJsonDto[] optionalStageDiagnostics;
            public BackendMetadataJsonDto backendMetadata;
        }

        [Serializable]
        private class UncertaintyIntervalJsonDto
        {
            public int startFrame;
            public int endFrame;
            public string reason;
            public float confidence;
            public string recommendedAction;
            public string primaryHypothesis;
            public string[] alternativeHypotheses;
            public float maxJointDisagreementMeters;
        }

        [Serializable]
        private class OptionalStageDiagnosticJsonDto
        {
            public string stage;
            public string status;
            public string reason;
        }

        [Serializable]
        private class BackendMetadataJsonDto
        {
            public int schemaVersion;
            public string backendRequested;
            public string backendActual;
            public string fusionMode;
            public bool backendFallback;
            public string fallbackFrom;
            public string fallbackReason;
            public string officialRunnerStatus;
            public string hmr2ImageFeaturesStatus;
            public string vitpose2DStatus;
            public string overlayBackend;
            public string overlaySource;
            public string coordinateSystem;
            public bool worldMotionAvailable;
            public bool scaleIsRelative = true;
            public bool fusionAvailable;
            public string fusionError;
            public string primaryBackend;
            public string[] auxiliaryBackends;
            public int qualityFrameFallbackCount;
            public string qualityFallbackReason;
            public string whamAssetDiagnostic;
            public string[] whamMissingStages;
            public string selectedBackend;
            public bool backendReady;
            public string preflightStatus;
            public string preflightPhase;
            public string preflightError;
            public int preflightFrames;
            public string[] missingAssets;
            public string[] incompatibleAssets;
            public string[] assetDiagnostics;
            public string[] missingOptionalAssets;
            public string[] optionalErrors;
            public OptionalStageDiagnosticJsonDto[] optionalStageDiagnostics;
            public string imageFeaturePreprocess;
            public bool imageFeatureRunnerActive;
            public string imageFeatureRunnerStatus;
            public string imageFeatureRunner;
            public string imageFeatureRunnerModule;
            public int imageFeatureRunnerFeatureDim;
            public int imageFeatureInputFrameCount;
            public int imageFeatureAdoptedFrameCount;
            public string imageFeatureRequestedRunner;
            public string imageFeatureStageStatus;
            public string imageFeatureRuntime;
            public string imageFeatureCheckpointPath;
            public string imageFeatureCheckpointSha256;
            public string imageFeatureErrorCode;
            public string imageFeatureNextAction;
            public int[] imageFeatureInputFrameIndices;
            public int[] imageFeatureAdoptedFrameIndices;
            public string imageFeatureRunnerError;
            public string imageFeatureFallback;
            public string imageFeatureFallbackReason;
            public bool imageFeatureFallbackConsumed;
            public ImageFeatureRunnerMetadataJsonDto imageFeatureStage;
            public ImageFeatureRunnerMetadataJsonDto imageFeatureRunnerMetadata;
            public string cameraMotionRunner;
            public bool dpvoVerified;
            public CameraMotionProvenanceJsonDto cameraMotionProvenance;
            public string whamPreprocessManifestPath;
            public string[] runtimeWarnings;
            public bool whamFullParity;
            public string whamInferencePolicy;
            public string whamMissingAssetBehavior;
            public string smplInitializationSource;
            public int smplInitializationFrame = -1;
            public bool mediapipeSeedTrackerReset;
            public VideoFusionMetadataJsonDto fusion;
            public BackendSourceMetadataJsonDto wham;
            public BackendSourceMetadataJsonDto mediapipe;
            public BackendSourceMetadataJsonDto rtmpose;
        }

        [Serializable]
        private class ImageFeatureRunnerMetadataJsonDto
        {
            public string requestedRunner;
            public string runner;
            public string status;
            public string runtime;
            public string runtimeKind;
            public string runtimePath;
            public string checkpointPath;
            public string checkpointSha256;
            public int featureDim;
            public int frameCount;
            public int inputFrameCount;
            public int acceptedFrameCount;
            public int adoptedFrameCount;
            public string modelFactory;
            public string errorCode;
            public string error;
            public string fallbackReason;
            public string nextAction;
            public int[] inputFrameIndices;
            public int[] adoptedFrameIndices;
        }

        [Serializable]
        private class VideoFusionMetadataJsonDto
        {
            public string cameraModel;
            public float windowSeconds;
            public int weightsVersion;
            public float meanReprojectionError;
            public float meanWhamCorrection;
        }

        [Serializable]
        private class BackendSourceMetadataJsonDto
        {
            public string implementation;
            public string modelPath;
            public string checkpoint;
            public string modelHash;
            public string coordinateSystem;
            public int observationCount;
            public string error;
        }

        [Serializable]
        private class CameraMotionProvenanceJsonDto
        {
            public string source;
            public bool verified;
            public string runner;
            public string asset;
        }

        [Serializable]
        private struct Vector3Dto
        {
            public float x;
            public float y;
            public float z;
        }

        [Serializable]
        private struct Vector4Dto
        {
            public float x;
            public float y;
            public float z;
            public float w;
        }
#pragma warning restore CS0649

        #endregion

        #region Python Runtime Detection

        /// <summary>
        /// Detects available Python runtimes, testing virtual environments, PATH commands,
        /// and probing whether required packages (mediapipe, opencv-python, numpy, scipy) are installed.
        /// </summary>
        public static PythonRuntimeInfo DetectPythonRuntime(string explicitPath = null)
        {
            var candidates = FindPotentialPythonPaths(explicitPath);
            PythonRuntimeInfo bestFallback = null;

            foreach (var candidate in candidates)
            {
                var info = ProbePythonExecutable(candidate);
                if (info.IsAvailable)
                {
                    if (info.IsFullyConfigured)
                    {
                        return info; // Found complete runtime with MediaPipe and OpenCV!
                    }
                    if (bestFallback == null)
                    {
                        bestFallback = info;
                    }
                }
            }

            if (bestFallback != null)
            {
                bestFallback.ErrorMessage = TexMotionLocalization.Tr(
                    TexMotionLocalization.RuntimeDependenciesMissing);
                return bestFallback;
            }

            return new PythonRuntimeInfo
            {
                IsAvailable = false,
                ErrorMessage = TexMotionLocalization.Tr(TexMotionLocalization.NoPythonRuntimeDetected)
            };
        }

        /// <summary>
        /// Async wrapper for Python runtime detection.
        /// </summary>
        public static Task<PythonRuntimeInfo> DetectPythonRuntimeAsync(string explicitPath = null, CancellationToken cancellationToken = default)
        {
            return Task.Run(() => DetectPythonRuntime(explicitPath), cancellationToken);
        }

        /// <summary>
        /// Gathers potential Python executable paths by inspecting settings, virtual environments,
        /// PATH, and standard Windows/Unix installation directories.
        /// </summary>
        public static List<string> FindPotentialPythonPaths(string customExecutablePath = null)
        {
            var paths = new List<string>();

            // 1. Explicit argument
            if (!string.IsNullOrEmpty(customExecutablePath))
            {
                paths.Add(customExecutablePath);
            }

            // 2. TexMotionSettings stored preference
            try
            {
                string settingsPath = TexMotionSettings.instance?.GetEffectiveVideoPythonExecutablePath();
                if (!string.IsNullOrEmpty(settingsPath) && !paths.Contains(settingsPath))
                {
                    paths.Add(settingsPath);
                }
            }
            catch {}

            // 3. Active virtual environment or Conda environment variables
            string venvEnv = Environment.GetEnvironmentVariable("VIRTUAL_ENV");
            if (!string.IsNullOrEmpty(venvEnv))
            {
                string venvPy = Path.Combine(venvEnv, IsWindowsPlatform ? "Scripts\\python.exe" : "bin/python");
                if (File.Exists(venvPy) && !paths.Contains(venvPy)) paths.Add(venvPy);
            }

            string condaPrefix = Environment.GetEnvironmentVariable("CONDA_PREFIX");
            if (!string.IsNullOrEmpty(condaPrefix))
            {
                string condaPy = Path.Combine(condaPrefix, IsWindowsPlatform ? "python.exe" : "bin/python");
                if (File.Exists(condaPy) && !paths.Contains(condaPy)) paths.Add(condaPy);
            }

            // 4. Project-local virtual environments (.texmotion-venv, .venv,
            // venv, env, TexMotion-venv)
            var searchRoots = new List<string> { Directory.GetCurrentDirectory() };
            string dataPath = SafeGetDataPath();
            if (!string.IsNullOrEmpty(dataPath))
            {
                searchRoots.Add(dataPath);
                string parent = Path.GetDirectoryName(dataPath);
                if (!string.IsNullOrEmpty(parent)) searchRoots.Add(parent);
            }

            string[] venvFolderNames = new[] { ".texmotion-venv", ".venv", "venv", "env", "TexMotion-venv" };
            foreach (var root in searchRoots)
            {
                if (string.IsNullOrEmpty(root) || !Directory.Exists(root)) continue;

                foreach (var venvName in venvFolderNames)
                {
                    string venvDir = Path.Combine(root, venvName);
                    if (!Directory.Exists(venvDir)) continue;

                    string pyWin = Path.Combine(venvDir, "Scripts", "python.exe");
                    string pyUnix = Path.Combine(venvDir, "bin", "python");

                    if (File.Exists(pyWin) && !paths.Contains(pyWin)) paths.Add(pyWin);
                    if (File.Exists(pyUnix) && !paths.Contains(pyUnix)) paths.Add(pyUnix);
                }
            }

            // Keep the uv-managed project environment discoverable even when Unity's
            // current working directory is the package cache rather than the project
            // root. This also preserves the legacy venv/conda/PATH candidates below.
            try
            {
                foreach (string localPython in PythonEnvironmentManager.FindLocalPythonPaths())
                {
                    if (!string.IsNullOrEmpty(localPython) && !paths.Contains(localPython)) paths.Add(localPython);
                }
            }
            catch { }

            // 5. System PATH commands & standard installation folders
            if (IsWindowsPlatform)
            {
                if (!paths.Contains("python")) paths.Add("python");
                if (!paths.Contains("py")) paths.Add("py");
                if (!paths.Contains("python3")) paths.Add("python3");

                // Check standard Windows AppData and Program Files paths
                string localAppData = Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData);
                string[] versions = new[] { "Python311", "Python312", "Python310", "Python313", "Python39" };

                foreach (var v in versions)
                {
                    string appDataPy = Path.Combine(localAppData, "Programs", "Python", v, "python.exe");
                    if (File.Exists(appDataPy) && !paths.Contains(appDataPy)) paths.Add(appDataPy);

                    string progPy = $@"C:\Program Files\{v}\python.exe";
                    if (File.Exists(progPy) && !paths.Contains(progPy)) paths.Add(progPy);

                    string rootPy = $@"C:\{v}\python.exe";
                    if (File.Exists(rootPy) && !paths.Contains(rootPy)) paths.Add(rootPy);
                }

                // Conda in UserProfile
                string userProfile = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);
                string[] condaFolders = new[] { "miniconda3", "anaconda3", "miniforge3" };
                foreach (var c in condaFolders)
                {
                    string p = Path.Combine(userProfile, c, "python.exe");
                    if (File.Exists(p) && !paths.Contains(p)) paths.Add(p);
                }
            }
            else
            {
                if (!paths.Contains("python3")) paths.Add("python3");
                if (!paths.Contains("python")) paths.Add("python");

                string[] unixPaths = new[]
                {
                    "/usr/bin/python3",
                    "/usr/local/bin/python3",
                    "/opt/homebrew/bin/python3"
                };
                foreach (var p in unixPaths)
                {
                    if (File.Exists(p) && !paths.Contains(p)) paths.Add(p);
                }
            }

            return paths;
        }

        /// <summary>
        /// Probes a specific Python executable and inspects its version, virtualenv status,
        /// and package availability for mediapipe, opencv, numpy, and scipy.
        /// </summary>
        public static PythonRuntimeInfo ProbePythonExecutable(string pythonExe, int timeoutMs = 4000)
        {
            var info = new PythonRuntimeInfo { ExecutablePath = pythonExe };
            if (string.IsNullOrEmpty(pythonExe))
            {
                info.IsAvailable = false;
                info.ErrorMessage = TexMotionLocalization.Tr(TexMotionLocalization.PythonExecutablePathEmpty);
                return info;
            }

            // Check if absolute path exists on disk
            if (pythonExe.Contains(Path.DirectorySeparatorChar.ToString()) || pythonExe.Contains(Path.AltDirectorySeparatorChar.ToString()))
            {
                if (!File.Exists(pythonExe))
                {
                    info.IsAvailable = false;
                    info.ErrorMessage = TexMotionLocalization.TrFormat(
                        TexMotionLocalization.PythonFileNotFound,
                        pythonExe);
                    return info;
                }
            }

            // Base64 encoded probe script to avoid escaping/multiline SyntaxError across platforms:
            // import sys, json, importlib.util
            // torch_spec = bool(importlib.util.find_spec('torch'))
            // cuda_avail = False
            // if torch_spec:
            //     try:
            //         import torch
            //         cuda_avail = bool(torch.cuda.is_available())
            //     except Exception:
            //         pass
            // print(json.dumps({
            //     'version': sys.version.split()[0],
            //     'executable': sys.executable,
            //     'prefix': sys.prefix,
            //     'is_venv': sys.prefix != getattr(sys, 'base_prefix', sys.prefix),
            //     'mp': bool(importlib.util.find_spec('mediapipe')),
            //     'cv': bool(importlib.util.find_spec('cv2')),
            //     'np': bool(importlib.util.find_spec('numpy')),
            //     'sp': bool(importlib.util.find_spec('scipy')),
            //     'ort': bool(importlib.util.find_spec('onnxruntime')),
            //     'torch': torch_spec,
            //     'cuda': cuda_avail
            // }))
            const string probeScriptB64 =
                "aW1wb3J0IHN5cywganNvbiwgaW1wb3J0bGliLnV0aWwKdG9yY2hfc3BlYyA9IGJvb2woaW1wb3J0" +
                "bGliLnV0aWwuZmluZF9zcGVjKCd0b3JjaCcpKQpjdWRhX2F2YWlsID0gRmFsc2UKaWYgdG9yY2hf" +
                "c3BlYzoKICAgIHRyeToKICAgICAgICBpbXBvcnQgdG9yY2gKICAgICAgICBjdWRhX2F2YWlsID0g" +
                "Ym9vbCh0b3JjaC5jdWRhLmlzX2F2YWlsYWJsZSgpKQogICAgZXhjZXB0IEV4Y2VwdGlvbjoKICAg" +
                "ICAgICBwYXNzCnByaW50KGpzb24uZHVtcHMoewogICAgJ3ZlcnNpb24nOiBzeXMudmVyc2lvbi5z" +
                "cGxpdCgpWzBdLAogICAgJ2V4ZWN1dGFibGUnOiBzeXMuZXhlY3V0YWJsZSwKICAgICdwcmVmaXgn" +
                "OiBzeXMucHJlZml4LAogICAgJ2lzX3ZlbnYnOiBzeXMucHJlZml4ICE9IGdldGF0dHIoc3lzLCAn" +
                "YmFzZV9wcmVmaXgnLCBzeXMucHJlZml4KSwKICAgICdtcCc6IGJvb2woaW1wb3J0bGliLnV0aWwu" +
                "ZmluZF9zcGVjKCdtZWRpYXBpcGUnKSksCiAgICAnY3YnOiBib29sKGltcG9ydGxpYi51dGlsLmZp" +
                "bmRfc3BlYygnY3YyJykpLAogICAgJ25wJzogYm9vbChpbXBvcnRsaWIudXRpbC5maW5kX3NwZWMo" +
                "J251bXB5JykpLAogICAgJ3NwJzogYm9vbChpbXBvcnRsaWIudXRpbC5maW5kX3NwZWMoJ3NjaXB5" +
                "JykpLAogICAgJ29ydCc6IGJvb2woaW1wb3J0bGliLnV0aWwuZmluZF9zcGVjKCdvbm54cnVudGlt" +
                "ZScpKSwKICAgICd0b3JjaCc6IHRvcmNoX3NwZWMsCiAgICAnY3VkYSc6IGN1ZGFfYXZhaWwKfSkp";

            var psi = new ProcessStartInfo
            {
                FileName = pythonExe,
                Arguments = $"-c \"import base64; exec(base64.b64decode('{probeScriptB64}'))\"",
                UseShellExecute = false,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                CreateNoWindow = true,
                StandardOutputEncoding = Encoding.UTF8,
                StandardErrorEncoding = Encoding.UTF8
            };

            try
            {
                using var process = Process.Start(psi);
                if (process == null)
                {
                    info.IsAvailable = false;
                    info.ErrorMessage = TexMotionLocalization.Tr(TexMotionLocalization.PythonProcessLaunchFailed);
                    return info;
                }

                string stdout = process.StandardOutput.ReadToEnd();
                string stderr = process.StandardError.ReadToEnd();

                if (!process.WaitForExit(timeoutMs))
                {
                    KillProcessTree(process);
                    info.IsAvailable = false;
                    info.ErrorMessage = TexMotionLocalization.Tr(TexMotionLocalization.PythonProbeTimedOut);
                    return info;
                }

                if (process.ExitCode != 0)
                {
                    info.IsAvailable = false;
                    info.ErrorMessage = TexMotionLocalization.TrFormat(
                        TexMotionLocalization.PythonProcessExited,
                        process.ExitCode,
                        stderr);
                    return info;
                }

                string jsonLine = stdout.Trim();
                if (jsonLine.StartsWith("{") && jsonLine.EndsWith("}"))
                {
                    PythonProbeDto probeDto = null;
                    try
                    {
                        probeDto = JsonUtility.FromJson<PythonProbeDto>(jsonLine);
                    }
                    catch
                    {
                    }

                    if (probeDto != null && !string.IsNullOrEmpty(probeDto.version))
                    {
                        info.IsAvailable = true;
                        info.Version = probeDto.version;
                        info.ExecutablePath = !string.IsNullOrEmpty(probeDto.executable) ? probeDto.executable : pythonExe;
                        info.EnvironmentPath = probeDto.prefix;
                        info.IsVirtualEnv = probeDto.is_venv;
                        info.HasMediaPipe = probeDto.mp;
                        info.HasOpenCV = probeDto.cv;
                        info.HasNumPy = probeDto.np;
                        info.HasSciPy = probeDto.sp;
                        info.HasOnnxRuntime = probeDto.ort;
                        info.HasTorch = probeDto.torch;
                        info.HasCuda = probeDto.cuda;
                        return info;
                    }

                    // Fallback string parsing if JsonUtility is unavailable
                    info.IsAvailable = true;
                    info.Version = ExtractString(jsonLine, "version", "3.x");
                    info.ExecutablePath = ExtractString(jsonLine, "executable", pythonExe);
                    info.EnvironmentPath = ExtractString(jsonLine, "prefix", null);
                    info.IsVirtualEnv = jsonLine.Contains("\"is_venv\": true");
                    info.HasMediaPipe = jsonLine.Contains("\"mp\": true");
                    info.HasOpenCV = jsonLine.Contains("\"cv\": true");
                    info.HasNumPy = jsonLine.Contains("\"np\": true");
                    info.HasSciPy = jsonLine.Contains("\"sp\": true");
                    info.HasOnnxRuntime = jsonLine.Contains("\"ort\": true");
                    info.HasTorch = jsonLine.Contains("\"torch\": true");
                    info.HasCuda = jsonLine.Contains("\"cuda\": true");
                    return info;
                }

                info.IsAvailable = true;
                info.Version = TexMotionLocalization.Tr(TexMotionLocalization.Unknown);
                return info;
            }
            catch (Exception ex)
            {
                info.IsAvailable = false;
                info.ErrorMessage = ex.Message;
                return info;
            }
        }

#pragma warning disable CS0649
        [Serializable]
        private class PythonProbeDto
        {
            public string version;
            public string executable;
            public string prefix;
            public bool is_venv;
            public bool mp;
            public bool cv;
            public bool np;
            public bool sp;
            public bool ort;
            public bool torch;
            public bool cuda;
        }
#pragma warning restore CS0649

        /// <summary>
        /// Automatically installs requirements via pip (mediapipe, opencv-python, numpy, scipy).
        /// </summary>
        public static async Task<bool> InstallRequirementsAsync(
            string pythonExe,
            string requirementsPath = null,
            IProgress<float> progress = null,
            Action<string> onLog = null,
            CancellationToken cancellationToken = default)
        {
            if (string.IsNullOrEmpty(pythonExe))
            {
                var runtime = DetectPythonRuntime();
                if (!runtime.IsAvailable)
                {
                    throw new InvalidOperationException(TexMotionLocalization.Tr(
                        TexMotionLocalization.RequirementsPythonMissing));
                }
                pythonExe = runtime.ExecutablePath;
            }

            if (string.IsNullOrEmpty(requirementsPath) || !File.Exists(requirementsPath))
            {
                requirementsPath = FindRequirementsPath();
            }

            string arguments;
            if (!string.IsNullOrEmpty(requirementsPath) && File.Exists(requirementsPath))
            {
                arguments = $"-m pip install -r \"{requirementsPath}\"";
            }
            else
            {
                arguments = "-m pip install \"mediapipe>=0.10.0\" \"opencv-python>=4.8.0\" \"numpy>=1.24.0\" \"scipy>=1.10.0\"";
            }

            var psi = new ProcessStartInfo
            {
                FileName = pythonExe,
                Arguments = arguments,
                UseShellExecute = false,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                CreateNoWindow = true,
                StandardOutputEncoding = Encoding.UTF8,
                StandardErrorEncoding = Encoding.UTF8
            };

            using var process = new Process { StartInfo = psi, EnableRaisingEvents = true };
            var tcs = new TaskCompletionSource<int>();
            process.Exited += (s, e) => tcs.TrySetResult(process.ExitCode);

            process.OutputDataReceived += (s, e) =>
            {
                if (e.Data != null)
                {
                    onLog?.Invoke(e.Data);
                }
            };

            process.ErrorDataReceived += (s, e) =>
            {
                if (e.Data != null)
                {
                    onLog?.Invoke(e.Data);
                }
            };

            using (cancellationToken.Register(() =>
            {
                KillProcessTree(process);
                tcs.TrySetCanceled(cancellationToken);
            }))
            {
                progress?.Report(0.1f);
                if (!process.Start())
                {
                    throw new InvalidOperationException(TexMotionLocalization.TrFormat(
                        TexMotionLocalization.PipProcessLaunchFailed,
                        pythonExe));
                }

                process.BeginOutputReadLine();
                process.BeginErrorReadLine();

                if (process.HasExited)
                {
                    tcs.TrySetResult(process.ExitCode);
                }

                int exitCode = await tcs.Task.ConfigureAwait(false);
                process.WaitForExit();

                progress?.Report(1.0f);
                return exitCode == 0;
            }
        }

        #endregion

        #region Script & Requirements Path Resolution

        /// <summary>
        /// Finds the absolute path to video_pose_extractor.py in package or Assets.
        /// </summary>
        public static string FindScriptPath()
        {
            // 1. Package cache path
            string p1 = Path.GetFullPath("Packages/com.k0ta0uchi.texmotion/Editor/Video/video_pose_extractor.py");
            if (File.Exists(p1)) return p1;

            // 2. Assets directory path
            string dataPath = SafeGetDataPath();
            if (!string.IsNullOrEmpty(dataPath))
            {
                string p2 = Path.GetFullPath(Path.Combine(dataPath, "TexMotion/Editor/Video/video_pose_extractor.py"));
                if (File.Exists(p2)) return p2;
            }

            // 3. Workspace relative path
            string p3 = Path.GetFullPath("Editor/Video/video_pose_extractor.py");
            if (File.Exists(p3)) return p3;

            // 4. AssetDatabase search
            try
            {
                string[] guids = AssetDatabase.FindAssets("video_pose_extractor");
                foreach (var guid in guids)
                {
                    string assetPath = AssetDatabase.GUIDToAssetPath(guid);
                    if (assetPath.EndsWith("video_pose_extractor.py", StringComparison.OrdinalIgnoreCase))
                    {
                        return Path.GetFullPath(assetPath);
                    }
                }
            }
            catch {}

            // 5. Recursive search in dataPath
            if (!string.IsNullOrEmpty(dataPath))
            {
                try
                {
                    string[] files = Directory.GetFiles(dataPath, "video_pose_extractor.py", SearchOption.AllDirectories);
                    if (files.Length > 0) return Path.GetFullPath(files[0]);
                }
                catch {}
            }

            return p3;
        }

        /// <summary>
        /// Finds the absolute path to requirements.txt in Editor/Video.
        /// </summary>
        public static string FindRequirementsPath()
        {
            string scriptPath = FindScriptPath();
            if (!string.IsNullOrEmpty(scriptPath))
            {
                string reqNextToScript = Path.Combine(Path.GetDirectoryName(scriptPath), "requirements.txt");
                if (File.Exists(reqNextToScript)) return reqNextToScript;
            }

            string p1 = Path.GetFullPath("Packages/com.k0ta0uchi.texmotion/Editor/Video/requirements.txt");
            if (File.Exists(p1)) return p1;

            string p2 = Path.GetFullPath("Editor/Video/requirements.txt");
            if (File.Exists(p2)) return p2;

            return p2;
        }

        #endregion

        #region Process Tree Termination

        /// <summary>
        /// Cleanly terminates a process and all of its spawned child processes.
        /// </summary>
        public static void KillProcessTree(Process process)
        {
            if (process == null) return;

            try
            {
                if (process.HasExited) return;

                int pid = process.Id;

                // Modern .NET Core Kill(entireProcessTree: true) via reflection
                try
                {
                    var killTreeMethod = typeof(Process).GetMethod("Kill", new[] { typeof(bool) });
                    if (killTreeMethod != null)
                    {
                        killTreeMethod.Invoke(process, new object[] { true });
                        return;
                    }
                }
                catch {}

                // Windows taskkill /PID {pid} /T /F
                if (IsWindowsPlatform)
                {
                    try
                    {
                        var psi = new ProcessStartInfo
                        {
                            FileName = "taskkill",
                            Arguments = $"/PID {pid} /T /F",
                            CreateNoWindow = true,
                            UseShellExecute = false,
                            RedirectStandardOutput = true,
                            RedirectStandardError = true
                        };
                        using var killProc = Process.Start(psi);
                        killProc?.WaitForExit(3000);
                    }
                    catch {}
                }

                if (!process.HasExited)
                {
                    process.Kill();
                }
            }
            catch (Exception ex)
            {
                Debug.LogWarning(TexMotionLocalization.TrFormat(
                    TexMotionLocalization.ProcessTerminationWarning,
                    ex.Message));
            }
        }

        #endregion
    }
}
