using System;
using UnityEngine;
using TexMotion.Runtime.Native;

namespace TexMotion.Runtime.Motion
{
    /// <summary>
    /// Represents motion extracted from video (SMPL-X22 joint hierarchy compatible),
    /// supporting timestamps, root positions, 22 joint local rotations, and frame/joint confidence metrics.
    /// Inherits from GeneratedMotionData for universal compatibility with AnimationClipBuilder and TexMotion pipelines.
    /// </summary>
    public class VideoMotionData : GeneratedMotionData
    {
        /// <summary>
        /// Exact timestamp in seconds for each frame.
        /// </summary>
        public float[] Timestamps { get; }

        /// <summary>
        /// Frame-level pose estimation confidence score [0.0, 1.0].
        /// </summary>
        public float[] Confidences { get; }

        /// <summary>
        /// Optional per-joint confidence score [frame, joint] in [0.0, 1.0].
        /// </summary>
        public float[,] JointConfidences { get; }

        /// <summary>
        /// Source video file path or identifier if available.
        /// </summary>
        public string SourceVideoPath { get; set; }

        /// <summary>
        /// Intervals of ambiguity or self-occlusion detected during extraction.
        /// </summary>
        public UncertaintyInterval[] UncertaintyIntervals { get; }

        /// <summary>
        /// Path to the preview overlay video (with the active detector's
        /// skeleton drawn) if generated.
        /// </summary>
        public string OverlayVideoPath { get; set; }

        /// <summary>
        /// Name of the pose detector/backend actually used for extraction
        /// (e.g. "mediapipe", "rtmpose", "wham", "synthetic").
        /// </summary>
        public string DetectorName { get; set; } = "mediapipe";

        /// <summary>
        /// Concrete backend selected after initialization/fallback resolution.
        /// This is separate from <see cref="DetectorName"/> so a fused overlay
        /// can keep WHAM as the primary backend while retaining its mode label.
        /// </summary>
        public string BackendActual { get; set; } = "mediapipe";

        /// <summary>
        /// Runtime status of the selected official/temporal runner.  This is
        /// execution provenance (for example "active" or "fallback"), not a
        /// claim that a checkpoint merely exists on disk.
        /// </summary>
        public string OfficialRunnerStatus { get; set; }

        /// <summary>
        /// Runtime status of the HMR2 image-feature/initial-SMPL stage.
        /// </summary>
        public string Hmr2ImageFeaturesStatus { get; set; }

        /// <summary>
        /// Runtime status of the ViTPose 2D detector stage.
        /// </summary>
        public string VitPose2DStatus { get; set; }

        /// <summary>
        /// Explicit fusion mode used by the extractor ("off" for legacy JSON).
        /// </summary>
        public string FusionMode { get; set; } = "off";

        /// <summary>
        /// Backend requested by the user before any automatic fallback.
        /// </summary>
        public string BackendRequested { get; set; } = "auto";

        /// <summary>
        /// Backend provenance when the requested backend was unavailable.
        /// Empty when no fallback occurred.
        /// </summary>
        public string BackendFallbackReason { get; set; }

        /// <summary>True when the extractor explicitly recorded a fallback.</summary>
        public bool BackendFallback { get; set; }

        /// <summary>Requested backend that led to a fallback, when available.</summary>
        public string FallbackFrom { get; set; }

        /// <summary>Alias used by the JSON contract and result-card diagnostics.</summary>
        public string FallbackReason
        {
            get => BackendFallbackReason;
            set => BackendFallbackReason = value;
        }

        /// <summary>Backend name used to render the overlay video.</summary>
        public string OverlayBackend { get; set; }

        /// <summary>Overlay provenance (for example fused_3d_projection).</summary>
        public string OverlaySource { get; set; }

        /// <summary>
        /// Structured per-backend/fusion metadata.  Unknown fields from newer
        /// extractor versions remain optional so old JSON continues to load.
        /// </summary>
        public VideoBackendMetadata BackendMetadata { get; set; }

        /// <summary>
        /// Flat, JSON-friendly diagnostics for optional WHAM stages.  The
        /// detailed adapter metadata remains available in BackendMetadata.
        /// </summary>
        public VideoOptionalStageDiagnostic[] OptionalStageDiagnostics { get; }

        /// <summary>
        /// True when extraction could not initialize the requested backend and
        /// explicitly switched to another backend.  Per-frame quality fallbacks
        /// (for example WHAM geometry gates selecting MediaPipe for a span) are
        /// represented by BackendMetadata.QualityFrameFallbackCount and must
        /// not relabel the whole clip as a backend initialization fallback.
        /// </summary>
        public bool UsedBackendFallback => BackendFallback;

        /// <summary>True when the output is the explicit WHAM + MediaPipe fusion.</summary>
        public bool IsWhamMediaPipeFusion => string.Equals(
            FusionMode, "wham_mediapipe", StringComparison.OrdinalIgnoreCase);

        /// <summary>
        /// True only when the persisted metadata proves that fusion actually
        /// ran.  A requested fusion mode can legitimately fall back to a
        /// single backend or fail during the temporal prepass.
        /// </summary>
        public bool IsActualWhamMediaPipeFusion =>
            IsWhamMediaPipeFusion &&
            !UsedBackendFallback &&
            string.Equals(BackendActual, "wham", StringComparison.OrdinalIgnoreCase) &&
            string.Equals(OverlayBackend, "wham_mediapipe", StringComparison.OrdinalIgnoreCase) &&
            OverlaySource != null &&
            OverlaySource.StartsWith("fused_3d_projection", StringComparison.OrdinalIgnoreCase) &&
            BackendMetadata != null &&
            (BackendMetadata.FusionAvailable || BackendMetadata.Fusion != null);

        /// <summary>
        /// Original video width in pixels.
        /// </summary>
        public int VideoWidth { get; set; }

        /// <summary>
        /// Original video height in pixels.
        /// </summary>
        public int VideoHeight { get; set; }

        /// <summary>
        /// Original video frame rate.
        /// </summary>
        public float VideoFps { get; set; }

        /// <summary>
        /// True if frame timestamps have variable spacing (VFR or dropped frames).
        /// </summary>
        public bool HasVariableTimestamps { get; }

        /// <summary>
        /// Mean confidence score across all frames.
        /// </summary>
        public float AverageConfidence { get; }

        /// <summary>
        /// Minimum confidence score across all frames.
        /// </summary>
        public float MinConfidence { get; }

        /// <summary>
        /// Maximum confidence score across all frames.
        /// </summary>
        public float MaxConfidence { get; }

        /// <summary>
        /// Total motion duration in seconds.
        /// </summary>
        public float Duration
        {
            get
            {
                if (Frames <= 0) return 0f;
                if (Timestamps != null && Timestamps.Length > 0)
                {
                    return Timestamps[Frames - 1] - Timestamps[0];
                }
                return FrameRate > 0f ? Frames / FrameRate : 0f;
            }
        }

        public VideoMotionData(
            int frames,
            int jointCount,
            Vector3[] rootPositions,
            Quaternion[,] localRotations,
            float frameRate = 30.0f,
            float[] timestamps = null,
            float[] confidences = null,
            float[,] jointConfidences = null,
            string sourceVideoPath = null,
            int videoWidth = 0,
            int videoHeight = 0,
            float videoFps = 0f,
            string overlayVideoPath = null,
            UncertaintyInterval[] uncertaintyIntervals = null,
            string detectorName = "mediapipe",
            string backendRequested = null,
            string backendFallbackReason = null,
            string backendActual = null,
            string fusionMode = null,
            bool backendFallback = false,
            string fallbackFrom = null,
            string overlayBackend = null,
            string overlaySource = null,
            VideoBackendMetadata backendMetadata = null,
            string officialRunnerStatus = null,
            string hmr2ImageFeaturesStatus = null,
            string vitPose2DStatus = null,
            VideoOptionalStageDiagnostic[] optionalStageDiagnostics = null)
            : base(frames, jointCount, rootPositions, localRotations, frameRate)
        {
            SourceVideoPath = sourceVideoPath;
            OverlayVideoPath = overlayVideoPath;
            DetectorName = !string.IsNullOrEmpty(detectorName) ? detectorName : "mediapipe";
            BackendRequested = !string.IsNullOrEmpty(backendRequested) ? backendRequested : DetectorName;
            BackendFallbackReason = backendFallbackReason;
            BackendActual = !string.IsNullOrEmpty(backendActual) ? backendActual : DetectorName;
            FusionMode = !string.IsNullOrEmpty(fusionMode) ? fusionMode : "off";
            // A reason can describe a per-frame quality gate (or another
            // diagnostic) while the requested backend still initialized and
            // produced the clip.  Keep the boolean tied to the explicit
            // backendFallback field so the HUD does not display "WHAM -> WHAM
            // fallback" for a quality warning.
            BackendFallback = backendFallback;
            FallbackFrom = fallbackFrom;
            OverlayBackend = !string.IsNullOrEmpty(overlayBackend) ? overlayBackend : DetectorName;
            OverlaySource = !string.IsNullOrEmpty(overlaySource) ? overlaySource : "backend_output";
            BackendMetadata = backendMetadata ?? VideoBackendMetadata.CreateLegacy(
                BackendRequested, BackendActual, FusionMode, BackendFallback,
                FallbackFrom, BackendFallbackReason, OverlayBackend, OverlaySource);
            OfficialRunnerStatus = officialRunnerStatus ?? BackendMetadata.OfficialRunnerStatus;
            Hmr2ImageFeaturesStatus = hmr2ImageFeaturesStatus ?? BackendMetadata.Hmr2ImageFeaturesStatus;
            VitPose2DStatus = vitPose2DStatus ?? BackendMetadata.VitPose2DStatus;
            OptionalStageDiagnostics = optionalStageDiagnostics ??
                BackendMetadata.OptionalStageDiagnostics ?? Array.Empty<VideoOptionalStageDiagnostic>();
            BackendMetadata.OfficialRunnerStatus = OfficialRunnerStatus;
            BackendMetadata.Hmr2ImageFeaturesStatus = Hmr2ImageFeaturesStatus;
            BackendMetadata.VitPose2DStatus = VitPose2DStatus;
            BackendMetadata.OptionalStageDiagnostics = OptionalStageDiagnostics;
            VideoWidth = videoWidth;
            VideoHeight = videoHeight;
            VideoFps = videoFps > 0f ? videoFps : frameRate;
            UncertaintyIntervals = uncertaintyIntervals ?? Array.Empty<UncertaintyInterval>();

            // Initialize or assign timestamps
            float timestampDt = frameRate > 0f ? 1.0f / frameRate : 1.0f / 30.0f;
            Timestamps = new float[frames];
            for (int i = 0; i < frames; i++)
            {
                float fallback = i * timestampDt;
                float value = timestamps != null && i < timestamps.Length
                    ? timestamps[i]
                    : fallback;
                if (float.IsNaN(value) || float.IsInfinity(value)) value = fallback;
                if (i > 0 && value < Timestamps[i - 1]) value = Timestamps[i - 1];
                Timestamps[i] = value;
            }
            HasVariableTimestamps = timestamps != null && timestamps.Length == frames &&
                CheckVariableTimestamps(Timestamps, frameRate);

            // Initialize or assign confidences
            Confidences = new float[frames];
            for (int i = 0; i < frames; i++)
            {
                float value = confidences == null
                    ? 1.0f
                    : i < confidences.Length
                    ? confidences[i]
                    : 0.0f;
                Confidences[i] = float.IsNaN(value) || float.IsInfinity(value)
                    ? 0.0f
                    : Mathf.Clamp01(value);
            }

            JointConfidences = jointConfidences;

            // Compute confidence statistics
            float sum = 0f;
            float min = float.MaxValue;
            float max = float.MinValue;
            for (int i = 0; i < frames; i++)
            {
                float c = Confidences[i];
                sum += c;
                if (c < min) min = c;
                if (c > max) max = c;
            }
            AverageConfidence = frames > 0 ? sum / frames : 0f;
            MinConfidence = frames > 0 ? min : 0f;
            MaxConfidence = frames > 0 ? max : 0f;

            // Ensure quaternion hemisphere continuity across all frames and joints
            EnforceQuaternionContinuity();
        }

        public VideoMotionData(
            int frames,
            Vector3[] rootPositions,
            Quaternion[,] localRotations,
            float frameRate = 30.0f,
            float[] timestamps = null,
            float[] confidences = null,
            float[,] jointConfidences = null,
            string sourceVideoPath = null)
            : this(frames, SmplxJointDefinitions.JointCount, rootPositions, localRotations, frameRate, timestamps, confidences, jointConfidences, sourceVideoPath)
        {
        }

        /// <summary>
        /// Gets exact timestamp in seconds for a given frame index.
        /// </summary>
        public override float GetTimestamp(int frameIndex)
        {
            if (Timestamps != null && frameIndex >= 0 && frameIndex < Timestamps.Length)
            {
                return Timestamps[frameIndex];
            }
            return base.GetTimestamp(frameIndex);
        }

        /// <summary>
        /// Gets frame-level confidence score in [0.0, 1.0].
        /// </summary>
        public float GetFrameConfidence(int frameIndex)
        {
            if (Confidences != null && frameIndex >= 0 && frameIndex < Confidences.Length)
            {
                return Confidences[frameIndex];
            }
            return 1.0f;
        }

        /// <summary>
        /// Gets joint-level confidence score in [0.0, 1.0], falling back to frame confidence if joint confidences not present.
        /// </summary>
        public float GetJointConfidence(int frameIndex, int jointIndex)
        {
            if (JointConfidences != null &&
                frameIndex >= 0 && frameIndex < JointConfidences.GetLength(0) &&
                jointIndex >= 0 && jointIndex < JointConfidences.GetLength(1))
            {
                return JointConfidences[frameIndex, jointIndex];
            }
            return GetFrameConfidence(frameIndex);
        }

        /// <summary>
        /// Checks whether a frame meets a given confidence threshold.
        /// </summary>
        public bool IsFrameReliable(int frameIndex, float minConfidence = 0.5f)
        {
            return GetFrameConfidence(frameIndex) >= minConfidence;
        }

        /// <summary>
        /// Smooths out low-confidence outlier frames by interpolating from adjacent reliable frames.
        /// </summary>
        public void InterpolateOutliers(float minConfidenceThreshold = 0.3f)
        {
            if (Frames <= 2 || LocalRotations == null) return;

            for (int t = 0; t < Frames; t++)
            {
                if (Confidences[t] >= minConfidenceThreshold) continue;

                // Find previous valid frame
                int prevIdx = -1;
                for (int p = t - 1; p >= 0; p--)
                {
                    if (Confidences[p] >= minConfidenceThreshold)
                    {
                        prevIdx = p;
                        break;
                    }
                }

                // Find next valid frame
                int nextIdx = -1;
                for (int n = t + 1; n < Frames; n++)
                {
                    if (Confidences[n] >= minConfidenceThreshold)
                    {
                        nextIdx = n;
                        break;
                    }
                }

                if (prevIdx >= 0 && nextIdx >= 0)
                {
                    float alpha = (float)(t - prevIdx) / (nextIdx - prevIdx);
                    if (RootPositions != null && RootPositions.Length == Frames)
                    {
                        RootPositions[t] = LerpVector3(RootPositions[prevIdx], RootPositions[nextIdx], alpha);
                    }
                    for (int j = 0; j < JointCount; j++)
                    {
                        LocalRotations[t, j] = SlerpQuaternion(LocalRotations[prevIdx, j], LocalRotations[nextIdx, j], alpha);
                    }
                }
                else if (prevIdx >= 0)
                {
                    if (RootPositions != null && RootPositions.Length == Frames)
                    {
                        RootPositions[t] = RootPositions[prevIdx];
                    }
                    for (int j = 0; j < JointCount; j++)
                    {
                        LocalRotations[t, j] = LocalRotations[prevIdx, j];
                    }
                }
                else if (nextIdx >= 0)
                {
                    if (RootPositions != null && RootPositions.Length == Frames)
                    {
                        RootPositions[t] = RootPositions[nextIdx];
                    }
                    for (int j = 0; j < JointCount; j++)
                    {
                        LocalRotations[t, j] = LocalRotations[nextIdx, j];
                    }
                }
            }
        }

        /// <summary>
        /// Validates structural integrity of motion data arrays against frame and joint dimensions.
        /// </summary>
        public bool Validate(out string errorMessage)
        {
            if (Frames <= 0)
            {
                errorMessage = "Frame count must be greater than zero.";
                return false;
            }
            if (JointCount != SmplxJointDefinitions.JointCount)
            {
                errorMessage = $"Joint count must be {SmplxJointDefinitions.JointCount}, but was {JointCount}.";
                return false;
            }
            if (RootPositions == null || RootPositions.Length != Frames)
            {
                errorMessage = $"Root positions array length ({RootPositions?.Length ?? 0}) does not match Frames ({Frames}).";
                return false;
            }
            if (LocalRotations == null || LocalRotations.GetLength(0) != Frames || LocalRotations.GetLength(1) != JointCount)
            {
                errorMessage = "Local rotations dimensions do not match [Frames, JointCount].";
                return false;
            }
            if (Timestamps != null && Timestamps.Length != Frames)
            {
                errorMessage = "Timestamps array length does not match Frames.";
                return false;
            }
            if (Confidences != null && Confidences.Length != Frames)
            {
                errorMessage = "Confidences array length does not match Frames.";
                return false;
            }

            errorMessage = null;
            return true;
        }

        /// <summary>
        /// Creates an empty VideoMotionData instance with default allocations.
        /// </summary>
        public static VideoMotionData CreateEmpty(int frames, float frameRate = 30.0f)
        {
            int jointCount = SmplxJointDefinitions.JointCount;
            var rootPositions = new Vector3[frames];
            var localRotations = new Quaternion[frames, jointCount];
            for (int t = 0; t < frames; t++)
            {
                for (int j = 0; j < jointCount; j++)
                {
                    localRotations[t, j] = Quaternion.identity;
                }
            }
            return new VideoMotionData(frames, jointCount, rootPositions, localRotations, frameRate);
        }

        /// <summary>
        /// Wraps or converts an existing GeneratedMotionData into VideoMotionData.
        /// </summary>
        public static VideoMotionData FromGeneratedMotion(
            GeneratedMotionData source,
            float[] confidences = null,
            string sourceVideoPath = null,
            string overlayVideoPath = null)
        {
            if (source == null) return null;

            var vmdSource = source as VideoMotionData;
            var vmd = new VideoMotionData(
                source.Frames,
                source.JointCount,
                source.RootPositions,
                source.LocalRotations,
                source.FrameRate,
                vmdSource?.Timestamps,
                confidences ?? vmdSource?.Confidences,
                null,
                sourceVideoPath ?? vmdSource?.SourceVideoPath,
                vmdSource?.VideoWidth ?? 0,
                vmdSource?.VideoHeight ?? 0,
                vmdSource?.VideoFps ?? 0f,
                overlayVideoPath ?? vmdSource?.OverlayVideoPath,
                vmdSource?.UncertaintyIntervals,
                vmdSource?.DetectorName,
                vmdSource?.BackendRequested,
                vmdSource?.BackendFallbackReason,
                vmdSource?.BackendActual,
                vmdSource?.FusionMode,
                vmdSource?.BackendFallback ?? false,
                vmdSource?.FallbackFrom,
                vmdSource?.OverlayBackend,
                vmdSource?.OverlaySource,
                vmdSource?.BackendMetadata,
                vmdSource?.OfficialRunnerStatus,
                vmdSource?.Hmr2ImageFeaturesStatus,
                vmdSource?.VitPose2DStatus,
                vmdSource?.OptionalStageDiagnostics
            );
            return vmd;
        }

        private static bool CheckVariableTimestamps(float[] timestamps, float frameRate)
        {
            if (timestamps == null || timestamps.Length <= 1) return false;
            float expectedDt = frameRate > 0f ? 1.0f / frameRate : 0.03333f;
            for (int i = 1; i < timestamps.Length; i++)
            {
                float dt = timestamps[i] - timestamps[i - 1];
                if (Math.Abs(dt - expectedDt) > 0.005f)
                {
                    return true;
                }
            }
            return false;
        }

        private static Vector3 LerpVector3(Vector3 a, Vector3 b, float t)
        {
            return new Vector3(
                a.x + (b.x - a.x) * t,
                a.y + (b.y - a.y) * t,
                a.z + (b.z - a.z) * t
            );
        }

        private static Quaternion SlerpQuaternion(Quaternion a, Quaternion b, float t)
        {
            float dot = a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
            if (dot < 0.0f)
            {
                b = new Quaternion(-b.x, -b.y, -b.z, -b.w);
                dot = -dot;
            }

            if (dot > 0.9995f)
            {
                return NormalizeQuaternion(new Quaternion(
                    a.x + (b.x - a.x) * t,
                    a.y + (b.y - a.y) * t,
                    a.z + (b.z - a.z) * t,
                    a.w + (b.w - a.w) * t
                ));
            }

            float theta = (float)Math.Acos(Math.Max(-1f, Math.Min(1f, dot)));
            float sinTheta = (float)Math.Sin(theta);
            if (Math.Abs(sinTheta) < 0.0001f)
            {
                return a;
            }

            float wa = (float)Math.Sin((1f - t) * theta) / sinTheta;
            float wb = (float)Math.Sin(t * theta) / sinTheta;

            return new Quaternion(
                wa * a.x + wb * b.x,
                wa * a.y + wb * b.y,
                wa * a.z + wb * b.z,
                wa * a.w + wb * b.w
            );
        }

        private static Quaternion NormalizeQuaternion(Quaternion q)
        {
            float mag = (float)Math.Sqrt(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w);
            if (mag > 0.00001f)
            {
                return new Quaternion(q.x / mag, q.y / mag, q.z / mag, q.w / mag);
            }
            return Quaternion.identity;
        }

        /// <summary>
        /// Checks if a frame falls within an uncertainty or occlusion interval.
        /// </summary>
        public bool HasUncertainty(int frame, out UncertaintyInterval interval)
        {
            interval = null;
            if (UncertaintyIntervals == null || UncertaintyIntervals.Length == 0) return false;
            for (int i = 0; i < UncertaintyIntervals.Length; i++)
            {
                var ui = UncertaintyIntervals[i];
                if (ui != null && ui.ContainsFrame(frame))
                {
                    interval = ui;
                    return true;
                }
            }
            return false;
        }

        /// <summary>
        /// Checks if a frame falls within an uncertainty interval, returning the reason string.
        /// </summary>
        public bool HasUncertainty(int frame, out string reason)
        {
            reason = string.Empty;
            if (HasUncertainty(frame, out UncertaintyInterval interval) && interval != null)
            {
                reason = interval.Reason ?? string.Empty;
                return true;
            }
            return false;
        }
    }

    /// <summary>
    /// Represents an ambiguous or occluded interval in video pose estimation.
    /// </summary>
    [Serializable]
    public class UncertaintyInterval
    {
        public int StartFrame;
        public int EndFrame;
        public string Reason;
        public float Confidence;
        public string RecommendedAction;
        public string PrimaryHypothesis;
        public string[] AlternativeHypotheses;
        public float MaxJointDisagreementMeters;

        public bool ContainsFrame(int frame) => frame >= StartFrame && frame <= EndFrame;

        public UncertaintyInterval() { }

        public UncertaintyInterval(int startFrame, int endFrame, string reason, float confidence, string recommendedAction = "")
        {
            StartFrame = startFrame;
            EndFrame = endFrame;
            Reason = reason ?? string.Empty;
            Confidence = confidence;
            RecommendedAction = recommendedAction ?? string.Empty;
            PrimaryHypothesis = string.Empty;
            AlternativeHypotheses = Array.Empty<string>();
            MaxJointDisagreementMeters = 0f;
        }
    }

    /// <summary>
    /// JSON-friendly backend provenance retained by VideoMotionData.  The
    /// extractor may include additional fields; these are the stable fields
    /// consumed by the editor result card and Timeline badge.
    /// </summary>
    [Serializable]
    public class VideoBackendMetadata
    {
        public int SchemaVersion = 1;
        public string BackendRequested = "legacy";
        public string BackendActual = "legacy";
        public string OfficialRunnerStatus;
        public string Hmr2ImageFeaturesStatus;
        public string VitPose2DStatus;
        public string FusionMode = "off";
        public bool BackendFallback;
        public string FallbackFrom;
        public string FallbackReason;
        public string OverlayBackend = "legacy";
        public string OverlaySource = "legacy";
        public string CoordinateSystem = "smplx";
        public bool WorldMotionAvailable;
        public bool ScaleIsRelative = true;
        public bool FusionAvailable;
        public string FusionError;
        public string PrimaryBackend;
        public string[] AuxiliaryBackends = Array.Empty<string>();
        /// <summary>
        /// Number of individual frames where a quality backend (for example
        /// WHAM) failed the auxiliary silhouette-alignment gate and the
        /// extractor used a safe MediaPipe 3D pose for retargeting/overlay.
        /// This is distinct from <see cref="BackendFallback"/>, which means
        /// that the requested backend could not initialize at all.
        /// </summary>
        public int QualityFrameFallbackCount;
        /// <summary>
        /// Human-readable explanation for per-frame quality fallback.  The
        /// detailed frame spans remain in <see cref="VideoMotionData.UncertaintyIntervals"/>.
        /// </summary>
        public string QualityFallbackReason;
        /// <summary>
        /// Runtime WHAM capability diagnostic. Static files may be present
        /// while an image-feature extractor/archive or exported camera poses
        /// are still missing; keep that distinction visible to the editor.
        /// </summary>
        public string WhamAssetDiagnostic;
        public string[] WhamMissingStages = Array.Empty<string>();
        /// <summary>
        /// Explicit runtime selection and preflight state.  These fields are
        /// kept separate from <see cref="BackendFallback"/> so a backend that
        /// produced frames can still report an optional stage that was not
        /// executable.
        /// </summary>
        public string SelectedBackend;
        public bool BackendReady;
        public string PreflightStatus;
        public string PreflightPhase;
        public string PreflightError;
        public int PreflightFrames;
        public string[] MissingAssets = Array.Empty<string>();
        public string[] IncompatibleAssets = Array.Empty<string>();
        public string[] AssetDiagnostics = Array.Empty<string>();
        public string[] MissingOptionalAssets = Array.Empty<string>();
        public string[] OptionalErrors = Array.Empty<string>();
        public VideoOptionalStageDiagnostic[] OptionalStageDiagnostics = Array.Empty<VideoOptionalStageDiagnostic>();
        /// <summary>Per-video offline WHAM feature/camera preprocessing provenance.</summary>
        public string ImageFeaturePreprocess;
        /// <summary>
        /// Whether the configured ViTPose/image-feature runner actually ran.
        /// A false value is still a valid extraction when the native WHAM bridge
        /// continued with an explicitly labeled fallback; unverified local
        /// descriptors are never presented as learned WHAM features.
        /// </summary>
        public bool ImageFeatureRunnerActive;
        public string ImageFeatureRunnerStatus;
        public string ImageFeatureRunner;
        public string ImageFeatureRunnerModule;
        /// <summary>
        /// Typed evidence emitted by the image-feature runner.  These fields
        /// deliberately remain separate from the WHAM/ViTPose source labels:
        /// a configured checkpoint is not evidence that a feature row was
        /// accepted by the Integrator.
        /// </summary>
        public int ImageFeatureRunnerFeatureDim;
        public int ImageFeatureInputFrameCount;
        public int ImageFeatureAdoptedFrameCount;
        public string ImageFeatureRequestedRunner;
        public string ImageFeatureStageStatus;
        public string ImageFeatureRuntime;
        public string ImageFeatureCheckpointPath;
        public string ImageFeatureCheckpointSha256;
        public string ImageFeatureErrorCode;
        public string ImageFeatureNextAction;
        public string ImageFeatureRunnerError;
        public string ImageFeatureFallback;
        public string ImageFeatureFallbackReason;
        public bool ImageFeatureFallbackConsumed;
        public string CameraMotionRunner;
        public VideoCameraMotionProvenance CameraMotionProvenance;
        public bool DpvoVerified;
        public string WhamPreprocessManifestPath;
        public string[] RuntimeWarnings = Array.Empty<string>();
        public bool WhamFullParity;
        /// <summary>Stable policy text for missing optional WHAM assets/runners.</summary>
        public string WhamInferencePolicy;
        public string WhamMissingAssetBehavior;
        /// <summary>
        /// Seed used by the native WHAM recurrent state.  A
        /// <c>mediapipe_observation</c> value confirms that the subject's
        /// first camera-space 3D pose initialized the rollout instead of a
        /// neutral checkpoint pose.
        /// </summary>
        public string SmplInitializationSource;
        public int SmplInitializationFrame = -1;
        public bool MediapipeSeedTrackerReset;
        public VideoFusionMetadata Fusion;
        public VideoBackendSourceMetadata Wham;
        public VideoBackendSourceMetadata MediaPipe;
        public VideoBackendSourceMetadata RTMPose;

        public float MeanReprojectionError => Fusion != null ? Fusion.MeanReprojectionError : 0f;
        public float MeanWhamCorrection => Fusion != null ? Fusion.MeanWhamCorrection : 0f;

        public static VideoBackendMetadata CreateLegacy(
            string requested,
            string actual,
            string fusionMode,
            bool fallback,
            string fallbackFrom,
            string fallbackReason,
            string overlayBackend,
            string overlaySource)
        {
            return new VideoBackendMetadata
            {
                SchemaVersion = 1,
                BackendRequested = string.IsNullOrEmpty(requested) ? "legacy" : requested,
                BackendActual = string.IsNullOrEmpty(actual) ? "legacy" : actual,
                FusionMode = string.IsNullOrEmpty(fusionMode) ? "off" : fusionMode,
                BackendFallback = fallback,
                FallbackFrom = fallbackFrom,
                FallbackReason = fallbackReason,
                OverlayBackend = string.IsNullOrEmpty(overlayBackend) ? "legacy" : overlayBackend,
                OverlaySource = string.IsNullOrEmpty(overlaySource) ? "legacy" : overlaySource,
                PrimaryBackend = string.IsNullOrEmpty(actual) ? "legacy" : actual,
                AuxiliaryBackends = Array.Empty<string>()
            };
        }
    }

    [Serializable]
    public class VideoFusionMetadata
    {
        public string CameraModel = "orthographic";
        public float WindowSeconds;
        public int WeightsVersion;
        public float MeanReprojectionError;
        public float MeanWhamCorrection;
    }

    [Serializable]
    public class VideoBackendSourceMetadata
    {
        public string Implementation;
        public string ModelPath;
        public string Checkpoint;
        public string ModelHash;
        public string CoordinateSystem;
        public int ObservationCount;
        public string Error;
    }

    /// <summary>
    /// Provenance for the camera-motion stage.  A DPVO checkpoint being present
    /// is not evidence that DPVO ran, so the source and verification bit are
    /// retained in the exported motion data.
    /// </summary>
    [Serializable]
    public class VideoCameraMotionProvenance
    {
        public string Source;
        public bool Verified;
        public string Runner;
        public string Asset;
    }

    /// <summary>
    /// Compact optional-stage diagnostic that Unity can deserialize without
    /// relying on dictionary support in JsonUtility.
    /// </summary>
    [Serializable]
    public class VideoOptionalStageDiagnostic
    {
        public string Stage;
        public string Status;
        public string Reason;
    }
}
