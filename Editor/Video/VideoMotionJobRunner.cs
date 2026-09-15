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
                if (IsError) return $"[Video Error] {ErrorMessage}";
                if (IsCompleted) return "Extraction complete!";
                if (TotalFrames > 0) return $"{Stage} ({Percentage:F0}%) - Frame {CurrentFrame}/{TotalFrames}";
                return $"{Stage} ({Percentage:F0}%)";
            }
            set => _message = value;
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
    /// Configuration options for invoking the video pose extraction pipeline.
    /// </summary>
    [Serializable]
    public class VideoJobOptions
    {
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

            return sb.ToString();
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
        public string ErrorMessage;

        /// <summary>
        /// True if Python is available and required core dependencies (MediaPipe, OpenCV) are installed.
        /// </summary>
        public bool IsFullyConfigured => IsAvailable && HasMediaPipe && HasOpenCV;

        public string GetSummary()
        {
            if (!IsAvailable)
            {
                return $"Python Unavailable: {ErrorMessage ?? "Not detected"}";
            }

            var sb = new StringBuilder();
            sb.Append($"Python {Version ?? "Unknown"} ({ExecutablePath})");
            if (IsVirtualEnv) sb.Append(" [VirtualEnv]");

            var missing = new List<string>();
            if (!HasMediaPipe) missing.Add("mediapipe");
            if (!HasOpenCV) missing.Add("opencv-python");
            if (!HasNumPy) missing.Add("numpy");
            if (!HasSciPy) missing.Add("scipy");

            if (missing.Count == 0)
            {
                sb.Append(" - All dependencies ready");
            }
            else
            {
                sb.Append($" - Missing packages: {string.Join(", ", missing)}");
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
                    "video_pose_extractor.py script could not be located in package or Assets. " +
                    "Ensure Editor/Video/video_pose_extractor.py exists.");
            }

            // 2. Resolve Python executable
            string pythonExe = options.PythonExecutable;
            if (string.IsNullOrEmpty(pythonExe))
            {
                try
                {
                    pythonExe = TexMotionSettings.instance?.CustomPythonExecutablePath;
                }
                catch {}
            }

            if (string.IsNullOrEmpty(pythonExe) || !File.Exists(pythonExe))
            {
                var runtime = DetectPythonRuntime();
                if (!runtime.IsAvailable)
                {
                    throw new InvalidOperationException(
                        "No working Python runtime found on system. " +
                        "Please install Python 3.10+ (with mediapipe and opencv-python) " +
                        "or specify the Python path in TexMotion Settings.\nDetails: " + runtime.ErrorMessage);
                }
                pythonExe = runtime.ExecutablePath;
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
                        throw new InvalidOperationException($"Failed to start Python process: {pythonExe}");
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
                        throw new OperationCanceledException("Video motion extraction was canceled.", cancellationToken);
                    }

                    string stderrText;
                    lock (stderrBuilder)
                    {
                        stderrText = stderrBuilder.ToString();
                    }

                    if (exitCode != 0)
                    {
                        string msg = string.IsNullOrEmpty(stageErrorMessage)
                            ? $"Video pose extraction failed with exit code {exitCode}.\nDiagnostics:\n{stderrText}"
                            : $"Video pose extraction failed during stage '{lastStage}': {stageErrorMessage}\nDiagnostics:\n{stderrText}";
                        throw new VideoExtractionException(msg, exitCode, stderrText, lastStage);
                    }

                    if (!string.IsNullOrEmpty(stageErrorMessage))
                    {
                        throw new VideoExtractionException(
                            $"Video pose extraction reported error: {stageErrorMessage}\nDiagnostics:\n{stderrText}",
                            exitCode, stderrText, lastStage);
                    }
                }
                catch (Exception) when (cancellationToken.IsCancellationRequested)
                {
                    KillProcessTree(process);
                    throw new OperationCanceledException("Video motion extraction was canceled.", cancellationToken);
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
                throw new FileNotFoundException(
                    $"Expected output motion JSON file not found at: {outputPath}.\nDiagnostics:\n{stderrText}");
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
                throw new FileNotFoundException($"Motion JSON file not found: {filePath}");
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
                throw new ArgumentNullException(nameof(jsonText), "JSON content cannot be null or empty.");
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
                throw new FormatException("Failed to deserialize motion JSON into MotionDataJsonDto. Ensure valid JSON payload.");
            }

            int frames = dto.frames;
            int jointCount = dto.jointCount > 0 ? dto.jointCount : SmplxJointDefinitions.JointCount;
            float frameRate = dto.frameRate > 0f ? dto.frameRate : 30.0f;

            // 1. Root positions
            var rootPositions = new Vector3[frames];
            if (dto.rootPositions != null && dto.rootPositions.Length >= frames)
            {
                for (int t = 0; t < frames; t++)
                {
                    rootPositions[t] = new Vector3(dto.rootPositions[t].x, dto.rootPositions[t].y, dto.rootPositions[t].z);
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
            if (dto.flatLocalRotations != null && dto.flatLocalRotations.Length >= frames * jointCount)
            {
                for (int t = 0; t < frames; t++)
                {
                    for (int j = 0; j < jointCount; j++)
                    {
                        var q = dto.flatLocalRotations[t * jointCount + j];
                        localRotations[t, j] = new Quaternion(q.x, q.y, q.z, q.w);
                    }
                }
            }
            else
            {
                // Fallback: parse from nested localRotations if flatLocalRotations wasn't present
                var fallbackRotations = ParseLocalRotationsFallback(jsonText, frames, jointCount);
                if (fallbackRotations != null)
                {
                    localRotations = fallbackRotations;
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
                dto.overlayVideoPath
            );

            if (!motionData.Validate(out string validationError))
            {
                Debug.LogWarning($"[TexMotion VideoJobRunner] Deserialized VideoMotionData warning: {validationError}");
            }

            return motionData;
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
                overlayVideoPath = ExtractString(jsonText, "overlayVideoPath", null)
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

            return dto;
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
            public bool inPlace;
            public bool smoothed;
            public bool footLocking;
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
                bestFallback.ErrorMessage =
                    "Python runtime was found, but required video dependencies (mediapipe, opencv-python) are missing. " +
                    "Run requirements installation to enable full pose extraction.";
                return bestFallback;
            }

            return new PythonRuntimeInfo
            {
                IsAvailable = false,
                ErrorMessage = "No Python runtime detected on system. Please install Python 3.10+ (with mediapipe and opencv-python)."
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
                string settingsPath = TexMotionSettings.instance?.CustomPythonExecutablePath;
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

            // 4. Project-local virtual environments (.venv, venv, env, TexMotion-venv)
            var searchRoots = new List<string> { Directory.GetCurrentDirectory() };
            string dataPath = SafeGetDataPath();
            if (!string.IsNullOrEmpty(dataPath))
            {
                searchRoots.Add(dataPath);
                string parent = Path.GetDirectoryName(dataPath);
                if (!string.IsNullOrEmpty(parent)) searchRoots.Add(parent);
            }

            string[] venvFolderNames = new[] { ".venv", "venv", "env", "TexMotion-venv" };
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
                info.ErrorMessage = "Python executable path is null or empty.";
                return info;
            }

            // Check if absolute path exists on disk
            if (pythonExe.Contains(Path.DirectorySeparatorChar.ToString()) || pythonExe.Contains(Path.AltDirectorySeparatorChar.ToString()))
            {
                if (!File.Exists(pythonExe))
                {
                    info.IsAvailable = false;
                    info.ErrorMessage = $"File not found: {pythonExe}";
                    return info;
                }
            }

            string probeScript =
                "import sys, json, importlib.util; " +
                "print(json.dumps({" +
                "'version': sys.version.split()[0], " +
                "'executable': sys.executable, " +
                "'prefix': sys.prefix, " +
                "'is_venv': sys.prefix != getattr(sys, 'base_prefix', sys.prefix), " +
                "'mp': bool(importlib.util.find_spec('mediapipe')), " +
                "'cv': bool(importlib.util.find_spec('cv2')), " +
                "'np': bool(importlib.util.find_spec('numpy')), " +
                "'sp': bool(importlib.util.find_spec('scipy'))" +
                "}))";

            var psi = new ProcessStartInfo
            {
                FileName = pythonExe,
                Arguments = $"-c \"{probeScript}\"",
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
                    info.ErrorMessage = "Failed to launch Python process.";
                    return info;
                }

                string stdout = process.StandardOutput.ReadToEnd();
                string stderr = process.StandardError.ReadToEnd();

                if (!process.WaitForExit(timeoutMs))
                {
                    KillProcessTree(process);
                    info.IsAvailable = false;
                    info.ErrorMessage = "Python probe execution timed out.";
                    return info;
                }

                if (process.ExitCode != 0)
                {
                    info.IsAvailable = false;
                    info.ErrorMessage = $"Python process exited with code {process.ExitCode}: {stderr}";
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
                    return info;
                }

                info.IsAvailable = true;
                info.Version = "Unknown";
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
                    throw new InvalidOperationException("Cannot install requirements: No Python executable detected.");
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
                    throw new InvalidOperationException($"Failed to launch pip process with {pythonExe}");
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
                Debug.LogWarning($"[TexMotion VideoJobRunner] Notice while terminating process: {ex.Message}");
            }
        }

        #endregion
    }
}
