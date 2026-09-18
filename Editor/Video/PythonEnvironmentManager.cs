using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using UnityEngine;

namespace TexMotion.Editor.Video
{
    /// <summary>
    /// Optional dependency profile used when provisioning the local pose runtime.
    /// Lightweight is the supported offline extraction path; PyTorch is intentionally
    /// opt-in because its wheel is large and hardware-specific.
    /// </summary>
    public enum PythonDependencyProfile
    {
        Lightweight,
        PyTorch
    }

    /// <summary>
    /// Options for creating or updating a project-local Python environment.
    /// </summary>
    [Serializable]
    public sealed class PythonEnvironmentOptions
    {
        /// <summary>Project root used to resolve the default .texmotion-venv and requirements paths.</summary>
        public string ProjectRoot;

        /// <summary>Environment directory. Defaults to &lt;ProjectRoot&gt;/.texmotion-venv.</summary>
        public string EnvironmentDirectory;

        /// <summary>Lightweight requirements file. Defaults to Editor/Video/requirements.txt.</summary>
        public string RequirementsPath;

        /// <summary>Optional heavy profile requirements file.</summary>
        public string PyTorchRequirementsPath;

        /// <summary>Dependency profile to install.</summary>
        public PythonDependencyProfile Profile = PythonDependencyProfile.Lightweight;

        /// <summary>Python version requested from uv when creating a new environment.</summary>
        public string PythonVersion = "3.11";

        /// <summary>
        /// Use only uv's local cache while installing. Set this after a machine has been
        /// provisioned when extraction must remain fully offline.
        /// </summary>
        public bool Offline;

        /// <summary>If true, continue when an optional requirements file is absent.</summary>
        public bool AllowMissingOptionalProfile = true;

        public static PythonEnvironmentOptions CreateDefault(PythonDependencyProfile profile = PythonDependencyProfile.Lightweight)
        {
            string root = PythonEnvironmentManager.FindProjectRoot();
            return new PythonEnvironmentOptions
            {
                ProjectRoot = root,
                EnvironmentDirectory = PythonEnvironmentManager.GetDefaultEnvironmentDirectory(root),
                RequirementsPath = PythonEnvironmentManager.FindRequirementsPath(root),
                PyTorchRequirementsPath = PythonEnvironmentManager.FindPyTorchRequirementsPath(root),
                Profile = profile
            };
        }

        internal void Normalize()
        {
            if (string.IsNullOrEmpty(ProjectRoot)) ProjectRoot = PythonEnvironmentManager.FindProjectRoot();
            if (string.IsNullOrEmpty(EnvironmentDirectory))
            {
                EnvironmentDirectory = PythonEnvironmentManager.GetDefaultEnvironmentDirectory(ProjectRoot);
            }
            if (string.IsNullOrEmpty(RequirementsPath)) RequirementsPath = PythonEnvironmentManager.FindRequirementsPath(ProjectRoot);
            if (string.IsNullOrEmpty(PyTorchRequirementsPath))
            {
                PyTorchRequirementsPath = PythonEnvironmentManager.FindPyTorchRequirementsPath(ProjectRoot);
            }
            if (string.IsNullOrEmpty(PythonVersion)) PythonVersion = "3.11";

            ProjectRoot = PythonEnvironmentManager.NormalizePath(ProjectRoot);
            EnvironmentDirectory = PythonEnvironmentManager.NormalizePath(EnvironmentDirectory);
            if (!string.IsNullOrEmpty(RequirementsPath)) RequirementsPath = PythonEnvironmentManager.NormalizePath(RequirementsPath);
            if (!string.IsNullOrEmpty(PyTorchRequirementsPath)) PyTorchRequirementsPath = PythonEnvironmentManager.NormalizePath(PyTorchRequirementsPath);
        }
    }

    /// <summary>Information returned by a uv executable probe.</summary>
    [Serializable]
    public sealed class UvRuntimeInfo
    {
        public bool IsAvailable;
        public string ExecutablePath;
        public string Version;
        public string ErrorMessage;

        public string GetSummary()
        {
            if (!IsAvailable)
            {
                return TexMotionLocalization.TrFormat(
                    TexMotionLocalization.UvUnavailable,
                    ErrorMessage ?? TexMotionLocalization.Tr(TexMotionLocalization.InstallUvOrUseExistingRuntime));
            }
            return TexMotionLocalization.TrFormat(
                TexMotionLocalization.UvSummary,
                Version ?? "unknown",
                ExecutablePath);
        }
    }

    /// <summary>Result returned by the optional official uv installer.</summary>
    [Serializable]
    public sealed class UvInstallResult
    {
        public bool Success;
        public string ExecutablePath;
        public int ExitCode;
        public string ErrorMessage;
        public string Diagnostics;
        public UvRuntimeInfo Runtime;

        public bool IsUsable => Success && Runtime != null && Runtime.IsAvailable;
    }

    /// <summary>Structured status emitted while provisioning the environment.</summary>
    [Serializable]
    public struct PythonEnvironmentProgress
    {
        public float Progress;
        public string Stage;
        public string Message;

        public PythonEnvironmentProgress(float progress, string stage, string message)
        {
            Progress = Mathf.Clamp01(progress);
            Stage = stage ?? string.Empty;
            Message = message ?? string.Empty;
        }

        public override string ToString() => Message;
    }

    /// <summary>Result of an environment setup operation.</summary>
    [Serializable]
    public sealed class PythonEnvironmentSetupResult
    {
        public bool Success;
        public bool UsedUv;
        public bool WasAlreadyConfigured;
        public string EnvironmentDirectory;
        public string PythonExecutable;
        public string RequirementsPath;
        public PythonDependencyProfile Profile;
        public int ExitCode;
        public string ErrorMessage;
        public string Diagnostics;
        public PythonRuntimeInfo Runtime;

        public bool IsUsable => Success && !string.IsNullOrEmpty(PythonExecutable) && File.Exists(PythonExecutable);
    }

    /// <summary>
    /// Provisions the local Windows/desktop Python runtime with uv. This class never
    /// downloads or contacts a service during extraction: network access is confined to
    /// the explicit setup call, and <see cref="PythonEnvironmentOptions.Offline"/> passes
    /// uv's --offline switch for repeatable air-gapped setup.
    /// </summary>
    public static class PythonEnvironmentManager
    {
        // Keep the environment isolated from a user's generic project venv and
        // align this default with TexMotionSettings.GetDefaultVideoVenvPath().
        private const string DefaultVenvName = ".texmotion-venv";

        #region Discovery

        /// <summary>Finds the project/package root used by TexMotion.</summary>
        public static string FindProjectRoot()
        {
            var roots = new List<string>();
            try { roots.Add(Directory.GetCurrentDirectory()); } catch { }
            try
            {
                string dataPath = Application.dataPath;
                if (!string.IsNullOrEmpty(dataPath)) roots.Add(Directory.GetParent(dataPath)?.FullName);
            }
            catch { }

            foreach (string candidate in roots)
            {
                string found = FindRootFrom(candidate);
                if (!string.IsNullOrEmpty(found)) return found;
            }

            return NormalizePath(Directory.GetCurrentDirectory());
        }

        private static string FindRootFrom(string start)
        {
            if (string.IsNullOrEmpty(start)) return null;
            string current;
            try { current = Path.GetFullPath(start); } catch { return null; }

            for (int i = 0; i < 12 && !string.IsNullOrEmpty(current); i++)
            {
                bool isUnityProject = Directory.Exists(Path.Combine(current, "Assets")) || Directory.Exists(Path.Combine(current, "ProjectSettings"));
                bool isPackageRoot = File.Exists(Path.Combine(current, "package.json")) && Directory.Exists(Path.Combine(current, "Editor"));
                if (isUnityProject || isPackageRoot) return NormalizePath(current);

                string parent;
                try { parent = Directory.GetParent(current)?.FullName; } catch { parent = null; }
                if (string.Equals(parent, current, StringComparison.OrdinalIgnoreCase)) break;
                current = parent;
            }
            return null;
        }

        public static string GetDefaultEnvironmentDirectory(string projectRoot = null)
        {
            string root = string.IsNullOrEmpty(projectRoot) ? FindProjectRoot() : projectRoot;
            return NormalizePath(Path.Combine(root, DefaultVenvName));
        }

        public static string GetPythonExecutablePath(string environmentDirectory = null)
        {
            string env = string.IsNullOrEmpty(environmentDirectory)
                ? GetDefaultEnvironmentDirectory()
                : environmentDirectory;
            string windowsPath = Path.Combine(env, "Scripts", "python.exe");
            if (File.Exists(windowsPath) || IsWindowsPlatform) return NormalizePath(windowsPath);
            return NormalizePath(Path.Combine(env, "bin", "python"));
        }

        /// <summary>Returns the known local venv interpreter candidates, including legacy names.</summary>
        public static List<string> FindLocalPythonPaths(string projectRoot = null)
        {
            string root = string.IsNullOrEmpty(projectRoot) ? FindProjectRoot() : projectRoot;
            var paths = new List<string>();
            string[] names = { ".texmotion-venv", ".venv", "venv", "env", "TexMotion-venv" };
            foreach (string name in names)
            {
                string dir = Path.Combine(root, name);
                string win = Path.Combine(dir, "Scripts", "python.exe");
                string unix = Path.Combine(dir, "bin", "python");
                if (File.Exists(win) && !paths.Contains(win)) paths.Add(NormalizePath(win));
                if (File.Exists(unix) && !paths.Contains(unix)) paths.Add(NormalizePath(unix));
            }
            return paths;
        }

        /// <summary>
        /// Returns conventional uv executable locations used by the official
        /// installer and common Windows package managers. The standalone
        /// installer currently defaults to %USERPROFILE%\\.local\\bin\\uv.exe;
        /// the pre-0.5 Cargo location is retained for upgrades.
        /// </summary>
        public static List<string> GetUvExecutableCandidates()
        {
            var candidates = new List<string>();
            var seen = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
            Action<string> add = value =>
            {
                if (string.IsNullOrWhiteSpace(value)) return;
                string candidate = value.Trim().Trim('"');
                if (string.IsNullOrEmpty(candidate)) return;
                try { candidate = NormalizePath(candidate); } catch { }
                if (seen.Add(candidate)) candidates.Add(candidate);
            };

            // The installer honours UV_INSTALL_DIR and UV_UNMANAGED_INSTALL.
            string installDir = Environment.GetEnvironmentVariable("UV_INSTALL_DIR");
            string unmanaged = Environment.GetEnvironmentVariable("UV_UNMANAGED_INSTALL");
            foreach (string configured in new[] { installDir, unmanaged })
            {
                if (string.IsNullOrWhiteSpace(configured)) continue;
                string value = configured.Trim().Trim('"');
                if (value.EndsWith(".exe", StringComparison.OrdinalIgnoreCase)) add(value);
                else add(Path.Combine(value, "uv.exe"));
            }

            string xdgBinHome = Environment.GetEnvironmentVariable("XDG_BIN_HOME");
            if (!string.IsNullOrEmpty(xdgBinHome)) add(Path.Combine(xdgBinHome, "uv.exe"));
            string xdgDataHome = Environment.GetEnvironmentVariable("XDG_DATA_HOME");
            if (!string.IsNullOrEmpty(xdgDataHome)) add(Path.Combine(xdgDataHome, "..", "bin", "uv.exe"));

            string userProfile = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);
            string localAppData = Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData);
            string roamingAppData = Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData);
            string programData = Environment.GetFolderPath(Environment.SpecialFolder.CommonApplicationData);
            string programFiles = Environment.GetFolderPath(Environment.SpecialFolder.ProgramFiles);
            if (!string.IsNullOrEmpty(userProfile))
            {
                // Official standalone installer and legacy Cargo install.
                add(Path.Combine(userProfile, ".local", "bin", "uv.exe"));
                add(Path.Combine(userProfile, ".cargo", "bin", "uv.exe"));
                add(Path.Combine(userProfile, "scoop", "shims", "uv.exe"));
            }
            if (!string.IsNullOrEmpty(localAppData))
            {
                add(Path.Combine(localAppData, "uv", "uv.exe"));
                add(Path.Combine(localAppData, "uv", "bin", "uv.exe"));
                add(Path.Combine(localAppData, "Programs", "uv", "uv.exe"));
                // WinGet exposes installed command-line tools through this
                // per-user links directory.
                add(Path.Combine(localAppData, "Microsoft", "WinGet", "Links", "uv.exe"));
            }
            if (!string.IsNullOrEmpty(programData))
            {
                add(Path.Combine(programData, "chocolatey", "bin", "uv.exe"));
            }
            if (!string.IsNullOrEmpty(programFiles))
            {
                add(Path.Combine(programFiles, "uv", "uv.exe"));
                add(Path.Combine(programFiles, "uv", "bin", "uv.exe"));
            }

            // ``pip install --user uv`` places scripts under a versioned
            // roaming Python directory.
            foreach (string root in new[] { roamingAppData, localAppData })
            {
                if (string.IsNullOrEmpty(root) || !Directory.Exists(root)) continue;
                string pythonRoot = Path.Combine(root, "Python");
                try
                {
                    foreach (string versionDir in Directory.GetDirectories(pythonRoot, "Python*"))
                        add(Path.Combine(versionDir, "Scripts", "uv.exe"));
                }
                catch { }
            }

            // Inspect PATH directly so package managers and existing shells are
            // detected without invoking a shell process.
            string pathEnv = Environment.GetEnvironmentVariable("PATH");
            if (!string.IsNullOrEmpty(pathEnv))
            {
                foreach (string segment in pathEnv.Split(Path.PathSeparator))
                {
                    if (!string.IsNullOrWhiteSpace(segment)) add(Path.Combine(segment.Trim().Trim('\"'), "uv.exe"));
                }
            }
            return candidates;
        }

        /// <summary>Returns the current official standalone-installer target on Windows.</summary>
        public static string GetDefaultUvExecutablePath()
        {
            string userProfile = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);
            return string.IsNullOrEmpty(userProfile)
                ? null
                : NormalizePath(Path.Combine(userProfile, ".local", "bin", "uv.exe"));
        }

        /// <summary>Resolves uv without invoking a shell, checking explicit and common locations.</summary>
        public static string FindUvExecutable(string explicitPath = null)
        {
            var candidates = new List<string>();
            string explicitCommand = null;
            if (!string.IsNullOrEmpty(explicitPath))
            {
                bool explicitIsPath = explicitPath.IndexOf(Path.DirectorySeparatorChar) >= 0 ||
                                      explicitPath.IndexOf(Path.AltDirectorySeparatorChar) >= 0;
                if (explicitIsPath) candidates.Add(explicitPath);
                else
                {
                    explicitCommand = explicitPath.Trim();
                    // The default setting is the bare "uv" command. Defer it
                    // until absolute locations are checked, while preserving an
                    // explicitly named command's priority.
                    if (!string.Equals(explicitCommand, "uv", StringComparison.OrdinalIgnoreCase))
                    {
                        candidates.Add(explicitCommand);
                        explicitCommand = null;
                    }
                }
            }

            candidates.AddRange(GetUvExecutableCandidates());
            candidates.Add(string.IsNullOrEmpty(explicitCommand) ? "uv" : explicitCommand);

            foreach (string candidate in candidates)
            {
                if (string.IsNullOrEmpty(candidate)) continue;
                bool isPath = candidate.IndexOf(Path.DirectorySeparatorChar) >= 0 || candidate.IndexOf(Path.AltDirectorySeparatorChar) >= 0;
                if (!isPath || File.Exists(candidate)) return candidate;
            }
            return null;
        }

        public static UvRuntimeInfo DetectUv(string explicitPath = null, int timeoutMs = 4000)
        {
            string executable = FindUvExecutable(explicitPath);
            var info = new UvRuntimeInfo { ExecutablePath = executable };
            if (string.IsNullOrEmpty(executable))
            {
                info.ErrorMessage = TexMotionLocalization.Tr(TexMotionLocalization.UvPathLookupFailed);
                return info;
            }

            ProcessResult result;
            try { result = RunProcessAsync(executable, "--version", null, timeoutMs, CancellationToken.None).GetAwaiter().GetResult(); }
            catch (Exception ex)
            {
                info.ErrorMessage = ex.Message;
                return info;
            }

            if (result.ExitCode != 0)
            {
                info.ErrorMessage = string.IsNullOrEmpty(result.Stderr)
                    ? TexMotionLocalization.Tr(TexMotionLocalization.UvVersionFailed)
                    : result.Stderr.Trim();
                return info;
            }

            info.IsAvailable = true;
            info.Version = FirstNonEmptyLine(result.Stdout) ?? FirstNonEmptyLine(result.Stderr) ?? "unknown";
            return info;
        }

        /// <summary>
        /// Installs uv with Astral's official Windows PowerShell installer. The
        /// network is used only after the user presses the Settings button; pose
        /// extraction itself never calls this method.
        /// </summary>
        public static async Task<UvInstallResult> InstallUvAsync(
            IProgress<PythonEnvironmentProgress> progress = null,
            Action<string> onLog = null,
            CancellationToken cancellationToken = default,
            string powershellExecutable = null)
        {
            var result = new UvInstallResult();
            if (!IsWindowsPlatform)
            {
                result.ErrorMessage = TexMotionLocalization.Tr(TexMotionLocalization.UvInstallerWindowsOnly);
                progress?.Report(new PythonEnvironmentProgress(0f, "error", result.ErrorMessage));
                return result;
            }

            string shell = string.IsNullOrWhiteSpace(powershellExecutable)
                ? FindPowerShellExecutable()
                : powershellExecutable;
            if (string.IsNullOrEmpty(shell))
            {
                result.ErrorMessage = TexMotionLocalization.Tr(TexMotionLocalization.PowerShellNotFound);
                progress?.Report(new PythonEnvironmentProgress(0f, "error", result.ErrorMessage));
                return result;
            }

            const string installerUrl = "https://astral.sh/uv/install.ps1";
            string command = "$ErrorActionPreference='Stop'; Invoke-RestMethod '" + installerUrl + "' | Invoke-Expression";
            string arguments = "-NoProfile -ExecutionPolicy Bypass -Command \"" + command + "\"";
            progress?.Report(new PythonEnvironmentProgress(
                0.05f,
                "uv-install",
                TexMotionLocalization.Tr(TexMotionLocalization.RunningUvInstaller)));
            onLog?.Invoke(TexMotionLocalization.TrFormat(
                TexMotionLocalization.InstallingUvWithOfficial,
                installerUrl));

            try
            {
                ProcessResult process = await RunProcessAsync(shell, arguments, onLog, 0, cancellationToken).ConfigureAwait(false);
                result.ExitCode = process.ExitCode;
                result.Diagnostics = CombineDiagnostics(process);
                if (process.ExitCode != 0)
                {
                    result.ErrorMessage = TexMotionLocalization.TrFormat(
                        TexMotionLocalization.UvInstallFailed,
                        process.ExitCode);
                    progress?.Report(new PythonEnvironmentProgress(0f, "error", result.ErrorMessage));
                    return result;
                }

                progress?.Report(new PythonEnvironmentProgress(
                    0.85f,
                    "uv-probe",
                    TexMotionLocalization.Tr(TexMotionLocalization.VerifyingUv)));
                UvRuntimeInfo runtime = DetectUv(null);
                result.Runtime = runtime;
                result.ExecutablePath = runtime.ExecutablePath;
                result.Success = runtime.IsAvailable;
                if (!result.Success)
                {
                    result.ErrorMessage = TexMotionLocalization.Tr(TexMotionLocalization.UvExecutableMissingAfterInstall);
                    progress?.Report(new PythonEnvironmentProgress(0f, "error", result.ErrorMessage));
                    return result;
                }

                progress?.Report(new PythonEnvironmentProgress(
                    1f,
                    "uv-installed",
                    TexMotionLocalization.Tr(TexMotionLocalization.UvInstalledReady)));
                onLog?.Invoke("[TexMotion] " + runtime.GetSummary());
                return result;
            }
            catch (OperationCanceledException)
            {
                result.ErrorMessage = TexMotionLocalization.Tr(TexMotionLocalization.UvInstallationCancelled);
                progress?.Report(new PythonEnvironmentProgress(0f, "cancelled", result.ErrorMessage));
                throw;
            }
            catch (Exception ex)
            {
                result.ErrorMessage = ex.Message;
                result.Diagnostics = ex.ToString();
                progress?.Report(new PythonEnvironmentProgress(0f, "error", result.ErrorMessage));
                return result;
            }
        }

        private static string FindPowerShellExecutable()
        {
            string[] candidates = { "powershell.exe", "pwsh.exe", "powershell", "pwsh" };
            foreach (string candidate in candidates)
            {
                try
                {
                    if (candidate.IndexOf(Path.DirectorySeparatorChar) >= 0 || candidate.IndexOf(Path.AltDirectorySeparatorChar) >= 0)
                    {
                        if (File.Exists(candidate)) return candidate;
                    }
                    else
                    {
                        // Keep command names for Process.Start; PATH resolution is
                        // handled by the OS even when no absolute path is known.
                        return candidate;
                    }
                }
                catch { }
            }
            return null;
        }

        #endregion

        #region Setup

        /// <summary>
        /// Creates/updates the local venv and installs the selected dependency profile.
        /// A missing uv executable is returned as a failed result with an actionable
        /// message, allowing the Settings UI to offer installation instructions without
        /// crashing the editor.
        /// </summary>
        public static async Task<PythonEnvironmentSetupResult> EnsureEnvironmentAsync(
            PythonEnvironmentOptions options = null,
            IProgress<PythonEnvironmentProgress> progress = null,
            Action<string> onLog = null,
            CancellationToken cancellationToken = default,
            string uvExecutable = null)
        {
            cancellationToken.ThrowIfCancellationRequested();
            options = options ?? PythonEnvironmentOptions.CreateDefault();
            options.Normalize();
            // A machine-wide UV_CACHE_DIR can point at a stale or read-only
            // drive. Resolve it once for this setup so every uv command uses
            // the same writable cache and the decision is visible in the log.
            string uvCacheDirectory = ResolveUvCacheDirectory(options.ProjectRoot, options.Profile, onLog);
            if (string.IsNullOrEmpty(uvCacheDirectory))
            {
                onLog?.Invoke(TexMotionLocalization.Tr(TexMotionLocalization.UvCacheDefault));
            }

            var result = new PythonEnvironmentSetupResult
            {
                EnvironmentDirectory = options.EnvironmentDirectory,
                RequirementsPath = options.RequirementsPath,
                Profile = options.Profile,
                ExitCode = 0
            };

            string pythonPath = GetPythonExecutablePath(options.EnvironmentDirectory);
            result.PythonExecutable = pythonPath;
            bool hadPythonBeforeSetup = File.Exists(pythonPath);
            PythonRuntimeInfo existingRuntime = null;
            if (hadPythonBeforeSetup)
            {
                progress?.Report(new PythonEnvironmentProgress(
                    0.15f,
                    "probe",
                    TexMotionLocalization.Tr(TexMotionLocalization.ExistingPythonEnvironment)));
                onLog?.Invoke(TexMotionLocalization.TrFormat(TexMotionLocalization.ExistingEnvironmentLog, pythonPath));
                existingRuntime = VideoMotionJobRunner.ProbePythonExecutable(pythonPath);
            }

            UvRuntimeInfo uv = DetectUv(uvExecutable);
            if (!uv.IsAvailable)
            {
                result.ErrorMessage = uv.GetSummary() + " " +
                    TexMotionLocalization.Tr(TexMotionLocalization.InstallUvFromDocsOrSelectRuntime);
                result.Diagnostics = result.ErrorMessage;
                onLog?.Invoke("[TexMotion] " + result.ErrorMessage);
                progress?.Report(new PythonEnvironmentProgress(0f, "error", result.ErrorMessage));
                return result;
            }

            result.UsedUv = true;
            progress?.Report(new PythonEnvironmentProgress(
                0.05f,
                "uv",
                TexMotionLocalization.TrFormat(TexMotionLocalization.UsingUv, uv.GetSummary())));
            onLog?.Invoke("[TexMotion] " + uv.GetSummary());

            cancellationToken.ThrowIfCancellationRequested();
            if (!File.Exists(pythonPath))
            {
                progress?.Report(new PythonEnvironmentProgress(
                    0.15f,
                    "venv",
                    TexMotionLocalization.Tr(TexMotionLocalization.CreatingLocalVenv)));
                string createArguments = "venv --python " + QuoteArgument(options.PythonVersion) + " " + QuoteArgument(options.EnvironmentDirectory);
                if (options.Offline) createArguments += " --offline";
                var create = await RunProcessAsync(
                    uv.ExecutablePath,
                    createArguments,
                    onLog,
                    0,
                    cancellationToken,
                    uvCacheDirectory: uvCacheDirectory).ConfigureAwait(false);
                result.ExitCode = create.ExitCode;
                if (create.ExitCode != 0 || !File.Exists(pythonPath))
                {
                    return Fail(
                        result,
                        TexMotionLocalization.Tr(TexMotionLocalization.EnvironmentCreateFailed),
                        create,
                        progress,
                        onLog);
                }
            }

            cancellationToken.ThrowIfCancellationRequested();
            var requirementFiles = new List<string>();
            if (!string.IsNullOrEmpty(options.RequirementsPath) && File.Exists(options.RequirementsPath))
            {
                requirementFiles.Add(options.RequirementsPath);
            }
            else
            {
                return Fail(
                    result,
                    TexMotionLocalization.Tr(TexMotionLocalization.RequirementsMissing),
                    null,
                    progress,
                    onLog);
            }

            if (options.Profile == PythonDependencyProfile.PyTorch)
            {
                if (!string.IsNullOrEmpty(options.PyTorchRequirementsPath) && File.Exists(options.PyTorchRequirementsPath))
                {
                    requirementFiles.Add(options.PyTorchRequirementsPath);
                }
                else if (!options.AllowMissingOptionalProfile)
                {
                    return Fail(
                        result,
                        TexMotionLocalization.Tr(TexMotionLocalization.PyTorchRequirementsMissing),
                        null,
                        progress,
                        onLog);
                }
                else
                {
                    onLog?.Invoke(TexMotionLocalization.Tr(TexMotionLocalization.OptionalPyTorchRequirementsMissing));
                }
            }

            for (int i = 0; i < requirementFiles.Count; i++)
            {
                float start = 0.25f + (0.65f * i / requirementFiles.Count);
                float end = 0.25f + (0.65f * (i + 1) / requirementFiles.Count);
                string file = requirementFiles[i];
                progress?.Report(new PythonEnvironmentProgress(
                    start,
                    "install",
                    TexMotionLocalization.TrFormat(
                        TexMotionLocalization.InstallingRequirements,
                        Path.GetFileName(file))));
                onLog?.Invoke(TexMotionLocalization.TrFormat(
                    TexMotionLocalization.InstallingRequirementsFrom,
                    file));
                // Keep uv's output line-oriented so it can be streamed into the
                // Settings log even when its interactive progress renderer is
                // unavailable inside the Unity Editor.
                string args = "pip install --verbose --no-progress --color never --python " +
                    QuoteArgument(pythonPath) + " -r " + QuoteArgument(file);

                bool isPyTorchReq = (!string.IsNullOrEmpty(options.PyTorchRequirementsPath) &&
                                     string.Equals(file, options.PyTorchRequirementsPath, StringComparison.OrdinalIgnoreCase)) ||
                                    file.EndsWith("requirements-pytorch.txt", StringComparison.OrdinalIgnoreCase);

                if (isPyTorchReq && IsCudaHardwareAvailable())
                {
                    // Prioritize PyTorch official CUDA wheel index for NVIDIA hardware
                    args += " --extra-index-url https://download.pytorch.org/whl/cu121";

                    // If existing environment had a CPU-only build of torch installed,
                    // force uv to upgrade torch and torchvision to the CUDA build.
                    if (existingRuntime != null && existingRuntime.HasTorch && !existingRuntime.HasCuda)
                    {
                        onLog?.Invoke("[TexMotion] Upgrading existing CPU PyTorch build to CUDA 12.1 build...");
                        args += " --reinstall-package torch --reinstall-package torchvision";
                    }
                }

                if (options.Offline) args += " --offline";
                var install = await RunProcessAsync(uv.ExecutablePath, args, onLog, 0, cancellationToken, p =>
                {
                    // uv does not provide a stable machine-readable progress format;
                    // map process liveness to this profile's bounded range.
                    progress?.Report(new PythonEnvironmentProgress(
                        Mathf.Lerp(start, end, p),
                        "install",
                        TexMotionLocalization.TrFormat(
                            TexMotionLocalization.InstallingRequirements,
                            Path.GetFileName(file))));
                }, uvCacheDirectory).ConfigureAwait(false);
                result.ExitCode = install.ExitCode;
                if (install.ExitCode != 0)
                {
                    string detail = options.Offline
                        ? TexMotionLocalization.Tr(TexMotionLocalization.OfflineInstallFailed)
                        : TexMotionLocalization.Tr(TexMotionLocalization.DependencyInstallFailed);
                    return Fail(result, detail, install, progress, onLog);
                }
                onLog?.Invoke(TexMotionLocalization.TrFormat(
                    TexMotionLocalization.FinishedInstalling,
                    Path.GetFileName(file)));
            }

            cancellationToken.ThrowIfCancellationRequested();
            result.Runtime = VideoMotionJobRunner.ProbePythonExecutable(pythonPath);
            result.Success = result.Runtime != null && result.Runtime.IsAvailable;
            result.WasAlreadyConfigured = hadPythonBeforeSetup && result.Success;
            if (!result.Success)
            {
                result.ErrorMessage = TexMotionLocalization.Tr(TexMotionLocalization.EnvironmentVerificationFailed);
                result.Diagnostics = result.Runtime?.ErrorMessage;
                onLog?.Invoke("[TexMotion] " + result.ErrorMessage);
                progress?.Report(new PythonEnvironmentProgress(0f, "error", result.ErrorMessage));
                return result;
            }

            progress?.Report(new PythonEnvironmentProgress(
                1f,
                "completed",
                TexMotionLocalization.Tr(TexMotionLocalization.EnvironmentReady)));
            onLog?.Invoke(TexMotionLocalization.TrFormat(
                TexMotionLocalization.LocalEnvironmentReadyLog,
                pythonPath));
            return result;
        }

        /// <summary>Convenience wrapper for the default lightweight offline-capable profile.</summary>
        public static Task<PythonEnvironmentSetupResult> EnsureLightweightEnvironmentAsync(
            IProgress<PythonEnvironmentProgress> progress = null,
            Action<string> onLog = null,
            CancellationToken cancellationToken = default,
            string projectRoot = null,
            bool offline = false)
        {
            var options = PythonEnvironmentOptions.CreateDefault(PythonDependencyProfile.Lightweight);
            if (!string.IsNullOrEmpty(projectRoot))
            {
                options.ProjectRoot = projectRoot;
                options.EnvironmentDirectory = GetDefaultEnvironmentDirectory(projectRoot);
                options.RequirementsPath = FindRequirementsPath(projectRoot);
            }
            options.Offline = offline;
            return EnsureEnvironmentAsync(options, progress, onLog, cancellationToken);
        }

        /// <summary>Convenience wrapper for the opt-in heavyweight PyTorch profile.</summary>
        public static Task<PythonEnvironmentSetupResult> EnsurePyTorchEnvironmentAsync(
            IProgress<PythonEnvironmentProgress> progress = null,
            Action<string> onLog = null,
            CancellationToken cancellationToken = default,
            string projectRoot = null,
            bool offline = false)
        {
            var options = PythonEnvironmentOptions.CreateDefault(PythonDependencyProfile.PyTorch);
            if (!string.IsNullOrEmpty(projectRoot))
            {
                options.ProjectRoot = projectRoot;
                options.EnvironmentDirectory = GetDefaultEnvironmentDirectory(projectRoot);
                options.RequirementsPath = FindRequirementsPath(projectRoot);
                options.PyTorchRequirementsPath = FindPyTorchRequirementsPath(projectRoot);
            }
            options.Offline = offline;
            options.AllowMissingOptionalProfile = false;
            return EnsureEnvironmentAsync(options, progress, onLog, cancellationToken);
        }

        private static PythonEnvironmentSetupResult Fail(
            PythonEnvironmentSetupResult result,
            string message,
            ProcessResult process,
            IProgress<PythonEnvironmentProgress> progress,
            Action<string> onLog)
        {
            result.Success = false;
            result.ErrorMessage = message;
            result.Diagnostics = process == null ? null : CombineDiagnostics(process);
            if (process != null && process.ExitCode != 0) result.ExitCode = process.ExitCode;
            onLog?.Invoke("[TexMotion] " + message);
            if (process != null && !string.IsNullOrEmpty(process.Stdout))
            {
                onLog?.Invoke(TexMotionLocalization.TrFormat(
                    TexMotionLocalization.CapturedUvStdout,
                    process.Stdout));
            }
            if (process != null && !string.IsNullOrEmpty(process.Stderr))
            {
                // Keep stderr last so the actionable uv error remains in the
                // visible tail of the Settings log.
                onLog?.Invoke(TexMotionLocalization.TrFormat(
                    TexMotionLocalization.CapturedUvStderr,
                    process.Stderr));
            }
            progress?.Report(new PythonEnvironmentProgress(0f, "error", message));
            return result;
        }

        #endregion

        #region Requirements paths

        public static string FindRequirementsPath(string projectRoot = null)
        {
            string root = string.IsNullOrEmpty(projectRoot) ? FindProjectRoot() : projectRoot;
            string[] candidates =
            {
                Path.Combine(root, "Editor", "Video", "requirements.txt"),
                Path.Combine(root, "Packages", "com.k0ta0uchi.texmotion", "Editor", "Video", "requirements.txt")
            };
            foreach (string candidate in candidates) if (File.Exists(candidate)) return NormalizePath(candidate);
            return NormalizePath(candidates[0]);
        }

        public static string FindPyTorchRequirementsPath(string projectRoot = null)
        {
            string root = string.IsNullOrEmpty(projectRoot) ? FindProjectRoot() : projectRoot;
            string[] candidates =
            {
                Path.Combine(root, "Editor", "Video", "requirements-pytorch.txt"),
                Path.Combine(root, "Packages", "com.k0ta0uchi.texmotion", "Editor", "Video", "requirements-pytorch.txt")
            };
            foreach (string candidate in candidates) if (File.Exists(candidate)) return NormalizePath(candidate);
            return NormalizePath(candidates[0]);
        }

        #endregion

        #region Process helpers

        private sealed class ProcessResult
        {
            public int ExitCode;
            public string Stdout;
            public string Stderr;
        }

        private static async Task<ProcessResult> RunProcessAsync(
            string executable,
            string arguments,
            Action<string> onLog,
            int timeoutMs,
            CancellationToken cancellationToken,
            Action<float> onProgress = null,
            string uvCacheDirectory = null)
        {
            var psi = new ProcessStartInfo
            {
                FileName = executable,
                Arguments = arguments ?? string.Empty,
                UseShellExecute = false,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                CreateNoWindow = true,
                StandardOutputEncoding = Encoding.UTF8,
                StandardErrorEncoding = Encoding.UTF8
            };
            // These environment overrides complement the explicit CLI flags and
            // prevent carriage-return progress bars from hiding useful output.
            psi.EnvironmentVariables["UV_NO_PROGRESS"] = "1";
            psi.EnvironmentVariables["UV_COLOR"] = "never";
            if (!string.IsNullOrEmpty(uvCacheDirectory))
            {
                psi.EnvironmentVariables["UV_CACHE_DIR"] = uvCacheDirectory;
            }

            using var process = new Process { StartInfo = psi, EnableRaisingEvents = true };
            var stdout = new StringBuilder();
            var stderr = new StringBuilder();
            var completion = new TaskCompletionSource<int>();
            process.OutputDataReceived += (s, e) =>
            {
                if (e.Data == null) return;
                lock (stdout) stdout.AppendLine(e.Data);
                onLog?.Invoke(e.Data);
            };
            process.ErrorDataReceived += (s, e) =>
            {
                if (e.Data == null) return;
                lock (stderr) stderr.AppendLine(e.Data);
                onLog?.Invoke(e.Data);
            };
            process.Exited += (s, e) => completion.TrySetResult(process.ExitCode);

            using var cancellation = cancellationToken.Register(() =>
            {
                VideoMotionJobRunner.KillProcessTree(process);
                completion.TrySetCanceled(cancellationToken);
            });

            onLog?.Invoke(TexMotionLocalization.TrFormat(
                TexMotionLocalization.StartingProcess,
                executable,
                arguments ?? string.Empty));
            if (!process.Start()) throw new InvalidOperationException("Failed to launch process: " + executable);
            process.BeginOutputReadLine();
            process.BeginErrorReadLine();
            onProgress?.Invoke(0.05f);

            Task completed = completion.Task;
            if (timeoutMs > 0)
            {
                // Keep timeout and cancellation as separate signals. Passing the
                // cancellation token to Task.Delay can race the process callback
                // and turn a user cancellation into a misleading timeout error.
                Task timeout = Task.Delay(timeoutMs);
                Task winner = await Task.WhenAny(completed, timeout).ConfigureAwait(false);
                cancellationToken.ThrowIfCancellationRequested();
                if (winner != completed)
                {
                    VideoMotionJobRunner.KillProcessTree(process);
                    throw new TimeoutException("Process timed out: " + executable);
                }
            }
            else
            {
                await completed.ConfigureAwait(false);
            }

            process.WaitForExit();
            onProgress?.Invoke(1f);
            string outText;
            string errText;
            lock (stdout) outText = stdout.ToString();
            lock (stderr) errText = stderr.ToString();
            onLog?.Invoke(TexMotionLocalization.TrFormat(
                TexMotionLocalization.ProcessExited,
                process.ExitCode));
            return new ProcessResult { ExitCode = process.ExitCode, Stdout = outText, Stderr = errText };
        }

        /// <summary>
        /// Returns the configured uv cache when it is usable. A global cache
        /// override is common on build machines, but a read-only/removable
        /// drive makes uv fail before dependency resolution can finish. In
        /// that case keep setup self-contained by using a project-local cache.
        /// </summary>
        private static string ResolveUvCacheDirectory(
            string projectRoot,
            PythonDependencyProfile profile,
            Action<string> onLog)
        {
            string configured = null;
            try { configured = Environment.GetEnvironmentVariable("UV_CACHE_DIR"); } catch { }
            if (string.IsNullOrWhiteSpace(configured)) return null;

            long minimumFreeBytes = profile == PythonDependencyProfile.PyTorch
                ? 6L * 1024L * 1024L * 1024L
                : 1L * 1024L * 1024L * 1024L;
            configured = NormalizePath(configured.Trim().Trim('"'));
            string configuredFailure;
            if (IsWritableUvCache(configured, minimumFreeBytes, out configuredFailure)) return configured;

            string root = string.IsNullOrEmpty(projectRoot) ? FindProjectRoot() : projectRoot;
            if (string.IsNullOrEmpty(root)) root = Path.GetTempPath();
            string fallback = NormalizePath(Path.Combine(root, ".texmotion-uv-cache"));
            string fallbackFailure;
            if (IsWritableUvCache(fallback, minimumFreeBytes, out fallbackFailure))
            {
                onLog?.Invoke(TexMotionLocalization.TrFormat(
                    TexMotionLocalization.UvCacheUnavailableDetailed,
                    configured,
                    configuredFailure));
                onLog?.Invoke(TexMotionLocalization.TrFormat(
                    TexMotionLocalization.UsingProjectUvCache,
                    fallback));
                return fallback;
            }

            onLog?.Invoke(TexMotionLocalization.TrFormat(
                TexMotionLocalization.UvCacheRetained,
                fallbackFailure,
                configured));
            return configured;
        }

        private static bool IsWritableUvCache(string directory, long minimumFreeBytes, out string failureReason)
        {
            failureReason = null;
            if (string.IsNullOrEmpty(directory)) return false;
            try
            {
                Directory.CreateDirectory(directory);
                string root = Path.GetPathRoot(Path.GetFullPath(directory));
                if (!string.IsNullOrEmpty(root))
                {
                    var drive = new DriveInfo(root);
                    if (drive.IsReady && drive.AvailableFreeSpace < minimumFreeBytes)
                    {
                        failureReason = "only " + FormatUvByteCount(drive.AvailableFreeSpace) +
                            " free; requires at least " + FormatUvByteCount(minimumFreeBytes);
                        return false;
                    }
                }

                string probe = Path.Combine(directory, ".texmotion-write-test-" + Guid.NewGuid().ToString("N"));
                using (var stream = new FileStream(probe, FileMode.CreateNew, FileAccess.Write, FileShare.None))
                {
                    stream.WriteByte(0);
                }
                File.Delete(probe);

                // uv may touch the marker in an existing source-distribution
                // cache. Check it too, because the cache root can be writable
                // while a stale marker has stricter permissions.
                string[] sourceCaches = Directory.GetDirectories(directory, "sdists-v*", SearchOption.TopDirectoryOnly);
                foreach (string sourceCache in sourceCaches)
                {
                    string marker = Path.Combine(sourceCache, ".git");
                    if (!File.Exists(marker)) continue;
                    using (var markerStream = new FileStream(marker, FileMode.Open, FileAccess.ReadWrite, FileShare.ReadWrite)) { }
                }
                return true;
            }
            catch (Exception ex)
            {
                failureReason = ex.Message;
                return false;
            }
        }

        private static string FormatUvByteCount(long bytes)
        {
            const double gigabyte = 1024d * 1024d * 1024d;
            return (bytes / gigabyte).ToString("0.##") + " GB";
        }

        private static string CombineDiagnostics(ProcessResult result)
        {
            if (result == null) return null;
            return (result.Stdout ?? string.Empty) + (result.Stderr ?? string.Empty);
        }

        private static string FirstNonEmptyLine(string text)
        {
            if (string.IsNullOrEmpty(text)) return null;
            string[] lines = text.Split(new[] { '\r', '\n' }, StringSplitOptions.RemoveEmptyEntries);
            return lines.Length == 0 ? null : lines[0].Trim();
        }

        private static string QuoteArgument(string value)
        {
            if (value == null) return "\"\"";
            return "\"" + value.Replace("\\", "\\\\").Replace("\"", "\\\"") + "\"";
        }

        internal static string NormalizePath(string path)
        {
            if (string.IsNullOrEmpty(path)) return path;
            try
            {
                string fullPath = Path.GetFullPath(path);
                string root = Path.GetPathRoot(fullPath);
                if (!string.IsNullOrEmpty(root) && string.Equals(fullPath, root, StringComparison.OrdinalIgnoreCase)) return root;
                return fullPath.TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar);
            }
            catch { return path; }
        }

        private static bool IsWindowsPlatform => Application.platform == RuntimePlatform.WindowsEditor ||
                                                  Environment.OSVersion.Platform == PlatformID.Win32NT;

        /// <summary>
        /// Checks whether the local machine has an NVIDIA GPU and driver capable of running CUDA.
        /// On Windows, the presence of nvcuda.dll in System32 indicates NVIDIA driver installation.
        /// </summary>
        public static bool IsCudaHardwareAvailable()
        {
            try
            {
                if (IsWindowsPlatform)
                {
                    string systemDir = Environment.GetFolderPath(Environment.SpecialFolder.System);
                    if (!string.IsNullOrEmpty(systemDir) && File.Exists(Path.Combine(systemDir, "nvcuda.dll")))
                    {
                        return true;
                    }
                }
                else
                {
                    string[] linuxLibs = { "/usr/lib/x86_64-linux-gnu/libcuda.so", "/usr/local/cuda/lib64/libcudart.so" };
                    foreach (string lib in linuxLibs)
                    {
                        if (File.Exists(lib)) return true;
                    }
                }
            }
            catch { }
            return false;
        }

        #endregion
    }
}
