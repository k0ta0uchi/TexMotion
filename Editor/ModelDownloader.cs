using System;
using System.Collections.Generic;
using System.IO;
using System.Threading;
using System.Threading.Tasks;
using UnityEngine;
using UnityEngine.Networking;

#if UNITY_EDITOR
using UnityEditor;
#endif

namespace TexMotion.Editor
{
    public struct DownloadProgress
    {
        public float OverallProgress; // 0.0 to 1.0
        public float FileProgress;    // 0.0 to 1.0
        public int CompletedFiles;
        public int TotalFiles;
        public string CurrentFileName;
        public string StatusText;
    }

    /// <summary>
    /// Handles asynchronous downloading of GGUF model files and text bundle components from Hugging Face.
    /// </summary>
    public static class ModelDownloader
    {
        public static async Task DownloadAllModelsAsync(
            TexMotionSettings settings,
            IProgress<DownloadProgress> progress,
            CancellationToken cancellationToken = default)
        {
            string repo = settings.HuggingFaceRepo;
            string baseDestDir = settings.GetEffectiveModelDirectory();
            string bundleDestDir = Path.Combine(baseDestDir, settings.TextBundleDirName);

            Directory.CreateDirectory(baseDestDir);
            Directory.CreateDirectory(bundleDestDir);

            var filesToDownload = new List<(string remotePath, string localPath)>();

            // 1. Motion model GGUF
            filesToDownload.Add((
                settings.MotionModelFileName,
                Path.Combine(baseDestDir, settings.MotionModelFileName)
            ));

            // 2. Text bundle files
            foreach (var bundleFile in settings.GetRequiredBundleFiles())
            {
                filesToDownload.Add((
                    $"{settings.TextBundleDirName}/{bundleFile}",
                    Path.Combine(bundleDestDir, bundleFile)
                ));
            }

            int totalFiles = filesToDownload.Count;

            for (int i = 0; i < totalFiles; i++)
            {
                var (remoteRelPath, localFullPath) = filesToDownload[i];
                string fileName = Path.GetFileName(localFullPath);

                // Skip if file exists and has size > 0
                if (File.Exists(localFullPath) && new FileInfo(localFullPath).Length > 0)
                {
                    progress?.Report(new DownloadProgress
                    {
                        OverallProgress = (float)(i + 1) / totalFiles,
                        FileProgress = 1.0f,
                        CompletedFiles = i + 1,
                        TotalFiles = totalFiles,
                        CurrentFileName = fileName,
                        StatusText = TexMotionLocalization.TrFormat(
                            TexMotionLocalization.SkippedExistingFile,
                            fileName,
                            i + 1,
                            totalFiles)
                    });
                    continue;
                }

                await DownloadSingleFileAsync(
                    repo,
                    remoteRelPath,
                    localFullPath,
                    i,
                    totalFiles,
                    progress,
                    cancellationToken);
            }

            progress?.Report(new DownloadProgress
            {
                OverallProgress = 1.0f,
                FileProgress = 1.0f,
                CompletedFiles = totalFiles,
                TotalFiles = totalFiles,
                CurrentFileName = TexMotionLocalization.Tr(TexMotionLocalization.Complete),
                StatusText = TexMotionLocalization.Tr(TexMotionLocalization.AllModelsReady)
            });
        }

        private static async Task DownloadSingleFileAsync(
            string repoName,
            string remoteRelativePath,
            string destinationPath,
            int fileIndex,
            int totalFiles,
            IProgress<DownloadProgress> progress,
            CancellationToken cancellationToken)
        {
            string downloadUrl = $"https://huggingface.co/{repoName}/resolve/main/{remoteRelativePath}";
            string tempPath = destinationPath + ".tmp";
            string fileName = Path.GetFileName(destinationPath);

            string dir = Path.GetDirectoryName(destinationPath);
            if (!Directory.Exists(dir)) Directory.CreateDirectory(dir);

            if (File.Exists(tempPath)) File.Delete(tempPath);

            using (var request = UnityWebRequest.Get(downloadUrl))
            {
                request.downloadHandler = new DownloadHandlerFile(tempPath);
                var operation = request.SendWebRequest();

                while (!operation.isDone)
                {
                    if (cancellationToken.IsCancellationRequested)
                    {
                        request.Abort();
                        if (File.Exists(tempPath)) File.Delete(tempPath);
                        throw new OperationCanceledException(TexMotionLocalization.Tr(TexMotionLocalization.DownloadCancelled));
                    }

                    float fileProg = request.downloadProgress;
                    float overall = ((float)fileIndex + fileProg) / totalFiles;
                    long downloaded = (long)request.downloadedBytes;

                    progress?.Report(new DownloadProgress
                    {
                        OverallProgress = overall,
                        FileProgress = fileProg,
                        CompletedFiles = fileIndex,
                        TotalFiles = totalFiles,
                        CurrentFileName = fileName,
                        StatusText = TexMotionLocalization.TrFormat(
                            TexMotionLocalization.DownloadingFile,
                            fileIndex + 1,
                            totalFiles,
                            fileName,
                            FormatBytes(downloaded),
                            fileProg * 100f)
                    });

                    await Task.Delay(100, cancellationToken);
                }

                if (request.result != UnityWebRequest.Result.Success)
                {
                    if (File.Exists(tempPath)) File.Delete(tempPath);
                    throw new Exception(TexMotionLocalization.TrFormat(
                        TexMotionLocalization.DownloadFailed,
                        fileName,
                        request.error,
                        downloadUrl));
                }

                if (File.Exists(destinationPath))
                {
                    File.Delete(destinationPath);
                }
                File.Move(tempPath, destinationPath);
            }
        }

        private static string FormatBytes(long bytes)
        {
            if (bytes <= 0) return "0 B";
            string[] units = { "B", "KB", "MB", "GB", "TB" };
            int digit = (int)Math.Floor(Math.Log(bytes, 1024));
            digit = Math.Min(digit, units.Length - 1);
            return $"{bytes / Math.Pow(1024, digit):F2} {units[digit]}";
        }
    }
}
