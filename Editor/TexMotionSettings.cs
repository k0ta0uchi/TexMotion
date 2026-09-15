using System;
using System.Collections.Generic;
using System.IO;
using UnityEngine;

#if UNITY_EDITOR
using UnityEditor;
#endif

namespace TexMotion.Editor
{
    [FilePath("ProjectSettings/TexMotionSettings.asset", FilePathAttribute.Location.ProjectFolder)]
    public class TexMotionSettings : ScriptableSingleton<TexMotionSettings>
    {
        [Header("Hugging Face Repository")]
        public string HuggingFaceRepo = "k0ta0uchi/Llama-3-Kimodo-SMPLX-RP-v1-GGUF";
        public string MotionModelFileName = "kimodo-smplx-rp-v1-f32.gguf";
        public string TextBundleDirName = "llm2vec-text-bundle";

        [Header("Model Storage Mode")]
        public bool UseCustomLocalPath = false;
        public string CustomLocalModelDirectory = "";

        [Header("Python Video Pose Extraction")]
        public string CustomPythonExecutablePath = "";

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
