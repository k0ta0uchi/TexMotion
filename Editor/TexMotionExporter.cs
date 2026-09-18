using System;
using System.IO;
using UnityEditor;
using UnityEngine;

namespace TexMotion.Editor
{
    public static class TexMotionExporter
    {
        private const string PackageFileName = "TexMotion.unitypackage";

        [MenuItem("Tools/TexMotion/Export UnityPackage", false, 100)]
        public static void ExportPackageMenu()
        {
            string exportPath = EditorUtility.SaveFilePanel(
                TexMotionLocalization.Tr(TexMotionLocalization.ExportUnityPackage),
                "",
                PackageFileName,
                "unitypackage");
            if (!string.IsNullOrEmpty(exportPath))
            {
                ExportPackageTo(exportPath);
                EditorUtility.RevealInFinder(exportPath);
            }
        }

        public static void ExportPackageBatch()
        {
            string workspacePackage = @"C:\Workspace\TexMotion\TexMotion.unitypackage";
            ExportPackageTo(workspacePackage, false);

            string projectRoot = Path.GetFullPath(Path.Combine(Application.dataPath, ".."));
            string outputPath = Path.Combine(projectRoot, PackageFileName);
            if (!string.Equals(outputPath, workspacePackage, StringComparison.OrdinalIgnoreCase))
            {
                try
                {
                    File.Copy(workspacePackage, outputPath, true);
                }
                catch {}
            }
        }

        public static void ExportPackageTo(string outputPath, bool interactive = false)
        {
            string packageRoot = "Packages/com.k0ta0uchi.texmotion";
            if (!Directory.Exists(packageRoot))
            {
                // Fallback to local path
                packageRoot = "Assets/TexMotion";
            }

            string[] assetPaths = new string[]
            {
                "Packages/com.k0ta0uchi.texmotion",
                "Assets/TexMotion"
            };

            var existingAssets = new System.Collections.Generic.List<string>();
            foreach (var p in assetPaths)
            {
                if (Directory.Exists(p) || File.Exists(p))
                {
                    existingAssets.Add(p);
                }
            }

            if (existingAssets.Count == 0)
            {
                // If in stand-alone package directory
                existingAssets.Add("Editor");
                existingAssets.Add("Plugins");
                existingAssets.Add("Runtime");
                existingAssets.Add("package.json");
                existingAssets.Add("README.md");
            }

            Debug.Log(TexMotionLocalization.TrFormat(TexMotionLocalization.ExportingPackage, outputPath));
            ExportPackageOptions options = ExportPackageOptions.Recurse;
            if (interactive && !Application.isBatchMode)
            {
                options |= ExportPackageOptions.Interactive;
            }

            AssetDatabase.ExportPackage(
                existingAssets.ToArray(),
                outputPath,
                options
            );
            Debug.Log(TexMotionLocalization.Tr(TexMotionLocalization.ExportComplete));
        }
    }
}
