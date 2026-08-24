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
            string exportPath = EditorUtility.SaveFilePanel("Export TexMotion UnityPackage", "", PackageFileName, "unitypackage");
            if (!string.IsNullOrEmpty(exportPath))
            {
                ExportPackageTo(exportPath);
                EditorUtility.RevealInFinder(exportPath);
            }
        }

        public static void ExportPackageBatch()
        {
            string projectRoot = Path.GetFullPath(Path.Combine(Application.dataPath, ".."));
            string outputPath = Path.Combine(projectRoot, PackageFileName);
            ExportPackageTo(outputPath);
        }

        public static void ExportPackageTo(string outputPath)
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

            Debug.Log($"[TexMotion] Exporting package to: {outputPath}");
            AssetDatabase.ExportPackage(
                existingAssets.ToArray(),
                outputPath,
                ExportPackageOptions.Recurse | ExportPackageOptions.Interactive
            );
            Debug.Log("[TexMotion] Export complete!");
        }
    }
}
