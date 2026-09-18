using System;
using System.Collections.Generic;
using System.IO;
using TexMotion.Editor.VRChat;
using TexMotion.Runtime.VRChat;
using UnityEditor;
using UnityEditor.Animations;
using UnityEngine;

namespace TexMotion.Editor.Motion
{
    public class MotionLibraryItem
    {
        public string Name;
        public string AssetPath;
        public AnimationClip Clip;
        public float Duration;
        public float FrameRate;
        public bool IsLoop;
        public bool IsAppliedToAvatar;
        public GameObject ModularAvatarObject;
    }

    public static class MotionLibraryManager
    {
        public const string GeneratedDirectory = "Assets/TexMotion/Generated";

        public static List<MotionLibraryItem> ScanLibrary(GameObject targetAvatar = null)
        {
            return ScanGeneratedMotions(targetAvatar);
        }

        public static List<MotionLibraryItem> ScanGeneratedMotions(GameObject targetAvatar = null)
        {
            var list = new List<MotionLibraryItem>();

            if (!Directory.Exists(GeneratedDirectory))
            {
                return list;
            }

            string[] animFiles = Directory.GetFiles(GeneratedDirectory, "Anim_*.anim", SearchOption.AllDirectories);

            foreach (string file in animFiles)
            {
                string unifiedPath = file.Replace("\\", "/");
                var clip = AssetDatabase.LoadAssetAtPath<AnimationClip>(unifiedPath);
                if (clip == null) continue;

                string rawName = Path.GetFileNameWithoutExtension(unifiedPath);
                string motionName = rawName.StartsWith("Anim_") ? rawName.Substring(5) : rawName;

                var item = new MotionLibraryItem
                {
                    Name = motionName,
                    AssetPath = unifiedPath,
                    Clip = clip,
                    Duration = clip.length,
                    FrameRate = clip.frameRate,
                    IsLoop = clip.isLooping
                };

                // Check if applied to target avatar
                if (targetAvatar != null)
                {
                    CheckAvatarApplicationStatus(targetAvatar, item);
                }

                list.Add(item);
            }

            return list;
        }

        private static void CheckAvatarApplicationStatus(GameObject targetAvatar, MotionLibraryItem item)
        {
            // 1. Check Modular Avatar GameObject
            string maObjName = $"TexMotion_{item.Name}";
            Transform maChild = targetAvatar.transform.Find(maObjName);
            if (maChild != null)
            {
                item.IsAppliedToAvatar = true;
                item.ModularAvatarObject = maChild.gameObject;
                return;
            }

            // 2. Check Direct VRCExpressionsMenu & Parameters
            Component descriptor = FindAvatarDescriptor(targetAvatar);
            if (descriptor != null)
            {
                var exprParams = descriptor.GetType().GetField("expressionParameters")?.GetValue(descriptor) as ScriptableObject;
                if (exprParams != null)
                {
                    string paramName = $"TexMotion_{item.Name}";
                    var paramsField = exprParams.GetType().GetField("parameters");
                    var arr = paramsField?.GetValue(exprParams) as Array;
                    if (arr != null)
                    {
                        foreach (object p in arr)
                        {
                            var n = p.GetType().GetField("name")?.GetValue(p) as string;
                            if (n == paramName)
                            {
                                item.IsAppliedToAvatar = true;
                                return;
                            }
                        }
                    }
                }
            }
        }

        public static void RemoveFromAvatar(GameObject targetAvatar, MotionLibraryItem item)
        {
            if (targetAvatar == null || item == null) return;

            string paramName = $"TexMotion_{item.Name}";

            // 1. Remove Modular Avatar GameObject if exists
            string maObjName = $"TexMotion_{item.Name}";
            Transform maChild = targetAvatar.transform.Find(maObjName);
            if (maChild != null)
            {
                Undo.DestroyObjectImmediate(maChild.gameObject);
            }

            // 2. Clean from Direct VRCAvatarDescriptor, ExpressionsMenu, Parameters
            Component descriptor = FindAvatarDescriptor(targetAvatar);
            if (descriptor != null)
            {
                // Remove from ExpressionParameters
                var exprParams = descriptor.GetType().GetField("expressionParameters")?.GetValue(descriptor) as ScriptableObject;
                if (exprParams != null)
                {
                    RemoveParameterFromExpressionParameters(exprParams, paramName);
                }

                // Remove from ExpressionsMenu
                var exprMenu = descriptor.GetType().GetField("expressionsMenu")?.GetValue(descriptor) as ScriptableObject;
                if (exprMenu != null)
                {
                    RemoveControlFromExpressionsMenu(exprMenu, item.Name, paramName);
                }

                // Remove state from Action/FX AnimatorControllers
                RemoveStateFromAvatarControllers(descriptor, item.Name, paramName);
            }

            item.IsAppliedToAvatar = false;
            item.ModularAvatarObject = null;

            EditorUtility.SetDirty(targetAvatar);
            AssetDatabase.SaveAssets();
        }

        public static void DeleteMotionFiles(MotionLibraryItem item, GameObject targetAvatar = null)
        {
            if (item == null) return;

            if (targetAvatar != null && item.IsAppliedToAvatar)
            {
                RemoveFromAvatar(targetAvatar, item);
            }

            // Delete .anim file
            if (File.Exists(item.AssetPath))
            {
                AssetDatabase.DeleteAsset(item.AssetPath);
            }

            // Delete controller if exists
            string ctrlPath = $"{GeneratedDirectory}/Ctrl_{item.Name}.controller";
            if (File.Exists(ctrlPath))
            {
                AssetDatabase.DeleteAsset(ctrlPath);
            }

            AssetDatabase.Refresh();
        }

        private static Component FindAvatarDescriptor(GameObject avatar)
        {
            Type descriptorType = Type.GetType("VRC.SDK3.Avatars.Components.VRCAvatarDescriptor, VRC.SDK3.Avatars");
            if (descriptorType == null)
            {
                foreach (var asm in AppDomain.CurrentDomain.GetAssemblies())
                {
                    descriptorType = asm.GetType("VRC.SDK3.Avatars.Components.VRCAvatarDescriptor");
                    if (descriptorType != null) break;
                }
            }
            return descriptorType != null ? avatar.GetComponent(descriptorType) : null;
        }

        private static void RemoveParameterFromExpressionParameters(ScriptableObject exprParams, string paramName)
        {
            var paramsField = exprParams.GetType().GetField("parameters");
            var arr = paramsField?.GetValue(exprParams) as Array;
            if (arr == null) return;

            Type paramType = exprParams.GetType().GetNestedType("Parameter");
            var keptList = new List<object>();

            foreach (object p in arr)
            {
                string pName = p.GetType().GetField("name")?.GetValue(p) as string;
                if (pName != paramName)
                {
                    keptList.Add(p);
                }
            }

            if (keptList.Count != arr.Length)
            {
                Array newArr = Array.CreateInstance(paramType, keptList.Count);
                for (int i = 0; i < keptList.Count; i++)
                {
                    newArr.SetValue(keptList[i], i);
                }
                paramsField.SetValue(exprParams, newArr);
                EditorUtility.SetDirty(exprParams);
            }
        }

        private static void RemoveControlFromExpressionsMenu(ScriptableObject rootMenu, string motionName, string paramName)
        {
            if (rootMenu == null) return;
            RemoveControlsRecursive(rootMenu, motionName, paramName, new HashSet<ScriptableObject>());
        }

        /// <summary>
        /// Removes matching controls from the whole menu tree (root + all TexMotion NextPage entries),
        /// then collapses pages back up: a page left empty, or holding only its originally
        /// displaced entry, is merged back into the parent's NextPage slot.
        /// </summary>
        private static bool RemoveControlsRecursive(ScriptableObject menu, string motionName, string paramName, HashSet<ScriptableObject> visited)
        {
            if (menu == null || !visited.Add(menu)) return false;

            var so = new SerializedObject(menu);
            var controls = so.FindProperty("controls");
            if (controls == null || !controls.isArray) return false;

            bool removedAny = false;
            for (int i = controls.arraySize - 1; i >= 0; i--)
            {
                var elem = controls.GetArrayElementAtIndex(i);
                var nameProp = elem.FindPropertyRelative("name");
                var paramProp = elem.FindPropertyRelative("parameter")?.FindPropertyRelative("name");

                string n = nameProp != null ? nameProp.stringValue : null;
                string p = paramProp != null ? paramProp.stringValue : null;
                if (n == motionName || p == paramName)
                {
                    controls.DeleteArrayElementAtIndex(i);
                    removedAny = true;
                }
            }
            if (removedAny)
            {
                so.ApplyModifiedProperties();
                EditorUtility.SetDirty(menu);
            }

            // Recurse into managed pages first (post-order), then collapse upward.
            var childPages = CollectTexMotionPages(menu, new HashSet<ScriptableObject>());
            foreach (var page in childPages)
            {
                RemoveControlsRecursive(page, motionName, paramName, visited);
            }

            CollapseChildPages(menu, new HashSet<ScriptableObject>());
            return removedAny;
        }

        private static List<ScriptableObject> CollectTexMotionPages(ScriptableObject menu, HashSet<ScriptableObject> visited)
        {
            var pages = new List<ScriptableObject>();
            var controls = new SerializedObject(menu).FindProperty("controls");
            if (controls == null || !controls.isArray) return pages;

            for (int i = 0; i < controls.arraySize; i++)
            {
                if (VrcMenuUtility.TryGetTexMotionPage(controls.GetArrayElementAtIndex(i), out ScriptableObject page) && visited.Add(page))
                {
                    pages.Add(page);
                }
            }
            return pages;
        }

        /// <summary>
        /// If a managed page has no entries left, or only its originally displaced emote,
        /// that entry is moved back into the parent's NextPage slot (restoring the original
        /// layout) and the page link is removed together with its generated asset.
        /// Collapses recursively up the chain.
        /// </summary>
        private static void CollapseChildPages(ScriptableObject menu, HashSet<ScriptableObject> visited)
        {
            if (menu == null || !visited.Add(menu)) return;

            // Depth-first: fold grandchildren into children before judging this menu.
            foreach (var page in CollectTexMotionPages(menu, new HashSet<ScriptableObject>()))
            {
                CollapseChildPages(page, visited);
            }

            bool changed = true;
            while (changed)
            {
                changed = false;
                var so = new SerializedObject(menu);
                var controls = so.FindProperty("controls");
                if (controls == null || !controls.isArray) return;

                for (int i = 0; i < controls.arraySize; i++)
                {
                    var elem = controls.GetArrayElementAtIndex(i);
                    if (!VrcMenuUtility.TryGetTexMotionPage(elem, out ScriptableObject page)) continue;

                    var pageControls = new SerializedObject(page).FindProperty("controls");
                    int count = pageControls != null && pageControls.isArray ? pageControls.arraySize : 0;
                    if (count > 1) continue;

                    if (count == 1)
                    {
                        // Move the surviving entry back into the parent's NextPage slot.
                        VrcMenuUtility.ApplyControlData(elem, VrcMenuUtility.CaptureControl(pageControls.GetArrayElementAtIndex(0)));
                    }
                    else
                    {
                        controls.DeleteArrayElementAtIndex(i);
                    }

                    so.ApplyModifiedProperties();
                    EditorUtility.SetDirty(menu);
                    changed = true;

                    string pagePath = AssetDatabase.GetAssetPath(page);
                    if (!string.IsNullOrEmpty(pagePath) &&
                        pagePath.StartsWith(MotionLibraryManager.GeneratedDirectory, StringComparison.OrdinalIgnoreCase))
                    {
                        AssetDatabase.DeleteAsset(pagePath);
                    }
                    break; // indices shifted; rescan
                }
            }
        }

        private static void RemoveStateFromAvatarControllers(Component descriptor, string motionName, string paramName)
        {
            var field = descriptor.GetType().GetField("baseAnimationLayers");
            var layersArray = field?.GetValue(descriptor) as Array;
            if (layersArray == null) return;

            foreach (object layerObj in layersArray)
            {
                if (layerObj == null) continue;
                var animCtrlField = layerObj.GetType().GetField("animatorController");
                var ctrl = animCtrlField?.GetValue(layerObj) as AnimatorController;
                if (ctrl == null) continue;

                foreach (var l in ctrl.layers)
                {
                    if (l.name == "TexMotion")
                    {
                        var sm = l.stateMachine;
                        for (int s = sm.states.Length - 1; s >= 0; s--)
                        {
                            if (sm.states[s].state.name == motionName)
                            {
                                sm.RemoveState(sm.states[s].state);
                                EditorUtility.SetDirty(ctrl);
                            }
                        }
                    }
                }
            }
        }
    }
}
