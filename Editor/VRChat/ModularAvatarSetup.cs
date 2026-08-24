using System;
using System.IO;
using System.Reflection;
using TexMotion.Runtime.VRChat;
using UnityEditor;
using UnityEditor.Animations;
using UnityEditor.PackageManager;
using UnityEditor.PackageManager.Requests;
using UnityEngine;

namespace TexMotion.Editor.VRChat
{
    /// <summary>
    /// Automates non-destructive VRChat avatar setup using Modular Avatar, with fallback to direct VRCAvatarDescriptor setup.
    /// </summary>
    public static class ModularAvatarSetup
    {
        private const string ModularAvatarGitUrl = "https://github.com/bdunderscore/modular-avatar.git?path=Packages/nadena.dev.modular-avatar";
        private static AddRequest _installRequest;
        private static Action _onInstallComplete;

        /// <summary>
        /// Checks if Modular Avatar package is installed and active in the project.
        /// </summary>
        public static bool IsModularAvatarInstalled()
        {
            Type mergeAnimatorType = Type.GetType("nadena.dev.modular_avatar.core.ModularAvatarMergeAnimator, nadena.dev.modular-avatar.core");
            if (mergeAnimatorType != null) return true;

            // Also check all loaded assemblies
            foreach (var asm in AppDomain.CurrentDomain.GetAssemblies())
            {
                if (asm.GetType("nadena.dev.modular_avatar.core.ModularAvatarMergeAnimator") != null)
                {
                    return true;
                }
            }
            return false;
        }

        /// <summary>
        /// Triggers automated installation of Modular Avatar via Unity Package Manager.
        /// </summary>
        public static void InstallModularAvatar(Action onComplete = null)
        {
            _onInstallComplete = onComplete;
            _installRequest = Client.Add(ModularAvatarGitUrl);
            EditorApplication.update += MonitorInstallProgress;
        }

        private static void MonitorInstallProgress()
        {
            if (_installRequest == null)
            {
                EditorApplication.update -= MonitorInstallProgress;
                return;
            }

            if (_installRequest.IsCompleted)
            {
                EditorApplication.update -= MonitorInstallProgress;

                if (_installRequest.Status == StatusCode.Success)
                {
                    Debug.Log("[TexMotion] Modular Avatar installed successfully!");
                    EditorUtility.DisplayDialog("Modular Avatar Installed", "Modular Avatar was successfully installed into your project! You can now apply animations non-destructively.", "OK");
                    _onInstallComplete?.Invoke();
                }
                else
                {
                    Debug.LogError($"[TexMotion] Failed to install Modular Avatar: {_installRequest.Error.message}");
                    EditorUtility.DisplayDialog("Installation Failed", $"Could not install Modular Avatar automatically:\n{_installRequest.Error.message}\n\nPlease install Modular Avatar via VCC (VRChat Creator Companion).", "OK");
                }

                _installRequest = null;
                _onInstallComplete = null;
            }
        }

        /// <summary>
        /// Creates an AnimatorController and setups Modular Avatar components under the target avatar.
        /// </summary>
        public static GameObject SetupAvatarMotion(GameObject targetAvatar, AnimationClip motionClip, VrcMotionConfig config, string saveDirectory = "Assets/TexMotion/Generated")
        {
            if (targetAvatar == null) throw new ArgumentNullException(nameof(targetAvatar));
            if (motionClip == null) throw new ArgumentNullException(nameof(motionClip));

            if (!Directory.Exists(saveDirectory))
            {
                Directory.CreateDirectory(saveDirectory);
                AssetDatabase.Refresh();
            }

            string sanitizedName = SanitizeFileName(config.MotionName);
            string controllerPath = $"{saveDirectory}/Ctrl_{sanitizedName}.controller";

            // 1. Create AnimatorController
            var controller = CreateMotionAnimatorController(controllerPath, motionClip, config);

            // 2. Create or find child GameObject under avatar root
            string objectName = $"TexMotion_{sanitizedName}";
            Transform existingChild = targetAvatar.transform.Find(objectName);
            GameObject motionObj;
            if (existingChild != null)
            {
                motionObj = existingChild.gameObject;
            }
            else
            {
                motionObj = new GameObject(objectName);
                motionObj.transform.SetParent(targetAvatar.transform, false);
            }

            Undo.RegisterCreatedObjectUndo(motionObj, $"Setup TexMotion: {config.MotionName}");

            // 3. Attach Modular Avatar Components
            AttachModularAvatarComponents(motionObj, controller, config);

            EditorUtility.SetDirty(targetAvatar);
            AssetDatabase.SaveAssets();

            return motionObj;
        }

        private static AnimatorController CreateMotionAnimatorController(string path, AnimationClip clip, VrcMotionConfig config)
        {
            var controller = AnimatorController.CreateAnimatorControllerAtPath(path);

            string paramName = $"TexMotion_{SanitizeFileName(config.MotionName)}";
            controller.AddParameter(paramName, AnimatorControllerParameterType.Bool);

            var layer = controller.layers[0];
            layer.name = config.MotionName;
            layer.defaultWeight = 1.0f;

            var stateMachine = layer.stateMachine;

            // State: Idle (Empty)
            var idleState = stateMachine.AddState("Idle", new Vector3(250, 0, 0));
            idleState.motion = null;
            stateMachine.defaultState = idleState;

            // State: Motion
            var motionState = stateMachine.AddState(config.MotionName, new Vector3(250, 100, 0));
            motionState.motion = clip;

            // Attach VRChat Action Layer behaviours (identical semantics after Modular Avatar merge).

            // Idle: Action weight 0, tracking restored; one-shots also reset the parameter here.
            VrcLayerBehaviours.ApplyPlayableLayerControl(idleState, 0.0f, 0.2f);
            VrcLayerBehaviours.ApplyTrackingControl(idleState, false);
            if (config.MotionType == VrcMotionType.OneShotEmote)
            {
                VrcLayerBehaviours.ApplyParameterResetDriver(idleState, paramName);
            }

            // Motion: Action weight 1, limbs driven by animation, one-shot param reset on entry.
            VrcLayerBehaviours.ApplyPlayableLayerControl(motionState, 1.0f, 0.1f);
            VrcLayerBehaviours.ApplyTrackingControl(motionState, true);
            if (config.MotionType == VrcMotionType.OneShotEmote)
            {
                VrcLayerBehaviours.ApplyParameterResetDriver(motionState, paramName);
            }

            if (config.MotionType == VrcMotionType.OneShotEmote)
            {
                var toMotion = idleState.AddTransition(motionState);
                toMotion.AddCondition(AnimatorConditionMode.If, 0, paramName);
                toMotion.hasExitTime = false;
                toMotion.duration = 0.2f;

                var toIdle = motionState.AddTransition(idleState);
                toIdle.hasExitTime = true;
                toIdle.exitTime = 0.95f;
                toIdle.duration = 0.25f;
            }
            else // ToggleLoopPose
            {
                var toMotion = idleState.AddTransition(motionState);
                toMotion.AddCondition(AnimatorConditionMode.If, 0, paramName);
                toMotion.hasExitTime = false;
                toMotion.duration = 0.25f;

                var toIdle = motionState.AddTransition(idleState);
                toIdle.AddCondition(AnimatorConditionMode.IfNot, 0, paramName);
                toIdle.hasExitTime = false;
                toIdle.duration = 0.25f;
            }

            return controller;
        }

        private static void AttachModularAvatarComponents(GameObject motionObj, RuntimeAnimatorController controller, VrcMotionConfig config)
        {
            string paramName = $"TexMotion_{SanitizeFileName(config.MotionName)}";

            // 1. ModularAvatarMergeAnimator
            Type mergeAnimatorType = Type.GetType("nadena.dev.modular_avatar.core.ModularAvatarMergeAnimator, nadena.dev.modular-avatar.core");
            if (mergeAnimatorType == null)
            {
                foreach (var asm in AppDomain.CurrentDomain.GetAssemblies())
                {
                    mergeAnimatorType = asm.GetType("nadena.dev.modular_avatar.core.ModularAvatarMergeAnimator");
                    if (mergeAnimatorType != null) break;
                }
            }

            if (mergeAnimatorType != null)
            {
                var mergeComp = motionObj.GetComponent(mergeAnimatorType) ?? motionObj.AddComponent(mergeAnimatorType);
                var animatorField = mergeAnimatorType.GetField("animator");
                animatorField?.SetValue(mergeComp, controller);

                var layerTypeField = mergeAnimatorType.GetField("layerType");
                if (layerTypeField != null)
                {
                    int layerIndex = config.TargetLayer == VrcTargetLayer.ActionLayer ? 3 : 4;
                    layerTypeField.SetValue(mergeComp, Enum.ToObject(layerTypeField.FieldType, layerIndex));
                }

                var deleteAttachedField = mergeAnimatorType.GetField("deleteAttachedAnimator");
                deleteAttachedField?.SetValue(mergeComp, true);
            }

            // 2. ModularAvatarMenuItem
            Type menuItemType = Type.GetType("nadena.dev.modular_avatar.core.ModularAvatarMenuItem, nadena.dev.modular-avatar.core");
            if (menuItemType == null)
            {
                foreach (var asm in AppDomain.CurrentDomain.GetAssemblies())
                {
                    menuItemType = asm.GetType("nadena.dev.modular_avatar.core.ModularAvatarMenuItem");
                    if (menuItemType != null) break;
                }
            }

            if (menuItemType != null)
            {
                var menuComp = motionObj.GetComponent(menuItemType) ?? motionObj.AddComponent(menuItemType);
                var controlField = menuItemType.GetField("Control");
                if (controlField != null)
                {
                    object controlObj = controlField.GetValue(menuComp);
                    if (controlObj != null)
                    {
                        // All VRCExpressionsMenu.Control members are FIELDS (not properties).
                        controlObj.GetType().GetField("name")?.SetValue(controlObj, config.MotionName);

                        var typeField = controlObj.GetType().GetField("type");
                        if (typeField != null)
                        {
                            bool isOneShot = config.MotionType == VrcMotionType.OneShotEmote;
                            object controlType = isOneShot
                                ? VrcLayerBehaviours.ResolveEnumMember(typeField.FieldType, "Button", 101)
                                : VrcLayerBehaviours.ResolveEnumMember(typeField.FieldType, "Toggle", 102);
                            typeField.SetValue(controlObj, controlType);
                        }

                        var paramField = controlObj.GetType().GetField("parameter");
                        if (paramField != null)
                        {
                            object paramObj = paramField.GetValue(controlObj);
                            if (paramObj != null)
                            {
                                var paramNameField = paramObj.GetType().GetField("name");
                                paramNameField?.SetValue(paramObj, paramName);
                            }
                        }

                        controlField.SetValue(menuComp, controlObj);
                    }
                }
            }

            // 3. ModularAvatarParameters
            Type paramsType = Type.GetType("nadena.dev.modular_avatar.core.ModularAvatarParameters, nadena.dev.modular-avatar.core");
            if (paramsType == null)
            {
                foreach (var asm in AppDomain.CurrentDomain.GetAssemblies())
                {
                    paramsType = asm.GetType("nadena.dev.modular_avatar.core.ModularAvatarParameters");
                    if (paramsType != null) break;
                }
            }

            if (paramsType != null)
            {
                if (motionObj.GetComponent(paramsType) == null)
                {
                    motionObj.AddComponent(paramsType);
                }
            }
        }

        private static string SanitizeFileName(string name)
        {
            foreach (char c in Path.GetInvalidFileNameChars())
            {
                name = name.Replace(c, '_');
            }
            return name.Replace(" ", "_");
        }
    }
}
