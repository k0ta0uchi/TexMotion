using System;
using System.Collections.Generic;
using System.IO;
using TexMotion.Runtime.VRChat;
using UnityEditor;
using UnityEditor.Animations;
using UnityEngine;

namespace TexMotion.Editor.VRChat
{
    /// <summary>
    /// Handles robust direct setup of motions into VRCAvatarDescriptor, VRCExpressionsMenu, and VRCExpressionParameters
    /// using SerializedObject for guaranteed Prefab & Scene persistence and proper VRChat Action Layer behaviours.
    /// </summary>
    public static class VrcDirectSetup
    {
        public static bool SetupDirectAvatarMotion(
            GameObject targetAvatar,
            AnimationClip motionClip,
            VrcMotionConfig config,
            ScriptableObject customTargetMenu = null,
            string saveDirectory = "Assets/TexMotion/Generated")
        {
            if (targetAvatar == null) throw new ArgumentNullException(nameof(targetAvatar));
            if (motionClip == null) throw new ArgumentNullException(nameof(motionClip));

            if (!Directory.Exists(saveDirectory))
            {
                Directory.CreateDirectory(saveDirectory);
                AssetDatabase.Refresh();
            }

            // 1. Find VRCAvatarDescriptor
            Component descriptor = FindAvatarDescriptor(targetAvatar);
            if (descriptor == null)
            {
                throw new Exception("VRCAvatarDescriptor component was not found on the target avatar.");
            }

            string sanitizedName = SanitizeFileName(config.MotionName);
            string paramName = $"TexMotion_{sanitizedName}";

            // 2. Add Parameter to VRCExpressionParameters
            ScriptableObject exprParams = GetExpressionParameters(descriptor);
            if (exprParams != null)
            {
                AddParameterToExpressionParameters(exprParams, paramName);
            }

            // 3. Determine Target Menu (Custom / Emote SubMenu / Root Menu)
            ScriptableObject targetMenu = customTargetMenu;
            if (targetMenu == null)
            {
                targetMenu = GetExpressionsMenu(descriptor);
            }

            if (targetMenu != null)
            {
                if (!AddControlToExpressionsMenu(targetMenu, config, paramName, saveDirectory))
                {
                    Debug.LogError(TexMotionLocalization.TrFormat(
                        TexMotionLocalization.MenuMotionNotAdded,
                        config.MotionName,
                        targetMenu.name));
                }
            }

            // 4. Ensure & Setup AnimatorController in VRCAvatarDescriptor (Action or FX Layer) using SerializedObject
            AnimatorController animController = EnsureAndAssignLayerControllerSerialized(descriptor, config.TargetLayer, saveDirectory);
            if (animController != null)
            {
                AddStateToAnimatorController(animController, motionClip, config, paramName);
            }

            EditorUtility.SetDirty(targetAvatar);
            if (exprParams != null) EditorUtility.SetDirty(exprParams);
            if (targetMenu != null) EditorUtility.SetDirty(targetMenu);
            AssetDatabase.SaveAssets();

            return true;
        }

        /// <summary>
        /// Directly configures synchronized dual-layer playback in VRCAvatarDescriptor:
        /// - Body AnimationClip in Action Layer
        /// - Face AnimationClip in FX Layer
        /// Both driven concurrently by the same Expression Parameter and Menu item.
        /// </summary>
        public static bool SetupDirectAvatarMotionWithFace(
            GameObject targetAvatar,
            AnimationClip bodyClip,
            AnimationClip faceClip,
            VrcMotionConfig config,
            ScriptableObject customTargetMenu = null,
            string saveDirectory = "Assets/TexMotion/Generated")
        {
            if (targetAvatar == null) throw new ArgumentNullException(nameof(targetAvatar));
            if (bodyClip == null) throw new ArgumentNullException(nameof(bodyClip));

            if (faceClip == null)
            {
                return SetupDirectAvatarMotion(targetAvatar, bodyClip, config, customTargetMenu, saveDirectory);
            }

            if (!Directory.Exists(saveDirectory))
            {
                Directory.CreateDirectory(saveDirectory);
                AssetDatabase.Refresh();
            }

            // 1. Find VRCAvatarDescriptor
            Component descriptor = FindAvatarDescriptor(targetAvatar);
            if (descriptor == null)
            {
                throw new Exception("VRCAvatarDescriptor component was not found on the target avatar.");
            }

            string sanitizedName = SanitizeFileName(config.MotionName);
            string paramName = $"TexMotion_{sanitizedName}";

            // 2. Add Parameter to VRCExpressionParameters
            ScriptableObject exprParams = GetExpressionParameters(descriptor);
            if (exprParams != null)
            {
                AddParameterToExpressionParameters(exprParams, paramName);
            }

            // 3. Determine Target Menu & Add Menu Item
            ScriptableObject targetMenu = customTargetMenu ?? GetExpressionsMenu(descriptor);
            if (targetMenu != null)
            {
                if (!AddControlToExpressionsMenu(targetMenu, config, paramName, saveDirectory))
                {
                    Debug.LogError(TexMotionLocalization.TrFormat(
                        TexMotionLocalization.MenuMotionNotAdded,
                        config.MotionName,
                        targetMenu.name));
                }
            }

            // 4. Setup Action Layer Controller (Body)
            AnimatorController actionController = EnsureAndAssignLayerControllerSerialized(descriptor, VrcTargetLayer.ActionLayer, saveDirectory);
            if (actionController != null)
            {
                var actionConfig = new VrcMotionConfig
                {
                    MotionName = config.MotionName,
                    MotionType = config.MotionType,
                    TargetLayer = VrcTargetLayer.ActionLayer
                };
                AddStateToAnimatorController(actionController, bodyClip, actionConfig, paramName);
            }

            // 5. Setup FX Layer Controller (Face)
            AnimatorController fxController = EnsureAndAssignLayerControllerSerialized(descriptor, VrcTargetLayer.FXLayer, saveDirectory);
            if (fxController != null)
            {
                var fxConfig = new VrcMotionConfig
                {
                    MotionName = $"{config.MotionName}_Face",
                    MotionType = config.MotionType,
                    TargetLayer = VrcTargetLayer.FXLayer
                };
                AddStateToFxAnimatorController(fxController, faceClip, fxConfig, paramName);
            }

            EditorUtility.SetDirty(targetAvatar);
            if (exprParams != null) EditorUtility.SetDirty(exprParams);
            if (targetMenu != null) EditorUtility.SetDirty(targetMenu);
            AssetDatabase.SaveAssets();

            return true;
        }

        private static void AddStateToFxAnimatorController(
            AnimatorController controller,
            AnimationClip clip,
            VrcMotionConfig config,
            string paramName)
        {
            // 1. Ensure parameter exists
            bool hasParam = false;
            foreach (var p in controller.parameters)
            {
                if (p.name == paramName) { hasParam = true; break; }
            }
            if (!hasParam)
            {
                controller.AddParameter(paramName, AnimatorControllerParameterType.Bool);
            }

            // 2. Ensure FX layer exists
            if (controller.layers.Length == 0)
            {
                controller.AddLayer("Base Layer");
            }

            var layers = controller.layers;
            var fxLayer = layers[0];
            fxLayer.defaultWeight = 1.0f;
            controller.layers = layers;

            var sm = fxLayer.stateMachine;

            // Ensure Idle state
            AnimatorState idleState = sm.defaultState;
            if (idleState == null)
            {
                idleState = sm.AddState("Idle", new Vector3(250, 0, 0));
                idleState.motion = null;
                sm.defaultState = idleState;
            }

            // Add Face Motion State
            string stateName = config.MotionName;
            var motionState = sm.AddState(stateName, new Vector3(250, 100, 0));
            motionState.motion = clip;

            if (config.MotionType == VrcMotionType.OneShotEmote)
            {
                var toMotion = idleState.AddTransition(motionState);
                toMotion.AddCondition(AnimatorConditionMode.If, 0, paramName);
                toMotion.hasExitTime = false;
                toMotion.duration = 0.1f;

                var toIdle = motionState.AddTransition(idleState);
                toIdle.hasExitTime = true;
                toIdle.exitTime = 0.95f;
                toIdle.duration = 0.2f;
            }
            else // ToggleLoopPose
            {
                var toMotion = idleState.AddTransition(motionState);
                toMotion.AddCondition(AnimatorConditionMode.If, 0, paramName);
                toMotion.hasExitTime = false;
                toMotion.duration = 0.15f;

                var toIdle = motionState.AddTransition(idleState);
                toIdle.AddCondition(AnimatorConditionMode.IfNot, 0, paramName);
                toIdle.hasExitTime = false;
                toIdle.duration = 0.15f;
            }

            EditorUtility.SetDirty(controller);
        }

        public static List<ScriptableObject> FindAllAvatarMenus(GameObject avatar)
        {
            var menus = new List<ScriptableObject>();
            if (avatar == null) return menus;

            Component descriptor = FindAvatarDescriptor(avatar);
            if (descriptor == null) return menus;

            ScriptableObject rootMenu = GetExpressionsMenu(descriptor);
            if (rootMenu != null)
            {
                menus.Add(rootMenu);
                CollectSubMenusRecursive(rootMenu, menus);
            }

            return menus;
        }

        private static void CollectSubMenusRecursive(ScriptableObject menu, List<ScriptableObject> list)
        {
            if (menu == null) return;
            var so = new SerializedObject(menu);
            var controlsProp = so.FindProperty("controls");
            if (controlsProp == null || !controlsProp.isArray) return;

            for (int i = 0; i < controlsProp.arraySize; i++)
            {
                var elem = controlsProp.GetArrayElementAtIndex(i);
                var subMenuProp = elem.FindPropertyRelative("subMenu");
                if (subMenuProp != null && subMenuProp.objectReferenceValue is ScriptableObject subMenu)
                {
                    if (!list.Contains(subMenu))
                    {
                        list.Add(subMenu);
                        CollectSubMenusRecursive(subMenu, list);
                    }
                }
            }
        }

        public static Component FindAvatarDescriptor(GameObject avatar)
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

        public static ScriptableObject GetExpressionsMenu(Component descriptor)
        {
            var so = new SerializedObject(descriptor);
            var prop = so.FindProperty("expressionsMenu");
            return prop?.objectReferenceValue as ScriptableObject;
        }

        public static ScriptableObject GetExpressionParameters(Component descriptor)
        {
            var so = new SerializedObject(descriptor);
            var prop = so.FindProperty("expressionParameters");
            return prop?.objectReferenceValue as ScriptableObject;
        }

        private static AnimatorController EnsureAndAssignLayerControllerSerialized(Component descriptor, VrcTargetLayer targetLayer, string saveDirectory)
        {
            var descriptorSO = new SerializedObject(descriptor);
            descriptorSO.Update();

            var baseLayersProp = descriptorSO.FindProperty("baseAnimationLayers");
            if (baseLayersProp == null || !baseLayersProp.isArray)
            {
                Debug.LogError(TexMotionLocalization.Tr(TexMotionLocalization.BaseAnimationLayersMissing));
                return null;
            }

            // Target layer index in baseAnimationLayers: Action = 3, FX = 4
            int targetIdx = targetLayer == VrcTargetLayer.ActionLayer ? 3 : 4;
            if (baseLayersProp.arraySize <= targetIdx)
            {
                Debug.LogError(TexMotionLocalization.TrFormat(
                    TexMotionLocalization.BaseAnimationLayerIndexMissing,
                    targetIdx));
                return null;
            }

            var layerElem = baseLayersProp.GetArrayElementAtIndex(targetIdx);
            var animCtrlProp = layerElem.FindPropertyRelative("animatorController");
            var isDefaultProp = layerElem.FindPropertyRelative("isDefault");

            RuntimeAnimatorController currentCtrl = animCtrlProp.objectReferenceValue as RuntimeAnimatorController;
            if (currentCtrl is AnimatorController existingController)
            {
                // Ensure isDefault is false
                isDefaultProp.boolValue = false;
                descriptorSO.ApplyModifiedProperties();
                PrefabUtility.RecordPrefabInstancePropertyModifications(descriptor);
                return existingController;
            }

            // Create new controller
            string layerName = targetLayer.ToString();
            string newControllerPath = $"{saveDirectory}/Ctrl_Avatar_{layerName}.controller";

            AnimatorController newController;
            if (File.Exists(newControllerPath))
            {
                newController = AssetDatabase.LoadAssetAtPath<AnimatorController>(newControllerPath);
            }
            else
            {
                newController = AnimatorController.CreateAnimatorControllerAtPath(newControllerPath);
            }

            // Critical: Set isDefault = false and assign controller via SerializedProperty
            isDefaultProp.boolValue = false;
            animCtrlProp.objectReferenceValue = newController;

            var customizeProp = descriptorSO.FindProperty("customizeAnimationLayers");
            if (customizeProp != null && customizeProp.isArray && customizeProp.arraySize > targetIdx)
            {
                customizeProp.GetArrayElementAtIndex(targetIdx).boolValue = true;
            }

            descriptorSO.ApplyModifiedProperties();
            PrefabUtility.RecordPrefabInstancePropertyModifications(descriptor);

            EditorUtility.SetDirty(descriptor);
            return newController;
        }

        private static void AddParameterToExpressionParameters(ScriptableObject exprParams, string paramName)
        {
            var so = new SerializedObject(exprParams);
            so.Update();

            var paramsProp = so.FindProperty("parameters");
            if (paramsProp == null || !paramsProp.isArray) return;

            // Check if already exists
            for (int i = 0; i < paramsProp.arraySize; i++)
            {
                var elem = paramsProp.GetArrayElementAtIndex(i);
                var nameProp = elem.FindPropertyRelative("name");
                if (nameProp != null && nameProp.stringValue == paramName)
                {
                    return; // Already present
                }
            }

            // Add new parameter
            int newIdx = paramsProp.arraySize;
            paramsProp.InsertArrayElementAtIndex(newIdx);
            var newElem = paramsProp.GetArrayElementAtIndex(newIdx);

            newElem.FindPropertyRelative("name").stringValue = paramName;
            newElem.FindPropertyRelative("valueType").enumValueIndex = 2; // Bool
            newElem.FindPropertyRelative("saved").boolValue = false;
            newElem.FindPropertyRelative("defaultValue").floatValue = 0f;

            var networkProp = newElem.FindPropertyRelative("networkSynced");
            if (networkProp != null) networkProp.boolValue = true;

            so.ApplyModifiedProperties();
            EditorUtility.SetDirty(exprParams);
        }

        private static bool AddControlToExpressionsMenu(
            ScriptableObject exprMenu,
            VrcMotionConfig config,
            string paramName,
            string saveDirectory,
            int depth = 0)
        {
            var so = new SerializedObject(exprMenu);
            so.Update();

            var controlsProp = so.FindProperty("controls");
            if (controlsProp == null || !controlsProp.isArray) return false;

            // 1. Update existing control / claim a garbage "Name" placeholder
            for (int i = 0; i < controlsProp.arraySize; i++)
            {
                var elem = controlsProp.GetArrayElementAtIndex(i);
                var nameProp = elem.FindPropertyRelative("name");
                var paramProp = elem.FindPropertyRelative("parameter")?.FindPropertyRelative("name");

                if (nameProp != null && (nameProp.stringValue == config.MotionName || (paramProp != null && paramProp.stringValue == paramName)))
                {
                    ApplyControlValues(elem, config.MotionName, GetMenuControlTypeValue(config.MotionType), paramName, null);
                    so.ApplyModifiedProperties();
                    EditorUtility.SetDirty(exprMenu);
                    return true;
                }

                if (nameProp != null && nameProp.stringValue == "Name")
                {
                    ApplyControlValues(elem, config.MotionName, GetMenuControlTypeValue(config.MotionType), paramName, null);
                    so.ApplyModifiedProperties();
                    EditorUtility.SetDirty(exprMenu);
                    return true;
                }
            }

            // 2. Free slot available -> append directly
            if (controlsProp.arraySize < VrcMenuUtility.MaxControls)
            {
                int newIdx = controlsProp.arraySize;
                controlsProp.InsertArrayElementAtIndex(newIdx);
                ApplyControlValues(controlsProp.GetArrayElementAtIndex(newIdx), config.MotionName, GetMenuControlTypeValue(config.MotionType), paramName, null);

                so.ApplyModifiedProperties();
                EditorUtility.SetDirty(exprMenu);
                return true;
            }

            // 3. Menu is full -> delegate into an already-linked NextPage / TexMotion page
            for (int i = 0; i < controlsProp.arraySize; i++)
            {
                if (VrcMenuUtility.TryGetTexMotionPage(controlsProp.GetArrayElementAtIndex(i), out ScriptableObject linkedPage))
                {
                    return AddControlToExpressionsMenu(linkedPage, config, paramName, saveDirectory, depth + 1);
                }
            }

            // 4. Menu is full and no page exists yet -> paginate:
            //    evict the LAST control into a fresh "NextPage" submenu, then keep adding there.
            //    The evicted emote is preserved as the first entry of the new page.
            int lastIndex = controlsProp.arraySize - 1;
            var evicted = VrcMenuUtility.CaptureControl(controlsProp.GetArrayElementAtIndex(lastIndex));

            ScriptableObject pageAsset = CreateTexMotionNextPageAsset(saveDirectory, exprMenu.name, exprMenu.GetType());
            VrcMenuUtility.ApplyControlData(controlsProp.GetArrayElementAtIndex(lastIndex), VrcMenuUtility.CreateNextPageControl(pageAsset));
            so.ApplyModifiedProperties();
            EditorUtility.SetDirty(exprMenu);

            var pageSO = new SerializedObject(pageAsset);
            var pageControls = pageSO.FindProperty("controls");
            if (pageControls != null && pageControls.isArray && evicted != null && !string.IsNullOrEmpty(evicted.Name))
            {
                pageControls.arraySize = 1;
                VrcMenuUtility.ApplyControlData(pageControls.GetArrayElementAtIndex(0), evicted);
                pageSO.ApplyModifiedProperties();
                EditorUtility.SetDirty(pageAsset);
            }

            return AddControlToExpressionsMenu(pageAsset, config, paramName, saveDirectory, depth + 1);
        }

        private static void ApplyControlValues(SerializedProperty elem, string displayName, int typeValue, string parameterName, ScriptableObject subMenu)
        {
            VrcMenuUtility.ApplyControlData(elem, new VrcMenuControlData
            {
                Name = displayName,
                Type = typeValue,
                Value = 1f,
                ParameterName = parameterName ?? "",
                SubMenu = subMenu
            });
        }

        private static ScriptableObject CreateTexMotionNextPageAsset(string saveDirectory, string parentMenuName, Type menuType)
        {
            string baseName = $"Menu_TexMotion_NextPage_{SanitizeFileName(parentMenuName)}";
            string path = AssetDatabase.GenerateUniqueAssetPath($"{saveDirectory}/{baseName}.asset");

            var created = ScriptableObject.CreateInstance(menuType);
            AssetDatabase.CreateAsset(created, path);
            return created;
        }

        private static void AddStateToAnimatorController(
            AnimatorController controller,
            AnimationClip clip,
            VrcMotionConfig config,
            string paramName)
        {
            // 1. Ensure parameter exists in controller
            bool hasParam = false;
            foreach (var p in controller.parameters)
            {
                if (p.name == paramName) { hasParam = true; break; }
            }
            if (!hasParam)
            {
                controller.AddParameter(paramName, AnimatorControllerParameterType.Bool);
            }

            // 2. Use Layer 0 (Base Layer) directly with defaultWeight = 1.0f
            if (controller.layers.Length == 0)
            {
                controller.AddLayer("Base Layer");
            }

            var layers = controller.layers;
            var baseLayer = layers[0];
            baseLayer.name = "TexMotion";
            baseLayer.defaultWeight = 1.0f;
            controller.layers = layers;

            var sm = baseLayer.stateMachine;

            // Ensure Idle state
            AnimatorState idleState = sm.defaultState;
            if (idleState == null)
            {
                idleState = sm.AddState("Idle", new Vector3(250, 0, 0));
                idleState.motion = null;
                sm.defaultState = idleState;
            }

            // Attach VRChat Action Layer behaviours to Idle state (LayerWeight: 0, Tracking: Tracking, ParamDriver: Reset)
            ConfigureIdleStateBehaviours(idleState, paramName, config.MotionType);

            // Check if state already exists, else create new
            AnimatorState motionState = null;
            foreach (var s in sm.states)
            {
                if (s.state.name == config.MotionName)
                {
                    motionState = s.state;
                    break;
                }
            }

            if (motionState == null)
            {
                motionState = sm.AddState(config.MotionName, new Vector3(250, 100 + (sm.states.Length * 35), 0));
            }

            motionState.motion = clip;

            // Attach VRChat Action Layer behaviours to Motion state (LayerWeight: 1, Tracking: Animation, ParamDriver: Reset)
            ConfigureMotionStateBehaviours(motionState, paramName, config.MotionType);

            // Rebuild transitions cleanly
            motionState.transitions = new AnimatorStateTransition[0];
            idleState.transitions = new AnimatorStateTransition[0];

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
            else // Toggle
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

            EditorUtility.SetDirty(controller);
        }

        private static void ConfigureIdleStateBehaviours(AnimatorState idleState, string paramName, VrcMotionType motionType)
        {
            // Action layer weight back to 0 so the avatar returns to normal tracking.
            VrcLayerBehaviours.ApplyPlayableLayerControl(idleState, 0.0f, 0.2f);
            VrcLayerBehaviours.ApplyTrackingControl(idleState, false);

            // Defense-in-depth for one-shots: guarantee the emote parameter is false
            // whenever Idle is entered (covers interrupted transitions and stale states).
            if (motionType == VrcMotionType.OneShotEmote)
            {
                VrcLayerBehaviours.ApplyParameterResetDriver(idleState, paramName);
            }
        }

        private static void ConfigureMotionStateBehaviours(AnimatorState motionState, string paramName, VrcMotionType motionType)
        {
            // Raise the Action layer weight to 1 while the motion plays.
            VrcLayerBehaviours.ApplyPlayableLayerControl(motionState, 1.0f, 0.1f);
            // Switch all tracked body parts to Animation while the motion plays.
            VrcLayerBehaviours.ApplyTrackingControl(motionState, true);

            // Immediately reset the one-shot parameter on state entry so that a latched
            // menu control can never re-trigger Idle -> Motion after the emote finishes.
            if (motionType == VrcMotionType.OneShotEmote)
            {
                VrcLayerBehaviours.ApplyParameterResetDriver(motionState, paramName);
            }
        }

        private static int ResolveMenuControlType(string memberName, int fallback)
        {
            return VrcLayerBehaviours.TryGetMenuControlTypeValue(memberName, out int value) ? value : fallback;
        }

        /// <summary>
        /// One-shot emotes must use a BUTTON control (fires once); loop poses use a TOGGLE.
        /// Values are resolved by member name because ControlType does not start at 0
        /// (Button = 101, Toggle = 102 in the current SDK).
        /// </summary>
        private static int GetMenuControlTypeValue(VrcMotionType motionType)
        {
            bool isOneShot = motionType == VrcMotionType.OneShotEmote;
            return VrcLayerBehaviours.TryGetMenuControlTypeValue(isOneShot ? "Button" : "Toggle", out int value)
                ? value
                : (isOneShot ? 101 : 102);
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
