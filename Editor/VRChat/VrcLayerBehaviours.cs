using System;
using UnityEditor;
using UnityEditor.Animations;
using UnityEngine;

namespace TexMotion.Editor.VRChat
{
    /// <summary>
    /// Configures VRChat StateMachineBehaviours on AnimatorStates in a version-proof way.
    /// Enum members are resolved BY NAME (never by hardcoded integer) because VRC SDK enum
    /// values do not follow playable-layer indices (e.g. BlendableLayer.Action == 0),
    /// and VRCAvatarParameterDriver.Parameter lives on the SDK BASE class
    /// (VRC.SDKBase.VRC_AvatarParameterDriver), not on the derived type.
    /// </summary>
    internal static class VrcLayerBehaviours
    {
        public static Type FindVrcType(string fullName)
        {
            Type t = Type.GetType(fullName);
            if (t != null) return t;
            foreach (var asm in AppDomain.CurrentDomain.GetAssemblies())
            {
                t = asm.GetType(fullName);
                if (t != null) return t;
            }
            return null;
        }

        /// <summary>Resolves an enum member's boxed value by member name, falling back to a raw value.</summary>
        public static object ResolveEnumMember(Type enumType, string memberName, int fallbackValue)
        {
            if (enumType == null) return fallbackValue;
            foreach (var name in Enum.GetNames(enumType))
            {
                if (string.Equals(name, memberName, StringComparison.OrdinalIgnoreCase))
                {
                    return Enum.Parse(enumType, name);
                }
            }
            Debug.LogWarning(TexMotionLocalization.TrFormat(
                TexMotionLocalization.EnumMemberMissing,
                memberName,
                enumType.Name,
                fallbackValue));
            return Enum.ToObject(enumType, fallbackValue);
        }

        /// <summary>
        /// Resolves the serialized int VALUE of VRCExpressionsMenu.Control.ControlType by member name.
        /// (Button = 101, Toggle = 102 in current SDK; never hardcode.)
        /// </summary>
        public static bool TryGetMenuControlTypeValue(string memberName, out int value)
        {
            value = 0;
            Type controlType = FindVrcType("VRC.SDK3.Avatars.ScriptableObjects.VRCExpressionsMenu+Control+ControlType");
            if (controlType == null) return false;

            foreach (var name in Enum.GetNames(controlType))
            {
                if (string.Equals(name, memberName, StringComparison.OrdinalIgnoreCase))
                {
                    value = Convert.ToInt32(Enum.Parse(controlType, name));
                    return true;
                }
            }
            Debug.LogWarning(TexMotionLocalization.TrFormat(
                TexMotionLocalization.ControlTypeMissing,
                memberName));
            return false;
        }

        public static void ApplyPlayableLayerControl(AnimatorState state, float goalWeight, float blendDuration)
        {
            Type t = FindVrcType("VRC.SDK3.Avatars.Components.VRCPlayableLayerControl");
            if (t == null)
            {
                Debug.LogWarning(TexMotionLocalization.Tr(TexMotionLocalization.PlayableLayerControlMissing));
                return;
            }

            var behaviour = GetOrAddBehaviour(state, t);
            if (behaviour == null)
            {
                Debug.LogWarning($"[TexMotion] Could not add StateMachineBehaviour {t.Name} to state {state?.name}");
                return;
            }

            var layerField = t.GetField("layer");
            if (layerField != null)
            {
                // NOTE: BlendableLayer.Action is NOT the playable layer index; resolve by name.
                layerField.SetValue(behaviour, ResolveEnumMember(layerField.FieldType, "Action", 0));
            }
            t.GetField("goalWeight")?.SetValue(behaviour, goalWeight);
            t.GetField("blendDuration")?.SetValue(behaviour, blendDuration);
            EditorUtility.SetDirty(behaviour);
        }

        public static void ApplyTrackingControl(AnimatorState state, bool useAnimation)
        {
            Type t = FindVrcType("VRC.SDK3.Avatars.Components.VRCAnimatorTrackingControl");
            if (t == null)
            {
                Debug.LogWarning(TexMotionLocalization.Tr(TexMotionLocalization.TrackingControlMissing));
                return;
            }

            var behaviour = GetOrAddBehaviour(state, t);
            if (behaviour == null)
            {
                Debug.LogWarning($"[TexMotion] Could not add StateMachineBehaviour {t.Name} to state {state?.name}");
                return;
            }

            string memberName = useAnimation ? "Animation" : "Tracking";
            var headField = t.GetField("trackingHead");
            if (headField != null)
            {
                object val = ResolveEnumMember(headField.FieldType, memberName, useAnimation ? 2 : 1);
                foreach (var f in t.GetFields())
                {
                    if (f.FieldType == headField.FieldType && f.Name.StartsWith("tracking", StringComparison.Ordinal))
                    {
                        f.SetValue(behaviour, val);
                    }
                }
            }
            EditorUtility.SetDirty(behaviour);
        }

        /// <summary>
        /// Attaches a VRCAvatarParameterDriver that sets paramName to 0 when the state is entered.
        /// Uses SerializedObject because the Parameter nested class lives on the SDK base class.
        /// </summary>
        public static void ApplyParameterResetDriver(AnimatorState state, string paramName)
        {
            Type t = FindVrcType("VRC.SDK3.Avatars.Components.VRCAvatarParameterDriver");
            if (t == null)
            {
                Debug.LogWarning(TexMotionLocalization.Tr(TexMotionLocalization.ParameterDriverMissing));
                return;
            }

            var behaviour = GetOrAddBehaviour(state, t);
            if (behaviour == null)
            {
                Debug.LogWarning($"[TexMotion] Could not add StateMachineBehaviour {t.Name} to state {state?.name}");
                return;
            }

            var so = new SerializedObject(behaviour);
            var listProp = so.FindProperty("parameters");
            if (listProp != null && listProp.isArray)
            {
                listProp.arraySize = 1;
                var elem = listProp.GetArrayElementAtIndex(0);

                var typeProp = elem.FindPropertyRelative("type");
                if (typeProp != null && typeProp.propertyType == SerializedPropertyType.Enum)
                {
                    // ChangeType.Set — resolve index by name to survive reordering.
                    Type changeType = FindVrcType("VRC.SDKBase.VRC_AvatarParameterDriver+ChangeType");
                    int setIndex = changeType != null
                        ? Array.FindIndex(Enum.GetNames(changeType), n => string.Equals(n, "Set", StringComparison.OrdinalIgnoreCase))
                        : 0;
                    typeProp.enumValueIndex = setIndex >= 0 ? setIndex : 0;
                }

                var nameProp = elem.FindPropertyRelative("name");
                if (nameProp != null) nameProp.stringValue = paramName;

                var valueProp = elem.FindPropertyRelative("value");
                if (valueProp != null) valueProp.floatValue = 0f;

                so.ApplyModifiedPropertiesWithoutUndo();
            }
            else
            {
                Debug.LogError(TexMotionLocalization.Tr(TexMotionLocalization.ParameterDriverSerializeFailed));
            }

            EditorUtility.SetDirty(behaviour);
        }

        public static StateMachineBehaviour GetOrAddBehaviour(AnimatorState state, Type behaviourType)
        {
            if (state == null || behaviourType == null) return null;

            if (state.behaviours != null)
            {
                foreach (var b in state.behaviours)
                {
                    if (b != null && b.GetType() == behaviourType)
                    {
                        return b;
                    }
                }
            }

            StateMachineBehaviour smb = null;
            try
            {
                smb = state.AddStateMachineBehaviour(behaviourType);
            }
            catch (Exception ex)
            {
                Debug.LogWarning($"[TexMotion] state.AddStateMachineBehaviour({behaviourType.Name}) threw: {ex.Message}");
            }

            if (smb == null)
            {
                try
                {
                    smb = ScriptableObject.CreateInstance(behaviourType) as StateMachineBehaviour;
                    if (smb != null)
                    {
                        smb.name = behaviourType.Name;
                        if (EditorUtility.IsPersistent(state))
                        {
                            AssetDatabase.AddObjectToAsset(smb, state);
                        }
                        var list = new System.Collections.Generic.List<StateMachineBehaviour>();
                        if (state.behaviours != null) list.AddRange(state.behaviours);
                        list.Add(smb);
                        state.behaviours = list.ToArray();
                        EditorUtility.SetDirty(state);
                    }
                }
                catch (Exception ex)
                {
                    Debug.LogWarning($"[TexMotion] Fallback CreateInstance({behaviourType.Name}) failed: {ex.Message}");
                }
            }

            return smb;
        }
    }
}
