using System;
using System.Collections.Generic;
using UnityEditor;
using UnityEngine;

namespace TexMotion.Editor.VRChat
{
    internal sealed class VrcMenuControlData
    {
        public string Name;
        public Texture Icon;
        public int Type;
        public float Value = 1f;
        public int Style;
        public string ParameterName = "";
        public ScriptableObject SubMenu;
        public List<KeyValuePair<string, float>> SubParameters = new List<KeyValuePair<string, float>>();
        public List<VrcMenuLabelData> Labels = new List<VrcMenuLabelData>();
    }

    internal sealed class VrcMenuLabelData
    {
        public string Name;
        public Texture Icon;
    }

    /// <summary>
    /// Shared helpers for manipulating VRCExpressionsMenu controls through SerializedProperty,
    /// including full-fidelity capture/restore used by the NextPage pagination system.
    /// </summary>
    internal static class VrcMenuUtility
    {
        public static int MaxControls
        {
            get
            {
                Type menuType = VrcLayerBehaviours.FindVrcType("VRC.SDK3.Avatars.ScriptableObjects.VRCExpressionsMenu");
                var field = menuType?.GetField("MAX_CONTROLS");
                if (field != null)
                {
                    try { return Convert.ToInt32(field.GetRawConstantValue()); }
                    catch { /* fall through */ }
                }
                return 8;
            }
        }

        public static int ResolveControlType(string memberName, int fallback)
        {
            return VrcLayerBehaviours.TryGetMenuControlTypeValue(memberName, out int value) ? value : fallback;
        }

        /// <summary>True when the control is a TexMotion-managed page link ("NextPage" or legacy "TexMotion" submenu).</summary>
        public static bool TryGetTexMotionPage(SerializedProperty controlElem, out ScriptableObject page)
        {
            page = null;
            var nameProp = controlElem.FindPropertyRelative("name");
            var typeProp = controlElem.FindPropertyRelative("type");
            var subProp = controlElem.FindPropertyRelative("subMenu");
            if (nameProp == null || typeProp == null || subProp == null) return false;

            string n = nameProp.stringValue;
            bool isPageName = n == "NextPage" || (n != null && n.StartsWith("TexMotion", StringComparison.Ordinal));
            if (!isPageName) return false;

            if (typeProp.intValue != ResolveControlType("SubMenu", 103)) return false;

            page = subProp.objectReferenceValue as ScriptableObject;
            return page != null;
        }

        /// <summary>Captures every serialized field of a Control element so it can be moved between menus.</summary>
        public static VrcMenuControlData CaptureControl(SerializedProperty elem)
        {
            var data = new VrcMenuControlData();

            var p = elem.FindPropertyRelative("name");
            if (p != null) data.Name = p.stringValue;

            p = elem.FindPropertyRelative("icon");
            if (p != null) data.Icon = p.objectReferenceValue as Texture;

            p = elem.FindPropertyRelative("type");
            if (p != null) data.Type = p.intValue;

            p = elem.FindPropertyRelative("value");
            if (p != null) data.Value = p.floatValue;

            p = elem.FindPropertyRelative("style");
            if (p != null) data.Style = p.intValue;

            p = elem.FindPropertyRelative("parameter")?.FindPropertyRelative("name");
            if (p != null) data.ParameterName = p.stringValue ?? "";

            p = elem.FindPropertyRelative("subMenu");
            if (p != null) data.SubMenu = p.objectReferenceValue as ScriptableObject;

            var subParams = elem.FindPropertyRelative("subParameters");
            if (subParams != null && subParams.isArray)
            {
                for (int i = 0; i < subParams.arraySize; i++)
                {
                    var sp = subParams.GetArrayElementAtIndex(i);
                    string spName = sp.FindPropertyRelative("name")?.stringValue;
                    float spValue = sp.FindPropertyRelative("value")?.floatValue ?? 1f;
                    data.SubParameters.Add(new KeyValuePair<string, float>(spName, spValue));
                }
            }

            var labels = elem.FindPropertyRelative("labels");
            if (labels != null && labels.isArray)
            {
                for (int i = 0; i < labels.arraySize; i++)
                {
                    var lp = labels.GetArrayElementAtIndex(i);
                    data.Labels.Add(new VrcMenuLabelData
                    {
                        Name = lp.FindPropertyRelative("name")?.stringValue,
                        Icon = lp.FindPropertyRelative("icon")?.objectReferenceValue as Texture
                    });
                }
            }

            return data;
        }

        /// <summary>Writes captured control data into a Control element.</summary>
        public static void ApplyControlData(SerializedProperty elem, VrcMenuControlData data)
        {
            if (data == null)
            {
                ClearControl(elem);
                return;
            }

            var p = elem.FindPropertyRelative("name");
            if (p != null) p.stringValue = data.Name;

            p = elem.FindPropertyRelative("icon");
            if (p != null) p.objectReferenceValue = data.Icon;

            p = elem.FindPropertyRelative("type");
            if (p != null) p.intValue = data.Type;

            p = elem.FindPropertyRelative("value");
            if (p != null) p.floatValue = data.Value;

            p = elem.FindPropertyRelative("style");
            if (p != null) p.intValue = data.Style;

            var paramObj = elem.FindPropertyRelative("parameter");
            var paramNameProp = paramObj?.FindPropertyRelative("name");
            if (paramNameProp != null) paramNameProp.stringValue = data.ParameterName ?? "";

            p = elem.FindPropertyRelative("subMenu");
            if (p != null) p.objectReferenceValue = data.SubMenu;

            var subParams = elem.FindPropertyRelative("subParameters");
            if (subParams != null && subParams.isArray)
            {
                subParams.arraySize = data.SubParameters.Count;
                for (int i = 0; i < data.SubParameters.Count; i++)
                {
                    var sp = subParams.GetArrayElementAtIndex(i);
                    var spName = sp.FindPropertyRelative("name");
                    if (spName != null) spName.stringValue = data.SubParameters[i].Key;
                    var spValue = sp.FindPropertyRelative("value");
                    if (spValue != null) spValue.floatValue = data.SubParameters[i].Value;
                }
            }

            var labels = elem.FindPropertyRelative("labels");
            if (labels != null && labels.isArray)
            {
                labels.arraySize = data.Labels.Count;
                for (int i = 0; i < data.Labels.Count; i++)
                {
                    var lp = labels.GetArrayElementAtIndex(i);
                    var lpName = lp.FindPropertyRelative("name");
                    if (lpName != null) lpName.stringValue = data.Labels[i].Name;
                    var lpIcon = lp.FindPropertyRelative("icon");
                    if (lpIcon != null) lpIcon.objectReferenceValue = data.Labels[i].Icon;
                }
            }
        }

        public static VrcMenuControlData CreateNextPageControl(ScriptableObject pageAsset)
        {
            return new VrcMenuControlData
            {
                Name = "NextPage",
                Type = ResolveControlType("SubMenu", 103),
                Value = 1f,
                ParameterName = "",
                SubMenu = pageAsset
            };
        }

        private static void ClearControl(SerializedProperty elem)
        {
            ApplyControlData(elem, new VrcMenuControlData { Name = "", Type = ResolveControlType("Button", 101), SubMenu = null });
        }
    }
}
