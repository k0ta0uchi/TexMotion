using System;
using System.Collections.Generic;
using System.Reflection;
using UnityEditor;
using UnityEngine;

namespace TexMotion.Editor.VRChat
{
    /// <summary>
    /// Represents an animation curve that conflicts with a runtime PhysBone component.
    /// </summary>
    public class PhysBoneConflict
    {
        public EditorCurveBinding Binding;
        public string Path;
        public string PropertyName;
        public string PhysBoneName;
        public Transform Transform;

        public override string ToString()
        {
            return $"[{PhysBoneName}] {Path} -> {PropertyName}";
        }
    }

    /// <summary>
    /// Represents a boundary discontinuity or velocity pop between first and last frames of a loopable clip.
    /// </summary>
    public class LoopDiscontinuity
    {
        public EditorCurveBinding Binding;
        public string Path;
        public string PropertyName;
        public float StartValue;
        public float EndValue;
        public float DeltaValue;
        public float VelocityDiff;
        public string Severity; // "Warning" | "Error"
        public string Description;

        public override string ToString()
        {
            return $"{Path}.{PropertyName}: Delta={DeltaValue:F4}, VelDiff={VelocityDiff:F4} ({Severity})";
        }
    }

    /// <summary>
    /// Comprehensive diagnostic report detailing PhysBone conflicts and loop boundary discontinuities.
    /// </summary>
    public class PhysBoneDiagnosticReport
    {
        public GameObject TargetAvatar;
        public AnimationClip Clip;
        public List<PhysBoneConflict> Conflicts = new List<PhysBoneConflict>();
        public List<LoopDiscontinuity> Discontinuities = new List<LoopDiscontinuity>();

        public int TotalCurvesScanned;
        public int PhysBoneComponentsFound;
        public int ControlledTransformsCount;

        public bool HasConflicts => Conflicts.Count > 0;
        public bool HasDiscontinuities => Discontinuities.Count > 0;
        public bool HasIssues => HasConflicts || HasDiscontinuities;

        public string SummaryText
        {
            get
            {
                if (!HasIssues)
                {
                    return $"Clean: Scanned {TotalCurvesScanned} curves across {PhysBoneComponentsFound} PhysBone components. No conflicts or loop pops detected.";
                }
                return $"Issues Detected: {Conflicts.Count} PhysBone conflicts, {Discontinuities.Count} loop boundary discontinuities across {TotalCurvesScanned} curves.";
            }
        }
    }

    /// <summary>
    /// Diagnostics and safety utility for detecting VRChat PhysBone conflicts and loop boundary pops in AnimationClips.
    /// Uses safe reflection to inspect VRCPhysBone components without hard compile-time SDK dependencies.
    /// </summary>
    public static class PhysBoneConflictDetector
    {
        private static Type _cachedPhysBoneType;
        private static bool _hasSearchedPhysBoneType;

        /// <summary>
        /// Gets the VRCPhysBone type safely via reflection across all loaded assemblies.
        /// Returns null if VRChat SDK is not installed in the project.
        /// </summary>
        public static Type GetPhysBoneType()
        {
            if (_hasSearchedPhysBoneType) return _cachedPhysBoneType;
            _hasSearchedPhysBoneType = true;

            // Direct type lookup for standard VRCSDK3A assembly
            _cachedPhysBoneType = Type.GetType("VRC.SDK3.Dynamics.PhysBone.Components.VRCPhysBone, VRCSDK3A")
                               ?? Type.GetType("VRC.SDK3.Dynamics.PhysBone.Components.VRCPhysBone, VRC.SDK3.Dynamics.PhysBone");

            if (_cachedPhysBoneType == null)
            {
                // Fallback scan across all loaded assemblies in current domain
                foreach (var assembly in AppDomain.CurrentDomain.GetAssemblies())
                {
                    try
                    {
                        var type = assembly.GetType("VRC.SDK3.Dynamics.PhysBone.Components.VRCPhysBone", false);
                        if (type != null)
                        {
                            _cachedPhysBoneType = type;
                            break;
                        }
                    }
                    catch
                    {
                        // Ignore reflection load errors on dynamically emitted assemblies
                    }
                }
            }

            return _cachedPhysBoneType;
        }

        /// <summary>
        /// Collects all Transforms governed by PhysBone components on the target avatar.
        /// </summary>
        public static Dictionary<Transform, string> CollectPhysBoneTransforms(GameObject avatarRoot)
        {
            var result = new Dictionary<Transform, string>();
            if (avatarRoot == null) return result;

            Type pbType = GetPhysBoneType();
            if (pbType == null)
            {
                // VRChat SDK not present in project; return empty mapping safely
                return result;
            }

            Component[] pbComponents = avatarRoot.GetComponentsInChildren(pbType, true);
            PropertyInfo rootTransformProp = pbType.GetProperty("rootTransform", BindingFlags.Public | BindingFlags.Instance);
            FieldInfo ignoreTransformsField = pbType.GetField("ignoreTransforms", BindingFlags.Public | BindingFlags.Instance);

            foreach (var pb in pbComponents)
            {
                if (pb == null) continue;

                string pbName = pb.gameObject.name;
                Transform rootT = null;

                if (rootTransformProp != null)
                {
                    rootT = rootTransformProp.GetValue(pb) as Transform;
                }
                if (rootT == null)
                {
                    rootT = pb.transform;
                }

                // Collect ignored sub-transforms if defined
                var ignored = new HashSet<Transform>();
                if (ignoreTransformsField != null)
                {
                    var ignoreList = ignoreTransformsField.GetValue(pb) as System.Collections.IEnumerable;
                    if (ignoreList != null)
                    {
                        foreach (var item in ignoreList)
                        {
                            if (item is Transform t && t != null) ignored.Add(t);
                        }
                    }
                }

                // Recursively register all transforms in this PhysBone hierarchy
                RegisterTransformHierarchy(rootT, pbName, ignored, result);
            }

            return result;
        }

        private static void RegisterTransformHierarchy(
            Transform current,
            string physBoneName,
            HashSet<Transform> ignored,
            Dictionary<Transform, string> registry)
        {
            if (current == null || ignored.Contains(current)) return;

            if (!registry.ContainsKey(current))
            {
                registry[current] = physBoneName;
            }

            for (int i = 0; i < current.childCount; i++)
            {
                RegisterTransformHierarchy(current.GetChild(i), physBoneName, ignored, registry);
            }
        }

        /// <summary>
        /// Runs a full diagnostic scan against an AnimationClip on the specified avatar.
        /// Identifies PhysBone curve conflicts and loop boundary jump discontinuities.
        /// </summary>
        public static PhysBoneDiagnosticReport RunDiagnostics(
            GameObject avatarRoot,
            AnimationClip clip,
            float posJumpThreshold = 0.02f,
            float rotJumpThresholdDeg = 4.0f)
        {
            var report = new PhysBoneDiagnosticReport
            {
                TargetAvatar = avatarRoot,
                Clip = clip
            };

            if (clip == null) return report;

            EditorCurveBinding[] bindings = AnimationUtility.GetCurveBindings(clip);
            report.TotalCurvesScanned = bindings.Length;

            // 1. Diagnose PhysBone conflicts
            if (avatarRoot != null)
            {
                var pbMap = CollectPhysBoneTransforms(avatarRoot);
                report.ControlledTransformsCount = pbMap.Count;

                Type pbType = GetPhysBoneType();
                if (pbType != null)
                {
                    report.PhysBoneComponentsFound = avatarRoot.GetComponentsInChildren(pbType, true).Length;
                }

                foreach (var binding in bindings)
                {
                    Transform t = string.IsNullOrEmpty(binding.path)
                        ? avatarRoot.transform
                        : avatarRoot.transform.Find(binding.path);

                    if (t != null && pbMap.TryGetValue(t, out string pbName))
                    {
                        // Check if the animated property affects Transform physics (Position, Rotation, Scale)
                        if (binding.propertyName.StartsWith("m_LocalPosition") ||
                            binding.propertyName.StartsWith("m_LocalRotation") ||
                            binding.propertyName.StartsWith("m_LocalScale") ||
                            binding.propertyName.StartsWith("localEulerAngles"))
                        {
                            report.Conflicts.Add(new PhysBoneConflict
                            {
                                Binding = binding,
                                Path = binding.path,
                                PropertyName = binding.propertyName,
                                PhysBoneName = pbName,
                                Transform = t
                            });
                        }
                    }
                }
            }

            // 2. Diagnose Loop Boundary Continuity
            report.Discontinuities.AddRange(CheckLoopContinuity(clip, posJumpThreshold, rotJumpThresholdDeg));

            return report;
        }

        /// <summary>
        /// Evaluates all curves in an AnimationClip at t=0 and t=length to detect loop discontinuities or velocity pops.
        /// </summary>
        public static List<LoopDiscontinuity> CheckLoopContinuity(
            AnimationClip clip,
            float posJumpThreshold = 0.02f,
            float rotJumpThresholdDeg = 4.0f)
        {
            var discontinuities = new List<LoopDiscontinuity>();
            if (clip == null || clip.length <= 0.001f) return discontinuities;

            EditorCurveBinding[] bindings = AnimationUtility.GetCurveBindings(clip);
            float length = clip.length;

            foreach (var binding in bindings)
            {
                AnimationCurve curve = AnimationUtility.GetEditorCurve(clip, binding);
                var discontinuity = CheckCurveContinuity(curve, binding.propertyName, binding.path, binding, length, posJumpThreshold, rotJumpThresholdDeg);
                if (discontinuity != null)
                {
                    discontinuities.Add(discontinuity);
                }
            }

            return discontinuities;
        }

        /// <summary>
        /// Evaluates a single animation curve at boundary t=0 and t=length to detect jump discontinuities.
        /// </summary>
        public static LoopDiscontinuity CheckCurveContinuity(
            AnimationCurve curve,
            string propertyName,
            string path = "",
            EditorCurveBinding binding = default,
            float length = -1f,
            float posJumpThreshold = 0.02f,
            float rotJumpThresholdDeg = 4.0f)
        {
            if (curve == null || curve.length < 2) return null;
            if (length <= 0.001f)
            {
                length = curve.keys[curve.length - 1].time;
            }
            if (length <= 0.001f) return null;

            float dt = Mathf.Min(0.016f, length * 0.1f);
            float startVal = curve.Evaluate(0f);
            float endVal = curve.Evaluate(length);
            float delta = Mathf.Abs(startVal - endVal);

            // Compute instantaneous boundary velocity (first derivative)
            float velStart = (curve.Evaluate(dt) - startVal) / dt;
            float velEnd = (endVal - curve.Evaluate(length - dt)) / dt;

            return EvaluateBoundaryDiscontinuity(startVal, endVal, velStart, velEnd, propertyName, path, binding, posJumpThreshold, rotJumpThresholdDeg);
        }

        /// <summary>
        /// Pure managed calculation of boundary jump and velocity discontinuity.
        /// Evaluates start/end differences against position or rotation thresholds.
        /// </summary>
        public static LoopDiscontinuity EvaluateBoundaryDiscontinuity(
            float startVal,
            float endVal,
            float velStart,
            float velEnd,
            string propertyName,
            string path = "",
            EditorCurveBinding binding = default,
            float posJumpThreshold = 0.02f,
            float rotJumpThresholdDeg = 4.0f)
        {
            float delta = Mathf.Abs(startVal - endVal);
            float velDiff = Mathf.Abs(velStart - velEnd);

            bool isPos = propertyName != null && propertyName.StartsWith("m_LocalPosition");
            bool isRot = propertyName != null && (propertyName.StartsWith("m_LocalRotation") || propertyName.StartsWith("localEulerAngles"));

            float jumpThreshold = isPos ? posJumpThreshold : (isRot ? rotJumpThresholdDeg : 0.05f);

            if (delta > jumpThreshold || velDiff > jumpThreshold * 8.0f)
            {
                string severity = (delta > jumpThreshold * 2.5f || velDiff > jumpThreshold * 16.0f) ? "Error" : "Warning";
                string desc = isPos
                    ? $"Position jump: {delta:F3}m (Vel diff: {velDiff:F2}m/s)"
                    : (isRot ? $"Rotation jump: {delta:F1} deg (Vel diff: {velDiff:F1} deg/s)" : $"Property jump: {delta:F3}");

                return new LoopDiscontinuity
                {
                    Binding = binding,
                    Path = path,
                    PropertyName = propertyName,
                    StartValue = startVal,
                    EndValue = endVal,
                    DeltaValue = delta,
                    VelocityDiff = velDiff,
                    Severity = severity,
                    Description = desc
                };
            }

            return null;
        }

        /// <summary>
        /// Removes the specified conflicting curve bindings from an AnimationClip.
        /// Returns the number of curves successfully removed.
        /// </summary>
        public static int RemoveConflictingCurves(AnimationClip clip, IEnumerable<EditorCurveBinding> bindingsToRemove)
        {
            if (clip == null || bindingsToRemove == null) return 0;

            Undo.RecordObject(clip, "Strip PhysBone Conflicting Curves");
            int count = 0;

            foreach (var binding in bindingsToRemove)
            {
                AnimationUtility.SetEditorCurve(clip, binding, null);
                count++;
            }

            EditorUtility.SetDirty(clip);
            return count;
        }

        /// <summary>
        /// Automatically detects and removes all PhysBone-conflicting curves from an AnimationClip.
        /// </summary>
        public static int AutoStripPhysBoneCurves(GameObject avatarRoot, AnimationClip clip)
        {
            if (avatarRoot == null || clip == null) return 0;

            var report = RunDiagnostics(avatarRoot, clip);
            if (!report.HasConflicts) return 0;

            var toRemove = new List<EditorCurveBinding>();
            foreach (var conflict in report.Conflicts)
            {
                toRemove.Add(conflict.Binding);
            }

            return RemoveConflictingCurves(clip, toRemove);
        }
    }
}
