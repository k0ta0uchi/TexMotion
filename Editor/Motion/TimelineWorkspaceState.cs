using System;
using UnityEngine;

namespace TexMotion.Editor.Motion
{
    /// <summary>
    /// Four primary task modes for the Motion Timeline Inspector.
    /// Replaces the monolithic feature stack with focused, single-purpose workflows.
    /// </summary>
    public enum TimelineInspectorMode
    {
        Pose = 0,
        Repair = 1,
        Timing = 2,
        Polish = 3
    }

    /// <summary>
    /// Sub-tools / pages available within the Pose mode.
    /// </summary>
    public enum TimelinePoseTool
    {
        Joint = 0,
        PoseActions = 1,
        Palette = 2,
        HandFace = 3,
        IkPins = 4
    }

    /// <summary>
    /// Focused repair tools for fixing tracking issues and physical artifacts.
    /// </summary>
    public enum TimelineRepairTool
    {
        TrackingAmbiguity = 0,
        MotionGlitches = 1,
        FootGrounding = 2,
        PenetrationLimiter = 3,
        Offset = 4
    }

    /// <summary>
    /// Timing and temporal editing tools.
    /// </summary>
    public enum TimelineTimingTool
    {
        Tween = 0,
        Loop = 1,
        Retime = 2
    }

    /// <summary>
    /// Stylized polish filter categories for progressive disclosure.
    /// </summary>
    public enum TimelinePolishCategory
    {
        TimingSpacing = 0,
        WeightBalance = 1,
        OverlapDrag = 2,
        PoseSilhouette = 3
    }

    /// <summary>
    /// Presets for stylized motion polish.
    /// </summary>
    public enum TimelinePolishPreset
    {
        Action = 0,
        Weight = 1,
        Subtle = 2,
        Custom = 3
    }

    /// <summary>
    /// UI-only workspace state for the redesigned Motion Timeline.
    /// Holds current modes, active tools per mode, scroll positions, and disclosure states.
    /// </summary>
    [Serializable]
    public class TimelineWorkspaceState
    {
        // User-adjustable inspector split; always constrained by the shell.
        public float InspectorWidth = 380f;
        // Active mode
        public TimelineInspectorMode Mode = TimelineInspectorMode.Pose;

        // Active tool per mode
        public TimelinePoseTool ActivePoseTool = TimelinePoseTool.Joint;
        public TimelineRepairTool ActiveRepairTool = TimelineRepairTool.TrackingAmbiguity;
        public TimelineTimingTool ActiveTimingTool = TimelineTimingTool.Tween;
        public TimelinePolishCategory ActivePolishCategory = TimelinePolishCategory.TimingSpacing;
        public TimelinePolishPreset ActivePolishPreset = TimelinePolishPreset.Action;

        // Scroll positions per mode to preserve user view when switching tabs
        public Vector2 PoseScroll = Vector2.zero;
        public Vector2 RepairScroll = Vector2.zero;
        public Vector2 TimingScroll = Vector2.zero;
        public Vector2 PolishScroll = Vector2.zero;

        // Joint picker & filtering in Pose mode
        public string JointSearchFilter = string.Empty;
        public int JointCategoryFilter = 0; // 0: All, 1: Spine/Head, 2: Arms, 3: Legs

        // Polish detail foldout states
        public bool PolishShowCustomDetails = false;
        public int SelectedPolishFilterIndex = -1; // -1: summary, >=0: individual filter detail

        // Glitch navigation state in Repair mode
        public Vector2 GlitchListScroll = Vector2.zero;
        public int SelectedGlitchIndex = -1;

        // Overlay popover state in Viewport
        public bool ShowOverlaysPopover = false;

        // Reset all scroll states when loading a new clip
        public void ResetOnNewClip()
        {
            PoseScroll = Vector2.zero;
            RepairScroll = Vector2.zero;
            TimingScroll = Vector2.zero;
            PolishScroll = Vector2.zero;
            GlitchListScroll = Vector2.zero;
            SelectedGlitchIndex = -1;
        }
    }
}
