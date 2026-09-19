using System;
using System.Collections.Generic;
using UnityEditor;
using UnityEngine;

namespace TexMotion.Editor.Motion
{
    /// <summary>
    /// Shared IMGUI theme for the Motion Timeline window.
    ///
    /// DESIGN.md describes the editor as a midnight surface with a small number of
    /// deliberate accents. Keeping the tokens and generated textures here means the
    /// timeline can use the same surfaces in every panel without allocating a new
    /// Texture2D or GUIStyle during OnGUI.
    /// </summary>
    public static class MotionTimelineTheme
    {
        // DESIGN.md color tokens. Keep these as named values so callers do not need
        // to repeat subtly different near-black colors in individual controls.
        public static readonly Color Void = Rgb(8, 9, 10);
        public static readonly Color Carbon = Rgb(15, 16, 17);
        public static readonly Color Obsidian = Rgb(22, 23, 24);
        public static readonly Color Graphite = Rgb(35, 37, 42);
        public static readonly Color Smoke = Rgb(56, 59, 63);
        public static readonly Color Ash = Rgb(98, 102, 109);
        public static readonly Color Fog = Rgb(138, 143, 152);
        public static readonly Color Mist = Rgb(208, 214, 224);
        public static readonly Color Bone = Rgb(229, 229, 230);
        public static readonly Color Paper = Color.white;

        public static readonly Color AcidLime = Rgb(228, 242, 34);
        public static readonly Color PulseGreen = Rgb(39, 166, 68);
        public static readonly Color CoralRed = Rgb(235, 87, 87);
        public static readonly Color SignalTeal = Rgb(2, 184, 204);
        public static readonly Color IrisViolet = Rgb(99, 102, 241);
        public static readonly Color Lavender = Rgb(139, 92, 246);

        // Slate is the name used for the interactive surface in DESIGN.md. It is
        // intentionally an alias of Graphite, whose token is also the border color.
        public static readonly Color Slate = Graphite;

        /// <summary>Common cached style roles used by the Timeline UI.</summary>
        public enum StyleRole
        {
            Panel,
            Card,
            Toolbar,
            Heading,
            Label,
            MutedLabel,
            Button,
            GhostButton,
            PrimaryButton,
            Badge,
            SuccessBadge,
            WarningBadge,
            ErrorBadge,
            InfoBadge,
            TabActive,
            TabInactive,
            SectionHeader
        }

        private struct TextureKey : IEquatable<TextureKey>
        {
            public readonly Color32 Color;
            public readonly int Radius;

            public TextureKey(Color32 color, int radius)
            {
                Color = color;
                Radius = radius;
            }

            public bool Equals(TextureKey other)
            {
                return Color.Equals(other.Color) && Radius == other.Radius;
            }

            public override bool Equals(object obj)
            {
                return obj is TextureKey && Equals((TextureKey)obj);
            }

            public override int GetHashCode()
            {
                unchecked
                {
                    int hash = Color.r;
                    hash = (hash * 397) ^ Color.g;
                    hash = (hash * 397) ^ Color.b;
                    hash = (hash * 397) ^ Color.a;
                    return (hash * 397) ^ Radius;
                }
            }
        }

        private static readonly Dictionary<TextureKey, Texture2D> TextureCache =
            new Dictionary<TextureKey, Texture2D>();
        private static readonly Dictionary<StyleRole, GUIStyle> StyleCache =
            new Dictionary<StyleRole, GUIStyle>();

        // Styles are only requested after the editor domain is initialized, but the
        // reload hook ensures generated textures do not survive an assembly reload.
        static MotionTimelineTheme()
        {
            AssemblyReloadEvents.beforeAssemblyReload += ClearCaches;
        }

        /// <summary>
        /// Gets a cached, 9-slice-compatible rounded texture for a surface.
        /// Radius is clamped to keep the generated texture small and deterministic.
        /// </summary>
        public static Texture2D RoundedTexture(Color color, int radius = 6)
        {
            radius = Mathf.Clamp(radius, 0, 16);
            var key = new TextureKey((Color32)color, radius);
            Texture2D texture;
            if (TextureCache.TryGetValue(key, out texture) && texture != null)
                return texture;

            int size = Mathf.Max(1, radius * 2 + 1);
            texture = new Texture2D(size, size, TextureFormat.RGBA32, false)
            {
                name = "TexMotion Timeline Theme Surface",
                hideFlags = HideFlags.HideAndDontSave,
                wrapMode = TextureWrapMode.Clamp,
                filterMode = FilterMode.Bilinear
            };

            Color32 baseColor = color;
            var pixels = new Color32[size * size];
            for (int y = 0; y < size; y++)
            {
                for (int x = 0; x < size; x++)
                {
                    byte alpha = baseColor.a;
                    if (radius > 0)
                    {
                        float dx = Mathf.Max(radius - x - 0.5f, x - (size - radius - 0.5f), 0f);
                        float dy = Mathf.Max(radius - y - 0.5f, y - (size - radius - 0.5f), 0f);
                        float distance = Mathf.Sqrt(dx * dx + dy * dy);
                        // One pixel of coverage smoothing avoids jagged corners while
                        // preserving a fully opaque center for 9-slice stretching.
                        float coverage = Mathf.Clamp01(radius + 0.5f - distance);
                        alpha = (byte)Mathf.RoundToInt(alpha * coverage);
                    }

                    pixels[y * size + x] = new Color32(baseColor.r, baseColor.g, baseColor.b, alpha);
                }
            }

            texture.SetPixels32(pixels);
            texture.Apply(false, true);
            TextureCache[key] = texture;
            return texture;
        }

        /// <summary>Gets a cached style following the specified DESIGN.md role.</summary>
        public static GUIStyle GetStyle(StyleRole role)
        {
            GUIStyle style;
            if (StyleCache.TryGetValue(role, out style) && style != null)
                return style;

            style = BuildStyle(role);
            StyleCache[role] = style;
            return style;
        }

        // Named accessors keep call sites readable and make role intent obvious in
        // code that is laying out a panel or a primary action.
        public static GUIStyle Panel { get { return GetStyle(StyleRole.Panel); } }
        public static GUIStyle Card { get { return GetStyle(StyleRole.Card); } }
        public static GUIStyle Toolbar { get { return GetStyle(StyleRole.Toolbar); } }
        public static GUIStyle Heading { get { return GetStyle(StyleRole.Heading); } }
        public static GUIStyle Label { get { return GetStyle(StyleRole.Label); } }
        public static GUIStyle MutedLabel { get { return GetStyle(StyleRole.MutedLabel); } }
        public static GUIStyle Button { get { return GetStyle(StyleRole.Button); } }
        public static GUIStyle GhostButton { get { return GetStyle(StyleRole.GhostButton); } }
        public static GUIStyle PrimaryButton { get { return GetStyle(StyleRole.PrimaryButton); } }
        public static GUIStyle Badge { get { return GetStyle(StyleRole.Badge); } }
        public static GUIStyle SuccessBadge { get { return GetStyle(StyleRole.SuccessBadge); } }
        public static GUIStyle WarningBadge { get { return GetStyle(StyleRole.WarningBadge); } }
        public static GUIStyle ErrorBadge { get { return GetStyle(StyleRole.ErrorBadge); } }
        public static GUIStyle InfoBadge { get { return GetStyle(StyleRole.InfoBadge); } }
        public static GUIStyle TabActive { get { return GetStyle(StyleRole.TabActive); } }
        public static GUIStyle TabInactive { get { return GetStyle(StyleRole.TabInactive); } }
        public static GUIStyle SectionHeader { get { return GetStyle(StyleRole.SectionHeader); } }

        /// <summary>
        /// Creates a style for a one-off control while reusing the cached surface
        /// texture. Use GetStyle for repeated controls; this method is useful when a
        /// panel needs custom padding or alignment.
        /// </summary>
        public static GUIStyle CreateStyle(
            GUIStyle baseStyle,
            Color background,
            Color foreground,
            int radius = 6,
            int padding = 8,
            TextAnchor alignment = TextAnchor.MiddleLeft,
            int fontSize = 0,
            int verticalPadding = -1)
        {
            return BuildStyle(
                baseStyle ?? GUIStyle.none,
                background,
                foreground,
                radius,
                padding,
                alignment,
                fontSize,
                verticalPadding);
        }

        /// <summary>Returns a DESIGN.md status tint with a caller-selected alpha.</summary>
        public static Color WithAlpha(Color color, float alpha)
        {
            color.a = Mathf.Clamp01(alpha);
            return color;
        }

        /// <summary>
        /// Releases generated editor textures. Normally invoked by Unity's assembly
        /// reload hook; it is public for tests and editor shutdown paths.
        /// </summary>
        public static void ClearCaches()
        {
            foreach (var pair in TextureCache)
            {
                if (pair.Value != null)
                    UnityEngine.Object.DestroyImmediate(pair.Value);
            }
            TextureCache.Clear();
            StyleCache.Clear();
        }

        private static GUIStyle BuildStyle(StyleRole role)
        {
            switch (role)
            {
                case StyleRole.Panel:
                    return BuildStyle(EditorStyles.helpBox, Carbon, Mist, 12, 12, TextAnchor.UpperLeft, 0);
                case StyleRole.Card:
                    return BuildStyle(EditorStyles.helpBox, Obsidian, Mist, 6, 8, TextAnchor.UpperLeft, 0);
                case StyleRole.Toolbar:
                    return BuildStyle(EditorStyles.toolbar, Carbon, Mist, 6, 6, TextAnchor.MiddleLeft, 11, 1);
                case StyleRole.Heading:
                    return BuildStyle(EditorStyles.label, Color.clear, Paper, 0, 0, TextAnchor.MiddleLeft, 13);
                case StyleRole.Label:
                    return BuildStyle(EditorStyles.label, Color.clear, Mist, 0, 0, TextAnchor.MiddleLeft, 0);
                case StyleRole.MutedLabel:
                    return BuildStyle(EditorStyles.miniLabel, Color.clear, Fog, 0, 0, TextAnchor.MiddleLeft, 0);
                case StyleRole.PrimaryButton:
                    // Buttons are commonly given a fixed 24-28px IMGUI height. Keep
                    // vertical padding compact so the label retains a full text row.
                    // EditorStyles.miniButton carries skin-specific font metrics on
                    // some Unity versions. Cloning the label style gives the custom
                    // surface a stable font/line-height while we still provide our
                    // own normal/hover/active backgrounds below.
                    return BuildStyle(EditorStyles.label, AcidLime, Void, 6, 6, TextAnchor.MiddleCenter, 12, 1);
                case StyleRole.GhostButton:
                    return BuildStyle(EditorStyles.label, Carbon, Mist, 6, 4, TextAnchor.MiddleCenter, 11, 1);
                case StyleRole.Button:
                    return BuildStyle(EditorStyles.label, Slate, Mist, 6, 4, TextAnchor.MiddleCenter, 11, 1);
                case StyleRole.SuccessBadge:
                    return BuildStyle(EditorStyles.miniLabel, WithAlpha(PulseGreen, 0.16f), PulseGreen, 4, 4, TextAnchor.MiddleCenter, 10, 1);
                case StyleRole.WarningBadge:
                    return BuildStyle(EditorStyles.miniLabel, WithAlpha(AcidLime, 0.16f), AcidLime, 4, 4, TextAnchor.MiddleCenter, 10, 1);
                case StyleRole.ErrorBadge:
                    return BuildStyle(EditorStyles.miniLabel, WithAlpha(CoralRed, 0.16f), CoralRed, 4, 4, TextAnchor.MiddleCenter, 10, 1);
                case StyleRole.InfoBadge:
                    return BuildStyle(EditorStyles.miniLabel, WithAlpha(SignalTeal, 0.16f), SignalTeal, 4, 4, TextAnchor.MiddleCenter, 10, 1);
                case StyleRole.TabActive:
                    return BuildStyle(EditorStyles.label, Obsidian, Paper, 4, 8, TextAnchor.MiddleCenter, 12, 4);
                case StyleRole.TabInactive:
                    return BuildStyle(EditorStyles.label, Carbon, Fog, 4, 8, TextAnchor.MiddleCenter, 12, 4);
                case StyleRole.SectionHeader:
                    return BuildStyle(EditorStyles.boldLabel, Color.clear, Bone, 0, 0, TextAnchor.MiddleLeft, 12);
                case StyleRole.Badge:
                default:
                    return BuildStyle(EditorStyles.miniLabel, WithAlpha(Color.white, 0.05f), Fog, 4, 4, TextAnchor.MiddleCenter, 10, 1);
            }
        }

        /// <summary>Draws a 1-pixel hairline border or separator using Graphite or custom color.</summary>
        public static void DrawHairline(Rect rect, Color color)
        {
            EditorGUI.DrawRect(rect, color);
        }

        private static GUIStyle BuildStyle(
            GUIStyle baseStyle,
            Color background,
            Color foreground,
            int radius,
            int padding,
            TextAnchor alignment,
            int fontSize,
            int verticalPadding = -1)
        {
            if (verticalPadding < 0)
                verticalPadding = padding;
            var style = new GUIStyle(baseStyle ?? GUIStyle.none)
            {
                alignment = alignment,
                padding = new RectOffset(padding, padding, verticalPadding, verticalPadding),
                // Layout callers provide their own 4/8px rhythm. Margins inherited
                // from EditorStyles shrink fixed-height buttons before Unity lays
                // out their text, which is especially destructive for Japanese UI.
                margin = new RectOffset(0, 0, 0, 0),
                border = new RectOffset(radius, radius, radius, radius),
                wordWrap = false,
                clipping = TextClipping.Clip,
                stretchHeight = false,
                stretchWidth = false,
                fixedHeight = 0f,
                fixedWidth = 0f,
                contentOffset = Vector2.zero
            };
            // A miniButton's font can be null or carry a one-pixel line metric on
            // older editor skins. Use the regular editor label font as a stable
            // fallback; this is what keeps button labels from rendering as hairlines.
            if (style.font == null)
                style.font = EditorStyles.label.font;
            if (fontSize > 0)
                style.fontSize = fontSize;

            Texture2D normal = RoundedTexture(background, radius);
            Texture2D hover = RoundedTexture(Blend(background, Color.white, 0.08f), radius);
            Texture2D active = RoundedTexture(Blend(background, Color.black, 0.12f), radius);
            style.normal.background = normal;
            style.hover.background = hover;
            style.active.background = active;
            style.focused.background = hover;
            style.onNormal.background = normal;
            style.onHover.background = hover;
            style.onActive.background = active;
            style.onFocused.background = hover;
            style.normal.textColor = foreground;
            style.hover.textColor = foreground;
            style.active.textColor = foreground;
            style.focused.textColor = foreground;
            // Toggle controls use the on* states; without these assignments a
            // selected toolbar button can inherit a transparent skin color even
            // while its background remains visible.
            style.onNormal.textColor = foreground;
            style.onHover.textColor = foreground;
            style.onActive.textColor = foreground;
            style.onFocused.textColor = foreground;
            return style;
        }

        private static Color Blend(Color from, Color to, float amount)
        {
            return Color.Lerp(from, to, Mathf.Clamp01(amount));
        }

        private static Color Rgb(byte red, byte green, byte blue)
        {
            return new Color(red / 255f, green / 255f, blue / 255f, 1f);
        }
    }
}
