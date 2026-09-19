using System;
using System.Collections.Generic;
using TexMotion.Runtime.Motion;
using UnityEditor;
using UnityEngine;

namespace TexMotion.Editor.Motion
{
    /// <summary>
    /// Background viewport modes for the Motion Timeline Editor.
    /// </summary>
    public enum TimelineBackgroundMode
    {
        Stage = 0,        // 3D Studio Stage (Ground grid, floor plane, contact shadows, penetration guides)
        Dark = 1,         // Dark Void (Classic minimalist black)
        GreenScreen = 2   // Chroma Key Green (#00FF00, for clean video cutout compositing)
    }

    /// <summary>
    /// Manages and renders the 3D stage environment inside PreviewRenderUtility.
    /// Provides ground elevation awareness, scale grid, contact shadows, and foot contact indicators.
    /// </summary>
    public class TimelineStageRenderer : IDisposable
    {
        // Viewport Background Colors
        public static readonly Color StageBgColor = new Color(0.10f, 0.12f, 0.15f, 1.0f);
        public static readonly Color DarkBgColor = new Color(0.05f, 0.07f, 0.09f, 1.0f);
        public static readonly Color GreenScreenColor = new Color(0.0f, 1.0f, 0.0f, 1.0f);

        private Mesh _gridMesh;
        private Mesh _floorMesh;
        private Mesh _discMesh;
        private Material _coloredMaterial;
        private bool _isDisposed;

        public TimelineStageRenderer()
        {
            EnsureResources();
        }

        private void EnsureResources()
        {
            if (_coloredMaterial == null)
            {
                Shader shader = Shader.Find("Hidden/Internal-Colored");
                if (shader == null) shader = Shader.Find("Sprites/Default");
                if (shader == null) shader = Shader.Find("UI/Default");

                _coloredMaterial = new Material(shader);
                _coloredMaterial.hideFlags = HideFlags.HideAndDontSave;
            }

            if (_gridMesh == null)
            {
                _gridMesh = BuildGridMesh(5.0f, 0.2f, 1.0f);
            }

            if (_floorMesh == null)
            {
                _floorMesh = BuildFloorDiscMesh(5.5f, 36);
            }

            if (_discMesh == null)
            {
                _discMesh = BuildUnitDiscMesh(24);
            }
        }

        /// <summary>
        /// Renders the 3D stage floor, grid, and dynamic foot contact shadows into the PreviewRenderUtility scene.
        /// Must be called before pru.Render().
        /// </summary>
        public void RenderStage(
            PreviewRenderUtility pru,
            Vector3 lFootWorldPos,
            Vector3 rFootWorldPos,
            bool hasValidFeet)
        {
            if (pru == null || _isDisposed) return;
            EnsureResources();

            if (_coloredMaterial == null) return;

            // 1. Draw subtle studio floor plane vignette
            if (_floorMesh != null)
            {
                pru.DrawMesh(_floorMesh, Matrix4x4.Translate(new Vector3(0f, -0.001f, 0f)), _coloredMaterial, 0);
            }

            // 2. Draw Ground Grid (Scale & Orientation)
            if (_gridMesh != null)
            {
                pru.DrawMesh(_gridMesh, Matrix4x4.identity, _coloredMaterial, 0);
            }

            // 3. Draw Dynamic Foot Contact Shadows and Penetration Guides
            if (hasValidFeet && _discMesh != null)
            {
                DrawFootContactShadow(pru, lFootWorldPos);
                DrawFootContactShadow(pru, rFootWorldPos);
            }
        }

        private void DrawFootContactShadow(PreviewRenderUtility pru, Vector3 footPos)
        {
            float height = footPos.y; // 0 = exactly touching ground
            float clampedHeight = Mathf.Max(0f, height);

            // Contact shadow parameters: close to ground = small & crisp & dark; high in air = large & diffuse & faint
            float shadowRadius = Mathf.Lerp(0.14f, 0.45f, Mathf.Clamp01(clampedHeight / 1.0f));
            float shadowAlpha = Mathf.Lerp(0.65f, 0.05f, Mathf.Clamp01(clampedHeight / 0.8f));

            if (shadowAlpha > 0.01f)
            {
                Vector3 shadowPos = new Vector3(footPos.x, 0.001f, footPos.z);
                Matrix4x4 shadowMatrix = Matrix4x4.TRS(
                    shadowPos,
                    Quaternion.identity,
                    new Vector3(shadowRadius, 1f, shadowRadius));

                pru.DrawMesh(_discMesh, shadowMatrix, _coloredMaterial, 0);
            }

            // If foot penetrates below ground level (height < -0.015m), draw an alert ring at ground level
            if (height < -0.015f)
            {
                float penRadius = Mathf.Lerp(0.12f, 0.28f, Mathf.Clamp01(-height / 0.2f));
                Vector3 penPos = new Vector3(footPos.x, 0.003f, footPos.z);
                Matrix4x4 penMatrix = Matrix4x4.TRS(
                    penPos,
                    Quaternion.identity,
                    new Vector3(penRadius, 1f, penRadius));

                pru.DrawMesh(_discMesh, penMatrix, _coloredMaterial, 0);
            }
        }

        #region Mesh Generation

        private static Mesh BuildGridMesh(float halfSize, float subStep, float mainStep)
        {
            var vertices = new List<Vector3>();
            var colors = new List<Color>();
            var indices = new List<int>();

            int count = Mathf.RoundToInt(halfSize / subStep);
            int idx = 0;

            for (int i = -count; i <= count; i++)
            {
                float coord = i * subStep;
                bool isOrigin = Mathf.Abs(coord) < 0.001f;
                bool isMain = Mathf.Abs(coord % mainStep) < 0.01f || Mathf.Abs(coord % mainStep - mainStep) < 0.01f;

                // X-axis line (along X, fixed Z)
                Color colZ;
                if (isOrigin)
                {
                    colZ = new Color(0.90f, 0.35f, 0.35f, 0.90f); // X-Axis (Red)
                }
                else if (isMain)
                {
                    colZ = new Color(0.42f, 0.48f, 0.58f, 0.55f);
                }
                else
                {
                    colZ = new Color(0.24f, 0.28f, 0.35f, 0.24f);
                }

                // Fade edges along X
                float edgeFadeZ = 1f - Mathf.Clamp01(Mathf.Abs(coord) / halfSize);
                AddLine(vertices, colors, indices, ref idx,
                    new Vector3(-halfSize, 0f, coord),
                    new Vector3(halfSize, 0f, coord),
                    colZ, edgeFadeZ);

                // Z-axis line (along Z, fixed X)
                Color colX;
                if (isOrigin)
                {
                    colX = new Color(0.35f, 0.60f, 0.95f, 0.90f); // Z-Axis (Blue)
                }
                else if (isMain)
                {
                    colX = new Color(0.42f, 0.48f, 0.58f, 0.55f);
                }
                else
                {
                    colX = new Color(0.24f, 0.28f, 0.35f, 0.24f);
                }

                float edgeFadeX = 1f - Mathf.Clamp01(Mathf.Abs(coord) / halfSize);
                AddLine(vertices, colors, indices, ref idx,
                    new Vector3(coord, 0f, -halfSize),
                    new Vector3(coord, 0f, halfSize),
                    colX, edgeFadeX);
            }

            var mesh = new Mesh
            {
                name = "TimelineStage_Grid",
                hideFlags = HideFlags.HideAndDontSave
            };
            mesh.SetVertices(vertices);
            mesh.SetColors(colors);
            mesh.SetIndices(indices.ToArray(), MeshTopology.Lines, 0);
            mesh.UploadMeshData(false);
            return mesh;
        }

        private static void AddLine(List<Vector3> vertices, List<Color> colors, List<int> indices, ref int idx, Vector3 p1, Vector3 p2, Color baseColor, float edgeFade)
        {
            // Center of the line is less faded than tips
            Color tipColor = baseColor;
            tipColor.a *= (0.15f * edgeFade);

            Color centerColor = baseColor;
            centerColor.a *= edgeFade;

            Vector3 mid = (p1 + p2) * 0.5f;

            // Segment 1: p1 to mid
            vertices.Add(p1);
            colors.Add(tipColor);
            indices.Add(idx++);

            vertices.Add(mid);
            colors.Add(centerColor);
            indices.Add(idx++);

            // Segment 2: mid to p2
            vertices.Add(mid);
            colors.Add(centerColor);
            indices.Add(idx++);

            vertices.Add(p2);
            colors.Add(tipColor);
            indices.Add(idx++);
        }

        private static Mesh BuildFloorDiscMesh(float radius, int segments)
        {
            var vertices = new List<Vector3>(segments + 1);
            var colors = new List<Color>(segments + 1);
            var indices = new List<int>(segments * 3);

            // Center vertex
            vertices.Add(Vector3.zero);
            colors.Add(new Color(0.13f, 0.16f, 0.20f, 0.85f));

            // Perimeter vertices (transparent fade)
            Color rimColor = new Color(StageBgColor.r, StageBgColor.g, StageBgColor.b, 0.0f);
            for (int i = 0; i < segments; i++)
            {
                float angle = (i / (float)segments) * Mathf.PI * 2f;
                vertices.Add(new Vector3(Mathf.Cos(angle) * radius, 0f, Mathf.Sin(angle) * radius));
                colors.Add(rimColor);
            }

            for (int i = 0; i < segments; i++)
            {
                int next = (i + 1) % segments;
                indices.Add(0);
                indices.Add(i + 1);
                indices.Add(next + 1);
            }

            var mesh = new Mesh
            {
                name = "TimelineStage_Floor",
                hideFlags = HideFlags.HideAndDontSave
            };
            mesh.SetVertices(vertices);
            mesh.SetColors(colors);
            mesh.SetTriangles(indices, 0);
            mesh.UploadMeshData(false);
            return mesh;
        }

        private static Mesh BuildUnitDiscMesh(int segments)
        {
            var vertices = new List<Vector3>(segments + 1);
            var colors = new List<Color>(segments + 1);
            var indices = new List<int>(segments * 3);

            // Center: dark shadow core
            vertices.Add(Vector3.zero);
            colors.Add(new Color(0.02f, 0.03f, 0.05f, 0.65f));

            // Outer rim: soft falloff
            Color rimColor = new Color(0.02f, 0.03f, 0.05f, 0.0f);
            for (int i = 0; i < segments; i++)
            {
                float angle = (i / (float)segments) * Mathf.PI * 2f;
                vertices.Add(new Vector3(Mathf.Cos(angle), 0f, Mathf.Sin(angle)));
                colors.Add(rimColor);
            }

            for (int i = 0; i < segments; i++)
            {
                int next = (i + 1) % segments;
                indices.Add(0);
                indices.Add(i + 1);
                indices.Add(next + 1);
            }

            var mesh = new Mesh
            {
                name = "TimelineStage_ContactShadow",
                hideFlags = HideFlags.HideAndDontSave
            };
            mesh.SetVertices(vertices);
            mesh.SetColors(colors);
            mesh.SetTriangles(indices, 0);
            mesh.UploadMeshData(false);
            return mesh;
        }

        #endregion

        public void Dispose()
        {
            if (_isDisposed) return;
            _isDisposed = true;

            if (_coloredMaterial != null)
            {
                UnityEngine.Object.DestroyImmediate(_coloredMaterial);
                _coloredMaterial = null;
            }
            if (_gridMesh != null)
            {
                UnityEngine.Object.DestroyImmediate(_gridMesh);
                _gridMesh = null;
            }
            if (_floorMesh != null)
            {
                UnityEngine.Object.DestroyImmediate(_floorMesh);
                _floorMesh = null;
            }
            if (_discMesh != null)
            {
                UnityEngine.Object.DestroyImmediate(_discMesh);
                _discMesh = null;
            }
        }
    }
}
