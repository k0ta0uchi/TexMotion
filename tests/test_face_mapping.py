"""Unit verification tests for FaceMappingProfile, AnimationClipBuilder face extensions, and VRChat dual-layer synchronization.
"""

from __future__ import annotations

import re
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]
RUNTIME_FACE_MAPPING = ROOT / "Runtime" / "Motion" / "FaceMappingProfile.cs"
RUNTIME_VIDEO_MOTION = ROOT / "Runtime" / "Motion" / "VideoMotionData.cs"
RUNTIME_VRC_CONFIG = ROOT / "Runtime" / "VRChat" / "VrcMotionConfig.cs"
EDITOR_CLIP_BUILDER = ROOT / "Editor" / "Motion" / "AnimationClipBuilder.cs"
EDITOR_FACE_EDITOR = ROOT / "Editor" / "Motion" / "FaceMappingProfileEditor.cs"
EDITOR_MA_SETUP = ROOT / "Editor" / "VRChat" / "ModularAvatarSetup.cs"
EDITOR_DIRECT_SETUP = ROOT / "Editor" / "VRChat" / "VrcDirectSetup.cs"


class TestFaceMappingProfileSource:
    """Verifies FaceMappingProfile.cs source definitions and logic."""

    @pytest.fixture(autouse=True)
    def setup(self):
        assert RUNTIME_FACE_MAPPING.exists(), f"Missing {RUNTIME_FACE_MAPPING}"
        self.content = RUNTIME_FACE_MAPPING.read_text(encoding="utf-8")

    def test_file_structure_and_inheritance(self):
        assert "class FaceMappingProfile : ScriptableObject" in self.content
        assert "class FaceBlendShapeMapping" in self.content
        assert "namespace TexMotion.Runtime.Motion" in self.content

    def test_standard_52_blendshapes_count_and_unique(self):
        # Extract shapes from MediaPipe52BlendShapes array
        match = re.search(r"public static readonly string\[\] MediaPipe52BlendShapes = new string\[\]\s*\{(.*?)\};", self.content, re.DOTALL)
        assert match is not None, "MediaPipe52BlendShapes array not found"
        raw_shapes = re.findall(r'"([A-Za-z0-9_]+)"', match.group(1))
        assert len(raw_shapes) == 52, f"Expected 52 shapes, found {len(raw_shapes)}"
        assert len(set(raw_shapes)) == 52, "Duplicate shape names detected in MediaPipe52BlendShapes"

    def test_opposite_shape_name_symmetry(self):
        assert "GetOppositeShapeName" in self.content
        # Verify specific replacements
        assert 'EndsWith("Left"' in self.content or 'EndsWith("Left"' in self.content
        assert 'EndsWith("Right"' in self.content or 'EndsWith("Right"' in self.content
        assert '_L' in self.content and '_R' in self.content

    def test_weight_evaluation_math(self):
        assert "EvaluateWeight(FaceBlendShapeMapping mapping, float rawWeight01)" in self.content
        assert "DeadZone" in self.content
        assert "Multiplier" in self.content
        assert "GlobalMultiplier" in self.content
        assert "100.0f" in self.content

    def test_auto_generate_mapping_naming_conventions(self):
        assert "AutoGenerateMapping(SkinnedMeshRenderer renderer)" in self.content
        assert "vrc.blink_l" in self.content
        assert "vrc.blink_r" in self.content
        assert "vrc.v_aa" in self.content
        assert "fcl_eye_close_l" in self.content or "Fcl_EYE_Close_L" in self.content
        assert "fcl_mth_a" in self.content or "Fcl_MTH_A" in self.content
        assert "まばたき" in self.content
        assert "ウィンク" in self.content


class TestVideoMotionDataFaceTrack:
    """Verifies FaceTrackData in VideoMotionData.cs."""

    @pytest.fixture(autouse=True)
    def setup(self):
        assert RUNTIME_VIDEO_MOTION.exists(), f"Missing {RUNTIME_VIDEO_MOTION}"
        self.content = RUNTIME_VIDEO_MOTION.read_text(encoding="utf-8")

    def test_face_track_data_helpers(self):
        assert "public FaceTrackData FaceTrack" in self.content
        assert "class FaceTrackData" in self.content
        assert "HasHeadRotations" in self.content
        assert "GetHeadRotation(int frameIndex)" in self.content
        assert "FindShape(string name)" in self.content
        assert "GetWeight(string shapeName, int frameIndex)" in self.content


class TestAnimationClipBuilderExtensions:
    """Verifies AnimationClipBuilder extensions for facial sync and FX Layer export."""

    @pytest.fixture(autouse=True)
    def setup(self):
        assert EDITOR_CLIP_BUILDER.exists(), f"Missing {EDITOR_CLIP_BUILDER}"
        self.content = EDITOR_CLIP_BUILDER.read_text(encoding="utf-8")

    def test_animation_build_options_face_fields(self):
        assert "public FaceMappingProfile FaceMapping;" in self.content
        assert "public SkinnedMeshRenderer TargetFaceRenderer;" in self.content
        assert "public bool ApplyHeadRotationFromFace;" in self.content
        assert "public float HeadRotationBlendWeight;" in self.content
        assert "public bool ExportFaceOnly;" in self.content

    def test_build_clip_aliases(self):
        assert "public static AnimationClip BuildClip(VideoMotionData videoMotionData, AnimationBuildOptions options)" in self.content
        assert "public static AnimationClip BuildClip(GeneratedMotionData motionData, AnimationBuildOptions options)" in self.content

    def test_head_rotation_blending(self):
        assert "options.ApplyHeadRotationFromFace" in self.content
        assert "videoData.FaceTrack.HasHeadRotations" in self.content
        assert "videoData.FaceTrack.GetHeadRotation" in self.content
        assert "Quaternion.Slerp" in self.content

    def test_dynamic_face_track_curves(self):
        assert "BuildFaceTrackCurves" in self.content
        assert "clip.SetCurve(facePath, typeof(SkinnedMeshRenderer)" in self.content
        assert '"blendShape." + mapping.TargetShapeName' in self.content

    def test_face_only_and_synchronized_clip_builders(self):
        assert "BuildFaceAnimationClip" in self.content
        assert "BuildSynchronizedClips" in self.content


class TestVrcDualLayerOutput:
    """Verifies VRChat Action + FX layer dual clip export and synchronization."""

    def test_vrc_motion_config_fields(self):
        content = RUNTIME_VRC_CONFIG.read_text(encoding="utf-8")
        assert "SyncFaceToFxLayer" in content
        assert "FaceMappingProfile FaceMapping" in content

    def test_modular_avatar_setup_with_face(self):
        content = EDITOR_MA_SETUP.read_text(encoding="utf-8")
        assert "SetupAvatarMotionWithFace" in content
        assert "Ctrl_{sanitizedName}_Action.controller" in content
        assert "Ctrl_{sanitizedName}_FX.controller" in content
        assert "Action_Body" in content
        assert "FX_Face" in content

    def test_vrc_direct_setup_with_face(self):
        content = EDITOR_DIRECT_SETUP.read_text(encoding="utf-8")
        assert "SetupDirectAvatarMotionWithFace" in content
        assert "VrcTargetLayer.ActionLayer" in content
        assert "VrcTargetLayer.FXLayer" in content
        assert "AddStateToFxAnimatorController" in content


class TestFaceMappingProfileEditor:
    """Verifies FaceMappingProfileEditor GUI features."""

    def test_editor_gui_elements(self):
        content = EDITOR_FACE_EDITOR.read_text(encoding="utf-8")
        assert "[CustomEditor(typeof(FaceMappingProfile))]" in content
        assert "AutoGenerateMapping" in content
        assert "ResetToStandard52" in content
        assert "TargetMeshPath" in content
        assert "GlobalMultiplier" in content
        assert "ApplyHeadRotation" in content
