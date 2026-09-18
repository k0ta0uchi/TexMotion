"""Static contract tests for the Unity Video2Motion asset catalog.

The Unity editor assembly is not loaded by the Python test environment.  These
checks keep the catalog contract honest nevertheless: every runtime role must
be named in the downloader, the bundled manifest must describe the same WHAM
stages, and the settings/downloader must expose deterministic preflight APIs.
"""

from __future__ import annotations

import json
from pathlib import Path


VIDEO_DIR = Path(__file__).resolve().parents[1]
ROOT = VIDEO_DIR.parents[1]
DOWNLOADER = VIDEO_DIR / "VideoModelDownloader.cs"
SETTINGS = ROOT / "Editor" / "TexMotionSettings.cs"
MANIFEST = VIDEO_DIR / "models" / "wham_asset_manifest.json"


def test_catalog_names_every_runtime_role_and_hmr2_contract() -> None:
    source = DOWNLOADER.read_text(encoding="utf-8-sig")
    required_roles = {
        "WhamCheckpoint",
        "Hmr2Checkpoint",
        "WhamBodyModel",
        "WhamImageFeatureBackbone",
        "WhamImageFeatureArchive",
        "WhamImageFeatureModelDefinition",
        "WhamImageFeatureConfig",
        "WhamCamera",
        "WhamDpvo",
        "WhamAdapter",
        "Hmr2Runtime",
        "PythonRequirements",
        "PythonRuntime",
        "Hmr2Adapter",
        "HybrIKAdapter",
    }
    for role in required_roles:
        assert role in source
    assert "hmr2a_model.tar.gz" in source
    assert "https://github.com/shubham-goel/4D-Humans" in source


def test_catalog_and_settings_expose_deterministic_preflight_paths() -> None:
    downloader = DOWNLOADER.read_text(encoding="utf-8-sig")
    settings = SETTINGS.read_text(encoding="utf-8-sig")
    for method in (
        "GetRuntimeAssetStatuses",
        "GetPreflightReport",
        "GetMissingAssetDetails",
        "GetConfiguredModelPath",
    ):
        assert method in downloader
    for method in (
        "GetEffectiveVideoModelDirectory",
        "GetVideoRequirementsPath",
        "GetEffectiveVideoPythonExecutablePath",
        "GetHMR2RuntimePath",
    ):
        assert method in settings
    assert "HMR2RuntimePath" in settings


def test_downloader_uses_catalog_asset_validation_scope() -> None:
    """Download transactions must call the shared catalog validator.

    ``ValidateAssetFile`` lives on ``VideoModelCatalog`` so preflight and
    downloaded-file validation use the same checksum/size policy.  Keeping
    the qualified calls here prevents Unity's CS0103 regression when the
    downloader is compiled as a separate class in the editor assembly.
    """
    source = DOWNLOADER.read_text(encoding="utf-8-sig")
    catalog_start = source.index("public static class VideoModelCatalog")
    downloader_start = source.index("public static class VideoModelDownloader")
    catalog = source[catalog_start:downloader_start]
    downloader = source[downloader_start:]

    assert "internal static bool ValidateAssetFile" in catalog
    assert "VideoModelCatalog.ValidateAssetFile(destinationPath" in downloader
    assert "VideoModelCatalog.ValidateAssetFile(tempPath" in downloader
    # Unqualified calls are valid in the catalog itself, but must not leak
    # into the separate downloader class.
    assert "if (!ValidateAssetFile(" not in downloader


def test_job_runner_explains_windows_native_access_violation() -> None:
    """The Unity exception must decode the native crash code users see."""
    runner = (VIDEO_DIR / "VideoMotionJobRunner.cs").read_text(encoding="utf-8-sig")
    assert "DescribeNativeExitCode" in runner
    assert "0xC0000005u" in runner
    assert "Native process access violation" in runner
    assert 'msg += "\\nDiagnostic: "' in runner


def test_job_runner_vitpose_status_dto_field_uses_json_casing() -> None:
    """The DTO field must match the extractor's ``vitpose2DStatus`` key."""
    runner = (VIDEO_DIR / "VideoMotionJobRunner.cs").read_text(encoding="utf-8-sig")
    assert "public string vitpose2DStatus;" in runner
    assert "dto.vitPose2DStatus" not in runner
    assert "dto.backendMetadata?.vitPose2DStatus" not in runner
    assert "vitPose2DStatus = ExtractString" not in runner
    assert "if (settings != null)" in (ROOT / "Editor" / "TexMotionWindow.cs").read_text(encoding="utf-8-sig")


def test_video_preview_timeout_does_not_fire_during_extraction() -> None:
    """A stale VideoPlayer must not mask an asynchronous extraction failure."""
    window = (ROOT / "Editor" / "TexMotionWindow.cs").read_text(encoding="utf-8-sig")
    timeout_check = window.index("double elapsed = currentTime - _videoPrepareStartTime")
    extracting_guard = window.rfind("if (_isVideoExtracting)", 0, timeout_check)
    assert extracting_guard >= 0
    assert timeout_check - extracting_guard < 2000
    assert "motion extraction is unaffected" in window


def test_manifest_lists_exact_wham_stage_roles() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    roles = {asset["role"] for asset in manifest["assets"]}
    assert {
        "checkpoint",
        "body_model",
        "image_feature_backbone",
        "image_feature_model_definition",
        "image_feature_config",
        "camera",
        "dpvo",
        "adapter",
    } <= roles


def test_manual_asset_rows_expose_testable_in_editor_guides() -> None:
    downloader = DOWNLOADER.read_text(encoding="utf-8-sig")
    window = (ROOT / "Editor" / "TexMotionWindow.cs").read_text(encoding="utf-8-sig")
    guide = (ROOT / "Editor" / "Video" / "VideoAssetGuideWindow.cs").read_text(encoding="utf-8-sig")

    for asset_id in (
        "hmr2",
        "hmr2-adapter",
        "hmr2-runtime",
        "hmr2-smpl-body",
        "wham-smplx-body",
        "wham-image-feature-archive",
        "wham-vitpose-model-definition",
        "wham-vitpose-runner-config",
        "hybrik-adapter",
    ):
        assert asset_id in downloader

    assert "VideoAssetGuide" in downloader
    assert "Hmr2BodyModel" in downloader
    assert "HMR2BodyModelPath" in (ROOT / "Editor" / "TexMotionSettings.cs").read_text(encoding="utf-8-sig")
    assert "HMR2AdapterPath" in (ROOT / "Editor" / "TexMotionSettings.cs").read_text(encoding="utf-8-sig")
    assert "HybrIKAdapterPath" in (ROOT / "Editor" / "TexMotionSettings.cs").read_text(encoding="utf-8-sig")
    assert "VideoAssetGuideWindow.Show" in window
    assert "OfficialUrl" in guide
    assert "RecommendedInstallDirectory" in guide
    assert "Preflight" in guide
    assert "License" in guide


def test_manual_asset_guide_follows_editor_language_setting() -> None:
    guide = (ROOT / "Editor" / "Video" / "VideoAssetGuideWindow.cs").read_text(encoding="utf-8-sig")
    localization = (ROOT / "Editor" / "Localization" / "TexMotionLocalization.cs").read_text(encoding="utf-8-sig")

    assert "DrawLanguageToolbar" in guide
    assert "TexMotionLanguage.Japanese" in guide
    assert "GetPreflightSteps" in guide
    assert "GetLicenseReason" in guide
    assert "TexMotionLocalization.TrLiteral" in guide
    assert '"Video Asset Guide"' in localization
    assert '"公式ダウンロード／登録ページを開く"' in localization


def test_vitpose_feature_archive_has_one_click_export_contract() -> None:
    window = (ROOT / "Editor" / "TexMotionWindow.cs").read_text(encoding="utf-8-sig")
    runner = (VIDEO_DIR / "VideoMotionJobRunner.cs").read_text(encoding="utf-8-sig")
    extractor = (VIDEO_DIR / "video_pose_extractor.py").read_text(encoding="utf-8-sig")
    preprocess = (VIDEO_DIR / "pose_pipeline" / "wham_preprocess.py").read_text(encoding="utf-8-sig")

    assert "DrawViTPoseFeatureExportControl" in window
    assert "StartViTPoseFeatureExportAsync" in window
    assert "RunViTPoseFeatureExportAsync" in runner
    assert "--export-vitpose-features" in runner
    assert "export_vitpose_features_from_video" in extractor
    assert "frame_indices" in preprocess
    assert "settings.WHAMImageFeaturePath = resultPath" in window


def test_settings_consolidates_wham_workflow_and_hides_other_assets() -> None:
    """The Settings surface must keep WHAM setup discoverable as one workflow."""
    window = (ROOT / "Editor" / "TexMotionWindow.cs").read_text(encoding="utf-8-sig")

    assert 'FoldoutSettingsWham = "settings.video.wham"' in window
    assert 'FoldoutSettingsVideoAdvancedAssets = "settings.video.advanced-assets"' in window
    assert "DrawWhamWorkflowCard" in window
    assert "DrawWhamCompanionAssetRows" in window
    assert "DrawWhamCompanionPathSettings(settings)" in window
    assert '"Other Video2Motion Assets"' in window
    # The generic catalog must not render the WHAM checkpoint a second time.
    assert "if (model.Kind == VideoModelKind.WHAM) continue;" in window


def test_simplified_wham_setup_card_is_the_settings_entry_point() -> None:
    """The WHAM setup surface must expose one compact workflow.

    Keep this contract at the source boundary because Unity IMGUI is not
    available in the Python test process.  The implementation is expected to
    keep the legacy detailed helpers for explicit advanced settings, but the
    normal Video2Motion card must call the simplified entry point directly.
    """
    window = (ROOT / "Editor" / "TexMotionWindow.cs").read_text(encoding="utf-8-sig")
    start = window.index("private void DrawVideo2MotionModelSettings")
    end = window.index("private void DrawWhamWorkflowCard", start)
    entry = window[start:end].split("return;", 1)[0]

    assert "DrawSimplifiedWhamSetupCard(settings" in entry
    assert "DrawWhamCompanionPathSettings(settings)" not in entry
    assert "DrawViTPoseFeatureExportControl(settings)" not in entry
    assert "DrawWhamSetupStatus" in window
    assert "GetWhamSetupSummary" in window
    assert "RequiredAssets" in window
    assert "OptionalAssets" in window


def test_wham_ui_separates_official_roles_and_runtime_provenance() -> None:
    window = (ROOT / "Editor" / "TexMotionWindow.cs").read_text(encoding="utf-8-sig")
    downloader = (DOWNLOADER).read_text(encoding="utf-8-sig")

    for label in (
        "Official WHAM",
        "HMR2 image feature runner",
        "ViTPose 2D detector",
        "WHAM + MediaPipe",
        "Fallback reason",
        "Actual backend",
    ):
        assert label in window
    assert "VideoWhamSetupSummary" in downloader
    assert "GetWhamSetupSummary" in downloader
    assert "OfficialRunnerStatus" in downloader
    assert "Hmr2ImageFeaturesStatus" in downloader
    assert "VitPose2DStatus" in downloader


def test_wham_summary_classifies_optional_manual_and_incompatible_rows() -> None:
    """Manual/invalid optional stages must remain visible without blocking WHAM."""
    source = DOWNLOADER.read_text(encoding="utf-8-sig")
    summary_start = source.index("public static VideoWhamSetupSummary GetWhamSetupSummary")
    summary_end = source.index("private static void AddWhamSetupRequired", summary_start)
    summary = source[summary_start:summary_end]

    assert "ClassifyWhamSetupAssets(required, manual, incompatible)" in summary
    assert "ClassifyWhamSetupAssets(optional, manual, incompatible)" in summary
    assert "GetWhamSetupState(required)" in summary
    assert "VideoAssetStatus.Manual" in source
    assert "VideoAssetStatus.Incompatible" in source


def test_wham_image_feature_roles_are_explicitly_separate() -> None:
    """The HMR2 feature runner and ViTPose detector cannot share a role path."""
    source = DOWNLOADER.read_text(encoding="utf-8-sig")
    assert "IsHmr2ImageFeatureRunnerAsset" in source
    assert "IsWhamViTPose2DDetector" in source
    assert "IsHmr2ImageFeatureArchive" in source
    assert "Hmr2ImageFeatureRunner" in source
    assert "WhamViTPose2DDetector" in source
    assert "GetWhamSetupRoleStatus(required, VideoAssetRole.Hmr2Checkpoint" in source
    assert "GetWhamSetupRoleStatus(required, VideoAssetRole.WhamImageFeatureBackbone" in source
    assert 'paths["hmr2_image_feature"]' in source
    assert 'paths["vitpose_2d"]' in source


def test_manual_assets_are_not_automatic_download_candidates() -> None:
    """A documentation URL must not make a licensed/manual row downloadable."""
    source = DOWNLOADER.read_text(encoding="utf-8-sig")
    assert "IsAutomaticallyDownloadable" in source
    assert "CanDownload => !IsManualProvisioning && IsAutomaticallyDownloadable" in source
    assert "if (IsManualProvisioning) return false;" in source


def test_incompatible_companion_rows_route_to_replacement_setup() -> None:
    """Incompatible templates must offer a replacement path, not reinstall themselves.

    The bundled ViTPose hook and camera file are intentionally placeholders.  A
    regression here makes the Settings buttons look clickable while restoring
    the same placeholder and leaving the user with no next step.
    """
    window = (ROOT / "Editor" / "TexMotionWindow.cs").read_text(encoding="utf-8-sig")
    assert "HandleIncompatibleWhamAssetAction" in window
    assert "Choose compatible file" in window
    assert "Open manual setup guide" in window
    assert "Incompatible templates cannot be fixed by Download" in window


def _method_slice(source: str, start: str, end: str) -> str:
    start_index = source.index(start)
    return source[start_index : source.index(end, start_index)]


def test_compact_wham_surface_has_one_primary_action_and_grouped_setup() -> None:
    """The normal WHAM path stays scannable without hiding setup semantics."""
    window = (ROOT / "Editor" / "TexMotionWindow.cs").read_text(encoding="utf-8-sig")
    content = _method_slice(
        window,
        "private void DrawSimplifiedWhamSetupContent",
        "private void DrawWhamSetupStatus",
    )

    assert content.count("_primaryActionStyle") == 1
    assert 'TrLiteral("Required")' in content
    assert 'TrLiteral("Optional")' in content
    assert 'TrLiteral("Advanced")' in content or 'TrLiteral("Advanced paths")' in content


def test_compact_wham_buttons_do_not_use_fixed_width_layout() -> None:
    """Compact cards stack actions so localized labels remain readable."""
    window = (ROOT / "Editor" / "TexMotionWindow.cs").read_text(encoding="utf-8-sig")
    compact_helpers = "\n".join(
        (
            _method_slice(window, "private void DrawSimplifiedWhamSetupCard", "private void DrawSimplifiedWhamSetupContent"),
            _method_slice(window, "private void DrawSimplifiedWhamSetupContent", "private void DrawWhamSetupStatus"),
            _method_slice(window, "private void DrawWhamSetupGroup", "private void DrawWhamSetupDiagnostics"),
            _method_slice(window, "private void DrawCompactBackendAssetRow", "private void DrawVideoModelDownloadStatus"),
            _method_slice(window, "private void DrawWhamCompanionPathField", "private void DrawViTPoseFeatureExportControl"),
            _method_slice(window, "private void DrawViTPoseFeatureExportControl", "private static bool IsBundledViTPoseDefinitionPlaceholder"),
        )
    )

    assert "GUILayout.Width(" not in compact_helpers


def test_video_backend_card_links_to_settings_without_internal_paths() -> None:
    """Video Motion exposes a settings route, not cache or checkpoint paths."""
    window = (ROOT / "Editor" / "TexMotionWindow.cs").read_text(encoding="utf-8-sig")
    card = _method_slice(
        window,
        "private void DrawVideoBackendSummaryCard",
        "private static string GetVideoBackendDisplayName",
    )

    assert 'TrLiteral("Open Settings")' in card
    assert "_currentTab = Tab.Settings" in card
    assert "GetEffectiveRTMPoseModelPath" not in card
    assert "GetEffectiveVideoModelDirectory" not in card
    assert "Path.GetFileName" not in card
    assert "Path." not in card
    assert "GUILayout.Width(" not in card
