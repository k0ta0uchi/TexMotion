using System;
using System.Collections.Generic;

namespace TexMotion.Editor
{
    /// <summary>
    /// Small, editor-safe localization facade. English is always the fallback.
    ///
    /// The public constants are the canonical keys used by new call sites. The
    /// legacy literal map is intentionally generated from the English catalog so
    /// existing TrLiteral("English text") callers keep working while the catalog
    /// can evolve without making English display text part of the key contract.
    /// </summary>
    public static class TexMotionLocalization
    {
        // Existing keys are kept byte-for-byte compatible with the first
        // localization pass. Do not rename these values: serialized/editor
        // callers may already use them.
        public const string Settings = "settings";
        public const string VideoMotion = "video_motion";
        public const string ModelDownloader = "model_downloader";
        public const string Environment = "environment";
        public const string Status = "status";
        public const string Navigation = "navigation";
        public const string Language = "language";
        public const string English = "english";
        public const string Japanese = "japanese";

        // Shared/header/navigation keys.
        public const string HeaderTitle = "ui.header.title";
        public const string HeaderSubtitle = "ui.header.subtitle";
        public const string MotionGenerator = "ui.motion_generator";
        public const string Generator = "ui.generator";
        public const string Library = "ui.library";
        public const string SettingsLabel = "ui.settings_label";
        public const string TargetAvatar = "ui.target_avatar";
        public const string AvatarGameObject = "ui.avatar_game_object";
        public const string PleaseSelectAvatar = "ui.please_select_avatar";
        public const string MotionPrompt = "ui.motion_prompt";
        public const string PromptDescription = "ui.prompt_description";
        public const string PresetPunch = "ui.preset.punch";
        public const string PresetWave = "ui.preset.wave";
        public const string PresetDance = "ui.preset.dance";
        public const string PresetStretch = "ui.preset.stretch";
        public const string HandFaceAssistance = "ui.hand_face_assistance";
        public const string HandPose = "ui.hand_pose";
        public const string FaceEmotion = "ui.face_emotion";
        public const string EmotionIntensity = "ui.emotion_intensity";
        public const string GenerationQuality = "ui.generation_quality";
        public const string DurationFrames = "ui.duration_frames";
        public const string DiffusionSteps = "ui.diffusion_steps";
        public const string TextCfgScale = "ui.text_cfg_scale";
        public const string RandomizeSeed = "ui.randomize_seed";
        public const string Seed = "ui.seed";
        public const string GenerateMotionPreview = "ui.generate_motion_preview";
        public const string GeneratingMotion = "ui.generating_motion";
        public const string ApproxDuration = "ui.approx_duration";

        // Video Motion keys.
        public const string VideoSourceFile = "ui.video.source_file";
        public const string VideoAssetClip = "ui.video.asset_clip";
        public const string BrowseVideo = "ui.video.browse";
        public const string UseTestMotion = "ui.video.use_test_motion";
        public const string NoVideoSelected = "ui.video.no_video_selected";
        public const string VideoSource = "ui.video.source";
        public const string SourceKinematicSynthetic = "ui.video.source_kinematic_synthetic";
        public const string ExtractionSettingsVideoInfo = "ui.video.extraction_settings";
        public const string TargetFps = "ui.video.target_fps";
        public const string VideoTrimmingSeconds = "ui.video.trimming_seconds";
        public const string StartTimeSeconds = "ui.video.start_time";
        public const string EndTimeFull = "ui.video.end_time";
        public const string TrimmedSegment = "ui.video.trimmed_segment";
        public const string FullVideoDuration = "ui.video.full_duration";
        public const string ExtractionQualityProcessing = "ui.video.extraction_quality";
        public const string InPlaceRoot = "ui.video.in_place_root";
        public const string TemporalSmoothing = "ui.video.temporal_smoothing";
        public const string FootLocking = "ui.video.foot_locking";
        public const string MinConfidenceCutoff = "ui.video.min_confidence";
        public const string ExtractMotionFromVideo = "ui.video.extract_motion";
        public const string CancelExtraction = "ui.video.cancel_extraction";
        public const string ExtractingHumanoidMotion = "ui.video.extracting_humanoid";
        public const string ExtractionInProgress = "ui.video.extraction_in_progress";
        public const string ExtractedMotionMetrics = "ui.video.extracted_metrics";
        public const string ExtractionComplete = "ui.video.extraction_complete";
        public const string VideoError = "ui.video.error";
        public const string ExtractionCompleteInitializing = "ui.video.extraction_complete_initializing";
        public const string ExtractionCompleteFallback = "ui.video.extraction_complete_fallback";
        public const string ExtractedFrames = "ui.video.extracted_frames";
        public const string InitializingPosePipeline = "ui.video.initializing_pose_pipeline";
        public const string StartingPoseExtractor = "ui.video.starting_pose_extractor";
        public const string PoseEstimationBackend = "ui.video.pose_backend";
        public const string ChoosePoseBackend = "ui.video.choose_pose_backend";
        public const string BackendSettingsHint = "ui.video.backend_settings_hint";
        public const string Checkpoint = "ui.video.checkpoint";
        public const string NoCheckpointSelected = "ui.video.no_checkpoint";
        public const string RtmposeDescription = "ui.video.rtmpose_description";
        public const string MediaPipeDescription = "ui.video.mediapipe_description";
        public const string AutomaticBackendDescription = "ui.video.automatic_backend_description";
        public const string ModelsReady = "ui.video.models_ready";
        public const string ModelsMissing = "ui.video.models_missing";
        public const string GoToSettingsToDownload = "ui.video.go_to_settings_download";
        public const string PythonChecking = "ui.video.python_checking";
        public const string PythonDependenciesInstalling = "ui.video.python_dependencies_installing";
        public const string PythonDependenciesReady = "ui.video.python_dependencies_ready";
        public const string PythonDependenciesMissing = "ui.video.python_dependencies_missing";
        public const string InstallDependencies = "ui.video.install_dependencies";
        public const string Recheck = "ui.video.recheck";
        public const string VideoExtractionDisabled = "ui.video.extraction_disabled";
        public const string PythonEnvironmentSetupFailed = "ui.video.python_environment_setup_failed";
        public const string VideoModelDownloadFailed = "ui.video.model_download_failed";
        public const string VideoExtractionFailed = "ui.video.extraction_failed";
        public const string VideoExtractionCancelled = "ui.video.extraction_cancelled";
        public const string ExtractorScriptMissing = "ui.video.extractor_script_missing";
        public const string NoWorkingPythonRuntime = "ui.video.no_working_python_runtime";
        public const string SelectedPythonRuntimeFailed = "ui.video.selected_python_runtime_failed";
        public const string MissingVideoDependencies = "ui.video.missing_dependencies";
        public const string FailedToStartPythonProcess = "ui.video.failed_to_start_python";
        public const string ExtractionCancelled = "ui.video.extraction_cancelled_error";
        public const string ExtractionFailedExitCode = "ui.video.extraction_failed_exit";
        public const string ExtractionFailedStage = "ui.video.extraction_failed_stage";
        public const string ExtractionReportedError = "ui.video.extraction_reported_error";
        public const string OutputMotionFileMissing = "ui.video.output_motion_file_missing";
        public const string RuntimeDependenciesMissing = "ui.video.runtime_dependencies_missing";
        public const string NoPythonRuntimeDetected = "ui.video.no_python_runtime";
        public const string PythonExecutablePathEmpty = "ui.video.python_executable_empty";
        public const string PythonFileNotFound = "ui.video.python_file_not_found";
        public const string PythonProcessLaunchFailed = "ui.video.python_process_launch_failed";
        public const string PythonProbeTimedOut = "ui.video.python_probe_timed_out";
        public const string PythonProcessExited = "ui.video.python_process_exited";
        public const string MotionJsonNotFound = "ui.video.motion_json_not_found";
        public const string JsonContentEmpty = "ui.video.json_content_empty";
        public const string MotionJsonDeserializeFailed = "ui.video.motion_json_deserialize_failed";
        public const string RequirementsPythonMissing = "ui.video.requirements_python_missing";
        public const string PipProcessLaunchFailed = "ui.video.pip_process_launch_failed";
        public const string PythonUnavailableSummary = "ui.video.python_unavailable_summary";
        public const string PythonSummary = "ui.video.python_summary";
        public const string VirtualEnvironmentSuffix = "ui.video.virtual_environment_suffix";
        public const string DependenciesReadySummary = "ui.video.dependencies_ready_summary";
        public const string MissingPackagesSummary = "ui.video.missing_packages_summary";
        public const string NotDetected = "ui.video.not_detected";
        public const string Unknown = "ui.video.unknown";

        // Model/settings keys.
        public const string ReloadEngine = "ui.settings.reload_engine";
        public const string DownloadModelsHuggingFace = "ui.settings.download_models_hugging_face";
        public const string CancelDownload = "ui.settings.cancel_download";
        public const string AutoOpenTimelineEditor = "ui.settings.auto_open_timeline";
        public const string LaunchMotionTimelineEditor = "ui.settings.launch_timeline";
        public const string DownloadDirectory = "ui.settings.download_directory";
        public const string Download = "ui.settings.download";
        public const string Redownload = "ui.settings.redownload";
        public const string Use = "ui.settings.use";
        public const string Source = "ui.settings.source";
        public const string Clear = "ui.settings.clear";
        public const string Browse = "ui.settings.browse";
        public const string Cancel = "ui.settings.cancel";
        public const string Ready = "ui.settings.ready";
        public const string Complete = "ui.settings.complete";
        public const string Missing = "ui.settings.missing";
        public const string Error = "ui.settings.error";
        public const string OpenSettings = "ui.settings.open";
        public const string DownloadingFile = "ui.download.downloading_file";
        public const string SkippedExistingFile = "ui.download.skipped_existing";
        public const string AllModelsReady = "ui.download.all_models_ready";
        public const string DownloadCancelled = "ui.download.cancelled";
        public const string DownloadFailed = "ui.download.failed";
        public const string ReadyModel = "ui.download.ready_model";
        public const string ReadyModelWithAdapter = "ui.download.ready_model_adapter";
        public const string AllVideoModelsReady = "ui.download.all_video_models_ready";
        public const string InstallingWhamAdapter = "ui.download.installing_wham_adapter";
        public const string WhamAdapterReady = "ui.download.wham_adapter_ready";
        public const string InstallingWhamManifest = "ui.download.installing_wham_manifest";
        public const string WhamManifestReady = "ui.download.wham_manifest_ready";
        public const string InstallingWhamCamera = "ui.download.installing_wham_camera";
        public const string WhamCameraReady = "ui.download.wham_camera_ready";
        public const string WhamContractReady = "ui.download.wham_contract_ready";
        public const string WhamContractIncomplete = "ui.download.wham_contract_incomplete";
        public const string ManualWhamAsset = "ui.download.manual_wham_asset";
        public const string DownloadedFileTooSmall = "ui.download.file_too_small";
        public const string DownloadingModel = "ui.download.downloading_model";
        public const string DownloadingModelProgress = "ui.download.downloading_model_progress";
        public const string BundledWhamAdapterMissing = "ui.download.bundled_wham_adapter_missing";
        public const string WhamAdapterDestinationEmpty = "ui.download.wham_adapter_destination_empty";
        public const string BundledWhamManifestMissing = "ui.download.bundled_wham_manifest_missing";
        public const string WhamManifestDestinationEmpty = "ui.download.wham_manifest_destination_empty";
        public const string BundledWhamCameraMissing = "ui.download.bundled_wham_camera_missing";
        public const string WhamCameraDestinationEmpty = "ui.download.wham_camera_destination_empty";
        public const string VideoModelDestinationEmpty = "ui.download.video_model_destination_empty";

        // Built-in Video2Motion catalog names/descriptions. The model catalog
        // stores English display text for compatibility; these stable keys let
        // TrLiteral(model.DisplayName/Description) translate those rows too.
        public const string ModelRtmposeMName = "ui.model.rtmpose_m.name";
        public const string ModelRtmposeSName = "ui.model.rtmpose_s.name";
        public const string ModelRtmposeLName = "ui.model.rtmpose_l.name";
        public const string ModelRtmposeXName = "ui.model.rtmpose_x.name";
        public const string ModelRtmposeWholeBodyName = "ui.model.rtmpose_wholebody.name";
        public const string ModelRtmposeHandName = "ui.model.rtmpose_hand.name";
        public const string ModelDwposeName = "ui.model.dwpose.name";
        public const string ModelMediapipeLiteName = "ui.model.mediapipe_lite.name";
        public const string ModelMediapipeFullName = "ui.model.mediapipe_full.name";
        public const string ModelMediapipeHeavyName = "ui.model.mediapipe_heavy.name";
        public const string ModelWhamName = "ui.model.wham.name";
        public const string ModelHmr2Name = "ui.model.hmr2.name";
        public const string ModelHybrikName = "ui.model.hybrik.name";
        public const string ModelWhamBodyName = "ui.model.wham_body.name";
        public const string ModelWhamFeatureName = "ui.model.wham_feature.name";
        public const string ModelWhamCameraName = "ui.model.wham_camera.name";
        public const string ModelWhamDpvoName = "ui.model.wham_dpvo.name";
        public const string ModelWhamAdapterName = "ui.model.wham_adapter.name";
        public const string ModelWhamManifestName = "ui.model.wham_manifest.name";
        public const string ModelRtmposeMDescription = "ui.model.rtmpose_m.description";
        public const string ModelRtmposeSDescription = "ui.model.rtmpose_s.description";
        public const string ModelRtmposeLDescription = "ui.model.rtmpose_l.description";
        public const string ModelRtmposeXDescription = "ui.model.rtmpose_x.description";
        public const string ModelRtmposeWholeBodyDescription = "ui.model.rtmpose_wholebody.description";
        public const string ModelRtmposeHandDescription = "ui.model.rtmpose_hand.description";
        public const string ModelDwposeDescription = "ui.model.dwpose.description";
        public const string ModelMediapipeLiteDescription = "ui.model.mediapipe_lite.description";
        public const string ModelMediapipeFullDescription = "ui.model.mediapipe_full.description";
        public const string ModelMediapipeHeavyDescription = "ui.model.mediapipe_heavy.description";
        public const string ModelWhamDescription = "ui.model.wham.description";
        public const string ModelHmr2Description = "ui.model.hmr2.description";
        public const string ModelHybrikDescription = "ui.model.hybrik.description";
        public const string ModelWhamBodyDescription = "ui.model.wham_body.description";
        public const string ModelWhamFeatureDescription = "ui.model.wham_feature.description";
        public const string ModelWhamCameraDescription = "ui.model.wham_camera.description";
        public const string ModelWhamDpvoDescription = "ui.model.wham_dpvo.description";
        public const string ModelWhamAdapterDescription = "ui.model.wham_adapter.description";
        public const string ModelWhamManifestDescription = "ui.model.wham_manifest.description";

        // Python environment/provisioning keys.
        public const string UvUnavailable = "ui.python.uv_unavailable";
        public const string UvSummary = "ui.python.uv_summary";
        public const string InstallUvOrUseExistingRuntime = "ui.python.install_uv_or_existing_runtime";
        public const string InstallUvFromDocsOrSelectRuntime = "ui.python.install_uv_from_docs_or_select_runtime";
        public const string InstallUv = "ui.python.install_uv";
        public const string DetectUv = "ui.python.detect_uv";
        public const string UvDetected = "ui.python.uv_detected";
        public const string UvNotDetected = "ui.python.uv_not_detected";
        public const string UvInstallerWindowsOnly = "ui.python.uv_windows_only";
        public const string PowerShellNotFound = "ui.python.powershell_not_found";
        public const string RunningUvInstaller = "ui.python.running_uv_installer";
        public const string InstallingUvWithOfficial = "ui.python.installing_uv_official";
        public const string VerifyingUv = "ui.python.verifying_uv";
        public const string UvInstallFailed = "ui.python.uv_install_failed";
        public const string UvExecutableMissingAfterInstall = "ui.python.uv_executable_missing";
        public const string UvInstalledReady = "ui.python.uv_installed_ready";
        public const string UvInstallationCancelled = "ui.python.uv_installation_cancelled";
        public const string ExistingPythonEnvironment = "ui.python.existing_environment";
        public const string ExistingEnvironmentLog = "ui.python.existing_environment_log";
        public const string UsingUv = "ui.python.using_uv";
        public const string CreatingLocalVenv = "ui.python.creating_local_venv";
        public const string EnvironmentCreateFailed = "ui.python.environment_create_failed";
        public const string RequirementsMissing = "ui.python.requirements_missing";
        public const string PyTorchRequirementsMissing = "ui.python.pytorch_requirements_missing";
        public const string OptionalPyTorchRequirementsMissing = "ui.python.optional_pytorch_requirements_missing";
        public const string InstallingRequirements = "ui.python.installing_requirements";
        public const string FinishedInstalling = "ui.python.finished_installing";
        public const string InstallingRequirementsFrom = "ui.python.installing_requirements_from";
        public const string LocalEnvironmentReadyLog = "ui.python.local_environment_ready_log";
        public const string UvCacheDefault = "ui.python.uv_cache_default";
        public const string CapturedUvStdout = "ui.python.captured_uv_stdout";
        public const string CapturedUvStderr = "ui.python.captured_uv_stderr";
        public const string StartingProcess = "ui.python.starting_process";
        public const string ProcessExited = "ui.python.process_exited";
        public const string UvCacheUnavailable = "ui.python.uv_cache_unavailable";
        public const string UvCacheUnavailableDetailed = "ui.python.uv_cache_unavailable_detailed";
        public const string UsingProjectUvCache = "ui.python.using_project_uv_cache";
        public const string UvCachePreparationFailed = "ui.python.uv_cache_preparation_failed";
        public const string UvCacheRetained = "ui.python.uv_cache_retained";
        public const string OfflineInstallFailed = "ui.python.offline_install_failed";
        public const string DependencyInstallFailed = "ui.python.dependency_install_failed";
        public const string EnvironmentReady = "ui.python.environment_ready";
        public const string EnvironmentVerificationFailed = "ui.python.environment_verification_failed";
        public const string UvPathLookupFailed = "ui.python.uv_path_lookup_failed";
        public const string UvVersionFailed = "ui.python.uv_version_failed";

        // Timeline keys. These are also used by the timeline worker's literal
        // calls; keeping them here lets that window remain ownership-isolated.
        public const string TimelineEditor = "ui.timeline.editor";
        public const string OpenTexMotionStudio = "ui.timeline.open_studio";
        public const string TimelinePoseEditor = "ui.timeline.pose_editor";
        public const string TimelineEmptyState = "ui.timeline.empty_state";
        public const string TimelineGettingStarted = "ui.timeline.getting_started";
        public const string TimelineStepOne = "ui.timeline.step_one";
        public const string TimelineStepTwo = "ui.timeline.step_two";
        public const string TimelineStepThree = "ui.timeline.step_three";
        public const string Clip = "ui.timeline.clip";
        public const string Avatar = "ui.timeline.avatar";
        public const string FirstFrameHome = "ui.timeline.first_frame";
        public const string PreviousFrameLeft = "ui.timeline.previous_frame";
        public const string Play = "ui.timeline.play";
        public const string Pause = "ui.timeline.pause";
        public const string TogglePlaybackSpace = "ui.timeline.toggle_playback";
        public const string NextFrameRight = "ui.timeline.next_frame";
        public const string LastFrameEnd = "ui.timeline.last_frame";
        public const string LoopPlayback = "ui.timeline.loop";
        public const string Speed = "ui.timeline.speed";
        public const string DecreaseSpeed = "ui.timeline.decrease_speed";
        public const string ResetSpeed = "ui.timeline.reset_speed";
        public const string IncreaseSpeed = "ui.timeline.increase_speed";
        public const string ModifiedCount = "ui.timeline.modified_count";
        public const string Clean = "ui.timeline.clean";
        public const string Undo = "ui.timeline.undo";
        public const string Redo = "ui.timeline.redo";
        public const string RevertAll = "ui.timeline.revert_all";
        public const string RevertAllTitle = "ui.timeline.revert_all_title";
        public const string RevertAllMessage = "ui.timeline.revert_all_message";
        public const string Revert = "ui.timeline.revert";
        public const string SaveAnim = "ui.timeline.save_anim";
        public const string ApplyToAvatar = "ui.timeline.apply_avatar";
        public const string ViewportVideo = "ui.timeline.viewport_video";
        public const string MissingAvatar = "ui.timeline.missing_avatar";
        public const string LoadingVideo = "ui.timeline.loading_video";
        public const string Frame = "ui.timeline.frame";
        public const string Time = "ui.timeline.time";
        public const string PreviousFrameButton = "ui.timeline.previous_frame_button";
        public const string NextFrameButton = "ui.timeline.next_frame_button";
        public const string FramePoseTools = "ui.timeline.frame_pose_tools";
        public const string Copy = "ui.timeline.copy";
        public const string CopyTooltip = "ui.timeline.copy_tooltip";
        public const string Paste = "ui.timeline.paste";
        public const string PasteTooltip = "ui.timeline.paste_tooltip";
        public const string Mirror = "ui.timeline.mirror";
        public const string MirrorTooltip = "ui.timeline.mirror_tooltip";
        public const string Smooth = "ui.timeline.smooth";
        public const string SmoothTooltip = "ui.timeline.smooth_tooltip";
        public const string ResetFrame = "ui.timeline.reset_frame";
        public const string ResetFrameTooltip = "ui.timeline.reset_frame_tooltip";
        public const string TPose = "ui.timeline.t_pose";
        public const string TPoseTooltip = "ui.timeline.t_pose_tooltip";
        public const string RootPosition = "ui.timeline.root_position";
        public const string Active = "ui.timeline.active";
        public const string Select = "ui.timeline.select";
        public const string XLeftRight = "ui.timeline.x_left_right";
        public const string YHeight = "ui.timeline.y_height";
        public const string ZForwardBack = "ui.timeline.z_forward_back";
        public const string Pitch = "ui.timeline.pitch";
        public const string Yaw = "ui.timeline.yaw";
        public const string Roll = "ui.timeline.roll";
        public const string OcclusionTools = "ui.timeline.occlusion_tools";
        public const string Ambiguity = "ui.timeline.ambiguity";
        public const string TrackingConfidence = "ui.timeline.tracking_confidence";
        public const string ConfidentPoseHint = "ui.timeline.confident_pose_hint";
        public const string SwapLegCrossing = "ui.timeline.swap_leg_crossing";
        public const string SwapLegTooltip = "ui.timeline.swap_leg_tooltip";
        public const string ArmBehind = "ui.timeline.arm_behind";
        public const string ArmFront = "ui.timeline.arm_front";
        public const string PreviousModifiedFrame = "ui.timeline.previous_modified_frame";
        public const string NextModifiedFrame = "ui.timeline.next_modified_frame";
        public const string Zoom = "ui.timeline.zoom";
        public const string LiveHandFaceAssistance = "ui.timeline.live_hand_face_assistance";
        public const string LiveHandFaceAssistanceHint = "ui.timeline.live_hand_face_assistance_hint";

        // Dialog/log keys used outside the two large editor windows.
        public const string ExportUnityPackage = "ui.dialog.export_unitypackage";
        public const string ExportingPackage = "ui.dialog.exporting_package";
        public const string ExportComplete = "ui.dialog.export_complete";
        public const string ModularAvatarInstalled = "ui.dialog.modular_avatar_installed";
        public const string ModularAvatarInstalledMessage = "ui.dialog.modular_avatar_installed_message";
        public const string InstallationFailed = "ui.dialog.installation_failed";
        public const string InstallationFailedMessage = "ui.dialog.installation_failed_message";
        public const string Ok = "ui.dialog.ok";
        public const string MenuMotionNotAdded = "ui.dialog.menu_motion_not_added";
        public const string BaseAnimationLayersMissing = "ui.dialog.base_animation_layers_missing";
        public const string BaseAnimationLayerIndexMissing = "ui.dialog.base_animation_layer_index_missing";
        public const string EnumMemberMissing = "ui.dialog.enum_member_missing";
        public const string ControlTypeMissing = "ui.dialog.control_type_missing";
        public const string PlayableLayerControlMissing = "ui.dialog.playable_layer_control_missing";
        public const string TrackingControlMissing = "ui.dialog.tracking_control_missing";
        public const string ParameterDriverMissing = "ui.dialog.parameter_driver_missing";
        public const string ParameterDriverSerializeFailed = "ui.dialog.parameter_driver_serialize_failed";
        public const string VideoDataValidationWarning = "ui.dialog.video_data_validation_warning";
        public const string ProcessTerminationWarning = "ui.dialog.process_termination_warning";

        // Runtime validation/engine messages are catalogued here so editor
        // call sites can translate them without making the Runtime assembly
        // depend on the Editor assembly.
        public const string FrameCountInvalid = "runtime.validation.frame_count_invalid";
        public const string JointCountInvalid = "runtime.validation.joint_count_invalid";
        public const string RootPositionsLengthInvalid = "runtime.validation.root_positions_length_invalid";
        public const string LocalRotationsDimensionsInvalid = "runtime.validation.local_rotations_invalid";
        public const string TimestampsLengthInvalid = "runtime.validation.timestamps_length_invalid";
        public const string ConfidencesLengthInvalid = "runtime.validation.confidences_length_invalid";
        public const string UnknownKimodoLoadError = "runtime.kimodo.unknown_load_error";
        public const string KimodoModelNotLoaded = "runtime.kimodo.model_not_loaded";
        public const string KimodoGenerationFailed = "runtime.kimodo.generation_failed";

        private static readonly Dictionary<string, string> EnglishText = new Dictionary<string, string>(StringComparer.Ordinal)
        {
            { Settings, "Settings & Models" },
            { VideoMotion, "Video Motion" },
            { ModelDownloader, "Model Downloader" },
            { Environment, "Environment" },
            { Status, "Status" },
            { Navigation, "Navigation" },
            { Language, "Language" },
            { English, "English" },
            { Japanese, "Japanese" },

            { HeaderTitle, "✨ TexMotion Studio" },
            { HeaderSubtitle, "AI-Powered Text & Video-to-Motion for VRChat Avatars" },
            { MotionGenerator, "Motion Generator" },
            { Generator, "Generator" },
            { Library, "Library" },
            { SettingsLabel, "Settings" },
            { TargetAvatar, "Target Avatar" },
            { AvatarGameObject, "Avatar GameObject" },
            { PleaseSelectAvatar, "Please select your VRChat Avatar from the Hierarchy." },
            { MotionPrompt, "Motion Prompt" },
            { PromptDescription, "Describe the animation in natural language:" },
            { PresetPunch, "🥊 Punch" },
            { PresetWave, "👋 Wave" },
            { PresetDance, "🕺 Dance" },
            { PresetStretch, "🧘 Stretch" },
            { HandFaceAssistance, "Hand & Face Assistance" },
            { HandPose, "Hand Pose" },
            { FaceEmotion, "Face Emotion" },
            { EmotionIntensity, "Emotion Intensity" },
            { GenerationQuality, "Generation Quality" },
            { DurationFrames, "Duration (Frames)" },
            { DiffusionSteps, "Diffusion Steps" },
            { TextCfgScale, "Text CFG Scale" },
            { RandomizeSeed, "Randomize Seed" },
            { Seed, "Seed" },
            { GenerateMotionPreview, "Generate Motion Preview" },
            { GeneratingMotion, "Generating Motion..." },
            { ApproxDuration, "Approx. Duration: {0:F1} seconds" },

            { VideoSourceFile, "Video Source File" },
            { VideoAssetClip, "Video Asset / Clip" },
            { BrowseVideo, "Browse Video..." },
            { UseTestMotion, "Use Test Motion" },
            { NoVideoSelected, "No video file selected. You can select an MP4 video or test with a synthetic walking motion." },
            { VideoSource, "Source" },
            { SourceKinematicSynthetic, "Kinematic Synthetic Generator" },
            { ExtractionSettingsVideoInfo, "Extraction Settings & Video Info" },
            { TargetFps, "Target FPS" },
            { VideoTrimmingSeconds, "Video Trimming (Seconds)" },
            { StartTimeSeconds, "Start Time (s)" },
            { EndTimeFull, "End Time (0=full)" },
            { TrimmedSegment, "Trimmed Segment: {0:F2}s to {1:F2}s ({2:F2}s, ~{3} frames)" },
            { FullVideoDuration, "Trimming: Full video duration will be processed." },
            { ExtractionQualityProcessing, "Extraction Quality & Processing" },
            { InPlaceRoot, "In-Place Root (Keep hips centered)" },
            { TemporalSmoothing, "Temporal Smoothing (Savitzky-Golay)" },
            { FootLocking, "Foot Locking & Floor Snapping" },
            { MinConfidenceCutoff, "Min Confidence Cutoff" },
            { ExtractMotionFromVideo, "Extract Motion from Video" },
            { CancelExtraction, "Cancel Extraction" },
            { ExtractingHumanoidMotion, "Extracting 3D Humanoid Motion..." },
            { ExtractionInProgress, "Extraction in Progress" },
            { ExtractedMotionMetrics, "Extracted Motion Metrics" },
            { ExtractionComplete, "Extraction complete!" },
            { VideoError, "[Video Error] {0}" },
            { ExtractionCompleteInitializing, "Extraction complete! Initializing 3D viewport..." },
            { ExtractionCompleteFallback, "Extraction complete with fallback ({0})." },
            { ExtractedFrames, "Extracted {0} frames successfully!" },
            { InitializingPosePipeline, "Initializing pose extraction pipeline..." },
            { StartingPoseExtractor, "Starting video pose extractor..." },
            { PoseEstimationBackend, "Pose Estimation Backend" },
            { ChoosePoseBackend, "Choose the pose backend for this extraction." },
            { BackendSettingsHint, "Choose the backend here; configure model paths, adapters, and device in Settings → Video2Motion." },
            { Checkpoint, "Checkpoint" },
            { NoCheckpointSelected, "No checkpoint selected" },
            { RtmposeDescription, "RTMPose 2D ONNX  •  {0}" },
            { MediaPipeDescription, "MediaPipe Pose Landmarker  •  Complexity {0}" },
            { AutomaticBackendDescription, "Automatic selection uses local RTMPose/MediaPipe assets when available." },
            { ModelsReady, "Models Ready" },
            { ModelsMissing, "Models Missing" },
            { GoToSettingsToDownload, "Go to Settings to Download" },
            { PythonChecking, "Checking Python environment..." },
            { PythonDependenciesInstalling, "Installing Python dependencies via pip..." },
            { PythonDependenciesReady, "● Video Pose Dependencies Ready ({0})" },
            { PythonDependenciesMissing, "Python Dependencies Missing" },
            { InstallDependencies, "Install Dependencies" },
            { Recheck, "Re-check" },
            { VideoExtractionDisabled, "Video extraction is disabled until the selected Python runtime has NumPy. {0}" },
            { PythonEnvironmentSetupFailed, "Video Motion environment setup failed." },
            { VideoModelDownloadFailed, "Video2Motion model download failed." },
            { VideoExtractionFailed, "Video extraction failed: {0}" },
            { VideoExtractionCancelled, "Video motion extraction cancelled." },
            { ExtractorScriptMissing, "video_pose_extractor.py script could not be located in package or Assets. Ensure Editor/Video/video_pose_extractor.py exists." },
            { NoWorkingPythonRuntime, "No working Python runtime found on system. Please install Python 3.10+ (with mediapipe, opencv-python, and numpy) or specify the Python path in TexMotion Settings.\nDetails: {0}" },
            { SelectedPythonRuntimeFailed, "The selected Python runtime could not be started: {0}" },
            { MissingVideoDependencies, "Video extraction cannot start because {0} is missing from {1}. Open Settings → Video Motion Python Environment and create/update the project venv, or install the Video Motion requirements for this interpreter." },
            { FailedToStartPythonProcess, "Failed to start Python process: {0}" },
            { ExtractionCancelled, "Video motion extraction was canceled." },
            { ExtractionFailedExitCode, "Video pose extraction failed with exit code {0}.\nDiagnostics:\n{1}" },
            { ExtractionFailedStage, "Video pose extraction failed during stage '{0}': {1}\nDiagnostics:\n{2}" },
            { ExtractionReportedError, "Video pose extraction reported error: {0}\nDiagnostics:\n{1}" },
            { OutputMotionFileMissing, "Expected output motion JSON file not found at: {0}.\nDiagnostics:\n{1}" },
            { RuntimeDependenciesMissing, "Python runtime was found, but required video dependencies (mediapipe, opencv-python, numpy) are missing. Run requirements installation to enable full pose extraction." },
            { NoPythonRuntimeDetected, "No Python runtime detected on system. Please install Python 3.10+ (with mediapipe, opencv-python, and numpy)." },
            { PythonExecutablePathEmpty, "Python executable path is null or empty." },
            { PythonFileNotFound, "File not found: {0}" },
            { PythonProcessLaunchFailed, "Failed to launch Python process." },
            { PythonProbeTimedOut, "Python probe execution timed out." },
            { PythonProcessExited, "Python process exited with code {0}: {1}" },
            { MotionJsonNotFound, "Motion JSON file not found: {0}" },
            { JsonContentEmpty, "JSON content cannot be null or empty." },
            { MotionJsonDeserializeFailed, "Failed to deserialize motion JSON into MotionDataJsonDto. Ensure valid JSON payload." },
            { RequirementsPythonMissing, "Cannot install requirements: No Python executable detected." },
            { PipProcessLaunchFailed, "Failed to launch pip process with {0}" },
            { PythonUnavailableSummary, "Python Unavailable: {0}" },
            { PythonSummary, "Python {0} ({1})" },
            { VirtualEnvironmentSuffix, " [VirtualEnv]" },
            { DependenciesReadySummary, " - All dependencies ready (MediaPipe + RTMPose)" },
            { MissingPackagesSummary, " - Missing packages: {0}" },
            { NotDetected, "Not detected" },
            { Unknown, "Unknown" },

            { ReloadEngine, "Reload Engine" },
            { DownloadModelsHuggingFace, "Download Models from Hugging Face" },
            { CancelDownload, "Cancel Download" },
            { AutoOpenTimelineEditor, "Auto Open Timeline Editor" },
            { LaunchMotionTimelineEditor, "Launch Motion Timeline Editor" },
            { DownloadDirectory, "Download Directory" },
            { Download, "Download" },
            { Redownload, "Redownload" },
            { Use, "Use" },
            { Source, "Source" },
            { Clear, "Clear" },
            { Browse, "Browse..." },
            { Cancel, "Cancel" },
            { Ready, "Ready" },
            { Complete, "Complete" },
            { Missing, "Missing" },
            { Error, "Error" },
            { OpenSettings, "Open Settings" },
            { DownloadingFile, "({0}/{1}) Downloading {2}: {3} ({4:F0}%)" },
            { SkippedExistingFile, "[Skipped existing] {0} ({1}/{2})" },
            { AllModelsReady, "All models downloaded and ready!" },
            { DownloadCancelled, "Download canceled by user." },
            { DownloadFailed, "Failed to download {0}: {1} (URL: {2})" },
            { ReadyModel, "Ready: {0}" },
            { ReadyModelWithAdapter, "Ready: {0} + TexMotion adapter" },
            { AllVideoModelsReady, "All Video2Motion models and downloadable WHAM companions are ready." },
            { InstallingWhamAdapter, "Installing WHAM adapter..." },
            { WhamAdapterReady, "WHAM adapter bridge ready" },
            { InstallingWhamManifest, "Installing WHAM asset manifest..." },
            { WhamManifestReady, "WHAM asset manifest ready" },
            { InstallingWhamCamera, "Installing WHAM camera calibration template..." },
            { WhamCameraReady, "Camera template ready (replace intrinsics before DPVO/SLAM)" },
            { WhamContractReady, "WHAM full-parity offline asset contract is ready." },
            { WhamContractIncomplete, "WHAM full-parity offline asset contract is incomplete; missing local stages: {0}. Downloadable ViTPose/DPVO stages can be fetched above; SMPL/SMPL-X data and calibrated camera/pose exports must be supplied explicitly." },
            { ManualWhamAsset, "{0} is a separately licensed/manual WHAM asset. Place the user-owned file at {1} (or one of its declared aliases); TexMotion does not download it automatically." },
            { DownloadedFileTooSmall, "Downloaded file is unexpectedly small ({0} bytes)." },
            { DownloadingModel, "Downloading {0}..." },
            { DownloadingModelProgress, "Downloading {0} ({1:F0}%)" },
            { BundledWhamAdapterMissing, "The bundled WHAM adapter source is missing. Reimport the TexMotion package before downloading WHAM." },
            { WhamAdapterDestinationEmpty, "WHAM adapter destination is empty." },
            { BundledWhamManifestMissing, "The bundled WHAM asset manifest is missing. Reimport the TexMotion package before installing it." },
            { WhamManifestDestinationEmpty, "WHAM asset manifest destination is empty." },
            { BundledWhamCameraMissing, "The bundled WHAM camera template is missing. Reimport the TexMotion package before installing it." },
            { WhamCameraDestinationEmpty, "WHAM camera template destination is empty." },
            { VideoModelDestinationEmpty, "Video2Motion model destination is empty." },

            { ModelRtmposeMName, "RTMPose-M (COCO 2D)" },
            { ModelRtmposeSName, "RTMPose-S (COCO 2D)" },
            { ModelRtmposeLName, "RTMPose-L (COCO 2D)" },
            { ModelRtmposeXName, "RTMPose-X (COCO 2D)" },
            { ModelRtmposeWholeBodyName, "RTMPose-M WholeBody" },
            { ModelRtmposeHandName, "RTMPose-M Hand" },
            { ModelDwposeName, "DWPose-L (WholeBody ONNX)" },
            { ModelMediapipeLiteName, "MediaPipe Pose Landmarker Lite" },
            { ModelMediapipeFullName, "MediaPipe Pose Landmarker Full" },
            { ModelMediapipeHeavyName, "MediaPipe Pose Landmarker Heavy" },
            { ModelWhamName, "WHAM (World-grounded 3D)" },
            { ModelHmr2Name, "HMR2 / 4D-Humans" },
            { ModelHybrikName, "HybrIK-X (SMPL-X)" },
            { ModelWhamBodyName, "WHAM SMPL/SMPL-X Body Model" },
            { ModelWhamFeatureName, "WHAM Image Feature Backbone (ViTPose)" },
            { ModelWhamCameraName, "WHAM Camera Calibration/Config" },
            { ModelWhamDpvoName, "WHAM DPVO Camera-Motion Assets" },
            { ModelWhamAdapterName, "WHAM Adapter Bridge" },
            { ModelWhamManifestName, "WHAM Offline Asset Manifest" },
            { ModelRtmposeMDescription, "Recommended CPU/DirectML 2D keypoint model for the hybrid pipeline." },
            { ModelRtmposeSDescription, "Smaller and faster RTMPose variant for low-power machines." },
            { ModelRtmposeLDescription, "Higher-capacity 2D detector for difficult or small subjects." },
            { ModelRtmposeXDescription, "Largest COCO 2D variant; highest memory and latency cost." },
            { ModelRtmposeWholeBodyDescription, "WholeBody graph for body, hands, and face auxiliary observations." },
            { ModelRtmposeHandDescription, "Hand-focused graph; download for custom hand-pose adapters." },
            { ModelDwposeDescription, "Optional DWPose/RTMW graph; use with a compatible ONNX adapter." },
            { ModelMediapipeLiteDescription, "Fastest 33-joint MediaPipe task model (complexity 0)." },
            { ModelMediapipeFullDescription, "Balanced 33-joint MediaPipe task model (complexity 1)." },
            { ModelMediapipeHeavyDescription, "Highest-quality 33-joint MediaPipe task model (complexity 2)." },
            { ModelWhamDescription, "Optional temporal 3D WHAM checkpoint. TexMotion includes a native temporal WHAM core and bundled adapter bridge for the local path; an official/external WHAM implementation is optional for the full research preprocessing and camera-motion stages." },
            { ModelHmr2Description, "Optional HMR2 image-conditioned body-model checkpoint. Select HMR2 and install the PyTorch Quality profile to use it." },
            { ModelHybrikDescription, "Optional analytical/neural HybrIK-X checkpoint. Google Drive can require a browser confirmation; use Source for manual provisioning if needed." },
            { ModelWhamBodyDescription, "Required for full WHAM parity; manually provision a licensed SMPL-X or compatible SMPL body model." },
            { ModelWhamFeatureDescription, "Optional downloadable ViTPose-Huge checkpoint. A compatible HMR2/ViTPose runner or verified frame-feature archive is required; raw weights are never treated as features and unverified local descriptors are rejected." },
            { ModelWhamCameraDescription, "Installs a starter camera.yaml template. Replace its intrinsics for the input video; calibration alone does not contain motion." },
            { ModelWhamDpvoDescription, "Optional downloadable DPVO checkpoint. A local DPVO runtime must export camera poses; the native adapter can consume those pose archives directly." },
            { ModelWhamAdapterDescription, "Bundled TexMotion adapter bridge; an external full implementation may be selected explicitly." },
            { ModelWhamManifestDescription, "Contract metadata for deterministic local WHAM stage resolution." },

            { UvUnavailable, "uv unavailable: {0}" },
            { UvSummary, "uv {0} ({1})" },
            { InstallUvOrUseExistingRuntime, "Install uv or choose an existing Python runtime." },
            { InstallUvFromDocsOrSelectRuntime, "Install uv from https://docs.astral.sh/uv/ or select an existing Python executable." },
            { InstallUv, "Install uv" },
            { DetectUv, "Detect uv" },
            { UvDetected, "✓ uv detected" },
            { UvNotDetected, "⚠ uv not detected" },
            { UvInstallerWindowsOnly, "The automatic uv installer is currently available for Windows only." },
            { PowerShellNotFound, "PowerShell was not found. Install uv manually from https://docs.astral.sh/uv/getting-started/installation/." },
            { RunningUvInstaller, "Running the official uv installer..." },
            { InstallingUvWithOfficial, "[TexMotion] Installing uv with Astral's official installer: {0}" },
            { VerifyingUv, "Verifying the installed uv executable..." },
            { UvInstallFailed, "The uv installer failed (exit code {0})." },
            { UvExecutableMissingAfterInstall, "uv installation finished, but no executable was found. Restart Unity or choose uv.exe with Browse." },
            { UvInstalledReady, "uv is installed and ready." },
            { UvInstallationCancelled, "uv installation was cancelled." },
            { ExistingPythonEnvironment, "Existing local Python environment found." },
            { ExistingEnvironmentLog, "[TexMotion] Existing environment: {0}" },
            { UsingUv, "Using {0}" },
            { CreatingLocalVenv, "Creating project-local .venv with uv..." },
            { EnvironmentCreateFailed, "uv could not create the local environment." },
            { RequirementsMissing, "Lightweight requirements.txt was not found. Expected Editor/Video/requirements.txt." },
            { PyTorchRequirementsMissing, "PyTorch profile requirements file was not found." },
            { OptionalPyTorchRequirementsMissing, "[TexMotion] Optional PyTorch profile file not found; installing lightweight profile only." },
            { InstallingRequirements, "Installing {0}..." },
            { FinishedInstalling, "[TexMotion] Finished installing: {0}" },
            { InstallingRequirementsFrom, "[TexMotion] Installing requirements from: {0}" },
            { LocalEnvironmentReadyLog, "[TexMotion] Local environment ready: {0}" },
            { UvCacheDefault, "[TexMotion] UV_CACHE_DIR is not set; using uv's default cache." },
            { CapturedUvStdout, "[TexMotion] Captured uv stdout:\n{0}" },
            { CapturedUvStderr, "[TexMotion] Captured uv stderr:\n{0}" },
            { StartingProcess, "[TexMotion] Starting: {0} {1}" },
            { ProcessExited, "[TexMotion] Process exited with code {0}." },
            { UvCacheUnavailable, "[TexMotion] UV_CACHE_DIR is unavailable: {0}; falling back to project-local cache." },
            { UvCacheUnavailableDetailed, "[TexMotion] UV_CACHE_DIR is unavailable: {0} ({1})." },
            { UsingProjectUvCache, "[TexMotion] Using project-local uv cache: {0}" },
            { UvCachePreparationFailed, "[TexMotion] Could not prepare a writable uv cache (fallback: {0})." },
            { UvCacheRetained, "[TexMotion] Could not prepare a writable uv cache (fallback: {0}). Retaining UV_CACHE_DIR: {1}" },
            { OfflineInstallFailed, "Offline install failed; required wheels are not present in uv's cache." },
            { DependencyInstallFailed, "Dependency installation failed." },
            { EnvironmentReady, "Local Python environment is ready." },
            { EnvironmentVerificationFailed, "The environment was created, but its Python executable could not be verified." },
            { UvPathLookupFailed, "uv was not found on PATH or in the standard user install locations." },
            { UvVersionFailed, "uv --version failed." },

            { TimelineEditor, "Timeline Editor" },
            { OpenTexMotionStudio, "✨ Open TexMotion Studio" },
            { TimelinePoseEditor, "TexMotion Timeline & Pose Editor" },
            { TimelineEmptyState, "No motion is currently loaded in the timeline.\nGenerate a motion clip from Text or extract poses from Video in TexMotion Studio to start visual keyframe and bone posing here." },
            { TimelineGettingStarted, "HOW TO GET STARTED" },
            { TimelineStepOne, "1. Open TexMotion Studio and generate or extract a motion clip." },
            { TimelineStepTwo, "2. Preview the animation and click \"Edit in Timeline\"." },
            { TimelineStepThree, "3. Directly manipulate 3D joints, scrub frames, and save as .anim." },
            { Clip, "🎬 Clip:" },
            { Avatar, "Avatar:" },
            { FirstFrameHome, "First Frame (Home)" },
            { PreviousFrameLeft, "Previous Frame (Left Arrow)" },
            { Play, "▶ Play" },
            { Pause, "⏸ Pause" },
            { TogglePlaybackSpace, "Toggle Playback (Space)" },
            { NextFrameRight, "Next Frame (Right Arrow)" },
            { LastFrameEnd, "Last Frame (End)" },
            { LoopPlayback, "Loop Playback" },
            { Speed, "Speed:" },
            { DecreaseSpeed, "Decrease Speed (-0.1x)" },
            { ResetSpeed, "Click to reset to 1.0x" },
            { IncreaseSpeed, "Increase Speed (+0.1x)" },
            { ModifiedCount, "● {0} mod" },
            { Clean, "✓ Clean" },
            { Undo, "↩ Undo" },
            { Redo, "↪ Redo" },
            { RevertAll, "↩ Revert All" },
            { RevertAllTitle, "Revert All Frames" },
            { RevertAllMessage, "Discard all frame modifications and revert back to original generated motion?" },
            { Revert, "Revert" },
            { SaveAnim, "💾 Save .anim" },
            { ApplyToAvatar, "✨ Apply to Avatar" },
            { ViewportVideo, "3D Viewport & Video" },
            { MissingAvatar, "⚠️ No Avatar Assigned\nPlease select a Humanoid Avatar in the toolbar above." },
            { LoadingVideo, "Loading Video Stream..." },
            { Frame, "📍 Frame {0} / {1}" },
            { Time, "Time: {0:F2}s" },
            { PreviousFrameButton, "◀ Prev Frame" },
            { NextFrameButton, "Next Frame ▶" },
            { FramePoseTools, "🛠️ Frame Pose Tools" },
            { Copy, "📋 Copy" },
            { CopyTooltip, "Copy current frame pose to clipboard" },
            { Paste, "📄 Paste" },
            { PasteTooltip, "Paste clipboard pose onto this frame" },
            { Mirror, "🔄 Mirror" },
            { MirrorTooltip, "Mirror pose across left and right limbs" },
            { Smooth, "⚖️ Smooth" },
            { SmoothTooltip, "Interpolate with neighbor frames (smoothing)" },
            { ResetFrame, "↩️ Reset Frame" },
            { ResetFrameTooltip, "Revert this frame to original generated pose" },
            { TPose, "🧘 T-Pose" },
            { TPoseTooltip, "Set this frame to neutral T-Pose" },
            { RootPosition, "Root Position (Offset):" },
            { Active, "🎯 Active" },
            { Select, "🎯 Select" },
            { XLeftRight, "X (Left/Right)" },
            { YHeight, "Y (Height)" },
            { ZForwardBack, "Z (Fwd/Back)" },
            { Pitch, "Pitch (X)" },
            { Yaw, "Yaw (Y)" },
            { Roll, "Roll (Z)" },
            { OcclusionTools, "🔮 Occlusion & Ambiguity Tools (オクルージョン補正)" },
            { Ambiguity, "⚠ Ambiguity: {0}\nFrames: {1} - {2} (Confidence: {3:P0})" },
            { TrackingConfidence, "Tracking Ambiguity" },
            { ConfidentPoseHint, "Pose tracking confident on this frame. Use tools below to manually resolve occlusion orders." },
            { SwapLegCrossing, "🔄 Swap Leg Crossing (脚前後反転)" },
            { SwapLegTooltip, "Reverses anterior/posterior leg crossing order (Swaps which leg is in front)" },
            { ArmBehind, "✋ {0}-Arm Behind" },
            { ArmFront, "✋ {0}-Arm Front" },
            { PreviousModifiedFrame, "Previous Modified Frame" },
            { NextModifiedFrame, "Next Modified Frame" },
            { Zoom, "Zoom:" },
            { LiveHandFaceAssistance, "🖐 Hand & Face Assistance (Live Preview)" },
            { LiveHandFaceAssistanceHint, "Changes apply to the avatar preview and exported AnimationClip." },

            { ExportUnityPackage, "Export TexMotion UnityPackage" },
            { ExportingPackage, "[TexMotion] Exporting package to: {0}" },
            { ExportComplete, "[TexMotion] Export complete!" },
            { ModularAvatarInstalled, "Modular Avatar Installed" },
            { ModularAvatarInstalledMessage, "Modular Avatar was successfully installed into your project! You can now apply animations non-destructively." },
            { InstallationFailed, "Installation Failed" },
            { InstallationFailedMessage, "Could not install Modular Avatar automatically:\n{0}\n\nPlease install Modular Avatar via VCC (VRChat Creator Companion)." },
            { Ok, "OK" },
            { MenuMotionNotAdded, "[TexMotion] '{0}' was not added to menu '{1}' (full and replacement was declined). Animator setup completed; run again after freeing a slot." },
            { BaseAnimationLayersMissing, "[TexMotion] baseAnimationLayers property not found on descriptor." },
            { BaseAnimationLayerIndexMissing, "[TexMotion] baseAnimationLayers does not have layer index {0}." },
            { EnumMemberMissing, "[TexMotion] Enum member '{0}' not found on {1}; using raw value {2}." },
            { ControlTypeMissing, "[TexMotion] ControlType member '{0}' not found." },
            { PlayableLayerControlMissing, "[TexMotion] VRCPlayableLayerControl not found; Action layer weight will not be controlled." },
            { TrackingControlMissing, "[TexMotion] VRCAnimatorTrackingControl not found; tracking will not be switched." },
            { ParameterDriverMissing, "[TexMotion] VRCAvatarParameterDriver not found; one-shot parameter cannot be reset." },
            { ParameterDriverSerializeFailed, "[TexMotion] Could not serialize ParameterDriver.parameters via SerializedObject." },
            { VideoDataValidationWarning, "[TexMotion VideoJobRunner] Deserialized VideoMotionData warning: {0}" },
            { ProcessTerminationWarning, "[TexMotion VideoJobRunner] Notice while terminating process: {0}" },

            { FrameCountInvalid, "Frame count must be greater than zero." },
            { JointCountInvalid, "Joint count must be {0}, but was {1}." },
            { RootPositionsLengthInvalid, "Root positions array length ({0}) does not match Frames ({1})." },
            { LocalRotationsDimensionsInvalid, "Local rotations dimensions do not match [Frames, JointCount]." },
            { TimestampsLengthInvalid, "Timestamps array length does not match Frames." },
            { ConfidencesLengthInvalid, "Confidences array length does not match Frames." },
            { UnknownKimodoLoadError, "Unknown error while loading Kimodo model." },
            { KimodoModelNotLoaded, "Kimodo model is not loaded. Call LoadModel first." },
            { KimodoGenerationFailed, "Kimodo generation failed: {0}" }
        };

        private static readonly Dictionary<string, string> JapaneseText = new Dictionary<string, string>(StringComparer.Ordinal)
        {
            { Settings, "設定とモデル" },
            { VideoMotion, "動画モーション" },
            { ModelDownloader, "モデルダウンローダー" },
            { Environment, "環境" },
            { Status, "ステータス" },
            { Navigation, "ナビゲーション" },
            { Language, "言語" },
            { English, "英語" },
            { Japanese, "日本語" },

            // Video asset setup guide (legacy literal keys kept local so the
            // guide can be translated without changing serialized catalog data).
            { "Video Asset Guide", "Videoアセットガイド" },
            { "This catalog row has no manual setup guide.", "このカタログ行には手動設定ガイドがありません。" },
            { "Catalog id: ", "カタログID: " },
            { "  •  Role: ", "  •  役割: " },
            { "License / manual reason", "ライセンス／手動設定の理由" },
            { "Official download / registration", "公式ダウンロード／登録" },
            { "Open official download / registration page", "公式ダウンロード／登録ページを開く" },
            { "Recommended install directory", "推奨インストール先" },
            { "Required file or directory", "必要なファイルまたはフォルダー" },
            { "Browse setting", "参照設定" },
            { "Preflight steps", "事前チェック手順" },
            { "What to do next", "次に行うこと" },
            { "Open TexMotion Settings", "TexMotion設定を開く" },
            { "Copy install directory", "インストール先をコピー" },
            { "Close", "閉じる" },
            { "<not specified>", "未指定" },
            { "Create ViTPose Feature Archive", "ViTPose特徴量アーカイブを作成" },
            { "One button samples the selected video with the same FPS/trim settings, runs the configured local ViTPose runner, and registers vitpose_features.npz for WHAM.", "1つのボタンで選択動画を同じFPS／トリム設定でサンプリングし、設定済みのローカルViTPoseランナーを実行して、vitpose_features.npzをWHAMへ登録します。" },
            { "Source video: {0}", "ソース動画: {0}" },
            { "not selected", "未選択" },
            { "Choose Video...", "動画を選択..." },
            { "Select source video for ViTPose features", "ViTPose特徴量用のソース動画を選択" },
            { "Output archive: {0}", "出力アーカイブ: {0}" },
            { "Sampling: {0:F0} FPS, trim {1:F2}s–{2}", "サンプリング: {0:F0} FPS、トリム {1:F2}秒–{2}" },
            { "end", "末尾" },
            { "Select a ViTPose checkpoint and compatible model-definition hook above. A raw .pth file alone cannot run inference.", "上でViTPoseチェックポイントと互換モデル定義フックを選択してください。.pthファイルだけでは推論できません。" },
            { "The bundled ViTPose model-definition file is a contract template, not an architecture. Replace it with a compatible create_model/model factory before extracting.", "同梱のViTPoseモデル定義ファイルは契約テンプレートであり、アーキテクチャ本体ではありません。抽出前に互換性のあるcreate_model／modelファクトリーへ置き換えてください。" },
            { "Extracting ViTPose features...", "ViTPose特徴量を抽出中..." },
            { "Extract Features (one click)", "特徴量を抽出（ワンクリック）" },
            { "ViTPose feature archive ready: {0}", "ViTPose特徴量アーカイブを準備しました: {0}" },
            { "Preparing ViTPose feature extraction...", "ViTPose特徴量抽出を準備中..." },
            { "ViTPose feature extraction cancelled.", "ViTPose特徴量抽出をキャンセルしました。" },
            { "ViTPose feature extraction failed.", "ViTPose特徴量抽出に失敗しました。" },
            { "Cancelling ViTPose feature extraction...", "ViTPose特徴量抽出をキャンセル中..." },
            { "ViTPose Features (WHAM Quality)", "ViTPose 特徴量 (WHAM高精度用)" },
            { "Extracting ViTPose features in background ({0:P0})...", "バックグラウンドでViTPose特徴量を抽出中 ({0:P0})..." },
            { "ViTPose Features Cached", "ViTPose特徴量キャッシュ済み" },
            { "CPU environment detected. Auto-extraction of ViTPose features (650M params) is skipped to avoid high load.", "CPU環境が検出されました。高負荷を避けるため、ViTPose特徴量（約6.5億パラメータ）の自動抽出はスキップされました。" },
            { "Extract Features in Background (GPU)", "バックグラウンドで特徴量を抽出 (GPU)" },
            { "ViTPose checkpoint or model definition is missing. Configure it in Settings.", "ViTPoseチェックポイントまたはモデル定義が未設定です。Settingsで設定してください。" },
            { "Background extraction in progress. You can continue other work.", "バックグラウンドで特徴量を抽出中です。他の操作を継続できます。" },
            { "Cached archive: {0}", "キャッシュ済みアーカイブ: {0}" },
            { "GPU (CUDA) detected. Starting automatic background feature extraction...", "GPU (CUDA) を検出しました。ViTPose特徴量のバックグラウンド自動抽出を開始します..." },
            { "Processing", "処理中" },
            { "GPU is performing deep temporal sequence optimization across all video frames. This calculation is actively running on your GPU and takes several minutes for high-accuracy motion. Please keep this window open.", "GPU上で動画全フレームの時系列3D骨格最適化（WHAM）を実行中です。高精度なモーション生成のため数分〜十数分かかります。このままウィンドウを開いてお待ちください。" },
            { "Show Live Process Log", "リアルタイム処理ログを表示" },
            { "Live Process Output:", "リアルタイム処理ログ:" },
            { "Waiting for process output...", "プロセスの出力を待機中..." },

            // Compact WHAM setup/result card literals.  Keep these direct
            // mappings localizable even though they are intentionally not
            // serialized settings keys.
            { "Official WHAM", "公式WHAM" },
            { "Run WHAM preflight", "WHAMの動作確認を実行" },
            { "Open manual setup guide", "手動設定ガイドを開く" },
            { "Choose compatible file", "互換ファイルを選択" },
            { "Auto-configure from Video / Standard FOV", "動画解像度・標準画角から自動設定" },
            { "Choose compatible {0}", "互換性のある{0}を選択" },
            { "No compatible file was selected. Open Guide for the required format.", "互換ファイルが選択されていません。必要な形式はガイドで確認してください。" },
            { "Incompatible templates cannot be fixed by Download. Choose a compatible local file, then run Recheck.", "互換性エラーのテンプレートはダウンロードでは直りません。互換性のあるローカルファイルを選択してから再検査してください。" },
            { "Run Recheck to validate the selected file.", "選択したファイルを検証するには再検査を実行してください。" },
            { "Previous status: {0}. Run Recheck to validate the selected file.", "以前の状態: {0}。選択したファイルを検証するには再検査を実行してください。" },
            { "Selected {0}. {1}", "{0}を選択しました。{1}" },
            { "Download missing WHAM assets", "不足しているWHAMアセットをダウンロード" },
            { "Recheck", "再検査" },
            { "Required", "必須" },
            { "Optional", "任意" },
            { "Advanced", "詳細設定" },
            { "Diagnostic", "診断" },
            { "Actual backend: {0}", "実行バックエンド: {0}" },
            { "Requested backend: {0}", "要求バックエンド: {0}" },
            { "WHAM image-feature runner failed → MediaPipe fallback. {0}", "WHAM画像特徴ランナー失敗 → MediaPipeへフォールバック。{0}" },
            { "See the diagnostic rows below for the error code and next action.", "下の診断行でエラーコードと次の操作を確認してください。" },
            { "Fallback reason: {0}", "フォールバック理由: {0}" },
            { "Official WHAM runner: {0}", "公式WHAMランナー: {0}" },
            { "HMR2 image features: {0}", "HMR2画像特徴量: {0}" },
            { "ViTPose 2D detector: {0}", "ViTPose 2D検出器: {0}" },
            { "HMR2 image feature runner", "HMR2画像特徴ランナー" },
            { "Official WHAM temporal 3D runner.", "公式WHAM時系列3Dランナー。" },
            { "WHAM + MediaPipe (WHAM primary, MediaPipe auxiliary).", "WHAM + MediaPipe（WHAM主体、MediaPipe補助）。" },
            { "Preparing required WHAM downloads...", "必要なWHAMダウンロードを準備中..." },
            { "Preparing the official WHAM setup...", "公式WHAMセットアップを準備中..." },
            { "No automatic WHAM download is available. Open the manual setup guide for the remaining items.", "自動ダウンロードできるWHAMアセットがありません。残りの項目は手動設定ガイドを開いてください。" },
            { "WHAM setup: {0} ({1}/{2})", "WHAMセットアップ: {0}（{1}/{2}）" },
            { "Downloadable WHAM setup assets are ready. Complete any manual items, then run Preflight.", "ダウンロード可能なWHAMセットアップアセットは準備完了です。手動項目を完了してから事前チェックを実行してください。" },
            { "WHAM setup download cancelled.", "WHAMセットアップのダウンロードをキャンセルしました。" },
            { "WHAM setup download failed.", "WHAMセットアップのダウンロードに失敗しました。" },
            { "{0}  •  {1}", "{0}  •  {1}" },
            { "Setup unavailable", "セットアップを利用できません" },
            { "RTMPose 2D ONNX. Configure the model in Settings.", "RTMPose 2D ONNX。モデルは設定で指定してください。" },
            { "MediaPipe Pose Landmarker. Configure model quality in Settings.", "MediaPipe Pose Landmarker。モデル品質は設定で指定してください。" },
            { "Choose the backend here; configure model details, adapters, and device in Settings → Video2Motion.", "ここでバックエンドを選択し、モデル詳細、アダプター、デバイスは設定 → Video2Motionで指定してください。" },
            { "not reported", "未報告" },
            { " ({0})", "（{0}）" },
            { "WHAM preflight: {0}{1} | ready={2} | smoke frames={3}", "WHAM事前チェック: {0}{1} | 準備完了={2} | スモークフレーム={3}" },
            { "image-feature stage", "画像特徴量ステージ" },
            { "Image-feature fallback: {0} (consumed by WHAM integrator: {1}).", "画像特徴量フォールバック: {0}（WHAMインテグレーターで使用: {1}）。" },
            { "Camera-motion provenance: source={0}, runner={1}, DPVO verified={2}", "カメラモーション出所: ソース={0}、ランナー={1}、DPVO検証済み={2}" },
            { "not available", "利用不可" },
            { "Missing optional files are reported and extraction continues with an explicit fallback; inference is never silently skipped.", "任意ファイルの不足を報告し、明示的なフォールバックで抽出を続けます。推論ステージを黙って省略しません。" },
            { "WHAM image-feature runner inactive; extraction continued with {0}.", "WHAM画像特徴ランナーは非アクティブです。{0}で抽出を続行しました。" },
            { "WHAM image-feature runner active.", "WHAM画像特徴ランナーはアクティブです。" },
            { "Runner error:", "ランナーエラー:" },
            { "WHAM per-video preprocessing: image features = ", "WHAM動画ごとの前処理: 画像特徴量 = " },
            { "WHAM local runner notes:", "WHAMローカルランナーの注記:" },
            { "Official WHAM Temporal 3D", "公式WHAM時系列3D" },
            { "WHAM + MediaPipe", "WHAM + MediaPipe" },
            { "{0} required item(s) need attention before official inference can run.", "公式推論を実行する前に必須項目{0}件の対応が必要です。" },
            { "Browse Folder", "フォルダーを参照" },
            { "Official WHAM: {0}", "公式WHAM: {0}" },
            { "HMR2 image feature runner: {0}", "HMR2画像特徴ランナー: {0}" },
            { "{0} optional stage(s) are unavailable; basic extraction remains possible.", "任意ステージ{0}件は利用できませんが、基本抽出は実行できます。" },
            { "Download or select the model for the backend chosen in Video Motion.", "Video Motionで選択したバックエンドのモデルをダウンロードまたは指定してください。" },
            { "Cancel download", "ダウンロードをキャンセル" },
            { "No manual guide is available for the selected WHAM state.", "選択したWHAM状態に利用できる手動ガイドはありません。" },
            { "Incompatible", "互換性なし" },
            { "ViTPose 2D Detector", "ViTPose 2D検出器" },
            { "ViTPose checkpoint used for official WHAM 2D preprocessing. It is separate from the HMR2 image-feature runner.", "公式WHAM 2D前処理で使うViTPoseチェックポイント。HMR2画像特徴ランナーとは別です。" },
            { "ViTPose Model Definition", "ViTPoseモデル定義" },
            { "Python model-definition hook required to construct a raw ViTPose state-dict checkpoint. The catalog installs a contract template; replace it with a compatible MMPose/Transformers factory.", "raw ViTPose state-dictチェックポイントを構築するPythonモデル定義フック。同梱テンプレートを互換MMPose/Transformersファクトリーへ置き換えてください。" },
            { "Select ViTPose model-definition module", "ViTPoseモデル定義モジュールを選択" },
            { "ViTPose Runner Config", "ViTPoseランナー設定" },
            { "Optional JSON/YAML/Python runner config. A config does not replace the model-definition implementation.", "任意のJSON/YAML/Pythonランナー設定。設定だけではモデル定義の実装を置き換えられません。" },
            { "Select ViTPose runner config", "ViTPoseランナー設定を選択" },
            { "Intrinsics configure projection; exported DPVO/SLAM poses provide world motion. If no pose archive is supplied, a local per-video optical-flow cache is generated.", "内部パラメーターは投影を設定し、出力済みDPVO/SLAMポーズがワールド移動を提供します。ポーズアーカイブがない場合は動画ごとのローカル光学フローキャッシュを生成します。" },
            { "DPVO weights can be used by a configured runner; pose exports can be consumed directly. Without a runner, TexMotion generates a conservative local optical-flow cache and labels it.", "DPVO重みは設定済みランナーで使用でき、ポーズ出力は直接利用できます。ランナーがない場合、TexMotionは保守的なローカル光学フローキャッシュを生成して明示します。" },
            { "Compatible ViTPose image feature runner", "互換ViTPose画像特徴ランナー" },
            { "Use only an explicitly selected compatible runtime; the normal WHAM path does not require a Python model definition.", "明示的に選択した互換ランタイムだけを使用します。通常のWHAM経路ではPythonモデル定義は不要です。" },
            { "Precomputed HMR2 image feature archive", "事前計算済みHMR2画像特徴量アーカイブ" },
            { "Use a frame-aligned archive only after video hash, FPS, trim, frame count, and feature dimension validation.", "動画ハッシュ、FPS、トリム、フレーム数、特徴量次元を検証して一致したアーカイブだけを使用します。" },
            { "Requested image-feature runner: {0}", "要求画像特徴ランナー: {0}" },
            { "MediaPipe fallback allowed: {0} | Require image features: {1}", "MediaPipeフォールバック許可: {0} | 画像特徴量を必須化: {1}" },
            { "Official HMR2 (recommended)", "公式HMR2（推奨）" },
            { "Compatible ViTPose/MMPose", "互換ViTPose/MMPose" },
            { "Precomputed HMR2 archive", "事前計算済みHMR2アーカイブ" },
            { "Image feature runner", "画像特徴ランナー" },
            { "Image-feature evidence: accepted {0}/{1} frame(s), dim={2}, status={3}", "画像特徴量の証跡: 採用 {0}/{1} フレーム、次元={2}、状態={3}" },
            { "Image-feature error code: {0}", "画像特徴量エラーコード: {0}" },
            { "Allow MediaPipe fallback", "MediaPipeフォールバックを許可" },
            { "Require image features", "画像特徴量を必須化" },
            { "Error code: {0}", "エラーコード: {0}" },
            { "Next action: {0}", "次の操作: {0}" },
            { "Official HMR2 image features are resolved automatically during WHAM extraction. This path does not require a Python model-definition file; run Runtime Preflight instead.", "公式HMR2画像特徴量はWHAM抽出時に自動解決されます。この経路ではPythonモデル定義ファイルは不要です。代わりにランタイム事前チェックを実行してください。" },
            { "Archive mode uses the selected frame-aligned HMR2 archive. Create a compatible archive in compatible_vitpose mode, then switch back and run Runtime Preflight.", "アーカイブモードでは選択したフレーム整列済みHMR2アーカイブを使用します。compatible_vitposeで互換アーカイブを作成してから戻り、ランタイム事前チェックを実行してください。" },
            { "HMR2 uses the TexMotion bridge automatically. Configure the official 4D-Humans runtime, checkpoint and licensed neutral SMPL body; HybrIK still requires its own local bridge.", "HMR2ではTexMotion同梱ブリッジを自動で使用します。公式4D-Humansランタイム、チェックポイント、ライセンス済み中立SMPLボディを設定してください。HybrIKは別途ローカルブリッジが必要です。" },
            { "TexMotion's bundled official HMR2 bridge is used automatically; browse only to override it.", "TexMotion同梱の公式HMR2ブリッジを自動で使用します。置き換える場合だけ参照してください。" },
            { "TexMotion's official HMR2 bridge is bundled. Set the official runtime, checkpoint, and licensed neutral SMPL body; the bridge does not require a user-authored .py file.", "TexMotionの公式HMR2ブリッジは同梱されています。公式ランタイム、チェックポイント、ライセンス済み中立SMPLボディを設定してください。ユーザーが.pyファイルを作成する必要はありません。" },
            { "ViTPose 2D detector", "ViTPose 2D検出器" },
            { "Choose a backend to see its local assets", "バックエンドを選ぶとローカルアセットが表示されます" },
            { "Choose the backend in Video Motion. Prepare its local assets here, then run extraction offline.", "Video Motionでバックエンドを選択し、ここでローカルアセットを準備してからオフライン抽出を実行します。" },
            { "Video2Motion Setup", "Video2Motionセットアップ" },
            { "Python environment", "Python環境" },
            { "The local runner is started only during extraction; CUDA is optional.", "ローカルランナーは抽出時だけ起動します。CUDAは任意です。" },
            { "WHAM temporal model", "WHAM時系列モデル" },
            { "Official WHAM temporal 3D checkpoint and adapter.", "公式WHAM時系列3Dチェックポイントとアダプター。" },
            { "HMR2 ViT image features and the initial SMPL estimate used by official WHAM.", "公式WHAMが使うHMR2 ViT画像特徴量と初期SMPL推定。" },
            { "2D keypoint detection for WHAM preprocessing. This is separate from HMR2 image features.", "WHAM前処理用の2D関節検出。HMR2画像特徴量とは別の役割です。" },
            { "MediaPipe auxiliary task", "MediaPipe補助タスク" },
            { "Visibility and 3D observations used only by WHAM + MediaPipe fusion.", "WHAM + MediaPipe融合だけで使う可視性と3D観測。" },
            { "Camera and research stages", "カメラ・研究ステージ" },
            { "These assets improve camera/world motion or preserve an offline cache. Basic WHAM extraction can start without them.", "これらはカメラ／ワールド移動やオフラインキャッシュを補います。基本的なWHAM抽出はなくても開始できます。" },
            { "DPVO camera motion", "DPVOカメラ移動" },
            { "World camera motion; without it the result stays camera-relative.", "ワールドカメラ移動。ない場合はカメラ相対の結果になります。" },
            { "Camera calibration", "カメラキャリブレーション" },
            { "Video-specific intrinsics for projection.", "動画ごとの投影内部パラメータ。" },
            { "Precomputed image feature archive", "事前計算済み画像特徴量アーカイブ" },
            { "Optional per-video cache for an offline feature stage.", "オフライン特徴量ステージ用の動画ごとの任意キャッシュ。" },
            { "SMPL/SMPL-X body model", "SMPL/SMPL-Xボディモデル" },
            { "Licensed body data can be supplied when a local runtime requires it.", "ローカルランタイムが必要とする場合は、ライセンス済みのボディデータを指定できます。" },
            { "Custom runtimes only", "カスタムランタイムのみ" },
            { "Most users do not need these fields. Use them only when a licensed runtime or external cache is outside the default model directory.", "通常は不要です。ライセンス済みランタイムや外部キャッシュを既定フォルダー外に置く場合だけ使用してください。" },
            { "Ready to extract offline.", "オフライン抽出の準備完了。" },
            { "Setup is required.", "セットアップが必要です。" },
            { "Download", "ダウンロード" },
            { "Guide", "ガイド" },
            { "Install environment", "環境をインストール" },
            { "Manual setup needed", "手動設定が必要" },
            { "Compatibility error", "互換性エラー" },
            { "Checking", "検査中" },
            { "Missing", "不足" },
            { "{0} required item(s) missing", "必須項目が{0}件不足" },

            { HeaderTitle, "✨ TexMotion Studio" },
            { HeaderSubtitle, "VRChatアバター向け AIテキスト・動画モーション" },
            { MotionGenerator, "モーション生成" },
            { Generator, "生成" },
            { Library, "ライブラリ" },
            { SettingsLabel, "設定" },
            { TargetAvatar, "対象アバター" },
            { AvatarGameObject, "アバター GameObject" },
            { PleaseSelectAvatar, "HierarchyからVRChatアバターを選択してください。" },
            { MotionPrompt, "モーションプロンプト" },
            { PromptDescription, "アニメーションを自然言語で説明してください:" },
            { PresetPunch, "🥊 パンチ" },
            { PresetWave, "👋 ウェーブ" },
            { PresetDance, "🕺 ダンス" },
            { PresetStretch, "🧘 ストレッチ" },
            { HandFaceAssistance, "手と表情の補助" },
            { HandPose, "手のポーズ" },
            { FaceEmotion, "表情" },
            { EmotionIntensity, "表情の強さ" },
            { GenerationQuality, "生成品質" },
            { DurationFrames, "長さ（フレーム）" },
            { DiffusionSteps, "拡散ステップ数" },
            { TextCfgScale, "テキスト CFG スケール" },
            { RandomizeSeed, "シードをランダム化" },
            { Seed, "シード" },
            { GenerateMotionPreview, "モーションプレビューを生成" },
            { GeneratingMotion, "モーションを生成中..." },
            { ApproxDuration, "おおよその長さ: {0:F1} 秒" },

            { VideoSourceFile, "動画ソースファイル" },
            { VideoAssetClip, "動画アセット / クリップ" },
            { BrowseVideo, "動画を参照..." },
            { UseTestMotion, "テストモーションを使用" },
            { NoVideoSelected, "動画ファイルが選択されていません。MP4動画を選択するか、合成歩行モーションを試せます。" },
            { VideoSource, "ソース" },
            { SourceKinematicSynthetic, "キネマティック合成ジェネレーター" },
            { ExtractionSettingsVideoInfo, "抽出設定と動画情報" },
            { TargetFps, "目標 FPS" },
            { VideoTrimmingSeconds, "動画トリミング（秒）" },
            { StartTimeSeconds, "開始時間（秒）" },
            { EndTimeFull, "終了時間（0=全体）" },
            { TrimmedSegment, "トリミング範囲: {0:F2}秒 ～ {1:F2}秒（{2:F2}秒、約{3}フレーム）" },
            { FullVideoDuration, "トリミング: 動画全体を処理します。" },
            { ExtractionQualityProcessing, "抽出品質と処理" },
            { InPlaceRoot, "インプレース（腰を中央に維持）" },
            { TemporalSmoothing, "時間方向の平滑化（Savitzky-Golay）" },
            { FootLocking, "足の固定と床へのスナップ" },
            { MinConfidenceCutoff, "最低信頼度" },
            { ExtractMotionFromVideo, "動画からモーションを抽出" },
            { CancelExtraction, "抽出をキャンセル" },
            { ExtractingHumanoidMotion, "3Dヒューマノイドモーションを抽出中..." },
            { ExtractionInProgress, "抽出中" },
            { ExtractedMotionMetrics, "抽出モーション指標" },
            { ExtractionComplete, "抽出完了!" },
            { VideoError, "[動画エラー] {0}" },
            { ExtractionCompleteInitializing, "抽出完了! 3Dビューポートを初期化中..." },
            { ExtractionCompleteFallback, "フォールバックで抽出完了（{0}）。" },
            { ExtractedFrames, "{0}フレームの抽出に成功しました!" },
            { InitializingPosePipeline, "姿勢抽出パイプラインを初期化中..." },
            { StartingPoseExtractor, "動画姿勢抽出を開始中..." },
            { PoseEstimationBackend, "姿勢推定バックエンド" },
            { ChoosePoseBackend, "この抽出で使用する姿勢バックエンドを選択します。" },
            { BackendSettingsHint, "ここでバックエンドを選択し、設定 → Video2Motionでモデルパス、アダプター、デバイスを設定します。" },
            { Checkpoint, "チェックポイント" },
            { NoCheckpointSelected, "チェックポイント未選択" },
            { RtmposeDescription, "RTMPose 2D ONNX  •  {0}" },
            { MediaPipeDescription, "MediaPipe Pose Landmarker  •  複雑度 {0}" },
            { AutomaticBackendDescription, "利用可能なローカルRTMPose/MediaPipeアセットを自動選択します。" },
            { ModelsReady, "モデル準備完了" },
            { ModelsMissing, "モデル不足" },
            { GoToSettingsToDownload, "設定でダウンロード" },
            { PythonChecking, "Python環境を確認中..." },
            { PythonDependenciesInstalling, "pipでPython依存関係をインストール中..." },
            { PythonDependenciesReady, "● 動画姿勢依存関係の準備完了（{0}）" },
            { PythonDependenciesMissing, "Python依存関係が不足しています" },
            { InstallDependencies, "依存関係をインストール" },
            { Recheck, "再確認" },
            { VideoExtractionDisabled, "選択したPython環境にNumPyが入るまで動画抽出は無効です。{0}" },
            { PythonEnvironmentSetupFailed, "Video Motion環境のセットアップに失敗しました。" },
            { VideoModelDownloadFailed, "Video2Motionモデルのダウンロードに失敗しました。" },
            { VideoExtractionFailed, "動画抽出に失敗しました: {0}" },
            { VideoExtractionCancelled, "動画モーションの抽出をキャンセルしました。" },
            { ExtractorScriptMissing, "video_pose_extractor.pyスクリプトがパッケージまたはAssetsに見つかりません。Editor/Video/video_pose_extractor.pyが存在することを確認してください。" },
            { NoWorkingPythonRuntime, "動作するPython環境が見つかりません。Python 3.10以降（mediapipe、opencv-python、numpy）をインストールするか、TexMotion設定でPythonパスを指定してください。\n詳細: {0}" },
            { SelectedPythonRuntimeFailed, "選択したPython環境を起動できませんでした: {0}" },
            { MissingVideoDependencies, "動画抽出を開始できません。不足している依存関係: {0}（{1}）。設定 → Video Motion Python Environmentでプロジェクトvenvを作成/更新するか、この環境にVideo Motionのrequirementsをインストールしてください。" },
            { FailedToStartPythonProcess, "Pythonプロセスを開始できませんでした: {0}" },
            { ExtractionCancelled, "動画モーションの抽出をキャンセルしました。" },
            { ExtractionFailedExitCode, "動画姿勢抽出が終了コード{0}で失敗しました。\n診断情報:\n{1}" },
            { ExtractionFailedStage, "ステージ'{0}'で動画姿勢抽出に失敗しました: {1}\n診断情報:\n{2}" },
            { ExtractionReportedError, "動画姿勢抽出でエラーが報告されました: {0}\n診断情報:\n{1}" },
            { OutputMotionFileMissing, "出力モーションJSONが見つかりません: {0}\n診断情報:\n{1}" },
            { RuntimeDependenciesMissing, "Python環境は見つかりましたが、必要な動画依存関係（mediapipe、opencv-python、numpy）が不足しています。完全な姿勢抽出を有効にするにはrequirementsをインストールしてください。" },
            { NoPythonRuntimeDetected, "Python環境が検出されません。Python 3.10以降（mediapipe、opencv-python、numpy）をインストールしてください。" },
            { PythonExecutablePathEmpty, "Python実行ファイルのパスがnullまたは空です。" },
            { PythonFileNotFound, "ファイルが見つかりません: {0}" },
            { PythonProcessLaunchFailed, "Pythonプロセスの起動に失敗しました。" },
            { PythonProbeTimedOut, "Pythonの検査がタイムアウトしました。" },
            { PythonProcessExited, "Pythonプロセスが終了コード{0}で終了しました: {1}" },
            { MotionJsonNotFound, "モーションJSONが見つかりません: {0}" },
            { JsonContentEmpty, "JSONの内容をnullまたは空にできません。" },
            { MotionJsonDeserializeFailed, "モーションJSONをMotionDataJsonDtoにデシリアライズできませんでした。有効なJSONか確認してください。" },
            { RequirementsPythonMissing, "requirementsをインストールできません。Python実行ファイルが検出されません。" },
            { PipProcessLaunchFailed, "pipプロセスを起動できませんでした: {0}" },
            { PythonUnavailableSummary, "Pythonを利用できません: {0}" },
            { PythonSummary, "Python {0}（{1}）" },
            { VirtualEnvironmentSuffix, " [VirtualEnv]" },
            { DependenciesReadySummary, " - すべての依存関係の準備完了（MediaPipe + RTMPose）" },
            { MissingPackagesSummary, " - 不足パッケージ: {0}" },
            { NotDetected, "未検出" },
            { Unknown, "不明" },

            { ReloadEngine, "エンジンを再読み込み" },
            { DownloadModelsHuggingFace, "Hugging Faceからモデルをダウンロード" },
            { CancelDownload, "ダウンロードをキャンセル" },
            { AutoOpenTimelineEditor, "タイムラインエディターを自動で開く" },
            { LaunchMotionTimelineEditor, "モーションタイムラインエディターを起動" },
            { DownloadDirectory, "ダウンロード先" },
            { Download, "ダウンロード" },
            { Redownload, "再ダウンロード" },
            { Use, "使用" },
            { Source, "ソース" },
            { Clear, "クリア" },
            { Browse, "参照..." },
            { Cancel, "キャンセル" },
            { Ready, "準備完了" },
            { Complete, "完了" },
            { Missing, "不足" },
            { Error, "エラー" },
            { OpenSettings, "設定を開く" },
            { DownloadingFile, "（{0}/{1}）{2}をダウンロード中: {3}（{4:F0}%）" },
            { SkippedExistingFile, "［既存のためスキップ］{0}（{1}/{2}）" },
            { AllModelsReady, "すべてのモデルをダウンロードして準備完了!" },
            { DownloadCancelled, "ユーザーによりダウンロードをキャンセルしました。" },
            { DownloadFailed, "{0}のダウンロードに失敗しました: {1}（URL: {2}）" },
            { ReadyModel, "準備完了: {0}" },
            { ReadyModelWithAdapter, "準備完了: {0} + TexMotionアダプター" },
            { AllVideoModelsReady, "すべてのVideo2Motionモデルとダウンロード可能なWHAM補助ファイルの準備が完了しました。" },
            { InstallingWhamAdapter, "WHAMアダプターをインストール中..." },
            { WhamAdapterReady, "WHAMアダプターブリッジ準備完了" },
            { InstallingWhamManifest, "WHAMアセットマニフェストをインストール中..." },
            { WhamManifestReady, "WHAMアセットマニフェスト準備完了" },
            { InstallingWhamCamera, "WHAMカメラキャリブレーションテンプレートをインストール中..." },
            { WhamCameraReady, "カメラテンプレート準備完了（DPVO/SLAMの前に内部パラメーターを置き換えてください）" },
            { WhamContractReady, "WHAM完全互換オフラインアセット契約の準備完了。" },
            { WhamContractIncomplete, "WHAM完全互換オフラインアセット契約は未完了です。不足ステージ: {0}。ダウンロード可能なViTPose/DPVOステージは上から取得できます。SMPL/SMPL-Xデータとキャリブレーション済みカメラ/ポーズ書き出しは明示的に用意してください。" },
            { ManualWhamAsset, "{0}は別途ライセンスが必要な手動WHAMアセットです。ユーザー所有のファイルを{1}（または宣言済みエイリアス）に配置してください。TexMotionは自動ダウンロードしません。" },
            { DownloadedFileTooSmall, "ダウンロードしたファイルが小さすぎます（{0}バイト）。" },
            { DownloadingModel, "{0}をダウンロード中..." },
            { DownloadingModelProgress, "{0}をダウンロード中（{1:F0}%）" },
            { BundledWhamAdapterMissing, "同梱のWHAMアダプターソースが見つかりません。WHAMをダウンロードする前にTexMotionパッケージを再インポートしてください。" },
            { WhamAdapterDestinationEmpty, "WHAMアダプターの保存先が空です。" },
            { BundledWhamManifestMissing, "同梱のWHAMアセットマニフェストが見つかりません。インストール前にTexMotionパッケージを再インポートしてください。" },
            { WhamManifestDestinationEmpty, "WHAMアセットマニフェストの保存先が空です。" },
            { BundledWhamCameraMissing, "同梱のWHAMカメラテンプレートが見つかりません。インストール前にTexMotionパッケージを再インポートしてください。" },
            { WhamCameraDestinationEmpty, "WHAMカメラテンプレートの保存先が空です。" },
            { VideoModelDestinationEmpty, "Video2Motionモデルの保存先が空です。" },

            { ModelRtmposeMName, "RTMPose-M（COCO 2D）" },
            { ModelRtmposeSName, "RTMPose-S（COCO 2D）" },
            { ModelRtmposeLName, "RTMPose-L（COCO 2D）" },
            { ModelRtmposeXName, "RTMPose-X（COCO 2D）" },
            { ModelRtmposeWholeBodyName, "RTMPose-M WholeBody" },
            { ModelRtmposeHandName, "RTMPose-M Hand" },
            { ModelDwposeName, "DWPose-L（WholeBody ONNX）" },
            { ModelMediapipeLiteName, "MediaPipe Pose Landmarker Lite" },
            { ModelMediapipeFullName, "MediaPipe Pose Landmarker Full" },
            { ModelMediapipeHeavyName, "MediaPipe Pose Landmarker Heavy" },
            { ModelWhamName, "WHAM（World-grounded 3D）" },
            { ModelHmr2Name, "HMR2 / 4D-Humans" },
            { ModelHybrikName, "HybrIK-X（SMPL-X）" },
            { ModelWhamBodyName, "WHAM SMPL/SMPL-Xボディモデル" },
            { ModelWhamFeatureName, "WHAM画像特徴バックボーン（ViTPose）" },
            { ModelWhamCameraName, "WHAMカメラキャリブレーション/設定" },
            { ModelWhamDpvoName, "WHAM DPVOカメラモーションアセット" },
            { ModelWhamAdapterName, "WHAMアダプターブリッジ" },
            { ModelWhamManifestName, "WHAMオフラインアセットマニフェスト" },
            { ModelRtmposeMDescription, "ハイブリッドパイプライン向けの推奨CPU/DirectML 2Dキーポイントモデル。" },
            { ModelRtmposeSDescription, "低スペック環境向けの小型・高速RTMPoseバリアント。" },
            { ModelRtmposeLDescription, "難しい被写体や小さな被写体向けの高容量2D検出器。" },
            { ModelRtmposeXDescription, "最大のCOCO 2Dバリアント。メモリと遅延のコストが最も高くなります。" },
            { ModelRtmposeWholeBodyDescription, "身体・手・顔の補助観測に対応するWholeBodyグラフ。" },
            { ModelRtmposeHandDescription, "手に特化したグラフ。カスタム手ポーズアダプター用にダウンロードします。" },
            { ModelDwposeDescription, "オプションのDWPose/RTMWグラフ。互換性のあるONNXアダプターと併用します。" },
            { ModelMediapipeLiteDescription, "最速の33ジョイントMediaPipeタスクモデル（複雑度0）。" },
            { ModelMediapipeFullDescription, "バランス型の33ジョイントMediaPipeタスクモデル（複雑度1）。" },
            { ModelMediapipeHeavyDescription, "最高品質の33ジョイントMediaPipeタスクモデル（複雑度2）。" },
            { ModelWhamDescription, "オプションの時間方向3D WHAMチェックポイント。TexMotionにはローカル用のネイティブ時間方向WHAMコアとアダプターブリッジが含まれます。完全な研究用前処理とカメラモーションステージでは公式/外部WHAM実装も選択できます。" },
            { ModelHmr2Description, "オプションのHMR2画像条件付きボディモデルチェックポイント。使用するにはHMR2を選択し、PyTorch Qualityプロファイルをインストールします。" },
            { ModelHybrikDescription, "オプションの解析/ニューラルHybrIK-Xチェックポイント。Google Driveではブラウザー確認が必要な場合があります。必要ならSourceから手動で用意してください。" },
            { ModelWhamBodyDescription, "WHAM完全互換に必要です。ライセンス済みのSMPL-Xまたは互換SMPLボディモデルを手動で用意してください。" },
            { ModelWhamFeatureDescription, "オプションでダウンロード可能なViTPose-Hugeチェックポイント。ネイティブアダプターはローカルのフレーム特徴アーカイブまたは抽出器を使用し、設定時にこのバックボーンを渡します。" },
            { ModelWhamCameraDescription, "camera.yamlの初期テンプレートをインストールします。入力動画に合わせて内部パラメーターを置き換えてください。キャリブレーションだけではモーションを取得できません。" },
            { ModelWhamDpvoDescription, "オプションでダウンロード可能なDPVOチェックポイント。ローカルDPVOランタイムでカメラ姿勢を書き出し、ネイティブアダプターがその姿勢アーカイブを読み込みます。" },
            { ModelWhamAdapterDescription, "同梱のTexMotionアダプターブリッジ。外部の完全実装も明示的に選択できます。" },
            { ModelWhamManifestDescription, "WHAMステージを決定的にローカル解決するための契約メタデータ。" },

            { UvUnavailable, "uvを利用できません: {0}" },
            { UvSummary, "uv {0}（{1}）" },
            { InstallUvOrUseExistingRuntime, "uvをインストールするか、既存のPython環境を選択してください。" },
            { InstallUvFromDocsOrSelectRuntime, "https://docs.astral.sh/uv/ からuvをインストールするか、既存のPython実行ファイルを選択してください。" },
            { InstallUv, "uvをインストール" },
            { DetectUv, "uvを検出" },
            { UvDetected, "✓ uvを検出しました" },
            { UvNotDetected, "⚠ uvが見つかりません" },
            { UvInstallerWindowsOnly, "uvの自動インストーラーは現在Windowsでのみ利用できます。" },
            { PowerShellNotFound, "PowerShellが見つかりません。https://docs.astral.sh/uv/getting-started/installation/ からuvを手動でインストールしてください。" },
            { RunningUvInstaller, "公式uvインストーラーを実行中..." },
            { InstallingUvWithOfficial, "[TexMotion] Astral公式インストーラーでuvをインストール中: {0}" },
            { VerifyingUv, "インストール済みuv実行ファイルを確認中..." },
            { UvInstallFailed, "uvインストーラーに失敗しました（終了コード {0}）。" },
            { UvExecutableMissingAfterInstall, "uvのインストールは完了しましたが、実行ファイルが見つかりません。Unityを再起動するか、Browseでuv.exeを選択してください。" },
            { UvInstalledReady, "uvのインストールと準備が完了しました。" },
            { UvInstallationCancelled, "uvのインストールをキャンセルしました。" },
            { ExistingPythonEnvironment, "既存のローカルPython環境が見つかりました。" },
            { ExistingEnvironmentLog, "[TexMotion] 既存の環境: {0}" },
            { UsingUv, "{0}を使用中" },
            { CreatingLocalVenv, "uvでプロジェクトローカル.venvを作成中..." },
            { EnvironmentCreateFailed, "uvでローカル環境を作成できませんでした。" },
            { RequirementsMissing, "Lightweight requirements.txtが見つかりません。Editor/Video/requirements.txtが必要です。" },
            { PyTorchRequirementsMissing, "PyTorchプロファイルのrequirementsファイルが見つかりません。" },
            { OptionalPyTorchRequirementsMissing, "[TexMotion] オプションのPyTorchプロファイルが見つかりません。Lightweightプロファイルのみをインストールします。" },
            { InstallingRequirements, "{0}をインストール中..." },
            { FinishedInstalling, "[TexMotion] インストール完了: {0}" },
            { InstallingRequirementsFrom, "[TexMotion] requirementsをインストール中: {0}" },
            { LocalEnvironmentReadyLog, "[TexMotion] ローカル環境の準備完了: {0}" },
            { UvCacheDefault, "[TexMotion] UV_CACHE_DIRが未設定のため、uvのデフォルトキャッシュを使用します。" },
            { CapturedUvStdout, "[TexMotion] uv標準出力:\n{0}" },
            { CapturedUvStderr, "[TexMotion] uv標準エラー出力:\n{0}" },
            { StartingProcess, "[TexMotion] 開始: {0} {1}" },
            { ProcessExited, "[TexMotion] プロセス終了コード: {0}。" },
            { UvCacheUnavailable, "[TexMotion] UV_CACHE_DIRを利用できません: {0}。プロジェクトローカルキャッシュに切り替えます。" },
            { UvCacheUnavailableDetailed, "[TexMotion] UV_CACHE_DIRを利用できません: {0}（{1}）。" },
            { UsingProjectUvCache, "[TexMotion] プロジェクトローカルuvキャッシュを使用: {0}" },
            { UvCachePreparationFailed, "[TexMotion] 書き込み可能なuvキャッシュを準備できませんでした（フォールバック: {0}）。" },
            { UvCacheRetained, "[TexMotion] 書き込み可能なuvキャッシュを準備できませんでした（フォールバック: {0}）。UV_CACHE_DIRを維持します: {1}" },
            { OfflineInstallFailed, "オフラインインストールに失敗しました。必要なwheelがuvのキャッシュにありません。" },
            { DependencyInstallFailed, "依存関係のインストールに失敗しました。" },
            { EnvironmentReady, "ローカルPython環境の準備が完了しました。" },
            { EnvironmentVerificationFailed, "環境は作成されましたが、Python実行ファイルを確認できませんでした。" },
            { UvPathLookupFailed, "PATHまたは標準ユーザーインストール先にuvが見つかりません。" },
            { UvVersionFailed, "uv --versionに失敗しました。" },

            { TimelineEditor, "タイムラインエディター" },
            { OpenTexMotionStudio, "✨ TexMotion Studioを開く" },
            { TimelinePoseEditor, "TexMotionタイムライン＆ポーズエディター" },
            { TimelineEmptyState, "タイムラインにモーションが読み込まれていません。\nTexMotion Studioでテキストからモーションクリップを生成するか、動画からポーズを抽出すると、ビジュアルキーフレームとボーンポーズを編集できます。" },
            { TimelineGettingStarted, "はじめに" },
            { TimelineStepOne, "1. TexMotion Studioを開き、モーションクリップを生成または抽出します。" },
            { TimelineStepTwo, "2. アニメーションをプレビューし、「タイムラインで編集」をクリックします。" },
            { TimelineStepThree, "3. 3Dジョイントを直接操作し、フレームをスクラブして.animとして保存します。" },
            { Clip, "🎬 クリップ:" },
            { Avatar, "アバター:" },
            { FirstFrameHome, "最初のフレーム（Home）" },
            { PreviousFrameLeft, "前のフレーム（左矢印）" },
            { Play, "▶ 再生" },
            { Pause, "⏸ 一時停止" },
            { TogglePlaybackSpace, "再生/一時停止（Space）" },
            { NextFrameRight, "次のフレーム（右矢印）" },
            { LastFrameEnd, "最後のフレーム（End）" },
            { LoopPlayback, "ループ再生" },
            { Speed, "速度:" },
            { DecreaseSpeed, "速度を下げる（-0.1x）" },
            { ResetSpeed, "クリックで速度を1.0xに戻す" },
            { IncreaseSpeed, "速度を上げる（+0.1x）" },
            { ModifiedCount, "● {0}件を変更" },
            { Clean, "✓ 変更なし" },
            { Undo, "↩ 元に戻す" },
            { Redo, "↪ やり直す" },
            { RevertAll, "↩ すべて元に戻す" },
            { RevertAllTitle, "全フレームを元に戻す" },
            { RevertAllMessage, "すべてのフレーム変更を破棄し、生成直後のモーションに戻しますか？" },
            { Revert, "元に戻す" },
            { SaveAnim, "💾 .animを保存" },
            { ApplyToAvatar, "✨ アバターに適用" },
            { ViewportVideo, "3Dビューポート＆動画" },
            { MissingAvatar, "⚠️ アバター未設定\n上のツールバーでHumanoidアバターを選択してください。" },
            { LoadingVideo, "動画ストリームを読み込み中..." },
            { Frame, "📍 フレーム {0} / {1}" },
            { Time, "時間: {0:F2}秒" },
            { PreviousFrameButton, "◀ 前のフレーム" },
            { NextFrameButton, "次のフレーム ▶" },
            { FramePoseTools, "🛠️ フレームポーズツール" },
            { Copy, "📋 コピー" },
            { CopyTooltip, "現在のフレームポーズをクリップボードにコピー" },
            { Paste, "📄 貼り付け" },
            { PasteTooltip, "クリップボードのポーズをこのフレームに貼り付け" },
            { Mirror, "🔄 ミラー" },
            { MirrorTooltip, "左右の手足を反転したポーズにする" },
            { Smooth, "⚖️ スムーズ" },
            { SmoothTooltip, "隣接フレームと補間して滑らかにする" },
            { ResetFrame, "↩️ フレームをリセット" },
            { ResetFrameTooltip, "このフレームを生成直後のポーズに戻す" },
            { TPose, "🧘 Tポーズ" },
            { TPoseTooltip, "このフレームをニュートラルなTポーズにする" },
            { RootPosition, "ルート位置（オフセット）:" },
            { Active, "🎯 選択中" },
            { Select, "🎯 選択" },
            { XLeftRight, "X（左右）" },
            { YHeight, "Y（高さ）" },
            { ZForwardBack, "Z（前後）" },
            { Pitch, "ピッチ（X）" },
            { Yaw, "ヨー（Y）" },
            { Roll, "ロール（Z）" },
            { OcclusionTools, "🔮 オクルージョンと曖昧さの補正" },
            { Ambiguity, "⚠ 曖昧さ: {0}\nフレーム: {1} - {2}（信頼度: {3:P0}）" },
            { TrackingConfidence, "トラッキングの曖昧さ" },
            { ConfidentPoseHint, "このフレームの姿勢トラッキングは安定しています。下のツールでオクルージョンの前後関係を手動調整できます。" },
            { SwapLegCrossing, "🔄 脚の交差を反転（脚の前後）" },
            { SwapLegTooltip, "脚の交差における前後関係を反転します。" },
            { ArmBehind, "✋ {0}腕を後ろ" },
            { ArmFront, "✋ {0}腕を前" },
            { PreviousModifiedFrame, "前の変更フレーム" },
            { NextModifiedFrame, "次の変更フレーム" },
            { Zoom, "ズーム:" },
            { LiveHandFaceAssistance, "🖐 手と表情の補助（リアルタイムプレビュー）" },
            { LiveHandFaceAssistanceHint, "変更はアバタープレビューと書き出すAnimationClipに反映されます。" },

            { ExportUnityPackage, "TexMotion UnityPackageをエクスポート" },
            { ExportingPackage, "[TexMotion] パッケージをエクスポート中: {0}" },
            { ExportComplete, "[TexMotion] エクスポート完了!" },
            { ModularAvatarInstalled, "Modular Avatarをインストールしました" },
            { ModularAvatarInstalledMessage, "Modular Avatarをプロジェクトにインストールしました。アニメーションを非破壊で適用できます。" },
            { InstallationFailed, "インストールに失敗しました" },
            { InstallationFailedMessage, "Modular Avatarを自動インストールできませんでした:\n{0}\n\nVCC（VRChat Creator Companion）からModular Avatarをインストールしてください。" },
            { Ok, "OK" },
            { MenuMotionNotAdded, "[TexMotion] '{0}'をメニュー'{1}'に追加できませんでした（満杯で置き換えも拒否）。アニメーター設定は完了しています。空きを作って再実行してください。" },
            { BaseAnimationLayersMissing, "[TexMotion] descriptorにbaseAnimationLayersプロパティがありません。" },
            { BaseAnimationLayerIndexMissing, "[TexMotion] baseAnimationLayersにレイヤーインデックス{0}がありません。" },
            { EnumMemberMissing, "[TexMotion] {1}にEnumメンバー'{0}'がありません。raw値{2}を使用します。" },
            { ControlTypeMissing, "[TexMotion] ControlTypeメンバー'{0}'が見つかりません。" },
            { PlayableLayerControlMissing, "[TexMotion] VRCPlayableLayerControlが見つかりません。Actionレイヤーのウェイトは制御されません。" },
            { TrackingControlMissing, "[TexMotion] VRCAnimatorTrackingControlが見つかりません。トラッキングは切り替えられません。" },
            { ParameterDriverMissing, "[TexMotion] VRCAvatarParameterDriverが見つかりません。ワンショットパラメーターをリセットできません。" },
            { ParameterDriverSerializeFailed, "[TexMotion] SerializedObject経由でParameterDriver.parametersをシリアライズできませんでした。" },
            { VideoDataValidationWarning, "[TexMotion VideoJobRunner] VideoMotionDataのデシリアライズ警告: {0}" },
            { ProcessTerminationWarning, "[TexMotion VideoJobRunner] プロセス終了中の通知: {0}" },

            { FrameCountInvalid, "フレーム数は0より大きくしてください。" },
            { JointCountInvalid, "ジョイント数は{0}である必要がありますが、{1}でした。" },
            { RootPositionsLengthInvalid, "ルート位置配列の長さ（{0}）がフレーム数（{1}）と一致しません。" },
            { LocalRotationsDimensionsInvalid, "ローカル回転の次元が[Frames, JointCount]と一致しません。" },
            { TimestampsLengthInvalid, "タイムスタンプ配列の長さがフレーム数と一致しません。" },
            { ConfidencesLengthInvalid, "信頼度配列の長さがフレーム数と一致しません。" },
            { UnknownKimodoLoadError, "Kimodoモデルの読み込み中に不明なエラーが発生しました。" },
            { KimodoModelNotLoaded, "Kimodoモデルが読み込まれていません。先にLoadModelを呼び出してください。" },
            { KimodoGenerationFailed, "Kimodoの生成に失敗しました: {0}" },
        

            // Legacy literal aliases retained for UI call sites that still pass display text.
            { "Drag & Drop Video File Here (.mp4, .mov, .webm, .avi)\n— or click 'Browse Video' below —" , "ここに動画ファイル（.mp4、.mov、.webm、.avi）をドラッグ＆ドロップ\n— または下の「動画を参照」をクリック —" },
            { "Selected: {0}\n(Drag & drop another video file to replace)" , "選択済み: {0}\n（別の動画ファイルをドラッグ＆ドロップして置き換え）" },
            { "{0} was unavailable; the extraction fell back to {1}.\n{2} {3}" , "{0}は利用できなかったため、抽出は{1}へフォールバックしました。\n{2} {3}" },
            { "WHAM geometry fallback: {0}/{1} frame(s) used MediaPipe 3D for safe retargeting and overlay.\n{2} {3}\n{4}" , "WHAMジオメトリのフォールバック: {0}/{1}フレームで安全なリターゲットとオーバーレイのためMediaPipe 3Dを使用しました。\n{2} {3}\n{4}" },
            { "Video extraction failed:\n{0}" , "動画抽出に失敗しました:\n{0}" },
            { "{0}\n{1:F2}s" , "{0}\n{1:F2}秒" },
            { "Successfully exported edited AnimationClip to:\n{0}" , "編集済みAnimationClipを次へ書き出しました:\n{0}" },
            { "Failed to save clip:\n{0}" , "クリップの保存に失敗しました:\n{0}" },
            { "Failed to apply motion:\n{0}" , "モーションの適用に失敗しました:\n{0}" },
            { "Browse", "参照" },
            { "General", "一般" },
            { "Motion Generation", "モーション生成" },
            { "Model Repository & Files", "モデルリポジトリとファイル" },
            { "Copy", "コピー" },
            { "  + {0} additional interval(s) highlighted in the timeline." , "  + {0} 追加区間をタイムラインで強調表示しました。" },
            { "  Suggested action: {0}" , "  推奨アクション: {0}" },
            { " (frame {0})" , " (フレーム {0})" },
            { " | Duration: {0:F2}s | Resolution: {1}x{2} | FPS: {3:F1}" , " | 長さ: {0:F2}s | 解像度: {1}x{2} | FPS: {3:F1}" },
            { "0 - Fast / Lightweight" , "0 - 高速 / 軽量" },
            { "0 keeps the full sampled sequence; cap this for constrained memory." , "0の場合はサンプリングした全シーケンスを保持します。メモリ制約がある場合は上限を設定してください。" },
            { "1 - Balanced" , "1 - バランス" },
            { "100% Offline" , "100%オフライン" },
            { "2 - High Precision" , "2 - 高精度" },
            { "3D Avatar (SMPL-X 22)" , "3Dアバター（SMPL-X 22）" },
            { "3D Avatar Motion Playing Smoothly" , "3Dアバターモーションを滑らかに再生中" },
            { "3D Motion Preview (Drag to Rotate, Scroll to Zoom)" , "3Dモーションプレビュー（ドラッグで回転、スクロールでズーム）" },
            { "Active RTMPose/DWPose file" , "アクティブなRTMPose/DWPoseファイル" },
            { "Active file: {0}" , "アクティブファイル: {0}" },
            { "Adapter" , "アダプター" },
            { "Adjust Root Position Frame {0}" , "フレーム{0}のルート位置を調整" },
            { "Advanced Extraction" , "詳細抽出" },
            { "Ambiguous pose hypothesis" , "曖昧なポーズ仮説" },
            { "AnimationClip saved to {0}" , "AnimationClipを保存しました: {0}" },
            { "Applied '{0}' to {1} via Modular Avatar!" , "適用済み '{0}' へ {1} 経由 Modular アバター!" },
            { "Applied '{0}' to {1}!" , "適用済み '{0}' へ {1}!" },
            { "Applied to Avatar" , "アバターに適用済み" },
            { "Apply" , "適用" },
            { "Apply Error" , "適用エラー" },
            { "Apply to Avatar:" , "アバターに適用:" },
            { "Are you sure you want to completely delete '{0}' from project?" , "プロジェクトから「{0}」を完全に削除しますか？" },
            { "Asset Only" , "アセットのみ" },
            { "Auto" , "自動" },
            { "Auto selects RTMPose + MediaPipe from the local catalog when dependencies and weights are ready." , "依存関係と重みが準備できている場合、ローカルカタログからRTMPose + MediaPipeを自動選択します。" },
            { "Auto-open and launch" , "自動で開いて起動" },
            { "Automatically opens the dedicated Motion Timeline Editor upon completing text motion generation or video pose extraction." , "テキストモーション生成または動画姿勢抽出が完了すると、専用のモーションタイムラインエディターを自動で開きます。" },
            { "Avatar Missing" , "アバター未設定" },
            { "Avatar Only" , "アバターのみ" },
            { "Backend fallback" , "バックエンドフォールバック" },
            { "Buffering video stream{0}" , "動画ストリームをバッファ中{0}" },
            { "Building" , "構築中" },
            { "CPU" , "CPU" },
            { "CUDA" , "CUDA" },
            { "CUDA is optional; Auto selects CUDA when available and otherwise CPU." , "CUDAは任意です。利用可能な場合は自動でCUDAを選択し、それ以外はCPUを使用します。" },
            { "Cache and resolved paths" , "キャッシュと解決済みパス" },
            { "Camera Config / Poses" , "カメラ設定 / ポーズ" },
            { "Cancel Video2Motion model download" , "キャンセル 動画2モーション モデル ダウンロード" },
            { "Cancel setup" , "セットアップをキャンセル" },
            { "Cancel uv installation" , "uvのインストールをキャンセル" },
            { "Cancelling Video Motion environment setup..." , "動画モーション環境のセットアップをキャンセル中..." },
            { "Cancelling Video2Motion model download..." , "動画2モーションモデルのダウンロードをキャンセル中..." },
            { "Cancelling uv installation..." , "uvのインストールをキャンセル中..." },
            { "Checkpoint not found; install it from the catalog or select a local file." , "チェックポイントが見つかりません。カタログからインストールするかローカルファイルを選択してください。" },
            { "Checkpoint ready: {0}" , "チェックポイント準備完了: {0}" },
            { "Chest (Middle)" , "胸（中央）" },
            { "Choose the matching MediaPipe task asset in the catalog above (lite/full/heavy)." , "上のカタログから対応するMediaPipeタスクアセット（lite/full/heavy）を選択してください。" },
            { "Choose the pose backend for this extraction. Detailed model paths remain in Settings." , "この抽出で使用する姿勢バックエンドを選択します。詳細なモデルパスは設定にあります。" },
            { "Click Bone to Select | Drag: Orbit | Scroll: Zoom" , "クリックでボーンを選択 | ドラッグ: オービット | スクロール: ズーム" },
            { "Click to reset speed to 1.0x" , "クリックして速度を1.0xに戻します" },
            { "Compatibility mode selected: MediaPipe only. Choose Auto or RTMPose to enable the hybrid 2D auxiliary pass." , "互換モード（MediaPipeのみ）が選択されています。ハイブリッド2D補助パスを有効にするには自動またはRTMPoseを選択してください。" },
            { "Conf: {0:F0}%" , "信頼度: {0:F0}%" },
            { "Contract: {0}" , "契約: {0}" },
            { "Create an isolated project-local venv with uv. The lightweight profile runs MediaPipe + RTMPose/ONNX on CPU or DirectML; the optional quality profile adds the larger PyTorch lifting/refinement stack (WHAM/HybrIK/HMR2 adapters when installed). Live uv output opens below while setup runs and can be copied for diagnostics." , "uvでプロジェクト専用の分離venvを作成します。軽量プロファイルはCPUまたはDirectML上でMediaPipe + RTMPose/ONNXを実行し、任意の品質プロファイルではPyTorchのリフティング/精密化（WHAM/HybrIK/HMR2アダプター）を追加します。セットアップ中のuv出力は下に表示され、診断用にコピーできます。" },
            { "DPVO Weights / Poses" , "DPVOの重み / ポーズ" },
            { "DPVO weights require a local DPVO runner; pose exports can be consumed directly." , "DPVOの重みにはローカルDPVOランナーが必要です。ポーズのエクスポートは直接読み込めます。" },
            { "Decrease Speed by 0.1x" , "速度を下げる by 0.1x" },
            { "Default Cache Directory (AppData):" , "既定のキャッシュディレクトリ（AppData）:" },
            { "Delete" , "削除" },
            { "Delete Motion" , "モーションを削除" },
            { "Dependencies installed successfully. Rechecking Python..." , "依存関係をインストールしました。Pythonを再確認中..." },
            { "Destination Expressions Menu:" , "遷移先Expressions Menu:" },
            { "Destination: {0}" , "保存先: {0}" },
            { "Detector Backend" , "検出バックエンド" },
            { "Detector: {0} | Frames: {1} | Length: {2:F2}s | FPS: {3:F0} | Confidence Range: [{4:F0}% - {5:F0}%]" , "検出器: {0} | フレーム: {1} | 長さ: {2:F2}s | FPS: {3:F0} | 信頼度 Range: [{4:F0}% - {5:F0}%]" },
            { "Download All Video2Motion Models (estimated {0})" , "動画2モーションモデルをすべてダウンロード (estimated {0})" },
            { "Download RTMPose/DWPose ONNX, MediaPipe task assets, or optional WHAM/HMR2/HybrIK quality checkpoints once, then use the local files for offline Video2Motion extraction. WHAM companion rows below also download ViTPose/DPVO assets and install a camera template; SMPL/SMPL-X body data remains user-provided. Each asset can be downloaded separately or all together; Use configures the inference backend below." , "RTMPose/DWPose ONNX、MediaPipeタスクアセット、任意のWHAM/HMR2/HybrIK品質チェックポイントを一度ダウンロードすれば、ローカルファイルでVideo2Motionをオフライン抽出できます。下のWHAMコンパニオン行ではViTPose/DPVOアセットとカメラテンプレートも取得します。SMPL/SMPL-Xボディデータはユーザーが用意してください。各アセットは個別または一括でダウンロードでき、「使用」で下の推論バックエンドを設定します。" },
            { "Download failed: {0}" , "ダウンロード 失敗: {0}" },
            { "Downloading Video2Motion model..." , "動画2モーションモデルをダウンロード中..." },
            { "Downloading all Video2Motion models..." , "動画2モーション全モデルをダウンロード中..." },
            { "Drag Ring to Rotate ({0})" , "リングをドラッグして回転 ({0})" },
            { "Drag: Orbit | Scroll: Zoom" , "ドラッグ: オービット | スクロール: ズーム" },
            { "Duration: {0:F2}s | FPS: {1:F0}" , "長さ: {0:F2}s | FPS: {1:F0}" },
            { "Edit" , "編集" },
            { "Experimental: " , "実験的: " },
            { "Extraction Error" , "抽出エラー" },
            { "Extraction cancelled." , "抽出をキャンセルしました。" },
            { "Extraction failed." , "抽出に失敗しました。" },
            { "Failed to apply motion: {0}" , "モーションの適用に失敗しました: {0}" },
            { "Failed to load model: {0}" , "モデルの読み込みに失敗しました: {0}" },
            { "Failed to save clip: {0}" , "クリップの保存に失敗しました: {0}" },
            { "Feature Archive" , "特徴量アーカイブ" },
            { "File Path:" , "ファイルパス:" },
            { "Fixes left arm posture to behind head" , "左腕を頭の後ろの姿勢に補正します" },
            { "Fixes left arm posture to front of chest" , "左腕を胸の前の姿勢に補正します" },
            { "Fixes right arm posture to behind head" , "右腕を頭の後ろの姿勢に補正します" },
            { "Fixes right arm posture to front of chest" , "右腕を胸の前の姿勢に補正します" },
            { "Frame {0}: Occlusion / Ambiguity Warning" , "フレーム{0}: オクルージョン / 曖昧さの警告" },
            { "Frame {0}: {1} (Confidence: {2:P0})" , "Frame {0}: {1} (信頼度: {2:P0})" },
            { "Frame: {0} / {1}" , "フレーム: {0} / {1}" },
            { "Frames: {0} | Length: {1:F2}s | FPS: {2:F0} | Confidence: {3:F1}%" , "フレーム: {0} | 長さ: {1:F2}s | FPS: {2:F0} | 信頼度: {3:F1}%" },
            { "Frames: {0} | Length: {1:F2}s | Hand: {2} | Face: {3}" , "フレーム: {0} | 長さ: {1:F2}s | 手: {2} | 表情: {3}" },
            { "Frames: {0} | {1:F2}s ({2:F0} FPS)" , "フレーム: {0} | {1:F2}s ({2:F0} FPS)" },
            { "Fusion: {0} camera | Reprojection error {1:F3} | WHAM correction {2:F3}m" , "融合: {0}カメラ | 再投影誤差 {1:F3} | WHAM補正 {2:F3}m" },
            { "Generation Error" , "生成エラー" },
            { "Generation error: {0}" , "生成エラー: {0}" },
            { "Generic TorchScript temporal 3D backend." , "汎用TorchScript時間3Dバックエンド。" },
            { "Great!" , "完了!" },
            { "HF Repo (User/Repo)" , "HFリポジトリ（User/Repo）" },
            { "HMR2 / 4D-Humans 3D" , "HMR2 / 4D-Humans 3D" },
            { "HMR2 / 4D-Humans image-conditioned SMPL backend." , "HMR2 / 4D-Humans画像条件付きSMPLバックエンド。" },
            { "Hand pose • emotion" , "手のポーズ • 表情" },
            { "Head" , "頭" },
            { "High Confidence ({0:F1}%)" , "高信頼度 ({0:F1}%)" },
            { "Hips / Root" , "腰 / ルート" },
            { "Hugging Face repository" , "Hugging Faceリポジトリ" },
            { "HybrIK 3D + IK" , "HybrIK 3D + IK" },
            { "HybrIK analytical/neural IK backend." , "HybrIK解析/ニューラルIKバックエンド。" },
            { "Hybrid Video-to-Motion" , "ハイブリッド動画モーション" },
            { "In-Place (Stay on origin)" , "インプレース（原点に留まる）" },
            { "Increase Speed by 0.1x" , "速度を上げる by 0.1x" },
            { "Inference Profile" , "推論プロファイル" },
            { "Installation failed: {0}" , "インストールに失敗しました: {0}" },
            { "Installing Python dependencies (mediapipe, opencv-python, numpy, scipy)..." , "Python依存関係（mediapipe、opencv-python、numpy、scipy）をインストール中..." },
            { "Installing uv" , "uvをインストール中" },
            { "Interpolate Low-Confidence Outliers" , "低信頼度外れ値を補間" },
            { "Interpolated low-confidence outlier frames." , "低信頼度の外れ値フレームを補間しました。" },
            { "Intrinsics configure projection; exported DPVO/SLAM poses provide world motion." , "内部パラメーターは投影を設定し、エクスポートしたDPVO/SLAMポーズがワールドの動きを提供します。" },
            { "It requires the PyTorch Quality environment, a compatible checkpoint, and the local MediaPipe auxiliary asset. " , "PyTorch品質環境、互換チェックポイント、ローカルMediaPipe補助アセットが必要です。 " },
            { "Kimodo + LLM2Vec Text Bundle Loaded" , "Kimodo + LLM2Vecテキストバンドルを読み込みました" },
            { "Kimodo engine loaded & ready." , "Kimodoエンジンの読み込み完了。準備完了です。" },
            { "Latest uv output:" , "最新のuv出力:" },
            { "Leave empty for <project>/.texmotion-venv." , "空欄の場合: <プロジェクト>/.texmotion-venv." },
            { "Left Ankle" , "左足首" },
            { "Left Ankle (Foot)" , "左足首（足）" },
            { "Left Collar (Shoulder)" , "左鎖骨（肩）" },
            { "Left Elbow" , "左肘" },
            { "Left Elbow (Forearm)" , "左肘（前腕）" },
            { "Left Hip" , "左股関節" },
            { "Left Hip (Upper Leg)" , "左股関節（上脚）" },
            { "Left Knee" , "左膝" },
            { "Left Knee (Lower Leg)" , "左膝（下腿）" },
            { "Left Shoulder (Collar)" , "左肩（鎖骨）" },
            { "Left Toes" , "左つま先" },
            { "Left Upper Arm" , "左上腕" },
            { "Left Wrist (Hand)" , "左手首（手）" },
            { "Lightweight ONNX" , "軽量ONNX" },
            { "Lightweight ONNX — recommended default; no PyTorch or CUDA dependency required." , "軽量ONNX — 推奨の既定値です。PyTorchやCUDAの依存関係は不要です。" },
            { "Loading 3D Preview..." , "3Dプレビューを読み込み中..." },
            { "Loading frames & buffering stream..." , "フレームを読み込みストリームをバッファ中..." },
            { "Local Directory" , "ローカルディレクトリ" },
            { "Local Storage Configuration" , "ローカルストレージ設定" },
            { "Local runtime ready — no network access is used while extracting." , "ローカルランタイム準備完了 — 抽出中にネットワークアクセスは使用しません。" },
            { "Loop" , "ループ" },
            { "Low Confidence ({0:F1}%)" , "低信頼度（{0:F1}%）" },
            { "Manage, preview, and add/remove generated motions on your avatar." , "アバター上の生成済みモーションを管理、プレビュー、追加/削除します。" },
            { "Manual" , "手動" },
            { "Max Sequence Frames" , "最大シーケンスフレーム数" },
            { "MediaPipe Complexity" , "MediaPipe複雑度" },
            { "MediaPipe Pose Overlay" , "MediaPipeポーズオーバーレイ" },
            { "Model checkpoint, adapter, device, and temporal settings are stored here and used by the Video Motion tab. " , "モデルチェックポイント、アダプター、デバイス、時間設定をここで管理し、動画モーションタブで使用します。 " },
            { "Model checkpoint, adapter, device, and temporal settings are stored here and used by the Video Motion tab. The backend can also be changed from the Video Motion dropdown; both tabs persist the same choice. The model catalog above handles one-click provisioning for public RTMPose, DWPose, MediaPipe, WHAM, and HybrIK assets; HMR2's official checkpoint, runtime, and licensed body stages remain manual with Guide actions.", "モデルチェックポイント、アダプター、デバイス、時間設定をここで管理し、動画モーションタブで使用します。バックエンドは動画モーションのドロップダウンからも変更でき、両タブで同じ選択を保存します。上のモデルカタログでは公開RTMPose、DWPose、MediaPipe、WHAM、HybrIKアセットをワンクリックで用意できます。HMR2の公式チェックポイント、ランタイム、ライセンス済みボディはガイドから手動で設定します。" },
            { "Model is not loaded. Please check settings." , "Model は not loaded. Please check 設定." },
            { "Models downloaded successfully!" , "Models downloaded 正常に!" },
            { "Moderate Confidence ({0:F1}%)" , "中信頼度 ({0:F1}%)" },
            { "Modular Avatar Mode: Will create a non-destructive child object under your avatar." , "Modular Avatarモード: アバターの下に非破壊の子オブジェクトを作成します。" },
            { "Motion '{0}' was successfully added directly into '{1}', ExpressionParameters, and {2}!" , "モーション「{0}」を「{1}」、ExpressionParameters、{2}に直接追加しました。" },
            { "Motion '{0}' was successfully applied to {1} via Modular Avatar!" , "Modular Avatar経由でモーション「{0}」を{1}に適用しました。" },
            { "Motion GGUF File" , "モーションGGUFファイル" },
            { "Motion Model: {0}" , "モーション Model: {0}" },
            { "Motion Name" , "モーション名" },
            { "Motion Playback" , "モーション再生" },
            { "Motion Saved" , "モーションを保存しました" },
            { "Motion Timeline Editor" , "モーションタイムラインエディター" },
            { "Motion button will be added directly into '{0}'." , "モーションボタンを直接追加します: '{0}'." },
            { "Motion ready! Previewing on {0}." , "モーション準備完了。プレビュー対象: {0}." },
            { "Neck" , "首" },
            { "Next Frame" , "次のフレーム" },
            { "No generated motions found in Assets/TexMotion/Generated. Generate a new motion in 'Motion Generator' or extract one in 'Video Motion'!" , "Assets/TexMotion/Generatedに生成済みモーションがありません。「モーション生成」で作成するか「動画モーション」で抽出してください。" },
            { "ONNX Runtime available" , "ONNX Runtime利用可能" },
            { "ONNX Runtime not detected" , "ONNX Runtime未検出" },
            { "OS codec decode unavailable or timed out." , "OSコーデックのデコードが利用できないかタイムアウトしました。" },
            { "Official Windows installer default: {0}" , "公式Windowsインストーラーの既定値: {0}" },
            { "One-Shot" , "ワンショット" },
            { "One-time local setup pending — cache the model/runtime once, then extract offline." , "初回のローカルセットアップ待ち — モデル/ランタイムを一度キャッシュすればオフラインで抽出できます。" },
            { "Open uv docs" , "uvドキュメントを開く" },
            { "Optional .pt/.pth/.ckpt/.tar checkpoint. Adapters may load their own weights." , "任意の.pt/.pth/.ckpt/.tarチェックポイント。アダプターが独自の重みを読み込む場合があります。" },
            { "Overlay" , "オーバーレイ" },
            { "Overlay source: {0}" , "オーバーレイソース: {0}" },
            { "Pelvis / Hips Rotation" , "骨盤 / 腰の回転" },
            { "Please assign a humanoid Target Avatar above to edit this clip." , "このクリップを編集するには上でHumanoid対象アバターを割り当ててください。" },
            { "Please select your target Avatar in the scene first." , "先にシーンで対象アバターを選択してください。" },
            { "Precomputed WHAM frame features (.npy/.npz/.pt), one row per sampled frame." , "事前計算済みWHAMフレーム特徴量（.npy/.npz/.pt）。サンプリングした各フレームが1行です。" },
            { "Preparing Pose Overlay Video..." , "ポーズオーバーレイ動画を準備中..." },
            { "Preparing Video2Motion model download..." , "動画2モーションモデルのダウンロードを準備中..." },
            { "Preparing all Video2Motion model downloads..." , "動画2モーション全モデルのダウンロードを準備中..." },
            { "Preview discarded." , "プレビューを破棄しました。" },
            { "Previewing Library Motion: '{0}'" , "Previewing Library モーション: '{0}'" },
            { "Previous Frame" , "前のフレーム" },
            { "Probe: project venv is the active Video Motion runtime." , "プローブ: プロジェクトvenvがアクティブな動画モーションランタイムです。" },
            { "Probe: {0}" , "プローブ: {0}" },
            { "Profile changed. Build or update the venv to apply it." , "プロファイルが変更されました。 venvを構築または更新してください へ apply it." },
            { "Provision {0}" , "用意 {0}" },
            { "PyTorch / CUDA: Not required  •  Execution: CPU or DirectML via ONNX Runtime" , "PyTorch / CUDA: 不要  •  実行: ONNX Runtime経由のCPUまたはDirectML" },
            { "PyTorch Device" , "PyTorchデバイス" },
            { "PyTorch Quality" , "PyTorch品質" },
            { "PyTorch Quality — larger install; enables optional temporal 3D lifting/refinement experiments." , "PyTorch品質 — 大きなインストールです。任意の時間3Dリフティング/精密化実験を有効にします。" },
            { "PyTorch Temporal 3D" , "PyTorch時間3D" },
            { "PyTorch: Required for the selected quality backend  •  CUDA: Optional  •  CPU fallback remains available" , "PyTorch: 選択した品質バックエンドに必要  •  CUDA: 任意  •  CPUフォールバックも利用可能" },
            { "Python" , "Python" },
            { "Python module name or .py adapter implementing infer_sequence." , "infer_sequenceを実装するPythonモジュール名または.pyアダプター。" },
            { "Quality pose failed auxiliary silhouette alignment." , "品質ポーズと補助シルエットの位置合わせに失敗しました。" },
            { "Quick Select from Avatar's Menus:" , "アバターのメニューからクイック選択:'s Menus:" },
            { "RTMPose 2D  +  MediaPipe 3D  +  Multi-Hypothesis Fit (Stage B/C)" , "RTMPose 2D + MediaPipe 3D + 多重仮説フィッティング（ステージB/C）" },
            { "RTMPose 2D ONNX" , "RTMPose 2D ONNX" },
            { "RTMPose 2D Overlay" , "RTMPose 2Dオーバーレイ" },
            { "RTMPose 2D observations were fused with MediaPipe 3D auxiliaries and Stage B/C multi-hypothesis fitting." , "RTMPose 2D観測をMediaPipe 3D補助情報およびステージB/C多重仮説フィッティングと融合しました。" },
            { "RTMPose Hybrid (2D + 3D Auxiliary)" , "RTMPoseハイブリッド（2D + 3D補助）" },
            { "Reason:" , "理由:" },
            { "Refresh" , "更新" },
            { "Remove" , "削除" },
            { "Reset" , "リセット" },
            { "Resolved Paths:" , "解決済みパス:" },
            { "Resolved directory: {0}" , "解決済みディレクトリ: {0}" },
            { "Resolved venv" , "解決済みvenv" },
            { "Resolved: {0}" , "解決済み: {0}" },
            { "Retry Video Playback" , "動画再生を再試行" },
            { "Review Uncertainty Intervals in Timeline Editor" , "タイムラインエディターで不確実性区間を確認" },
            { "Review the highlighted spans in the Timeline Editor. Scrub the interval and add a correction anchor when the two pose hypotheses are visually indistinguishable." , "タイムラインエディターで強調された区間を確認してください。2つのポーズ仮説を見分けられない場合は、区間をスクラブして補正アンカーを追加します。" },
            { "Right Ankle" , "右足首" },
            { "Right Ankle (Foot)" , "右足首（足）" },
            { "Right Collar (Shoulder)" , "右鎖骨（肩）" },
            { "Right Elbow" , "右肘" },
            { "Right Elbow (Forearm)" , "右肘（前腕）" },
            { "Right Hip" , "右股関節" },
            { "Right Hip (Upper Leg)" , "右股関節（上脚）" },
            { "Right Knee" , "右膝" },
            { "Right Knee (Lower Leg)" , "右膝（下腿）" },
            { "Right Shoulder (Collar)" , "右肩（鎖骨）" },
            { "Right Toes" , "右つま先" },
            { "Right Upper Arm" , "右上腕" },
            { "Right Wrist (Hand)" , "右手首（手）" },
            { "Root Menu" , "ルート Menu" },
            { "Rotate {0} Frame {1}" , "回転 {0} Frame {1}" },
            { "Rotate {0} on Frame {1}" , "回転 {0} 上 Frame {1}" },
            { "Running motion diffusion..." , "実行中 motion diffusion..." },
            { "Save Error" , "保存エラー" },
            { "Saved" , "保存しました" },
            { "Saved AnimationClip to {0}" , "AnimationClipを保存しました: {0}" },
            { "Saved Motion Library" , "保存済みモーションライブラリ" },
            { "Select DPVO checkpoint or pose archive" , "選択 DPVO チェックポイント または ポーズ アーカイブ" },
            { "Select Model Directory" , "選択 Model Directory" },
            { "Select RTMPose/DWPose ONNX model" , "選択 RTMPose/DWPose ONNX モデル" },
            { "Select ViTPose image-feature backbone" , "選択 ViTPose image-feature backbone" },
            { "Select Video Motion venv directory" , "選択 動画 モーション venv ディレクトリ" },
            { "Select Video for Motion Extraction" , "モーション抽出用動画を選択" },
            { "Select Video2Motion model directory" , "選択 動画2モーション モデル ディレクトリ" },
            { "Select WHAM camera calibration or pose archive" , "選択 WHAM カメラ calibration または ポーズ アーカイブ" },
            { "Select WHAM image feature archive" , "選択 WHAM image feature アーカイブ" },
            { "Select quality adapter" , "品質アダプターを選択" },
            { "Select the inference path used for the next extraction." , "推論経路を選択 used 用  next 抽出." },
            { "Select uv executable" , "uv実行ファイルを選択" },
            { "Select {0} checkpoint" , "{0}チェックポイントを選択" },
            { "Selected path: hybrid observations with temporal multi-hypothesis optimization; heavy PyTorch/CUDA backends are not part of this path." , "選択経路: 時間方向の多重仮説最適化によるハイブリッド観測。重いPyTorch/CUDAバックエンドはこの経路では使用しません。" },
            { "Selected path: temporal PyTorch quality inference keeps competing 3D hypotheses through occlusion; RTMPose and MediaPipe remain auxiliary evidence for the Stage B/C fit." , "選択経路: 時間PyTorch品質推論でオクルージョン中も競合する3D仮説を保持します。RTMPoseとMediaPipeはステージB/Cフィッティングの補助証拠として使用します。" },
            { "Selected {0} for Video Motion inference." , "選択ed {0} 用 動画 モーション inference." },
            { "Separate cache for Video2Motion model assets." , "専用キャッシュ: 動画2モーション モデル アセット." },
            { "Setting up 3D preview viewport..." , "3Dプレビュービューポートを準備中..." },
            { "Setup Complete" , "セットアップ完了" },
            { "Setup Target" , "セットアップ対象" },
            { "Setup required" , "セットアップが必要" },
            { "Show live setup log" , "セットアップログを表示" },
            { "Side-by-Side" , "左右比較" },
            { "Size: {0:F1} MB" , "サイズ: {0:F1} MB" },
            { "Source: {0}" , "ソース: {0}" },
            { "Spine (Lower)" , "背骨（下部）" },
            { "Stage B/C" , "ステージB/C" },
            { "Stage B/C Fit" , "ステージB/Cフィッティング" },
            { "Starting download from Hugging Face..." , "Starting ダウンロード から Hugging Face..." },
            { "Starting the official uv installer..." , "公式uvインストーラーを開始中..." },
            { "Successfully added '{0}' directly to {1} & Animator!" , "追加に成功: '{0}' directly へ {1} & Animator!" },
            { "Successfully applied '{0}' to {1} via Modular Avatar!" , "適用に成功: '{0}' へ {1} 経由 Modular アバター!" },
            { "Successfully built and applied '{0}' to {1}!" , "ビルドして適用しました: '{0}' へ {1}!" },
            { "Synchronized Side-by-Side Motion Preview" , "左右比較モーションプレビュー" },
            { "Synthetic Pose Overlay" , "合成ポーズオーバーレイ" },
            { "Target Avatar Required" , "対象アバターが必要" },
            { "Target Avatar is not configured. Please assign an avatar." , "対象アバターが設定されていません。アバターを割り当ててください。" },
            { "Target Avatar is required to apply motion." , "対象アバターが必要です へ apply motion." },
            { "Target Avatar:" , "対象アバター:" },
            { "Target Layer" , "対象レイヤー" },
            { "Target Menu Asset" , "対象メニューアセット" },
            { "Target Repo: {0}" , "対象リポジトリ: {0}" },
            { "Temporal PyTorch 3D  +  RTMPose 2D  +  MediaPipe 3D Auxiliary  +  Multi-Hypothesis Fit (Stage B/C)" , "時間PyTorch 3D + RTMPose 2D + MediaPipe 3D補助 + 多重仮説フィッティング（ステージB/C）" },
            { "Temporal PyTorch 3D hypotheses were fused with available RTMPose/MediaPipe auxiliary observations and Stage B/C fitting." , "時間PyTorch 3D仮説を利用可能なRTMPose/MediaPipe補助観測およびステージB/Cフィッティングと融合しました。" },
            { "Text Bundle Folder" , "テキストバンドルフォルダー" },
            { "Text Bundle: {0}" , "テキストバンドル: {0}" },
            { "The affected spans are listed below and highlighted in the Timeline Editor." , "影響区間を下に一覧表示し、タイムラインエディターで強調表示しています。" },
            { "The bundled WHAM adapter bridge is installed. Download a compatible WHAM checkpoint to enable TexMotion's native temporal core. " , "同梱WHAMアダプターブリッジがインストールされています。互換WHAMチェックポイントをダウンロードするとTexMotionのネイティブ時間コアを有効にできます。 " },
            { "The cache is scanned automatically. Explicit paths let you keep licensed assets, ViTPose features, or DPVO exports outside the cache." , "キャッシュは自動スキャンされます。明示的なパスを指定すると、ライセンス済みアセット、ViTPose特徴量、DPVOエクスポートをキャッシュ外に保持できます。" },
            { "The optimizer found no ambiguous spans. You can still inspect and edit every frame in the Timeline Editor." , "最適化で曖昧な区間は見つかりませんでした。タイムラインエディターで全フレームを確認・編集できます。" },
            { "The quality pose failed silhouette alignment with the auxiliary observation." , "品質ポーズと補助観測のシルエット位置合わせに失敗しました。" },
            { "The selected backend was unavailable." , "選択したバックエンドを利用できません。" },
            { "Timeline" , "タイムライン" },
            { "Trim • quality • backend" , "トリミング • 品質 • バックエンド" },
            { "Uncertainty Intervals  •  None detected" , "不確実性区間  •  None 検出" },
            { "Uncertainty Intervals  •  {0} detected" , "不確実性区間  •  {0} 検出" },
            { "Upper Chest" , "胸（上部）" },
            { "Use 'uv' for PATH lookup or select uv.exe." , "PATH検索には'uv'を使用するかuv.exeを選択してください。" },
            { "Use Custom Local Path" , "カスタムローカルパスを使用" },
            { "Use detected uv path" , "検出したuvパスを使用" },
            { "Use project venv for Video Motion" , "動画モーションにプロジェクトvenvを使用" },
            { "VRCExpressionsMenu" , "VRCExpressionsMenu" },
            { "VRChat Setup Target & Destination Menu" , "VRChat Setup 対象 & Destination Menu" },
            { "VRChat Setup Target & Menu" , "VRChat Setup 対象 & Menu" },
            { "Variable frame timestamps detected — synchronized with precision time interpolation." , "可変フレームタイムスタンプを検出 — 高精度な時間補間で同期しました。" },
            { "Venv Path" , "venvパス" },
            { "ViTPose Backbone" , "ViTPoseバックボーン" },
            { "ViTPose Image Features" , "ViTPose画像特徴量" },
            { "ViTPose features cached (ready for accelerated WHAM extraction)" , "ViTPose特徴量キャッシュ済み (高速WHAM姿勢抽出の準備完了)" },
            { "Extracting ViTPose features in background..." , "バックグラウンドでViTPose特徴量を抽出中..." },
            { "CUDA (GPU) is not available. ViTPose feature extraction on CPU takes a significant amount of time (several minutes). You can proceed directly with MediaPipe/RTMPose fallback, or extract manually below if needed." , "CUDA (GPU) が検出されませんでした。CPUでのViTPose特徴量抽出には非常に長い時間（数分〜十数分）がかかります。通常のMediaPipe/RTMPoseを使用するか、必要な場合のみ手動で抽出してください。" },
            { "Extract Features Manually (CPU)" , "特徴量を手動抽出 (CPUで実行)" },
            { "Extract Features (GPU)" , "特徴量を抽出 (GPU)" },
            { "Re-extract Features" , "特徴量を再抽出" },
            { "ViTPose and DPVO checkpoints are downloaded into the configured cache and passed to the local WHAM contract. Install Camera Template to create camera.yaml, then replace its intrinsics; a template alone does not estimate motion. Precomputed feature/pose archives and SMPL/SMPL-X body data can be supplied with Browse." , "ViTPoseとDPVOのチェックポイントは設定したキャッシュに保存され、ローカルWHAM契約へ渡されます。カメラテンプレートをインストールしてcamera.yamlを作成し、内部パラメーターを置き換えてください。テンプレートだけでは動きを推定できません。事前計算済み特徴量/ポーズアーカイブとSMPL/SMPL-Xボディデータは「参照」から指定できます。" },
            { "ViTPose checkpoint used by a configured local image-feature extractor." , "設定したローカル画像特徴量抽出器で使用するViTPoseチェックポイント。" },
            { "Video Motion Python Environment" , "動画 モーション Python 環境" },
            { "Video Motion backend set to {0}." , "動画モーションのバックエンドを設定: {0}." },
            { "Video Motion environment ready ({0})." , "動画モーション環境の準備完了 ({0})." },
            { "Video Motion environment setup cancelled." , "動画 モーション environment セットアップ cancelled." },
            { "Video Motion environment setup did not complete." , "動画モーション環境のセットアップが完了しませんでした。" },
            { "Video Motion venv ready. Re-probing Python capabilities..." , "動画モーションvenvの準備完了。Python機能を再検出中..." },
            { "Video Only" , "動画のみ" },
            { "Video Overlay Playback Skipped" , "動画オーバーレイ再生をスキップしました" },
            { "Video Pose Dependencies Ready ({0})" , "動画姿勢依存関係の準備完了 ({0})" },
            { "Video Stream Not Available" , "動画ストリームを利用できません" },
            { "Video motion extracted ({0} frames, {1:F2}s). Ready to preview!" , "動画モーションを抽出しました（{0}フレーム、{1:F2}秒）。プレビューの準備完了です。" },
            { "Video motion extracted with fallback: {0} → {1}. See the result card for the reason." , "フォールバックで動画モーションを抽出: {0} → {1}. 理由は結果カードを確認してください。" },
            { "Video2Motion Inference" , "Video2Motion推論" },
            { "Video2Motion Model Assets" , "Video2Motionモデルアセット" },
            { "Video2Motion model and WHAM adapter are ready: {0}" , "動画2モーション モデル と WHAM アダプター は 準備完了: {0}" },
            { "Video2Motion model download cancelled." , "動画2モーションモデルのダウンロードをキャンセルしました。" },
            { "Video2Motion model is ready: {0}" , "動画2モーション モデル は 準備完了: {0}" },
            { "WHAM + MediaPipe Fusion" , "WHAM + MediaPipe融合" },
            { "WHAM + MediaPipe temporal 3D fusion (WHAM primary, MediaPipe auxiliary)." , "WHAM + MediaPipe時間3D融合（WHAMを主、MediaPipeを補助に使用）。" },
            { "WHAM Setup" , "WHAMセットアップ" },
            { "Checkpoint ready" , "チェックポイント準備完了" },
            { "Checkpoint required" , "チェックポイントが必要" },
            { "{0} companion assets need attention" , "コンパニオンアセット {0} 件を確認してください" },
            { "WHAM ready" , "WHAM準備完了" },
            { "WHAM asset details are grouped in the WHAM Setup card above." , "WHAMアセットの詳細は上のWHAMセットアップカードにまとめています。" },
            { "WHAM checkpoint + adapter" , "WHAMチェックポイント + アダプター" },
            { "The adapter is installed together with this checkpoint." , "アダプターはチェックポイントと一緒にインストールされます。" },
            { "WHAM companion assets" , "WHAMコンパニオンアセット" },
            { "Downloadable ViTPose/DPVO files and camera templates are listed here. Licensed body data and precomputed archives use Browse." , "ダウンロード可能なViTPose/DPVOファイルとカメラテンプレートを表示します。ライセンスが必要なボディデータと事前計算アーカイブは「参照」から指定します。" },
            { "Other Video2Motion Assets" , "その他のVideo2Motionアセット" },
            { "RTMPose / MediaPipe / HMR2 / HybrIK" , "RTMPose / MediaPipe / HMR2 / HybrIK" },
            { "Optional backends and shared cache" , "任意バックエンドと共有キャッシュ" },
            { "These assets are kept out of the WHAM workflow until you need another backend." , "別のバックエンドを使うまで、これらのアセットはWHAMの手順から分離されています。" },
            { "WHAM is the focused setup path: choose the checkpoint, provision companion assets, run the preflight, then create a ViTPose feature archive when the optional image-feature stage is needed. Every fallback or missing file is reported in the result card; nothing is skipped silently." , "WHAMでは、チェックポイント選択、コンパニオンアセット準備、事前チェック、必要な場合のViTPose特徴量アーカイブ作成を順番に行います。フォールバックや不足ファイルは結果カードに表示され、推論が黙って省略されることはありません。" },
            { "1. Install or browse the WHAM checkpoint and adapter.  2. Download or provide the companion assets.  3. Run Runtime Preflight.  4. Create a ViTPose feature archive only when the image-feature stage needs it." , "1. WHAMチェックポイントとアダプターをインストールまたは参照します。 2. コンパニオンアセットをダウンロードまたは指定します。 3. ランタイム事前チェックを実行します。 4. 画像特徴ステージが必要な場合だけViTPose特徴量アーカイブを作成します。" },
            { "WHAM Full-Parity Companion Assets" , "WHAM完全互換コンパニオンアセット" },
            { "WHAM Stage Paths (optional explicit overrides)" , "WHAMステージパス（任意の明示設定）" },
            { "WHAM Temporal 3D" , "WHAM時間3D" },
            { "WHAM adapter configured: {0}" , "WHAMアダプター設定済み: {0}" },
            { "WHAM adapter missing — download WHAM from the catalog to install the bundled adapter." , "WHAMアダプターがありません — カタログからWHAMをダウンロードして同梱アダプターをインストールしてください。" },
            { "WHAM asset/runtime diagnostic:" , "WHAMアセット/ランタイム診断:" },
            { "WHAM checkpoint and bundled adapter bridge are installed. TexMotion's native temporal WHAM core is available for the local path; " , "WHAMチェックポイントと同梱アダプターブリッジがインストールされています。ローカル経路でTexMotionのネイティブ時間WHAMコアを利用できます。 " },
            { "WHAM companion is ready: {0}" , "WHAMコンパニオンの準備完了: {0}" },
            { "WHAM initialization: {0}{1}" , "WHAM初期化: {0}{1}" },
            { "WHAM runtime stages unavailable: " , "WHAMランタイムステージを利用できません: " },
            { "WHAM temporal 3D is primary; MediaPipe visibility-gated observations refine the canonical fused result." , "WHAM時間3Dを主結果とし、MediaPipeの可視性ゲート観測で融合結果を精密化します。" },
            { "WHAM temporal 3D motion backend (native core; full SMPL stages optional)." , "WHAM時間3Dモーションバックエンド（ネイティブコア、完全なSMPLステージは任意）。" },
            { "Waiting for uv output..." , "uv出力を待機中..." },
            { "When enabled, extraction uses the venv Python after setup." , "有効にすると、セットアップ後にvenvのPythonで抽出します。" },
            { "[Video] {0}" , "[動画] {0}" },
            { "auto" , "自動" },
            { "download a model from the catalog above" , "上のカタログからモデルをダウンロード" },
            { "fallback" , "フォールバック" },
            { "local model ready" , "ローカルモデル準備完了" },
            { "ready" , "準備完了" },
            { "selected backend" , "選択中のバックエンド" },
            { "source video" , "ソース動画" },
            { "unknown" , "不明" },
            { "uv Executable" , "uv実行ファイル" },
            { "uv detected" , "uvを検出" },
            { "uv installation cancelled." , "uvのインストールをキャンセルしました。" },
            { "uv installation did not complete." , "uvのインストールが完了しませんでした。" },
            { "uv installation failed." , "uvのインストールに失敗しました。" },
            { "uv installed. You can now create the Video Motion venv." , "uvをインストールしました。動画モーションvenvを作成できます。" },
            { "uv is installed and ready: {0}" , "uvをインストールしました。準備完了: {0}" },
            { "uv not detected" , "uv未検出" },
            { "via uv" , "uv経由" },
            { "{0:F2}s ({1:F0}fps) | {2}" , "{0:F2}秒（{1:F0}fps） | {2}" },
            { "{0:F2}s / {1:F2}s" , "{0:F2}秒 / {1:F2}秒" },
            { "{0}  •  Checkpoint: {1}" , "{0}  •  チェックポイント: {1}" },
            { "{0} Requires the PyTorch Quality environment profile. If unavailable, extraction reports the reason and falls back safely." , "{0}にはPyTorch品質環境プロファイルが必要です。利用できない場合は理由を表示して安全にフォールバックします。" },
            { "{0} chars" , "{0}文字" },
            { "{0} {1}" , "{0} {1}" },
            { "{0}/{1} {2}" , "{0}/{1} {2}" },
            { "• Frames {0}–{1}  |  {2}  |  Confidence {3:F0}%" , "• フレーム {0}–{1}  |  {2}  |  信頼度 {3:F0}%" },
            { "⏹ Reset" , "⏹リセット" },
            { "● HMR2 / 4D-Humans 3D" , "● HMR2 / 4D-Humans 3D" },
            { "● HybrIK 3D + IK" , "● HybrIK 3D + IK" },
            { "● MediaPipe Pose" , "● MediaPipeポーズ" },
            { "● Modified" , "● 変更済み" },
            { "● PyTorch Temporal 3D" , "● PyTorch時間3D" },
            { "● RTMPose 2D" , "● RTMPose 2D" },
            { "● RTMPose Hybrid (2D + 3D Auxiliary)" , "● RTMPoseハイブリッド（2D + 3D補助）" },
            { "● WHAM + MediaPipe Fusion" , "● WHAM + MediaPipe融合" },
            { "● WHAM Temporal 3D" , "● WHAM時間3D" },
            { "⚙️ Create Video Motion venv" , "⚙️動画モーションvenvを作成" },
            { "⚠ WHAM geometry → MediaPipe ({0} frame(s))" , "⚠ WHAM geometry → MediaPipe ({0} フレーム(s))" },
            { "⚠ {0} → {1} {2}" , "⚠ {0} → {1} {2}" },
            { "✅ Found" , "✅検出済み" },
            { "✋ L-Arm Behind" , "✋左腕を後ろ" },
            { "✋ L-Arm Front" , "✋左腕を前" },
            { "✋ R-Arm Behind" , "✋右腕を後ろ" },
            { "✋ R-Arm Front" , "✋右腕を前" },
            { "✏️ Edit in Timeline" , "✏️タイムラインで編集" },
            { "✓ Original" , "✓元データ" },
            { "❌ Not Found" , "❌未検出" },
            { "⬇ Install uv" , "⬇uvをインストール" },
            { "🎯 Gizmos" , "🎯ギズモ" },
            { "👤 Root & Hips" , "👤ルートと腰" },
            { "💪 Left Arm" , "💪左腕" },
            { "💪 Right Arm" , "💪右腕" },
            { "💾 Save .anim Only" , "💾 .animのみ保存" },
            { "🔁 Loop" , "🔁ループ" },
            { "🔄 Reset Cam" , "🔄カメラをリセット" },
            { "🔄 Reset View" , "🔄表示をリセット" },
            { "🔄 Swap Leg Crossing" , "🔄脚の交差を反転" },
            { "🔄 Swap Leg Crossing ({0}-{1})" , "🔄脚の交差を反転（{0}-{1}）" },
            { "🔄 Update Video Motion venv" , "🔄動画モーションvenvを更新" },
            { "🔍 Re-probe Python capabilities" , "🔍Python機能を再検出" },
            { "🔮 Occlusion & Ambiguity Tools" , "🔮オクルージョンと曖昧さの補正" },
            { "🗑️ Discard" , "🗑️破棄" },
            { "🛠️ Secondary Pose Tools" , "🛠️補助ポーズツール" },
            { "🦴 Skeleton" , "🦴スケルトン" },
            { "🦴 Torso & Head" , "🦴胴体と頭" },
            { "🦵 Left Leg" , "🦵左脚" },
            { "🦵 Right Leg" , "🦵右脚" },};

        // This alias map is deliberately derived from the English catalog. It
        // means every old English literal accepted by TrLiteral remains a
        // supported input, while only stable constants are used internally.
        // The JapaneseText dictionary also contains direct legacy literal rows
        // for display strings that have not migrated to a named constant yet;
        // those rows keep older call sites localizable without changing their
        // serialized/editor-facing values.
        private static readonly Dictionary<string, string> LegacyLiteralKeys = BuildLegacyLiteralKeys();

        private static Dictionary<string, string> BuildLegacyLiteralKeys()
        {
            var aliases = new Dictionary<string, string>(StringComparer.Ordinal);
            foreach (var pair in EnglishText)
            {
                // If two canonical keys ever share an English label, preserving
                // the first mapping is deterministic and keeps literal callers
                // backward-compatible. New call sites should use constants.
                if (!aliases.ContainsKey(pair.Value)) aliases.Add(pair.Value, pair.Key);
            }
            return aliases;
        }

        /// <summary>Translates a visible English literal without requiring a serialized key.</summary>
        public static string TrLiteral(string english)
        {
            return Tr(english);
        }

        /// <summary>
        /// Translates a canonical key. Legacy English display literals are also
        /// accepted so callers can migrate incrementally. Unknown keys return
        /// their input, preserving the English fallback behavior.
        /// </summary>
        public static string Tr(string key)
        {
            if (string.IsNullOrEmpty(key)) return key ?? string.Empty;

            string canonicalKey;
            if (LegacyLiteralKeys.TryGetValue(key, out canonicalKey)) key = canonicalKey;

            string english;
            if (!EnglishText.TryGetValue(key, out english)) english = key;

            if (IsJapanese())
            {
                string japanese;
                if (JapaneseText.TryGetValue(key, out japanese)) return japanese;
            }
            return english;
        }

        /// <summary>Formats a localized string while retaining safe fallback behavior.</summary>
        public static string TrFormat(string key, params object[] args)
        {
            try { return string.Format(Tr(key), args ?? new object[0]); }
            catch (FormatException) { return Tr(key); }
        }

        /// <summary>Convenience alias for formatting a legacy English literal.</summary>
        public static string TrLiteralFormat(string englishFormat, params object[] args)
        {
            return TrFormat(englishFormat, args);
        }

        private static bool IsJapanese()
        {
            try
            {
                var settings = TexMotionSettings.instance;
                return settings != null && settings.Language == TexMotionLanguage.Japanese;
            }
            catch
            {
                // Localization must never prevent the editor from opening when
                // ProjectSettings are unavailable during domain reload/startup.
                return false;
            }
        }
    }
}
