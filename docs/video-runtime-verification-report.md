# Video runtime verification report

Date: 2026-09-17 (Asia/Tokyo)  
Scope: Python video extraction, the six-role WHAM offline asset contract,
WHAM/MediaPipe fusion, preprocessing provenance, overlay export, and the
`Editor/Video/tests` suite. Settings now also exposes a metadata-driven
manual-asset guide for each upstream/licensed file or runtime bridge.

## Executive result

The current code completed a real-video WHAM smoke on the available 164-frame
Unity-editor recording. It processed **164/164 sampled frames**, returned
`backendRequested=wham`, `backendActual=wham`, `backendFallback=false`, and
exported a decodable H.264 overlay; the process exited 0 and reported no
sequence inference error.

This is a successful native WHAM-core integration run, not a claim of full
upstream WHAM parity. The six declared asset files all resolved, but the
downloaded ViTPose checkpoint did not have a compatible executable model
factory and the downloaded DPVO checkpoint was not run by DPVO. The runtime
rejected the unverified `local_frame_descriptor` instead of feeding it to the
learned image integrator, and used a clearly labelled local optical-flow
camera fallback. These limitations are retained in the output JSON and in the
preprocessing manifest.

The validation run itself did not mutate runtime outputs; the implementation
changes that make the provenance, preflight, and guide contracts explicit are
recorded in the implementation sources and regression tests.

## Six-role requirement audit

The six requirements below are the six roles in
`Editor/Video/models/wham_asset_manifest.json`. “Present” means that the
resolver found a non-empty local file. It does not, by itself, prove that the
corresponding upstream stage executed.

| Role / requirement | Current evidence | Result and limitation |
| --- | --- | --- |
| WHAM checkpoint | Explicit path resolved; native adapter loaded `83` parameters; output metadata reports `adapterImplementation=texmotion_native_wham`, `nativeWhamCore=true`, `preflightStatus=passed`. | **Pass for native core inference.** This is not evidence that every official WHAM stage ran. |
| SMPL/SMPL-X body model | `SMPLX_NEUTRAL.npz` resolved and was passed explicitly. Runtime reports `embeddedSMPL=true`, `officialSmplDecoder=true`, and `smplInitializationSource=mediapipe_observation`. | **Resolved/native decoder evidence.** External body-model presence is not treated as proof of separately licensed upstream body-data parity; the native checkpoint-embedded buffers remain part of this local path. |
| Image-feature backbone | `vitpose-huge.pth` resolved. The supplied model-definition file is the bundled contract template; the runner recorded `runnerKind=unavailable`, adopted `0/164` frames, and did not load the checkpoint. | **Not executed.** `imageFeatureSource=unverified_local_descriptor_rejected`, `imageFeatureFallbackConsumed=false`, and `officialImageFeatureIntegrator=false`. A compatible ViTPose/HMR2 factory or verified `(frames, feature_dim)` archive is still required. |
| Camera calibration/configuration | `camera.yaml` resolved and `cameraCalibrationAvailable=true`. The run generated `camera_poses.npz` with 164 rows of 7 values. | **Calibration resolved; motion fallback used.** The generated motion source is `local_optical_flow`, not DPVO. |
| DPVO camera-motion asset | `dpvo.pth` resolved and `dpvoWeightsAvailable=true`. | **Weights present only.** `dpvoVerified=false`; no DPVO-capable runner was configured, so the checkpoint was not used to produce camera poses. |
| WHAM adapter | Explicit `texmotion_wham_adapter.py` resolved and selected; native adapter initialized and completed the sequence. | **Pass for the TexMotion native adapter bridge.** It is not an assertion that the full official research repository/runtime was installed. |

### Resolver versus runtime readiness

The resolver command was run from the repository root with the package path
made explicit:

```powershell
python -c "import sys,json; sys.path.insert(0,'Editor/Video'); from pose_pipeline.backends.pytorch_backend import WHAM_ASSET_ROLES,resolve_wham_assets; p=r'C:\Users\k0ta0\AppData\Roaming\TexMotion\Models\Video'; r=resolve_wham_assets(model_directory=p,asset_manifest_path=r'Editor\Video\models\wham_asset_manifest.json'); print(json.dumps({'roles':WHAM_ASSET_ROLES,'paths':r.paths,'missingStages':r.missing_stages,'nativeAvailable':r.native_available,'fullParityAvailable':r.full_parity_available,'manifestPath':r.manifest_path,'diagnostic':r.diagnostic_text},indent=2,default=str))"
```

Observed result: all six paths present, `missingStages=[]`,
`nativeAvailable=true`, and `fullParityAvailable=true`. The resolver's
`fullParityAvailable` is a declared-file contract result only. The actual run
reported `whamFullParity=false` with runtime-missing stages
`image_feature_extractor_for_backbone` and `camera_motion_or_slam`; the report
uses the runtime result for parity claims.

## Real-video smoke

### Input and environment

Input:

```text
X:\Pictures\Screenshot\2026-09\Unity_2026_09_16_19_45_33.mp4
```

`ffprobe` reported H.264, 1,486x896, 30 FPS, 164 frames, and 5.466667
seconds. Python was 3.11.8; the smoke was forced to CPU for reproducibility.

### Command

The production-sized smoke used all six explicit role paths plus the local
ViTPose definition/config contract files:

```powershell
python Editor/Video/video_pose_extractor.py --video "X:\Pictures\Screenshot\2026-09\Unity_2026_09_16_19_45_33.mp4" --output "C:\Users\k0ta0\AppData\Local\Temp\texmotion_validation_current_wham_20260917_192051.json" --overlay-video "C:\Users\k0ta0\AppData\Local\Temp\texmotion_validation_current_wham_20260917_192051.mp4" --fps 30 --trim-start 0 --trim-end 5.5 --backend wham --fusion-mode wham_mediapipe --pytorch-device cpu --video-model-dir "C:\Users\k0ta0\AppData\Roaming\TexMotion\Models\Video" --wham-asset-manifest "Editor/Video/models/wham_asset_manifest.json" --pytorch-model "C:\Users\k0ta0\AppData\Roaming\TexMotion\Models\Video\wham_vit_w_3dpw.pth.tar" --pytorch-adapter "C:\Users\k0ta0\AppData\Roaming\TexMotion\Models\Video\texmotion_wham_adapter.py" --wham-body-model "C:\Users\k0ta0\AppData\Roaming\TexMotion\Models\Video\SMPLX_NEUTRAL.npz" --wham-image-feature-backbone "C:\Users\k0ta0\AppData\Roaming\TexMotion\Models\Video\vitpose-huge.pth" --wham-image-feature-model-definition "C:\Users\k0ta0\AppData\Roaming\TexMotion\Models\Video\wham_vitpose_model_definition.py" --wham-image-feature-config "C:\Users\k0ta0\AppData\Roaming\TexMotion\Models\Video\wham_vitpose_runner_config.json" --wham-camera "C:\Users\k0ta0\AppData\Roaming\TexMotion\Models\Video\camera.yaml" --wham-dpvo "C:\Users\k0ta0\AppData\Roaming\TexMotion\Models\Video\dpvo.pth" --wham-preprocess-dir "C:\Users\k0ta0\AppData\Local\Temp\.texmotion_validation_current_wham_20260917_192051_wham_preprocess"
```

### Artifacts and observed output

The exact disposable artifacts are:

```text
C:\Users\k0ta0\AppData\Local\Temp\texmotion_validation_current_wham_20260917_192051.json
C:\Users\k0ta0\AppData\Local\Temp\texmotion_validation_current_wham_20260917_192051.mp4
C:\Users\k0ta0\AppData\Local\Temp\.texmotion_validation_current_wham_20260917_192051_wham_preprocess\manifest.json
C:\Users\k0ta0\AppData\Local\Temp\.texmotion_validation_current_wham_20260917_192051_wham_preprocess\camera_poses.npz
C:\Users\k0ta0\AppData\Local\Temp\texmotion_validation_current_wham_20260917_192051.stdout.log
C:\Users\k0ta0\AppData\Local\Temp\texmotion_validation_current_wham_20260917_192051.stderr.log
```

| Check | Observed result |
| --- | --- |
| Process / sequence | Exit 0; 25.96 s wall time; 164 frames and 164 timestamps; `sequenceInferenceError=null`. |
| Backend selection | `backendRequested=wham`, `backendActual=wham`, `backendFallback=false`, `backendReady=true`. |
| Fusion / export | `fusionMode=wham_mediapipe`; `overlayBackend=wham_mediapipe`; `overlaySource=fused_3d_projection_with_mediapipe_fallback`. |
| Native evidence | `adapterImplementation=texmotion_native_wham`; `loadedParameterCount=83`; `inputDetector=rtmpose`; `smplInitializationSource=mediapipe_observation`. |
| Image stage | `imageFeatureRunnerActive=false`; `imageFeatureRunnerStatus=fallback`; `imageFeatureAdoptedFrameCount=0`; local descriptor rejected and not consumed. |
| Camera stage | `cameraMotionSource=local_optical_flow`; `cameraMotionRunner=local_optical_flow`; `dpvoVerified=false`; `worldMotionAvailable=false`. |
| Runtime parity | `whamFullParity=false`; missing stages are `image_feature_extractor_for_backbone` and `camera_motion_or_slam`. |
| Overlay container | `ffprobe`: H.264, 1,486x896, 30 FPS, 164 frames, 5.466667 s. |

The output contains 5,412 joint-provenance records (164 x 33): 3,444 WHAM
records and 1,968 MediaPipe records. The 1,968 MediaPipe records are explicit
per-joint fallback/adoption records with reason
`wham_invalid_mediapipe_selected`; this is not described as whole-frame WHAM
failure. Separately, 30 frames were recorded in
`qualityFrameFallbacks` for retargeting/overlay safety:

```text
10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 33, 35, 36, 37, 38, 39, 40,
41, 42, 52, 53, 54, 98, 129, 131, 148, 161, 162, 163
```

The reasons are retained per frame and include `median_2d_error`,
`p90_2d_error`, and `body_extent_collapsed`/`body_extent_exploded`.

### Overlay spot check

`ffmpeg` extracted frames 0 and 82 to PNG and both decoded successfully. A
visual spot check of those images found the source viewport, skeleton, and
`TexMotion WHAM + MEDIAPIPE FUSION Overlay` HUD present in the exported video.
The images also show the expected ambiguity/geometry-disagreement diagnostics;
this spot check demonstrates readable end-to-end export, not pose-accuracy or
full-parity certification.

## Provenance and limitations

The run's image-stage warning was:

```text
The bundled ViTPose model-definition template is not an architecture; the checkpoint was not loaded. Configure a compatible HMR2/ViTPose model factory or replace the template with declared model code.
```

The camera-stage warning stated that the DPVO checkpoint was detected but
camera poses were generated by the bundled offline optical-flow runner because
no external DPVO runtime was configured. The report therefore does not call
the local runner DPVO and does not treat `dpvoWeightsAvailable=true` as DPVO
inference evidence.

To reach upstream/full-stage parity, supply a compatible ViTPose/HMR2 model
factory or verified precomputed image-feature archive, and a DPVO-capable
runner/exported camera-motion archive. Separately licensed SMPL/SMPL-X body
data/terms remain an external responsibility. The native path intentionally
continues with explicit fallbacks when optional stages are unavailable.

Unity-side C# compilation and in-Editor visual QA were not executable in this
workspace: there is no `ProjectSettings` directory and neither `Unity` nor
`UnityHub` is on PATH. This report does not claim those checks passed.

## Settings manual-asset guidance

Rows that require an upstream runtime, adapter, model-definition hook,
precomputed archive, DPVO export, or separately licensed body file expose a
`Guide` button in the Settings > Video2Motion catalog. The button opens
`VideoAssetGuideWindow` inside the Unity Editor and shows the official
download/registration URL, the resolved recommended install directory, the
required filename or directory, the Browse setting to edit, role-specific
preflight steps, and the licensing/manual-provisioning reason. The window can
open TexMotion Settings and copy the resolved destination. Public artifacts
still use the existing Download button; licensed or executable-runtime assets
remain explicitly manual so a raw checkpoint is never presented as a complete
runtime.

## Regression verification

All relevant Python checks passed after the smoke:

```text
python -m pytest -q
165 passed in 13.62s

python -m pytest -q Editor/Video/tests/test_wham_native.py Editor/Video/tests/test_pytorch_backend.py
55 passed in focused backend/extractor/native verification

python -m pytest -q Editor/Video/tests
158 passed in 12.95s

python -m compileall -q Editor/Video tests
exit 0
```

The pytest runs emit one pre-existing `RequestsDependencyWarning` from the
installed Python environment; it does not fail the suite.
