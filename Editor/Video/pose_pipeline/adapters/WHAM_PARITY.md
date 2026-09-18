# WHAM adapter parity and offline assets

This adapter tracks the official WHAM implementation at commit
`2b54f7797391c94876848b905ed875b154c4a295` (2024-04-18),
[`yohanshin/WHAM`](https://github.com/yohanshin/WHAM), under the upstream MIT
license.  The state-dict names and recurrent dimensions below are kept
compatible with `wham_vit_w_3dpw.pth.tar`.

## Ported inference stages

The native module contains the official `Regressor`,
`NeuralInitialization`, and `MotionEncoder` path, including WHAM's masked
keypoint preprocessing and first-frame recurrent seed.  It also ports the
checkpoint-compatible `Integrator`, `TrajectoryDecoder`, `MotionDecoder`, and
`TrajectoryRefiner` modules.  Full checkpoints therefore load these keys
without importing the research repository:

```
motion_encoder.*
trajectory_decoder.*
integrator.*
motion_decoder.*
trajectory_refiner.*
```

The official `CustomDataset` forms `init_kp` from root-centered COCO-17 and
`init_smpl` from full-pose 6D rotations.  TexMotion accepts that same data via
`smplInitialization`/`smplModel` and, when present, reads the neutral SMPL
buffers stored in the official checkpoint.  The latter is a small inference
LBS implementation and makes the 527 MB checkpoint sufficient for CPU
SMPL-decoder inference; it does not download `smplx` or body-model files.
When neither source is available, the extractor seeds the recurrent state from
the first valid MediaPipe camera-space 3D observation
(`smplInitializationSource=mediapipe_observation`).  A clearly labelled
neutral seed can still keep the temporal 3D core running when no observation
is available (`smplInitializationSource=neutral_fallback`).
Set `requireSMPLInitialization` or `strictOptionalAssets` to turn that missing
asset into an explicit initialization error.

The official DPVO conversion is available without DPVO itself: pass either
`cameraPoses`/`slamResults` with `[tx, ty, tz, qx, qy, qz, qw]` rows or a
precomputed `cameraAngularVelocity` `[frames, 6]` array.  Without that input,
the trajectory decoder receives the same zero angular-velocity local-only
signal used by the upstream `--estimate_local_only` path, and metadata keeps
`worldMotionAvailable=false`.

The official image `Integrator` is used when `imageFeatures` or an offline
feature extractor/archive is configured.  `.npy`, `.npz`, `.pt`, and `.pth`
archives are accepted, and the feature width is checked against the loaded
checkpoint. Settings can download the public ViTPose-Huge backbone, but the
research ViTPose feature extractor is not bundled or silently substituted;
configure a compatible local extractor or archive. `requireImageFeatures` makes
absence or a dimension mismatch an explicit error.

### Inline ViTPose execution

The optional PyTorch profile can execute a local ViTPose `.pth` checkpoint
directly through the inline runner. A `.pth` file is normally a state dict, not
a self-describing executable model: provide the matching local model definition
and config, and make the runner emit one finite `(frame_count, feature_dim)`
row per input frame. No model download or network access is performed at
runtime. The configured runner is preferred; if it fails, the deterministic
per-video descriptor remains an explicit fallback and its name/reason is kept
in preprocessing metadata rather than being presented as ViTPose output.

Optional ONNX export is a deployment optimization, not a prerequisite. Install
the exporter noted in `requirements-pytorch.txt`, export the same model with
fixed preprocessing, and validate numerical parity (BGR/RGB order, resize,
normalization, output width, and frame ordering) before using the resulting
graph. The base video profile already supplies ONNX Runtime for local ONNX
execution; changing a `.pth` suffix or silently treating weights as an
`(N,D)` archive is unsupported.

## Intentional boundary

The adapter preserves the native API (`infer_sequence`, `infer_frame`, and
`get_metadata`) and the TexMotion coordinate contract.  WHAM camera-space
coordinates (`+Y` down, `+Z` away) are converted once to canonical SMPL-X
(`+Y` up, `+Z` forward); trajectory translations use the same boundary
conversion.  Native `keypoints2d` and `landmarks3d` are both COCO-17. External
adapters may still advertise the historical `wham_j17` topology explicitly;
the generic backend keeps that mapping for compatibility.

The official research preprocessing is not bundled verbatim in this offline
adapter.  The native path now includes an explicit per-video preprocessing
runner, however:

* A configured `imageFeatureExtractor`/`extract_sequence` implementation is
  preferred. A raw `vitpose-huge.pth` is never treated as a feature archive,
  and the bundled deterministic CPU descriptor is **not** fed to the learned
  WHAM integrator. When no compatible runner/archive is available, extraction
  continues with an explicit `local_frame_descriptor` rejection and records
  the exact reason in metadata. Install a compatible HMR2/ViTPose runner or
  supply a verified, frame-aligned archive to enable this stage.
* A configured `cameraPoseExtractor`/`dpvoRunner` is preferred. When only
  `dpvo.pth` is installed, the bundled OpenCV optical-flow runner exports
  DPVO-compatible `camera_poses.npz` rows. Its conservative fallback holds
  the previous pose when tracking is lost and records
  `cameraMotionRunner=local_optical_flow` plus a runtime warning. Projects
  with the official DPVO runtime can replace this runner without changing the
  adapter contract.
* HMR2 first-frame image-conditioned SMPL initialization (the built-in
  MediaPipe camera-space seed is used when available; a configured
  `smplInitialization` remains authoritative);
* Temporal SMPLify optimization and research visualization.

RTMPose ONNX is preferred when a local model exists, with a local MediaPipe
task fallback for the 2D input contract.  That detector choice does not
replace failed WHAM inference: checkpoint, optional-stage, and asset errors
remain in `error`, `optionalErrors`, and `missingOptionalAssets` metadata.

Generated archives are written to the configured
`whamPreprocessDirectory`. If it is omitted, the extractor creates a hidden
`.<output-stem>_wham_preprocess` directory beside the motion JSON so each
video receives frame-aligned features and camera poses.

## Capability metadata

`get_metadata()` exposes both `capabilities` and the compatibility alias
`capabilityFlags`.  The flags distinguish loaded decoder weights from stages
actually used (`trajectoryDecoderWeights` vs `trajectoryDecoder`, and so on),
as well as `smplDerivedInitialization`, `embeddedSMPL`, `cameraMotion`, and
`worldTrajectory`.  Consumers should use `worldTrajectory`/`worldMotionAvailable`
to avoid presenting a camera-relative result as recovered world motion.
