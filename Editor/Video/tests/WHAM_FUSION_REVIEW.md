# WHAM / fusion adversarial review

Date: 2026-09-16  
Scope: `Editor/Video/tests` regression coverage plus a review of the current
Python WHAM, backend, camera, fusion, and extractor integration changes. The
review also wires the frame-level geometry safety gate into the extractor and
preserves explicit frame identity for out-of-order temporal adapter rows.

## Added coverage

`test_wham_adversarial.py` adds nine deterministic tests covering the failure
seams that can turn a valid temporal pose into a collapsed avatar:

- WHAM J17 semantic ordering, including explicit ankle/foot and head aliases;
- one-time WHAM/MediaPipe camera-axis conversion without changing confidence or
  other metadata columns;
- zero geometry, all-missing observations, degenerate perspective projection,
  and finite/no-runtime-warning guarantees;
- independent partial-joint 2D/3D fusion and per-joint provenance;
- sparse frame IDs through fusion intervals and `PyTorchPoseBackend` sequence
  calls, plus frame-identity alignment over out-of-order records;
- requested-backend preservation and concrete initialization fallback errors;
- deterministic collapse/edge-spanning alignment decisions and bounded
  fallback-overlay geometry;
- explicit out-of-order WHAM rows staying attached to their source frames.

The read-only review covered `pose_pipeline/adapters/wham_native.py`,
`pose_pipeline/adapters/texmotion_wham_adapter.py`,
`pose_pipeline/backends/pytorch_backend.py`, `pose_pipeline/camera.py`,
`pose_pipeline/fusion.py`, `video_pose_extractor.py`, the existing Python
regression tests, and the WHAM/fusion metadata seams in the Unity data/job
classes.

## Verification

Targeted tests:

```text
python -m pytest -q Editor/Video/tests/test_wham_adversarial.py
9 passed
```

Full Python suite:

```text
python -m pytest -q
110 passed, 1 warning in 10.13s
```

The warning is PyTorch's expected single-layer RNN dropout warning from the
native WHAM parity fixture. Earlier runs also showed an environment-level
`RequestsDependencyWarning`; neither warning failed a test.

## Target MP4 execution and overlay inspection

The supplied 810-frame target was rerun with the local WHAM checkpoint and
bundled adapter:

```text
C:\Users\k0ta0\Downloads\x_x_user_2099358411762851866.mp4
```

The exact exported result was:

```text
frames=810
backendRequested=wham
backendActual=wham
backendFallback=False
fusionMode=wham_mediapipe
overlaySource=fused_3d_projection_with_mediapipe_fallback
qualityFrameFallbackCount=798
direct WHAM frames=12
```

The migrated native WHAM path ran its checkpoint-compatible SMPL initialization,
trajectory decoder, and SMPL decoder on CPU. The optional image-feature
integrator and camera/DPVO stages were not available locally, so the geometry
gate identified 798 finite-but-inconsistent frames and switched those frames
to the aligned MediaPipe world observation. Each switch is exported in
`qualityFrameFallbacks` with the exact median/p90 reprojection and collapse
reason; corresponding spans are emitted as `wham_geometry_fallback` uncertainty
intervals.

The inspected overlay is:

```text
C:\Users\k0ta0\AppData\Local\Temp\texmotion_full_wham_final_overlay.mp4
```

Frames 1, 101, 301, 501, 701, and 810 were sampled. The skeleton remained
inside the subject silhouette, and the HUD displayed
`WHAM -> MEDIAPIPE FALLBACK` on guarded frames. Raw MediaPipe world landmarks
are converted from camera axes (+Y down, +Z away) to canonical SMPL-X axes
exactly once before retargeting, while the overlay itself stays in image-space
coordinates. This keeps the fallback path upright without applying a second
WHAM/RTMPose conversion.

## Residual findings / exact blockers

1. **Full WHAM parity still depends on optional local assets.** The checkpoint,
   embedded SMPL buffers, trajectory decoder, and SMPL decoder run locally.
   Provisioning the image-feature backbone/archive and camera/DPVO assets is
   required to enable the image integrator, world-camera motion, and remaining
   official stages without the safety gate.

2. **Frame identity is explicit end to end.** Native WHAM rows include
   `frameIndex`/`timestampSec`; `PyTorchPoseBackend.detect_sequence` consumes
   those identities before using a legacy positional fallback. The adversarial
   test `test_pytorch_sequence_preserves_explicit_adapter_frame_identity`
   verifies reversed adapter output remains correctly timed.

3. **Safety-gated fallback is wired into selection.** When the alignment gate
   fails and MediaPipe has a finite aligned world row, the extractor retargets
   and renders that row, records `qualityFrameFallbacks`, and keeps WHAM
   diagnostics/provenance for review. If the auxiliary row is missing or
   non-finite, WHAM is retained and that reason is recorded.

The generated MP4/PNG and JSON outputs are disposable artifacts under the
system temp directory and are not repository changes.
