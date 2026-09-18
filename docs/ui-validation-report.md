# TexMotion UI/Test Validation Report

Date: 2026-09-17 (Asia/Tokyo)

Scope: `DESIGN.md`, the existing `Editor/Video/tests` suite, and the changed
Unity UI/localization sources (`TexMotionWindow.cs`,
`MotionTimelineEditorWindow.cs`, `MotionTimelineTheme.cs`, and
`TexMotionLocalization.cs`). The design reference describes a dark, compact
Linear-style surface with hairline borders and a restrained acid-lime primary
action. The checks below validate source contracts; visual fidelity still needs
a Unity Editor pass.

## Added validation

`tests/validate_ui_static.py` is a report-only source check by default. It
checks LIFO ordering for IMGUI Begin/End groups, reports missing Japanese keys
for `TrLiteral`/`TrFormat`, and inventories direct user-facing IMGUI literals.
Use `--strict` to fail on deterministic layout or localization findings.

## Results

- The four top-level tabs use a responsive equal-width layout, so localized
  labels stay inside the window at narrow widths. Video Motion keeps Target FPS,
  the backend dropdown, and the extraction action visible; secondary trim,
  quality, assistance, and Settings surfaces use persisted foldouts.
- The Timeline Editor now uses cached `MotionTimelineTheme` tokens for Void,
  Carbon, Obsidian, Graphite, and the documented accents. Its header is split
  into identity/action and playback rows, viewport/inspector/timeline are
  independently surfaced cards, and uncertainty/current/modified frames use
  Coral Red/Acid Lime/Pulse Green signals.

- `python tests/validate_ui_static.py --strict`: passed; 0 deterministic IMGUI
  layout findings and 0 uncovered `TrLiteral`/`TrFormat` Japanese keys. The
  validator reports one intentional direct Japanese Modular Avatar prompt for
  manual review.
- `python -m compileall -q Editor/Video tests`: passed (exit 0).
- `python -m pytest -q Editor/Video/tests Editor/Video/pose_pipeline`: 121
  passed in 12.41s.
- Roslyn syntax parsing passed for `TexMotionWindow.cs`,
  `MotionTimelineEditorWindow.cs`, `TexMotionLocalization.cs`,
  `VideoModelDownloader.cs`, `PythonEnvironmentManager.cs`,
  `VideoMotionJobRunner.cs`, and `ModelDownloader.cs`.
- `ruff check Editor/Video tests`: exit 1 with 53 existing style/lint errors,
  primarily unused imports/locals and E701 one-line statements in the pose
  pipeline. No lint edits were made because they are outside this focused UI
  validation scope.

## Blockers and follow-up

The Unity Editor executable and a `ProjectSettings` directory are not present
in this package workspace, so a full Unity assembly compile and screenshot/
manual IMGUI rendering pass could not be run. The source-level checks are
clean; exercise the four tabs at narrow and normal window widths in Unity when
the project is opened in the Editor.
