# Adversarial UI and regression audit

Audit scope: `DESIGN.md`, `Editor/TexMotionWindow.cs`,
`Editor/TexMotionSettings.cs`, `Editor/Localization/TexMotionLocalization.cs`,
and the Python tests under `Editor/Video/tests`.

## Findings

### Medium — language switching still has secondary strings to cover

The localization completion pass now provides Japanese entries for the major
Generator, Video Motion, Library, Settings, header, status, and timeline labels
and wires 40 visible call sites through `TrLiteral`; English remains the safe
fallback.  A static scan still finds secondary labels, dialogs, dynamic status
messages, and model descriptions that are intentionally or temporarily
English.  Add keys incrementally when those strings become user-facing
requirements, and keep representative strings from each tab covered by UI
smoke checks.

### Low — some navigation metadata remains English

The tab bar now localizes Generator, Video Motion, Library, and Settings routes
through the shared facade.  Technical metadata such as enum names and model
descriptions can remain English until a product terminology glossary is
approved.  A Unity editor smoke test (or a source contract test that does not
load Unity) should assert the four route identifiers and localized labels stay
reachable.

### Medium — settings persistence is not directly regression-tested

`TexMotionSettings` is a `ScriptableSingleton` and exposes the shared
`VideoBackend`, quality-model paths, Python environment settings, and language.
The existing Python suite cannot instantiate Unity settings, and no static or
editor test currently verifies that the Video Motion controls and Settings tab
read/write the same fields.  A Unity EditMode test should mutate each shared
field, call `Save`, reload, and verify the value; this is especially important
to preserve existing functionality during UI reorganization.

### Low — design tokens are not enforceable in the IMGUI surface

`DESIGN.md` specifies Linear-style dark surfaces, compact spacing, restrained
weights, and an acid-lime primary action.  The window now caches those surfaces,
hairline borders, tabs, cards, badges, and primary action styles; some legacy
`EditorStyles.boldLabel` controls remain by design.  There is no automated
visual/token contract, so keep this as a manual Unity visual QA item.

## Existing safeguards verified by inspection

The Python tests already exercise backend selection, missing-model fallback,
requested/actual backend metadata, per-joint fusion provenance, and synthetic
CLI output.  These are the safest current regression boundary because they run
without Unity and should remain green while the editor UI changes.

## Validation

Commands run from repository root:

```text
pytest Editor/Video/tests
python -m py_compile Editor/Video/video_pose_extractor.py Editor/Video/pose_pipeline/*.py Editor/Video/pose_pipeline/backends/*.py Editor/Video/pose_pipeline/adapters/*.py
```

Results: `python -m pytest -q Editor/Video/tests Editor/Video/pose_pipeline`
passed 121 tests; explicit PowerShell-expanded `py_compile` passed; and
`git diff --check` passed.  The report itself intentionally adds no executable
Unity tests, because settings persistence still requires an EditMode fixture and
source-text assertions would encode implementation details rather than behavior.

