# Timeline Editor redesign recommendation

Replace the feature stack with four task modes—**Pose, Repair, Timing, Polish**—inside a fixed workspace. Keep the avatar and timeline visible while the inspector presents exactly one selected joint or tool. Adding a feature must add a discoverable tool entry, never another permanent vertical section.

This is a source-based proposal, not an implemented or visually tested change. Reviewed `Editor/Motion/MotionTimelineEditorWindow.cs` and `DESIGN.md`; implementation should preserve the existing motion algorithms and preview lifecycle initially.

## Diagnosis grounded in the source

- `DrawPoseInspector` (line 1695) starts its scroll before the frame header, so even context and navigation disappear. Its `height` argument is not used to partition a fixed header, body, and footer.
- Hand/face, occlusion, advanced correction, and polish all precede joint groups (lines 1841–1905). Advanced tools, polish, occlusion, root, and torso default expanded. Selecting a joint only opens its body foldout (`FocusJointInInspector`, line 1462); it does not bring that joint into view.
- One Advanced Correction foldout exposes eight heterogeneous sections (line 2190): scope, tween, loop, offset, grounding, glitches, palette, retiming. Scope is configuration shared by operations, not another tool.
- Polish (line 2565) exposes nine filter mechanisms and their parameters together. Presets still leave the entire parameter matrix visible.
- Transport is repeated in the header (490), viewport HUD (1333), timeline (2749), and inspector frame stepping. The viewport HUD uses fixed coordinates whose controls can collide in narrow split views.
- Width clamps in `OnGUI` (353) cannot solve content density; the declared minimum is 850 × 640. Several horizontal control rows are already too ambitious for that size, especially localized labels.
- Actual rendering deviates from the contract: Apply to Avatar uses a hard-coded green background, polish uses teal, range edges use teal, and emoji-heavy headings compete with data. The frame inspector uses lime for Modified and green for Original, inconsistent with the specified timeline meanings.
- Operations have materially different scope semantics: occlusion uses an uncertainty interval or nearby frames; arm repair uses current ±3; loop blending operates on clip boundaries; grounding and palette range application loop over individual frames. A universal-looking scope control must not imply unsupported masks or ranges.

## Layout and interaction contract

Use three independent bounded regions: viewport, task inspector, bottom timeline. Only inspector tool content may scroll vertically; mode navigation, scope, and the operation footer remain anchored. Timeline scrolling stays independent and horizontal. Do not wrap the complete window or inspector chrome in a scroll view.

At a typical 1200 × 800, reserve 80px for the existing two-row header, 144px for the timeline, 8px gutters, and a resizable 340–420px inspector. At 850 × 640, use a 310px inspector and a 128px timeline; remaining height belongs to the viewport and inspector body. Size all rectangles from available bounds and clamp splitters rather than relying on nested minimum widths. These are proposed layout budgets to validate in Unity, not measured results.

```text
+--------------------------------------------------------------------------------------+
| walk_03 *   240 frames / 8.0s    Avatar [Hero v]       [Save .anim] [APPLY TO AVATAR]   |
| [|<] [<] [Play] [>] [>|] [Loop] [1.0x v]    048 / 240    [Undo] [Redo] [Revert...]     |
+------------------------------------------------------+-------------------------------+
| [Avatar | Video | Split]            [Overlays v] [Home]| Pose | Repair | Timing | Polish|
|                                                      +-------------------------------+
|                                                      | Target: Left forearm          |
|                                                      | Frame 048 / Modified          |
|                                                      +-------------------------------+
|                   AVATAR VIEWPORT                    | [Search joints...]            |
|                                                      | [Left arm v] [Forearm v]      |
|         click joint -> select; drag -> rotate         |                               |
|                                                      | Rotation                      |
|                                                      | X  --------o----   [ 12.0 deg] |
|                                                      | Y  -----o-------   [ -4.0 deg] |
|                                                      | Z  ------o------   [  2.0 deg] |
|                                                      |                               |
|                                                      | [Pose actions v] [Palette...] |
|                                                      | [Hand & Face...] [IK Pins...] |
|                                                      +-------------------------------+
|  Left forearm                          Pins: 2 [Edit] | Current frame / Left forearm  |
+------------------------------------------------------+-------------------------------+
| In [024] [Set]  Out [072] [Set]  49 frames   [All frames]      [Issues: 3] [Zoom - +]  |
|  001       024           048 |       072                  120                 240     |
| ----------------------------|------------------------------------------------------- |
| modified:      . . .        |      .       uncertainty:  =====                         |
+--------------------------------------------------------------------------------------+
```

In Repair/Timing/Polish, the inspector replaces joint content with this bounded template:

```text
| Pose | REPAIR | Timing | Polish |
| Tool [Foot grounding        v] |
| Scope [In-Out v] [24 .. 72]     | <- visible before execution
| Body: Feet (fixed)              |
|--------------------------------|
| Ground plane     [0.00 m]       |
| Method [Snap v]                 | <- only selected tool parameters
| > Advanced                      |
|                                |
|--------------------------------|
| Affects 49 frames / Feet        |
| [Run grounding]                | <- neutral outlined action
```

Use one compact tool chooser with categorized entries, optional search, and a remembered selection per mode. A tool chooser replaces page content; it never expands the list into another long form. Use a bounded results list for diagnostics. Avoid tool cards nested inside section cards nested inside inspector cards.

At narrow widths, clip name truncates with a tooltip; metadata moves to a clip-info popover, while Save, Apply, transport, Undo/Redo, and Revert remain reachable. Reserve the action widths first. Use compact labeled controls and tooltips for playback icons; move speed fine adjustment into its popup. Do not force horizontal scrolling of the header. Tabs remain one short row; localized labels may use a mode dropdown if they cannot fit. Below the supported minimum, retain clamped bounds rather than drawing overlapping controls.

## Categorized modes and complete feature mapping

| Surface | Category / selected page | Existing functionality |
|---|---|---|
| Pose | Joint | Root position; pelvis, torso/head, both arms and both legs; show only selected joint XYZ, or Root translation when selected |
| Pose | Pose actions | Copy, Paste, Mirror, Smooth current pose, Reset frame, T-pose; expose mask where actually supported |
| Pose | Palette | Eight pose slots, Capture/Clear, blend weight, current/range blend; select a slot, then show its controls |
| Pose | Hand & Face | Hand pose, expression, conditional intensity; explicitly label clip assistance and preserve existing live preview behavior |
| Pose | IK Pins | Enable solving, inspect active pins, discover current pin/unpin gestures; preserve current semantics until solver behavior is audited |
| Repair | Tracking ambiguity | Confidence/reason summary, swap crossing, left/right arm front/behind; show actual affected interval |
| Repair | Motion glitches | Scan, full bounded issue list, Go, Fix selected, Fix all; replace the current first-five-only display |
| Repair | Contact | Foot grounding and foot-position lock |
| Repair | Penetration | Armpit limiter |
| Repair | Offset | Additive root height and arm opening, fade edges |
| Timing | Tween | Existing interpolation/easing controls over In-Out |
| Timing | Loop | Blend clip start/end; explicitly display Entire clip / boundary margin |
| Timing | Retime | Existing range resampling and new frame count; show old/new duration |
| Polish | Presets | Action, Weight, Subtle; selecting a preset changes settings, execution remains explicit |
| Polish | Timing & spacing | Snap & Ease, Anime Frame Stepping, Keyframe Decimator |
| Polish | Weight & balance | Landing Cushion, Contrapposto |
| Polish | Overlap & drag | Kinematic Drag, Overshoot & Settling |
| Polish | Pose & silhouette | Pose Exaggeration, Trajectory Arc Smoother |
| Viewport overlays popup | Display | Skeleton, rotation gizmos, onion skin; secondary display options when available |
| Timeline | Shared temporal selection | In/Out fields, Set In/Out, Select All, zoom, uncertainty and modified-frame navigation |

Polish initially shows a preset selector, an enabled-filter summary, scope/mask, and the neutral Run Polish footer. A “Customize” control opens one of the four category pages; each filter is a compact enabled row and selecting it reveals its parameters. Enabled filters in other categories remain enabled. Display “Custom” after changing a preset parameter, and show all active filter names in an accessible summary. Do not invent a master-strength slider unless algorithm support is deliberately added later.

Joint selection should be immediate: clicking the avatar selects the joint and, in Pose mode, replaces the inspector target without scrolling. Provide searchable joint selection for joints difficult to click. During another mode, a viewport click updates selection without discarding tool parameters; an explicit “Edit joint” action enters Pose. Keep selection and operation mask separate and clearly named. A selected left wrist must never silently change a whole-body operation into a wrist-only operation.

Onion skin is a visualization, so it belongs in Overlays. IK Pins can change manipulation behavior, so its enable state and active-pin count remain visible while its editing controls live in Pose. Do not disguise solver activation as a visibility toggle. Preserve gizmo axis colors where they encode direction; do not use them as navigation accents.

## Visual rules translated from DESIGN.md

- Canvas Void `#08090a`; header/timeline Carbon `#0f1011`; viewport/inspector Obsidian `#161718`; structural borders Graphite `#23252a`. Use Smoke sparingly for focused boundaries.
- Acid Lime `#e4f222` fills only Apply to Avatar. Its playhead/current-frame role is retained as specified. Tabs and selected tools use neutral surface changes plus a thin underline and clear text. Run Repair/Polish remains a neutral ghost/outline button.
- Modified-frame markers use Pulse Green; uncertainty/glitch spans use Coral Red. Range selection uses a subtle neutral wash and labeled In/Out brackets, removing the extra teal signal. Original frame status is neutral. Add textual or geometric distinctions so color is not the only information channel.
- Use a device-aligned hairline (approximately one physical pixel, with DPI-aware drawing); avoid accumulating nested outlines. Apply 8px gutters, 8–12px padding, 4px internal spacing, 6px control corners and restrained 12px outer-card corners where existing cached styles support them.
- Replace emojis with consistent monochrome editor icons and localized labels. Paper/Mist for essential text, Fog for secondary copy; do not use low-contrast Ash for required editable values. Keyboard focus needs a visible neutral outline.
- Use compact 13px labels and readable numeric values, not the web document's hero typography. Prefer the existing editor font fallback unless appropriate fonts are already bundled; a typography asset migration is not a prerequisite.
- Keep current localized empty-state guidance and single lime Studio action when no clip is loaded. Error/backend details should be concise status summaries that open details, not permanent overlays covering the motion.

The Timeline-specific DESIGN.md rules take precedence over general token descriptions that describe green/red as decorative accents. Amend its inspector paragraph to describe mode routing and selected-tool disclosure instead of a single secondary-tools foldout. The UI/UX skill's generated generic website palette/layout is unsuitable here; the explicit project design contract governs this desktop editor.

## Concrete implementation architecture

Retain IMGUI for the first delivery. A simultaneous UI Toolkit and layout rewrite adds unnecessary migration scope; the cognitive improvement comes from routing and ownership, not a new rendering framework.

1. `MotionTimelineEditorWindow` remains lifecycle/composition owner: load data, editor update, preview resources, keyboard routing, save/apply. It computes header, viewport, inspector, timeline rectangles and draws each surface once.
2. `TimelineWorkspaceState` owns UI-only mode, selected tool per mode, selected joint, per-tool scroll, splitter position, and disclosure state. Persist suitable preferences with namespaced `EditorPrefs`; clip-relative frame/range and operation options remain session/document state as appropriate. Never persist stale frame indices globally.
3. `TimelineSelectionContext` exposes validated current frame, inclusive In/Out, selected joint, operation body mask, and resolved affected-frame count. Present frames as 1-based everywhere; convert exactly once at the model boundary. Retime and new clip loading must clamp/reset range and selection and invalidate diagnostics.
4. `TimelineInspectorRouter` draws fixed tabs and dispatches exactly one active panel: Pose, Repair, Timing, or Polish. Each panel draws its fixed context header, bounded content, and fixed footer; each tool has independent scroll state. No central call chain through every feature renderer remains.
5. A small `TimelineToolDescriptor` registry contains stable ID, localized name/category, supported scopes, mask capability, availability explanation, and parameter-drawing/execution adapters. Keep it simple and local; it is a feature index, not a generalized plugin framework.
6. A command adapter layer routes edits to existing `EditableMotionData` methods, refreshes preview once, invalidates affected diagnostics, and requests repaint. Audit the model's existing undo recording before introducing grouped transactions; do not blindly add another snapshot around methods already recording history. One range operation should eventually undo as one user action, especially current per-frame loops.
7. Reuse `MotionTimelineTheme` as the token source. Add cached tab/field/section styles there rather than scattering colors through new panels. Cache styles/textures and fixed GUI content; avoid new per-repaint resources. Use scoped enabled/color/indent handling to prevent one panel leaking GUI state into another.

Initially, tool descriptors must report what existing algorithms actually do. Occlusion can use a read-only “Detected interval” or “Nearby frames” scope; loop blending uses fixed clip boundaries; unsupported masks display the actual affected body parts. Expanding their algorithm scope is a separate functional change, not something to imply through UI consolidation.

Do not promise non-destructive previews in the first release. Existing tools mutate data; keep explicit execution and Undo. A later optional preview transaction can support Compare/Commit/Cancel using an isolated working copy and reliable cleanup on frame, tool, or clip changes.

## Implementation roadmap and acceptance

1. **Freeze behavior and inventory.** Map each current button to its model method, actual scope, mask, undo behavior, and preview refresh. Record representative editing scenarios and known limitations. Resolve the DESIGN.md inspector paragraph and color semantics before styling.
2. **Ship the bounded shell.** Introduce modes, fixed inspector chrome, one content scroll, and a single top transport row. Keep timeline selection/zoom below; remove duplicate viewport, inspector, and bottom transport. Move existing feature blocks behind mode/tool routing without changing algorithms. This delivers the largest immediate reduction in clutter.
3. **Make Pose direct.** Replace all-body slider stacks with selected-joint controls and searchable selection. Move palette, assistance, and pin controls to their selected pages. Replace `FocusJointInInspector` foldout expansion with explicit selection routing.
4. **Split operations into tools.** Break `DrawAdvancedPoseCorrectionTools` into individual adapters. Move range editing to the timeline and expose an accurate fixed scope summary per tool. Add bounded diagnostics navigation and revalidate state after retiming.
5. **Make Polish progressive.** Add preset-first content, active-filter summary, four customization categories, one filter detail at a time, and one neutral execute footer. Preserve all nine filters and current parameter ranges.
6. **Finish visual consistency and verify.** Remove ad hoc colored backgrounds/emoji, apply cached tokens, localize new strings, verify keyboard focus, document actions, preview cleanup, and empty/error states. Consider transactional preview only after the core release is stable.

Acceptance requires the following observable results:

- At 850 × 640 and 1200 × 800, tabs, scope, operation action, header actions, and timeline stay visible with no clipped/overlapping controls; repeat with Japanese labels and 100%/150%/200% display scaling.
- Clicking any selectable joint in Pose reveals its numeric fields without scrolling past tools. Default Pose content fits the minimum inspector body; long diagnostics or localization may use its bounded scroll.
- No tab contains an all-features accordion. Reaching any named operation takes a mode selection and a tool/category selection; changing the selected tool does not reset other tool parameters.
- Save and Apply preserve current behavior. Playback, scrubbing, viewport manipulation, video split, shortcuts, undo/redo, range boundaries, empty clips, missing avatar/video, and domain reload remain functional. Shortcuts do not fire while typing into numeric/search controls.
- Repair, timing, palette and polish affect exactly the displayed frame interval/body parts. Retime leaves valid frame/range indices and refreshed diagnostic results. Undo/redo restores an entire logical operation or an explicitly documented existing limitation is fixed before claiming that guarantee.
- Hidden panels cannot execute changes; changing mode/disclosure alone does not alter motion, assistance, enabled polish filters, or IK state. No per-frame GUIStyle/texture allocations are introduced.

No production source changes or Unity tests were performed for this analysis task. The next deliverable should be the shell plus selected-joint interaction, reviewed at the actual minimum window size before the remaining panels are migrated.
