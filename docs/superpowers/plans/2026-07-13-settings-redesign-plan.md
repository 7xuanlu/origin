# Settings redesign — plan (2026-07-13)

Design read: settings surface of a local-first personal knowledge desktop app;
calm, literary-technical (serif display, mono machine values); Linear-grade
precision executed entirely inside the existing `--mem-*` system. Mode:
**redesign-preserve** — the token system (Fraunces / Instrument Sans /
JetBrains Mono, ink-navy dark / cool-paper light, indigo accent, radius lock,
two-band type scale) is the brand and stays. The defects are in how sections
*use* the system, not the system.

## Ranked weaknesses (verified by reading every file at 4e63c86)

Tier 1 — structural / immediately visible:

1. **Duplicate headings.** The page `h1` names the active group, then the first
   `SectionHeader` repeats it: Intelligence renders "Intelligence" twice
   stacked (`IntelligenceSection.tsx:14-21`); Sources renders "Import
   memories" as eyebrow AND as row title 40px apart (`SourcesSection.tsx:14`
   vs `:21`); General repeats the gear icon already in the sidebar.
2. **Eyebrow-per-card monotony.** Every card cluster gets icon + mono
   uppercase eyebrow (4× in Agents alone). Icons duplicate the sidebar's
   iconography; rhythm reads templated and dilutes the eyebrows that matter.
3. **Sources list is off-system.** `memory/sources/SourcesSection.tsx`
   (SourcesList, 396 lines) uses zero shared primitives, raw Tailwind palette
   colors (`emerald-*`, `red-*`, `bg-black/50`) bypassing every
   `--mem-status-*` token, emoji glyphs (↻ ⋯ ✓ →) against the app's SVG icon
   language, and one-sentence metadata walls.
4. **Theme switcher weight.** Accent-filled sliding thumb with fragile
   `translateX(calc(...))` math (`GeneralSection.tsx:248-257`) — the loudest
   control on the page for a tertiary choice. Premium desktop apps use a
   neutral raised thumb.
5. **`window.confirm` for "re-run setup"** (`GeneralSection.tsx:308`) — native
   OS dialog inside an otherwise fully token-styled surface.
6. **Trust-level explainer wall.** Permanent 4-row legend above the agents
   list (`AgentsSection.tsx:63-104`), shown even with zero agents.

Tier 2 — consistency debt (mechanical, high propagation value):

7. Hand-rolled card shells in 6 of 8 embedded components (RemoteAccessPanel,
   WebPlatformCards, ClientRow, AnthropicFields guidance box, SourcesList,
   ImportFlow strip) — `Card` changes propagate to none of them.
8. Hardcoded font sizes (`"10px"`–`"13px"`) in every embedded component except
   ActiveIntelligenceStrip; worst: IntelligenceSetup (7×), AnyProviderCard,
   RemoteAccessPanel. Also `SettingsPage.tsx:44` (`13px`) and `:64` (`11px`).
9. AnyProviderCard preset pills hand-roll status dots (`●`/`○`/`…`) beside a
   StatusChip system built exactly for that vocabulary.
10. Duplicated warning-triangle SVG (primitives SettingRow + RemoteAccessPanel);
    ImportFlow ships a private `@keyframes pulse` duplicating `mem-node-pulse`.

Tier 3 — polish gaps:

11. No skeleton loading anywhere (Diagnostics/Agents show bare "Loading…"
    text); `animate-pulse` is already available.
12. Empty states are plain centered text (Agents) or a dashed box (Sources).
13. Agent rows: unlabeled trash icon button (violates the repo's own a11y
    floor), crowded Select + Toggle + delete cluster, tiny two-step confirm.
14. Page header underweight (20px serif + hardcoded 13px hint); privacy footer
    at hardcoded 11px with opacity-on-tertiary double-muting.
15. Diagnostics wire state is a flat list although the data is a topology
    (daemon → MCP binary → clients).

## Design decisions

- **Header**: group title moves up to `--mem-text-2xl` Fraunces (parity with
  wizard step titles), hint on `--mem-text-sm` token. Back arrow stays (keyboard
  muscle memory) but aligns to the title block grid.
- **Heading rule**: the page h1 names the group; a `SectionHeader` appears only
  when a group has ≥2 clusters and never repeats the group name. SectionHeader
  loses its icon slot usage in settings (type carries hierarchy; sidebar
  carries iconography). Intelligence loses its self-titled header; Sources
  loses the "Import memories" eyebrow.
- **New primitives** (all in `settings/primitives.tsx`, tokens only):
  - `SegmentedControl` — neutral raised thumb (`--mem-surface` +
    `--mem-shadow-raised` + border), radiogroup semantics, replaces the theme
    switcher; reduced-motion safe (thumb is the button's own background, no
    translate animation needed — each segment is a button, active one raised).
  - `Skeleton` — `animate-pulse` bars on `--mem-hover` fill, radius-sm;
    `aria-hidden`, container carries the loading semantics.
  - `ConfirmActionButton` — generalizes AgentsSection's inline two-step
    confirm (danger tone, auto-reset on blur/timeout); replaces
    `window.confirm` in GeneralSection and the agent-row delete cluster.
  - `Button` gains `active:scale-[0.98]` tactile press (one class).
- **Agents**: trust explainer folds into a native `<details>` disclosure line
  under the header ("What trust levels mean"); agent rows get labeled controls
  (`aria-label` on delete), pending-client chip switches to the Tag/StatusChip
  vocabulary; "Set up another tool" becomes a ghost `Button`.
- **Sources (settings group)**: SourcesList brought on-system — status tokens
  replace emerald/red raws, SVG icons replace glyph emoji, rows/cards/buttons
  move onto primitives, metadata becomes a structured mono meta-line. Logic,
  dialogs, and data flow untouched.
- **Intelligence**: dup header dropped; hardcoded sizes → tokens; preset-pill
  dots aligned to StatusChip dot classes (they DO carry probe semantics, so
  they stay — as tokens, not glyphs).
- **Diagnostics — the signature.** The wiring card renders the real topology
  as the app's existing rail motif: daemon → MCP binary → clients as nodes on
  a vertical rail (sage ink for healthy segments, `mem-node-pulse` while
  probing, danger tint on a broken hop). Reuses the onboarding rail's
  animation vocabulary — the one place the settings page is memorable, and it
  encodes true structure, not decoration. All existing content (URLs,
  candidates, chips, copy-report) stays, hung off the rail. Skeletons for
  loading.
- **Embedded components** (RemoteAccessPanel, WebPlatformCards, ClientRow,
  ClientSetupList, IntelligenceSetup, AnyProviderCard, ImportFlow/DropZone):
  mechanical alignment only — Card shells, token sizes, shared warning icon,
  drop the private keyframes. No layout rework.

## Non-goals

No new dependencies. No i18n key removals (additions in en/zh-Hans/zh-Hant
with exact parity). No changes to settings logic, IPC, dialogs' behavior, or
information architecture (sidebar groups unchanged). No daemon work. The
standalone Sources page (`SourcesView.tsx`) is out of scope.

## Verification floor

`pnpm exec tsc -b` · `pnpm test` · `pnpm test:i18n` · preview screenshots of
all 5 sections × dark/light via `:1421?mode=settings&section=…&theme=…&bar=0`.
Any load-bearing test added/changed gets mutation-proved (break product code,
watch it fail, paste failure, targeted revert). Draft PR from
`worktree-settings-redesign`; never merge.
