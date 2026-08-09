# AGENTS.md - src/components/memory/

## OVERVIEW

Main product UI cluster: home, search, spaces, memories, pages, citations,
sources, settings, remote access, imports, profile, and review surfaces.

## STRUCTURE

```text
src/components/memory/
|-- Main.tsx             # navigation shell, view state, invalidations
|-- HomePage.tsx         # overview/review lanes
|-- MemoryDetail.tsx     # memory detail and enrichment status
|-- PageDetail.tsx       # distilled page detail shell
|-- page/                # citations, related pages, page metadata
|-- settings/            # diagnostics/settings side panels
|-- sources/             # source add/list UI
`-- __tests__/           # grouped component suites
```

## WHERE TO LOOK

| Task | Location | Notes |
| --- | --- | --- |
| Navigation or view state | `Main.tsx` | high blast radius; many query invalidations |
| Page citations | `PageDetail.tsx`, `page/PageInfo.tsx`, `page/CitationChip.tsx` | verified/unverified states and source ordering |
| Settings/config | `SettingsPage.tsx`, `settings/` | profile, capture, sources, diagnostics, remote access |
| Sources UI | `SourcesView.tsx`, `sources/` | add/list/sync source workflows |
| Memory rendering | `MemoryCard.tsx`, `ContentRenderer.tsx`, `MemoryDetail.tsx` | classifier/rendering invariants |
| Remote access UI | `RemoteAccessPanel.tsx` | talks to Rust remote-access commands |
| Review lanes | `DistillReviewPanel.tsx`, `RefiningList.tsx`, `WorthAGlanceScroll.tsx` | pending/refinement flows |

## CONVENTIONS

- Preserve citation diagnosability. Existing tests cover verified/unverified
  chips, page-source ordering, missing/mismatched citation states, and popovers.
- Keep settings mutations going through the shared Tauri client wrappers; the
  config-boundary test exists to catch local bypasses.
- Treat `Main.tsx` changes as cross-view changes. Verify the affected view plus
  any query invalidation or keyboard shortcut behavior you touched.
- Visible fixed copy still belongs in `src/i18n/resources.ts`, not inline JSX.
- Prefer focused component tests near the edited file; many invariants here are
  easier to lock with Testing Library than with snapshots.

## ANTI-PATTERNS

- Do not hide unverified citations or collapse them into verified styling.
- Do not add navigation state that depends on localized placeholder text.
- Do not weaken `PageDetail.*.test.tsx`, `page/PageInfo.test.tsx`, or
  `SettingsPage.config-boundary.test.ts` to make UI changes pass.
- Do not treat `RefiningList.tsx` TODOs as cleanup-only; they mark unresolved
  behavior.

## COMMANDS

```bash
pnpm vitest run src/components/memory
pnpm vitest run src/components/memory/page
pnpm test:i18n
pnpm exec tsc -b
```
