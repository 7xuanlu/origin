# AGENTS.md - src/components/memory/

## OVERVIEW

Main product UI cluster: home, search, spaces, memories, pages, citations,
sources, settings, remote access, imports, profile, and review surfaces.

## WHERE TO LOOK

| Task | Location | Notes |
| --- | --- | --- |
| Navigation or view state | `Main.tsx` | high blast radius; many query invalidations |
| Page citations | `PageDetail.tsx`, `page/PageInfo.tsx`, `page/CitationChip.tsx` | verified/unverified states and source ordering |
| Settings/config | `SettingsPage.tsx`, `settings/` | profile, capture, sources, diagnostics, remote access |
| Sources UI | `SourcesView.tsx`, `sources/` | add/list/sync source workflows |
| Memory rendering | `MemoryCard.tsx`, `ContentRenderer.tsx`, `MemoryDetail.tsx` | classifier/rendering invariants |
| Remote access UI | `RemoteAccessPanel.tsx` | talks to Rust remote-access commands |
| Review lanes | `DistillReviewPanel.tsx`, `HomePage.tsx` needs-review rail | pending/refinement flows |

## CONVENTIONS

- Preserve citation diagnosability. Existing tests cover verified/unverified
  chips, page-source ordering, missing/mismatched citation states, and popovers.
- Keep settings mutations going through the shared Tauri client wrappers.
- Treat `Main.tsx` changes as cross-view changes. Verify the affected view plus
  any query invalidation or keyboard shortcut behavior you touched.
- Prefer focused component tests near the edited file; many invariants here are
  easier to lock with Testing Library than with snapshots.

## ANTI-PATTERNS

- Do not hide unverified citations or collapse them into verified styling.
- Do not add navigation state that depends on localized placeholder text.
- Do not weaken `PageDetail.*.test.tsx` or `page/PageInfo.test.tsx` to make UI
  changes pass.
- Do not reintroduce a second home layout. `HomePage.tsx` always renders
  `WikiHome`; a library with no pages gets the empty state inside the page
  slot, never a separate greeting screen.
- Do not gate that empty state on the `intelligence-ready` milestone. It is a
  permanent latch, so it keeps claiming a provider that has since been removed
  and denies one that was configured a minute ago. Read the live provider
  through `useProviderConfigured`.
- Do not hoist that read above the empty state. The three commands are settings
  IPC and Home also renders in the fixture-only Review flavor, whose command
  contract (`review/commandCapabilities.ts`) fails closed on them — so calling
  the hook in `HomePage` rejects three commands on every Review home and breaks
  `e2e/review-flavor.review.spec.ts`. Same rule for any other settings read: ask
  from the component whose copy branches on the answer.
- Do not put a retrieval list ("Where AI looked") back on the home surface.
  Agents reading from the library are the Activity view's subject
  (`ActivityFeed.tsx`, route kind `activity`); home stays the library's own
  surface. `HomePage.redesign.test.tsx` asserts the section is absent even when
  the daemon has retrieval events. Recaps remain reachable from Memories; do
  not make their entry point depend on a retrieval surface.
- Do not promise pages will compile when no provider is configured. Page
  synthesis is LLM-gated, so that copy is a lie without a local model or an
  API key.

## COMMANDS

```bash
pnpm vitest run src/components/memory
pnpm vitest run src/components/memory/page
pnpm test:i18n
pnpm exec tsc -b
```
