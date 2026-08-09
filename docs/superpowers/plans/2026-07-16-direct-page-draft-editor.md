# Direct Page draft editor and microcontrol convergence implementation plan

> Execute only in `/Users/lucian/.codex/worktrees/f8d2/wenlan-app`. Preserve every unrelated dirty change. Use RED-first tests. Do not install Figma, edit `.wenlan-backend-version` without a published checksummed daemon asset, push, publish, deploy, or touch production data.

**Goal:** Replace the New Page modal with a direct, durable Page editor; make drafts discoverable in Wiki; preserve Page as first-class with optional Space; and make the Wiki/Spaces/Page microcontrols visibly consistent without changing Wenlan's typography, 72px main gutter, 72px Wiki rows, 64px Space rows, or 730px reading measure.

**Architecture:** `Main` gains a first-class `page-draft` view. Wiki opens it with no Space and Space detail opens it with the current Space. A dedicated `PageDraftEditor` serializes debounced snapshots through the new daemon lifecycle so its own writes cannot race; no server row exists until title or body is meaningful. Publish replaces the editor route with Page detail while preserving the originating Wiki/Space in history. Wiki reads active and draft Pages separately, combines them for inventory, and routes draft rows back to the editor. Preview and E2E runtimes remain synthetic and never mutate production data.

**Design contract:**

- Editor title uses the existing object-detail hierarchy, not the 34px destination title.
- Title autofocuses; body is the primary writing surface; Space is a compact nullable selector with no “Optional” copy.
- Autosave states are quiet `Saving`, `Saved`, and actionable error copy; there is no Save button or creation form.
- Publish is the sole explicit lifecycle action and is enabled only with non-empty title and body.
- A publish title conflict offers `Open existing` and `Rename draft`; it never merges content.
- The global 72px view gutter stays unchanged. The editor aligns to a 730px writing axis, creating local right-side breathing room.
- Text/header controls are 34px high, 6px radius, 13px; quiet icon actions are 32×32 with 16px glyphs; menus share one 6px-radius popover grammar.

---

### Task 1: Lock the Tauri and mock contract RED

**Files:**

- Modify: `src/lib/tauri.test.ts`
- Modify: `src/lib/tauri.ts`
- Modify: `app/src/search.rs`
- Modify: `app/src/lib.rs`
- Modify: `preview/mocks/live-invoke.test.ts`
- Modify: `preview/mocks/live-invoke.ts`
- Modify: `e2e/tauriMock/runtime.ts`
- Modify: `src/reviewTauriCore.test.ts`

**Interfaces:**

- `createPageDraft({ clientDraftId, title, content, space }) -> Page`
- `updatePageDraft({ id, expectedVersion, title, content, space }) -> Page`
- `publishPageDraft({ id, expectedVersion }) -> Page`
- `discardPageDraft({ id, expectedVersion }) -> void`
- `PageDraftApiError` preserves structured conflict code and details across Tauri's string error boundary.
- `clientDraftId` is a stable `page_<uuid>` generated once per editor session. Rust serializes it as `draft_id`; the daemon must treat a repeated create with the same ID as the same draft so an ambiguous network retry cannot duplicate Pages.

- [x] Write TS wrapper tests proving body whitespace is preserved, Space is normalized to nullable, version fields use Tauri camelCase, and structured daemon 409 JSON becomes `PageDraftApiError`.
- [x] Write Rust command tests for exact HTTP payloads/routes plus response unwrapping; keep local DTOs so the app does not depend on an unpublished `wenlan-types`.
- [x] Write Preview and E2E runtime tests for partial snapshots, CAS, discard, publish, and explicit title conflict.
- [x] Write an ambiguous-response retry test proving the same client-generated draft ID is reused and creates only one draft.
- [x] Reverse the E2E mock drift: legacy `create_page` always mints like the real authored route and never silently appends to an exact-title Page.
- [x] Run the focused TS/Rust suites and inspect the expected missing-command failures.
- [x] Implement and register the four commands; keep Preview drafts in the existing synthetic in-memory map and never proxy draft writes to the live daemon.
- [x] Re-run focused contract tests to green.

### Task 2: Build the serialized autosave editor RED→GREEN

**Files:**

- Create: `src/components/memory/pages/usePageDraftAutosave.ts`
- Create: `src/components/memory/pages/usePageDraftAutosave.test.ts`
- Create: `src/components/memory/pages/PageDraftEditor.tsx`
- Create: `src/components/memory/pages/PageDraftEditor.test.tsx`
- Create: `src/components/memory/pages/pageDraftEditor.css`
- Create: `src/components/memory/pages/pageActions.css`
- Modify: `src/i18n/resources.ts`
- Modify: `src/i18n/resources.test.ts`

- [x] Test title autofocus, Wiki null Space, current-Space preselection, clearable Space, and no “Optional” copy.
- [x] Test that a resumed draft hydrates through `getPage(draftId)`, keeps its selected Space even when it is absent from `listSpaces`, and renders explicit loading, missing, and load-error states.
- [x] Test no write for untouched or Space-only state.
- [x] Test title-only/body-only first persistence after the debounce and raw body whitespace preservation.
- [x] Test serialized writes: edits made during an in-flight create/update produce one ordered follow-up snapshot and never a self-inflicted CAS conflict.
- [x] Test both-empty cleanup of an existing draft, network-error field preservation, and retry.
- [x] Reconcile ambiguous create responses against the returned server snapshot, then CAS-update any newer local edit instead of falsely marking it saved.
- [x] Treat `page_draft_not_found` on a discard retry as an already-completed delete, rotate the client ID, and allow the next edit to create a fresh draft.
- [x] Rotate a collided client-generated ID once, and reconcile an ambiguous publish by reading the same Page ID before surfacing an error.
- [x] Test that Back and global Escape await an editor-owned flush gate before navigation; do not claim an async flush from passive unmount cleanup.
- [x] Test stale-version conflicts as a blocking state with an explicit reload-latest recovery; generic retry must never repeat the same stale `expectedVersion`.
- [x] Test Publish disabled until title and body are meaningful, flush-before-publish, same-id transition, and route replacement.
- [x] Test title conflict actions: Open existing routes without deleting the draft; Rename clears the conflict and focuses/selects the title.
- [x] Run focused hook/component tests and inspect the expected missing-file failures.
- [x] Implement the minimal editor and serialized autosave loop with a 700ms idle debounce, mounted-state guards, and React Query invalidation for `["pages"]`, active/draft inventory, Space counts, and the published Page.
- [x] Add English, Simplified Chinese, and Traditional Chinese copy and re-run i18n parity.
- [x] Move the shared `.page-create-action` contract into `pageActions.css` before deleting the modal-owned stylesheet in Task 3.

### Task 3: Lock direct navigation and inventory semantics RED

**Files:**

- Modify: `src/components/memory/navigation/viewState.test.ts`
- Modify: `src/components/memory/navigation/viewState.ts`
- Modify: `src/components/memory/pages/listAllPages.test.ts`
- Modify: `src/components/memory/pages/listAllPages.ts`
- Modify: `src/components/memory/pages/PagesOverview.test.tsx`
- Modify: `src/components/memory/pages/PagesOverview.tsx`
- Modify: `src/components/memory/space-detail/SpaceDetail.dossier.test.tsx`
- Modify: `src/components/memory/SpaceDetail.tsx`
- Modify: `src/components/memory/Main.search.test.tsx`
- Modify: `src/components/memory/Main.tsx`
- Delete: `src/components/memory/pages/PageComposerDialog.test.tsx`
- Delete: `src/components/memory/pages/PageComposerDialog.tsx`
- Delete: `src/components/memory/pages/pageComposer.css`

**View contract:**

- `{ kind: "page-draft"; draftId?: string; space: string | null }`

- [x] Test that `page-draft` keeps Wiki selected.
- [x] Test that Wiki New Page calls `onCreatePage(null)` and renders no dialog.
- [x] Test that Space detail New Page calls `onCreatePage(currentSpace)`.
- [x] Preserve `listAllActivePages` for current consumers; add or parameterize a draft helper with pagination and dedupe coverage.
- [x] Test that active and draft lists are fetched separately, draft rows show only a Draft marker, untitled drafts use localized fallback copy, active rows route to Page detail, and draft rows route to the editor.
- [x] Test Main history: Wiki/Space → editor → Back returns to origin; publish/open-existing replaces the editor so Page Back also returns to origin.
- [x] Route global Escape through the same editor-owned async flush gate as Back.
- [x] Run focused tests and inspect expected prop/view failures.
- [x] Implement the view and routing seam, combine inventory rows without conflating `status=draft` with `review_status=unconfirmed`, and remove the obsolete modal ownership only after the editor and shared action stylesheet exist.
- [x] Re-run focused navigation/inventory tests to green.

### Task 4: Converge microcontrols without changing layout typography RED→GREEN

**Files:**

- Create: `src/components/memory/pages/wikiSpaceControls.test.ts`
- Modify: `src/index.css`
- Modify: `src/components/memory/navigation/navigation-shell.css`
- Modify: `src/components/memory/spaces/spaces.css`
- Modify: `src/components/memory/spaces/spacesInventory.css`
- Modify: `src/components/memory/spaces/SpacesOverview.tsx`
- Modify: `src/components/memory/spaces/SpaceRow.tsx`
- Modify: `src/components/memory/space-detail/space-detail-header.css`
- Modify: `src/components/memory/space-detail/SpaceDossierHeader.tsx`
- Modify: `src/components/memory/PageDetail.tsx`
- Modify focused component tests beside those files

- [x] Lock shared `--mem-control-height: 34px`, `--mem-icon-action-size: 32px`, `--mem-icon-glyph-size: 16px`, 6px radius, 13px text, and common menu surface/border/shadow.
- [x] Lock unchanged 72px gutter, 72px Wiki row, 64px Space row, 730px Page measure, current font families, and destination/object title tokens.
- [x] Make New Page and New Space the same text-only quiet outlined control; remove the metric-dependent `+` glyph from New Space.
- [x] Normalize Wiki filters, Spaces search, Space row/detail actions, Page detail actions, and their focus states to the shared contracts.
- [x] Give Page export, Space row, and Space detail menus common semantics, Escape/focus return, and common visual grammar.
- [x] Run focused CSS/component tests, inspect expected failures, implement surgical classes/tokens, and re-run to green.

### Task 5: Integration and rendered verification

- [x] Run focused Page draft, Wiki, Space, navigation, Page detail, controls, mock, and i18n Vitest suites.
- [x] Run `pnpm test`, `pnpm test:i18n`, `pnpm build`, `cd app && cargo test`, `cargo fmt --check --all`, and `git diff --check`.
- [x] Build/run against `/Users/lucian/Repos/wenlan/.worktrees/page-draft-lifecycle` via `WENLAN_BACKEND_DIR`; do not edit the release pin.
- [x] Clean-restart only the fixture-isolated Review/browser preview; capture Wiki with drafts, new standalone editor, current-Space editor, saving/saved/error, collision, published Page, Spaces, and Page detail at desktop plus compact width.
- [x] Run inline Visual QA twice: first for hierarchy/spacing/state comprehension, then for regressions/accessibility/responsiveness.
- [x] Ask Fable for a fresh-eye design-lead verdict from the final captures; fix every critical/high-confidence issue and re-run affected gates.
- [x] Ask an independent code reviewer to inspect autosave serialization, unmount behavior, CAS/conflict propagation, mock-production parity, and unrelated dirty-change safety.
- [x] Leave `.wenlan-backend-version` unchanged unless a published `wenlan-darwin-arm64.tar.gz` exists and its sha256 plus app version lockstep can be verified; report that release gate explicitly rather than fabricating a pin.
- [x] Keep the app PR in draft until the daemon accepts `draft_id` on `POST /api/pages/drafts`, returns the existing draft for a repeated ID, and ships the whole Page draft lifecycle in a checksummed release asset.
