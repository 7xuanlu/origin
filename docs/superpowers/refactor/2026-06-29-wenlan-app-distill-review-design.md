# Wenlan App Distill Review Panel Design

- **Date:** 2026-06-29
- **Worktree:** `/Users/lucian/Repos/wenlan-app/.worktrees/wenlan-app-api-parity-audit`
- **Branch:** `codex/wenlan-app-distill-design`
- **Decision:** implement the review-first distill surface before any page rebuild action.

## Goal

Close the desktop app parity gap for the safe part of `/api/distill` by adding a
non-destructive Distill Review panel. The panel lets the user inspect daemon
distillation work that needs attention without rewriting pages or clearing
user-edited content.

This is the first distill checkpoint. It deliberately does not expose force
rebuild or `POST /api/distill/{page_id}`.

## Evidence

Current route diff:

```text
backend routes: 123
app source routes: 112
missing in app: 11
app-only routes: 0
```

The relevant missing routes are:

- `/api/distill`
- `/api/distill/{page_id}`

Backend behavior:

- `POST /api/distill` accepts `target?: string` and `force?: bool`.
- Normal `POST /api/distill` never invokes the daemon LLM for synthesis. It
  returns review data: `pending`, `stale_pages`, `stale_truncated`, and
  `orphan_topics`.
- If a non-empty `target` cannot be resolved, the backend returns a different
  hint payload without the normal review arrays.
- `force=true` clears `user_edited` on a resolved page before recompile.
- `POST /api/distill/{page_id}` also clears `user_edited` before re-distilling.

App behavior:

- `PageDetail` already shows page state, source memory evidence, links,
  unlinked mentions, and revision history.
- The app has established Rust client, Tauri command, and TypeScript wrapper
  seams for page-related routes.
- There is no callable distill wrapper or visible manual distill review surface.

## User Experience

Add a Distill Review entry point on Home near the existing `RefiningList`
section, then open a dedicated `Main` view such as
`{ kind: "distill-review" }`. This keeps the review queue in the same place as
other memory refinement work while avoiding extra weight in `PageDetail`.

The first version should be work-focused and quiet:

- A user-triggered refresh action runs the global `/api/distill` review pass.
- There is no target input in this checkpoint. Scoped distill can be designed
  later with a response union for unresolved-target hints.
- The panel renders three sections:
  - **Pending pages:** clusters not fully covered by an existing page.
  - **Stale pages:** pages with updated sources that may need attention.
  - **Unlinked topics:** repeated orphan page-link labels from the daemon.
- Empty sections stay collapsed or show a compact empty state.
- Rows are navigational or informational only:
  - Pending clusters show a deterministic label, source count, short source
    preview, existing page hint when present, and new-memory count when present.
  - Stale pages link to existing Page Detail.
  - Unlinked topics show label and mention count.
- No row performs page rewrite, synthesis, force rebuild, or page deletion.

The copy should avoid promising that the app can synthesize a page in this
checkpoint. The panel is a review queue, not an editor.

## Data Contract

Add app-local strict response types because `wenlan-types 0.9.2` does not expose
a stable typed distill response for this route yet.

Minimum TypeScript/Rust shape:

```text
DistillReviewResponse {
  pages_created: number
  scoped: boolean
  created_ids: string[]
  pending: DistillPendingCluster[]
  stale_pages: DistillStalePage[]
  stale_truncated: boolean
  orphan_topics: DistillOrphanTopic[]
}

DistillPendingCluster {
  source_ids: string[]
  contents: string[]
  entity_id?: string | null
  entity_name?: string | null
  space?: string | null
  estimated_tokens: number
  centroid_embedding?: number[] | null
  existing_page_id?: string | null
  existing_page_title?: string | null
  new_memory_count?: number | null
}

DistillStalePage {
  page_id: string
  title: string
  summary?: string | null
  source_memory_ids: string[]
  sources_updated_count?: number | null
  stale_reason?: string | null
  user_edited?: boolean | null
}

DistillOrphanTopic {
  label: string
  count: number
}
```

The implementation should preserve strict envelopes at the command boundary:
daemon shape drift should fail tests rather than silently rendering nonsense.
`centroid_embedding` is part of the current daemon payload and should be typed
but ignored by UI rendering.

Pending cluster display contract:

- Label priority: `existing_page_title`, then `entity_name`, then `space`, then
  the first non-empty `contents` item truncated to 72 characters, then
  `Untitled cluster`.
- Preview: show up to two non-empty `contents` snippets, each truncated to 140
  characters.
- Count copy: use `source_ids.length`; if `new_memory_count` is present, show it
  as the number of new source memories.

## Architecture

Use the existing app seam pattern:

1. `app/src/api.rs`
   - Add `WenlanClient::distill_review()`.
   - Use `POST /api/distill` with an empty JSON body or default request.
   - Do not send `target` or `force`.

2. `app/src/search.rs`
   - Add a `#[tauri::command] distill_review`.
   - Clone `WenlanClient` out of app state before awaiting.
   - Return the typed response directly.

3. `app/src/lib.rs`
   - Register the new command.

4. `src/lib/tauri.ts`
   - Add response interfaces and `distillReview()`.

5. Frontend component
   - Add `src/components/memory/DistillReviewPanel.tsx` as a dedicated routed
     view in `Main.tsx`.
   - Add a compact Home entry point near `RefiningList`.
   - Use a user-triggered React Query mutation or disabled query for loading,
     refresh, error, and retry states. Do not run the POST on mount, window
     focus, polling, or automatic retries.
   - Navigate stale pages into existing `PageDetail` instead of introducing a
     rebuild action.

## Error Handling

- If a refresh fails after a previous success, keep the last successful results
  visible and show a retryable error banner.
- If `stale_truncated` is true, show a small note that the daemon returned the
  first 10 stale pages and more may exist.
- Older daemon incompatibility should surface as a clear route/shape error, not
  as an empty queue.

## Testing

Use TDD before implementation.

Required RED tests before code:

- Rust API client posts to `/api/distill` and does not include `force`.
- Rust API client does not send `target`.
- Rust code has no wrapper or call path for `/api/distill/{page_id}` in this
  checkpoint.
- Rust response type deserializes a daemon-shaped payload containing pending,
  stale pages, orphan topics, `stale_truncated`, and `centroid_embedding`.
- Tauri command returns the typed response without holding a state guard across
  await.
- TypeScript wrapper invokes `distill_review` without `target` or `force`.
- Frontend renders the three sections, empty states, truncated note, and
  stale-page navigation.
- Frontend exposes no rebuild, synthesize, or force controls.
- Stale-page clicks only navigate to Page Detail.

Required verification after implementation:

- `pnpm test`
- `cargo test --manifest-path app/Cargo.toml`
- `pnpm build`
- `cargo clippy --manifest-path app/Cargo.toml --all-targets -- -D warnings`
- `pnpm refactor:api-routes --json`

Expected route diff after this checkpoint:

```text
backend routes: 123
app source routes: 113
missing in app: 10
app-only routes: 0
```

## Tool Boundary

- Use CodeGraph to locate impact surfaces before implementation:
  - `codegraph sync .`
  - `codegraph query WenlanClient --json`
  - `codegraph query PageDetail --json`
- Use local ast-grep (`sg`) for structural checks in Rust and TypeScript.
- Use `rust-analyzer` and Cargo tests as semantic validation.
- Use `rg` as the fallback when CodeGraph or ast-grep cannot express the query.

## Explicit Non-Goals

- No `force=true` UI.
- No scoped or targeted distill input.
- No `POST /api/distill/{page_id}` wrapper or button in this slice.
- No page synthesis editor.
- No relation/entity graph authoring.
- No WebSocket migration.
- No operator route panel for `/api/debug/pipeline`, `/api/steep`, or
  `/api/context`.

## Boule Review Questions

Run `/boule:debate` on this design before implementation planning. Review should
focus on:

- Whether the panel accidentally implies that the app can synthesize or rebuild
  pages in this slice.
- Whether any UI action can clear `user_edited` content without explicit future
  confirmation.
- Whether the app-local response types are strict enough to catch daemon route
  drift.
- Whether the dedicated Home-to-review view is better than embedding the review
  list into Page Detail.
- Whether pending clusters need a richer source-memory preview before they are
  useful.

## Approval State

Approved by user on 2026-06-29 with direction `1`: review panel first.
