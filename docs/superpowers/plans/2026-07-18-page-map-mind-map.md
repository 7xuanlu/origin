# Page Map (mind map) — carve-out for its own session

**Status:** not started — deliberately excluded from the KG redesign branch
(PR #90). This file is the hand-off so a future session starts with the full
picture instead of re-deriving it. Tooling settled 2026-07-18 (see "Tooling
decision" below); direction fork settled 2026-07-18: **hybrid**;
council-reviewed 2026-07-18: **approve-with-changes**, changes folded in.
Mockup approved 2026-07-18 (Lucian) with two follow-ups, both folded in
below: the edge-anchoring criterion and the full CRUD surface (edge ops
were the gap). Next steps: radial-layout spike, then the daemon API work in
`7xuanlu/wenlan`.

**Design source of truth:** "Knowledge Atlas — design mockups" artifact,
screen 03 ("Page Map — AI drafts, you curate"). Screens 01 (Atlas) and
02 (Focus) shipped on `worktree-kg-foundation`.

## What screen 03 specifies

A **Map tab on every wiki page** (Read | Map | Sources), rendering the page as
a spatial outline the AI drafts and the user curates:

- Center node = the page ("Wenlan Architecture · 12 sources · updated 3d
  ago"), first-ring nodes = its sections/claims, leaves = supporting items
  with citation counts (`[n]` badges open the source).
- **A view, not a copy** — every node points at a real claim, citation,
  entity, memory, or page. No free-floating text. This is the grounding
  NotebookLM has and editable canvas tools (Xmind, Whimsical, Miro) don't.
- **Per-node ghost suggestions** — "✦ suggested · from 3 memories" nodes and
  "✦ suggested link" edges render ghosted with per-node ✓ Accept / ✕ Dismiss.
  Surveyed products only review whole maps; per-node review is Wenlan's
  review-queue DNA on a canvas.
- **Your edits are pinned** — once touched, a node is curated; AI never
  rewrites an accepted node, it only proposes new ones.
- **"✦ Improve map"** runs an on-demand AI pass (toolbar button, next to
  "Suggestions (2)" and a "Scope: this page ▾" selector).
- **Cross-page maps** reuse the same canvas at space level — hubs are pages,
  connectors are shared entities and wikilinks.
- Toolbar: Outline toggle, zoom − 100% +, suggestion count, scope selector.

## Direction fork — SETTLED 2026-07-18: hybrid

Lucian's mental model is **an Obsidian-Canvas-style editable canvas**: the
user can create boxes, edit them, and arrange freely. The artifact's design
is **AI-drafts-then-curate**: nodes are grounded views of existing objects;
the user accepts/dismisses/pins but doesn't author free content on the map.

These pull the data model in different directions:

| | AI-drafts-curate (artifact) | Freeform canvas (Obsidian-like) |
|---|---|---|
| Node identity | pointer to claim/entity/memory/page | user-authored box (text), optionally linked |
| Persistence | map layout + accept/pin state | full node/edge/position/content documents |
| AI role | drafts + suggests, never edits pinned | assistant on top of user content |
| Grounding invariant | every node resolves to a source | broken unless enforced separately |

**Settled 2026-07-18 (Lucian, via session 57b66472): Hybrid.** User-created
nodes are allowed but must link to an existing object
(claim/entity/memory/page), keeping "a view, not a copy" true. AI drafting +
per-node accept/dismiss/pin stays as the artifact specifies, and AI never
rewrites a curated node. The daemon API therefore includes the
create/update/delete surface from day one, with `ref` required on create.

**Auto-suggest is user-configurable** (Lucian, 2026-07-18): a setting
toggles whether ghost suggestions appear automatically (daemon may run map
passes proactively) or only on demand via "✦ Improve map". Default: on.
The setting gates only the proactive pass — `improve_page_map` stays
available either way. Surface the toggle in Settings and echo current state
on the map toolbar.

Council-specified on-ramp (so the ref requirement doesn't make the canvas
feel broken): typing loose text into a new node auto-creates a **backing
memory through the existing capture pipeline** — visible, undoable, subject
to normal curation — and the node refs that. Never auto-mint claims with
implied citation provenance; raw refless content is still rejected at the
API boundary.

## What exists today (verified on this branch, 2026-07-18)

- `getPageLinks(pageId)` — `src/lib/tauri.ts:1133`; `listOrphanLinks` at
  `:1137`. Wikilink graph data per page exists.
- Entity graph + relations (Atlas/Focus) — shipped; one renderer (sigma) for
  whole-graph, SVG for ego view.
- Accept/dismiss APIs exist **only for entity suggestions/revisions** — they
  do not cover map nodes.
- The daemon (`7xuanlu/wenlan` repo) has **no** APIs for: map persistence
  (layout, pin state), per-node suggestions, accept/dismiss of map nodes, or
  an improve-map pass. All four are prerequisites; all are daemon-repo work.

## Suggested daemon API sketch (fork settled; council-trimmed)

- `get_page_map(page_id) -> { nodes, edges, layout, updated_at }` — nodes
  carry `{ id, kind: claim|entity|memory|page|section, ref_id, pinned,
  suggested, provenance }`.
- `put_page_map_layout(page_id, positions)` — user drag persistence.
- `update_map_node(page_id, node_id, patch)` — ONE endpoint for
  accept / dismiss / pin / content-edit (council: shrink the endpoint
  fan-out; every endpoint is app↔daemon version-skew surface in a repo with
  no runtime version handshake). Dismissals write **tombstones** so
  `improve_page_map` can never re-propose a rejected suggestion.
- `improve_page_map(page_id, scope)` — enqueue the AI pass; suggestions land
  as `suggested: true` nodes on the next `get_page_map`. Must define a
  winner when racing user edits: curated state always wins.
- Dangling refs (council): when a backing object is deleted, merged, or
  superseded, its nodes enter a defined tombstone/ghost state — never a
  silent break of the grounding invariant.
- Hybrid (settled): `create_map_node(page_id, { content, ref })` — `ref`
  REQUIRED (claim/entity/memory/page id); `update_map_node`,
  `delete_map_node`. Refless free content is rejected at the API boundary —
  that is where the grounding invariant is enforced.
- Edge CRUD (Lucian, 2026-07-18 — this was the gap in the sketch): edges
  carry `{ id, from, to, label?, kind: section|citation|wikilink|user|
  suggested }`. User-drawn connections via `create_map_edge(page_id,
  { from, to, label? })`; `update_map_edge` folds accept/dismiss of a
  suggested link plus relabel (same shrunk fan-out as nodes);
  `delete_map_edge`. Dismissed suggested links get tombstones, same as
  nodes — never re-proposed.
- `put_page_map_layout` persists per-node `x/y/width/height` (JSON Canvas
  convention), not positions alone — user resize is layout state too.

## Tooling decision (2026-07-18) — two-agent survey, verified against npm/GitHub/docs

**Renderer: `@xyflow/react` (React Flow) 12.x.** MIT (clean with our
AGPL-3.0-only app), fully controlled nodes/edges — the arrays ARE app state,
the structural prerequisite for the pointer invariant (single source of
truth, one place to validate; the daemon API boundary is the actual
enforcer) —
first-class custom React nodes (citation badges, Accept/Dismiss chips, ghost
styling), built-in drag/zoom/pan/minimap, largest community in the field
(37.7k★, daily commits). Known nit: transitive `zustand ^4` peer-dep warning
under React 19 (xyflow #5229/#4893) — cosmetic, pnpm override if noisy.

**Auto-layout: `d3-flextree`.** Purpose-built for variable-height boxes
(per-node `nodeSize` accessor), synchronous O(n) — fine at page-map scale.
WTFPL, stale since ~2018: **vendor the source in-tree** (pinning is not
vendoring — council correction; WTFPL permits it, and packaging/typing
drift becomes our deliberate ownership cost). **Council gate: spike the
radial case FIRST** — a naive x/y→angle/radius transform can re-collide
variable-width boxes near the center; angular spacing must account for box
size at radius. Fallback shape: bilateral (mirrored two-run) layout.
Fallback engine: `elkjs` (active, native `mrtree`/`radial`, variable sizes;
~8 MB unpacked, worker execution; its `EPL-2.0 OR GPL-3.0-or-later` dual
license is AGPL-compatible either way — council-confirmed).

**Persistence: first-party JSON schema; skip JSON Canvas.** The v1.0 spec is
effectively closed (`text|file|link|group` nodes only, no extension
mechanism) — no room for provenance, accept/pin state, or the ghost flag.
Borrow its conventions (per-node `x/y/width/height`, edges as
`fromNode/toNode/label`) without claiming compliance.

**Fork-agnostic:** React Flow is a common base for Obsidian-Canvas-style
freeform editors AND fits the curate-only model — the renderer choice does
not constrain the direction fork either way.

Rejected:

- tldraw — proprietary license (verified against LICENSE.md); production use
  requires a paid or discretionary watermarked grant. Disqualified for an
  AGPL app.
- GoJS — commercial per-developer license. Disqualified.
- Excalidraw — MIT, but no interactive React content inside canvas elements
  (open FR excalidraw#8424); can't do badge/chip nodes.
- `@antv/g6` 5.1.1 (already in devDeps) — core active, but
  `g6-extension-react` breaks at runtime under React 19 (stale
  react-reconciler / @antv/react-g peers); imperative model besides.
- cytoscape 3.34 (already in devDeps) — canvas-rendered; HTML nodes only via
  single-maintainer plugins; imperative model.
- JointJS (`@joint/core` + `@joint/react` 4.3, MPL-2.0) — the runner-up:
  explicitly built for rich interactive HTML nodes and actively shipping,
  but the React wrapper is brand-new, state lives in its `dia.Graph` (sync
  adapter needed), and auto-layout is fully bring-your-own.
- reaflow — right shape (controlled, React nodes, elk-based layout) but no
  commits since 2025-04; adoption risk. (Not archived — a search claim of
  Jan-2026 archival was checked and is false.)
- Dedicated mind-map libs (mind-elixir, jsMind, markmap, simple-mind-map) —
  all topic-string renderers with fixed decoration schemas; none render
  arbitrary React per node. mind-elixir is healthiest but its React wrapper
  is near-dead; simple-mind-map's maintainer has moved to closed-source.

Housekeeping (council-upgraded: do WITH this work, not at ship): remove the
unused `@antv/g6` and `cytoscape` devDeps.

## Council review (2026-07-18, /boule:debate: Claude + gpt-5.6-sol + Gemini 3 Pro)

Verdict: **approve-with-changes** (tally 2 approve-with-changes / 1 approve;
stake-free judge, medium confidence, position-stable across swapped
orderings). The stack survived attack: React Flow over JointJS was endorsed
unanimously on independent grounds; the JSON Canvas rejection hardened (an
attacker's "prefixed-properties extensibility" counter-citation was exposed
as fabricated by a primary-source check); the elkjs license worry dissolved.
All surviving change items are folded into the sections above. Recorded
dissents / deliberate defaults:

- **Tree vs DAG**: a supporting item citing multiple claims is rendered by
  duplicating the leaf node (standard mind-map convention) — deliberate
  modeling default, revisit only if duplication confuses in practice.
- **.canvas export**: optional and user-story-driven, not a requirement —
  a lossy Obsidian export would strip curation state; add only if users ask.
- **Vendoring depth**: one member held that pinning finished pure-math
  d3-flextree suffices and an in-tree fork is busywork; we vendor anyway
  (majority) — the cost is one small file of algorithm source.

## Codex gap review (2026-07-18, gpt-5.6-sol xhigh, cross-model)

Asked one question — "what is the plan silent on?" — with settled decisions
excluded. 22 ranked gaps, most load-bearing first. None block the radial
spike; items 1–17 are daemon-API-design inputs, 18–22 are frontend/UX
inputs. Resolve or consciously defer each one when its layer gets designed.

1. **Version-skew contract** — versioned routes/payloads, unknown-field and
   unsupported-endpoint behavior (no runtime handshake exists between repos).
2. **Optimistic concurrency** — monotonic map revision, conditional writes,
   client op IDs; second-resolution `updated_at` + "curated wins" can't
   prevent lost/replayed edits.
3. **Content ownership** — does `update_map_node` content-edit change a
   map-local label or the referenced object? Editable copies would break the
   grounding invariant.
4. **Suggestion/curation state machine** — replace `suggested`/`pinned`
   booleans with explicit states (pending/accepted/curated/dismissed/
   dangling/superseded); keep edge kind separate from suggestion state.
5. **Occurrence identity** — node occurrence IDs ≠ backing-object IDs;
   tombstones need semantic fingerprints or duplicate leaves and fresh IDs
   bypass dismissals.
6. **Schema migration** — storage schema version, forward migration,
   interrupted-migration recovery, downgrade policy, before maps are durable
   user data.
7. **Atomicity + undo/redo** — transactions and inverse ops, especially for
   backing-memory-plus-node creation (no orphans on failure or undo).
8. **Improve-map job lifecycle** — job ID, base revision, status/cancel/
   retry, dedup, stale-run-result discard.
9. **Space-map API** — cross-page mode is promised but every endpoint is
   keyed by `page_id`.
10. **Whole-map lifecycle** — absent-map init, regenerate/reset, delete;
    uninitialized vs empty vs all-dismissed.
11. **Ref type completeness** — discriminated `ref` schema; citation/source
    and section refs are missing from the four create types.
12. **Graph invariants** — root uniqueness, cycles, self-loops, duplicate
    edges, endpoint validation, root editability.
13. **Provenance schema** — auditable source IDs, run/model, timestamps,
    confidence — not one opaque field.
14. **Referential integrity ops** — edge cascade on node delete; explicit
    relink/remove/resolve for merged or superseded backing objects.
15. **Layout ownership** — persist viewport + collapsed state; pinning ≠
    position locking; auto-layout vs manually placed nodes.
16. **Auto-suggest preference API** — where the setting lives (global/space/
    page scope) and how the app reads/writes it; daemon sketch has no
    preference surface today.
17. **Error/conflict contract** — machine-readable errors for missing refs,
    conflicts, failed AI jobs, daemon down.
18. **UX state matrix** — loading, empty, generating, saving, offline,
    partial-ref failure, conflict, undo-failure; saved/unsaved feedback
    during drag.
19. **Accessibility** — Outline as the semantic keyboard-operable tree
    alternative; keyboard CRUD; suggestion state never color-only.
20. **Scale budgets** — tested node/edge limits, bulk ref hydration,
    coalesced layout writes, layout off the UI thread.
21. **Proactive-AI scheduling** — debounce, per-page coalescing, concurrency
    caps, offline/battery behavior, so default-on can't build an unbounded
    queue.
22. **i18n beyond strings** — AI output locale, RTL canvas, locale-aware
    counts/times, re-layout after translation.

## Frontend notes for the future session

- Renderer choice: settled — see "Tooling decision" above. sigma remains
  wrong for this; the earlier "plain absolutely-positioned DOM/SVG is likely
  enough" note is superseded (drag/zoom/pan/minimap are exactly the wheel
  React Flow already ships). Obsidian Canvas is DOM boxes; same grain.
- **Acceptance criterion — edges anchor to node bounds, structurally**
  (Lucian, 2026-07-18, after catching edge drift in the static mockup):
  edges must be computed from node geometry (React Flow handles), never
  hand-placed coordinates, so they stay attached through drag, zoom,
  resize, and text reflow. The mockup's drift came from a fixed-coordinate
  SVG stretching with the container — a bug class React Flow makes
  impossible, but verify in the spike anyway: drag a node, edges follow.
- Reuse the artifact's visual tokens: ghost styling for suggestions
  (`--mem-indigo-bg` + dashed border), `✦` marker, per-node Accept/Dismiss
  chips, `[n]` citation badges.
- The Read | Map | Sources tab row is **net-new UI** — checked 2026-07-18:
  `src/components/memory/PageDetail.tsx` (652 lines) has zero tab structure
  today (single Read view). Page sources data is already fetched there via
  `getPageSources`, so the Sources tab has its data wired.
- i18n: all new copy in all three locales (`src/i18n/resources.ts`), and the
  hardcoded-copy guard will catch strays.

## Related decisions already made on the KG branch

- Timestamps: daemon delivers Unix **seconds** app-wide.
- Entity palette: validated 5-slot (`slotForEntityType`,
  `src/lib/graph/palette.ts`) — do not extend to 7 slots without re-running
  the dataviz validator.
- Top-20 detail-fetch cap in Atlas: daemon has no bulk-relations API — a
  bulk endpoint would also serve the map's cross-page mode.
