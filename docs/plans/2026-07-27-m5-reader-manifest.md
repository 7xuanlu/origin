# M5 Stage 0 — executable reader manifest

Date: 2026-07-27. Binding for M5 PR-B. Implements D3 of
`2026-07-27-kg-m5-goal-prompt.md`.

D3's rule: **automatic readers exclude provisional pages; explicit readers may
expose them only to a client that declared M5 truth-contract support and renders
both axes.** Old or unversioned clients receive no provisional title, summary,
prose, excerpt, or Markdown projection.

The prompt's binding constraint on this artifact: *"No single endpoint or shared
helper is assumed to cover this set. Stage 0 must produce an executable manifest
from actual callers."*

## 1. Surface, and the seam that already exists

Verified surface on the merge base (`5ba8a3b4`):

| Surface | Count | Source |
|---|---|---|
| HTTP `(method, path, handler)` triples in `router.rs` | 155 | paren-balanced parse of `.route("…", …)` |
| in `repair_routes.rs` | 5 | `repair_routes.rs:24-36` |
| in `lint_routes.rs` | 2 | missed by the first pass |
| **total** | **162** | every file, `#[cfg(test)]` modules stripped |
| distinct `(method, path)` pairs | 160 | 2 duplicates, below |
| MCP tools | 29 | `#[tool(` in `wenlan-mcp/src/tools.rs` |
| CLI subcommands | 19 top-level | `Commands` enum, `wenlan-cli/src/main.rs:29` |

The two duplicate `(method, path)` pairs are `GET /api/health` and
`GET /api/status`, registered once in `build_router` (`router.rs:47`) and again
in `build_repair_router` (`router.rs:592`). Two separate `Router` instances, not
an overlapping registration — but the inventory's set-equality check must key on
`(builder, method, path)` or it will report a phantom drift.

These counts come from a paren-balanced parse across **every** file in the
crate, with `#[cfg(test)]` modules stripped. Three counts preceded this one —
151 (hand-picked file list, single-line regex), then 163 (both fixed, but
counting a route registered inside a test module). All three are recorded in the
inventory, because the parser is now part of the contract.

`router.rs` is not the whole route table — `/api/repairs/*` registers in its own
module. But it composes through the **same** wrapper, and that wrapper is the
seam M5 must use rather than reinvent:

`TrackedRouter` (`wenlan-server/src/route_registry.rs`) intercepts every
`.route()` call and already fails closed:

- `route()` asserts `"unclassified router path: {path}"` unless the path
  resolves in `wenlan_core::lint::serving::routes` or appears in
  `NON_SENSITIVE_PATHS` / `NON_SENSITIVE_MIXED_ROUTES`;
- `finish()` asserts `"sensitive route registration drift"` — the registered set
  must **exactly equal** `routes::sensitive_read_routes()`, so a route that is
  classified but never registered fails too;
- `repair_routes.rs:24` takes and returns `TrackedRouter`, so the separate
  module is covered by the same assert.

An M5-specific source-scanning test would be a second, weaker copy of this —
the exact hand-copied-canonical-thing pattern that produced three separate
defects in the M4 review rounds. M5 extends the existing table instead.

## 2. Enforcement mechanism

### HTTP: extend `SensitiveReadRoute`, reuse the existing assert

`SensitiveReadRoute` (`lint/serving/routes.rs:57`) already carries per-route
classification — `data_class`, `selector_precedence`, `capability`,
`scope_binding`, `selection_gate`, `unknown_scope`, `cross_scope_policy`.
M5 adds one field:

```rust
pub truth_class: TruthClass,   // Automatic | NotPageBearing
```

and one derived check, mirroring the existing `scope_contract_violation()`.

### The opt-out list is the trap

`truth_class` must **not** be derived from membership in
`sensitive_read_routes()`. Verified: `NON_SENSITIVE_PATHS`
(`route_registry.rs:190`) contains paths that are page-bearing readers under D3.
Found by inspection, **non-exhaustive**:

| Path in `NON_SENSITIVE_PATHS` | D3 class |
|---|---|
| `/api/steep` | automatic |
| `/api/distill`, `/api/distill/{page_id}` | automatic |
| `/api/lint`, `/api/repairs/*` (5 routes) | automatic |
| `/api/debug/pipeline` | automatic |
| `/api/communities/proposals/{id}/accept`, `/reject` | automatic |
| `/api/pages/{id}/map/*` (5 routes) | automatic |
| `/api/pages/{id}/archive`, `/api/memory/{id}/update-page` | automatic |

**Two paths an earlier draft wrongly listed here**, both corrected by reading
the response types instead of reasoning from the route name:

- `/ws/updates` is **not** page-bearing. `WsServerMessage` carries only
  index progress (`files_indexed`/`files_total`), ingest completion
  (`document_id`, `chunks`), and an error string (`websocket.rs:34`). The draft
  called it "a reader nobody requested" and made it the headline example.
- `/api/knowledge/path` returns a **filesystem path string**
  (`knowledge_routes.rs:11`), not a graph traversal. Its exposure risk is the
  projection directory it names, which §5's invariant owns — not a
  page-response adapter.

Those are opted out of *scope-sensitivity* classification, which is a different
question from *truth exposure*. A page-bearing route can be scope-insensitive
and still leak provisional prose. So M5's truth classification must be **total
over every registered path**, including the opt-out lists, with its own
fail-closed assert in `TrackedRouter::route`.

**The table above is not the contract** —
`2026-07-27-m5-reader-manifest-inventory.md` is. An earlier draft asserted a
specific count; re-reading `NON_SENSITIVE_PATHS` found more, and reading the
response types then removed two that did not belong. A hand-enumeration was
wrong in **both directions** within one day of being written, which is why the
inventory is generated and this table is kept only as the motivating example.

Deciding `page_bearing` requires reading the **response type**, not inferring
from the route name; where the response type is `serde_json::Value` even that is
not enough; and where the route writes prose to a destination, the response is
the wrong place to look entirely. The inventory applies all three tests.

### MCP, CLI, projection, internal: no registry exists

These have no `TrackedRouter` equivalent, so each needs its own seam:

| Surface | Seam |
|---|---|
| MCP | a `const` table keyed by tool name; a test asserts it covers every `#[tool(` fn in `tools.rs` and nothing more |
| CLI | a `const` table keyed by subcommand path; a test asserts it covers every `Commands` variant |
| projection | the directory invariant (§5), not a call site |
| internal | a `const` table of reader call sites, asserted against a source scan for the page-read helpers |

The internal-reader scan is the weakest of the four, because "reads a page" is
not a syntactic property. It is bounded by routing every page read through a
small set of named helpers and asserting no other module calls the underlying
query directly — the same containment `TrackedRouter` gives HTTP.

## 3. Classification table shape

Each entry carries seven fields. All seven are required; none may be inferred.

| Field | Values |
|---|---|
| `method` | `GET \| POST \| PUT \| DELETE \| PATCH`; `n/a` off-HTTP |
| `path` | route / tool name / CLI command |
| `surface` | `http \| mcp \| cli \| projection \| export \| internal` |
| `page_bearing` | `yes \| no` |
| `class` | `automatic \| not_applicable` — never `explicit`, see §4 |
| `adapter` | the exact enforcement call site |
| `evidence` | the response fields, opacity, or effect that decided `page_bearing` |

`method` is required because `TrackedRouter`'s authoritative identity is
`(method, path)`, not path alone (`route_registry.rs:145`). Several paths carry
both a read and a mutation — `GET` vs `DELETE /api/pages/{id}` — and keying the
manifest on path would silently merge them into one classification.

`page_bearing = no` entries still appear in the table. Marking something
not-page-bearing is a claim that must be reviewed, and silence is not a claim.

### Classification rule

| Reader | Class | Provisional pages |
|---|---|---|
| every route, tool, and subcommand | **automatic** | **excluded, unconditionally** |
| a call carrying a per-call human-intent marker | **explicit**, for that call only | allowed, both axes rendered |

Under-exposing a provisional page costs a user one click; over-exposing one
attaches unearned trust to unverified prose, which is the failure this rung
exists to prevent. §4 explains why intent cannot be a route property.

## 4. Classification — the enumeration lives in the inventory file

**The manifest is `2026-07-27-m5-reader-manifest-inventory.md`**, generated from
the merge base: all 162 registered `(method, path, handler)` triples, all 29
`#[tool(` declarations, all 19 `Commands` variants, and the internal
page-prose readers, each with `page_bearing`, a class, a marker shape, an
adapter, and the evidence the classification rests on. That file is the artifact
this section used to only promise.

The internal readers are **not enumerated in prose here or there**. They come
from `scripts/m5-reader-sweep.py`, which is the predicate; the inventory carries
its output and the count is whatever it prints. Four hand-written drafts of that
set were wrong in four different ways, which is the whole argument for making it
a script.

Three earlier counts were wrong — 151, then 163 — from a hand-picked file list,
a single-line regex, and a route registered inside a `#[cfg(test)]` module.
**Enumerate by pattern over the whole crate, and strip test modules.**

Generating it also surfaced five things a reviewed prose list did not:

- `label` is a page title by another name — `PageLinkOutbound`,
  `PageLinkInbound`, `OrphanLink`, `PageMapNode` all carry one, and a scan keyed
  on `title` misses every one;
- word-boundary matching misses `delta_summary`, because `_` is a word
  character — so the field scan must match substrings;
- routes are statically opaque where they return `serde_json::Value` or a bare
  `Response`, including `GET /api/pages`, `POST /api/pages/search`, and
  `GET /api/pages/{id}` — the three primary page readers;
- **response scanning is blind to effects.** `POST /api/pages/export` returns
  only `ExportStats` and writes full page prose into the user's Obsidian vault.
  The most consequential page reader in the product exposes nothing through its
  response, so `page_bearing` needs an effects test alongside the response test;
- **and blind to the error arm** — though this one is *not* a `page_bearing`
  test, and a draft that made it one contradicted its own table. Nearly every
  handler returns `Result<_, ServerError>` and every variant carries a free-form
  `String` (`error.rs:11`) — the exact count is in the inventory, not restated
  here — so the rule would classify almost every route page-bearing while the
  table says 77. D4's stale-base conflict is precisely where an
  implementer writes `current version: <title>`. The leak is real; the axis was
  wrong. It is now **one cross-cutting invariant at the error-serialization
  seam** — no `ServerError` body may carry a provisional page's title or prose —
  enforced once, covering routes added later, and far less to maintain than 158
  classifications.

### `explicit` is a property of the call, not the route

Two drafts got this wrong in opposite directions, and the correction is the
load-bearing part of this artifact.

The first said "all 16 opaque routes classify automatic, fail-closed." That
conflated *not knowing whether prose is in the payload* with *not knowing
whether a human named the page*, and forcing both to automatic would have denied
provisional content to M5-aware browse clients.

The second fixed that by reading intent off the route — `GET /api/pages/{id}`
names a page, so it is explicit. Live app code disproves it:

| Route | Classified | What production actually does |
|---|---|---|
| `GET /api/pages` | explicit (browse) | `SpaceList.tsx:76` polls it every 10 s for sidebar counts |
| `GET /api/pages/recent-changes` | explicit (browse) | `HomePage.tsx:75` loads it every 30 s |
| `GET /api/pages/orphan-links` | explicit | feeds candidate generation (`memory_routes.rs:3464`) |

A route is not one reader. The same path serves a human who clicked and a timer
that polls, and a client being *globally* M5-aware proves nothing about whether
*this* request was a human naming a page. Route-level intent cannot be made
sound by classifying more carefully; the signal is not in the route.

So M5 does not classify intent per route at all:

- **every reader is `automatic` by default**, and the inventory's class column
  is `automatic` throughout;
- **`explicit` is a per-call signal** — the request carries an explicit
  human-intent marker, the server records it, and only then may provisional
  content appear, with both axes rendered;
- absent the marker, automatic. No exceptions, no route allowlist.

This is simpler than what it replaces: one server-verifiable signal instead of a
per-route intent judgment that a UI change can silently falsify. It also removes
the entire class of "the app started polling an explicit route" regressions,
which nothing in the previous design would have caught.

### The marker binds to the pages the call names

A per-call marker is not a blanket header. It **binds to the page IDs the call
names**, which makes spraying it on every request structurally meaningless on
routes that name no page. That is the cheap structural defence; the marker is
otherwise cooperative-tier (below).

Embedded other-page content follows the automatic rule.
`PageLinkOutbound.label`, `PageLinkInbound.label`, `OrphanLink.label`, and
`PageMapNode.label` all carry *other* pages' titles with no per-item axes, so a
grant for page A never covers page B's title riding along in A's payload.

### Collections may list provisional entries; only prose is named-page-only

Named-page-only, applied to collections, produces a dead end: post-migration
every page is `provisional` (artifact 2 §4), lists and search exclude
provisional, and **a human cannot name a page they cannot see**. The review loop
this rung exists to feed would have no entry point — the same curation-death
failure artifact 6 §2a names for human prose. It also contradicts G6, which
requires "M5-aware explicit browse returns all four with both axes" for every
manifest row, unsatisfiable for collection rows under a named-page-only rule.

The line that resolves both:

| Call | Provisional pages |
|---|---|
| marker-bearing **collection** (list, search, recent) | **entries visible** — page ID + title + both axes per item, no prose |
| marker-bearing **named-page** fetch | full prose, both axes |
| any call without the marker | excluded entirely |
| embedded other-page labels inside either | excluded — they carry no axes |

`/api/pages/orphan-links` is deliberately **not** in that first row. Its items
are `OrphanLink { label, count }` (`wenlan-types/src/responses.rs:1128`) — no
page identity, no axes — and the carve-out is conditional on rendering both.
A route qualifies for `collection` only if its item type can carry a page
identity and both axes.

Discovery is restored without weakening the prose rule: a provisional page can
be *found* and its state seen, and reading it is still a deliberate act. Per-item
axes are what make this safe — an entry that appears without its state is the
unearned trust this rung exists to prevent, which is why the carve-out is
conditional on rendering them.

### Two gates: shape is hard, authenticity is cooperative

Stated once so it is not re-litigated. The intent marker is **not** a D7
presence capability. Artifact 5 §1 already puts hostile-same-user out of scope,
and a client willing to forge the marker can equally lie about its contract
version — both sit in the same trust tier, so nonce-consuming machinery buys
nothing against the conceded attacker.

An earlier draft stopped there and made eligibility a boolean "enforced
client-side by a test on each surface," which is another way of saying the
server does not enforce it. Every page-bearing route ended up eligible,
including `POST /api/context` and both exports — the shapes D3 excludes
unconditionally. That does not close the mixed-caller hole; it relocates it.

Splitting the concern makes one half hard:

| Gate | Question | Enforcement |
|---|---|---|
| **route shape** | may a marker do anything here, and what? | **server-side, no client cooperation.** `none` **refuses** the request. |
| **marker authenticity** | did a human actually gesture? | cooperative-tier; the daemon cannot tell forged from real |

`marker_shape` is therefore a three-valued per-route column in the inventory —
`none` (153 routes), `collection` (4), `named_page` (5) — fail-closed by
construction, so a route added tomorrow is `none` until someone deliberately
gives it a shape.

The gate stops the **careless** integration: an MCP tool wired to transmit the
marker still gets nothing from `/api/context`, `/api/search`, or an export,
because those refuse it regardless of caller. It does **not** bound total
exposure, and a draft of this section wrongly claimed it did — `collection` and
`named_page` compose, so a forging agent can enumerate provisional IDs and then
fetch each one. That requires forging, which artifact 5 §1 concedes at T11; it
is made auditable rather than prevented. The inventory states the full
disposition.

Per-surface transmission (MCP, internal readers, non-interactive CLI: never)
remains the soft gate, tested per surface. It is no longer load-bearing.

The two-table prose enumeration that used to sit here was deleted, not moved: it
was a second copy of the inventory's Class column, which is precisely the
hand-copied-canonical-thing pattern that produced the misses above.

## 5. The projection path is not wire-negotiable

`wenlan pages` reads Markdown from the legacy projection directory directly, so
frontmatter and wire negotiation cannot protect it. Per D3 the directory itself
becomes the enforcement boundary: after cutover it contains **supported pages
only**, maintained through the durable outbox/reconciler and the fenced cutover
ceremony (artifact 7). M5-aware explicit browsing fetches provisional content
from the daemon, never from that directory.

Manifest entries for the projection surface therefore record
`adapter = projection_directory_invariant`, not a code call site.

## 6. Contract negotiation

- The client declares support with an explicit truth-contract version. No
  caller-selected content filter — a client cannot ask to see provisional pages;
  it can only declare that it renders both axes.
- Absent, malformed, or unknown version ⇒ treated as legacy ⇒ no provisional
  content.
- A declared version newer than the daemon supports ⇒ treated as legacy. Trust
  contracts do not negotiate upward.

Contract declaration is **necessary and not sufficient**. It says the client can
render both axes; it does not say a human named a page. Provisional content
requires both the declared contract **and** the per-call intent marker (§4). A
declared client polling on a timer gets automatic behavior, which is the whole
point of separating the two.

## 7. Mutation test per entry

D3 requires a mutation test per adapter, not one blanket test. For each
page-bearing entry:

- **automatic**: seed one provisional and one supported page; assert the
  response contains the supported page and **no field** of the provisional one —
  not title, not summary, not prose, not excerpt, not ID-with-content. Then
  remove that adapter's filter and assert the test goes RED.
- **per-call explicit**: four cases — no contract declared (provisional absent),
  contract declared but **no intent marker** (provisional absent), contract and
  marker both present (provisional present **with both axes rendered**), unknown
  version (provisional absent). Removing the negotiation check must turn case 1
  or 4 RED; removing the intent-marker check must turn case 2 RED.
- **embedded other-page content**: with contract and marker both present for
  page A, seed a *provisional* page B linked from A, and assert B's title is
  absent from A's links, map, and orphan-link payloads. The grant covers the
  named page only.

A blanket "search excludes provisional" test passing while one adapter is
missing is exactly the false PASS the per-entry requirement prevents. Case 2 is
the one the previous design could not express at all: it had no way to
distinguish an M5-aware client's human click from its 10-second poll.

## 8. PR-B does not activate D3

PR-B installs every adapter and mutation-tests it with `cutover_generation=off`.
Behavior is unchanged. PR-C advances the durable cutover generation through the
two-phase fenced ceremony only after readiness reaches 100%.

This split means the adapters are proven correct while inert, then switched on
by one durable generation advance — the same shape as the M4 reader cutover,
which is the precedent this follows deliberately.

## 9. Mutation checks

Rows marked **[gate]** are human review gates, not executable tests. They are
listed here because they must happen, but they must not be counted as teeth — a
table that mixes the two lets a process promise stand in for a failing build.
Every unmarked row is an executable test that goes RED under its weakening.

| Weakening | Must fail |
|---|---|
| derive `truth_class` from `sensitive_read_routes()` membership | §2 opt-out test — `/ws/updates` and every other opt-out path goes unclassified |
| replace total coverage with a hand-enumerated page-bearing list | §2 — add a page-bearing route to `NON_SENSITIVE_PATHS`, assert must fire |
| build a second source-scanning manifest beside `TrackedRouter` | **[gate]** §1 duplicate seam |
| allow an unclassified registered path | `TrackedRouter::route` assert |
| allow a table entry with no registered path | `finish()` drift assert |
| leave an MCP tool or CLI subcommand uncovered | §2 per-surface coverage test |
| default ambiguous readers to `explicit` | §3 classification test |
| treat an unknown contract version as supported | §6 test |
| let a marked call skip negotiation | §7 per-call case 1 |
| grant `collection` to an item type that cannot carry both axes | inventory teeth 12 — `OrphanLink { label, count }` |
| claim the shape gate bounds total exposure | §4 — `collection` + `named_page` compose |
| let a marked call leave no durable audit row | §4 — the audit record is the *only* compensating control for the conceded composition attack; without a tooth it is a sentence |
| remove any single adapter's filter | that entry's own §7 test goes RED |
| leave provisional files in the legacy projection directory | §5 invariant test |
| omit `/ws/updates` from the manifest | §2.3 |
| add a route without an inventory row | inventory §teeth check 2 (set equality) |
| delete an inventory row for a live route | inventory §teeth check 2 |
| leave an inventory row with an empty class | inventory §teeth check 3 |
| classify an opaque route `page_bearing = no` | inventory §teeth check 4 |
| scan response fields with word-boundary instead of substring matching | inventory — `delta_summary` must still be found |
| drop `label` from the prose-field pattern | inventory — `PageLinkOutbound.label` must still be found |
| strip the per-call marker check | §7 case 2 — a declared client's unmarked poll must not see provisional |
| treat the marker as a blanket header | §4 — a marker naming no page grants nothing |
| let an MCP tool transmit the marker | inventory teeth 9 — per-surface test |
| give a new route a marker shape by default | inventory teeth 7 — fail-closed allowlist |
| ignore a marker on a `none` route instead of refusing | inventory teeth 8 — `/api/context`, `/api/search`, both exports |
| exclude provisional entries from a marker-bearing collection call | §4 collection carve-out; G6 explicit-browse bullet |
| list a provisional entry without its axes | §4 — per-item axes are the carve-out's precondition |
| let page prose reach an error body | inventory §teeth check 8 sentinel |
