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
| HTTP `(method, path, handler)` triples in `router.rs` | 146 | parsed `.route("…", method(handler))` |
| HTTP triples in **other** server files | 5 | `repair_routes.rs:24-36` |
| distinct `(method, path)` pairs | 149 | 2 duplicates, below |
| distinct paths | 142 | 7 paths carry more than one method |
| MCP tools | 29 | `#[tool(` in `wenlan-mcp/src/tools.rs` |
| CLI subcommands | 19 top-level | `Commands` enum, `wenlan-cli/src/main.rs:29` |

The two duplicate `(method, path)` pairs are `GET /api/health` and
`GET /api/status`, registered once in `build_router` (`router.rs:47`) and again
in `build_repair_router` (`router.rs:592`). Two separate `Router` instances, not
an overlapping registration — but the inventory's set-equality check must key on
`(builder, method, path)` or it will report a phantom drift.

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
pub truth_class: TruthClass,   // Automatic | Explicit | NotPageBearing
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
| `/api/pages/{id}/map/*` (5 routes) | explicit |
| `/api/pages/{id}/archive`, `/api/memory/{id}/update-page` | explicit |

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

Classifying a path requires reading its **response type**, not inferring from
its name — and where the response type is `serde_json::Value`, even that is not
enough (inventory, "the opaque routes").

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

Each entry carries five fields. All five are required; none may be inferred.

| Field | Values |
|---|---|
| `method` | `GET \| POST \| PUT \| DELETE \| PATCH`; `n/a` off-HTTP |
| `path` | route / tool name / CLI command |
| `surface` | `http \| mcp \| cli \| projection \| export \| internal` |
| `page_bearing` | `yes \| no` |
| `class` | `automatic \| explicit \| not_applicable` |
| `adapter` | the exact enforcement call site |

`method` is required because `TrackedRouter`'s authoritative identity is
`(method, path)`, not path alone (`route_registry.rs:145`). Several paths carry
both a read and a mutation — `GET` vs `DELETE /api/pages/{id}` — and keying the
manifest on path would silently merge them into one classification.

`page_bearing = no` entries still appear in the table. Marking something
not-page-bearing is a claim that must be reviewed, and silence is not a claim.

### Classification rule

| Reader intent | Class | Provisional pages |
|---|---|---|
| a machine decides what to show, without a human naming the page | **automatic** | **excluded, unconditionally** |
| a human explicitly named the page or asked to browse | **explicit** | allowed **only** with a declared M5 contract |

Ambiguous readers classify **automatic**. Under-exposing a provisional page
costs a user one click; over-exposing one attaches unearned trust to unverified
prose, which is the failure this rung exists to prevent.

## 4. Classification — the enumeration lives in the inventory file

**The manifest is `2026-07-27-m5-reader-manifest-inventory.md`**, generated from
the merge base: all 151 registered `(method, path, handler)` triples, all 29
`#[tool(` declarations, all 19 `Commands` variants, each with `page_bearing`, a
class, an adapter, and the return-type evidence the classification rests on.
That file is the artifact this section used to only promise.

Generating it surfaced three things a reviewed prose list did not:

- `label` is a page title by another name — `PageLinkOutbound`,
  `PageLinkInbound`, `OrphanLink`, `PageMapNode` all carry one, and a scan keyed
  on `title` misses every one;
- word-boundary matching misses `delta_summary`, because `_` is a word
  character — so the field scan must match substrings;
- **16 routes are statically opaque**, returning `serde_json::Value` or a bare
  `Response`, and they include `GET /api/pages`, `POST /api/pages/search`, and
  `GET /api/pages/{id}` — the three primary page readers.

### The two axes are independent, including under opacity

An opaque return type answers `page_bearing`, not `class`. A first pass at this
section said "all 16 opaque routes classify automatic, fail-closed," which reads
like caution and is wrong in a way worth naming: it conflates *not knowing
whether prose is in the payload* with *not knowing whether a human named the
page*. Only the first is unknown. `GET /api/pages/{id}` is explicit whatever its
return type says, because reader intent is a property of the route, not the
struct.

Forcing those to automatic would have denied provisional content to M5-aware
explicit browse clients — a functional regression dressed as a safety win. The
rules are therefore separate:

| Unknown | Fail-closed default | Because |
|---|---|---|
| is prose in the payload? | `page_bearing = yes` | an unguarded reader is unrecoverable |
| did a human name the page? | read it off the route, as usual | opacity says nothing about intent |

**No provisional content is exposed anywhere without a declared M5 contract**,
so an opaque explicit route is not a hole: it degrades to automatic behavior for
every caller that has not declared support (§6).

### The classification rule

| Reader intent | Class |
|---|---|
| a human named the page, or asked to browse | **explicit** |
| everything else, including every ambiguous case | **automatic** |

Applied per row in the inventory. The two-table prose enumeration that used to
sit here was deleted, not moved: it was a second copy of the inventory's Class
column, which is precisely the hand-copied-canonical-thing pattern that produced
the misses above. One source, one positive control.

**Every** explicit path degrades to automatic behavior when the caller has not
declared the M5 contract. There is no "explicit therefore safe" path.

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

## 7. Mutation test per entry

D3 requires a mutation test per adapter, not one blanket test. For each
page-bearing entry:

- **automatic**: seed one provisional and one supported page; assert the
  response contains the supported page and **no field** of the provisional one —
  not title, not summary, not prose, not excerpt, not ID-with-content. Then
  remove that adapter's filter and assert the test goes RED.
- **explicit**: three cases — no contract declared (provisional absent),
  contract declared (provisional present **with both axes rendered**), unknown
  version (provisional absent). Removing the negotiation check must turn the
  first or third RED.

A blanket "search excludes provisional" test passing while one adapter is
missing is exactly the false PASS the per-entry requirement prevents.

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
| let an explicit path skip negotiation | §7 explicit case 1 |
| remove any single adapter's filter | that entry's own §7 test goes RED |
| leave provisional files in the legacy projection directory | §5 invariant test |
| omit `/ws/updates` from the manifest | §2.3 |
| add a route without an inventory row | inventory §teeth check 2 (set equality) |
| delete an inventory row for a live route | inventory §teeth check 2 |
| leave an inventory row with an empty class | inventory §teeth check 3 |
| classify an opaque route `page_bearing = no` | inventory §teeth check 4 |
| scan response fields with word-boundary instead of substring matching | inventory — `delta_summary` must still be found |
| drop `label` from the prose-field pattern | inventory — `PageLinkOutbound.label` must still be found |
| force opaque routes to `automatic` | §4 — `GET /api/pages/{id}` must stay explicit |
