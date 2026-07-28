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
| HTTP routes in `router.rs` | 139 | parsed `.route("…", method(handler))` |
| HTTP routes in **other** server files | 5 | `repair_routes.rs:24-36` |
| MCP tools | 29 | `#[tool(` in `wenlan-mcp/src/tools.rs` |
| CLI subcommands | 18 top-level | `Commands` enum, `wenlan-cli/src/main.rs:31` |

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
| `/ws/updates` | automatic |
| `/api/steep` | automatic |
| `/api/distill`, `/api/distill/{page_id}` | automatic |
| `/api/lint`, `/api/repairs/*` (5 routes) | automatic |
| `/api/knowledge/path` | automatic |
| `/api/debug/pipeline` | automatic |
| `/api/communities/proposals/{id}/accept`, `/reject` | automatic |
| `/api/pages/{id}/map/*` (5 routes) | explicit |
| `/api/pages/{id}/archive`, `/api/memory/{id}/update-page` | explicit |

Those are opted out of *scope-sensitivity* classification, which is a different
question from *truth exposure*. A page-bearing route can be scope-insensitive
and still leak provisional prose. So M5's truth classification must be **total
over every registered path**, including the opt-out lists, with its own
fail-closed assert in `TrackedRouter::route`.

**The table above is deliberately not the contract.** An earlier draft of this
document asserted a specific count, and re-reading `NON_SENSITIVE_PATHS`
immediately found more. That is the argument for total coverage in one
observation: any hand-enumeration of page-bearing readers is wrong on the day
it is written, so PR-B classifies **every** registered path and lets the assert
find what a human list misses. Treat this table as motivation, never as the set.

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
| `path` | route / tool name / CLI command |
| `surface` | `http \| mcp \| cli \| projection \| export \| internal` |
| `page_bearing` | `yes \| no` |
| `class` | `automatic \| explicit \| not_applicable` |
| `adapter` | the exact enforcement call site |

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

## 4. Classification of the known page-bearing surface

Derived from the route table above. The executable test owns the authoritative
copy; this section is the reviewed seed.

### Automatic — provisional excluded unconditionally

| Path | Why automatic |
|---|---|
| `POST /api/context` | the canonical automatic context constructor |
| `POST /api/search` | supplementation, no page named |
| `POST /api/memory/search` | same |
| `POST /api/pages/search` | machine-ranked, no page named |
| `GET /api/briefing` | synthesized digest |
| `GET /api/home-stats` | synthesized counts + surfaced pages |
| `GET /api/decisions`, `/api/decisions/domains` | synthesis |
| `GET /api/profile/narrative` | synthesis |
| `GET /api/knowledge/path`, `/recent-relations` | graph traversal |
| `GET /api/communities/*` | routing/graph voting inputs |
| `GET /api/memory/nurture`, `/api/memory/entity-suggestions` | candidate generation |
| `GET /api/refinery/queue` | candidate generation |
| `POST /api/steep`, `POST /api/distill`, `POST /api/distill/{page_id}` | downstream synthesis inputs |
| `POST /api/lint`, `/api/repairs/*` | maintenance readers |
| MCP `context`, `recall`, `search_pages`, `list_pending`, `list_refinements`, `list_nurture`, `lint`, `distill`, `get_lint_agent_work_page`, `prepare_lint_repair*`, `verify_lint_repair` | agent-facing automatic reads |
| `/ws/updates` | push payloads carry page fields |
| internal: RRF page channel, graph voting, community routing, export builders, refinery, summary | not reachable by URL; still readers |

`/ws/updates` is called out because a push channel is easy to forget: it is a
reader that no one requested, which is the definition of automatic.

### Explicit — provisional allowed only with a declared M5 contract

| Path | Why explicit |
|---|---|
| `GET /api/pages/{id}` | a page was named |
| `GET /api/pages/{id}/sources`, `/links`, `/revisions`, `/map*` | subresources of a named page |
| `GET /api/pages`, `/api/pages/recent`, `/api/pages/recent-changes`, `/api/pages/orphan-links` | human browse surfaces |
| `POST /api/pages/export`, `POST /api/pages/{id}/export` | human-initiated export |
| MCP `get_page`, `get_page_sources`, `get_page_revisions`, `get_page_links`, `list_pages_recent` | agent acting on a named page |
| CLI `wenlan pages …` | human at a terminal |

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

| Weakening | Must fail |
|---|---|
| derive `truth_class` from `sensitive_read_routes()` membership | §2 opt-out test — `/ws/updates` and every other opt-out path goes unclassified |
| replace total coverage with a hand-enumerated page-bearing list | §2 — add a page-bearing route to `NON_SENSITIVE_PATHS`, assert must fire |
| build a second source-scanning manifest beside `TrackedRouter` | §1 — review gate, duplicate seam |
| allow an unclassified registered path | `TrackedRouter::route` assert |
| allow a table entry with no registered path | `finish()` drift assert |
| leave an MCP tool or CLI subcommand uncovered | §2 per-surface coverage test |
| default ambiguous readers to `explicit` | §3 classification test |
| treat an unknown contract version as supported | §6 test |
| let an explicit path skip negotiation | §7 explicit case 1 |
| replace per-entry mutation tests with one blanket test | §7 — remove one adapter, blanket test still passes |
| leave provisional files in the legacy projection directory | §5 invariant test |
| omit `/ws/updates` from the manifest | §2.3 |
