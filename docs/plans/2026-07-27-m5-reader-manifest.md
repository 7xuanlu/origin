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

## 1. Why a hand-written list is not acceptable

Verified surface on the merge base (`5ba8a3b4`):

| Surface | Count | Source |
|---|---|---|
| HTTP routes in `router.rs` | 139 | parsed `.route("…", method(handler))` |
| HTTP routes in **other** server files | 5+ | `repair_routes.rs:27-36`, and a `/api/lint` registration outside `router.rs` |
| MCP tools | 29 | `#[tool(` in `wenlan-mcp/src/tools.rs` |
| CLI subcommands | 18 top-level | `Commands` enum, `wenlan-cli/src/main.rs:31` |

`router.rs` is **not** the whole route table. `/api/repairs/*` and `/api/lint`
register in separate modules and are composed in. Any manifest built by reading
one file is already wrong today, before M5 adds anything. That is the concrete
reason the manifest must be generated and enforced, not transcribed.

## 2. Enforcement mechanism

Axum exposes no public API to enumerate a composed `Router`'s paths, so the
manifest is enforced at the **source level**, in the established `drift_guard.rs`
idiom (teeth #2 already scans `crates/*/src` for `WENLAN_*` flags and fails the
build on a new undocumented one).

`m5_reader_manifest` is a `#[cfg(test)]` lib test that:

1. scans every `crates/*/src/**.rs` for route registrations
   (`.route("<path>", <method>(<handler>))`), MCP `#[tool(` declarations with
   their `async fn` names, and CLI subcommand variants;
2. joins each discovered path against a checked-in classification table;
3. **fails on any discovered path absent from the table** — fail-closed, exactly
   like teeth #2. A new endpoint is unclassified until someone classifies it;
4. fails on any table entry that no longer matches a discovered path, so the
   table cannot rot into fiction.

The scan runs on the same `cargo test --workspace --lib` that CI and pre-push
already run. No new wiring.

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
| build the manifest from `router.rs` alone | §2 scan test — `/api/repairs/*` and `/api/lint` go missing |
| allow an unclassified discovered path | §2.3 fail-closed test |
| allow a table entry with no matching source path | §2.4 rot test |
| default ambiguous readers to `explicit` | §3 classification test |
| treat an unknown contract version as supported | §6 test |
| let an explicit path skip negotiation | §7 explicit case 1 |
| replace per-entry mutation tests with one blanket test | §7 — remove one adapter, blanket test still passes |
| leave provisional files in the legacy projection directory | §5 invariant test |
| omit `/ws/updates` from the manifest | §2.3 |
