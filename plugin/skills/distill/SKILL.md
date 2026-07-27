---
name: distill
description: >
  Synthesize wiki pages from related memories. One endpoint, one flow:
  daemon clusters and synthesizes what it can; agent finishes whatever
  the daemon couldn't (no LLM or cluster too big). Invoked as
  `/distill [target]`.
argument-hint: "[target | rebuild <page-id> | deep]"
allowed-tools: ["mcp__plugin_wenlan_wenlan__recall", "mcp__plugin_wenlan_wenlan__distill", "mcp__plugin_wenlan_wenlan__write_page", "mcp__plugin_wenlan_wenlan__delete_page", "mcp__plugin_wenlan_wenlan__get_page_sources", "Bash"]
---

# /distill

Force a distillation pass now. The daemon's background distill cycles run
on its own clock; `/distill` is the explicit user-triggered pass.

## Mental model

Distillation is four operations bundled into one flow:

- **emerge** — cluster new memories into new pages
- **absorb** — assign orphan memories to existing pages, propose new pages
  from topics that 2+ existing pages link to but no page is named for
- **refresh** — regenerate stale pages from their source memories (only when
  the user has not edited the page; pages you have touched stay locked)
- **merge** — combine duplicate pages flagged by the daemon's global review

The default flow runs all four. The `rebuild` verb is a destructive opt-in
that overrides the lock on a single page.

## Single flow

One POST to the daemon. Response splits into:

- `pages_created` / `created_ids`: pages the daemon synthesized itself
  (only when daemon has an LLM).
- `pending`: clusters the daemon couldn't finish. The agent
  synthesizes each in this session and POSTs them back to `/api/pages`.

Trigger timing is the only thing that differs between background distill
cycles and this skill. Code path is the same; daemon hands back
clusters when it can't synthesize; whoever called fills in the rest.

## Argument parsing

The `/distill` skill accepts one optional inline token of the form
`space:<name>` anywhere in the argument string. Extract it before
treating the rest as the target:

    raw_args="<the full argument string passed to /distill>"
    space_arg="$(printf '%s\n' "$raw_args" | grep -oE 'space:[A-Za-z0-9_-]+' | head -1 | cut -d: -f2)"
    target="$(printf '%s\n' "$raw_args" | sed -E 's/[[:space:]]*space:[A-Za-z0-9_-]+[[:space:]]*/ /g' | sed -E 's/^[[:space:]]+|[[:space:]]+$//g')"

If `space_arg` is non-empty, pass it to the resolver as `--arg "$space_arg"`.

## Resolve the active space

Call the bundled resolver:

    resolved="$("$CLAUDE_PLUGIN_ROOT/bin/resolve-space.sh" --cwd "$PWD" \
        ${space_arg:+--arg "$space_arg"} 2>/dev/null)"
    space="$(printf '%s\n' "$resolved" | cut -f1)"
    source_layer="$(printf '%s\n' "$resolved" | cut -f2)"

The `distill` MCP tool has no separate `space` parameter. Use the resolved
non-empty space as `target` unless an explicit target argument or rebuild target
is already being forwarded. Print one line before the call:

    Resolved space: <space> (from <source-layer>)

If `space` is empty, print `Resolved space: none (unscoped)` and omit `target`
unless the user supplied an explicit target.

## Flow

### 1. Pick the scope

Use exactly one target rule:

- Reserved keyword sequence `rebuild <page-id>` → call
  `distill(target=<page-id>, force=true)`. Confirms "Rebuild page <id>?
  Your edits will be wiped, page regenerates from sources." before
  proceeding. Skip the rest of this skill — single-page rebuild does
  not produce `pending` clusters; the daemon's response shape is
  `{"status": "ok", "force": true, "page_id": ..., "updated": true}`.
  Report verbatim.
- Reserved keyword `deep` → no target (global pass).
- `/distill <arg>` after removing any `space:<name>` token → forward `<arg>`
  to `target`.
- Bare `/distill` with a non-empty resolved space → use the resolved space as
  `target`.
- Bare `/distill` with no resolved space → omit `target`.

### 2. Call the MCP tool

```
distill(target="<scope>")
```

The tool returns the daemon's full JSON payload as text. Parse it as
JSON. Possible shapes:

```
{
  "pages_created": 0,
  "scoped": true,
  "created_ids": [],
  "pending": [
    { "source_ids": [...], "contents": [...], "entity_id": ...,
      "entity_name": ..., "space": ..., "estimated_tokens": ... },
    ...
  ],
  "stale_pages": [
    { "page_id": ..., "title": ..., "summary": ...,
      "source_memory_ids": [...], "stale_reason": "source_updated",
      "user_edited": false, "sources_updated_count": 3 },
    ...
  ],
  "stale_truncated": false,
  "orphan_topics": [
    { "label": "Topic Z", "count": 3 },
    ...
  ]
}
```

The route never invokes the daemon LLM. `created_ids` is always empty
when called from this skill; `pending` carries every cluster the
daemon found. The agent synthesizes them in this session — that's why
the LLM choice is consistent with how the user invoked the skill.

`unresolved` + `hint`: relay to user verbatim and stop.

### 3. Synthesize each `pending` cluster

The daemon route filters out clusters fully covered by an existing
page (subset or Jaccard ≥ 0.8). What remains is either:

- A **brand-new cluster** (no existing page) → create a new page.
- A **refresh candidate** (`existing_page_id` is set) → the cluster
  has new memories beyond what's in the matched page. The agent has
  LLM access, so the right move is to refresh the existing page in
  the same pass.

Cluster shape:

```
pending: [
  {
    source_ids, contents, entity_id, entity_name, space,
    estimated_tokens,
    existing_page_id?, existing_page_title?, new_memory_count?
  },
  ...
]
```

For each cluster, first run a **coherence check** before synthesizing:

- Skim every memory in `cluster.contents`.
- If the cluster has ≥ ~4 memories and the topics scatter (entity
  shared but the memories cover unrelated sub-topics — e.g. all tagged
  `Wenlan` but spanning RwLock bugs, schema choices, onboarding UI,
  migrations, and CSS), the cluster is **incoherent**. Skip
  synthesizing it. Record it for the report under "Skipped (low
  coherence)" with the existing page title (if refresh) or a short
  topic hint (if new).
- Coherent cluster (memories share an actual topic, not just an entity
  tag) → proceed to synthesis.

The coherence judgement is something only the agent can do — it needs
to read the prose. Daemon clustering is heuristic; agent is the
final filter against producing a grab-bag page.

For each coherent cluster:

- Title: short noun phrase. Use `existing_page_title` when refreshing
  unless the new memories materially change the topic. For new
  clusters: `cluster.entity_name` if specific, otherwise derive from
  the first memory's content.
- Summary: one sentence — the durable claim.
- Body: 3-7 paragraphs of wiki prose. Use `[[wikilinks]]`. Cite source
  ids inline with `(source: mem_XXX)`.

**New cluster** (no `existing_page_id`) — call the MCP tool:

```
write_page(title="...", summary="...", content="...",
           entity_id="<cluster.entity_id or omit>",
           space="<cluster.space>",
           source_memory_ids=[...])
```

**Refresh candidate** (`existing_page_id` is set) — replace the body
in place via the `write_page` MCP tool, passing `page_id`. This is a single atomic
call: replaces content + source list + optional summary, clears the
daemon's `stale_reason`, bumps version, preserves page_id +
created_at so external `[[wikilinks]]` keep working.

```
write_page(page_id=cluster.existing_page_id,
           content="...",
           source_memory_ids=cluster.source_ids,
           summary="...")
```

### 3.5 Refresh stale pages

The `stale_pages` block in the response lists pages whose sources
changed since last compile. Shape:

```
stale_pages: [
  { page_id, title, summary, source_memory_ids,
    sources_updated_count, stale_reason, user_edited },
  ...
]
stale_truncated: <bool>   # true when 10+ stale pages exist
```

For each stale page:

- **`user_edited == true`** → never auto-rewrite. The user touched
  the page; the upstream memories also changed. Surface in the
  "Conflict" report block and stop. The user resolves by hand, OR
  runs `/distill rebuild <page-id>` to wipe their edits and
  regenerate from sources.
- **`user_edited == false`** → fetch source memories via
  `get_page_sources(page_id="<id>")`, run the same coherence check
  used for clusters, then call `write_page` with `page_id` and the existing
  `source_memory_ids` and freshly synthesized prose.

```
write_page(page_id=stale.page_id,
           content="<refreshed prose>",
           source_memory_ids=stale.source_memory_ids,
           summary="<optional refreshed claim>")
```

When `stale_truncated == true`, tell the user "more stale pages
remain — re-run `/distill` after this pass to continue."

### 3.6 Resolve markdown paths for the report

After every successful `write_page`, resolve the
on-disk markdown file by page id. Never derive the slug client-side;
the daemon owns filename collision handling and punctuation rules.

Use `~/.wenlan/pages/.wenlan/state.json`:

```
Bash: python3 -c '
import json, os, sys
state_path = os.path.expanduser("~/.wenlan/pages/.wenlan/state.json")
pid = sys.argv[1]
filename = None
try:
    with open(state_path) as f:
        filename = json.load(f).get("pages", {}).get(pid, {}).get("file")
except FileNotFoundError:
    pass
print(f"~/.wenlan/pages/{filename}" if filename else "(no md projection on disk)")
' "<page_id>"
```

For new-page calls, parse `<page_id>` from either `Created page <id>` or
`Attached to existing page <id>`. For refreshes, use the known
`existing_page_id` or `stale.page_id`.

### 3.7 Surface orphan-topic suggestions

`orphan_topics` lists wikilink labels that 2+ existing pages reach
for but no page is named for. Each entry is a topic-discovery
signal — other pages are asking for this page.

Do **not** auto-create pages from this list — the agent doesn't have
the source memories at hand, and an empty-stub page is worse than no
page. Surface them in the report so the user can choose to run
`/distill <label>` intentionally:

```
Topic suggestions (other pages link here, no page yet):
  - "Topic Z"  (3 pages reference it)
  - "Other"    (2 pages reference it)
```

Skip the section when `orphan_topics` is empty.

### 4. Report terse

Three output shapes. Pick the one that matches what happened.

**If `pending` is empty (every cluster already fully covered):**

```
Scope `<scope>` is up to date — no new memories to distill.
```

**If at least one cluster was synthesized:**

```
Distilled N page(s) from <total> memories in scope `<scope>`:
  - <Title>  v1, synthesized from <K> sources
    Open:     ~/.wenlan/pages/<slug>.md
  - <Title>  v3 → v4: +mem_xyz, +250 chars
    Open:     ~/.wenlan/pages/<slug>.md
  ...
```

For each page, parse the `write_page` result text:

- A new-page call returns `Created page <id>` or
  `Attached to existing page <id>`, plus optional `warning: <delta>` lines.
- A refresh call returns `Refreshed page <id>` today. Use the known page id
  and fall back to `refreshed` unless the tool later returns a
  `warning: <delta>` line.

Render the first `warning: <delta>` payload verbatim after the title
when present. When no warning line is present, fall back to:

- New page: `v1, synthesized from <K> sources` (K = source_ids length)
- Refreshed page: `refreshed` (bare, as before)

This lets the user see exactly what changed per page without opening
each file.

**If at least one cluster was skipped on the coherence check:**

```
Skipped M cluster(s) — low coherence (memories share entity but
topics scatter; would produce a grab-bag page):
  - "<existing_page_title or topic hint>"  (<N> memories)
  ...
```

**If at least one stale page was refreshed:**

```
Refreshed K stale page(s):
  - <Title>  v2 → v3: +mem_abc, +180 chars
    Open:     ~/.wenlan/pages/<slug>.md
  - <Title>  refreshed
    Open:     ~/.wenlan/pages/<slug>.md
  ...
```

Same delta-line rule as new/refresh clusters: render the first
`warning: <delta>` payload from the `write_page` tool output verbatim;
fall back to `refreshed` when absent.

**If at least one stale page was skipped because `user_edited`:**

```
Conflict on L stale page(s) — page has user edits, sources also
changed. Open and reconcile manually:
  - <Title>  (~/.wenlan/pages/<slug>.md)
```

Distinct wording from the coherence-skip block so the user can tell
the two reasons apart at a glance.

Emit the blocks back-to-back when more than one outcome happened in
the same pass.

When the only outcome is skipped clusters (and `pending` was
non-empty), still emit the Skipped block. Do **not** report "up to
date" in that case — the scope isn't up to date, the candidates were
just too low quality.

Rules:
- **Titles, not page ids.** Ids visually truncate; titles read clean.
- Every synthesized or refreshed page gets its own `Open:` line with
  the daemon-resolved markdown path. No page body in chat — `/pages
  "<title>"` to preview one.
- `<total>` = sum of `source_ids.len()` across the clusters that were
  actually synthesized.
- If the pass produced fewer pages than expected, it's the clustering
  thresholds. Most memories sit alone without enough peers to form a
  cluster of 3+. Capture more on the same topic to grow them.

## Auto-commit ~/.wenlan/

After writing the pages above, snapshot the change so the user can `git
log` their memory's life timeline. Defensive — silent skip if `git` is
missing or `~/.wenlan/` is not a repo yet.

```
Bash: git -C ~/.wenlan add -A && \
      git -C ~/.wenlan -c user.name=Wenlan -c user.email=daemon@wenlan.local \
          commit --quiet -m "distill: <N> pages" 2>/dev/null || \
      (sleep 1 && git -C ~/.wenlan add -A && \
       git -C ~/.wenlan -c user.name=Wenlan -c user.email=daemon@wenlan.local \
           commit --quiet -m "distill: <N> pages" 2>/dev/null) || true
```

The retry handles index.lock races — the daemon may be writing to
`~/.wenlan/` at the same moment. Commits land at session boundaries
(handoff or daemon events), not per capture; uncommitted page edits
between sessions are normal. One second is enough for the daemon to
release the lock.

## When to use

- User says "distill", "synthesize", "rebuild the page on X".
- After a bulk import — daemon distill cycles handle this in the background;
  user can force a pass for immediate visibility.
- `/distill rebuild <page-id>` when you want to wipe a page you
  previously edited and have the daemon regenerate from current
  sources. Destructive: your prose goes away. Use after you are
  done curating a page and want it back on the auto-refresh path.

## When NOT to use

- Trivial / one-off interactions. The background scheduler covers
  periodic refresh.
- Single memory write → daemon's post-ingest enrichment already
  covers it.

## Cost

Each cluster the agent synthesizes counts against this session's
tokens. Daemon-side clusters (when an LLM is present) cost daemon LLM
tokens instead (cents on API, seconds on-device). Either way, keep
cluster sizes reasonable — the daemon already enforces a per-cluster
token budget via its tuning config.
