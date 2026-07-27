---
name: recall
description: >
  Search Wenlan's local memory by query. Targeted lookup, not orientation.
  Invoked as `/recall <query>`. Use when the user asks "do you
  remember", "what do you know about", "look up".
argument-hint: "<query>"
allowed-tools: ["mcp__plugin_wenlan_wenlan__recall", "mcp__plugin_wenlan_wenlan__get_memory_revisions"]
---

# /recall

Search Wenlan's memory by natural-language query. Returns matching memories
ranked by hybrid vector + FTS search, then re-ordered by the agent if it
helps.

## Argument parsing

The `/recall` skill accepts one optional inline token of the form
`space:<name>` anywhere in the argument string. Extract it before
treating the rest as the query:

    raw_args="<the full argument string passed to /recall>"
    space_arg="$(printf '%s\n' "$raw_args" | grep -oE 'space:[A-Za-z0-9_-]+' | head -1 | cut -d: -f2)"
    query="$(printf '%s\n' "$raw_args" | sed -E 's/[[:space:]]*space:[A-Za-z0-9_-]+[[:space:]]*/ /g' | sed -E 's/^[[:space:]]+|[[:space:]]+$//g')"

If `space_arg` is non-empty, pass it to the resolver as `--arg "$space_arg"`.

## Resolve the active space

Call the bundled resolver:

    resolved="$("$CLAUDE_PLUGIN_ROOT/bin/resolve-space.sh" --cwd "$PWD" \
        ${space_arg:+--arg "$space_arg"} 2>/dev/null)"
    space="$(printf '%s\n' "$resolved" | cut -f1)"
    source_layer="$(printf '%s\n' "$resolved" | cut -f2)"

Pass `space="$space"` to the `recall` MCP tool only when `space` is
non-empty. Print one line before the call:

    Resolved space: <space> (from <source-layer>)

If `space` is empty, print `Resolved space: none (unscoped)` and omit the
space filter.

## Two phases

When a local model or API key is configured, the daemon can rerank and
expand server-side. In local memory mode it cannot. The skill always does
**agent-side expansion and rerank** itself — cheap, makes results good in
both modes.

### Phase 1 — expand the query (agent-side)

Before calling `recall`, rewrite the user's query into a more
search-friendly form:

- Replace pronouns with the referent ("it" → the actual thing).
- Expand abbreviations the embedder is unlikely to know.
- Add the obvious synonym when the original term is too narrow (e.g.
  "auth" → "auth OR authentication").

Don't over-expand. If the query is already specific, leave it alone.
One recall call per `/recall` invocation — duplicate calls double
embedding load and the merge step is rarely worth it. The daemon's
own `search_memory_expanded` exists for the multi-query case; if it
matters, use that endpoint instead of issuing parallel calls here.

### Phase 2 — call the MCP tool

```
recall(query="<expanded query>", space=<resolved if non-empty>, memory_type=<inferred>)
```

Inferences (do not ask the user):

- `space`: current working directory (e.g. `~/Repos/wenlan/...` → `"wenlan"`),
  the topic being discussed, or whatever space was mentioned in recent turns.
  Pass only when scope is known; omit when uncertain or unscoped.
- `memory_type`: only when the query itself names a type ("decision on X",
  "lesson about Y", "preference for Z"). Otherwise omit and let hybrid
  search rank.
- `limit`: default 10. Use 3-5 for quick lookups, 10-20 for exploration.

### Phase 3 — rerank (agent-side)

The daemon returns hits ranked by hybrid search. That ranking is good but
not perfect — it doesn't know the user's exact intent.

Re-read the returned memories against the *original* query. Promote the
ones that directly answer the question; demote ones that just share
keywords.

Show the user the top 3-5 reranked hits. Surface the rest only if asked.

If the user explicitly asks how one returned memory changed, for its full
history, or whether a correction was recorded, call
`get_memory_revisions(memory_id="<source_id>")`. Do not fetch revision chains
for ordinary recall results.

### Phase 4 — render revision context (per result)

Each memory may carry revision fields: `version`, `pending_revision`,
`merged_from`, `last_delta_summary`. Most memories are fresh (v1, none
set) — render nothing extra for those. Only add a tag line when
something meaningful is present.

**Condition:** emit the tag line when any of these holds:
- `version > 1`
- `merged_from` is non-empty
- `pending_revision == true`

**Format** — one compact line above the memory body:

```
<id>  v<N> (merged <K> memories)         ← merged_from has K entries
<id>  v<N>, pending revision against <id> ← pending_revision true
<id>  v<N> — <last_delta_summary>         ← version > 1, delta populated
<id>  v<N>                                ← version > 1, no delta
```

Rules:
- Merged takes precedence over pending_revision in the label.
- Omit `— <delta>` when `last_delta_summary` is empty or null.
- Skip the tag line entirely when version == 1 (or null) and no other
  flag is set. Preserves current output for fresh memories.

## When to use

- "What did I say about X?"
- "Do you remember the decision on Y?"
- Need a specific fact before continuing.

## When NOT to use

- Broad session orientation → use `/brief` instead.
- Storing a new memory → use `/capture`.

## Hint: write specific queries

"Alice database preference" finds more than "database stuff". The semantic
matcher rewards specificity. If too many results return, add filters rather
than making the query longer.
