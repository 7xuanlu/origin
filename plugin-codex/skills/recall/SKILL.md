---
name: recall
description: >
  Search Wenlan's local memory from Codex by query. Targeted lookup, not
  session orientation. Invoked as /recall <query>.
argument-hint: "<query>"
allowed-tools: ["Bash", "mcp__wenlan__recall", "mcp__wenlan__get_memory_revisions"]
user-invocable: true
---

# /recall

Search Wenlan memory by natural-language query. Use `/brief` for broad session
orientation and `/recall` for a specific fact, decision, lesson, or preference.

## Argument parsing

Accept one optional inline token of the form `space:<name>` anywhere in the
argument string. Extract it before treating the rest as the query:

```bash
raw_args="<the full argument string passed to /recall>"
space_arg="$(printf '%s\n' "$raw_args" | grep -oE 'space:[A-Za-z0-9_-]+' | head -1 | cut -d: -f2)"
query="$(printf '%s\n' "$raw_args" | sed -E 's/[[:space:]]*space:[A-Za-z0-9_-]+[[:space:]]*/ /g' | sed -E 's/^[[:space:]]+|[[:space:]]+$//g')"
```

If `query` is empty, ask the user what they want to search for.

## Resolve the active space

Call the Codex resolver:

```bash
resolved="$(plugin-codex/bin/resolve-space.sh --cwd "$PWD" ${space_arg:+--arg "$space_arg"} 2>/dev/null)"
space="$(printf '%s\n' "$resolved" | cut -f1)"
source_layer="$(printf '%s\n' "$resolved" | cut -f2)"
```

If `space` is non-empty, print:

```text
Resolved space: <space> (from <source-layer>)
```

If `space` is empty, print:

```text
Resolved space: none (unscoped)
```

Pass `space="<resolved>"` to `mcp__wenlan__recall` only when `space` is
non-empty. Do not substitute `personal` for an unscoped result.

## Expand once, then call recall

Rewrite the query only enough to improve retrieval:

- Replace pronouns with the referent from the current thread.
- Expand an abbreviation when the embedder may miss it.
- Add one obvious synonym when the user's wording is too narrow.

Do not issue multiple recall calls. Use one call:

```text
mcp__wenlan__recall(
  query="<expanded query>",
  space="<resolved if non-empty>",
  memory_type="<only if the query names a type>"
)
```

Infer `memory_type` only when the user names a type, such as "decision on X",
"lesson about Y", or "preference for Z". Otherwise omit it.

## Rerank and render

Rerank the returned memories against the user's original query:

- Promote direct answers.
- Demote keyword overlaps that do not answer the question.
- Show the top 3-5 hits unless the user asked for a deeper audit.

Render revision context only when it matters:

```text
<id>  v<N> (merged <K> memories)
<id>  v<N>, pending revision against <id>
<id>  v<N> - <last_delta_summary>
<id>  v<N>
```

Skip that tag line when the memory is a fresh v1 with no merge or pending
revision fields.

Only when the user explicitly asks for one memory's history or evolution, or
whether a correction was recorded, call:

```text
mcp__wenlan__get_memory_revisions(memory_id="<source_id>")
```

Do not fetch revision chains for ordinary recall results.

## When to use

- "What did I say about X?"
- "Do you remember the decision on Y?"
- "Look up my preference for Z."

## When not to use

- Session start context: use `/brief`.
- Storing a new memory: use `/capture`.
