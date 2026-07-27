---
name: brief
description: >
  Session-start briefing from Wenlan for Codex. Reads the project status file,
  loads identity/preferences/topic memories, and surfaces pending revisions.
  Invoked as /brief [topic].
argument-hint: "[topic]"
allowed-tools: ["Bash", "mcp__wenlan__context", "mcp__wenlan__recall", "mcp__wenlan__list_pending_revisions", "mcp__wenlan__accept_revision", "mcp__wenlan__dismiss_revision"]
user-invocable: true
---

# /brief

Pull a curated session brief from Wenlan. Use it at session start or after a
major topic shift.

Sources, in order:

1. Project status file: what `/handoff` last wrote.
2. `context` MCP: identity, preferences, and topic-relevant memories.
3. `list_pending_revisions`: daemon-flagged memories awaiting human review.

The status file wins for "what is next" because it is the live ledger.

## 1. Read project status first

Detect project root:

```bash
cd_repo="$(git -C "$PWD" rev-parse --show-toplevel 2>/dev/null || true)"
if [ -n "$cd_repo" ]; then
  project="$(basename "$cd_repo")"
else
  project="$(basename "$PWD")"
fi
printf '%s\n' "$project"
```

Read `~/.wenlan/sessions/_status/<project>.md`:

```bash
cat "$HOME/.wenlan/sessions/_status/<project>.md"
```

If the file exists, render its `## Last session`, `## Active`, and `## Backlog`
sections verbatim at the top under `Status`. If the file is missing, say
nothing about it.

## 2. Resolve topic and space

Accept an optional topic argument and an optional inline `space:<name>` token.
Extract the space token before using the rest as topic text:

```bash
raw_args="<the full argument string passed to /brief>"
space_arg="$(printf '%s\n' "$raw_args" | grep -oE 'space:[A-Za-z0-9_-]+' | head -1 | cut -d: -f2)"
topic_arg="$(printf '%s\n' "$raw_args" | sed -E 's/[[:space:]]*space:[A-Za-z0-9_-]+[[:space:]]*/ /g' | sed -E 's/^[[:space:]]+|[[:space:]]+$//g')"
```

Call the Codex resolver:

```bash
resolved="$(plugin-codex/bin/resolve-space.sh --cwd "$PWD" ${space_arg:+--arg "$space_arg"} 2>/dev/null)"
space="$(printf '%s\n' "$resolved" | cut -f1)"
source_layer="$(printf '%s\n' "$resolved" | cut -f2)"
```

If `space` is non-empty, print `Resolved space: <space> (from <source-layer>)`
and pass it to the `context` MCP call. If it is empty, print
`Resolved space: none (unscoped)` and omit the `space` parameter.

## 3. Call context

Call the Wenlan MCP `context` tool:

```text
context(topic="<topic_arg or inferred topic>", space="<resolved if non-empty>")
```

If the user omitted a topic, infer it from the working directory and recent
conversation. Do not ask unless inference would be misleading.

Use the result to model how the user thinks, not just to retrieve facts. Their
preferences, corrections, and past decisions shape how you should work.

## 4. Pending revisions check

After loading context, call:

```text
list_pending_revisions(limit=10)
```

If the result is empty, say nothing. If the call errors, print one warning and
continue.

If non-empty, show the top three:

```text
Pending revisions (<N> total, top 3 shown):

1. target: <target id>  (proposed by <source_agent or "daemon">)
   revision: "<revision text>"
   Action: accept (replace original) | dismiss (drop revision) | skip
```

Inline verbs map to:

- accept: `accept_revision(target_source_id="<id>")`
- dismiss: `dismiss_revision(target_source_id="<id>")`
- skip: no call

If there are more than three, end with:

```text
Run /curate revisions to walk the full queue.
```

Do not auto-action revisions.
