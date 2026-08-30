---
name: brief
description: >
  Read the current Space-owned project Brief from Wenlan for Codex. With an
  optional topic, appends separately labeled related context from the same
  Space. Invoked as /brief [topic] when resuming work or asking to catch up.
argument-hint: "[topic]"
allowed-tools: ["Bash", "mcp__wenlan__brief", "mcp__wenlan__list_pending_revisions", "mcp__wenlan__accept_revision", "mcp__wenlan__dismiss_revision"]
user-invocable: true
---

# /brief

Read one Space's current project snapshot. The daemon Brief is authoritative;
`~/.wenlan/sessions/_status/*.md` is only a one-way human-readable receipt and must
never be read as product state.

## 1. Resolve the Space

```bash
raw_args="<the full argument string passed to /brief>"
space_arg="$(printf '%s\n' "$raw_args" | grep -oE 'space:[A-Za-z0-9_-]+' | head -1 | cut -d: -f2)"
topic_arg="$(printf '%s\n' "$raw_args" | sed -E 's/[[:space:]]*space:[A-Za-z0-9_-]+[[:space:]]*/ /g' | sed -E 's/^[[:space:]]+|[[:space:]]+$//g')"
# The resolver ships inside the installed plugin; the relative path only
# exists in a wenlan checkout.
resolver="$(find "$HOME/.codex/plugins/cache" -path '*/wenlan/*/bin/resolve-space.sh' 2>/dev/null | head -1)"
resolved="$("${resolver:-plugin-codex/bin/resolve-space.sh}" --cwd "$PWD" ${space_arg:+--arg "$space_arg"} 2>/dev/null)"
space="$(printf '%s\n' "$resolved" | cut -f1)"
source_layer="$(printf '%s\n' "$resolved" | cut -f2)"
```

Print `Resolved space: <space> (from <source-layer>)`. If no Space resolves,
print `Resolved space: none (unscoped)` and omit `space`.

## 2. Read the Brief

Call:

```text
mcp__wenlan__brief(
  topic="<topic_arg only when the user supplied one>",
  space="<resolved only when non-empty>"
)
```

Do not invent or infer a topic when the argument is absent. No topic means the
complete Brief alone. A topic means the same complete Brief plus a separate
`Related Context` section scoped to that Space.

Brief reads never create state. If the state is `brief_not_created`, explain
that the first `/handoff` update will create it. If the state is
`space_not_resolved`, ask for a Space only when the working directory and
configuration cannot resolve one safely.

Render the last-session summary, Active, Backlog, then Related Context only
when returned. Keep stable item IDs and versions available for later
`/handoff` reconciliation, but omit them from the normal user-facing list.

## 3. Pending revisions

Call:

```text
mcp__wenlan__list_pending_revisions(limit=10)
```

If empty, say nothing. If it fails, emit one warning and keep the Brief. If
non-empty, show at most three. Never auto-action them.

- accept: `mcp__wenlan__accept_revision(target_source_id="<id>")`
- dismiss: `mcp__wenlan__dismiss_revision(target_source_id="<id>")`
- skip: no call

If more than three exist, point to `/curate revisions`.

## Boundary

Use `/brief` to resume a Space or answer "catch me up." It is not a mandatory
every-session boot step. Use `/recall` for a specific fact, `/capture` for a
durable memory, and `/handoff` to close a work session.
