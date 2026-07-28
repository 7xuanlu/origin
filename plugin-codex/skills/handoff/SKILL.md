---
name: handoff
description: >
  End a Codex work session. Stores durable captures, writes a narrative session
  log, and automatically applies typed item-level deltas to the current Space
  Brief. Invoked as /handoff.
allowed-tools: ["Bash", "mcp__wenlan__capture", "mcp__wenlan__list_pending"]
user-invocable: true
---

# /handoff

Close the current work session with three distinct artifacts:

1. Durable MCP captures in the daemon.
2. A chronological session log in `~/.wenlan/sessions/`.
3. A typed update to the Space-owned Brief.

The Brief in the daemon is the source of truth for current project state.
`~/.wenlan/sessions/_status/<space>.md` is a one-way human receipt written by
the daemon. Never read, edit, or overwrite that receipt as authority.

## 1. Resolve repository and Space

```bash
repo="$(git -C "$PWD" rev-parse --show-toplevel 2>/dev/null || true)"
if [ -n "$repo" ]; then project="$(basename "$repo")"; else project="$(basename "$PWD")"; fi
resolved="$(plugin-codex/bin/resolve-space.sh --cwd "$PWD" 2>/dev/null)"
space="$(printf '%s\n' "$resolved" | cut -f1)"
source_layer="$(printf '%s\n' "$resolved" | cut -f2)"
```

Print the resolution. If no Space resolves, continue with captures and the
session log, but stop before the Brief update and report that precise gap.
Do not guess a different Space.

## 2. Read the Brief before composing deltas

When `space` is non-empty:

```bash
W="$(command -v wenlan || echo "$HOME/.wenlan/bin/wenlan")"
brief_before="$("$W" --format json --space "$space" brief)"
```

This read is mandatory before any Brief delta is authored. Retain the Brief
version plus every item's exact ID, version, state, text, added date, and gate.
Use `last_handoff_at` for the pending-capture window.

`brief_not_created` is valid: use summary `expected_version: 0`; the update may
create the Space and Brief. Reads themselves never create state.

## 3. Preview recent pending captures

Call:

```text
mcp__wenlan__list_pending(limit=50)
```

Filter by `created_at >= last_handoff_at`; use 12 hours ago when absent. If
none match, say nothing. Otherwise show at most three and proceed
automatically. `/curate captures` remains opt-in.

## 4. Gather evidence and capture durable knowledge

For a git repository, inspect recent log, short status, a bounded diff stat,
and worktree list. Combine them with the conversation.

Store one atomic durable item per call:

```text
mcp__wenlan__capture(
  content="<self-contained statement with why>",
  memory_type="<decision|lesson|gotcha|preference|fact>",
  space="<resolved only when non-empty>"
)
```

Skip transient state and facts recoverable from git. Do not ask about ordinary
captures. Pause only for a contradiction, critical incident, irreversible
production action, or genuine durability ambiguity.

## 5. Write the chronological session log

Write `~/.wenlan/sessions/<YYYY-MM-DD-HHmm>-<slug>.md` with Accomplished,
Decisions, Lessons & Gotchas, Open Threads, Captures stored, and Git summary.
This is narrative history, not current-work authority.

## 6. Build one typed Brief update

Compare the outcome with the Brief read in step 2. Create one
`BriefUpdateRequest` JSON file:

```json
{
  "space": "<resolved Space>",
  "caller_id": "codex",
  "operation_id": "<unique id retained for retries of this handoff>",
  "summary": {
    "text": "<concise last-session summary>",
    "expected_version": 0
  },
  "mutations": []
}
```

Use the existing Brief version instead of `0` when present.

- `add`: genuinely new open work, state `active` or `backlog`, optional gate.
- `edit`, `move`, `set_gate`, `complete`: exact existing `item_id` and read
  `expected_version`.
- Completion removes the item; there is no Done state.
- Never fuzzy-match. If identity is ambiguous, leave the existing item
  unchanged.
- Never auto-demote untouched Active work.
- Do not add an unchanged duplicate.

## 7. Apply automatically and inspect the receipt

```bash
"$W" --format json --space "$space" brief update --file "$update_file"
```

Do not ask for approval for this normal handoff update. Submit once. Interpret
`applied`, `conflicts`, `projection_path`, and `warnings` independently.
Non-overlapping changes may commit while a stale same-item change conflicts.
Never resolve a conflict by guessing; re-read the Brief first.

## 8. Snapshot and report

Best-effort commit the logical `~/.wenlan/` file batch at the session boundary;
do not fail if no repository is configured or a commit races.

Report capture counts, session-log path, applied/conflicted Brief deltas,
Brief version, and projection path or warning. The Markdown projection is an
inspectable receipt, never the authority.
