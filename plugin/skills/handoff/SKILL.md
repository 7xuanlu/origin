---
name: handoff
description: >
  End a work session. Stores durable captures, writes a narrative session log,
  and automatically applies typed item-level deltas to the current Space Brief.
  Invoked as `/handoff`.
allowed-tools: ["Bash", "mcp__plugin_wenlan_wenlan__capture", "mcp__plugin_wenlan_wenlan__list_pending"]
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
resolved="$("$CLAUDE_PLUGIN_ROOT/bin/resolve-space.sh" --cwd "$PWD" 2>/dev/null)"
space="$(printf '%s\n' "$resolved" | cut -f1)"
source_layer="$(printf '%s\n' "$resolved" | cut -f2)"
```

Print the resolution. If no Space resolves, continue with captures and the
session log, but stop before the Brief update and report that precise gap.
Do not guess a different Space.

## 2. Read the Brief before composing deltas

When `space` is non-empty, run:

```bash
W="$(command -v wenlan || echo "$HOME/.wenlan/bin/wenlan")"
brief_before="$("$W" --format json --space "$space" brief)"
```

This read is mandatory before any Brief delta is authored. Retain:

- Brief `version` for the summary CAS.
- Every active/backlog item `id`, `version`, `state`, `text`, `added_at`, and
  optional `gate`.
- `last_handoff_at` for the pending-capture window.

`brief_not_created` is valid: use summary `expected_version: 0`; the update may
create the Space and Brief. Reads themselves never create state.

## 3. Preview recent pending captures

Call:

```text
mcp__plugin_wenlan_wenlan__list_pending(limit=50)
```

Filter by `created_at >= last_handoff_at`. If the Brief has no
`last_handoff_at`, use 12 hours ago. If none match, say nothing. Otherwise show
at most three and proceed automatically; `/curate captures` remains opt-in.

## 4. Gather evidence and infer durable captures

For a git repository, inspect:

```bash
git -C "$repo" log --oneline -20
git -C "$repo" status --short
git -C "$repo" diff --stat HEAD~5..HEAD 2>/dev/null || true
git -C "$repo" worktree list
```

Combine this with the conversation. Store only durable decisions, lessons,
gotchas, corrections, and facts. Skip transient state and facts recoverable
from git.

For each durable item, call one atomic capture:

```text
mcp__plugin_wenlan_wenlan__capture(
  content="<self-contained statement with why>",
  memory_type="<decision|lesson|gotcha|preference|fact>",
  space="<resolved only when non-empty>"
)
```

Do not ask about ordinary captures. Pause only for a contradiction, critical
incident, irreversible production action, or genuine durability ambiguity.

## 5. Write the chronological session log

Write `~/.wenlan/sessions/<YYYY-MM-DD-HHmm>-<slug>.md` with Accomplished,
Decisions, Lessons & Gotchas, Open Threads, Captures stored, and Git summary.
This log is narrative history, not the current-work authority.

## 6. Build one typed Brief update

Compare the session outcome with the Brief read in step 2. Create one
`BriefUpdateRequest` JSON file:

```json
{
  "space": "<resolved Space>",
  "caller_id": "claude-code",
  "operation_id": "<unique id retained for retries of this handoff>",
  "summary": {
    "text": "<concise last-session summary>",
    "expected_version": 0
  },
  "mutations": []
}
```

Use the read Brief version instead of `0` when the Brief already exists.
Mutation rules:

- `add`: a genuinely new open item; choose `active` or `backlog`, with optional
  gate. The daemon supplies the added date when omitted.
- `edit`, `move`, `set_gate`, `complete`: use the exact existing `item_id` and
  its read `expected_version`.
- Completed work uses `complete`; it does not become a hidden third state.
- Never fuzzy-match an existing item. If identity is ambiguous, leave it
  unchanged and do not manufacture an edit or completion.
- Never auto-demote an untouched Active item. Active/Backlog changes must come
  from actual session evidence.
- Do not add an item that already exists unchanged.

## 7. Apply automatically and inspect the receipt

```bash
"$W" --format json --space "$space" brief update --file "$update_file"
```

Do not ask for approval for this normal handoff update. Submit exactly once,
then parse the typed receipt:

- `applied` confirms item-level changes.
- `conflicts` contains only stale summary/item versions or missing IDs.
- `projection_path` identifies the human Markdown receipt.
- `warnings` report projection failures without rolling back committed state.

Non-overlapping mutations may apply even when another mutation conflicts.
Never retry a conflict by guessing. Re-read the Brief and report the exact
conflicted item if a safe reconciliation is not mechanical.

## 8. Snapshot and report

Best-effort commit the logical `~/.wenlan/` file batch at the session boundary;
do not fail the handoff if no repository is configured or the commit races.

Report capture counts, session-log path, applied/conflicted Brief deltas,
Brief version, and projection path or warning. Do not claim the receipt is the
authority; it is only the inspectable projection.
