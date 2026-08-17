---
name: handoff
description: >
  End a Codex work session. Stores durable captures, writes a narrative session
  log, and automatically applies typed item-level deltas to the current Space
  Brief. Invoked as /handoff.
allowed-tools: ["Bash"]
user-invocable: true
---

# /handoff

Close the session with three separate artifacts:

1. A typed update to the daemon-owned Space Brief.
2. Durable MCP captures in that Space.
3. A chronological session log in `~/.wenlan/sessions/`.

The daemon Brief is current-work authority. Its
`~/.wenlan/sessions/_status/<space>.md` projection is a one-way human receipt.
Never read, edit, or overwrite that receipt as authority.

## 1. Resolve repository and Space

```bash
repo="$(git -C "$PWD" rev-parse --show-toplevel 2>/dev/null || true)"
if [ -n "$repo" ]; then
  common="$(git -C "$PWD" rev-parse --path-format=absolute --git-common-dir 2>/dev/null || true)"
  case "$common" in
    */.git) project="$(basename "$(dirname "$common")")" ;;
    *) project="$(basename "$repo")" ;;
  esac
else
  project=""
fi
handoff_arg="<the Space name passed to /handoff, empty when none>"
resolved="$(plugin-codex/bin/resolve-space.sh --cwd "$PWD" --arg "$handoff_arg" 2>/dev/null)"
space="$(printf '%s\n' "$resolved" | cut -f1)"
source_layer="$(printf '%s\n' "$resolved" | cut -f2)"
if [ -z "$space" ] && [ -n "$project" ]; then
  space="$project"
  source_layer="cwd-repo-new"
fi
```

If the user passed a Space name (`/wenlan:handoff <space>`), pass it as
`--arg <space>` to resolve-space.sh.

Print `space` and `source_layer`. Explicit pins, defaults, and mappings still
win. `cwd-repo-new` is the approved first-handoff fallback: use the canonical
repository basename, which the user can override through normal Space config.
Do not invent any other Space name.

Outside a Git repository, do not derive a new Space from the directory
basename. If resolution still leaves `space` empty, skip the Brief read and
typed update, do not issue Space-scoped captures, and continue with the
unscoped session log and any unscoped durable captures. Outside a Git
repository, a pin, explicit argument, default, or mapping still resolves a
Space, and the Brief update and captures still run.

## 2. Read the Brief before composing deltas

```bash
W="$(command -v wenlan || echo "$HOME/.wenlan/bin/wenlan")"
brief_before=""
brief_absent=0
daemon_down=0
if [ -n "$space" ]; then
  if [ "$source_layer" = "cwd-repo-new" ]; then
    space_probe_status=0
    space_probe="$("$W" --format json spaces show "$space" 2>&1)" || space_probe_status=$?
    if [ "$space_probe_status" -eq 0 ]; then
      brief_before="$("$W" --format json --space "$space" brief)"
      source_layer="cwd-repo"
    elif [ "$space_probe" = "Error: space '$space' not found" ]; then
      brief_absent=1
    elif printf '%s' "$space_probe" | grep -qE 'tcp connect error|daemon not reachable'; then
      daemon_down=1
      brief_before=""
      echo "wenlan daemon unreachable — this handoff will queue its writes"
    else
      printf "%s\n" "$space_probe" >&2
      exit "$space_probe_status"
    fi
  else
    brief_status=0
    brief_output="$("$W" --format json --space "$space" brief 2>&1)" || brief_status=$?
    if [ "$brief_status" -eq 0 ]; then
      brief_before="$brief_output"
    elif printf '%s' "$brief_output" | grep -qE 'tcp connect error|daemon not reachable'; then
      daemon_down=1
      brief_before=""
      echo "wenlan daemon unreachable — this handoff will queue its writes"
    else
      printf "%s\n" "$brief_output" >&2
      exit "$brief_status"
    fi
  fi
fi
```

Read the Brief before composing deltas. This read is mandatory before any Brief
delta is authored for a registered Space. Retain the Brief version and every
item's exact ID, version, state, text, added date, and gate. Use
`last_handoff_at` for the pending-capture window.

`brief_not_created` is valid and write-free. Use summary
`expected_version: 0`. For `cwd-repo-new`, prove the Space is absent with
`spaces show` before composing deltas. Accept only the exact CLI error
`Error: space '<name>' not found` as first-handoff absence; any other probe
failure stops the handoff. An absent Space cannot have a Brief or existing
items, so use `expected_version: 0` and author no existing-item mutations. The
typed update may then create the Space and Brief.

## 3. Preview pending captures and gather evidence

Best-effort; skip silently when `daemon_down=1` or the command fails:

```bash
"$W" --format json --space "$space" memories --pending -l 50
```

Filter by `created_at >= last_handoff_at`, or 12 hours ago when absent. Show
at most three when any match, then continue automatically; `/curate captures`
remains opt-in.

For a repository, inspect a bounded recent log, short status, diff stat, and
worktree list. Combine that evidence with the conversation. Draft atomic
captures only for durable decisions, lessons, gotchas, corrections,
preferences, and facts. Skip transient or git-recoverable state.

## 4. Write the session log

Write `~/.wenlan/sessions/<YYYY-MM-DD-HHmm>-<slug>.md` with Accomplished,
Decisions, Lessons & Gotchas, Open Threads, Captures stored, and Git summary.
This is narrative history, not current-work authority. It is a plain file and
must never depend on the daemon, so write it before the Brief update.

Best-effort commit the logical `~/.wenlan/` file batch; do not fail if no
repository is configured or a commit races.

## 5. Build and apply one typed Brief update

Compare the session outcome with `brief_before` and write one
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

Use the existing Brief version instead of `0` when present. When
`daemon_down=1`, use `expected_version: 0` — the daemon reconciles the
summary version when it replays the queued file — and author only `add`
mutations; do not author `edit`, `move`, `set_gate`, or `complete` without a
Brief snapshot.
If `space` is empty, skip this typed update entirely.

- `add`: genuinely new open work, in `active` or `backlog`, with an optional
  gate.
- `edit`, `move`, `set_gate`, and `complete`: use the exact existing item ID.
- Every delta for one existing item uses the same version from the pre-handoff Brief snapshot.
  Do not chain versions generated by earlier deltas in the same request.
- `complete` removes the item; there is no Done state.
- Never fuzzy-match. Leave an ambiguous item unchanged.
- Never auto-demote untouched Active work.
- Do not add an unchanged duplicate.

Apply exactly once:

```bash
"$W" --format json --space "$space" brief update --file "$update_file"
```

Do not ask approval for this normal handoff update. Interpret `applied`,
`conflicts`, `projection_path`, and `warnings` independently. A `"status":
"queued"` result is success-queued: record the outbox path and report it as
queued, not applied. Non-overlapping changes may commit while a stale
same-item delta conflicts. Re-read before any safe mechanical reconciliation;
never guess.

Apply the Brief update before Space-scoped captures when this fallback is new.
That creates the basename Space through the typed handoff path without making a
read or a capture create state. If this first update fails, stop Space-scoped
captures and report the exact failure. A queued result is not a failure.

## 6. Store durable captures

For each drafted durable item, run one command:

```bash
"$W" --format json --space "$space" capture -t <type> "<content>"
```

Use one atomic item per call. A `"status": "queued"` result counts as
queued, not failed — record the path. Do not ask about ordinary captures.
Pause only for a contradiction, critical incident, irreversible production
action, or genuine durability ambiguity.

## 7. Report

Report capture counts (stored vs queued), session-log path,
applied/conflicted/queued Brief deltas, Brief version or outbox path, and
projection path or warning. When anything queued, add: `N handoff write(s)
queued in the outbox — they apply when the daemon is back (`wenlan outbox
status`).`
