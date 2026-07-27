---
name: curate
description: >
  Review pending Wenlan captures, revisions, or daemon refinements from Codex.
  Use for explicit audit walks after /brief or /handoff surfaces pending work.
  Invoked as /curate captures, /curate revisions, or /curate refinements.
argument-hint: "captures | revisions | refinements"
allowed-tools: ["Bash", "mcp__wenlan__list_pending", "mcp__wenlan__confirm_memory", "mcp__wenlan__forget", "mcp__wenlan__capture", "mcp__wenlan__recall", "mcp__wenlan__list_refinements", "mcp__wenlan__accept_refinement", "mcp__wenlan__reject_refinement"]
user-invocable: true
---

# /curate

Power-user audit of Wenlan pending surfaces. Daily flow should stay light:
`/brief` shows the top pending revisions and `/handoff` previews recent pending
captures. Use `/curate` only when the user asks to walk the queue.

## Safety protocol

Perform no mutation until the user replies. Ambiguous replies do not mutate.
Cancellation, no reply, or unclear syntax is a no-op.

Render at most four items at a time. Number every item. Ask for explicit action
syntax:

```text
1 accept
2 dismiss
3 skip
4 edit: <replacement text>
```

Do not rely on hidden pagination state across turns. After applying a batch,
re-list before showing another batch.

## `/curate revisions`

Use the local CLI, not MCP, because revision actions must key on the exact
`revision_source_id`.

Resolve `wenlan`:

```bash
if command -v wenlan >/dev/null 2>&1; then
  W="$(command -v wenlan)"
elif [ -x "$HOME/.wenlan/bin/wenlan" ]; then
  W="$HOME/.wenlan/bin/wenlan"
else
  echo "wenlan not found"
  exit 127
fi
"$W" --format json curate
```

If `wenlan` is missing, tell the user to run `/setup`.

The JSON array contains:

- `revision_source_id`: staged revision id; action key for accept/dismiss.
- `target_source_id`: original memory id; context only.
- `content`: revised text.
- `source_agent`: proposer.
- `original`: current text when available.
- `diff`: ready-made OLD/NEW preview when available.

Render up to four revisions:

```text
Pending revisions:

1. revision_source_id: <rev_id>
   target: <target_source_id>
   proposed: "<content>"
   diff:
   <diff or "(original unavailable)">

Reply with: 1 accept, 1 dismiss, 1 skip
```

Apply only explicit actions:

```bash
"$W" curate accept <revision_source_id>
"$W" curate dismiss <revision_source_id>
```

`accept` replaces the original memory with that named revision. `dismiss`
unstages the false revision link and keeps both memories as independent rows.
`skip` does nothing.

## `/curate captures`

If the user names a space, parse `space:<name>` and pass it to
`mcp__wenlan__list_pending`. Otherwise leave `space` off so the audit covers
all pending captures.

Call:

```text
mcp__wenlan__list_pending(limit=50, space="<only when user named one>")
```

Render up to four:

```text
Pending captures:

1. <source_id>  <memory_type>  <source_agent>
   "<content>"

Reply with: 1 accept, 1 reject, 1 skip, 1 edit: <replacement text>
```

Apply only explicit actions:

- `accept`: `mcp__wenlan__confirm_memory(memory_id="<source_id>")`
- `reject`: `mcp__wenlan__forget(memory_id="<source_id>")`
- `skip`: no call
- `edit: <replacement>`:
  1. `mcp__wenlan__capture(content="<replacement>", supersedes="<source_id>")`
  2. `mcp__wenlan__forget(memory_id="<source_id>")`

Reject and edit delete the original pending capture, so require the numbered
item and action verb in the same reply.

## `/curate refinements`

Use `/curate refinements` only when the user explicitly asks to inspect or
review the daemon proposal/refinement queue. Never poll the refinement queue
ambiently.

List first with `mcp__wenlan__list_refinements(limit=50)` and show at most four
items. Include each proposal id, action, confidence, and a bounded payload
summary. The queue includes `vocab_promote` proposals.

Perform no mutation until the user gives an unambiguous item-level accept or
reject decision. Apply only numbered item decisions:

- `accept`: `mcp__wenlan__accept_refinement(id="<proposal-id>")`
  (include `space` for `cross_space_discovery`).
- `reject`: `mcp__wenlan__reject_refinement(id="<proposal-id>")`
- `skip`: no call

Skip or cancel is a no-op. Do not generically accept `lint_repair_review`;
route it through `/lint repair`. Re-list after every mutation batch.

## Bare `/curate`

Default to `/curate revisions`. Revisions are the conflict surface that most
often needs human judgment. Mention `/curate captures` only as an opt-in audit.

## When not to use

- Specific lookup: use `/recall`.
- One-off correction: use `/capture` with the corrected fact.
