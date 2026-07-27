---
name: capture
description: >
  Save a durable memory to Wenlan from Codex. Use proactively when the user
  states a preference, makes a decision, corrects you, or shares a durable
  fact. Invoked as /capture <content>.
argument-hint: "<content>"
allowed-tools: ["Bash", "mcp__wenlan__capture", "mcp__wenlan__recall", "mcp__wenlan__create_entity", "mcp__wenlan__create_relation", "mcp__wenlan__list_rejections", "mcp__wenlan__accept_revision", "mcp__wenlan__dismiss_revision"]
user-invocable: true
---

# /capture

Capture one memory in the moment. Write the memory as a self-contained
statement with the reason it matters.

## Argument parsing

Accept one optional inline token of the form `space:<name>` anywhere in the
argument string. Extract it before treating the rest as content:

```bash
raw_args="<the full argument string passed to /capture>"
space_arg="$(printf '%s\n' "$raw_args" | grep -oE 'space:[A-Za-z0-9_-]+' | head -1 | cut -d: -f2)"
content="$(printf '%s\n' "$raw_args" | sed -E 's/[[:space:]]*space:[A-Za-z0-9_-]+[[:space:]]*/ /g' | sed -E 's/^[[:space:]]+|[[:space:]]+$//g')"
```

If `content` is empty, ask the user what they want to capture.

## Resolve the active space

Call the Codex resolver:

```bash
resolved="$(plugin-codex/bin/resolve-space.sh --cwd "$PWD" ${space_arg:+--arg "$space_arg"} --topic "$content" 2>/dev/null)"
space="$(printf '%s\n' "$resolved" | cut -f1)"
source_layer="$(printf '%s\n' "$resolved" | cut -f2)"
```

If `space` is non-empty, print `Resolved space: <space> (from <source-layer>)`
and pass it to the `capture` MCP tool. If `space` is empty, print
`Resolved space: none (unscoped)` and omit the `space` parameter.

If `source_layer` is `arg`, also print:

```text
Created new space '<space>' from arg. Register it later if you want it pinned.
```

## How to invoke

Call the Wenlan MCP `capture` tool with the user's content as a complete,
self-contained statement. Attach topic from cwd or the conversation when useful.

```text
capture(
  content="<content, written as a full sentence with WHY>",
  memory_type="<picked from the 6 types>",
  entity="<primary entity name, if any>",
  space="<resolved if non-empty>"
)
```

## Pick `memory_type`

The daemon classifies when a model or API key is configured. In local memory
mode it may not, so pick the type from the content itself.

| Type | Use for |
|---|---|
| `identity` | Durable facts about the user |
| `preference` | A habit, correction, or stylistic choice with a reason |
| `decision` | A specific choice with rationale |
| `lesson` | Root cause, workaround, or technical insight |
| `gotcha` | Sharp edge or surprising behavior |
| `fact` | Durable info about people, projects, tools, or places |

If two types fit, pick the one closest to why the memory matters.

## Extract `entity`

Pick the single most important named anchor: person, project, tool, or place.
Use the exact name. If there is no named anchor, omit `entity`.

Pass the single most important named anchor through `capture.entity`. Ordinary
captures stop there; daemon enrichment handles routine extraction. Do not call
`mcp__wenlan__create_entity` for every capture, and never infer a relation the
user did not state.

Only when the user explicitly states a durable relation:

1. Call `mcp__wenlan__capture` first with the complete relation statement.
2. Call `mcp__wenlan__create_entity` for both named endpoints to resolve their
   stable ids. The call is idempotent and may return an existing id.
3. Call `mcp__wenlan__create_relation` with `from_entity_id`, `to_entity_id`,
   `relation_type`, and the capture result's required `source_memory_id`.

For a durable named entity explicitly established by the user,
`mcp__wenlan__create_entity` may also be used alone when its stable id is
needed. `Entity <id> ready` does not imply that a new row was created.

## What to capture

- Decisions: "Going with approach A because B."
- Preferences: "Prefers TDD because it catches regressions early."
- Corrections: "Actually it is C, not D."
- Project facts: "Wenlan is the local memory daemon for AI tools."

## What not to capture

- System prompts, boot logs, and command output.
- Transient task state.
- File paths or git history the user can re-derive.
- Agent operating rules that belong in AGENTS.md or another obey-tier file.
- Single-word acknowledgments.

If a capture the user expected is missing or the user explicitly asks why it
was rejected, call `mcp__wenlan__list_rejections`. Do not list rejections after
successful ordinary captures.

## Post-capture contradiction signal

After `capture` returns, check `triggered_revisions` and `auto_superseded`.

If `auto_superseded` is non-empty, surface it as informational. No follow-up
tool call is needed.

If `triggered_revisions` is non-empty and `auto_superseded` is empty, render:

```text
Stored <new id>.

This capture topic-matches a protected memory now flagged for revision:
  - <target id>

Action: accept (replace original) | dismiss (drop revision) | leave (decide later)
```

Inline verb map:

- accept: `accept_revision(target_source_id="<target id>")`
- dismiss: `dismiss_revision(target_source_id="<target id>")`
- leave: no call
