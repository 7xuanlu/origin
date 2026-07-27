---
name: capture
description: >
  Save a memory to Wenlan in flow. Active capture verb — use proactively
  when the user states a preference, makes a decision, corrects you, or
  shares a durable fact. Invoked as `/capture <content>`.
argument-hint: "<content>"
allowed-tools: ["mcp__plugin_wenlan_wenlan__capture", "mcp__plugin_wenlan_wenlan__recall", "mcp__plugin_wenlan_wenlan__create_entity", "mcp__plugin_wenlan_wenlan__create_relation", "mcp__plugin_wenlan_wenlan__list_rejections", "mcp__plugin_wenlan_wenlan__accept_revision", "mcp__plugin_wenlan_wenlan__dismiss_revision", "Bash"]
---

# /capture

Capture a single memory in the moment. Active verb: agent captures the
moment of insight, like a photograph.

## Argument parsing

The `/capture` skill accepts one optional inline token of the form
`space:<name>` anywhere in the argument string. Extract it before
treating the rest as content:

    raw_args="<the full argument string passed to /capture>"
    space_arg="$(printf '%s\n' "$raw_args" | grep -oE 'space:[A-Za-z0-9_-]+' | head -1 | cut -d: -f2)"
    content="$(printf '%s\n' "$raw_args" | sed -E 's/[[:space:]]*space:[A-Za-z0-9_-]+[[:space:]]*/ /g' | sed -E 's/^[[:space:]]+|[[:space:]]+$//g')"

If `space_arg` is non-empty, pass it to the resolver as `--arg "$space_arg"`.

## Resolve the active space

Call the bundled resolver:

    resolved="$("$CLAUDE_PLUGIN_ROOT/bin/resolve-space.sh" --cwd "$PWD" \
        ${space_arg:+--arg "$space_arg"} 2>/dev/null)"
    space="$(printf '%s\n' "$resolved" | cut -f1)"
    source_layer="$(printf '%s\n' "$resolved" | cut -f2)"

Pass `space="$space"` to the `capture` MCP tool only when `space` is
non-empty. Before every capture, also print:

    Resolved space: <space> (from <source-layer>)

If `space` is empty, print:

    Resolved space: none (unscoped)

Unknown spaces are not auto-created. Register one through the resolved CLI
binary, or omit `space` to store uncategorized:

```bash
W="$(command -v wenlan || echo "$HOME/.wenlan/bin/wenlan")"
"$W" spaces add <space>
```

## How to invoke

Call the `wenlan` MCP server's `capture` tool with the user's content as a
complete, self-contained statement. Attach `topic` from cwd or the
conversation — don't make the user type it.

```
capture(content="<args, written as a full sentence with WHY>",
        memory_type="<picked from the 6 types>",
        entity="<primary entity name, if any>",
        space=<resolved if non-empty>)
```

### `memory_type` — agent picks one of 6

The daemon classifies when a local model or API key is configured. In
local memory mode it does not, so the agent picks the type from the content itself. Use this
mapping:

| Type | Use for |
|---|---|
| `identity` | Durable facts about the user (role, company, language preference) |
| `preference` | "I prefer X because Y" — a habit, a correction, a stylistic choice |
| `decision` | "Going with A over B because C" — a specific choice with rationale |
| `lesson` | Root cause found, workaround discovered, technical insight earned |
| `gotcha` | Sharp edge, surprising behavior, a thing to watch out for |
| `fact` | Durable info about people, projects, tools — anchor to `entity` when possible |

If two types fit, pick the one closest to *why the memory matters*. A
decision *also* implies a preference, but `decision` is more specific.

### `entity` — extract the anchor

Pick the single most important named thing in the content: a person,
project, tool, place. Use the exact name. Example: "Alice prefers TDD
because…" → `entity="Alice"`. If the content has no named anchor,
omit `entity`.

### `topic` / `space` inference

- cwd inside a repo → repo name (e.g. `~/Repos/wenlan/...` → `"wenlan"`).
- Outside any repo → most recent topic from the conversation, or omit.
- Pass `space` only when scope is known. If uncertain, omit it or inspect the
  registered spaces first:

```bash
W="$(command -v wenlan || echo "$HOME/.wenlan/bin/wenlan")"
"$W" spaces list
```

### Multiple entities or relations

Ordinary captures stop after `capture`: pass the single most important named
anchor through `capture.entity` and let daemon enrichment handle routine
extraction. Do not call `create_entity` for every capture, and never infer a
relation the user did not state.

Use the explicit KG tools only when the user directly states a durable relation:

1. Call `create_entity` for both named endpoints first and collect their stable
   ids. This is idempotent and may return an existing id.
2. Call `capture` with the complete relation statement and pass the primary
   entity name as `entity` so the memory resolves and links to it.
3. Call `create_relation` with `from_entity_id`, `to_entity_id`,
   `relation_type`, and the capture result's required `source_memory_id`.

For a durable named entity explicitly established by the user, `create_entity`
may also be used alone when its stable id is needed. Its `Entity <id> ready`
result does not imply that a new row was created.

## What to capture

- Decisions: "Going with approach A because B"
- Preferences: "Prefers TDD because catches regressions early"
- Corrections: "Actually it's C, not D"
- Identity / project facts: "Works on Wenlan, a local memory daemon for AI tools"

## What NOT to capture

- System prompts, boot logs, heartbeats
- Transient task state ("currently working on...")
- Tool output, command results, architecture dumps
- Single-word acknowledgments
- Things the user can trivially re-derive (file paths, recent git history)
- Agent operating rules — "always X" / "never Y" directives about how the
  agent should behave. Those belong in CLAUDE.md / AGENTS.md / MEMORY.md (the
  obey tier), not Wenlan. Capture the user's *preference* ("prefers TDD
  because…"), not the agent-facing *rule* ("always run TDD first").

## Atomic ideas

One capture = one idea. "Prefers TDD" and "Uses pytest" are two captures, not
one.

## When to use

- User explicitly says "remember this", "save that", "capture this".
- User states a durable preference / decision / correction proactively (no
  ask required — that's the floor, not the trigger).

## When NOT to use

- End of session bulk store → use `/handoff` (multi-item batch).
- Pulling memories back out → use `/recall`.

If a capture the user expected is missing or the user explicitly asks why it
was rejected, call `list_rejections` to diagnose the quality gate. Do not list
rejections after successful ordinary captures.

## Post-capture contradiction signal

After `capture` returns, check `response.triggered_revisions` and `response.auto_superseded`.

### auto_superseded (no action needed)

If `auto_superseded` is non-empty, the daemon already resolved the contradiction. Surface it as informational:

```
Note: auto-superseded mem_X. Wenlan replaced a prior protected memory because
trust=high and similarity > 0.9. No action needed.
```

No accept/dismiss call required. The revision was applied automatically.

### triggered_revisions (human review needed)

If `triggered_revisions` is non-empty (and `auto_superseded` is empty), render an inline block to the user:

```
Stored mem_new.

This capture topic-matches a protected memory now flagged for revision:
  - mem_target_abc

Action: accept (replace original content) | dismiss (drop the revision) | leave (decide later)
```

Inline verb map:

- accept: `accept_revision(target_source_id="mem_target_abc")`
- dismiss: `dismiss_revision(target_source_id="mem_target_abc")`
- leave: no call; surfaces again in next `/brief`

Both fields can technically be non-empty in a single response (multiple protected matches), but in practice only one fires per capture: `auto_superseded` fires when trust=full and similarity > 0.9, `triggered_revisions` fires otherwise.

If neither field is non-empty, the capture stored cleanly with no conflicts.
