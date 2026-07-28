# Wenlan Skills

Claude Code workflow skills installed by the Wenlan plugin.

These skills keep the daily interface short:

```text
/setup       verify setup end-to-end
/help        one-screen reference
/brief       load session context
/capture     save one durable memory
/recall      search local memory
/lint [deep|repair] [scope]   diagnose, or resolve all findings safely
/distill     refresh wiki pages
/pages [q]   browse + open distilled pages (wenlan pages)
/curate captures|revisions   power-user deep audit; daily flow is /brief
/forget      delete a memory by ID
/handoff     end-of-session debrief
```

The skills do not store data themselves. They guide Claude Code to use the local `wenlan-mcp` tools, which talk to the Wenlan daemon on `127.0.0.1:7878`.

## Files

| Skill | Purpose |
| --- | --- |
| `setup` | End-to-end setup verifier (local runtime + MCP + round-trip). |
| `help` | One-screen quick reference of the daily verbs and flow. |
| `brief` | Load working context at session start or topic shifts. |
| `capture` | Save one durable memory: decision, lesson, gotcha, preference, fact, or correction. |
| `recall` | Query Wenlan for focused context. |
| `lint` | Run read-only diagnostics, or resolve all findings into ready, review, system-action, or blocked items. |
| `distill` | Refresh wiki pages from accumulated memories. |
| `pages` | Browse + open distilled pages by delegating to the `wenlan pages` CLI; query to open by title. |
| `curate` | Power-user deep audit of pending surfaces (captures, revisions). Daily flow handled by `/brief`. |
| `forget` | Delete a memory by ID. |
| `handoff` | End-session capture for decisions, lessons, gotchas, and open threads. |

Plugin metadata lives in [`.claude-plugin`](../.claude-plugin/README.md).

## Choosing the active space

Every space-aware skill resolves the active memory bucket through the
ordered chain below. Higher layers override lower ones:

| Layer | Mechanism | Example |
|---|---|---|
| 1 | `WENLAN_SPACE` strict process pin | `WENLAN_SPACE=career claude` |
| 2 | `space:X` inline arg | `/capture space:health "slept 5hrs"` |
| 3 | `WENLAN_DEFAULT_SPACE` overridable process context | `WENLAN_DEFAULT_SPACE=career claude` |
| 4 | `~/.wenlan/spaces.toml` cwd-prefix mapping (longest prefix wins; ties go to first-defined) | see `plugin/examples/spaces.toml` |
| 5 | registered cwd git-repo basename | `~/Repos/wenlan/...` → `wenlan`, only if that Space exists |
| 6 | none | omit client context; daemon Default decides new writes |

To pin a session to a specific bucket regardless of cwd, set
`WENLAN_SPACE` before invoking Claude Code. To pin by working directory
declaratively, copy `plugin/examples/spaces.toml` to
`~/.wenlan/spaces.toml` and edit. To override per call, prefix any
space-aware skill arg with `space:<name>`.

On the first space-aware skill call of a session, the skill prints one
line so the user can confirm the active bucket:

    Resolved space: <name> (from <layer>)

If the resolver reports no space, the skill omits the space parameter. The
daemon then applies its Default save space to new writes; reads remain All
Spaces. The tool's returned receipt is authoritative if it differs from the
client's proposed context.

## Links

- [wenlan.app](https://wenlan.app) — project home
- [wenlan.app/docs/commands](https://wenlan.app/docs/commands) — full Claude Code commands and MCP tools reference
- [wenlan.app/docs/daily-workflow](https://wenlan.app/docs/daily-workflow) — brief/capture/recall/handoff loop
- [wenlan.app/learn/claude-code-memory](https://wenlan.app/learn/claude-code-memory) — Claude Code memory concept article
