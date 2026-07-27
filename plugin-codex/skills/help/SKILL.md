---
name: help
description: >
  Show a one-screen Codex reference for the Wenlan plugin. Use when the user
  asks for Wenlan help, command list, invokes /help, or explicitly asks about
  import progress.
allowed-tools: ["mcp__wenlan__list_pending_imports"]
user-invocable: true
---

# /help

Print the Wenlan Codex command card. The default help path is read-only and
calls no tools.

```text
Wenlan for Codex

  /setup          set up or repair the local runtime and MCP bridge
  /brief [topic]  load session status, identity, preferences, and memories
  /capture <x>    save one durable memory
  /recall <q>     search local memory
  /lint [deep|repair] [scope]   diagnose, or resolve all findings safely
  /distill [t]    synthesize or refresh source-backed pages
  /pages [q]      list or open distilled pages in the OS editor
  /curate <s>     review pending captures or revisions (s = captures|revisions)
  /forget <id>    delete one memory by exact id after confirmation
  /handoff        close a session with captures, session log, and status
  /help           show this card

Import progress: ask explicitly; Wenlan checks `list_pending_imports` on demand.

Daily flow:

  1. /setup once after install, or when Wenlan looks broken
  2. /brief at session start
  3. /capture durable decisions, corrections, lessons, or preferences
  4. /recall when you need a specific memory
  5. /handoff before ending the session

Data lives under ~/.wenlan/:

  pages/              source-backed wiki pages
  sessions/           narrative session logs
  sessions/_status/   current per-project status
  bin/                installed wenlan and wenlan-mcp binaries

Open pages with /pages. Inspect history with:

  git -C ~/.wenlan log --oneline

Optional models and keys do not enable background inference by themselves:

  wenlan enrichment status        show Everyday + Synthesis as off/ready/paused
  wenlan enrichment configure --everyday <source> --synthesis <source>
                                  review the exact mapping, disclosure, and confirm
  wenlan enrichment disable       turn model-backed background work off
```

Only when the user explicitly asks whether an import/export is still running,
call `mcp__wenlan__list_pending_imports`. Never call it during ordinary
`/help`, `/brief`, setup, or session-start flows.

If the local runtime or MCP bridge is down, tell the user to run `/setup`.
