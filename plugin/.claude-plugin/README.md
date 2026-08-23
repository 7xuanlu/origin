# Wenlan

A living knowledge base your agents build as they work. What they learn becomes source-cited wiki pages that refresh between sessions, so each new thread starts from your latest work.

## 30-Second Setup

```text
0s   /plugin marketplace add 7xuanlu/wenlan
     /plugin install wenlan@7xuanlu-wenlan
5s   restart Claude Code
10s  /setup  auto-installs local runtime if missing, configures local memory,
            verifies runtime + MCP + round-trip, prints "Wenlan ready"
30s  /brief  (or /capture <something to remember>)
```

`/setup` is self-healing — if the local runtime isn't running and the `wenlan`
CLI isn't on PATH, it runs the install one-liner for you. No copy/paste,
no restart loop. The `SessionStart` hook only nudges you toward `/setup`
if the runtime ever stops.

## Install

```text
/plugin marketplace add 7xuanlu/wenlan
/plugin install wenlan@7xuanlu-wenlan
```

`7xuanlu-wenlan` is the marketplace name declared by this repository.

The marketplace is defined in [`../../.claude-plugin/marketplace.json`](../../.claude-plugin/marketplace.json) (at the repo root). The plugin metadata is defined in [`plugin.json`](plugin.json). MCP configuration is in [`../.mcp.json`](../.mcp.json) (this plugin's `.mcp.json`), which delegates to [`../bin/wenlan-mcp-runner.sh`](../bin/wenlan-mcp-runner.sh).

The runner picks the MCP server binary from four paths, in order:

1. **Filesystem override** — if `plugin/bin/wenlan-mcp.local` exists (typically a symlink to a locally-built binary, gitignored), the runner exec's it. Most reliable: survives plugin reloads that don't re-read env.
2. **Env var override** — `WENLAN_MCP_DEV_BIN=/abs/path/to/wenlan-mcp` (or the legacy `ORIGIN_MCP_DEV_BIN`). Convenient if you already export it; requires Claude Code to inherit the var at startup.
3. **Installed runtime** — `~/.wenlan/bin/wenlan-mcp`, when `/setup` has installed it.
4. **Default** — `npx -y wenlan-mcp@^<version>`, where the version is read from the sibling `plugin.json` at spawn time (`@latest` if it cannot be read). What end users get before local setup completes.

To set up the filesystem override during dev:

```
cargo build -p wenlan-mcp --release
ln -s $(pwd)/target/release/wenlan-mcp plugin/bin/wenlan-mcp.local
```

Reload the plugin (`/reload-plugins`) and the wrapper picks the local binary on the next MCP spawn.

## Daily Commands

```text
/setup      set up + verify Wenlan works (run once, or to diagnose)
/help       one-screen reference
/brief      load identity + topic context (start of session)
/capture    save one durable memory in flow
/recall     search local memory
/distill    synthesize pages from clusters (scoped to current repo)
/pages      browse + open distilled pages (wenlan pages)
/curate     audit pending captures or revisions
/forget     delete a memory by ID
/handoff    end-of-session debrief
```

A `SessionStart` hook (`hooks/check-daemon.sh`) probes the local runtime at `127.0.0.1:7878`. If down, it prints a single line: `local runtime not running. Handoff writes will queue in the outbox; run /wenlan:setup or \`wenlan background on\`.` The skill owns the install logic — the hook is just a nudge. Hook never blocks the session.

## Where your data lives

```text
~/.wenlan/pages/               wiki pages distilled from memories (md)
~/.wenlan/sessions/            session logs by date (md)
~/.wenlan/sessions/_status/    current per-project goals + last-handoff timestamp
~/.wenlan/db/                  symlink to the libSQL store
~/.wenlan/bin/                 installed binaries
```

Browse with `open ~/.wenlan/` (Finder), `code ~/.wenlan/` (VS Code), or symlink `~/.wenlan/pages/` into an Obsidian vault for the graph view. No Tauri app required.

## Local memory and agent-side model phases

By default `/setup` configures **local memory**: no model download, no API
key, no prompts. The daemon stores, embeds, dedupes, and serves hybrid
search. Model-backed work like classification, entity extraction, page
synthesis, and reranking stays opt-in.

The skills compensate. Where the daemon would normally call a model, the
skill asks Claude itself to do the equivalent step and posts the result
back via HTTP:

| Phase | Model-equipped daemon | Local memory + skill |
|---|---|---|
| Pick `memory_type` | daemon classifier | `/capture` picks one of 6 types from content |
| Extract entities/relations | daemon `extract.rs` | `/capture` POSTs to `/api/memory/entities` and `/api/memory/relations` |
| Synthesize a page | daemon distill cycles + `/distill` | `/distill` reads the cluster, writes the page, POSTs to `/api/pages` |
| Expand a query / rerank hits | daemon expansion + rerank | `/recall` rewrites the query before search and reorders hits after |

The daemon stays the single writer and storage owner. Claude does the
thinking. This is why local memory works without an API key or
on-device model — and why the same skills also work when you later add
one. The skills detect the daemon's capabilities and fall through to
the agent path when needed.

## Skill Files

The actual skill instructions live in [`../skills`](../skills):

- `setup`: end-to-end setup verifier (local runtime + MCP + round-trip)
- `help`: one-screen quick reference
- `brief`: load session context
- `capture`: save one durable memory
- `recall`: targeted lookup
- `distill`: refresh wiki pages
- `pages`: browse + open distilled pages (wenlan pages)
- `curate`: audit pending captures or revisions
- `forget`: delete a memory by ID
- `handoff`: capture end-of-session decisions, lessons, gotchas, and open threads

## Links

- [wenlan.app](https://wenlan.app) — project home
- [wenlan.app/learn/claude-code-memory](https://wenlan.app/learn/claude-code-memory) — Claude Code memory concept article
- [wenlan.app/docs/daily-workflow](https://wenlan.app/docs/daily-workflow) — the brief/capture/recall/handoff loop
- [wenlan.app/docs/get-started](https://wenlan.app/docs/get-started) — install + verify

## License

Apache-2.0.
