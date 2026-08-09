# wenlan-server

Local daemon for Wenlan. It owns the database, embeddings, search, distill cycles, page distillation, knowledge graph, and HTTP API on `127.0.0.1:7878`.

## Headless install

Most users install Wenlan through the [Claude Code plugin](../../plugin/.claude-plugin/README.md), which auto-runs the install script the first time `/setup` runs. This page is for daemon internals. On macOS Apple Silicon, terminal setup uses the product CLI package:

```bash
npx -y wenlan setup
```

The npm installer downloads `wenlan`, `wenlan-server`, and `wenlan-mcp` into `~/.wenlan/bin/`. Linux x64/ARM64 users can run the automated [shell setup](../../docs/setup-with-ai.md#install-the-runtime); Windows x64 users install the matching archive from [Releases](https://github.com/7xuanlu/wenlan/releases/latest). macOS Intel currently has no supported complete-runtime install. The `wenlan` CLI owns setup and service management; `wenlan-server` is the daemon binary the host's service manager runs (launchd on macOS, systemd-user on Linux, Task Scheduler ONLOGON task on Windows).

## Setup modes

```bash
wenlan setup                  # interactive (1=local memory, 2=on-device model, 3=Anthropic key)
wenlan setup --basic          # non-interactive local memory setup used by the plugin's /setup
wenlan models install         # opt into a local Qwen model (llama.cpp + Metal)
wenlan keys set anthropic     # opt into Anthropic-backed extraction (BYOK)
wenlan doctor                 # diagnose runtime, model, and key state
```

Local memory works without a local model or API key: store, search, recall, and MCP memory are available immediately. On-device models and Anthropic keys unlock distill cycles: auto entity extraction, page synthesis, recaps, and knowledge-graph rethink.

## Data layout

All user-facing data lives under `~/.wenlan/`:

```text
~/.wenlan/pages/               wiki pages distilled from your memories (md)
~/.wenlan/sessions/            session logs by date (md, written by /handoff)
~/.wenlan/sessions/_status/    current per-project goals + last-handoff timestamps
~/.wenlan/db/                  symlink to the libSQL store
~/.wenlan/bin/                 installed binaries
~/.wenlan/.git/                git repo — skills auto-commit per logical batch
```

The libSQL store lives under the platform data directory (`dirs::data_local_dir()/origin/memorydb/origin_memory.db`). On macOS that resolves to `~/Library/Application Support/wenlan/memorydb/`; on Linux `~/.local/share/wenlan/memorydb/`; on Windows `%LOCALAPPDATA%\origin\memorydb\`. `~/.wenlan/db` is a symlink on macOS + Linux so the user-facing tree stays single-rooted (Windows skips the symlink since it needs Developer Mode or admin and the alias is cosmetic). Use `open ~/.wenlan/`, `code ~/.wenlan/`, or symlink `~/.wenlan/pages/` into an Obsidian vault for the graph view.

## Service Commands

```bash
wenlan background on   # register with the host service manager (launchd / systemd-user / schtasks)
wenlan background off  # stop the current daemon while preserving its registration
wenlan status           # service + runtime status
```

`background off` is a reversible runtime stop, not an uninstall. Run `wenlan background on` to start the registered daemon again.

On Windows the install path uses `schtasks.exe` to register a per-user `WenlanServer` ONLOGON task and triggers it immediately. `background off` ends the running task without deleting that registration. wenlan-server stays a plain console app — no Windows Service Control Protocol dispatcher is needed.

The daemon listens on `127.0.0.1:7878`. MCP clients, the Claude Code plugin, and local tools call that local API.

## Build From Source

```bash
cargo build -p wenlan-server
cargo run -p wenlan-server
```

Or build a release binary and register it with the host service manager:

```bash
cargo build --release -p wenlan -p wenlan-server
./target/release/wenlan setup --basic
./target/release/wenlan background on
./target/release/wenlan status
```

First build on macOS takes several minutes while `llama.cpp` compiles for Metal. On Linux + Windows the build is CPU-only and finishes faster.

## Main HTTP Surfaces

- Health/status: `/api/health`, `/api/status`, `/api/setup/status`
- Memory ingest/search: `/api/memory/store`, `/api/memory/search`, `/api/context`
- Review and pages: `/api/memory/list`, `/api/memory/confirm/{id}`, `/api/distill`
- Model/key setup: `/api/on-device-model`, `/api/on-device-model/download`, `/api/setup/anthropic-key`

See [CLAUDE.md](../../CLAUDE.md) for the full route and module map.

## Links

- [wenlan.app](https://wenlan.app) — project home
- [wenlan.app/docs/get-started](https://wenlan.app/docs/get-started) — install + verify the first memory loop
- [wenlan.app/docs/mcp-clients](https://wenlan.app/docs/mcp-clients) — connect Claude Code, Cursor, Codex, Claude Desktop, Gemini CLI
- [github.com/7xuanlu/wenlan](https://github.com/7xuanlu/wenlan) — source

## License

Apache-2.0.
