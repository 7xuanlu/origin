![Origin social preview](docs/assets/social-preview.png)

[![CI](https://github.com/7xuanlu/origin/actions/workflows/ci.yml/badge.svg)](https://github.com/7xuanlu/origin/actions/workflows/ci.yml)
[![Release](https://img.shields.io/github/v/release/7xuanlu/origin)](https://github.com/7xuanlu/origin/releases/latest)
[![License](https://img.shields.io/badge/license-AGPL--3.0-blue)](LICENSE)
![Platform](https://img.shields.io/badge/platform-macOS%20Apple%20Silicon-black)

https://github.com/user-attachments/assets/6407e96a-9b84-4506-b702-c4a5f8da2920

[Origin](https://useorigin.app) runs quietly behind the AI tools you already use. It gives your AI a place to carry decisions, lessons, gotchas, and project context instead of rediscovering them in every new chat.

The daemon does the memory chores in the background: deduplicating, linking related ideas, compiling concepts, and keeping provenance attached.

Use optional desktop app to search, inspect, edit, and delete what Origin learned.

**Status:** Early preview for macOS Apple Silicon. Expect fast iteration and some sharp edges.

---

## Quickstart

**Platform:** macOS Apple Silicon (M1+). Linux, Intel Mac, and Windows are not supported yet.

### 1. Use with Claude Code, Cursor, Codex, or another MCP client

For Claude Code, Cursor, Codex, Claude Desktop, Windsurf, Gemini CLI, or another client that accepts a JSON `mcpServers` entry, add:

```json
{
  "mcpServers": {
    "origin": {
      "command": "npx",
      "args": ["-y", "origin-mcp"]
    }
  }
}
```

On first run, `npx origin-mcp` downloads the MCP binary from the [origin-mcp repo](https://github.com/7xuanlu/origin-mcp). It connects to the local Origin daemon on `127.0.0.1:7878`, started by the desktop app or the headless install below.

### 2. Add the desktop app

Use the desktop app when you want Origin in the menu bar, memory inspection/editing, and Remote Access for Claude.ai or ChatGPT web.

1. Download the `.dmg` from [GitHub Releases](https://github.com/7xuanlu/origin/releases/latest).
2. Drag **Origin** into Applications.
3. Clear quarantine (the build is currently unsigned):
   ```bash
   sudo xattr -cr /Applications/Origin.app
   ```
4. Launch Origin. It runs from the menu bar and starts the local daemon on `127.0.0.1:7878`.

If you already use the MCP config above, `origin-mcp` connects to the desktop app's daemon.

### 3. Claude.ai and ChatGPT web

Use **Remote Access** from the desktop app. Web clients do not use the local stdio `npx` config above.

### 4. Headless daemon only

Use this path for automation, servers, or no-GUI setups.

```bash
curl -fsSL https://raw.githubusercontent.com/7xuanlu/origin/main/install.sh | bash
export PATH="$HOME/.origin/bin:$PATH"
origin-server install
origin-server status
```

---

## Why Origin?

AI work has a continuity problem. Most memory tools ask you to trust a cloud API, maintain a notes/wiki workflow, or move into another app. Origin takes a different shape: a local daemon, a desktop inspection lens, and MCP access from the AI tools you already use.

**Your AI starts from scratch too often.** Origin carries decisions, preferences, gotchas, and project context across chats, projects, and time.

**Memory gets worse when nobody maintains it.** Origin runs a background refinery that deduplicates captures, links related ideas, compiles concepts, and keeps provenance attached.

**You need to see and correct what it learned.** The desktop app lets you search, inspect, edit, and delete memories instead of trusting a black box.

Developers already have `origin` for code. Origin gives AI work a source of truth too: local, traceable, and readable by agents through MCP.

For people whose work spans projects, clients, and jobs. Your context should not disappear when a chat ends or when you switch gears.

Origin keeps the useful parts together:

- **Capture:** decisions, lessons, observations, gotchas, and project context.
- **Refine:** deduplicate, link, and compile memories in the background.
- **Recall:** relevant context through MCP when your AI needs it.
- **Inspect:** every memory stays editable and traceable to where it came from.

**96% fewer tokens per query.** Same cost as basic vector search, but 19% more relevant context. 168 tokens instead of 4,505 for full replay. Measured on LoCoMo (2,531 memories, 1,540 queries). Eval harness at `crates/origin-core/src/eval/`.

---

## How Origin works

Use your AI tools normally. Origin runs in the background, keeps the useful parts, and makes them available when context is needed.

1. **Install Origin.** The desktop app lives in the macOS menu bar and keeps the local daemon running.
2. **Connect your AI tools.** Claude Code, Cursor, Codex, Claude Desktop, Windsurf, Gemini CLI, and other MCP clients connect through `origin-mcp`. Claude.ai and ChatGPT web use Remote Access.
3. **Capture useful context.** Agents save decisions, preferences, project facts, gotchas, and lessons while you work.
4. **Refine in the background.** Origin deduplicates captures, links related ideas, compiles concepts, and preserves where each memory came from.
5. **Recall when needed.** Retrieval combines vector search, full-text search, and knowledge graph signals without replaying full chat history.
6. **Inspect and export.** Use the desktop app to search, edit, delete, verify, and export what Origin learned.

---

## Evaluation

Retrieval quality on standard long-memory benchmarks. Numbers come from BGE-Base-EN-v1.5-Q embeddings combined with FTS5 and Reciprocal Rank Fusion. Harness at `crates/origin-core/src/eval/`; update workflow in [docs/eval](docs/eval/README.md).


| Benchmark                   | Recall@5 | MRR   | NDCG@10 |
| --------------------------- | -------- | ----- | ------- |
| LongMemEval (oracle, 500 Q) | 88.0%    | 74.2% | 79.0%   |
| LoCoMo (locomo10)           | 67.3%    | 58.9% | 64.0%   |


---

## Local by default

- Memories are stored locally at `~/Library/Application Support/origin/memorydb/origin_memory.db` by default.
- The daemon listens on `127.0.0.1:7878`; the desktop app and MCP clients call that local API.
- There is no cloud sync or telemetry by default. Remote access and external LLM endpoints are opt-in settings.
- On-device Qwen models download only when requested and use the `hf-hub` cache.
- Security reports: [SECURITY.md](SECURITY.md).

---

## Architecture

Origin is daemon-first. `origin-server` owns the local database, embeddings, refinery, knowledge graph, and HTTP API on `127.0.0.1:7878`. The desktop app and MCP clients are thin clients over that daemon.

Stack: Rust, Tauri 2, libSQL, Tokio, FastEmbed, llama-cpp-2, Axum, React, Tailwind CSS. Full module map: [CLAUDE.md](CLAUDE.md).

---

## Build from source

```bash
git clone https://github.com/7xuanlu/origin.git
cd origin
pnpm install
```

Single command builds the daemon, starts it, and launches the Tauri app with Vite:

```bash
pnpm dev:all
```

Or run daemon and app separately:

```bash
cargo run -p origin-server          # terminal 1
pnpm tauri dev                      # terminal 2
```

For local `.dmg` builds:

```bash
pnpm release            # builds daemon + app bundle
pnpm release:dmg                # wraps .app into DMG
```

First build takes several minutes while `llama.cpp` compiles for Metal.

---

## Boundaries

- Not a chat UI. Keep using Claude, ChatGPT, Cursor, or your agent of choice.
- Not a notes app or Notion / Obsidian replacement. Markdown export exists so you can read the artifact anywhere.
- Not a memory infrastructure SDK. Origin is meant for people using AI, not as a backend for other apps.
- macOS Apple Silicon only for now.
- Best for work that spans sessions, projects, and weeks. One-off chats may not need it.

---

## Contributing

Bug fixes, eval cases, docs, and features are welcome. Start with [CONTRIBUTING.md](CONTRIBUTING.md). Please also read the [Code of Conduct](CODE_OF_CONDUCT.md). Architecture is in [CLAUDE.md](CLAUDE.md).


| [Bug reports](https://github.com/7xuanlu/origin/issues/new/choose) | [Feature requests](https://github.com/7xuanlu/origin/issues/new?template=feature_request.yml) | [Good first issues](https://github.com/7xuanlu/origin/labels/good%20first%20issue) |
| ------------------------------------------------------------------ | --------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------- |


---

## License

- `origin-types`, `origin-core`, `origin-server`: **Apache-2.0**
- Desktop app (`app/`) and root frontend UI: **AGPL-3.0-only**
- [origin-mcp](https://github.com/7xuanlu/origin-mcp) (MCP server, npm package, Homebrew): **MIT**

The split keeps the data layer permissively licensed for downstream tools while the shipped desktop app stays AGPL.

---

## Acknowledgments

Adjacent work shaping this space:

- Andrej Karpathy's note on the LLM-wiki pattern, parallel work in this space.
- Claude Code's `MEMORY.md`, the simplest version of the idea, and the one Origin aims to cooperate with.
- [PAI](https://github.com/danielmiessler/PAI), [claude-memory-compiler](https://github.com/coleam00/claude-memory-compiler), Palinode: different shapes of the same direction.
