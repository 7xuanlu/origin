# Origin. Where Understanding Compounds.

> *"A large fraction of my recent token throughput is going less into manipulating code, and more into manipulating knowledge. The LLM is rediscovering knowledge from scratch on every question. There's no accumulation."* -- [Andrej Karpathy](https://x.com/karpathy/status/2039805659525644595), April 2026

Origin is a local-first companion for people who work with AI every day. It's where your AI thinking takes shape and keeps sharpening into something you can trust. Conversations across Claude, ChatGPT, Cursor, and other tools become connected, deduplicated, and editable. Any agent you grant access can read them through MCP, with distilled context that saves tokens every session.

![Origin demo](https://github.com/user-attachments/assets/d77806a4-69c2-4580-b95d-f8152323d122)

<details>
<summary>Watch full demo (90 seconds)</summary>

https://github.com/user-attachments/assets/6407e96a-9b84-4506-b702-c4a5f8da2920

</details>

**Status:** Early and exploratory. `v0.1.0` is a research preview running on macOS Apple Silicon. The shape of the product is still moving with usage. Expect changes, sharp edges, and fast iteration.

---

## Quickstart

**Platform:** macOS Apple Silicon (M1+) at `v0.1.0`. Linux, Intel Mac, and Windows are not supported yet.

### Use with Claude Code, Cursor, or Claude Desktop

Add to your MCP config:

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

On first run, `npx origin-mcp` downloads `origin-mcp` and `origin-server` into `~/.origin/bin/` and starts the daemon. Source: `[packages/origin-mcp-npm](packages/origin-mcp-npm)` in this repo bootstraps the companion [origin-mcp](https://github.com/7xuanlu/origin-mcp) binary.

### Desktop app

1. Download the `.dmg` from [GitHub Releases](https://github.com/7xuanlu/origin/releases/latest).
2. Drag **Origin** into Applications.
3. Clear quarantine (the build is currently unsigned):
   ```bash
   sudo xattr -cr /Applications/Origin.app
   ```
4. Launch. The daemon starts on `127.0.0.1:7878`.

### Headless daemon only

```bash
curl -fsSL https://raw.githubusercontent.com/7xuanlu/origin/main/install.sh | bash
export PATH="$HOME/.origin/bin:$PATH"
origin-server install
origin-server status
```

---

## Why Origin?

Native memory in Claude, ChatGPT, and Cursor works. It's a good start. Origin makes it much better in three ways:

**Trust.** Native memory stores what the AI decided was important. You can't inspect the full picture, correct what's wrong, or trace where a fact came from. Origin gives you an editable, searchable layer where every memory is visible, correctable, and yours.

**Token efficiency.** Every conversation starts from scratch. Agents re-explain context you've already established, burning tokens on repetition. Origin distills your history into compact, structured concepts that agents retrieve on demand. A 200-token concept summary replaces thousands of tokens of re-sent conversation. The longer you use Origin, the more you save per session.

**Understanding compounds.** Origin doesn't just store, it refines. A background engine deduplicates, links related memories into concepts, detects contradictions, and distills patterns. The quality of what your agents know improves every day without you doing anything. What you figured out in March is sharper in April. What you told Claude, Cursor can see too.

---

## What you actually get

Origin keeps many things from your work with AI: memories, concepts, decisions, observations, gotchas, learnings, and more. Each one is editable, inspectable, and traceable back to the conversation it came from.

- **Self-refining.** A background engine dedups overlapping captures, links related items into concepts, and surfaces contradictions for your attention. The longer you use Origin, the cleaner and more interconnected everything becomes.
- **Import and go.** Drop in your ChatGPT export or Obsidian vault and Origin starts refining immediately. No waiting period before it's useful.
- **MCP-native.** Add `origin-mcp` to Claude Code, Claude Desktop, Cursor, ChatGPT (App Directory), or Gemini CLI. The agent then recalls, stores, and searches across your shared context.
- **Local-first.** Everything stays on your machine. Nothing leaves by default.
- **Curated by you.** The desktop app gives you a timeline, concept browser, knowledge graph, and edit-anything controls. The daemon doesn't need it; most days it just runs.
- **Markdown export.** Concepts and decisions export to Obsidian or any vault as plain markdown.

---

## Evaluation

Retrieval quality on standard long-memory benchmarks. Numbers come from BGE-Base-EN-v1.5-Q embeddings combined with FTS5 and Reciprocal Rank Fusion. Harness at `crates/origin-core/src/eval/`.


| Benchmark                   | Recall@5 | MRR   | NDCG@10 |
| --------------------------- | -------- | ----- | ------- |
| LongMemEval (oracle, 500 Q) | 88.0%    | 74.2% | 79.0%   |
| LoCoMo (locomo10)           | 67.3%    | 58.9% | 64.0%   |


The optional LLM reranker (`search_memory_reranked`) is wired in but does not currently lift these benchmarks; reranker prompt and configuration are an active research area. LoCoMo-Plus (semantic-disconnect variant) deferred for v0.1.1.

---

## Architecture

The daemon owns all data and business logic. The Tauri app and MCP clients are thin HTTP clients that come and go; the daemon runs continuously.


| Crate                  | Role                                                      | License       |
| ---------------------- | --------------------------------------------------------- | ------------- |
| `origin-types`         | Shared request/response types                             | Apache-2.0    |
| `origin-core`          | Business logic: db, embeddings, refinery, knowledge graph | Apache-2.0    |
| `origin-server`        | Axum HTTP daemon on `127.0.0.1:7878`                      | Apache-2.0    |
| `app/` + root frontend | Tauri desktop client, frontend UI, macOS sensors          | AGPL-3.0-only |


**Stack:** Rust · Tauri 2 · libSQL · Tokio · FastEmbed (BGE-Base-EN-v1.5-Q, 768-dim) · llama-cpp-2 (Qwen3-4B via Metal GPU) · Axum 0.8 · React 19 · Tailwind CSS 4

Full module-by-module map is in [CLAUDE.md](CLAUDE.md).

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

## Where it's going

These are directions informed by usage, not a committed roadmap. Direction shifts when the data does.

- `**MEMORY.md` cooperation**: read from and write into Claude Code's per-project `MEMORY.md` so the two systems stay aligned rather than duplicating.
- **Skills grounded in what Origin holds**: workflows that operate against what you've worked through, not generic prompts.
- **Working context**: narrowing AI retrieval to the project you're currently in, not your whole history.
- **Spaces**: auto-scoped per repository or project so work-machine context and hobby code don't bleed into each other.
- **Team layer**: much later. Shared trust for groups of two or more.

---

## What it isn't

- Not a notes app or a Notion / Obsidian replacement.
- Not a chat UI. The conversation stays in Claude / ChatGPT / Cursor.
- Not a graph visualization tool. The graph is a means, not the product.
- Not a memory infrastructure SDK. Origin is meant for people, not as a backend for other apps.
- Not a browser extension.
- Not Windows or Linux at `v0.1.0`.

If you want to assemble a different architecture, [PAI](https://github.com/danielmiessler/PAI), [claude-memory-compiler](https://github.com/coleam00/claude-memory-compiler), and Palinode are good starting points and are explicitly building in this space.

---

## Honest caveats

- **Pays off over time.** Origin gets better the more you use it. If most of what you do is one-off chats, platform memory may be enough. The value shows when you carry work across sessions, tools, and weeks.
- **Quality of compile reflects quality of input.** Origin can structure what you bring to it; it can't invent depth that isn't there.
- **Opinionated by design.** Origin makes specific choices about what to keep and how to organize it. The Apache-licensed crates make alternative shells possible, but the desktop app expects this shape.

---

## Contributing

Bug fixes, eval cases, docs, and features are welcome. Start with [CONTRIBUTING.md](CONTRIBUTING.md). Architecture is in [CLAUDE.md](CLAUDE.md).


| [Bug reports](https://github.com/7xuanlu/origin/issues/new/choose) | [Feature requests](https://github.com/7xuanlu/origin/issues/new?template=feature_request.yml) | [Good first issues](https://github.com/7xuanlu/origin/labels/good%20first%20issue) |
| ----------------------------------------------------------- | ----------------------------------------------------------------- | ---------------------------------------------------------------------------------- |


---

## License

- `origin-types`, `origin-core`, `origin-server`: **Apache-2.0**
- Desktop app (`app/`) and root frontend UI: **AGPL-3.0-only**
- npm wrapper (`packages/origin-mcp-npm`): **MIT**
- Companion MCP server binary ([origin-mcp](https://github.com/7xuanlu/origin-mcp)): **MIT**

The split keeps the data layer permissively licensed for downstream tools while the shipped desktop app stays AGPL.

---

## Acknowledgments

Adjacent work shaping this space:

- Andrej Karpathy's note on the LLM-wiki pattern, the prompt that defined this category.
- Claude Code's `MEMORY.md`, the simplest version of the idea, and the one Origin aims to cooperate with.
- [PAI](https://github.com/danielmiessler/PAI), [claude-memory-compiler](https://github.com/coleam00/claude-memory-compiler), Palinode: different shapes of the same direction.

