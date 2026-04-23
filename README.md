# Origin. Where Understanding Compounds.

> *"A large fraction of my recent token throughput is going less into manipulating code, and more into manipulating knowledge. The LLM is rediscovering knowledge from scratch on every question. There's no accumulation."* -- [Andrej Karpathy](https://x.com/karpathy/status/2039805659525644595), April 2026

Origin is a local-first memory app that captures knowledge, decisions, and insights from every AI conversation. Everything you've figured out across Claude, ChatGPT, Cursor, and other tools, compounding instead of disappearing. It distills what matters, makes every memory visible and editable, and gets sharper the longer you use it.

![Origin demo](https://github.com/user-attachments/assets/d77806a4-69c2-4580-b95d-f8152323d122)

<details>
<summary>Watch full demo (90 seconds)</summary>

https://github.com/user-attachments/assets/6407e96a-9b84-4506-b702-c4a5f8da2920

</details>

**Status:** Early and exploratory. The [current release](https://github.com/7xuanlu/origin/releases/latest) is a research preview running on macOS Apple Silicon. The shape of the product is still moving with usage. Expect changes, sharp edges, and fast iteration.

---

## Quickstart

**Platform:** macOS Apple Silicon (M1+). Linux, Intel Mac, and Windows are not supported yet.

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

On first run, `npx origin-mcp` downloads `origin-mcp` and `origin-server` into `~/.origin/bin/` and starts the daemon. The npm package is published from the [origin-mcp repo](https://github.com/7xuanlu/origin-mcp).

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

AI was supposed to reduce work. Instead: more to manage, more to keep up with. Every conversation starts from scratch. What you figured out yesterday doesn't exist today. And the memory that does exist? Stale facts, contradicted decisions, wrong inferences you can't tell from real ones.

**Distills what matters.** Less to track, less to worry about. Origin reduces the noise to what you actually need.

**Visible and yours.** Every memory editable, traceable to its source. You control what your AI knows.

**Gets sharper over time.** Links, deduplicates, detects contradictions. March's insight is sharper in April.

**96% fewer tokens per query.** Same cost as basic vector search, but 19% more relevant context. 168 tokens instead of 4,505 for full replay. Measured on LoCoMo (2,531 memories, 1,540 queries). Eval harness at `crates/origin-core/src/eval/`.

---

## What you actually get

Origin keeps memories, concepts, decisions, observations, gotchas, and learnings from your work with AI. Each one is editable, inspectable, and traceable back to the conversation it came from.

- **Self-evolving.** Deduplicates, links related memories into concepts, detects contradictions. Your understanding matures while you work.
- **Hybrid memory engine.** Vector search, full-text search, and knowledge graph unified in one local database with Reciprocal Rank Fusion.
- **Associative recall.** Ask about one thing, get related context you didn't search for. Entities and relations link your knowledge so retrieval goes beyond keyword matching.
- **On-device intelligence.** Qwen3-4B and Qwen3.5-9B run on Apple Silicon Metal GPU. Your data never leaves your machine for processing.
- **MCP-native.** Any MCP-compatible agent reads and writes your memory. Claude Code, Claude Desktop, Cursor, ChatGPT (App Directory), Gemini CLI.
- **Memory lineage.** Every memory traces back to the conversation it came from. Full provenance: see where it was learned, when it was refined, and why it's there.
- **Import and go.** Drop in your ChatGPT export or Obsidian vault and Origin starts refining immediately. No cold start.
- **Markdown export.** Concepts and decisions export to Obsidian or any vault as plain markdown.

---

## Evaluation

Retrieval quality on standard long-memory benchmarks. Numbers come from BGE-Base-EN-v1.5-Q embeddings combined with FTS5 and Reciprocal Rank Fusion. Harness at `crates/origin-core/src/eval/`.


| Benchmark                   | Recall@5 | MRR   | NDCG@10 |
| --------------------------- | -------- | ----- | ------- |
| LongMemEval (oracle, 500 Q) | 88.0%    | 74.2% | 79.0%   |
| LoCoMo (locomo10)           | 67.3%    | 58.9% | 64.0%   |


The optional LLM reranker (`search_memory_reranked`) is wired in but does not currently lift these benchmarks; reranker prompt and configuration are an active research area. LoCoMo-Plus (semantic-disconnect variant) deferred to a future release.

---

## Architecture

The daemon owns all data and business logic. The Tauri app and MCP clients are thin HTTP clients that come and go; the daemon runs continuously.


| Crate                  | Role                                                      | License       |
| ---------------------- | --------------------------------------------------------- | ------------- |
| `origin-types`         | Shared request/response types                             | Apache-2.0    |
| `origin-core`          | Business logic: db, embeddings, refinery, knowledge graph | Apache-2.0    |
| `origin-server`        | Axum HTTP daemon on `127.0.0.1:7878`                      | Apache-2.0    |
| `app/` + root frontend | Tauri desktop client, frontend UI, macOS sensors          | AGPL-3.0-only |


**Stack:** Rust · Tauri 2 · libSQL · Tokio · FastEmbed (BGE-Base-EN-v1.5-Q, 768-dim) · llama-cpp-2 (Qwen3-4B / Qwen3.5-9B via Metal GPU) · Axum 0.8 · React 19 · Tailwind CSS 4

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

- **`MEMORY.md` cooperation**: read from and write into Claude Code's per-project `MEMORY.md` so the two systems stay aligned rather than duplicating.
- **Skills grounded in what Origin holds**: workflows that operate against what you've worked through, not generic prompts.
- **Working context**: narrowing AI retrieval to the project you're currently in, not your whole history.
- **Spaces**: auto-scoped per repository or project so work-machine context and hobby code don't bleed into each other.
- **Team layer**: much later. Shared trust for groups of two or more.

---

## What it isn't

- Not another memory MCP. Origin is a product built on top of memory, not just a store.
- Not a notes app or a Notion / Obsidian replacement.
- Not a chat UI. The conversation stays in Claude / ChatGPT / Cursor.
- Not a graph visualization tool. The graph is a means, not the product.
- Not a memory infrastructure SDK. Origin is meant for people, not as a backend for other apps.
- Not Windows or Linux yet.

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
- [origin-mcp](https://github.com/7xuanlu/origin-mcp) (MCP server, npm package, Homebrew): **MIT**

The split keeps the data layer permissively licensed for downstream tools while the shipped desktop app stays AGPL.

---

## Acknowledgments

Adjacent work shaping this space:

- Andrej Karpathy's note on the LLM-wiki pattern, the prompt that defined this category.
- Claude Code's `MEMORY.md`, the simplest version of the idea, and the one Origin aims to cooperate with.
- [PAI](https://github.com/danielmiessler/PAI), [claude-memory-compiler](https://github.com/coleam00/claude-memory-compiler), Palinode: different shapes of the same direction.

