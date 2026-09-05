<p align="center">
  <picture>
    <source media="(max-width: 600px)" srcset="./docs/assets/readme-banner-mobile.png">
    <img src="./docs/assets/readme-banner.png" alt="Wenlan: your source-backed knowledge base, built to compound." width="100%">
  </picture>
</p>

Useful work with AI shouldn't disappear when a conversation ends. Wenlan builds the right pages and keeps them current as sources change, asking only when judgment is needed.

<p align="center">
  English | <a href="./README.zh-Hans.md">简体中文</a> | <a href="./README.zh-Hant.md">繁體中文</a> | <a href="./README.es-ES.md">Español</a>
</p>

<p align="center">
  <a href="https://github.com/7xuanlu/wenlan/actions/workflows/ci.yml?query=branch%3Amain"><img alt="CI" src="https://github.com/7xuanlu/wenlan/actions/workflows/ci.yml/badge.svg?branch=main&event=push"></a>
  <a href="https://github.com/7xuanlu/wenlan/releases/latest"><img alt="Latest release" src="https://img.shields.io/github/v/release/7xuanlu/wenlan?sort=semver&label=release"></a>
  <a href="#license"><img alt="License: Apache-2.0 and AGPL-3.0" src="https://img.shields.io/badge/license-Apache--2.0%20%2B%20AGPL--3.0-blue.svg"></a>
</p>

<p align="center">
  <a href="#start-in-30-seconds">Get&nbsp;started</a> ·
  <a href="#what-does-wenlan-build">What&nbsp;is&nbsp;this?</a> ·
  <a href="#what-can-it-do">Capabilities</a> ·
  <a href="#how-does-it-work">Daily&nbsp;workflow</a> ·
  <a href="#evaluation">Evaluation</a> ·
  <a href="#learn-more">Learn&nbsp;more</a>
</p>

https://github.com/user-attachments/assets/d8b2ad4a-f97a-4a15-97a8-9105478de18a

<p align="center">
  <sub>A maintained Page in the desktop app: open any citation to inspect the Source or Memory behind the claim.</sub>
</p>

---

<a id="quickstart"></a>
<a id="start-in-30-seconds"></a>

## Get started

Wenlan runs as one local daemon. The desktop app carries that daemon inside it; the headless install gives you the same daemon without a window. Your AI clients reach the same knowledge base either way.

<a id="start-with-the-app"></a>
<a id="open-the-wiki"></a>
<a id="desktop-app"></a>

### Desktop app

Download from the [Releases page](https://github.com/7xuanlu/wenlan/releases/latest):

- **macOS (Apple Silicon):** open the `.dmg` and drag Wenlan to Applications. The app is signed and notarized, so there is no warning on first launch. From the terminal instead: `/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/7xuanlu/wenlan/main/scripts/install-macos-app.sh)"` (downloads, checks the SHA-256, moves it to Applications).
- **Windows x64:** run the `-setup.exe`. It is not signed yet, so when SmartScreen says "Windows protected your PC", choose "More info", then "Run anyway".
- **Linux:** no desktop build yet; use the headless runtime below.

The app bundles the daemon, CLI, and MCP connector, starts the daemon on launch, and offers to connect the AI clients it detects: the plugin for Claude Code and Codex, an MCP entry for the rest. To upgrade, drag the new app over the old one and open it (Wenlan 0.17.0 and older must be quit by hand first).

<a id="claude-code-in-30-seconds"></a>

<a id="codex-plugin"></a>

<a id="mcp-setup"></a>
<a id="mcp-clients"></a>

### Set up with your AI

Paste this into Claude Code, Codex, or another tool that can follow a setup guide:

```text
Set up Wenlan for this AI client by following:
https://raw.githubusercontent.com/7xuanlu/wenlan/main/docs/setup-with-ai.md

Install only what this client needs. Then verify the local runtime,
its Wenlan connection, and a capture/recall round trip.
```

The guide detects which client you are using and keeps client-specific commands out of this README. It does not configure every AI tool unless you ask it to.

Need only the headless runtime on macOS Apple Silicon?

```bash
npx -y wenlan setup
```

`npx` requires Node.js; without it, run `curl -fsSL https://raw.githubusercontent.com/7xuanlu/wenlan/main/install.sh | bash` then `wenlan setup --basic`.

This downloads the prebuilt CLI, daemon, and MCP connector, starts the local runtime, and verifies it. No Rust toolchain or Cargo is required. Linux x64/ARM64 with glibc has an automated [shell setup path](docs/setup-with-ai.md#install-the-runtime); Windows x64 uses the matching archive from [Releases](https://github.com/7xuanlu/wenlan/releases/latest). macOS Intel currently has [no supported complete-runtime install](crates/wenlan-cli/README.md#macos-intel).

Manual and client-specific instructions: [AI-assisted setup](docs/setup-with-ai.md) · [Claude Code plugin](plugin/.claude-plugin/README.md) · [Codex plugin](plugin-codex/README.md) · [CLI and MCP](crates/wenlan-cli/README.md).

---

<a id="what-does-wenlan-build"></a>
<a id="why-it-compounds"></a>

## What is this?

Wenlan turns documents, notes, and past AI conversations into a source-backed knowledge base that stays current as your work evolves. Sources remain traceable; decisions, lessons, and corrections become durable memories; both can support the same maintained Pages.

<p align="center">
  <picture>
    <source media="(max-width: 600px)" srcset="./docs/assets/wenlan-system-mobile.png">
    <img src="./docs/assets/wenlan-system.png" alt="Sources and memories independently support a maintained Page. Wenlan can rebuild a stale Page from its current support; optional conflict review can surface protected conflicts, and changes to human writing wait for the user." width="100%">
  </picture>
</p>

<a id="what-wenlan-is-not"></a>

**Built for work that continues.** Wenlan is for researchers, writers, consultants, product teams, and software teams whose knowledge is scattered across documents, notes, and AI conversations. It turns that material into inspectable Pages that can improve across projects and weeks, not another chat history or isolated memory store. It is not a life-management system or a memory SDK embedded inside another product.

**One knowledge system, three roles:**

- **Sources keep the material Wenlan reads traceable.** Imported conversations remain as captured records; registered files sync their current contents as they change.
- **Memories preserve what work teaches you.** Agents capture atomic decisions, lessons, corrections, and supersession with provenance.
- **Pages compile current knowledge.** Wenlan turns relevant Sources and Memories into source-cited Markdown you can reuse, refresh, and review.

**The LLM-wiki foundation, extended:**

- **[LLM-wiki v1](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f):** Karpathy defined immutable Sources, an AI-maintained Markdown Wiki, and a co-evolving Schema of rules for structuring and maintaining it. Wenlan implements that foundation with [typed Memory fields](docs/technical-foundations.md#typed-memory-schema) and built-in rules for Page structure, provenance, citations, refresh, ownership, and review.
- **[LLM-wiki v2](https://gist.github.com/rohitg00/2067ab416f7bbe447c1977edaaa681e2):** Rohitg00 added a memory lifecycle. Wenlan makes that direction concrete with traceable Sources, agent-captured Zettelkasten-style atomic Memories (one complete idea each), and maintained Pages built from both.

For the complete workflow, see the [LLM-wiki implementation guide](https://wenlan.app/learn/distilled-wiki-pages-ai-memory).

**Wenlan's distinctive move:** Sources and atomic Memories independently support maintained Pages. Memory history preserves how knowledge changed; Page history shows which current evidence supports the synthesis. Machine-maintained Pages can rebuild from current support, while changes to human writing wait as reviewable revisions.

<a id="knowledge-graph"></a>

### A knowledge graph that gets more useful over time

The entity-relation graph is one part of Wenlan's wider connected wiki. **Knowledge Pages** hold maintained synthesis, **Entities** anchor reusable people, projects, and concepts, **Source Pages** make imported or synchronized material inspectable, and atomic **Memories** preserve decisions and changes. They work through separate, explicit links: Page-to-Page wikilinks, Page evidence, Memory-to-Entity links, and directed Entity relations.

<p align="center">
  <picture>
    <source media="(max-width: 600px)" srcset="./docs/assets/wenlan-knowledge-network-mobile.png">
    <img src="./docs/assets/wenlan-knowledge-network.png" alt="Conceptual model of Wenlan's connected knowledge system, with Knowledge Pages, Source Pages, atomic Memories, and Entities connected through Page links, evidence, Memory-to-Entity links, and Entity relations." width="100%">
  </picture>
</p>

Within the entity graph, a configured enrichment model extracts typed Entities, observations, and directed relations from Memories. Entity linking and resolution reuse existing nodes instead of treating every mention as new; each Memory keeps its Source and can link to multiple Entities. [How the connected model is stored ->](docs/technical-foundations.md#connected-knowledge-model)

- **Meaning and direction:** Relations use a seeded vocabulary such as `uses`, `part_of`, `contradicts`, and `replaced_by`; unknown types fall back to `related_to` and become reviewable vocabulary proposals.
- **Strength and provenance:** A relation can store confidence, an explanation, and its source Memory, so stronger and weaker claims remain distinguishable and inspectable.
- **Communities that compound:** Label propagation groups Entities by relation density, weighted by the relation count between each pair. These groups can organize optional corpus summaries while Entity links add retrieval context.
- **Correction without erasure:** Related claims, corrections, and explicit supersession stay inspectable together while original Sources and Memory history remain.

During retrieval, dense entity matching finds query-relevant entities. When eligible graph links exist, the default graph-memory stream boosts linked Memories as a third [RRF](https://cormack.uwaterloo.ca/cormacksigir09-rrf.pdf) signal. The path is data- and scope-dependent, and Space boundaries (Spaces are defined under Capabilities) still apply. [How the graph path works ->](docs/technical-foundations.md#graph-assisted-retrieval)

<a id="retrieval"></a>

### Retrieval across words, meaning, and connections

Wenlan's core search is a local hybrid pipeline, not a single vector lookup. Each stage has a different job:

- **Exact wording, [SQLite FTS5](https://www.sqlite.org/fts5.html):** a full-text index finds literal terms, identifiers, and phrases.
- **Similar meaning, FastEmbed + [`Qdrant/bge-base-en-v1.5-onnx-Q`](https://huggingface.co/Qdrant/bge-base-en-v1.5-onnx-Q):** a quantized English model creates 768-dimensional embeddings; [libSQL cosine DiskANN](https://turso.tech/blog/approximate-nearest-neighbor-search-with-diskann-in-libsql) indexes them for approximate nearest-neighbor retrieval.
- **Combined ranking, weighted [RRF](https://cormack.uwaterloo.ca/cormacksigir09-rrf.pdf) (`k = 60`):** lexical and semantic rank lists are fused without pretending their raw scores share a scale; cosine similarity also weights the vector contribution.
- **Connected context, graph-memory stream:** eligible entity links add a third RRF signal while the active read scope still filters returned Memories.
- **Optional precision, cross-encoder reranking:** unlike embeddings, [`jinaai/jina-reranker-v1-turbo-en`](https://huggingface.co/jinaai/jina-reranker-v1-turbo-en) or [`BAAI/bge-reranker-base`](https://huggingface.co/BAAI/bge-reranker-base) reads each query-candidate pair and reorders the smaller pool; reranking is off by default.

Page, episodic, and fact channels are opt-in and degrade to the remaining search signals if unavailable. Space still limits the read scope. [Methods, defaults, and limitations ->](docs/technical-foundations.md)

<a id="what-makes-wenlan-distinct"></a>
<a id="why-is-wenlan-different"></a>
<a id="two-lifecycles"></a>

### Two lifecycles, one maintained knowledge system

A generated wiki can go stale; a memory store can fragment into disconnected facts. Wenlan links two lifecycles without collapsing them into one layer.

<p align="center">
  <picture>
    <source media="(max-width: 600px)" srcset="./docs/assets/wenlan-lifecycle-mobile.png">
    <img src="./docs/assets/wenlan-lifecycle.png" alt="An earlier memory remains linked after an explicit superseding capture. When a Page is stale, Wenlan rebuilds it from current Sources and Memories, records the revision, and stages changes to human writing for review." width="100%">
  </picture>
</p>

#### Atomic Memory

`CAPTURE -> CLASSIFY -> ENRICH -> LINK -> RECONCILE`

Capture and explicit supersession are core. Model-backed stages run only when the matching model is configured, and the reconcile pass is off by default.

| Operation | What Wenlan does |
|---|---|
| **Capture** | Agents write one complete, self-contained idea per Memory, following the Zettelkasten atomic-note principle instead of saving the whole conversation. |
| **Classify** | With the on-device model, Wenlan assigns `identity`, `preference`, `decision`, `lesson`, `gotcha`, or `fact`; a precise type supplied by the caller remains authoritative. |
| **Enrich** | With the on-device model, adds structured fields, retrieval cues, event dates, quality, importance, and tags when available. |
| **Link** | Retains provenance and, when enrichment is enabled, connects Memories to entities and relations in the knowledge graph. |
| **Reconcile** | Explicit replacements preserve a `supersedes` chain. A replacement from an agent whose trust level is below full queues for human review automatically, no flag required. An optional on-device pass can also queue protected conflicts for review instead of overwriting history; that pass is off by default and must be explicitly enabled. |

Advanced configuration: set `WENLAN_ENABLE_DUAL_POOL_RESOLVE=1` to enable that reconcile pass.

#### Maintained Page

`DISTILL -> CITE -> TRACK -> REFRESH -> REVIEW`

| Operation | What Wenlan does |
|---|---|
| **Distill** | Compiles related Sources and Memories into one Markdown Page. |
| **Cite** | Retains citation records and verification status; automatic refresh discards a draft when its citation-support check fails. |
| **Track** | Records which evidence supports the Page, why it became stale, and a bounded changelog. |
| **Refresh** | When a Page is marked stale, rebuilds the eligible machine-maintained Page from current evidence. |
| **Review** | Turns changes to a Page you edited into a proposed revision instead of a silent rewrite. |

For example, import a design document and capture a debugging decision in Codex. Wenlan can compile one Page that cites both. When that Page is refreshed, it rebuilds from its current support; if you have edited it, the proposed change waits for review.

<a id="local-markdown"></a>

### Local Markdown that works with Obsidian

Your durable synthesis remains ordinary files rather than a proprietary editor format:

- **Plain files:** Pages and session notes stay as Markdown under `~/.wenlan/`.
- **Inspectable history:** Distill and handoff workflows can commit logical file batches to a local git repository.
- **Obsidian coexistence:** Wenlan reads an existing vault as a source. Symlink `~/.wenlan/pages/` into the vault or export a Page from the desktop app; your edits remain human-owned, and later machine refreshes become reviewable revisions.

The local history is directly inspectable:

```text
$ git -C ~/.wenlan log --oneline
a1b2c3d distill: 4 pages
9f8e7d6 session: embedding-work
```

---

<a id="what-you-get"></a>
<a id="what-can-it-do"></a>
<a id="what-can-i-bring-in"></a>

## Capabilities

- **Chat import:** Bring in ChatGPT or Claude export ZIPs; Wenlan automatically skips conversations already imported.
- **Document Sources:** Ingest one `.md`, `.txt`, or text-extractable `.pdf` file; recurse through a folder of them; or index Markdown from an Obsidian vault.
- **Incremental sync:** Regular file and folder Sources track changes in the background; Obsidian vaults stay read-only and resync on demand.
- **Atomic Memory:** MCP clients save one complete decision, lesson, correction, preference, or fact, with [provenance and supersession](https://wenlan.app/learn/ai-memory-provenance) recording where it came from and what it replaces.
- **[Typed enrichment](docs/technical-foundations.md#typed-memory-schema):** A configured model classifies each Memory, then adds the structured fields defined for its type, plus dates, tags, retrieval cues, and graph links.
- **[Source-backed Pages](https://wenlan.app/docs/source-backed-pages):** Distill related Sources and Memories into Markdown Pages with source references and `[[wikilinks]]`; the daemon can verify and record per-claim citations.
- **Citation-gated refresh:** Automatic refresh rejects citation-poor drafts; machine Pages update while human edits become reviewable revisions.
- **[Hybrid retrieval](docs/technical-foundations.md#retrieval-pipeline):** FTS5 finds exact words, local BGE embeddings find meaning, and RRF fuses their ranks; graph links can add context.
- **[Retrieval channels](docs/technical-foundations.md#optional-channels-and-defaults):** Optional Page, episodic, and per-fact channels widen recall; cross-encoder reranking can improve precision.
- **[Knowledge graph](docs/technical-foundations.md#graph-data-and-entity-resolution):** Typed entities, relations, and observations connect people, projects, claims, and supporting Memories.
- **[Human-in-the-loop review](https://wenlan.app/docs/review-and-trust):** Routine work stays automatic; protected conflicts, Page revisions, entity merges, and new vocabulary wait for judgment.
- **[Spaces](https://wenlan.app/docs/spaces):** Keep work, personal, client, and repository knowledge inside an explicit retrieval scope.
- **[Local daemon + MCP](https://wenlan.app/docs/architecture):** One lightweight Rust daemon remains the local source of truth. The desktop app and CLI call it directly; AI clients use small MCP connectors to reach the same knowledge.
- **Custom integrations:** The localhost HTTP API accepts prepared text, webpage content, and Memories from other capture workflows.
- **Background maintenance:** The daemon keeps working after the desktop app closes, running configured sync, enrichment, citation work, and eligible Page refresh.
- **[Model choice](docs/technical-foundations.md#model-roles):** Base retrieval stays local; enrichment and synthesis can use on-device Qwen, a local endpoint, or a configured cloud model.
- **[Inspectable ownership](https://wenlan.app/learn/markdown-local-index-ai-memory):** Memories and graph data stay in local libSQL; Markdown, citations, revisions, git history, and Obsidian exports remain inspectable.
- **Read-only health checks:** [`doctor`](https://wenlan.app/docs/diagnostics-and-issue-reports) verifies the runtime; [`lint`](plugin/skills/lint/SKILL.md) finds malformed citations, orphan links, broken embeddings, and search-index or graph integrity problems without rewriting knowledge.

---

<a id="how-wenlan-works"></a>
<a id="how-does-it-work"></a>

## Daily workflow

The system above becomes a small daily loop: start with relevant knowledge, capture what matters while you work, close with a handoff, and let Wenlan refine what should return next time. Each pass leaves the same knowledge base sharper instead of creating another disconnected history.

The loop has four steps:

1. **Capture and find knowledge while you work.** `/capture <thing>` saves a decision, lesson, gotcha, or fact with its source. `/recall <query>` retrieves only what is relevant instead of loading your whole history.
2. **Find current knowledge.** Open a relevant Page, search, or use `/recall <query>`; `/brief [topic]` reads the Brief — the Space's rolling project snapshot, first written by `/handoff` — and a topic appends separately labeled context from that same Space. Clients without plugin commands use the equivalent page, search, recall, and brief tools.
3. **Close the loop.** `/handoff` records what changed and applies typed item-level updates to the current Space Brief.
4. **Keep the wiki current.** `/distill` deliberately creates or refreshes pages. Between sessions, optional model-backed passes can enrich captures, connect related entities, and refresh eligible pages. `/lint` checks knowledge health; `/curate` brings proposed revisions and any conflict-review items created by the optional reconcile pass to you.

### Offline queue (outbox)

If the local daemon is unreachable, `wenlan capture` and `wenlan brief update` write their requests to a durable local outbox and exit successfully. When the daemon returns, it drains those writes through the normal HTTP routes; inspect the queue with `wenlan outbox status` or request an immediate replay with `wenlan outbox drain`. A write the daemon rejects outright (a 4xx, such as failing the content quality gate) moves to `outbox/failed/` with a receipt instead of retrying forever; a transport failure or server error (5xx) leaves it queued for the next drain, which runs automatically every 60 seconds.

### Models and privacy

- **Local base retrieval:** The [BGE embedding model](https://huggingface.co/Qdrant/bge-base-en-v1.5-onnx-Q) runs through FastEmbed on your machine for hybrid search and needs no API key.
- **Optional on-device synthesis:** Enrichment and Page synthesis can use user-selected [`Qwen3 4B`](https://huggingface.co/unsloth/Qwen3-4B-Instruct-2507-GGUF) or [`Qwen3.5 9B`](https://huggingface.co/unsloth/Qwen3.5-9B-GGUF) through [llama.cpp](https://github.com/ggml-org/llama.cpp). Wenlan does not download or activate a language model until you choose one.
- **Other providers:** An OpenAI-compatible local endpoint such as Ollama or LM Studio, or a configured cloud provider, can supply model-backed enrichment and synthesis instead.
- **Cloud disclosure:** If the model endpoint you select is remote, Wenlan sends that task's system and user prompts to it. Local retrieval and on-device synthesis stay on your machine.
- **No telemetry:** Wenlan sends no telemetry.

Full workflow reference: [plugin/skills](plugin/skills/README.md). Technical model roles: [technical foundations](docs/technical-foundations.md#model-roles).

### Your data and uninstall

Nothing is locked in. Pages and session notes are Markdown under `~/.wenlan/`; memories live in one libSQL database under the platform data directory (`~/Library/Application Support/wenlan/` on macOS, `~/.local/share/wenlan/` on Linux, `%LOCALAPPDATA%\wenlan\` on Windows). Copy those two folders to back up or move a Wenlan. An install upgraded from Origin still holds a full copy of its data in `~/.origin/` and in the sibling `origin` data folder (`~/Library/Application Support/origin/` on macOS, `~/.local/share/origin/` on Linux, `%LOCALAPPDATA%\origin\` on Windows); delete or copy those two as well.

To uninstall: the app's *Run Wenlan in background at login* toggle removes the launch registration — turn it off, quit, and delete `Wenlan.app` or run the Windows uninstaller, then delete the folders above. `wenlan background off` only stops the daemon and disables autostart; it does not remove the launch registration, so a CLI-only install should instead follow the daemon uninstall bullet in [PRIVACY.md](docs/PRIVACY.md). The paths Wenlan writes are listed there.

---

<a id="evaluation"></a>

## Evaluation

This is a retrieval-only snapshot, not a claim about end-to-end answer quality. Method, environment receipts, and the update workflow live in [docs/eval](docs/eval/README.md).

<!-- EVAL_SNAPSHOT_START -->
| Benchmark | Recall@5 | MRR | NDCG@10 |
|---|---:|---:|---:|
| LME_Oracle (500 Q) | 93.6% | 0.857 | 0.883 |
| LME_S (deep, 90 Q) | 87.7% | 0.815 | 0.822 |
<!-- EVAL_SNAPSHOT_END -->

---

<a id="learn-more"></a>

## Learn more

More detailed documentation, concepts, and comparisons:

### Docs

- [Get started](https://wenlan.app/docs/get-started): install and verify the first local loop.
- [Daily workflow](https://wenlan.app/docs/daily-workflow): brief, capture, recall, handoff, distill, lint, and curate.
- [MCP clients](https://wenlan.app/docs/mcp-clients): connect Claude Code, Codex, Cursor, Claude Desktop, and other clients.

### Workflow guides

- [Build a client project knowledge base for consulting](https://wenlan.app/learn/build-client-project-knowledge-base-for-consulting)
- [Build an investment research knowledge base](https://wenlan.app/learn/build-investment-research-knowledge-base)
- [Build a product research knowledge base before writing a PRD](https://wenlan.app/learn/build-product-research-knowledge-base-for-prd)
- [Build an SRE incident knowledge base](https://wenlan.app/learn/build-sre-incident-knowledge-base)
- [Build a business metric definition knowledge base](https://wenlan.app/learn/build-business-metric-definition-knowledge-base): turn approved KPI specifications into a source-backed data dictionary with formula text, grain, exclusions, owners, revisions, and review state.

### Concepts

- [Why a living wiki, not just AI memory](https://wenlan.app/learn/ai-work-memory): the problem and product model in depth.
- [MCP memory server](https://wenlan.app/learn/mcp-memory-server): how Wenlan exposes knowledge across AI tools.
- [Local-first AI memory](https://wenlan.app/learn/local-first-ai-memory): data, privacy, and control.
- [Markdown and local index](https://wenlan.app/learn/markdown-local-index-ai-memory): storage, retrieval, and ownership.
- [AI agent handoff loop](https://wenlan.app/learn/ai-agent-handoff-loop): carrying work cleanly into the next session.
- [Research knowledge base from papers](https://wenlan.app/learn/source-backed-research-knowledge-base): build an inspectable literature matrix and source-backed synthesis from papers you already have.

### Comparisons

- [Wenlan vs Basic Memory](https://wenlan.app/learn/wenlan-vs-basic-memory)
- [Wenlan vs claude-mem](https://wenlan.app/learn/wenlan-vs-claude-mem)
- [Wenlan vs Superlocal Memory](https://wenlan.app/learn/wenlan-vs-superlocal-memory)

---

## Contributing

Bug fixes, eval cases, docs, and features are welcome. Installing Wenlan does not require building from source. For local development, run these commands from this repository's root:

```bash
# daemon crates (default-members — the desktop app is not compiled)
cargo build
cargo test

# desktop app (Cargo target and root-level frontend tooling)
pnpm install
pnpm dev:all
pnpm build:all
```

`pnpm dev:all` is the supported development entry point for the desktop app. It keeps development ports, data, process ownership, app identity, MCP sockets, and Remote Access state separate from the installed production runtime; a debug app started without that isolation refuses to run. See this repository's [AGENTS.md](AGENTS.md) and [CONTRIBUTING.md](.github/CONTRIBUTING.md), plus the in-tree [app/AGENTS.md](app/AGENTS.md), for the complete development workflow. Security reports: [SECURITY.md](.github/SECURITY.md). Privacy policy: [PRIVACY.md](docs/PRIVACY.md). Please also read the [Code of Conduct](.github/CODE_OF_CONDUCT.md).

---

<a id="code-signing-policy"></a>

## Code signing policy

Free code signing provided by [SignPath.io](https://about.signpath.io), certificate by [SignPath Foundation](https://signpath.org).

- **Authors:** [@7xuanlu](https://github.com/7xuanlu), who may commit to this repository without a further review.
- **Reviewers:** [@7xuanlu](https://github.com/7xuanlu). Every change from someone who is not a committer arrives as a pull request and is reviewed before it merges.
- **Approvers:** [@7xuanlu](https://github.com/7xuanlu), who approves each signing request and so decides which release is signed.

Multi-factor authentication is required of every maintainer, on GitHub and on SignPath, and nobody is added to either without it. Releases are built only by the tagged release workflow in this repository, on GitHub-hosted runners, from the commit the tag points at.

**Privacy policy:** [PRIVACY.md](docs/PRIVACY.md) — what Wenlan stores, where it stores it, and each case we know of in which it reaches the network. How each platform is signed: [docs/code-signing.md](docs/code-signing.md).

The SignPath application is pending. Windows installers are not signed yet.

---

<a id="license"></a>

## License

Wenlan uses two licenses, one per part of the repository.

- **Apache-2.0** ([`LICENSE`](LICENSE)) covers the local runtime, CLI, MCP server, shared types, and the Claude Code and Codex plugin files. Build on these freely.
- **AGPL-3.0-only** ([`app/LICENSE`](app/LICENSE)) covers the desktop app: the `app/` crate and the React frontend it ships. If you run a modified version of the app as a network service, the AGPL asks you to offer that modified source to its users.

The split is deliberate. Apache-2.0 code may be used inside an AGPL-3.0 program, so the desktop app builds on the runtime without either license being violated.

---

<a id="acknowledgments"></a>

## Lineage and peers

Wenlan (文瀾) takes its name from 文瀾閣, an imperial library that held 四庫全書 as part of one of China's largest book collections.

Wenlan's llm-wiki v2 model is its own product direction, informed by the LLM-wiki and agent-memory lineages:

- [Karpathy's LLM-wiki note](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f) established the raw-source-to-maintained-wiki pattern.
- [Rohitg00's LLM Wiki v2 proposal](https://gist.github.com/rohitg00/2067ab416f7bbe447c1977edaaa681e2) extends that pattern with memory lifecycle, confidence, graph, and retrieval mechanisms. [agentmemory](https://github.com/rohitg00/agentmemory) is its concrete agent-memory implementation.
- [nashsu/llm_wiki](https://github.com/nashsu/llm_wiki) is a full desktop implementation of the document-centered LLM-wiki pattern.
- [basic-memory](https://github.com/basicmachines-co/basic-memory), [obsidian-mind](https://github.com/breferrari/obsidian-mind), [mcp-memory-service](https://pypi.org/project/mcp-memory-service/), [Memoria](https://github.com/matrixorigin/Memoria), and [OpenMemory](https://github.com/CaviraOSS/OpenMemory) explore adjacent local knowledge and agent-memory shapes.
