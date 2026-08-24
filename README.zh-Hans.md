<!-- README_SYNC: source=README.md sha256=7b7171a095587f65f249b9f82d918776815fd3a1325bdc1eb74ee83823d8fa53 -->

<p align="center">
  <picture>
    <source media="(max-width: 600px)" srcset="./docs/assets/readme-banner-mobile.png">
    <img src="./docs/assets/readme-banner.png" alt="Wenlan：有来源支撑、会持续积累的知识库。" width="100%">
  </picture>
</p>

和 AI 聊出的成果，不该在对话结束后消失。Wenlan 会建立真正需要的页面，并在来源变化时让它们保持最新；只有需要判断时才找你。

<p align="center">
  <a href="./README.md">English</a> | 简体中文 | <a href="./README.zh-Hant.md">繁體中文</a> | <a href="./README.es-ES.md">Español</a>
</p>

<p align="center">
  <a href="https://github.com/7xuanlu/wenlan/actions/workflows/ci.yml?query=branch%3Amain"><img alt="CI" src="https://github.com/7xuanlu/wenlan/actions/workflows/ci.yml/badge.svg?branch=main&event=push"></a>
  <a href="https://github.com/7xuanlu/wenlan/releases/latest"><img alt="最新版本" src="https://img.shields.io/github/v/release/7xuanlu/wenlan?sort=semver&label=release"></a>
  <a href="#license"><img alt="许可证：Apache-2.0" src="https://img.shields.io/badge/license-Apache--2.0-blue.svg"></a>
</p>

<p align="center">
  <a href="#start-in-30-seconds">开&#8288;始&#8288;使&#8288;用</a> ·
  <a href="#what-does-wenlan-build">这&#8288;是&#8288;什&#8288;么？</a> ·
  <a href="#what-can-it-do">能&#8288;力</a> ·
  <a href="#how-does-it-work">日&#8288;常&#8288;流&#8288;程</a> ·
  <a href="#evaluation">评&#8288;估</a> ·
  <a href="#learn-more">进&#8288;一&#8288;步&#8288;了&#8288;解</a>
</p>

<p align="center">
  <img src="./docs/assets/desktop-wiki-preview.png" alt="Wenlan 桌面 app，展示有来源支撑的 wiki 页面与可检查的引用。" width="100%">
</p>

<p align="center">
  <sub>桌面 app 中持续维护的页面：打开任意引用，就能检查这条结论背后的来源或记忆。</sub>
</p>

---

<a id="quickstart"></a>
<a id="start-in-30-seconds"></a>

## 开始使用

Wenlan 以单个本地 daemon 运行。桌面 app 内置这个 daemon；无 GUI 的安装方式提供的是同一个 daemon，只是没有窗口。两种方式下，你的 AI 客户端访问的都是同一个知识库。

<a id="start-with-the-app"></a>
<a id="open-the-wiki"></a>
<a id="desktop-app"></a>

### 桌面 app

**[下载 macOS 版 Wenlan](https://github.com/7xuanlu/wenlan/releases/latest)**（Apple Silicon），打开 `.dmg`，把 app 拖进「应用程序」。

不需要再安装别的东西。App 内已打包 daemon、CLI 与 MCP 连接器，启动时会自动运行 daemon，并会为检测到的 AI 客户端提供接入：Claude Code 与 Codex 安装 plugin，其余客户端写入 MCP 配置。之后你就可以阅读 Page、检查任一引用背后的 Source，并整理整个知识体系。

这个预览版尚未经过 Apple notarization，首次启动会被 macOS 拦下，在「系统设置」的「隐私与安全性」里点一次「仍要打开」即可。一条命令可以跳过这一步：它会用 GitHub 发布的 SHA-256 核对下载文件，只为这一个 app 清除 quarantine，不会更改任何 macOS 安全设置。

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/7xuanlu/wenlan/main/scripts/install-macos-app.sh)"
```

Windows x64 请在同一个 [Releases](https://github.com/7xuanlu/wenlan/releases/latest) 页面下载 `-setup.exe` 并运行。安装包内含 daemon、CLI 与 MCP 连接器，以及它们运行时要加载的库，无需再装别的东西。Linux 暂时没有桌面版，请使用下面的无 GUI runtime。

<a id="claude-code-in-30-seconds"></a>

<a id="codex-plugin"></a>

<a id="mcp-setup"></a>
<a id="mcp-clients"></a>

### 让你的 AI 完成设置

把下面这段贴给 Claude Code、Codex，或其他能够读取设置指南的工具：

```text
请为当前 AI 客户端设置 Wenlan，并严格遵循：
https://raw.githubusercontent.com/7xuanlu/wenlan/main/docs/setup-with-ai.md

只安装这个客户端需要的内容。完成后验证本地 runtime、
Wenlan connection，以及一次 capture/recall round trip。
```

指南会识别当前使用的 client，把各平台命令留在专门文档中。除非你明确要求，否则它不会设置所有 AI 工具。

只需要在 macOS Apple Silicon 上运行的无 GUI 本地服务？

```bash
npx -y wenlan setup
```

这个命令会下载预编译的 CLI、后台服务（daemon）与 MCP 连接器，启动并验证本地服务；不需要安装 Rust 或 Cargo。使用 glibc 的 Linux x64/ARM64 可以采用自动化的 [shell 设置流程](docs/setup-with-ai.md#install-the-runtime)；Windows x64 请从 [Releases](https://github.com/7xuanlu/wenlan/releases/latest) 下载对应的 archive。macOS Intel 目前[没有受支持的完整 runtime 安装方式](crates/wenlan-cli/README.md#macos-intel)。

手动与各 client 设置说明：[AI 辅助设置](docs/setup-with-ai.md) · [Claude Code plugin](plugin/.claude-plugin/README.md) · [Codex plugin](plugin-codex/README.md) · [CLI 与 MCP](crates/wenlan-cli/README.md)。

---

<a id="what-does-wenlan-build"></a>
<a id="why-it-compounds"></a>

## 这是什么？

Wenlan 把文档、笔记和过去的 AI 对话整理成会随工作持续更新、每个结论都能追溯来源的知识库。原始材料保留为来源；工作中的决策、经验与修正成为长期记忆；两者都能支撑同一批持续维护的页面。

<p align="center">
  <picture>
    <source media="(max-width: 600px)" srcset="./docs/assets/wenlan-system-zh-Hans-mobile.png">
    <img src="./docs/assets/wenlan-system-zh-Hans.png" alt="来源与记忆分别支撑同一个持续维护的页面。页面过时后，Wenlan 可以依当前依据重建；可选的冲突审核可以让受保护内容的冲突浮现，对人工文字的改动则等待用户判断。" width="100%">
  </picture>
</p>

<a id="what-wenlan-is-not"></a>

**适合需要长期延续的工作。** Wenlan 面向研究者、写作者、顾问、产品团队与软件团队：当知识散落在文档、笔记和 AI 对话里，它会把这些材料变成可检查、能随项目持续改进的页面，而不是另一份聊天记录或孤立的记忆库。它不是生活管理系统，也不是嵌入其他产品的 memory SDK。

**一个知识系统，三种角色：**

- **来源让 Wenlan 读到的材料始终可追溯。** 导入的对话保留为捕获时的记录；已注册文件会随内容变化同步当前版本。
- **记忆保留工作真正教会你的内容。** AI agent 捕获原子的决策、经验、修正与取代关系，并保留出处。
- **页面汇总当前知识。** Wenlan 把相关来源与记忆整理成带引用的 Markdown，让你反复使用、刷新与审核。

**在 LLM-wiki 的基础上继续推进：**

- **[LLM-wiki v1](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f)：** Karpathy 提出不可变的来源、由 AI 维护的 Markdown Wiki，以及会随你和 AI 一起演进、规定组织与维护方式的 Schema（规则层）。Wenlan 以[类型化记忆字段](docs/technical-foundations.md#typed-memory-schema)与内建规则，落实页面结构、出处、引用、刷新、归属和审核。
- **[LLM-wiki v2](https://gist.github.com/rohitg00/2067ab416f7bbe447c1977edaaa681e2)：** Rohitg00 加入记忆生命周期。Wenlan 把这个方向做成可以直接使用的产品：可追溯的来源、由 AI agent 按 Zettelkasten（卡片盒笔记法）捕获的原子记忆（每条只表达一个完整想法），以及同时由两者建立并持续维护的页面。

**Wenlan 最独特的做法：** 来源与原子记忆会分别支撑持续维护的页面。记忆历史保留知识如何改变；页面历史说明当前结论由哪些依据支撑。机器维护的页面可以依当前依据重建；对人工文字的改动则成为可审核的修订，不会直接覆盖。

<p align="center">
  <img src="./docs/assets/feature-reel.gif" alt="Wenlan feature reel，展示有来源支撑的页面、来源检查、graph context、agent capture 与 curation。" width="100%">
</p>

<a id="knowledge-graph"></a>

### 越用越有价值的知识图谱

实体关系图谱只是 Wenlan 更大连接式 wiki 的一部分。**知识页面**保留持续维护的结论，**实体**固定可复用的人物、项目与概念，**来源页面**让导入或同步的材料可检查，原子**记忆**则保留决策与变化。它们通过彼此分开的明确连接协作：页面间的 wikilink、页面依据、记忆到实体的连接，以及实体间的有向关系。

<p align="center">
  <picture>
    <source media="(max-width: 600px)" srcset="./docs/assets/wenlan-knowledge-network-zh-Hans-mobile.png">
    <img src="./docs/assets/wenlan-knowledge-network-zh-Hans.png" alt="Wenlan 连接式知识系统的概念图：知识页面、来源页面、原子记忆与实体通过页面链接、依据、记忆到实体的连接和实体关系互相连接。" width="100%">
  </picture>
</p>

在实体图谱这一层，配置 enrichment 模型后，Wenlan 会从记忆中提取带类型的实体、观察与有方向的关系。实体链接与解析会复用已有节点，而不是把每次提及都当成新事物；每条记忆仍保留来源，并可连接多个实体。[查看连接模型如何存储 ->](docs/technical-foundations.md#connected-knowledge-model)

- **含义与方向：** 关系使用 `uses`、`part_of`、`contradicts`、`replaced_by` 等预置词汇；未知类型会回退为 `related_to`，并成为可审核的词汇提案。
- **强度与出处：** 关系可以保存置信度、解释与对应的来源记忆，让强弱不同的主张仍可区分、可检查。
- **形成可复用群组：** 标签传播会依关系密度为实体分组，并按每对实体之间的关系数量加权。这些群组可组织可选的全局摘要，实体链接也会为检索补充脉络。
- **修正但不抹除：** 相关说法、修正与明确的取代关系可以放在一起检查，原始来源与记忆历史仍会保留。

检索时，Wenlan 会用实体向量匹配找到与问题相关的实体。存在符合条件的图谱链接时，默认开启的图谱记忆信号（graph-memory stream）会把相连记忆作为第三路 [RRF](https://cormack.uwaterloo.ca/cormacksigir09-rrf.pdf) 排名信号加以提升。这个路径取决于现有图谱数据与读取范围，Space 边界仍然有效。[查看图谱检索如何工作 ->](docs/technical-foundations.md#graph-assisted-retrieval)

<a id="retrieval"></a>

### 从关键词、语义与关联找回正确内容

Wenlan 的核心搜索是本地混合检索流程，不是单一的向量查询。每个阶段负责不同工作：

- **原词匹配，[SQLite FTS5](https://www.sqlite.org/fts5.html)：** 全文索引查找字面关键词、标识符与短语。
- **相近含义，FastEmbed + [`Qdrant/bge-base-en-v1.5-onnx-Q`](https://huggingface.co/Qdrant/bge-base-en-v1.5-onnx-Q)：** 量化的英文模型会产生 768 维语义向量；[libSQL cosine DiskANN](https://turso.tech/blog/approximate-nearest-neighbor-search-with-diskann-in-libsql) 再以近似最近邻搜索（ANN）快速取得候选。
- **合并排名，加权 [RRF](https://cormack.uwaterloo.ca/cormacksigir09-rrf.pdf)（`k = 60`）：** 融合原词与语义排名，不假设两者的原始分数采用同一尺度；向量信号还会由余弦相似度加权。
- **关联脉络，图谱记忆信号（graph-memory stream）：** 符合条件的实体链接会加入第三路 RRF 信号，返回的记忆仍受当前读取范围限制。
- **可选精排，交叉编码器（cross-encoder）：** 与分别编码查询和记忆的 embedding 不同，[`jinaai/jina-reranker-v1-turbo-en`](https://huggingface.co/jinaai/jina-reranker-v1-turbo-en) 或 [`BAAI/bge-reranker-base`](https://huggingface.co/BAAI/bge-reranker-base) 会同时读取查询与单个候选，再对较小的候选池重新排名；默认关闭。

页面、情节记忆与事实（fact）通道都需要主动启用；不可用时会退回其余搜索信号。Space 仍负责限制读取范围。[查看方法、默认值与限制 ->](docs/technical-foundations.md)

<a id="what-makes-wenlan-distinct"></a>
<a id="why-is-wenlan-different"></a>
<a id="two-lifecycles"></a>

### 两套生命周期，一个持续维护的知识系统

一次生成的 wiki 会过时；只存记忆又容易碎成互不相连的事实。Wenlan 连接两套生命周期，但不把它们混成同一层。

<p align="center">
  <picture>
    <source media="(max-width: 600px)" srcset="./docs/assets/wenlan-lifecycle-zh-Hans-mobile.png">
    <img src="./docs/assets/wenlan-lifecycle-zh-Hans.png" alt="明确取代旧说法的新记忆仍会保留前后关联。页面过时后，Wenlan 会依当前来源与记忆重建、记录修订，并把对人工文字的改动变成审核提案。" width="100%">
  </picture>
</p>

#### 原子记忆

`CAPTURE -> CLASSIFY -> ENRICH -> LINK -> RECONCILE`

Capture 与明确的 supersession 属于核心流程。模型支持的阶段只会在配置相应模型后运行，Reconcile 默认关闭。

| 操作 | Wenlan 做什么 |
|---|---|
| **Capture** | AI agent 每次写入一条完整、自足的想法，遵循 Zettelkasten 的原子笔记原则，而不是保存整段对话。 |
| **Classify** | 配置本地模型后，Wenlan 将记忆分为 `identity`、`preference`、`decision`、`lesson`、`gotcha` 或 `fact`；调用方明确提供的准确类型优先。 |
| **Enrich** | 配置本地模型后，在可用时补充结构化字段、检索提示、事件日期、质量、重要性与标签。 |
| **Link** | 保留出处；启用 enrichment 后，把记忆连接到知识图谱中的实体与关系。 |
| **Reconcile** | 明确取代旧说法时保留 `supersedes` 链。若发起替换的 agent 信任级别低于 full，该替换会自动进入人工审核队列，无需任何开关。可选的本地模型流程还可以把受保护内容的冲突放入审核，而不是覆盖历史；这个流程默认关闭，必须明确启用。 |

高级设置：使用 `WENLAN_ENABLE_DUAL_POOL_RESOLVE=1` 启用这个 Reconcile 流程。

#### 持续维护的页面

`DISTILL -> CITE -> TRACK -> REFRESH -> REVIEW`

| 操作 | Wenlan 做什么 |
|---|---|
| **Distill** | 把相关来源与记忆汇总成一个 Markdown 页面。 |
| **Cite** | 保留引用记录与验证状态；自动 refresh 若未通过引用支撑检查，就会丢弃草稿。 |
| **Track** | 记录哪些证据支撑页面、页面为何过时，以及有上限的变更记录。 |
| **Refresh** | 页面被标记为过时后，依当前证据重建符合条件、由机器维护的页面。 |
| **Review** | 对你编辑过的页面提出修订，而不是静默改写。 |

例如，导入一份设计文档，再让 Codex 捕获一次调试决策。Wenlan 可以把两者整理成一个同时引用两份依据的页面。这个页面 refresh 时，会依当前依据重建；如果你已经编辑过它，改动提案会等待审核。

<a id="local-markdown"></a>

### 与 Obsidian 共存的本地 Markdown

长期知识保留为普通文件，不被锁在专有编辑器格式里：

- **纯文本文件：** 页面与 session notes 都以 Markdown 保存在 `~/.wenlan/`。
- **可检查的历史：** Distill 与 handoff 可以把逻辑上属于同一批的文件提交到本地 git repository。
- **与 Obsidian 共存：** Wenlan 把现有 vault 当作来源读取。你可以把 `~/.wenlan/pages/` symlink 到 vault，或从桌面 app 导出页面；你的编辑仍由你拥有，之后的机器更新会成为可审核的修订建议。

本地历史可以直接检查：

```text
$ git -C ~/.wenlan log --oneline
a1b2c3d distill: 4 pages
9f8e7d6 session: embedding-work
```

---

<a id="what-you-get"></a>
<a id="what-can-it-do"></a>
<a id="what-can-i-bring-in"></a>

## 能力

- **对话导入：** 导入 ChatGPT 或 Claude 的 ZIP；Wenlan 会自动跳过已经导入的对话。
- **文档来源：** 导入单个 `.md`、`.txt` 或可提取文本的 `.pdf`，递归读取包含它们的文件夹，或索引 Obsidian 仓库中的 Markdown。
- **增量同步：** 单个文件与普通文件夹来源会在后台追踪变化；Obsidian 仓库保持只读，按需重新同步。
- **原子记忆（Atomic Memory）：** MCP 客户端把一个完整的决策、经验、更正、偏好或事实存成一条，并用[溯源与取代](https://wenlan.app/learn/ai-memory-provenance)记录它来自哪里、替代了什么。
- **[类型化补全（Typed enrichment）](docs/technical-foundations.md#typed-memory-schema)：** 配好模型后，Wenlan 会分类每条记忆，并补上该类型定义的结构化字段、日期、标签、检索提示与图谱链接。
- **[来源支撑页面（Source-backed Pages）](https://wenlan.app/docs/source-backed-pages)：** 把相关来源与记忆提炼成带来源标记与 `[[wikilinks]]` 的 Markdown 页面；daemon 还能验证并记录逐条引用。
- **引用门控更新（Citation-gated refresh）：** 自动刷新时，引用支撑不足的草稿会被拒绝；机器页面可更新，用户文字只进入待审修订。
- **[混合检索（Hybrid retrieval）](docs/technical-foundations.md#retrieval-pipeline)：** FTS5 找原词，本地 BGE embedding 找语义，RRF 融合排序，图谱链接补充脉络。
- **[检索通道（Retrieval channels）](docs/technical-foundations.md#optional-channels-and-defaults)：** 可选的页面、情节记忆（episodic）与逐事实（per-fact）通道扩大召回；cross-encoder 重排提高精度。
- **[知识图谱（Knowledge graph）](docs/technical-foundations.md#graph-data-and-entity-resolution)：** 类型化实体、关系与观察连接人物、项目、主张及其支撑记忆。
- **[人在回路审查（Human-in-the-loop）](https://wenlan.app/docs/review-and-trust)：** 日常工作保持自动；受保护冲突、页面修订、实体合并与新词汇才等待判断。
- **[空间（Spaces）](https://wenlan.app/docs/spaces)：** 用明确范围隔开工作、个人、客户与代码库的记忆、页面和检索结果。
- **[本地 daemon + MCP](https://wenlan.app/docs/architecture)：** 一个轻量 Rust daemon 作为本地唯一事实来源。桌面 app 与 CLI 直接调用它；各 AI client 通过小型 MCP 连接器访问同一份知识。
- **自定义集成：** localhost HTTP API 可接收其他采集流程准备好的文本、网页内容与记忆。
- **后台维护：** 关闭桌面 app 后，daemon 仍会继续执行已配置的同步、补全、引用与页面更新。
- **[模型选择](docs/technical-foundations.md#model-roles)：** 基础检索留在本机；补全与合成可用设备端 Qwen、兼容 OpenAI 的本地接口或云端模型。
- **[可检查的所有权](https://wenlan.app/learn/markdown-local-index-ai-memory)：** 记忆与图谱留在本地 libSQL；Markdown、引用、修订、git 历史与 Obsidian 导出都可检查。
- **只读健康检查：** [`doctor`](https://wenlan.app/docs/diagnostics-and-issue-reports) 检查本地服务；[`lint`](plugin/skills/lint/SKILL.md) 找出格式错误的引用、孤立链接、损坏的 embedding，以及搜索索引或图谱完整性问题，但不会改写知识。

---

<a id="how-wenlan-works"></a>
<a id="how-does-it-work"></a>

## 日常流程

日常使用分成一个小循环：取回相关知识、保存工作重点、以 handoff 收尾，再由 Wenlan 整理下次需要的内容。每一轮都改善同一个知识库，不再累积互不相连的历史。

这个循环分成四步：

1. **找到最新知识。** 打开相关 Page、搜索，或使用 `/recall <query>`；`/brief [topic]` 读取当前 Space Brief；提供 topic 时，会另行附上同一 Space 的相关上下文。其他 AI 工具可使用等价的 page、search、recall 与 brief 工具。
2. **工作时随手保存与查找。** `/capture <thing>` 保存决策、经验、踩坑或事实，并记录来源。`/recall <query>` 只取回相关内容，不加载全部历史。
3. **闭合循环。** `/handoff` 记录本次改动，并把类型化的逐项更新应用到当前 Space Brief。
4. **让 wiki 保持最新。** `/distill` 主动建立或刷新页面。可选的模型流程会在两次工作之间补充已保存内容、连接相关知识，并刷新符合条件的页面。`/lint` 检查知识库健康状态；`/curate` 让你审核页面更新提案，以及可选 Reconcile 流程产生的冲突项目。

### 离线队列（outbox）

如果本地守护进程无法访问，`wenlan capture` 和 `wenlan brief update` 会把请求写入本地持久化队列（outbox）并正常退出。守护进程恢复后，它会通过常规 HTTP 路由排空这些写入；用 `wenlan outbox status` 查看队列，或用 `wenlan outbox drain` 立即重放。被守护进程直接拒绝的写入（4xx，例如未通过内容质量检查）会带着回执移动到 `outbox/failed/`，而不是无限重试；传输失败或服务器错误（5xx）则留在队列中等待下一次排空，排空每 60 秒自动运行一次。

### 模型与隐私

- **本地基础检索：** [BGE 向量模型（embedding model）](https://huggingface.co/Qdrant/bge-base-en-v1.5-onnx-Q) 通过 FastEmbed 在你的设备上运行，用于混合搜索，不需要 API key。
- **可选的设备端整理：** 内容补充（enrichment）与页面汇总可以使用你选择的 [`Qwen3 4B`](https://huggingface.co/unsloth/Qwen3-4B-Instruct-2507-GGUF) 或 [`Qwen3.5 9B`](https://huggingface.co/unsloth/Qwen3.5-9B-GGUF)，并通过 [llama.cpp](https://github.com/ggml-org/llama.cpp) 运行。你没有选择前，Wenlan 不会下载或启用语言模型。
- **其他模型来源：** Ollama 或 LM Studio 等 OpenAI 兼容的本地端点，或已设置的云端 provider，也可以提供模型支持的内容补充与页面汇总。
- **云端说明：** 如果你选择的模型端点位于远端，Wenlan 会把该任务需要的 system prompt 与 user prompt 发给它。本地检索与设备端整理仍留在你的设备上。
- **无遥测：** Wenlan 不发送使用遥测。

完整 workflow 参考：[plugin/skills](plugin/skills/README.md)。模型分工与限制见：[技术基础（英文）](docs/technical-foundations.md#model-roles)。

---

<a id="evaluation"></a>

## 评估

以下是 retrieval-only snapshot，不代表 end-to-end answer quality。方法、环境 receipts 与更新流程见 [docs/eval](docs/eval/README.md)。

<!-- EVAL_SNAPSHOT_START -->
| Benchmark | Recall@5 | MRR | NDCG@10 |
|---|---:|---:|---:|
| LME_Oracle (500 Q) | 93.6% | 0.857 | 0.883 |
| LME_S (deep, 90 Q) | 87.7% | 0.815 | 0.822 |
<!-- EVAL_SNAPSHOT_END -->

---

<a id="learn-more"></a>

## 进一步了解

更完整的文档、概念说明与比较：

### 文档

- [开始使用](https://wenlan.app/docs/get-started)：安装并验证第一个本地循环。
- [日常工作流程](https://wenlan.app/docs/daily-workflow)：brief、capture、recall、handoff、distill、lint 与 curate。
- [MCP 客户端](https://wenlan.app/docs/mcp-clients)：连接 Claude Code、Codex、Cursor、Claude Desktop 与其他工具。

### 概念

- [为什么需要持续演进的 wiki，而不只是 AI 记忆](https://wenlan.app/learn/ai-work-memory)：深入理解问题与产品模型。
- [MCP 记忆服务器](https://wenlan.app/learn/mcp-memory-server)：Wenlan 如何让知识跨 AI 工具使用。
- [本地优先的 AI 记忆](https://wenlan.app/learn/local-first-ai-memory)：数据、隐私与控制权。
- [Markdown 与本地索引](https://wenlan.app/learn/markdown-local-index-ai-memory)：存储、检索与所有权。
- [AI agent 的交接循环](https://wenlan.app/learn/ai-agent-handoff-loop)：把工作完整带到下一次会话。

### 比较

- [Wenlan 与 Basic Memory](https://wenlan.app/learn/wenlan-vs-basic-memory)
- [Wenlan 与 claude-mem](https://wenlan.app/learn/wenlan-vs-claude-mem)
- [Wenlan 与 Superlocal Memory](https://wenlan.app/learn/wenlan-vs-superlocal-memory)

---

## 贡献

欢迎 bug fixes、eval cases、文档与功能。安装 Wenlan 不需要从源码构建。本地开发时，请从本 repository 的根目录运行以下命令：

```bash
# daemon crates（default-members——不会编译桌面 app）
cargo build
cargo test

# 桌面 app（Cargo target 与根目录的前端工具链）
pnpm install
pnpm dev:all
pnpm build:all
```

`pnpm dev:all` 是桌面 app 受支持的开发入口。它让开发用的端口、数据、进程归属、app 标识、MCP socket 与 Remote Access 状态都与已安装的生产运行时隔离；未处于该隔离环境的 debug build 会拒绝启动。完整开发流程见本 repository 的 [AGENTS.md](AGENTS.md) 与 [CONTRIBUTING.md](CONTRIBUTING.md)，以及仓库内的 [app/AGENTS.md](app/AGENTS.md)。安全性问题请见 [SECURITY.md](SECURITY.md)，隐私政策请见 [PRIVACY.md](PRIVACY.md)，也请阅读 [Code of Conduct](CODE_OF_CONDUCT.md)。

---

<a id="license"></a>

## 许可

Wenlan 采用 **Apache-2.0** 许可，包括本 repository 内的 local runtime、CLI、MCP server、shared types，以及 Claude Code/Codex plugin files。

---

<a id="acknowledgments"></a>

## 源流与同类项目

Wenlan（文澜）的名字来自文澜阁。这座皇家藏书楼收藏《四库全书》，曾是中国最大的藏书之一。

Wenlan 的 llm-wiki v2 模型是自己的产品方向，并受到 LLM-wiki 与 agent-memory 两条脉络启发：

- [Karpathy 的 LLM-wiki note](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f) 建立了从 raw sources 到持续维护 wiki 的模式。
- [Rohitg00 的 LLM Wiki v2 proposal](https://gist.github.com/rohitg00/2067ab416f7bbe447c1977edaaa681e2) 加入 memory lifecycle、confidence、graph 与 retrieval mechanisms。[agentmemory](https://github.com/rohitg00/agentmemory) 是其具体的 agent-memory implementation。
- [nashsu/llm_wiki](https://github.com/nashsu/llm_wiki) 是以文档为核心的 LLM-wiki 完整桌面实现。
- [basic-memory](https://github.com/basicmachines-co/basic-memory)、[obsidian-mind](https://github.com/breferrari/obsidian-mind)、[mcp-memory-service](https://pypi.org/project/mcp-memory-service/)、[Memoria](https://github.com/matrixorigin/Memoria) 和 [OpenMemory](https://github.com/CaviraOSS/OpenMemory) 探索相邻的本地知识与 agent-memory 方向。
