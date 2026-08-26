# wenlan-mcp

MCP server for [Wenlan](https://github.com/7xuanlu/wenlan). It lets Claude Code, Cursor, Codex, Claude Desktop, Gemini CLI, and other MCP clients read and write to the local Wenlan daemon through the [Model Context Protocol](https://modelcontextprotocol.io).

Wenlan owns storage, search, embeddings, pages, and distill cycles. `wenlan-mcp` is the connector.

## Install

Most users should install the runtime through the root README (`npx -y wenlan setup` on macOS Apple Silicon, the automated shell setup on Linux, or the matching release archive on Windows). Then use the product CLI to configure supported clients:

```bash
wenlan connect codex              # or: claude-code, cursor, claude-desktop, vscode, gemini
wenlan connect cursor --dry-run   # preview before editing JSON config
```

MCP-only setup gives agents tools for capture, recall, a Space-owned Brief, and page distillation. It does not install Claude Code slash skills like `/brief`, `/handoff`, `/distill`, or `/setup`; use the Wenlan plugin for that workflow.

If you only need the raw MCP connector config, add this to your MCP client:

```json
{
  "mcpServers": {
    "wenlan": {
      "command": "npx",
      "args": ["-y", "wenlan-mcp"]
    }
  }
}
```

The npm wrapper auto-detects the host platform and downloads the matching prebuilt binary from the Wenlan release. Supported: macOS (arm64), Linux (x64, arm64; glibc), Windows (x64). Other targets require building the connector from source via `cargo install --locked wenlan-mcp`; macOS Intel does not currently have a supported complete local runtime.

Or install a binary directly:

```bash
brew install 7xuanlu/tap/wenlan-mcp
cargo install --locked wenlan-mcp
```

`--locked` builds with the dependency versions the release was tested with; a plain `cargo install wenlan-mcp` also works, since the crate pins its proc-macro crate to the same minor as `rmcp`.

Then use:

```json
{
  "mcpServers": {
    "wenlan": {
      "command": "wenlan-mcp"
    }
  }
}
```

`wenlan-mcp` expects the Wenlan daemon at `http://127.0.0.1:7878` by default. Override it with:

```bash
wenlan-mcp --origin-url http://127.0.0.1:7879
```

## Tools

The local stdio surface is locked at exactly 29 tools. The table below calls
out the primary memory loop and the unique refinement-review queue.

| Tool | Purpose |
| --- | --- |
| `brief` | Read the current Space Brief; an optional topic appends separately labeled same-Space context. |
| `capture` | Save one durable memory and return its explicit `source_memory_id`. |
| `recall` | Search memories and pages by natural-language query. |
| `distill` | Trigger page distillation for new clusters or a specific `page_id`. |
| `list_pending` | List unconfirmed memories waiting for review. |
| `confirm_memory` | Confirm a pending memory by `source_id`. |
| `forget` | Delete a memory by ID. Destructive. |
| `list_refinements` | Explicitly inspect the daemon's unique proposal queue, including `vocab_promote`; never polled ambiently. |
| `accept_refinement` | Accept one listed proposal after an unambiguous item-level decision. Local stdio only. |
| `reject_refinement` | Reject one listed proposal after an unambiguous item-level decision. Local stdio only. |

The refinement trio remains because this review queue has no CLI or replacement
path. Remote HTTP clients can list it, but `accept_refinement` and
`reject_refinement` are hidden and hard-rejected remotely.

Runtime diagnostics live in the CLI: `wenlan doctor`. They are not part of the
MCP memory loop.

## Setup Modes

Wenlan works immediately in **local memory** mode: storage, search, recall, and MCP memory are available without a local model or API key.

Users can opt into more expensive distill cycles:

- **On-device model:** private extraction and distillation after `wenlan models install`.
- **Anthropic key:** richer extraction and page synthesis after `wenlan keys set anthropic`.

## Agent Guidance

The MCP server ships tool instructions that tell agents to capture durable state proactively:

- One idea per capture.
- Include the why, not just the what.
- Name people, projects, and tools explicitly.
- Omit `memory_type` unless the agent is certain.
- Do not store tool output, command logs, filler, or transient task state.

See [`src/tools.rs`](src/tools.rs) for the full instructions.

## Links

- [wenlan.app](https://wenlan.app) — project home
- [wenlan.app/learn/mcp-memory-server](https://wenlan.app/learn/mcp-memory-server) — concept article on Wenlan as an MCP memory server
- [wenlan.app/docs/mcp-clients](https://wenlan.app/docs/mcp-clients) — connect Claude Code, Cursor, Codex, Claude Desktop, Gemini CLI
- [npm: wenlan-mcp](https://www.npmjs.com/package/wenlan-mcp) — standalone npm package
- [github.com/7xuanlu/wenlan](https://github.com/7xuanlu/wenlan) — source

## License

Apache-2.0.
