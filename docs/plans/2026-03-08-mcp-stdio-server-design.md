# MCP Server (stdio transport) — Design

## Overview

Add a standalone `origin-mcp` binary that exposes Origin's memory and knowledge graph capabilities via the Model Context Protocol (stdio transport). AI tools (Claude Code, Cursor, etc.) connect to it as an MCP server. The binary is a thin JSON-RPC ↔ REST translator — it calls the running Origin Tauri app's HTTP/UDS API with zero shared code.

Designed for future open-sourcing: only the MCP crate is published, not the Origin backend or UI.

## Architecture

```
AI Tool (Claude Code, Cursor, etc.)
  ↕ stdio (JSON-RPC 2.0)
origin-mcp binary (separate crate)
  ↕ HTTP or Unix Domain Socket (reqwest)
Origin Tauri app (127.0.0.1:7878 or /tmp/origin-server-*.sock)
```

## Crate Structure

```
origin-mcp/
├── Cargo.toml        # standalone, publishable independently
├── src/
│   ├── main.rs       # CLI args + stdio JSON-RPC loop via rmcp
│   ├── client.rs     # HTTP/UDS client to Origin REST API
│   ├── tools.rs      # MCP tool definitions + dispatch
│   └── types.rs      # request/response types (mirrors Origin API contract)
```

### Dependencies

- `rmcp` — Rust MCP SDK (stdio transport, JSON-RPC, tool macros)
- `reqwest` — HTTP client
- `serde` / `serde_json` — serialization
- `tokio` — async runtime
- `clap` — CLI args (`--origin-url`, `--origin-socket`)
- `tracing` / `tracing-subscriber` — logging to stderr

### No dependency on `origin_lib`

The open-source boundary is the HTTP API contract. The MCP crate only knows request/response JSON shapes.

## Server Discovery

In order:
1. `--origin-url` CLI flag if provided
2. Scan `/tmp/origin-server-*.sock` for existing Unix socket
3. Fall back to `http://127.0.0.1:7878`
4. If nothing responds, tools return `isError: true` with "Origin app is not running"

## Error Handling

- Origin API errors → MCP tool error responses (`isError: true`, human-readable message)
- Origin unreachable → same pattern, MCP connection stays alive
- Protocol errors → handled by `rmcp`
- Logging → stderr only (stdout reserved for JSON-RPC)

## Tool Surface (10 tools)

### Memory CRUD

| Tool | Annotation | Params | Origin Endpoint |
|------|-----------|--------|-----------------|
| `store_memory` | destructive: false | `text` (req), `memory_type?`, `domain?`, `source_agent?`, `confidence?` | `POST /api/memory/store` |
| `search_memory` | readOnly: true | `query` (req), `limit?`, `memory_type?`, `domain?`, `source_agent?` | `POST /api/memory/search` |
| `list_memories` | readOnly: true | `memory_type?`, `domain?`, `limit?` | `GET /api/memory/list` |
| `delete_memory` | destructive: true | `source_id` (req) | `DELETE /api/memory/delete/{source_id}` |

### Knowledge Graph

| Tool | Annotation | Params | Origin Endpoint |
|------|-----------|--------|-----------------|
| `create_entities` | destructive: false | `entities[]` (req): `{name, entity_type, domain?, source_agent?, confidence?}` | `POST /api/memory/entities` (loop) |
| `create_relations` | destructive: false | `relations[]` (req): `{from_entity, to_entity, relation_type, source_agent?}` | `POST /api/memory/relations` (loop) |
| `add_observations` | destructive: false | `observations[]` (req): `{entity_id, content, source_agent?, confidence?}` | `POST /api/memory/observations` (loop) |

### Retrieval

| Tool | Annotation | Params | Origin Endpoint |
|------|-----------|--------|-----------------|
| `search` | readOnly: true | `query` (req), `limit?`, `source?` | `POST /api/search` |
| `chat_context` | readOnly: true | `messages[]` (req): `{role, content}`, `limit?`, `threshold?` | `POST /api/chat-context` |

### Meta

| Tool | Annotation | Params | Origin Endpoint |
|------|-----------|--------|-----------------|
| `health` | readOnly: true | (none) | `GET /api/health` |

## User Configuration

```json
{
  "mcpServers": {
    "origin": {
      "command": "origin-mcp",
      "args": []
    }
  }
}
```

## Future Tools (not in initial release)

- `confirm_memory` — mark memory as human-verified
- `update_memory` — edit existing memory content
- `delete_entities` / `delete_relations` / `delete_observations` — KG cleanup
- `read_graph` / `open_nodes` — KG traversal and retrieval
- `ingest_text` / `ingest_webpage` — raw content ingest

## Conventions

- Batch inputs for KG tools (arrays, not single items)
- Tool annotations (`readOnlyHint`, `destructiveHint`) on all tools
- Snake_case tool names
- Tools only — no MCP resources or prompts initially
