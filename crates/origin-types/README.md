# origin-types

Shared wire-format types for [Origin](https://github.com/7xuanlu/origin) — a personal agent memory system.

This crate defines the HTTP API request/response types and core enums used by:
- `origin-server` (HTTP backend daemon)
- `origin-mcp` (MCP server wrapper for AI tools)
- the Origin desktop app (Tauri IPC)

## Stability

Pre-1.0 — expect minor version bumps to include breaking changes, per Rust 0.x convention. Types marked `#[doc(hidden)]` are not part of the stability contract and may change without a version bump.

## License

Apache-2.0
