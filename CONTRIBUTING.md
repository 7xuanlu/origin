# Contributing to Wenlan

Wenlan is a local-first personal AI memory layer. We welcome bug fixes, features, tests, docs, and design feedback.

This repo holds the daemon (`wenlan-server`), the CLI (`wenlan`), the MCP server (`wenlan-mcp`), the shared types/core (`wenlan-types`, `wenlan-core`), and the Tauri desktop app (`wenlan-app`, in `app/`). Bug reports for the local runtime, CLI, MCP server, desktop app, and plugin are welcome here.

## Development Setup

**Requirements:** macOS arm64, Linux (x86_64 + arm64; glibc), or Windows x86_64; platform build tools ([Xcode Command Line Tools](https://developer.apple.com/xcode/resources/) on macOS, MSVC Build Tools on Windows, gcc + make on Linux); [Rust](https://rustup.rs/) (stable).

Building the **desktop app** additionally needs [Node.js](https://nodejs.org/) with [pnpm](https://pnpm.io/), and [`cloudflared`](https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/) on `PATH` (or `CLOUDFLARED_BIN` pointing at it). Wenlan ships `cloudflared` as a Tauri `externalBin` for Remote Access, so `pnpm dev:all`, `pnpm build:all`, and `pnpm tauri build` all fail without it. The daemon, CLI, and MCP crates build without either.

macOS Intel is not currently a supported stock build target because the pinned ONNX Runtime dependency has no prebuilt x86_64 macOS binary. See the [platform note](crates/wenlan-cli/README.md#macos-intel).

```bash
git clone https://github.com/7xuanlu/wenlan.git
cd wenlan
cargo build -p wenlan-server
```

Run the daemon directly:

```bash
cargo run -p wenlan-server
```

Or register it as the per-user background service:

```bash
cargo build --release -p wenlan -p wenlan-server
./target/release/wenlan setup --basic
./target/release/wenlan background on
./target/release/wenlan status
```

> First build can take several minutes while `llama.cpp` compiles for Metal.

### Running Tests

```bash
# Workspace tests (excludes wenlan-app; see Architecture Overview)
cargo test --workspace --exclude wenlan-app

# Per-crate
cargo test -p wenlan-types
cargo test -p wenlan-core --lib
cargo test -p wenlan-server
cargo test -p wenlan
```

### Linting

```bash
cargo fmt --check --all
cargo clippy --workspace --exclude wenlan-app --all-targets -- -D warnings
```

## Architecture Overview

- **Shared types**: `crates/wenlan-types` (Apache-2.0). Lightweight wire types shared with `wenlan-mcp` and `wenlan-app` as in-workspace path deps.
- **Core logic**: `crates/wenlan-core` (Apache-2.0). DB, embeddings, LLM engine, search, knowledge graph, distill cycles, eval. No tauri / no axum dependencies.
- **HTTP daemon**: `crates/wenlan-server` (Apache-2.0), serves `127.0.0.1:7878`.
- **CLI binary**: `crates/wenlan-cli` (Apache-2.0). The `wenlan` command for setup, service management, search, recall, etc.
- **MCP server**: `crates/wenlan-mcp` (Apache-2.0). The connector spawned by Claude Code, Cursor, Codex, and other MCP clients.
- **Desktop app**: `wenlan-app` in `app/` (AGPL-3.0-only). Tauri + React; folded into this monorepo on 2026-07-20.
- **Database**: libSQL (vectors + knowledge graph + FTS).

See `crates/wenlan-core/REFERENCE.md` and `crates/wenlan-server/REFERENCE.md` for the module-by-module breakdown.

## Finding Work

Look for issues labeled [`good first issue`](https://github.com/7xuanlu/wenlan/labels/good%20first%20issue) or [`help wanted`](https://github.com/7xuanlu/wenlan/labels/help%20wanted).

## Pull Request Process

1. Fork the repo and create a branch from `main`
2. Make your changes, keeping PRs small and focused (one logical change per PR)
3. Ensure all tests pass and linting is clean
4. Open a PR using the template, describing what changed and how to test it

CI runs `cargo fmt --check --all`, `cargo clippy --workspace --exclude wenlan-app --all-targets -- -D warnings`, and cargo-nextest slices across the daemon crates. The `wenlan-app` desktop crate is checked separately, after sidecar binaries are staged.

## Code Conventions

These conventions keep the codebase consistent.

- **SQL safety**: Always use parameterized queries. Never interpolate user input into SQL strings.
- **NULL semantics**: Store `Option<T>` as SQL NULL, not empty string
- **UTF-8 safety**: Never byte-index Rust strings (`&s[..n]`). Use `chars().take(n)` instead.
- **Batch SQL**: Wrap multi-row insert/delete loops in `BEGIN`/`COMMIT` transactions
- **License headers**: The workspace is still normalizing SPDX headers after the package split. For new files, use the header that matches the package/file license even if nearby legacy files have not been cleaned up yet.

## Docs Layout

- In-repo docs live under `docs/` and are documentation only. Nothing there is read by code or CI.
- Working design docs (plans, specs, research) are deliberately not tracked. `docs/plans/` and `docs/superpowers/` are gitignored local scratch space, so nothing you need to contribute lives there.
- `crates/wenlan-core/contracts/` holds two markdown files that are test fixtures rather than documentation: `crates/wenlan-core/src/truth_manifest_test.rs` compiles one in with `include_str!`, and `crates/wenlan-core/src/m6/catalog_test.rs` reads the other at test time. Editing either changes what a test asserts.

## License

This repo is Apache-2.0: `crates/wenlan-types`, `crates/wenlan-core`, `crates/wenlan-server`, `crates/wenlan-cli`, `crates/wenlan-mcp`, and the Claude Code plugin files. The `wenlan-app` desktop crate (`app/`) is AGPL-3.0-only via its own `license` field.

By contributing, you agree that your changes will be licensed under the license that applies to the files you modify.

## Links

- [wenlan.app](https://wenlan.app): project home
- [wenlan.app/docs/get-started](https://wenlan.app/docs/get-started): install and verify the local memory loop before opening a PR
- [wenlan.app/docs/daily-workflow](https://wenlan.app/docs/daily-workflow): the workflow your changes will fit into
