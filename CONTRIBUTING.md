# Contributing to Origin

Origin is a local-first personal AI memory layer. We welcome bug fixes, features, tests, docs, and design feedback.

## Development Setup

**Requirements:** macOS Apple Silicon (M1+), [Xcode Command Line Tools](https://developer.apple.com/xcode/resources/), [Rust](https://rustup.rs/) (stable), [Node.js](https://nodejs.org/) 20+, [pnpm](https://pnpm.io/)

```bash
git clone https://github.com/7xuanlu/origin.git
cd origin
pnpm install
```

Single command builds the daemon, starts it, and launches the Tauri app:

```bash
pnpm dev:all
```

Or run daemon and app separately:

```bash
cargo run -p origin-server          # terminal 1
pnpm tauri dev                      # terminal 2
```

> First build can take several minutes while `llama.cpp` compiles for Metal.

### Building a release

```bash
pnpm release            # builds release daemon, copies to sidecar, builds .app
pnpm release:dmg                # creates DMG (uses hdiutil, no external deps)
```

Output: `target/release/bundle/macos/Origin.app` and `target/release/bundle/dmg/Origin_0.1.0_aarch64.dmg`.

`CXXFLAGS="-std=c++17"` is set automatically by `release` (required on macOS 26.x for llama.cpp C++17 features). Do not run `pnpm dev:daemon` before `release` as it overwrites sidecar binaries with debug builds.

### Running Tests

```bash
# Rust workspace
cargo test --workspace

# Frontend (React)
pnpm test

# Optional convenience script (if present in package.json)
pnpm test:all
```

### Linting

```bash
cargo fmt --check --all
cargo clippy --workspace --all-targets -- -D warnings
```

## Architecture Overview

- **Shared types**: `crates/origin-types` (Apache-2.0)
- **Core logic**: `crates/origin-core` (Apache-2.0)
- **HTTP daemon**: `crates/origin-server` (Apache-2.0), serves `127.0.0.1:7878`
- **Desktop app**: `app/` (AGPL-3.0-only), thin Tauri client + macOS integrations
- **Frontend**: `src/` React app consumed by Tauri webview
- **Database**: libSQL (vectors + knowledge graph + FTS)

See `CLAUDE.md` for a full module-by-module breakdown.

## Finding Work

Look for issues labeled [`good first issue`](https://github.com/7xuanlu/origin/labels/good%20first%20issue) or [`help wanted`](https://github.com/7xuanlu/origin/labels/help%20wanted).

## Pull Request Process

1. Fork the repo and create a branch from `main`
2. Make your changes — keep PRs small and focused (one logical change per PR)
3. Ensure all tests pass and linting is clean
4. Open a PR using the template — describe what and how to test

CI runs `cargo fmt --check --all`, `cargo clippy --workspace --all-targets`, `cargo test --workspace`, and `pnpm test`.

## Code Conventions

These conventions keep the codebase consistent. See `CLAUDE.md` for the full list.

- **SQL safety**: Always use parameterized queries — never interpolate user input into SQL strings
- **NULL semantics**: Store `Option<T>` as SQL NULL, not empty string
- **UTF-8 safety**: Never byte-index Rust strings (`&s[..n]`) — use `chars().take(n)` instead
- **Batch SQL**: Wrap multi-row insert/delete loops in `BEGIN`/`COMMIT` transactions
- **License headers**: The workspace is still normalizing SPDX headers after the package split. For new files, use the header that matches the package/file license even if nearby legacy files have not been cleaned up yet.

## Docs Layout

- In-repo docs live under `docs/` (especially `docs/plans/` for historical implementation context).
- Some personal/internal notes may exist outside the repository and are not required for contributors.

## License

Origin is mixed-license: `crates/origin-types`, `crates/origin-core`, and `crates/origin-server` are Apache-2.0; `app/` and the frontend UI are AGPL-3.0-only.

By contributing, you agree that your changes will be licensed under the license that applies to the files you modify.
