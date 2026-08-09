# Wenlan App Refactor Tooling Preflight

- **Date:** 2026-06-25
- **Worktree:** `/Users/lucian/Repos/wenlan-app/.worktrees/wenlan-app-convergence`
- **Branch:** `codex/wenlan-app-convergence`
- **Base:** `af618e5 frontend: align activity strings + CSS var with daemon page rename`
- **Purpose:** establish repeatable structural search and baseline checks before API/identity refactor edits.
- **Current status:** refreshed on 2026-06-26 after the local/GitHub repo rename to `wenlan-app`.

## Baseline

Frontend setup remains repeatable:

```bash
pnpm install --frozen-lockfile --offline
pnpm test
```

Observed frontend result:

```text
Test Files 35 passed (35)
Tests 319 passed | 1 skipped (320)
```

Rust sidecar setup has moved past the original stale `origin-server` failure. Current direct build evidence:

```bash
cargo build
```

Observed on 2026-06-26:

```text
Finished `dev` profile [unoptimized + debuginfo] target(s) in 50.77s
```

Tauri debug bundle generation also reaches a built `.app`; it still exits at the updater bundle signing step when `TAURI_SIGNING_PRIVATE_KEY` is unset. That signing error is not an app compile or sidecar-contract failure.

## Structural Tools

CodeGraph was evaluated against the exact requested project:

```bash
CODEGRAPH_TELEMETRY=0 DO_NOT_TRACK=1 npx -y @colbymchenry/codegraph init /Users/lucian/Repos/wenlan-app/.worktrees/wenlan-app-convergence
CODEGRAPH_TELEMETRY=0 DO_NOT_TRACK=1 npx -y @colbymchenry/codegraph status /Users/lucian/Repos/wenlan-app/.worktrees/wenlan-app-convergence
```

Observed result:

```text
Indexed 153 files
2,002 nodes, 4,814 edges in 910ms
DB Size: 6.24 MB
```

Use it for symbol navigation, impact checks, and refactor blast-radius discovery:

```bash
CODEGRAPH_TELEMETRY=0 DO_NOT_TRACK=1 npx -y @colbymchenry/codegraph sync .
CODEGRAPH_TELEMETRY=0 DO_NOT_TRACK=1 npx -y @colbymchenry/codegraph query WenlanClient --json
CODEGRAPH_TELEMETRY=0 DO_NOT_TRACK=1 npx -y @colbymchenry/codegraph impact WenlanClient --json
CODEGRAPH_TELEMETRY=0 DO_NOT_TRACK=1 npx -y @colbymchenry/codegraph affected app/src/api.rs --json
```

Do not commit `.codegraph/`; it is a generated local cache. Do not run `codegraph install` during this prereq phase because it mutates shared agent configuration.

`ast-grep` is available through:

```bash
npx -y -p @ast-grep/cli sg --version
```

Verified version:

```text
ast-grep 0.44.0
```

The reusable inventory command is:

```bash
bash scripts/refactor/inventory.sh
```

It writes:

- `docs/superpowers/refactor/wenlan-app-inventory/summary.md`
- `docs/superpowers/refactor/wenlan-app-inventory/tauri-ts-outline.txt`
- `docs/superpowers/refactor/wenlan-app-inventory/api-rs-outline.txt`
- `docs/superpowers/refactor/wenlan-app-inventory/search-rs-outline.txt`
- `docs/superpowers/refactor/wenlan-app-inventory/frontend-invokes.txt`
- `docs/superpowers/refactor/wenlan-app-inventory/rust-structs.txt`

## Current Inventory Counts

From `bash scripts/refactor/inventory.sh` after the typed-client and sidecar convergence work:

| Surface | Count | Meaning |
|---|---:|---|
| `src/lib/tauri.ts` `invoke(...)` calls | 122 | frontend-to-Tauri wrapper surface |
| `app/src/lib.rs` registered `search::...` commands | 166 | Rust command registration surface |
| Rust `origin_types` references | 0 | typed contract migration is no longer blocked by stale crate imports |
| runtime identity references | 222 | package/service/MCP/remote/app-name migration surface |
| stale taxonomy references | 239 | `concept`/`goal`/`domain` review surface |
| source files under `app/src` and `src` | 151 | working code surface size |

## Refactor Optimizer Rules

Use CodeGraph and structural search first, then text search:

1. Use `codegraph sync .` before cross-cutting edits.
2. Use `codegraph query <symbol> --json` and `codegraph impact <symbol> --json` for blast-radius orientation.
3. Use `codegraph affected <file> --json` to seed targeted test selection, but treat it as advisory.
4. Use `sg outline` to map exported TypeScript wrappers, Rust client methods, and Rust Tauri commands.
5. Use `sg run -p 'invoke($CMD, $$$ARGS)' -l ts src` to inventory frontend command calls.
6. Use `sg run -p 'pub struct $NAME { $$$FIELDS }' -l rs app/src` to find local DTO shadows before replacing them with `wenlan-types`.
7. Use `rg` residual checks only after graph and structural surfaces are mapped.
8. Keep the first refactor steps contract-only: no shell redesign, no visual rewrite, no public rename before typed API and compatibility gates are in place.

## Tool Authority Boundaries

| Tool | Trust it for | Do not trust it for | Typical command |
|---|---|---|---|
| CodeGraph | "What code probably depends on this symbol/file?" and "which tests might be affected?" | deterministic counts, syntactic rewrite completeness, type correctness | `codegraph query WenlanClient --json`; `codegraph impact WenlanClient --json`; `codegraph affected app/src/api.rs --json` |
| ast-grep | "Which code has this exact syntax shape?" and "what is the repeatable inventory/codemod surface?" | semantic call graph, type/import validity, runtime behavior | `sg run -p 'invoke($CMD, $$$ARGS)' -l ts src`; `sg run -p 'pub struct $NAME { $$$FIELDS }' -l rs app/src` |
| LSP/compiler | "Do imports, types, signatures, and references still make sense?" | migration scope discovery, feature parity, runtime side effects | rust-analyzer/tsserver diagnostics; `cargo check`; `pnpm build` |
| Tests/builds | "Does the edited behavior work?" | finding every impacted file before the edit | `pnpm test`; targeted Vitest; Rust tests; Tauri build checks |
| `rg` | "Are stale strings or allowed legacy tokens still present?" | primary planning for cross-file behavior | `rg -n 'origin-types|origin_types::|OriginClient|com\\.origin|Origin\\.app' app/Cargo.toml app/src` |
| `grep` | "What text can I still find when the better tools are unavailable?" | structural inventory, semantic correctness, rewrite safety | `grep -RIn --exclude-dir=node_modules --exclude-dir=target --exclude-dir=.codegraph 'OriginClient' app src` |

The boundary is intentional: CodeGraph reduces token-heavy exploration, ast-grep makes inventory reproducible, LSP/compiler catches semantic breakage, and tests/builds provide evidence. `rg` is the normal residual text check. `grep` is the fallback if none of the higher-signal lanes work for the current surface. A task is not complete merely because one lane is green.

## Known Hotspots

- `src/lib/tauri.ts`: public frontend wrapper and many local TypeScript DTO shadows.
- `app/src/api.rs`: `OriginClient`, `origin_types`, and daemon HTTP helper methods.
- `app/src/search.rs`: Tauri command implementations and many app-local request/response DTOs.
- `app/src/lib.rs`: sidecar spawn and command registration.
- `app/src/lifecycle.rs`: LaunchAgent labels, stable app path validation, sidecar install/uninstall.
- `app/src/remote_access.rs`: Origin relay URL, token/relay-id paths, port-only orphan cleanup.
- `app/src/mcp_config.rs` and `src/components/SetupWizard.tsx`: MCP key/package naming and compatibility bridge.
- `app/tauri.conf.json`, `app/capabilities/default.json`, `package.json`: public/runtime identity and sidecar names.

## Stop Conditions

- Do not start typed API edits until `scripts/refactor/inventory.sh` runs and its summary is committed or intentionally regenerated.
- Do not start a cross-cutting edit until CodeGraph is either synced and queried for the target symbol/file or explicitly marked unavailable for that edit.
- Do not start public identity rename until the sidecar contract has a tested `wenlan-server`/`wenlan-mcp` path and Origin bridge behavior.
- Do not delete old MCP config entries automatically; migration UI must detect and report both old and new keys.
