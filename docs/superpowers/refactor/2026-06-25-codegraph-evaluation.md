# CodeGraph Evaluation for Wenlan App Refactor

- **Date:** 2026-06-25
- **Package:** `@colbymchenry/codegraph` 1.1.1
- **Source:** <https://github.com/colbymchenry/codegraph>
- **Worktree:** `/Users/lucian/Repos/wenlan-app/.worktrees/wenlan-app-convergence`
- **Purpose:** validate whether CodeGraph improves token efficiency and refactor accuracy before the large `origin-app` to `wenlan-app` migration.
- **Current status:** this evaluation started before the `WenlanClient` rename. For new probes, query `WenlanClient`, `wenlan_types`, sidecar symbols, and runtime identity files. The historical `OriginClient`/`origin_types` results remain useful only as pre-migration evidence.

## Decision

Use CodeGraph as a project-local navigation and blast-radius tool for the convergence refactor.

Do not require a global install or MCP install yet. In particular, do not run `codegraph install` during this prereq phase because it mutates shared agent configuration. Use the `npx` CLI with telemetry disabled and keep `.codegraph/` ignored as local cache.

Keep `ast-grep` as the deterministic inventory gate, and keep `rust-analyzer`, `tsserver`, builds, and tests as the correctness authorities. CodeGraph is useful, but it is not a proof oracle.

## Commands Tested

```bash
CODEGRAPH_TELEMETRY=0 DO_NOT_TRACK=1 npx -y @colbymchenry/codegraph version
CODEGRAPH_TELEMETRY=0 DO_NOT_TRACK=1 npx -y @colbymchenry/codegraph init /Users/lucian/Repos/wenlan-app/.worktrees/wenlan-app-convergence
CODEGRAPH_TELEMETRY=0 DO_NOT_TRACK=1 npx -y @colbymchenry/codegraph status /Users/lucian/Repos/wenlan-app/.worktrees/wenlan-app-convergence
CODEGRAPH_TELEMETRY=0 DO_NOT_TRACK=1 npx -y @colbymchenry/codegraph query OriginClient --json
CODEGRAPH_TELEMETRY=0 DO_NOT_TRACK=1 npx -y @colbymchenry/codegraph query origin_types --json
CODEGRAPH_TELEMETRY=0 DO_NOT_TRACK=1 npx -y @colbymchenry/codegraph callers OriginClient --json
CODEGRAPH_TELEMETRY=0 DO_NOT_TRACK=1 npx -y @colbymchenry/codegraph impact OriginClient --json
CODEGRAPH_TELEMETRY=0 DO_NOT_TRACK=1 npx -y @colbymchenry/codegraph affected app/src/api.rs --json
CODEGRAPH_TELEMETRY=0 DO_NOT_TRACK=1 npx -y @colbymchenry/codegraph explore "OriginClient daemon API boundary and pending revision commands"
```

## Observed Results

Indexing completed cleanly:

```text
Indexed 153 files
2,002 nodes, 4,814 edges in 910ms
DB Size: 6.24 MB
Backend: node:sqlite - built-in (full WAL)
```

Language split:

| Language | Files |
|---|---:|
| tsx | 95 |
| rust | 32 |
| typescript | 21 |
| yaml | 3 |
| xml | 2 |

The `OriginClient` query returned the struct plus method line ranges and signatures in `app/src/api.rs`, including `OriginClient::new`, `OriginClient::url`, `OriginClient::health`, `OriginClient::get_config`, and config helpers.

The `origin_types` query identified Rust import sites across `app/src/search.rs`, `app/src/api.rs`, `app/src/events.rs`, `app/src/router/intent.rs`, `app/src/sources/*`, `app/src/state.rs`, and `app/src/system_info.rs`. This is a useful seed list for the `origin-types` to `wenlan-types` contract migration.

The `callers OriginClient` query found `app/src/state.rs` as the direct caller.

The `impact OriginClient` query returned a 140-node, 153-edge blast radius covering the HTTP client, many `app/src/search.rs` Tauri commands, `app/src/lib.rs`, `app/src/router/intent.rs`, and source sync paths. It is useful for refactor orientation, but broad enough that follow-up queries should target concrete methods or files.

The `affected app/src/api.rs` query reported:

```json
{
  "changedFiles": ["app/src/api.rs"],
  "affectedTests": ["src/components/memory/RemoteAccessPanel.test.tsx"],
  "totalDependentsTraversed": 18
}
```

The natural-language `explore` query surfaced the daemon API boundary and pending revision bridge across `app/src/api.rs`, `app/src/search.rs`, and `src/lib/tauri.ts` without opening the full files manually.

## Caveats

- `.codegraph/` is generated local state and must stay ignored.
- `impact OriginClient` can become too broad for a large client facade. Prefer method-specific or file-specific queries after the first orientation pass.
- Test-impact hints are advisory only. The `explore` output flagged some symbols as having no covering tests, while `affected app/src/api.rs` still returned one dependent test. Real `pnpm test`, `cargo test`, and targeted test additions remain required.
- CodeGraph's natural-language output can include large source excerpts. Use specific queries first; use `explore` only when a compact multi-file orientation is worth the output.

## Refactor Workflow Placement

Before a cross-cutting typed-client, sidecar, MCP, or runtime-identity edit:

1. Run `codegraph sync .`.
2. Run `codegraph query <symbol> --json`.
3. Run `codegraph impact <symbol> --json` or `codegraph affected <file> --json`.
4. Use `ast-grep` to generate deterministic inventories and residual checks.
5. Verify with LSP/build/test output before treating the change as safe.
6. Use `rg` for residual text checks after the graph and structural surfaces are known.
7. Use bounded `grep` only if CodeGraph, ast-grep, LSP/compiler, tests/builds, and `rg` are unavailable or unsuitable for the current surface.

Initial high-value queries for Phase A/B:

- `OriginClient`
- `origin_types`
- `PendingRevision`
- `ConfigResponse`
- `UpdateConfigRequest`
- `setup_mcp`
- `remote_access`
- `spawn_sidecar`

## Task 4 Execution Note

During Task 4, the required CodeGraph sync command was attempted:

```bash
CODEGRAPH_TELEMETRY=0 DO_NOT_TRACK=1 npx -y @colbymchenry/codegraph sync .
```

The sandbox reviewer blocked the command because it was considered an external
untrusted network action with possible repository code/metadata exfiltration.
Per the task fallback decision, CodeGraph was not retried and no further
CodeGraph escalation was requested.

Task 4 continued with the approved fallback ladder:

- ast-grep structural surfaces
- `env TAURI_CONFIG='{"bundle":{"externalBin":[]}}' RUSTC_WRAPPER= cargo check`
- `pnpm build`
- residual `rg`
- `git diff --check`

## Task 5 Execution Note

Task 5 attempted the requested CodeGraph sidecar/lifecycle probes:

```bash
CODEGRAPH_TELEMETRY=0 DO_NOT_TRACK=1 npx -y @colbymchenry/codegraph sync .
CODEGRAPH_TELEMETRY=0 DO_NOT_TRACK=1 npx -y @colbymchenry/codegraph query spawn_sidecar --json
```

The `sync` command produced no output for 40 seconds and was interrupted. The
first `query` command also produced no output for 10 seconds and was
interrupted. The remaining CodeGraph commands were intentionally skipped
because this matched the same unavailable/sandboxed CodeGraph path captured in
Task 4, and the Task 5 instructions said not to request escalation for that
case.

The ast-grep `npx` fallback commands were also attempted:

```bash
npx -y -p @ast-grep/cli sg outline app/src/lib.rs
npx -y -p @ast-grep/cli sg outline app/src/lifecycle.rs
```

Both commands produced no output for 30 seconds and were interrupted. Task 5
therefore used the deterministic fallback surface:

- residual `rg` over sidecar/config/lifecycle strings
- `cargo build` before and after sidecar-name edits
- no-sidecar `TAURI_CONFIG` semantic `cargo check`
- focused lifecycle tests for stable app target validation
- JSON parse checks for Tauri/package/capability config
- `git diff --check`

Task 5 follow-up added `scripts/prepare-sidecars.sh` so raw Tauri builds can
prepare generated, ignored `app/binaries/*-$TRIPLE` sidecars before running
`cargo build`.

## MCP Entry Typing Checkpoint Note

During the MCP entry typing checkpoint, `codegraph sync .` exited successfully
and reported "Already up to date", but `codegraph query wenlan_mcp_entry --json`
still returned the pre-edit Rust signatures:

- `wenlan_mcp_entry() -> serde_json::Value`
- `get_wenlan_mcp_entry() -> Result<serde_json::Value, String>`

The actual source, ast-grep, Rust compiler, and tests all saw the updated typed
surface (`WenlanMcpEntry`). Treat this as another example of CodeGraph being an
orientation tool, not a proof gate. For type/signature proof, prefer ast-grep
for syntax and `cargo test`/`cargo clippy` for compiler authority.
