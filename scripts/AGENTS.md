# AGENTS.md - scripts/

## OVERVIEW

Release, sidecar, and repo-inventory contracts. These scripts are
part of packaging behavior, not generic local helpers.

## WHERE TO LOOK

| Task | Location | Notes |
| --- | --- | --- |
| Stage sidecars | `prepare-sidecars.sh` | tree-build only; compiles from the checked-out backend |
| Tauri build hook | `prepare-tauri-build-sidecars.sh` | picks debug vs release based on `TAURI_ENV_DEBUG` |
| Resolve backend checkout | `resolve-backend-dir.sh` | validates sibling or `WENLAN_BACKEND_DIR` shape |
| Isolated dev runtime | `dev-runtime.sh`, `dev-all.sh` | worktree-owned daemon/UI ports, data dir, debug MCP socket, PID, and teardown |
| Version lockstep | `release-version-sync.test.ts` | app, Cargo, Tauri versions must match |
| Sidecar tests | `prepare-sidecars.test.ts` | locks path and cloudflared behavior |
| API route inventory | `refactor/api-route-diff.mjs` | route coverage signal, not a product requirement |

## CONVENTIONS

- Sidecars always come from a backend checkout in the same tree, found by
  `resolve-backend-dir.sh` (sibling checkout or `WENLAN_BACKEND_DIR`). The old
  pinned-download mode (`.wenlan-backend-version`, `prepare-sidecars.sh
  --download`) was deleted once the unified release (v0.15.7) proved the
  in-tree build.
- `prepare-tauri-build-sidecars.sh` is the Tauri hook; keep it aligned with
  `app/tauri.conf.json` `beforeBuildCommand`.
- `cloudflared` is required for a full Tauri bundle:
  `binaries/cloudflared-$TRIPLE`.
- Update scripts, tests, and workflows together when release or sidecar behavior
  changes. The workflow comments are part of the operational contract.

## ANTI-PATTERNS

- Do not let CI placeholder binaries become a release substitute.
- Do not make `resolve-backend-dir.sh` silently accept a directory that lacks
  `crates/wenlan-server`, `crates/wenlan-mcp`, and `crates/wenlan-cli`.

## COMMANDS

```bash
bash -n scripts/prepare-sidecars.sh
bash -n scripts/prepare-tauri-build-sidecars.sh
bash -n scripts/resolve-backend-dir.sh
bash -n scripts/dev-runtime.sh
bash -n scripts/dev-all.sh
bash scripts/prepare-sidecars.sh --print-paths
pnpm vitest run scripts/prepare-sidecars.test.ts scripts/release-version-sync.test.ts scripts/dev-runtime.test.ts
```
