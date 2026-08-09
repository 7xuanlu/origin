# AGENTS.md - scripts/

## OVERVIEW

Release, backend-pin, sidecar, and repo-inventory contracts. These scripts are
part of packaging behavior, not generic local helpers.

## WHERE TO LOOK

| Task | Location | Notes |
| --- | --- | --- |
| Stage sidecars | `prepare-sidecars.sh` | local build and pinned download modes |
| Tauri build hook | `prepare-tauri-build-sidecars.sh` | `WENLAN_DOWNLOAD_SIDECARS=1` switches to download |
| Resolve backend checkout | `resolve-backend-dir.sh` | validates sibling or `WENLAN_BACKEND_DIR` shape |
| Version lockstep | `release-version-sync.test.ts` | app, Cargo, Tauri, daemon pin must match |
| Sidecar tests | `prepare-sidecars.test.ts` | locks path, cloudflared, download, checksum behavior |
| API route inventory | `refactor/api-route-diff.mjs` | route coverage signal, not a product requirement |

## CONVENTIONS

- `.wenlan-backend-version` format is exact: line 1 is the daemon release tag,
  line 2 is the sha256 for `wenlan-darwin-arm64.tar.gz`.
- Local/dev sidecars come from a sibling backend checkout found by
  `resolve-backend-dir.sh`; release/download sidecars come from the pinned
  public `7xuanlu/wenlan` asset.
- `prepare-tauri-build-sidecars.sh` is the Tauri hook; keep it aligned with
  `app/tauri.conf.json` `beforeBuildCommand`.
- `cloudflared` is optional only for download smoke paths. Full Tauri bundles
  need `binaries/cloudflared-$TRIPLE`.
- Update scripts, tests, and workflows together when release or sidecar behavior
  changes. The workflow comments are part of the operational contract.

## ANTI-PATTERNS

- Do not remove sha256 verification, archive extraction checks, or `--help`
  smoke semantics from backend-pin/download flows.
- Do not let CI placeholder binaries become a release substitute.
- Do not make `resolve-backend-dir.sh` silently accept a directory that lacks
  `crates/wenlan-server`, `crates/wenlan-mcp`, and `crates/wenlan-cli`.
- Do not edit `.wenlan-backend-version` as casual metadata; it drives release
  validation and sidecar downloads.

## COMMANDS

```bash
bash -n scripts/prepare-sidecars.sh
bash -n scripts/prepare-tauri-build-sidecars.sh
bash -n scripts/resolve-backend-dir.sh
bash scripts/prepare-sidecars.sh --print-paths
pnpm vitest run scripts/prepare-sidecars.test.ts scripts/release-version-sync.test.ts
```
