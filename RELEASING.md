# Releasing Wenlan (daemon side)

This document covers releases of the local runtime: `wenlan` CLI, `wenlan-server` daemon, `wenlan-mcp` connector, and shared crates (`wenlan-types`, `wenlan-core`). The desktop app ships from [7xuanlu/wenlan-app](https://github.com/7xuanlu/wenlan-app) on its own release cadence.

## How release-please works

Merge conventional commits to `main` (e.g. `feat:`, `fix:`, `chore:`). The `release-please` workflow opens a "Release PR" automatically, bumping the version and updating `CHANGELOG.md`. Merge that PR to cut the release. Release-please then creates the git tag, which triggers the `release.yml` build workflow.

> The coding-time rules — why every release bumps patch (`"versioning": "always-bump-patch"`), how to force a deliberate minor via `release-as`, the "review the squash-merge PR title before merging" warning, the version-file-sync rule, and how to undo a release — live in the root [`AGENTS.md`](AGENTS.md) 'Releasing (release-please)' section so every agent has them in-context. This document is the human operator procedure.

The `.release-please-manifest.json` is the canonical version source; check the pending version with `cat .release-please-manifest.json`. The release-please workflow syncs Cargo manifests, npm package manifests, plugin metadata, and pinned install URLs from `version.txt`. It also syncs the daemon workspace `Cargo.toml` version on the release branch, because release-please can't handle Cargo workspaces reliably with the `simple` release type.

**Config files:**
- `release-please-config.json` — release type, version-bump behavior
- `.release-please-manifest.json` — current version
- `.github/workflows/release-please.yml` — creates/updates the release PR, syncs daemon Cargo.toml versions
- `.github/workflows/release.yml` — builds the daemon + uploads artifacts on `v*` tag push

## Manual override: bump-version.sh

If you need to cut a release without release-please (hotfix, first release, version correction):

```bash
bash scripts/bump-version.sh 0.2.0
```

This updates workspace Cargo versions, npm package manifests, plugin metadata, and pinned plugin URLs. Review the diff, stage the files, and push. Then create and push the tag manually:

```bash
git tag v0.2.0
git push origin v0.2.0
```

The `release.yml` workflow triggers on any `v*` tag push.

## Version consistency gate

The `release.yml` workflow validates that the pushed tag version matches `version.txt`, workspace Cargo, npm package manifests, and plugin metadata before building. If out of sync, the build fails with instructions to run `bump-version.sh`.

## What the release workflow does

The `v*` tag push triggers `.github/workflows/release.yml`. Its **first** job immediately demotes the freshly-created release to a **prerelease**, so `releases/latest` keeps resolving to the last good version while the build runs; only after every build + publish step below succeeds does the `finalize-release` job clear the prerelease flag.

1. Validates version consistency.
2. Builds `wenlan`, `wenlan-server`, and `wenlan-mcp` for `aarch64-apple-darwin`.
3. Smoke-tests `wenlan --help` and `wenlan-server --help`.
4. Creates the GitHub release with standalone binaries attached.
5. Publishes `wenlan-types` and `wenlan-mcp` to crates.io.
6. Publishes `wenlan-mcp` and `wenlan` to npm.
7. Updates the Homebrew tap for `wenlan-mcp`.

`wenlan-mcp` now lives in this monorepo under `crates/wenlan-mcp` and shares the workspace Apache-2.0 license. The desktop DMG is still built from [wenlan-app](https://github.com/7xuanlu/wenlan-app); see its `RELEASING.md` for that pipeline.

Nothing is notified when the prerelease flag clears: the Claude Code plugin ships from this repo's own `.claude-plugin/marketplace.json`, which sources `plugin/` by `git-subdir` with no `ref` pin, so it tracks the default branch and has no release-time pin to sync.

## Required secrets

Configure these in the repository settings (Settings, Secrets and variables, Actions):

| Secret | Purpose |
| ------ | ------- |
| `CARGO_REGISTRY_TOKEN` | Publish `wenlan-types` to crates.io. Create at crates.io under Account Settings, API Tokens. |
| `RELEASE_TOKEN` | Fine-grained PAT (contents:write) used by release-please-action so its push triggers the next workflow run. GITHUB_TOKEN-driven pushes never fire downstream workflows. |
| `GITHUB_TOKEN` | Built-in. Used for GitHub release creation and release-please PR management. No setup needed. |
