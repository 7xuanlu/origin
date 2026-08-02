# Releasing Wenlan (daemon side)

This document covers releases of the local runtime: `wenlan` CLI, `wenlan-server` daemon, `wenlan-mcp` connector, and shared crates (`wenlan-types`, `wenlan-core`). The desktop app ships from [7xuanlu/wenlan-app](https://github.com/7xuanlu/wenlan-app) on its own release cadence.

## How release-please works

Merge conventional commits to `main` (for example, `feat:`, `fix:`, or `chore:`). The `release-please` workflow opens a Release PR automatically, bumps the version, and updates `CHANGELOG.md`. Its ordinary-main path sets `skip-github-release: true`: it maintains the PR but cannot create a tag or GitHub Release. Merging the Release PR cuts a release only after the trusted promotion gate binds that exact merge tree to the archives already built and tested by the PR. The gate then creates the exact `v*` tag, which triggers artifact-only publication in `release.yml`.

> The coding-time rules—why every release bumps patch (`"versioning": "always-bump-patch"`), how to force a deliberate minor via `release-as`, the version-file-sync rule, and how to undo a release—live in the root [`AGENTS.md`](AGENTS.md) "Releasing (release-please)" section. This document is the human operator procedure.

The `.release-please-manifest.json` is the canonical version source; check the pending version with `cat .release-please-manifest.json`. The release-please workflow syncs Cargo manifests, npm package manifests, plugin metadata, and pinned install URLs from `version.txt`. It also syncs the workspace `Cargo.toml` version on the release branch because release-please cannot reliably handle Cargo workspaces with the `simple` release type.

Config files:

- `release-please-config.json` — release type and version-bump behavior
- `.release-please-manifest.json` — current version
- `.github/workflows/release-please.yml` — maintains the Release PR and creates a tag only for a receipt-validated Release PR merge
- `.github/workflows/release.yml` — consumes exact validated archives and publishes them on the receipt-derived `v*` tag; it does not recompile Rust

## Manual version override

For a hotfix, first release, or version correction, use the version script on a branch:

```bash
bash scripts/bump-version.sh 0.2.0
```

This updates workspace Cargo versions, npm package manifests, plugin metadata, and pinned plugin URLs. Review the diff and send it through the normal Release PR CI, observer, merge, and promotion path. Do not manually push a release tag: a tag without the exact main-promotion receipt has no validated archive identity and `release.yml` fails closed. To recover from a transient failure, rerun the original failed workflow attempt; do not manufacture a replacement tag or select an artifact named "latest".

## Version consistency gate

Release PR CI validates that the proposed version is synchronized across `version.txt`, workspace Cargo, npm package manifests, and plugin metadata before producing archives. The observer independently verifies the strict release-managed diff and version policy. The tag workflow rechecks the receipt-derived tag, version, commit, and asset inventory before publishing; it never repairs drift or rebuilds different bytes.

## Validated release-candidate promotion

The exact same-repository `release-please--branches--main` PR builds and smoke-tests the six canonical final archives once during `release-preflight`. Those hostile-PR outputs are named by source run, producing attempt, and target. A separate default-branch `workflow_run` observer checks out trusted default-branch code, has read-only permissions, treats every downloaded byte as untrusted data, and verifies the upstream CI identity through the Actions API. GitHub can rerun only failed or selected jobs, so a successful terminal run may legitimately contain target artifacts from different attempts; the attempt jobs API also re-emits inherited jobs with the new attempt number. The observer therefore indexes the attempt-scoped artifacts first, selects each target's newest available artifact, then requires the exact matrix job on that artifact's attempt to have completed successfully and requires its embedded manifest attempt to agree. The terminal source run must itself be successful, so a genuinely failed rerun cannot fall back to an older artifact. It then reconciles the Release PR and strict version-only diff, checks the exact four-wrapper/six-asset inventory, hashes every layer, and safely inspects each archive without executing it. To tolerate the merge/observer scheduling race, it accepts either the still-open Release PR or a merged Release PR whose merge tree is byte-for-byte the candidate head tree; a closed-unmerged PR or different tree fails.

A passing observer uploads the exact six inspected files as `validated-release-assets-{source-run}-{source-attempt}-{observer-run}-{observer-attempt}` and closes a receipt over the upload action's artifact ID and SHA-256 digest. Both observer outputs are retained for 30 days. The receipt artifact name, `validated-release-receipt-{source-run}-{source-attempt}`, is deliberately only a locator. On an observer rerun, `upload-artifact` replaces this source-only locator with a new immutable artifact ID because GitHub requires artifact names to be unique within one workflow run. Consumers still distrust the name: they require the exact active observer workflow identity and newest terminal attempt, then validate the closed schema, artifact ID/digest, and source/observer identities. Semantically conflicting evidence fails the release instead of letting the consumer choose whichever one passes.

On the Release PR merge, main CI waits up to 12 minutes for this small closed receipt. Ordinary commits without a Release PR stay on the normal CI path. A release-like merge with missing, invalid, expired, or conflicting evidence fails closed; it cannot silently fall back to a full rebuild or ordinary release-please behavior. For valid evidence, the successful `detect-changes` execution emits `main-release-promotion-receipt-{main-run}-{producer-attempt}`, bound to the exact main SHA/tree, source CI run/attempt, observer run/attempt, validated-assets artifact ID/digest, PR, version, and tag. A failed-jobs-only rerun may leave that producer in an earlier attempt; the later `release-please` workflow therefore validates the terminal successful main run, locates the receipt's claimed producer attempt, proves that exact `detect-changes` job succeeded in that attempt, and freshly revalidates the receipt before it creates the exact tag. Future, failed-producer, or semantically conflicting receipts fail closed.

The closed receipt is an internal chain-of-custody record, not a Sigstore attestation or third-party provenance statement. Its trust comes from the default-branch workflow code, exact GitHub workflow/run identities, immutable artifact IDs, and verified SHA-256 digests. This design follows GitHub's warning that `workflow_run` can receive secrets and write tokens even when its source run cannot, so the observer is kept read-only and never executes candidate bytes.

Official behavior relied on by this design:

- [GitHub `workflow_run` security and default-branch behavior](https://docs.github.com/en/actions/reference/workflows-and-actions/events-that-trigger-workflows#workflow_run)
- [GitHub rerunning failed or selected jobs and run-attempt semantics](https://docs.github.com/en/actions/how-tos/manage-workflow-runs/re-run-workflows-and-jobs)
- [GitHub REST API for the jobs in one workflow-run attempt](https://docs.github.com/en/rest/actions/workflow-runs?apiVersion=2026-03-10#get-jobs-for-a-workflow-run-attempt)
- [GitHub Actions artifact REST metadata and exact-name query](https://docs.github.com/en/rest/actions/artifacts?apiVersion=2026-03-10)
- [GitHub `push` event commit SHA semantics](https://docs.github.com/en/actions/reference/workflows-and-actions/events-that-trigger-workflows#push)
- [GitHub REST API for reading an exact tag ref](https://docs.github.com/en/rest/git/refs?apiVersion=2026-03-10#get-a-reference)
- [`upload-artifact` immutability, replacement semantics, artifact ID/digest outputs, and retention](https://github.com/actions/upload-artifact)
- [`upload-artifact` same-run artifact-name uniqueness](https://github.com/actions/upload-artifact#not-uploading-to-the-same-artifact)
- [Release Please's `skip-github-release` split between PR creation and tagging](https://github.com/googleapis/release-please-action#configuration)
- [Cargo publish packaging, verification, upload, and `--dry-run` semantics](https://doc.rust-lang.org/cargo/commands/cargo-publish.html)

## What the release workflow does

The receipt-derived `v*` tag triggers `.github/workflows/release.yml`. Every source checkout is pinned to the tag-push event's immutable `${{ github.sha }}`, never to the mutable tag name, and every publication boundary re-reads the live tag ref and requires it still to equal that event SHA. `resolve-promotion` then locates the exact main-promotion receipt for the event commit and revalidates its owning successful main CI run. `promote-assets` downloads the one receipt-bound `validated-release-assets-*` artifact by exact ID and digest, safely extracts its six archives, and verifies every archive size and SHA-256 from the closed receipt. There is no `cargo build`, toolchain setup, cache restore, or source checkout that could produce different release binaries. Short-lived artifacts used only inside this trusted tag workflow (`release-promotion-plan-{run}`, Homebrew inputs, Docker inputs, and per-architecture digest receipts) deliberately use stable run-scoped names with `overwrite: true`: a full rerun replaces the old immutable artifact with a new ID, while a failed-jobs-only rerun can reuse a successful producer. These names are retry locators, not trust anchors; the run's concurrency is serialized and downstream steps still strictly parse the plan, verify the closed receipt and archive hashes, or inspect the registry digests before public promotion.

`prepare-release` immediately places a new GitHub Release behind a prerelease gate, or refreshes an existing prerelease during a retry, and checks that `7xuanlu/homebrew-tap` is anonymously readable and its credential exists. An existing stable release is never incrementally filled: the workflow fails closed, because a partial public release cannot be made safe by uploading whichever assets happen to be missing. Recover a failed publication by rerunning its failed jobs while the release remains a prerelease. Package publishing and per-architecture runtime-image work can then fan out from the same verified archive directory. The Docker image copies the already validated `wenlan-server` from the matching Linux archive into a digest-pinned distroless runtime base; it does not compile or strip it. Before registry login/push, each architecture is loaded locally, checked for exact binary hash, numeric non-root user, environment and entrypoint, writable `/data`, architecture, and successful health/store/search behavior. Public GHCR manifests and `latest` remain behind the full publication barrier. Only after all channels succeed does `finalize-release` promote the GitHub prerelease and transition the Release PR label by adding `autorelease: tagged` before removing `autorelease: pending`.

1. Resolves and validates the exact main-promotion and closed candidate receipts.
2. Downloads the receipt-bound validated assets once and verifies all six archive digests and sizes.
3. Creates or refreshes the gated GitHub prerelease and release notes; an existing stable release fails closed.
4. Uploads the already tested standalone archives to the gated GitHub prerelease.
5. Publishes `wenlan-types` and `wenlan-mcp` to crates.io. Each real `cargo publish` retains Cargo's package-build verification; there is no duplicate preflight `--dry-run` compilation. A missing registry token fails unless the exact version is already present, and both packages must be visible in the sparse index before this channel succeeds.
6. Publishes `wenlan-mcp` and `wenlan` to npm.
7. Updates the public Homebrew tap for `wenlan` and `wenlan-mcp`, then anonymously taps, installs, and runs both formula tests on macOS arm64.
8. Builds runtime-only GHCR images from the exact validated Linux server binaries, proves binary equality and runtime behavior, then publishes the multi-architecture version manifest and, for stable tags only, moves `latest`.
9. Promotes the GitHub prerelease and moves the Release PR lifecycle from pending to tagged.

The complete four-target Rust compilation and archive smoke happen only once, in Release PR CI. An ordinary main run retains only the Windows release-profile build so it can warm the one expensive target cache that CI persists; a validated Release PR merge downloads only a small receipt. The tag workflow downloads the validated asset bundle once rather than compiling four targets again. The expected release critical path is therefore external registry publication, Homebrew installation, and runtime-image smoke—not duplicate Cargo work. If release time regresses, inspect those job durations separately before changing test coverage or weakening artifact validation.

Release workflow changes have static mutation-tested contracts in `scripts/release-workflow-contract.test.py`, but external registries and a revised job DAG are proved end-to-end only by the next real tag. Do not cite an older successful tag as evidence for a newer workflow topology.

`wenlan-mcp` lives in this monorepo under `crates/wenlan-mcp` and shares the workspace Apache-2.0 license. The desktop DMG is still built from [wenlan-app](https://github.com/7xuanlu/wenlan-app); see its `RELEASING.md` for that pipeline.

Nothing is notified when the prerelease flag clears: the Claude Code plugin ships from this repo's own `.claude-plugin/marketplace.json`, which sources `plugin/` by `git-subdir` with no `ref` pin, so it tracks the default branch and has no release-time pin to sync.

## Required secrets

Configure these in repository Settings → Secrets and variables → Actions:

| Secret | Purpose |
| ------ | ------- |
| `CARGO_REGISTRY_TOKEN` | Publishes `wenlan-types` and `wenlan-mcp` to crates.io. Create it under crates.io Account Settings → API Tokens. |
| `RELEASE_TOKEN` | Fine-grained PAT with contents:write and pull-requests:write. It updates the Release PR and creates the validated tag; unlike `GITHUB_TOKEN`, its pushes trigger downstream workflows. |
| `HOMEBREW_TAP_TOKEN` | Fine-grained PAT with contents:write on the public `7xuanlu/homebrew-tap` repository. The workflow verifies anonymous clone/install separately. |
| `GITHUB_TOKEN` | Built in. Reads workflow and artifact identity, uploads GitHub release assets, and publishes GHCR. No setup is needed. |

Before merging a Release PR, verify the tap is public without credentials:

```bash
GIT_TERMINAL_PROMPT=0 git ls-remote https://github.com/7xuanlu/homebrew-tap.git HEAD
```

If this fails or prompts for credentials, make the tap public before cutting the tag. The release preflight stops while the GitHub release is still a prerelease and before crates.io, npm, or GHCR promotion.
