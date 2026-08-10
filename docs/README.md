# Documentation Guide

This directory contains project documentation intended for contributors and maintainers.

## Structure

- `assets/`: images used by the READMEs.
- `eval/`: evaluation method, receipts, and the README snapshot workflow.

`plans/` and `superpowers/` are gitignored local scratch space for working design docs. They are not part of the repository. Nothing under `docs/` is read by code or CI: the two specs that are live test fixtures live in `crates/wenlan-core/contracts/`, next to the tests that parse them.

## Reading a local plan file safely

If you have a local `plans/` or `superpowers/` directory, treat everything in it as a snapshot of the architecture at the time it was written, never as current guidance. Many are superseded by the daemon-centric layout (`crates/wenlan-types`, `crates/wenlan-core`, `crates/wenlan-server`, `crates/wenlan-cli`), and the Tauri desktop app that older plans describe as a separate repository is now the in-tree `app/` crate. Two of them moved to `crates/wenlan-core/contracts/` precisely because they stopped being plans and became things the build reads.

## Current sources of truth

- Repository overview and quickstart: `README.md`
- Retrieval, graph, and model details: `technical-foundations.md`
- Contributor workflow and CI commands: `CONTRIBUTING.md`
- Agent and developer conventions: `AGENTS.md` at the repo root (`CLAUDE.md` re-imports it)
- Test layers (what runs at L1-L8, where, when, whether it blocks): `test-layers.md`
- Platform code (per-OS data dirs, service registration, GPU backends): `cross-platform.md`
- Release operator runbook: `RELEASING.md` at the repo root
- AI-assisted install path used by the README: `setup-with-ai.md`
- Intermittent CI failure policy: `ci-flake-policy.md`
