# Documentation Guide

This directory contains project documentation intended for contributors and maintainers.

## Structure

- `plans/`: historical implementation plans and design snapshots.

## Reading `plans/` safely

Many plan files reflect the architecture at the time they were authored. Some are now superseded by the current daemon-centric layout (`crates/wenlan-types`, `crates/wenlan-core`, `crates/wenlan-server`, `crates/wenlan-cli`). The Tauri desktop app referenced by older plans now lives in [7xuanlu/wenlan-app](https://github.com/7xuanlu/wenlan-app).

If a plan starts with a **Superseded** note, treat it as historical context rather than current implementation guidance.

## Current sources of truth

- Repository overview and quickstart: `README.md`
- Retrieval, graph, and model details: `technical-foundations.md`
- Contributor workflow and CI commands: `CONTRIBUTING.md`
- Agent and developer conventions: `AGENTS.md` at the repo root (`CLAUDE.md` re-imports it)
- Test layers — what runs at L1-L8, where, when, whether it blocks: `test-layers.md`
- Platform code — per-OS data dirs, service registration, GPU backends: `cross-platform.md`
