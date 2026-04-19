# Documentation Guide

This directory contains project documentation intended for contributors and maintainers.

## Structure

- `plans/`: historical implementation plans and design snapshots.

## Reading `plans/` safely

Many plan files reflect the architecture at the time they were authored. Some are now superseded by the current daemon-centric workspace split (`crates/origin-core`, `crates/origin-server`, `app/`, `src/`).

If a plan starts with a **Superseded** note, treat it as historical context rather than current implementation guidance.

## Current sources of truth

- Repository overview and quickstart: `README.md`
- Contributor workflow and CI commands: `CONTRIBUTING.md`
- Detailed developer architecture conventions: `CLAUDE.md`
