# Fixture-only Review flavor and Wiki/Space refinement

**Date:** 2026-07-16
**Status:** Approved for implementation by the user
**Scope:** a permanent native Review flavor, Wiki/Spaces title hierarchy, Wiki row semantics, representative fixture data, and the Memories navigation glyph.

> **Final supersession (2026-07-17):** Later user review selected the fixed
> `34px` destination-title token and the durable `Page | Entity` Wiki kinds.
> `Kind` is non-interactive metadata; the full row opens the Page and Space is
> the sole secondary row link. This replaces the earlier responsive title,
> heuristic Topic/Decision/Recap glyphs, and Entity-chip navigation below.

## Outcome

Wenlan gains a native `Wenlan Review.app` that is safe to leave open while reviewing new UI. It renders deterministic test fixtures and cannot attach to or mutate the production daemon or user data. In the same pass, Wiki and Space surfaces adopt the agreed title scale, Wiki rows expose durable Page/Entity kinds and contextual destinations, and Memories keeps a recognisable brain mark that is visually distinct from Wiki, Graph, Space, and Sources.

## Product decisions

### Review is an environment, not a badge over production

- Product name: `Wenlan Review`.
- Bundle identifier: `com.wenlan.desktop.review`.
- Every launch starts from `createSpacesNavigationFixture()`; fixture mutations are process-local and reset on relaunch.
- A persistent header proof-stamp reads `TEST DATA` and `Fixture data · resets on relaunch` (localized in English, Simplified Chinese, and Traditional Chinese).
- A separate reset control clears Review-only browser state and reloads the fixture.
- Production builds hard-code the Review flag to `false`; Review builds hard-code it to `true`. No exported runtime environment variable can turn the marker on accidentally.

The Review Rust entry point is compile-time selected and deliberately minimal. It does not install plugins, register production IPC commands, manage the daemon, repair launchd, spawn sidecars, start file watchers or source sync, open a tunnel, install a tray, register global shortcuts, open the MCP debug socket, or check for updates. The Review Tauri overlay additionally removes sidecars, updater configuration, tray configuration, and production capabilities.

This rejects two weaker alternatives:

- Runtime `if review` guards inside the production startup path are too porous: a newly added side effect can bypass the guard.
- A browser-only preview is safe but cannot validate the native bundle, title bar, WebView, or app identity.

### Title hierarchy

- Home, Wiki, and Spaces overview titles: fixed `34px`.
- Space detail title and rename input: `clamp(24px, 1.6vw + 14px, 30px)`, matching the existing Wiki Page and Entity detail scale.
- Fraunces, Instrument Sans, JetBrains Mono, and the current warm Wenlan tokens remain unchanged.

### Wiki rows and destinations

- Every item is a Page. `entity_id` is the only durable distinction currently available, so `Kind` exposes `Page` or `Entity`; prose is never guessed into Topic, Decision, or Recap schemas.
- The full row is one Page destination and opens the corresponding Wiki Page.
- `Kind` is quiet, non-interactive metadata.
- A Space name opens that Space detail, in both table and compact/mobile presentations.
- Pages with no Space render an empty Space field. There is no `Independent` label.

The Review fixture must visibly contain Entity and ordinary Page rows, varied content, and at least one Page without a Space so reviewers can verify the semantics.

### Memories mark

- Memories remains a brain; it does not revert to a document/list glyph.
- The geometry must come from a recognised icon system selected with the design lead, not an improvised path.
- The chosen view box, paths, 14 px rendered size, and stroke width are locked in a focused test.
- Graph, Wiki, Space, Sources, and the Home Page layered glyph remain unchanged.

## Acceptance checks

1. Focused tests fail before the implementation and pass afterward.
2. Production build and its identity/config remain unchanged except for a hard-coded `__WENLAN_REVIEW__ = false` definition.
3. The Review web build rejects unknown Tauri commands and never proxies to `127.0.0.1:7878`.
4. The native Review bundle contains no Wenlan daemon, MCP, cloudflared, or other sidecar executable.
5. While Review is open, it has no TCP connection to port 7878, no daemon child process, and does not modify production logs or data paths.
6. Fresh desktop and compact screenshots show the smaller title hierarchy, Page/Entity Wiki glyphs, working Page/Space destinations, correct Memories mark, and persistent test-data proof-stamp.
