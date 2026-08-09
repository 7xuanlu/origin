# Page authoring, Graph camera, and title hierarchy design

> **Authoring flow superseded (2026-07-17):** The approved direct Page draft
> editor replaces the modal composer and `attached_to` banner sections below.
> The fixed `34px` title and Graph camera decisions remain current.

## Approved outcome

- Repair the Graph pane so zero- and one-entity data never becomes a stale or full-canvas color field.
- Keep Wenlan's existing Fraunces, Instrument Sans, and JetBrains Mono families; change only the destination-title component token.
- Add direct authored Page creation to Wiki and Space detail without making Space mandatory.

## Fable convergence

Fable reviewed the verified decision brief and returned `REVISE`. The composer flow was approved unchanged. The accepted refinements are:

- Use one fixed `34px` top-level destination title for Home, Wiki, and Spaces. This stays visibly above the `30px` Space-detail object title at every supported desktop width and follows Wenlan's existing fixed token scale.
- When the daemon attaches authored content to an existing Page, show a dismissible inline banner on that Page naming the target Page. Do not rely on a transient toast.
- Define the zero-node Graph camera case explicitly and keep `nodeRelSize={1}` coupled to `nodeVal = radius²`.

## Title hierarchy

- Home, Wiki, and Spaces use Fraunces `34px / 500`, `line-height: 1.12`, `letter-spacing: -0.03em`.
- Home remains a semantic `h1`; it is not optically demoted below Wiki or Spaces.
- Home's title-to-stats gap is `16px` because it has no descriptive paragraph.
- Space detail remains Fraunces `clamp(24px, 1.6vw + 14px, 30px)`.

## Page composer

- Entry points use the same quiet outlined `New page` treatment as the existing Spaces header action.
- Wiki places the action at the right of its title/description header.
- Space detail places the action before the overflow button in its header action cluster.
- Both entry points open one shared centered dialog, maximum width `680px`.
- Fields are Title, Content, and Space. Title and Content are required. Content has a minimum height of `220px`.
- The blank Space option is `No space`; no `Optional` label is shown.
- Wiki defaults to no Space. Space detail defaults to the current Space and remains clearable.
- Title receives initial focus. `Cmd/Ctrl+Enter` creates the Page.
- Backdrop clicks never discard entered writing. Cancel or Escape asks for confirmation only when the form is dirty.
- During submission, the primary action reads `Creating…` and is disabled.
- Errors remain inline and preserve every field.
- Success navigates to `attached_to ?? id`. When `attached_to` is present, the destination Page shows a dismissible `Added to “<Page title>”` banner.

## Daemon contract

The desktop command posts to the existing `POST /api/pages` route with:

```text
title
content
summary: null
entity_id: null
space: string | null
source_memory_ids: []
creation_kind: "authored"
workspace: null
```

No backend schema or daemon route changes are in scope.

## Graph camera contract

- Keep one centralized camera-plan helper shared by resize and engine-settle paths.
- Zero nodes: reset to the default center and zoom; never inherit the previous graph transform.
- One node: `centerAt(node.x, node.y, 400)` then `zoom(3.5, 400)`; never call `zoomToFit`.
- Two or more nodes: `zoomToFit(400, 64)` with positive padding.
- Keep `nodeRelSize={1}` and return painted radius squared from `nodeVal` so force-graph bounds match custom glyph geometry.
- Repair the Review fixture so every declared entity has a detail row; add a parity test.

## Verification floor

- RED-first focused tests for API serialization, composer safety, both entry points, attached-content feedback, title CSS, Review fixture parity, and zero/one/many Graph camera plans.
- Full TypeScript tests, i18n parity, build, Rust tests/formatting, and `git diff --check`.
- Fresh rendered captures of Home, Graph, Wiki, Wiki composer, Space detail, and Space-detail composer from the fixture-isolated Review runtime.
- Exercise at least one actual Page-create interaction in Review and confirm navigation, preserved optional-Space semantics, console health, and non-orange Graph pixels.
