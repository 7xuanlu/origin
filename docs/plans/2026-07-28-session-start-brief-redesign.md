# Session-start brief redesign

**Status:** Approved product contract
**Scope:** Wenlan daemon, MCP, CLI/plugin skills, and legacy status migration
**Explicitly out of scope:** wenlan-app UI

## Summary

Replace the current prompt-maintained project status file and the synthetic
`"recent context"` semantic query with one Space-owned, daemon-authoritative
Brief.

Wenlan has one scope concept: **Space**. A repo, worktree, or explicit client
setting resolves to a Space; there is no separate Project entity. The Brief is
the current work state for that Space, while `recall` remains the semantic
knowledge lookup for the same Space.

The user-facing contract is:

- `brief()` returns the current Space's Brief and performs no semantic search.
- `brief(topic)` returns the complete Brief plus a separately labelled
  Related Context result produced by `recall(topic)` in the same Space.
- `handoff` updates the Brief automatically and returns a concise mutation
  receipt.
- `context` remains a temporary compatibility adapter. It must stop issuing the
  fake `"recent context"` query and is no longer the recommended session-start
  tool.

Calling `brief` at session start is optional. Agents use it when continuation
state is useful; it is not a mandatory ritual for every new conversation.

## Product model

### Space is the project boundary

Space owns both durable knowledge and current work state:

```text
Space
├── Memories / Pages / Entities
└── Brief
    ├── Last session summary
    ├── Active items
    └── Backlog items
```

There is no second Project table or Project-to-Space synchronization layer.

Resolution keeps the existing precedence model: strict process pin, explicit
argument, configured default, cwd mapping, then a registered repo basename.
The redesign adds these guarantees:

- Worktrees of the same Git repository resolve to the same default Space.
- Multiple repositories may be explicitly mapped to one Space.
- A repo basename is a convenient default, not a globally unique identity.
- A read never creates a Space. The first successful handoff may create the
  resolved basename Space when no explicit/default/mapped Space exists.

### Brief is operational state, not memory

Brief items answer "what still needs attention?" They are not memories and do
not participate in semantic retrieval.

Minimum daemon model:

```text
Brief {
  space_id
  last_session_summary
  last_handoff_at
  version
}

BriefItem {
  id
  space_id
  text
  state: active | backlog
  added_at
  gate?
  version
}
```

`space_id` references the existing stable Space identity so a Space rename does
not orphan its Brief.

Deliberately absent from the first version:

- `done` rows: completed work leaves the Brief and remains in session history.
- workstream/worktree/branch fields.
- priorities, dependencies, assignees, or due dates.
- automatic closure, aging, or Active-to-Backlog demotion.
- fuzzy daemon-side matching of item text.

Stable item IDs and versions are machine-facing. Normal Markdown and app
surfaces do not display them.

## Read contract

### `brief()`

Returns structured fields for:

- resolved Space
- latest session summary and handoff time
- all Active items
- all Backlog items
- inline gate information

It does not load recent memories, decisions, agent activity, identity, or
preferences. In particular, a no-topic brief must not invoke semantic search.

If the Space exists without a Brief, the result states that no Brief exists.
If no Space resolves, the result asks the caller to specify one. Neither case
creates data.

### `brief(topic)`

Returns the same complete Brief plus:

```text
related_context {
  query
  results
}
```

Related Context uses the same resolved Space as the Brief. It is visibly
separate retrieval evidence and never filters, rewrites, or hides Brief items.

This is a convenience composition, not a second retrieval implementation:
`brief(topic) = brief() + recall(topic, same_space)`.

### Legacy `context`

During the compatibility window:

- `context()` adapts the new no-topic brief to the legacy response shape.
- `context(topic)` adapts the new topic brief to the legacy response shape.
- the literal `"recent context"` is never sent to semantic search.
- plugin documentation and session-start instructions stop recommending
  `context`.

Removal of the legacy route/tool is a later versioned change after CLI and
external callers have migrated.

## Write contract

### Automatic handoff

Handoff reads the current Brief first, then submits item-level deltas:

- add a new item
- edit an existing item by stable ID
- move an item between Active and Backlog by stable ID
- set or clear its gate by stable ID
- complete an item by stable ID
- update the latest session summary

The agent infers these deltas from the session and applies them without asking
for routine confirmation. If it cannot identify an existing item confidently,
it leaves that item unchanged rather than closing or merging it by text.

The daemon never fuzzy-matches item text. New items receive IDs from the
daemon; mutations of existing items require their IDs.

### Concurrency

Item versions prevent silent last-writer-wins behavior:

- mutations to different items merge normally
- simultaneous edits to the same item compare the expected item version
- only the conflicting mutation is rejected
- all unrelated safe mutations still commit
- the receipt names the conflicted item and leaves its stored value unchanged

The full handoff does not fail merely because one item conflicts.

### Receipt projection

After the brief update transaction commits, Wenlan writes:

```text
~/.wenlan/sessions/_status/<space>.md
```

This file is a human-readable handoff receipt:

- it contains the latest session, Active, and Backlog sections
- it carries a generated-file header and timestamp
- the handoff result provides a clickable path
- it is never parsed during normal `brief` or handoff operation
- projection failure produces a warning but does not roll back committed state

The existing narrative session log remains the historical record. The existing
best-effort `~/.wenlan` Git commit keeps the user's inspectable timeline.

## Migration

On upgrade, import each legacy status Markdown file once when its
stored Brief does not yet exist:

- `## Active` bullets become Active items
- `## Backlog` bullets become Backlog items
- `(added YYYY-MM-DD)` is preserved
- `(gated: ...)` is preserved as the gate field
- `## Last session` becomes `last_session_summary`, not an item
- every imported item receives a stable ID

The importer recognizes only the existing Wenlan-generated shape. A malformed
bullet is preserved as plain item text rather than dropped. The original file
remains available and is replaced by generated receipts only after a successful
handoff.

Import is idempotent and must never overwrite an existing stored Brief.

## Surface changes

### Included

- daemon storage and read/write routes
- shared request/response contracts
- MCP `brief` read surface and a daemon-backed CLI update surface used by
  handoff
- CLI/plugin skill migration for Claude and Codex
- legacy `context` adapter and caller migration
- one-time status Markdown importer
- generated receipt projection

### Excluded

- any wenlan-app route, screen, component, navigation item, or settings UI
- a Brief viewer or editor in wenlan-app
- cross-Space brief search
- a general task manager
- a workstream hierarchy

MCP-only clients are sufficient justification for daemon ownership. A desktop
UI requires a separate demonstrated user job and separate design.

## Acceptance contract

The implementation must leave executable evidence for these behaviors:

1. A no-topic brief returns Brief data without invoking semantic search.
2. A topic brief returns the same complete Brief and a separate
   same-Space recall result.
3. A brief with no resolved Space does not create a Space or Brief.
4. The first handoff may create the basename Space and its Brief.
5. Worktrees of one repository resolve to one Space.
6. Explicit mappings can route multiple repositories to one Space.
7. Legacy Active/Backlog/date/gate values survive the one-time import.
8. A second import cannot overwrite stored Brief state.
9. Concurrent non-overlapping item mutations merge.
10. A stale same-item mutation is isolated and cannot overwrite the newer item.
11. Completing an item removes it from Brief reads while session history
    retains the completion record.
12. Receipt write failure cannot undo the committed Brief mutation.
13. Legacy no-topic `context` never performs a `"recent context"` search.
14. Claude and Codex skills use the same Brief contract.

## Alternatives rejected

### Keep Markdown authoritative

Rejected because MCP-only clients cannot rely on direct filesystem access, and
whole-file rewrites provide no safe item-level concurrency.

### Add a separate Project entity

Rejected because the product already uses Space as the project-like context.
A second scope would create naming, mapping, and synchronization problems
without a distinct user concept.

### Persist workstreams

Rejected for the first version. A workstream field would currently only reorder
a short list, becomes stale after branch/worktree cleanup, and has no observed
failure that requires it.

### Add Brief UI to wenlan-app

Rejected because no user job has been demonstrated. The app does not gain UI
merely because the daemon gains a structured contract.

### Remove Markdown entirely

Rejected because the post-handoff file is a low-cost reassurance and inspection
artifact. It remains a one-way receipt, not a second authority.

