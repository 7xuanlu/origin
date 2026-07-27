# Default Save Space and Client Context Design

## Goal

Give Wenlan one truthful, cross-client rule for where new scoped content is
saved, while keeping reads broad by default and preserving explicit per-call
control.

The user-visible outcome is:

- Wenlan has one daemon-owned **Default save space**.
- Explicit choices and local working context can override that default.
- New Memories, Pages, and Entities use the resolved write destination.
- Reads remain All Spaces unless the caller explicitly supplies a narrower
  context.
- Wenlan always reports the destination that actually received or already
  owned the object.

## Context

The current surfaces have overlapping but inconsistent behavior:

- `wenlan spaces default` reads and writes a top-level `default` name in
  `~/.wenlan/spaces.toml`. The daemon and desktop app do not own or expose that
  setting.
- The Claude and Codex resolver scripts combine explicit input,
  `WENLAN_SPACE`, cwd mappings, a top-level file default, repo basename, and a
  topic fallback. The top-level default currently runs before repo inference.
- `wenlan-mcp` treats `WENLAN_SPACE` as a lock: it hides the `space` field from
  tool schemas and ignores an explicit tool argument that differs.
- The daemon already treats a request-body `space` as stronger than the
  `X-Wenlan-Space` header in most scoped handlers.
- `wenlan-app` can create and browse Spaces and can pass an explicit space on
  several operations, but it has no app-wide default-save setting. Its
  standalone Quick Capture sends `domain` as ingest metadata, while
  `/api/ingest/memory` stores canonical `space` as unset.
- `create_relation` and observations do not own an independent space. They
  attach to existing entities and source memories.

The result is a fragmented mental model: "default" can mean a CLI file value,
an environment lock, a request scope, an app selector, or no canonical scope at
all.

## Product Decisions

### Default means write destination, not read filter

**Default save space** applies only when creating new top-level scoped content:

- Memory capture and Memory import.
- New Page creation.
- New Entity creation.

It does not silently filter Home, Search, Recall, Memory Stream, or other read
surfaces. A read without explicit or local context remains All Spaces.

This separation preserves the app's whole-library overview while still making
new writes predictable.

### Explicit choice beats every default

The client-side write resolution order is:

1. Explicit per-call space.
2. `WENLAN_SPACE`.
3. Longest matching cwd mapping from `~/.wenlan/spaces.toml`.
4. Current git repository basename, only when a registered Space has that
   exact name.
5. Omit client context and let the daemon use Default save space.
6. Uncategorized when no daemon default exists.

The daemon-side write resolution order is:

1. Request-body `space`.
2. `X-Wenlan-Space` header.
3. Daemon-owned Default save space.
4. Uncategorized.

An explicit request-body value therefore overrides `WENLAN_SPACE`, because the
environment value reaches the daemon as a fallback header or a lower-priority
client resolution input. `WENLAN_SPACE` is a process context, not a lock.

### Reads do not consult Default save space

The read resolution order is:

1. Explicit per-call space.
2. `X-Wenlan-Space` or an equivalent client-resolved local context.
3. All Spaces.

CLI read commands expose `--all-spaces` so a user can bypass
`WENLAN_SPACE`, cwd mapping, or repo inference. Default save space is never in
the read chain.

### Uncategorized is the safe fallback

If no Default save space exists, or if the default Space is deleted, creation
continues in Uncategorized. The caller must display that destination; it must
not describe the result as globally scoped or omit the destination.

Unknown explicit, environment, or configured Space names fail with an
actionable error instead of silently filing content into Uncategorized.

Repo inference is different: an unregistered repo basename is not an error and
does not create a Space. The resolver skips that layer and continues to the
daemon default.

### App navigation is not a hidden write-context switch

Opening a Space page in `wenlan-app` changes the visible page, not the global
save preference. App writes use:

1. The action's explicit selector.
2. Default save space.
3. Uncategorized.

Quick Capture and Add Memory show the selected destination before saving and
permit a one-off override. A one-off override does not mutate Default save
space.

## Alternatives Considered

### Recommended: daemon-owned default plus client-local overrides

The daemon owns the durable default. Clients own local context that only they
can know, such as cwd and repository. This gives every official client the same
fallback without asking the daemon to guess a caller's workspace.

### Rejected: keep every client's own default

Keeping the CLI file default, a separate app preference, and MCP environment
behavior avoids a daemon change but guarantees drift. A default changed in the
app would not affect CLI or MCP writes.

### Rejected: apply the daemon default to every unscoped request

This is mechanically simple but changes global reads into scoped reads and
would make Home and Search silently incomplete. It also removes the distinction
between "where new content goes" and "what the user can currently see."

## Durable Storage

The canonical default is the stable ID of one registered Space.

Add an `is_default INTEGER NOT NULL DEFAULT 0` column to `spaces` and a partial
unique index that permits at most one nonzero row. The effective
`default_space_id` is the ID of that row.

This representation has the desired lifecycle:

- Rename keeps the same row and therefore preserves the default.
- Delete removes the row and therefore clears the default automatically.
- The Uncategorized sentinel cannot become the default.
- A database invariant, not application timing, enforces at most one default.

Extend the shared `Space` wire type with:

```rust
#[serde(default)]
pub is_default: bool,
```

The default-management API is:

```text
GET    /api/spaces/default
PUT    /api/spaces/default       {"space_id":"<registered-id>"}
DELETE /api/spaces/default
```

`GET` returns:

```json
{
  "space": {
    "id": "space-id",
    "name": "Wenlan",
    "is_default": true
  }
}
```

When no default exists it returns `{"space": null}`. Setting the sentinel or
an unknown ID fails validation.

## Legacy `spaces.toml` Migration

`~/.wenlan/spaces.toml` remains the user-editable home for cwd-prefix mappings.
Its top-level `default` name becomes a compatibility input, not the canonical
store.

For the first release carrying this contract:

1. After database initialization, the daemon checks the legacy top-level
   `default` only when no database default exists.
2. If the name resolves to a registered Space, the daemon marks that Space as
   default.
3. If the name is missing or unregistered, the daemon leaves the default
   unset, logs one actionable warning, and writes nothing.
4. The new default API mirrors set and clear operations to the legacy key so a
   same-machine older CLI does not immediately display contradictory state.
5. Rename rewrites the mirrored legacy name. Clear removes the legacy key, so a
   later restart cannot re-import a value the user cleared.

New CLI code reads and writes the daemon API. Resolver scripts continue reading
only `[[mapping]]` blocks from `spaces.toml`; they no longer treat its
top-level `default` as a resolution layer.

## Shared Write-Space Contract

Add a focused write resolver in `wenlan-core`:

```rust
pub enum WriteSpaceSource {
    Request,
    Header,
    Default,
    Uncategorized,
    Existing,
}

pub struct ResolvedWriteSpace {
    pub space: Option<String>,
    pub source: WriteSpaceSource,
}
```

The resolver accepts body and header candidates, validates any named candidate
against registered Spaces, reads the daemon default when both are absent, and
returns `None` plus `Uncategorized` when no default exists.

`Existing` is used only when an idempotent top-level create resolves an object
that already exists. The response reports the object's persisted Space; the
resolver must not move it to the requested/default Space.

The shared resolver is used at the request boundary by:

- `/api/memory/store`.
- The canonical Memory import path.
- `/api/memory/entities` before explicit Entity creation enters the canonical
  resolve-or-create capability.
- `/api/pages` when a new Page is created.

It is not used by:

- Relation creation.
- Observation creation.
- Daemon-internal Entity extraction, which inherits the source Memory's
  already-resolved Space rather than independently consulting the global
  default.
- Updates, confirmation, deletion, or explicit move operations.
- Distill and refinement decisions with their own target semantics.
- Any read handler.
- Raw source/document ingestion, which keeps its existing explicit assignment
  workflow.

## Entity and Derived-Write Semantics

Entity identity resolution remains global. Supplying or resolving a write
destination does not force a duplicate Entity into that Space.

For `create_entity`:

- A genuinely new Entity is created in the resolved write destination.
- An existing alias, exact-name match, MinHash match, or vector match remains
  in its current Space.
- The response distinguishes `created` from `resolved_existing` and returns
  the persisted Space.

For `create_relation`:

- The relation does not accept or resolve a Default save space.
- Both endpoint Entity IDs and the source Memory continue to define its
  provenance and visibility.

For observations:

- The observation inherits the owning Entity's context.
- It does not independently consume Default save space.

For daemon-derived Entities:

- Post-ingest extraction and Memory import thread the source Memory's persisted
  Space into genuinely new Entities.
- Existing Entity resolution remains global and never moves the match.
- A derived write never re-runs Default save space after its source Memory has
  already been filed.

For Page refresh, Memory update, confirmation, deletion, distill, and
refinement actions:

- Existing scope stays unchanged unless the operation is an explicit move or
  an explicit curation action whose contract already asks for a destination.

## Truthful Write Receipts

Successful top-level writes return the actual persisted destination and
outcome. Additive wire fields remain backward compatible for older clients.

Memory store responses add:

```json
{
  "space": "Wenlan",
  "space_source": "default",
  "write_outcome": "created"
}
```

Entity create responses add:

```json
{
  "space": "Work",
  "space_source": "existing",
  "write_outcome": "resolved_existing"
}
```

Page create responses add the same three fields. A new Page returns
`write_outcome: "created"`; a dedup match that attaches sources to an existing
Page returns `write_outcome: "attached_existing"` and the existing Page's
persisted Space.

Memory import responses add batch-level `space` and `space_source`. Every newly
imported Memory in that batch uses that destination; skipped duplicates are not
moved. The existing imported/skipped counters remain the outcome contract, so
the batch response does not add a singular `write_outcome`.

For Uncategorized, `space` is `null` and `space_source` is
`"uncategorized"`.

Allowed `space_source` values are:

- `request`
- `header`
- `default`
- `uncategorized`
- `existing`

Allowed top-level `write_outcome` values in this feature are:

- `created`
- `resolved_existing`
- `attached_existing`

Human-facing clients translate these fields, for example:

```text
Saved to Wenlan (repo)
Saved to Personal (default)
Saved to Uncategorized
Resolved existing entity in Work
```

Client-local layers such as `repo` and `cwd-config` may replace the generic
wire label `request` in presentation because the client knows how it produced
the body value. The persisted `space` always comes from the daemon response,
not from the client's prediction.

JSON output exposes literal `space`, `space_source`, and `write_outcome`
fields. `--quiet` emits no success prose.

## CLI Design

### Caller identity

Add a global `--agent-name <id>` option and resolve the caller identity in this
order:

1. `--agent-name`.
2. `WENLAN_AGENT_NAME`.
3. No `X-Agent-Name` header.

The CLI does not default to `wenlan-cli`, because a human, Codex, Claude Code,
or automation may invoke it. Honest absence is more useful than false
attribution.

### Space controls

Add global context options:

```text
--space <registered-name>
--all-spaces
```

`--space` is the explicit highest-priority context for scope-aware commands.
`--all-spaces` is accepted only by reads and bypasses environment, cwd, and
repo context. Passing both is a usage error. Passing `--all-spaces` to a write
is a usage error.

Write commands send no space when all client-local layers miss; the daemon then
applies Default save space or Uncategorized.

Read commands send no space when `--all-spaces` is present or when every
client-local layer misses. The daemon then performs a global read.

### Space management

Keep the existing command vocabulary:

```text
wenlan spaces default
wenlan spaces default <name>
```

Add:

```text
wenlan spaces default --clear
```

These commands use the daemon default API. `spaces list` and `spaces show`
render `is_default` from daemon data rather than reading the TOML file.

## MCP and Plugin Design

`WENLAN_SPACE` becomes a fallback:

- Keep the `space` property in tool schemas even when the environment variable
  exists.
- `effective_space` returns an explicit non-empty tool argument first, then
  `WENLAN_SPACE`.
- The common `X-Wenlan-Space` header may continue carrying the environment
  value because the daemon gives a request-body space higher priority.
- Startup and documentation call this a process default/context, never a lock.

The Claude and Codex resolver scripts:

- Keep explicit argument, environment, and longest-prefix cwd mapping layers.
- Check that a repo basename is registered before returning it.
- Remove the top-level TOML default and topic fallback layers.
- Return no client space when no local layer resolves, allowing the daemon
  default to decide writes and global semantics to decide reads.
- Continue producing a source label so skills can render truthful receipts.

MCP `capture`, new Page creation through `write_page`, and new Entity creation
consume the daemon default when no higher layer exists.

MCP `create_relation` does not gain a space argument. Its required endpoint IDs
and source Memory remain the contract.

## `wenlan-app` Design

The app remains a first-party human interface. It does not globally attach
`X-Agent-Name: wenlan-app` to ordinary requests; absence continues to mean a
local first-party operation. Explicit machine-authored flows may retain their
existing source/caller identities.

### Space management

The Spaces management surface shows one Default badge and provides:

- Set as default on a registered Space.
- Clear default.
- An explanatory line: "New memories, pages, and entities save here unless
  you choose another Space."

The app reads and mutates the daemon default API. It does not read
`~/.wenlan/spaces.toml`.

### Add Memory

The selector initially shows:

1. An action-provided explicit Space, when present.
2. Default save space.
3. Uncategorized.

Changing it affects only that submission.

### Quick Capture

Both embedded and standalone Quick Capture show a compact destination control.
The control loads registered Spaces and Default save space through existing
Tauri wrappers.

Quick Capture writes through the canonical Memory store route rather than
encoding `domain` into `/api/ingest/memory` metadata. Its success state names
the persisted destination returned by the daemon.

If the default changes while a Quick Capture window is already open, the
window refreshes its suggestion before the next submission unless the user has
already chosen a one-off override.

### Reads

Home, Search, Memory Stream, and Space navigation keep their current read
semantics:

- Home and global search remain All Spaces.
- Entering a Space shows that Space where the screen explicitly requests it.
- Entering a Space does not mutate Default save space.

## Error Handling

- Unknown explicit, environment, or cwd-mapped Space: reject the operation and
  name the invalid value.
- Unknown repo basename: skip repo inference and continue.
- Invalid legacy default during migration: warn and leave default unset.
- Default deleted: the database invariant clears it; the next write goes to
  Uncategorized.
- Explicit body/header conflict: body wins and the receipt reports the body
  destination.
- Existing Entity in another Space: return `resolved_existing` and the actual
  Space; do not move or duplicate it.
- App default lookup failure because the daemon is unavailable: preserve the
  draft and show the normal daemon-unavailable error; do not claim a
  destination.
- Write response missing additive fields from an older daemon: clients retain
  their current success behavior but mark the destination unavailable rather
  than guessing.

## Compatibility and Rollout

### Stage 1: daemon and shared wire contract

- Add default persistence and API.
- Add the write-space resolver.
- Apply it to canonical Memory, Memory import, new Entity, and new Page writes.
- Add truthful response fields.
- Preserve current read behavior.

This stage must ship before clients rely on the default API or response fields.

### Stage 2: CLI, MCP, and plugin surfaces

- Move CLI default commands from direct TOML editing to the daemon API.
- Add CLI identity and space controls.
- Convert MCP environment locking to fallback precedence.
- Update both resolver scripts and their parity tests.
- Update skill copy and structured/human receipts.

### Stage 3: `wenlan-app`

- Update the app's pinned shared wire dependency to a release containing Stage
  1.
- Add default management and one-off selectors.
- Move Quick Capture to canonical Memory store.
- Add destination-aware success UI.

The app stage is implemented in its own repository and pull request. The
current app checkout must not be reused destructively when it contains unrelated
local changes.

## Testing

### Core and daemon

- Migration produces at most one default and excludes the Uncategorized
  sentinel.
- Set, replace, clear, rename, and delete preserve the lifecycle above.
- Legacy valid default imports; invalid default does not.
- Body beats header; header beats default; default beats Uncategorized.
- Named unknown inputs fail.
- Reads without body/header remain global even when a default exists.
- Memory store, Memory import, Entity create, and Page create return persisted
  destination and outcome.
- Existing Entity resolution reports its actual Space without moving it.
- Page dedup reports `attached_existing` and the matched Page's actual Space.
- Entity extraction and Memory import derive new Entity scope from the
  persisted source Memory instead of consulting the default a second time.
- Relation and observation writes do not consume the default.

### CLI

- `--agent-name` beats `WENLAN_AGENT_NAME`; absence sends no header.
- Explicit space beats environment, mapping, repo, and daemon default.
- Environment beats mapping and repo.
- Longest cwd mapping wins.
- Registered repo basename resolves; unregistered basename is skipped.
- `--all-spaces` bypasses local read context and is rejected for writes.
- JSON fields and quiet output obey the receipt contract.
- Default set, get, clear, rename, and delete are daemon-backed.

### MCP and plugins

- Tool schemas retain explicit `space` under `WENLAN_SPACE`.
- Explicit tool space beats the environment fallback.
- Header fallback still scopes calls without explicit tool space.
- Resolver parity tests cover the new layer order.
- No resolver returns a top-level TOML default or topic as a Space.
- Capture, Entity, and Page outputs show actual destinations.
- Relation remains source-backed and has no space parameter.

### `wenlan-app`

- Spaces management sets and clears the daemon default.
- Add Memory and both Quick Capture modes preselect the default.
- One-off overrides do not mutate the default.
- Missing default selects Uncategorized.
- Quick Capture persists canonical space through Memory store.
- Success UI renders the daemon-returned destination.
- Home and global search remain All Spaces.
- Existing user changes in the app checkout remain untouched.

## Verification Gates

The Wenlan monorepo implementation must pass focused tests during each task and
the repository floor before PR:

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace --lib
cargo test -p wenlan-mcp --all-targets
python3 scripts/validate-plugin-contract.py
bash scripts/validate-plugin-contract.test.sh
python3 scripts/validate-codex-plugin-slice.py
git diff --check
```

The `wenlan-app` implementation must run its focused Rust and frontend tests,
then the repository's documented build/test floor in its own clean worktree.

## Non-Goals

- Do not make Default save space a read filter.
- Do not auto-create Spaces from repository names.
- Do not infer a repository inside `wenlan-app`.
- Do not add an independent space to Relation or Observation.
- Do not move an existing Entity when create resolves it.
- Do not make an app one-off selector mutate the global default.
- Do not redesign Home or the visual language of Spaces.
- Do not change raw source/document ingestion assignment in this feature.
- Do not use an agent-name guess when caller identity is absent.
