# Spaces & Smart Tags Redesign

## Summary

Replace the flat category system with **Spaces** (auto-detected + user-defined containers) and **Smart Tags** (LLM-generated with user curation). Add **Activity Streams** as temporal groupings within spaces, with a global timeline view across all spaces.

## Goals

- Memories are attributed to places that match how users actually think (project, tool, topic — not a generic "code" bucket)
- Tags are generated automatically with minimal friction; users curate, not create
- Temporal browsing via activity streams gives a "what was I doing" view
- Server-side filtering replaces client-side category filtering

## Data Model

### Space

```rust
struct Space {
    id: String,           // slug, e.g. "origin-dev"
    name: String,         // display name, e.g. "Origin Dev"
    icon: String,         // emoji
    color: String,        // accent color
    rules: Vec<SpaceRule>,// auto-routing rules
    pinned: bool,         // user pinned
    auto_detected: bool,  // system-created vs user-created
    created_at: u64,      // timestamp
}

struct SpaceRule {
    kind: SpaceRuleKind,  // App, Path, Keyword, UrlPattern
    pattern: String,      // regex or glob
}
```

### Activity Stream

```rust
struct ActivityStream {
    id: String,
    space_id: String,
    name: String,         // auto-generated, e.g. "Debugging auth flow"
    started_at: u64,
    ended_at: Option<u64>,
    app_sequence: Vec<String>,
}
```

### Memory Attribution

Each memory (chunk) gains these fields in VectorDB:

- `space_id: String` — primary space (auto-assigned)
- `tags: Vec<String>` — LLM-generated semantic tags
- `stream_id: String` — which activity stream it belongs to

### Storage

`tags.json` evolves to `spaces.json`:

```json
{
  "spaces": [...],
  "activity_streams": [...],
  "document_spaces": { "source::source_id": "space-id" },
  "document_tags": { "source::source_id": ["tag1", "tag2"] },
  "tags": ["debugging", "api-design", ...]
}
```

## Attribution Pipeline

### Step 1: Space Resolution (immediate, no LLM)

1. Match capture's app name, file path, URL against space rules. First match wins.
2. For code editors, extract git repo / project directory. For browsers, extract domain.
3. No match → create auto-detected space from app+workspace combo (unpinned, `auto_detected: true`).
4. Fallback → "Unsorted" space.

### Step 2: Activity Stream Assignment (immediate)

Uses existing AFK detection (60s idle) as stream boundaries:

- AFK → come back → new stream in the active space
- Switch to different space → close stream in old space, open in new
- Same space, not AFK → same stream continues
- Stream name initially set to app + window title

### Step 3: LLM Enrichment (async, Pass 2)

LLM formatter produces:

- **2-4 semantic tags** — meaningful descriptors ("debugging", "API design", "reading documentation")
- **Space confirmation/override** — can suggest a different primary space
- **Stream name** — short description of the activity ("Investigating LanceDB migration")

### Step 4: User Refinement (on-demand)

- Reassign memory to different space
- Pin/remove tags
- Merge or split activity streams
- Promote auto-detected space → pinned space with custom name/icon

## UX Design

### Memory View — Evolved Layout

Top bar: Space pills replace category pills. Click to filter.

```
[All] [Origin Dev] [ML Research] [Team Comms] [+]
[Search...]

Tags: rust  tauri  debugging  ocr
(contextual — shows tags for active space, or top tags for "All")

── Today ────────────────────────
▼ Debugging auth flow · 2h
  capture — VS Code        rust tauri
  capture — Safari         docs

▼ Reviewing PR #42 · 35m
  capture — GitHub         review

── Yesterday ────────────────────
▶ Implementing vector search · 3h
  3 captures
```

- **"All"** = global timeline (default). Shows streams across all spaces with space badges.
- **Click a space pill** = filtered view of just that space's streams.
- **Activity streams** = collapsible groups within time-based layout.
- **Tags** shown inline per memory; aggregated at top for filtering.
- **Per-memory**: space badge (click to reassign) + inline tags (click to filter).

### Settings

Space management (rename, icon, rules, delete) in Settings/SourceManager page.

## Migration

1. Existing categories become seed spaces with auto-generated rules:
   - `code` → Space with rules matching VS Code, terminals, IDEs
   - `communication` → Space with rules matching Slack, Mail, Messages
   - `research` → Space with rules matching browsers
   - `writing` → Space with rules matching text editors, Notion
   - `design` → Space with rules matching Figma, Sketch
   - `other` → "Unsorted"
2. `document_categories` mappings transfer to `document_spaces`
3. Existing tags carry over as-is
4. Add `space_id`, `tags`, `stream_id` columns to LanceDB chunks table
5. Backfill existing documents with space from category mapping
6. Update LLM prompt: ask for space, tags, stream_name instead of category

## Deferred (not v1)

- Linked/cross-referenced spaces (memory in multiple spaces)
- Space rules editor in settings (v1: auto-detection + rename/pin only)
- Tag vocabulary management UI
- Stream merging/splitting UI
