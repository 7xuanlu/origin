# Profiles & Agent Connections Design

**Date**: 2026-03-10
**Status**: Approved
**Ticket**: Add profiles + agent_connections tables (P1)

## Problem

`source_agent` is a free-form string on chunks, entities, observations, and relations. No registry of known agents, no trust gating, no per-agent stats. No user identity model.

## Decisions

- **Profile**: Minimal — single-row table with name/display_name. Auto-created on first launch.
- **Agent connections**: Registry table with name, type, trust level, enabled flag, stats.
- **Registration**: Auto-register on first write. Default trust = `"review"`.
- **FK strategy**: No FK constraint. `source_agent` stays TEXT, joined by convention on `agent_connections.name`.
- **Implementation location**: Both tables in `memory_db.rs`, same libSQL database.
- **Frontend**: New sections in Settings (SourceManager) for profile editing and agent management.

## Schema

```sql
CREATE TABLE IF NOT EXISTS profiles (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    display_name TEXT,
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS agent_connections (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    agent_type TEXT NOT NULL DEFAULT 'api',
    description TEXT,
    enabled INTEGER NOT NULL DEFAULT 1,
    trust_level TEXT NOT NULL DEFAULT 'review',
    last_seen_at INTEGER,
    memory_count INTEGER NOT NULL DEFAULT 0,
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_agent_connections_name ON agent_connections(name);
```

### Field details

**profiles**: Single row. `id` is UUID generated once. Bootstrap on `MemoryDB::initialize()` if zero rows exist (default `name = "User"`).

**agent_connections**:
- `name` (UNIQUE) — join key matching `source_agent` on chunks/entities
- `agent_type` — `"cli"`, `"ide"`, `"chat"`, `"api"` (for filtering/display)
- `trust_level` — `"full"` (auto-confirm), `"review"` (default, confirmed=0), `"untrusted"` (stored but excluded from search)
- `enabled` — kill switch, rejects writes when false
- `memory_count` — denormalized, incremented on store, decremented on delete
- `last_seen_at` — updated on each write

## Auto-register logic

```
store_memory(content, source_agent="claude-code", ...)
    → lookup agent_connections WHERE name = "claude-code"
    → if not found: INSERT with defaults (type="api", trust="review")
    → update last_seen_at, increment memory_count
    → if enabled = false: reject write (return error)
    → if trust_level = "review": set confirmed = 0 on chunk
    → if trust_level = "full": set confirmed = 1
    → if trust_level = "untrusted": store but exclude from search
```

Same logic for `create_entities`, `add_observations`, `create_relations`.

Writes with `source_agent = None` (local captures, file indexer) skip agent lookup — treated as first-party/trusted.

## API surface

### REST endpoints

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/api/profile` | Get user profile |
| PUT | `/api/profile` | Update name/display_name |
| GET | `/api/agents` | List all registered agents |
| GET | `/api/agents/:name` | Get single agent by name |
| PUT | `/api/agents/:name` | Update agent settings |
| DELETE | `/api/agents/:name` | Remove agent connection |

### Tauri commands

`get_profile`, `update_profile`, `list_agents`, `get_agent`, `update_agent`, `delete_agent`

## Frontend

New sections in SourceManager (Settings page):

- **Profile section** (top): Name and display name edit fields.
- **Agents section**: List of registered agents. Each shows name, type badge, trust level dropdown, enabled toggle, memory count, last seen timestamp. Delete button per agent.

## Out of scope

- Per-agent permissions (noted for future)
- Auth/API keys on agents
- Multi-user support
- FK constraint on `source_agent`
- Migration/backfill of existing `source_agent` data (lazy registration handles this)
