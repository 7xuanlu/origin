# Verifying wenlan-server changes at the HTTP surface

Build, then boot an isolated daemon (never port 7878 — a launchd daemon or a
stale app may own it):

```bash
cargo build -p wenlan-server
WENLAN_PORT=7879 WENLAN_DATA_DIR=/tmp/<unique> RUST_LOG=warn,wenlan_server=info \
  ./target/debug/wenlan-server   # run in background; poll /api/health until 200
```

- Health: `curl http://127.0.0.1:7879/api/health` → `{"status":"ok","db_initialized":true,"version":"0.x.y+g<sha>"}`.
  The `+g<sha>` suffix confirms which build is running.
- Fresh `WENLAN_DATA_DIR` exercises the full migration chain on boot; check
  with `sqlite3 <dir>/memorydb/origin_memory.db "PRAGMA user_version"`.
- Boot needs no GPU/LLM: FastEmbed loads from cache, LLM degrades gracefully.

## Seeding fixtures

Seed through the API, not sqlite3 — the `pages` table has a libsql-only
vector index (`libsql_vector_idx`), so system sqlite3 cannot INSERT into it
(reads are fine).

- Memories: `POST /api/memory/store {"content": "..."}` → `source_id`.
- Pages: `POST /api/pages {"title", "content", "source_memory_ids": [...], "creation_kind": "authored"}`
  — `source_memory_ids` are validated against real memories, so store those first.

## Gotchas

- A sandboxed `kill` does not reach the daemon; kill unsandboxed and confirm
  with `lsof -ti :<port>` (probe the port, not the PID).
- Curl payloads needing control characters (e.g. the page-map U+001F
  fingerprint separator) must be written to a file via python and sent with
  `-d @file` — literal control chars in a Bash command are rejected.
