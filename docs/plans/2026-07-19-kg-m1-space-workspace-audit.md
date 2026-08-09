# M1 mapping / collision audit — `pages.space` × `pages.workspace`

Rung M1 ("honest columns") of the KG unified-model spec v3. This audit is the
first deliverable of M1 and gates the schema change: no column is folded until
the numbers below are on the record.

**Direction (GT3, non-negotiable):** `pages.workspace` is the authoritative
page-scope axis (migration 63; backfilled as the modal `space` of the page's
source memories). `pages.space` is a rename of the old `domain` column and was
never a scope column. Scope migrates **FROM `workspace` INTO `space`**; the
`pages.space` residue is *classified*, never assumed to be scope.

## Method

Snapshot taken with the SQLite **online-backup API** against the live
production database, read-only:

```
sqlite3 "file:$HOME/Library/Application Support/wenlan/memorydb/origin_memory.db?mode=ro" \
  ".backup '<snapshot>'"
```

A raw file copy is unsound under WAL (§6.9); the backup API is the same
mechanism M1's pre-migration backup must use.

> `SELECT COUNT(*) FROM pages` returns **0** on this database — the known libSQL
> vector-index `COUNT(*)` bug. Every count below comes from `GROUP BY`
> aggregation over enumerated rows, which is unaffected. Row totals reconcile
> to **197** on both axes independently.

## 1. Distinct `pages.space` values (n = 197 pages)

| `pages.space` | pages | in `spaces` registry? | classification |
|---|---:|---|---|
| `<NULL>` | 104 | — | no scope recorded |
| `wenlan` | 49 | yes | origin |
| `writing` | 27 | yes | origin |
| `engineering` | 4 | yes | origin |
| `work` | 2 | yes | origin |
| `lucian-site` | 2 | yes | origin |
| `workflow` | 1 | yes | origin |
| `ultrapowers` | 1 | yes | origin |
| `strategy` | 1 | yes | origin |
| `ruru` | 1 | yes | origin |
| `personal` | 1 | yes | origin |
| `origin-website` | 1 | yes | origin |
| `origin` | 1 | yes | origin |
| `meridian` | 1 | yes | origin |
| `frontend` | 1 | yes | origin |

**Headline: zero category residue.** Every one of the 14 distinct non-NULL
`pages.space` values is a registered space name:

```sql
SELECT DISTINCT p.space FROM pages p
WHERE p.space IS NOT NULL AND p.space NOT IN (SELECT name FROM spaces);
-- 0 rows
```

The spec anticipated page-type categories (`recap` / `decision` / `people`)
squatting in this column. On this database there are **none**. The same query
returns 0 rows for `pages.workspace` and for `memories.space`.

**Consequence for the migration:** the category→`kind` classification rule must
be *data-driven*, not value-listed — membership in the `spaces` registry is the
discriminator (`in registry ⇒ origin`, `not in registry ⇒ category residue`).
That rule is correct on this database (where it moves nothing) and stays correct
on a database that does carry categories.

## 2. Distinct `pages.workspace` values

| `pages.workspace` | pages |
|---|---:|
| `wenlan` | 90 |
| `<NULL>` | 65 |
| `writing` | 15 |
| `personal` | 10 |
| `workflow` | 5 |
| `ruru` | 5 |
| `work` | 2 |
| `engineering` | 2 |
| `testing` | 1 |
| `origin-website` | 1 |
| `lucian-site` | 1 |

## 3. Cross-tab `space` × `workspace`

Agreement classes (the load-bearing summary):

| class | pages | share |
|---|---:|---:|
| `space` NULL, `workspace` set | 67 | 34.0% |
| both set and equal | 60 | 30.5% |
| both NULL | 37 | 18.8% |
| `workspace` NULL, `space` set | 28 | 14.2% |
| **both set and DIFFERENT** | **5** | **2.5%** |
| total | 197 | 100% |

Full cell-level cross-tab:

| `space` | `workspace` | pages |
|---|---|---:|
| `<NULL>` | `wenlan` | 47 |
| `wenlan` | `wenlan` | 41 |
| `<NULL>` | `<NULL>` | 37 |
| `writing` | `writing` | 14 |
| `writing` | `<NULL>` | 13 |
| `<NULL>` | `personal` | 10 |
| `wenlan` | `<NULL>` | 6 |
| `<NULL>` | `ruru` | 4 |
| `<NULL>` | `workflow` | 4 |
| `engineering` | `<NULL>` | 2 |
| `work` | `work` | 2 |
| `<NULL>` | `engineering` | 1 |
| `<NULL>` | `testing` | 1 |
| `engineering` | `engineering` | 1 |
| `engineering` | `wenlan` | 1 |
| `frontend` | `wenlan` | 1 |
| `lucian-site` | `<NULL>` | 1 |
| `lucian-site` | `lucian-site` | 1 |
| `meridian` | `<NULL>` | 1 |
| `origin` | `<NULL>` | 1 |
| `origin-website` | `<NULL>` | 1 |
| `personal` | `<NULL>` | 1 |
| `ruru` | `ruru` | 1 |
| `strategy` | `writing` | 1 |
| `ultrapowers` | `<NULL>` | 1 |
| `wenlan` | `origin-website` | 1 |
| `wenlan` | `workflow` | 1 |
| `workflow` | `<NULL>` | 1 |

### NULL counts that drive the `NOT NULL` rebuild

- `workspace IS NULL`: **65** pages (37 both-NULL + 28 space-only).
- `space IS NULL`: **104** pages.
- Pages with **no scope on either axis**: **37** → these are the rows that
  normalize to `unfiled`.
- Pages recoverable from `space` when `workspace` is NULL: **28**.

### 4. The 5 collisions

Both columns set and disagreeing. Under GT3 `workspace` wins and the `space`
value is residue; all 10 values involved are registered spaces, so none is a
category.

| page id | `space` | `workspace` | `creation_kind` | title |
|---|---|---|---|---|
| `page_4ae12d66…fefa3e7` | `wenlan` | `origin-website` | distilled | useorigin.app Website Hosting and SEO |
| `page_6f4a3d1f…32a89146` | `engineering` | `wenlan` | distilled | Single-LibSQL Triple-Hybrid |
| `page_177ed6fa…46e8321d` | `frontend` | `wenlan` | distilled | Theme-Aware Card Styling Patterns |
| `page_e24c0e69…6cc413` | `wenlan` | `workflow` | distilled | Code review discipline in Origin development |
| `page_2ee8e599…bd8dac59` | `strategy` | `writing` | distilled | Software-free thesis — solo founder hedges |

**Not a stop condition.** Every collision is *origin vs origin*, which GT3
already resolves (`workspace` is authoritative). The spec's unresolved case
would be a collision where the `space` side is a category — that case has zero
rows here.

## 5. `unfiled` target space

The `spaces` registry is `(id TEXT PRIMARY KEY, name TEXT NOT NULL UNIQUE, …)`
and both `pages.space` and `memories.space` store the **name**, not the id.

```sql
SELECT id, name FROM spaces WHERE name LIKE '%unfiled%';  -- 0 rows
```

`unfiled` is free; the migration must mint the registry row and use its **name**
as the column value.

## 6. Code-side inventory

### 6.1 Category producers: none

Exactly one **production** insert path writes `pages.space`:
`post_write.rs:2622` (`insert_page_with_kind`, inside `create_page_impl`). Every
other `insert_page*` call site sits below its file's `#[cfg(test)]` marker —
`maintenance.rs:514,539`, `post_ingest.rs:1472`, `refinement_queue.rs:1440`,
`distill.rs:2594,2717`, `page_watcher.rs:308-767`, `citations.rs:753+`,
`refinery/mod.rs:1605+`, `detect.rs:193`. Every literal `"recap"`-as-space is a
fixture (`post_write.rs:4627+`; `db.rs:41231,44654,48085,51317+`, all past the
top-level test module at `db.rs:31691`).

The `insert_page` wrapper at `db.rs:25676` — the second NULL producer named in
the M1 brief — has **zero production callers**; its `workspace: None` default is
reachable only from tests.

**Therefore the category→`kind` classification step has no live input.** This is
a property of the code, not of this snapshot.

### 6.2 The category reader is live, and can never match

`db.rs:27916-27919` filters `AND c.space = ?3` under the comment *"pages store
their category in `space`"*. It is reachable from `requests.rs:425` →
`memory_routes.rs:2050`, and MCP `tools.rs:3127` actively instructs agents to
pass `page_type: 'recap' | 'decision'`. A documented, agent-facing filter that
matches nothing. Out of M1 scope (removing it changes the wire contract) —
recorded here as a follow-up.

### 6.3 `pages.space` is already read as scope in production

`COALESCE(workspace, space)` appears at `post_write.rs:638`, `723-729`,
`1083-1086`, `1162-1163`, `1266`, and `lint/pages/db_checks.rs:236-244`. These
resolve exactly the **28** workspace-NULL-only rows in §3. The migration must
preserve that resolution, not discard it.

### 6.4 NULL producers (production paths only)

| site | what it does | disposition |
|---|---|---|
| `wenlan-mcp/src/tools.rs:2074` | `workspace: None` | not fatal end-to-end — the server mirrors `space`→`workspace` at `memory_routes.rs:2084-2096`; still fixed |
| `synthesis/overview.rs:78-90` | `ensure_overview_page` hardcodes **both** `space: None` and `workspace: None` | **not in the M1 brief.** The reserved Overview page is global by design; needs an explicit scope decision |
| `db.rs:25676` | `insert_page` wrapper defaults both | test-only reachable; fixed for hygiene |

Already honest: `distill.rs:756,759`, `refinement_queue.rs:299,302`,
`eval/shared.rs:2529,2532`. The M0 write gate passes both columns through
verbatim (`post_write.rs:2585,2603,2628,2633`) and invents no NULL — every NULL
originates upstream of it.

### 6.5 Indexes

Both scope indexes are partial and both predicates go vacuous under `NOT NULL`:

- `idx_pages_workspace` — `db.rs:6907`, `ON pages(workspace) WHERE workspace IS NOT NULL`
- `idx_pages_space` — `db.rs:6347` (m50), also partial. **Not in the M1 brief**; rebuilt alongside.

Untouched: `idx_pages_entity_id` / `idx_pages_status` (`db.rs:6058,6060`),
`idx_pages_embedding` DiskANN (`6130`, self-heal `6195`). `idx_pages_domain` was
dropped by m50 (`6331`).

### 6.6 Rename / merge cascades — the double-apply hypothesis does not hold

Three identical statements: `update_space` (`db.rs:8071-8085`), `delete_space`
move-branch (`8252-8265`), `reassign_memories_space` (`8376-8389`). Each is a
**single UPDATE**, so every matched row is touched once and the draft `version+1`
bumps once. There is no row-level double-apply. The real failures after the fold:

1. **Duplicate SET target** at all three sites — `space = CASE…` and
   `workspace = CASE…` collapse onto one column. Must be hand-collapsed.
2. **WHERE becomes a tautology** — `space=?2 OR space=?2`.
3. **Rename semantics change on the 5 collisions.** Today a rename half-applies
   (rewrites `space`, leaves `workspace`). Post-fold it applies fully or not at
   all. Safe only if the 5 rows are reconciled *before* the fold lands.
4. **`delete_space` never rescopes pages** — `db.rs:8141-8207`. The
   `keep`/`unassign`/`delete` branches touch `memories` and `entities` only
   (`8145,8153,8192,8200`); only `move:` reaches pages (`8252`). A supported user
   action leaves pages carrying a scope with no `spaces` row — illegal under M1's
   model and produced on demand. **Not in the M1 brief**; fixed here.

### 6.7 Wire surfaces

**Outputs** — the only page-carrying type is `Page`:

- `wenlan-types/src/pages.rs:16-17` — `#[serde(default, alias = "domain")] pub space: Option<String>`
- `wenlan-types/src/pages.rs:52-53` — `#[serde(default)] pub workspace: Option<String>`

Neither has `skip_serializing_if`, so both keys are always emitted. Making the DB
column `NOT NULL` while leaving the wire field `Option<String>` means `space`
simply stops ever being `null` — still valid against the app's `string | null`.
The `wenlan-app` frontend never reads `workspace` at all (only unrelated CSS and
theme identifiers) and models page `space` as `string | null`.

> **Do not retype these fields to `String` in this PR.** The JSON is unchanged
> either way, but it silently removes `#[serde(default)]`, turning any client or
> older row that omits the key into a hard deserialize error.

**Inputs** — all safe to leave alone: `CreateConceptRequest.space/.workspace`
(`requests.rs:251-252,257-260`), `SearchPagesRequest.space` (`419-428`), headers
`x-wenlan-space` + legacy `x-origin-space` (`space_header.rs:14-15`, feeds
`space` only), MCP `CreatePageParams.space` (`tools.rs:924-926`).
`CreatePageDraftRequest` / `UpdatePageDraftRequest` (`requests.rs:268-397`) carry
`space` but no server route consumes either type. The CLI is clean —
`wenlan pages` reads local `.md` and never the API.

**Conclusion: M1 requires no wire-contract change.**

### 6.8 `spaces` registry and the `unfiled` target

`db.rs:2393-2400` (m12), extended by `sort_order` (m14, `3622-3642`) and
`starred` (m15, `3645-3660`). **No CHECK, no FK, no UNIQUE beyond `name`.** The
identity that matters is `name`, not `id` — every cascade keys on `s.name`
(`7896-7930`, `8047`, `8281`); `id` is an inert UUID. There is no validation of
space names at all: `create_space` (`7984-8029`) inserts verbatim.

`unfiled` exists nowhere and is free to mint. `uncategorized` is the incumbent
"no space" sentinel and a collision risk — `read_scope.rs:41-53` returns
`AmbiguousUncategorized` if it is ever registered as a real space, which is why
`unfiled` (not `uncategorized`) is the target name.

**The binding constraint — writes fail closed.** `registered_space_or_none`
(`db.rs:7969-7982`) returns `Some` only when a matching `spaces` row exists, and
silently NULLs otherwise. It is enforced at `memory_routes.rs:211-230` and
`27-50`, `ingest.rs:135-151`, `importer.rs:665`, `distill.rs:65`,
`import_routes.rs:337`. So a migration that backfills `space='unfiled'` **must
insert the `spaces` row first, in the same transaction**, or every later write
naming `unfiled` is nulled — reintroducing precisely the NULLs M1 removes.
Precedent for minting registry rows inside a migration: m12 at `db.rs:3576-3585`.

### 6.9 Migration mechanics — house style

Mixed, and the split is principled:

- **Additive** (76 / 78 / 79): bare `execute_batch` + `PRAGMA user_version`,
  replay-safe via `IF NOT EXISTS` / `INSERT OR IGNORE` (`7331-7352`, `7378-7414`,
  `7430-7448`).
- **Destructive / multi-statement** (m50, and the 73/77 helper): explicit
  transaction — `PRAGMA foreign_keys=OFF` + `BEGIN` + `COMMIT`/`ROLLBACK`
  (`6309-6360`), helper `BEGIN TRANSACTION` at `7460`.

M1 is squarely the second class.

**The codebase has already ruled against rebuilding `pages`, in writing.** m67 at
`db.rs:7056-7064`: *"a full table rebuild of `pages` would have to recreate its
FTS mirror, vector index, and triggers (rowid-coupled) — high risk."* The
coupling is real: `pages_fts` is contentless (`content='pages'`,
`content_rowid='rowid'`) with 3 triggers (`6070-6086`), and `search_pages` joins
`vector_top_k(...) ON c.rowid = vt.id` (`27928-27929`). m67 instead patches
`sqlite_master.sql` under `PRAGMA writable_schema=ON`/`RESET` (`7094-7107`).

Two precedents:

| approach | precedent | cost |
|---|---|---|
| full table rebuild | m46 (`5988-6086`) — had to recreate FTS + 3 triggers + 4 scalar indexes + the vector index across phases E–L | high risk, explicitly warned against by m67 |
| `writable_schema` text patch | m67 (`7094-7107`) | cheap, rowid-preserving, **but does not validate existing rows** |

The text patch stamps `NOT NULL` without checking anything, so it is safe **only
if the backfill provably clears every NULL in the same transaction**. M1 takes
that path and pairs it with a post-backfill `NULL`-count assertion inside the
transaction, aborting the migration if any row survives.

**No backup or integrity helper exists to reuse.** A grep for `integrity_check` /
`VACUUM INTO` / `.backup(` across core + server returns only `spaces.rs:39-46`
(renames a legacy file). `operation_receipts` (m79, `7433-7440`) is an **API
idempotency** receipt keyed `(caller_id, operation_id)` — a false friend, not
migration integrity. M1 writes this fresh.

## 7. Dispositions

| finding | disposition |
|---|---|
| 5 origin-vs-origin collisions | GT3 resolves: `workspace` wins. Reconciled before the fold lands. |
| zero category residue; 3 of 5 `kind` values already have homes | **`kind` deferred to M3** (decided 2026-07-19). M1 ships the scope fold only. The fold ledger still ships — it is what makes the fold reversible. |
| `delete_space` manufactures pages scoped to a deleted space | fixed in M1 — M1 owns scope legality |
| `idx_pages_space` also partial | rebuilt alongside `idx_pages_workspace` |
| `overview.rs` Overview page has no scope | explicit decision required; see impl spec |
| live category reader that can never match (`db.rs:27916`, MCP `tools.rs:3127`) | **follow-up, out of M1** — removing it changes the wire contract |
| newer-schema refusal (spec says missing) | **already implemented** at `db.rs:2955-2958`; spec text is stale |

