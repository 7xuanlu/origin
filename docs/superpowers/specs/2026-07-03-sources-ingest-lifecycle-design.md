# Sources ingest lifecycle — daemon-native (design)

Date: 2026-07-03
Status: design v2 — pressure-tested by boule debate (verdict: approve-with-changes,
medium) + backend verification. Awaiting user review before writing-plans.
Scope: `wenlan-app` frontend + `app/src` Rust glue only. No backend
(`7xuanlu/wenlan`) changes.

## Problem

The Sources tab lets a user register folders, but the lifecycle around them is
thin: single loose files can't be added, there is no ingest feedback, removal
punts to Settings, and the "Sync" button errors on directory sources. The
original plan assumed the **app** had to own file ingestion (extract text,
POST to the daemon, run an in-process watcher) and therefore needed an
app-managed storage location.

That assumption is **wrong**, and validating it inverted the design.

## Review outcome (boule debate, 2026-07-03)

Three heterogeneous reviewers (Claude / Codex gpt-5.5 / Gemini 3.1 Pro) ran a
form → attack → defend → judge council on the v1 spec. **Verdict:
approve-with-changes (medium).** All three endorsed the core pivot (daemon owns
ingest, app is a thin client); none challenged the architecture. Gemini revised
its own vote to reject on an accumulation of *fixable* defects, not an
architectural objection — hence approve-with-changes, not reject.

The conceded must-fixes, each then re-verified against the backend (HEAD
`e6787221`, tag `v0.10.1`), are folded into this v2:

1. `files_total` is hardcoded `0` daemon-side → the determinate spine-fill can
   never engage; **shimmer-only** now (§3, §Signature).
2. No daemon version-gate → the live 0.9.5 daemon shows perpetual "Indexing…";
   add version detection via `/api/health` (§0).
3. Remove leaks the copied upload blob; memories persist by design (§4).
4. Loose-file copy is non-atomic against the 30s scheduler → temp+rename (§1).
5. Mock-only tests → add a real ≥v0.10.0 daemon integration test (§Testing).

Attacks that did **not** land (do not over-correct): DELETE keeps memory rows
(the "notes stay" copy is correct); a custom `knowledge_path` does not stop
`~/.wenlan/sources` from indexing; consolidating uploads into one managed dir
beats registering N single files in place.

## Key finding — the daemon already owns file ingestion

Verified against the backend repo (`7xuanlu/wenlan`, HEAD `e6787221`, tag
`v0.10.1`) and the running daemon:

- **Daemon-side folder + multi-format document ingest landed in v0.10.0**
  (`#320`, commit `4cdc4561`) — the version the app already pins.
  - `POST /api/sources {source_type:"directory", path}` registers a directory
    **or a single file** (`sources/source_routes.rs:96-101`). Dedupe is **by
    path**: re-POSTing an already-registered path returns a ValidationError
    `"Source already registered"` (`:126-131`), it does not create a duplicate.
  - The daemon walks the path, diffs by mtime/hash, extracts text from
    `.md/.txt/.pdf` — **server-side PDF extraction** via `pdf_extract`
    (`sources/directory.rs:399`) — and ingests derived chunks.
  - An always-on **30s scheduler** re-syncs every Directory source
    (`scheduler.rs:25`), so files dropped into a registered dir are picked up
    within ~30s with no explicit call.
- **CLI parity, app-free**: `wenlan sources add <path>` does the same over HTTP
  (`wenlan-cli/.../commands/ingest.rs`). This is the answer to "what if the
  user never installs the app" — the daemon + CLI fully cover local
  directory/file indexing without the desktop app.
- **The daemon does not store raw blobs** — it indexes files in place and
  persists only extracted text + the page projection + `(path, mtime, hash)`
  sync-state. There is no need for an app-managed blob store.

### Version drift (the only reason it looks broken today)

The **running** daemon is **0.9.5** — *before* v0.10.0 — so it registers
Directory sources but never indexes them (proof: the live `directory-books`
source is inert, `file_count: 0`, `last_sync: null`). The app's current
in-process file watcher (`app/src/indexer.rs`) exists as a **bridge** for
pre-v0.10.0 daemons. On a daemon ≥ v0.10.0 it is redundant.

Latest daemon is v0.11.0. Updating the live daemon from 0.9.5 → ≥ v0.10.0 is a
**user action** (never restart/kill the daemon on `:7878` from here). Until
then, files stage correctly but do not index — which is exactly why §0 exists.

## Design

The daemon is the single source of truth for sources. The app is a thin client
that **registers** sources with the daemon and **displays** daemon-reported
state. No app-side extraction, no app-side blob store, no parallel watcher for
new sources.

### 0. Daemon version gate (precondition)

The one hard failure mode the debate surfaced: on a **< v0.10.0** daemon, the
whole feature stages files that never index, producing a permanent "Indexing…"
with no explanation.

- Read the daemon version from `GET /api/health` (`HealthResponse.version`,
  `routes.rs:49,64` — the only endpoint that carries it; `/api/status` and
  `/api/config` do not).
- If `< v0.10.0`: the Add menu and upload path show an **update-daemon** state
  instead of eternal Indexing, e.g. `Your daemon needs an update to index
  files.` with a link to the updater. Registration still works (harmless), but
  the UI is honest that nothing will index yet.
- If `≥ v0.10.0`: normal lifecycle below.

### 1. Add source — two entry points, one mechanism

Both routes end at `POST /api/sources` (the app already has this via
`api.rs` `add_source` / `AddSourceRequest`); the change is to stop diverting
directory sources to the local watcher (`add_directory_source` in `search.rs`)
and send them to the daemon like Obsidian sources already are.

- **Folder** — native directory picker → register the folder **in place**
  (e.g. `~/Documents/Books`). Not copied; it is already the user's folder.
- **Single file ("upload")** — native file picker (`{directory:false}`,
  filter `pdf/md/txt`) → **atomically** place the file in the stable managed
  dir (below): write/copy to a temp name on the same filesystem, then `rename`
  into place, so the 30s scheduler never sees a partially-written file. Then
  ensure `~/.wenlan/sources/` is a registered Directory source. The 30s
  auto-sync ingests it.

### 2. Stable managed path for loose uploads

Aligned under the daemon's knowledge home, `~/.wenlan`, next to `pages/`:

```
~/.wenlan/
├── pages/                 daemon page projection (existing)
├── sources/               NEW — stable home for uploaded loose files
│   ├── .gitignore         contains "*"  → keeps binary PDFs out of git history
│   ├── 西藏生死书.pdf
│   └── paper.pdf
└── db → ~/Library/Application Support/wenlan/memorydb
```

- The app resolves the path as `home_dir()/.wenlan/sources`, mirroring the
  daemon's own `~/.wenlan` resolution (`dirs::home_dir()`).
  `ponytail:` hardcoded to `~/.wenlan`; the daemon's knowledge home can in
  principle be customized (`knowledge_path`) and is **not** exposed via any
  API today. Verified safe: `scan_directory` walks the source's absolute path
  regardless of `knowledge_path`, and only the exact `knowledge_path` / pages
  dir is a reserved ingest root (`directory.rs:139-147`) — `~/.wenlan/sources`
  (a distinct subdir) is not reserved and indexes normally.
- `~/.wenlan/sources/` is registered **once** as a Directory source
  (id `directory-sources`). On repeat registration the daemon returns
  `"Source already registered"` — the app **treats that as success**
  (check-or-ignore), it is not an error path. Subsequent uploads just place a
  file in and let auto-sync pick it up.
- A self-contained `~/.wenlan/sources/.gitignore` (`*`) prevents the daemon's
  git repo from tracking uploaded blobs — no edit to the daemon's root
  `.gitignore`.

### 3. Ingest feedback (aggregate, not per-file)

Poll while a source is settling and show real numbers:

- Per source: `file_count` + `memory_count` from `GET /api/sources`.
- **`GET /api/status` is not usable for per-source progress.** Verified:
  `files_total` is hardcoded `0` (`routes.rs:111`) and `files_indexed` is a
  **global** `db.count()` — neither maps to one source. Do not build the fill
  ratio on them.
- **"Still indexing" signal** = a source with `last_sync == null` or whose
  `memory_count` is still climbing between polls. When counts stop changing,
  it is settled.
- Surface `last_sync_errors` / `last_sync_error_detail` (already on the source
  object) as a quiet "N skipped/failed" so silent skips (image-only PDF, or a
  file over the daemon's size cap) are visible.

No per-file progress bar. Refetch faster (~3s) while a source reports as
syncing, back to the existing slow interval when idle.

### 4. Remove source (CRUD)

Inline "Remove source" in the FolderBrowser header → `DELETE /api/sources/{id}`
(the app already exposes `remove_source`). Confirm before removing.

- Verified: `DELETE` clears only the config entry + `source_sync_state`
  (`source_routes.rs:159-185`); it **does not** delete ingested memory rows.
  So already-indexed notes persist by design — which is what the confirm copy
  promises. Keep that copy.
- **App-side cleanup for uploads:** when removing the managed `directory-sources`
  source (or a single uploaded file), the app deletes the copied blob(s) from
  `~/.wenlan/sources/` so removed uploads do not accumulate as orphaned files.
  (Folder sources are in-place and never copied, so nothing to clean.)

### 5. Sync button fix

The app's `sync_registered_source` (`search.rs`) hard-errors for Directory
sources (`"Only Obsidian sources support manual sync…"`). A daemon ≥ v0.10.0
does support Directory sync, and directory sources auto-sync every 30s anyway.
Fix: for directory sources, drop the manual "Sync" button and show the
auto-synced state instead; keep "Sync" for Obsidian.

### 6. Retire the app's in-process watcher (for new sources)

New directory/file sources go to the daemon, not `add_directory_source` +
`create_file_watcher`. The watcher stays only as a legacy bridge for
pre-v0.10.0 daemons; document that it is redundant on ≥ v0.10.0 and can be
removed once the minimum supported daemon is v0.10.0. `ponytail:` do not rip it
out in this slice — leave it for existing app-registered directory sources
until the daemon floor is raised.

**Positioning reason, not just cleanliness:** the app watcher only indexes
while the app is open. Wenlan's load-bearing differentiator is **Compounding**
— "writes itself while the app is closed" (the autonomous daemon; see
Positioning alignment below). A source that stops indexing when the window
closes contradicts the core pitch. Daemon-native sourcing is the positioning,
so no new UX is built on the watcher.

## Frontend / UX design

Applies the frontend-design pass **inside the existing Sources identity**, not a
new palette. The bookshelf is already this view's signature; the design extends
it to cover the ingest lifecycle rather than bolting on generic progress UI.

### Design language (inherited)

- **Tokens:** the `--mem-*` system (`--mem-surface`, `--mem-border`,
  `--mem-text` / `-secondary` / `-tertiary`, `--mem-accent-indigo`,
  `--mem-hover`, `--mem-shadow-toast`). No new colors.
- **Type:** Fraunces (`--mem-font-heading`) for headings, body face for prose,
  mono (`--mem-font-mono`) for counts and eyebrows.
- **Metaphor:** the shelf. A source is a book spine; spine height already
  scales with `memory_count` (`spineHeight`), so the shelf silhouette reads as
  "how much each source knows."

### Signature: the spine *is* the ingest indicator

The one deliberate risk. A source being indexed is a book being shelved — no
spinner, no progress bar:

- **Added** → the spine appears as a hairline outline (a ghost book) at its
  shelf slot.
- **Indexing** (`last_sync` null or `memory_count` still climbing) → the spine
  **fills from the base upward as an indeterminate shimmer**. (Determinate
  percent is *not possible*: the daemon reports no per-source total —
  `files_total` is hardcoded 0. The fill communicates "still arriving", not a
  percentage.) A mono caption reads `Indexing…`.
- **Settled** → spine solid at full height; caption flips to `142 notes`.
- **Skipped/failed** → a thin notch near the base + caption `2 skipped`
  (hover reveals the reason from `last_sync_error_detail`, e.g. an image-only
  PDF).

Structure encodes truth: silhouette = knowledge-per-source; fill = still
arriving. `prefers-reduced-motion` → no fill animation, static `Indexing…`
label instead.

### Moments

1. **Add** — one quiet `＋` on the shelf-rail header opens a two-item menu:
   **Add a folder** / **Add files**. Not "Register source." File picker filters
   to `pdf / md / txt` (the daemon's supported set). On a < v0.10.0 daemon the
   menu shows the update-daemon state (§0).
2. **Upload a loose file** — atomically placed in `~/.wenlan/sources/`, joins
   the managed **Uploads** spine (created if absent). Toast in the
   MilestoneToaster idiom (eyebrow · heading · body): `Added` / `paper.pdf is
   on the shelf` / `Indexing in the background.`
3. **Autonomy cue (Compounding, made visible)** — a persistent shelf-header
   line when ≥1 directory source exists: `Syncs in the background, even when
   Wenlan is closed.` This surfaces the differentiator instead of hiding it in
   a tooltip.
4. **Remove** — inline **Remove** in the FolderBrowser header; confirm reads
   `Remove Books? Indexed notes stay in your library; this source stops
   syncing.`
5. **Auto-sync state** — directory sources replace the (broken) manual "Sync"
   button with a quiet `Auto-synced · updated 2m ago` (from `last_sync`).
   Obsidian keeps **Sync**.
6. **Empty state** — keep `Nothing on the shelf yet`; primary action
   `Add your first source` opens the same two-item menu.

**Deferred — Provenance ("grounds N notes"):** the v1 spec assumed
`GET /api/pages/{id}/sources` could power a source→pages "grounds 142 notes"
readout. Verified mismatch: that endpoint returns page→**memory-source**
provenance (which memories fed a distilled page), not registered-source→pages,
and there is **no existing endpoint** that counts the notes a given file-source
grounds. Shipping it would need a backend change (out of scope). Moved to
fast-follow. Source-cited positioning is still honored by keeping sources
visible and traceable in the browser; the numeric "grounds N" readout waits.

### Shelf with an indexing source (wireframe)

```
┌ Sources ─────────────────────────────────────────── ＋ ─┐
│ Syncs in the background, even when Wenlan is closed.     │
│                                                          │
│    ██     ██     ▁▁                                      │
│    ██     ██     ▓▓  ← filling (indeterminate shimmer)   │
│    ██     ██     ▓▓                                      │
│   Books  Vault  Uploads                                  │
│   142     210   Indexing… 12                             │
└──────────────────────────────────────────────────────────┘
```

### Copy (user-side, active voice, no em-dashes)

| Element | Text |
|---|---|
| Add menu | `Add a folder` · `Add files` |
| Upload toast | `paper.pdf is on the shelf` / `Indexing in the background.` |
| Indexing caption | `Indexing…` → `142 notes` |
| Skipped | `2 skipped` (hover: reason) |
| Autonomy line | `Syncs in the background, even when Wenlan is closed.` |
| Remove confirm | `Remove Books? Indexed notes stay in your library; this source stops syncing.` |
| Auto-sync state | `Auto-synced · updated 2m ago` |
| Daemon too old | `Your daemon needs an update to index files.` |
| Empty | `Nothing on the shelf yet` / `Add your first source` |

### Quality floor

Responsive (shelf scrolls horizontally on narrow widths, matching the existing
`overflow-x` pattern), visible keyboard focus on spines / menu / buttons,
`prefers-reduced-motion` honored (no spine-fill animation), errors give
direction not apology.

### Self-critique vs generic defaults

The spine-as-ingest-state is subject-specific (books being shelved), not the
default spinner/progress-bar, and it reuses the app's real `--mem-*` identity
rather than the AI-generic cream/serif/terracotta look. The one bold element is
the spine fill; everything else stays quiet.

## Positioning alignment (llm-wiki)

Checked against the locked positioning (`wenlan-positioning-lane-llm-wiki`,
2026-06-23 / re-verified 2026-07-01). Triad: **Source-cited · Compounding ·
Reconciled.** What this feature must honor:

- **Compounding** ("writes itself while the app is closed") is the load-bearing
  differentiator. → Daemon-native auto-sync, surfaced in the UI (moment 3).
  The app watcher is positioning-wrong and gets no new UX.
- **Source-cited** (provenance). → Ingested files stay visible and traceable in
  the browser; the numeric `grounds N notes` readout is deferred (no endpoint),
  but sources are never treated as write-only.
- **Copy discipline** — no em-dashes in user-facing strings
  (`no-em-dashes-in-user-facing-docs`); frame ingestion as an enjoyable
  auto-wiki that cites everything, never as a chore.
- **Reconciled** (human trust gate) is out of scope here (it lives in the
  refinery/review queue), but nothing in this feature should auto-merge or
  silently drop a source's conflicts.

## Data flow

```
Folder    ─ picker ─────────────► POST /api/sources {directory, <folder>}  (in place)
Loose file ─ picker ─ temp+rename ─► ~/.wenlan/sources/ ─► (dir registered once) ─► daemon
                                                                              │
                                       daemon: walk → mtime/hash diff → PDF/text extract
                                              → ingest chunks → 30s auto-sync
                                                                              │
UI ◄── GET /api/health  (version gate, ≥ v0.10.0 required) ──────────────────┤
   ◄── poll GET /api/sources (file_count, memory_count, last_sync_errors) ────┘
Remove ─────────────────────► DELETE /api/sources/{id}  + delete copied blob (uploads)
```

## Constraints & caveats

- **Requires daemon ≥ v0.10.0.** The live daemon is **0.9.5** — the binary
  bundled in the installed `/Applications/Wenlan.app`, launched via the
  `com.wenlan.server` LaunchAgent (`KeepAlive`, `RunAtLoad`,
  `WENLAN_DATA_DIR=~/Library/Application Support/wenlan`). Latest release is
  **v0.11.0** (backend HEAD is `v0.10.1`). On 0.9.5, §0 shows the update-daemon
  state and nothing indexes. Updating swaps a production binary and may migrate
  the live db (9158 memories) — back up first; treat as a confirmed step, not
  automatic. The version-gate uses `GET /api/health`'s `version`.
- The daemon extracts `.md/.txt/.pdf` only — no docx/rtf/OCR server-side.
  The upload picker filter must match (`pdf/md/txt`).
- No blob copy for whole folders — only loose single-file uploads are copied.
- Writing into `~/.wenlan/sources/` places app-managed files inside the
  daemon's git home; the self-gitignore keeps history clean, and verification
  confirmed the subdir indexes fine under a custom `knowledge_path`. Still a
  deliberate boundary crossing chosen for `.wenlan` alignment per user
  preference.

## Out of scope (deferred)

- **Provenance `grounds N notes`** — no source→pages count endpoint; needs a
  backend change (see Moments).
- Per-file "index this / skip" choice (needs a daemon-side skip list).
- Per-file progress rows (queued→indexing→indexed→error); the daemon reports
  aggregate counts, not per-file events.
- Per-file removal / re-index from the browser.
- Removing the in-process watcher entirely (blocked on raising the daemon floor).
- Exposing the daemon's knowledge home via an API (would remove the hardcoded
  `~/.wenlan`).

## Testing

- Frontend (Vitest): upload picker registers a source via the daemon client
  (mocked); the `< v0.10.0` health response shows the update-daemon state and
  suppresses eternal Indexing; remove calls `DELETE /api/sources/{id}` and
  cleans the copied blob; feedback renders `file_count`/`memory_count`/error
  counts from mocked `/api/sources`; directory sources show no manual Sync
  button, Obsidian do.
- Rust (`cargo test`): the upload command places into `~/.wenlan/sources/`
  atomically (temp+rename), ensures the `.gitignore`, registers the dir once,
  and treats `"Source already registered"` as success on repeat.
- **Integration (new, addresses the mock-only gap the debate flagged):** against
  a real ≥ v0.10.0 daemon (build v0.11.0 locally from `../wenlan`), register a
  temp directory, drop a `.md`/`.pdf`, and assert `memory_count` climbs and
  `last_sync` populates within the scheduler window. This is the only test that
  exercises the actual daemon contract the whole design rests on.
- Revert the 3 dev-only TEMP preview edits (`SourcesView.tsx`, `Main.tsx`,
  `UpdaterDialog.tsx`) before the implementation PR.

## Open items for review

1. Managed dir name: `~/.wenlan/sources/` vs `uploads/` vs `library/`.
2. ~~Keep the app watcher?~~ **Resolved by positioning:** daemon-native only;
   watcher is a legacy bridge with no new UX. Full removal deferred until the
   minimum daemon is raised to v0.10.0.
3. ~~Provenance in this slice or fast-follow?~~ **Resolved by verification:**
   deferred — no source→pages endpoint exists; it needs a backend change.
4. Version-gate copy/affordance: is `Your daemon needs an update to index
   files.` + updater link the right treatment, or should the whole Add flow be
   disabled on < v0.10.0?
