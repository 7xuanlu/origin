# Sources: left tree + right details — design

**Date:** 2026-07-03
**Scope:** Frontend-only. `src/components/memory/SourcesView.tsx` + its test file.
No daemon/Rust change. Storage unchanged (`~/.wenlan/sources`).

## Problem

Uploading a PDF registers the managed dir `~/.wenlan/sources` as its own
directory source, so a redundant **"sources"** tile appears on the shelf and the
uploaded file is buried one level inside it — instead of sitting next to the
folder-sources (Books, notes, vault) as a peer.

## Confirmed design (user: "yes build it")

Keep the **two-pane** layout. Change what each pane does:

```
┌ LEFT: folder/file TREE ─────┐ ┌ RIGHT: details of selected ───────┐
│ ▸ notes            (folder) │ │  report.pdf                       │
│ ▸ vault            (folder) │ │  pdf · In your library            │
│   report.pdf   (loose file) │ │  [ Open ]                         │
│   notes.md     (loose file) │ │                                   │
│   … no "sources" node …     │ │  (folder selected → file count,   │
└─────────────────────────────┘ │   sync status, Remove)            │
                                 └───────────────────────────────────┘
```

- **LEFT** = a folder/file tree.
  - Root nodes: each **folder-source** (Books, notes, vault) as a folder node,
    **plus** the loose files inside the managed dir hoisted to the root as
    **peers** — the managed `~/.wenlan/sources` source is **not** shown as a node.
  - Folders **expand inline (accordion)**; children (files + subfolders) nest
    under them, lazily read on first expand. Subfolders expand recursively.
  - Clicking a row **selects** it (updates the right pane). Clicking a **folder**
    row also toggles its expansion.
- **RIGHT** = details only. No browsing here.
  - **File** → name, extension, index status, `Open`.
  - **Folder source** → name, on-disk file count, memories, sync status,
    `Remove`, `Reveal`. `Sync` button for obsidian; `Auto-synced` for directory.
  - **Subfolder** (not a source root) → name, file count, `Reveal` only.

## Managed-dir detection

`isManagedSourcePath(path)` → true when path ends in `.wenlan/sources`
(optionally trailing slash). Regex: `/\.wenlan\/sources\/?$/`.
Folder-sources = registered sources where this is false. The managed source (if
present) is read with `readSourceDir(managed.path)` and its entries become root
peers; it is never itself a node.

## Index status wording (kept from prior work)

A supported file (`md`/`txt`/`pdf`) not yet in `listIndexedFiles()` reads
**"Indexing…"** in calm tertiary color — never "not indexed". Gated on the
indexed-file list having loaded (`indexReady`), so nothing is flagged mid-load.
Per-file mtime is unavailable, so a genuinely unreadable file reads "Indexing…"
indefinitely (documented ceiling; a per-file status DTO is the upgrade path).

## Node model

```ts
type SourcesNode =
  | { kind: "folder"; name: string; path: string; source: RegisteredSource; isSourceRoot: boolean }
  | { kind: "file";   name: string; path: string; source: RegisteredSource };
```

- Root folder nodes: `isSourceRoot: true`, `path = source.path`.
- Managed children: `source = managed`, `isSourceRoot: false`, `path = managed.path + "/" + name`.
- Deeper children (on expand): `source` inherited from parent, `isSourceRoot: false`.
- `isIndexed(source, absPath)` = `listIndexedFiles()` contains `${source.id}::${absPath}`.

## Keep / change

**KEEP (do not alter behavior):** two-pane flex container; all `--mem-*` tokens
and font vars; `AddSourceMenu` open/close wiring; `EmptyShelf` ("Nothing on the
shelf yet"); `onManageSources` prop; the "Indexing…" calm wording; helpers
`folderName`, `ext`, `statusLabel`, `STATUS_COLORS`, `relTime`; on-disk file
count (not daemon `file_count`); Remove-with-confirm calling `removeSource`;
Auto-synced (directory) vs Sync button (obsidian).

**CHANGE:** LEFT aside becomes the recursive tree (no book-spine bars). RIGHT
`FolderBrowser` becomes a details-only `DetailPane` (no breadcrumb drill).
Export `sourceCaption` (renamed from `spineCaption`, same body) + `isManagedSourcePath`.
Drop `spineVisual` / `spineHeight` / `SpineVisual` (spine-only, now unused).

## Tests (the contract)

Rewritten `__tests__/SourcesView.test.tsx`:
1. Left tree hides the managed dir and hoists its files as root peers (no "sources" node; `notes`/`vault` folders + `report.pdf`/`notes.md` files present).
2. Clicking a folder node expands it (accordion) → `readSourceDir(folderPath)` called, children shown.
3. Expanding a subfolder reads the joined path and shows its children.
4. Selecting a folder source shows on-disk file count (not daemon `file_count`) and a Remove button; count excludes subfolders.
5. Directory source → `Auto-synced`, no Sync button; obsidian → Sync button.
6. Selecting a loose file shows `Open` and its index status; unindexed supported file reads "Indexing…", never "not indexed".
7. Remove calls `removeSource(id)` after confirm.
8. Empty state (no sources) → "Nothing on the shelf yet" + add button.
9. `isManagedSourcePath` unit: matches `.wenlan/sources` (± trailing slash), rejects `notes` and `.wenlan/sources-backup`.
10. `sourceCaption` unit: "Indexing…" while `last_sync===null`; "N notes" settled; appends ", M skipped".
