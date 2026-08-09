# Sources Ingest Lifecycle Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the Sources tab a full daemon-native ingest lifecycle — add folders and loose files, see them index, remove them — with the app as a thin client to the daemon.

**Architecture:** The daemon (`7xuanlu/wenlan` ≥ v0.10.0) already owns file ingestion (walk, mtime/hash diff, PDF/text extract, 30s auto-sync). This app change stops diverting directory sources to the local in-process watcher and sends them to the daemon like Obsidian sources already are, adds a loose-file "upload" path that atomically stages files into `~/.wenlan/sources/`, gates the UI on daemon version, and reworks the shelf UX so a book spine encodes ingest state. No backend (`7xuanlu/wenlan`) changes.

**Tech Stack:** Rust (Tauri 2 commands, `reqwest` HTTP client to daemon on `:7878`), React 19 + `@tanstack/react-query` + `sonner` toasts, Tailwind v4 (`--mem-*` CSS tokens), Vitest + Testing Library, `cargo test`.

## Global Constraints

Every task's requirements implicitly include this section. Values are copied verbatim from the spec (`docs/superpowers/specs/2026-07-03-sources-ingest-lifecycle-design.md`).

- **Requires daemon ≥ v0.10.0.** Live daemon is `0.9.5`; latest release is `v0.11.0`; backend HEAD is `v0.10.1`. Never restart, kill, or update the daemon on `:7878` from code — that is the user's action.
- **Version floor read from `GET /api/health` only** — field `HealthResponse.version: String`. `/api/status` and `/api/config` do not carry it.
- **Managed uploads dir = `~/.wenlan/sources/`** (resolved as `dirs::home_dir().join(".wenlan/sources")` — note the leading dot; distinct from the app's own `~/Wenlan/knowledge`). Contains a self-gitignore file whose only content is `*`.
- **Version-gate treatment = warn + updater link, non-blocking** (registration still allowed on an old daemon; UI is honest that nothing indexes). Copy: `Your daemon needs an update to index files.`
- **Supported extensions: `pdf`, `md`, `txt` only.** The upload picker filter must match this set.
- **Register-once id for the managed dir is `directory-sources`.** The daemon dedupes by path; a repeat `POST /api/sources` returns the string `Source already registered`, which the app treats as success (check-or-ignore), never an error.
- **DELETE keeps memory rows by design.** Removed sources stop syncing but indexed notes persist. Confirm copy: `Remove Books? Indexed notes stay in your library; this source stops syncing.`
- **No em-dashes in any user-facing string.** Use commas or periods.
- **`--mem-*` tokens only, no new colors.** Fonts: Fraunces (`--mem-font-heading`), body (`--mem-font-body`), mono (`--mem-font-mono`). Reuse `--mem-shimmer-color` + `@keyframes mem-shimmer` (already in `src/index.css`).
- **CI is strict (`ci.yml`): `cargo fmt --check`, `cargo clippy --workspace --all-targets -- -D warnings`, `cargo test`, `pnpm exec tsc -b`, `pnpm test`.** Any unused function/param after the directory pivot must be handled (`#[allow(dead_code)]` + comment, or `_`-prefixed param) or clippy fails the build.
- **User-facing copy strings** come verbatim from the spec's Copy table (§Frontend / Copy).

**Two defaulted product decisions** (surfaced for one-tap override; both baked into this plan): managed dir name `~/.wenlan/sources/`, and non-blocking warn+link version gate. If the user overrides either, only Task 2 (dir name) or Task 4 (gate affordance) changes.

---

### Task 1: Daemon version gate — command + floor helper

**Files:**
- Modify: `app/src/search.rs` (add `daemon_version` command near the other source commands, ~line 3547)
- Modify: `app/src/lib.rs:800` (register `search::daemon_version` in `generate_handler!`)
- Modify: `src/lib/tauri.ts` (add `getDaemonVersion` + `daemonMeetsFloor`, after the Registered Sources block ~line 177)
- Test: `src/lib/tauri.test.ts` (Vitest for `daemonMeetsFloor`)

**Interfaces:**
- Produces: Rust command `daemon_version() -> Result<String, String>`; TS `getDaemonVersion(): Promise<string>`; TS `daemonMeetsFloor(version: string, floor?: string): boolean`.

- [ ] **Step 1: Write the failing test**

Add to `src/lib/tauri.test.ts`:

```ts
import { daemonMeetsFloor } from "./tauri";

describe("daemonMeetsFloor", () => {
  it("rejects the live 0.9.5 daemon (below floor)", () => {
    expect(daemonMeetsFloor("0.9.5")).toBe(false);
  });
  it("accepts exactly the floor and above", () => {
    expect(daemonMeetsFloor("0.10.0")).toBe(true);
    expect(daemonMeetsFloor("0.10.1")).toBe(true);
    expect(daemonMeetsFloor("0.11.0")).toBe(true);
    expect(daemonMeetsFloor("1.0.0")).toBe(true);
  });
  it("tolerates a pre-release suffix", () => {
    expect(daemonMeetsFloor("0.11.0-rc.1")).toBe(true);
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pnpm test -- src/lib/tauri.test.ts -t daemonMeetsFloor`
Expected: FAIL with "daemonMeetsFloor is not a function".

- [ ] **Step 3: Write minimal implementation**

Add to `src/lib/tauri.ts` (after `syncRegisteredSource`, ~line 177):

```ts
/** Read the running daemon's version string from GET /api/health. */
export async function getDaemonVersion(): Promise<string> {
  return invoke("daemon_version");
}

/**
 * True when the daemon version is >= floor. Daemon-native file ingest landed
 * in 0.10.0; below that, files register but never index (§0 version gate).
 */
export function daemonMeetsFloor(version: string, floor = "0.10.0"): boolean {
  const parse = (v: string) => v.split(".").map((n) => parseInt(n, 10) || 0);
  const a = parse(version);
  const b = parse(floor);
  for (let i = 0; i < 3; i++) {
    const d = (a[i] ?? 0) - (b[i] ?? 0);
    if (d !== 0) return d > 0;
  }
  return true;
}
```

Add the Rust command to `app/src/search.rs` (after `sync_registered_source`, ~line 3547):

```rust
#[tauri::command]
pub async fn daemon_version(state: tauri::State<'_, State>) -> Result<String, String> {
    let client = {
        let s = state.read().await;
        s.client.clone()
    };
    Ok(client.health().await?.version)
}
```

Register it in `app/src/lib.rs` `generate_handler!` (in the source-commands run, next to `search::sync_registered_source`):

```rust
            search::daemon_version,
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pnpm test -- src/lib/tauri.test.ts -t daemonMeetsFloor`
Expected: PASS (5 assertions).

- [ ] **Step 5: Commit**

```bash
git add src/lib/tauri.ts src/lib/tauri.test.ts app/src/search.rs app/src/lib.rs
git commit -m "feat(sources): daemon version gate command + floor helper"
```

---

### Task 2: Managed uploads dir — atomic file placement

**Files:**
- Create: `app/src/sources/uploads.rs`
- Modify: `app/src/sources/mod.rs` (add `pub mod uploads;`)
- Test: inline `#[cfg(test)]` in `app/src/sources/uploads.rs`

**Interfaces:**
- Consumes: nothing (leaf module).
- Produces: `sources::uploads::sources_dir() -> PathBuf`; `sources::uploads::place_upload_file(sources_dir: &Path, src: &Path) -> std::io::Result<PathBuf>`.

- [ ] **Step 1: Write the failing test**

Create `app/src/sources/uploads.rs` with only the test first:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn place_upload_is_atomic_and_gitignored() {
        let tmp = tempfile::tempdir().unwrap();
        let sources = tmp.path().join("sources");
        let src = tmp.path().join("paper.pdf");
        std::fs::write(&src, b"%PDF-1.4 body").unwrap();

        let dest = place_upload_file(&sources, &src).unwrap();

        assert_eq!(dest, sources.join("paper.pdf"));
        assert_eq!(std::fs::read(&dest).unwrap(), b"%PDF-1.4 body");
        // self-gitignore keeps blobs out of the daemon's git home
        assert_eq!(std::fs::read_to_string(sources.join(".gitignore")).unwrap(), "*\n");
        // no temp file left behind
        let leftover: Vec<_> = std::fs::read_dir(&sources)
            .unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| e.file_name().to_string_lossy().ends_with(".tmp"))
            .collect();
        assert!(leftover.is_empty(), "temp file must be renamed away");
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd app && cargo test -p wenlan-app sources::uploads::tests::place_upload_is_atomic_and_gitignored`
Expected: FAIL to compile — `place_upload_file` not defined.

- [ ] **Step 3: Write minimal implementation**

Prepend to `app/src/sources/uploads.rs` (above the test module):

```rust
// SPDX-License-Identifier: AGPL-3.0-only
//! Managed home for loose uploaded files (§2). The daemon indexes files in
//! place and stores no blobs, so the ONLY copies the app makes are loose
//! single-file uploads staged here for the daemon's 30s scheduler.

use std::fs;
use std::path::{Path, PathBuf};

/// `~/.wenlan/sources` — aligned under the daemon's knowledge home, next to
/// `pages/`.
/// ponytail: hardcoded to `~/.wenlan`; the daemon's knowledge_path can in
/// principle be customized but is not exposed via any API today. Verified
/// safe: this subdir is not a reserved ingest root and indexes normally.
pub fn sources_dir() -> PathBuf {
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".wenlan/sources")
}

/// Atomically place `src` into `sources_dir`: copy to a temp name on the same
/// filesystem, then `rename` into place, so the daemon's 30s scheduler never
/// sees a partially written file. Ensures the dir and its self-gitignore exist.
pub fn place_upload_file(sources_dir: &Path, src: &Path) -> std::io::Result<PathBuf> {
    fs::create_dir_all(sources_dir)?;
    let gitignore = sources_dir.join(".gitignore");
    if !gitignore.exists() {
        fs::write(&gitignore, "*\n")?;
    }
    let name = src
        .file_name()
        .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::InvalidInput, "source has no file name"))?;
    let dest = sources_dir.join(name);
    let tmp = sources_dir.join(format!(".{}.tmp", name.to_string_lossy()));
    fs::copy(src, &tmp)?;
    fs::rename(&tmp, &dest)?;
    Ok(dest)
}
```

Add to `app/src/sources/mod.rs` (with the other `pub mod` lines):

```rust
pub mod uploads;
```

Confirm `tempfile` is a dev-dependency in `app/Cargo.toml` (it is already used by `config.rs` tests). If missing, add under `[dev-dependencies]`: `tempfile = "3"`.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd app && cargo test -p wenlan-app sources::uploads::tests::place_upload_is_atomic_and_gitignored`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add app/src/sources/uploads.rs app/src/sources/mod.rs
git commit -m "feat(sources): atomic managed-dir placement for loose uploads"
```

---

### Task 3: Directory sources → daemon (retire the watcher for new sources)

**Files:**
- Modify: `app/src/search.rs` — `add_source` `"directory"` arm (`:3383`), add `register_directory_source_with_daemon` + `already_registered` helpers; mark `add_directory_source` (`:3388`) legacy; add `upload_source_file` command
- Modify: `app/src/lib.rs:800` (register `search::upload_source_file`)
- Modify: `src/lib/tauri.ts` (add `uploadSourceFile`)
- Test: inline `#[cfg(test)]` in `app/src/search.rs` for `already_registered`

**Interfaces:**
- Consumes: `api.rs` `client.add_source(source_type, path)`, `client.list_sources()`; `sources::uploads::{sources_dir, place_upload_file}` (Task 2).
- Produces: Rust `register_directory_source_with_daemon(state, path: &Path) -> Result<Source, String>`; `already_registered(err: &str) -> bool`; command `upload_source_file(path: String) -> Result<Source, String>`; TS `uploadSourceFile(path: string): Promise<RegisteredSource>`.

- [ ] **Step 1: Write the failing test**

Add to the `#[cfg(test)]` module in `app/src/search.rs`:

```rust
    #[test]
    fn already_registered_matches_daemon_dedupe_string() {
        assert!(super::already_registered("Source already registered"));
        assert!(super::already_registered("ValidationError: Source already registered for path"));
        assert!(!super::already_registered("Path does not exist"));
        assert!(!super::already_registered("connection refused"));
    }
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd app && cargo test -p wenlan-app already_registered_matches_daemon_dedupe_string`
Expected: FAIL to compile — `already_registered` not defined.

- [ ] **Step 3: Write minimal implementation**

Add helpers to `app/src/search.rs` (above `add_source`, ~line 3352):

```rust
/// The daemon dedupes sources by path; a repeat POST returns this string. The
/// app treats it as success (check-or-ignore), not an error path (§2).
fn already_registered(err: &str) -> bool {
    err.contains("Source already registered")
}

/// Register a directory (folder in place, or the managed uploads dir) with the
/// daemon, which owns ingestion (§1, §6). On repeat registration the daemon
/// returns "Source already registered" — resolve the existing source instead
/// of erroring.
async fn register_directory_source_with_daemon(
    state: &tauri::State<'_, State>,
    path: &std::path::Path,
) -> Result<crate::sources::Source, String> {
    let client = {
        let s = state.read().await;
        s.client.clone()
    };
    let path_str = path.to_string_lossy().to_string();
    match client.add_source("directory".to_string(), path_str).await {
        Ok(source) => Ok(source),
        Err(e) if already_registered(&e) => client
            .list_sources()
            .await?
            .into_iter()
            .find(|s| s.path == path)
            .ok_or_else(|| "source registered but not returned by daemon".to_string()),
        Err(e) => Err(e),
    }
}
```

Replace the `"directory"` arm in `add_source` (`:3383`) with:

```rust
        "directory" => register_directory_source_with_daemon(&state, &path_buf).await,
```

Change the `add_source` command's watcher param to unused (the directory arm no longer touches the watcher; Tauri still injects it by type):

```rust
    _watcher: tauri::State<'_, WatcherState>,
```

Mark `add_directory_source` as legacy so clippy's `-D warnings` (dead_code) passes:

```rust
// ponytail: legacy bridge for pre-v0.10.0 daemons only; new directory sources
// go straight to the daemon (register_directory_source_with_daemon). Remove
// once the minimum supported daemon is raised to v0.10.0 (§6).
#[allow(dead_code)]
async fn add_directory_source(
```

Add the upload command (after `daemon_version`, from Task 1):

```rust
#[tauri::command]
pub async fn upload_source_file(
    state: tauri::State<'_, State>,
    path: String,
) -> Result<crate::sources::Source, String> {
    let src = std::path::PathBuf::from(&path);
    if !src.is_file() {
        return Err(format!("Not a file: {}", path));
    }
    let dir = crate::sources::uploads::sources_dir();
    crate::sources::uploads::place_upload_file(&dir, &src).map_err(|e| e.to_string())?;
    register_directory_source_with_daemon(&state, &dir).await
}
```

Register in `app/src/lib.rs`:

```rust
            search::upload_source_file,
```

Add to `src/lib/tauri.ts` (next to `addSource`):

```ts
/** Stage a loose file into the managed dir and ensure it is a daemon source. */
export async function uploadSourceFile(path: string): Promise<RegisteredSource> {
  return invoke("upload_source_file", { path });
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd app && cargo test -p wenlan-app already_registered_matches_daemon_dedupe_string && cargo clippy -p wenlan-app --all-targets -- -D warnings`
Expected: test PASS; clippy clean (no dead_code/unused_variables from the pivot).

- [ ] **Step 5: Commit**

```bash
git add app/src/search.rs app/src/lib.rs src/lib/tauri.ts
git commit -m "feat(sources): route directory sources to the daemon; add upload command"
```

---

### Task 4: Add menu (folder / files) + version-gate notice

**Files:**
- Create: `src/components/memory/sources/AddSourceMenu.tsx`
- Modify: `src/components/memory/SourcesView.tsx` — swap the `+ Add source` button (`:210`) and `EmptyShelf` action (`:595`) to open the menu; render `<AddSourceMenu>` instead of `<AddSourceDialog>` directly
- Modify: `src/components/memory/sources/AddSourceDialog.tsx` — drop the auto-sync-on-add (`:32-36`), which now hard-errors is gone but daemon auto-syncs anyway
- Test: `src/components/memory/sources/__tests__/AddSourceMenu.test.tsx`

**Interfaces:**
- Consumes: `uploadSourceFile`, `getDaemonVersion`, `daemonMeetsFloor`, `openFile` (Tasks 1, 3); `AddSourceDialog`.
- Produces: `AddSourceMenu({ onClose }: { onClose: () => void })`.

- [ ] **Step 1: Write the failing test**

Create `src/components/memory/sources/__tests__/AddSourceMenu.test.tsx`:

```tsx
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { vi, describe, it, expect, beforeEach } from "vitest";
import AddSourceMenu from "../AddSourceMenu";

const mocks = vi.hoisted(() => ({
  getDaemonVersion: vi.fn(),
  uploadSourceFile: vi.fn(),
  openFile: vi.fn(),
  openDialog: vi.fn(),
}));

vi.mock("../../../../lib/tauri", () => ({
  getDaemonVersion: mocks.getDaemonVersion,
  daemonMeetsFloor: (v: string) => v.split(".").map(Number)[1] >= 10,
  uploadSourceFile: mocks.uploadSourceFile,
  openFile: mocks.openFile,
}));
vi.mock("@tauri-apps/plugin-dialog", () => ({ open: mocks.openDialog }));
vi.mock("sonner", () => ({ toast: vi.fn() }));

function wrap(ui: React.ReactNode) {
  return <QueryClientProvider client={new QueryClient()}>{ui}</QueryClientProvider>;
}

describe("AddSourceMenu", () => {
  beforeEach(() => vi.clearAllMocks());

  it("shows both entry points on a current daemon", async () => {
    mocks.getDaemonVersion.mockResolvedValue("0.11.0");
    render(wrap(<AddSourceMenu onClose={() => {}} />));
    await screen.findByText("Add a folder");
    expect(screen.getByText("Add files")).toBeInTheDocument();
    expect(screen.queryByText("Your daemon needs an update to index files.")).toBeNull();
  });

  it("shows the update-daemon notice on an old daemon", async () => {
    mocks.getDaemonVersion.mockResolvedValue("0.9.5");
    render(wrap(<AddSourceMenu onClose={() => {}} />));
    await screen.findByText("Your daemon needs an update to index files.");
  });

  it("picking Add files stages the chosen file", async () => {
    mocks.getDaemonVersion.mockResolvedValue("0.11.0");
    mocks.openDialog.mockResolvedValue("/Users/me/paper.pdf");
    mocks.uploadSourceFile.mockResolvedValue({ id: "directory-sources" });
    render(wrap(<AddSourceMenu onClose={() => {}} />));
    fireEvent.click(await screen.findByText("Add files"));
    await waitFor(() => expect(mocks.uploadSourceFile).toHaveBeenCalledWith("/Users/me/paper.pdf"));
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pnpm test -- src/components/memory/sources/__tests__/AddSourceMenu.test.tsx`
Expected: FAIL — cannot find module `../AddSourceMenu`.

- [ ] **Step 3: Write minimal implementation**

Create `src/components/memory/sources/AddSourceMenu.tsx`:

```tsx
// SPDX-License-Identifier: AGPL-3.0-only
import { useState } from "react";
import { open as openDialog } from "@tauri-apps/plugin-dialog";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { toast } from "sonner";
import {
  uploadSourceFile,
  getDaemonVersion,
  daemonMeetsFloor,
  openFile,
} from "../../../lib/tauri";
import AddSourceDialog from "./AddSourceDialog";

export default function AddSourceMenu({ onClose }: { onClose: () => void }) {
  const qc = useQueryClient();
  const [showFolder, setShowFolder] = useState(false);

  const { data: version } = useQuery({ queryKey: ["daemonVersion"], queryFn: getDaemonVersion });
  // Optimistic until the version is known, so the menu never flickers a warning.
  const ready = version === undefined ? true : daemonMeetsFloor(version);

  const upload = useMutation({
    mutationFn: async () => {
      const picked = await openDialog({
        directory: false,
        multiple: false,
        filters: [{ name: "Documents", extensions: ["pdf", "md", "txt"] }],
      });
      if (!picked || typeof picked !== "string") return null;
      const name = picked.split("/").pop() ?? "file";
      await uploadSourceFile(picked);
      // Toast idiom: eyebrow (title) / heading + body (description).
      toast("Added", { description: `${name} is on the shelf. Indexing in the background.` });
      return name;
    },
    onSuccess: (name) => {
      if (!name) return;
      qc.invalidateQueries({ queryKey: ["registeredSources"] });
      onClose();
    },
  });

  if (showFolder) {
    return <AddSourceDialog onClose={onClose} onSuccess={onClose} />;
  }

  const item = "w-full text-left rounded-md px-3 py-2 text-sm hover:bg-[var(--mem-hover)]";
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50" onClick={onClose}>
      <div
        className="w-[22rem] rounded-lg bg-[var(--mem-surface)] p-4 shadow-xl"
        onClick={(e) => e.stopPropagation()}
      >
        <h3 className="text-sm font-medium text-[var(--mem-text)] mb-2" style={{ fontFamily: "var(--mem-font-heading)" }}>
          Add a source
        </h3>
        {!ready && (
          <div className="mb-3 rounded-md p-3 text-xs" style={{ background: "var(--mem-hover)", color: "var(--mem-text-secondary)" }}>
            <p className="mb-2">Your daemon needs an update to index files.</p>
            <button className="underline" style={{ color: "var(--mem-accent-indigo)" }} onClick={() => openFile("https://wenlan.app")}>
              Update Wenlan
            </button>
          </div>
        )}
        <button className={item} style={{ color: "var(--mem-text)" }} onClick={() => setShowFolder(true)}>
          Add a folder
        </button>
        <button
          className={item}
          style={{ color: "var(--mem-text)" }}
          disabled={upload.isPending}
          onClick={() => upload.mutate()}
        >
          {upload.isPending ? "Adding…" : "Add files"}
        </button>
      </div>
    </div>
  );
}
```

In `src/components/memory/SourcesView.tsx`: replace the two `AddSourceDialog` render sites (`:102-104` and `:256-258`) and the `EmptyShelf` block usage with `AddSourceMenu`, and update the import (`:13`):

```tsx
import AddSourceMenu from "./sources/AddSourceMenu";
```

Both render sites become:

```tsx
{adding && <AddSourceMenu onClose={() => setAdding(false)} />}
```

In `src/components/memory/sources/AddSourceDialog.tsx`, delete the auto-sync-on-add so a directory add does not fire the (now removed) manual sync — the daemon auto-syncs within 30s. Replace `onSuccess` (`:30-37`) with:

```tsx
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["registeredSources"] });
      onSuccess();
    },
```

Remove the now-unused `syncRegisteredSource` import in `AddSourceDialog.tsx:5`.

- [ ] **Step 4: Run test to verify it passes**

Run: `pnpm test -- src/components/memory/sources/__tests__/AddSourceMenu.test.tsx`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add src/components/memory/sources/AddSourceMenu.tsx src/components/memory/sources/__tests__/AddSourceMenu.test.tsx src/components/memory/SourcesView.tsx src/components/memory/sources/AddSourceDialog.tsx
git commit -m "feat(sources): two-entry Add menu with folder/files + version-gate notice"
```

---

### Task 5: Spine ingest states — the signature UX

**Files:**
- Modify: `src/components/memory/SourcesView.tsx` — add `spineVisual`/`spineCaption` helpers, prev-count ref, dynamic `refetchInterval`, and spine JSX (`:163-199`)
- Modify: `src/index.css` — add a spine-fill rise animation keyframe (reuse `--mem-shimmer-color`)
- Test: `src/components/memory/__tests__/SourcesView.test.tsx` (extend existing)

**Interfaces:**
- Consumes: `RegisteredSource` (has `last_sync`, `memory_count`, `last_sync_errors`).
- Produces: `spineVisual(s, prevMemoryCount): "ghost" | "indexing" | "settled"`; `spineCaption(s): string`.

- [ ] **Step 1: Write the failing test**

Add to `src/components/memory/__tests__/SourcesView.test.tsx` (import the helpers once they exist):

```tsx
import { spineVisual, spineCaption } from "../SourcesView";

const base = { id: "s", source_type: "directory" as const, path: "/x/Books", status: "Active" as const };

describe("spineVisual", () => {
  it("ghost before any memories arrive", () => {
    expect(spineVisual({ ...base, last_sync: null, memory_count: 0 }, undefined)).toBe("ghost");
  });
  it("indexing while last_sync is null and memories exist", () => {
    expect(spineVisual({ ...base, last_sync: null, memory_count: 4 }, undefined)).toBe("indexing");
  });
  it("indexing while the count is still climbing between polls", () => {
    expect(spineVisual({ ...base, last_sync: 100, memory_count: 20 }, 12)).toBe("indexing");
  });
  it("settled once synced and the count is stable", () => {
    expect(spineVisual({ ...base, last_sync: 100, memory_count: 20 }, 20)).toBe("settled");
  });
});

describe("spineCaption", () => {
  it("says Indexing while settling", () => {
    expect(spineCaption({ ...base, last_sync: null, memory_count: 0 })).toBe("Indexing…");
  });
  it("shows notes when settled", () => {
    expect(spineCaption({ ...base, last_sync: 100, memory_count: 142 })).toBe("142 notes");
  });
  it("appends skipped count", () => {
    expect(spineCaption({ ...base, last_sync: 100, memory_count: 142, last_sync_errors: 2 })).toBe("142 notes, 2 skipped");
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pnpm test -- src/components/memory/__tests__/SourcesView.test.tsx -t spine`
Expected: FAIL — `spineVisual`/`spineCaption` are not exported.

- [ ] **Step 3: Write minimal implementation**

Add exported helpers near the top of `src/components/memory/SourcesView.tsx` (after `spineHeight`, ~line 46):

```tsx
export type SpineVisual = "ghost" | "indexing" | "settled";

/** Ingest state for the spine (§Signature). Determinate percent is impossible
 *  (daemon reports no per-source total); the fill means "still arriving". */
export function spineVisual(s: RegisteredSource, prevMemoryCount: number | undefined): SpineVisual {
  if (s.last_sync === null) return s.memory_count === 0 ? "ghost" : "indexing";
  if (prevMemoryCount !== undefined && s.memory_count > prevMemoryCount) return "indexing";
  return "settled";
}

/** Mono caption under a source: Indexing… while settling, else "N notes" (+ skipped). */
export function spineCaption(s: RegisteredSource): string {
  if (s.last_sync === null) return "Indexing…";
  const skipped = s.last_sync_errors ?? 0;
  const notes = `${s.memory_count.toLocaleString()} notes`;
  return skipped > 0 ? `${notes}, ${skipped} skipped` : notes;
}
```

In `SourcesView`, track previous counts and drive polling by settling state. Replace the query (`:68-72`) with:

```tsx
  const prevCounts = useRef<Record<string, number>>({});

  const { data: fetchedSources = [] } = useQuery({
    queryKey: ["registeredSources"],
    queryFn: listRegisteredSources,
    // Refetch fast while anything is still settling; slow when idle (§3).
    refetchInterval: (q) => {
      const list = (q.state.data as RegisteredSource[] | undefined) ?? [];
      return list.some((s) => s.last_sync === null) ? 3000 : 10000;
    },
  });

  useEffect(() => {
    const next: Record<string, number> = {};
    for (const s of fetchedSources) next[s.id] = s.memory_count;
    prevCounts.current = next;
  }, [fetchedSources]);
```

Add `useRef, useEffect` to the React import (`:2`).

Replace the spine `<span>` (`:164-175`) so the fill and caption reflect `spineVisual`. The rail spine keeps `spineHeight` for silhouette; add the fill state via class + inline reduced-motion guard:

```tsx
                {(() => {
                  const visual = spineVisual(s, prevCounts.current[s.id]);
                  const h = spineHeight(s.memory_count, maxMemories);
                  return (
                    <span className="shrink-0 w-1.5 flex items-end justify-center" style={{ height: 30 }}>
                      <span
                        data-testid="source-spine"
                        data-visual={visual}
                        className={visual === "indexing" ? "spine-indexing" : undefined}
                        style={{
                          width: 3,
                          borderRadius: 1.5,
                          height: visual === "ghost" ? 30 : h,
                          backgroundColor: label ? STATUS_COLORS[label] : "var(--mem-accent-indigo)",
                          border: visual === "ghost" ? "1px solid var(--mem-accent-indigo)" : undefined,
                          background: visual === "ghost" ? "transparent" : undefined,
                          opacity: visual === "ghost" ? 0.5 : label ? 0.9 : active ? 0.85 : 0.5,
                        }}
                      />
                    </span>
                  );
                })()}
```

Replace the sub-label (`:189-199`) to use `spineCaption` for directory sources (keep the status label precedence):

```tsx
                    {label ?? spineCaption(s)}
```

Add to `src/index.css` (near `@keyframes mem-shimmer`, ~line 216):

```css
/* Spine "still arriving" fill: a soft indeterminate glow, not a percentage (§3). */
.spine-indexing {
  animation: mem-shimmer 1.6s ease-in-out infinite;
}
@media (prefers-reduced-motion: reduce) {
  .spine-indexing {
    animation: none;
  }
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pnpm test -- src/components/memory/__tests__/SourcesView.test.tsx -t spine`
Expected: PASS (7 assertions across the two describes).

- [ ] **Step 5: Commit**

```bash
git add src/components/memory/SourcesView.tsx src/components/memory/__tests__/SourcesView.test.tsx src/index.css
git commit -m "feat(sources): spine encodes ingest state (ghost/indexing/settled)"
```

---

### Task 6: Sync-button fix + auto-sync state + autonomy line

**Files:**
- Modify: `src/components/memory/SourcesView.tsx` — `FolderBrowser` header actions (`:388-405`) and shelf header (`:116-138`)
- Test: `src/components/memory/__tests__/SourcesView.test.tsx` (extend)

**Interfaces:**
- Consumes: `RegisteredSource.source_type`, `.last_sync`; `relTime` (existing).
- Produces: no new exports; behavior only.

- [ ] **Step 1: Write the failing test**

Add to `src/components/memory/__tests__/SourcesView.test.tsx` (render-level; mock `listRegisteredSources`):

```tsx
import { render, screen, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import SourcesView from "../SourcesView";

vi.mock("../../../lib/tauri", async (orig) => ({
  ...(await orig<typeof import("../../../lib/tauri")>()),
  listRegisteredSources: vi.fn(),
  readSourceDir: vi.fn().mockResolvedValue([]),
}));
import { listRegisteredSources } from "../../../lib/tauri";

function view() {
  return render(
    <QueryClientProvider client={new QueryClient()}>
      <SourcesView onManageSources={() => {}} />
    </QueryClientProvider>,
  );
}

describe("sync affordance by source type", () => {
  it("directory sources show auto-synced state, no manual Sync button", async () => {
    (listRegisteredSources as vi.Mock).mockResolvedValue([
      { id: "directory-books", source_type: "directory", path: "/x/Books", status: "Active", last_sync: 1_700_000_000, file_count: 3, memory_count: 42 },
    ]);
    view();
    await waitFor(() => expect(screen.queryByRole("button", { name: "Sync" })).toBeNull());
    expect(screen.getByText(/Auto-synced/)).toBeInTheDocument();
    expect(screen.getByText("Syncs in the background, even when Wenlan is closed.")).toBeInTheDocument();
  });

  it("obsidian sources keep the manual Sync button", async () => {
    (listRegisteredSources as vi.Mock).mockResolvedValue([
      { id: "obsidian-vault", source_type: "obsidian", path: "/x/Vault", status: "Active", last_sync: null, file_count: 3, memory_count: 0 },
    ]);
    view();
    await waitFor(() => expect(screen.getByRole("button", { name: "Sync" })).toBeInTheDocument());
  });
});
```

Note: the `import.meta.env.DEV` preview override at `SourcesView.tsx:73-79` must already be reverted (Task 8) for this test to use mocked data. If running this task before Task 8, temporarily guard the test by setting the sources through the mock only (the DEV block short-circuits it) — prefer doing Task 8's revert of that block first. **Do Task 8's SourcesView revert before this test.**

- [ ] **Step 2: Run test to verify it fails**

Run: `pnpm test -- src/components/memory/__tests__/SourcesView.test.tsx -t "sync affordance"`
Expected: FAIL — directory source still renders a Sync button; autonomy line absent.

- [ ] **Step 3: Write minimal implementation**

In `FolderBrowser`, gate the Sync button on Obsidian and show auto-sync state for directories. Replace the Sync `<button>` (`:388-405`) with:

```tsx
            {source.source_type === "obsidian" ? (
              <button
                onClick={() => syncMutation.mutate()}
                disabled={syncMutation.isPending}
                className="rounded-md transition-colors duration-150"
                style={{
                  padding: "6px 13px",
                  fontFamily: "var(--mem-font-body)",
                  fontSize: "12px",
                  fontWeight: 500,
                  color: "white",
                  background: "var(--mem-accent-indigo)",
                  border: "none",
                  cursor: syncMutation.isPending ? "default" : "pointer",
                  opacity: syncMutation.isPending ? 0.6 : 1,
                }}
              >
                {syncedFlash ? "✓ Synced" : syncMutation.isPending ? "Syncing…" : "Sync"}
              </button>
            ) : (
              <span
                style={{
                  fontFamily: "var(--mem-font-mono)",
                  fontSize: "11px",
                  color: "var(--mem-text-tertiary)",
                }}
              >
                Auto-synced{source.last_sync ? ` · updated ${relTime(source.last_sync).replace(/^synced /, "")}` : ""}
              </span>
            )}
```

Add the autonomy line under the shelf header (after the `sources.length … sources` count block, ~`:138`), shown only when at least one directory source exists:

```tsx
          {sources.some((s) => s.source_type === "directory") && (
            <div
              style={{
                fontFamily: "var(--mem-font-body)",
                fontSize: "11px",
                color: "var(--mem-text-tertiary)",
                marginTop: 8,
                lineHeight: 1.4,
              }}
            >
              Syncs in the background, even when Wenlan is closed.
            </div>
          )}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pnpm test -- src/components/memory/__tests__/SourcesView.test.tsx -t "sync affordance"`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add src/components/memory/SourcesView.tsx src/components/memory/__tests__/SourcesView.test.tsx
git commit -m "feat(sources): auto-synced state for directories, keep Sync for Obsidian"
```

---

### Task 7: Inline Remove + upload blob cleanup

**Files:**
- Modify: `src/components/memory/SourcesView.tsx` — add a Remove control + confirm in `FolderBrowser` header (`:371-406`)
- Modify: `app/src/search.rs` — `remove_directory_source` (`:3465`) deletes managed blobs; add `managed_blob_paths` helper
- Test: `src/components/memory/__tests__/SourcesView.test.tsx` (extend) + `#[cfg(test)]` in `app/src/search.rs`

**Interfaces:**
- Consumes: `removeSource` (existing TS), `sources::uploads::sources_dir` (Task 2).
- Produces: Rust `managed_blob_paths(sources_dir: &Path, source: &Source) -> Vec<PathBuf>`; frontend Remove flow.

- [ ] **Step 1: Write the failing tests**

Rust — add to `#[cfg(test)]` in `app/src/search.rs`:

```rust
    #[test]
    fn managed_blob_paths_targets_only_the_managed_dir() {
        let sources_dir = std::path::Path::new("/home/u/.wenlan/sources");
        let managed = crate::sources::Source {
            id: "directory-sources".into(),
            source_type: crate::sources::SourceType::Directory,
            path: sources_dir.to_path_buf(),
            status: crate::sources::SyncStatus::Active,
            last_sync: None,
            file_count: 0,
            memory_count: 0,
            last_sync_errors: 0,
            last_sync_error_detail: None,
        };
        // The managed dir itself is cleaned; an in-place folder source is not.
        assert_eq!(super::managed_blob_paths(sources_dir, &managed), vec![sources_dir.to_path_buf()]);

        let in_place = crate::sources::Source { path: "/home/u/Documents/Books".into(), ..managed.clone() };
        assert!(super::managed_blob_paths(sources_dir, &in_place).is_empty());
    }
```

Frontend — add to `SourcesView.test.tsx`:

```tsx
import { fireEvent } from "@testing-library/react";
import { removeSource } from "../../../lib/tauri";
// (add removeSource: vi.fn() to the tauri mock above)

it("Remove calls DELETE after confirm", async () => {
  vi.spyOn(window, "confirm").mockReturnValue(true);
  (listRegisteredSources as vi.Mock).mockResolvedValue([
    { id: "directory-books", source_type: "directory", path: "/x/Books", status: "Active", last_sync: 1, file_count: 1, memory_count: 5 },
  ]);
  view();
  fireEvent.click(await screen.findByRole("button", { name: "Remove" }));
  await waitFor(() => expect(removeSource).toHaveBeenCalledWith("directory-books"));
});
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd app && cargo test -p wenlan-app managed_blob_paths_targets_only_the_managed_dir`
Expected: FAIL — `managed_blob_paths` not defined.
Run: `pnpm test -- src/components/memory/__tests__/SourcesView.test.tsx -t "Remove calls DELETE"`
Expected: FAIL — no Remove button.

- [ ] **Step 3: Write minimal implementation**

Rust — add the helper and call it from `remove_directory_source`. Add near `already_registered`:

```rust
/// Blobs to delete on removal. Only the app-managed uploads dir holds copies;
/// in-place folder sources are never copied, so nothing to clean (§4).
fn managed_blob_paths(sources_dir: &std::path::Path, source: &crate::sources::Source) -> Vec<std::path::PathBuf> {
    if source.path == sources_dir {
        vec![sources_dir.to_path_buf()]
    } else {
        Vec::new()
    }
}
```

Extend `remove_directory_source` (`:3465`) to clean managed blobs after config removal:

```rust
    let sources_dir = crate::sources::uploads::sources_dir();
    for blob in managed_blob_paths(&sources_dir, &source) {
        let _ = std::fs::remove_dir_all(&blob); // best-effort; missing dir is fine
    }
```

Note: the daemon-side directory sources are removed via `client.remove_source` in the command's else branch. The managed `directory-sources` source is a daemon source (registered via the pivot), so ensure the cleanup also runs for it: after `client.remove_source(&id).await?` in `remove_source` (`:3462`), add the same managed-blob cleanup when the removed id is `directory-sources`. Concretely, replace the tail of `remove_source`:

```rust
    let client = {
        let s = state.read().await;
        s.client.clone()
    };
    client.remove_source(&id).await?;
    if id == "directory-sources" {
        let dir = crate::sources::uploads::sources_dir();
        let _ = std::fs::remove_dir_all(&dir); // managed uploads only
    }
    Ok(())
```

Frontend — add a Remove button to the `FolderBrowser` header actions (before Reveal, `:371`), wired to `removeSource` with a confirm:

```tsx
            <button
              onClick={() => {
                const name = folderName(source.path);
                if (!window.confirm(`Remove ${name}? Indexed notes stay in your library; this source stops syncing.`)) return;
                removeSource(source.id).then(() => {
                  queryClient.invalidateQueries({ queryKey: ["registeredSources"] });
                });
              }}
              className="rounded-md transition-colors duration-150 hover:bg-[var(--mem-hover)]"
              style={{
                padding: "6px 11px",
                fontFamily: "var(--mem-font-body)",
                fontSize: "12px",
                color: "var(--mem-text-secondary)",
                background: "transparent",
                border: "1px solid var(--mem-border)",
                cursor: "pointer",
              }}
            >
              Remove
            </button>
```

Add `removeSource` to the imports from `../../lib/tauri` in `SourcesView.tsx:4-12`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd app && cargo test -p wenlan-app managed_blob_paths_targets_only_the_managed_dir`
Expected: PASS.
Run: `pnpm test -- src/components/memory/__tests__/SourcesView.test.tsx -t "Remove calls DELETE"`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add app/src/search.rs src/components/memory/SourcesView.tsx src/components/memory/__tests__/SourcesView.test.tsx
git commit -m "feat(sources): inline Remove with confirm + managed-upload blob cleanup"
```

---

### Task 8: Pre-merge — revert dev previews + real-daemon integration test

**Files:**
- Modify: `src/components/memory/SourcesView.tsx:73-79` (remove the TEMP DEV preview block)
- Modify: `src/components/memory/Main.tsx:58` (remove the TEMP DEV initial-view override)
- Modify: `src/components/UpdaterDialog.tsx:24` (remove the TEMP DEV early return)
- Create: `app/tests/sources_integration.rs` (ignored-by-default integration test)

**Interfaces:**
- Consumes: nothing new.
- Produces: a runnable `#[ignore]` integration test documenting the real daemon contract.

- [ ] **Step 1: Revert the three dev-only TEMP edits**

In `src/components/memory/SourcesView.tsx`, delete the block at `:73-79` and use the fetched sources directly:

```tsx
  const sources: RegisteredSource[] = fetchedSources;
```

(or inline `fetchedSources` and drop the alias — keep whichever reads cleaner after Task 5's edits.)

In `src/components/memory/Main.tsx:57-58`, remove the DEV branch so the initial view is not forced to Sources:

```tsx
    : initialView === "import" ? { kind: "import" }
    : { kind: "..." /* the original non-DEV default that preceded the TEMP line */ };
```

Read `Main.tsx:52-62` first to restore the exact original default (the line the TEMP `import.meta.env.DEV ? { kind: "sources" }` replaced or preceded).

In `src/components/UpdaterDialog.tsx:24`, delete the line:

```tsx
  if (import.meta.env.DEV) return null; // TEMP (do not commit)
```

- [ ] **Step 2: Verify no TEMP markers remain**

Run: `git grep -n "do not commit\|TEMP (do not commit)" -- src/`
Expected: no output.

- [ ] **Step 3: Write the integration test**

Create `app/tests/sources_integration.rs`:

```rust
// SPDX-License-Identifier: AGPL-3.0-only
//! Exercises the actual daemon contract the whole Sources design rests on.
//! Requires a real daemon >= v0.10.0 on :7878 (build v0.11.0 locally from
//! ../wenlan). Ignored by default; run with:
//!   cargo test -p wenlan-app --test sources_integration -- --ignored --nocapture

#[tokio::test]
#[ignore = "requires a live daemon >= v0.10.0 on :7878"]
async fn directory_source_indexes_within_scheduler_window() {
    let client = reqwest::Client::new();
    let base = "http://127.0.0.1:7878";

    // Version floor.
    let health: serde_json::Value = client.get(format!("{base}/api/health")).send().await.unwrap().json().await.unwrap();
    let version = health["version"].as_str().unwrap();
    let parts: Vec<u32> = version.split('.').map(|n| n.parse().unwrap_or(0)).collect();
    assert!(parts[0] > 0 || parts[1] >= 10, "daemon {version} is below the v0.10.0 floor");

    // Register a temp dir with a markdown file.
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("note.md"), "# Hello\n\nintegration body").unwrap();
    let path = dir.path().to_string_lossy().to_string();
    let reg = client.post(format!("{base}/api/sources"))
        .json(&serde_json::json!({ "source_type": "directory", "path": path }))
        .send().await.unwrap();
    assert!(reg.status().is_success(), "register failed: {}", reg.status());

    // Poll until memory_count climbs and last_sync populates (scheduler is 30s).
    let mut indexed = false;
    for _ in 0..40 {
        tokio::time::sleep(std::time::Duration::from_secs(2)).await;
        let sources: serde_json::Value = client.get(format!("{base}/api/sources")).send().await.unwrap().json().await.unwrap();
        if let Some(s) = sources.as_array().and_then(|a| a.iter().find(|s| s["path"] == path)) {
            if s["last_sync"].as_i64().is_some() && s["memory_count"].as_i64().unwrap_or(0) > 0 {
                indexed = true;
                break;
            }
        }
    }
    assert!(indexed, "source did not index within the scheduler window");
}
```

Confirm `reqwest`, `tokio` (with `rt`/`macros`/`time`), `serde_json`, `tempfile` are available for tests in `app/Cargo.toml` (`reqwest`/`tokio`/`serde_json` are runtime deps; `tempfile` is a dev-dep). Add any missing to `[dev-dependencies]`.

- [ ] **Step 4: Verify the suite (integration test compiles, is skipped by default)**

Run: `cd app && cargo test -p wenlan-app --test sources_integration`
Expected: compiles; `1 ignored` (not run without `--ignored`).
Run full gates: `cargo fmt --check --all && cargo clippy --workspace --all-targets -- -D warnings && cd .. && pnpm exec tsc -b && pnpm test`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/components/memory/SourcesView.tsx src/components/memory/Main.tsx src/components/UpdaterDialog.tsx app/tests/sources_integration.rs
git commit -m "chore(sources): revert dev previews; add real-daemon integration test"
```

---

## Self-Review

**Spec coverage:**
- §0 version gate → Task 1 (command + floor helper) + Task 4 (notice UI). ✓
- §1 add source, two entry points → Task 3 (folder→daemon, upload command) + Task 4 (menu). ✓
- §2 managed path + register-once + gitignore → Task 2 (placement + gitignore) + Task 3 (`register_directory_source_with_daemon`, `already_registered`). ✓
- §3 ingest feedback (aggregate, shimmer, dynamic poll, skipped) → Task 5. ✓
- §4 remove + blob cleanup → Task 7. ✓
- §5 sync button fix → Task 6. ✓
- §6 retire watcher for new sources → Task 3 (pivot + `#[allow(dead_code)]` legacy). ✓
- §Signature spine states → Task 5. ✓
- §Moments (add, upload toast, autonomy line, remove confirm, auto-sync, empty) → Tasks 4, 5, 6, 7. ✓
- §Copy table → strings used verbatim across Tasks 4-7. ✓
- §Testing (Vitest, cargo, integration, revert dev edits) → every task's tests + Task 8. ✓
- Deferred/out-of-scope (provenance, per-file rows, watcher removal, knowledge-path API) → not planned, correct. ✓

**Placeholder scan:** No TBD/TODO. The one intentional read-before-edit is Task 8 Step 1 for `Main.tsx`'s original default (the exact prior value must be read from the file, not guessed) — flagged, not a placeholder.

**Type consistency:** `spineVisual`/`spineCaption` signatures identical between Task 5 def and its test. `register_directory_source_with_daemon`/`already_registered`/`managed_blob_paths` names consistent across Tasks 3 and 7. `uploadSourceFile`/`getDaemonVersion`/`daemonMeetsFloor` consistent across Tasks 1, 3, 4. `RegisteredSource` fields (`last_sync`, `memory_count`, `last_sync_errors`) match `src/lib/tauri.ts:140-150`.

**Ordering note:** Task 6 and Task 7 render-level tests need the `SourcesView.tsx:73-79` DEV preview block reverted first (it short-circuits mocked data). Do **Task 8 Step 1's SourcesView revert before Task 6.** The rest of Task 8 (Main.tsx, UpdaterDialog.tsx, integration test) can stay last.
