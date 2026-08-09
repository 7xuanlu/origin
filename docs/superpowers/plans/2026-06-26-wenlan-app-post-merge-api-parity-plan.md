# Wenlan App Post-Merge API Parity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Land the next `wenlan-app` parity slice after PR #2 by adding daemon-backed enrichment status, page-link, and revision-history consumers without starting the product/runtime identity rename.

**Architecture:** Keep the Tauri app as a thin client over the Wenlan daemon. Add typed Rust client methods first, expose them through Tauri commands, mirror the response shapes in `src/lib/tauri.ts`, then attach narrow UI consumers that hide unsupported/empty daemon surfaces rather than inventing app-local inference.

**Tech Stack:** Tauri 2, Rust, React 19, TypeScript, TanStack Query, Vitest, Cargo, CodeGraph (`npx -y @colbymchenry/codegraph`), ast-grep (`npx -y -p @ast-grep/cli sg`), rust-analyzer/compiler diagnostics, bounded `rg` fallback.

---

## Current Baseline

- Repo: `/Users/lucian/Repos/wenlan-app`
- Active worktree: `/Users/lucian/Repos/wenlan-app/.worktrees/wenlan-app-convergence`
- Current branch for this plan: `codex/wenlan-app-next-refactor`
- Base: `origin/main` after PR #2 merge commit `9d8d40df10f5e3a3d2bddca730b57aed73cc954b`
- Completed before this plan: typed client migration, sidecar bridge, MCP bridge, avatar migration, dock/app activation fix, Home pending-revision/refinery queue review lanes, daemon-backed setup/model/external LLM config reads and writes.
- Refreshed inventory on this branch:
  - frontend invoke calls: 126
  - registered Tauri commands: 170
  - Rust `origin_types` references: 0
  - runtime identity references: 222
  - stale taxonomy references: 239
  - source files under `app/src` and `src`: 151

## Tooling Protocol

Run this before each implementation task:

```bash
CODEGRAPH_TELEMETRY=0 DO_NOT_TRACK=1 npx -y @colbymchenry/codegraph sync .
CODEGRAPH_TELEMETRY=0 DO_NOT_TRACK=1 npx -y @colbymchenry/codegraph query WenlanClient --json
CODEGRAPH_TELEMETRY=0 DO_NOT_TRACK=1 npx -y @colbymchenry/codegraph impact WenlanClient --json
npx -y -p @ast-grep/cli sg outline app/src/api.rs
npx -y -p @ast-grep/cli sg outline app/src/search.rs
```

If CodeGraph fails, record the exact failure in the task note and continue with ast-grep, compiler diagnostics, and bounded `rg`. CodeGraph is advisory; tests and live daemon probes are the evidence gates.

Use this fail-closed daemon probe pattern for every live route check:

```bash
set -euo pipefail
curl -fsS "http://127.0.0.1:7878/api/health" | jq -e '.status == "ok" and (.version | startswith("0.9."))'
```

For ID-bearing probes, use `jq -er`, reject empty/null IDs, and assert the response envelope shape:

```bash
MEMORY_ID=$(curl -fsS "http://127.0.0.1:7878/api/memory/recent?limit=1" | jq -er '.[0].id')
test -n "${MEMORY_ID}" && test "${MEMORY_ID}" != "null"
export MEMORY_ID
curl -fsS "http://127.0.0.1:7878/api/memory/${MEMORY_ID}/enrichment-status" | jq -e '.source_id == env.MEMORY_ID and (.steps | type == "array")'
```

If the local daemon lacks representative memory/page data, record `live probe skipped: no representative data` and do not count that probe as passing.

## File Structure

- `app/src/api.rs`: typed daemon HTTP methods on `WenlanClient`; no Tauri state.
- `app/src/search.rs`: Tauri commands; clone `state.client` before `.await`.
- `app/src/lib.rs`: command registration only.
- `src/lib/tauri.ts`: public frontend wrappers and TypeScript response interfaces.
- `src/lib/tauri.test.ts`: wrapper argument-shape tests.
- `src/components/memory/MemoryDetail.tsx`: non-blocking enrichment status consumer.
- `src/components/memory/MemoryDetail.enrichment-status.test.tsx`: enrichment UI contract.
- `src/components/memory/PageDetail.tsx`: daemon page-link and page-revision consumer.
- `src/components/memory/PageDetail.links-revisions.test.tsx`: page-link and revision UI contract.
- `docs/superpowers/refactor/wenlan-app-parity-matrix.md`: update only when a route is implemented or explicitly deferred.

## Task 1: Enrichment Status Wrapper and Memory Detail Badge

**Files:**
- Modify: `app/src/api.rs`
- Modify: `app/src/search.rs`
- Modify: `app/src/lib.rs`
- Modify: `src/lib/tauri.ts`
- Modify: `src/lib/tauri.test.ts`
- Modify: `src/components/memory/MemoryDetail.tsx`
- Create: `src/components/memory/MemoryDetail.enrichment-status.test.tsx`

- [ ] **Step 1: Write the failing TypeScript wrapper test**

Add this block to `src/lib/tauri.test.ts`:

```ts
describe("enrichment status", () => {
  it("getEnrichmentStatus passes sourceId", async () => {
    mockInvoke.mockResolvedValue({
      source_id: "mem-1",
      summary: "complete",
      steps: [{ step: "classify", status: "done", error: null, attempts: 1 }],
    });

    const result = await tauri.getEnrichmentStatus("mem-1");

    expect(mockInvoke).toHaveBeenCalledWith("get_enrichment_status", {
      sourceId: "mem-1",
    });
    expect(result.summary).toBe("complete");
  });
});
```

Run:

```bash
pnpm vitest run src/lib/tauri.test.ts
```

Expected: FAIL with `tauri.getEnrichmentStatus is not a function`.

- [ ] **Step 2: Write the failing Rust client/command tests**

Add method-presence coverage to `app/src/api.rs` tests:

```rust
#[test]
fn wenlan_client_exposes_enrichment_status_method() {
    let _get_enrichment_status = WenlanClient::get_enrichment_status;
}

#[test]
fn enrichment_status_response_deserializes_daemon_payload() {
    let status: wenlan_types::EnrichmentStatusResponse =
        serde_json::from_value(serde_json::json!({
            "source_id": "mem-1",
            "summary": "complete",
            "steps": [
                { "step": "classify", "status": "done", "error": null, "attempts": 1 }
            ]
        }))
        .unwrap();

    assert_eq!(status.source_id, "mem-1");
    assert_eq!(status.steps[0].step, "classify");
}
```

Run:

```bash
cargo test -p origin-app --lib api::tests -- --nocapture
```

Expected: FAIL with missing `WenlanClient::get_enrichment_status`.

- [ ] **Step 3: Implement the typed Rust method and Tauri command**

In `app/src/api.rs`, add:

```rust
pub async fn get_enrichment_status(
    &self,
    source_id: &str,
) -> Result<wenlan_types::EnrichmentStatusResponse, String> {
    let path = format!("/api/memory/{}/enrichment-status", source_id);
    self.get_json(&path).await
}
```

In `app/src/search.rs`, add:

```rust
#[tauri::command]
pub async fn get_enrichment_status(
    state: tauri::State<'_, State>,
    source_id: String,
) -> Result<wenlan_types::EnrichmentStatusResponse, String> {
    let client = {
        let s = state.read().await;
        s.client.clone()
    };
    client.get_enrichment_status(&source_id).await
}
```

In `app/src/lib.rs`, register the command in the `tauri::generate_handler!` list:

```rust
search::get_enrichment_status,
```

- [ ] **Step 4: Implement the TypeScript response and wrapper**

In `src/lib/tauri.ts`, add:

```ts
export interface EnrichmentStepStatus {
  step: string;
  status: string;
  error?: string | null;
  attempts: number;
}

export interface EnrichmentStatusResponse {
  source_id: string;
  summary: string;
  steps: EnrichmentStepStatus[];
}

export async function getEnrichmentStatus(
  sourceId: string
): Promise<EnrichmentStatusResponse> {
  return invoke("get_enrichment_status", { sourceId });
}
```

Run:

```bash
pnpm vitest run src/lib/tauri.test.ts
cargo test -p origin-app --lib api::tests -- --nocapture
pnpm build
```

Expected: PASS.

- [ ] **Step 5: Write the failing MemoryDetail UI test**

Create `src/components/memory/MemoryDetail.enrichment-status.test.tsx`:

```tsx
// SPDX-License-Identifier: AGPL-3.0-only
import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import MemoryDetail from "./MemoryDetail";
import * as tauri from "../../lib/tauri";

vi.mock("../../lib/tauri");

const memory: tauri.MemoryItem = {
  source_id: "mem-1",
  title: "Memory",
  content: "A memory",
  summary: null,
  memory_type: "fact",
  domain: null,
  source_agent: null,
  confidence: null,
  confirmed: false,
  pinned: false,
  supersedes: null,
  last_modified: 1,
  chunk_count: 1,
};

function wrapper({ children }: { children: React.ReactNode }) {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return <QueryClientProvider client={qc}>{children}</QueryClientProvider>;
}

describe("MemoryDetail enrichment status", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.mocked(tauri.getMemoryDetail).mockResolvedValue(memory);
    vi.mocked(tauri.listSpaces).mockResolvedValue([]);
    vi.mocked(tauri.listEntities).mockResolvedValue([]);
    vi.mocked(tauri.listAllTags).mockResolvedValue({
      tags: [],
      document_tags: {},
      categories: [],
      document_categories: {},
    });
    vi.mocked(tauri.search).mockResolvedValue([]);
  });

  it("shows daemon enrichment status without blocking the memory body", async () => {
    vi.mocked(tauri.getEnrichmentStatus).mockResolvedValue({
      source_id: "mem-1",
      summary: "complete",
      steps: [{ step: "classify", status: "done", error: null, attempts: 1 }],
    });

    render(
      <MemoryDetail
        sourceId="mem-1"
        onBack={vi.fn()}
        onNavigateEntity={vi.fn()}
        onNavigateMemory={vi.fn()}
      />,
      { wrapper },
    );

    expect(await screen.findByText("A memory")).toBeInTheDocument();
    await waitFor(() => {
      expect(screen.getByText(/enrichment/i)).toBeInTheDocument();
      expect(screen.getByText(/complete/i)).toBeInTheDocument();
    });
  });

  it("keeps rendering the memory when enrichment status route is unavailable", async () => {
    vi.mocked(tauri.getEnrichmentStatus).mockRejectedValue(new Error("404"));

    render(
      <MemoryDetail
        sourceId="mem-1"
        onBack={vi.fn()}
        onNavigateEntity={vi.fn()}
        onNavigateMemory={vi.fn()}
      />,
      { wrapper },
    );

    expect(await screen.findByText("A memory")).toBeInTheDocument();
    expect(screen.queryByText(/enrichment/i)).toBeNull();
  });
});
```

Run:

```bash
pnpm vitest run src/components/memory/MemoryDetail.enrichment-status.test.tsx
```

Expected: FAIL because `MemoryDetail` does not call or render `getEnrichmentStatus`.

- [ ] **Step 6: Implement the non-blocking MemoryDetail consumer**

In `MemoryDetail.tsx`, import `getEnrichmentStatus` and add:

```tsx
const { data: enrichmentStatus } = useQuery({
  queryKey: ["enrichment-status", sourceId],
  queryFn: () => getEnrichmentStatus(sourceId),
  enabled: !!sourceId,
  staleTime: 30_000,
  retry: false,
});
```

Render this near the existing metadata chips, only when data is present:

```tsx
{enrichmentStatus && (
  <span title={enrichmentStatus.steps.map((s) => `${s.step}: ${s.status}`).join("\n")}>
    Enrichment {enrichmentStatus.summary}
  </span>
)}
```

Do not show an error banner when this optional route fails; detail rendering must remain usable on older daemons.

- [ ] **Step 7: Update parity matrix**

In `docs/superpowers/refactor/wenlan-app-parity-matrix.md`, update the `/api/memory/{source_id}/enrichment-status` row:

```markdown
| `/api/memory/{source_id}/enrichment-status` | `EnrichmentStatusResponse` | typed Rust/Tauri/TS wrapper present; Memory Detail shows non-blocking status when available | keep hidden on old daemons and do not block memory render | optional per-memory route; show unknown state if absent |
```

- [ ] **Step 8: Verify and commit Task 1**

Run:

```bash
pnpm vitest run src/lib/tauri.test.ts src/components/memory/MemoryDetail.enrichment-status.test.tsx
pnpm build
cargo test -p origin-app --lib api::tests -- --nocapture
cargo test -p origin-app --lib
```

Live daemon probe:

```bash
set -euo pipefail
curl -fsS "http://127.0.0.1:7878/api/health" | jq -e '.status == "ok" and (.version | startswith("0.9."))'
MEMORY_ID=$(curl -fsS "http://127.0.0.1:7878/api/memory/recent?limit=1" | jq -er '.[0].id')
test -n "${MEMORY_ID}" && test "${MEMORY_ID}" != "null"
export MEMORY_ID
curl -fsS "http://127.0.0.1:7878/api/memory/${MEMORY_ID}/enrichment-status" | jq -e '.source_id == env.MEMORY_ID and (.steps | type == "array")'
```

Commit:

```bash
git add app/src/api.rs app/src/search.rs app/src/lib.rs src/lib/tauri.ts src/lib/tauri.test.ts src/components/memory/MemoryDetail.tsx src/components/memory/MemoryDetail.enrichment-status.test.tsx docs/superpowers/refactor/wenlan-app-parity-matrix.md
git commit -m "fix: surface memory enrichment status"
```

## Task 2: Source Registry Daemon Ownership

**Files:**
- Modify: `app/src/api.rs`
- Modify: `app/src/search.rs`
- Modify: `app/src/lib.rs`
- Modify: `src/lib/tauri.ts`
- Modify: `src/lib/tauri.test.ts`
- Modify: `src/components/memory/sources/SourcesSection.tsx`
- Modify: `src/components/memory/sources/__tests__/SourcesSection.test.tsx`
- Modify: `src/components/memory/sources/__tests__/AddSourceDialog.test.tsx`
- Modify: `docs/superpowers/refactor/wenlan-app-parity-matrix.md`

- [ ] **Step 1: Inventory current app-local source writes**

Run:

```bash
rg -n "list_registered_sources|add_source|remove_source|sync_registered_source|save_current_config|load_config\\(|config::save_config|sources" app/src/search.rs src/lib/tauri.ts src/components/memory/sources
```

Expected classification:
- `connect_source`, `disconnect_source`, and `sync_source` are legacy source-name commands.
- `list_registered_sources`, `add_source`, `remove_source`, and `sync_registered_source` must move to daemon `/api/sources` for Obsidian sources.
- `config::load_config()` and `config::save_config()` must no longer appear inside registered-source add/list/remove/sync command bodies after this task, except inside an explicit `"directory"` compatibility branch that keeps local file watching.

- [ ] **Step 2: Write failing wrapper and Rust method tests**

Add to `src/lib/tauri.test.ts` or update existing source tests:

```ts
describe("registered sources", () => {
  it("addSource passes daemon source type and path", async () => {
    await tauri.addSource("obsidian", "/Users/test/vault");
    expect(mockInvoke).toHaveBeenCalledWith("add_source", {
      sourceType: "obsidian",
      path: "/Users/test/vault",
    });
  });

  it("listRegisteredSources invokes the daemon-backed command", async () => {
    mockInvoke.mockResolvedValue([]);
    await tauri.listRegisteredSources();
    expect(mockInvoke).toHaveBeenCalledWith("list_registered_sources");
  });
});
```

Add to `app/src/api.rs` tests:

```rust
#[test]
fn wenlan_client_exposes_source_registry_methods() {
    let _list_sources = WenlanClient::list_sources;
    let _add_source = WenlanClient::add_source;
    let _remove_source = WenlanClient::remove_source;
    let _sync_source = WenlanClient::sync_source;
}
```

Run:

```bash
pnpm vitest run src/lib/tauri.test.ts
cargo test -p origin-app --lib api::tests -- --nocapture
pnpm build
```

Expected: FAIL with missing `WenlanClient` source registry methods.

- [ ] **Step 3: Add typed daemon source methods**

In `app/src/api.rs`, add:

```rust
pub async fn list_sources(&self) -> Result<Vec<wenlan_types::sources::Source>, String> {
    self.get_json("/api/sources").await
}

pub async fn add_source(
    &self,
    source_type: String,
    path: String,
) -> Result<wenlan_types::sources::Source, String> {
    let req = wenlan_types::requests::AddSourceRequest { source_type, path };
    self.post_json("/api/sources", &req).await
}

pub async fn delete_empty(&self, path: &str) -> Result<(), String> {
    let resp = self
        .client
        .delete(self.url(path))
        .send()
        .await
        .map_err(|e| format!("HTTP DELETE {}: {}", path, e))?;
    if !resp.status().is_success() {
        return Err(format!("HTTP DELETE {} returned {}", path, resp.status()));
    }
    Ok(())
}

pub async fn remove_source(&self, id: &str) -> Result<(), String> {
    let path = format!("/api/sources/{}", id);
    self.delete_empty(&path).await
}

pub async fn sync_source(
    &self,
    id: &str,
) -> Result<wenlan_types::responses::SyncStatsResponse, String> {
    let path = format!("/api/sources/{}/sync", id);
    self.post_empty(&path).await
}
```

The no-body delete helper is required because daemon `DELETE /api/sources/{id}` returns `204 No Content`; do not use `delete_path::<serde_json::Value>` for this route. The sync route must use `wenlan_types::responses::SyncStatsResponse`, not `serde_json::Value`.

- [ ] **Step 4: Route registered-source Tauri commands through the daemon**

In `app/src/search.rs`, change `list_registered_sources`, `add_source`, `remove_source`, and `sync_registered_source` to take `tauri::State<'_, State>`, clone `s.client` before `.await`, and call the new `WenlanClient` methods. Keep the frontend command names and argument casing unchanged.

Directory-source compatibility gate:

- The current visible `AddSourceDialog` only calls `addSource("obsidian", path)`.
- The public wrapper type still allows `"directory"`, and the old Rust `add_source` path started the local file watcher for directory sources.
- In this task, either keep `"directory"` routed through the existing local watcher path or add a failing test that proves daemon-owned directory sources still activate `state.watch_paths` and `indexer::watch_path`.
- Do not silently route `"directory"` through `/api/sources` unless live directory watching is intentionally moved to the daemon and a test proves the app no longer owns that watcher.

Expected shape:

```rust
#[tauri::command]
pub async fn list_registered_sources(
    state: tauri::State<'_, State>,
) -> Result<Vec<wenlan_types::sources::Source>, String> {
    let client = {
        let s = state.read().await;
        s.client.clone()
    };
    client.list_sources().await
}
```

After this step, run:

```bash
rg -n "config::load_config\\(|config::save_config\\(" app/src/search.rs
```

Expected: no hits inside Obsidian source-registry add/list/remove/sync command bodies. Hits are allowed only for unrelated app-local settings or an explicit `"directory"` compatibility branch that starts the local watcher.

- [ ] **Step 5: Update source UI tests for daemon-owned source registry**

In `src/components/memory/sources/__tests__/SourcesSection.test.tsx` and `AddSourceDialog.test.tsx`, keep existing user-facing expectations but ensure mocks still use `listRegisteredSources`, `addSource`, `removeSource`, and `syncRegisteredSource`. Add one assertion that a successful add invalidates the `registeredSources` query.

Run:

```bash
pnpm vitest run src/lib/tauri.test.ts src/components/memory/sources/__tests__/SourcesSection.test.tsx src/components/memory/sources/__tests__/AddSourceDialog.test.tsx
```

Expected: PASS after implementation.

- [ ] **Step 6: Update parity matrix**

In `docs/superpowers/refactor/wenlan-app-parity-matrix.md`, update the `/api/sources` row:

```markdown
| `/api/sources` | daemon source routes | registered source list/add/remove/sync route through daemon-backed Tauri commands; no app-local config writes for source registry | keep legacy source-name commands only where still needed for compatibility | required for source management |
```

- [ ] **Step 7: Verify and commit Task 2**

Run:

```bash
pnpm vitest run src/lib/tauri.test.ts src/components/memory/sources/__tests__/SourcesSection.test.tsx src/components/memory/sources/__tests__/AddSourceDialog.test.tsx
pnpm build
cargo test -p origin-app --lib api::tests -- --nocapture
cargo test -p origin-app --lib
```

Live daemon probes:

```bash
set -euo pipefail
curl -fsS "http://127.0.0.1:7878/api/health" | jq -e '.status == "ok" and (.version | startswith("0.9."))'
curl -fsS "http://127.0.0.1:7878/api/sources" | jq -e 'type == "array"'

if [ "${WENLAN_APP_MUTATION_PROBE:-0}" = "1" ]; then
  TMP_DIR=$(mktemp -d)
  trap 'rm -rf "${TMP_DIR}"' EXIT
  printf '# Wenlan source probe\n' > "${TMP_DIR}/probe.md"
  REQ=$(jq -n --arg path "${TMP_DIR}" '{source_type:"obsidian", path:$path}')
  SOURCE_ID=$(curl -fsS -H 'content-type: application/json' -d "${REQ}" "http://127.0.0.1:7878/api/sources" | jq -er '.id')
  test -n "${SOURCE_ID}" && test "${SOURCE_ID}" != "null"
  export SOURCE_ID
  curl -fsS -X POST "http://127.0.0.1:7878/api/sources/${SOURCE_ID}/sync" | jq -e 'has("files_found") and has("ingested") and has("skipped") and has("errors")'
  DELETE_STATUS=$(curl -fsS -o /dev/null -w '%{http_code}' -X DELETE "http://127.0.0.1:7878/api/sources/${SOURCE_ID}")
  test "${DELETE_STATUS}" = "204"
else
  echo "source mutation/sync/delete live probe skipped: set WENLAN_APP_MUTATION_PROBE=1 to run against a temporary markdown vault"
fi
```

Do not mark source mutation/sync/delete as live-probe passed unless `WENLAN_APP_MUTATION_PROBE=1` was run successfully.

Commit:

```bash
git add app/src/api.rs app/src/search.rs app/src/lib.rs src/lib/tauri.ts src/lib/tauri.test.ts src/components/memory/sources/SourcesSection.tsx src/components/memory/sources/__tests__/SourcesSection.test.tsx src/components/memory/sources/__tests__/AddSourceDialog.test.tsx docs/superpowers/refactor/wenlan-app-parity-matrix.md
git commit -m "fix: route source registry through daemon"
```

## Task 3: Page Links Wrapper and PageDetail Related Links

**Files:**
- Modify: `app/src/api.rs`
- Modify: `app/src/search.rs`
- Modify: `app/src/lib.rs`
- Modify: `src/lib/tauri.ts`
- Modify: `src/lib/tauri.test.ts`
- Modify: `src/components/memory/PageDetail.tsx`
- Modify: `src/components/memory/PageDetail.test.tsx`
- Create: `src/components/memory/PageDetail.links-revisions.test.tsx`
- Modify: `docs/superpowers/refactor/wenlan-app-parity-matrix.md`

- [ ] **Step 1: Write failing TypeScript wrapper tests**

Add to `src/lib/tauri.test.ts`:

```ts
describe("page links", () => {
  it("getPageLinks passes pageId", async () => {
    mockInvoke.mockResolvedValue({ outbound: [], inbound: [] });
    await tauri.getPageLinks("page-1");
    expect(mockInvoke).toHaveBeenCalledWith("get_page_links", { pageId: "page-1" });
  });

  it("listOrphanLinks passes minCount", async () => {
    mockInvoke.mockResolvedValue({ min_count: 2, orphan_labels: [] });
    await tauri.listOrphanLinks(2);
    expect(mockInvoke).toHaveBeenCalledWith("list_orphan_links", { minCount: 2 });
  });
});
```

Run:

```bash
pnpm vitest run src/lib/tauri.test.ts
```

Expected: FAIL with missing `getPageLinks` and `listOrphanLinks`.

- [ ] **Step 2: Add Rust client methods and commands**

In `app/src/api.rs`, add:

```rust
pub async fn get_page_links(
    &self,
    page_id: &str,
) -> Result<wenlan_types::responses::PageLinksResponse, String> {
    let path = format!("/api/pages/{}/links", page_id);
    self.get_json(&path).await
}

pub async fn list_orphan_links(
    &self,
    min_count: Option<usize>,
) -> Result<wenlan_types::responses::OrphanLinksResponse, String> {
    let path = match min_count {
        Some(n) => format!("/api/pages/orphan-links?min_count={}", n),
        None => "/api/pages/orphan-links".to_string(),
    };
    self.get_json(&path).await
}
```

In `app/src/search.rs`, add:

```rust
#[tauri::command]
pub async fn get_page_links(
    state: tauri::State<'_, State>,
    page_id: String,
) -> Result<wenlan_types::responses::PageLinksResponse, String> {
    let client = {
        let s = state.read().await;
        s.client.clone()
    };
    client.get_page_links(&page_id).await
}

#[tauri::command]
pub async fn list_orphan_links(
    state: tauri::State<'_, State>,
    min_count: Option<usize>,
) -> Result<wenlan_types::responses::OrphanLinksResponse, String> {
    let client = {
        let s = state.read().await;
        s.client.clone()
    };
    client.list_orphan_links(min_count).await
}
```

Register both commands in `app/src/lib.rs`.

- [ ] **Step 3: Add TypeScript types and wrappers**

In `src/lib/tauri.ts`, add:

```ts
export interface PageLinkOutbound {
  label: string;
  target_page_id: string | null;
}

export interface PageLinkInbound {
  source_page_id: string;
  label: string;
}

export interface PageLinksResponse {
  outbound: PageLinkOutbound[];
  inbound: PageLinkInbound[];
}

export interface OrphanLink {
  label: string;
  count: number;
}

export interface OrphanLinksResponse {
  min_count: number;
  orphan_labels: OrphanLink[];
}

export async function getPageLinks(pageId: string): Promise<PageLinksResponse> {
  return invoke("get_page_links", { pageId });
}

export async function listOrphanLinks(minCount?: number): Promise<OrphanLinksResponse> {
  return invoke("list_orphan_links", { minCount: minCount ?? null });
}
```

Run:

```bash
pnpm vitest run src/lib/tauri.test.ts
cargo test -p origin-app --lib api::tests -- --nocapture
```

Expected: PASS.

- [ ] **Step 4: Write failing PageDetail link consumer tests**

In `src/components/memory/PageDetail.links-revisions.test.tsx`, start with:

```tsx
// SPDX-License-Identifier: AGPL-3.0-only
import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, within, fireEvent } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import PageDetail from "./PageDetail";
import * as tauri from "../../lib/tauri";

vi.mock("../../lib/tauri");

const page: tauri.Page = {
  id: "page-1",
  title: "Temporal channel design",
  content: "Body without daemon link labels",
  summary: "summary",
  domain: "work",
  entity_id: null,
  version: 2,
  status: "active",
  created_at: "2026-06-26T00:00:00Z",
  last_compiled: "2026-06-26T00:00:00Z",
  last_modified: "2026-06-26T00:00:00Z",
  source_memory_ids: [],
};

function wrapper({ children }: { children: React.ReactNode }) {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return <QueryClientProvider client={qc}>{children}</QueryClientProvider>;
}

describe("PageDetail links", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.mocked(tauri.getPage).mockResolvedValue(page);
    vi.mocked(tauri.listRegisteredSources).mockResolvedValue([]);
    vi.mocked(tauri.getPageSources).mockResolvedValue([]);
    vi.mocked(tauri.listPages).mockResolvedValue([]);
    vi.mocked(tauri.getPageLinks).mockResolvedValue({
      outbound: [
        { label: "Resolved Page", target_page_id: "page-2" },
        { label: "Missing Page", target_page_id: null },
      ],
      inbound: [{ source_page_id: "page-3", label: "Backlink" }],
    });
  });

  it("renders daemon page links and navigates resolved outbound links", async () => {
    const onPageClick = vi.fn();
    render(
      <PageDetail
        pageId="page-1"
        onBack={vi.fn()}
        onMemoryClick={vi.fn()}
        onPageClick={onPageClick}
      />,
      { wrapper },
    );

    expect(await screen.findByText("Temporal channel design")).toBeInTheDocument();
    expect(tauri.getPageLinks).toHaveBeenCalledWith("page-1");
    expect(tauri.listPages).not.toHaveBeenCalled();
    const linksRegion = await screen.findByLabelText("Page links");
    const resolved = within(linksRegion).getByRole("button", { name: "Resolved Page" });
    fireEvent.click(resolved);
    expect(onPageClick).toHaveBeenCalledWith("page-2");
    expect(within(linksRegion).getByText("Missing Page")).toBeInTheDocument();
    expect(within(linksRegion).getByText("Backlink")).toBeInTheDocument();
  });

  it("keeps rendering the page when page links route is unavailable", async () => {
    vi.mocked(tauri.getPageLinks).mockRejectedValue(new Error("404"));

    render(
      <PageDetail
        pageId="page-1"
        onBack={vi.fn()}
        onMemoryClick={vi.fn()}
        onPageClick={vi.fn()}
      />,
      { wrapper },
    );

    expect(await screen.findByText("Temporal channel design")).toBeInTheDocument();
    expect(screen.queryByLabelText("Page links")).toBeNull();
  });
});
```

Run:

```bash
pnpm vitest run src/components/memory/PageDetail.links-revisions.test.tsx
```

Expected: FAIL because `PageDetail` still derives links through `listPages` and local parsing.

- [ ] **Step 5: Replace PageDetail local link inference with daemon links**

In `PageDetail.tsx`:

- Import `getPageLinks`.
- Remove `parseRelatedConcepts`, `buildPageLookup`, `allPages`, and `pageLookup` if no other code uses them.
- Add `aria-label="Page links"` to the related-link section so tests and assistive tech can target daemon links without matching raw body text.
- Add:

```tsx
const { data: pageLinks } = useQuery({
  queryKey: ["page-links", pageId],
  queryFn: () => getPageLinks(pageId),
  enabled: !!pageId,
  staleTime: 30_000,
  retry: false,
});
```

Use `pageLinks?.outbound ?? []` for outbound links and `pageLinks?.inbound ?? []` for backlinks. Resolved outbound links call `onPageClick?.(target_page_id)`. Unresolved outbound links render as inert text.

Keep content click handling for `#memory:` links. For `#concept:` links, use daemon `pageLinks.outbound` to resolve a label to a target page id rather than `listPages("active")`.

- [ ] **Step 6: Update existing PageDetail tests**

In `src/components/memory/PageDetail.test.tsx`:

- Add `getPageLinks: vi.fn().mockResolvedValue({ outbound: [], inbound: [] })` to the `vi.mock("../../lib/tauri", ...)` factory.
- Add `getPageRevisions: vi.fn().mockResolvedValue({ page_id: "concept_abc", current_version: 3, user_edited: false, stale_reason: null, entries: [] })` only after Task 4 introduces page revisions.
- Replace the old `shows Related Pages only for resolved wikilinks` assertion that mocks `listPages` with a daemon-link assertion:

```tsx
it("shows Related Pages from daemon page links", async () => {
  const { getPageLinks, listPages } = await import("../../lib/tauri");
  (getPageLinks as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
    outbound: [{ label: "Entity Graph", target_page_id: "concept_eg" }],
    inbound: [],
  });
  renderWithQuery(<PageDetail {...defaultProps} />);
  expect(await screen.findByText("Related Pages")).toBeTruthy();
  expect(await screen.findByText("Entity Graph")).toBeTruthy();
  expect(listPages).not.toHaveBeenCalled();
});
```

- [ ] **Step 7: Update parity matrix**

In `docs/superpowers/refactor/wenlan-app-parity-matrix.md`, update these rows:

```markdown
| `/api/pages/{id}/links` | `PageLinksResponse` | typed Rust/Tauri/TS wrapper present; Page Detail uses daemon outbound/inbound links | keep unresolved outbound labels inert; use orphan-links for diagnostics later | hide links section if route absent |
| `/api/pages/orphan-links` | `OrphanLinksResponse` | typed Rust/Tauri/TS wrapper present; no primary UI yet | surface in diagnostics/review screen later | optional diagnostics until route exists |
```

- [ ] **Step 8: Verify and commit Task 3**

Run:

```bash
pnpm vitest run src/lib/tauri.test.ts src/components/memory/PageDetail.links-revisions.test.tsx src/components/memory/PageDetail.test.tsx src/components/memory/__tests__/PageDetail.export.test.tsx
pnpm build
cargo test -p origin-app --lib api::tests -- --nocapture
cargo test -p origin-app --lib
```

Live daemon probes:

```bash
set -euo pipefail
curl -fsS "http://127.0.0.1:7878/api/health" | jq -e '.status == "ok" and (.version | startswith("0.9."))'
PAGE_ID=$(curl -fsS "http://127.0.0.1:7878/api/pages?limit=1" | jq -er '.pages[0].id')
test -n "${PAGE_ID}" && test "${PAGE_ID}" != "null"
export PAGE_ID
curl -fsS "http://127.0.0.1:7878/api/pages/${PAGE_ID}/links" | jq -e '(.outbound | type == "array") and (.inbound | type == "array")'
curl -fsS "http://127.0.0.1:7878/api/pages/orphan-links?min_count=2" | jq -e '.min_count == 2 and (.orphan_labels | type == "array")'
```

Commit:

```bash
git add app/src/api.rs app/src/search.rs app/src/lib.rs src/lib/tauri.ts src/lib/tauri.test.ts src/components/memory/PageDetail.tsx src/components/memory/PageDetail.test.tsx src/components/memory/PageDetail.links-revisions.test.tsx docs/superpowers/refactor/wenlan-app-parity-matrix.md
git commit -m "fix: use daemon page links"
```

## Task 4: Revision History Wrappers and Detail Panels

**Files:**
- Modify: `app/src/api.rs`
- Modify: `app/src/search.rs`
- Modify: `app/src/lib.rs`
- Modify: `src/lib/tauri.ts`
- Modify: `src/lib/tauri.test.ts`
- Modify: `src/components/memory/MemoryDetail.tsx`
- Modify: `src/components/memory/PageDetail.tsx`
- Modify: `src/components/memory/MemoryDetail.enrichment-status.test.tsx`
- Modify: `src/components/memory/PageDetail.links-revisions.test.tsx`
- Modify: `docs/superpowers/refactor/wenlan-app-parity-matrix.md`

- [ ] **Step 1: Write failing TypeScript wrapper tests**

Add to `src/lib/tauri.test.ts`:

```ts
describe("revision history", () => {
  it("getMemoryRevisions passes sourceId", async () => {
    mockInvoke.mockResolvedValue({ current_source_id: "mem-1", chain_depth: 1, entries: [] });
    await tauri.getMemoryRevisions("mem-1");
    expect(mockInvoke).toHaveBeenCalledWith("get_memory_revisions", { sourceId: "mem-1" });
  });

  it("getPageRevisions passes pageId", async () => {
    mockInvoke.mockResolvedValue({
      page_id: "page-1",
      current_version: 2,
      user_edited: false,
      stale_reason: null,
      entries: [],
    });
    await tauri.getPageRevisions("page-1");
    expect(mockInvoke).toHaveBeenCalledWith("get_page_revisions", { pageId: "page-1" });
  });
});
```

Run:

```bash
pnpm vitest run src/lib/tauri.test.ts
```

Expected: FAIL with missing revision wrappers.

- [ ] **Step 2: Add Rust client methods and commands**

In `app/src/api.rs`, add:

```rust
pub async fn get_memory_revisions(
    &self,
    source_id: &str,
) -> Result<wenlan_types::responses::ListMemoryRevisionsResponse, String> {
    let path = format!("/api/memory/{}/revisions", source_id);
    self.get_json(&path).await
}

pub async fn get_page_revisions(
    &self,
    page_id: &str,
) -> Result<wenlan_types::responses::ListPageRevisionsResponse, String> {
    let path = format!("/api/pages/{}/revisions", page_id);
    self.get_json(&path).await
}
```

In `app/src/search.rs`, add:

```rust
#[tauri::command]
pub async fn get_memory_revisions(
    state: tauri::State<'_, State>,
    source_id: String,
) -> Result<wenlan_types::responses::ListMemoryRevisionsResponse, String> {
    let client = {
        let s = state.read().await;
        s.client.clone()
    };
    client.get_memory_revisions(&source_id).await
}

#[tauri::command]
pub async fn get_page_revisions(
    state: tauri::State<'_, State>,
    page_id: String,
) -> Result<wenlan_types::responses::ListPageRevisionsResponse, String> {
    let client = {
        let s = state.read().await;
        s.client.clone()
    };
    client.get_page_revisions(&page_id).await
}
```

Register both commands in `app/src/lib.rs`.

- [ ] **Step 3: Add TypeScript revision types and wrappers**

In `src/lib/tauri.ts`, add:

```ts
export interface MemoryRevisionEntry {
  source_id: string;
  depth: number;
  title: string;
  content_preview: string;
  last_modified: number;
  source_agent?: string | null;
  supersede_mode?: string | null;
  delta_summary?: string | null;
}

export interface ListMemoryRevisionsResponse {
  current_source_id: string;
  chain_depth: number;
  entries: MemoryRevisionEntry[];
}

export interface PageChangelogEntry {
  version: number;
  at: number;
  edited_by: string;
  delta_summary?: string | null;
  incoming_source_ids?: string[] | null;
}

export interface ListPageRevisionsResponse {
  page_id: string;
  current_version: number;
  user_edited: boolean;
  stale_reason?: string | null;
  entries: PageChangelogEntry[];
}

export async function getMemoryRevisions(
  sourceId: string
): Promise<ListMemoryRevisionsResponse> {
  return invoke("get_memory_revisions", { sourceId });
}

export async function getPageRevisions(
  pageId: string
): Promise<ListPageRevisionsResponse> {
  return invoke("get_page_revisions", { pageId });
}
```

Run:

```bash
pnpm vitest run src/lib/tauri.test.ts
cargo test -p origin-app --lib api::tests -- --nocapture
```

Expected: PASS.

- [ ] **Step 4: Add revision panel tests**

Extend `MemoryDetail.enrichment-status.test.tsx`:

```tsx
it("renders daemon memory revision history", async () => {
  vi.mocked(tauri.getEnrichmentStatus).mockRejectedValue(new Error("old daemon"));
  vi.mocked(tauri.getMemoryRevisions).mockResolvedValue({
    current_source_id: "mem-1",
    chain_depth: 2,
    entries: [
      {
        source_id: "mem-1",
        depth: 0,
        title: "Current",
        content_preview: "Current version",
        last_modified: 10,
        source_agent: "claude-code",
        supersede_mode: "protected_revision",
        delta_summary: "Clarified wording",
      },
    ],
  });

  render(
    <MemoryDetail
      sourceId="mem-1"
      onBack={vi.fn()}
      onNavigateEntity={vi.fn()}
      onNavigateMemory={vi.fn()}
    />,
    { wrapper },
  );

  expect(await screen.findByText(/revision history/i)).toBeInTheDocument();
  expect(screen.getByText(/clarified wording/i)).toBeInTheDocument();
});

it("keeps rendering the memory when memory revisions route is unavailable", async () => {
  vi.mocked(tauri.getEnrichmentStatus).mockRejectedValue(new Error("old daemon"));
  vi.mocked(tauri.getMemoryRevisions).mockRejectedValue(new Error("404"));

  render(
    <MemoryDetail
      sourceId="mem-1"
      onBack={vi.fn()}
      onNavigateEntity={vi.fn()}
      onNavigateMemory={vi.fn()}
    />,
    { wrapper },
  );

  expect(await screen.findByText("A memory")).toBeInTheDocument();
  expect(screen.queryByText(/revision history/i)).toBeNull();
});
```

Extend `PageDetail.links-revisions.test.tsx`:

```tsx
it("renders daemon page revision history", async () => {
  vi.mocked(tauri.getPageRevisions).mockResolvedValue({
    page_id: "page-1",
    current_version: 2,
    user_edited: false,
    stale_reason: null,
    entries: [
      {
        version: 2,
        at: 1782490000000,
        edited_by: "distill",
        delta_summary: "Added backlinks",
        incoming_source_ids: ["mem-1"],
      },
    ],
  });

  render(
    <PageDetail
      pageId="page-1"
      onBack={vi.fn()}
      onMemoryClick={vi.fn()}
      onPageClick={vi.fn()}
    />,
    { wrapper },
  );

  expect(await screen.findByText(/revision history/i)).toBeInTheDocument();
  expect(screen.getByText(/added backlinks/i)).toBeInTheDocument();
});

it("keeps rendering the page when page revisions route is unavailable", async () => {
  vi.mocked(tauri.getPageRevisions).mockRejectedValue(new Error("404"));

  render(
    <PageDetail
      pageId="page-1"
      onBack={vi.fn()}
      onMemoryClick={vi.fn()}
      onPageClick={vi.fn()}
    />,
    { wrapper },
  );

  expect(await screen.findByText("Temporal channel design")).toBeInTheDocument();
  expect(screen.queryByText(/revision history/i)).toBeNull();
});
```

Run:

```bash
pnpm vitest run src/components/memory/MemoryDetail.enrichment-status.test.tsx src/components/memory/PageDetail.links-revisions.test.tsx
```

Expected: FAIL until UI panels are implemented.

- [ ] **Step 5: Implement MemoryDetail and PageDetail revision panels**

In `MemoryDetail.tsx`, import `getMemoryRevisions` and query:

```tsx
const { data: memoryRevisions } = useQuery({
  queryKey: ["memory-revisions", sourceId],
  queryFn: () => getMemoryRevisions(sourceId),
  enabled: !!sourceId,
  staleTime: 30_000,
  retry: false,
});
```

Render the panel only when `memoryRevisions.entries.length > 0`. Each row should show `title`, `content_preview`, `delta_summary` when present, and call `onNavigateMemory(entry.source_id)` for non-current entries.

In `PageDetail.tsx`, import `getPageRevisions` and query:

```tsx
const { data: pageRevisions } = useQuery({
  queryKey: ["page-revisions", pageId],
  queryFn: () => getPageRevisions(pageId),
  enabled: !!pageId,
  staleTime: 30_000,
  retry: false,
});
```

Render only when `pageRevisions.entries.length > 0`. Show `version`, `edited_by`, `delta_summary`, and `incoming_source_ids.length` when present. Do not block page rendering on this optional route.

- [ ] **Step 6: Update parity matrix**

In `docs/superpowers/refactor/wenlan-app-parity-matrix.md`, update these rows:

```markdown
| `/api/memory/{id}/revisions` | `ListMemoryRevisionsResponse` | typed Rust/Tauri/TS wrapper present; Memory Detail shows non-blocking revision history | keep old version-chain wrapper only as legacy fallback until removed in a later cleanup | hide revisions panel if route absent |
| `/api/pages/{id}/revisions` | `ListPageRevisionsResponse` | typed Rust/Tauri/TS wrapper present; Page Detail shows non-blocking revision history | reuse for future page diff UI | hide revisions panel if route absent |
```

- [ ] **Step 7: Verify and commit Task 4**

Run:

```bash
pnpm vitest run src/lib/tauri.test.ts src/components/memory/MemoryDetail.enrichment-status.test.tsx src/components/memory/PageDetail.links-revisions.test.tsx src/components/memory/PageDetail.test.tsx src/components/memory/__tests__/PageDetail.export.test.tsx
pnpm build
cargo test -p origin-app --lib
```

Live daemon probes:

```bash
set -euo pipefail
curl -fsS "http://127.0.0.1:7878/api/health" | jq -e '.status == "ok" and (.version | startswith("0.9."))'
MEMORY_ID=$(curl -fsS "http://127.0.0.1:7878/api/memory/recent?limit=1" | jq -er '.[0].id')
PAGE_ID=$(curl -fsS "http://127.0.0.1:7878/api/pages?limit=1" | jq -er '.pages[0].id')
test -n "${MEMORY_ID}" && test "${MEMORY_ID}" != "null"
test -n "${PAGE_ID}" && test "${PAGE_ID}" != "null"
export MEMORY_ID PAGE_ID
curl -fsS "http://127.0.0.1:7878/api/memory/${MEMORY_ID}/revisions" | jq -e '.current_source_id == env.MEMORY_ID and (.entries | type == "array")'
curl -fsS "http://127.0.0.1:7878/api/pages/${PAGE_ID}/revisions" | jq -e '.page_id == env.PAGE_ID and (.entries | type == "array")'
```

Commit:

```bash
git add app/src/api.rs app/src/search.rs app/src/lib.rs src/lib/tauri.ts src/lib/tauri.test.ts src/components/memory/MemoryDetail.tsx src/components/memory/MemoryDetail.enrichment-status.test.tsx src/components/memory/PageDetail.tsx src/components/memory/PageDetail.links-revisions.test.tsx docs/superpowers/refactor/wenlan-app-parity-matrix.md
git commit -m "fix: surface daemon revision history"
```

## Deferred Boundaries

- Do not rename `Origin.app`, `com.origin.desktop`, `com.origin.server`, updater endpoints, relay URLs, remote token paths, app data roots, app config paths, or LaunchAgent cleanup labels in this plan.
- Do not change default data/config path behavior from legacy Origin paths to fresh Wenlan paths until a dedicated bridge plan proves old state is detected, imported, and preserved.
- Do not globally replace `concept`, `domain`, or `goal`; taxonomy cleanup is a separate plan after API parity.
- Do not remove `getVersionChain` until `getMemoryRevisions` has shipped and is validated against old and current daemons.
- Do not treat CodeGraph as proof. It is only a discovery and blast-radius tool.

## Adversarial Review Notes

| Attack | Defense in this plan | Residual risk |
|---|---|---|
| Optional daemon routes fail and blank the detail screen | All new detail queries use `retry: false` and render only when data exists | A missing route can still hide useful diagnostics on old daemons |
| PageDetail silently regresses resolved wiki-link navigation | The PageDetail tests use an `aria-label="Page links"` region, assert `getPageLinks("page-1")`, assert `listPages` is not used, and click a resolved outbound button | ContentRenderer link interception still needs careful regression testing |
| Revision wrappers compile but use wrong id names | TS tests assert `sourceId` and `pageId`; Rust methods use typed response envelopes | Path ids are not percent-encoded; acceptable only while daemon ids remain path-safe |
| App invents local page-link semantics again | Task 3 removes `listPages("active")` link lookup and uses `/api/pages/{id}/links` | Unresolved link presentation can still become too noisy |
| The plan drifts into rename work | Deferred boundaries explicitly exclude bundle/product/runtime identity | Separate identity plan still required before public release |
| Tooling reduces tokens but misses behavior | CodeGraph/ast-grep are pre-edit discovery only; tests, builds, and live daemon probes remain gates | Live probes need a daemon with representative page/memory data |

## Boule Handoff

Use this exact prompt for adversarial design review before implementation:

```text
/boule:debate Review docs/superpowers/plans/2026-06-26-wenlan-app-post-merge-api-parity-plan.md for the next wenlan-app migration slice after PR #2. The target is API parity with Wenlan v0.9.x, not a shallow Origin rename. Attack the plan on typed-client correctness, Tauri command argument drift, optional-route failure behavior, page-link/revision UI regressions, CodeGraph/ast-grep/LSP boundaries, and whether the plan should include or defer runtime identity rename. Identify missing tests, unsafe ordering, false dependencies, and any path that could silently strand user data, config, or review proposals.
```
