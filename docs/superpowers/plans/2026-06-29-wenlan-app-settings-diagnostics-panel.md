# Wenlan App Settings Diagnostics Panel Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Surface the daemon's read-only pipeline status in Wenlan App Settings without exposing the mutating `/api/steep` maintenance route.

**Architecture:** Add a typed app-local DTO and Tauri command for `GET /api/debug/pipeline`, then render it through a focused Settings diagnostics section. Keep diagnostics read-only, scoped to Settings, and covered by Rust wrapper tests plus React rendering tests.

**Tech Stack:** Tauri 2, Rust, reqwest, serde, React 19, TanStack Query, Vitest, Testing Library, CodeGraph, ast-grep.

---

## Tool Boundary

- CodeGraph: use for blast-radius and dependency lookup. It requires escalated local index access in this sandbox because `.codegraph/codegraph.db` uses SQLite WAL.
- ast-grep: use `sg outline` to place symbols and keep file edits focused.
- LSP/compiler/tests: source of truth for type/import correctness.
- `rg`/`sed`: fallback for exact source reads when CodeGraph or ast-grep is too broad.

## File Structure

- Modify `app/src/api.rs`: add `PipelineStatusResponse`, `PipelineEntityLinkingStatus`, `PipelineQueueEntry`, and `WenlanClient::pipeline_status()`.
- Modify `app/src/search.rs`: add `get_pipeline_status` Tauri command and a compile-time type test.
- Modify `app/src/lib.rs`: register `search::get_pipeline_status` in `tauri::generate_handler!`.
- Modify `src/lib/tauri.ts`: add TypeScript pipeline status types and `getPipelineStatus()`.
- Modify `src/lib/tauri.test.ts`: add the IPC wrapper test.
- Modify `src/components/memory/settings/SettingsSidebar.tsx`: add the `diagnostics` Settings group.
- Create `src/components/memory/settings/DiagnosticsSection.tsx`: focused read-only diagnostics UI.
- Create `src/components/memory/settings/DiagnosticsSection.test.tsx`: React coverage for rendered fields, old-daemon error, and no maintenance button.
- Modify `src/components/memory/SettingsPage.tsx`: render `DiagnosticsSection` when `section === "diagnostics"`.
- Update `docs/superpowers/refactor/wenlan-app-inventory/api-route-classifications.json`: remove `/api/debug/pipeline` after it is directly surfaced.
- Regenerate `docs/superpowers/refactor/wenlan-app-inventory/api-route-diff.{json,md}` with `pnpm refactor:api-routes --json`.

## Task 1: Typed Pipeline Status Boundary

**Files:**
- Modify: `app/src/api.rs`
- Modify: `app/src/search.rs`
- Modify: `app/src/lib.rs`
- Modify: `src/lib/tauri.ts`
- Modify: `src/lib/tauri.test.ts`

- [ ] **Step 1: Write failing Rust client tests**

In `app/src/api.rs`, change the import:

```rust
use std::collections::{BTreeMap, HashMap};
```

Then add these tests near `capture_stats_uses_daemon_capture_stats_endpoint`:

```rust
#[tokio::test]
async fn pipeline_status_uses_daemon_debug_pipeline_endpoint() {
    let body = r#"{"enrichment":{"done":3,"pending":1},"entity_linking":{"linked":7,"unlinked":2},"refinement_queue":[{"action":"merge","status":"pending","count":4}],"recaps":5,"types":{"fact":6},"quality":{"high":2}}"#;
    let (base_url, request) = serve_json_once(body).await;
    let client = WenlanClient {
        client: reqwest::Client::new(),
        base_url,
    };

    let status = client.pipeline_status().await.unwrap();

    assert_eq!(status.enrichment.get("done"), Some(&3));
    assert_eq!(status.entity_linking.linked, 7);
    assert_eq!(status.entity_linking.unlinked, 2);
    assert_eq!(status.refinement_queue[0].action, "merge");
    assert_eq!(status.refinement_queue[0].status, "pending");
    assert_eq!(status.refinement_queue[0].count, 4);
    assert_eq!(status.recaps, 5);
    assert_eq!(status.types.get("fact"), Some(&6));
    assert_eq!(status.quality.get("high"), Some(&2));
    let request = request.await.unwrap();
    assert_eq!(
        request.lines().next().unwrap_or_default(),
        "GET /api/debug/pipeline HTTP/1.1"
    );
}

#[test]
fn pipeline_status_response_deserializes_daemon_payload() {
    let status: PipelineStatusResponse = serde_json::from_value(serde_json::json!({
        "enrichment": {"raw": 1, "classified": 2},
        "entity_linking": {"linked": 3, "unlinked": 4},
        "refinement_queue": [
            {"action": "merge", "status": "pending", "count": 5}
        ],
        "recaps": 6,
        "types": {"fact": 7},
        "quality": {"trusted": 8},
        "future_additive_key": {"ignored": true}
    }))
    .expect("daemon pipeline status payload should deserialize");

    assert_eq!(status.enrichment.get("classified"), Some(&2));
    assert_eq!(status.entity_linking.linked, 3);
    assert_eq!(status.entity_linking.unlinked, 4);
    assert_eq!(status.refinement_queue.len(), 1);
    assert_eq!(status.recaps, 6);
    assert_eq!(status.types.get("fact"), Some(&7));
    assert_eq!(status.quality.get("trusted"), Some(&8));
}
```

- [ ] **Step 2: Run Rust tests and confirm they fail for missing symbols**

Run:

```bash
cargo test --manifest-path app/Cargo.toml pipeline_status -- --nocapture
```

Expected: FAIL because `PipelineStatusResponse` and `WenlanClient::pipeline_status` do not exist yet.

- [ ] **Step 3: Implement the Rust DTO and client method**

In `app/src/api.rs`, add these structs after `CaptureStatsResponse`:

```rust
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PipelineStatusResponse {
    pub enrichment: BTreeMap<String, u64>,
    pub entity_linking: PipelineEntityLinkingStatus,
    pub refinement_queue: Vec<PipelineQueueEntry>,
    pub recaps: u64,
    pub types: BTreeMap<String, u64>,
    pub quality: BTreeMap<String, u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PipelineEntityLinkingStatus {
    pub linked: u64,
    pub unlinked: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PipelineQueueEntry {
    pub action: String,
    pub status: String,
    pub count: u64,
}
```

Add this method near `get_capture_stats`:

```rust
pub async fn pipeline_status(&self) -> Result<PipelineStatusResponse, String> {
    self.get_json("/api/debug/pipeline").await
}
```

- [ ] **Step 4: Add the Tauri command and command type test**

In `app/src/search.rs`, add this command after `get_capture_stats`:

```rust
#[tauri::command]
pub async fn get_pipeline_status(
    state: tauri::State<'_, State>,
) -> Result<crate::api::PipelineStatusResponse, String> {
    let client = {
        let s = state.read().await;
        s.client.clone()
    };
    client.pipeline_status().await
}
```

Add this test module near the command:

```rust
#[cfg(test)]
mod pipeline_status_command_type_tests {
    use super::*;

    #[allow(dead_code)]
    async fn get_pipeline_status_uses_typed_response(state: tauri::State<'_, State>) {
        let _: Result<crate::api::PipelineStatusResponse, String> =
            get_pipeline_status(state).await;
    }

    #[test]
    fn pipeline_status_command_response_type_is_checked() {}
}
```

In `app/src/lib.rs`, add this handler next to `search::get_index_status`:

```rust
search::get_pipeline_status,
```

- [ ] **Step 5: Add the TypeScript wrapper test first**

In `src/lib/tauri.test.ts`, add this block after `getIndexStatus`:

```ts
describe("getPipelineStatus", () => {
  it("calls invoke with no args", async () => {
    mockInvoke.mockResolvedValue({
      enrichment: {},
      entity_linking: { linked: 0, unlinked: 0 },
      refinement_queue: [],
      recaps: 0,
      types: {},
      quality: {},
    });

    await tauri.getPipelineStatus();

    expect(mockInvoke).toHaveBeenCalledWith("get_pipeline_status");
  });
});
```

Run:

```bash
pnpm vitest run src/lib/tauri.test.ts -t getPipelineStatus
```

Expected: FAIL because `getPipelineStatus` is not exported yet.

- [ ] **Step 6: Implement the TypeScript types and wrapper**

In `src/lib/tauri.ts`, add these types near `CaptureStats`:

```ts
export interface PipelineEntityLinkingStatus {
  linked: number;
  unlinked: number;
}

export interface PipelineQueueEntry {
  action: string;
  status: string;
  count: number;
}

export interface PipelineStatusResponse {
  enrichment: Record<string, number>;
  entity_linking: PipelineEntityLinkingStatus;
  refinement_queue: PipelineQueueEntry[];
  recaps: number;
  types: Record<string, number>;
  quality: Record<string, number>;
}
```

Add this wrapper:

```ts
export async function getPipelineStatus(): Promise<PipelineStatusResponse> {
  return invoke("get_pipeline_status");
}
```

- [ ] **Step 7: Run focused wrapper verification**

Run:

```bash
cargo test --manifest-path app/Cargo.toml pipeline_status -- --nocapture
pnpm vitest run src/lib/tauri.test.ts -t getPipelineStatus
```

Expected: both pass.

- [ ] **Step 8: Commit Task 1**

Run:

```bash
git add app/src/api.rs app/src/search.rs app/src/lib.rs src/lib/tauri.ts src/lib/tauri.test.ts
git commit -m "fix: add pipeline diagnostics command"
```

## Task 2: Read-Only Settings Diagnostics UI

**Files:**
- Modify: `src/components/memory/settings/SettingsSidebar.tsx`
- Create: `src/components/memory/settings/DiagnosticsSection.tsx`
- Create: `src/components/memory/settings/DiagnosticsSection.test.tsx`
- Modify: `src/components/memory/SettingsPage.tsx`

- [ ] **Step 1: Write failing UI tests**

Create `src/components/memory/settings/DiagnosticsSection.test.tsx`:

```tsx
// SPDX-License-Identifier: AGPL-3.0-only
import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import type { ReactNode } from "react";
import DiagnosticsSection from "./DiagnosticsSection";
import { getPipelineStatus } from "../../../lib/tauri";

vi.mock("../../../lib/tauri", () => ({
  getPipelineStatus: vi.fn(),
}));

function renderDiagnostics() {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  function Wrapper({ children }: { children: ReactNode }) {
    return <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>;
  }
  return render(<DiagnosticsSection />, { wrapper: Wrapper });
}

describe("DiagnosticsSection", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.mocked(getPipelineStatus).mockResolvedValue({
      enrichment: { classified: 9, raw: 2 },
      entity_linking: { linked: 7, unlinked: 3 },
      refinement_queue: [{ action: "merge", status: "pending", count: 4 }],
      recaps: 5,
      types: { fact: 6, preference: 1 },
      quality: { trusted: 8, low: 1 },
    });
  });

  it("renders the pipeline snapshot fields", async () => {
    renderDiagnostics();

    expect(await screen.findByText("Pipeline Snapshot")).toBeInTheDocument();
    expect(screen.getByText("classified")).toBeInTheDocument();
    expect(screen.getByText("9")).toBeInTheDocument();
    expect(screen.getByText("Entity linking")).toBeInTheDocument();
    expect(screen.getByText("70% linked")).toBeInTheDocument();
    expect(screen.getByText("merge")).toBeInTheDocument();
    expect(screen.getByText("pending")).toBeInTheDocument();
    expect(screen.getByText("Recaps")).toBeInTheDocument();
    expect(screen.getByText("5")).toBeInTheDocument();
    expect(screen.getByText("fact")).toBeInTheDocument();
    expect(screen.getByText("trusted")).toBeInTheDocument();
  });

  it("shows a scoped old-daemon message when the route is missing", async () => {
    vi.mocked(getPipelineStatus).mockRejectedValue(new Error("HTTP GET /api/debug/pipeline returned 404: not found"));

    renderDiagnostics();

    expect(await screen.findByText("Diagnostics require a newer daemon")).toBeInTheDocument();
    expect(screen.queryByText("Run maintenance")).not.toBeInTheDocument();
  });

  it("does not expose the manual steep maintenance action", async () => {
    renderDiagnostics();

    await waitFor(() => expect(getPipelineStatus).toHaveBeenCalled());
    expect(screen.queryByText("Run maintenance")).not.toBeInTheDocument();
    expect(screen.queryByText("Steep")).not.toBeInTheDocument();
  });
});
```

Run:

```bash
pnpm vitest run src/components/memory/settings/DiagnosticsSection.test.tsx
```

Expected: FAIL because `DiagnosticsSection.tsx` does not exist yet.

- [ ] **Step 2: Add the Diagnostics Settings group**

In `src/components/memory/settings/SettingsSidebar.tsx`, extend `SettingsSection`:

```ts
export type SettingsSection =
  | "capture"
  | "sources"
  | "agents"
  | "intelligence"
  | "diagnostics"
  | "general";
```

Add this group after `intelligence`:

```tsx
{
  id: "diagnostics",
  label: "Diagnostics",
  hint: "Daemon pipeline health",
  icon: (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round">
      <path d="M4 19V5" />
      <path d="M4 19h16" />
      <path d="M8 15l3-3 3 2 4-6" />
    </svg>
  ),
},
```

- [ ] **Step 3: Implement `DiagnosticsSection`**

Create `src/components/memory/settings/DiagnosticsSection.tsx`:

```tsx
// SPDX-License-Identifier: AGPL-3.0-only
import { useQuery } from "@tanstack/react-query";
import { getPipelineStatus, type PipelineStatusResponse } from "../../../lib/tauri";

function sortedEntries(values: Record<string, number>): [string, number][] {
  return Object.entries(values).sort(([leftKey, leftValue], [rightKey, rightValue]) => {
    if (rightValue !== leftValue) return rightValue - leftValue;
    return leftKey.localeCompare(rightKey);
  });
}

function isOldDaemonError(error: unknown): boolean {
  const message = error instanceof Error ? error.message : String(error);
  return message.includes("/api/debug/pipeline") && (message.includes("404") || message.includes("not found"));
}

function StatList({ title, values, empty }: { title: string; values: Record<string, number>; empty: string }) {
  const entries = sortedEntries(values);
  return (
    <div className="px-5 py-4">
      <div className="mb-2" style={{ fontFamily: "var(--mem-font-body)", fontSize: "13px", fontWeight: 600, color: "var(--mem-text)" }}>
        {title}
      </div>
      {entries.length === 0 ? (
        <p style={{ fontFamily: "var(--mem-font-body)", fontSize: "12px", color: "var(--mem-text-tertiary)" }}>{empty}</p>
      ) : (
        <div className="flex flex-col gap-1.5">
          {entries.map(([key, count]) => (
            <div key={key} className="flex items-center justify-between gap-3">
              <span style={{ fontFamily: "var(--mem-font-mono)", fontSize: "12px", color: "var(--mem-text-secondary)" }}>{key}</span>
              <span style={{ fontFamily: "var(--mem-font-mono)", fontSize: "12px", color: "var(--mem-text)" }}>{count}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function EntityLinking({ data }: { data: PipelineStatusResponse }) {
  const total = data.entity_linking.linked + data.entity_linking.unlinked;
  const percent = total === 0 ? null : Math.round((data.entity_linking.linked / total) * 100);
  return (
    <div className="px-5 py-4">
      <div className="mb-2" style={{ fontFamily: "var(--mem-font-body)", fontSize: "13px", fontWeight: 600, color: "var(--mem-text)" }}>
        Entity linking
      </div>
      <div className="flex items-baseline gap-3">
        <span style={{ fontFamily: "var(--mem-font-heading)", fontSize: "22px", color: "var(--mem-text)" }}>{data.entity_linking.linked}</span>
        <span style={{ fontFamily: "var(--mem-font-body)", fontSize: "12px", color: "var(--mem-text-secondary)" }}>
          linked / {data.entity_linking.unlinked} unlinked
        </span>
      </div>
      {percent !== null && (
        <p style={{ fontFamily: "var(--mem-font-body)", fontSize: "12px", color: "var(--mem-text-tertiary)", marginTop: 4 }}>
          {percent}% linked
        </p>
      )}
    </div>
  );
}

function RefineryQueue({ data }: { data: PipelineStatusResponse }) {
  return (
    <div className="px-5 py-4">
      <div className="mb-2" style={{ fontFamily: "var(--mem-font-body)", fontSize: "13px", fontWeight: 600, color: "var(--mem-text)" }}>
        Refinery queue
      </div>
      {data.refinement_queue.length === 0 ? (
        <p style={{ fontFamily: "var(--mem-font-body)", fontSize: "12px", color: "var(--mem-text-tertiary)" }}>No pending refinery work.</p>
      ) : (
        <div className="flex flex-col gap-1.5">
          {data.refinement_queue.map((entry) => (
            <div key={`${entry.action}:${entry.status}`} className="grid grid-cols-[1fr_auto_auto] items-center gap-3">
              <span style={{ fontFamily: "var(--mem-font-mono)", fontSize: "12px", color: "var(--mem-text-secondary)" }}>{entry.action}</span>
              <span style={{ fontFamily: "var(--mem-font-mono)", fontSize: "12px", color: "var(--mem-text-tertiary)" }}>{entry.status}</span>
              <span style={{ fontFamily: "var(--mem-font-mono)", fontSize: "12px", color: "var(--mem-text)" }}>{entry.count}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

export default function DiagnosticsSection() {
  const pipelineQuery = useQuery({
    queryKey: ["pipelineStatus"],
    queryFn: getPipelineStatus,
    retry: false,
  });

  return (
    <section className="mem-fade-up" style={{ animationDelay: "0ms" }}>
      <div className="flex items-center justify-between gap-3 mb-3 px-1">
        <h3 style={{ fontFamily: "var(--mem-font-heading)", fontSize: "12px", fontWeight: 600, letterSpacing: "0.05em", color: "var(--mem-text-tertiary)", textTransform: "uppercase" as const }}>
          Pipeline Snapshot
        </h3>
        <button
          onClick={() => pipelineQuery.refetch()}
          className="px-2.5 py-1 rounded-md transition-colors hover:bg-[var(--mem-hover)]"
          style={{ fontFamily: "var(--mem-font-body)", fontSize: "12px", color: "var(--mem-text-secondary)", border: "1px solid var(--mem-border)" }}
        >
          Refresh
        </button>
      </div>
      <div className="bg-[var(--mem-surface)] rounded-xl overflow-hidden border border-[var(--mem-border)]">
        {pipelineQuery.isLoading && (
          <p className="px-5 py-4" style={{ fontFamily: "var(--mem-font-body)", fontSize: "12px", color: "var(--mem-text-secondary)" }}>
            Loading diagnostics...
          </p>
        )}
        {pipelineQuery.isError && (
          <p className="px-5 py-4" style={{ fontFamily: "var(--mem-font-body)", fontSize: "12px", color: "#ef4444", lineHeight: "1.5" }}>
            {isOldDaemonError(pipelineQuery.error) ? "Diagnostics require a newer daemon" : pipelineQuery.error instanceof Error ? pipelineQuery.error.message : "Diagnostics unavailable"}
          </p>
        )}
        {pipelineQuery.data && (
          <>
            <StatList title="Enrichment" values={pipelineQuery.data.enrichment} empty="No enrichment rows." />
            <div className="mx-5 border-t border-[var(--mem-border)]" style={{ opacity: 0.4 }} />
            <EntityLinking data={pipelineQuery.data} />
            <div className="mx-5 border-t border-[var(--mem-border)]" style={{ opacity: 0.4 }} />
            <RefineryQueue data={pipelineQuery.data} />
            <div className="mx-5 border-t border-[var(--mem-border)]" style={{ opacity: 0.4 }} />
            <div className="px-5 py-4">
              <div className="mb-1" style={{ fontFamily: "var(--mem-font-body)", fontSize: "13px", fontWeight: 600, color: "var(--mem-text)" }}>Recaps</div>
              <span style={{ fontFamily: "var(--mem-font-heading)", fontSize: "22px", color: "var(--mem-text)" }}>{pipelineQuery.data.recaps}</span>
            </div>
            <div className="mx-5 border-t border-[var(--mem-border)]" style={{ opacity: 0.4 }} />
            <StatList title="Memory types" values={pipelineQuery.data.types} empty="No memory type rows." />
            <div className="mx-5 border-t border-[var(--mem-border)]" style={{ opacity: 0.4 }} />
            <StatList title="Quality" values={pipelineQuery.data.quality} empty="No quality rows." />
          </>
        )}
      </div>
    </section>
  );
}
```

- [ ] **Step 4: Wire `DiagnosticsSection` into SettingsPage**

In `src/components/memory/SettingsPage.tsx`, add this import:

```ts
import DiagnosticsSection from "./settings/DiagnosticsSection";
```

Add this section before the persistent privacy footer:

```tsx
{/* Diagnostics (group: diagnostics) */}
{section === "diagnostics" && <DiagnosticsSection />}
```

- [ ] **Step 5: Run focused UI verification**

Run:

```bash
pnpm vitest run src/components/memory/settings/DiagnosticsSection.test.tsx
pnpm build
```

Expected: both pass.

- [ ] **Step 6: Commit Task 2**

Run:

```bash
git add src/components/memory/settings/SettingsSidebar.tsx src/components/memory/settings/DiagnosticsSection.tsx src/components/memory/settings/DiagnosticsSection.test.tsx src/components/memory/SettingsPage.tsx
git commit -m "fix: surface pipeline diagnostics in settings"
```

## Task 3: Route Inventory and Full Verification

**Files:**
- Modify: `docs/superpowers/refactor/wenlan-app-inventory/api-route-classifications.json`
- Regenerate: `docs/superpowers/refactor/wenlan-app-inventory/api-route-diff.json`
- Regenerate: `docs/superpowers/refactor/wenlan-app-inventory/api-route-diff.md`

- [ ] **Step 1: Remove the implemented route classification**

In `docs/superpowers/refactor/wenlan-app-inventory/api-route-classifications.json`, delete only this entry:

```json
"/api/debug/pipeline": {
  "category": "operator_diagnostics",
  "status": "design_ready",
  "rationale": "This returns raw pipeline state; the app already has user-facing status and reranker diagnostics.",
  "next_action": "Implement the read-only Settings diagnostics panel described in docs/superpowers/refactor/2026-06-29-wenlan-app-settings-diagnostics-design.md."
},
```

Keep `/api/steep` deferred.

- [ ] **Step 2: Regenerate the API route diff**

Run:

```bash
pnpm refactor:api-routes --json
```

Expected JSON:

```json
{"backendRoutes":123,"appSourceRoutes":115,"missingInApp":8,"classifiedMissingInApp":8,"unclassifiedMissingInApp":0,"appOnly":0}
```

If `backendRoutes` changes because the daemon moved, verify the changed backend routes directly before accepting the new count.

- [ ] **Step 3: Run complete verification**

Run:

```bash
cargo test --manifest-path app/Cargo.toml
pnpm test
pnpm build
pnpm refactor:api-routes --json
```

Expected:

- Rust tests pass.
- Vitest passes.
- TypeScript/Vite build passes.
- Route diff has zero unclassified gaps and zero app-only routes.

- [ ] **Step 4: Commit Task 3**

Run:

```bash
git add docs/superpowers/refactor/wenlan-app-inventory/api-route-classifications.json docs/superpowers/refactor/wenlan-app-inventory/api-route-diff.json docs/superpowers/refactor/wenlan-app-inventory/api-route-diff.md
git commit -m "docs: update diagnostics route inventory"
```

## Final Integrated Review

- [ ] Run CodeGraph affected-test lookup after implementation:

```bash
codegraph sync /Users/lucian/Repos/wenlan-app/.worktrees/wenlan-app-diagnostics-panel
codegraph affected app/src/api.rs app/src/search.rs app/src/lib.rs src/lib/tauri.ts src/components/memory/settings/DiagnosticsSection.tsx src/components/memory/SettingsPage.tsx
```

- [ ] Run final verification:

```bash
cargo test --manifest-path app/Cargo.toml
pnpm test
pnpm build
pnpm refactor:api-routes --json
```

- [ ] Launch the Tauri app against the local v0.9 daemon and visually confirm:
  - app opens,
  - daemon connects,
  - Settings sidebar includes Diagnostics,
  - Diagnostics shows the pipeline snapshot or scoped old-daemon message,
  - no manual maintenance or Steep button appears.

- [ ] Open a checkpoint PR against `7xuanlu/wenlan-app:main` after final review passes.

## Self-Review

- Spec coverage: implements read-only `/api/debug/pipeline`; does not expose `/api/steep`; adds typed Rust and TS wrappers; adds focused UI and tests; updates route inventory.
- Placeholder scan: no deferred implementation blanks are present in the tasks.
- Type consistency: Rust, TS, and UI all use `enrichment`, `entity_linking`, `refinement_queue`, `recaps`, `types`, and `quality`.
- Tool boundary: CodeGraph, ast-grep, compiler/tests, and grep fallback are assigned to distinct roles.
