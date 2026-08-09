# Wenlan App Distill Review Panel Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a non-destructive Distill Review panel that closes the safe `/api/distill` parity gap without exposing force rebuild, targeted distill, or page-specific redistill.

**Architecture:** Keep the existing thin-client seam: Rust daemon client -> Tauri command -> TypeScript wrapper -> routed React view. The route is global and user-triggered only, sending `{}` to `POST /api/distill` and rendering the review payload as navigation/information, not synthesis controls.

**Tech Stack:** Tauri 2, Rust, `reqwest`, `serde`, React 19, React Query, Vitest, Testing Library, CodeGraph, ast-grep, rust-analyzer, Cargo, pnpm.

---

## Current Baseline

- Worktree: `/Users/lucian/Repos/wenlan-app/.worktrees/wenlan-app-api-parity-audit`
- Branch: `codex/wenlan-app-distill-design`
- Design spec: `docs/superpowers/refactor/2026-06-29-wenlan-app-distill-review-design.md`
- Current route diff:

```bash
pnpm refactor:api-routes --json
```

Expected current output before this plan is implemented:

```text
{"backendRoutes":123,"appSourceRoutes":112,"missingInApp":11,"appOnly":0}
```

Expected output after implementation:

```text
{"backendRoutes":123,"appSourceRoutes":113,"missingInApp":10,"appOnly":0}
```

## Tool Boundary For This Slice

Use these in order. CodeGraph is for orientation, not proof.

```bash
codegraph status .
codegraph query WenlanClient --json
codegraph query HomePage --json
codegraph affected app/src/api.rs --json
```

Expected: `codegraph status .` reports the index path as this worktree, not `/Users/lucian/Repos/wenlan-app`.

Use ast-grep for structural checks:

```bash
sg run -p 'invoke($CMD, $$$ARGS)' -l ts src/lib/tauri.ts
sg run -p 'pub async fn $NAME($$$ARGS) -> Result<$RET, String> { $$$BODY }' -l rs app/src/search.rs
```

Use `rust-analyzer`, Cargo, TypeScript, and tests as semantic authorities:

```bash
rust-analyzer --version
cargo test --manifest-path app/Cargo.toml
pnpm test
pnpm build
```

Use `rg` as the residual check lane:

```bash
rg -n '/api/distill/' app/src src
rg -n 'force\s*:' app/src/api.rs src/lib/tauri.ts src/components/memory
rg -n 'target\s*:' app/src/api.rs src/lib/tauri.ts src/components/memory
```

Expected after implementation: no production call path for `/api/distill/`, `force:`, or `target:` in the new app distill surface.

## File Map

- Modify `app/src/api.rs`: app-local strict DTOs and `WenlanClient::distill_review()`.
- Modify `app/src/search.rs`: Tauri command `distill_review`.
- Modify `app/src/lib.rs`: register `search::distill_review`.
- Modify `src/lib/tauri.ts`: TypeScript interfaces and `distillReview()`.
- Modify `src/lib/tauri.test.ts`: wrapper test that the command has no args.
- Create `src/components/memory/DistillReviewPanel.tsx`: routed review view with user-triggered refresh.
- Create `src/components/memory/DistillReviewPanel.test.tsx`: panel behavior tests.
- Modify `src/components/memory/Main.tsx`: route `{ kind: "distill-review" }`.
- Modify `src/components/memory/HomePage.tsx`: entry point near `RefiningList`.
- Modify `src/components/memory/HomePage.redesign.test.tsx`: entry-point test.

## Task 0: Confirm Local Refactor Tools

**Files:**
- Read: `docs/superpowers/refactor/2026-06-25-codegraph-evaluation.md`
- Read: `docs/superpowers/refactor/2026-06-25-wenlan-app-tooling.md`
- No source edits.

- [ ] **Step 1: Confirm CodeGraph is worktree-local**

Run:

```bash
codegraph status .
```

Expected: the status output names `/Users/lucian/Repos/wenlan-app/.worktrees/wenlan-app-api-parity-audit` as the active index tree. If it names `/Users/lucian/Repos/wenlan-app`, run:

```bash
codegraph init -i .
```

Then re-run:

```bash
codegraph status .
```

- [ ] **Step 2: Confirm graph queries resolve current symbols**

Run:

```bash
codegraph query WenlanClient --json
codegraph query HomePage --json
```

Expected:

- `WenlanClient` resolves to `app/src/api.rs`.
- `HomePage` resolves to `src/components/memory/HomePage.tsx`.

- [ ] **Step 3: Confirm structural tools**

Run:

```bash
sg --version
rust-analyzer --version
pnpm refactor:api-routes --json
```

Expected:

- `sg --version` prints `ast-grep 0.44.0` or newer.
- `rust-analyzer --version` prints a version string.
- route diff prints `{"backendRoutes":123,"appSourceRoutes":112,"missingInApp":11,"appOnly":0}` before implementation.

- [ ] **Step 4: Commit only if generated docs changed**

This task normally creates no tracked changes. If route-diff or inventory output is intentionally regenerated, commit it separately:

```bash
git add docs/superpowers/refactor/wenlan-app-inventory
git commit -m "docs: refresh wenlan app refactor inventory"
```

## Task 1: Rust Client Distill Review Contract

**Files:**
- Modify: `app/src/api.rs`

- [ ] **Step 1: Write failing Rust API tests**

Add these tests inside `#[cfg(test)] mod tests` in `app/src/api.rs`, after `move_space_percent_encodes_space_names_as_path_segments`:

```rust
    #[tokio::test]
    async fn distill_review_posts_empty_global_request_to_daemon() {
        let body = r#"{"pages_created":0,"scoped":false,"created_ids":[],"pending":[],"stale_pages":[],"stale_truncated":false,"orphan_topics":[]}"#;
        let (base_url, request) = serve_json_once(body).await;
        let client = WenlanClient {
            client: reqwest::Client::new(),
            base_url,
        };

        let resp = client.distill_review().await.unwrap();

        assert_eq!(resp.pages_created, 0);
        assert!(!resp.scoped);
        let request = request.await.unwrap();
        assert_eq!(
            request.lines().next().unwrap_or_default(),
            "POST /api/distill HTTP/1.1"
        );
        assert_eq!(request_body(&request), serde_json::json!({}));
    }

    #[test]
    fn distill_review_deserializes_daemon_review_payload_with_centroid_embedding() {
        let payload = serde_json::json!({
            "pages_created": 1,
            "scoped": false,
            "created_ids": ["page_new"],
            "pending": [{
                "source_ids": ["mem_1", "mem_2"],
                "contents": ["First source", "Second source"],
                "entity_id": "entity_rust",
                "entity_name": "Rust",
                "space": "Engineering",
                "estimated_tokens": 220,
                "centroid_embedding": [0.1, 0.2],
                "existing_page_id": "page_rust",
                "existing_page_title": "Rust notes",
                "new_memory_count": 1
            }],
            "stale_pages": [{
                "page_id": "page_old",
                "title": "Old page",
                "summary": "Needs source review",
                "source_memory_ids": ["mem_old"],
                "sources_updated_count": 3,
                "stale_reason": "source_updated",
                "user_edited": false
            }],
            "stale_truncated": true,
            "orphan_topics": [{"label": "Vector clocks", "count": 4}]
        });

        let resp: DistillReviewResponse = serde_json::from_value(payload).unwrap();

        assert_eq!(resp.created_ids, vec!["page_new"]);
        assert_eq!(resp.pending[0].centroid_embedding, Some(vec![0.1, 0.2]));
        assert_eq!(resp.pending[0].existing_page_title.as_deref(), Some("Rust notes"));
        assert_eq!(resp.pending[0].new_memory_count, Some(1));
        assert_eq!(resp.stale_pages[0].page_id, "page_old");
        assert_eq!(resp.orphan_topics[0].label, "Vector clocks");
        assert!(resp.stale_truncated);
    }

    #[test]
    fn distill_review_rejects_unresolved_target_hint_shape() {
        let payload = serde_json::json!({
            "pages_created": 0,
            "pages_updated": 0,
            "unresolved": "unknown target",
            "hint": "target must be a page id"
        });

        let err = serde_json::from_value::<DistillReviewResponse>(payload).unwrap_err();

        assert!(err.to_string().contains("missing field") || err.to_string().contains("unknown field"));
    }
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
cargo test --manifest-path app/Cargo.toml distill_review -- --nocapture
```

Expected: fails because `DistillReviewResponse` and `WenlanClient::distill_review` do not exist.

- [ ] **Step 3: Add strict DTOs and client method**

Change the import near the top of `app/src/api.rs`:

```rust
use serde::{Deserialize, Serialize};
```

Replace the existing `use serde::Serialize;`.

Add these DTOs after `MoveSpaceResponse`:

```rust
#[derive(Debug, Clone, Serialize)]
struct DistillReviewRequest {}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct DistillReviewResponse {
    pub pages_created: usize,
    pub scoped: bool,
    pub created_ids: Vec<String>,
    pub pending: Vec<DistillPendingCluster>,
    pub stale_pages: Vec<DistillStalePage>,
    pub stale_truncated: bool,
    pub orphan_topics: Vec<DistillOrphanTopic>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct DistillPendingCluster {
    pub source_ids: Vec<String>,
    pub contents: Vec<String>,
    pub entity_id: Option<String>,
    pub entity_name: Option<String>,
    pub space: Option<String>,
    pub estimated_tokens: usize,
    pub centroid_embedding: Option<Vec<f32>>,
    pub existing_page_id: Option<String>,
    pub existing_page_title: Option<String>,
    pub new_memory_count: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct DistillStalePage {
    pub page_id: String,
    pub title: String,
    pub summary: Option<String>,
    pub source_memory_ids: Vec<String>,
    pub sources_updated_count: Option<usize>,
    pub stale_reason: Option<String>,
    pub user_edited: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct DistillOrphanTopic {
    pub label: String,
    pub count: usize,
}
```

Add this method near the other endpoint methods in `impl WenlanClient`:

```rust
    pub async fn distill_review(&self) -> Result<DistillReviewResponse, String> {
        self.post_json("/api/distill", &DistillReviewRequest {})
            .await
    }
```

- [ ] **Step 4: Run focused Rust API tests**

Run:

```bash
cargo test --manifest-path app/Cargo.toml distill_review -- --nocapture
```

Expected: all `distill_review` tests pass.

- [ ] **Step 5: Run route residual check**

Run:

```bash
rg -n '/api/distill/' app/src src
```

Expected: no output. The only app production path should be the global literal `/api/distill`.

- [ ] **Step 6: Commit**

```bash
git add app/src/api.rs
git commit -m "fix: add distill review daemon client"
```

## Task 2: Tauri Command Registration

**Files:**
- Modify: `app/src/search.rs`
- Modify: `app/src/lib.rs`

- [ ] **Step 1: Write failing compile-time command type test**

Add this helper inside `#[cfg(test)] mod space_command_type_tests` in `app/src/search.rs`, after `move_space_command_uses_typed_affected_envelope`:

```rust
    #[allow(dead_code)]
    async fn distill_review_command_uses_typed_review_envelope(state: tauri::State<'_, State>) {
        let _: Result<crate::api::DistillReviewResponse, String> = distill_review(state).await;
    }
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
cargo test --manifest-path app/Cargo.toml space_command_type_tests -- --nocapture
```

Expected: fails because `distill_review` is not defined.

- [ ] **Step 3: Add the Tauri command**

Add this command in `app/src/search.rs` near page/refinement commands or next to `ingest_webpage` if keeping all thin daemon wrappers together:

```rust
#[tauri::command]
pub async fn distill_review(
    state: tauri::State<'_, State>,
) -> Result<crate::api::DistillReviewResponse, String> {
    let client = {
        let s = state.read().await;
        s.client.clone()
    };
    client.distill_review().await
}
```

This command clones `WenlanClient` before awaiting the daemon call.

- [ ] **Step 4: Register the command**

Add `search::distill_review,` to the `tauri::generate_handler!` list in `app/src/lib.rs` near page/refinement commands:

```rust
            search::ingest_webpage,
            search::distill_review,
            search::get_clipboard_enabled,
```

The exact nearby line can differ; keep the command grouped with daemon wrapper commands.

- [ ] **Step 5: Run focused Rust tests**

Run:

```bash
cargo test --manifest-path app/Cargo.toml distill_review -- --nocapture
cargo test --manifest-path app/Cargo.toml space_command_type_tests -- --nocapture
```

Expected: tests pass.

- [ ] **Step 6: Run registration residual checks**

Run:

```bash
rg -n 'distill_review' app/src/search.rs app/src/lib.rs
rg -n '/api/distill/' app/src src
```

Expected:

- `distill_review` appears in `app/src/search.rs` and `app/src/lib.rs`.
- `/api/distill/` produces no output.

- [ ] **Step 7: Commit**

```bash
git add app/src/search.rs app/src/lib.rs
git commit -m "fix: expose distill review tauri command"
```

## Task 3: TypeScript Wrapper

**Files:**
- Modify: `src/lib/tauri.ts`
- Modify: `src/lib/tauri.test.ts`

- [ ] **Step 1: Write failing wrapper test**

Add this test in `src/lib/tauri.test.ts` near the other wrapper tests:

```ts
  it('distillReview invokes the review command without target or force args', async () => {
    const payload: tauri.DistillReviewResponse = {
      pages_created: 0,
      scoped: false,
      created_ids: [],
      pending: [],
      stale_pages: [],
      stale_truncated: false,
      orphan_topics: [],
    };
    mockInvoke.mockResolvedValue(payload);

    const result = await tauri.distillReview();

    expect(result).toEqual(payload);
    expect(mockInvoke).toHaveBeenCalledWith('distill_review');
  });
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
pnpm test src/lib/tauri.test.ts
```

Expected: fails because `DistillReviewResponse` and `distillReview` do not exist.

- [ ] **Step 3: Add TypeScript interfaces and wrapper**

Add these interfaces in `src/lib/tauri.ts` near page-related types or after the space wrappers:

```ts
export interface DistillReviewResponse {
  pages_created: number;
  scoped: boolean;
  created_ids: string[];
  pending: DistillPendingCluster[];
  stale_pages: DistillStalePage[];
  stale_truncated: boolean;
  orphan_topics: DistillOrphanTopic[];
}

export interface DistillPendingCluster {
  source_ids: string[];
  contents: string[];
  entity_id?: string | null;
  entity_name?: string | null;
  space?: string | null;
  estimated_tokens: number;
  centroid_embedding?: number[] | null;
  existing_page_id?: string | null;
  existing_page_title?: string | null;
  new_memory_count?: number | null;
}

export interface DistillStalePage {
  page_id: string;
  title: string;
  summary?: string | null;
  source_memory_ids: string[];
  sources_updated_count?: number | null;
  stale_reason?: string | null;
  user_edited?: boolean | null;
}

export interface DistillOrphanTopic {
  label: string;
  count: number;
}

export async function distillReview(): Promise<DistillReviewResponse> {
  return invoke("distill_review");
}
```

- [ ] **Step 4: Run focused wrapper test**

Run:

```bash
pnpm test src/lib/tauri.test.ts
```

Expected: wrapper tests pass.

- [ ] **Step 5: Run structural no-arg check**

Run:

```bash
sg run -p 'invoke("distill_review", { $$$ARGS })' -l ts src/lib/tauri.ts
```

Expected: no output.

- [ ] **Step 6: Commit**

```bash
git add src/lib/tauri.ts src/lib/tauri.test.ts
git commit -m "fix: add distill review frontend wrapper"
```

## Task 4: Distill Review Panel Component

**Files:**
- Create: `src/components/memory/DistillReviewPanel.tsx`
- Create: `src/components/memory/DistillReviewPanel.test.tsx`

- [ ] **Step 1: Write failing panel tests**

Create `src/components/memory/DistillReviewPanel.test.tsx`:

```tsx
// SPDX-License-Identifier: AGPL-3.0-only
import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import DistillReviewPanel from "./DistillReviewPanel";
import type { DistillReviewResponse } from "../../lib/tauri";

vi.mock("../../lib/tauri", () => ({
  distillReview: vi.fn(),
}));

import { distillReview } from "../../lib/tauri";

function renderPanel(props: Partial<React.ComponentProps<typeof DistillReviewPanel>> = {}) {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
  });
  const onBack = props.onBack ?? vi.fn();
  const onPageClick = props.onPageClick ?? vi.fn();
  const user = userEvent.setup();
  render(
    <QueryClientProvider client={client}>
      <DistillReviewPanel onBack={onBack} onPageClick={onPageClick} />
    </QueryClientProvider>,
  );
  return { user, onBack, onPageClick };
}

const reviewPayload: DistillReviewResponse = {
  pages_created: 0,
  scoped: false,
  created_ids: [],
  pending: [
    {
      source_ids: ["mem_1", "mem_2"],
      contents: [
        "This is a detailed source memory about temporal page refresh behavior.",
        "A second source memory adds routing context for the distill review panel.",
      ],
      entity_id: "entity_temporal",
      entity_name: "Temporal refresh",
      space: "Engineering",
      estimated_tokens: 180,
      centroid_embedding: [0.1, 0.2],
      existing_page_id: "page_temporal",
      existing_page_title: "Temporal page refresh",
      new_memory_count: 1,
    },
    {
      source_ids: ["mem_3"],
      contents: ["Fallback content label should appear when no title or entity exists."],
      estimated_tokens: 80,
    },
  ],
  stale_pages: [
    {
      page_id: "page_stale",
      title: "Retrieval Pipeline",
      summary: "Source memories changed after the page compiled.",
      source_memory_ids: ["mem_old"],
      sources_updated_count: 3,
      stale_reason: "source_updated",
      user_edited: false,
    },
  ],
  stale_truncated: true,
  orphan_topics: [{ label: "Vector clocks", count: 4 }],
};

beforeEach(() => {
  vi.clearAllMocks();
});

describe("DistillReviewPanel", () => {
  it("does not run the distill POST on mount", () => {
    vi.mocked(distillReview).mockResolvedValue(reviewPayload);

    renderPanel();

    expect(distillReview).not.toHaveBeenCalled();
    expect(screen.getByRole("button", { name: /refresh review/i })).toBeInTheDocument();
  });

  it("renders review sections after a user-triggered refresh", async () => {
    vi.mocked(distillReview).mockResolvedValue(reviewPayload);
    const { user } = renderPanel();

    await user.click(screen.getByRole("button", { name: /refresh review/i }));

    expect(await screen.findByText("Temporal page refresh")).toBeInTheDocument();
    expect(screen.getByText(/1 new source/)).toBeInTheDocument();
    expect(screen.getByText(/Fallback content label should appear/)).toBeInTheDocument();
    expect(screen.getByText("Retrieval Pipeline")).toBeInTheDocument();
    expect(screen.getByText(/first 10 stale pages/i)).toBeInTheDocument();
    expect(screen.getByText("Vector clocks")).toBeInTheDocument();
    expect(screen.getByText(/4 mentions/)).toBeInTheDocument();
  });

  it("navigates stale pages without exposing rebuild controls", async () => {
    vi.mocked(distillReview).mockResolvedValue(reviewPayload);
    const { user, onPageClick } = renderPanel();

    await user.click(screen.getByRole("button", { name: /refresh review/i }));
    await user.click(await screen.findByRole("button", { name: /open Retrieval Pipeline/i }));

    expect(onPageClick).toHaveBeenCalledWith("page_stale");
    expect(screen.queryByText(/force rebuild/i)).toBeNull();
    expect(screen.queryByText(/synthesize page/i)).toBeNull();
    expect(screen.queryByText(/^rebuild$/i)).toBeNull();
  });

  it("keeps the last successful result visible after refresh failure", async () => {
    vi.mocked(distillReview)
      .mockResolvedValueOnce(reviewPayload)
      .mockRejectedValueOnce(new Error("HTTP POST /api/distill returned 500"));
    const { user } = renderPanel();

    await user.click(screen.getByRole("button", { name: /refresh review/i }));
    expect(await screen.findByText("Temporal page refresh")).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: /refresh review/i }));

    expect(await screen.findByText(/HTTP POST \/api\/distill returned 500/)).toBeInTheDocument();
    expect(screen.getByText("Temporal page refresh")).toBeInTheDocument();
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
pnpm test src/components/memory/DistillReviewPanel.test.tsx
```

Expected: fails because `DistillReviewPanel.tsx` does not exist.

- [ ] **Step 3: Create the component**

Create `src/components/memory/DistillReviewPanel.tsx`:

```tsx
// SPDX-License-Identifier: AGPL-3.0-only
import { useState } from "react";
import { useMutation } from "@tanstack/react-query";
import {
  distillReview,
  type DistillPendingCluster,
  type DistillReviewResponse,
} from "../../lib/tauri";

interface DistillReviewPanelProps {
  onBack: () => void;
  onPageClick: (pageId: string) => void;
}

function truncateText(value: string, max: number): string {
  if (value.length <= max) return value;
  return `${value.slice(0, max - 1).trimEnd()}...`;
}

function firstNonEmpty(values: Array<string | null | undefined>): string | null {
  for (const value of values) {
    const trimmed = value?.trim();
    if (trimmed) return trimmed;
  }
  return null;
}

function pendingLabel(cluster: DistillPendingCluster): string {
  const fromContent = firstNonEmpty(cluster.contents);
  return (
    firstNonEmpty([
      cluster.existing_page_title,
      cluster.entity_name,
      cluster.space,
      fromContent ? truncateText(fromContent, 72) : null,
    ]) ?? "Untitled cluster"
  );
}

function sourceCopy(count: number): string {
  return count === 1 ? "1 source" : `${count} sources`;
}

function newSourceCopy(count: number): string {
  return count === 1 ? "1 new source" : `${count} new sources`;
}

export default function DistillReviewPanel({ onBack, onPageClick }: DistillReviewPanelProps) {
  const [lastResult, setLastResult] = useState<DistillReviewResponse | null>(null);
  const review = useMutation({
    mutationFn: distillReview,
    retry: false,
    onSuccess: (result) => setLastResult(result),
  });
  const result = lastResult;
  const error = review.error instanceof Error ? review.error.message : review.error ? String(review.error) : null;

  return (
    <div style={{ padding: "36px 56px", color: "var(--mem-text-primary)" }}>
      <button
        type="button"
        onClick={onBack}
        aria-label="Back"
        style={{
          color: "var(--mem-text-secondary)",
          background: "transparent",
          border: 0,
          fontSize: 24,
          cursor: "pointer",
          marginBottom: 24,
        }}
      >
        <-
      </button>

      <div style={{ display: "flex", justifyContent: "space-between", gap: 16, alignItems: "center" }}>
        <div>
          <h1 style={{ fontFamily: "var(--mem-font-heading)", fontSize: 28, margin: 0 }}>
            Distill Review
          </h1>
          <p style={{ margin: "8px 0 0", color: "var(--mem-text-secondary)" }}>
            Review pending page work from the daemon.
          </p>
        </div>
        <button
          type="button"
          onClick={() => review.mutate()}
          disabled={review.isPending}
          style={{
            border: "1px solid var(--mem-border)",
            background: "var(--mem-surface)",
            color: "var(--mem-text-primary)",
            borderRadius: 8,
            padding: "9px 14px",
            cursor: review.isPending ? "default" : "pointer",
            opacity: review.isPending ? 0.7 : 1,
          }}
        >
          {review.isPending ? "Refreshing..." : "Refresh review"}
        </button>
      </div>

      {error && (
        <div
          role="alert"
          style={{
            marginTop: 20,
            border: "1px solid var(--mem-border)",
            background: "var(--mem-surface)",
            color: "var(--mem-text-primary)",
            borderRadius: 8,
            padding: 12,
          }}
        >
          {error}
        </div>
      )}

      {!result && !review.isPending && (
        <p style={{ color: "var(--mem-text-secondary)", marginTop: 28 }}>
          Run a refresh to load the current review queue.
        </p>
      )}

      {result && (
        <div style={{ display: "grid", gap: 24, marginTop: 28 }}>
          <section>
            <h2 style={{ fontSize: 17, margin: "0 0 10px" }}>Pending pages</h2>
            {result.pending.length === 0 ? (
              <p style={{ color: "var(--mem-text-secondary)", margin: 0 }}>No pending page clusters.</p>
            ) : (
              <div style={{ display: "grid", gap: 10 }}>
                {result.pending.map((cluster) => {
                  const label = pendingLabel(cluster);
                  const preview = cluster.contents.filter((item) => item.trim().length > 0).slice(0, 2);
                  return (
                    <article
                      key={`${label}-${cluster.source_ids.join("-")}`}
                      style={{
                        border: "1px solid var(--mem-border)",
                        borderRadius: 8,
                        padding: 14,
                        background: "var(--mem-surface)",
                      }}
                    >
                      <h3 style={{ margin: 0, fontSize: 16 }}>{label}</h3>
                      <p style={{ margin: "6px 0", color: "var(--mem-text-secondary)" }}>
                        {cluster.new_memory_count != null
                          ? newSourceCopy(cluster.new_memory_count)
                          : sourceCopy(cluster.source_ids.length)}
                        {cluster.existing_page_id ? " linked to an existing page" : ""}
                      </p>
                      {preview.map((item, index) => (
                        <p key={index} style={{ margin: "6px 0 0", color: "var(--mem-text-secondary)" }}>
                          {truncateText(item, 140)}
                        </p>
                      ))}
                    </article>
                  );
                })}
              </div>
            )}
          </section>

          <section>
            <h2 style={{ fontSize: 17, margin: "0 0 10px" }}>Stale pages</h2>
            {result.stale_truncated && (
              <p style={{ color: "var(--mem-text-secondary)", margin: "0 0 10px" }}>
                The daemon returned the first 10 stale pages; more may exist.
              </p>
            )}
            {result.stale_pages.length === 0 ? (
              <p style={{ color: "var(--mem-text-secondary)", margin: 0 }}>No stale pages.</p>
            ) : (
              <div style={{ display: "grid", gap: 10 }}>
                {result.stale_pages.map((page) => (
                  <button
                    key={page.page_id}
                    type="button"
                    aria-label={`Open ${page.title}`}
                    onClick={() => onPageClick(page.page_id)}
                    style={{
                      textAlign: "left",
                      border: "1px solid var(--mem-border)",
                      borderRadius: 8,
                      padding: 14,
                      background: "var(--mem-surface)",
                      color: "var(--mem-text-primary)",
                      cursor: "pointer",
                    }}
                  >
                    <strong>{page.title}</strong>
                    {page.summary && (
                      <span style={{ display: "block", marginTop: 6, color: "var(--mem-text-secondary)" }}>
                        {page.summary}
                      </span>
                    )}
                  </button>
                ))}
              </div>
            )}
          </section>

          <section>
            <h2 style={{ fontSize: 17, margin: "0 0 10px" }}>Unlinked topics</h2>
            {result.orphan_topics.length === 0 ? (
              <p style={{ color: "var(--mem-text-secondary)", margin: 0 }}>No repeated unlinked topics.</p>
            ) : (
              <div style={{ display: "grid", gap: 10 }}>
                {result.orphan_topics.map((topic) => (
                  <article
                    key={topic.label}
                    style={{
                      border: "1px solid var(--mem-border)",
                      borderRadius: 8,
                      padding: 14,
                      background: "var(--mem-surface)",
                    }}
                  >
                    <strong>{topic.label}</strong>
                    <span style={{ display: "block", marginTop: 6, color: "var(--mem-text-secondary)" }}>
                      {topic.count === 1 ? "1 mention" : `${topic.count} mentions`}
                    </span>
                  </article>
                ))}
              </div>
            )}
          </section>
        </div>
      )}
    </div>
  );
}
```

If the ASCII back button visually needs cleanup during implementation, replace `"<- "` with the app's existing back glyph pattern from `Main.tsx` or `PageDetail.tsx` while keeping the accessible label `Back`.

- [ ] **Step 4: Run focused panel tests**

Run:

```bash
pnpm test src/components/memory/DistillReviewPanel.test.tsx
```

Expected: tests pass.

- [ ] **Step 5: Run no-control residual check**

Run:

```bash
rg -n 'force rebuild|synthesize page|rebuild page|force\s*:|target\s*:' src/components/memory/DistillReviewPanel.tsx
```

Expected: no output.

- [ ] **Step 6: Commit**

```bash
git add src/components/memory/DistillReviewPanel.tsx src/components/memory/DistillReviewPanel.test.tsx
git commit -m "fix: add distill review panel"
```

## Task 5: Home Entry And Main Routing

**Files:**
- Modify: `src/components/memory/Main.tsx`
- Modify: `src/components/memory/HomePage.tsx`
- Modify: `src/components/memory/HomePage.redesign.test.tsx`

- [ ] **Step 1: Write failing Home entry test**

In `src/components/memory/HomePage.redesign.test.tsx`, update the `renderHome()` helper so it can accept a distill callback:

```tsx
function renderHome(props: { onOpenDistillReview?: () => void } = {}) {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={qc}>
      <HomePage
        onNavigateMemory={() => {}}
        onNavigateStream={() => {}}
        onNavigateLog={() => {}}
        onNavigateGraph={() => {}}
        onOpenDistillReview={props.onOpenDistillReview}
      />
    </QueryClientProvider>,
  );
}
```

Add this test under `describe("HomePage redesign", () => {`:

```tsx
  it("opens the distill review route from the home refinement area", async () => {
    const onOpenDistillReview = vi.fn();
    const user = userEvent.setup();

    renderHome({ onOpenDistillReview });

    await user.click(await screen.findByRole("button", { name: /review distillation/i }));

    expect(onOpenDistillReview).toHaveBeenCalledTimes(1);
  });
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
pnpm test src/components/memory/HomePage.redesign.test.tsx
```

Expected: fails because `HomePageProps` has no `onOpenDistillReview` and no button renders.

- [ ] **Step 3: Add HomePage prop and button**

Update `HomePageProps` in `src/components/memory/HomePage.tsx`:

```tsx
interface HomePageProps {
  onNavigateMemory: (sourceId: string) => void;
  onNavigateStream: () => void;
  onNavigateLog: () => void;
  onNavigateGraph: () => void;
  onSelectPage?: (pageId: string) => void;
  onOpenDistillReview?: () => void;
}
```

Update the function parameters:

```tsx
export default function HomePage({
  onNavigateMemory,
  onNavigateStream,
  onNavigateLog: _onNavigateLog,
  onNavigateGraph: _onNavigateGraph,
  onSelectPage,
  onOpenDistillReview,
}: HomePageProps) {
```

Add this button immediately before the existing `<RefiningList ... />`:

```tsx
          {onOpenDistillReview && (
            <div style={{ display: "flex", justifyContent: "flex-end", margin: "-4px 0 10px" }}>
              <button
                type="button"
                onClick={onOpenDistillReview}
                style={{
                  border: "1px solid var(--mem-border)",
                  background: "var(--mem-surface)",
                  color: "var(--mem-text-primary)",
                  borderRadius: 8,
                  padding: "7px 11px",
                  fontSize: 13,
                  cursor: "pointer",
                }}
              >
                Review distillation
              </button>
            </div>
          )}
          <RefiningList changes={changes} pages={recentConcepts} onSelectPage={onSelectPage} />
```

- [ ] **Step 4: Add Main routed view**

Add import in `src/components/memory/Main.tsx`:

```tsx
import DistillReviewPanel from "./DistillReviewPanel";
```

Extend the `View` union:

```tsx
type View = { kind: "home" } | { kind: "stream" } | { kind: "activity" } | { kind: "recaps" } | { kind: "entity"; entityId: string } | { kind: "profile" } | { kind: "memory"; sourceId: string } | { kind: "settings"; section?: SettingsSection } | { kind: "import" } | { kind: "connect-agent" } | { kind: "space"; spaceName: string } | { kind: "graph" } | { kind: "page"; pageId: string } | { kind: "distill-review" } | { kind: "decisions" };
```

Add a render branch before the `home` branch:

```tsx
          ) : view.kind === "distill-review" ? (
            <DistillReviewPanel
              onBack={navigateBack}
              onPageClick={(id) => navigateTo({ kind: "page", pageId: id })}
            />
          ) : view.kind === "home" ? (
```

Pass the Home callback:

```tsx
              onOpenDistillReview={() => navigateTo({ kind: "distill-review" })}
```

The `HomePage` usage should look like this:

```tsx
            <HomePage
              onNavigateMemory={(sid) => navigateTo({ kind: "memory", sourceId: sid })}
              onNavigateStream={() => navigateTo({ kind: "recaps" })}
              onNavigateLog={() => navigateTo({ kind: "stream" })}
              onNavigateGraph={() => navigateTo({ kind: "graph" })}
              onSelectPage={(id) => navigateTo({ kind: "page", pageId: id })}
              onOpenDistillReview={() => navigateTo({ kind: "distill-review" })}
            />
```

- [ ] **Step 5: Run focused Home and panel tests**

Run:

```bash
pnpm test src/components/memory/HomePage.redesign.test.tsx src/components/memory/DistillReviewPanel.test.tsx
```

Expected: tests pass.

- [ ] **Step 6: Run TypeScript build**

Run:

```bash
pnpm build
```

Expected: TypeScript and Vite build pass.

- [ ] **Step 7: Commit**

```bash
git add src/components/memory/Main.tsx src/components/memory/HomePage.tsx src/components/memory/HomePage.redesign.test.tsx
git commit -m "fix: route distill review from home"
```

## Task 6: Full Verification And Route Parity Checkpoint

**Files:**
- No planned source edits unless verification exposes a defect.

- [ ] **Step 1: Sync graph and query new symbols**

Run:

```bash
codegraph sync .
codegraph query DistillReviewPanel --json
codegraph query distillReview --json
codegraph query distill_review --json
```

Expected:

- `DistillReviewPanel` resolves to `src/components/memory/DistillReviewPanel.tsx`.
- `distillReview` resolves to `src/lib/tauri.ts`.
- `distill_review` resolves to `app/src/search.rs`.

- [ ] **Step 2: Run structural residual checks**

Run:

```bash
sg run -p 'invoke("distill_review", { $$$ARGS })' -l ts src/lib/tauri.ts
rg -n '/api/distill/' app/src src
rg -n 'force\s*:|target\s*:' app/src/api.rs src/lib/tauri.ts src/components/memory/DistillReviewPanel.tsx
```

Expected: no output from all three commands.

- [ ] **Step 3: Run Rust checks**

Run:

```bash
cargo test --manifest-path app/Cargo.toml
cargo clippy --manifest-path app/Cargo.toml --all-targets -- -D warnings
```

Expected: both pass.

- [ ] **Step 4: Run frontend checks**

Run:

```bash
pnpm test
pnpm build
```

Expected: both pass.

- [ ] **Step 5: Run route diff**

Run:

```bash
pnpm refactor:api-routes --json
```

Expected:

```text
{"backendRoutes":123,"appSourceRoutes":113,"missingInApp":10,"appOnly":0}
```

- [ ] **Step 6: Inspect diff**

Run:

```bash
git diff --check
git status --short
git log --oneline --decorate -n 8
```

Expected:

- `git diff --check` passes.
- Worktree has no unstaged source changes after commits.
- Log shows the task commits on `codex/wenlan-app-distill-design`.

## Task 7: Final Review And PR Checkpoint

**Files:**
- No planned source edits unless review exposes a defect.

- [ ] **Step 1: Run fresh-eye integrated code review**

Dispatch one fresh reviewer with this brief:

```text
Review the integrated distill review panel implementation in /Users/lucian/Repos/wenlan-app/.worktrees/wenlan-app-api-parity-audit against docs/superpowers/refactor/2026-06-29-wenlan-app-distill-review-design.md and docs/superpowers/plans/2026-06-29-wenlan-app-distill-review-panel.md.

Evaluate on the merits. Do not force a pessimistic stance. Findings must cite concrete file:line evidence and explain the defect.

Checklist:
- No UI or wrapper can send target, force, or /api/distill/{page_id}.
- POST /api/distill is user-triggered only, never on mount/focus/polling.
- DTOs match the daemon global review payload including centroid_embedding.
- Stale page rows only navigate to PageDetail.
- Error state preserves the last successful review result.
- Tests and route diff prove the checkpoint.
```

Expected: reviewer either approves or reports concrete issues. Fix critical and important issues before PR.

- [ ] **Step 2: Create PR checkpoint**

Run after review issues are fixed and verification is green:

```bash
git status --short
git push -u origin codex/wenlan-app-distill-design
gh pr create --draft --base main --head codex/wenlan-app-distill-design --title "fix: add distill review panel" --body "Adds the non-destructive /api/distill review checkpoint for wenlan-app. No force rebuild, no targeted distill, and no /api/distill/{page_id} app wrapper."
```

Expected: draft PR opens against `7xuanlu/wenlan-app`.

## Self-Review Notes

- Spec coverage:
  - Global-only `/api/distill`: Tasks 1, 2, 3, and residual checks in Task 6.
  - No target, no force, no `/api/distill/{page_id}`: Tests in Tasks 1 and 3, residual checks in Tasks 1, 2, 4, and 6, reviewer checklist in Task 7.
  - Strict daemon payload including `centroid_embedding`: Task 1 DTOs and tests, Task 4 UI ignores that field.
  - User-triggered UI only: Task 4 mutation test proves no POST on mount.
  - Home entry and routed view: Task 5.
  - Stale-page navigation only: Task 4 panel test and Task 7 review checklist.
  - Route diff target: Task 6.
- Type consistency:
  - Rust command: `distill_review`.
  - TypeScript wrapper: `distillReview`.
  - Component: `DistillReviewPanel`.
  - Response fields use daemon snake_case unchanged.
- Planned commit cadence:
  - Client contract.
  - Tauri command.
  - TypeScript wrapper.
  - Panel.
  - Route integration.
