# Page Detail Citations Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the wiki page-detail view into a citation-first reading page: inline per-claim citation chips with hover popovers, a Related Pages strip, and one collapsed Page info disclosure replacing the four always-open bottom sections.

**Architecture:** The Rust Tauri commands `get_page` / `get_page_revisions` switch to raw `serde_json::Value` passthrough (the pinned `wenlan-types 0.9.2` structs silently drop the daemon's new `citations` field). A pure TS util (`src/lib/pageCitations.ts`) mirrors the backend's marker-counting contract and rewrites `[N]` markers into `[k](#citation:k)` links; `ContentRenderer` maps those links to `CitationChip` components. New subcomponents under `src/components/memory/page/` absorb the old bottom sections; `PageDetail.tsx` stays the orchestrator.

**Tech Stack:** React 19, TypeScript, Tailwind v4 (inline `--mem-*` CSS vars idiom), react-markdown + remark-gfm, @tanstack/react-query, Vitest 4 + Testing Library (jsdom), Rust/Tauri 2, serde_json.

**Spec:** `docs/superpowers/specs/2026-07-03-page-detail-citations-redesign-design.md` (boule-debate amended). Read it if a behavior question is not answered here.

## Global Constraints

- `app/Cargo.toml:83` keeps `wenlan-types = "0.9.2"` — do NOT bump, patch, or git-pin it (explicit spec decision).
- No new npm or cargo dependencies. Popover positioning is minimal manual flip logic. "Open in browser" uses the already-installed `@tauri-apps/plugin-shell` (capability `shell:allow-open` is already granted in `app/capabilities/default.json:14`).
- All marker stripping/rewriting is display-only. Never modify stored page content.
- Unverified citations are never hidden — they render dashed/muted with an explicit "unverified" note.
- Backend-parity rule: occurrence counting is a plain `\[(\d+)\]` regex over the raw stored body — no code-fence, wikilink, or context awareness. Matches inside code ARE counted (consume occurrence indices) but are NOT rewritten. Do not "improve" this; parity with the backend is the contract.
- Diagnosability strings are exact (spec section 5): `Citations: N (M unverified)` / `Citations cleared by edit — re-distill to restore` / `Citation data mismatched — re-distill to repair` / omitted.
- Every new file starts with `// SPDX-License-Identifier: AGPL-3.0-only` (repo convention, enforced by review).
- CI gates that must pass before the final commit: `cargo fmt --check --all` (run from repo root), `cargo clippy --workspace --all-targets -- -D warnings`, `pnpm exec tsc -b`, `pnpm test`, `cd app && cargo test`.
- Out of scope (do not touch): the actions toolbar, edit mode, the TLDR sentence heuristic (beyond marker stripping), `get_page_sources` typing, the app shell.

## File Structure

| File | Action | Responsibility |
|---|---|---|
| `app/src/search.rs` | Modify (~2932-2962, ~3198-3208, tests mod) | `get_page` / `get_page_revisions` raw-JSON passthrough; remove `GetPageWire` |
| `app/src/api.rs` | Modify (549-555) | client `get_page_revisions` returns `serde_json::Value` |
| `src/lib/tauri.ts` | Modify | `PageCitation` type; `Page.citations`; `PageChangelogEntry.citations_summary` |
| `src/lib/pageCitations.ts` | Create | pure marker-processing util (counting, rewriting, stripping, labels) |
| `src/lib/pageCitations.test.ts` | Create | util unit tests |
| `src/components/memory/ContentRenderer.tsx` | Modify | optional `renderCitation` prop mapping `#citation:` anchors to chips |
| `src/components/memory/ContentRenderer.test.tsx` | Modify | renderCitation cases |
| `src/components/memory/page/format.ts` | Create | shared display helpers (`relativeMs`, `prettyAgent`, `sourceKindLabel`) |
| `src/components/memory/page/CitationChip.tsx` | Create | inline chip button + hover/focus popover lifecycle |
| `src/components/memory/page/CitationPopover.tsx` | Create | popover content per source_kind + viewport flip |
| `src/components/memory/page/CitationChip.test.tsx` | Create | chip + popover tests |
| `src/components/memory/page/RelatedPages.tsx` | Create | outbound-link cards |
| `src/components/memory/page/RelatedPages.test.tsx` | Create | related-pages tests |
| `src/components/memory/page/PageInfo.tsx` | Create | collapsed disclosure: sources / backlinks / revisions / diagnosability |
| `src/components/memory/page/PageInfo.test.tsx` | Create | page-info tests |
| `src/components/memory/PageDetail.tsx` | Modify | orchestrator: citations pipeline, mount new components, delete old sections |
| `src/components/memory/PageDetail.test.tsx` | Modify | update for new structure |
| `src/components/memory/PageDetail.links-revisions.test.tsx` | Modify | update for new structure; invert orphan-links test |
| `src/components/memory/PageDetail.citations.test.tsx` | Create | citations integration tests |
| `src/test/setup.ts` | Modify | global mock for `@tauri-apps/plugin-shell` |

Task order matters: 1 (Rust) and 2 (util) are independent; 3-6 depend on 2; 7 depends on 4-6; 8 depends on all.

---

### Task 1: Rust raw-JSON passthrough for `get_page` and `get_page_revisions`

The daemon's new `citations` field never reaches the frontend today: `get_page` deserializes into the pinned `wenlan-types 0.9.2` `Page` struct and serde silently drops unknown fields. Switch both page-read commands to `serde_json::Value` passthrough. The Rust layer consumes no fields from these responses; TypeScript types them.

**Files:**
- Modify: `app/src/search.rs:2932-2962` (get_page + GetPageWire), `app/src/search.rs:3198-3208` (get_page_revisions), tests `mod tests` in the same file
- Modify: `app/src/api.rs:549-555` (client method return type)

**Interfaces:**
- Consumes: `WenlanClient::get_json<T: DeserializeOwned>(&self, path: &str) -> Result<T, String>` (`app/src/api.rs:173`).
- Produces: Tauri commands `get_page(id: String) -> Result<Option<serde_json::Value>, String>` and `get_page_revisions(page_id: String) -> Result<serde_json::Value, String>`. Wire shapes are unchanged JSON, so the TS functions `getPage` / `getPageRevisions` in `src/lib/tauri.ts` keep their signatures — later tasks only add optional fields to the TS interfaces.
- Note: `app/src/api.rs:1519` has a compile-time assertion `let _page = WenlanClient::get_page_revisions;` — it stays valid because only the return type changes.

- [ ] **Step 1: Write the failing tests**

In `app/src/search.rs`, inside the existing `mod tests` block (it already uses `serde_json::json!` — search for `mod tests` in the file), add:

```rust
    #[test]
    fn page_from_wire_extracts_page_object_with_unknown_fields() {
        let wire = serde_json::json!({
            "page": { "id": "p1", "citations": [{ "occurrence": 1, "marker": 1 }] }
        });
        let page = page_from_wire(wire).expect("page present");
        assert_eq!(page["id"], "p1");
        assert_eq!(page["citations"][0]["occurrence"], 1);
    }

    #[test]
    fn page_from_wire_maps_null_and_missing_page_to_none() {
        assert!(page_from_wire(serde_json::json!({ "page": null })).is_none());
        assert!(page_from_wire(serde_json::json!({})).is_none());
    }
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd app && cargo test page_from_wire`
Expected: compile error — `cannot find function page_from_wire`.

- [ ] **Step 3: Implement the passthrough**

In `app/src/search.rs`, replace the `GetPageWire` struct and `get_page` command (currently lines 2934-2962) with:

```rust
/// Extract the `page` object from the daemon's `{ "page": {...} }` wrapper.
/// Raw-JSON passthrough: the pinned wenlan-types structs would silently drop
/// fields the daemon added after 0.9.2 (e.g. `citations`); the Rust layer
/// consumes no Page fields on this path, so TypeScript types the response.
fn page_from_wire(mut wire: serde_json::Value) -> Option<serde_json::Value> {
    match wire.get_mut("page") {
        Some(v) if !v.is_null() => Some(v.take()),
        _ => None,
    }
}

#[tauri::command]
pub async fn get_page(
    state: tauri::State<'_, State>,
    id: String,
) -> Result<Option<serde_json::Value>, String> {
    let client = state.read().await.client.clone();
    // The daemon returns 404 when the page doesn't exist, which reqwest
    // turns into an error. Distinguish "not found" from real errors so the
    // frontend sees None for the former and a real error for the latter —
    // rather than the previous silent `Err(_) => Ok(None)` which hid
    // wrapper/deserialization bugs behind a "not found" UI.
    match client
        .get_json::<serde_json::Value>(&format!("/api/pages/{}", id))
        .await
    {
        Ok(wire) => Ok(page_from_wire(wire)),
        Err(e) => {
            let msg = e.to_string();
            if msg.contains("404") || msg.to_lowercase().contains("not found") {
                Ok(None)
            } else {
                Err(format!("get_page failed: {}", msg))
            }
        }
    }
}
```

In `app/src/api.rs`, change the client method (lines 549-555) to:

```rust
    pub async fn get_page_revisions(
        &self,
        page_id: &str,
    ) -> Result<serde_json::Value, String> {
        let path = format!("/api/pages/{}/revisions", page_id);
        self.get_json(&path).await
    }
```

In `app/src/search.rs`, change the command (lines 3198-3208) to:

```rust
#[tauri::command]
pub async fn get_page_revisions(
    state: tauri::State<'_, State>,
    page_id: String,
) -> Result<serde_json::Value, String> {
    let client = {
        let s = state.read().await;
        s.client.clone()
    };
    client.get_page_revisions(&page_id).await
}
```

If removing `GetPageWire` orphans a `Page` import warning, leave the glob `use wenlan_types::*;` alone — `Page` is still used by `search_pages`/`list_pages`.

- [ ] **Step 4: Run tests and lints to verify they pass**

Run: `cd app && cargo test page_from_wire && cargo test`
Expected: PASS (all pre-existing tests too — `wenlan_client_exposes_revision_history_methods` at `app/src/api.rs:1517` still compiles because it only takes a function pointer).
Run: `cargo fmt --all && cargo clippy --workspace --all-targets -- -D warnings` (from repo root)
Expected: clean.

- [ ] **Step 5: Commit**

```bash
git add app/src/search.rs app/src/api.rs
git commit -m "feat: raw-JSON passthrough for get_page and get_page_revisions"
```

---

### Task 2: Citation types + `pageCitations` util

Pure functions that mirror the backend's marker contract. This is the correctness core — test it hard.

**Files:**
- Modify: `src/lib/tauri.ts` (add types near the `Page` interface, currently line 915, and `PageChangelogEntry`, line 787)
- Create: `src/lib/pageCitations.ts`
- Test: `src/lib/pageCitations.test.ts`

**Interfaces:**
- Produces (types in `src/lib/tauri.ts`):

```ts
export interface PageCitation {
  occurrence: number;
  marker: number;
  source_kind: "memory" | "external_url" | "external_file" | "authored";
  locator: string;
  score: number;
  status: "verified" | "unverified";
  scope: "sentence" | "paragraph";
}
```

plus `citations?: PageCitation[];` added to `Page` and `citations_summary?: string | null;` added to `PageChangelogEntry`.

- Produces (`src/lib/pageCitations.ts`):

```ts
export const CITATION_ANCHOR_PREFIX = "#citation:";
export type CitationState = "cited" | "stripped-empty" | "stripped-mismatch" | "none";
export interface ProcessedCitations {
  content: string;
  state: CitationState;
  byOccurrence: Map<number, PageCitation>;
}
export function processCitations(content: string, citations: PageCitation[] | undefined): ProcessedCitations;
export function stripCitationLinks(text: string): string; // for the TLDR pull-quote
export function citationDisplayLabel(c: PageCitation): string; // chip label
```

- [ ] **Step 1: Write the failing tests**

Create `src/lib/pageCitations.test.ts`:

```ts
// SPDX-License-Identifier: AGPL-3.0-only
import { describe, it, expect } from "vitest";
import type { PageCitation } from "./tauri";
import {
  processCitations,
  stripCitationLinks,
  citationDisplayLabel,
} from "./pageCitations";

const cite = (
  occurrence: number,
  marker: number,
  over: Partial<PageCitation> = {},
): PageCitation => ({
  occurrence,
  marker,
  source_kind: "memory",
  locator: `mem-${marker}`,
  score: 0.9,
  status: "verified",
  scope: "sentence",
  ...over,
});

describe("processCitations", () => {
  it("returns state none and unchanged content when no markers and no citations", () => {
    const r = processCitations("Plain prose. No markers here.", undefined);
    expect(r.state).toBe("none");
    expect(r.content).toBe("Plain prose. No markers here.");
    expect(r.byOccurrence.size).toBe(0);
  });

  it("rewrites markers to occurrence-indexed citation links in body order", () => {
    const r = processCitations("A claim.[1] Another.[2]", [cite(1, 1), cite(2, 2)]);
    expect(r.state).toBe("cited");
    expect(r.content).toBe("A claim.[1](#citation:1) Another.[2](#citation:2)");
    expect(r.byOccurrence.get(1)?.locator).toBe("mem-1");
    expect(r.byOccurrence.get(2)?.locator).toBe("mem-2");
  });

  it("numbers multi-source runs like [1][3] by occurrence, not marker", () => {
    const r = processCitations("Claim.[1][3]", [cite(1, 1), cite(2, 3)]);
    expect(r.state).toBe("cited");
    expect(r.content).toBe("Claim.[1](#citation:1)[2](#citation:2)");
  });

  it("counts matches inside fenced code but does not rewrite them", () => {
    const content = "```\narr[1] access\n```\nA claim.[2]";
    const r = processCitations(content, [cite(1, 1), cite(2, 2)]);
    expect(r.state).toBe("cited");
    expect(r.content).toBe("```\narr[1] access\n```\nA claim.[2](#citation:2)");
  });

  it("counts matches inside inline code but does not rewrite them", () => {
    const content = "Use `x[1]` here.[2]";
    const r = processCitations(content, [cite(1, 1), cite(2, 2)]);
    expect(r.state).toBe("cited");
    expect(r.content).toBe("Use `x[1]` here.[2](#citation:2)");
  });

  it("falls back to strip-all when match count disagrees with citations length", () => {
    const r = processCitations("A.[1] B.[2]", [cite(1, 1)]);
    expect(r.state).toBe("stripped-mismatch");
    expect(r.content).toBe("A. B.");
    expect(r.byOccurrence.size).toBe(0);
  });

  it("falls back to strip-all when a marker value disagrees", () => {
    const r = processCitations("A.[5]", [cite(1, 1)]);
    expect(r.state).toBe("stripped-mismatch");
    expect(r.content).toBe("A.");
    expect(r.content).not.toContain("#citation");
  });

  it("display-strips markers when citations are empty (user-edited page)", () => {
    const r = processCitations("A. [1] B.[2]", []);
    expect(r.state).toBe("stripped-empty");
    expect(r.content).toBe("A. B.");
  });

  it("display-strips markers when citations are absent (old daemon shape)", () => {
    const r = processCitations("A.[1] B.", undefined);
    expect(r.state).toBe("stripped-empty");
    expect(r.content).toBe("A. B.");
  });

  it("collapses doubled spaces left by stripping, mirroring backend strip_markers", () => {
    const r = processCitations("A. [1] middle [2] B.", []);
    expect(r.content).toBe("A. middle B.");
  });
});

describe("stripCitationLinks", () => {
  it("removes rewritten citation links and collapses spacing", () => {
    expect(stripCitationLinks("First claim [1](#citation:1) done.")).toBe(
      "First claim done.",
    );
    expect(stripCitationLinks("Tail.[2](#citation:2)")).toBe("Tail.");
  });
});

describe("citationDisplayLabel", () => {
  it("shows memory locators as-is", () => {
    expect(citationDisplayLabel(cite(1, 1))).toBe("mem-1");
  });

  it("truncates URLs to hostname", () => {
    const c = cite(1, 1, {
      source_kind: "external_url",
      locator: "https://docs.rs/serde/latest/serde/",
    });
    expect(citationDisplayLabel(c)).toBe("docs.rs");
  });

  it("truncates file paths to basename", () => {
    const c = cite(1, 1, {
      source_kind: "external_file",
      locator: "/Users/l/notes/design.md",
    });
    expect(citationDisplayLabel(c)).toBe("design.md");
  });

  it("labels authored citations", () => {
    expect(citationDisplayLabel(cite(1, 1, { source_kind: "authored" }))).toBe(
      "authored",
    );
  });
});
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pnpm vitest run src/lib/pageCitations.test.ts`
Expected: FAIL — `Cannot find module './pageCitations'`.

- [ ] **Step 3: Add the types and implement the util**

In `src/lib/tauri.ts`: insert the `PageCitation` interface directly above `export interface Page` (line 915); add `citations?: PageCitation[];` as the last field of `Page`; add `citations_summary?: string | null;` as the last field of `PageChangelogEntry` (line 787).

Create `src/lib/pageCitations.ts`:

```ts
// SPDX-License-Identifier: AGPL-3.0-only
// Display-side mirror of the backend's per-claim citation contract
// (7xuanlu/wenlan crates/wenlan-core/src/citations.rs). Occurrence k is the
// k-th plain \[(\d+)\] regex match over the raw stored body — deliberately no
// code-fence/wikilink awareness, because the backend counts the same way and
// parity is the contract. Matches inside code consume occurrence indices but
// keep their raw [N] display (a rewritten link inside a code span would render
// as literal garbage). All transforms are display-only.
import type { PageCitation } from "./tauri";

export const CITATION_ANCHOR_PREFIX = "#citation:";

export type CitationState = "cited" | "stripped-empty" | "stripped-mismatch" | "none";

export interface ProcessedCitations {
  /** Body with markers rewritten to [k](#citation:k), or display-stripped. */
  content: string;
  state: CitationState;
  /** 1-based occurrence -> citation; empty unless state === "cited". */
  byOccurrence: Map<number, PageCitation>;
}

const MARKER_RE = /\[(\d+)\]/g;

/** Character ranges covered by fenced blocks or inline code spans. */
function codeRanges(content: string): Array<[number, number]> {
  const ranges: Array<[number, number]> = [];
  const fence = /^```.*$/gm;
  let openAt: number | null = null;
  let m: RegExpExecArray | null;
  while ((m = fence.exec(content)) !== null) {
    if (openAt === null) {
      openAt = m.index;
    } else {
      ranges.push([openAt, m.index + m[0].length]);
      openAt = null;
    }
  }
  if (openAt !== null) ranges.push([openAt, content.length]);
  const inline = /`[^`\n]*`/g;
  while ((m = inline.exec(content)) !== null) {
    const start = m.index;
    if (!ranges.some(([s, e]) => start >= s && start < e)) {
      ranges.push([start, start + m[0].length]);
    }
  }
  return ranges;
}

/** Mirror the backend's strip_markers: remove [N], collapse doubled spaces. */
function stripMarkers(text: string): string {
  return text.replace(/\[\d+\]/g, "").replace(/ {2,}/g, " ");
}

/** Remove rewritten citation links from plain-text contexts (TLDR pull-quote). */
export function stripCitationLinks(text: string): string {
  return text
    .replace(/\[\d+\]\(#citation:\d+\)/g, "")
    .replace(/ {2,}/g, " ")
    .trim();
}

export function processCitations(
  content: string,
  citations: PageCitation[] | undefined,
): ProcessedCitations {
  const matches = [...content.matchAll(MARKER_RE)];
  if (matches.length === 0 && (!citations || citations.length === 0)) {
    return { content, state: "none", byOccurrence: new Map() };
  }
  if (!citations || citations.length === 0) {
    // Verified backend behavior: a user content edit resets citations to []
    // but stores the markers verbatim — without this strip every edited page
    // renders permanent [N] noise.
    return { content: stripMarkers(content), state: "stripped-empty", byOccurrence: new Map() };
  }
  const byOccurrence = new Map<number, PageCitation>();
  for (const c of citations) byOccurrence.set(c.occurrence, c);
  // Conservative fallback: any count or marker disagreement means the mapping
  // is untrustworthy — misattributed citations are worse than none.
  const mismatch =
    matches.length !== citations.length ||
    matches.some((m, i) => byOccurrence.get(i + 1)?.marker !== Number(m[1]));
  if (mismatch) {
    return { content: stripMarkers(content), state: "stripped-mismatch", byOccurrence: new Map() };
  }
  const ranges = codeRanges(content);
  const inCode = (idx: number) => ranges.some(([s, e]) => idx >= s && idx < e);
  let out = "";
  let last = 0;
  matches.forEach((m, i) => {
    const at = m.index ?? 0;
    if (inCode(at)) return; // counted, but displayed raw
    const k = i + 1;
    out += content.slice(last, at) + `[${k}](${CITATION_ANCHOR_PREFIX}${k})`;
    last = at + m[0].length;
  });
  out += content.slice(last);
  return { content: out, state: "cited", byOccurrence };
}

/** Short chip label: memory ids as-is, URLs to hostname, paths to basename. */
export function citationDisplayLabel(c: PageCitation): string {
  if (c.source_kind === "external_url") {
    try {
      return new URL(c.locator).hostname;
    } catch {
      return c.locator.slice(0, 24);
    }
  }
  if (c.source_kind === "external_file") {
    return c.locator.split("/").filter(Boolean).pop() ?? c.locator;
  }
  if (c.source_kind === "authored") return "authored";
  return c.locator;
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pnpm vitest run src/lib/pageCitations.test.ts && pnpm exec tsc -b`
Expected: all PASS, tsc clean. Note: `src/lib/tauri.ts` is coverage-gated (90/90/85/90) but the additions are type-only — no runtime lines added, so the gate is unaffected. Verify with `pnpm test:coverage` if in doubt.

- [ ] **Step 5: Commit**

```bash
git add src/lib/tauri.ts src/lib/pageCitations.ts src/lib/pageCitations.test.ts
git commit -m "feat: page citation types and marker-processing util"
```

---

### Task 3: `ContentRenderer` renderCitation hook

`ContentRenderer` already owns the markdown `a` component override (`src/components/memory/ContentRenderer.tsx:79`). Add an optional `renderCitation` prop: when set (detail variant), `#citation:k` hrefs render the caller's node instead of an anchor. All other consumers of ContentRenderer are untouched (prop is optional).

**Files:**
- Modify: `src/components/memory/ContentRenderer.tsx`
- Test: `src/components/memory/ContentRenderer.test.tsx` (append a describe block)

**Interfaces:**
- Consumes: `CITATION_ANCHOR_PREFIX` from `src/lib/pageCitations` (Task 2).
- Produces: `ContentRendererProps` gains `renderCitation?: (occurrence: number) => React.ReactNode;`. Callers pass content whose citation links were produced by `processCitations` (link text `[k]`, href `#citation:k`).

- [ ] **Step 1: Write the failing tests**

Append to `src/components/memory/ContentRenderer.test.tsx`:

```tsx
describe("ContentRenderer citation links", () => {
  it("renders #citation: links through renderCitation", () => {
    render(
      <ContentRenderer
        content="A claim.[1](#citation:1) More text."
        variant="detail"
        renderCitation={(k) => <button data-testid={`chip-${k}`}>chip {k}</button>}
      />,
    );
    expect(screen.getByTestId("chip-1")).toBeInTheDocument();
    expect(screen.queryByRole("link", { name: "1" })).toBeNull();
  });

  it("leaves ordinary links alone when renderCitation is set", () => {
    const { container } = render(
      <ContentRenderer
        content="See [docs](https://example.com) and a claim.[1](#citation:1)"
        variant="detail"
        renderCitation={(k) => <span data-testid={`chip-${k}`} />}
      />,
    );
    const link = container.querySelector('a[href="https://example.com"]');
    expect(link).not.toBeNull();
    expect(link!.getAttribute("target")).toBe("_blank");
  });

  it("renders #citation: hrefs as plain anchors when renderCitation is absent", () => {
    const { container } = render(
      <ContentRenderer content="A claim.[1](#citation:1)" variant="detail" />,
    );
    expect(container.querySelector('a[href="#citation:1"]')).not.toBeNull();
  });
});
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pnpm vitest run src/components/memory/ContentRenderer.test.tsx`
Expected: FAIL — TS error / chip testid not found (prop doesn't exist yet).

- [ ] **Step 3: Implement**

In `src/components/memory/ContentRenderer.tsx`:

1. Add the import: `import { CITATION_ANCHOR_PREFIX } from "../../lib/pageCitations";`
2. Extend the props interface:

```tsx
interface ContentRendererProps {
  content: string;
  structuredFields?: string | null;
  variant: "card" | "detail";
  className?: string;
  /** Render #citation:k links as inline chips (page detail). */
  renderCitation?: (occurrence: number) => React.ReactNode;
}
```

3. In the component, destructure `renderCitation` and build the components map right before the detail-variant `return`:

```tsx
  const components = renderCitation
    ? {
        ...markdownComponents,
        a: (props: { href?: string; children?: React.ReactNode }) => {
          const href = props.href ?? "";
          if (href.startsWith(CITATION_ANCHOR_PREFIX)) {
            const k = Number(href.slice(CITATION_ANCHOR_PREFIX.length));
            if (Number.isInteger(k) && k > 0) return <>{renderCitation(k)}</>;
          }
          return markdownComponents.a(props);
        },
      }
    : markdownComponents;
```

and pass `components={components}` to `ReactMarkdown` instead of `components={markdownComponents}`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pnpm vitest run src/components/memory/ContentRenderer.test.tsx && pnpm exec tsc -b`
Expected: all PASS (including the pre-existing cases), tsc clean.

- [ ] **Step 5: Commit**

```bash
git add src/components/memory/ContentRenderer.tsx src/components/memory/ContentRenderer.test.tsx
git commit -m "feat: ContentRenderer renderCitation hook for #citation: links"
```

---

### Task 4: `CitationChip` + `CitationPopover` (+ shared format helpers)

The inline chip and its popover. Accessibility is required, not decorative: real `<button>`, `role="tooltip"` + `aria-describedby`, opens on hover (~150 ms) and focus, closes on blur/mouse-out/Escape, viewport flip, focus stays on the chip.

**Files:**
- Create: `src/components/memory/page/format.ts` (copies of PageDetail's display helpers — PageDetail's own copies are deleted in Task 7)
- Create: `src/components/memory/page/CitationChip.tsx`
- Create: `src/components/memory/page/CitationPopover.tsx`
- Modify: `src/test/setup.ts` (add plugin-shell mock)
- Test: `src/components/memory/page/CitationChip.test.tsx`

**Interfaces:**
- Consumes: `PageCitation`, `MemoryItem` from `src/lib/tauri`; `citationDisplayLabel` from `src/lib/pageCitations` (Task 2); `open` from `@tauri-apps/plugin-shell`.
- Produces:

```tsx
// CitationChip.tsx (default export)
interface CitationChipProps {
  occurrence: number;
  citation: PageCitation;
  sourceMemory: MemoryItem | null; // resolved from page-sources; null = not found
  sourcesLoading: boolean;         // page-sources query still in flight
  onOpenMemory: (sourceId: string) => void;
}
```

```ts
// format.ts
export function relativeMs(ms: number): string;
export function prettyAgent(name: string | null | undefined): string;
export function sourceKindLabel(mem: MemoryItem): string;
```

Chip primary action (click): `memory` → `onOpenMemory(citation.locator)`; `external_url` → shell `open(locator)`; `external_file`/`authored` → toggle popover. Touch first-tap opens the popover instead.

- [ ] **Step 1: Add the plugin-shell mock and format helpers**

Append to `src/test/setup.ts` (after the plugin-fs mock):

```ts
vi.mock('@tauri-apps/plugin-shell', () => ({
  open: vi.fn(),
}));
```

Create `src/components/memory/page/format.ts` (bodies copied verbatim from `src/components/memory/PageDetail.tsx:58-94` — `KNOWN_AGENTS`, `prettyAgent`, `SOURCE_KIND_LABEL`, `sourceKindLabel`, `relativeMs`):

```ts
// SPDX-License-Identifier: AGPL-3.0-only
// Shared display helpers for the page-detail subcomponents.
import type { MemoryItem } from "../../../lib/tauri";

const KNOWN_AGENTS: Record<string, string> = {
  "claude-code": "Claude Code",
  "claude-desktop": "Claude Desktop",
  cursor: "Cursor",
  "chatgpt-mcp": "ChatGPT",
  chatgpt: "ChatGPT",
  "gemini-cli": "Gemini CLI",
  windsurf: "Windsurf",
  zed: "Zed",
};

export function prettyAgent(name: string | null | undefined): string {
  if (!name) return "unknown agent";
  const key = name.trim().toLowerCase();
  return KNOWN_AGENTS[key] ?? name;
}

const SOURCE_KIND_LABEL: Record<string, string> = {
  memory: "memory",
  chat: "chat",
  file: "file",
  obsidian: "obsidian",
  web: "web",
};

export function sourceKindLabel(mem: MemoryItem): string {
  const mt = mem.memory_type?.toLowerCase() ?? "";
  return SOURCE_KIND_LABEL[mt] ?? (mt || "memory");
}

export function relativeMs(ms: number): string {
  const delta = Date.now() - ms;
  if (delta < 60_000) return "just now";
  if (delta < 3_600_000) return `${Math.floor(delta / 60_000)}m ago`;
  if (delta < 86_400_000) return `${Math.floor(delta / 3_600_000)}h ago`;
  return `${Math.floor(delta / 86_400_000)}d ago`;
}
```

- [ ] **Step 2: Write the failing tests**

Create `src/components/memory/page/CitationChip.test.tsx`:

```tsx
// SPDX-License-Identifier: AGPL-3.0-only
import { describe, it, expect, vi, afterEach } from "vitest";
import { render, screen, fireEvent, act } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { open as shellOpen } from "@tauri-apps/plugin-shell";
import type { MemoryItem, PageCitation } from "../../../lib/tauri";
import CitationChip from "./CitationChip";

const cite = (over: Partial<PageCitation> = {}): PageCitation => ({
  occurrence: 1,
  marker: 1,
  source_kind: "memory",
  locator: "mem-1",
  score: 0.9,
  status: "verified",
  scope: "sentence",
  ...over,
});

const memory = (over: Partial<MemoryItem> = {}): MemoryItem => ({
  source_id: "mem-1",
  title: "Design decision",
  content: "We decided to keep the daemon local-first because it is simpler.",
  summary: null,
  memory_type: "memory",
  domain: null,
  source_agent: "claude-code",
  confidence: null,
  confirmed: true,
  pinned: false,
  supersedes: null,
  last_modified: Math.floor(Date.now() / 1000),
  chunk_count: 1,
  ...over,
});

function renderChip(over: Partial<React.ComponentProps<typeof CitationChip>> = {}) {
  const onOpenMemory = vi.fn();
  const utils = render(
    <CitationChip
      occurrence={1}
      citation={cite()}
      sourceMemory={memory()}
      sourcesLoading={false}
      onOpenMemory={onOpenMemory}
      {...over}
    />,
  );
  return { onOpenMemory, ...utils };
}

afterEach(() => {
  vi.useRealTimers();
});

describe("CitationChip", () => {
  it("renders a focusable button with locator label and occurrence superscript", () => {
    renderChip();
    const chip = screen.getByRole("button", { name: /mem-1/ });
    expect(chip).toBeInTheDocument();
    expect(chip.querySelector("sup")?.textContent).toBe("1");
    expect(chip).toHaveAttribute("data-status", "verified");
  });

  it("marks unverified citations", () => {
    renderChip({ citation: cite({ status: "unverified" }) });
    expect(screen.getByRole("button", { name: /mem-1/ })).toHaveAttribute(
      "data-status",
      "unverified",
    );
  });

  it("opens the popover on focus and links it via aria-describedby", () => {
    renderChip();
    const chip = screen.getByRole("button", { name: /mem-1/ });
    fireEvent.focus(chip);
    const tip = screen.getByRole("tooltip");
    expect(tip).toBeInTheDocument();
    expect(chip.getAttribute("aria-describedby")).toBe(tip.getAttribute("id"));
  });

  it("closes the popover on Escape", () => {
    renderChip();
    const chip = screen.getByRole("button", { name: /mem-1/ });
    fireEvent.focus(chip);
    expect(screen.getByRole("tooltip")).toBeInTheDocument();
    fireEvent.keyDown(chip, { key: "Escape" });
    expect(screen.queryByRole("tooltip")).toBeNull();
  });

  it("opens on hover after a delay and closes on mouse-out", () => {
    vi.useFakeTimers();
    renderChip();
    const wrapper = screen.getByRole("button", { name: /mem-1/ }).parentElement!;
    fireEvent.mouseEnter(wrapper);
    expect(screen.queryByRole("tooltip")).toBeNull();
    act(() => vi.advanceTimersByTime(200));
    expect(screen.getByRole("tooltip")).toBeInTheDocument();
    fireEvent.mouseLeave(wrapper);
    act(() => vi.advanceTimersByTime(200));
    expect(screen.queryByRole("tooltip")).toBeNull();
  });

  it("shows memory details and opens the memory from the popover action", async () => {
    const user = userEvent.setup();
    const { onOpenMemory } = renderChip();
    fireEvent.focus(screen.getByRole("button", { name: /mem-1/ }));
    expect(screen.getByText("Design decision")).toBeInTheDocument();
    expect(screen.getByText("Source memory")).toBeInTheDocument();
    expect(screen.getByText(/We decided to keep the daemon/)).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: /Open memory/ }));
    expect(onOpenMemory).toHaveBeenCalledWith("mem-1");
  });

  it("clicking a memory chip opens the memory directly", async () => {
    const user = userEvent.setup();
    const { onOpenMemory } = renderChip();
    await user.click(screen.getByRole("button", { name: /mem-1/ }));
    expect(onOpenMemory).toHaveBeenCalledWith("mem-1");
  });

  it("shows 'source not available' when the locator does not resolve", () => {
    renderChip({ sourceMemory: null, sourcesLoading: false });
    fireEvent.focus(screen.getByRole("button", { name: /mem-1/ }));
    expect(screen.getByText(/source not available/i)).toBeInTheDocument();
  });

  it("shows a skeleton while sources are loading", () => {
    renderChip({ sourceMemory: null, sourcesLoading: true });
    fireEvent.focus(screen.getByRole("button", { name: /mem-1/ }));
    expect(screen.getByTestId("citation-popover-skeleton")).toBeInTheDocument();
  });

  it("notes unverified status in the popover", () => {
    renderChip({ citation: cite({ status: "unverified" }) });
    fireEvent.focus(screen.getByRole("button", { name: /mem-1/ }));
    expect(screen.getByText(/unverified/i)).toBeInTheDocument();
  });

  it("opens external urls in the browser", async () => {
    const user = userEvent.setup();
    renderChip({
      citation: cite({ source_kind: "external_url", locator: "https://docs.rs/serde" }),
      sourceMemory: null,
    });
    await user.click(screen.getByRole("button", { name: /docs\.rs/ }));
    expect(vi.mocked(shellOpen)).toHaveBeenCalledWith("https://docs.rs/serde");
  });

  it("shows the file path with no action for external_file", () => {
    renderChip({
      citation: cite({ source_kind: "external_file", locator: "/notes/design.md" }),
      sourceMemory: null,
    });
    fireEvent.focus(screen.getByRole("button", { name: /design\.md/ }));
    expect(screen.getByText("/notes/design.md")).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /Open/ })).toBeNull();
  });

  it("labels authored citations as written directly", () => {
    renderChip({ citation: cite({ source_kind: "authored" }), sourceMemory: null });
    fireEvent.focus(screen.getByRole("button", { name: /authored/ }));
    expect(screen.getByText(/written directly/i)).toBeInTheDocument();
  });
});
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `pnpm vitest run src/components/memory/page/CitationChip.test.tsx`
Expected: FAIL — `Cannot find module './CitationChip'`.

- [ ] **Step 4: Implement the popover**

Create `src/components/memory/page/CitationPopover.tsx`:

```tsx
// SPDX-License-Identifier: AGPL-3.0-only
import { useLayoutEffect, useRef, useState } from "react";
import { open as shellOpen } from "@tauri-apps/plugin-shell";
import type { MemoryItem, PageCitation } from "../../../lib/tauri";
import { relativeMs } from "./format";

interface CitationPopoverProps {
  id: string;
  citation: PageCitation;
  sourceMemory: MemoryItem | null;
  sourcesLoading: boolean;
  anchorRef: React.RefObject<HTMLButtonElement | null>;
  onOpenMemory: (sourceId: string) => void;
}

const WIDTH = 280;

// Spec: external_url shows a *domain* badge, other kinds a fixed label.
function kindBadge(citation: PageCitation): string {
  if (citation.source_kind === "external_url") {
    try {
      return new URL(citation.locator).hostname;
    } catch {
      return "Web";
    }
  }
  return { memory: "Source memory", external_file: "File", authored: "Authored" }[
    citation.source_kind
  ];
}

export default function CitationPopover({
  id,
  citation,
  sourceMemory,
  sourcesLoading,
  anchorRef,
  onOpenMemory,
}: CitationPopoverProps) {
  const boxRef = useRef<HTMLDivElement>(null);
  const [pos, setPos] = useState<{ top: number; left: number } | null>(null);

  // Minimal viewport collision handling: below the chip by default, flip
  // above when it would overflow the bottom, clamp horizontally.
  useLayoutEffect(() => {
    const anchor = anchorRef.current?.getBoundingClientRect();
    if (!anchor) return;
    const height = boxRef.current?.getBoundingClientRect().height ?? 120;
    const flip =
      anchor.bottom + height + 8 > window.innerHeight && anchor.top - height - 8 > 0;
    setPos({
      top: flip ? anchor.top - height - 8 : anchor.bottom + 4,
      left: Math.min(Math.max(anchor.left, 8), Math.max(window.innerWidth - WIDTH - 8, 8)),
    });
  }, [anchorRef]);

  const mono = {
    fontFamily: "var(--mem-font-mono)",
    fontSize: "10px",
    color: "var(--mem-text-tertiary)",
  } as const;
  const bodyText = {
    fontFamily: "var(--mem-font-body)",
    fontSize: "12px",
    color: "var(--mem-text-secondary)",
    lineHeight: 1.5,
  } as const;
  const actionStyle = {
    fontFamily: "var(--mem-font-body)",
    fontSize: "11px",
    fontWeight: 500,
    color: "var(--mem-accent-indigo)",
    background: "none",
    border: "none",
    padding: 0,
    cursor: "pointer",
  } as const;

  const snippet = sourceMemory?.content
    ? sourceMemory.content.replace(/\s+/g, " ").trim().slice(0, 200)
    : null;

  function body() {
    if (citation.source_kind === "authored") {
      return <p style={bodyText}>Written directly in this page.</p>;
    }
    if (citation.source_kind === "external_file") {
      return <p style={mono}>{citation.locator}</p>;
    }
    if (citation.source_kind === "external_url") {
      return (
        <>
          <p style={{ ...mono, wordBreak: "break-all" }}>{citation.locator}</p>
          <button style={actionStyle} onClick={() => void shellOpen(citation.locator)}>
            Open in browser →
          </button>
        </>
      );
    }
    // memory
    if (sourcesLoading && !sourceMemory) {
      return (
        <div data-testid="citation-popover-skeleton" className="flex flex-col gap-1.5">
          <div style={{ width: "70%", height: "10px", background: "var(--mem-hover)", borderRadius: "4px" }} />
          <div style={{ width: "90%", height: "10px", background: "var(--mem-hover)", borderRadius: "4px" }} />
        </div>
      );
    }
    if (!sourceMemory) {
      return (
        <>
          <p style={mono}>{citation.locator}</p>
          <p style={{ ...bodyText, fontStyle: "italic" }}>source not available</p>
        </>
      );
    }
    return (
      <>
        {sourceMemory.title && (
          <p
            style={{
              fontFamily: "var(--mem-font-heading)",
              fontSize: "13px",
              fontWeight: 500,
              color: "var(--mem-text)",
              lineHeight: 1.4,
            }}
          >
            {sourceMemory.title}
          </p>
        )}
        <p style={mono}>
          {citation.locator}
          {sourceMemory.last_modified
            ? ` · ${relativeMs(sourceMemory.last_modified * 1000)}`
            : ""}
        </p>
        {snippet && <p style={bodyText}>{snippet}</p>}
        <button style={actionStyle} onClick={() => onOpenMemory(citation.locator)}>
          Open memory →
        </button>
      </>
    );
  }

  return (
    <div
      ref={boxRef}
      id={id}
      role="tooltip"
      className="flex flex-col gap-1.5 rounded-lg p-3"
      style={{
        position: "fixed",
        top: pos?.top ?? -9999,
        left: pos?.left ?? -9999,
        width: `${WIDTH}px`,
        zIndex: 50,
        backgroundColor: "var(--mem-surface)",
        border: "1px solid var(--mem-border)",
        boxShadow: "0 4px 12px rgba(0,0,0,0.3)",
      }}
    >
      <div className="flex items-center gap-2">
        <span
          style={{
            fontFamily: "var(--mem-font-mono)",
            fontSize: "10px",
            color: "var(--mem-text-tertiary)",
            background: "var(--mem-hover)",
            padding: "1px 5px",
            borderRadius: "3px",
          }}
        >
          {kindBadge(citation)}
        </span>
        {citation.status === "unverified" && (
          <span
            style={{
              fontFamily: "var(--mem-font-mono)",
              fontSize: "10px",
              color: "var(--mem-accent-amber)",
            }}
          >
            unverified
          </span>
        )}
      </div>
      {body()}
    </div>
  );
}
```

- [ ] **Step 5: Implement the chip**

Create `src/components/memory/page/CitationChip.tsx`:

```tsx
// SPDX-License-Identifier: AGPL-3.0-only
import { useEffect, useId, useRef, useState } from "react";
import { open as shellOpen } from "@tauri-apps/plugin-shell";
import type { MemoryItem, PageCitation } from "../../../lib/tauri";
import { citationDisplayLabel } from "../../../lib/pageCitations";
import CitationPopover from "./CitationPopover";

interface CitationChipProps {
  occurrence: number;
  citation: PageCitation;
  sourceMemory: MemoryItem | null;
  sourcesLoading: boolean;
  onOpenMemory: (sourceId: string) => void;
}

const HOVER_OPEN_DELAY_MS = 150;
const HOVER_CLOSE_GRACE_MS = 120;

export default function CitationChip({
  occurrence,
  citation,
  sourceMemory,
  sourcesLoading,
  onOpenMemory,
}: CitationChipProps) {
  const [open, setOpen] = useState(false);
  const chipRef = useRef<HTMLButtonElement>(null);
  const openTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const closeTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const lastPointerType = useRef("mouse");
  const popoverId = useId();

  const clearTimers = () => {
    if (openTimer.current) clearTimeout(openTimer.current);
    if (closeTimer.current) clearTimeout(closeTimer.current);
  };
  useEffect(() => clearTimers, []);

  const activate = () => {
    // Touch has no hover: first tap opens the popover, its buttons navigate.
    if (lastPointerType.current === "touch" && !open) {
      setOpen(true);
      return;
    }
    if (citation.source_kind === "memory") {
      onOpenMemory(citation.locator);
    } else if (citation.source_kind === "external_url") {
      void shellOpen(citation.locator);
    } else {
      setOpen((v) => !v);
    }
  };

  const unverified = citation.status === "unverified";

  return (
    <span
      style={{ position: "relative", display: "inline-block" }}
      onMouseEnter={() => {
        if (closeTimer.current) clearTimeout(closeTimer.current);
        openTimer.current = setTimeout(() => setOpen(true), HOVER_OPEN_DELAY_MS);
      }}
      onMouseLeave={() => {
        if (openTimer.current) clearTimeout(openTimer.current);
        closeTimer.current = setTimeout(() => setOpen(false), HOVER_CLOSE_GRACE_MS);
      }}
      onFocus={() => setOpen(true)}
      onBlur={(e) => {
        // Keep open while focus moves into the popover (its action button).
        if (!e.currentTarget.contains(e.relatedTarget as Node | null)) setOpen(false);
      }}
      onKeyDown={(e) => {
        if (e.key === "Escape") {
          clearTimers();
          setOpen(false);
        }
      }}
    >
      <button
        ref={chipRef}
        type="button"
        data-status={citation.status}
        aria-describedby={open ? popoverId : undefined}
        onPointerDown={(e) => {
          lastPointerType.current = e.pointerType;
        }}
        onClick={activate}
        className="focus-visible:outline-2 focus-visible:outline-[var(--mem-accent-indigo)]"
        style={{
          fontFamily: "var(--mem-font-mono)",
          fontSize: "10px",
          lineHeight: 1,
          color: unverified ? "var(--mem-text-tertiary)" : "var(--mem-accent-indigo)",
          background: "var(--mem-hover)",
          border: unverified
            ? "1px dashed var(--mem-border)"
            : "1px solid transparent",
          borderRadius: "4px",
          padding: "1px 4px",
          margin: "0 2px",
          verticalAlign: "baseline",
          cursor: "pointer",
        }}
      >
        {citationDisplayLabel(citation)}
        <sup style={{ marginLeft: "1px" }}>{occurrence}</sup>
      </button>
      {open && (
        <CitationPopover
          id={popoverId}
          citation={citation}
          sourceMemory={sourceMemory}
          sourcesLoading={sourcesLoading}
          anchorRef={chipRef}
          onOpenMemory={onOpenMemory}
        />
      )}
    </span>
  );
}
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `pnpm vitest run src/components/memory/page/CitationChip.test.tsx && pnpm exec tsc -b`
Expected: all PASS. If the hover test flakes on timer interplay, the fix is in the test (use `fireEvent` + fake timers as written, not `userEvent.hover`), not in the component.

- [ ] **Step 7: Commit**

```bash
git add src/test/setup.ts src/components/memory/page/
git commit -m "feat: CitationChip and CitationPopover with a11y and per-kind content"
```

---

### Task 5: `RelatedPages` component

Outbound links as cards. Resolved targets are clickable buttons; unresolved targets are muted, inert `div`s (today they're styled identically to clickable rows — the audit finding). Renders nothing when there are zero outbound links.

**Files:**
- Create: `src/components/memory/page/RelatedPages.tsx`
- Test: `src/components/memory/page/RelatedPages.test.tsx`

**Interfaces:**
- Consumes: `PageLinkOutbound { label: string; target_page_id: string | null }` from `src/lib/tauri`.
- Produces:

```tsx
// RelatedPages.tsx (default export)
interface RelatedPagesProps {
  outbound: PageLinkOutbound[];
  onPageClick?: (pageId: string) => void;
}
```

The section root carries `aria-label="Related pages"`.

- [ ] **Step 1: Write the failing tests**

Create `src/components/memory/page/RelatedPages.test.tsx`:

```tsx
// SPDX-License-Identifier: AGPL-3.0-only
import { describe, it, expect, vi } from "vitest";
import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import RelatedPages from "./RelatedPages";

describe("RelatedPages", () => {
  it("renders nothing when there are no outbound links", () => {
    const { container } = render(<RelatedPages outbound={[]} onPageClick={vi.fn()} />);
    expect(container).toBeEmptyDOMElement();
  });

  it("renders resolved links as clickable cards", async () => {
    const user = userEvent.setup();
    const onPageClick = vi.fn();
    render(
      <RelatedPages
        outbound={[{ label: "Resolved Link", target_page_id: "page-2" }]}
        onPageClick={onPageClick}
      />,
    );
    const section = screen.getByLabelText("Related pages");
    await user.click(within(section).getByRole("button", { name: /Resolved Link/ }));
    expect(onPageClick).toHaveBeenCalledWith("page-2");
  });

  it("renders unresolved links muted and inert", () => {
    render(
      <RelatedPages
        outbound={[{ label: "Missing Link", target_page_id: null }]}
        onPageClick={vi.fn()}
      />,
    );
    const section = screen.getByLabelText("Related pages");
    expect(within(section).getByText("Missing Link")).toBeInTheDocument();
    expect(within(section).queryByRole("button", { name: /Missing Link/ })).toBeNull();
  });
});
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pnpm vitest run src/components/memory/page/RelatedPages.test.tsx`
Expected: FAIL — `Cannot find module './RelatedPages'`.

- [ ] **Step 3: Implement**

Create `src/components/memory/page/RelatedPages.tsx`:

```tsx
// SPDX-License-Identifier: AGPL-3.0-only
import type { PageLinkOutbound } from "../../../lib/tauri";

interface RelatedPagesProps {
  outbound: PageLinkOutbound[];
  onPageClick?: (pageId: string) => void;
}

export default function RelatedPages({ outbound, onPageClick }: RelatedPagesProps) {
  // An empty "Related pages" header is noise, not information.
  if (outbound.length === 0) return null;

  return (
    <div aria-label="Related pages">
      <h3
        className="mb-2"
        style={{
          fontFamily: "var(--mem-font-mono)",
          fontSize: "11px",
          fontWeight: 600,
          letterSpacing: "0.05em",
          textTransform: "uppercase",
          color: "var(--mem-text-tertiary)",
        }}
      >
        Related Pages
      </h3>
      <div className="flex flex-wrap gap-1.5">
        {outbound.map((link, idx) => {
          const key = `${link.label}-${link.target_page_id ?? idx}`;
          const inner = (
            <span className="flex items-center gap-2">
              <svg
                width="14"
                height="14"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="1.5"
                style={{ color: "var(--mem-page-icon)" }}
                className="shrink-0"
              >
                <path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5" />
              </svg>
              <span
                style={{
                  fontFamily: "var(--mem-font-body)",
                  fontSize: "13px",
                  fontWeight: 500,
                  color: "var(--mem-text)",
                }}
              >
                {link.label}
              </span>
            </span>
          );
          const targetPageId = link.target_page_id;
          if (!targetPageId || !onPageClick) {
            return (
              <div
                key={key}
                className="rounded-lg px-3 py-2"
                style={{
                  backgroundColor: "var(--mem-surface)",
                  border: "1px solid var(--mem-border)",
                  opacity: 0.55,
                }}
                title="No page exists for this link yet"
              >
                {inner}
              </div>
            );
          }
          return (
            <button
              key={key}
              onClick={() => onPageClick(targetPageId)}
              className="rounded-lg px-3 py-2 text-left transition-colors duration-150 cursor-pointer hover:bg-[var(--mem-hover)]"
              style={{
                backgroundColor: "var(--mem-surface)",
                border: "1px solid var(--mem-border)",
              }}
            >
              {inner}
            </button>
          );
        })}
      </div>
    </div>
  );
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pnpm vitest run src/components/memory/page/RelatedPages.test.tsx && pnpm exec tsc -b`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/components/memory/page/RelatedPages.tsx src/components/memory/page/RelatedPages.test.tsx
git commit -m "feat: RelatedPages cards with inert unresolved links"
```

---

### Task 6: `PageInfo` disclosure

One collapsed `<details>` disclosure replacing Source Memories, inbound Page Links, and Revision History. Always rendered — zero counts are informative (they replace today's vanishing sections). Native `<details>/<summary>` gives collapse + keyboard behavior for free; do not build a custom disclosure.

**Files:**
- Create: `src/components/memory/page/PageInfo.tsx`
- Test: `src/components/memory/page/PageInfo.test.tsx`

**Interfaces:**
- Consumes: `PageSourceWithMemory`, `PageLinkInbound`, `PageChangelogEntry`, `PageCitation` from `src/lib/tauri`; `CitationState` from `src/lib/pageCitations`; `relativeMs`, `prettyAgent`, `sourceKindLabel` from `./format` (Task 4).
- Produces:

```tsx
// PageInfo.tsx (default export)
interface PageInfoProps {
  sourceCount: number;                        // header-line count (fallback-aware, computed by parent)
  sources: PageSourceWithMemory[] | undefined; // undefined = query in flight
  inbound: PageLinkInbound[];
  revisions: PageChangelogEntry[];
  citations: PageCitation[] | undefined;
  citationState: CitationState;
  onMemoryClick: (sourceId: string) => void;
  onPageClick?: (pageId: string) => void;
}
```

Behavior contract:
- Summary line: `Page info` + `{sourceCount} source(s) · {inbound.length} backlink(s) · {revisions.length} revision(s)`.
- Sources rows ordered by first citation occurrence (memory-kind citations only), then uncited rows by `last_modified` descending. Rows show locator chip, title, date, kind badge, `v{n}` tag when `version > 1`, and an `unverified` tag when any citation for that locator is unverified. Click → `onMemoryClick(memory_source_id)`.
- Backlinks show `label` only — never `source_page_id` (the raw-UUID audit finding). Click → `onPageClick(source_page_id)`.
- Revisions rows: existing changelog fields plus a `citations_summary` chip when present.
- Diagnosability line (muted, last row of expanded content), exact strings:
  - `citationState === "cited"` → `Citations: {citations.length} ({M} unverified)`
  - `"stripped-empty"` → `Citations cleared by edit — re-distill to restore`
  - `"stripped-mismatch"` → `Citation data mismatched — re-distill to repair`
  - `"none"` → omitted.

- [ ] **Step 1: Write the failing tests**

Create `src/components/memory/page/PageInfo.test.tsx`:

```tsx
// SPDX-License-Identifier: AGPL-3.0-only
import { describe, it, expect, vi } from "vitest";
import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import type {
  MemoryItem,
  PageChangelogEntry,
  PageCitation,
  PageSourceWithMemory,
} from "../../../lib/tauri";
import PageInfo from "./PageInfo";

const memory = (id: string, over: Partial<MemoryItem> = {}): MemoryItem => ({
  source_id: id,
  title: `Title ${id}`,
  content: `Content of ${id}.`,
  summary: null,
  memory_type: "memory",
  domain: null,
  source_agent: "claude-code",
  confidence: null,
  confirmed: true,
  pinned: false,
  supersedes: null,
  last_modified: 1_700_000_000,
  chunk_count: 1,
  ...over,
});

const source = (id: string, over: Partial<MemoryItem> = {}): PageSourceWithMemory => ({
  source: { page_id: "page-1", memory_source_id: id, linked_at: 0 },
  memory: memory(id, over),
});

const cite = (
  occurrence: number,
  marker: number,
  locator: string,
  over: Partial<PageCitation> = {},
): PageCitation => ({
  occurrence,
  marker,
  source_kind: "memory",
  locator,
  score: 0.9,
  status: "verified",
  scope: "sentence",
  ...over,
});

const revision = (over: Partial<PageChangelogEntry> = {}): PageChangelogEntry => ({
  version: 2,
  at: Math.floor(Date.now() / 1000),
  edited_by: "distill",
  delta_summary: "Added backlinks",
  incoming_source_ids: ["mem-1"],
  ...over,
});

function renderInfo(over: Partial<React.ComponentProps<typeof PageInfo>> = {}) {
  const onMemoryClick = vi.fn();
  const onPageClick = vi.fn();
  const utils = render(
    <PageInfo
      sourceCount={0}
      sources={[]}
      inbound={[]}
      revisions={[]}
      citations={undefined}
      citationState="none"
      onMemoryClick={onMemoryClick}
      onPageClick={onPageClick}
      {...over}
    />,
  );
  return { onMemoryClick, onPageClick, user: userEvent.setup(), ...utils };
}

describe("PageInfo", () => {
  it("always renders, with zero counts in the summary line", () => {
    renderInfo();
    expect(
      screen.getByText("0 sources · 0 backlinks · 0 revisions"),
    ).toBeInTheDocument();
  });

  it("is collapsed by default and expands on summary click", async () => {
    const { user } = renderInfo({
      sourceCount: 1,
      sources: [source("mem-1")],
    });
    expect(screen.getByText("Title mem-1")).not.toBeVisible();
    await user.click(screen.getByText(/Page info/i));
    expect(screen.getByText("Title mem-1")).toBeVisible();
  });

  it("orders sources by first citation occurrence, then uncited by recency", async () => {
    const { user } = renderInfo({
      sourceCount: 3,
      sources: [
        source("mem-a", { last_modified: 300 }),
        source("mem-b", { last_modified: 100 }),
        source("mem-c", { last_modified: 200 }),
      ],
      citations: [cite(1, 1, "mem-b"), cite(2, 2, "mem-a")],
      citationState: "cited",
    });
    await user.click(screen.getByText(/Page info/i));
    const rows = screen.getAllByTestId("page-info-source-row");
    expect(within(rows[0]).getByText("Title mem-b")).toBeInTheDocument();
    expect(within(rows[1]).getByText("Title mem-a")).toBeInTheDocument();
    expect(within(rows[2]).getByText("Title mem-c")).toBeInTheDocument();
  });

  it("tags sources that carry unverified citations and opens memories on click", async () => {
    const { user, onMemoryClick } = renderInfo({
      sourceCount: 1,
      sources: [source("mem-a")],
      citations: [cite(1, 1, "mem-a", { status: "unverified" })],
      citationState: "cited",
    });
    await user.click(screen.getByText(/Page info/i));
    const row = screen.getByTestId("page-info-source-row");
    expect(within(row).getByText("unverified")).toBeInTheDocument();
    await user.click(row);
    expect(onMemoryClick).toHaveBeenCalledWith("mem-a");
  });

  it("shows backlinks by label without raw page ids", async () => {
    const { user, onPageClick } = renderInfo({
      inbound: [{ source_page_id: "page-uuid-42", label: "Inbound Mention" }],
    });
    await user.click(screen.getByText(/Page info/i));
    await user.click(screen.getByRole("button", { name: "Inbound Mention" }));
    expect(onPageClick).toHaveBeenCalledWith("page-uuid-42");
    expect(screen.queryByText(/page-uuid-42/)).toBeNull();
  });

  it("renders revisions with a citations_summary chip", async () => {
    const { user } = renderInfo({
      revisions: [revision({ citations_summary: "3 verified, 1 unverified" })],
    });
    await user.click(screen.getByText(/Page info/i));
    expect(screen.getByText("v2")).toBeInTheDocument();
    expect(screen.getByText("Added backlinks")).toBeInTheDocument();
    expect(screen.getByText("3 verified, 1 unverified")).toBeInTheDocument();
  });

  it("shows the citation count diagnosability line", async () => {
    const { user } = renderInfo({
      citations: [cite(1, 1, "mem-a"), cite(2, 2, "mem-b", { status: "unverified" })],
      citationState: "cited",
    });
    await user.click(screen.getByText(/Page info/i));
    expect(screen.getByText("Citations: 2 (1 unverified)")).toBeInTheDocument();
  });

  it("explains stripped states", async () => {
    const { user } = renderInfo({ citationState: "stripped-empty" });
    await user.click(screen.getByText(/Page info/i));
    expect(
      screen.getByText("Citations cleared by edit — re-distill to restore"),
    ).toBeInTheDocument();
  });

  it("explains mismatch fallback", async () => {
    const { user } = renderInfo({ citationState: "stripped-mismatch" });
    await user.click(screen.getByText(/Page info/i));
    expect(
      screen.getByText("Citation data mismatched — re-distill to repair"),
    ).toBeInTheDocument();
  });

  it("omits the diagnosability line when there are no citations and no markers", async () => {
    const { user } = renderInfo({ citationState: "none" });
    await user.click(screen.getByText(/Page info/i));
    expect(screen.queryByText(/Citations/)).toBeNull();
  });
});
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pnpm vitest run src/components/memory/page/PageInfo.test.tsx`
Expected: FAIL — `Cannot find module './PageInfo'`.

- [ ] **Step 3: Implement**

Create `src/components/memory/page/PageInfo.tsx`:

```tsx
// SPDX-License-Identifier: AGPL-3.0-only
import type {
  PageChangelogEntry,
  PageCitation,
  PageLinkInbound,
  PageSourceWithMemory,
} from "../../../lib/tauri";
import type { CitationState } from "../../../lib/pageCitations";
import { prettyAgent, relativeMs, sourceKindLabel } from "./format";

interface PageInfoProps {
  sourceCount: number;
  sources: PageSourceWithMemory[] | undefined;
  inbound: PageLinkInbound[];
  revisions: PageChangelogEntry[];
  citations: PageCitation[] | undefined;
  citationState: CitationState;
  onMemoryClick: (sourceId: string) => void;
  onPageClick?: (pageId: string) => void;
}

const groupHeading = {
  fontFamily: "var(--mem-font-mono)",
  fontSize: "10px",
  fontWeight: 600,
  letterSpacing: "0.05em",
  textTransform: "uppercase",
  color: "var(--mem-text-tertiary)",
} as const;

function plural(n: number, word: string): string {
  return `${n} ${word}${n === 1 ? "" : "s"}`;
}

/** Cited rows first (by first occurrence), then uncited by recency. */
function sortSourceRows(
  rows: PageSourceWithMemory[],
  citations: PageCitation[] | undefined,
): PageSourceWithMemory[] {
  const firstOccurrence = new Map<string, number>();
  for (const c of [...(citations ?? [])].sort((a, b) => a.occurrence - b.occurrence)) {
    if (c.source_kind === "memory" && !firstOccurrence.has(c.locator)) {
      firstOccurrence.set(c.locator, c.occurrence);
    }
  }
  return [...rows].sort((a, b) => {
    const ao = firstOccurrence.get(a.source.memory_source_id);
    const bo = firstOccurrence.get(b.source.memory_source_id);
    if (ao != null && bo != null) return ao - bo;
    if (ao != null) return -1;
    if (bo != null) return 1;
    return (b.memory?.last_modified ?? 0) - (a.memory?.last_modified ?? 0);
  });
}

export default function PageInfo({
  sourceCount,
  sources,
  inbound,
  revisions,
  citations,
  citationState,
  onMemoryClick,
  onPageClick,
}: PageInfoProps) {
  const rows = sortSourceRows(
    (sources ?? []).filter((s) => s.memory !== null),
    citations,
  );
  const unverifiedLocators = new Set(
    (citations ?? []).filter((c) => c.status === "unverified").map((c) => c.locator),
  );
  const unverifiedCount = (citations ?? []).filter(
    (c) => c.status === "unverified",
  ).length;
  const diagnosability =
    citationState === "cited"
      ? `Citations: ${(citations ?? []).length} (${unverifiedCount} unverified)`
      : citationState === "stripped-empty"
        ? "Citations cleared by edit — re-distill to restore"
        : citationState === "stripped-mismatch"
          ? "Citation data mismatched — re-distill to repair"
          : null;

  return (
    <details
      aria-label="Page info"
      className="rounded-lg"
      style={{ border: "1px solid var(--mem-border)" }}
    >
      <summary
        className="flex items-center gap-2 px-4 py-3 cursor-pointer select-none list-none"
        style={{
          fontFamily: "var(--mem-font-mono)",
          fontSize: "11px",
          fontWeight: 600,
          letterSpacing: "0.05em",
          textTransform: "uppercase",
          color: "var(--mem-text-tertiary)",
        }}
      >
        <span>Page info</span>
        <span style={{ fontWeight: 400, textTransform: "none", letterSpacing: 0 }}>
          {plural(sourceCount, "source")} · {plural(inbound.length, "backlink")} ·{" "}
          {plural(revisions.length, "revision")}
        </span>
      </summary>
      <div className="flex flex-col gap-4 px-4 pb-4">
        {rows.length > 0 && (
          <div>
            <h4 className="mb-1" style={groupHeading}>
              Sources
            </h4>
            <ul>
              {rows.map((row, idx) => {
                const mem = row.memory!;
                const locator = row.source.memory_source_id;
                return (
                  <li
                    key={locator}
                    data-testid="page-info-source-row"
                    className="py-2 px-2 transition-colors duration-150 hover:bg-[var(--mem-hover)]"
                    style={{
                      borderBottom:
                        idx === rows.length - 1
                          ? "none"
                          : "1px solid color-mix(in srgb, var(--mem-border) 60%, transparent)",
                      cursor: "pointer",
                    }}
                    onClick={() => onMemoryClick(locator)}
                  >
                    <div className="flex items-center gap-2 flex-wrap">
                      <span
                        style={{
                          fontFamily: "var(--mem-font-mono)",
                          fontSize: "10px",
                          color: "var(--mem-text-tertiary)",
                          background: "var(--mem-hover)",
                          padding: "1px 5px",
                          borderRadius: "3px",
                          whiteSpace: "nowrap",
                        }}
                      >
                        {locator}
                      </span>
                      {mem.title && (
                        <span
                          className="truncate"
                          style={{
                            fontFamily: "var(--mem-font-heading)",
                            fontSize: "13px",
                            fontWeight: 500,
                            color: "var(--mem-text)",
                          }}
                        >
                          {mem.title}
                        </span>
                      )}
                    </div>
                    <div className="flex items-center gap-2 mt-0.5">
                      {mem.last_modified && (
                        <span
                          style={{
                            fontFamily: "var(--mem-font-body)",
                            fontSize: "11px",
                            color: "var(--mem-text-tertiary)",
                          }}
                        >
                          {relativeMs(mem.last_modified * 1000)}
                        </span>
                      )}
                      <span
                        style={{
                          fontFamily: "var(--mem-font-body)",
                          fontSize: "11px",
                          color: "var(--mem-text-secondary)",
                        }}
                      >
                        {prettyAgent(mem.source_agent)}
                      </span>
                      <span
                        style={{
                          fontFamily: "var(--mem-font-mono)",
                          fontSize: "10px",
                          color: "var(--mem-text-tertiary)",
                        }}
                      >
                        {sourceKindLabel(mem)}
                      </span>
                      {mem.version != null && mem.version > 1 && (
                        <span
                          style={{
                            fontFamily: "var(--mem-font-mono)",
                            fontSize: "10px",
                            color: "var(--mem-accent-blue, #60a5fa)",
                          }}
                        >
                          v{mem.version}
                        </span>
                      )}
                      {unverifiedLocators.has(locator) && (
                        <span
                          style={{
                            fontFamily: "var(--mem-font-mono)",
                            fontSize: "10px",
                            color: "var(--mem-accent-amber)",
                          }}
                        >
                          unverified
                        </span>
                      )}
                    </div>
                  </li>
                );
              })}
            </ul>
          </div>
        )}
        {inbound.length > 0 && (
          <div>
            <h4 className="mb-1" style={groupHeading}>
              Backlinks
            </h4>
            <div className="flex flex-wrap gap-1.5">
              {inbound.map((link, idx) => (
                <button
                  key={`${link.source_page_id}-${idx}`}
                  onClick={() => onPageClick?.(link.source_page_id)}
                  className="rounded-md px-2.5 py-1.5 transition-colors duration-150 cursor-pointer hover:bg-[var(--mem-hover)]"
                  style={{
                    backgroundColor: "var(--mem-surface)",
                    border: "1px solid var(--mem-border)",
                    fontFamily: "var(--mem-font-body)",
                    fontSize: "12px",
                    fontWeight: 500,
                    color: "var(--mem-text)",
                  }}
                >
                  {link.label}
                </button>
              ))}
            </div>
          </div>
        )}
        {revisions.length > 0 && (
          <div>
            <h4 className="mb-1" style={groupHeading}>
              Revisions
            </h4>
            <div className="flex flex-col gap-1.5">
              {revisions.map((entry) => {
                const incomingCount = entry.incoming_source_ids?.length ?? 0;
                return (
                  <div
                    key={`${entry.version}-${entry.at}`}
                    className="rounded-lg px-3 py-2"
                    style={{
                      backgroundColor: "var(--mem-surface)",
                      border: "1px solid var(--mem-border)",
                    }}
                  >
                    <div className="flex items-center gap-2 mb-0.5 flex-wrap">
                      <span
                        style={{
                          fontFamily: "var(--mem-font-mono)",
                          fontSize: "11px",
                          fontWeight: 600,
                          color: "var(--mem-accent-page)",
                        }}
                      >
                        v{entry.version}
                      </span>
                      <span
                        style={{
                          fontFamily: "var(--mem-font-body)",
                          fontSize: "12px",
                          color: "var(--mem-text-secondary)",
                        }}
                      >
                        {entry.edited_by}
                      </span>
                      <span
                        style={{
                          fontFamily: "var(--mem-font-mono)",
                          fontSize: "10px",
                          color: "var(--mem-text-tertiary)",
                        }}
                      >
                        {relativeMs(entry.at * 1000)}
                      </span>
                      {incomingCount > 0 && (
                        <span
                          style={{
                            fontFamily: "var(--mem-font-mono)",
                            fontSize: "10px",
                            color: "var(--mem-text-tertiary)",
                          }}
                        >
                          {incomingCount} incoming{" "}
                          {incomingCount === 1 ? "memory" : "memories"}
                        </span>
                      )}
                      {entry.citations_summary && (
                        <span
                          style={{
                            fontFamily: "var(--mem-font-mono)",
                            fontSize: "10px",
                            color: "var(--mem-text-tertiary)",
                            background: "var(--mem-hover)",
                            padding: "1px 5px",
                            borderRadius: "3px",
                          }}
                        >
                          {entry.citations_summary}
                        </span>
                      )}
                    </div>
                    {entry.delta_summary && (
                      <p
                        style={{
                          fontFamily: "var(--mem-font-body)",
                          fontSize: "13px",
                          color: "var(--mem-text)",
                          lineHeight: "1.5",
                        }}
                      >
                        {entry.delta_summary}
                      </p>
                    )}
                  </div>
                );
              })}
            </div>
          </div>
        )}
        {diagnosability && (
          <p
            style={{
              fontFamily: "var(--mem-font-mono)",
              fontSize: "10px",
              color: "var(--mem-text-tertiary)",
            }}
          >
            {diagnosability}
          </p>
        )}
      </div>
    </details>
  );
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pnpm vitest run src/components/memory/page/PageInfo.test.tsx && pnpm exec tsc -b`
Expected: all PASS. Note: `toBeVisible()` understands closed `<details>` (jest-dom), which is what the collapse test relies on.

- [ ] **Step 5: Commit**

```bash
git add src/components/memory/page/PageInfo.tsx src/components/memory/page/PageInfo.test.tsx
git commit -m "feat: PageInfo collapsed disclosure (sources, backlinks, revisions, diagnosability)"
```

---

### Task 7: PageDetail structure swap

Delete the four old bottom sections and the global orphan-links query; mount `RelatedPages` + `PageInfo`. Citations wiring comes in Task 8 — this task passes inert citation props (`citations={undefined}`, `citationState="none"`). Update the two existing test files to the new structure in the same task (they'd be red otherwise).

**Files:**
- Modify: `src/components/memory/PageDetail.tsx`
- Modify: `src/components/memory/PageDetail.test.tsx`
- Modify: `src/components/memory/PageDetail.links-revisions.test.tsx`

**Interfaces:**
- Consumes: `RelatedPages` (Task 5), `PageInfo` (Task 6) with the exact props defined there.
- Produces: `PageDetail` renders, after the content block and only when `!editing`: `<RelatedPages …/>` then `<PageInfo …/>`. `aria-label`s: `"Related pages"`, `"Page info"`. The labels `"Page links"`, `"Orphan page links"`, `"Revision history"` and the heading `"Source Memories"` no longer exist anywhere in the component.

- [ ] **Step 1: Update the existing tests to the new structure (failing first)**

In `src/components/memory/PageDetail.links-revisions.test.tsx`:

1. Replace the test `"uses daemon page links for related links and wikilink navigation"` body's section assertions: change `screen.findByLabelText("Page links")` to `screen.findByLabelText("Related pages")`; keep the resolved-button click assertion; keep the Missing-Link inert assertions; the "Inbound Mention" assertion moves to a Page info expansion:

```tsx
  it("uses daemon page links for related pages and wikilink navigation", async () => {
    tauriMocks.getPageLinks.mockResolvedValue({
      outbound: [
        { label: "Resolved Link", target_page_id: "page-2" },
        { label: "Missing Link", target_page_id: null },
      ],
      inbound: [{ source_page_id: "page-3", label: "Inbound Mention" }],
    });

    const { user } = renderWithQuery(<PageDetail {...defaultProps} />);

    expect(await screen.findByText("Link Test Page")).toBeInTheDocument();
    await waitFor(() => {
      expect(tauriMocks.getPageLinks).toHaveBeenCalledWith("page-1");
    });
    expect(tauriMocks.listPages).not.toHaveBeenCalled();

    const contentLink = await screen.findByRole("link", { name: "Resolved Link" });
    await user.click(contentLink);
    expect(defaultProps.onPageClick).toHaveBeenCalledWith("page-2");

    const related = await screen.findByLabelText("Related pages");
    await user.click(within(related).getByRole("button", { name: /Resolved Link/ }));
    expect(defaultProps.onPageClick).toHaveBeenCalledWith("page-2");
    expect(within(related).getByText("Missing Link")).toBeInTheDocument();
    expect(within(related).queryByRole("button", { name: /Missing Link/ })).toBeNull();

    await user.click(screen.getByText(/Page info/i));
    expect(screen.getByRole("button", { name: "Inbound Mention" })).toBeInTheDocument();
  });
```

2. Replace `"shows source identity for duplicate inbound link labels"` — duplicate labels now render as two identical backlink buttons and raw ids are gone:

```tsx
  it("renders duplicate inbound labels without raw page ids", async () => {
    tauriMocks.getPageLinks.mockResolvedValue({
      outbound: [],
      inbound: [
        { source_page_id: "source-page-a", label: "Shared Mention" },
        { source_page_id: "source-page-b", label: "Shared Mention" },
      ],
    });

    const { user } = renderWithQuery(<PageDetail {...defaultProps} />);
    expect(await screen.findByText("Link Test Page")).toBeInTheDocument();
    await user.click(screen.getByText(/Page info/i));
    expect(screen.getAllByRole("button", { name: "Shared Mention" })).toHaveLength(2);
    expect(screen.queryByText(/source-page-a/)).toBeNull();
  });
```

3. In `"keeps the page visible and hides links when the daemon route fails"`, change the final assertion to `expect(screen.queryByLabelText("Related pages")).toBeNull();` and add `expect(screen.getByText(/Page info/i)).toBeInTheDocument();` (Page info always renders).

4. Replace `"renders daemon orphan link diagnostics"` with the inversion:

```tsx
  it("does not query orphan links and renders no Unlinked Mentions section", async () => {
    renderWithQuery(<PageDetail {...defaultProps} />);
    expect(await screen.findByText("Link Test Page")).toBeInTheDocument();
    expect(tauriMocks.listOrphanLinks).not.toHaveBeenCalled();
    expect(screen.queryByText("Unlinked Mentions")).toBeNull();
  });
```

5. In `"renders daemon page revision history"`, expand the disclosure before asserting:

```tsx
    const { user } = renderWithQuery(<PageDetail {...defaultProps} />);
    expect(await screen.findByText("Link Test Page")).toBeInTheDocument();
    await user.click(screen.getByText(/Page info/i));
    expect(screen.getByText(/added backlinks/i)).toBeInTheDocument();
    expect(screen.getByText("just now")).toBeInTheDocument();
```

6. In `"keeps rendering the page when page revisions route is unavailable"`, replace the final assertion with a summary-count assertion: `expect(screen.getByText(/0 revisions/)).toBeInTheDocument();`

In `src/components/memory/PageDetail.test.tsx`, translate these specific tests (names as they exist today; read the file for their current bodies):
- `"renders source memories section with count"` (~line 220): assert the Page info summary line instead — `expect(screen.getByText(/2 sources/)).toBeInTheDocument()` (match the fixture's count).
- `"shows Page Links from daemon page links without listPages inference"` (~234): retarget `Page links` → `Related pages`; outbound assertions move to the Related pages section, inbound assertions expand Page info first.
- `"hides Page Links when the daemon returns no links"` (~248) and `"hides Page Links section when no daemon links in content"` (~256): assert `screen.queryByLabelText("Related pages")` is null AND `screen.getByText(/Page info/i)` is present (Page info always renders).
- `"shows loading placeholders while source memories are fetching"` (~277): delete it — the collapsed disclosure intentionally drops the placeholder skeletons (behavior removed, not overlooked).
- `"renders one evidence card per source memory after fetch"` (~291): expand Page info, then `expect(screen.getAllByTestId("page-info-source-row")).toHaveLength(n)`.
- `"clicking an evidence card calls onMemoryClick with the right source_id"` (~299): expand Page info, click a `page-info-source-row`, assert `onMemoryClick`.
- `"uses getPageSources (join table) not listMemoriesByIds"` (~307): keep, assertions unchanged (query wiring is untouched).
- Keep every toolbar/redistill/edit/meta/back-button/CSS-variable test untouched.

- [ ] **Step 2: Run tests to verify they fail**

Run: `pnpm vitest run src/components/memory/PageDetail.test.tsx src/components/memory/PageDetail.links-revisions.test.tsx`
Expected: FAIL — the component still renders the old sections.

- [ ] **Step 3: Implement the swap in `PageDetail.tsx`**

All references are to current line numbers in `src/components/memory/PageDetail.tsx`:

1. Imports: remove `listOrphanLinks` from the `../../lib/tauri` import (line 8); remove `type MemoryItem` if now unused; add:

```tsx
import RelatedPages from "./page/RelatedPages";
import PageInfo from "./page/PageInfo";
```

2. Delete the local helper block now owned by `page/format.ts`: `KNOWN_AGENTS` + `prettyAgent` (lines 58-73), `SOURCE_KIND_LABEL` + `sourceKindLabel` (lines 75-86), `relativeMs` (lines 88-94). Keep `relativeTimeFromISO` (used by the header).
3. Delete the orphan-links query (lines 130-135).
4. Delete the `sourceMemories` extraction/sort block (lines 168-178) — `PageInfo` owns filtering and ordering now.
5. Delete `hasPageLinks` (line 359) and `orphanLinkLabels` (line 361). Keep `outboundLinks`, `inboundLinks`, `pageRevisionEntries`.
6. Delete the four JSX sections: Page Links (lines 631-737), Unlinked Mentions (739-785), Revision History (787-870), Source Memories (872-914), and the whole `EvidenceCard` component (919-1037).
7. In their place (directly after the content `{editing ? … : …}` block, inside the root flex column), add:

```tsx
      {!editing && <RelatedPages outbound={outboundLinks} onPageClick={onPageClick} />}

      {!editing && (
        <PageInfo
          sourceCount={sourceCount}
          sources={pageSources}
          inbound={inboundLinks}
          revisions={pageRevisionEntries}
          citations={undefined}
          citationState="none"
          onMemoryClick={onMemoryClick}
          onPageClick={onPageClick}
        />
      )}
```

`sourceCount` already exists (line 316: `pageSources?.length ?? page.source_memory_ids.length`).

- [ ] **Step 4: Run tests to verify they pass**

Run: `pnpm vitest run src/components/memory/ && pnpm exec tsc -b`
Expected: all PASS, including `__tests__/PageDetail.export.test.tsx` (toolbar untouched). If tsc flags unused imports in PageDetail, remove them.

- [ ] **Step 5: Commit**

```bash
git add src/components/memory/PageDetail.tsx src/components/memory/PageDetail.test.tsx src/components/memory/PageDetail.links-revisions.test.tsx
git commit -m "feat: replace page-detail bottom sections with RelatedPages and PageInfo"
```

---

### Task 8: Citations wiring in PageDetail

Connect the pipeline: process markers, render chips through `ContentRenderer`, strip markers from the TLDR, feed real citation state to `PageInfo`.

**Files:**
- Modify: `src/components/memory/PageDetail.tsx`
- Test (create): `src/components/memory/PageDetail.citations.test.tsx`

**Interfaces:**
- Consumes: `processCitations`, `stripCitationLinks` (Task 2); `ContentRenderer`'s `renderCitation` prop (Task 3); `CitationChip` (Task 4); `PageInfo` props (Task 6).
- Produces: the user-visible feature. Pipeline order is load-bearing: `processCitations(page.content, page.citations)` runs on the RAW content (backend counted occurrences over the raw stored body), THEN the existing title-heading strip, `## Sources` strip, and wikilink rewrite run on `processed.content`.

- [ ] **Step 1: Write the failing integration tests**

Create `src/components/memory/PageDetail.citations.test.tsx`:

```tsx
// SPDX-License-Identifier: AGPL-3.0-only
import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import type { PageCitation } from "../../lib/tauri";
import PageDetail from "./PageDetail";

const tauriMocks = vi.hoisted(() => ({
  getPage: vi.fn(),
  getPageSources: vi.fn(),
  listRegisteredSources: vi.fn(),
  getPageLinks: vi.fn(),
  getPageRevisions: vi.fn(),
  redistillPage: vi.fn(),
  updatePage: vi.fn(),
  deletePage: vi.fn(),
  clipboardWrite: vi.fn(),
  exportPageToObsidian: vi.fn(),
}));

vi.mock("../../lib/tauri", () => ({
  ...tauriMocks,
  FACET_COLORS: {},
  STABILITY_TIERS: {},
}));

const cite = (
  occurrence: number,
  marker: number,
  over: Partial<PageCitation> = {},
): PageCitation => ({
  occurrence,
  marker,
  source_kind: "memory",
  locator: `mem-${marker}`,
  score: 0.9,
  status: "verified",
  scope: "sentence",
  ...over,
});

const BASE_PAGE = {
  id: "page-1",
  title: "Cited Page",
  summary: null,
  content:
    "# Cited Page\n\nIntro sentence stands alone. The daemon is local-first.[1] It uses libSQL.[2]",
  entity_id: null,
  domain: "testing",
  source_memory_ids: ["mem-1", "mem-2"],
  version: 1,
  status: "active",
  created_at: "2026-06-26T00:00:00+00:00",
  last_compiled: "2026-06-26T00:00:00+00:00",
  last_modified: "2026-06-26T00:00:00+00:00",
  citations: [cite(1, 1), cite(2, 2, { status: "unverified" })],
};

const SOURCES = [
  {
    source: { page_id: "page-1", memory_source_id: "mem-1", linked_at: 0 },
    memory: {
      source_id: "mem-1",
      title: "Local-first decision",
      content: "We keep the daemon local-first.",
      summary: null,
      memory_type: "memory",
      domain: null,
      source_agent: "claude-code",
      confidence: null,
      confirmed: true,
      pinned: false,
      supersedes: null,
      last_modified: 1_700_000_000,
      chunk_count: 1,
    },
  },
];

function renderPage() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  const props = {
    pageId: "page-1",
    onBack: vi.fn(),
    onMemoryClick: vi.fn(),
    onPageClick: vi.fn(),
  };
  render(
    <QueryClientProvider client={client}>
      <PageDetail {...props} />
    </QueryClientProvider>,
  );
  return { props, user: userEvent.setup() };
}

beforeEach(() => {
  vi.clearAllMocks();
  tauriMocks.getPage.mockResolvedValue(BASE_PAGE);
  tauriMocks.getPageSources.mockResolvedValue(SOURCES);
  tauriMocks.listRegisteredSources.mockResolvedValue([]);
  tauriMocks.getPageLinks.mockResolvedValue({ outbound: [], inbound: [] });
  tauriMocks.getPageRevisions.mockResolvedValue({
    page_id: "page-1",
    current_version: 1,
    user_edited: false,
    stale_reason: null,
    entries: [],
  });
  tauriMocks.redistillPage.mockResolvedValue({ status: "ok", updated: true });
});

describe("PageDetail citations", () => {
  it("renders one chip per citation and no raw markers in the body", async () => {
    renderPage();
    expect(await screen.findByText("Cited Page")).toBeInTheDocument();
    const chip1 = await screen.findByRole("button", { name: /mem-1/ });
    const chip2 = screen.getByRole("button", { name: /mem-2/ });
    expect(chip1).toHaveAttribute("data-status", "verified");
    expect(chip2).toHaveAttribute("data-status", "unverified");
    expect(screen.queryByText(/\[1\]/)).toBeNull();
  });

  it("resolves the popover from page-sources and opens the memory", async () => {
    const { props, user } = renderPage();
    const chip = await screen.findByRole("button", { name: /mem-1/ });
    fireEvent.focus(chip);
    expect(await screen.findByText("Local-first decision")).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: /Open memory/ }));
    expect(props.onMemoryClick).toHaveBeenCalledWith("mem-1");
  });

  it("shows 'source not available' for a locator missing from page-sources", async () => {
    renderPage();
    const chip = await screen.findByRole("button", { name: /mem-2/ });
    fireEvent.focus(chip);
    expect(await screen.findByText(/source not available/i)).toBeInTheDocument();
  });

  it("display-strips markers when citations were cleared by an edit", async () => {
    tauriMocks.getPage.mockResolvedValue({ ...BASE_PAGE, citations: undefined });
    const { user } = renderPage();
    expect(await screen.findByText("Cited Page")).toBeInTheDocument();
    expect(screen.getByText(/It uses libSQL\./)).toBeInTheDocument();
    expect(screen.queryByText(/\[2\]/)).toBeNull();
    expect(screen.queryByRole("button", { name: /mem-1/ })).toBeNull();
    await user.click(screen.getByText(/Page info/i));
    expect(
      screen.getByText("Citations cleared by edit — re-distill to restore"),
    ).toBeInTheDocument();
  });

  it("falls back to strip-all on count mismatch and reports it", async () => {
    tauriMocks.getPage.mockResolvedValue({ ...BASE_PAGE, citations: [cite(1, 1)] });
    const { user } = renderPage();
    expect(await screen.findByText("Cited Page")).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /mem-1/ })).toBeNull();
    expect(screen.queryByText(/\[1\]/)).toBeNull();
    await user.click(screen.getByText(/Page info/i));
    expect(
      screen.getByText("Citation data mismatched — re-distill to repair"),
    ).toBeInTheDocument();
  });

  it("keeps the TLDR pull-quote free of markers and citation links", async () => {
    // First `.\s` sentence boundary lands AFTER marker [1], so the extracted
    // pull-quote contains a rewritten citation link that must be stripped.
    tauriMocks.getPage.mockResolvedValue({
      ...BASE_PAGE,
      content:
        "# Cited Page\n\nThe daemon is local-first.[1] It stays fast under load. Second paragraph here.[2]",
    });
    renderPage();
    expect(await screen.findByText("Cited Page")).toBeInTheDocument();
    const quote = screen.getByText(/It stays fast under load\./);
    expect(quote.textContent).not.toMatch(/\[\d+\]/);
    expect(quote.textContent).not.toContain("#citation");
    // Known ceiling (spec §4.3): the first-sentence citation gets no inline
    // chip; the second citation still renders in the body.
    expect(screen.queryByRole("button", { name: /mem-1/ })).toBeNull();
    expect(screen.getByRole("button", { name: /mem-2/ })).toBeInTheDocument();
  });
});
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pnpm vitest run src/components/memory/PageDetail.citations.test.tsx`
Expected: FAIL — raw `[1]` markers render, no chips exist.

- [ ] **Step 3: Implement the wiring in `PageDetail.tsx`**

1. Add imports:

```tsx
import { processCitations, stripCitationLinks } from "../../lib/pageCitations";
import CitationChip from "./page/CitationChip";
```

2. Replace the `cleanedContent` computation (post-Task-7 location of old lines 318-329) so marker processing runs FIRST, on raw content:

```tsx
  // Citations first (occurrence counting mirrors the backend and runs over the
  // raw stored body), then the existing display transforms.
  const processed = processCitations(page.content, page.citations);

  // Strip ## Sources (shown in Page info below)
  // Convert [[wikilinks]] to markdown links if they resolve to pages, else plain text
  const cleanedContent = processed.content
    .replace(/^#\s+.*\n+/, "") // Strip title heading (displayed separately by UI)
    .replace(/## Sources\n[\s\S]*?(?=\n## |\s*$)/, "")
    .replace(/\[\[([^\]]+)\]\]/g, (_match, inner) => {
      const link = parseWikilink(inner);
      const cid = outboundTargetByLabel.get(normalizeLinkLabel(link.targetLabel));
      if (cid) return `[${link.displayText}](${PAGE_LINK_ANCHOR_PREFIX}${cid})`;
      return link.displayText;
    })
    .trim();
```

3. TLDR: keep the extraction untouched, strip citation links from the displayed string only (known ceiling per spec: a first-sentence citation gets no inline chip):

```tsx
  const sentenceEnd = cleanedContent.search(/\.\s/);
  const tldr = sentenceEnd > 0 && sentenceEnd < 400
    ? stripCitationLinks(cleanedContent.slice(0, sentenceEnd + 1).trim())
    : "";
  const displayContent = tldr
    ? cleanedContent.slice(sentenceEnd + 1).trim()
    : cleanedContent;
```

Note `sentenceEnd` still indexes `cleanedContent`, so `displayContent` slicing is unchanged.

4. Build the locator→memory map (after the `pageSources` query, before the early returns is NOT needed — plain computation next to `sourceCount`):

```tsx
  const sourceMemoryByLocator = new Map(
    (pageSources ?? [])
      .filter((cs) => cs.memory !== null)
      .map((cs) => [cs.source.memory_source_id, cs.memory!]),
  );
```

5. Pass the chip renderer to the content renderer (the `<ContentRenderer content={displayContent} variant="detail" />` call):

```tsx
          <ContentRenderer
            content={displayContent}
            variant="detail"
            renderCitation={(k) => {
              const c = processed.byOccurrence.get(k);
              if (!c) return null;
              return (
                <CitationChip
                  occurrence={k}
                  citation={c}
                  sourceMemory={sourceMemoryByLocator.get(c.locator) ?? null}
                  sourcesLoading={pageSources === undefined}
                  onOpenMemory={onMemoryClick}
                />
              );
            }}
          />
```

6. Feed `PageInfo` the real state — replace `citations={undefined}` / `citationState="none"` from Task 7 with:

```tsx
          citations={page.citations}
          citationState={processed.state}
```

7. The redistill mutation's `onSuccess` already invalidates `["page", pageId]` and `["page-sources", pageId]`, so chips refresh after re-distilling — no change needed. The update (edit) mutation invalidates `["page", pageId]`; the daemon clears citations on edit, so the stripped-empty path exercises automatically — no change needed.

- [ ] **Step 4: Run the full frontend suite**

Run: `pnpm test && pnpm exec tsc -b`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/components/memory/PageDetail.tsx src/components/memory/PageDetail.citations.test.tsx
git commit -m "feat: inline per-claim citation chips on page detail"
```

---

### Task 9: Full verification gate

No new code — run every CI gate locally and fix anything that surfaces. `ci.yml` is strict (`-D warnings`), not advisory.

- [ ] **Step 1: Rust gates**

Run from the repo/worktree root:

```bash
cargo fmt --check --all
cargo clippy --workspace --all-targets -- -D warnings
cd app && cargo test && cd ..
```

Expected: all clean/PASS.

- [ ] **Step 2: Frontend gates**

```bash
pnpm exec tsc -b
pnpm test
```

Expected: clean, all suites PASS (including the untouched `__tests__/PageDetail.export.test.tsx` and `src/lib/tauri.test.ts`).

- [ ] **Step 3: Coverage sanity for the gated module**

```bash
pnpm test:coverage
```

Expected: `src/lib/tauri.ts` still meets 90/90/85/90 (the additions were type-only).

- [ ] **Step 4: Commit any fixes**

```bash
git add -A && git commit -m "chore: satisfy fmt/clippy/tsc gates for citations redesign"
```

(Skip the commit if nothing changed.)

**Manual QA (needs a human/dev environment, not blocking the PR):** `git -C ../wenlan pull` (PR #332 is on the backend's `origin/main`; the local sibling checkout must be at ≥ commit `2f6ee4bd`), then `pnpm dev:all`, re-distill a page, verify chips + popovers + Page info states. No e2e harness exists in this repo — known limitation. **Release gate reminder:** the app cannot ship this feature until the backend tags a release containing PR #332 (`release.yml` resolves the backend at its latest GitHub release tag); until then the app degrades gracefully (no chips).
