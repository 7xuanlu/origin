// SPDX-License-Identifier: AGPL-3.0-only
import { beforeEach, describe, expect, it } from "vitest";
import { act, renderHook } from "@testing-library/react";
import { reviewSuppressKey, useSuppressedReviewItems } from "./reviewSuppression";
import { EXAMPLE_REVIEW_ITEMS, exampleReviewLabel } from "./reviewExamples";
import type { ReviewItem } from "./useReviewQueue";

const STORAGE_KEY = "wenlan.review.hidden.v1";

function topicItem(label: string): ReviewItem {
  return { kind: "topic", id: label, label, count: 1, timestampMs: null };
}

function candidateItem(title: string): ReviewItem {
  return {
    kind: "page_candidate",
    id: `cluster-${title}`,
    title,
    cluster: { source_ids: [], contents: [], estimated_tokens: 0 },
    timestampMs: null,
  };
}

function staleItem(id: string, title: string): ReviewItem {
  return { kind: "stale_page", id, title, summary: null, sourcesUpdated: null, timestampMs: null };
}

function revisionItem(id: string): ReviewItem {
  return {
    kind: "revision",
    id,
    targetSourceId: "t1",
    revisionSourceId: "rs1",
    content: "content",
    agent: null,
    timestampMs: null,
  };
}

function captureItem(id: string): ReviewItem {
  return { kind: "capture", id, title: "t", snippet: null, timestampMs: null };
}

function refinementItem(id: string): ReviewItem {
  return {
    kind: "refinement",
    id,
    action: "detect_contradiction",
    sourceIds: [],
    payload: { action: "detect_contradiction" },
    confidence: 1,
    timestampMs: null,
  };
}

describe("reviewSuppressKey", () => {
  it("keys topic items by their label", () => {
    expect(reviewSuppressKey(topicItem("Preview harness"))).toBe("topic:Preview harness");
  });

  it("keys page_candidate items by normalized title", () => {
    expect(reviewSuppressKey(candidateItem("  Threads Retro Notes  "))).toBe(
      "candidate:threads retro notes",
    );
  });

  it("keys stale_page items by page id", () => {
    expect(reviewSuppressKey(staleItem("page-1", "Any title"))).toBe("stale:page-1");
  });

  it("returns null for kinds that already persist dismissal through the daemon", () => {
    expect(reviewSuppressKey(revisionItem("r1"))).toBeNull();
    expect(reviewSuppressKey(captureItem("c1"))).toBeNull();
    expect(reviewSuppressKey(refinementItem("f1"))).toBeNull();
  });

  it("keys example items (revision and refinement kinds) by their own id", () => {
    for (const item of EXAMPLE_REVIEW_ITEMS) {
      expect(reviewSuppressKey(item)).toBe(item.id);
    }
  });
});

describe("useSuppressedReviewItems", () => {
  beforeEach(() => {
    localStorage.clear();
  });

  it("hides an item and persists it to localStorage", () => {
    const { result } = renderHook(() => useSuppressedReviewItems());

    act(() => result.current.hide(topicItem("Preview harness")));

    expect(result.current.hiddenKeys.has("topic:Preview harness")).toBe(true);
    const stored = JSON.parse(localStorage.getItem(STORAGE_KEY) ?? "[]");
    expect(stored).toHaveLength(1);
    expect(stored[0].key).toBe("topic:Preview harness");
  });

  it("evicts the oldest entry once the store exceeds 200", () => {
    const { result } = renderHook(() => useSuppressedReviewItems());

    // One hide per act() — mirrors real usage (one click, then a render)
    // rather than batching 201 state updates off a single stale snapshot.
    for (let i = 0; i < 201; i++) {
      act(() => result.current.hide(topicItem(`topic-${i}`)));
    }

    expect(result.current.hiddenEntries).toHaveLength(200);
    expect(result.current.hiddenKeys.has("topic:topic-0")).toBe(false);
    expect(result.current.hiddenKeys.has("topic:topic-200")).toBe(true);

    const stored = JSON.parse(localStorage.getItem(STORAGE_KEY) ?? "[]");
    expect(stored).toHaveLength(200);
  });

  it("restore removes a key from the hidden set", () => {
    const { result } = renderHook(() => useSuppressedReviewItems());

    act(() => result.current.hide(topicItem("Preview harness")));
    expect(result.current.hiddenKeys.has("topic:Preview harness")).toBe(true);

    act(() => result.current.restore("topic:Preview harness"));

    expect(result.current.hiddenKeys.has("topic:Preview harness")).toBe(false);
    const stored = JSON.parse(localStorage.getItem(STORAGE_KEY) ?? "[]");
    expect(stored).toHaveLength(0);
  });

  it("hides and restores an example item, keyed and labeled by reviewExamples.ts", () => {
    const [coffee] = EXAMPLE_REVIEW_ITEMS;
    const { result } = renderHook(() => useSuppressedReviewItems());

    act(() => result.current.hide(coffee));

    expect(result.current.hiddenKeys.has(coffee.id)).toBe(true);
    const stored = JSON.parse(localStorage.getItem(STORAGE_KEY) ?? "[]");
    expect(stored).toHaveLength(1);
    expect(stored[0]).toMatchObject({
      key: coffee.id,
      label: exampleReviewLabel(coffee),
      kind: coffee.kind,
    });

    act(() => result.current.restore(coffee.id));

    expect(result.current.hiddenKeys.has(coffee.id)).toBe(false);
  });
});
