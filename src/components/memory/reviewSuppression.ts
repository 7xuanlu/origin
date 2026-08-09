// SPDX-License-Identifier: AGPL-3.0-only
import { useMemo, useState } from "react";
import { exampleReviewLabel, isExampleReviewItem } from "./reviewExamples";
import type { ReviewItem } from "./useReviewQueue";

const STORAGE_KEY = "wenlan.review.hidden.v1";
const MAX_HIDDEN_ENTRIES = 200;

export interface HiddenReviewEntry {
  key: string;
  label: string;
  kind: string;
  at: number;
}

// Dev-only sample items (see reviewExamples.ts) key on their own id — real
// revision/refinement/capture kinds persist dismissal through daemon calls
// instead (see useReviewQueue.ts), so only topic/page_candidate/stale_page
// fall through to the switch below.
function suppressionKey(item: ReviewItem): { key: string; label: string } | null {
  if (isExampleReviewItem(item)) {
    return { key: item.id, label: exampleReviewLabel(item) };
  }
  switch (item.kind) {
    case "topic":
      // The topic's id IS its label already — content-stable across distill runs.
      return { key: `topic:${item.label}`, label: item.label };
    case "page_candidate":
      // Cluster ids are `source_ids.join("-")` and regenerate when the
      // cluster gains a member, but the user recognizes "the same item" by
      // title, so keying on the normalized title keeps a re-proposal hidden.
      //
      // Caveat: two candidates can share the fallback title "Untitled
      // cluster" — hiding one hides both. Accepted: rare, and Restore
      // recovers it. A candidate whose cluster later grows but keeps its
      // title also stays hidden — that's the requested behavior ("prevent
      // them showing again"), not a bug; Restore is the escape, not
      // auto-resurface.
      return {
        key: `candidate:${item.title.trim().toLowerCase()}`,
        label: item.title,
      };
    case "stale_page":
      // Page ids are stable — "stop nagging me about this page" is per-page.
      return { key: `stale:${item.id}`, label: item.title };
    default:
      return null;
  }
}

/** null means the item's kind already persists dismissal through the daemon. */
export function reviewSuppressKey(item: ReviewItem): string | null {
  return suppressionKey(item)?.key ?? null;
}

function readHiddenEntries(): HiddenReviewEntry[] {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    // Corrupt or inaccessible storage degrades to "nothing hidden" — the
    // next write overwrites it, so bad state never wedges the panel.
    return [];
  }
}

function writeHiddenEntries(entries: HiddenReviewEntry[]): void {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(entries));
  } catch {
    // Hiding is a convenience, not a critical path — disabled/full storage
    // should not break the review panel.
  }
}

/** Local (per-device) "Hide" for the three read-only review kinds that have
 * no daemon dismiss verb. State-backed so hides re-render immediately; every
 * mutation writes through to localStorage. Cross-window sync is a non-goal —
 * only one panel instance runs in practice. */
export function useSuppressedReviewItems(): {
  hiddenKeys: Set<string>;
  hiddenEntries: HiddenReviewEntry[];
  hide: (item: ReviewItem) => void;
  restore: (key: string) => void;
  restoreAll: () => void;
} {
  const [entries, setEntries] = useState<HiddenReviewEntry[]>(() => readHiddenEntries());

  const persist = (next: HiddenReviewEntry[]) => {
    setEntries(next);
    writeHiddenEntries(next);
  };

  const hide = (item: ReviewItem) => {
    const found = suppressionKey(item);
    if (found === null) return;
    if (entries.some((entry) => entry.key === found.key)) return;
    const next = [
      ...entries,
      { key: found.key, label: found.label, kind: item.kind, at: Date.now() },
    ];
    // Cap at MAX_HIDDEN_ENTRIES, evicting the oldest by `at`.
    persist(
      next.length <= MAX_HIDDEN_ENTRIES
        ? next
        : next.sort((a, b) => a.at - b.at).slice(next.length - MAX_HIDDEN_ENTRIES),
    );
  };

  const restore = (key: string) => {
    persist(entries.filter((entry) => entry.key !== key));
  };

  const restoreAll = () => {
    persist([]);
  };

  const hiddenKeys = useMemo(() => new Set(entries.map((entry) => entry.key)), [entries]);

  return { hiddenKeys, hiddenEntries: entries, hide, restore, restoreAll };
}
