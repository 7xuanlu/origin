// SPDX-License-Identifier: AGPL-3.0-only
// Dev-only sample review items shown when the real queue is empty, so a fresh
// install's Review page isn't a blank slate. Inlined out of production builds
// via REVIEW_EXAMPLES_ENABLED (import.meta.env.DEV is compile-time false in
// `vite build`). MODE !== "test" keeps Vitest (which sets DEV=true,
// MODE="test") from injecting samples into existing panel tests.
import type { QueryClient } from "@tanstack/react-query";
import type { ReviewItem } from "./useReviewQueue";
import type { MemoryItem } from "../../lib/tauri";

export const REVIEW_EXAMPLES_ENABLED =
  import.meta.env.DEV && import.meta.env.MODE !== "test";

const EXAMPLE_PREFIX = "example:";

export function isExampleReviewItem(item: ReviewItem): boolean {
  return item.id.startsWith(EXAMPLE_PREFIX);
}

// ---- sample content (intentionally hard-coded English: fixture data, not UI copy) ----

const COFFEE_ID = "example:memory-coffee";
const COFFEE_TITLE = "Coffee routine";
const COFFEE_CURRENT =
  "Prefers pour-over coffee in the morning and switches to black tea after 2pm to protect sleep.";
const COFFEE_REVISED =
  "Prefers pour-over coffee in the morning — usually a single-origin Ethiopian — and switches to herbal tea after 2pm to protect sleep. No caffeine at all during on-call weeks.";

const STANDUP_NEW_ID = "example:memory-standup-new";
const STANDUP_NEW_TITLE = "Standup schedule (updated)";
const STANDUP_NEW_CONTENT =
  "Standup moved to 9:30 on Tuesdays and Thursdays only, to keep mornings free for deep work.";
const STANDUP_OLD_ID = "example:memory-standup-old";
const STANDUP_OLD_TITLE = "Daily standup";
const STANDUP_OLD_CONTENT = "Daily standup is at 10:00 every weekday on Zoom.";

const HOURS = 60 * 60 * 1000;

export const EXAMPLE_REVIEW_ITEMS: ReviewItem[] = [
  {
    kind: "revision",
    id: COFFEE_ID, // revision items key on target_source_id
    targetSourceId: COFFEE_ID,
    revisionSourceId: "example:memory-coffee-rev1",
    content: COFFEE_REVISED,
    agent: "claude-desktop",
    timestampMs: Date.now() - 2 * HOURS,
  },
  {
    kind: "refinement",
    id: "example:refinement-standup",
    action: "detect_contradiction",
    sourceIds: [STANDUP_NEW_ID, STANDUP_OLD_ID], // [0]=new, [1]=existing (daemon order)
    payload: { action: "detect_contradiction" },
    confidence: 0.82,
    timestampMs: Date.now() - 5 * HOURS,
  },
];

/** Human label for the hidden-items footer (fixture data — English like the content). */
export function exampleReviewLabel(item: ReviewItem): string {
  return item.kind === "revision" ? COFFEE_TITLE : STANDUP_NEW_TITLE;
}

function exampleMemory(sourceId: string, title: string, content: string): MemoryItem {
  return {
    source_id: sourceId,
    title,
    content,
    summary: content,
    memory_type: "preference",
    domain: null,
    source_agent: "claude-desktop",
    confidence: 0.9,
    confirmed: true,
    pinned: false,
    supersedes: null,
    last_modified: Math.floor(Date.now() / 1000) - 86_400,
    chunk_count: 1,
  };
}

/** Seed every cache the dialog reads for example ids, so no query ever
 * fetches (staleTime Infinity + pre-set data = mounted queries stay idle). */
export function seedReviewExampleCaches(queryClient: QueryClient): void {
  const seed = (key: unknown[], value: unknown) => {
    queryClient.setQueryDefaults(key, { staleTime: Infinity, gcTime: Infinity });
    queryClient.setQueryData(key, value);
  };
  const coffee = exampleMemory(COFFEE_ID, COFFEE_TITLE, COFFEE_CURRENT);
  const standupNew = exampleMemory(STANDUP_NEW_ID, STANDUP_NEW_TITLE, STANDUP_NEW_CONTENT);
  const standupOld = exampleMemory(STANDUP_OLD_ID, STANDUP_OLD_TITLE, STANDUP_OLD_CONTENT);

  seed(["memory-detail", COFFEE_ID], coffee);
  seed(["memory-detail", STANDUP_NEW_ID], standupNew);
  seed(["memory-detail", STANDUP_OLD_ID], standupOld);

  // Card/dialog summary names + word-delta base text (useReviewItemSummary).
  seed(["review-summary", "memory", COFFEE_ID], { name: COFFEE_TITLE, text: COFFEE_CURRENT });
  seed(["review-summary", "memory", STANDUP_NEW_ID], { name: STANDUP_NEW_TITLE, text: STANDUP_NEW_CONTENT });
  seed(["review-summary", "memory", STANDUP_OLD_ID], { name: STANDUP_OLD_TITLE, text: STANDUP_OLD_CONTENT });

  // MemoryRevisionChain: empty chain renders null without an invoke
  // (ReviewHistory.tsx — chainDepth <= 1 && entries.length === 0). Only the
  // revision dialog mounts it today, but seed all three ids defensively.
  const emptyChain = (sourceId: string) => ({
    current_source_id: sourceId,
    chain_depth: 0,
    entries: [],
  });
  seed(["memory-revisions", COFFEE_ID], emptyChain(COFFEE_ID));
  seed(["memory-revisions", STANDUP_NEW_ID], emptyChain(STANDUP_NEW_ID));
  seed(["memory-revisions", STANDUP_OLD_ID], emptyChain(STANDUP_OLD_ID));
}
