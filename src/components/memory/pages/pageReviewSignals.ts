// SPDX-License-Identifier: AGPL-3.0-only
import type { QueryClient } from "@tanstack/react-query";
import type {
  DistillPendingCluster,
  DistillReviewResponse,
  ListRefinementsResponse,
} from "../../../lib/tauri";
import type { ReviewItem } from "../useReviewQueue";

/** Session-only bridge from the user's explicit Review run to the Wiki.
 * Reading this query never invokes distill_review. */
export const DISTILL_REVIEW_SESSION_QUERY_KEY = [
  "distill-review",
  "session",
] as const;

/** Review discoveries last for the lifetime of this app session. Both the
 * producer and consumer use this policy so a delayed trip from Review to Wiki
 * cannot fall back to TanStack Query's five-minute garbage-collection window. */
export const DISTILL_REVIEW_SESSION_QUERY_POLICY = {
  gcTime: Infinity,
  staleTime: Infinity,
} as const;

export function cacheDistillReviewSession(
  queryClient: QueryClient,
  result: DistillReviewResponse,
): void {
  queryClient.setQueryDefaults(
    DISTILL_REVIEW_SESSION_QUERY_KEY,
    DISTILL_REVIEW_SESSION_QUERY_POLICY,
  );
  queryClient.setQueryData(DISTILL_REVIEW_SESSION_QUERY_KEY, result);
}

function firstNonEmpty(values: Array<string | null | undefined>): string | null {
  for (const value of values) {
    const trimmed = value?.trim();
    if (trimmed) return trimmed;
  }
  return null;
}

function truncate(value: string, max: number): string {
  if (value.length <= max) return value;
  return `${value.slice(0, max - 3).trimEnd()}...`;
}

export function pageCandidateTitle(cluster: DistillPendingCluster): string | null {
  const content = firstNonEmpty(cluster.contents);
  const contentTitle = content
    ? truncate(content.replace(/^#+\s*/, "").replace(/\s+/g, " "), 72)
    : null;
  return firstNonEmpty([
    cluster.existing_page_title,
    cluster.entity_name,
    cluster.space,
    contentTitle,
  ]);
}

export function pageCandidateItems(
  result: DistillReviewResponse | undefined,
  untitledLabel: string,
): ReviewItem[] {
  return (result?.pending ?? []).map(
    (cluster, index): ReviewItem => ({
      kind: "page_candidate",
      id: cluster.source_ids.join("-") || `cluster-${index}`,
      title: pageCandidateTitle(cluster) ?? untitledLabel,
      cluster,
      timestampMs: null,
    }),
  );
}

export function pageCleanupSuggestionIds(
  result: ListRefinementsResponse | undefined,
): ReadonlySet<string> {
  const ids = new Set<string>();
  for (const proposal of result?.proposals ?? []) {
    if (proposal.action !== "page_keep_or_archive") continue;
    const payloadId =
      proposal.payload?.action === "page_keep_or_archive"
        ? proposal.payload.page_id
        : null;
    if (payloadId) ids.add(payloadId);
  }
  return ids;
}
