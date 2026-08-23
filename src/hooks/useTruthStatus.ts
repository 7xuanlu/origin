// SPDX-License-Identifier: AGPL-3.0-only
import { useQuery } from "@tanstack/react-query";
import { getTruthStatus, type TruthStatus } from "../lib/tauri";

export const TRUTH_STATUS_QUERY_KEY = ["truth-status"] as const;

/**
 * TanStack Query options for any query built on an explicit-browse fetcher
 * (`listPagesExplicitBrowse`, `getPage(id, "explicit")`). A machine-triggered
 * refetch — reconnect, window
 * refocus — must not resend the human-intent marker on behalf of a call the
 * user did not make; see the policy application in `PagesOverview.tsx` and
 * the rationale in `crates/wenlan-core/src/truth_contract.rs`.
 */
export const EXPLICIT_BROWSE_QUERY_POLICY = {
  refetchOnReconnect: false,
  refetchOnWindowFocus: false,
} as const;

export interface TruthStatusInfo {
  status: TruthStatus | null;
  /** Whether a live daemon has begun the M5 truth cutover. Conservative on
   *  failure or an unreachable daemon: `false`, same rule `Page.truth`
   *  itself follows — no cutover state is read as "not live", not "unknown,
   *  proceed anyway" (`crates/wenlan-types/src/responses.rs::TruthStatus`). */
  cutoverLive: boolean;
}

/** Whether truth badges should render at all. Call once per view and pass
 *  `cutoverLive` down — a badge on a page with no `truth` is a stale read,
 *  not "not yet supported", so components must not render one without it. */
export function useTruthStatus(): TruthStatusInfo {
  const { data } = useQuery({
    queryKey: TRUTH_STATUS_QUERY_KEY,
    queryFn: getTruthStatus,
    staleTime: 30_000,
    retry: false,
  });
  const status = data ?? null;
  return { status, cutoverLive: status != null && status.cutover_generation > 0 };
}
