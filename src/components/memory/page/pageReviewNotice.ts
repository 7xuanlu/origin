// SPDX-License-Identifier: AGPL-3.0-only
import type { TFunction } from "i18next";
import type { PageReviewOutcome } from "../../../lib/tauri";

export interface PageReviewNotice {
  kind: "success" | "warning" | "error";
  message: string;
  /**
   * Whether to offer a reload alongside the message. True only for the stale
   * case, which is the one outcome a reload actually fixes — the page moved on
   * and the reader needs the current text before they can approve anything.
   * Offering it elsewhere would suggest the reader can act on a refusal that
   * has nothing to do with what they are looking at.
   */
  offerReload: boolean;
}

/**
 * Turns the daemon's answer into what the reader is told.
 *
 * Kept out of the component so the mapping can be checked without rendering
 * one: the whole point of the tagged outcome is that each refusal leads
 * somewhere different, and a switch buried in JSX is where that quietly
 * collapses back into "something went wrong".
 */
export function pageReviewNotice(
  outcome: PageReviewOutcome,
  t: TFunction,
): PageReviewNotice {
  switch (outcome.kind) {
    case "applied":
      return { kind: "success", message: t("pageDetail.reviewApplied"), offerReload: false };
    // Already recorded is a success from the reader's side: the page carries
    // the mark they were asking for. Reporting it as a failure would invite
    // them to retry something that is already done.
    case "replayed":
      return { kind: "success", message: t("pageDetail.reviewReplayed"), offerReload: false };
    case "stale":
      return { kind: "warning", message: t("pageDetail.reviewStale"), offerReload: true };
    // The capability aged out in flight. Nothing is wrong with the page or the
    // install, and clicking again mints a fresh one — so this is a nudge, not
    // an error.
    case "expired":
      return { kind: "warning", message: t("pageDetail.reviewExpired"), offerReload: false };
    case "unsupported":
      return { kind: "warning", message: t("pageDetail.reviewUnsupported"), offerReload: false };
    case "unavailable":
      return { kind: "error", message: t("pageDetail.reviewUnavailable"), offerReload: false };
    case "invalid":
      return { kind: "error", message: t("pageDetail.reviewInvalid"), offerReload: false };
  }
}
