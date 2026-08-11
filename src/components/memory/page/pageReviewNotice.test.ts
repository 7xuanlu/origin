// SPDX-License-Identifier: AGPL-3.0-only
import { describe, expect, it } from "vitest";
import type { TFunction } from "i18next";
import type { PageReviewOutcome } from "../../../lib/tauri";
import { pageReviewNotice } from "./pageReviewNotice";

const t = ((key: string) => key) as unknown as TFunction;

describe("pageReviewNotice", () => {
  it("gives every outcome its own message", () => {
    // The refusals are coarse on the wire precisely so that none of them leaks
    // detail. That only helps if this side still tells them apart — a mapping
    // that collapsed two of them would show the reader a reason that is not
    // the reason.
    const outcomes: PageReviewOutcome[] = [
      { kind: "applied", reviewed_page_version: 3 },
      { kind: "stale" },
      { kind: "expired" },
      { kind: "invalid" },
      { kind: "replayed" },
      { kind: "unavailable" },
      { kind: "unsupported" },
    ];

    const messages = outcomes.map((outcome) => pageReviewNotice(outcome, t).message);

    expect(new Set(messages).size).toBe(outcomes.length);
    expect(messages.every((message) => message.startsWith("pageDetail.review"))).toBe(true);
  });

  it("reads an already-recorded review as done, not as a failure", () => {
    expect(pageReviewNotice({ kind: "replayed" }, t).kind).toBe("success");
    expect(pageReviewNotice({ kind: "applied", reviewed_page_version: 1 }, t).kind).toBe("success");
  });

  it("offers a reload only when the page moved on", () => {
    // A reload fixes exactly one of these. Anywhere else it would be an action
    // that cannot help, offered as though it could.
    expect(pageReviewNotice({ kind: "stale" }, t).offerReload).toBe(true);
    for (const outcome of [
      { kind: "expired" },
      { kind: "invalid" },
      { kind: "unavailable" },
      { kind: "unsupported" },
      { kind: "replayed" },
    ] as PageReviewOutcome[]) {
      expect(pageReviewNotice(outcome, t).offerReload).toBe(false);
    }
  });

  it("keeps a daemon or install problem louder than a timing one", () => {
    // Expiry is ordinary and self-healing; an unreadable install secret is not.
    expect(pageReviewNotice({ kind: "expired" }, t).kind).toBe("warning");
    expect(pageReviewNotice({ kind: "unavailable" }, t).kind).toBe("error");
    expect(pageReviewNotice({ kind: "invalid" }, t).kind).toBe("error");
  });
});
