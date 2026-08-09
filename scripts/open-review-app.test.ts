// SPDX-License-Identifier: AGPL-3.0-only
import { describe, expect, it } from "vitest";

import { restartReviewApp } from "./open-review-app.mjs";

describe("Review app launcher", () => {
  it("waits for an existing Review process to exit before opening the bundle", async () => {
    const events: string[] = [];
    let polls = 0;
    await restartReviewApp({
      getPids: () => {
        polls += 1;
        return polls < 3 ? [52406] : [];
      },
      terminate: (pid) => events.push(`terminate:${pid}`),
      sleep: async (milliseconds) => events.push(`sleep:${milliseconds}`),
      launch: () => events.push("launch"),
      pollIntervalMs: 50,
    });

    expect(events).toEqual([
      "terminate:52406",
      "sleep:50",
      "launch",
    ]);
  });
});
