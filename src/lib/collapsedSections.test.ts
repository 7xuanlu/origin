// SPDX-License-Identifier: AGPL-3.0-only
import { beforeEach, describe, expect, it, vi } from "vitest";
import { getCollapsedSections } from "./collapsedSections";

describe("collapsed section preference bridge", () => {
  beforeEach(() => {
    localStorage.clear();
    vi.useRealTimers();
  });

  it("imports legacy Origin collapsed-section state into the Wenlan key", () => {
    const collapsedAt = Date.now();
    localStorage.setItem(
      "origin:collapsed-sections",
      JSON.stringify([{ id: "group-1", collapsedAt }]),
    );

    expect(getCollapsedSections()).toEqual(new Set(["group-1"]));
    expect(localStorage.getItem("wenlan:collapsed-sections")).toBe(
      JSON.stringify([{ id: "group-1", collapsedAt }]),
    );
    expect(localStorage.getItem("origin:collapsed-sections")).not.toBeNull();
  });
});
