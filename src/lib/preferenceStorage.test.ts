// SPDX-License-Identifier: AGPL-3.0-only
import { beforeEach, describe, expect, it, vi } from "vitest";
import { readPreference, writePreference } from "./preferenceStorage";

describe("preference storage bridge", () => {
  beforeEach(() => {
    localStorage.clear();
    vi.restoreAllMocks();
  });

  it("prefers the Wenlan key over legacy Origin state", () => {
    localStorage.setItem("origin-example", "legacy");
    localStorage.setItem("wenlan-example", "current");

    expect(readPreference("wenlan-example", "origin-example")).toBe("current");
  });

  it("imports legacy Origin state into the Wenlan key when missing", () => {
    localStorage.setItem("origin-example", "legacy");

    expect(readPreference("wenlan-example", "origin-example")).toBe("legacy");
    expect(localStorage.getItem("wenlan-example")).toBe("legacy");
    expect(localStorage.getItem("origin-example")).toBe("legacy");
  });

  it("writes only the Wenlan key for new state", () => {
    writePreference("wenlan-example", "current");

    expect(localStorage.getItem("wenlan-example")).toBe("current");
    expect(localStorage.getItem("origin-example")).toBeNull();
  });

  it("returns null when preference reads throw", () => {
    vi.spyOn(Storage.prototype, "getItem").mockImplementation(() => {
      throw new Error("storage disabled");
    });

    expect(readPreference("wenlan-example", "origin-example")).toBeNull();
  });

  it("still returns legacy state when import writes throw", () => {
    localStorage.setItem("origin-example", "legacy");
    vi.spyOn(Storage.prototype, "setItem").mockImplementation(() => {
      throw new Error("storage full");
    });

    expect(readPreference("wenlan-example", "origin-example")).toBe("legacy");
  });

  it("does not throw when preference writes fail", () => {
    vi.spyOn(Storage.prototype, "setItem").mockImplementation(() => {
      throw new Error("storage full");
    });

    expect(() => writePreference("wenlan-example", "current")).not.toThrow();
  });
});
