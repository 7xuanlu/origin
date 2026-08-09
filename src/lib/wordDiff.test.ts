// SPDX-License-Identifier: AGPL-3.0-only
import { describe, expect, it } from "vitest";
import { diffWords, diffWordCounts } from "./wordDiff";

describe("diffWords", () => {
  it("returns a single same segment for identical text", () => {
    expect(diffWords("keep it simple", "keep it simple")).toEqual([
      { kind: "same", text: "keep it simple" },
    ]);
  });

  it("marks appended words as inserted", () => {
    expect(diffWords("use pnpm", "use pnpm for installs")).toEqual([
      { kind: "same", text: "use pnpm" },
      { kind: "ins", text: " for installs" },
    ]);
  });

  it("marks removed words as deleted", () => {
    expect(diffWords("use pnpm for installs", "use pnpm")).toEqual([
      { kind: "same", text: "use pnpm" },
      { kind: "del", text: " for installs" },
    ]);
  });

  it("pairs a replacement as del followed by ins", () => {
    const segments = diffWords("prefers npm always", "prefers pnpm always");
    expect(segments).toEqual([
      { kind: "same", text: "prefers" },
      { kind: "del", text: " npm" },
      { kind: "ins", text: " pnpm" },
      { kind: "same", text: " always" },
    ]);
  });

  it("handles fully different strings", () => {
    expect(diffWords("old fact", "new statement entirely")).toEqual([
      { kind: "del", text: "old fact" },
      { kind: "ins", text: "new statement entirely" },
    ]);
  });

  it("handles empty before and after", () => {
    expect(diffWords("", "added")).toEqual([{ kind: "ins", text: "added" }]);
    expect(diffWords("gone", "")).toEqual([{ kind: "del", text: "gone" }]);
    expect(diffWords("", "")).toEqual([]);
  });
});

describe("diffWordCounts", () => {
  it("counts added and removed words", () => {
    const segments = diffWords("a b c d", "a x y c d z");
    expect(diffWordCounts(segments)).toEqual({ added: 3, removed: 1 });
  });
});
