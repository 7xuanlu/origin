// SPDX-License-Identifier: AGPL-3.0-only
//
// These assertions are a contract with the daemon, not a style preference.
// Each case below mirrors the Rust behaviour in
// `wenlan_core::export::obsidian::slugify` /
// `wenlan_core::page_map_improve::extract_headings`; if one starts failing
// because someone "cleaned up" the port, boxes go dangling in the UI.
import { describe, expect, it } from "vitest";
import { extractHeadings, slugify, withHeading } from "./slug";

describe("slugify", () => {
  it("lowercases and hyphenates spaces", () => {
    expect(slugify("Open Questions")).toBe("open-questions");
  });

  it("drops punctuation rather than replacing it", () => {
    expect(slugify("What's next?")).toBe("whats-next");
  });

  it("collapses dash pairs the way Rust's split(\"--\") does, not fully", () => {
    // "a---b" -> split on "--" -> ["a", "-b"] -> join("-") -> "a--b".
    expect(slugify("a---b")).toBe("a--b");
  });

  it("trims leading and trailing dashes", () => {
    expect(slugify("  spaced  ")).toBe("spaced");
  });

  it("keeps CJK characters (is_alphanumeric is Unicode-aware)", () => {
    expect(slugify("问蓝 设计")).toBe("问蓝-设计");
  });

  it("returns empty for a heading with nothing sluggable", () => {
    expect(slugify("!!!")).toBe("");
  });
});

describe("extractHeadings", () => {
  it("reads every ATX level and strips the hashes", () => {
    expect(extractHeadings("# One\n\ntext\n\n### Three\n")).toEqual(["One", "Three"]);
  });

  it("de-duplicates case-insensitively, keeping first seen", () => {
    expect(extractHeadings("## Alpha\n## ALPHA\n")).toEqual(["Alpha"]);
  });

  it("ignores hashes that are not headings", () => {
    expect(extractHeadings("no heading here\n")).toEqual([]);
  });

  it("survives CRLF line endings", () => {
    expect(extractHeadings("## One\r\n## Two\r\n")).toEqual(["One", "Two"]);
  });
});

describe("withHeading", () => {
  it("appends a section when the anchor is missing", () => {
    expect(withHeading("Body.", "Next steps")).toBe("Body.\n\n## Next steps\n");
  });

  it("is a no-op when a heading already shares the slug", () => {
    const content = "## Next Steps\n\nbody";
    expect(withHeading(content, "next steps")).toBe(content);
  });

  it("does not lead with blank lines on an empty page", () => {
    expect(withHeading("", "First")).toBe("## First\n");
  });

  it("refuses a heading that slugifies to nothing", () => {
    expect(withHeading("Body.", "???")).toBe("Body.");
  });
});
