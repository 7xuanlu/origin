// SPDX-License-Identifier: AGPL-3.0-only
import { describe, it, expect } from "vitest";
import type { PageCitation } from "./tauri";
import {
  processCitations,
  stripCitationLinks,
  citationDisplayLabel,
} from "./pageCitations";

const cite = (
  occurrence: number,
  marker: number,
  over: Partial<PageCitation> = {},
): PageCitation => ({
  occurrence,
  marker,
  source_kind: "memory",
  locator: `mem-${marker}`,
  score: 0.9,
  status: "verified",
  scope: "sentence",
  ...over,
});

describe("processCitations", () => {
  it("returns state none and unchanged content when no markers and no citations", () => {
    const r = processCitations("Plain prose. No markers here.", undefined);
    expect(r.state).toBe("none");
    expect(r.content).toBe("Plain prose. No markers here.");
    expect(r.byOccurrence.size).toBe(0);
  });

  it("rewrites markers to occurrence-indexed citation links in body order", () => {
    const r = processCitations("A claim.[1] Another.[2]", [cite(1, 1), cite(2, 2)]);
    expect(r.state).toBe("cited");
    expect(r.content).toBe("A claim.[1](#citation:1) Another.[2](#citation:2)");
    expect(r.byOccurrence.get(1)?.locator).toBe("mem-1");
    expect(r.byOccurrence.get(2)?.locator).toBe("mem-2");
  });

  it("numbers multi-source runs like [1][3] by occurrence, not marker", () => {
    const r = processCitations("Claim.[1][3]", [cite(1, 1), cite(2, 3)]);
    expect(r.state).toBe("cited");
    expect(r.content).toBe("Claim.[1](#citation:1)[2](#citation:2)");
  });

  it("counts matches inside fenced code but does not rewrite them", () => {
    const content = "```\narr[1] access\n```\nA claim.[2]";
    const r = processCitations(content, [cite(1, 1), cite(2, 2)]);
    expect(r.state).toBe("cited");
    expect(r.content).toBe("```\narr[1] access\n```\nA claim.[2](#citation:2)");
  });

  it("counts matches inside inline code but does not rewrite them", () => {
    const content = "Use `x[1]` here.[2]";
    const r = processCitations(content, [cite(1, 1), cite(2, 2)]);
    expect(r.state).toBe("cited");
    expect(r.content).toBe("Use `x[1]` here.[2](#citation:2)");
  });

  it("falls back to strip-all when match count disagrees with citations length", () => {
    const r = processCitations("A.[1] B.[2]", [cite(1, 1)]);
    expect(r.state).toBe("stripped-mismatch");
    expect(r.content).toBe("A. B.");
    expect(r.byOccurrence.size).toBe(0);
  });

  it("falls back to strip-all when a marker value disagrees", () => {
    const r = processCitations("A.[5]", [cite(1, 1)]);
    expect(r.state).toBe("stripped-mismatch");
    expect(r.content).toBe("A.");
    expect(r.content).not.toContain("#citation");
  });

  it("display-strips markers when citations are empty (user-edited page)", () => {
    const r = processCitations("A. [1] B.[2]", []);
    expect(r.state).toBe("stripped-empty");
    expect(r.content).toBe("A. B.");
  });

  it("display-strips markers when citations are absent (old daemon shape)", () => {
    const r = processCitations("A.[1] B.", undefined);
    expect(r.state).toBe("stripped-empty");
    expect(r.content).toBe("A. B.");
  });

  it("collapses doubled spaces left by stripping, mirroring backend strip_markers", () => {
    const r = processCitations("A. [1] middle [2] B.", []);
    expect(r.content).toBe("A. middle B.");
  });
});

describe("stripCitationLinks", () => {
  it("removes rewritten citation links and collapses spacing", () => {
    expect(stripCitationLinks("First claim [1](#citation:1) done.")).toBe(
      "First claim done.",
    );
    expect(stripCitationLinks("Tail.[2](#citation:2)")).toBe("Tail.");
  });
});

describe("citationDisplayLabel", () => {
  it("shows memory locators as-is", () => {
    expect(citationDisplayLabel(cite(1, 1))).toBe("mem-1");
  });

  it("truncates URLs to hostname", () => {
    const c = cite(1, 1, {
      source_kind: "external_url",
      locator: "https://docs.rs/serde/latest/serde/",
    });
    expect(citationDisplayLabel(c)).toBe("docs.rs");
  });

  it("truncates file paths to basename", () => {
    const c = cite(1, 1, {
      source_kind: "external_file",
      locator: "/Users/l/notes/design.md",
    });
    expect(citationDisplayLabel(c)).toBe("design.md");
  });

  it("labels authored citations", () => {
    expect(citationDisplayLabel(cite(1, 1, { source_kind: "authored" }))).toBe(
      "authored",
    );
  });
});
