// SPDX-License-Identifier: AGPL-3.0-only
import { describe, it, expect } from "vitest";
import {
  classifyContent,
  prepareForRender,
  extractPreview,
  normalizeContent,
} from "./contentClassifier";

describe("classifyContent", () => {
  it("returns 'structured' when structuredFields has keys", () => {
    expect(
      classifyContent("some content", '{"context":"auth","decision":"JWT"}')
    ).toBe("structured");
  });

  it("returns 'structured' only for non-empty JSON objects", () => {
    expect(classifyContent("content", "{}")).not.toBe("structured");
    expect(classifyContent("content", "")).not.toBe("structured");
    expect(classifyContent("content", null)).not.toBe("structured");
  });

  it("returns 'key-value' for lines matching 'label: value'", () => {
    const text = "Context: auth rewrite\nDecision: use JWT\nRationale: stateless";
    expect(classifyContent(text)).toBe("key-value");
  });

  it("does not classify single key-value line as key-value", () => {
    expect(classifyContent("Context: auth rewrite")).not.toBe("key-value");
  });

  it("returns 'list' for bullet list content", () => {
    const text = "- Use libSQL\n- Keep local-first\n- AGPL license";
    expect(classifyContent(text)).toBe("list");
  });

  it("returns 'list' for numbered list content", () => {
    const text = "1. First step\n2. Second step\n3. Third step";
    expect(classifyContent(text)).toBe("list");
  });

  it("returns 'list' for asterisk lists", () => {
    const text = "* Item one\n* Item two";
    expect(classifyContent(text)).toBe("list");
  });

  it("returns 'mixed' for prose followed by a list", () => {
    const text = "We decided the following approach:\n- Use libSQL\n- Keep local-first\n- AGPL license";
    expect(classifyContent(text)).toBe("mixed");
  });

  it("returns 'prose' for multi-paragraph content", () => {
    const text = "This is the first paragraph about a decision we made.\n\nThis is the second paragraph with more context about why.";
    expect(classifyContent(text)).toBe("prose");
  });

  it("returns 'prose' for long single-paragraph text over 200 chars", () => {
    const text = "A".repeat(201);
    expect(classifyContent(text)).toBe("prose");
  });

  it("returns 'single-fact' for short content", () => {
    expect(classifyContent("Lucian prefers TDD")).toBe("single-fact");
  });

  it("returns 'single-fact' for short content with trailing newline", () => {
    expect(classifyContent("Lucian prefers TDD\n")).toBe("single-fact");
  });

  it("returns 'single-fact' for empty-ish content", () => {
    expect(classifyContent("")).toBe("single-fact");
  });

  it("handles key-value with URLs in values (not confused by http://)", () => {
    const text = "Source: https://example.com\nTopic: API design";
    expect(classifyContent(text)).toBe("key-value");
  });

  it("does not misclassify prose with a single colon as key-value", () => {
    const text = "The reason: we needed something faster for production workloads.";
    expect(classifyContent(text)).not.toBe("key-value");
  });
});

describe("prepareForRender", () => {
  it("bolds key-value labels", () => {
    const text = "Context: auth rewrite\nDecision: use JWT";
    const result = prepareForRender(text, "key-value");
    expect(result).toBe("**Context:** auth rewrite\n**Decision:** use JWT");
  });

  it("is a no-op for list content", () => {
    const text = "- Use libSQL\n- Keep local-first";
    expect(prepareForRender(text, "list")).toBe(text);
  });

  it("is a no-op for prose", () => {
    const text = "First paragraph.\n\nSecond paragraph.";
    expect(prepareForRender(text, "prose")).toBe(text);
  });

  it("is a no-op for single-fact", () => {
    const text = "Lucian prefers TDD";
    expect(prepareForRender(text, "single-fact")).toBe(text);
  });

  it("is a no-op for structured shape", () => {
    expect(prepareForRender("whatever", "structured")).toBe("whatever");
  });

  it("bolds key-value lines in mixed content, leaves others untouched", () => {
    const text = "We decided:\nContext: auth rewrite\nDecision: use JWT";
    const result = prepareForRender(text, "mixed");
    expect(result).toBe("We decided:\n**Context:** auth rewrite\n**Decision:** use JWT");
  });

  it("converts pipe-delimited lines to bold key-value", () => {
    const text = "context | auth rewrite\ndecision | use JWT";
    const result = prepareForRender(text, "key-value");
    expect(result).toBe("**context:** auth rewrite\n**decision:** use JWT");
  });

  it("is idempotent on already-bold key-value", () => {
    const text = "**Context:** auth rewrite\n**Decision:** use JWT";
    const result = prepareForRender(text, "key-value");
    expect(result).toBe(text);
  });
});

describe("normalizeContent", () => {
  it("is a no-op for content with existing newlines", () => {
    const text = "Line one\nLine two\nLine three";
    expect(normalizeContent(text)).toBe(text);
  });

  it("splits pipe-delimited content into lines", () => {
    const text = "claim: Memory is important | domain: origin | verified: true";
    const result = normalizeContent(text);
    expect(result).toBe("claim: Memory is important\ndomain: origin\nverified: true");
  });

  it("splits pipe-delimited with key | value format", () => {
    const text = "context | auth rewrite | decision | use JWT";
    const result = normalizeContent(text);
    expect(result).toContain("\n");
    expect(result.split("\n").length).toBe(4);
  });

  it("does not split on single pipe (not pipe-delimited)", () => {
    const text = "This is a sentence with a single | pipe in it.";
    expect(normalizeContent(text)).not.toContain("\n");
  });

  it("converts inline numbering to list items", () => {
    const text = "Origin needs: (1) auto relation creation, (2) 2-hop traversal, (3) session grouping";
    const result = normalizeContent(text);
    expect(result).toContain("\n");
    expect(result).toMatch(/^Origin needs:\n/);
    expect(result).toContain("- auto relation creation");
    expect(result).toContain("- 2-hop traversal");
    expect(result).toContain("- session grouping");
  });

  it("handles numbered items with closing paren: 1) 2) 3)", () => {
    const text = "Steps: 1) install deps, 2) run tests, 3) deploy";
    const result = normalizeContent(text);
    expect(result).toContain("- install deps");
    expect(result).toContain("- run tests");
  });

  it("inserts paragraph breaks in long prose", () => {
    const text = "First sentence about decisions. Second sentence with more context. Third sentence about rationale. Fourth sentence wrapping up the discussion. Fifth sentence adding even more detail to reach the threshold.";
    const result = normalizeContent(text);
    expect(result).toContain("\n\n");
    expect(result.split("\n\n").length).toBeGreaterThan(1);
  });

  it("does not break short content", () => {
    const text = "Lucian prefers TDD.";
    expect(normalizeContent(text)).toBe(text);
  });

  it("does not break content under 300 chars even with sentences", () => {
    const text = "First sentence. Second sentence. Third sentence.";
    expect(normalizeContent(text)).toBe(text);
  });

  it("preserves content that already has structure", () => {
    const text = "# Heading\n\nSome paragraph.\n\n- List item";
    expect(normalizeContent(text)).toBe(text);
  });
});

describe("extractPreview", () => {
  it("returns full text for single-fact", () => {
    expect(extractPreview("Lucian prefers TDD", "single-fact")).toBe(
      "Lucian prefers TDD"
    );
  });

  it("returns first key-value pair for key-value shape", () => {
    const text = "Context: auth rewrite\nDecision: use JWT";
    const result = extractPreview(text, "key-value");
    expect(result).toEqual({ key: "Context", value: "auth rewrite" });
  });

  it("returns first item for list shape", () => {
    const text = "- Use libSQL\n- Keep local-first";
    expect(extractPreview(text, "list")).toBe("Use libSQL");
  });

  it("returns first sentence for prose", () => {
    const text = "We decided to use libSQL for storage. It supports vectors natively.";
    expect(extractPreview(text, "prose")).toBe(
      "We decided to use libSQL for storage."
    );
  });

  it("truncates long first sentence in prose to 120 chars", () => {
    const text = "A".repeat(200) + ". Next sentence.";
    const result = extractPreview(text, "prose");
    expect(typeof result).toBe("string");
    expect((result as string).length).toBeLessThanOrEqual(123); // 120 + "..."
  });

  it("returns first non-empty line for mixed", () => {
    const text = "We decided the following:\n- Use libSQL\n- Keep local-first";
    expect(extractPreview(text, "mixed")).toBe("We decided the following:");
  });

  it("returns null for structured (handled separately)", () => {
    expect(extractPreview("content", "structured")).toBeNull();
  });
});
