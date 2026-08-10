// SPDX-License-Identifier: AGPL-3.0-only
import { describe, expect, it } from "vitest";
import { leadingMarkdownH1MatchesTitle } from "./pageEditorPresentation";

describe("leadingMarkdownH1MatchesTitle", () => {
  it("matches the editable leading H1 to the Page title without rewriting source", () => {
    expect(
      leadingMarkdownH1MatchesTitle(
        "# Native Editor Review\n",
        "Native Editor Review",
      ),
    ).toBe(true);
    expect(
      leadingMarkdownH1MatchesTitle(
        "\uFEFF\n#  Native   Editor Review  ##\r\n",
        "Native Editor Review",
      ),
    ).toBe(true);
  });

  it("does not treat a different Markdown construct as the Page title", () => {
    expect(
      leadingMarkdownH1MatchesTitle(
        "## Native Editor Review\n",
        "Native Editor Review",
      ),
    ).toBe(false);
    expect(
      leadingMarkdownH1MatchesTitle(
        "Native Editor Review\n====================\n",
        "Native Editor Review",
      ),
    ).toBe(false);
    expect(
      leadingMarkdownH1MatchesTitle(
        "Paragraph\n\n# Native Editor Review\n",
        "Native Editor Review",
      ),
    ).toBe(false);
  });

  it("keeps comparison case-sensitive and source-visible", () => {
    expect(
      leadingMarkdownH1MatchesTitle("# Different\n", "Native Editor Review"),
    ).toBe(false);
    expect(
      leadingMarkdownH1MatchesTitle(
        "# native editor review\n",
        "Native Editor Review",
      ),
    ).toBe(false);
    expect(
      leadingMarkdownH1MatchesTitle(
        "# **Native Editor Review**\n",
        "Native Editor Review",
      ),
    ).toBe(false);
  });

  it("rejects empty titles and code-indented hashes", () => {
    expect(leadingMarkdownH1MatchesTitle("# \n", "")).toBe(false);
    expect(
      leadingMarkdownH1MatchesTitle(
        "    # Native Editor Review\n",
        "Native Editor Review",
      ),
    ).toBe(false);
  });
});
