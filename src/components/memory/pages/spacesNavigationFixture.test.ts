import { describe, expect, it } from "vitest";
import { createSpacesNavigationFixture } from "../../../../e2e/fixtures/spacesNavigation";
import { classifyPage, pageSpaceContext } from "./pagePresentation";

describe("Spaces navigation fixture Wiki coverage", () => {
  it("shows Page and Entity kinds plus an explicitly no-Space Page", () => {
    const pages = createSpacesNavigationFixture().pages;

    expect(new Set(pages.map(classifyPage))).toEqual(new Set(["page", "entity"]));
    expect(pages.find((page) => classifyPage(page) === "entity")?.entity_id).toBe("entity-ada");
    expect(pages.some((page) => page.space === null && page.domain === null && pageSpaceContext(page) === undefined)).toBe(true);
  });
});
