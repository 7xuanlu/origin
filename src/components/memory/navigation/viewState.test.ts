import { describe, expect, it } from "vitest";
import { activeNavigationForView } from "./viewState";

describe("activeNavigationForView", () => {
  it("keeps the Page library and Page detail first-class instead of nesting them under Spaces", () => {
    expect(activeNavigationForView({ kind: "pages" })).toBe("pages");
    expect(activeNavigationForView({ kind: "page", pageId: "page-1" })).toBe("pages");
    expect(activeNavigationForView({
      kind: "page-draft",
      draftId: "draft-1",
      space: "Launch",
    })).toBe("pages");
    expect(activeNavigationForView({ kind: "entity", entityId: "entity-1" })).toBe("pages");
    expect(activeNavigationForView({ kind: "space", spaceId: "space-1", spaceName: "Launch" })).toBe("spaces");
  });
});
