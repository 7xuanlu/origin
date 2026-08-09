import { describe, expect, it } from "vitest";
import type { Page } from "../../../lib/tauri";
import { classifyPage, pageSpaceContext } from "./pagePresentation";

function page(overrides: Partial<Page> = {}): Page {
  return {
    id: "page-1",
    title: "Ambient memory",
    summary: null,
    content: "Long-lived context for ongoing work.",
    entity_id: null,
    domain: null,
    space: null,
    source_memory_ids: [],
    version: 1,
    status: "active",
    created_at: "2026-07-01T00:00:00Z",
    last_compiled: "2026-07-01T00:00:00Z",
    last_modified: "2026-07-10T00:00:00Z",
    ...overrides,
  };
}

describe("page presentation", () => {
  it("uses the durable Entity relationship as the only specialized Page kind", () => {
    expect(classifyPage(page({ entity_id: "entity-1", title: "A decision recap" }))).toBe("entity");
    expect(classifyPage(page())).toBe("page");
  });

  it.each([
    ["English decision body", page({ title: "Why citations stay visible", content: "Decision: keep citations visible." })],
    ["English recap title", page({ title: "July research recap" })],
    ["Simplified Chinese decision title", page({ title: "认证机制决策" })],
    ["Traditional Chinese decision body", page({ title: "認證機制", content: "## 決策：保留來源引用。" })],
    ["Simplified Chinese recap title", page({ title: "七月研究回顾" })],
    ["Traditional Chinese recap title", page({ title: "七月研究回顧" })],
  ] as const)("keeps %s as a Page instead of guessing a schema from prose", (_name, candidate) => {
    expect(classifyPage(candidate)).toBe("page");
  });

  it("keeps explicit no-Space pages blank without reviving a legacy domain", () => {
    expect(pageSpaceContext(page({ space: null, domain: "Legacy" }))).toBeUndefined();
    expect(pageSpaceContext(page({ space: undefined, domain: "Legacy" }))).toBe("Legacy");
    expect(pageSpaceContext(page({ space: "  Wenlan  ", domain: "Legacy" }))).toBe("Wenlan");
  });
});
