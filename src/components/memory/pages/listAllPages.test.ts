import { beforeEach, describe, expect, it, vi } from "vitest";
import { listPages, type Page } from "../../../lib/tauri";
import { listAllActivePages, listAllDraftPages } from "./listAllPages";

vi.mock("../../../lib/tauri", () => ({ listPages: vi.fn() }));

function page(id: string): Page {
  return {
    id,
    title: id,
    summary: null,
    content: "",
    entity_id: null,
    domain: null,
    space: null,
    source_memory_ids: [],
    version: 1,
    status: "active",
    created_at: "",
    last_compiled: "",
    last_modified: "",
  };
}

describe("listAllActivePages", () => {
  beforeEach(() => vi.mocked(listPages).mockReset());

  it("paginates until the final partial batch instead of presenting a capped slice as all pages", async () => {
    vi.mocked(listPages)
      .mockResolvedValueOnce(Array.from({ length: 500 }, (_, index) => page(`page-${index}`)))
      .mockResolvedValueOnce(Array.from({ length: 500 }, (_, index) => page(`page-${index + 500}`)))
      .mockResolvedValueOnce([page("page-1000")]);

    const result = await listAllActivePages();

    expect(result).toHaveLength(1001);
    expect(listPages).toHaveBeenNthCalledWith(1, "active", undefined, 500, 0);
    expect(listPages).toHaveBeenNthCalledWith(2, "active", undefined, 500, 500);
    expect(listPages).toHaveBeenNthCalledWith(3, "active", undefined, 500, 1000);
  });

  it("stops safely if a legacy backend ignores the offset and repeats a full batch", async () => {
    const repeated = Array.from({ length: 500 }, (_, index) => page(`page-${index}`));
    vi.mocked(listPages).mockResolvedValue(repeated);

    const result = await listAllActivePages();

    expect(result).toHaveLength(500);
    expect(listPages).toHaveBeenCalledTimes(2);
  });
});

describe("listAllDraftPages", () => {
  beforeEach(() => vi.mocked(listPages).mockReset());

  it("paginates drafts independently from the active inventory", async () => {
    vi.mocked(listPages)
      .mockResolvedValueOnce(Array.from({ length: 500 }, (_, index) => ({
        ...page(`draft-${index}`),
        status: "draft",
      })))
      .mockResolvedValueOnce([{
        ...page("draft-500"),
        status: "draft",
      }]);

    const result = await listAllDraftPages();

    expect(result).toHaveLength(501);
    expect(listPages).toHaveBeenNthCalledWith(1, "draft", undefined, 500, 0);
    expect(listPages).toHaveBeenNthCalledWith(2, "draft", undefined, 500, 500);
  });

  it("dedupes repeated ids and stops if a legacy backend repeats a full page", async () => {
    const repeated = Array.from({ length: 500 }, (_, index) => ({
      ...page(`draft-${index}`),
      status: "draft",
    }));
    vi.mocked(listPages).mockResolvedValue(repeated);

    const result = await listAllDraftPages();

    expect(result).toHaveLength(500);
    expect(listPages).toHaveBeenCalledTimes(2);
  });
});
