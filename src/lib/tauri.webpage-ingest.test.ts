import { beforeEach, describe, expect, it, vi } from "vitest";

const invokeMock = vi.hoisted(() => vi.fn());

vi.mock("@tauri-apps/api/core", () => ({
  invoke: invokeMock,
}));

import { ingestWebpage } from "./tauri";

describe("ingestWebpage", () => {
  beforeEach(() => {
    invokeMock.mockReset();
  });

  it("invokes the typed webpage ingest Tauri command", async () => {
    invokeMock.mockResolvedValue({
      chunks_created: 3,
      document_id: "https://example.com/post",
    });

    const result = await ingestWebpage({
      url: "https://example.com/post",
      title: "Example Post",
      content: "A durable article body.",
      metadata: { source: "manual-url" },
    });

    expect(result).toEqual({
      chunks_created: 3,
      document_id: "https://example.com/post",
    });
    expect(invokeMock).toHaveBeenCalledWith("ingest_webpage", {
      req: {
        url: "https://example.com/post",
        title: "Example Post",
        content: "A durable article body.",
        metadata: { source: "manual-url" },
      },
    });
  });
});
