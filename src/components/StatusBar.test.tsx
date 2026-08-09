// SPDX-License-Identifier: AGPL-3.0-only
import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi, beforeEach } from "vitest";
import StatusBar from "./StatusBar";
import { useIndexStatus } from "../hooks/useSearch";
import type { IndexStatus } from "../lib/tauri";

vi.mock("../hooks/useSearch", () => ({
  useIndexStatus: vi.fn(),
}));

const mockUseIndexStatus = vi.mocked(useIndexStatus);

beforeEach(() => {
  mockUseIndexStatus.mockReset();
});

describe("StatusBar daemon diagnostics", () => {
  it("surfaces daemon reranker failures", () => {
    const status: IndexStatus = {
      is_running: false,
      files_indexed: 42,
      files_total: 0,
      last_error: null,
      sources_connected: [],
      reranker: { state: "failed", reason: "model missing" },
      reranker_light: { state: "disabled" },
      reranker_mode: "full",
    };
    mockUseIndexStatus.mockReturnValue({
      data: status,
    } as unknown as ReturnType<typeof useIndexStatus>);

    render(<StatusBar resultCount={0} />);

    expect(screen.getByText("Reranker failed")).toBeInTheDocument();
    expect(screen.getByText("Reranker failed")).toHaveAttribute("title", "model missing");
  });
});
