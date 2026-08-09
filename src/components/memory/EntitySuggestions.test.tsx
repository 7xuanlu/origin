// SPDX-License-Identifier: AGPL-3.0-only
import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import EntitySuggestions from "./EntitySuggestions";

vi.mock("../../lib/tauri", () => ({
  getEntitySuggestions: vi.fn(),
  approveEntitySuggestion: vi.fn(),
  dismissEntitySuggestion: vi.fn(),
}));

import * as tauri from "../../lib/tauri";

function renderEntitySuggestions() {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={queryClient}>
      <EntitySuggestions />
    </QueryClientProvider>,
  );
}

beforeEach(() => {
  vi.mocked(tauri.getEntitySuggestions).mockReset();
  vi.mocked(tauri.approveEntitySuggestion).mockReset();
  vi.mocked(tauri.dismissEntitySuggestion).mockReset();
});

describe("EntitySuggestions", () => {
  it("offers dismiss but not create for daemon suggest_entity proposals", async () => {
    vi.mocked(tauri.getEntitySuggestions).mockResolvedValue([
      {
        id: "ref-suggest",
        entity_name: "Wenlan",
        source_ids: ["mem-a"],
        confidence: 0.8,
        created_at: "2026-06-28T00:00:00Z",
      },
    ]);
    vi.mocked(tauri.dismissEntitySuggestion).mockResolvedValue({ id: "ref-suggest" });
    renderEntitySuggestions();

    expect(await screen.findByText("Wenlan")).toBeTruthy();
    expect(screen.queryByRole("button", { name: /create/i })).toBeNull();

    await userEvent.click(screen.getByRole("button", { name: /dismiss/i }));

    await waitFor(() => {
      expect(tauri.dismissEntitySuggestion).toHaveBeenCalledWith("ref-suggest");
    });
    expect(tauri.approveEntitySuggestion).not.toHaveBeenCalled();
  });
});
