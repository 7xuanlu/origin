// SPDX-License-Identifier: AGPL-3.0-only
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { i18n } from "../../../i18n";
import type { EntityDetail as EntityDetailRecord } from "../../../lib/tauri";
import EntityDetail from "../EntityDetail";

vi.mock("../../../lib/tauri", () => ({
  getEntityDetail: vi.fn(),
  updateObservation: vi.fn(),
  deleteObservation: vi.fn().mockResolvedValue(undefined),
  addObservation: vi.fn().mockResolvedValue(undefined),
  confirmObservation: vi.fn().mockResolvedValue(undefined),
  confirmEntity: vi.fn().mockResolvedValue(undefined),
  deleteEntity: vi.fn(),
  search: vi.fn().mockResolvedValue([]),
  FACET_COLORS: {},
}));

import {
  confirmObservation,
  deleteEntity,
  getEntityDetail,
  updateObservation,
} from "../../../lib/tauri";

const detail: EntityDetailRecord = {
  entity: {
    id: "entity-ada",
    name: "Ada Lovelace",
    entity_type: "person",
    domain: "computing",
    space: null,
    source_agent: "research-agent",
    confidence: 0.87,
    confirmed: false,
    created_at: 1_700_000_000,
    updated_at: 1_700_086_400,
  },
  observations: [
    {
      id: "obs-1",
      entity_id: "entity-ada",
      content: "Original observation",
      source_agent: "research-agent",
      confidence: 0.8,
      confirmed: false,
      created_at: 1_700_000_100,
    },
  ],
  relations: [],
};

function renderEntity(onBack = vi.fn()) {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: { retry: false, gcTime: 0, refetchOnWindowFocus: false },
      mutations: { retry: false },
    },
  });
  return {
    onBack,
    queryClient,
    user: userEvent.setup(),
    ...render(
      <QueryClientProvider client={queryClient}>
        <EntityDetail
          entityId="entity-ada"
          onBack={onBack}
          onEntityClick={vi.fn()}
          onMemoryClick={vi.fn()}
        />
      </QueryClientProvider>,
    ),
  };
}

describe("EntityDetail mutation errors", () => {
  beforeEach(async () => {
    vi.clearAllMocks();
    await i18n.changeLanguage("en");
    vi.mocked(getEntityDetail).mockResolvedValue(detail);
  });

  it("keeps a failed observation edit open with its draft and an inline error", async () => {
    vi.mocked(updateObservation).mockRejectedValueOnce(new Error("write failed"));
    const { user } = renderEntity();
    await screen.findByRole("heading", { name: "Ada Lovelace" });

    await user.click(screen.getByRole("button", { name: "Original observation" }));
    const input = screen.getByRole("textbox", { name: "Edit note" });
    await user.clear(input);
    await user.type(input, "Draft survives{Enter}");

    expect(await screen.findByRole("alert")).toHaveTextContent("Couldn't save. Try again.");
    expect(screen.getByRole("textbox", { name: "Edit note" })).toHaveValue("Draft survives");
  });

  it("clears a failed observation edit error when Escape cancels the editor", async () => {
    // Given a failed edit whose draft and single inline error remain visible.
    vi.mocked(updateObservation).mockRejectedValueOnce(new Error("write failed"));
    const { user } = renderEntity();
    await screen.findByRole("heading", { name: "Ada Lovelace" });
    await user.click(screen.getByRole("button", { name: "Original observation" }));
    const input = screen.getByRole("textbox", { name: "Edit note" });
    await user.clear(input);
    await user.type(input, "Draft survives{Enter}");
    expect(await screen.findAllByRole("alert")).toHaveLength(1);
    expect(screen.getByRole("textbox", { name: "Edit note" })).toHaveValue("Draft survives");

    // When the user cancels that editing intent with Escape.
    await user.type(screen.getByRole("textbox", { name: "Edit note" }), "{Escape}");

    // Then both the editor and the canceled mutation error leave the current UI state.
    expect(screen.queryByRole("textbox", { name: "Edit note" })).not.toBeInTheDocument();
    await waitFor(() => expect(screen.queryByRole("alert")).not.toBeInTheDocument());
  });

  it("shows only the later entity delete error after a failed observation edit is canceled", async () => {
    // Given a failed observation edit that the user has canceled.
    vi.mocked(updateObservation).mockRejectedValueOnce(new Error("write failed"));
    vi.mocked(deleteEntity).mockRejectedValueOnce(new Error("delete failed"));
    const { user } = renderEntity();
    await screen.findByRole("heading", { name: "Ada Lovelace" });
    await user.click(screen.getByRole("button", { name: "Original observation" }));
    const input = screen.getByRole("textbox", { name: "Edit note" });
    await user.clear(input);
    await user.type(input, "Draft survives{Enter}");
    expect(await screen.findAllByRole("alert")).toHaveLength(1);
    await user.type(screen.getByRole("textbox", { name: "Edit note" }), "{Escape}");

    // When an independent entity deletion subsequently fails.
    await user.click(screen.getByRole("button", { name: "Delete entity" }));
    await user.click(screen.getByRole("button", { name: /^Delete$/ }));

    // Then the parent deletion is the only current error owner.
    await waitFor(() => expect(deleteEntity).toHaveBeenCalledWith("entity-ada"));
    const alerts = await screen.findAllByRole("alert");
    expect(alerts).toHaveLength(1);
    expect(alerts[0]).toHaveTextContent("Couldn't save. Try again.");
  });

  it("clears a failed observation edit error when the user retries the save", async () => {
    // Given a failed edit whose draft remains active for recovery.
    vi.mocked(updateObservation)
      .mockRejectedValueOnce(new Error("write failed"))
      .mockResolvedValueOnce(undefined);
    const { user } = renderEntity();
    await screen.findByRole("heading", { name: "Ada Lovelace" });
    await user.click(screen.getByRole("button", { name: "Original observation" }));
    const input = screen.getByRole("textbox", { name: "Edit note" });
    await user.clear(input);
    await user.type(input, "Draft survives{Enter}");
    expect(await screen.findAllByRole("alert")).toHaveLength(1);

    // When the user retries the same draft.
    await user.type(screen.getByRole("textbox", { name: "Edit note" }), "{Enter}");

    // Then the successful retry closes the editor and clears the stale error.
    await waitFor(() => expect(updateObservation).toHaveBeenCalledTimes(2));
    await waitFor(() => {
      expect(screen.queryByRole("textbox", { name: "Edit note" })).not.toBeInTheDocument();
      expect(screen.queryByRole("alert")).not.toBeInTheDocument();
    });
  });

  it("clears a failed observation edit error when the draft is restored and closed", async () => {
    // Given a failed edit whose draft and error remain active.
    vi.mocked(updateObservation).mockRejectedValueOnce(new Error("write failed"));
    const { user } = renderEntity();
    await screen.findByRole("heading", { name: "Ada Lovelace" });
    await user.click(screen.getByRole("button", { name: "Original observation" }));
    const input = screen.getByRole("textbox", { name: "Edit note" });
    await user.clear(input);
    await user.type(input, "Draft survives{Enter}");
    expect(await screen.findAllByRole("alert")).toHaveLength(1);

    // When the user restores the original content and closes the edit with Enter.
    await user.clear(screen.getByRole("textbox", { name: "Edit note" }));
    await user.type(
      screen.getByRole("textbox", { name: "Edit note" }),
      "Original observation{Enter}",
    );

    // Then the canceled edit does not retry and its stale error is cleared.
    expect(updateObservation).toHaveBeenCalledTimes(1);
    expect(screen.queryByRole("textbox", { name: "Edit note" })).not.toBeInTheDocument();
    await waitFor(() => expect(screen.queryByRole("alert")).not.toBeInTheDocument());
  });

  it("keeps the entity visible and does not navigate back when deletion fails", async () => {
    vi.mocked(deleteEntity).mockRejectedValueOnce(new Error("delete failed"));
    const { onBack, user } = renderEntity();
    await screen.findByRole("heading", { name: "Ada Lovelace" });

    await user.click(screen.getByRole("button", { name: "Delete entity" }));
    await user.click(screen.getByRole("button", { name: /^Delete$/ }));

    expect(await screen.findByRole("alert")).toHaveTextContent("Couldn't save. Try again.");
    expect(screen.getByRole("heading", { name: "Ada Lovelace" })).toBeInTheDocument();
    expect(onBack).not.toHaveBeenCalled();
  });

  it("evicts entity-specific caches and refreshes every entity index after deletion", async () => {
    vi.mocked(deleteEntity).mockResolvedValueOnce(undefined);
    const { queryClient, user } = renderEntity();
    const removeSpy = vi.spyOn(queryClient, "removeQueries");
    const invalidateSpy = vi.spyOn(queryClient, "invalidateQueries");
    await screen.findByRole("heading", { name: "Ada Lovelace" });

    await user.click(screen.getByRole("button", { name: "Delete entity" }));
    await user.click(screen.getByRole("button", { name: /^Delete$/ }));

    await waitFor(() => expect(deleteEntity).toHaveBeenCalledWith("entity-ada"));
    expect(removeSpy).toHaveBeenCalledWith({ queryKey: ["entityDetail", "entity-ada"] });
    expect(removeSpy).toHaveBeenCalledWith({
      queryKey: ["entity-linked-memories", "entity-ada"],
    });
    for (const queryKey of [
      "entities",
      "space-entities",
      "constellation-entities",
      "constellation-relations",
      "connections-entities",
      "searchEntities",
    ]) {
      expect(invalidateSpy).toHaveBeenCalledWith({ queryKey: [queryKey] });
    }
  });

  it("blocks repeated observation confirmation while the mutation is pending", async () => {
    let resolveConfirmation: (() => void) | undefined;
    vi.mocked(confirmObservation).mockImplementationOnce(
      () => new Promise<void>((resolve) => {
        resolveConfirmation = resolve;
      }),
    );
    const { user } = renderEntity();
    const confirmButton = await screen.findByRole("button", { name: "Mark note confirmed" });

    await user.click(confirmButton);
    await waitFor(() => expect(confirmButton).toBeDisabled());
    await user.click(confirmButton);
    expect(confirmObservation).toHaveBeenCalledTimes(1);
    resolveConfirmation?.();
    await waitFor(() => expect(confirmButton).not.toBeDisabled());
  });
});
