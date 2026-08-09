// SPDX-License-Identifier: AGPL-3.0-only
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { i18n } from "../../i18n";
import type { EntityDetail as EntityDetailRecord, SearchResult } from "../../lib/tauri";
import EntityDetail from "./EntityDetail";

vi.mock("../../lib/tauri", () => ({
  getEntityDetail: vi.fn(),
  updateObservation: vi.fn().mockResolvedValue(undefined),
  deleteObservation: vi.fn().mockResolvedValue(undefined),
  addObservation: vi.fn().mockResolvedValue(undefined),
  confirmObservation: vi.fn().mockResolvedValue(undefined),
  confirmEntity: vi.fn().mockResolvedValue(undefined),
  deleteEntity: vi.fn().mockResolvedValue(undefined),
  search: vi.fn(),
  FACET_COLORS: { fact: "facet-fact" },
}));

vi.mock("./FocusGraph", () => ({
  default: (props: { onEntityClick?: (id: string) => void }) => (
    <div data-testid="focus-graph">
      <button type="button" onClick={() => props.onEntityClick?.("entity-babbage")}>
        Focus node
      </button>
    </div>
  ),
}));

vi.mock("./AtlasView", () => ({
  default: (props: {
    focusEntityId?: string;
    onNodeClick?: (id: string) => void;
  }) => (
    <div data-focus-entity-id={props.focusEntityId ?? ""} data-testid="atlas-view">
      <button type="button" onClick={() => props.onNodeClick?.("entity-hopper")}>
        Atlas node
      </button>
    </div>
  ),
}));

import {
  addObservation,
  confirmEntity,
  confirmObservation,
  deleteEntity,
  deleteObservation,
  getEntityDetail,
  search,
  updateObservation,
} from "../../lib/tauri";

const detail: EntityDetailRecord = {
  entity: {
    id: "entity-ada",
    name: "Ada Lovelace",
    entity_type: "person",
    domain: "computing",
    space: "History of Computing",
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
      content: "Wrote the first published algorithm",
      source_agent: "research-agent",
      confidence: 0.8,
      confirmed: false,
      created_at: 1_700_000_100,
    },
  ],
  relations: [
    {
      id: "relation-1",
      relation_type: "collaborated with",
      direction: "outgoing",
      entity_id: "entity-babbage",
      entity_name: "Charles Babbage",
      entity_type: "person",
      source_agent: "research-agent",
      created_at: 1_700_000_200,
    },
    {
      id: "relation-2",
      relation_type: "inspired",
      direction: "incoming",
      entity_id: "entity-hopper",
      entity_name: "Grace Hopper",
      entity_type: "person",
      source_agent: null,
      created_at: 1_700_000_300,
    },
  ],
};

const linkedMemory: SearchResult = {
  id: "chunk-1",
  content: "Ada's notes described a general-purpose machine.",
  source: "notes.md",
  source_id: "memory-ada",
  title: "Ada's notes",
  url: null,
  chunk_index: 0,
  last_modified: 1_700_086_500,
  score: 0.94,
  memory_type: "fact",
  entity_id: "entity-ada",
  is_archived: false,
};

const defaultProps = {
  entityId: "entity-ada",
  onBack: vi.fn(),
  onEntityClick: vi.fn(),
  onMemoryClick: vi.fn(),
};

function renderEntity(props = defaultProps) {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: { retry: false, gcTime: 0 },
      mutations: { retry: false },
    },
  });
  return {
    user: userEvent.setup(),
    ...render(
      <QueryClientProvider client={queryClient}>
        <EntityDetail {...props} />
      </QueryClientProvider>,
    ),
  };
}

describe("EntityDetail characterization", () => {
  beforeEach(async () => {
    vi.clearAllMocks();
    await i18n.changeLanguage("en");
    vi.mocked(getEntityDetail).mockResolvedValue(detail);
    vi.mocked(search).mockResolvedValue([linkedMemory]);
  });

  it("loads the entity record with identity, metadata, observations, relations, and linked memories", async () => {
    renderEntity();

    expect(await screen.findByRole("heading", { name: "Ada Lovelace" })).toBeInTheDocument();
    expect(getEntityDetail).toHaveBeenCalledWith("entity-ada");
    expect(screen.getAllByText("person").length).toBeGreaterThan(0);
    expect(screen.getAllByText("History of Computing").length).toBeGreaterThan(0);
    expect(screen.getByText("research-agent")).toBeInTheDocument();
    expect(screen.getByText("0.87")).toBeInTheDocument();
    expect(screen.getByText("Wrote the first published algorithm")).toBeInTheDocument();
    expect(screen.getAllByText("Charles Babbage").length).toBeGreaterThan(0);
    expect(
      await screen.findByText("Ada's notes described a general-purpose machine."),
    ).toBeInTheDocument();
  });

  it("exposes back navigation and recovers from a load error", async () => {
    vi.mocked(getEntityDetail)
      .mockRejectedValueOnce(new Error("offline"))
      .mockResolvedValue(detail);
    const { user } = renderEntity();

    await user.click(screen.getByRole("button", { name: "Back" }));
    expect(defaultProps.onBack).toHaveBeenCalledOnce();
    expect(await screen.findByText("Couldn't load this entity.")).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "Retry" }));
    expect(await screen.findByRole("heading", { name: "Ada Lovelace" })).toBeInTheDocument();
    expect(getEntityDetail).toHaveBeenCalledTimes(2);
  });

  it("confirms the entity and an observation", async () => {
    const { user } = renderEntity();

    await user.click(await screen.findByRole("button", { name: "Confirm entity" }));
    await user.click(screen.getByRole("button", { name: "Mark note confirmed" }));

    await waitFor(() => {
      expect(confirmEntity).toHaveBeenCalledWith("entity-ada", true);
      expect(confirmObservation).toHaveBeenCalledWith("obs-1", true);
    });
  });

  it("adds, edits, cancels, and deletes observations without leaking Escape to back", async () => {
    const { user } = renderEntity();
    await screen.findByRole("heading", { name: "Ada Lovelace" });

    await user.click(screen.getByRole("button", { name: "Add note" }));
    const addInput = screen.getByRole("textbox", { name: "Add note" });
    await user.type(addInput, "A precise new observation{Enter}");
    await waitFor(() => {
      expect(addObservation).toHaveBeenCalledWith(
        "entity-ada",
        "A precise new observation",
        "human",
        1,
      );
    });

    await user.click(screen.getByRole("button", { name: /Wrote the first published algorithm/ }));
    const editInput = screen.getByRole("textbox", { name: "Edit note" });
    await user.clear(editInput);
    await user.type(editInput, "Edited observation{Enter}");
    await waitFor(() => {
      expect(updateObservation).toHaveBeenCalledWith("obs-1", "Edited observation");
    });

    await user.click(screen.getByRole("button", { name: /Wrote the first published algorithm/ }));
    await user.type(screen.getByRole("textbox", { name: "Edit note" }), " ignored{Escape}");
    expect(defaultProps.onBack).not.toHaveBeenCalled();

    await user.click(screen.getByRole("button", { name: "Delete note" }));
    await waitFor(() => expect(deleteObservation).toHaveBeenCalledWith("obs-1"));
  });

  it("routes graph, ledger, and linked-memory controls through their existing callbacks", async () => {
    const { user } = renderEntity();
    await screen.findByRole("heading", { name: "Ada Lovelace" });

    const charlesControls = screen.getAllByRole("button", { name: /Charles Babbage/ });
    expect(charlesControls).toHaveLength(2);
    await user.click(charlesControls[0]);
    await user.click(charlesControls[1]);
    await user.click(
      await screen.findByRole("button", {
        name: /Ada's notes described a general-purpose machine/,
      }),
    );

    expect(defaultProps.onEntityClick).toHaveBeenNthCalledWith(1, "entity-babbage");
    expect(defaultProps.onEntityClick).toHaveBeenNthCalledWith(2, "entity-babbage");
    expect(defaultProps.onMemoryClick).toHaveBeenCalledWith("memory-ada");
  });

  it("opens the Focus graph overlay and closes it on Escape", async () => {
    const { user } = renderEntity();
    await screen.findByRole("heading", { name: "Ada Lovelace" });

    await user.click(screen.getByRole("button", { name: "Full screen" }));

    expect(screen.getByRole("dialog", { name: "Full screen" })).toBeInTheDocument();
    expect(screen.getByTestId("focus-graph")).toBeInTheDocument();

    await user.keyboard("{Escape}");
    expect(screen.queryByRole("dialog", { name: "Full screen" })).not.toBeInTheDocument();
  });

  it("switches the overlay to Atlas with the current entity focused", async () => {
    const { user } = renderEntity();
    await screen.findByRole("heading", { name: "Ada Lovelace" });
    await user.click(screen.getByRole("button", { name: "Full screen" }));
    const dialog = screen.getByRole("dialog", { name: "Full screen" });

    await user.click(within(dialog).getByRole("button", { name: "Atlas" }));
    expect(await screen.findByTestId("atlas-view")).toHaveAttribute(
      "data-focus-entity-id",
      "entity-ada",
    );

    await user.click(within(dialog).getByRole("button", { name: "Atlas node" }));
    expect(defaultProps.onEntityClick).toHaveBeenCalledWith("entity-hopper");
    expect(screen.queryByRole("dialog", { name: "Full screen" })).not.toBeInTheDocument();
  });

  it("uses a two-step entity delete and returns after success", async () => {
    const { user } = renderEntity();
    await screen.findByRole("heading", { name: "Ada Lovelace" });

    await user.click(screen.getByRole("button", { name: "Delete entity" }));
    expect(deleteEntity).not.toHaveBeenCalled();
    await user.click(screen.getByRole("button", { name: /^Delete$/ }));

    await waitFor(() => {
      expect(deleteEntity).toHaveBeenCalledWith("entity-ada");
      expect(defaultProps.onBack).toHaveBeenCalledOnce();
    });
  });
});
