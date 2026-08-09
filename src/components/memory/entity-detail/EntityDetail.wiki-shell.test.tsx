// SPDX-License-Identifier: AGPL-3.0-only
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen, within } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { i18n } from "../../../i18n";
import type { EntityDetail as EntityDetailRecord } from "../../../lib/tauri";
import EntityDetail from "../EntityDetail";

vi.mock("../../../lib/tauri", () => ({
  getEntityDetail: vi.fn(),
  updateObservation: vi.fn().mockResolvedValue(undefined),
  deleteObservation: vi.fn().mockResolvedValue(undefined),
  addObservation: vi.fn().mockResolvedValue(undefined),
  confirmObservation: vi.fn().mockResolvedValue(undefined),
  confirmEntity: vi.fn().mockResolvedValue(undefined),
  deleteEntity: vi.fn().mockResolvedValue(undefined),
  search: vi.fn().mockResolvedValue([]),
  FACET_COLORS: {},
}));

import { getEntityDetail } from "../../../lib/tauri";

const detail: EntityDetailRecord = {
  entity: {
    id: "entity-ada",
    name: "Ada Lovelace",
    entity_type: "person",
    domain: "computing",
    space: "History of Computing",
    source_agent: "research-agent",
    confidence: 0.87,
    confirmed: true,
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
      confirmed: true,
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
  ],
};

function renderEntity(record: EntityDetailRecord = detail) {
  vi.mocked(getEntityDetail).mockResolvedValue(record);
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: { retry: false, gcTime: 0, refetchOnWindowFocus: false },
      mutations: { retry: false },
    },
  });
  return render(
    <QueryClientProvider client={queryClient}>
      <EntityDetail
        entityId={record.entity.id}
        onBack={vi.fn()}
        onEntityClick={vi.fn()}
        onMemoryClick={vi.fn()}
      />
    </QueryClientProvider>,
  );
}

describe("EntityDetail wiki shell", () => {
  beforeEach(async () => {
    vi.clearAllMocks();
    await i18n.changeLanguage("en");
  });

  it("composes the entity dossier from the shared page root, grid, prose, and rail", async () => {
    const { container } = renderEntity();
    await screen.findByRole("heading", { name: "Ada Lovelace" });

    expect(container.firstElementChild).toHaveClass("page-detail", "entity-detail-dossier");
    expect(container.querySelector(".page-detail-grid")).toBeInTheDocument();
    expect(container.querySelector(".page-detail-prose")).toBeInTheDocument();
    expect(container.querySelector(".page-detail-rail")).toBeInTheDocument();
  });

  it("uses one primary heading and keeps the sage entity identity visible", async () => {
    renderEntity();
    const title = await screen.findByRole("heading", { level: 1, name: "Ada Lovelace" });

    expect(screen.getAllByRole("heading", { level: 1 })).toHaveLength(1);
    expect(title).toHaveClass("page-detail-title");
    expect(screen.getByText("AL")).toHaveClass("entity-detail-seal");
    expect(screen.getAllByText("person").length).toBeGreaterThan(0);
  });

  it("places Connections before About in DOM order", async () => {
    renderEntity();
    const connections = await screen.findByRole("heading", { name: "Connections" });
    const about = screen.getByRole("heading", { name: "About" });

    expect(
      connections.compareDocumentPosition(about) & Node.DOCUMENT_POSITION_FOLLOWING,
    ).toBeTruthy();
  });

  it("labels both the graph and raw relation ledger and leaves their controls tabbable", async () => {
    renderEntity();
    const graph = await screen.findByRole("group", { name: "Connection map for Ada Lovelace" });
    const ledger = screen.getByRole("group", { name: "Connections" });
    const graphControl = within(graph).getByRole("button", { name: /Charles Babbage/ });
    const ledgerControl = within(ledger).getByRole("button", { name: /Charles Babbage/ });

    expect(graphControl.tabIndex).toBe(0);
    expect(ledgerControl.tabIndex).toBe(0);
    expect(ledgerControl).toHaveAccessibleName(
      "Charles Babbage (person) · outgoing · collaborated with",
    );
  });

  it("renders empty relationships and malformed numeric metadata without invalid text", async () => {
    renderEntity({
      entity: {
        ...detail.entity,
        confidence: Number.NaN,
        created_at: Number.MAX_VALUE,
        updated_at: Number.NEGATIVE_INFINITY,
      },
      observations: [{ ...detail.observations[0], confidence: Number.POSITIVE_INFINITY }],
      relations: [],
    });

    expect(await screen.findByText("No connections recorded yet.")).toBeInTheDocument();
    expect(screen.queryAllByText(/NaN|Infinity|Invalid Date/)).toHaveLength(0);
  });
});
