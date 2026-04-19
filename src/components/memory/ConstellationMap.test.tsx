// SPDX-License-Identifier: AGPL-3.0-only
import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";

// jsdom does not provide ResizeObserver — stub it
class ResizeObserverStub {
  observe() {}
  unobserve() {}
  disconnect() {}
}
vi.stubGlobal("ResizeObserver", ResizeObserverStub);

vi.mock("../../lib/tauri", () => ({
  listEntities: vi.fn(),
  getEntityDetail: vi.fn(),
}));

// react-force-graph-2d requires canvas — mock it in jsdom
vi.mock("react-force-graph-2d", () => ({
  __esModule: true,
  default: vi.fn(({ graphData }: any) => (
    <div data-testid="force-graph">
      {graphData?.nodes?.map((n: any) => (
        <span key={n.id} className="constellation-node">{n.name}</span>
      ))}
    </div>
  )),
}));

import { listEntities, getEntityDetail } from "../../lib/tauri";
import ConstellationMap from "./ConstellationMap";

const mockListEntities = vi.mocked(listEntities);
const mockGetEntityDetail = vi.mocked(getEntityDetail);

function renderWithQuery(ui: React.ReactElement) {
  const qc = new QueryClient({
    defaultOptions: { queries: { retry: false, gcTime: 0 } },
  });
  return render(<QueryClientProvider client={qc}>{ui}</QueryClientProvider>);
}

function makeEntity(overrides: Partial<import("../../lib/tauri").Entity> = {}): import("../../lib/tauri").Entity {
  return {
    id: overrides.id ?? "e1",
    name: overrides.name ?? "Entity",
    entity_type: overrides.entity_type ?? "concept",
    domain: overrides.domain ?? null,
    source_agent: overrides.source_agent ?? null,
    confidence: overrides.confidence ?? null,
    confirmed: overrides.confirmed ?? false,
    created_at: overrides.created_at ?? Date.now(),
    updated_at: overrides.updated_at ?? Date.now(),
  };
}

describe("ConstellationMap", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("renders the force graph when entities exist", async () => {
    const entities = [
      makeEntity({ id: "e1", name: "Alice", entity_type: "person" }),
      makeEntity({ id: "e2", name: "Origin", entity_type: "project" }),
    ];
    mockListEntities.mockResolvedValue(entities);
    mockGetEntityDetail.mockResolvedValue({
      entity: entities[0],
      observations: [],
      relations: [],
    });

    renderWithQuery(<ConstellationMap />);

    expect(await screen.findByTestId("force-graph")).toBeInTheDocument();
  });

  it("renders a circle with class constellation-node for each entity", async () => {
    const entities = [
      makeEntity({ id: "e1", name: "Alice", entity_type: "person" }),
      makeEntity({ id: "e2", name: "Origin", entity_type: "project" }),
      makeEntity({ id: "e3", name: "Memory", entity_type: "concept" }),
    ];
    mockListEntities.mockResolvedValue(entities);
    mockGetEntityDetail.mockImplementation(async (id) => ({
      entity: entities.find((e) => e.id === id) ?? entities[0],
      observations: [],
      relations: [],
    }));

    const { container } = renderWithQuery(<ConstellationMap />);

    await screen.findByText("3 entities");

    const nodes = container.querySelectorAll(".constellation-node");
    expect(nodes).toHaveLength(3);
  });

  it("shows entity count label", async () => {
    const entities = [
      makeEntity({ id: "e1", name: "A" }),
      makeEntity({ id: "e2", name: "B" }),
      makeEntity({ id: "e3", name: "C" }),
    ];
    mockListEntities.mockResolvedValue(entities);
    mockGetEntityDetail.mockResolvedValue({
      entity: entities[0],
      observations: [],
      relations: [],
    });

    renderWithQuery(<ConstellationMap />);

    expect(await screen.findByText("3 entities")).toBeInTheDocument();
  });

  it("renders empty state message when no entities", async () => {
    mockListEntities.mockResolvedValue([]);

    renderWithQuery(<ConstellationMap />);

    expect(
      await screen.findByText("Your constellation will appear as knowledge grows"),
    ).toBeInTheDocument();
  });
});
