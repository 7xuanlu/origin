// SPDX-License-Identifier: AGPL-3.0-only
import { describe, it, expect, vi, beforeEach } from "vitest";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { i18n } from "../../i18n";

vi.mock("../../lib/tauri", async () => {
  const actual = await vi.importActual<typeof import("../../lib/tauri")>("../../lib/tauri");
  return {
    ...actual,
    getKnowledgeGraph: vi.fn(),
    listCommunities: vi.fn(),
    listCommunityMembers: vi.fn(),
  };
});

// jsdom has no WebGL context — a real Sigma would throw trying to acquire
// one. Mocked at module level per repo convention for canvas/WebGL surfaces;
// the drawn graph itself is verified live in preview. This test only proves
// the three states, retry, mount/teardown, and the click handoff.
const capturedSigmaInstances = vi.hoisted(() => [] as any[]);
/** The mock viewport. Tests that watch the names of a far component widen
 *  it (see graphToViewport below). */
let mockDimensions = { width: 400, height: 600 };
vi.mock("sigma", () => {
  class MouseCaptorMock {
    handlers = new Map<string, (payload: any) => void>();
    on(event: string, handler: (payload: any) => void) {
      this.handlers.set(event, handler);
      return this;
    }
  }
  class SigmaMock {
    handlers = new Map<string, (payload: any) => void>();
    mouseCaptor = new MouseCaptorMock();
    customBBox: unknown = null;
    constructor(
      public graph: any,
      public container: any,
      public settings: any,
    ) {
      capturedSigmaInstances.push(this);
    }
    on(event: string, handler: (payload: any) => void) {
      this.handlers.set(event, handler);
      return this;
    }
    getMouseCaptor() {
      return this.mouseCaptor;
    }
    // 100-unit graph span in a 400px-min-dim container: the fill-aware
    // default density is 0.6 * 400 / 100 = 2.4 px/unit (inside the [1.5, 3]
    // clamp), so the mount zoom-out must be exactly 6 / 2.4 = 2.5.
    getBBox() {
      return { x: [-50, 50], y: [-40, 40] };
    }
    getDimensions() {
      return mockDimensions;
    }
    getCustomBBox() {
      return this.customBBox;
    }
    setCustomBBox(bbox: unknown) {
      this.customBBox = bbox;
    }
    viewportToGraph(coords: { x: number; y: number }) {
      return coords;
    }
    // Fake fit density: 6 px per graph unit, so the density cap (target 1.5)
    // must zoom out by exactly 4x. Centred on the middle of the mock
    // viewport, as the real projection is, so a node near the graph origin
    // lands ON screen — the place-name pass culls names whose box falls
    // outside the viewport. At 6 px/unit a wing component (60+ graph units
    // out) projects past the default 400x600; tests that watch its name
    // widen mockDimensions instead, the way a real fit keeps the whole map
    // in view.
    graphToViewport(coords: { x: number; y: number }) {
      return {
        x: mockDimensions.width / 2 + coords.x * 6,
        y: mockDimensions.height / 2 + coords.y * 6,
      };
    }
    camera = {
      ratio: 1,
      setState: vi.fn(),
      animate: vi.fn(),
      getBoundedRatio: (r: number) => r,
      // The zoom level-of-detail listener (AtlasView mount) subscribes here.
      on: vi.fn(),
    };
    getCamera() {
      return this.camera;
    }
    // Fixed display coords — enough for the search-fly tests to pin that the
    // camera target comes from getNodeDisplayData, not raw graph coords.
    getNodeDisplayData(_node: string) {
      return { x: 0.42, y: 0.24 };
    }
    getViewportZoomedState = vi.fn((pos: { x: number; y: number }, ratio: number) => ({
      x: pos.x,
      y: pos.y,
      ratio,
    }));
    // Real sigma renders synchronously on refresh() and fires afterRender at
    // the end of the frame. The overlay repaint hangs off that event, so a
    // no-op here would silently pass any "it repainted" assertion.
    refresh() {
      this.handlers.get("afterRender")?.({});
    }
    setSetting(_key: string, _value: unknown) {}
    kill() {}
  }
  return { default: SigmaMock };
});

import { getKnowledgeGraph, listCommunities, listCommunityMembers } from "../../lib/tauri";
import type {
  Entity,
  EntityDetail,
  GraphMemoryLink,
  GraphMemoryNode,
  GraphPageLink,
  GraphPageNode,
  GraphRelation,
  KnowledgeGraph,
} from "../../lib/tauri";
import AtlasView from "./AtlasView";

const mockGetKnowledgeGraph = vi.mocked(getKnowledgeGraph);
const mockListCommunities = vi.mocked(listCommunities);
const mockListCommunityMembers = vi.mocked(listCommunityMembers);

// The view makes ONE bulk read now. These two stand-ins keep the fixtures in
// the entities-plus-details shape the per-entity routes used to serve, and
// fold them into that single response — so a test still declares a graph
// shape rather than a wire payload, and the fold enforces the bulk read's own
// contract (a relation whose other endpoint is out of scope never ships).
let entitiesSource: () => Promise<Entity[]>;
let detailSource: (id: string) => Promise<EntityDetail>;
let memorySource: { memories: GraphMemoryNode[]; memory_links: GraphMemoryLink[] };
let pageSource: { pages: GraphPageNode[]; page_links: GraphPageLink[] };

const mockListEntities = {
  mockResolvedValue(entities: Entity[]) {
    entitiesSource = async () => entities;
  },
  mockReturnValue(promise: Promise<Entity[]>) {
    entitiesSource = () => promise;
  },
  mockResolvedValueOnce(entities: Entity[]) {
    const previous = entitiesSource;
    let used = false;
    entitiesSource = async () => {
      if (used) return previous();
      used = true;
      return entities;
    };
  },
  mockRejectedValueOnce(error: Error) {
    const previous = entitiesSource;
    let used = false;
    entitiesSource = async () => {
      if (used) return previous();
      used = true;
      throw error;
    };
  },
};

const mockGetEntityDetail = {
  mockResolvedValue(detail: EntityDetail) {
    detailSource = async () => detail;
  },
  mockImplementation(provider: (id: string) => Promise<EntityDetail>) {
    detailSource = provider;
  },
};

/** Memory nodes and their entity links for the next bulk read. */
function mockMemories(memories: GraphMemoryNode[], links: GraphMemoryLink[]) {
  memorySource = { memories, memory_links: links };
}

/** Wiki-page nodes and their typed links for the next bulk read. */
function mockPages(pages: GraphPageNode[], links: GraphPageLink[]) {
  pageSource = { pages, page_links: links };
}

function makePage(overrides: Partial<GraphPageNode> = {}): GraphPageNode {
  return {
    id: overrides.id ?? "p1",
    title: overrides.title ?? "Page",
    space: overrides.space ?? null,
    creation_kind: overrides.creation_kind ?? "distilled",
    entity_id: overrides.entity_id ?? null,
    last_modified: overrides.last_modified ?? "2026-08-15T00:00:00Z",
  };
}

/** Turn the memory layer on before the first render. Memories ship OFF by
 *  default, so a case that is about memory nodes has to ask for them. */
function withMemoryLayerOn() {
  window.localStorage.setItem(
    "atlas.layers",
    JSON.stringify({ page: true, entity: true, memory: true }),
  );
}

function makeMemory(overrides: Partial<GraphMemoryNode> = {}): GraphMemoryNode {
  return {
    source_id: overrides.source_id ?? "m1",
    title: overrides.title ?? "Memory",
    memory_type: overrides.memory_type ?? "fact",
    space: overrides.space ?? null,
    confirmed: overrides.confirmed ?? true,
    last_modified: overrides.last_modified ?? Math.floor(Date.now() / 1000),
  };
}

function foldGraph(entities: Entity[], details: EntityDetail[]): KnowledgeGraph {
  const known = new Set(entities.map((entity) => entity.id));
  const relations = new Map<string, GraphRelation>();
  for (const detail of details) {
    const home = detail.entity.id;
    if (!known.has(home)) continue;
    for (const relation of detail.relations) {
      // Both endpoints in scope or the relation does not exist — the daemon
      // never ships a dangling one, so neither does this fixture.
      if (!known.has(relation.entity_id)) continue;
      const incoming = relation.direction === "incoming";
      const from = incoming ? relation.entity_id : home;
      const to = incoming ? home : relation.entity_id;
      const id = relation.id || `${from}:${relation.relation_type}:${to}`;
      if (relations.has(id)) continue;
      relations.set(id, {
        id,
        from_entity: from,
        to_entity: to,
        relation_type: relation.relation_type,
        source_agent: relation.source_agent,
        created_at: relation.created_at,
      });
    }
  }
  return {
    entities,
    relations: Array.from(relations.values()),
    memories: memorySource.memories,
    memory_links: memorySource.memory_links,
    pages: pageSource.pages,
    page_links: pageSource.page_links,
  };
}

function renderWithQuery(ui: React.ReactElement) {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false, gcTime: 0 } } });
  return { ...render(<QueryClientProvider client={qc}>{ui}</QueryClientProvider>), qc };
}

function makeEntity(overrides: Partial<import("../../lib/tauri").Entity> = {}): import("../../lib/tauri").Entity {
  return {
    id: overrides.id ?? "e1",
    name: overrides.name ?? "Entity",
    entity_type: overrides.entity_type ?? "concept",
    domain: overrides.domain ?? null,
    // Kept distinct from domain: an entity can carry space: null or the empty
    // string while domain says nothing, and all of those mean "unscoped".
    space: overrides.space ?? null,
    source_agent: overrides.source_agent ?? null,
    confidence: overrides.confidence ?? null,
    confirmed: overrides.confirmed ?? false,
    created_at: overrides.created_at ?? Math.floor(Date.now() / 1000),
    updated_at: overrides.updated_at ?? Date.now(),
  };
}

describe("AtlasView", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    capturedSigmaInstances.length = 0;
    mockDimensions = { width: 400, height: 600 };
    entitiesSource = async () => [];
    detailSource = async (id: string) => ({
      entity: makeEntity({ id }),
      observations: [],
      relations: [],
    });
    memorySource = { memories: [], memory_links: [] };
    pageSource = { pages: [], page_links: [] };
    window.localStorage.removeItem("atlas.layers");
    window.localStorage.removeItem("atlas.smallGroups");
    mockGetKnowledgeGraph.mockImplementation(async () => {
      const entities = await entitiesSource();
      const details = await Promise.all(entities.map((entity) => detailSource(entity.id)));
      return foldGraph(entities, details);
    });
    // Ordinary "no durable data published yet" default — most tests carry
    // no space at all (so this never fires), and the ones that do only care
    // about entities/relations, not cartography.
    mockListCommunities.mockResolvedValue({
      schema_version: "community-read-v1",
      communities: [],
      next_cursor: null,
    });
    mockListCommunityMembers.mockResolvedValue({
      schema_version: "community-read-v1",
      members: [],
      next_cursor: null,
    });
  });

  it("shows a distinct loading state while entities are in flight, then resolves to empty", async () => {
    let resolveEntities!: (value: import("../../lib/tauri").Entity[]) => void;
    mockListEntities.mockReturnValue(
      new Promise((resolve) => {
        resolveEntities = resolve;
      }),
    );

    renderWithQuery(<AtlasView />);

    expect(await screen.findByText("Loading your knowledge graph…")).toBeInTheDocument();
    expect(screen.queryByText("Your constellation will appear as knowledge grows")).not.toBeInTheDocument();
    expect(capturedSigmaInstances).toHaveLength(0);

    resolveEntities([]);

    expect(
      await screen.findByText("Your constellation will appear as knowledge grows"),
    ).toBeInTheDocument();
    expect(screen.queryByText("Loading your knowledge graph…")).not.toBeInTheDocument();
    expect(capturedSigmaInstances).toHaveLength(0);
  });

  it("renders the empty state when there are no entities", async () => {
    mockListEntities.mockResolvedValue([]);

    renderWithQuery(<AtlasView />);

    expect(
      await screen.findByText("Your constellation will appear as knowledge grows"),
    ).toBeInTheDocument();
    expect(capturedSigmaInstances).toHaveLength(0);
  });

  it("shows an error panel with retry on query failure, distinct from empty, and retry recovers", async () => {
    mockListEntities.mockRejectedValueOnce(new Error("daemon unreachable"));

    renderWithQuery(<AtlasView />);

    expect(await screen.findByText("Couldn't load your knowledge graph.")).toBeInTheDocument();
    expect(screen.queryByText("Your constellation will appear as knowledge grows")).not.toBeInTheDocument();
    expect(capturedSigmaInstances).toHaveLength(0);

    const entities = [makeEntity({ id: "e1", name: "Alice" })];
    mockListEntities.mockResolvedValueOnce(entities);
    mockGetEntityDetail.mockResolvedValue({ entity: entities[0], observations: [], relations: [] });

    screen.getByRole("button", { name: "Retry" }).click();

    await waitFor(() => expect(capturedSigmaInstances).toHaveLength(1));
  });

  it("mounts a sigma renderer once entities and details resolve, and tears it down on unmount", async () => {
    const entities = [makeEntity({ id: "e1", name: "Alice", entity_type: "person" })];
    mockListEntities.mockResolvedValue(entities);
    mockGetEntityDetail.mockResolvedValue({ entity: entities[0], observations: [], relations: [] });

    const { unmount } = renderWithQuery(<AtlasView />);

    await waitFor(() => expect(capturedSigmaInstances).toHaveLength(1));
    const killSpy = vi.spyOn(capturedSigmaInstances[0], "kill");

    unmount();

    expect(killSpy).toHaveBeenCalledTimes(1);
  });

  it("fires onNodeClick with the sigma node id on clickNode", async () => {
    // A drawn node needs an edge now that degree-0 nodes are hidden, so this
    // interacts with the connected pair rather than a lone entity.
    mockConnectedPair();
    const onNodeClick = vi.fn();

    renderWithQuery(<AtlasView onNodeClick={onNodeClick} />);
    await waitFor(() => expect(capturedSigmaInstances).toHaveLength(1));

    capturedSigmaInstances[0].handlers.get("clickNode")?.({ node: "e1" });

    expect(onNodeClick).toHaveBeenCalledWith({ kind: "entity", id: "e1" });
  });

  it("wires nodeReducer and edgeReducer functions into the sigma constructor settings", async () => {
    // A drawn node needs an edge now that degree-0 nodes are hidden, so this
    // interacts with the connected pair rather than a lone entity.
    mockConnectedPair();

    renderWithQuery(<AtlasView />);
    await waitFor(() => expect(capturedSigmaInstances).toHaveLength(1));

    const { settings } = capturedSigmaInstances[0];
    expect(typeof settings.nodeReducer).toBe("function");
    expect(typeof settings.edgeReducer).toBe("function");
    // Sigma's built-in hover renderer hardcodes a #FFF label box that's
    // unreadable under the dark theme's label ink; ours is the same painter
    // the labels layer uses, so a lifted (highlighted) name draws on the
    // hover layer above every other label, and a blank label paints nothing.
    expect(typeof settings.defaultDrawNodeHover).toBe("function");
    expect(settings.defaultDrawNodeHover).toBe(settings.defaultDrawNodeLabel);
    const ctx = { fillText: vi.fn(), strokeText: vi.fn(), measureText: vi.fn(() => ({ width: 0 })) };
    settings.defaultDrawNodeHover(ctx, { key: "e1", label: "", x: 0, y: 0, size: 4 }, {});
    expect(ctx.fillText).not.toHaveBeenCalled();
  });

  it("wires the radial label drawer over the live graph and lowers sigma's edge-thickness floor", async () => {
    // A drawn node needs an edge now that degree-0 nodes are hidden, so this
    // interacts with the connected pair rather than a lone entity.
    mockConnectedPair();
    // jsdom resolves no theme tokens; give the palette a ground colour so the
    // label halo (stroked in --mem-surface) has an ink to stroke with.
    document.documentElement.style.setProperty("--mem-surface", "#11131a");

    renderWithQuery(<AtlasView />);
    await waitFor(() => expect(capturedSigmaInstances).toHaveLength(1));

    const { settings } = capturedSigmaInstances[0];
    // Edges are 1 CSS px (0.6 shared-source) — sigma's default floor (1.7)
    // would bump them up.
    expect(settings.minEdgeThickness).toBe(0.5);

    // Drives the real drawRadialNodeLabel through the graph closure: the
    // drawer reads e1's graph position for the sector, so this throws if the
    // wiring doesn't reach the mounted graph.
    const order: string[] = [];
    const ctx = {
      font: "",
      fillStyle: "",
      strokeStyle: "",
      lineWidth: 0,
      lineJoin: "",
      globalAlpha: 1,
      textAlign: "",
      textBaseline: "",
      strokeText: vi.fn(() => order.push("stroke")),
      fillText: vi.fn(() => order.push("fill")),
    };
    settings.defaultDrawNodeLabel(
      ctx,
      { key: "e1", label: "Alice", size: 4, x: 10, y: 20 },
      { labelColor: { color: "#123456" } },
    );
    expect(ctx.fillText).toHaveBeenCalledWith("Alice", expect.any(Number), expect.any(Number));
    expect(ctx.font).toBe('12px "Instrument Sans", -apple-system, sans-serif');
    expect(ctx.fillStyle).toBe("#123456");
    // Ground-coloured halo under the name, so labels stay legible over
    // neighbouring nodes: the stroke lands before the fill.
    expect(ctx.strokeText).toHaveBeenCalledWith("Alice", expect.any(Number), expect.any(Number));
    expect(ctx.lineWidth).toBe(3);
    expect(ctx.strokeStyle).toBe("#11131a");
    expect(order).toEqual(["stroke", "fill"]);
    document.documentElement.style.removeProperty("--mem-surface");
  });

  it("renders the legend chips, connection sample, and count line over the graph", async () => {
    mockConnectedPair();

    renderWithQuery(<AtlasView />);
    await waitFor(() => expect(capturedSigmaInstances).toHaveLength(1));

    for (const label of ["Project", "Technology", "Organization", "Person", "Theme", "Connection"]) {
      expect(screen.getByText(label)).toBeInTheDocument();
    }
    // Alice, Bob and Carol form one community of three, so the toolbar count
    // line reports the entities and the single region (artifact format).
    expect(screen.getByText("3 entities · 1 region")).toBeInTheDocument();
    // Map affordance hint, bottom-left.
    expect(screen.getByText("scroll to zoom · more labels appear as you approach")).toBeInTheDocument();
  });

  it("refreshes on enterNode and again on leaveNode", async () => {
    // A drawn node needs an edge now that degree-0 nodes are hidden, so this
    // interacts with the connected pair rather than a lone entity.
    mockConnectedPair();

    renderWithQuery(<AtlasView />);
    await waitFor(() => expect(capturedSigmaInstances).toHaveLength(1));
    const instance = capturedSigmaInstances[0];
    const refreshSpy = vi.spyOn(instance, "refresh");

    instance.handlers.get("enterNode")?.({ node: "e1" });
    expect(refreshSpy).toHaveBeenCalledTimes(1);

    instance.handlers.get("leaveNode")?.({});
    expect(refreshSpy).toHaveBeenCalledTimes(2);
  });

  it("releases fx/fy for every node on mouseup, pinned or not", async () => {
    // Every component is live now (see atlas.ts's groupCenterForce): its own
    // anchor holds it at its shelf slot, so mouseup no longer has to leave a
    // shelved node pinned to keep the shelf out of the core — it always
    // clears fx/fy, the same as a core drag always did.
    mockConnectedPair();

    renderWithQuery(<AtlasView />);
    await waitFor(() => expect(capturedSigmaInstances).toHaveLength(1));
    const instance = capturedSigmaInstances[0];
    const mouseCaptor = instance.getMouseCaptor();
    const sim = (window as any).__ATLAS_SIM;
    const pinned = sim.nodes().find((n: { id: string }) => n.id === "e1");
    pinned.fx = pinned.x;
    pinned.fy = pinned.y;

    instance.handlers.get("downNode")?.({ node: "e1" });
    mouseCaptor.handlers.get("mousemovebody")?.(dragEvent(300, 300));
    mouseCaptor.handlers.get("mouseup")?.({});
    expect(pinned.fx).toBeNull();
    expect(pinned.fy).toBeNull();

    const free = sim.nodes().find((n: { id: string }) => n.id === "e2");
    instance.handlers.get("downNode")?.({ node: "e2" });
    mouseCaptor.handlers.get("mousemovebody")?.(dragEvent(120, 120));
    mouseCaptor.handlers.get("mouseup")?.({});
    expect(free.fx).toBeNull();
    expect(free.fy).toBeNull();
  });

  it("locks hover while a drag is live — enter/leave neither repaint nor flip the cursor until release", async () => {
    mockConnectedPair();

    renderWithQuery(<AtlasView />);
    await waitFor(() => expect(capturedSigmaInstances).toHaveLength(1));
    const instance = capturedSigmaInstances[0];
    const container = instance.container as HTMLElement;
    const mouseCaptor = instance.getMouseCaptor();

    instance.handlers.get("downNode")?.({ node: "e1" });
    expect(container.style.cursor).toBe("grabbing");

    const refreshSpy = vi.spyOn(instance, "refresh");
    instance.handlers.get("enterNode")?.({ node: "e2" });
    instance.handlers.get("leaveNode")?.({});
    expect(refreshSpy).not.toHaveBeenCalled();
    expect(container.style.cursor).toBe("grabbing");

    mouseCaptor.handlers.get("mouseup")?.({});
    // Hover re-arms the moment the drag ends.
    instance.handlers.get("enterNode")?.({ node: "e2" });
    expect(container.style.cursor).toBe("pointer");
  });

  it("applies every wheel delta directly to the camera — no animated stepped zoom", async () => {
    mockConnectedPair();

    renderWithQuery(<AtlasView />);
    await waitFor(() => expect(capturedSigmaInstances).toHaveLength(1));
    const instance = capturedSigmaInstances[0];
    const mouseCaptor = instance.getMouseCaptor();
    const preventSigmaDefault = vi.fn();

    // The mount zoom-out already called setState once; count from here.
    instance.camera.setState.mockClear();
    instance.camera.ratio = 1;
    mouseCaptor.handlers.get("wheel")?.({
      x: 100,
      y: 50,
      delta: 1,
      preventSigmaDefault,
      original: { deltaY: -120, deltaMode: 0, ctrlKey: false },
    });

    // Sigma's own eased-lurch handler must be suppressed...
    expect(preventSigmaDefault).toHaveBeenCalled();
    // ...and the camera set synchronously with d3-zoom's delta scale:
    // ratio 1 * 2^(-120 * 0.002) — zoomed toward the cursor.
    expect(instance.getViewportZoomedState).toHaveBeenCalledWith(
      { x: 100, y: 50 },
      Math.pow(2, -120 * 0.002),
    );
    expect(instance.camera.setState).toHaveBeenCalledWith({
      x: 100,
      y: 50,
      ratio: Math.pow(2, -120 * 0.002),
    });
  });

  it("suppresses onNodeClick when clickNode follows a moved drag", async () => {
    // A drawn node needs an edge now that degree-0 nodes are hidden, so this
    // interacts with the connected pair rather than a lone entity.
    mockConnectedPair();
    const onNodeClick = vi.fn();

    renderWithQuery(<AtlasView onNodeClick={onNodeClick} />);
    await waitFor(() => expect(capturedSigmaInstances).toHaveLength(1));
    const instance = capturedSigmaInstances[0];
    const mouseCaptor = instance.getMouseCaptor();

    instance.handlers.get("downNode")?.({ node: "e1" });
    mouseCaptor.handlers.get("mousemovebody")?.({
      x: 10,
      y: 10,
      preventSigmaDefault: () => {},
      original: { preventDefault: () => {}, stopPropagation: () => {} },
    });
    mouseCaptor.handlers.get("mouseup")?.({});
    instance.handlers.get("clickNode")?.({ node: "e1" });

    expect(onNodeClick).not.toHaveBeenCalled();
  });

  it("still fires onNodeClick for a plain click with no prior drag", async () => {
    // A drawn node needs an edge now that degree-0 nodes are hidden, so this
    // interacts with the connected pair rather than a lone entity.
    mockConnectedPair();
    const onNodeClick = vi.fn();

    renderWithQuery(<AtlasView onNodeClick={onNodeClick} />);
    await waitFor(() => expect(capturedSigmaInstances).toHaveLength(1));

    capturedSigmaInstances[0].handlers.get("clickNode")?.({ node: "e1" });

    expect(onNodeClick).toHaveBeenCalledWith({ kind: "entity", id: "e1" });
  });

  // A hub connected to two neighbors — enough to prove drag-follow moves a
  // neighbor, without pulling in the whole buildGraphModel relation-shape test
  // surface (that's model.test.ts's job). Three nodes, not two, because the
  // map only draws components of at least MIN_COMPONENT_SIZE (model.ts): a
  // bare pair is a crumb and goes behind the hidden chip.
  function mockConnectedPair() {
    const entities = [
      makeEntity({ id: "e1", name: "Alice" }),
      makeEntity({ id: "e2", name: "Bob" }),
      makeEntity({ id: "e3", name: "Carol" }),
    ];
    mockListEntities.mockResolvedValue(entities);
    mockGetEntityDetail.mockImplementation(async (id: string) => {
      if (id === "e1") {
        return {
          entity: entities[0],
          observations: [],
          relations: [
            {
              id: "rel-1",
              relation_type: "knows",
              direction: "outgoing" as const,
              entity_id: "e2",
              entity_name: "Bob",
              entity_type: "person",
              source_agent: null,
              created_at: Math.floor(Date.now() / 1000),
            },
            {
              id: "rel-2",
              relation_type: "knows",
              direction: "outgoing" as const,
              entity_id: "e3",
              entity_name: "Carol",
              entity_type: "person",
              source_agent: null,
              created_at: Math.floor(Date.now() / 1000),
            },
          ],
        };
      }
      return { entity: entities.find((e) => e.id === id)!, observations: [], relations: [] };
    });
    return entities;
  }

  // The connected pair plus one memory linked to Alice — the memory is a
  // drawn node in its own right (id "mem:m1") with an edge to the entity.
  function mockConnectedPairWithMemory() {
    const entities = mockConnectedPair();
    mockMemories(
      [makeMemory({ source_id: "m1", title: "A decision I made", memory_type: "decision" })],
      [{ memory_id: "m1", entity_id: "e1" }],
    );
    return entities;
  }

  // Same connected trio plus a fourth, unrelated entity — degree 0, so it is
  // never drawn and never joins the sim (see atlas.ts's createAtlasSimulation).
  /** A five-node star around Alice plus one unconnected entity. Five is
   *  MIN_COMPONENT_SIZE: the star is a real component the map draws, so the
   *  lone "Isolate" is genuinely small beside it and gets hidden. Names avoid
   *  the letter "e" apart from Alice and Isolate, which the search tests
   *  match on. */
  function mockConnectedPairWithIsolate() {
    const entities = [
      makeEntity({ id: "e1", name: "Alice" }),
      makeEntity({ id: "e2", name: "Bob" }),
      makeEntity({ id: "e3", name: "Carol" }),
      makeEntity({ id: "e4", name: "Isolate" }),
      makeEntity({ id: "e5", name: "Ravi" }),
      makeEntity({ id: "e6", name: "Hugo" }),
    ];
    const spoke = (id: string, target: string, name: string) => ({
      id,
      relation_type: "knows",
      direction: "outgoing" as const,
      entity_id: target,
      entity_name: name,
      entity_type: "person",
      source_agent: null,
      created_at: Math.floor(Date.now() / 1000),
    });
    mockListEntities.mockResolvedValue(entities);
    mockGetEntityDetail.mockImplementation(async (id: string) => {
      if (id === "e1") {
        return {
          entity: entities[0],
          observations: [],
          relations: [
            spoke("rel-1", "e2", "Bob"),
            spoke("rel-2", "e3", "Carol"),
            spoke("rel-3", "e5", "Ravi"),
            spoke("rel-4", "e6", "Hugo"),
          ],
        };
      }
      return { entity: entities.find((e) => e.id === id)!, observations: [], relations: [] };
    });
    return entities;
  }

  const dragEvent = (x: number, y: number) => ({
    x,
    y,
    preventSigmaDefault: () => {},
    original: { preventDefault: () => {}, stopPropagation: () => {} },
  });

  it("zooms the fitted camera out to the fill-aware default density (60% fill, clamped [1.5, 3])", async () => {
    mockConnectedPair();

    renderWithQuery(<AtlasView />);
    await waitFor(() => expect(capturedSigmaInstances).toHaveLength(1));
    const instance = capturedSigmaInstances[0];
    // Mock fit density 6 px/unit (graphToViewport scales by 6); target is
    // 0.6 * min(400, 600) / span 100 = 2.4 px/unit → ratio 6 / 2.4 = 2.5.
    expect(instance.camera.setState).toHaveBeenCalledWith({ ratio: 2.5 });
  });

  it("paints synchronously on every physics tick — the writeback drives sigma's refresh", async () => {
    mockConnectedPair();

    renderWithQuery(<AtlasView />);
    await waitFor(() => expect(capturedSigmaInstances).toHaveLength(1));
    const instance = capturedSigmaInstances[0];
    const sim = (window as any).__ATLAS_SIM;

    const refreshSpy = vi.spyOn(instance, "refresh");
    sim.tick(1);
    // One frame late (sigma's own event-scheduled render) is the round-8
    // drag-latency bug — the writeback must paint in the same frame.
    expect(refreshSpy).toHaveBeenCalled();
  });

  it("moves a connected non-dragged node as the reheated sim ticks", async () => {
    mockConnectedPair();

    renderWithQuery(<AtlasView />);
    await waitFor(() => expect(capturedSigmaInstances).toHaveLength(1));
    const instance = capturedSigmaInstances[0];
    const graph = instance.graph;
    const mouseCaptor = instance.getMouseCaptor();
    // d3-timer's automatic per-frame loop never fires on its own inside a
    // test — drive the DEV-only captured sim ref directly instead.
    // sim.tick() is synchronous and writes back to the graph (see atlas.ts's
    // createAtlasSimulation).
    const sim = (window as any).__ATLAS_SIM;
    const restartSpy = vi.spyOn(sim, "restart");

    const before = { x: graph.getNodeAttribute("e2", "x"), y: graph.getNodeAttribute("e2", "y") };

    instance.handlers.get("downNode")?.({ node: "e1" });

    // downNode pins the pressed node and reheats the sim: alpha JUMPS to 0.3
    // (not just alphaTarget's 3%/tick ramp from the settled 0 — that ramp is
    // the round-7 "drag feels laggy" bug).
    expect(restartSpy).toHaveBeenCalledTimes(1);
    expect(sim.alphaTarget()).toBeCloseTo(0.3);
    expect(sim.alpha()).toBeCloseTo(0.3);

    mouseCaptor.handlers.get("mousemovebody")?.(dragEvent(200, 200));
    sim.tick(30);

    const after = { x: graph.getNodeAttribute("e2", "x"), y: graph.getNodeAttribute("e2", "y") };
    expect(after).not.toEqual(before);
  });

  it("stops the sim (skipping the decay tail) on mouseup under prefers-reduced-motion, without gating the dragged node's own instant response", async () => {
    const matchMediaMock = vi.fn().mockReturnValue({ matches: true } as MediaQueryList);
    vi.stubGlobal("matchMedia", matchMediaMock);
    try {
      mockConnectedPair();

      renderWithQuery(<AtlasView />);
      await waitFor(() => expect(capturedSigmaInstances).toHaveLength(1));
      const instance = capturedSigmaInstances[0];
      const graph = instance.graph;
      const mouseCaptor = instance.getMouseCaptor();
      const sim = (window as any).__ATLAS_SIM;
      const stopSpy = vi.spyOn(sim, "stop");
      const settleSpy = vi.spyOn(sim, "settleShelf");

      const before = { x: graph.getNodeAttribute("e1", "x"), y: graph.getNodeAttribute("e1", "y") };

      instance.handlers.get("downNode")?.({ node: "e1" });
      mouseCaptor.handlers.get("mousemovebody")?.(dragEvent(200, 200));

      // Drag-follow is direct manipulation, not an animation — the dragged
      // node's own position updates instantly regardless of reduced motion.
      const afterDrag = { x: graph.getNodeAttribute("e1", "x"), y: graph.getNodeAttribute("e1", "y") };
      expect(afterDrag).not.toEqual(before);

      mouseCaptor.handlers.get("mouseup")?.({});

      expect(matchMediaMock).toHaveBeenCalledWith("(prefers-reduced-motion: reduce)");
      expect(stopSpy).toHaveBeenCalledTimes(1);
      // Stopping outright skips the tail that walks a released shelf component
      // back onto its slot, so the same correction has to be applied in one
      // step here or the component stays where the drag left it.
      expect(settleSpy).toHaveBeenCalledTimes(1);
    } finally {
      vi.unstubAllGlobals();
    }
  });

  it("keeps degree-0 entities out of the drawn graph and reports them in the chip", async () => {
    // The isolate ring is gone: an unconnected entity is not drawn at all,
    // it is counted.
    mockConnectedPairWithIsolate();

    renderWithQuery(<AtlasView />);
    await waitFor(() => expect(capturedSigmaInstances).toHaveLength(1));
    const graph = capturedSigmaInstances[0].graph;

    expect(graph.hasNode("e1")).toBe(true);
    expect(graph.hasNode("e2")).toBe(true);
    expect(graph.hasNode("e3")).toBe(true);
    expect(graph.hasNode("e4")).toBe(false);
    expect(graph.order).toBe(5);
    expect(screen.getByText("1 node in small groups hidden")).toBeInTheDocument();
    // The count line describes what is drawn; the sixth entity is
    // unconnected, so it is reported by the chip beside it rather than
    // counted here.
    expect(screen.getByText("5 entities · 1 region")).toBeInTheDocument();
  });

  it("hides a component below MIN_COMPONENT_SIZE and folds it into the small-groups chip", async () => {
    // Round 3, section C, retuned in round 5: a bare pair off on its own is a
    // crumb — it carries no shape a reader can navigate by, and dozens of them
    // are what made the map read as confetti. It goes behind the chip with the
    // isolates.
    const entities = [
      makeEntity({ id: "e1", name: "Alice" }),
      makeEntity({ id: "e2", name: "Bob" }),
      makeEntity({ id: "e3", name: "Carol" }),
      makeEntity({ id: "e4", name: "Dan" }),
      makeEntity({ id: "e5", name: "Elle" }),
      makeEntity({ id: "p1", name: "Pat" }),
      makeEntity({ id: "p2", name: "Quinn" }),
    ];
    const relation = (id: string, target: string, name: string) => ({
      id,
      relation_type: "knows",
      direction: "outgoing" as const,
      entity_id: target,
      entity_name: name,
      entity_type: "person",
      source_agent: null,
      created_at: Math.floor(Date.now() / 1000),
    });
    mockListEntities.mockResolvedValue(entities);
    mockGetEntityDetail.mockImplementation(async (id: string) => ({
      entity: entities.find((e) => e.id === id)!,
      observations: [],
      relations:
        id === "e1"
          ? [
              relation("rel-1", "e2", "Bob"),
              relation("rel-2", "e3", "Carol"),
              relation("rel-3", "e4", "Dan"),
              relation("rel-4", "e5", "Elle"),
            ]
          : id === "p1"
            ? [relation("rel-5", "p2", "Quinn")]
            : [],
    }));

    renderWithQuery(<AtlasView />);
    await waitFor(() => expect(capturedSigmaInstances).toHaveLength(1));
    const graph = capturedSigmaInstances[0].graph;

    // The star is drawn; the pair is not, even though both its nodes have a
    // connection and neither is an isolate.
    expect(graph.order).toBe(5);
    expect(graph.hasNode("e1")).toBe(true);
    expect(graph.hasNode("p1")).toBe(false);
    expect(graph.hasNode("p2")).toBe(false);
    expect(graph.hasEdge("p1", "p2")).toBe(false);
    expect(screen.getByText("2 nodes in small groups hidden")).toBeInTheDocument();
    expect(screen.getByText("5 entities · 1 region")).toBeInTheDocument();
  });

  /** A five-node star (the core) plus a two-node group that MIN_COMPONENT_SIZE
   *  keeps off the map by default. */
  function mockStarWithSmallPair() {
    const entities = [
      makeEntity({ id: "e1", name: "Alice" }),
      makeEntity({ id: "e2", name: "Bob" }),
      makeEntity({ id: "e3", name: "Carol" }),
      makeEntity({ id: "e4", name: "Dan" }),
      makeEntity({ id: "e5", name: "Elle" }),
      makeEntity({ id: "p1", name: "Pat" }),
      makeEntity({ id: "p2", name: "Quinn" }),
    ];
    const relation = (id: string, target: string, name: string) => ({
      id,
      relation_type: "knows",
      direction: "outgoing" as const,
      entity_id: target,
      entity_name: name,
      entity_type: "person",
      source_agent: null,
      created_at: Math.floor(Date.now() / 1000),
    });
    mockListEntities.mockResolvedValue(entities);
    mockGetEntityDetail.mockImplementation(async (id: string) => ({
      entity: entities.find((e) => e.id === id)!,
      observations: [],
      relations:
        id === "e1"
          ? [
              relation("rel-1", "e2", "Bob"),
              relation("rel-2", "e3", "Carol"),
              relation("rel-3", "e4", "Dan"),
              relation("rel-4", "e5", "Elle"),
            ]
          : id === "p1"
            ? [relation("rel-5", "p2", "Quinn")]
            : [],
    }));
    return entities;
  }

  const latestGraph = () => capturedSigmaInstances[capturedSigmaInstances.length - 1].graph;

  it("draws the small groups when the chip is clicked, and remembers the choice", async () => {
    mockStarWithSmallPair();

    renderWithQuery(<AtlasView />);
    await waitFor(() => expect(capturedSigmaInstances).toHaveLength(1));
    expect(latestGraph().hasNode("p1")).toBe(false);

    const chip = screen.getByRole("button", { name: "Show small groups" });
    expect(chip).toHaveAttribute("aria-pressed", "false");
    expect(chip).toHaveTextContent("2 nodes in small groups hidden");

    fireEvent.click(chip);

    await waitFor(() => expect(latestGraph().hasNode("p1")).toBe(true));
    const graph = latestGraph();
    expect(graph.hasNode("p2")).toBe(true);
    expect(graph.hasEdge("p1", "p2")).toBe(true);
    // The chip keeps its place and now offers the way back.
    const shown = screen.getByRole("button", { name: "Hide small groups" });
    expect(shown).toHaveAttribute("aria-pressed", "true");
    expect(shown).toHaveTextContent("Hide small groups");
    expect(window.localStorage.getItem("atlas.smallGroups")).toBe("true");

    fireEvent.click(shown);
    await waitFor(() => expect(latestGraph().hasNode("p1")).toBe(false));
    expect(window.localStorage.getItem("atlas.smallGroups")).toBe("false");
  });

  it("puts the revealed small group on a wing beside the core, clear of its bounds", async () => {
    window.localStorage.setItem("atlas.smallGroups", "true");
    mockStarWithSmallPair();

    renderWithQuery(<AtlasView />);
    await waitFor(() => expect(capturedSigmaInstances).toHaveLength(1));
    const graph = latestGraph();
    expect(graph.hasNode("p1")).toBe(true);

    const x = (id: string) => graph.getNodeAttribute(id, "x") as number;
    const size = (id: string) => graph.getNodeAttribute(id, "size") as number;
    const core = ["e1", "e2", "e3", "e4", "e5"];
    const coreLeft = Math.min(...core.map((id) => x(id) - size(id)));
    const coreRight = Math.max(...core.map((id) => x(id) + size(id)));
    // The pair is the largest shelf component, so it takes the first wing:
    // wholly to one side of the core, never tucked into a gap inside it.
    const pair = ["p1", "p2"];
    const pairLeft = Math.min(...pair.map((id) => x(id) - size(id)));
    const pairRight = Math.max(...pair.map((id) => x(id) + size(id)));
    expect(pairLeft > coreRight || pairRight < coreLeft).toBe(true);
  });

  /** A five-node star (the core) plus a four-node star that MIN_COMPONENT_SIZE
   *  keeps off the map by default. Unlike the pair above, the small star is
   *  big enough to earn a region of its own once it is drawn (MIN_REGION_SIZE
   *  is 3), and its hub outranks its leaves so the fallback climb gathers all
   *  four under one community instead of splitting them into singletons. */
  function mockStarWithSmallStar() {
    const entities = [
      makeEntity({ id: "e1", name: "Alice" }),
      makeEntity({ id: "e2", name: "Bob" }),
      makeEntity({ id: "e3", name: "Carol" }),
      makeEntity({ id: "e4", name: "Dan" }),
      makeEntity({ id: "e5", name: "Elle" }),
      makeEntity({ id: "s1", name: "Shelf Hub" }),
      makeEntity({ id: "s2", name: "Shelf Two" }),
      makeEntity({ id: "s3", name: "Shelf Three" }),
      makeEntity({ id: "s4", name: "Shelf Four" }),
    ];
    const relation = (id: string, target: string, name: string) => ({
      id,
      relation_type: "knows",
      direction: "outgoing" as const,
      entity_id: target,
      entity_name: name,
      entity_type: "person",
      source_agent: null,
      created_at: Math.floor(Date.now() / 1000),
    });
    mockListEntities.mockResolvedValue(entities);
    mockGetEntityDetail.mockImplementation(async (id: string) => ({
      entity: entities.find((e) => e.id === id)!,
      observations: [],
      relations:
        id === "e1"
          ? [
              relation("rel-1", "e2", "Bob"),
              relation("rel-2", "e3", "Carol"),
              relation("rel-3", "e4", "Dan"),
              relation("rel-4", "e5", "Elle"),
            ]
          : id === "s1"
            ? [
                relation("rel-5", "s2", "Shelf Two"),
                relation("rel-6", "s3", "Shelf Three"),
                relation("rel-7", "s4", "Shelf Four"),
              ]
            : [],
    }));
    return entities;
  }

  it("paints the rebuilt renderer's overlay when the small-groups chip flips", async () => {
    mockStarWithSmallStar();

    renderWithQuery(<AtlasView />);
    await waitFor(() => expect(capturedSigmaInstances).toHaveLength(1));

    // Stub the prototype, not one canvas: the chip rebuilds the renderer, and
    // the overlay this has to watch does not exist until that happens.
    const painter = recordingOverlayCtx();
    const originalGetContext = HTMLCanvasElement.prototype.getContext;
    HTMLCanvasElement.prototype.getContext = vi.fn().mockReturnValue(painter.ctx) as any;
    try {
      fireEvent.click(screen.getByRole("button", { name: "Show small groups" }));
      await waitFor(() => expect(capturedSigmaInstances).toHaveLength(2));

      // Nothing fires afterRender by hand here, and nothing else can: the chip
      // changes only the DRAWN model, while `communities` and the palette are
      // derived from the full model and the theme, so neither of the two
      // repaint effects re-runs. The rebuilt renderer has to paint its own
      // first frame or the map loses every place name.
      expect(painter.state.paints).toBeGreaterThan(0);
      expect(painter.state.regionNames.length).toBeGreaterThan(0);
    } finally {
      HTMLCanvasElement.prototype.getContext = originalGetContext;
    }
  });

  it("names an island small group only once the view is zoomed in enough for the islands to come up solid", async () => {
    mockStarWithSmallStar();
    mockDimensions = { width: 2000, height: 1200 }; // the island sits out on the rim; keep it in view

    renderWithQuery(<AtlasView />);
    await waitFor(() => expect(capturedSigmaInstances).toHaveLength(1));

    const painter = recordingOverlayCtx();
    const originalGetContext = HTMLCanvasElement.prototype.getContext;
    HTMLCanvasElement.prototype.getContext = vi.fn().mockReturnValue(painter.ctx) as any;
    try {
      fireEvent.click(screen.getByRole("button", { name: "Show small groups" }));
      await waitFor(() => expect(capturedSigmaInstances).toHaveLength(2));

      // At the opening view the island is dim and nameless: only the core's
      // own name is on the map.
      expect(painter.state.regionNames).toContain("Alice");
      expect(painter.state.regionNames).not.toContain("Shelf Hub");

      // Zoom in twice (camera ratio halves): the islands come up solid and
      // the small star earns its name. Its leader is its degree-3 hub.
      const instance = capturedSigmaInstances[1]!;
      const onCamera = instance.camera.on.mock.calls.find((call: unknown[]) => call[0] === "updated")?.[1] as
        | ((state: { ratio: number }) => void)
        | undefined;
      expect(onCamera).toBeDefined();
      onCamera!({ ratio: 0.5 });
      expect(painter.state.regionNames).toContain("Shelf Hub");
      expect(painter.state.regionNames).toContain("Alice");
    } finally {
      HTMLCanvasElement.prototype.getContext = originalGetContext;
    }
  });

  it("counts only regions that are drawn: a hidden small group adds none until it is shown", async () => {
    mockStarWithSmallStar();

    renderWithQuery(<AtlasView />);
    await waitFor(() => expect(capturedSigmaInstances).toHaveLength(1));

    // The small star is its own community in the full model, but while its
    // nodes are hidden nothing draws or names it — so it must not be counted.
    expect(screen.getByText("5 entities · 1 region")).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "Show small groups" }));
    await waitFor(() => expect(capturedSigmaInstances).toHaveLength(2));
    expect(screen.getByText("9 entities · 2 regions")).toBeInTheDocument();
  });

  it("starts hidden when the stored small-groups choice is malformed", async () => {
    window.localStorage.setItem("atlas.smallGroups", "{oops");
    mockStarWithSmallPair();

    renderWithQuery(<AtlasView />);
    await waitFor(() => expect(capturedSigmaInstances).toHaveLength(1));

    expect(latestGraph().hasNode("p1")).toBe(false);
    expect(screen.getByRole("button", { name: "Show small groups" })).toHaveAttribute(
      "aria-pressed",
      "false",
    );
  });

  it("offers no small-groups chip when nothing is hidden", async () => {
    mockConnectedPair();

    renderWithQuery(<AtlasView />);
    await waitFor(() => expect(capturedSigmaInstances).toHaveLength(1));

    expect(screen.queryByRole("button", { name: "Show small groups" })).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Hide small groups" })).not.toBeInTheDocument();
  });

  it("mounts the place-name overlay canvas above sigma, and nothing under it, and removes it on unmount", async () => {
    mockConnectedPair();

    const { unmount } = renderWithQuery(<AtlasView />);
    await waitFor(() => expect(capturedSigmaInstances).toHaveLength(1));

    const container = capturedSigmaInstances[0].container as HTMLElement;
    const overlay = document.querySelector('canvas[data-testid="atlas-region-names"]');
    expect(overlay).toBeInTheDocument();
    // Appended AFTER the sigma mock ran, so it stacks above sigma's canvases.
    expect(overlay!.parentElement).toBe(container);
    // The overlay is the only canvas the Atlas adds: no terrain or wash
    // underlay exists to put a shadow or an aura under the nodes.
    expect(container.querySelectorAll("canvas")).toHaveLength(1);
    expect(container.querySelector('canvas[data-testid="atlas-cartography"]')).toBeNull();

    unmount();
    // Query the DETACHED container, not the document: React unmount removes
    // the whole subtree from the document either way, so a document-level
    // query passes even if the cleanup leaks the canvas (mutation-proven).
    expect(container.querySelector('canvas[data-testid="atlas-region-names"]')).toBeNull();
  });

  it("repaints the place-name overlay on every sigma afterRender, names over the nodes", async () => {
    mockConnectedPair();

    renderWithQuery(<AtlasView />);
    await waitFor(() => expect(capturedSigmaInstances).toHaveLength(1));
    const instance = capturedSigmaInstances[0];

    const handler = instance.handlers.get("afterRender");
    expect(typeof handler).toBe("function");

    // jsdom's getContext("2d") is null (the mount-time paint no-ops on that);
    // hand the handler a recording ctx and re-fire it like a render would.
    const overlay = document.querySelector(
      'canvas[data-testid="atlas-region-names"]',
    ) as HTMLCanvasElement;
    // The overlay is appended AFTER sigma mounts so it stacks above sigma's
    // canvases — names sit on the map, never under the nodes.
    expect(overlay).toBeInTheDocument();
    const ctx = {
      strokeStyle: "",
      fillStyle: "",
      lineWidth: 0,
      lineJoin: "",
      lineCap: "",
      font: "",
      letterSpacing: "",
      textAlign: "",
      textBaseline: "",
      setTransform: vi.fn(),
      clearRect: vi.fn(),
      save: vi.fn(),
      restore: vi.fn(),
      beginPath: vi.fn(),
      closePath: vi.fn(),
      moveTo: vi.fn(),
      lineTo: vi.fn(),
      bezierCurveTo: vi.fn(),
      arc: vi.fn(),
      setLineDash: vi.fn(),
      stroke: vi.fn(),
      fill: vi.fn(),
      drawImage: vi.fn(),
      strokeText: vi.fn(),
      fillText: vi.fn(),
      measureText: vi.fn((text: string) => ({ width: text.length * 7 })),
    };
    const order: string[] = [];
    const overCtx = {
      ...ctx,
      setTransform: vi.fn(),
      clearRect: vi.fn(),
      arc: vi.fn(),
      fill: vi.fn(),
      strokeText: vi.fn(() => order.push("stroke")),
      fillText: vi.fn(() => order.push("fill")),
    };
    overlay.getContext = vi.fn().mockReturnValue(overCtx) as any;

    handler!({});

    // Sized to sigma's viewport and cleared before painting.
    expect(overlay.width).toBe(400);
    expect(overlay.height).toBe(600);
    expect(overCtx.clearRect).toHaveBeenCalledWith(0, 0, 400, 600);
    // One community of three, so exactly one place name — the label placer
    // keeps it because nothing else competes for the space — and its
    // ground-coloured halo is stroked before the fill. Text is ALL the 2D
    // layer paints: no discs, no washes, no outlines under the nodes.
    expect(overCtx.fillText).toHaveBeenCalledTimes(1);
    expect(overCtx.strokeText).toHaveBeenCalledTimes(1);
    expect(order).toEqual(["stroke", "fill"]);
    expect(overCtx.arc).not.toHaveBeenCalled();
    expect(overCtx.fill).not.toHaveBeenCalled();
    expect(overCtx.stroke).not.toHaveBeenCalled();
    expect(overCtx.drawImage).not.toHaveBeenCalled();
    expect(ctx.clearRect).not.toHaveBeenCalled();
  });

  // Two 3-cliques joined by one bridge (a1–b1). Peak-climbing puts each
  // triangle in its own community (a1/b1 are their own degree peaks — the
  // bridge neighbor ties at 3, never strictly higher), so 2 regions.
  function mockTwoTriangles() {
    const names: Record<string, string> = {
      a1: "Alice",
      a2: "Anna",
      a3: "Ada",
      b1: "Bob",
      b2: "Ben",
      b3: "Bea",
    };
    const entities = Object.entries(names).map(([id, name]) => makeEntity({ id, name }));
    const rel = (id: string, target: string) => ({
      id,
      relation_type: "knows",
      direction: "outgoing" as const,
      entity_id: target,
      entity_name: names[target],
      entity_type: "concept",
      source_agent: null,
      created_at: Math.floor(Date.now() / 1000),
    });
    mockListEntities.mockResolvedValue(entities);
    mockGetEntityDetail.mockImplementation(async (id: string) => {
      const relations =
        id === "a1"
          ? [rel("r1", "a2"), rel("r2", "a3"), rel("rb", "b1")]
          : id === "a2"
            ? [rel("r3", "a3")]
            : id === "b1"
              ? [rel("r4", "b2"), rel("r5", "b3")]
              : id === "b2"
                ? [rel("r6", "b3")]
                : [];
      return { entity: entities.find((e) => e.id === id)!, observations: [], relations };
    });
  }

  it("shows the artifact count line — entities · regions — in the toolbar", async () => {
    mockTwoTriangles();

    renderWithQuery(<AtlasView />);
    await waitFor(() => expect(capturedSigmaInstances).toHaveLength(1));

    expect(screen.getByText("6 entities · 2 regions")).toBeInTheDocument();
  });

  it("opens a hidden isolate's page from search instead of flying the camera to nothing", async () => {
    mockConnectedPairWithIsolate();
    const onNodeClick = vi.fn();

    renderWithQuery(<AtlasView onNodeClick={onNodeClick} />);
    await waitFor(() => expect(capturedSigmaInstances).toHaveLength(1));
    const instance = capturedSigmaInstances[0];

    // The isolate is not drawn (degree 0, see visibleModel), but search still
    // lists it; Enter navigates to its entity page rather than animating the
    // camera onto empty map.
    const input = screen.getByPlaceholderText("Jump to anything…");
    fireEvent.focus(input);
    fireEvent.change(input, { target: { value: "isol" } });
    fireEvent.keyDown(input, { key: "Enter" });

    expect(onNodeClick).toHaveBeenCalledWith({ kind: "entity", id: "e4" });
    expect(instance.camera.animate).not.toHaveBeenCalled();
  });

  it("focuses the search input on ⌘K", async () => {
    mockConnectedPair();

    renderWithQuery(<AtlasView />);
    await waitFor(() => expect(capturedSigmaInstances).toHaveLength(1));

    const input = screen.getByPlaceholderText("Jump to anything…");
    expect(document.activeElement).not.toBe(input);

    fireEvent.keyDown(window, { key: "k", metaKey: true });
    expect(document.activeElement).toBe(input);
  });

  it("matches case-insensitively and Enter flies the camera to the match with hover emphasis", async () => {
    mockConnectedPairWithIsolate();

    renderWithQuery(<AtlasView />);
    await waitFor(() => expect(capturedSigmaInstances).toHaveLength(1));
    const instance = capturedSigmaInstances[0];
    const { settings } = instance;

    // e3 is not in Alice's neighborhood — its reducer output must change
    // once the search emphasis lands (the hover-dim path).
    const attrs = { color: "#abc", size: 4, entityType: "concept", confirmed: false };
    const before = settings.nodeReducer("e3", attrs);

    const input = screen.getByPlaceholderText("Jump to anything…");
    fireEvent.focus(input);
    // Lowercase query against "Alice" — pins the case-insensitive match.
    fireEvent.change(input, { target: { value: "ali" } });
    expect(screen.getByRole("option", { name: /Alice/ })).toBeInTheDocument();

    fireEvent.keyDown(input, { key: "Enter" });

    // Camera target comes from getNodeDisplayData (mock: 0.42/0.24); ratio
    // never grows past the current view (mock camera ratio 1).
    expect(instance.camera.animate).toHaveBeenCalledWith(
      { x: 0.42, y: 0.24, ratio: 1 },
      { duration: 450 },
    );
    expect(settings.nodeReducer("e3", attrs)).not.toEqual(before);
  });

  it("points aria-activedescendant at the active option and tracks arrow keys", async () => {
    mockConnectedPairWithIsolate();

    renderWithQuery(<AtlasView />);
    await waitFor(() => expect(capturedSigmaInstances).toHaveLength(1));

    const input = screen.getByPlaceholderText("Jump to anything…");
    // Closed dropdown → no dangling descendant reference.
    expect(input).not.toHaveAttribute("aria-activedescendant");

    fireEvent.focus(input);
    // "e" hits both Alice and Isolate.
    fireEvent.change(input, { target: { value: "e" } });
    expect(screen.getAllByRole("option")).toHaveLength(2);

    expect(input).toHaveAttribute("aria-activedescendant", "atlas-search-option-0");
    expect(document.getElementById("atlas-search-option-0")).toHaveTextContent("Alice");

    fireEvent.keyDown(input, { key: "ArrowDown" });
    expect(input).toHaveAttribute("aria-activedescendant", "atlas-search-option-1");
    expect(document.getElementById("atlas-search-option-1")).toHaveTextContent("Isolate");
  });

  it("Regions chip hides the place-name overlay and repaints it on re-show", async () => {
    mockConnectedPair();

    renderWithQuery(<AtlasView />);
    await waitFor(() => expect(capturedSigmaInstances).toHaveLength(1));
    const instance = capturedSigmaInstances[0];

    const chip = screen.getByRole("button", { name: "Regions" });
    const overlay = document.querySelector('[data-testid="atlas-region-names"]') as HTMLCanvasElement;
    expect(chip).toHaveAttribute("aria-pressed", "true");
    expect(overlay.style.display).not.toBe("none");

    fireEvent.click(chip);
    expect(chip).toHaveAttribute("aria-pressed", "false");
    expect(overlay.style.display).toBe("none");

    // Re-show must refresh: the hidden canvas kept its stale last frame.
    const refreshSpy = vi.spyOn(instance, "refresh");
    fireEvent.click(chip);
    expect(overlay.style.display).not.toBe("none");
    expect(refreshSpy).toHaveBeenCalled();
  });

  it("Space selector scopes the graph to one space and back", async () => {
    // Five connected entities in one space, because a component smaller than
    // MIN_COMPONENT_SIZE (model.ts) is not drawn at all — plus one isolate in
    // another space, which is what the selector has to scope away.
    const names: Record<string, string> = { e2: "Bob", e3: "Carol", e5: "Ravi", e6: "Hugo" };
    const entities = [
      makeEntity({ id: "e1", name: "Alice", domain: "wenlan-dev" }),
      makeEntity({ id: "e2", name: "Bob", domain: "wenlan-dev" }),
      makeEntity({ id: "e3", name: "Carol", domain: "wenlan-dev" }),
      makeEntity({ id: "e4", name: "Isolate", domain: "personal" }),
      makeEntity({ id: "e5", name: "Ravi", domain: "wenlan-dev" }),
      makeEntity({ id: "e6", name: "Hugo", domain: "wenlan-dev" }),
    ];
    mockListEntities.mockResolvedValue(entities);
    mockGetEntityDetail.mockImplementation(async (id: string) => ({
      entity: entities.find((e) => e.id === id)!,
      observations: [],
      relations:
        id === "e1"
          ? Object.entries(names).map(([target, name], i) => ({
              id: `rel-${i + 1}`,
              relation_type: "knows",
              direction: "outgoing" as const,
              entity_id: target,
              entity_name: name,
              entity_type: "person",
              source_agent: null,
              created_at: Math.floor(Date.now() / 1000),
            }))
          : [],
    }));

    renderWithQuery(<AtlasView />);
    await waitFor(() => expect(capturedSigmaInstances).toHaveLength(1));

    const select = screen.getByRole("combobox", { name: "Space" });
    expect(
      Array.from((select as HTMLSelectElement).options).map((o) => o.textContent),
    ).toEqual(["All spaces", "personal", "wenlan-dev"]);
    // Only the wenlan-dev star is connected, so the count line reads 5 either
    // way; the scoping is visible in the graph itself and in the hidden chip.
    expect(screen.getByText("5 entities · 1 region")).toBeInTheDocument();
    expect(screen.getByText("1 node in small groups hidden")).toBeInTheDocument();

    fireEvent.change(select, { target: { value: "wenlan-dev" } });
    expect(await screen.findByText("5 entities · 1 region")).toBeInTheDocument();
    expect(screen.queryByText("1 node in small groups hidden")).not.toBeInTheDocument();

    fireEvent.change(select, { target: { value: "" } });
    expect(await screen.findByText("1 node in small groups hidden")).toBeInTheDocument();
  });

  it("hides the Space selector when no entity carries a space", async () => {
    mockConnectedPair();

    renderWithQuery(<AtlasView />);
    await waitFor(() => expect(capturedSigmaInstances).toHaveLength(1));

    // Toolbar is provably rendered (Regions chip present) before the absence claim.
    expect(screen.getByRole("button", { name: "Regions" })).toBeInTheDocument();
    expect(screen.queryByRole("combobox", { name: "Space" })).not.toBeInTheDocument();
  });

  // A single entity carrying a space — the minimum needed to put the
  // cartography query in flight for the badge tests below.
  function mockOneSpaceEntity(domain = "wenlan-dev") {
    const entity = makeEntity({ id: "e1", domain });
    mockListEntities.mockResolvedValue([entity]);
    mockGetEntityDetail.mockResolvedValue({ entity, observations: [], relations: [] });
    return entity;
  }

  function readyCommunityPages(space: string) {
    return {
      communities: {
        schema_version: "community-read-v1" as const,
        communities: [
          {
            community_id: "c1",
            space,
            display_name: null,
            member_count: 1,
            published_generation: 1,
            algo_version: "v1",
            projection_version: "v1",
          },
        ],
        next_cursor: null,
      },
      members: {
        schema_version: "community-read-v1" as const,
        members: [
          {
            space,
            node_id: "e1",
            node_kind: "entity",
            community_id: "c1",
            published_generation: 1,
            attachment: "core",
          },
        ],
        next_cursor: null,
      },
    };
  }

  it("badges the toolbar with durable-regions once the space's community read comes back ready", async () => {
    mockOneSpaceEntity();
    const pages = readyCommunityPages("wenlan-dev");
    mockListCommunities.mockResolvedValue(pages.communities);
    mockListCommunityMembers.mockResolvedValue(pages.members);

    renderWithQuery(<AtlasView />);
    await waitFor(() => expect(capturedSigmaInstances).toHaveLength(1));

    expect(await screen.findByText("Durable regions")).toBeInTheDocument();
  });

  it("keeps the badge honest when a space-less entity sits beside an otherwise-ready space", async () => {
    // "e1" carries a real space and its own community read comes back ready;
    // "ghost" is a real entity with no space at all, drawn on the fallback
    // climb — the aggregate must never read as all-durable while that
    // unreported node is on the map.
    const entity = makeEntity({ id: "e1", domain: "wenlan-dev" });
    const ghost = makeEntity({ id: "ghost", name: "Ghost" });
    mockListEntities.mockResolvedValue([entity, ghost]);
    mockGetEntityDetail.mockImplementation(async (id: string) =>
      id === "e1"
        ? {
            entity,
            observations: [],
            relations: [
              {
                id: "rel-1",
                relation_type: "knows",
                direction: "outgoing" as const,
                entity_id: "ghost",
                entity_name: "Ghost",
                entity_type: "concept",
                source_agent: null,
                created_at: Math.floor(Date.now() / 1000),
              },
            ],
          }
        : { entity: ghost, observations: [], relations: [] },
    );
    const pages = readyCommunityPages("wenlan-dev");
    mockListCommunities.mockResolvedValue(pages.communities);
    mockListCommunityMembers.mockResolvedValue(pages.members);

    renderWithQuery(<AtlasView />);
    await waitFor(() => expect(capturedSigmaInstances).toHaveLength(1));

    expect(await screen.findByText("Estimated regions")).toBeInTheDocument();
    expect(screen.queryByText("Durable regions")).not.toBeInTheDocument();
  });

  it("badges the toolbar with estimated-regions while no durable data has been published for the space yet", async () => {
    mockOneSpaceEntity();

    renderWithQuery(<AtlasView />);
    await waitFor(() => expect(capturedSigmaInstances).toHaveLength(1));

    expect(await screen.findByText("Estimated regions")).toBeInTheDocument();
  });

  // An entity's OWN space can be null or empty (model.ts's GraphNode.space) —
  // it does not take a relation-only neighbor to put unscoped nodes on the
  // map. Such a graph renders entirely on the fallback climb, so the badge
  // must say so even though there is no relation, and no known space, to give
  // it away.
  it.each([
    ["null", null],
    ["empty-string", ""],
  ])("badges the toolbar as estimated for a relation-free graph whose entity carries a %s space", async (_label, space) => {
    const entity = makeEntity({ id: "e1", name: "Alice", domain: null, space });
    mockListEntities.mockResolvedValue([entity]);
    mockGetEntityDetail.mockResolvedValue({ entity, observations: [], relations: [] });

    renderWithQuery(<AtlasView />);
    await waitFor(() => expect(capturedSigmaInstances).toHaveLength(1));

    // Toolbar is provably rendered before the badge claim.
    expect(screen.getByRole("button", { name: "Regions" })).toBeInTheDocument();
    expect(await screen.findByText("Estimated regions")).toBeInTheDocument();
  });

  // Scoping to a space drops the other space's entities AND every relation
  // that pointed at them. Nothing is synthesized any more: the bulk read
  // already carries every entity, so a half-visible relation is a filtered-out
  // edge, not a node the view has to invent.
  it("drops a cross-space relation under a space filter instead of synthesizing an unscoped neighbor", async () => {
    // A five-node Work star around Alice, with Bob in Personal hanging off
    // her too: unfiltered they are one component, and Work on its own still
    // clears MIN_COMPONENT_SIZE (model.ts) so the scoped graph is drawn.
    const workNames: Record<string, string> = { e3: "Carol", e5: "Ravi", e6: "Hugo", e7: "Wu" };
    const entities = [
      makeEntity({ id: "e1", name: "Alice", domain: "Work" }),
      makeEntity({ id: "e2", name: "Bob", domain: "Personal" }),
      makeEntity({ id: "e3", name: "Carol", domain: "Work" }),
      makeEntity({ id: "e5", name: "Ravi", domain: "Work" }),
      makeEntity({ id: "e6", name: "Hugo", domain: "Work" }),
      makeEntity({ id: "e7", name: "Wu", domain: "Work" }),
    ];
    mockListEntities.mockResolvedValue(entities);
    mockGetEntityDetail.mockImplementation(async (id: string) => ({
      entity: entities.find((e) => e.id === id)!,
      observations: [],
      relations:
        id === "e1"
          ? [["e2", "Bob"] as const, ...Object.entries(workNames)].map(([target, name], i) => ({
              id: `rel-${i + 1}`,
              relation_type: "knows",
              direction: "outgoing" as const,
              entity_id: target,
              entity_name: name,
              entity_type: "person",
              source_agent: null,
              created_at: Math.floor(Date.now() / 1000),
            }))
          : [],
    }));
    // Both spaces publish durable cartography, so nothing about the daemon
    // reads is degraded — only the filtered view is.
    const communityId = (space: string) => (space === "Work" ? "c1" : "c2");
    const nodeIds = (space: string) =>
      space === "Work" ? ["e1", "e3", "e5", "e6", "e7"] : ["e2"];
    mockListCommunities.mockImplementation(async (space: string) => ({
      schema_version: "community-read-v1" as const,
      communities: [
        {
          community_id: communityId(space),
          space,
          display_name: null,
          member_count: nodeIds(space).length,
          published_generation: 1,
          algo_version: "v1",
          projection_version: "v1",
        },
      ],
      next_cursor: null,
    }));
    mockListCommunityMembers.mockImplementation(async (space: string) => ({
      schema_version: "community-read-v1" as const,
      members: nodeIds(space).map((node_id) => ({
        space,
        node_id,
        node_kind: "entity",
        community_id: communityId(space),
        published_generation: 1,
        attachment: "core",
      })),
      next_cursor: null,
    }));

    renderWithQuery(<AtlasView />);
    await waitFor(() => expect(capturedSigmaInstances).toHaveLength(1));

    // Unfiltered, every rendered node carries a ready space.
    expect(await screen.findByText("Durable regions")).toBeInTheDocument();
    expect(capturedSigmaInstances[0].graph.hasEdge("e1", "e2")).toBe(true);

    fireEvent.change(screen.getByRole("combobox", { name: "Space" }), {
      target: { value: "Work" },
    });

    // Bob is gone and so is the edge to him — and no space-less node was
    // invented to stand in for him, so the badge stays durable.
    await waitFor(() => expect(capturedSigmaInstances.length).toBeGreaterThan(1));
    const scoped = capturedSigmaInstances[capturedSigmaInstances.length - 1].graph;
    expect(scoped.hasNode("e2")).toBe(false);
    expect(scoped.order).toBe(5);
    expect(scoped.hasEdge("e1", "e2")).toBe(false);
    expect(screen.getByText("Durable regions")).toBeInTheDocument();
    expect(screen.queryByText("Estimated regions")).not.toBeInTheDocument();
  });

  it("badges the toolbar with an alerting sync-issue state when a space's community read fails", async () => {
    mockOneSpaceEntity();
    mockListCommunities.mockRejectedValue(new Error("connection reset"));

    renderWithQuery(<AtlasView />);
    await waitFor(() => expect(capturedSigmaInstances).toHaveLength(1));

    const badge = await screen.findByText("Region sync issue");
    expect(badge).toHaveAttribute("role", "alert");
  });

  // Same two-triangles-one-bridge shape as mockTwoTriangles, but every
  // entity carries a space so the cartography query actually fires (a bare
  // mockTwoTriangles never puts the constellation-cartography query in
  // flight — see mockOneSpaceEntity's comment above).
  function mockTwoTrianglesInSpace(domain = "wenlan-dev") {
    const names: Record<string, string> = {
      a1: "Alice",
      a2: "Anna",
      a3: "Ada",
      b1: "Bob",
      b2: "Ben",
      b3: "Bea",
    };
    const entities = Object.entries(names).map(([id, name]) => makeEntity({ id, name, domain }));
    const rel = (id: string, target: string) => ({
      id,
      relation_type: "knows",
      direction: "outgoing" as const,
      entity_id: target,
      entity_name: names[target],
      entity_type: "concept",
      source_agent: null,
      created_at: Math.floor(Date.now() / 1000),
    });
    mockListEntities.mockResolvedValue(entities);
    mockGetEntityDetail.mockImplementation(async (id: string) => {
      const relations =
        id === "a1"
          ? [rel("r1", "a2"), rel("r2", "a3"), rel("rb", "b1")]
          : id === "a2"
            ? [rel("r3", "a3")]
            : id === "b1"
              ? [rel("r4", "b2"), rel("r5", "b3")]
              : id === "b2"
                ? [rel("r6", "b3")]
                : [];
      return { entity: entities.find((e) => e.id === id)!, observations: [], relations };
    });
  }

  // Records what the place-name overlay actually painted. The overlay clears
  // once per paint before drawing, so resetting on clearRect leaves
  // `regionNames` holding the MOST RECENT paint only — immune to however many
  // extra repaints a render happens to trigger — while `paints` counts them.
  function recordingOverlayCtx() {
    const state = { paints: 0, regionNames: [] as string[] };
    const ctx = {
      strokeStyle: "",
      fillStyle: "",
      lineWidth: 0,
      lineJoin: "",
      lineCap: "",
      font: "",
      letterSpacing: "",
      textAlign: "",
      textBaseline: "",
      setTransform: vi.fn(),
      clearRect: vi.fn(() => {
        state.paints += 1;
        state.regionNames.length = 0;
      }),
      save: vi.fn(),
      restore: vi.fn(),
      beginPath: vi.fn(),
      closePath: vi.fn(),
      moveTo: vi.fn(),
      lineTo: vi.fn(),
      bezierCurveTo: vi.fn(),
      arc: vi.fn(),
      setLineDash: vi.fn(),
      stroke: vi.fn(),
      fill: vi.fn(),
      strokeText: vi.fn(),
      drawImage: vi.fn(),
      fillText: vi.fn((text: string) => {
        state.regionNames.push(text);
      }),
      // The region-label placement pass measures every candidate name; jsdom
      // has no text engine, so report a deterministic 7px per character.
      measureText: vi.fn((text: string) => ({ width: text.length * 7 })),
    };
    return { ctx, state };
  }

  it("paints the names on mount through the post-mount effects, with no second draw of its own", async () => {
    mockTwoTrianglesInSpace();
    mockDimensions = { width: 2000, height: 1200 }; // both triangles in view, like a real fit
    // The helper canvases only exist once the mount effect has run, so stub
    // the prototype to catch the very first paint.
    const painter = recordingOverlayCtx();
    const originalGetContext = HTMLCanvasElement.prototype.getContext;
    HTMLCanvasElement.prototype.getContext = vi.fn().mockReturnValue(painter.ctx) as any;
    try {
      renderWithQuery(<AtlasView />);
      await waitFor(() => expect(capturedSigmaInstances).toHaveLength(1));

      // Nothing here fired afterRender by hand: the theme and cartography
      // effects both refresh() right after the mount effect, and that is the
      // whole initial paint path — so the mount effect must not draw as well.
      expect(painter.state.paints).toBeGreaterThan(0);
      expect(painter.state.regionNames).toEqual(["Alice", "Bob"]);
    } finally {
      HTMLCanvasElement.prototype.getContext = originalGetContext;
    }
  });

  it("re-renders the place names when a space's cartography status changes, without a full remount", async () => {
    mockTwoTrianglesInSpace();
    mockDimensions = { width: 2000, height: 1200 }; // both triangles in view, like a real fit
    // Starts on the default (empty) fallback climb: two 3-cliques joined by
    // "rb" — an ordinary edge; nothing is painted as a bridge any more.
    const { qc } = renderWithQuery(<AtlasView />);
    await waitFor(() => expect(capturedSigmaInstances).toHaveLength(1));
    const instance = capturedSigmaInstances[0];
    const graph = instance.graph;

    expect(graph.getEdgeAttribute("rb", "bridge")).toBeUndefined();

    // Watch the names: on the fallback climb the two triangles are two named
    // regions (leaders a1/b1 — the degree-3 hubs).
    const overlay = document.querySelector(
      'canvas[data-testid="atlas-region-names"]',
    ) as HTMLCanvasElement;
    const painter = recordingOverlayCtx();
    overlay.getContext = vi.fn().mockReturnValue(painter.ctx) as any;
    instance.handlers.get("afterRender")!({});
    expect(painter.state.regionNames).toEqual(["Alice", "Bob"]);
    const paintsBeforeChange = painter.state.paints;

    // Durable arrives: the daemon puts all 6 members in ONE community, so
    // the two regions merge into one.
    const allOneCommunity = {
      communities: {
        schema_version: "community-read-v1" as const,
        communities: [
          {
            community_id: "one",
            space: "wenlan-dev",
            display_name: null,
            member_count: 6,
            published_generation: 1,
            algo_version: "v1",
            projection_version: "v1",
          },
        ],
        next_cursor: null,
      },
      members: {
        schema_version: "community-read-v1" as const,
        members: ["a1", "a2", "a3", "b1", "b2", "b3"].map((node_id) => ({
          space: "wenlan-dev",
          node_id,
          node_kind: "entity",
          community_id: "one",
          published_generation: 1,
          attachment: "core",
        })),
        next_cursor: null,
      },
    };
    mockListCommunities.mockResolvedValue(allOneCommunity.communities);
    mockListCommunityMembers.mockResolvedValue(allOneCommunity.members);
    await qc.invalidateQueries({ queryKey: ["constellation-cartography"] });

    // The names repainted on their OWN — no test-fired afterRender here — and
    // the two regions have become the one merged region the daemon declared.
    await waitFor(() => expect(painter.state.regionNames).toEqual(["Alice"]));
    expect(painter.state.paints).toBeGreaterThan(paintsBeforeChange);
    // Same sigma instance and the same graphology graph — no remount, no
    // layout/camera reset.
    expect(capturedSigmaInstances).toHaveLength(1);
    expect(capturedSigmaInstances[0].graph).toBe(graph);

    // ready -> error: the space's community read starts failing, so the
    // renderer must fall back to the client-side climb again — two names back.
    mockListCommunities.mockRejectedValue(new Error("connection reset"));
    await qc.invalidateQueries({ queryKey: ["constellation-cartography"] });

    await waitFor(() => expect(painter.state.regionNames).toEqual(["Alice", "Bob"]));
    expect(capturedSigmaInstances).toHaveLength(1);
  });

  it("aggregates worst-first across spaces: one failed space taints the badge even though another is ready", async () => {
    const entities = [
      makeEntity({ id: "e1", domain: "wenlan-dev" }),
      makeEntity({ id: "e2", domain: "personal" }),
    ];
    mockListEntities.mockResolvedValue(entities);
    mockGetEntityDetail.mockImplementation(async (id: string) => ({
      entity: entities.find((e) => e.id === id)!,
      observations: [],
      relations: [],
    }));
    const ready = readyCommunityPages("wenlan-dev");
    mockListCommunities.mockImplementation(async (space: string) =>
      space === "wenlan-dev" ? ready.communities : Promise.reject(new Error("connection reset")),
    );
    mockListCommunityMembers.mockImplementation(async (space: string) =>
      space === "wenlan-dev"
        ? ready.members
        : { schema_version: "community-read-v1" as const, members: [], next_cursor: null },
    );

    renderWithQuery(<AtlasView />);
    await waitFor(() => expect(capturedSigmaInstances).toHaveLength(1));

    expect(await screen.findByText("Region sync issue")).toBeInTheDocument();
  });

  it("renders the badge text in zh-Hans and zh-Hant, not just English", async () => {
    mockOneSpaceEntity();
    const pages = readyCommunityPages("wenlan-dev");
    mockListCommunities.mockResolvedValue(pages.communities);
    mockListCommunityMembers.mockResolvedValue(pages.members);

    await i18n.changeLanguage("zh-Hans");
    const simplified = renderWithQuery(<AtlasView />);
    await waitFor(() => expect(capturedSigmaInstances).toHaveLength(1));
    expect(await screen.findByText("稳定区域")).toBeInTheDocument();
    simplified.unmount();
    capturedSigmaInstances.length = 0;

    await i18n.changeLanguage("zh-Hant");
    renderWithQuery(<AtlasView />);
    await waitFor(() => expect(capturedSigmaInstances).toHaveLength(1));
    expect(await screen.findByText("穩定區域")).toBeInTheDocument();

    await i18n.changeLanguage("en");
  });

  it("jumps the camera instantly on search select under prefers-reduced-motion", async () => {
    const matchMediaMock = vi.fn().mockReturnValue({ matches: true } as MediaQueryList);
    vi.stubGlobal("matchMedia", matchMediaMock);
    try {
      mockConnectedPair();

      renderWithQuery(<AtlasView />);
      await waitFor(() => expect(capturedSigmaInstances).toHaveLength(1));
      const instance = capturedSigmaInstances[0];

      const input = screen.getByPlaceholderText("Jump to anything…");
      fireEvent.focus(input);
      fireEvent.change(input, { target: { value: "alice" } });
      fireEvent.keyDown(input, { key: "Enter" });

      expect(instance.camera.setState).toHaveBeenCalledWith({ x: 0.42, y: 0.24, ratio: 1 });
      expect(instance.camera.animate).not.toHaveBeenCalled();
    } finally {
      vi.unstubAllGlobals();
    }
  });

  it("lands centered on focusEntityId at mount with emphasis, never animating", async () => {
    mockConnectedPairWithIsolate();

    // Baseline: an unfocused mount, for the reducer comparison below.
    const attrs = { color: "#abc", size: 4, entityType: "concept", confirmed: false };
    const { unmount } = renderWithQuery(<AtlasView />);
    await waitFor(() => expect(capturedSigmaInstances).toHaveLength(1));
    const plain = capturedSigmaInstances[0].settings.nodeReducer("e3", attrs);
    unmount();
    capturedSigmaInstances.length = 0;

    renderWithQuery(<AtlasView focusEntityId="e1" />);
    await waitFor(() => expect(capturedSigmaInstances).toHaveLength(1));
    const instance = capturedSigmaInstances[0];

    // First frame, not a transition: setState with the display coords.
    expect(instance.camera.setState).toHaveBeenCalledWith({ x: 0.42, y: 0.24, ratio: 1 });
    expect(instance.camera.animate).not.toHaveBeenCalled();

    // e3 is outside e1's neighborhood — the mounted reducer already dims it.
    expect(instance.settings.nodeReducer("e3", attrs)).not.toEqual(plain);
  });

  it("draws the small groups when focusEntityId sits in one, so the focus lands instead of silently vanishing", async () => {
    mockStarWithSmallPair();

    // A core entity as focus: the small pair stays hidden as usual.
    const { unmount } = renderWithQuery(<AtlasView focusEntityId="e1" />);
    await waitFor(() => expect(capturedSigmaInstances).toHaveLength(1));
    expect(latestGraph().hasNode("p1")).toBe(false);
    unmount();
    capturedSigmaInstances.length = 0;

    renderWithQuery(<AtlasView focusEntityId="p1" />);
    await waitFor(() => expect(capturedSigmaInstances).toHaveLength(1));
    const instance = capturedSigmaInstances[0];
    expect(instance.graph.hasNode("p1")).toBe(true);
    expect(instance.graph.hasNode("p2")).toBe(true);
    expect(instance.camera.setState).toHaveBeenCalledWith({ x: 0.42, y: 0.24, ratio: 1 });
  });

  it("renders a Back toolbar button only when onBack is passed, and clicking it fires the callback", async () => {
    mockConnectedPairWithIsolate();

    const { unmount } = renderWithQuery(<AtlasView />);
    await waitFor(() => expect(capturedSigmaInstances).toHaveLength(1));
    expect(screen.queryByRole("button", { name: "Back" })).not.toBeInTheDocument();
    unmount();
    capturedSigmaInstances.length = 0;

    const onBack = vi.fn();
    renderWithQuery(<AtlasView onBack={onBack} />);
    await waitFor(() => expect(capturedSigmaInstances).toHaveLength(1));
    fireEvent.click(screen.getByRole("button", { name: "Back" }));
    expect(onBack).toHaveBeenCalledTimes(1);
  });

  it("drags a leaf memory by direct manipulation, without reheating the sim", async () => {
    // A memory with exactly one link is a satellite, not a sim member (see
    // nonSimulatedIds in atlas.ts): a single link has one rest position, so
    // simulating it buys nothing. Dragging one still moves it — the
    // mousemovebody handler writes straight onto the graph — but there is no
    // pin to set and nothing to reheat.
    withMemoryLayerOn();
    mockConnectedPairWithMemory();

    renderWithQuery(<AtlasView />);
    await waitFor(() => expect(capturedSigmaInstances).toHaveLength(1));
    const instance = capturedSigmaInstances[0];
    const graph = instance.graph;
    const mouseCaptor = instance.getMouseCaptor();
    const sim = (window as any).__ATLAS_SIM;
    const restartSpy = vi.spyOn(sim, "restart");

    expect(graph.hasNode("mem:m1")).toBe(true);
    expect(sim.nodes().some((n: { id: string }) => n.id === "mem:m1")).toBe(false);
    const before = {
      x: graph.getNodeAttribute("mem:m1", "x"),
      y: graph.getNodeAttribute("mem:m1", "y"),
    };

    instance.handlers.get("downNode")?.({ node: "mem:m1" });
    expect(restartSpy).not.toHaveBeenCalled();

    mouseCaptor.handlers.get("mousemovebody")?.(dragEvent(500, 500));

    const after = {
      x: graph.getNodeAttribute("mem:m1", "x"),
      y: graph.getNodeAttribute("mem:m1", "y"),
    };
    expect(after).not.toEqual(before);
    expect(after).toEqual({ x: 500, y: 500 });
  });

  it("holds a dragged leaf memory in place while the sim keeps ticking", async () => {
    // The satellite orbit is rewritten on every writeback, so without telling
    // the sim which node the pointer holds, a warm tick would snap the leaf
    // straight back onto its anchor's ring mid-drag.
    withMemoryLayerOn();
    mockConnectedPairWithMemory();

    renderWithQuery(<AtlasView />);
    await waitFor(() => expect(capturedSigmaInstances).toHaveLength(1));
    const instance = capturedSigmaInstances[0];
    const graph = instance.graph;
    const mouseCaptor = instance.getMouseCaptor();
    const sim = (window as any).__ATLAS_SIM;

    instance.handlers.get("downNode")?.({ node: "mem:m1" });
    mouseCaptor.handlers.get("mousemovebody")?.(dragEvent(500, 500));
    sim.alpha(0.3);
    sim.tick(3);

    expect(graph.getNodeAttribute("mem:m1", "x")).toBe(500);
    expect(graph.getNodeAttribute("mem:m1", "y")).toBe(500);

    // Released, it belongs to its anchor again.
    mouseCaptor.handlers.get("mouseup")?.({});
    sim.tick(1);
    expect(graph.getNodeAttribute("mem:m1", "x")).not.toBe(500);
  });

  it("carries a leaf memory along when its anchor entity is dragged", async () => {
    // The flip side of the satellite rule: the orbit is rewritten on every
    // tick writeback, so a memory follows the entity it hangs off instead of
    // being dragged around on its own.
    withMemoryLayerOn();
    mockConnectedPairWithMemory();

    renderWithQuery(<AtlasView />);
    await waitFor(() => expect(capturedSigmaInstances).toHaveLength(1));
    const instance = capturedSigmaInstances[0];
    const graph = instance.graph;
    const mouseCaptor = instance.getMouseCaptor();
    const sim = (window as any).__ATLAS_SIM;

    instance.handlers.get("downNode")?.({ node: "e1" });
    mouseCaptor.handlers.get("mousemovebody")?.(dragEvent(500, 500));
    sim.tick(1);

    const anchorRadius = (graph.getNodeAttribute("e1", "size") as number) + 10;
    const dx = (graph.getNodeAttribute("mem:m1", "x") as number) - 500;
    const dy = (graph.getNodeAttribute("mem:m1", "y") as number) - 500;
    expect(Math.hypot(dx, dy)).toBeCloseTo(anchorRadius, 6);
  });

  it("draws memories as nodes linked to their entities, counts them, and legends them", async () => {
    withMemoryLayerOn();
    mockConnectedPairWithMemory();

    renderWithQuery(<AtlasView />);
    await waitFor(() => expect(capturedSigmaInstances).toHaveLength(1));
    const graph = capturedSigmaInstances[0].graph;

    expect(graph.hasNode("mem:m1")).toBe(true);
    expect(graph.hasEdge("mem:m1", "e1")).toBe(true);
    expect(graph.getNodeAttribute("mem:m1", "label")).toBe("A decision I made");
    // Memories are drawn smaller than the smallest entity.
    expect(graph.getNodeAttribute("mem:m1", "size")).toBeLessThan(
      graph.getNodeAttribute("e1", "size") as number,
    );
    expect(screen.getByText("3 entities · 1 memory · 1 region")).toBeInTheDocument();
    expect(screen.getByText("Memory")).toBeInTheDocument();
  });

  it("routes a click on a memory node to the memory, not to an entity", async () => {
    mockConnectedPairWithMemory();
    const onNodeClick = vi.fn();

    renderWithQuery(<AtlasView onNodeClick={onNodeClick} />);
    await waitFor(() => expect(capturedSigmaInstances).toHaveLength(1));

    capturedSigmaInstances[0].handlers.get("clickNode")?.({ node: "mem:m1" });

    // The "mem:" prefix keeps memory ids from colliding with entity ids and
    // is stripped here, so the caller gets the memory's own source_id.
    expect(onNodeClick).toHaveBeenCalledWith({ kind: "memory", id: "m1" });
  });

  it("hides a memory whose only entity is filtered out by the space selector", async () => {
    withMemoryLayerOn();
    // Three Work entities, so the scoped map still has a component big enough
    // to draw (MIN_COMPONENT_SIZE in model.ts) once the memory is filtered out.
    const entities = [
      makeEntity({ id: "e1", name: "Alice", domain: "Work" }),
      makeEntity({ id: "e2", name: "Bob", domain: "Work" }),
      makeEntity({ id: "e3", name: "Carol", domain: "Work" }),
    ];
    mockListEntities.mockResolvedValue(entities);
    mockGetEntityDetail.mockImplementation(async (id: string) => ({
      entity: entities.find((entity) => entity.id === id)!,
      observations: [],
      relations:
        id === "e1"
          ? [
              {
                id: "rel-1",
                relation_type: "knows",
                direction: "outgoing" as const,
                entity_id: "e2",
                entity_name: "Bob",
                entity_type: "person",
                source_agent: null,
                created_at: Math.floor(Date.now() / 1000),
              },
              {
                id: "rel-2",
                relation_type: "knows",
                direction: "outgoing" as const,
                entity_id: "e3",
                entity_name: "Carol",
                entity_type: "person",
                source_agent: null,
                created_at: Math.floor(Date.now() / 1000),
              },
            ]
          : [],
    }));
    // The memory itself lives in another space, so a Work-scoped map must not
    // draw it even though the entity it links to is in scope.
    mockMemories(
      [makeMemory({ source_id: "m1", title: "Personal note", space: "Personal" })],
      [{ memory_id: "m1", entity_id: "e1" }],
    );

    renderWithQuery(<AtlasView />);
    await waitFor(() => expect(capturedSigmaInstances).toHaveLength(1));
    expect(capturedSigmaInstances[0].graph.hasNode("mem:m1")).toBe(true);

    fireEvent.change(screen.getByRole("combobox", { name: "Space" }), {
      target: { value: "Work" },
    });

    await waitFor(() => expect(capturedSigmaInstances.length).toBeGreaterThan(1));
    const scoped = capturedSigmaInstances[capturedSigmaInstances.length - 1].graph;
    expect(scoped.hasNode("mem:m1")).toBe(false);
    expect(scoped.hasEdge("e1", "e2")).toBe(true);
  });
  // The connected pair plus two wiki pages: Alpha links to Beta, Alpha is
  // about Alice, and both pages cite the one memory (so the memory-off map
  // can still join them with a synthetic shared-source edge).
  function mockPairWithPages() {
    const entities = mockConnectedPair();
    mockMemories(
      [makeMemory({ source_id: "m1", title: "A decision I made", memory_type: "decision" })],
      [{ memory_id: "m1", entity_id: "e1" }],
    );
    mockPages(
      [makePage({ id: "p1", title: "Alpha" }), makePage({ id: "p2", title: "Beta" })],
      [
        { from: { kind: "page", id: "p1" }, to: { kind: "page", id: "p2" }, link_type: "wikilink" },
        { from: { kind: "page", id: "p1" }, to: { kind: "entity", id: "e1" }, link_type: "about" },
        { from: { kind: "page", id: "p1" }, to: { kind: "memory", id: "m1" }, link_type: "cites" },
        { from: { kind: "page", id: "p2" }, to: { kind: "memory", id: "m1" }, link_type: "cites" },
      ],
    );
    return entities;
  }

  it("draws wiki pages by default and leaves memories out until the chip is lit", async () => {
    mockPairWithPages();

    renderWithQuery(<AtlasView />);
    await waitFor(() => expect(capturedSigmaInstances).toHaveLength(1));
    const graph = capturedSigmaInstances[0].graph;

    expect(graph.hasNode("page:p1")).toBe(true);
    expect(graph.hasNode("page:p2")).toBe(true);
    expect(graph.getNodeAttribute("page:p1", "label")).toBe("Alpha");
    // Memories are the off-by-default layer, so the cited memory is absent and
    // the two pages that cite it are joined directly instead.
    expect(graph.hasNode("mem:m1")).toBe(false);
    expect(graph.hasEdge("page:p1", "page:p2")).toBe(true);
    expect(screen.getByText("2 pages · 3 entities · 1 region")).toBeInTheDocument();
    expect(screen.getByText("Wiki page")).toBeInTheDocument();
  });

  it("gives each layer a chip whose pressed state matches what is drawn", async () => {
    mockPairWithPages();

    renderWithQuery(<AtlasView />);
    await waitFor(() => expect(capturedSigmaInstances).toHaveLength(1));

    const pages = screen.getByRole("button", { name: "Wiki pages" });
    const entities = screen.getByRole("button", { name: "Entities" });
    const memories = screen.getByRole("button", { name: "Memories" });
    expect(pages).toHaveAttribute("aria-pressed", "true");
    expect(entities).toHaveAttribute("aria-pressed", "true");
    expect(memories).toHaveAttribute("aria-pressed", "false");

    fireEvent.click(memories);
    await waitFor(() => expect(capturedSigmaInstances.length).toBeGreaterThan(1));
    const withMemories = capturedSigmaInstances[capturedSigmaInstances.length - 1].graph;
    expect(withMemories.hasNode("mem:m1")).toBe(true);
    expect(memories).toHaveAttribute("aria-pressed", "true");
    expect(screen.getByText("2 pages · 3 entities · 1 memory · 1 region")).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "Entities" }));
    await waitFor(() => {
      const graph = capturedSigmaInstances[capturedSigmaInstances.length - 1].graph;
      expect(graph.hasNode("e1")).toBe(false);
    });
    expect(screen.getByText("2 pages · 1 memory")).toBeInTheDocument();
  });

  it("refuses to turn off the last lit layer", async () => {
    mockPairWithPages();

    renderWithQuery(<AtlasView />);
    await waitFor(() => expect(capturedSigmaInstances).toHaveLength(1));

    fireEvent.click(screen.getByRole("button", { name: "Entities" }));
    const pages = screen.getByRole("button", { name: "Wiki pages" });
    // Pages are now the only lit layer: the chip stays pressed and stops
    // taking clicks, because an empty map is not a view.
    expect(pages).toHaveAttribute("aria-pressed", "true");
    expect(pages).toBeDisabled();
    fireEvent.click(pages);
    expect(pages).toHaveAttribute("aria-pressed", "true");
  });

  it("persists the layer choice and ignores a malformed stored value", async () => {
    mockPairWithPages();

    const first = renderWithQuery(<AtlasView />);
    await waitFor(() => expect(capturedSigmaInstances).toHaveLength(1));
    fireEvent.click(screen.getByRole("button", { name: "Memories" }));
    expect(JSON.parse(window.localStorage.getItem("atlas.layers")!)).toEqual({
      page: true,
      entity: true,
      memory: true,
    });

    first.unmount();
    capturedSigmaInstances.length = 0;
    renderWithQuery(<AtlasView />);
    await waitFor(() => expect(capturedSigmaInstances).toHaveLength(1));
    expect(capturedSigmaInstances[0].graph.hasNode("mem:m1")).toBe(true);

    window.localStorage.setItem("atlas.layers", "{not json");
    cleanup();
    capturedSigmaInstances.length = 0;
    renderWithQuery(<AtlasView />);
    await waitFor(() => expect(capturedSigmaInstances).toHaveLength(1));
    // Back to the default: pages and entities, no memories.
    expect(capturedSigmaInstances[0].graph.hasNode("mem:m1")).toBe(false);
    expect(capturedSigmaInstances[0].graph.hasNode("page:p1")).toBe(true);
  });

  it("routes a click on a page node to the page, not to an entity", async () => {
    mockPairWithPages();
    const onNodeClick = vi.fn();

    renderWithQuery(<AtlasView onNodeClick={onNodeClick} />);
    await waitFor(() => expect(capturedSigmaInstances).toHaveLength(1));

    capturedSigmaInstances[0].handlers.get("clickNode")?.({ node: "page:p1" });

    expect(onNodeClick).toHaveBeenCalledWith({ kind: "page", id: "p1" });
  });
});
