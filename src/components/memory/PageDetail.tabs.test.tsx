// SPDX-License-Identifier: AGPL-3.0-only
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, cleanup } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import PageDetail from "./PageDetail";

// Same reasoning as PageCanvas.test.tsx: React Flow needs real dimensions.
// These tests only care that the canvas tab mounts, so the node state hook is
// stubbed down to plain useState — selection and dragging are exercised in
// PageCanvas.test.tsx, not here.
vi.mock("@xyflow/react", async () => {
  const React = await import("react");
  return {
    ReactFlow: ({ nodes, nodeTypes }: any) => (
      <div data-testid="react-flow">
        {nodes.map((n: any) => {
          const NodeComponent = nodeTypes[n.type];
          return <NodeComponent key={n.id} id={n.id} data={n.data} />;
        })}
      </div>
    ),
    ReactFlowProvider: ({ children }: any) => <>{children}</>,
    Background: () => null,
    Controls: () => null,
    Handle: () => null,
    Position: { Top: "top", Bottom: "bottom", Left: "left", Right: "right" },
    useReactFlow: () => ({ getViewport: () => ({ x: 0, y: 0, zoom: 1 }) }),
    useNodesState: (initial: any) => {
      const [ns, setNs] = React.useState(initial);
      return [ns, setNs, () => {}];
    },
  };
});

vi.mock("../../lib/tauri", async (importOriginal) => ({
  ...(await importOriginal<typeof import("../../lib/tauri")>()),
  // Inlined on purpose: vi.mock factories are hoisted above every top-level
  // binding, so a shared fixture const would be in its temporal dead zone.
  getPage: vi.fn().mockResolvedValue({
    id: "concept_abc",
    title: "libSQL Architecture",
    summary: "Core database layer",
    content: "libSQL is the core database layer.\n\nMore prose here.",
    entity_id: null,
    domain: null,
    source_memory_ids: ["mem_1"],
    version: 3,
    status: "active",
    created_at: "2026-04-01T00:00:00+00:00",
    last_compiled: "2026-04-07T12:00:00+00:00",
    last_modified: "2026-04-07T12:00:00+00:00",
  }),
  getPageSources: vi.fn().mockResolvedValue([
    {
      source: { page_id: "concept_abc", memory_source_id: "mem_1", linked_at: 1, link_reason: "page_growth" },
      memory: {
        source_id: "mem_1",
        title: "libSQL stores vectors",
        content: "libSQL stores vectors in F32_BLOB columns",
        summary: null,
        memory_type: "fact",
        domain: null,
        source_agent: "claude",
        confidence: 0.9,
        confirmed: true,
        last_modified: 1,
      },
    },
  ]),
  getPageLinks: vi.fn().mockResolvedValue({ outbound: [], inbound: [] }),
  getPageRevisions: vi.fn().mockResolvedValue({ entries: [], user_edited: false }),
  listRegisteredSources: vi.fn().mockResolvedValue([]),
  getEntityDetail: vi.fn().mockResolvedValue(null),
  getPageMap: vi.fn().mockResolvedValue({
    page_id: "concept_abc",
    revision: 3,
    map_schema: 1,
    viewport: null,
    nodes: [
      {
        id: "n_root", parent_id: null, rank: 0, ref_kind: "page", ref_id: "concept_abc",
        label: null, status: "active", pinned: false, placed: false, collapsed: false,
        x: null, y: null, width: null, height: null, ref_state: "live",
      },
      {
        id: "n_mem", parent_id: "n_root", rank: 0, ref_kind: "memory", ref_id: "mem_1",
        label: null, status: "active", pinned: false, placed: false, collapsed: false,
        x: null, y: null, width: null, height: null, ref_state: "live",
      },
    ],
    edges: [],
  }),
}));

class ResizeObserverStub {
  observe() {}
  unobserve() {}
  disconnect() {}
}
vi.stubGlobal("ResizeObserver", ResizeObserverStub);

function renderDetail() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return {
    user: userEvent.setup(),
    ...render(
      <QueryClientProvider client={client}>
        <PageDetail
          pageId="concept_abc"
          onBack={vi.fn()}
          onMemoryClick={vi.fn()}
          onPageClick={vi.fn()}
          onEntityClick={vi.fn()}
        />
      </QueryClientProvider>,
    ),
  };
}

beforeEach(() => vi.clearAllMocks());
afterEach(() => cleanup());

describe("PageDetail canvas toggle", () => {
  it("opens reading with one unpressed Canvas control and no tab row", async () => {
    renderDetail();
    await screen.findByText("libSQL Architecture");

    // Reading is the default, so it gets no control of its own.
    expect(screen.queryAllByRole("tab")).toHaveLength(0);
    expect(screen.queryByRole("button", { name: "Read" })).toBeNull();
    expect(
      screen.getByRole("button", { name: "Canvas" }).getAttribute("aria-pressed"),
    ).toBe("false");
    expect(screen.getByText(/Page info/i)).toBeTruthy();
  });

  it("swaps the reading column and Page info for the canvas", async () => {
    const { user } = renderDetail();
    await screen.findByText("libSQL Architecture");

    await user.click(screen.getByRole("button", { name: "Canvas" }));

    expect(
      await screen.findByRole("region", { name: "Canvas for libSQL Architecture" }),
    ).toBeTruthy();
    // the page title stays in the header; the prose and Page info do not
    expect(screen.queryByText(/Page info/i)).toBeNull();
    expect(screen.queryByText("More prose here.")).toBeNull();
    expect(
      screen.getByRole("button", { name: "Canvas" }).getAttribute("aria-pressed"),
    ).toBe("true");
  });

  it("goes back to reading when the same control is pressed again", async () => {
    const { user } = renderDetail();
    await screen.findByText("libSQL Architecture");

    const toggle = screen.getByRole("button", { name: "Canvas" });
    await user.click(toggle);
    await screen.findByRole("region", { name: "Canvas for libSQL Architecture" });

    // Without a Read tab, this control is the only way back — if it does not
    // toggle, the canvas is a trap.
    await user.click(toggle);
    expect(
      screen.queryByRole("region", { name: "Canvas for libSQL Architecture" }),
    ).toBeNull();
    expect(await screen.findByText("More prose here.")).toBeTruthy();
    expect(toggle.getAttribute("aria-pressed")).toBe("false");
  });

  it("keeps a door to the canvas in the actions menu, which is the whole toolbar on a narrow window", async () => {
    const { user } = renderDetail();
    await screen.findByText("libSQL Architecture");

    // Under 600px the icon row is display:none and this menu is all that is
    // left, so an icon-only control is no control at all there.
    await user.click(screen.getByRole("button", { name: "Page actions" }));
    await user.click(screen.getByRole("menuitem", { name: "Canvas" }));
    await screen.findByRole("region", { name: "Canvas for libSQL Architecture" });

    await user.click(screen.getByRole("button", { name: "Page actions" }));
    await user.click(screen.getByRole("menuitem", { name: "Close canvas" }));
    expect(
      screen.queryByRole("region", { name: "Canvas for libSQL Architecture" }),
    ).toBeNull();
  });

  it("sits with the other page controls, not in a band of its own", async () => {
    renderDetail();
    await screen.findByText("libSQL Architecture");

    const toggle = screen.getByRole("button", { name: "Canvas" });
    const cluster = toggle.closest(".page-detail-icon-actions");
    expect(cluster).toBeTruthy();
    // The controls it was asked to join.
    expect(cluster?.querySelector('[title="Edit page"]')).toBeTruthy();
  });

  it("resolves the root node label from the page title it already loaded", async () => {
    const { user } = renderDetail();
    await screen.findByText("libSQL Architecture");
    await user.click(screen.getByRole("button", { name: "Canvas" }));

    // Both nodes arrive with label: null — the daemon stores refs, the client
    // renders the backing objects PageDetail already has in hand.
    await screen.findByTestId("react-flow");
    expect(screen.getAllByText("libSQL Architecture").length).toBeGreaterThan(1);
    expect(screen.getByText("libSQL stores vectors")).toBeTruthy();
  });

  it("hides the canvas control while editing so nobody types into an unseen page", async () => {
    const { user } = renderDetail();
    await screen.findByText("libSQL Architecture");

    await user.click(screen.getByRole("button", { name: "Canvas" }));
    await screen.findByRole("region", { name: "Canvas for libSQL Architecture" });

    await user.click(screen.getByTitle("Edit page"));
    expect(screen.queryByRole("button", { name: "Canvas" })).toBeNull();
    // Edit mode engaged: the whole header actions row goes with it. Asserted
    // through the row rather than through a field, because what the editor
    // itself resolves to is the editor's business — under this env's daemon
    // check it is the page-editor floor notice, not a text box.
    expect(screen.queryByTitle("Edit page")).toBeNull();
    expect(screen.queryByRole("region", { name: "Canvas for libSQL Architecture" })).toBeNull();
  });
});
