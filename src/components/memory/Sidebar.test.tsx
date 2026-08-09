// SPDX-License-Identifier: AGPL-3.0-only
import { beforeEach, describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import type { ComponentProps } from "react";
import { i18n } from "../../i18n";
import { RECENT_PAGES_STORAGE_KEY } from "../../lib/recentPages";
import { RECENT_SPACES_STORAGE_KEY } from "../../lib/recentSpaces";
import type { Page, Space } from "../../lib/tauri";
import Sidebar, { SidebarHeaderDivider, SidebarToggleButton } from "./Sidebar";

const { listAgentsMock, listAllActivePagesMock, listSpacesMock } = vi.hoisted(() => ({
  listAgentsMock: vi.fn().mockResolvedValue([]),
  listAllActivePagesMock: vi.fn().mockResolvedValue([]),
  listSpacesMock: vi.fn().mockResolvedValue([]),
}));

vi.mock("../../lib/tauri", () => ({
  getMemoryStats: vi.fn().mockResolvedValue({ total: 0, new_today: 0, confirmed: 0, domains: [] }),
  listAgents: listAgentsMock,
  listSpaces: listSpacesMock,
}));

vi.mock("./pages/listAllPages", () => ({
  listAllActivePages: listAllActivePagesMock,
}));

vi.mock("./IdentityCard", async () => {
  const { useState } = await vi.importActual<typeof import("react")>("react");

  return {
    default: ({
      onOpenAbout,
      onOpenDetail,
      onOpenSettings,
    }: {
      readonly onOpenAbout?: () => void;
      readonly onOpenDetail: (entityId: string) => void;
      readonly onOpenSettings?: () => void;
    }) => {
      const [menuOpen, setMenuOpen] = useState(false);
      return (
        <div data-testid="identity-card">
          <button type="button" aria-expanded={menuOpen} onClick={() => setMenuOpen((open) => !open)}>Identity menu</button>
          <button type="button" onClick={() => onOpenDetail("person-1")}>Open identity detail</button>
          <button type="button" onClick={() => onOpenSettings?.()}>Open identity settings</button>
          <button type="button" onClick={() => onOpenAbout?.()}>Open identity about</button>
        </div>
      );
    },
  };
});
vi.mock("./SpaceList", () => ({ default: () => <div data-testid="space-list">Spaces</div> }));
vi.mock("./EntitySuggestions", () => ({ default: () => <div data-testid="entity-suggestions" /> }));
vi.mock("./RecentSpaces", () => ({
  RecentSpaces: ({ onSelectSpace, spaces }: { readonly onSelectSpace: (space: Space) => void; readonly spaces: readonly Space[] }) => (
    <nav aria-label="Recent spaces" data-count={spaces.length}>
      {spaces.map((space) => <button key={space.id} onClick={() => onSelectSpace(space)}>{space.name}</button>)}
    </nav>
  ),
}));
vi.mock("./RecentPages", () => ({
  RecentPages: ({ onSelectPage, pages }: { readonly onSelectPage: (page: Page) => void; readonly pages: readonly Page[] }) => (
    <nav aria-label="Recent pages" data-count={pages.length}>
      {pages.map((page) => <button key={page.id} onClick={() => onSelectPage(page)}>{page.title}</button>)}
    </nav>
  ),
}));

function page(id: string, title: string): Page {
  return {
    id,
    title,
    summary: null,
    content: "",
    entity_id: null,
    domain: null,
    source_memory_ids: [],
    version: 1,
    status: "active",
    created_at: "2026-07-16T00:00:00Z",
    last_compiled: "2026-07-16T00:00:00Z",
    last_modified: "2026-07-16T00:00:00Z",
  };
}

function renderSidebar(extraProps: Partial<ComponentProps<typeof Sidebar>> = {}) {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={queryClient}>
      <Sidebar
        collapsed={false}
        onSelectSpace={() => {}}
        onEntityClick={() => {}}
        onNavigateHome={() => {}}
        onNavigateLog={() => {}}
        onNavigateGraph={() => {}}
        onNavigatePages={() => {}}
        onNavigateSpaces={() => {}}
        onSelectPage={() => {}}
        {...extraProps}
      />
    </QueryClientProvider>,
  );
}

describe("Sidebar", () => {
  beforeEach(async () => {
    listAgentsMock.mockResolvedValue([]);
    listAllActivePagesMock.mockResolvedValue([]);
    listSpacesMock.mockResolvedValue([]);
    localStorage.clear();
    await i18n.changeLanguage("en");
  });

  it("places Home in the primary nav and routes it through onNavigateHome", async () => {
    const user = userEvent.setup();
    const onNavigateHome = vi.fn();
    renderSidebar({ onNavigateHome });

    const home = screen.getByRole("button", { name: "Home" });
    const spaces = screen.getByRole("button", { name: "Spaces" });

    expect(home.compareDocumentPosition(spaces) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();

    await user.click(home);

    expect(onNavigateHome).toHaveBeenCalledTimes(1);
  });

  it("requests overlay closure after every successful primary or Recent navigation", async () => {
    // Given an open overlay with one actually visited Page and Space
    const recentPage = page("page-1", "Notes");
    const recentSpace: Space = {
      id: "space-1",
      name: "Work",
      description: null,
      suggested: false,
      starred: false,
      sort_order: 0,
      memory_count: 1,
      entity_count: 0,
      created_at: 0,
      updated_at: 1,
    };
    listAllActivePagesMock.mockResolvedValue([recentPage]);
    listSpacesMock.mockResolvedValue([recentSpace]);
    localStorage.setItem(RECENT_PAGES_STORAGE_KEY, JSON.stringify({
      version: 1,
      entries: [{ id: recentPage.id, title: recentPage.title, visitedAt: Date.now() - 2 }],
    }));
    localStorage.setItem(RECENT_SPACES_STORAGE_KEY, JSON.stringify({
      version: 1,
      entries: [{ id: recentSpace.id, name: recentSpace.name, visitedAt: Date.now() - 1 }],
    }));
    const user = userEvent.setup();
    const onNavigateHome = vi.fn();
    const onNavigateLog = vi.fn();
    const onNavigateGraph = vi.fn();
    const onNavigateSources = vi.fn();
    const onNavigateSpaces = vi.fn();
    const onSelectPage = vi.fn();
    const onSelectSpace = vi.fn();
    const onRequestClose = vi.fn();
    renderSidebar({
      onNavigateGraph,
      onNavigateHome,
      onNavigateLog,
      onNavigateSources,
      onNavigateSpaces,
      onRequestClose,
      onSelectPage,
      onSelectSpace,
      open: true,
      presentation: "overlay",
    });

    // When each navigation control succeeds
    await user.click(screen.getByRole("button", { name: "Home" }));
    await user.click(screen.getByRole("button", { name: "Memories" }));
    await user.click(screen.getByRole("button", { name: "Graph" }));
    await user.click(screen.getByRole("button", { name: "Sources" }));
    await user.click(screen.getByRole("button", { name: "Spaces" }));
    await user.click(await screen.findByRole("button", { name: "Notes" }));
    await user.click(await screen.findByRole("button", { name: "Work" }));

    // Then every route callback runs and each one requests drawer closure
    expect(onNavigateHome).toHaveBeenCalledTimes(1);
    expect(onNavigateLog).toHaveBeenCalledTimes(1);
    expect(onNavigateGraph).toHaveBeenCalledTimes(1);
    expect(onNavigateSources).toHaveBeenCalledTimes(1);
    expect(onNavigateSpaces).toHaveBeenNthCalledWith(1, false);
    expect(onSelectPage).toHaveBeenCalledWith(recentPage);
    expect(onSelectSpace).toHaveBeenCalledWith(recentSpace);
    expect(onRequestClose).toHaveBeenCalledTimes(7);
  });

  it.each([
    ["detail", "Open identity detail", "onEntityClick", "person-1"],
    ["Settings", "Open identity settings", "onNavigateSettings", undefined],
    ["About", "Open identity about", "onOpenAbout", undefined],
  ] as const)("requests overlay closure after IdentityCard %s navigation", async (_label, buttonName, callbackProp, expectedArgument) => {
    // Given an open overlay with a wired IdentityCard destination
    const user = userEvent.setup();
    const callback = vi.fn();
    const onRequestClose = vi.fn();
    renderSidebar({
      [callbackProp]: callback,
      onRequestClose,
      open: true,
      presentation: "overlay",
    });

    // When the identity destination is selected
    await user.click(screen.getByRole("button", { name: buttonName }));

    // Then the intended destination runs before the drawer requests closure
    expectedArgument === undefined
      ? expect(callback).toHaveBeenCalledWith()
      : expect(callback).toHaveBeenCalledWith(expectedArgument);
    expect(onRequestClose).toHaveBeenCalledTimes(1);
    expect(callback.mock.invocationCallOrder[0]).toBeLessThan(onRequestClose.mock.invocationCallOrder[0]);
  });

  it("keeps the overlay open for IdentityCard menu interaction", async () => {
    // Given an open overlay with the identity menu closed
    const user = userEvent.setup();
    const onRequestClose = vi.fn();
    renderSidebar({ onRequestClose, open: true, presentation: "overlay" });
    const menu = screen.getByRole("button", { name: "Identity menu" });
    expect(menu).toHaveAttribute("aria-expanded", "false");

    // When the identity menu is opened without navigating
    await user.click(menu);

    // Then the menu action still works and the drawer remains open
    expect(menu).toHaveAttribute("aria-expanded", "true");
    expect(onRequestClose).not.toHaveBeenCalled();
  });

  it("labels the memory view as Memories and keeps Decisions out of the left nav", () => {
    renderSidebar();

    expect(screen.getByRole("button", { name: "Memories" })).toBeInTheDocument();
    expect(screen.queryByText("Memory Log")).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Decisions" })).not.toBeInTheDocument();
  });

  it("uses the approved Home, Wiki, Spaces, Graph, Memories, Sources order", () => {
    renderSidebar({ onNavigateSources: () => {} });

    const destinations = ["Home", "Wiki", "Spaces", "Graph", "Memories", "Sources"].map((name) => screen.getByRole("button", { name }));
    for (const [index, destination] of destinations.entries()) {
      const next = destinations[index + 1];
      if (next) expect(destination.compareDocumentPosition(next) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
    }
  });

  it("replaces inline SpaceList management with one full-width Spaces destination", async () => {
    // Given the integrated global sidebar
    const user = userEvent.setup();
    const onNavigateSpaces = vi.fn();
    renderSidebar({ onNavigateSpaces });

    // When the Spaces destination is used
    await user.click(screen.getByRole("button", { name: "Spaces" }));

    // Then creation stays inside the overview and no management list remains
    expect(onNavigateSpaces).toHaveBeenNthCalledWith(1, false);
    expect(screen.queryByRole("button", { name: "New space" })).not.toBeInTheDocument();
    expect(screen.queryByTestId("space-list")).toBeNull();
  });

  it("places only visited Recent Pages then Recent Spaces beneath Sources without a suggestion badge", async () => {
    // Given one visited Page, one visited Space, and an unrelated suggestion
    const recentPage = page("page-1", "Notes");
    const recentSpace: Space = {
      id: "confirmed",
      name: "Work",
      description: null,
      suggested: false,
      starred: false,
      sort_order: 0,
      memory_count: 0,
      entity_count: 0,
      created_at: 0,
      updated_at: 0,
    };
    listAllActivePagesMock.mockResolvedValue([recentPage]);
    listSpacesMock.mockResolvedValue([
      { id: "suggested", suggested: true },
      recentSpace,
    ]);
    localStorage.setItem(RECENT_PAGES_STORAGE_KEY, JSON.stringify({
      version: 1,
      entries: [{ id: recentPage.id, title: recentPage.title, visitedAt: Date.now() - 2 }],
    }));
    localStorage.setItem(RECENT_SPACES_STORAGE_KEY, JSON.stringify({
      version: 1,
      entries: [{ id: recentSpace.id, name: recentSpace.name, visitedAt: Date.now() - 1 }],
    }));

    // When the sidebar renders on a standard view
    renderSidebar({ onNavigateSources: () => {} });

    // Then only the contextual histories follow Sources, in Page-first order
    const recentPages = await screen.findByRole("navigation", { name: "Recent pages" });
    const recentSpaces = await screen.findByRole("navigation", { name: "Recent spaces" });
    const sources = screen.getByRole("button", { name: "Sources" });
    expect(sources.compareDocumentPosition(recentPages) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
    expect(recentPages.compareDocumentPosition(recentSpaces) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
    expect(screen.getByRole("button", { name: "Spaces" })).toHaveTextContent(/^Spaces$/);
    expect(screen.queryByText(/^1$/)).not.toBeInTheDocument();
  });

  it("omits both Recent groups when no destination has been visited", async () => {
    listAllActivePagesMock.mockResolvedValue([page("page-1", "Notes")]);
    listSpacesMock.mockResolvedValue([{ id: "space-1", name: "Work", suggested: false }]);

    renderSidebar();

    expect(await screen.findByRole("button", { name: "Spaces" })).toBeInTheDocument();
    expect(screen.queryByRole("navigation", { name: "Recent pages" })).toBeNull();
    expect(screen.queryByRole("navigation", { name: "Recent spaces" })).toBeNull();
  });

  it("exposes exactly one current global destination", () => {
    // Given the Space family is active
    renderSidebar({ activeNavigation: "spaces" });

    // Then only the global Spaces destination is current
    const current = screen.getAllByRole("button").filter((button) => button.getAttribute("aria-current") === "page");
    expect(current).toHaveLength(1);
    expect(current[0]).toHaveAccessibleName("Spaces");
    expect(current[0]).toHaveStyle({ fontWeight: "500" });
    const activeMarkers = document.querySelectorAll('[data-primary-navigation-active-marker="true"]');
    expect(activeMarkers).toHaveLength(1);
    expect(activeMarkers[0]).toHaveAttribute("aria-hidden", "true");

    for (const name of ["Home", "Wiki", "Memories", "Graph"]) {
      const inactive = screen.getByRole("button", { name });
      expect(inactive).not.toHaveStyle({ fontWeight: "500" });
      expect(inactive.querySelector('[data-primary-navigation-active-marker="true"]')).toBeNull();
    }
  });

  it("keeps Sources in the primary nav and routes it through onNavigateSources", async () => {
    const user = userEvent.setup();
    const onNavigateSources = vi.fn();
    renderSidebar({ onNavigateSources });

    const sources = screen.getByRole("button", { name: "Sources" });
    const memories = screen.getByRole("button", { name: "Memories" });

    expect(memories.compareDocumentPosition(sources) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();

    await user.click(sources);
    expect(onNavigateSources).toHaveBeenCalledTimes(1);
  });

  it("omits the Sources tab when source navigation is not wired", () => {
    renderSidebar();

    expect(screen.queryByRole("button", { name: "Sources" })).not.toBeInTheDocument();
  });

  it("uses the account card as the only sidebar footer item", () => {
    renderSidebar();

    const spaces = screen.getByRole("button", { name: "Spaces" });
    const account = screen.getByTestId("identity-card");

    expect(spaces.compareDocumentPosition(account) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
    expect(screen.queryByRole("button", { name: "Wenlan" })).not.toBeInTheDocument();
  });

  it("keeps the account card as the footer after locale changes", async () => {
    await i18n.changeLanguage("zh-Hant");
    renderSidebar();

    expect(screen.getByTestId("identity-card")).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Wenlan 文瀾" })).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Wenlan" })).not.toBeInTheDocument();
  });

  it("does not show agent connection status in the footer", async () => {
    listAgentsMock.mockResolvedValue([
      { id: "agent-1", name: "Codex", enabled: true },
      { id: "agent-2", name: "Claude", enabled: true },
    ]);

    renderSidebar();

    expect(await screen.findByTestId("identity-card")).toBeInTheDocument();
    expect(screen.queryByText(/agents? connected/i)).not.toBeInTheDocument();
    expect(screen.queryByText(/No agents connected/i)).not.toBeInTheDocument();
  });

  it("keeps the sidebar toggle on the shared header centerline", () => {
    render(<SidebarToggleButton collapsed={false} onToggle={() => {}} />);

    const toggle = screen.getByTitle("Hide sidebar");
    expect(toggle).toHaveAttribute("data-sidebar-toggle", "true");
    expect(toggle.style.alignSelf).toBe("");
    expect(toggle.style.marginTop).toBe("");
  });

  it("continues the desktop sidebar divider through the full header", () => {
    const { container, rerender } = render(<SidebarHeaderDivider visible />);

    expect(container.querySelector('[data-sidebar-header-divider="true"]')).toHaveStyle({
      backgroundColor: "var(--mem-border)",
      height: "52px",
      left: "239px",
      position: "absolute",
      top: "0px",
      width: "1px",
    });

    rerender(<SidebarHeaderDivider visible={false} />);
    expect(container.querySelector('[data-sidebar-header-divider="true"]')).not.toBeInTheDocument();
  });
});
