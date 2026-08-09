// SPDX-License-Identifier: AGPL-3.0-only
import { beforeEach, describe, expect, it, vi } from "vitest";
import { act, fireEvent, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { useState, type ComponentProps } from "react";
import { i18n } from "../../i18n";
import { RECENT_PAGES_STORAGE_KEY } from "../../lib/recentPages";
import { RECENT_SPACES_STORAGE_KEY } from "../../lib/recentSpaces";
import type { Page, SearchResult, Space } from "../../lib/tauri";
import Main from "./Main";

const eventListeners = vi.hoisted(
  () => new Map<string, (payload?: unknown) => void>(),
);
const listSpacesMock = vi.hoisted(() => vi.fn<() => Promise<readonly Space[]>>());
const openFileMock = vi.hoisted(() => vi.fn<(url: string) => Promise<void>>());
const draftRequestBackMock = vi.hoisted(
  () => vi.fn<(onBack: () => void) => Promise<void>>(),
);
const draftFlushMock = vi.hoisted(
  () => vi.fn<() => Promise<boolean>>(),
);
const draftIdentityMock = vi.hoisted(
  () => vi.fn<() => { readonly draftId: string | null; readonly version: number | null }>(),
);
const setSearchQueryMock = vi.hoisted(() => vi.fn());
const useSearchMock = vi.hoisted(() => vi.fn(() => ({
  query: "",
  setQuery: setSearchQueryMock,
  results: [] as SearchResult[],
})));

vi.mock("@tauri-apps/api/event", () => ({
  listen: vi.fn((event: string, handler: (payload?: unknown) => void) => {
    eventListeners.set(event, handler);
    return Promise.resolve(() => eventListeners.delete(event));
  }),
}));

vi.mock("../../hooks/useSearch", () => ({
  useSearch: useSearchMock,
}));

vi.mock("../../lib/tauri", () => ({
  listMemoriesRich: vi.fn().mockResolvedValue([]),
  getMemoryStats: vi.fn().mockResolvedValue({ total: 0, new_today: 0, confirmed: 0, domains: [] }),
  searchEntities: vi.fn().mockResolvedValue([]),
  searchPages: vi.fn().mockResolvedValue([]),
  listSpaces: listSpacesMock,
  openFile: openFileMock,
  deleteFileChunks: vi.fn().mockResolvedValue(undefined),
}));

vi.mock("./ActivityFeed", () => ({ default: () => <div data-testid="activity-feed" /> }));
vi.mock("./IdentityDetail", () => ({ default: () => <div /> }));
vi.mock("./MemoryStream", () => ({ default: () => <div /> }));
vi.mock("./HomePage", () => ({
  default: (props: { onNavigateGraph?: () => void }) => (
    <div data-testid="home-page">
      <button type="button" onClick={() => props.onNavigateGraph?.()}>
        Open graph view
      </button>
    </div>
  ),
}));
vi.mock("./AtlasView", () => ({
  default: (props: { onBack?: () => void; onNodeClick?: (id: string) => void }) => (
    <div data-testid="atlas-view">
      <button type="button" onClick={() => props.onBack?.()}>
        Atlas back
      </button>
      <button type="button" onClick={() => props.onNodeClick?.("ent-1")}>
        Atlas node
      </button>
    </div>
  ),
}));
vi.mock("./EntityDetail", () => ({ default: () => <div data-testid="entity-detail" /> }));
vi.mock("./MemoryStatusBar", () => ({ default: () => <div /> }));
vi.mock("./MemorySearchResult", () => ({
  default: (props: { result: { content: string; source_id: string }; onClick?: (sourceId: string) => void }) => (
    <button type="button" onClick={() => props.onClick?.(props.result.source_id)}>
      {props.result.content}
    </button>
  ),
}));
vi.mock("./MemoryDetail", () => ({ default: () => <div data-testid="memory-detail" /> }));
vi.mock("./PageDetail", () => ({
  default: (props: {
    onBack?: () => void;
    onPageLoaded?: (page: Pick<Page, "id" | "status" | "title">) => void;
    pageId: string;
  }) => (
    <div
      data-page-id={props.pageId}
      data-testid="page-detail"
    >
      <button
        onClick={() => props.onPageLoaded?.({ id: props.pageId, status: "active", title: "Visited page" })}
        type="button"
      >
        Finish loading page
      </button>
      <button onClick={props.onBack} type="button">Page back</button>
    </div>
  ),
}));
vi.mock("./DistillReviewPanel", () => ({ default: () => <div /> }));
vi.mock("./SettingsPage", () => ({
  default: (props: { onSetupAgent?: () => void }) => (
    <div data-testid="settings-page">
      <button type="button" onClick={props.onSetupAgent}>Connect agent</button>
    </div>
  ),
}));
vi.mock("../SetupWizard", () => ({ SetupWizard: () => <div /> }));
vi.mock("../ViewToggle", () => ({ default: () => <div /> }));
vi.mock("./Sidebar", () => ({
  default: (props: {
    activeNavigation?: string | null;
    collapsed: boolean;
    currentPageId?: string | null;
    currentSpaceId?: string | null;
    open?: boolean;
    onEntityClick: (entityId: string) => void;
    onNavigateLog?: () => void;
    onNavigateGraph?: () => void;
    onNavigatePages?: () => void;
    onNavigateSpaces?: (create: boolean) => void;
    onRequestClose?: () => void;
    onSelectPage?: (page: Page) => void;
    onSelectSpace?: (space: { readonly id: string; readonly name: string }) => void;
    presentation?: string;
  }) => (
    <aside
      data-active={props.activeNavigation ?? "none"}
      data-collapsed={props.collapsed ? "true" : "false"}
      data-current-page={props.currentPageId ?? "none"}
      data-current-space={props.currentSpaceId ?? "none"}
      data-open={props.open ? "true" : "false"}
      data-presentation={props.presentation ?? "desktop"}
    >
      <button type="button" onClick={() => props.onEntityClick("__create_profile__")}>
        Open avatar menu destination
      </button>
      <button type="button" onClick={props.onNavigateLog}>Open memories</button>
      <button type="button" onClick={props.onNavigatePages}>Open wiki</button>
      <button type="button" onClick={props.onNavigateGraph}>Open graph</button>
      <button type="button" onClick={() => props.onNavigateSpaces?.(false)}>Open spaces</button>
      <button type="button" onClick={() => props.onNavigateSpaces?.(true)}>Create space</button>
      <button type="button" onClick={() => props.onSelectPage?.({ id: "page-1", status: "active", title: "Recent page" } as Page)}>
        Open recent page
      </button>
      <button type="button" onClick={() => props.onSelectSpace?.({ id: "space-1", name: "Work" })}>
        Open recent space
      </button>
      {props.presentation === "overlay" && props.open && (
        <button type="button" aria-label="Close sidebar" onClick={props.onRequestClose} />
      )}
    </aside>
  ),
  SidebarToggleButton: (props: { onToggle: () => void; ref?: React.Ref<HTMLButtonElement> }) => (
    <button ref={props.ref} type="button" aria-label="Toggle sidebar" onClick={props.onToggle} />
  ),
  SidebarHeaderDivider: (props: { visible: boolean }) => (
    props.visible ? <div data-sidebar-header-divider="true" /> : null
  ),
}));
vi.mock("./pages/PagesOverview", () => ({
  PagesOverview: (props: {
    onCreatePage: (space: string | null) => void;
    onSelectDraft: (draftId: string, space: string | null) => void;
  }) => (
    <section data-testid="pages-overview">
      <button
        onClick={() => props.onCreatePage(null)}
        type="button"
      >
        Create standalone draft
      </button>
      <button
        onClick={() => props.onSelectDraft("draft-resume", "Research")}
        type="button"
      >
        Resume draft
      </button>
    </section>
  ),
}));
vi.mock("./pages/PageDraftEditor", async () => {
  const React = await import("react");
  return {
    PageDraftEditor: React.forwardRef((props: {
      draftId?: string;
      onBack: () => void;
      onEscapeBeforeLeave?: () => boolean;
      onOpenExisting: (pageId: string) => void;
      onPublished: (pageId: string) => void;
      space: string | null;
    }, ref) => {
      const requestBack = () => draftRequestBackMock(props.onBack);
      React.useImperativeHandle(ref, () => ({
        flush: draftFlushMock,
        getIdentity: draftIdentityMock,
        requestBack,
      }));
      React.useEffect(() => {
        const handleEscape = (event: KeyboardEvent) => {
          if (event.key !== "Escape") return;
          event.preventDefault();
          event.stopPropagation();
          if (props.onEscapeBeforeLeave?.()) return;
          void requestBack();
        };
        window.addEventListener("keydown", handleEscape, true);
        return () => window.removeEventListener("keydown", handleEscape, true);
      });
      return (
        <section
          data-draft-id={props.draftId ?? "new"}
          data-space={props.space ?? "none"}
          data-testid="page-draft-editor"
        >
          <button onClick={() => void requestBack()} type="button">Draft back</button>
          <button onClick={() => props.onPublished(props.draftId ?? "draft-new")} type="button">
            Publish draft
          </button>
          <button onClick={() => props.onOpenExisting("page-existing")} type="button">
            Open existing conflict
          </button>
        </section>
      );
    }),
  };
});
vi.mock("./settings/SettingsSidebar", () => ({ default: () => <aside /> }));
vi.mock("./navigation/ReviewEnvironmentBadge", () => ({
  ReviewEnvironmentBadge: (props: { compact?: boolean }) => (
    <div data-compact={props.compact ? "true" : "false"} data-testid="review-environment-badge" />
  ),
}));
vi.mock("./SpaceDetail", () => ({
  default: (props: {
    onBack: () => void;
    onCreatePage?: (space: string) => void;
    onSpaceDeleted?: (spaceId: string) => void;
    onSpaceLoaded?: (space: Space) => void;
    onSpaceRenamed?: (space: Pick<Space, "id" | "name">) => void;
    spaceName: string;
  }) => (
    <div data-space-name={props.spaceName} data-testid="space-detail">
      <button type="button" onClick={props.onBack}>Space parent</button>
      <button type="button" onClick={() => props.onCreatePage?.(props.spaceName)}>Create Space draft</button>
      <button type="button" onClick={() => props.onSpaceLoaded?.(space("space-1", props.spaceName))}>Finish loading space</button>
      <button type="button" onClick={() => props.onSpaceRenamed?.({ id: "space-1", name: "Renamed Work" })}>Rename loaded space</button>
      <button type="button" onClick={() => props.onSpaceDeleted?.("space-1")}>Delete loaded space</button>
    </div>
  ),
}));
vi.mock("./spaces", () => ({
  SpacesOverview: (props: { createIntent?: boolean; onSelectSpace: (name: string) => void }) => (
    <section data-testid="spaces-overview">
      {props.createIntent && <input aria-label="Space name" autoFocus />}
      <button type="button" onClick={() => props.onSelectSpace("Work")}>Open managed space</button>
    </section>
  ),
}));
vi.mock("./DecisionLog", () => ({ default: () => <div /> }));
vi.mock("./MemoryCard", () => ({ default: () => <div /> }));
vi.mock("./ImportView", () => ({ ImportView: () => <div data-testid="import-view" /> }));

function space(id: string, name: string): Space {
  return {
    id,
    name,
    description: null,
    suggested: false,
    starred: false,
    sort_order: 0,
    memory_count: 1,
    entity_count: 0,
    created_at: 0,
    updated_at: 1,
  };
}

function stubResponsiveViewport(initialMatches: boolean) {
  let matches = initialMatches;
  const listeners = new Set<(event: { readonly matches: boolean }) => void>();
  vi.stubGlobal("matchMedia", vi.fn((query: string) => ({
    get matches() { return matches; },
    media: query,
    onchange: null,
    addEventListener: (_type: string, listener: (event: { readonly matches: boolean }) => void) => listeners.add(listener),
    removeEventListener: (_type: string, listener: (event: { readonly matches: boolean }) => void) => listeners.delete(listener),
    addListener: vi.fn(),
    removeListener: vi.fn(),
    dispatchEvent: vi.fn(),
  })));
  return {
    setMatches(next: boolean) {
      matches = next;
      for (const listener of listeners) listener({ matches: next });
    },
  };
}

function renderMain(props: ComponentProps<typeof Main> = {}) {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  const view = render(
    <QueryClientProvider client={queryClient}>
      <Main {...props} />
    </QueryClientProvider>,
  );
  return {
    ...view,
    rerenderMain: (nextProps: ComponentProps<typeof Main> = {}) =>
      view.rerender(
        <QueryClientProvider client={queryClient}>
          <Main {...nextProps} />
        </QueryClientProvider>,
      ),
  };
}

function deferred<T>() {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((next) => {
    resolve = next;
  });
  return { promise, resolve };
}

describe("Main search", () => {
  beforeEach(async () => {
    eventListeners.clear();
    listSpacesMock.mockReset();
    listSpacesMock.mockResolvedValue([]);
    openFileMock.mockReset();
    openFileMock.mockResolvedValue(undefined);
    setSearchQueryMock.mockReset();
    draftRequestBackMock.mockReset();
    draftRequestBackMock.mockImplementation(async (onBack) => onBack());
    draftFlushMock.mockReset();
    draftFlushMock.mockResolvedValue(true);
    draftIdentityMock.mockReset();
    draftIdentityMock.mockReturnValue({ draftId: "draft-new", version: 1 });
    useSearchMock.mockReset();
    useSearchMock.mockReturnValue({ query: "", setQuery: setSearchQueryMock, results: [] });
    localStorage.clear();
    vi.unstubAllGlobals();
    await i18n.changeLanguage("en");
  });

  it("reads and updates the persisted desktop sidebar preference", async () => {
    // Given a persisted collapsed desktop sidebar
    localStorage.setItem("wenlan-sidebar-collapsed", "true");
    const user = userEvent.setup();

    // When Main mounts and the user toggles it
    renderMain();
    const sidebar = screen.getByRole("complementary");
    expect(sidebar).toHaveAttribute("data-collapsed", "true");
    await user.click(screen.getByRole("button", { name: "Toggle sidebar" }));

    // Then the desktop state and preference both become expanded
    expect(sidebar).toHaveAttribute("data-collapsed", "false");
    expect(localStorage.getItem("wenlan-sidebar-collapsed")).toBe("false");
  });

  it("continues the expanded desktop sidebar divider through the header", async () => {
    const user = userEvent.setup();
    renderMain();

    expect(document.querySelector('[data-sidebar-header-divider="true"]')).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "Toggle sidebar" }));
    expect(document.querySelector('[data-sidebar-header-divider="true"]')).not.toBeInTheDocument();
  });

  it("routes Spaces and create intent to one overview with focused creation", async () => {
    // Given the Home view
    const user = userEvent.setup();
    renderMain();

    // When the sidebar add action is used
    await user.click(screen.getByRole("button", { name: "Create space" }));

    // Then the shared overview opens and focuses its create form
    expect(screen.getByTestId("spaces-overview")).toBeInTheDocument();
    expect(screen.getByRole("textbox", { name: "Space name" })).toHaveFocus();
    expect(screen.getByRole("complementary")).toHaveAttribute("data-active", "spaces");
  });

  it("uses parent replacement for Spaces and Escape for pushed history", async () => {
    // Given a recent Space opened from Home
    const user = userEvent.setup();
    renderMain();
    await user.click(screen.getByRole("button", { name: "Open recent space" }));

    // When Escape is pressed
    fireEvent.keyDown(window, { key: "Escape" });

    // Then pushed history returns Home
    expect(screen.getByTestId("home-page")).toBeInTheDocument();

    // Given the Space is opened again and its parent is used
    await user.click(screen.getByRole("button", { name: "Open recent space" }));
    await user.click(screen.getByRole("button", { name: "Space parent" }));

    // Then the parent always replaces with the Spaces overview
    expect(screen.getByTestId("spaces-overview")).toBeInTheDocument();
  });

  it("closes a narrow drawer without mutating desktop preference and returns focus", async () => {
    // Given a narrow viewport and a collapsed desktop preference
    localStorage.setItem("wenlan-sidebar-collapsed", "true");
    stubResponsiveViewport(true);
    const user = userEvent.setup();
    renderMain();
    const toggle = screen.getByRole("button", { name: "Toggle sidebar" });

    // When the drawer opens and its backdrop closes it
    await user.click(toggle);
    expect(screen.getByRole("complementary")).toHaveAttribute("data-presentation", "overlay");
    expect(screen.getByRole("complementary")).toHaveAttribute("data-open", "true");
    await user.click(screen.getByRole("button", { name: "Close sidebar" }));

    // Then the desktop preference is untouched and focus returns to the toggle
    expect(localStorage.getItem("wenlan-sidebar-collapsed")).toBe("true");
    expect(toggle).toHaveFocus();
  });

  it("uses the first narrow Escape only for the drawer and the next Escape for Space history", async () => {
    // Given a Space opened from Home with the 899px drawer open
    stubResponsiveViewport(true);
    const user = userEvent.setup();
    renderMain();
    await user.click(screen.getByRole("button", { name: "Open recent space" }));
    const toggle = screen.getByRole("button", { name: "Toggle sidebar" });
    await user.click(toggle);
    expect(screen.getByRole("complementary")).toHaveAttribute("data-open", "true");

    // When Escape is pressed once
    fireEvent.keyDown(window, { key: "Escape" });

    // Then only the overlay closes and the Space plus its history remain
    expect(screen.getByRole("complementary")).toHaveAttribute("data-open", "false");
    expect(screen.getByTestId("space-detail")).toBeInTheDocument();
    expect(toggle).toHaveFocus();

    // When Escape is pressed again after the drawer has closed
    fireEvent.keyDown(window, { key: "Escape" });

    // Then the pushed history returns Home
    expect(screen.getByTestId("home-page")).toBeInTheDocument();
  });

  it("returns focus when 899px overlay content becomes a collapsed 900px desktop sidebar", async () => {
    // Given a persisted collapsed desktop sidebar with overlay focus at 899px
    localStorage.setItem("wenlan-sidebar-collapsed", "true");
    const viewport = stubResponsiveViewport(true);
    const user = userEvent.setup();
    renderMain();
    const toggle = screen.getByRole("button", { name: "Toggle sidebar" });
    await user.click(toggle);
    const drawerDestination = screen.getByRole("button", { name: "Open spaces" });
    drawerDestination.focus();
    expect(drawerDestination).toHaveFocus();

    // When the viewport crosses from overlay to desktop at 900px
    act(() => viewport.setMatches(false));

    // Then focus leaves the aria-hidden sidebar without changing the desktop preference
    expect(screen.getByRole("complementary")).toHaveAttribute("data-presentation", "desktop");
    expect(screen.getByRole("complementary")).toHaveAttribute("data-open", "false");
    expect(toggle).toHaveFocus();
    expect(localStorage.getItem("wenlan-sidebar-collapsed")).toBe("true");
  });

  it("prunes a missing MRU id on the next unrelated successful Space visit", async () => {
    // Given valid MRU entries for one current and one deleted Space
    const current = space("space-1", "Work");
    listSpacesMock.mockResolvedValue([current]);
    localStorage.setItem(RECENT_SPACES_STORAGE_KEY, JSON.stringify({
      version: 1,
      entries: [
        { id: "missing", name: "Deleted elsewhere", visitedAt: Date.now() - 2 },
        { id: current.id, name: current.name, visitedAt: Date.now() - 3 },
      ],
    }));
    const user = userEvent.setup();
    renderMain();
    await user.click(screen.getByRole("button", { name: "Open recent space" }));

    // When an unrelated current Space finishes loading and records its visit
    await user.click(screen.getByRole("button", { name: "Finish loading space" }));

    // Then the successful write reconciles history against the current inventory
    const stored: unknown = JSON.parse(localStorage.getItem(RECENT_SPACES_STORAGE_KEY) ?? "null");
    expect(stored).toEqual({
      version: 1,
      entries: [expect.objectContaining({ id: current.id, name: current.name })],
    });
  });

  it("records a loaded Space when its inventory query is still pending", async () => {
    // Given Space detail can load before the shared inventory query resolves
    const current = space("space-1", "Work");
    const other = space("space-2", "Previously visited");
    const inventory = deferred<readonly Space[]>();
    listSpacesMock.mockReturnValue(inventory.promise);
    localStorage.setItem(RECENT_SPACES_STORAGE_KEY, JSON.stringify({
      version: 1,
      entries: [
        { id: other.id, name: other.name, visitedAt: Date.now() - 100 },
      ],
    }));
    const user = userEvent.setup();
    renderMain();
    await user.click(screen.getByRole("button", { name: "Open recent space" }));

    // When the selected Space finishes loading first, followed by the inventory
    await user.click(screen.getByRole("button", { name: "Finish loading space" }));
    await act(async () => {
      inventory.resolve([current, other]);
      await inventory.promise;
    });

    // Then the genuine visit persists without adding an unvisited inventory item
    const stored: unknown = JSON.parse(localStorage.getItem(RECENT_SPACES_STORAGE_KEY) ?? "null");
    expect(stored).toEqual({
      version: 1,
      entries: [
        expect.objectContaining({ id: current.id, name: current.name }),
        expect.objectContaining({ id: other.id, name: other.name }),
      ],
    });
  });

  it("preserves unrelated Recent Spaces when rename happens before inventory resolves", async () => {
    const current = space("space-1", "Work");
    const other = space("space-2", "Previously visited");
    const inventory = deferred<readonly Space[]>();
    listSpacesMock.mockReturnValue(inventory.promise);
    localStorage.setItem(RECENT_SPACES_STORAGE_KEY, JSON.stringify({
      version: 1,
      entries: [
        { id: current.id, name: current.name, visitedAt: Date.now() - 50 },
        { id: other.id, name: other.name, visitedAt: Date.now() - 100 },
      ],
    }));
    const user = userEvent.setup();
    renderMain();
    await user.click(screen.getByRole("button", { name: "Open recent space" }));

    await user.click(screen.getByRole("button", { name: "Rename loaded space" }));

    const stored: unknown = JSON.parse(localStorage.getItem(RECENT_SPACES_STORAGE_KEY) ?? "null");
    expect(stored).toEqual({
      version: 1,
      entries: [
        expect.objectContaining({ id: current.id, name: "Renamed Work" }),
        expect.objectContaining({ id: other.id, name: other.name }),
      ],
    });
    await act(async () => {
      inventory.resolve([current, other]);
      await inventory.promise;
    });
  });

  it("removes only the deleted Recent Space before inventory resolves", async () => {
    const current = space("space-1", "Work");
    const other = space("space-2", "Previously visited");
    const inventory = deferred<readonly Space[]>();
    listSpacesMock.mockReturnValue(inventory.promise);
    localStorage.setItem(RECENT_SPACES_STORAGE_KEY, JSON.stringify({
      version: 1,
      entries: [
        { id: current.id, name: current.name, visitedAt: Date.now() - 50 },
        { id: other.id, name: other.name, visitedAt: Date.now() - 100 },
      ],
    }));
    const user = userEvent.setup();
    renderMain();
    await user.click(screen.getByRole("button", { name: "Open recent space" }));

    await user.click(screen.getByRole("button", { name: "Delete loaded space" }));

    const stored: unknown = JSON.parse(localStorage.getItem(RECENT_SPACES_STORAGE_KEY) ?? "null");
    expect(stored).toEqual({
      version: 1,
      entries: [expect.objectContaining({ id: other.id, name: other.name })],
    });
    await act(async () => {
      inventory.resolve([other]);
      await inventory.promise;
    });
  });

  it("keeps a selected Recent Space active by id when its name changes", async () => {
    // Given a Recent Space selected from its stable production object
    const current = space("space-1", "Work");
    listSpacesMock.mockResolvedValue([current]);
    const user = userEvent.setup();
    renderMain();
    await user.click(screen.getByRole("button", { name: "Open recent space" }));
    expect(screen.getByRole("complementary")).toHaveAttribute("data-current-space", current.id);

    // When the loaded Space is renamed under the same id
    await user.click(screen.getByRole("button", { name: "Rename loaded space" }));

    // Then selection remains on that id while the detail route refreshes its name
    expect(screen.getByRole("complementary")).toHaveAttribute("data-current-space", current.id);
    expect(screen.getByTestId("space-detail")).toHaveAttribute("data-space-name", "Renamed Work");
  });

  it("records a Recent Page only after its detail finishes loading", async () => {
    const user = userEvent.setup();
    renderMain({ initialPageId: "page-1" });

    expect(screen.getByRole("complementary")).toHaveAttribute("data-current-page", "page-1");
    expect(localStorage.getItem(RECENT_PAGES_STORAGE_KEY)).toBeNull();
    await user.click(screen.getByRole("button", { name: "Finish loading page" }));

    const stored: unknown = JSON.parse(localStorage.getItem(RECENT_PAGES_STORAGE_KEY) ?? "null");
    expect(stored).toEqual({
      version: 1,
      entries: [expect.objectContaining({ id: "page-1", title: "Visited page" })],
    });
  });

  it("replaces a Wiki draft editor with the published Page while preserving Wiki history", async () => {
    const user = userEvent.setup();
    renderMain();

    await user.click(screen.getByRole("button", { name: "Open wiki" }));
    await user.click(screen.getByRole("button", { name: "Create standalone draft" }));

    expect(screen.getByTestId("page-draft-editor")).toHaveAttribute("data-space", "none");
    expect(screen.getByRole("complementary")).toHaveAttribute("data-active", "pages");
    draftFlushMock.mockResolvedValue(false);
    await user.click(screen.getByRole("button", { name: "Publish draft" }));
    expect(screen.getByTestId("page-detail")).toHaveAttribute("data-page-id", "draft-new");

    await user.click(screen.getByRole("button", { name: "Page back" }));
    expect(screen.getByTestId("pages-overview")).toBeInTheDocument();
  });

  it("keeps the editor mounted until a sidebar destination passes the draft flush gate", async () => {
    const flush = deferred<boolean>();
    draftFlushMock.mockReturnValue(flush.promise);
    const user = userEvent.setup();
    renderMain();

    await user.click(screen.getByRole("button", { name: "Open wiki" }));
    await user.click(screen.getByRole("button", { name: "Create standalone draft" }));
    await user.click(screen.getByRole("button", { name: "Open spaces" }));

    expect(screen.getByTestId("page-draft-editor")).toBeInTheDocument();
    expect(draftFlushMock).toHaveBeenCalledTimes(1);

    await act(async () => {
      flush.resolve(true);
      await flush.promise;
    });
    expect(await screen.findByTestId("spaces-overview")).toBeInTheDocument();
  });

  it("registers a quit guard that flushes only while a draft editor is active", async () => {
    const onRegisterQuitGuard = vi.fn();
    const user = userEvent.setup();
    renderMain({ onRegisterQuitGuard });
    const lastRegistration = onRegisterQuitGuard.mock.calls[
      onRegisterQuitGuard.mock.calls.length - 1
    ];
    const guard = lastRegistration?.[0] as () => Promise<boolean>;

    await expect(guard()).resolves.toBe(true);
    expect(draftFlushMock).not.toHaveBeenCalled();

    await user.click(screen.getByRole("button", { name: "Open wiki" }));
    await user.click(screen.getByRole("button", { name: "Create standalone draft" }));
    draftFlushMock.mockResolvedValue(false);

    await expect(guard()).resolves.toBe(false);
    expect(draftFlushMock).toHaveBeenCalledTimes(1);
    expect(screen.getByTestId("page-draft-editor")).toBeInTheDocument();
  });

  it("promotes a newly saved draft into history before leaving for Graph", async () => {
    const user = userEvent.setup();
    renderMain();

    await user.click(screen.getByRole("button", { name: "Open wiki" }));
    await user.click(screen.getByRole("button", { name: "Create standalone draft" }));
    await user.click(screen.getByRole("button", { name: "Open graph" }));

    expect(await screen.findByTestId("atlas-view")).toBeInTheDocument();
    fireEvent.keyDown(window, { key: "Escape" });

    expect(await screen.findByTestId("page-draft-editor")).toHaveAttribute(
      "data-draft-id",
      "draft-new",
    );
  });

  it("flushes and promotes a new draft before the first global search query unmounts it", async () => {
    const flush = deferred<boolean>();
    draftFlushMock.mockReturnValue(flush.promise);
    useSearchMock.mockImplementation(() => {
      const [query, setLocalQuery] = useState("");
      return {
        query,
        results: [],
        setQuery: ((next: string) => {
          setSearchQueryMock(next);
          setLocalQuery(next);
        }) as typeof setSearchQueryMock,
      };
    });
    const user = userEvent.setup();
    renderMain();

    await user.click(screen.getByRole("button", { name: "Open wiki" }));
    await user.click(screen.getByRole("button", { name: "Create standalone draft" }));
    await user.type(
      screen.getByPlaceholderText("Search pages, memories, sources..."),
      "architecture",
    );

    expect(draftFlushMock).toHaveBeenCalledTimes(1);
    expect(screen.getByTestId("page-draft-editor")).toBeInTheDocument();
    await act(async () => {
      flush.resolve(true);
      await flush.promise;
    });

    expect(screen.queryByTestId("page-draft-editor")).not.toBeInTheDocument();
    const search = screen.getByPlaceholderText("Search pages, memories, sources...");
    expect(search).toHaveValue(
      "architecture",
    );
    await user.clear(search);
    expect(await screen.findByTestId("page-draft-editor")).toHaveAttribute(
      "data-draft-id",
      "draft-new",
    );
  });

  it("cancels a pending draft search without resurrecting it or promoting the mounted editor", async () => {
    const flush = deferred<boolean>();
    draftFlushMock.mockReturnValue(flush.promise);
    useSearchMock.mockImplementation(() => {
      const [query, setLocalQuery] = useState("");
      return {
        query,
        results: [],
        setQuery: ((next: string) => {
          setSearchQueryMock(next);
          setLocalQuery(next);
        }) as typeof setSearchQueryMock,
      };
    });
    const user = userEvent.setup();
    renderMain();
    await user.click(screen.getByRole("button", { name: "Open wiki" }));
    await user.click(screen.getByRole("button", { name: "Create standalone draft" }));
    const search = screen.getByPlaceholderText("Search pages, memories, sources...");

    await user.type(search, "architecture");
    await user.clear(search);
    expect(search).toHaveValue("");
    expect(screen.getByTestId("page-draft-editor")).toHaveAttribute("data-draft-id", "new");

    await act(async () => {
      flush.resolve(true);
      await flush.promise;
    });

    expect(search).toHaveValue("");
    expect(screen.getByTestId("page-draft-editor")).toHaveAttribute("data-draft-id", "new");
  });

  it("keeps a new draft editor visible when its first global search query cannot flush", async () => {
    draftFlushMock.mockResolvedValue(false);
    useSearchMock.mockImplementation(() => {
      const [query, setLocalQuery] = useState("");
      return {
        query,
        results: [],
        setQuery: ((next: string) => {
          setSearchQueryMock(next);
          setLocalQuery(next);
        }) as typeof setSearchQueryMock,
      };
    });
    const user = userEvent.setup();
    renderMain();

    await user.click(screen.getByRole("button", { name: "Open wiki" }));
    await user.click(screen.getByRole("button", { name: "Create standalone draft" }));
    await user.type(
      screen.getByPlaceholderText("Search pages, memories, sources..."),
      "architecture",
    );

    expect(await screen.findByTestId("page-draft-editor")).toBeInTheDocument();
    expect(screen.getByPlaceholderText("Search pages, memories, sources...")).toHaveValue(
      "architecture",
    );
  });

  it("stays in the editor when a sidebar destination cannot save the draft", async () => {
    draftFlushMock.mockResolvedValue(false);
    const user = userEvent.setup();
    renderMain();

    await user.click(screen.getByRole("button", { name: "Open wiki" }));
    await user.click(screen.getByRole("button", { name: "Create standalone draft" }));
    await user.click(screen.getByRole("button", { name: "Open spaces" }));

    expect(screen.getByTestId("page-draft-editor")).toBeInTheDocument();
    expect(screen.queryByTestId("spaces-overview")).not.toBeInTheDocument();
  });

  it("routes a cross-window memory arrival through the draft flush gate", async () => {
    const flush = deferred<boolean>();
    draftFlushMock.mockReturnValue(flush.promise);
    const user = userEvent.setup();
    const view = renderMain();
    await user.click(screen.getByRole("button", { name: "Open wiki" }));
    await user.click(screen.getByRole("button", { name: "Create standalone draft" }));

    view.rerenderMain({ initialMemoryId: "memory-1" });

    expect(screen.getByTestId("page-draft-editor")).toBeInTheDocument();
    await act(async () => {
      flush.resolve(true);
      await flush.promise;
    });
    expect(await screen.findByTestId("memory-detail")).toBeInTheDocument();
  });

  it("keeps the editor when a cross-window memory arrival cannot save the draft", async () => {
    draftFlushMock.mockResolvedValue(false);
    const user = userEvent.setup();
    const view = renderMain();
    await user.click(screen.getByRole("button", { name: "Open wiki" }));
    await user.click(screen.getByRole("button", { name: "Create standalone draft" }));

    view.rerenderMain({ initialMemoryId: "memory-1" });

    expect(await screen.findByTestId("page-draft-editor")).toBeInTheDocument();
    expect(screen.queryByTestId("memory-detail")).not.toBeInTheDocument();
  });

  it.each([
    ["memory", { initialMemoryId: "memory-1" }, "memory-detail"],
    ["page", { initialPageId: "page-1" }, "page-detail"],
    ["import", { initialView: "import" as const }, "import-view"],
  ])("cancels a withdrawn external %s destination while draft flush is pending", async (
    _label,
    externalProps,
    destinationTestId,
  ) => {
    const flush = deferred<boolean>();
    draftFlushMock.mockReturnValue(flush.promise);
    const user = userEvent.setup();
    const view = renderMain();
    await user.click(screen.getByRole("button", { name: "Open wiki" }));
    await user.click(screen.getByRole("button", { name: "Create standalone draft" }));

    view.rerenderMain(externalProps);
    view.rerenderMain({});
    await act(async () => {
      flush.resolve(true);
      await flush.promise;
    });

    expect(screen.queryByTestId(destinationTestId)).not.toBeInTheDocument();
    expect(screen.getByTestId("page-draft-editor")).toHaveAttribute("data-draft-id", "new");
  });

  it("clears a discarded resumed draft identity before putting it in navigation history", async () => {
    draftIdentityMock.mockReturnValue({ draftId: null, version: null });
    const user = userEvent.setup();
    renderMain();
    await user.click(screen.getByRole("button", { name: "Open wiki" }));
    await user.click(screen.getByRole("button", { name: "Resume draft" }));

    await user.click(screen.getByRole("button", { name: "Open graph" }));
    fireEvent.keyDown(window, { key: "Escape" });

    expect(await screen.findByTestId("page-draft-editor")).toHaveAttribute(
      "data-draft-id",
      "new",
    );
  });

  it("replaces a resumed draft with an existing conflict destination without pushing history", async () => {
    const user = userEvent.setup();
    renderMain();

    await user.click(screen.getByRole("button", { name: "Open wiki" }));
    await user.click(screen.getByRole("button", { name: "Resume draft" }));

    expect(screen.getByTestId("page-draft-editor")).toHaveAttribute("data-draft-id", "draft-resume");
    await user.click(screen.getByRole("button", { name: "Open existing conflict" }));
    expect(screen.getByTestId("page-detail")).toHaveAttribute("data-page-id", "page-existing");

    await user.click(screen.getByRole("button", { name: "Page back" }));
    expect(screen.getByTestId("pages-overview")).toBeInTheDocument();
  });

  it("awaits the editor-owned Escape gate before returning to the originating Space", async () => {
    const gate = deferred<void>();
    draftRequestBackMock.mockImplementationOnce(async (onBack) => {
      await gate.promise;
      onBack();
    });
    const user = userEvent.setup();
    renderMain();

    await user.click(screen.getByRole("button", { name: "Open recent space" }));
    await user.click(screen.getByRole("button", { name: "Create Space draft" }));
    expect(screen.getByTestId("page-draft-editor")).toHaveAttribute("data-space", "Work");
    expect(screen.getByRole("complementary")).toHaveAttribute("data-active", "pages");

    fireEvent.keyDown(window, { key: "Escape" });
    expect(screen.getByTestId("page-draft-editor")).toBeInTheDocument();
    await act(async () => {
      gate.resolve();
      await gate.promise;
    });
    expect(screen.getByTestId("space-detail")).toHaveAttribute("data-space-name", "Work");
  });

  it("uses the first narrow Escape only for the drawer while a Page draft is open", async () => {
    stubResponsiveViewport(true);
    const user = userEvent.setup();
    renderMain();
    await user.click(screen.getByRole("button", { name: "Open wiki" }));
    await user.click(screen.getByRole("button", { name: "Create standalone draft" }));
    await user.click(screen.getByRole("button", { name: "Toggle sidebar" }));
    expect(screen.getByRole("complementary")).toHaveAttribute("data-open", "true");

    fireEvent.keyDown(window, { key: "Escape" });

    expect(screen.getByRole("complementary")).toHaveAttribute("data-open", "false");
    expect(screen.getByTestId("page-draft-editor")).toBeInTheDocument();
    expect(draftRequestBackMock).not.toHaveBeenCalled();
  });

  it("keeps the Wenlan brand out of the top chrome", () => {
    renderMain();

    expect(screen.queryByAltText("Wenlan")).toBeNull();
    expect(screen.queryByRole("button", { name: "Go to home" })).toBeNull();
  });

  it("keeps Settings out of the top chrome because the account menu owns it", () => {
    renderMain();

    expect(screen.queryByRole("button", { name: "Settings" })).toBeNull();
  });

  it("labels the global search around pages, memories, and sources", () => {
    renderMain();

    expect(screen.getByPlaceholderText("Search pages, memories, sources...")).toBeInTheDocument();
    expect(useSearchMock).toHaveBeenCalledWith();
    expect(useSearchMock).not.toHaveBeenCalledWith("memory");
  });

  it("offers a compact search action below the desktop breakpoint", async () => {
    const user = userEvent.setup();
    renderMain();

    const searchAction = screen.getByRole("button", { name: "Search" });
    expect(searchAction).toHaveAttribute("aria-expanded", "false");
    await user.click(searchAction);

    expect(searchAction).toHaveAttribute("aria-expanded", "true");
    expect(screen.getByPlaceholderText("Search pages, memories, sources...")).toHaveFocus();
  });

  it("opens URL-backed source search results through the file bridge", async () => {
    useSearchMock.mockReturnValue({
      query: "source",
      setQuery: setSearchQueryMock,
      results: [{
        id: "source-hit",
        content: "Source credibility notes",
        source: "local_files",
        source_id: "source-1",
        title: "source.md",
        url: "/tmp/source.md",
        chunk_index: 0,
        last_modified: 0,
        score: 0.9,
      }],
    });
    const user = userEvent.setup();
    renderMain();

    await user.click(screen.getByText("Source credibility notes"));

    expect(openFileMock).toHaveBeenCalledWith("/tmp/source.md");
  });

  it("uses one Activity action instead of a Home and Activity segmented control", async () => {
    const user = userEvent.setup();
    renderMain();

    expect(screen.queryByRole("button", { name: "Home" })).not.toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Activity" }));

    expect(await screen.findByTestId("activity-feed")).toBeInTheDocument();
  });

  it("routes the former profile sentinel to General settings instead of the old profile page", async () => {
    const user = userEvent.setup();
    renderMain();

    await user.click(screen.getByRole("button", { name: "Open avatar menu destination" }));

    expect(await screen.findByTestId("settings-page")).toBeInTheDocument();
    expect(screen.queryByTestId("profile-page")).not.toBeInTheDocument();
  });

  it("keeps the compact Review marker visible when Settings or Connect Agent replaces the standard sidebar", async () => {
    const user = userEvent.setup();
    renderMain();

    expect(screen.queryByTestId("review-environment-badge")).not.toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "Open avatar menu destination" }));
    expect(await screen.findByTestId("settings-page")).toBeInTheDocument();
    expect(screen.getByTestId("review-environment-badge")).toHaveAttribute("data-compact", "true");

    await user.click(screen.getByRole("button", { name: "Connect agent" }));
    expect(screen.getByTestId("review-environment-badge")).toHaveAttribute("data-compact", "true");
  });

  it("opens and focuses responsive search from the Tauri event when the placeholder is translated", async () => {
    await i18n.changeLanguage("zh-Hant");
    renderMain();

    const searchInput = screen.getByPlaceholderText(
      "搜尋頁面、記憶、來源...",
    );
    const searchAction = screen.getByRole("button", { name: "搜尋" });
    act(() => eventListeners.get("focus-search")?.());

    expect(searchAction).toHaveAttribute("aria-expanded", "true");
    expect(searchInput).toHaveFocus();
  });

  it("renders AtlasView as the Graph view — back returns home, node clicks open the entity", async () => {
    const user = userEvent.setup();
    renderMain();

    await user.click(screen.getByRole("button", { name: "Open graph view" }));
    expect(screen.getByTestId("atlas-view")).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "Atlas back" }));
    expect(screen.getByTestId("home-page")).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "Open graph view" }));
    await user.click(screen.getByRole("button", { name: "Atlas node" }));
    expect(screen.getByTestId("entity-detail")).toBeInTheDocument();
  });

  it("opens memory detail when initialMemoryId arrives after mount", async () => {
    const view = renderMain();

    expect(screen.getByTestId("home-page")).toBeInTheDocument();

    view.rerenderMain({ initialMemoryId: "memory-1" });

    expect(await screen.findByTestId("memory-detail")).toBeInTheDocument();
  });
});
