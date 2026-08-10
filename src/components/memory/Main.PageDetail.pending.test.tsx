// SPDX-License-Identifier: AGPL-3.0-only
import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import Main from "./Main";

const eventListeners = vi.hoisted(
  () => new Map<string, (payload?: unknown) => void>(),
);

vi.mock("@tauri-apps/api/event", () => ({
  listen: vi.fn((event: string, handler: (payload?: unknown) => void) => {
    eventListeners.set(event, handler);
    return Promise.resolve(() => {
      if (eventListeners.get(event) === handler) eventListeners.delete(event);
    });
  }),
}));

vi.mock("../../lib/tauri", async (importOriginal) => ({
  ...(await importOriginal<typeof import("../../lib/tauri")>()),
  FACET_COLORS: {},
  STABILITY_TIERS: {},
  listMemoriesRich: vi.fn().mockResolvedValue([]),
  listSpaces: vi.fn().mockResolvedValue([]),
  getMemoryStats: vi.fn().mockResolvedValue({
    total: 0,
    new_today: 0,
    confirmed: 0,
    domains: [],
  }),
  search: vi.fn().mockResolvedValue([]),
  searchEntities: vi.fn().mockResolvedValue([]),
  searchPages: vi.fn().mockResolvedValue([]),
  openFile: vi.fn().mockResolvedValue(undefined),
  deleteFileChunks: vi.fn().mockResolvedValue(undefined),
}));

vi.mock("./ActivityFeed", () => ({ default: () => <div /> }));
vi.mock("./IdentityDetail", () => ({ default: () => <div /> }));
vi.mock("./MemoryStream", () => ({ default: () => <div /> }));
vi.mock("./HomePage", () => ({
  default: ({ onSelectPage }: { onSelectPage: (id: string) => void }) => (
    <div data-testid="home-page">
      <button type="button" onClick={() => onSelectPage("page-two")}>
        Open second page
      </button>
    </div>
  ),
}));
vi.mock("./AtlasView", () => ({ default: () => <div /> }));
vi.mock("./MemoryStatusBar", () => ({ default: () => <div /> }));
vi.mock("./MemorySearchResult", () => ({ default: () => <div /> }));
vi.mock("./MemoryDetail", () => ({
  default: ({ sourceId }: { sourceId: string }) => (
    <div data-testid="memory-detail">{sourceId}</div>
  ),
}));
vi.mock("./PageDetail", () => ({
  default: (props: {
    onBack?: () => void;
    onEditDirtyChange?: (dirty: boolean) => void;
    onSavePendingChange?: (pending: boolean) => void;
    pageId: string;
  }) => (
    <section data-testid="page-detail">
      <h1>{props.pageId === "page-two" ? "Second page" : "Pending page"}</h1>
      <button type="button" onClick={() => props.onEditDirtyChange?.(true)}>
        Make page dirty
      </button>
      <button type="button" onClick={() => props.onSavePendingChange?.(true)}>
        Start page save
      </button>
      <button type="button" onClick={() => props.onSavePendingChange?.(false)}>
        Finish page save
      </button>
      <button
        type="button"
        onClick={() => {
          if (confirm("Discard your unsaved draft?")) props.onBack?.();
        }}
      >
        PageDetail back
      </button>
      <div aria-label="Page editor" contentEditable role="textbox" />
      <textarea aria-label="Page notes" />
      <select aria-label="Block style" defaultValue="paragraph">
        <option value="paragraph">Paragraph</option>
      </select>
    </section>
  ),
}));
vi.mock("./DistillReviewPanel", () => ({ default: () => <div /> }));
vi.mock("./SettingsPage", () => ({ default: () => <div /> }));
vi.mock("../SetupWizard", () => ({ SetupWizard: () => <div /> }));
vi.mock("./Sidebar", () => ({
  default: (props: {
    currentPageId?: string | null;
    onNavigateHome: () => void;
    onSelectPage?: (page: { id: string }) => void;
  }) => (
    <aside>
      <button type="button" onClick={props.onNavigateHome}>
        Sidebar home
      </button>
      {props.currentPageId && (
        <button
          type="button"
          onClick={() => props.onSelectPage?.({ id: props.currentPageId! })}
        >
          Reselect current page
        </button>
      )}
    </aside>
  ),
  SidebarToggleButton: (props: {
    onToggle: () => void;
    ref?: React.Ref<HTMLButtonElement>;
  }) => (
    <button ref={props.ref} type="button" onClick={props.onToggle}>
      Toggle sidebar
    </button>
  ),
  SidebarHeaderDivider: () => null,
}));
vi.mock("./pages/PagesOverview", () => ({ PagesOverview: () => <div /> }));
vi.mock("./pages/PageDraftEditor", async () => {
  const React = await import("react");
  return { PageDraftEditor: React.forwardRef(() => <div />) };
});
vi.mock("./spaces", () => ({ SpacesOverview: () => <div /> }));
vi.mock("./settings/SettingsSidebar", () => ({ default: () => <aside /> }));
vi.mock("./SpaceDetail", () => ({ default: () => <div /> }));
vi.mock("./SourcesView", () => ({ default: () => <div /> }));
vi.mock("./DecisionLog", () => ({ default: () => <div /> }));
vi.mock("./RecapsList", () => ({ RecapsList: () => <div /> }));
vi.mock("./ImportView", () => ({
  ImportView: () => <div data-testid="import-view" />,
}));
vi.mock("./AboutWenlanDialog", () => ({ default: () => <div /> }));

interface RenderMainProps {
  initialMemoryId?: string | null;
  initialPageId?: string | null;
  initialView?: "import" | null;
  onRegisterQuitGuard?: (guard: (() => Promise<boolean>) | null) => void;
}

function renderMain(props: RenderMainProps = { initialPageId: "page-one" }) {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  const rendered = render(
    <QueryClientProvider client={client}>
      <Main {...props} />
    </QueryClientProvider>,
  );
  return {
    user: userEvent.setup(),
    ...rendered,
    rerenderMain: (nextProps: RenderMainProps) =>
      rendered.rerender(
        <QueryClientProvider client={client}>
          <Main {...nextProps} />
        </QueryClientProvider>,
      ),
  };
}

describe("Main published PageDetail navigation guards", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    eventListeners.clear();
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("blocks sidebar, search focus, and quit while a page save is pending", async () => {
    let quitGuard: (() => Promise<boolean>) | null = null;
    const { user } = renderMain({
      initialPageId: "page-one",
      onRegisterQuitGuard: (guard) => {
        quitGuard = guard;
      },
    });
    expect(await screen.findByText("Pending page")).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "Start page save" }));
    const search = screen.getByPlaceholderText(
      "Search pages, memories, sources...",
    );
    expect(search).toBeDisabled();

    await user.click(screen.getByRole("button", { name: "Sidebar home" }));
    expect(screen.getByText("Pending page")).toBeInTheDocument();

    eventListeners.get("focus-search")?.();
    expect(search).not.toHaveFocus();
    expect(quitGuard).not.toBeNull();
    await expect(quitGuard!()).resolves.toBe(false);

    await user.click(screen.getByRole("button", { name: "Finish page save" }));
    await waitFor(() => expect(search).toBeEnabled());
    await expect(quitGuard!()).resolves.toBe(true);

    await user.click(screen.getByRole("button", { name: "Make page dirty" }));
    await expect(quitGuard!()).resolves.toBe(false);
  });

  it("defers an external page replacement until the pending save settles", async () => {
    const { user, rerenderMain } = renderMain();
    expect(await screen.findByText("Pending page")).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Start page save" }));

    rerenderMain({ initialPageId: "page-two" });
    expect(screen.getByText("Pending page")).toBeInTheDocument();
    expect(screen.queryByText("Second page")).toBeNull();

    await user.click(screen.getByRole("button", { name: "Finish page save" }));
    expect(await screen.findByText("Second page")).toBeInTheDocument();
  });

  it("does not replay the consumed initial page after internal navigation", async () => {
    const { user } = renderMain();
    expect(await screen.findByText("Pending page")).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Sidebar home" }));
    await user.click(screen.getByRole("button", { name: "Open second page" }));
    expect(await screen.findByText("Second page")).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "Start page save" }));
    await user.click(screen.getByRole("button", { name: "Finish page save" }));

    expect(screen.getByText("Second page")).toBeInTheDocument();
    expect(screen.queryByText("Pending page")).toBeNull();
  });

  it("keeps a dirty draft guarded when the active page is reselected", async () => {
    let quitGuard: (() => Promise<boolean>) | null = null;
    const confirmSpy = vi.spyOn(window, "confirm").mockReturnValue(true);
    const { user } = renderMain({
      initialPageId: "page-one",
      onRegisterQuitGuard: (guard) => {
        quitGuard = guard;
      },
    });
    expect(await screen.findByText("Pending page")).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Make page dirty" }));

    await user.click(
      screen.getByRole("button", { name: "Reselect current page" }),
    );

    expect(confirmSpy).not.toHaveBeenCalled();
    expect(screen.getByText("Pending page")).toBeInTheDocument();
    expect(quitGuard).not.toBeNull();
    await expect(quitGuard!()).resolves.toBe(false);
  });

  it("leaves slash and Escape to editor input, textarea, select, and contenteditable targets", async () => {
    const confirmSpy = vi.spyOn(window, "confirm").mockReturnValue(false);
    const { user } = renderMain();
    expect(await screen.findByText("Pending page")).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Make page dirty" }));

    const search = screen.getByPlaceholderText(
      "Search pages, memories, sources...",
    );
    const editor = screen.getByRole("textbox", { name: "Page editor" });
    const textarea = screen.getByRole("textbox", { name: "Page notes" });
    const select = screen.getByRole("combobox", { name: "Block style" });
    fireEvent.keyDown(editor, { key: "/" });
    fireEvent.keyDown(editor, { key: "Escape" });
    fireEvent.keyDown(textarea, { key: "Escape" });
    fireEvent.keyDown(select, { key: "Escape" });
    fireEvent.keyDown(search, { key: "Escape" });

    expect(search).not.toHaveFocus();
    expect(confirmSpy).not.toHaveBeenCalled();
    expect(screen.getByText("Pending page")).toBeInTheDocument();
  });

  it("ignores default-prevented and composing global shortcuts", async () => {
    const confirmSpy = vi.spyOn(window, "confirm").mockReturnValue(false);
    renderMain();
    expect(await screen.findByText("Pending page")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Make page dirty" }));

    const preventedEscape = new KeyboardEvent("keydown", {
      bubbles: true,
      cancelable: true,
      key: "Escape",
    });
    preventedEscape.preventDefault();
    act(() => window.dispatchEvent(preventedEscape));
    fireEvent.keyDown(window, { isComposing: true, key: "Escape" });

    expect(confirmSpy).not.toHaveBeenCalled();
    expect(screen.getByText("Pending page")).toBeInTheDocument();
  });

  it("confirms dirty global Escape but avoids a second confirmation for PageDetail Back", async () => {
    const confirmSpy = vi.spyOn(window, "confirm").mockReturnValue(false);
    const { user } = renderMain();
    expect(await screen.findByText("Pending page")).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Make page dirty" }));

    const sidebarToggle = screen.getByRole("button", { name: "Toggle sidebar" });
    sidebarToggle.focus();
    fireEvent.keyDown(window, { key: "Escape" });
    expect(confirmSpy).toHaveBeenCalledOnce();
    expect(screen.getByText("Pending page")).toBeInTheDocument();

    confirmSpy.mockReturnValue(true);
    await user.click(screen.getByRole("button", { name: "PageDetail back" }));
    expect(confirmSpy).toHaveBeenCalledTimes(2);
    expect(screen.getByTestId("home-page")).toBeInTheDocument();
  });

  it("confirms before dirty header search and sidebar navigation", async () => {
    const confirmSpy = vi.spyOn(window, "confirm").mockReturnValue(false);
    const { user } = renderMain();
    expect(await screen.findByText("Pending page")).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Make page dirty" }));

    const search = screen.getByPlaceholderText(
      "Search pages, memories, sources...",
    );
    fireEvent.change(search, { target: { value: "blocked" } });
    expect(search).toHaveValue("");
    expect(confirmSpy).toHaveBeenCalledOnce();

    await user.click(screen.getByRole("button", { name: "Sidebar home" }));
    expect(confirmSpy).toHaveBeenCalledTimes(2);
    expect(screen.getByText("Pending page")).toBeInTheDocument();

    confirmSpy.mockReturnValue(true);
    fireEvent.change(search, { target: { value: "accepted" } });
    expect(search).toHaveValue("accepted");
    expect(confirmSpy).toHaveBeenCalledTimes(3);
  });

  it("guards prop-driven page, memory, and import replacements while dirty", async () => {
    const confirmSpy = vi.spyOn(window, "confirm").mockReturnValue(false);
    const { user, rerenderMain } = renderMain();
    expect(await screen.findByText("Pending page")).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Make page dirty" }));

    rerenderMain({ initialPageId: "page-two" });
    await waitFor(() => expect(confirmSpy).toHaveBeenCalledOnce());
    expect(screen.getByText("Pending page")).toBeInTheDocument();

    rerenderMain({
      initialMemoryId: "memory-one",
      initialPageId: "page-one",
    });
    await waitFor(() => expect(confirmSpy).toHaveBeenCalledTimes(2));
    expect(screen.getByText("Pending page")).toBeInTheDocument();

    rerenderMain({ initialPageId: "page-one", initialView: "import" });
    await waitFor(() => expect(confirmSpy).toHaveBeenCalledTimes(3));
    expect(screen.getByText("Pending page")).toBeInTheDocument();

    confirmSpy.mockReturnValue(true);
    rerenderMain({ initialPageId: "page-two" });
    expect(await screen.findByText("Second page")).toBeInTheDocument();
  });
});
