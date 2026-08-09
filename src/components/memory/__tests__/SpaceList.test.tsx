import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import SpaceList from "../SpaceList";
import { deleteSpace } from "../../../lib/tauri";
import { i18n } from "../../../i18n";

vi.mock("../../../lib/tauri", () => ({
  listSpaces: vi.fn().mockResolvedValue([
    { id: "1", name: "work", description: "Work stuff", suggested: false, memory_count: 45, entity_count: 5, created_at: 0, updated_at: 0 },
    { id: "2", name: "health", description: null, suggested: true, memory_count: 12, entity_count: 0, created_at: 0, updated_at: 0 },
  ]),
  listPages: vi.fn().mockResolvedValue([
    { id: "page-1", title: "Work page 1", domain: "work", space: "work", source_memory_ids: [], last_modified: new Date(Date.now() - 26 * 60 * 60 * 1000).toISOString() },
    { id: "page-2", title: "Work page 2", domain: "work", space: "work", source_memory_ids: [], last_modified: new Date(Date.now() - 30 * 60 * 60 * 1000).toISOString() },
    { id: "page-3", title: "Health page", domain: "health", space: "health", source_memory_ids: [], last_modified: new Date(Date.now() - 2 * 60 * 60 * 1000).toISOString() },
  ]),
  createSpace: vi.fn().mockResolvedValue({ id: "3", name: "new", description: null, suggested: false, memory_count: 0, entity_count: 0, created_at: 0, updated_at: 0 }),
  deleteSpace: vi.fn().mockResolvedValue(undefined),
  updateSpace: vi.fn().mockResolvedValue({ id: "1", name: "renamed", description: null, suggested: false, memory_count: 45, entity_count: 5, created_at: 0, updated_at: 0 }),
  reorderSpace: vi.fn().mockResolvedValue(undefined),
  toggleSpaceStarred: vi.fn().mockResolvedValue(undefined),
}));

function wrapper({ children }: { children: React.ReactNode }) {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return <QueryClientProvider client={qc}>{children}</QueryClientProvider>;
}

describe("SpaceList", () => {
  const onSelectSpace = vi.fn();
  const mockDeleteSpace = vi.mocked(deleteSpace);

  beforeEach(async () => {
    vi.clearAllMocks();
    await i18n.changeLanguage("en");
  });

  it("renders confirmed and suggested spaces", async () => {
    render(<SpaceList onSelectSpace={onSelectSpace} />, { wrapper });
    await waitFor(() => {
      expect(screen.getByText("work")).toBeTruthy();
      expect(screen.getByText("health")).toBeTruthy();
    });
  });

  it("shows page counts instead of memory counts", async () => {
    render(<SpaceList onSelectSpace={onSelectSpace} />, { wrapper });

    await waitFor(() => {
      expect(screen.getByText("2")).toBeTruthy();
      expect(screen.getByText("1")).toBeTruthy();
    });
    expect(screen.queryByText("45")).toBeNull();
    expect(screen.queryByText("12")).toBeNull();
  });

  it("keeps stable space order without rendering recent activity dates", async () => {
    render(<SpaceList onSelectSpace={onSelectSpace} />, { wrapper });

    await waitFor(() => {
      expect(screen.getByText("work")).toBeTruthy();
      expect(screen.getByText("health")).toBeTruthy();
    });

    expect(screen.queryByText("1d")).toBeNull();
    expect(screen.queryByText("2h")).toBeNull();

    const workRow = screen.getByRole("button", { name: /work/ });
    const healthRow = screen.getByRole("button", { name: /health/ });
    expect(workRow.compareDocumentPosition(healthRow) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
  });

  it("calls onSelectSpace with name when clicked", async () => {
    render(<SpaceList onSelectSpace={onSelectSpace} />, { wrapper });
    await waitFor(() => screen.getByText("work"));
    fireEvent.click(screen.getByText("work"));
    expect(onSelectSpace).toHaveBeenCalledWith("work");
  });

  it("shows new space form on button click", async () => {
    render(<SpaceList onSelectSpace={onSelectSpace} />, { wrapper });
    await waitFor(() => screen.getByTitle("New space"));
    fireEvent.click(screen.getByTitle("New space"));
    expect(screen.getByPlaceholderText("Name")).toBeTruthy();
  });

  it("shows context menu on right-click", async () => {
    render(<SpaceList onSelectSpace={onSelectSpace} />, { wrapper });
    await waitFor(() => screen.getByText("work"));
    fireEvent.contextMenu(screen.getByText("work"));
    expect(screen.getByText("Rename")).toBeTruthy();
    expect(screen.getByText("Delete space")).toBeTruthy();
  });

  it("localizes the space section controls", async () => {
    await i18n.changeLanguage("zh-Hant");
    render(<SpaceList onSelectSpace={onSelectSpace} />, { wrapper });

    expect(await screen.findByText("空間")).toBeInTheDocument();

    fireEvent.click(screen.getByTitle("新增空間"));
    expect(screen.getByPlaceholderText("名稱")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "新增" })).toBeInTheDocument();

    fireEvent.contextMenu(await screen.findByText("work"));
    expect(screen.getByText("加上星號")).toBeInTheDocument();
    expect(screen.getByText("重新命名")).toBeInTheDocument();
    expect(screen.getByText("刪除空間")).toBeInTheDocument();
  });

  it("deletes a space without unsupported memoryAction arguments", async () => {
    render(<SpaceList onSelectSpace={onSelectSpace} />, { wrapper });
    await waitFor(() => screen.getByText("work"));

    fireEvent.contextMenu(screen.getByText("work"));
    fireEvent.click(screen.getByText("Delete space"));

    await waitFor(() => {
      const calls = mockDeleteSpace.mock.calls;
      expect(calls[calls.length - 1]).toEqual(["work"]);
    });
  });
});
