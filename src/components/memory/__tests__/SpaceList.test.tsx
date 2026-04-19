import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import SpaceList from "../SpaceList";

vi.mock("../../../lib/tauri", () => ({
  listSpaces: vi.fn().mockResolvedValue([
    { id: "1", name: "work", description: "Work stuff", suggested: false, memory_count: 45, entity_count: 5, created_at: 0, updated_at: 0 },
    { id: "2", name: "health", description: null, suggested: true, memory_count: 12, entity_count: 0, created_at: 0, updated_at: 0 },
  ]),
  createSpace: vi.fn().mockResolvedValue({ id: "3", name: "new", description: null, suggested: false, memory_count: 0, entity_count: 0, created_at: 0, updated_at: 0 }),
  deleteSpace: vi.fn().mockResolvedValue(undefined),
  updateSpace: vi.fn().mockResolvedValue({ id: "1", name: "renamed", description: null, suggested: false, memory_count: 45, entity_count: 5, created_at: 0, updated_at: 0 }),
}));

function wrapper({ children }: { children: React.ReactNode }) {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return <QueryClientProvider client={qc}>{children}</QueryClientProvider>;
}

describe("SpaceList", () => {
  const onSelectSpace = vi.fn();

  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("renders confirmed and suggested spaces", async () => {
    render(<SpaceList onSelectSpace={onSelectSpace} />, { wrapper });
    await waitFor(() => {
      expect(screen.getByText("work")).toBeTruthy();
      expect(screen.getByText("health")).toBeTruthy();
    });
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
});
