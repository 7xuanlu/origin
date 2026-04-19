import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import AddSourceDialog from "../AddSourceDialog";

vi.mock("../../../../lib/tauri");
vi.mock("@tauri-apps/plugin-dialog", () => ({
  open: vi.fn(),
}));
vi.mock("@tauri-apps/plugin-fs", () => ({
  readDir: vi.fn(),
}));

function wrapper({ children }: { children: React.ReactNode }) {
  const qc = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  return <QueryClientProvider client={qc}>{children}</QueryClientProvider>;
}

describe("AddSourceDialog", () => {
  const onClose = vi.fn();
  const onSuccess = vi.fn();

  beforeEach(() => {
    vi.resetAllMocks();
  });

  it("submit is disabled when path is empty", () => {
    render(<AddSourceDialog onClose={onClose} onSuccess={onSuccess} />, {
      wrapper,
    });
    const submitBtn = screen.getByRole("button", { name: /add source/i });
    expect(submitBtn).toBeDisabled();
  });

  it("ESC key calls onClose", () => {
    render(<AddSourceDialog onClose={onClose} onSuccess={onSuccess} />, {
      wrapper,
    });
    fireEvent.keyDown(window, { key: "Escape" });
    expect(onClose).toHaveBeenCalled();
  });

  it("browse opens native dialog and detects obsidian vault", async () => {
    const { open } = await import("@tauri-apps/plugin-dialog");
    const { readDir } = await import("@tauri-apps/plugin-fs");
    (open as ReturnType<typeof vi.fn>).mockResolvedValue("/Users/test/vault");
    (readDir as ReturnType<typeof vi.fn>).mockResolvedValue([
      { name: ".obsidian", isDirectory: true, isFile: false, isSymlink: false },
      { name: "note.md", isDirectory: false, isFile: true, isSymlink: false },
    ]);

    render(<AddSourceDialog onClose={onClose} onSuccess={onSuccess} />, {
      wrapper,
    });

    fireEvent.click(screen.getByRole("button", { name: /browse/i }));

    await waitFor(() => {
      expect(screen.getByText(/Obsidian vault/)).toBeInTheDocument();
    });
  });

  it("detects markdown files without obsidian", async () => {
    const { open } = await import("@tauri-apps/plugin-dialog");
    const { readDir } = await import("@tauri-apps/plugin-fs");
    (open as ReturnType<typeof vi.fn>).mockResolvedValue("/Users/test/notes");
    (readDir as ReturnType<typeof vi.fn>).mockResolvedValue([
      { name: "note1.md", isDirectory: false, isFile: true, isSymlink: false },
      { name: "note2.md", isDirectory: false, isFile: true, isSymlink: false },
      { name: "readme.txt", isDirectory: false, isFile: true, isSymlink: false },
    ]);

    render(<AddSourceDialog onClose={onClose} onSuccess={onSuccess} />, {
      wrapper,
    });

    fireEvent.click(screen.getByRole("button", { name: /browse/i }));

    await waitFor(() => {
      expect(screen.getByText(/2 markdown files/)).toBeInTheDocument();
    });
  });

  it("shows error when no markdown files found", async () => {
    const { open } = await import("@tauri-apps/plugin-dialog");
    const { readDir } = await import("@tauri-apps/plugin-fs");
    (open as ReturnType<typeof vi.fn>).mockResolvedValue("/Users/test/empty");
    (readDir as ReturnType<typeof vi.fn>).mockResolvedValue([
      { name: "readme.txt", isDirectory: false, isFile: true, isSymlink: false },
    ]);

    render(<AddSourceDialog onClose={onClose} onSuccess={onSuccess} />, {
      wrapper,
    });

    fireEvent.click(screen.getByRole("button", { name: /browse/i }));

    await waitFor(() => {
      expect(screen.getByText(/no markdown files/i)).toBeInTheDocument();
    });
    expect(screen.getByRole("button", { name: /add source/i })).toBeDisabled();
  });
});
