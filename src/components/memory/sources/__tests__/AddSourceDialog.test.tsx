import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import AddSourceDialog from "../AddSourceDialog";
import * as tauri from "../../../../lib/tauri";

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

function createQueryClient() {
  return new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
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

  it("counts markdown, text, and pdf files without obsidian", async () => {
    const { open } = await import("@tauri-apps/plugin-dialog");
    const { readDir } = await import("@tauri-apps/plugin-fs");
    (open as ReturnType<typeof vi.fn>).mockResolvedValue("/Users/test/notes");
    (readDir as ReturnType<typeof vi.fn>).mockResolvedValue([
      { name: "note1.md", isDirectory: false, isFile: true, isSymlink: false },
      { name: "note2.md", isDirectory: false, isFile: true, isSymlink: false },
      { name: "readme.txt", isDirectory: false, isFile: true, isSymlink: false },
      { name: "paper.pdf", isDirectory: false, isFile: true, isSymlink: false },
      { name: "photo.jpg", isDirectory: false, isFile: true, isSymlink: false },
    ]);

    render(<AddSourceDialog onClose={onClose} onSuccess={onSuccess} />, {
      wrapper,
    });

    fireEvent.click(screen.getByRole("button", { name: /browse/i }));

    await waitFor(() => {
      expect(screen.getByText(/4 supported files/)).toBeInTheDocument();
    });
  });

  it("warns but does not block submit when no supported files are found (council change e)", async () => {
    const { open } = await import("@tauri-apps/plugin-dialog");
    const { readDir } = await import("@tauri-apps/plugin-fs");
    (open as ReturnType<typeof vi.fn>).mockResolvedValue("/Users/test/empty");
    (readDir as ReturnType<typeof vi.fn>).mockResolvedValue([
      { name: "photo.jpg", isDirectory: false, isFile: true, isSymlink: false },
    ]);

    render(<AddSourceDialog onClose={onClose} onSuccess={onSuccess} />, {
      wrapper,
    });

    fireEvent.click(screen.getByRole("button", { name: /browse/i }));

    await waitFor(() => {
      expect(screen.getByText(/No notes found/)).toBeInTheDocument();
    });
    expect(screen.getByRole("button", { name: /add source/i })).toBeEnabled();
  });

  it("registers an obsidian vault as obsidian and a plain folder as directory", async () => {
    const { open } = await import("@tauri-apps/plugin-dialog");
    const { readDir } = await import("@tauri-apps/plugin-fs");
    vi.mocked(tauri.addSource).mockResolvedValue({
      id: "src",
      source_type: "directory",
      path: "/Users/test/papers",
      status: "Active",
      last_sync: null,
      file_count: 0,
      memory_count: 0,
    });
    vi.mocked(tauri.syncRegisteredSource).mockReturnValue(new Promise(() => {}));

    // Plain folder → directory
    (open as ReturnType<typeof vi.fn>).mockResolvedValue("/Users/test/papers");
    (readDir as ReturnType<typeof vi.fn>).mockResolvedValue([
      { name: "paper.pdf", isDirectory: false, isFile: true, isSymlink: false },
    ]);
    const first = render(
      <AddSourceDialog onClose={onClose} onSuccess={onSuccess} />,
      { wrapper },
    );
    fireEvent.click(screen.getByRole("button", { name: /browse/i }));
    await waitFor(() => {
      expect(screen.getByText(/1 supported file/)).toBeInTheDocument();
    });
    fireEvent.click(screen.getByRole("button", { name: /add source/i }));
    await waitFor(() => {
      expect(tauri.addSource).toHaveBeenCalledWith("directory", "/Users/test/papers");
    });
    // Directory sources ride the daemon's background scheduler — no on-add sync.
    expect(tauri.syncRegisteredSource).not.toHaveBeenCalled();
    first.unmount();

    // Vault → obsidian
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
    fireEvent.click(screen.getByRole("button", { name: /add source/i }));
    await waitFor(() => {
      expect(tauri.addSource).toHaveBeenCalledWith("obsidian", "/Users/test/vault");
    });
    // Obsidian vaults are NOT on the daemon scheduler, so a one-shot first index
    // must fire on add or the vault sits at "Indexing…" forever.
    await waitFor(() => {
      expect(tauri.syncRegisteredSource).toHaveBeenCalledWith("src");
    });
  });

  it("invalidates registered sources after a successful add before background sync completes", async () => {
    const { open } = await import("@tauri-apps/plugin-dialog");
    const { readDir } = await import("@tauri-apps/plugin-fs");
    (open as ReturnType<typeof vi.fn>).mockResolvedValue("/Users/test/vault");
    (readDir as ReturnType<typeof vi.fn>).mockResolvedValue([
      { name: "note.md", isDirectory: false, isFile: true, isSymlink: false },
    ]);
    vi.mocked(tauri.addSource).mockResolvedValue({
      id: "obsidian-vault",
      source_type: "obsidian",
      path: "/Users/test/vault",
      status: "Active",
      last_sync: null,
      file_count: 0,
      memory_count: 0,
    });
    vi.mocked(tauri.syncRegisteredSource).mockReturnValue(new Promise(() => {}));

    const qc = createQueryClient();
    const invalidateSpy = vi.spyOn(qc, "invalidateQueries");
    render(
      <QueryClientProvider client={qc}>
        <AddSourceDialog onClose={onClose} onSuccess={onSuccess} />
      </QueryClientProvider>,
    );

    fireEvent.click(screen.getByRole("button", { name: /browse/i }));
    await waitFor(() => {
      expect(screen.getByText(/1 supported file/)).toBeInTheDocument();
    });
    fireEvent.click(screen.getByRole("button", { name: /add source/i }));

    await waitFor(() => {
      expect(onSuccess).toHaveBeenCalled();
    });
    expect(invalidateSpy).toHaveBeenCalledWith({
      queryKey: ["registeredSources"],
    });
  });
});
