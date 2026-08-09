// SPDX-License-Identifier: AGPL-3.0-only
import { beforeEach, describe, expect, it, vi } from "vitest";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import SourcesView, { sourceCaption, isManagedSourcePath } from "../SourcesView";
import {
  listRegisteredSources,
  openFile,
  readSourceDir,
  readSourceText,
  removeSource,
  listIndexedFiles,
  type RegisteredSource,
  type IndexedFileInfo,
} from "../../../lib/tauri";

vi.mock("../../../lib/tauri", () => ({
  listRegisteredSources: vi.fn(),
  syncRegisteredSource: vi.fn(),
  openFile: vi.fn(),
  readSourceDir: vi.fn(),
  readSourceText: vi.fn(),
  addSource: vi.fn(),
  removeSource: vi.fn(),
  listIndexedFiles: vi.fn(),
}));

/** Minimal IndexedFileInfo for a given source id + absolute file path. */
function indexed(sourceId: string, absPath: string): IndexedFileInfo {
  return {
    source: "directory",
    source_id: `${sourceId}::${absPath}`,
    title: absPath.split("/").pop() ?? absPath,
    chunk_count: 1,
    last_modified: 1,
  };
}

const MANAGED = "/Users/me/.wenlan/sources";

// notes: directory folder-source; daemon file_count (5) deliberately disagrees
// with the on-disk non-dir count (2) so the "on-disk count" test can tell them
// apart. vault: obsidian, most memories → default selection. managed: the
// hoisted dir — must never appear as its own node.
const SOURCES: RegisteredSource[] = [
  { id: "notes", source_type: "directory", path: "/Users/me/notes", status: "Active", last_sync: 1, file_count: 5, memory_count: 12 },
  { id: "vault", source_type: "obsidian", path: "/Users/me/vault", status: "Active", last_sync: null, file_count: 9, memory_count: 210 },
  { id: "managed", source_type: "directory", path: MANAGED, status: "Active", last_sync: 1, file_count: 2, memory_count: 4 },
];

/** Path-aware directory listing shared by all rendering tests. */
async function fakeReadDir(p: string) {
  const s = String(p);
  if (s === MANAGED)
    return [
      { name: "report.pdf", isDirectory: false },
      { name: "notes.md", isDirectory: false },
    ];
  if (s === "/Users/me/notes")
    return [
      { name: "research", isDirectory: true },
      { name: "index.md", isDirectory: false },
      { name: "cover.png", isDirectory: false },
    ];
  if (s === "/Users/me/notes/research") return [{ name: "paper.md", isDirectory: false }];
  if (s === "/Users/me/vault") return [{ name: "vault-note.md", isDirectory: false }];
  return [];
}

function renderView() {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={qc}>
      <SourcesView onManageSources={() => {}} />
    </QueryClientProvider>,
  );
}

beforeEach(() => {
  vi.clearAllMocks(); // isolate call history — openFile / removeSource asserted per test
  vi.mocked(listRegisteredSources).mockResolvedValue(SOURCES);
  vi.mocked(readSourceDir).mockImplementation(fakeReadDir);
  vi.mocked(readSourceText).mockResolvedValue("");
  vi.mocked(openFile).mockResolvedValue(undefined);
  // report.pdf (loose) and notes/index.md are in the library; notes.md is not.
  vi.mocked(listIndexedFiles).mockResolvedValue([
    indexed("managed", `${MANAGED}/report.pdf`),
    indexed("notes", "/Users/me/notes/index.md"),
  ]);
});

describe("Sources left tree", () => {
  it("hides the managed sources dir and hoists its files as root peers", async () => {
    renderView();

    // Folder-sources render as folder nodes. "vault" is also the default
    // selection, so its bare name doubles as the detail-pane heading.
    expect(await screen.findByText("notes")).toBeInTheDocument();
    expect(screen.getAllByText("vault").length).toBeGreaterThan(0);

    // The managed dir's loose files are hoisted to the root as peers.
    expect(await screen.findByText("report.pdf")).toBeInTheDocument();
    expect(screen.getByText("notes.md")).toBeInTheDocument();

    // The managed `~/.wenlan/sources` dir is NOT shown as its own node.
    // (folderName(MANAGED) === "sources", lowercase.)
    expect(screen.queryByText("sources")).toBeNull();
  });

  it("expands a folder node on click, revealing its children (accordion)", async () => {
    const user = userEvent.setup();
    renderView();

    await user.click(await screen.findByText("notes"));

    expect(vi.mocked(readSourceDir)).toHaveBeenCalledWith("/Users/me/notes");
    expect(await screen.findByText("index.md")).toBeInTheDocument();
    expect(screen.getByText("research")).toBeInTheDocument();
  });

  it("expands a subfolder and reads the joined path", async () => {
    const user = userEvent.setup();
    renderView();

    await user.click(await screen.findByText("notes"));
    await user.click(await screen.findByText("research"));

    expect(vi.mocked(readSourceDir)).toHaveBeenCalledWith("/Users/me/notes/research");
    expect(await screen.findByText("paper.md")).toBeInTheDocument();
  });
});

describe("detail pane", () => {
  it("shows a folder source's on-disk file count, not the daemon's stale count", async () => {
    const user = userEvent.setup();
    renderView();

    await user.click(await screen.findByText("notes"));

    // On disk: research(dir) + index.md + cover.png → 2 files. Daemon says 5.
    expect(await screen.findByText(/\b2 files\b/i)).toBeInTheDocument();
    expect(screen.queryByText(/\b5 files\b/i)).toBeNull();
  });

  it("directory source shows Auto-synced, no manual Sync button", async () => {
    const user = userEvent.setup();
    renderView();

    await user.click(await screen.findByText("notes"));

    expect(await screen.findByText(/Auto-synced/)).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Sync" })).toBeNull();
  });

  it("obsidian source keeps the manual Sync button", async () => {
    renderView();

    // vault has the most memories → default-selected on load.
    expect(await screen.findByRole("button", { name: "Sync" })).toBeInTheDocument();
  });

  it("selecting a loose file shows Open and an in-library status", async () => {
    const user = userEvent.setup();
    renderView();

    await user.click(await screen.findByText("report.pdf"));

    expect(await screen.findByRole("button", { name: "Open" })).toBeInTheDocument();
    // report.pdf is indexed → calm "in your library" status.
    expect(screen.getByText(/in your library/i)).toBeInTheDocument();
  });

  it("renders a text file's content inline below the header", async () => {
    vi.mocked(readSourceText).mockResolvedValue("# Heading\n\nThe body text.");
    const user = userEvent.setup();
    renderView();

    await user.click(await screen.findByText("notes.md"));

    // The real file content is fetched and rendered, not just the summary.
    expect(await screen.findByText("The body text.")).toBeInTheDocument();
    expect(vi.mocked(readSourceText)).toHaveBeenCalledWith(`${MANAGED}/notes.md`);
  });

  it("does not read file contents for a PDF — Open handles it", async () => {
    const user = userEvent.setup();
    renderView();

    await user.click(await screen.findByText("report.pdf"));
    await screen.findByRole("button", { name: "Open" });

    // PDFs have no inline renderer; the pane never reads their bytes.
    expect(vi.mocked(readSourceText)).not.toHaveBeenCalled();
  });

  it("marks a not-yet-indexed supported file as Indexing, never 'not indexed'", async () => {
    const user = userEvent.setup();
    renderView();

    await user.click(await screen.findByText("notes.md"));

    expect(await screen.findByText(/indexing/i)).toBeInTheDocument();
    expect(screen.queryByText(/not indexed/i)).toBeNull();
  });

  it("Remove calls removeSource after confirm", async () => {
    const user = userEvent.setup();
    vi.spyOn(window, "confirm").mockReturnValue(true);
    vi.mocked(removeSource).mockResolvedValue(undefined);
    renderView();

    await user.click(await screen.findByText("notes"));
    fireEvent.click(await screen.findByRole("button", { name: "Remove" }));
    await waitFor(() => expect(removeSource).toHaveBeenCalledWith("notes"));
  });

  it("folder detail heading shows the bare name, not the full path", async () => {
    const user = userEvent.setup();
    renderView();
    await user.click(await screen.findByText("notes"));
    // The heading is the bare folder name…
    expect(await screen.findByRole("heading", { name: "notes" })).toBeInTheDocument();
    // …and the full path is demoted to a caption (still present, not the h2).
    expect(screen.getByText("/Users/me/notes")).toBeInTheDocument();
  });
});

describe("file detail: index internals", () => {
  beforeEach(() => {
    // report.pdf carries a passage count and an AI summary; the rest is default.
    vi.mocked(listIndexedFiles).mockResolvedValue([
      { ...indexed("managed", `${MANAGED}/report.pdf`), chunk_count: 3, summary: "A quarterly report." },
    ]);
  });

  it("shows the passage count and AI summary for an indexed file", async () => {
    const user = userEvent.setup();
    renderView();

    await user.click(await screen.findByText("report.pdf"));

    expect(await screen.findByText(/indexed as 3 passages/i)).toBeInTheDocument();
    expect(screen.getByText("A quarterly report.")).toBeInTheDocument();
  });

  it("does not expose a raw chunk list — chunks are an internal unit", async () => {
    const user = userEvent.setup();
    renderView();

    await user.click(await screen.findByText("report.pdf"));
    await screen.findByText(/indexed as 3 passages/i);

    // The pane shows the passage count, never the raw chunk cards or a toggle.
    expect(screen.queryByRole("button", { name: /chunks/i })).toBeNull();
  });

  it("shows the domain when the indexed file has one", async () => {
    vi.mocked(listIndexedFiles).mockResolvedValue([
      { ...indexed("managed", `${MANAGED}/report.pdf`), chunk_count: 3, summary: "A quarterly report.", domain: "finance" },
    ]);
    const user = userEvent.setup();
    renderView();
    await user.click(await screen.findByText("report.pdf"));
    expect(await screen.findByText("finance")).toBeInTheDocument();
  });
});

describe("empty state", () => {
  it("shows the empty-shelf state when there are no sources", async () => {
    vi.mocked(listRegisteredSources).mockResolvedValue([]);
    renderView();

    expect(await screen.findByText("Nothing on the shelf yet")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /add your first source/i })).toBeInTheDocument();
  });
});

describe("isManagedSourcePath", () => {
  it("matches the managed uploads dir, with or without a trailing slash", () => {
    expect(isManagedSourcePath("/Users/me/.wenlan/sources")).toBe(true);
    expect(isManagedSourcePath("/Users/me/.wenlan/sources/")).toBe(true);
  });
  it("rejects ordinary folders and near-misses", () => {
    expect(isManagedSourcePath("/Users/me/notes")).toBe(false);
    expect(isManagedSourcePath("/x/.wenlan/sources-backup")).toBe(false);
  });
});

const base = {
  id: "s",
  source_type: "directory" as const,
  path: "/x/Books",
  status: "Active" as const,
  file_count: 0,
};

describe("sourceCaption", () => {
  it("says Indexing while settling", () => {
    expect(sourceCaption({ ...base, last_sync: null, memory_count: 0 })).toBe("Indexing…");
  });
  it("shows notes when settled", () => {
    expect(sourceCaption({ ...base, last_sync: 100, memory_count: 142 })).toBe("142 notes");
  });
  it("appends skipped count", () => {
    expect(sourceCaption({ ...base, last_sync: 100, memory_count: 142, last_sync_errors: 2 })).toBe("142 notes, 2 skipped");
  });
});
