import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, act } from "@testing-library/react";
import { open } from "@tauri-apps/plugin-dialog";

vi.mock("@tauri-apps/api/event", () => ({
  listen: vi.fn(() => Promise.resolve(() => {})),
  emit: vi.fn(),
}));

vi.mock("@tauri-apps/api/core", () => ({
  invoke: vi.fn(),
}));

vi.mock("@tauri-apps/plugin-notification", () => ({
  sendNotification: vi.fn(),
  isPermissionGranted: vi.fn(() => Promise.resolve(true)),
  requestPermission: vi.fn(() => Promise.resolve("granted")),
}));

vi.mock("@tauri-apps/plugin-dialog", () => ({
  open: vi.fn(),
}));

const mockImportChatExport = vi.fn();
const mockSaveTempFile = vi.fn();
const mockListPendingImports = vi.fn();

vi.mock("../../../lib/tauri", () => {
  const labels: Record<string, string> = {
    parsing: "Reading archive",
    stage_a: "Importing conversations",
    stage_b: "Classifying and extracting entities",
    done: "Complete",
    error: "Failed",
  };
  return {
    importChatExport: (...args: unknown[]) => mockImportChatExport(...args),
    saveTempFile: (...args: unknown[]) => mockSaveTempFile(...args),
    listPendingImports: (...args: unknown[]) => mockListPendingImports(...args),
    importStageLabel: (stage: string) => labels[stage] ?? stage,
    IMPORT_STAGE_LABELS: labels,
  };
});

import { ImportFlow } from "../ImportFlow";

describe("ImportFlow", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.useFakeTimers();
    // Default: no pending imports
    mockListPendingImports.mockResolvedValue([]);
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it("renders DropZone in idle state", async () => {
    const { getByTestId } = render(<ImportFlow />);
    expect(getByTestId("chat-import-drop-zone")).toBeTruthy();
  });

  it("shows the drop prompt text", async () => {
    const { getByText } = render(<ImportFlow />);
    expect(getByText("Drop export ZIP here")).toBeTruthy();
  });

  it("shows DropZone always (not replaced during import)", async () => {
    mockListPendingImports.mockResolvedValue([
      { id: "imp_1", vendor: "chatgpt", stage: "stage_b", total_conversations: 77 },
    ]);
    const { getByTestId } = render(<ImportFlow />);
    // DropZone is always present regardless of import state
    expect(getByTestId("chat-import-drop-zone")).toBeTruthy();
  });

  it("polls daemon for pending imports on mount", async () => {
    mockListPendingImports.mockResolvedValue([]);
    render(<ImportFlow />);
    await act(async () => { await vi.advanceTimersByTimeAsync(100); });
    expect(mockListPendingImports).toHaveBeenCalled();
  });

  it("shows idle (no strip) when no pending imports", async () => {
    mockListPendingImports.mockResolvedValue([]);
    const { queryByText } = render(<ImportFlow />);
    // Wait for the effect to run
    await vi.advanceTimersByTimeAsync(100);
    expect(queryByText(/Refining/)).toBeNull();
    expect(queryByText(/Importing/)).toBeNull();
  });

  it("marks the dismiss icon aria-hidden while the dismiss button keeps its own accessible name", async () => {
    const mockOpen = open as ReturnType<typeof vi.fn>;
    mockOpen.mockResolvedValue("/tmp/export.zip");
    mockImportChatExport.mockRejectedValue(new Error("boom"));

    const { getByRole, container } = render(<ImportFlow />);
    const chooseFile = getByRole("button", { name: /choose file/i });
    await act(async () => {
      chooseFile.click();
      await vi.advanceTimersByTimeAsync(50);
    });

    const dismissButton = getByRole("button", { name: "Dismiss" });
    expect(dismissButton).toHaveAttribute("aria-label", "Dismiss");
    container.querySelectorAll("svg").forEach((svg) => {
      expect(svg).toHaveAttribute("aria-hidden", "true");
    });
  });

  it("marks the success status icon aria-hidden", async () => {
    const mockOpen = open as ReturnType<typeof vi.fn>;
    mockOpen.mockResolvedValue("/tmp/export.zip");
    mockImportChatExport.mockResolvedValue({
      import_id: "imp_1",
      vendor: "chatgpt",
      conversations_total: 3,
      conversations_new: 3,
      conversations_skipped_existing: 0,
      memories_stored: 5,
    });

    const { getByRole, getByText, container } = render(<ImportFlow />);
    const chooseFile = getByRole("button", { name: /choose file/i });
    await act(async () => {
      chooseFile.click();
      await vi.advanceTimersByTimeAsync(50);
    });

    expect(getByText(/imported/i)).toBeInTheDocument();
    const icons = container.querySelectorAll("svg");
    expect(icons.length).toBeGreaterThan(0);
    icons.forEach((svg) => expect(svg).toHaveAttribute("aria-hidden", "true"));
  });
});
