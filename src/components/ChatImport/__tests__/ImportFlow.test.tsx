import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, act } from "@testing-library/react";

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
});
