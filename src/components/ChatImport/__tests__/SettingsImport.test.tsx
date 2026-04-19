import { describe, it, expect, vi } from "vitest";
import { render, screen } from "@testing-library/react";

vi.mock("@tauri-apps/api/event", () => ({
  listen: vi.fn(() => Promise.resolve(() => {})),
  emit: vi.fn(),
}));

vi.mock("@tauri-apps/api/core", () => ({
  invoke: vi.fn(),
}));

vi.mock("@tauri-apps/plugin-dialog", () => ({
  open: vi.fn(),
}));

vi.mock("../../../lib/tauri", () => ({
  importChatExport: vi.fn(),
  saveTempFile: vi.fn(),
  listPendingImports: vi.fn(() => Promise.resolve([])),
  importStageLabel: (s: string) => s,
  IMPORT_STAGE_LABELS: {},
}));

import { ImportFlow } from "../ImportFlow";

describe("Settings page chat import section", () => {
  it("ImportFlow renders in idle state with DropZone", () => {
    render(<ImportFlow />);
    expect(screen.getByTestId("chat-import-drop-zone")).toBeInTheDocument();
  });

  it("ImportFlow shows the drop prompt text", () => {
    render(<ImportFlow />);
    expect(
      screen.getByText(/drop export zip here/i),
    ).toBeInTheDocument();
  });
});
