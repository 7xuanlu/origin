import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, act } from "@testing-library/react";
import IconOverlay from "./IconOverlay";

// Mock Tauri APIs
vi.mock("@tauri-apps/api/event", () => ({
  listen: vi.fn(() => Promise.resolve(() => {})),
}));
vi.mock("@tauri-apps/api/window", () => ({
  getCurrentWindow: vi.fn(() => ({
    setSize: vi.fn(() => Promise.resolve()),
    setPosition: vi.fn(() => Promise.resolve()),
    show: vi.fn(() => Promise.resolve()),
    hide: vi.fn(() => Promise.resolve()),
  })),
  primaryMonitor: vi.fn(() =>
    Promise.resolve({ size: { width: 2560, height: 1600 }, scaleFactor: 2 })
  ),
}));
vi.mock("@tauri-apps/api/dpi", () => ({
  LogicalSize: vi.fn(),
  LogicalPosition: vi.fn(),
}));
vi.mock("../lib/ambient", () => ({
  listenShowIcon: vi.fn(() => Promise.resolve(() => {})),
  triggerIconClick: vi.fn(() => Promise.resolve()),
}));

describe("IconOverlay", () => {
  beforeEach(() => vi.clearAllMocks());

  it("renders without crashing", async () => {
    await act(async () => {
      render(<IconOverlay />);
    });
  });

  it("renders null initially (no icon payload)", async () => {
    let container!: HTMLElement;
    await act(async () => {
      ({ container } = render(<IconOverlay />));
    });
    expect(container.firstChild).toBeNull();
  });
});
