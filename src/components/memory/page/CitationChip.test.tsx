// SPDX-License-Identifier: AGPL-3.0-only
import { describe, it, expect, vi, afterEach } from "vitest";
import { render, screen, fireEvent, act } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { open as shellOpen } from "@tauri-apps/plugin-shell";
import type { MemoryItem, PageCitation } from "../../../lib/tauri";
import CitationChip from "./CitationChip";

const cite = (over: Partial<PageCitation> = {}): PageCitation => ({
  occurrence: 1,
  marker: 1,
  source_kind: "memory",
  locator: "mem-1",
  score: 0.9,
  status: "verified",
  scope: "sentence",
  ...over,
});

const memory = (over: Partial<MemoryItem> = {}): MemoryItem => ({
  source_id: "mem-1",
  title: "Design decision",
  content: "We decided to keep the daemon local-first because it is simpler.",
  summary: null,
  memory_type: "memory",
  domain: null,
  source_agent: "claude-code",
  confidence: null,
  confirmed: true,
  pinned: false,
  supersedes: null,
  last_modified: Math.floor(Date.now() / 1000),
  chunk_count: 1,
  ...over,
});

function renderChip(over: Partial<React.ComponentProps<typeof CitationChip>> = {}) {
  const onOpenMemory = vi.fn();
  const utils = render(
    <CitationChip
      occurrence={1}
      citation={cite()}
      sourceMemory={memory()}
      sourcesLoading={false}
      onOpenMemory={onOpenMemory}
      {...over}
    />,
  );
  return { onOpenMemory, ...utils };
}

afterEach(() => {
  vi.useRealTimers();
});

describe("CitationChip", () => {
  it("renders a focusable button with locator label and occurrence superscript", () => {
    renderChip();
    const chip = screen.getByRole("button", { name: /mem-1/ });
    expect(chip).toBeInTheDocument();
    expect(chip.querySelector("sup")?.textContent).toBe("1");
    expect(chip).toHaveAttribute("data-status", "verified");
  });

  it("marks unverified citations", () => {
    renderChip({ citation: cite({ status: "unverified" }) });
    expect(screen.getByRole("button", { name: /mem-1/ })).toHaveAttribute(
      "data-status",
      "unverified",
    );
  });

  it("opens the popover on focus and links it via aria-describedby", () => {
    renderChip();
    const chip = screen.getByRole("button", { name: /mem-1/ });
    fireEvent.focus(chip);
    const tip = screen.getByRole("tooltip");
    expect(tip).toBeInTheDocument();
    expect(chip.getAttribute("aria-describedby")).toBe(tip.getAttribute("id"));
  });

  it("closes the popover on Escape", () => {
    renderChip();
    const chip = screen.getByRole("button", { name: /mem-1/ });
    fireEvent.focus(chip);
    expect(screen.getByRole("tooltip")).toBeInTheDocument();
    fireEvent.keyDown(chip, { key: "Escape" });
    expect(screen.queryByRole("tooltip")).toBeNull();
  });

  it("opens on hover after a delay and closes on mouse-out", () => {
    vi.useFakeTimers();
    renderChip();
    const wrapper = screen.getByRole("button", { name: /mem-1/ }).parentElement!;
    fireEvent.mouseEnter(wrapper);
    expect(screen.queryByRole("tooltip")).toBeNull();
    act(() => vi.advanceTimersByTime(200));
    expect(screen.getByRole("tooltip")).toBeInTheDocument();
    fireEvent.mouseLeave(wrapper);
    act(() => vi.advanceTimersByTime(200));
    expect(screen.queryByRole("tooltip")).toBeNull();
  });

  it("shows memory details and opens the memory from the popover action", async () => {
    const user = userEvent.setup();
    const { onOpenMemory } = renderChip();
    fireEvent.focus(screen.getByRole("button", { name: /mem-1/ }));
    expect(screen.getByText("Design decision")).toBeInTheDocument();
    expect(screen.getByText("Source memory")).toBeInTheDocument();
    expect(screen.getByText(/We decided to keep the daemon/)).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: /Open memory/ }));
    expect(onOpenMemory).toHaveBeenCalledWith("mem-1");
  });

  it("clicking a memory chip opens the memory directly", async () => {
    const user = userEvent.setup();
    const { onOpenMemory } = renderChip();
    await user.click(screen.getByRole("button", { name: /mem-1/ }));
    expect(onOpenMemory).toHaveBeenCalledWith("mem-1");
  });

  it("explains a missing source and does not navigate on click", async () => {
    const user = userEvent.setup();
    const { onOpenMemory } = renderChip({ sourceMemory: null, sourcesLoading: false });
    const chip = screen.getByRole("button", { name: /mem-1/ });
    fireEvent.focus(chip);
    expect(screen.getByText(/no longer exists/i)).toBeInTheDocument();
    expect(screen.getByText(/re-distill/i)).toBeInTheDocument();
    await user.click(chip);
    expect(onOpenMemory).not.toHaveBeenCalled();
  });

  it("shows a skeleton while sources are loading", () => {
    renderChip({ sourceMemory: null, sourcesLoading: true });
    fireEvent.focus(screen.getByRole("button", { name: /mem-1/ }));
    expect(screen.getByTestId("citation-popover-skeleton")).toBeInTheDocument();
  });

  it("notes unverified status in the popover", () => {
    renderChip({ citation: cite({ status: "unverified" }) });
    fireEvent.focus(screen.getByRole("button", { name: /mem-1/ }));
    expect(screen.getByText(/unverified/i)).toBeInTheDocument();
  });

  it("opens external urls in the browser", async () => {
    const user = userEvent.setup();
    renderChip({
      citation: cite({ source_kind: "external_url", locator: "https://docs.rs/serde" }),
      sourceMemory: null,
    });
    await user.click(screen.getByRole("button", { name: /docs\.rs/ }));
    expect(vi.mocked(shellOpen)).toHaveBeenCalledWith("https://docs.rs/serde");
  });

  it("shows the file path and opens it from the popover action", async () => {
    const user = userEvent.setup();
    renderChip({
      citation: cite({ source_kind: "external_file", locator: "/notes/design.md" }),
      sourceMemory: null,
    });
    fireEvent.focus(screen.getByRole("button", { name: /design\.md/ }));
    expect(screen.getByText("/notes/design.md")).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: /Open file/ }));
    expect(vi.mocked(shellOpen)).toHaveBeenCalledWith("/notes/design.md");
  });

  it("explains that authored content survives re-distillation", () => {
    renderChip({ citation: cite({ source_kind: "authored" }), sourceMemory: null });
    fireEvent.focus(screen.getByRole("button", { name: /authored/ }));
    expect(screen.getByText(/kept unchanged when the page is re-distilled/i)).toBeInTheDocument();
  });
});
