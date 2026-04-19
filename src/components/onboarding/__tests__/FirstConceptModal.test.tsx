// SPDX-License-Identifier: AGPL-3.0-only
import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { FirstConceptModal } from "../FirstConceptModal";

const sampleConcept = {
  id: "c1",
  title: "Rust ownership",
  summary: "A concept about Rust's ownership rules",
  source_memory_ids: ["m1", "m2", "m3"],
};

describe("FirstConceptModal", () => {
  it("renders concept title and summary", () => {
    render(
      <FirstConceptModal concept={sampleConcept} onOpen={vi.fn()} onDismiss={vi.fn()} />
    );
    expect(screen.getByText(/Rust ownership/)).toBeInTheDocument();
    expect(screen.getByText(/ownership rules/)).toBeInTheDocument();
    expect(screen.getByText(/compiled from 3 memories/i)).toBeInTheDocument();
  });

  it("calls onOpen when Open concept clicked", () => {
    const onOpen = vi.fn();
    render(<FirstConceptModal concept={sampleConcept} onOpen={onOpen} onDismiss={vi.fn()} />);
    fireEvent.click(screen.getByRole("button", { name: /open concept/i }));
    expect(onOpen).toHaveBeenCalledWith("c1");
  });

  it("calls onDismiss when Dismiss clicked", () => {
    const onDismiss = vi.fn();
    render(<FirstConceptModal concept={sampleConcept} onOpen={vi.fn()} onDismiss={onDismiss} />);
    fireEvent.click(screen.getByRole("button", { name: /dismiss/i }));
    expect(onDismiss).toHaveBeenCalled();
  });

  it("calls onDismiss on Escape key", () => {
    const onDismiss = vi.fn();
    render(<FirstConceptModal concept={sampleConcept} onOpen={vi.fn()} onDismiss={onDismiss} />);
    fireEvent.keyDown(document, { key: "Escape" });
    expect(onDismiss).toHaveBeenCalled();
  });

  it("focuses the first focusable element on mount", () => {
    render(<FirstConceptModal concept={sampleConcept} onOpen={vi.fn()} onDismiss={vi.fn()} />);
    // Dismiss is the first button in the dialog, so it should receive focus.
    expect(document.activeElement).toBe(screen.getByRole("button", { name: /dismiss/i }));
  });

  it("traps Tab at the last focusable element (cycles to first)", () => {
    render(<FirstConceptModal concept={sampleConcept} onOpen={vi.fn()} onDismiss={vi.fn()} />);
    const dismissBtn = screen.getByRole("button", { name: /dismiss/i });
    const openBtn = screen.getByRole("button", { name: /open concept/i });
    // Move focus to the last element.
    openBtn.focus();
    expect(document.activeElement).toBe(openBtn);
    // Tab at the end should cycle back to the first focusable.
    fireEvent.keyDown(document, { key: "Tab" });
    expect(document.activeElement).toBe(dismissBtn);
  });

  it("traps Shift+Tab at the first focusable element (cycles to last)", () => {
    render(<FirstConceptModal concept={sampleConcept} onOpen={vi.fn()} onDismiss={vi.fn()} />);
    const dismissBtn = screen.getByRole("button", { name: /dismiss/i });
    const openBtn = screen.getByRole("button", { name: /open concept/i });
    // First element should already be focused on mount.
    expect(document.activeElement).toBe(dismissBtn);
    // Shift+Tab at the start should cycle to the last focusable.
    fireEvent.keyDown(document, { key: "Tab", shiftKey: true });
    expect(document.activeElement).toBe(openBtn);
  });
});
