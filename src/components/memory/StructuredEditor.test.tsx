// SPDX-License-Identifier: AGPL-3.0-only
import { render, screen, fireEvent } from "@testing-library/react";
import { describe, test, expect, vi } from "vitest";
import StructuredEditor from "./StructuredEditor";

describe("StructuredEditor", () => {
  test("renders required fields for decision type", () => {
    const onChange = vi.fn();
    render(<StructuredEditor memoryType="decision" onChange={onChange} />);
    expect(screen.getByPlaceholderText("We decided to...")).toBeInTheDocument();
    expect(screen.getByPlaceholderText("Because...")).toBeInTheDocument();
  });

  test("renders fields for lesson type", () => {
    const onChange = vi.fn();
    render(<StructuredEditor memoryType="lesson" onChange={onChange} />);
    expect(screen.getByPlaceholderText("What was learned")).toBeInTheDocument();
    expect(screen.getByPlaceholderText("When to apply it")).toBeInTheDocument();
  });

  test("renders fields for gotcha type", () => {
    const onChange = vi.fn();
    render(<StructuredEditor memoryType="gotcha" onChange={onChange} />);
    expect(screen.getByPlaceholderText("What can go wrong")).toBeInTheDocument();
    expect(screen.getByPlaceholderText("How to avoid it")).toBeInTheDocument();
  });

  test("renders legacy goal fields for old rows", () => {
    const onChange = vi.fn();
    render(<StructuredEditor memoryType="goal" onChange={onChange} />);
    expect(screen.getByPlaceholderText("Legacy goal")).toBeInTheDocument();
    expect(
      screen.getByText(/legacy goal rows are displayed for migration/i),
    ).toBeInTheDocument();
  });

  test("calls onChange when field is updated", () => {
    const onChange = vi.fn();
    render(<StructuredEditor memoryType="fact" onChange={onChange} />);
    fireEvent.change(screen.getByPlaceholderText("The fact"), { target: { value: "test claim" } });
    expect(onChange).toHaveBeenCalledWith({ claim: "test claim" });
  });

  test("falls back to fact schema for unknown type", () => {
    const onChange = vi.fn();
    render(<StructuredEditor memoryType="unknown" onChange={onChange} />);
    expect(screen.getByPlaceholderText("The fact")).toBeInTheDocument();
  });

  test("renders with initial fields populated", () => {
    const onChange = vi.fn();
    render(
      <StructuredEditor
        memoryType="identity"
        initialFields={{ claim: "I am a developer" }}
        onChange={onChange}
      />
    );
    expect(screen.getByDisplayValue("I am a developer")).toBeInTheDocument();
  });

  test("marks required fields with asterisk", () => {
    const onChange = vi.fn();
    render(<StructuredEditor memoryType="preference" onChange={onChange} />);
    // Required fields should have " *" suffix in their label
    const labels = screen.getAllByText(/\*/);
    expect(labels.length).toBeGreaterThanOrEqual(2); // preference and applies_when
  });
});
