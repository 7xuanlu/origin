import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import { GhostConceptsRow } from "../GhostConceptsRow";

describe("GhostConceptsRow", () => {
  it("renders the hint line", () => {
    render(<GhostConceptsRow />);
    expect(screen.getByText(/Concepts will appear here/i)).toBeInTheDocument();
  });

  it("renders exactly 3 ghost cards", () => {
    const { container } = render(<GhostConceptsRow />);
    const ghosts = container.querySelectorAll("[data-ghost-card]");
    expect(ghosts.length).toBe(3);
  });
});
