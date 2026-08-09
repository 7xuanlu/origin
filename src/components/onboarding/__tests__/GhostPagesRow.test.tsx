import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import { GhostPagesRow } from "../GhostPagesRow";

describe("GhostPagesRow", () => {
  it("renders the hint line", () => {
    render(<GhostPagesRow />);
    expect(screen.getByText(/Pages will appear here/i)).toBeInTheDocument();
  });

  it("renders exactly 3 ghost cards", () => {
    const { container } = render(<GhostPagesRow />);
    const ghosts = container.querySelectorAll("[data-ghost-card]");
    expect(ghosts.length).toBe(3);
  });
});
