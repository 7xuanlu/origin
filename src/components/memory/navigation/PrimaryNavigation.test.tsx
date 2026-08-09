import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { PrimaryNavigation } from "./PrimaryNavigation";

const labels = {
  graph: "Graph",
  home: "Home",
  memories: "Memories",
  navigation: "Primary navigation",
  pages: "Wiki",
  sources: "Sources",
  spaces: "Spaces",
} as const;

const preservedIconGeometry = {
  Graph: "M7 6h10M6 8l5 8M18 8l-5 8",
  Home: "M3 10.5L12 3l9 7.5M5 9.5V21h14V9.5M9.5 21v-6h5v6",
} as const;

function renderNavigation(active: "home" | "spaces" | null = null) {
  const callbacks = {
    graph: vi.fn(),
    home: vi.fn(),
    memories: vi.fn(),
    pages: vi.fn(),
    sources: vi.fn(),
    spaces: vi.fn(),
  };
  const view = render(
    <PrimaryNavigation
      active={active}
      labels={labels}
      onNavigateGraph={callbacks.graph}
      onNavigateHome={callbacks.home}
      onNavigateLog={callbacks.memories}
      onNavigatePages={callbacks.pages}
      onNavigateSources={callbacks.sources}
      onNavigateSpaces={callbacks.spaces}
    />,
  );
  return { callbacks, ...view };
}

describe("PrimaryNavigation", () => {
  it("uses a stable navigation label instead of naming the landmark after Home", () => {
    renderNavigation();

    expect(screen.getByRole("navigation", { name: "Primary navigation" })).toBeInTheDocument();
    expect(screen.queryByRole("navigation", { name: "Home" })).not.toBeInTheDocument();
  });

  it("keeps Wiki first-class and places Recent Pages then Recent Spaces after every primary destination", () => {
    render(
      <PrimaryNavigation
        active={null}
        labels={labels}
        onNavigateGraph={() => {}}
        onNavigateHome={() => {}}
        onNavigateLog={() => {}}
        onNavigatePages={() => {}}
        onNavigateSources={() => {}}
        onNavigateSpaces={() => {}}
        recentPagesSection={<div>Recent pages</div>}
        recentSpacesSection={<div>Recent spaces</div>}
      />,
    );

    const destinations = ["Home", "Wiki", "Spaces", "Graph", "Memories", "Sources"].map((name) => screen.getByRole("button", { name }));
    for (const [index, destination] of destinations.entries()) {
      const next = destinations[index + 1];
      if (next) expect(destination.compareDocumentPosition(next) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
    }
    const recentPages = screen.getByText("Recent pages");
    const recentSpaces = screen.getByText("Recent spaces");
    expect(destinations[5]?.compareDocumentPosition(recentPages) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
    expect(recentPages.compareDocumentPosition(recentSpaces) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
    expect(recentPages.closest("nav")).not.toBe(destinations[0]?.closest("nav"));
  });

  it("does not expose a suggestion count in primary navigation", () => {
    renderNavigation();

    expect(screen.getByRole("button", { name: "Spaces" })).toHaveTextContent(/^Spaces$/);
    expect(screen.queryByText(/^\d+$/)).not.toBeInTheDocument();
  });

  it("keeps space creation inside the Spaces page instead of duplicating it in navigation", () => {
    renderNavigation();

    expect(screen.queryByRole("button", { name: "New space" })).not.toBeInTheDocument();
  });

  it("renders the selected self-contained-world Planet mark instead of map or folder geometry", () => {
    // Given the Spaces destination is available in the primary navigation
    renderNavigation();

    // When its decorative icon is inspected
    const spaces = screen.getByRole("button", { name: "Spaces" });
    const mark = spaces.querySelector<SVGSVGElement>("[data-space-mark='self-contained-world']");

    // Then the official Tabler Planet geometry is present and the rejected metaphors are absent
    expect(mark).not.toBeNull();
    expect(Array.from(mark?.querySelectorAll("path") ?? [], (path) => path.getAttribute("d"))).toEqual([
      "M18.816 13.58c2.292 2.138 3.546 4 3.092 4.9c-.745 1.46 -5.783 -.259 -11.255 -3.838c-5.47 -3.579 -9.304 -7.664 -8.56 -9.123c.464 -.91 2.926 -.444 5.803 .805",
      "M5 12a7 7 0 1 0 14 0a7 7 0 1 0 -14 0",
    ]);
    expect(mark?.querySelector("path")).toHaveAttribute("stroke-width", "2");
    expect(spaces.querySelector('path[d="M3 7l6 -3l6 3l6 -3v13l-6 3l-6 -3l-6 3v-13"]')).toBeNull();
    expect(spaces.querySelector('path[d="M4 5.5h6l2 2h8v11H4z"]')).toBeNull();
  });

  it("keeps the mark decorative while the selected button exposes its name and page state", () => {
    // Given Spaces is the selected global destination
    renderNavigation("spaces");

    // When assistive semantics are inspected
    const spaces = screen.getByRole("button", { name: "Spaces", current: "page" });
    const mark = spaces.querySelector<SVGSVGElement>("[data-space-mark='self-contained-world']");

    // Then the button owns navigation semantics and the SVG stays decorative
    expect(spaces).toHaveAccessibleName("Spaces");
    expect(spaces).toHaveAttribute("aria-current", "page");
    expect(mark).toHaveAttribute("aria-hidden", "true");
  });

  it("uses the exact 14px navigation footprint and documented state tokens", () => {
    // Given Spaces first renders as a default destination
    const { rerender } = renderNavigation();
    const defaultMark = screen.getByRole("button", { name: "Spaces" }).querySelector<SVGSVGElement>("[data-space-mark='self-contained-world']");

    // When the destination changes to selected
    expect(defaultMark).toHaveAttribute("height", "14");
    expect(defaultMark).toHaveAttribute("width", "14");
    expect(defaultMark).toHaveAttribute("viewBox", "0 0 24 24");
    expect(defaultMark).toHaveStyle({ color: "var(--mem-text-tertiary)" });
    rerender(
      <PrimaryNavigation
        active="spaces"
        labels={labels}
        onNavigateSpaces={() => {}}
      />,
    );

    // Then selected identity maps to indigo without changing geometry
    const selectedMark = screen.getByRole("button", { name: "Spaces" }).querySelector<SVGSVGElement>("[data-space-mark='self-contained-world']");
    expect(selectedMark).toHaveAttribute("height", "14");
    expect(selectedMark).toHaveAttribute("width", "14");
    expect(selectedMark).toHaveAttribute("viewBox", "0 0 24 24");
    expect(selectedMark).toHaveStyle({ color: "var(--mem-accent-indigo)" });
  });

  it("gives the selected row a quiet full-width wash, outer rail, breathing room, and distinct focus treatment", () => {
    renderNavigation("spaces");

    const selected = screen.getByRole("button", { name: "Spaces", current: "page" });
    const rail = selected.querySelector('[data-primary-navigation-active-marker="true"]');

    expect(selected).toHaveStyle({
      backgroundColor: "var(--mem-indigo-bg)",
      fontWeight: "500",
    });
    expect(selected.className).toContain("px-3");
    expect(selected.className).toContain("focus-visible:outline-2");
    expect(selected.className).toContain("focus-visible:outline-offset-2");
    expect(selected.className).toContain("focus-visible:outline-[var(--mem-accent-page)]");
    expect(selected.className).not.toContain("focus-visible:ring-");
    expect(rail?.className).toContain("left-0");
    expect(rail?.className).not.toContain("-left-");
    expect(rail?.className).toContain("w-0.5");
  });

  it("uses the canonical layered Page glyph for the Wiki destination", () => {
    // Given Wiki is the browse-all destination for every Page presentation type
    renderNavigation();

    // When its decorative navigation mark is inspected
    const mark = screen
      .getByRole("button", { name: "Wiki" })
      .querySelector<SVGSVGElement>('[data-navigation-icon="wiki-page"]');

    // Then it reuses the Page-family geometry instead of the topic subtype document
    expect(mark).not.toBeNull();
    expect(mark?.querySelector("path")).toHaveAttribute(
      "d",
      "M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5",
    );
    expect(mark?.querySelector("path")).toHaveAttribute("stroke-width", "1.8");
    expect(mark).toHaveAttribute("height", "14");
    expect(mark).toHaveAttribute("width", "14");
    expect(mark?.querySelector('path[d="M6 3h8l4 4v14H6zM14 3v5h5M9 12h6M9 16h6"]')).toBeNull();
  });

  it("uses the approved intake tray for Sources without import-arrow semantics", () => {
    // Given the Sources destination is available in primary navigation
    renderNavigation();

    // When its decorative mark is inspected
    const mark = screen
      .getByRole("button", { name: "Sources" })
      .querySelector<SVGSVGElement>('[data-navigation-icon="sources-intake-tray"]');

    // Then three plain input strokes enter a shallow tray with no arrowhead geometry
    expect(mark).not.toBeNull();
    expect(Array.from(mark?.querySelectorAll("path") ?? [], (path) => path.getAttribute("d"))).toEqual([
      "M7 4v6M12 4v6M17 4v6",
      "M5 13l1.5 5h11l1.5-5",
    ]);
    expect(mark?.querySelector("polygon, polyline")).toBeNull();
    expect(mark).toHaveAttribute("aria-hidden", "true");
    expect(mark).toHaveAttribute("height", "14");
    expect(mark).toHaveAttribute("width", "14");
  });

  it("preserves the unaffected icon geometry and keeps Wiki, Memories, and Graph visually distinct", () => {
    // Given the redesigned primary navigation
    renderNavigation();

    // When every unaffected destination icon is inspected
    for (const [name, geometry] of Object.entries(preservedIconGeometry)) {
      const icon = screen.getByRole("button", { name }).querySelector("svg");

      // Then its original path remains byte-for-byte unchanged
      expect(icon).not.toBeNull();
      expect(icon?.querySelector("path")?.getAttribute("d")).toBe(geometry);
    }

    const wiki = screen.getByRole("button", { name: "Wiki" }).querySelector("svg");
    const memories = screen.getByRole("button", { name: "Memories" }).querySelector("svg");
    const graph = screen.getByRole("button", { name: "Graph" }).querySelector("svg");
    expect(wiki).toHaveAttribute("data-navigation-icon", "wiki-page");
    expect(memories).toHaveAttribute("data-navigation-icon", "brain");
    expect(graph).toHaveAttribute("data-navigation-icon", "graph");
    expect(wiki?.innerHTML).not.toBe(memories?.innerHTML);
    expect(memories?.innerHTML).not.toBe(graph?.innerHTML);
  });

  it("uses the exact official Tabler Brain geometry selected for Memories", () => {
    renderNavigation();

    const mark = screen
      .getByRole("button", { name: "Memories" })
      .querySelector<SVGSVGElement>('[data-navigation-icon="brain"]');

    expect(mark).not.toBeNull();
    expect(Array.from(mark?.querySelectorAll("path") ?? [], (path) => path.getAttribute("d"))).toEqual([
      "M15.5 13a3.5 3.5 0 0 0 -3.5 3.5v1a3.5 3.5 0 0 0 7 0v-1.8",
      "M8.5 13a3.5 3.5 0 0 1 3.5 3.5v1a3.5 3.5 0 0 1 -7 0v-1.8",
      "M17.5 16a3.5 3.5 0 0 0 0 -7h-.5",
      "M19 9.3v-2.8a3.5 3.5 0 0 0 -7 0",
      "M6.5 16a3.5 3.5 0 0 1 0 -7h.5",
      "M5 9.3v-2.8a3.5 3.5 0 0 1 7 0v10",
    ]);
    expect(mark).toHaveAttribute("height", "14");
    expect(mark).toHaveAttribute("viewBox", "0 0 24 24");
    expect(mark).toHaveAttribute("width", "14");
    for (const path of mark?.querySelectorAll("path") ?? []) {
      expect(path).toHaveAttribute("stroke-width", "1.8");
      expect(path).toHaveAttribute("stroke-linecap", "round");
      expect(path).toHaveAttribute("stroke-linejoin", "round");
    }
  });

  it("preserves the existing navigation callbacks", async () => {
    // Given every global destination is wired
    const user = userEvent.setup();
    const { callbacks } = renderNavigation();

    // When each destination is chosen once
    await user.click(screen.getByRole("button", { name: "Home" }));
    await user.click(screen.getByRole("button", { name: "Memories" }));
    await user.click(screen.getByRole("button", { name: "Wiki" }));
    await user.click(screen.getByRole("button", { name: "Graph" }));
    await user.click(screen.getByRole("button", { name: "Sources" }));
    await user.click(screen.getByRole("button", { name: "Spaces" }));

    // Then each existing callback receives its original invocation
    expect(callbacks.home).toHaveBeenCalledTimes(1);
    expect(callbacks.memories).toHaveBeenCalledTimes(1);
    expect(callbacks.pages).toHaveBeenCalledTimes(1);
    expect(callbacks.graph).toHaveBeenCalledTimes(1);
    expect(callbacks.sources).toHaveBeenCalledTimes(1);
    expect(callbacks.spaces).toHaveBeenCalledWith(false);
  });
});
