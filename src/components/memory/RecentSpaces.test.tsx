// SPDX-License-Identifier: AGPL-3.0-only
import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import type { Space } from "../../lib/tauri";
import { RecentSpaces } from "./RecentSpaces";

function space(id: string, name: string, suggested = false, memoryCount = 0): Space {
  return {
    id,
    name,
    description: null,
    suggested,
    starred: false,
    sort_order: 0,
    memory_count: memoryCount,
    entity_count: 0,
    created_at: 0,
    updated_at: 0,
  };
}

describe("RecentSpaces", () => {
  it("caps confirmed rows at four and excludes suggested spaces", () => {
    // Given
    const spaces = [
      space("suggested", "Suggestion", true),
      ...Array.from({ length: 6 }, (_, index) =>
        space(`confirmed-${index}`, `Confirmed ${index}`),
      ),
    ];

    // When
    render(
      <RecentSpaces
        ariaLabel="Recent spaces"
        spaces={spaces}
        currentSpaceId={null}
        onSelectSpace={() => undefined}
      />,
    );

    // Then
    expect(screen.getAllByRole("button")).toHaveLength(4);
    expect(screen.queryByRole("button", { name: "Suggestion" })).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Confirmed 4" })).not.toBeInTheDocument();
  });

  it("keeps active state on a stable id across rename and returns the selected space", () => {
    // Given
    const spaces = [space("alpha-id", "Alpha"), space("beta-id", "Beta")];
    let selected: Space | null = null;
    const onSelectSpace = (next: Space) => { selected = next; };
    const { rerender } = render(
      <RecentSpaces
        ariaLabel="Recent spaces"
        spaces={spaces}
        currentSpaceId="beta-id"
        onSelectSpace={onSelectSpace}
      />,
    );

    // When / Then
    const beta = screen.getByRole("button", { name: "Beta" });
    expect(beta).toHaveAttribute("aria-current", "page");
    expect(beta).not.toHaveAttribute("aria-pressed");
    expect(beta).toHaveAttribute("data-selected", "true");
    fireEvent.click(beta);
    expect(selected).toEqual(spaces[1]);

    // Given / When
    const renamedSpaces = [space("alpha-id", "Renamed Alpha"), spaces[1]];
    rerender(
      <RecentSpaces
        ariaLabel="Recent spaces"
        spaces={renamedSpaces}
        currentSpaceId="alpha-id"
        onSelectSpace={onSelectSpace}
      />,
    );

    // Then
    expect(screen.getByRole("button", { name: "Renamed Alpha" })).toHaveAttribute("aria-current", "page");
    expect(screen.getByRole("button", { name: "Beta" })).not.toHaveAttribute("aria-current");
  });

  it("does not expose an unlabeled memory count", () => {
    render(
      <RecentSpaces
        ariaLabel="Recent spaces"
        spaces={[space("alpha", "Alpha", false, 42)]}
        currentSpaceId={null}
        onSelectSpace={() => undefined}
      />,
    );

    expect(screen.getByRole("button", { name: "Alpha" })).not.toHaveTextContent("42");
    expect(screen.queryByText("42")).not.toBeInTheDocument();
  });
});
