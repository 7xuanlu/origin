// SPDX-License-Identifier: AGPL-3.0-only
import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import type { Page } from "../../lib/tauri";
import { RecentPages } from "./RecentPages";

function page(id: string, title: string): Page {
  return {
    id,
    title,
    summary: null,
    content: "",
    entity_id: null,
    domain: null,
    source_memory_ids: [],
    version: 1,
    status: "active",
    created_at: "2026-07-16T00:00:00Z",
    last_compiled: "2026-07-16T00:00:00Z",
    last_modified: "2026-07-16T00:00:00Z",
  };
}

describe("RecentPages", () => {
  it("caps rows at four", () => {
    render(
      <RecentPages
        ariaLabel="Recent pages"
        currentPageId={null}
        onSelectPage={() => undefined}
        pages={Array.from({ length: 5 }, (_, index) => page(`page-${index}`, `Page ${index}`))}
      />,
    );

    expect(screen.getAllByRole("button")).toHaveLength(4);
    expect(screen.queryByRole("button", { name: "Page 4" })).not.toBeInTheDocument();
  });

  it("marks the current destination and returns the selected Page", () => {
    const pages = [page("alpha", "Alpha"), page("beta", "Beta")];
    let selected: Page | null = null;
    render(
      <RecentPages
        ariaLabel="Recent pages"
        currentPageId="beta"
        onSelectPage={(next) => { selected = next; }}
        pages={pages}
      />,
    );

    const beta = screen.getByRole("button", { name: "Beta" });
    expect(beta).toHaveAttribute("aria-current", "page");
    expect(beta).not.toHaveAttribute("aria-pressed");
    expect(screen.getByRole("button", { name: "Alpha" })).not.toHaveAttribute("aria-current");
    fireEvent.click(beta);
    expect(selected).toEqual(pages[1]);
  });
});
