import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { fireEvent, screen, within } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { labels, makeSpace, renderOverview } from "./SpacesOverview.testUtils";

const api = vi.hoisted(() => ({
  listSpaces: vi.fn(),
  listPages: vi.fn(),
  createSpace: vi.fn(),
  updateSpace: vi.fn(),
  deleteSpace: vi.fn(),
  confirmSpace: vi.fn(),
  reorderSpace: vi.fn(),
  toggleSpaceStarred: vi.fn(),
}));

vi.mock("../../../lib/tauri", () => api);

const confirmed = makeSpace({ id: "work", name: "Work" });
const suggested = makeSpace({ id: "suggested", name: "Suggested", suggested: true });
const desktopColumns = ["drag", "name", "pages", "memories", "updated", "menu"];

function directColumnNames(element: HTMLElement): string[] {
  return Array.from(element.children)
    .map((child) => child.getAttribute("data-space-column"))
    .filter((column): column is string => column !== null);
}

describe("SpacesOverview layout contract", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    api.listSpaces.mockResolvedValue([confirmed, suggested]);
    api.listPages.mockResolvedValue([]);
  });

  it("shares one named desktop column contract between the header and confirmed rows", async () => {
    renderOverview();
    const inventory = await screen.findByRole("region", { name: labels.confirmedHeading });
    const header = within(inventory).getByRole("row");
    const row = screen.getByTestId("space-row-work");

    expect(directColumnNames(header)).toEqual(desktopColumns);
    expect(directColumnNames(row)).toEqual(desktopColumns);
    expect(header.querySelector('[data-space-column="updated"]')).not.toBeNull();
    expect(header.querySelector('[data-space-column="menu"]')).not.toBeNull();
    expect(row.querySelector('[data-space-column="updated"]')).not.toBeNull();
    expect(row.querySelector('[data-space-column="menu"]')).not.toBeNull();

    const css = readFileSync(resolve("src/components/memory/spaces/spacesInventory.css"), "utf8");
    expect(css).toMatch(/--spaces-desktop-columns:/);
    expect(css).toMatch(/grid-template-columns:\s*var\(--spaces-desktop-columns\)/);
  });

  it("keeps New space outlined while Suggested decisions stay compact, tonal, and marker-free", async () => {
    renderOverview();
    const newSpace = screen.getByRole("button", { name: labels.newSpace });
    const suggestedRow = await screen.findByTestId("space-row-suggested");
    const keep = within(suggestedRow).getByRole("button", { name: labels.keep });
    const discard = within(suggestedRow).getByRole("button", { name: labels.discard });

    expect(newSpace).toHaveClass("spaces-new-action");
    expect(newSpace).toHaveClass("page-create-action");
    expect(newSpace).not.toHaveClass("spaces-primary-action");
    expect(newSpace.querySelector('[aria-hidden="true"]')).not.toBeInTheDocument();
    expect(keep).toHaveClass("spaces-suggestion-action", "spaces-suggestion-keep");
    expect(discard).toHaveClass("spaces-suggestion-action", "spaces-suggestion-discard");
    expect(keep).not.toHaveClass("spaces-primary-action");
    expect(discard).not.toHaveClass("spaces-primary-action");

    const css = readFileSync(resolve("src/components/memory/spaces/spacesInventory.css"), "utf8");
    const baseCss = readFileSync(resolve("src/components/memory/spaces/spaces.css"), "utf8");
    expect(css).not.toMatch(/\.spaces-row-suggested\s+\.spaces-row-main::before/);
    expect(css).toMatch(/\.spaces-overview-header \.spaces-new-action\s*\{[^}]*align-self:\s*flex-start/s);
    expect(css).toMatch(/\.spaces-overview-header \.spaces-new-action\s*\{[^}]*background:\s*transparent/s);
    expect(css).toMatch(/\.spaces-suggestion-action\s*\{[^}]*font-size:\s*12px[^}]*padding:\s*4px 10px/s);
    expect(baseCss).toMatch(/\.spaces-suggestion-keep\s*\{[^}]*background:\s*transparent[^}]*color:\s*var\(--mem-accent-indigo\)/s);
    expect(baseCss).toMatch(/\.spaces-suggestion-keep:hover\s*\{[^}]*background:\s*var\(--mem-indigo-bg\)/s);
  });

  it("keeps mobile row controls beside the Space identity instead of stacking them as loose rows", () => {
    const css = readFileSync(resolve("src/components/memory/spaces/spacesInventory.css"), "utf8");

    expect(css).toMatch(
      /@media \(max-width: 699px\) \{[\s\S]*?\.spaces-overview \.spaces-row:not\(\.spaces-row-suggested\):not\(\.spaces-row-edit\)\s*\{[^}]*grid-template-columns:\s*28px minmax\(0, 1fr\) 36px;/,
    );
    expect(css).toMatch(/\.spaces-row:not\(\.spaces-row-suggested\) \.spaces-drag-handle\s*\{[^}]*grid-column:\s*1;[^}]*grid-row:\s*1;/s);
    expect(css).toMatch(/\.spaces-row:not\(\.spaces-row-suggested\) \.spaces-row-main\s*\{[^}]*grid-column:\s*2;[^}]*grid-row:\s*1;/s);
    expect(css).toMatch(/\.spaces-row:not\(\.spaces-row-suggested\) \.spaces-menu-anchor\s*\{[^}]*grid-column:\s*3;[^}]*grid-row:\s*1;/s);
    expect(css).toMatch(/\.spaces-row:not\(\.spaces-row-suggested\) \.spaces-mobile-metadata\s*\{[^}]*grid-column:\s*2 \/ 4;[^}]*grid-row:\s*2;/s);
  });

  it("compacts only sub-600 Suggested rows so owned Spaces enter the initial phone viewport", () => {
    const css = readFileSync(resolve("src/components/memory/spaces/spacesInventory.css"), "utf8");

    expect(css).toMatch(
      /@media \(max-width:\s*599px\)\s*\{[\s\S]*?\.spaces-overview \.spaces-row-suggested\s*\{[^}]*grid-template-columns:\s*minmax\(0,\s*1fr\) auto;[^}]*min-height:\s*52px;[^}]*padding:\s*0;/,
    );
    expect(css).toMatch(
      /@media \(max-width:\s*599px\)\s*\{[\s\S]*?\.spaces-row-suggested \.spaces-row-main\s*\{[^}]*padding:\s*8px;/,
    );
    expect(css).toMatch(
      /@media \(max-width:\s*599px\)\s*\{[\s\S]*?\.spaces-row-suggested \.spaces-row-decisions\s*\{[^}]*padding:\s*0 4px 0 0;/,
    );
    expect(css).toMatch(
      /@media \(max-width:\s*599px\)\s*\{[\s\S]*?\.spaces-confirmed-section\s*\{[^}]*padding-top:\s*18px;/,
    );
  });

  it("gives the inline rename editor the full inventory width", async () => {
    renderOverview();
    fireEvent.click(await screen.findByRole("button", { name: labels.actionsFor("Work") }));
    fireEvent.click(screen.getByRole("menuitem", { name: labels.rename }));

    const row = screen.getByTestId("space-row-work");
    const editor = row.querySelector(":scope > .spaces-editor");
    expect(row).toHaveClass("spaces-row-edit");
    expect(editor).not.toBeNull();

    const css = readFileSync(resolve("src/components/memory/spaces/spacesInventory.css"), "utf8");
    const editRowRule = css.match(/\.spaces-overview \.spaces-row-edit\s*\{([^}]*)\}/)?.[1] ?? "";
    const editorRule = css.match(/\.spaces-row-edit\s*>\s*\.spaces-editor\s*\{([^}]*)\}/)?.[1] ?? "";

    expect(editRowRule).toMatch(/grid-template-columns:\s*minmax\(0,\s*1fr\)/);
    expect(editorRule).toMatch(/grid-column:\s*1 \/ -1/);
    expect(editorRule).toMatch(/min-width:\s*0/);
    expect(editorRule).toMatch(/width:\s*100%/);
  });
});
