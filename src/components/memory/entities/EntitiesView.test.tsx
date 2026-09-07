// SPDX-License-Identifier: AGPL-3.0-only
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { cleanup, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import type { Entity, ListEntitiesRequest } from "../../../lib/tauri";
import { EntitiesView } from "./EntitiesView";

vi.mock("sonner", () => ({ toast: { success: vi.fn(), error: vi.fn() } }));

vi.mock("../../../lib/tauri", async (importOriginal) => ({
  ...(await importOriginal<typeof import("../../../lib/tauri")>()),
  queryEntities: vi.fn(),
  archiveEntities: vi.fn(),
  restoreEntities: vi.fn(),
  confirmEntity: vi.fn(),
  deleteEntity: vi.fn(),
}));

function entity(overrides: Partial<Entity> & { id: string; name: string }): Entity {
  return {
    entity_type: "concept",
    domain: null,
    source_agent: null,
    confidence: null,
    confirmed: false,
    created_at: 1_700_000_000,
    updated_at: 1_700_000_000,
    memory_count: 0,
    status: "detected",
    established_by: null,
    ...overrides,
  };
}

// A small in-memory stand-in for the daemon's /entities/query and
// /entities/archive|restore routes (crates/wenlan-server/src/entity_graph_routes.rs),
// faithful enough to exercise the view's real request-building and its
// archive-all-matching / restore round trip end to end.
function matchesFilter(candidate: Entity, filter: ListEntitiesRequest): boolean {
  if (filter.status && candidate.status !== filter.status) return false;
  if (filter.entity_type && candidate.entity_type !== filter.entity_type) return false;
  if (typeof filter.min_memories === "number" && candidate.memory_count < filter.min_memories) return false;
  if (typeof filter.max_memories === "number" && candidate.memory_count > filter.max_memories) return false;
  if (filter.query && !candidate.name.toLowerCase().includes(filter.query.toLowerCase())) return false;
  return true;
}

let fixture: Entity[];

function seedFixture(): Entity[] {
  return [
    entity({ id: "ada", name: "Ada Lovelace", entity_type: "person", status: "detected", memory_count: 0 }),
    entity({ id: "babbage", name: "Charles Babbage", entity_type: "person", status: "detected", memory_count: 2 }),
    entity({
      id: "engine",
      name: "Analytical Engine",
      entity_type: "concept",
      status: "established",
      memory_count: 5,
      established_by: "auto:memories",
      confirmed: true,
    }),
    entity({
      id: "countess",
      name: "Countess of Lovelace",
      entity_type: "person",
      status: "archived",
      memory_count: 1,
      confirmed: true,
      established_by: "manual",
    }),
  ];
}

beforeEach(async () => {
  fixture = seedFixture();
  const tauri = await import("../../../lib/tauri");

  vi.mocked(tauri.queryEntities).mockImplementation(async (filter) => {
    const all = fixture.filter((candidate) => matchesFilter(candidate, filter));
    const offset = filter.offset ?? 0;
    const limit = filter.limit ?? 100;
    return { entities: all.slice(offset, offset + limit), total: all.length };
  });

  vi.mocked(tauri.archiveEntities).mockImplementation(async (req) => {
    const eligible = fixture.filter((candidate) => candidate.status !== "archived");
    const selected = req.ids
      ? eligible.filter((candidate) => req.ids?.includes(candidate.id))
      : eligible.filter((candidate) => matchesFilter(candidate, req.filter ?? {}));
    if (!req.dry_run) {
      const ids = new Set(selected.map((candidate) => candidate.id));
      fixture = fixture.map((candidate) =>
        ids.has(candidate.id) ? { ...candidate, status: "archived" } : candidate,
      );
    }
    return { count: selected.length, entity_ids: selected.map((candidate) => candidate.id), dry_run: req.dry_run };
  });

  vi.mocked(tauri.restoreEntities).mockImplementation(async (req) => {
    const eligible = fixture.filter((candidate) => candidate.status === "archived");
    const selected = req.ids
      ? eligible.filter((candidate) => req.ids?.includes(candidate.id))
      : eligible.filter((candidate) => matchesFilter(candidate, req.filter ?? {}));
    if (!req.dry_run) {
      const ids = new Set(selected.map((candidate) => candidate.id));
      fixture = fixture.map((candidate) =>
        ids.has(candidate.id)
          ? { ...candidate, status: candidate.confirmed ? "established" : "detected" }
          : candidate,
      );
    }
    return { count: selected.length, entity_ids: selected.map((candidate) => candidate.id), dry_run: req.dry_run };
  });

  vi.mocked(tauri.confirmEntity).mockImplementation(async (id, confirmed) => {
    fixture = fixture.map((candidate) =>
      candidate.id === id
        ? { ...candidate, confirmed, status: confirmed ? "established" : candidate.status, established_by: confirmed ? "manual" : candidate.established_by }
        : candidate,
    );
  });

  vi.mocked(tauri.deleteEntity).mockImplementation(async (id) => {
    fixture = fixture.filter((candidate) => candidate.id !== id);
  });
});

afterEach(() => {
  cleanup();
  vi.clearAllMocks();
});

function renderView(onEntityClick = vi.fn()) {
  return { onEntityClick, user: userEvent.setup(), ...render(<EntitiesView onEntityClick={onEntityClick} />) };
}

async function openTab(user: ReturnType<typeof userEvent.setup>, name: RegExp) {
  await user.click(screen.getByRole("tab", { name }));
}

describe("EntitiesView", () => {
  it("opens on the Detected tab and lists its rows with tab counts", async () => {
    renderView();

    expect(await screen.findByText("Ada Lovelace")).toBeInTheDocument();
    expect(screen.getByText("Charles Babbage")).toBeInTheDocument();
    expect(screen.queryByText("Analytical Engine")).not.toBeInTheDocument();
    expect(screen.getByRole("tab", { name: /Detected/ })).toHaveAttribute("aria-selected", "true");
    // 1 established (Engine), 2 detected (Ada, Babbage), 1 archived (Countess).
    expect(screen.getByRole("tab", { name: /Established/ })).toHaveTextContent("1");
    expect(screen.getByRole("tab", { name: /Detected/ })).toHaveTextContent("2");
    expect(screen.getByRole("tab", { name: /Archived/ })).toHaveTextContent("1");
  });

  it("filters the Detected tab by the Type chip", async () => {
    const { user } = renderView();
    await screen.findByText("Ada Lovelace");

    await user.click(screen.getByRole("button", { name: "Concept" }));

    expect(await screen.findByText("No detected entities match")).toBeInTheDocument();
    expect(screen.queryByText("Ada Lovelace")).not.toBeInTheDocument();
  });

  it("establishes a selected entity from the selection bar", async () => {
    const { user } = renderView();
    await screen.findByText("Ada Lovelace");

    await user.click(screen.getByRole("checkbox", { name: "Select Ada Lovelace" }));
    await user.click(screen.getByRole("button", { name: "Establish selected" }));

    await screen.findByText("Charles Babbage");
    expect(screen.queryByText("Ada Lovelace")).not.toBeInTheDocument();

    await openTab(user, /Established/);
    expect(await screen.findByText("Ada Lovelace")).toBeInTheDocument();
  });

  it("opens the dossier from an Established row", async () => {
    const { user, onEntityClick } = renderView();
    await screen.findByText("Ada Lovelace");
    await openTab(user, /Established/);

    await user.click(await screen.findByRole("button", { name: "Analytical Engine" }));
    expect(onEntityClick).toHaveBeenCalledWith("engine");
    expect(screen.getByText("5 memories")).toBeInTheDocument();
  });

  it("archives all matching via dry-run-then-confirm, and restores them back by their own state", async () => {
    const { user } = renderView();
    await screen.findByText("Ada Lovelace");

    await user.click(screen.getByRole("button", { name: "Person" }));
    await screen.findByText("Ada Lovelace");
    await screen.findByText("Charles Babbage");

    await user.click(screen.getByRole("button", { name: "Archive all matching" }));

    const dialog = await screen.findByRole("dialog");
    expect(within(dialog).getByText("Archive 2 detected entities?")).toBeInTheDocument();
    expect(within(dialog).getByText("Filter")).toBeInTheDocument();
    expect(within(dialog).getByText(/Person/)).toBeInTheDocument();
    // Babbage (2 memories) is among the Person matches, so the dialog warns
    // that archiving takes memories with it.
    expect(within(dialog).getByText("Includes")).toBeInTheDocument();
    expect(within(dialog).getByText("entities that already have memories")).toBeInTheDocument();
    expect(
      within(dialog).getByText(
        "To keep those, set Memories to None first. Archived entities can be restored from the Archived tab.",
      ),
    ).toBeInTheDocument();

    await user.click(within(dialog).getByRole("button", { name: "Archive" }));

    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
    await screen.findByText("No detected entities match");

    await openTab(user, /Archived/);
    await screen.findByText("Ada Lovelace");
    await screen.findByText("Charles Babbage");
    await screen.findByText("Countess of Lovelace");

    const selectAll = screen.getByRole("checkbox", { name: "Select all" });
    await user.click(selectAll);
    await user.click(screen.getByRole("button", { name: "Restore selected" }));

    // Ada and Babbage were never established, so they land back on Detected;
    // the Countess was confirmed before archiving, so she returns Established.
    await openTab(user, /Detected/);
    await screen.findByText("Ada Lovelace");
    await screen.findByText("Charles Babbage");
    await openTab(user, /Established/);
    await screen.findByText("Countess of Lovelace");
  });

  it("shows an Includes line when the matched entities still have memories", async () => {
    const { user } = renderView();
    await screen.findByText("Ada Lovelace");
    await screen.findByText("Charles Babbage");

    // Default Memories chip is "any"; Babbage (2 memories) matches alongside Ada (0).
    await user.click(screen.getByRole("button", { name: "Archive all matching" }));

    const dialog = await screen.findByRole("dialog");
    expect(within(dialog).getByText("Includes")).toBeInTheDocument();
    expect(within(dialog).getByText("entities that already have memories")).toBeInTheDocument();
    expect(within(dialog).getByText("1")).toBeInTheDocument();
    expect(
      within(dialog).getByText(
        "To keep those, set Memories to None first. Archived entities can be restored from the Archived tab.",
      ),
    ).toBeInTheDocument();
  });

  it("omits the Includes line once the Memories chip already excludes entities with memories", async () => {
    const { user } = renderView();
    await screen.findByText("Ada Lovelace");

    await user.click(screen.getByRole("button", { name: "None" }));
    await screen.findByText("Ada Lovelace");
    expect(screen.queryByText("Charles Babbage")).not.toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "Archive all matching" }));

    const dialog = await screen.findByRole("dialog");
    expect(within(dialog).getByText("Archive 1 detected entity?")).toBeInTheDocument();
    expect(within(dialog).queryByText("Includes")).not.toBeInTheDocument();
    expect(
      within(dialog).getByText("Archived entities can be restored from the Archived tab."),
    ).toBeInTheDocument();
  });

  it("deletes one archived entity permanently from its row, with an irreversible confirm", async () => {
    const { user } = renderView();
    await screen.findByText("Ada Lovelace");
    await openTab(user, /Archived/);
    await screen.findByText("Countess of Lovelace");

    // Permanent delete is per row (spec): the selection bar only restores.
    await user.click(screen.getByRole("checkbox", { name: "Select Countess of Lovelace" }));
    expect(screen.getByRole("button", { name: "Restore selected" })).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Delete permanently" })).not.toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "Delete Countess of Lovelace permanently" }));

    const dialog = await screen.findByRole("dialog");
    expect(within(dialog).getByText("Delete Countess of Lovelace permanently?")).toBeInTheDocument();
    expect(within(dialog).getByText("This cannot be undone.")).toBeInTheDocument();
    // The safe action takes focus, not the destructive one.
    expect(within(dialog).getByRole("button", { name: "Cancel" })).toHaveFocus();

    await user.click(within(dialog).getByRole("button", { name: "Delete permanently" }));

    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
    expect(await screen.findByText("No archived entities")).toBeInTheDocument();
    const tauri = await import("../../../lib/tauri");
    expect(tauri.deleteEntity).toHaveBeenCalledTimes(1);
    expect(tauri.deleteEntity).toHaveBeenCalledWith("countess");
  });

  it("keeps the confirm open on Escape while the delete is in flight", async () => {
    const tauri = await import("../../../lib/tauri");
    let release: () => void = () => {};
    vi.mocked(tauri.deleteEntity).mockImplementation(
      (id) =>
        new Promise<void>((resolve) => {
          release = () => {
            fixture = fixture.filter((candidate) => candidate.id !== id);
            resolve();
          };
        }),
    );
    const { user } = renderView();
    await screen.findByText("Ada Lovelace");
    await openTab(user, /Archived/);
    await screen.findByText("Countess of Lovelace");
    await user.click(screen.getByRole("button", { name: "Delete Countess of Lovelace permanently" }));
    const dialog = await screen.findByRole("dialog");
    await user.click(within(dialog).getByRole("button", { name: "Delete permanently" }));

    await user.keyboard("{Escape}");
    expect(screen.getByRole("dialog")).toBeInTheDocument();

    release();
    await waitFor(() => expect(screen.queryByRole("dialog")).not.toBeInTheDocument());
    expect(await screen.findByText("No archived entities")).toBeInTheDocument();
  });

  it("drops a list response that arrives after the tab changed", async () => {
    const tauri = await import("../../../lib/tauri");
    let releaseDetected: () => void = () => {};
    vi.mocked(tauri.queryEntities).mockImplementation(async (filter) => {
      const all = fixture.filter((candidate) => matchesFilter(candidate, filter));
      const offset = filter.offset ?? 0;
      const limit = filter.limit ?? 100;
      const page = { entities: all.slice(offset, offset + limit), total: all.length };
      // The Detected LIST (not the limit-1 count probe) hangs until released.
      if (filter.status === "detected" && limit !== 1) {
        await new Promise<void>((resolve) => {
          releaseDetected = resolve;
        });
      }
      return page;
    });

    const { user } = renderView();
    await screen.findByRole("tab", { name: /Archived/ });
    await openTab(user, /Archived/);
    await screen.findByText("Countess of Lovelace");

    releaseDetected();
    // Give the stale promise every chance to land, then assert it did not.
    await new Promise((resolve) => setTimeout(resolve, 20));
    expect(screen.getByText("Countess of Lovelace")).toBeInTheDocument();
    expect(screen.queryByText("Ada Lovelace")).not.toBeInTheDocument();
    expect(screen.queryByText("Charles Babbage")).not.toBeInTheDocument();
  });

  it("reads back and applies the search term as typed, even before the debounce lands", async () => {
    const tauri = await import("../../../lib/tauri");
    const { user } = renderView();
    await screen.findByText("Ada Lovelace");

    await user.type(screen.getByRole("searchbox", { name: "Find a name" }), "Ada");
    await user.click(screen.getByRole("button", { name: "Archive all matching" }));

    const dialog = await screen.findByRole("dialog");
    expect(within(dialog).getByText(/"Ada"/)).toBeInTheDocument();
    expect(within(dialog).getByText("Archive 1 detected entity?")).toBeInTheDocument();
    const dryRuns = vi.mocked(tauri.archiveEntities).mock.calls.filter(([req]) => req.dry_run);
    expect(dryRuns.length).toBeGreaterThan(0);
    for (const [req] of dryRuns) expect(req.filter?.query).toBe("Ada");

    await user.click(within(dialog).getByRole("button", { name: "Archive" }));
    const applied = vi.mocked(tauri.archiveEntities).mock.calls.find(([req]) => !req.dry_run);
    expect(applied?.[0].filter?.query).toBe("Ada");
    await openTab(user, /Archived/);
    await screen.findByText("Ada Lovelace");
    // Only the "Ada" match went; Babbage is still detected (the search box
    // still says "Ada", so he is filtered out of the list, not archived).
    expect(fixture.find((candidate) => candidate.id === "babbage")?.status).toBe("detected");
  });

  it("moves between tabs with the arrow keys", async () => {
    const { user } = renderView();
    await screen.findByText("Ada Lovelace");

    screen.getByRole("tab", { name: /Detected/ }).focus();
    await user.keyboard("{ArrowRight}");
    expect(screen.getByRole("tab", { name: /Archived/ })).toHaveAttribute("aria-selected", "true");
    expect(screen.getByRole("tab", { name: /Archived/ })).toHaveFocus();
    expect(await screen.findByText("Countess of Lovelace")).toBeInTheDocument();

    await user.keyboard("{ArrowRight}");
    expect(screen.getByRole("tab", { name: /Established/ })).toHaveAttribute("aria-selected", "true");
    await user.keyboard("{End}");
    expect(screen.getByRole("tab", { name: /Archived/ })).toHaveAttribute("aria-selected", "true");
    expect(screen.getByRole("tabpanel", { name: /Archived/ })).toBeInTheDocument();
  });

  it("paginates the Detected tab 100 rows at a time with Load more", async () => {
    fixture = Array.from({ length: 130 }, (_, index) =>
      entity({ id: `d${index}`, name: `Detected Entity ${index}`, status: "detected" }),
    );
    const tauri = await import("../../../lib/tauri");
    vi.mocked(tauri.queryEntities).mockImplementation(async (filter) => {
      const all = fixture.filter((candidate) => matchesFilter(candidate, filter));
      const offset = filter.offset ?? 0;
      const limit = filter.limit ?? 100;
      return { entities: all.slice(offset, offset + limit), total: all.length };
    });

    const { user } = renderView();
    await screen.findByText("Detected Entity 0");
    expect(screen.queryByText("Detected Entity 100")).not.toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "Load more" }));

    expect(await screen.findByText("Detected Entity 100")).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Load more" })).not.toBeInTheDocument();
  });
});
