import { fireEvent, screen, waitFor, within } from "@testing-library/react";
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

const REQUIRED_QUERY_KEYS = [
  "spaces",
  "spaces-page-counts",
  "pages",
  "sidebar-space-page-counts",
  "space",
  "space-pages",
  "space-memories",
  "space-entities",
  "memories",
] as const;

const work = makeSpace({ id: "work", name: "Work", sort_order: 0 });
const personal = makeSpace({ id: "personal", name: "Personal", sort_order: 1 });
const suggested = makeSpace({ id: "suggested", name: "Suggested", suggested: true });
const mutationKinds = ["confirm", "create", "delete", "rename", "reorder", "star"] as const;
type MutationKind = (typeof mutationKinds)[number];

function assertNever(value: never): never {
  throw new TypeError(`Unsupported test mutation: ${String(value)}`);
}

function performMutation(kind: MutationKind): void {
  switch (kind) {
    case "confirm":
      fireEvent.click(within(screen.getByTestId("space-row-suggested")).getByRole("button", { name: labels.keep }));
      return;
    case "create":
      fireEvent.click(screen.getByRole("button", { name: labels.newSpace }));
      fireEvent.change(screen.getByLabelText(labels.nameLabel), { target: { value: "Studio" } });
      fireEvent.click(screen.getByRole("button", { name: labels.create }));
      return;
    case "delete":
      fireEvent.click(screen.getByRole("button", { name: labels.actionsFor("Work") }));
      fireEvent.click(screen.getByRole("menuitem", { name: labels.delete }));
      fireEvent.click(screen.getByRole("button", { name: labels.confirmDelete }));
      return;
    case "rename":
      fireEvent.click(screen.getByRole("button", { name: labels.actionsFor("Work") }));
      fireEvent.click(screen.getByRole("menuitem", { name: labels.rename }));
      fireEvent.change(screen.getByLabelText(labels.nameLabel), { target: { value: "Studio" } });
      fireEvent.keyDown(screen.getByLabelText(labels.nameLabel), { key: "Enter" });
      return;
    case "reorder":
      fireEvent.click(screen.getByRole("button", { name: labels.actionsFor("Personal") }));
      fireEvent.click(screen.getByRole("menuitem", { name: labels.moveUp }));
      return;
    case "star":
      fireEvent.click(screen.getByRole("button", { name: labels.actionsFor("Work") }));
      fireEvent.click(screen.getByRole("menuitem", { name: labels.star }));
      return;
    default:
      return assertNever(kind);
  }
}

describe("SpacesOverview dependent cache invalidation", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    api.listSpaces.mockResolvedValue([work, personal, suggested]);
    api.listPages.mockResolvedValue([]);
    api.createSpace.mockResolvedValue(undefined);
    api.updateSpace.mockResolvedValue(work);
    api.deleteSpace.mockResolvedValue(undefined);
    api.confirmSpace.mockResolvedValue(undefined);
    api.reorderSpace.mockResolvedValue(undefined);
    api.toggleSpaceStarred.mockResolvedValue(true);
  });

  it.each(mutationKinds)("invalidates the exact dependent query set after successful %s", async (kind) => {
    // Given every Spaces query is active and invalidation calls are observed
    const { queryClient } = renderOverview();
    const invalidate = vi.spyOn(queryClient, "invalidateQueries");
    await screen.findByTestId("space-row-suggested");

    // When one mutation succeeds
    performMutation(kind);

    // Then each required cache family is invalidated exactly once
    await waitFor(() => expect(invalidate).toHaveBeenCalledTimes(REQUIRED_QUERY_KEYS.length));
    const keys = invalidate.mock.calls.map(([filters]) => filters?.queryKey?.[0]);
    expect([...keys].sort()).toEqual([...REQUIRED_QUERY_KEYS].sort());
  });

  it("refreshes page counts under the renamed Space name", async () => {
    // Given the active queries initially map one page to Work, then return Studio after rename
    const studio = { ...work, name: "Studio" };
    api.listSpaces.mockResolvedValueOnce([work]).mockResolvedValue([studio]);
    api.listPages
      .mockResolvedValueOnce([{ id: "page-1", title: "One", space: "Work", domain: null }])
      .mockResolvedValue([{ id: "page-1", title: "One", space: "Studio", domain: null }]);
    renderOverview();
    const workRow = await screen.findByTestId("space-row-work");
    expect(within(workRow).getByTestId("space-pages")).toHaveTextContent("1");

    // When Work is renamed to Studio
    fireEvent.click(screen.getByRole("button", { name: labels.actionsFor("Work") }));
    fireEvent.click(screen.getByRole("menuitem", { name: labels.rename }));
    fireEvent.change(screen.getByLabelText(labels.nameLabel), { target: { value: "Studio" } });
    fireEvent.keyDown(screen.getByLabelText(labels.nameLabel), { key: "Enter" });

    // Then the refreshed Space row uses the refreshed page-count fixture instead of stale zero
    const studioRow = await screen.findByTestId("space-row-work");
    await waitFor(() => expect(within(studioRow).getByRole("button", { name: "Studio" })).toBeInTheDocument());
    await waitFor(() => expect(within(studioRow).getByTestId("space-pages")).toHaveTextContent("1"));
    expect(api.listSpaces).toHaveBeenCalledTimes(2);
    expect(api.listPages).toHaveBeenCalledTimes(2);
  });

  it("does not invalidate dependent queries after a failed mutation", async () => {
    // Given a failed Keep mutation and an invalidation observer
    api.confirmSpace.mockRejectedValueOnce("offline");
    const { queryClient } = renderOverview();
    const invalidate = vi.spyOn(queryClient, "invalidateQueries");
    const suggestedRow = await screen.findByTestId("space-row-suggested");

    // When Keep rejects
    fireEvent.click(within(suggestedRow).getByRole("button", { name: labels.keep }));

    // Then the safe failure state appears without invalidating last-good data
    expect(await screen.findByRole("alert")).toHaveTextContent(labels.mutationError);
    expect(invalidate).not.toHaveBeenCalled();
  });
});
