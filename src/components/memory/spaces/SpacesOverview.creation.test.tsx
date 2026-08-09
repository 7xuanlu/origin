import { act, fireEvent, screen, waitFor } from "@testing-library/react";
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

function submitNearestForm(element: HTMLElement): void {
  const form = element.closest("form");
  if (form === null) throw new TypeError("Expected the input to belong to a form");
  fireEvent.submit(form);
}

describe("SpacesOverview creation and query states", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    api.listSpaces.mockResolvedValue([]);
    api.listPages.mockResolvedValue([]);
    api.createSpace.mockResolvedValue(makeSpace({ id: "new", name: "New" }));
  });

  it("replaces loading with the inventory empty state and omits empty suggestions", async () => {
    // Given an unresolved initial request
    let resolveSpaces: (value: readonly ReturnType<typeof makeSpace>[]) => void = () => undefined;
    api.listSpaces.mockReturnValue(new Promise((resolve) => { resolveSpaces = resolve; }));

    // When the overview renders and the request later succeeds empty
    renderOverview();
    expect(screen.getByText(labels.loading)).toBeInTheDocument();
    await act(async () => resolveSpaces([]));

    // Then the inventory state replaces loading without an empty Suggested section
    expect(await screen.findByText(labels.noConfirmed)).toBeInTheDocument();
    expect(screen.queryByText(labels.noSuggestions)).not.toBeInTheDocument();
  });

  it("autofocuses create, submits trimmed fields on Enter, and cancels on Escape", async () => {
    // Given an empty overview
    renderOverview();
    await screen.findByText(labels.noConfirmed);

    // When the create form is opened and submitted from the name field
    fireEvent.click(screen.getByRole("button", { name: labels.newSpace }));
    const nameInput = screen.getByLabelText(labels.nameLabel);
    expect(nameInput).toHaveFocus();
    fireEvent.change(nameInput, { target: { value: "  Research  " } });
    fireEvent.change(screen.getByLabelText(labels.descriptionLabel), {
      target: { value: "  Reading queue  " },
    });
    fireEvent.keyDown(nameInput, { key: "Enter" });

    // Then the API receives normalized values
    await waitFor(() => expect(api.createSpace).toHaveBeenCalledWith("Research", "Reading queue"));

    // When another form is cancelled
    fireEvent.click(screen.getByRole("button", { name: labels.newSpace }));
    fireEvent.keyDown(screen.getByLabelText(labels.nameLabel), { key: "Escape" });

    // Then the form closes without another mutation
    expect(screen.queryByLabelText(labels.nameLabel)).not.toBeInTheDocument();
    await waitFor(() => expect(api.createSpace).toHaveBeenCalledTimes(1));
  });

  it("blocks blank, duplicate, and repeated pending create submissions", async () => {
    // Given an existing space and a create request that stays pending
    api.listSpaces.mockResolvedValue([makeSpace({ id: "work", name: "Work" })]);
    api.createSpace.mockReturnValue(new Promise(() => undefined));
    renderOverview();
    await screen.findByText("Work");
    fireEvent.click(screen.getByRole("button", { name: labels.newSpace }));
    const input = screen.getByLabelText(labels.nameLabel);

    // When blank and duplicate values are submitted
    fireEvent.change(input, { target: { value: "   " } });
    submitNearestForm(input);
    expect(screen.getByText(labels.nameRequired)).toBeInTheDocument();
    fireEvent.change(input, { target: { value: "  wOrK " } });
    submitNearestForm(input);
    expect(screen.getByText(labels.duplicateName)).toBeInTheDocument();
    expect(api.createSpace).not.toHaveBeenCalled();

    // When a valid value is submitted twice before it resolves
    fireEvent.change(input, { target: { value: "Personal" } });
    submitNearestForm(input);
    submitNearestForm(input);

    // Then exactly one mutation starts and unrelated row navigation remains enabled
    await waitFor(() => expect(api.createSpace).toHaveBeenCalledTimes(1));
    expect(screen.getByRole("button", { name: "Work" })).toBeEnabled();
  });

  it("filters both sections without running a mutation", async () => {
    // Given confirmed and suggested spaces
    api.listSpaces.mockResolvedValue([
      makeSpace({ id: "work", name: "Work", description: "Client projects" }),
      makeSpace({ id: "health", name: "Health", description: "Fitness", suggested: true }),
    ]);
    renderOverview();
    await screen.findByText("Work");

    // When filtering by description and then an absent term
    fireEvent.change(screen.getByLabelText(labels.filterLabel), { target: { value: "  FITness " } });
    expect(screen.getByText("Health")).toBeInTheDocument();
    expect(screen.queryByText("Work")).not.toBeInTheDocument();
    fireEvent.change(screen.getByLabelText(labels.filterLabel), { target: { value: "missing" } });

    // Then the no-results state appears without API side effects
    expect(screen.getByText(labels.noResults)).toBeInTheDocument();
    expect(api.createSpace).not.toHaveBeenCalled();
    expect(api.updateSpace).not.toHaveBeenCalled();
    expect(api.deleteSpace).not.toHaveBeenCalled();
  });

  it("retains last-good rows when retry fails and exposes recovery", async () => {
    // Given one successful load followed by a failed retry
    api.listSpaces
      .mockResolvedValueOnce([makeSpace({ id: "work", name: "Work" })])
      .mockRejectedValueOnce(new Error("offline"));
    const { queryClient } = renderOverview();
    await screen.findByText("Work");

    // When the query is retried
    await queryClient.invalidateQueries({ queryKey: ["spaces"] });

    // Then the error is inline and the last-good inventory remains usable
    expect(await screen.findByRole("alert")).toHaveTextContent(labels.loadError);
    expect(screen.getByRole("button", { name: labels.retry })).toBeEnabled();
    expect(screen.getByRole("button", { name: "Work" })).toBeEnabled();
  });
});
