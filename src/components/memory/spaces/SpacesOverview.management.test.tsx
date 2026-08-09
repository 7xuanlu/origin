import { fireEvent, screen, waitFor, within } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { i18n } from "../../../i18n";
import { formatLocaleDate } from "../../../lib/dateFormat";
import { createSpacesOverviewLabels } from "../navigation/copy";
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

const work = makeSpace({ id: "work", name: "Work", sort_order: 0 });
const personal = makeSpace({ id: "personal", name: "Personal", sort_order: 1 });
const starred = makeSpace({ id: "starred", name: "Starred", starred: true, sort_order: 0 });
const suggested = makeSpace({
  id: "suggested",
  name: "Suggested",
  description: "Suggested description",
  suggested: true,
});

describe("SpacesOverview management", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    api.listSpaces.mockResolvedValue([work, personal, starred, suggested]);
    api.listPages.mockResolvedValue([]);
    api.updateSpace.mockResolvedValue(work);
    api.confirmSpace.mockResolvedValue(undefined);
    api.deleteSpace.mockResolvedValue(undefined);
    api.reorderSpace.mockResolvedValue(undefined);
    api.toggleSpaceStarred.mockResolvedValue(true);
  });

  it("opens row detail and maps Suggested Keep and Discard directly", async () => {
    // Given a mixed inventory
    const onSelectSpace = vi.fn();
    renderOverview({ onSelectSpace });
    await screen.findByTestId("space-row-suggested");

    // When the row and its decision actions are used
    fireEvent.click(screen.getByRole("button", { name: "Work" }));
    const suggestedRow = screen.getByTestId("space-row-suggested");
    fireEvent.click(within(suggestedRow).getByRole("button", { name: labels.keep }));
    await waitFor(() => expect(api.confirmSpace).toHaveBeenCalledWith("Suggested"));
    await waitFor(() => expect(within(suggestedRow).getByRole("button", { name: labels.discard })).toBeEnabled());
    fireEvent.click(within(suggestedRow).getByRole("button", { name: labels.discard }));

    // Then navigation and direct command mapping are preserved without confirmation
    expect(onSelectSpace).toHaveBeenCalledWith("Work");
    await waitFor(() => expect(api.deleteSpace).toHaveBeenCalledWith("Suggested"));
    expect(within(suggestedRow).queryByRole("button", { name: labels.confirmDelete })).not.toBeInTheDocument();
  });

  it("renders Suggested before the unlabeled inventory with filter and real metadata columns", async () => {
    vi.setSystemTime(new Date("2026-07-10T12:00:00Z"));
    api.listSpaces.mockResolvedValue([
      { ...work, memory_count: 44, updated_at: 1_720_569_600 },
      suggested,
    ]);
    api.listPages.mockResolvedValue([
      { id: "page-1", title: "One", space: "Work", domain: null },
      { id: "page-2", title: "Two", space: null, domain: "Work" },
      { id: "page-3", title: "Legacy", domain: "Work" },
    ]);

    renderOverview();
    const suggestedSection = await screen.findByRole("region", {
      name: `${labels.suggestedHeading} (1)`,
    });
    const inventorySection = screen.getByRole("region", { name: labels.confirmedHeading });
    const workRow = screen.getByTestId("space-row-work");

    expect(
      suggestedSection.compareDocumentPosition(inventorySection) & Node.DOCUMENT_POSITION_FOLLOWING,
    ).toBeTruthy();
    expect(within(suggestedSection).getByText("Suggested description")).toBeInTheDocument();
    expect(within(inventorySection).queryByRole("heading", { name: labels.confirmedHeading })).not.toBeInTheDocument();
    expect(within(inventorySection).getByLabelText(labels.filterLabel)).toBeInTheDocument();
    expect(within(inventorySection).getByRole("columnheader", { name: labels.pages })).toBeInTheDocument();
    expect(within(inventorySection).getByRole("columnheader", { name: labels.memories })).toBeInTheDocument();
    expect(within(inventorySection).getByRole("columnheader", { name: labels.updated })).toBeInTheDocument();
    expect(within(workRow).getByTestId("space-pages")).toHaveTextContent("2");
    expect(within(workRow).getByTestId("space-memories")).toHaveTextContent("44");
    expect(within(workRow).getByTestId("space-updated")).toHaveTextContent(
      formatLocaleDate(new Date(1_720_569_600_000)).label,
    );
  });

  it("omits the Suggested section when no suggestions match", async () => {
    api.listSpaces.mockResolvedValue([work]);

    renderOverview();

    await screen.findByTestId("space-row-work");
    expect(screen.queryByRole("region", { name: /Suggested/ })).not.toBeInTheDocument();
    expect(screen.queryByText(labels.noSuggestions)).not.toBeInTheDocument();
  });

  it.each([
    ["en", "Pages", "Memories", "Updated"],
    ["zh-Hans", "页面", "记忆", "更新"],
    ["zh-Hant", "頁面", "記憶", "更新"],
  ] as const)(
    "renders localized desktop and mobile metadata labels in %s",
    async (locale, pagesLabel, memoriesLabel, updatedLabel) => {
      // Given one confirmed space and labels built from the selected locale
      api.listSpaces.mockResolvedValue([work]);
      const localizedLabels = createSpacesOverviewLabels(i18n.getFixedT(locale));

      // When the overview renders both responsive metadata structures
      renderOverview({ labels: localizedLabels });
      const inventory = await screen.findByRole("region", {
        name: localizedLabels.confirmedHeading,
      });
      const row = screen.getByTestId("space-row-work");

      // Then desktop column headers and mobile definition labels are localized
      expect(within(inventory).getByRole("columnheader", { name: pagesLabel })).toBeInTheDocument();
      expect(within(inventory).getByRole("columnheader", { name: memoriesLabel })).toBeInTheDocument();
      expect(within(inventory).getByRole("columnheader", { name: updatedLabel })).toBeInTheDocument();
      expect(within(row).getByTestId("space-mobile-pages")).toHaveTextContent(pagesLabel);
      expect(within(row).getByTestId("space-mobile-memories")).toHaveTextContent(memoriesLabel);
      expect(within(row).getByTestId("space-mobile-updated")).toHaveTextContent(updatedLabel);
    },
  );

  it("supports star, rename Enter/Escape, and explicit confirmed delete", async () => {
    // Given a confirmed row menu
    renderOverview();
    await screen.findByText("Work");
    const actions = screen.getByRole("button", { name: labels.actionsFor("Work") });

    // When Star is selected
    fireEvent.click(actions);
    fireEvent.click(screen.getByRole("menuitem", { name: labels.star }));
    await waitFor(() => expect(api.toggleSpaceStarred).toHaveBeenCalledWith("Work"));
    await waitFor(() => expect(actions).toBeEnabled());

    // When rename is cancelled and then submitted
    fireEvent.click(actions);
    fireEvent.click(screen.getByRole("menuitem", { name: labels.rename }));
    const renameInput = screen.getByLabelText(labels.nameLabel);
    fireEvent.change(renameInput, { target: { value: "Ignored" } });
    fireEvent.keyDown(renameInput, { key: "Escape" });
    expect(api.updateSpace).not.toHaveBeenCalled();
    const actionsAfterCancel = screen.getByRole("button", { name: labels.actionsFor("Work") });
    fireEvent.click(actionsAfterCancel);
    fireEvent.click(screen.getByRole("menuitem", { name: labels.rename }));
    fireEvent.change(screen.getByLabelText(labels.nameLabel), { target: { value: "  Studio  " } });
    fireEvent.keyDown(screen.getByLabelText(labels.nameLabel), { key: "Enter" });
    await waitFor(() => expect(api.updateSpace).toHaveBeenCalledWith("Work", "Studio", "Projects and planning"));

    // When Delete is selected, the API waits for explicit confirmation
    const actionsAfterRename = screen.getByRole("button", { name: labels.actionsFor("Work") });
    fireEvent.click(actionsAfterRename);
    fireEvent.click(screen.getByRole("menuitem", { name: labels.delete }));
    expect(api.deleteSpace).not.toHaveBeenCalled();
    fireEvent.click(screen.getByRole("button", { name: labels.confirmDelete }));
    await waitFor(() => expect(api.deleteSpace).toHaveBeenCalledWith("Work"));
  });

  it("restores focus to the menu trigger when Escape closes the menu", async () => {
    // Given an open row menu
    renderOverview();
    await screen.findByText("Work");
    const trigger = screen.getByRole("button", { name: labels.actionsFor("Work") });
    fireEvent.click(trigger);
    expect(screen.getByRole("menu")).toBeInTheDocument();

    // When Escape is pressed inside the menu
    fireEvent.keyDown(screen.getByRole("menu"), { key: "Escape" });

    // Then the menu closes and focus returns to its trigger
    expect(screen.queryByRole("menu")).not.toBeInTheDocument();
    expect(trigger).toHaveFocus();
  });

  it("routes menu and pointer moves through the same within-group reorder command", async () => {
    // Given ordered starred and unstarred groups
    renderOverview();
    await screen.findByText("Work");

    // When moving Personal up through the menu
    fireEvent.click(screen.getByRole("button", { name: labels.actionsFor("Personal") }));
    fireEvent.click(screen.getByRole("menuitem", { name: labels.moveUp }));
    await waitFor(() => expect(api.reorderSpace).toHaveBeenCalledWith("Personal", 0));

    // When dragging Work onto Personal
    const workRow = screen.getByTestId("space-row-work");
    const personalRow = screen.getByTestId("space-row-personal");
    fireEvent.pointerDown(within(workRow).getByRole("button", { name: labels.dragSpace("Work") }));
    fireEvent.pointerEnter(personalRow);
    fireEvent.pointerUp(personalRow);

    // Then the same API path receives the target order, while group boundary moves stay disabled
    await waitFor(() => expect(api.reorderSpace).toHaveBeenLastCalledWith("Work", 1));
    fireEvent.click(screen.getByRole("button", { name: labels.actionsFor("Work") }));
    expect(screen.getByRole("menuitem", { name: labels.moveUp })).toBeDisabled();
  });

  it("isolates pending and error state to the affected row and invalidates dependent data", async () => {
    // Given a Keep request that remains pending while another row exists
    let resolveKeep: () => void = () => undefined;
    api.confirmSpace.mockReturnValue(new Promise<void>((resolve) => { resolveKeep = resolve; }));
    const { queryClient } = renderOverview();
    const invalidate = vi.spyOn(queryClient, "invalidateQueries");
    await screen.findByTestId("space-row-suggested");
    const suggestedRow = screen.getByTestId("space-row-suggested");

    // When Keep is double-submitted
    const keep = within(suggestedRow).getByRole("button", { name: labels.keep });
    fireEvent.click(keep);
    fireEvent.click(keep);

    // Then only one request starts and other rows remain actionable
    await waitFor(() => expect(api.confirmSpace).toHaveBeenCalledTimes(1));
    expect(screen.getByRole("button", { name: labels.actionsFor("Work") })).toBeEnabled();

    // Given a later failed star mutation
    resolveKeep();
    await waitFor(() => expect(keep).toBeEnabled());
    api.toggleSpaceStarred.mockRejectedValue(new Error("offline"));
    fireEvent.click(screen.getByRole("button", { name: labels.actionsFor("Work") }));
    fireEvent.click(screen.getByRole("menuitem", { name: labels.star }));

    // Then last-good rows remain with an inline error
    expect(await screen.findByRole("alert")).toHaveTextContent(labels.mutationError);
    expect(screen.getByRole("button", { name: "Work" })).toBeEnabled();

    // When a successful delete completes
    api.toggleSpaceStarred.mockResolvedValue(true);
    fireEvent.click(screen.getByRole("button", { name: labels.actionsFor("Personal") }));
    fireEvent.click(screen.getByRole("menuitem", { name: labels.delete }));
    fireEvent.click(screen.getByRole("button", { name: labels.confirmDelete }));
    await waitFor(() => expect(api.deleteSpace).toHaveBeenCalledWith("Personal"));

    // Then spaces and dependent page/cache families are invalidated
    await waitFor(() => {
      const keys = invalidate.mock.calls.map(([filters]) => filters?.queryKey?.[0]);
      expect(keys).toEqual(expect.arrayContaining(["spaces", "pages", "space-pages", "memories"]));
    });
  });
});
