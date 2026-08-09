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

const work = makeSpace({ id: "work", name: "Work" });
const suggested = makeSpace({ id: "suggested", name: "Suggested", suggested: true });
const rejectionCases = [
  { name: "Error", value: new Error("internal Error detail") },
  { name: "odd object", value: { message: "internal object detail" } },
  { name: "null", value: null },
] satisfies readonly { readonly name: string; readonly value: unknown }[];

describe("SpacesOverview mutation errors", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    api.listSpaces.mockResolvedValue([work, suggested]);
    api.listPages.mockResolvedValue([]);
    api.confirmSpace.mockResolvedValue(undefined);
  });

  it("recovers from a string rejection without duplicate submission or inventory loss", async () => {
    // Given a Tauri mutation that stays pending before rejecting with Result<_, String>
    let rejectKeep: (reason?: unknown) => void = () => undefined;
    api.confirmSpace.mockReturnValueOnce(new Promise<void>((_resolve, reject) => {
      rejectKeep = reject;
    }));
    renderOverview();
    const suggestedRow = await screen.findByTestId("space-row-suggested");
    const keep = within(suggestedRow).getByRole("button", { name: labels.keep });

    // When the affected action is rapidly submitted and then rejects
    fireEvent.click(keep);
    fireEvent.click(keep);
    await waitFor(() => expect(api.confirmSpace).toHaveBeenCalledTimes(1));
    expect(keep).toBeDisabled();
    expect(screen.getByRole("button", { name: "Work" })).toBeEnabled();
    rejectKeep("daemon internals: connection refused");

    // Then recovery is localized, safe, and retryable against the last-good inventory
    expect(await screen.findByRole("alert")).toHaveTextContent(labels.mutationError);
    expect(screen.queryByText(/daemon internals/i)).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Suggested" })).toBeEnabled();
    await waitFor(() => expect(keep).toBeEnabled());
    api.confirmSpace.mockResolvedValueOnce(undefined);
    fireEvent.click(keep);
    await waitFor(() => expect(api.confirmSpace).toHaveBeenCalledTimes(2));
    await waitFor(() => expect(screen.queryByRole("alert")).not.toBeInTheDocument());
  });

  it.each(rejectionCases)("normalizes a $name rejection to the safe mutation error", async ({ value }) => {
    // Given a mutation command that rejects with an arbitrary boundary value
    api.confirmSpace.mockRejectedValueOnce(value);
    renderOverview();
    const suggestedRow = await screen.findByTestId("space-row-suggested");

    // When the mutation is submitted
    fireEvent.click(within(suggestedRow).getByRole("button", { name: labels.keep }));

    // Then one API call settles into localized recovery without leaking details
    expect(await screen.findByRole("alert")).toHaveTextContent(labels.mutationError);
    expect(api.confirmSpace).toHaveBeenCalledTimes(1);
    expect(screen.queryByText(/internal .* detail/i)).not.toBeInTheDocument();
    await waitFor(() => {
      expect(within(suggestedRow).getByRole("button", { name: labels.keep })).toBeEnabled();
    });
  });
});
