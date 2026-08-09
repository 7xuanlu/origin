import { fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { ConfirmedSpaces } from "./ConfirmedSpaces";
import { labels, makeSpace } from "./SpacesOverview.testUtils";

const work = makeSpace({ id: "work", name: "Work", sort_order: 0 });
const personal = makeSpace({ id: "personal", name: "Personal", sort_order: 1 });
const spaces = [work, personal] as const;

function renderConfirmed(onReorder = vi.fn()) {
  const result = render(
    <ConfirmedSpaces
      spaces={spaces}
      allSpaces={spaces}
      labels={labels}
      filter=""
      onFilterChange={() => undefined}
      noResults={false}
      pageCounts={new Map()}
      pendingIds={[]}
      onSelect={() => undefined}
      onStar={() => undefined}
      onRename={async () => true}
      onReorder={onReorder}
      onDelete={() => undefined}
    />,
  );
  return { ...result, onReorder };
}

function startWorkDrag(pointerId: number): HTMLElement {
  const handle = screen.getByRole("button", { name: labels.dragSpace("Work") });
  fireEvent.pointerDown(handle, { pointerId });
  return handle;
}

afterEach(() => {
  vi.restoreAllMocks();
});

describe("ConfirmedSpaces pointer drag lifecycle", () => {
  it("does not reorder after the active pointer is cancelled", () => {
    // Given an active drag that the browser cancels
    const { onReorder } = renderConfirmed();
    const handle = startWorkDrag(1);

    // When the cancelled pointer later releases over another row
    fireEvent.pointerCancel(handle, { pointerId: 1 });
    fireEvent.pointerUp(screen.getByTestId("space-row-personal"), { pointerId: 1 });

    // Then the stale source cannot trigger a reorder
    expect(onReorder).not.toHaveBeenCalled();
  });

  it("does not reorder after the drag handle loses pointer capture", () => {
    // Given an active drag whose handle loses capture
    const { onReorder } = renderConfirmed();
    const handle = startWorkDrag(2);

    // When that pointer later releases over another row
    fireEvent.lostPointerCapture(handle, { pointerId: 2 });
    fireEvent.pointerUp(screen.getByTestId("space-row-personal"), { pointerId: 2 });

    // Then the abandoned drag is no longer actionable
    expect(onReorder).not.toHaveBeenCalled();
  });

  it("clears the drag when pointerup occurs outside the inventory", () => {
    // Given an active drag
    const { onReorder } = renderConfirmed();
    startWorkDrag(3);

    // When the pointer releases outside and a later release hits a row
    fireEvent.pointerUp(window, { pointerId: 3 });
    fireEvent.pointerUp(screen.getByTestId("space-row-personal"), { pointerId: 3 });

    // Then the outside release prevents a stale reorder
    expect(onReorder).not.toHaveBeenCalled();
  });

  it("clears the drag when pointerup bubbles from the document", () => {
    // Given an active drag
    const { onReorder } = renderConfirmed();
    startWorkDrag(7);

    // When the pointer releases on the document before a later row release
    fireEvent.pointerUp(document, { pointerId: 7 });
    fireEvent.pointerUp(screen.getByTestId("space-row-personal"), { pointerId: 7 });

    // Then the document-level release prevents a stale reorder
    expect(onReorder).not.toHaveBeenCalled();
  });

  it("allows only the pointer that started the drag to finish it", () => {
    // Given pointer 4 owns the active drag
    const { onReorder } = renderConfirmed();
    startWorkDrag(4);
    const target = screen.getByTestId("space-row-personal");

    // When another pointer releases over the target
    fireEvent.pointerUp(target, { pointerId: 5 });

    // Then it cannot finish pointer 4's drag
    expect(onReorder).not.toHaveBeenCalled();

    // When the owning pointer releases over the target
    fireEvent.pointerUp(target, { pointerId: 4 });

    // Then the intended reorder runs exactly once
    expect(onReorder).toHaveBeenCalledTimes(1);
    expect(onReorder).toHaveBeenCalledWith(work, personal);
  });

  it("routes a normal drag through the existing reorder callback exactly once", () => {
    // Given an active drag over a compatible target
    const { onReorder } = renderConfirmed();
    startWorkDrag(6);
    const target = screen.getByTestId("space-row-personal");
    fireEvent.pointerEnter(target, { pointerId: 6 });

    // When the active pointer releases over the target twice
    fireEvent.pointerUp(target, { pointerId: 6 });
    fireEvent.pointerUp(target, { pointerId: 6 });

    // Then only the first release reorders
    expect(onReorder).toHaveBeenCalledTimes(1);
    expect(onReorder).toHaveBeenCalledWith(work, personal);
  });

  it("removes global pointer lifecycle listeners on unmount", () => {
    // Given a mounted inventory with global pointer lifecycle listeners
    const addListener = vi.spyOn(window, "addEventListener");
    const removeListener = vi.spyOn(window, "removeEventListener");
    const { unmount } = renderConfirmed();
    const pointerUpHandler = addListener.mock.calls.find(([type]) => type === "pointerup")?.[1];
    const pointerCancelHandler = addListener.mock.calls.find(([type]) => type === "pointercancel")?.[1];

    // When the inventory unmounts
    unmount();

    // Then both global listeners are removed with their registered identities
    expect(pointerUpHandler).toBeDefined();
    expect(pointerCancelHandler).toBeDefined();
    expect(removeListener).toHaveBeenCalledWith("pointerup", pointerUpHandler);
    expect(removeListener).toHaveBeenCalledWith("pointercancel", pointerCancelHandler);
  });
});
