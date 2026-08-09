import { render, screen, within } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { i18n } from "../../../i18n";
import { SpaceRow } from "./SpaceRow";
import { labels, makeSpace } from "./SpacesOverview.testUtils";
import { formatLocaleDate } from "../../../lib/dateFormat";

function renderRow(updatedAt: number): HTMLElement {
  const space = makeSpace({ updated_at: updatedAt });
  render(
    <SpaceRow
      space={space}
      spaces={[space]}
      labels={labels}
      pageCount={2}
      pending={false}
      canMoveUp={false}
      canMoveDown={false}
      onSelect={vi.fn()}
      onStar={vi.fn()}
      onRename={vi.fn()}
      onMoveUp={vi.fn()}
      onMoveDown={vi.fn()}
      onDelete={vi.fn()}
      onDragStart={vi.fn()}
    />,
  );
  return screen.getByTestId(`space-row-${space.id}`);
}

const invalidTimestamps = [
  ["zero", 0],
  ["negative", -1],
  ["NaN", Number.NaN],
  ["positive infinity", Number.POSITIVE_INFINITY],
  ["negative infinity", Number.NEGATIVE_INFINITY],
  ["out of Date range", 8_640_000_000_001],
] as const;

describe("SpaceRow updated timestamp", () => {
  beforeEach(async () => {
    await i18n.changeLanguage("en");
  });

  it.each(invalidTimestamps)("renders a safe fallback for %s seconds", (_name, updatedAt) => {
    // Given a Space timestamp that cannot represent a meaningful updated date
    // When the inventory row renders
    const row = renderRow(updatedAt);

    // Then desktop and mobile metadata use the neutral fallback without invalid machine values
    const desktop = within(row).getByTestId("space-updated");
    const mobile = within(row).getByTestId("space-mobile-updated");
    expect(desktop).toHaveTextContent("—");
    expect(desktop).not.toHaveAttribute("datetime");
    expect(mobile).toHaveTextContent("—");
    expect(mobile.querySelector("[datetime]")).toBeNull();
    expect(row).not.toHaveTextContent(/NaN|Invalid Date|Infinity/);
  });

  it("retains locale text and ISO machine datetime for valid Unix seconds", () => {
    // Given a valid Unix-seconds timestamp
    const updatedAt = 1_720_569_600;

    // When the inventory row renders
    const row = renderRow(updatedAt);

    // Then both responsive variants share its locale label and ISO machine value
    const expectedDate = new Date(updatedAt * 1000);
    const expected = formatLocaleDate(expectedDate);
    const desktop = within(row).getByTestId("space-updated");
    const mobileTime = within(row).getByTestId("space-mobile-updated").querySelector("time");
    expect(desktop).toHaveTextContent(expected.label);
    expect(desktop).toHaveAttribute("datetime", expected.dateTime);
    expect(mobileTime).toHaveTextContent(expected.label);
    expect(mobileTime).toHaveAttribute("datetime", expected.dateTime);
  });

  it("formats the visible date with Wenlan's active locale", async () => {
    const updatedAt = 1_720_569_600;
    await i18n.changeLanguage("zh-Hant");

    const row = renderRow(updatedAt);

    expect(within(row).getByTestId("space-updated")).toHaveTextContent(
      new Date(updatedAt * 1000).toLocaleDateString("zh-Hant", {
        day: "numeric",
        month: "short",
        year: "numeric",
      }),
    );
  });
});
