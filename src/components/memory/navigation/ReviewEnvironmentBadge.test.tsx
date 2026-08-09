import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import { ReviewEnvironmentBadge } from "./ReviewEnvironmentBadge";

describe("ReviewEnvironmentBadge", () => {
  it("stays absent from production surfaces", () => {
    render(<ReviewEnvironmentBadge enabled={false} />);

    expect(screen.queryByText("TEST DATA")).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Reset test data" })).not.toBeInTheDocument();
  });

  it("keeps the fixture boundary and reset semantics visible in Review", () => {
    render(<ReviewEnvironmentBadge enabled />);

    expect(screen.getByText("TEST DATA")).toBeVisible();
    expect(screen.getByText("Fixture data · resets on relaunch")).toBeVisible();
    expect(screen.getByRole("button", { name: "Reset test data" })).toBeVisible();
  });

  it("requires an explicit reset action", async () => {
    const onReset = vi.fn();
    const user = userEvent.setup();
    render(<ReviewEnvironmentBadge enabled onReset={onReset} />);

    await user.click(screen.getByRole("button", { name: "Reset test data" }));

    expect(onReset).toHaveBeenCalledTimes(1);
  });

  it("retains a compact proof mark when the sidebar is unavailable", () => {
    render(<ReviewEnvironmentBadge compact enabled />);

    expect(screen.getByText("TEST DATA")).toBeVisible();
    expect(screen.queryByText("Fixture data · resets on relaunch")).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Reset test data" })).not.toBeInTheDocument();
    expect(screen.getByRole("status")).toHaveAccessibleName("Fixture data · resets on relaunch");
  });
});
