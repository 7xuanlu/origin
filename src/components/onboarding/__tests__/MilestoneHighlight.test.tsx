// SPDX-License-Identifier: AGPL-3.0-only
import { describe, it, expect, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import { MilestoneHighlight } from "../MilestoneHighlight";

// jsdom does not ship IntersectionObserver. The component constructs one in its
// effect when `active`, so stub it with a no-op for tests. Callback is never
// invoked, so onSeen scheduling stays inert — matching the test expectations.
if (typeof IntersectionObserver === "undefined") {
  (globalThis as unknown as { IntersectionObserver: unknown }).IntersectionObserver =
    class {
      observe() {}
      unobserve() {}
      disconnect() {}
    };
}

describe("MilestoneHighlight", () => {
  it("wraps its children", () => {
    render(
      <MilestoneHighlight active={true} onSeen={vi.fn()}>
        <div>inner</div>
      </MilestoneHighlight>
    );
    expect(screen.getByText("inner")).toBeInTheDocument();
  });

  it("does not call onSeen when inactive", () => {
    const onSeen = vi.fn();
    render(
      <MilestoneHighlight active={false} onSeen={onSeen}>
        <div>inner</div>
      </MilestoneHighlight>
    );
    expect(onSeen).not.toHaveBeenCalled();
  });
});
