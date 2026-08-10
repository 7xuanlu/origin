// SPDX-License-Identifier: AGPL-3.0-only
import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import { PageTruthBadges } from "./PageTruthBadges";
import type { PageTruth } from "../../lib/tauri";

const supportedAndReviewed: PageTruth = { supported: true, human_reviewed: true };
const provisionalAndUnreviewed: PageTruth = { supported: false, human_reviewed: false };

describe("PageTruthBadges", () => {
  it("renders both axes once the cutover is live and the page carries truth data", () => {
    render(<PageTruthBadges cutoverLive truth={supportedAndReviewed} />);

    expect(screen.getByTestId("page-truth-support")).toHaveTextContent("Supported");
    expect(screen.getByTestId("page-truth-review")).toHaveTextContent("Reviewed");
  });

  it("renders nothing before the cutover goes live, even with truth data present", () => {
    // The safety property: a page must never show a vault-wide "Provisional"
    // badge ahead of the daemon's own cutover ceremony, however complete the
    // truth data on the wire looks.
    const { container } = render(<PageTruthBadges cutoverLive={false} truth={supportedAndReviewed} />);

    expect(container).toBeEmptyDOMElement();
    expect(screen.queryByTestId("page-truth-support")).not.toBeInTheDocument();
    expect(screen.queryByTestId("page-truth-review")).not.toBeInTheDocument();
  });

  it("renders nothing when the page carries no truth data", () => {
    const { container } = render(<PageTruthBadges cutoverLive truth={null} />);

    expect(container).toBeEmptyDOMElement();
    expect(screen.queryByTestId("page-truth-support")).not.toBeInTheDocument();
  });

  it("renders nothing when truth is undefined", () => {
    const { container } = render(<PageTruthBadges cutoverLive />);

    expect(container).toBeEmptyDOMElement();
  });

  it("renders the negative label variants for an unsupported, unreviewed page", () => {
    render(<PageTruthBadges cutoverLive truth={provisionalAndUnreviewed} />);

    const support = screen.getByTestId("page-truth-support");
    const review = screen.getByTestId("page-truth-review");
    expect(support).toHaveTextContent("Provisional");
    expect(support.className).toContain("wiki-page-state--truth-provisional");
    expect(review).toHaveTextContent("Unreviewed");
    expect(review.className).toContain("wiki-page-state--truth-unreviewed");
  });
});
