// SPDX-License-Identifier: AGPL-3.0-only
import { describe, it, expect, vi } from "vitest";
import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import RelatedPages from "./RelatedPages";

describe("RelatedPages", () => {
  it("renders nothing when there are no outbound links", () => {
    const { container } = render(<RelatedPages outbound={[]} onPageClick={vi.fn()} />);
    expect(container).toBeEmptyDOMElement();
  });

  it("renders resolved links as clickable cards", async () => {
    const user = userEvent.setup();
    const onPageClick = vi.fn();
    render(
      <RelatedPages
        outbound={[{ label: "Resolved Link", target_page_id: "page-2" }]}
        onPageClick={onPageClick}
      />,
    );
    const section = screen.getByLabelText("Related pages");
    await user.click(within(section).getByRole("button", { name: /Resolved Link/ }));
    expect(onPageClick).toHaveBeenCalledWith("page-2");
  });

  it("renders unresolved links muted and inert", () => {
    render(
      <RelatedPages
        outbound={[{ label: "Missing Link", target_page_id: null }]}
        onPageClick={vi.fn()}
      />,
    );
    const section = screen.getByLabelText("Related pages");
    expect(within(section).getByText("Missing Link")).toBeInTheDocument();
    expect(within(section).queryByRole("button", { name: /Missing Link/ })).toBeNull();
    expect(within(section).getByText(/not distilled yet/i)).toBeInTheDocument();
  });
});
