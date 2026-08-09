// SPDX-License-Identifier: AGPL-3.0-only
import { describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import SourcesSection from "./SourcesSection";

vi.mock("../../sources/SourcesSection", () => ({ default: () => <div /> }));
vi.mock("../../../ChatImport/ImportFlow", () => ({ ImportFlow: () => <div /> }));

// R4.2/S5: `onImport` used to be optional, and the Import Memories button
// only rendered when a handler was passed in. It's required now — this
// proves the button is unconditional, not just that the type changed.
describe("SourcesSection import button", () => {
  it("renders the Import Memories button unconditionally", () => {
    render(<SourcesSection onImport={vi.fn()} />);
    expect(screen.getByRole("button", { name: "Import" })).toBeInTheDocument();
  });
});
