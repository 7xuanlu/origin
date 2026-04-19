// SPDX-License-Identifier: AGPL-3.0-only
import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import { WhatHappensNextCard } from "../WhatHappensNextCard";

describe("WhatHappensNextCard", () => {
  it("renders Seed copy when intelligence is loading", () => {
    render(<WhatHappensNextCard state="seed" memoryCount={0} daysInListening={0} />);
    expect(screen.getByText(/on-device intelligence/i)).toBeInTheDocument();
  });

  it("renders Listening copy when ready but no memories", () => {
    render(<WhatHappensNextCard state="listening" memoryCount={0} daysInListening={0} />);
    expect(screen.getByText(/keep using your AI tools/i)).toBeInTheDocument();
  });

  it("renders Gathering copy with count when memories exist but no concepts", () => {
    render(<WhatHappensNextCard state="gathering" memoryCount={3} daysInListening={0} />);
    expect(screen.getByText(/3 memor/i)).toBeInTheDocument();
  });

  it("renders Gathering 5+ copy variant when memories >= 5", () => {
    render(<WhatHappensNextCard state="gathering" memoryCount={12} daysInListening={0} />);
    expect(screen.getByText(/patterns emerge/i)).toBeInTheDocument();
  });

  it("renders stuck-in-listening variant after 3 days", () => {
    render(<WhatHappensNextCard state="listening" memoryCount={0} daysInListening={4} />);
    expect(screen.getByText(/still quiet/i)).toBeInTheDocument();
  });

  it("renders nothing in Alive state", () => {
    const { container } = render(
      <WhatHappensNextCard state="alive" memoryCount={20} daysInListening={30} />
    );
    expect(container.firstChild).toBeNull();
  });
});
