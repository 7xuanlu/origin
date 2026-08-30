// SPDX-License-Identifier: AGPL-3.0-only
import { describe, it, expect } from "vitest";
import { stripLedeLabel } from "./pageLede";

describe("stripLedeLabel", () => {
  it("drops a plain TLDR label", () => {
    expect(stripLedeLabel("TLDR: Tally stores all data in one SQLite file.")).toBe(
      "Tally stores all data in one SQLite file.",
    );
  });

  it("drops a bold TL;DR label with a dash", () => {
    expect(stripLedeLabel("**TL;DR** — Stripe over PayPal for EU clients.")).toBe(
      "Stripe over PayPal for EU clients.",
    );
  });

  it("drops the label but keeps the sentence's citation links", () => {
    expect(stripLedeLabel("TLDR: One file [5](#citation:5).")).toBe(
      "One file [5](#citation:5).",
    );
  });

  it("leaves prose that merely starts with those letters", () => {
    const prose = "TLDRs are no substitute for the page.";
    expect(stripLedeLabel(prose)).toBe(prose);
  });

  it("leaves an ordinary sentence untouched", () => {
    const prose = "Tally is a single-user invoicing app.";
    expect(stripLedeLabel(prose)).toBe(prose);
  });
});
