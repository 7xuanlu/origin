import { describe, expect, it } from "vitest";
import { formatLocaleDate } from "./dateFormat";

describe("formatLocaleDate", () => {
  it("returns one unambiguous locale-aware calendar style and an ISO machine value", () => {
    const date = new Date("2026-07-10T12:00:00Z");
    const options: Intl.DateTimeFormatOptions = { day: "numeric", month: "short", year: "numeric" };

    expect(formatLocaleDate(date, "en-US")).toEqual({
      label: date.toLocaleDateString("en-US", options),
      dateTime: date.toISOString(),
    });
    expect(formatLocaleDate(date, "zh-TW")).toEqual({
      label: date.toLocaleDateString("zh-TW", options),
      dateTime: date.toISOString(),
    });
  });

  it("uses one neutral fallback for invalid dates", () => {
    expect(formatLocaleDate(new Date(Number.NaN))).toEqual({ label: "—" });
  });
});
