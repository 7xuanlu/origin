// SPDX-License-Identifier: AGPL-3.0-only
import { describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import { i18n } from "../i18n";
import ViewToggle from "./ViewToggle";

describe("ViewToggle", () => {
  it.each([
    ["zh-Hans", "首页", "活动"],
    ["zh-Hant", "首頁", "活動"],
  ])(
    "uses translated labels in %s",
    async (locale, homeLabel, activityLabel) => {
      await i18n.changeLanguage(locale);

      render(<ViewToggle active="home" onSwitch={vi.fn()} />);

      expect(
        screen.getByRole("button", { name: homeLabel }),
      ).toBeInTheDocument();
      expect(
        screen.getByRole("button", { name: activityLabel }),
      ).toBeInTheDocument();
    },
  );
});
