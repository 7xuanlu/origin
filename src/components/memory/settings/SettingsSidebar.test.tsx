// SPDX-License-Identifier: AGPL-3.0-only
import { describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import SettingsSidebar from "./SettingsSidebar";

vi.mock("@tauri-apps/api/app", () => ({
  getVersion: vi.fn(() => new Promise(() => {})),
}));

function renderSettingsSidebar(extraProps: Partial<React.ComponentProps<typeof SettingsSidebar>> = {}) {
  return render(
    <SettingsSidebar
      collapsed={false}
      active="general"
      onSelect={() => {}}
      onNavigateHome={() => {}}
      {...extraProps}
    />,
  );
}

describe("SettingsSidebar", () => {
  it("uses Home as the top return affordance instead of a Wenlan heading", async () => {
    const user = userEvent.setup();
    const onNavigateHome = vi.fn();
    renderSettingsSidebar({ onNavigateHome });

    expect(screen.queryByRole("heading", { name: "Wenlan" })).toBeNull();

    await user.click(screen.getByRole("button", { name: "Home" }));

    expect(onNavigateHome).toHaveBeenCalledTimes(1);
  });

  it("keeps the Wenlan brand in the footer", () => {
    renderSettingsSidebar();

    const settingsLabel = screen.getByText("Settings");
    const brand = screen.getByRole("button", { name: "Wenlan" });

    expect(settingsLabel.compareDocumentPosition(brand) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
  });
});
