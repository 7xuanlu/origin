// SPDX-License-Identifier: AGPL-3.0-only
import { describe, expect, it, vi, beforeEach } from "vitest";
import { fireEvent, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { i18n } from "../../../i18n";
import IdentityCard from "../IdentityCard";

vi.mock("../../../lib/tauri", () => ({
  getProfile: vi.fn(),
  listEntities: vi.fn(),
  getEntityDetail: vi.fn(),
}));

import * as tauri from "../../../lib/tauri";

function renderIdentityCard() {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  const onOpenSettings = vi.fn();
  const onOpenAbout = vi.fn();
  return render(
    <QueryClientProvider client={qc}>
      <IdentityCard
        onOpenDetail={() => {}}
        onOpenSettings={onOpenSettings}
        onOpenAbout={onOpenAbout}
      />
    </QueryClientProvider>,
  );
}

beforeEach(async () => {
  await i18n.changeLanguage("en");
  vi.mocked(tauri.getProfile).mockResolvedValue({
    id: "p1",
    name: "Lucian",
    display_name: "Lucian",
    email: null,
    bio: null,
    avatar_path: "/missing/avatar.png",
    created_at: 0,
    updated_at: 0,
  } as any);
  vi.mocked(tauri.listEntities).mockResolvedValue([
    { id: "person-lucian", name: "Lucian", entity_type: "person" },
  ] as any);
  vi.mocked(tauri.getEntityDetail).mockResolvedValue({ observations: [] } as any);
});

describe("IdentityCard", () => {
  it("falls back to initials when the saved avatar cannot be loaded", async () => {
    renderIdentityCard();

    const avatar = await screen.findByRole("img", { name: "Lucian" });
    fireEvent.error(avatar);

    expect(screen.queryByRole("img", { name: "Lucian" })).not.toBeInTheDocument();
    expect(screen.getByText("L")).toBeInTheDocument();
  });

  it("renders a horizontal mini account card without observation text", async () => {
    vi.mocked(tauri.getEntityDetail).mockResolvedValue({
      observations: [{ content: "The user is a senior engineer working on Wenlan." }],
    } as any);

    renderIdentityCard();

    const trigger = await screen.findByRole("button", { name: /Lucian account menu/ });
    await new Promise((resolve) => setTimeout(resolve, 0));

    expect(tauri.getEntityDetail).not.toHaveBeenCalled();
    expect(trigger).toHaveClass("flex", "items-center", "gap-3");
    expect(trigger).toHaveStyle({ backgroundColor: "var(--mem-account-card)" });
    expect(trigger.getAttribute("style")).toContain("--mem-account-card-border");
    expect(screen.getByText("Lucian")).toBeInTheDocument();
    expect(screen.queryByText(/senior engineer/i)).not.toBeInTheDocument();
    expect(screen.queryByText("Set up your profile")).not.toBeInTheDocument();
  });

  it("opens a minimal avatar menu without profile-specific or tool-connection shortcuts", async () => {
    const user = userEvent.setup();
    const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    const onOpenSettings = vi.fn();
    const onOpenAbout = vi.fn();

    render(
      <QueryClientProvider client={qc}>
        <IdentityCard
          onOpenDetail={() => {}}
          onOpenSettings={onOpenSettings}
          onOpenAbout={onOpenAbout}
        />
      </QueryClientProvider>,
    );

    await user.click(await screen.findByRole("button", { name: /Lucian account menu/ }));

    expect(screen.getByRole("menu")).toHaveStyle({ backgroundColor: "var(--mem-popover)" });
    expect(screen.getByRole("menuitem", { name: "Settings" })).toBeInTheDocument();
    expect(screen.getByRole("menuitem", { name: "About Wenlan" })).toBeInTheDocument();
    expect(screen.queryByText("Profile settings")).not.toBeInTheDocument();
    expect(screen.queryByText("Connect tools")).not.toBeInTheDocument();
    expect(screen.queryByText("Local account")).not.toBeInTheDocument();

    await user.click(screen.getByRole("menuitem", { name: "Settings" }));
    expect(onOpenSettings).toHaveBeenCalledTimes(1);
    expect(onOpenAbout).not.toHaveBeenCalled();
  });

  it("localizes the empty account menu state", async () => {
    await i18n.changeLanguage("zh-Hant");
    vi.mocked(tauri.getProfile).mockResolvedValue(null);
    vi.mocked(tauri.listEntities).mockResolvedValue([]);

    renderIdentityCard();

    expect(await screen.findByRole("button", { name: "帳戶選單" })).toBeInTheDocument();
    expect(screen.getByText("帳戶")).toBeInTheDocument();
    expect(screen.queryByText("設定你的個人資料")).not.toBeInTheDocument();
    expect(screen.queryByText("Set up your profile")).not.toBeInTheDocument();
  });
});
