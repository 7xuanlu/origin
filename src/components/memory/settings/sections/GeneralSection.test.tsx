// SPDX-License-Identifier: AGPL-3.0-only
import { describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import GeneralSection from "./GeneralSection";

vi.mock("../../../../lib/tauri", () => ({
  getProfile: vi.fn(() =>
    Promise.resolve({
      id: "p1",
      name: "Lucian",
      display_name: "Lucian",
      email: null,
      bio: null,
      avatar_path: null,
      created_at: 0,
    }),
  ),
  updateProfile: vi.fn(() => Promise.resolve()),
  setAvatar: vi.fn(() => Promise.resolve()),
  removeAvatar: vi.fn(() => Promise.resolve()),
  setSetupCompleted: vi.fn(() => Promise.resolve()),
  isRunAtLoginEnabled: vi.fn(() => Promise.resolve(false)),
  setRunAtLogin: vi.fn(() => Promise.resolve()),
}));

vi.mock("../../../../lib/theme", () => ({
  useTheme: () => ["system", vi.fn()] as const,
}));

function renderGeneralSection() {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
  });
  return render(
    <QueryClientProvider client={queryClient}>
      <GeneralSection />
    </QueryClientProvider>,
  );
}

// S6: run-at-login, theme, and language used to be three separate
// `Card padding="rows"` blocks. They now share one. `.divide-y` is the class
// `Card` only applies for padding="rows", so counting it proves the merge
// instead of just trusting the JSX shape. ProfileSettingsBlock also renders
// a `rows` Card (Photo + Display name), so scope to the card that contains
// "Language" — unique to the App section — rather than every `.divide-y`.
describe("GeneralSection app card merge", () => {
  it("renders run-at-login, theme, and language inside a single rows Card", async () => {
    renderGeneralSection();

    await screen.findByLabelText("Language");

    const rowsCards = Array.from(document.querySelectorAll(".divide-y")).filter((el) =>
      el.textContent?.includes("Language"),
    );
    expect(rowsCards).toHaveLength(1);

    const merged = rowsCards[0];
    expect(merged.textContent).toContain("Run Wenlan in background at login");
    expect(merged.textContent).toContain("Theme");
    expect(merged.textContent).toContain("Language");
  });
});
