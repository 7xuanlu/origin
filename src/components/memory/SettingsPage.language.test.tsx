// SPDX-License-Identifier: AGPL-3.0-only
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { i18n } from "../../i18n";
import { APP_LOCALE_STORAGE_KEY } from "../../i18n/locales";
import SettingsPage from "./SettingsPage";

vi.mock("../../lib/theme", () => ({
  useTheme: () => ["system", vi.fn()] as const,
}));

vi.mock("../../lib/tauri", () => ({
  deleteAgent: vi.fn(() => Promise.resolve()),
  detectMcpClients: vi.fn(() => Promise.resolve([])),
  getCaptureStats: vi.fn(() => Promise.resolve({ clipboard: 0, screen: 0 })),
  getProfile: vi.fn(() => Promise.resolve({
    id: "p1",
    name: "Lucian",
    display_name: "Lucian",
    email: null,
    bio: null,
    avatar_path: null,
    created_at: 0,
  })),
  isRunAtLoginEnabled: vi.fn(() => Promise.resolve(false)),
  listAgents: vi.fn(() => Promise.resolve([])),
  setAvatar: vi.fn(() => Promise.resolve()),
  setRunAtLogin: vi.fn(() => Promise.resolve()),
  setSetupCompleted: vi.fn(() => Promise.resolve()),
  removeAvatar: vi.fn(() => Promise.resolve()),
  updateAgent: vi.fn(() => Promise.resolve()),
  updateProfile: vi.fn(() => Promise.resolve()),
}));

vi.mock("./sources/SourcesSection", () => ({
  default: () => <div />,
}));

vi.mock("../ChatImport/ImportFlow", () => ({
  ImportFlow: () => <div />,
}));

vi.mock("./RemoteAccessPanel", () => ({
  RemoteAccessPanel: () => <div />,
}));

vi.mock("../intelligence/IntelligenceSetup", () => ({
  AnthropicFields: () => <div />,
  OnDeviceModelCard: () => <div />,
  useApiKeyStatus: () => ({
    data: { hasAnthropic: false, hasOpenAI: false },
    isLoading: false,
  }),
}));

vi.mock("./settings/sections/DiagnosticsSection", () => ({
  default: () => <div />,
}));

function renderSettingsPage() {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: { retry: false },
      mutations: { retry: false },
    },
  });

  return render(
    <QueryClientProvider client={queryClient}>
      <SettingsPage section="general" onBack={() => {}} onImport={() => {}} />
    </QueryClientProvider>,
  );
}

describe("SettingsPage language selector", () => {
  beforeEach(async () => {
    window.localStorage.clear();
    await i18n.changeLanguage("en");
  });

  it("shows all supported interface language choices", () => {
    renderSettingsPage();

    expect(screen.getByLabelText("Language")).toHaveValue("system");
    expect(screen.getByRole("option", { name: "System" })).toBeInTheDocument();
    expect(screen.getByRole("option", { name: "English" })).toBeInTheDocument();
    expect(
      screen.getByRole("option", { name: "简体中文" }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("option", { name: "繁體中文" }),
    ).toBeInTheDocument();
  });

  it("persists language changes and updates the active i18n language", async () => {
    const user = userEvent.setup();
    renderSettingsPage();

    await user.selectOptions(screen.getByLabelText("Language"), "zh-Hant");

    expect(window.localStorage.getItem(APP_LOCALE_STORAGE_KEY)).toBe("zh-Hant");
    expect(i18n.language).toBe("zh-Hant");
  });
});
