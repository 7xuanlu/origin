// SPDX-License-Identifier: AGPL-3.0-only
//
// Pins the one true privacy claim in the Settings screen. The app used to
// state a categorical "no data ever leaves your machine" in two places
// (the settings sidebar's bottom strip and the per-section footer) while the
// same screen offers an Anthropic API key and an unauthenticated Remote
// Access tunnel — both of which do send data off the device. The fix keeps
// the claim in exactly one place (the per-section footer) and makes it
// conditional. This test renders the sidebar and the page as the siblings
// they are in Main.tsx's settings view, so "exactly once" is checked against
// the real composed screen, not each component in isolation.
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import SettingsSidebar from "./settings/SettingsSidebar";
import SettingsPage from "./SettingsPage";

vi.mock("@tauri-apps/api/app", () => ({
  getVersion: vi.fn(() => new Promise(() => {})),
}));

vi.mock("../../lib/theme", () => ({
  useTheme: () => ["system", vi.fn()] as const,
}));

vi.mock("../../lib/tauri", () => ({
  deleteAgent: vi.fn(() => Promise.resolve()),
  detectMcpClients: vi.fn(() => Promise.resolve([])),
  getCaptureStats: vi.fn(() => Promise.resolve({ clipboard: 0, screen: 0 })),
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

function renderSettingsScreen() {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: { retry: false },
      mutations: { retry: false },
    },
  });

  // Mirrors Main.tsx: SettingsSidebar and SettingsPage are mounted as
  // siblings under the same "settings" view — not two isolated trees.
  return render(
    <QueryClientProvider client={queryClient}>
      <div>
        <SettingsSidebar
          collapsed={false}
          active="general"
          onSelect={() => {}}
          onNavigateHome={() => {}}
        />
        <SettingsPage section="general" onBack={() => {}} onImport={() => {}} />
      </div>
    </QueryClientProvider>,
  );
}

describe("Settings privacy claim", () => {
  it("states the truthful, conditional claim exactly once", () => {
    renderSettingsScreen();

    // getByText throws if more than one match exists, so this alone proves
    // singularity; the exact string also pins the wording so a partial
    // rewrite (e.g. dropping the condition) fails the test.
    const claim = screen.getByText(
      "Your memories live on this machine. Nothing is sent anywhere unless you connect a cloud model or turn on Remote Access.",
    );
    expect(claim).toBeInTheDocument();
  });

  it("no longer states the old categorical claim anywhere on the screen", () => {
    renderSettingsScreen();

    expect(
      screen.queryByText("Local-only. Your data never leaves this machine."),
    ).not.toBeInTheDocument();
    expect(
      screen.queryByText(
        "Everything stays on your device. No data leaves your machine — no cloud sync, no API calls, no telemetry.",
      ),
    ).not.toBeInTheDocument();
  });
});
