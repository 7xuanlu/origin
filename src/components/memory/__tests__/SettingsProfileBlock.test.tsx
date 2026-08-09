// SPDX-License-Identifier: AGPL-3.0-only
import { describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import SettingsPage from "../SettingsPage";

const updateProfileMock = vi.hoisted(() => vi.fn().mockResolvedValue(null));
const setRunAtLoginMock = vi.hoisted(() => vi.fn().mockResolvedValue(null));

vi.mock("../../../lib/tauri", () => ({
  getProfile: vi.fn().mockResolvedValue({
    id: "1",
    name: "Lucian",
    display_name: "Lucian",
    email: null,
    bio: null,
    avatar_path: null,
    created_at: 1709251200,
  }),
  updateProfile: updateProfileMock,
  setAvatar: vi.fn().mockResolvedValue(null),
  removeAvatar: vi.fn().mockResolvedValue(null),
  getCaptureStats: vi.fn().mockResolvedValue({ clipboard: 0, screen: 0 }),
  listAgents: vi.fn().mockResolvedValue([]),
  updateAgent: vi.fn().mockResolvedValue(null),
  deleteAgent: vi.fn().mockResolvedValue(null),
  detectMcpClients: vi.fn().mockResolvedValue([]),
  setSetupCompleted: vi.fn().mockResolvedValue(null),
  isRunAtLoginEnabled: vi.fn().mockResolvedValue(false),
  setRunAtLogin: setRunAtLoginMock,
}));

vi.mock("../../../lib/theme", () => ({
  useTheme: () => ["light", vi.fn()],
}));

vi.mock("../sources/SourcesSection", () => ({ default: () => <div /> }));
vi.mock("../../ChatImport/ImportFlow", () => ({ ImportFlow: () => <div /> }));
vi.mock("../RemoteAccessPanel", () => ({ RemoteAccessPanel: () => <div /> }));
vi.mock("../../intelligence/IntelligenceSetup", () => ({
  AnthropicFields: () => <div />,
  OnDeviceModelCard: () => <div />,
  useApiKeyStatus: () => ({ isConfigured: false }),
}));
vi.mock("../settings/sections/DiagnosticsSection", () => ({ default: () => <div /> }));

function renderSettingsPage() {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={qc}>
      <SettingsPage section="general" onBack={() => {}} onImport={() => {}} />
    </QueryClientProvider>,
  );
}

describe("SettingsPage profile display block", () => {
  it("keeps avatar and name editing inside General settings", async () => {
    renderSettingsPage();

    expect(await screen.findByText("Profile")).toBeInTheDocument();
    expect(screen.getByDisplayValue("Lucian")).toBeInTheDocument();
    expect(screen.queryByText("Email")).not.toBeInTheDocument();
    expect(screen.queryByPlaceholderText("Add email")).not.toBeInTheDocument();
    expect(screen.getByText("Joined March 2024")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Change photo" })).toBeInTheDocument();
    expect(screen.getByText("App")).toBeInTheDocument();
    expect(screen.getAllByText("General")).toHaveLength(1);

    expect(screen.queryByText(/How Wenlan sees you/i)).not.toBeInTheDocument();
    expect(screen.queryByText(/What Wenlan knows/i)).not.toBeInTheDocument();
    expect(screen.queryByText(/Local account/i)).not.toBeInTheDocument();
  });

  it("saves display name edits through the existing profile API", async () => {
    const user = userEvent.setup();
    renderSettingsPage();

    const name = await screen.findByDisplayValue("Lucian");
    await user.clear(name);
    await user.type(name, "Qi-Xuan Lu");
    await user.tab();

    expect(updateProfileMock).toHaveBeenCalledWith(
      "1",
      "Qi-Xuan Lu",
      "Qi-Xuan Lu",
      undefined,
      undefined,
    );
  });

  // Primitives migration (Step 6a): the display-name field moved from a
  // hand-rolled <label><input> pair to the shared Field/Input primitives,
  // and the run-at-login row moved into a Card wrapper. Both changes could
  // silently break the label/control association or the toggle's click
  // wiring without any visual sign, so pin them explicitly.
  it("keeps the display-name label wired to its input after the Field migration", async () => {
    renderSettingsPage();

    const input = await screen.findByLabelText("Display name");
    expect(input).toHaveValue("Lucian");
  });

  it("still fires the run-at-login toggle after the Card migration", async () => {
    const user = userEvent.setup();
    renderSettingsPage();

    const toggle = await screen.findByRole("button", {
      name: /run wenlan in background at login/i,
      pressed: false,
    });
    await user.click(toggle);

    expect(setRunAtLoginMock).toHaveBeenCalledWith(true, expect.anything());
  });
});
