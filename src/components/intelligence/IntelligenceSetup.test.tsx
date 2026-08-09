// SPDX-License-Identifier: AGPL-3.0-only
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { open as shellOpen } from "@tauri-apps/plugin-shell";
import "../../i18n";

const mocks = vi.hoisted(() => ({
  getApiKey: vi.fn(),
  setApiKey: vi.fn(),
  getModelChoice: vi.fn(),
  setModelChoice: vi.fn(),
  getOnDeviceModel: vi.fn(),
  getSystemInfo: vi.fn(),
}));
vi.mock("../../lib/tauri", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../../lib/tauri")>();
  return { ...actual, ...mocks };
});

import { AnthropicFields, OnDeviceModelCard } from "./IntelligenceSetup";

function renderCard(qc: QueryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } })) {
  render(
    <QueryClientProvider client={qc}>
      <AnthropicFields />
    </QueryClientProvider>
  );
  return qc;
}

describe("AnthropicFields", () => {
  // NOTE: reset runs in afterEach, not beforeEach — see
  // src/hooks/useDaemonVersion.test.tsx for why. Resetting a vi.fn() in
  // beforeEach immediately before it's reconfigured with a resolved value
  // and consumed by a react-query queryFn deterministically triggers a
  // false-positive "unhandled rejection" failure on this project's pinned
  // vitest 4.1.5 + @tanstack/react-query 5.100.9 + React 19 combination.
  afterEach(() => {
    Object.values(mocks).forEach((m) => m.mockReset());
  });

  beforeEach(() => {
    mocks.getApiKey.mockResolvedValue(null);
    mocks.setApiKey.mockResolvedValue(undefined);
    mocks.getModelChoice.mockResolvedValue([null, null]);
    mocks.setModelChoice.mockResolvedValue(undefined);
  });

  it("invalidates apiKey, external-llm, and external-llm-key-configured after saving a key (job rows re-read the routing)", async () => {
    const qc = renderCard();
    const invalidateSpy = vi.spyOn(qc, "invalidateQueries");

    await userEvent.type(screen.getByPlaceholderText("sk-ant-api03-..."), "sk-ant-test-key");
    await userEvent.click(screen.getByRole("button", { name: "Save" }));

    await waitFor(() => expect(mocks.setApiKey).toHaveBeenCalledWith("sk-ant-test-key"));
    expect(invalidateSpy).toHaveBeenCalledWith({ queryKey: ["apiKey"] });
    expect(invalidateSpy).toHaveBeenCalledWith({ queryKey: ["external-llm"] });
    expect(invalidateSpy).toHaveBeenCalledWith({ queryKey: ["external-llm-key-configured"] });
  });

  it("invalidates the same keys after clearing a configured key", async () => {
    mocks.getApiKey.mockResolvedValue("sk-ant-***configured");
    const qc = renderCard();
    const invalidateSpy = vi.spyOn(qc, "invalidateQueries");

    await screen.findByText("sk-ant-***configured");
    await userEvent.click(screen.getByRole("button", { name: "Clear" }));

    await waitFor(() => expect(mocks.setApiKey).toHaveBeenCalledWith(""));
    expect(invalidateSpy).toHaveBeenCalledWith({ queryKey: ["apiKey"] });
    expect(invalidateSpy).toHaveBeenCalledWith({ queryKey: ["external-llm"] });
    expect(invalidateSpy).toHaveBeenCalledWith({ queryKey: ["external-llm-key-configured"] });
  });

  it("offers a Get-a-key link to the Anthropic console when no key is set", async () => {
    // getApiKey → null (set in beforeEach) renders the password-input
    // branch (not the masked state).
    vi.mocked(shellOpen).mockClear();
    renderCard();
    await userEvent.click(await screen.findByRole("button", { name: /Get a key/ }));
    expect(shellOpen).toHaveBeenCalledWith(
      "https://console.anthropic.com/settings/keys",
    );
  });

  it("gives the routine and synthesis model selects a real accessible name (not just an adjacent, unassociated row label)", async () => {
    mocks.getApiKey.mockResolvedValue("sk-ant-***configured");
    mocks.getModelChoice.mockResolvedValue([null, null]);
    renderCard();
    await screen.findByText("sk-ant-***configured");
    expect(await screen.findByRole("combobox", { name: "Choose everyday model" })).toBeInTheDocument();
    expect(screen.getByRole("combobox", { name: "Choose synthesis model" })).toBeInTheDocument();
  });
});

describe("OnDeviceModelCard", () => {
  afterEach(() => {
    mocks.getOnDeviceModel.mockReset();
    mocks.getSystemInfo.mockReset();
  });

  beforeEach(() => {
    mocks.getOnDeviceModel.mockResolvedValue({
      loaded: null,
      selected: "qwen3-4b-instruct-2507",
      models: [{
        id: "qwen3-4b-instruct-2507",
        display_name: "Qwen3 4B",
        param_count: "4B",
        ram_required_gb: 8,
        file_size_gb: 2.7,
        cached: false,
      }],
    });
    mocks.getSystemInfo.mockResolvedValue({
      total_ram_gb: 16,
      available_ram_gb: 10,
      has_metal: true,
      has_cuda: false,
      os: "macOS",
      arch: "arm64",
      recommended_builtin: "qwen3-4b-instruct-2507",
    });
  });

  // Defect 6 / spec §3.6 "metadata never truncates": the model spec line
  // (params · download size · RAM) is a system-authored fact line — it must
  // wrap, never truncate mid-word. `truncate` on this span was the bug.
  it("does not truncate the on-device model spec line", async () => {
    const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    render(
      <QueryClientProvider client={qc}>
        <OnDeviceModelCard />
      </QueryClientProvider>,
    );

    const specLine = await screen.findByTestId("on-device-model-spec");
    expect(specLine).toHaveTextContent("4B params");
    expect(specLine.className.split(/\s+/)).not.toContain("truncate");
  });

  it("gives the model picker Select a real accessible name", async () => {
    const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    render(
      <QueryClientProvider client={qc}>
        <OnDeviceModelCard />
      </QueryClientProvider>,
    );
    expect(await screen.findByRole("combobox", { name: "Choose on-device model" })).toBeInTheDocument();
  });

  it("shows an honest unavailable message instead of an empty, dead-end Select when the catalog is empty", async () => {
    mocks.getOnDeviceModel.mockResolvedValue({ loaded: null, selected: null, models: [] });
    const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    render(
      <QueryClientProvider client={qc}>
        <OnDeviceModelCard />
      </QueryClientProvider>,
    );
    expect(await screen.findByText("Model catalog unavailable — check your connection.")).toBeInTheDocument();
    expect(screen.queryByRole("combobox")).not.toBeInTheDocument();
  });
});
