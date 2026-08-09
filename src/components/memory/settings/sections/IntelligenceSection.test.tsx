// SPDX-License-Identifier: AGPL-3.0-only
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import "../../../../i18n";

const mocks = vi.hoisted(() => ({
  getApiKey: vi.fn(),
  setApiKey: vi.fn(),
  getModelChoice: vi.fn(),
  setModelChoice: vi.fn(),
  getExternalLlm: vi.fn(),
  setExternalLlm: vi.fn(),
  testExternalLlm: vi.fn(),
  listExternalModels: vi.fn(),
  getExternalLlmKeyConfigured: vi.fn(),
  getDaemonVersion: vi.fn(),
  getOnDeviceModel: vi.fn(),
  getSystemInfo: vi.fn(),
  downloadOnDeviceModel: vi.fn(),
  getResolvedRouting: vi.fn(),
  setSourcePin: vi.fn(),
}));
vi.mock("../../../../lib/tauri", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../../../../lib/tauri")>();
  return { ...actual, ...mocks };
});

import IntelligenceSection from "./IntelligenceSection";

function renderSection(qc: QueryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } })) {
  render(
    <QueryClientProvider client={qc}>
      <IntelligenceSection delay={0} />
    </QueryClientProvider>
  );
  return qc;
}

// A resolved-routing payload the app treats as PINNED mode. Callers override the
// per-job routes; the pool below has all three sources configured so source
// options render enabled.
function pinnedRouting(over: {
  everyday?: { source: string; model: string | null; mode: string; pin?: string | null };
  synthesis?: { source: string; model: string | null; mode: string; pin?: string | null };
} = {}) {
  return {
    everyday: over.everyday ?? { source: "anthropic", model: "claude-opus-4-6", mode: "pinned" },
    synthesis: over.synthesis ?? { source: "anthropic", model: "claude-sonnet-4-6", mode: "pinned" },
    pool: {
      anthropic: { configured: true, everyday_model: "claude-opus-4-6", synthesis_model: "claude-sonnet-4-6" },
      external: { endpoint: "https://api.openai.com/v1", model: "gpt-5.2" },
      on_device: { selected: "qwen3-4b-instruct-2507", loaded: true },
    },
  };
}

describe("IntelligenceSection", () => {
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
    mocks.getExternalLlm.mockResolvedValue([null, null]);
    mocks.setExternalLlm.mockResolvedValue(undefined);
    mocks.testExternalLlm.mockResolvedValue({ response: "pong" });
    mocks.listExternalModels.mockResolvedValue([]);
    mocks.getExternalLlmKeyConfigured.mockResolvedValue(false);
    mocks.getDaemonVersion.mockResolvedValue("0.12.0");
    mocks.getOnDeviceModel.mockResolvedValue({ loaded: null, selected: null, models: [] });
    mocks.getSystemInfo.mockResolvedValue({
      total_ram_gb: 16,
      available_ram_gb: 10,
      has_metal: true,
      has_cuda: false,
      os: "macOS",
      arch: "arm64",
      recommended_builtin: "qwen3-4b-instruct-2507",
    });
    mocks.downloadOnDeviceModel.mockResolvedValue(undefined);
    // Default to LEGACY mode — the live 0.13.2 daemon has no routing endpoint.
    mocks.getResolvedRouting.mockResolvedValue(null);
    mocks.setSourcePin.mockResolvedValue(undefined);
  });

  it("shows the connection-only cloud row meta (provider + masked key), not per-job models", async () => {
    mocks.getApiKey.mockResolvedValue("sk-ant-***configured");
    mocks.getModelChoice.mockResolvedValue(["claude-opus-4-6", "claude-haiku-4-5-20251001"]);
    renderSection();

    const cloudRow = (await screen.findByText("Cloud provider")).closest("button")!;
    // Connection identity only — the per-job model names moved to the job rows.
    expect(await within(cloudRow).findByText("Anthropic · sk-ant-***configured")).toBeInTheDocument();
    expect(within(cloudRow).queryByText(/Everyday model:/)).not.toBeInTheDocument();
    expect(within(cloudRow).queryByText(/Synthesis model:/)).not.toBeInTheDocument();
  });

  it("attributes an Ollama endpoint to the Local server row, not the Cloud provider row", async () => {
    mocks.getExternalLlm.mockResolvedValue(["http://localhost:11434/v1", "llama3.2:3b"]);
    renderSection();

    // The Synthesis job row also surfaces "Ollama (local) · llama3.2:3b"
    // (mix-and-match), so scope to the Local server row itself.
    const localRow = screen.getByText("Local server").closest("button")!;
    expect(await within(localRow).findByText("Ollama (local) · llama3.2:3b")).toBeInTheDocument();
    // Cloud row stays on its unconfigured hint — the same saved slot never
    // shows up in both provider rows.
    const cloudRow = screen.getByText("Cloud provider").closest("button")!;
    expect(within(cloudRow).getByText(/Anthropic, OpenAI, Gemini, Groq, and more/)).toBeInTheDocument();
  });

  // Mutation-proof target: reintroducing the old `!isConfigured &&` gate
  // around the on-device row must make this assertion fail.
  it("keeps the on-device row present even while Anthropic is configured", async () => {
    mocks.getApiKey.mockResolvedValue("sk-ant-***configured");
    renderSection();

    // Wait for the cloud row to render its configured meta, proving
    // isConfigured has resolved true, before checking the on-device row.
    const cloudRow = (await screen.findByText("Cloud provider")).closest("button")!;
    await within(cloudRow).findByText("Anthropic ·", { exact: false });
    expect(screen.getByText("On-device model")).toBeInTheDocument();
  });

  it("hides the provider body until the cloud row is expanded, then reveals its preset chips", async () => {
    renderSection();

    // Wait for the cloud row's meta to settle before asserting absence —
    // asserting before queries resolve would pass regardless of the bug.
    const cloudRow = (await screen.findByText("Cloud provider")).closest("button")!;
    await within(cloudRow).findByText(/Anthropic, OpenAI, Gemini, Groq, and more/);
    expect(screen.queryByRole("group", { name: "Provider" })).not.toBeInTheDocument();

    await userEvent.click(cloudRow);

    expect(await screen.findByRole("group", { name: "Provider" })).toBeInTheDocument();
  });

  // Pins the mix-and-match display logic: the two job rows walk independent
  // priority chains, so their rows can show different sources at once.
  it("shows the Anthropic routine model in the Everyday row's meta when Anthropic is configured", async () => {
    mocks.getApiKey.mockResolvedValue("sk-ant-***configured");
    mocks.getModelChoice.mockResolvedValue(["claude-haiku-4-5-20251001", "claude-sonnet-4-6"]);
    renderSection();

    const everydayRow = (await screen.findByText("Everyday model")).closest("button")!;
    expect(await within(everydayRow).findByText("Anthropic · Haiku 4.5")).toBeInTheDocument();
  });

  it("falls back the Synthesis row to the cloud-required hint, while the Everyday row still shows the on-device model", async () => {
    mocks.getOnDeviceModel.mockResolvedValue({
      loaded: "qwen3-4b-instruct-2507",
      selected: "qwen3-4b-instruct-2507",
      models: [{ id: "qwen3-4b-instruct-2507", display_name: "Qwen3 4B Instruct", cached: true }],
    });
    renderSection();

    // Wait for the on-device query to settle via the Everyday row first —
    // asserting on the Synthesis row before it resolves would pass on the
    // still-loading placeholder regardless of the bug.
    const everydayRow = (await screen.findByText("Everyday model")).closest("button")!;
    expect(await within(everydayRow).findByText("Qwen3 4B Instruct")).toBeInTheDocument();

    const synthesisRow = screen.getByText("Synthesis model").closest("button")!;
    expect(within(synthesisRow).getByText("No model is assigned — pages still update whenever your AI tools use Wenlan.")).toBeInTheDocument();
  });

  it("shows the On-device row's capability hint alongside its state-derived meta", async () => {
    mocks.getOnDeviceModel.mockResolvedValue({
      loaded: "qwen3-4b-instruct-2507",
      selected: "qwen3-4b-instruct-2507",
      models: [{ id: "qwen3-4b-instruct-2507", display_name: "Qwen3 4B Instruct", cached: true }],
    });
    renderSection();

    const onDeviceRow = screen.getByText("On-device model").closest("button")!;
    expect(
      within(onDeviceRow).getByText("Runs entirely on this device. Larger cloud or server models usually give better synthesis quality.")
    ).toBeInTheDocument();
    expect(await within(onDeviceRow).findByText("Qwen3 4B Instruct · Running")).toBeInTheDocument();
  });

  // ── Headline (a): LEGACY mode — routing endpoint absent (null). The source
  // is read-only with the updated-runtime hint, but the model select still
  // works. Mutation proof: dropping the `isPinned ?` branch (rendering the
  // interactive source select in legacy) makes the "Choose source" absence
  // assertion fail; dropping the model select breaks the setModelChoice call.
  it("legacy mode: source is read-only with a runtime hint, model select still functional", async () => {
    mocks.getApiKey.mockResolvedValue("sk-ant-***configured");
    mocks.getModelChoice.mockResolvedValue(["claude-opus-4-6", null]);
    // routing stays null (LEGACY) from beforeEach.
    renderSection();

    const everydayRow = (await screen.findByText("Everyday model")).closest("button")!;
    await within(everydayRow).findByText("Anthropic · Opus 4.6");
    await userEvent.click(everydayRow);

    // Read-only source line + the "needs the updated runtime" hint.
    expect(await screen.findByText("Changing the source needs the updated Wenlan runtime.")).toBeInTheDocument();
    // No interactive source picker in legacy mode.
    expect(screen.queryByLabelText("Choose source")).not.toBeInTheDocument();

    // The model select is live: switching off the current Opus value writes
    // through setModelChoice (a different value, so onChange actually fires).
    const modelSelect = screen.getByLabelText("Choose everyday model");
    await userEvent.selectOptions(modelSelect, "claude-haiku-4-5-20251001");
    expect(mocks.setModelChoice).toHaveBeenCalledWith("claude-haiku-4-5-20251001", null);
    // A source pin is never written at a legacy daemon.
    expect(mocks.setSourcePin).not.toHaveBeenCalled();
  });

  // ── Headline (b): PINNED mode — the row reflects the routing endpoint (not
  // local derivation), and picking a source issues the pin write. Mutation
  // proof: re-deriving the meta from modelChoice shows "Haiku 4.5" and fails
  // the "Opus 4.6" assertion; a no-op source select never calls setSourcePin.
  it("pinned mode: reflects endpoint routing and pins the picked source", async () => {
    mocks.getApiKey.mockResolvedValue("sk-ant-***configured");
    // Local knobs say Haiku; the endpoint says Opus. The row must show Opus.
    mocks.getModelChoice.mockResolvedValue(["claude-haiku-4-5-20251001", "claude-sonnet-4-6"]);
    mocks.getResolvedRouting.mockResolvedValue(
      pinnedRouting({ everyday: { source: "anthropic", model: "claude-opus-4-6", mode: "pinned" } })
    );
    renderSection();

    const everydayRow = (await screen.findByText("Everyday model")).closest("button")!;
    // Reflected from routing, not re-derived from the Haiku modelChoice.
    expect(await within(everydayRow).findByText("Anthropic · Opus 4.6")).toBeInTheDocument();
    expect(within(everydayRow).queryByText("Anthropic · Haiku 4.5")).not.toBeInTheDocument();

    await userEvent.click(everydayRow);
    const sourceSelect = await screen.findByLabelText("Choose source");
    await userEvent.selectOptions(sourceSelect, "external");
    expect(mocks.setSourcePin).toHaveBeenCalledWith("external", null);
  });

  // ── Headline (c): PINNED_DEGRADED — the pinned source is unavailable, so the
  // amber hint names it: "Pinned to X — using Y for now". Mutation proof:
  // removing the `isPinned && degraded` branch drops the hint and fails this
  // assertion.
  it("pinned_degraded mode: renders the amber hint naming the pinned source", async () => {
    mocks.getResolvedRouting.mockResolvedValue(
      pinnedRouting({
        everyday: { source: "on_device", model: "qwen3-4b-instruct-2507", mode: "pinned" },
        // Pinned to Anthropic, but the daemon resolved it as degraded, so the
        // route fell back to the connected external provider (OpenAI). The
        // component trusts the wire values as-is — see the next test for the
        // no-pin (pre-#357 daemon) fallback case.
        synthesis: { source: "external", model: "gpt-5.2", mode: "pinned_degraded", pin: "anthropic" },
      })
    );
    renderSection();

    const synthesisRow = (await screen.findByText("Synthesis model")).closest("button")!;
    await userEvent.click(synthesisRow);

    expect(await screen.findByText("Pinned to Anthropic — currently unavailable, using OpenAI for now.")).toBeInTheDocument();
  });

  // ── Headline (d): PINNED_DEGRADED with no pin on the wire (a daemon that
  // predates #357's pin field) — falls back to the generic, unnamed hint
  // rather than rendering "Pinned to null". Mutation proof: forcing
  // `pinnedDisplay` to always resolve from `pin` (dropping the `pin ?` guard)
  // renders the named hint here and fails the generic-hint assertion.
  it("pinned_degraded mode with no pin on the wire: falls back to the generic hint", async () => {
    mocks.getResolvedRouting.mockResolvedValue(
      pinnedRouting({
        everyday: { source: "on_device", model: "qwen3-4b-instruct-2507", mode: "pinned" },
        synthesis: { source: "external", model: "gpt-5.2", mode: "pinned_degraded", pin: null },
      })
    );
    renderSection();

    const synthesisRow = (await screen.findByText("Synthesis model")).closest("button")!;
    await userEvent.click(synthesisRow);

    expect(await screen.findByText("The pinned source is currently unavailable — using OpenAI for now.")).toBeInTheDocument();
    expect(screen.queryByText(/Pinned to/)).not.toBeInTheDocument();
  });

  // ── Empty-state defect: nothing connected, Synthesis resolves to "none" in
  // LEGACY mode. The expanded body must give an actionable next step, and the
  // none-state meta string must appear EXACTLY ONCE (collapsed meta only) —
  // never duplicated by a body branch. Mutation proof: restoring the old
  // `source === "none"` body line that re-rendered the none-state meta
  // (synthesisNoSourceHint) makes the count 2 and fails toHaveLength(1).
  it("synthesis 'none' expanded: shows connect-provider guidance, none-state meta appears once", async () => {
    // All beforeEach defaults: no key, no external, no on-device, routing null
    // (LEGACY). Everyday resolves to "basic", Synthesis to "none".
    renderSection();

    const synthesisRow = (await screen.findByText("Synthesis model")).closest("button")!;
    const rowRoot = synthesisRow.parentElement!;
    await userEvent.click(synthesisRow);

    // Waiting on the guidance settles the queries before the count assertion,
    // so the count can't false-green on a still-loading row.
    expect(
      await screen.findByText("Connect a cloud provider or local server below to serve this job.")
    ).toBeInTheDocument();
    // The collapsed meta keeps the cloud-required string; the body no longer repeats it.
    expect(
      within(rowRoot).getAllByText("No model is assigned — pages still update whenever your AI tools use Wenlan.")
    ).toHaveLength(1);
  });

  // ── Empty-state defect: PINNED mode with nothing configured must still show
  // the source picker (a fresh user has to SEE what's choosable), with every
  // option disabled. Mutation proof: re-gating the pinned select on
  // isProviderSource drops it for a "none" source and fails findByLabelText.
  it("pinned mode, nothing configured: the source select renders with every option disabled", async () => {
    mocks.getResolvedRouting.mockResolvedValue({
      everyday: { source: "basic", model: null, mode: "auto", pin: null },
      synthesis: { source: "none", model: null, mode: "auto", pin: null },
      pool: {
        anthropic: { configured: false, everyday_model: null, synthesis_model: null },
        external: null,
        on_device: null,
      },
    });
    renderSection();

    const synthesisRow = (await screen.findByText("Synthesis model")).closest("button")!;
    await userEvent.click(synthesisRow);

    const sourceSelect = await screen.findByLabelText("Choose source");
    const options = within(sourceSelect).getAllByRole("option") as HTMLOptionElement[];
    expect(options.length).toBeGreaterThan(0);
    expect(options.filter((o) => !o.disabled)).toHaveLength(0);

    // PINNED mode points at the select above (not the legacy connect-below-only
    // wording — that renders in the "synthesis 'none' expanded" legacy test).
    expect(
      screen.getByText("Choose a source above, or connect a provider below to serve this job.")
    ).toBeInTheDocument();
  });

  // ── Synthesis can now use on-device (daemon #357 makes synthesis=on_device a
  // valid pin). The Synthesis row must OFFER on-device as an enabled source
  // when a model is loaded, and picking it must pin synthesis→on_device (the
  // (everyday, synthesis) arg pair means null in the everyday slot). Mutation
  // proof: dropping "on_device" from the synthesis branch of buildOptions'
  // `order` removes the option and fails the getByRole("option", { name:
  // "On-device" }) query.
  it("synthesis offers an available On-device source and pins synthesis→on_device", async () => {
    // pinnedRouting()'s pool has on_device configured, so the option is enabled.
    mocks.getResolvedRouting.mockResolvedValue(pinnedRouting());
    renderSection();

    const synthesisRow = (await screen.findByText("Synthesis model")).closest("button")!;
    await userEvent.click(synthesisRow);

    const sourceSelect = await screen.findByLabelText("Choose source");
    const onDeviceOption = within(sourceSelect).getByRole("option", { name: "On-device" }) as HTMLOptionElement;
    expect(onDeviceOption.disabled).toBe(false);

    await userEvent.selectOptions(sourceSelect, "on_device");
    expect(mocks.setSourcePin).toHaveBeenCalledWith(null, "on_device");
  });

  // ── The on-device job row exposes a model dropdown (jobs choose — no "managed
  // below" deflection). Cached models are selectable, uncached ones disabled
  // with a pointer to the provider row, and switching reuses OnDeviceModelCard's
  // exact mechanism: downloadOnDeviceModel(id) (loads if cached). Mutation proof:
  // no-op'ing the `await downloadOnDeviceModel(e.target.value)` in
  // OnDeviceJobModelSelect.onChange makes the switch write nothing and fails
  // toHaveBeenCalledWith("qwen3-8b") — and only this test.
  it("on-device job row: dropdown lists cached (enabled) + uncached (disabled), switching loads the picked model", async () => {
    mocks.getResolvedRouting.mockResolvedValue(
      pinnedRouting({ everyday: { source: "on_device", model: "qwen3-4b-instruct-2507", mode: "pinned" } })
    );
    mocks.getOnDeviceModel.mockResolvedValue({
      loaded: "qwen3-4b-instruct-2507",
      selected: "qwen3-4b-instruct-2507",
      models: [
        { id: "qwen3-4b-instruct-2507", display_name: "Qwen3 4B", param_count: "4B", ram_required_gb: 6, file_size_gb: 2.7, cached: true },
        { id: "qwen3-8b", display_name: "Qwen3 8B", param_count: "8B", ram_required_gb: 10, file_size_gb: 5.2, cached: true },
        { id: "qwen3-14b", display_name: "Qwen3 14B", param_count: "14B", ram_required_gb: 16, file_size_gb: 9.0, cached: false },
      ],
    });
    renderSection();

    const everydayRow = (await screen.findByText("Everyday model")).closest("button")!;
    await userEvent.click(everydayRow);

    const modelSelect = await screen.findByLabelText("Choose on-device model");
    expect((within(modelSelect).getByRole("option", { name: "Qwen3 8B" }) as HTMLOptionElement).disabled).toBe(false);
    expect((within(modelSelect).getByRole("option", { name: "Qwen3 14B" }) as HTMLOptionElement).disabled).toBe(true);
    // The uncached entry surfaces the download pointer to the provider row.
    expect(screen.getByText("Uncached models download from the On-device model row below.")).toBeInTheDocument();

    await userEvent.selectOptions(modelSelect, "qwen3-8b");
    expect(mocks.downloadOnDeviceModel).toHaveBeenCalledWith("qwen3-8b");
  });

  // ── Value disclosure (with a model): the everyday row leads with its "what you
  // get" summary and the ⓘ carries the background-organization detail. Mutation
  // proof: flipping jobRow's `hint={on ? summary : undefined}` to always-undefined
  // drops the summary and fails the getByText; swapping the connected tip branch
  // (using the no-model tip) changes the ⓘ's accessible name and fails getByRole.
  it("everyday row with a model: shows the value summary and the with-model ⓘ tip", async () => {
    mocks.getApiKey.mockResolvedValue("sk-ant-***configured");
    mocks.getModelChoice.mockResolvedValue(["claude-haiku-4-5-20251001", "claude-sonnet-4-6"]);
    renderSection();

    const everydayRow = (await screen.findByText("Everyday model")).closest("button")!;
    // Model resolved → the meta names it and the summary sits above it.
    await within(everydayRow).findByText("Anthropic · Haiku 4.5");
    expect(within(everydayRow).getByText("Files and links new memories in the background")).toBeInTheDocument();
    // The ⓘ is a sibling of the row button, so scope to the header row. Its
    // accessible name is the with-model tip title, not the no-model one.
    const header = everydayRow.parentElement!;
    expect(within(header).getByRole("button", { name: "Background organization" })).toBeInTheDocument();
    expect(within(header).queryByRole("button", { name: "What a model adds" })).not.toBeInTheDocument();
  });

  // ── Value disclosure (no model): the everyday row drops the summary (the meta
  // already states the no-model reality) and the ⓘ flips to "what a model adds".
  // Mutation proof: flipping `hint={on ? summary : undefined}` to always-`summary`
  // renders the summary here and fails the queryByText absence; swapping the tip
  // branch (using the with-model tip) changes the ⓘ name and fails getByRole.
  it("everyday row without a model: drops the summary and the ⓘ becomes 'what a model adds'", async () => {
    // All beforeEach defaults → everyday resolves to "basic" (no model serves it).
    renderSection();

    const everydayRow = (await screen.findByText("Everyday model")).closest("button")!;
    await within(everydayRow).findByText("No background model — your wiki updates through your AI tools.");
    // The with-model summary must NOT appear when nothing serves the job.
    expect(within(everydayRow).queryByText("Files and links new memories in the background")).not.toBeInTheDocument();
    const header = everydayRow.parentElement!;
    expect(within(header).getByRole("button", { name: "What a model adds" })).toBeInTheDocument();
    expect(within(header).queryByRole("button", { name: "Background organization" })).not.toBeInTheDocument();
  });
});
