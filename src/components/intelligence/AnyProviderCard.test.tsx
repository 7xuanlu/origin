// SPDX-License-Identifier: AGPL-3.0-only
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, waitFor, fireEvent, act, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import "../../i18n";

const mocks = vi.hoisted(() => ({
  getDaemonVersion: vi.fn(),
  getExternalLlm: vi.fn(),
  setExternalLlm: vi.fn(),
  testExternalLlm: vi.fn(),
  listExternalModels: vi.fn(),
  getExternalLlmKeyConfigured: vi.fn(),
  getApiKey: vi.fn(),
  setApiKey: vi.fn(),
  getModelChoice: vi.fn(),
  setModelChoice: vi.fn(),
}));
vi.mock("../../lib/tauri", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../../lib/tauri")>();
  return { ...actual, ...mocks };
});

import AnyProviderCard from "./AnyProviderCard";
import type { PresetGroup } from "./providerPresets";

const CLOUD_GROUPS: PresetGroup[] = ["cloud"];
const LOCAL_GROUPS: PresetGroup[] = ["local", "custom"];

// This suite's describe block predates the unified chip row and exercises
// local-server behavior specifically, so its default keeps that scope —
// Anthropic isn't in LOCAL_GROUPS, so none of these calls see it. Tests that
// need the cloud row or the unscoped (Settings) card pass `groups` explicitly.
function renderCard(
  qc: QueryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } }),
  groups: PresetGroup[] | undefined = LOCAL_GROUPS,
) {
  render(
    <QueryClientProvider client={qc}>
      <AnyProviderCard groups={groups} />
    </QueryClientProvider>
  );
  return qc;
}

function renderAllScope(qc: QueryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } })) {
  render(
    <QueryClientProvider client={qc}>
      <AnyProviderCard />
    </QueryClientProvider>
  );
  return qc;
}

describe("AnyProviderCard — the Local-server card (spec §5.2)", () => {
  // NOTE: reset runs in afterEach, not beforeEach — see
  // src/hooks/useDaemonVersion.test.tsx for why. Resetting a vi.fn() in
  // beforeEach immediately before it's reconfigured with mockRejectedValue
  // and consumed by a react-query queryFn deterministically triggers a
  // false-positive "unhandled rejection" failure on this project's pinned
  // vitest 4.1.5 + @tanstack/react-query 5.100.9 + React 19 combination.
  afterEach(() => {
    Object.values(mocks).forEach((m) => m.mockReset());
  });

  beforeEach(() => {
    mocks.getDaemonVersion.mockResolvedValue("0.12.0");
    mocks.getExternalLlm.mockResolvedValue([null, null]);
    mocks.getExternalLlmKeyConfigured.mockResolvedValue(false);
    mocks.getApiKey.mockResolvedValue(null);
    mocks.listExternalModels.mockResolvedValue(["llama3.2:3b"]);
    mocks.setExternalLlm.mockResolvedValue(undefined);
    mocks.testExternalLlm.mockResolvedValue({ response: "pong" });
    mocks.setApiKey.mockResolvedValue(undefined);
    mocks.getModelChoice.mockResolvedValue([null, null]);
    mocks.setModelChoice.mockResolvedValue(undefined);
  });

  // Defect 1 (spec §5.2/§5.2a): the 7 keyed cloud vendors the daemon cannot
  // actually authenticate must not exist anywhere in this card's UI, and the
  // key Field must not render — not "disabled with an explanation", gone.
  it("no keyed vendor appears in the preset picker, no API-key Field renders, and the initial preset is Ollama", async () => {
    renderCard();
    await screen.findByRole("group", { name: "Provider" });
    expect(screen.queryByText("OpenAI")).not.toBeInTheDocument();
    expect(screen.queryByText("Groq")).not.toBeInTheDocument();
    expect(screen.queryByText("Gemini")).not.toBeInTheDocument();
    expect(screen.queryByLabelText("API key")).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Ollama" })).toHaveAttribute("aria-pressed", "true");
    expect(screen.getByLabelText("Endpoint URL")).toHaveValue("http://localhost:11434/v1");
  });

  // Defect 1: no permanently-dead button state survives — Test/Save are
  // disabled only for a genuinely-incomplete form, and become enabled the
  // moment the form is complete (never locked behind a daemon version).
  it("Test and Save enable once endpoint and model are both set — no permanently-dead button state", async () => {
    renderCard();
    expect(screen.getByRole("button", { name: "Test" })).toBeDisabled();
    expect(screen.getByRole("button", { name: "Save" })).toBeDisabled();

    await screen.findByText(/Connected to Ollama/);
    await userEvent.selectOptions(screen.getByLabelText("Model"), "llama3.2:3b");

    expect(screen.getByRole("button", { name: "Test" })).toBeEnabled();
    expect(screen.getByRole("button", { name: "Save" })).toBeEnabled();
  });

  it("disables Save and Test while the model field is empty, even with a valid endpoint (canAct guard)", async () => {
    renderCard();
    expect(screen.getByLabelText("Endpoint URL")).toHaveValue("http://localhost:11434/v1");
    expect(screen.getByRole("button", { name: "Test" })).toBeDisabled();
    expect(screen.getByRole("button", { name: "Save" })).toBeDisabled();
  });

  it("keyless save omits the key and shows the restart note", async () => {
    renderCard();
    await screen.findByText(/Connected to Ollama/);
    await userEvent.selectOptions(screen.getByLabelText("Model"), "llama3.2:3b");
    await userEvent.click(screen.getByRole("button", { name: "Save" }));
    await waitFor(() =>
      expect(mocks.setExternalLlm).toHaveBeenCalledWith(
        "http://localhost:11434/v1", "llama3.2:3b", undefined
      )
    );
    expect(await screen.findByText(/Restart Wenlan to apply/)).toBeInTheDocument();
  });

  it("test button shows verbatim daemon error", async () => {
    mocks.testExternalLlm.mockRejectedValue(new Error("LLM request failed: 401 Unauthorized"));
    renderCard();
    await screen.findByText(/Connected to Ollama/);
    await userEvent.selectOptions(screen.getByLabelText("Model"), "llama3.2:3b");
    await userEvent.click(screen.getByRole("button", { name: "Test" }));
    expect(await screen.findByText(/401 Unauthorized/)).toBeInTheDocument();
  });

  it("invalidates external-llm and external-llm-key-configured after a successful save (job rows re-read the routing)", async () => {
    const qc = renderCard();
    const invalidateSpy = vi.spyOn(qc, "invalidateQueries");
    await screen.findByText(/Connected to Ollama/);
    await userEvent.selectOptions(screen.getByLabelText("Model"), "llama3.2:3b");
    await userEvent.click(screen.getByRole("button", { name: "Save" }));
    await waitFor(() => expect(mocks.setExternalLlm).toHaveBeenCalled());

    expect(invalidateSpy).toHaveBeenCalledWith({ queryKey: ["external-llm"] });
    expect(invalidateSpy).toHaveBeenCalledWith({ queryKey: ["external-llm-key-configured"] });
  });

  it("shows the Anthropic precedence warning when an Anthropic key is configured", async () => {
    mocks.getApiKey.mockResolvedValue("sk-ant-configured");
    renderCard();
    expect(await screen.findByText(/Anthropic takes precedence/)).toBeInTheDocument();
  });

  // The exact interpolated singular/plural string (i18next _one/_other) —
  // the default mock resolves exactly one model.
  it("singular: exactly one discovered model reads '1 model', not '1 models'", async () => {
    renderCard();
    expect(await screen.findByText("Connected to Ollama — 1 model")).toBeInTheDocument();
    expect(screen.queryByText(/1 models/)).not.toBeInTheDocument();
  });

  it("plural: two discovered models reads '2 models'", async () => {
    mocks.listExternalModels.mockResolvedValue(["llama3.2:3b", "qwen2.5:7b"]);
    renderCard();
    expect(await screen.findByText("Connected to Ollama — 2 models")).toBeInTheDocument();
  });

  // Thread #3: the probe genuinely succeeded (server reachable — chip stays
  // "up", never red) but a server with zero models can't serve anything, so
  // the label must not read as success.
  it("zero models: reachable server with no models installed gets the no-models label, not the connected label", async () => {
    mocks.listExternalModels.mockResolvedValue([]);
    renderCard();
    expect(await screen.findByText("Ollama is running — no models installed yet")).toBeInTheDocument();
    expect(screen.queryByText(/Connected to Ollama/)).not.toBeInTheDocument();
  });

  // Endpoint normalization: a hand-typed endpoint missing a scheme and a
  // /v1 path must still resolve to the same probe target as the canonical
  // form, not silently drop to a no-match (component-level integration
  // check; the pure-function cases live in providerPresets.test.ts).
  it("normalizes a hand-typed scheme-less, /v1-less endpoint before probing", async () => {
    renderCard();
    await screen.findByText(/Connected to Ollama/);
    mocks.listExternalModels.mockClear();

    await userEvent.click(await screen.findByRole("button", { name: "Custom…" }));
    await userEvent.type(screen.getByLabelText("Endpoint URL"), "192.168.1.5:11434");

    await waitFor(() =>
      expect(mocks.listExternalModels).toHaveBeenCalledWith("http://192.168.1.5:11434/v1", null)
    );
  });

  it("both servers up → both pills connected, no auto-switch", async () => {
    // both probes succeed; resolve 2 models each so the chip's exact model count is meaningful.
    mocks.listExternalModels.mockResolvedValue(["llama3.2:3b", "qwen2.5:7b"]);
    renderCard();
    // Exact interpolated text — guards the {{count}} (not modelCount) i18next key.
    expect(await screen.findByText("Connected to Ollama — 2 models")).toBeInTheDocument();
    // Exact accessible name: the status-dot span must not fold into the pill's name.
    const ollamaPill = screen.getByRole("button", { name: "Ollama" });
    const lmStudioPill = screen.getByRole("button", { name: "LM Studio" });
    expect(ollamaPill).toBeInTheDocument();
    expect(lmStudioPill).toBeInTheDocument();
    // Selection is communicated to assistive tech via aria-pressed, not just color.
    expect(ollamaPill).toHaveAttribute("aria-pressed", "true");
    expect(lmStudioPill).toHaveAttribute("aria-pressed", "false");
  });

  it("both probes still pending → probing chip renders", async () => {
    mocks.listExternalModels.mockImplementation(() => new Promise(() => {}));
    renderCard();
    expect(await screen.findByText("Checking Ollama…")).toBeInTheDocument();
  });

  it("exactly one server up → auto-selects it", async () => {
    mocks.listExternalModels.mockImplementation((ep: string) =>
      ep.includes("1234")
        ? Promise.resolve(["qwen2.5:7b"])
        : Promise.reject(new Error("ECONNREFUSED")),
    );
    renderCard();
    // LM Studio (1234) is the sole responder → its chip is shown.
    expect(await screen.findByText(/Connected to LM Studio/)).toBeInTheDocument();
    expect(screen.getByLabelText("Endpoint URL")).toHaveValue("http://localhost:1234/v1");
  });

  it("no server up → not-detected chip for the selected pill", async () => {
    mocks.listExternalModels.mockRejectedValue(new Error("ECONNREFUSED"));
    renderCard();
    expect(
      await screen.findByText(/Not detected at localhost:11434 — is Ollama running\?/),
    ).toBeInTheDocument();
  });

  // Chip-never-lies, color channel: the label text alone can read correctly
  // even if the chip's rendered TONE were wired to something other than the
  // real probe (e.g. the selected preset) — the label and the color are
  // computed by two separate ternaries in the component. This pins the tone
  // independently so that divergence can't hide behind a passing text match.
  it("chip-never-lies: a failed probe renders the danger tone, never success, regardless of which preset is selected", async () => {
    mocks.listExternalModels.mockRejectedValue(new Error("ECONNREFUSED"));
    renderCard();
    const chipLabel = await screen.findByText(/Not detected at localhost:11434/);
    const chip = chipLabel.closest('[aria-live="polite"]');
    expect(chip).not.toBeNull();
    expect(chip?.className).toContain("mem-status-danger-text");
    expect(chip?.className).not.toContain("mem-status-success-text");
  });

  it("discovered models render as a <select>, not free text", async () => {
    mocks.listExternalModels.mockResolvedValue(["qwen2.5:7b", "llama3.2:3b"]);
    renderCard();
    await screen.findByText(/Connected to Ollama/);
    const modelField = await screen.findByLabelText("Model");
    expect(modelField.tagName).toBe("SELECT");
    await userEvent.selectOptions(modelField, "llama3.2:3b");
    expect(modelField).toHaveValue("llama3.2:3b");
  });

  it("discovery failure keeps free-text model entry with hint", async () => {
    mocks.listExternalModels.mockRejectedValue(new Error("ECONNREFUSED"));
    renderCard();
    await screen.findByText(/Not detected at localhost:11434/);
    const modelField = await screen.findByLabelText("Model");
    expect(modelField.tagName).toBe("INPUT");
    expect(screen.getByText(/type a model name/i)).toBeInTheDocument();
  });

  it("does not duplicate-fetch a probed preset's endpoint via the generic discovery query", async () => {
    // Default mock (["llama3.2:3b"]) covers both probes; the probed preset
    // (Ollama, the default selection) must not ALSO trigger the generic
    // `discovery` query against the same endpoint.
    renderCard();
    await screen.findByText(/Connected to Ollama/);
    expect(mocks.listExternalModels).toHaveBeenCalledTimes(2);
    expect(mocks.listExternalModels).toHaveBeenCalledWith("http://localhost:11434/v1", null);
    expect(mocks.listExternalModels).toHaveBeenCalledWith("http://localhost:1234/v1", null);
  });

  it("a hand-typed Custom endpoint still gets model discovery (no discovery hole for unprobed presets)", async () => {
    mocks.listExternalModels.mockResolvedValue(["custom-model-x"]);
    renderCard();
    await userEvent.click(await screen.findByRole("button", { name: "Custom…" }));
    await userEvent.type(screen.getByLabelText("Endpoint URL"), "http://localhost:9999/v1");
    await waitFor(() =>
      expect(mocks.listExternalModels).toHaveBeenCalledWith("http://localhost:9999/v1", null)
    );
    // Custom/unprobed presets have no dedicated probe, so they never get the
    // polished <select> — but discovery still runs and feeds the datalist,
    // so free-text entry is never left without autocomplete.
    const modelField = screen.getByLabelText("Model");
    expect(modelField.tagName).toBe("INPUT");
    expect(
      document.getElementById("any-provider-models")?.querySelector('option[value="custom-model-x"]'),
    ).toBeTruthy();
  });

  it("custom endpoint discovery failure keeps free text with the discovery-failed hint", async () => {
    mocks.listExternalModels.mockRejectedValue(new Error("ECONNREFUSED"));
    renderCard();
    await userEvent.click(await screen.findByRole("button", { name: "Custom…" }));
    await userEvent.type(screen.getByLabelText("Endpoint URL"), "http://localhost:9999/v1");
    expect(await screen.findByText(/type a model name/i)).toBeInTheDocument();
    expect(screen.getByLabelText("Model").tagName).toBe("INPUT");
  });

  // Chip-never-lies (the invariant, pinned hard): the chip and the model
  // dropdown must follow the endpoint that `discovery` actually probed, not
  // the merely-selected preset — hand-editing the endpoint off Ollama's
  // default must drop the stale "Connected to Ollama" claim immediately.
  it("chip-never-lies: hand-editing the endpoint off the probed Ollama preset drops the stale chip and dropdown", async () => {
    renderCard();
    // Ollama pill is the default selection and is probed/connected.
    await screen.findByText(/Connected to Ollama/);
    expect(screen.getByLabelText("Model").tagName).toBe("SELECT");
    mocks.listExternalModels.mockClear();

    await userEvent.clear(screen.getByLabelText("Endpoint URL"));
    await userEvent.type(screen.getByLabelText("Endpoint URL"), "http://192.168.1.5:11434/v1");

    // The edited endpoint no longer matches the Ollama preset's endpoint, so
    // the probe association must drop: no stale chip, no dropdown of the
    // WRONG server's models, and the generic discovery query must pick up
    // the live value.
    expect(screen.queryByText(/Connected to Ollama/)).not.toBeInTheDocument();
    expect(screen.getByLabelText("Model").tagName).toBe("INPUT");
    await waitFor(() =>
      expect(mocks.listExternalModels).toHaveBeenCalledWith("http://192.168.1.5:11434/v1", null)
    );
  });

  it("a saved model no longer discoverable stays visible and selectable, discovered ids still choosable", async () => {
    mocks.getExternalLlm.mockResolvedValue(["http://localhost:11434/v1", "llama3.2:1b"]);
    renderCard();
    await screen.findByText(/Connected to Ollama/);

    const modelField = await screen.findByLabelText("Model");
    expect(modelField.tagName).toBe("SELECT");
    // Saved model ("llama3.2:1b") isn't among the discovered ids
    // (["llama3.2:3b"], from the beforeEach default) — it must still render
    // as the selected value instead of a blank select.
    expect(modelField).toHaveValue("llama3.2:1b");
    // The discovered id must still be a choosable option.
    await userEvent.selectOptions(modelField, "llama3.2:3b");
    expect(modelField).toHaveValue("llama3.2:3b");
  });

  // Thread #5: the discovery query key is [trimmedEndpoint, apiKey], so
  // every keystroke used to fire its own fetch. Debouncing must collapse a
  // burst of rapid edits into exactly one fetch, fired only after the
  // burst settles.
  it("debounces discovery: a rapid burst of endpoint edits fires exactly one fetch, not one per edit", async () => {
    renderCard();
    await screen.findByText(/Connected to Ollama/);
    // Grab references while still on real timers — findBy*/waitFor poll via
    // setTimeout internally and must not run once fake timers take over.
    const customButton = screen.getByRole("button", { name: "Custom…" });
    const input = screen.getByLabelText("Endpoint URL");
    mocks.listExternalModels.mockClear();

    vi.useFakeTimers();
    try {
      // Switching to Custom also resets the endpoint (to "") through the
      // same debounced state, so it must happen under fake time too — a
      // debounce timer scheduled on real timers would fire mid-test.
      fireEvent.click(customButton);
      fireEvent.change(input, { target: { value: "h" } });
      fireEvent.change(input, { target: { value: "http://localhost:9999" } });
      fireEvent.change(input, { target: { value: "http://localhost:9999/v1" } });

      // Still within the debounce window — no fetch yet.
      expect(mocks.listExternalModels).not.toHaveBeenCalled();

      await act(async () => {
        await vi.advanceTimersByTimeAsync(400);
      });
    } finally {
      vi.useRealTimers();
    }

    expect(mocks.listExternalModels).toHaveBeenCalledTimes(1);
    expect(mocks.listExternalModels).toHaveBeenCalledWith("http://localhost:9999/v1", null);
  });

  // §5.2a: the widened preset picker (cloud vendors + key auth) is gated on
  // the daemon-0.13 `supportsExternalKey` floor. Below the floor this card
  // must behave exactly as before the widening. These tests exercise the
  // unscoped (Settings) card, since `groups` is what now decides which
  // vendors are eligible — the gate alone no longer determines cloud
  // presence (Anthropic, native, is exempt from the gate entirely).
  describe("cloud preset gating (§5.2a)", () => {
    it("gate CLOSED (0.12 daemon): no OpenAI pill renders, but Anthropic still does (native bypasses the gate)", async () => {
      renderAllScope();
      await screen.findByRole("button", { name: "Anthropic" });
      expect(screen.queryByRole("button", { name: "OpenAI" })).not.toBeInTheDocument();
    });

    it("gate OPEN (0.13 daemon): an OpenAI pill renders alongside Anthropic", async () => {
      mocks.getDaemonVersion.mockResolvedValue("0.13.0");
      renderAllScope();
      expect(await screen.findByRole("button", { name: "OpenAI" })).toBeInTheDocument();
      expect(screen.getByRole("button", { name: "Anthropic" })).toBeInTheDocument();
    });

    it("gate OPEN: the default selected pill stays Ollama, never OpenAI or Anthropic", async () => {
      mocks.getDaemonVersion.mockResolvedValue("0.13.0");
      renderAllScope();
      await screen.findByRole("button", { name: "OpenAI" });
      expect(screen.getByRole("button", { name: "Ollama" })).toHaveAttribute("aria-pressed", "true");
      expect(screen.getByRole("button", { name: "OpenAI" })).toHaveAttribute("aria-pressed", "false");
      expect(screen.getByRole("button", { name: "Anthropic" })).toHaveAttribute("aria-pressed", "false");
    });

    it("gate OPEN: selecting the OpenAI pill reveals the API key Field and the Get-a-key link", async () => {
      mocks.getDaemonVersion.mockResolvedValue("0.13.0");
      renderAllScope();
      await userEvent.click(await screen.findByRole("button", { name: "OpenAI" }));
      expect(await screen.findByLabelText("API key")).toBeInTheDocument();
      expect(screen.getByRole("button", { name: "Get a key →" })).toBeInTheDocument();
    });

    it("gate OPEN: local presets still render before cloud presets in DOM order", async () => {
      mocks.getDaemonVersion.mockResolvedValue("0.13.0");
      renderAllScope();
      await screen.findByRole("button", { name: "OpenAI" });
      const lmStudioPill = screen.getByRole("button", { name: "LM Studio" });
      const openAiPill = screen.getByRole("button", { name: "OpenAI" });
      expect(
        lmStudioPill.compareDocumentPosition(openAiPill) & Node.DOCUMENT_POSITION_FOLLOWING,
      ).toBeTruthy();
    });
  });

  // §Unified chip row: Anthropic is now a chip in the SAME row as the other
  // cloud vendors (no separate Anthropic-only card), scoped per host via
  // `groups`. These pin the 6 behaviors the redesign depends on: dispatch
  // never crosses the native/external-llm boundary, scope never leaks a
  // vendor from the wrong tile, and the daemon-version gate never empties
  // the cloud row (Anthropic is exempt from it).
  describe("unified chip row — Anthropic as a cloud-scoped chip", () => {
    it("cloud-scoped card: Anthropic renders in the same chip row as OpenAI and friends, no local-server pills", async () => {
      mocks.getDaemonVersion.mockResolvedValue("0.13.0");
      renderCard(undefined, CLOUD_GROUPS);
      // Anthropic (native) renders on the very first synchronous paint, before
      // the daemon-version query resolves — it's not a valid "gate is open"
      // anchor. OpenAI only appears once the gate opens, so wait on it instead.
      await screen.findByRole("button", { name: "OpenAI" });
      expect(screen.getByRole("button", { name: "Anthropic" })).toBeInTheDocument();
      expect(screen.getByRole("button", { name: "Gemini" })).toBeInTheDocument();
      expect(screen.queryByRole("button", { name: "Ollama" })).not.toBeInTheDocument();
      expect(screen.queryByRole("button", { name: "LM Studio" })).not.toBeInTheDocument();
    });

    it("cloud-scoped card: selecting Anthropic shows the key field with no Endpoint or Model field, and Save calls setApiKey — never setExternalLlm", async () => {
      mocks.getDaemonVersion.mockResolvedValue("0.13.0");
      renderCard(undefined, CLOUD_GROUPS);
      await userEvent.click(await screen.findByRole("button", { name: "OpenAI" }));
      // Neither preset shows an Endpoint field anymore (defect fix: known
      // cloud endpoints are constants, not user input) — the generic form's
      // Model field is the reliable anchor that OpenAI actually dispatched,
      // since AnthropicFields never renders one.
      expect(await screen.findByLabelText("Model")).toBeInTheDocument();

      await userEvent.click(screen.getByRole("button", { name: "Anthropic" }));
      expect(screen.queryByLabelText("Model")).not.toBeInTheDocument();
      expect(screen.queryByLabelText("Endpoint URL")).not.toBeInTheDocument();
      await userEvent.type(screen.getByLabelText("API key"), "sk-ant-test-key");
      await userEvent.click(screen.getByRole("button", { name: "Save" }));

      await waitFor(() => expect(mocks.setApiKey).toHaveBeenCalledWith("sk-ant-test-key"));
      expect(mocks.setExternalLlm).not.toHaveBeenCalled();
    });

    // Defect fix (user report): a known cloud vendor's endpoint is a
    // constant from the preset table, not a user decision — asking for it
    // was noise. Hiding the field must not change what actually gets sent.
    // Model is dropdown-only for cloud vendors (user report), so there is no
    // free-text input to type into once a key resolves models — Save must
    // still send the fixed endpoint and the model SELECTED from the
    // dropdown. This is the one that matters most: hiding the text input
    // must not break the value that actually gets sent.
    it("cloud-scoped card: OpenAI shows key + Model fields but no Endpoint field, and Save still calls setExternalLlm with the fixed OpenAI endpoint and the selected model", async () => {
      mocks.getDaemonVersion.mockResolvedValue("0.13.0");
      mocks.listExternalModels.mockResolvedValue(["gpt-4o-mini", "gpt-4o"]);
      renderCard(undefined, CLOUD_GROUPS);
      await userEvent.click(await screen.findByRole("button", { name: "OpenAI" }));
      expect(screen.queryByLabelText("Endpoint URL")).not.toBeInTheDocument();

      await userEvent.type(screen.getByLabelText("API key"), "sk-proj-test-key");
      await waitFor(() => expect(screen.getByLabelText("Model")).toBeEnabled());
      await userEvent.selectOptions(screen.getByLabelText("Model"), "gpt-4o-mini");
      await userEvent.click(screen.getByRole("button", { name: "Save" }));

      await waitFor(() =>
        expect(mocks.setExternalLlm).toHaveBeenCalledWith(
          "https://api.openai.com/v1", "gpt-4o-mini", "sk-proj-test-key",
        )
      );
      expect(mocks.setApiKey).not.toHaveBeenCalled();
    });

    it("the custom chip still renders its Endpoint field — the escape hatch has no fixed endpoint to hide", async () => {
      mocks.getDaemonVersion.mockResolvedValue("0.13.0");
      renderCard(undefined, LOCAL_GROUPS);
      await userEvent.click(await screen.findByRole("button", { name: "Custom…" }));
      expect(screen.getByLabelText("Endpoint URL")).toBeInTheDocument();
    });

    it("the ollama chip still renders its Endpoint field — no regression to the local card", async () => {
      renderCard(undefined, LOCAL_GROUPS);
      expect(await screen.findByLabelText("Endpoint URL")).toBeInTheDocument();
    });

    // The Model field only carries a placeholder in its free-text fallback
    // state (dropdown-only otherwise) — force discovery to error with a key
    // present to reach that state, then check the placeholder is vendor-shaped.
    it("OpenAI's Model field placeholder is an OpenAI-shaped id, never the Ollama example", async () => {
      mocks.getDaemonVersion.mockResolvedValue("0.13.0");
      mocks.listExternalModels.mockRejectedValue(new Error("ECONNREFUSED"));
      renderCard(undefined, CLOUD_GROUPS);
      await userEvent.click(await screen.findByRole("button", { name: "OpenAI" }));
      await userEvent.type(screen.getByLabelText("API key"), "sk-proj-test-key");
      await waitFor(() => expect(screen.getByLabelText("Model").tagName).toBe("INPUT"));
      expect(screen.getByLabelText("Model")).toHaveAttribute("placeholder", "gpt-4o-mini");
      expect(screen.getByLabelText("Model")).not.toHaveAttribute("placeholder", "Model name (e.g. llama3.2)");
    });

    it("Ollama's Model field placeholder stays the generic llama3.2 example", async () => {
      mocks.listExternalModels.mockRejectedValue(new Error("ECONNREFUSED"));
      renderCard(undefined, LOCAL_GROUPS);
      await screen.findByText(/Not detected at localhost:11434/);
      expect(screen.getByLabelText("Model")).toHaveAttribute("placeholder", "Model name (e.g. llama3.2)");
    });

    // §Dropdown-only Model field (user report: "I thought even the model
    // don't need to type, only need to select from dropdown"). Before a key,
    // the control is a disabled <select> — never free text to type into;
    // once a key resolves models, it becomes the enabled, selectable
    // dropdown that already existed for local presets.
    it("a cloud vendor's Model field is a disabled dropdown before a key, and becomes selectable with discovered ids once a key is present", async () => {
      mocks.getDaemonVersion.mockResolvedValue("0.13.0");
      mocks.listExternalModels.mockResolvedValue(["gpt-4o-mini", "gpt-4o"]);
      renderCard(undefined, CLOUD_GROUPS);
      await userEvent.click(await screen.findByRole("button", { name: "OpenAI" }));
      // No key yet — a disabled dropdown with a needs-key hint, not free
      // text, and no failed-discovery noise either (covered by the dedicated
      // test below).
      const noKeyField = screen.getByLabelText("Model");
      expect(noKeyField.tagName).toBe("SELECT");
      expect(noKeyField).toBeDisabled();
      expect(screen.getByText("Add your API key to see available models")).toBeInTheDocument();

      await userEvent.type(screen.getByLabelText("API key"), "sk-proj-test-key");
      // The select stays a <select> throughout (disabled → enabled), so
      // waiting on the enabled state is enough — no element-type swap to
      // race here, unlike the old Input → Select case.
      await waitFor(() => expect(screen.getByLabelText("Model")).toBeEnabled());
      const modelField = screen.getByLabelText("Model");
      expect(modelField.tagName).toBe("SELECT");
      await userEvent.selectOptions(modelField, "gpt-4o");
      expect(modelField).toHaveValue("gpt-4o");
    });

    // Anti-brick escape hatch: if a vendor's /models call errors even with a
    // key present, dropdown-only would permanently strand that vendor — the
    // free-text fallback (with the vendor's own placeholder) must reappear.
    it("discovery ERRORS with a key present: the free-text fallback reappears with the vendor's modelPlaceholder", async () => {
      mocks.getDaemonVersion.mockResolvedValue("0.13.0");
      mocks.listExternalModels.mockRejectedValue(new Error("500 Internal Server Error"));
      renderCard(undefined, CLOUD_GROUPS);
      await userEvent.click(await screen.findByRole("button", { name: "OpenAI" }));

      await userEvent.type(screen.getByLabelText("API key"), "sk-proj-test-key");
      await waitFor(() => expect(screen.getByLabelText("Model").tagName).toBe("INPUT"));
      expect(screen.getByLabelText("Model")).toHaveAttribute("placeholder", "gpt-4o-mini");
      expect(await screen.findByText(/Couldn.t list models/)).toBeInTheDocument();
    });

    it("modelDiscoveryFailed is absent on a keyRequired preset with an empty key — no noise before a key exists", async () => {
      mocks.getDaemonVersion.mockResolvedValue("0.13.0");
      mocks.listExternalModels.mockRejectedValue(new Error("401 Unauthorized"));
      renderCard(undefined, CLOUD_GROUPS);
      await userEvent.click(await screen.findByRole("button", { name: "OpenAI" }));
      // Discovery still fires unauthenticated (unchanged) and this mock
      // always rejects — awaiting the call is the positive settle-anchor
      // before the absence check below (never assert absence "too early").
      await waitFor(() =>
        expect(mocks.listExternalModels).toHaveBeenCalledWith("https://api.openai.com/v1", null)
      );
      expect(screen.queryByText(/Couldn.t list models/)).not.toBeInTheDocument();

      // Same failing mock, but now with a key present — the guard flips and
      // the message appears, proving the earlier absence wasn't a fluke of
      // being too early.
      await userEvent.type(screen.getByLabelText("API key"), "sk-proj-test-key");
      expect(await screen.findByText(/Couldn.t list models/, {}, { timeout: 2000 })).toBeInTheDocument();
    });

    it("cloud-scoped card on a sub-0.13 daemon shows Anthropic only — never an empty chip row", async () => {
      // getDaemonVersion resolves "0.12.0" from the outer beforeEach (gate closed).
      renderCard(undefined, CLOUD_GROUPS);
      const chipRow = await screen.findByRole("group", { name: "Provider" });
      expect(within(chipRow).getByRole("button", { name: "Anthropic" })).toBeInTheDocument();
      expect(within(chipRow).getAllByRole("button")).toHaveLength(1);
    });

    it("local-scoped card shows Ollama, LM Studio, and Custom — no cloud vendors — and defaults to Ollama", async () => {
      mocks.getDaemonVersion.mockResolvedValue("0.13.0"); // gate open — scope must still exclude cloud vendors
      renderCard(undefined, LOCAL_GROUPS);
      await screen.findByText(/Connected to Ollama/);
      expect(screen.getByRole("button", { name: "Ollama" })).toHaveAttribute("aria-pressed", "true");
      expect(screen.getByRole("button", { name: "LM Studio" })).toBeInTheDocument();
      expect(screen.getByRole("button", { name: "Custom…" })).toBeInTheDocument();
      expect(screen.queryByRole("button", { name: "Anthropic" })).not.toBeInTheDocument();
      expect(screen.queryByRole("button", { name: "OpenAI" })).not.toBeInTheDocument();
    });

    it("precedence warning: shown when Anthropic is configured and a non-Anthropic chip is selected, absent when the Anthropic chip is selected", async () => {
      mocks.getDaemonVersion.mockResolvedValue("0.13.0");
      mocks.getApiKey.mockResolvedValue("sk-ant-***configured");
      renderCard(undefined, CLOUD_GROUPS);

      await userEvent.click(await screen.findByRole("button", { name: "OpenAI" }));
      expect(await screen.findByText(/Anthropic takes precedence/)).toBeInTheDocument();

      await userEvent.click(screen.getByRole("button", { name: "Anthropic" }));
      await waitFor(() => expect(screen.queryByText(/Anthropic takes precedence/)).not.toBeInTheDocument());
    });

    // Defect fix (returning user): `list_external_models` talks to the
    // provider directly and only sends a key when the frontend passes one —
    // a STORED key never reaches it. `keyConfigured` is only a presence
    // flag, so it must never be read as "we have a key to discover with".
    // Reopening Settings with a saved OpenAI key must show the saved model,
    // not brick into the free-text "couldn't list models" hatch (which used
    // to happen because a null-keyed discovery call 401'd).
    describe("returning user / stored key (no live discovery)", () => {
      it("a saved OpenAI key+model renders a disabled select showing the saved model, the stored-key hint, and no discovery-error noise", async () => {
        mocks.getDaemonVersion.mockResolvedValue("0.13.0");
        mocks.getExternalLlm.mockResolvedValue(["https://api.openai.com/v1", "gpt-4o-mini"]);
        mocks.getExternalLlmKeyConfigured.mockResolvedValue(true);
        // A null-keyed discovery call is a guaranteed 401 — this is what
        // actually happens on this path; the assertion is that it stays
        // silent, not that it doesn't happen.
        mocks.listExternalModels.mockRejectedValue(new Error("401 Unauthorized"));
        renderCard(undefined, CLOUD_GROUPS);

        // Positive settle-anchor: the saved endpoint must move the selected
        // chip off the mount-time default (Anthropic, cloud scope's first
        // preset) onto OpenAI before any absence check is meaningful.
        await waitFor(() =>
          expect(screen.getByRole("button", { name: "OpenAI" })).toHaveAttribute("aria-pressed", "true"),
        );

        const modelField = screen.getByLabelText("Model");
        expect(modelField.tagName).toBe("SELECT");
        expect(modelField).toBeDisabled();
        expect(modelField).toHaveValue("gpt-4o-mini");
        expect(screen.getByText("Re-enter your API key to choose a different model")).toBeInTheDocument();
        expect(screen.queryByText(/Couldn.t list models/)).not.toBeInTheDocument();
      });

      it("wrong-vendor: a stored OpenAI key does not carry over to the Groq chip — needsKey disabled select, no discovery-error noise", async () => {
        mocks.getDaemonVersion.mockResolvedValue("0.13.0");
        mocks.getExternalLlm.mockResolvedValue(["https://api.openai.com/v1", "gpt-4o-mini"]);
        mocks.getExternalLlmKeyConfigured.mockResolvedValue(true);
        mocks.listExternalModels.mockRejectedValue(new Error("401 Unauthorized"));
        renderCard(undefined, CLOUD_GROUPS);

        // Positive settle-anchor before switching chips.
        await waitFor(() =>
          expect(screen.getByRole("button", { name: "OpenAI" })).toHaveAttribute("aria-pressed", "true"),
        );

        await userEvent.click(screen.getByRole("button", { name: "Groq" }));

        const modelField = screen.getByLabelText("Model");
        expect(modelField.tagName).toBe("SELECT");
        expect(modelField).toBeDisabled();
        expect(screen.getByText("Add your API key to see available models")).toBeInTheDocument();
        expect(screen.queryByText("Re-enter your API key to choose a different model")).not.toBeInTheDocument();
        expect(screen.queryByText(/Couldn.t list models/)).not.toBeInTheDocument();
      });
    });
  });
});
