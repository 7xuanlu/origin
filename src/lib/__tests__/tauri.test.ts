import { describe, it, expect, vi, beforeEach } from "vitest";

vi.mock("@tauri-apps/api/core", () => ({
  invoke: vi.fn(),
}));

describe("sources, page export, and knowledge wrappers", () => {
  beforeEach(() => {
    vi.resetAllMocks();
  });

  it("listRegisteredSources calls list_registered_sources", async () => {
    const { invoke } = await import("@tauri-apps/api/core");
    const { listRegisteredSources } = await import("../tauri");
    (invoke as ReturnType<typeof vi.fn>).mockResolvedValue([]);
    await listRegisteredSources();
    expect(invoke).toHaveBeenCalledWith("list_registered_sources");
  });

  it("addSource passes sourceType and path", async () => {
    const { invoke } = await import("@tauri-apps/api/core");
    const { addSource } = await import("../tauri");
    (invoke as ReturnType<typeof vi.fn>).mockResolvedValue({
      id: "obsidian-vault",
      source_type: "obsidian",
      path: "/vault",
      status: "Active",
      last_sync: null,
      file_count: 0,
      memory_count: 0,
    });
    await addSource("obsidian", "/vault");
    expect(invoke).toHaveBeenCalledWith("add_source", {
      sourceType: "obsidian",
      path: "/vault",
    });
  });

  it("removeSource passes id", async () => {
    const { invoke } = await import("@tauri-apps/api/core");
    const { removeSource } = await import("../tauri");
    (invoke as ReturnType<typeof vi.fn>).mockResolvedValue(undefined);
    await removeSource("obsidian-vault");
    expect(invoke).toHaveBeenCalledWith("remove_source", {
      id: "obsidian-vault",
    });
  });

  it("syncRegisteredSource passes id", async () => {
    const { invoke } = await import("@tauri-apps/api/core");
    const { syncRegisteredSource } = await import("../tauri");
    (invoke as ReturnType<typeof vi.fn>).mockResolvedValue({
      files_found: 10,
      ingested: 5,
      skipped: 5,
      errors: 0,
    });
    await syncRegisteredSource("obsidian-vault");
    expect(invoke).toHaveBeenCalledWith("sync_registered_source", {
      id: "obsidian-vault",
    });
  });

  it("exportPageToObsidian passes pageId and vaultPath", async () => {
    const { invoke } = await import("@tauri-apps/api/core");
    const { exportPageToObsidian } = await import("../tauri");
    (invoke as ReturnType<typeof vi.fn>).mockResolvedValue({
      path: "/path/to/file.md",
    });
    const result = await exportPageToObsidian("c123", "/vault");
    expect(result).toEqual({ path: "/path/to/file.md" });
    expect(invoke).toHaveBeenCalledWith("export_page_to_obsidian", {
      pageId: "c123",
      vaultPath: "/vault",
    });
  });

  it("testExternalLlm preserves the daemon response envelope", async () => {
    const { invoke } = await import("@tauri-apps/api/core");
    const { testExternalLlm } = await import("../tauri");
    (invoke as ReturnType<typeof vi.fn>).mockResolvedValue({
      response: "hello",
    });

    const result = await testExternalLlm("http://localhost:11434/v1", "qwen3");

    expect(result.response).toBe("hello");
    expect(invoke).toHaveBeenCalledWith("test_external_llm", {
      endpoint: "http://localhost:11434/v1",
      model: "qwen3",
      apiKey: null,
    });
  });

  it("testExternalLlm passes apiKey through when supplied", async () => {
    const { invoke } = await import("@tauri-apps/api/core");
    const { testExternalLlm } = await import("../tauri");
    (invoke as ReturnType<typeof vi.fn>).mockResolvedValue({
      response: "hello",
    });

    await testExternalLlm("http://localhost:11434/v1", "qwen3", "sk-secret");

    expect(invoke).toHaveBeenCalledWith("test_external_llm", {
      endpoint: "http://localhost:11434/v1",
      model: "qwen3",
      apiKey: "sk-secret",
    });
  });

  it("setExternalLlm preserves the stored key when apiKey is omitted", async () => {
    const { invoke } = await import("@tauri-apps/api/core");
    const { setExternalLlm } = await import("../tauri");
    (invoke as ReturnType<typeof vi.fn>).mockResolvedValue(undefined);

    await setExternalLlm("http://localhost:11434/v1", "qwen3");

    expect(invoke).toHaveBeenCalledWith("set_external_llm", {
      endpoint: "http://localhost:11434/v1",
      model: "qwen3",
      apiKey: null,
    });
  });

  it("setExternalLlm clears the stored key with an empty string", async () => {
    const { invoke } = await import("@tauri-apps/api/core");
    const { setExternalLlm } = await import("../tauri");
    (invoke as ReturnType<typeof vi.fn>).mockResolvedValue(undefined);

    await setExternalLlm(null, null, "");

    expect(invoke).toHaveBeenCalledWith("set_external_llm", {
      endpoint: null,
      model: null,
      apiKey: "",
    });
  });

  it("setExternalLlm replaces the stored key when apiKey is supplied", async () => {
    const { invoke } = await import("@tauri-apps/api/core");
    const { setExternalLlm } = await import("../tauri");
    (invoke as ReturnType<typeof vi.fn>).mockResolvedValue(undefined);

    await setExternalLlm(null, null, "sk-secret");

    expect(invoke).toHaveBeenCalledWith("set_external_llm", {
      endpoint: null,
      model: null,
      apiKey: "sk-secret",
    });
  });

  it("getExternalLlmKeyConfigured calls get_external_llm_key_configured", async () => {
    const { invoke } = await import("@tauri-apps/api/core");
    const { getExternalLlmKeyConfigured } = await import("../tauri");
    (invoke as ReturnType<typeof vi.fn>).mockResolvedValue(true);

    const result = await getExternalLlmKeyConfigured();

    expect(result).toBe(true);
    expect(invoke).toHaveBeenCalledWith("get_external_llm_key_configured");
  });

  it("getOnDeviceModel calls get_on_device_model and returns model state", async () => {
    const { invoke } = await import("@tauri-apps/api/core");
    const { getOnDeviceModel } = await import("../tauri");
    (invoke as ReturnType<typeof vi.fn>).mockResolvedValue({
      loaded: null,
      selected: "qwen3-4b-instruct-2507",
      models: [
        {
          id: "qwen3-4b-instruct-2507",
          display_name: "Qwen3 4B",
          param_count: "4B",
          ram_required_gb: 8,
          file_size_gb: 2.7,
          cached: false,
        },
      ],
    });

    const result = await getOnDeviceModel();

    expect(result.models[0].id).toBe("qwen3-4b-instruct-2507");
    expect(invoke).toHaveBeenCalledWith("get_on_device_model");
  });

  it("downloadOnDeviceModel passes modelId", async () => {
    const { invoke } = await import("@tauri-apps/api/core");
    const { downloadOnDeviceModel } = await import("../tauri");
    (invoke as ReturnType<typeof vi.fn>).mockResolvedValue(undefined);

    await downloadOnDeviceModel("qwen3-4b-instruct-2507");

    expect(invoke).toHaveBeenCalledWith("download_on_device_model", {
      modelId: "qwen3-4b-instruct-2507",
    });
  });

  it("getKnowledgePath calls get_knowledge_path", async () => {
    const { invoke } = await import("@tauri-apps/api/core");
    const { getKnowledgePath } = await import("../tauri");
    (invoke as ReturnType<typeof vi.fn>).mockResolvedValue(
      "/home/user/Origin/knowledge",
    );
    await getKnowledgePath();
    expect(invoke).toHaveBeenCalledWith("get_knowledge_path");
  });

  it("countKnowledgeFiles calls count_knowledge_files", async () => {
    const { invoke } = await import("@tauri-apps/api/core");
    const { countKnowledgeFiles } = await import("../tauri");
    (invoke as ReturnType<typeof vi.fn>).mockResolvedValue(12);
    await countKnowledgeFiles();
    expect(invoke).toHaveBeenCalledWith("count_knowledge_files");
  });

  it("quitWenlanFull calls the Wenlan lifecycle command", async () => {
    const { invoke } = await import("@tauri-apps/api/core");
    const { quitWenlanFull } = await import("../tauri");
    (invoke as ReturnType<typeof vi.fn>).mockResolvedValue(undefined);
    await quitWenlanFull();
    expect(invoke).toHaveBeenCalledWith("quit_wenlan_full");
  });

  it("cancelGuardedQuitRequest synchronously clears the native quit guard", async () => {
    const { invoke } = await import("@tauri-apps/api/core");
    const { cancelGuardedQuitRequest } = await import("../tauri");
    (invoke as ReturnType<typeof vi.fn>).mockResolvedValue(undefined);
    await cancelGuardedQuitRequest();
    expect(invoke).toHaveBeenCalledWith("cancel_guarded_quit_request");
  });

  it("quitOriginFull remains a legacy alias", async () => {
    const { invoke } = await import("@tauri-apps/api/core");
    const { quitOriginFull } = await import("../tauri");
    (invoke as ReturnType<typeof vi.fn>).mockResolvedValue(undefined);
    await quitOriginFull();
    expect(invoke).toHaveBeenCalledWith("quit_origin_full");
  });

  it("getWenlanMcpEntry calls the typed MCP entry command", async () => {
    const { invoke } = await import("@tauri-apps/api/core");
    const { getWenlanMcpEntry } = await import("../tauri");
    (invoke as ReturnType<typeof vi.fn>).mockResolvedValue({
      command: "npx",
      args: ["-y", "wenlan-mcp"],
    });

    const result = await getWenlanMcpEntry();

    expect(result.command).toBe("npx");
    expect(result.args).toEqual(["-y", "wenlan-mcp"]);
    expect(invoke).toHaveBeenCalledWith("get_wenlan_mcp_entry");
  });
});
