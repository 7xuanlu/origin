// SPDX-License-Identifier: AGPL-3.0-only
import { describe, it, expect } from "vitest";
import {
  PROVIDER_PRESETS,
  keyPrefixMismatch,
  normalizeEndpoint,
  presetForEndpoint,
  visiblePresets,
  type ProviderPreset,
} from "./providerPresets";

const byId = (id: string): ProviderPreset => {
  const p = PROVIDER_PRESETS.find((x) => x.id === id);
  if (!p) throw new Error(`preset ${id} missing`);
  return p;
};

describe("providerPresets — §9.1 data shape", () => {
  it("carries the exact per-provider key metadata", () => {
    expect(byId("openai")).toMatchObject({
      name: "OpenAI",
      keyPlaceholder: "sk-proj-...",
      keyPrefixes: ["sk-"],
      getKeyUrl: "https://platform.openai.com/api-keys",
    });
    expect(byId("gemini")).toMatchObject({
      name: "Gemini",
      keyPlaceholder: "AIzaSy... or AQ....",
      keyPrefixes: ["AIzaSy", "AQ."],
      getKeyUrl: "https://aistudio.google.com/apikey",
    });
    expect(byId("groq")).toMatchObject({
      name: "Groq",
      keyPrefixes: ["gsk_"],
      getKeyUrl: "https://console.groq.com/keys",
    });
    expect(byId("openrouter")).toMatchObject({
      name: "OpenRouter",
      keyPrefixes: ["sk-or-"],
      getKeyUrl: "https://openrouter.ai/keys",
    });
    expect(byId("deepseek")).toMatchObject({
      name: "DeepSeek",
      keyPrefixes: ["sk-"],
      getKeyUrl: "https://platform.deepseek.com/api_keys",
    });
    expect(byId("xai")).toMatchObject({
      name: "xAI (Grok)",
      keyPlaceholder: "xai-...",
      keyPrefixes: ["xai-"],
      getKeyUrl: "https://console.x.ai",
    });
  });

  it("Mistral is keyed but opaque — a getKeyUrl, no prefixes", () => {
    const mistral = byId("mistral");
    expect(mistral.name).toBe("Mistral");
    expect(mistral.getKeyUrl).toBe("https://console.mistral.ai");
    expect(mistral.keyPrefixes).toBeUndefined();
  });

  // Defect fix (user report): the old placeholder was a random-looking
  // 32-char string that read like a real (possibly leaked) credential.
  it("Mistral's key placeholder is not the old fabricated-looking token string", () => {
    expect(byId("mistral").keyPlaceholder).not.toBe("hDx3mQ7tRkP1sLb9vNc5wEa2fGz8jTy4");
  });

  it("keeps ids stable and endpoints unchanged after the renames", () => {
    expect(byId("gemini").endpoint).toBe(
      "https://generativelanguage.googleapis.com/v1beta/openai",
    );
    expect(byId("xai").endpoint).toBe("https://api.x.ai/v1");
  });
});

describe("keyPrefixMismatch — §9.1 soft validation", () => {
  it("no hint for an empty key", () => {
    expect(keyPrefixMismatch(byId("openai"), "")).toBe(false);
    expect(keyPrefixMismatch(byId("openai"), "   ")).toBe(false);
  });
  it("no hint when a prefix matches", () => {
    expect(keyPrefixMismatch(byId("openai"), "sk-proj-abc")).toBe(false);
    expect(keyPrefixMismatch(byId("xai"), "xai-abc")).toBe(false);
  });
  it("hint when no prefix matches", () => {
    expect(keyPrefixMismatch(byId("openai"), "nope-123")).toBe(true);
    expect(keyPrefixMismatch(byId("groq"), "sk-123")).toBe(true);
  });
  it("Gemini accepts either live format — neither mismatches", () => {
    expect(keyPrefixMismatch(byId("gemini"), "AIzaSyABC")).toBe(false);
    expect(keyPrefixMismatch(byId("gemini"), "AQ.abc")).toBe(false);
    expect(keyPrefixMismatch(byId("gemini"), "xyz")).toBe(true);
  });
  it("Mistral never hints (no documented prefix)", () => {
    expect(keyPrefixMismatch(byId("mistral"), "anything")).toBe(false);
  });
});

describe("normalizeEndpoint", () => {
  it("scheme-defaults and path-defaults a bare host:port", () => {
    expect(normalizeEndpoint("192.168.1.5:11434")).toBe("http://192.168.1.5:11434/v1");
  });
  it("path-defaults a scheme-prefixed host with no path", () => {
    expect(normalizeEndpoint("http://192.168.1.5:11434")).toBe("http://192.168.1.5:11434/v1");
  });
  it("leaves a fully-qualified endpoint unchanged", () => {
    expect(normalizeEndpoint("192.168.1.5:11434/v1")).toBe("http://192.168.1.5:11434/v1");
  });
  it("all three hand-typed forms of the same server converge", () => {
    const forms = ["192.168.1.5:11434", "http://192.168.1.5:11434", "192.168.1.5:11434/v1"];
    const normalized = new Set(forms.map(normalizeEndpoint));
    expect(normalized.size).toBe(1);
  });
  it("strips trailing slashes before deciding whether a path exists", () => {
    expect(normalizeEndpoint("http://localhost:11434/v1/")).toBe("http://localhost:11434/v1");
  });
  it("passes through an empty string (no endpoint typed yet)", () => {
    expect(normalizeEndpoint("")).toBe("");
    expect(normalizeEndpoint("   ")).toBe("");
  });
});

describe("presetForEndpoint — normalization applied before matching", () => {
  it("matches Ollama from a scheme-less, /v1-less hand-typed endpoint", () => {
    expect(presetForEndpoint("localhost:11434").id).toBe("ollama");
  });
  it("still falls back to custom for a genuinely different host", () => {
    expect(presetForEndpoint("192.168.1.5:11434").id).toBe("custom");
  });
});

describe("presetForEndpoint — host-alias matching (Thread #2: 127.0.0.1/[::1] == localhost)", () => {
  it("127.0.0.1:11434 matches the Ollama preset", () => {
    expect(presetForEndpoint("127.0.0.1:11434").id).toBe("ollama");
  });
  it("[::1]:11434 matches the Ollama preset", () => {
    expect(presetForEndpoint("[::1]:11434").id).toBe("ollama");
  });
  it("localhost:1234 matches the LM Studio preset", () => {
    expect(presetForEndpoint("localhost:1234").id).toBe("lmstudio");
  });
  it("192.168.1.5:11434 is a genuinely different machine — still custom, not aliased to Ollama", () => {
    expect(presetForEndpoint("192.168.1.5:11434").id).toBe("custom");
  });
  it("boundary: normalizeEndpoint does NOT rewrite 127.0.0.1 to localhost — the probed endpoint stays exactly as typed", () => {
    expect(normalizeEndpoint("127.0.0.1:11434")).toBe("http://127.0.0.1:11434/v1");
  });
});

describe("Anthropic — the native preset (unified chip row)", () => {
  it("is first in PROVIDER_PRESETS, native, cloud-grouped, with no endpoint", () => {
    expect(PROVIDER_PRESETS[0]).toMatchObject({
      id: "anthropic",
      name: "Anthropic",
      endpoint: "",
      keyRequired: true,
      group: "cloud",
      native: true,
    });
  });

  it("does not disturb the custom-preset fallback (still the last entry)", () => {
    expect(PROVIDER_PRESETS[PROVIDER_PRESETS.length - 1].id).toBe("custom");
  });

  it("presetForEndpoint never matches Anthropic's empty endpoint — an empty saved endpoint still falls back to custom", () => {
    expect(presetForEndpoint("").id).toBe("custom");
    expect(presetForEndpoint(null).id).toBe("custom");
  });
});

describe("visiblePresets — native exemption and groups scoping", () => {
  it("includes Anthropic even when supportsExternalKey is false (native bypasses the gate)", () => {
    const ids = visiblePresets(false).map((p) => p.id);
    expect(ids).toContain("anthropic");
    expect(ids).not.toContain("openai");
  });

  it("includes every keyed vendor once the gate is open", () => {
    const ids = visiblePresets(true).map((p) => p.id);
    expect(ids).toEqual(expect.arrayContaining(["anthropic", "openai", "gemini", "groq", "openrouter", "mistral", "deepseek", "xai"]));
  });

  it("a cloud-only scope excludes every local/custom preset, gate open or closed", () => {
    expect(visiblePresets(false, ["cloud"]).map((p) => p.id)).toEqual(["anthropic"]);
    const openIds = visiblePresets(true, ["cloud"]).map((p) => p.id);
    expect(openIds).not.toContain("ollama");
    expect(openIds).not.toContain("lmstudio");
    expect(openIds).not.toContain("custom");
  });

  it("a local+custom scope excludes Anthropic and every keyed cloud vendor", () => {
    const ids = visiblePresets(true, ["local", "custom"]).map((p) => p.id);
    expect(ids).toEqual(["ollama", "lmstudio", "custom"]);
  });

  it("group render order is local, then cloud, then custom, regardless of which groups are requested", () => {
    const ids = visiblePresets(true).map((p) => p.id);
    const lastLocalIdx = Math.max(ids.indexOf("ollama"), ids.indexOf("lmstudio"));
    const firstCloudIdx = ids.indexOf("anthropic");
    const customIdx = ids.indexOf("custom");
    expect(lastLocalIdx).toBeLessThan(firstCloudIdx);
    expect(customIdx).toBe(ids.length - 1);
  });

  // The Model placeholder is the only in-product hint of what a model id even
  // looks like for a given vendor, so a shared one is actively misleading —
  // "llama3.2" under OpenAI told users to type an Ollama model. Pin the
  // placeholders as present and pairwise distinct, so a future edit can't
  // quietly collapse them back to one.
  it("every keyed cloud vendor carries its own distinct model placeholder", () => {
    const vendors = PROVIDER_PRESETS.filter((p) => p.group === "cloud" && p.keyRequired && !p.native);
    expect(vendors).toHaveLength(7);

    const placeholders = vendors.map((p) => p.modelPlaceholder);
    for (const value of placeholders) {
      expect(value).toBeTruthy();
      // The generic i18n fallback is Ollama-shaped and belongs to local/custom.
      expect(value).not.toMatch(/llama3\.2/);
    }
    expect(new Set(placeholders).size).toBe(vendors.length);
  });
});
