// SPDX-License-Identifier: AGPL-3.0-only

export type PresetGroup = "cloud" | "local" | "custom";

/** Name for the native Anthropic preset entry below. Kept as its own export
 *  (rather than inlined) so other call sites can reference the proper noun
 *  without importing the whole preset table — same convention as every
 *  other ProviderPreset.name, none of which are translated. */
export const ANTHROPIC_VENDOR_NAME = "Anthropic";

export interface ProviderPreset {
  id: string;
  /** Vendor names are proper nouns — not translated. */
  name: string;
  endpoint: string;
  keyRequired: boolean;
  group: PresetGroup;
  /** True for the Anthropic preset: not an OpenAI-compatible endpoint, no
   *  external-llm slot involved — it's handled by the pre-existing native
   *  Anthropic key path (setApiKey/getApiKey). A native preset is never
   *  gated by the daemon's external-key floor, and AnyProviderCard renders
   *  it as AnthropicFields instead of the generic endpoint/model/key form. */
  native?: boolean;
  /** Provider-shaped example shown when no key is stored (§9.1). */
  keyPlaceholder?: string;
  /** Vendor-specific example model id shown in the Model field before
   *  discovery has anything to offer. A proper noun / id, like keyPlaceholder
   *  — never translated. Absent = falls back to the generic (Ollama-shaped)
   *  i18n placeholder, correct for the local/custom presets that have no
   *  fixed vendor. */
  modelPlaceholder?: string;
  /** Soft-check prefixes; absent = no format check (§9.1). */
  keyPrefixes?: string[];
  /** Provider console, opened in the system browser (§9.1). */
  getKeyUrl?: string;
}

// Spec §1 preset table. Every keyed preset's /models + /chat/completions
// compatibility is validated live during implementation (council dissent
// note); a preset whose /models shape drifts ships free-text-only (the card
// already falls back to free text when discovery fails, so no code change —
// just note the finding in the PR description).
export const PROVIDER_PRESETS: ProviderPreset[] = [
  {
    id: "anthropic",
    name: ANTHROPIC_VENDOR_NAME,
    // No external-llm endpoint — the native Anthropic key path has none.
    // presetForEndpoint skips empty-endpoint presets when matching a saved
    // endpoint back to a preset, and its custom-preset fallback depends on
    // PROVIDER_PRESETS[length-1] staying "custom" — inserting Anthropic
    // here, at index 0, preserves both.
    endpoint: "",
    keyRequired: true,
    group: "cloud",
    native: true,
    keyPlaceholder: "sk-ant-api03-...",
    keyPrefixes: ["sk-ant-"],
    getKeyUrl: "https://console.anthropic.com/settings/keys",
  },
  {
    id: "openai",
    name: "OpenAI",
    endpoint: "https://api.openai.com/v1",
    keyRequired: true,
    group: "cloud",
    keyPlaceholder: "sk-proj-...",
    keyPrefixes: ["sk-"],
    getKeyUrl: "https://platform.openai.com/api-keys",
    modelPlaceholder: "gpt-4o-mini",
  },
  {
    id: "gemini",
    name: "Gemini",
    endpoint: "https://generativelanguage.googleapis.com/v1beta/openai",
    keyRequired: true,
    group: "cloud",
    keyPlaceholder: "AIzaSy... or AQ....",
    keyPrefixes: ["AIzaSy", "AQ."],
    getKeyUrl: "https://aistudio.google.com/apikey",
    modelPlaceholder: "gemini-2.0-flash",
  },
  {
    id: "groq",
    name: "Groq",
    endpoint: "https://api.groq.com/openai/v1",
    keyRequired: true,
    group: "cloud",
    keyPlaceholder: "gsk_...",
    keyPrefixes: ["gsk_"],
    getKeyUrl: "https://console.groq.com/keys",
    modelPlaceholder: "llama-3.3-70b-versatile",
  },
  {
    id: "openrouter",
    name: "OpenRouter",
    endpoint: "https://openrouter.ai/api/v1",
    keyRequired: true,
    group: "cloud",
    keyPlaceholder: "sk-or-v1-...",
    keyPrefixes: ["sk-or-"],
    getKeyUrl: "https://openrouter.ai/keys",
    modelPlaceholder: "anthropic/claude-3.5-sonnet",
  },
  {
    id: "mistral",
    name: "Mistral",
    endpoint: "https://api.mistral.ai/v1",
    keyRequired: true,
    group: "cloud",
    // Mistral keys are an opaque token with no fixed prefix (unlike sk-,
    // gsk_, xai-, sk-or-) — keyPrefixes intentionally stays unset (see
    // keyPrefixMismatch's doc comment, which names Mistral as the example).
    // This placeholder still shows the token's actual shape/length, using
    // repeated "x" rather than a random-looking string so it can't be
    // mistaken for a real (possibly leaked) credential.
    keyPlaceholder: "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
    getKeyUrl: "https://console.mistral.ai",
    modelPlaceholder: "mistral-large-latest",
  },
  {
    id: "deepseek",
    name: "DeepSeek",
    endpoint: "https://api.deepseek.com/v1",
    keyRequired: true,
    group: "cloud",
    keyPlaceholder: "sk-...",
    keyPrefixes: ["sk-"],
    getKeyUrl: "https://platform.deepseek.com/api_keys",
    modelPlaceholder: "deepseek-chat",
  },
  {
    id: "xai",
    name: "xAI (Grok)",
    endpoint: "https://api.x.ai/v1",
    keyRequired: true,
    group: "cloud",
    keyPlaceholder: "xai-...",
    keyPrefixes: ["xai-"],
    getKeyUrl: "https://console.x.ai",
    modelPlaceholder: "grok-2-latest",
  },
  { id: "ollama", name: "Ollama (local)", endpoint: "http://localhost:11434/v1", keyRequired: false, group: "local" },
  { id: "lmstudio", name: "LM Studio (local)", endpoint: "http://localhost:1234/v1", keyRequired: false, group: "local" },
  { id: "custom", name: "Custom…", endpoint: "", keyRequired: false, group: "custom" },
];

/** Normalize a hand-typed endpoint so equivalent forms resolve to the same
 *  probe target: default the scheme to http:// and the path to /v1 when
 *  absent. No further host validation — malformed input still falls
 *  through to free text/no-match, same as before this existed. */
export function normalizeEndpoint(raw: string): string {
  let ep = raw.trim().replace(/\/+$/, "");
  if (ep === "") return ep;
  if (!/^https?:\/\//.test(ep)) ep = `http://${ep}`;
  if (!/^https?:\/\/[^/]+\/.+/.test(ep)) ep = `${ep}/v1`;
  return ep;
}

const ENDPOINT_HOST_RE = /^(https?:\/\/)(\[[^\]]+\]|[^/:]+)(.*)$/;

/** localhost, 127.0.0.1, and [::1] are the same host for PRESET MATCHING
 *  only (Thread #2). Never used to rewrite the endpoint that actually gets
 *  probed — a server can bind IPv4-only while `localhost` resolves to ::1,
 *  so rewriting a user's typed host could break a probe that would
 *  otherwise have worked. See `endpointsMatch`. */
export function canonicalHostForMatch(host: string): string {
  const bare = host.startsWith("[") && host.endsWith("]") ? host.slice(1, -1) : host;
  return bare === "127.0.0.1" || bare === "::1" ? "localhost" : host;
}

/** True when two already-normalized endpoints denote the same server for
 *  preset-matching purposes, treating localhost/127.0.0.1/[::1] as one
 *  host. Match-only: does not affect what gets fetched. */
export function endpointsMatch(a: string, b: string): boolean {
  const canon = (ep: string) => {
    const m = ENDPOINT_HOST_RE.exec(ep);
    if (!m) return ep;
    const [, scheme, host, rest] = m;
    return `${scheme}${canonicalHostForMatch(host)}${rest}`;
  };
  return canon(a) === canon(b);
}

/** Match a saved endpoint back to its preset ("custom" when no match). */
export function presetForEndpoint(endpoint: string | null): ProviderPreset {
  const norm = normalizeEndpoint(endpoint ?? "");
  return (
    PROVIDER_PRESETS.find((p) => p.endpoint !== "" && endpointsMatch(normalizeEndpoint(p.endpoint), norm)) ??
    PROVIDER_PRESETS[PROVIDER_PRESETS.length - 1]
  );
}

// Group render order for the widened (§5.2a) preset picker: local servers
// first, then cloud vendors, then the free-text "Custom…" escape hatch last.
// Within "cloud", Anthropic renders first because it's PROVIDER_PRESETS[0].
const VISIBLE_GROUP_ORDER: PresetGroup[] = ["local", "cloud", "custom"];

/** The presets AnyProviderCard should render. Anthropic (native) is never
 *  gated — it works on every daemon version via the pre-existing native key
 *  path, so the closed-gate result set is never empty even on a sub-0.13
 *  daemon. Below the daemon-0.13 key-auth floor, the other 7 keyed cloud
 *  vendors are excluded (they need a real external-key slot to
 *  authenticate); at/above the floor every preset is eligible.
 *
 *  An optional `groups` further scopes the result to a subset of
 *  PresetGroup — e.g. a wizard pane that must show only cloud vendors, or
 *  only local servers. Final order is always local-first / cloud-second /
 *  custom-last, regardless of which groups are included. */
export function visiblePresets(
  supportsExternalKey: boolean,
  groups?: PresetGroup[],
): ProviderPreset[] {
  const gated = PROVIDER_PRESETS.filter((p) => p.native || !p.keyRequired || supportsExternalKey);
  const scoped = groups ? gated.filter((p) => groups.includes(p.group)) : gated;
  return VISIBLE_GROUP_ORDER.flatMap((group) => scoped.filter((p) => p.group === group));
}

/** Soft key-format check (§9.1): true only when a non-empty key matches none
 *  of the preset's prefixes. Presets without prefixes (e.g. Mistral) never
 *  mismatch. Never used to block Save/Test — hint only. */
export function keyPrefixMismatch(preset: ProviderPreset, key: string): boolean {
  const k = key.trim();
  if (k === "" || !preset.keyPrefixes || preset.keyPrefixes.length === 0) return false;
  return !preset.keyPrefixes.some((prefix) => k.startsWith(prefix));
}
