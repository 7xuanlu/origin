// SPDX-License-Identifier: AGPL-3.0-only
import { useQuery } from "@tanstack/react-query";
import { getExternalLlm, getOnDeviceModel } from "../../lib/tauri";
import { useApiKeyStatus } from "./IntelligenceSetup";

export interface ProviderConfiguredStatus {
  /** At least one LLM provider can serve an inference right now. */
  configured: boolean;
  /** Every provider query has answered; until then `configured` is a guess. */
  isResolved: boolean;
}

/**
 * Whether an LLM provider is configured *right now*.
 *
 * Deliberately not the `intelligence-ready` onboarding milestone. That
 * milestone is a permanent latch: the daemon inserts it once, on the first
 * successful inference (`mark_llm_ready`), with `ON CONFLICT DO NOTHING`
 * (`crates/wenlan-core/src/db/onboarding_milestones.rs`), and never clears
 * it. So it stays true after a key is removed, and stays false while a
 * freshly configured provider has not yet served a request — both wrong
 * answers to "can Wenlan compile a page for me".
 *
 * The three queries below, their keys, and the derivation are the same ones
 * Settings → Intelligence builds its availability triple from
 * (`settings/sections/IntelligenceSection.tsx`), so the two surfaces read one
 * shared cache and cannot disagree about what is turned on.
 */
export function useProviderConfigured(): ProviderConfiguredStatus {
  const anthropic = useApiKeyStatus();
  const external = useQuery({
    queryKey: ["external-llm"],
    queryFn: getExternalLlm,
  });
  const onDevice = useQuery({
    queryKey: ["onDeviceModel"],
    queryFn: getOnDeviceModel,
  });

  // A saved endpoint is an external provider; an on-device model counts only
  // once the daemon reports that exact model loaded, not merely selected or
  // downloaded.
  const externalConfigured = Boolean(external.data?.[0]);
  const loadedId = onDevice.data?.loaded ?? null;
  const onDeviceLoaded =
    loadedId != null &&
    (onDevice.data?.models ?? []).some((model) => model.id === loadedId);

  return {
    configured: anthropic.isConfigured || externalConfigured || onDeviceLoaded,
    // A pending or failed query leaves the answer unknown, and an unknown
    // answer must not pick a variant: callers stay on copy that is true
    // either way until every source has actually replied.
    isResolved: anthropic.isResolved && external.isSuccess && onDevice.isSuccess,
  };
}
