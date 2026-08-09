// SPDX-License-Identifier: AGPL-3.0-only
import { useQuery } from "@tanstack/react-query";
import { getDaemonVersion, daemonMeetsFloor } from "../lib/tauri";

export interface DaemonVersionInfo {
  version: string | null;
  /** Daemon accepts `external_llm_api_key` / `api_key` fields (≥ 0.13). */
  supportsExternalKey: boolean;
  /** `PUT /api/config` hot-swaps the external provider (≥ 0.13). */
  supportsHotSwap: boolean;
}

/** Version gate for daemon-0.13 features (spec §8). Conservative on failure:
 *  an unreachable daemon reports both capabilities as false. */
export function useDaemonVersion(): DaemonVersionInfo {
  const { data } = useQuery({
    queryKey: ["daemon-version"],
    queryFn: getDaemonVersion,
    staleTime: 60_000,
    retry: 1,
  });
  const version = data ?? null;
  const atLeast013 = version !== null && daemonMeetsFloor(version, "0.13.0");
  return { version, supportsExternalKey: atLeast013, supportsHotSwap: atLeast013 };
}
