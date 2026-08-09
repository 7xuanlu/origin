// SPDX-License-Identifier: AGPL-3.0-only
import { useEffect } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { listen, type UnlistenFn } from "@tauri-apps/api/event";
import {
  listOnboardingMilestones,
  acknowledgeOnboardingMilestone,
  type MilestoneId,
  type MilestoneRecord,
} from "../../lib/tauri";

const QUERY_KEY = ["onboarding-milestones"] as const;

export function useMilestones() {
  const qc = useQueryClient();

  const query = useQuery({
    queryKey: QUERY_KEY,
    queryFn: listOnboardingMilestones,
    staleTime: 30_000,
    refetchInterval: 5_000,
  });

  const ack = useMutation({
    mutationFn: (id: MilestoneId) => acknowledgeOnboardingMilestone(id),
    onSuccess: () => qc.invalidateQueries({ queryKey: QUERY_KEY }),
  });

  // Subscribe to backend-emitted milestone events to refresh the cache.
  useEffect(() => {
    let un: UnlistenFn | undefined;
    listen<MilestoneRecord>("onboarding-milestone", () => {
      qc.invalidateQueries({ queryKey: QUERY_KEY });
    }).then((f) => {
      un = f;
    });
    return () => {
      if (un) un();
    };
  }, [qc]);

  return {
    milestones: query.data ?? [],
    isLoading: query.isLoading,
    acknowledge: (id: MilestoneId) => ack.mutate(id),
  };
}
