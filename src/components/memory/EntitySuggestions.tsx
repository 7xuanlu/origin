// SPDX-License-Identifier: AGPL-3.0-only
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import {
  getEntitySuggestions,
  approveEntitySuggestion,
  dismissEntitySuggestion,
} from "../../lib/tauri";

interface EntitySuggestionsProps {
  onEntityCreated?: (entityId: string) => void;
}

export default function EntitySuggestions({ onEntityCreated }: EntitySuggestionsProps) {
  const queryClient = useQueryClient();

  const { data: suggestions = [] } = useQuery({
    queryKey: ["entity-suggestions"],
    queryFn: getEntitySuggestions,
    refetchInterval: 30000,
  });

  const approveMutation = useMutation({
    mutationFn: (id: string) => approveEntitySuggestion(id),
    onSuccess: (result) => {
      queryClient.invalidateQueries({ queryKey: ["entity-suggestions"] });
      queryClient.invalidateQueries({ queryKey: ["entities"] });
      queryClient.invalidateQueries({ queryKey: ["memories"] });
      onEntityCreated?.(result.entity_id);
    },
  });

  const dismissMutation = useMutation({
    mutationFn: (id: string) => dismissEntitySuggestion(id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["entity-suggestions"] });
    },
  });

  const visible = suggestions.slice(0, 5);
  if (visible.length === 0) return null;

  return (
    <div className="flex flex-col gap-1">
      <h4
        className="px-2 mb-1"
        style={{
          fontFamily: "var(--mem-font-mono)",
          fontSize: "10px",
          fontWeight: 600,
          letterSpacing: "0.05em",
          textTransform: "uppercase" as const,
          color: "var(--mem-text-tertiary)",
        }}
      >
        Suggested Entities
      </h4>
      {visible.map((s) => (
        <div
          key={s.id}
          className="flex items-center gap-2 px-2 py-1.5 rounded-md"
          style={{
            backgroundColor: "var(--mem-surface)",
            border: "1px solid var(--mem-border)",
          }}
        >
          <span
            className="flex-1 truncate"
            style={{
              fontFamily: "var(--mem-font-body)",
              fontSize: "12px",
              color: "var(--mem-text)",
            }}
          >
            {s.entity_name ?? "Unknown"}
          </span>
          <span
            style={{
              fontFamily: "var(--mem-font-mono)",
              fontSize: "10px",
              color: "var(--mem-text-tertiary)",
            }}
          >
            {s.source_ids.length}
          </span>
          <button
            onClick={() => approveMutation.mutate(s.id)}
            disabled={approveMutation.isPending}
            className="px-1.5 py-0.5 rounded text-[10px] font-medium transition-colors duration-150 hover:bg-emerald-500/20"
            style={{ color: "var(--mem-accent-sage)" }}
          >
            Create
          </button>
          <button
            onClick={() => dismissMutation.mutate(s.id)}
            disabled={dismissMutation.isPending}
            className="px-1.5 py-0.5 rounded text-[10px] font-medium transition-colors duration-150 hover:bg-zinc-500/20"
            style={{ color: "var(--mem-text-tertiary)" }}
          >
            Dismiss
          </button>
        </div>
      ))}
    </div>
  );
}
