// SPDX-License-Identifier: AGPL-3.0-only
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { getProfileNarrative, regenerateNarrative } from "../../lib/tauri";

function timeAgo(ts: number): string {
  const now = Date.now() / 1000;
  const diff = now - ts;
  if (diff < 60) return "just now";
  if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
  if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`;
  if (diff < 604800) return `${Math.floor(diff / 86400)}d ago`;
  return new Date(ts * 1000).toLocaleDateString();
}

export default function ProfileNarrative() {
  const queryClient = useQueryClient();

  const { data: narrative, isLoading } = useQuery({
    queryKey: ["profile-narrative"],
    queryFn: getProfileNarrative,
    refetchInterval: 120000,
    staleTime: 60000,
  });

  const regenMutation = useMutation({
    mutationFn: regenerateNarrative,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["profile-narrative"] });
    },
  });

  if (isLoading) {
    return (
      <div
        className="rounded-lg px-4 py-5"
        style={{
          backgroundColor: "var(--mem-surface)",
          border: "1px solid var(--mem-border)",
        }}
      >
        <div
          className="animate-pulse h-4 rounded"
          style={{ backgroundColor: "var(--mem-hover)", width: "60%" }}
        />
      </div>
    );
  }

  if (!narrative || !narrative.content) {
    return (
      <div
        className="rounded-lg px-4 py-5 text-center"
        style={{
          backgroundColor: "var(--mem-surface)",
          border: "1px solid var(--mem-border)",
          fontFamily: "var(--mem-font-body)",
          fontSize: "13px",
          color: "var(--mem-text-tertiary)",
        }}
      >
        No narrative yet — confirm some memories to get started.
      </div>
    );
  }

  return (
    <div
      className="rounded-lg px-4 py-4"
      style={{
        backgroundColor: "var(--mem-surface)",
        border: "1px solid var(--mem-border)",
      }}
    >
      {/* Header */}
      <div className="flex items-center justify-between mb-3">
        <span
          style={{
            fontFamily: "var(--mem-font-heading)",
            fontSize: "13px",
            color: "var(--mem-text-secondary)",
            letterSpacing: "0.03em",
          }}
        >
          AI's Understanding of You
        </span>
        <button
          onClick={() => regenMutation.mutate()}
          disabled={regenMutation.isPending}
          className="flex items-center gap-1 px-2 py-1 rounded transition-colors duration-150 hover:bg-[var(--mem-hover)]"
          style={{
            fontFamily: "var(--mem-font-body)",
            fontSize: "11px",
            color: "var(--mem-text-tertiary)",
            border: "none",
            background: "none",
            cursor: "pointer",
            opacity: regenMutation.isPending ? 0.5 : 1,
          }}
        >
          <svg
            width="12"
            height="12"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            style={{
              animation: regenMutation.isPending
                ? "spin 1s linear infinite"
                : "none",
            }}
          >
            <path d="M21 2v6h-6M3 12a9 9 0 0 1 15-6.7L21 8M3 22v-6h6M21 12a9 9 0 0 1-15 6.7L3 16" />
          </svg>
          Regenerate
        </button>
      </div>

      {/* Narrative — smooth flowing paragraph */}
      <p
        style={{
          fontFamily: "var(--mem-font-body)",
          fontSize: "14px",
          lineHeight: "1.65",
          color: "var(--mem-text)",
          margin: 0,
        }}
      >
        {narrative.content}
      </p>

      {/* Metadata: updated time + source count */}
      <div
        className="flex items-center justify-between mt-3"
        style={{
          fontFamily: "var(--mem-font-mono)",
          fontSize: "10px",
          color: "var(--mem-text-tertiary)",
        }}
      >
        <span>
          Updated {timeAgo(narrative.generated_at)}
          {narrative.memory_count > 0 && ` · from ${narrative.memory_count} memories`}
        </span>
      </div>

      {/* Stale indicator */}
      {narrative.is_stale && (
        <div
          className="mt-2 text-right"
          style={{
            fontFamily: "var(--mem-font-mono)",
            fontSize: "10px",
            color: "var(--mem-text-tertiary)",
          }}
        >
          Updating...
        </div>
      )}
    </div>
  );
}
