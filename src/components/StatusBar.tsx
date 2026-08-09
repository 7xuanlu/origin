// SPDX-License-Identifier: AGPL-3.0-only
import { useIndexStatus } from "../hooks/useSearch";
import type { IndexStatus, RerankerStatus } from "../lib/tauri";

interface StatusBarProps {
  resultCount: number;
}

export default function StatusBar({
  resultCount,
}: StatusBarProps) {
  const { data: status } = useIndexStatus();
  const rerankerFailure = status ? failedRerankerReason(status) : null;

  return (
    <div className="flex items-center justify-between px-4 py-2 border-t border-[var(--separator)]/30 text-[10px] text-[var(--text-tertiary)]/70">
      <div className="flex items-center gap-3">
        {status?.is_running && (
          <span className="flex items-center gap-1.5">
            <span className="w-1 h-1 bg-green-400/80 rounded-full animate-pulse" />
            Indexing
          </span>
        )}
        {status && !status.is_running && (
          <span>{status.files_indexed} indexed</span>
        )}
        {status?.last_error && (
          <span className="text-red-400/80 truncate" title={status.last_error}>
            {status.last_error}
          </span>
        )}
        {rerankerFailure && (
          <span className="text-red-400/80 truncate" title={rerankerFailure}>
            Reranker failed
          </span>
        )}
        {resultCount > 0 && <span>{resultCount} results</span>}
      </div>
      <div className="flex items-center gap-2">
        <span>
          ↑↓ navigate · ↵ open · ⌘C copy · ⌘↵ all · esc close
        </span>
      </div>
    </div>
  );
}

function failedRerankerReason(status: IndexStatus): string | null {
  for (const reranker of [status.reranker, status.reranker_light]) {
    const reason = failedReason(reranker);
    if (reason) return reason;
  }
  return null;
}

function failedReason(status: RerankerStatus): string | null {
  return status.state === "failed" ? status.reason : null;
}
