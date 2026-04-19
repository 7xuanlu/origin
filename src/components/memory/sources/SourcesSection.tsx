import { useState, useMemo, useCallback, useEffect } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import {
  listRegisteredSources,
  syncRegisteredSource,
  removeSource,
  getKnowledgePath,
  countKnowledgeFiles,
  openFile,
  type RegisteredSource,
} from "../../../lib/tauri";
import AddSourceDialog from "./AddSourceDialog";

function shortenPath(p: string): string {
  const home = "~";
  // Replace /Users/<user> prefix with ~
  return p.replace(/^\/Users\/[^/]+/, home);
}

function folderName(p: string): string {
  return p.split("/").filter(Boolean).pop() || p;
}

function relativeTime(ts: number | null): string {
  if (!ts) return "Never synced";
  const diff = Math.floor(Date.now() / 1000 - ts);
  if (diff < 60) return "Last synced just now";
  if (diff < 3600) return `Last synced ${Math.floor(diff / 60)}m ago`;
  if (diff < 86400) return `Last synced ${Math.floor(diff / 3600)}h ago`;
  return `Last synced ${Math.floor(diff / 86400)}d ago`;
}

export default function SourcesSection() {
  const queryClient = useQueryClient();
  const [dialogOpen, setDialogOpen] = useState(false);
  const [confirmRemove, setConfirmRemove] = useState<RegisteredSource | null>(null);
  const [syncingId, setSyncingId] = useState<string | null>(null);
  const [justSyncedId, setJustSyncedId] = useState<string | null>(null);
  const [menuOpenId, setMenuOpenId] = useState<string | null>(null);

  const { data: sources = [] } = useQuery({
    queryKey: ["registeredSources"],
    queryFn: listRegisteredSources,
    refetchInterval: 10_000,
  });

  const obsidianSources = useMemo(
    () => sources.filter((s) => s.source_type === "obsidian"),
    [sources],
  );

  const { data: knowledgePath } = useQuery({
    queryKey: ["knowledgePath"],
    queryFn: getKnowledgePath,
  });

  const { data: knowledgeCount } = useQuery({
    queryKey: ["knowledgeCount"],
    queryFn: countKnowledgeFiles,
  });

  const syncMutation = useMutation({
    mutationFn: (id: string) => syncRegisteredSource(id),
    onMutate: (id) => setSyncingId(id),
    onSuccess: (_data, id) => {
      setSyncingId(null);
      setJustSyncedId(id);
      setTimeout(() => setJustSyncedId(null), 2000);
      queryClient.invalidateQueries({ queryKey: ["registeredSources"] });
    },
    onError: () => setSyncingId(null),
  });

  const removeMutation = useMutation({
    mutationFn: (id: string) => removeSource(id),
    onSuccess: () => {
      setConfirmRemove(null);
      queryClient.invalidateQueries({ queryKey: ["registeredSources"] });
    },
  });

  const handleReveal = useCallback((path: string) => {
    openFile(path);
    setMenuOpenId(null);
  }, []);

  // Close the kebab menu on outside click or Escape. The listeners are only
  // attached while a menu is open, and the mousedown target is matched against
  // `data-menu-id` on the menu container so clicks on the trigger button
  // itself (which is inside the container) don't close-then-reopen the menu.
  useEffect(() => {
    if (!menuOpenId) return;
    const handleMouseDown = (e: MouseEvent) => {
      const target = e.target as HTMLElement | null;
      if (!target?.closest(`[data-menu-id="${menuOpenId}"]`)) {
        setMenuOpenId(null);
      }
    };
    const handleKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setMenuOpenId(null);
    };
    document.addEventListener("mousedown", handleMouseDown);
    document.addEventListener("keydown", handleKey);
    return () => {
      document.removeEventListener("mousedown", handleMouseDown);
      document.removeEventListener("keydown", handleKey);
    };
  }, [menuOpenId]);

  // Escape closes the Remove-source confirmation dialog. The dialog has its
  // own Cancel button, but keyboard dismiss is the expected pattern for
  // modal dialogs across the app.
  useEffect(() => {
    if (!confirmRemove) return;
    const handleKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setConfirmRemove(null);
    };
    document.addEventListener("keydown", handleKey);
    return () => document.removeEventListener("keydown", handleKey);
  }, [confirmRemove]);

  return (
    <section className="space-y-3">
      {obsidianSources.length === 0 ? (
        <div className="rounded-lg border border-dashed border-[var(--mem-border)] p-8 text-center">
          <p className="text-sm text-[var(--mem-text-secondary)] mb-3">
            No sources yet
          </p>
          <button
            onClick={() => setDialogOpen(true)}
            className="rounded-md bg-[var(--mem-accent-indigo)] px-4 py-2 text-sm text-white hover:opacity-90"
          >
            Add your first source
          </button>
        </div>
      ) : (
        <div className="space-y-2">
          {obsidianSources.map((source) => {
            const isSyncing = syncingId === source.id;
            const justSynced = justSyncedId === source.id;
            const errorCount = source.last_sync_errors ?? 0;
            const hasErrors = errorCount > 0 && !isSyncing;
            return (
              <div
                key={source.id}
                className="rounded-lg border border-[var(--mem-border)] bg-[var(--mem-surface)]"
              >
                <div className="flex items-center justify-between px-4 py-3">
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2">
                      <span className="text-sm font-medium text-[var(--mem-text)] truncate">
                        {folderName(source.path)}
                      </span>
                    </div>
                    <p className="text-xs text-[var(--mem-text-secondary)] mt-0.5">
                      {isSyncing
                        ? "Syncing…"
                        : `${source.file_count.toLocaleString()} files · ${source.memory_count.toLocaleString()} memories · ${relativeTime(source.last_sync)}`}
                    </p>
                  </div>

                  <div className="flex items-center gap-2 ml-3">
                    <button
                      onClick={() => syncMutation.mutate(source.id)}
                      disabled={isSyncing}
                      className={`rounded-md px-3 py-1.5 text-xs font-medium transition-colors ${
                        justSynced
                          ? "bg-emerald-50 text-emerald-700 dark:bg-emerald-900/30 dark:text-emerald-400"
                          : isSyncing
                            ? "bg-[var(--mem-accent-indigo)]/10 text-[var(--mem-accent-indigo)]"
                            : "border border-[var(--mem-border)] text-[var(--mem-text-secondary)] hover:border-[var(--mem-accent-indigo)] hover:text-[var(--mem-accent-indigo)]"
                      }`}
                    >
                      {justSynced ? "✓ Synced" : isSyncing ? "↻ Syncing…" : "↻ Sync"}
                    </button>

                    <div className="relative" data-menu-id={source.id}>
                      <button
                        onClick={() =>
                          setMenuOpenId(menuOpenId === source.id ? null : source.id)
                        }
                        className="rounded p-1 text-[var(--mem-text-secondary)] hover:bg-[var(--mem-hover-strong)]"
                        aria-label="More"
                        aria-haspopup="menu"
                        aria-expanded={menuOpenId === source.id}
                      >
                        ⋯
                      </button>
                      {menuOpenId === source.id && (
                        <div
                          role="menu"
                          className="absolute right-0 top-full mt-1 z-10 w-44 rounded-md border border-[var(--mem-border)] bg-[var(--mem-surface)] shadow-lg py-1"
                        >
                          <button
                            role="menuitem"
                            onClick={() => handleReveal(source.path)}
                            className="w-full px-3 py-1.5 text-left text-xs text-[var(--mem-text)] hover:bg-[var(--mem-hover)]"
                          >
                            Reveal in Finder
                          </button>
                          <button
                            role="menuitem"
                            onClick={() => {
                              setConfirmRemove(source);
                              setMenuOpenId(null);
                            }}
                            className="w-full px-3 py-1.5 text-left text-xs text-red-500 hover:bg-[var(--mem-hover)]"
                          >
                            Remove source
                          </button>
                        </div>
                      )}
                    </div>
                  </div>
                </div>

                {hasErrors && (
                  <SourceErrorCallout
                    source={source}
                    errorCount={errorCount}
                    isSyncing={isSyncing}
                    onRetry={() => syncMutation.mutate(source.id)}
                    onReveal={() => handleReveal(source.path)}
                  />
                )}
              </div>
            );
          })}
          <button
            onClick={() => setDialogOpen(true)}
            className="w-full rounded-lg border border-dashed border-[var(--mem-border)] py-2 text-xs text-[var(--mem-text-tertiary)] hover:border-[var(--mem-accent-indigo)] hover:text-[var(--mem-accent-indigo)] transition-colors"
          >
            + Add source…
          </button>
        </div>
      )}

      {/* Knowledge Directory block */}
      {knowledgePath && (
        <div className="rounded-lg border border-[var(--mem-border)] bg-[var(--mem-surface)] px-4 py-3 mt-4">
          <h4 className="text-xs font-medium text-[var(--mem-text-secondary)] mb-1">
            Knowledge Directory
          </h4>
          <p className="text-sm text-[var(--mem-text)]">
            {shortenPath(knowledgePath)}
          </p>
          <p className="text-xs text-[var(--mem-text-secondary)] mt-0.5">
            {knowledgeCount !== undefined
              ? `${knowledgeCount} concept files`
              : "Loading…"}{" "}
            · updates automatically
          </p>
          <button
            onClick={() => openFile(knowledgePath)}
            className="mt-2 text-xs text-[var(--mem-accent-indigo)] hover:underline"
          >
            Reveal in Finder
          </button>
        </div>
      )}

      {/* Add Source Dialog */}
      {dialogOpen && (
        <AddSourceDialog
          onClose={() => setDialogOpen(false)}
          onSuccess={() => {
            setDialogOpen(false);
            queryClient.invalidateQueries({ queryKey: ["registeredSources"] });
          }}
        />
      )}

      {/* Remove Confirmation Dialog */}
      {confirmRemove && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50">
          <div className="w-96 rounded-lg bg-[var(--mem-surface)] p-6 shadow-xl">
            <h3 className="text-sm font-medium text-[var(--mem-text)] mb-2">
              Remove &ldquo;{folderName(confirmRemove.path)}&rdquo;?
            </h3>
            <p className="text-xs text-[var(--mem-text-secondary)] mb-4">
              {confirmRemove.file_count.toLocaleString()} files ·{" "}
              {confirmRemove.memory_count.toLocaleString()} ingested memories will
              remain in your Origin library but stop updating.
            </p>
            <div className="flex justify-end gap-2">
              <button
                onClick={() => setConfirmRemove(null)}
                className="rounded-md px-3 py-1.5 text-xs text-[var(--mem-text-secondary)] border border-[var(--mem-border)] hover:bg-[var(--mem-hover)]"
              >
                Cancel
              </button>
              <button
                onClick={() => removeMutation.mutate(confirmRemove.id)}
                disabled={removeMutation.isPending}
                className="rounded-md bg-red-500 px-3 py-1.5 text-xs text-white hover:bg-red-600"
              >
                {removeMutation.isPending ? "Removing…" : "Remove source"}
              </button>
            </div>
          </div>
        </div>
      )}
    </section>
  );
}

// ── Error callout ─────────────────────────────────────────────────────
// Shown below a source row when last_sync_errors > 0. Uses Origin's warm
// accent as the left-border signal + a subtle tinted background, with a
// friendly headline, detail message, and two actions (retry + reveal).

interface SourceErrorCalloutProps {
  source: RegisteredSource;
  errorCount: number;
  isSyncing: boolean;
  onRetry: () => void;
  onReveal: () => void;
}

function SourceErrorCallout({
  source,
  errorCount,
  isSyncing,
  onRetry,
  onReveal,
}: SourceErrorCalloutProps) {
  const isGoogleDrive = source.last_sync_error_detail === "google_drive_offline";
  const headline =
    errorCount === 1
      ? "1 file couldn't be read"
      : `${errorCount.toLocaleString()} files couldn't be read`;
  const detail = isGoogleDrive
    ? "These files are stored online in Google Drive. Right-click the folder in Finder and choose “Available offline” so Origin can index them."
    : "Origin couldn't open these files. Check that the folder is accessible and not locked by another application, then retry.";

  return (
    <div
      className="border-t border-l-[3px] px-4 py-3 flex items-start gap-3 rounded-b-lg"
      style={{
        // Set each side explicitly — using `borderColor` shorthand here
        // would also color the (zero-width) right and bottom sides with
        // --mem-border, which would be wrong if a later change adds widths
        // to those sides.
        borderTopColor: "var(--mem-border)",
        borderLeftColor: "var(--mem-accent-warm)",
        backgroundColor: "rgba(212, 136, 74, 0.06)",
      }}
    >
      <svg
        className="w-4 h-4 shrink-0 mt-0.5"
        fill="none"
        stroke="currentColor"
        strokeWidth={2}
        viewBox="0 0 24 24"
        style={{ color: "var(--mem-accent-warm)" }}
        aria-hidden="true"
      >
        <path
          strokeLinecap="round"
          strokeLinejoin="round"
          d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"
        />
      </svg>
      <div className="flex-1 min-w-0">
        <p
          className="text-xs font-medium mb-1"
          style={{ color: "var(--mem-text)" }}
        >
          {headline}
        </p>
        <p
          className="text-xs leading-relaxed mb-2.5"
          style={{ color: "var(--mem-text-secondary)" }}
        >
          {detail}
        </p>
        <div className="flex items-center gap-2">
          <button
            onClick={onRetry}
            disabled={isSyncing}
            className="px-2.5 py-1 rounded-md text-xs font-medium border transition-colors disabled:opacity-60 disabled:cursor-not-allowed"
            style={{
              borderColor: "var(--mem-border)",
              color: "var(--mem-text)",
              backgroundColor: "var(--mem-surface)",
            }}
          >
            {isSyncing ? "Syncing…" : "Retry sync"}
          </button>
          <button
            onClick={onReveal}
            className="px-2.5 py-1 text-xs font-medium transition-colors"
            style={{ color: "var(--mem-text-secondary)" }}
          >
            Open in Finder →
          </button>
        </div>
      </div>
    </div>
  );
}
