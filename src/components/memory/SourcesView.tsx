// SPDX-License-Identifier: AGPL-3.0-only
import { useState, useMemo, useEffect } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import {
  listRegisteredSources,
  syncRegisteredSource,
  openFile,
  readSourceDir,
  readSourceText,
  removeSource,
  listIndexedFiles,
  type RegisteredSource,
  type SourceDirEntry,
  type SyncStatusStr,
  type IndexedFileInfo,
} from "../../lib/tauri";
import AddSourceMenu from "./sources/AddSourceMenu";
import ContentRenderer from "./ContentRenderer";
import { toast } from "sonner";

// The daemon's directory-ingest filter (wenlan-core sources/directory.rs).
// Files with these extensions feed the wiki; everything else is shown but dimmed.
const SUPPORTED_EXTENSIONS = ["md", "txt", "pdf"];

// Extensions we can render inline in the detail pane. Subset of the supported
// set — PDFs and binaries stay Open-only (no inline viewer). Plain text goes
// through the markdown renderer too; with no markup it just renders as prose.
const TEXT_RENDER_EXTENSIONS = ["md", "markdown", "mdx", "txt", "text"];

function folderName(p: string): string {
  return p.split("/").filter(Boolean).pop() || p;
}

function ext(name: string): string {
  const dot = name.lastIndexOf(".");
  return dot > 0 ? name.slice(dot + 1).toLowerCase() : "";
}

/** Collapse the daemon's SyncStatus enum to a short label, or null when Active. */
function statusLabel(status: SyncStatusStr): string | null {
  if (status === "Active") return null;
  if (status === "Paused") return "Paused";
  if (typeof status === "object" && "Unavailable" in status) return "Unavailable";
  return "Sync error";
}

const STATUS_COLORS: Record<string, string> = {
  Paused: "var(--mem-accent-amber)",
  "Sync error": "#ef4444",
  Unavailable: "var(--mem-text-tertiary)",
};

/** Mono caption under a source: Indexing… while settling, else "N notes" (+ skipped). */
export function sourceCaption(s: RegisteredSource): string {
  if (s.last_sync === null) return "Indexing…";
  const skipped = s.last_sync_errors ?? 0;
  const notes = `${s.memory_count.toLocaleString()} notes`;
  return skipped > 0 ? `${notes}, ${skipped} skipped` : notes;
}

function relTime(ts: number | null): string {
  if (!ts) return "never synced";
  const secs = Math.floor(Date.now() / 1000 - ts);
  if (secs < 60) return "synced just now";
  if (secs < 3600) return `synced ${Math.floor(secs / 60)}m ago`;
  if (secs < 86400) return `synced ${Math.floor(secs / 3600)}h ago`;
  return `synced ${Math.floor(secs / 86400)}d ago`;
}

/** True for the managed uploads dir (`~/.wenlan/sources`) — hoisted to the root
 *  as peers of its entries, never shown as a node of its own. */
export function isManagedSourcePath(path: string): boolean {
  return /\.wenlan\/sources\/?$/.test(path);
}

/** Folders first, then alphabetical — shared by the root build and every
 *  lazily-loaded subfolder. */
function sortDirEntries(entries: SourceDirEntry[]): SourceDirEntry[] {
  return [...entries].sort((a, b) => {
    if (a.isDirectory !== b.isDirectory) return a.isDirectory ? -1 : 1;
    return a.name.localeCompare(b.name);
  });
}

type SourcesNode =
  | { kind: "folder"; name: string; path: string; source: RegisteredSource; isSourceRoot: boolean }
  | { kind: "file"; name: string; path: string; source: RegisteredSource };

interface SourcesViewProps {
  /** Settings › Sources, for remove and advanced source management. */
  onManageSources: () => void;
}

export default function SourcesView({ onManageSources }: SourcesViewProps) {
  const [adding, setAdding] = useState(false);
  const [selectedPath, setSelectedPath] = useState<string | null>(null);
  const [selectedNode, setSelectedNode] = useState<SourcesNode | null>(null);

  const { data: fetchedSources = [] } = useQuery({
    queryKey: ["registeredSources"],
    queryFn: listRegisteredSources,
    // Refetch fast while anything is still settling; slow when idle (§3).
    refetchInterval: (q) => {
      const list = (q.state.data as RegisteredSource[] | undefined) ?? [];
      return list.some((s) => s.last_sync === null) ? 3000 : 10000;
    },
  });
  const sources: RegisteredSource[] = fetchedSources;

  // Which files the daemon actually indexed. The daemon silently drops files
  // it can't extract text from (e.g. a scanned PDF), so a folder listing can
  // show files that never made it into the library — surface that instead of
  // leaving a count quietly wrong.
  const { data: indexedFiles } = useQuery({
    queryKey: ["indexedFiles"],
    queryFn: listIndexedFiles,
    // Poll while on the Sources tab so a file that finishes indexing flips from
    // "Indexing…" to "In your library" without waiting for a window blur/focus.
    // Pauses when the window is unfocused (React Query default) and stops when
    // this view unmounts. ponytail: steady 5s is plenty at desktop scale.
    refetchInterval: 5000,
  });
  const indexReady = indexedFiles !== undefined;
  const indexedSet = useMemo(
    () => new Set((indexedFiles ?? []).map((f) => f.source_id)),
    [indexedFiles],
  );
  function isIndexed(source: RegisteredSource, absPath: string): boolean {
    return indexedSet.has(`${source.id}::${absPath}`);
  }

  // Tallest source first — the tree reads as a clear silhouette, the source
  // with the most memories on top (and the default selection).
  const folderSources = useMemo(
    () =>
      sources
        .filter((s) => !isManagedSourcePath(s.path))
        .sort((a, b) => b.memory_count - a.memory_count),
    [sources],
  );
  const managed = useMemo(() => sources.find((s) => isManagedSourcePath(s.path)), [sources]);

  const { data: managedEntries } = useQuery({
    queryKey: ["sourceDir", managed?.path ?? ""],
    queryFn: () => readSourceDir(managed?.path ?? ""),
    enabled: managed !== undefined,
  });

  const rootNodes: SourcesNode[] = useMemo(() => {
    const nodes: SourcesNode[] = folderSources.map((s) => ({
      kind: "folder",
      name: folderName(s.path),
      path: s.path,
      source: s,
      isSourceRoot: true,
    }));
    if (managed) {
      for (const e of sortDirEntries(managedEntries ?? [])) {
        const path = `${managed.path}/${e.name}`;
        nodes.push(
          e.isDirectory
            ? { kind: "folder", name: e.name, path, source: managed, isSourceRoot: false }
            : { kind: "file", name: e.name, path, source: managed },
        );
      }
    }
    return nodes;
  }, [folderSources, managed, managedEntries]);

  function selectNode(node: SourcesNode) {
    setSelectedPath(node.path);
    setSelectedNode(node);
  }

  useEffect(() => {
    if (rootNodes.length === 0) return;
    if (selectedPath === null) {
      selectNode(rootNodes[0]);
      return;
    }
    // The selected source may have been removed — fall back to root rather
    // than pointing the detail pane at a source that no longer exists.
    if (selectedNode && !sources.some((s) => s.id === selectedNode.source.id)) {
      selectNode(rootNodes[0]);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [rootNodes, selectedPath, selectedNode, sources]);

  // Re-derive the selected node's source from the live list on every render so
  // sync status / memory counts stay fresh across refetches without resetting
  // which node is selected.
  const liveSelectedNode: SourcesNode | null = useMemo(() => {
    if (!selectedNode) return null;
    const liveSource = sources.find((s) => s.id === selectedNode.source.id) ?? selectedNode.source;
    return { ...selectedNode, source: liveSource } as SourcesNode;
  }, [selectedNode, sources]);

  if (sources.length === 0) {
    return (
      <>
        <EmptyShelf onAdd={() => setAdding(true)} />
        {adding && <AddSourceMenu onClose={() => setAdding(false)} />}
      </>
    );
  }

  return (
    <div className="flex h-full w-full overflow-hidden">
      {/* ── LEFT: folder/file tree ── */}
      <aside
        className="flex-shrink-0 flex flex-col"
        style={{ width: 260, borderRight: "1px solid var(--mem-border)" }}
      >
        <div className="px-5 pt-6 pb-3">
          <div
            style={{
              fontFamily: "var(--mem-font-heading)",
              fontSize: "15px",
              fontWeight: 500,
              color: "var(--mem-text)",
              letterSpacing: "-0.01em",
            }}
          >
            Sources
          </div>
          <div
            style={{
              fontFamily: "var(--mem-font-mono)",
              fontSize: "11px",
              color: "var(--mem-text-tertiary)",
              marginTop: 4,
            }}
          >
            {folderSources.length} {folderSources.length === 1 ? "source" : "sources"}
          </div>
        </div>

        <div className="flex-1 overflow-y-auto px-3 pb-3 flex flex-col gap-0.5">
          {rootNodes.map((node) =>
            node.kind === "folder" ? (
              <FolderRow
                key={node.path}
                node={node}
                depth={0}
                selectedPath={selectedPath}
                onSelect={selectNode}
                isIndexed={isIndexed}
                indexReady={indexReady}
              />
            ) : (
              <FileRow
                key={node.path}
                node={node}
                depth={0}
                selectedPath={selectedPath}
                onSelect={selectNode}
                isIndexed={isIndexed}
                indexReady={indexReady}
              />
            ),
          )}
        </div>

        <div
          className="px-3 py-3 flex flex-col gap-1"
          style={{ borderTop: "1px solid var(--mem-border)" }}
        >
          <button
            onClick={() => setAdding(true)}
            className="w-full rounded-md border border-dashed px-3 py-2 text-left transition-colors duration-150"
            style={{
              borderColor: "var(--mem-border)",
              fontFamily: "var(--mem-font-body)",
              fontSize: "12px",
              color: "var(--mem-text-tertiary)",
            }}
            onMouseEnter={(e) => {
              e.currentTarget.style.borderColor = "var(--mem-accent-indigo)";
              e.currentTarget.style.color = "var(--mem-accent-indigo)";
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.borderColor = "var(--mem-border)";
              e.currentTarget.style.color = "var(--mem-text-tertiary)";
            }}
          >
            + Add source
          </button>
          <button
            onClick={onManageSources}
            className="w-full rounded-md px-3 py-1.5 text-left transition-colors duration-150 hover:bg-[var(--mem-hover)]"
            style={{
              fontFamily: "var(--mem-font-body)",
              fontSize: "11.5px",
              color: "var(--mem-text-tertiary)",
            }}
          >
            Manage sources ⚙
          </button>
        </div>
      </aside>

      {/* ── RIGHT: details of the selected node ── */}
      <DetailPane
        node={liveSelectedNode}
        isIndexed={isIndexed}
        indexReady={indexReady}
        indexedFiles={indexedFiles}
      />

      {adding && <AddSourceMenu onClose={() => setAdding(false)} />}
    </div>
  );
}

function Chevron({ expanded }: { expanded: boolean }) {
  return (
    <svg
      width="10"
      height="10"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="3"
      strokeLinecap="round"
      strokeLinejoin="round"
      style={{
        color: "var(--mem-text-tertiary)",
        transform: expanded ? "rotate(90deg)" : undefined,
        transition: "transform 150ms",
        flexShrink: 0,
      }}
    >
      <polyline points="9 18 15 12 9 6" />
    </svg>
  );
}

interface FolderRowProps {
  node: Extract<SourcesNode, { kind: "folder" }>;
  depth: number;
  selectedPath: string | null;
  onSelect: (node: SourcesNode) => void;
  isIndexed: (source: RegisteredSource, absPath: string) => boolean;
  indexReady: boolean;
}

function FolderRow({ node, depth, selectedPath, onSelect, isIndexed, indexReady }: FolderRowProps) {
  const [expanded, setExpanded] = useState(false);
  const active = selectedPath === node.path;

  // Raw queryFn (no sort) so it matches FolderDetail's observer on the same
  // ["sourceDir", path] key — React Query caches one result per key, so a
  // clicked folder (selected + expanded) must not depend on which observer's
  // queryFn wins the fetch. Order is applied at render below.
  const { data: children, isLoading: childrenLoading } = useQuery({
    queryKey: ["sourceDir", node.path],
    queryFn: () => readSourceDir(node.path),
    enabled: expanded,
  });

  return (
    <>
      <button
        onClick={() => {
          setExpanded((e) => !e);
          onSelect(node);
        }}
        title={node.path}
        className="w-full flex items-center gap-2 rounded-md text-left transition-colors duration-150"
        style={{
          padding: "7px 8px",
          paddingLeft: 8 + depth * 16,
          background: active ? "var(--mem-indigo-bg)" : "transparent",
          cursor: "pointer",
        }}
        onMouseEnter={(e) => {
          if (!active) e.currentTarget.style.background = "var(--mem-hover)";
        }}
        onMouseLeave={(e) => {
          if (!active) e.currentTarget.style.background = "transparent";
        }}
      >
        <Chevron expanded={expanded} />
        <FileGlyph isDir supported={false} />
        <span
          className="flex-1 min-w-0 truncate"
          style={{
            fontFamily: "var(--mem-font-heading)",
            fontSize: "14px",
            fontWeight: 500,
            color: active ? "var(--mem-text)" : "var(--mem-text-secondary)",
            letterSpacing: "-0.005em",
          }}
        >
          {node.name}
        </span>
      </button>
      {expanded &&
        (childrenLoading ? (
          <div style={{ paddingLeft: 8 + (depth + 1) * 16 }}>
            <SkeletonRows />
          </div>
        ) : (children ?? []).length === 0 ? (
          <div style={{ paddingLeft: 8 + (depth + 1) * 16 }}>
            <EmptyLine text="This folder is empty." />
          </div>
        ) : (
          sortDirEntries(children ?? []).map((e) => {
            const path = `${node.path}/${e.name}`;
            return e.isDirectory ? (
              <FolderRow
                key={path}
                node={{ kind: "folder", name: e.name, path, source: node.source, isSourceRoot: false }}
                depth={depth + 1}
                selectedPath={selectedPath}
                onSelect={onSelect}
                isIndexed={isIndexed}
                indexReady={indexReady}
              />
            ) : (
              <FileRow
                key={path}
                node={{ kind: "file", name: e.name, path, source: node.source }}
                depth={depth + 1}
                selectedPath={selectedPath}
                onSelect={onSelect}
                isIndexed={isIndexed}
                indexReady={indexReady}
              />
            );
          })
        ))}
    </>
  );
}

interface FileRowProps {
  node: Extract<SourcesNode, { kind: "file" }>;
  depth: number;
  selectedPath: string | null;
  onSelect: (node: SourcesNode) => void;
  isIndexed: (source: RegisteredSource, absPath: string) => boolean;
  indexReady: boolean;
}

function FileRow({ node, depth, selectedPath, onSelect, isIndexed, indexReady }: FileRowProps) {
  const active = selectedPath === node.path;
  const supported = SUPPORTED_EXTENSIONS.includes(ext(node.name));
  // A supported file not yet in the index — gated on indexReady so nothing is
  // flagged while the indexed-file list is still loading. Suppressed on the
  // active row: the detail pane already states this file's index status.
  const indexing = supported && indexReady && !isIndexed(node.source, node.path) && !active;

  return (
    <button
      onClick={() => onSelect(node)}
      title={node.path}
      className="w-full flex items-center gap-2 rounded-md text-left transition-colors duration-150"
      style={{
        padding: "7px 8px",
        paddingLeft: 8 + 14 + depth * 16,
        background: active ? "var(--mem-indigo-bg)" : "transparent",
        cursor: "pointer",
      }}
      onMouseEnter={(e) => {
        if (!active) e.currentTarget.style.background = "var(--mem-hover)";
      }}
      onMouseLeave={(e) => {
        if (!active) e.currentTarget.style.background = "transparent";
      }}
    >
      <FileGlyph isDir={false} supported={supported} />
      <span
        className="flex-1 min-w-0 truncate"
        style={{
          fontFamily: "var(--mem-font-body)",
          fontSize: "13.5px",
          color: indexing
            ? "var(--mem-text-tertiary)"
            : supported
              ? "var(--mem-text-secondary)"
              : "var(--mem-text-tertiary)",
        }}
      >
        {node.name}
      </span>
      {indexing && (
        <span
          style={{
            fontFamily: "var(--mem-font-mono)",
            fontSize: "10px",
            letterSpacing: "0.02em",
            color: "var(--mem-text-tertiary)",
            whiteSpace: "nowrap",
          }}
        >
          Indexing…
        </span>
      )}
    </button>
  );
}

interface DetailPaneProps {
  node: SourcesNode | null;
  isIndexed: (source: RegisteredSource, absPath: string) => boolean;
  indexReady: boolean;
  indexedFiles?: IndexedFileInfo[];
}

function DetailPane({ node, isIndexed, indexReady, indexedFiles }: DetailPaneProps) {
  if (!node) return <section className="flex-1 flex flex-col overflow-hidden" />;
  return node.kind === "file" ? (
    <FileDetail
      key={node.path}
      node={node}
      isIndexed={isIndexed}
      indexReady={indexReady}
      indexedFiles={indexedFiles}
    />
  ) : (
    <FolderDetail key={node.path} node={node} />
  );
}

function FileDetail({
  node,
  isIndexed,
  indexReady,
  indexedFiles,
}: {
  node: Extract<SourcesNode, { kind: "file" }>;
  isIndexed: (source: RegisteredSource, absPath: string) => boolean;
  indexReady: boolean;
  indexedFiles?: IndexedFileInfo[];
}) {
  const e = ext(node.name);
  const supported = SUPPORTED_EXTENSIONS.includes(e);
  const indexed = indexReady && isIndexed(node.source, node.path);
  const info = indexedFiles?.find((f) => f.source_id === `${node.source.id}::${node.path}`);

  // Render the real file inline for text formats; PDFs/binaries stay Open-only.
  const isText = TEXT_RENDER_EXTENSIONS.includes(e);
  const {
    data: fileText,
    isLoading: textLoading,
    isError: textError,
  } = useQuery({
    queryKey: ["fileText", node.path],
    queryFn: () => readSourceText(node.path),
    enabled: isText,
  });

  // Gate on indexReady before deciding indexed-vs-indexing; never claim
  // "not indexed" — an auto-syncing source is almost always mid-flight.
  let statusText: string;
  if (!supported) statusText = "Unsupported type";
  else if (indexReady && indexed) statusText = "In your library";
  else statusText = "Indexing…";

  return (
    <section className="flex-1 flex flex-col overflow-hidden">
      <div className="px-8 pt-6 pb-4" style={{ borderBottom: "1px solid var(--mem-border)" }}>
        <div className="flex items-start justify-between gap-4">
          <div className="min-w-0">
            <div className="flex items-center gap-2 flex-wrap">
              <h2
                style={{
                  fontFamily: "var(--mem-font-heading)",
                  fontSize: "18px",
                  fontWeight: 500,
                  color: "var(--mem-text)",
                  letterSpacing: "-0.01em",
                  margin: 0,
                }}
              >
                {node.name}
              </h2>
              {e && (
                <span
                  style={{
                    fontFamily: "var(--mem-font-mono)",
                    fontSize: "10px",
                    letterSpacing: "0.04em",
                    color: supported ? "var(--mem-accent-indigo)" : "var(--mem-text-tertiary)",
                    opacity: supported ? 0.9 : 0.5,
                  }}
                >
                  {e}
                </span>
              )}
            </div>
            <div
              style={{
                fontFamily: "var(--mem-font-mono)",
                fontSize: "11px",
                color: "var(--mem-text-tertiary)",
                marginTop: 6,
              }}
            >
              {statusText}
            </div>
            {info && (
              <div className="flex items-center gap-2 flex-wrap" style={{ marginTop: 4 }}>
                <span
                  style={{
                    fontFamily: "var(--mem-font-mono)",
                    fontSize: "11px",
                    color: "var(--mem-text-tertiary)",
                  }}
                >
                  Indexed as {info.chunk_count} {info.chunk_count === 1 ? "passage" : "passages"}
                  {info.last_modified
                    ? ` · updated ${relTime(info.last_modified).replace(/^synced /, "")}`
                    : ""}
                </span>
                {info.memory_type && (
                  <span
                    style={{
                      fontFamily: "var(--mem-font-mono)",
                      fontSize: "10px",
                      letterSpacing: "0.02em",
                      padding: "1px 6px",
                      borderRadius: 4,
                      background: "var(--mem-indigo-bg)",
                      color: "var(--mem-accent-indigo)",
                    }}
                  >
                    {info.memory_type}
                  </span>
                )}
                {info.domain && (
                  <span
                    style={{
                      fontFamily: "var(--mem-font-mono)",
                      fontSize: "10px",
                      letterSpacing: "0.02em",
                      padding: "1px 6px",
                      borderRadius: 4,
                      background: "var(--mem-hover)",
                      color: "var(--mem-text-tertiary)",
                    }}
                  >
                    {info.domain}
                  </span>
                )}
              </div>
            )}
          </div>
          <div className="shrink-0">
            <button
              onClick={() => openFile(node.path)}
              className="rounded-md transition-colors duration-150 hover:bg-[var(--mem-hover)]"
              style={{
                padding: "6px 13px",
                fontFamily: "var(--mem-font-body)",
                fontSize: "12px",
                fontWeight: 500,
                color: "white",
                background: "var(--mem-accent-indigo)",
                border: "none",
                cursor: "pointer",
              }}
            >
              Open
            </button>
          </div>
        </div>
      </div>
      <div className="flex-1 overflow-y-auto px-8 pt-4 pb-8 flex flex-col gap-4">
        {info?.summary && (
          <p
            style={{
              fontFamily: "var(--mem-font-body)",
              fontSize: "13px",
              color: "var(--mem-text-secondary)",
              lineHeight: 1.6,
              margin: 0,
              maxWidth: 560,
            }}
          >
            {info.summary}
          </p>
        )}

        {isText && textLoading && (
          <div
            style={{
              fontFamily: "var(--mem-font-mono)",
              fontSize: "11px",
              color: "var(--mem-text-tertiary)",
            }}
          >
            Loading…
          </div>
        )}
        {isText && textError && (
          <div
            style={{
              fontFamily: "var(--mem-font-mono)",
              fontSize: "11px",
              color: "var(--mem-text-tertiary)",
            }}
          >
            Couldn't read this file.
          </div>
        )}
        {isText && !textLoading && !textError && fileText && (
          <div style={{ maxWidth: 680 }}>
            <ContentRenderer content={fileText} variant="detail" />
          </div>
        )}
      </div>
    </section>
  );
}

function FolderDetail({ node }: { node: Extract<SourcesNode, { kind: "folder" }> }) {
  const queryClient = useQueryClient();
  const [syncedFlash, setSyncedFlash] = useState(false);
  const { source, isSourceRoot, path, name } = node;

  const {
    data: entries,
    isLoading,
    isError,
  } = useQuery({
    queryKey: ["sourceDir", path],
    queryFn: () => readSourceDir(path),
  });
  // Ground truth for "how many files are here" is the on-disk listing, not the
  // daemon's source.file_count (whole-source, and known to miscount — it drops
  // files it can't extract text from without adjusting the total).
  const fileCount = (entries ?? []).filter((e) => !e.isDirectory).length;

  const syncMutation = useMutation({
    mutationFn: () => syncRegisteredSource(source.id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["registeredSources"] });
      queryClient.invalidateQueries({ queryKey: ["sourceDir"] });
      setSyncedFlash(true);
      setTimeout(() => setSyncedFlash(false), 1600);
    },
  });

  const label = statusLabel(source.status);

  function handleRemove() {
    if (
      !window.confirm(
        `Remove ${name}? Indexed notes stay in your library; this source stops syncing.`,
      )
    )
      return;
    removeSource(source.id)
      .then(() => {
        queryClient.invalidateQueries({ queryKey: ["registeredSources"] });
      })
      .catch((err) => {
        toast("Couldn't remove source", { description: String(err) });
      });
  }

  return (
    <section className="flex-1 flex flex-col overflow-hidden">
      <div className="px-8 pt-6 pb-4" style={{ borderBottom: "1px solid var(--mem-border)" }}>
        <div className="flex items-start justify-between gap-4">
          <div className="min-w-0">
            {/* Name is the headline; the full path is demoted to a muted
                caption below. */}
            <h2
              className="truncate"
              style={{
                fontFamily: "var(--mem-font-heading)",
                fontSize: "18px",
                fontWeight: 500,
                color: "var(--mem-text)",
                letterSpacing: "-0.01em",
                margin: 0,
              }}
              title={path}
            >
              {name}
            </h2>
            <div
              className="truncate"
              style={{
                fontFamily: "var(--mem-font-mono)",
                fontSize: "11px",
                color: "var(--mem-text-tertiary)",
                marginTop: 3,
              }}
              title={path}
            >
              {path}
            </div>
            <div
              style={{
                fontFamily: "var(--mem-font-mono)",
                fontSize: "11px",
                color: label ? STATUS_COLORS[label] : "var(--mem-text-tertiary)",
                marginTop: 6,
              }}
            >
              {isLoading ? "…" : `${fileCount.toLocaleString()} ${fileCount === 1 ? "file" : "files"}`}
              {isSourceRoot && ` · ${source.memory_count.toLocaleString()} memories`}
              {isSourceRoot && ` · ${label ?? relTime(source.last_sync)}`}
            </div>
            {isSourceRoot && (source.last_sync_errors ?? 0) > 0 && (
              <div
                style={{
                  fontFamily: "var(--mem-font-mono)",
                  fontSize: "11px",
                  color: "var(--mem-accent-amber)",
                  marginTop: 2,
                }}
              >
                {source.last_sync_errors} file{source.last_sync_errors === 1 ? "" : "s"} couldn't be indexed
              </div>
            )}
          </div>

          <div className="flex items-center gap-2 shrink-0">
            {isSourceRoot && (
              <button
                onClick={handleRemove}
                className="rounded-md transition-colors duration-150 hover:bg-[var(--mem-hover)]"
                style={{
                  padding: "6px 11px",
                  fontFamily: "var(--mem-font-body)",
                  fontSize: "12px",
                  color: "var(--mem-text-secondary)",
                  background: "transparent",
                  border: "1px solid var(--mem-border)",
                  cursor: "pointer",
                }}
              >
                Remove
              </button>
            )}
            <button
              onClick={() => openFile(path)}
              className="rounded-md transition-colors duration-150 hover:bg-[var(--mem-hover)]"
              title="Reveal in Finder"
              style={{
                padding: "6px 11px",
                fontFamily: "var(--mem-font-body)",
                fontSize: "12px",
                color: "var(--mem-text-secondary)",
                background: "transparent",
                border: "1px solid var(--mem-border)",
                cursor: "pointer",
              }}
            >
              Reveal
            </button>
            {isSourceRoot &&
              (source.source_type === "obsidian" ? (
                <button
                  onClick={() => syncMutation.mutate()}
                  disabled={syncMutation.isPending}
                  className="rounded-md transition-colors duration-150"
                  style={{
                    padding: "6px 13px",
                    fontFamily: "var(--mem-font-body)",
                    fontSize: "12px",
                    fontWeight: 500,
                    color: "white",
                    background: "var(--mem-accent-indigo)",
                    border: "none",
                    cursor: syncMutation.isPending ? "default" : "pointer",
                    opacity: syncMutation.isPending ? 0.6 : 1,
                  }}
                >
                  {syncedFlash ? "✓ Synced" : syncMutation.isPending ? "Syncing…" : "Sync"}
                </button>
              ) : (
                <span
                  style={{
                    fontFamily: "var(--mem-font-mono)",
                    fontSize: "11px",
                    color: "var(--mem-text-tertiary)",
                  }}
                >
                  Auto-synced
                  {source.last_sync ? ` · updated ${relTime(source.last_sync).replace(/^synced /, "")}` : ""}
                </span>
              ))}
          </div>
        </div>
      </div>

      <div className="flex-1 overflow-y-auto px-8 py-6">
        {isError && <FolderError onReveal={() => openFile(path)} atRoot={isSourceRoot} />}
      </div>
    </section>
  );
}

function FileGlyph({ isDir, supported }: { isDir: boolean; supported: boolean }) {
  if (isDir) {
    return (
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" style={{ color: "var(--mem-accent-indigo)", opacity: 0.8 }}>
        <path d="M3 7a2 2 0 0 1 2-2h4l2 2h8a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z" />
      </svg>
    );
  }
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" style={{ color: supported ? "var(--mem-text-secondary)" : "var(--mem-text-tertiary)", opacity: supported ? 0.85 : 0.5 }}>
      <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
      <polyline points="14 2 14 8 20 8" />
    </svg>
  );
}

function SkeletonRows() {
  return (
    <div className="flex flex-col gap-2 pt-1">
      {[0.9, 0.7, 0.8, 0.55, 0.65].map((w, i) => (
        <div
          key={i}
          style={{
            height: 18,
            width: `${w * 100}%`,
            maxWidth: 320,
            borderRadius: 5,
            background: "var(--mem-shimmer-color)",
          }}
        />
      ))}
    </div>
  );
}

function EmptyLine({ text }: { text: string }) {
  return (
    <div
      className="pt-8"
      style={{ fontFamily: "var(--mem-font-body)", fontSize: "13px", color: "var(--mem-text-tertiary)" }}
    >
      {text}
    </div>
  );
}

function FolderError({ onReveal, atRoot }: { onReveal: () => void; atRoot: boolean }) {
  return (
    <div className="pt-8 flex flex-col items-start gap-3">
      <p style={{ fontFamily: "var(--mem-font-body)", fontSize: "13px", color: "var(--mem-text-secondary)", maxWidth: 380, lineHeight: 1.55 }}>
        {atRoot
          ? "This source can't be read — the folder may have been moved, renamed, or unmounted."
          : "This folder can't be read — it may have been moved or renamed."}
      </p>
      <button
        onClick={onReveal}
        className="rounded-md transition-colors duration-150 hover:bg-[var(--mem-hover)]"
        style={{
          padding: "6px 12px",
          fontFamily: "var(--mem-font-body)",
          fontSize: "12px",
          color: "var(--mem-text-secondary)",
          background: "transparent",
          border: "1px solid var(--mem-border)",
          cursor: "pointer",
        }}
      >
        Reveal in Finder
      </button>
    </div>
  );
}

function EmptyShelf({ onAdd }: { onAdd: () => void }) {
  return (
    <div className="flex h-full w-full items-center justify-center px-8">
      <div className="flex flex-col items-center text-center" style={{ maxWidth: 420 }}>
        <h2
          style={{
            fontFamily: "var(--mem-font-heading)",
            fontSize: "26px",
            fontWeight: 500,
            color: "var(--mem-text)",
            margin: "10px 0 0 0",
            letterSpacing: "-0.02em",
          }}
        >
          Nothing on the shelf yet
        </h2>
        <p
          style={{
            fontFamily: "var(--mem-font-body)",
            fontSize: "14px",
            color: "var(--mem-text-secondary)",
            margin: "10px 0 22px 0",
            lineHeight: 1.6,
          }}
        >
          Wenlan builds its wiki from the folders and vaults you add here. Point it at a notes
          folder or an Obsidian vault to lay the first stone.
        </p>
        <button
          onClick={onAdd}
          className="rounded-md transition-colors duration-150"
          style={{
            padding: "9px 18px",
            fontFamily: "var(--mem-font-body)",
            fontSize: "13px",
            fontWeight: 500,
            color: "white",
            background: "var(--mem-accent-indigo)",
            border: "none",
            cursor: "pointer",
          }}
        >
          + Add your first source
        </button>
      </div>
    </div>
  );
}
