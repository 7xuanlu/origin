// SPDX-License-Identifier: AGPL-3.0-only
import { useState, useMemo, useSyncExternalStore } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import Markdown from "react-markdown";
import {
  getChunks,
  updateChunk,
  clipboardWrite,
  type IndexedFileInfo,
  type ChunkDetail,
} from "../lib/tauri";
import { isProcessing, subscribe, getSnapshot } from "../lib/processingStore";

const SOURCE_LABELS: Record<string, string> = {
  local_files: "File",
  clipboard: "Clipboard",
  manual: "Capture",
  screen_capture: "Screen",
};

const SOURCE_COLORS: Record<string, string> = {
  local_files: "bg-blue-500/15 text-blue-400",
  clipboard: "bg-amber-500/15 text-amber-400",
  manual: "bg-purple-500/15 text-purple-400",
  screen_capture: "bg-green-500/15 text-green-400",
};

const CHUNK_TYPE_COLORS: Record<string, string> = {
  code: "bg-blue-500/10 text-blue-300",
  prose: "bg-zinc-500/10 text-zinc-400",
  markdown: "bg-teal-500/10 text-teal-400",
  text: "bg-zinc-500/10 text-zinc-400",
};

interface ChunkViewerProps {
  file: IndexedFileInfo;
  onBack: () => void;
}

function highlight(text: string, term: string) {
  if (!term) return <>{text}</>;
  const parts = text.split(new RegExp(`(${term.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")})`, "gi"));
  return (
    <>
      {parts.map((part, i) =>
        part.toLowerCase() === term.toLowerCase()
          ? <mark key={i} className="bg-[var(--accent)]/30 text-[var(--text-primary)] rounded-sm">{part}</mark>
          : part
      )}
    </>
  );
}

export default function ChunkViewer({ file, onBack }: ChunkViewerProps) {
  const queryClient = useQueryClient();
  const [editingId, setEditingId] = useState<string | null>(null);
  const [editContent, setEditContent] = useState("");
  const [search, setSearch] = useState("");
  const [copiedId, setCopiedId] = useState<string | null>(null);

  // Processing state from persistent store (survives navigation)
  useSyncExternalStore(subscribe, getSnapshot);
  const fileProcessing = file.processing || isProcessing(file.source_id);

  const { data: allChunks = [], isLoading } = useQuery({
    queryKey: ["chunks", file.source, file.source_id],
    queryFn: () => getChunks(file.source, file.source_id),
  });

  const updateMutation = useMutation({
    mutationFn: ({ id, content }: { id: string; content: string }) =>
      updateChunk(id, content),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["chunks", file.source, file.source_id] });
      setEditingId(null);
    },
  });

  const visibleChunks = useMemo(() => {
    if (!search.trim()) return allChunks;
    const term = search.toLowerCase();
    return allChunks.filter((c) => c.content.toLowerCase().includes(term));
  }, [allChunks, search]);

  async function copyChunk(chunk: ChunkDetail) {
    await clipboardWrite(chunk.content);
    setCopiedId(chunk.id);
    setTimeout(() => setCopiedId(null), 1500);
  }

  function startEditing(chunk: ChunkDetail) {
    setEditingId(chunk.id);
    setEditContent(chunk.content);
  }

  function cancelEditing() {
    setEditingId(null);
    setEditContent("");
  }

  function saveEditing(id: string) {
    updateMutation.mutate({ id, content: editContent });
  }

  return (
    // h-screen anchors to the viewport so flex-1 can distribute remaining height
    <div className="w-full h-screen flex flex-col bg-[var(--bg-primary)] overflow-hidden">
      {/* Header */}
      <div className="flex items-start gap-3 pl-[130px] pr-14 py-3 border-b border-[var(--separator)] shrink-0">
        <button
          onClick={onBack}
          className="text-[var(--text-secondary)] hover:text-[var(--text-primary)] transition-colors text-sm shrink-0 mt-0.5"
        >
          ‹ Back
        </button>
        <div className="flex-1 min-w-0">
          <h2 className="text-sm font-semibold text-[var(--text-primary)] truncate">
            {file.title}
          </h2>
          <div className="flex items-center gap-1.5 mt-1">
            <span className={`text-[10px] font-medium px-1.5 py-0.5 rounded ${SOURCE_COLORS[file.source] ?? "bg-zinc-500/15 text-zinc-400"}`}>
              {SOURCE_LABELS[file.source] ?? file.source}
            </span>
            <span className="text-xs text-[var(--text-tertiary)]">
              {search.trim()
                ? `${visibleChunks.length} of ${allChunks.length} chunk${allChunks.length !== 1 ? "s" : ""}`
                : `${allChunks.length} chunk${allChunks.length !== 1 ? "s" : ""}`}
            </span>
          </div>
        </div>
      </div>

      {/* Summary */}
      {file.summary ? (
        <div className="px-5 py-3 border-b border-[var(--separator)] shrink-0">
          <p className="text-xs text-[var(--text-secondary)] leading-relaxed">
            {file.summary}
          </p>
        </div>
      ) : fileProcessing ? (
        <div className="px-5 py-3 border-b border-[var(--separator)] shrink-0 flex items-center gap-2">
          <span className="w-1.5 h-1.5 bg-[var(--accent)] rounded-full animate-pulse" />
          <p className="text-xs text-[var(--text-tertiary)]">AI is summarizing…</p>
        </div>
      ) : null}

      {/* Search bar */}
      <div className="px-4 py-2 border-b border-[var(--separator)] shrink-0">
        <div className="flex items-center gap-2 bg-[var(--bg-secondary)] rounded-lg px-3 py-1.5">
          <svg className="w-3.5 h-3.5 text-[var(--text-tertiary)] shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
          </svg>
          <input
            type="text"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder="Search chunks…"
            className="flex-1 bg-transparent text-xs text-[var(--text-primary)] outline-none placeholder:text-[var(--text-tertiary)]"
            spellCheck={false}
          />
          {search && (
            <button
              onClick={() => setSearch("")}
              className="text-[var(--text-tertiary)] hover:text-[var(--text-primary)] transition-colors text-xs"
            >
              ✕
            </button>
          )}
        </div>
      </div>

      {/* Chunks — scrollable */}
      <div className="flex-1 overflow-y-auto p-4 space-y-3">
        {isLoading ? (
          <div className="px-4 py-6 text-center text-sm text-[var(--text-tertiary)]">
            Loading chunks...
          </div>
        ) : visibleChunks.length === 0 ? (
          <div className="px-4 py-6 text-center text-sm text-[var(--text-tertiary)]">
            {search.trim() ? "No chunks match" : "No chunks found"}
          </div>
        ) : (
          visibleChunks.map((chunk) => (
            <div
              key={chunk.id}
              className="bg-[var(--bg-secondary)] rounded-[10px] overflow-hidden"
            >
              {/* Chunk header */}
              <div className="flex items-center justify-between px-4 py-2 border-b border-[var(--separator)]">
                <div className="flex items-center gap-2 flex-wrap">
                  <span className="text-xs text-[var(--text-tertiary)]">
                    #{chunk.chunk_index}
                  </span>
                  {chunk.chunk_type && (
                    <span className={`text-[10px] font-medium px-1.5 py-0.5 rounded ${CHUNK_TYPE_COLORS[chunk.chunk_type] ?? "bg-zinc-500/10 text-zinc-400"}`}>
                      {chunk.chunk_type}
                    </span>
                  )}
                  {chunk.language && (
                    <span className="text-[10px] font-medium px-1.5 py-0.5 rounded bg-blue-500/10 text-blue-300">
                      {chunk.language}
                    </span>
                  )}
                </div>
                {editingId === chunk.id ? (
                  <div className="flex items-center gap-2">
                    <button
                      onClick={cancelEditing}
                      className="text-xs text-[var(--text-secondary)] hover:text-[var(--text-primary)] transition-colors"
                    >
                      Cancel
                    </button>
                    <button
                      onClick={() => saveEditing(chunk.id)}
                      disabled={updateMutation.isPending}
                      className="text-xs text-[var(--accent)] hover:text-[var(--accent-hover)] transition-colors disabled:opacity-50"
                    >
                      {updateMutation.isPending ? "Saving..." : "Save"}
                    </button>
                  </div>
                ) : (
                  <div className="flex items-center gap-2">
                    <button
                      onClick={() => copyChunk(chunk)}
                      className={`text-xs transition-colors ${
                        copiedId === chunk.id
                          ? "text-green-400"
                          : "text-[var(--text-secondary)] hover:text-[var(--text-primary)]"
                      }`}
                    >
                      {copiedId === chunk.id ? "Copied!" : "Copy"}
                    </button>
                    <button
                      onClick={() => startEditing(chunk)}
                      className="text-xs text-[var(--text-secondary)] hover:text-[var(--text-primary)] transition-colors"
                    >
                      Edit
                    </button>
                  </div>
                )}
              </div>

              {/* Chunk content */}
              <div className="px-4 py-3">
                {editingId === chunk.id ? (
                  <textarea
                    value={editContent}
                    onChange={(e) => setEditContent(e.target.value)}
                    className="w-full px-3 py-2 bg-[var(--bg-tertiary)] rounded text-xs text-[var(--text-primary)] focus:outline-none focus:ring-1 focus:ring-[var(--accent)] resize-none font-mono"
                    rows={Math.min(20, Math.max(4, editContent.split("\n").length + 1))}
                    autoFocus
                  />
                ) : (chunk.chunk_type === "markdown" || file.source === "screen_capture") && !search.trim() ? (
                  <div className="chunk-markdown text-xs text-[var(--text-secondary)] leading-relaxed">
                    <Markdown
                      components={{
                        h1: ({ children }) => <p className="text-sm font-semibold text-[var(--text-primary)] mb-1">{children}</p>,
                        h2: ({ children }) => <p className="text-xs font-semibold text-[var(--text-primary)] mb-1">{children}</p>,
                        h3: ({ children }) => <p className="text-xs font-medium text-[var(--text-primary)] mb-0.5">{children}</p>,
                        p: ({ children }) => <p className="mb-1.5">{children}</p>,
                        ul: ({ children }) => <ul className="list-disc list-inside mb-1.5 space-y-0.5">{children}</ul>,
                        ol: ({ children }) => <ol className="list-decimal list-inside mb-1.5 space-y-0.5">{children}</ol>,
                        li: ({ children }) => <li>{children}</li>,
                        code: ({ className, children }) => {
                          const isBlock = className?.includes("language-");
                          return isBlock
                            ? <pre className="bg-[var(--bg-tertiary)] rounded px-2 py-1.5 mb-1.5 overflow-x-auto"><code className="text-[11px] font-mono">{children}</code></pre>
                            : <code className="bg-[var(--bg-tertiary)] rounded px-1 py-0.5 text-[11px] font-mono">{children}</code>;
                        },
                        pre: ({ children }) => <>{children}</>,
                        blockquote: ({ children }) => <blockquote className="border-l-2 border-[var(--text-tertiary)] pl-2 mb-1.5 opacity-80">{children}</blockquote>,
                        hr: () => <hr className="border-[var(--separator)] my-2" />,
                        strong: ({ children }) => <strong className="font-semibold text-[var(--text-primary)]">{children}</strong>,
                      }}
                    >
                      {chunk.content}
                    </Markdown>
                  </div>
                ) : (
                  <div className="text-xs text-[var(--text-secondary)] whitespace-pre-wrap break-words leading-relaxed">
                    {highlight(chunk.content, search.trim())}
                  </div>
                )}
              </div>
            </div>
          ))
        )}
      </div>
    </div>
  );
}
