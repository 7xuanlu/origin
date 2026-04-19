// SPDX-License-Identifier: AGPL-3.0-only
import { useState, useRef, useEffect } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import {
  getMemoryDetail,
  getVersionChain,
  listEntities,
  listAllTags,
  search,
  updateMemory,
  reclassifyMemory,
  pinMemory,
  unpinMemory,
  deleteFileChunks,
  clipboardWrite,
  getPendingRevision,
  acceptPendingRevision,
  dismissPendingRevision,
  listSpaces,
  MEMORY_FACETS,
  FACET_COLORS,
  STABILITY_TIERS,
  type MemoryType,
  type PendingRevision,
} from "../../lib/tauri";
import TagEditor from "../TagEditor";
import ContentRenderer from "./ContentRenderer";

interface MemoryDetailProps {
  sourceId: string;
  onBack: () => void;
  onNavigateEntity: (entityId: string) => void;
  onNavigateMemory: (sourceId: string) => void;
}

function timeAgo(ts: number): string {
  const now = Date.now() / 1000;
  const diff = now - ts;
  if (diff < 60) return "just now";
  if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
  if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`;
  if (diff < 604800) return `${Math.floor(diff / 86400)}d ago`;
  return new Date(ts * 1000).toLocaleDateString();
}

function absoluteDate(ts: number): string {
  return new Date(ts * 1000).toLocaleString();
}

export default function MemoryDetail({
  sourceId,
  onBack,
  onNavigateEntity,
  onNavigateMemory,
}: MemoryDetailProps) {
  const queryClient = useQueryClient();
  const [editing, setEditing] = useState(false);
  const [editContent, setEditContent] = useState("");
  const [reclassifyOpen, setReclassifyOpen] = useState(false);
  const [spacePickerOpen, setSpacePickerOpen] = useState(false);
  const [expandedVersion, setExpandedVersion] = useState<string | null>(null);
  const [pendingRevision, setPendingRevision] = useState<PendingRevision | null>(null);
  const [copied, setCopied] = useState(false);
  const [editingTags, setEditingTags] = useState(false);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const dropdownRef = useRef<HTMLDivElement>(null);
  const spaceDropdownRef = useRef<HTMLDivElement>(null);

  const { data: spaces = [] } = useQuery({
    queryKey: ["spaces"],
    queryFn: listSpaces,
  });

  const { data: memory } = useQuery({
    queryKey: ["memoryDetail", sourceId],
    queryFn: () => getMemoryDetail(sourceId),
    refetchInterval: 5000,
  });

  const { data: versionChain = [] } = useQuery({
    queryKey: ["version-chain", sourceId],
    queryFn: () => getVersionChain(sourceId),
    enabled: !!memory?.supersedes,
  });

  const { data: relatedEntities = [] } = useQuery({
    queryKey: ["entities", undefined, memory?.domain],
    queryFn: () => listEntities(undefined, memory?.domain ?? undefined),
    enabled: !!memory?.domain,
  });

  // For recaps: fetch exact source memories from structured_fields.source_ids
  // For regular memories: semantic search for related
  const recapSourceIds: string[] = (() => {
    if (!memory?.is_recap || !memory.structured_fields) return [];
    try {
      const parsed = JSON.parse(memory.structured_fields);
      return Array.isArray(parsed.source_ids) ? parsed.source_ids : [];
    } catch { return []; }
  })();

  const { data: relatedSearchResults = [] } = useQuery({
    queryKey: ["relatedMemories", sourceId, recapSourceIds],
    queryFn: async () => {
      if (!memory) return [];
      // Recaps: only show source memories when we have exact IDs
      if (memory.is_recap) {
        if (recapSourceIds.length === 0) return [];
        const results = await Promise.all(
          recapSourceIds.map((id) => getMemoryDetail(id))
        );
        return results.filter((r): r is NonNullable<typeof r> => r != null);
      }
      // Regular memories: semantic search for related
      const results = await search(memory.content.substring(0, 200), 6, "memory");
      return results.filter((r) => r.source_id !== sourceId);
    },
    enabled: !!memory,
    staleTime: 30000,
  });

  const { data: tagData } = useQuery({
    queryKey: ["tags"],
    queryFn: listAllTags,
    refetchInterval: 5000,
  });

  const docTagKey = `memory::${sourceId}`;
  const currentTags = tagData?.document_tags[docTagKey] ?? [];
  const allTags = tagData?.tags ?? [];

  const invalidate = () => {
    queryClient.invalidateQueries({ queryKey: ["memoryDetail", sourceId] });
    queryClient.invalidateQueries({ queryKey: ["memories"] });
    queryClient.invalidateQueries({ queryKey: ["memoryStats"] });
  };

  const updateMutation = useMutation({
    mutationFn: (content: string) => updateMemory(sourceId, content),
    onSuccess: invalidate,
  });

  const confirmMutation = useMutation({
    mutationFn: (confirmed: boolean) => updateMemory(sourceId, undefined, undefined, confirmed),
    onSuccess: invalidate,
  });

  const reclassifyMutation = useMutation({
    mutationFn: (memoryType: MemoryType) => reclassifyMemory(sourceId, memoryType),
    onSuccess: () => {
      setReclassifyOpen(false);
      invalidate();
    },
  });

  const pinMutation = useMutation({
    mutationFn: () => memory?.pinned ? unpinMemory(sourceId) : pinMemory(sourceId),
    onSuccess: invalidate,
  });

  const deleteMutation = useMutation({
    mutationFn: () => deleteFileChunks("memory", sourceId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["memories"] });
      queryClient.invalidateQueries({ queryKey: ["memoryStats"] });
      onBack();
    },
  });

  const changeSpaceMutation = useMutation({
    mutationFn: (newDomain: string | undefined) =>
      updateMemory(sourceId, undefined, newDomain, undefined, undefined),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["memoryDetail", sourceId] });
      queryClient.invalidateQueries({ queryKey: ["memories"] });
      queryClient.invalidateQueries({ queryKey: ["spaces"] });
      setSpacePickerOpen(false);
    },
  });

  // Focus textarea on edit
  useEffect(() => {
    if (editing && textareaRef.current) {
      textareaRef.current.focus();
      textareaRef.current.selectionStart = textareaRef.current.value.length;
    }
  }, [editing]);

  // Close reclassify dropdown on click outside
  useEffect(() => {
    if (!reclassifyOpen) return;
    const handler = (e: MouseEvent) => {
      if (dropdownRef.current && !dropdownRef.current.contains(e.target as Node)) {
        setReclassifyOpen(false);
      }
    };
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, [reclassifyOpen]);

  // Close space picker on click outside
  useEffect(() => {
    if (!spacePickerOpen) return;
    const handler = (e: MouseEvent) => {
      if (spaceDropdownRef.current && !spaceDropdownRef.current.contains(e.target as Node)) {
        setSpacePickerOpen(false);
      }
    };
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, [spacePickerOpen]);

  // Fetch pending revision for protected confirmed memories
  useEffect(() => {
    if (!memory) return;
    const facetType = memory.memory_type ?? "fact";
    const tier = STABILITY_TIERS[facetType] ?? "ephemeral";
    if (tier === "protected" && memory.confirmed) {
      getPendingRevision(memory.source_id).then(setPendingRevision).catch(() => {});
    }
  }, [memory]);

  const handleAcceptRevision = async () => {
    if (!pendingRevision) return;
    await acceptPendingRevision(sourceId);
    setPendingRevision(null);
    invalidate();
  };

  const handleDismissRevision = async () => {
    if (!pendingRevision) return;
    await dismissPendingRevision(sourceId);
    setPendingRevision(null);
  };

  const handleSave = () => {
    if (editContent.trim() && editContent !== memory?.content) {
      updateMutation.mutate(editContent.trim());
    }
    setEditing(false);
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && e.metaKey) {
      e.preventDefault();
      handleSave();
    }
    if (e.key === "Escape") {
      setEditContent(memory?.content ?? "");
      setEditing(false);
    }
  };

  if (!memory) return null;

  const facetType = (memory.memory_type ?? "fact") as MemoryType;
  const isConfirmed = memory.confirmed;
  const relatedMemories = relatedSearchResults.slice(0, 5);

  return (
    <div className="flex flex-col gap-6">
      {/* Back + Header */}
      <div>
        <button onClick={onBack} className="p-1.5 -ml-1.5 rounded-md transition-colors duration-150 hover:bg-[var(--mem-hover)]" style={{ color: "var(--mem-text-tertiary)", background: "none", border: "none", cursor: "pointer", lineHeight: 0, marginBottom: "12px" }}>
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5"><path d="M19 12H5M12 19l-7-7 7-7" /></svg>
        </button>
      <div className="flex items-start justify-between gap-4">
        <div className="flex-1 min-w-0">
          <h2
            style={{
              fontFamily: "var(--mem-font-heading)",
              fontSize: "20px",
              color: "var(--mem-text)",
              fontWeight: 500,
              lineHeight: "1.4",
            }}
          >
            {memory.title || memory.content.split("\n")[0]?.substring(0, 80) || "Untitled memory"}
          </h2>
          <span
            title={absoluteDate(memory.last_modified)}
            style={{
              fontFamily: "var(--mem-font-mono)",
              fontSize: "11px",
              color: "var(--mem-text-tertiary)",
            }}
          >
            {timeAgo(memory.last_modified)}
          </span>
        </div>

        {/* Actions */}
        <div className="flex items-center gap-1 shrink-0">
          {/* Copy as context — recap only */}
          {memory.is_recap && (
            <button
              onClick={async () => {
                const domain = memory.domain ? `**Domain:** ${memory.domain}` : "";
                const time = `**Time:** ${absoluteDate(memory.last_modified)}`;
                const agent = memory.source_agent ? `**Generated by:** ${memory.source_agent}` : "";
                const meta = [domain, time, agent].filter(Boolean).join("\n");
                const contentSection = `### Content\n${memory.content}`;
                const text = [
                  "## Activity Recap",
                  meta,
                  "",
                  contentSection,
                ].join("\n");
                await clipboardWrite(text);
                setCopied(true);
                setTimeout(() => setCopied(false), 2000);
              }}
              className={`flex items-center gap-1 px-2 py-1 rounded-md text-[11px] font-medium transition-all ${
                copied
                  ? "bg-emerald-500/15 text-emerald-400"
                  : "hover:bg-[var(--mem-hover-strong)]"
              }`}
              style={copied ? undefined : { color: "var(--mem-accent-indigo)" }}
              title="Copy structured recap for LLM context"
            >
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="shrink-0">
                <rect x="9" y="9" width="13" height="13" rx="2" ry="2" />
                <path d="M5 15H4a2 2 0 01-2-2V4a2 2 0 012-2h9a2 2 0 012 2v1" />
              </svg>
              {copied ? "Copied!" : "Copy as context"}
            </button>
          )}

          {/* Pin toggle */}
          <button
            onClick={() => pinMutation.mutate()}
            className="p-1.5 rounded transition-colors duration-150 hover:bg-[var(--mem-hover-strong)]"
            style={{ color: "var(--mem-text-tertiary)" }}
            title={memory.pinned ? "Unpin" : "Pin"}
          >
            <span className={memory.pinned ? "text-amber-400" : "text-zinc-400"} style={{ fontSize: "14px" }}>
              {memory.pinned ? "\u2605" : "\u2606"}
            </span>
          </button>

          {/* Delete */}
          <button
            onClick={() => { if (confirm("Delete this memory?")) deleteMutation.mutate(); }}
            className="p-1.5 rounded transition-colors duration-150 hover:bg-[var(--mem-hover-strong)]"
            style={{ color: "var(--mem-text-tertiary)" }}
            title="Delete memory"
          >
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M3 6h18M8 6V4a2 2 0 012-2h4a2 2 0 012 2v2m3 0v14a2 2 0 01-2 2H7a2 2 0 01-2-2V6h14" />
            </svg>
          </button>
        </div>
      </div>
      </div>

      {/* Content section */}
      <section className="space-y-4">
        {/* Summary (recap) or Content (regular) */}
        <div>
          <div className="flex items-center justify-between mb-2">
            <h3
              className="pb-1"
              style={{
                fontFamily: "var(--mem-font-heading)",
                fontSize: "14px",
                color: "var(--mem-text)",
              }}
            >
              Content
            </h3>
            {!editing && memory.memory_type !== "recap" && (
              <button
                onClick={() => { setEditContent(memory.content); setEditing(true); }}
                className="px-2 py-0.5 rounded text-xs transition-colors duration-150 hover:bg-[var(--mem-hover)]"
                style={{
                  fontFamily: "var(--mem-font-body)",
                  color: "var(--mem-text-secondary)",
                }}
              >
                Edit
              </button>
            )}
          </div>
          <div
            className="rounded-lg px-4 py-3"
            style={{
              backgroundColor: memory.is_recap ? "var(--mem-indigo-bg)" : "var(--mem-surface)",
              border: "1px solid var(--mem-border)",
            }}
          >
            {editing ? (
              <div>
                <textarea
                  ref={textareaRef}
                  value={editContent}
                  onChange={(e) => setEditContent(e.target.value)}
                  onKeyDown={handleKeyDown}
                  className="w-full bg-transparent resize-none outline-none leading-relaxed"
                  style={{
                    color: "var(--mem-text)",
                    fontFamily: "var(--mem-font-body)",
                    fontSize: "14px",
                    lineHeight: "1.6",
                  }}
                  rows={Math.max(3, editContent.split("\n").length)}
                />
                <div
                  className="flex items-center gap-2 mt-2 pt-2"
                  style={{ borderTop: "1px solid var(--mem-border)" }}
                >
                  <button
                    onClick={handleSave}
                    className="px-2.5 py-1 rounded text-xs font-medium transition-colors duration-150"
                    style={{
                      backgroundColor: "var(--mem-accent-warm)",
                      color: "white",
                    }}
                  >
                    Save
                  </button>
                  <button
                    onClick={() => { setEditContent(memory.content); setEditing(false); }}
                    className="px-2.5 py-1 rounded text-xs transition-colors duration-150"
                    style={{ color: "var(--mem-text-tertiary)" }}
                  >
                    Cancel
                  </button>
                  <span
                    className="ml-auto"
                    style={{
                      fontFamily: "var(--mem-font-mono)",
                      fontSize: "10px",
                      color: "var(--mem-text-tertiary)",
                    }}
                  >
                    Cmd+Enter to save
                  </span>
                </div>
              </div>
            ) : (
              <ContentRenderer
                content={memory.source_text || memory.content}
                structuredFields={memory.structured_fields}
                variant="detail"
              />
            )}
          </div>
        </div>

      </section>

      {/* Structured fields */}
      {memory.structured_fields && memory.structured_fields !== "{}" && (() => {
        try {
          const fields: Record<string, string> = JSON.parse(memory.structured_fields);
          const entries = Object.entries(fields).filter(([, v]) => v && v.trim());
          if (entries.length === 0) return null;
          return (
            <section>
              <h3
                className="mb-2 uppercase tracking-wider"
                style={{
                  fontFamily: "var(--mem-font-mono)",
                  fontSize: "10px",
                  fontWeight: 600,
                  letterSpacing: "0.05em",
                  color: "var(--mem-text-tertiary)",
                }}
              >
                Details
              </h3>
              <div
                className="rounded-lg px-4 py-3 space-y-2"
                style={{
                  backgroundColor: "var(--mem-surface)",
                  border: "1px solid var(--mem-border)",
                }}
              >
                {entries.map(([key, val]) => (
                  <div key={key} className="flex gap-3">
                    <span
                      className="shrink-0"
                      style={{
                        fontFamily: "var(--mem-font-mono)",
                        fontSize: "11px",
                        color: "var(--mem-text-tertiary)",
                        minWidth: "100px",
                        textTransform: "capitalize",
                      }}
                    >
                      {key.replace(/_/g, " ")}
                    </span>
                    <span
                      style={{
                        fontFamily: "var(--mem-font-body)",
                        fontSize: "13px",
                        color: "var(--mem-text)",
                        lineHeight: "1.5",
                      }}
                    >
                      {val}
                    </span>
                  </div>
                ))}
              </div>
            </section>
          );
        } catch { return null; }
      })()}

      {/* Pending revision */}
      {pendingRevision && (
        <div
          className="rounded-lg px-4 py-3 text-xs"
          style={{
            backgroundColor: "rgba(245, 158, 11, 0.1)",
            border: "1px solid rgba(245, 158, 11, 0.3)",
            color: "rgb(245, 158, 11)",
          }}
        >
          <div className="font-medium mb-1">
            Proposed update{pendingRevision.source_agent ? ` from ${pendingRevision.source_agent}` : ""}
          </div>
          <p className="mb-2 whitespace-pre-wrap" style={{ color: "var(--mem-text-secondary)" }}>
            {pendingRevision.content}
          </p>
          <div className="flex gap-2">
            <button
              onClick={handleAcceptRevision}
              className="px-2 py-0.5 rounded text-[10px] font-medium transition-colors"
              style={{
                backgroundColor: "rgba(245, 158, 11, 0.2)",
                color: "rgb(245, 158, 11)",
              }}
            >
              Accept
            </button>
            <button
              onClick={handleDismissRevision}
              className="px-2 py-0.5 rounded text-[10px] font-medium transition-colors"
              style={{
                backgroundColor: "rgba(161, 161, 170, 0.15)",
                color: "var(--mem-text-tertiary)",
              }}
            >
              Dismiss
            </button>
          </div>
        </div>
      )}

      {/* Metadata section */}
      <section>
        <h3
          className="mb-3 pb-2"
          style={{
            fontFamily: "var(--mem-font-heading)",
            fontSize: "14px",
            color: "var(--mem-text)",
            borderBottom: "1px solid var(--mem-border)",
          }}
        >
          Metadata
        </h3>
        <div className="flex flex-col gap-2">
          {/* Facet type */}
          <div className="flex items-center gap-3">
            <span
              style={{
                fontFamily: "var(--mem-font-mono)",
                fontSize: "11px",
                color: "var(--mem-text-tertiary)",
                width: "80px",
              }}
            >
              Type
            </span>
            <div className="relative" ref={dropdownRef}>
              <button
                onClick={() => setReclassifyOpen(!reclassifyOpen)}
                className={`px-1.5 py-0.5 rounded text-[10px] font-medium border cursor-pointer ${FACET_COLORS[facetType]}`}
              >
                {facetType}
                <span className="ml-1 opacity-50">&#x25BE;</span>
              </button>
              {reclassifyOpen && (
                <div
                  className="absolute left-0 top-7 z-50 rounded-lg shadow-xl py-1 min-w-[140px]"
                  style={{
                    backgroundColor: "var(--mem-surface)",
                    border: "1px solid var(--mem-border)",
                  }}
                >
                  {MEMORY_FACETS.map((facet) => (
                    <button
                      key={facet.type}
                      className="w-full text-left px-3 py-1.5 text-xs flex items-center gap-2 transition-colors duration-150 hover:bg-[var(--mem-hover)]"
                      style={{
                        fontFamily: "var(--mem-font-body)",
                        color: facetType === facet.type ? "var(--mem-text)" : "var(--mem-text-secondary)",
                      }}
                      onClick={() => reclassifyMutation.mutate(facet.type)}
                    >
                      <span className={`w-2 h-2 rounded-full ${FACET_COLORS[facet.type].split(" ")[0]}`} />
                      {facet.label}
                    </button>
                  ))}
                </div>
              )}
            </div>
          </div>

          {/* Space */}
          <div className="flex items-center gap-3">
            <span
              style={{
                fontFamily: "var(--mem-font-mono)",
                fontSize: "11px",
                color: "var(--mem-text-tertiary)",
                width: "80px",
              }}
            >
              Space
            </span>
            <div className="relative" ref={spaceDropdownRef}>
              <button
                onClick={() => setSpacePickerOpen(!spacePickerOpen)}
                className="px-2 py-0.5 rounded text-[12px] cursor-pointer transition-colors duration-150 hover:bg-[var(--mem-hover)]"
                style={{
                  fontFamily: "var(--mem-font-body)",
                  color: memory.domain ? "var(--mem-text)" : "var(--mem-text-tertiary)",
                  border: "1px solid var(--mem-border)",
                }}
              >
                {memory.domain ? (
                  <span className="capitalize">{memory.domain}</span>
                ) : (
                  <span>Assign space</span>
                )}
              </button>
              {spacePickerOpen && (
                <div
                  className="absolute left-0 top-8 z-50 rounded-lg shadow-xl py-1 min-w-[140px]"
                  style={{
                    backgroundColor: "var(--mem-surface)",
                    border: "1px solid var(--mem-border)",
                    animation: "mem-fade-up 120ms ease both",
                  }}
                >
                  {memory.domain && (
                    <>
                      <button
                        onClick={() => changeSpaceMutation.mutate(undefined)}
                        className="w-full text-left px-3 py-1.5 text-xs transition-colors duration-150 hover:bg-[var(--mem-hover)]"
                        style={{ fontFamily: "var(--mem-font-body)", color: "var(--mem-text-tertiary)" }}
                      >
                        Remove from space
                      </button>
                      <div style={{ height: "1px", backgroundColor: "var(--mem-border)", margin: "2px 0" }} />
                    </>
                  )}
                  {spaces.filter((s) => s.name !== memory.domain).map((s) => (
                    <button
                      key={s.id}
                      onClick={() => changeSpaceMutation.mutate(s.name)}
                      className="w-full text-left px-3 py-1.5 text-xs transition-colors duration-150 hover:bg-[var(--mem-hover)]"
                      style={{ fontFamily: "var(--mem-font-body)", color: "var(--mem-text-secondary)" }}
                    >
                      <span className="capitalize">{s.name}</span>
                    </button>
                  ))}
                </div>
              )}
            </div>
          </div>

          {/* Entity link */}
          {memory.entity_id && (
            <div className="flex items-center gap-3">
              <span
                style={{
                  fontFamily: "var(--mem-font-mono)",
                  fontSize: "11px",
                  color: "var(--mem-text-tertiary)",
                  width: "80px",
                }}
              >
                Entity
              </span>
              <button
                onClick={() => onNavigateEntity(memory.entity_id!)}
                className="px-1.5 py-0.5 rounded text-[10px] font-medium transition-colors duration-150 hover:bg-[var(--mem-hover-strong)]"
                style={{
                  color: "var(--mem-accent-sage)",
                  textDecoration: "underline",
                  textDecorationColor: "color-mix(in srgb, var(--mem-accent-sage) 30%, transparent)",
                  textUnderlineOffset: "2px",
                }}
              >
                Linked
              </button>
            </div>
          )}

          {/* Quality */}
          {memory.quality && memory.quality === "low" && (
            <div className="flex items-center gap-3">
              <span
                style={{
                  fontFamily: "var(--mem-font-mono)",
                  fontSize: "11px",
                  color: "var(--mem-text-tertiary)",
                  width: "80px",
                }}
              >
                Quality
              </span>
              <span
                className="px-1.5 py-0.5 rounded text-[10px] font-medium"
                style={{
                  backgroundColor: "rgba(245, 158, 11, 0.1)",
                  color: "rgb(245, 158, 11)",
                }}
              >
                Low quality
              </span>
            </div>
          )}

          {/* Source agent */}
          {memory.source_agent && (
            <div className="flex items-center gap-3">
              <span
                style={{
                  fontFamily: "var(--mem-font-mono)",
                  fontSize: "11px",
                  color: "var(--mem-text-tertiary)",
                  width: "80px",
                }}
              >
                Agent
              </span>
              <span className="px-1.5 py-0.5 rounded text-[10px] font-medium bg-cyan-500/10 text-cyan-400">
                {memory.source_agent}
              </span>
            </div>
          )}

          {/* Confirmed toggle */}
          <div className="flex items-center gap-3">
            <span
              style={{
                fontFamily: "var(--mem-font-mono)",
                fontSize: "11px",
                color: "var(--mem-text-tertiary)",
                width: "80px",
              }}
            >
              Confirmed
            </span>
            <button
              onClick={() => confirmMutation.mutate(!isConfirmed)}
              className="flex items-center gap-1.5"
            >
              <span
                className="w-3 h-3 rounded-full border transition-all duration-300"
                style={{
                  borderColor: isConfirmed ? "var(--mem-accent-warm)" : "var(--mem-accent-amber)",
                  backgroundColor: isConfirmed ? "var(--mem-accent-warm)" : "transparent",
                }}
              />
              <span
                style={{
                  fontFamily: "var(--mem-font-body)",
                  fontSize: "12px",
                  color: isConfirmed ? "var(--mem-text)" : "var(--mem-text-tertiary)",
                }}
              >
                {isConfirmed ? "Yes" : "No"}
              </span>
            </button>
          </div>

          {/* Pinned */}
          <div className="flex items-center gap-3">
            <span
              style={{
                fontFamily: "var(--mem-font-mono)",
                fontSize: "11px",
                color: "var(--mem-text-tertiary)",
                width: "80px",
              }}
            >
              Pinned
            </span>
            <button
              onClick={() => pinMutation.mutate()}
              className="flex items-center gap-1.5"
            >
              <span className={memory.pinned ? "text-amber-400" : "text-zinc-400"} style={{ fontSize: "12px" }}>
                {memory.pinned ? "\u2605" : "\u2606"}
              </span>
              <span
                style={{
                  fontFamily: "var(--mem-font-body)",
                  fontSize: "12px",
                  color: memory.pinned ? "var(--mem-text)" : "var(--mem-text-tertiary)",
                }}
              >
                {memory.pinned ? "Yes" : "No"}
              </span>
            </button>
          </div>

          {/* Tags */}
          <div className="flex items-start gap-3">
            <span
              style={{
                fontFamily: "var(--mem-font-mono)",
                fontSize: "11px",
                color: "var(--mem-text-tertiary)",
                width: "80px",
                paddingTop: "2px",
              }}
            >
              Tags
            </span>
            <div className="relative flex-1">
              <div className="flex items-center gap-1.5 flex-wrap">
                {currentTags.map((tag) => (
                  <span
                    key={tag}
                    className="px-1.5 py-0.5 rounded text-[10px] font-medium bg-indigo-500/10 text-indigo-400"
                  >
                    {tag}
                  </span>
                ))}
                <button
                  onClick={() => setEditingTags(!editingTags)}
                  className="px-1.5 py-0.5 rounded text-[10px] font-medium transition-colors duration-150 hover:bg-[var(--mem-hover-strong)]"
                  style={{ color: "var(--mem-text-tertiary)" }}
                  title="Edit tags"
                >
                  {editingTags ? "Done" : "+"}
                </button>
              </div>
              {editingTags && (
                <TagEditor
                  source="memory"
                  sourceId={sourceId}
                  lastModified={memory.last_modified}
                  currentTags={currentTags}
                  allTags={allTags}
                  onClose={() => setEditingTags(false)}
                  onTagsChanged={() => queryClient.invalidateQueries({ queryKey: ["spaces"] })}
                />
              )}
            </div>
          </div>
        </div>
      </section>

      {/* Version history */}
      {memory.supersedes && versionChain.length > 0 && (
        <section>
          <h3
            className="mb-3 pb-2"
            style={{
              fontFamily: "var(--mem-font-heading)",
              fontSize: "14px",
              color: "var(--mem-text)",
              borderBottom: "1px solid var(--mem-border)",
            }}
          >
            Version History
          </h3>
          <div className="flex flex-col gap-1">
            {versionChain.map((v, i) => (
              <div key={v.source_id}>
                <button
                  onClick={() => setExpandedVersion(expandedVersion === v.source_id ? null : v.source_id)}
                  className="w-full flex items-center gap-2 px-3 py-2 rounded-md text-left transition-colors duration-150 hover:bg-[var(--mem-hover)]"
                >
                  <span
                    style={{
                      fontFamily: "var(--mem-font-mono)",
                      fontSize: "11px",
                      color: "var(--mem-text-tertiary)",
                      width: "24px",
                    }}
                  >
                    v{i + 1}
                  </span>
                  <span
                    className="flex-1 truncate"
                    style={{
                      fontFamily: "var(--mem-font-body)",
                      fontSize: "13px",
                      color: v.source_id === sourceId ? "var(--mem-text)" : "var(--mem-text-secondary)",
                      fontWeight: v.source_id === sourceId ? 500 : 400,
                    }}
                  >
                    {v.title}
                  </span>
                  {v.confirmed && (
                    <span
                      style={{
                        fontFamily: "var(--mem-font-mono)",
                        fontSize: "10px",
                        color: "var(--mem-accent-warm)",
                      }}
                    >
                      confirmed
                    </span>
                  )}
                  <span
                    style={{
                      fontFamily: "var(--mem-font-mono)",
                      fontSize: "10px",
                      color: "var(--mem-text-tertiary)",
                    }}
                  >
                    {timeAgo(v.last_modified)}
                  </span>
                  <svg
                    width="12" height="12"
                    viewBox="0 0 24 24"
                    fill="none"
                    stroke="currentColor"
                    strokeWidth="2"
                    style={{
                      color: "var(--mem-text-tertiary)",
                      transform: expandedVersion === v.source_id ? "rotate(90deg)" : "rotate(0deg)",
                      transition: "transform 150ms ease",
                    }}
                  >
                    <path d="M9 18l6-6-6-6" />
                  </svg>
                </button>
                {expandedVersion === v.source_id && (
                  <div
                    className="mx-3 mb-1 px-3 py-2 rounded"
                    style={{
                      backgroundColor: "var(--mem-hover)",
                      fontFamily: "var(--mem-font-body)",
                      fontSize: "13px",
                      color: "var(--mem-text-secondary)",
                      lineHeight: "1.6",
                      whiteSpace: "pre-wrap",
                    }}
                  >
                    {v.content}
                  </div>
                )}
              </div>
            ))}
          </div>
        </section>
      )}

      {/* Related entities */}
      {relatedEntities.length > 0 && (
        <section>
          <h3
            className="mb-3 pb-2"
            style={{
              fontFamily: "var(--mem-font-heading)",
              fontSize: "14px",
              color: "var(--mem-text)",
              borderBottom: "1px solid var(--mem-border)",
            }}
          >
            Related Entities
          </h3>
          <div className="flex flex-col gap-1">
            {relatedEntities.map((entity) => (
              <button
                key={entity.id}
                onClick={() => onNavigateEntity(entity.id)}
                className="flex items-center gap-2 px-3 py-2 rounded-md text-left transition-colors duration-150 hover:bg-[var(--mem-hover)]"
              >
                <span
                  style={{
                    fontFamily: "var(--mem-font-body)",
                    fontSize: "13px",
                    color: "var(--mem-accent-sage)",
                    textDecoration: "underline",
                    textDecorationColor: "color-mix(in srgb, var(--mem-accent-sage) 30%, transparent)",
                    textUnderlineOffset: "2px",
                  }}
                >
                  {entity.name}
                </span>
                <span
                  style={{
                    fontFamily: "var(--mem-font-mono)",
                    fontSize: "10px",
                    color: "var(--mem-text-tertiary)",
                  }}
                >
                  {entity.entity_type}
                </span>
              </button>
            ))}
          </div>
        </section>
      )}

      {/* Related / Source memories */}
      {relatedMemories.length > 0 && (
        <section className="pb-8">
          <h3
            className="mb-3 pb-2"
            style={{
              fontFamily: "var(--mem-font-heading)",
              fontSize: "14px",
              color: "var(--mem-text)",
              borderBottom: "1px solid var(--mem-border)",
            }}
          >
            {memory.is_recap ? "Source Memories" : "Related Memories"}
          </h3>
          <div className="flex flex-col gap-1">
            {relatedMemories.map((r) => {
              const rFacet = r.memory_type ?? null;
              const rColor = rFacet ? FACET_COLORS[rFacet] : null;
              return (
                <button
                  key={'id' in r ? r.id : r.source_id}
                  onClick={() => onNavigateMemory(r.source_id)}
                  className="flex items-start gap-3 px-3 py-2 rounded-md text-left transition-colors duration-150 hover:bg-[var(--mem-hover)]"
                >
                  <div className="flex-1 min-w-0">
                    <p
                      className="line-clamp-2"
                      style={{
                        fontFamily: "var(--mem-font-body)",
                        fontSize: "13px",
                        color: "var(--mem-text)",
                        lineHeight: "1.5",
                      }}
                    >
                      {r.content.length > 200 ? r.content.substring(0, 200) + "\u2026" : r.content}
                    </p>
                    <div
                      className="flex items-center gap-2 mt-1"
                      style={{
                        fontFamily: "var(--mem-font-mono)",
                        fontSize: "10px",
                        color: "var(--mem-text-tertiary)",
                      }}
                    >
                      {rFacet && rColor && (
                        <span className={`px-1 py-0.5 rounded text-[9px] font-medium border ${rColor}`}>
                          {rFacet}
                        </span>
                      )}
                      <span>{timeAgo(r.last_modified)}</span>
                    </div>
                  </div>
                  <svg
                    width="12" height="12"
                    viewBox="0 0 24 24"
                    fill="none"
                    stroke="currentColor"
                    strokeWidth="2"
                    className="shrink-0 mt-1"
                    style={{ color: "var(--mem-text-tertiary)" }}
                  >
                    <path d="M9 18l6-6-6-6" />
                  </svg>
                </button>
              );
            })}
          </div>
        </section>
      )}
    </div>
  );
}
