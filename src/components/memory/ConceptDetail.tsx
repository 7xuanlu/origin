// SPDX-License-Identifier: AGPL-3.0-only
import { useState, useCallback, useMemo, useRef, useEffect } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import {
  getConcept,
  listConcepts,
  updateConcept,
  deleteConcept,
  clipboardWrite,
  exportConceptToObsidian,
  listRegisteredSources,
  getConceptSources,
  type MemoryItem,
  type Concept,
} from "../../lib/tauri";
import ContentRenderer from "./ContentRenderer";

interface ConceptDetailProps {
  conceptId: string;
  onBack: () => void;
  onMemoryClick: (sourceId: string) => void;
  onConceptClick?: (conceptId: string) => void;
}

function relativeTimeFromISO(iso: string): string {
  const diff = Math.floor((Date.now() - new Date(iso).getTime()) / 1000);
  if (diff < 60) return "just now";
  if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
  if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`;
  return `${Math.floor(diff / 86400)}d ago`;
}

/** Parse [[wikilinks]] from content */
function parseRelatedConcepts(content: string): Array<{ title: string }> {
  const links: Array<{ title: string }> = [];
  const seen = new Set<string>();
  const wikiLinkRegex = /\[\[([^\]]+)\]\]/g;
  let match;
  while ((match = wikiLinkRegex.exec(content)) !== null) {
    const title = match[1];
    if (!seen.has(title.toLowerCase())) {
      seen.add(title.toLowerCase());
      links.push({ title });
    }
  }
  return links;
}

/** Build a map of lowercase title → concept id for resolved wikilinks */
function buildConceptLookup(concepts: Concept[]): Map<string, string> {
  const map = new Map<string, string>();
  for (const c of concepts) {
    map.set(c.title.toLowerCase(), c.id);
  }
  return map;
}

function folderName(path: string): string {
  return path.split("/").filter(Boolean).pop() || path;
}

const KNOWN_AGENTS: Record<string, string> = {
  "claude-code": "Claude Code",
  "claude-desktop": "Claude Desktop",
  cursor: "Cursor",
  "chatgpt-mcp": "ChatGPT",
  chatgpt: "ChatGPT",
  "gemini-cli": "Gemini CLI",
  windsurf: "Windsurf",
  zed: "Zed",
};

function prettyAgent(name: string | null | undefined): string {
  if (!name) return "unknown agent";
  const key = name.trim().toLowerCase();
  return KNOWN_AGENTS[key] ?? name;
}

const SOURCE_KIND_LABEL: Record<string, string> = {
  memory: "memory",
  chat: "chat",
  file: "file",
  obsidian: "obsidian",
  web: "web",
};

function sourceKindLabel(mem: MemoryItem): string {
  const mt = mem.memory_type?.toLowerCase() ?? "";
  return SOURCE_KIND_LABEL[mt] ?? (mt || "memory");
}

function relativeMs(ms: number): string {
  const delta = Date.now() - ms;
  if (delta < 60_000) return "just now";
  if (delta < 3_600_000) return `${Math.floor(delta / 60_000)}m ago`;
  if (delta < 86_400_000) return `${Math.floor(delta / 3_600_000)}h ago`;
  return `${Math.floor(delta / 86_400_000)}d ago`;
}

export default function ConceptDetail({ conceptId, onBack, onMemoryClick, onConceptClick }: ConceptDetailProps) {
  const queryClient = useQueryClient();
  const [copied, setCopied] = useState(false);
  const [exported, setExported] = useState(false);
  const [editing, setEditing] = useState(false);
  const [editContent, setEditContent] = useState("");
  const [exportMenuOpen, setExportMenuOpen] = useState(false);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const { data: concept, isLoading } = useQuery({
    queryKey: ["concept", conceptId],
    queryFn: () => getConcept(conceptId),
  });

  const { data: allConcepts = [] } = useQuery({
    queryKey: ["concepts-for-links"],
    queryFn: () => listConcepts("active"),
    staleTime: 30000,
  });

  const conceptLookup = buildConceptLookup(allConcepts);

  const { data: registeredSources = [] } = useQuery({
    queryKey: ["registeredSources"],
    queryFn: () => listRegisteredSources(),
    staleTime: 30000,
  });

  const obsidianSources = useMemo(
    () => registeredSources.filter((s) => s.source_type === "obsidian"),
    [registeredSources],
  );

  const { data: conceptSources } = useQuery({
    queryKey: ["concept-sources", conceptId],
    queryFn: () => getConceptSources(conceptId),
    enabled: !!conceptId,
  });

  // Extract MemoryItems from the join table result for backward-compatible rendering.
  const sourceMemories: MemoryItem[] | undefined = conceptSources
    ?.filter((cs) => cs.memory !== null)
    .map((cs) => cs.memory as MemoryItem);

  const updateMutation = useMutation({
    mutationFn: (content: string) => updateConcept(conceptId, content),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["concept", conceptId] });
      queryClient.invalidateQueries({ queryKey: ["concepts"] });
      setEditing(false);
    },
  });

  const deleteMutation = useMutation({
    mutationFn: () => deleteConcept(conceptId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["concepts"] });
      onBack();
    },
  });

  const copyAsContext = useCallback(async () => {
    if (!concept) return;
    const domain = concept.domain ? `**Domain:** ${concept.domain}` : "";
    const version = `**Version:** ${concept.version}`;
    const compiled = `**Last compiled:** ${concept.last_compiled}`;
    const meta = [domain, version, compiled].filter(Boolean).join("\n");
    const text = [
      `## ${concept.title}`,
      meta,
      "",
      concept.content,
    ].join("\n");
    await clipboardWrite(text);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  }, [concept]);

  const handleExportToVault = useCallback(
    async (vaultPath: string) => {
      setExportMenuOpen(false);
      await exportConceptToObsidian(conceptId, `${vaultPath}/Origin/concepts`);
      setExported(true);
      setTimeout(() => setExported(false), 2000);
    },
    [conceptId],
  );

  const handleSave = () => {
    if (editContent.trim() && editContent !== concept?.content) {
      updateMutation.mutate(editContent.trim());
    } else {
      setEditing(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && e.metaKey) {
      e.preventDefault();
      handleSave();
    }
    if (e.key === "Escape") {
      setEditing(false);
    }
  };

  // Focus textarea on edit
  useEffect(() => {
    if (editing && textareaRef.current) {
      textareaRef.current.focus();
      textareaRef.current.selectionStart = textareaRef.current.value.length;
    }
  }, [editing]);

  if (isLoading) return null;

  if (!concept) {
    return (
      <div className="flex flex-col items-center justify-center gap-2 py-20">
        <span
          style={{
            fontFamily: "var(--mem-font-body)",
            fontSize: "13px",
            color: "var(--mem-text-tertiary)",
          }}
        >
          Concept not found
        </span>
        <button
          onClick={onBack}
          className="transition-colors text-sm"
          style={{ color: "var(--mem-text-secondary)" }}
        >
          Back
        </button>
      </div>
    );
  }

  const sourceCount = conceptSources?.length ?? concept.source_memory_ids.length;
  const relatedConcepts = parseRelatedConcepts(concept.content);

  // Strip ## Sources (shown as MemoryCard UI below)
  // Convert [[wikilinks]] to markdown links if they resolve to concepts, else plain text
  const displayContent = concept.content
    .replace(/## Sources\n[\s\S]*?(?=\n## |\s*$)/, "")
    .replace(/\[\[([^\]]+)\]\]/g, (_match, title) => {
      const cid = conceptLookup.get(title.toLowerCase());
      if (cid) return `[${title}](#concept:${cid})`;
      return title;
    })
    .trim();

  // Intercept concept/memory link clicks in rendered content (capture phase beats target="_blank")
  const handleContentClick = (e: React.MouseEvent) => {
    const anchor = (e.target as HTMLElement).closest("a");
    if (!anchor) return;
    const href = anchor.getAttribute("href") || "";
    if (href.startsWith("#concept:")) {
      e.preventDefault();
      e.stopPropagation();
      onConceptClick?.(href.replace("#concept:", ""));
    } else if (href.startsWith("#memory:")) {
      e.preventDefault();
      e.stopPropagation();
      onMemoryClick(href.replace("#memory:", ""));
    }
  };

  // Only show Related Concepts for wikilinks that resolve to actual concepts
  const resolvedConcepts = relatedConcepts
    .map((rc) => ({ title: rc.title, id: conceptLookup.get(rc.title.toLowerCase()) }))
    .filter((rc): rc is { title: string; id: string } => rc.id != null);

  return (
    <div className="flex flex-col gap-6">
      {/* Back + Header */}
      <div>
        <button
          onClick={onBack}
          className="p-1.5 -ml-1.5 rounded-md transition-colors duration-150 hover:bg-[var(--mem-hover)]"
          style={{ color: "var(--mem-text-tertiary)", background: "none", border: "none", cursor: "pointer", lineHeight: 0, marginBottom: "12px" }}
        >
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
              {concept.title}
            </h2>
            <div
              className="flex items-center gap-2 mt-1.5 flex-wrap"
              style={{ fontFamily: "var(--mem-font-mono)", fontSize: "11px", color: "var(--mem-text-tertiary)" }}
            >
              <span>Last distilled {relativeTimeFromISO(concept.last_compiled)}</span>
              <span style={{ opacity: 0.4 }}>&middot;</span>
              <span>from {sourceCount} {sourceCount === 1 ? "memory" : "memories"}</span>
              {concept.stale_reason && (
                <>
                  <span style={{ opacity: 0.4 }}>&middot;</span>
                  <span style={{ color: concept.stale_reason === "source_conflict" ? "var(--mem-accent-amber)" : "var(--mem-text-tertiary)" }}>
                    {concept.stale_reason === "source_conflict" ? "needs review" : "updating..."}
                  </span>
                </>
              )}
            </div>
          </div>

          {/* Actions — icon-only for clean top bar */}
          <div className="flex items-center gap-0.5 shrink-0">
            {!editing && (
              <button
                onClick={() => { setEditContent(concept.content); setEditing(true); }}
                className="p-1.5 rounded-md transition-colors duration-150 hover:bg-[var(--mem-hover-strong)]"
                style={{ color: "var(--mem-text-tertiary)" }}
                title="Edit concept"
              >
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <path d="M11 4H4a2 2 0 00-2 2v14a2 2 0 002 2h14a2 2 0 002-2v-7" />
                  <path d="M18.5 2.5a2.121 2.121 0 013 3L12 15l-4 1 1-4 9.5-9.5z" />
                </svg>
              </button>
            )}
            <button
              onClick={copyAsContext}
              className={`p-1.5 rounded-md transition-colors duration-150 ${
                copied ? "text-emerald-400" : "hover:bg-[var(--mem-hover-strong)]"
              }`}
              style={copied ? undefined : { color: "var(--mem-text-tertiary)" }}
              title={copied ? "Copied!" : "Copy as context"}
            >
              {copied ? (
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <polyline points="20 6 9 17 4 12" />
                </svg>
              ) : (
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <rect x="9" y="9" width="13" height="13" rx="2" ry="2" />
                  <path d="M5 15H4a2 2 0 01-2-2V4a2 2 0 012-2h9a2 2 0 012 2v1" />
                </svg>
              )}
            </button>
            {/* Export button: 0 sources = disabled, 1 = direct, 2+ = popover */}
            <div className="relative">
              {obsidianSources.length === 0 ? (
                <button
                  disabled
                  className="p-1.5 rounded-md opacity-40 cursor-not-allowed"
                  style={{ color: "var(--mem-text-tertiary)" }}
                  title="Add an Obsidian source in Settings to export"
                >
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4" />
                    <polyline points="7 10 12 15 17 10" />
                    <line x1="12" y1="15" x2="12" y2="3" />
                  </svg>
                </button>
              ) : (
                <button
                  onClick={() => {
                    if (obsidianSources.length === 1) {
                      handleExportToVault(obsidianSources[0].path);
                    } else {
                      setExportMenuOpen((v) => !v);
                    }
                  }}
                  className={`p-1.5 rounded-md transition-colors duration-150 ${
                    exported ? "text-emerald-400" : "hover:bg-[var(--mem-hover-strong)]"
                  }`}
                  style={exported ? undefined : { color: "var(--mem-text-tertiary)" }}
                  title={exported ? "Exported!" : "Export to Obsidian"}
                >
                  {exported ? (
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                      <polyline points="20 6 9 17 4 12" />
                    </svg>
                  ) : (
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                      <path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4" />
                      <polyline points="7 10 12 15 17 10" />
                      <line x1="12" y1="15" x2="12" y2="3" />
                    </svg>
                  )}
                </button>
              )}
              {exportMenuOpen && obsidianSources.length >= 2 && (
                <div
                  className="absolute right-0 top-full mt-1 z-50 rounded-lg py-1 min-w-[180px]"
                  style={{
                    backgroundColor: "var(--mem-surface)",
                    border: "1px solid var(--mem-border)",
                    boxShadow: "0 4px 12px rgba(0,0,0,0.3)",
                  }}
                >
                  {obsidianSources.map((s) => (
                    <button
                      key={s.id}
                      onClick={() => handleExportToVault(s.path)}
                      className="w-full text-left px-3 py-2 text-[13px] transition-colors hover:bg-[var(--mem-hover)]"
                      style={{ color: "var(--mem-text)", fontFamily: "var(--mem-font-body)" }}
                    >
                      {folderName(s.path)}
                    </button>
                  ))}
                </div>
              )}
            </div>
            <button
              onClick={() => { if (confirm("Delete this concept?")) deleteMutation.mutate(); }}
              className="p-1.5 rounded-md transition-colors duration-150 hover:bg-red-500/10"
              style={{ color: "var(--mem-text-tertiary)" }}
              title="Delete concept"
            >
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M3 6h18M8 6V4a2 2 0 012-2h4a2 2 0 012 2v2m3 0v14a2 2 0 01-2 2H7a2 2 0 01-2-2V6h14" />
              </svg>
            </button>
          </div>
        </div>
      </div>

      {/* Summary / TLDR */}
      {concept.summary && !editing && (
        <div
          className="pl-4 py-2"
          style={{ borderLeft: "3px solid var(--mem-accent-concept)" }}
        >
          <p
            style={{
              fontFamily: "var(--mem-font-body)",
              fontSize: "15px",
              color: "var(--mem-text)",
              lineHeight: "1.7",
              fontStyle: "italic",
            }}
          >
            {concept.summary}
          </p>
        </div>
      )}

      {/* Content — edit mode or rendered */}
      {editing ? (
        <div className="flex flex-col gap-2">
          <textarea
            ref={textareaRef}
            value={editContent}
            onChange={(e) => setEditContent(e.target.value)}
            onKeyDown={handleKeyDown}
            className="w-full rounded-lg p-4 resize-y outline-none"
            style={{
              minHeight: "300px",
              backgroundColor: "var(--mem-surface)",
              border: "1px solid var(--mem-border)",
              color: "var(--mem-text)",
              fontFamily: "var(--mem-font-mono)",
              fontSize: "13px",
              lineHeight: "1.6",
            }}
          />
          <div className="flex items-center gap-2">
            <button
              onClick={handleSave}
              disabled={updateMutation.isPending}
              className="text-[11px] font-medium px-3 py-1.5 rounded-md transition-all"
              style={{ backgroundColor: "rgba(99, 102, 241, 0.15)", color: "var(--mem-accent-concept)" }}
            >
              {updateMutation.isPending ? "Saving..." : "Save (Cmd+Enter)"}
            </button>
            <button
              onClick={() => setEditing(false)}
              className="text-[11px] font-medium px-3 py-1.5 rounded-md transition-all hover:bg-[var(--mem-hover-strong)]"
              style={{ color: "var(--mem-text-tertiary)" }}
            >
              Cancel (Esc)
            </button>
          </div>
        </div>
      ) : (
        <div onClickCapture={handleContentClick}>
          <ContentRenderer content={displayContent} variant="detail" />
        </div>
      )}

      {/* Related Concepts — from [[wikilinks]] in content */}
      {!editing && resolvedConcepts.length > 0 && (
        <div>
          <h3
            className="mb-2"
            style={{
              fontFamily: "var(--mem-font-mono)",
              fontSize: "11px",
              fontWeight: 600,
              letterSpacing: "0.05em",
              textTransform: "uppercase",
              color: "var(--mem-text-tertiary)",
            }}
          >
            Related Concepts
          </h3>
          <div className="flex flex-col gap-1.5">
            {resolvedConcepts.map((rc) => (
              <button
                key={rc.id}
                onClick={() => onConceptClick?.(rc.id)}
                className="w-full text-left rounded-lg px-4 py-3 transition-colors duration-150 cursor-pointer hover:bg-[var(--mem-hover)] group"
                style={{ backgroundColor: "var(--mem-surface)", border: "1px solid var(--mem-border)" }}
              >
                <div className="flex items-center gap-2">
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" style={{ color: "var(--mem-accent-concept)" }} className="shrink-0">
                    <path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5" />
                  </svg>
                  <span
                    className="group-hover:text-[var(--mem-accent-indigo)] transition-colors"
                    style={{ fontFamily: "var(--mem-font-body)", fontSize: "13px", fontWeight: 500, color: "var(--mem-text)" }}
                  >
                    {rc.title}
                  </span>
                </div>
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Source Memories: evidence cards */}
      {!editing && sourceCount > 0 && (
        <div>
          <h3
            className="mb-1"
            style={{
              fontFamily: "var(--mem-font-mono)",
              fontSize: "11px",
              fontWeight: 600,
              letterSpacing: "0.05em",
              textTransform: "uppercase",
              color: "var(--mem-text-tertiary)",
            }}
          >
            Source Memories ({sourceCount})
          </h3>
          <ul>
            {sourceMemories?.map((mem, idx) => (
              <EvidenceCard
                key={mem.source_id}
                mem={mem}
                isLast={idx === (sourceMemories?.length ?? 0) - 1}
                onClick={onMemoryClick}
              />
            ))}
            {/* While loading show placeholders matching source count */}
            {!sourceMemories &&
              concept.source_memory_ids.map((id) => (
                <li
                  key={id}
                  style={{
                    borderBottom: "1px solid color-mix(in srgb, var(--mem-border) 60%, transparent)",
                    padding: "10px 8px",
                    opacity: 0.4,
                  }}
                >
                  <div style={{ width: "60%", height: "10px", background: "var(--mem-hover)", borderRadius: "4px" }} />
                </li>
              ))
            }
          </ul>
        </div>
      )}
    </div>
  );
}

// Evidence card for a single source memory.
// Hairline border-bottom row, hover background shift, click opens Memory Log.
function EvidenceCard({
  mem,
  isLast,
  onClick,
}: {
  mem: MemoryItem;
  isLast: boolean;
  onClick: (sourceId: string) => void;
}) {
  const [hover, setHover] = useState(false);
  const ts = mem.last_modified ? relativeMs(mem.last_modified * 1000) : null;
  const agent = mem.source_agent ? prettyAgent(mem.source_agent) : null;
  const kind = sourceKindLabel(mem);
  const snippet = mem.content
    ? mem.content.replace(/\s+/g, " ").trim().slice(0, 160)
    : null;

  return (
    <li
      className="py-3 px-2 transition-colors duration-150"
      style={{
        backgroundColor: hover ? "var(--mem-hover)" : "transparent",
        borderBottom: isLast
          ? "none"
          : "1px solid color-mix(in srgb, var(--mem-border) 60%, transparent)",
        cursor: "pointer",
      }}
      onMouseEnter={() => setHover(true)}
      onMouseLeave={() => setHover(false)}
      onClick={() => onClick(mem.source_id)}
    >
      <div className="flex items-center gap-2 mb-0.5">
        {ts && (
          <span
            style={{
              fontFamily: "var(--mem-font-body)",
              fontSize: "11px",
              color: "var(--mem-text-tertiary)",
              whiteSpace: "nowrap",
            }}
          >
            {ts}
          </span>
        )}
        {agent && (
          <span
            style={{
              fontFamily: "var(--mem-font-body)",
              fontSize: "11px",
              fontWeight: 500,
              color: "var(--mem-text-secondary)",
              whiteSpace: "nowrap",
            }}
          >
            {agent}
          </span>
        )}
        <span
          style={{
            fontFamily: "var(--mem-font-mono)",
            fontSize: "10px",
            color: "var(--mem-text-tertiary)",
            background: "var(--mem-hover)",
            padding: "1px 5px",
            borderRadius: "3px",
            whiteSpace: "nowrap",
          }}
        >
          {kind}
        </span>
      </div>
      {mem.title && (
        <p
          className="truncate"
          style={{
            fontFamily: "var(--mem-font-heading)",
            fontSize: "13px",
            fontWeight: 500,
            color: "var(--mem-text)",
            lineHeight: 1.4,
          }}
        >
          {mem.title}
        </p>
      )}
      {snippet && (
        <p
          className="line-clamp-2"
          style={{
            fontFamily: "var(--mem-font-body)",
            fontSize: "12px",
            color: "var(--mem-text-secondary)",
            lineHeight: 1.5,
            marginTop: "2px",
          }}
        >
          {snippet}
        </p>
      )}
    </li>
  );
}
