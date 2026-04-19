// SPDX-License-Identifier: AGPL-3.0-only
import { useState } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { correctMemory, updateMemory, type MemoryItem } from "../../lib/tauri";

interface NurtureCardProps {
  memory: MemoryItem;
  onConfirm: (sourceId: string) => void;
  onDismiss: () => void;
  onDelete: (sourceId: string) => void;
}

function smartSummary(memory: MemoryItem): string {
  // 1. Try structured field values
  if (memory.structured_fields) {
    try {
      const fields = JSON.parse(memory.structured_fields);
      const primary = fields.decision || fields.claim || fields.preference || fields.objective;
      if (primary && typeof primary === "string" && primary.length > 5) {
        // Strip field-name prefix if present (e.g. "claim: ...")
        return primary.replace(/^\w+:\s*/i, "");
      }
    } catch { /* fall through */ }
  }
  // 2. First sentence (keep short)
  const dot = memory.content.indexOf(". ");
  if (dot > 10 && dot < 80) return memory.content.slice(0, dot + 1);
  // 3. Word-boundary truncation at 80 chars
  if (memory.content.length <= 80) return memory.content;
  const cut = memory.content.lastIndexOf(" ", 80);
  return memory.content.slice(0, cut > 40 ? cut : 80).replace(/[.\u2026]+$/, "").trim() + "\u2026";
}

export default function NurtureCard({ memory, onConfirm, onDismiss: _onDismiss, onDelete }: NurtureCardProps) {
  const queryClient = useQueryClient();
  const [expanded, setExpanded] = useState(false);
  const [showCorrection, setShowCorrection] = useState(false);
  const [editContent, setEditContent] = useState(memory.content);
  const [correctionText, setCorrectionText] = useState("");
  const [dismissState, setDismissState] = useState<null | "confirmed" | "out">(null);

  const correctionMutation = useMutation({
    mutationFn: (prompt: string) => correctMemory(memory.source_id, prompt),
    onSuccess: (corrected) => {
      setEditContent(corrected);
      setCorrectionText("");
    },
  });

  const handleConfirm = async () => {
    if (editContent !== memory.content) {
      await updateMemory(memory.source_id, editContent);
    }
    onConfirm(memory.source_id);
    // Phase 1: brief "confirmed" flash
    setDismissState("confirmed");
    // Phase 2: slide out
    setTimeout(() => setDismissState("out"), 400);
    // Phase 3: invalidate after animation
    setTimeout(() => {
      queryClient.invalidateQueries({ queryKey: ["nurture-cards"] });
      queryClient.invalidateQueries({ queryKey: ["memories"] });
      queryClient.invalidateQueries({ queryKey: ["memoryStats"] });
    }, 800);
  };

  const handleDelete = () => {
    onDelete(memory.source_id);
    queryClient.invalidateQueries({ queryKey: ["nurture-cards"] });
  };

  const summary = smartSummary(memory);
  // For expanded view, prefer source_text (clean prose) over content (pipe-delimited)
  const fullText = memory.source_text
    || memory.content.split(" | ").filter((s) => !s.match(/^(domain|source|verified|memory_type|entity):/i)).join(". ").replace(/\.\./g, ".");
  // Only show toggle when expanded text adds meaningful content beyond the summary
  const hasMore = fullText.length > summary.length + 15;
  const facet = memory.memory_type ?? "fact";
  const domain = memory.domain;

  return (
    <div
      className="rounded-lg overflow-hidden"
      style={{
        border: dismissState === "confirmed"
          ? "1px solid var(--mem-accent-warm)"
          : "1px solid var(--mem-border)",
        backgroundColor: dismissState === "confirmed"
          ? "var(--mem-confirm-bg)"
          : "var(--mem-surface)",
        opacity: dismissState === "out" ? 0 : 1,
        transform: dismissState === "out"
          ? "translateY(-12px) scale(0.97)"
          : dismissState === "confirmed"
            ? "scale(1.005)"
            : "none",
        transition: dismissState === "out"
          ? "opacity 350ms ease-out, transform 350ms ease-out, border-color 200ms, background-color 200ms"
          : "opacity 200ms, transform 200ms ease-out, border-color 300ms, background-color 300ms",
      }}
    >
      <div className="px-4 py-3">
        {/* Type badge + domain */}
        <div className="flex items-center gap-2 mb-2">
          <span style={{ fontFamily: "var(--mem-font-mono)", fontSize: "10px", color: "var(--mem-text-tertiary)" }}>
            {facet}
          </span>
          {domain && (
            <>
              <span style={{ color: "var(--mem-text-tertiary)", fontSize: "10px" }}>&middot;</span>
              <span style={{ fontFamily: "var(--mem-font-mono)", fontSize: "10px", color: "var(--mem-text-tertiary)" }}>
                {domain}
              </span>
            </>
          )}
          {/* Confirmed flash badge */}
          {dismissState === "confirmed" && (
            <span
              style={{
                marginLeft: "auto",
                fontFamily: "var(--mem-font-mono)",
                fontSize: "10px",
                color: "var(--mem-accent-warm)",
                animation: "mem-fade-up 200ms ease-out",
              }}
            >
              &#10003; confirmed
            </span>
          )}
        </div>

        {/* Smart summary / full content */}
        {!showCorrection && (
          <div>
            <p
              style={{
                fontFamily: "var(--mem-font-body)",
                fontSize: "13px",
                color: "var(--mem-text)",
                lineHeight: "1.6",
                margin: 0,
              }}
            >
              {expanded ? fullText : summary}
            </p>
            {hasMore && (
              <button
                onClick={() => setExpanded(!expanded)}
                className="flex items-center gap-1 mt-1.5 rounded px-2 py-0.5 transition-colors duration-150 hover:bg-[var(--mem-hover)]"
                style={{
                  fontFamily: "var(--mem-font-body)",
                  fontSize: "11px",
                  color: "var(--mem-text-secondary)",
                  background: "none",
                  border: "none",
                  cursor: "pointer",
                  padding: "2px 0",
                }}
              >
                <svg
                  width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5"
                  style={{ transition: "transform 200ms", transform: expanded ? "rotate(180deg)" : "none" }}
                >
                  <polyline points="6 9 12 15 18 9" />
                </svg>
                <span>{expanded ? "Show less" : "Show more"}</span>
              </button>
            )}
          </div>
        )}

        {/* Correction flow */}
        {showCorrection && (
          <div className="flex flex-col gap-2 mt-1">
            <textarea
              value={editContent}
              onChange={(e) => setEditContent(e.target.value)}
              rows={3}
              className="w-full rounded-md px-3 py-2 outline-none resize-none"
              style={{ fontFamily: "var(--mem-font-body)", fontSize: "13px", color: "var(--mem-text)", backgroundColor: "var(--mem-hover)", border: "1px solid var(--mem-border)" }}
            />
            <div className="flex items-center gap-2">
              <input
                value={correctionText}
                onChange={(e) => setCorrectionText(e.target.value)}
                onKeyDown={(e) => { if (e.key === "Enter" && correctionText.trim()) correctionMutation.mutate(correctionText.trim()); }}
                placeholder="Describe what's wrong..."
                className="flex-1 rounded-md px-3 py-1.5 outline-none"
                style={{ fontFamily: "var(--mem-font-body)", fontSize: "12px", color: "var(--mem-text)", backgroundColor: "var(--mem-hover)", border: "1px solid var(--mem-border)" }}
              />
              {correctionMutation.isPending && (
                <span style={{ fontSize: "11px", color: "var(--mem-text-tertiary)" }}>fixing...</span>
              )}
            </div>
          </div>
        )}

        {/* Actions */}
        <div className="flex items-center gap-3 mt-3">
          {!showCorrection ? (
            <>
              <button
                onClick={handleConfirm}
                disabled={dismissState !== null}
                className="px-3 py-1 rounded-full transition-colors duration-150"
                style={{ fontFamily: "var(--mem-font-body)", fontSize: "12px", fontWeight: 500, color: "var(--mem-accent-warm)", border: "1px solid var(--mem-accent-warm)", background: "none", cursor: "pointer" }}
              >
                Yes, that&apos;s right
              </button>
              <button
                onClick={() => setShowCorrection(true)}
                className="px-3 py-1 rounded-full transition-colors duration-150"
                style={{ fontFamily: "var(--mem-font-body)", fontSize: "12px", color: "var(--mem-text-tertiary)", border: "1px solid var(--mem-border)", background: "none", cursor: "pointer" }}
              >
                Not quite&hellip;
              </button>
            </>
          ) : (
            <>
              <button
                onClick={handleConfirm}
                disabled={dismissState !== null}
                className="px-3 py-1 rounded-full transition-colors duration-150"
                style={{ fontFamily: "var(--mem-font-body)", fontSize: "12px", fontWeight: 500, color: "var(--mem-accent-warm)", border: "1px solid var(--mem-accent-warm)", background: "none", cursor: "pointer" }}
              >
                Save &amp; Confirm
              </button>
              <button
                onClick={handleDelete}
                className="px-3 py-1 rounded-full transition-colors duration-150"
                style={{ fontFamily: "var(--mem-font-body)", fontSize: "12px", color: "var(--mem-text-tertiary)", border: "1px solid var(--mem-border)", background: "none", cursor: "pointer" }}
              >
                Delete
              </button>
              <button
                onClick={() => { setShowCorrection(false); setEditContent(memory.content); setCorrectionText(""); }}
                style={{ fontFamily: "var(--mem-font-body)", fontSize: "12px", color: "var(--mem-text-tertiary)", background: "none", border: "none", cursor: "pointer", marginLeft: "auto" }}
              >
                Cancel
              </button>
            </>
          )}
        </div>
      </div>
    </div>
  );
}
