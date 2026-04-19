// SPDX-License-Identifier: AGPL-3.0-only
import { useState, useRef, useEffect } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import {
  getSpace,
  updateSpace,
  deleteSpace,
  confirmSpace,
  listMemoriesRich,
  listEntities,
  getNurtureCards,
  setStability,
  deleteFileChunks,
  listConcepts,
} from "../../lib/tauri";
import MemoryStream from "./MemoryStream";
import type { SortMode } from "./MemoryStream";
import NurtureCard from "./NurtureCard";

interface SpaceDetailProps {
  spaceName: string;
  onBack: () => void;
  onSelectMemory: (sourceId: string) => void;
  onSelectConcept: (conceptId: string) => void;
  onEntityClick: (entityId: string) => void;
}

export default function SpaceDetail({
  spaceName,
  onBack,
  onSelectMemory,
  onSelectConcept,
  onEntityClick,
}: SpaceDetailProps) {
  const queryClient = useQueryClient();
  const [editingName, setEditingName] = useState(false);
  const [editingDesc, setEditingDesc] = useState(false);
  const [nameValue, setNameValue] = useState("");
  const [descValue, setDescValue] = useState("");
  const [deleteStep, setDeleteStep] = useState<null | "options">(null);
  const [sortMode, setSortMode] = useState<SortMode>("curated");
  const [showAllEntities, setShowAllEntities] = useState(false);
  const nameRef = useRef<HTMLInputElement>(null);
  const descRef = useRef<HTMLTextAreaElement>(null);
  const chipsRef = useRef<HTMLDivElement>(null);
  const [hasChipsOverflow, setHasChipsOverflow] = useState(false);
  // Measured height of a single chip in the current font/zoom, used as the
  // collapsed `maxHeight`. A static 28px worked for default root font size
  // but drifts if the user bumps OS/accessibility font scaling. Falls back
  // to 28 on the first render before measurement lands.
  const [collapsedChipHeight, setCollapsedChipHeight] = useState(28);

  const { data: space } = useQuery({
    queryKey: ["space", spaceName],
    queryFn: () => getSpace(spaceName),
    refetchInterval: 5000,
  });

  const { data: memories = [] } = useQuery({
    queryKey: ["space-memories", spaceName],
    queryFn: () => listMemoriesRich(spaceName, undefined, undefined, 200),
    refetchInterval: 5000,
  });

  const { data: entities = [] } = useQuery({
    queryKey: ["space-entities", spaceName],
    queryFn: () => listEntities(undefined, spaceName),
    refetchInterval: 10000,
  });

  const { data: concepts = [] } = useQuery({
    queryKey: ["space-concepts", spaceName],
    queryFn: () => listConcepts("active", spaceName, 50),
    refetchInterval: 10000,
  });

  const [memoriesExpanded, setMemoriesExpanded] = useState(false);

  const invalidate = () => {
    queryClient.invalidateQueries({ queryKey: ["space", spaceName] });
    queryClient.invalidateQueries({ queryKey: ["spaces"] });
    queryClient.invalidateQueries({ queryKey: ["space-memories", spaceName] });
    queryClient.invalidateQueries({ queryKey: ["space-entities", spaceName] });
    queryClient.invalidateQueries({ queryKey: ["memories"] });
  };

  const renameMutation = useMutation({
    mutationFn: ({ newName, desc }: { newName: string; desc?: string }) =>
      updateSpace(spaceName, newName, desc),
    onSuccess: (_data, { newName }) => {
      invalidate();
      if (newName !== spaceName) onBack();
    },
  });

  const deleteMutation = useMutation({
    mutationFn: (memoryAction: string) => deleteSpace(spaceName, memoryAction),
    onSuccess: () => {
      invalidate();
      onBack();
    },
  });

  const confirmMutation = useMutation({
    mutationFn: () => confirmSpace(spaceName),
    onSuccess: () => invalidate(),
  });

  // Nurture cards hidden — confirmation moving to concept level
  const SHOW_NURTURE = false;
  const { data: nurtureCards = [] } = useQuery({
    queryKey: ["nurture-cards", space?.name],
    queryFn: () => getNurtureCards(2, space?.name),
    enabled: SHOW_NURTURE && !!space?.name,
    refetchInterval: 30000,
  });

  const nurtureConfirmMutation = useMutation({
    mutationFn: (sourceId: string) => setStability(sourceId, "confirmed"),
    // NurtureCard controls invalidation timing for dismiss animation
  });

  const nurtureDeleteMutation = useMutation({
    mutationFn: (sourceId: string) => deleteFileChunks("memory", sourceId),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["nurture-cards"] }),
  });

  // Auto-focus inputs when entering edit mode
  useEffect(() => {
    if (editingName && nameRef.current) {
      nameRef.current.focus();
      nameRef.current.select();
    }
  }, [editingName]);

  useEffect(() => {
    if (editingDesc && descRef.current) {
      descRef.current.focus();
      descRef.current.setSelectionRange(descRef.current.value.length, descRef.current.value.length);
    }
  }, [editingDesc]);

  useEffect(() => {
    if (chipsRef.current) {
      // Measure the first chip's rendered height so the collapsed maxHeight
      // matches the actual chip size in the current font/zoom.
      const firstChip = chipsRef.current.querySelector("button") as HTMLElement | null;
      if (firstChip && firstChip.offsetHeight > 0) {
        setCollapsedChipHeight(firstChip.offsetHeight);
      }
      setHasChipsOverflow(chipsRef.current.scrollHeight > chipsRef.current.clientHeight);
    }
  }, [entities]);

  if (!space) return null;

  const startEditName = () => {
    setNameValue(space.name);
    setEditingName(true);
  };

  const saveName = () => {
    const trimmed = nameValue.trim();
    if (trimmed && trimmed !== space.name) {
      renameMutation.mutate({ newName: trimmed, desc: space.description ?? undefined });
    }
    setEditingName(false);
  };

  const startEditDesc = () => {
    setDescValue(space.description ?? "");
    setEditingDesc(true);
  };

  const saveDesc = () => {
    renameMutation.mutate({ newName: space.name, desc: descValue.trim() || undefined });
    setEditingDesc(false);
  };

  return (
    <div
      className="flex flex-col"
      style={{ animation: "mem-fade-up 350ms cubic-bezier(0.16, 1, 0.3, 1) both" }}
    >
      {/* ── Header: back button ── */}
      <div className="flex items-center mb-3">
        <button onClick={onBack} className="p-1.5 -ml-1.5 rounded-md transition-colors duration-150 hover:bg-[var(--mem-hover)]" style={{ color: "var(--mem-text-tertiary)", background: "none", border: "none", cursor: "pointer", lineHeight: 0 }}>
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5"><path d="M19 12H5M12 19l-7-7 7-7" /></svg>
        </button>
      </div>

      {/* ── Suggested banner ── */}
      {space.suggested && (
        <div
          className="flex items-center justify-between rounded-lg px-4 py-3 mb-6"
          style={{
            backgroundColor: "rgba(217, 168, 83, 0.06)",
            border: "1px solid rgba(217, 168, 83, 0.15)",
            animation: "mem-fade-up 350ms cubic-bezier(0.16, 1, 0.3, 1) 60ms both",
          }}
        >
          <div className="flex items-center gap-2.5">
            <span
              className="w-2 h-2 rounded-full shrink-0"
              style={{ backgroundColor: "var(--mem-accent-indigo)" }}
            />
            <span style={{ fontFamily: "var(--mem-font-body)", fontSize: "13px", color: "var(--mem-text-secondary)" }}>
              Auto-created by an agent
            </span>
          </div>
          <div className="flex gap-2">
            <button
              onClick={() => confirmMutation.mutate()}
              className="px-3 py-1 rounded-md text-xs font-medium transition-all duration-150 hover:brightness-110"
              style={{ backgroundColor: "var(--mem-accent-sage)", color: "white" }}
            >
              Keep
            </button>
            <button
              onClick={() => deleteMutation.mutate("unassign")}
              className="px-3 py-1 rounded-md text-xs transition-colors duration-150 hover:bg-red-500/10"
              style={{ color: "var(--mem-text-tertiary)" }}
            >
              Discard
            </button>
          </div>
        </div>
      )}

      {/* ── Space identity: title + actions ── */}
      <div
        className="flex items-start justify-between gap-4 mb-6"
        style={{ animation: "mem-fade-up 350ms cubic-bezier(0.16, 1, 0.3, 1) 80ms both" }}
      >
        <div className="flex-1 min-w-0">
          {editingName ? (
            <input
              ref={nameRef}
              value={nameValue}
              onChange={(e) => setNameValue(e.target.value)}
              onBlur={saveName}
              onKeyDown={(e) => {
                if (e.key === "Enter") saveName();
                if (e.key === "Escape") setEditingName(false);
              }}
              className="w-full bg-transparent outline-none"
              style={{
                fontFamily: "var(--mem-font-heading)",
                fontSize: "24px",
                fontWeight: 500,
                color: "var(--mem-text)",
                borderBottom: "1px solid var(--mem-accent-indigo)",
                paddingBottom: "4px",
              }}
            />
          ) : (
            <h1
              onClick={startEditName}
              className="capitalize cursor-pointer"
              style={{
                fontFamily: "var(--mem-font-heading)",
                fontSize: "24px",
                fontWeight: 500,
                color: "var(--mem-text)",
              }}
            >
              {space.name}
            </h1>
          )}
        </div>

        {/* Actions: edit + delete */}
        {!space.suggested && (
          <div className="flex items-center gap-0.5 shrink-0 pt-1">
            <button
              onClick={startEditDesc}
              className="p-1.5 rounded-md transition-colors duration-150 hover:bg-[var(--mem-hover)]"
              style={{ color: "var(--mem-text-tertiary)" }}
              title="Edit description"
            >
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M17 3a2.85 2.85 0 114 4L7.5 20.5 2 22l1.5-5.5L17 3z" />
              </svg>
            </button>
            <button
              onClick={() => setDeleteStep("options")}
              className="p-1.5 rounded-md transition-colors duration-150 hover:bg-[var(--mem-hover)]"
              style={{ color: "var(--mem-text-tertiary)" }}
              title="Delete space"
            >
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M3 6h18M8 6V4a2 2 0 012-2h4a2 2 0 012 2v2m3 0v14a2 2 0 01-2 2H7a2 2 0 01-2-2V6h14" />
              </svg>
            </button>
          </div>
        )}
      </div>

      {/* ── Delete confirmation ── */}
      {deleteStep === "options" && (
        <div
          className="flex items-center gap-2 rounded-lg px-3 py-2 mb-4"
          style={{
            backgroundColor: "rgba(239, 68, 68, 0.04)",
            border: "1px solid rgba(239, 68, 68, 0.12)",
            animation: "mem-fade-up 200ms ease both",
          }}
        >
          <span style={{ fontFamily: "var(--mem-font-body)", fontSize: "12px", color: "var(--mem-text-secondary)" }}>
            Keep {space.memory_count} {space.memory_count === 1 ? "memory" : "memories"}?
          </span>
          <button
            onClick={() => deleteMutation.mutate("unassign")}
            className="px-2 py-0.5 rounded text-xs font-medium transition-colors duration-150 hover:brightness-110"
            style={{ backgroundColor: "var(--mem-accent-sage)", color: "white" }}
          >
            Keep
          </button>
          <button
            onClick={() => deleteMutation.mutate("delete")}
            className="px-2 py-0.5 rounded text-xs font-medium transition-colors duration-150 hover:bg-red-500/15"
            style={{ color: "#ef4444" }}
          >
            Delete
          </button>
          <button
            onClick={() => setDeleteStep(null)}
            className="px-2 py-0.5 rounded text-xs transition-colors duration-150 hover:bg-[var(--mem-hover)]"
            style={{ color: "var(--mem-text-tertiary)" }}
          >
            Cancel
          </button>
        </div>
      )}

      {/* ── Description ── */}
      {(space.description || editingDesc) && (
        <div
          className="rounded-lg px-4 py-3 mb-4"
          style={{
            backgroundColor: "var(--mem-surface)",
            border: `1px solid ${editingDesc ? "var(--mem-accent-indigo)" : "var(--mem-border)"}`,
            animation: "mem-fade-up 350ms cubic-bezier(0.16, 1, 0.3, 1) 100ms both",
            transition: "border-color 150ms ease",
          }}
        >
          {editingDesc ? (
            <div>
              <textarea
                ref={descRef}
                value={descValue}
                onChange={(e) => setDescValue(e.target.value)}
                onKeyDown={(e) => {
                  if (e.metaKey && e.key === "Enter") { e.preventDefault(); saveDesc(); }
                  if (e.key === "Escape") setEditingDesc(false);
                }}
                placeholder="Add a description..."
                className="w-full bg-transparent resize-none outline-none leading-relaxed"
                style={{
                  color: "var(--mem-text)",
                  fontFamily: "var(--mem-font-body)",
                  fontSize: "13px",
                  lineHeight: "1.6",
                }}
                rows={Math.max(2, descValue.split("\n").length)}
              />
              <div
                className="flex items-center gap-2 mt-2 pt-2"
                style={{ borderTop: "1px solid var(--mem-border)" }}
              >
                <button
                  onClick={saveDesc}
                  className="px-2.5 py-1 rounded text-xs font-medium transition-colors duration-150"
                  style={{ backgroundColor: "var(--mem-accent-warm)", color: "white" }}
                >
                  Save
                </button>
                <button
                  onClick={() => setEditingDesc(false)}
                  className="px-2.5 py-1 rounded text-xs transition-colors duration-150"
                  style={{ color: "var(--mem-text-tertiary)" }}
                >
                  Cancel
                </button>
                <span
                  className="ml-auto"
                  style={{ fontFamily: "var(--mem-font-mono)", fontSize: "10px", color: "var(--mem-text-tertiary)" }}
                >
                  Cmd+Enter
                </span>
              </div>
            </div>
          ) : (
            <p
              style={{
                fontFamily: "var(--mem-font-body)",
                fontSize: "13px",
                lineHeight: "1.6",
                color: "var(--mem-text-secondary)",
                margin: 0,
              }}
            >
              {space.description}
            </p>
          )}
        </div>
      )}

      {/* ── Stats strip ── */}
      <div
        className="flex items-center gap-5 pb-3 mb-0"
        style={{
          borderBottom: "none",
          animation: "mem-fade-up 350ms cubic-bezier(0.16, 1, 0.3, 1) 120ms both",
        }}
      >
        <div className="flex items-center gap-1.5">
          <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" style={{ color: "var(--mem-text-tertiary)" }}>
            <circle cx="12" cy="12" r="10" />
            <path d="M12 6v6l4 2" />
          </svg>
          <span style={{ fontFamily: "var(--mem-font-mono)", fontSize: "11px", color: "var(--mem-text-tertiary)" }}>
            {space.memory_count} {space.memory_count === 1 ? "memory" : "memories"}
          </span>
        </div>
        {space.entity_count > 0 && (
          <div className="flex items-center gap-1.5">
            <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" style={{ color: "var(--mem-text-tertiary)" }}>
              <path d="M16 21v-2a4 4 0 00-4-4H6a4 4 0 00-4-4v2" />
              <circle cx="9" cy="7" r="4" />
              <path d="M22 21v-2a4 4 0 00-3-3.87" />
              <path d="M16 3.13a4 4 0 010 7.75" />
            </svg>
            <span style={{ fontFamily: "var(--mem-font-mono)", fontSize: "11px", color: "var(--mem-text-tertiary)" }}>
              {space.entity_count} {space.entity_count === 1 ? "entity" : "entities"}
            </span>
          </div>
        )}
      </div>

      {/* ── Entities ── */}
      {entities.length > 0 && (
        <section
          className="py-4"
          style={{
            borderBottom: "1px solid var(--mem-border)",
            animation: "mem-fade-up 350ms cubic-bezier(0.16, 1, 0.3, 1) 160ms both",
          }}
        >
          <div className="flex items-center justify-between mb-2.5">
            <h3
              style={{
                fontFamily: "var(--mem-font-mono)",
                fontSize: "11px",
                fontWeight: 600,
                letterSpacing: "0.05em",
                textTransform: "uppercase" as const,
                color: "var(--mem-text-tertiary)",
              }}
            >
              Entities
            </h3>
            <span style={{ fontFamily: "var(--mem-font-mono)", fontSize: "11px", color: "var(--mem-text-tertiary)" }}>
              {entities.length}
            </span>
          </div>
          <div
            ref={chipsRef}
            className="flex flex-wrap gap-1.5"
            style={{
              // `collapsedChipHeight` is measured from the first chip's
              // rendered height on mount. That keeps this tight enough to
              // hide the second row (whose top edge would otherwise sit at
              // chipHeight + gap-1.5 = chipHeight + 6px) while surviving
              // font-size / OS zoom changes that would break a hardcoded
              // pixel value.
              maxHeight: showAllEntities ? "none" : `${collapsedChipHeight}px`,
              overflow: "hidden",
              transition: "max-height 300ms ease",
            }}
          >
            {entities.map((e) => (
              <button
                key={e.id}
                onClick={() => onEntityClick(e.id)}
                className="px-2.5 py-1 rounded-full text-xs transition-all duration-150 hover:brightness-110"
                style={{
                  fontFamily: "var(--mem-font-body)",
                  color: "var(--mem-text-secondary)",
                  backgroundColor: "var(--mem-surface)",
                  border: "1px solid var(--mem-border)",
                }}
              >
                {e.name}
              </button>
            ))}
          </div>
          {hasChipsOverflow && !showAllEntities && (
            <button
              onClick={() => setShowAllEntities(true)}
              className="mt-2 text-xs transition-colors duration-150 hover:underline"
              style={{
                fontFamily: "var(--mem-font-body)",
                color: "var(--mem-text-secondary)",
                background: "none",
                border: "none",
                cursor: "pointer",
              }}
            >
              Show all
            </button>
          )}
          {showAllEntities && (
            <button
              onClick={() => setShowAllEntities(false)}
              className="mt-2 text-xs transition-colors duration-150 hover:underline"
              style={{
                fontFamily: "var(--mem-font-body)",
                color: "var(--mem-text-secondary)",
                background: "none",
                border: "none",
                cursor: "pointer",
              }}
            >
              Show less
            </button>
          )}
        </section>
      )}

      {/* ── Concepts — card grid ── */}
      {concepts.length > 0 && (
        <section
          className="pt-4 pb-4"
          style={{
            borderBottom: "1px solid var(--mem-border)",
            animation: "mem-fade-up 350ms cubic-bezier(0.16, 1, 0.3, 1) 180ms both",
          }}
        >
          <h3
            className="mb-3"
            style={{
              fontFamily: "var(--mem-font-heading)",
              fontSize: "14px",
              color: "var(--mem-text)",
            }}
          >
            Concepts
          </h3>
          <div
            className="grid gap-3"
            style={{ gridTemplateColumns: "repeat(auto-fill, minmax(220px, 1fr))" }}
          >
            {concepts.map((c, i) => (
              <button
                key={c.id}
                onClick={() => onSelectConcept(c.id)}
                className="text-left px-4 py-3.5 rounded-xl transition-colors duration-200 hover:bg-[var(--mem-hover)] group flex flex-col"
                style={{
                  border: "1px solid var(--mem-border)",
                  animation: `mem-fade-up 400ms cubic-bezier(0.16, 1, 0.3, 1) ${i * 40}ms both`,
                }}
              >
                <div className="flex items-start gap-2.5">
                  <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" style={{ color: "var(--mem-accent-concept)", marginTop: "2px" }} className="shrink-0">
                    <path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5" />
                  </svg>
                  <span
                    style={{
                      fontFamily: "var(--mem-font-heading)",
                      fontSize: "13px",
                      fontWeight: 500,
                      color: "var(--mem-text)",
                      lineHeight: "1.4",
                    }}
                    className="group-hover:text-[var(--mem-accent-concept)] transition-colors"
                  >
                    {c.title}
                  </span>
                </div>
                {c.summary && (
                  <p
                    className="line-clamp-3 flex-1"
                    style={{
                      fontFamily: "var(--mem-font-body)",
                      fontSize: "12px",
                      color: "var(--mem-text-tertiary)",
                      margin: "8px 0 0 0",
                      lineHeight: "1.5",
                    }}
                  >
                    {c.summary}
                  </p>
                )}
                <span
                  className="mt-3"
                  style={{ fontFamily: "var(--mem-font-mono)", fontSize: "10px", color: "var(--mem-text-tertiary)" }}
                >
                  from {c.source_memory_ids.length} memories
                </span>
              </button>
            ))}
          </div>
        </section>
      )}

      {/* ── Nurture cards (hidden — confirmation moving to concept level) ── */}
      {SHOW_NURTURE && nurtureCards.length > 0 && (
        <section className="flex flex-col gap-2 pt-4">
          <p style={{ fontFamily: "var(--mem-font-body)", fontSize: "13px", color: "var(--mem-text-tertiary)", margin: 0 }}>
            Origin learned something new
          </p>
          {nurtureCards.map((card) => (
            <NurtureCard
              key={card.source_id}
              memory={card}
              onConfirm={(id) => nurtureConfirmMutation.mutate(id)}
              onDismiss={() => {}}
              onDelete={(id) => nurtureDeleteMutation.mutate(id)}
            />
          ))}
        </section>
      )}

      {/* ── Memory stream (collapsed by default) ── */}
      <div
        className="pt-4"
        style={{ animation: "mem-fade-up 350ms cubic-bezier(0.16, 1, 0.3, 1) 200ms both" }}
      >
        <button
          onClick={() => setMemoriesExpanded(!memoriesExpanded)}
          className="flex items-center gap-2 mb-3 w-full text-left group"
        >
          <svg
            width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"
            style={{
              color: "var(--mem-text-tertiary)",
              transform: memoriesExpanded ? "rotate(90deg)" : "rotate(0deg)",
              transition: "transform 150ms ease",
            }}
          >
            <path d="M9 18l6-6-6-6" />
          </svg>
          <span
            style={{
              fontFamily: "var(--mem-font-heading)",
              fontSize: "14px",
              color: "var(--mem-text-secondary)",
            }}
          >
            Memories ({space.memory_count})
          </span>
        </button>
        {memoriesExpanded && memories.length > 0 && (
          <MemoryStream
            memories={memories}
            selectedDomain={null}
            onSelectMemory={onSelectMemory}
            cardVariant="insight"
            sortMode={sortMode}
            onSortChange={setSortMode}
          />
        )}
        {memoriesExpanded && memories.length === 0 && (
          <div
            className="flex flex-col items-center justify-center py-12 gap-3"
            style={{ opacity: 0.6 }}
          >
            <span style={{ fontFamily: "var(--mem-font-body)", fontSize: "14px", color: "var(--mem-text-tertiary)" }}>
              No memories in this space yet
            </span>
          </div>
        )}
      </div>
    </div>
  );
}
