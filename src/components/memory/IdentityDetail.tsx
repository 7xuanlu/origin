// SPDX-License-Identifier: AGPL-3.0-only
import { useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import {
  getEntityDetail,
  updateObservation,
  deleteObservation,
  addObservation,
  confirmObservation,
  deleteEntity,
  search,
  FACET_COLORS,
  type Observation,
} from "../../lib/tauri";

interface IdentityDetailProps {
  entityId: string;
  onBack: () => void;
  onEntityClick: (entityId: string) => void;
  onMemoryClick?: (sourceId: string) => void;
}

export default function IdentityDetail({ entityId, onBack, onEntityClick, onMemoryClick }: IdentityDetailProps) {
  const queryClient = useQueryClient();
  const [editingObs, setEditingObs] = useState<string | null>(null);
  const [editContent, setEditContent] = useState("");
  const [newObs, setNewObs] = useState("");
  const [showAddForm, setShowAddForm] = useState(false);
  const [confirmDelete, setConfirmDelete] = useState(false);

  const { data: detail } = useQuery({
    queryKey: ["entityDetail", entityId],
    queryFn: () => getEntityDetail(entityId),
    refetchInterval: 5000,
  });

  const invalidate = () => {
    queryClient.invalidateQueries({ queryKey: ["entityDetail", entityId] });
    queryClient.invalidateQueries({ queryKey: ["entities"] });
  };

  const updateMutation = useMutation({
    mutationFn: ({ id, content }: { id: string; content: string }) =>
      updateObservation(id, content),
    onSuccess: invalidate,
  });

  const deleteMutation = useMutation({
    mutationFn: (id: string) => deleteObservation(id),
    onSuccess: invalidate,
  });

  const addMutation = useMutation({
    mutationFn: (content: string) => addObservation(entityId, content, "human", 1.0),
    onSuccess: () => {
      setNewObs("");
      setShowAddForm(false);
      invalidate();
    },
  });

  const confirmMutation = useMutation({
    mutationFn: ({ id, confirmed }: { id: string; confirmed: boolean }) =>
      confirmObservation(id, confirmed),
    onSuccess: invalidate,
  });

  const deleteEntityMutation = useMutation({
    mutationFn: () => deleteEntity(entityId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["entities"] });
      queryClient.invalidateQueries({ queryKey: ["constellation-entities"] });
      onBack();
    },
  });

  // Linked memories: search by entity name to find memories referencing this entity
  const { data: linkedMemories = [] } = useQuery({
    queryKey: ["entity-linked-memories", entityId, detail?.entity?.name],
    queryFn: async () => {
      if (!detail?.entity?.name) return [];
      const results = await search(detail.entity.name, 10, "memory");
      // Filter to results that have this entity_id, or are semantically close
      return results.filter((r) => r.entity_id === entityId || r.score > 0.7).slice(0, 8);
    },
    enabled: !!detail?.entity?.name,
    staleTime: 30000,
  });

  if (!detail) return null;

  const { entity, observations: rawObservations, relations: rawRelations } = detail;

  // Deduplicate observations by content (safety net for DB-level dupes)
  const observations = rawObservations.filter(
    (obs, i, arr) => arr.findIndex((o) => o.content.toLowerCase() === obs.content.toLowerCase()) === i,
  );

  // Deduplicate relations by (entity_id, relation_type, direction)
  const relations = rawRelations.filter(
    (rel, i, arr) =>
      arr.findIndex(
        (r) => r.entity_id === rel.entity_id && r.relation_type === rel.relation_type && r.direction === rel.direction,
      ) === i,
  );

  const startEdit = (obs: Observation) => {
    setEditingObs(obs.id);
    setEditContent(obs.content);
  };

  const saveEdit = (id: string) => {
    if (editContent.trim() && editContent !== observations.find((o) => o.id === id)?.content) {
      updateMutation.mutate({ id, content: editContent.trim() });
    }
    setEditingObs(null);
  };

  return (
    <div className="flex flex-col gap-6">
      {/* Back + Entity header */}
      <div>
        <button onClick={onBack} className="p-1.5 -ml-1.5 rounded-md transition-colors duration-150 hover:bg-[var(--mem-hover)]" style={{ color: "var(--mem-text-tertiary)", background: "none", border: "none", cursor: "pointer", lineHeight: 0, marginBottom: "12px" }}>
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5"><path d="M19 12H5M12 19l-7-7 7-7" /></svg>
        </button>
      <div className="flex items-center gap-4">
        <div
          className="w-12 h-12 rounded-full flex items-center justify-center flex-shrink-0"
          style={{
            background: "linear-gradient(135deg, var(--mem-accent-warm), var(--mem-accent-amber))",
            fontFamily: "var(--mem-font-heading)",
            fontSize: "18px",
            color: "white",
          }}
        >
          {entity.name.split(" ").map((w) => w[0]).join("").slice(0, 2).toUpperCase()}
        </div>
        <div>
          <h2
            style={{
              fontFamily: "var(--mem-font-heading)",
              fontSize: "24px",
              color: "var(--mem-text)",
              fontWeight: 500,
            }}
          >
            {entity.name}
          </h2>
          <span
            style={{
              fontFamily: "var(--mem-font-mono)",
              fontSize: "11px",
              color: "var(--mem-text-tertiary)",
            }}
          >
            {entity.entity_type}{entity.domain ? ` · ${entity.domain}` : ""}
          </span>
        </div>
        <div style={{ marginLeft: "auto" }}>
          {confirmDelete ? (
            <div className="flex items-center gap-2">
              <span style={{ fontSize: 11, color: "var(--mem-text-secondary)", fontFamily: "var(--mem-font-body)" }}>
                Delete this entity?
              </span>
              <button
                onClick={() => deleteEntityMutation.mutate()}
                style={{
                  fontSize: 11, fontFamily: "var(--mem-font-body)", fontWeight: 500,
                  color: "#D05050", background: "none", border: "1px solid #D05050",
                  borderRadius: 5, padding: "2px 8px", cursor: "pointer",
                }}
              >
                Delete
              </button>
              <button
                onClick={() => setConfirmDelete(false)}
                style={{
                  fontSize: 11, fontFamily: "var(--mem-font-body)",
                  color: "var(--mem-text-tertiary)", background: "none", border: "1px solid var(--mem-border)",
                  borderRadius: 5, padding: "2px 8px", cursor: "pointer",
                }}
              >
                Cancel
              </button>
            </div>
          ) : (
            <button
              onClick={() => setConfirmDelete(true)}
              className="transition-opacity duration-150"
              style={{
                opacity: 0.4, background: "none", border: "none", cursor: "pointer",
                color: "var(--mem-text-tertiary)", padding: 4,
              }}
              onMouseEnter={(e) => { e.currentTarget.style.opacity = "0.8"; }}
              onMouseLeave={(e) => { e.currentTarget.style.opacity = "0.4"; }}
            >
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <polyline points="3 6 5 6 21 6" /><path d="M19 6v14a2 2 0 01-2 2H7a2 2 0 01-2-2V6m3 0V4a2 2 0 012-2h4a2 2 0 012 2v2" />
              </svg>
            </button>
          )}
        </div>
      </div>
      </div>

      {/* About section */}
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
          About
        </h3>
        <div className="flex flex-col gap-1">
          {observations.map((obs) => (
            <div
              key={obs.id}
              className="group flex items-start gap-3 px-3 py-2 rounded-md transition-colors duration-150 hover:bg-[var(--mem-hover)]"
            >
              <button
                onClick={() => confirmMutation.mutate({ id: obs.id, confirmed: !obs.confirmed })}
                className="mt-1.5 flex-shrink-0 w-2.5 h-2.5 rounded-full border transition-all duration-300"
                style={{
                  borderColor: obs.confirmed ? "var(--mem-accent-warm)" : "var(--mem-accent-amber)",
                  backgroundColor: obs.confirmed ? "var(--mem-accent-warm)" : "transparent",
                }}
              />

              {editingObs === obs.id ? (
                <input
                  value={editContent}
                  onChange={(e) => setEditContent(e.target.value)}
                  onBlur={() => saveEdit(obs.id)}
                  onKeyDown={(e) => {
                    if (e.key === "Enter") saveEdit(obs.id);
                    if (e.key === "Escape") setEditingObs(null);
                  }}
                  autoFocus
                  className="flex-1 bg-transparent outline-none"
                  style={{
                    fontFamily: "var(--mem-font-body)",
                    fontSize: "13px",
                    color: "var(--mem-text)",
                  }}
                />
              ) : (
                <span
                  onClick={() => startEdit(obs)}
                  className="flex-1 cursor-text"
                  style={{
                    fontFamily: "var(--mem-font-body)",
                    fontSize: "13px",
                    color: "var(--mem-text)",
                    opacity: obs.confirmed ? 1 : 0.7,
                  }}
                >
                  {obs.content}
                </span>
              )}

              {obs.confidence != null && (
                <span
                  className="flex-shrink-0 mt-0.5"
                  style={{
                    fontFamily: "var(--mem-font-mono)",
                    fontSize: "10px",
                    color: "var(--mem-text-tertiary)",
                  }}
                >
                  {obs.confidence.toFixed(1)}
                </span>
              )}

              <button
                onClick={() => deleteMutation.mutate(obs.id)}
                className="opacity-0 group-hover:opacity-100 transition-opacity duration-150 flex-shrink-0 mt-0.5"
                style={{ color: "var(--mem-text-tertiary)" }}
              >
                <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <path d="M18 6L6 18M6 6l12 12" />
                </svg>
              </button>
            </div>
          ))}

          {/* Add observation */}
          {showAddForm ? (
            <div className="flex items-center gap-2 px-3 py-2">
              <input
                value={newObs}
                onChange={(e) => setNewObs(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && newObs.trim()) addMutation.mutate(newObs.trim());
                  if (e.key === "Escape") { setShowAddForm(false); setNewObs(""); }
                }}
                placeholder="New observation..."
                autoFocus
                className="flex-1 bg-transparent outline-none placeholder:text-[var(--mem-text-tertiary)]"
                style={{
                  fontFamily: "var(--mem-font-body)",
                  fontSize: "13px",
                  color: "var(--mem-text)",
                }}
              />
            </div>
          ) : (
            <button
              onClick={() => setShowAddForm(true)}
              className="px-3 py-1.5 text-left transition-colors duration-150"
              style={{
                fontFamily: "var(--mem-font-body)",
                fontSize: "13px",
                color: "var(--mem-text-tertiary)",
              }}
            >
              + add observation
            </button>
          )}
        </div>
      </section>

      {/* Linked memories */}
      {linkedMemories.length > 0 && (
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
            Linked Memories
          </h3>
          <div className="flex flex-col gap-1">
            {linkedMemories.map((mem) => {
              const facet = mem.memory_type ?? null;
              const color = facet ? FACET_COLORS[facet] : null;
              const isSuperseded = mem.is_archived;
              return (
                <button
                  key={mem.id}
                  onClick={() => onMemoryClick?.(mem.source_id)}
                  className="flex items-start gap-3 px-3 py-2 rounded-md text-left transition-colors duration-150 hover:bg-[var(--mem-hover)]"
                  style={{ opacity: isSuperseded ? 0.7 : 1 }}
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
                      {mem.content.length > 200 ? mem.content.substring(0, 200) + "\u2026" : mem.content}
                    </p>
                    <div
                      className="flex items-center gap-2 mt-1"
                      style={{
                        fontFamily: "var(--mem-font-mono)",
                        fontSize: "10px",
                        color: "var(--mem-text-tertiary)",
                      }}
                    >
                      {facet && color && (
                        <span className={`px-1 py-0.5 rounded text-[9px] font-medium border ${color}`}>
                          {facet}
                        </span>
                      )}
                      {isSuperseded && (
                        <span className="text-zinc-500">archived</span>
                      )}
                    </div>
                  </div>
                </button>
              );
            })}
          </div>
        </section>
      )}

      {/* Connections section */}
      {relations.length > 0 && (
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
            Connections
          </h3>
          <div className="flex flex-col gap-1">
            {relations.map((rel) => (
              <button
                key={rel.id}
                onClick={() => onEntityClick(rel.entity_id)}
                className="flex items-center gap-2 px-3 py-2 rounded-md text-left transition-colors duration-150 hover:bg-[var(--mem-hover)]"
              >
                <span
                  style={{
                    fontFamily: "var(--mem-font-body)",
                    fontSize: "13px",
                    color: "var(--mem-text-secondary)",
                  }}
                >
                  {rel.direction === "outgoing" ? rel.relation_type : `← ${rel.relation_type}`}
                </span>
                <span style={{ color: "var(--mem-text-tertiary)" }}>&rarr;</span>
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
                  {rel.entity_name}
                </span>
                <span
                  style={{
                    fontFamily: "var(--mem-font-mono)",
                    fontSize: "10px",
                    color: "var(--mem-text-tertiary)",
                  }}
                >
                  {rel.entity_type}
                </span>
              </button>
            ))}
          </div>
        </section>
      )}
    </div>
  );
}
