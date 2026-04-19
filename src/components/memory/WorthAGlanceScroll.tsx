// SPDX-License-Identifier: AGPL-3.0-only
import { useState, useRef, useEffect } from "react";
import { createPortal } from "react-dom";
import type { ActivityKind, RecentActivityItem } from "../../lib/tauri";
import { updateMemory, getMemoryDetail } from "../../lib/tauri";

interface Props {
  items: RecentActivityItem[];
  onConfirm: (id: string) => void;
  onDelete: (id: string) => void;
  onEdit: (kind: ActivityKind, id: string) => void;
  onNavigate: (kind: ActivityKind, id: string) => void;
  recapCount?: number;
  onViewRecaps?: () => void;
}

function SproutIcon() {
  return (
    <svg
      width="14"
      height="14"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      style={{ color: "var(--mem-text-tertiary)" }}
      className="shrink-0"
      aria-hidden="true"
    >
      <path d="M7 20h10" />
      <path d="M10 20c5.5-2.5.8-6.4 3-10" />
      <path d="M9.5 9.4c1.1.8 1.8 2.2 2.3 3.7-2 .4-3.5.4-4.8-.3-1.2-.6-2.3-1.9-3-4.2 2.8-.5 4.4 0 5.5.8z" />
      <path d="M14.1 6a7 7 0 0 0-1.1 4c1.9.1 3.3-.2 4.3-.9 1-.6 1.9-1.8 2.7-3.6-2.4-.6-3.9-.4-4.9.3-1 .7-.9 1-1 .2z" />
    </svg>
  );
}

export function WorthAGlanceScroll({ items, onConfirm, onDelete, onEdit, onNavigate, recapCount, onViewRecaps }: Props) {
  const [editingItem, setEditingItem] = useState<RecentActivityItem | null>(null);

  if (items.length === 0) return null;

  return (
    <section data-testid="worth-a-glance">
      <div className="flex items-baseline justify-between mb-2">
        <h2 className="text-sm font-medium uppercase tracking-wide text-neutral-500">
          Worth a glance
        </h2>
        {onViewRecaps && recapCount != null && recapCount > 0 && (
          <button
            type="button"
            onClick={onViewRecaps}
            className="transition-colors duration-150"
            style={{
              fontFamily: "var(--mem-font-mono)",
              fontSize: 10,
              fontWeight: 500,
              letterSpacing: "0.03em",
              color: "var(--mem-text-tertiary)",
              background: "none",
              border: "none",
              cursor: "pointer",
              padding: "2px 0",
            }}
            onMouseEnter={(e) => (e.currentTarget.style.color = "var(--mem-text-secondary)")}
            onMouseLeave={(e) => (e.currentTarget.style.color = "var(--mem-text-tertiary)")}
          >
            {recapCount} recap{recapCount !== 1 ? "s" : ""} &rarr;
          </button>
        )}
      </div>
      <p
        style={{
          fontFamily: "var(--mem-font-body)",
          fontSize: 12,
          fontStyle: "italic",
          color: "var(--mem-text-tertiary)",
          marginTop: 2,
        }}
        className="mb-3"
      >
        unconfirmed memories and items Origin flagged for a second look
      </p>
      <div
        className="flex gap-2 overflow-x-auto pb-2"
        style={{
          scrollSnapType: "x mandatory",
          scrollbarWidth: "none",
          msOverflowStyle: "none",
          WebkitOverflowScrolling: "touch",
          marginRight: -72,
          paddingRight: 72,
        }}
      >
        {items.map((item) => (
          <WorthAGlanceCard
            key={`${item.kind}:${item.id}`}
            item={item}
            onConfirm={onConfirm}
            onDelete={onDelete}
            onEditClick={() => setEditingItem(item)}
            onNavigate={onNavigate}
          />
        ))}
      </div>

      {editingItem &&
        createPortal(
          <EditMemoryModal
            item={editingItem}
            onDismiss={() => setEditingItem(null)}
            onSave={async (newContent, _newTitle) => {
              await updateMemory(editingItem.id, newContent);
              onConfirm(editingItem.id);
              onEdit(editingItem.kind, editingItem.id);
              setEditingItem(null);
            }}
          />,
          document.body,
        )}
    </section>
  );
}

function WorthAGlanceCard({
  item,
  onConfirm,
  onDelete,
  onEditClick,
  onNavigate,
}: {
  item: RecentActivityItem;
  onConfirm: (id: string) => void;
  onDelete: (id: string) => void;
  onEditClick: () => void;
  onNavigate: (kind: ActivityKind, id: string) => void;
}) {
  const [hover, setHover] = useState(false);
  const [confirming, setConfirming] = useState(false);
  const title = item.title || item.snippet || "(untitled)";
  const showSnippet = Boolean(item.snippet && item.title);

  const baseBg = "var(--mem-surface)";
  const hoverBg = "var(--mem-hover)";

  return (
    <article
      data-testid={`worth-a-glance-card-${item.kind}`}
      className="relative flex flex-col gap-2 rounded-xl border px-3 py-2.5 shrink-0 transition-[border-color,box-shadow,background-color] duration-200 hover:shadow-[0_1px_2px_rgba(0,0,0,0.03),0_3px_10px_rgba(0,0,0,0.04)]"
      style={{
        width: 240,
        scrollSnapAlign: "start",
        backgroundColor: hover ? hoverBg : baseBg,
        borderColor: "var(--mem-hover-strong)",
        cursor: "pointer",
        opacity: confirming ? 0.5 : 1,
        transform: confirming ? "scale(0.97)" : "scale(1)",
        transition:
          "border-color 200ms, box-shadow 200ms, background-color 200ms, opacity 300ms, transform 300ms",
      }}
      onMouseEnter={() => setHover(true)}
      onMouseLeave={() => setHover(false)}
      onClick={() => onNavigate(item.kind, item.id)}
    >
      <button
        type="button"
        onClick={(e) => {
          e.stopPropagation();
          setConfirming(true);
          setTimeout(() => onDelete(item.id), 300);
        }}
        aria-label="Delete"
        className="absolute right-2 top-2 text-xs leading-none p-1 rounded transition-colors"
        style={{
          color: "var(--mem-text-tertiary)",
          opacity: hover ? 1 : 0,
          transition: "opacity 150ms, color 150ms",
        }}
        onMouseEnter={(e) => (e.currentTarget.style.color = "var(--mem-text)")}
        onMouseLeave={(e) => (e.currentTarget.style.color = "var(--mem-text-tertiary)")}
      >
        &times;
      </button>

      <div className="flex items-start gap-2 pr-6">
        <div className="mt-0.5">
          <SproutIcon />
        </div>
        <span
          className="flex-1 line-clamp-2"
          style={{
            fontFamily: "var(--mem-font-heading)",
            fontSize: 13,
            fontWeight: 500,
            color: "var(--mem-text)",
            lineHeight: 1.4,
          }}
        >
          {title}
        </span>
      </div>

      {showSnippet && (
        <p
          className="line-clamp-2"
          style={{
            fontFamily: "var(--mem-font-body)",
            fontSize: 12,
            color: "var(--mem-text-tertiary)",
            lineHeight: 1.4,
          }}
        >
          {item.snippet}
        </p>
      )}

      <div
        className="flex gap-2 mt-auto"
        style={{
          opacity: hover ? 1 : 0,
          transition: "opacity 150ms",
          pointerEvents: hover ? "auto" : "none",
        }}
      >
        <button
          type="button"
          onClick={(e) => {
            e.stopPropagation();
            setConfirming(true);
            setTimeout(() => onConfirm(item.id), 300);
          }}
          className="rounded-md px-3 py-1.5 text-xs font-medium transition-colors"
          style={{
            color: "var(--mem-accent-warm)",
            backgroundColor:
              "color-mix(in srgb, var(--mem-accent-warm) 12%, transparent)",
          }}
          onMouseEnter={(e) =>
            (e.currentTarget.style.backgroundColor =
              "color-mix(in srgb, var(--mem-accent-warm) 22%, transparent)")
          }
          onMouseLeave={(e) =>
            (e.currentTarget.style.backgroundColor =
              "color-mix(in srgb, var(--mem-accent-warm) 12%, transparent)")
          }
        >
          Looks good
        </button>
        <button
          type="button"
          onClick={(e) => {
            e.stopPropagation();
            onEditClick();
          }}
          className="rounded-md border px-3 py-1.5 text-xs font-medium transition-colors"
          style={{
            color: "var(--mem-text-secondary)",
            backgroundColor: "transparent",
            borderColor: "var(--mem-border)",
          }}
          onMouseEnter={(e) =>
            (e.currentTarget.style.backgroundColor = "var(--mem-hover)")
          }
          onMouseLeave={(e) =>
            (e.currentTarget.style.backgroundColor = "transparent")
          }
        >
          Edit
        </button>
      </div>
    </article>
  );
}

// ── Edit Memory Modal ──
// Follows QuickCapture's visual language: rounded card, border accent,
// monospace status line, indigo save button with shimmer.

function EditMemoryModal({
  item,
  onDismiss,
  onSave,
}: {
  item: RecentActivityItem;
  onDismiss: () => void;
  onSave: (content: string, title?: string) => Promise<void>;
}) {
  const [title, setTitle] = useState(item.title || "");
  const [content, setContent] = useState(item.snippet || item.title || "");
  const [saving, setSaving] = useState(false);
  const [loading, setLoading] = useState(true);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // Fetch full memory content on open (snippet is truncated)
  useEffect(() => {
    let cancelled = false;
    getMemoryDetail(item.id).then((detail) => {
      if (cancelled || !detail) { setLoading(false); return; }
      setContent(detail.content || detail.summary || item.snippet || "");
      if (detail.title) setTitle(detail.title);
      setLoading(false);
      setTimeout(() => textareaRef.current?.focus(), 50);
    }).catch(() => setLoading(false));
    return () => { cancelled = true; };
  }, [item.id, item.snippet]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        e.preventDefault();
        onDismiss();
      } else if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) {
        e.preventDefault();
        handleSave();
      }
    };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [content, title, saving]);

  const handleSave = async () => {
    if (saving || !content.trim()) return;
    setSaving(true);
    await onSave(content.trim(), title.trim() || undefined);
  };

  const isEmpty = content.trim().length === 0;

  const borderAccent = saving
    ? "var(--mem-accent-amber)"
    : !isEmpty
      ? "var(--mem-accent-indigo)"
      : "var(--mem-border)";

  return (
    <div
      className="fixed inset-0 flex items-center justify-center z-50"
      onClick={onDismiss}
      style={{
        backgroundColor: "rgba(0,0,0,0.45)",
        backdropFilter: "blur(4px)",
        WebkitBackdropFilter: "blur(4px)",
        animation: "wag-overlay-in 0.15s ease-out",
      }}
    >
      <div
        className="w-[680px] max-h-[80vh] flex flex-col"
        onClick={(e) => e.stopPropagation()}
        style={{ animation: "mem-fade-up 0.2s cubic-bezier(0.16, 1, 0.3, 1)" }}
      >
        <div
          className="flex flex-col overflow-hidden rounded-xl"
          style={{
            backgroundColor: "var(--mem-surface)",
            border: "1px solid",
            borderColor: borderAccent,
            boxShadow: !isEmpty
              ? "0 0 20px var(--mem-shimmer-color), 0 8px 32px rgba(0,0,0,0.25)"
              : "0 8px 32px rgba(0,0,0,0.25)",
            transition: "border-color 0.3s ease, box-shadow 0.3s ease",
          }}
        >
          {/* Header */}
          <div
            className="flex items-center justify-between px-5 pt-3 pb-1"
            style={{ opacity: 0.7 }}
          >
            <div className="flex items-center gap-2">
              <div
                className="w-1.5 h-1.5 rounded-full transition-colors duration-200"
                style={{
                  backgroundColor: saving
                    ? "var(--mem-accent-amber)"
                    : !isEmpty
                      ? "var(--mem-accent-indigo)"
                      : "var(--mem-text-tertiary)",
                  opacity: isEmpty && !saving ? 0.4 : 1,
                }}
              />
              <span
                style={{
                  fontFamily: "var(--mem-font-mono)",
                  fontSize: 10,
                  fontWeight: 600,
                  letterSpacing: "0.05em",
                  color: "var(--mem-text-tertiary)",
                  textTransform: "uppercase",
                }}
              >
                {saving ? "Saving..." : "Edit memory"}
              </span>
            </div>
            <button
              onClick={onDismiss}
              className="p-0.5 rounded transition-colors duration-150 hover:bg-[var(--mem-hover-strong)]"
              style={{ color: "var(--mem-text-tertiary)" }}
              title="Dismiss (Esc)"
            >
              <svg
                className="w-3.5 h-3.5"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M6 18L18 6M6 6l12 12"
                />
              </svg>
            </button>
          </div>

          {/* Editable title */}
          <div className="px-5 pt-2">
            <input
              type="text"
              value={title}
              onChange={(e) => setTitle(e.target.value)}
              placeholder="Title"
              className="w-full bg-transparent focus:outline-none"
              style={{
                fontFamily: "var(--mem-font-heading)",
                fontSize: 16,
                fontWeight: 600,
                color: "var(--mem-text)",
                lineHeight: 1.4,
                borderBottom: "1px solid var(--mem-border)",
                paddingBottom: 8,
              }}
              disabled={saving}
            />
          </div>

          {/* Editing surface */}
          <div className="flex flex-col px-5 pb-4 min-h-0 flex-1">
            {loading ? (
              <div className="flex items-center justify-center py-8" style={{ color: "var(--mem-text-tertiary)" }}>
                <span style={{ fontFamily: "var(--mem-font-mono)", fontSize: 11 }}>Loading...</span>
              </div>
            ) : (
              <textarea
                ref={textareaRef}
                value={content}
                onChange={(e) => setContent(e.target.value)}
                placeholder="Memory content..."
                className="w-full flex-1 bg-transparent focus:outline-none resize-none pt-3 overflow-y-auto"
                style={{
                  fontFamily: "var(--mem-font-body)",
                  fontSize: 14,
                  lineHeight: 1.7,
                  color: "var(--mem-text)",
                  caretColor: "var(--mem-accent-indigo)",
                  minHeight: 240,
                  maxHeight: "55vh",
                }}
                disabled={saving}
              />
            )}

            {/* Bottom bar */}
            <div
              className="flex items-center justify-end pt-3 shrink-0"
              style={{ opacity: 0.7 }}
            >
              <div className="flex items-center gap-3">
                <span
                  className="select-none"
                  style={{
                    fontFamily: "var(--mem-font-mono)",
                    fontSize: 10,
                    color: "var(--mem-text-tertiary)",
                    opacity: 0.5,
                  }}
                >
                  esc dismiss
                </span>
                <button
                  onClick={onDismiss}
                  disabled={saving}
                  className="transition-all duration-200"
                  style={{
                    fontFamily: "var(--mem-font-body)",
                    fontSize: 12,
                    fontWeight: 500,
                    padding: "5px 16px",
                    borderRadius: 8,
                    color: "var(--mem-text-secondary)",
                    backgroundColor: "transparent",
                    border: "1px solid var(--mem-border)",
                    cursor: saving ? "default" : "pointer",
                  }}
                  onMouseEnter={(e) =>
                    (e.currentTarget.style.backgroundColor = "var(--mem-hover)")
                  }
                  onMouseLeave={(e) =>
                    (e.currentTarget.style.backgroundColor = "transparent")
                  }
                >
                  Dismiss
                </button>
                <button
                  onClick={handleSave}
                  disabled={isEmpty || saving}
                  className="transition-all duration-200"
                  style={{
                    fontFamily: "var(--mem-font-body)",
                    fontSize: 12,
                    fontWeight: 500,
                    padding: "5px 16px",
                    borderRadius: 8,
                    color:
                      isEmpty || saving ? "var(--mem-text-tertiary)" : "white",
                    backgroundColor:
                      isEmpty || saving
                        ? "var(--mem-hover)"
                        : "var(--mem-accent-indigo)",
                    cursor: isEmpty || saving ? "default" : "pointer",
                    boxShadow:
                      !isEmpty && !saving
                        ? "0 2px 10px var(--mem-shimmer-color)"
                        : "none",
                  }}
                >
                  {saving ? "..." : "Confirm"}
                </button>
              </div>
            </div>
          </div>
        </div>
      </div>

      <style>{`
        @keyframes wag-overlay-in {
          from { opacity: 0; }
          to { opacity: 1; }
        }
      `}</style>
    </div>
  );
}
