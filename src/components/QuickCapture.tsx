// SPDX-License-Identifier: AGPL-3.0-only
import { useState, useRef, useEffect } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { quickCapture } from "../lib/tauri";

interface QuickCaptureProps {
  isOpen: boolean;
  onClose: () => void;
  standalone?: boolean;
}

export default function QuickCapture({ isOpen, onClose, standalone }: QuickCaptureProps) {
  const [content, setContent] = useState("");
  const [saved, setSaved] = useState(false);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const queryClient = useQueryClient();

  const captureMutation = useMutation({
    mutationFn: quickCapture,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["indexedFiles"] });
      queryClient.invalidateQueries({ queryKey: ["memories"] });
      setSaved(true);
      setTimeout(() => {
        setSaved(false);
        setContent("");
        onClose();
      }, 500);
    },
  });

  const handleSubmit = () => {
    if (!content.trim() || captureMutation.isPending) return;
    captureMutation.mutate({ content: content.trim() });
  };

  // Document-level keydown so Esc / Cmd+Enter work regardless of focus target.
  // The textarea's onKeyDown used to own this, which broke after the window was
  // hidden and re-shown: the textarea lost focus and never regained it (because
  // `isOpen` is constant-true in standalone mode, so the auto-focus effect
  // never re-ran), leaving Esc unhandled.
  useEffect(() => {
    if (!isOpen) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        e.preventDefault();
        onClose();
      } else if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) {
        e.preventDefault();
        handleSubmit();
      }
    };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
    // handleSubmit closes over content/captureMutation; re-bind on change so the
    // ⌘↵ submit path sees the latest draft.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isOpen, onClose, content]);

  // Re-focus the textarea whenever the window becomes visible.
  // In standalone (Tauri window) mode, hiding the window doesn't unmount the
  // component — it just hides the webview. When it comes back via
  // `WebviewWindow.show()` the document fires `visibilitychange`; we use that
  // to restore text-entry focus so the user can type immediately.
  useEffect(() => {
    if (!isOpen) return;
    const refocus = () => {
      if (!document.hidden && textareaRef.current) {
        textareaRef.current.focus();
      }
    };
    refocus(); // initial mount + any state toggle
    document.addEventListener("visibilitychange", refocus);
    window.addEventListener("focus", refocus);
    return () => {
      document.removeEventListener("visibilitychange", refocus);
      window.removeEventListener("focus", refocus);
    };
  }, [isOpen]);

  if (!isOpen) return null;

  const charCount = content.trim().length;
  const isEmpty = charCount === 0;
  const isPending = captureMutation.isPending;

  // Left border accent shifts with state — mirrors MemoryCard pattern
  const borderAccent = saved
    ? "var(--mem-accent-warm)"
    : isPending
      ? "var(--mem-accent-amber)"
      : !isEmpty
        ? "var(--mem-accent-indigo)"
        : "var(--mem-border)";

  const card = (
    <div
      className="flex-1 flex flex-col overflow-hidden rounded-xl"
      style={{
        backgroundColor: "var(--mem-surface)",
        border: "1px solid var(--mem-border)",
        borderColor: borderAccent,
        boxShadow: !isEmpty && !saved
          ? "0 0 20px var(--mem-shimmer-color), 0 8px 32px rgba(0,0,0,0.25)"
          : "0 8px 32px rgba(0,0,0,0.25)",
        transition: "border-color 0.3s ease, box-shadow 0.3s ease",
      }}
    >
      {/* Header */}
      <div
        className="flex items-center justify-between px-4 pt-3 pb-1 shrink-0"
        style={{ opacity: 0.7 }}
      >
        <div className="flex items-center gap-2">
          <div
            className={`w-1.5 h-1.5 rounded-full transition-colors duration-200 ${
              saved
                ? "bg-[var(--mem-accent-warm)]"
                : isPending
                  ? "bg-[var(--mem-accent-amber)] animate-pulse"
                  : !isEmpty
                    ? "bg-[var(--mem-accent-indigo)]"
                    : ""
            }`}
            style={isEmpty && !saved && !isPending ? { backgroundColor: "var(--mem-text-tertiary)", opacity: 0.4 } : undefined}
          />
          <span
            style={{
              fontFamily: "var(--mem-font-mono)",
              fontSize: "10px",
              fontWeight: 600,
              letterSpacing: "0.05em",
              color: "var(--mem-text-tertiary)",
              textTransform: "uppercase" as const,
            }}
          >
            {saved ? "Saved to memory" : isPending ? "Saving..." : "Quick capture"}
          </span>
        </div>
        <button
          onClick={onClose}
          className="p-0.5 rounded transition-colors duration-150 hover:bg-[var(--mem-hover-strong)]"
          style={{ color: "var(--mem-text-tertiary)" }}
          title="Close (Esc)"
        >
          <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
          </svg>
        </button>
      </div>

      {/* Writing surface */}
      <div className="flex-1 flex flex-col px-4 pb-3 min-h-0">
        <textarea
          ref={textareaRef}
          value={content}
          onChange={(e) => setContent(e.target.value)}
          placeholder="What's on your mind?"
          rows={standalone ? undefined : 7}
          className={`w-full bg-transparent focus:outline-none resize-none pt-3 ${standalone ? "flex-1" : ""}`}
          style={{
            fontFamily: "var(--mem-font-body)",
            fontSize: "14px",
            lineHeight: "1.6",
            color: "var(--mem-text)",
            caretColor: "var(--mem-accent-indigo)",
          }}
          autoFocus
          disabled={isPending || saved}
        />

        {/* Bottom bar */}
        <div
          className="flex items-center justify-between pt-2 mt-auto shrink-0"
          style={{ opacity: 0.7 }}
        >
          <span
            className="transition-opacity duration-150"
            style={{
              fontFamily: "var(--mem-font-mono)",
              fontSize: "10px",
              color: "var(--mem-text-tertiary)",
              opacity: isEmpty ? 0 : 1,
            }}
          >
            {charCount} char{charCount !== 1 ? "s" : ""}
          </span>
          <div className="flex items-center gap-3">
            <span
              className="select-none"
              style={{
                fontFamily: "var(--mem-font-mono)",
                fontSize: "10px",
                color: "var(--mem-text-tertiary)",
                opacity: 0.5,
              }}
            >
              {standalone ? "esc close" : "\u2318\u23CE save"}
            </span>
            <button
              onClick={handleSubmit}
              disabled={isEmpty || isPending || saved}
              className="transition-all duration-200"
              style={{
                fontFamily: "var(--mem-font-body)",
                fontSize: "12px",
                fontWeight: 500,
                padding: "4px 14px",
                borderRadius: 8,
                color:
                  saved
                    ? "var(--mem-accent-warm)"
                    : isEmpty || isPending
                      ? "var(--mem-text-tertiary)"
                      : "white",
                backgroundColor:
                  saved
                    ? "var(--mem-confirm-bg)"
                    : isEmpty || isPending
                      ? "var(--mem-hover)"
                      : "var(--mem-accent-indigo)",
                cursor: isEmpty || isPending || saved ? "default" : "pointer",
                boxShadow:
                  !isEmpty && !isPending && !saved
                    ? "0 2px 10px var(--mem-shimmer-color)"
                    : "none",
              }}
            >
              {saved ? "Saved" : isPending ? "..." : "Save"}
            </button>
          </div>
        </div>
      </div>
    </div>
  );

  if (standalone) {
    return (
      <div className="w-full h-screen p-[12px] flex flex-col" style={{ background: "transparent" }}>
        {card}
      </div>
    );
  }

  return (
    <div
      className="fixed inset-0 flex items-center justify-center z-50"
      onClick={onClose}
      style={{
        backgroundColor: "rgba(0,0,0,0.5)",
        backdropFilter: "blur(4px)",
        WebkitBackdropFilter: "blur(4px)",
        animation: "qc-overlay-in 0.15s ease-out",
      }}
    >
      <div
        className="w-[500px] max-h-[55vh] flex flex-col"
        onClick={(e) => e.stopPropagation()}
        style={{ animation: "mem-fade-up 0.2s cubic-bezier(0.16, 1, 0.3, 1)" }}
      >
        {card}
      </div>
      <style>{`
        @keyframes qc-overlay-in {
          from { opacity: 0; }
          to { opacity: 1; }
        }
      `}</style>
    </div>
  );
}
