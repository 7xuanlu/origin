// SPDX-License-Identifier: AGPL-3.0-only

import { useEffect, useState, useCallback, useRef, useLayoutEffect } from "react";
import { getCurrentWindow, primaryMonitor } from "@tauri-apps/api/window";
import { emit, listen } from "@tauri-apps/api/event";
import { invoke } from "@tauri-apps/api/core";
import { LogicalSize, LogicalPosition } from "@tauri-apps/api/dpi";
import {
  type AmbientCard,
  type MemorySnippet,
  type SelectionCardPayload,
  listenAmbientCard,
  listenSelectionCard,
  dismissAmbientCard,
  triggerAmbient,
} from "../lib/ambient";

const CARD_WIDTH = 344; // 400 (QC width) - 2 * WIN_PADDING(28) = 344
const CARD_GAP = 8;
const SCREEN_PADDING = 16;
/** Padding around card — must be large enough that shadow isn't clipped */
const WIN_PADDING = 28;
const MAX_CARDS = 3;
const WIN_WIDTH = CARD_WIDTH + WIN_PADDING * 2;
const MAX_WIN_HEIGHT = 600;
/** Ambient proactive cards auto-dismiss after 10s */
const AMBIENT_DISMISS_MS = 10000;

interface SelectionState {
  card: AmbientCard;
  cursor_x: number;
  cursor_y: number;
}

export default function AmbientOverlay() {
  const [ambientCards, setAmbientCards] = useState<AmbientCard[]>([]);
  const [selection, setSelection] = useState<SelectionState | null>(null);
  const ambientCardsRef = useRef<AmbientCard[]>([]);
  const selectionRef = useRef<SelectionState | null>(null);
  /** True while a trigger-ambient search is in flight — prevents hide race */
  const [triggerLoading, setTriggerLoading] = useState(false);

  useEffect(() => { ambientCardsRef.current = ambientCards; }, [ambientCards]);
  useEffect(() => { selectionRef.current = selection; }, [selection]);

  // Listen for proactive/on-demand ambient cards
  useEffect(() => {
    const unlisten = listenAmbientCard((card) => {
      setAmbientCards((prev) => {
        if (prev.some((c) => c.card_id === card.card_id)) return prev;
        return [card, ...prev].slice(0, MAX_CARDS);
      });
    });
    return () => { unlisten.then((fn) => fn()); };
  }, []);

  // Listen for icon-triggered selection cards — NO auto-dismiss
  useEffect(() => {
    const unlisten = listenSelectionCard((payload: SelectionCardPayload) => {
      setAmbientCards([]);
      setSelection({ card: payload.card, cursor_x: payload.cursor_x, cursor_y: payload.cursor_y });
    });
    return () => { unlisten.then((fn) => fn()); };
  }, []);

  // Listen for on-demand trigger (Cmd+Shift+O)
  useEffect(() => {
    const unlisten = listen("trigger-ambient", async () => {
      if (ambientCardsRef.current.length > 0 || selectionRef.current !== null) {
        setAmbientCards([]);
        setSelection(null);
        return;
      }
      setTriggerLoading(true);
      try {
        const result = await triggerAmbient();
        if (result.cards_emitted === 0 && result.reason) {
          const now = Math.floor(Date.now() / 1000);
          setAmbientCards([{
            card_id: `status-${now}`,
            kind: "person_context",
            title: result.context_summary || "No context",
            topic: "",
            body: result.reason,
            sources: [],
            memory_count: 0,
            primary_source_id: "",
            created_at: now,
          }]);
          setTimeout(() => setAmbientCards([]), 4000);
        }
      } catch (e) {
        console.error("[ambient] trigger error:", e);
      } finally {
        setTriggerLoading(false);
      }
    });
    return () => { unlisten.then((fn) => fn()); };
  }, []);

  // Position and show/hide window
  useEffect(() => {
    const win = getCurrentWindow();
    if (selection) {
      positionNearCursor(win, selection.cursor_x, selection.cursor_y);
    } else if (ambientCards.length === 0 && !triggerLoading) {
      // No cards and not loading — hide the window.
      // When ambientCards.length > 0, useLayoutEffect handles show+position.
      win.hide();
    }
  }, [selection, ambientCards.length, triggerLoading]);

  const handleDismissAmbient = useCallback((card: AmbientCard) => {
    setAmbientCards((prev) => prev.filter((c) => c.card_id !== card.card_id));
    dismissAmbientCard(card.title);
  }, []);

  const handleDismissSelection = useCallback(() => {
    setSelection(null);
  }, []);

  // Click on transparent backdrop → dismiss
  const handleBackdropClick = useCallback(() => {
    if (selection) handleDismissSelection();
    else setAmbientCards([]);
  }, [selection, handleDismissSelection]);

  const handleOpenDetail = useCallback(
    (card: AmbientCard, isSelection: boolean) => {
      emit("navigate-to-memory", { sourceId: card.primary_source_id });
      if (isSelection) handleDismissSelection();
      else handleDismissAmbient(card);
    },
    [handleDismissAmbient, handleDismissSelection],
  );

  // Measure actual card content and resize window to fit exactly.
  // The window grows upward from the bottom-right anchor.
  const cardContainerRef = useRef<HTMLDivElement>(null);
  useLayoutEffect(() => {
    if (!cardContainerRef.current) return;
    const contentH = cardContainerRef.current.offsetHeight;
    if (contentH <= 0) return;
    const winH = contentH + WIN_PADDING * 2;
    const win = getCurrentWindow();
    // Re-position with exact height — bottom edge stays pinned
    invoke("position_ambient_bottom_right", { width: WIN_WIDTH, height: winH }).then(() => {
      win.show();
    });
  }, [ambientCards, selection]);

  if (!selection && ambientCards.length === 0) return null;

  return (
    <div
      className="w-full h-full flex flex-col items-end justify-end"
      style={{ padding: WIN_PADDING }}
      onClick={handleBackdropClick}
    >
      <div ref={cardContainerRef} className="flex flex-col" style={{ gap: CARD_GAP, width: CARD_WIDTH }}>
        {selection && (
          <SelectionCard
            key={selection.card.card_id}
            card={selection.card}
            onDismiss={handleDismissSelection}
            onOpenDetail={() => handleOpenDetail(selection.card, true)}
          />
        )}
        {!selection && ambientCards.map((card) => (
          <AmbientCardView
            key={card.card_id}
            card={card}
            onDismiss={() => handleDismissAmbient(card)}
            onOpenDetail={() => handleOpenDetail(card, false)}
          />
        ))}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Selection card — synthesis + memory chips, no auto-dismiss
// ---------------------------------------------------------------------------

function SelectionCard({
  card,
  onDismiss,
  onOpenDetail,
}: {
  card: AmbientCard;
  onDismiss: () => void;
  onOpenDetail: () => void;
}) {
  return (
    <div
      className="rounded-xl overflow-hidden animate-[mem-fade-up_220ms_ease-out]"
      style={{
        background: "var(--mem-surface)",
        border: "1px solid rgba(255,255,255,0.12)",
        boxShadow: "0 4px 20px rgba(0,0,0,0.55), 0 1px 0 rgba(255,255,255,0.06) inset",
      }}
      onClick={(e) => e.stopPropagation()}
    >
      {/* Header */}
      <div className="flex items-center gap-2 px-4 pt-4 pb-2.5">
        <span style={{ color: "var(--mem-accent-warm)", fontSize: 13, lineHeight: 1, flexShrink: 0 }}>◈</span>
        <span
          className="text-[12px] font-semibold flex-1 truncate"
          style={{ color: "var(--mem-text)", fontFamily: "var(--mem-font-body)", letterSpacing: "-0.01em" }}
        >
          {card.title}
        </span>
        <div className="flex items-center gap-0.5 flex-shrink-0">
          {card.primary_source_id && !card.loading && (
            <button
              onClick={onOpenDetail}
              title="Open in app"
              className="flex items-center justify-center w-6 h-6 rounded-md transition-colors"
              style={{ color: "var(--mem-text-tertiary)" }}
              onMouseEnter={(e) => (e.currentTarget.style.background = "var(--mem-hover-strong)")}
              onMouseLeave={(e) => (e.currentTarget.style.background = "transparent")}
            >
              <svg width="11" height="11" viewBox="0 0 12 12" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round">
                <path d="M3 9l6-6M5 3h4v4" />
              </svg>
            </button>
          )}
          <button
            onClick={onDismiss}
            title="Close"
            className="flex items-center justify-center w-6 h-6 rounded-md transition-colors"
            style={{ color: "var(--mem-text-tertiary)" }}
            onMouseEnter={(e) => (e.currentTarget.style.background = "var(--mem-hover-strong)")}
            onMouseLeave={(e) => (e.currentTarget.style.background = "transparent")}
          >
            <svg width="10" height="10" viewBox="0 0 12 12" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
              <path d="M3 3l6 6M9 3l-6 6" />
            </svg>
          </button>
        </div>
      </div>

      {/* Synthesis body */}
      <div className="px-4 pb-3.5">
        {card.loading ? (
          <div className="flex flex-col gap-2 py-0.5">
            <div className="animate-pulse h-3 rounded-md" style={{ background: "var(--mem-border)", width: "92%" }} />
            <div className="animate-pulse h-3 rounded-md" style={{ background: "var(--mem-border)", width: "75%" }} />
            <div className="animate-pulse h-3 rounded-md" style={{ background: "var(--mem-border)", width: "60%" }} />
          </div>
        ) : (
          <p
            className="text-[13px] leading-[1.7]"
            style={{ color: "var(--mem-text-secondary)", fontFamily: "var(--mem-font-body)" }}
          >
            {card.body}
          </p>
        )}
      </div>

      {/* Memory chips */}
      {!card.loading && card.snippets && card.snippets.length > 0 && (
        <>
          <div style={{ height: 1, background: "var(--mem-border)", margin: "0 0" }} />
          <div className="px-4 pt-3 pb-3.5 flex flex-col gap-1.5">
            <span
              className="text-[9px] uppercase mb-0.5"
              style={{
                color: "var(--mem-text-tertiary)",
                fontFamily: "var(--mem-font-mono)",
                letterSpacing: "0.12em",
              }}
            >
              from memory
            </span>
            {card.snippets.map((snippet, i) => (
              <MemoryChip key={i} snippet={snippet} />
            ))}
          </div>
        </>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Ambient card — proactive, auto-dismisses
// ---------------------------------------------------------------------------

function AmbientCardView({
  card,
  onDismiss,
  onOpenDetail,
}: {
  card: AmbientCard;
  onDismiss: () => void;
  onOpenDetail: () => void;
}) {
  const [hovered, setHovered] = useState(false);

  useEffect(() => {
    if (hovered) return;
    const t = setTimeout(onDismiss, AMBIENT_DISMISS_MS);
    return () => clearTimeout(t);
  }, [hovered, onDismiss]);

  return (
    <div
      className="rounded-xl overflow-hidden animate-[mem-fade-up_300ms_ease-out]"
      style={{
        background: "var(--mem-surface)",
        border: "1px solid var(--mem-border)",
        boxShadow: "0 4px 24px rgba(0,0,0,0.35)",
      }}
      onClick={(e) => e.stopPropagation()}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
    >
      <div className="flex items-center gap-2 px-4 pt-3.5 pb-1.5">
        <span
          className="w-1.5 h-1.5 rounded-full flex-shrink-0"
          style={{ background: card.kind === "person_context" ? "var(--mem-accent-warm)" : "var(--mem-accent-amber)" }}
        />
        <span
          className="text-[12px] font-medium flex-1 truncate"
          style={{ color: "var(--mem-text)", fontFamily: "var(--mem-font-body)" }}
        >
          {card.title}
        </span>
        <div className="flex items-center gap-0.5">
          {card.primary_source_id && (
            <button
              onClick={onOpenDetail}
              className="flex items-center justify-center w-6 h-6 rounded-md opacity-0 group-hover:opacity-100 transition-all"
              style={{ color: "var(--mem-text-tertiary)" }}
            >
              <svg width="11" height="11" viewBox="0 0 12 12" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round">
                <path d="M3 9l6-6M5 3h4v4" />
              </svg>
            </button>
          )}
          <button
            onClick={onDismiss}
            className="flex items-center justify-center w-6 h-6 rounded-md"
            style={{ color: "var(--mem-text-tertiary)" }}
          >
            <svg width="10" height="10" viewBox="0 0 12 12" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
              <path d="M3 3l6 6M9 3l-6 6" />
            </svg>
          </button>
        </div>
      </div>
      <div
        className="px-4 pb-3 text-[12px] leading-[1.6]"
        style={{ color: "var(--mem-text-secondary)", fontFamily: "var(--mem-font-body)" }}
      >
        {card.body}
      </div>
      {card.sources.length > 0 && (
        <div className="flex items-center gap-1.5 px-4 pb-3">
          {card.sources.map((src) => <SourceIcon key={src} name={src} />)}
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Memory chip
// ---------------------------------------------------------------------------

function MemoryChip({ snippet }: { snippet: MemorySnippet }) {
  return (
    <div
      className="flex items-center gap-2.5 px-2.5 py-1.5 rounded-lg"
      style={{ background: "var(--mem-hover)" }}
    >
      <SourceIcon name={snippet.source} />
      <span
        className="text-[11px] truncate"
        style={{ color: "var(--mem-text-secondary)", fontFamily: "var(--mem-font-body)" }}
      >
        {snippet.text}
      </span>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Source icon — colored circle with initials
// ---------------------------------------------------------------------------

const SOURCE_PALETTE = [
  "#6C63FF", "#FF6B6B", "#43B89C", "#F5A623", "#4A90D9",
  "#A78BFA", "#FB923C", "#34D399", "#60A5FA", "#F472B6",
];

function sourceColor(name: string): string {
  let h = 0;
  for (let i = 0; i < name.length; i++) h = (h * 31 + name.charCodeAt(i)) >>> 0;
  return SOURCE_PALETTE[h % SOURCE_PALETTE.length];
}

function SourceIcon({ name }: { name: string }) {
  const initials = name.replace(/[^a-zA-Z0-9]/g, "").slice(0, 2).toUpperCase() || "?";
  return (
    <span
      title={name}
      style={{
        display: "inline-flex",
        alignItems: "center",
        justifyContent: "center",
        width: 16,
        height: 16,
        borderRadius: "50%",
        background: sourceColor(name),
        color: "#fff",
        fontSize: 7,
        fontWeight: 700,
        flexShrink: 0,
        letterSpacing: "-0.3px",
      }}
    >
      {initials}
    </span>
  );
}

// ---------------------------------------------------------------------------
// Window positioning
// ---------------------------------------------------------------------------

async function positionNearCursor(
  win: ReturnType<typeof getCurrentWindow>,
  cursorX: number,
  cursorY: number,
) {
  try {
    const monitor = await primaryMonitor();
    if (!monitor) return;
    const scale = monitor.scaleFactor;
    const screenW = monitor.size.width / scale;
    const screenH = monitor.size.height / scale;

    const winH = MAX_WIN_HEIGHT;

    // CG coordinates and Tauri both use top-left origin — no Y flip needed.
    // Place card above the cursor.
    let x = cursorX - WIN_WIDTH / 2;
    let y = cursorY - winH - 12;

    x = Math.max(SCREEN_PADDING, Math.min(x, screenW - WIN_WIDTH - SCREEN_PADDING));
    y = Math.max(SCREEN_PADDING, Math.min(y, screenH - winH - SCREEN_PADDING));

    await win.setSize(new LogicalSize(WIN_WIDTH, winH));
    await win.setPosition(new LogicalPosition(x, y));
    await win.show();
  } catch (e) {
    console.error("[ambient] cursor-position error:", e);
  }
}
