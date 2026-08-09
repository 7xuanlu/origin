// SPDX-License-Identifier: AGPL-3.0-only
import { useLayoutEffect, useRef, useState } from "react";
import { open as shellOpen } from "@tauri-apps/plugin-shell";
import type { MemoryItem, PageCitation } from "../../../lib/tauri";
import { relativeMs } from "./format";

interface CitationPopoverProps {
  id: string;
  citation: PageCitation;
  sourceMemory: MemoryItem | null;
  sourcesLoading: boolean;
  anchorRef: React.RefObject<HTMLButtonElement | null>;
  onOpenMemory: (sourceId: string) => void;
}

const WIDTH = 280;

// Spec: external_url shows a *domain* badge, other kinds a fixed label.
function kindBadge(citation: PageCitation): string {
  if (citation.source_kind === "external_url") {
    try {
      return new URL(citation.locator).hostname;
    } catch {
      return "Web";
    }
  }
  return { memory: "Source memory", external_file: "File", authored: "Authored" }[
    citation.source_kind
  ];
}

export default function CitationPopover({
  id,
  citation,
  sourceMemory,
  sourcesLoading,
  anchorRef,
  onOpenMemory,
}: CitationPopoverProps) {
  const boxRef = useRef<HTMLDivElement>(null);
  const [pos, setPos] = useState<{ top: number; left: number } | null>(null);

  // Minimal viewport collision handling: below the chip by default, flip
  // above when it would overflow the bottom, clamp horizontally.
  useLayoutEffect(() => {
    const anchor = anchorRef.current?.getBoundingClientRect();
    if (!anchor) return;
    const height = boxRef.current?.getBoundingClientRect().height ?? 120;
    const flip =
      anchor.bottom + height + 8 > window.innerHeight && anchor.top - height - 8 > 0;
    setPos({
      top: flip ? anchor.top - height - 8 : anchor.bottom + 4,
      left: Math.min(Math.max(anchor.left, 8), Math.max(window.innerWidth - WIDTH - 8, 8)),
    });
  }, [anchorRef]);

  const mono = {
    fontFamily: "var(--mem-font-mono)",
    fontSize: "10px",
    color: "var(--mem-text-tertiary)",
  } as const;
  const bodyText = {
    fontFamily: "var(--mem-font-body)",
    fontSize: "12px",
    color: "var(--mem-text-secondary)",
    lineHeight: 1.5,
  } as const;
  const actionStyle = {
    fontFamily: "var(--mem-font-body)",
    fontSize: "11px",
    fontWeight: 500,
    color: "var(--mem-accent-indigo)",
    background: "none",
    border: "none",
    padding: 0,
    cursor: "pointer",
  } as const;

  const snippet = sourceMemory?.content
    ? sourceMemory.content.replace(/\s+/g, " ").trim().slice(0, 200)
    : null;

  function body() {
    if (citation.source_kind === "authored") {
      return (
        <p style={bodyText}>
          Written by you in this page — kept unchanged when the page is
          re-distilled from its sources.
        </p>
      );
    }
    if (citation.source_kind === "external_file") {
      return (
        <>
          <p style={{ ...mono, wordBreak: "break-all" }}>{citation.locator}</p>
          <button style={actionStyle} onClick={() => void shellOpen(citation.locator)}>
            Open file →
          </button>
        </>
      );
    }
    if (citation.source_kind === "external_url") {
      return (
        <>
          <p style={{ ...mono, wordBreak: "break-all" }}>{citation.locator}</p>
          <button style={actionStyle} onClick={() => void shellOpen(citation.locator)}>
            Open in browser →
          </button>
        </>
      );
    }
    // memory
    if (sourcesLoading && !sourceMemory) {
      return (
        <div data-testid="citation-popover-skeleton" className="flex flex-col gap-1.5">
          <div style={{ width: "70%", height: "10px", background: "var(--mem-hover)", borderRadius: "4px" }} />
          <div style={{ width: "90%", height: "10px", background: "var(--mem-hover)", borderRadius: "4px" }} />
        </div>
      );
    }
    if (!sourceMemory) {
      return (
        <>
          <p style={mono}>{citation.locator}</p>
          <p style={{ ...bodyText, fontStyle: "italic" }}>
            This source memory no longer exists — it was deleted or merged
            after distillation. Re-distill the page to refresh its citations.
          </p>
        </>
      );
    }
    return (
      <>
        {sourceMemory.title && (
          <p
            style={{
              fontFamily: "var(--mem-font-heading)",
              fontSize: "13px",
              fontWeight: 500,
              color: "var(--mem-text)",
              lineHeight: 1.4,
            }}
          >
            {sourceMemory.title}
          </p>
        )}
        <p style={mono}>
          {citation.locator}
          {sourceMemory.last_modified
            ? ` · ${relativeMs(sourceMemory.last_modified * 1000)}`
            : ""}
        </p>
        {snippet && <p style={bodyText}>{snippet}</p>}
        <button style={actionStyle} onClick={() => onOpenMemory(citation.locator)}>
          Open memory →
        </button>
      </>
    );
  }

  return (
    <div
      ref={boxRef}
      id={id}
      role="tooltip"
      className="flex flex-col gap-1.5 rounded-lg p-3"
      style={{
        position: "fixed",
        top: pos?.top ?? -9999,
        left: pos?.left ?? -9999,
        width: `${WIDTH}px`,
        zIndex: 50,
        backgroundColor: "var(--mem-surface)",
        border: "1px solid var(--mem-border)",
        boxShadow: "0 4px 12px rgba(0,0,0,0.3)",
      }}
    >
      <div className="flex items-center gap-2">
        <span
          style={{
            fontFamily: "var(--mem-font-mono)",
            fontSize: "10px",
            color: "var(--mem-text-tertiary)",
            background: "var(--mem-hover)",
            padding: "1px 5px",
            borderRadius: "3px",
          }}
        >
          {kindBadge(citation)}
        </span>
        {citation.status === "unverified" && (
          <span
            style={{
              fontFamily: "var(--mem-font-mono)",
              fontSize: "10px",
              color: "var(--mem-accent-amber)",
            }}
          >
            unverified
          </span>
        )}
      </div>
      {body()}
    </div>
  );
}
