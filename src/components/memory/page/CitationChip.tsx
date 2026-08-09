// SPDX-License-Identifier: AGPL-3.0-only
import { useEffect, useId, useRef, useState } from "react";
import { open as shellOpen } from "@tauri-apps/plugin-shell";
import type { MemoryItem, PageCitation } from "../../../lib/tauri";
import { citationDisplayLabel } from "../../../lib/pageCitations";
import CitationPopover from "./CitationPopover";

interface CitationChipProps {
  occurrence: number;
  citation: PageCitation;
  sourceMemory: MemoryItem | null;
  sourcesLoading: boolean;
  onOpenMemory: (sourceId: string) => void;
}

const HOVER_OPEN_DELAY_MS = 150;
const HOVER_CLOSE_GRACE_MS = 120;

export default function CitationChip({
  occurrence,
  citation,
  sourceMemory,
  sourcesLoading,
  onOpenMemory,
}: CitationChipProps) {
  const [open, setOpen] = useState(false);
  const chipRef = useRef<HTMLButtonElement>(null);
  const openTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const closeTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const lastPointerType = useRef("mouse");
  const popoverId = useId();

  const clearTimers = () => {
    if (openTimer.current) clearTimeout(openTimer.current);
    if (closeTimer.current) clearTimeout(closeTimer.current);
  };
  useEffect(() => clearTimers, []);

  const activate = () => {
    // Touch has no hover: first tap opens the popover, its buttons navigate.
    if (lastPointerType.current === "touch" && !open) {
      setOpen(true);
      return;
    }
    if (citation.source_kind === "memory") {
      // A deleted/merged source has no detail view to land on — explain in
      // the popover instead of navigating to a blank page.
      if (!sourceMemory && !sourcesLoading) {
        setOpen((v) => !v);
        return;
      }
      onOpenMemory(citation.locator);
    } else if (citation.source_kind === "external_url") {
      void shellOpen(citation.locator);
    } else {
      setOpen((v) => !v);
    }
  };

  const unverified = citation.status === "unverified";

  return (
    <span
      style={{ position: "relative", display: "inline-block" }}
      onMouseEnter={() => {
        if (closeTimer.current) clearTimeout(closeTimer.current);
        openTimer.current = setTimeout(() => setOpen(true), HOVER_OPEN_DELAY_MS);
      }}
      onMouseLeave={() => {
        if (openTimer.current) clearTimeout(openTimer.current);
        closeTimer.current = setTimeout(() => setOpen(false), HOVER_CLOSE_GRACE_MS);
      }}
      onFocus={() => setOpen(true)}
      onBlur={(e) => {
        // Keep open while focus moves into the popover (its action button).
        if (!e.currentTarget.contains(e.relatedTarget as Node | null)) setOpen(false);
      }}
      onKeyDown={(e) => {
        if (e.key === "Escape") {
          clearTimers();
          setOpen(false);
        }
      }}
    >
      <button
        ref={chipRef}
        type="button"
        data-status={citation.status}
        aria-describedby={open ? popoverId : undefined}
        onPointerDown={(e) => {
          lastPointerType.current = e.pointerType;
        }}
        onClick={activate}
        className="focus-visible:outline-2 focus-visible:outline-[var(--mem-accent-indigo)]"
        style={{
          fontFamily: "var(--mem-font-mono)",
          fontSize: "11px",
          lineHeight: 1,
          color: unverified ? "var(--mem-text-tertiary)" : "var(--mem-accent-indigo)",
          background: "var(--mem-hover)",
          border: unverified
            ? "1px dashed var(--mem-border)"
            : "1px solid transparent",
          borderRadius: "4px",
          padding: "1px 4px",
          margin: "0 2px",
          verticalAlign: "baseline",
          cursor: "pointer",
        }}
      >
        {citationDisplayLabel(citation)}
        <sup style={{ marginLeft: "1px" }}>{occurrence}</sup>
      </button>
      {open && (
        <CitationPopover
          id={popoverId}
          citation={citation}
          sourceMemory={sourceMemory}
          sourcesLoading={sourcesLoading}
          anchorRef={chipRef}
          onOpenMemory={onOpenMemory}
        />
      )}
    </span>
  );
}
