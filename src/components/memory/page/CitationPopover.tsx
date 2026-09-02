// SPDX-License-Identifier: AGPL-3.0-only
import { useLayoutEffect, useRef, useState } from "react";
import { useTranslation } from "react-i18next";
import { open as shellOpen } from "@tauri-apps/plugin-shell";
import { openFile, type MemoryItem, type PageCitation } from "../../../lib/tauri";
import { relativeMs } from "./format";

interface CitationPopoverProps {
  id: string;
  citation: PageCitation;
  sourceMemory: MemoryItem | null;
  sourcesLoading: boolean;
  anchorRef: React.RefObject<HTMLButtonElement | null>;
  onOpenMemory: (sourceId: string) => void;
  /** Why the last attempt to open this citation's file or link was refused. */
  openFailure: string | null;
  onOpenTarget: () => void;
}

const WIDTH = 280;

// A document citation's locator is the document source id
// (`<source id>::<absolute path>`); only the path part is a file the shell can
// open or a human wants to read. Web and memory locators pass through unchanged.
export function citationFilePath(locator: string): string {
  const sep = locator.lastIndexOf("::");
  return sep === -1 ? locator : locator.slice(sep + 2);
}

// Opens the file or link a citation points at and returns the refusal text
// instead of dropping it. Files go through the app's own `open_file` command:
// the shell plugin's `open` only admits URL schemes (its default validator is
// `^((mailto:\w+)|(tel:\w+)|(https?://\w+)).+`), so a bare path handed to it
// is refused inside the plugin, and a discarded rejection is exactly how #656
// shipped a button that did nothing.
export async function openCitationTarget(citation: PageCitation): Promise<string | null> {
  try {
    if (citation.source_kind === "external_file") {
      await openFile(citationFilePath(citation.locator));
    } else {
      await shellOpen(citation.locator);
    }
    return null;
  } catch (err) {
    return err instanceof Error ? err.message : String(err);
  }
}

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
  openFailure,
  onOpenTarget,
}: CitationPopoverProps) {
  const { t } = useTranslation();
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

  function failureNotice(headline: "citation.openFileFailed" | "citation.openLinkFailed") {
    if (openFailure === null) return null;
    return (
      <div role="alert" className="flex flex-col gap-1">
        <p style={{ ...bodyText, color: "var(--mem-accent-amber)" }}>{t(headline)}</p>
        <p style={{ ...mono, wordBreak: "break-word" }}>{openFailure}</p>
      </div>
    );
  }

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
      const filePath = citationFilePath(citation.locator);
      return (
        <>
          <p style={{ ...mono, wordBreak: "break-all" }}>{filePath}</p>
          <button style={actionStyle} onClick={onOpenTarget}>
            Open file →
          </button>
          {failureNotice("citation.openFileFailed")}
        </>
      );
    }
    if (citation.source_kind === "external_url") {
      return (
        <>
          <p style={{ ...mono, wordBreak: "break-all" }}>{citation.locator}</p>
          <button style={actionStyle} onClick={onOpenTarget}>
            Open in browser →
          </button>
          {failureNotice("citation.openLinkFailed")}
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
        // A chip inside the italic pull quote must not hand its slant down.
        fontStyle: "normal",
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
