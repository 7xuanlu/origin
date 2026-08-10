// SPDX-License-Identifier: AGPL-3.0-only
import { useLayoutEffect, useRef, useState } from "react";
import { ArrowClockwiseIcon } from "@phosphor-icons/react/dist/csr/ArrowClockwise";
import { ArrowCounterClockwiseIcon } from "@phosphor-icons/react/dist/csr/ArrowCounterClockwise";
import { CodeIcon } from "@phosphor-icons/react/dist/csr/Code";
import { LinkIcon } from "@phosphor-icons/react/dist/csr/Link";
import { ListBulletsIcon } from "@phosphor-icons/react/dist/csr/ListBullets";
import { ListNumbersIcon } from "@phosphor-icons/react/dist/csr/ListNumbers";
import { QuotesIcon } from "@phosphor-icons/react/dist/csr/Quotes";
import { TextBIcon } from "@phosphor-icons/react/dist/csr/TextB";
import { TextItalicIcon } from "@phosphor-icons/react/dist/csr/TextItalic";
import type { Icon } from "@phosphor-icons/react";
import type {
  MarkdownEditorCommand,
  MarkdownEditorStatus,
} from "./MarkdownEditor";

export interface MarkdownEditorToolbarLabels {
  undo: string;
  redo: string;
  blockStyle: string;
  paragraph: string;
  heading1: string;
  heading2: string;
  heading3: string;
  bold: string;
  italic: string;
  inlineCode: string;
  link: string;
  blockquote: string;
  bulletList: string;
  numberedList: string;
  save: string;
  cancel: string;
}

export interface MarkdownEditorToolbarProps {
  status: MarkdownEditorStatus;
  labels: MarkdownEditorToolbarLabels;
  saveDisabled: boolean;
  cancelDisabled: boolean;
  onCommand(command: MarkdownEditorCommand): void;
  onSave(): void;
  onCancel(): void;
}

type FormatAction = Readonly<{
  command: MarkdownEditorCommand;
  label: keyof MarkdownEditorToolbarLabels;
  icon: Icon;
}>;

const INLINE_ACTIONS: readonly FormatAction[] = [
  { command: "bold", label: "bold", icon: TextBIcon },
  { command: "italic", label: "italic", icon: TextItalicIcon },
  { command: "inlineCode", label: "inlineCode", icon: CodeIcon },
  { command: "link", label: "link", icon: LinkIcon },
];

const BLOCK_ACTIONS: readonly FormatAction[] = [
  { command: "blockquote", label: "blockquote", icon: QuotesIcon },
  { command: "bulletList", label: "bulletList", icon: ListBulletsIcon },
  { command: "numberedList", label: "numberedList", icon: ListNumbersIcon },
];

const HISTORY_ACTIONS: readonly FormatAction[] = [
  { command: "undo", label: "undo", icon: ArrowCounterClockwiseIcon },
  { command: "redo", label: "redo", icon: ArrowClockwiseIcon },
];

type HorizontalOverflowCue = "none" | "start" | "end" | "both";

function horizontalOverflowCue(element: HTMLElement): HorizontalOverflowCue {
  if (element.scrollWidth <= element.clientWidth + 1) return "none";
  const atStart = element.scrollLeft <= 1;
  const atEnd =
    element.scrollLeft + element.clientWidth >= element.scrollWidth - 1;
  if (atStart) return "end";
  if (atEnd) return "start";
  return "both";
}

export function MarkdownEditorToolbar({
  status,
  labels,
  saveDisabled,
  cancelDisabled,
  onCommand,
  onSave,
  onCancel,
}: MarkdownEditorToolbarProps) {
  const formatRowRef = useRef<HTMLDivElement>(null);
  const [formatOverflowCue, setFormatOverflowCue] =
    useState<HorizontalOverflowCue>("none");
  const formatDisabled = !status.ready || status.compositionActive;
  const saveBlockedByComposition =
    status.compositionActive && status.ready && !saveDisabled;
  const cancelBlockedByComposition = status.compositionActive && !cancelDisabled;
  const formatLabelsKey = [
    labels.blockStyle,
    labels.bold,
    labels.italic,
    labels.inlineCode,
    labels.link,
    labels.blockquote,
    labels.bulletList,
    labels.numberedList,
    labels.undo,
    labels.redo,
  ].join("\u0000");

  useLayoutEffect(() => {
    const row = formatRowRef.current;
    if (!row) {
      setFormatOverflowCue("none");
      return;
    }

    const updateCue = () => setFormatOverflowCue(horizontalOverflowCue(row));
    updateCue();
    window.addEventListener("resize", updateCue);
    const resizeObserver =
      typeof ResizeObserver === "undefined"
        ? null
        : new ResizeObserver(updateCue);
    resizeObserver?.observe(row);
    return () => {
      window.removeEventListener("resize", updateCue);
      resizeObserver?.disconnect();
    };
  }, [formatLabelsKey, status.engine]);

  const renderActions = (actions: readonly FormatAction[]) =>
    actions.map(({ command, label, icon: ActionIcon }) => {
      const actionLabel = labels[label];
      const historyDisabled =
        (command === "undo" && !status.canUndo) ||
        (command === "redo" && !status.canRedo);
      const blockedByComposition =
        status.compositionActive && status.ready && !historyDisabled;
      return (
        <button
          key={command}
          type="button"
          className="page-editor-format-action"
          title={actionLabel}
          aria-label={actionLabel}
          data-composition-blocked={blockedByComposition ? "true" : undefined}
          disabled={formatDisabled || historyDisabled}
          onPointerDown={(event) => event.preventDefault()}
          onClick={() => onCommand(command)}
        >
          <ActionIcon
            size={16}
            weight="regular"
            color="currentColor"
            aria-hidden="true"
            focusable="false"
          />
        </button>
      );
    });

  return (
    <div
      className="page-editor-toolbar"
      role="toolbar"
      aria-label={labels.blockStyle}
      data-composition-active={status.compositionActive ? "true" : undefined}
    >
      {status.engine === "codemirror" ? (
        <div
          ref={formatRowRef}
          className="page-editor-format-row"
          role="group"
          aria-label={labels.blockStyle}
          data-overflow-cue={formatOverflowCue}
          onScroll={(event) =>
            setFormatOverflowCue(horizontalOverflowCue(event.currentTarget))
          }
        >
          <div className="page-editor-format-group" data-toolbar-group="inline">
            {renderActions(INLINE_ACTIONS)}
          </div>
          <div
            className="page-editor-format-group"
            data-toolbar-group="block-style"
          >
            <select
              aria-label={labels.blockStyle}
              value=""
              data-composition-blocked={
                status.compositionActive && status.ready ? "true" : undefined
              }
              disabled={formatDisabled}
              onChange={(event) => {
                const command = event.currentTarget.value as MarkdownEditorCommand;
                if (command) onCommand(command);
              }}
            >
              <option value="" disabled>
                {labels.blockStyle}
              </option>
              <option value="paragraph">{labels.paragraph}</option>
              <option value="heading1">{labels.heading1}</option>
              <option value="heading2">{labels.heading2}</option>
              <option value="heading3">{labels.heading3}</option>
            </select>
          </div>
          <div className="page-editor-format-group" data-toolbar-group="block">
            {renderActions(BLOCK_ACTIONS)}
          </div>
          <div className="page-editor-format-group" data-toolbar-group="history">
            {renderActions(HISTORY_ACTIONS)}
          </div>
        </div>
      ) : null}

      <div className="page-editor-persistence-actions">
        <button
          type="button"
          title={labels.save}
          aria-label={labels.save}
          data-composition-blocked={
            saveBlockedByComposition ? "true" : undefined
          }
          disabled={saveDisabled || !status.ready || status.compositionActive}
          onClick={onSave}
        >
          {labels.save}
        </button>
        <button
          type="button"
          title={labels.cancel}
          aria-label={labels.cancel}
          data-composition-blocked={
            cancelBlockedByComposition ? "true" : undefined
          }
          disabled={cancelDisabled || status.compositionActive}
          onClick={onCancel}
        >
          {labels.cancel}
        </button>
      </div>
    </div>
  );
}
