// SPDX-License-Identifier: AGPL-3.0-only
import { forwardRef, useImperativeHandle, useLayoutEffect, useRef } from "react";
import type { MarkdownEditorHandle, MarkdownEditorProps } from "./MarkdownEditor";

function resizeToDocument(textarea: HTMLTextAreaElement): void {
  textarea.style.height = "auto";
  textarea.style.height = `${Math.max(textarea.scrollHeight, 300)}px`;
}

export const NativeMarkdownEditor = forwardRef<MarkdownEditorHandle, MarkdownEditorProps>(
  function NativeMarkdownEditor(props, ref) {
    return <NativeMarkdownEditorSession key={props.sessionId} ref={ref} {...props} />;
  },
);

const NativeMarkdownEditorSession = forwardRef<MarkdownEditorHandle, MarkdownEditorProps>(
  function NativeMarkdownEditorSession(props, ref) {
    const textareaRef = useRef<HTMLTextAreaElement>(null);
    const documentRef = useRef(props.initialDocument);
    const composingRef = useRef(false);
    const disabledRef = useRef(props.disabled);
    const callbacksRef = useRef({
      onDocumentChange: props.onDocumentChange,
      onSave: props.onSave,
      onCancel: props.onCancel,
      onStatusChange: props.onStatusChange,
    });

    const publishStatus = (compositionActive: boolean): void => {
      callbacksRef.current.onStatusChange?.({
        sessionId: props.sessionId,
        engine: "native",
        ready: true,
        compositionActive,
        canUndo: false,
        canRedo: false,
      });
    };

    const requestSave = (): boolean => {
      if (disabledRef.current || composingRef.current) return false;
      callbacksRef.current.onSave(documentRef.current);
      return true;
    };

    const requestCancel = (): boolean => {
      if (disabledRef.current || composingRef.current) return false;
      callbacksRef.current.onCancel();
      return true;
    };

    useImperativeHandle(
      ref,
      () => ({
        focus(): void {
          textareaRef.current?.focus();
        },
        runCommand(): boolean {
          return false;
        },
        requestSave,
        requestCancel,
      }),
      [],
    );

    useLayoutEffect(() => {
      callbacksRef.current = {
        onDocumentChange: props.onDocumentChange,
        onSave: props.onSave,
        onCancel: props.onCancel,
        onStatusChange: props.onStatusChange,
      };
      disabledRef.current = props.disabled;
    }, [props.disabled, props.onCancel, props.onDocumentChange, props.onSave, props.onStatusChange]);

    useLayoutEffect(() => {
      publishStatus(false);
      if (textareaRef.current) resizeToDocument(textareaRef.current);
      // A NativeMarkdownEditorSession is keyed by sessionId, so this effect runs
      // once for the live session and cannot leak composition state forward.
      // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [props.sessionId]);

    return (
      <textarea
        ref={textareaRef}
        className="page-editor-native-surface"
        defaultValue={documentRef.current}
        disabled={props.disabled}
        aria-label={props.ariaLabel}
        aria-describedby={props.describedBy}
        aria-disabled={props.disabled}
        aria-multiline="true"
        style={{
          boxSizing: "border-box",
          width: "100%",
          minHeight: "300px",
          resize: "none",
          overflow: "hidden",
          padding: "0.75rem",
          color: "var(--mem-text)",
          background: "var(--mem-detail-surface)",
          border: "1px solid var(--mem-border)",
          borderRadius: "var(--mem-radius-md)",
          fontFamily:
            "ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace",
          fontSize: "var(--mem-text-md)",
          lineHeight: 1.6,
          outline: "none",
        }}
        onChange={(event) => {
          documentRef.current = event.currentTarget.value;
          resizeToDocument(event.currentTarget);
          callbacksRef.current.onDocumentChange(documentRef.current);
        }}
        onCompositionStart={() => {
          composingRef.current = true;
          publishStatus(true);
        }}
        onCompositionEnd={() => {
          composingRef.current = false;
          publishStatus(false);
        }}
        onKeyDown={(event) => {
          const mod = event.metaKey || event.ctrlKey;
          if (mod && event.key.toLowerCase() === "s") {
            event.preventDefault();
            requestSave();
            return;
          }
          if (mod && event.key === "Enter") {
            if (!composingRef.current) event.preventDefault();
            requestSave();
            return;
          }
          if (event.key === "Escape" && !composingRef.current) {
            event.preventDefault();
            requestCancel();
          }
        }}
      />
    );
  },
);
