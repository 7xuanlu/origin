// SPDX-License-Identifier: AGPL-3.0-only
import {
  forwardRef,
  useImperativeHandle,
  useLayoutEffect,
  useRef,
} from "react";
import {
  defaultKeymap,
  history,
  historyKeymap,
  redoDepth,
  undoDepth,
} from "@codemirror/commands";
import { markdown, markdownLanguage } from "@codemirror/lang-markdown";
import { HighlightStyle, syntaxHighlighting } from "@codemirror/language";
import { Compartment, EditorState } from "@codemirror/state";
import {
  drawSelection,
  dropCursor,
  EditorView,
  highlightSpecialChars,
  keymap,
} from "@codemirror/view";
import { tags } from "@lezer/highlight";
import type {
  MarkdownEditorHandle,
  MarkdownEditorProps,
  MarkdownEditorStatus,
} from "./MarkdownEditor";
import { runMarkdownCommand } from "./markdownCommands";
import {
  setWritingCompositionActive,
  writingPresentation,
} from "./writingPresentation";

const sourcePreservingDefaultKeymap = defaultKeymap.filter(
  (binding) => binding.key !== "Mod-Enter" && binding.key !== "Escape",
);

export interface CodeMirrorMarkdownEditorProps extends MarkdownEditorProps {
  onConstructionFailure?(): void;
}

export const CodeMirrorMarkdownEditor = forwardRef<
  MarkdownEditorHandle,
  CodeMirrorMarkdownEditorProps
>(function CodeMirrorMarkdownEditor(props, ref) {
  const mountRef = useRef<HTMLDivElement>(null);
  const viewRef = useRef<EditorView | null>(null);
  const focusRequestedRef = useRef(false);
  const disabledCompartmentRef = useRef(new Compartment());
  const attributesCompartmentRef = useRef(new Compartment());
  const compositionActiveRef = useRef(false);
  const lastStatusRef = useRef<{
    callback: ((status: MarkdownEditorStatus) => void) | undefined;
    key: string;
  } | null>(null);
  const callbacksRef = useRef({
    sessionId: props.sessionId,
    onDocumentChange: props.onDocumentChange,
    onSave: props.onSave,
    onCancel: props.onCancel,
    onConstructionFailure: props.onConstructionFailure,
    onStatusChange: props.onStatusChange,
  });

  const publishStatus = (view: EditorView): void => {
    const status: MarkdownEditorStatus = {
      sessionId: callbacksRef.current.sessionId,
      engine: "codemirror",
      ready: true,
      compositionActive: compositionActiveRef.current,
      canUndo: undoDepth(view.state) > 0,
      canRedo: redoDepth(view.state) > 0,
    };
    const callback = callbacksRef.current.onStatusChange;
    if (!callback) return;
    const key = JSON.stringify(status);
    if (
      lastStatusRef.current?.callback === callback &&
      lastStatusRef.current.key === key
    ) {
      return;
    }
    lastStatusRef.current = { callback, key };
    callback(status);
  };

  const actionsBlocked = (view: EditorView): boolean =>
    view.state.readOnly || compositionActiveRef.current || view.compositionStarted;

  const requestSave = (): boolean => {
    const view = viewRef.current;
    if (!view || actionsBlocked(view)) return false;
    callbacksRef.current.onSave(view.state.doc.toString());
    return true;
  };

  const requestCancel = (): boolean => {
    const view = viewRef.current;
    if (!view || actionsBlocked(view)) return false;
    callbacksRef.current.onCancel();
    return true;
  };

  useImperativeHandle(
    ref,
    () => ({
      focus(): void {
        focusRequestedRef.current = true;
        if (viewRef.current) {
          viewRef.current.focus();
        }
      },
      runCommand(command): boolean {
        const view = viewRef.current;
        if (!view || actionsBlocked(view)) return false;
        const applied = runMarkdownCommand(view, command);
        publishStatus(view);
        return applied;
      },
      requestSave,
      requestCancel,
    }),
    [],
  );

  useLayoutEffect(() => {
    callbacksRef.current = {
      sessionId: props.sessionId,
      onDocumentChange: props.onDocumentChange,
      onSave: props.onSave,
      onCancel: props.onCancel,
      onConstructionFailure: props.onConstructionFailure,
      onStatusChange: props.onStatusChange,
    };
  }, [
    props.onCancel,
    props.onConstructionFailure,
    props.onDocumentChange,
    props.onSave,
    props.onStatusChange,
    props.sessionId,
  ]);

  useLayoutEffect(() => {
    const mount = mountRef.current;
    if (!mount) return;

    let view: EditorView;
    compositionActiveRef.current = false;
    lastStatusRef.current = null;
    try {
      view = new EditorView({
        parent: mount,
        state: EditorState.create({
          doc: props.initialDocument,
          extensions: [
            history(),
            drawSelection(),
            dropCursor(),
            highlightSpecialChars(),
            EditorView.lineWrapping,
            markdown({
              base: markdownLanguage,
              pasteURLAsLink: false,
              completeHTMLTags: false,
            }),
            syntaxHighlighting(
              HighlightStyle.define([
                { tag: tags.heading, color: "var(--mem-accent-page)", fontWeight: "600" },
                { tag: tags.strong, color: "var(--mem-text)", fontWeight: "700" },
                { tag: tags.emphasis, color: "var(--mem-text-secondary)", fontStyle: "italic" },
                { tag: [tags.link, tags.url], color: "var(--mem-accent-indigo)" },
                { tag: tags.monospace, color: "var(--mem-accent-warm)" },
                { tag: [tags.comment, tags.meta], color: "var(--mem-text-tertiary)" },
                { tag: [tags.atom, tags.keyword], color: "var(--mem-accent-sage)" },
              ]),
            ),
            EditorView.theme({
              "&": {
                boxSizing: "border-box",
                width: "100%",
                minHeight: "300px",
                overflow: "hidden",
                color: "var(--mem-text)",
                backgroundColor: "var(--mem-detail-surface)",
                border: "1px solid var(--mem-border)",
                borderRadius: "var(--mem-radius-md)",
                fontSize: "var(--mem-text-md)",
              },
              "&.cm-focused": {
                outline: "none",
                borderColor: "var(--mem-accent-page)",
              },
              ".cm-scroller": {
                minHeight: "300px",
                height: "auto",
                overflow: "visible",
                fontFamily:
                  "ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace",
              },
              ".cm-content": {
                minHeight: "300px",
                padding: "0.75rem",
                caretColor: "var(--mem-accent-indigo)",
              },
              ".cm-cursor": {
                borderLeftColor: "var(--mem-accent-indigo)",
              },
              ".cm-selectionBackground, &.cm-focused .cm-selectionBackground, ::selection": {
                backgroundColor: "var(--mem-indigo-bg)",
              },
            }),
            writingPresentation(),
            EditorView.domEventHandlers({
              compositionstart: (_event, currentView) => {
                if (viewRef.current !== currentView) return false;
                compositionActiveRef.current = true;
                currentView.dispatch({
                  effects: setWritingCompositionActive.of(true),
                });
                publishStatus(currentView);
                return false;
              },
              compositionend: (_event, currentView) => {
                if (viewRef.current !== currentView) return false;
                Promise.resolve().then(() => {
                  if (
                    viewRef.current !== currentView ||
                    currentView.compositionStarted
                  ) {
                    return;
                  }
                  compositionActiveRef.current = false;
                  currentView.dispatch({
                    effects: setWritingCompositionActive.of(false),
                  });
                  publishStatus(currentView);
                });
                return false;
              },
            }),
            keymap.of([
              {
                key: "Mod-Enter",
                run: (currentView) => {
                  if (currentView.state.readOnly) return true;
                  if (actionsBlocked(currentView)) return false;
                  requestSave();
                  return true;
                },
              },
              {
                key: "Escape",
                run: (currentView) => {
                  if (currentView.state.readOnly) return true;
                  if (actionsBlocked(currentView)) return false;
                  requestCancel();
                  return true;
                },
              },
            ]),
            keymap.of(sourcePreservingDefaultKeymap),
            keymap.of(historyKeymap),
            EditorView.updateListener.of((update) => {
              if (update.docChanged) {
                callbacksRef.current.onDocumentChange(update.state.doc.toString());
              }
              publishStatus(update.view);
            }),
            disabledCompartmentRef.current.of(disabledExtensions(props.disabled)),
            attributesCompartmentRef.current.of(contentAttributes(props)),
          ],
        }),
      });
    } catch {
      mount.replaceChildren();
      callbacksRef.current.onConstructionFailure?.();
      return;
    }

    viewRef.current = view;
    publishStatus(view);
    if (focusRequestedRef.current) view.focus();
    return () => {
      if (viewRef.current === view) viewRef.current = null;
      compositionActiveRef.current = false;
      view.destroy();
      mount.replaceChildren();
    };
    // EditorView lifetime is intentionally keyed only by the edit session.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [props.sessionId]);

  useLayoutEffect(() => {
    const view = viewRef.current;
    if (!view) return;
    view.dispatch({
      effects: disabledCompartmentRef.current.reconfigure(disabledExtensions(props.disabled)),
    });
  }, [props.disabled, props.sessionId]);

  useLayoutEffect(() => {
    const view = viewRef.current;
    if (!view) return;
    view.dispatch({
      effects: attributesCompartmentRef.current.reconfigure(contentAttributes(props)),
    });
  }, [props.ariaLabel, props.describedBy, props.disabled, props.sessionId]);

  return (
    <div
      ref={mountRef}
      data-markdown-editor-engine="codemirror"
      style={{ width: "100%", minHeight: "300px" }}
      onKeyDownCapture={(event) => {
        if (
          (event.metaKey || event.ctrlKey) &&
          event.key.toLowerCase() === "s"
        ) {
          event.preventDefault();
          requestSave();
        }
      }}
    />
  );
});

function disabledExtensions(disabled: boolean) {
  return [EditorState.readOnly.of(disabled), EditorView.editable.of(!disabled)];
}

function contentAttributes(props: Pick<MarkdownEditorProps, "ariaLabel" | "describedBy" | "disabled">) {
  return EditorView.contentAttributes.of({
    "aria-label": props.ariaLabel,
    "aria-multiline": "true",
    "aria-disabled": String(props.disabled),
    ...(props.describedBy ? { "aria-describedby": props.describedBy } : {}),
  });
}
