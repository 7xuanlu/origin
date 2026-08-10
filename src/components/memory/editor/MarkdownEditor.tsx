// SPDX-License-Identifier: AGPL-3.0-only
import {
  forwardRef,
  useCallback,
  useEffect,
  useImperativeHandle,
  useLayoutEffect,
  useRef,
  useState,
} from "react";
import type { ComponentType, RefAttributes } from "react";
import { NativeMarkdownEditor } from "./NativeMarkdownEditor";
import { loadCodeMirrorEditor } from "./loadCodeMirrorEditor";

export interface MarkdownEditorHandle {
  focus(): void;
  runCommand(command: MarkdownEditorCommand): boolean;
  requestSave(): boolean;
  requestCancel(): boolean;
}

export type MarkdownEditorCommand =
  | "undo"
  | "redo"
  | "paragraph"
  | "heading1"
  | "heading2"
  | "heading3"
  | "bold"
  | "italic"
  | "inlineCode"
  | "link"
  | "blockquote"
  | "bulletList"
  | "numberedList";

export interface MarkdownEditorStatus {
  sessionId: string;
  engine: "loading" | "codemirror" | "native";
  ready: boolean;
  compositionActive: boolean;
  canUndo: boolean;
  canRedo: boolean;
}

export interface MarkdownEditorProps {
  initialDocument: string;
  sessionId: string;
  disabled: boolean;
  ariaLabel: string;
  describedBy?: string;
  onDocumentChange(document: string): void;
  onSave(document: string): void;
  onCancel(): void;
  onFallback(reason: "load" | "construction"): void;
  onStatusChange?(status: MarkdownEditorStatus): void;
}

export const MarkdownEditor = forwardRef<MarkdownEditorHandle, MarkdownEditorProps>(
  function MarkdownEditor(props, ref) {
    return <MarkdownEditorSession key={props.sessionId} ref={ref} {...props} />;
  },
);

const MarkdownEditorSession = forwardRef<MarkdownEditorHandle, MarkdownEditorProps>(
  function MarkdownEditorSession(props, ref) {
    const editorRef = useRef<MarkdownEditorHandle>(null);
    const pendingFocusRef = useRef(false);
    const focusRequestedRef = useRef(false);
    const layoutLifetimeRef = useRef(false);
    const callbacksRef = useRef({
      onFallback: props.onFallback,
      onCancel: props.onCancel,
      onStatusChange: props.onStatusChange,
    });
    const originalDocumentRef = useRef(props.initialDocument);
    const fallbackEmittedRef = useRef(false);
    const [loadState, setLoadState] = useState<LoadState>({
      sessionId: props.sessionId,
      status: "loading",
    });

    useImperativeHandle(
      ref,
      () => ({
        focus(): void {
          focusRequestedRef.current = true;
          if (editorRef.current) {
            editorRef.current.focus();
          } else {
            pendingFocusRef.current = true;
          }
        },
        runCommand(command): boolean {
          return editorRef.current?.runCommand(command) ?? false;
        },
        requestSave(): boolean {
          return editorRef.current?.requestSave() ?? false;
        },
        requestCancel(): boolean {
          if (editorRef.current) return editorRef.current.requestCancel();
          callbacksRef.current.onCancel();
          return true;
        },
      }),
      [],
    );

    useLayoutEffect(() => {
      layoutLifetimeRef.current = true;
      return () => {
        layoutLifetimeRef.current = false;
      };
    }, []);

    useLayoutEffect(() => {
      callbacksRef.current = {
        onFallback: props.onFallback,
        onCancel: props.onCancel,
        onStatusChange: props.onStatusChange,
      };
    }, [props.onCancel, props.onFallback, props.onStatusChange]);

    useLayoutEffect(() => {
      callbacksRef.current.onStatusChange?.({
        sessionId: props.sessionId,
        engine: "loading",
        ready: false,
        compositionActive: false,
        canUndo: false,
        canRedo: false,
      });
      // Loading belongs to the keyed edit session.
      // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [props.sessionId]);

    const setEditorRef = useCallback((editor: MarkdownEditorHandle | null) => {
      editorRef.current = editor;
      if (editor && pendingFocusRef.current) {
        pendingFocusRef.current = false;
        editor.focus();
      }
    }, []);

    const activateFallback = useCallback(
      (sessionId: string, reason: "load" | "construction") => {
        if (sessionId !== props.sessionId) return;
        setLoadState({ sessionId, status: "fallback" });
        if (!fallbackEmittedRef.current) {
          fallbackEmittedRef.current = true;
          callbacksRef.current.onFallback(reason);
        }
      },
      [props.sessionId],
    );

    useEffect(() => {
      let active = true;
      setLoadState({ sessionId: props.sessionId, status: "loading" });

      loadCodeMirrorEditor().then(
        (module) => {
          if (!active || !layoutLifetimeRef.current) return;
          if (!module.CodeMirrorMarkdownEditor) {
            activateFallback(props.sessionId, "load");
            return;
          }
          setLoadState({
            sessionId: props.sessionId,
            status: "loaded",
            Editor: module.CodeMirrorMarkdownEditor,
          });
        },
        () => {
          if (active && layoutLifetimeRef.current) {
            activateFallback(props.sessionId, "load");
          }
        },
      );

      return () => {
        active = false;
      };
    }, [activateFallback, props.sessionId]);

    if (loadState.sessionId !== props.sessionId || loadState.status === "loading") {
      return null;
    }

    const sessionProps = {
      ...props,
      initialDocument: originalDocumentRef.current,
    };

    if (loadState.status === "fallback") {
      return (
        <>
          <span
            data-testid="markdown-editor-fallback"
            data-markdown-editor-fallback="true"
            aria-hidden="true"
          />
          <NativeMarkdownEditor ref={setEditorRef} {...sessionProps} />
        </>
      );
    }

    const LoadedEditor = loadState.Editor;
    return (
      <LoadedEditor
        ref={setEditorRef}
        {...sessionProps}
        onConstructionFailure={() => {
          if (focusRequestedRef.current) pendingFocusRef.current = true;
          activateFallback(props.sessionId, "construction");
        }}
      />
    );
  },
);

type LoadedEditor = ComponentType<
  MarkdownEditorProps &
    RefAttributes<MarkdownEditorHandle> & {
      onConstructionFailure?(): void;
    }
>;

type LoadState =
  | { sessionId: string; status: "loading" }
  | { sessionId: string; status: "fallback" }
  | { sessionId: string; status: "loaded"; Editor: LoadedEditor };
