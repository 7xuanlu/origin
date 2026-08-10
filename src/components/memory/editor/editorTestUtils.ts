// SPDX-License-Identifier: AGPL-3.0-only
import { EditorSelection } from "@codemirror/state";
import { EditorView } from "@codemirror/view";

export function installCodeMirrorDomPolyfills(): void {
  if (!Range.prototype.getClientRects) {
    Object.defineProperty(Range.prototype, "getClientRects", {
      configurable: true,
      value: () => [],
    });
  }
}

export function editorViewFromTextbox(textbox: HTMLElement): EditorView {
  const view = EditorView.findFromDOM(textbox);
  if (!view) throw new Error("Expected a live CodeMirror EditorView");
  return view;
}

export function replaceDocument(view: EditorView, document: string): void {
  view.dispatch({
    changes: { from: 0, to: view.state.doc.length, insert: document },
  });
}

export function selectRange(view: EditorView, anchor: number, head = anchor): void {
  view.dispatch({ selection: EditorSelection.single(anchor, head) });
}

export function pressKey(
  target: HTMLElement,
  key: string,
  options: KeyboardEventInit = {},
): KeyboardEvent {
  const event = new KeyboardEvent("keydown", {
    bubbles: true,
    cancelable: true,
    key,
    ...options,
  });
  target.dispatchEvent(event);
  return event;
}

export function pasteText(target: HTMLElement, text: string): ClipboardEvent {
  const event = new Event("paste", {
    bubbles: true,
    cancelable: true,
  }) as ClipboardEvent;
  Object.defineProperty(event, "clipboardData", {
    value: {
      getData: (format: string) =>
        format === "text/plain" || format === "Text" ? text : "",
      types: ["text/plain"],
    },
  });
  target.dispatchEvent(event);
  return event;
}
