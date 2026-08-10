// SPDX-License-Identifier: AGPL-3.0-only
import { Suspense, startTransition, useState } from "react";
import { act, fireEvent, render, screen } from "@testing-library/react";
import { beforeAll, describe, expect, it, vi } from "vitest";
import { CodeMirrorMarkdownEditor } from "./CodeMirrorMarkdownEditor";
import { NativeMarkdownEditor } from "./NativeMarkdownEditor";
import {
  editorViewFromTextbox,
  installCodeMirrorDomPolyfills,
  pressKey,
  replaceDocument,
} from "./editorTestUtils";

const never = new Promise<void>(() => {});

function NeverCommits(): never {
  throw never;
}

beforeAll(() => {
  installCodeMirrorDomPolyfills();
});

describe("editor concurrent-render safety", () => {
  it("keeps the committed Native session document and callbacks after a new session suspends", async () => {
    const oldChange = vi.fn();
    const newChange = vi.fn();
    const oldSave = vi.fn();
    const newSave = vi.fn();
    let beginAbandonedRender!: () => void;

    function Harness() {
      const [next, setNext] = useState(false);
      beginAbandonedRender = () => startTransition(() => setNext(true));
      return (
        <Suspense fallback={null}>
          <NativeMarkdownEditor
            initialDocument={next ? "new original" : "old original"}
            sessionId={next ? "page:2" : "page:1"}
            disabled={false}
            ariaLabel={next ? "New native" : "Old native"}
            onDocumentChange={next ? newChange : oldChange}
            onSave={next ? newSave : oldSave}
            onCancel={vi.fn()}
            onFallback={vi.fn()}
          />
          {next ? <NeverCommits /> : null}
        </Suspense>
      );
    }

    render(<Harness />);
    const textbox = screen.getByRole("textbox", { name: "Old native" });
    await act(async () => {
      beginAbandonedRender();
      await Promise.resolve();
    });

    expect(screen.getByRole("textbox", { name: "Old native" })).toBe(textbox);
    fireEvent.keyDown(textbox, { key: "s", ctrlKey: true });
    fireEvent.change(textbox, { target: { value: "old changed" } });

    expect(oldSave).toHaveBeenCalledWith("old original");
    expect(oldChange).toHaveBeenCalledWith("old changed");
    expect(newSave).not.toHaveBeenCalled();
    expect(newChange).not.toHaveBeenCalled();
  });

  it("keeps the committed CodeMirror view and callbacks after a new session suspends", async () => {
    const oldChange = vi.fn();
    const newChange = vi.fn();
    const oldSave = vi.fn();
    const newSave = vi.fn();
    let beginAbandonedRender!: () => void;

    function Harness() {
      const [next, setNext] = useState(false);
      beginAbandonedRender = () => startTransition(() => setNext(true));
      return (
        <Suspense fallback={null}>
          <CodeMirrorMarkdownEditor
            initialDocument={next ? "new original" : "old original"}
            sessionId={next ? "page:2" : "page:1"}
            disabled={false}
            ariaLabel={next ? "New CodeMirror" : "Old CodeMirror"}
            onDocumentChange={next ? newChange : oldChange}
            onSave={next ? newSave : oldSave}
            onCancel={vi.fn()}
            onFallback={vi.fn()}
          />
          {next ? <NeverCommits /> : null}
        </Suspense>
      );
    }

    render(<Harness />);
    const textbox = screen.getByRole("textbox", { name: "Old CodeMirror" });
    const view = editorViewFromTextbox(textbox);
    await act(async () => {
      beginAbandonedRender();
      await Promise.resolve();
    });

    expect(screen.getByRole("textbox", { name: "Old CodeMirror" })).toBe(textbox);
    act(() => pressKey(textbox, "s", { ctrlKey: true }));
    act(() => replaceDocument(view, "old changed"));

    expect(oldSave).toHaveBeenCalledWith("old original");
    expect(oldChange).toHaveBeenCalledWith("old changed");
    expect(newSave).not.toHaveBeenCalled();
    expect(newChange).not.toHaveBeenCalled();
  });
});
