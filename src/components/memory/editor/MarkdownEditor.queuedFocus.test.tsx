// SPDX-License-Identifier: AGPL-3.0-only
import { createRef, forwardRef, useImperativeHandle } from "react";
import { act, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { MarkdownEditor } from "./MarkdownEditor";
import { loadCodeMirrorEditor } from "./loadCodeMirrorEditor";
import type { MarkdownEditorHandle } from "./MarkdownEditor";

vi.mock("./loadCodeMirrorEditor", () => ({ loadCodeMirrorEditor: vi.fn() }));

function deferredLoad() {
  let resolve!: (module: Record<string, unknown>) => void;
  const promise = new Promise<Record<string, unknown>>((done) => {
    resolve = done;
  });
  const focus = vi.fn();
  return { promise, resolve, focus };
}

afterEach(() => vi.clearAllMocks());

describe("MarkdownEditor queued focus", () => {
  it("flushes focus requested during loading exactly once, then forwards later calls directly", async () => {
    const load = deferredLoad();
    vi.mocked(loadCodeMirrorEditor).mockReturnValue(load.promise as never);
    const ref = createRef<MarkdownEditorHandle>();
    render(
      <MarkdownEditor
        ref={ref}
        initialDocument="source"
        sessionId="page:1"
        disabled={false}
        ariaLabel="Source"
        onDocumentChange={vi.fn()}
        onSave={vi.fn()}
        onCancel={vi.fn()}
        onFallback={vi.fn()}
      />,
    );

    act(() => ref.current?.focus());
    expect(load.focus).not.toHaveBeenCalled();

    await act(async () => {
      load.resolve({
        CodeMirrorMarkdownEditor: forwardRef<MarkdownEditorHandle>(function MockEditor(
          _props,
          childRef,
        ) {
          useImperativeHandle(childRef, () => ({
            focus: load.focus,
            runCommand: () => false,
            requestSave: () => false,
            requestCancel: () => false,
          }));
          return <div role="textbox" aria-label="Source" />;
        }),
      });
      await load.promise;
    });
    await screen.findByRole("textbox", { name: "Source" });
    expect(load.focus).toHaveBeenCalledTimes(1);

    act(() => ref.current?.focus());
    expect(load.focus).toHaveBeenCalledTimes(2);
  });
});
