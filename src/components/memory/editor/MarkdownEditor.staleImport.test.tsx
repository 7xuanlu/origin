// SPDX-License-Identifier: AGPL-3.0-only
import { useLayoutEffect } from "react";
import { act, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { MarkdownEditor } from "./MarkdownEditor";
import { loadCodeMirrorEditor } from "./loadCodeMirrorEditor";

vi.mock("./loadCodeMirrorEditor", () => ({ loadCodeMirrorEditor: vi.fn() }));

function deferredLoad() {
  let resolve!: (module: Record<string, unknown>) => void;
  const promise = new Promise<Record<string, unknown>>((done) => {
    resolve = done;
  });
  return { promise, resolve };
}

function controlledRejection() {
  let rejectContinuation: ((reason: Error) => void) | undefined;
  const thenable = {
    then(
      _resolve: (module: Record<string, unknown>) => void,
      reject: (reason: Error) => void,
    ) {
      rejectContinuation = reject;
    },
  };
  return {
    thenable,
    reject(reason: Error) {
      if (!rejectContinuation) throw new Error("Import continuation was not registered");
      rejectContinuation(reason);
    },
  };
}

afterEach(() => vi.clearAllMocks());

describe("MarkdownEditor stale import handling", () => {
  it("ignores loads after unmount/session change and falls back with the current original", async () => {
    const load = deferredLoad();
    vi.mocked(loadCodeMirrorEditor).mockReturnValue(load.promise as never);
    const unmountedFallback = vi.fn();
    const old = render(
      <MarkdownEditor
        initialDocument="unmounted"
        sessionId="gone:1"
        disabled={false}
        ariaLabel="Gone"
        onDocumentChange={vi.fn()}
        onSave={vi.fn()}
        onCancel={vi.fn()}
        onFallback={unmountedFallback}
      />,
    );
    old.unmount();

    const onFallback = vi.fn();
    const current = render(
      <MarkdownEditor
        initialDocument="old original"
        sessionId="page:1"
        disabled={false}
        ariaLabel="Source"
        onDocumentChange={vi.fn()}
        onSave={vi.fn()}
        onCancel={vi.fn()}
        onFallback={onFallback}
      />,
    );
    current.rerender(
      <MarkdownEditor
        initialDocument="new original"
        sessionId="page:2"
        disabled={false}
        ariaLabel="Source"
        onDocumentChange={vi.fn()}
        onSave={vi.fn()}
        onCancel={vi.fn()}
        onFallback={onFallback}
      />,
    );

    await act(async () => {
      load.resolve({});
      await load.promise;
    });

    await waitFor(() => expect(onFallback).toHaveBeenCalledTimes(1));
    expect(unmountedFallback).not.toHaveBeenCalled();
    expect(onFallback).toHaveBeenCalledWith("load");
    expect(screen.getByRole("textbox", { name: "Source" })).toHaveValue("new original");
  });

  it("ignores an old import rejected during replacement layout before passive cleanup", async () => {
    const oldLoad = controlledRejection();
    const nextLoad = deferredLoad();
    vi.mocked(loadCodeMirrorEditor).mockReset();
    vi.mocked(loadCodeMirrorEditor)
      .mockReturnValueOnce(oldLoad.thenable as never)
      .mockReturnValueOnce(nextLoad.promise as never);
    const oldFallback = vi.fn();
    const nextFallback = vi.fn();

    function RejectOldLoadInLayout() {
      useLayoutEffect(() => {
        oldLoad.reject(new Error("old session load failed"));
      }, []);
      return null;
    }

    const view = render(
      <MarkdownEditor
        initialDocument="old"
        sessionId="page:old"
        disabled={false}
        ariaLabel="Old source"
        onDocumentChange={vi.fn()}
        onSave={vi.fn()}
        onCancel={vi.fn()}
        onFallback={oldFallback}
      />,
    );

    view.rerender(
      <>
        <MarkdownEditor
          initialDocument="new"
          sessionId="page:new"
          disabled={false}
          ariaLabel="New source"
          onDocumentChange={vi.fn()}
          onSave={vi.fn()}
          onCancel={vi.fn()}
          onFallback={nextFallback}
        />
        <RejectOldLoadInLayout />
      </>,
    );
    await act(async () => {
      await Promise.resolve();
    });

    expect(oldFallback).not.toHaveBeenCalled();
    expect(nextFallback).not.toHaveBeenCalled();
    expect(screen.queryByTestId("markdown-editor-fallback")).toBeNull();
  });
});
