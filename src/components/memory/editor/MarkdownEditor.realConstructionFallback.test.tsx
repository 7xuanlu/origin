// SPDX-License-Identifier: AGPL-3.0-only
import { createRef } from "react";
import { act, cleanup, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { MarkdownEditor } from "./MarkdownEditor";
import type { MarkdownEditorHandle } from "./MarkdownEditor";

vi.mock("@codemirror/view", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@codemirror/view")>();

  class ThrowingEditorView {
    static contentAttributes = actual.EditorView.contentAttributes;
    static editable = actual.EditorView.editable;
    static lineWrapping = actual.EditorView.lineWrapping;
    static theme = actual.EditorView.theme;
    static updateListener = actual.EditorView.updateListener;

    constructor(config: { parent?: Element }) {
      const partialMount = document.createElement("span");
      partialMount.dataset.partialCodemirrorMount = "true";
      config.parent?.append(partialMount);
      throw new Error("forced EditorView construction failure");
    }
  }

  return { ...actual, EditorView: ThrowingEditorView };
});

afterEach(() => cleanup());

describe("MarkdownEditor real construction fallback", () => {
  it("clears a partial EditorView mount, reports once, and preserves the original draft", async () => {
    const onFallback = vi.fn();
    const view = render(
      <MarkdownEditor
        initialDocument="preserved construction draft"
        sessionId="page:1"
        disabled={false}
        ariaLabel="Construction fallback"
        onDocumentChange={vi.fn()}
        onSave={vi.fn()}
        onCancel={vi.fn()}
        onFallback={onFallback}
      />,
    );

    expect(await screen.findByRole("textbox", { name: "Construction fallback" })).toHaveValue(
      "preserved construction draft",
    );
    expect(document.querySelector("[data-partial-codemirror-mount]")).toBeNull();
    expect(onFallback).toHaveBeenCalledTimes(1);
    expect(onFallback).toHaveBeenCalledWith("construction");

    view.unmount();
    expect(screen.queryByRole("textbox", { name: "Construction fallback" })).toBeNull();
    expect(document.querySelector("[data-partial-codemirror-mount]")).toBeNull();
  });

  it("carries focus queued for the failed CodeMirror view into the Native fallback", async () => {
    const ref = createRef<MarkdownEditorHandle>();
    render(
      <MarkdownEditor
        ref={ref}
        initialDocument="focused fallback draft"
        sessionId="page:focus"
        disabled={false}
        ariaLabel="Focused construction fallback"
        onDocumentChange={vi.fn()}
        onSave={vi.fn()}
        onCancel={vi.fn()}
        onFallback={vi.fn()}
      />,
    );

    act(() => ref.current?.focus());

    expect(
      await screen.findByRole("textbox", { name: "Focused construction fallback" }),
    ).toHaveFocus();
  });
});
