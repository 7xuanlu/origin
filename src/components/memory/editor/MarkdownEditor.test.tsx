// SPDX-License-Identifier: AGPL-3.0-only
import { createRef, StrictMode } from "react";
import { act, render, screen, waitFor } from "@testing-library/react";
import { beforeAll, describe, expect, it, vi } from "vitest";
import { MarkdownEditor } from "./MarkdownEditor";
import type { MarkdownEditorHandle, MarkdownEditorProps } from "./MarkdownEditor";
import type { MarkdownEditorStatus } from "./MarkdownEditor";
import {
  editorViewFromTextbox,
  installCodeMirrorDomPolyfills,
  replaceDocument,
  selectRange,
} from "./editorTestUtils";

beforeAll(() => {
  installCodeMirrorDomPolyfills();
});

const baseProps = (overrides: Partial<MarkdownEditorProps> = {}): MarkdownEditorProps => ({
  initialDocument: "source",
  sessionId: "page:1",
  disabled: false,
  ariaLabel: "Source",
  onDocumentChange: vi.fn(),
  onSave: vi.fn(),
  onCancel: vi.fn(),
  onFallback: vi.fn(),
  ...overrides,
});

describe("MarkdownEditor", () => {
  it("reports loading once per session across same-session rerenders", async () => {
    const statuses: MarkdownEditorStatus[] = [];
    const props = baseProps({
      onStatusChange: (status) => statuses.push(status),
    });
    const { rerender } = render(<MarkdownEditor {...props} />);
    await screen.findByRole("textbox", { name: "Source" });
    const loadingCount = statuses.filter((status) => status.engine === "loading").length;

    rerender(
      <MarkdownEditor
        {...props}
        disabled
      />,
    );

    expect(statuses.filter((status) => status.engine === "loading")).toHaveLength(
      loadingCount,
    );
  });

  it("forwards save, cancel, and commands to the loaded engine handle", async () => {
    const ref = createRef<MarkdownEditorHandle>();
    const onSave = vi.fn();
    const onCancel = vi.fn();
    render(
      <MarkdownEditor
        ref={ref}
        {...baseProps({ onSave, onCancel })}
      />,
    );
    const textbox = await screen.findByRole("textbox", { name: "Source" });
    act(() => replaceDocument(editorViewFromTextbox(textbox), "live document"));

    expect(ref.current?.requestSave).toBeTypeOf("function");
    expect(ref.current?.requestSave()).toBe(true);
    expect(onSave).toHaveBeenCalledWith("live document");
    act(() => selectRange(editorViewFromTextbox(textbox), 0, "live document".length));
    expect(ref.current?.runCommand("bold")).toBe(true);
    expect(editorViewFromTextbox(textbox).state.doc.toString()).toBe("**live document**");
    expect(ref.current?.requestCancel()).toBe(true);
    expect(onCancel).toHaveBeenCalledOnce();
  });

  it("preserves focus requested before the real lazy CodeMirror view exists", async () => {
    const ref = createRef<MarkdownEditorHandle>();
    render(<MarkdownEditor ref={ref} {...baseProps()} />);

    act(() => ref.current?.focus());

    await screen.findByRole("textbox", { name: "Source" });
    await waitFor(() => expect(screen.getByRole("textbox", { name: "Source" })).toHaveFocus());
  });

  it("preserves queued focus on the surviving real view after StrictMode effect replay", async () => {
    const ref = createRef<MarkdownEditorHandle>();
    render(
      <StrictMode>
        <MarkdownEditor ref={ref} {...baseProps({ sessionId: "strict:1" })} />
      </StrictMode>,
    );

    act(() => ref.current?.focus());

    await screen.findByRole("textbox", { name: "Source" });
    await waitFor(() => expect(screen.getByRole("textbox", { name: "Source" })).toHaveFocus());
  });

  it("manually loads the real CodeMirror leaf and forwards later focus calls", async () => {
    const ref = createRef<MarkdownEditorHandle>();
    render(<MarkdownEditor ref={ref} {...baseProps()} />);
    const textbox = await screen.findByRole("textbox", { name: "Source" });

    act(() => ref.current?.focus());

    expect(textbox).toHaveFocus();
  });
});
