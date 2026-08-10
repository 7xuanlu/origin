// SPDX-License-Identifier: AGPL-3.0-only
import { StrictMode, useLayoutEffect } from "react";
import { render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { MarkdownEditor } from "./MarkdownEditor";
import { loadCodeMirrorEditor } from "./loadCodeMirrorEditor";

vi.mock("./loadCodeMirrorEditor", () => ({ loadCodeMirrorEditor: vi.fn() }));

afterEach(() => vi.clearAllMocks());

describe("MarkdownEditor construction fallback", () => {
  it("falls back once with the original document when construction reports failure", async () => {
    function ConstructionFailure(props: { onConstructionFailure(): void }) {
      useLayoutEffect(() => {
        props.onConstructionFailure();
      }, [props.onConstructionFailure]);
      return null;
    }
    vi.mocked(loadCodeMirrorEditor).mockResolvedValue({
      CodeMirrorMarkdownEditor: ConstructionFailure,
    } as never);
    const onFallback = vi.fn();
    render(
      <StrictMode>
        <MarkdownEditor
          initialDocument="construction original"
          sessionId="page:1"
          disabled={false}
          ariaLabel="Native after construction"
          onDocumentChange={vi.fn()}
          onSave={vi.fn()}
          onCancel={vi.fn()}
          onFallback={onFallback}
        />
      </StrictMode>,
    );

    expect(await screen.findByRole("textbox", { name: "Native after construction" })).toHaveValue(
      "construction original",
    );
    expect(onFallback).toHaveBeenCalledTimes(1);
    expect(onFallback).toHaveBeenCalledWith("construction");
  });
});
