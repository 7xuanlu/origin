// SPDX-License-Identifier: AGPL-3.0-only
import { describe, it, expect, vi, afterEach } from "vitest";
import { render, screen } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import type { ReactNode } from "react";
import { i18n } from "../../i18n";
import AddMemoryForm from "./AddMemoryForm";

vi.mock("../../lib/tauri", () => ({
  quickCapture: vi.fn().mockResolvedValue({}),
}));

function renderForm() {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  function Wrapper({ children }: { children: ReactNode }) {
    return <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>;
  }
  return render(<AddMemoryForm spaces={[]} onClose={vi.fn()} />, { wrapper: Wrapper });
}

describe("AddMemoryForm", () => {
  afterEach(async () => {
    await i18n.changeLanguage("en");
  });

  it("renders English copy by default", () => {
    renderForm();
    expect(screen.getByPlaceholderText("What do you want to remember?")).toBeInTheDocument();
    expect(screen.getByText("Space:")).toBeInTheDocument();
    expect(screen.getByText("No space")).toBeInTheDocument();
    expect(screen.getByText("Cancel")).toBeInTheDocument();
    expect(screen.getByText("Save")).toBeInTheDocument();
  });

  // i18n (S8): all 5 strings used to be hardcoded English — a zh-Hans/zh-Hant
  // user saw an English form. This pins that switching locale actually
  // changes the rendered text, not just that resources.ts has the keys.
  it("renders translated copy in zh-Hans — not the hardcoded English fallback", async () => {
    await i18n.changeLanguage("zh-Hans");
    renderForm();

    expect(screen.getByPlaceholderText("你想记住什么?")).toBeInTheDocument();
    expect(screen.getByText("空间:")).toBeInTheDocument();
    expect(screen.getByText("无空间")).toBeInTheDocument();
    expect(screen.getByText("取消")).toBeInTheDocument();
    expect(screen.getByText("保存")).toBeInTheDocument();
    expect(screen.queryByPlaceholderText("What do you want to remember?")).not.toBeInTheDocument();
    expect(screen.queryByText("Save")).not.toBeInTheDocument();
  });

  // S8-visual mutation-proof (c): raw color: white is banned outright, and
  // the native <select> must be gone in favor of the Select primitive.
  it("has no raw color: white and no native <select> left in the source", async () => {
    const fs = await import("node:fs/promises");
    const path = await import("node:path");
    const filePath = path.join(process.cwd(), "src/components/memory/AddMemoryForm.tsx");
    const source = await fs.readFile(filePath, "utf-8");
    expect(source).not.toMatch(/color:\s*["']white["']/i);
    expect(source).not.toMatch(/<select[\s>]/);
  });
});
