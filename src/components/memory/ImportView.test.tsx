import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { ImportView } from "./ImportView";

vi.mock("../../lib/tauri", () => ({
  importMemories: vi.fn(),
}));

import { importMemories } from "../../lib/tauri";

function renderImport(props = {}) {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  return render(
    <QueryClientProvider client={queryClient}>
      <ImportView
        onBack={vi.fn()}
        onComplete={vi.fn()}
        {...props}
      />
    </QueryClientProvider>,
  );
}

describe("ImportView", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("renders input form by default", () => {
    renderImport();
    expect(screen.getByText("Import Memories")).toBeInTheDocument();
    expect(screen.getByPlaceholderText(/paste your memories/i)).toBeInTheDocument();
    expect(screen.getByText("ChatGPT")).toBeInTheDocument();
    expect(screen.getByText("Claude")).toBeInTheDocument();
  });

  it("disables import button when textarea is empty", () => {
    renderImport();
    const button = screen.getByText("Import");
    expect(button).toBeDisabled();
  });

  it("enables import button when textarea has content", () => {
    renderImport();
    const textarea = screen.getByPlaceholderText(/paste your memories/i);
    fireEvent.change(textarea, { target: { value: "User is an engineer" } });
    const button = screen.getByText("Import");
    expect(button).not.toBeDisabled();
  });

  it("shows progress state while importing", async () => {
    let resolveImport: (value: unknown) => void;
    (importMemories as ReturnType<typeof vi.fn>).mockImplementation(
      () => new Promise((resolve) => { resolveImport = resolve; }),
    );

    renderImport();
    const textarea = screen.getByPlaceholderText(/paste your memories/i);
    fireEvent.change(textarea, { target: { value: "Memory 1" } });
    fireEvent.click(screen.getByText("Import"));

    expect(screen.getByText(/Processing your memories/)).toBeInTheDocument();

    resolveImport!({
      imported: 1, skipped: 0,
      breakdown: { fact: 1 },
      entities_created: 0, observations_added: 0, relations_created: 0,
      batch_id: "test",
    });
  });

  it("shows summary after successful import", async () => {
    const mockResult = {
      imported: 3, skipped: 1,
      breakdown: { identity: 1, fact: 2 },
      entities_created: 2, observations_added: 3, relations_created: 1,
      batch_id: "import_123",
    };
    (importMemories as ReturnType<typeof vi.fn>).mockResolvedValue(mockResult);

    renderImport();
    const textarea = screen.getByPlaceholderText(/paste your memories/i);
    fireEvent.change(textarea, { target: { value: "Memory 1\nMemory 2\nMemory 3" } });
    fireEvent.click(screen.getByText("Import"));

    await waitFor(() => {
      expect(screen.getByText(/3 memories imported/i)).toBeInTheDocument();
    });
    expect(screen.getByText(/1 skipped/i)).toBeInTheDocument();
  });

  it("shows type breakdown badges in summary", async () => {
    const mockResult = {
      imported: 3, skipped: 0,
      breakdown: { identity: 1, fact: 2 },
      entities_created: 0, observations_added: 0, relations_created: 0,
      batch_id: "test",
    };
    (importMemories as ReturnType<typeof vi.fn>).mockResolvedValue(mockResult);

    renderImport();
    const textarea = screen.getByPlaceholderText(/paste your memories/i);
    fireEvent.change(textarea, { target: { value: "a\nb\nc" } });
    fireEvent.click(screen.getByText("Import"));

    await waitFor(() => {
      expect(screen.getByText("identity")).toBeInTheDocument();
      expect(screen.getByText("fact")).toBeInTheDocument();
    });
  });

  it("shows KG stats when entities or observations are created", async () => {
    const mockResult = {
      imported: 2, skipped: 0,
      breakdown: { fact: 2 },
      entities_created: 3, observations_added: 5, relations_created: 0,
      batch_id: "test",
    };
    (importMemories as ReturnType<typeof vi.fn>).mockResolvedValue(mockResult);

    renderImport();
    const textarea = screen.getByPlaceholderText(/paste your memories/i);
    fireEvent.change(textarea, { target: { value: "a\nb" } });
    fireEvent.click(screen.getByText("Import"));

    await waitFor(() => {
      expect(screen.getByText(/3 entities/)).toBeInTheDocument();
      expect(screen.getByText(/5 observations/)).toBeInTheDocument();
    });
  });

  it("shows error on import failure", async () => {
    (importMemories as ReturnType<typeof vi.fn>).mockRejectedValue("Import failed: too large");

    renderImport();
    const textarea = screen.getByPlaceholderText(/paste your memories/i);
    fireEvent.change(textarea, { target: { value: "Memory" } });
    fireEvent.click(screen.getByText("Import"));

    await waitFor(() => {
      expect(screen.getByText(/Import failed/i)).toBeInTheDocument();
    });
    // Should go back to input form after error
    expect(screen.getByPlaceholderText(/paste your memories/i)).toBeInTheDocument();
  });

  it("calls onBack when back button clicked", () => {
    const onBack = vi.fn();
    renderImport({ onBack });
    // Back button is the first button (arrow icon, no text)
    const buttons = screen.getAllByRole("button");
    fireEvent.click(buttons[0]);
    expect(onBack).toHaveBeenCalled();
  });

  it("calls onComplete when View memories clicked", async () => {
    const onComplete = vi.fn();
    (importMemories as ReturnType<typeof vi.fn>).mockResolvedValue({
      imported: 1, skipped: 0, breakdown: { fact: 1 },
      entities_created: 0, observations_added: 0, relations_created: 0,
      batch_id: "test",
    });

    renderImport({ onComplete });
    const textarea = screen.getByPlaceholderText(/paste your memories/i);
    fireEvent.change(textarea, { target: { value: "Memory" } });
    fireEvent.click(screen.getByText("Import"));

    await waitFor(() => screen.getByText("View memories"));
    fireEvent.click(screen.getByText("View memories"));
    expect(onComplete).toHaveBeenCalledWith("chatgpt", expect.objectContaining({ imported: expect.any(Number) }));
  });

  it("resets to input form when Import more clicked", async () => {
    (importMemories as ReturnType<typeof vi.fn>).mockResolvedValue({
      imported: 1, skipped: 0, breakdown: { fact: 1 },
      entities_created: 0, observations_added: 0, relations_created: 0,
      batch_id: "test",
    });

    renderImport();
    const textarea = screen.getByPlaceholderText(/paste your memories/i);
    fireEvent.change(textarea, { target: { value: "Memory" } });
    fireEvent.click(screen.getByText("Import"));

    await waitFor(() => screen.getByText("Import more"));
    fireEvent.click(screen.getByText("Import more"));

    // Should be back to input form with empty textarea
    expect(screen.getByText("Import Memories")).toBeInTheDocument();
    expect(screen.getByPlaceholderText(/paste your memories/i)).toHaveValue("");
  });

  it("switches source and shows correct help text", () => {
    renderImport();

    // Default is ChatGPT — shows export prompt with ChatGPT instruction
    expect(screen.getByText(/paste into ChatGPT/i)).toBeInTheDocument();

    // Switch to Claude
    fireEvent.click(screen.getByText("Claude"));
    expect(screen.getByText(/paste into Claude/i)).toBeInTheDocument();

    // Switch to Other
    fireEvent.click(screen.getByText("Other"));
    expect(screen.getByText(/Paste any list/)).toBeInTheDocument();
  });

  it("passes selected source to importMemories", async () => {
    (importMemories as ReturnType<typeof vi.fn>).mockResolvedValue({
      imported: 1, skipped: 0, breakdown: { fact: 1 },
      entities_created: 0, observations_added: 0, relations_created: 0,
      batch_id: "test",
    });

    renderImport();
    fireEvent.click(screen.getByText("Claude"));
    const textarea = screen.getByPlaceholderText(/paste your memories/i);
    fireEvent.change(textarea, { target: { value: "Memory" } });
    fireEvent.click(screen.getByText("Import"));

    await waitFor(() => {
      expect(importMemories).toHaveBeenCalledWith("claude", "Memory");
    });
  });
});
