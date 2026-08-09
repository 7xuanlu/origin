import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { UpgradePrompt } from "../UpgradePrompt";

describe("UpgradePrompt", () => {
  it("shows the archive size + estimated local time", () => {
    render(
      <UpgradePrompt
        memoryCount={2143}
        estimatedLocalHours={4}
        onCloud={() => {}}
        onLocal={() => {}}
      />,
    );
    expect(screen.getByText(/2,143-memory archive/i)).toBeInTheDocument();
    expect(screen.getByText(/approximately 4 hours/i)).toBeInTheDocument();
  });

  it("calls onCloud with API key when cloud selected", () => {
    const onCloud = vi.fn();
    render(
      <UpgradePrompt
        memoryCount={2143}
        estimatedLocalHours={4}
        onCloud={onCloud}
        onLocal={() => {}}
      />,
    );
    fireEvent.change(screen.getByPlaceholderText(/anthropic api key/i), {
      target: { value: "sk-ant-test123" },
    });
    fireEvent.click(screen.getByRole("button", { name: /use cloud/i }));
    expect(onCloud).toHaveBeenCalledWith("sk-ant-test123");
  });

  it("calls onLocal when local selected", () => {
    const onLocal = vi.fn();
    render(
      <UpgradePrompt
        memoryCount={2143}
        estimatedLocalHours={4}
        onCloud={() => {}}
        onLocal={onLocal}
      />,
    );
    fireEvent.click(screen.getByRole("button", { name: /continue with local/i }));
    expect(onLocal).toHaveBeenCalled();
  });
});
