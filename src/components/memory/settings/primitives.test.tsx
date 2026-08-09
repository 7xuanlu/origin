// SPDX-License-Identifier: AGPL-3.0-only
import { useState } from "react";
import { describe, expect, it, beforeEach, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { i18n } from "../../../i18n";
import {
  Button,
  Field,
  Input,
  Select,
  SectionHeader,
  SettingRow,
  StatusChip,
  Tag,
  Toggle,
} from "./primitives";

beforeEach(async () => {
  await i18n.changeLanguage("en");
});

describe("StatusChip", () => {
  it("renders the success token triplet for an up state", () => {
    render(<StatusChip state={{ kind: "up", detail: "Ollama 0.6.2" }} label="Ollama" />);
    const chip = screen.getByText(/Ollama/).closest("span[aria-live]") as HTMLElement;
    expect(chip.className).toContain("bg-[var(--mem-status-success-bg)]");
    expect(chip.className).toContain("text-[var(--mem-status-success-text)]");
    expect(chip.className).toContain("border-[var(--mem-status-success-border)]");
    expect(screen.getByText("Ollama · Ollama 0.6.2")).toBeInTheDocument();
  });

  it("renders the danger token triplet for a down state", () => {
    render(<StatusChip state={{ kind: "down", detail: "connection refused" }} label="Ollama" />);
    const chip = screen.getByText(/Ollama/).closest("span[aria-live]") as HTMLElement;
    expect(chip.className).toContain("bg-[var(--mem-status-danger-bg)]");
    expect(chip.className).toContain("text-[var(--mem-status-danger-text)]");
    expect(chip.className).toContain("border-[var(--mem-status-danger-border)]");
  });

  it("shows the not-verified i18n string for a stale state", () => {
    render(<StatusChip state={{ kind: "stale" }} label="Ollama" />);
    expect(screen.getByText("Ollama · Not verified yet")).toBeInTheDocument();
  });

  it("marks the container aria-live=polite and the dot aria-hidden", () => {
    const { container } = render(<StatusChip state={{ kind: "idle" }} label="Ollama" />);
    const chip = screen.getByText("Ollama").closest("span[aria-live]");
    expect(chip).toHaveAttribute("aria-live", "polite");
    const dot = container.querySelector('[aria-hidden="true"]');
    expect(dot).not.toBeNull();
  });
});

describe("Field", () => {
  it("wires the control's aria-describedby to the description id", () => {
    render(
      <Field label="Endpoint" htmlFor="endpoint" description="Where the daemon listens">
        <Input />
      </Field>,
    );
    const input = screen.getByLabelText("Endpoint");
    expect(input).toHaveAttribute("aria-describedby", "endpoint-desc");
    expect(screen.getByText("Where the daemon listens")).toHaveAttribute("id", "endpoint-desc");
  });

  it("renders error text outside the <label> element and wins over description", () => {
    render(
      <Field label="Endpoint" htmlFor="endpoint" description="helper" error="Invalid host">
        <Input />
      </Field>,
    );
    const input = screen.getByLabelText("Endpoint");
    expect(input).toHaveAttribute("aria-describedby", "endpoint-error");
    const errorEl = screen.getByRole("alert");
    expect(errorEl).toHaveTextContent("Invalid host");
    const label = screen.getByText("Endpoint");
    expect(label.tagName).toBe("LABEL");
    expect(label).not.toContainElement(errorEl);
    expect(screen.queryByText("helper")).not.toBeInTheDocument();
  });

  it("uses the shared eyebrow tracking token, not a hardcoded letter-spacing", () => {
    render(
      <Field label="Endpoint" htmlFor="endpoint">
        <Input />
      </Field>,
    );
    const label = screen.getByText("Endpoint");
    expect(label).toHaveStyle({ letterSpacing: "var(--mem-tracking-eyebrow)" });
  });
});

describe("Toggle", () => {
  function ControlledToggle() {
    const [enabled, setEnabled] = useState(false);
    return <Toggle enabled={enabled} onToggle={() => setEnabled((e) => !e)} />;
  }

  it("reflects enabled via aria-pressed and flips it on click", async () => {
    const user = userEvent.setup();
    render(<ControlledToggle />);
    const button = screen.getByRole("button");
    expect(button).toHaveAttribute("aria-pressed", "false");
    await user.click(button);
    expect(button).toHaveAttribute("aria-pressed", "true");
  });
});

describe("Button", () => {
  it("disables the button when disabled is set", () => {
    render(<Button disabled>Save</Button>);
    expect(screen.getByRole("button", { name: "Save" })).toBeDisabled();
  });

  it("renders every variant", () => {
    const variants = ["primary", "secondary", "ghost", "danger"] as const;
    for (const variant of variants) {
      render(<Button variant={variant}>{variant}</Button>);
      expect(screen.getByRole("button", { name: variant })).toBeInTheDocument();
    }
  });

  it("wraps icon+text children in a flex span, so preflight's svg{display:block} can't force a line break", () => {
    render(
      <Button>
        <svg data-testid="btn-icon" />
        <span>Show config</span>
      </Button>,
    );
    const icon = screen.getByTestId("btn-icon");
    const wrapper = icon.parentElement as HTMLElement;
    expect(wrapper.className).toContain("inline-flex");
    expect(wrapper.className).toContain("items-center");
  });
});

describe("Select", () => {
  it("renders its options and fires onChange with the selected value", () => {
    const onChange = vi.fn();
    render(
      <Select aria-label="Language" value="en" onChange={onChange}>
        <option value="en">English</option>
        <option value="zh-Hans">简体中文</option>
      </Select>,
    );
    const select = screen.getByRole("combobox", { name: "Language" }) as HTMLSelectElement;
    expect(screen.getByRole("option", { name: "English" })).toBeInTheDocument();
    expect(screen.getByRole("option", { name: "简体中文" })).toBeInTheDocument();
    fireEvent.change(select, { target: { value: "zh-Hans" } });
    expect(onChange).toHaveBeenCalledTimes(1);
  });

  it("wires into Field like Input does: id + aria-describedby via cloneElement", () => {
    render(
      <Field label="Language" htmlFor="lang" description="Interface language">
        <Select value="en" onChange={() => {}}>
          <option value="en">English</option>
        </Select>
      </Field>,
    );
    const select = screen.getByLabelText("Language");
    expect(select).toHaveAttribute("id", "lang");
    expect(select).toHaveAttribute("aria-describedby", "lang-desc");
  });

  it("hides the chevron glyph from assistive tech", () => {
    const { container } = render(
      <Select aria-label="Language" value="en" onChange={() => {}}>
        <option value="en">English</option>
      </Select>,
    );
    const chevron = container.querySelector('svg')?.closest('[aria-hidden="true"]');
    expect(chevron).not.toBeNull();
  });
});

describe("Tag — chip-never-lies invariant", () => {
  it("renders its label and defaults to the neutral tone", () => {
    render(<Tag>Beta</Tag>);
    const tag = screen.getByText("Beta");
    expect(tag.className).toContain("bg-transparent");
    expect(tag.className).toContain("text-[var(--mem-text-secondary)]");
    expect(tag.className).toContain("border-[var(--mem-border)]");
  });

  it("renders the accent tone on request", () => {
    render(<Tag tone="accent">New</Tag>);
    const tag = screen.getByText("New");
    expect(tag.className).toContain("bg-[var(--mem-indigo-bg)]");
    expect(tag.className).toContain("text-[var(--mem-accent-indigo)]");
    expect(tag.className).toContain("border-transparent");
  });

  it("the tone union has no probe vocabulary, and StatusChip's state cannot be faked with a string (compile-time only, see @ts-expect-error below; pnpm exec tsc -b fails if either stops being an error)", () => {
    // @ts-expect-error — Tag has no probe vocabulary
    <Tag tone="up">Configured</Tag>;
    // @ts-expect-error — StatusChip color cannot be faked with a string
    <StatusChip state="recommended" label="x" />;
    expect(true).toBe(true);
  });
});

describe("SectionHeader", () => {
  it("renders without an icon", () => {
    const { container } = render(<SectionHeader label="Profile" />);
    expect(screen.getByText("Profile")).toBeInTheDocument();
    expect(container.querySelector('[aria-hidden="true"]')).toBeNull();
  });

  it("still renders a given icon", () => {
    const { container } = render(
      <SectionHeader label="Profile" icon={<svg data-testid="icon" />} />,
    );
    expect(container.querySelector('[aria-hidden="true"]')).not.toBeNull();
    expect(screen.getByTestId("icon")).toBeInTheDocument();
  });

  it("renders an action slot, right-aligned via justify-between", () => {
    const { container } = render(
      <SectionHeader label="Diagnostics" action={<button>Refresh</button>} />,
    );
    expect(screen.getByRole("button", { name: "Refresh" })).toBeInTheDocument();
    expect(container.firstElementChild?.className).toContain("justify-between");
  });
});

describe("Toggle — accessible name", () => {
  // The switch renders only a decorative knob, so its accessible name is its
  // ONLY description. Unnamed, a screen reader says "button, pressed" and never
  // says what was toggled. This was true of the hand-rolled role="switch" it
  // replaced, so the bug predates the primitive — pin it so it stays fixed.
  it("is reachable by name, not just by role", () => {
    render(<Toggle enabled={false} onToggle={() => {}} aria-label="Launch at login" />);
    expect(screen.getByRole("button", { name: "Launch at login" })).toBeInTheDocument();
  });

  it("SettingRow names its Toggle with the row title", () => {
    render(
      <SettingRow title="Launch at login" description="Start Wenlan on boot" enabled onToggle={() => {}} />,
    );
    const toggle = screen.getByRole("button", { name: "Launch at login" });
    expect(toggle).toHaveAttribute("aria-pressed", "true");
  });
});

describe("SettingRow", () => {
  it("toggle arm: renders title/description, flips via the Toggle, and wires error to aria-describedby (unchanged behavior)", async () => {
    function ControlledRow() {
      const [enabled, setEnabled] = useState(false);
      return (
        <SettingRow
          title="Run at login"
          description="Start Wenlan automatically"
          enabled={enabled}
          onToggle={() => setEnabled((e) => !e)}
          error="Could not save"
        />
      );
    }
    const user = userEvent.setup();
    render(<ControlledRow />);
    expect(screen.getByText("Run at login")).toBeInTheDocument();
    expect(screen.getByText("Start Wenlan automatically")).toBeInTheDocument();
    const toggle = screen.getByRole("button");
    expect(toggle).toHaveAttribute("aria-pressed", "false");
    expect(toggle).toHaveAttribute("aria-describedby", expect.stringContaining("error"));
    await user.click(toggle);
    expect(toggle).toHaveAttribute("aria-pressed", "true");
  });

  it("control arm: renders the given control instead of a Toggle", () => {
    render(
      <SettingRow
        title="Language"
        description="Interface language"
        control={<div data-testid="custom-control">picker</div>}
      />,
    );
    expect(screen.getByText("Language")).toBeInTheDocument();
    expect(screen.getByTestId("custom-control")).toBeInTheDocument();
    expect(screen.queryByRole("button")).not.toBeInTheDocument();
  });
});
