// SPDX-License-Identifier: AGPL-3.0-only
import { cloneElement, useId, useState } from "react";
import { useTranslation } from "react-i18next";

export function Toggle({
  enabled,
  onToggle,
  "aria-label": ariaLabel,
  "aria-describedby": ariaDescribedby,
}: {
  enabled: boolean;
  onToggle: () => void;
  /** Required in practice: the switch renders only a decorative knob, so
   *  without this a screen reader announces "button, pressed" and never says
   *  WHAT is being toggled. Callers pass the visible row/panel title. */
  "aria-label"?: string;
  "aria-describedby"?: string;
}) {
  return (
    <button
      onClick={onToggle}
      aria-pressed={enabled}
      aria-label={ariaLabel}
      aria-describedby={ariaDescribedby}
      className={`relative w-11 h-[26px] rounded-full transition-colors shrink-0 focus-visible:outline-2 focus-visible:outline-[var(--mem-focus-ring)] focus-visible:outline-offset-2 ${
        enabled ? "bg-[var(--mem-accent-indigo)]" : "bg-[var(--mem-hover-strong)]"
      }`}
    >
      <span
        className={`absolute top-[3px] w-5 h-5 rounded-full bg-white shadow-sm transition-transform ${
          enabled ? "left-[22px]" : "left-[3px]"
        }`}
      />
    </button>
  );
}

export function WarningTriangleIcon({ className }: { className?: string }) {
  return (
    <svg aria-hidden="true" className={className} fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeWidth={2}
        d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.964-.833-2.732 0L4.072 16.5c-.77.833.192 2.5 1.732 2.5z"
      />
    </svg>
  );
}

type SettingRowBaseProps = {
  title: string;
  description: string;
  statusLine?: React.ReactNode;
  warning?: string | null;
  error?: string | null;
};

type SettingRowProps = SettingRowBaseProps &
  (
    | { enabled: boolean; onToggle: () => void; control?: never } // toggle row (today's API)
    | { control: React.ReactNode; enabled?: never; onToggle?: never } // custom-control row
  );

export function SettingRow(props: SettingRowProps) {
  const { title, description, statusLine, warning, error } = props;
  const rowId = useId();
  const errorId = `${rowId}-error`;
  const warningId = `${rowId}-warning`;
  const describedBy =
    [error ? errorId : null, warning ? warningId : null].filter(Boolean).join(" ") || undefined;

  return (
    <div className="px-5 py-4">
      <div className="flex items-start justify-between gap-4">
        <div className="min-w-0">
          <div style={{ fontFamily: "var(--mem-font-body)", fontSize: "var(--mem-text-md)", fontWeight: 500, color: "var(--mem-text)" }}>
            {title}
          </div>
          <p style={{ fontFamily: "var(--mem-font-body)", fontSize: "var(--mem-text-sm)", color: "var(--mem-text-secondary)", marginTop: "2px", lineHeight: "1.5" }}>
            {description}
          </p>
        </div>
        <div className="mt-0.5">
          {"control" in props ? (
            props.control
          ) : (
            <Toggle
              enabled={props.enabled}
              onToggle={props.onToggle}
              aria-label={title}
              aria-describedby={describedBy}
            />
          )}
        </div>
      </div>
      {statusLine && <div className="mt-2">{statusLine}</div>}
      {error && (
        <p
          id={errorId}
          style={{ fontFamily: "var(--mem-font-body)", fontSize: "11px", color: "var(--mem-status-danger-text)", marginTop: "8px", lineHeight: "1.5", wordBreak: "break-all" }}
        >
          {error}
        </p>
      )}
      {warning && (
        <div className="flex items-start gap-2 mt-2">
          <WarningTriangleIcon className="w-3.5 h-3.5 text-[var(--mem-status-warning-text)] shrink-0 mt-px" />
          <p
            id={warningId}
            style={{ fontFamily: "var(--mem-font-body)", fontSize: "11px", color: "var(--mem-status-warning-text)", lineHeight: "1.5" }}
          >
            {warning}
          </p>
        </div>
      )}
    </div>
  );
}

export function SectionHeader({
  icon,
  label,
  action,
}: {
  icon?: React.ReactNode;
  label: string;
  action?: React.ReactNode; // right-aligned slot, e.g. Diagnostics' Refresh button
}) {
  return (
    <div className="flex items-center justify-between gap-2 mb-3 px-1">
      <div className="flex items-center gap-2">
        {icon && (
          <span aria-hidden="true" style={{ color: "var(--mem-text-tertiary)" }}>
            {icon}
          </span>
        )}
        <h3
          style={{
            fontFamily: "var(--mem-font-mono)",
            fontSize: "var(--mem-text-2xs)",
            fontWeight: 500,
            lineHeight: 1.2,
            letterSpacing: "var(--mem-tracking-eyebrow)",
            color: "var(--mem-text-tertiary)",
            textTransform: "uppercase" as const,
          }}
        >
          {label}
        </h3>
      </div>
      {action}
    </div>
  );
}

// ── Button ──────────────────────────────────────────────────────────────

export type ButtonVariant = "primary" | "secondary" | "ghost" | "danger";
export type ButtonSize = "sm" | "md";

export interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: ButtonVariant; // default "secondary"
  size?: ButtonSize; // default "md"
  loading?: boolean; // disables; spinner overlays label; width preserved
}

const BUTTON_VARIANT_CLASS: Record<ButtonVariant, string> = {
  primary:
    "bg-[var(--mem-accent-indigo)] text-[var(--mem-text-on-accent)] border border-transparent hover:bg-[var(--mem-accent-indigo-hover)]",
  secondary:
    "bg-transparent text-[var(--mem-text)] border border-[var(--mem-border)] hover:bg-[var(--mem-hover)]",
  ghost:
    "bg-transparent text-[var(--mem-text-secondary)] border border-transparent hover:text-[var(--mem-text)] hover:bg-[var(--mem-hover)]",
  danger:
    "bg-[var(--mem-status-danger-bg)] text-[var(--mem-status-danger-text)] border border-[var(--mem-status-danger-border)] hover:border-[var(--mem-status-danger-text)]",
};

const BUTTON_SIZE_CLASS: Record<ButtonSize, string> = {
  md: "h-[32px] px-[14px] rounded-[var(--mem-radius-md)]",
  sm: "h-[26px] px-[10px] rounded-[var(--mem-radius-sm)]",
};

export function Button({
  variant = "secondary",
  size = "md",
  loading = false,
  disabled,
  className,
  style,
  children,
  ...rest
}: ButtonProps) {
  return (
    <button
      {...rest}
      disabled={disabled || loading}
      className={[
        "relative inline-flex items-center justify-center gap-1.5 font-medium",
        "transition-[background-color,border-color,color,transform] duration-[var(--mem-dur-fast)]",
        "active:scale-[0.98] motion-reduce:active:scale-100",
        "focus-visible:outline-2 focus-visible:outline-[var(--mem-focus-ring)] focus-visible:outline-offset-2",
        "disabled:opacity-45 disabled:cursor-default",
        BUTTON_VARIANT_CLASS[variant],
        BUTTON_SIZE_CLASS[size],
        className ?? "",
      ]
        .filter(Boolean)
        .join(" ")}
      style={{
        fontFamily: "var(--mem-font-body)",
        fontSize: size === "md" ? "var(--mem-text-base)" : "var(--mem-text-sm)",
        ...style,
      }}
    >
      <span className={["inline-flex items-center gap-1.5", loading ? "invisible" : ""].filter(Boolean).join(" ")}>
        {children}
      </span>
      {loading && (
        <svg
          aria-hidden="true"
          className="absolute w-3.5 h-3.5 animate-spin"
          fill="none"
          viewBox="0 0 24 24"
        >
          <circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="3" opacity="0.25" />
          <path d="M4 12a8 8 0 018-8" stroke="currentColor" strokeWidth="3" strokeLinecap="round" />
        </svg>
      )}
    </button>
  );
}

// ── SegmentedControl ────────────────────────────────────────────────────
// One choice among 2–4 peers. Neutral raised thumb: the active segment sits
// on a surface fill + raised shadow instead of an accent slab — accent is
// reserved for primary actions, and a theme picker is not one. No sliding
// thumb element: each segment paints its own active state, so there is no
// translateX math to break and nothing to suppress for reduced motion.

export interface SegmentedOption<T extends string> {
  value: T;
  label: string;
  icon?: React.ReactNode;
}

export function SegmentedControl<T extends string>({
  options,
  value,
  onChange,
  "aria-label": ariaLabel,
}: {
  options: SegmentedOption<T>[];
  value: T;
  onChange: (value: T) => void;
  /** Group name for screen readers — the visible segments only name the choices. */
  "aria-label": string;
}) {
  return (
    <div
      role="group"
      aria-label={ariaLabel}
      className="flex gap-0.5 p-0.5 rounded-[var(--mem-radius-md)] bg-[var(--mem-hover)] w-fit shrink-0"
    >
      {options.map((opt) => {
        const active = opt.value === value;
        return (
          <button
            key={opt.value}
            type="button"
            aria-pressed={active}
            onClick={() => onChange(opt.value)}
            className={[
              "flex items-center gap-1.5 px-3 h-[26px] rounded-[var(--mem-radius-sm)] border",
              "transition-[background-color,border-color,color,box-shadow] duration-[var(--mem-dur-fast)]",
              "focus-visible:outline-2 focus-visible:outline-[var(--mem-focus-ring)] focus-visible:outline-offset-0",
              active
                ? "bg-[var(--mem-surface)] border-[var(--mem-border)] text-[var(--mem-text)] shadow-[var(--mem-shadow-raised)]"
                : "bg-transparent border-transparent text-[var(--mem-text-secondary)] hover:text-[var(--mem-text)]",
            ].join(" ")}
            style={{ fontFamily: "var(--mem-font-body)", fontSize: "var(--mem-text-xs)", fontWeight: 500 }}
          >
            {opt.icon}
            {opt.label}
          </button>
        );
      })}
    </div>
  );
}

// ── Skeleton ────────────────────────────────────────────────────────────
// Loading placeholder bar. aria-hidden: the container that owns the load
// carries the busy semantics; bars are decoration.

export function Skeleton({
  width = "100%",
  height = 12,
  className,
}: {
  width?: string | number;
  height?: number;
  className?: string;
}) {
  return (
    <span
      aria-hidden="true"
      className={[
        "block animate-pulse rounded-[var(--mem-radius-sm)] bg-[var(--mem-hover-strong)]",
        className ?? "",
      ]
        .filter(Boolean)
        .join(" ")}
      style={{ width, height }}
    />
  );
}

// ── ConfirmActionButton — inline two-step destructive confirm ──────────
// First activation swaps the trigger, in place, for a danger Confirm plus a
// ghost Cancel. Replaces window.confirm and hand-rolled confirm clusters:
// the decision happens where the click happened, in the app's own idiom.

export function ConfirmActionButton({
  confirmLabel,
  cancelLabel,
  onConfirm,
  variant = "ghost",
  size = "sm",
  "aria-label": ariaLabel,
  className,
  children,
}: {
  confirmLabel: string;
  cancelLabel: string;
  onConfirm: () => void;
  variant?: ButtonVariant;
  size?: ButtonSize;
  "aria-label"?: string;
  className?: string;
  children: React.ReactNode; // trigger content
}) {
  const [armed, setArmed] = useState(false);
  if (armed) {
    return (
      <span
        className="inline-flex items-center gap-1"
        onKeyDown={(event) => {
          if (event.key === "Escape") setArmed(false);
        }}
      >
        <Button
          variant="danger"
          size={size}
          autoFocus
          onClick={() => {
            setArmed(false);
            onConfirm();
          }}
        >
          {confirmLabel}
        </Button>
        <Button variant="ghost" size={size} onClick={() => setArmed(false)}>
          {cancelLabel}
        </Button>
      </span>
    );
  }
  return (
    <Button
      variant={variant}
      size={size}
      aria-label={ariaLabel}
      className={className}
      onClick={() => setArmed(true)}
    >
      {children}
    </Button>
  );
}

// ── Card ────────────────────────────────────────────────────────────────

export interface CardProps {
  padding?: "rows" | "card" | "none"; // default "card"
  tone?: "default" | "warning" | "danger"; // default "default"
  interactive?: boolean; // default false
  children: React.ReactNode;
}

const CARD_TONE_BORDER: Record<NonNullable<CardProps["tone"]>, string> = {
  default: "border-[var(--mem-border)]",
  warning: "border-[var(--mem-status-warning-border)]",
  danger: "border-[var(--mem-status-danger-border)]",
};

export function Card({
  padding = "card",
  tone = "default",
  interactive = false,
  children,
}: CardProps) {
  return (
    <div
      className={[
        "bg-[var(--mem-surface)] border rounded-[var(--mem-radius-lg)] overflow-hidden",
        CARD_TONE_BORDER[tone],
        padding === "card" ? "p-[var(--mem-space-5)]" : "",
        padding === "rows" ? "divide-y divide-[var(--mem-border)]" : "",
        interactive
          ? "cursor-pointer transition-[background-color,box-shadow] duration-[var(--mem-dur-fast)] hover:bg-[var(--mem-hover)] hover:shadow-[var(--mem-shadow-raised)]"
          : "",
      ]
        .filter(Boolean)
        .join(" ")}
    >
      {children}
    </div>
  );
}

// ── Field + Input ──────────────────────────────────────────────────────

export interface FieldProps {
  label: string;
  htmlFor: string;
  description?: string; // static helper, id `${htmlFor}-desc`
  error?: string; // id `${htmlFor}-error`, wins over description
  children: React.ReactElement<{ id?: string; "aria-describedby"?: string }>; // input/select; cloned with id + aria-describedby
}

export function Field({ label, htmlFor, description, error, children }: FieldProps) {
  const descId = `${htmlFor}-desc`;
  const errorId = `${htmlFor}-error`;
  const describedBy = error ? errorId : description ? descId : undefined;

  return (
    <div className="flex flex-col">
      <label
        htmlFor={htmlFor}
        style={{
          fontFamily: "var(--mem-font-mono)",
          fontSize: "var(--mem-text-2xs)",
          fontWeight: 500,
          lineHeight: 1.2,
          letterSpacing: "var(--mem-tracking-eyebrow)",
          color: "var(--mem-text-tertiary)",
          textTransform: "uppercase" as const,
          marginBottom: "6px",
        }}
      >
        {label}
      </label>
      {cloneElement(children, { id: htmlFor, "aria-describedby": describedBy })}
      {error ? (
        <p
          id={errorId}
          role="alert"
          style={{
            fontFamily: "var(--mem-font-body)",
            fontSize: "var(--mem-text-xs)",
            lineHeight: 1.4,
            color: "var(--mem-status-danger-text)",
            marginTop: "4px",
          }}
        >
          {error}
        </p>
      ) : description ? (
        <p
          id={descId}
          style={{
            fontFamily: "var(--mem-font-body)",
            fontSize: "var(--mem-text-xs)",
            lineHeight: 1.4,
            color: "var(--mem-text-secondary)",
            marginTop: "4px",
          }}
        >
          {description}
        </p>
      ) : null}
    </div>
  );
}

export interface InputProps extends React.InputHTMLAttributes<HTMLInputElement> {
  mono?: boolean; // endpoints, API keys, model ids
  invalid?: boolean; // aria-invalid + danger border
}

export function Input({ mono = false, invalid = false, className, style, ...rest }: InputProps) {
  return (
    <input
      {...rest}
      aria-invalid={invalid || undefined}
      className={[
        "h-[32px] px-[10px] py-[8px] rounded-[var(--mem-radius-md)] bg-[var(--mem-bg)] outline-none",
        "border",
        invalid ? "border-[var(--mem-status-danger-border)]" : "border-[var(--mem-border)]",
        "text-[var(--mem-text)] placeholder:text-[var(--mem-text-tertiary)]",
        "transition-[border-color] duration-[var(--mem-dur-fast)]",
        invalid ? "" : "focus-visible:border-[var(--mem-accent-indigo)]",
        "focus-visible:outline-2 focus-visible:outline-[var(--mem-focus-ring)] focus-visible:outline-offset-0",
        className ?? "",
      ]
        .filter(Boolean)
        .join(" ")}
      style={{
        fontFamily: mono ? "var(--mem-font-mono)" : "var(--mem-font-body)",
        fontSize: mono ? "var(--mem-text-sm)" : "var(--mem-text-base)",
        ...style,
      }}
    />
  );
}

// ── Select ─────────────────────────────────────────────────────────────
// Styled native <select> (appearance:none + custom chevron), not a
// hand-rolled listbox — native keeps the OS keyboard/VoiceOver contract.

export interface SelectProps
  extends Omit<React.SelectHTMLAttributes<HTMLSelectElement>, "size"> {
  size?: "sm" | "md"; // md (default): h-32px, radius-md — Input/Button parity
                      // sm: h-26px, radius-sm — Button sm parity
  mono?: boolean; // model ids, machine values (mirrors Input.mono)
  invalid?: boolean; // aria-invalid + danger border (mirrors Input.invalid)
}

const SELECT_SIZE_CLASS: Record<NonNullable<SelectProps["size"]>, string> = {
  md: "h-[32px] pl-[10px] pr-[28px] rounded-[var(--mem-radius-md)]",
  sm: "h-[26px] pl-[8px] pr-[24px] rounded-[var(--mem-radius-sm)]",
};

export function Select({
  size = "md",
  mono = false,
  invalid = false,
  className,
  style,
  ...rest
}: SelectProps) {
  return (
    <span className="relative inline-flex w-full">
      <select
        {...rest}
        aria-invalid={invalid || undefined}
        className={[
          // No vertical padding: the size classes below set a fixed height, and
          // py on top of it shrinks the content box under the line height —
          // 32px - 16px padding - 2px border = 14px for 13px text, which Chrome
          // renders by clipping the glyphs. Height alone centers the text.
          "appearance-none cursor-pointer w-full bg-[var(--mem-bg)] outline-none",
          "border",
          invalid ? "border-[var(--mem-status-danger-border)]" : "border-[var(--mem-border)]",
          "text-[var(--mem-text)]",
          "transition-[border-color] duration-[var(--mem-dur-fast)]",
          invalid ? "" : "focus-visible:border-[var(--mem-accent-indigo)]",
          "focus-visible:outline-2 focus-visible:outline-[var(--mem-focus-ring)] focus-visible:outline-offset-0",
          SELECT_SIZE_CLASS[size],
          className ?? "",
        ]
          .filter(Boolean)
          .join(" ")}
        style={{
          fontFamily: mono ? "var(--mem-font-mono)" : "var(--mem-font-body)",
          fontSize: mono ? "var(--mem-text-sm)" : size === "md" ? "var(--mem-text-base)" : "var(--mem-text-sm)",
          ...style,
        }}
      />
      <span
        aria-hidden="true"
        className="pointer-events-none absolute right-[10px] top-1/2 -translate-y-1/2 text-[var(--mem-text-tertiary)]"
      >
        <svg width="12" height="12" viewBox="0 0 12 12" fill="none">
          <path
            d="M3 4.5L6 7.5L9 4.5"
            stroke="currentColor"
            strokeWidth="1.5"
            strokeLinecap="round"
            strokeLinejoin="round"
          />
        </svg>
      </span>
    </span>
  );
}

// ── Tag — static label, closed tone union, no probe vocabulary ─────────
// The chip-never-lies invariant expressed in the type system: TagTone has
// no member that maps to the success/danger palette, so a Tag can never
// impersonate a probe result (see StatusChip below, the only primitive
// allowed to speak that vocabulary).

export type TagTone = "neutral" | "accent";

export interface TagProps {
  tone?: TagTone; // default "neutral"
  children: React.ReactNode;
}

const TAG_TONE_CLASS: Record<TagTone, string> = {
  neutral: "bg-transparent text-[var(--mem-text-secondary)] border-[var(--mem-border)]",
  accent: "bg-[var(--mem-indigo-bg)] text-[var(--mem-accent-indigo)] border-transparent",
};

export function Tag({ tone = "neutral", children }: TagProps) {
  return (
    <span
      className={[
        "inline-flex items-center rounded-full border py-[2px] px-[8px]",
        TAG_TONE_CLASS[tone],
      ].join(" ")}
      style={{
        fontFamily: "var(--mem-font-body)",
        fontSize: "var(--mem-text-xs)",
        fontWeight: 500,
        lineHeight: 1.2,
      }}
    >
      {children}
    </span>
  );
}

// ── StatusChip — the chip-never-lies API ──────────────────────────────

export type ProbeState =
  | { kind: "idle" } // never probed → muted, "not checked"
  | { kind: "probing" }
  | { kind: "up"; detail?: string } // e.g. "Ollama 0.6.2"
  | { kind: "down"; detail?: string } // e.g. "connection refused"
  | { kind: "stale" }; // input edited since last probe

export interface StatusChipProps {
  state: ProbeState; // the ONLY color input — no variant/intent/color prop exists
  label: string; // WHAT was probed (host, tool name), not what was selected
}

const STATUS_CHIP_TONE: Record<
  ProbeState["kind"],
  { dot: string; bg: string; text: string; border: string }
> = {
  up: {
    dot: "bg-[var(--mem-status-success-text)] rounded-full",
    bg: "bg-[var(--mem-status-success-bg)]",
    text: "text-[var(--mem-status-success-text)]",
    border: "border-[var(--mem-status-success-border)]",
  },
  down: {
    dot: "bg-[var(--mem-status-danger-text)] rounded-full",
    bg: "bg-[var(--mem-status-danger-bg)]",
    text: "text-[var(--mem-status-danger-text)]",
    border: "border-[var(--mem-status-danger-border)]",
  },
  probing: {
    dot: "bg-[var(--mem-text-tertiary)] rounded-full animate-pulse",
    bg: "bg-transparent",
    text: "text-[var(--mem-text-tertiary)]",
    border: "border-[var(--mem-border)]",
  },
  idle: {
    dot: "border border-[var(--mem-text-tertiary)] rounded-full",
    bg: "bg-transparent",
    text: "text-[var(--mem-text-tertiary)]",
    border: "border-[var(--mem-border)]",
  },
  stale: {
    dot: "border border-[var(--mem-text-tertiary)] rounded-full",
    bg: "bg-transparent",
    text: "text-[var(--mem-text-tertiary)]",
    border: "border-[var(--mem-border)]",
  },
};

export function StatusChip({ state, label }: StatusChipProps) {
  const { t } = useTranslation();
  const tone = STATUS_CHIP_TONE[state.kind];
  const detail =
    state.kind === "up" || state.kind === "down"
      ? state.detail
      : state.kind === "stale"
        ? t("status.notVerified")
        : undefined;

  return (
    <span
      aria-live="polite"
      className={[
        "inline-flex items-center gap-1.5 rounded-full border py-[2px] px-[8px] uppercase",
        tone.bg,
        tone.text,
        tone.border,
      ].join(" ")}
      style={{
        fontFamily: "var(--mem-font-mono)",
        fontSize: "var(--mem-text-2xs)",
        fontWeight: 500,
        lineHeight: 1.2,
        letterSpacing: "var(--mem-tracking-caps)",
      }}
    >
      <span aria-hidden="true" className={`w-1.5 h-1.5 shrink-0 ${tone.dot}`} />
      <span>
        {label}
        {detail ? ` · ${detail}` : ""}
      </span>
    </span>
  );
}
