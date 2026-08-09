// SPDX-License-Identifier: AGPL-3.0-only
import type { ReactNode } from "react";

interface PinIconProps {
  readonly filled: boolean;
  readonly size?: number;
}

interface RailPanelTitleProps {
  readonly children: ReactNode;
}

interface MetadataRowProps {
  readonly align?: "center" | "start";
  readonly children: ReactNode;
  readonly label: string;
}

interface DisclosureButtonProps {
  readonly ariaLabel: string;
  readonly children: ReactNode;
  readonly count?: number;
  readonly onClick: () => void;
}

export function PinIcon({ filled, size = 16 }: PinIconProps) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill={filled ? "currentColor" : "none"}
      stroke="currentColor"
      strokeWidth="1.7"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      <path d="M14.5 4.5l5 5-3.25 1.25-3 6.75-2.75-2.75L6 19l4.25-4.5-2.75-2.75 6.75-3L14.5 4.5z" />
    </svg>
  );
}

export function RailPanelTitle({ children }: RailPanelTitleProps) {
  return <h3 className="memory-detail-rail-title">{children}</h3>;
}

export function MetadataRow({ align = "center", children, label }: MetadataRowProps) {
  return (
    <div className={`memory-detail-metadata-row ${align === "start" ? "is-start" : ""}`}>
      <span className="memory-detail-metadata-label">{label}</span>
      <div className="memory-detail-metadata-value">{children}</div>
    </div>
  );
}

export function DisclosureButton({ ariaLabel, children, count, onClick }: DisclosureButtonProps) {
  return (
    <button
      type="button"
      className="memory-detail-disclosure-button"
      onClick={onClick}
      aria-label={ariaLabel}
    >
      <span className="memory-detail-disclosure-label">{children}</span>
      {count !== undefined && (
        <span className="memory-detail-disclosure-count">{count}</span>
      )}
    </button>
  );
}
