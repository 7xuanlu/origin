// SPDX-License-Identifier: AGPL-3.0-only
import type { PageLinkOutbound } from "../../../lib/tauri";

interface RelatedPagesProps {
  outbound: PageLinkOutbound[];
  onPageClick?: (pageId: string) => void;
}

export default function RelatedPages({ outbound, onPageClick }: RelatedPagesProps) {
  // An empty "Related pages" header is noise, not information.
  if (outbound.length === 0) return null;

  return (
    <div aria-label="Related pages" className="memory-detail-rail-section">
      <h3 className="memory-detail-rail-title">
        Related Pages
      </h3>
      <div className="flex flex-wrap gap-1.5">
        {outbound.map((link, idx) => {
          const key = `${link.label}-${link.target_page_id ?? idx}`;
          const inner = (
            <span className="flex items-center gap-2">
              <svg
                width="14"
                height="14"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="1.5"
                style={{ color: "var(--mem-page-icon)" }}
                className="shrink-0"
              >
                <path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5" />
              </svg>
              <span
                style={{
                  fontFamily: "var(--mem-font-body)",
                  fontSize: "13px",
                  fontWeight: 500,
                  color: "var(--mem-text)",
                }}
              >
                {link.label}
              </span>
            </span>
          );
          const targetPageId = link.target_page_id;
          if (!targetPageId || !onPageClick) {
            return (
              <div
                key={key}
                className="flex items-center gap-2 rounded-lg px-3 py-2"
                style={{
                  backgroundColor: "var(--mem-surface)",
                  border: "1px solid var(--mem-border)",
                  opacity: 0.55,
                }}
                title="No page exists for this link yet"
              >
                {inner}
                <span
                  style={{
                    fontFamily: "var(--mem-font-mono)",
                    fontSize: "10px",
                    color: "var(--mem-text-tertiary)",
                  }}
                >
                  not distilled yet
                </span>
              </div>
            );
          }
          return (
            <button
              key={key}
              onClick={() => onPageClick(targetPageId)}
              className="rounded-lg px-3 py-2 text-left transition-colors duration-150 cursor-pointer hover:bg-[var(--mem-hover)]"
              style={{
                backgroundColor: "var(--mem-surface)",
                border: "1px solid var(--mem-border)",
              }}
            >
              {inner}
            </button>
          );
        })}
      </div>
    </div>
  );
}
