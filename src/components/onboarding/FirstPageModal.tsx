// SPDX-License-Identifier: AGPL-3.0-only
import { useEffect, useRef } from "react";
import { useTranslation } from "react-i18next";

interface Page {
  id: string;
  title: string;
  summary?: string | null;
  source_memory_ids: string[];
}

interface Props {
  page: Page;
  onOpen: (pageId: string) => void;
  onDismiss: () => void;
}

/** CSS selector matching all focusable elements we care about cycling through. */
const FOCUSABLE_SELECTOR =
  'a[href], button:not([disabled]), textarea:not([disabled]), input:not([disabled]), select:not([disabled]), [tabindex]:not([tabindex="-1"])';

export function FirstPageModal({ page, onOpen, onDismiss }: Props) {
  const { t } = useTranslation();
  const dialogRef = useRef<HTMLDivElement | null>(null);

  // Focus trap: on mount focus the first focusable element inside the dialog,
  // and on Tab/Shift+Tab at the boundaries cycle to the other end so focus
  // never escapes the modal.
  useEffect(() => {
    const dialog = dialogRef.current;
    if (!dialog) return;
    const focusables = dialog.querySelectorAll<HTMLElement>(FOCUSABLE_SELECTOR);
    const first = focusables[0];
    const last = focusables[focusables.length - 1];

    // Move focus into the dialog on mount so keyboard users land inside.
    if (first) first.focus();

    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        onDismiss();
        return;
      }
      if (e.key !== "Tab" || !first || !last) return;
      // Cycle at the boundaries.
      if (e.shiftKey) {
        if (document.activeElement === first) {
          e.preventDefault();
          last.focus();
        }
      } else {
        if (document.activeElement === last) {
          e.preventDefault();
          first.focus();
        }
      }
    };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [onDismiss]);

  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-labelledby="first-page-title"
      ref={dialogRef}
      style={{
        position: "fixed",
        inset: 0,
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        backgroundColor: "rgba(0,0,0,0.4)",
        zIndex: 1100,
        animation: "fade-in 400ms ease both",
      }}
      onClick={(e) => {
        if (e.target === e.currentTarget) onDismiss();
      }}
    >
      <div
        style={{
          maxWidth: 480,
          width: "90%",
          padding: "28px 32px",
          borderRadius: 16,
          backgroundColor: "var(--mem-bg)",
          border: "1px solid var(--mem-accent-warm)",
          boxShadow: "0 24px 48px rgba(0,0,0,0.35)",
        }}
      >
        <p
          style={{
            fontFamily: "var(--mem-font-body)",
            fontSize: "12px",
            color: "var(--mem-accent-warm)",
            textTransform: "uppercase",
            letterSpacing: "0.08em",
            margin: "0 0 8px 0",
          }}
        >
          {t("onboarding.firstPage.eyebrow")}
        </p>
        <h2
          id="first-page-title"
          style={{
            fontFamily: "var(--mem-font-heading)",
            fontSize: "22px",
            fontWeight: 500,
            margin: "0 0 12px 0",
            color: "var(--mem-text)",
          }}
        >
          {page.title}
        </h2>
        {page.summary && (
          <p
            style={{
              fontFamily: "var(--mem-font-body)",
              fontSize: "14px",
              color: "var(--mem-text-secondary)",
              lineHeight: 1.6,
              margin: "0 0 16px 0",
            }}
          >
            {page.summary}
          </p>
        )}
        <p
          style={{
            fontFamily: "var(--mem-font-mono)",
            fontSize: "11px",
            color: "var(--mem-text-tertiary)",
            margin: "0 0 24px 0",
          }}
        >
          {t("onboarding.firstPage.compiledFrom", {
            count: page.source_memory_ids.length,
          })}
        </p>
        <div className="flex gap-3 justify-end">
          <button
            onClick={onDismiss}
            style={{
              fontFamily: "var(--mem-font-body)",
              fontSize: "13px",
              color: "var(--mem-text-secondary)",
              background: "none",
              border: "none",
              padding: "8px 12px",
              cursor: "pointer",
            }}
          >
            {t("onboarding.firstPage.dismiss")}
          </button>
          <button
            onClick={() => onOpen(page.id)}
            style={{
              fontFamily: "var(--mem-font-body)",
              fontSize: "13px",
              color: "white",
              backgroundColor: "var(--mem-accent-warm)",
              border: "none",
              padding: "8px 16px",
              borderRadius: 8,
              cursor: "pointer",
            }}
          >
            {t("onboarding.firstPage.openPage")}
          </button>
        </div>
      </div>
    </div>
  );
}
