// SPDX-License-Identifier: AGPL-3.0-only
import { useTranslation } from "react-i18next";

interface AboutWenlanDialogProps {
  open: boolean;
  onClose: () => void;
}

export default function AboutWenlanDialog({ open, onClose }: AboutWenlanDialogProps) {
  const { t } = useTranslation();

  if (!open) return null;

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/20 px-4"
      onClick={(event) => {
        if (event.target === event.currentTarget) {
          onClose();
        }
      }}
    >
      <div
        role="dialog"
        aria-modal="true"
        aria-labelledby="about-wenlan-title"
        className="w-full max-w-sm rounded-xl p-5 shadow-xl"
        style={{
          backgroundColor: "var(--mem-surface)",
          border: "1px solid var(--mem-border)",
        }}
      >
        <div className="flex items-start justify-between gap-4">
          <div>
            <h2
              id="about-wenlan-title"
              style={{
                fontFamily: "var(--mem-font-heading)",
                fontSize: "20px",
                fontWeight: 500,
                color: "var(--mem-text)",
              }}
            >
              {t("aboutWenlan.title")}
            </h2>
            <p
              style={{
                fontFamily: "var(--mem-font-body)",
                fontSize: "13px",
                color: "var(--mem-text-secondary)",
                lineHeight: 1.6,
                marginTop: 8,
              }}
            >
              {t("aboutWenlan.description")}
            </p>
          </div>
          <button
            type="button"
            aria-label={t("common.close")}
            onClick={onClose}
            className="rounded-md p-1 transition-colors duration-150 hover:bg-[var(--mem-hover)]"
            style={{
              color: "var(--mem-text-tertiary)",
            }}
          >
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M18 6 6 18M6 6l12 12" />
            </svg>
          </button>
        </div>
      </div>
    </div>
  );
}
