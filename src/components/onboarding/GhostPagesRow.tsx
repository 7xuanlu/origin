// SPDX-License-Identifier: AGPL-3.0-only
import { useTranslation } from "react-i18next";

export function GhostPagesRow() {
  const { t } = useTranslation();
  return (
    <div>
      <p
        style={{
          fontFamily: "var(--mem-font-body)",
          fontSize: "13px",
          color: "var(--mem-text-tertiary)",
          margin: "0 0 10px 0",
        }}
      >
        {t("onboarding.ghostPages")}
      </p>
      {/*
        All three ghosts must fit the column they sit in — the `minmax(0, 1fr)`
        half of HomePage's `.wiki-content-grid`, 508px at the app's default
        1280x720 window. The tracks below are what guarantee that; fixed card
        widths overflowed it and sliced the second card in half.
      */}
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(3, minmax(0, 1fr))",
          gap: "12px",
        }}
      >
        {[0, 1, 2].map((i) => (
          <div
            key={i}
            data-ghost-card
            className="rounded-xl"
            style={{
              height: "110px",
              border: "1px solid var(--mem-border)",
              opacity: 0.4,
              backgroundColor: "transparent",
            }}
          />
        ))}
      </div>
    </div>
  );
}
