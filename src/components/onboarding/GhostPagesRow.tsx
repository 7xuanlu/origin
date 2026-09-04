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
        The three ghosts share whatever width the column has, rather than each
        claiming a fixed 280px.

        They used to be `shrink-0` cards 280px wide inside an `overflow-x: auto`
        strip with `scrollbar-width: none`. That needs 864px (3x280 + 2x12) and
        the column it sits in is the `minmax(0, 1fr)` half of
        HomePage's `.wiki-content-grid`, which is 508px at the app's own default
        window size (app/tauri.conf.json: 1280x720, sidebar expanded). So the
        second ghost was sliced down the middle at the column edge and the third
        was off-screen entirely — with the scrollbar hidden, nothing said the
        strip scrolled, so it just read as a broken card.

        A 3-track grid of `minmax(0, 1fr)` can never outgrow its column: the
        ghosts are ~161px each at the default size and widen back to ~268px on a
        large window, which is where the original 280px was tuned.
      */}
      <div
        className="pb-2"
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
