// SPDX-License-Identifier: AGPL-3.0-only
import { useTranslation } from "react-i18next";
import SourcesList from "../../sources/SourcesSection";
import { ImportFlow } from "../../../ChatImport/ImportFlow";
import { Button, Card, SectionHeader } from "../primitives";

export default function SourcesSection({ onImport }: { onImport: () => void }) {
  const { t } = useTranslation();
  return (
    <>
      {/* ── Import Memories — no eyebrow: the row inside carries the same
          title, and a heading that repeats its only child's title is noise. */}
      <section className="mem-fade-up" style={{ animationDelay: "0ms" }}>
        <Card padding="rows">
          <div className="px-5 py-4">
            <div className="flex items-center justify-between">
              <div className="min-w-0">
                <div style={{ fontFamily: "var(--mem-font-body)", fontSize: "var(--mem-text-md)", fontWeight: 500, color: "var(--mem-text)" }}>{t("settings.sources.importMemoriesTitle")}</div>
                <p style={{ fontFamily: "var(--mem-font-body)", fontSize: "var(--mem-text-sm)", color: "var(--mem-text-secondary)", marginTop: "2px", lineHeight: "1.5" }}>
                  {t("settings.sources.importMemoriesDescription")}
                </p>
              </div>
              <Button variant="secondary" size="sm" className="shrink-0" onClick={onImport}>
                {t("settings.sources.import")}
              </Button>
            </div>
          </div>
        </Card>
      </section>

      {/* ── Import Chat History ──────────────────────────────── */}
      <section className="mem-fade-up" style={{ animationDelay: "30ms" }}>
        <SectionHeader label={t("settings.sources.importChatHistoryTitle")} />
        <Card padding="card">
          <p style={{ fontFamily: "var(--mem-font-body)", fontSize: "var(--mem-text-sm)", color: "var(--mem-text-secondary)", marginBottom: "12px", lineHeight: "1.5" }}>
            {t("settings.sources.importChatHistoryDescription")}
          </p>
          <ImportFlow />
        </Card>
      </section>

      {/* ── Sources ────────────────────────────────────────────── */}
      <section className="mem-fade-up" style={{ animationDelay: "60ms" }}>
        <SectionHeader label={t("settings.sources.connectedTitle")} />
        <SourcesList />
      </section>
    </>
  );
}
