// SPDX-License-Identifier: AGPL-3.0-only
import { useTranslation } from "react-i18next";
import type { SettingsSection } from "./settings/SettingsSidebar";
import { SETTINGS_GROUPS } from "./settings/SettingsSidebar";
import GeneralSection from "./settings/sections/GeneralSection";
import SourcesSection from "./settings/sections/SourcesSection";
import AgentsSection from "./settings/sections/AgentsSection";
import IntelligenceSection from "./settings/sections/IntelligenceSection";
import DiagnosticsSection from "./settings/sections/DiagnosticsSection";

interface SettingsPageProps {
  /** Which group to display. Driven by the Settings sidebar in Main.tsx. */
  section?: SettingsSection;
  onBack: () => void;
  onSetupAgent?: () => void;
  /** Required: SourcesSection's only affordance is the import button, so a
   *  caller that omits this would render a card whose entire purpose is a
   *  button that does nothing. Keep it required — that is a compile error,
   *  which is the point. */
  onImport: () => void;
}

export default function SettingsPage({
  section = "general",
  onBack,
  onSetupAgent,
  onImport,
}: SettingsPageProps) {
  const { t } = useTranslation();

  const activeGroup = SETTINGS_GROUPS.find((g) => g.id === section);

  return (
    <div className="flex flex-col gap-6 max-w-2xl mx-auto py-4">
      {/* Back + Heading. The heading now names the active group; the sidebar
          handles navigating between groups. */}
      <div>
        <button onClick={onBack} className="p-1.5 -ml-1.5 rounded-md transition-colors duration-150 hover:bg-[var(--mem-hover)]" style={{ color: "var(--mem-text-tertiary)", background: "none", border: "none", cursor: "pointer", lineHeight: 0, marginBottom: "12px" }}>
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5"><path d="M19 12H5M12 19l-7-7 7-7" /></svg>
        </button>
        <h1 style={{ fontFamily: "var(--mem-font-heading)", fontSize: "var(--mem-text-2xl)", fontWeight: 500, lineHeight: 1.2, color: "var(--mem-text)" }}>
          {activeGroup ? t(activeGroup.labelKey) : t("settings.title")}
        </h1>
        <p style={{ fontFamily: "var(--mem-font-body)", fontSize: "var(--mem-text-sm)", color: "var(--mem-text-secondary)", marginTop: "6px" }}>
          {activeGroup ? t(activeGroup.hintKey) : t("settings.manageHint")}
        </p>
      </div>

      {section === "general" && <GeneralSection />}
      {section === "sources" && <SourcesSection onImport={onImport} />}
      {section === "agents" && <AgentsSection onSetupAgent={onSetupAgent} />}
      {section === "intelligence" && <IntelligenceSection delay={0} />}
      {section === "diagnostics" && <DiagnosticsSection />}

      {/* ── Privacy note — persistent footer on all groups. This is the only
          place this claim appears in the UI; keep it truthful and singular. */}
      <div
        className="flex items-start gap-2.5 px-2 pt-2 mem-fade-up"
        style={{ animationDelay: "120ms" }}
      >
        <svg aria-hidden="true" className="w-3.5 h-3.5 shrink-0 mt-0.5" style={{ color: "var(--mem-text-tertiary)" }} fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z" />
        </svg>
        <p style={{ fontFamily: "var(--mem-font-body)", fontSize: "var(--mem-text-xs)", color: "var(--mem-text-tertiary)", lineHeight: "1.5" }}>
          {t("settings.footer")}
        </p>
      </div>

    </div>
  );
}
