// SPDX-License-Identifier: AGPL-3.0-only
import { useEffect, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { open } from "@tauri-apps/plugin-dialog";
import { useTranslation } from "react-i18next";
import {
  getProfile,
  updateProfile,
  setAvatar,
  removeAvatar,
  setSetupCompleted,
  isRunAtLoginEnabled,
  setRunAtLogin,
} from "../../../../lib/tauri";
import { type Theme, useTheme } from "../../../../lib/theme";
import {
  readStoredLocalePreference,
  setLocalePreference,
  type StoredLocale,
} from "../../../../i18n";
import {
  Button,
  Card,
  ConfirmActionButton,
  Input,
  SectionHeader,
  SegmentedControl,
  Select,
  SettingRow,
} from "../primitives";
import ProfileAvatar from "../../ProfileAvatar";

type ThemeLabelKey =
  | "settings.theme.auto"
  | "settings.theme.light"
  | "settings.theme.dark";

const THEME_OPTIONS: { value: Theme; labelKey: ThemeLabelKey; icon: React.ReactNode }[] = [
  {
    value: "system",
    labelKey: "settings.theme.auto",
    icon: (
      <svg aria-hidden="true" className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17h14a2 2 0 002-2V5a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
      </svg>
    ),
  },
  {
    value: "light",
    labelKey: "settings.theme.light",
    icon: (
      <svg aria-hidden="true" className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M12 3v1m0 16v1m9-9h-1M4 12H3m15.364 6.364l-.707-.707M6.343 6.343l-.707-.707m12.728 0l-.707.707M6.343 17.657l-.707.707M16 12a4 4 0 11-8 0 4 4 0 018 0z" />
      </svg>
    ),
  },
  {
    value: "dark",
    labelKey: "settings.theme.dark",
    icon: (
      <svg aria-hidden="true" className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M20.354 15.354A9 9 0 018.646 3.646 9.003 9.003 0 0012 21a9.003 9.003 0 008.354-5.646z" />
      </svg>
    ),
  },
];

function formatProfileMonth(ts: number): string {
  return new Intl.DateTimeFormat(undefined, {
    month: "long",
    year: "numeric",
    timeZone: "UTC",
  }).format(ts * 1000);
}

interface ProfileUpdateFields {
  name?: string;
  displayName?: string;
  bio?: string;
}

function ProfileSettingsBlock() {
  const { t } = useTranslation();
  const queryClient = useQueryClient();
  const { data: profile } = useQuery({
    queryKey: ["profile"],
    queryFn: getProfile,
  });
  const [nameDraft, setNameDraft] = useState("");

  const displayName = profile?.display_name || profile?.name || "";

  useEffect(() => {
    setNameDraft(displayName);
  }, [displayName]);

  const profileMutation = useMutation({
    mutationFn: (fields: ProfileUpdateFields) => {
      if (!profile) return Promise.resolve();
      return updateProfile(profile.id, fields.name, fields.displayName, undefined, fields.bio);
    },
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["profile"] }),
  });

  const avatarMutation = useMutation({
    mutationFn: (path: string) => setAvatar(path),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["profile"] }),
  });

  const removeAvatarMutation = useMutation({
    mutationFn: removeAvatar,
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["profile"] }),
  });

  const saveName = () => {
    if (!profile) return;
    const next = nameDraft.trim();
    const current = displayName.trim();
    if (!next || next === current) {
      setNameDraft(displayName);
      return;
    }
    profileMutation.mutate({ name: next, displayName: next });
  };

  const handlePickAvatar = async () => {
    const selected = await open({
      title: t("settings.profile.choosePhoto"),
      filters: [{ name: t("settings.profile.images"), extensions: ["png", "jpg", "jpeg", "webp", "gif"] }],
    });
    if (typeof selected === "string") {
      avatarMutation.mutate(selected);
    }
  };

  if (!profile) return null;

  return (
    <section className="mem-fade-up" style={{ animationDelay: "0ms" }}>
      {/* No icon: in settings the sidebar owns iconography; eyebrows are type-only. */}
      <SectionHeader label={t("settings.profile.label")} />
      <Card padding="rows">
        <div className="px-5 py-4">
          <div className="flex items-start justify-between gap-4">
            <div className="min-w-0">
              <div style={{ fontFamily: "var(--mem-font-body)", fontSize: "var(--mem-text-md)", fontWeight: 500, color: "var(--mem-text)" }}>
                {t("settings.profile.photo")}
              </div>
            </div>
            <div className="mt-0.5 flex items-center gap-2">
              <ProfileAvatar
                avatarPath={profile.avatar_path}
                displayName={displayName}
                size={32}
                fontSize={13}
              />
              <Button variant="secondary" size="sm" onClick={handlePickAvatar}>
                {t("settings.profile.changePhoto")}
              </Button>
              {profile.avatar_path && (
                <Button variant="ghost" size="sm" onClick={() => removeAvatarMutation.mutate()}>
                  {t("settings.profile.removePhoto")}
                </Button>
              )}
            </div>
          </div>
        </div>
        <div className="px-5 py-4">
          <div className="flex items-start justify-between gap-4">
            <div className="min-w-0">
              <div style={{ fontFamily: "var(--mem-font-body)", fontSize: "var(--mem-text-md)", fontWeight: 500, color: "var(--mem-text)" }}>
                {t("settings.profile.displayName")}
              </div>
            </div>
            <div className="mt-0.5">
              <Input
                aria-label={t("settings.profile.displayName")}
                value={nameDraft}
                onChange={(event) => setNameDraft(event.target.value)}
                onBlur={saveName}
                onKeyDown={(event) => {
                  if (event.key === "Enter") event.currentTarget.blur();
                  if (event.key === "Escape") {
                    setNameDraft(displayName);
                    event.currentTarget.blur();
                  }
                }}
                style={{ width: "200px" }}
              />
            </div>
          </div>
        </div>
      </Card>
      <p
        className="px-1 pt-2"
        style={{
          fontFamily: "var(--mem-font-mono)",
          fontSize: "var(--mem-text-xs)",
          color: "var(--mem-text-tertiary)",
        }}
      >
        {t("settings.profile.joined", { date: formatProfileMonth(profile.created_at) })}
      </p>
    </section>
  );
}

export default function GeneralSection() {
  const { t } = useTranslation();
  const queryClient = useQueryClient();
  const [theme, setThemeValue] = useTheme();
  const [languagePreference, setLanguagePreference] = useState<StoredLocale>(
    () => readStoredLocalePreference(),
  );

  // ── Run at login ───────────────────────────────────────────────────
  const runAtLoginQuery = useQuery({
    queryKey: ["runAtLogin"],
    queryFn: isRunAtLoginEnabled,
  });
  const runAtLoginMutation = useMutation({
    mutationFn: setRunAtLogin,
    onSuccess: () => runAtLoginQuery.refetch(),
  });

  return (
    <>
      <ProfileSettingsBlock />
      <section className="mem-fade-up" style={{ animationDelay: "0ms" }}>
        <SectionHeader label={t("settings.general.appSection")} />
        <Card padding="rows">
          {/* Theme — folded into General; previously its own "Appearance" sidebar entry. */}
          <SettingRow
            title={t("settings.theme.label")}
            description={t("settings.theme.description")}
            control={
              <SegmentedControl
                aria-label={t("settings.theme.label")}
                options={THEME_OPTIONS.map((opt) => ({
                  value: opt.value,
                  label: t(opt.labelKey),
                  icon: opt.icon,
                }))}
                value={theme}
                onChange={setThemeValue}
              />
            }
          />
          <SettingRow
            title={t("settings.language.label")}
            description={t("settings.language.description")}
            control={
              <div className="w-fit shrink-0">
                <Select
                  size="sm"
                  aria-label={t("settings.language.label")}
                  value={languagePreference}
                  onChange={(event) => {
                    const nextPreference = event.currentTarget.value as StoredLocale;
                    setLanguagePreference(nextPreference);
                    void setLocalePreference(nextPreference);
                  }}
                >
                  <option value="system">{t("settings.language.system")}</option>
                  <option value="en">{t("settings.language.english")}</option>
                  <option value="zh-Hans">{t("settings.language.simplifiedChinese")}</option>
                  <option value="zh-Hant">{t("settings.language.traditionalChinese")}</option>
                </Select>
              </div>
            }
          />
          <SettingRow
            title={t("settings.general.runAtLoginTitle")}
            description={t("settings.general.runAtLoginDescription")}
            enabled={runAtLoginQuery.data ?? false}
            onToggle={() => runAtLoginMutation.mutate(!(runAtLoginQuery.data ?? false))}
          />
          {/* Re-run setup wizard — a proper row with an inline two-step
              confirm; data is preserved regardless. */}
          <SettingRow
            title={t("settings.general.rerunSetup")}
            description={t("settings.general.rerunSetupConfirm")}
            control={
              <ConfirmActionButton
                variant="secondary"
                size="sm"
                confirmLabel={t("settings.agents.confirm")}
                cancelLabel={t("settings.agents.cancel")}
                onConfirm={async () => {
                  await setSetupCompleted(false);
                  queryClient.invalidateQueries({ queryKey: ["shouldShowWizard"] });
                }}
              >
                {t("settings.general.rerunSetupGo")}
              </ConfirmActionButton>
            }
          />
        </Card>
      </section>
    </>
  );
}
