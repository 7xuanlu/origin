// SPDX-License-Identifier: AGPL-3.0-only
import { useEffect, useRef, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { useTranslation } from "react-i18next";
import { listEntities, getProfile } from "../../lib/tauri";
import ProfileAvatar from "./ProfileAvatar";

interface IdentityCardProps {
  onOpenDetail: (entityId: string) => void;
  onOpenSettings?: () => void;
  onOpenAbout?: () => void;
}

export default function IdentityCard({
  onOpenDetail,
  onOpenSettings,
  onOpenAbout,
}: IdentityCardProps) {
  const { t } = useTranslation();
  const [menuOpen, setMenuOpen] = useState(false);
  const rootRef = useRef<HTMLDivElement>(null);

  const { data: profile } = useQuery({
    queryKey: ["profile"],
    queryFn: getProfile,
  });

  const { data: entities = [] } = useQuery({
    queryKey: ["entities", "person"],
    queryFn: () => listEntities("person"),
    refetchInterval: 10000,
  });

  const selfEntity = entities[0];
  // Profile name is canonical — entity name may be partial (e.g. "Lu" vs "Qi-Xuan Lu")
  const displayName = profile?.display_name || profile?.name || selfEntity?.name || "";
  const labelText = displayName || t("identityCard.account");
  const triggerLabel = displayName
    ? t("identityCard.namedAccountMenu", { name: displayName })
    : t("identityCard.accountMenu");

  useEffect(() => {
    if (!menuOpen) return;

    const closeOnOutsideClick = (event: MouseEvent) => {
      if (rootRef.current?.contains(event.target as Node)) return;
      setMenuOpen(false);
    };

    const closeOnEscape = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        setMenuOpen(false);
      }
    };

    document.addEventListener("mousedown", closeOnOutsideClick);
    document.addEventListener("keydown", closeOnEscape);
    return () => {
      document.removeEventListener("mousedown", closeOnOutsideClick);
      document.removeEventListener("keydown", closeOnEscape);
    };
  }, [menuOpen]);

  const openSettings = () => {
    setMenuOpen(false);
    if (onOpenSettings) {
      onOpenSettings();
      return;
    }
    onOpenDetail("__create_profile__");
  };

  const openAbout = () => {
    setMenuOpen(false);
    onOpenAbout?.();
  };

  return (
    <div className="relative w-full" ref={rootRef}>
      <button
        type="button"
        aria-label={triggerLabel}
        aria-haspopup="menu"
        aria-expanded={menuOpen}
        onClick={() => setMenuOpen((open) => !open)}
        className="flex w-full items-center gap-3 rounded-xl px-3 py-2.5 text-left transition-all duration-200 hover:shadow-sm focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[var(--mem-accent-sage)]"
        style={{
          backgroundColor: "var(--mem-account-card)",
          border: "1px solid var(--mem-account-card-border)",
        }}
      >
        {displayName ? (
          <span
            className="shrink-0 rounded-full p-px"
            style={{ background: "linear-gradient(135deg, var(--mem-accent-page), var(--mem-accent-warm))" }}
          >
            <ProfileAvatar
              avatarPath={profile?.avatar_path}
              displayName={displayName}
              size={34}
              fontSize={13}
            />
          </span>
        ) : (
          <div
            className="flex h-[34px] w-[34px] shrink-0 items-center justify-center rounded-full"
            style={{
              backgroundColor: "var(--mem-border)",
              color: "var(--mem-text-tertiary)",
            }}
          >
            <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
              <path d="M20 21a8 8 0 10-16 0" />
              <circle cx="12" cy="7" r="4" />
            </svg>
          </div>
        )}
        <span className="min-w-0 flex-1">
          <span
            className="block truncate"
            style={{
              fontFamily: "var(--mem-font-body)",
              fontSize: "13px",
              fontWeight: 500,
              color: "var(--mem-text)",
            }}
          >
            {labelText}
          </span>
        </span>
        <svg
          width="14"
          height="14"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="1.7"
          strokeLinecap="round"
          strokeLinejoin="round"
          aria-hidden
          className="shrink-0 transition-transform duration-150"
          style={{
            color: "var(--mem-text-tertiary)",
            transform: menuOpen ? "rotate(180deg)" : "none",
          }}
        >
          <path d="m6 9 6 6 6-6" />
        </svg>
      </button>

      {menuOpen && (
        <div
          role="menu"
          className="absolute bottom-full right-0 z-20 mb-2 w-44 rounded-lg p-1 shadow-lg"
          style={{
            backgroundColor: "var(--mem-popover)",
            border: "1px solid var(--mem-popover-border)",
            boxShadow: "var(--mem-shadow-toast)",
          }}
        >
          <button
            type="button"
            role="menuitem"
            onClick={openSettings}
            className="flex w-full items-center gap-2 rounded-md px-3 py-2 text-left transition-colors duration-150 hover:bg-[var(--mem-hover)]"
            style={{
              fontFamily: "var(--mem-font-body)",
              fontSize: "13px",
              color: "var(--mem-text)",
            }}
          >
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round" style={{ color: "var(--mem-text-tertiary)" }}>
              <circle cx="12" cy="12" r="3" />
              <path d="M19.4 15a1.65 1.65 0 00.33 1.82l.06.06a2 2 0 010 2.83 2 2 0 01-2.83 0l-.06-.06a1.65 1.65 0 00-1.82-.33 1.65 1.65 0 00-1 1.51V21a2 2 0 01-4 0v-.09A1.65 1.65 0 009 19.4a1.65 1.65 0 00-1.82.33l-.06.06A2 2 0 014.35 17l.06-.06A1.65 1.65 0 004.74 15a1.65 1.65 0 00-1.51-1H3a2 2 0 010-4h.23A1.65 1.65 0 004.74 9a1.65 1.65 0 00-.33-1.94l-.06-.06A2 2 0 017.17 4.17l.06.06A1.65 1.65 0 009 4.56a1.65 1.65 0 001-1.51V3a2 2 0 014 0v.05a1.65 1.65 0 001 1.51 1.65 1.65 0 001.77-.33l.06-.06A2 2 0 0119.65 7l-.06.06A1.65 1.65 0 0019.26 9c.2.61.75 1 1.4 1H21a2 2 0 010 4h-.34a1.65 1.65 0 00-1.26 1z" />
            </svg>
            {t("identityCard.settings")}
          </button>
          <button
            type="button"
            role="menuitem"
            onClick={openAbout}
            className="flex w-full items-center gap-2 rounded-md px-3 py-2 text-left transition-colors duration-150 hover:bg-[var(--mem-hover)]"
            style={{
              fontFamily: "var(--mem-font-body)",
              fontSize: "13px",
              color: "var(--mem-text)",
            }}
          >
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round" style={{ color: "var(--mem-text-tertiary)" }}>
              <circle cx="12" cy="12" r="9" />
              <path d="M12 11v5" />
              <path d="M12 8h.01" />
            </svg>
            {t("identityCard.aboutWenlan")}
          </button>
        </div>
      )}
    </div>
  );
}
