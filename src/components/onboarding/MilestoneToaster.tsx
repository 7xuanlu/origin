// SPDX-License-Identifier: AGPL-3.0-only
import { useEffect, useMemo, useRef, useState } from "react";
import type { TFunction } from "i18next";
import { useTranslation } from "react-i18next";
import { useMilestones } from "./useMilestones";
import type { MilestoneId, MilestoneRecord } from "../../lib/tauri";

const TOAST_CHANNEL_IDS: Record<MilestoneId, boolean> = {
  "intelligence-ready": true,
  "first-memory": true,
  "first-recall": true,
  // first-concept is celebrated via FirstConceptModal; toast would be redundant.
  "first-concept": false,
  "graph-alive": false,
  "second-agent": true,
};

const TWENTY_FOUR_HOURS_S = 24 * 60 * 60;
const AUTO_DISMISS_MS = 8_000;

/** Accent color per milestone — keeps indigo for most events (cool, quiet
 *  "it worked") and warm amber for intelligence-ready to mark the moment
 *  Wenlan's mind comes online. */
function accentFor(id: MilestoneId): string {
  switch (id) {
    case "intelligence-ready":
      return "var(--mem-accent-warm)";
    default:
      return "var(--mem-accent-indigo)";
  }
}

function eyebrowFor(t: TFunction, id: MilestoneId): string {
  switch (id) {
    case "intelligence-ready":
      return t("onboarding.milestone.eyebrow.intelligenceReady");
    case "first-memory":
      return t("onboarding.milestone.eyebrow.firstMemory");
    case "first-recall":
      return t("onboarding.milestone.eyebrow.firstRecall");
    case "second-agent":
      return t("onboarding.milestone.eyebrow.secondAgent");
    case "first-concept":
    case "graph-alive":
      return "";
  }
}

function titleFor(t: TFunction, id: MilestoneId): string {
  switch (id) {
    case "intelligence-ready":
      return t("onboarding.milestone.title.intelligenceReady");
    case "first-memory":
      return t("onboarding.milestone.title.firstMemory");
    case "first-recall":
      return t("onboarding.milestone.title.firstRecall");
    case "second-agent":
      return t("onboarding.milestone.title.secondAgent");
    case "first-concept":
    case "graph-alive":
      return "";
  }
}

function toastKey(record: MilestoneRecord): string {
  return `${record.id}@${record.first_triggered_at}`;
}

/** Shapes a secondary line from the payload, or returns null when the
 *  milestone has no useful subtitle. Each branch treats missing/empty
 *  fields as "don't render" rather than inventing placeholder copy. */
function subtitleFor(t: TFunction, record: MilestoneRecord): {
  kind: "quote" | "plain";
  source?: string;
  text: string;
} | null {
  const p = (record.payload ?? {}) as Record<string, unknown>;
  const nonEmpty = (v: unknown): string | null =>
    typeof v === "string" && v.trim().length > 0 ? v.trim() : null;

  switch (record.id) {
    case "intelligence-ready":
      return {
        kind: "plain",
        text: t("onboarding.milestone.classificationLocal"),
      };
    case "first-memory": {
      const preview = nonEmpty(p.preview);
      if (!preview) return null;
      const source = nonEmpty(p.source);
      return {
        kind: "quote",
        source: source ?? undefined,
        text: preview,
      };
    }
    case "first-recall": {
      const agent = nonEmpty(p.agent);
      const preview = nonEmpty(p.preview);
      if (preview) {
        return {
          kind: "quote",
          source: agent ?? undefined,
          text: preview,
        };
      }
      return agent ? { kind: "plain", text: t("onboarding.milestone.calledBy", { agent }) } : null;
    }
    case "second-agent": {
      const agent = nonEmpty(p.agent);
      return agent
        ? {
            kind: "plain",
            text: t("onboarding.milestone.agentJoined", { agent }),
          }
        : null;
    }
    case "first-concept":
    case "graph-alive":
      return null;
  }
}

export function MilestoneToaster() {
  const { milestones, acknowledge } = useMilestones();
  const [dismissed, setDismissed] = useState<Set<string>>(new Set());

  const visible = useMemo(() => {
    const now = Math.floor(Date.now() / 1000);
    return milestones.filter((m) => {
      if (m.acknowledged_at != null) return false;
      if (!TOAST_CHANNEL_IDS[m.id as MilestoneId]) return false;
      if (now - m.first_triggered_at > TWENTY_FOUR_HOURS_S) return false;
      // Keyed by id + trigger time so a re-fire (new trigger_at) bypasses
      // any stale dismissal from a prior firing of the same id.
      const key = toastKey(m);
      if (dismissed.has(key)) return false;
      return true;
    });
  }, [milestones, dismissed]);

  return (
    <div
      style={{
        position: "fixed",
        bottom: 24,
        right: 24,
        display: "flex",
        flexDirection: "column",
        gap: 10,
        zIndex: 1000,
      }}
    >
      {visible.map((m, i) => (
        <Toast
          key={toastKey(m)}
          record={m}
          index={i}
          onClick={() => {
            acknowledge(m.id as MilestoneId);
            setDismissed((p) => new Set(p).add(toastKey(m)));
          }}
          onExpire={() => {
            setDismissed((p) => new Set(p).add(toastKey(m)));
          }}
        />
      ))}
    </div>
  );
}

function Toast({
  record,
  index,
  onClick,
  onExpire,
}: {
  record: MilestoneRecord;
  index: number;
  onClick: () => void;
  onExpire: () => void;
}) {
  const { t } = useTranslation();
  const id = record.id as MilestoneId;
  const accent = accentFor(id);
  const eyebrow = eyebrowFor(t, id);
  const title = titleFor(t, id);
  const sub = subtitleFor(t, record);
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const timerStartedAtRef = useRef<number | null>(null);
  const remainingMsRef = useRef(AUTO_DISMISS_MS);
  const hoveredRef = useRef(false);
  const focusedRef = useRef(false);
  const documentHiddenRef = useRef(document.hidden);
  const expiredRef = useRef(false);
  const onExpireRef = useRef(onExpire);
  onExpireRef.current = onExpire;

  const expire = () => {
    if (expiredRef.current) return;
    expiredRef.current = true;
    remainingMsRef.current = 0;
    timerRef.current = null;
    timerStartedAtRef.current = null;
    onExpireRef.current();
  };

  const pauseTimer = () => {
    if (timerRef.current == null) return;
    clearTimeout(timerRef.current);
    timerRef.current = null;
    if (timerStartedAtRef.current != null) {
      remainingMsRef.current = Math.max(
        0,
        remainingMsRef.current - (Date.now() - timerStartedAtRef.current),
      );
      timerStartedAtRef.current = null;
    }
  };

  const startTimer = () => {
    if (
      timerRef.current != null ||
      expiredRef.current ||
      hoveredRef.current ||
      focusedRef.current ||
      documentHiddenRef.current
    ) {
      return;
    }
    if (remainingMsRef.current <= 0) {
      expire();
      return;
    }

    timerStartedAtRef.current = Date.now();
    timerRef.current = setTimeout(() => {
      timerRef.current = null;
      if (timerStartedAtRef.current != null) {
        remainingMsRef.current = Math.max(
          0,
          remainingMsRef.current - (Date.now() - timerStartedAtRef.current),
        );
        timerStartedAtRef.current = null;
      }
      if (remainingMsRef.current <= 0) {
        expire();
      } else {
        // Keep the countdown accurate if the platform wakes the timer early.
        startTimer();
      }
    }, remainingMsRef.current);
  };

  useEffect(() => {
    const onVisibilityChange = () => {
      documentHiddenRef.current = document.hidden;
      if (document.hidden) {
        pauseTimer();
      } else {
        startTimer();
      }
    };

    document.addEventListener("visibilitychange", onVisibilityChange);
    startTimer();

    return () => {
      document.removeEventListener("visibilitychange", onVisibilityChange);
      pauseTimer();
    };
  }, []);

  return (
    <div
      data-testid="milestone-toast"
      onClick={onClick}
      className="text-left group"
      style={{
        position: "relative",
        fontFamily: "var(--mem-font-body)",
        color: "var(--mem-text)",
        backgroundColor: "var(--mem-surface)",
        border: "1px solid var(--mem-border)",
        borderRadius: 10,
        padding: "14px 44px 14px 18px",
        boxShadow: "var(--mem-shadow-toast)",
        animation: `mem-fade-up 400ms cubic-bezier(0.16, 1, 0.3, 1) ${index * 70}ms both`,
        maxWidth: 380,
        width: "min(380px, calc(100vw - 48px))",
        minWidth: 0,
        boxSizing: "border-box",
        transition: "transform 180ms ease, border-color 180ms ease",
        cursor: "pointer",
      }}
      onFocusCapture={() => {
        focusedRef.current = true;
        pauseTimer();
      }}
      onBlurCapture={(event) => {
        const relatedTarget = event.relatedTarget as Node | null;
        if (!relatedTarget || !event.currentTarget.contains(relatedTarget)) {
          focusedRef.current = false;
          startTimer();
        }
      }}
      onMouseEnter={(e) => {
        hoveredRef.current = true;
        pauseTimer();
        e.currentTarget.style.transform = "translateY(-1px)";
      }}
      onMouseLeave={(e) => {
        hoveredRef.current = false;
        startTimer();
        e.currentTarget.style.transform = "translateY(0)";
      }}
    >
      <button
        type="button"
        aria-label={t("common.close")}
        title={`${t("common.close")}: ${title}`}
        onClick={(event) => {
          event.stopPropagation();
          onClick();
        }}
        className="rounded-md p-1 transition-colors duration-150 hover:bg-[var(--mem-hover)]"
        style={{
          position: "absolute",
          top: 8,
          right: 8,
          color: "var(--mem-text-tertiary)",
          cursor: "pointer",
        }}
      >
        <svg
          aria-hidden="true"
          width="16"
          height="16"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="2"
        >
          <path d="M18 6 6 18M6 6l12 12" />
        </svg>
      </button>
      {eyebrow && (
        <div
          style={{
            fontFamily: "var(--mem-font-mono)",
            fontSize: "10px",
            letterSpacing: "0.14em",
            textTransform: "uppercase",
            color: accent,
            marginBottom: 6,
            opacity: 0.85,
          }}
        >
          {eyebrow}
        </div>
      )}
      <div
        style={{
          fontFamily: "var(--mem-font-heading)",
          fontSize: "15px",
          fontWeight: 500,
          lineHeight: 1.35,
          color: "var(--mem-text)",
          letterSpacing: "-0.005em",
        }}
      >
        {title}
      </div>
      {sub && sub.kind === "quote" && (
        <div
          style={{
            marginTop: 10,
            paddingLeft: 10,
            borderLeft: "1.5px solid var(--mem-border)",
            fontFamily: "var(--mem-font-heading)",
            fontStyle: "italic",
            fontSize: "13px",
            lineHeight: 1.5,
            color: "var(--mem-text-secondary)",
            display: "-webkit-box",
            WebkitLineClamp: 2,
            WebkitBoxOrient: "vertical",
            overflow: "hidden",
          }}
        >
          <span aria-hidden style={{ color: "var(--mem-text-tertiary)" }}>
            “
          </span>
          {sub.text}
          <span aria-hidden style={{ color: "var(--mem-text-tertiary)" }}>
            ”
          </span>
          {sub.source && (
            <span
              style={{
                display: "block",
                marginTop: 4,
                fontFamily: "var(--mem-font-mono)",
                fontStyle: "normal",
                fontSize: "10px",
                letterSpacing: "0.08em",
                textTransform: "uppercase",
                color: "var(--mem-text-tertiary)",
              }}
            >
              — {sub.source}
            </span>
          )}
        </div>
      )}
      {sub && sub.kind === "plain" && (
        <div
          style={{
            marginTop: 6,
            fontSize: "12.5px",
            lineHeight: 1.5,
            color: "var(--mem-text-secondary)",
            display: "-webkit-box",
            WebkitLineClamp: 2,
            WebkitBoxOrient: "vertical",
            overflow: "hidden",
          }}
        >
          {sub.text}
        </div>
      )}
    </div>
  );
}
