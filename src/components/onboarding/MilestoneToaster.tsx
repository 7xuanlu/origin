// SPDX-License-Identifier: AGPL-3.0-only
import { useState, useMemo } from "react";
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

/** Small uppercase mono eyebrow — mirrors the FirstConceptModal pattern
 *  ("FIRST CONCEPT COMPILED") so onboarding surfaces share one voice. */
const EYEBROW: Record<MilestoneId, string> = {
  "intelligence-ready": "Intelligence ready",
  "first-memory": "First memory",
  "first-recall": "Recall",
  "first-concept": "",
  "graph-alive": "",
  "second-agent": "Second agent",
};

const TITLE: Record<MilestoneId, string> = {
  "intelligence-ready": "On-device intelligence is ready.",
  "first-memory": "First memory saved.",
  "first-recall": "Origin just recalled for you.",
  "first-concept": "", // celebrated via FirstConceptModal instead
  "graph-alive": "", // no toast
  "second-agent": "A second AI is writing too.",
};

const TWENTY_FOUR_HOURS_S = 24 * 60 * 60;

/** Accent color per milestone — keeps indigo for most events (cool, quiet
 *  "it worked") and warm amber for intelligence-ready to mark the moment
 *  Origin's mind comes online. */
function accentFor(id: MilestoneId): string {
  switch (id) {
    case "intelligence-ready":
      return "var(--mem-accent-warm)";
    default:
      return "var(--mem-accent-indigo)";
  }
}

/** Shapes a secondary line from the payload, or returns null when the
 *  milestone has no useful subtitle. Each branch treats missing/empty
 *  fields as "don't render" rather than inventing placeholder copy. */
function subtitleFor(record: MilestoneRecord): {
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
        text: "Classification and search now run locally.",
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
      return agent ? { kind: "plain", text: `Called by ${agent}.` } : null;
    }
    case "second-agent": {
      const agent = nonEmpty(p.agent);
      return agent
        ? {
            kind: "plain",
            text: `${agent} joined. Your memories follow you across tools.`,
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
      const key = `${m.id}@${m.first_triggered_at}`;
      if (dismissed.has(key)) return false;
      return true;
    });
  }, [milestones, dismissed]);

  const dismissKey = (m: MilestoneRecord) =>
    `${m.id}@${m.first_triggered_at}`;

  // No auto-dismiss timer: toast persists until user clicks (which acks
  // via API) or the 24h window in the filter above excludes it. A timer
  // would race with concurrent modal attention — user looks at modal,
  // misses the bottom-right toast, and local dismissal makes it
  // un-replayable until app restart.

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
          key={m.id}
          record={m}
          index={i}
          onClick={() => {
            acknowledge(m.id as MilestoneId);
            setDismissed((p) => new Set(p).add(dismissKey(m)));
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
}: {
  record: MilestoneRecord;
  index: number;
  onClick: () => void;
}) {
  const id = record.id as MilestoneId;
  const accent = accentFor(id);
  const eyebrow = EYEBROW[id];
  const title = TITLE[id];
  const sub = subtitleFor(record);

  return (
    <button
      onClick={onClick}
      className="text-left group"
      style={{
        fontFamily: "var(--mem-font-body)",
        color: "var(--mem-text)",
        backgroundColor: "var(--mem-surface)",
        border: "1px solid var(--mem-border)",
        borderRadius: 10,
        padding: "14px 18px",
        boxShadow: "var(--mem-shadow-toast)",
        animation: `mem-fade-up 400ms cubic-bezier(0.16, 1, 0.3, 1) ${index * 70}ms both`,
        maxWidth: 380,
        minWidth: 280,
        transition: "transform 180ms ease, border-color 180ms ease",
        cursor: "pointer",
      }}
      onMouseEnter={(e) => {
        e.currentTarget.style.transform = "translateY(-1px)";
      }}
      onMouseLeave={(e) => {
        e.currentTarget.style.transform = "translateY(0)";
      }}
    >
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
    </button>
  );
}
