// SPDX-License-Identifier: AGPL-3.0-only
import type { TFunction } from "i18next";
import { useTranslation } from "react-i18next";

export type HomePageState = "seed" | "listening" | "gathering" | "alive";

interface Props {
  state: HomePageState;
  memoryCount: number;
  daysInListening: number;
}

function copyFor(
  t: TFunction,
  state: HomePageState,
  memoryCount: number,
  daysInListening: number,
): string | null {
  if (state === "alive") return null;
  if (state === "seed") {
    return t("onboarding.next.seed");
  }
  if (state === "listening") {
    if (daysInListening > 3) {
      return t("onboarding.next.listeningQuiet");
    }
    return t("onboarding.next.listening");
  }
  // gathering
  if (memoryCount < 5) {
    return t("onboarding.next.gatheringFew", { count: memoryCount });
  }
  return t("onboarding.next.gatheringMany", { count: memoryCount });
}

export function WhatHappensNextCard({ state, memoryCount, daysInListening }: Props) {
  const { t } = useTranslation();
  const text = copyFor(t, state, memoryCount, daysInListening);
  if (!text) return null;
  return (
    <section
      data-testid="what-happens-next"
      className="rounded-xl px-5 py-4"
      style={{
        backgroundColor: "var(--mem-surface)",
        border: "1px solid var(--mem-border)",
        animation: "mem-fade-up 400ms cubic-bezier(0.16, 1, 0.3, 1) 50ms both",
      }}
    >
      <p
        style={{
          fontFamily: "var(--mem-font-body)",
          fontSize: "14px",
          color: "var(--mem-text-secondary)",
          lineHeight: "1.6",
          margin: 0,
        }}
      >
        {text}
      </p>
    </section>
  );
}
