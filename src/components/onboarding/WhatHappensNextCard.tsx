// SPDX-License-Identifier: AGPL-3.0-only
export type HomePageState = "seed" | "listening" | "gathering" | "alive";

interface Props {
  state: HomePageState;
  memoryCount: number;
  daysInListening: number;
}

function copyFor(state: HomePageState, memoryCount: number, daysInListening: number): string | null {
  if (state === "alive") return null;
  if (state === "seed") {
    return "Origin is loading its on-device intelligence. Once it's ready, memories from your agents will start arriving here and compile into concepts.";
  }
  if (state === "listening") {
    if (daysInListening > 3) {
      return "Still quiet here. You can connect more AI tools from Settings.";
    }
    return "Keep using your AI tools. Memories will appear here as agents save what they learn. Concepts usually start compiling within a day.";
  }
  // gathering
  if (memoryCount < 5) {
    const plural = memoryCount === 1 ? "memory" : "memories";
    return `${memoryCount} ${plural} saved. Concepts are compiled automatically — usually within a day of regular use.`;
  }
  return `${memoryCount} memories saved. Concepts are compiled when patterns emerge — you should see the first ones soon.`;
}

export function WhatHappensNextCard({ state, memoryCount, daysInListening }: Props) {
  const text = copyFor(state, memoryCount, daysInListening);
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
