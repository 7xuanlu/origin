// SPDX-License-Identifier: AGPL-3.0-only
import { useQuery } from "@tanstack/react-query";
import { getProfile } from "../../lib/tauri";

function timeOfDay(): "morning" | "afternoon" | "evening" {
  const h = new Date().getHours();
  if (h < 12) return "morning";
  if (h < 18) return "afternoon";
  return "evening";
}

function formatDate(): string {
  return new Date().toLocaleDateString(undefined, {
    weekday: "long",
    month: "long",
    day: "numeric",
  });
}

interface Props {
  memoryCount?: number;
  conceptCount?: number;
}

export function Greeting({ memoryCount, conceptCount }: Props) {
  const { data: profile } = useQuery({
    queryKey: ["profile"],
    queryFn: getProfile,
    staleTime: 5 * 60_000,
  });

  const fullName = profile?.display_name || profile?.name || "";
  const firstName = fullName ? fullName.trim().split(/\s+/)[0] : "";
  const period = timeOfDay();
  const date = formatDate();

  const stats: string[] = [];
  if (conceptCount && conceptCount > 0) {
    stats.push(`${conceptCount} ${conceptCount === 1 ? "concept" : "concepts"}`);
  }
  if (memoryCount && memoryCount > 0) {
    stats.push(`${memoryCount} ${memoryCount === 1 ? "memory" : "memories"}`);
  }

  return (
    <header data-testid="greeting" className="pt-2">
      <h1
        style={{
          fontFamily: "var(--mem-font-heading)",
          fontSize: 30,
          fontWeight: 400,
          color: "var(--mem-text)",
          letterSpacing: "-0.015em",
          lineHeight: 1.1,
        }}
      >
        Good {period}{firstName ? `, ${firstName}` : ""}.
      </h1>
      <p
        className="mt-2"
        style={{
          fontFamily: "var(--mem-font-body)",
          fontSize: 13,
          color: "var(--mem-text-secondary)",
          letterSpacing: "0.005em",
        }}
      >
        <span style={{ fontStyle: "italic" }}>{date}</span>
        {stats.length > 0 && (
          <>
            <span
              className="mx-2"
              style={{ color: "var(--mem-text-tertiary)" }}
              aria-hidden="true"
            >
              ·
            </span>
            <span>your library holds {stats.join(" and ")}</span>
          </>
        )}
      </p>
    </header>
  );
}
