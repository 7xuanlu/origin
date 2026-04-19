// SPDX-License-Identifier: AGPL-3.0-only
import { useQuery } from "@tanstack/react-query";
import { convertFileSrc } from "@tauri-apps/api/core";
import { listEntities, getEntityDetail, getProfile } from "../../lib/tauri";

interface IdentityCardProps {
  onOpenDetail: (entityId: string) => void;
}

export default function IdentityCard({ onOpenDetail }: IdentityCardProps) {

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

  const { data: detail } = useQuery({
    queryKey: ["entityDetail", selfEntity?.id],
    queryFn: () => getEntityDetail(selfEntity!.id),
    enabled: !!selfEntity,
    refetchInterval: 10000,
  });

  const observations = detail?.observations ?? [];

  const roleObs = observations.find(
    (o) =>
      o.content.toLowerCase().includes("engineer") ||
      o.content.toLowerCase().includes("developer") ||
      o.content.toLowerCase().includes("designer") ||
      o.content.toLowerCase().includes("manager") ||
      o.content.toLowerCase().includes("founder"),
  );

  if (!selfEntity) {
    return (
      <button
        onClick={() => onOpenDetail("__create_profile__")}
        className="w-full rounded-xl p-5 text-center text-left transition-all duration-200 hover:shadow-md"
        style={{
          border: "2px dashed var(--mem-border)",
        }}
      >
        <div
          className="w-12 h-12 rounded-full mx-auto mb-3 flex items-center justify-center"
          style={{ backgroundColor: "var(--mem-border)" }}
        >
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" style={{ color: "var(--mem-text-tertiary)" }}>
            <path d="M12 5v14M5 12h14" />
          </svg>
        </div>
        <p
          style={{
            fontFamily: "var(--mem-font-body)",
            fontSize: "12px",
            color: "var(--mem-text-tertiary)",
            lineHeight: "1.5",
          }}
        >
          Set up your profile
        </p>
      </button>
    );
  }

  const initials = displayName
    .split(" ")
    .map((w) => w[0])
    .join("")
    .slice(0, 2)
    .toUpperCase();
  const avatarUrl = profile?.avatar_path ? convertFileSrc(profile.avatar_path) : null;

  return (
    <button
      onClick={() => onOpenDetail("__create_profile__")}
      className="w-full rounded-xl p-5 text-left transition-all duration-200 hover:shadow-md"
      style={{
        backgroundColor: "var(--mem-surface)",
        border: "1px solid var(--mem-border)",
      }}
    >
      {avatarUrl ? (
        <img
          src={avatarUrl}
          alt={displayName}
          className="mx-auto mb-3"
          style={{ width: 56, height: 56, borderRadius: "50%", objectFit: "cover" }}
        />
      ) : (
        <div
          className="w-14 h-14 rounded-full mx-auto mb-3 flex items-center justify-center"
          style={{
            background: "linear-gradient(135deg, var(--mem-accent-warm), var(--mem-accent-amber))",
            fontFamily: "var(--mem-font-heading)",
            fontSize: "20px",
            color: "white",
            fontWeight: 500,
          }}
        >
          {initials}
        </div>
      )}

      <p
        className="text-center font-medium"
        style={{
          fontFamily: "var(--mem-font-heading)",
          fontSize: "18px",
          color: "var(--mem-text)",
        }}
      >
        {displayName}
      </p>

      {roleObs && (
        <p
          className="text-center mt-0.5"
          style={{
            fontFamily: "var(--mem-font-body)",
            fontSize: "12px",
            color: "var(--mem-text-secondary)",
          }}
        >
          {roleObs.content.length > 40
            ? roleObs.content.slice(0, 40) + "..."
            : roleObs.content}
        </p>
      )}

    </button>
  );
}
