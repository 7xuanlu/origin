// SPDX-License-Identifier: AGPL-3.0-only
import { useState, useRef, useEffect } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { convertFileSrc } from "@tauri-apps/api/core";
import { open } from "@tauri-apps/plugin-dialog";
import {
  getProfile,
  updateProfile,
  setAvatar,
  removeAvatar,
  getProfileNarrative,
  regenerateNarrative,
  listMemoriesRich,
} from "../../lib/tauri";

interface ProfilePageProps {
  onBack: () => void;
  onSelectMemory?: (sourceId: string) => void;
}

function formatDate(ts: number): string {
  return new Date(ts * 1000).toLocaleDateString(undefined, {
    year: "numeric",
    month: "long",
  });
}

function relativeDate(ts: number): string {
  const delta = Date.now() - ts * 1000;
  const days = Math.floor(delta / 86_400_000);
  if (days <= 0) return "today";
  if (days === 1) return "yesterday";
  if (days < 7) return `${days}d ago`;
  const weeks = Math.floor(days / 7);
  return weeks === 1 ? "1w ago" : `${weeks}w ago`;
}

// ── Section header (matches homepage Fraunces + italic sub-line) ─────

const SECTION_TITLE_STYLE: React.CSSProperties = {
  fontFamily: "var(--mem-font-heading)",
  fontSize: 19,
  fontWeight: 400,
  color: "var(--mem-text)",
  letterSpacing: "-0.005em",
  lineHeight: 1.2,
};

const SECTION_SUB_STYLE: React.CSSProperties = {
  fontFamily: "var(--mem-font-body)",
  fontSize: 12,
  fontStyle: "italic",
  color: "var(--mem-text-tertiary)",
  marginTop: 2,
};

// ── Editable field ──────────────────────────────────────────────────────

function EditableField({
  value,
  placeholder,
  onSave,
  fontSize = "14px",
  fontFamily = "var(--mem-font-body)",
  color = "var(--mem-text)",
  multiline = false,
}: {
  value: string;
  placeholder: string;
  onSave: (val: string) => void;
  fontSize?: string;
  fontFamily?: string;
  color?: string;
  multiline?: boolean;
}) {
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState(value);
  const ref = useRef<HTMLInputElement | HTMLTextAreaElement>(null);

  useEffect(() => { setDraft(value); }, [value]);
  useEffect(() => { if (editing) ref.current?.focus(); }, [editing]);

  const save = () => {
    setEditing(false);
    if (draft.trim() !== value) onSave(draft.trim());
  };

  const style: React.CSSProperties = {
    fontFamily,
    fontSize,
    color: draft ? color : "var(--mem-text-tertiary)",
    background: "transparent",
    border: "none",
    outline: "none",
    width: "100%",
    padding: 0,
    resize: "none" as const,
  };

  if (editing) {
    if (multiline) {
      return (
        <textarea
          ref={ref as React.RefObject<HTMLTextAreaElement>}
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          onBlur={save}
          onKeyDown={(e) => { if (e.key === "Escape") { setDraft(value); setEditing(false); } }}
          rows={2}
          style={{ ...style, lineHeight: "1.5" }}
        />
      );
    }
    return (
      <input
        ref={ref as React.RefObject<HTMLInputElement>}
        value={draft}
        onChange={(e) => setDraft(e.target.value)}
        onBlur={save}
        onKeyDown={(e) => {
          if (e.key === "Enter") save();
          if (e.key === "Escape") { setDraft(value); setEditing(false); }
        }}
        style={style}
      />
    );
  }

  return (
    <span
      onClick={() => setEditing(true)}
      style={{ ...style, cursor: "text", display: "block" }}
    >
      {draft || placeholder}
    </span>
  );
}

// ── Memory row (hairline, matches RefiningList/ConnectionsList) ──────

function MemoryRow({
  title,
  snippet,
  timestamp,
  accentColor,
  onClick,
}: {
  title: string;
  snippet?: string;
  timestamp?: number;
  accentColor: string;
  onClick?: () => void;
}) {
  const [hover, setHover] = useState(false);

  return (
    <li
      role={onClick ? "button" : undefined}
      tabIndex={onClick ? 0 : undefined}
      className="py-2.5 px-2 transition-colors duration-150"
      style={{
        backgroundColor: hover ? "var(--mem-hover)" : "transparent",
        borderBottom: "1px solid color-mix(in srgb, var(--mem-border) 60%, transparent)",
        cursor: onClick ? "pointer" : "default",
      }}
      onClick={onClick}
      onKeyDown={(e) => {
        if (onClick && (e.key === "Enter" || e.key === " ")) {
          e.preventDefault();
          onClick();
        }
      }}
      onMouseEnter={() => setHover(true)}
      onMouseLeave={() => setHover(false)}
    >
      <div className="flex items-baseline gap-2">
        <span
          className="inline-block w-1.5 h-1.5 rounded-full flex-shrink-0"
          style={{ backgroundColor: accentColor, marginTop: 1 }}
        />
        <span
          className="flex-1 truncate"
          style={{
            fontFamily: "var(--mem-font-heading)",
            fontSize: 14,
            fontWeight: 500,
            color: "var(--mem-text)",
          }}
        >
          {title}
        </span>
        {timestamp != null && (
          <span
            style={{
              fontFamily: "var(--mem-font-body)",
              fontSize: 11,
              color: "var(--mem-text-tertiary)",
              whiteSpace: "nowrap",
            }}
          >
            {relativeDate(timestamp)}
          </span>
        )}
      </div>
      {snippet && (
        <p
          className="line-clamp-1 ml-3.5"
          style={{
            fontFamily: "var(--mem-font-body)",
            fontSize: 12,
            color: "var(--mem-text-tertiary)",
            lineHeight: 1.4,
            marginTop: 2,
          }}
        >
          {snippet}
        </p>
      )}
    </li>
  );
}

// ── Main Profile Page ───────────────────────────────────────────────────

export default function ProfilePage({ onBack, onSelectMemory }: ProfilePageProps) {
  const queryClient = useQueryClient();

  const { data: profile } = useQuery({
    queryKey: ["profile"],
    queryFn: getProfile,
  });

  const { data: narrative, isLoading: narrativeLoading } = useQuery({
    queryKey: ["profile-narrative"],
    queryFn: getProfileNarrative,
    refetchInterval: 120000,
    staleTime: 60000,
  });

  const { data: identities = [] } = useQuery({
    queryKey: ["memories", "identity"],
    queryFn: () => listMemoriesRich(undefined, "identity", undefined, 20),
  });

  const { data: preferences = [] } = useQuery({
    queryKey: ["memories", "preference"],
    queryFn: () => listMemoriesRich(undefined, "preference", undefined, 20),
  });

  const { data: goals = [] } = useQuery({
    queryKey: ["memories", "goal"],
    queryFn: () => listMemoriesRich(undefined, "goal", undefined, 10),
  });

  const profileMutation = useMutation({
    mutationFn: (fields: { name?: string; displayName?: string; email?: string; bio?: string }) =>
      updateProfile(profile!.id, fields.name, fields.displayName, fields.email, fields.bio),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["profile"] }),
  });

  const avatarMutation = useMutation({
    mutationFn: (path: string) => setAvatar(path),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["profile"] }),
  });

  const removeAvatarMutation = useMutation({
    mutationFn: () => removeAvatar(),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["profile"] }),
  });

  const regenMutation = useMutation({
    mutationFn: regenerateNarrative,
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["profile-narrative"] }),
  });

  const avatarUrl = profile?.avatar_path ? convertFileSrc(profile.avatar_path) : null;

  const handlePickAvatar = async () => {
    try {
      const selected = await open({
        title: "Choose a profile photo",
        filters: [{ name: "Images", extensions: ["png", "jpg", "jpeg", "webp", "gif"] }],
      });
      if (typeof selected === "string") {
        avatarMutation.mutate(selected);
      }
    } catch (e) {
      console.error("Avatar picker failed:", e);
    }
  };

  if (!profile) return null;

  const displayName = profile.display_name || profile.name;
  const initials = displayName
    .split(" ")
    .map((w: string) => w[0])
    .join("")
    .slice(0, 2)
    .toUpperCase();

  return (
    <div className="flex flex-col gap-8 max-w-2xl mx-auto py-4">
      {/* Back button */}
      <div>
        <button
          onClick={onBack}
          className="p-1.5 -ml-1.5 rounded-md transition-colors duration-150 hover:bg-[var(--mem-hover)]"
          style={{ color: "var(--mem-text-tertiary)", background: "none", border: "none", cursor: "pointer", lineHeight: 0, marginBottom: "12px" }}
        >
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5"><path d="M19 12H5M12 19l-7-7 7-7" /></svg>
        </button>

        {/* Header: avatar + editable fields */}
        <div className="flex items-start gap-5">
          <div className="flex-shrink-0 relative group">
            <button
              onClick={handlePickAvatar}
              className="rounded-full overflow-hidden transition-opacity duration-150 hover:opacity-80"
              style={{ width: 72, height: 72, cursor: "pointer", border: "none", padding: 0, background: "none" }}
              title="Change profile photo"
            >
              {avatarUrl ? (
                <img
                  src={avatarUrl}
                  alt={displayName}
                  style={{ width: 72, height: 72, borderRadius: "50%", objectFit: "cover" }}
                />
              ) : (
                <div
                  style={{
                    width: 72,
                    height: 72,
                    borderRadius: "50%",
                    background: "linear-gradient(135deg, var(--mem-accent-warm), var(--mem-accent-amber))",
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    fontFamily: "var(--mem-font-heading)",
                    fontSize: "24px",
                    color: "white",
                    fontWeight: 500,
                  }}
                >
                  {initials}
                </div>
              )}
            </button>
            {avatarUrl && (
              <button
                onClick={() => removeAvatarMutation.mutate()}
                className="absolute -bottom-1 -right-1 w-5 h-5 rounded-full flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity duration-150"
                style={{
                  backgroundColor: "var(--mem-surface)",
                  border: "1px solid var(--mem-border)",
                  color: "var(--mem-text-tertiary)",
                }}
                title="Remove photo"
              >
                <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
                  <path d="M18 6L6 18M6 6l12 12" />
                </svg>
              </button>
            )}
          </div>

          <div className="flex-1 min-w-0 flex flex-col gap-1">
            <EditableField
              value={profile.name}
              placeholder="Name"
              onSave={(v) => profileMutation.mutate({ name: v, displayName: v })}
              fontSize="20px"
              fontFamily="var(--mem-font-heading)"
              color="var(--mem-text)"
            />
            <EditableField
              value={profile.email || ""}
              placeholder="Add email"
              onSave={(v) => profileMutation.mutate({ email: v })}
              fontSize="13px"
              color="var(--mem-text-secondary)"
            />
            <span
              style={{
                fontFamily: "var(--mem-font-mono)",
                fontSize: "11px",
                color: "var(--mem-text-tertiary)",
                marginTop: "6px",
              }}
            >
              Joined {formatDate(profile.created_at)}
            </span>
          </div>
        </div>
      </div>

      {/* ── Narrative portrait ──────────────────────────────────────── */}
      <section>
        <div className="flex items-center justify-between mb-2">
          <h2 style={SECTION_TITLE_STYLE}>How Origin sees you</h2>
          <button
            onClick={() => regenMutation.mutate()}
            disabled={regenMutation.isPending}
            className="flex items-center gap-1 px-2 py-1 rounded transition-colors duration-150 hover:bg-[var(--mem-hover)]"
            style={{
              fontFamily: "var(--mem-font-body)",
              fontSize: "11px",
              color: "var(--mem-text-tertiary)",
              border: "none",
              background: "none",
              cursor: "pointer",
              opacity: regenMutation.isPending ? 0.5 : 1,
            }}
          >
            <svg
              width="12"
              height="12"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              style={{
                animation: regenMutation.isPending ? "spin 1s linear infinite" : "none",
              }}
            >
              <path d="M21 2v6h-6M3 12a9 9 0 0 1 15-6.7L21 8M3 22v-6h6M21 12a9 9 0 0 1-15 6.7L3 16" />
            </svg>
            Regenerate
          </button>
        </div>
        <p style={SECTION_SUB_STYLE} className="mb-3">
          synthesized from {narrative?.memory_count ?? 0} memories
        </p>

        {narrativeLoading ? (
          <div className="space-y-2">
            <div className="animate-pulse h-3 rounded" style={{ backgroundColor: "var(--mem-hover-strong)", width: "90%" }} />
            <div className="animate-pulse h-3 rounded" style={{ backgroundColor: "var(--mem-hover-strong)", width: "75%" }} />
            <div className="animate-pulse h-3 rounded" style={{ backgroundColor: "var(--mem-hover-strong)", width: "60%" }} />
          </div>
        ) : narrative?.content ? (
          <div
            style={{
              fontFamily: "var(--mem-font-body)",
              fontSize: "14px",
              lineHeight: "1.7",
              color: "var(--mem-text)",
            }}
          >
            {narrative.content.split("\n\n").map((para, i) => (
              <p key={i} style={{ margin: 0, marginBottom: i < narrative.content.split("\n\n").length - 1 ? "12px" : 0 }}>
                {para}
              </p>
            ))}
          </div>
        ) : (
          <p
            style={{
              fontFamily: "var(--mem-font-body)",
              fontSize: 13,
              fontStyle: "italic",
              color: "var(--mem-text-tertiary)",
            }}
          >
            No narrative yet. Confirm some memories to get started.
          </p>
        )}
      </section>

      {/* ── Identity memories ──────────────────────────────────────── */}
      {identities.length > 0 && (
        <section>
          <h2 style={SECTION_TITLE_STYLE}>What Origin knows</h2>
          <p style={SECTION_SUB_STYLE} className="mb-3">
            identity facts that shape every conversation
          </p>
          <ul>
            {identities.map((mem, _i) => (
              <MemoryRow
                key={mem.source_id}
                title={mem.title || mem.content.split("\n")[0].slice(0, 80)}
                snippet={mem.content && mem.title ? mem.content.split("\n")[0].slice(0, 120) : undefined}
                timestamp={mem.last_modified}
                accentColor="var(--mem-accent-indigo)"
                onClick={onSelectMemory ? () => onSelectMemory(mem.source_id) : undefined}
              />
            ))}
          </ul>
        </section>
      )}

      {/* ── Preferences ────────────────────────────────────────────── */}
      {preferences.length > 0 && (
        <section>
          <h2 style={SECTION_TITLE_STYLE}>Preferences</h2>
          <p style={SECTION_SUB_STYLE} className="mb-3">
            how you like things done
          </p>
          <ul>
            {preferences.map((pref) => (
              <MemoryRow
                key={pref.source_id}
                title={pref.title || pref.content.split("\n")[0].slice(0, 80)}
                snippet={pref.content && pref.title ? pref.content.split("\n")[0].slice(0, 120) : undefined}
                timestamp={pref.last_modified}
                accentColor="var(--mem-accent-warm)"
                onClick={onSelectMemory ? () => onSelectMemory(pref.source_id) : undefined}
              />
            ))}
          </ul>
        </section>
      )}

      {/* ── Current focus (goals) ──────────────────────────────────── */}
      <section>
        <h2 style={SECTION_TITLE_STYLE}>Current focus</h2>
        <p style={SECTION_SUB_STYLE} className="mb-3">
          what you are working toward
        </p>
        {goals.length > 0 ? (
          <ul>
            {goals.map((goal) => (
              <MemoryRow
                key={goal.source_id}
                title={goal.title || goal.content.split("\n")[0].slice(0, 80)}
                snippet={goal.content && goal.title ? goal.content.split("\n")[0].slice(0, 120) : undefined}
                timestamp={goal.last_modified}
                accentColor="var(--mem-accent-sage)"
                onClick={onSelectMemory ? () => onSelectMemory(goal.source_id) : undefined}
              />
            ))}
          </ul>
        ) : (
          <p
            style={{
              fontFamily: "var(--mem-font-body)",
              fontSize: 13,
              fontStyle: "italic",
              color: "var(--mem-text-tertiary)",
              paddingTop: 8,
            }}
          >
            No active goals yet
          </p>
        )}
      </section>
    </div>
  );
}
