// SPDX-License-Identifier: AGPL-3.0-only
interface Props {
  narrative: string;
  lastUpdatedMs: number;
}

function formatRelative(ms: number): string {
  const deltaMs = Date.now() - ms;
  const days = Math.floor(deltaMs / 86_400_000);
  if (days <= 0) return "today";
  if (days === 1) return "1d ago";
  if (days < 30) return `${days}d ago`;
  const months = Math.floor(days / 30);
  return months === 1 ? "1mo ago" : `${months}mo ago`;
}

export function ProfileNarrativeCompact({ narrative, lastUpdatedMs }: Props) {
  const text = narrative?.trim() ?? "";
  if (!text) return null;
  return (
    <section data-testid="profile-compact">
      <p className="line-clamp-4 text-sm text-neutral-700 leading-relaxed">{text}</p>
      <span className="mt-1 inline-block text-xs text-neutral-500">
        Updated {formatRelative(lastUpdatedMs)}
      </span>
    </section>
  );
}
