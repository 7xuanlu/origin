// SPDX-License-Identifier: AGPL-3.0-only
import { useEffect, useState } from "react";
import { useTranslation } from "react-i18next";
import {
  FACET_COLORS,
  STABILITY_TIERS,
  acceptPendingRevision,
  agentDisplayName,
  dismissPendingRevision,
  getPendingRevision,
  type MemoryItem,
  type PendingRevision,
} from "../../lib/tauri";
import ContentRenderer from "./ContentRenderer";

interface MemoryListRowProps {
  memory: MemoryItem;
  onConfirm: (sourceId: string, confirmed: boolean) => void;
  onDelete: (sourceId: string) => void;
  onPin?: (sourceId: string) => void;
  onUnpin?: (sourceId: string) => void;
  onClick?: (sourceId: string) => void;
  style?: React.CSSProperties;
}

function timeAgo(ts: number): string {
  const now = Date.now() / 1000;
  const diff = now - ts;
  if (diff < 60) return "just now";
  if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
  if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`;
  if (diff < 604800) return `${Math.floor(diff / 86400)}d ago`;
  return new Date(ts * 1000).toLocaleDateString();
}

export default function MemoryListRow({
  memory,
  onConfirm,
  onDelete,
  onPin,
  onUnpin,
  onClick,
  style,
}: MemoryListRowProps) {
  const { t } = useTranslation();
  const [deleting, setDeleting] = useState(false);
  const [pendingRevision, setPendingRevision] = useState<PendingRevision | null>(null);

  const facetType = memory.memory_type ?? "fact";
  const tier = STABILITY_TIERS[facetType] ?? "ephemeral";
  const isConfirmed = memory.stability === "confirmed" || (!memory.stability && memory.confirmed);
  const stability = memory.stability ?? (memory.confirmed ? "confirmed" : "new");
  const displayText = memory.source_text || memory.summary || memory.content;
  const rowTitle = memory.title || displayText || t("memoryList.untitledMemory");
  const agentLabel = agentDisplayName(memory.source_agent);
  const statusLabel = (() => {
    if (isConfirmed) return t("memoryList.statusConfirmed");
    switch (stability) {
      case "learned":
        return t("memoryList.statusLearned");
      case "new":
        return t("memoryList.statusNew");
      default:
        return stability;
    }
  })();
  const handleOpen = () => onClick?.(memory.source_id);
  const handleKeyDown = (event: React.KeyboardEvent<HTMLElement>) => {
    if (event.target !== event.currentTarget) return;
    if (event.key !== "Enter" && event.key !== " ") return;
    event.preventDefault();
    handleOpen();
  };

  useEffect(() => {
    if (tier === "protected" && isConfirmed) {
      getPendingRevision(memory.source_id).then(setPendingRevision).catch(() => {});
    }
  }, [tier, isConfirmed, memory.source_id]);

  const handleAcceptRevision = async () => {
    if (!pendingRevision) return;
    await acceptPendingRevision(memory.source_id);
    setPendingRevision(null);
    window.dispatchEvent(new CustomEvent("memory-updated"));
  };

  const handleDismissRevision = async () => {
    if (!pendingRevision) return;
    await dismissPendingRevision(memory.source_id);
    setPendingRevision(null);
  };

  if (deleting) return null;

  return (
    <article
      aria-label={rowTitle}
      className="memory-list-row"
      onKeyDown={handleKeyDown}
      tabIndex={0}
      style={style}
    >
      <div className="memory-list-row-body">
        <div className="memory-list-row-copy">
          <button
            type="button"
            aria-label={t("memoryList.openMemory")}
            className="memory-list-row-title"
            onClick={handleOpen}
          >
            {rowTitle}
          </button>
          {displayText && displayText !== rowTitle && (
            <p className="memory-list-row-preview">
              <ContentRenderer
                content={displayText}
                structuredFields={memory.structured_fields}
                variant="card"
              />
            </p>
          )}
        </div>

        <dl className="memory-list-row-metadata">
          <div>
            <dt>{t("memoryList.type")}</dt>
            <dd>
              <span className={`memory-facet-pill ${FACET_COLORS[facetType] ?? FACET_COLORS.fact}`}>
                {facetType}
              </span>
            </dd>
          </div>
          <div>
            <dt>{t("memoryList.space")}</dt>
            <dd className="capitalize">{memory.domain ?? "—"}</dd>
          </div>
          <div>
            <dt>{t("memoryList.agent")}</dt>
            <dd>
              {agentLabel ? (
                <span className="memory-chip indigo">{agentLabel}</span>
              ) : (
                t("memoryList.manual")
              )}
            </dd>
          </div>
          <div>
            <dt>{t("memoryList.status")}</dt>
            <dd>{statusLabel}</dd>
          </div>
          <div>
            <dt>{t("memoryList.updated")}</dt>
            <dd>{timeAgo(memory.last_modified)}</dd>
          </div>
        </dl>

        {pendingRevision && (
          <div className="memory-list-row-update">
            <div>
              <span className="memory-list-row-update-label">
                {pendingRevision.source_agent
                  ? t("memoryList.proposedUpdateFrom", { agent: pendingRevision.source_agent })
                  : t("memoryList.proposedUpdate")}
              </span>
              <p>{pendingRevision.content}</p>
            </div>
            <div className="memory-list-row-update-actions">
              <button type="button" aria-label={t("memoryList.acceptUpdate")} onClick={handleAcceptRevision}>
                {t("memoryList.acceptUpdate")}
              </button>
              <button type="button" aria-label={t("memoryList.dismissUpdate")} onClick={handleDismissRevision}>
                {t("memoryList.dismissUpdate")}
              </button>
            </div>
          </div>
        )}
      </div>

      <div className="memory-list-row-actions">
        <button
          type="button"
          aria-label={isConfirmed ? t("memoryList.unconfirmMemory") : t("memoryList.confirmMemory")}
          onClick={() => onConfirm(memory.source_id, !isConfirmed)}
        >
          {isConfirmed ? t("memoryList.confirmed") : t("memoryList.confirm")}
        </button>
        {(onPin || onUnpin) && (
          memory.pinned ? (
            <button type="button" aria-label={t("memoryList.unpinMemory")} onClick={() => onUnpin?.(memory.source_id)}>
              {t("memoryList.pinned")}
            </button>
          ) : (
            <button type="button" aria-label={t("memoryList.pinMemory")} onClick={() => onPin?.(memory.source_id)}>
              {t("memoryList.pin")}
            </button>
          )
        )}
        <button
          type="button"
          aria-label={t("memoryList.deleteMemory")}
          onClick={() => {
            setDeleting(true);
            onDelete(memory.source_id);
          }}
        >
          {t("memoryList.deleteMemory")}
        </button>
      </div>
    </article>
  );
}
