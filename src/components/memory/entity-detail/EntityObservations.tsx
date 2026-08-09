// SPDX-License-Identifier: AGPL-3.0-only
import { useMemo, useRef, useState } from "react";
import { useMutation } from "@tanstack/react-query";
import { useTranslation } from "react-i18next";
import {
  addObservation,
  confirmObservation,
  deleteObservation,
  updateObservation,
  type Observation,
} from "../../../lib/tauri";
import { formatConfidence } from "./formatEntityMetadata";

type EntityObservationsProps = {
  readonly entityId: string;
  readonly entityName: string;
  readonly observations: readonly Observation[];
  readonly onInvalidate: () => void;
};

export function EntityObservations({
  entityId,
  entityName,
  observations: rawObservations,
  onInvalidate,
}: EntityObservationsProps) {
  const { t } = useTranslation();
  const [editingObservationId, setEditingObservationId] = useState<string | null>(null);
  const [editContent, setEditContent] = useState("");
  const [newObservation, setNewObservation] = useState("");
  const [showAddForm, setShowAddForm] = useState(false);
  const cancelEditRef = useRef(false);
  const observations = useMemo(
    () =>
      rawObservations.filter(
        (observation, index, all) =>
          all.findIndex(
            (candidate) => candidate.content.toLowerCase() === observation.content.toLowerCase(),
          ) === index,
      ),
    [rawObservations],
  );
  const updateMutation = useMutation({
    mutationFn: ({ id, content }: { id: string; content: string }) =>
      updateObservation(id, content),
    onSuccess: onInvalidate,
  });
  const deleteMutation = useMutation({
    mutationFn: (id: string) => deleteObservation(id),
    onSuccess: onInvalidate,
  });
  const addMutation = useMutation({
    mutationFn: (content: string) => addObservation(entityId, content, "human", 1),
    onSuccess: () => {
      setNewObservation("");
      setShowAddForm(false);
      onInvalidate();
    },
  });
  const confirmMutation = useMutation({
    mutationFn: ({ id, confirmed }: { id: string; confirmed: boolean }) =>
      confirmObservation(id, confirmed),
    onSuccess: onInvalidate,
  });
  const hasMutationError =
    updateMutation.isError ||
    deleteMutation.isError ||
    addMutation.isError ||
    confirmMutation.isError;

  const saveEdit = (id: string) => {
    if (updateMutation.isPending) return;
    const trimmed = editContent.trim();
    const original = observations.find((observation) => observation.id === id)?.content;
    if (!trimmed || trimmed === original) {
      updateMutation.reset();
      setEditingObservationId(null);
      return;
    }
    updateMutation.mutate(
      { id, content: trimmed },
      { onSuccess: () => setEditingObservationId(null) },
    );
  };

  return (
    <section className="memory-detail-card" aria-labelledby="entity-about-title">
      <div className="memory-detail-card-header">
        <h2 id="entity-about-title" className="memory-detail-section-title">
          {t("entityDetail.aboutTitle")}
        </h2>
        {!showAddForm ? (
          <button
            type="button"
            className="memory-detail-text-button"
            onClick={() => setShowAddForm(true)}
          >
            {t("entityDetail.addNote")}
          </button>
        ) : null}
      </div>
      <div className="memory-detail-card-body">
        {hasMutationError ? (
          <p className="entity-error" role="alert">
            {t("entityDetail.saveError")}
          </p>
        ) : null}
        {observations.length === 0 && !showAddForm ? (
          <p className="entity-empty">{t("entityDetail.emptyAbout")}</p>
        ) : null}
        <div className="entity-obs-list">
          {observations.map((observation) => {
            const confidence = formatConfidence(observation.confidence, 1);
            return (
              <div key={observation.id} className="entity-obs-row">
                <button
                  type="button"
                  className="memory-detail-state-button entity-obs-state"
                  disabled={confirmMutation.isPending}
                  onClick={() =>
                    confirmMutation.mutate({
                      id: observation.id,
                      confirmed: !observation.confirmed,
                    })
                  }
                  aria-label={
                    observation.confirmed
                      ? t("entityDetail.unconfirmObservation")
                      : t("entityDetail.confirmObservation")
                  }
                  title={
                    observation.confirmed
                      ? t("entityDetail.unconfirmObservation")
                      : t("entityDetail.confirmObservation")
                  }
                >
                  <span
                    className={`memory-detail-state-dot ${observation.confirmed ? "is-on" : ""}`}
                  />
                </button>
                {editingObservationId === observation.id ? (
                  <input
                    value={editContent}
                    onChange={(event) => setEditContent(event.target.value)}
                    onBlur={() => {
                      if (cancelEditRef.current) {
                        cancelEditRef.current = false;
                      } else {
                        saveEdit(observation.id);
                      }
                    }}
                    onKeyDown={(event) => {
                      if (event.key === "Enter") saveEdit(observation.id);
                      if (event.key === "Escape") {
                        event.stopPropagation();
                        cancelEditRef.current = true;
                        updateMutation.reset();
                        setEditingObservationId(null);
                      }
                    }}
                    autoFocus
                    disabled={updateMutation.isPending}
                    className="entity-obs-input"
                    aria-label={t("entityDetail.editNote")}
                  />
                ) : (
                  <button
                    type="button"
                    className={`entity-obs-content ${observation.confirmed ? "" : "is-unconfirmed"}`}
                    onClick={() => {
                      setEditingObservationId(observation.id);
                      setEditContent(observation.content);
                    }}
                    title={
                      observation.source_agent
                        ? `${t("entityDetail.editNote")} · ${observation.source_agent}`
                        : t("entityDetail.editNote")
                    }
                  >
                    {observation.content}
                  </button>
                )}
                {confidence ? <span className="entity-obs-confidence">{confidence}</span> : null}
                <button
                  type="button"
                  disabled={deleteMutation.isPending}
                  onClick={() => deleteMutation.mutate(observation.id)}
                  className="entity-obs-delete"
                  aria-label={t("entityDetail.deleteNote")}
                  title={t("entityDetail.deleteNote")}
                >
                  <svg
                    aria-hidden="true"
                    width="12"
                    height="12"
                    viewBox="0 0 24 24"
                    fill="none"
                    stroke="currentColor"
                    strokeWidth="2"
                  >
                    <path d="M18 6L6 18M6 6l12 12" />
                  </svg>
                </button>
              </div>
            );
          })}
          {showAddForm ? (
            <div className="entity-obs-add">
              <input
                value={newObservation}
                onChange={(event) => setNewObservation(event.target.value)}
                onKeyDown={(event) => {
                  if (event.key === "Enter" && newObservation.trim() && !addMutation.isPending) {
                    addMutation.mutate(newObservation.trim());
                  }
                  if (event.key === "Escape") {
                    event.stopPropagation();
                    setShowAddForm(false);
                    setNewObservation("");
                  }
                }}
                placeholder={t("entityDetail.notePlaceholder", { name: entityName })}
                autoFocus
                disabled={addMutation.isPending}
                className="entity-obs-input"
                aria-label={t("entityDetail.addNote")}
              />
            </div>
          ) : null}
        </div>
      </div>
    </section>
  );
}
