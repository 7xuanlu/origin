import {
  forwardRef,
  useCallback,
  useEffect,
  useImperativeHandle,
  useMemo,
  useRef,
  useState,
} from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { useTranslation } from "react-i18next";

import {
  getPage,
  listSpaces,
  publishPageDraft,
  type Page,
  type Space,
} from "../../../lib/tauri";
import {
  usePageDraftAutosave,
  type PageDraftSnapshot,
} from "./usePageDraftAutosave";
import "./pageActions.css";
import "./pageDraftEditor.css";

export type PageDraftEditorHandle = {
  readonly flush: () => Promise<boolean>;
  readonly getIdentity: () => {
    readonly draftId: string | null;
    readonly version: number | null;
  };
  readonly requestBack: () => Promise<void>;
};

type PageDraftEditorProps = {
  readonly draftId?: string;
  readonly onBack: () => void;
  readonly onEscapeBeforeLeave?: () => boolean;
  readonly onOpenExisting: (pageId: string) => void;
  readonly onPublished: (pageId: string) => void;
  readonly space: string | null;
};

type PublishConflict = {
  readonly existingPageId: string;
  readonly existingPageTitle: string | null;
};

function errorProperty(error: unknown, ...keys: string[]): string | null {
  if (!error || typeof error !== "object") return null;
  const record = error as Record<string, unknown>;
  for (const key of keys) {
    if (typeof record[key] === "string") return record[key] as string;
  }
  return null;
}

function errorCode(error: unknown): string | null {
  return errorProperty(error, "code");
}

function selectableSpaces(spaces: readonly Space[], selected: string | null): Space[] {
  if (!selected || spaces.some((candidate) => candidate.name === selected)) return [...spaces];
  return [{
    id: `current-${selected}`,
    name: selected,
    description: null,
    suggested: false,
    starred: false,
    sort_order: -1,
    memory_count: 0,
    entity_count: 0,
    created_at: 0,
    updated_at: 0,
  }, ...spaces];
}

type HydratedEditorProps = PageDraftEditorProps & {
  readonly initialPage: Page | null;
};

const HydratedPageDraftEditor = forwardRef<PageDraftEditorHandle, HydratedEditorProps>(
  function HydratedPageDraftEditor({
    initialPage,
    onBack,
    onEscapeBeforeLeave,
    onOpenExisting,
    onPublished,
    space: initialSpace,
  }, ref) {
    const { t } = useTranslation();
    const queryClient = useQueryClient();
    const titleRef = useRef<HTMLInputElement>(null);
    const [title, setTitle] = useState(initialPage?.title ?? "");
    const [content, setContent] = useState(initialPage?.content ?? "");
    const [space, setSpace] = useState(initialPage?.space ?? initialSpace ?? "");
    const [operationKind, setOperationKind] = useState<"idle" | "leaving" | "publishing">("idle");
    const [publishError, setPublishError] = useState<Error | null>(null);
    const [publishConflict, setPublishConflict] = useState<PublishConflict | null>(null);
    const [publishVersionConflict, setPublishVersionConflict] = useState(false);
    const [reloadError, setReloadError] = useState(false);
    const leavePromiseRef = useRef<Promise<void> | null>(null);
    const operationRef = useRef<
      | { readonly kind: "leaving"; readonly promise: Promise<boolean> }
      | { readonly kind: "publishing"; readonly promise: Promise<void> }
      | null
    >(null);

    const snapshot = useMemo<PageDraftSnapshot>(() => ({
      title,
      content,
      space: space || null,
    }), [content, space, title]);
    const initialSnapshot = useMemo<PageDraftSnapshot>(() => ({
      title: initialPage?.title ?? "",
      content: initialPage?.content ?? "",
      space: initialPage?.space ?? initialSpace ?? null,
    }), [initialPage, initialSpace]);
    const reconcileSpace = useCallback((reconciledSpace: string | null) => {
      setSpace(reconciledSpace ?? "");
    }, []);
    const autosave = usePageDraftAutosave({
      draftId: initialPage?.id,
      initial: initialSnapshot,
      initialVersion: initialPage?.version,
      onSpaceReconciled: reconcileSpace,
      snapshot,
    });

    const beginLeaving = useCallback((): Promise<boolean> => {
      const active = operationRef.current;
      if (active?.kind === "leaving") return active.promise;
      if (active?.kind === "publishing") {
        return active.promise.then(() => false, () => false);
      }

      setOperationKind("leaving");
      const leaving = autosave.flush();
      operationRef.current = { kind: "leaving", promise: leaving };
      const finish = () => {
        if (operationRef.current?.promise === leaving) {
          operationRef.current = null;
          setOperationKind("idle");
        }
      };
      void leaving.then(finish, finish);
      return leaving;
    }, [autosave.flush]);

    const requestBack = useCallback((): Promise<void> => {
      if (leavePromiseRef.current) return leavePromiseRef.current;
      const leave = (async () => {
        if (await beginLeaving()) onBack();
      })();
      leavePromiseRef.current = leave;
      void leave.finally(() => {
        if (leavePromiseRef.current === leave) leavePromiseRef.current = null;
      });
      return leave;
    }, [beginLeaving, onBack]);

    useImperativeHandle(ref, () => ({
      flush: beginLeaving,
      getIdentity: autosave.getIdentity,
      requestBack,
    }), [autosave.getIdentity, beginLeaving, requestBack]);

    useEffect(() => {
      const handleEscape = (event: KeyboardEvent) => {
        if (event.key !== "Escape") return;
        event.preventDefault();
        event.stopPropagation();
        if (onEscapeBeforeLeave?.()) return;
        void requestBack();
      };
      window.addEventListener("keydown", handleEscape, true);
      return () => window.removeEventListener("keydown", handleEscape, true);
    }, [onEscapeBeforeLeave, requestBack]);

    const spacesQuery = useQuery({
      queryKey: ["spaces"],
      queryFn: listSpaces,
      staleTime: 30_000,
    });
    const options = useMemo(
      () => selectableSpaces(spacesQuery.data ?? [], space || null),
      [space, spacesQuery.data],
    );

    const reloadLatest = async () => {
      const identity = autosave.getIdentity();
      if (!identity.draftId) return;
      setReloadError(false);
      try {
        const latest = await getPage(identity.draftId);
        if (!latest || latest.status !== "draft") {
          setReloadError(true);
          return;
        }
        const latestSnapshot = {
          title: latest.title,
          content: latest.content,
          space: latest.space ?? null,
        };
        setTitle(latest.title);
        setContent(latest.content);
        setSpace(latest.space ?? "");
        autosave.adoptRemote({
          draftId: latest.id,
          version: latest.version,
          snapshot: latestSnapshot,
        });
        setPublishError(null);
        setPublishVersionConflict(false);
      } catch {
        setReloadError(true);
      }
    };

    const finishPublished = async (published: Page) => {
      await Promise.all([
        queryClient.invalidateQueries({ queryKey: ["pages"] }),
        queryClient.invalidateQueries({ queryKey: ["pages", "active"] }),
        queryClient.invalidateQueries({ queryKey: ["pages", "draft"] }),
        queryClient.invalidateQueries({ queryKey: ["page", published.id] }),
        queryClient.invalidateQueries({ queryKey: ["spaces-page-counts"] }),
        queryClient.invalidateQueries({ queryKey: ["sidebar-space-page-counts"] }),
      ]);
      onPublished(published.id);
    };

    const publish = (): Promise<void> => {
      const active = operationRef.current;
      if (active?.kind === "publishing") return active.promise;
      if (active?.kind === "leaving") {
        return active.promise.then(() => {}, () => {});
      }
      if (title.trim().length === 0 || content.trim().length === 0) {
        return Promise.resolve();
      }

      setOperationKind("publishing");
      const publishingPromise = (async () => {
        setPublishError(null);
        setPublishConflict(null);
        setPublishVersionConflict(false);
        let attemptedDraftId: string | null = null;
        let attemptedVersion: number | null = null;
        try {
          if (!await autosave.flush()) return;

          const identity = autosave.getIdentity();
          if (!identity.draftId || identity.version == null) return;
          attemptedDraftId = identity.draftId;
          attemptedVersion = identity.version;
          const published = await publishPageDraft({
            id: identity.draftId,
            expectedVersion: identity.version,
          });
          await finishPublished(published);
        } catch (cause) {
          const nextError = cause instanceof Error ? cause : new Error(String(cause));
          const code = errorCode(cause);
          if (
            attemptedDraftId
            && code !== "page_title_conflict"
          ) {
            try {
              const reconciled = await getPage(attemptedDraftId);
              if (
                reconciled?.status === "active"
                && attemptedVersion !== null
                && reconciled.version === attemptedVersion + 1
              ) {
                await finishPublished(reconciled);
                return;
              }
              if (reconciled?.status === "active") {
                await finishPublished(reconciled);
                return;
              }
            } catch {
              // Keep the original publish failure as the actionable error.
            }
          }
          if (code === "page_title_conflict") {
            const existingPageId = errorProperty(
              cause,
              "existingPageId",
              "existing_page_id",
            );
            if (existingPageId) {
              setPublishConflict({
                existingPageId,
                existingPageTitle: errorProperty(
                  cause,
                  "existingPageTitle",
                  "existing_page_title",
                ),
              });
            }
          } else if (code === "draft_version_conflict") {
            setPublishVersionConflict(true);
          }
          setPublishError(nextError);
        }
      })();
      operationRef.current = { kind: "publishing", promise: publishingPromise };
      const finish = () => {
        if (operationRef.current?.promise === publishingPromise) {
          operationRef.current = null;
          setOperationKind("idle");
        }
      };
      void publishingPromise.then(finish, finish);
      return publishingPromise;
    };

    const renameDraft = () => {
      setPublishConflict(null);
      setPublishError(null);
      titleRef.current?.focus();
      titleRef.current?.select();
    };

    const locked = operationKind !== "idle";
    const publishing = operationKind === "publishing";
    const canPublish = title.trim().length > 0
      && content.trim().length > 0
      && !locked
      && autosave.status !== "conflict"
      && !publishConflict
      && !publishVersionConflict;

    return (
      <section className="page-draft-editor" aria-labelledby="page-draft-editor-heading">
        <div className="page-draft-editor-axis">
          <header className="page-draft-editor-header">
            <button
              className="page-draft-back"
              disabled={locked}
              onClick={() => void requestBack()}
              type="button"
            >
              {t("pages.editor.back")}
            </button>
            <div className="page-draft-status" aria-live="polite">
              {autosave.status === "saving" && t("pages.editor.saving")}
              {autosave.status === "saved" && t("pages.editor.saved")}
            </div>
            <button
              className="page-draft-publish"
              disabled={!canPublish}
              onClick={() => void publish()}
              type="button"
            >
              {publishing ? t("pages.editor.publishing") : t("pages.editor.publish")}
            </button>
          </header>

          <h1 className="sr-only" id="page-draft-editor-heading">
            {t("pages.editor.heading")}
          </h1>
          <div className="page-draft-notices">
            {autosave.status === "error" && (
              <div className="page-draft-notice page-draft-notice-error" role="alert">
                <span>{t("pages.editor.saveError")}</span>
                <button onClick={() => void autosave.retry()} type="button">
                  {t("pages.editor.retrySave")}
                </button>
              </div>
            )}

            {(autosave.status === "conflict" || publishVersionConflict) && (
              <div className="page-draft-notice page-draft-notice-error" role="alert">
                <span>
                  {reloadError
                    ? t("pages.editor.reloadError")
                    : t("pages.editor.versionConflict")}
                </span>
                <button onClick={() => void reloadLatest()} type="button">
                  {t("pages.editor.reloadLatest")}
                </button>
              </div>
            )}

            {publishConflict && (
              <div className="page-draft-notice page-draft-notice-error" role="alert">
                <span>{t("pages.editor.titleConflict")}</span>
                <div>
                  <button
                    className="page-draft-conflict-action"
                    onClick={() => onOpenExisting(publishConflict.existingPageId)}
                    type="button"
                  >
                    {t("pages.editor.openExisting")}
                  </button>
                  <button
                    className="page-draft-conflict-action"
                    onClick={renameDraft}
                    type="button"
                  >
                    {t("pages.editor.renameDraft")}
                  </button>
                </div>
              </div>
            )}

            {publishError && !publishConflict && !publishVersionConflict && (
              <div className="page-draft-notice page-draft-notice-error" role="alert">
                {t("pages.editor.publishError")}
              </div>
            )}
          </div>
          <input
            aria-label={t("pages.editor.titleLabel")}
            autoFocus
            className="page-draft-title"
            disabled={locked}
            onChange={(event) => setTitle(event.target.value)}
            placeholder={t("pages.editor.titlePlaceholder")}
            ref={titleRef}
            value={title}
          />

          <label className="page-draft-space">
            <span>{t("pages.editor.spaceLabel")}</span>
            <select
              disabled={locked}
              onChange={(event) => setSpace(event.target.value)}
              value={space}
            >
              <option value="">{t("pages.editor.noSpace")}</option>
              {options.map((candidate) => (
                <option key={candidate.id} value={candidate.name}>{candidate.name}</option>
              ))}
            </select>
          </label>

          <textarea
            aria-label={t("pages.editor.contentLabel")}
            className="page-draft-content"
            disabled={locked}
            onChange={(event) => setContent(event.target.value)}
            placeholder={t("pages.editor.contentPlaceholder")}
            value={content}
          />
        </div>
      </section>
    );
  },
);

export const PageDraftEditor = forwardRef<PageDraftEditorHandle, PageDraftEditorProps>(
  function PageDraftEditor({ draftId, ...props }, ref) {
    const { t } = useTranslation();
    const draftQuery = useQuery({
      queryKey: ["page-draft", draftId],
      queryFn: () => getPage(draftId!),
      enabled: Boolean(draftId),
      retry: false,
    });

    if (draftId && draftQuery.isPending) {
      return <div className="page-draft-load-state">{t("pages.editor.loading")}</div>;
    }
    if (draftId && draftQuery.isError) {
      return (
        <div className="page-draft-load-state page-draft-notice-error" role="alert">
          <p>{t("pages.editor.loadError")}</p>
          <button onClick={() => void draftQuery.refetch()} type="button">
            {t("pages.editor.tryAgain")}
          </button>
        </div>
      );
    }
    if (draftId && !draftQuery.data) {
      return (
        <div className="page-draft-load-state">
          <p>{t("pages.editor.missing")}</p>
          <button onClick={props.onBack} type="button">{t("pages.editor.back")}</button>
        </div>
      );
    }
    if (draftId && draftQuery.data?.status !== "draft") {
      return (
        <div className="page-draft-load-state">
          <p>{t("pages.editor.notDraft")}</p>
          <button onClick={props.onBack} type="button">{t("pages.editor.back")}</button>
        </div>
      );
    }

    const initialPage = draftId ? draftQuery.data ?? null : null;
    return (
      <HydratedPageDraftEditor
        {...props}
        initialPage={initialPage}
        ref={ref}
      />
    );
  },
);
