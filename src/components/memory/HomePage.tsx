// SPDX-License-Identifier: AGPL-3.0-only
import { useEffect, useMemo, useRef } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import {
  confirmMemory,
  deleteFileChunks,
  dismissContradiction,
  getMemoryStats,
  listConcepts,
  listMemoriesRich,
  listRecentChanges,
  listRecentConcepts,
  listRecentMemories,
  listRecentRetrievals,
  listUnconfirmedMemories,
} from "../../lib/tauri";
import { Greeting } from "./Greeting";
import { WorthAGlanceScroll } from "./WorthAGlanceScroll";
import { RefiningList } from "./RefiningList";
import { ConnectionsList } from "./ConnectionsList";
import { RetrievalsList } from "./RetrievalsList";
import { WhatHappensNextCard, type HomePageState } from "../onboarding/WhatHappensNextCard";
import { GhostConceptsRow } from "../onboarding/GhostConceptsRow";
import { useMilestones } from "../onboarding/useMilestones";
import { FirstConceptModal } from "../onboarding/FirstConceptModal";
import { MilestoneHighlight } from "../onboarding/MilestoneHighlight";

interface HomePageProps {
  onNavigateMemory: (sourceId: string) => void;
  onNavigateStream: () => void;
  onNavigateLog: () => void;
  onNavigateGraph: () => void;
  onSelectConcept?: (conceptId: string) => void;
}

const FIRST_CONCEPT_SHOWN_KEY = "onboarding:firstConceptShownCount";
const MAX_MODAL_SHOWS = 3;

function deriveHomePageState(params: {
  intelligenceReady: boolean;
  memoryCount: number;
  conceptCount: number;
}): HomePageState {
  // Concepts take precedence over memories: if any concept exists the home
  // page is "alive", even if memories were subsequently deleted (which can
  // leave conceptCount > 0 while memoryCount == 0).
  if (params.conceptCount > 0) return "alive";
  if (params.memoryCount > 0) return "gathering";
  if (params.intelligenceReady) return "listening";
  return "seed";
}

export default function HomePage({
  onNavigateMemory,
  onNavigateStream,
  onNavigateLog: _onNavigateLog,
  onNavigateGraph: _onNavigateGraph,
  onSelectConcept,
}: HomePageProps) {
  const queryClient = useQueryClient();

  // Snapshot lastVisitMs ONCE at first render so badges don't drift while the
  // user sits on the page. Default first-time users to a 7-day window.
  //
  // Debounce against HMR / Cmd+R / quick navigation: if the stored anchor is
  // very fresh (< 1h) it's almost certainly a re-mount rather than a genuine
  // return visit, and using it would collapse the "since last visit" window
  // to seconds. Fall back to the 7-day window in that case, and only advance
  // the anchor on unmount when the user actually spent >= 10 min on the page.
  const mountedAtRef = useRef(Date.now());
  const lastVisitMs = useMemo(() => {
    const stored = parseInt(localStorage.getItem("home:lastVisitMs") ?? "0", 10);
    const now = Date.now();
    if (!stored || now - stored < 60 * 60_000) {
      return now - 7 * 86_400_000;
    }
    return stored;
  }, []);

  useEffect(() => {
    return () => {
      if (Date.now() - mountedAtRef.current >= 10 * 60_000) {
        localStorage.setItem("home:lastVisitMs", String(Date.now()));
      }
    };
  }, []);

  const { data: recentConceptItems = [] } = useQuery({
    queryKey: ["recentConceptItems", lastVisitMs],
    queryFn: () => listRecentConcepts(10, lastVisitMs),
    refetchInterval: 30_000,
  });

  const { data: recentMemoryItems = [] } = useQuery({
    queryKey: ["recentMemoryItems", lastVisitMs],
    queryFn: () => listRecentMemories(10, lastVisitMs),
    refetchInterval: 30_000,
  });

  const { data: unconfirmedItems = [] } = useQuery({
    queryKey: ["unconfirmedMemories"],
    queryFn: () => listUnconfirmedMemories(6),
    refetchInterval: 30_000,
  });

  const { data: retrievals = [] } = useQuery({
    queryKey: ["recentRetrievals"],
    queryFn: () => listRecentRetrievals(12),
    refetchInterval: 30_000,
  });

  const { data: changes = [] } = useQuery({
    queryKey: ["recentChanges"],
    queryFn: () => listRecentChanges(3),
    refetchInterval: 30_000,
  });

  const { data: recentConcepts = [] } = useQuery({
    queryKey: ["recent-concepts"],
    queryFn: () => listConcepts("active", undefined, 10),
    refetchInterval: 10_000,
  });

  const { data: stats } = useQuery({
    queryKey: ["memoryStats"],
    queryFn: getMemoryStats,
    refetchInterval: 10_000,
  });

  const { data: recapCount = 0 } = useQuery({
    queryKey: ["recap-count"],
    queryFn: async () => {
      const all = await listMemoriesRich(undefined, undefined, undefined, 200);
      return all.filter((m) => m.is_recap === true).length;
    },
    refetchInterval: 30_000,
  });

  const activityItems = useMemo(
    () =>
      [...recentConceptItems, ...recentMemoryItems]
        .sort((a, b) => b.timestamp_ms - a.timestamp_ms)
        .slice(0, 12),
    [recentConceptItems, recentMemoryItems],
  );

  // Worth-a-glance surfaces items that benefit from a quick confirm/edit pass:
  //   (1) contradiction-flagged items from the recent activity stream
  //   (2) explicitly unconfirmed memories (confirmed=0), regardless of age
  // Dedupe by id so a contradiction-flagged unconfirmed memory doesn't appear
  // twice. Contradictions take precedence in ordering.
  const worthAGlanceItems = useMemo(() => {
    const contradictions = activityItems.filter((i) => i.badge.kind === "needs_review");
    const seen = new Set(contradictions.map((i) => i.id));
    const extras = unconfirmedItems.filter((i) => !seen.has(i.id));
    return [...contradictions, ...extras].slice(0, 6);
  }, [activityItems, unconfirmedItems]);

  const memoryCount = stats?.total ?? 0;
  const conceptCount = recentConcepts.length;

  const { milestones, acknowledge } = useMilestones();
  const intelligenceReady = milestones.some(
    (m) => m.id === "intelligence-ready" && m.first_triggered_at != null,
  );
  const firstTriggeredAt = milestones.find((m) => m.id === "intelligence-ready")
    ?.first_triggered_at;
  const daysInListening = firstTriggeredAt
    ? Math.floor((Date.now() / 1000 - firstTriggeredAt) / 86_400)
    : 0;

  const firstConceptMs = milestones.find(
    (m) => m.id === "first-concept" && m.acknowledged_at == null,
  );
  const firstConceptData = firstConceptMs?.payload as
    | { concept_id?: string; title?: string }
    | undefined;
  const firstConcept = firstConceptData?.concept_id
    ? recentConcepts.find((c) => c.id === firstConceptData.concept_id)
    : null;

  const homePageState = deriveHomePageState({
    intelligenceReady,
    memoryCount,
    conceptCount,
  });

  // Track whether we've incremented shown_count for the current first-concept
  // modal instance, so StrictMode double-invokes don't double-count.
  const incrementedOnMountRef = useRef(false);
  useEffect(() => {
    if (!firstConcept || !firstConceptMs) return;
    if (incrementedOnMountRef.current) return;
    incrementedOnMountRef.current = true;
    const current = parseInt(
      localStorage.getItem(FIRST_CONCEPT_SHOWN_KEY) || "0",
      10,
    );
    const next = current + 1;
    localStorage.setItem(FIRST_CONCEPT_SHOWN_KEY, String(next));
    if (next > MAX_MODAL_SHOWS) {
      acknowledge("first-concept");
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [firstConcept?.id, firstConceptMs?.id]);

  const firstConceptShownCount = parseInt(
    localStorage.getItem(FIRST_CONCEPT_SHOWN_KEY) || "0",
    10,
  );
  const shouldShowFirstConceptModal =
    !!firstConcept &&
    !!firstConceptMs &&
    firstConceptShownCount <= MAX_MODAL_SHOWS;

  const isEmpty =
    activityItems.length === 0 && retrievals.length === 0 && changes.length === 0;

  return (
    <div className="flex flex-col gap-8 pb-16">
      <Greeting memoryCount={memoryCount} conceptCount={conceptCount} />

      {isEmpty ? (
        <>
          <WhatHappensNextCard
            state={homePageState}
            memoryCount={memoryCount}
            daysInListening={daysInListening}
          />
          <MilestoneHighlight
            active={
              !shouldShowFirstConceptModal &&
              milestones.some(
                (m) => m.id === "first-concept" && m.acknowledged_at == null,
              )
            }
            onSeen={() => {}}
            intensity="full"
          >
            <section>
              <GhostConceptsRow />
            </section>
          </MilestoneHighlight>
        </>
      ) : (
        <>
          <WorthAGlanceScroll
            items={worthAGlanceItems}
            onConfirm={async (id) => {
              await Promise.all([
                dismissContradiction(id).catch(() => undefined),
                confirmMemory(id, true).catch(() => undefined),
              ]);
              await queryClient.invalidateQueries({ queryKey: ["recentConceptItems"] });
              await queryClient.invalidateQueries({ queryKey: ["recentMemoryItems"] });
              await queryClient.invalidateQueries({ queryKey: ["unconfirmedMemories"] });
            }}
            onDelete={async (id) => {
              await deleteFileChunks("memory", id);
              await queryClient.invalidateQueries({ queryKey: ["recentConceptItems"] });
              await queryClient.invalidateQueries({ queryKey: ["recentMemoryItems"] });
              await queryClient.invalidateQueries({ queryKey: ["unconfirmedMemories"] });
            }}
            onEdit={async (_kind, _id) => {
              await queryClient.invalidateQueries({ queryKey: ["recentMemoryItems"] });
              await queryClient.invalidateQueries({ queryKey: ["unconfirmedMemories"] });
            }}
            onNavigate={(kind, id) => {
              if (kind === "concept") onSelectConcept?.(id);
              else onNavigateMemory(id);
            }}
            recapCount={recapCount}
            onViewRecaps={onNavigateStream}
          />
          <RefiningList changes={changes} concepts={recentConcepts} onSelectConcept={onSelectConcept} />
          <RetrievalsList
            events={retrievals}
            onSelectConceptById={(conceptId) => onSelectConcept?.(conceptId)}
          />
          <ConnectionsList
            onSelectConcept={onSelectConcept}
            onSelectEntity={undefined}
          />
        </>
      )}

      {shouldShowFirstConceptModal && firstConcept && (
        <FirstConceptModal
          concept={firstConcept}
          onOpen={(id) => {
            localStorage.removeItem(FIRST_CONCEPT_SHOWN_KEY);
            acknowledge("first-concept");
            onSelectConcept?.(id);
          }}
          onDismiss={() => {
            localStorage.removeItem(FIRST_CONCEPT_SHOWN_KEY);
            acknowledge("first-concept");
          }}
        />
      )}
    </div>
  );
}
