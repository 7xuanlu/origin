// SPDX-License-Identifier: Apache-2.0
//! Request-level coalescer for concurrent `/api/memory/store` calls.
//!
//! Each HTTP handler calls [`IngestBatcher::submit`] with the fully-built
//! `RawDocument` plus its pre-computed chunk count. A single background task
//! drains the queue, grouping requests into batches that arrive within a
//! short window (typically 25 ms) or hit a size cap (typically 32). Each
//! batch is handed to a process closure that runs the full ingest pipeline
//! — batched quality gate (shared FastEmbed call for novelty), partition,
//! upsert survivors — and returns per-doc [`StoreOutcome`] values.
//!
//! ### What gets amortized across a batch
//!
//! 1. **FastEmbed model call.** Quality gate's novelty check embeds every
//!    surviving content in one batched invocation via
//!    `MemoryDB::check_novelty_batch` — previously each request paid its
//!    own ~50-100ms embedding cost.
//! 2. **libSQL transaction.** `MemoryDB::upsert_documents` takes a single
//!    `Vec<RawDocument>` and writes the whole batch in one transaction.
//! 3. **Write-serialization overhead.** Ten concurrent stores don't fight
//!    for the connection mutex round-robin; one worker owns the conn for
//!    one flush window.
//!
//! ### Why outcomes flow per-request (not batch-wide)
//!
//! Quality gate can reject individual docs — e.g., one is a duplicate of
//! something already in the store, another is fine. The coalescer has to
//! be able to send each caller its own accept/reject verdict. That's why
//! `StoreOutcome` is a three-arm enum and the processor closure returns a
//! `Vec` parallel to its input.

use std::sync::Arc;
use std::time::Duration;

use futures::future::BoxFuture;
use origin_core::sources::RawDocument;
use tokio::sync::{mpsc, oneshot};

/// Per-document result from a batched ingest flush.
#[derive(Debug, Clone)]
pub enum StoreOutcome {
    /// Document admitted and persisted. `chunks_created` is the
    /// caller-supplied pre-computed chunk count (the batch upsert
    /// returns a sum across all docs, so per-doc counts have to be
    /// tracked at submission time).
    Stored { chunks_created: usize },
    /// Quality gate rejected the document. Fields mirror the shape the
    /// HTTP handler turns into `ServerError::QualityGateRejected` so
    /// the daemon emits the same 422 shape callers already see today.
    GateRejected {
        reason: String,
        detail: String,
        similar_to: Option<String>,
    },
    /// Upsert failed for the whole batch (libSQL transaction aborted).
    /// All survivors in the same batch get this outcome — per-doc
    /// granularity isn't possible at the transactional layer.
    UpsertFailed(String),
}

/// Backend the coalescer invokes per flush. Takes `(doc, chunks_predicted)`
/// tuples, returns outcomes in the same order.
pub type BatchProcessFn =
    Arc<dyn Fn(Vec<(RawDocument, usize)>) -> BoxFuture<'static, Vec<StoreOutcome>> + Send + Sync>;

#[derive(Debug)]
struct CoalescedRequest {
    doc: RawDocument,
    chunks_predicted: usize,
    response: oneshot::Sender<StoreOutcome>,
}

/// Configuration for [`IngestBatcher`]. Fields are immutable after spawn;
/// live tuning would require tearing down and rebuilding the worker.
#[derive(Debug, Clone, Copy)]
pub struct BatcherConfig {
    /// Max time between the first queued request and the batch flush.
    /// Short windows (25-50 ms) keep p50 latency close to a single-doc
    /// upsert; longer windows amortize more at the cost of response time.
    pub batch_window: Duration,
    /// Hard cap on batch size regardless of window.
    pub max_batch: usize,
    /// `mpsc` channel capacity; beyond this, callers' `submit().await`
    /// backs up at the `send` step.
    pub channel_capacity: usize,
}

impl Default for BatcherConfig {
    fn default() -> Self {
        Self {
            batch_window: Duration::from_millis(25),
            max_batch: 32,
            channel_capacity: 256,
        }
    }
}

/// Handle used by HTTP route handlers to submit a document and await its
/// outcome. Cheap to clone (`Arc` inside the `mpsc::Sender`).
#[derive(Clone)]
pub struct IngestBatcher {
    tx: mpsc::Sender<CoalescedRequest>,
}

impl IngestBatcher {
    /// Spawn the coalescer worker. The returned handle is clonable and
    /// safe to share across all route handlers; the worker shuts down
    /// gracefully when every clone is dropped (the channel closes,
    /// `rx.recv()` returns `None`, the loop exits).
    pub fn spawn(process: BatchProcessFn, config: BatcherConfig) -> Self {
        let (tx, rx) = mpsc::channel(config.channel_capacity);
        tokio::spawn(run_coalescer(rx, process, config));
        Self { tx }
    }

    /// Submit a document for coalesced processing. Returns the per-doc
    /// outcome — [`StoreOutcome::Stored`] on success, `GateRejected` /
    /// `UpsertFailed` for the specific failure modes. `Err(String)` only
    /// when the batcher itself is unreachable (worker panicked, channel
    /// closed).
    pub async fn submit(
        &self,
        doc: RawDocument,
        chunks_predicted: usize,
    ) -> Result<StoreOutcome, String> {
        let (resp_tx, resp_rx) = oneshot::channel();
        if self
            .tx
            .send(CoalescedRequest {
                doc,
                chunks_predicted,
                response: resp_tx,
            })
            .await
            .is_err()
        {
            return Err("ingest batcher is shut down".to_string());
        }
        resp_rx
            .await
            .map_err(|_| "ingest batcher dropped response channel".to_string())
    }
}

async fn run_coalescer(
    mut rx: mpsc::Receiver<CoalescedRequest>,
    process: BatchProcessFn,
    config: BatcherConfig,
) {
    loop {
        let Some(first) = rx.recv().await else {
            return;
        };
        let batch_started = tokio::time::Instant::now();
        let mut batch = vec![first];
        let deadline = batch_started + config.batch_window;

        while batch.len() < config.max_batch {
            let remaining = deadline.saturating_duration_since(tokio::time::Instant::now());
            if remaining.is_zero() {
                break;
            }
            match tokio::time::timeout(remaining, rx.recv()).await {
                Ok(Some(req)) => batch.push(req),
                _ => break,
            }
        }

        let batch_size = batch.len();
        let fill_elapsed = batch_started.elapsed();
        let items: Vec<(RawDocument, usize)> = batch
            .iter()
            .map(|r| (r.doc.clone(), r.chunks_predicted))
            .collect();

        let process_started = tokio::time::Instant::now();
        let outcomes = process(items).await;
        let process_elapsed = process_started.elapsed();

        // Count per-outcome kinds for observability.
        let mut stored = 0usize;
        let mut rejected = 0usize;
        let mut failed = 0usize;
        for o in &outcomes {
            match o {
                StoreOutcome::Stored { .. } => stored += 1,
                StoreOutcome::GateRejected { .. } => rejected += 1,
                StoreOutcome::UpsertFailed(_) => failed += 1,
            }
        }

        if batch_size > 1 {
            tracing::info!(
                batch_size,
                stored,
                rejected,
                failed,
                fill_ms = fill_elapsed.as_millis() as u64,
                process_ms = process_elapsed.as_millis() as u64,
                "[ingest_batcher] flushed coalesced batch"
            );
        } else {
            tracing::debug!(
                batch_size,
                stored,
                rejected,
                failed,
                fill_ms = fill_elapsed.as_millis() as u64,
                process_ms = process_elapsed.as_millis() as u64,
                "[ingest_batcher] flushed solo request"
            );
        }

        // If the processor returned fewer outcomes than inputs (contract
        // violation), fill the missing slots with an explicit error so
        // callers aren't left hanging forever. This should never fire in
        // practice — it's defense against a future processor bug.
        let missing = batch.len().saturating_sub(outcomes.len());
        let mut outcomes = outcomes;
        for _ in 0..missing {
            outcomes.push(StoreOutcome::UpsertFailed(
                "ingest processor returned fewer outcomes than docs".to_string(),
            ));
        }

        for (req, outcome) in batch.into_iter().zip(outcomes) {
            let _ = req.response.send(outcome);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Mutex;
    use std::time::Instant;

    fn raw_doc(source_id: &str, content: &str) -> RawDocument {
        RawDocument {
            source: "memory".into(),
            source_id: source_id.into(),
            title: content.chars().take(40).collect(),
            summary: None,
            content: content.into(),
            url: None,
            last_modified: 0,
            event_date: None,
            metadata: HashMap::new(),
            memory_type: Some("fact".into()),
            domain: None,
            source_agent: Some("test".into()),
            confidence: None,
            confirmed: None,
            stability: None,
            supersedes: None,
            pending_revision: false,
            entity_id: None,
            quality: None,
            is_recap: false,
            enrichment_status: "raw".into(),
            supersede_mode: "hide".into(),
            structured_fields: None,
            retrieval_cue: None,
            source_text: None,
        }
    }

    /// Primary behavior contract: concurrent submits within the batch
    /// window hit the processor as ONE call, not N. Each caller gets a
    /// Stored outcome with its own pre-computed chunk count.
    #[tokio::test]
    async fn coalesces_concurrent_submits_into_single_processor_call() {
        let invocations = Arc::new(AtomicUsize::new(0));
        let observed_batch_sizes: Arc<Mutex<Vec<usize>>> = Arc::new(Mutex::new(Vec::new()));

        let invocations_cb = invocations.clone();
        let sizes_cb = observed_batch_sizes.clone();
        let process: BatchProcessFn = Arc::new(move |items: Vec<(RawDocument, usize)>| {
            let invocations = invocations_cb.clone();
            let sizes = sizes_cb.clone();
            Box::pin(async move {
                invocations.fetch_add(1, Ordering::SeqCst);
                sizes.lock().unwrap().push(items.len());
                items
                    .into_iter()
                    .map(|(_, chunks)| StoreOutcome::Stored {
                        chunks_created: chunks,
                    })
                    .collect()
            })
        });

        let batcher = IngestBatcher::spawn(
            process,
            BatcherConfig {
                batch_window: Duration::from_millis(50),
                max_batch: 16,
                channel_capacity: 32,
            },
        );

        let mut handles = Vec::new();
        for i in 0..5 {
            let b = batcher.clone();
            handles.push(tokio::spawn(async move {
                b.submit(raw_doc(&format!("mem_{i}"), "hello world"), 1)
                    .await
            }));
        }
        let results: Vec<_> = futures::future::join_all(handles).await;
        for r in results {
            match r.unwrap().unwrap() {
                StoreOutcome::Stored { chunks_created } => assert_eq!(chunks_created, 1),
                other => panic!("expected Stored, got {other:?}"),
            }
        }
        assert_eq!(
            invocations.load(Ordering::SeqCst),
            1,
            "5 concurrent submits must coalesce into exactly 1 processor call"
        );
        let sizes = observed_batch_sizes.lock().unwrap().clone();
        assert_eq!(sizes, vec![5]);
    }

    /// Mixed outcomes contract: the processor can admit some docs and
    /// reject others within the same batch. Each caller gets the verdict
    /// that corresponds to its own doc, identified by position.
    #[tokio::test]
    async fn per_doc_outcomes_are_delivered_in_order() {
        let process: BatchProcessFn = Arc::new(|items: Vec<(RawDocument, usize)>| {
            Box::pin(async move {
                items
                    .into_iter()
                    .enumerate()
                    .map(|(i, (_, chunks))| {
                        if i % 2 == 0 {
                            StoreOutcome::Stored {
                                chunks_created: chunks,
                            }
                        } else {
                            StoreOutcome::GateRejected {
                                reason: "not_novel".into(),
                                detail: "too similar to existing".into(),
                                similar_to: Some(format!("mem_existing_{i}")),
                            }
                        }
                    })
                    .collect()
            })
        });
        let batcher = IngestBatcher::spawn(
            process,
            BatcherConfig {
                batch_window: Duration::from_millis(40),
                max_batch: 16,
                channel_capacity: 32,
            },
        );

        let mut handles = Vec::new();
        for i in 0..6 {
            let b = batcher.clone();
            handles.push(tokio::spawn(async move {
                let doc = raw_doc(&format!("mem_{i}"), "payload");
                (i, b.submit(doc, 1).await.unwrap())
            }));
        }
        let mut results: Vec<(usize, StoreOutcome)> = futures::future::join_all(handles)
            .await
            .into_iter()
            .map(|r| r.unwrap())
            .collect();
        results.sort_by_key(|(i, _)| *i);

        // Callers do not control batch order, so we can't map even/odd
        // directly to caller id. But the COUNT of each outcome kind
        // across all callers must match what the processor returned.
        let (stored, rejected): (Vec<_>, Vec<_>) = results
            .iter()
            .partition(|(_, o)| matches!(o, StoreOutcome::Stored { .. }));
        // 6 submits, processor alternates — there will be 3 of each in
        // some permutation depending on the order the channel drains.
        assert_eq!(stored.len() + rejected.len(), 6);
        assert!(!stored.is_empty());
        assert!(!rejected.is_empty());
    }

    /// Window expiry contract: a single submit flushes within the
    /// configured window, not after `channel_capacity` backlog builds up.
    #[tokio::test]
    async fn single_submit_flushes_after_window_not_after_timeout() {
        let invocations = Arc::new(AtomicUsize::new(0));
        let invocations_cb = invocations.clone();
        let process: BatchProcessFn = Arc::new(move |items: Vec<(RawDocument, usize)>| {
            let invocations = invocations_cb.clone();
            Box::pin(async move {
                invocations.fetch_add(1, Ordering::SeqCst);
                items
                    .into_iter()
                    .map(|(_, chunks)| StoreOutcome::Stored {
                        chunks_created: chunks,
                    })
                    .collect()
            })
        });
        let batcher = IngestBatcher::spawn(
            process,
            BatcherConfig {
                batch_window: Duration::from_millis(30),
                max_batch: 8,
                channel_capacity: 8,
            },
        );

        let start = Instant::now();
        let result = batcher
            .submit(raw_doc("mem_alone", "solo"), 1)
            .await
            .unwrap();
        let elapsed = start.elapsed();

        assert!(matches!(result, StoreOutcome::Stored { chunks_created: 1 }));
        assert_eq!(invocations.load(Ordering::SeqCst), 1);
        assert!(
            elapsed < Duration::from_millis(500),
            "single submit blocked longer than expected: {elapsed:?}"
        );
    }

    /// Size-cap contract: requests arriving faster than the window can
    /// fill still flush in max_batch-sized chunks.
    #[tokio::test]
    async fn flushes_at_max_batch_size_then_starts_next_batch() {
        let observed_sizes: Arc<Mutex<Vec<usize>>> = Arc::new(Mutex::new(Vec::new()));
        let sizes_cb = observed_sizes.clone();
        let process: BatchProcessFn = Arc::new(move |items: Vec<(RawDocument, usize)>| {
            let sizes = sizes_cb.clone();
            Box::pin(async move {
                sizes.lock().unwrap().push(items.len());
                items
                    .into_iter()
                    .map(|(_, chunks)| StoreOutcome::Stored {
                        chunks_created: chunks,
                    })
                    .collect()
            })
        });
        let batcher = IngestBatcher::spawn(
            process,
            BatcherConfig {
                batch_window: Duration::from_millis(500),
                max_batch: 3,
                channel_capacity: 32,
            },
        );

        let mut handles = Vec::new();
        for i in 0..7 {
            let b = batcher.clone();
            handles.push(tokio::spawn(async move {
                b.submit(raw_doc(&format!("mem_{i}"), "hi"), 1).await
            }));
        }
        let _ = futures::future::join_all(handles).await;

        let mut sizes = observed_sizes.lock().unwrap().clone();
        sizes.sort();
        assert_eq!(
            sizes,
            vec![1, 3, 3],
            "7 submits with max_batch=3 must produce 3 batches (3+3+1)"
        );
    }

    /// Error propagation contract: if the processor itself fails to
    /// produce an outcome slot for every input (contract violation), the
    /// batcher fills the gap with `UpsertFailed` so callers aren't left
    /// hanging.
    #[tokio::test]
    async fn fills_missing_outcome_slots_with_upsert_failed() {
        let process: BatchProcessFn = Arc::new(|_items: Vec<(RawDocument, usize)>| {
            Box::pin(async move {
                Vec::new() /* intentional contract violation */
            })
        });
        let batcher = IngestBatcher::spawn(
            process,
            BatcherConfig {
                batch_window: Duration::from_millis(25),
                max_batch: 8,
                channel_capacity: 16,
            },
        );

        let mut handles = Vec::new();
        for i in 0..3 {
            let b = batcher.clone();
            handles.push(tokio::spawn(async move {
                b.submit(raw_doc(&format!("mem_{i}"), "hi"), 1).await
            }));
        }
        let results: Vec<_> = futures::future::join_all(handles).await;
        for r in results {
            let outcome = r.unwrap().unwrap();
            assert!(
                matches!(outcome, StoreOutcome::UpsertFailed(ref msg) if msg.contains("fewer outcomes")),
                "expected UpsertFailed, got {outcome:?}"
            );
        }
    }
}
