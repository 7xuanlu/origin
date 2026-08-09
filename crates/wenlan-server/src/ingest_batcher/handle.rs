use super::{CoalescedRequest, IngestBatcher, RawDocument, ResolvedWriteSpace, StoreOutcome};
use tokio::sync::oneshot;

impl IngestBatcher {
    pub fn is_closed(&self) -> bool {
        self.tx.is_closed()
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
        self.submit_inner(doc, chunks_predicted, None).await
    }

    pub async fn submit_with_space(
        &self,
        doc: RawDocument,
        chunks_predicted: usize,
        write_space: ResolvedWriteSpace,
    ) -> Result<StoreOutcome, String> {
        self.submit_inner(doc, chunks_predicted, Some(write_space))
            .await
    }

    async fn submit_inner(
        &self,
        doc: RawDocument,
        chunks_predicted: usize,
        write_space: Option<ResolvedWriteSpace>,
    ) -> Result<StoreOutcome, String> {
        let (resp_tx, resp_rx) = oneshot::channel();
        if self
            .tx
            .send(CoalescedRequest {
                doc,
                chunks_predicted,
                write_space,
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
