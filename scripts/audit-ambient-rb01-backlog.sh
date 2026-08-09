#!/bin/bash
set -euo pipefail

usage() {
  echo "usage: scripts/audit-ambient-rb01-backlog.sh <wenlan-db-path>" >&2
}

if [[ $# -ne 1 ]]; then
  usage
  exit 2
fi

DB_PATH="$1"
SQLITE_BIN="$(command -v sqlite3 || true)"
JQ_BIN="$(command -v jq || true)"
if [[ -z "${SQLITE_BIN}" || -z "${JQ_BIN}" ]]; then
  echo "sqlite3 and jq are required" >&2
  exit 1
fi
if [[ ! -f "${DB_PATH}" ]]; then
  echo "database not found: ${DB_PATH}" >&2
  exit 1
fi

# `mode=ro`, `-readonly`, and `query_only` fail closed against migrations or
# accidental writes. Do not use immutable=1: a running daemon may have current
# committed rows in the WAL.
DB_URI="file:${DB_PATH}?mode=ro"
schema_metadata="$(
  "${SQLITE_BIN}" -readonly -noheader -separator '|' "${DB_URI}" \
    "PRAGMA query_only=ON;
     SELECT
       (SELECT user_version FROM pragma_user_version),
       (SELECT COUNT(*) FROM sqlite_master
         WHERE type='table' AND name='enrichment_origin'),
       (SELECT COUNT(*) FROM pragma_table_info('enrichment_steps')
         WHERE name='input_version'),
       (SELECT COUNT(*) FROM pragma_table_info('enrichment_origin')
         WHERE name='service_class');"
)"
IFS='|' read -r schema_version origin_present input_version_present service_class_present \
  <<<"${schema_metadata}"

current_head_predicate="
  m.source='memory'
  AND m.chunk_index=0
  AND COALESCE(m.pending_revision,0)=0
  AND NOT EXISTS (
    SELECT 1 FROM memories superseder
    WHERE superseder.supersedes=m.source_id
      AND superseder.pending_revision=0
      AND superseder.source='memory'
  )"
classification_predicate="
  ${current_head_predicate}
  AND COALESCE(m.is_recap,0)=0
  AND COALESCE(m.source_agent,'')<>'folder'"
page_growth_predicate="
  ${classification_predicate}
  AND m.source_id NOT LIKE 'merged_%'"
retry_predicate="
  es.source_id IS NULL
  OR es.input_version IS NULL
  OR es.input_version<>m.version
  OR es.status='skipped'
  OR (
    es.input_version=m.version
    AND es.status IN ('failed','needs_retry')
    AND es.attempts<3
  )"

selector_mode="legacy_population_projection"
exact_classification_sql="NULL"
exact_structured_sql="NULL"
exact_entity_sql="NULL"
exact_page_growth_sql="NULL"
if [[ "${origin_present}" == "1" &&
      "${input_version_present}" == "1" &&
      "${service_class_present}" == "1" ]]; then
  # Service class affects ORDER BY, not the number of eligible rows. The
  # aggregate deliberately reports eligibility population rather than claiming
  # to reproduce which single row the production selector would choose first.
  selector_mode="exact_eligibility_population"
  exact_classification_sql="
    (SELECT COUNT(*)
       FROM memories m
       LEFT JOIN enrichment_origin eo ON eo.source_id=m.source_id
       LEFT JOIN enrichment_steps es
         ON es.source_id=m.source_id AND es.step_name='classify'
      WHERE ${classification_predicate} AND (${retry_predicate}))"
  exact_structured_sql="
    (SELECT COUNT(*)
       FROM memories m
       LEFT JOIN enrichment_origin eo ON eo.source_id=m.source_id
       INNER JOIN enrichment_steps classified
         ON classified.source_id=m.source_id
        AND classified.step_name='classify'
        AND classified.status='ok'
        AND classified.input_version=m.version
       LEFT JOIN enrichment_steps es
         ON es.source_id=m.source_id
        AND es.step_name='structured_extract'
      WHERE ${classification_predicate} AND (${retry_predicate}))"
  exact_entity_sql="
    (SELECT COUNT(*)
       FROM memories m
       LEFT JOIN enrichment_origin eo ON eo.source_id=m.source_id
       LEFT JOIN enrichment_steps es
         ON es.source_id=m.source_id AND es.step_name='entity_extract'
      WHERE ${current_head_predicate} AND (${retry_predicate}))"
  exact_page_growth_sql="
    (SELECT COUNT(*)
       FROM memories m
       LEFT JOIN enrichment_origin eo ON eo.source_id=m.source_id
       INNER JOIN enrichment_steps entity_done
         ON entity_done.source_id=m.source_id
        AND entity_done.step_name='entity_extract'
        AND entity_done.input_version=m.version
        AND entity_done.status IN ('ok','skipped','abandoned')
       LEFT JOIN enrichment_steps es
         ON es.source_id=m.source_id AND es.step_name='page_growth'
      WHERE ${page_growth_predicate} AND (${retry_predicate}))"
fi

# Every aggregate count below is evaluated by one sqlite connection inside one
# read transaction. No raw content, title, path, or identifier leaves sqlite;
# content/title columns are not evaluated even as predicates.
aggregate_row="$(
  "${SQLITE_BIN}" -readonly -noheader -separator '|' "${DB_URI}" "
    PRAGMA query_only=ON; BEGIN;
    SELECT
      (SELECT COUNT(*) FROM memories),
      (SELECT COUNT(*) FROM memories WHERE chunk_index=0),
      (SELECT COUNT(*) FROM pages WHERE status='active'),
      (SELECT COUNT(*) FROM document_enrichment_queue
        WHERE status IN ('pending','in_progress','paused')),
      (SELECT COUNT(*) FROM memories
        WHERE chunk_index=0 AND COALESCE(pending_revision,0)=1),
      (SELECT COUNT(*) FROM enrichment_steps),
      (SELECT COUNT(*) FROM pages
        WHERE status='active' AND citations IS NULL),
      (SELECT COUNT(*) FROM pages p
        WHERE p.status='active' AND p.citations IS NULL
          AND EXISTS (
            SELECT 1 FROM page_evidence pe WHERE pe.page_id=p.id
          )),
      (SELECT COUNT(*) FROM memories m WHERE ${classification_predicate}),
      (SELECT COUNT(*) FROM memories m WHERE ${page_growth_predicate}),
      (SELECT COUNT(*) FROM memories m WHERE ${current_head_predicate}),
      (SELECT COUNT(*) FROM memories m
        WHERE ${current_head_predicate}
          AND COALESCE(
            (SELECT e.id FROM entities e WHERE e.id=m.entity_id),
            (SELECT me.entity_id
             FROM memory_entities me
             JOIN entities e ON e.id=me.entity_id
             WHERE me.memory_id=m.source_id
             ORDER BY me.entity_id
             LIMIT 1)
          ) IS NOT NULL),
      (SELECT COUNT(*) FROM memories m
        WHERE ${classification_predicate} AND m.supersedes IS NOT NULL),
      ${exact_classification_sql},
      ${exact_structured_sql},
      ${exact_entity_sql},
      ${exact_page_growth_sql};
    COMMIT;"
)"
IFS='|' read -r \
  memory_rows \
  memory_heads \
  active_pages \
  document_queue \
  pending_revision_heads \
  legacy_receipts \
  missing_citations \
  missing_citations_with_evidence \
  classification_population \
  page_growth_population \
  entity_candidates \
  entity_candidates_linked \
  accepted_revision_heads \
  exact_classification \
  exact_structured \
  exact_entity \
  exact_page_growth \
  <<<"${aggregate_row}"
entity_candidates_unlinked=$((entity_candidates - entity_candidates_linked))

"${JQ_BIN}" -n \
  --arg observed_at_epoch "$(/bin/date +%s)" \
  --arg schema_version "${schema_version}" \
  --arg selector_mode "${selector_mode}" \
  --argjson enrichment_origin_present "$([[ "${origin_present}" == "1" ]] && echo true || echo false)" \
  --argjson input_version_present "$([[ "${input_version_present}" == "1" ]] && echo true || echo false)" \
  --argjson service_class_present "$([[ "${service_class_present}" == "1" ]] && echo true || echo false)" \
  --arg memory_rows "${memory_rows}" \
  --arg memory_heads "${memory_heads}" \
  --arg active_pages "${active_pages}" \
  --arg document_queue "${document_queue}" \
  --arg pending_revision_heads "${pending_revision_heads}" \
  --arg legacy_receipts "${legacy_receipts}" \
  --arg missing_citations "${missing_citations}" \
  --arg missing_citations_with_evidence "${missing_citations_with_evidence}" \
  --arg classification_population "${classification_population}" \
  --arg page_growth_population "${page_growth_population}" \
  --arg entity_candidates "${entity_candidates}" \
  --arg entity_candidates_linked "${entity_candidates_linked}" \
  --arg entity_candidates_unlinked "${entity_candidates_unlinked}" \
  --arg accepted_revision_heads "${accepted_revision_heads}" \
  --arg exact_classification "${exact_classification}" \
  --arg exact_structured "${exact_structured}" \
  --arg exact_entity "${exact_entity}" \
  --arg exact_page_growth "${exact_page_growth}" \
  '{
    observed_at_epoch: ($observed_at_epoch | tonumber),
    schema_version: ($schema_version | tonumber),
    selector_mode: $selector_mode,
    boundary: {
      read_only: true,
      one_read_transaction_for_counts: true,
      raw_content_titles_paths_or_ids_returned: false,
      content_or_title_predicates_evaluated: false,
      database_migrated: false
    },
    substrate: {
      enrichment_origin_present: $enrichment_origin_present,
      enrichment_steps_input_version_present: $input_version_present,
      enrichment_origin_service_class_present: $service_class_present
    },
    counts: {
      memory_rows: ($memory_rows | tonumber),
      memory_heads: ($memory_heads | tonumber),
      active_pages: ($active_pages | tonumber),
      document_queue: ($document_queue | tonumber),
      pending_revision_heads: ($pending_revision_heads | tonumber),
      legacy_enrichment_receipts: ($legacy_receipts | tonumber),
      active_pages_missing_citations: ($missing_citations | tonumber),
      missing_citation_pages_with_evidence: ($missing_citations_with_evidence | tonumber),
      classification_population: ($classification_population | tonumber),
      page_growth_population: ($page_growth_population | tonumber),
      title_population_upper_bound: ($classification_population | tonumber),
      entity_candidates: ($entity_candidates | tonumber),
      entity_candidates_linked: ($entity_candidates_linked | tonumber),
      entity_candidates_unlinked: ($entity_candidates_unlinked | tonumber),
      accepted_revision_heads: ($accepted_revision_heads | tonumber)
    },
    exact_current_eligible: {
      classification: (if $exact_classification == "" then null else ($exact_classification | tonumber) end),
      structured_extract: (if $exact_structured == "" then null else ($exact_structured | tonumber) end),
      entity: (if $exact_entity == "" then null else ($exact_entity | tonumber) end),
      title: null,
      page_growth: (if $exact_page_growth == "" then null else ($exact_page_growth | tonumber) end)
    }
  }'
