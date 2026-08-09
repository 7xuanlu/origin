#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TARGET="${ROOT}/scripts/audit-ambient-rb01-backlog.sh"
SQLITE_BIN="$(command -v sqlite3)"
JQ_BIN="$(command -v jq)"
TMP_DIR="$(/usr/bin/mktemp -d /private/tmp/wenlan-rb01-audit-test.XXXXXX)"
DB="${TMP_DIR}/fixture.db"
trap '/bin/rm -rf "${TMP_DIR}"' EXIT

"${SQLITE_BIN}" "${DB}" "
PRAGMA user_version=79;
CREATE TABLE memories (
  source_id TEXT,
  source TEXT,
  chunk_index INTEGER,
  pending_revision INTEGER,
  supersedes TEXT,
  is_recap INTEGER,
  source_agent TEXT,
  title TEXT,
  content TEXT,
  version INTEGER,
  entity_id TEXT
);
CREATE TABLE pages (id TEXT, status TEXT, citations TEXT);
CREATE TABLE document_enrichment_queue (status TEXT);
CREATE TABLE enrichment_steps (
  source_id TEXT,
  step_name TEXT,
  status TEXT,
  error TEXT,
  attempts INTEGER,
  updated_at INTEGER,
  PRIMARY KEY(source_id, step_name)
);
CREATE TABLE entities (id TEXT PRIMARY KEY);
CREATE TABLE memory_entities (memory_id TEXT, entity_id TEXT);
CREATE TABLE page_evidence (
  page_id TEXT,
  source_kind TEXT,
  locator TEXT,
  title TEXT,
  linked_at INTEGER,
  link_reason TEXT
);
INSERT INTO memories VALUES
  ('a','memory',0,0,NULL,0,'codex','Alpha body','Alpha body',1,NULL),
  ('b','memory',0,1,NULL,0,'codex','Pending','Pending body',1,NULL),
  ('c','memory',0,0,NULL,0,'codex','User title','Charlie body',1,'entity-c');
INSERT INTO entities VALUES ('entity-c');
INSERT INTO pages VALUES ('page-a','active',NULL);
INSERT INTO page_evidence VALUES ('page-a','memory','a',NULL,1,'test');
INSERT INTO document_enrichment_queue VALUES ('pending');
INSERT INTO enrichment_steps VALUES ('a','entity_extract','skipped',NULL,1,1);
"

legacy="$("${TARGET}" "${DB}")"
"${JQ_BIN}" -e '
  .schema_version == 79
  and .selector_mode == "legacy_population_projection"
  and .substrate.enrichment_origin_service_class_present == false
  and .counts.memory_rows == 3
  and .counts.memory_heads == 3
  and .counts.pending_revision_heads == 1
  and .counts.document_queue == 1
  and .counts.classification_population == 2
  and .counts.page_growth_population == 2
  and .counts.title_population_upper_bound == 2
  and .counts.entity_candidates == 2
  and .counts.entity_candidates_linked == 1
  and .counts.entity_candidates_unlinked == 1
  and .counts.active_pages_missing_citations == 1
  and .counts.missing_citation_pages_with_evidence == 1
  and .exact_current_eligible.classification == null
  and .exact_current_eligible.title == null
' >/dev/null <<<"${legacy}"

"${SQLITE_BIN}" "${DB}" "
ALTER TABLE enrichment_steps ADD COLUMN input_version INTEGER;
CREATE TABLE enrichment_origin (
  source_id TEXT PRIMARY KEY,
  memory_type_explicit INTEGER,
  structured_fields_explicit INTEGER,
  space_rejected INTEGER,
  service_class INTEGER
);
UPDATE enrichment_steps
SET step_name='classify', status='ok', input_version=1
WHERE source_id='a';
PRAGMA user_version=85;
"

current="$("${TARGET}" "${DB}")"
"${JQ_BIN}" -e '
  .schema_version == 85
  and .selector_mode == "exact_eligibility_population"
  and .substrate.enrichment_origin_service_class_present == true
  and .exact_current_eligible.classification == 1
  and .exact_current_eligible.structured_extract == 1
  and .exact_current_eligible.entity == 2
  and .exact_current_eligible.title == null
  and .exact_current_eligible.page_growth == 0
' >/dev/null <<<"${current}"

if /usr/bin/grep -Eq 'm\\.content|m\\.title' "${TARGET}"; then
  echo "aggregate audit must not evaluate raw memory content or titles" >&2
  exit 1
fi
origin_joins="$(
  /usr/bin/grep -Fc 'LEFT JOIN enrichment_origin eo' "${TARGET}" || true
)"
if [[ "${origin_joins}" -lt 4 ]] ||
  ! /usr/bin/grep -Fq \
    "COUNT(*) FROM pragma_table_info('enrichment_origin')" \
    "${TARGET}"; then
  echo "exact populations must require and join the production service-class substrate" >&2
  exit 1
fi
if /usr/bin/grep -q '^scalar()' "${TARGET}" ||
  ! /usr/bin/grep -q 'PRAGMA query_only=ON; BEGIN;' "${TARGET}"; then
  echo "aggregate counts must share one read transaction" >&2
  exit 1
fi
if ! "${JQ_BIN}" -e '
  .boundary.raw_content_titles_paths_or_ids_returned == false
  and .boundary.content_or_title_predicates_evaluated == false
' >/dev/null <<<"${current}"; then
  echo "aggregate audit must publish its strict privacy boundary" >&2
  exit 1
fi

echo "RB-01 backlog audit fixtures passed"
