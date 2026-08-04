# G3 measurement receipts — KG revamp close plan

Close plan: `docs/plans/2026-08-04-kg-revamp-close-plan.md` item G3.
Gate bars: `docs/plans/2026-07-25-m3g-gate-criteria.md` (Gates 1–3).
Tree measured: `feat/g2-origin-honesty` @ 23a7c011 (post-G2 daemon-authoritative origin).
Env: macOS (Darwin 25.5.0), Apple Silicon, Metal GPU; pinned entailment model
`Qwen3-4B-Instruct-2507-Q4_K_M.gguf`. Single-run receipts tagged "scaffold" per
eval citation discipline (headline claims need N≥3; these are gate pass/fail
receipts, not benchmark claims).

## 1. Live-corpus document-vs-capture mix (read-only copy, 2026-08-04)

Source: scratch copy of `~/Library/Application Support/wenlan/memorydb/origin_memory.db`
(user_version=110, pre-migration-112).

| Measure | Value |
|---|---|
| Total memories | 5,968 |
| `source_agent='folder'` (document ingest today) | 248 (4.2%) |
| `source_agent='obsidian'` (document ingest after re-sync) | 253 (4.2%) |
| Everything else (agent captures, refinery, NULL) | 5,467 (91.6%) |
| `relates` relations total | 301 |
| — with no source memory recorded (NULL/empty `source_memory_id`) | 259 |
| — sourced from agent-capture memories | 42 |
| — sourced from document-ingest memories | **0** |

**Implication for G5:** on today's live corpus the grounding sweep promotes
nothing — zero existing `relates` rows have a document-ingest source. The
groundable graph share is 0% of the current backlog; grounding coverage will
come from NEW document ingests (and from an Obsidian vault re-sync, which
reclassifies the 253 pre-112 vault notes to `document_ingest`). Flipping the
sweep ON is therefore load-free on day one; the coverage gate is proven on the
labelled fixture set below, not on live rows.

## 2. Parity sweep receipts (memory + foreground latency)

### 2.1 Page-cites parity (`bench_reconcile_parity_at_scale`, §6.5 corpus 100k/5k/100k)

**PASS, drift 0** (2026-08-04):

```
[bench §6.5] memories=100000 pages=5000 cites=100000
[bench §6.5] seed=19.8s backfill=1352.6s reconcile=1.68s gate=0.29ms
[bench §6.5] expected_active=100000 actual_active=100000 drift=0
1375.41 real   max RSS 1,093,156,864 bytes (~1.02 GiB)
```

- Recurring foreground costs: reconcile 1.68s, cutover-gate predicate 0.29ms.
- The 1352.6s is the ONE-TIME edges backfill at §6.5 scale (already applied on
  the live DB by the G1 transplant); it is not a recurring sweep cost.
- Memory ceiling: ~1.02 GiB peak RSS for the whole test process at 20× the
  live corpus size (5,968 memories live vs 100k benched).

### 2.2 Entity↔page parity (`bench_reconcile_entity_page_parity_at_scale`, 100k/5k/10k)

**PASS, drift 0** (2026-08-04):

```
[bench §6.5 entity] memories=100000 pages=5000 entities=10000
[bench §6.5 entity] reconcile=18.26s (548 entities/sec) drift=0
[bench §6.5 entity] list_entities_scoped (n=10000): off=0.036s on=0.069s
[bench §6.5 entity] get_entity_detail_scoped (n=1000): off=0.088s on=0.128s
[bench §6.5 entity] reconcile_after_perturb=18.24s drift=10   <- detector fires
749.34 real   max RSS 1,085,603,840 bytes (~1.01 GiB)
```

- Flipped (cutover-ON) reads asserted byte-identical to legacy in-test; both
  EXPLAIN plans ride indexes (`idx_entities_space`, `entity_page_map`, `pages`
  autoindexes) — no table scans.
- Recurring foreground cost: reconcile 18.26s per full pass at 10k entities
  (live corpus is far smaller); read-path overhead ~2× on list (36→69ms) and
  ~1.5× on detail (88→128ms) at bench scale, both within interactive budgets.
- Memory ceiling: ~1.01 GiB peak RSS, same shape as §2.1.

## 3. Grounding sweep — Gate 2.2 + Gate 3 (`edge_grounding_scale_and_latency_bench`)

**PASS** (2026-08-04, post-G2 tree):

```
[m3g bench] corpus: memories=100000 pages=5000 backlog=750
[m3g bench] seed=20.3s backlog_seed=0.8s
[m3g bench] ticks=31 full_ticks=30 promoted=750 scanned=779 promoted/scanned=0.963
[m3g bench] db_mutex_hold over 30 promoting ticks: p95=48.3ms max=50.2ms
```

- Gate 2.2: every tick within bounds, monotone, drained re-run promotes 0 (asserted in-test).
- Gate 3: p95 48.3ms ≤ 500ms; max 50.2ms ≤ 2s hard-fail bar. ~10× headroom under the ceiling.
- Second run (bare test binary, no compile in-process): p95 48.7ms / max 49.6ms —
  repeatable; peak RSS 388,481,024 bytes (~370 MiB) for the whole test process.

Bench fixture updated on the G2 branch: the raw-SQL §6.5 seed now stamps
`origin_class='document_ingest'`, mirroring `upsert_documents` — post-G2 the
sweep's candidate gate reads `origin_class`, failing closed on NULL, so the
pre-G2 seeding produced zero candidates (first run failed with 0 full ticks;
that failure is itself evidence the fail-closed gate works).

Structural companion (no LLM/embedding inside a transaction):
`edge_grounding::tests::sweep_holds_no_db_mutex_across_entailment` — green in
the focused suite run 2026-08-04.

## 4. Gate 1 re-run vs the G2 origin gate (real model)

**PASS — unqualified exact zero** (2026-08-04, post-G2 tree, pinned
`Qwen3-4B-Instruct-2507-Q4_K_M.gguf` on Metal / Apple M2 Pro):

```
[gate1] real-model: cases=35 entailment_calls=25 ticks=2 promoted=0
test result: ok (36.29s)   max RSS ~4.4 GiB (model weights resident)
```

Zero false-groundings across all injection classes with the candidate gate now
reading daemon-stamped `origin_class`. Matches the G4-resolution numbers
(35 cases / 25 calls / promoted=0, leak set empty) — the `HV3_H2` fixture fix
holds on the post-G2 tree.

## 5. Gate 2.1 coverage floor (real model)

**PASS — 100% recall** (2026-08-04, post-G2 tree, same pinned model/Metal):

```
[gate2] real-model: grounded=57/57 recall=100.0% promoted=57 calls=57 ticks=4 missed=[]
test result: ok (77.50s)   max RSS ~4.5 GiB (model weights resident)
```

Floor is >=80% (>=46/57); measured 57/57 with every promoted edge root-correct
(asserted in-test: real `provenance_roots` rows, `root_kind='document_ingest'`,
shared-source edges converge on one root, distinct chunks share an independence
group). The 4 backlog span-null cases (PB1-PB4, full-narration evidence) all
ground - no collateral over-rejection from the v3 prompt on the post-G2 tree.

## 6. Proposed G5 flip order (agent proposal — user confirms)

Live cutover state (from the 2026-08-04 corpus copy): `community_reader_cutover`
has `summary_buckets` and `summary_eligibility` enabled (parked overview
surfaces); `edges_reader_cutover` and `entity_reader_cutover` are empty — no
consumer flipped yet.

1. **`WENLAN_ENABLE_EDGE_GROUNDING_PROMOTE=1`** (grounding sweep ON). Safe by
   receipts: Gate 1 zero false-grounding on the pinned model, Gate 3 latency
   ~10× under ceiling, and on the live corpus the sweep has zero candidates
   today (section 1), so day-one load is nil.
2. **`scoped_entities` entity↔page reader** (`set_entity_reader_cutover`),
   gated on a clean `reconcile_entity_page_parity` watermark — the §2.2 bench
   proves parity + byte-identical flipped reads at scale.
3. **`communities` edges reader** (`set_reader_cutover`), gated on a clean
   parity watermark — last because its consumer surfaces are parked anyway.

Soak window between each flip; drift must stay zero (the weekly sweep plus the
parity gate's own predicate). Rollback per flip is the same lever back to 0.

## 7. G6 restore-drill rehearsal (2026-08-04)

G6's gate requires a pre-migration SQLite online-backup plus a rehearsed
restore drill. Rehearsed end-to-end against the live daemon:

1. **Online backup with the daemon live:** `sqlite3 <live> ".backup <dest>"`
   completed in **1.2s** for the 376 MB database, no daemon pause.
2. **Backup verified:** row counts identical to live (memories 5,968, edges
   4,932, entities 885, pages 1,084; `user_version=110`). `PRAGMA quick_check`
   flags the three libSQL vector indexes — a plain-sqlite3 tooling artifact
   (it cannot parse libSQL's vector index format), adjudicated by the next step.
3. **Restore served by the production binary** (`wenlan-server 0.15.4`),
   isolated on all three axes (`WENLAN_PORT=7879`, `WENLAN_DATA_DIR=<scratch>`,
   `config.json` `knowledge_path=<scratch>/pages`, confirmed via
   `GET /api/knowledge/path`): health ok, `files_indexed=5968`, search returns
   real content. The isolated daemon projected pages into the scratch vault —
   confirming the three-axes isolation rule is load-bearing for any future drill.
4. **Teardown clean:** drill daemon killed; the live daemon on :7878 was never
   disturbed (healthy before, during, after).

The drill validates the restore PATH. The real G6 ceremony takes a fresh
backup immediately before the retirement migration runs; this receipt does not
substitute for that backup.
