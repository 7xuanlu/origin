# M5 Stage 0 — edge assignment and rebuild matrix

Date: 2026-07-27. Binding for M5 PR-A. Implements D5 of
`2026-07-27-kg-m5-goal-prompt.md`; amends §2 of the unified-model spec.

**One edge store.** Support and attestation live in `edges`. No second support
table, no second attestation table. The cost of that decision is a CHECK-widening
rebuild, specified below.

## 1. Current shape (verified, `db.rs:8910`, migration 81)

```
src_kind, dst_kind  IN ('page','memory','entity','external')
edge_type           IN ('mentions','relates','cites','supports','links')
lineage             IN ('assertion','evidence','synthesis','legacy')
grounded            IN (0,1)
root_id             REFERENCES provenance_roots(root_id)
space               NOT NULL
```

`provenance_roots` (`db.rs:8894`) has `root_kind IN ('document_ingest',
'human_capture','human_edit_delta','generated')` and **no `space` column** — a
fact that drives §4.

## 2. Amended enumeration

Added: endpoint kinds `claim_revision`, `root`; edge type `attests`.

| Edge type | src_kind | dst_kind | lineage | grounded | root_id | Writer |
|---|---|---|---|---|---|---|
| `mentions` | page | entity | assertion \| legacy | inherit | optional | existing |
| `relates` | entity | entity | assertion \| legacy | inherit | optional | existing |
| `cites` | page | external | evidence \| synthesis \| legacy | 0 | optional | existing |
| `cites` | page | **memory** | evidence \| synthesis \| legacy | inherit | optional | existing |
| `links` | page | page | synthesis \| legacy | 0 | optional | existing |
| `supports` | **claim_revision** | **memory** | **evidence** | see §3 | required | D8 finalizer only |
| `attests` | **root** | **claim_revision** | **assertion** | 0 | required, = src | UI-presence txn only |

`cites page→memory` is **live production behavior**, not a hypothetical: the
backfill emits it (`db.rs:17937`) and both `dual_write_edge` call sites emit it
with lineage `evidence`, or `legacy` on a cross-space downgrade
(`db.rs:41004`, `db.rs:45178`). An earlier draft of this table listed only
`cites page→external`, which would have made PR-A's assignment guard reject
existing rows and break the row-for-row no-behavior-change contract. The guard
is the dangerous half of "one edge store": it must be derived from what the
tree actually writes, never from what the spec's prose enumerates.

### The lineage column had the same defect as the kind columns

Adding `cites page→memory` fixed the *tuple* and left the *lineage* set wrong,
which would have failed identically. Every live writer computes lineage from a
same-space test and downgrades to `legacy` when the spaces differ, so **every**
existing shape can appear with `legacy`:

| Writer | Shape | Lineage it emits |
|---|---|---|
| citation dual-write (`db.rs:17809`, `:41004`, `:45178`) | `cites page→memory\|external` | `evidence`, `synthesis`, or `legacy` |
| page-link dual-write (`db.rs:43861`) | `links page→page` | `synthesis`, or `legacy` |
| relation backfill (`db.rs:17858`) | `relates entity→entity` | `assertion`, or `legacy` |

A guard enumerating `cites page→memory | evidence` alone rejects the same live
rows the missing tuple would have. The enumeration must be built by reading
every `dual_write_edge` call site, not by extending the table one counterexample
at a time — which is what the previous two rounds of this artifact did.

Every tuple not in this table is rejected. The CHECK constraints permit the
column values; a separate writer-side assignment guard rejects unlisted
combinations, because a widened CHECK alone would newly permit nonsense like
`root → external` or `claim_revision → page`.

**PR-A must verify the guard against a full census of live tuples**
(`SELECT DISTINCT edge_type, src_kind, dst_kind, lineage FROM edges`) on a
migrated production-shaped database before the guard is enabled. Any tuple the
census finds and this table omits is a defect in this table.

### Legacy `supports` — the type is reserved, never written

An earlier draft of this section said pre-M5 `supports` rows are
`page`/`memory`-shaped and survive the rebuild with `lineage='legacy'`. That
claim was never verified, and it is **false**. It also contradicted the
completeness rule above: it described a surviving tuple absent from the allowed
table, so the guard would have rejected rows the migration promised to keep.

The verified fact: across **executable source** — every `.rs` and `.sql` file in
the repository — the literal `'supports'` occurs exactly once, in the
`edge_type` CHECK constraint (`db.rs:8916`). No `dual_write_edge` call site and
no backfill emits it; every production writer emits `cites`, `relates`, or
`links`. The type was reserved in the constraint and never used. (Tracked
markdown, including this file, mentions the word; the claim is about code, and
saying "the whole tree" overstated it.)

Two consequences:

- the allowed-tuple table is complete as written; `supports` appears there only
  in its M5 form, `claim_revision → memory`;
- PR-A's assignment guard rejects any `supports` row whose `src_kind` is not
  `claim_revision`, and that rejection is **not** expected to fire on migration,
  because the census should find zero pre-M5 `supports` rows.

If the §2 census (`SELECT DISTINCT edge_type, src_kind, dst_kind, lineage FROM
edges`) does find a `supports` row on a real database, that is a **discovery
that invalidates this section**, not a row to wave through: the migration halts
and this table is widened deliberately. Fail-closed, because a `supports` edge
of unknown provenance is exactly the shape M5 reads as truth.

M5 support queries additionally predicate on `src_kind='claim_revision'`, so
even a row that somehow existed could not be read as a claim support. Type name
alone never grants trust; the source kind is the discriminator.

### Writer exclusivity (D8)

- Only the model-work finalizer writes `supports`. It can never write `attests`
  and never sets `human_reviewed`.
- Only a receipt-authorized UI-presence transaction writes `attests`.
- A machine job or public/MCP caller reaching either path is a contract
  violation, enforced at the writer, not merely by convention.

## 3. Grounding rule

`grounded` is inherited, never asserted:

- `supports` is `grounded=1` **only if** the destination memory span is itself
  grounded. A support edge cannot manufacture grounding for evidence that has
  none.
- `attests` is always `grounded=0`. Human presence is provenance, not grounding.
  A human clicking approve does not make an ungrounded claim grounded — that
  conflation is precisely the D2 axis collapse this rung forbids.
- Existing types keep their current inheritance.

## 4. Space rule — the fence must be extended, and today rejects everything new

The space fence (`edges_space_fence` / `_update`, `db.rs:8993`) resolves an
endpoint's space with `CASE kind WHEN 'page' … WHEN 'memory' … WHEN 'entity' …
ELSE NULL END`, then aborts when the resolved value `IS NOT NEW.space`. Because
`IS NOT` is null-safe, an unknown kind resolves to NULL and **always aborts**.

Verified consequence: with the CHECK widened but the fence untouched, every
**non-legacy** `supports` and `attests` edge is rejected at insert.

### The legacy-lineage bypass — a fail-open lane

Both fence triggers are guarded by `WHEN NEW.lineage != 'legacy'`
(`db.rs:8995`). A row with `lineage='legacy'` **skips the trigger body
entirely** — no space check, no unknown-kind rejection. So "every new support
edge aborts" is false as stated, and the gap is not theoretical:

a `claim_revision → memory` row written with `lineage='legacy'` bypasses the
SQL fence completely. If the M5 support indexes and queries discriminate only
on `edge_type` and `src_kind` — as an earlier draft of §6 did — that row lands
directly in the **trusted** support set, and the only thing standing between it
and false `supported` is every writer being perfect. A fail-closed design must
not rest on that.

### The lineage tooth

Lineage is therefore part of the M5 contract, not incidental metadata:

- a claim support edge is **`lineage='evidence'`**, always;
- an attestation edge is **`lineage='assertion'`**, always;
- **every** M5 support/attestation query and index filters on lineage in
  addition to `edge_type` and `src_kind` (§6);
- a `claim_revision`- or `root`-endpoint row carrying `lineage='legacy'` is
  invalid by construction. Since the SQL fence cannot see it, a **separate
  CHECK constraint** on the rebuilt table enforces it — CHECKs, unlike these
  triggers, have no lineage exemption.

The CHECK is what makes this fail-closed at the storage layer rather than at
the writer. The rebuild in §7 is the one chance to add it.

Required extensions:

- `claim_revision` resolves to the space of its owning page. Claims are
  page-scoped, so this is exact.
- `root` has no space (`provenance_roots` carries none). Roots are therefore
  **source-side exempt** for `attests` only, exactly mirroring how `cites` →
  `external` is destination-side exempt. The exemption is scoped to
  `edge_type='attests' AND src_kind='root'`; the destination claim_revision is
  still fenced, so a root cannot attest into a foreign space.

The existing external exemption was already narrowed once (per the `db.rs:8976`
comment, an earlier WHEN clause disabled the whole trigger body and skipped the
source check). The new exemption is written in the same narrow form for the
same reason.

## 4a. Support edges must bind the exact span and the exact verdict

D2 requires a support edge "to an exact source span"
(`2026-07-27-kg-m5-goal-prompt.md:88`). The `edges` table has **no span
columns** (`db.rs:8910`) — only `src_id`/`dst_id`. A `claim_revision → memory`
edge therefore names a *memory*, not a span within it, which is not what D2
asks for and is not enough to prove later that the thing judged is the thing
cited.

Nothing in Stage 0 previously closed this: artifact 1 §3 defines anchors on
*claim revisions*, and artifact 6 §2 puts `source_span_digest` in the
*entailment cache key* — but no column bound the **edge** to either. Three
independent records of "the evidence" with nothing forcing them to agree is the
same drift pattern this program has hit repeatedly.

The support edge's immutable payload therefore carries, and PR-A freezes:

| Field | Binds |
|---|---|
| `source_version` | the exact memory version the span was read from |
| `span_start`, `span_end` | offsets into that version |
| `span_digest` | SHA-256 of the exact span bytes |
| `model_id`, `model_version`, `prompt_version` | which judge produced this |
| `score`, `threshold_at_write` | the verdict and the bar it cleared |

Two invariants, both testable:

1. **Same-evidence invariant.** `span_digest` on the edge equals the
   `source_span_digest` component of the entailment-cache key that produced the
   verdict, and equals the digest of the bytes at `[span_start, span_end)` in
   `source_version`. If any pair disagrees, the edge is invalid and its claim is
   `provisional`.
2. **Same-verdict invariant.** The edge's `model_id`/`model_version`/
   `prompt_version`/`score` are the ones the cache recorded. A support edge may
   never be written from a verdict it cannot name.

Anchor validity (artifact 1 §3) applies here unchanged: offsets alone are never
trusted, so a span whose digest no longer matches invalidates the support edge
rather than silently re-pointing it at whatever text now occupies those offsets.

### The human-delta destination

Artifact 6 §2a's human edit delta must be reachable as a `memory` destination
with the same five span fields, because §2 permits no other destination kind for
`supports`. PR-A stores the delta as a memory with a provenance root of kind
`human_edit_delta`; the delta's span is then addressable exactly like any other
evidence span. Without that, §2a has no representable edge.

## 5. Root rule

`root_id` is required on `supports` and `attests`.

- `supports.root_id` is the provenance root of the evidence being cited.
- `attests.root_id` **equals `src_id`** — the attesting human-presence root is
  both the edge source and its provenance. A row where these differ is rejected;
  otherwise an attestation could claim one root's authority while pointing at
  another's identity.
- The attesting root's `root_kind` must be `human_capture` or
  `human_edit_delta`. A `generated` root can never attest.

**Verified gap.** Neither human root kind is minted anywhere in production. The
`CHECK` permits them (`db.rs:8898`) and the only minter in the tree is
`acquire_provenance_root("document_ingest", …)` (`edge_grounding.rs:537`).
Of the two human kinds, only `human_capture` appears anywhere else at all, in a
`provenance.rs:232` unit test; `human_edit_delta` exists **solely** as a CHECK
literal. So attestation
has **no valid source root today**. PR-A must deliver the human-root minter —
see artifact 6 §2a, which needs the same minter for the human-prose evidence
path. One missing component blocks both.

Immutable `attests` payload carries: viewed page version, revision digest,
caller/operation identity, protocol version, nonce digest, verification time.
It **never** carries the HMAC, raw capability, or install secret (D7).

## 6. Indexes

Existing `idx_edges_src (src_kind, src_id)` and `idx_edges_dst (dst_kind,
dst_id)` already cover point lookups for the new kinds. M5 adds partial indexes
for the two hot traversals:

| Index | Definition | Serves |
|---|---|---|
| `idx_edges_supports_fwd` | `(src_id, dst_id)` WHERE `edge_type='supports' AND src_kind='claim_revision' AND lineage='evidence' AND valid_until IS NULL` | is this revision supported? |
| `idx_edges_supports_rev` | `(dst_id)` WHERE `edge_type='supports' AND src_kind='claim_revision' AND lineage='evidence' AND valid_until IS NULL` | what does this memory support? |
| `idx_edges_attests_fwd` | `(dst_id)` WHERE `edge_type='attests' AND src_kind='root' AND lineage='assertion' AND valid_until IS NULL` | who attested this revision? |
| `idx_edges_attests_rev` | `(src_id, dst_id)` WHERE `edge_type='attests' AND src_kind='root' AND lineage='assertion' AND valid_until IS NULL` | what has this root attested? |

All four are active-only, and all four carry the **lineage predicate** from §4.
The index predicates and the query predicates must be written once and shared —
an index that filters lineage while a query does not would still read the
bypassed row via a scan, which defeats the tooth without failing any test that
only inspects the index definition.

Support-status evaluation (§1 of the truth-state matrix) runs per page version
and must not degrade to a scan; the benchmark in artifact 6 measures it.

## 7. Rebuild

SQLite cannot alter a CHECK constraint in place, so widening requires a table
rebuild. It runs as a guarded, replay-safe, row-for-row migration.

1. Pre-migration online backup + integrity receipt; restore drill executed.
2. Record source row count and a content checksum over the full ordered row set.
3. Create `edges_new` with the widened CHECKs and every index.
4. Copy in deterministic `edge_id` order, batched, recording cursor and
   `batch_checksum` per batch in `edges_migration_state` (the table already
   exists at `db.rs:8936` with `stage`, `cursor`, `batch_checksum`, `epoch`).
   Bump `epoch` for this rebuild.
5. Verify destination count == source count **and** destination checksum ==
   source checksum. Mismatch aborts and rolls back.
6. Drop old, rename, recreate **every** index and trigger the old table
   carried, **before** stamping the new `user_version`. See below — the set is
   larger than an earlier draft of this step claimed.
7. Stamp `user_version` last.

### Recreate the whole set, and diff it — do not trust this list

An earlier draft of step 6 read "recreate triggers (space fence, both twins)".
That names two. The live schema attaches **eight** triggers to `edges`,
verified by querying `sqlite_master` on a freshly migrated database
(`db/edges_rebuild_test.rs`):

| Trigger | Source |
|---|---|
| `edges_space_fence` | `db.rs:8996` |
| `edges_space_fence_update` | `db.rs:9021` |
| `m4_grouping_edge_insert` | `db.rs:10607` |
| `m4_grouping_edge_delete` | `db.rs:10628` |
| `m4_grouping_edge_update` | `db.rs:10649` |
| `m4_page_community_edge_insert_invalidate` | `db.rs:11099` |
| `m4_page_community_edge_delete_invalidate` | `db.rs:11114` |
| `m4_page_community_edge_update_invalidate` | `db.rs:11129` |

Following the prose would have dropped the six M4 triggers that keep community
grouping and page-community route inputs invalidated — and **nothing
downstream would fail loudly.** The tables would stay correct-looking while
their invalidation silently stopped firing, which is the worst available
failure shape: no error, no alarm, just a control plane that quietly stops
noticing edits.

Six explicit indexes come with the same problem (`idx_edges_src`,
`idx_edges_dst`, `idx_edges_root`, `idx_edges_operation`,
`idx_edges_superseded`, `idx_edges_active_grounded_space_type`), plus the four
new partial indexes from §6. `sqlite_autoindex_edges_1` is *not* on the list:
it is SQLite's implicit index for `edge_id TEXT PRIMARY KEY` and returns with
the `CREATE TABLE`.

**The rule, not the list:** the rebuild captures the attached-object set from
`sqlite_master` before the drop and asserts the post-rebuild set equals it,
plus exactly the four new indexes. The table above is a record of what was true
at authoring time, not the input to the migration. A hand-maintained list is
the same prose-number defect this program has hit repeatedly; the census is a
query.

Step 6 before step 7 is the M4 lesson applied: migration 96 committed guards
before stamping the version, which is exactly what made an interrupted upgrade
provably safe. The same ordering here means an interrupted rebuild leaves the
old schema version and never a widened-but-unfenced `edges`.

**Byte-for-byte, row-for-row.** Every existing edge survives with identical
column values. The rebuild adds no rows, drops no rows, and rewrites no payload.

**Resumability.** A restart mid-copy resumes from the recorded cursor and
re-verifies the completed prefix's checksum before continuing. A checksum
mismatch on the prefix restarts the epoch from scratch rather than trusting a
partially-written table.

**Downgrade barrier.** After the rebuild the database refuses older daemon
binaries, because an old binary would write rows the new fence expects and the
old fence cannot enforce.

## 8. Rollback

| Point of failure | Recovery |
|---|---|
| before `edges_new` created | nothing to undo |
| during copy | drop `edges_new`, clear cursor, old `edges` untouched |
| checksum mismatch at verify | drop `edges_new`, abort migration, alert |
| after rename, before triggers | **refuse to serve**; recreate triggers and re-verify |
| after `user_version` stamp | forward-only; restore from the §7.1 backup |

The "after rename, before triggers" row is the dangerous window: a widened table
with no space fence. The daemon refuses to serve rather than accepting writes
into an unfenced table — the same choice `open_for_repair` makes for the parity
guards in M4.

## 9. Mutation checks

| Weakening | Must fail |
|---|---|
| extend the fence to resolve an unknown kind as "no space" instead of aborting | §4: cross-space `supports` insert must abort |
| write a `claim_revision` support edge with `lineage='legacy'` | §4 CHECK test — the SQL fence cannot catch this one |
| drop the lineage predicate from a support index **or** its query | §6 — bypassed row must not enter the trusted set |
| omit `cites page→memory` from the assignment guard | §2 live-tuple census |
| recreate only the triggers §7 names in prose | §7 — post-rebuild `sqlite_master` trigger set must equal the pre-rebuild set |
| lose an index in the rebuild | §7 — same census, index half, plus the four new partial indexes |
| omit span/verdict fields from the support payload | §4a same-evidence invariant |
| let edge `span_digest` differ from the cache key's | §4a invariant 1 |
| write a support edge whose verdict it cannot name | §4a invariant 2 |
| exempt `root` on both sides instead of source-only | cross-space `attests` accepted |
| allow `attests` where `root_id != src_id` | §5 test |
| allow a `generated` root to attest | §5 test |
| let `supports` set `grounded=1` from an ungrounded memory | §3 test |
| let the D8 finalizer write `attests` | writer-exclusivity test |
| treat a non-`claim_revision` `supports` row as a claim support | §2 discriminator test — insert one directly (no writer emits them), assert every M5 support query excludes it |
| let the migration continue after the census finds a pre-M5 `supports` row | §2 — census halt, not a silent widen |
| skip the row-count or checksum verify | §7.5 **with fault injection** — drop and alter a row mid-copy; a clean corpus stays green either way, so the oracle must corrupt something |
| stamp `user_version` before recreating triggers | §7 ordering test |
| resume from cursor without re-verifying the prefix | §7 resumability test |
