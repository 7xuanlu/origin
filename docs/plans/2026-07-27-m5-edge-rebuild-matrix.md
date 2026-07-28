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
| `mentions` | page | entity | assertion | inherit | optional | existing |
| `relates` | entity | entity | assertion | inherit | optional | existing |
| `cites` | page | external | synthesis | 0 | optional | existing |
| `links` | page | page | synthesis | 0 | optional | existing |
| `supports` | **claim_revision** | **memory** | evidence | see §3 | required | D8 finalizer only |
| `attests` | **root** | **claim_revision** | assertion | 0 | required, = src | UI-presence txn only |

Every tuple not in this table is rejected. The CHECK constraints permit the
column values; a separate writer-side assignment guard rejects unlisted
combinations, because a widened CHECK alone would newly permit nonsense like
`root → external` or `claim_revision → page`.

### Legacy `supports`

`supports` already exists as an edge type. Pre-M5 `supports` rows are
`page`/`memory`-shaped and are **not** claim supports. They keep
`lineage='legacy'` or their existing lineage and are excluded from every M5
support query by the `src_kind='claim_revision'` predicate. M5 never reads a
`supports` edge whose source is not a claim revision. This is the one place
where reusing an existing type name could silently widen trust, so the
discriminator is the source kind, never the type alone.

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

Verified consequence: with the CHECK widened but the fence untouched, *every*
`supports` and `attests` edge is rejected at insert. That is the correct default
— fail-closed — and it means the fence extension is a deliberate, reviewable
act rather than an oversight that silently permits cross-space edges.

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
`acquire_provenance_root("document_ingest", …)` (`edge_grounding.rs:537`); the
human kinds otherwise appear only in a `provenance.rs` unit test. So attestation
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
| `idx_edges_supports_fwd` | `(src_id, dst_id)` WHERE `edge_type='supports' AND src_kind='claim_revision' AND valid_until IS NULL` | is this revision supported? |
| `idx_edges_supports_rev` | `(dst_id)` WHERE `edge_type='supports' AND src_kind='claim_revision' AND valid_until IS NULL` | what does this memory support? |
| `idx_edges_attests_fwd` | `(dst_id)` WHERE `edge_type='attests' AND valid_until IS NULL` | who attested this revision? |
| `idx_edges_attests_rev` | `(src_id, dst_id)` WHERE `edge_type='attests' AND valid_until IS NULL` | what has this root attested? |

All four are active-only. Support-status evaluation (§1 of the truth-state
matrix) runs per page version and must not degrade to a scan; the benchmark in
artifact 6 measures it.

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
6. Drop old, rename, recreate triggers (space fence, both twins) **before**
   stamping the new `user_version`.
7. Stamp `user_version` last.

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
| widen CHECK without extending the fence | §4 test: `supports` insert must abort |
| exempt `root` on both sides instead of source-only | cross-space `attests` accepted |
| allow `attests` where `root_id != src_id` | §5 test |
| allow a `generated` root to attest | §5 test |
| let `supports` set `grounded=1` from an ungrounded memory | §3 test |
| let the D8 finalizer write `attests` | writer-exclusivity test |
| treat legacy `supports` rows as claim supports | §2 discriminator test |
| skip the row-count or checksum verify | §7.5 test on a seeded corpus |
| stamp `user_version` before recreating triggers | §7 ordering test |
| resume from cursor without re-verifying the prefix | §7 resumability test |
