# M6 PR-A follow-up — migration 109 substrate inventory

Status: implementation receipt; fresh independent review, latest-main rebase,
post-rebase workspace gates, CI, and merge remain pending.

Migration 108 intentionally stopped at the five genesis objects whose Stage-0
contracts were internally coherent. The D1-D8 amendment ratified on 2026-08-02
resolved the remaining contradictions. The uncontested schema questions follow
their recorded recommendations unless this PR's independent review rejects one.

## Complete migration-109 inventory

The earlier schema sweep counted fifteen remaining storage objects. Formal
proof-reset review required one additional stable-space binding, so migration
109 now creates fourteen tables and adds two columns to the existing coverage
table:

| # | Object | Contract responsibility |
|---|---|---|
| 1 | `genesis_suppression` | repeatable retained history; at most one live row |
| 2 | `genesis_card_binding` | per-group retained card binding with explicit closure |
| 3 | `genesis_quarantine` | repeatable retained quarantine with explicit lift |
| 4 | `genesis_refresh_jobs` | row-less queue followed by atomic durable lease/retry |
| 5 | `m6_refresh_dependencies` | exact page-version/claim/root snapshot; missing roots remain observable |
| 6 | `m6_readiness` | stage/signal identity plus independent epoch/phase fence |
| 7 | `m6_readiness_soak_receipts` | separately named, epoch-bound soak evidence |
| 8 | `m6_overview_subscriptions` | retained subscription identity plus live scope/page exclusion |
| 9 | `m6_pair_stats` | decayed cells plus current raw distinct-group count |
| 10 | `m6_adjacency` | deterministic top-64 ranks plus neighbor uniqueness |
| 11 | `m6_deploy_attestation` | durable signed deployment evidence |
| 12 | `m6_genesis_provenance` | immutable genesis byte/version origin |
| 13 | `page_projection_outbox` | version-keyed pending/handed-off/complete projection intent |
| 14 | `m6_counters` | monotone per-space relevance and normalization counters |
| 15 | `genesis_coverage_state.m6_mutation_count` | monotone per-space proof that no M6 mutation has occurred |
| 16 | `genesis_coverage_state.space_id` | immutable `spaces.id` binding that survives a display-name rename |

All objects are installed inside one migration-owned `IMMEDIATE` transaction.
The tables start empty; the added counter backfills to zero. Migration 109 does
not enable genesis, schedule work, or flip any reader/writer fence.

## Load-bearing controls

- Pair qualification derives the three-group floor from
  `distinct_group_count`; the normalized digest includes that field.
- Suppression, card, and quarantine history use stable nullable liveness
  markers and partial unique indexes rather than wall-clock index predicates.
- The refresh coalescing index names only durable `leased` and `retry` states.
- Readiness stores `stage`, `signal`, `epoch`, and `phase` as independent axes;
  soak evidence and operational disable/forward-fix state have separate homes.
- Overview install rows require `space IS NULL`; every non-install row requires
  a space. Subscription identity is explicitly non-null, and detached history
  never occupies the live-scope unique key.
- Deployment evidence has a collision-safe non-null `attestation_id`; its
  required `attested_at` remains data, so two attestations in one second do not
  collide.
- Genesis provenance is non-null and immutable against update, delete, and
  replacement. Both counter homes reject decreasing updates/replacements and
  deletion, so their monotone proofs cannot be reset through alternate SQL
  statement shapes.
- Every monotone proof is bound to immutable `spaces.id`. Migration backfills
  the binding from the live name and refuses an orphan it cannot resolve;
  inserts require a live `id ↔ name` pair. A rename changes only the stored
  name while retaining the same stable ID.
- Root IDs in refresh dependencies have no cascading foreign key, so deletion
  cannot make the readiness anti-join vacuously pass.
- Space rename covers every new `space` column. Retained history refuses a
  destination collision; re-derivable destination rows are retired before the
  source rows move.

### Proof-store DML guard matrix

The two monotone proof stores are guarded by mutation mechanism, not by one
expected writer statement:

| Mutation shape | Storage guard | Causal control |
|---|---|---|
| lower `UPDATE` of the proof value | value-update trigger | direct decreasing update fails |
| lower `INSERT OR REPLACE` on the same key | pre-insert comparison with the existing value | replace attempt fails |
| `DELETE` followed by a reset insert | delete prohibition | direct delete fails |
| key-moving `UPDATE OR REPLACE` onto an occupied identity | key-update destination-collision guard | attempted space/name moves fail and both rows retain their original values |
| key-moving DML onto an unoccupied identity, followed by reset at the old key | counter name and `space_id` are immutable; the new name must resolve to the same `spaces.id` | combined space/name/value parking fails, zero reinsertion fails, and the original proof remains exact |
| NULL genesis space through insert or key update | identity insert/update guards | both NULL statement shapes fail |
| `delete_space("keep")`, then move/reset or recreate the same name | retained proof keeps its old stable ID; a different live ID cannot adopt it | real repository deletion path cannot park/reset, and same-name recreation cannot replace retained proof |
| non-colliding authorized space rename | `update_space` renames the same live `spaces.id` first, then the destination-fenced proof cascade updates only the display name | complete space-rename suite remains green |

The refused-mint review artifact remains owned by the later edge-grounding
writer lane, as its Stage-0 decision states. The owner-only `card_handle_salt`
belongs to the app/secret-store rung rather than this database migration.

## Review-round-5 gate receipt

| Gate | Result |
|---|---|
| D1/D6 relevance controls | 6 passed |
| D2/D3/J1 frontier controls | 4 passed |
| D4/D7/D8 refresh/readiness controls | 7 passed |
| D5 overview controls | initial RED 0/3 missing table; review RED 3/4; GREEN 4/4 |
| J4-J7/J10 remaining controls | initial RED 0/5 missing objects; review RED 1/5; GREEN 5/5 |
| proof-store conflict matrix | round-2 RED 0/2 key moves and RED 0/1 NULL identity; GREEN 3/3 |
| proof-store identity authorization | round-3 RED 0/2 non-colliding park/reset; GREEN 2/2 |
| stable space identity | round-4 RED 0/1 repository delete-keep park/reset; GREEN 1/1; same-name recreation GREEN |
| all M6 module tests | 91 passed |
| migration-109 integration | 3 passed |
| migration-107 through current follow-up | 1 passed |
| genesis schema/migration module | 16 passed |
| space-rename/delete closure | 13 passed |
| Rustfmt / diff whitespace | passed |
| scoped Clippy (`-D warnings`) | passed |
| rust-analyzer diagnostics on every changed Rust file | zero errors/warnings; one expected test-only inactive-code hint |
| fresh independent Sol round 5 | `CLOSED` |

Round 1 returned `FIX`: decreasing `INSERT OR REPLACE` bypassed the two
update-only monotonicity triggers; genesis provenance was mutable and nullable;
subscription identity was nullable; deployment attestations collided at
one-second resolution; one D4 control used an invalid key type; and the
state-machine status was stale. The focused REDs above failed for those exact
reasons. The current subject adds insert/delete guards around the monotone
proofs, immutable provenance guards plus non-null identity, non-null retained
subscription identity, a separate attestation identity, a numeric D4 control,
and the corrected status. No runtime writer or reader was enabled.

Round 2 resolved all round-1 findings, then found the same conflict-resolution
class through a key-moving `UPDATE OR REPLACE`: SQLite could delete an occupied
destination proof without firing Wenlan's delete trigger. Per the two-strike
rule, the response is the bounded mechanism matrix above rather than another
isolated statement patch. The new REDs reproduce destination erasure for both
proof stores; the identity-update guards refuse any occupied destination while
the existing rename fence remains the positive non-colliding path.

Round 3 resolved the occupied-destination bypass, then showed that a proof could
still be parked under an unoccupied key and recreated at zero under its old
identity. The identity authorization now freezes `m6_counters.name` and permits
a proof's `space` change only in the canonical `update_space` transaction state:
the old registered name is already absent and the new registered name exists.
The new controls combine space/name/value changes, attempt the zero reset, and
assert the original proof remains byte-for-byte exact. The existing 11-test
space-rename suite is the positive authorization control.

Round 4 showed that old-absent/new-present was not exclusive to rename:
`delete_space("keep")` creates the same state. The durable mechanism now binds
both proof homes to immutable `spaces.id`; display-name changes must resolve to
that same ID, and `m6_counters.name` is immutable. Migration backfills valid
coverage rows and fails closed on an orphan. The repository deletion RED first
reproduced the park/reset, then turned green; a same-name recreation control
proves a new `spaces.id` cannot adopt retained proof.

Round 5 returned `CLOSED`: the stable-ID binding closes delete-keep,
space-ID swap, same-name recreation, replace/delete/reset, migration replay,
and canonical rename paths within this schema-first rung. Latest-main rebase,
inventory regeneration, and the one post-rebase full workspace suite remain
pending at this receipt boundary.
