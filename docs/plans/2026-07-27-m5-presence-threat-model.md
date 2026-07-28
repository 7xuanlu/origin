# M5 Stage 0 — UI-presence threat model and protocol

Date: 2026-07-27. Binding for M5 PR-A (nonce table) and the App PR. Implements
D7 of `2026-07-27-kg-m5-goal-prompt.md`.

## 1. What this is, and what it is not

Wenlan's daemon binds loopback and is **unauthenticated**. Any process running as
the same macOS user can reach it. This protocol does **not** change that.

| Claim | Status |
|---|---|
| a mutation marked `human_reviewed` came from a real UI gesture in the normal case | **in scope** |
| an ordinary MCP/HTTP client cannot mint a presence capability | **in scope** |
| a bug or a careless integration cannot accidentally forge presence | **in scope** |
| a replayed or reordered request cannot double-consume presence | **in scope** |
| a **hostile** process running as the same user cannot forge presence | **explicitly out of scope** |

The last row is the honest part. A hostile same-user process can read the
per-install secret from disk, because it can read anything that user can read.
Hard isolation would require an OS-specific code-signing / XPC-equivalent
design, which is out of M5 scope and is not smuggled in as a claim here.

**This is a cooperative local-client provenance boundary.** Any documentation,
UI copy, or commit message that describes it as security against a hostile local
attacker is wrong and must be corrected. Overclaiming here is worse than the gap
itself: a user who believes `human_reviewed` is tamper-proof will trust it more
than it deserves.

## 2. Threats and dispositions

| # | Threat | Disposition |
|---|---|---|
| T1 | frontend JS exfiltrates a capability | **closed** — JS never receives one (§3) |
| T2 | MCP/HTTP client mints its own capability | **closed** — owner-only secret/ACL (§5) |
| T3 | capability replayed to attest twice | **closed** — one-shot nonce, transactionally consumed (§6) |
| T4 | capability reused for a different action or target | **closed** — bound to action + target IDs + digests |
| T5 | capability captured and used later | **closed** — 60-second expiry |
| T6 | page edited between gesture and submit | **closed** — bound to exact base/revision digests |
| T7 | network retry double-applies | **closed** — receipt replay returns the stored response without re-consuming (§6) |
| T8 | same operation ID reused with different content | **closed** — conflict, not silent overwrite (§6) |
| T9 | capability leaks via logs/errors/receipts/exports | **closed** — redaction contract (§7) |
| T10 | machine job writes `attests` or `human_reviewed` | **closed** — writer exclusivity (D8, artifact 3 §2) |
| T11 | hostile same-user process reads the secret | **OUT OF SCOPE** — stated, not mitigated |
| T12 | user is socially engineered into clicking approve | out of scope; no technical control |

## 3. Protocol

1. A concrete user gesture reaches the **Tauri Rust backend**. Not a JS event
   handler — the backend.
2. The backend mints a capability, HMAC'd with the per-install secret, bound to:

   | Field | Purpose |
   |---|---|
   | action | attest / page-review; a capability for one is invalid for the other |
   | target IDs | page ID, claim revision ID |
   | base/revision digests | exact viewed content (T6) |
   | caller ID | which client |
   | operation ID | idempotency key |
   | nonce | one-shot (T3) |
   | expiry | mint time + 60s (T5) |
   | protocol version | so the format can change without ambiguity |

3. **The Tauri backend submits the mutation itself.** Frontend JavaScript never
   receives the capability (T1). This is the single most load-bearing step: a
   capability that reaches JS is a capability that reaches anything JS reaches.
4. The daemon validates, consumes the nonce inside the mutation transaction, and
   writes the `attests` edge and/or `human_reviewed`.

## 4. Ordering — replay lookup precedes validation

D7 and D8 both require this exact order, and it is easy to get backwards:

```
receipt replay / collision lookup   ← FIRST
        ↓
capability validation + nonce consumption   ← only on first execution
        ↓
mutation
```

| Case | Behavior |
|---|---|
| same caller + operation + request digest | replay the **stored response**; do **not** re-consume the nonce |
| same caller + operation, **different** digest | **conflict**; write nothing |
| new operation | validate capability, consume nonce inside the transaction, execute |

Validating first would burn the nonce on a retry of an already-applied
mutation, turning an idempotent retry into a hard failure — the client would
correctly conclude the write failed when it had actually succeeded.

## 5. Secret lifecycle

- Generated once per install, on first run.
- Stored owner-only (`0600`), outside any exported or synced path.
- Never transmitted; only HMAC outputs cross the wire.
- Rotation invalidates every outstanding capability. Since capabilities live 60
  seconds, rotation needs no drain.
- Absent or unreadable secret ⇒ presence minting is **unavailable**, and every
  presence-requiring mutation is refused. It does not degrade to "trust the
  caller."
- An ACL check gates minting to the owning client, so an ordinary MCP or HTTP
  client cannot mint even while holding a valid-looking request (T2).

## 6. Nonce table

| Column | Notes |
|---|---|
| `nonce_digest` | PRIMARY KEY — the **digest**, never the nonce |
| `caller_id`, `operation_id` | replay/collision lookup key |
| `request_digest` | distinguishes replay (T7) from collision (T8) |
| `consumed_at` | set inside the mutation transaction |
| `expires_at` | for reaping |

Consumption is a row insert **inside the same transaction** as the mutation. If
the mutation rolls back, the nonce is not consumed; if the nonce insert
conflicts, the mutation rolls back. There is no window where one succeeded and
the other did not.

Expired rows are reaped on a schedule. Reaping never resurrects a consumed
nonce's ability to be reused — a reaped consumed nonce whose capability has also
expired is unusable by expiry alone, and the reaper only removes rows already
past `expires_at`.

## 7. Redaction contract

Never logged, exported, serialized into receipts, or returned in errors:

- the HMAC,
- the raw capability,
- the install secret,
- the raw nonce.

The immutable `attests` payload may carry **only**: protocol version, nonce
**digest**, verification time, viewed page version, revision digest, and
caller/operation identity.

Error messages are deliberately coarse: `presence_invalid`, `presence_expired`,
`presence_replayed`, `presence_conflict`. They never echo the submitted
capability, and they never distinguish "bad HMAC" from "unknown nonce" in a way
that would let a caller probe the secret one byte at a time.

Enforced by a test that submits a capability containing a distinctive sentinel
string, then asserts the sentinel appears in no log line, no error body, no
receipt, and no export.

## 8. Mutation checks

Rows marked **[gate]** are human review gates, not executable tests. They are
listed because they must happen, but they are not teeth — a table that mixes the
two lets a process promise stand in for a failing build. Every unmarked row is
an executable test that goes RED under its weakening.

| Weakening | Must fail |
|---|---|
| pass the capability to frontend JS | T1 test |
| let a plain HTTP/MCP client mint | T2 test |
| accept a second use of one nonce | T3 test |
| accept a capability for a different action or target | T4 test |
| drop the expiry check | T5 test |
| drop the digest binding | T6 test |
| validate before replay lookup | T7 test — retry must not fail |
| treat a digest collision as a replay | T8 test |
| consume the nonce outside the mutation transaction | crash-injection test |
| log or return the capability | §7 sentinel test |
| let a machine job write `attests` | T10 / writer-exclusivity test |
| degrade to trusting the caller when the secret is missing | §5 test |
| describe this as hostile-same-user protection | **[gate]** §1 scope statement |
