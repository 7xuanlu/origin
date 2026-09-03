#!/usr/bin/env bash
# attest.sh — portable evidence wrapper for /verify and /prove.
#
# Runs a verification command and appends one JSON line to `.claude/attest.jsonl`,
# the ledger the weekly sweep audits (`prove/references/sweep.md` step 4: "any
# week where surfaces shipped but their smoke never ran attested is itself a
# finding"). It replaces the personal `~/.claude/bin/attest.sh`, which is a macOS
# helper outside the repo and simply does not exist on a fresh checkout or on
# Windows — where the verification contract was therefore unsatisfiable.
#
#   bash scripts/attest.sh bash scripts/smoke-cli.sh
#   bash scripts/attest.sh cargo test -p wenlan-core
#
# THE INVARIANT, the same one `scripts/lib/host-process.sh` carries:
#
#   AN UNRECORDED RUN MUST NEVER BE INDISTINGUISHABLE FROM A RECORDED ONE.
#
# A ledger that cannot be written is not an empty ledger. The sweep reads a
# missing row as "the smoke never ran"; if this wrapper exited 0 after failing to
# write, it would manufacture exactly that finding out of a green run. So the
# final status is the command's status when non-zero, and the ledger write's
# status otherwise — a passing command with an unwritable ledger exits non-zero
# and says why. The command's own stdout/stderr are never captured, redirected,
# or piped: its exit status must reach here unmodified, and its output must reach
# the terminal unchanged.
#
# Environment:
#   WENLAN_ATTEST_LEDGER  ledger path (default: <repo>/.claude/attest.jsonl)
#   WENLAN_ATTEST_SURFACE optional label recorded as "surface" (e.g. cli, mcp)

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LEDGER="${WENLAN_ATTEST_LEDGER:-$ROOT/.claude/attest.jsonl}"

if [ "$#" -eq 0 ] || [ "$1" = "--help" ] || [ "$1" = "-h" ]; then
    echo "usage: bash scripts/attest.sh <command> [args...]" >&2
    echo "       records the run to ${LEDGER}" >&2
    exit 2
fi

# JSON string escaping. Backslash and quote first, so the escapes introduced
# afterwards are not themselves re-escaped, then EVERY control character.
#
# The first version escaped only \\ \" \n \r \t on the reasoning that nothing
# else can appear in a command line or a path. That is wrong: an argument can
# carry any byte, and JSON forbids every raw U+0000-U+001F. A backspace in an
# argument produced a row that appended fine, exited 0, and then failed
# JSON.parse -- a run recorded as evidence that no reader can read, which is the
# unrecorded case wearing a green badge. `LC_ALL=C tr` maps the remaining
# controls to \uXXXX one byte at a time; the named five are already gone by then.
json_escape() {
    local s="$1"
    s="${s//\\/\\\\}"
    s="${s//\"/\\\"}"
    s="${s//$'\n'/\\n}"
    s="${s//$'\r'/\\r}"
    s="${s//$'\t'/\\t}"
    s="${s//$'\b'/\\b}"
    s="${s//$'\f'/\\f}"
    # Anything still in the control range has no short escape. Bash cannot hold
    # a NUL in a variable at all, so it can never reach this point.
    if [[ "$s" == *[$'\x01'-$'\x1f']* ]]; then
        local out="" i c
        for (( i = 0; i < ${#s}; i++ )); do
            c="${s:i:1}"
            if [[ "$c" == [$'\x01'-$'\x1f'] ]]; then
                printf -v c '\\u%04x' "'$c"
            fi
            out="$out$c"
        done
        s="$out"
    fi
    printf '%s' "$s"
}

# Fields that come from an external command are best-effort: a value that cannot
# be measured is recorded as the empty string rather than as a plausible-looking
# guess. `git` is absent from plenty of CI images, and a wrong commit in an
# evidence ledger is worse than no commit.
git_field() {
    local out=""
    command -v git >/dev/null 2>&1 || { printf ''; return 0; }
    out="$(git -C "$ROOT" "$@" 2>/dev/null)" || out=""
    printf '%s' "$out" | tr -d '\r\n'
}

case "$(uname -s 2>/dev/null || echo unknown)" in
    MINGW* | MSYS* | CYGWIN*) PLATFORM=windows ;;
    Darwin) PLATFORM=macos ;;
    Linux) PLATFORM=linux ;;
    *) PLATFORM=unknown ;;
esac

# The command as one shell-readable string, so the ledger row can be re-run.
COMMAND_STR=""
for arg in "$@"; do
    case "$arg" in
        *[!A-Za-z0-9_./:=-]*) COMMAND_STR="${COMMAND_STR:+$COMMAND_STR }'${arg//\'/\'\\\'\'}'" ;;
        *) COMMAND_STR="${COMMAND_STR:+$COMMAND_STR }$arg" ;;
    esac
done

STARTED_AT="$(date -u +%Y-%m-%dT%H:%M:%SZ 2>/dev/null || printf '')"
START_EPOCH="$(date +%s 2>/dev/null || printf '')"

# No pipe and no capture: the command owns the terminal, and its status arrives
# here intact.
#
# It DOES run in a subshell, and that parenthesis is the whole of the fix. Run
# bare, the arguments are executed by THIS shell, so any argument that is a
# shell builtin capable of ending the shell ends the wrapper instead of the
# command:
#
#   bash scripts/attest.sh exit 0    -- `exit` returns from the wrapper here,
#                                       at this line, with no ledger row
#   bash scripts/attest.sh exec true -- `exec` REPLACES the wrapper with true,
#                                       which exits 0, also with no ledger row
#
# Both exited 0 having recorded nothing. That is the one outcome this file
# exists to make impossible: `/prove`'s weekly sweep reads a missing ledger
# entry as "the smoke never ran" and calls it a finding, so silence is the
# unacceptable answer, and these two spellings produced silence behind a green
# exit status. Neither is exotic -- `exec` is how a caller avoids a process, and
# `exit` is what a generated command list ends with.
#
# The subshell confines both. `( exit 0 )` ends the subshell and yields status
# 0 here; `( exec true )` replaces the subshell and yields true's status here.
# Everything after this line still runs, and the row is still written. The
# command keeps the terminal because nothing is redirected -- a subshell is not
# a capture.
status=0
( "$@" ) || status=$?

END_EPOCH="$(date +%s 2>/dev/null || printf '')"
DURATION_S=""
# `-n` is not "is a number", and the difference is not cosmetic. A clock that
# answers something non-numeric passes the old emptiness test, and the
# arithmetic below then evaluates that word as a NAME: under `set -u` that ends
# the wrapper HERE -- after the command has run, before the row is written --
# announcing an unrecorded run as `not: unbound variable`. Found by the stubbed
# clock in scripts/attest.test.ts. A duration nobody could compute is null in
# the row; the row itself is never the thing that goes missing.
if [[ "$START_EPOCH" =~ ^[0-9]+$ ]] && [[ "$END_EPOCH" =~ ^[0-9]+$ ]]; then
    DURATION_S=$(( END_EPOCH - START_EPOCH ))
fi

row="{"
row="$row\"ts\":\"$(json_escape "$STARTED_AT")\""
row="$row,\"platform\":\"$PLATFORM\""
row="$row,\"repo\":\"$(json_escape "$ROOT")\""
row="$row,\"branch\":\"$(json_escape "$(git_field rev-parse --abbrev-ref HEAD)")\""
row="$row,\"commit\":\"$(json_escape "$(git_field rev-parse --short HEAD)")\""
row="$row,\"surface\":\"$(json_escape "${WENLAN_ATTEST_SURFACE:-}")\""
row="$row,\"command\":\"$(json_escape "$COMMAND_STR")\""
row="$row,\"status\":$status"
row="$row,\"duration_s\":${DURATION_S:-null}"
row="$row}"

# --- the append is a critical section ---------------------------------------
#
# Two attest runs against one ledger are ordinary, not exotic: `/verify` drives
# the CLI and MCP smokes side by side, and the sweep re-runs surfaces in
# parallel. The append below is not one operation but three -- write, read the
# last line back, compare -- and unlocked, every damaging interleaving is
# reachable. Measured on this host with twelve concurrent `attest.sh true`
# invocations: all twelve rows landed, and THREE workers exited 1 reporting
# "the append was truncated or interleaved", because another writer's row
# became the last line between their write and their `tail`. A passing command
# turned red, and its evidence declared lost, by a neighbour doing nothing
# wrong. The other two directions are worse and quieter: a partial write from
# one process interleaved inside another's line leaves a row nothing can parse,
# and a row verified by one writer can be appended to by another a microsecond
# later, so what was verified is not what is on disk.
#
# `flock` is absent from Git Bash on Windows, which is where the surfaces this
# wrapper exists for are run, so the lock is an atomic `mkdir`: the one
# create-or-fail primitive every filesystem here provides. `set -C` /
# `noclobber` is the other candidate and is weaker -- it is documented as
# unreliable over NFS, and Bash's own manual says so.
#
# A lock that CANNOT be taken is a loud failure and never a silent skip. An
# unlocked append is precisely the unrecorded-run risk this file exists to
# remove, so falling back to one would be the defect wearing the remedy's name.
LOCK_DIR="$LEDGER.lock"
LOCK_HELD=0
# Whole seconds. The wait is bounded so a wedged holder cannot hang a smoke
# forever; the stale age is what lets a CRASHED holder's lock be reclaimed at
# all, since a directory left by a killed process is not released by anything.
LOCK_WAIT_S="${WENLAN_ATTEST_LOCK_WAIT_S:-30}"
LOCK_STALE_S="${WENLAN_ATTEST_LOCK_STALE_S:-300}"

# ROUND 6. THE OWNER TOKEN NAMES A GENERATION, NOT A PROCESS.
#
# It used to be `$$ $stamp`: a pid and a second. Neither is a name for the
# DIRECTORY that was created, and two ABA races followed from that.
#
#   CREATION. H runs `mkdir` and is descheduled before it can stamp. A waiter
#   reaches the unstamped-lock breaker and removes the empty directory. N
#   creates it again, stamps it, verifies its own text and believes it holds
#   the lock. H resumes, writes ITS owner through the same pathname -- over
#   N's -- reads back its own text, and believes the same thing. Two writers,
#   one lock, both appending.
#
#   PID REUSE. Inside one stale window `$$ $stamp` can repeat, so a re-read
#   that is supposed to prove "the holder I measured is still the holder" can
#   be satisfied by a different run.
#
# Two things close them. The token carries a per-acquisition NONCE, so no two
# generations of the directory can ever compare equal; and the owner file is
# created under `set -C` (noclobber), so writing the stamp is a create-or-fail
# act and a holder that finds an owner already there knows the directory is not
# the one it made. See `acquire_lock`.
#
# What no `mkdir` lock can offer is a removal that tests and destroys in one
# step, so a waiter breaking a stale lock can still, in a microsecond-wide
# window, remove a directory that has just been retaken. That half is answered
# from the VICTIM's side: `lock_generation_state` lets a holder discover that
# the lock it is holding is no longer the one it took, and `LOCK_STOLEN` is how
# it says so instead of appending in silence.
LOCK_TOKEN=""
LOCK_GEN=0
LOCK_STOLEN=0
LOCK_OWNER_REMOVED=0

lock_new_token() {
    LOCK_GEN=$(( LOCK_GEN + 1 ))
    printf '%s %s %s.%s.%s' "$$" "$1" "$LOCK_GEN" "$RANDOM" "$RANDOM"
}

# Is the lock directory still holding THIS run's generation?
# exit: 0 yes, 1 no -- it is gone, or it now carries somebody else's token,
#       2 the question could not be answered.
#
# Three answers, and the third is not folded into either neighbour: a holder
# that cannot tell whose lock it is looking at must not remove it (that is the
# destruction the waiter above must not do, one step later) and must not report
# a clean release either. The errno TEXT is read in the C locale for the same
# reason `errno_says_no_such_process` in scripts/lib/host-process.sh reads it:
# `cat`'s status alone cannot tell "the file is gone" from "the file would not
# open", and only the first of those is an answer.
lock_generation_state() {
    local cur rc=0
    cur="$(LC_ALL=C LANGUAGE=C cat "$LOCK_DIR/owner" 2>&1)" || rc=$?
    if [ "$rc" -eq 0 ]; then
        [ "$cur" = "$LOCK_TOKEN" ] || return 1
        return 0
    fi
    case "$cur" in
        *"No such file or directory"*) return 1 ;;
        *) return 2 ;;
    esac
}

# 0 released, 1 the lock could NOT be shown to be gone.
#
# ROUND 5. Both removals ran with their statuses dropped and `LOCK_HELD=0` set
# unconditionally underneath them, so a lock that could not be removed -- an ACL
# change, an antivirus holding the directory open, an unexpected entry inside it
# that `rmdir` refuses -- was spelled exactly like one that was: the wrapper
# exited 0 announcing a released lock that is still on disk. Every later writer
# then waits LOCK_WAIT_S for it and refuses to record ITS run, and nothing in
# this run's output says why.
#
# `rmdir` returning 0 IS the measurement that the directory is gone. A `[ ! -d
# "$LOCK_DIR" ]` afterwards would be the weaker question, because a parent that
# cannot be searched answers it "absent" too -- the two-answer test standing in
# for a three-answer question, which is this file's whole subject.
#
# ROUND 6 adds the question that has to come FIRST: is this still our lock?
# Removing a directory that was broken under us and retaken destroys a lock we
# do not hold -- the same destruction the stale-breaker must not do, committed
# by the other party -- and reporting a clean release for a generation that is
# gone hides the fact that the append it guarded may have interleaved. So the
# three states are all spelled: ours (remove it), not ours (remove NOTHING and
# say the hold was lost), unknown (remove nothing and say it could not be told).
LOCK_RELEASE_REPORTED=0
release_lock() {
    [ "$LOCK_HELD" -eq 1 ] || return 0
    local rc=0 gen=0
    if [ "$LOCK_OWNER_REMOVED" -eq 0 ]; then
        lock_generation_state || gen=$?
        if [ "$gen" -eq 1 ]; then
            # Our generation is gone. There is nothing of ours left to remove,
            # and whatever is there now belongs to somebody else.
            LOCK_HELD=0
            LOCK_STOLEN=1
            echo "attest: the lock at $LOCK_DIR is no longer the one this run took" >&2
            echo "attest: it was broken while this run held it, so the append it was" >&2
            echo "attest: guarding may have interleaved with another writer's" >&2
            return 1
        fi
        if [ "$gen" -eq 2 ]; then
            # LOCK_HELD stays 1 so the EXIT trap asks again; nothing is removed,
            # because a lock that cannot be shown to be ours is one that may be
            # somebody else's.
            if [ "$LOCK_RELEASE_REPORTED" -eq 0 ]; then
                LOCK_RELEASE_REPORTED=1
                echo "attest: cannot tell whether the lock at $LOCK_DIR is still this" >&2
                echo "attest: run's, so it is left alone rather than removed blind" >&2
            fi
            return 1
        fi
        if rm -f "$LOCK_DIR/owner" 2>/dev/null; then
            # Recorded, because the retry from the EXIT trap must not re-ask
            # `lock_generation_state` about an owner file THIS function deleted
            # and read the absence as a stolen lock.
            LOCK_OWNER_REMOVED=1
        else
            rc=1
        fi
    fi
    rmdir "$LOCK_DIR" 2>/dev/null || rc=1
    if [ "$rc" -ne 0 ]; then
        # LOCK_HELD deliberately stays 1: the EXIT trap gets one more attempt at
        # it. The message is printed once, so the retry is quiet.
        if [ "$LOCK_RELEASE_REPORTED" -eq 0 ]; then
            LOCK_RELEASE_REPORTED=1
            echo "attest: the ledger lock at $LOCK_DIR could not be released" >&2
            echo "attest: it is still there, so the next writer waits ${LOCK_WAIT_S}s" >&2
            echo "attest: for it and only breaks it once it is ${LOCK_STALE_S}s old" >&2
        fi
        return 1
    fi
    LOCK_HELD=0
    return 0
}
# The lock outlives no failure path: every exit from here on releases it -- and
# a release that did not happen leaves the trap with a status rather than in
# silence. A handler that discarded it would put the collapse back at the
# process boundary, one step outside the function that just stopped having it.
# Only a 0 is escalated: a run that already failed keeps its own status, because
# the command's status is the more specific answer.
on_exit() {
    local st=$?
    if ! release_lock && [ "$st" -eq 0 ]; then
        st=1
    fi
    exit "$st"
}
trap on_exit EXIT

# Age of the owner stamp in $1, in seconds, on stdout. Exit 1 when the age
# cannot be established -- a stamp that is not a number, or a clock this host
# does not provide. Age that could not be measured must never be read as "old
# enough to break", or a live holder's lock gets taken out from under it and the
# interleaving comes back.
#
# The owner TEXT is passed in rather than read here, because the caller has to
# hold on to the exact bytes the age was computed from: a stale lock is only
# breakable if the holder that was measured is still the holder at the moment of
# the removal. See the re-read in `acquire_lock`.
# The stamp is FIELD TWO of `<pid> <stamp>[ <nonce>]`, not the last field: the
# token grew a third component in round 6 and `${1##* }` would have started
# reading the nonce as a time. A two-field owner written by an older run, or by
# a test fixture, still parses -- `${1#* }` is the whole tail and `%% *` is a
# no-op on it.
lock_age_s() {
    local stamp now
    stamp="${1#* }"
    stamp="${stamp%% *}"
    [[ "$stamp" =~ ^[0-9]+$ ]] || return 1
    now="$(date +%s 2>/dev/null)" || return 1
    [[ "$now" =~ ^[0-9]+$ ]] || return 1
    printf '%s' "$(( now - stamp ))"
}

# 0 acquired, 1 not. Callers treat 1 as a failed write, which it is.
acquire_lock() {
    local attempt=0 attempts=$(( LOCK_WAIT_S * 10 )) age unstamped=0 stamp
    local back read_rc owner owner_rc recheck owner_again token other other_rc
    local broke_rc
    while [ "$attempt" -lt "$attempts" ]; do
        if mkdir "$LOCK_DIR" 2>/dev/null; then
            # THE DIRECTORY ALONE IS NOT THE LOCK. A holder whose stamp is
            # missing, unreadable or nonsense is indistinguishable from a
            # crashed one, and the unstamped-lock breaker below removes such a
            # lock after about two seconds. So a holder that cannot stamp does
            # not hold anything, and setting LOCK_HELD=1 anyway is this file's
            # own defect wearing the lock's name: the append goes ahead
            # believing it is serialised while a second writer is free to take
            # the same lock and interleave with it.
            #
            # Round 15 measured two ways in:
            #   * the owner write's status was DROPPED (`>"$LOCK_DIR/owner"
            #     2>/dev/null` followed unconditionally by LOCK_HELD=1), so a
            #     failed write left a live holder permanently unstamped;
            #   * the stamp was `$(date +%s || printf 0)`, so a clock that
            #     could not be read wrote 0 -- and `now - 0` is about 1.7
            #     billion seconds, older than any stale threshold, which makes
            #     a LIVE holder's lock immediately reclaimable by anyone.
            # Both restore concurrent appends while the first holder still
            # believes it owns the lock. The clock is read and checked BEFORE
            # the directory is claimed as held, and the write is checked and
            # read back, exactly as the ledger append itself is.
            if ! stamp="$(date +%s 2>/dev/null)" || [[ ! "$stamp" =~ ^[0-9]+$ ]]; then
                rmdir "$LOCK_DIR" 2>/dev/null
                echo "attest: cannot read a usable clock, so the lock cannot be stamped" >&2
                echo "attest: an unstamped lock is one any other writer may break" >&2
                return 1
            fi
            token="$(lock_new_token "$stamp")"
            # NOCLOBBER, and this subshell is the whole of the creation-ABA fix.
            # `set -C` makes `>` a CREATE-OR-FAIL act, so the stamp cannot land
            # in a directory that already carries somebody else's owner file --
            # which is exactly the state left behind when this run's `mkdir`
            # was broken by a waiter and the directory recreated and claimed by
            # a third writer. Under the old plain `>` that write SUCCEEDED, over
            # the new holder's stamp, and the read-back below then confirmed the
            # writer's own text and returned "acquired" to both of them.
            #
            # `set -C` is scoped to the subshell so nothing else in this file
            # acquires noclobber semantics; the redirection's status is the
            # subshell's status, so it is still read the same way.
            if ! ( set -C; printf '%s\n' "$token" >"$LOCK_DIR/owner" ) 2>/dev/null; then
                # Two different failures wear this status, and only one of them
                # is ours to clean up after.
                other_rc=0
                other="$(cat "$LOCK_DIR/owner" 2>/dev/null)" || other_rc=$?
                if [ "$other_rc" -eq 0 ] && [ -n "$other" ] && [ "$other" != "$token" ]; then
                    # An owner that is not ours: the directory we created is
                    # gone and this one belongs to another writer. Touch NOTHING
                    # -- removing it here is precisely the destruction this
                    # round is closing -- and go round again, where `mkdir` is
                    # what arbitrates.
                    echo "attest: the lock at $LOCK_DIR was taken by another writer" >&2
                    echo "attest: between this run's mkdir and its stamp; not touching it" >&2
                    sleep 0.1 2>/dev/null || sleep 1
                    attempt=$(( attempt + 1 ))
                    continue
                fi
                # No readable owner: the stamp could not be written at all (a
                # full disk, a read-only parent, an `owner` that is not a file).
                # THE RESIDUAL, stated: if the directory was removed by a waiter
                # and recreated by a writer that has not stamped it yet, and our
                # write failed for an unrelated I/O reason in that same instant,
                # the `rmdir` below removes that writer's directory. It cannot
                # be told apart from our own from here, and the unstamped-lock
                # breaker is what recovers it either way.
                rm -f "$LOCK_DIR/owner" 2>/dev/null
                rmdir "$LOCK_DIR" 2>/dev/null
                echo "attest: cannot stamp the lock at $LOCK_DIR" >&2
                echo "attest: an unstamped lock is one any other writer may break" >&2
                return 1
            fi
            # ROUND 5. The read-back is a MEASUREMENT and its STATUS is half of
            # it. This was one `[ "$(cat …)" != "$$ $stamp" ]`, which compares
            # the TEXT and throws the status away: a `cat` that prints the
            # expected `PID timestamp` and then exits non-zero -- a read error
            # after the buffer was already flushed, a shim, a filesystem that
            # failed on the way out -- satisfies "not unequal", and LOCK_HELD
            # becomes 1 on the strength of a read that failed. The whole point
            # of reading the stamp back is to distinguish a stamp that is on
            # disk from one that only appeared to be written, and a discarded
            # status is exactly the half of that question that was not asked.
            read_rc=0
            back="$(cat "$LOCK_DIR/owner" 2>/dev/null)" || read_rc=$?
            if [ "$read_rc" -ne 0 ] || [ "$back" != "$token" ]; then
                rm -f "$LOCK_DIR/owner" 2>/dev/null
                rmdir "$LOCK_DIR" 2>/dev/null
                if [ "$read_rc" -ne 0 ]; then
                    echo "attest: cannot stamp the lock at $LOCK_DIR" >&2
                    echo "attest: the stamp was written and reading it back exited $read_rc," >&2
                    echo "attest: so nothing here has seen the stamp on disk" >&2
                else
                    echo "attest: cannot stamp the lock at $LOCK_DIR" >&2
                fi
                echo "attest: an unstamped lock is one any other writer may break" >&2
                return 1
            fi
            LOCK_TOKEN="$token"
            LOCK_HELD=1
            return 0
        fi
        owner_rc=0
        owner="$(cat "$LOCK_DIR/owner" 2>/dev/null)" || owner_rc=$?
        if [ "$owner_rc" -eq 0 ] && age="$(lock_age_s "$owner")"; then
            unstamped=0
            if [ "$age" -gt "$LOCK_STALE_S" ]; then
                # ROUND 5. THE AGE IS A PRE-STATE, AND A PRE-STATE IS NOT A
                # LICENCE TO DESTROY. The age above was measured from the owner
                # value read at the top of this iteration; between that read and
                # these removals the old holder can release normally and a NEW
                # one can take the lock -- `mkdir` succeeds the moment the
                # directory is gone. This waiter would then delete a lock that
                # is seconds old and being held right now, and both writers
                # would append to the ledger at once, which is the interleaving
                # the whole lock exists to remove.
                #
                # So the owner is read AGAIN, immediately before the removal,
                # and must still be the exact value whose age was measured. A
                # re-read that fails is not a match either. Since round 6 that
                # value carries a per-generation nonce, so "the same owner" is
                # a statement about the same generation of the directory and no
                # longer merely about the same pid in the same second.
                #
                # THE RESIDUAL, stated exactly, and ROUND 6 corrected the
                # previous wording, which claimed the window was "the two lines
                # below" while an `echo` sat inside it as well. The window is
                # the two removals below and the scheduling boundaries around
                # them: the diagnostic has been moved out, after the removal, so
                # nothing but `rm` and `rmdir` runs between the comparison and
                # the destruction. It is NOT closed -- `mkdir` offers no
                # primitive that tests and removes in one step, so a holder that
                # releases and is replaced inside that window still has its lock
                # broken. What round 6 adds is that the victim FINDS OUT:
                # `release_lock` asks `lock_generation_state` before it removes
                # anything, so the writer whose lock was broken here reports it
                # (LOCK_STOLEN) instead of finishing green over an append that
                # may have interleaved. What is gone from round 5 is the
                # SECONDS-wide window, which is the one an ordinary release
                # lands in.
                recheck=0
                owner_again="$(cat "$LOCK_DIR/owner" 2>/dev/null)" || recheck=$?
                if [ "$recheck" -eq 0 ] && [ "$owner_again" = "$owner" ]; then
                    # Reclaim, then go round again: `mkdir` is what decides who
                    # gets it, so two waiters breaking the same stale lock still
                    # produce exactly one holder.
                    broke_rc=0
                    rm -f "$LOCK_DIR/owner" 2>/dev/null || broke_rc=1
                    rmdir "$LOCK_DIR" 2>/dev/null || broke_rc=1
                    if [ "$broke_rc" -eq 0 ]; then
                        echo "attest: breaking a stale lock at $LOCK_DIR (held ${age}s)" >&2
                    else
                        echo "attest: a stale lock at $LOCK_DIR (held ${age}s) could NOT be" >&2
                        echo "attest: removed, so this run keeps waiting for it" >&2
                    fi
                fi
            fi
        else
            # No readable stamp. A live holder writes one the instant after it
            # creates the directory, so a stamp still absent after two seconds
            # of polling belongs to a process that died in that window -- the
            # one way an unbreakable lock could otherwise be left behind.
            unstamped=$(( unstamped + 1 ))
            if [ "$unstamped" -gt 20 ]; then
                echo "attest: breaking an unstamped lock at $LOCK_DIR" >&2
                rmdir "$LOCK_DIR" 2>/dev/null
                unstamped=0
            fi
        fi
        sleep 0.1 2>/dev/null || sleep 1
        attempt=$(( attempt + 1 ))
    done
    return 1
}

# Every step of the write is checked. `mkdir -p` on an unwritable parent, a
# ledger that is a directory, a full disk — each of these produces a run that
# happened and was not recorded, which is the one outcome this wrapper exists to
# make impossible to mistake for a clean sweep.
ledger_rc=0
ledger_dir="$(dirname "$LEDGER")"
if [[ "$row" == *[$'\x01'-$'\x1f']* ]]; then
    # A row that appends cleanly and then fails JSON.parse is an unrecorded run
    # wearing a green badge, so the row is checked before it can reach the file
    # rather than after somebody tries to read the ledger.
    echo "attest: refusing to append a row containing raw control characters" >&2
    ledger_rc=1
elif ! mkdir -p "$ledger_dir" 2>/dev/null; then
    echo "attest: cannot create ledger directory $ledger_dir" >&2
    ledger_rc=1
elif ! acquire_lock; then
    # "waited up to", not "within": acquire_lock also returns 1 the moment
    # it finds it cannot stamp the lock, and reporting that as a timeout
    # would misdescribe the one failure this message exists to explain.
    echo "attest: could not take the ledger lock at $LOCK_DIR (waited up to ${LOCK_WAIT_S}s)" >&2
    echo "attest: appending without it could interleave this row with another" >&2
    echo "attest: writer's, so the row is NOT written" >&2
    ledger_rc=1
elif ! printf '%s\n' "$row" >>"$LEDGER" 2>/dev/null; then
    echo "attest: cannot append to ledger $LEDGER" >&2
    ledger_rc=1
else
    # Read back what was written. A short write, a full disk that took part of
    # the line, or another writer interleaving all leave a ledger that looks
    # appended-to and parses as something else.
    #
    # ROUND 6. THE READ-BACK IS A MEASUREMENT AND ITS STATUS IS HALF OF IT --
    # the same finding the owner read-back got in round 5, one function away and
    # missed then. This was `[ "$(tail -n 1 …)" != "$row" ]`, which compares the
    # TEXT and drops the status: a `tail` that prints the expected row and then
    # exits non-zero -- a read error after the buffer was flushed, a shim, a
    # filesystem that failed on the way out -- satisfies "not unequal", so
    # `ledger_rc` stayed 0 and a passing command exited 0 on a verification read
    # that failed. The whole point of reading the row back is to tell a row that
    # is on disk from one that only appeared to be written.
    tail_rc=0
    last_line="$(tail -n 1 "$LEDGER" 2>/dev/null)" || tail_rc=$?
    if [ "$tail_rc" -ne 0 ]; then
        echo "attest: the row was appended to $LEDGER and reading it back exited" >&2
        echo "attest: $tail_rc, so nothing here has seen the row on disk" >&2
        ledger_rc=1
    elif [ "$last_line" != "$row" ]; then
        echo "attest: the row just written is not the last line of $LEDGER" >&2
        echo "attest: the append was truncated or interleaved" >&2
        ledger_rc=1
    fi
fi
# Held across the write AND the read-back, because the read-back is half of the
# measurement: releasing between them is the interleaving this lock removes.
lock_rc=0
release_lock || lock_rc=1

# A lock that was broken while this run held it is a statement about the ROW,
# not about the lock file: the append it was guarding may have interleaved with
# another writer's, so the row on disk is not evidence that this run's row is
# what is on disk. That is the unrecorded case, and it gets the unrecorded
# case's verdict rather than the "could not remove a directory" one below.
if [ "$LOCK_STOLEN" -eq 1 ] && [ "$ledger_rc" -eq 0 ]; then
    echo "attest: the row was written without a lock this run still held" >&2
    ledger_rc=1
fi

if [ "$ledger_rc" -ne 0 ]; then
    echo "attest: the run was NOT recorded — treat this as unverified, not as a pass" >&2
    echo "attest: command exited $status" >&2
fi

# Command status wins when it is non-zero; otherwise an unwritable ledger is the
# failure. A green command whose evidence was lost is not a green check.
if [ "$status" -ne 0 ]; then
    exit "$status"
fi
if [ "$ledger_rc" -ne 0 ]; then
    exit "$ledger_rc"
fi
# And last, a lock this run could not take off the ledger. The row IS written
# here, so this is not the unrecorded case -- it is the wrapper declining to
# exit 0 on a claim ("the lock is released") it just measured to be false. The
# EXIT trap retries the removal and would escalate a 0 for the same reason; this
# line is what makes the ordinary path say so before it gets there.
exit "$lock_rc"
