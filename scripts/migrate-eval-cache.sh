#!/usr/bin/env bash
# Migrate eval per-scenario cache + sibling JSONL caches to a
# worktree-agnostic shared location.
#
# Usage:
#   bash scripts/migrate-eval-cache.sh <source-baselines-dir> [--force]
#
# Example (from any worktree):
#   bash scripts/migrate-eval-cache.sh \
#     ~/Repos/origin/.worktrees/eval-per-scenario/app/eval/baselines
#
# Defaults destination to ~/.cache/origin-eval. Override via EVAL_BASELINES_DIR.
# By default refuses to run if destination already exists; use --force to
# overwrite (after manual review of which copy is canonical).
set -euo pipefail

SRC="${1:?Usage: $0 <source-baselines-dir> [--force]}"
FORCE=0
[[ "${2:-}" == "--force" ]] && FORCE=1

DEST="${EVAL_BASELINES_DIR:-$HOME/.cache/origin-eval}"

if [[ ! -d "$SRC/fullpipeline" ]]; then
    echo "ERROR: $SRC/fullpipeline not found"
    exit 1
fi

# Idempotency safety: refuse to silently overwrite a populated destination.
if [[ -d "$DEST/fullpipeline" ]] && [[ $FORCE -eq 0 ]]; then
    src_mtime=$(stat -f "%m" "$SRC/fullpipeline")
    dst_mtime=$(stat -f "%m" "$DEST/fullpipeline")
    echo "ERROR: $DEST/fullpipeline already exists."
    echo "  source mtime: $(date -r "$src_mtime")"
    echo "  dest mtime:   $(date -r "$dst_mtime")"
    if [[ "$src_mtime" -gt "$dst_mtime" ]]; then
        echo "  Source is NEWER -- destination may be stale."
    fi
    echo "Re-run with --force to overwrite, after manual review of which copy is canonical."
    exit 1
fi

# DEST sanity check: refuse destructive ops on dangerous paths.
# Catches accidental EVAL_BASELINES_DIR=$HOME or =/ before --force rm -rf.
if [[ -z "$DEST" ]] || [[ "$DEST" == "/" ]] || [[ "$DEST" == "$HOME" ]]; then
    echo "ERROR: refusing to operate on dangerous DEST: '$DEST'"
    echo "  EVAL_BASELINES_DIR must be a dedicated subdirectory, not / or \$HOME."
    exit 1
fi

mkdir -p "$DEST"
echo "Copying $SRC/fullpipeline -> $DEST/fullpipeline ..."
[[ $FORCE -eq 1 ]] && rm -rf "$DEST/fullpipeline"
cp -R "$SRC/fullpipeline" "$DEST/fullpipeline"

# Also migrate sibling JSONL caches (Phase 3 + judge response caches).
for jsonl in "$SRC"/fullpipeline_*_phase3_answers_batch.jsonl "$SRC"/fullpipeline_*_judgments_batch.jsonl; do
    [[ -f "$jsonl" ]] || continue
    base=$(basename "$jsonl")
    if [[ -f "$DEST/$base" ]] && [[ $FORCE -eq 0 ]]; then
        echo "WARN: $DEST/$base already exists; skipping (use --force)"
        continue
    fi
    cp "$jsonl" "$DEST/$base"
    echo "Copied: $base"
done

# Verify per-scenario DB integrity (the 1.2 GB asset that matters most).
src_count=$(find "$SRC/fullpipeline" -name "origin_memory.db" | wc -l | tr -d ' ')
dst_count=$(find "$DEST/fullpipeline" -name "origin_memory.db" | wc -l | tr -d ' ')
src_bytes=$(find "$SRC/fullpipeline" -type f -exec stat -f "%z" {} + | awk '{s+=$1} END {print s}')
dst_bytes=$(find "$DEST/fullpipeline" -type f -exec stat -f "%z" {} + | awk '{s+=$1} END {print s}')

echo ""
echo "Source: $src_count DBs, $src_bytes bytes"
echo "Dest:   $dst_count DBs, $dst_bytes bytes"

if [[ "$src_count" != "$dst_count" ]] || [[ "$src_bytes" != "$dst_bytes" ]]; then
    echo "ERROR: counts/bytes mismatch -- abort, do not delete source"
    exit 1
fi

echo ""
echo "OK. Set this in your shell:"
echo "  export EVAL_BASELINES_DIR=$DEST"
echo ""
echo "Source preserved at $SRC. Remove manually only after confirming new path works."
