#!/usr/bin/env bash
# C1/C2: mlmf-core's transitive dependency set is pinned.
# Regenerate deliberately with: scripts/check-deps.sh --bless
set -euo pipefail

SNAPSHOT="crates/mlmf-core/tests/transitive-deps.snapshot"
CEILING=50

current() {
  cargo tree -p mlmf-core --edges normal --no-default-features --prefix none \
    | sed 's/ (.*//' | sed 's/ v[0-9].*//' | sort -u
}

if [[ "${1:-}" == "--bless" ]]; then
  current > "$SNAPSHOT"
  echo "blessed $(wc -l < "$SNAPSHOT") nodes into $SNAPSHOT"
  exit 0
fi

count=$(current | wc -l)
if (( count > CEILING )); then
  echo "C1 FAILED: mlmf-core has $count transitive nodes, ceiling is $CEILING" >&2
  exit 1
fi

if ! diff -u "$SNAPSHOT" <(current); then
  echo "" >&2
  echo "C2 FAILED: mlmf-core's transitive dependency set changed." >&2
  echo "If intended, run: scripts/check-deps.sh --bless" >&2
  exit 1
fi

echo "C1/C2 OK: $count transitive nodes, snapshot matches"
