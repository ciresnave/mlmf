#!/usr/bin/env bash
# C1/C2: mlmf-core's transitive dependency set is pinned.
# Regenerate deliberately with: scripts/check-deps.sh --bless
#
# The same assertion also runs as a Rust test
# (crates/mlmf-core/tests/transitive_deps.rs) so that `cargo test` alone is
# sufficient. This script stays because it owns --bless and is the CI entry
# point. The two MUST normalise identically.
#
# Three corrections against the first version, each of which let a real
# change through with this script printing "snapshot matches":
#
#   * `--target all`     — a host-only tree cannot see
#                          [target.'cfg(unix)'.dependencies], so memmap2
#                          reached every Linux and macOS consumer green.
#   * `--edges normal,build`
#                        — `normal` alone excludes build edges, which is
#                          exactly the C5 codegen vector (prost-build).
#   * versions kept      — the old `sed 's/ v[0-9].*//'` accepted a move
#                          from thiserror 2 to thiserror 1.0.30 as
#                          unchanged, and `sort -u` on names alone collapses
#                          two coexisting majors into one line, which spec
#                          §2 says must be counted separately.
#
# Comparison is line-ending agnostic: the snapshot is generated on a POSIX
# shell and checked out on a Windows host with core.autocrlf=true. (.gitattributes
# pins it to LF as well — belt and braces, because a gate that cries wolf on
# a clean checkout is the fastest way to teach a team to ignore it.)
set -euo pipefail

SNAPSHOT="crates/mlmf-core/tests/transitive-deps.snapshot"
# C1, reset from the retired placeholder of 50 to measured + 5 per spec §3.3.
# A backstop, not a target: C2 is the operative control.
CEILING=13

# mlmf-core itself is dropped: it is the root of the tree, not one of its own
# dependencies, and leaving it in would turn every lockstep version bump (C7)
# into a false "dependency set changed" — the same cry-wolf failure the
# .gitattributes rule above exists to stop.
current() {
  cargo tree -p mlmf-core --edges normal,build --no-default-features \
    --target all --prefix none \
    | sed 's/ (.*//' \
    | sed 's/[[:space:]]*$//' \
    | grep -v '^[[:space:]]*$' \
    | grep -v '^mlmf-core ' \
    | sort -u
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

if ! diff -u --strip-trailing-cr "$SNAPSHOT" <(current); then
  echo "" >&2
  echo "C2 FAILED: mlmf-core's transitive dependency set changed." >&2
  echo "If intended, run: scripts/check-deps.sh --bless" >&2
  exit 1
fi

echo "C1/C2 OK: $count transitive nodes, snapshot matches"
