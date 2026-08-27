#!/usr/bin/env bash
# Run exactly what CI runs, by READING what CI runs.
#
# A local verification set is a claim about CI's job list, and it decays
# silently every time CI gains a job. A sibling project shipped a docs break
# after reporting full verification — fmt, clippy and both test suites green,
# and none of them build documentation. Their words: "my 'all exit 0' was
# true of the four things I ran and silent about the fifth thing CI runs."
#
# The same gap existed here: CI runs `scripts/check-deps.sh` and my standing
# local set did not include it. Nothing was missed, but nothing would have
# told me if it had been.
#
# So this does not hardcode the list. It extracts every `run:` line from the
# workflow and executes them in order. It cannot drift from CI, because it
# has no list of its own to drift.
#
# It also sets CARGO_TERM_COLOR=always, which CI sets and an interactive
# shell does not. That variable has produced a red in this repo that no plain
# local run reproduces: a dependency-snapshot gate parsed coloured
# `cargo tree` output and reported crates as added that were already in the
# snapshot.
#
# Never pipe an invocation of this script to another command. Piping makes
# the pipeline's exit status the LAST command's, and a check whose exit
# status is discarded is not a check.

set -uo pipefail

WORKFLOW="${1:-.github/workflows/ci.yml}"
[ -f "$WORKFLOW" ] || { echo "no workflow at $WORKFLOW" >&2; exit 2; }

export CARGO_TERM_COLOR=always
export RUSTDOCFLAGS="-D warnings"

mapfile -t CMDS < <(grep -oE '^\s*run: .*' "$WORKFLOW" | sed -E 's/^\s*run: //' | grep -v '^\s*$')

if [ "${#CMDS[@]}" -eq 0 ]; then
  echo "extracted ZERO commands from $WORKFLOW — refusing to report success" >&2
  echo "an empty job list is what a passing run looks like, which is the" >&2
  echo "failure this script exists to prevent" >&2
  exit 2
fi

echo "running ${#CMDS[@]} commands, extracted from $WORKFLOW"
echo

failed=0
for cmd in "${CMDS[@]}"; do
  printf '  %-64s ' "$cmd"
  if eval "$cmd" >/dev/null 2>&1; then
    echo "ok"
  else
    echo "FAILED"
    failed=$((failed + 1))
  fi
done

echo
if [ "$failed" -eq 0 ]; then
  echo "all ${#CMDS[@]} CI commands pass locally"
else
  echo "$failed of ${#CMDS[@]} CI commands FAILED"
fi
exit "$failed"
