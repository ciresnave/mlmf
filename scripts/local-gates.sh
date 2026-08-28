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
# RUSTDOCFLAGS is a DELIBERATE INEXACTNESS and the only one here. CI sets it
# per-step, on the `cargo doc` steps alone; this script exports it once for
# every command. The direction is safe -- a broader `-D warnings` cannot let
# something through that CI would catch -- but this script's whole claim is
# that it has no list of its own to drift from CI's, and an environment it
# sets differently is a list of its own however small. Recorded rather than
# left for someone to rediscover as a divergence.
#
# Never pipe an invocation of this script to another command. Piping makes
# the pipeline's exit status the LAST command's, and a check whose exit
# status is discarded is not a check.

set -uo pipefail

WORKFLOW="${1:-.github/workflows/ci.yml}"
[ -f "$WORKFLOW" ] || { echo "no workflow at $WORKFLOW" >&2; exit 2; }

export CARGO_TERM_COLOR=always
export RUSTDOCFLAGS="-D warnings"

# REFUSE on a block scalar before extracting anything.
#
# `run: |` and `run: >` put the commands on FOLLOWING lines, which the
# extraction below cannot see. Measured, on a workflow whose only step was a
# `run: |` block holding two real gates: the extraction yields the single
# token `|`, `eval "|"` is a syntax error, and the script reports
# "1 of 1 CI commands FAILED". So it does not silently pass -- but it runs
# nothing, blames a command line reading `|`, and tells you nothing about
# the two gates it never saw.
#
# Refusing is what the script already claims to do. Before this check it
# refused only on ZERO extracted commands; a step it could not parse was
# executed as garbage instead. A published note asserted the stronger
# property, which is the defect that note is about.
if grep -qE '^[[:space:]]*run:[[:space:]]*[|>]' "$WORKFLOW"; then
  echo "$WORKFLOW uses a multi-line \`run: |\` or \`run: >\` block." >&2
  echo "This script extracts single-line \`run:\` steps only, so the commands" >&2
  echo "inside such a block are INVISIBLE to it. Refusing rather than running" >&2
  echo "a subset: a partial job list is what a passing run looks like." >&2
  echo "Teach the extraction to read block scalars, or keep CI single-line." >&2
  exit 2
fi

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
notices=""
for cmd in "${CMDS[@]}"; do
  printf '  %-64s ' "$cmd"
  # stdout to /dev/null, stderr CAPTURED -- not discarded.
  #
  # This used to be `>/dev/null 2>&1`, which threw fd 2 away for all of
  # these commands. `crates/mlmf-safetensors/tests/corpus.rs` and
  # `crates/mlmf-gguf/tests/corpus.rs` both write a SKIPPED notice to the
  # raw `std::io::Stderr` handle, specifically so it survives libtest's
  # capture of a PASSING test -- and this script, which the plan mandates as
  # the verification command, then swallowed it and printed "all N CI
  # commands pass locally". Two deliberate designs pointing opposite ways,
  # and this one was louder.
  #
  # A corpus differential that silently skips is the exact failure the
  # notice exists to prevent: the run is green, the byte-level check never
  # happened, and nothing on screen distinguishes the two.
  err=$(eval "$cmd" 2>&1 >/dev/null)
  status=$?
  if [ "$status" -eq 0 ]; then
    echo "ok"
  else
    echo "FAILED"
    failed=$((failed + 1))
    # PRINT it. Capturing stderr and then dropping it is worse than never
    # capturing: the diagnosis existed and was thrown away, and the comment
    # above said "CAPTURED -- not discarded" while this branch discarded it.
    # A failing gate whose output you cannot see is a gate you will rerun by
    # hand to find out what it said.
    if [ -n "$err" ]; then
      printf '%s
' "$err" | sed 's/^/      /' >&2
    fi
  fi
  # Surfaced whether the command passed or failed. A notice is not an error
  # and must not need one to be seen.
  case "$err" in
    *SKIPPED*) notices="${notices}${err}
" ;;
  esac
done

echo
if [ -n "$notices" ]; then
  printf '%s' "$notices" | grep -a SKIPPED
  echo
fi
if [ "$failed" -eq 0 ]; then
  echo "all ${#CMDS[@]} CI commands pass locally"
  if [ -n "$notices" ]; then
    echo "...but read the SKIPPED notice(s) above: a check that skipped is"
    echo "   not a check that passed, and this line cannot tell them apart."
  fi
else
  echo "$failed of ${#CMDS[@]} CI commands FAILED"
fi
exit "$failed"
