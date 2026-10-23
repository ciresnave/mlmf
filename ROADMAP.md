# mlmf roadmap

Baseline findings from the portfolio-wide CodeRipper sweep (PR #130, first
run on `origin/main` at `c5ae1c2f`, 2026-10-23, run
https://github.com/ciresnave/mlmf/actions/runs/40008808856). Advisory-only
and non-blocking (CireSnave, 2026-10-23); nothing here gates a merge. One row
per failing rule — see that run's log for the full per-finding list.

A second sweep (after `ciresnave/coderipper`#127 repinned `portfolio.yml` to
sweep a cargo workspace member by member, 2026-10-23, run
https://github.com/ciresnave/mlmf/actions/runs/40024100747, on PR #132's
branch) surfaced `reachability` findings it could not see before (added as
its own row below) and confirmed the `lint-suppression` and most of the
`licence` fixes landed: "12 allow and 0 ignore attribute(s) carry a reason
and pass", "5 of 169 source files carry no SPDX-License-Identifier" (down
from 169 of 169).

| Rule | Findings | Status | Notes |
|---|---|---|---|
| `SUP-004` (file/dependency reach outside its own boundary) | 11 | open | 2 git deps (`candlelight`, `fuel-formats`) and 9 `../../` test-file reads outside their crate. |
| `lint-suppression` (allow/ignore with no reason) | 14 (corrected from an earlier miscount of 12 — recounted directly off the sweep log) | DONE | 12 of 14 got a `reason = "..."`; the other 2 (`src/validation.rs`, both `unreachable_patterns`) had their `#[allow(...)]` removed outright after `cargo check` with the attribute removed showed no warning either way — the suppression was already unnecessary, so a fabricated reason would have been worse than deleting it. Confirmed by the second sweep: 12 passing, 0 failing. |
| `licence` | 2 | SPDX done, copyright wording open | SPDX: every git-tracked `.rs` file (165, via `git ls-files`) plus the 3 tracked `.py` files and 2 tracked `.sh` files (CodeRipper's "source file" count of 169 includes those 5, which `git ls-files '*.rs'` alone does not) now carry an SPDX comment. The second sweep, run before the `.py`/`.sh` fix was pushed, still showed "5 of 169" missing — exactly those 5; not yet re-confirmed at 0. Copyright-holder wording in `LICENSE-MIT` waits on board 263 (CireSnave). |
| `reachability` (no caller found within its own crate) | 73 across `mlmf-awq` (23), `mlmf-gptq` (23), `mlmf-hf-tokenizer` (18), `mlmf-exl2` (4), `mlmf-core` (3), `mlmf-gguf` (1), `mlmf-safetensors` (1) | open, not started | Only visible once the sweep started analysing member crates individually (`ciresnave/coderipper`#127); each one needs a judgement call (genuinely dead vs. reachable only from a caller outside the crate the per-crate analysis can't see, e.g. a doctest, another workspace member, or a planned call site) rather than a mechanical fix — out of scope for this "cheap rows first" pass. |

## Known infrastructure defect in the same run (not a code finding)

The sweep's `reachability`, `unused-return-values`, and
`unused-parameters` checks fail to run on the root `mlmf` crate specifically
(not failed *with* findings — failed to run at all, confirmed still true in
the second, per-member-sweeping run): its `build.rs` needs `protoc`, which
the sweep's sandbox (`ciresnave/coderipper/.github/workflows/portfolio.yml`)
does not install. This repo's own `core-gates`/`root-crate-tests` jobs
install it (`.github/workflows/ci.yml`, "install protoc" step) but that step
isn't available to a reusable workflow owned by another repo. Every other
workspace member (14/14) analysed cleanly in the second run. Reported to the
PM; fix belongs in `ciresnave/coderipper`, not here.
