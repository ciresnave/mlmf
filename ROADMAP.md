# mlmf roadmap

Baseline findings from the portfolio-wide CodeRipper sweep (PR #130, first
run on `origin/main` at `c5ae1c2f`, 2026-10-23, run
https://github.com/ciresnave/mlmf/actions/runs/40008808856). Advisory-only
and non-blocking (CireSnave, 2026-10-23); nothing here gates a merge. One row
per failing rule — see that run's log for the full per-finding list.

| Rule | Findings | Status | Notes |
|---|---|---|---|
| `SUP-004` (file/dependency reach outside its own boundary) | 11 | open | 2 git deps (`candlelight`, `fuel-formats`) and 9 `../../` test-file reads outside their crate. |
| `lint-suppression` (allow/ignore with no reason) | 14 (corrected from an earlier miscount of 12 — recounted directly off the sweep log) | DONE | 12 of 14 got a `reason = "..."`; the other 2 (`src/validation.rs`, both `unreachable_patterns`) had their `#[allow(...)]` removed outright after `cargo check` with the attribute removed showed no warning either way — the suppression was already unnecessary, so a fabricated reason would have been worse than deleting it. |
| `licence` | 2 | 1 of 2 done | SPDX: every git-tracked `.rs` file (`git ls-files`, 165 of them) now carries `// SPDX-License-Identifier: MIT OR Apache-2.0` in its first line. The sweep's own count was "169 of 169" — 4 more than `git ls-files` finds; unexplained, possibly generated/cached files its sandbox saw that aren't in this repo's tree. Copyright-holder wording in `LICENSE-MIT` waits on board 263 (CireSnave). |

## Known infrastructure defect in the same run (not a code finding)

The same sweep's `reachability`, `unused-return-values`, and
`unused-parameters` checks failed to run (not failed *with* findings — failed
to run at all): mlmf's `build.rs` needs `protoc`, which the sweep's sandbox
(`ciresnave/coderipper/.github/workflows/portfolio.yml`) does not install.
This repo's own `core-gates`/`root-crate-tests` jobs install it
(`.github/workflows/ci.yml`, "install protoc" step) but that step isn't
available to a reusable workflow owned by another repo. Reported to the PM;
fix belongs in `ciresnave/coderipper`, not here.
