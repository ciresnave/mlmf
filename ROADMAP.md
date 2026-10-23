# mlmf roadmap

Baseline findings from the portfolio-wide CodeRipper sweep (PR #130, first
run on `origin/main` at `c5ae1c2f`, 2026-10-23, run
https://github.com/ciresnave/mlmf/actions/runs/40008808856). Advisory-only
and non-blocking (CireSnave, 2026-10-23); nothing here gates a merge. One row
per failing rule — see that run's log for the full per-finding list.

| Rule | Findings | Notes |
|---|---|---|
| `SUP-004` (file/dependency reach outside its own boundary) | 11 | 2 git deps (`candlelight`, `fuel-formats`) and 9 `../../` test-file reads outside their crate. |
| `lint-suppression` (allow/ignore with no reason) | 12 | `#[allow(...)]` attributes across src and tests with no reason string. |
| `licence` | 2 | `LICENSE-MIT` names a person as copyright holder; 169 of 169 source files carry no `SPDX-License-Identifier`. |

## Known infrastructure defect in the same run (not a code finding)

The same sweep's `reachability`, `unused-return-values`, and
`unused-parameters` checks failed to run (not failed *with* findings — failed
to run at all): mlmf's `build.rs` needs `protoc`, which the sweep's sandbox
(`ciresnave/coderipper/.github/workflows/portfolio.yml`) does not install.
This repo's own `core-gates`/`root-crate-tests` jobs install it
(`.github/workflows/ci.yml`, "install protoc" step) but that step isn't
available to a reusable workflow owned by another repo. Reported to the PM;
fix belongs in `ciresnave/coderipper`, not here.
