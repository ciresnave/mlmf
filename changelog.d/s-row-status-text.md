kind: docs: ROADMAP S-57 names merged PRs, not live branches
---
### Changed

- `ROADMAP.md` S-57: the row said slices 1, 1b/2 and 3 were "on branch" `s57-stablehlo-attrs` and `s57-resource-dense`; both merged (#252, #253) and neither branch exists on origin (`git ls-remote --heads origin "s57-*"` is empty; the same query for `s68-*` finds one branch). The row now cites the PRs. Docs only; no code changed.
- `ROADMAP.md` S-58, S-61, S-62, S-68: "this PR" and "proposed" replaced by the merged PR numbers #161, #174, #187, #255 (found by `git log origin/main --grep` on each row's subject). Other rows' "N proposed" version tags are left as written.
