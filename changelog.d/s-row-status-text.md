kind: docs: ROADMAP S-57 names merged PRs, not live branches
---
### Changed

- `ROADMAP.md` S-57: the row said slices 1, 1b/2 and 3 were "on branch" `s57-stablehlo-attrs` and `s57-resource-dense`; both merged (#252, #253) and neither branch exists on origin (`git ls-remote --heads origin "s57-*"` is empty; the same query for `s68-*` finds one branch). The row now cites the PRs. Docs only; no code changed.
