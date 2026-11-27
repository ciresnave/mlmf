kind: docs: ROADMAP S-15 and S-64 status say done
---
### Changed

- `ROADMAP.md` S-15: the status cell said "IN PROGRESS, 3 PRs" although all three merged (#200, #201, #202; `git log origin/main --grep S-15`). Checked at `origin/main`: `HfLayout::from_files`, `parse_chat_template_jinja` and `SpecialToken::text_source` exist.
- `ROADMAP.md` S-64: the status cell began "open" while its own text ends "S-64 complete"; it now says done. Checked: `list_files`, `fetch_to`, `fetch_to_with_progress` and `RemoteHash::verify` exist in `crates/mlmf-source-hub` at `origin/main`.

Docs only; no code changed.
