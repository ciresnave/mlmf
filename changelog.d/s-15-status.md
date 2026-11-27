kind: docs: ROADMAP S-15 status says done
---
### Changed

- `ROADMAP.md` S-15: the status cell said "IN PROGRESS, 3 PRs" although all three merged (#200, #201, #202; `git log origin/main --grep S-15`). Checked in code at `origin/main`: `HfLayout::from_files`, `parse_chat_template_jinja` and `SpecialToken::text_source` exist. Docs only.
