kind: CI: CodeRipper pin moved to 0.47.5
---
### Changed

- `.github/workflows/ci.yml` calls the shared portfolio workflow at coderipper `c72a6e03479d914de8ea8a73609ba73e93a0de92` (0.47.5, was `33cc3b7e` 0.43.49).
- Rule-id migration: `tools/migrate-rule-ids.py` (from coderipper `c72a6e0`) reports nothing to migrate in this repo; no `.coderipper.toml` names a renamed check.
