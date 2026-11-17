kind: ci: dependency-security job (cargo-deny)
---
### Added
- `deny.toml` (template: fishstringer's) and a `dependency-security` CI job running `cargo deny check`: advisories, licences, bans, sources. Additions to the template, ruled by the PM: `CDLA-Permissive-2.0` (webpki-roots 1.0.11 via ureq 3.5.0 <- `crates/mlmf-source-hub`) and `allow-wildcard-paths = true` for the workspace's `mlmf-test-support` dev-dependency paths.
- Verified on `origin/main` 0a60a90: `cargo audit` 0 vulnerabilities (1527 advisories, 142 crates); `cargo deny check` with this file passes, duplicates warn only.
