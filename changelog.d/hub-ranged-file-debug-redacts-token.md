kind: **Fixed**: `HubRangedFile`'s `Debug` no longer prints the bearer token
---
### Fixed

- `mlmf_source_hub::HubRangedFile` derived `Debug`, which printed its `token` (a Hugging Face bearer token) in any `{:?}` or log line. `Debug` is now written by hand and shows `Some("<redacted>")` or `None`. Found by CodeRipper 0.47.5 SEC-012 on mlmf#247. Test `debug_output_never_contains_the_bearer_token` in `crates/mlmf-source-hub/tests/ranged.rs`.
