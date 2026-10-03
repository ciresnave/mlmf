//! HuggingFace checkpoint layout and declared sidecar contents.
//!
//! Answers two things from bytes the caller supplies: where does each
//! tensor live (`shards`), and what did a checkpoint's JSON sidecar files
//! declare (`generation_config`, `special_tokens_map`, `config`)? Finding
//! and reading the file belongs to `mlmf-source-file`; spec line 90 is
//! explicit that this crate never enumerates a directory.
//!
//! `generation_config`/`special_tokens_map` are standalone fixed-field
//! parsers, each naming the keys it extracts. `config` is different, by
//! ruling (board #111, 2026-10-03): `config.json`'s field set varies by
//! model family in ways that are architecture knowledge (`hidden_size` /
//! `n_embd` / `d_model` naming one hyperparameter is exactly the
//! "interpreting a model file's content" this workspace's charter puts
//! with Fuel), so `config::ConfigJson` is a `MetadataSource` over the
//! keys exactly as declared -- this crate's own `tests/
//! part_b_deferral.rs` named this "Part B" and deferred it; `config.rs`'s
//! own module doc states exactly how much of that deferred scope this
//! closes and how much it does not (a single-file `MetadataSource`, not
//! the originally-envisioned `<filename>:<key>` union across all three
//! sidecars).
//!
//! NO INTRA-DOC LINKS IN THIS COMMENT, deliberately. A link to a module
//! that does not exist yet fails `cargo doc -D warnings` -- a CI step, and
//! one `scripts/local-gates.sh` runs -- at the commit that introduces it.
//! And no `///` on the `pub mod` lines below: an outer doc merges with the
//! module's own `//!` and the merged text resolves in THIS module's scope,
//! breaking the module's own links. Both measured, both cost a red commit.
#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod config;
pub mod generation_config;
pub mod shards;
pub mod special_tokens_map;
