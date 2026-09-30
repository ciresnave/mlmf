//! HuggingFace checkpoint layout and declared sidecar contents.
//!
//! Answers two things from bytes the caller supplies: where does each
//! tensor live (`shards`), and what did a checkpoint's JSON sidecar files
//! declare (`generation_config`, `special_tokens_map`)? Finding and
//! reading the file belongs to `mlmf-source-file`; spec line 90 is
//! explicit that this crate never enumerates a directory. Each reader here
//! is a standalone bytes-to-structure parser for one named file — not a
//! `MetadataSource` over the general HuggingFace JSON key space, which
//! stays deferred (see `tests/part_b_deferral.rs`).
//!
//! NO INTRA-DOC LINKS IN THIS COMMENT, deliberately. A link to a module
//! that does not exist yet fails `cargo doc -D warnings` -- a CI step, and
//! one `scripts/local-gates.sh` runs -- at the commit that introduces it.
//! And no `///` on the `pub mod` lines below: an outer doc merges with the
//! module's own `//!` and the merged text resolves in THIS module's scope,
//! breaking the module's own links. Both measured, both cost a red commit.
#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod generation_config;
pub mod shards;
pub mod special_tokens_map;
