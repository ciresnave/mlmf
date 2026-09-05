//! HuggingFace checkpoint layout.
//!
//! Answers one question from bytes the caller supplies: where does each
//! tensor live? Finding and reading the file belongs to `mlmf-source-file`;
//! spec line 90 is explicit that this crate never enumerates a directory.
//!
//! NO INTRA-DOC LINKS IN THIS COMMENT, deliberately. A link to a module
//! that does not exist yet fails `cargo doc -D warnings` -- a CI step, and
//! one `scripts/local-gates.sh` runs -- at the commit that introduces it.
//! And no `///` on the `pub mod` line below: an outer doc merges with the
//! module's own `//!` and the merged text resolves in THIS module's scope,
//! breaking the module's own links. Both measured, both cost a red commit.
#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod shards;
