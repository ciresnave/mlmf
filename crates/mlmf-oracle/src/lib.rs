// SPDX-License-Identifier: MIT OR Apache-2.0
//! Offline emitter of tensor-name mapping files.
//!
//! Some checkpoints use tensor names that `mlmf-tensor-names` does not list.
//! This crate holds the **mapping file** that records extra name pairs with
//! their evidence, and the [`Oracle`] seam through which a model-backed tool
//! proposes them. See `docs/superpowers/specs/2026-10-25-mlmf-oracle-design.md`.
//!
//! * **Never on a load path.** No crate that reads a model file depends on
//!   this one.
//! * **Names only, byte-exact.** No normalization of a name before it is
//!   matched, proposed or looked up (ROADMAP S-60, patent sweep Part E, H34).
//! * **No model-fail-then-ask loop.** An unplaced name stays absent
//!   (ROADMAP S-60). A `human` entry is a person writing a row.
//! * **A wrong guess must surface as an error or an absent value, never an
//!   invented one:** a proposal is checked against the request before it can
//!   become an entry, and a file is validated on read and on write.
//!
//! This crate does no I/O: a caller reads and writes the file's bytes and
//! supplies the prompt's SHA-256.
#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod oracle;

pub use mlmf_tensor_names::Direction;
pub use mlmf_tensor_names::mapping_file::{Entry, Evidence, EvidenceKind, FileError, MappingFile};
pub use oracle::{
    Accepted, Oracle, OracleError, Proposal, RejectReason, Rejected, Request, accept,
};
