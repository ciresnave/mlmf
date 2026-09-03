//! Canonical key vocabulary and declared-metadata extraction.
//!
//! This crate answers **"what did this file declare?"** It never answers
//! "what does that mean?" — that is Fuel's question, and the boundary is
//! the charter rather than a style preference.
//!
//! It sits above [`mlmf_core`] and below every format crate, holding spec
//! §5's canonical key vocabulary **as data**. Knowing that GGUF spells a
//! key `tokenizer.chat_template` is a table row; knowing how to find that
//! key in a byte stream is a parser, and lives elsewhere.
#![forbid(unsafe_code)]
#![warn(missing_docs)]

/// The spec §5 canonical key vocabulary, as a bidirectional data table.
pub mod vocab;

/// Chat-template extraction, including the GGUF multi-template indirection.
pub mod template;

/// Special-token declarations: what the file said, not what to do.
pub mod tokens;
