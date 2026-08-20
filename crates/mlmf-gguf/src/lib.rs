//! Reads GGUF metadata.
//!
//! A GGUF file is parsed in stages — magic and version, then the key-value
//! block, then the tensor directory — and **only the stage that fails,
//! fails**. Reading `tokenizer.chat_template` out of a file full of
//! quantizations this build has never heard of is not merely supported, it
//! is structurally guaranteed: the metadata stage has no access to a type
//! table, so it cannot fail against one.
//!
//! The key-value block is **indexed** at open, not decoded. A value is
//! decoded when asked for. This is not a micro-optimization: the largest
//! file in the reference corpus declares 777,056 strings, so decoding
//! eagerly costs about 26 MB of allocations to answer a question about one
//! key.
//!
//! # What this crate will not do
//!
//! It does not interpret keys. There is no chat-template accessor, no
//! architecture detection, no config struct. Resolving
//! `tokenizer.ggml.bos_token_id` against `tokenizer.ggml.tokens` has seven
//! distinct failure modes, and getting them right needs knowledge of the
//! ecosystem that MLMF deliberately does not hold.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod cursor;
pub mod error;
pub mod header;
pub mod value;

pub use error::{GgufError, Stage};
pub use header::{Header, parse_header};
pub use value::ValueType;
