//! Reads GGUF metadata and tensor directories.
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
//! # The tensor directory
//!
//! [`parse_tensors`] is a separate call taking a parsed
//! [`GgufMetadata`], because R1 requires that reading metadata cannot
//! fail on tensor content and a caller who never calls it cannot be
//! failed by it. It yields [`GgufTensors`], which implements
//! `mlmf_core::TensorContainer` and rebases every offset once, so a
//! consumer adds nothing to the range — and reads it through
//! [`mlmf_core::TensorContainer::tensor_bytes`], which is fallible, rather
//! than slicing, which is not.
//!
//! **This paragraph licensed `&blob[d.bytes]` until that licence was
//! withdrawn** from `mlmf_core::TensorDescriptor::bytes`, whose doc now
//! rules that a descriptor records what the file DECLARES including a range
//! the file cannot honour. The licence was survivable while it was merely
//! wrong; the fourth bullet below made it reachable, because this crate now
//! hands out a descriptor whose range is past the end of the file and
//! slicing that panics.
//!
//! Four behaviours a consumer should know before trusting the tensor list,
//! all of which report rather than refuse:
//!
//! - A tensor whose ggml type code this build cannot resolve is
//!   **omitted from the list** and named in the report.
//!   `TensorDescriptor` has no way to say "length unknown", and a
//!   fabricated length is a byte range for a tensor whose extent is
//!   genuinely unknown.
//! - A **duplicate tensor name** keeps the first record and reports
//!   the second. Taking the last would make the file's meaning depend
//!   on parse order.
//! - **Overlapping byte ranges** are reported and both tensors stay
//!   readable. Refusing the open would make one bad tensor cost the
//!   whole file.
//! - A tensor whose declared range **runs past the end of the file** is
//!   reported and **KEPT**, with the range the file declares, and fails at
//!   `tensor_bytes`. A descriptor records what the file DECLARES, so
//!   dropping it would be this crate deciding a declaration does not count.
//!
//! That list was **three** until the fourth behaviour was added and nothing
//! brought this file with it — no task's file list named `lib.rs`, so no
//! per-task review opened it. A counted list beside the thing it counts is
//! checked by nothing; this one is short enough to enumerate and is
//! enumerated.
//!
//! # What this crate will not do
//!
//! It does not interpret keys. There is no chat-template accessor, no
//! architecture detection, no config struct. Resolving
//! `tokenizer.ggml.bos_token_id` against `tokenizer.ggml.tokens` has seven
//! distinct failure modes, and getting them right needs knowledge of the
//! ecosystem that MLMF deliberately does not hold.
//!
//! # Cost of opening a file
//!
//! Measured on the reference corpus. The largest key-value block is 15.78 MB
//! across 42 keys, declaring **777,056 strings** — `ggml-vocab-gemma-4.gguf`,
//! whose `tokenizer.ggml.merges` alone holds 514,906 entries. Decoding that
//! eagerly costs roughly 26 MB of allocations, all of it to answer a
//! question about one key.
//!
//! Opening indexes the keys and decodes none of them, so the cost of an
//! open is proportional to the number of keys — at most 42 in the corpus —
//! rather than to the size of the vocabulary. `array_get` decodes one
//! element without materializing its array.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod cursor;
pub mod error;
pub mod header;
pub mod metadata;
pub mod tensors;
pub mod value;

pub use error::{GgufError, Stage};
pub use header::{Header, parse_header};
pub use metadata::GgufMetadata;
pub use tensors::{GgufTensors, parse_tensors};
pub use value::ValueType;
