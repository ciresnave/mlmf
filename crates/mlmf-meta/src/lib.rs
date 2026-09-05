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

// NO `///` DOC ON THESE DECLARATIONS. Each module carries its own `//!`
// inner doc, which is what satisfies `missing_docs`. An OUTER doc here
// would be merged with that inner one and the merged text resolved in
// THIS module's scope, so `vocab`'s own `[`TABLE`]` link stops resolving
// and `cargo doc -D warnings` -- a CI step -- fails. Measured, after I
// added three such comments as an embellishment the plan did not ask for.
pub mod template;
pub mod tokens;
pub mod vocab;
