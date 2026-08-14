//! Vocabulary and traits for machine learning model files.
//!
//! `mlmf-core` describes **what is in a model file**. It does not obtain
//! files: it contains no filesystem access, no memory mapping and no
//! networking. Acquisition lives on a separate axis of source crates.
//!
//! It also contains no tensor type, no device and no backend trait.
//! Consumers build their own tensors from borrowed bytes.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod error;

pub use error::{Error, ErrorKind, Result};
