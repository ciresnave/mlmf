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

pub mod align;
pub mod dtype;
pub mod encoding;
pub mod error;
pub mod meta;
pub mod report;
pub mod shape;
pub mod tensor;
pub mod traits;

pub use dtype::DType;
pub use encoding::{BlockSpec, Encoding};
pub use error::{Error, ErrorKind, Result};
pub use meta::MetaValue;
pub use report::{Declaration, DeclaredType, Report, Unrecognized, UnrecognizedKind};
pub use shape::Shape;
pub use tensor::TensorDescriptor;
pub use traits::{ByteSource, MetadataSource, RangedSource, TensorContainer};
