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

/// The marker a test puts in a notice it needs a human to see.
///
/// `scripts/local-gates.sh` redirects each command's stdout and captures its
/// stderr, and surfaces only lines carrying this token — otherwise every
/// `Compiling`/`Finished` line cargo writes to fd 2 would drown the run.
///
/// **It lives here, in the library, on purpose.** It was briefly a file at
/// `scripts/notice-token.txt` that tests reached with
/// `include_str!("../../../scripts/…")`. That escapes the package root, so
/// `cargo package` shipped the test source and not the file it includes —
/// caught in review. Every gated crate already depends on `mlmf-core`, so a
/// `const` here is one definition that ships with all of them and reaches
/// outside nothing. The gate script derives its value from this line.
pub const NOTICE_TOKEN: &str = "MLMF-NOTICE";
pub use shape::Shape;
pub use tensor::TensorDescriptor;
pub use traits::{ByteSource, MetadataSource, RangedSource, TensorContainer};
