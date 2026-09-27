//! The GGUF-embedded imatrix layout: parse, validate, enumerate — nothing
//! else.
//!
//! Board item 63a, ruled by CireSnave: *"ditch legacy GGML and split iMatrix
//! between MLMF and Fuel as you and I both think it should be."* MLMF takes
//! the file-format half — can we read this file and is it well-formed —
//! never what the numbers mean, how good a calibration run was, or how
//! quantization should consume them. Those stay with Fuel.
//!
//! # Only one of two wire formats
//!
//! `llama-imatrix` produces two layouts that share nothing but a name:
//! this GGUF-embedded one, and an older raw positional layout (no magic, no
//! version field) that `fuel-formats/src/imatrix.rs` already reads and
//! keeps reading — both are still in real use today (verified against real
//! Hub files, 2026-09-26/27; see the design spec's OQ-3). **This module
//! covers the GGUF-embedded layout only.** Nothing here reads the raw
//! layout, and nothing in this crate should be read as "imatrix support"
//! being complete on the strength of this module alone.
//!
//! # Why this exists despite the crate's own "no interpretation" rule
//!
//! [`crate`]'s module doc says there is no chat-template accessor and no
//! architecture detection, because resolving those needs ecosystem
//! knowledge this crate deliberately does not hold. This module is
//! different in kind, not just in size: [`read`] performs exactly two
//! mechanical checks — a literal string compare against `general.type`,
//! and pairing tensor names by a documented suffix convention
//! (`<name>.in_sum2` / `<name>.counts`) — and returns [`TensorDescriptor`]s,
//! never decoded values. It asks "does this file follow the shape its own
//! format documents", the same question [`crate::tensors`] already asks of
//! every GGUF tensor directory; it does not ask what the statistics mean.
//!
//! # What this file cannot tell you without the model
//!
//! ⚠️ **An imatrix's entry keys ARE the source model's tensor names, and
//! neither wire format carries that model's shape or dtype.** [`Imatrix`]
//! is not a standalone, checkable artifact — a tensor name here proves
//! nothing about whether it, or the model it names, still exists, has the
//! same shape, or was renamed since the calibration run. There is
//! deliberately no entry point here named anything like "read this
//! imatrix" that would suggest otherwise. A caller who needs to know
//! whether an entry still applies to a given checkpoint must open that
//! checkpoint and compare names themselves — this module has no way to do
//! that for you, and would be lying about its own scope if it tried.

use mlmf_core::{MetaValue, MetadataSource, Shape, TensorContainer, TensorDescriptor};

use crate::tensors::GgufTensors;

/// One calibration statistic pair, for one tensor name in the source model
/// the imatrix was computed against.
///
/// Both descriptors are exactly as [`crate::parse_tensors`] would hand them
/// out individually — this struct only records that the two names pair up.
/// Nothing here decodes a byte.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ImatrixEntry {
    /// The source model's tensor name, exactly as declared in this file.
    /// **Not validated against any model** — see this module's doc.
    pub tensor_name: String,
    /// Descriptor for the `<tensor_name>.in_sum2` tensor: per-input-channel
    /// sums of squared activations, as declared. Whether or how to
    /// normalize these is Fuel's half, not this crate's — this descriptor
    /// is handed out raw.
    pub in_sum2: TensorDescriptor,
    /// Descriptor for the `<tensor_name>.counts` tensor. Every file this
    /// build has read against declares this as a single-element tensor
    /// (`shape.dims() == [1]`), which [`read`] checks as a structural
    /// invariant of the file — not an interpretation of what the count
    /// means.
    pub counts: TensorDescriptor,
}

/// A GGUF-embedded imatrix file's declared structure.
///
/// See this module's doc for what this type deliberately cannot tell a
/// caller.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Imatrix {
    /// `imatrix.datasets`, as declared. Empty if the key is absent —
    /// GGUF's own `Declaration::Absent` distinction is available through
    /// [`MetadataSource::declaration`] if a caller needs to tell "absent"
    /// from "declared empty".
    pub datasets: Vec<String>,
    /// `imatrix.chunk_count`, as declared.
    pub chunk_count: Option<u32>,
    /// `imatrix.chunk_size`, as declared.
    pub chunk_size: Option<u32>,
    /// Every paired statistic this file declares. **Never empty** — [`read`]
    /// refuses a file with zero entries rather than return one (see
    /// [`ImatrixError::NoEntries`]).
    pub entries: Vec<ImatrixEntry>,
}

const IN_SUM2_SUFFIX: &str = ".in_sum2";
const COUNTS_SUFFIX: &str = ".counts";

/// Why a file that opened as GGUF is not a well-formed GGUF-embedded
/// imatrix.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum ImatrixError {
    /// `general.type` is not the string `"imatrix"` — either absent, or
    /// declared as something else. This is an ordinary GGUF file this
    /// module was asked to read as an imatrix and is not one; it says
    /// nothing about whether the file is well-formed for its own actual
    /// type.
    NotAnImatrixFile {
        /// What `general.type` actually declared, if anything.
        declared_type: Option<String>,
    },
    /// A tensor ending in `.in_sum2` has no matching `.counts` tensor, or
    /// vice versa. Names the orphan and which half is missing.
    UnpairedStatistic {
        /// The tensor name that has no partner.
        name: String,
        /// `true` if `name` is the `.in_sum2` half and `.counts` is
        /// missing; `false` if `name` is the `.counts` half and
        /// `.in_sum2` is missing.
        in_sum2_present: bool,
    },
    /// A `.counts` tensor is not a single-element tensor. Every file this
    /// build has read against declares one count per tensor
    /// (`shape.dims() == [1]`); a different shape is a fact about this
    /// file worth refusing on rather than silently accepting a convention
    /// no reader has seen.
    CountsNotScalar {
        /// The tensor's declared name.
        name: String,
        /// Its declared shape.
        shape: Shape,
    },
    /// `general.type` was `"imatrix"` but zero `.in_sum2`/`.counts` pairs
    /// were found. An empty enumeration is a defect in the file or in this
    /// read, never a legitimate "no statistics" answer — a real
    /// calibration run always produces at least one entry.
    NoEntries,
}

impl std::fmt::Display for ImatrixError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ImatrixError::NotAnImatrixFile { declared_type } => write!(
                f,
                "general.type is {}, not \"imatrix\"",
                declared_type
                    .as_deref()
                    .map_or("absent".to_string(), |t| format!("{t:?}"))
            ),
            ImatrixError::UnpairedStatistic {
                name,
                in_sum2_present,
            } => {
                let (have, missing) = if *in_sum2_present {
                    (IN_SUM2_SUFFIX, COUNTS_SUFFIX)
                } else {
                    (COUNTS_SUFFIX, IN_SUM2_SUFFIX)
                };
                write!(
                    f,
                    "{name} has a {have} tensor but no matching {missing} tensor"
                )
            }
            ImatrixError::CountsNotScalar { name, shape } => {
                write!(
                    f,
                    "{name}: counts tensor has shape {shape:?}, expected a single element"
                )
            }
            ImatrixError::NoEntries => {
                write!(
                    f,
                    "general.type is \"imatrix\" but no in_sum2/counts pairs were found"
                )
            }
        }
    }
}

impl std::error::Error for ImatrixError {}

fn string_array(metadata: &impl MetadataSource, key: &str) -> Vec<String> {
    match metadata.get(key) {
        Some(MetaValue::Array(items)) => items
            .iter()
            .filter_map(|v| match v {
                MetaValue::String(s) => Some(s.clone()),
                _ => None,
            })
            .collect(),
        Some(MetaValue::String(s)) => vec![s.clone()],
        _ => Vec::new(),
    }
}

fn u32_value(metadata: &impl MetadataSource, key: &str) -> Option<u32> {
    match metadata.get(key)? {
        MetaValue::U32(v) => Some(*v),
        _ => None,
    }
}

/// Read a GGUF-embedded imatrix's structure: validate it declares
/// `general.type = "imatrix"`, pair every `.in_sum2`/`.counts` tensor by
/// name, and enumerate the KV metadata — no bytes decoded, nothing about
/// what the statistics mean.
///
/// `metadata` and `tensors` come from the same file, already opened with
/// [`crate::GgufMetadata::parse`] and [`crate::parse_tensors`] — this
/// function performs no I/O of its own.
///
/// # Errors
///
/// [`ImatrixError::NotAnImatrixFile`] if `general.type` is not
/// `"imatrix"`. [`ImatrixError::UnpairedStatistic`] or
/// [`ImatrixError::CountsNotScalar`] if the tensor directory does not
/// follow the documented pairing convention. [`ImatrixError::NoEntries`]
/// if zero pairs were found — an empty result is refused, not returned, so
/// a caller cannot mistake "this read found nothing" for "this model has no
/// statistics".
pub fn read(
    metadata: &impl MetadataSource,
    tensors: &GgufTensors<'_>,
) -> Result<Imatrix, ImatrixError> {
    let declared_type = match metadata.get("general.type") {
        Some(MetaValue::String(s)) => Some(s.clone()),
        _ => None,
    };
    if declared_type.as_deref() != Some("imatrix") {
        return Err(ImatrixError::NotAnImatrixFile { declared_type });
    }

    let mut entries = Vec::new();
    for d in tensors.tensors() {
        let Some(base) = d.name.strip_suffix(IN_SUM2_SUFFIX) else {
            continue;
        };
        let counts_name = format!("{base}{COUNTS_SUFFIX}");
        let Some(counts) = tensors.tensor(&counts_name) else {
            return Err(ImatrixError::UnpairedStatistic {
                name: d.name.clone(),
                in_sum2_present: true,
            });
        };
        if counts.shape.dims() != [1usize] {
            return Err(ImatrixError::CountsNotScalar {
                name: counts.name.clone(),
                shape: counts.shape.clone(),
            });
        }
        entries.push(ImatrixEntry {
            tensor_name: base.to_string(),
            in_sum2: d.clone(),
            counts: counts.clone(),
        });
    }

    // A `.counts` tensor with no matching `.in_sum2` is the same defect
    // from the other side, and the loop above never visits it because it
    // only starts from `.in_sum2` names.
    for d in tensors.tensors() {
        if let Some(base) = d.name.strip_suffix(COUNTS_SUFFIX) {
            let in_sum2_name = format!("{base}{IN_SUM2_SUFFIX}");
            if tensors.tensor(&in_sum2_name).is_none() {
                return Err(ImatrixError::UnpairedStatistic {
                    name: d.name.clone(),
                    in_sum2_present: false,
                });
            }
        }
    }

    if entries.is_empty() {
        return Err(ImatrixError::NoEntries);
    }

    Ok(Imatrix {
        datasets: string_array(metadata, "imatrix.datasets"),
        chunk_count: u32_value(metadata, "imatrix.chunk_count"),
        chunk_size: u32_value(metadata, "imatrix.chunk_size"),
        entries,
    })
}

// Tests live in `tests/imatrix.rs`, alongside this crate's other domain
// checks (`tests/requirements.rs`), so they can share `tests/fixture`'s
// `GgufBuilder` the way every other test in this crate does — a `#[path]`
// reach from `src/` into `tests/` was tried first and rejected: the
// relative path resolves against a virtual `src/imatrix/tests/` directory
// that does not exist on disk, and getting it right needed one more
// leading `../` per level of module nesting than intuition suggested.
