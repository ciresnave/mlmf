//! The second stage: the tensor directory, and safetensors' own base.
//!
//! # The base is the point
//!
//! Every format records a tensor offset against a different origin, and
//! none of them is the start of the file. GGUF's is a padded region
//! boundary computed from `general.alignment`. Safetensors' is the end of
//! the JSON header — `8 + header_len` — and nothing else.
//!
//! [`mlmf_core::TensorDescriptor::bytes`] is absolute into the slice, so
//! this module rebases **once, here**, and a consumer thereafter adds
//! nothing to the range — it reads through
//! [`mlmf_core::TensorContainer::tensor_bytes`], which is fallible, rather
//! than slicing, which is not. This paragraph said "writes
//! `&blob[d.bytes]`" and that licence has since been withdrawn from the
//! field's own doc; see the ruling below. The rest of the argument stands:
//! that doc says a consumer guessing the wrong base "would read
//! plausible-looking floats, and only one of them would be right", and
//! until this crate existed the argument had been validated against exactly
//! one base.
//!
//! # What is refused, and what is merely reported
//!
//! Two different failures, split the way `mlmf-gguf` splits them:
//!
//! - An entry that is **not shaped like a tensor record** — no
//!   `data_offsets`, a `shape` that is not an array of non-negative
//!   integers, a `dtype` that is not a string — fails the whole call with
//!   [`SafetensorsError::MalformedEntry`]. The header is not a safetensors
//!   header.
//! - An entry that parses and then **makes a claim this build will not act
//!   on** — an inverted range, a width that disagrees with the shape, a
//!   dtype this build does not know — costs exactly that one tensor. It is
//!   named in the [`Report`], and for six of the seven such claims also
//!   omitted from [`mlmf_core::TensorContainer::tensors`], so
//!   `__metadata__` and every other tensor stay readable. **The seventh is
//!   KEPT** — see the ruling two sections below, and `resolve`, which
//!   enumerates all seven rather than counting them.
//!
//! An unknown dtype is the one of those reported as
//! [`mlmf_core::UnrecognizedKind::TensorEncoding`] rather than
//! `TensorDeclined`: it is the seam's own "cannot resolve the encoding",
//! and it could not be said that way until `mlmf_core::DeclaredType` grew
//! an arm for a type declared as a NAME. See this module's private
//! `resolve`, whose doc enumerates every outcome.
//!
//! # A range past the end of the file is KEPT, and that is a ruling
//!
//! A tensor whose rebased range runs past the last byte of the file stays
//! in [`mlmf_core::TensorContainer::tensors`] carrying the range the file
//! declares, is named in the [`Report`], and fails at
//! [`mlmf_core::TensorContainer::tensor_bytes`].
//!
//! It was omitted here until Task 2b, on a licence
//! [`mlmf_core::TensorDescriptor::bytes`] used to give and no longer does.
//! `mlmf-gguf` keeps such a descriptor and always has, so for as long as
//! this crate dropped it the two backends answered one question two ways,
//! each citing a different seam doc, and both docs were right about what
//! they said. The seam now rules: **a descriptor records what the file
//! DECLARES, including a range the file cannot honour**, and dropping it
//! would be a reader deciding a declaration does not count.
//!
//! **Both backends now keep AND report.** For one task they differed on the
//! second half — `mlmf-gguf` had no end-of-file check anywhere in its
//! directory parse, so it kept such a descriptor silently while this crate
//! named it — and Task 2c closed that. `mlmf-conformance`'s
//! `a_tensor_declared_past_the_end_of_the_file_is_kept_and_reported_by_both`
//! is what holds the two together now; this paragraph is history, and it is
//! written as history because the live version of it went stale one task
//! after it was written.
//!
//! # Tied weights are not an error, and that is a ruling
//!
//! Two tensors declaring **identical `data_offsets`** — `lm_head` and
//! `embed_tokens` sharing one buffer — are a standard safetensors layout
//! that real models ship. This crate resolves both, keeps both readable,
//! and reports **nothing**.
//!
//! `mlmf-gguf` sweeps for overlapping ranges and reports them, because in
//! GGUF every tensor carries its own explicit offset and a writer has no
//! reason to make two collide. **That is a fact about GGUF, not a rule of
//! the seam** — [`mlmf_core::TensorContainer::tensors`] says so — and
//! inheriting it here would blame a valid file. There is deliberately no
//! overlap sweep in this module.
//!
//! # Order is lexicographic, not declaration order
//!
//! `serde_json` without its `preserve_order` feature backs an object with a
//! `BTreeMap`, so [`mlmf_core::TensorContainer::tensors`] yields names
//! sorted by their UTF-8 bytes regardless of the order the header declared
//! them in. Measured, pinned by a test, and stated here rather than left to
//! be discovered: it is a real divergence from `mlmf-gguf`, which yields
//! declaration order.

use std::borrow::Cow;
use std::collections::HashMap;

use mlmf_core::{
    DeclaredType, Encoding, Error, ErrorKind, Report, Shape, TensorContainer, TensorDescriptor,
    Unrecognized, UnrecognizedKind,
};

use crate::dtype::dtype_of;
use crate::error::{SafetensorsError, Stage};
use crate::header::{Header, json_kind};

/// The header key that is not a tensor.
///
/// Safetensors puts its free-form `str -> str` metadata under this key in
/// the same object as the tensor records, so the tensor directory has to
/// know the name in order to skip it. `mlmf-gguf` needs no equivalent: its
/// metadata is a different byte region.
pub(crate) const METADATA_KEY: &str = "__metadata__";

/// A safetensors file's tensor directory, parsed and rebased.
#[derive(Debug)]
pub struct SafetensorsTensors<'a> {
    bytes: &'a [u8],
    descriptors: Vec<TensorDescriptor>,
    index: HashMap<String, usize>,
    data_start: u64,
}

impl SafetensorsTensors<'_> {
    /// Where tensor data begins, absolute in the slice: `8 + header_len`.
    ///
    /// The base every descriptor in this container was rebased from,
    /// reported so a consumer can check the crate's arithmetic against the
    /// file rather than take it on trust.
    #[must_use]
    pub fn data_start(&self) -> u64 {
        self.data_start
    }

    /// How many names the lookup index holds.
    ///
    /// Test surface for the assertion that an index exists covering every
    /// descriptor — which is strictly weaker than "the lookup used it", as
    /// `mlmf-gguf`'s equivalent records.
    #[must_use]
    pub fn index_len(&self) -> usize {
        self.index.len()
    }
}

/// One header entry read as a tensor record, and not yet resolved.
///
/// `dtype` stays a string and the offsets stay relative, deliberately: the
/// stage that resolves a type table is the stage that can fail against one,
/// and reading a record must not. Same split `mlmf-gguf`'s `RawInfo` makes,
/// for the same reason.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct RawEntry {
    pub(crate) name: String,
    pub(crate) dtype: String,
    pub(crate) dims: Vec<u64>,
    pub(crate) start: u64,
    pub(crate) end: u64,
}

/// The whole call fails, and the entry is named because no offset survives.
fn malformed(name: &str, detail: String) -> SafetensorsError {
    SafetensorsError::MalformedEntry {
        stage: Stage::TensorDirectory,
        name: name.to_string(),
        detail,
    }
}

/// Read one header entry as a tensor record.
///
/// Every failure here means the entry is not a tensor record **at all** —
/// a missing field, or one of the wrong JSON kind. Nothing here judges the
/// values: an inverted range and a dtype this build has never heard of both
/// read fine and are [`resolve`]'s business.
fn read_entry(name: &str, value: &serde_json::Value) -> Result<RawEntry, SafetensorsError> {
    let serde_json::Value::Object(record) = value else {
        return Err(malformed(
            name,
            format!(
                "a tensor record must be a JSON object, this is a {}",
                json_kind(value)
            ),
        ));
    };

    let Some(dtype) = record.get("dtype") else {
        return Err(malformed(name, "dtype is missing".to_string()));
    };
    let Some(dtype) = dtype.as_str() else {
        return Err(malformed(
            name,
            format!("dtype must be a string, this is a {}", json_kind(dtype)),
        ));
    };

    let Some(shape) = record.get("shape") else {
        return Err(malformed(name, "shape is missing".to_string()));
    };
    let Some(shape) = shape.as_array() else {
        return Err(malformed(
            name,
            format!("shape must be an array, this is a {}", json_kind(shape)),
        ));
    };
    // `shape.len()` is a MEASURED count of values serde has already
    // materialised, not a count the file declared, so this is not the
    // attacker-controlled allocation `header.rs` bounds before trusting.
    // The header's own length prefix bounded all of it.
    let mut dims = Vec::with_capacity(shape.len());
    for (i, d) in shape.iter().enumerate() {
        let Some(d) = d.as_u64() else {
            return Err(malformed(
                name,
                format!(
                    "shape[{i}] must be a non-negative integer, this is {d} \
                     (a JSON {})",
                    json_kind(d)
                ),
            ));
        };
        dims.push(d);
    }

    let Some(offsets) = record.get("data_offsets") else {
        return Err(malformed(name, "data_offsets is missing".to_string()));
    };
    let Some(offsets) = offsets.as_array() else {
        return Err(malformed(
            name,
            format!(
                "data_offsets must be an array, this is a {}",
                json_kind(offsets)
            ),
        ));
    };
    let [start, end] = offsets.as_slice() else {
        return Err(malformed(
            name,
            format!("data_offsets has {} elements, not 2", offsets.len()),
        ));
    };
    let (Some(start), Some(end)) = (start.as_u64(), end.as_u64()) else {
        return Err(malformed(
            name,
            format!("data_offsets [{start}, {end}] are not both non-negative integers"),
        ));
    };

    Ok(RawEntry {
        name: name.to_string(),
        dtype: dtype.to_string(),
        dims,
        start,
        end,
    })
}

/// Turn a record into a descriptor, or report why it cannot be one.
///
/// **SEVEN outcomes push a report entry. Six of the seven also return
/// `None`; the seventh keeps its descriptor.** Enumerated rather than
/// counted, because this doc said "six" while the code had seven — a bare
/// number in prose is checked by nothing, and `mlmf-gguf`'s twin of this
/// comment carries a parenthesis recording the identical defect being found
/// there.
///
/// **Omitted** — `None`, so the consumer sees a shorter list and the report
/// is the only other signal:
///
/// 1. a `dtype` string this build does not know;
/// 2. `data_offsets` that end before they start;
/// 3. a dimension that does not fit this target's `usize`;
/// 4. a shape product or byte size that overflows a `u64`;
/// 5. a declared width that disagrees with the shape and dtype;
/// 6. offsets that overflow a `u64` once the base is added.
///
/// **Kept** — the descriptor goes in the list and reading it is what fails:
///
/// 7. a rebased range past the end of the file.
///
/// Outcome 1 is [`UnrecognizedKind::TensorEncoding`]; 2 through 7 are
/// [`UnrecognizedKind::TensorDeclined`].
///
/// **Outcome 1 was `TensorDeclined` until the seam grew an arm for it, and
/// that was forced rather than chosen.**
/// [`UnrecognizedKind::TensorEncoding`] is the kind whose doc describes
/// exactly this case — "a tensor whose declared encoding this build cannot
/// resolve" — but it carried `code: u32`, and safetensors declares a type
/// *string*. There was no honest number to put in that field, and a
/// dishonest one would have named a ggml code this build recognises. So the
/// distinction `mlmf-gguf` draws between the two kinds could not be drawn
/// here at all, and this crate's report was uniform where GGUF's was not.
/// The field is now [`mlmf_core::DeclaredType`], which carries a code or a
/// name; this crate takes the name arm and reports the dtype string the
/// file spells.
///
/// **Outcome 7 is kept rather than omitted, and that is the seam's ruling
/// rather than this crate's taste.** [`TensorDescriptor::bytes`] says a
/// descriptor records what the file DECLARES, including a range the file
/// cannot honour; dropping it would be this crate deciding a declaration
/// does not count, which is interpretation and the charter forbids it. It
/// returned `None` until Task 2b, on a licence `TensorDescriptor::bytes`
/// used to give — slice the blob with nothing added — which contradicted
/// [`TensorContainer::tensor_bytes`], documented from the start to error on
/// a range outside the container's data. `mlmf-gguf` read the second doc
/// and kept its descriptor, so two backends answered the same file two ways
/// and nothing in the seam decided which was right.
///
/// Every reason names the offsets **as the file declares them**, not as
/// this function rebases them, except where the rebased pair is the fact
/// being reported. An operator holding a report goes looking in the JSON
/// header, and `83..71` appears nowhere in it.
fn resolve(
    entry: &RawEntry,
    data_start: u64,
    file_len: u64,
    origin: &str,
    report: &mut Report,
) -> Option<TensorDescriptor> {
    // Built rather than pushed, because outcome 7 pushes one of these and
    // then goes on to return a descriptor. A closure that both pushed and
    // returned `None` cannot be reused there, and the alternative was a
    // second copy of this literal.
    let declined = |reason: String| Unrecognized {
        kind: UnrecognizedKind::TensorDeclined {
            name: entry.name.clone(),
            reason,
        },
        origin: origin.to_string(),
    };
    let decline = |report: &mut Report, reason: String| {
        report.push(declined(reason));
        None::<TensorDescriptor>
    };

    let Some(dtype) = dtype_of(&entry.dtype) else {
        // `TensorEncoding`, not `TensorDeclined`: this is the seam's own
        // "cannot resolve the encoding", and the two are different facts a
        // consumer acts on differently — an unknown dtype points at a
        // library upgrade, a bad range points at a corrupt file. The
        // declared type is a NAME because safetensors declares names; there
        // is no number in the file and none is invented.
        report.push(Unrecognized {
            kind: UnrecognizedKind::TensorEncoding {
                name: entry.name.clone(),
                family: "safetensors",
                declared: DeclaredType::Name(entry.dtype.clone()),
            },
            origin: origin.to_string(),
        });
        // Still omitted, and that promise is seam-level and unchanged:
        // `TensorDescriptor::encoding` is not optional, so with no dtype
        // there is no descriptor to keep.
        return None;
    };

    // ORDERING FIRST, and on the file's own numbers before anything is
    // rebased. `TensorDescriptor::validate` makes the same check in the
    // same position and says why: the width below saturates rather than
    // panicking, so with this check removed an inverted range reports a
    // width of 0 and the reader is sent after a missing tensor rather than
    // a swapped pair of offsets.
    if entry.end < entry.start {
        return decline(
            report,
            format!(
                "data_offsets [{}, {}] end before they start",
                entry.start, entry.end
            ),
        );
    }

    let dims: Option<Vec<usize>> = entry
        .dims
        .iter()
        .map(|d| usize::try_from(*d).ok())
        .collect();
    let Some(dims) = dims else {
        // Unreachable on a 64-bit target, where every `u64` is a `usize`,
        // so no sabotage on this machine can drive it. Said plainly rather
        // than left looking covered: on a 32-bit target this is the arm
        // that runs.
        return decline(
            report,
            format!(
                "declared shape {:?} does not fit this target's usize; that is \
                 a limit of the machine reading the file, not of the file",
                entry.dims
            ),
        );
    };
    let shape = Shape::new(dims);
    // Safetensors has no block-quantized types, so this is the only arm of
    // `Encoding` this crate can ever produce — see [`crate::dtype`].
    let encoding = Encoding::Dense(dtype);

    // Core's arithmetic, not a second copy of it: a shape product that
    // overflows a `u64` and a byte size that overflows one are both refused
    // there rather than wrapped, and the message they carry names the
    // tensor.
    let expected = match shape
        .elem_count()
        .and_then(|n| encoding.byte_size(n, &entry.name))
    {
        Ok(n) => n,
        Err(e) => return decline(report, e.to_string()),
    };
    // Saturating, exactly as `TensorDescriptor::byte_len` is, and for the
    // same reason: this must never be a crash. The ordering check above is
    // what makes it a true width rather than an invented one.
    let declared = entry.end.saturating_sub(entry.start);
    if expected != declared {
        // The file disagreeing with ITSELF, which no GGUF path has an
        // analogue for: GGUF derives a tensor's length from its type and
        // shape and never declares it a second time, so there is nothing
        // there to disagree.
        return decline(
            report,
            format!(
                "shape {:?} of {} needs {expected} bytes; data_offsets [{}, {}] span {declared}",
                shape.dims(),
                entry.dtype,
                entry.start,
                entry.end
            ),
        );
    }

    // ---- THE REBASE ---------------------------------------------------
    // `data_start` is `8 + header_len` and nothing else. This is the one
    // fact in the descriptor a consumer cannot check for itself.
    let (Some(start), Some(end)) = (
        data_start.checked_add(entry.start),
        data_start.checked_add(entry.end),
    ) else {
        return decline(
            report,
            format!(
                "data_offsets [{}, {}] plus the {data_start}-byte base overflow a u64",
                entry.start, entry.end
            ),
        );
    };
    // -------------------------------------------------------------------

    if end > file_len {
        // KEPT, not omitted — see outcome 7 in this function's doc. The
        // declaration survives, this entry names the problem, and
        // `tensor_bytes` is where reading it fails. There is deliberately no
        // `return` here: falling through to the descriptor below is the
        // whole of the behaviour change.
        report.push(declined(format!(
            "data_offsets [{}, {}] rebase to {start}..{end}, past the end \
             of the {file_len}-byte file",
            entry.start, entry.end
        )));
    }

    Some(TensorDescriptor {
        name: entry.name.clone(),
        shape,
        encoding,
        bytes: start..end,
    })
}

/// Read the tensor directory out of an already-parsed header.
///
/// Separate from [`crate::parse_header`] by construction: one JSON object
/// carries both stages here (D2), so the split that matters is *parsed the
/// header* versus *derived the tensors from it*, and a caller who never
/// calls this cannot be failed by it.
///
/// `bytes` must be the same slice `header` was parsed from — every offset
/// this produces is absolute into it, and the end-of-file bound is measured
/// against it.
///
/// # Errors
///
/// [`SafetensorsError::MalformedEntry`] if an entry in the header object is
/// not shaped like a tensor record. A record that parses and then makes a
/// claim this build will not act on is **not** an error: it is named in the
/// returned [`Report`], and — for six of the seven such claims — omitted
/// from the container. The seventh, a range past the end of the file, keeps
/// its descriptor and fails at
/// [`mlmf_core::TensorContainer::tensor_bytes`]. This module's private
/// `resolve` enumerates all seven.
pub fn parse_tensors<'a>(
    bytes: &'a [u8],
    header: &Header,
    origin: &str,
) -> Result<(SafetensorsTensors<'a>, Report), SafetensorsError> {
    let file_len = bytes.len() as u64;
    let mut report = Report::new();
    let mut descriptors = Vec::new();
    let mut index = HashMap::new();

    for (name, value) in &header.entries {
        if name.as_str() == METADATA_KEY {
            continue;
        }
        let entry = read_entry(name, value)?;
        let Some(d) = resolve(&entry, header.data_start, file_len, origin, &mut report) else {
            continue;
        };
        // No duplicate-name check, and its absence is a fact about JSON
        // rather than a decision: `serde_json::Map`'s keys are unique by
        // construction, so a header declaring `"weight"` twice was already
        // collapsed — last one wins — before this function saw it.
        // `mlmf-gguf` reports a duplicate because its directory is a
        // sequence of records where two can genuinely both be present.
        // A duplicate key here is INVISIBLE to this crate, which is worth
        // knowing and is not something this crate can fix.
        index.insert(d.name.clone(), descriptors.len());
        descriptors.push(d);
    }

    // Deliberately no overlap sweep. See this module's header: tied
    // weights are a standard safetensors layout and reporting them would
    // blame a valid file.

    Ok((
        SafetensorsTensors {
            bytes,
            descriptors,
            index,
            data_start: header.data_start,
        },
        report,
    ))
}

impl TensorContainer for SafetensorsTensors<'_> {
    fn tensors(&self) -> &[TensorDescriptor] {
        &self.descriptors
    }

    /// Indexed, not a scan. `TensorContainer::tensor`'s own doc requires a
    /// format crate parsing a real model to override the default: the two
    /// files this plan was measured against declare 290 and 201 tensors,
    /// and a caller walking every layer turns the default linear scan into
    /// a quadratic walk MLMF cannot see.
    fn tensor(&self, name: &str) -> Option<&TensorDescriptor> {
        self.index.get(name).map(|i| &self.descriptors[*i])
    }

    fn tensor_bytes(&self, descriptor: &TensorDescriptor) -> mlmf_core::Result<Cow<'_, [u8]>> {
        // `ErrorKind::Truncated`, for the reason `mlmf-gguf` records at the
        // same seam: nothing here is structurally invalid, the file simply
        // does not carry the bytes it declares, and `Truncated`'s two
        // numbers are what let an operator tell a cut-off download from a
        // nonsense offset.
        //
        // Reachable for a descriptor THIS container produced, and that is
        // the point. `parse_tensors` no longer omits a tensor whose rebased
        // range runs past the end of the file: it keeps the declaration and
        // reports it, so this is where such a tensor fails. Also reachable
        // for a descriptor a caller built or carried over from another
        // container, which the signature permits.
        //
        // This comment claimed the first case was unreachable, and it was,
        // for exactly as long as `resolve` dropped those tensors.
        let out_of_range = || {
            Error::from(ErrorKind::Truncated {
                needed: descriptor.bytes.end,
                available: self.bytes.len() as u64,
            })
        };
        let start = usize::try_from(descriptor.bytes.start).map_err(|_| out_of_range())?;
        let end = usize::try_from(descriptor.bytes.end).map_err(|_| out_of_range())?;
        self.bytes
            .get(start..end)
            .map(Cow::Borrowed)
            .ok_or_else(out_of_range)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::header::parse_header;
    use mlmf_core::DType;

    const ORIGIN: &str = "model.safetensors";

    // ---- fixtures ------------------------------------------------------
    //
    // Every byte-range literal asserted below is written out, not computed.
    // `8 + header_len + data_offsets.0` in a test body is the production
    // expression copied into the control, and a control that recomputes the
    // thing it is checking agrees with itself however wrong both are. The
    // independent check on these numbers is
    // `docs/superpowers/plans/safetensors_recon.py`, which shares no code
    // with this crate.

    /// 94 bytes of JSON. One `__metadata__` block and one BF16 tensor of
    /// 2 x 3 = 6 elements x 2 bytes = 12 bytes at `data_offsets [0, 12]`.
    /// Byte-identical to the fixture `header.rs` pins `header_len: 94` and
    /// `data_start: 102` against.
    const WELL_FORMED: &[u8] =
        br#"{"__metadata__":{"format":"pt"},"weight":{"dtype":"BF16","shape":[2,3],"data_offsets":[0,12]}}"#;

    /// 132 bytes. Two tensors, identical `data_offsets`: tied weights.
    const TIED: &[u8] = br#"{"embed_tokens":{"dtype":"BF16","shape":[2,3],"data_offsets":[0,12]},"lm_head":{"dtype":"BF16","shape":[2,3],"data_offsets":[0,12]}}"#;

    /// 63 bytes. `data_offsets` in the wrong order.
    const INVERTED: &[u8] = br#"{"weight":{"dtype":"BF16","shape":[2,3],"data_offsets":[12,0]}}"#;

    /// 63 bytes. Well-formed offsets; the file will be too short for them.
    const PAST_END: &[u8] = br#"{"weight":{"dtype":"BF16","shape":[2,3],"data_offsets":[0,12]}}"#;

    /// 63 bytes. 6 BF16 elements need 12 bytes; the range spans 10.
    const SIZE_MISMATCH: &[u8] =
        br#"{"weight":{"dtype":"BF16","shape":[2,3],"data_offsets":[0,10]}}"#;

    /// 66 bytes. A dtype string safetensors does not define.
    const UNKNOWN_DTYPE: &[u8] =
        br#"{"weight":{"dtype":"F4_E2M1","shape":[2,3],"data_offsets":[0,12]}}"#;

    /// 41 bytes. A record with no `data_offsets` at all.
    const NO_OFFSETS: &[u8] = br#"{"weight":{"dtype":"BF16","shape":[2,3]}}"#;

    /// 12 bytes. A header entry that is not a record at all.
    const NOT_A_RECORD: &[u8] = br#"{"weight":5}"#;

    /// 116 bytes. Declares `z` first and `a` second.
    const OUT_OF_ORDER: &[u8] = br#"{"z":{"dtype":"BF16","shape":[2,3],"data_offsets":[0,12]},"a":{"dtype":"BF16","shape":[2,3],"data_offsets":[12,24]}}"#;

    /// Twelve bytes of tensor data, all distinct, so a payload read from
    /// the wrong offset is not silently identical to the right one.
    const PAYLOAD: [u8; 12] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];

    /// A safetensors image: a truthful 8-byte length prefix, `json`, then
    /// `data_len` zero bytes of tensor data.
    fn image(json: &[u8], data_len: usize) -> Vec<u8> {
        let mut v = (json.len() as u64).to_le_bytes().to_vec();
        v.extend_from_slice(json);
        v.extend(std::iter::repeat_n(0u8, data_len));
        v
    }

    /// The same, with `PAYLOAD` as the tensor data.
    fn image_with_payload(json: &[u8]) -> Vec<u8> {
        let mut v = (json.len() as u64).to_le_bytes().to_vec();
        v.extend_from_slice(json);
        v.extend_from_slice(&PAYLOAD);
        v
    }

    // ---- the production path, and only the production path -------------

    /// Run both stages exactly as a consumer does and reduce the outcome to
    /// values that compare. **No arithmetic here.** Every offset in an
    /// assertion below came out of `parse_header` and `parse_tensors`.
    ///
    /// Returning a `Result` rather than unwrapping is what keeps a sabotage
    /// that stops an error being produced from panicking before any
    /// comparison happens.
    fn directory(image: &[u8]) -> Result<(Vec<TensorDescriptor>, Report), SafetensorsError> {
        let header = parse_header(image)?;
        let (tensors, report) = parse_tensors(image, &header, ORIGIN)?;
        Ok((tensors.tensors().to_vec(), report))
    }

    /// As [`directory`], and additionally reads every tensor's bytes back
    /// through [`TensorContainer::tensor_bytes`] and looks every name up
    /// through [`TensorContainer::tensor`].
    ///
    /// `Result<Vec<u8>, String>` rather than `mlmf_core::Result`, because
    /// `mlmf_core::Error` is not `PartialEq` and this must compare as a
    /// whole value.
    #[allow(clippy::type_complexity)]
    fn directory_with_bytes(
        image: &[u8],
    ) -> Result<
        (
            Vec<TensorDescriptor>,
            Report,
            Vec<Result<Vec<u8>, String>>,
            Vec<Option<TensorDescriptor>>,
        ),
        SafetensorsError,
    > {
        let header = parse_header(image)?;
        let (tensors, report) = parse_tensors(image, &header, ORIGIN)?;
        let payloads = tensors
            .tensors()
            .iter()
            .map(|d| {
                tensors
                    .tensor_bytes(d)
                    .map(Cow::into_owned)
                    .map_err(|e| e.to_string())
            })
            .collect();
        let by_name = tensors
            .tensors()
            .iter()
            .map(|d| tensors.tensor(&d.name).cloned())
            .collect();
        Ok((tensors.tensors().to_vec(), report, payloads, by_name))
    }

    /// One `TensorDeclined` entry: six of the seven complaints this module
    /// emits. The seventh is [`unresolvable_dtype`].
    fn declined(name: &str, reason: &str) -> Report {
        let mut r = Report::new();
        r.push(Unrecognized {
            kind: UnrecognizedKind::TensorDeclined {
                name: name.to_string(),
                reason: reason.to_string(),
            },
            origin: ORIGIN.to_string(),
        });
        r
    }

    /// One `TensorEncoding` entry, which is what an unknown dtype is: the
    /// seam's "cannot resolve the encoding", carrying the type the file
    /// declares as a NAME because safetensors declares names.
    fn unresolvable_dtype(name: &str, dtype: &str) -> Report {
        let mut r = Report::new();
        r.push(Unrecognized {
            kind: UnrecognizedKind::TensorEncoding {
                name: name.to_string(),
                family: "safetensors",
                declared: DeclaredType::Name(dtype.to_string()),
            },
            origin: ORIGIN.to_string(),
        });
        r
    }

    fn bf16_2x3(name: &str, start: u64, end: u64) -> TensorDescriptor {
        TensorDescriptor {
            name: name.to_string(),
            shape: Shape::new([2usize, 3]),
            encoding: Encoding::Dense(DType::BF16),
            bytes: start..end,
        }
    }

    // ---- D4: the different base ----------------------------------------

    #[test]
    fn a_resolvable_tensor_is_rebased_onto_the_slice_from_the_end_of_the_header() {
        // THE assertion this task exists for. The header is 94 bytes, so
        // safetensors' base is 8 + 94 = 102, and a tensor at
        // `data_offsets [0, 12]` occupies 102..114 of the slice.
        //
        // 102 and 114 are literals. A test that wrote `8 + h.header_len`
        // would agree with a rebase from 0, from 94, or from the moon,
        // because it would be evaluating the implementation's own
        // expression a second time.
        //
        // The whole `Result` is compared in one `assert_eq!` rather than
        // unwrapped and picked apart: a chain of field assertions cannot
        // see two descriptors swapped, and `unwrap` panics before any
        // comparison happens when a sabotage removes the value entirely.
        assert_eq!(
            directory(&image_with_payload(WELL_FORMED)),
            Ok((vec![bf16_2x3("weight", 102, 114)], Report::new()))
        );
    }

    #[test]
    fn the_container_reports_the_base_it_rebased_from() {
        // `data_start` is the number the descriptor above was built on.
        // Asserted separately, and as a literal, so a wrong rebase and a
        // wrong `data_start` cannot cancel out.
        let image = image_with_payload(WELL_FORMED);
        let header = parse_header(&image).expect("Task 1 pins this fixture");
        let (tensors, _) = parse_tensors(&image, &header, ORIGIN).expect("well-formed");
        assert_eq!((tensors.data_start(), tensors.index_len()), (102, 1));
    }

    // ---- D1's second ruling ---------------------------------------------

    #[test]
    fn tied_weights_both_resolve_both_are_readable_and_the_report_is_empty() {
        // `lm_head` and `embed_tokens` on one buffer is a standard
        // safetensors layout, not a defect, so BOTH descriptors exist,
        // BOTH read back the same twelve bytes, and the report is EMPTY.
        //
        // This is the control for a RULING rather than for a line of code.
        // `mlmf-gguf` reports an overlap because a GGUF writer has no
        // reason to produce one; importing that answer here would decline a
        // valid file with a verdict that reads as a defect in the file
        // rather than in the reader. The empty `Report` in this literal is
        // the assertion that this crate did not inherit it.
        //
        // Base is 8 + 132 = 140; both tensors are `[0, 12]`, so both are
        // 140..152. Literals, not arithmetic.
        assert_eq!(
            directory_with_bytes(&image_with_payload(TIED)),
            Ok((
                vec![
                    bf16_2x3("embed_tokens", 140, 152),
                    bf16_2x3("lm_head", 140, 152),
                ],
                Report::new(),
                vec![Ok(PAYLOAD.to_vec()), Ok(PAYLOAD.to_vec())],
                vec![
                    Some(bf16_2x3("embed_tokens", 140, 152)),
                    Some(bf16_2x3("lm_head", 140, 152)),
                ],
            ))
        );
    }

    // ---- records that parse and then make a false claim ------------------

    #[test]
    fn data_offsets_that_end_before_they_start_are_declined_not_accepted() {
        // Reported with the file's OWN numbers, `[12, 0]`, rather than with
        // the rebased pair 83..71 — which is the true absolute range and
        // appears nowhere in the header the operator is about to open.
        assert_eq!(
            directory(&image_with_payload(INVERTED)),
            Ok((
                vec![],
                declined("weight", "data_offsets [12, 0] end before they start")
            ))
        );
    }

    #[test]
    fn data_offsets_running_past_the_end_of_the_file_keep_the_declaration_and_fail_at_read() {
        // Base 8 + 63 = 71, so `[0, 12]` is 71..83 — and this image carries
        // only 4 bytes of tensor data, so the file ends at 75. Every number
        // below is a literal; 71, 83 and 75 came out of that arithmetic
        // written here in the comment and NOT in the test body, because a
        // body that recomputes `8 + header_len + offset` agrees with the
        // implementation however wrong both are.
        //
        // This test asserted `vec![]` — omission — until Task 2b, and the
        // inversion is a seam ruling rather than a preference. A descriptor
        // records what the file DECLARES, including a range the file cannot
        // honour, so the declaration survives, the report names it, and
        // `tensor_bytes` is the one place it fails. The comment that used
        // to sit here cited `TensorDescriptor::bytes`'s licence to write
        // `&blob[d.bytes]`; that licence has been withdrawn from the doc it
        // was quoting, which is why the answer changed.
        //
        // Three facts in one whole-value comparison, because any two of
        // them holding while the third does not is precisely the state the
        // two backends were in: the descriptor is IN the list with the
        // declared range, the report NAMES it, and reading it ERRORS.
        assert_eq!(
            directory_with_bytes(&image(PAST_END, 4)),
            Ok((
                vec![bf16_2x3("weight", 71, 83)],
                declined(
                    "weight",
                    "data_offsets [0, 12] rebase to 71..83, past the end of the 75-byte file"
                ),
                vec![Err("truncated: needed 83 bytes, 75 available".to_string())],
                vec![Some(bf16_2x3("weight", 71, 83))],
            ))
        );
    }

    #[test]
    fn a_shape_that_disagrees_with_the_declared_width_is_declined() {
        // The file contradicting ITSELF: 2 x 3 BF16 elements need 12 bytes
        // and the range spans 10. No GGUF path has an analogue — GGUF
        // derives a tensor's length from its type and shape and never
        // declares it a second time, so there is nothing there to disagree.
        assert_eq!(
            directory(&image(SIZE_MISMATCH, 10)),
            Ok((
                vec![],
                declined(
                    "weight",
                    "shape [2, 3] of BF16 needs 12 bytes; data_offsets [0, 10] span 10"
                )
            ))
        );
    }

    #[test]
    fn a_dtype_this_build_does_not_know_is_an_unresolvable_encoding_not_a_generic_decline() {
        // `TensorEncoding`, carrying `DeclaredType::Name("F4_E2M1")`.
        //
        // This asserted `TensorDeclined` until Task 2b, and that was forced
        // rather than chosen: the variant carried `code: u32`, which
        // assumes a format declaring a NUMERIC type code, and safetensors
        // declares a string. There was no honest u32 to put there, so the
        // seam's own distinction — cannot resolve the encoding, versus
        // declined for another reason — was inexpressible here and the two
        // facts collapsed into one kind. They are different facts: an
        // unknown dtype points a consumer at a library upgrade, a bad range
        // points at a corrupt file.
        //
        // The whole `Report` is compared, not the kind's discriminant. A
        // `matches!(declared, DeclaredType::Name(_))` is satisfied by the
        // tensor's own name, by the empty string, and by every other string
        // a body could return; only the value pins which string the file
        // actually declared.
        assert_eq!(
            directory(&image_with_payload(UNKNOWN_DTYPE)),
            Ok((vec![], unresolvable_dtype("weight", "F4_E2M1")))
        );
    }

    // ---- entries that are not records at all -----------------------------

    #[test]
    fn an_entry_with_no_data_offsets_fails_the_call_and_names_the_key() {
        assert_eq!(
            directory(&image(NO_OFFSETS, 0)),
            Err(SafetensorsError::MalformedEntry {
                stage: Stage::TensorDirectory,
                name: "weight".to_string(),
                detail: "data_offsets is missing".to_string(),
            })
        );
    }

    #[test]
    fn an_entry_that_is_not_an_object_fails_the_call_and_says_what_it_found() {
        assert_eq!(
            directory(&image(NOT_A_RECORD, 0)),
            Err(SafetensorsError::MalformedEntry {
                stage: Stage::TensorDirectory,
                name: "weight".to_string(),
                detail: "a tensor record must be a JSON object, this is a number".to_string(),
            })
        );
    }

    // ---- __metadata__ is not a tensor ------------------------------------

    #[test]
    fn the_metadata_key_is_not_a_tensor() {
        // `WELL_FORMED` declares `__metadata__` alongside `weight`. The
        // rebase test above already shows only one descriptor survives;
        // this pins the negative directly, because "the list has one entry"
        // is also satisfied by a build that dropped `weight` and kept
        // `__metadata__` under a fabricated shape.
        let image = image_with_payload(WELL_FORMED);
        let header = parse_header(&image).expect("Task 1 pins this fixture");
        let (tensors, report) = parse_tensors(&image, &header, ORIGIN).expect("well-formed");
        assert_eq!(
            (
                tensors
                    .tensors()
                    .iter()
                    .map(|d| d.name.clone())
                    .collect::<Vec<_>>(),
                tensors.tensor(METADATA_KEY).cloned(),
                report,
            ),
            (vec!["weight".to_string()], None, Report::new())
        );
    }

    // ---- order, measured rather than assumed ------------------------------

    #[test]
    fn tensors_come_back_in_lexicographic_order_not_declaration_order() {
        // Not a preference — a measurement of what `serde_json` without
        // `preserve_order` does, pinned so Task 5's cross-backend
        // comparison meets a documented divergence rather than a surprise.
        // `mlmf-gguf` yields declaration order; this yields sorted order.
        //
        // Base is 8 + 116 = 124. `a` is declared second and comes first.
        assert_eq!(
            directory(&image(OUT_OF_ORDER, 24)),
            Ok((
                vec![bf16_2x3("a", 136, 148), bf16_2x3("z", 124, 136)],
                Report::new()
            ))
        );
    }
}
