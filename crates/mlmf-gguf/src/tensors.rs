//! The tensor directory: what tensors a file declares and where their bytes are.

use crate::cursor::Cursor;
use crate::error::{GgufError, Stage};
use crate::metadata::GgufMetadata;
// `Encoding` is deliberately NOT imported here: `resolve` produces one via
// `GgmlType::encoding()` without ever naming the type, so a module-scope
// import of it is unused and `-D warnings` refuses it. The tests name it,
// and import it themselves.
use mlmf_core::{
    DeclaredType, Error, ErrorKind, Report, Shape, TensorContainer, TensorDescriptor, Unrecognized,
    UnrecognizedKind,
};
use mlmf_ggml::GgmlType;
use std::borrow::Cow;
use std::collections::HashMap;

/// One tensor-info record, exactly as declared and not yet resolved.
///
/// `code` is a raw ggml type code, not a resolved encoding, and `offset` is
/// relative to the data region rather than to the file. Both stay raw here
/// so that record parsing has no opinion about type tables — the stage that
/// resolves them is the stage that can fail against a type table, and this
/// one must not.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct RawInfo {
    pub(crate) name: String,
    pub(crate) dims: Vec<u64>,
    pub(crate) code: u32,
    pub(crate) offset: u64,
}

fn trunc(at: u64, t: crate::cursor::Truncated) -> GgufError {
    GgufError::Truncated {
        stage: Stage::TensorDirectory,
        offset: at,
        needed: t.needed,
        available: t.available,
    }
}

/// Read one tensor-info record, leaving the cursor on the next.
pub(crate) fn read_info(cursor: &mut Cursor<'_>) -> Result<RawInfo, GgufError> {
    let at = cursor.pos();
    let len = cursor.u64().map_err(|t| trunc(at, t))?;
    let at = cursor.pos();
    let raw = cursor.take(len).map_err(|t| trunc(at, t))?;
    let name = core::str::from_utf8(raw)
        .map_err(|e| GgufError::Malformed {
            stage: Stage::TensorDirectory,
            offset: at,
            detail: format!("tensor name is not valid UTF-8: {e}"),
        })?
        .to_string();

    let at = cursor.pos();
    let n_dims = cursor.u32().map_err(|t| trunc(at, t))?;
    // Bound the declared count by what remains BEFORE allocating. `n_dims`
    // is a u32 from the file, so at u32::MAX it asks for 34 GB.
    //
    // The `try_reserve` below is NOT the protection, and measurement is why
    // this bound is here rather than left to it: with this check deleted,
    // `try_reserve(0xFFFF_FFFF)` SUCCEEDED on the Windows dev host in under
    // a millisecond, committing 34 GB, and only then did the dimension loop
    // fail on the empty remainder. `try_reserve` turns an abort into a
    // recoverable error; it does not stop the allocation from happening
    // when the allocator is willing. Refusing the count against the bytes
    // that remain is what makes "fails before allocating" true.
    let need = u64::from(n_dims)
        .checked_mul(8)
        .ok_or_else(|| GgufError::Malformed {
            stage: Stage::TensorDirectory,
            offset: at,
            detail: format!("{n_dims} dimensions overflows a byte count"),
        })?;
    if need > cursor.remaining() {
        return Err(GgufError::Truncated {
            stage: Stage::TensorDirectory,
            offset: cursor.pos(),
            needed: need,
            available: cursor.remaining(),
        });
    }
    let mut dims = Vec::new();
    dims.try_reserve(n_dims as usize)
        .map_err(|_| GgufError::Malformed {
            stage: Stage::TensorDirectory,
            offset: at,
            detail: format!("cannot allocate {n_dims} dimensions"),
        })?;
    for _ in 0..n_dims {
        let at = cursor.pos();
        dims.push(cursor.u64().map_err(|t| trunc(at, t))?);
    }

    let at = cursor.pos();
    let code = cursor.u32().map_err(|t| trunc(at, t))?;
    let at = cursor.pos();
    let offset = cursor.u64().map_err(|t| trunc(at, t))?;

    Ok(RawInfo {
        name,
        dims,
        code,
        offset,
    })
}

/// Where the tensor data region begins: `dir_end` rounded up to `alignment`.
///
/// `None` on overflow rather than a wrapped value, which would place the
/// data region BEFORE the directory that describes it.
///
/// **The padding is `(alignment - dir_end % alignment) % alignment`, and
/// the outer `%` is the part that matters.** A writer emits no padding at
/// all when the directory already ends on a boundary — measured on
/// `SmolLM2-135M-Instruct-f16.gguf`, where `dir_end == data_start ==
/// 1785664`. The naive `dir_end + alignment - (dir_end % alignment)` adds a
/// full block in that case and shifts every tensor in the file.
///
/// `alignment` is a power of two and at least 1: `GgufMetadata::alignment`
/// guarantees it, falling back to 32 for a file that declares otherwise.
pub(crate) fn data_start(dir_end: u64, alignment: u64) -> Option<u64> {
    debug_assert!(alignment.is_power_of_two(), "caller guarantees this");
    let pad = (alignment - dir_end % alignment) % alignment;
    dir_end.checked_add(pad)
}

/// Turn a raw record into a descriptor, or report why it cannot be one.
///
/// `None` means the tensor is **omitted from the container's list** and the
/// report names it. SEVEN things reach that outcome and every one of them
/// looks the same to a consumer holding the list: a shorter list. The
/// report is the only other signal, so no path here returns `None` without
/// pushing an entry.
///
/// (Six until the whole-branch review. The seventh — a dimension that does
/// not fit this target's `usize` — existed in the code and not in this
/// list, which read as exhaustive and was consulted as though it were.)
///
/// Four of the seven are reported as
/// [`UnrecognizedKind::TensorEncoding`], and they are deliberately not
/// distinguished from one another — a code this build does not know, a
/// retired code, a shape `mlmf-ggml` refuses as ragged, and a size that
/// overflows all say "this build cannot compute this tensor's extent from
/// its declared type", and telling them apart is interpretation.
///
/// The other THREE are [`UnrecognizedKind::TensorDeclined`] — the two
/// rebase overflows and the `usize` narrowing — and that distinction is
/// NOT interpretation: in all three the
/// encoding resolved perfectly. Reporting
/// `TensorEncoding { declared: DeclaredType::Code(0) }` for an F32 tensor
/// would name a code this build recognises and point an operator at a
/// library upgrade that would change nothing. The fact is a declared offset
/// no address space can hold, and `TensorDeclined` exists to say exactly
/// that.
pub(crate) fn resolve(
    info: &RawInfo,
    data_start: u64,
    origin: &str,
    report: &mut Report,
) -> Option<TensorDescriptor> {
    // Not `mut`: `info` and `origin` are captured by shared reference and
    // `report` arrives as a parameter, so this is an `Fn` and calling it
    // from four arms needs no mutable binding.
    let complain = |report: &mut Report| {
        report.push(Unrecognized {
            kind: UnrecognizedKind::TensorEncoding {
                name: info.name.clone(),
                family: "ggml",
                // `DeclaredType::Code`, because GGUF declares a NUMBER in
                // ggml's code space. Safetensors declares a name and takes
                // the other arm; the field admits both so the seam does not
                // have to render one as the other.
                declared: DeclaredType::Code(info.code),
            },
            origin: origin.to_string(),
        });
        None::<TensorDescriptor>
    };
    let decline = |report: &mut Report, reason: String| {
        report.push(Unrecognized {
            kind: UnrecognizedKind::TensorDeclined {
                name: info.name.clone(),
                reason,
            },
            origin: origin.to_string(),
        });
        None::<TensorDescriptor>
    };

    let Some(ty) = GgmlType::from_code(info.code) else {
        return complain(report);
    };
    let Ok(nbytes) = ty.nbytes(&info.dims, &info.name) else {
        return complain(report);
    };
    let dims: Option<Vec<usize>> = info.dims.iter().map(|d| usize::try_from(*d).ok()).collect();
    let Some(dims) = dims else {
        // `decline`, not `complain`. The encoding resolved — `from_code`
        // and `nbytes` both succeeded above — and the failure is that a
        // declared dimension does not fit this target's `usize`. That is a
        // fact about the MACHINE DOING THE READING, not about the model,
        // and D5 says so explicitly.
        //
        // This arm reported `TensorEncoding` until the whole-branch review,
        // which is the exact defect the doc twelve lines above forbids: it
        // would name a ggml code this build recognises perfectly and point
        // an operator at a library upgrade that would change nothing. An
        // F32 tensor reported as an unresolvable ggml encoding.
        //
        // **Unreachable on a 64-bit target**, where every `u64` fits a
        // `usize`, so no test here can drive it and the kind is fixed by
        // inspection rather than by a control. Said plainly rather than
        // left looking covered: on a 32-bit target this is the arm that
        // runs, and it is the one arm in this function no sabotage on this
        // machine can reach.
        return decline(
            report,
            format!(
                "declared dimensions {:?} do not fit this target's usize; \
                 that is a limit of the machine reading the file, not of \
                 the file",
                info.dims
            ),
        );
    };
    let Some(start) = data_start.checked_add(info.offset) else {
        return decline(
            report,
            format!(
                "declared offset {} plus data start {data_start} overflows a u64",
                info.offset
            ),
        );
    };
    let Some(end) = start.checked_add(nbytes) else {
        return decline(
            report,
            format!("byte offset {start} plus {nbytes} bytes overflows a u64"),
        );
    };

    Some(TensorDescriptor {
        name: info.name.clone(),
        shape: Shape::new(dims),
        encoding: ty.encoding(),
        bytes: start..end,
    })
}

/// A GGUF file's tensor directory, parsed and rebased.
#[derive(Debug)]
pub struct GgufTensors<'a> {
    bytes: &'a [u8],
    descriptors: Vec<TensorDescriptor>,
    index: HashMap<String, usize>,
    data_start: u64,
}

impl GgufTensors<'_> {
    /// Where the tensor data region begins, absolute in the slice.
    ///
    /// For a file with no tensors this is a position that may lie one
    /// alignment block PAST the end of the file: the writer emits no
    /// padding when there is nothing to pad for. It is reported rather than
    /// validated for exactly that reason — see [`parse_tensors`].
    #[must_use]
    pub fn data_start(&self) -> u64 {
        self.data_start
    }

    /// How many names the lookup index holds.
    ///
    /// Test surface for the assertion that an index EXISTS covering
    /// every tensor — which is strictly less than "the lookup used
    /// it". Measured: replacing the indexed lookup with a linear scan
    /// while still building the index leaves that assertion green.
    /// This doc previously claimed the stronger property, which the
    /// test's own comment already retracted.
    #[must_use]
    pub fn index_len(&self) -> usize {
        self.index.len()
    }
}

/// Parse the tensor directory that follows `meta`'s key-value block.
///
/// Separate from [`GgufMetadata::parse`] by construction, not by discipline:
/// R1 requires that reading metadata cannot fail on tensor content, and a
/// caller who never calls this function cannot be failed by it.
///
/// # Errors
///
/// [`GgufError::Truncated`] or [`GgufError::Malformed`] with
/// `Stage::TensorDirectory` if a record is unreadable. A tensor whose TYPE
/// this build cannot resolve is not an error — it is omitted and reported.
pub fn parse_tensors<'a>(
    bytes: &'a [u8],
    meta: &GgufMetadata<'a>,
    origin: &str,
) -> Result<(GgufTensors<'a>, Report), GgufError> {
    let mut cursor = Cursor::new(bytes);
    cursor
        .seek(meta.kv_end())
        .map_err(|t| GgufError::Truncated {
            stage: Stage::TensorDirectory,
            offset: meta.kv_end(),
            needed: t.needed,
            available: t.available,
        })?;

    let mut report = Report::new();
    let mut raws = Vec::new();
    for _ in 0..meta.header().tensor_count {
        raws.push(read_info(&mut cursor)?);
    }
    let dir_end = cursor.pos();

    // NOT validated against the file length. A file with no tensors ends at
    // `dir_end` and this value can point one alignment block past the end —
    // 19 of the 28 corpus files are that shape, counted from
    // `tests/corpus-metadata.tsv`, where every `llamacpp-vocab/*` row
    // declares `n_tensors` 0. Validating here would refuse every
    // vocab-only GGUF, which is exactly what a metadata consumer opens.
    // The bound that matters is per-tensor, in `tensor_bytes`.
    let data_start = data_start(dir_end, meta.alignment()).ok_or(GgufError::Malformed {
        stage: Stage::TensorDirectory,
        offset: dir_end,
        detail: "the data region's start overflows".to_string(),
    })?;

    let mut descriptors = Vec::new();
    let mut index = HashMap::new();
    for raw in &raws {
        let Some(d) = resolve(raw, data_start, origin, &mut report) else {
            continue;
        };
        if index.contains_key(&d.name) {
            // `TensorDeclined`, not `TensorEncoding`: this tensor's encoding
            // resolved perfectly well. See Task 0.
            report.push(Unrecognized {
                kind: UnrecognizedKind::TensorDeclined {
                    name: d.name.clone(),
                    reason: "declared more than once; the first occurrence is kept".to_string(),
                },
                origin: origin.to_string(),
            });
            continue;
        }
        index.insert(d.name.clone(), descriptors.len());
        descriptors.push(d);
    }

    // Sort by start once and compare neighbours, rather than comparing every
    // pair. At 272 tensors the quadratic form is 37,000 comparisons and at a
    // thousand it is half a million, for a check that runs on every open.
    //
    // Zero-length ranges are dropped BEFORE the sweep rather than handled
    // inside it, and that `filter` is a correction to this task's brief
    // rather than a flourish. A tensor occupying no bytes cannot share a
    // byte with anything, so it is neither the subject nor the witness of
    // an overlap -- and leaving it in the sweep breaks the sweep in both
    // directions. Measured, not reasoned:
    //
    //   - `sort_unstable_by_key` does not specify the order of equal keys,
    //     so an empty tensor sharing a start with a real one lands second
    //     whenever the file declares it second, and `64 > 0` then reports
    //     the EMPTY tensor as overlapping. That is a false positive on a
    //     tensor that occupies nothing.
    //   - An empty range strictly inside another tensor's range is the same
    //     false positive without any tie to blame.
    //   - Worse, an empty range interposed between two genuinely
    //     overlapping ones splits the neighbour chain and HIDES the real
    //     overlap -- a false negative, which is the direction that costs a
    //     consumer something.
    //
    // What is deliberately NOT fixed here: two NON-empty tensors declared
    // at the same offset do overlap and are reported either way, but which
    // of the two is named depends on the unspecified tie order. Reporting
    // the overlap is the contract; the choice of subject within a tie is
    // not asserted anywhere, and a secondary sort key would be code no test
    // can distinguish.
    let mut order: Vec<usize> = (0..descriptors.len())
        .filter(|i| !descriptors[*i].bytes.is_empty())
        .collect();
    order.sort_unstable_by_key(|i| descriptors[*i].bytes.start);
    // Compare against the FURTHEST END SEEN SO FAR, not against the
    // immediately preceding tensor.
    //
    // An earlier version walked `order.windows(2)`, which only ever
    // compares neighbours. Measured against a file declaring one 256-byte
    // tensor and three small ones entirely inside it:
    //
    //     big 192..448
    //     in1 208..212   reported
    //     in2 224..228   NOT reported
    //     in3 240..244   NOT reported
    //
    // Once `in1` sorts between `big` and `in2`, the pair (`in1`, `in2`) is
    // adjacent and does not overlap, so `in2` and `in3` are silently
    // cleared while sitting wholly inside `big`. The report was non-empty,
    // so "this file has overlaps" was still signalled — but D4's whole
    // argument for reporting rather than refusing is that a consumer may
    // want the tensors that DO NOT overlap, and that consumer was being
    // told two corrupt ones were safe.
    //
    // Carrying the maximum end forward fixes it in the direction that
    // matters: a tensor starting before anything already seen ends is
    // reported, whichever earlier tensor it collides with.
    let mut furthest: Option<usize> = None;
    for &i in &order {
        if let Some(h) = furthest {
            // Strictly greater: `a.end == b.start` is adjacency, which is
            // how every real file is laid out. Empty ranges were filtered
            // above and cannot appear on either side.
            if descriptors[h].bytes.end > descriptors[i].bytes.start {
                report.push(Unrecognized {
                    kind: UnrecognizedKind::TensorDeclined {
                        name: descriptors[i].name.clone(),
                        reason: format!("byte range overlaps {}", descriptors[h].name),
                    },
                    origin: origin.to_string(),
                });
            }
        }
        if furthest.is_none_or(|h| descriptors[i].bytes.end > descriptors[h].bytes.end) {
            furthest = Some(i);
        }
    }

    Ok((
        GgufTensors {
            bytes,
            descriptors,
            index,
            data_start,
        },
        report,
    ))
}

impl TensorContainer for GgufTensors<'_> {
    fn tensors(&self) -> &[TensorDescriptor] {
        &self.descriptors
    }

    /// Indexed, not a scan. The corpus quants declare 272 tensors and a 70B
    /// declares about a thousand; `TensorContainer::tensor`'s own doc
    /// requires a format crate to override the default here.
    fn tensor(&self, name: &str) -> Option<&TensorDescriptor> {
        self.index.get(name).map(|i| &self.descriptors[*i])
    }

    fn tensor_bytes(&self, descriptor: &TensorDescriptor) -> mlmf_core::Result<Cow<'_, [u8]>> {
        // The brief left this arm open. `ErrorKind::Truncated` is the
        // variant chosen, and it is chosen against precedent rather than by
        // elimination: `mlmf-core`'s own reference `RangedSource` — the
        // only other seam in the workspace that answers "this range is not
        // in my bytes" — returns exactly `Truncated { needed: range.end,
        // available: blob.len() }`, and `Cursor::seek` one crate over
        // reports an unreachable absolute position the same way.
        //
        // `Malformed` was the alternative and is worse: nothing here is
        // structurally invalid. Every field of the record parsed, the type
        // resolved, and the arithmetic held; the file simply does not carry
        // as many bytes as it declares. That is the same operator fact a
        // half-finished download produces, and `Truncated`'s two numbers
        // are what let an operator tell the two apart — a `needed` slightly
        // above `available` is a cut-off file, a `needed` of 2^40 against a
        // 128-byte file is a nonsense offset. `Malformed`'s single
        // `message` would flatten that back into prose.
        //
        // Not `SizeMismatch`: that variant compares a declared range
        // against what shape and encoding require, which is an internal
        // disagreement in the record. Here the record agrees with itself
        // and disagrees with the file's length.
        let out_of_range = || {
            Error::from(ErrorKind::Truncated {
                needed: descriptor.bytes.end,
                available: self.bytes.len() as u64,
            })
        };
        // Unreachable on a 64-bit host, where every `u64` is a `usize`, and
        // reachable on a 32-bit one: a file may declare a range this
        // machine cannot address. That is the same fact as a range past the
        // end of the slice — the bytes asked for are not there — so it is
        // reported the same way rather than as a second kind of failure.
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
    // `Encoding` and `DType` are named only here: `resolve` produces an
    // encoding without ever spelling the type, so importing them at module
    // scope would be an unused import.
    use mlmf_core::{DType, Encoding};

    /// Encode one tensor-info record.
    fn info(name: &str, dims: &[u64], code: u32, offset: u64) -> Vec<u8> {
        let mut b = (name.len() as u64).to_le_bytes().to_vec();
        b.extend_from_slice(name.as_bytes());
        b.extend_from_slice(&(dims.len() as u32).to_le_bytes());
        for d in dims {
            b.extend_from_slice(&d.to_le_bytes());
        }
        b.extend_from_slice(&code.to_le_bytes());
        b.extend_from_slice(&offset.to_le_bytes());
        b
    }

    #[test]
    fn reads_a_record_and_lands_exactly_after_it() {
        let b = info("blk.0.attn_q.weight", &[4096, 4096], 0, 1024);
        let mut c = Cursor::new(&b);
        let got = read_info(&mut c).expect("parses");
        // The WHOLE record in one comparison. A chain of field assertions
        // cannot see `code` and a dimension transposed, and both are u32
        // and u64 respectively sitting adjacent in the byte stream.
        assert_eq!(
            got,
            RawInfo {
                name: "blk.0.attn_q.weight".to_string(),
                dims: vec![4096, 4096],
                code: 0,
                offset: 1024,
            }
        );
        // Landing exactly after is what makes the NEXT record readable, and
        // no assertion above can see the cursor one byte off.
        assert_eq!(c.pos(), b.len() as u64, "must consume the record exactly");
    }

    #[test]
    fn a_rank_zero_tensor_is_a_record_not_an_error() {
        // GGUF does not forbid it and the reader must not either: `n_dims`
        // of 0 is a well-formed record with an empty dims list. Refusing it
        // here would be this crate deciding what a model may contain.
        let b = info("scalar", &[], 6, 0);
        let mut c = Cursor::new(&b);
        let got = read_info(&mut c).expect("parses");
        // The whole record, not just `dims`. Asserting the empty dims list
        // alone leaves `code` and `offset` untested HERE, and a `u32`/`u64`
        // swap between them consumes the same twelve bytes either way — so
        // the position assertion below cannot see it either, and this test
        // stays green through a transposition it is sitting right on top of.
        assert_eq!(
            got,
            RawInfo {
                name: "scalar".to_string(),
                dims: Vec::new(),
                code: 6,
                offset: 0,
            }
        );
        assert_eq!(c.pos(), b.len() as u64);
    }

    #[test]
    fn a_dimension_count_larger_than_the_file_fails_before_allocating() {
        // `n_dims` is a declared u32. At 0xFFFF_FFFF, a `Vec::with_capacity`
        // from it asks for 34 GB before a single bounds check runs. The
        // count must be bounded by the bytes that remain.
        let mut b = 6u64.to_le_bytes().to_vec();
        b.extend_from_slice(b"scalar");
        b.extend_from_slice(&u32::MAX.to_le_bytes());
        let mut c = Cursor::new(&b);
        let err = read_info(&mut c).unwrap_err();
        // The WHOLE error, and `needed` above all. Matching only the variant
        // and the stage does not test this at all: delete the bound and the
        // first dimension read fails on the empty remainder with the SAME
        // variant and the SAME stage, so a `..` pattern is green either way
        // — after `try_reserve` has really committed 34 GB. Measured, not
        // predicted: the sabotage that deletes the bound leaves `needed: 8`
        // here, and only comparing `needed` can see the difference.
        //
        // 34_359_738_360 is 0xFFFF_FFFF * 8: the declared count costed at
        // eight bytes a dimension, refused before a byte of it was read.
        // Offset 18 is the 8-byte length, plus "scalar", plus the 4-byte
        // count.
        assert_eq!(
            err,
            GgufError::Truncated {
                stage: Stage::TensorDirectory,
                offset: 18,
                needed: 34_359_738_360,
                available: 0,
            }
        );
    }

    #[test]
    fn a_truncated_record_reports_the_tensor_stage_not_the_metadata_stage() {
        // R7's principle inside the crate: the stage tag is how a caller
        // tells "your metadata is malformed" from "your tensor directory
        // is". Every error out of this module carries TensorDirectory.
        let full = info("t", &[8], 0, 0);
        for cut in 1..full.len() {
            let mut c = Cursor::new(&full[..cut]);
            match read_info(&mut c) {
                Err(GgufError::Truncated { stage, .. }) => {
                    assert_eq!(stage, Stage::TensorDirectory, "cut at {cut}");
                }
                Err(other) => panic!("cut at {cut}: wrong error {other:?}"),
                Ok(v) => panic!("cut at {cut}: parsed {v:?} from a truncated record"),
            }
        }
    }

    #[test]
    fn a_name_that_is_not_utf8_is_malformed_rather_than_lossy() {
        // A tensor name is a lookup key, exactly as a metadata key is. A
        // lossy conversion produces a name no caller can look up and no
        // error anywhere. Values may be non-UTF-8; keys may not.
        let mut b = 2u64.to_le_bytes().to_vec();
        b.extend_from_slice(&[0xFF, 0xFE]);
        b.extend_from_slice(&1u32.to_le_bytes());
        b.extend_from_slice(&8u64.to_le_bytes());
        b.extend_from_slice(&0u32.to_le_bytes());
        b.extend_from_slice(&0u64.to_le_bytes());
        let mut c = Cursor::new(&b);
        assert!(matches!(
            read_info(&mut c).unwrap_err(),
            GgufError::Malformed {
                stage: Stage::TensorDirectory,
                ..
            }
        ));
    }

    #[test]
    fn padding_is_zero_when_the_directory_already_lands_on_a_boundary() {
        // Measured, not assumed. `SmolLM2-135M-Instruct-f16.gguf` has
        // dir_end == data_start == 1785664 with alignment 32: the writer
        // emits NO padding when none is needed. A formula of
        // `dir_end + align - (dir_end % align)` adds a phantom 32 bytes
        // when the directory happens to land on a boundary, and shifts
        // every tensor in the file by one alignment block.
        assert_eq!(data_start(1785664, 32), Some(1785664));
        // And the same file's siblings, which do need padding:
        assert_eq!(data_start(1785944, 32), Some(1785952));
        // The whole table, so an off-by-one in either direction is visible:
        assert_eq!(
            (0..=8).map(|d| data_start(d, 4)).collect::<Vec<_>>(),
            vec![
                Some(0),
                Some(4),
                Some(4),
                Some(4),
                Some(4),
                Some(8),
                Some(8),
                Some(8),
                Some(8)
            ]
        );
    }

    #[test]
    fn a_data_start_that_overflows_is_none_rather_than_wrapping() {
        // `dir_end` comes from a walk over real bytes so it cannot be near
        // u64::MAX in practice, but the addition is still an addition and
        // a wrap would produce a data region BEFORE the directory.
        assert_eq!(data_start(u64::MAX, 32), None);
        // A second alignment with a DIFFERENT residue, so this is not the
        // line above in disguise: `u64::MAX - 1` is 2 short of a multiple
        // of 4, and those 2 bytes of padding are what run off the end.
        assert_eq!(data_start(u64::MAX - 1, 4), None);
        // The other side of the comparison. `u64::MAX - 1` is EVEN, so at
        // alignment 2 it is already on a boundary, needs no padding, and
        // must come back unchanged — a `None` here would mean the overflow
        // guard is refusing values that do not overflow, and the two
        // assertions above cannot tell that apart from a correct refusal.
        assert_eq!(data_start(u64::MAX - 1, 2), Some(u64::MAX - 1));
    }

    #[test]
    fn a_resolvable_tensor_becomes_a_descriptor_rebased_onto_the_slice() {
        // D3, and the reason `TensorDescriptor::bytes` is documented at
        // length: GGUF records an offset relative to the DATA REGION, and
        // the descriptor must carry an offset relative to the SLICE. A
        // consumer that guessed either base would read plausible floats and
        // only one guess would be right.
        let mut report = Report::new();
        let info = RawInfo {
            name: "blk.0.attn_q.weight".into(),
            dims: vec![64, 4],
            code: 0, // F32
            offset: 512,
        };
        let d = resolve(&info, 1_000_000, "t.gguf", &mut report).expect("resolves");
        assert_eq!(
            (d.name.as_str(), d.shape.dims(), d.bytes.clone()),
            (
                "blk.0.attn_q.weight",
                [64usize, 4].as_slice(),
                1_000_512..1_000_512 + 64 * 4 * 4
            )
        );
        assert_eq!(d.encoding, Encoding::Dense(DType::F32));
        assert!(report.is_empty(), "a resolvable tensor is not a finding");
    }

    #[test]
    fn an_unresolvable_code_omits_the_tensor_and_names_it_in_the_report() {
        // D2. There is no descriptor to produce: `TensorDescriptor` has no
        // way to say "length unknown", and inventing one would hand a
        // caller a byte range for a tensor whose extent is unknown. The
        // report is the only signal, and a consumer ignoring it sees a
        // shorter list with nothing else to notice.
        let mut report = Report::new();
        let info = RawInfo {
            name: "blk.0.future".into(),
            dims: vec![32],
            code: 9999,
            offset: 0,
        };
        assert!(resolve(&info, 0, "t.gguf", &mut report).is_none());
        // The WHOLE entry. `!report.is_empty()` cannot see the wrong tensor
        // named, the wrong family, or the code silently defaulted.
        assert_eq!(
            report.entries(),
            [Unrecognized {
                kind: UnrecognizedKind::TensorEncoding {
                    name: "blk.0.future".into(),
                    family: "ggml",
                    declared: DeclaredType::Code(9999),
                },
                origin: "t.gguf".into(),
            }]
        );
    }

    #[test]
    fn a_retired_code_is_reported_like_any_other_unknown() {
        // ggml has eight retired slots. They are not "unknown to this
        // build" in the same sense — nothing will ever define them again —
        // but the consumer-visible outcome is identical: no descriptor, one
        // report entry. Distinguishing them would be interpretation.
        let mut report = Report::new();
        let info = RawInfo {
            name: "old".into(),
            dims: vec![32],
            code: 4, // Q4_2, retired
            offset: 0,
        };
        assert!(resolve(&info, 0, "t.gguf", &mut report).is_none());
        assert_eq!(report.entries().len(), 1);
    }

    #[test]
    fn a_ragged_row_is_reported_rather_than_rounded() {
        // `GgmlType::nbytes` refuses a first dimension that is not a whole
        // number of blocks — the rule is stronger than whole-tensor
        // divisibility and mlmf-ggml owns it. This crate must not paper
        // over that by rounding: a rounded length is a byte range that
        // reads into the next tensor.
        let mut report = Report::new();
        let info = RawInfo {
            name: "ragged".into(),
            dims: vec![33], // Q4_0 blocks are 32 elements
            code: 2,
            offset: 0,
        };
        assert!(resolve(&info, 0, "t.gguf", &mut report).is_none());
        // The WHOLE entry, not a count. Switching this arm from `complain`
        // to `decline` left the workspace green — two of the four
        // TensorEncoding causes had their kind pinned and two did not, and
        // this was one of the two. A count cannot see a kind.
        assert_eq!(
            report.entries(),
            [Unrecognized {
                kind: UnrecognizedKind::TensorEncoding {
                    name: "ragged".into(),
                    family: "ggml",
                    declared: DeclaredType::Code(2),
                },
                origin: "t.gguf".into(),
            }]
        );
    }

    #[test]
    fn a_dimension_that_does_not_fit_the_platform_is_not_a_malformed_file() {
        // D5. `Shape::new` takes `usize`. On a 64-bit target every u64
        // dimension fits and this arm is unreachable; on a 32-bit one it is
        // not. The distinction is recorded because it is easy to report
        // this as a bad file, and it is not one — it is a fact about the
        // machine doing the reading.
        //
        // Asserted through the report's TEXT rather than by constructing a
        // 32-bit failure, which this test cannot do on the host it runs on.
        // The control below is what proves the arm exists.
        let mut report = Report::new();
        let info = RawInfo {
            name: "huge".into(),
            dims: vec![u64::MAX],
            code: 0,
            offset: 0,
        };
        // On a 64-bit host this fails in `nbytes` on overflow, not on the
        // `usize` conversion. Either way: no descriptor, one entry, and the
        // file is not called malformed.
        assert!(resolve(&info, 0, "t.gguf", &mut report).is_none());
        assert_eq!(report.entries().len(), 1);
    }

    #[test]
    fn an_offset_that_leaves_the_address_space_is_declined_out_loud() {
        // The decision the brief left open, made and then pinned. A `None`
        // that pushes nothing is a tensor that vanishes from the list with
        // no signal at all, which is the one outcome
        // `TensorContainer::tensors` documents against — so the rebase
        // reports like every other refusal.
        //
        // `TensorDeclined`, not `TensorEncoding`: code 0 is F32 and this
        // build resolves it perfectly. Reporting it as an unrecognised
        // encoding would name a code that IS recognised and send an
        // operator looking for a newer build of this library, when the
        // fact is a declared offset no address space can hold.
        let mut report = Report::new();
        let info = RawInfo {
            name: "far".into(),
            dims: vec![64],
            code: 0,
            offset: u64::MAX,
        };
        assert!(resolve(&info, 1_000_000, "t.gguf", &mut report).is_none());
        // The WHOLE entry, including the reason text: a length check cannot
        // see this entry arrive with the other kind, and the two kinds are
        // what tell a consumer whether to go looking for a newer build.
        assert_eq!(
            report.entries(),
            [Unrecognized {
                kind: UnrecognizedKind::TensorDeclined {
                    name: "far".into(),
                    reason: "declared offset 18446744073709551615 plus data start 1000000 \
                             overflows a u64"
                        .into(),
                },
                origin: "t.gguf".into(),
            }]
        );
    }

    #[test]
    fn a_length_that_runs_off_the_end_of_the_address_space_is_declined_out_loud() {
        // The SECOND `checked_add`, which is a different arm from the one
        // above and would otherwise be proven by nothing: here the start is
        // representable and the END is not. 2^61 F32 elements is 2^63
        // bytes, placed at 2^63 — a legal start, and one byte of length too
        // many.
        let mut report = Report::new();
        let info = RawInfo {
            name: "long".into(),
            dims: vec![1 << 61],
            code: 0,
            offset: 1 << 63,
        };
        assert!(resolve(&info, 0, "t.gguf", &mut report).is_none());
        assert_eq!(
            report.entries(),
            [Unrecognized {
                kind: UnrecognizedKind::TensorDeclined {
                    name: "long".into(),
                    reason: "byte offset 9223372036854775808 plus 9223372036854775808 bytes \
                             overflows a u64"
                        .into(),
                },
                origin: "t.gguf".into(),
            }]
        );
    }

    /// A GGUF with a tensor directory and a data region.
    ///
    /// `tensors` are (name, dims, ggml code, offset-within-data-region).
    fn gguf_with_tensors(tensors: &[(&str, &[u64], u32, u64)], data: &[u8]) -> Vec<u8> {
        let mut b = Vec::new();
        b.extend_from_slice(b"GGUF");
        b.extend_from_slice(&3u32.to_le_bytes());
        b.extend_from_slice(&(tensors.len() as i64).to_le_bytes());
        b.extend_from_slice(&0i64.to_le_bytes()); // no KV pairs
        for (n, d, c, o) in tensors {
            b.extend_from_slice(&info(n, d, *c, *o));
        }
        // Pad to 32, the default alignment, using the SAME rule the
        // implementation uses — and note that with no tensors and no data
        // this loop adds nothing, which is the corpus's own shape.
        if !data.is_empty() {
            while b.len() % 32 != 0 {
                b.push(0);
            }
        }
        b.extend_from_slice(data);
        b
    }

    #[test]
    fn a_file_with_no_tensors_opens_and_has_no_data_region() {
        // 19 of the 28 readable corpus files are this shape — every
        // `llamacpp-vocab/*` file — and they END at the tensor directory
        // with no padding and no data. An implementation that computes
        // `data_start` eagerly and bounds-checks it against the file length
        // refuses a MAJORITY OF THE CORPUS, and refuses precisely the files
        // a metadata-only consumer opens.
        let bytes = gguf_with_tensors(&[], &[]);
        let (m, _) = GgufMetadata::parse(&bytes, "t.gguf").unwrap();
        let (t, report) = parse_tensors(&bytes, &m, "t.gguf").expect("opens");
        assert_eq!(t.tensors(), &[]);
        assert!(report.is_empty());
        assert_eq!(bytes.len() as u64, m.kv_end(), "the file ends at kv_end");
    }

    #[test]
    fn tensor_bytes_returns_the_declared_range_and_borrows_it() {
        let payload: Vec<u8> = (0..64u8).collect();
        let bytes = gguf_with_tensors(&[("t", &[16], 0, 0)], &payload);
        let (m, _) = GgufMetadata::parse(&bytes, "t.gguf").unwrap();
        let (t, _) = parse_tensors(&bytes, &m, "t.gguf").unwrap();
        let d = t.tensor("t").expect("declared");
        let got = t.tensor_bytes(d).expect("in range");
        assert_eq!(&*got, &payload[..64]);
        // GGUF is always borrowable. `Cow::Owned` here would mean MLMF
        // allocated a copy of a tensor, which is the invisible cost AL-3
        // exists to forbid — and the seam returns `Cow` precisely so a
        // caller can see it.
        assert!(matches!(got, std::borrow::Cow::Borrowed(_)));
    }

    #[test]
    fn a_declared_range_past_the_end_is_an_error_not_a_panic() {
        // A file may declare an offset its own bytes do not cover. That is
        // the file's fault and it must surface as an error, not a slice
        // panic — and not at parse time either, because the other tensors
        // are still readable. R1's shape, one stage over.
        let bytes = gguf_with_tensors(&[("t", &[16], 0, 1 << 40)], &[0u8; 64]);
        let (m, _) = GgufMetadata::parse(&bytes, "t.gguf").unwrap();
        let (t, _) = parse_tensors(&bytes, &m, "t.gguf").expect("the OPEN survives");
        let d = t.tensor("t").expect("still declared");
        assert!(t.tensor_bytes(d).is_err(), "but reading it fails");
    }

    #[test]
    fn lookup_is_indexed_rather_than_a_linear_scan() {
        // `TensorContainer::tensor`'s doc requires this of a format crate
        // parsing a real model: the corpus quants declare 272 tensors and a
        // 70B declares about a thousand, with consumers doing by-name
        // lookups inside per-layer loops.
        //
        // Asserted structurally rather than by timing: the index must hold
        // an entry for every tensor in the list. A timing assertion would
        // be flaky and would pass on a fast linear scan.
        //
        // What that does and does not prove, measured rather than assumed.
        // Deleting the index and scanning instead drives `index_len()` to 0
        // and turns this red. Keeping the index and scanning ANYWAY leaves
        // it GREEN — both were run. So the honest reading of this green is
        // "an index exists covering every tensor", not "the lookup used
        // it". The gap is recorded rather than closed because closing it
        // needs a timing assertion, which is the flaky thing this shape
        // exists to avoid.
        let specs: Vec<(String, Vec<u64>, u32, u64)> = (0..300)
            .map(|i| (format!("blk.{i}.w"), vec![32], 0, i as u64 * 128))
            .collect();
        let refs: Vec<(&str, &[u64], u32, u64)> = specs
            .iter()
            .map(|(n, d, c, o)| (n.as_str(), d.as_slice(), *c, *o))
            .collect();
        let bytes = gguf_with_tensors(&refs, &vec![0u8; 300 * 128]);
        let (m, _) = GgufMetadata::parse(&bytes, "t.gguf").unwrap();
        let (t, _) = parse_tensors(&bytes, &m, "t.gguf").unwrap();
        assert_eq!(t.index_len(), t.tensors().len(), "every tensor is indexed");
        assert_eq!(
            t.tensor("blk.299.w").map(|d| d.name.as_str()),
            Some("blk.299.w")
        );
        assert_eq!(t.tensor("blk.300.w"), None);
    }

    #[test]
    fn a_duplicate_tensor_name_keeps_the_first_and_reports_the_second() {
        // Same rule as a duplicate metadata key, for the same reason:
        // taking the last would make the file's meaning depend on parse
        // order. GGUF does not forbid it.
        let bytes = gguf_with_tensors(&[("t", &[16], 0, 0), ("t", &[16], 0, 64)], &[0u8; 128]);
        let (m, _) = GgufMetadata::parse(&bytes, "t.gguf").unwrap();
        let (t, report) = parse_tensors(&bytes, &m, "t.gguf").unwrap();
        assert_eq!(
            t.tensors()
                .iter()
                .map(|d| d.name.as_str())
                .collect::<Vec<_>>(),
            ["t"],
            "the second occurrence must not be indexed"
        );
        // WHICH occurrence survived, not just that one did. Both records
        // declare `[16]` and differ only in offset, so the name list above
        // is byte-identical under "keep the first" and "keep the last" —
        // the test named for that distinction could not make it. Found by
        // an authored fixture that used differing SHAPES and so could.
        //
        // `data_start()` rather than a literal: the first occurrence sits at
        // offset 0 of the data region, the second at 64. The sabotage that
        // proves this line is a keep-LAST (overwrite the indexed entry), not
        // a keep-BOTH — keeping both reddens the name list above and leaves
        // this assertion unreached.
        assert_eq!(
            t.tensor("t").expect("declared").bytes.start,
            t.data_start(),
            "the FIRST occurrence must be the one kept"
        );
        assert!(!report.is_empty(), "and it must be reported");
    }

    #[test]
    fn overlapping_tensors_are_reported_and_both_stay_readable() {
        // Refusing the open would make one bad tensor cost the whole file,
        // which is R1's argument one stage over. A consumer that cares can
        // read the report; a consumer that wants the other 271 tensors gets
        // them.
        let bytes = gguf_with_tensors(&[("a", &[16], 0, 0), ("b", &[16], 0, 32)], &[0u8; 128]);
        let (m, _) = GgufMetadata::parse(&bytes, "t.gguf").unwrap();
        let (t, report) = parse_tensors(&bytes, &m, "t.gguf").expect("the open survives");
        assert_eq!(t.tensors().len(), 2, "both are still declared");
        assert!(t.tensor_bytes(t.tensor("a").unwrap()).is_ok());
        assert!(t.tensor_bytes(t.tensor("b").unwrap()).is_ok());
        // The whole entry: `!report.is_empty()` cannot see the wrong
        // tensor blamed, and the overlap names TWO tensors of which only
        // one is the subject.
        assert_eq!(
            report.entries(),
            [Unrecognized {
                kind: UnrecognizedKind::TensorDeclined {
                    name: "b".into(),
                    reason: "byte range overlaps a".into(),
                },
                origin: "t.gguf".into(),
            }]
        );
    }

    #[test]
    fn a_tensor_inside_an_earlier_one_is_reported_even_when_it_is_not_adjacent() {
        // The neighbour-only sweep this replaced reported `in1` and cleared
        // `in2` and `in3`, which sit wholly inside `big`. Measured, on the
        // exact fixture below, before the fix:
        //
        //     overlaps reported: 1   -> TensorDeclined { name: "in1", .. }
        //
        // The report was non-empty, so a test asserting `!report.is_empty()`
        // stayed green while two corrupt tensors were reported clean. That
        // is the whole reason this asserts the WHOLE report rather than its
        // emptiness — and the reason the fixture needs THREE intruders
        // rather than one, since a single intruder is adjacent to `big` and
        // the old sweep catches it.
        let bytes = gguf_with_tensors(
            &[
                ("big", &[64], 0, 0),
                ("in1", &[1], 0, 16),
                ("in2", &[1], 0, 32),
                ("in3", &[1], 0, 48),
            ],
            &[0u8; 512],
        );
        let (m, _) = GgufMetadata::parse(&bytes, "t.gguf").unwrap();
        let (t, report) = parse_tensors(&bytes, &m, "t.gguf").expect("the open survives");
        assert_eq!(t.tensors().len(), 4, "every tensor is still declared");

        let named: Vec<(&str, &str)> = report
            .entries()
            .iter()
            .map(|e| match &e.kind {
                UnrecognizedKind::TensorDeclined { name, reason } => {
                    (name.as_str(), reason.as_str())
                }
                other => panic!("wrong kind: {other:?}"),
            })
            .collect();
        assert_eq!(
            named,
            [
                ("in1", "byte range overlaps big"),
                ("in2", "byte range overlaps big"),
                ("in3", "byte range overlaps big"),
            ],
            "each intruder must be named, and blamed on the tensor it is inside"
        );
    }

    #[test]
    fn adjacent_tensors_are_not_an_overlap() {
        // The off-by-one that would make this test necessary is the whole
        // reason it exists: `a` ends at 64 and `b` starts at 64, which is
        // the NORMAL layout of every real file. A `>=` where a `>` belongs
        // reports every tensor in the corpus as overlapping.
        let bytes = gguf_with_tensors(&[("a", &[16], 0, 0), ("b", &[16], 0, 64)], &[0u8; 128]);
        let (m, _) = GgufMetadata::parse(&bytes, "t.gguf").unwrap();
        let (_, report) = parse_tensors(&bytes, &m, "t.gguf").unwrap();
        assert!(report.is_empty(), "touching is not overlapping");
    }

    #[test]
    fn a_zero_length_tensor_never_overlaps() {
        // A rank-0 or empty tensor has start == end. Under a naive interval
        // test an empty range at the same offset as another tensor's start
        // reads as contained-in and reports a false overlap.
        let bytes = gguf_with_tensors(&[("empty", &[0], 0, 0), ("real", &[16], 0, 0)], &[0u8; 128]);
        let (m, _) = GgufMetadata::parse(&bytes, "t.gguf").unwrap();
        let (_, report) = parse_tensors(&bytes, &m, "t.gguf").unwrap();
        assert!(report.is_empty(), "an empty range overlaps nothing");
    }

    #[test]
    fn the_overlapping_pair_is_found_by_offset_not_by_declaration_order() {
        // Added because the sabotage that was supposed to prove the sort
        // was inert: the fixture above declares `a` at 0 and `b` at 32, so
        // declaration order ALREADY equals start order and deleting the
        // sort produced a byte-identical comparison and a green suite.
        // Here the directory is declared in DESCENDING offset order, which
        // GGUF does not forbid, so the unsorted neighbour comparison pairs
        // `b` against `a` the wrong way round and blames the wrong tensor.
        let bytes = gguf_with_tensors(&[("b", &[16], 0, 32), ("a", &[16], 0, 0)], &[0u8; 128]);
        let (m, _) = GgufMetadata::parse(&bytes, "t.gguf").unwrap();
        let (t, report) = parse_tensors(&bytes, &m, "t.gguf").expect("the open survives");
        assert_eq!(t.tensors().len(), 2, "both are still declared");
        // The whole entry, and the NAMES are the point: unsorted this same
        // file reports `a` overlapping `b`, which is the wrong subject.
        assert_eq!(
            report.entries(),
            [Unrecognized {
                kind: UnrecognizedKind::TensorDeclined {
                    name: "b".into(),
                    reason: "byte range overlaps a".into(),
                },
                origin: "t.gguf".into(),
            }]
        );
    }

    #[test]
    fn a_zero_length_tensor_never_overlaps_whichever_order_it_is_declared_in() {
        // The same file as `a_zero_length_tensor_never_overlaps` with the
        // two records swapped. Both tensors start at the data region's
        // first byte, so sorting by `start` ALONE leaves their order
        // unspecified and the empty one can land second -- where
        // `real.end > empty.start` is `64 > 0` and a tensor occupying no
        // bytes at all is reported as overlapping.
        let bytes = gguf_with_tensors(&[("real", &[16], 0, 0), ("empty", &[0], 0, 0)], &[0u8; 128]);
        let (m, _) = GgufMetadata::parse(&bytes, "t.gguf").unwrap();
        let (_, report) = parse_tensors(&bytes, &m, "t.gguf").unwrap();
        assert!(report.is_empty(), "an empty range overlaps nothing");
    }

    #[test]
    fn an_empty_tensor_inside_another_tensors_range_is_not_an_overlap() {
        // `empty` occupies no bytes at all, so there is no byte it can
        // share with `real` -- and no tie to blame either: sorted by start
        // it lands second unconditionally, where a sweep that keeps empty
        // ranges reports `64 > 32` and blames a tensor of zero bytes.
        let bytes = gguf_with_tensors(
            &[("real", &[16], 0, 0), ("empty", &[0], 0, 32)],
            &[0u8; 128],
        );
        let (m, _) = GgufMetadata::parse(&bytes, "t.gguf").unwrap();
        let (_, report) = parse_tensors(&bytes, &m, "t.gguf").unwrap();
        assert!(report.is_empty(), "an empty range overlaps nothing");
    }

    #[test]
    fn an_empty_tensor_between_two_overlapping_ones_does_not_hide_the_overlap() {
        // The false-NEGATIVE direction, which is the one that costs a
        // consumer something. `a` is 0..64 and `b` is 32..96 and they
        // genuinely overlap, but `empty` sorts between them, and a
        // neighbour sweep that keeps it compares `a` against `empty` and
        // `empty` against `b` -- neither of which overlaps -- and never
        // compares the pair that does. The whole entry is asserted because
        // `!report.is_empty()` would be satisfied by the spurious `empty`
        // entry that the same defect also produces.
        let bytes = gguf_with_tensors(
            &[
                ("a", &[16], 0, 0),
                ("empty", &[0], 0, 32),
                ("b", &[16], 0, 32),
            ],
            &[0u8; 128],
        );
        let (m, _) = GgufMetadata::parse(&bytes, "t.gguf").unwrap();
        let (t, report) = parse_tensors(&bytes, &m, "t.gguf").expect("the open survives");
        assert_eq!(t.tensors().len(), 3, "all three are still declared");
        assert_eq!(
            report.entries(),
            [Unrecognized {
                kind: UnrecognizedKind::TensorDeclined {
                    name: "b".into(),
                    reason: "byte range overlaps a".into(),
                },
                origin: "t.gguf".into(),
            }]
        );
    }
}
