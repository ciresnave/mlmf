// SPDX-License-Identifier: MIT OR Apache-2.0
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
/// **An EIGHTH outcome pushes an entry and KEEPS the descriptor**: a
/// rebased range running past the end of the file. It is separated from the
/// seven above rather than counted with them, because the count above is
/// about omission and this one is not — and a doc that folded it in would
/// be the "six versus seven" defect again in a new place.
///
/// That outcome is [`UnrecognizedKind::TensorDeclined`], and it exists
/// because the seam's ruling is **keep and report**
/// ([`TensorDescriptor::bytes`]): a descriptor records what the file
/// DECLARES, including a range the file cannot honour. This crate kept such
/// a descriptor from the start and said nothing about it, which is half the
/// ruling — a consumer handed the same defect by this backend and by
/// `mlmf-safetensors` got silence from one and a diagnostic from the other.
///
/// **The bound in [`TensorContainer::tensor_bytes`] is unaffected and must
/// stay.** This adds a report entry; it removes no check. The entry is what
/// an operator reads at open time, the error is what a caller gets at read
/// time, and neither substitutes for the other — a descriptor built by a
/// caller, or carried over from another container, never passes through
/// here at all.
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
    file_len: u64,
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

    if end > file_len {
        // KEPT, not omitted — the eighth outcome in this function's doc.
        // There is deliberately no `return`: falling through to the
        // descriptor below is the whole point, because the seam rules that
        // a declaration the file cannot honour is still a declaration.
        //
        // `>` and not `>=`: a tensor ending exactly at the last byte of the
        // file is entirely present, and reporting it would blame every
        // well-formed model whose final tensor runs to the end — which is
        // most of them.
        //
        // The reason names the file's OWN number first, `info.offset`,
        // which is what an operator finds in the tensor directory, and then
        // the rebased pair, which appears nowhere in the file. The same
        // rule `mlmf-safetensors` follows.
        report.push(Unrecognized {
            kind: UnrecognizedKind::TensorDeclined {
                name: info.name.clone(),
                reason: format!(
                    "declared offset {} rebases to {start}..{end}, past the \
                     end of the {file_len}-byte file",
                    info.offset
                ),
            },
            origin: origin.to_string(),
        });
    }

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
/// A tensor whose declared range runs past the end of the file is neither
/// an error nor omitted: it is KEPT with the range the file declares, named
/// in the report, and fails at [`TensorContainer::tensor_bytes`].
pub fn parse_tensors<'a>(
    bytes: &'a [u8],
    meta: &GgufMetadata<'a>,
    origin: &str,
) -> Result<(GgufTensors<'a>, Report), GgufError> {
    let (descriptors, index, data_start, report) =
        directory(bytes, meta, bytes.len() as u64, origin)?;
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

/// A directory as `parse_tensors` and the ranged parse both build it:
/// descriptors, name index, data start, report.
pub(crate) type Directory = (Vec<TensorDescriptor>, HashMap<String, usize>, u64, Report);

/// Read the tensor directory out of `bytes` (which need only reach the end
/// of the directory) and resolve it against a file of `file_len` bytes.
/// `file_len` is the only thing that differs between the slice parse (its
/// length) and the ranged parse (the source's, or `u64::MAX` when unknown).
pub(crate) fn directory(
    bytes: &[u8],
    meta: &GgufMetadata<'_>,
    file_len: u64,
    origin: &str,
) -> Result<Directory, GgufError> {
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
    // `mlmf-test-support/data/corpus-metadata.tsv`, where every `llamacpp-vocab/*` row
    // declares `n_tensors` 0. Validating here would refuse every
    // vocab-only GGUF, which is exactly what a metadata consumer opens.
    // The bound that matters is per-tensor, in `tensor_bytes`.
    //
    // Task 2c added a SECOND per-tensor bound, in `resolve`, which reports
    // a past-end-of-file range rather than erroring on it. It cannot fire
    // on any of those 19 files: a bound that runs once per tensor runs zero
    // times on a file declaring zero tensors. That is why the measurement
    // above does not transfer, and why this line is still deliberately
    // unvalidated rather than validated now that a length is in scope.
    let data_start = data_start(dir_end, meta.alignment()).ok_or(GgufError::Malformed {
        stage: Stage::TensorDirectory,
        offset: dir_end,
        detail: "the data region's start overflows".to_string(),
    })?;

    let mut descriptors = Vec::new();
    let mut index = HashMap::new();
    for raw in &raws {
        let Some(d) = resolve(raw, data_start, file_len, origin, &mut report) else {
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

    Ok((descriptors, index, data_start, report))
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
mod tests;
