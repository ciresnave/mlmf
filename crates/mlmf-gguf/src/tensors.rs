//! The tensor directory: what tensors a file declares and where their bytes are.

use crate::cursor::Cursor;
use crate::error::{GgufError, Stage};
// `Encoding` is deliberately NOT imported here: `resolve` produces one via
// `GgmlType::encoding()` without ever naming the type, so a module-scope
// import of it is unused and `-D warnings` refuses it. The tests name it,
// and import it themselves.
use mlmf_core::{Report, Shape, TensorDescriptor, Unrecognized, UnrecognizedKind};
use mlmf_ggml::GgmlType;

/// One tensor-info record, exactly as declared and not yet resolved.
///
/// `code` is a raw ggml type code, not a resolved encoding, and `offset` is
/// relative to the data region rather than to the file. Both stay raw here
/// so that record parsing has no opinion about type tables — the stage that
/// resolves them is the stage that can fail against a type table, and this
/// one must not.
#[derive(Debug, Clone, PartialEq, Eq)]
// Nothing outside this module's own tests reads a record yet, and
// `pub(crate)` items with no in-crate consumer are `dead_code`. The
// directory walk that consumes them is the next task; this allow comes off
// then. `expect` is wrong here: under `cfg(test)` the tests DO use these, so
// the expectation would go unfulfilled and `--all-targets` would fail from
// the other side.
#[allow(dead_code)]
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
// See `RawInfo`: no in-crate caller until the directory walk lands. Allowing
// it here also makes `trunc` reachable, so `trunc` needs no attribute.
#[allow(dead_code)]
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
// See `RawInfo`: the directory walk that will call this lands in a later
// task, so there is no in-crate consumer yet. `expect` is wrong for the
// same reason it is wrong there — under `cfg(test)` the tests DO call it.
#[allow(dead_code)]
pub(crate) fn data_start(dir_end: u64, alignment: u64) -> Option<u64> {
    debug_assert!(alignment.is_power_of_two(), "caller guarantees this");
    let pad = (alignment - dir_end % alignment) % alignment;
    dir_end.checked_add(pad)
}

/// Turn a raw record into a descriptor, or report why it cannot be one.
///
/// `None` means the tensor is **omitted from the container's list** and the
/// report names it. Six things reach that outcome and every one of them
/// looks the same to a consumer holding the list: a shorter list. The
/// report is the only other signal, so no path here returns `None` without
/// pushing an entry.
///
/// Four of the six are reported as
/// [`UnrecognizedKind::TensorEncoding`], and they are deliberately not
/// distinguished from one another — a code this build does not know, a
/// retired code, a shape `mlmf-ggml` refuses as ragged, and a size that
/// overflows all say "this build cannot compute this tensor's extent from
/// its declared type", and telling them apart is interpretation.
///
/// The other two are [`UnrecognizedKind::TensorDeclined`], and that
/// distinction is NOT interpretation: when the rebase overflows, the
/// encoding resolved perfectly. Reporting `TensorEncoding { code: 0 }` for
/// an F32 tensor would name a code this build recognises and point an
/// operator at a library upgrade that would change nothing. The fact is a
/// declared offset no address space can hold, and `TensorDeclined` exists
/// to say exactly that.
// See `RawInfo`: the directory walk that will call this lands in a later
// task, so there is no in-crate consumer yet. `expect` is wrong for the
// same reason it is wrong there — under `cfg(test)` the tests DO call it.
#[allow(dead_code)]
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
                code: info.code,
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
        return complain(report);
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
                    code: 9999,
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
        assert_eq!(report.entries().len(), 1);
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
}
