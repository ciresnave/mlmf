//! The tensor directory: what tensors a file declares and where their bytes are.

use crate::cursor::Cursor;
use crate::error::{GgufError, Stage};

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

#[cfg(test)]
mod tests {
    use super::*;

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
}
