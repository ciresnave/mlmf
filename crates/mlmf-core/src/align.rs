//! The alignment contract (spec §4.6).
//!
//! MLMF gives **no blanket alignment guarantee**, because the formats do
//! not agree: GGUF pads tensor data to `general.alignment` (default 32),
//! while safetensors guarantees nothing — its `data_offsets` are
//! cumulative byte offsets, so an F32 tensor following an odd-length U8
//! tensor is misaligned. A memory map yields a page-aligned *base*, which
//! says nothing about per-tensor offsets.
//!
//! In practice nearly all safetensors tensors land aligned because their
//! sizes are large powers of two. That is exactly what makes this
//! dangerous: it works until a checkpoint with an odd-length tensor
//! appears, and then it is undefined behaviour rather than a wrong answer.

use bytemuck::Pod;

use crate::{Error, ErrorKind, Result};

/// The largest alignment [`alignment_of`] will report.
///
/// Beyond this the answer stops being useful and starts being an artifact of
/// the allocator. It is also what an empty range reports, because an empty
/// range satisfies every alignment.
pub const MAX_REPORTED_ALIGN: usize = 64;

/// The alignment these bytes actually have (AL-1).
///
/// Reports what is true, capped at [`MAX_REPORTED_ALIGN`] — beyond that the
/// answer stops being useful and starts being an artifact of the allocator.
///
/// An **empty** range reports the cap, because an empty range constrains
/// nothing: its pointer is dangling and its address is an artifact of where
/// the enclosing region happened to start. Reporting `1` for it stated a
/// constraint that does not exist, so a consumer branching on the fact
/// copied a zero-byte buffer for no reason — and, worse, the same logical
/// zero-element tensor cast successfully at one offset and failed at
/// another.
#[must_use]
pub fn alignment_of(bytes: &[u8]) -> usize {
    if bytes.is_empty() {
        return MAX_REPORTED_ALIGN;
    }
    // `<[u8]>::as_ptr` is never null — Rust guarantees slice pointers are
    // non-null — so the old `if addr == 0` branch was unreachable, and its
    // presence disguised the fact that the empty case was never handled.
    let addr = bytes.as_ptr() as usize;
    let tz = addr.trailing_zeros().min(MAX_REPORTED_ALIGN.trailing_zeros());
    1usize << tz
}

/// Reinterpret bytes as `&[T]`, or fail (AL-2).
///
/// There is deliberately **no infallible typed accessor**. A caller that
/// wants the bytes regardless must choose [`to_aligned_vec`] by name and
/// accept the copy.
///
/// The two ways this can fail are reported as two different kinds, because
/// they call for two different responses: a misalignment is recoverable by
/// copying, a ragged length means the declared range is wrong and copying
/// would only launder the error. Collapsing them produced messages like
/// "4-byte alignment required, address is 64-byte aligned", which is false
/// on its face and which a consumer branching on AL-1 cannot act on.
///
/// # Errors
///
/// [`ErrorKind::RaggedCast`] if the length is not a whole number of `T`;
/// [`ErrorKind::Misaligned`] if the address does not satisfy `T`'s
/// alignment. After the length check, `Misaligned` is reachable only when
/// `actual < required`.
pub fn try_as_slice<T: Pod>(bytes: &[u8]) -> Result<&[T]> {
    let width = size_of::<T>();
    if width == 0 || !bytes.len().is_multiple_of(width) {
        return Err(Error::from(ErrorKind::RaggedCast {
            len: bytes.len(),
            width,
        }));
    }
    if bytes.is_empty() {
        return Ok(&[]);
    }
    bytemuck::try_cast_slice(bytes).map_err(|_| {
        Error::from(ErrorKind::Misaligned {
            required: align_of::<T>(),
            actual: alignment_of(bytes),
        })
    })
}

/// Copy bytes into an owned, correctly-aligned `Vec<T>` (AL-3).
///
/// This is the escape hatch for misaligned data, and it is a separate
/// named call precisely so the copy is a decision the caller makes and can
/// see. A silent copy is the same class of defect as a silent promoting
/// cast: correct output, invisible cost, found only by whoever profiles it.
///
/// A trailing partial element is **refused**, not dropped. Truncating and
/// returning `Ok` made this call strictly more permissive than the fallible
/// accessor it is the fallback for: a caller doing what the docs say —
/// try to borrow, fall back to the named copy — received a silently short
/// buffer, and a 4094-byte range yielded 1023 `f32` instead of an error.
/// Spec §7 admits only two dispositions for something unexpected, fatal or
/// preserved-and-reported; dropping two bytes quietly is neither.
///
/// # Errors
///
/// [`ErrorKind::RaggedCast`] if the length is not a whole number of `T`.
pub fn to_aligned_vec<T: Pod>(bytes: &[u8]) -> Result<Vec<T>> {
    let width = size_of::<T>();
    if width == 0 || !bytes.len().is_multiple_of(width) {
        return Err(Error::from(ErrorKind::RaggedCast {
            len: bytes.len(),
            width,
        }));
    }
    let count = bytes.len() / width;
    let mut out = vec![T::zeroed(); count];
    let dst: &mut [u8] = bytemuck::cast_slice_mut(&mut out);
    dst.copy_from_slice(bytes);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ErrorKind;

    /// A buffer whose base is at least 8-aligned, so a sub-slice's alignment
    /// is decided by the offset we choose and not by allocator luck.
    fn aligned_backing(len: usize) -> Vec<u64> {
        (0..len).map(|i| i as u64).collect()
    }

    #[test]
    fn alignment_of_reports_the_actual_alignment() {
        // AL-1 says every descriptor reports the ACTUAL alignment as a fact
        // that consumers may branch on. The previous version of this test
        // asserted only `>= 1` and `is_power_of_two()`, which is satisfied
        // by EVERY constant the function could return — a body of
        // `{ let _ = bytes; 64 }` and one of `{ let _ = bytes; 1 }` both
        // passed it. Pin real numbers at controlled offsets instead.
        let v = aligned_backing(8);
        let b: &[u8] = bytemuck::cast_slice(&v);
        assert!(alignment_of(b) >= 8, "backing buffer should be 8-aligned");
        assert_eq!(alignment_of(&b[1..]), 1);
        assert_eq!(alignment_of(&b[2..]), 2);
        assert_eq!(alignment_of(&b[3..]), 1);
        assert_eq!(alignment_of(&b[4..]), 4);
        assert_eq!(alignment_of(&b[6..]), 2);
    }

    #[test]
    fn alignment_of_is_capped_at_64() {
        let v = aligned_backing(64);
        let b: &[u8] = bytemuck::cast_slice(&v);
        for off in 0..b.len() {
            let a = alignment_of(&b[off..]);
            assert!(a.is_power_of_two(), "offset {off} reported {a}");
            assert!(a <= MAX_REPORTED_ALIGN, "offset {off} reported {a}");
        }
    }

    #[test]
    fn an_empty_range_constrains_nothing() {
        // A zero-element tensor is legal in safetensors: shape [0, 768],
        // data_offsets [x, x]. Its bytes are `&region[x..x]`, whose pointer
        // is dangling-but-aligned, so the low bits of x decided whether the
        // same logical tensor cast or failed. Reporting `1` was also not
        // honest under AL-1: it states a constraint that does not exist.
        let v = aligned_backing(4);
        let b: &[u8] = bytemuck::cast_slice(&v);
        let empty_at_odd = &b[3..3];
        assert!(empty_at_odd.is_empty());
        assert_eq!(alignment_of(empty_at_odd), MAX_REPORTED_ALIGN);
        assert_eq!(alignment_of(&[][..]), MAX_REPORTED_ALIGN);

        let cast: &[f32] = try_as_slice(empty_at_odd).expect("an empty range casts");
        assert!(cast.is_empty());
        assert!(to_aligned_vec::<f32>(&[]).expect("empty copy").is_empty());
    }

    #[test]
    fn try_as_slice_succeeds_on_aligned_bytes() {
        let src: Vec<u32> = vec![1, 2, 3, 4];
        let bytes: &[u8] = bytemuck::cast_slice(&src);
        let back: &[u32] = try_as_slice(bytes).expect("aligned");
        assert_eq!(back, &[1, 2, 3, 4]);
    }

    #[test]
    fn try_as_slice_refuses_misaligned_bytes_instead_of_copying() {
        let src: Vec<u32> = vec![1, 2, 3, 4];
        let bytes: &[u8] = bytemuck::cast_slice(&src);
        // Offset by one byte: guaranteed not 4-aligned.
        let skewed = &bytes[1..13];
        let err = try_as_slice::<u32>(skewed).unwrap_err();
        // Assert the payload, not just the variant: an error that carries
        // any number rather than a true one is what let `alignment_of`
        // return a constant undetected.
        match err.kind() {
            ErrorKind::Misaligned { required, actual } => {
                assert_eq!(*required, 4);
                assert_eq!(*actual, 1);
            }
            other => panic!("wrong kind: {other:?}"),
        }
    }

    #[test]
    fn misaligned_is_only_ever_reported_when_it_is_true() {
        // The invariant the split buys: after the length check runs first,
        // `Misaligned` cannot be reached with actual >= required.
        let v = aligned_backing(8);
        let b: &[u8] = bytemuck::cast_slice(&v);
        for off in 0..16 {
            for len in 0..12 {
                if let Err(e) = try_as_slice::<u32>(&b[off..off + len])
                    && let ErrorKind::Misaligned { required, actual } = e.kind()
                {
                    assert!(
                        actual < required,
                        "offset {off}, length {len}: reported misaligned with \
                         actual {actual} >= required {required}, which is not \
                         a misalignment"
                    );
                }
            }
        }
    }

    #[test]
    fn realigning_is_a_separate_named_call() {
        // AL-3: MLMF never silently copies. The copy exists, but the
        // caller has to ask for it by name.
        //
        // This used to assert only `owned.len() == 3` and never looked at a
        // single element, so deleting the copy entirely — returning
        // `vec![T::zeroed(); count]` — left the suite green. The one
        // function in mlmf-core that moves tensor bytes had no assertion on
        // the bytes it moves, and a wrong copy here is silent all-zero
        // weights: the model loads, runs and produces garbage.
        let src: Vec<u32> = vec![7, 8, 9, 10];
        let bytes: &[u8] = bytemuck::cast_slice(&src);
        let skewed = &bytes[1..13];
        assert!(try_as_slice::<u32>(skewed).is_err());

        let owned: Vec<u32> = to_aligned_vec(skewed).expect("copy succeeds");
        let expected: Vec<u32> = skewed
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| u32::from_le_bytes(*c))
            .collect();
        assert_eq!(owned.len(), 3);
        assert_eq!(owned, expected);
        assert_eq!(
            owned.as_ptr() as usize % align_of::<u32>(),
            0,
            "the realigned buffer must actually be aligned"
        );
    }

    #[test]
    fn to_aligned_vec_preserves_bytes_exactly() {
        let src: Vec<u32> = vec![0xDEAD_BEEF, 1, 0, u32::MAX];
        let bytes: &[u8] = bytemuck::cast_slice(&src);
        let owned: Vec<u32> = to_aligned_vec(bytes).expect("aligned copy");
        assert_eq!(owned, src);

        let as_u8: Vec<u8> = to_aligned_vec(bytes).expect("byte copy");
        assert_eq!(as_u8, bytes);
    }

    #[test]
    fn a_partial_trailing_element_is_refused_even_when_perfectly_aligned() {
        // The old version of this test used `[0u8; 6]`, whose alignment the
        // compiler owes nothing beyond 1. bytemuck therefore took its
        // *alignment* branch and never reached the slop branch, so the
        // ragged-length behaviour this test names had no coverage at all —
        // which is why both the misreported error kind and the silent
        // truncation in `to_aligned_vec` survived the suite.
        let v = aligned_backing(4);
        let b: &[u8] = bytemuck::cast_slice(&v);
        let six = &b[..6];
        assert!(
            alignment_of(six) >= align_of::<u32>(),
            "alignment must NOT be the problem in this test"
        );

        let err = try_as_slice::<u32>(six).unwrap_err();
        match err.kind() {
            ErrorKind::RaggedCast { len, width } => {
                assert_eq!(*len, 6);
                assert_eq!(*width, 4);
            }
            other => panic!("wrong kind: {other:?}"),
        }
        assert!(err.to_string().contains("whole number"), "{err}");

        // And the copy path agrees rather than being strictly more
        // permissive than the fallible accessor it is the fallback for.
        let err = to_aligned_vec::<u32>(six).unwrap_err();
        assert!(matches!(err.kind(), ErrorKind::RaggedCast { .. }));
    }

    #[test]
    fn a_partial_trailing_element_is_refused_when_misaligned_too() {
        // 6 bytes at an odd offset: both things are wrong. Refusing is what
        // matters; which reason is reported is not asserted here beyond it
        // being one of the two honest ones.
        let bytes = [0u8; 6];
        assert!(try_as_slice::<u32>(&bytes).is_err());
        assert!(to_aligned_vec::<u32>(&bytes).is_err());
    }
}
