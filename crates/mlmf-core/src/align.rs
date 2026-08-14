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

/// The alignment these bytes actually have (AL-1).
///
/// Reports what is true, capped at 64 — beyond that the answer stops being
/// useful and starts being an artifact of the allocator.
#[must_use]
pub fn alignment_of(bytes: &[u8]) -> usize {
    let addr = bytes.as_ptr() as usize;
    if addr == 0 {
        return 1;
    }
    let tz = addr.trailing_zeros().min(6);
    1usize << tz
}

/// Reinterpret bytes as `&[T]`, or fail (AL-2).
///
/// There is deliberately **no infallible typed accessor**. A caller that
/// wants the bytes regardless must choose [`to_aligned_vec`] by name and
/// accept the copy.
///
/// # Errors
///
/// [`ErrorKind::Misaligned`] if the address does not satisfy `T`'s
/// alignment, or if the length is not a whole number of `T`.
pub fn try_as_slice<T: Pod>(bytes: &[u8]) -> Result<&[T]> {
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
/// Trailing bytes that do not fill a whole `T` are dropped.
///
/// # Errors
///
/// Never returns `Err` today; the signature is `Result` so that adding a
/// bound later is not a breaking change.
pub fn to_aligned_vec<T: Pod>(bytes: &[u8]) -> Result<Vec<T>> {
    let width = size_of::<T>();
    let count = bytes.len() / width;
    let mut out = vec![T::zeroed(); count];
    let dst: &mut [u8] = bytemuck::cast_slice_mut(&mut out);
    dst.copy_from_slice(&bytes[..count * width]);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ErrorKind;

    #[test]
    fn alignment_of_reports_the_actual_alignment() {
        let buf = vec![0u8; 64];
        // A Vec<u8> allocation is at least 1-aligned; report what is true,
        // never what is convenient (spec AL-1).
        assert!(alignment_of(&buf) >= 1);
        assert!(alignment_of(&buf).is_power_of_two());
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
        assert!(matches!(err.kind(), ErrorKind::Misaligned { .. }));
    }

    #[test]
    fn realigning_is_a_separate_named_call() {
        // AL-3: MLMF never silently copies. The copy exists, but the
        // caller has to ask for it by name.
        let src: Vec<u32> = vec![7, 8, 9, 10];
        let bytes: &[u8] = bytemuck::cast_slice(&src);
        let skewed = &bytes[1..13];
        assert!(try_as_slice::<u32>(skewed).is_err());
        let owned: Vec<u32> = to_aligned_vec(skewed).expect("copy succeeds");
        assert_eq!(owned.len(), 3);
    }

    #[test]
    fn a_partial_trailing_element_is_refused() {
        let bytes = [0u8; 6];
        assert!(try_as_slice::<u32>(&bytes).is_err());
    }
}
