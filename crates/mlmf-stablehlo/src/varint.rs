// SPDX-License-Identifier: MIT OR Apache-2.0
//! MLIR bytecode's prefix-coded variable-width integer.
//!
//! The position of the lowest set bit of the first byte gives the byte count
//! `n` (1 to 8): the value is `raw >> n` where `raw` is the `n` bytes read
//! little-endian. A first byte of zero means the next eight bytes are the
//! value, little-endian. The writer uses the shortest form; the reader
//! accepts any.

/// Decode one integer at `pos`; returns the value and the next position, or
/// `None` if the bytes run out.
#[must_use]
pub fn read_varint(bytes: &[u8], pos: usize) -> Option<(u64, usize)> {
    let first = *bytes.get(pos)?;
    if first == 0 {
        let raw = bytes.get(pos + 1..pos + 9)?;
        return Some((le(raw), pos + 9));
    }
    let n = first.trailing_zeros() as usize + 1;
    let raw = bytes.get(pos..pos + n)?;
    Some((le(raw) >> n, pos + n))
}

fn le(raw: &[u8]) -> u64 {
    raw.iter().rev().fold(0u64, |a, &b| (a << 8) | u64::from(b))
}

/// Append `v` in its shortest form.
pub fn write_varint(out: &mut Vec<u8>, v: u64) {
    for n in 1..=8u32 {
        if v < 1u64 << (7 * n) {
            let raw = v << n | 1u64 << (n - 1);
            out.extend_from_slice(&raw.to_le_bytes()[..n as usize]);
            return;
        }
    }
    out.push(0);
    out.extend_from_slice(&v.to_le_bytes());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips_at_every_width_boundary() {
        for n in 1..=8u32 {
            for v in [(1u64 << (7 * n)) - 1, 1u64 << (7 * n)] {
                let mut b = Vec::new();
                write_varint(&mut b, v);
                assert_eq!(read_varint(&b, 0), Some((v, b.len())), "{v}");
            }
        }
        let mut b = Vec::new();
        write_varint(&mut b, u64::MAX);
        assert_eq!((b.len(), read_varint(&b, 0)), (9, Some((u64::MAX, 9))));
    }

    #[test]
    fn known_encodings() {
        let enc = |v| {
            let mut b = Vec::new();
            write_varint(&mut b, v);
            b
        };
        assert_eq!(enc(0), vec![0x01]);
        assert_eq!(enc(1), vec![0x03]);
        assert_eq!(enc(127), vec![0xff]);
        assert_eq!(enc(128), vec![0x02, 0x02]);
    }

    #[test]
    fn truncated_input_is_none() {
        assert_eq!(read_varint(&[], 0), None);
        assert_eq!(read_varint(&[0x02], 0), None);
        assert_eq!(read_varint(&[0x00, 1, 2], 0), None);
    }
}
