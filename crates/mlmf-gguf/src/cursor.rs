//! Bounds-checked little-endian reads over a byte slice.

/// A read that ran off the end of the slice.
///
/// Carries both numbers because "failed to read" does not distinguish a
/// truncated download from a file whose declared lengths are nonsense, and
/// those are different operator problems.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Truncated {
    /// Bytes the read required.
    pub needed: u64,
    /// Bytes actually available from the current position.
    pub available: u64,
}

/// A bounds-checked little-endian reader over borrowed bytes.
///
/// Every method that can run off the end returns [`Truncated`] and **leaves
/// the position unchanged**, so the caller can report the offset at which
/// the file stopped making sense.
#[derive(Debug)]
pub struct Cursor<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    /// A cursor at the start of `bytes`.
    #[must_use]
    pub fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, pos: 0 }
    }

    /// Current offset from the start of the slice.
    #[must_use]
    pub fn pos(&self) -> u64 {
        self.pos as u64
    }

    /// Bytes remaining.
    fn remaining(&self) -> u64 {
        (self.bytes.len() - self.pos) as u64
    }

    /// Move to an absolute position.
    ///
    /// # Errors
    ///
    /// [`Truncated`] if `pos` is past the end. The end itself is legal.
    pub fn seek(&mut self, pos: u64) -> Result<(), Truncated> {
        let want = usize::try_from(pos).map_err(|_| Truncated {
            needed: pos,
            available: self.bytes.len() as u64,
        })?;
        if want > self.bytes.len() {
            return Err(Truncated {
                needed: pos,
                available: self.bytes.len() as u64,
            });
        }
        self.pos = want;
        Ok(())
    }

    /// Borrow the next `n` bytes.
    ///
    /// # Errors
    ///
    /// [`Truncated`] if fewer than `n` bytes remain. **Checked before any
    /// allocation**, so a declared length of `u64::MAX` costs a comparison
    /// rather than an out-of-memory abort.
    pub fn take(&mut self, n: u64) -> Result<&'a [u8], Truncated> {
        if n > self.remaining() {
            return Err(Truncated {
                needed: n,
                available: self.remaining(),
            });
        }
        let n = n as usize; // <= remaining, so it fits
        let out = &self.bytes[self.pos..self.pos + n];
        self.pos += n;
        Ok(out)
    }
}

/// Generates the fixed-width little-endian readers.
///
/// A macro rather than eight hand-written near-identical bodies: the
/// bounds check and the position update must be the same in all of them,
/// and eight copies is eight chances for one to differ.
macro_rules! fixed_width {
    ($($name:ident => $ty:ty),* $(,)?) => {$(
        impl Cursor<'_> {
            #[doc = concat!("Read a little-endian `", stringify!($ty), "`.")]
            ///
            /// # Errors
            ///
            /// [`Truncated`] if too few bytes remain; the position is unchanged.
            pub fn $name(&mut self) -> Result<$ty, Truncated> {
                const N: usize = core::mem::size_of::<$ty>();
                let raw = self.take(N as u64)?;
                let mut buf = [0u8; N];
                buf.copy_from_slice(raw);
                Ok(<$ty>::from_le_bytes(buf))
            }
        }
    )*};
}

fixed_width! {
    u8 => u8, u16 => u16, u32 => u32, u64 => u64,
    i64 => i64, f32 => f32, f64 => f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_generated_reader_reads_its_own_width() {
        // All seven, by identity rather than by sampling two. The macro
        // guarantees the SHAPE of the generated bodies is identical; it does
        // not guarantee the type list is right. `u64 => u32` in the
        // invocation would produce a method named `u64` that reads four
        // bytes, and no amount of testing `u16` and `u32` would notice.
        //
        // `u64` matters most: it is the width GGUF uses for every string
        // length and every count, so Task 5 and Task 6 build directly on it.
        let b = [0x11u8, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88];
        assert_eq!(Cursor::new(&b).u8().unwrap(), 0x11);
        assert_eq!(Cursor::new(&b).u16().unwrap(), 0x2211);
        assert_eq!(Cursor::new(&b).u32().unwrap(), 0x4433_2211);
        assert_eq!(Cursor::new(&b).u64().unwrap(), 0x8877_6655_4433_2211);
        assert_eq!(
            Cursor::new(&b).i64().unwrap(),
            0x8877_6655_4433_2211u64 as i64
        );

        // And each advances by its own width. A reader that returns the
        // right value after consuming the wrong number of bytes leaves the
        // cursor desynchronised, and the symptom appears at the NEXT field
        // rather than at this one.
        let mut c = Cursor::new(&b);
        c.u8().unwrap();
        assert_eq!(c.pos(), 1, "u8");
        let mut c = Cursor::new(&b);
        c.u16().unwrap();
        assert_eq!(c.pos(), 2, "u16");
        let mut c = Cursor::new(&b);
        c.u32().unwrap();
        assert_eq!(c.pos(), 4, "u32");
        let mut c = Cursor::new(&b);
        c.u64().unwrap();
        assert_eq!(c.pos(), 8, "u64");
        let mut c = Cursor::new(&b);
        c.i64().unwrap();
        assert_eq!(c.pos(), 8, "i64");
        let mut c = Cursor::new(&b);
        c.f32().unwrap();
        assert_eq!(c.pos(), 4, "f32");
        let mut c = Cursor::new(&b);
        c.f64().unwrap();
        assert_eq!(c.pos(), 8, "f64");
    }

    #[test]
    fn a_short_read_reports_what_it_needed_and_what_was_there() {
        // Not just "failed": a truncated file is common and the numbers are
        // what let an operator tell a truncated download from a corrupt one.
        let bytes = [0x01u8, 0x02];
        let mut c = Cursor::new(&bytes);
        // One comparison over the whole error. Chaining `needed` then
        // `available` means a transposition in the construction — the two
        // sit adjacent, so it is a plausible slip — trips the first
        // assertion and the second never runs, blaming the wrong field.
        assert_eq!(
            c.u32().unwrap_err(),
            Truncated {
                needed: 4,
                available: 2
            }
        );
        // Position is a property of the cursor rather than of the error, so
        // it cannot fold into the comparison above and stays its own check.
        assert_eq!(c.pos(), 0);
    }

    #[test]
    fn take_borrows_rather_than_copies() {
        let bytes = [0xAAu8; 16];
        let mut c = Cursor::new(&bytes);
        let s = c.take(8).unwrap();
        assert_eq!(s.len(), 8);
        // Same allocation, not a copy — this is what lets a 15 MB KV block
        // be indexed without being duplicated.
        assert_eq!(s.as_ptr() as usize, bytes.as_ptr() as usize);
        assert_eq!(c.pos(), 8);
    }

    #[test]
    fn a_length_that_cannot_fit_the_file_fails_before_allocating() {
        // The adversarial case: a declared string length of u64::MAX. A
        // reader that trusts it and allocates first is a denial of service
        // triggered by four bytes of a header.
        let bytes = [0u8; 8];
        let mut c = Cursor::new(&bytes);
        assert_eq!(
            c.take(u64::MAX).unwrap_err(),
            Truncated {
                needed: u64::MAX,
                available: 8
            }
        );
        // The same guarantee its sibling short-read test pins: a refused
        // read must not move the cursor, or every later error reports an
        // offset that is wrong by however much the failed read consumed.
        assert_eq!(c.pos(), 0);
    }

    #[test]
    fn seek_refuses_a_position_past_the_end() {
        let bytes = [0u8; 8];
        let mut c = Cursor::new(&bytes);
        c.seek(8).expect("the end is a legal position");
        assert_eq!(c.pos(), 8);
        assert!(c.seek(9).is_err());
        // A refused seek must not move the cursor.
        assert_eq!(c.pos(), 8);
    }

    #[test]
    fn floats_are_bit_exact_not_approximately_decoded() {
        // Compare BITS, and pick values that a lossy route would damage.
        //
        // A first draft used -1.5, which is exactly representable in f32,
        // f64 and decimal — so it survives a detour through any of them and
        // the assertion could not distinguish `from_le_bytes` from the very
        // thing its comment warned against. It passed for an uninteresting
        // reason.
        let third = f32::from_bits(0x3EAA_AAAB); // nearest f32 to 1/3
        assert_eq!(
            Cursor::new(&third.to_le_bytes()).f32().unwrap().to_bits(),
            0x3EAA_AAAB
        );

        // A subnormal. An implementation that normalises on conversion —
        // or runs with flush-to-zero — turns this into 0.0, and comparing
        // bits is what catches it. Comparing values would too, but only
        // because 0.0 != this; comparing bits also catches a sign or
        // payload change that compares equal as a float.
        let subnormal = f32::from_bits(0x0000_0001);
        assert_eq!(
            Cursor::new(&subnormal.to_le_bytes())
                .f32()
                .unwrap()
                .to_bits(),
            0x0000_0001
        );

        let w = f64::from_bits(0x3FD5_5555_5555_5555); // nearest f64 to 1/3
        assert_eq!(
            Cursor::new(&w.to_le_bytes()).f64().unwrap().to_bits(),
            0x3FD5_5555_5555_5555
        );
    }
}
