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
    fn reads_little_endian_widths_and_advances() {
        let bytes = [0x01u8, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08];
        let mut c = Cursor::new(&bytes);
        assert_eq!(c.pos(), 0);
        assert_eq!(c.u16().unwrap(), 0x0201);
        assert_eq!(c.pos(), 2);
        assert_eq!(c.u32().unwrap(), 0x06050403);
        assert_eq!(c.pos(), 6);
    }

    #[test]
    fn a_short_read_reports_what_it_needed_and_what_was_there() {
        // Not just "failed": a truncated file is common and the numbers are
        // what let an operator tell a truncated download from a corrupt one.
        let bytes = [0x01u8, 0x02];
        let mut c = Cursor::new(&bytes);
        let e = c.u32().unwrap_err();
        assert_eq!(e.needed, 4);
        assert_eq!(e.available, 2);
        // And the cursor must not have moved, so the caller can report the
        // offset the failure happened at.
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
        let e = c.take(u64::MAX).unwrap_err();
        assert_eq!(e.needed, u64::MAX);
        assert_eq!(e.available, 8);
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
        // f32::from_le_bytes, not a cast through f64 or a parse. Exact
        // equality is the right assertion here: any transformation at all
        // shows up as inequality.
        let v: f32 = -1.5;
        let bytes = v.to_le_bytes();
        assert_eq!(Cursor::new(&bytes).f32().unwrap(), v);
        let w: f64 = 1.0 / 3.0;
        let wb = w.to_le_bytes();
        assert_eq!(Cursor::new(&wb).f64().unwrap(), w);
    }
}
