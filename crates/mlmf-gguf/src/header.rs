//! The first stage: magic, version, and the two counts.

use crate::cursor::Cursor;
use crate::error::{GgufError, Stage};

/// Versions this build reads. v1 used 32-bit counts and is refused rather
/// than misparsed — see the crate docs.
const SUPPORTED: &[u32] = &[2, 3];

/// The fixed-size prologue of a GGUF file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Header {
    /// Declared format version.
    pub version: u32,
    /// Number of tensors the file declares.
    pub tensor_count: u64,
    /// Number of key-value pairs the file declares.
    pub kv_count: u64,
    /// Offset just past the header, where the KV block begins.
    pub end: u64,
}

/// Read magic, version and counts.
///
/// # Errors
///
/// [`GgufError::NotGguf`] if the magic is wrong — which is a different fact
/// from a malformed GGUF and is reported differently (R7).
/// [`GgufError::ByteSwapped`] if the version indicates opposite-endian
/// authorship. [`GgufError::UnsupportedVersion`] for v1 or anything newer
/// than this build. [`GgufError::Truncated`] or [`GgufError::Malformed`]
/// otherwise.
pub fn parse_header(cursor: &mut Cursor<'_>) -> Result<Header, GgufError> {
    let at = cursor.pos();
    let magic = cursor.take(4).map_err(|t| GgufError::Truncated {
        stage: Stage::Header,
        offset: at,
        needed: t.needed,
        available: t.available,
    })?;
    if magic != b"GGUF" {
        let mut found = [0u8; 4];
        found.copy_from_slice(magic);
        return Err(GgufError::NotGguf { found });
    }

    let at = cursor.pos();
    let version = cursor.u32().map_err(|t| GgufError::Truncated {
        stage: Stage::Header,
        offset: at,
        needed: t.needed,
        available: t.available,
    })?;

    // llama.cpp's own sniff (gguf.cpp:497). Checked BEFORE the supported
    // list, so a byte-swapped v3 is reported as byte-swapped rather than as
    // "unsupported version 50331648".
    if version & 0x0000_FFFF == 0 {
        return Err(GgufError::ByteSwapped {
            raw_version: version,
        });
    }
    if !SUPPORTED.contains(&version) {
        return Err(GgufError::UnsupportedVersion { version });
    }

    let tensor_count = read_count(cursor, "tensor count")?;
    let kv_count = read_count(cursor, "key-value count")?;

    Ok(Header {
        version,
        tensor_count,
        kv_count,
        end: cursor.pos(),
    })
}

/// Read one `i64` count and refuse a negative.
///
/// The counts are signed on the wire. Reinterpreting -1 as `u64` yields
/// 18446744073709551615, and a parser that then loops that many times has
/// turned eight bytes of a header into a hang.
fn read_count(cursor: &mut Cursor<'_>, what: &str) -> Result<u64, GgufError> {
    let at = cursor.pos();
    let raw = cursor.i64().map_err(|t| GgufError::Truncated {
        stage: Stage::Header,
        offset: at,
        needed: t.needed,
        available: t.available,
    })?;
    u64::try_from(raw).map_err(|_| GgufError::Malformed {
        stage: Stage::Header,
        offset: at,
        detail: format!("{what} is negative: {raw}"),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cursor::Cursor;
    use crate::error::{GgufError, Stage};

    /// A well-formed v3 header declaring `n_tensors` and `n_kv`.
    fn header_bytes(version: u32, n_tensors: i64, n_kv: i64) -> Vec<u8> {
        let mut v = Vec::new();
        v.extend_from_slice(b"GGUF");
        v.extend_from_slice(&version.to_le_bytes());
        v.extend_from_slice(&n_tensors.to_le_bytes());
        v.extend_from_slice(&n_kv.to_le_bytes());
        v
    }

    #[test]
    fn reads_a_well_formed_v3_header() {
        let b = header_bytes(3, 291, 42);
        let mut c = Cursor::new(&b);
        let h = parse_header(&mut c).expect("valid header");
        assert_eq!(h.version, 3);
        assert_eq!(h.tensor_count, 291);
        assert_eq!(h.kv_count, 42);
        assert_eq!(h.end, 24, "magic 4 + version 4 + two i64 = 24");
    }

    #[test]
    fn a_file_that_is_not_a_gguf_says_so_rather_than_reporting_corruption() {
        // R7. This is the residue of the consumer's own extension check: a
        // file NAMED .gguf whose magic is something else. "Not what its name
        // claims" and "malformed GGUF" are different operator remedies, and
        // collapsing them makes the warning hedge across both.
        let mut b = header_bytes(3, 0, 0);
        b[0..4].copy_from_slice(b"PK\x03\x04"); // a zip, the common mistake
        let mut c = Cursor::new(&b);
        match parse_header(&mut c).unwrap_err() {
            GgufError::NotGguf { found } => assert_eq!(&found, b"PK\x03\x04"),
            other => panic!("expected NotGguf, got {other:?}"),
        }
    }

    #[test]
    fn a_byte_swapped_file_is_named_rather_than_reported_as_a_huge_version() {
        // llama.cpp's own check (gguf.cpp:497): version 3 read with the
        // wrong endianness is 0x03000000. Without this the error says
        // "unsupported version 50331648", which sends the reader looking
        // for a version that has never existed.
        let mut b = Vec::new();
        b.extend_from_slice(b"GGUF");
        b.extend_from_slice(&0x0300_0000u32.to_le_bytes());
        b.extend_from_slice(&0i64.to_le_bytes());
        b.extend_from_slice(&0i64.to_le_bytes());
        let mut c = Cursor::new(&b);
        match parse_header(&mut c).unwrap_err() {
            GgufError::ByteSwapped { raw_version } => assert_eq!(raw_version, 0x0300_0000),
            other => panic!("expected ByteSwapped, got {other:?}"),
        }
    }

    #[test]
    fn v1_is_refused_by_version_rather_than_misparsed() {
        // v1 used 32-bit counts. Parsing it with 64-bit reads produces
        // plausible-looking garbage rather than an error, which is worse
        // than refusing. Out of scope for this crate; the error says which
        // version so the message is actionable.
        let b = header_bytes(1, 0, 0);
        let mut c = Cursor::new(&b);
        match parse_header(&mut c).unwrap_err() {
            GgufError::UnsupportedVersion { version } => assert_eq!(version, 1),
            other => panic!("expected UnsupportedVersion, got {other:?}"),
        }
    }

    #[test]
    fn v2_and_v3_are_both_accepted() {
        // v2 differs from v3 only below the header, and the corpus contains
        // one (ggml-vocab-aquila.gguf). Refusing it would reject real files.
        for v in [2u32, 3] {
            let b = header_bytes(v, 1, 1);
            let mut c = Cursor::new(&b);
            assert_eq!(parse_header(&mut c).expect("accepted").version, v);
        }
    }

    #[test]
    fn a_future_version_is_refused_with_its_number() {
        let b = header_bytes(4, 0, 0);
        let mut c = Cursor::new(&b);
        assert!(matches!(
            parse_header(&mut c).unwrap_err(),
            GgufError::UnsupportedVersion { version: 4 }
        ));
    }

    #[test]
    fn a_negative_count_is_malformed_rather_than_a_huge_unsigned_one() {
        // The counts are i64 on the wire. Reading -1 as u64 gives
        // 18446744073709551615, and a parser that then loops that many
        // times has turned eight bytes into a hang.
        let b = header_bytes(3, -1, 0);
        let mut c = Cursor::new(&b);
        // Whole-value comparison, for two reasons. A chain of field
        // assertions carries ordering bias — one on `stage` above one on
        // `detail` means a mutation touching only `detail` is masked or
        // misattributed. And this test's own sabotage stops the error being
        // produced at all, so it panics at `unwrap_err` before any inner
        // assertion is reached: those assertions were never proven live.
        assert_eq!(
            parse_header(&mut c).unwrap_err(),
            GgufError::Malformed {
                stage: Stage::Header,
                offset: 8,
                detail: "tensor count is negative: -1".to_string(),
            }
        );
    }

    #[test]
    fn a_truncated_header_reports_the_stage_and_the_offset() {
        let b = b"GGUF\x03\x00\x00\x00\x01".to_vec(); // 9 bytes: magic, version, one stray
        let mut c = Cursor::new(&b);
        // Whole-value comparison rather than a chain of field assertions.
        // Chaining means a transposition of `needed` and `available` — a
        // plausible one-line slip, since they are adjacent in the
        // construction — trips the `needed` assertion and the `available`
        // one never runs, so the failure names the wrong field.
        assert_eq!(
            parse_header(&mut c).unwrap_err(),
            GgufError::Truncated {
                stage: Stage::Header,
                offset: 8, // the tensor count starts at byte 8
                needed: 8,
                available: 1,
            }
        );
    }

    #[test]
    fn an_empty_slice_is_not_gguf_rather_than_a_panic() {
        let mut c = Cursor::new(&[]);
        assert!(matches!(
            parse_header(&mut c).unwrap_err(),
            GgufError::Truncated {
                stage: Stage::Header,
                ..
            }
        ));
    }

    #[test]
    fn a_wrong_magic_sharing_gguf_s_first_byte_is_still_not_a_gguf() {
        // The positive control for the magic check, kept as its own test so
        // that the claim and its control are two names rather than one name
        // in two states. "GGJT" is the real legacy ggml magic, so this is
        // the file a user actually holds when they hit this path.
        let mut b = header_bytes(3, 0, 0);
        b[0..4].copy_from_slice(b"GGJT");
        let mut c = Cursor::new(&b);
        match parse_header(&mut c).unwrap_err() {
            GgufError::NotGguf { found } => assert_eq!(&found, b"GGJT"),
            other => panic!("expected NotGguf, got {other:?}"),
        }
    }
}
