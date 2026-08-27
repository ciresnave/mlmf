//! The first stage: an 8-byte length prefix and the JSON header it sizes.
//!
//! ```text
//! 8 bytes    u64 little-endian header length N
//! N bytes    JSON header
//! remainder  tensor data
//! ```
//!
//! Verified byte-exact against two real models, 491 tensors, by
//! `docs/superpowers/plans/safetensors_recon.py` — an independent reader
//! sharing no code with this crate. The falsification test was that the
//! furthest `data_offsets[1]`, plus 8, plus the header length, equals the
//! file length exactly; it did, on 2 of 2 files, one of them 2,200,119,864
//! bytes.
//!
//! # The length prefix is attacker-controlled
//!
//! Eight bytes chosen by whoever wrote the file decide how much this
//! function is about to address. At `u64::MAX` a naive `&bytes[8..8 + n]`
//! panics and a naive `Vec::with_capacity(n)` aborts the process, and
//! neither is a parse error a caller can handle.
//!
//! So [`parse_header`] bounds the declared length against the slice
//! **before** any slicing, any allocation and any conversion to `usize`.
//! `try_reserve` is deliberately not the mechanism: this project measured
//! `try_reserve(0xFFFF_FFFF)` on a `Vec<u64>` returning `Ok` and committing
//! 34 GB. An allocation guard that succeeds is not a guard. The explicit
//! bound is.

use crate::error::{SafetensorsError, Stage};

/// Bytes of little-endian `u64` at the start of every safetensors file.
const LENGTH_PREFIX: usize = 8;

/// The parsed safetensors header.
///
/// One header carries both the metadata and the tensor directory — see D2
/// in the plan — so this is what both later stages read from.
#[derive(Debug, Clone, PartialEq)]
pub struct Header {
    /// Length in bytes of the JSON header, exactly as the prefix declared
    /// it. Bounded against the slice before this value was trusted.
    pub header_len: u64,
    /// Offset of the first byte of tensor data: `8 + header_len`.
    ///
    /// This is safetensors' own base for `data_offsets`, and it is a
    /// **different base from GGUF's**, which derives its data start from a
    /// padded region boundary. D4 in the plan is about not confusing them.
    pub data_start: u64,
    /// The header's top-level object, exactly as declared.
    ///
    /// Crate-visible rather than public: `serde_json::Map` is a foreign
    /// type and MLMF does not put foreign types in its public API. The
    /// tensor directory and the metadata source read it from inside the
    /// crate; a consumer gets `TensorContainer` and `MetadataSource`.
    pub(crate) entries: serde_json::Map<String, serde_json::Value>,
}

/// Name the JSON kind of a value, for an error message that says what was
/// found instead of an object.
pub(crate) fn json_kind(value: &serde_json::Value) -> &'static str {
    match value {
        serde_json::Value::Null => "null",
        serde_json::Value::Bool(_) => "boolean",
        serde_json::Value::Number(_) => "number",
        serde_json::Value::String(_) => "string",
        serde_json::Value::Array(_) => "array",
        serde_json::Value::Object(_) => "object",
    }
}

/// Read the length prefix and the JSON header it sizes.
///
/// # Errors
///
/// [`SafetensorsError::Truncated`] if the slice is shorter than the 8-byte
/// prefix, or if the declared header length does not fit in what remains —
/// **checked before anything is sliced or allocated**.
/// [`SafetensorsError::Malformed`] for a declared length of zero, which
/// cannot be JSON, and for a header that is not valid JSON.
/// [`SafetensorsError::NotUtf8`] if the header bytes are not UTF-8.
/// [`SafetensorsError::NotAnObject`] if the header parses and its top level
/// is not an object.
pub fn parse_header(bytes: &[u8]) -> Result<Header, SafetensorsError> {
    let Some(prefix) = bytes.get(..LENGTH_PREFIX) else {
        return Err(SafetensorsError::Truncated {
            stage: Stage::Header,
            offset: 0,
            needed: LENGTH_PREFIX as u64,
            available: bytes.len() as u64,
        });
    };
    let mut le = [0u8; LENGTH_PREFIX];
    le.copy_from_slice(prefix);
    let header_len = u64::from_le_bytes(le);

    // Zero passes any bound trivially, so it gets its own refusal here
    // rather than downstream. An empty byte string is not JSON, and letting
    // serde report "EOF while parsing a value" would blame the JSON for
    // what the length prefix did.
    if header_len == 0 {
        return Err(SafetensorsError::Malformed {
            stage: Stage::Header,
            offset: 0,
            detail: "declared header length is zero".to_string(),
        });
    }

    // ---- THE BOUND ----------------------------------------------------
    // Everything past this point indexes or allocates using a number the
    // FILE chose. Nothing past this point runs until that number fits in
    // the bytes actually present. This is not belt-and-braces around a
    // fallible allocator: `try_reserve(0xFFFF_FFFF)` on a `Vec<u64>` was
    // measured in this project returning `Ok` and committing 34 GB, so an
    // allocation guard is not the guard. The explicit bound is.
    let available = bytes.len() - LENGTH_PREFIX;
    let too_big = || SafetensorsError::Truncated {
        stage: Stage::Header,
        offset: LENGTH_PREFIX as u64,
        needed: header_len,
        available: available as u64,
    };
    // On a 32-bit host a `u64` length can exceed `usize` outright, which is
    // the same fact -- it does not fit in this file -- reached by a
    // different route. On a 64-bit host this conversion always succeeds,
    // `u64::MAX` included, which is precisely why the comparison below
    // cannot be left to it.
    let Ok(declared) = usize::try_from(header_len) else {
        return Err(too_big());
    };
    if declared > available {
        return Err(too_big());
    }
    let end = LENGTH_PREFIX + declared;
    // -------------------------------------------------------------------

    let text =
        std::str::from_utf8(&bytes[LENGTH_PREFIX..end]).map_err(|e| SafetensorsError::NotUtf8 {
            stage: Stage::Header,
            offset: (LENGTH_PREFIX + e.valid_up_to()) as u64,
        })?;

    let value: serde_json::Value =
        serde_json::from_str(text).map_err(|e| SafetensorsError::Malformed {
            stage: Stage::Header,
            offset: LENGTH_PREFIX as u64,
            detail: e.to_string(),
        })?;

    match value {
        serde_json::Value::Object(entries) => Ok(Header {
            header_len,
            data_start: end as u64,
            entries,
        }),
        other => Err(SafetensorsError::NotAnObject {
            stage: Stage::Header,
            offset: LENGTH_PREFIX as u64,
            found: json_kind(&other),
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The positive fixture: one `__metadata__` block and one tensor, in
    /// the shape the recon script found in both real files.
    const WELL_FORMED: &[u8] =
        br#"{"__metadata__":{"format":"pt"},"weight":{"dtype":"BF16","shape":[2,3],"data_offsets":[0,12]}}"#;

    /// The smallest legal header: an object with no tensors.
    const EMPTY_OBJECT: &[u8] = b"{}";

    /// A safetensors image: a truthful 8-byte length prefix, then `json`.
    fn image(json: &[u8]) -> Vec<u8> {
        let mut v = (json.len() as u64).to_le_bytes().to_vec();
        v.extend_from_slice(json);
        v
    }

    /// The same, with the prefix declaring `declared` rather than the truth.
    fn image_declaring(declared: u64, json: &[u8]) -> Vec<u8> {
        let mut v = declared.to_le_bytes().to_vec();
        v.extend_from_slice(json);
        v
    }

    // Every case below compares the WHOLE `Result` in one `assert_eq!`,
    // rather than `unwrap_err()` followed by field assertions. Two reasons,
    // both learned here. A chain of field assertions carries ordering bias:
    // a transposition of `needed` and `available` trips the first assertion
    // and the second never runs, so the failure names the wrong field. And
    // `unwrap_err()` PANICS when a sabotage stops the error being produced
    // at all -- a panic before any comparison, which proves nothing about
    // the comparisons the test exists to make.

    #[test]
    fn a_well_formed_header_yields_its_length_its_data_start_and_its_entries() {
        let mut entries = serde_json::Map::new();
        entries.insert(
            "__metadata__".to_string(),
            serde_json::json!({"format": "pt"}),
        );
        entries.insert(
            "weight".to_string(),
            serde_json::json!({
                "dtype": "BF16",
                "shape": [2, 3],
                "data_offsets": [0, 12],
            }),
        );

        assert_eq!(
            parse_header(&image(WELL_FORMED)),
            Ok(Header {
                header_len: 94,
                // 8 + 94. The literal is spelled out rather than computed
                // from `header_len`, because a `data_start` derived from
                // the same expression the implementation uses would agree
                // with a wrong base as readily as with the right one.
                data_start: 102,
                entries,
            })
        );
    }

    #[test]
    fn a_length_prefix_larger_than_the_file_is_truncated_not_an_allocation() {
        // The attacker case. `u64::MAX` is both the largest declarable
        // length and, on a 64-bit host, exactly `usize::MAX` -- so
        // `usize::try_from` SUCCEEDS and only the explicit bound stands
        // between this input and `8 + usize::MAX` overflowing, or a
        // 16-exabyte allocation. `needed` is reported verbatim so the
        // operator sees the absurd number rather than a saturated one.
        assert_eq!(
            parse_header(&image_declaring(u64::MAX, EMPTY_OBJECT)),
            Err(SafetensorsError::Truncated {
                stage: Stage::Header,
                offset: 8,
                needed: u64::MAX,
                available: 2,
            })
        );
    }

    #[test]
    fn a_length_prefix_one_byte_past_the_end_is_truncated() {
        // The off-by-one companion to the `u64::MAX` case. A bound written
        // `>=` instead of `>`, or one comparing against `bytes.len()`
        // rather than against what remains after the prefix, is invisible
        // to an input that overshoots by exabytes and caught by this one.
        assert_eq!(
            parse_header(&image_declaring(3, EMPTY_OBJECT)),
            Err(SafetensorsError::Truncated {
                stage: Stage::Header,
                offset: 8,
                needed: 3,
                available: 2,
            })
        );
    }

    #[test]
    fn a_length_prefix_exactly_filling_the_file_is_accepted() {
        // Two claims in one: the smallest legal header is an object with
        // no tensors, and the largest legal length -- one that leaves zero
        // bytes of tensor data -- must not be refused. Without the second,
        // `>=` written for `>` passes every other test in this module.
        assert_eq!(
            parse_header(&image(EMPTY_OBJECT)),
            Ok(Header {
                header_len: 2,
                data_start: 10,
                entries: serde_json::Map::new(),
            })
        );
    }

    #[test]
    fn a_declared_length_of_zero_is_refused_rather_than_read_as_an_empty_header() {
        // Zero passes any bound check trivially, so it needs its own
        // refusal. An empty byte string is not valid JSON, and letting
        // serde report "EOF while parsing a value" would blame the JSON
        // for what the length prefix did. The offset is 0 -- the prefix --
        // not 8, because that is where the wrong number is.
        assert_eq!(
            parse_header(&image_declaring(0, EMPTY_OBJECT)),
            Err(SafetensorsError::Malformed {
                stage: Stage::Header,
                offset: 0,
                detail: "declared header length is zero".to_string(),
            })
        );
    }

    #[test]
    fn a_slice_shorter_than_the_length_prefix_is_truncated_rather_than_a_panic() {
        assert_eq!(
            parse_header(&[0u8; 4]),
            Err(SafetensorsError::Truncated {
                stage: Stage::Header,
                offset: 0,
                needed: 8,
                available: 4,
            })
        );
    }

    #[test]
    fn an_empty_slice_is_truncated_rather_than_a_panic() {
        assert_eq!(
            parse_header(&[]),
            Err(SafetensorsError::Truncated {
                stage: Stage::Header,
                offset: 0,
                needed: 8,
                available: 0,
            })
        );
    }

    #[test]
    fn a_header_that_is_not_utf8_is_refused_at_the_offending_byte() {
        // 0xFF is not a legal UTF-8 byte in any position. `{"na` is four
        // valid bytes, so the first invalid one is at header offset 4 and
        // slice offset 12. Reported as its own kind: a header that is not
        // UTF-8 came from something that did not emit JSON, or emitted it
        // in another encoding, and neither is "this JSON has a syntax
        // error".
        let json = b"{\"na\xFFme\":[]}";
        assert_eq!(
            parse_header(&image(json)),
            Err(SafetensorsError::NotUtf8 {
                stage: Stage::Header,
                offset: 12,
            })
        );
    }

    #[test]
    fn valid_utf8_that_is_not_json_is_malformed_and_carries_serde_s_message() {
        // The `detail` is serde_json's own message, pinned as a literal.
        // It is the actionable half of the error -- it says WHAT was wrong
        // and where in the JSON -- and this repository commits Cargo.lock,
        // so it is deterministic here: serde_json 1.0.151. A version bump
        // that rewords it reddens this line, which is a true statement
        // about a changed dependency and is the reason to pin rather than
        // to assert something weaker.
        assert_eq!(
            parse_header(&image(b"{\"a\": }")),
            Err(SafetensorsError::Malformed {
                stage: Stage::Header,
                offset: 8,
                detail: "expected value at line 1 column 7".to_string(),
            })
        );
    }

    #[test]
    fn a_top_level_array_is_refused_rather_than_read_as_a_tensor_directory() {
        // Well-formed JSON, wrong shape. Distinct from `Malformed` because
        // an operator holding a JSON array holds a different kind of file,
        // not a corrupt safetensors one, and looking for corruption in it
        // wastes their time.
        assert_eq!(
            parse_header(&image(b"[\"weight\"]")),
            Err(SafetensorsError::NotAnObject {
                stage: Stage::Header,
                offset: 8,
                found: "array",
            })
        );
    }

    #[test]
    fn every_json_kind_has_a_name() {
        // `json_kind` feeds `NotAnObject.found`, and the array case above
        // exercises exactly one of its six arms. Pinned arm by arm, by
        // identity: a mapping that returned "array" for everything would
        // satisfy that one test.
        assert_eq!(
            [
                json_kind(&serde_json::Value::Null),
                json_kind(&serde_json::json!(true)),
                json_kind(&serde_json::json!(1)),
                json_kind(&serde_json::json!("s")),
                json_kind(&serde_json::json!([])),
                json_kind(&serde_json::json!({})),
            ],
            ["null", "boolean", "number", "string", "array", "object"]
        );
    }
}
