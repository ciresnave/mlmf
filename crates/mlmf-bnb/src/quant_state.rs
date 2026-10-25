// SPDX-License-Identifier: MIT OR Apache-2.0
//! The `quant_state` JSON blob bitsandbytes stores beside a 4-bit weight.
//!
//! Every field is `Option`: a key the blob does not carry is `None`, never a
//! default, and nothing is judged here. In particular `quant_type` is
//! returned as a string whatever it says (`nf4`, `fp4`, or something newer);
//! deciding which codebook that implies, whether a blocksize divides a row,
//! or whether a shape has two dimensions belongs to the code that
//! dequantizes. A key that IS present with the wrong JSON type is an error,
//! not `None`: the file said something and it was not readable.

use crate::BnbError;

/// The fields of a bitsandbytes `quant_state` blob, as stored.
#[derive(Debug, Clone, PartialEq)]
pub struct BnbQuantState {
    /// `quant_type`, for example `"nf4"` or `"fp4"`.
    pub quant_type: Option<String>,
    /// `blocksize`: elements per first-level absmax.
    pub blocksize: Option<u64>,
    /// `shape` of the unquantized weight, every dimension as stored.
    pub shape: Option<Vec<u64>>,
    /// `dtype` of the unquantized weight, for example `"bfloat16"`.
    pub dtype: Option<String>,
    /// `nested_blocksize`: first-level absmax codes per second-level scale.
    pub nested_blocksize: Option<u64>,
    /// `nested_dtype`, for example `"float32"`.
    pub nested_dtype: Option<String>,
    /// `nested_offset`: the value bitsandbytes subtracted before nesting.
    pub nested_offset: Option<f64>,
    /// Every other top-level key, so a field this version does not know is
    /// visible rather than silently ignored.
    pub other_keys: Vec<String>,
}

type Map = serde_json::Map<String, serde_json::Value>;

fn bad(key: &'static str, expected: &'static str) -> BnbError {
    BnbError::QuantStateField { key, expected }
}

fn string(m: &Map, key: &'static str) -> Result<Option<String>, BnbError> {
    m.get(key).map_or(Ok(None), |v| {
        v.as_str()
            .map(|s| Some(s.to_string()))
            .ok_or_else(|| bad(key, "string"))
    })
}

fn count(m: &Map, key: &'static str) -> Result<Option<u64>, BnbError> {
    m.get(key).map_or(Ok(None), |v| {
        v.as_u64()
            .map(Some)
            .ok_or_else(|| bad(key, "non-negative integer"))
    })
}

fn float(m: &Map, key: &'static str) -> Result<Option<f64>, BnbError> {
    m.get(key).map_or(Ok(None), |v| {
        v.as_f64().map(Some).ok_or_else(|| bad(key, "number"))
    })
}

fn shape(m: &Map) -> Result<Option<Vec<u64>>, BnbError> {
    let Some(v) = m.get("shape") else {
        return Ok(None);
    };
    let dims = v
        .as_array()
        .ok_or_else(|| bad("shape", "array of integers"))?;
    dims.iter()
        .map(|d| d.as_u64().ok_or_else(|| bad("shape", "array of integers")))
        .collect::<Result<Vec<_>, _>>()
        .map(Some)
}

const KNOWN: [&str; 7] = [
    "quant_type",
    "blocksize",
    "shape",
    "dtype",
    "nested_blocksize",
    "nested_dtype",
    "nested_offset",
];

/// Parse a `quant_state` blob.
///
/// # Errors
///
/// [`BnbError::QuantStateJson`] if the bytes are not JSON or not an object;
/// [`BnbError::QuantStateField`] if a known key has the wrong JSON type.
pub fn parse_quant_state(json: &[u8]) -> Result<BnbQuantState, BnbError> {
    let value: serde_json::Value =
        serde_json::from_slice(json).map_err(|e| BnbError::QuantStateJson(e.to_string()))?;
    let m = value
        .as_object()
        .ok_or_else(|| BnbError::QuantStateJson("top level is not an object".to_string()))?;
    Ok(BnbQuantState {
        quant_type: string(m, "quant_type")?,
        blocksize: count(m, "blocksize")?,
        shape: shape(m)?,
        dtype: string(m, "dtype")?,
        nested_blocksize: count(m, "nested_blocksize")?,
        nested_dtype: string(m, "nested_dtype")?,
        nested_offset: float(m, "nested_offset")?,
        other_keys: m
            .keys()
            .filter(|k| !KNOWN.contains(&k.as_str()))
            .cloned()
            .collect(),
    })
}
