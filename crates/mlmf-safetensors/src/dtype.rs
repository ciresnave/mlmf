//! Safetensors dtype strings, mapped onto [`mlmf_core::DType`].
//!
//! Safetensors declares an element type as a **string** — `"BF16"`,
//! `"F8_E4M3"` — where GGUF declares a numeric ggml code. That is the
//! second instance of the seam's central question, one layer below
//! `MetaValue`: is [`mlmf_core::DType`] a neutral vocabulary, or is it
//! ggml's type table wearing a neutral name?
//!
//! Measured rather than argued: the fifteen dtype strings safetensors
//! defines map **one to one, onto and into**, the fifteen variants of
//! `DType::ALL`. Nothing had to be widened and nothing goes unused. That is
//! a stronger fit than `MetaValue` gets from this format, where twelve of
//! thirteen variants never appear.
//!
//! # Every safetensors tensor is [`mlmf_core::Encoding::Dense`]
//!
//! The format has no block-quantized types at all, so `Encoding::Blocked`
//! is unreachable from this crate. Stated here rather than discovered:
//! `Encoding`'s two arms are a GGUF/safetensors split, and a reader of this
//! crate should not have to work out which arm it lives on.
//!
//! # The one trap this module exists to hold shut
//!
//! `F8_E4M3` and `F8_E5M2` are the same width, the same kind, and mutually
//! byte-incompatible: reading one as the other yields finite floats with
//! wrong exponents rather than an error anyone would notice. This project's
//! notes recorded that exposure as dormant "until a format crate maps
//! declared type strings onto `DType`". **This is that crate**, so the
//! mapping is written arm by arm and must be tested arm by arm — a check
//! that compares widths is satisfied by a swapped pair.

use mlmf_core::DType;

/// The [`DType`] a safetensors `dtype` string declares.
///
/// `None` for a string this build does not know — never a guess and never a
/// default. A caller turns that into a report entry and omits the tensor;
/// fabricating a width would hand out a byte range for a tensor whose
/// extent is genuinely unknown.
///
/// Matched on the string exactly as declared, with no case folding: the
/// safetensors format defines these spellings, and accepting `"bf16"` would
/// be this crate guessing what a writer meant.
#[must_use]
pub fn dtype_of(declared: &str) -> Option<DType> {
    Some(match declared {
        "F64" => DType::F64,
        "F32" => DType::F32,
        "F16" => DType::F16,
        "BF16" => DType::BF16,
        // Adjacent on purpose, and the only two arms in this function that
        // a width-based test cannot tell apart.
        "F8_E4M3" => DType::F8E4M3,
        "F8_E5M2" => DType::F8E5M2,
        "I64" => DType::I64,
        "I32" => DType::I32,
        "I16" => DType::I16,
        "I8" => DType::I8,
        "U64" => DType::U64,
        "U32" => DType::U32,
        "U16" => DType::U16,
        "U8" => DType::U8,
        // Safetensors spells this one without the `EAN`, unlike every
        // other name here, which is why it is written out rather than
        // derived from the variant name.
        "BOOL" => DType::Bool,
        _ => return None,
    })
}
