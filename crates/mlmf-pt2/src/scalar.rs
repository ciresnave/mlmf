// SPDX-License-Identifier: MIT OR Apache-2.0
//! torch's `ScalarType` codes, as `torch._export.serde.schema` numbers them.
//!
//! These integers are what a `.pt2` weights config stores for a tensor's
//! dtype. The table was read from `torch/_export/serde/schema.py` in PyTorch
//! 2.14.1 (BSD-3-Clause; see NOTICE). A code with no [`DType`] here (the
//! half-precision complex type, the `fnuz` and `e8m0` float8 variants, or a
//! code a newer torch adds) is returned as an unmapped code, not guessed at.

use mlmf_core::DType;

/// The [`DType`] torch's `ScalarType` code `code` denotes, if this build has one.
#[must_use]
pub fn dtype_of_scalar_type(code: i64) -> Option<DType> {
    Some(match code {
        1 => DType::U8,
        2 => DType::I8,
        3 => DType::I16,
        4 => DType::I32,
        5 => DType::I64,
        6 => DType::F16,
        7 => DType::F32,
        8 => DType::F64,
        10 => DType::C64,
        11 => DType::C128,
        12 => DType::Bool,
        13 => DType::BF16,
        28 => DType::U16,
        29 => DType::F8E4M3,
        30 => DType::F8E5M2,
        34 => DType::U32,
        35 => DType::U64,
        _ => return None,
    })
}

/// The `ScalarType` code for a [`DType`], the inverse of
/// [`dtype_of_scalar_type`].
#[must_use]
pub fn scalar_type_of(dtype: DType) -> Option<i64> {
    Some(match dtype {
        DType::U8 => 1,
        DType::I8 => 2,
        DType::I16 => 3,
        DType::I32 => 4,
        DType::I64 => 5,
        DType::F16 => 6,
        DType::F32 => 7,
        DType::F64 => 8,
        DType::C64 => 10,
        DType::C128 => 11,
        DType::Bool => 12,
        DType::BF16 => 13,
        DType::U16 => 28,
        DType::F8E4M3 => 29,
        DType::F8E5M2 => 30,
        DType::U32 => 34,
        DType::U64 => 35,
        _ => return None,
    })
}
