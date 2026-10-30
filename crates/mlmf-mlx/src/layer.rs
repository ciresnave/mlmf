// SPDX-License-Identifier: MIT OR Apache-2.0
//! Locating MLX-packed linear layers in an already-open container.
//!
//! One logical layer is `<p>.weight` (`U32`, `[out, in*bits/32]`),
//! `<p>.scales` and `<p>.biases` (both `[out, in/group_size]`), plus the
//! Linear's own optional `<p>.bias`. See the crate docs for the measured
//! shapes. A layer is discovered by its `.scales` tensor: `.weight` alone
//! is also what an unquantized layer is called.

use std::fmt;

use mlmf_core::{DType, Encoding, TensorContainer, TensorDescriptor};

/// Why `locate_layers` refused to run: a bad CALL parameter. A problem
/// with what a FILE declares lands in the [`LocateReport`] instead, so one
/// bad layer never aborts every other layer's result.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LocateError {
    message: String,
}

impl fmt::Display for LocateError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for LocateError {}

impl LocateError {
    fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

/// One MLX-packed linear layer, located and geometrically validated.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PackedLinearLayer {
    /// The shared name prefix, e.g. `model.layers.0.self_attn.q_proj`.
    pub prefix: String,
    /// The packed weight tensor's full name.
    pub weight: String,
    /// The per-group scale tensor's full name.
    pub scales: String,
    /// The per-group affine zero-point tensor's full name (`.biases`).
    pub biases: String,
    /// The Linear's own bias (`.bias`), if present.
    pub bias: Option<String>,
    /// Logical input features: packed column count x pack factor.
    pub in_features: u64,
    /// Logical output features: `weight`'s row count.
    pub out_features: u64,
    /// `in_features / group_size`.
    pub group_count: u64,
}

/// Every candidate layer this scan found, sorted into three outcomes.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct LocateReport {
    /// Fully located, geometrically consistent layers, sorted by prefix.
    pub layers: Vec<PackedLinearLayer>,
    /// Prefixes with `.scales` but no `.weight` or no `.biases`. Sorted.
    pub incomplete: Vec<String>,
    /// Prefixes with every tensor present but inconsistent for the given
    /// `(bits, group_size)`: `(prefix, reason)`, sorted by prefix.
    pub malformed: Vec<(String, String)>,
}

/// Locate every MLX-packed linear layer `container` declares.
///
/// # Errors
///
/// `bits == 0`, `bits` does not evenly divide 32, or `group_size == 0`:
/// properties of the CALL, not of the container.
pub fn locate_layers(
    container: &dyn TensorContainer,
    bits: u64,
    group_size: u64,
) -> Result<LocateReport, LocateError> {
    if bits == 0 {
        return Err(LocateError::new("bits must be nonzero"));
    }
    if group_size == 0 {
        return Err(LocateError::new("group_size must be nonzero"));
    }
    if !32u64.is_multiple_of(bits) {
        return Err(LocateError::new(format!(
            "bits={bits} does not evenly divide a 32-bit pack word"
        )));
    }
    let pack_factor = 32 / bits;

    let mut prefixes: Vec<&str> = container
        .tensors()
        .iter()
        .filter_map(|d| d.name.strip_suffix(".scales"))
        .collect();
    prefixes.sort_unstable();

    let mut report = LocateReport::default();
    for prefix in prefixes {
        let (weight, scales, biases) = (
            container.tensor(&format!("{prefix}.weight")),
            container.tensor(&format!("{prefix}.scales")),
            container.tensor(&format!("{prefix}.biases")),
        );
        let (Some(weight), Some(scales), Some(biases)) = (weight, scales, biases) else {
            report.incomplete.push(prefix.to_string());
            continue;
        };
        let bias_name = format!("{prefix}.bias");
        let parts = Parts {
            weight,
            scales,
            biases,
        };
        match build_layer(prefix, &parts, pack_factor, group_size) {
            Ok(mut layer) => {
                layer.bias = container.tensor(&bias_name).map(|_| bias_name);
                report.layers.push(layer);
            }
            Err(reason) => report.malformed.push((prefix.to_string(), reason)),
        }
    }
    Ok(report)
}

/// The three descriptors one layer must have.
struct Parts<'a> {
    weight: &'a TensorDescriptor,
    scales: &'a TensorDescriptor,
    biases: &'a TensorDescriptor,
}

/// Validate one candidate's geometry, or name the first guard it fails.
fn build_layer(
    prefix: &str,
    parts: &Parts<'_>,
    pack_factor: u64,
    group_size: u64,
) -> Result<PackedLinearLayer, String> {
    let (out_features, packed_cols) = weight_guard(parts.weight)?;
    let in_features = packed_cols.checked_mul(pack_factor).ok_or_else(|| {
        format!(
            "weight's packed column count {packed_cols} * pack factor {pack_factor} overflows u64"
        )
    })?;
    let group_count = group_count_guard(in_features, group_size)?;
    let expected = [out_features as usize, group_count as usize];
    shape_guard("scales", parts.scales, expected)?;
    shape_guard("biases", parts.biases, expected)?;
    Ok(PackedLinearLayer {
        prefix: prefix.to_string(),
        weight: parts.weight.name.clone(),
        scales: parts.scales.name.clone(),
        biases: parts.biases.name.clone(),
        bias: None,
        in_features,
        out_features,
        group_count,
    })
}

/// Guard: `weight` is a rank-2, nonzero, `U32` tensor. Returns
/// `(out_features, packed_cols)`. A `[N, 0]` shape is legal for a parsed
/// container, so zero is checked explicitly.
fn weight_guard(weight: &TensorDescriptor) -> Result<(u64, u64), String> {
    if weight.encoding != Encoding::Dense(DType::U32) {
        return Err(format!(
            "weight encoding {:?} is not Dense(U32)",
            weight.encoding
        ));
    }
    let dims = weight.shape.dims();
    if dims.len() != 2 {
        return Err(format!("weight has rank {}, expected 2", dims.len()));
    }
    if dims[0] == 0 || dims[1] == 0 {
        return Err(format!("weight shape {dims:?} has a zero dimension"));
    }
    Ok((dims[0] as u64, dims[1] as u64))
}

/// Guard: `in_features` is a whole number of `group_size` groups.
fn group_count_guard(in_features: u64, group_size: u64) -> Result<u64, String> {
    if !in_features.is_multiple_of(group_size) {
        return Err(format!(
            "in_features {in_features} is not a whole number of groups of {group_size}"
        ));
    }
    Ok(in_features / group_size)
}

/// Guard: a per-group tensor is exactly `[out_features, group_count]`.
fn shape_guard(name: &str, t: &TensorDescriptor, expected: [usize; 2]) -> Result<(), String> {
    if t.shape.dims() != expected {
        return Err(format!(
            "{name} shape {:?} does not match the expected {expected:?}",
            t.shape.dims()
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::borrow::Cow;

    use mlmf_core::{Result, Shape};

    use super::*;

    struct FakeContainer(Vec<TensorDescriptor>);

    impl TensorContainer for FakeContainer {
        fn tensors(&self) -> &[TensorDescriptor] {
            &self.0
        }
        fn tensor_bytes(&self, _d: &TensorDescriptor) -> Result<Cow<'_, [u8]>> {
            unreachable!("locate_layers never reads tensor bytes")
        }
    }

    fn d(name: &str, dims: &[usize], dtype: DType) -> TensorDescriptor {
        TensorDescriptor {
            name: name.to_string(),
            shape: Shape::new(dims.iter().copied()),
            encoding: Encoding::Dense(dtype),
            bytes: 0..1,
        }
    }

    const P: &str = "model.layers.0.self_attn.q_proj";

    /// q_proj of mlx-community/Qwen2.5-0.5B-Instruct-4bit, shapes from its
    /// real safetensors header (2026-10-30): in 896, out 896, 4-bit, group 64.
    fn real() -> Vec<TensorDescriptor> {
        vec![
            d(&format!("{P}.weight"), &[896, 112], DType::U32),
            d(&format!("{P}.scales"), &[896, 14], DType::F16),
            d(&format!("{P}.biases"), &[896, 14], DType::F16),
        ]
    }

    fn run(t: Vec<TensorDescriptor>) -> LocateReport {
        locate_layers(&FakeContainer(t), 4, 64).expect("valid parameters")
    }

    #[test]
    fn finds_the_real_q_proj_layer() {
        let r = run(real());
        assert!(r.incomplete.is_empty() && r.malformed.is_empty());
        let l = &r.layers[0];
        assert_eq!(
            (l.in_features, l.out_features, l.group_count),
            (896, 896, 14)
        );
        assert_eq!(l.biases, format!("{P}.biases"));
        assert_eq!(l.bias, None);
    }

    /// down_proj is the non-square case: in 4864, out 896 -- `[896, 608]`
    /// weight and `[896, 76]` scales, so in/out cannot be swapped silently.
    #[test]
    fn the_non_square_down_proj_has_in_and_out_the_right_way_round() {
        let p = "model.layers.0.mlp.down_proj";
        let r = run(vec![
            d(&format!("{p}.weight"), &[896, 608], DType::U32),
            d(&format!("{p}.scales"), &[896, 76], DType::F16),
            d(&format!("{p}.biases"), &[896, 76], DType::F16),
        ]);
        let l = &r.layers[0];
        assert_eq!((l.in_features, l.out_features), (4864, 896));
    }

    #[test]
    fn the_linear_bias_is_recorded_apart_from_biases() {
        let mut t = real();
        t.push(d(&format!("{P}.bias"), &[896], DType::F16));
        let l = run(t).layers.remove(0);
        assert_eq!(l.bias, Some(format!("{P}.bias")));
        assert_eq!(l.biases, format!("{P}.biases"));
    }

    #[test]
    fn a_plain_weight_without_scales_is_not_a_layer() {
        let r = run(vec![d("model.norm.weight", &[896], DType::F16)]);
        assert_eq!(r, LocateReport::default());
    }

    #[test]
    fn missing_biases_or_weight_is_incomplete() {
        let mut t = real();
        t.remove(2);
        assert_eq!(run(t).incomplete, vec![P.to_string()]);
        let mut t = real();
        t.remove(0);
        assert_eq!(run(t).incomplete, vec![P.to_string()]);
    }

    #[test]
    fn bad_call_parameters_are_errors() {
        let c = FakeContainer(real());
        assert!(locate_layers(&c, 0, 64).is_err());
        assert!(locate_layers(&c, 3, 64).is_err());
        assert!(locate_layers(&c, 4, 0).is_err());
    }

    fn malformed_reason(t: Vec<TensorDescriptor>) -> String {
        let mut r = run(t);
        assert!(r.layers.is_empty(), "a bad layer must not be located");
        r.malformed.remove(0).1
    }

    #[test]
    fn a_non_u32_weight_is_malformed() {
        let mut t = real();
        t[0] = d(&format!("{P}.weight"), &[896, 112], DType::I32);
        assert!(malformed_reason(t).contains("not Dense(U32)"));
    }

    #[test]
    fn rank_zero_dimension_and_group_faults_are_malformed() {
        let mut t = real();
        t[0] = d(&format!("{P}.weight"), &[896], DType::U32);
        assert!(malformed_reason(t).contains("rank 1"));
        let mut t = real();
        t[0] = d(&format!("{P}.weight"), &[896, 0], DType::U32);
        assert!(malformed_reason(t).contains("zero dimension"));
        let mut t = real();
        t[0] = d(&format!("{P}.weight"), &[896, 113], DType::U32);
        assert!(malformed_reason(t).contains("whole number of groups"));
    }

    #[test]
    fn overflowing_columns_are_malformed_not_a_panic() {
        let mut t = real();
        t[0] = d(&format!("{P}.weight"), &[896, usize::MAX], DType::U32);
        assert!(malformed_reason(t).contains("overflows"));
    }

    #[test]
    fn scales_and_biases_shapes_are_each_checked() {
        let mut t = real();
        t[1] = d(&format!("{P}.scales"), &[14, 896], DType::F16);
        assert!(malformed_reason(t).starts_with("scales shape"));
        let mut t = real();
        t[2] = d(&format!("{P}.biases"), &[896, 13], DType::F16);
        assert!(malformed_reason(t).starts_with("biases shape"));
    }
}
