//! Locating AWQ-packed linear layers in an already-open container.
//!
//! AWQ's quantization is **inter-tensor**, like GPTQ's: one logical
//! linear layer is `qweight`/`qzeros`/`scales`/optional `bias`, each a
//! plain `Encoding::Dense` array as far as the container is concerned.
//! Unlike GPTQ, `qweight`'s packing axis is the OUTPUT dimension
//! (`[in_features, out_features/8]`), not the input one — see this
//! crate's plan document for the real shapes this was verified against.

use std::fmt;

use mlmf_core::{TensorContainer, TensorDescriptor};

/// Why `locate_layers` refused to run at all.
///
/// Exists only for bad CALL-TIME parameters (`bits`/`group_size`). A
/// problem with what a FILE declares never reaches here — it lands in
/// [`LocateReport::incomplete`] or [`LocateReport::malformed`] instead, by
/// design: one bad layer must not abort every other layer's result (the
/// #97 lesson, carried over from `mlmf-gptq`).
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

/// One AWQ-packed linear layer, located and geometrically validated.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PackedLinearLayer {
    /// The shared name prefix, e.g. `model.layers.0.self_attn.q_proj`.
    pub prefix: String,
    /// The packed weight tensor's full name.
    pub qweight: String,
    /// The packed zero-point tensor's full name.
    pub qzeros: String,
    /// The per-group scale tensor's full name.
    pub scales: String,
    /// The bias tensor's full name, if present.
    pub bias: Option<String>,
    /// Logical input features — read directly off `qweight`'s row count
    /// (AWQ does not pack the input dimension).
    pub in_features: u64,
    /// Logical output features (`qweight`'s packed column count × pack
    /// factor), not the packed column count itself.
    pub out_features: u64,
    /// `in_features / group_size`.
    pub group_count: u64,
}

/// Every candidate layer this scan found, sorted into three outcomes.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct LocateReport {
    /// Fully located and geometrically consistent layers, sorted by
    /// prefix.
    pub layers: Vec<PackedLinearLayer>,
    /// Prefixes with a `.qweight` tensor but missing `.qzeros` or
    /// `.scales` entirely. Sorted.
    pub incomplete: Vec<String>,
    /// Prefixes with all required tensors present but geometrically
    /// inconsistent for the given `(bits, group_size)` — `(prefix,
    /// reason)`. Sorted by prefix.
    pub malformed: Vec<(String, String)>,
}

/// Locate every AWQ-packed linear layer `container` declares.
///
/// # Errors
///
/// `bits == 0`, `bits` does not evenly divide 32, or `group_size == 0`.
/// Properties of the CALL, not of the container.
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
        .filter_map(|d| d.name.strip_suffix(".qweight"))
        .collect();
    prefixes.sort_unstable();

    let mut report = LocateReport::default();

    for prefix in prefixes {
        let qweight_name = format!("{prefix}.qweight");
        let qzeros_name = format!("{prefix}.qzeros");
        let scales_name = format!("{prefix}.scales");

        let Some(qweight) = container.tensor(&qweight_name) else {
            continue; // unreachable: prefix was derived from this exact name
        };
        let (Some(qzeros), Some(scales)) = (
            container.tensor(&qzeros_name),
            container.tensor(&scales_name),
        ) else {
            report.incomplete.push(prefix.to_string());
            continue;
        };

        match build_layer(
            container,
            prefix,
            qweight_name,
            qzeros_name,
            scales_name,
            qweight,
            qzeros,
            scales,
            pack_factor,
            group_size,
        ) {
            Ok(layer) => report.layers.push(layer),
            Err(reason) => report.malformed.push((prefix.to_string(), reason)),
        }
    }

    Ok(report)
}

/// Validate one candidate's geometry and build its [`PackedLinearLayer`],
/// or name the first guard it fails. Each guard is its own function so
/// this one stays a straight-line sequence of checks rather than one long
/// nest of `if`s -- the per-guard split this crate's Codacy review asked
/// for (none of the guards themselves, or their messages, changed).
#[allow(clippy::too_many_arguments)]
fn build_layer(
    container: &dyn TensorContainer,
    prefix: &str,
    qweight_name: String,
    qzeros_name: String,
    scales_name: String,
    qweight: &TensorDescriptor,
    qzeros: &TensorDescriptor,
    scales: &TensorDescriptor,
    pack_factor: u64,
    group_size: u64,
) -> Result<PackedLinearLayer, String> {
    // Transposed from mlmf-gptq: AWQ packs the OUTPUT dimension, so
    // in_features is read directly and out_features is derived.
    let (in_features, packed_out_cols) = qweight_rank_guard(qweight)?;
    zero_dimension_guard(in_features, packed_out_cols)?;
    let out_features = out_features_guard(packed_out_cols, pack_factor)?;
    let group_count = group_count_guard(in_features, group_size)?;
    scales_shape_guard(scales, group_count, out_features)?;
    qzeros_shape_guard(qzeros, group_count, packed_out_cols)?;

    let bias_name = format!("{prefix}.bias");
    Ok(PackedLinearLayer {
        prefix: prefix.to_string(),
        qweight: qweight_name,
        qzeros: qzeros_name,
        scales: scales_name,
        bias: container.tensor(&bias_name).map(|_| bias_name),
        in_features,
        out_features,
        group_count,
    })
}

/// Guard: `qweight` must be rank 2. Returns `(in_features,
/// packed_out_cols)` straight off its dims.
fn qweight_rank_guard(qweight: &TensorDescriptor) -> Result<(u64, u64), String> {
    let dims = qweight.shape.dims();
    if dims.len() != 2 {
        return Err(format!("qweight has rank {}, expected 2", dims.len()));
    }
    Ok((dims[0] as u64, dims[1] as u64))
}

/// Guard: neither of `qweight`'s dimensions may be zero. A
/// `mlmf-safetensors`-parsed container can legitimately produce a `[N, 0]`
/// shape (zero elements satisfies its own element-count check), so this
/// must be checked explicitly rather than left to cascade into a
/// zero-sized "valid" layer.
fn zero_dimension_guard(in_features: u64, packed_out_cols: u64) -> Result<(), String> {
    if in_features == 0 || packed_out_cols == 0 {
        return Err(format!(
            "qweight shape [{in_features}, {packed_out_cols}] has a zero dimension"
        ));
    }
    Ok(())
}

/// Guard: `packed_out_cols * pack_factor` must not overflow `u64` --
/// `checked_mul`, never a raw `*`, since both operands are read off an
/// attacker-controlled (downloaded-file) shape.
fn out_features_guard(packed_out_cols: u64, pack_factor: u64) -> Result<u64, String> {
    packed_out_cols.checked_mul(pack_factor).ok_or_else(|| {
        format!(
            "qweight's packed column count {packed_out_cols} * pack factor {pack_factor} overflows u64"
        )
    })
}

/// Guard: `in_features` must be a whole number of `group_size`-sized
/// groups.
fn group_count_guard(in_features: u64, group_size: u64) -> Result<u64, String> {
    if !in_features.is_multiple_of(group_size) {
        return Err(format!(
            "in_features {in_features} is not a whole number of groups of {group_size}"
        ));
    }
    Ok(in_features / group_size)
}

/// Guard: `scales` must be exactly `[group_count, out_features]`.
fn scales_shape_guard(
    scales: &TensorDescriptor,
    group_count: u64,
    out_features: u64,
) -> Result<(), String> {
    if scales.shape.dims() != [group_count as usize, out_features as usize] {
        return Err(format!(
            "scales shape {:?} does not match the expected [{group_count}, {out_features}]",
            scales.shape.dims()
        ));
    }
    Ok(())
}

/// Guard: `qzeros` must be exactly `[group_count, packed_out_cols]`.
fn qzeros_shape_guard(
    qzeros: &TensorDescriptor,
    group_count: u64,
    packed_out_cols: u64,
) -> Result<(), String> {
    if qzeros.shape.dims() != [group_count as usize, packed_out_cols as usize] {
        return Err(format!(
            "qzeros shape {:?} does not match the expected [{group_count}, {packed_out_cols}]",
            qzeros.shape.dims()
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::borrow::Cow;

    use mlmf_core::{DType, Encoding, Result, Shape, TensorContainer, TensorDescriptor};

    use super::*;

    struct FakeContainer(Vec<TensorDescriptor>);

    impl TensorContainer for FakeContainer {
        fn tensors(&self) -> &[TensorDescriptor] {
            &self.0
        }
        fn tensor_bytes(&self, _descriptor: &TensorDescriptor) -> Result<Cow<'_, [u8]>> {
            unreachable!("locate_layers never reads tensor bytes, only descriptors")
        }
    }

    fn descriptor(name: &str, dims: &[usize], dtype: DType) -> TensorDescriptor {
        TensorDescriptor {
            name: name.to_string(),
            shape: Shape::new(dims.iter().copied()),
            encoding: Encoding::Dense(dtype),
            bytes: 0..1,
        }
    }

    /// `model.layers.0.self_attn.q_proj`'s real tensor set, shapes
    /// verified against TheBloke/Llama-2-7B-Chat-AWQ, 2026-10-02.
    fn real_q_proj_layer() -> Vec<TensorDescriptor> {
        let p = "model.layers.0.self_attn.q_proj";
        vec![
            descriptor(&format!("{p}.qweight"), &[4096, 512], DType::I32),
            descriptor(&format!("{p}.qzeros"), &[32, 512], DType::I32),
            descriptor(&format!("{p}.scales"), &[32, 4096], DType::F16),
        ]
    }

    #[test]
    fn finds_the_real_q_proj_layer_with_correct_geometry() {
        let container = FakeContainer(real_q_proj_layer());
        let report = locate_layers(&container, 4, 128).expect("valid parameters");

        assert_eq!(report.incomplete, Vec::<String>::new());
        assert_eq!(report.malformed, Vec::<(String, String)>::new());
        assert_eq!(report.layers.len(), 1);

        let layer = &report.layers[0];
        assert_eq!(layer.prefix, "model.layers.0.self_attn.q_proj");
        assert_eq!(layer.in_features, 4096); // read directly off qweight's rows
        assert_eq!(layer.out_features, 4096); // 512 * pack_factor(8)
        assert_eq!(layer.group_count, 32); // 4096 / 128
        assert_eq!(layer.bias, None); // this real layer has none
    }

    #[test]
    fn a_layer_missing_scales_is_incomplete_not_dropped_or_panicking() {
        let p = "model.layers.0.self_attn.q_proj";
        let container = FakeContainer(vec![
            descriptor(&format!("{p}.qweight"), &[4096, 512], DType::I32),
            descriptor(&format!("{p}.qzeros"), &[32, 512], DType::I32),
            // scales deliberately omitted
        ]);
        let report = locate_layers(&container, 4, 128).expect("valid parameters");
        assert!(report.layers.is_empty());
        assert_eq!(report.incomplete, vec![p.to_string()]);
    }

    /// Final-review finding I3: only the missing-`scales` case was tested.
    /// The let-else at the top of the loop handles both tensors in one
    /// branch, but nothing pinned the `qzeros`-missing half of it.
    #[test]
    fn a_layer_missing_qzeros_is_incomplete_not_dropped_or_panicking() {
        let p = "model.layers.0.self_attn.q_proj";
        let container = FakeContainer(vec![
            descriptor(&format!("{p}.qweight"), &[4096, 512], DType::I32),
            // qzeros deliberately omitted
            descriptor(&format!("{p}.scales"), &[32, 4096], DType::F16),
        ]);
        let report = locate_layers(&container, 4, 128).expect("valid parameters");
        assert!(report.layers.is_empty());
        assert_eq!(report.incomplete, vec![p.to_string()]);
    }

    /// Final-review finding I3: the only assertion on `bias` anywhere
    /// (`finds_the_real_q_proj_layer_with_correct_geometry`) is
    /// `assert_eq!(layer.bias, None)` -- a constant expected value a
    /// sabotage to always return `None` would not be caught by. This test
    /// pins the other half: a layer that DOES carry `.bias`.
    #[test]
    fn a_layer_with_bias_records_its_tensor_name() {
        let p = "model.layers.0.self_attn.q_proj";
        let container = FakeContainer(vec![
            descriptor(&format!("{p}.qweight"), &[4096, 512], DType::I32),
            descriptor(&format!("{p}.qzeros"), &[32, 512], DType::I32),
            descriptor(&format!("{p}.scales"), &[32, 4096], DType::F16),
            descriptor(&format!("{p}.bias"), &[4096], DType::F16),
        ]);
        let report = locate_layers(&container, 4, 128).expect("valid parameters");
        assert_eq!(report.layers.len(), 1);
        assert_eq!(report.layers[0].bias, Some(format!("{p}.bias")));
    }

    #[test]
    fn bits_that_does_not_divide_32_is_a_call_error_not_a_panic() {
        let container = FakeContainer(Vec::new());
        let err = locate_layers(&container, 5, 128).unwrap_err();
        assert!(err.to_string().contains('5'));
    }

    #[test]
    fn zero_bits_is_a_call_error() {
        let container = FakeContainer(Vec::new());
        locate_layers(&container, 0, 128).unwrap_err();
    }

    #[test]
    fn zero_group_size_is_a_call_error() {
        let container = FakeContainer(Vec::new());
        locate_layers(&container, 4, 0).unwrap_err();
    }

    #[test]
    fn a_qweight_shape_that_would_overflow_out_features_is_malformed_not_a_panic() {
        let p = "model.layers.0.self_attn.q_proj";
        let container = FakeContainer(vec![
            descriptor(&format!("{p}.qweight"), &[4096, 1 << 62], DType::I32),
            descriptor(&format!("{p}.qzeros"), &[1, 1], DType::I32),
            descriptor(&format!("{p}.scales"), &[1, 1], DType::F16),
        ]);
        let report = locate_layers(&container, 4, 128).expect("valid parameters");
        assert!(report.layers.is_empty());
        assert_eq!(report.malformed.len(), 1);
        assert_eq!(report.malformed[0].0, p);
        assert!(report.malformed[0].1.contains("overflow"));
    }

    #[test]
    fn a_zero_sized_qweight_dimension_is_malformed_not_a_wrong_but_valid_layer() {
        let p = "model.layers.0.self_attn.q_proj";
        let container = FakeContainer(vec![
            descriptor(&format!("{p}.qweight"), &[4096, 0], DType::I32),
            descriptor(&format!("{p}.qzeros"), &[0, 0], DType::I32),
            descriptor(&format!("{p}.scales"), &[0, 0], DType::F16),
        ]);
        let report = locate_layers(&container, 4, 128).expect("valid parameters");
        assert!(report.layers.is_empty());
        assert_eq!(report.malformed.len(), 1);
        assert_eq!(report.malformed[0].0, p);
        assert!(report.malformed[0].1.contains("zero dimension"));
    }

    #[test]
    fn a_rank_1_qweight_is_malformed_not_an_index_panic() {
        let p = "model.layers.0.self_attn.q_proj";
        let container = FakeContainer(vec![
            descriptor(&format!("{p}.qweight"), &[4096], DType::I32),
            descriptor(&format!("{p}.qzeros"), &[32, 512], DType::I32),
            descriptor(&format!("{p}.scales"), &[32, 4096], DType::F16),
        ]);
        let report = locate_layers(&container, 4, 128).expect("valid parameters");
        assert!(report.layers.is_empty());
        assert_eq!(report.malformed.len(), 1);
        assert_eq!(report.malformed[0].0, p);
        assert!(report.malformed[0].1.contains("rank"));
    }

    #[test]
    fn an_in_features_not_a_whole_number_of_groups_is_malformed_not_an_abort() {
        // "bad": in_features 4096, group_size 127 does not divide it
        // (4096 % 127 == 32). "good": in_features 1016 (a DIFFERENT
        // qweight row count, 127), 1016 % 127 == 0 (group_count 8) --
        // a SECOND, well-formed layer in the same container must still
        // be found. Shapes double-checked against the fixture-consistency
        // mistake mlmf-gptq's Task 4 made and corrected (its ledger entry
        // explains why both layers must NOT share one qweight shape under
        // one group_size).
        let good = "model.layers.1.self_attn.q_proj";
        let container = FakeContainer(vec![
            descriptor("bad.qweight", &[4096, 512], DType::I32),
            descriptor("bad.qzeros", &[32, 512], DType::I32),
            descriptor("bad.scales", &[32, 4096], DType::F16),
            descriptor(&format!("{good}.qweight"), &[1016, 512], DType::I32),
            descriptor(&format!("{good}.qzeros"), &[8, 512], DType::I32),
            descriptor(&format!("{good}.scales"), &[8, 4096], DType::F16),
        ]);
        let report = locate_layers(&container, 4, 127).expect("valid parameters");
        assert_eq!(report.layers.len(), 1);
        assert_eq!(report.layers[0].prefix, good);
        assert_eq!(report.malformed.len(), 1);
        assert_eq!(report.malformed[0].0, "bad");
    }

    #[test]
    fn a_scales_shape_mismatch_is_malformed() {
        let p = "model.layers.0.self_attn.q_proj";
        let container = FakeContainer(vec![
            descriptor(&format!("{p}.qweight"), &[4096, 512], DType::I32),
            descriptor(&format!("{p}.qzeros"), &[32, 512], DType::I32),
            // Half the expected group count: [16, 4096] instead of [32, 4096].
            descriptor(&format!("{p}.scales"), &[16, 4096], DType::F16),
        ]);
        let report = locate_layers(&container, 4, 128).expect("valid parameters");
        assert!(report.layers.is_empty());
        assert_eq!(report.malformed.len(), 1);
        assert_eq!(report.malformed[0].0, p);
        assert!(report.malformed[0].1.contains("scales"));
    }

    #[test]
    fn a_qzeros_shape_mismatch_is_malformed() {
        let p = "model.layers.0.self_attn.q_proj";
        let container = FakeContainer(vec![
            descriptor(&format!("{p}.qweight"), &[4096, 512], DType::I32),
            // Half the expected column count: [32, 256] instead of [32, 512].
            descriptor(&format!("{p}.qzeros"), &[32, 256], DType::I32),
            descriptor(&format!("{p}.scales"), &[32, 4096], DType::F16),
        ]);
        let report = locate_layers(&container, 4, 128).expect("valid parameters");
        assert!(report.layers.is_empty());
        assert_eq!(report.malformed.len(), 1);
        assert_eq!(report.malformed[0].0, p);
        assert!(report.malformed[0].1.contains("qzeros"));
    }
}
