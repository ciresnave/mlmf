//! Locating GPTQ-packed linear layers in an already-open container.
//!
//! GPTQ's quantization is **inter-tensor**: one logical linear layer is
//! three to four cooperating tensors sharing a name prefix
//! (`qweight`/`qzeros`/`scales`/optional `g_idx`/`bias`), each a plain
//! `Encoding::Dense` array as far as the container is concerned. This
//! module's job is finding and jointly validating that set — see the
//! design spec §2 for why that does not fit `mlmf_core::BlockSpec`
//! (which describes one self-contained tensor's own bytes).

use std::fmt;

use mlmf_core::{TensorContainer, TensorDescriptor};

use crate::GptqGroupSize;

/// Why `locate_layers` refused to run at all.
///
/// Exists only for bad CALL-TIME parameters (`bits`/`group_size`). A
/// problem with what a FILE declares never reaches here — it lands in
/// [`LocateReport::incomplete`] or [`LocateReport::malformed`] instead,
/// by design: one bad layer must not abort every other layer's result,
/// the same lesson this repo already paid for on GGUF (#97).
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

/// One GPTQ-packed linear layer, located and geometrically validated.
///
/// Carries tensor NAMES, not bytes — a caller already holds the
/// `&dyn TensorContainer` this was located from and fetches bytes through
/// it via `tensor_bytes`, the same seam every other format crate uses.
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
    /// The activation-order index tensor's full name, if present.
    pub g_idx: Option<String>,
    /// The bias tensor's full name, if present.
    pub bias: Option<String>,
    /// Logical input features (`qweight`'s packed row count × pack
    /// factor), not the packed row count itself.
    pub in_features: u64,
    /// Logical output features.
    pub out_features: u64,
    /// `in_features / group_size`, or `1` for [`GptqGroupSize::NoGrouping`]
    /// (one group spans the whole input dimension).
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

/// Locate every GPTQ-packed linear layer `container` declares.
///
/// # Errors
///
/// `bits == 0`, `bits` does not evenly divide 32 (so there is no whole
/// pack factor), or `group_size` is [`GptqGroupSize::PerGroup`] of `0`.
/// These are properties of the CALL, not of the container — a bad file
/// never reaches this error; see [`LocateError`]'s own doc.
pub fn locate_layers(
    container: &dyn TensorContainer,
    bits: u64,
    group_size: GptqGroupSize,
) -> Result<LocateReport, LocateError> {
    if bits == 0 {
        return Err(LocateError::new("bits must be nonzero"));
    }
    if group_size == GptqGroupSize::PerGroup(0) {
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
/// nest of `if`s (none of the guards themselves, or their messages,
/// changed).
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
    group_size: GptqGroupSize,
) -> Result<PackedLinearLayer, String> {
    let (packed_rows, out_features) = qweight_rank_guard(qweight)?;
    zero_dimension_guard(packed_rows, out_features)?;
    let in_features = in_features_guard(packed_rows, pack_factor)?;
    let group_count = group_count_guard(in_features, group_size)?;
    scales_shape_guard(scales, group_count, out_features)?;
    let expected_qzeros_cols = qzeros_pack_divisibility_guard(out_features, pack_factor)?;
    qzeros_shape_guard(qzeros, group_count, expected_qzeros_cols)?;

    let g_idx_name = format!("{prefix}.g_idx");
    let bias_name = format!("{prefix}.bias");
    Ok(PackedLinearLayer {
        prefix: prefix.to_string(),
        qweight: qweight_name,
        qzeros: qzeros_name,
        scales: scales_name,
        g_idx: container.tensor(&g_idx_name).map(|_| g_idx_name),
        bias: container.tensor(&bias_name).map(|_| bias_name),
        in_features,
        out_features,
        group_count,
    })
}

/// Guard: `qweight` must be rank 2. Returns `(packed_rows, out_features)`
/// straight off its dims.
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
fn zero_dimension_guard(packed_rows: u64, out_features: u64) -> Result<(), String> {
    if packed_rows == 0 || out_features == 0 {
        return Err(format!(
            "qweight shape [{packed_rows}, {out_features}] has a zero dimension"
        ));
    }
    Ok(())
}

/// Guard: `packed_rows * pack_factor` must not overflow `u64` --
/// `checked_mul`, never a raw `*`, since both operands are read off an
/// attacker-controlled (downloaded-file) shape.
fn in_features_guard(packed_rows: u64, pack_factor: u64) -> Result<u64, String> {
    packed_rows.checked_mul(pack_factor).ok_or_else(|| {
        format!(
            "qweight's packed row count {packed_rows} * pack factor {pack_factor} overflows u64"
        )
    })
}

/// Guard: `GptqGroupSize::NoGrouping` is always one group spanning the
/// whole input dimension; `PerGroup(gs)` requires `in_features` to be a
/// whole number of `gs`-sized groups.
fn group_count_guard(in_features: u64, group_size: GptqGroupSize) -> Result<u64, String> {
    match group_size {
        GptqGroupSize::NoGrouping => Ok(1),
        GptqGroupSize::PerGroup(gs) => {
            if !in_features.is_multiple_of(gs) {
                return Err(format!(
                    "in_features {in_features} is not a whole number of groups of {gs}"
                ));
            }
            Ok(in_features / gs)
        }
    }
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

/// Guard: `out_features` must be a whole number of `pack_factor`-sized
/// packs. Returns the expected `qzeros` column count.
fn qzeros_pack_divisibility_guard(out_features: u64, pack_factor: u64) -> Result<u64, String> {
    if !out_features.is_multiple_of(pack_factor) {
        return Err(format!(
            "out_features {out_features} is not a whole number of {pack_factor}-packs"
        ));
    }
    Ok(out_features / pack_factor)
}

/// Guard: `qzeros` must be exactly `[group_count, expected_qzeros_cols]`.
fn qzeros_shape_guard(
    qzeros: &TensorDescriptor,
    group_count: u64,
    expected_qzeros_cols: u64,
) -> Result<(), String> {
    if qzeros.shape.dims() != [group_count as usize, expected_qzeros_cols as usize] {
        return Err(format!(
            "qzeros shape {:?} does not match the expected [{group_count}, {expected_qzeros_cols}]",
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

    /// A `TensorContainer` built from a plain list of descriptors, for
    /// tests that care about `locate_layers`'s logic and not about any
    /// real container format.
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
    /// verified against TheBloke/dolphin-2.2.1-mistral-7B-GPTQ, 2026-10-01.
    fn real_q_proj_layer() -> Vec<TensorDescriptor> {
        let p = "model.layers.0.self_attn.q_proj";
        vec![
            descriptor(&format!("{p}.g_idx"), &[4096], DType::I32),
            descriptor(&format!("{p}.qweight"), &[512, 4096], DType::I32),
            descriptor(&format!("{p}.qzeros"), &[32, 512], DType::I32),
            descriptor(&format!("{p}.bias"), &[4096], DType::F16),
            descriptor(&format!("{p}.scales"), &[32, 4096], DType::F16),
        ]
    }

    #[test]
    fn finds_the_real_q_proj_layer_with_correct_geometry() {
        let container = FakeContainer(real_q_proj_layer());
        let report =
            locate_layers(&container, 4, GptqGroupSize::PerGroup(128)).expect("valid parameters");

        assert_eq!(report.incomplete, Vec::<String>::new());
        assert_eq!(report.malformed, Vec::<(String, String)>::new());
        assert_eq!(report.layers.len(), 1);

        let layer = &report.layers[0];
        assert_eq!(layer.prefix, "model.layers.0.self_attn.q_proj");
        assert_eq!(layer.in_features, 4096); // 512 * pack_factor(8)
        assert_eq!(layer.out_features, 4096);
        assert_eq!(layer.group_count, 32); // 4096 / 128
        assert_eq!(
            layer.g_idx.as_deref(),
            Some("model.layers.0.self_attn.q_proj.g_idx")
        );
        assert_eq!(
            layer.bias.as_deref(),
            Some("model.layers.0.self_attn.q_proj.bias")
        );
    }

    #[test]
    fn a_layer_missing_scales_is_incomplete_not_dropped_or_panicking() {
        let p = "model.layers.0.self_attn.q_proj";
        let container = FakeContainer(vec![
            descriptor(&format!("{p}.qweight"), &[512, 4096], DType::I32),
            descriptor(&format!("{p}.qzeros"), &[32, 512], DType::I32),
            // scales deliberately omitted
        ]);
        let report =
            locate_layers(&container, 4, GptqGroupSize::PerGroup(128)).expect("valid parameters");
        assert!(report.layers.is_empty());
        assert_eq!(report.incomplete, vec![p.to_string()]);
    }

    #[test]
    fn bits_that_does_not_divide_32_is_a_call_error_not_a_panic() {
        let container = FakeContainer(Vec::new());
        let err = locate_layers(&container, 5, GptqGroupSize::PerGroup(128)).unwrap_err();
        assert!(err.to_string().contains("5"));
    }

    #[test]
    fn zero_bits_is_a_call_error() {
        let container = FakeContainer(Vec::new());
        locate_layers(&container, 0, GptqGroupSize::PerGroup(128)).unwrap_err();
    }

    #[test]
    fn zero_group_size_is_a_call_error() {
        let container = FakeContainer(Vec::new());
        locate_layers(&container, 4, GptqGroupSize::PerGroup(0)).unwrap_err();
    }

    #[test]
    fn an_in_features_not_a_whole_number_of_groups_is_malformed_not_an_abort() {
        // "bad": 512 packed rows * 8 = 4096 in_features; group_size 127
        // does not divide it evenly (4096 % 127 == 32).
        //
        // "good": 127 packed rows * 8 = 1016 in_features; 1016 % 127 == 0
        // (group_count 8), so it IS well-formed under the same
        // group_size=127 this test applies to both layers in one call.
        // A SECOND, well-formed layer in the same container must still be
        // found -- one bad layer must not abort the rest (the #97 lesson,
        // exercised directly).
        let good = "model.layers.1.self_attn.q_proj";
        let container = FakeContainer(vec![
            descriptor("bad.qweight", &[512, 4096], DType::I32),
            descriptor("bad.qzeros", &[32, 512], DType::I32),
            descriptor("bad.scales", &[32, 4096], DType::F16),
            descriptor(&format!("{good}.qweight"), &[127, 4096], DType::I32),
            descriptor(&format!("{good}.qzeros"), &[8, 512], DType::I32),
            descriptor(&format!("{good}.scales"), &[8, 4096], DType::F16),
        ]);
        let report =
            locate_layers(&container, 4, GptqGroupSize::PerGroup(127)).expect("valid parameters");
        assert_eq!(report.layers.len(), 1);
        assert_eq!(report.layers[0].prefix, good);
        assert_eq!(report.malformed.len(), 1);
        assert_eq!(report.malformed[0].0, "bad");
    }

    #[test]
    fn a_qweight_shape_that_would_overflow_in_features_is_malformed_not_a_panic() {
        // Final-review finding I1: packed_rows * pack_factor on an
        // attacker-controlled shape (a downloaded file's declared
        // dimensions) must not panic in debug or silently wrap in
        // release. out_features is nonzero here specifically so the
        // zero-dimension guard does not short-circuit before the
        // multiplication is reached -- this exercises checked_mul's
        // overflow path on its own, distinct from the zero-dimension case
        // covered below.
        let p = "model.layers.0.self_attn.q_proj";
        let container = FakeContainer(vec![
            descriptor(&format!("{p}.qweight"), &[1 << 62, 4096], DType::I32),
            // Shapes irrelevant: the overflow guard must fire before
            // these are ever consulted.
            descriptor(&format!("{p}.qzeros"), &[1, 1], DType::I32),
            descriptor(&format!("{p}.scales"), &[1, 1], DType::F16),
        ]);
        let report =
            locate_layers(&container, 4, GptqGroupSize::PerGroup(128)).expect("valid parameters");
        assert!(
            report.layers.is_empty(),
            "an overflowing qweight must never be reported as a valid layer: {:?}",
            report.layers
        );
        assert_eq!(report.malformed.len(), 1);
        assert_eq!(report.malformed[0].0, p);
        assert!(report.malformed[0].1.contains("overflow"));
    }

    #[test]
    fn a_zero_sized_qweight_dimension_is_malformed_not_a_wrong_but_valid_layer() {
        // mlmf-safetensors admits a [N, 0] or [0, N] shape because its own
        // element-count*dtype-size check (count 0 => span 0) is satisfied
        // by zero elements, so this shape genuinely reaches locate_layers
        // from a real container. Without the zero-dimension guard this
        // cascades to in_features=0, group_count=0, and scales/qzeros
        // shapes of [0, 0] "matching" -- reporting a bogus but internally
        // "consistent" zero-size layer as valid.
        let p = "model.layers.0.self_attn.q_proj";
        let container = FakeContainer(vec![
            descriptor(&format!("{p}.qweight"), &[512, 0], DType::I32),
            descriptor(&format!("{p}.qzeros"), &[0, 0], DType::I32),
            descriptor(&format!("{p}.scales"), &[0, 0], DType::F16),
        ]);
        let report =
            locate_layers(&container, 4, GptqGroupSize::PerGroup(128)).expect("valid parameters");
        assert!(
            report.layers.is_empty(),
            "a zero-sized qweight dimension must never be reported as a valid layer: {:?}",
            report.layers
        );
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
        let report =
            locate_layers(&container, 4, GptqGroupSize::PerGroup(128)).expect("valid parameters");
        assert!(report.layers.is_empty());
        assert_eq!(report.malformed.len(), 1);
        assert_eq!(report.malformed[0].0, p);
        assert!(report.malformed[0].1.contains("rank"));
    }

    #[test]
    fn a_scales_shape_mismatch_is_malformed() {
        let p = "model.layers.0.self_attn.q_proj";
        let container = FakeContainer(vec![
            descriptor(&format!("{p}.qweight"), &[512, 4096], DType::I32),
            descriptor(&format!("{p}.qzeros"), &[32, 512], DType::I32),
            // Half the expected group count: [16, 4096] instead of [32, 4096].
            descriptor(&format!("{p}.scales"), &[16, 4096], DType::F16),
        ]);
        let report =
            locate_layers(&container, 4, GptqGroupSize::PerGroup(128)).expect("valid parameters");
        assert!(report.layers.is_empty());
        assert_eq!(report.malformed.len(), 1);
        assert_eq!(report.malformed[0].0, p);
        assert!(report.malformed[0].1.contains("scales"));
    }

    #[test]
    fn an_out_features_not_a_whole_number_of_packs_is_malformed() {
        let p = "model.layers.0.self_attn.q_proj";
        // out_features = 4095 is not a multiple of pack_factor (8).
        // scales must match [group_count, out_features] = [32, 4095] to
        // reach the out_features%pack_factor check rather than failing
        // the scales-shape check first.
        let container = FakeContainer(vec![
            descriptor(&format!("{p}.qweight"), &[512, 4095], DType::I32),
            descriptor(&format!("{p}.qzeros"), &[32, 512], DType::I32),
            descriptor(&format!("{p}.scales"), &[32, 4095], DType::F16),
        ]);
        let report =
            locate_layers(&container, 4, GptqGroupSize::PerGroup(128)).expect("valid parameters");
        assert!(report.layers.is_empty());
        assert_eq!(report.malformed.len(), 1);
        assert_eq!(report.malformed[0].0, p);
        assert!(report.malformed[0].1.contains("out_features"));
    }

    #[test]
    fn a_qzeros_shape_mismatch_is_malformed() {
        let p = "model.layers.0.self_attn.q_proj";
        let container = FakeContainer(vec![
            descriptor(&format!("{p}.qweight"), &[512, 4096], DType::I32),
            // Half the expected column count: [32, 256] instead of [32, 512].
            descriptor(&format!("{p}.qzeros"), &[32, 256], DType::I32),
            descriptor(&format!("{p}.scales"), &[32, 4096], DType::F16),
        ]);
        let report =
            locate_layers(&container, 4, GptqGroupSize::PerGroup(128)).expect("valid parameters");
        assert!(report.layers.is_empty());
        assert_eq!(report.malformed.len(), 1);
        assert_eq!(report.malformed[0].0, p);
        assert!(report.malformed[0].1.contains("qzeros"));
    }

    #[test]
    fn no_grouping_reports_one_group_spanning_the_whole_input_dimension() {
        // Final-review finding I3: locate_layers must be able to express
        // AutoGPTQ's "-1 means no grouping" convention, not just read it
        // out of GptqConfig. With no grouping, group_count is always 1
        // regardless of in_features, and scales/qzeros carry one row.
        let p = "model.layers.0.self_attn.q_proj";
        let container = FakeContainer(vec![
            descriptor(&format!("{p}.qweight"), &[512, 4096], DType::I32),
            descriptor(&format!("{p}.qzeros"), &[1, 512], DType::I32),
            descriptor(&format!("{p}.scales"), &[1, 4096], DType::F16),
        ]);
        let report =
            locate_layers(&container, 4, GptqGroupSize::NoGrouping).expect("valid parameters");
        assert_eq!(report.malformed, Vec::<(String, String)>::new());
        assert_eq!(report.layers.len(), 1);
        assert_eq!(report.layers[0].group_count, 1);
        assert_eq!(report.layers[0].in_features, 4096);
    }
}
