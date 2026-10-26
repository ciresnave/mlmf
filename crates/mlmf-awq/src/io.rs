// SPDX-License-Identifier: MIT OR Apache-2.0
//! Reading and writing an AWQ layer's tensors as the file holds them.
//!
//! CireSnave, 2026-10-24: *"We are keeping AWQ loading and saving in MLMF."*
//! Unpacking the int4 nibbles, and packing floats into them, is conversion
//! and belongs to `ml-convertant` (ROADMAP S-53). This module moves the
//! stored bytes and checks that they are the size the layer's geometry says
//! they must be; it never reinterprets them.
//!
//! Reading takes an already-located [`PackedLinearLayer`]. Writing returns
//! [`AwqPart`]s (name, dtype, shape, bytes) for a caller to hand to a
//! container writer such as `mlmf-safetensors`; this crate stays free of I/O.

use std::borrow::Cow;
use std::fmt;

use mlmf_core::{DType, Encoding, TensorContainer, TensorDescriptor};

use crate::layer::PackedLinearLayer;

/// Why a layer could not be read or a write refused.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AwqIoError {
    message: String,
}

impl AwqIoError {
    fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl fmt::Display for AwqIoError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for AwqIoError {}

/// One tensor of a layer, as stored.
#[derive(Debug)]
pub struct AwqTensorBytes<'a> {
    /// The tensor's name in the container.
    pub name: String,
    /// The element type the bytes are stored in.
    pub dtype: DType,
    /// Dimensions, outermost first.
    pub shape: Vec<usize>,
    /// The bytes exactly as the file holds them.
    pub bytes: Cow<'a, [u8]>,
}

/// A located layer's tensors as the file's bytes.
#[derive(Debug)]
pub struct AwqLayerBytes<'a> {
    /// Packed weights, `[in_features, out_features / pack_factor]`.
    pub qweight: AwqTensorBytes<'a>,
    /// Packed zero points, `[group_count, out_features / pack_factor]`.
    pub qzeros: AwqTensorBytes<'a>,
    /// Per-group scales, `[group_count, out_features]`.
    pub scales: AwqTensorBytes<'a>,
    /// The bias, when the layer has one.
    pub bias: Option<AwqTensorBytes<'a>>,
}

fn dense_dtype(d: &TensorDescriptor) -> Result<DType, AwqIoError> {
    match &d.encoding {
        Encoding::Dense(dtype) => Ok(*dtype),
        Encoding::Blocked(_) => Err(AwqIoError::new(format!(
            "tensor `{}` is block-encoded; an AWQ tensor is dense",
            d.name
        ))),
    }
}

fn read_tensor<'a>(
    container: &'a dyn TensorContainer,
    name: &str,
) -> Result<AwqTensorBytes<'a>, AwqIoError> {
    let d = container
        .tensor(name)
        .ok_or_else(|| AwqIoError::new(format!("tensor `{name}` is not in the container")))?;
    let dtype = dense_dtype(d)?;
    let bytes = container
        .tensor_bytes(d)
        .map_err(|e| AwqIoError::new(format!("reading `{name}`: {e}")))?;
    Ok(AwqTensorBytes {
        name: name.to_string(),
        dtype,
        shape: d.shape.dims().to_vec(),
        bytes,
    })
}

/// Read a located layer's tensors as stored.
///
/// # Errors
///
/// A tensor the layer names is missing, is block-encoded, or its bytes
/// cannot be read.
pub fn read_layer<'a>(
    container: &'a dyn TensorContainer,
    layer: &PackedLinearLayer,
) -> Result<AwqLayerBytes<'a>, AwqIoError> {
    let bias = match &layer.bias {
        Some(name) => Some(read_tensor(container, name)?),
        None => None,
    };
    Ok(AwqLayerBytes {
        qweight: read_tensor(container, &layer.qweight)?,
        qzeros: read_tensor(container, &layer.qzeros)?,
        scales: read_tensor(container, &layer.scales)?,
        bias,
    })
}

/// A layer to write: its stored bytes and the declared quantization.
#[derive(Debug, Clone, Copy)]
pub struct AwqLayerToWrite<'a> {
    /// The name prefix, e.g. `model.layers.0.self_attn.q_proj`.
    pub prefix: &'a str,
    /// Bit width per weight.
    pub bits: u64,
    /// Weights per quantization group.
    pub group_size: u64,
    /// Logical input features.
    pub in_features: u64,
    /// Logical output features.
    pub out_features: u64,
    /// `I32` words holding packed weights, little-endian.
    pub qweight: &'a [u8],
    /// `I32` words holding packed zero points, little-endian.
    pub qzeros: &'a [u8],
    /// Scales in `scales_dtype`.
    pub scales: &'a [u8],
    /// The scales' element type.
    pub scales_dtype: DType,
    /// The bias, if any, with its element type.
    pub bias: Option<(DType, &'a [u8])>,
}

/// One tensor ready for a container writer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AwqPart {
    /// Full tensor name.
    pub name: String,
    /// Element type the bytes are encoded in.
    pub dtype: DType,
    /// Dimensions, outermost first.
    pub shape: Vec<usize>,
    /// The bytes, unchanged from the input.
    pub data: Vec<u8>,
}

fn product(a: u64, b: u64, what: &str) -> Result<u64, AwqIoError> {
    a.checked_mul(b)
        .ok_or_else(|| AwqIoError::new(format!("{what} overflows u64")))
}

fn to_usize(v: u64, what: &str) -> Result<usize, AwqIoError> {
    usize::try_from(v).map_err(|_| AwqIoError::new(format!("{what} does not fit in usize")))
}

fn expect_len(name: &str, got: usize, want: u64) -> Result<(), AwqIoError> {
    if got as u64 == want {
        Ok(())
    } else {
        Err(AwqIoError::new(format!(
            "`{name}` holds {got} bytes, its shape needs {want}"
        )))
    }
}

fn part(
    prefix: &str,
    suffix: &str,
    dtype: DType,
    shape: [u64; 2],
    data: &[u8],
) -> Result<AwqPart, AwqIoError> {
    let name = format!("{prefix}.{suffix}");
    let elems = product(shape[0], shape[1], &name)?;
    expect_len(
        &name,
        data.len(),
        product(elems, dtype.size() as u64, &name)?,
    )?;
    Ok(AwqPart {
        name,
        dtype,
        shape: vec![to_usize(shape[0], "rows")?, to_usize(shape[1], "columns")?],
        data: data.to_vec(),
    })
}

/// Check `layer` against its own geometry and return its tensors for a writer.
///
/// Nothing is repacked or defaulted: the bytes come back as given, and a
/// length that disagrees with the declared geometry is refused.
///
/// # Errors
///
/// `bits` is zero or does not divide 32, `group_size` or a feature count is
/// zero, `in_features` is not a whole number of groups, `out_features` is not
/// a whole number of pack words, or a byte length disagrees with its shape.
pub fn write_layer(layer: &AwqLayerToWrite<'_>) -> Result<Vec<AwqPart>, AwqIoError> {
    let (packed_cols, groups) = geometry(layer)?;
    let p = layer.prefix;
    let mut parts = vec![
        part(
            p,
            "qweight",
            DType::I32,
            [layer.in_features, packed_cols],
            layer.qweight,
        )?,
        part(p, "qzeros", DType::I32, [groups, packed_cols], layer.qzeros)?,
        part(
            p,
            "scales",
            layer.scales_dtype,
            [groups, layer.out_features],
            layer.scales,
        )?,
    ];
    if let Some((dtype, data)) = layer.bias {
        parts.push(bias_part(p, dtype, layer.out_features, data)?);
    }
    Ok(parts)
}

fn bias_part(prefix: &str, dtype: DType, out: u64, data: &[u8]) -> Result<AwqPart, AwqIoError> {
    let name = format!("{prefix}.bias");
    expect_len(&name, data.len(), product(out, dtype.size() as u64, &name)?)?;
    Ok(AwqPart {
        name,
        dtype,
        shape: vec![to_usize(out, "out_features")?],
        data: data.to_vec(),
    })
}

/// `(packed columns, group count)` for a layer, or the first thing wrong.
fn geometry(l: &AwqLayerToWrite<'_>) -> Result<(u64, u64), AwqIoError> {
    if l.bits == 0 || !32u64.is_multiple_of(l.bits) {
        return Err(AwqIoError::new(format!(
            "bits={} must be nonzero and divide 32",
            l.bits
        )));
    }
    if l.group_size == 0 || l.in_features == 0 || l.out_features == 0 {
        return Err(AwqIoError::new(
            "group_size, in_features and out_features must be nonzero",
        ));
    }
    let pack = 32 / l.bits;
    if !l.out_features.is_multiple_of(pack) {
        return Err(AwqIoError::new(format!(
            "out_features {} is not a whole number of {pack}-weight words",
            l.out_features
        )));
    }
    if !l.in_features.is_multiple_of(l.group_size) {
        return Err(AwqIoError::new(format!(
            "in_features {} is not a whole number of groups of {}",
            l.in_features, l.group_size
        )));
    }
    Ok((l.out_features / pack, l.in_features / l.group_size))
}

#[cfg(test)]
mod tests {
    use mlmf_core::{Result, Shape};

    use super::*;
    use crate::layer::locate_layers;

    /// A container over parts laid end to end in one buffer.
    struct Mem {
        descs: Vec<TensorDescriptor>,
        buf: Vec<u8>,
    }

    impl Mem {
        fn from_parts(parts: &[AwqPart]) -> Self {
            let (mut descs, mut buf) = (Vec::new(), Vec::new());
            for p in parts {
                let start = buf.len();
                buf.extend_from_slice(&p.data);
                descs.push(TensorDescriptor {
                    name: p.name.clone(),
                    shape: Shape::new(p.shape.iter().copied()),
                    encoding: Encoding::Dense(p.dtype),
                    bytes: start as u64..buf.len() as u64,
                });
            }
            Self { descs, buf }
        }
    }

    impl TensorContainer for Mem {
        fn tensors(&self) -> &[TensorDescriptor] {
            &self.descs
        }
        fn tensor_bytes(&self, d: &TensorDescriptor) -> Result<Cow<'_, [u8]>> {
            Ok(Cow::Borrowed(
                &self.buf[d.bytes.start as usize..d.bytes.end as usize],
            ))
        }
    }

    // in 256, out 64, 4-bit (8 per word), group 128: packed cols 8, groups 2.
    fn bytes(n: usize, seed: u8) -> Vec<u8> {
        (0..n)
            .map(|i| (i as u8).wrapping_mul(7).wrapping_add(seed))
            .collect()
    }

    fn layer<'a>(q: &'a [u8], z: &'a [u8], s: &'a [u8]) -> AwqLayerToWrite<'a> {
        AwqLayerToWrite {
            prefix: "m.q_proj",
            bits: 4,
            group_size: 128,
            in_features: 256,
            out_features: 64,
            qweight: q,
            qzeros: z,
            scales: s,
            scales_dtype: DType::F16,
            bias: None,
        }
    }

    #[test]
    fn written_parts_locate_and_read_back_byte_for_byte() {
        let (q, z, s, b) = (
            bytes(256 * 8 * 4, 1),
            bytes(2 * 8 * 4, 2),
            bytes(2 * 64 * 2, 3),
            bytes(64 * 2, 4),
        );
        let mut l = layer(&q, &z, &s);
        l.bias = Some((DType::F16, &b));
        let parts = write_layer(&l).unwrap();
        assert_eq!(parts.len(), 4);
        let mem = Mem::from_parts(&parts);
        let report = locate_layers(&mem, 4, 128).unwrap();
        assert_eq!(report.layers.len(), 1, "{report:?}");
        let got = read_layer(&mem, &report.layers[0]).unwrap();
        assert_eq!(&*got.qweight.bytes, &q[..]);
        assert_eq!(&*got.qzeros.bytes, &z[..]);
        assert_eq!(&*got.scales.bytes, &s[..]);
        assert_eq!(&*got.bias.unwrap().bytes, &b[..]);
        assert_eq!(got.qweight.shape, vec![256, 8]);
        assert_eq!(got.scales.dtype, DType::F16);
    }

    #[test]
    fn a_short_tensor_is_refused_by_name() {
        let (q, z, s) = (bytes(256 * 8 * 4 - 1, 1), bytes(64, 2), bytes(256, 3));
        let e = write_layer(&layer(&q, &z, &s)).unwrap_err().to_string();
        assert!(e.contains("m.q_proj.qweight") && e.contains("8192"), "{e}");
    }

    #[test]
    fn each_geometry_guard_refuses() {
        let (q, z, s) = (bytes(8192, 1), bytes(64, 2), bytes(256, 3));
        for (what, edit) in [
            (
                "bits 0",
                (|l: &mut AwqLayerToWrite<'_>| l.bits = 0) as fn(&mut AwqLayerToWrite<'_>),
            ),
            ("bits 5", |l| l.bits = 5),
            ("group 0", |l| l.group_size = 0),
            ("in not groups", |l| l.in_features = 255),
            ("out not words", |l| l.out_features = 63),
        ] {
            let mut l = layer(&q, &z, &s);
            edit(&mut l);
            assert!(write_layer(&l).is_err(), "{what} was accepted");
        }
        assert!(write_layer(&layer(&q, &z, &s)).is_ok());
    }

    #[test]
    fn a_missing_companion_is_a_read_error() {
        let (q, z, s) = (bytes(8192, 1), bytes(64, 2), bytes(256, 3));
        let mut parts = write_layer(&layer(&q, &z, &s)).unwrap();
        let mem_full = Mem::from_parts(&parts);
        let found = locate_layers(&mem_full, 4, 128).unwrap().layers.remove(0);
        parts.retain(|p| !p.name.ends_with("qzeros"));
        let e = read_layer(&Mem::from_parts(&parts), &found)
            .unwrap_err()
            .to_string();
        assert!(e.contains("qzeros"), "{e}");
    }
}
