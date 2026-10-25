// SPDX-License-Identifier: MIT OR Apache-2.0
//! Find a bitsandbytes 4-bit layer's tensors in a container, and read them.
//!
//! A layer is announced by `<p>.weight.quant_state.bitsandbytes__<type>` (a
//! `U8` tensor holding the JSON). Its other pieces sit beside it as
//! `<p>.weight` (the packed 4-bit codes, as `U8`), `<p>.weight.absmax`,
//! `<p>.weight.quant_map`, and, for double quantization,
//! `<p>.weight.nested_absmax` and `<p>.weight.nested_quant_map`. Each piece
//! that is not in the container is `None`: whether its absence matters is
//! the dequantizer's call (`nested_*` are simply absent without double
//! quantization).
//!
//! Nothing is decoded: the pieces come back as the file's bytes.

use std::borrow::Cow;

use mlmf_core::{TensorContainer, TensorDescriptor};

use crate::BnbError;
use crate::quant_state::{BnbQuantState, parse_quant_state};

const STATE_MARKER: &str = ".quant_state.bitsandbytes__";

/// One 4-bit layer's tensors, by descriptor.
#[derive(Debug, Clone, PartialEq)]
pub struct BnbLayer {
    /// The weight's name, `<p>.weight`; every other piece is named from it.
    pub weight_name: String,
    /// The suffix after `bitsandbytes__` in the state tensor's name:
    /// `nf4`, `fp4`, or whatever a newer writer uses.
    pub state_type: String,
    /// `<p>.weight`: the packed codes, if the container has it.
    pub weight: Option<TensorDescriptor>,
    /// `<p>.weight.absmax`.
    pub absmax: Option<TensorDescriptor>,
    /// `<p>.weight.quant_map`.
    pub quant_map: Option<TensorDescriptor>,
    /// `<p>.weight.nested_absmax` (double quantization only).
    pub nested_absmax: Option<TensorDescriptor>,
    /// `<p>.weight.nested_quant_map` (double quantization only).
    pub nested_quant_map: Option<TensorDescriptor>,
    /// `<p>.weight.quant_state.bitsandbytes__<type>`: the JSON.
    pub quant_state: TensorDescriptor,
}

/// A layer's pieces as the file's bytes, and its parsed `quant_state`.
#[derive(Debug)]
pub struct BnbLayerBytes<'a> {
    /// Packed 4-bit codes, two per byte, as stored.
    pub weight: Option<Cow<'a, [u8]>>,
    /// First-level absmax as stored: `U8` codes under double quantization,
    /// `F32` otherwise; see the descriptor's dtype.
    pub absmax: Option<Cow<'a, [u8]>>,
    /// The 16-entry codebook as stored.
    pub quant_map: Option<Cow<'a, [u8]>>,
    /// Second-level scales as stored.
    pub nested_absmax: Option<Cow<'a, [u8]>>,
    /// The 256-entry second-level codebook as stored.
    pub nested_quant_map: Option<Cow<'a, [u8]>>,
    /// The parsed JSON.
    pub state: BnbQuantState,
}

fn companion(c: &dyn TensorContainer, weight: &str, suffix: &str) -> Option<TensorDescriptor> {
    c.tensor(&format!("{weight}{suffix}")).cloned()
}

fn layer_of(c: &dyn TensorContainer, d: &TensorDescriptor) -> Option<BnbLayer> {
    let at = d.name.find(STATE_MARKER)?;
    let weight_name = d.name[..at].to_string();
    let state_type = d.name[at + STATE_MARKER.len()..].to_string();
    Some(BnbLayer {
        weight: c.tensor(&weight_name).cloned(),
        absmax: companion(c, &weight_name, ".absmax"),
        quant_map: companion(c, &weight_name, ".quant_map"),
        nested_absmax: companion(c, &weight_name, ".nested_absmax"),
        nested_quant_map: companion(c, &weight_name, ".nested_quant_map"),
        weight_name,
        state_type,
        quant_state: d.clone(),
    })
}

/// Every 4-bit layer `container` announces, sorted by weight name.
#[must_use]
pub fn locate_layers(container: &dyn TensorContainer) -> Vec<BnbLayer> {
    let mut layers: Vec<BnbLayer> = container
        .tensors()
        .iter()
        .filter_map(|d| layer_of(container, d))
        .collect();
    layers.sort_by(|a, b| a.weight_name.cmp(&b.weight_name));
    layers
}

fn bytes<'a>(
    c: &'a dyn TensorContainer,
    d: Option<&TensorDescriptor>,
) -> Result<Option<Cow<'a, [u8]>>, BnbError> {
    d.map(|d| c.tensor_bytes(d))
        .transpose()
        .map_err(|e| BnbError::Read(e.to_string()))
}

/// Read a located layer's pieces.
///
/// # Errors
///
/// [`BnbError::Read`] if a tensor's bytes cannot be read;
/// [`BnbError::QuantStateJson`] / [`BnbError::QuantStateField`] if the state
/// tensor is not a readable `quant_state` blob.
pub fn read_layer<'a>(
    container: &'a dyn TensorContainer,
    layer: &BnbLayer,
) -> Result<BnbLayerBytes<'a>, BnbError> {
    let json = container
        .tensor_bytes(&layer.quant_state)
        .map_err(|e| BnbError::Read(e.to_string()))?;
    Ok(BnbLayerBytes {
        weight: bytes(container, layer.weight.as_ref())?,
        absmax: bytes(container, layer.absmax.as_ref())?,
        quant_map: bytes(container, layer.quant_map.as_ref())?,
        nested_absmax: bytes(container, layer.nested_absmax.as_ref())?,
        nested_quant_map: bytes(container, layer.nested_quant_map.as_ref())?,
        state: parse_quant_state(&json)?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlmf_core::{DType, Encoding, Result, Shape};

    struct Fake(Vec<TensorDescriptor>);

    impl TensorContainer for Fake {
        fn tensors(&self) -> &[TensorDescriptor] {
            &self.0
        }
        fn tensor_bytes(&self, _d: &TensorDescriptor) -> Result<Cow<'_, [u8]>> {
            unreachable!("locate_layers reads descriptors only")
        }
    }

    fn d(name: &str) -> TensorDescriptor {
        TensorDescriptor {
            name: name.to_string(),
            shape: Shape::new([1usize]),
            encoding: Encoding::Dense(DType::U8),
            bytes: 0..1,
        }
    }

    #[test]
    fn layers_are_found_by_their_state_tensor_and_sorted() {
        let c = Fake(vec![
            d("b.weight.quant_state.bitsandbytes__nf4"),
            d("b.weight"),
            d("a.weight.quant_state.bitsandbytes__fp4"),
            d("a.weight"),
            d("a.weight.absmax"),
            d("c.weight"),
        ]);
        let layers = locate_layers(&c);
        let names: Vec<(&str, &str)> = layers
            .iter()
            .map(|l| (l.weight_name.as_str(), l.state_type.as_str()))
            .collect();
        assert_eq!(names, vec![("a.weight", "fp4"), ("b.weight", "nf4")]);
        assert!(layers[0].absmax.is_some() && layers[1].absmax.is_none());
    }

    #[test]
    fn a_missing_piece_is_none_not_an_error_or_a_default() {
        let c = Fake(vec![d("x.weight.quant_state.bitsandbytes__nf4")]);
        let l = &locate_layers(&c)[0];
        assert!(l.weight.is_none() && l.absmax.is_none() && l.quant_map.is_none());
        assert!(l.nested_absmax.is_none() && l.nested_quant_map.is_none());
    }

    #[test]
    fn a_container_with_no_state_tensor_has_no_layers() {
        assert!(locate_layers(&Fake(vec![d("w.weight"), d("w.weight.absmax")])).is_empty());
    }
}
