// SPDX-License-Identifier: MIT OR Apache-2.0
//! Writing a `.pt2` archive from caller-supplied parts.
//!
//! The exported graph (`models/<name>.json`) and the example inputs
//! (`data/sample_inputs/<name>.pt`, a `torch.save` blob) are opaque payloads
//! the caller produced; this crate lays them in the archive and writes the
//! weights, the configs and the markers around them. It does not build or edit
//! a graph.
//!
//! Nothing is defaulted: the archive root, `archive_version` and the stream
//! format version are arguments. `byteorder` is written as `little`, which is
//! what the raw tensor bytes must be.

use mlmf_core::DType;
use serde_json::{Value, json};

use crate::Pt2Error;
use crate::scalar::scalar_type_of;
use crate::zipw::write_stored_zip;

/// A tensor to store under `data/weights/` or `data/constants/`.
#[derive(Debug, Clone, Copy)]
pub struct Pt2TensorToWrite<'a> {
    /// The tensor's name, for example `lin.weight`.
    pub name: &'a str,
    /// Element type of `data`.
    pub dtype: DType,
    /// `sizes`.
    pub sizes: &'a [u64],
    /// `strides`, in elements.
    pub strides: &'a [u64],
    /// `storage_offset`.
    pub storage_offset: u64,
    /// `is_param`.
    pub is_param: bool,
    /// `requires_grad`.
    pub requires_grad: bool,
    /// `device.type`, for example `cpu`.
    pub device_type: &'a str,
    /// The storage bytes, little-endian, exactly `product(sizes) * dtype.size()`.
    pub data: &'a [u8],
}

/// One exported model.
#[derive(Debug, Clone, Copy)]
pub struct Pt2ModelToWrite<'a> {
    /// The model's name (`models/<name>.json`).
    pub name: &'a str,
    /// The exported-program JSON, opaque.
    pub graph_json: &'a [u8],
    /// Parameters and buffers.
    pub weights: &'a [Pt2TensorToWrite<'a>],
    /// Lifted tensor constants.
    pub constants: &'a [Pt2TensorToWrite<'a>],
    /// The `torch.save` blob of example inputs, opaque; `None` omits it
    /// (`torch.export.load` refuses an archive without it).
    pub sample_inputs: Option<&'a [u8]>,
}

/// Everything in an archive.
#[derive(Debug, Clone, Copy)]
pub struct Pt2ArchiveToWrite<'a> {
    /// The top-level directory name inside the zip.
    pub root: &'a str,
    /// The `archive_version` string.
    pub archive_version: &'a str,
    /// The PyTorch stream format version (`.data/version`, written as
    /// `<n>\n`).
    pub stream_version: u32,
    /// The models.
    pub models: &'a [Pt2ModelToWrite<'a>],
}

fn code_of(t: &Pt2TensorToWrite<'_>) -> Result<i64, Pt2Error> {
    scalar_type_of(t.dtype).ok_or_else(|| Pt2Error::UnspelledDtype {
        tensor: t.name.to_string(),
        dtype: t.dtype,
    })
}

fn check_len(t: &Pt2TensorToWrite<'_>) -> Result<(), Pt2Error> {
    let overflow = || Pt2Error::TooLarge {
        what: t.name.to_string(),
    };
    let numel = t
        .sizes
        .iter()
        .try_fold(1u64, |a, &d| a.checked_mul(d))
        .ok_or_else(overflow)?;
    let expected = numel
        .checked_mul(t.dtype.size() as u64)
        .ok_or_else(overflow)?;
    let got = t.data.len() as u64;
    if expected == got {
        return Ok(());
    }
    Err(Pt2Error::LengthMismatch {
        tensor: t.name.to_string(),
        expected,
        got,
    })
}

fn as_ints(v: &[u64]) -> Value {
    Value::Array(v.iter().map(|&n| json!({ "as_int": n })).collect())
}

/// One entry of a config's `config` object.
fn config_entry(t: &Pt2TensorToWrite<'_>, path_name: &str) -> Result<Value, Pt2Error> {
    check_len(t)?;
    Ok(json!({
        "path_name": path_name,
        "is_param": t.is_param,
        "use_pickle": false,
        "tensor_meta": {
            "dtype": code_of(t)?,
            "sizes": as_ints(t.sizes),
            "requires_grad": t.requires_grad,
            "device": { "type": t.device_type, "index": null },
            "strides": as_ints(t.strides),
            "storage_offset": { "as_int": t.storage_offset },
            "layout": 7,
        },
    }))
}

/// The blobs and the config for one directory (`weight_` or `tensor_`).
fn tensor_group(
    root: &str,
    dir: &str,
    prefix: &str,
    model: &str,
    kind: &str,
    tensors: &[Pt2TensorToWrite<'_>],
) -> Result<Vec<(String, Vec<u8>)>, Pt2Error> {
    let mut entries = Vec::new();
    let mut config = serde_json::Map::new();
    for (i, t) in tensors.iter().enumerate() {
        let path_name = format!("{prefix}{i}");
        if config
            .insert(t.name.to_string(), config_entry(t, &path_name)?)
            .is_some()
        {
            return Err(Pt2Error::DuplicateName {
                name: t.name.to_string(),
            });
        }
        entries.push((format!("{root}/{dir}{path_name}"), t.data.to_vec()));
    }
    let body = serde_json::to_vec(&json!({ "config": config }))
        .expect("a map of strings and integers serialises");
    entries.push((format!("{root}/{dir}{model}_{kind}_config.json"), body));
    Ok(entries)
}

fn model_entries(root: &str, m: &Pt2ModelToWrite<'_>) -> Result<Vec<(String, Vec<u8>)>, Pt2Error> {
    let mut e = tensor_group(
        root,
        "data/weights/",
        "weight_",
        m.name,
        "weights",
        m.weights,
    )?;
    e.extend(tensor_group(
        root,
        "data/constants/",
        "tensor_",
        m.name,
        "constants",
        m.constants,
    )?);
    e.push((
        format!("{root}/models/{}.json", m.name),
        m.graph_json.to_vec(),
    ));
    if let Some(inputs) = m.sample_inputs {
        e.push((
            format!("{root}/data/sample_inputs/{}.pt", m.name),
            inputs.to_vec(),
        ));
    }
    Ok(e)
}

/// Encode an archive.
///
/// Entry order: each model's weights, weights config, constants, constants
/// config, graph and sample inputs, then `archive_format`, `archive_version`,
/// `.data/version` and `byteorder`.
///
/// # Errors
///
/// [`Pt2Error`]: a dtype with no torch code, a byte length that is not
/// `product(sizes) * dtype.size()`, a duplicate tensor name in one group, or a
/// size plain zip cannot hold.
pub fn encode_pt2(a: &Pt2ArchiveToWrite<'_>) -> Result<Vec<u8>, Pt2Error> {
    let mut entries = Vec::new();
    for m in a.models {
        entries.extend(model_entries(a.root, m)?);
    }
    let root = a.root;
    entries.push((
        format!("{root}/archive_format"),
        crate::FORMAT_VALUE.as_bytes().to_vec(),
    ));
    entries.push((
        format!("{root}/archive_version"),
        a.archive_version.as_bytes().to_vec(),
    ));
    entries.push((
        format!("{root}/.data/version"),
        format!("{}\n", a.stream_version).into_bytes(),
    ));
    entries.push((format!("{root}/byteorder"), b"little".to_vec()));
    write_stored_zip(&entries)
}
