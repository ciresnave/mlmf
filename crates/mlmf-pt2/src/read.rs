// SPDX-License-Identifier: MIT OR Apache-2.0
//! Reading a `.pt2` archive: the layout, the configs, where the bytes are.
//!
//! Nothing is executed and the exported graph is not interpreted: each model's
//! `models/<name>.json` is reported as a byte range of the file.

use std::ops::Range;

use mlmf_core::DType;
use mlmf_pickle::{ZipEntry, read_zip_directory};
use serde_json::Value;

use crate::scalar::dtype_of_scalar_type;
use crate::{FORMAT_VALUE, Pt2Error};

/// One tensor a weights or constants config lists.
#[derive(Debug, Clone, PartialEq)]
pub struct Pt2Tensor {
    /// The model the config belongs to (`<model>_weights_config.json`).
    pub model: String,
    /// The tensor's name, for example `lin.weight`.
    pub name: String,
    /// The zip entry holding its bytes.
    pub path: String,
    /// `is_param`.
    pub is_param: bool,
    /// `use_pickle`: when true the entry is a pickled blob, not raw bytes,
    /// and `data` is the pickle's range.
    pub use_pickle: bool,
    /// torch's `ScalarType` code, exactly as stored.
    pub dtype_code: i64,
    /// The [`DType`] for that code, if this build has one.
    pub dtype: Option<DType>,
    /// `sizes`, each an `as_int`.
    pub sizes: Vec<u64>,
    /// `strides`, each an `as_int`.
    pub strides: Vec<u64>,
    /// `storage_offset`.
    pub storage_offset: u64,
    /// `requires_grad`, if the config states it.
    pub requires_grad: Option<bool>,
    /// `layout` code, if stated (7 is strided).
    pub layout: Option<i64>,
    /// `device.type`, if stated.
    pub device_type: Option<String>,
    /// Absolute byte range of the entry's data in the file.
    pub data: Range<u64>,
}

/// An exported model: its graph is an opaque JSON payload.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Pt2Model {
    /// The model's name (`models/<name>.json`).
    pub name: String,
    /// Absolute byte range of the graph JSON in the file.
    pub graph: Range<u64>,
}

/// A parsed archive.
#[derive(Debug, Clone, PartialEq)]
pub struct Pt2Archive {
    /// The archive's top-level directory name.
    pub root: String,
    /// The `archive_version` string.
    pub archive_version: String,
    /// The `byteorder` entry, if present.
    pub byteorder: Option<String>,
    /// Models, in archive order.
    pub models: Vec<Pt2Model>,
    /// Weights (parameters and buffers) of every model, by model and then
    /// by name (a config's JSON key order is not kept).
    pub weights: Vec<Pt2Tensor>,
    /// Lifted tensor constants of every model, ordered like `weights`.
    pub constants: Vec<Pt2Tensor>,
    /// Every entry of the zip, so nothing in the file is hidden.
    pub entries: Vec<ZipEntry>,
}

fn bad(detail: impl Into<String>) -> Pt2Error {
    Pt2Error::BadArchive {
        detail: detail.into(),
    }
}

fn bytes_of<'a>(file: &'a [u8], e: &ZipEntry) -> Result<&'a [u8], Pt2Error> {
    if e.method != 0 {
        return Err(Pt2Error::Compressed {
            entry: e.name.clone(),
        });
    }
    let (s, n) = (e.data_offset as usize, e.size as usize);
    file.get(s..s + n)
        .ok_or_else(|| bad(format!("entry {:?} runs past the end of the file", e.name)))
}

fn text_of(file: &[u8], e: &ZipEntry) -> Result<String, Pt2Error> {
    String::from_utf8(bytes_of(file, e)?.to_vec())
        .map_err(|_| bad(format!("entry {:?} is not UTF-8", e.name)))
}

/// The archive's single top-level directory.
fn root_of(entries: &[ZipEntry]) -> Result<String, Pt2Error> {
    let first = entries.first().ok_or_else(|| bad("the archive is empty"))?;
    let root = first
        .name
        .split('/')
        .next()
        .filter(|r| !r.is_empty() && first.name.contains('/'))
        .ok_or_else(|| bad("entries are not under a top-level directory"))?;
    let prefix = format!("{root}/");
    match entries.iter().find(|e| !e.name.starts_with(&prefix)) {
        None => Ok(root.to_string()),
        Some(e) => Err(bad(format!("entry {:?} is outside {root:?}", e.name))),
    }
}

fn find<'a>(entries: &'a [ZipEntry], root: &str, rel: &str) -> Option<&'a ZipEntry> {
    let want = format!("{root}/{rel}");
    entries.iter().find(|e| e.name == want)
}

/// Entries directly under `root/dir` whose file name ends with `suffix`,
/// with the part of the name before the suffix.
fn named<'a>(
    entries: &'a [ZipEntry],
    root: &str,
    dir: &str,
    suffix: &str,
) -> Vec<(String, &'a ZipEntry)> {
    let prefix = format!("{root}/{dir}");
    entries
        .iter()
        .filter_map(|e| {
            let rest = e.name.strip_prefix(&prefix)?;
            let stem = rest.strip_suffix(suffix)?;
            (!stem.contains('/')).then(|| (stem.to_string(), e))
        })
        .collect()
}

fn as_u64(v: &Value, what: &str) -> Result<u64, Pt2Error> {
    v.get("as_int")
        .and_then(Value::as_u64)
        .ok_or_else(|| bad(format!("{what} is not an as_int count")))
}

fn counts(meta: &Value, key: &str, who: &str) -> Result<Vec<u64>, Pt2Error> {
    let list = meta
        .get(key)
        .and_then(Value::as_array)
        .ok_or_else(|| bad(format!("{who}: {key} is missing")))?;
    list.iter()
        .map(|v| as_u64(v, &format!("{who}: {key} entry")))
        .collect()
}

/// `path_name`, the dtype code and the `tensor_meta` object of a config entry.
fn required<'a>(cfg: &'a Value, who: &str) -> Result<(&'a str, i64, &'a Value), Pt2Error> {
    let meta = cfg
        .get("tensor_meta")
        .ok_or_else(|| bad(format!("{who}: no tensor_meta")))?;
    let code = meta
        .get("dtype")
        .and_then(Value::as_i64)
        .ok_or_else(|| bad(format!("{who}: no dtype")))?;
    let path_name = cfg
        .get("path_name")
        .and_then(Value::as_str)
        .ok_or_else(|| bad(format!("{who}: no path_name")))?;
    Ok((path_name, code, meta))
}

/// A boolean config key; an absent key is `false` (the schema's own default
/// for `is_param` and `use_pickle`).
fn flag(cfg: &Value, key: &str) -> bool {
    cfg.get(key).and_then(Value::as_bool).unwrap_or(false)
}

/// `device.type`, if the metadata states it.
fn device_type(meta: &Value) -> Option<String> {
    meta.get("device")
        .and_then(|d| d.get("type"))
        .and_then(Value::as_str)
        .map(str::to_string)
}

/// `storage_offset`; an absent key is 0, the schema's default.
fn storage_offset(meta: &Value, who: &str) -> Result<u64, Pt2Error> {
    meta.get("storage_offset")
        .map_or(Ok(0), |v| as_u64(v, &format!("{who}: storage_offset")))
}

/// One config entry as a [`Pt2Tensor`] (without its data range).
fn tensor_of(
    model: &str,
    name: &str,
    cfg: &Value,
    root: &str,
    dir: &str,
) -> Result<Pt2Tensor, Pt2Error> {
    let who = format!("{model}/{name}");
    let (path_name, code, meta) = required(cfg, &who)?;
    Ok(Pt2Tensor {
        model: model.to_string(),
        name: name.to_string(),
        path: format!("{root}/{dir}{path_name}"),
        is_param: flag(cfg, "is_param"),
        use_pickle: flag(cfg, "use_pickle"),
        dtype_code: code,
        dtype: dtype_of_scalar_type(code),
        sizes: counts(meta, "sizes", &who)?,
        strides: counts(meta, "strides", &who)?,
        storage_offset: storage_offset(meta, &who)?,
        requires_grad: meta.get("requires_grad").and_then(Value::as_bool),
        layout: meta.get("layout").and_then(Value::as_i64),
        device_type: device_type(meta),
        data: 0..0,
    })
}

/// The `config` object of a weights or constants config entry.
fn config_of(
    file: &[u8],
    cfg_entry: &ZipEntry,
) -> Result<serde_json::Map<String, Value>, Pt2Error> {
    let json: Value = serde_json::from_str(&text_of(file, cfg_entry)?)
        .map_err(|e| bad(format!("{}: {e}", cfg_entry.name)))?;
    json.get("config")
        .and_then(Value::as_object)
        .cloned()
        .ok_or_else(|| bad(format!("{}: no config object", cfg_entry.name)))
}

/// Set a tensor's `data` range from the zip entry its `path` names.
fn locate(entries: &[ZipEntry], mut t: Pt2Tensor) -> Result<Pt2Tensor, Pt2Error> {
    let e = entries
        .iter()
        .find(|e| e.name == t.path)
        .ok_or_else(|| Pt2Error::MissingEntry {
            path: t.path.clone(),
        })?;
    t.data = e.data_offset..e.data_offset + e.size;
    Ok(t)
}

/// Every tensor listed by the `*_<kind>_config.json` files under `dir`.
fn tensors(
    file: &[u8],
    entries: &[ZipEntry],
    root: &str,
    dir: &str,
    config_suffix: &str,
) -> Result<Vec<Pt2Tensor>, Pt2Error> {
    let mut out = Vec::new();
    for (model, cfg_entry) in named(entries, root, dir, config_suffix) {
        for (name, cfg) in &config_of(file, cfg_entry)? {
            out.push(locate(entries, tensor_of(&model, name, cfg, root, dir)?)?);
        }
    }
    Ok(out)
}

/// Check `archive_format` and return the `archive_version` text.
fn check_markers(file: &[u8], entries: &[ZipEntry], root: &str) -> Result<String, Pt2Error> {
    let format = find(entries, root, "archive_format").ok_or(Pt2Error::NotPt2)?;
    if text_of(file, format)?.trim() != FORMAT_VALUE {
        return Err(Pt2Error::NotPt2);
    }
    let version = find(entries, root, "archive_version").ok_or_else(|| Pt2Error::MissingEntry {
        path: format!("{root}/archive_version"),
    })?;
    text_of(file, version)
}

/// The models: each `models/<name>.json`, as a byte range.
fn models_of(entries: &[ZipEntry], root: &str) -> Vec<Pt2Model> {
    named(entries, root, "models/", ".json")
        .into_iter()
        .map(|(name, e)| Pt2Model {
            name,
            graph: e.data_offset..e.data_offset + e.size,
        })
        .collect()
}

/// Read a `.pt2` archive held in memory.
///
/// # Errors
///
/// [`Pt2Error`]: not a zip, not a `pt2` archive, a missing required entry, a
/// compressed entry this read needs, or a config that is not shaped as the
/// writer's schema defines. A tensor whose dtype code has no [`DType`] is
/// returned with `dtype: None`, not refused.
pub fn parse_pt2(file: &[u8]) -> Result<Pt2Archive, Pt2Error> {
    let entries = read_zip_directory(file).map_err(|e| Pt2Error::NotZip(e.to_string()))?;
    let root = root_of(&entries)?;
    let archive_version = check_markers(file, &entries, &root)?;
    Ok(Pt2Archive {
        archive_version,
        byteorder: find(&entries, &root, "byteorder")
            .map(|e| text_of(file, e))
            .transpose()?,
        models: models_of(&entries, &root),
        weights: tensors(
            file,
            &entries,
            &root,
            "data/weights/",
            "_weights_config.json",
        )?,
        constants: tensors(
            file,
            &entries,
            &root,
            "data/constants/",
            "_constants_config.json",
        )?,
        root,
        entries,
    })
}
