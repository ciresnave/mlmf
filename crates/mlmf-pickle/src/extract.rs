// SPDX-License-Identifier: MIT OR Apache-2.0
//! From the parsed pickle to a tensor index.
//!
//! A checkpoint's pickle is a dict of name to tensor (optionally nested under
//! one key such as `state_dict`). Each tensor is the recorded call
//! `torch._utils._rebuild_tensor_v2(storage, offset, size, stride, ...)`,
//! possibly wrapped by `_rebuild_parameter` or `_rebuild_from_type_v2`, and
//! `storage` is a persistent id naming a storage class, a key and an element
//! count. Nothing here evaluates anything; it reads those arguments.

use mlmf_core::DType;

use crate::error::PickleError;
use crate::vm::{Id, Node, Parsed};

/// One tensor of a checkpoint.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PthTensorInfo {
    /// The tensor's name in the state dict.
    pub name: String,
    /// Element type, from the storage class.
    pub dtype: DType,
    /// The storage class as declared, for example `FloatStorage`.
    pub storage_class: String,
    /// Dimensions.
    pub shape: Vec<usize>,
    /// Strides, in elements.
    pub strides: Vec<usize>,
    /// Offset into the storage, in elements.
    pub storage_offset: u64,
    /// The zip entry holding the storage.
    pub storage_path: String,
    /// Elements in the whole storage (one storage can back several tensors).
    pub storage_numel: u64,
}

/// A tensor record before its storage is resolved against the archive.
pub type Rebuilt = PthTensorInfo;

fn bad(name: &str, detail: &str) -> PickleError {
    PickleError::BadTensor {
        name: name.to_string(),
        detail: detail.to_string(),
    }
}

fn is_global(p: &Parsed, id: Id, module: &str, name: &str) -> bool {
    matches!(&p.arena[id], Node::Global(m, n) if m == module && n == name)
}

fn int(p: &Parsed, id: Id) -> Option<i64> {
    match p.arena[id] {
        Node::Int(i) => Some(i),
        _ => None,
    }
}

fn tuple(p: &Parsed, id: Id) -> Option<&[Id]> {
    match &p.arena[id] {
        Node::Tuple(t) => Some(t),
        _ => None,
    }
}

fn string(p: &Parsed, id: Id) -> Option<&str> {
    match &p.arena[id] {
        Node::Str(s) => Some(s),
        _ => None,
    }
}

/// A tuple of non-negative integers.
fn dims(p: &Parsed, id: Id) -> Option<Vec<usize>> {
    tuple(p, id)?
        .iter()
        .map(|&i| int(p, i).and_then(|v| usize::try_from(v).ok()))
        .collect()
}

/// The dtype of a torch storage class. `None` for a class not in the table.
#[must_use]
pub fn dtype_of_storage(class: &str) -> Option<DType> {
    Some(match class {
        "FloatStorage" => DType::F32,
        "DoubleStorage" => DType::F64,
        "HalfStorage" => DType::F16,
        "BFloat16Storage" => DType::BF16,
        "LongStorage" => DType::I64,
        "IntStorage" => DType::I32,
        "ShortStorage" => DType::I16,
        "CharStorage" => DType::I8,
        "ByteStorage" => DType::U8,
        "BoolStorage" => DType::Bool,
        "ComplexFloatStorage" => DType::C64,
        "ComplexDoubleStorage" => DType::C128,
        _ => return None,
    })
}

fn need<T>(v: Option<T>, name: &str, detail: &str) -> Result<T, PickleError> {
    v.ok_or_else(|| bad(name, detail))
}

fn count(p: &Parsed, id: Id) -> Option<u64> {
    int(p, id).and_then(|n| u64::try_from(n).ok())
}

/// The tuple behind a persistent id: `('storage', class, key, location, numel)`.
fn storage_ref<'a>(p: &'a Parsed, name: &str, id: Id) -> Result<&'a [Id], PickleError> {
    let Node::PersId(inner) = p.arena[id] else {
        return Err(bad(name, "storage is not a persistent id"));
    };
    let t = need(tuple(p, inner), name, "persistent id is not a tuple")?;
    let is_storage = t.len() >= 5 && string(p, t[0]) == Some("storage");
    need(
        is_storage.then_some(t),
        name,
        "persistent id is not a storage reference",
    )
}

/// `(class, key, numel)` of the storage a persistent id names.
fn storage_of(p: &Parsed, name: &str, id: Id) -> Result<(String, String, u64), PickleError> {
    let t = storage_ref(p, name, id)?;
    let class = match &p.arena[t[1]] {
        Node::Global(module, class) if module == "torch" => class.clone(),
        _ => return Err(bad(name, "storage class is not a torch global")),
    };
    let key = need(string(p, t[2]), name, "storage key is not a string")?;
    let numel = need(count(p, t[4]), name, "storage size is not a count")?;
    Ok((class, key.to_string(), numel))
}

/// Shape, strides and storage offset from `_rebuild_tensor_v2`'s arguments.
fn geometry(
    p: &Parsed,
    name: &str,
    [offset, size, stride]: [Id; 3],
) -> Result<(Vec<usize>, Vec<usize>, u64), PickleError> {
    Ok((
        need(dims(p, size), name, "shape is not a tuple of counts")?,
        need(dims(p, stride), name, "strides are not a tuple of counts")?,
        need(count(p, offset), name, "storage offset is not a count")?,
    ))
}

/// The arguments of `_rebuild_tensor_v2`, as the record they describe.
fn rebuild_args(p: &Parsed, name: &str, args: Id, prefix: &str) -> Result<Rebuilt, PickleError> {
    let a = need(tuple(p, args), name, "arguments are not a tuple")?;
    let [storage, offset, size, stride, ..] = a else {
        return Err(bad(name, "too few arguments to _rebuild_tensor_v2"));
    };
    let (class, key, numel) = storage_of(p, name, *storage)?;
    let dtype = dtype_of_storage(&class).ok_or_else(|| PickleError::UnknownStorageType {
        name: name.to_string(),
        storage: class.clone(),
    })?;
    let (shape, strides, storage_offset) = geometry(p, name, [*offset, *size, *stride])?;
    Ok(Rebuilt {
        name: name.to_string(),
        dtype,
        storage_class: class,
        shape,
        strides,
        storage_offset,
        storage_path: format!("{prefix}data/{key}"),
        storage_numel: numel,
    })
}

/// How many `_rebuild_parameter` wrappers are looked through. torch writes
/// one; the bound keeps a crafted chain from recursing to the stack's end.
const MAX_WRAPPERS: usize = 4;

/// One step of unwrapping a tensor value.
enum Unwrapped {
    /// The `_rebuild_tensor_v2` arguments: done.
    Args(Id),
    /// A wrapped value to look through.
    Inner(Id),
}

fn unwrap_once(p: &Parsed, value: Id) -> Option<Unwrapped> {
    let Node::Reduce(callable, args) = p.arena[value] else {
        return None;
    };
    if is_global(p, callable, "torch._utils", "_rebuild_tensor_v2") {
        return Some(Unwrapped::Args(args));
    }
    let a = tuple(p, args)?;
    if is_global(p, callable, "torch._utils", "_rebuild_parameter") {
        return a.first().map(|&i| Unwrapped::Inner(i));
    }
    let typed = is_global(p, callable, "torch._tensor", "_rebuild_from_type_v2")
        && is_global(p, *a.first()?, "torch._utils", "_rebuild_tensor_v2");
    typed
        .then(|| a.get(2).copied())
        .flatten()
        .map(Unwrapped::Args)
}

/// Unwrap `_rebuild_parameter` and `_rebuild_from_type_v2` down to the
/// `_rebuild_tensor_v2` call, if that is what the value is.
fn tensor_call(p: &Parsed, mut value: Id) -> Option<Id> {
    for _ in 0..=MAX_WRAPPERS {
        match unwrap_once(p, value)? {
            Unwrapped::Args(args) => return Some(args),
            Unwrapped::Inner(next) => value = next,
        }
    }
    None
}

/// Look through `BUILD` wrappers (which only carry attributes) to the dict.
fn dict_entries(p: &Parsed, mut id: Id) -> Option<&[(Id, Id)]> {
    loop {
        match &p.arena[id] {
            Node::Build(obj, _) => id = *obj,
            Node::Dict(d) => return Some(d),
            _ => return None,
        }
    }
}

/// The dict that holds the tensors: the root, or the entry `key` of it.
fn state_dict<'a>(p: &'a Parsed, key: Option<&str>) -> Result<&'a [(Id, Id)], PickleError> {
    let root = dict_entries(p, p.root).ok_or_else(|| PickleError::NoStateDict {
        detail: "the top-level object is not a dict".to_string(),
    })?;
    let Some(key) = key else { return Ok(root) };
    let found = root.iter().find(|(k, _)| string(p, *k) == Some(key));
    found
        .and_then(|&(_, v)| dict_entries(p, v))
        .ok_or_else(|| PickleError::NoStateDict {
            detail: format!("no dict under key {key:?}"),
        })
}

/// The tensors of the checkpoint, and the names of entries that are not
/// tensors (a population that would otherwise shrink silently).
pub(crate) fn tensors(
    p: &Parsed,
    key: Option<&str>,
    prefix: &str,
) -> Result<(Vec<Rebuilt>, Vec<String>), PickleError> {
    let (mut out, mut skipped) = (Vec::new(), Vec::new());
    for &(k, v) in state_dict(p, key)? {
        let Some(name) = string(p, k) else {
            skipped.push("<non-string key>".to_string());
            continue;
        };
        match tensor_call(p, v) {
            Some(args) => out.push(rebuild_args(p, name, args, prefix)?),
            None => skipped.push(name.to_string()),
        }
    }
    Ok((out, skipped))
}
