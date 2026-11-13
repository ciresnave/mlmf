// SPDX-License-Identifier: MIT OR Apache-2.0
//! A real GGUF LoRA adapter, read, written back as read with
//! `mlmf_gguf::encode_gguf`, and read again (ROADMAP S-65 step 4, ruling 421:
//! same form, tensors arriving already encoded, no scale accessor).
//!
//! MLMF needs no adapter-specific writer for this form: an adapter is a GGUF
//! file whose `general.type` is `adapter`, so the existing writer carries every
//! key and every tensor. What this measures is that nothing is lost, added or
//! reordered into a different meaning on the way round (design note section 4):
//!
//! 1. every metadata key and value equal, so an absent `adapter.lora.alpha`
//!    stays absent and nothing is added;
//! 2. every tensor equal by name, ggml type, dims and payload bytes;
//! 3. `GgufAdapterInfo` and the A/B pairing equal before and after.
//!
//! Each comparison is sabotaged and must go red. The corpus file
//! (`MLMF_LORA_CORPUS`, default `C:/Models/lora-corpus`) is one adapter
//! (llama.cpp's converter output, F16), so this announces `SKIPPED` without it
//! and fails under `MLMF_CORPUS_REQUIRED`. Its limit, stated: one adapter file,
//! F16 tensors only; no F32 or quantized adapter is exercised.

use std::collections::BTreeMap;
use std::io::Write;
use std::path::PathBuf;

use mlmf_core::{Encoding, MetaValue, MetadataSource, NOTICE_TOKEN, TensorContainer};
use mlmf_ggml::GgmlType;
use mlmf_gguf::{GgufMetadata, GgufTensorToWrite, GgufVersion, encode_gguf, parse_tensors};
use mlmf_lora::{GgufAdapterInfo, pair_gguf_tensors};
use mlmf_test_support as armed;

const FILE: &str =
    "makaveli10--tinyllama-fc-lora-gguf/tinyllama-function-call-lora-adapter-250424-f16.gguf";

/// A GGUF file as held: all metadata, and every tensor's type, dims, bytes.
#[derive(Clone, Debug, PartialEq)]
struct Held {
    meta: BTreeMap<String, MetaValue>,
    tensors: BTreeMap<String, (String, Vec<usize>, Vec<u8>)>,
}

fn hold(bytes: &[u8]) -> Held {
    let (md, _) = GgufMetadata::parse(bytes, "lora_gguf_roundtrip").expect("metadata");
    let (ts, _) = parse_tensors(bytes, &md, "lora_gguf_roundtrip").expect("tensors");
    let meta = md
        .keys()
        .into_iter()
        .map(|k| (k.to_owned(), md.get(k).expect("listed key").clone()))
        .collect();
    let tensors = ts
        .tensors()
        .iter()
        .map(|d| {
            let ty = match d.encoding {
                Encoding::Dense(dt) => format!("{dt:?}"),
                Encoding::Blocked(b) => format!("{b:?}"),
            };
            let data = ts.tensor_bytes(d).expect("bytes").into_owned();
            (d.name.clone(), (ty, d.shape.dims().to_vec(), data))
        })
        .collect();
    Held { meta, tensors }
}

/// The writer under test: read as given, write back as given.
fn write_back(bytes: &[u8]) -> Vec<u8> {
    let (md, _) = GgufMetadata::parse(bytes, "lora_gguf_roundtrip").expect("metadata");
    let (ts, _) = parse_tensors(bytes, &md, "lora_gguf_roundtrip").expect("tensors");
    let keys = md.keys();
    let values: Vec<MetaValue> = keys
        .iter()
        .map(|k| md.get(k).expect("key").clone())
        .collect();
    let meta: Vec<(&str, &MetaValue)> = keys.iter().copied().zip(values.iter()).collect();
    let datas: Vec<_> = ts
        .tensors()
        .iter()
        .map(|d| ts.tensor_bytes(d).expect("bytes"))
        .collect();
    let dims: Vec<Vec<usize>> = ts
        .tensors()
        .iter()
        .map(|d| d.shape.dims().to_vec())
        .collect();
    let to_write: Vec<GgufTensorToWrite<'_>> = ts
        .tensors()
        .iter()
        .zip(&datas)
        .zip(&dims)
        .map(|((d, data), dims)| GgufTensorToWrite {
            name: &d.name,
            ggml_type: match d.encoding {
                Encoding::Dense(mlmf_core::DType::F16) => GgmlType::F16,
                Encoding::Dense(mlmf_core::DType::F32) => GgmlType::F32,
                other => panic!("{}: {other:?} not exercised by this test", d.name),
            },
            dims,
            data,
        })
        .collect();
    encode_gguf(&meta, &to_write, GgufVersion::V3).expect("encode")
}

/// `Err` names the first difference.
fn compare(a: &Held, b: &Held) -> Result<(), String> {
    compare_meta(&a.meta, &b.meta)?;
    compare_tensors(&a.tensors, &b.tensors)
}

type Meta = BTreeMap<String, MetaValue>;
type Tensors = BTreeMap<String, (String, Vec<usize>, Vec<u8>)>;

fn compare_meta(a: &Meta, b: &Meta) -> Result<(), String> {
    if a.len() != b.len() {
        return Err(format!("metadata key count {} vs {}", a.len(), b.len()));
    }
    for (k, v) in a {
        match b.get(k) {
            None => return Err(format!("metadata key {k} missing")),
            Some(w) if w != v => return Err(format!("metadata {k}: {v:?} vs {w:?}")),
            Some(_) => {}
        }
    }
    Ok(())
}

fn compare_tensors(a: &Tensors, b: &Tensors) -> Result<(), String> {
    if a.len() != b.len() {
        return Err(format!("tensor count {} vs {}", a.len(), b.len()));
    }
    for (n, (ty, dims, data)) in a {
        let Some((ty2, dims2, data2)) = b.get(n) else {
            return Err(format!("tensor {n} missing"));
        };
        if ty != ty2 {
            return Err(format!("{n}: type {ty} vs {ty2}"));
        }
        if dims != dims2 {
            return Err(format!("{n}: dims {dims:?} vs {dims2:?}"));
        }
        if data != data2 {
            return Err(format!("{n}: payload bytes differ"));
        }
    }
    Ok(())
}

fn corpus_file() -> Option<PathBuf> {
    let root =
        std::env::var("MLMF_LORA_CORPUS").unwrap_or_else(|_| "C:/Models/lora-corpus".to_string());
    let p = PathBuf::from(root).join(FILE);
    p.is_file().then_some(p)
}

/// The adapter bytes, or `None` after announcing the skip (a failure when required).
fn adapter(test: &str) -> Option<Vec<u8>> {
    match corpus_file() {
        Some(p) => Some(std::fs::read(p).expect("read adapter")),
        None => {
            assert!(
                !armed::armed("MLMF_CORPUS_REQUIRED"),
                "{test}: MLMF_CORPUS_REQUIRED is set and the GGUF adapter is absent"
            );
            let _ = writeln!(
                std::io::stderr(),
                "{NOTICE_TOKEN}: SKIPPED {test} -- no GGUF adapter (set MLMF_LORA_CORPUS)"
            );
            None
        }
    }
}

#[test]
fn a_real_gguf_adapter_round_trips_same_form() {
    let Some(bytes) = adapter("a_real_gguf_adapter_round_trips_same_form") else {
        return;
    };
    let before = hold(&bytes);
    // Non-vacuity: the measured population (lora-measurements.tsv).
    assert_eq!(
        before.meta["general.type"],
        MetaValue::String("adapter".into())
    );
    assert!(
        before.tensors.len() > 100,
        "{} tensors",
        before.tensors.len()
    );
    let out = write_back(&bytes);
    let after = hold(&out);
    compare(&before, &after).expect("same metadata, same tensors, byte for byte");
}

#[test]
fn nothing_is_added_and_the_declared_fields_and_pairing_survive() {
    let Some(bytes) = adapter("nothing_is_added_and_the_declared_fields_and_pairing_survive")
    else {
        return;
    };
    let out = write_back(&bytes);
    let (m0, _) = GgufMetadata::parse(&bytes, "in").expect("in");
    let (m1, _) = GgufMetadata::parse(&out, "out").expect("out");
    let (mut k0, mut k1) = (m0.keys(), m1.keys());
    k0.sort_unstable();
    k1.sort_unstable();
    assert_eq!(k0, k1, "no key added, none dropped");
    let (i0, i1) = (
        GgufAdapterInfo::from_source(&m0),
        GgufAdapterInfo::from_source(&m1),
    );
    assert_eq!(i0, i1);
    assert!(i1.is_adapter());
    assert_eq!(i1.lora_alpha, Some(16.0), "declared alpha, not defaulted");
    let (t0, _) = parse_tensors(&bytes, &m0, "in").expect("in");
    let (t1, _) = parse_tensors(&out, &m1, "out").expect("out");
    let (p0, p1) = (
        pair_gguf_tensors(t0.tensors()),
        pair_gguf_tensors(t1.tensors()),
    );
    assert!(!p0.pairs.is_empty(), "population: the file has pairs");
    assert_eq!(p0.pairs.len(), p1.pairs.len());
    assert_eq!(p0.orphans.len(), p1.orphans.len());
    assert_eq!(p0.others.len(), p1.others.len());
}

// ---- sabotage ---------------------------------------------------------------

fn held_or_skip(test: &str) -> Option<Held> {
    adapter(test).map(|b| hold(&b))
}

#[test]
fn sabotage_a_flipped_payload_byte_is_caught() {
    let Some(a) = held_or_skip("sabotage_payload") else {
        return;
    };
    let mut b = a.clone();
    let t = b.tensors.values_mut().next().expect("a tensor");
    let before = t.2[0];
    t.2[0] ^= 1;
    assert_ne!(t.2[0], before, "mutation applied");
    assert!(compare(&a, &b).expect_err("red").contains("payload"));
}

#[test]
fn sabotage_an_added_metadata_key_is_caught() {
    let Some(a) = held_or_skip("sabotage_added_key") else {
        return;
    };
    let mut b = a.clone();
    assert!(
        b.meta
            .insert("adapter.lora.scale".into(), MetaValue::F32(1.0))
            .is_none()
    );
    assert!(compare(&a, &b).expect_err("red").contains("key count"));
}

#[test]
fn sabotage_a_changed_alpha_is_caught() {
    let Some(a) = held_or_skip("sabotage_alpha") else {
        return;
    };
    let mut b = a.clone();
    let old = b
        .meta
        .insert("adapter.lora.alpha".into(), MetaValue::F32(17.0));
    assert!(old.is_some(), "anchor present exactly as measured");
    assert!(
        compare(&a, &b)
            .expect_err("red")
            .contains("adapter.lora.alpha")
    );
}

#[test]
fn sabotage_a_dropped_tensor_and_a_changed_dim_are_caught() {
    let Some(a) = held_or_skip("sabotage_tensor") else {
        return;
    };
    let mut b = a.clone();
    let name = b.tensors.keys().next().expect("a tensor").clone();
    assert!(b.tensors.remove(&name).is_some());
    assert!(compare(&a, &b).expect_err("red").contains("tensor count"));
    let mut c = a.clone();
    let t = c.tensors.values_mut().next().expect("a tensor");
    t.1.reverse();
    t.1.push(1);
    assert!(compare(&a, &c).expect_err("red").contains("dims"));
}
