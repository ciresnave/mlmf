// SPDX-License-Identifier: MIT OR Apache-2.0
//! What a LoRA adapter WRITER will later have to satisfy, stated on the read
//! side first (ROADMAP S-65, design `docs/superpowers/specs/2026-11-11-lora-adapter-files-design.md`
//! section 4 item 1). No writer exists and none is started here.
//!
//! The comparison is `compare`: two sets of held tensors (name, dtype name,
//! shape, payload bytes) are equal only if every tensor matches name by name,
//! compared as a SET, never a sequence. Here it holds what `mlmf-safetensors`
//! reads against what the upstream `safetensors` crate reads from the same
//! bytes, which share no code. When the writers exist, the round trip is the
//! same function applied to "read, write, read".
//!
//! Two populations:
//! - a synthetic adapter built in memory by the upstream crate, so the test
//!   always runs (CI included, NOT corpus-gated);
//! - the real adapter files in the LoRA corpus (`MLMF_LORA_CORPUS`, default
//!   `C:/Models/lora-corpus`), which announce `SKIPPED` or `PARTIAL: n of m`
//!   when the corpus is absent or short, and fail under `MLMF_CORPUS_REQUIRED`.
//!
//! Every comparison is sabotaged below and must go red.

use std::io::Write;
use std::path::{Path, PathBuf};

use mlmf_core::{Encoding, NOTICE_TOKEN, TensorContainer};
use mlmf_lora::pair_tensors;
use mlmf_safetensors::{name_of, parse_header, parse_tensors};
use mlmf_test_support as armed;
use safetensors::SafeTensors;
use safetensors::tensor::{Dtype, TensorView};

/// One tensor as held: the four facts a round trip must preserve.
#[derive(Clone, Debug, PartialEq, Eq)]
struct Held {
    name: String,
    dtype: String,
    shape: Vec<usize>,
    data: Vec<u8>,
}

/// Held tensors read by `mlmf-safetensors`.
fn hold_mlmf(bytes: &[u8]) -> Vec<Held> {
    let header = parse_header(bytes).expect("mlmf header");
    let (container, _report) =
        parse_tensors(bytes, &header, "lora_readback").expect("mlmf tensors");
    container
        .tensors()
        .iter()
        .map(|d| {
            let Encoding::Dense(dt) = d.encoding else {
                panic!("{}: adapter tensors are dense", d.name);
            };
            Held {
                name: d.name.clone(),
                dtype: name_of(dt)
                    .expect("dtype has a safetensors name")
                    .to_owned(),
                shape: d.shape.dims().to_vec(),
                data: container.tensor_bytes(d).expect("bytes").into_owned(),
            }
        })
        .collect()
}

/// Held tensors read by the upstream crate.
fn hold_upstream(bytes: &[u8]) -> Vec<Held> {
    let st = SafeTensors::deserialize(bytes).expect("upstream reads");
    st.tensors()
        .into_iter()
        .map(|(name, v)| Held {
            name,
            dtype: format!("{:?}", v.dtype()),
            shape: v.shape().to_vec(),
            data: v.data().to_vec(),
        })
        .collect()
}

/// Equal as sets, name by name. `Err` names the first difference.
fn compare(a: &[Held], b: &[Held]) -> Result<(), String> {
    let mut a = a.to_vec();
    let mut b = b.to_vec();
    a.sort_by(|x, y| x.name.cmp(&y.name));
    b.sort_by(|x, y| x.name.cmp(&y.name));
    if a.len() != b.len() {
        return Err(format!("tensor count {} vs {}", a.len(), b.len()));
    }
    for (x, y) in a.iter().zip(&b) {
        if x.name != y.name {
            return Err(format!("name {} vs {}", x.name, y.name));
        }
        if x.dtype != y.dtype {
            return Err(format!("{}: dtype {} vs {}", x.name, x.dtype, y.dtype));
        }
        if x.shape != y.shape {
            return Err(format!("{}: shape {:?} vs {:?}", x.name, x.shape, y.shape));
        }
        if x.data != y.data {
            return Err(format!("{}: payload bytes differ", x.name));
        }
    }
    Ok(())
}

/// A PEFT-shaped adapter: three modules of A/B pairs at different ranks and
/// widths, one non-LoRA tensor, distinct bytes per tensor so a swap cannot
/// compare equal. Built by the upstream crate.
fn synthetic_image() -> Vec<u8> {
    let specs: [(&str, Dtype, [usize; 2]); 7] = [
        (
            "base_model.model.layers.0.q_proj.lora_A.weight",
            Dtype::F32,
            [4, 16],
        ),
        (
            "base_model.model.layers.0.q_proj.lora_B.weight",
            Dtype::F32,
            [16, 4],
        ),
        (
            "base_model.model.layers.0.v_proj.lora_A.weight",
            Dtype::F16,
            [8, 16],
        ),
        (
            "base_model.model.layers.0.v_proj.lora_B.weight",
            Dtype::F16,
            [16, 8],
        ),
        (
            "base_model.model.layers.1.q_proj.lora_A.weight",
            Dtype::BF16,
            [2, 16],
        ),
        (
            "base_model.model.layers.1.q_proj.lora_B.weight",
            Dtype::BF16,
            [16, 2],
        ),
        (
            "base_model.model.lm_head.base_layer.weight",
            Dtype::F32,
            [3, 5],
        ),
    ];
    let owned: Vec<(String, Dtype, Vec<usize>, Vec<u8>)> = specs
        .iter()
        .enumerate()
        .map(|(i, (n, dt, sh))| {
            let len = sh[0] * sh[1] * (dt.bitsize() / 8);
            let data = (0..len).map(|b| (b * 13 + i * 41 + 3) as u8).collect();
            ((*n).to_owned(), *dt, sh.to_vec(), data)
        })
        .collect();
    let views: Vec<(String, TensorView<'_>)> = owned
        .iter()
        .map(|(n, dt, sh, d)| {
            (
                n.clone(),
                TensorView::new(*dt, sh.clone(), d).expect("view"),
            )
        })
        .collect();
    safetensors::serialize(views, None).expect("serialize")
}

#[test]
fn the_two_readers_hold_the_same_synthetic_adapter() {
    let image = synthetic_image();
    let (m, u) = (hold_mlmf(&image), hold_upstream(&image));
    assert_eq!(m.len(), 7, "population: all seven tensors read");
    compare(&m, &u).expect("mlmf and upstream agree byte for byte");
}

#[test]
fn held_tensors_pair_and_the_halves_share_a_rank() {
    let image = synthetic_image();
    let header = parse_header(&image).expect("header");
    let (container, _) = parse_tensors(&image, &header, "lora_readback").expect("tensors");
    let p = pair_tensors(container.tensors());
    assert_eq!(p.pairs.len(), 3);
    assert_eq!(p.others.len(), 1);
    assert!(p.orphans.is_empty());
    // A is [r, in], B is [out, r]: the rank is read from the shapes, not declared.
    let ranks: Vec<(usize, usize)> = p
        .pairs
        .iter()
        .map(|x| (x.a.shape.dims()[0], x.b.shape.dims()[1]))
        .collect();
    assert!(ranks.iter().all(|(a, b)| a == b), "{ranks:?}");
    let mut seen: Vec<usize> = ranks.iter().map(|r| r.0).collect();
    seen.sort_unstable();
    assert_eq!(seen, vec![2, 4, 8], "three different ranks were exercised");
}

// ---- sabotage: each comparison must be able to fail ------------------------

fn base() -> Vec<Held> {
    hold_mlmf(&synthetic_image())
}

#[test]
fn sabotage_a_flipped_payload_byte_is_caught() {
    let a = base();
    let mut b = a.clone();
    let before = b[2].data[0];
    b[2].data[0] ^= 0x01;
    assert_ne!(b[2].data[0], before, "mutation applied");
    let e = compare(&a, &b).expect_err("must go red");
    assert!(e.contains("payload bytes differ"), "{e}");
}

#[test]
fn sabotage_a_dropped_tensor_is_caught() {
    let a = base();
    let mut b = a.clone();
    assert!(b.pop().is_some(), "mutation applied");
    assert!(compare(&a, &b).expect_err("must go red").contains("count"));
}

#[test]
fn sabotage_a_changed_shape_with_the_same_bytes_is_caught() {
    let a = base();
    let mut b = a.clone();
    let i = b
        .iter()
        .position(|h| h.shape == [4, 16])
        .expect("anchor present once");
    b[i].shape = vec![16, 4];
    assert!(compare(&a, &b).expect_err("must go red").contains("shape"));
}

#[test]
fn sabotage_a_same_width_dtype_swap_is_caught() {
    let a = base();
    let mut b = a.clone();
    let i = b
        .iter()
        .position(|h| h.dtype == "F16")
        .expect("anchor present");
    assert_ne!(b[i].dtype, "BF16");
    b[i].dtype = "BF16".into();
    assert!(compare(&a, &b).expect_err("must go red").contains("dtype"));
}

#[test]
fn sabotage_a_renamed_tensor_is_caught() {
    let a = base();
    let mut b = a.clone();
    b[0].name.push('x');
    assert!(compare(&a, &b).is_err());
}

#[test]
fn a_reordered_copy_still_compares_equal() {
    let a = base();
    let mut b = a.clone();
    b.reverse();
    assert_ne!(a, b, "the order really changed");
    compare(&a, &b).expect("sets, not sequences");
}

// ---- the real adapters -----------------------------------------------------

fn corpus_root() -> Option<PathBuf> {
    let root =
        std::env::var("MLMF_LORA_CORPUS").unwrap_or_else(|_| "C:/Models/lora-corpus".to_string());
    let p = PathBuf::from(root);
    p.is_dir().then_some(p)
}

fn adapter_files(root: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let Ok(rd) = std::fs::read_dir(&dir) else {
            continue;
        };
        for e in rd.flatten() {
            let p = e.path();
            if p.is_dir() {
                // `.cache` holds download metadata, never adapters.
                if p.file_name().is_some_and(|n| n != ".cache") {
                    stack.push(p);
                }
            } else if p
                .file_name()
                .is_some_and(|n| n == "adapter_model.safetensors")
            {
                out.push(p);
            }
        }
    }
    out.sort();
    out
}

/// Real adapters measured by S-65 step 1 (`lora-measurements.tsv`).
const EXPECTED_FILES: usize = 7;

#[test]
fn real_adapters_read_back_bytewise_against_the_upstream_reader() {
    let name = "real_adapters_read_back_bytewise_against_the_upstream_reader";
    let Some(root) = corpus_root() else {
        assert!(
            !armed::armed("MLMF_CORPUS_REQUIRED"),
            "{name}: MLMF_CORPUS_REQUIRED is set and no corpus at MLMF_LORA_CORPUS"
        );
        let _ = writeln!(
            std::io::stderr(),
            "{NOTICE_TOKEN}: SKIPPED {name} -- no LoRA corpus (set MLMF_LORA_CORPUS)"
        );
        return;
    };
    let files = adapter_files(&root);
    let mut checked = 0usize;
    for f in &files {
        let bytes = std::fs::read(f).expect("read adapter");
        let (m, u) = (hold_mlmf(&bytes), hold_upstream(&bytes));
        assert!(!m.is_empty(), "{}: nothing read", f.display());
        compare(&m, &u).unwrap_or_else(|e| panic!("{}: {e}", f.display()));
        checked += m.len();
    }
    let _ = writeln!(
        std::io::stderr(),
        "{NOTICE_TOKEN}: {} {name}: {} of {EXPECTED_FILES} adapter files, {checked} tensors",
        if files.len() == EXPECTED_FILES {
            "FULL"
        } else {
            "PARTIAL"
        },
        files.len()
    );
    assert!(
        files.len() <= EXPECTED_FILES + 8 && !files.is_empty(),
        "population {} is not plausible",
        files.len()
    );
    assert!(
        files.len() == EXPECTED_FILES || !armed::armed("MLMF_CORPUS_REQUIRED"),
        "MLMF_CORPUS_REQUIRED: {} of {EXPECTED_FILES} adapter files present",
        files.len()
    );
}
