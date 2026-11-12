// SPDX-License-Identifier: MIT OR Apache-2.0
//! ROADMAP S-65 step 1: what real LoRA adapter files look like, replayed from
//! `data/lora-measurements.tsv`.
//!
//! The table was produced by `tools/lora_measure.py`, an independent reader that
//! shares no code with any mlmf crate. These tests read only the table, so they
//! pin what was MEASURED; they do not exercise a reader (none exists yet).
//! The set is three files, so every claim below is "in these files", not "in all
//! adapters": two PEFT (rank 16 and 32, one with an `lm_head` target) and one
//! llama.cpp-converted GGUF (rank 32, llama). No `rank_pattern`, `use_rslora`,
//! string `target_modules` or embedding adapter is in the set; step 2 must fetch
//! those before it fixes a type for them.

const TABLE: &str = include_str!("../data/lora-measurements.tsv");

const OPT: &str = "peft-internal-testing--opt-350m-lora/adapter_model.safetensors";
const LLAMA: &str =
    "ggml-org--lora-tests/LlamaForCausalLM/hidden_size=64/lora/adapter_model.safetensors";
const GGUF: &str =
    "makaveli10--tinyllama-fc-lora-gguf/tinyllama-function-call-lora-adapter-250424-f16.gguf";

struct Tensor {
    name: String,
    dtype: String,
    shape: Vec<u64>,
}

fn rows(file: &str, kind: &str) -> Vec<Vec<String>> {
    TABLE
        .lines()
        .filter(|l| !l.starts_with('#') && !l.starts_with("file\t"))
        .map(|l| l.split('\t').map(String::from).collect::<Vec<_>>())
        .inspect(|c| {
            assert_eq!(
                c.len(),
                if c[1] == "kv" { 4 } else { 5 },
                "malformed row {c:?}"
            )
        })
        .filter(|c| c[0] == file && c[1] == kind)
        .collect()
}

fn tensors(file: &str) -> Vec<Tensor> {
    rows(file, "tensor")
        .into_iter()
        .map(|c| Tensor {
            name: c[2].clone(),
            dtype: c[3].clone(),
            shape: c[4]
                .trim_matches(['[', ']'])
                .split(", ")
                .map(|d| d.parse().expect("dimension"))
                .collect(),
        })
        .collect()
}

/// Every `.lora_A.weight` pairs with a `.lora_B.weight` of the same module and
/// the shapes are `A = [r, in]`, `B = [out, r]`. Returns `(pairs, ranks seen)`.
fn peft_pairs(file: &str) -> (usize, Vec<u64>) {
    let all = tensors(file);
    let mut ranks = Vec::new();
    let mut pairs = 0;
    for a in all.iter().filter(|t| t.name.ends_with(".lora_A.weight")) {
        let module = a.name.trim_end_matches(".lora_A.weight");
        let b = all
            .iter()
            .find(|t| t.name == format!("{module}.lora_B.weight"))
            .unwrap_or_else(|| panic!("orphan A: {}", a.name));
        assert_eq!(a.shape.len(), 2, "{}", a.name);
        assert_eq!(b.shape.len(), 2, "{}", b.name);
        assert_eq!(a.shape[0], b.shape[1], "rank disagrees for {module}");
        assert!(
            a.name.starts_with("base_model.model."),
            "prefix: {}",
            a.name
        );
        assert_eq!((a.dtype.as_str(), b.dtype.as_str()), ("F32", "F32"));
        if !ranks.contains(&a.shape[0]) {
            ranks.push(a.shape[0]);
        }
        pairs += 1;
    }
    (pairs, ranks)
}

#[test]
fn peft_opt_is_all_pairs_under_the_peft_prefix() {
    assert_eq!(tensors(OPT).len(), 96);
    assert_eq!(peft_pairs(OPT), (48, vec![16]));
}

#[test]
fn peft_llama_has_a_third_tensor_class_that_is_not_an_orphan() {
    // 32 layers x 6 modules + lm_head = 193 pairs; plus `lm_head.base_layer.weight`,
    // a full-size base weight that PEFT saved beside the adapter when lm_head was targeted.
    assert_eq!(tensors(LLAMA).len(), 387);
    assert_eq!(peft_pairs(LLAMA), (193, vec![32]));
    let other: Vec<String> = tensors(LLAMA)
        .into_iter()
        .filter(|t| !t.name.contains(".lora_A.") && !t.name.contains(".lora_B."))
        .map(|t| format!("{} {:?}", t.name, t.shape))
        .collect();
    assert_eq!(
        other,
        ["base_model.model.lm_head.base_layer.weight [128256, 64]"]
    );
}

#[test]
fn peft_b_out_dimension_follows_the_projection_not_the_hidden_size() {
    // k_proj/v_proj B is [16, 32] against q_proj's [64, 32]: grouped-query attention.
    // A reader must take `out` from the tensor, never from a config's hidden size.
    let all = tensors(LLAMA);
    let b = |m: &str| {
        all.iter()
            .find(|t| {
                t.name == format!("base_model.model.model.layers.0.self_attn.{m}.lora_B.weight")
            })
            .expect(m)
            .shape
            .clone()
    };
    assert_eq!((b("q_proj"), b("k_proj")), (vec![64, 32], vec![16, 32]));
}

#[test]
fn gguf_adapter_names_dimension_order_and_rank() {
    let all = tensors(GGUF);
    assert_eq!(all.len(), 308);
    let mut ranks = Vec::new();
    let mut pairs = 0;
    for a in all.iter().filter(|t| t.name.ends_with(".weight.lora_a")) {
        let base = a.name.trim_end_matches(".lora_a");
        let b = all
            .iter()
            .find(|t| t.name == format!("{base}.lora_b"))
            .unwrap_or_else(|| panic!("orphan lora_a: {}", a.name));
        // ggml order is the reverse of PEFT's: lora_a ne = [in, r], lora_b ne = [r, out].
        assert_eq!(a.shape[1], b.shape[0], "rank disagrees for {base}");
        assert_eq!((a.dtype.as_str(), b.dtype.as_str()), ("F32", "F32"));
        if !ranks.contains(&a.shape[1]) {
            ranks.push(a.shape[1]);
        }
        pairs += 1;
    }
    assert_eq!((pairs, ranks), (154, vec![32]));
    assert!(
        all.iter().all(|t| t.name.starts_with("blk.")),
        "a non-block tensor appeared"
    );
}

#[test]
fn gguf_adapter_declares_alpha_and_no_rank() {
    let kv: Vec<(String, String)> = rows(GGUF, "kv")
        .into_iter()
        .map(|c| (c[2].clone(), c[3].clone()))
        .collect();
    let get = |k: &str| kv.iter().find(|(n, _)| n == k).map(|(_, v)| v.as_str());
    assert_eq!(get("general.type"), Some("'adapter'"));
    assert_eq!(get("adapter.type"), Some("'lora'"));
    assert_eq!(get("general.architecture"), Some("'llama'"));
    assert_eq!(get("adapter.lora.alpha"), Some("16.0"));
    // The rank is a tensor dimension, not a key; and no optional key was invented.
    assert!(
        kv.iter()
            .all(|(k, _)| k != "adapter.lora.r" && !k.ends_with(".rank"))
    );
    assert_eq!(get("adapter.lora.task_name"), None);
    assert_eq!(get("adapter.lora.prompt_prefix"), None);
}

#[test]
fn the_table_covers_exactly_the_three_files() {
    let mut files: Vec<&str> = TABLE
        .lines()
        .filter(|l| !l.starts_with('#') && !l.starts_with("file\t") && !l.is_empty())
        .filter_map(|l| l.split('\t').next())
        .collect();
    files.dedup();
    assert_eq!(files, [OPT, LLAMA, GGUF]);
}
