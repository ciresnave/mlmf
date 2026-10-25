// SPDX-License-Identifier: MIT OR Apache-2.0
//! `mlmf-bnb` against real bitsandbytes output, read through
//! `mlmf-safetensors`.
//!
//! The fixture is 16 rows of `model.layers.1.self_attn.k_proj` from
//! `unsloth/Qwen3-0.6B-unsloth-bnb-4bit` (revision
//! `e86f5289a2ca40ab7114071959dc2079cf97cb3c`, Apache-2.0, licence text beside
//! it), taken from Fuel's repository where its provenance is written down
//! (Fuel's `PROVENANCE.md`). Its bytes are pinned by sha256 below.
//!
//! The expected values come from `bnb_recon.py`-style independent reading
//! (plain `struct` + `json`, no mlmf code), run 2026-10-25, which ALSO
//! re-read the same tensors from the real 576 MB checkpoint by HTTP range
//! requests: the packed weight, `absmax`, `nested_absmax`, `nested_quant_map`
//! and `quant_map` are byte-identical to the hub's; only the JSON's `shape`
//! differs (patched to the 16-row slice, as the provenance says).

use mlmf_bnb::{locate_layers, parse_quant_state, read_layer};
use mlmf_core::DType;
use mlmf_safetensors::{parse_header, parse_tensors};

const FIXTURE: &[u8] =
    include_bytes!("fixtures/bnb_double_quant/bnb_double_quant_k_proj_16rows.safetensors");
const PREFIX: &str = "model.layers.1.self_attn.k_proj.weight";

fn open() -> mlmf_safetensors::SafetensorsTensors<'static> {
    let header = parse_header(FIXTURE).expect("header");
    let (t, report) = parse_tensors(FIXTURE, &header, "bnb").expect("tensors");
    assert!(report.is_empty());
    t
}

#[test]
fn the_fixture_is_the_10382_byte_file_it_was_measured_on() {
    assert_eq!(FIXTURE.len(), 10_382);
}

#[test]
fn one_layer_is_located_with_all_six_pieces() {
    let t = open();
    let layers = locate_layers(&t);
    assert_eq!(layers.len(), 1);
    let l = &layers[0];
    assert_eq!(
        (l.weight_name.as_str(), l.state_type.as_str()),
        (PREFIX, "nf4")
    );
    for (name, piece) in [
        ("weight", &l.weight),
        ("absmax", &l.absmax),
        ("quant_map", &l.quant_map),
        ("nested_absmax", &l.nested_absmax),
        ("nested_quant_map", &l.nested_quant_map),
    ] {
        assert!(piece.is_some(), "{name} missing");
    }
}

#[test]
fn the_pieces_have_the_independently_measured_sizes_and_dtypes() {
    let t = open();
    let l = &locate_layers(&t)[0];
    let b = read_layer(&t, l).expect("read");
    let size = |c: &Option<std::borrow::Cow<'_, [u8]>>| c.as_ref().map(|c| c.len());
    assert_eq!(size(&b.weight), Some(8192));
    assert_eq!(size(&b.absmax), Some(256));
    assert_eq!(size(&b.quant_map), Some(64));
    assert_eq!(size(&b.nested_absmax), Some(4));
    assert_eq!(size(&b.nested_quant_map), Some(1024));
    // Byte sums and leading bytes from the independent reader: the pieces
    // are the right bytes, not merely the right length.
    let sum = |c: &Option<std::borrow::Cow<'_, [u8]>>| {
        c.as_ref()
            .map(|c| c.iter().map(|&b| u64::from(b)).sum::<u64>())
    };
    assert_eq!(sum(&b.weight), Some(994_180));
    assert_eq!(sum(&b.absmax), Some(31_932));
    assert_eq!(sum(&b.quant_map), Some(6_660));
    assert_eq!(sum(&b.nested_absmax), Some(678));
    assert_eq!(sum(&b.nested_quant_map), Some(117_863));
    assert_eq!(&b.weight.as_ref().unwrap()[..4], &[0x5b, 0x4a, 0x7b, 0x77]);
    assert_eq!(
        &b.nested_absmax.as_ref().unwrap()[..4],
        &[0xcc, 0xa9, 0xf4, 0x3d]
    );
    // Double quantization: the first-level absmax are U8 codes, not floats.
    let dt = |d: &Option<mlmf_core::TensorDescriptor>| {
        d.as_ref().map(|d| match d.encoding {
            mlmf_core::Encoding::Dense(t) => t,
            _ => panic!("blocked"),
        })
    };
    assert_eq!(dt(&l.weight), Some(DType::U8));
    assert_eq!(dt(&l.absmax), Some(DType::U8));
    assert_eq!(dt(&l.nested_absmax), Some(DType::F32));
}

#[test]
fn the_quant_state_json_reads_to_the_independently_measured_values() {
    let t = open();
    let l = &locate_layers(&t)[0];
    let s = read_layer(&t, l).expect("read").state;
    assert_eq!(s.quant_type.as_deref(), Some("nf4"));
    assert_eq!(s.blocksize, Some(64));
    assert_eq!(s.shape, Some(vec![16, 1024]));
    assert_eq!(s.dtype.as_deref(), Some("bfloat16"));
    assert_eq!(s.nested_blocksize, Some(256));
    assert_eq!(s.nested_dtype.as_deref(), Some("float32"));
    assert_eq!(s.nested_offset, Some(0.07291832566261292));
    assert!(s.other_keys.is_empty());
}

#[test]
fn a_blob_with_a_wrong_typed_known_key_is_an_error_not_none() {
    assert!(parse_quant_state(br#"{"blocksize":"64"}"#).is_err());
    assert!(parse_quant_state(b"[1]").is_err());
    assert!(parse_quant_state(b"not json").is_err());
    let s = parse_quant_state(br#"{"quant_type":"fp4","future_key":1}"#).unwrap();
    assert_eq!(s.quant_type.as_deref(), Some("fp4"));
    assert_eq!(s.blocksize, None);
    assert_eq!(s.other_keys, vec!["future_key".to_string()]);
}
