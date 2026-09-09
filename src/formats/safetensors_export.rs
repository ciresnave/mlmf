// Copyright 2023 MLMF Contributors
// Licensed under Apache License v2.0

//! SafeTensors export functionality for saving models in the SafeTensors format.

use crate::{Error, LoadedModel};
use std::path::Path;

/// Save a loaded model as SafeTensors format.
///
/// # ⚠️ What this did until 2026-09-09
///
/// It ignored `model` and `preserve_metadata` entirely, wrote an 8-byte length
/// plus a header holding nothing but a `__metadata__` stamp, and returned
/// `Ok(())`. **Measured by round-trip: three tensors in, a 40-byte file out,
/// which then LOADS SUCCESSFULLY WITH ZERO TENSORS.** Not a corrupt file a
/// reader would reject -- a valid one that passes every check and contains
/// nothing. The caller's evidence that the save happened was a file.
///
/// Its comment said *"full tensor data conversion would be needed"*. The crate
/// already declared `safetensors = "0.7"` as a direct dependency and never
/// used it anywhere in `src/`, and candle implements `safetensors::View` for
/// `Tensor` -- which is how `candlelight::safetensors::save` works. The
/// conversion was one call.
pub fn save_as_safetensors(
    model: &LoadedModel,
    path: &Path,
    preserve_metadata: bool,
) -> Result<(), Error> {
    let metadata = preserve_metadata.then(|| {
        let mut m = std::collections::HashMap::new();
        m.insert("converted_by".to_string(), "mlmf".to_string());
        m.insert(
            "conversion_time".to_string(),
            chrono::Utc::now().to_rfc3339(),
        );
        m
    });

    write_safetensors(&model.raw_tensors, metadata, path)
}

/// Save tensors in SafeTensors format with custom metadata.
///
/// # ⚠️ What this did until 2026-09-09
///
/// It built a header containing only `__metadata__`, **never read the
/// `tensors` argument at all**, and returned `Ok(())`. Its caller
/// `lora::save_adapter` therefore reported that a LoRA adapter had been
/// written and produced a file with none of its weights -- and that file
/// loads without error, so nothing downstream reports a problem either.
///
/// ⚠️ **A bad read fails in front of the person who ran it. A bad write fails
/// later, in front of whoever opens the file, with the producer gone.**
///
/// The metadata matters as much as the tensors here: `save_adapter` passes
/// `format: "pt"` and `peft_type: "LORA"`, which is how PEFT tooling
/// identifies an adapter. Writing the tensors and dropping the metadata would
/// be the same defect one field over, so both are written.
pub fn save_safetensors_with_metadata(
    path: &Path,
    tensors: &std::collections::HashMap<String, candlelight::Tensor>,
    metadata: &std::collections::HashMap<String, String>,
) -> Result<(), Error> {
    let metadata = (!metadata.is_empty()).then(|| metadata.clone());
    write_safetensors(tensors, metadata, path)
}

/// The one place this module serializes.
///
/// Both public entry points route through here so that a fix or a defect
/// lands in one function rather than in two that drifted apart -- which is
/// how these two came to carry the same stub, worded differently.
fn write_safetensors(
    tensors: &std::collections::HashMap<String, candlelight::Tensor>,
    metadata: Option<std::collections::HashMap<String, String>>,
    path: &Path,
) -> Result<(), Error> {
    // `&Tensor` implements `safetensors::View` via candle, so the tensors are
    // serialized without a copy through an intermediate buffer.
    safetensors::tensor::serialize_to_file(tensors.iter(), metadata, path).map_err(|e| {
        Error::model_saving(format!(
            "Failed to write SafeTensors to {}: {e}",
            path.display()
        ))
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use tempfile::TempDir;

    /// Three tensors, small, with distinct shapes so a swap is visible.
    fn sample_tensors() -> HashMap<String, candlelight::Tensor> {
        let mut t = HashMap::new();
        for (n, r, c) in [("a.weight", 2, 3), ("b.weight", 4, 4), ("c.weight", 1, 5)] {
            t.insert(
                n.to_string(),
                candlelight::Tensor::zeros(
                    (r, c),
                    candlelight::DType::F32,
                    &candlelight::Device::Cpu,
                )
                .expect("tensor"),
            );
        }
        t
    }

    /// The `__metadata__` map as it sits ON DISK, parsed from the file's own
    /// header rather than from a library's reading of it.
    ///
    /// A SafeTensors file is an 8-byte little-endian header length followed by
    /// that many bytes of JSON. Reading it directly means this asserts what was
    /// written, not what a deserializer is willing to accept.
    fn metadata_on_disk(path: &std::path::Path) -> serde_json::Value {
        let raw = std::fs::read(path).expect("read back");
        let len = u64::from_le_bytes(raw[..8].try_into().expect("8 bytes")) as usize;
        let header: serde_json::Value =
            serde_json::from_slice(&raw[8..8 + len]).expect("header is JSON");
        header
            .get("__metadata__")
            .cloned()
            .unwrap_or(serde_json::Value::Null)
    }

    /// ⚠️ THE CONVERSION PATH, END TO END: load a real model, save it, load the
    /// result.
    ///
    /// `save_as_safetensors` is what `conversion.rs` calls for
    /// `ConversionFormat::SafeTensors`. The three tests above cover the shared
    /// writer; this one covers **this function's own job** — passing
    /// `model.raw_tensors` rather than, say, an empty map it happens to have in
    /// scope. Until 2026-09-09 it ignored `model` entirely, so a conversion
    /// produced a valid file with none of the model in it.
    #[test]
    fn a_converted_model_still_has_its_tensors() {
        let dir = TempDir::new().expect("temp dir");
        std::fs::write(
            dir.path().join("config.json"),
            r#"{
                "vocab_size": 32,
                "hidden_size": 8,
                "num_attention_heads": 2,
                "num_hidden_layers": 1,
                "intermediate_size": 16,
                "max_position_embeddings": 16
            }"#,
        )
        .expect("config.json");

        // LLaMA-style names so architecture detection succeeds; the load
        // refuses without it and this test is about a SUCCESSFUL conversion.
        let names = [
            "model.embed_tokens.weight",
            "model.layers.0.self_attn.q_proj.weight",
            "lm_head.weight",
        ];
        let mut source = HashMap::new();
        for n in names {
            source.insert(
                n.to_string(),
                candlelight::Tensor::zeros(
                    (2, 2),
                    candlelight::DType::F32,
                    &candlelight::Device::Cpu,
                )
                .expect("tensor"),
            );
        }
        candlelight::safetensors::save(&source, dir.path().join("model.safetensors"))
            .expect("fixture");

        let model =
            crate::loader::load_safetensors(dir.path(), crate::loader::LoadOptions::default())
                .expect("the fixture loads");
        assert_eq!(
            model.raw_tensors.len(),
            names.len(),
            "the model under test actually holds tensors, so a zero below \
             means the SAVE dropped them rather than the load"
        );

        let out = dir.path().join("converted.safetensors");
        save_as_safetensors(&model, &out, true).expect("save");

        let round_tripped = candlelight::safetensors::load(&out, &candlelight::Device::Cpu)
            .expect("the converted file loads");
        assert_eq!(
            round_tripped.len(),
            names.len(),
            "the conversion carries every tensor; got {} of {}",
            round_tripped.len(),
            names.len()
        );

        // `preserve_metadata: true` must actually put something there.
        let meta = metadata_on_disk(&out);
        assert_eq!(
            meta.get("converted_by").and_then(|v| v.as_str()),
            Some("mlmf"),
            "preserve_metadata stamps the file: {meta}"
        );
    }

    /// ⚠️ THE TENSORS MUST REACH THE FILE.
    ///
    /// Until this was fixed, `save_safetensors_with_metadata` never read its
    /// `tensors` argument. Measured by round-trip: three tensors in, a 40-byte
    /// file out, **which loaded successfully with ZERO tensors** — a valid file
    /// containing nothing, not a corrupt one a reader would reject.
    #[test]
    fn tensors_survive_the_round_trip() {
        let dir = TempDir::new().expect("temp dir");
        let path = dir.path().join("out.safetensors");
        let written = sample_tensors();

        let mut meta = HashMap::new();
        meta.insert("format".to_string(), "pt".to_string());
        meta.insert("peft_type".to_string(), "LORA".to_string());

        save_safetensors_with_metadata(&path, &written, &meta).expect("save");

        let read_back = candlelight::safetensors::load(&path, &candlelight::Device::Cpu)
            .expect("the file we just wrote loads");

        assert_eq!(
            read_back.len(),
            written.len(),
            "every tensor written comes back; got {} of {}",
            read_back.len(),
            written.len()
        );
        for (name, t) in &written {
            let got = read_back
                .get(name)
                .unwrap_or_else(|| panic!("{name} is in the file"));
            assert_eq!(
                got.dims(),
                t.dims(),
                "{name} keeps its shape, so a name/shape mix-up is visible"
            );
        }
    }

    /// ⚠️ AND SO MUST THE METADATA.
    ///
    /// `lora::save_adapter` passes `format: "pt"` and `peft_type: "LORA"`,
    /// which is how PEFT tooling identifies an adapter. Writing the tensors and
    /// dropping the metadata would be the same defect one field over, so this
    /// is a separate assertion rather than a clause of the one above.
    #[test]
    fn metadata_survives_the_round_trip() {
        let dir = TempDir::new().expect("temp dir");
        let path = dir.path().join("out.safetensors");
        let mut meta = HashMap::new();
        meta.insert("format".to_string(), "pt".to_string());
        meta.insert("peft_type".to_string(), "LORA".to_string());

        save_safetensors_with_metadata(&path, &sample_tensors(), &meta).expect("save");

        let on_disk = metadata_on_disk(&path);
        assert_eq!(on_disk.get("format").and_then(|v| v.as_str()), Some("pt"));
        assert_eq!(
            on_disk.get("peft_type").and_then(|v| v.as_str()),
            Some("LORA"),
            "the adapter marker PEFT reads is present: {on_disk}"
        );
    }

    /// ⚠️ NON-VACUITY FOR BOTH TESTS ABOVE.
    ///
    /// A writer that produced an empty-but-valid file would pass neither, but
    /// this states the discriminator explicitly: the old implementation wrote
    /// **40 bytes** regardless of input. Three F32 tensors of 6, 16 and 5
    /// elements are 108 bytes of data alone, so a byte count settles it without
    /// depending on any reader.
    #[test]
    fn the_file_is_not_a_bare_header() {
        let dir = TempDir::new().expect("temp dir");
        let path = dir.path().join("out.safetensors");
        save_safetensors_with_metadata(&path, &sample_tensors(), &HashMap::new()).expect("save");

        let len = std::fs::metadata(&path).expect("stat").len();
        assert!(
            len > 108,
            "a header-only file was 40 bytes; 27 f32 values are 108 bytes of \
             data alone, so {len} bytes means the data section is missing"
        );
    }

    // `test_save_as_safetensors` was deleted here, not moved.
    //
    // ⚠️ It made a temp directory, asserted NOTHING, and every line of its
    // body was commented out -- "Test temporarily disabled due to LoadedModel
    // complexity". It passed, and it carried the exact name a reader checks
    // when asking whether `save_as_safetensors` is covered. It sat beside a
    // function that silently discarded every tensor it was given.
    //
    // A TEST'S NAME IS A CLAIM ABOUT WHAT IT COVERS, and this one made that
    // claim while asserting nothing at all. Its one `assert!` was commented
    // out too, so a scan for assertion-free tests that counted the token
    // would have cleared it.
    //
    // Its stated blocker was constructing a `LoadedModel` by hand.
    // `a_converted_model_still_has_its_tensors` above avoids that entirely by
    // LOADING one from a fixture -- also a more honest subject, since it
    // exercises the path `conversion.rs` actually takes.
}
