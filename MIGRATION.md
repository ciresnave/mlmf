# Migration: the retired `mlmf` root package

**In 0.7.12 the single `mlmf` package that used to live at the root of this
repository was removed.** The repository is now a virtual workspace of the
`mlmf-*` crates (see [README.md](README.md)); those, at 0.7.11 and later, are
the supported crates.

- **The old code is not lost.** It is at the git tag `v0.7.11` (commit
  `5f977a725bfcf7583e9f9b813ac670babd6dbcdd`): `git show v0.7.11:src/lib.rs`.
- **crates.io `mlmf` stays at 0.2.0.** It was never released past that (0.1.0
  and 0.2.0 only), and nothing in this portfolio depended on it.
- **Why.** The root package depended on a tensor library (`candlelight`)
  through a git dependency that crates.io will not publish, so it could not be
  released; and most of what it did is no longer MLMF's job. CireSnave,
  2026-10-24, in `CLAUDE.md` section 0: *"If the code adds to MLMF's ability to
  read or write (but not interpret or guess about) machine learning model
  files, it should be in MLMF. Anything else should not."* The multimodal,
  distributed, quantization, caching and model-card code was already removed
  in 0.7.0.

## Where each module went

"Replaced" means a crate in this workspace now does the same file work, in the
backend-agnostic form (bytes in, structure out; no tensor library). "Partly"
says what is covered. "Removed" means no replacement here, with the reason.
Names below are the module file names the old package had.

| Old module | Fate |
|---|---|
| lib.rs | Replaced by the workspace. There is no umbrella crate; depend on the format crates you need. |
| error.rs | Replaced: each crate has its own error type. |
| loader.rs, universal_loader.rs, mmap_loader.rs | Replaced by a source crate (`mlmf-source-file`, `mlmf-source-hub`) plus a format crate (`mlmf-safetensors`, `mlmf-gguf`, `mlmf-pickle`). The old "load a model into framework tensors" step is the consumer's job (Fuel does it). |
| formats/gguf.rs | Replaced: `mlmf-gguf` and `mlmf-ggml`. GGUF values stay in their stored format; decoding them is conversion (ML-ConvertAnt). |
| formats/gguf_export.rs | Replaced: `mlmf-gguf` `encode_gguf`, which writes tensors already encoded in their file type. The old quantizing exporter is not carried over (encoding is ML-ConvertAnt's). |
| formats/safetensors_export.rs, saver.rs | Replaced: `mlmf-safetensors` `encode_safetensors`. |
| formats/onnx_export.rs | Partly: `mlmf-onnx` `encode_model` writes the container (tensors, metadata, opaque nodes). The old transformer-graph builder is removed: choosing operators and topology is not file work. |
| formats/onnx_import.rs | Removed. Reading an ONNX graph for evaluation is Fuel's. Reading initializers as a file operation is not built yet. |
| formats/pytorch_loader.rs | Replaced: `mlmf-pickle` `parse_pth` (it was a stub that always returned an error). |
| formats/pytorch_export.rs | Removed: it was a stub that always returned an error. No PyTorch writer exists. |
| formats/awq.rs, formats/awq_export.rs | Partly: `mlmf-awq` locates packed layers and parses the config. AWQ loading and saving are kept in scope by CireSnave but are not implemented anywhere (they were stubs). Unpacking and repacking is conversion. |
| config.rs, metadata.rs | Partly: `mlmf-meta` and the per-format metadata readers return declared values as `Option`. The old `ModelConfig` defaults are gone on purpose. |
| name_mapping.rs, smart_mapping.rs | Removed from this repository for now. Detecting how a file names its tensors is in scope (ROADMAP S-12, S-21, S-30) and is not built yet; the old code is at the tag. |
| lora.rs | Partly: reading adapter files is file work; merging, composing and quantizing adapters is conversion and is removed. |
| checkpoint.rs | Removed: training-checkpoint management is not model-file reading or writing. |
| conversion.rs | Removed: converting between formats is ML-ConvertAnt's. |
| validation.rs | Removed: it checked device and memory estimates, which are the consumer's (memory estimation belongs in Fuel; MLMF exposes shapes, dtypes and byte sizes). |
| progress.rs | Removed: progress callbacks belonged to the loaders that are gone. |
| mlmf.code-workspace | Removed: an editor file. |

The package's examples, its build script and the vendored copy of the ONNX
schema it compiled went with it. `mlmf-onnx` writes the wire format by hand and
credits `onnx.proto` in [NOTICE](NOTICE).

## If you used the old package

Depend on the specific crate for the format, for example
`mlmf-gguf = "0.7"` plus `mlmf-source-file = "0.7"`. Every crate hands you
`mlmf_core::TensorContainer`: descriptors (name, shape, encoding, byte range)
and the file's bytes, not framework tensors. If you need something this list
says is removed, open an issue describing the file operation you need; the
answer will be a file operation, or a pointer to the project that owns the rest.
