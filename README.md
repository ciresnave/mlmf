# MLMF - Machine Learning Model Files

[![License](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](https://github.com/CireSnave/mlmf#license)

**MLMF reads and writes machine learning model files.** It is a set of small
crates, one per file format, that turn bytes into structure and structure into
bytes. It does not interpret what a weight means, supply a value a file does
not declare, quantize, dequantize, or run anything. See
[CLAUDE.md](CLAUDE.md) for the rule and CireSnave's wording of it.

Since 0.7.12 this repository is a workspace of the crates below; the old
single `mlmf` package (a loader built on a tensor library) is retired.
[MIGRATION.md](MIGRATION.md) says where each of its modules went.

## Crates

All crates share one version number and are released together (0.7.12 at the
time of writing). The `mlmf-*` crates published at that version are the
supported ones; `mlmf` itself on crates.io stops at 0.2.0.

| Crate | What it does | Axis |
|---|---|---|
| `mlmf-core` | The seam: `TensorContainer`, `MetadataSource`, dtypes, shapes, descriptors, write-time requirement checks | - |
| `mlmf-safetensors` | Read safetensors headers and tensor directories; write a safetensors image | format |
| `mlmf-gguf` | Read GGUF metadata and tensor directories, raw and GGUF-embedded imatrix, tokenizer keys; write GGUF | format |
| `mlmf-ggml` | ggml type codes and block geometry | format |
| `mlmf-gguf-arch` | Name a GGUF file's architecture family from `general.architecture` or unambiguous tensor names; absent when the file does not say | format |
| `mlmf-tensor-names` | Translate tensor names between HuggingFace and GGUF spellings (dense Llama-style layout) from a declared table; a name with no entry is reported, never guessed; also reads a mapping file of extra name pairs (`translate_with`, table first) | format |
| `mlmf-oracle` | A checked `Oracle` seam for proposing tensor-name pairs offline, writing them as entries of the mapping file (the file type lives in `mlmf-tensor-names`); never on a load path, names compared byte-exactly | format |
| `mlmf-pickle` | Read PyTorch `.pth`/`.bin` checkpoints: zip directory and tensor index, nothing executed | format |
| `mlmf-onnx` | Write ONNX `ModelProto` files (nodes supplied as opaque bytes) | format |
| `mlmf-awq`, `mlmf-gptq` | Locate packed AWQ / GPTQ layers and parse their quantization configs | format |
| `mlmf-bnb` | Locate bitsandbytes 4-bit layers and read their raw pieces and `quant_state` | format |
| `mlmf-exl2` | Detect and refuse EXL2 | format |
| `mlmf-hf-layout` | HuggingFace shard index: where each tensor lives | format |
| `mlmf-hf-tokenizer` | Read `tokenizer.json` vocabulary and merges | format |
| `mlmf-meta` | Canonical metadata key vocabulary and declared-value extraction | format |
| `mlmf-source-file` | Open local files (memory-mapped by default) as byte sources | source |
| `mlmf-source-hub` | Fetch and cache from the HuggingFace Hub | source |
| `mlmf-conformance` | Cross-backend and differential tests (not published) | - |

Format-axis crates do no I/O: they take bytes and return structure (or the
reverse). Source-axis crates do I/O and know no formats. Any source composes
with any format.

## The rule

> *"MLMF's job is to read and write model files. MLMF is never intended to be an
> interpreter of the content of model files. That is for things like Fuel."*

Absent fields are `Option`, never a default (a format's *documented* default is
the one exception). Writers take data already encoded in its file dtype.
Encoding, decoding, quantization and conversion between formats belong to a
separate project (ML-ConvertAnt), not here.

## Building and testing

```bash
cargo build --workspace
cargo test --workspace
```

The tests that need real model files (a GGUF corpus, a PyTorch checkpoint, an
imatrix file) read them from a local path named by an environment variable
and print `SKIPPED` when the file is absent; a skip is not a pass. Each
crate's `tests/` directory says which variable.

## Documentation

- [MIGRATION.md](MIGRATION.md) - what happened to the old root package
- [GLOSSARY.md](GLOSSARY.md) - terms, types and format names
- [ROADMAP.md](ROADMAP.md) - what is done, open and out of scope
- [CHANGELOG.md](CHANGELOG.md)
- `docs/superpowers/specs/2026-08-14-backend-agnostic-mlmf-design.md` - the design

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md).

## Acknowledgements

Several tables and test fixtures are measured from, or quote, upstream
projects: the GGUF format and reference parser of
[ggml-org/ggml](https://github.com/ggml-org/ggml) and
[ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp), the ONNX protobuf
schema of [onnx/onnx](https://github.com/onnx/onnx), and the lineage of the
pickle reader in [huggingface/candle](https://github.com/huggingface/candle).
See [NOTICE](NOTICE) for the full credit, pinned sources and licence texts.

## License

This project is licensed under either of:

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT License ([LICENSE-MIT](LICENSE-MIT))

at your option.
