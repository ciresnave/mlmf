# Glossary

Terms this repository uses in its code, documentation and design specs, with
where each is defined. Entries are alphabetical. Paths are relative to the
repository root; `§` numbers refer to
`docs/superpowers/specs/2026-08-14-backend-agnostic-mlmf-design.md` (the
"design spec") unless another document is named.

**Absent field (`Option<T>` policy)** -- A field a supported format may not
supply is `Option<T>`, where `None` means only "MLMF had nothing to read".
Whether the format has no slot for it or a writer left it empty is not
distinguished. Source: `CLAUDE.md` §2.

**Alignment** -- The byte boundary a tensor's data starts on. MLMF gives no
blanket guarantee: GGUF pads tensor data to `general.alignment` (default 32),
safetensors guarantees nothing. Source: `crates/mlmf-core/src/align.rs`.

**Architecture** -- The legacy root crate's enum of model families detected
from tensor names: `LLaMA`, `GPT2`, `GPTNeoX` and `Unknown`. Source:
src/name_mapping.rs.

**AWQ** -- A quantization scheme for safetensors checkpoints. `mlmf-awq`
locates AWQ-packed linear layers and parses AWQ's quantization config; its
`qweight` packing axis is transposed relative to GPTQ's. The legacy root
crate does not implement AWQ loading or saving (it returns an error).
Source: `crates/mlmf-awq/src/lib.rs`, `README.md`.

**Block (quantization block)** -- A fixed group of elements stored together in
a quantized tensor, e.g. 32 elements in 18 bytes for `Q4_0`. See `BlockSpec`.
Source: `crates/mlmf-core/src/encoding.rs`.

**BlockSpec** -- The geometry of one quantization block family member:
family (e.g. `"ggml"`), the file's own declared type id, elements per block,
bytes per block and required alignment. New schemes are added as data (another
`BlockSpec` value), not as a new `Encoding` variant. Source:
`crates/mlmf-core/src/encoding.rs`.

**ByteSource** -- Trait for a contiguous region of bytes obtained from
somewhere (memory map, owned buffer, downloaded file), exposed as `&[u8]`.
Implemented by source crates; `mlmf-core` never implements it. Source:
`crates/mlmf-core/src/traits.rs`.

**CalibrationMethod** -- How a quantized model was calibrated, as recorded in
its metadata: `MinMax`,
`Percentile`, `Entropy` or `KLDivergence`. Source: src/metadata.rs.

**Charter** -- The project's founding rule: "MLMF's job is to read and write
model files. MLMF is never intended to be an interpreter of the content of
model files. That is for things like Fuel." Source: `CLAUDE.md`, design spec.

**CodeStatus** -- What a build knows about a ggml type code: `Known`,
`Retired` (ggml removed it, older files still carry it) or `Unassigned`
(never assigned, or newer than this build). Source:
`crates/mlmf-ggml/src/types.rs`.

**Conformance tests** -- Tests of the `mlmf-core` seam run against every
backend, held in the never-published `mlmf-conformance` crate because they
need two backends in one binary. Source:
`crates/mlmf-conformance/src/lib.rs`.

**ConversionFormat / ConversionOptions** -- The legacy root crate's target
format and settings for `convert_model`, which converts a model file between
formats. Source: src/conversion.rs.

**Declaration** -- What a `MetadataSource` knows about one key: `Absent`,
`Unreadable` (declared but could not be decoded) or `Declared`. It lets a
caller tell "the file says nothing" from "the file says something
unparseable". Source: `crates/mlmf-core/src/report.rs`.

**DeclaredType** -- A tensor's type exactly as the file declares it, before
resolution: either a numeric `Code` in a family's code space (GGUF) or a
`Name` (safetensors, e.g. `"BF16"`). Source:
`crates/mlmf-core/src/report.rs`.

**Encoding** -- How a tensor's elements are laid out in bytes: dense scalars
or block-quantized data. It also computes the byte size for an element
count. Source: `crates/mlmf-core/src/encoding.rs`.

**EXL2** -- exllamav2's quantization format. `mlmf-exl2` only detects it and
reports `Unsupported`, because its tensor layout has not been verified.
Source: `crates/mlmf-exl2/src/lib.rs`.

**Fence (§6 fence)** -- "MLMF may supply a format's documented default. MLMF
may never supply a model's value." Source: `CLAUDE.md` §1,
`crates/mlmf-core/src/write_check.rs`.

**Format axis / source axis** -- The two orthogonal kinds of crate in the
workspace. Format crates turn bytes into structure and do no I/O; source
crates only obtain bytes and know no formats, so any source composes with any
format. Source: `crates/mlmf-source-file/src/lib.rs`, design spec §3.1.

**Fuel / Lightbulb** -- Downstream consumer projects named throughout the
design documents. Interpreting what a model file's content means is Fuel's job
under the charter. Source: `docs/fuel-migration-inventory.md`, design spec.

**GGML / ggml type code** -- ggml owns a closed space of numeric type codes
for tensor element and quantization types; `mlmf-ggml` knows each code's block
geometry and nothing else. Source: `crates/mlmf-ggml/src/lib.rs`.

**GgmlType** -- Enum of the live ggml type codes. `GgmlType::from_code`
resolves a code into geometry; `GgmlType::status` reports why a code did not
resolve. Source: `crates/mlmf-ggml/src/types.rs`.

**GGUF** -- A model file format with a header (magic, version, counts), a
key-value metadata block and a tensor directory. `mlmf-gguf` reads it in
stages. Source: `crates/mlmf-gguf/src/lib.rs`.

**GgufMetadata / GgufTensors** -- `mlmf-gguf`'s parsed key-value block
(indexed at open, values decoded on request) and its tensor directory
(implements `TensorContainer`, offsets rebased once). Source:
`crates/mlmf-gguf/src/metadata.rs`, `crates/mlmf-gguf/src/tensors.rs`.

**GPTQ** -- A quantization scheme for safetensors checkpoints, stored across
cooperating tensors (`qweight`, `qzeros`, `scales`, optional `g_idx`).
`mlmf-gptq` locates its packed layers and parses `quantize_config.json`.
Source: `crates/mlmf-gptq/src/lib.rs`.

**GptqGroupSize** -- Group size in a GPTQ config: `PerGroup(n)` or
`NoGrouping` (AutoGPTQ's `-1`, one group spanning the whole input dimension).
Source: `crates/mlmf-gptq/src/config.rs`.

**Group count** -- In an AWQ packed layer, `in_features / group_size`.
Source: `crates/mlmf-awq/src/layer.rs`.

**HFConfig / ModelConfig** -- The legacy root crate's parse of a HuggingFace
`config.json` (with field aliases) and the model configuration derived from
it. Source: src/config.rs.

**HuggingFace layout** -- The directory convention of a HuggingFace
checkpoint: shard index plus JSON sidecars. `mlmf-hf-layout` reads
`model.safetensors.index.json`, `generation_config.json`,
`special_tokens_map.json` and `config.json` from supplied bytes. Source:
`crates/mlmf-hf-layout/src/lib.rs`.

**Hub / HubSource** -- The HuggingFace Hub fetch-and-cache source crate;
`HubSource` fetches one named file at a time into a local cache. Source:
`crates/mlmf-source-hub/src/lib.rs`.

**imatrix** -- Importance matrix produced by `llama-imatrix`. `mlmf-gguf`
reads only the GGUF-embedded layout, pairing `<name>.in_sum2` with
`<name>.counts` tensors, and never decodes the statistics. Source:
`crates/mlmf-gguf/src/imatrix.rs`.

**LoadedModel / LoadOptions** -- The legacy root crate's result of loading a
model (var builder, config, name mapper, raw tensors, metadata) and the
options controlling it (device, dtype, mmap, progress). Source:
src/loader.rs.

**LoRA** -- Low-Rank Adaptation: fine-tuning by learning low-rank updates to
weight matrices. src/lora.rs detects, loads, merges and saves LoRA
adapters. Source: src/lora.rs.

**MetaValue** -- The seam's metadata value type, whose variant reports how
the format declared the value (GGUF's thirteen value types plus `Bytes`), not
what it means. Source: `crates/mlmf-core/src/meta.rs`.

**MetadataSource** -- Trait for something that declares typed metadata under
string keys; GGUF's in-file metadata and HuggingFace's JSON sidecars both
implement it. Source: `crates/mlmf-core/src/traits.rs`.

**ONNX** -- A model file format. The legacy root crate loads tensors from it
and can export to it; its `ModelConfig` is partly guessed from tensor shapes.
Source: `README.md`, src/formats/onnx_import.rs,
src/formats/onnx_export.rs.

**Packed linear layer** -- A linear layer stored in a quantized, bit-packed
form, located by `mlmf-gptq` or `mlmf-awq`; carries tensor names, not bytes.
Source: `crates/mlmf-gptq/src/layer.rs`, `crates/mlmf-awq/src/layer.rs`.

**RangedSource** -- Trait for a source that can serve arbitrary byte ranges
without materializing the whole artifact. Source:
`crates/mlmf-core/src/traits.rs`.

**Rebasing** -- Converting a format's tensor offsets to absolute ranges into
the opened slice, done once at parse time. GGUF offsets are relative to the
data region; safetensors' to the end of its JSON header. Source:
`crates/mlmf-core/src/tensor.rs`.

**Report** -- The account, returned with every parse, of what the parse did
not understand (`Unrecognized` entries). Fatal unknowns are `ErrorKind`
variants instead. Source: `crates/mlmf-core/src/report.rs`.

**Requirement / Resolution / WriteCheck** -- The write-time mechanism that
sorts each key a target format requires into `Declared`, `Supplied` (with a
citation), `Missing` or `Unreadable`, and refuses when no citable default
exists. Source: `crates/mlmf-core/src/write_check.rs`.

**Revision** -- Which commit of a Hub repo to fetch: `Ref` (branch or tag,
resolved on each fetch) or `Pinned` (a commit SHA taken as-is). Source:
`crates/mlmf-source-hub/src/lib.rs`.

**Safetensors** -- A model file format: an 8-byte little-endian length
prefix, a JSON header of that length, then tensor data. `mlmf-safetensors`
reads headers and tensor directories without I/O. Source:
`crates/mlmf-safetensors/src/lib.rs`.

**Seam** -- The `mlmf-core` traits (`ByteSource`, `TensorContainer`,
`MetadataSource`) that format and source crates implement and consumers use,
so neither side knows the other. Source: `crates/mlmf-core/src/traits.rs`.

**Shape** -- A tensor's dimensions in declared order. Source:
`crates/mlmf-core/src/shape.rs`.

**ShardIndex** -- A parsed `model.safetensors.index.json`, mapping each tensor
name to the shard file that holds it. Source:
`crates/mlmf-hf-layout/src/shards.rs`.

**SmartTensorNameMapper** -- The legacy root crate's tensor name mapper with
fallbacks, used by `LoadedModel`. Source: src/smart_mapping.rs.

**Stage** -- The part of a parse that failed. GGUF has `Header`, `Metadata`
and `TensorDirectory`; safetensors has `Header` and `TensorDirectory`. Only
the failing stage fails, so metadata stays readable when tensors are not.
Source: `crates/mlmf-gguf/src/error.rs`,
`crates/mlmf-safetensors/src/error.rs`.

**TensorContainer** -- Trait for something that declares named tensors and
hands out their bytes through the fallible `tensor_bytes`. There is no tensor
type, device or backend trait. Source: `crates/mlmf-core/src/traits.rs`.

**TensorDescriptor** -- One tensor as the file declares it: name, `Shape`,
`Encoding` and the byte range of its data. A range the file cannot honour is
still recorded; `tensor_bytes` is where it fails. Source:
`crates/mlmf-core/src/tensor.rs`.

**TensorNameMapper** -- Maps tensor names from HuggingFace format to a target
format, with `Architecture` detected from tensor names. Source:
src/name_mapping.rs.

**TokenizerJson** -- `mlmf-hf-tokenizer`'s parse of `tokenizer.json`: the
declared vocabulary, BPE merge rules and added tokens. Only model type `"BPE"`
is read. Source: `crates/mlmf-hf-tokenizer/src/lib.rs`.
