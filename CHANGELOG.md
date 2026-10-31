# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.11.0] - 2026-10-31

### Added
- `mlmf-meta`: canonical keys `BOS_TOKEN_TEXT` and `EOS_TOKEN_TEXT`, and the HuggingFace rows for the token ids (`config.json:bos_token_id` / `eos_token_id`), the token texts (`tokenizer_config.json:bos_token` / `eos_token`), the two `add_*_token` flags and the chat-template names array (`tokenizer_config.json:chat_template_names`). The text keys have no GGUF row (GGUF declares an id and indexes its token table), and there is no `tokenizer_config.json:*_token_id` row, asserted by a test. `SpecialTokens::extract` now reads a token's text by two paths in order, a direct declaration first and then the token table, and records which in `SpecialToken::text_source` (`TextSource::{Declared, Indexed}`). New `SpecialTokens::unusable` names keys the file declared with a shape the extractor does not read, such as `config.json:eos_token_id` as a list of ids (Llama-3.1 declares three).
- `mlmf-conformance` `hf_meta`: `HfLayout` read through `mlmf-meta`, so the table's spellings are checked against what the producer emits, including the `chat_template.<name>` join `TemplateSet` needs; plus a local-checkpoint scan that announces its skip.

### Changed
- **Breaking** (`mlmf-meta`): `SpecialToken::id` is now `Option<u64>`, and `SpecialToken` has a new `text_source` field. A token declared by text alone, as HuggingFace's `tokenizer_config.json` does, has no id, and inventing `0` would be a supplied default. The GGUF path is unchanged in behaviour: `id` is `Some(n)` and `text_source` is `Some(Indexed)` when the table resolved it. Pre-1.0, so the second number moves; the PM allocates the version.

## [0.10.18] - 2026-10-31

### Added
- `mlmf-hf-layout`: `HfLayout::from_files` builds ONE `MetadataSource` over a checkpoint's sidecars, with keys spelled `<filename>:<key>` (S-15 PR2). `config.json`, `generation_config.json` and `special_tokens_map.json` are flattened to dotted paths exactly as `ConfigJson` does; `tokenizer_config.json` gives `tokenizer_config.json:bos_token` (and `eos_token`, `unk_token`, `pad_token`) as the token's text whether the file used a bare string or an `AddedToken` object, with declared flags as `...:bos_token.lstrip` and so on, the `add_bos_token` / `add_eos_token` booleans, every other leaf flattened, and the chat template under PM ruling A: the string, or the list entry named `default`, at `tokenizer_config.json:chat_template`, the other names in `tokenizer_config.json:chat_template_names` with one `tokenizer_config.json:chat_template.<name>` key per body. The `chat_template.jinja` and `chat_template.json` sidecars sit under their own filename keys and are never merged with the `tokenizer_config.json` ones. An explicit `null` token is absent; a typed key with the wrong shape, an unrepresentable leaf, and a template name declared twice are `Declaration::Unreadable`. A file name `HfLayout` does not read is refused by name. No `tokenizer_config.json:*_token_id` key exists.

## [0.10.17] - 2026-10-31

### Added
- `mlmf-hf-layout`: `TokenizerConfig::parse` reads `tokenizer_config.json` (`bos_token`, `eos_token`, `unk_token`, `pad_token` as text in either of their shapes, `add_bos_token`, `add_eos_token`, `chat_template`), with the chat template kept in the shape the file used (`ChatTemplate::Single`, or `Named` for the `{name, template}` list). An explicit `null` token is "declared nothing", not malformed (Qwen3 and Qwen2.5 declare `"bos_token": null`). `parse_chat_template_jinja` and `parse_chat_template_json` read the two sidecar spellings; sources are returned separately and never merged. First slice of ROADMAP S-15; measured over 16 Hub repos and 3 local checkpoints.

## [0.10.16] - 2026-10-31

### Added
- `mlmf-gguf`: `GgufModelConfig::effective_head_count_kv()` applies the one default the GGUF specification states for `head_count_kv`: absent with a declared scalar `head_count` gives `Uniform(head_count)`. The declared `head_count_kv` field is untouched, so declared and derived stay distinguishable. A declared `head_count_kv` is returned as declared; the result is `None` for per-layer or absent `head_count`, for a `head_count_kv` that was declared but unusable, and for an incomplete metadata index. Warrant: `ggml-org/ggml` `docs/gguf.md` at `6af560d55df03ad92116e3c0a697779584477e85`, line 465, "If not present or if present and equal to `[llm].attention.head_count`, the model does not use GQA", read as key/value heads equal to `head_count` (an inference, ruled acceptable by the PM).

## [0.10.15] - 2026-10-31

### Added
- `mlmf-source-hub`: `HubSource::open_ranged(repo, revision, filename)` returns a `HubRangedFile`, a `mlmf_core::RangedSource` that reads a Hub file by HTTP `Range` request instead of downloading it, so `mlmf-safetensors`' `parse_header_ranged` and `mlmf-gguf`'s `read_header_region` move only the header over the network. The length comes from a one-byte `Content-Range` probe, which also proves the server honours `Range`; a server answering 200 is an error, never a silent whole-file download. Every reply is checked (206, the `Content-Range` asked for, exactly the bytes asked for); a range past the end is `Truncated` before any request; nothing is cached. Also `HubSource::with_endpoint` (mirror or test server).
- `mlmf-source-hub` tests: `crates/mlmf-source-hub/tests/support/range_server.rs`, a loopback HTTP server that serves a blob with `Range` support and counts the bytes it sends, so "reads only the header" is measured on the wire. Closes the Hub half of ROADMAP S-08.

## [0.10.14] - 2026-10-31

### Changed
- ROADMAP: S-16 closed as superseded by S-13 (`GgufMetadata::model_config()`), with the reason recorded: the GGUF spelling is `{arch}`-templated and `mlmf-meta`'s lookups match static strings, and a one-sided cross-format row has no consumer. Reopens with the HuggingFace spellings (S-15). Docs only.

## [0.10.13] - 2026-10-30

### Added
- `mlmf-gguf`: `GgufMetadata::model_config()` returns `GgufModelConfig`, the model-shape keys the file DECLARES (`architecture`, `context_length`, `embedding_length`, `block_count`, `feed_forward_length`, `head_count`, `head_count_kv`, both layer-norm epsilons, `rope_freq_base`, `rope_dimension_count`, `vocab_size`). Every field is `Option`; nothing is derived from another field (`head_count_kv` is `None` when undeclared, never copied from `head_count`); the `{arch}` prefix is the file's own `general.architecture`; counts the file declares per layer are kept as `GgufCount::PerLayer`; a key present with a wrong shape or unreadable to the parser is named in `malformed` rather than folded into `None`. The key names were read from the 30 local corpus files (S-13).

## [0.10.12] - 2026-10-30

### Added
- `mlmf-safetensors`: `parse_header_ranged` and `parse_tensors_ranged` read the length prefix and header through a `mlmf_core::RangedSource` (exactly two ranges, `0..8` and `8..8 + header_len`) and return `RangedTensors`, whose `tensor_bytes` reads one tensor's range on demand. `SafetensorsError::Source` (non-exhaustive enum) reports a failing source.
- `mlmf-gguf`: `read_header_region` reads from a `RangedSource` until the key-value block and tensor directory both parse (doubling from `INITIAL_READ`, bounded by a caller `limit` and the source's length, never re-reading) and returns a `HeaderRegion`; `HeaderRegion::tensors` gives `RangedGgufTensors`. `GgufError::Source` (non-exhaustive enum) reports a failing source.
- ROADMAP S-08: the header half is done, the trait CALL is answered, and the hub HTTP Range implementation stays open. No tensor data is read while parsing, and the tests count the bytes read to prove it.

## [0.10.11] - 2026-10-30

### Changed
- `mlmf-hf-layout`: package description now names all four modules (shard index, `config.json`, `generation_config.json`, `special_tokens_map.json`) instead of only the shard index (ROADMAP D-10).
- ROADMAP: D-01, D-02 and D-03 closed on a measurement (the README claim is gone; `ONNX_SUPPORT.md` and `GGUF_EXPORT_SUMMARY.md` no longer exist). D-06, D-08, D-12 and D-13 stay open: they live in historical plan documents kept verbatim.

## [0.10.10] - 2026-10-30

### Changed
- ROADMAP: the nine open `src/...` rows of the file-length table are closed as obsolete. The root crate is retired (S-37), `git ls-tree -r origin/main` finds no file under `src/`, and no tracked `.rs` file exceeds 1000 lines. Docs only.

## [0.10.9] - 2026-10-30

### Added
- `mlmf-mlx` (S-23): new format-axis crate, `crates/mlmf-mlx`. `locate_layers` finds MLX-packed linear layers (`U32` `weight` `[out, in*bits/32]`, `scales` and `biases` `[out, in/group_size]`, plus the Linear's own optional `bias`) and `MlxConfig` parses `config.json`'s `quantization` section (`bits`, `group_size`, `mode`, each `Option`, none defaulted). Shapes were read from the real safetensors header of `mlx-community/Qwen2.5-0.5B-Instruct-4bit`. No I/O and no dequantization, so no packing-order claim is made beyond the shapes.

## [0.10.8] - 2026-10-30

### Changed
- `mlmf-test-support`: now holds the two corpus tables `data/corpus-metadata.tsv` and `data/corpus-safetensors.tsv` (moved from `crates/mlmf-gguf/tests` and `crates/mlmf-safetensors/tests`) and exports them as `GGUF_CORPUS_METADATA_TSV` and `SAFETENSORS_CORPUS_TSV`. The two `include_str!("../../...")` reads in `mlmf-conformance` that reached outside their crate (SUP-004) now go through it; no test reads outside its own package. No shipped API changed.

## [0.10.7] - 2026-10-30

### Changed
- ROADMAP: the `unused-parameters` / `unused-return-values` row closed on a measurement: 0 / 0 from `coderipper check` over all 22 members, with a seeded positive control. S-34 (provenance stamp) closed as moot under S-33.

## [0.10.6] - 2026-10-30

### Changed
- `Cargo.lock`: 30 packages updated to the latest compatible versions (`cargo update`; no manifest requirement changed). `crates/mlmf-core/tests/transitive-deps.snapshot` re-blessed for the five `mlmf-core` transitive bumps (smallvec, syn, thiserror, thiserror-impl, unicode-ident). `fuel-formats` stays at 0.24.0 on purpose, see ROADMAP S-61.

## [0.10.5] - 2026-10-30

### Changed
- ROADMAP: S-37 closed as obsolete; the root crate it describes was retired in 0.7.12.

## [0.10.4] - 2026-10-30

### Added
- `crates/mlmf-gguf/src/split.rs`: pure helpers for llama.cpp's split (sharded) GGUF layout, no I/O. `parse_split_file_name` reads `<prefix>-NNNNN-of-MMMMM.gguf`; `shard_file_names` builds the plain file names of every shard and refuses a prefix that could leave the model directory; `SplitInfo::from_source` reads `split.no` (u16, zero-based), `split.count` (u16) and `split.tensors.count` (i32, the total across shards) and rejects a missing `split.no`, `split.no >= split.count`, a `split.count` of 0 (the half-written placeholder) and a negative total; `SplitInfo::matches_file_name` compares a shard's declared position with its name. Key names, types and the zero-based numbering were checked against llama.cpp `master` on 2026-10-30. All new types are `#[non_exhaustive]`; additive. Requested by lightbulb.
- ROADMAP: S-62.
## [0.10.3] - 2026-10-30

### Changed
- `crates/mlmf-conformance/tests/fuel_differential.rs`: the GGUF/GGML AD-1 differential now compares payload bytes, in a weak form: mlmf `tensor_bytes` against the file buffer sliced by fuel's range, because `fuel-formats` 0.24 exposes offsets and no GGUF payload reader. Measured 2026-10-30 over the 30-file corpus: 30 files agreed, 2562 tensors, 1156925952 bytes identical; slicing one byte short turns it red. ROADMAP S-39 done. Test only; no library change.


## [0.10.2] - 2026-10-30

### Changed
- `crates/mlmf-conformance/tests/fuel_differential_safetensors.rs`: the AD-1 differential now compares tensor payload bytes (mlmf `tensor_bytes` against the reference `TensorView::data()`), not only descriptors and ranges. Measured 2026-10-30 over the 9-file corpus: 664 tensors, 2924751236 bytes, all identical; a one-byte truncation of the reference side turns it red. The GGUF/GGML half (`crates/mlmf-conformance/tests/fuel_differential.rs`) still compares ranges only, so ROADMAP S-39 is partial. Test only; no library change.

## [0.10.1] - 2026-10-30

### Changed
- ROADMAP: S-38 closed as obsolete. `loader.rs` (`load_safetensors_mmap`) no longer exists at `origin/main` and the `For now, use regular loading` comment is in no `.rs` file; `crates/mlmf-conformance/tests/mmap_zero_copy.rs` asserts `tensor_bytes` returns a borrowed `Cow` through a live mmap for both GGUF and safetensors. Documentation only.
## [0.10.0] - 2026-10-30

### Added
- **Breaking (struct shape):** `crates/mlmf-meta/src/template.rs` `TemplateSet` and `crates/mlmf-meta/src/tokens.rs` `SpecialTokens` gain `unreadable: Vec<String>` (format-spelled keys the file declared but the parse could not decode) and `index_incomplete: bool` (the source's `index_complete()` was `false`, so an empty result is not a fact about the file). Both extractors now read through `MetadataSource::declaration`, so an undecodable key is no longer indistinguishable from an absent one. Neither struct is `#[non_exhaustive]`, so a struct-literal construction outside this workspace breaks; `extract` callers are unaffected. The PM allocates the number.

### Changed
- ROADMAP: S-17 marked done.

## [0.9.1] - 2026-10-30

### Added
- `crates/mlmf-gguf/src/metadata.rs` `GgufMetadata::parse_from(&dyn ByteSource, origin)` and `crates/mlmf-safetensors/src/header.rs` `parse_header_from(&dyn ByteSource)`: the `ByteSource` entry points, equal to `parse` / `parse_header` over `source.as_bytes()`. Additive. The PM allocates the number.

### Changed
- `crates/mlmf-conformance/tests/mmap_zero_copy.rs` now reaches the mapped file through the new constructors.
- ROADMAP: S-10 marked done.


## [0.9.0] - 2026-10-30

### Added
- **Breaking (struct shape):** `crates/mlmf-meta/src/template.rs` `TemplateSet::malformed_names` records the three malformed template-name shapes that were dropped with no record: an empty-name key (`tokenizer.chat_template.`), a non-string element of the names array (`tokenizer.chat_templates[<i>]`) and a names key that is not an array (`tokenizer.chat_templates`). The field is new on a struct without `#[non_exhaustive]`, so a struct-literal construction outside this workspace breaks; `TemplateSet::extract` callers are unaffected. The PM allocates the number.
- `crates/mlmf-gguf/src/lib.rs`: a "Metadata only" doc section pointing at `GgufMetadata::parse`.

### Changed
- ROADMAP: S-18 marked done; S-09 closed as satisfied by `GgufMetadata::parse` (`crates/mlmf-gguf/src/metadata.rs:87`).
## [0.8.0] - 2026-10-30

### Changed
- **Breaking (behaviour):** `crates/mlmf-awq/src/config.rs` `AwqConfig::parse_from_model_config` now returns `Err` when `quantization_config.quant_method` is present but not a string, matching `mlmf-gptq`; it previously recorded `quant_method` in `malformed` and returned `Ok`. Every other wrong-shaped field still lands in `malformed`. Callers that matched on `malformed` containing `"quant_method"` must handle the `Err`; none exist in this workspace.
- ROADMAP: S-43 marked done.

## [0.7.35] - 2026-10-30

### Changed
- `crates/mlmf-core/tests/purity.rs`: the lexer, tokenizer and `use`-tree expander move to `crates/mlmf-core/tests/purity_scan/mod.rs` (788 lines to 488 plus 310), clearing the last File-length row. The scanner is moved unchanged apart from four `pub` markers; the policy and the five tests, including `the_gate_can_fail`, stay in `purity.rs` and run through it. No behaviour change.
- ROADMAP: that File-length row marked done.

## [0.7.34] - 2026-10-30

### Changed
- `crates/mlmf-gguf/src/metadata.rs` and `crates/mlmf-gguf/src/value.rs`: each `tests` module moves to `crates/mlmf-gguf/src/metadata/tests.rs` and `crates/mlmf-gguf/src/value/tests.rs` (1147 to 520 plus 626, 734 to 350 plus 383 lines). The one test edit: the case table in `decode_value_round_trips_every_type` becomes a `scalar_cases()` helper, because Codacy counted the moved function at 84 lines against a limit of 50. The same 99 `mlmf-gguf` unit tests run under the same names.
- ROADMAP: those two File-length rows marked done.

## [0.7.33] - 2026-10-30

### Added
- `SECURITY.md`: "Supported versions" (CireSnave 2026-10-29): the latest release line receives security fixes; an older line gets a backport only when it is small, otherwise the advisory says it is not fixed; no support period is promised.

## [0.7.32] - 2026-10-30

### Changed
- `crates/mlmf-gguf/src/tensors.rs`: the `tests` module moves to `crates/mlmf-gguf/src/tensors/tests/records.rs` and `crates/mlmf-gguf/src/tensors/tests/container.rs` (1419 lines to 560 plus two test files of about 430 lines each, each under Codacy's file-length limit), clearing the file-length row for it. No code change; the same 99 `mlmf-gguf` unit tests run under the same names.
- ROADMAP: that File-length row marked done.

## [0.7.31] - 2026-10-29

### Changed
- `mlmf-test-support` (new, `publish = false`): holds `armed` / `armed_by`, the REQUIRE-gate arming predicate that lived in `mlmf-core`'s `tests/support` directory. `mlmf-core`, `mlmf-gguf`, `mlmf-safetensors`, `mlmf-pickle` and `mlmf-conformance` take it as a `[dev-dependencies]` edge instead of a `#[path = "../../mlmf-core/..."]` include, which reached outside the including crate's package and would not resolve when its tests ran from the published tarball (ROADMAP SUP-004: 8 of the 10 reads; the 2 `include_str!` reads left are in the unpublished `mlmf-conformance`).
- `crates/mlmf-core/tests/deps.rs`: a plain `[dev-dependencies]` edge is allowed only to a workspace member whose own manifest says `publish = false`. A dev-dependency on a publishable member or an external crate, a `[dev-dependencies.NAME]` table and a `[target...dev-dependencies]` table are still refused. Tests: a publishable member, `publish = true`, a registry list and an external crate each refused; the real workspace read at least one edge.

## [0.7.30] - 2026-10-29

### Security
- Dependabot alert 1 (critical, `zip` <= 9.0.4, fixed in 9.0.5): `Cargo.lock` now resolves `zip` 9.1.3. `zip` was never a direct dependency; it came in through `fuel-formats` and `fuel-ir`, which `mlmf-conformance` takes behind the off-by-default `fuel-differential` feature.

### Changed
- `mlmf-conformance`: `fuel-formats` is now a crates.io dependency (`0.24.0`) instead of a git pin at commit 6609d15, which also meets the rule that cross-repo dependencies are published versions. Pinned to the 0.24 line on purpose: `fuel-formats` 0.25 deleted its own GGUF parser and reads through `mlmf-gguf`, so a differential against it would compare mlmf with itself. 0.24.0 is the newest release with the independent parser and `zip` 9. `fuel_differential` (30 corpus files) and `fuel_differential_safetensors` (9 files) pass unchanged.
- ROADMAP: S-61 marked done.

## [0.7.29] - 2026-10-29

### Changed
- `mlmf-tensor-names`: `crates/mlmf-tensor-names/src/mapping_file.rs` is declared public API for the reachability check, as `mlmf-oracle`'s `file.rs` was before the move (0.7.28). Clears 13 new advisory findings; no code change.

## [0.7.28] - 2026-10-28

### Added
- `mlmf-tensor-names`: `translate_with(direction, name, &MappingFile)` answers from the declared table first and falls back to a mapping file only for a name the table lacks. A file entry that disagrees with a table row is a `LookupError::Conflict`, a file for the other direction is `WrongDirection`, and nothing loads a file implicitly. Names match byte-exactly (ROADMAP S-60). ROADMAP S-21 (last piece).

### Changed
- The mapping-file type (`MappingFile`, `Entry`, `Evidence`, `EvidenceKind`, `FileError`) now lives in `mlmf_tensor_names::mapping_file`; `mlmf-oracle` re-exports it, so `mlmf_oracle::MappingFile` and friends still resolve. `mlmf-oracle` no longer depends on `serde_json` directly, and `mlmf-tensor-names` gains `serde_json` (the first dependency of that crate).

## [0.7.27] - 2026-10-26

### Added
- `mlmf-oracle` (ROADMAP S-21): the tensor-name mapping file (`MappingFile`, `Entry`, `Evidence`; JSON, versioned, unique sources and targets, evidence per entry with model, prompt SHA-256 and date for an oracle entry) and the `Oracle` trait with `accept`, which checks every proposal against its request and drops a source or target claimed twice rather than picking one. Names compare byte-exactly, with no normalization and no model-fail-then-ask loop (ROADMAP S-60). No I/O, never on a load path.

### Changed
- ROADMAP: S-59 marked done.

## [0.7.26] - 2026-10-26

### Changed
- `docs/PATENT-SWEEP.md`: Part E, the sweep for the `mlmf-oracle` design (ideas 25-28) and the `mlmf-awq` read/write (ideas 29-30). Three paced searches, three patents read. One to watch: US11556508B1 (Cigna, name-mapping by a model with name standardizing and a user-updated dictionary) is not asserted clear against the planned oracle; ROADMAP S-60 and the oracle design page bar the two things that would bring the crate closer to it. Nothing built reads on it.

## [0.7.25] - 2026-10-25

### Added
- `mlmf-awq`: `read_layer` (a located layer's qweight/qzeros/scales/bias as stored bytes), `write_layer` (checks byte lengths against the declared geometry and returns the tensors for a container writer), and `AwqConfig::to_standalone_json` / `to_quantization_config_json` (absent fields are omitted, never defaulted). Bytes are moved, never unpacked or repacked (ROADMAP S-53). ROADMAP S-05.

## [0.7.24] - 2026-10-25

### Added
- `docs/superpowers/specs/2026-10-25-mlmf-oracle-design.md`: design for `mlmf-oracle` (ROADMAP S-21), the offline mapping-file emitter. Docs only; no code.

## [0.7.23] - 2026-10-25

### Changed
- `docs/PATENT-SWEEP.md`: Part D, the sweep for `mlmf-tensor-names` (ROADMAP S-30). Two ideas (name table, all-or-nothing refusal), three paced searches, two patents read (US12271819B2 clear; WO2018058426A1 clear on the claims read, two independent claims unread and said so). No patent found on point; nothing needs review.

## [0.7.22] - 2026-10-25

### Added
- `mlmf-tensor-names` (ROADMAP S-30): `translate(Direction, name)` and `translate_all` convert tensor names between HuggingFace and GGUF spellings for the dense Llama-style layout (embedding, norms, Q/K/V/O, gate/up/down, Qwen3 Q/K norms, output head; `.weight` and `.bias`) from a declared table. A name with no entry is `None`, and `translate_all` refuses naming every missing name rather than returning a partial map. Names only: the Q/K permutation llama.cpp applies when converting Llama weights is value conversion and is not done. Not ported from the removed `name_mapping.rs`: `Architecture`/`uses_rope` (interpretation), the invented `wte`/`h.N` target naming (a consumer's convention), and detection (see `mlmf-gguf-arch`).

## [0.7.21] - 2026-10-25

### Changed
- `docs/PATENT-SWEEP.md`: re-sweep for `mlmf-gguf-arch` (ROADMAP S-59). Two ideas added (declared-architecture classification that is never overridden by a guess; tensor-name inference only when the key is absent from a complete metadata view). Three searches, two patents read claim by claim (US20250362963A1, US5444824A): both clear, nothing needs review. Google Patents was throttled, so search was a general web search and the record says so.

## [0.7.20] - 2026-10-25

### Added
- `mlmf-gguf-arch` (ROADMAP S-12): `classify(&dyn MetadataSource, &[TensorDescriptor])` names a GGUF file's family (`Llama`, `Qwen2`, `Qwen3`, `Qwen3Moe`, `Phi`, `Phi3`, `Gemma`, `Gemma3`, `Glm4`, `Lfm2`, `SmolLm3`, `Gpt2`, `GptNeoX`) with a `Basis` saying whether `general.architecture` declared it or tensor names showed it. The answer is `None` when the file does not say: an unlisted, non-string or undecodable declaration is never replaced by a tensor-name guess, an absent key in a partial metadata view is not inferred from, and `blk.*` names alone (the layout of nearly every llama.cpp architecture) name no family. Only the distinctive GPT-2 and GPT-NeoX layouts are inferred, and a file showing both is `None`. Differs from fuel's `arch.rs`, which answers `Llama`/`Qwen3Moe` from `blk.*`. Mutating each guard (partial-view check, declared-value precedence, the `blk.*` arm, the both-layouts arm) turns a test red.

## [0.7.19] - 2026-10-25

### Changed
- `docs/PATENT-SWEEP.md` redone: every independent claim of the 16 patents in the first record re-read with the portfolio's `patent-claims.py` (full claim text, not a paraphrasing reader), the six unread patents read, and the code removed before 0.7.0 (distributed loading, quantization and calibration, model cache, model-card generator, multimodal fusion, memory estimate) added to scope with 12 more patents read. 28 patents read, none needing a decision; one close call (US12700225B2 against the removed multimodal code) recorded for awareness.

## [0.7.18] - 2026-10-25

### Added
- `docs/PATENT-SWEEP.md`: the patent sweep of the `mlmf-*` workspace at 1f6e961 (14 ideas, 16 queries, 9 patents read to their independent claims, none needing review; limits stated in the file: general web search rather than a patent database, claims read through a paraphrasing page reader, 6 patents seen and not read). Required before any publish by the portfolio patent-sweep process (board 286).

## [0.7.17] - 2026-10-25

### Added
- `mlmf-onnx` reads (ROADMAP S-58): `parse_model` reads an ONNX `ModelProto` and reports `ir_version`, producer strings, `model_version`, `doc_string`, `opset_import`s, `metadata_props`, the graph name, each initializer (`name`, `data_type`, `dims`, and its elements as a `TensorData`: `raw_data` as a byte range of the input, external-data key/values, or the typed field that holds them, which is not decoded), inputs and outputs, and each node as a byte range. A field the file does not write is `None`, an `opset_import` domain the file leaves out is `None` and not the empty string, and a dimension with neither value nor name is kept as unknown. Both packed and unpacked `dims` are read. `dtype_of_onnx` is the inverse of `onnx_data_type`. Checked against the reference `onnx` package's reading of three files: the writer's golden model, a model PyTorch's exporter wrote (from the `onnx` repository's test data), and a generated file carrying typed, external, anonymous, packed, negative-varint and above-2**32 cases; mutating the reader (dropping packed `dims`, a typed field, external data, a dimension, the varint width) turns the tests red.

## [0.7.16] - 2026-10-25

### Changed
- ROADMAP: S-54 (ONNX writer, PR #156), S-55 (`.pt2`, PR #158) and S-56 (StableHLO envelope, PR #159) marked done; new row S-57 records the cost of reading weights out of a StableHLO module (MLIR attribute decoding) and leaves it open until a consumer asks.

## [0.7.15] - 2026-10-25

### Added
- New crate `mlmf-pt2` (ROADMAP S-55): `parse_pt2` reads a `torch.export` `.pt2` archive (zip layout, `archive_format`/`archive_version`/`byteorder`, per-model weights and constants configs, the raw tensor bytes as file ranges; the exported graph is reported as an opaque byte range) and `encode_pt2` writes one from caller-supplied parts (graph JSON and sample-inputs blob opaque; archive root, `archive_version` and stream version required, never defaulted). A tensor whose torch dtype code has no `DType` is returned with `dtype: None` rather than refused. Checked against a real PyTorch 2.14.1 archive (an independent `zipfile`+`json` reader gives the expected rows) and in the other direction: the archive this crate writes loads with `torch.export.load` and runs to the same output as the original.

## [0.7.14] - 2026-10-25

### Added
- New crate `mlmf-stablehlo` (ROADMAP S-56): `parse_bytecode` reads the MLIR bytecode envelope that StableHLO portable artifacts use (magic, bytecode version, producer string such as `StableHLO_v1.18.0`, and every section's id, declared alignment and byte range) and `encode_bytecode` writes one from a version, a producer and opaque section bytes (`0xCB` padding, shortest-form varints, power-of-two alignment as the specification requires). Section contents are not decoded. Checked against a StableHLO artifact written by JAX 0.12.0.1 and a file MLIR itself wrote (an independent stdlib reader gives the same section tables; parse-then-encode reproduces both byte for byte); a file this crate writes under a different producer string parses with MLIR's own reader and prints the same module, and a damaged copy is refused by it. Reading the weights out of a module's constants would mean decoding MLIR's attribute encoding and is left as an open question.

## [0.7.12] - 2026-10-25

### Removed (breaking)
- The legacy root `mlmf` package: `src/` (26 files), `examples/`, the build script, the vendored ONNX schema, `.cargo/config.toml`, the `root-crate-tests` CI job, `documented_imports` and `model_config_literals` (guards whose only subject was the root crate), and three root documents that described only that code (`FIXES_SUMMARY.md`, `GGUF_EXPORT_SUMMARY.md`, `ONNX_SUPPORT.md`). The repository root is now a virtual workspace. The package could not be published (its `candlelight` dependency was a git dependency, which crates.io refuses and which breaks the portfolio's no-git-dependencies rule), crates.io `mlmf` stays at 0.2.0, and nothing in the portfolio depended on it. The code is at tag `v0.7.11`; `MIGRATION.md` maps each module to its replacement or says why it was removed.

### Changed
- `README.md` rewritten for the workspace; `CONTRIBUTING.md` uses `--workspace`; `ROADMAP.md` says the root was retired. Backticks were removed from citations of root source files in the root documents (the documented-paths guard rejects backticked paths that no longer exist); the text is otherwise unchanged.

## [0.7.11] - 2026-10-25

### Added
- New crate `mlmf-onnx` (ROADMAP S-54, ONNX writing): `encode_model` writes a `ModelProto` from caller-supplied initializers (`raw_data` already in its element type), `opset_import`s, producer strings, `metadata_props` and graph inputs/outputs, with graph nodes as opaque pre-encoded `NodeProto` bytes. `ir_version` and the opsets are required arguments, never defaulted; optional strings stay absent when `None`; a wrong byte length, a duplicate initializer name or a dtype with no ONNX type is refused. Byte-identical to what the reference `onnx` package serializes for the same model (golden file, regenerated by `tests/make_golden.py`, accepted by `onnx.checker`); the golden check caught a real mismatch in the first draft (`dims` is unpacked in `onnx.proto`). Choosing operators or building a graph from weights is not done here, and the old root-crate exporter that did it is not carried over.

## [0.7.10] - 2026-10-25

### Changed
- `.github/workflows/ci.yml`: the coderipper `portfolio.yml` caller is repinned to `ee9c3950c510fae136d316fed3bd9ecda343cede` (`# 0.9.24`), the commit whose workflow installs coderipper 0.9.24 (the PM's coderipper notice of 2026-10-24; the commit exists and is one ahead of the `v0.9.24` tag).
- Design spec section 10: amended. Architecture detection, tensor-name mapping and the oracle are no longer non-goals (CireSnave 2026-10-24: detecting how a file is stored is in scope; supplying a value the file does not declare is not).
- `ROADMAP.md`: S-47 to S-52 (fuel row 42 slices F1, E1, F2, E3, E2, E4) marked done.

## [0.7.9] - 2026-10-25

### Added
- New crate `mlmf-bnb` (fuel row 42 slice E4, reading only): `locate_layers` finds bitsandbytes 4-bit layers in an open container by their `quant_state.bitsandbytes__<type>` tensor; `read_layer` returns the packed weight, `absmax`, `quant_map`, `nested_absmax` and `nested_quant_map` as the file's bytes (each `None` if absent) and the parsed `quant_state` JSON; `parse_quant_state` returns every field as `Option`, with `quant_type` unjudged and unknown keys listed. No dequantization (that is the conversions project's) and no check that a blocksize divides a row. Checked against real bitsandbytes output (16 rows of a Qwen3-0.6B NF4 double-quant layer, Apache-2.0, vendored with its licence and provenance), whose pieces were also re-read from the real 576 MB checkpoint by HTTP range requests and found byte-identical.

## [0.7.8] - 2026-10-25

### Added
- New crate `mlmf-pickle` (fuel row 42 slice E2): `parse_pth` reads a `torch.save` checkpoint's zip directory (zip64 aware, stored entries) and the tensor index in `data.pkl` with a bounded pickle reader that records what the pickle would build and executes nothing. Returns each tensor's name, dtype, shape, strides, storage offset and storage entry, plus the storage's byte range in the file; a tensor whose storage class has no dtype is an error, and non-tensor entries are listed rather than dropped. Objects live in an arena (a memo `GET` is an index, not a clone), every length is bounded by the remaining bytes, and `REDUCE`/`NEWOBJ` are gated by `ALLOWED_REDUCE_TARGETS`, whose exact content a test pins (spec AD-3). A rewrite, not a port, of Fuel's reader (itself derived from candle's `pickle.rs`); it does not carry over that reader's `DICT`-as-empty-dict or clone-on-GET. Checked against a real 3.5 MB checkpoint: all 152 tensors match Fuel's `read_pth_tensor_info` and an independent Python reader, storage byte sums included.

## [0.7.7] - 2026-10-25

### Added
- `mlmf-gguf`: `read_tokenizer_data`, `GgufTokenizerData`, `TokenizerDataError` read the `tokenizer.ggml.*` keys as plain data (fuel row 42 slice E3). Every field is `Option`: an absent key is `None`, with no defaulted `pre`, `add_bos_token` or `add_eos_token` and no case folding of the model name; merges stay unsplit strings; a wrong-typed key, an out-of-range id or a non-UTF-8 token is an error, never `None`. `index_complete` travels with the data. Checked against an independent Python reader over the 30 v2/v3 files of the GGUF corpus (`tests/corpus-tokenizer.tsv`).

## [0.7.6] - 2026-10-25

### Added
- `mlmf-gguf`: `encode_gguf`, `encode_gguf_head`, `GgufTensorToWrite`, `GgufVersion`, `GgufEncodeError`, `GgufHead` (fuel row 42 slice F2). Pure, no I/O. Tensors arrive already encoded in their ggml type and are laid down untouched, dims as stored (innermost first, what this crate's readers report), each tensor padded to the alignment (`general.alignment`, else the spec's 32). Refuses a missing `general.architecture` (and `general.quantization_version` when any tensor is quantized) rather than writing a guess, an empty or mixed array, duplicate keys or tensor names, and a byte length that is not what the type and dims require. For the same inputs the v2 output is byte-identical to a port of fuel's `write_gguf`, checked in `mlmf-conformance`.

### Removed
- The `write_check_has_no_consumer` test in `mlmf-core` (deleted; its path is not cited here because the documented-paths guard rejects a deleted file), the deferral detector for `WriteCheck`, and the "Nothing consumes this yet" section of `write_check.rs`: `encode_gguf` now consumes `WriteCheck`, which is the event the detector's header says to delete both on. The module doc now says who consumes it.

## [0.7.5] - 2026-10-25

### Added
- `mlmf-gguf`: `parse_raw_imatrix`, `RawImatrix`, `RawImatrixEntry`, `RawImatrixError` read `llama-imatrix`'s raw positional layout (fuel row 42 slice E1). Pure, no I/O; every length is bounded against the remaining bytes before allocation. Values come back as stored: unlike fuel's reader this does not divide by `ncall`; `ncall`, the trailer's `last_call` and the dataset name are returned beside them. Checked against a real 1.3 MB file (`bartowski/Llama-3.2-1B-Instruct-GGUF`) with facts measured by an independent reader; the test announces `SKIPPED` when the file is absent.

## [0.7.4] - 2026-10-25

### Added
- `mlmf-safetensors`: `encode_safetensors`, `encode_header`, `TensorToWrite`, `EncodeError`, `name_of`. Lays tensors already encoded in their file dtype into a safetensors image (fuel row 42 slice F1, CireSnave 2026-10-25). Pure, no I/O: the caller writes the bytes. Tensors are ordered widest dtype first then by name so each starts aligned; length, duplicate-name, `__metadata__` and overflow errors refuse rather than write. Checked in `mlmf-conformance` against the upstream `safetensors` crate in both directions.

## [0.7.3] - 2026-10-25

### Changed
- `ROADMAP.md`: rows S-54..S-56 (ONNX writing, `.pt2`, StableHLO) from CireSnave's board 282 ruling. Docs only.

## [0.7.1] - 2026-10-25

### Changed
- `ROADMAP.md`: rows S-47..S-53 record CireSnave's ruling on fuel row 42 (slices E and F come to MLMF as file I/O; slice C and the i-quant decoders go to ml-convertant). Docs only.

## [0.7.0] - 2026-10-24

### Removed (breaking)
CireSnave, 2026-10-24: MLMF's code must add to its ability to read or write model files,
without interpreting or guessing about them. The rule is now in `CLAUDE.md`. Removed because
it does not:
- Multi-modal support: `multimodal`, `multimodal_loader`, `multimodal_processor` (cross-modal
  attention, fusion, `MultiModalModel::infer`), their exports and examples. Fixes #143.
- Distributed loading: `distributed`, `distributed_core`, `distributed_loader`, including
  `ShardingStrategy` and its `ModalitySpecific` variant, and the distributed examples.
- Post-training quantization: `quantization` (`QuantizationEngine`, calibration datasets,
  activation statistics) and its examples. The metadata types that DESCRIBE a file's
  quantization (`ModelQuantizationInfo`, `CalibrationMethod`) stay.
- Model caching: `cache`, `cached_loader` (`ModelCache`, `CachedModelLoader`).
- Model cards: `model_card` (`ModelCardGenerator` and its types), and its example.
- The Feature 6 and Feature 8 summary docs.

The removed code is at `ciresnave/mlmf` commit `bfc84ac`, for any project that wants it.
`impl Display for Architecture`, which lived in `model_card`, moved to `name_mapping`.

### Changed
- All inter-crate `version` requirements are 0.7.0.
- `ROADMAP.md`: the integrate-or-retire section is replaced by CireSnave's ruling that AWQ
  loading and saving stay in MLMF, with an inventory of what exists.

## [0.6.22] - 2026-10-24

### Documentation
- `ROADMAP.md`: spec-to-roadmap audit. 46 rows for work the specs, plans and summary docs
  describe and the roadmap did not list (25 need a CireSnave call), 13 rows for docs that
  contradict the code, and a list of items verified done. Docs only; nothing started.

## [0.6.21] - 2026-10-24

### Changed
- `ONNXLoader::infer_model_config` (private) drops its never-read `name_mapper`
  parameter. The `modality` parameters of `load_distributed_modality` and
  `load_single_modality`, `deploy_shard_to_node`'s `model_path` and `select_node`'s
  `request` are underscore-prefixed as planned call sites. No behaviour change.
  Resolves 5 of the remaining 6 `unused-parameters` findings; the sixth,
  `process_modality`'s `model`, is a live no-op filed as an issue.

## [0.6.20] - 2026-10-24

### Fixed
- `save_as_onnx` ignored its `export_options`: it built fresh writer options and never read
  the argument. `opset_version` and `custom_metadata` now reach the writer;
  `optimize_graph` and `preserve_metadata`, which the writer cannot honour, are refused
  with an error instead of accepted and dropped. Behaviour change: `convert` to ONNX with
  `preserve_metadata: true` now returns that error (the ONNX writer itself still refuses
  to write a graph, so no export that succeeded before fails now).

## [0.6.19] - 2026-10-24

### Fixed
- `save_as_gguf` wrote the model's tensors as an empty map, ignoring its `model`,
  and returned `Ok` with a tensor-less file. It now writes `model.raw_tensors`, and
  refuses with an error when there are none. Behaviour change: a call that used to
  return `Ok` on an empty model now returns `Err`.

## [0.6.18] - 2026-10-24

### Changed
- `distributed_loader.rs` (removed in 0.7.0): the nine parameters of the private `todo!()` stubs
  (`deploy_model`, `scale_cluster`, `NodeManager::new`, `ShardManager::new` and
  `migrate_shard`, `LoadBalancer::new`, `HealthMonitor::new`) are underscore-prefixed.
  Resolves 9 of the root crate's 17 `unused-parameters` findings; no behaviour change,
  not breaking. The other 8 are triaged in `ROADMAP.md`.

## [0.6.17] - 2026-10-24

### Added
- `crates/mlmf-exl2`, `mlmf-core`, `mlmf-gguf` and `mlmf-safetensors` each gain a
  `.coderipper.toml` declaring the file holding their flagged `reachability`
  items as published API (9 findings, verified clean with CodeRipper 0.9.14).
  `ROADMAP.md` now records the root crate's 17 `unused-parameters` and 1
  `unused-return-values` findings. Config and docs only; not breaking.

## [0.6.16] - 2026-10-24

### Fixed
- `.coderipper.toml`: the `mlmf-awq`, `mlmf-gptq` and `mlmf-hf-tokenizer` public-API
  declarations now live in each crate's own `.coderipper.toml`, which is the file a
  per-member check reads; 0.6.14's root-file entries had no effect. Repinned the
  `portfolio.yml` caller to d14a22b (CodeRipper 0.9.14). Config and CI only; not breaking.

## [0.6.15] - 2026-10-23

### Added
- `ROADMAP.md`: recorded that `mlmf-awq`, `mlmf-gptq` and `mlmf-hf-tokenizer`
  are each independently published to crates.io with real downloads but
  have no in-repo consumer (root `mlmf`'s `awq` feature is a stub; its
  `tokenizers` feature wraps the external crate). Facts and two options
  (integrate vs. retire) only, no decision — flagged to CireSnave's board
  per the PM. Documentation only; not breaking.

## [0.6.14] - 2026-10-23

### Added
- `.coderipper.toml`: declared `mlmf-awq`, `mlmf-gptq` and `mlmf-hf-tokenizer`'s
  public API under `[reachability] public_api` (7 files, 64 of the 73
  `reachability` findings from the per-member portfolio sweep). All three
  crates are independently published to crates.io; their entry points
  (`parse_standalone`/`parse_from_model_config`/`locate_layers`/`parse`) are
  exercised only by each crate's own unit tests and have no in-repo
  consumer yet (root `mlmf`'s `awq` feature is still a stub; its
  `tokenizers` feature wraps the external crate, not `mlmf-hf-tokenizer`).
  Configuration only; no code change, not breaking.

## [0.6.13] - 2026-10-23

### Fixed
- `ROADMAP.md`: a peer notice had reported 17 `unused-parameters` findings;
  recounting raw finding lines directly from the sweep job log on
  `origin/main` found 0 for both `unused-parameters` and
  `unused-return-values`. The PM traced the "17" to an earlier,
  different-method baseline (root-package-only, pre-workspace-split) and
  confirmed 0/0 for the current per-member sweep; removed as a backlog item.
  `reachability`'s count (73) was re-verified and is unchanged.

## [0.6.12] - 2026-10-23

### Added
- `ROADMAP.md`: a file-length row per Codacy finding on PR #132 (15
  files, biggest non-comment-LOC first, none started) — pulled from
  the check run's own annotations rather than a peer notice that had
  undercounted at 8.

## [0.6.11] - 2026-10-23

### Fixed
- `lint-suppression` baseline rows (ROADMAP.md): 12 of 14 `#[allow(...)]`
  attributes across src and tests now carry a `reason = "..."`; the other
  2 (src/validation.rs's `unreachable_patterns` allows) had the
  attribute removed outright after `cargo check` confirmed, with the
  attribute removed, that neither produces a warning — the suppression
  was unneeded, not just unexplained.
- `licence` baseline row (ROADMAP.md), partial: every git-tracked `.rs`
  file (165) now carries `// SPDX-License-Identifier: MIT OR Apache-2.0`
  as its first line. `LICENSE-MIT`'s copyright-holder wording is
  unchanged, pending board 263.

## [0.6.10] - 2026-10-23

### Added
- `ROADMAP.md`: one row per failing rule from the first portfolio-wide
  CodeRipper sweep (PR #130) on `origin/main`, plus a note on the sweep's
  own `protoc`-related infrastructure defect (reported to the PM; fix
  belongs in `ciresnave/coderipper`, not here).

## [0.6.9] - 2026-10-23

### Added
- `.github/workflows/ci.yml`: an advisory-only, non-blocking `portfolio` job
  that runs CodeRipper's shared portfolio check (CireSnave, 2026-10-23:
  "Roll out advisory-only, non-blocking CodeRipper checks everywhere.").
  Not a required status check, and does not silence any existing rule.

## [0.6.8] - 2026-10-23

### Fixed
- `.github/workflows/ci.yml`'s `cargo metadata --locked` step moved to run
  first in `core-gates`, before any unlocked cargo invocation. It used to
  run fourth (after `cargo fmt`, two `cargo test -p mlmf-core` steps, and
  `scripts/check-deps.sh`), all of which resolve without `--locked` and so
  were free to silently rewrite the in-runner `Cargo.lock` before the
  dedicated check ever saw it — which is exactly how 0.6.7's own
  manifest/lockfile drift (fixed in #128) shipped on `main` with every
  `core-gates` leg green. Measured directly: restoring the committed
  drifted pair and running `cargo metadata --locked` first fails (exit
  101) as expected; running an unlocked `cargo test -p mlmf-core` first
  silently flips all 14 lockfile version lines to match the manifest, and
  the same `--locked` check then passes.
- Added a final `git diff --exit-code Cargo.lock` step at the end of
  `core-gates`, after every other cargo invocation in that job, as
  defense in depth: the moved check stops an existing drift from being
  masked, but nothing before stopped a later unlocked step in the same
  run from introducing a new one after the check passed.

## [0.6.7] - 2026-10-23

### Fixed
- `Cargo.lock` had drifted from `Cargo.toml`: every `mlmf-*` crate's
  recorded lock version still read `0.6.5` after the workspace version was
  bumped to `0.6.6`. `cargo update -p mlmf` resyncs it to this release's
  version.

## [0.6.6] - 2026-10-21

### Added
- `NOTICE` at the workspace root (also shipped inside `mlmf-ggml` and
  `mlmf-gguf`'s packages) crediting `ggml-org/llama.cpp` and `ggml-org/ggml`
  (MIT) for the type-geometry and required-key/version-support tables those
  two crates measure from and quote, with the upstream licence text copied
  from their LICENSE files and a README Acknowledgements section pointing
  to it.
- `crates/mlmf-ggml/src/types.rs` and `crates/mlmf-gguf/src/header.rs` now pin
  the exact upstream commit (`9d57ce456c94d241dde672b2db9cf18879766568`) their
  comments cite, instead of a file:line reference alone; `header.rs` also
  now names which parts of its header checks correspond to the GGUF spec's
  `## Version History` and which (the byte-swap heuristic) have no spec
  section at all. Not breaking: doc comments only, no behaviour change.

## [0.6.5] - 2026-10-21

### Security
- `rustls-webpki` 0.103.15 -> 0.103.16 (RUSTSEC-2026-0382: certificate
  verification may accept distrusted intermediates).
- `webpki-roots` 1.0.9 -> 1.0.10 (RUSTSEC-2026-0381: bundled roots trusted
  distrusted emSign intermediates).
- `rustls` 0.23.45 -> 0.23.46 (RUSTSEC-2026-0346, -0347, -0348, -0349).
  Dependency-only; no code change, not breaking.

## [0.6.4] - 2026-10-19

### Added
- `.coderipper.toml` declares the published API under `[reachability] public_api` (21 files). Configuration only; no code change, not breaking.

## [0.6.3] - 2026-10-18

### Added
- `GLOSSARY.md`: 53 terms the crates and docs use, each with a source pointer; linked from the README. Documentation only; not breaking.

## [0.6.2] - 2026-10-16

### Security
- CI hardening (Q14): GitHub Actions in `.github/workflows/ci.yml` are pinned to
  full commit SHAs, the workflow declares least-privilege `permissions: contents: read`,
  and `.github/CODEOWNERS` names a default owner. No code change; not breaking.

## [0.6.1] - 2026-10-08

### Fixed
- `TensorNameMapper` now recognizes qwen3's QK-Norm components -- GGUF
  `attn_q_norm.weight`/`attn_k_norm.weight` and HF/SafeTensors
  `self_attn.q_norm.*`/`self_attn.k_norm.*` (#96). Previously these hit the
  unrecognized-component refusal added in #97/#98; they are now ordinary
  recognized renames, consistent with the existing `q_proj`/`k_proj`/`v_proj`
  entries. Not a breaking change -- purely additive recognition.

## [0.6.0] - 2026-10-08

### Changed
- **Breaking:** `HFConfig`'s `vocab_size`, `hidden_size`, `num_attention_heads`
  and `num_hidden_layers` are now `Option<usize>` instead of `usize` (#118).
  An absent `vocab_size` previously decoded to `0`, indistinguishable from a
  file that declared `vocab_size: 0`; the other three were non-`Option`, so
  one absent key failed deserialization of the whole `config.json`.
  `HFConfig::to_model_config` now reports a missing field by name, and
  distinguishes "not declared" from "declared 0" with different error
  messages.

## [0.2.1] - 2024-11-12

### Added
- Added CHANGELOG.md to track version changes
- Added CONTRIBUTING.md with development guidelines
- Added comprehensive documentation for quantization API

### Fixed
- Updated README examples to use candlelight imports
- Fixed version specification in README examples
- Suppressed missing documentation warnings for cleaner builds

## [0.2.0] - 2024-11-12

### Added
- **Quantization API**: Complete post-training quantization (PTQ) support
  - Multiple quantization types: Int8, Int4, Mixed, Dynamic, Static
  - Calibration methods: minmax, percentile, entropy, KL divergence
  - Advanced features: block-wise quantization, symmetric quantization
  - Progress reporting and comprehensive configuration options
  - `advanced_quantization.rs` example (removed in 0.7.0) demonstrating the API
- **Candlelight Integration**: Migrated from direct candle dependencies to unified candlelight wrapper
  - Simplified dependency management
  - Consistent API across all Candle ecosystem crates
  - Better version compatibility

### Changed
- **BREAKING**: All `candle_core` imports must be changed to `candlelight`
- **BREAKING**: `LoadOptions` now includes `preserve_quantization` field
- **BREAKING**: `ModelConfig` now requires `raw_config` field
- Updated all examples to use candlelight imports
- Fixed all compilation errors after candlelight migration

### Fixed
- Resolved 51 compilation errors in quantization module
- Fixed doctest compilation issues across multiple files
- Corrected tensor name mapping for quantized models
- Fixed device validation and dtype checking

### Documentation
- Added comprehensive quantization API documentation
- Updated README with candlelight usage examples
- Fixed all doctest examples to use new API

## [0.1.0] - 2024-11-01

### Added
- Initial release of MLMF (Machine Learning Model Files)
- Support for multiple model formats: SafeTensors, GGUF, ONNX, PyTorch, AWQ
- Architecture detection for LLaMA, GPT-2, GPT-NeoX model families
- Dynamic tensor name mapping between formats
- Memory-efficient loading with memory mapping support
- Progress reporting system
- Comprehensive error handling and validation
- Model conversion capabilities
- Multi-modal support framework
