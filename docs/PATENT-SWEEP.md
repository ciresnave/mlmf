# Patent sweep: mlmf 0.7.17 @ 1f6e961

Date: 2026-10-25; Run by: mlmf lane, Claude Sonnet 5.5; Method: AI-assisted search and read; NOT legal advice; NOT a freedom-to-operate opinion.

Process: `C:\Projects\PATENT-SWEEP-PROCESS.md` (CireSnave, board 286 item 1). Covers the `mlmf-*` workspace at `origin/main` `1f6e961` (the 0.7.15 crates.io release plus the 0.7.16 and 0.7.17 changes: ONNX reading, ROADMAP). Version line: **0.7.x**. The previous releases (0.2.0 on crates.io for the root `mlmf` package, tag `v0.7.11` for its last state) contained a larger code base (distributed loading, quantization, caching, model cards, multimodal); most of it was removed in 0.7.0 and the root crate in 0.7.12. This sweep is of what is released and maintained now; it does **not** sweep that removed code.

## Ideas swept

What MLMF does that might be new or unusual. It reads and writes model files; it does not run, convert, or interpret them.

1. Parse a safetensors file's header and report each tensor's name, dtype, shape and byte range, without reading tensor data (`mlmf-safetensors`).
2. Index a GGUF file's key-value block at open without decoding values, bounding every allocation against the bytes that remain, and report a tensor directory with byte ranges (`mlmf-gguf`).
3. Write safetensors and GGUF files from tensors the caller already encoded in the file's dtype, computing the header or directory and offsets (`encode_safetensors`, `encode_gguf`).
4. A table of ggml quantization types (block size, bytes per block, alignment) measured from a reference implementation, used to compute a tensor's byte size without decoding it (`mlmf-ggml`).
5. Each format crate recognises its own file from its leading bytes and version and refuses anything else. (Architecture detection from tensor names and tensor-name mapping are in scope but **planned, not shipped**: ROADMAP S-12 and S-30. They are swept here only as a stated intent; sweep again when built.)
6. A format-agnostic seam: one trait for "a container of named tensors with byte ranges" and one for "declared key/value metadata", implemented by each format crate and composable with any byte source (`mlmf-core`).
7. Read a `torch.save` pickle checkpoint without running it: a small pickle interpreter that executes only an allow-list of four `REDUCE` targets, treats everything else as an error, and reports tensor storages as file ranges (`mlmf-pickle`).
8. Read bitsandbytes 4-bit (NF4/FP4) layers, including double-quantized state, as bytes plus the declared quantization state, without decoding to floats (`mlmf-bnb`).
9. Read the raw positional imatrix layout (no magic, no version) by bounding every length against the remaining bytes and refusing rather than guessing (`mlmf-gguf`).
10. Read and write ONNX `ModelProto` files: initializers as byte ranges of the input, external-data and typed-data tensors reported but not decoded, graph nodes opaque; byte-identical to the reference serializer (`mlmf-onnx`).
11. Read and write the `torch.export` `.pt2` zip archive layout (weights and constants configs, tensor byte ranges, graph opaque) (`mlmf-pt2`).
12. Read and write the MLIR bytecode container used by StableHLO portable artifacts: magic, version, producer and section table, sections opaque (`mlmf-stablehlo`).
13. Fetch files from a HuggingFace Hub repository into a local cache (`mlmf-source-hub`, I/O only), and report where each tensor of a HuggingFace checkpoint lives (shards) and what its JSON sidecar files declare, from bytes the caller supplies (`mlmf-hf-layout`). (Header-only range requests are not implemented; the range-request queries below were run as prior-art checks on an idea we may adopt.)
14. Treat an absent file field as `Option::None`, never a default, so a caller cannot mistake "the file said nothing" for a value (the standing policy in `CLAUDE.md` section 2).

None of these claims a new algorithm for training, inference, quantization, compression or scheduling; they are file-format work against formats others define (safetensors, GGUF, ONNX, pickle, `.pt2`, MLIR bytecode, bitsandbytes), recorded in `NOTICE`.

## Queries (source, query, date)

Search tool: a general web search (not a patent database), run 2026-10-25; unrestricted queries and queries restricted to `patents.google.com` / `patents.justia.com`. Patent pages were then read on Google Patents with a page-reading tool.

| # | idea | query (abridged) | restriction |
|---|---|---|---|
| Q1 | 7 | patent claims safe unpickling allowlist pickle opcodes ML model file loading restricted globals | none |
| Q2 | 1, 2 | patent method reading neural network model file tensor directory byte offsets without loading weights memory-mapped lazy loading | none |
| Q3 | 5 | patent detecting model architecture from tensor names in checkpoint file format identification | none |
| Q4 | 4, 8 | patent quantized weights file format block quantization 4-bit normal float storage and reading model weights container | none |
| Q5 | 6 | patent unified interface reading multiple ML model file formats abstraction container trait metadata tensors | none |
| Q6 | 13 | patent downloading model weights from model repository hub caching partial range requests safetensors header | none |
| Q7 | 1, 2, 3 | ML model file format parser tensor header offsets weights file loading method claims | Google Patents, Justia |
| Q8 | 7 | deserialization of serialized model object allowlist of permitted functions pickle security checkpoint | Google Patents, Justia |
| Q9 | 10, 11 | converting neural network model between file formats ONNX serialization graph initializers tensors | Google Patents, Justia |
| Q10 | 5 | identify type of neural network model file from file contents header magic number select loader | Google Patents, Justia |
| Q11 | 13 | retrieving portion of model weights file remote storage range request header determine tensor location | Google Patents, Justia |
| Q12 | 12 | compiler intermediate representation bytecode container sections serialization MLIR portable artifact versioned | Google Patents, Justia |
| Q13 | 1, 2 | loading trained model parameters from file; header records tensor name dtype shape data offsets; zero-copy memory mapping | Google Patents, Justia |
| Q14 | 4, 8 | block-wise quantized weight storage format scale per block 4-bit packed tensor file for LLM inference | Google Patents, Justia |
| Q15 | 7 | checkpoint file loading restrict deserialization to permitted object types prevent code execution ML model | Google Patents, Justia |
| Q16 | 13 | model repository downloading ML model files local cache directory snapshot revision hash deduplicated | Google Patents, Justia |

Ideas 9 and 14 were not searched separately (idea 9 is a layout reading rule, idea 14 is a type-level policy); they share Q2/Q7.

## Hits

"Claims read" means the claims section was read through a page-reading tool that returned paraphrases and short quotes, not the verbatim text; limitations are as that tool reported them.

| # | patent | title | assignee | status | claim(s) read | overlap | our difference | verdict |
|---|---|---|---|---|---|---|---|---|
| H1 | US11403080B2 | General machine learning model, and model file generation and parsing method | Shanghai Cambricon Information Technology | granted, in force (4th-year fee paid; est. expiry 2039) | independent 1 (method), 4 (device) | Generates a model file holding a model and a model directory, computing each model's storage offset from sizes, count and order (cf. our `encode_gguf` / `encode_safetensors` computing tensor offsets and a header/directory) | Every independent claim starts from machine-learning task parameters sorted into task instructions and model parameters, grouped by data type into **stack data and heap data**, and combined into a "general-purpose machine learning model"; then adds a check/error-correction code. We take tensors already encoded; there is no task, no stack/heap split, no check code | clear |
| H2 | US11334330B2 | same family (parsing) | Shanghai Cambricon | granted, in force (est. expiry 2039-06) | independent 1, 4 | Parses a model file by reading a model directory, then reading the target model according to the directory (cf. our tensor-directory readers) | Both independent claims require a file-header checker reading an identification code against a preset rule, a check code with a file-tailer error-correction step, **and a hardware matcher matching hardware in a device pool**. We have none of the three | clear |
| H3 | US11379199B2 | same family (parsing a "secondary model" file) | Shanghai Cambricon | granted (status not separately read) | independent 1, 5 | Reads a "secondary model directory" and a target secondary model; claim 2 recites a storage offset | Requires an identification-code rule check, a check code with error correction, and restoring a secondary model into a general-purpose model; we have no secondary-model construct | clear |
| H4 | US11036480B2 | same family | Shanghai Cambricon | active (assumed), est. expiry 2039-05 | independent 1, 10, 15 | None of the claims mention a storage offset or directory | Claim 1 executes a model on a hardware computing unit, splitting unsharable data into heap data; we run nothing | clear |
| H5 | US11726754B2 | same family | Shanghai Cambricon | active (assumed), est. expiry 2039-05 | independent 1, 19, 20; dependent 9-13, 16, 18 | Dependent claim 9 recites calculating a storage offset and generating a model directory; 16 and 18 read the directory | Independent claim 1 requires processing task parameters into shareable data (heap block) and unshareable data (stack block) and packing both into a "general-purpose machine learning model"; the offset/directory claims depend on it. No such packing here | clear |
| H6 | US11334329B2 | same family | Shanghai Cambricon | not separately read | independent 1, 10 | none on offset, directory or parsing | Claim 1 classifies task parameters, aggregates by data type into stack and heap data, integrates into a general machine learning model; not what we do | clear |
| H7 | US11611355B2 | Techniques for parameter set and header design for compressed neural network representation | Tencent America | active (assumed), est. expiry 2041 | independent 1, 11, 20 | A bitstream with headers and units describing a neural network (cf. any model container) | Claims a Neural Network Representation (NNR) bitstream of NNR units (model/layer parameter sets, topology, quantization and compressed data units) decoded to reconstruct the network; we read none of that format | clear |
| H8 | US11829850B2 | Reducing complexity of implementing machine learning models in software systems | RedCritter Corp | active (assumed), est. expiry 2042 | independent 1 (visible part; not all 20) | A model "meta file" and one API over many models (cf. our format-agnostic seam) | Claim 1 is a developer-authored model meta file for a specific student-practice model with converters that build a model request to a prediction service; we handle model files, not requests to services | clear |
| H9 | US12393861B1 | Systems and methods for securing untrusted code | not visible on the page read | granted (status not read) | independent 1, 10, 18 | Deserializes an ML model's computation graph and validates it before storing (cf. our restricted pickle read) | Claims also require storing in a pipelined-search system, running the model as an operator of a pipelined search query, and displaying results on a UI; the claims do not recite an allow list. Ours is a file reader | clear |

## Seen, claims not read (no verdict)

These surfaced in the searches; their claims were **not** read, so no verdict is given. They are listed so a later reader does not mistake them for cleared.

| patent | title | assignee | why seen | note |
|---|---|---|---|---|
| US11605020B2 (and continuation US12008448) | Documentation file-embedded machine learning models | AT&T Intellectual Property I | model-file search (Q11/Q12) | title suggests embedding documentation in a model file; we embed nothing. Claims unread |
| US10853110B2 | Constructor accessibility checks for deserialization | Oracle | allow-list deserialization (Q15) | Java deserialization; claim 1 unread |
| US12339986 | Runtime application self-protection | not read | allow lists for deserialization (Q15) | Java context per the search summary; unread |
| US7814124B1 | Architecture and method for serialization and deserialization of objects | not read | Q15 | general object serialization; unread |
| US11307836B2 | General machine learning model... (same Cambricon family) | Shanghai Cambricon | Q7 | claims section not retrieved; the other five readable family members are clear (H1-H6) |
| US11445051B2 | Configurable parser for information units | not read | Q7 | network packet parsing; irrelevant by title |

## Needs review (to the PM)

None from the hits read. Open items rather than reviews: finish reading the "seen, claims not read" table, and the unfetched members of the Cambricon family (US11307836B2, and any continuation later than US11726754B2).

## Limitations

- AI-assisted; not legal advice and not a freedom-to-operate opinion. A lawyer's review is a separate, later purchase.
- The search tool is a general web search, **not** a patent database. It does not run boolean or classification (CPC) queries, and it returned patents only for some queries. A null result for a query (no patent for idea 4/8/13, or most of idea 5) shows what this tool surfaced, not that none exist. Controls: the same tool found the Cambricon family for Q7, so it does reach patent pages for some phrasings.
- USPTO PPUBS, Espacenet and WIPO PATENTSCOPE were **not** queried directly. Only US patents were read; no EP, CN, JP or WO searches beyond what a US-only search returned (one WO hit, WO2022079545, was seen and not read).
- Claims were read through a summarizing page reader: limitations are the reader's paraphrase, quote-limited. Verdicts rest on limitations that excluded us (stack/heap packing of task parameters; hardware matcher; NNR bitstream; pipelined search), each reported for every independent claim of the patents read. A close re-read of verbatim claims is needed before relying on any "clear" here.
- Patents prior to a format (for example the Cambricon family, priority 2018-06) cannot be prior art against safetensors, GGUF, or `.pt2`, and the formats themselves are third-party specifications that we implement; this sweep says nothing about their owners' rights.
- Not swept: the code removed in 0.7.0 and 0.7.12 (distributed loading, quantization and calibration, caching, model cards, multimodal fusion); ML-ConvertAnt now owns quantization.
- Re-sweep due on a new family of capability, and by 2027-10-25.
