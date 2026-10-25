# Patent sweep: mlmf 0.7.x @ 1f6e961 (second record, 2026-10-25), plus `mlmf-gguf-arch` @ 54a8929 (Part C, third record)

Date: 2026-10-25; Run by: mlmf lane, Claude Sonnet 5.5; Method: AI-assisted search and read; NOT legal advice; NOT a freedom-to-operate opinion.

Process: `C:\Projects\PATENT-SWEEP-PROCESS.md` (CireSnave, board 286 item 1). This record replaces the first one (0.7.18), which read claims through a paraphrasing page reader. **Every patent below was re-read with `tools/patent-claims.py`, which prints the claim text in full, and the verdicts rest on the independent claims as printed.** Google Patents' text is machine-converted; for anything that matters, check the official record.

Scope, in two parts:

- **A. The maintained workspace**: the `mlmf-*` crates at `origin/main` `1f6e961` (the 0.7.15 release plus 0.7.16 and 0.7.17). Version line **0.7.x**.
- **B. Code removed before 0.7.0 and still in published releases**: the old root `mlmf` package had distributed loading, quantization with calibration, a model cache, a model-card generator, multimodal fusion and a memory estimator. It was removed in 0.7.0 and 0.7.12 (old code at tag `v0.7.11`), but it was shipped in tags `v0.2.1` and `v0.3.0`, and crates.io carries `mlmf` 0.2.0 (the root package; MIGRATION.md; whether that tarball includes each module was not checked). It is swept in part B because releases that shipped it were made without a sweep. Quantization lives on in ML-ConvertAnt; see "Hand-over" at the end.

## Part A: ideas swept (the maintained workspace)

MLMF reads and writes model files; it does not run, convert or interpret them.

1. Parse a safetensors file's header and report each tensor's name, dtype, shape and byte range, without reading tensor data (`mlmf-safetensors`).
2. Index a GGUF file's key-value block at open without decoding values, bounding every allocation against the bytes that remain, and report a tensor directory with byte ranges (`mlmf-gguf`).
3. Write safetensors and GGUF files from tensors the caller already encoded in the file's dtype, computing the header or directory and offsets (`encode_safetensors`, `encode_gguf`).
4. A table of ggml quantization types (block size, bytes per block, alignment) measured from a reference implementation, used to compute a tensor's byte size without decoding it (`mlmf-ggml`).
5. Each format crate recognises its own file from its leading bytes and version and refuses anything else. (Architecture detection from tensor names and tensor-name mapping are in scope but **planned, not shipped**: ROADMAP S-12 and S-30. Sweep again when built.)
6. A format-agnostic seam: one trait for "a container of named tensors with byte ranges" and one for "declared key/value metadata", implemented by each format crate and composable with any byte source (`mlmf-core`).
7. Read a `torch.save` pickle checkpoint without running it: a small pickle interpreter that executes only an allow-list of four `REDUCE` targets, treats everything else as an error, and reports tensor storages as file ranges (`mlmf-pickle`).
8. Read bitsandbytes 4-bit (NF4/FP4) layers, including double-quantized state, as bytes plus the declared quantization state, without decoding to floats (`mlmf-bnb`).
9. Read the raw positional imatrix layout (no magic, no version) by bounding every length against the remaining bytes and refusing rather than guessing (`mlmf-gguf`).
10. Read and write ONNX `ModelProto` files: initializers as byte ranges of the input, external-data and typed-data tensors reported but not decoded, graph nodes opaque; byte-identical to the reference serializer (`mlmf-onnx`).
11. Read and write the `torch.export` `.pt2` zip archive layout (weights and constants configs, tensor byte ranges, graph opaque) (`mlmf-pt2`).
12. Read and write the MLIR bytecode container used by StableHLO portable artifacts: magic, version, producer and section table, sections opaque (`mlmf-stablehlo`).
13. Fetch files from a HuggingFace Hub repository into a local cache (`mlmf-source-hub`, I/O only), and report where each tensor of a HuggingFace checkpoint lives (shards) and what its JSON sidecar files declare, from bytes the caller supplies (`mlmf-hf-layout`). (Header-only range requests are not implemented; the range-request queries were prior-art checks on an idea we may adopt.)
14. Treat an absent file field as `Option::None`, never a default, so a caller cannot mistake "the file said nothing" for a value (`CLAUDE.md` section 2).

None of these claims a new algorithm for training, inference, quantization, compression or scheduling; they are file-format work against formats others define (safetensors, GGUF, ONNX, pickle, `.pt2`, MLIR bytecode, bitsandbytes), recorded in `NOTICE`.

## Part B: ideas swept (removed code, read from `git show 34afd0e^:<path>` and `a3e37f7^:<path>`)

15. **Distributed loading** (`src/distributed*.rs`): node and device descriptions, a sharding strategy (none, by layers, by dimension, pipeline stages), placement and memory-allocation strategies, a communication and security configuration for a cluster.
16. **Post-training quantization with calibration** (`src/quantization.rs`; the orphaned `src/quantization_simple.rs`, which was never compiled in): quantization schemes (INT8, INT4, dynamic, static), a calibration dataset, min/max, percentile and entropy (histogram-bin) statistics, scale and zero-point computation; quantize-then-dequantize ("fake" quantization).
17. **A loaded-model cache** (`src/cache.rs`, `src/cached_loader.rs`): LRU eviction, a time-to-live, memory-pressure levels that trigger eviction.
18. **A model-card generator** (`src/model_card.rs`): a `ModelCard` structure (model, training, usage, evaluation, technical specs, memory requirements) filled from a loaded model and written as JSON.
19. **Multimodal processing** (`src/multimodal*.rs`): per-modality encoders; cross-modal attention for every ordered pair of modalities, each modality's embedding replaced by its attended version; a fusion layer (concatenation, mean with projection, or attention-weighted sum of modalities).
20. **A memory estimate for a model** (inside the model-card generator: `MemoryRequirements`, from `validation::estimate_memory_usage`): parameter and inference memory computed from the model's declared configuration (layer count, hidden size and the like), training memory taken as 4x parameters and recommended RAM as 2x inference.

## Part C: ideas swept (added for `mlmf-gguf-arch`, 0.7.20, `origin/main` `54a8929`)

21. Name the architecture family of a GGUF file from the family string the file itself declares (`general.architecture`), matched byte-exactly against a fixed list, and answer "none" when the declaration is unlisted, not a string or undecodable: a declaration is never overridden by a guess (`classify`).
22. When the key is absent from a metadata view known to be complete, name the family from distinctive tensor-name layouts only (GPT-2 `transformer.h.*.attn.c_attn`, GPT-NeoX `gpt_neox.layers.*`), answer "none" when both or neither show, and never infer from the near-universal `blk.*` layout; report whether the answer was declared or inferred (`Basis`).

This is the first shipped part of idea 5's planned architecture detection (ROADMAP S-12). Tensor-name mapping (S-30) is still unbuilt.

## Queries (source, query, date)

Search tool: a general web search (not a patent database), run 2026-10-25; unrestricted queries and queries restricted to `patents.google.com` / `patents.justia.com`. Candidate patents were then read in full with `tools/patent-claims.py`.

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
| Q17 | 15 | distributed model loading shard neural network weights across devices nodes tensor parallel pipeline parallel | Google Patents, Justia |
| Q18 | 16 | post-training quantization calibration dataset activation statistics min max entropy percentile scale zero point | Google Patents, Justia |
| Q19 | 17 | caching loaded machine learning models in memory eviction least recently used model cache inference server | Google Patents, Justia |
| Q20 | 18 | automatically generating model card documentation for machine learning model from model metadata | Google Patents, Justia |
| Q21 | 19 | multimodal model fusion cross-modal attention vision language image text embeddings combine | Google Patents, Justia |
| Q22 | 20 | estimating memory requirement of loading neural network model from weight file size dtype shape before loading | Google Patents, Justia |

Part C queries, run 2026-10-25 with the same general web search tool (see the throttle note under Limitations):

| # | idea | query (abridged) | restriction |
|---|---|---|---|
| Q23 | 21, 22 | patent identifying neural network model architecture type from model file metadata declared architecture string or tensor names layer naming | Google Patents, Justia |
| Q24 | 21, 22 | patent determining model family of machine learning model file from parameter names in checkpoint to select loader | Google Patents, Justia |
| Q25 | 21, 22 | patent method automatically detecting type of trained model from file contents header inspect tensor names select matching model architecture loader large language model weights | none |

Ideas 9 and 14 were not searched separately (a layout reading rule and a type-level policy); they share Q2/Q7.

## Hits

Each patent's **independent claims** were printed in full with `tools/patent-claims.py` on 2026-10-25. "Our difference" names the limitation that every independent claim requires and we lack.

### Part A hits

| # | patent | title | assignee | claims read | overlap | our difference | verdict |
|---|---|---|---|---|---|---|---|
| H1 | US11403080B2 | General machine learning model, and model file generation and parsing method | Shanghai Cambricon | indep. 1, 4 | Generates a model file holding a model and a model directory with a storage offset per model (cf. `encode_gguf` / `encode_safetensors` computing offsets and a header) | Both claims start from "task parameters of a machine learning task", classify them into task instructions and model parameters, aggregate by data type into "stack data and heap data" and integrate them into a "general-purpose machine learning model", then add "a check code or an error correction code". We take tensors already encoded; no task parameters, no stack/heap, no check code | clear |
| H2 | US11334330B2 | same family (parsing) | Shanghai Cambricon | indep. 1, 4 | Parses a model file by reading its model directory, then the target model per the directory (cf. our directory readers) | Both claims require a file header checker reading an identification code against a preset rule, a check code with a file-tailer error-correction step, **and a hardware matcher that "match[es] the corresponding hardware in a device pool"** | clear |
| H3 | US11379199B2 | same family (secondary model file) | Shanghai Cambricon | indep. 1, 5 | Reads a "secondary model directory" and a target secondary model | Requires identification-code rule, check code with error correction, and "restoring the target secondary model to obtain a target general-purpose machine learning model" | clear |
| H4 | US11036480B2 | same family | Shanghai Cambricon | indep. 1, 10, 15 | None on offsets or directories | "Executing a general-purpose machine learning model in a hardware computing unit", with stack and heap areas and dynamic/static heap sub-areas; we run nothing | clear |
| H5 | US11726754B2 | same family | Shanghai Cambricon | indep. 1, 19, 20 | Dependent claims 9-13, 16, 18 recite storage offsets and a model directory | Independent claim 1: "processing task parameters to obtain shareable data and unshareable data" (data shared among cores in a multi-core platform), arranged into heap and stack blocks and "packing" them into a general-purpose model | clear |
| H6 | US11334329B2 | same family | Shanghai Cambricon | indep. 1, 10 | none on offsets or directories | Classifies task parameters, allocates storage for input/output/intermediate data as heap and stack, integrates into a general model | clear |
| H7 | US11307836B2 | same family (generating a secondary model file) | Shanghai Cambricon | indep. 1, 5, 9 | Generates a file with a "secondary model directory" from computed storage offsets (cf. our writers) | The model is "generated by integrating a stack data and a heap data ... aggregating the task instructions and the model parameters of task parameters", then "storage optimization" produces the secondary model; we do neither | clear |
| H8 | US11611355B2 | Techniques for parameter set and header design for compressed neural network representation | Tencent America | indep. 1, 11, 20 | A bitstream of header+payload units describing a network (any model container) | Claims an NNR bitstream of a group of NNR units (model/layer parameter set, topology, quantization and compressed-data units) decoded to "reconstruct the independent neural network"; we read none of that format | clear |
| H9 | WO2022079545A1 | Carriage and signaling of neural network representations | Nokia | indep. 1, 26, 34, 59, 67 (the rest are dependent-form) | Stores NNR in a media file | Claims a metadata box for NNR item data with a configuration item property, or an ISO-base-media-file-style file of NNR tracks with a media handler; we read neither | clear |
| H10 | US11829850B2 | Reducing complexity of implementing machine learning models in software systems | RedCritter | indep. 1, 14 | A model "meta file" and one API over many models (cf. our seam) | Claims a model meta file editor for a model predicting whether a student needs more practice, converters, and submitting a model request to a machine learning service | clear |
| H11 | US12393861B1 | Systems and methods for securing untrusted code | Cisco Technology | indep. 1, 10, 18 | Deserializes an ML model's computation graph to verify it before storing it (cf. our restricted pickle read) | Also requires comparing input schemas, storing the representation, executing "a pipelined search query ... wherein the ML model is deployed as either the first operator or the second operator" and displaying results on a UI; no allow list is recited | clear |
| H12 | US10853110B2 | Constructor accessibility checks for deserialization | Oracle | indep. 1, 8, 15 | Deserialization gated by a check made before it runs | The check is on "a constructor of the ancestor type" being accessible to the target type, in a type system with constructors, "without calling any constructors"; a pickle reader has no ancestor-type constructor accessibility check | clear |
| H13 | US12339986B2 | Runtime application self-protection | Oracle | indep. 1, 11, 18 | Uses an allow list | A software agent in "logging mode" then "blocking mode", a security service that generalizes observed operations into the allow list and trains a "confidence estimator model"; our allow list is four fixed REDUCE targets | clear |
| H14 | US7814124B1 | Architecture and method for serialization and deserialization of objects | Microsoft | indep. 1, 5, 11 | Object-graph (de)serialization | Surrogate objects carrying user-defined rule sets, proxy references and forward references; not a model-file reader | clear |
| H15 | US11605020B2 and US12008448B2 | Documentation file-embedded machine learning models | AT&T Intellectual Property I | indep. 1, 8, 13 / 1, 9, 17 | Reads/writes a serialized model in a file | Embeds a serialized model into a documentation file "after an end-of-file indicator" using a delimiter byte sequence tied to a text code in a metadata field, and extracts it by that delimiter; we embed nothing in a documentation file | clear |
| H16 | US11445051B2 | Configurable parser and a method for parsing information units | Amazon Technologies | indep. 1, 11 | "Parsing" by field offset | A packet processing method (protocol field comparison, parse control table, completion descriptor for a forwarding engine) | clear |

### Part C hits (`mlmf-gguf-arch`)

Independent claims printed in full with `tools/patent-claims.py` on 2026-10-25. Q23 and Q24 returned no patent that claims identifying a model's family from a file's declared string or its tensor names; Q25 returned only open-source loaders (prior art for idea 22, not patents). Two patents surfaced as the nearest and were read.

| # | patent | title | assignee | claims read | overlap | our difference | verdict |
|---|---|---|---|---|---|---|---|
| H29 | US20250362963A1 | Dynamically selecting artificial intelligence models and hardware environments to execute tasks | Dropbox | indep. 1, 11 | Selects a machine-learning model | Requires receiving workload data, extracting workload features, "task routing metrics" from historical execution information, selection by "a hardware allocating machine-learning model", and executing the task; we classify a file's family and run nothing | clear |
| H30 | US5444824A (expired) | Enhanced neural network shell for application programs | IBM | indep. 1, 4 | A neural-network data structure naming its model type | Claims prompting a user for a problem type and an input data file, "selecting a neural network model based on said problem type and said input data format" and creating the data structure; we read a family from the file's own declaration. Google Patents lists it Expired; filed 1993 | clear |

### Part B hits (removed code)

| # | patent | title | assignee | claims read | overlap | difference (removed code) | verdict |
|---|---|---|---|---|---|---|---|
| H17 | US12700225B2 | Channel fusion for vision-language representation learning | Google | indep. 1, 7, 15, 18 | Cross-attention between two modalities, then fusion: close to `multimodal_processor.rs` | Claim 1/7 requires, per fusion layer, cross-attention with one modality's tokens as queries and the other's as keys/values, **concatenated "in a channel-wise fashion" with that same modality's original embedding tokens** to make "compound tokens", done in both directions. The removed code ran the cross-attention for each ordered pair and **replaced** the query modality's embedding with the attended output (`attended_embeddings.insert(*query_mod, attended)`); it never concatenated the attended output with the original embedding; its fusion step concatenated embeddings *across* modalities (`Tensor::cat` of the attended vectors), averaged them, or took an attention-weighted sum | clear on the claims as read; closest call in this sweep, see Needs review |
| H18 | US11657230B2 | Referring image segmentation | Adobe | indep. 1, 13, 18 | Combines image and language features | Claims generating an image mask for a target object from an image feature vector and language feature vectors "in a plurality of language embedding spaces having a different number of dimensions" and overlaying the mask; no segmentation here | clear |
| H19 | US12099930B2 | Parametric power-of-2 clipping activations for quantization for convolutional neural networks | Texas Instruments | indep. 1, 10, 17 | Calibration from histogram statistics (cf. entropy bins in `quantization_simple.rs`) | Requires arranging data as a histogram, **discarding the tail**, estimating a maximum, a **moving average** with prior sets' maxima, and a clipping value "expanded to a nearest power of two"; the removed code computed min/max, a percentile and histogram bins (the latter in the orphan file) with none of that (grep for histogram/moving average/power-of-two in `quantization.rs` is empty; in `quantization_simple.rs` only the histogram-bin function is present) | clear |
| H20 | US20230342613A1 | System and method for integer only quantization aware training on edge devices | Alphaics | indep. 1, 8 | Quantization | Quantization-**aware training**: pseudo cross entropy, gradient stabilization, back-propagation, residual weight error; the removed code trained nothing | clear |
| H21 | US11461300B2 | Dynamic model server for multi-model machine learning inference services | SAP | indep. 1, 7, 13 | Models loaded into memory with an LRU-style cache | Stateless API servers calling stateful model servers, and "adjusting a number of model servers" from the cache hit/miss ratio; our cache was one in-process LRU/TTL structure | clear |
| H22 | US20250110784A1 | Caching in a machine learning model hosting service | Amazon Technologies | indep. 1, 4, 13 | Loading models into memory by usage, with a cache manager | Hosts, a router with a hash function, host usage data, a second set of hosts with a software update, a priority order per model; ours is local | clear |
| H23 | US20200104749A1 and US20200104750A1 | Pre-warming scheme to load machine learning models; Hosting machine learning models | Amazon Technologies | indep. 1, 8, 15 / 1, 4, 14 | Loading models into RAM ahead of use | Candidate sets of hosts, groups of hosts, routing inference requests, a bot/utterance load-balanced path; not a local cache | clear |
| H24 | US11263188B2 | Generation and management of an AI model documentation throughout its life cycle | IBM | indep. 1, 8, 15 | Automatically generates documentation for a model (cf. the model-card generator) | Requires reading "a model facts policy ... indicating data to be collected" and "a factsheet template ... specif[ying] a subset of the AI model facts and their order"; the removed `model_card.rs` has no policy or template construct (grep for template/policy/facts in that file is empty) | clear |
| H25 | US11847574B2 | Systems and methods for enriching modeling tools and infrastructure with semantics | ZestFinance | indep. 1, 3, 4 | Metadata about a model stored in a database | Reference distributions of feature values over training/validation datasets and detection of unexpected production input by comparing them; not done | clear |
| H26 | US11500767B2 | Method and device for determining a global memory size ... for a neural network | STMicroelectronics | indep. 1, 7, 12, 19, 24 | Computes a memory size a network needs | Pairs of elementary memory areas per layer holding each layer's input and output data and the smallest size at the last layer; ours computed parameter and inference memory from declared configuration values for a card field, with fixed 4x and 2x multipliers, and no layer-by-layer buffer reuse | clear |
| H27 | US10698766B2 | Optimization of checkpoint operations for deep learning computing | EMC | indep. 1, 10, 18 | Checkpoints in distributed deep learning | Executing a distributed training process on accelerators and generating checkpoint images during training; we train nothing | clear |
| H28 | US20220083386A1 | Method and system for neural network execution distribution | Samsung | indep. 1, 15 | Spreads a network across devices | Identifies computing resources and their load on a user device, partitions the network to satisfy an optimisation constraint, assigns and schedules execution; the removed config only described shards and nodes | clear |

## Needs review (to the PM)

None of the 30 reads needed a decision. Flag for awareness only: **H17** is the one place where the removed code and a granted claim sit close (cross-attention between modalities followed by combining). The distinction rests on a specific limitation (channel-wise concatenation of the attended tokens with the same modality's original tokens, in both directions) that the removed code does not perform. That reading is mine and not a lawyer's, and the removed multimodal code was part of tags `v0.2.1`/`v0.3.0` (whether the crates.io root `mlmf` 0.2.0 tarball includes it was not checked). The maintained workspace has no such code. If CireSnave wants the question closed, a short FTO read of US12700225B2 claim 1 on `src/multimodal_processor.rs` as of `34afd0e^` is the check.

## Limitations

- AI-assisted; not legal advice and not a freedom-to-operate opinion. A lawyer's review is a separate, later purchase.
- The search tool is a general web search, **not** a patent database: no boolean or CPC queries, and it returns patents only for some phrasings. A query that surfaces no patent (ideas 4/8/13, most of 5) shows what this tool surfaced, not that none exist. Control: the same tool found the Cambricon family for Q7, so it does reach patent pages for some phrasings.
- USPTO PPUBS, Espacenet and WIPO PATENTSCOPE were **not** queried directly. Hits are US patents plus two non-US documents that surfaced (WO2022079545 read; CN117669701A and CN114222997A seen, not read).
- Claim text is Google Patents' machine conversion; the Cambricon family may have other members (continuations) not surfaced here. Seven members were read (H1-H7).
- Patents with priority before a format (the Cambricon family, 2018-06) are not prior art against safetensors, GGUF or `.pt2`, and those formats are third-party specifications we implement; this sweep says nothing about their owners' rights.
- Part B reads the removed code's headers and the passages quoted; it did not trace every code path. The removed code was largely scaffolding (configuration structures, placeholder INT4, simplified statistics), which is itself why most differences above are "never done".
- Part C was thin by necessity: Google Patents had throttled the shared IP on 2026-10-25, so only three web searches were made (two together, one later) and two patents fetched, with a 45 s pause between fetches. The third search surfaced no patents, so ideas 21 and 22 rest on two queries that returned nothing on point plus two reads. Control: the same tool returned patent pages (US5444824, US20250362963) for Q23/Q24, so it reaches patents for these phrasings. Open-source loaders that infer architecture from tensor names exist (ComfyUI-GGUF, Spandrel, llama.cpp itself); that is prior art for idea 22, not a patent read.
- Re-sweep due on a new family of capability (tensor-name mapping, ROADMAP S-30, when built; the detector S-12 is covered by Part C), and by 2027-10-25.

## Hand-over

Ideas 16 (quantization and calibration) and the rest of Part B concern code ML-ConvertAnt now starts from. This file is the mlmf lane's record; ML-ConvertAnt needs its own sweep that includes ideas 15-20 where it reuses them.
