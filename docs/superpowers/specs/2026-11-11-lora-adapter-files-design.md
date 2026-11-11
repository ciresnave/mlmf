# LoRA adapter files: read and write

Design note, ROADMAP S-65. No code yet; this note comes first (PM `[TASK]`, 2026-11-11).

## 1. The ask and where it sits

CireSnave, 2026-11-11, relayed verbatim by the PM: *"Have MLMF add reading and writing of LoRA adapters to its roadmap."*

An adapter file is a model file, so reading and writing it is MLMF's job under the charter (`CLAUDE.md` section 0: *"If the code adds to MLMF's ability to read or write (but not interpret or guess about) machine learning model files, it should be in MLMF."*). Three limits come with it:

- **Describe, do not interpret.** MLMF reports the fields the file declares. What a rank or an alpha *means* for a forward pass is a consumer's job.
- **No merge.** Folding an adapter into base weights computes on values, which is conversion, and CireSnave ruled that out of MLMF (S-33, S-53; ML-ConvertAnt's). An adapter stays a separate file and a separate object.
- **The retired root `src/lora.rs` is not the home.** The root package is gone (S-37, `git ls-tree -r origin/main` finds no `src/`). The work goes in the modular crates, as a format-axis concern.

The reason it is wanted: a consumer can apply an adapter at graph level at run time but has no file loader for one. The loader is this crate's side of that seam.

## 2. Two file forms, both public

Everything below is read from public sources; nothing else is cited.

### 2.1 PEFT: `adapter_config.json` plus `adapter_model.safetensors`

Source: Hugging Face PEFT, `src/peft/config.py` and `src/peft/tuners/lora/config.py` (read at `main`, 2026-11-11).

`adapter_config.json` fields the LoRA config class declares and that this work would report, each as `Option<T>` (standing policy, `CLAUDE.md` section 2):

| key | what the class says it is |
|---|---|
| `peft_type` | the PEFT method; the only key the loader treats as required (`MIN_EXPECTED_CONFIG_KEYS = {"peft_type"}`) |
| `base_model_name_or_path` | the base model the adapter was made for |
| `revision`, `task_type`, `inference_mode`, `auto_mapping` | base config |
| `r` | LoRA rank |
| `lora_alpha` | alpha |
| `target_modules`, `exclude_modules` | a list of names, or one string (a regex); both shapes occur |
| `use_rslora`, `fan_in_fan_out`, `bias`, `modules_to_save` | declared flags and lists |
| `rank_pattern`, `alpha_pattern` | per-module overrides of `r` and `lora_alpha` |
| `layers_to_transform`, `layers_pattern` | layer selection |

**The section 6 fence applies directly.** The PEFT class constructor defaults `r` to 8 and `lora_alpha` to 8. Those are defaults of a *Python class*, not of the file format: the file format has no "absent means 8" statement. So an `adapter_config.json` with no `r` reads as `r: None`, never 8. (This is the `mlmf-hf-config` distinction again: a citable default of the *format* may be filled; a default of a consuming library's class may not, unless a ruling like S-14's says so. None is asked for here.)

**No scale is computed.** The effective scaling is `lora_alpha / r`, or `lora_alpha / sqrt(r)` when `use_rslora` is set. That is an inference from declared fields. MLMF reports `r`, `lora_alpha` and `use_rslora` as declared and leaves the division to the consumer. If a ruling later wants it supplied, that is a question for CireSnave, not a default to slip in.

Tensors, in `adapter_model.safetensors`: per target module a pair named with a `.lora_A.weight` and a `.lora_B.weight` suffix (and `.lora_embedding_A` / `.lora_embedding_B` for embeddings), which is what llama.cpp's converter keys on (`convert_lora_to_gguf.py`, `is_lora_a = ".lora_A.weight" in name or ".lora_embedding_A" in name`). The prefix in front of the module path and the shapes of A and B are NOT measured here: **step 1 below measures them over real adapters before any type is fixed.** The tensor bytes stay in the file's dtype; nothing is cast.

Pairing is by name and is a *finding*: an A with no B, or a B with no A, is reported as an orphan, not dropped and not completed.

### 2.2 GGUF: `general.type = "adapter"`

Source: llama.cpp `gguf-py/gguf/constants.py` and `convert_lora_to_gguf.py` (read at `master`, 2026-11-11). A GGUF adapter exists and is declared by metadata:

| key | value |
|---|---|
| `general.type` | `"adapter"` (`GGUFType.ADAPTER`; models are `"model"`) |
| `general.architecture` | the base architecture the adapter targets |
| `adapter.type` | `"lora"` (the converter writes this string); the constants file also defines `adapter.alora.invocation_tokens`, which implies an activated-LoRA kind whose `adapter.type` spelling is not read here |
| `adapter.lora.alpha` | f32, written from the PEFT `lora_alpha` |
| `adapter.lora.task_name`, `adapter.lora.prompt_prefix` | optional strings |

Tensors are named `<base tensor name>.lora_a` and `<base tensor name>.lora_b`, where the base name is the GGUF name of the base weight (`blk.0.attn_q.weight` style). The converter transposes the embedding's `lora_a` on the way in; that is a producer-side detail MLMF reads as stored, and **how it shows in the dimension order is another step 1 measurement.**

Note what the GGUF form does NOT carry: no `r` key. The rank is a tensor dimension, so a reader derives it from a shape and says so (it is a shape fact, not a guess). `adapter.lora.alpha` can be absent, and then it is `None`.

The existing `mlmf-gguf-arch` and `mlmf-tensor-names` already map between HF and GGUF module names for the dense layout; the adapter code reuses them for the name half and adds nothing to them without a measurement.

## 3. Shape of the work

Not a new crate by default. The decision, recorded so it is not re-taken:

- **PEFT reader/writer:** `mlmf-safetensors` already reads and writes the tensor file (S-47) and `mlmf-hf-layout` reads sidecar JSON. A small `mlmf-lora` crate holding the two adapter *types* (`AdapterConfig` of `Option` fields, `AdapterPair { a, b }`, an orphan list) with PEFT and GGUF readers behind it follows the S-14 precedent (`mlmf-hf-config`: depends on `mlmf-core` alone, reads any `MetadataSource`). Recommended; the alternative of spreading it over two format crates duplicates the pairing rule.
- **GGUF side:** the metadata keys read through `mlmf-gguf`'s `MetadataSource`; writing through `encode_gguf` (S-49), which already takes tensors pre-encoded. No new GGUF primitive.
- **Dependency rule:** crates.io versions only (`CLAUDE.md` portfolio section 9); the direct-deps allow-list and axis tests apply to the new crate as to the others.

## 4. Round trip, measured

The acceptance test is a measurement, not a type that compiles. For each form, read a real adapter, write it, read the output, and compare:

1. **Tensor bytes equal**, name by name (the same payload-identity comparison S-39 uses), and the dtype and shape list equal.
2. **Declared fields equal**, as `Option` values, so a field absent on the way in is still absent on the way out (a writer that adds `"r": 8` fails this).
3. **Both ways round**: PEFT to PEFT and GGUF to GGUF. PEFT to GGUF is a different question: it renames tensors and writes a differently-shaped metadata block, which is format *translation* of names MLMF already maps. It is in scope only as far as the name tables go, and any case where the value of a field would have to be computed (the converter's alpha handling) is out. **Open question for CireSnave, section 6.**
4. **Sabotage** each comparison (flip a byte, drop a key) and see it go red, with the mutation verified applied, per `CLAUDE.md` section 4.
5. **Corpus caveat, stated up front:** the GGUF corpus described in `CLAUDE.md` section 5 holds models and vocab-only files, and no adapter is recorded in it, so step 1 below fetches a small public set (PEFT-saved adapters of varying rank, one with `rank_pattern`, one with a string `target_modules`, one with `use_rslora`, one embedding adapter; llama.cpp's converter output for at least two) and records them in `corpus-provenance.tsv` (S-35). A result that does not exercise those shapes must say so.

## 5. Steps

1. **Measure** (own PR, tests and data only): the tensor-name prefix, shapes and dtypes of PEFT adapters, and the converter's GGUF output for the same adapters, over the set in section 4.5. Record what each varies.
2. **PEFT read**: `adapter_config.json` into `Option` fields, `adapter_model.safetensors` into pairs and orphans.
3. **GGUF read**: the `adapter.*` keys and the `.lora_a` / `.lora_b` pairs, rank from shape.
4. **Writers** for both forms, tensors arriving already encoded.
5. **Round trip** as in section 4.

Each is its own PR at its own version (the PM allocates).

## 6. Questions needing a ruling

1. **PEFT to GGUF (and back) as a writer.** Section 4 item 3: MLMF renames tensors it already maps; it must not compute `alpha`. Is name-mapping translation in scope, or only same-form round trips? Recommend same-form first, translation only after the measurement shows what the converter does beyond renaming.
2. **A `scale` accessor.** Recommend none (section 2.1).

## 7. What this note does not claim

It states what the two public sources declare. It does not claim PEFT's saved tensor prefix, the A/B shapes, or the GGUF dimension order; those are step 1's output. It claims nothing about how any consumer applies an adapter.
