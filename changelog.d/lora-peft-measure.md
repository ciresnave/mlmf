kind: LoRA adapter measurements, PEFT variants (S-65 step 2, data only)
---
### Added

- `crates/mlmf-test-support/data/lora-measurements.tsv` now also measures five more real adapters (two `use_rslora`, two embedding adapters, one Gemma2) and the `adapter_config.json` of each, with revisions in `crates/mlmf-test-support/data/corpus-provenance.tsv`. `tools/lora_measure.py` prints a config's keys with Python repr.
- Measured: `use_rslora` is the string `'True'` in one real file and the boolean `True` in another; an embedding adapter stores `lora_embedding_A` `[r, vocab]` and `lora_embedding_B` `[dim, r]` with no `.weight` suffix, beside a full `embed_tokens.base_layer.weight`, in BF16 while the adapter halves are F32 in one file.
- NOT FOUND after searching the Hub: a non-empty `rank_pattern`, a string `target_modules`. No row claims them.
