kind: GGUF LoRA adapter written back as read, measured (S-65 step 4, GGUF form)
---
### Added

- `crates/mlmf-conformance/tests/lora_gguf_roundtrip.rs`: CireSnave's ruling 421 (same form, tensors already encoded, no `scale` accessor) needs no adapter-specific GGUF writer, because an adapter is a GGUF file and `mlmf_gguf::encode_gguf` already carries every key and tensor. This measures it on the one real GGUF adapter in the LoRA corpus: read, write back as read, read again, and every metadata key and value, every tensor's type, dims and payload bytes, `GgufAdapterInfo` and the A/B pairing are equal; no key is added and `adapter.lora.alpha` stays the declared 16.0. Each comparison is sabotaged (flipped byte, added key, changed alpha, dropped tensor, changed dim) and goes red with the mutation asserted applied. Limit: one adapter file, F16 tensors only. Announces `SKIPPED` without the corpus and fails under `MLMF_CORPUS_REQUIRED`.
