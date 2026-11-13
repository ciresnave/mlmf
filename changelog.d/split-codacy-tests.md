kind: three LoRA conformance tests split into helpers
---
### Changed

- Tests only, no behaviour change: `synthetic_image` (`lora_readback`), `compare` (`lora_gguf_roundtrip`) and `real_adapter_configs_are_written_back_as_read` (`lora_config_roundtrip`) are split into small helpers, the three functions Codacy flagged on #239, #240 and #241. Same assertions and error messages; the three test files give the same results as before (9, 6 and 1 tests pass; 7 of 7 adapter files with 1423 tensors, 7 of 7 configs with 175 top-level keys).
