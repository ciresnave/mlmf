kind: docs: patent re-sweep of 0.8.x-0.12.0, Part F (research, not counsel)
---
### Changed

- `docs/PATENT-SWEEP.md` Part F (ROADMAP S-67): a lane read of the independent claims of six documents surfaced by eight searches, against ten ideas that 0.8.x to 0.12.0 added (`mlmf-mlx`, `mlmf-hf-config`, `mlmf-lora`, ranged and split-GGUF reads, Hub listing and hash verification, token and chat-template rows, and the `mlmf-stablehlo` attribute, VHLO and builtin constant readers). Every item reads "outside as read" or "not read"; none is "clear". Seven of the ten ideas have no claim read against them, and the record says so.
- Not covered and still owed: `mlmf-gptq`, `mlmf-exl2`, `mlmf-hf-tokenizer` and the pre-0.8 `mlmf-meta` / `mlmf-hf-layout`, which no earlier record held. The "re-sweep owed" line was replaced with that statement, not removed.
- Limits: a general web search, not a patent database; claim text is Google Patents' machine conversion; Google returned HTTP 503 on claim fetches for about 15 minutes. No code changed.
