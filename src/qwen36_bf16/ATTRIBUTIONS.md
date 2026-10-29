# Qwen BF16 comparison path

This independent branch starts from Strata
`46c152a1310f77cd2f0ce2c7cd213962df14fe83`. Its existing Qwen3.6 engine,
`src/qwen35/q36_engine.cu`, is the family/shape and execution reference.
`platform::DirectFile` and `prefill::Gemm` are reused without changing the
normal quantized engine. Their Strata licenses remain in force.

Bounded BF16 storage/execution and causal CUDA building blocks were reused
from the experimental Clover branch at `803a6afaff6688fa05f084fb6ced070ddc306a73`.
This is source reuse, not a merge of that branch or use of its preview runtime.
Qwen-specific rounding boundaries were changed to follow the installed
Transformers 5.21.0 `modeling_qwen3_5_moe.py` implementation. Its source hash,
reference tests and immutable model revision are recorded in the task evidence.

The Qwen model and Transformers implementation carry Apache-2.0 notices,
Copyright 2025 The Qwen Team and The HuggingFace Inc. team; the checkpoint
retains its official LICENSE (2026 Alibaba Cloud). The reused Clover formulas
also derive from the Apache-2.0 DiffuSpace HF implementation. The accompanying
Apache-2.0 license is preserved. The JSON header is nlohmann/json from Strata's
pinned llama.cpp vendor tree (MIT, Niels Lohmann and contributors).
