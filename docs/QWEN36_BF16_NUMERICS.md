# Native Qwen BF16 numerical review

The opt-in adapter now uses cuBLAS BF16 dense projection output, FP32 accumulation
with reduced-precision reduction disabled, fixed vector RMS/GDN Q/K reductions,
and CUDA FP32 routing normalization followed by the official BF16 weight rounding.
Expert weight bytes remain official BF16. The original Qwen-specific convolution,
gated normalization and expert activation/output rounding boundaries are retained.
The unused FP32 dense output buffer was removed; new router buffers/transfers
are included in memory/traffic accounting. The independent branch still starts
at Strata 46c152a1310f77cd2f0ce2c7cd213962df14fe83.

This is not complete HF numerical parity. A fresh full Hello prefill has logits
relative RMS 0.040057 (max absolute 1.375, mean absolute 0.072737), versus 0.032094
in the historical build. More faithful individual operation boundaries do not
guarantee closer logits after 40 routed layers. The first expert-set difference
remains layer 4. Do not call this an overall logits accuracy improvement.

The selected layer-3 test retains a routing-weight failure: identical expert sets,
maximum two BF16 ULPs and relative RMS 0.004426 against the unchanged 0.004 bound.
All tensor and routing comparisons, including failures, remain in the run logs.
Native complete-model causal state replay passes, but tests native persistence
and chunking rather than independent HF generation parity.

Use `tools/qwen35/check_bf16_forward.py` with a one-output-token native trace
and matching state directory. It streams one dense layer and one expert at a
time, never a full HF model bank. The Windows build, download, index and run
commands remain in [the adapter guide](QWEN36_BF16_WINDOWS.md).

NVFP4 is excluded from this BF16 comparison. Static inspection found a generic
Strata NVFP4 expert kernel, but did not verify a complete loader for the specific
Qwen3.6 HF NVFP4 checkpoint (mixed FP8 dense and projection-scale metadata).
No inference or speed comparison from another framework is substituted.
