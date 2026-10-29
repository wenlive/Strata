# BF16 numerical review on Windows

This opt-in native adapter still fails the complete HF numerical oracle. Do not
interpret successful generation, exact speculative acceptance, or native state
replay as independent HF parity. Historical measurements remain in their original
files; the new review uses fresh runs and executable hashes.

The review changes dense projections to cuBLAS BF16 output with FP32 accumulation
and reduced-precision reduction disabled. It matches the HF linear output
boundary. Expert down projections intentionally retain FP32 output. RMS and GDN
Q/K normalization use a fixed vector reduction, preserving the same geometry
for causal replay. The unused FP32 dense output buffer was removed.

Given identical saved BF16 router logits, the old CPU weight calculation differed
from HF CUDA in 3180/5120 FP32 values (maximum 1.19209e-7). A CUDA softmax and
top-eight normalization now calculates the weights. Stable expert selection and
the official dynamic preview coreset are retained; cache residency never changes
routing. Forty real causal layers and eight synthetic cases matched HF CUDA
weights exactly in the initial test. The new control transfers and device buffers
are included in memory/traffic accounting. This does not remove upstream router
logit differences.

For the complete 16-token Hello prefill, relative RMS logits error changed from
0.148641 to 0.117060; last-position KL changed from 0.002358 to 0.000372.
The first *expert-set* difference is layer 4; earlier slot permutations at ties
must not be described as different expert sets. The new run differs at one
intermediate prefill argmax position. Its last-position argmax agrees with HF.
The original tolerance remains 0.004; the whole-model check fails.

Independent HF runs with only GEMM row padding changed have relative RMS logits
difference 0.090823 and three intermediate argmax differences. This demonstrates
sensitivity to BF16 rounding and routed expert selection. It does not prove the
remaining native differences are harmless or entirely explained by rounding.
Replacing HF routing with saved native logits is diagnostic only and cannot
qualify a passing reference result.

New `check_preview_trace.py` streams the official full 40-layer preview while
conditioning both paths on the same saved native committed history. Hello width
32 has logits RMS 0.060210 with all argmax positions equal. Sky width 32 has RMS
0.061279 and seven differing argmax positions. These checks fail the strict bound
and are not an end-to-end HF generate comparison. Preview isolates history,
right convolution, bidirectional block computation and dynamic coreset; the
native committed-state replay tests separately check persistence.

Reproduce diagnostics after building the targets in
[the Windows guide](CLOVER_BF16_WINDOWS.md):

```powershell
python tools/clover/check_forward_trace.py --index MODEL_INDEX --config RUN/config.json --trace RUN/trace/repeat0/0-prefill --state RUN/state --output NEW_ORACLE --isolate-each-layer
python tools/clover/check_preview_trace.py --index MODEL_INDEX --trace PREVIEW_RUN/trace/repeat0/1-denoise_preview --state PREVIEW_RUN/state --output NEW_PREVIEW_ORACLE --isolate-each-layer
python tools/clover/test_router_device.py --exe BUILD/strata-clover-kernel-test.exe --trace RUN/trace/repeat0/0-prefill --output NEW_ROUTER_TEST
```

Use a one-output-token run when comparing a prefill trace against its state
directory: a later generation state contains additional tokens. Diagnostic-only
padding and native-router options are explicitly marked in JSON and cannot pass
as unmodified reference checks. Synthetic tensors and forced threshold-zero
commit cases are correctness tests, never throughput measurements.
