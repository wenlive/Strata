# Experimental Clover BF16 adapter on Windows

This work is opt-in (`STRATA_BUILD_CLOVER=ON`). Current validated capability is
BF16 expert loading and numerical execution. The 40-layer native driver and
greedy exact CLI compile. **Complete Clover generation has not been validated
yet.** Synthetic attention/GDN/router and exact decisions have passed; actual
full-model execution remains a separate acceptance gate.

The adapter uses Strata's `platform::DirectFile` and `prefill::Gemm`. It does
not use Qwen3.6's all-expert host arena or the quantized native expert kernels.
The regular engine targets and their options remain separate.

## Sources

The tokenizer wrapper records native process working-set/commit peaks and
samples NVML whole-device memory every 100 ms into `telemetry.json`. The GPU
peak includes desktop/other processes and is a sampled device peak, not a
per-process GPU allocation counter. Known buffer allocations and native
phase-boundary memory observations remain separate. Optional tool dependency
pins are in `tools/clover/requirements.txt`; install CUDA Torch separately as
that file describes, then install those dependencies. Physical SSD bytes are
not inferred from any of these counters.

- Strata base: `46c152a1310f77cd2f0ce2c7cd213962df14fe83`.
- [HF checkpoint and implementation](https://huggingface.co/Diffuspace/Clover-35B-A3B-Preview/tree/4b0402b65e523d0183a14c431955438714cb16ff): `4b0402b65e523d0183a14c431955438714cb16ff`.
- [Clover project](https://github.com/DiffuSpace/Clover-35B-A3B-Preview/tree/e5a130a48e4a6d6c1d05c1c2338bce555d674044): `e5a130a48e4a6d6c1d05c1c2338bce555d674044`.
- [SGLang PR 46927](https://github.com/sgl-project/sglang/pull/46927), inspected head `058939af24d5cdc44bb4b8852d0ddfab555ed71e`.
- The updated Clover README links [llama.cpp PR 31306](https://github.com/ggml-org/llama.cpp/pull/31306); observed head `4838ffb0c8ade92f34d9157deea38d92758ab824`. This is a reference, not the engine used for this adapter.

## Checkpoint inventory

The official index lists 28 safetensors files: 26 main shards plus the right
convolution and softmax scalar files. Each parameter appears once, including
all parameters in both extra files. Every tensor is BF16. Header inspection
finds 34,661,350,848 parameters / 69,322,701,696 data bytes (64.561797 GiB).

| Category | BF16 bytes |
| --- | ---: |
| Routed experts | 64,424,509,440 |
| Shared experts and shared gates | 251,822,080 |
| Embedding and output head | 2,034,237,440 |
| Base GDN parameters | 2,023,107,840 |
| Full attention parameters | 545,269,760 |
| Layer/final norms and routers | 42,274,816 |
| Clover GDN right convolution and softmax scalars | 1,480,320 |

One routed expert is 6,291,456 bytes. Its gate/up slice is contiguous row-major
`[1024,2048]`, with gate first, followed by up; down is `[2048,512]`. Offsets are
absolute uint64 byte offsets in the original safetensors shards. No weight
repack is needed. Nonexpert tensors account for 4.561797 GiB before states,
activations, CUDA libraries, workspace, caches and staging.

## Download and index

Use an immutable revision. The downloader keeps unfinished files under
`.incomplete`, reads in 1 MiB chunks per worker, resumes using HTTP ranges and
checks file size and LFS SHA256 before renaming. It leaves a verified manifest
only after every file succeeds. It checks free disk space with 200 GiB reserve.

`--backend hf` selects installed HF Hub/Xet transport with the same full
SHA256 checks. HTTP is the bounded streaming fallback. Both transports keep
their partial files for resuming; their partial-file formats are different.

```powershell
python tools/clover/download.py --destination C:/workspace/models/inputs/clover-35b-a3b-bf16/4b0402b65e523d0183a14c431955438714cb16ff
python tools/clover/index.py --model C:/workspace/models/inputs/clover-35b-a3b-bf16/4b0402b65e523d0183a14c431955438714cb16ff --output C:/workspace/models/bundles/clover-bf16/4b0402b6/index.json
```

The indexer streams checksums, checks exact shapes and BF16 dtype, rejects
duplicates, missing tensors, overlaps and truncated files, and atomically
publishes the index. It never loads an expert bank. `--audit-headers FILE`
performs a metadata-only audit and produces `complete:false`; the native
loader rejects such an index for execution.

## Build

Enter a VS2022 x64 native tools environment with CMake, Ninja and CUDA 13.2 on
PATH. CUDA 13.2 DLLs are in `bin/x64`; include this directory too. The measured
RTX 5070 Ti reports compute capability 12.0. The following example explicitly
targets 120; probe a different machine before selecting its architecture.

```powershell
cmake -S . -B C:/workspace/builds/T-20261028-clover-bf16/baseline -G Ninja -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120 -DSTRATA_BUILD_TESTS=OFF -DSTRATA_BUILD_CLOVER=ON -DSTRATA_GGML_DIR=C:/workspace/repos/reference/llama-cpp-3cf03257
cmake --build C:/workspace/builds/T-20261028-clover-bf16/baseline --target strata-clover-expert-test -j 3
```

`STRATA_GGML_DIR` is an existing Strata option. This path contains the exact
Strata-pinned llama.cpp commit `3cf03257f219afbe7334045ff7c6a06ac68c627d` from
its GitHub source archive; it avoids a full-history Git clone. The adapter
also uses the MIT nlohmann JSON header from that pinned dependency.

## Storage and numerical contract

RAM cache, VRAM cache and staging budgets are independent byte counts. Slots
round down to whole experts; there is no full host expert bank. GPU residents
do not require permanent RAM copies. RAM slots use pageable memory; pinned
scratch is a distinct staging allocation. A zero VRAM cache uses one expert
GPU staging slot charged to the staging budget. The host read scratch is
4,198,400 bytes for Clover, including alignment; zero-cache total staging is
10,489,856 bytes. The native loader reports budgets and actual allocations.

`DirectFile` uses Windows unbuffered, overlapped, aligned reads. Source slices
can be unaligned; the bounded scratch includes alignment overhead. No model
file is memory mapped. Logical payload bytes, aligned requested/completed
bytes and H2D bytes are distinct counters. Device physical SSD traffic is
currently **unmeasured**, including for these unbuffered completions.

All transfers and expert work use one CUDA stream. Each GPU slot records an
event after its last computation. Eviction waits for that event. Host staging
waits for its H2D event before being overwritten. RAM cache hits also pass
through pinned staging. Prefetch and transfer overlap are not enabled yet.

Experts sharing tokens are grouped per expert. Gate/up GEMM has BF16 inputs,
FP32 accumulation and BF16 output rounding before FP32 SiLU/multiply. The
intermediate activation is BF16. Down GEMM keeps FP32 output through route
weighting; each contribution rounds to BF16, then accumulates in FP32 in
ascending expert id before the final BF16 output. Strata's GEMM has a separate
32 MiB workspace; activation workspaces are reported separately from expert
cache/staging budgets.

## Expert validation

```powershell
python tools/clover/test_experts.py --exe C:/workspace/builds/T-20261028-clover-bf16/baseline/strata-clover-expert-test.exe --output C:/workspace/runs/T-20261028-clover-bf16/<new-run>/results
```

Use CUDA PyTorch for this oracle. It is used only for tests, not as an inference
engine. The test executable takes exactly `JOB.json RESULT.json`; jobs are
generated by the test script. Tests cover the full 2048/512 expert dimensions,
1/8/16/32 rows, weighted experts, independent cache budgets, forced eviction,
byte-preserving source relocation to offsets above 4 GiB, and explicit errors
for incomplete indices, dtype, shape, truncation, bad expert offsets and
insufficient staging. Synthetic results do not establish complete-model or
state-machine correctness.

Expanded 24-case run on 2026-10-28 passed:
all 96 weighted MoE outputs matched the CUDA PyTorch BF16 oracle exactly
(maximum and mean absolute error zero); cache changes preserved outputs
exactly. All six corrupt/budget cases were rejected. The sparse source fixture
was 4,345,299,504 bytes with its first tensor above 4 GiB. Independent RAM/VRAM
budget coverage includes actual hits in both tiers. Evidence is in the task's
timestamped runs.

Selected official layer-0 expert weights also passed: 1/8/32 rows, 8/58/159
unique experts, four budget combinations and two passes each. All 24 weighted
outputs matched the CUDA Torch oracle exactly. These tests use synthetic BF16
activations and original SHA256-verified checkpoint slices. They do not prove
complete layer correctness.

Synthetic attention/GDN/router tests passed 154 comparisons against functions
extracted from the pinned HF source. Convolution outputs matched exactly;
recurrent state maximum absolute error was 1.49e-8; attention maximum error was
0.00390625 with maximum relative RMS 0.0023293. Router IDs and coreset sets
matched, with maximum route weight error 2.98e-8. Preview and read-only verify
preserved state. Exact decision logic passed 126 cases against pinned SGLang
utilities, including full/partial/zero acceptance, bonus, EOS and truncation.

Full-model download, actual single-layer comparisons, complete-state replay
validation and generation benchmarks remain outstanding. No token/s or
complete Clover support is claimed here.

## Native CLI (compiled, full-model validation pending)

Build with `cmake --build <build> --target strata-clover`. Use
`strata-clover --config RUN.json` or the tokenizer-only wrapper below. The
Python entry performs no model computation.

```powershell
python tools/clover/chat.py --exe C:/workspace/builds/T-20261028-clover-bf16/baseline/strata-clover.exe --index C:/workspace/models/bundles/clover-bf16/4b0402b6/index.json --prompt "Say hello briefly." --output C:/workspace/runs/T-20261028-clover-bf16/<new-run> --ram-cache-mib 4096 --vram-cache-mib 6144 --staging-mib 16 --context-tokens 256 --draft-width 8 --accept-rule exact --max-new-tokens 8 --audit-state
```

This implemented command requires a complete verified index; it has not yet
passed full-model acceptance. `--thinking` enables the official thinking chat
template; the default explicitly disables thinking. Sampling is greedy with
the checkpoint's response-penalty defaults unless overridden. `--repeats 2`
retains application expert caches between identical requests; this is labeled
warm application caching and makes no cold-SSD claim. `--dump-state` records
committed KV/conv/recurrent state for comparison. `--help` is authoritative.

Native JSON example (token IDs here are an example only):

```json
{
  "index": "C:/workspace/models/bundles/clover-bf16/4b0402b6/index.json",
  "input_tokens": [248045, 846, 198, 44240, 23066, 25899, 13, 248046, 198, 248045, 74455, 198, 248068, 271, 248069, 271],
  "mode": "selfspec",
  "max_new_tokens": 8,
  "ram_cache_mib": 4096,
  "vram_cache_mib": 6144,
  "staging_mib": 16,
  "context_tokens": 256,
  "draft_width": 8,
  "accept_rule": "exact",
  "audit_state": true,
  "repeats": 2
}
```

Preview and verification read persistent history. Accepted inputs are then
replayed causally; logs charge `commit_replay` as an additional forward. The
boundary bonus is emitted once and stays uncommitted until the next round.
EOS/length-truncated output replays only its returned accepted draft prefix.
Exact acceptance follows SGLang's verify-argmax rule, including its response
penalty snapshot. It does not assert equivalence to HF block denoising or to
a different numerical implementation.

## Actual layer comparison and diagnostic tools

With HF FP32 math SDPA selected explicitly, 190 comparisons across official
layer-0 GDN and layer-3 full-attention weights passed the same predefined BF16
bounds. Routed expert IDs matched in every step; preview/verify state hashes
were unchanged. First causal full-attention output and KV were byte-identical.
Later attention steps retained small differences (maximum relative RMS
0.002264). The initial default fused-SDPA comparison failed: some 16-token
steps had up to 10 differing route slots. Q/K and KV matched; attention output
was the first difference. That failed run is preserved. Native attention keeps
FP32 probabilities/value accumulation and was not changed for the math-oracle
pass. Complete-model argmax/routing comparisons remain outstanding.

`strata-clover-layer-test` accepts only an explicit selected-layer test index;
the generation CLI rejects that scope. The test does not substitute a reduced
model for complete Clover inference.

`--trace` in `chat.py` records per-layer correctness tensors with additional
synchronization and disk writes. Keep it off for performance measurements.
`--prefill-chunk` selects 1..32 causal rows per call (default 32), independently
of draft width and checkpoint block size. Native JSON fields are
`trace_directory` and `prefill_chunk`. The native `--help` includes both.

Prepared validation tools (full-model runs still pending):

- `check_forward_trace.py`: layer-streamed HF math oracle for a native short
  40-layer causal trace; one dense layer and selected expert at a time, with a
  streamed BF16 output head.
- `check_committed_state.py`: causally replay the exact committed token sequence
  and compare every KV/conv/SSM state; pending bonus is excluded.
- `report_run.py`: aggregate phase times, expert unions, hits, logical reads,
  H2D and memory. GPU memory is labeled as phase-boundary observations; physical
  SSD bytes remain unmeasured.
