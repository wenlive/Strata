# Experimental Clover BF16 on native Windows

The opt-in `strata-clover` CLI has generated short text with all 40 layers and
the complete official BF16 checkpoint on an RTX 5070 Ti 16 GB. It implements
causal inference and SGLang-style self-speculation with exact acceptance.
Experts stream from SSD through bounded RAM/VRAM caches. No quantization,
permanent expert pruning, changed router or all-expert host bank is used.

**Full cross-engine numerical validation is unfinished.** Expert tests,
selected-layer tests and complete native state-replay tests pass. A streamed
40-layer HF comparison exceeds its existing tensor-error bounds, despite the
same final-position argmax in the tested prompt. This branch is experimental;
it is not a claim of complete validated Clover support.

## Pinned sources and reuse

- Strata base: `46c152a1310f77cd2f0ce2c7cd213962df14fe83`.
- [Official HF weights and source](https://huggingface.co/Diffuspace/Clover-35B-A3B-Preview/tree/4b0402b65e523d0183a14c431955438714cb16ff): `4b0402b65e523d0183a14c431955438714cb16ff`.
- [Clover project](https://github.com/DiffuSpace/Clover-35B-A3B-Preview/tree/e5a130a48e4a6d6c1d05c1c2338bce555d674044): `e5a130a48e4a6d6c1d05c1c2338bce555d674044`.
- [SGLang PR 46927](https://github.com/sgl-project/sglang/pull/46927): inspected head `058939af24d5cdc44bb4b8852d0ddfab555ed71e`, including model, worker/draft/config/utils, attention backend and convolution/GDN/coreset kernels.
- The project's newer [llama.cpp PR 31306](https://github.com/ggml-org/llama.cpp/pull/31306), head `4838ffb0c8ade92f34d9157deea38d92758ab824`, was also inspected. It is not the runtime used here.
- Strata's pinned llama.cpp dependency: `3cf03257f219afbe7334045ff7c6a06ac68c627d`.

The adapter reuses Strata `platform::DirectFile` and `prefill::Gemm` and builds
inside its CMake project. Python handles downloading, indexing, tokenization,
reference calculations and measurement; it performs no production model forward.
At this base, `src/qwen35/q36_engine.cu` creates a complete host expert arena
before placing GPU experts. Its `native_expert_supported()` check in
`src/kernels/cuda/iq_kernels.cu` accepts specific quantized types, not BF16.
`FileExpertSource` has useful file/cache machinery but is not automatically the
Qwen family loader. Clover uses a separate bounded slice store over DirectFile.
It does not enter the MTP/CPU expert paths or change their `MAXT` constants.

## Machine and capacity

Measured on Windows 11 Pro 10.0.26200: Ryzen 7 9700X (8 cores/16 threads),
two 32 GiB DIMMs (64 GiB installed, 61.62 GiB OS-visible), RTX 5070 Ti,
16,303 MiB VRAM, queried compute capability **12.0**, driver 610.88.
Commit limit: 70,318,911,488 bytes. C: is NTFS on a Lexar ARES PRO 2 TB NVMe
with over 1 TiB free. No WSL or SGLang installation was used.
Build tools: CUDA Toolkit 13.2.78, VS2022 MSVC 19.44.35229, CMake 3.31.12,
Ninja 1.13.2, Python 3.12.10. Test dependencies are pinned in
[`tools/clover/requirements.txt`](../tools/clover/requirements.txt), with CUDA
Torch 2.11.0+cu130 installed separately. The original Strata target and CUDA
device smoke test built successfully before the adapter was added.

Config, every safetensors header and the official weight map give:

| BF16 weights | Bytes | GiB |
| --- | ---: | ---: |
| Routed experts | 64,424,509,440 | 60.0000 |
| Shared experts and their gates | 251,822,080 | 0.2345 |
| Embedding and LM head | 2,034,237,440 | 1.8945 |
| Base GDN | 2,023,107,840 | 1.8842 |
| Full attention | 545,269,760 | 0.5078 |
| Norms and routers | 42,274,816 | 0.0394 |
| Clover GDN extras | 1,480,320 | 0.0014 |
| **Total: 34,661,350,848 parameters** | **69,322,701,696** | **64.5618** |

There are 40 layers (30 GDN, 10 full attention), 256 routed experts per layer,
top-8 routing, hidden size 2048 and expert intermediate size 512. One expert
has 4 MiB gate/up `[1024,2048]` plus 2 MiB down `[2048,512]`, or 6 MiB;
routed experts occupy 1.5 GiB per layer. Shared experts are separate.
The 28 safetensors files comprise 26 main shards and two GDN-extra files for
right convolution and softmax parameters. Both extras are already in the
official index and are loaded once. All 28 passed full SHA256 verification.
The table excludes headers, tokenizer files, runtime state and workspaces.

## Prepare weights without a full host load

These commands reuse existing verified files or download missing ones. The
weights need approximately 69.3 GB plus small files. The downloader additionally
reserves 200 GiB free system disk space. Direct indexing avoids a second full
expert pack.

```powershell
python tools/clover/download.py --destination C:/workspace/models/inputs/clover-35b-a3b-bf16/4b0402b65e523d0183a14c431955438714cb16ff
python tools/clover/index.py --model C:/workspace/models/inputs/clover-35b-a3b-bf16/4b0402b65e523d0183a14c431955438714cb16ff --output C:/workspace/models/bundles/clover-bf16/4b0402b6/index.json
```

HTTP uses bounded 1 MiB chunks per worker, resumable `.incomplete` files,
size/SHA256 checks and final rename. `--backend hf` selects HF Hub/Xet with
the same verification; its partial-file format is separate. The indexer
streams checksums and validates shapes/dtypes, coverage, overlaps, revision
and config/tokenizer/source provenance. It atomically writes `complete:true`
only after success. `--audit-headers FILE` produces `complete:false`, which
the executable rejects. Keep source files read-only after indexing; changing
weights requires reindexing.
Offsets are 64-bit. Each expert entry records layer/expert, gate/up/down file,
offset, length, BF16 dtype and row-major shape. No weight bytes are rearranged
or rounded. Missing, truncated, wrong-shape/dtype and incomplete inputs fail.

## Build and run

Use a VS2022 x64 native tools shell, with CUDA, CMake and Ninja on PATH.
CUDA 13.2 DLLs also require `%CUDA_PATH%/bin/x64` on PATH. On this machine,
`CMAKE_CUDA_ARCHITECTURES=120` follows the queried capability.
Extract the pinned llama.cpp archive into the directory passed to
`STRATA_GGML_DIR`. The archive used here has SHA256
`c076d7534afa0e5d0ec2a0d425b11e791c16f3de0d727221aea071cef156a280`.
Using the archive avoids a full-history dependency clone.

```powershell
curl.exe -L https://github.com/ggml-org/llama.cpp/archive/3cf03257f219afbe7334045ff7c6a06ac68c627d.tar.gz -o llama-pinned.tar.gz
# Extract to C:/workspace/repos/reference/llama-cpp-3cf03257 first.
cmake -S . -B C:/workspace/builds/T-20261028-clover-bf16/baseline -G Ninja -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120 -DSTRATA_BUILD_TESTS=OFF -DSTRATA_BUILD_CLOVER=ON -DSTRATA_GGML_DIR=C:/workspace/repos/reference/llama-cpp-3cf03257
cmake --build C:/workspace/builds/T-20261028-clover-bf16/baseline --target strata-clover strata-clover-expert-test strata-clover-kernel-test strata-clover-layer-test strata-clover-state-test -j 3
python tools/clover/chat.py --exe C:/workspace/builds/T-20261028-clover-bf16/baseline/strata-clover.exe --index C:/workspace/models/bundles/clover-bf16/4b0402b6/index.json --prompt "Count from 1 to 10." --output C:/workspace/runs/T-20261028-clover-bf16/new-request --mode selfspec --accept-rule exact --ram-cache-mib 2048 --vram-cache-mib 6144 --staging-mib 16 --context-tokens 256 --draft-width 8 --max-new-tokens 16
```

`--output` must be new. It receives config, source/binary provenance, native
log, output IDs/text and memory telemetry. The official chat template explicitly
disables thinking by default; `--thinking` enables it. Tokenizer, mask 151669,
EOS 248046/248044 and checkpoint penalty defaults are retained. Sampling is
greedy only. Presence/frequency/repetition overrides appear in both help texts.
Native invocation is `strata-clover --config RUN.json`. Example:

```json
{
  "index": "C:/workspace/models/bundles/clover-bf16/4b0402b6/index.json",
  "input_tokens": [248045, 846, 198, 44240, 23066, 25899, 13, 248046, 198, 248045, 74455, 198, 248068, 271, 248069, 271],
  "mode": "selfspec",
  "accept_rule": "exact",
  "max_new_tokens": 16,
  "context_tokens": 256,
  "prefill_chunk": 32,
  "draft_width": 8,
  "ram_cache_mib": 2048,
  "vram_cache_mib": 6144,
  "staging_mib": 16,
  "repeats": 1,
  "audit_state": true,
  "result_file": "result.json"
}
```

`mode=causal` generates causally; `mode=forward` only commits input tokens.
`prefill_chunk` is 1..32. Context capacity is at most 8192, but only short
contexts have been validated. Draft widths are 2/4/8/16/32; checkpoint block
size stays 32. `dump_final_state` and `trace_directory` must be new directories;
the wrapper exposes them as `--dump-state` and `--trace`.

## Execution and budget contract

Non-expert BF16 weights (4.5618 GiB) stay on GPU. RAM expert cache, VRAM expert
cache and combined staging have independent budgets; cache sizes round down
to whole experts. RAM/VRAM are exclusive LRUs. GPU eviction copies an expert
to bounded pinned staging and then RAM; a RAM hit promotes it and frees its
RAM slot. With GPU cache zero, one GPU staging expert serves a bounded RAM
LRU. With RAM zero, misses read through host staging without retaining a copy.
Host read staging is 4,198,400 bytes. A 6,291,456-byte pinned eviction slot is
needed when both caches are enabled, or a GPU staging slot of that size when
GPU cache is zero. Their combined staging is 10,489,856 bytes, within 16 MiB.
There is no full host expert bank. CPU metadata, logits, CUDA/WDDM backing,
activations and states are additional process memory, measured separately.
At context 256: state 69,632,000 bytes, model workspace 51,439,808 bytes,
expert activations 819,456 bytes, plus two 32 MiB GEMM workspaces.

DirectFile uses native Windows unbuffered aligned reads; nothing is mmap'd.
One CUDA stream orders work. Slot events protect prior GPU computation,
a scratch event protects H2D source reuse, and completed D2H protects the
victim before overwriting GPU memory. RAM promotion consumes all source bytes
before freeing its slot. No background prefetch or unbounded transfer queue.
Tokens are grouped by expert before computation. BF16 inputs/weights use
Strata cuBLAS GEMM with FP32 accumulation. Gate/up rounds to BF16 before FP32
SiLU/multiply; hidden rounds to BF16; down output stays FP32 through route
weighting. Each contribution rounds to BF16, accumulates in FP32 in expert-ID
order, and rounds at MoE output. All GEMMs compute 32 zero-padded rows; only
actual rows enter attention, GDN, routing and state. This fixes a measured
batch-geometry replay discrepancy without changing checkpoint block metadata.
It adds 256 KiB dense input workspace.

## Preview and generation semantics

HF `generate()` iteratively denoises the whole block, including positions
already determined. That algorithm is not implemented here. This CLI follows
SGLang: a full-model parallel preview, causal verify, exact contiguous-prefix
acceptance and boundary bonus. There is no additional small draft model,
threshold acceptance or lightweight MTP substitute.
Full attention uses committed causal history and bidirectional preview blocks.
GDN preview uses right convolution, block-local memory and committed recurrent
history. History scaling retains trained block size 32 even at runtime width 8.
Preview/verify leave persistent KV/conv/SSM unchanged. Accepted inputs are
**replayed causally as a third forward**, charged separately. Only the returned
accepted prefix is committed. A bonus is emitted once and stays pending until
its input is committed later. EOS and length truncation are tested.
Official preview coreset aggregates tokens' top-8 routing mass and selects the
stable expert prefix covering 0.9 of that mass, then masks/reroutes. It varies
by layer/input/round. Causal verify/replay do not use it. No additional routing
approximation is enabled for cache hits.

## Validation and outstanding differences

Numeric checks report maximum/mean error, relative RMS and routing changes.
Cache-dependent outputs and native state replay require exact equality.
Full-model failures are not hidden by widening tolerances.

- Synthetic experts: 24 cache/row cases, 96 outputs, offsets above 4 GiB,
  byte-preserving reads, both cache tiers, eviction and six invalid inputs.
  Passed after exclusive caching; max absolute error 0.0078125, max relative
  RMS 0.00051882 versus variable-row CUDA Torch. All cache outputs were exactly
  equal and no expert was duplicated between cache tiers.
- Official layer-0 experts: 1/8/32 rows, 8/58/159 unique experts, four budgets,
  two passes each. Passed; max absolute error 0.00006103515625. Cache changes
  preserved outputs exactly.
- Synthetic attention/GDN/router: 154 comparisons passed against pinned HF
  functions. Exact decision logic: 126 cases passed against SGLang utilities.
- Official layer-0 GDN/layer-3 attention: 190 comparisons passed with math SDPA
  and unchanged bounds; no route-ID/coreset differences. Initial fused-SDPA
  comparison failed (up to 10 route slots) and remains recorded.
- Full 40-layer controlled acceptance: 7, 3 and 0 accepted drafts; preview/verify
  were read-only. All 240 state files matched direct causal execution by SHA256.
- Complete model, exclusive RAM 2 GiB / GPU 6 GiB: `Say hello briefly.` produced
  `Hello!` and EOS. All 80 final KV/conv/SSM tensors matched independent one-token
  causal replay with RAM 0 / GPU 1 GiB byte-for-byte.
- Full HF math-SDPA causal comparison on a 16-token prompt: **failed** the 0.004
  layer-output relative RMS bound. Logits max/mean absolute error 8.75/0.9963324,
  relative RMS 0.1486412. Final-position argmax matched (9419). First difference:
  layer-0 input RMSNorm, one of 32768 BF16 values, max 0.0009765625, RMS 0.000004934.
  Layer-0 output RMS 0.0008051; layer 1 then had four differing route slots.
  Its isolated native-input comparison had zero route differences. Small
  rounding differences accumulate and change sparse routing.
- A labeled diagnostic replacing HF linear operations with 32-row FP32-output
  GEMMs still failed (logits RMS 0.1103659). It is not an official-reference pass.
  Isolated all-depth HF comparisons retain intermediate and route differences
  in the measurement JSON; these also prevent a full numerical-support claim.

An earlier variable-GEMM state failure (76/80 tensors) was fixed; its logs are
preserved. Full cross-engine numerical agreement, long-context validation,
stochastic sampling and HF block denoising remain unfinished or unsupported.

Reproduce using new output directories:

```powershell
python tools/clover/test_experts.py --exe <build>/strata-clover-expert-test.exe --output <new-expert-test>
python tools/clover/test_official_experts.py --model <official-model> --exe <build>/strata-clover-expert-test.exe --output <new-official-test>
python tools/clover/test_official_layers.py --model <official-model> --exe <build>/strata-clover-layer-test.exe --output <new-layer-test> --sdpa-backend math
python tools/clover/check_committed_state.py --exe <build>/strata-clover.exe --run <chat-run-with-dump-state> --output <new-state-test>
python tools/clover/check_forward_trace.py --index <index.json> --config <chat-run>/config.json --trace <chat-run>/trace/repeat0/0-prefill --state <chat-run>/state --output <new-hf-test> --isolate-each-layer
python tools/clover/report_run.py --run <chat-run>
```

The controlled-state executable takes `JOB.json NEW_DIRECTORY`; the job has
`index` and `input_tokens`. It uses the complete checkpoint, RAM 0 / GPU 6144 MiB,
and alters only test candidates to cover acceptance branches. Production draft
generation has no candidate overrides. Full-forward oracle diagnostics are
optional `--linear-geometry padded32-fp32` and `--isolate-each-layer`; default
linear geometry leaves HF functions unchanged.

## Measured performance

[Measurement JSON](clover/measurements-20261029.json) records code/binary hashes,
phase totals, I/O, acceptance and memory. Prompt `Count from 1 to 10.`, 16 output
IDs, official greedy penalty defaults, context 256, prefill chunk 32, GPU cache
6144 MiB, staging 16 MiB, audit/trace off. Each process runs two identical requests.
All width-8 budgets returned the same IDs. Text: `1, 2, 3, 4, 5, 6` (the 16-ID
limit ends before 10).

| RAM MiB / width | TTFT first / warm (s) | Steady first / warm (token/s) | Logical reads first / warm (GiB) |
| --- | ---: | ---: | ---: |
| 0 / 8 | 3.460 / 3.435 | 2.038 / 2.024 | 47.637 / 47.637 |
| 2048 / 8 | 4.204 / 4.416 | 1.989 / 1.870 | 30.340 / 30.188 |
| 8192 / 8 | 4.485 / 3.954 | 2.229 / 2.060 | 24.100 / 20.379 |
| 8192 / 16 | 4.489 / 3.957 | 1.788 / 1.754 | 28.307 / 24.902 |
| 8192 / 32 | 4.470 / 4.087 | 1.200 / 1.180 | 43.107 / 40.371 |

TTFT is native first-token availability, excluding model opening/CUDA startup;
the wrapper prints completion after exit and is not a streaming client. Steady
rate excludes the first token and includes preview, verify, acceptance and
**extra commit replay**. First request means empty application expert caches,
not demonstrated cold SSD. Warm means retained application caches. No system
cache settings changed. Larger widths were slower for this short output.
These short runs are not statistical guarantees or extrapolations from other GPUs.
Exclusive caching reduces logical reads but costs D2H: width-8 warm H2D was
47.637 GiB in every RAM configuration, and RAM-enabled D2H also 47.637 GiB.
Extra RAM did not guarantee a faster request. Logs distinguish prefill/preview/
verify/replay, per-layer unique experts, cache counters, logical/aligned reads,
read calls, weight H2D/D2H and control H2D.
Process working set/commit are measured independently of expert budgets. NVML
samples whole-device memory every 100 ms, including desktop/other processes;
CUDA phase-boundary observations and known allocations are separate. Neither
is an exact per-process GPU high-water mark. **Physical SSD bytes and true cold
I/O are unmeasured**; logical ReadFile completions are not physical throughput.

| Width-8 RAM MiB | Process peak working set (MiB) | Process peak commit (GiB) | Sampled whole-device peak (GiB) |
| --- | ---: | ---: | ---: |
| 0 | 488.3 | 11.670 | 11.974 |
| 2048 | 2540.4 | 13.676 | 11.974 |
| 8192 | 8684.2 | 19.707 | 11.974 |

Windows commit includes CUDA/WDDM allocations and is not the resident RAM
expert cache. The measured 64 GiB DIMM capacity was probed, not assumed.
The normal Strata path defaults to `STRATA_BUILD_CLOVER=OFF`. This branch adds
no HTTP service, UI, multi-GPU, concurrency or CPU BF16 executor.

Additional store timers now report host_copy_bytes/host_copy_ms (CPU weight memcpy), ram_staging_wait_ms, and demotion_wait_ms (D2H enqueue plus synchronization). These are host wall-clock sections; do not add GPU copy duration again. Historical logs omit unavailable new counters; they are not reported as measured zeros. Arithmetic and cache policy are unchanged. Follow-up measurements will use the new executable hash.

## Follow-up BF16 comparison

The independent [Qwen BF16 branch comparison](https://github.com/wenlive/Strata/blob/feat/qwen36-bf16-windows/docs/BF16_COMPARISON_WINDOWS.md)
uses the same native storage/GEMM design, explicit greedy penalties 0/0/1,
equal prompts and RAM/VRAM/staging budgets. It records new phase counters,
Nsight analysis, complete Qwen checkpoint generation and remaining HF
numerical failures. This is separate from the earlier default-penalty table.

On the ordinary-text control with RAM/GPU expert caches 8/8 GiB, warm causal
rates were Qwen 4.818 and Clover 4.808 token/s; Clover exact selfspec width 8
was 2.010. Runtime widths 2/4 reached 3.088/2.931 in their warm requests and
matched causal IDs. The trained block size remained 32. For the counting
control, moving GPU cache from 6 to 8 GiB changed replay from zero GPU hits to
all 2,463 hits; exact selfspec rose from 1.994 to 3.853 token/s at RAM 8 GiB.
Host copying, synchronization, cache thrashing and rejected draft work are
measured costs. Physical SSD bytes remain unmeasured. Native state consistency
and these performance observations do not resolve the full HF mismatch.

