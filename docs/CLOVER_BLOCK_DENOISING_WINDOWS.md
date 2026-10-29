# Native BF16 Clover block denoising

`strata-clover` supports independent `mode=denoise` generation. It uses the
complete official BF16 checkpoint and existing bounded SSD/RAM/VRAM expert
store. Python handles tokenization and evidence only. Full cross-engine model
logit validation remains unfinished; this mode does not resolve that failure.

## References and semantics

The semantic baseline is HF `modeling_clover_moe.py::generate()` and
`sampling_clover_moe.py`, revision `4b0402b65e523d0183a14c431955438714cb16ff`.
Every block begins entirely masked. Each iteration executes the full preview,
including already resolved positions, using read-only committed history and
the checkpoint's dynamic per-layer coreset. It fills pending positions whose
predicted-token T=1 probability is **strictly greater** than the threshold.
If none qualify, it fills the highest-confidence pending position; a tie
chooses the first position. Filled positions are never overwritten.

After the complete block resolves, output is truncated at EOS or the output
limit. Causal replay commits the returned block only if generation continues.
The final output block remains uncommitted, as in HF. There is no causal
verification, exact accept rule, draft-prefix rejection or bonus token.
Checkpoint block size remains 32, including GDN history scaling. Runtime
denoising width must divide 32; even a final one-token output computes the
entire configured canvas. Greedy only; stochastic generation is not implemented.

SGLang PR46927 head `2243dc6edb416073b60be0765904cabae9eb61cf` contains generic
`python/sglang/srt/dllm/algorithm/low_confidence.py`, which corroborates the
selection rule. However, `arg_groups/model_overrides/clover_moe.py` explicitly
rejects `dllm_algorithm` and requires `speculative_algorithm=CLOVER`. Therefore
the project's README claim of both modes does not establish a runnable Clover
denoising path in that PR. llama.cpp PR31306 head
`4838ffb0c8ade92f34d9157deea38d92758ab824` implements self-speculation.
Neither backend is installed or used for production here.

## Build and run

Use the [native Windows setup](CLOVER_BF16_WINDOWS.md). Added target:

```powershell
cmake --build <build> --target strata-clover strata-clover-denoise-test -j 3
python tools/clover/chat.py --exe <build>/strata-clover.exe --index <bundle>/index.json --prompt "Explain why the sky is blue in two sentences." --output <new-run> --mode denoise --denoise-block-size 32 --denoise-threshold 0.95 --ram-cache-mib 8192 --vram-cache-mib 8192 --staging-mib 16 --context-tokens 256 --max-new-tokens 64 --repeats 2 --presence-penalty 0 --frequency-penalty 0 --repetition-penalty 1
python tools/clover/report_run.py --run <new-run>
```

JSON keys are `mode: "denoise"`, `denoise_block_size` (default checkpoint 32),
`denoise_threshold` (default checkpoint 0.95). They are independent of
`draft_width` and `accept_rule`. The wrapper emits neither speculative key in
denoising configs. All budgets, context and tokenizer options retain their
existing meanings. Presence/frequency/repetition penalties above are the
controlled comparison settings; omit them to use checkpoint defaults.

To run widths8/16/32 and causal/selfspec8 controls sequentially:

```powershell
python tools/clover/benchmark_denoising.py --exe <build>/strata-clover.exe --index <bundle>/index.json --output <new-control-root>
```

Optional `--qwen-source <qwen-worktree> --qwen-exe <qwen-build>/strata-q36-bf16.exe
--qwen-index <qwen-bundle>/index.json` adds the independent BF16 Qwen control.
Use `--help` for implemented options, `--summary-only --output <existing-root>`
to recompute summaries. Stop downloads and other model runs before measuring.

## Validation and measurements

```powershell
python tools/clover/test_denoising.py --exe <build>/strata-clover-denoise-test.exe --hf-sampling <model>/sampling_clover_moe.py --output <new-controller-test>
python tools/clover/check_denoising_run.py --run <generation-run> --replay <independent-state-replay>
```

The production helper passed 50 controller cases and four rejected inputs,
including real vocabulary width248320, strict threshold/tie/fallback rules,
resolved positions, response penalties and EOS/length truncation. Native
compensated FP32 softmax confidence max error versus FP64 was 1.79e-7.
Observed CUDA FP32 differences reached 6.02e-6 on nearly uniform small tails
around a dominant logit; CUDA itself differed from FP64 by 5.90e-6. Both
values and initial failed comparisons are retained. Predicted IDs and transfer
sets matched the HF CUDA oracle in all test cases. Positions extremely close
to the threshold can remain sensitive to reduction order.

Complete-model audit cases covered multiple blocks, a shortened final output,
EOS, threshold1 fallback at every step and threshold0 full filling. Preview
state audits passed. Four independent causal replays matched all 320 state
files by SHA256 under different caches, including a full 32-token commit and
a one-token final output on a 32-position canvas. That boundary case used
threshold0 deliberately as a correctness test, not as performance evidence.
All reported benchmarks use the official default threshold0.95.
These are controller/state results;
the earlier full HF model-logit mismatch is still unresolved.

Logs separate `denoise_preview`, `denoise_commit`, CPU selection and every
iteration's transferred positions, confidence and actual output count.
TTFT means first fully resolved block availability, excluding model opening.
E2E token/s includes prefill and all generation. Post-prefill token/s counts
all denoising output tokens; causal/selfspec excludes its first token produced
by prefill. Native denoising steady rate excludes the **entire first output
block** and is null for a one-block request. Do not treat simultaneously
available block tokens as near-infinite decode throughput.

The wrapper prints completion after native exit. Application first/warm cache
states are not proof of physical cold SSD. Read counts and bytes are logical;
physical SSD I/O is unmeasured. Memory observations include separate process
commit/working set and whole-device samples, not just expert cache budgets.

## Measured on RTX 5070 Ti 16 GB

Native implementation commit `0c27fb7b216bcf839c2927f4be704dda258268ff`, exe
SHA256 `ad3e6820d506cc8d8d758900f857cc1a139f7e53a3936e72a7073813a1d1acc5`.
Qwen control source/binary remain on the independent BF16 branch. Machine,
toolchain, all checkpoint revisions and store details are in the main notes.
Both official models remain BF16. RAM/GPU expert caches8/8GiB, staging16MiB,
context256, prefill chunk32, greedy, thinking off, penalties0/0/1, threshold0.95,
two identical requests per process, audit/trace off. No concurrent download or
reference oracle. Tables show the second request; both requests, native
per-layer expert counts, memory, phase times and source hashes are preserved.

`Count from 1 to 10.`, maximum32 output IDs: every mode emitted the **same31
IDs including EOS**, decoding to `1, 2, 3, 4, 5, 6, 7, 8, 9, 10.`

| Mode | TTFT s | E2E token/s | Post-prefill token/s | Preview iterations per block |
| --- | ---: | ---: | ---: | --- |
| Qwen causal | 3.238 | 3.986 | 6.610 | - |
| Clover causal | 3.492 | 3.789 | 6.397 | - |
| Clover selfspec8 exact | 3.651 | 2.405 | 3.248 | - |
| Clover denoise8 | 5.190 | 2.766 | 4.107 | 2,1,1,2 |
| Clover denoise16 | 5.318 | 3.750 | 6.698 | 2,1 |
| Clover denoise32 | 9.076 | 3.416 | 5.874 | 4 |

[Count evidence](clover/denoising-count-20261029.json).
Width16 uses three previews plus one causal commit instead of30 causal
decode calls. Its post-prefill time was4.628s versus4.690s causal; the rate
numerators differ (31 denoise output IDs versus30 new causal IDs after the
prefill-produced first ID). Thus6.698 versus6.397 is not a comparable5% latency
win. E2E was8.268s versus8.181s, effectively similar in these short observations.
Expert evaluations fell4258 versus9600, but H2D rose18.604 versus15.557GiB;
larger expert unions and weaker cache reuse offset much of the parallel gain.
The single commit cost2.187s of4.628s post-prefill time. No HF final-block
commit was added: width32's one-block run has none and steady rate is null.

`Explain why the sky is blue in two sentences.`, maximum64 output IDs:

| Mode | Actual output IDs, including EOS | TTFT s | E2E token/s | Post-prefill token/s | Preview iterations per block |
| --- | ---: | ---: | ---: | ---: | --- |
| Qwen causal | 49 | 4.044 | 3.906 | 5.646 | - |
| Clover causal | 36 | 4.119 | 3.442 | 5.519 | - |
| Clover selfspec8 exact | 36 | 4.220 | 1.641 | 1.975 | - |
| Clover denoise8 | 41 | 7.138 | 1.966 | 2.473 | 6,8,8,7,8,1 |
| Clover denoise16 | 48 | 13.894 | 1.643 | 1.932 | 16,16,15 |
| Clover denoise32 | 47 | 34.708 | 0.896 | 0.978 | 26,29 |

[Text evidence](clover/denoising-sky-20261029.json). Different denoising widths
and causal generation can emit different text and EOS lengths. These rows
are observed request performance, not identical-output latency comparisons or
quality rankings. Each repeated request was deterministic, and causal/exact
matched. Their first32 IDs also matched the earlier binaries.

On text, strict0.95 confidence often resolves only one position by fallback.
Width32 needed55 full previews for47 output IDs. Their expert unions ranged
668..2064 over the40 layers;20/55 previews exceeded the1365 GPU expert slots.
Post-prefill H2D was230.977GiB, logical reads36.990GiB, CPU weight copies20.581s,
file submit/wait6.792s and demotion enqueue/sync7.040s within48.061s. Preview
itself totaled42.596s, causal commit3.653s and CPU selection1.712s. GPU copy
time overlaps host waits and must not be added again. This measured path is
limited by repeated work and tier traffic; larger canvas width is not a
guaranteed speedup.

Width32 process peak working set was about8.50GiB, private commit21.72GiB,
sampled whole-device VRAM14.15GiB. Cache budgets are not total process memory.
No physical SSD byte count or cold-cache claim. Two short requests provide no
confidence interval. The unresolved native-vs-HF model logits may also affect
confidence and iteration counts; these are experimental implementation results,
not a conclusion about Clover's validated full-GPU decoding potential.

[Validation evidence, including initial failures](clover/denoising-validation-20261029.json).
