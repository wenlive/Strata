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
state audits passed. Three independent causal replays matched all 240 state
files by SHA256 under different caches. These are controller/state results;
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
