# Qwen3.6 BF16 text comparison on Windows

This is an opt-in experimental comparison path, independent of the Clover
branch. Base commit: 46c152a1310f77cd2f0ce2c7cd213962df14fe83. It uses Strata's
DirectFile and prefill::Gemm with bounded BF16 expert caching, and reuses causal
components from the Clover experiment with Qwen-specific numerical boundaries.
See src/qwen36_bf16/ATTRIBUTIONS.md. The normal Strata engine remains unchanged.

Current checkpoint revision: Qwen/Qwen3.6-35B-A3B at
995ad96eacd98c81ed38be0c5b274b04031597b0. All 26 safetensors files passed full SHA256 verification:
69,321,221,376 bytes for the complete 40-layer text model, including 60 GiB
routed experts; 893,142,496 bytes vision and 1,689,281,536 bytes MTP are classified
separately and retained on disk, not executed in this causal text comparison.
No quantization or expert pruning. No claim of vision or MTP support.

Qwen expert SiLU/multiply/down/route weighting and accumulation retain their
BF16 rounding boundaries. Router weights round to BF16. GDN convolution and
gated RMS norm also differ from Clover's reference and are implemented separately.
All GEMMs keep the same 32-row compute geometry as the Clover comparison path.
Sampling is explicitly greedy, overriding the Qwen checkpoint's sampling default.
This setting is recorded; it is not presented as the checkpoint's default generate.

Native Windows build passed on RTX 5070 Ti (queried sm_120), CUDA 13.2.78 and
VS2022. Synthetic expert tests passed 24 cache/row cases, including eviction,
byte equality, offsets above 4 GB and six invalid inputs. Selected real layer-0
GDN comparison passed against Transformers 5.21.0 with layer-streamed experts;
maximum relative RMS 0.0016257. Initial slot-order comparisons failed on tied
router scores; the report retains those differences and additionally compares
expert sets and per-ID weights, which passed the unchanged numerical bounds.
Layer-3 attention/KV/output comparisons also passed, but its first router-weight
integration comparison failed (relative RMS 0.0056448 versus the 0.004 bound).
The failure is retained. With identical saved native BF16 logits, isolated L0
and L3 routers matched Torch expert sets and BF16 per-ID weights exactly in all
12 cases. Upstream numerical differences remain under investigation.
Complete native 40-layer generation produced `Hello!` and EOS. All 80 committed
KV/conv/SSM files matched independent one-token replay by SHA256 under different
cache budgets. Full streamed HF validation still **failed**: logits max/mean
absolute error 0.984375/0.0599591, relative RMS 0.0320939, first differing expert
set at layer 4. Final-position argmax matched. These are unfinished numerical
differences, not a passing HF-equivalence result; tolerances were not widened.
Do not use metadata-only audit output for inference.

[The measured Qwen/Clover comparison](BF16_COMPARISON_WINDOWS.md) records
complete BF16 generation, controlled prompts, cache budgets, phase timings,
Nsight evidence and all numerical limits. At RAM/GPU expert caches 8/8 GiB,
warm causal throughput was 5.912 token/s for counting and 4.818 for ordinary
text; Clover causal was 5.607 and 4.808 respectively. These short observations
do not establish the BF16 hardware ceiling or cross-checkpoint text equality.

In a VS2022 x64 tools shell, with CUDA/CMake/Ninja on PATH and CUDA bin/x64 DLLs:

```powershell
cmake -S . -B C:/workspace/builds/T-20261029-qwen36-bf16/native -G Ninja -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120 -DSTRATA_BUILD_TESTS=OFF -DSTRATA_BUILD_Q36_BF16=ON -DSTRATA_GGML_DIR=C:/workspace/repos/reference/llama-cpp-3cf03257
cmake --build C:/workspace/builds/T-20261029-qwen36-bf16/native --target strata-q36-bf16 strata-q36-bf16-expert-test strata-q36-bf16-layer-test -j 3
python tools/qwen35/download_bf16.py --destination <model> --metadata-only
python tools/qwen35/download_bf16.py --destination <model> --revision 995ad96eacd98c81ed38be0c5b274b04031597b0 --backend hf --workers 3
python tools/qwen35/index_bf16.py --model <model> --output <bundle>/index.json
python tools/qwen35/chat_bf16.py --exe <build>/strata-q36-bf16.exe --index <bundle>/index.json --prompt "Count from 1 to 10." --output <new-run> --ram-cache-mib 8192 --vram-cache-mib 8192 --staging-mib 16 --context-tokens 256 --prefill-chunk 32 --max-new-tokens 16 --repeats 2 --presence-penalty 0 --frequency-penalty 0 --repetition-penalty 1
```

Use a dedicated Python environment. The measured inventory was Python 3.12.10,
Transformers 5.21.0, tokenizers 0.23.3, safetensors 0.8.0, huggingface_hub 2.3.0,
numpy 2.5.4, psutil 7.2.2, nvidia-ml-py 13.615.71 and Torch 2.11.0+cu130.
Torch is for the numerical oracles; production forward runs in the native exe.
The pinned llama.cpp dependency is `3cf03257f219afbe7334045ff7c6a06ac68c627d`;
`STRATA_GGML_DIR` points to that existing source tree. `120` is the compute
capability queried on this machine, not a default for other hardware.

The complete index covers 693 text tensors and classifies 352 vision/MTP
tensors separately. Data bytes (excluding safetensors headers):

| Component | Bytes |
| --- | ---: |
| Routed experts (10,240, each 6 MiB) | 64,424,509,440 |
| Shared experts and gates | 251,822,080 |
| Embedding and LM head | 2,034,237,440 |
| GDN | 2,023,107,840 |
| Full attention | 545,269,760 |
| Norms and routers | 42,274,816 |
| **Complete text model** | **69,321,221,376** |
| Vision, retained but not executed | 893,142,496 |
| MTP, retained but not executed | 1,689,281,536 |
| **All checkpoint tensors** | **71,903,645,408** |

The 26 files occupy 71,903,776,776 bytes. Direct slicing needs no second full
pack. The downloader reserves 200 GiB of additional free disk space. Headers,
weight map, dtypes, shapes and source revision are rechecked when indexing.

The download/index commands stream and verify immutable sources. Index writes
are atomic; the executable rejects incomplete/audit/selected-layer scopes.
`--output` must be new. `strata-q36-bf16 --help` describes the JSON config.
`mode=forward` only commits input IDs for state comparisons; `mode=causal` is
the generation mode. No self-speculation, coreset or changed router is inserted.
The tokenizer uses the official non-thinking template unless --thinking is set.

RAM/VRAM expert budgets and staging are separate. There is no full host bank.
Exclusive cache demotion/promotion, CUDA events and single-stream dependencies
follow the measured Clover baseline. Logs distinguish prefill/decode, expert
unions, cache hits, logical/read-alignment bytes, H2D/D2H and memory observations.
Physical SSD traffic and true cold-I/O state are not inferred from these logs.
`download_bf16.py --only-files FILE...` can fetch a named official subset;
its subset manifest never marks the full checkpoint complete. HF Hub locks
serialize concurrent access to the same file. The full index independently
checks every checkpoint shard and rejects missing files.
`check_bf16_forward.py` is a layer-streamed full HF trace oracle;
`check_bf16_router.py` checks routing on identical saved native logits;
`check_bf16_state.py` compares exact committed inputs against causal replay.
Use equal prompt/context/cache budgets and greedy settings for comparisons;
Clover's default response penalties require a separate controlled configuration.
Do not run the large checkpoint download during performance measurements.

Store timing counters additionally distinguish CPU weight memcpy, RAM-hit staging waits and GPU demotion waits. They match the Clover instrumentation and do not change arithmetic or cache policy. Do not add CUDA copy durations to host waits as independent costs. Historical missing counters remain absent.


Index metadata coverage tests reject eight corrupt layouts (missing, dtype, shape, range, weight-map, duplicate, unexpected and overlapping tensors). Production also rejects metadata-only and selected-layer scopes. Complete indexing now checks every file size before hashing any shard; an unfinished checkpoint publishes no runnable index.


HTTP Range fallback: when Xet transport stalls, `python tools/qwen35/download_bf16_ranges.py --destination <model> --files model-00010-of-00026.safetensors model-00024-of-00026.safetensors --workers 8 --segment-mib 128` downloads bounded segments, validates every response range, stream-assembles and verifies the complete official SHA256 before publishing. Segment files remain resumable; they are never runtime model files. The subset marker does not mark the whole checkpoint complete. This path is a transport fallback, not a weight conversion.

The control-run aggregator is tools/qwen35/summarize_bf16_comparison.py --runs <comparison-root>. It checks equal prompt IDs, repeated outputs and penalty-free Clover causal/exact equality, and reports decode-only reads/transfers separately from prefill. It does not infer physical disk traffic.

To reproduce the control matrix, build and prepare both branches independently,
then run this from the Qwen branch. Replace angle-bracket placeholders with
actual paths; output directories must be new. Stop weight downloads and other
model runs before measuring. The driver runs every case sequentially and
preserves commands, logs, both repeats and a summary:

```powershell
python tools/qwen35/compare_bf16.py --qwen-exe <qwen-build>/strata-q36-bf16.exe --qwen-index <qwen-bundle>/index.json --clover-source <clover-worktree> --clover-exe <clover-build>/strata-clover.exe --clover-index <clover-bundle>/index.json --output <new-count-controls>
python tools/qwen35/compare_bf16.py --qwen-exe <qwen-build>/strata-q36-bf16.exe --qwen-index <qwen-bundle>/index.json --clover-source <clover-worktree> --clover-exe <clover-build>/strata-clover.exe --clover-index <clover-bundle>/index.json --output <new-text-controls> --prompt "Explain why the sky is blue in two sentences." --max-new-tokens 32 --budget 8192 8192
```

Default budgets are RAM/GPU MiB 0/6144, 8192/6144 and 8192/8192. `--budget R G`
is repeatable; `--draft-width` chooses 2/4/8/16/32 (default 8), while trained
Clover block size stays 32. The driver fixes greedy, thinking off, penalties
0/0/1 and exact acceptance. A real full-checkpoint 4-output-token control at
8/8 GiB passed all three variants and the summary equality checks. This small
runner check is separate from the recorded 16/32-token benchmarks.

