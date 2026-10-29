# Qwen3.6 BF16 text comparison on Windows

This is an opt-in experimental comparison path, independent of the Clover
branch. Base commit: 46c152a1310f77cd2f0ce2c7cd213962df14fe83. It uses Strata's
DirectFile and prefill::Gemm with bounded BF16 expert caching, and reuses causal
components from the Clover experiment with Qwen-specific numerical boundaries.
See src/qwen36_bf16/ATTRIBUTIONS.md. The normal Strata engine remains unchanged.

Current checkpoint revision: Qwen/Qwen3.6-35B-A3B at
995ad96eacd98c81ed38be0c5b274b04031597b0. All 26 safetensors headers were audited:
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
Complete-model generation/performance remain pending until checkpoint download
and full verification finish. Do not use metadata-only audit output for inference.

In a VS2022 x64 tools shell, with CUDA/CMake/Ninja on PATH and CUDA bin/x64 DLLs:

```powershell
cmake -S . -B C:/workspace/builds/T-20261029-qwen36-bf16/native -G Ninja -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120 -DSTRATA_BUILD_TESTS=OFF -DSTRATA_BUILD_Q36_BF16=ON -DSTRATA_GGML_DIR=C:/workspace/repos/reference/llama-cpp-3cf03257
cmake --build C:/workspace/builds/T-20261029-qwen36-bf16/native --target strata-q36-bf16 strata-q36-bf16-expert-test strata-q36-bf16-layer-test -j 3
python tools/qwen35/download_bf16.py --destination <model> --metadata-only
python tools/qwen35/download_bf16.py --destination <model> --backend hf --workers 3
python tools/qwen35/index_bf16.py --model <model> --output <bundle>/index.json
python tools/qwen35/chat_bf16.py --exe <build>/strata-q36-bf16.exe --index <bundle>/index.json --prompt "Count from 1 to 10." --output <new-run> --ram-cache-mib 8192 --vram-cache-mib 8192 --context-tokens 256 --max-new-tokens 16 --repeats 2
```

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

