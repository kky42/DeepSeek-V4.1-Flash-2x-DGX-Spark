# Results

Measured 2026-10-17 on two ASUS Ascent GX10 nodes (NVIDIA GB10, 128 GB unified memory each, DGX OS 7.6.0, kernel
7.0.0-1019-nvidia, driver 580.178.04) joined by one ConnectX-7 cable (two RoCE rails at 200 Gb/s, MTU 9000).

| Component | Version |
|---|---|
| Engine | TensorFold TP2 Python engine, [bertholomus/TensorFold](https://github.com/bertholomus/TensorFold) `deepseek-v41-tp2` @ `8e0966e` (reports itself as TensorFold 0.6.6), on `nvcr.io/nvidia/pytorch:26.07-py3` |
| Weights | [`LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw`](https://huggingface.co/LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw) @ `afc7aa0a` (EXL3 mul1, 2.9 bpw, 196 GiB) |
| Engram tables | [`deepseek-ai/DeepSeek-V4.1-Flash-1017`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) @ `1bae0e10`, shards 47 and 48, unquantized |
| Chat template | DeepSeek's official V4.1 template, `deepseek-ai/DeepSeek-V4.1-Flash` @ `314113f8`, with its three image-placeholder guards removed (`scripts/install-chat-template.sh`) |
| Settings | TP 2, `--context 1048576 --parallel 4 --mtp-drafts 5 --vision`, both RoCE rails, as in `scripts/start.sh` |

Raw files: [`bench/`](bench/) and [`eval/`](eval/).

## Speed

[`bench/run-bench.sh`](../bench/run-bench.sh): the engine author's `kit_bench.py`, greedy, thinking off. Prompt set b,
384 tokens, median of 3. The right-hand column is bertholomus's published number for the same engine (v0.5.1) on the
original V4.1 with Mia-AiLab's 2.9 bpw pack, on his hardware.

| | This setup (1017) | Published (original V4.1) |
|---|---:|---:|
| Decode, 1 stream: code | **97.4 tok/s** | 101.5 |
| Decode, 1 stream: prose | **59.6 tok/s** | 63.4 |
| Decode, 1 stream: structured | **144.2 tok/s** | 143.4 |
| 2 streams, aggregate (2 reps) | 71.2 / 81.9 tok/s | 79.8 |
| 4 streams, aggregate (2 reps) | **118.3 / 118.2 tok/s** | 113.0 |
| Decode after a 128K-token prompt | **71.1 tok/s** | 49.8 |
| Prefill 8K, cold / warm best | 1,646 / 2,003 tok/s | 2,387 |
| Prefill 32K, cold / warm | 1,341 / 1,279–1,350 tok/s | 2,421 |
| Prefill 128K, cold | 1,319 tok/s (99 s to first token) | 2,242 |

Greedy output was identical across repetitions. Decode matches the published numbers. Prefill is 30–45% lower, a gap
we have not explained (see [TROUBLESHOOTING](../docs/TROUBLESHOOTING.md#open-questions)).

Memory while serving: ~7 GiB MemAvailable on each node. Boot: ~108 s once the weight caches exist.

## Correctness

`tests/smoke.py` passes: arithmetic, factual recall, a correct `is_prime`, reasoning with thinking on, a tool call
(`finish_reason: tool_calls`), and an image (a generated red/blue PNG described correctly). `tests/check_chat_template.py`
reproduces all five of 1017's golden prompt encodings and the image case through TensorFold's placeholder path. The
speed and quality runs on this page were made before the image-guard patch; the patch does not change text prompts (smoke
outputs were identical before and after), and a decode re-check after it gave 58.3 / 141.1 tok/s (prose / structured).

## Quality

[`eval/run-eval.sh`](../eval/run-eval.sh): a fixed, seeded subset of 200 MMLU questions and 100 GSM8K problems,
greedy, each run with thinking off and on (effort `high`). The same 600 prompts went to the local server and to
DeepSeek-V4.1-Flash-1017 on OpenRouter, pinned per provider with fallbacks off. DeepSeek (first-party) is the
reference.

| Test | Local (EXL3 2.9 bpw) | DeepSeek API | DeepInfra (FP8) |
|---|---:|---:|---:|
| MMLU, thinking off (n=200) | **85.0%** | 84.5% | 83.5% |
| MMLU, thinking on | **92.5%** | 93.0% | 94.0% |
| GSM8K, thinking off (n=100) | **96.0%** | 96.0% | 93.0%¹ |
| GSM8K, thinking on | **98.0%** | 97.0% | 97.0% |
| Same answer as DeepSeek | **96–98%** | — | 96–97% |

¹ Two of DeepInfra's requests timed out on the network and count as wrong (93/98 answered).

Every local-vs-DeepSeek difference is within noise (paired exact McNemar test, all p ≥ 0.25; full output in
[`eval/compare.txt`](eval/compare.txt)). The local quant agrees with DeepSeek's own servers as often as an FP8 API
provider does. With thinking on, three MMLU questions ran past the 8,192-token cap both locally and, for two of the same
three, on DeepSeek's API. Local thinking-on replies were 25–35% shorter than the API's at the same `high` effort, with
the same accuracy; the two likely map `high` to different reasoning budgets.

What this run can and cannot show: with 200 questions a single run moves by about ±3 points, so it rules out large
regressions, not subtle ones. Both benchmarks are near saturation for this model. IFEval and a tool-calling benchmark
would be the next, more sensitive tier. OpenRouter cost for both providers: $0.23.
