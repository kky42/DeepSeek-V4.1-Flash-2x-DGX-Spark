# How we got here

DeepSeek released [DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) on
2026-10-17. Goal for the day: serve it on two GB10 nodes with the fastest decode we could get, a large context and 1–4
concurrent sessions. This is the path, dead ends included, so others can skip them.

## 1. The checkpoint does not fit two nodes as released

The official checkpoint is 475 GiB: MXFP4 routed experts, FP8 dense weights and two ~95 GiB Engram n-gram tables. Even
with the Engram tables served from NVMe, the GPU-resident part (~290 GiB) needs ~145 GiB per rank at TP 2, more than a
GB10's ~121 GiB. [MiaAI-Lab's SGLang recipe](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-DGX-Sparks) runs the
native weights, but on three or four nodes; its preflight says so. Two nodes need a ~2.7–2.9 bpw quant of the routed
experts.

## 2. No 1017 quant for two nodes existed yet

On release day the 1017 quants on Hugging Face were GGUF (Q2_K is 246 GiB, too big), NVFP4 (400 GiB) and MLX (Apple
only). The fast two-node engines (TensorFold, sudoingX's vLLM build, Mia-AiLab's EXL3 kit) all load EXL3 packs, and the
existing EXL3 packs were of the original V4.1. Mia-AiLab's 2-node kit
([DeepSeek-v4.1-Flash-EXL3-2x-DGX-Sparks](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-EXL3-2x-DGX-Sparks)) only
serves its own pack and ships no quantizer, and stock exllamav3 (through 1.6.3) has no DeepSeek-V4.1 architecture.

## 3. Quantizing it ourselves on rented H200s (parked)

[sudoingX/dgx-spark-deepseek-v41](https://github.com/sudoingX/dgx-spark-deepseek-v41) publishes a full pipeline that
quantizes the official checkpoint into an EXL3 pack with a per-expert bit allocation (~16 h on two GB10s). We ported it
to [Hugging Face Jobs](https://huggingface.co/docs/hub/jobs): the reference model's Engram reads fall back to `np.memmap`
(no vLLM extension needed), and pass 1's expert split runs as 8 local processes on one 8×H200 node instead of across two
boxes. Inside a Job the 475 GB checkpoint downloads in under 3 minutes.

- A 2-layer smoke run on one H200 passed every stage (calibration set, routing statistics, sensitivity, encode,
  dense pass, assembly).
- The full routing statistics and sensitivity for 1017 ran on one H200 in 33 minutes and gave a 2.70 bpw plan
  (85.75 GiB of experts per rank; 95% of experts got the same bit width as the original V4.1's plan).
- The full encode was estimated at 5–6 h on 8×H200 (~$200–240). Its first launch failed at startup (that node's
  container had no `nvidia-smi`), and by then a ready-made pack had appeared, so it never ran.
- Spent on HF Jobs: about $8.

## 4. A drop-in 1017 pack appears

The same afternoon [LibertAIDAI](https://huggingface.co/LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw) published a
1017 EXL3 2.9 bpw pack in Mia-AiLab's layout: same tensor names and per-tensor bit widths, plus the 43 `gate.bias_vl`
tensors (image routing) and 6 bpw instead of 4 for the drafter's dense layers. Its 3.5 bpw sibling (240 GiB) does not fit
two nodes.

## 5. TensorFold instead of Mia-AiLab's kit

On the original V4.1, [bertholomus's TensorFold TP2 engine](https://github.com/bertholomus/deepseek-v4.1-tensorfold-tp2-2xgb10)
published ~100 / 63 / 143 tok/s (code / prose / structured) against ~39 / 32 for Mia-AiLab's vLLM kit, with a 1M
context and 4 parallel streams. It reads each tensor's bit width from its shape and has kernels for every width, so the
6 bpw drafter layers needed no engine change, and it picks up `bias_vl` from the pack. We used its Python engine
(branch `deepseek-v41-tp2`), which is also the prerequisite for the newer Zig engine.

The one fix: the first boot failed after loading everything, because 1017 ships no Jinja chat template. DeepSeek's
official V4.1 template turned out to be exactly right for 1017's text and tool prompts. Writing this repo's smoke test
later caught that it refused image requests through TensorFold; removing its three image-placeholder guards fixed that
without changing any text prompt (see [TROUBLESHOOTING](TROUBLESHOOTING.md)).

## 6. Measured

Decode matches the engine's published numbers on the original V4.1 (97 / 60 / 144 tok/s, 118 tok/s with 4 streams),
prefill trails them, and quality on a seeded MMLU + GSM8K run matches DeepSeek's own API within noise. Details in
[results/RESULTS.md](../results/RESULTS.md).

## Credits

- [bertholomus/TensorFold](https://github.com/bertholomus/TensorFold) (built on [ashhart/TensorFold](https://github.com/ashhart/TensorFold)): the engine, `kit_bench.py` and `quality_eval.py`.
- [LibertAIDAI](https://huggingface.co/LibertAIDAI): the 1017 EXL3 pack.
- [Mia-AiLab / MiaAI-Lab](https://github.com/MiaAI-Lab): the EXL3 layout and the original 2-node kits.
- [sudoingX](https://github.com/sudoingX/dgx-spark-deepseek-v41): the quantization pipeline we ported, and the GB10 lessons in its FINDINGS.
- [turboderp/exllamav3](https://github.com/turboderp-org/exllamav3): EXL3.
- [DeepSeek](https://huggingface.co/deepseek-ai): the model, its encoder and the V4.1 chat template (MIT).
