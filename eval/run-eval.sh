#!/usr/bin/env bash
# run-eval.sh — the quality comparison in results/RESULTS.md: the same seeded 200 MMLU + 100 GSM8K questions, thinking
# off and on, asked of the local server and of DeepSeek-V4.1-Flash-1017 on OpenRouter pinned to each provider in
# PROVIDERS (fallbacks off), then a paired comparison. ~$0.25 of OpenRouter credit for two providers; the local run takes
# ~20 min. Run anywhere that reaches the server (the head is simplest).
#   export OPENROUTER_API_KEY=...          (OPENROUTER_API_TOKEN works too)
#   python3 eval/prepare-data.py           (once)
#   eval/run-eval.sh
#   LOCAL=http://HEAD:8888 PROVIDERS="DeepSeek" eval/run-eval.sh
set -euo pipefail
cd "$(dirname "$0")"
LOCAL=${LOCAL:-http://127.0.0.1:8888}
MODEL=${MODEL:-deepseek-v4.1-flash}
OR_MODEL=${OR_MODEL:-deepseek/deepseek-v4.1-flash-1017}
PROVIDERS=${PROVIDERS:-DeepSeek DeepInfra}
KEY_ENV=${KEY_ENV:-$( [ -n "${OPENROUTER_API_KEY:-}" ] && echo OPENROUTER_API_KEY || echo OPENROUTER_API_TOKEN )}
[ -n "${!KEY_ENV:-}" ] || { echo "set OPENROUTER_API_KEY (or OPENROUTER_API_TOKEN)"; exit 1; }
[ -f data/mmlu_all_test.jsonl ] || { echo "run python3 eval/prepare-data.py first"; exit 1; }

OUT=../results/eval/run-$(date +%Y%m%d-%H%M)
mkdir -p "$OUT"
COMMON="--mmlu data/mmlu_all_test.jsonl --gsm8k data/gsm8k_test.jsonl --modes off,on --parallel 4"
echo "results: $OUT"
python3 quality_eval.py --base "$LOCAL" --model "$MODEL" $COMMON --out "$OUT/local.json" > "$OUT/local.log" 2>&1 &
pids=($!)
names=(local)
for p in $PROVIDERS; do
  n=$(echo "$p" | tr '[:upper:]' '[:lower:]')
  python3 quality_eval.py --base https://openrouter.ai/api --model "$OR_MODEL" --key-env "$KEY_ENV" --provider "$p" \
    $COMMON --out "$OUT/$n.json" > "$OUT/$n.log" 2>&1 &
  pids+=($!); names+=("$n")
done
for i in "${!pids[@]}"; do
  wait "${pids[$i]}" && echo "${names[$i]}: done" || echo "${names[$i]}: FAILED (see $OUT/${names[$i]}.log)"
done
files=("$OUT/local.json")
for n in "${names[@]:1}"; do files+=("$OUT/$n.json"); done
python3 compare.py "${files[@]}" | tee "$OUT/compare.txt"
