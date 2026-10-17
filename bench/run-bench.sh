#!/usr/bin/env bash
# run-bench.sh — the speed numbers in results/RESULTS.md, with the engine author's own client (kit_bench.py from
# bertholomus/TensorFold @ the pinned commit, fetched and hash-checked, not vendored). Run on the head with the server up
# and nothing else using it. Greedy, thinking off, the same prompt sets as the published TensorFold numbers.
#   bench/run-bench.sh               decode (set b), 2/4-stream concurrency, prefill 8K/32K/128K, decode after 128K
#   QUICK=1 bench/run-bench.sh       decode only (~1 min)
# Output: results/bench/run-<date>/*.json
set -euo pipefail
. "$(dirname "$0")/../scripts/common.sh"
KIT="$ROOT/bench/kit_bench.py"
KIT_SHA256=83a4fbeb987032f906cf00d3f6c9dd2fad47dd675e75dd498c4c4188d63a6c0d
if [ ! -f "$KIT" ]; then
  curl -fsSL "https://raw.githubusercontent.com/bertholomus/TensorFold/$TF_COMMIT/tools/dsv41/kit_bench.py" -o "$KIT"
fi
got=$(sha256sum "$KIT" 2>/dev/null || shasum -a 256 "$KIT")
[ "${got%% *}" = "$KIT_SHA256" ] || die "kit_bench.py hash mismatch (delete $KIT and rerun)"

OUT="$ROOT/results/bench/run-$(date +%Y%m%d-%H%M)"
mkdir -p "$OUT"
kb() { python3 "$KIT" --base "http://127.0.0.1:$PORT" --model "$SERVED" "$@"; }
curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null || die "server not answering on :$PORT (scripts/start.sh)"

say "decode, set b (code / prose / structured), 384 tokens, median of 3"
kb decode --set b --tokens 384 --reps 3 --out "$OUT/decode_b.json" | tail -3
[ "${QUICK:-0}" = 1 ] && { echo "results: $OUT"; exit 0; }
say "2 and 4 concurrent streams, set b"
kb concurrent --streams 2,4 --set b --tokens 384 --reps 2 --out "$OUT/conc_b.json" | tail -4
say "prefill 8K / 32K / 128K (cold, then 8K and 32K again warm)"
kb prefill --lengths 8192,32768,131072 --out "$OUT/prefill.json" | tail -3
kb prefill --lengths 8192,32768 --reps 2 --out "$OUT/prefill_warm.json" | tail -4
say "decode after a 128K prompt"
kb depth --lengths 131072 --tokens 256 --out "$OUT/depth.json" | tail -1
echo "results: $OUT"
