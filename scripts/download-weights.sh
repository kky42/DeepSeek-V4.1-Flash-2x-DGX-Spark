#!/usr/bin/env bash
# download-weights.sh — the EXL3 2.9 bpw pack (~211 GB) and DeepSeek's 1017 Engram shards (~203 GB) onto both nodes.
# Run on the head. Downloads land on the head, then are mirrored to the worker over the CX7 link (~650 MB/s).
#   scripts/download-weights.sh                    # everything through the head
#   scripts/download-weights.sh --engram-on-worker # the worker fetches the Engram shards itself, in parallel; worth it
#                                                  # only when each node has its own internet uplink (see TROUBLESHOOTING)
# Resumable: rerun after an interruption. Keep the GPUs idle while it runs (large file I/O beside a GPU job has stalled
# GB10 nodes).
set -euo pipefail
. "$(dirname "$0")/common.sh"
split=0; [ "${1:-}" = --engram-on-worker ] && split=1

command -v "$HF" >/dev/null || die "hf CLI not found ($HF); install: curl -LsSf https://hf.co/cli/install.sh | bash"
export HF_XET_HIGH_PERFORMANCE=1
mkdir -p "$MODEL_DIR" "$ENGRAM_DIR"
on_worker "mkdir -p '$MODEL_DIR' '$ENGRAM_DIR'"

# Engram shards: names passed positionally (hf 1.x drops the first name of a multi-value --include).
engram_cmd="$HF download $BASE_REPO $ENGRAM_SHARDS model.safetensors.index.json config.json --revision $BASE_REV --local-dir '$ENGRAM_DIR'"
if [ $split = 1 ]; then
  say "worker: fetching the 1017 Engram shards (background)"
  on_worker "export HF_XET_HIGH_PERFORMANCE=1; $engram_cmd" > /tmp/ds41-engram-worker.log 2>&1 &
  engram_pid=$!
fi

say "head: fetching $PACK_REPO @ ${PACK_REV:0:8}"
"$HF" download "$PACK_REPO" --revision "$PACK_REV" --local-dir "$MODEL_DIR"
[ "$(ls "$MODEL_DIR"/model-*.safetensors | wc -l)" = 43 ] || die "expected 43 pack shards in $MODEL_DIR"
say "mirroring the pack to the worker"
rsync -a --info=progress2 --exclude .cache/ "$MODEL_DIR/" "$WORKER:$MODEL_DIR/"

if [ $split = 1 ]; then
  wait "$engram_pid" || die "worker Engram download failed: /tmp/ds41-engram-worker.log"
  on_worker "cd '$ENGRAM_DIR' && printf '%s\n' '$ENGRAM_SHA256' | sha256sum -c" || die "Engram checksum mismatch on the worker"
  say "pulling the Engram shards to the head"
  rsync -a --info=progress2 --exclude .cache/ "$WORKER:$ENGRAM_DIR/" "$ENGRAM_DIR/"
else
  say "head: fetching the 1017 Engram shards"
  bash -c "$engram_cmd"
  say "mirroring the Engram shards to the worker"
  rsync -a --info=progress2 --exclude .cache/ "$ENGRAM_DIR/" "$WORKER:$ENGRAM_DIR/"
fi
say "verifying the Engram shards (SHA-256 from the release; ~2 min)"
( cd "$ENGRAM_DIR" && printf '%s\n' "$ENGRAM_SHA256" | sha256sum -c ) || die "Engram checksum mismatch on the head"
echo "done: pack and Engram on both nodes. Next: scripts/install-chat-template.sh"
