#!/usr/bin/env bash
# start.sh — DeepSeek-V4.1-Flash-1017 on two DGX Spark / GB10 nodes with TensorFold's TP2 engine. Run on the head.
#   scripts/start.sh                 start rank 0 here, then rank 1 on the worker; wait for the API
#   scripts/start.sh restart         stop both, then start (apply .env changes)
#   scripts/start.sh status          containers, memory, /v1/models
#   scripts/start.sh logs [worker]   last lines of rank 0 (or rank 1); FOLLOW=1 to follow
#   scripts/start.sh down            stop both ranks (same as scripts/stop.sh)
# The first start writes each rank's weight cache (~99 GB a node) into $CACHE and takes much longer; later starts are
# about two minutes. The API is OpenAI-compatible on http://<head>:$PORT/v1, model "$SERVED", open to the network.
set -euo pipefail
. "$(dirname "$0")/common.sh"

run_rank() {   # $1 rank, $2 "local" or "worker"
  local rank=$1 where=$2 host=127.0.0.1 cmd
  [ "$rank" = 0 ] && host=0.0.0.0
  cmd="mkdir -p '$CACHE' && docker rm -f $NAME >/dev/null 2>&1; docker run -d --name $NAME --gpus all --network host --ipc host \
    --ulimit memlock=-1 --ulimit stack=67108864 --cap-add IPC_LOCK --device /dev/infiniband --restart no \
    -v '$MODEL_DIR':/model:ro -v '$ENGRAM_DIR':/engram:ro -v '$CACHE':/root/.cache \
    -e TF_DS_REPLAY=1 -e TF_DS_REPLAY_FLOOR=1 -e TF_DS_PREFILL_CHUNK=2048 \
    -e TF_DS_RANK_CACHE=/root/.cache/tf-ds-rank -e TF_DS_RANK_CACHE_READERS=32 \
    -e TF_DS_WARM_LENGTHS=1,17,33,131,514,1024,2113 -e TF_DS_ENGRAM=/engram \
    -e NCCL_IB_HCA=$HCA -e NCCL_IB_GID_INDEX=$GID -e NCCL_SOCKET_IFNAME=$IFACE -e TF_RDMA_DEVICES=$HCA $EXTRA_ENV \
    $IMAGE tensorfold serve /model --tp 2 --rank $rank --master $HEAD_IP --host $host --port $PORT \
      --name $SERVED --context $CONTEXT --vision --parallel $PARALLEL --mtp-drafts $MTP_DRAFTS $EXTRA_ARGS >/dev/null \
    && echo 'rank $rank started'"
  if [ "$where" = local ]; then bash -c "$cmd"; else on_worker "$cmd"; fi
}

down() {
  docker rm -f $NAME >/dev/null 2>&1 && echo "head: rank 0 stopped" || true
  on_worker "docker rm -f $NAME >/dev/null 2>&1 && echo 'worker: rank 1 stopped' || true"
}

status() {
  echo "head:   $(docker ps --filter name=$NAME --format '{{.Status}}') $(free -g | awk '/Mem:/{print "avail", $7, "GiB"}')"
  echo "worker: $(on_worker "docker ps --filter name=$NAME --format '{{.Status}}'; free -g | awk '/Mem:/{print \"avail\", \$7, \"GiB\"}'" | tr '\n' ' ')"
  curl -s --max-time 5 "http://127.0.0.1:$PORT/v1/models" | head -c 300; echo
}

up() {
  for d in "$MODEL_DIR" "$ENGRAM_DIR"; do
    [ -d "$d" ] || die "missing $d on the head (scripts/download-weights.sh)"
    on_worker "test -d '$d'" || die "missing $d on the worker"
  done
  [ -f "$MODEL_DIR/chat_template.jinja" ] || die "no chat_template.jinja in $MODEL_DIR (scripts/install-chat-template.sh)"
  # GB10: CUDA allocations do not reclaim the page cache, so it is dropped before the weights load
  drop="sync; echo 3 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null || echo 'warning: drop_caches needs sudo -n'"
  bash -c "$drop"; on_worker "$drop"
  run_rank 0 local
  until ss -ltn | grep -q ":$STORE_PORT "; do       # rank 1 may only start once rank 0's store listens
    docker ps --format '{{.Names}}' | grep -qx $NAME || { echo "rank 0 stopped:"; docker logs --tail 40 $NAME; exit 1; }
    sleep 2
  done
  run_rank 1 worker
  t0=$(date +%s)
  until curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; do
    docker ps --format '{{.Names}}' | grep -qx $NAME || { echo "rank 0 stopped:"; docker logs --tail 40 $NAME; exit 1; }
    on_worker "docker ps --format '{{.Names}}' | grep -qx $NAME" \
      || { echo "rank 1 stopped:"; on_worker "docker logs --tail 40 $NAME"; exit 1; }
    sleep 15
  done
  echo "ready after $(( $(date +%s) - t0 )) s on http://$(hostname):$PORT/v1"
  status
}

case "${1:-up}" in
  up|start) up ;;
  restart) down; up ;;
  down|stop) down ;;
  status) status ;;
  logs) if [ "${2:-}" = worker ]; then
          if [ "${FOLLOW:-0}" = 1 ]; then exec ssh "$WORKER" docker logs -f --tail 200 $NAME; fi
          exec ssh "$WORKER" docker logs --tail "${3:-80}" $NAME
        fi
        if [ "${FOLLOW:-0}" = 1 ]; then exec docker logs -f --tail 200 $NAME; fi
        exec docker logs --tail "${3:-80}" $NAME ;;
  *) echo "usage: $0 [up|restart|status|logs [worker]|down]"; exit 1 ;;
esac
