#!/usr/bin/env bash
# preflight.sh — read-only checks of both nodes before downloading or serving. Run on the head.
# Prints PASS / WARN / FAIL per check; exits 1 if any check fails.
set -uo pipefail
. "$(dirname "$0")/common.sh"

fails=0
pass() { printf 'PASS  %-8s %s\n' "$1" "$2"; }
warn() { printf 'WARN  %-8s %s\n' "$1" "$2"; }
fail() { printf 'FAIL  %-8s %s\n' "$1" "$2"; fails=$((fails + 1)); }

# The same probe runs on both nodes; it prints key=value lines.
probe='
echo "gpu=$(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null | head -1)"
echo "busy=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)"
echo "ip=$(ip -4 -o addr show '"$IFACE"' 2>/dev/null | awk "{print \$4}" | cut -d/ -f1)"
echo "mtu=$(cat /sys/class/net/'"$IFACE"'/mtu 2>/dev/null)"
for d in $(echo '"$HCA"' | tr , " "); do
  P=/sys/class/infiniband/$d/ports/1
  echo "hca_$d=$(cat $P/state 2>/dev/null | tr -d " ") type=$(cat $P/gid_attrs/types/'"$GID"' 2>/dev/null | tr -d " ") gid=$(cat $P/gids/'"$GID"' 2>/dev/null)"
done
echo "disk_gb=$(df -BG --output=avail "$HOME" | tail -1 | tr -dc 0-9)"
echo "mem_gb=$(awk "/MemAvailable/{print int(\$2/1048576)}" /proc/meminfo)"
echo "docker=$(docker version --format "{{.Server.Version}}" 2>/dev/null)"
echo "image=$(docker image inspect -f "{{.Id}}" '"$IMAGE"' 2>/dev/null | cut -c8-19)"
sudo -n tee /proc/sys/vm/drop_caches </dev/null >/dev/null 2>&1 && echo "sudo_drop=yes" || echo "sudo_drop=no"
echo "model=$([ -d '"$MODEL_DIR"' ] && ls '"$MODEL_DIR"'/model-*.safetensors 2>/dev/null | wc -l || echo missing)"
echo "template=$([ -f '"$MODEL_DIR"'/chat_template.jinja ] && echo yes || echo no)"
echo "engram=$(ls '"$ENGRAM_DIR"'/model-0004[78]-of-00048.safetensors 2>/dev/null | wc -l)"
'

check_node() {   # $1 label, $2 probe output
  local n=$1 out=$2 v
  get() { echo "$out" | sed -n "s/^$1=//p" | head -1; }
  v=$(get gpu); [ -n "$v" ] && pass "$n" "GPU: $v" || fail "$n" "nvidia-smi found no GPU"
  [ "$(get busy)" = 0 ] && pass "$n" "GPU idle" || warn "$n" "GPU has $(get busy) compute process(es) (stop other servers before starting)"
  v=$(get ip); [ -n "$v" ] && pass "$n" "IPv4 $v on $IFACE" || fail "$n" "no IPv4 on $IFACE"
  [ "$(get mtu)" = 9000 ] && pass "$n" "MTU 9000 on $IFACE" || warn "$n" "MTU $(get mtu) on $IFACE (9000 measured here)"
  for d in $(echo "$HCA" | tr , ' '); do
    v=$(get "hca_$d")
    case "$v" in
      4:ACTIVE*RoCEv2*) pass "$n" "$d active, GID $GID is RoCE v2 (${v##*gid=})" ;;
      4:ACTIVE*) fail "$n" "$d active but GID $GID is not RoCE v2: $v (set GID in .env)" ;;
      *) fail "$n" "$d not active or missing: $v" ;;
    esac
  done
  v=$(get disk_gb); [ "${v:-0}" -ge 120 ] && pass "$n" "${v} GB free in \$HOME" || warn "$n" "${v} GB free; the weight cache alone needs ~100 GB"
  v=$(get mem_gb); [ "${v:-0}" -ge 110 ] && pass "$n" "MemAvailable ${v} GiB" || warn "$n" "MemAvailable ${v} GiB (serving needs ~108 GiB; stop other GPU jobs)"
  v=$(get docker); [ -n "$v" ] && pass "$n" "docker $v" || fail "$n" "docker not reachable"
  v=$(get image); [ -n "$v" ] && pass "$n" "image $IMAGE ($v)" || warn "$n" "image $IMAGE missing (scripts/build-image.sh)"
  [ "$(get sudo_drop)" = yes ] && pass "$n" "sudo -n can drop the page cache" \
    || warn "$n" "no sudo -n for drop_caches (see docs/TROUBLESHOOTING.md; CUDA can fail to allocate without it)"
  v=$(get model); [ "$v" = 43 ] && pass "$n" "EXL3 pack: 43 shards" || warn "$n" "EXL3 pack: $v/43 shards (scripts/download-weights.sh)"
  [ "$(get template)" = yes ] && pass "$n" "chat_template.jinja installed" || warn "$n" "no chat_template.jinja (scripts/install-chat-template.sh)"
  v=$(get engram); [ "$v" = 2 ] && pass "$n" "1017 Engram shards present" || warn "$n" "Engram shards: $v/2"
}

head_out=$(bash -c "$probe")
check_node head "$head_out"
if worker_out=$(on_worker "bash -s" <<<"$probe" 2>/dev/null); then
  pass worker "ssh $WORKER"
  check_node worker "$worker_out"
else
  fail worker "ssh $WORKER failed (passwordless ssh over the CX7 link is required)"
fi
hid=$(echo "$head_out" | sed -n 's/^image=//p'); wid=$(echo "${worker_out:-}" | sed -n 's/^image=//p')
[ -n "$hid" ] && [ "$hid" = "$wid" ] && pass both "same image id on both nodes" || warn both "image ids differ or missing: head '$hid', worker '$wid'"
[ "$(echo "$head_out" | sed -n 's/^ip=//p')" = "$HEAD_IP" ] && pass head "HEAD_IP matches $IFACE" || warn head "HEAD_IP=$HEAD_IP is not the IPv4 on $IFACE"
echo
[ "$fails" = 0 ] && echo "preflight: no failures" || { echo "preflight: $fails failure(s)"; exit 1; }
