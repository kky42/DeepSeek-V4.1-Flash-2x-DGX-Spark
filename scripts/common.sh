# Sourced by the scripts: settings from the repo's .env over the defaults below, and a few helpers.
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
if [ -f "$ROOT/.env" ]; then set -a; . "$ROOT/.env"; set +a; fi

: "${WORKER:?set WORKER in .env (the worker as ssh reaches it over the CX7 link, e.g. user@10.0.0.1)}"
: "${HEAD_IP:?set HEAD_IP in .env (the head IPv4 on the CX7 link)}"
HCA=${HCA:-rocep1s0f1,roceP2p1s0f1}
IFACE=${IFACE:-enp1s0f1np1}
GID=${GID:-3}
MODEL_DIR=${MODEL_DIR:-$HOME/Models/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw}
ENGRAM_DIR=${ENGRAM_DIR:-$HOME/Models/DeepSeek-V4.1-Flash-1017-engram}
CACHE=${CACHE:-$HOME/.cache/ds41-tf}
HF=${HF:-hf}
IMAGE=${IMAGE:-tensorfold-dsv41:8e0966e}
PORT=${PORT:-8888}
SERVED=${SERVED:-deepseek-v4.1-flash}
CONTEXT=${CONTEXT:-1048576}
PARALLEL=${PARALLEL:-4}
MTP_DRAFTS=${MTP_DRAFTS:-5}
EXTRA_ENV=${EXTRA_ENV:-}
EXTRA_ARGS=${EXTRA_ARGS:-}

# Pinned inputs: the revisions validated in results/RESULTS.md.
PACK_REPO=LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw
PACK_REV=afc7aa0aab98ee735c0f54e0a122019af1ac08f1           # weights identical in later revisions (card edits only)
BASE_REPO=deepseek-ai/DeepSeek-V4.1-Flash-1017
BASE_REV=1bae0e1088993b9b82de08f59de5be5a34e42170
ENGRAM_SHARDS="model-00047-of-00048.safetensors model-00048-of-00048.safetensors"
ENGRAM_SHA256="76a05106ece259be83d1bfea0dcb54f23377864ee60fd321690f35e6698a2af5  model-00047-of-00048.safetensors
9dc51433e195bc78b5ad846b68643cc0351d3b174e5e4309222d3b7402c83afd  model-00048-of-00048.safetensors"
TEMPLATE_URL=https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/resolve/314113f8a7dadb388215396ba5fbee2c610a04d6/chat_template.jinja
TEMPLATE_SHA256=d959d804d5101b79a49b6ff1bf3c54cd5affa9a5a78c503d925f3090040b31c3
TF_COMMIT=8e0966e2b09250abe2f6d9cc51edc20630e6eb1a                # bertholomus/TensorFold, branch deepseek-v41-tp2

NAME=ds41-tf                                                      # container name on both nodes
STORE_PORT=29551                                                  # rank 0's TCPStore

on_worker() { ssh -o BatchMode=yes "$WORKER" "$@"; }
say() { echo "== $*"; }
die() { echo "error: $*" >&2; exit 1; }
