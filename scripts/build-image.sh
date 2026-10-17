#!/usr/bin/env bash
# build-image.sh — build the TensorFold image on the head and copy it to the worker over the CX7 link.
# ~25 GB; the base (nvcr.io/nvidia/pytorch:26.07-py3) is pulled on the first build. Build with the GPUs idle.
set -euo pipefail
. "$(dirname "$0")/common.sh"

docker build --build-arg TF_COMMIT="$TF_COMMIT" -t "$IMAGE" "$ROOT/docker"
say "copying $IMAGE to the worker"
docker save "$IMAGE" | on_worker docker load
here=$(docker image inspect -f '{{.Id}}' "$IMAGE")
there=$(on_worker "docker image inspect -f '{{.Id}}' $IMAGE")
[ "$here" = "$there" ] || die "image ids differ: head $here, worker $there"
echo "$IMAGE on both nodes (${here:7:12})"
