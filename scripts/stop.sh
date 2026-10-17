#!/usr/bin/env bash
# stop.sh — stop both ranks of the server (see start.sh).
exec "$(dirname "$0")/start.sh" down
