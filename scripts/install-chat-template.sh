#!/usr/bin/env bash
# install-chat-template.sh — the 1017 release ships no Jinja chat template, and TensorFold renders every prompt through
# one. DeepSeek's encoder, tokenizer and golden encodings are byte-identical between V4.1 and V4.1-1017, so this takes
# the official V4.1 template (pinned revision, hash-checked) and removes its three image-placeholder guards: TensorFold
# lays images out as DeepSeek's placeholder token inside the message text before rendering (and rejects placeholders a
# user types itself), which the unpatched template refuses with "Message content contains image special token".
# The result reproduces all five of 1017's golden encodings, and its image case through TensorFold's placeholder path
# (tests/check_chat_template.py). Installs into the pack folder on both nodes; restart the server to load it.
set -euo pipefail
. "$(dirname "$0")/common.sh"
PATCHED_SHA256=13f33e7f7d5f7f7a6fea44cde335750201acae72af9ebc018b2cf2255201cf83

sha() { local s; s=$(sha256sum "$1" 2>/dev/null || shasum -a 256 "$1"); echo "${s%% *}"; }
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
curl -fsSL "$TEMPLATE_URL" -o "$tmp/official.jinja"
[ "$(sha "$tmp/official.jinja")" = "$TEMPLATE_SHA256" ] || die "official template hash mismatch"
python3 - "$tmp/official.jinja" "$tmp/chat_template.jinja" <<'EOF'
import sys
src = open(sys.argv[1]).read()
guard_str = ('    {%- if content is string and IMG in content -%}\n'
             '        {{- raise_exception("Message content contains image special token " ~ IMG ~ ". Images should be '
             'provided as image content blocks.") -}}\n'
             '    {%- endif -%}\n')
guard_blk = ('{%- if IMG in (b.text or "") -%}{{- raise_exception("Text block contains image placeholder " ~ IMG ~ '
             '". Images should be separate content blocks.") -}}{%- endif -%}')
assert src.count(guard_str) == 1 and src.count(guard_blk) == 2, "template guards not found"
open(sys.argv[2], "w").write(src.replace(guard_str, "").replace(guard_blk, ""))
EOF
[ "$(sha "$tmp/chat_template.jinja")" = "$PATCHED_SHA256" ] || die "patched template hash mismatch"
install -m 644 "$tmp/chat_template.jinja" "$MODEL_DIR/chat_template.jinja"
scp -q "$tmp/chat_template.jinja" "$WORKER:$MODEL_DIR/chat_template.jinja"
echo "chat_template.jinja installed on both nodes (sha256 ${PATCHED_SHA256:0:16}...); restart the server to load it"
