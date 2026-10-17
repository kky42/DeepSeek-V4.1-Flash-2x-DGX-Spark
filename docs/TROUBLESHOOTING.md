# Troubleshooting and lessons learned

Everything below was hit for real while setting this up.

## Serving

**The server loads, prints `DeepSeek-V4.1 engine ready`, then exits with `KeyError: 'chat_template'`.**
DeepSeek did not ship a Jinja chat template with the 1017 release (only `encoding/encoding.py`), and TensorFold renders
every prompt through one. Run `scripts/install-chat-template.sh`. It installs DeepSeek's official V4.1 template:
1017's encoder, tokenizer and golden encodings are byte-identical to V4.1's, and that template reproduces all five of
1017's golden encodings (`python3 tests/check_chat_template.py`). Mia-AiLab's V4.1 template, the other obvious
candidate, reproduces only two of five (it misses tool definitions, the `<｜action｜>` marker and image content).
On the failed boot rank 1 exits with `ConnectionResetError`; that is a consequence, not a second problem.

**Image requests fail with `400: the chat template rejected the request: Message content contains image special token`.**
That is the unpatched official template. TensorFold replaces each image with DeepSeek's placeholder token inside the
message text (joined exactly as DeepSeek's encoder joins blocks), renders, then grows each placeholder into the image's
tokens; the official template refuses placeholders in text. `install-chat-template.sh` removes those three guard lines,
and the check script verifies that the image golden case then renders byte-identically through TensorFold's path.
Restart the server after installing a template; it is read once at startup.

**The first start takes a long time.** It writes each rank's weights into one file (~99 GB a node under `$CACHE`),
builds two CUDA extensions, and warms up 7 prompt lengths plus 64 decode graphs. Later starts take about two minutes.
Delete `$CACHE/tf-ds-rank` only if you change the weights.

**`warning: drop_caches needs sudo -n`.** On GB10, CUDA allocations do not reclaim the page cache, so a node can fail to
allocate with plenty of memory "available". `start.sh` drops the page cache before loading. Allow just that one command
without a password:

```bash
echo "$USER ALL=(root) NOPASSWD: /usr/bin/tee /proc/sys/vm/drop_caches" | sudo tee /etc/sudoers.d/drop-caches
sudo chmod 440 /etc/sudoers.d/drop-caches
```

**Rank 1 cannot connect / NCCL hangs.** Rank 1 may only start after rank 0's TCPStore listens (port 29551); `start.sh`
waits for that. Check `HCA`, `IFACE` and `GID` with `scripts/preflight.sh`: the GID index must be the RoCE v2 entry
carrying the link's IPv4 (3 on our nodes; TensorFold's default is 5). List both RoCE devices of the cable in `HCA`.

**The server is open to the whole network.** It binds `0.0.0.0` without an API key. TensorFold accepts
`TF_API_KEY_FILE=<file>` (every route but `/health` then needs the key); pass it through `EXTRA_ENV` with a `-v` mount if
you need it. We have not exercised that path.

**Startup prints `TensorFold 1.0.12 is available`.** Ignore it: the pinned engine commit is the one measured here. Do not
run `tensorfold update` inside the container.

## Weights and downloads

**Use the 1017 Engram shards, not V4.1's.** 1017 changed all 48 shards, including the two Engram tables (47 and 48).
Their names and sizes are the same, so V4.1's copies load without an error and silently run a mismatched model.
`download-weights.sh` fetches them from the 1017 repo and checks their SHA-256.

**`hf download` fetched 3 of 4 files and reported success.** `huggingface_hub` 1.x (1.32 here) treats only the first
name after `--include` as a pattern and then ignores `--include` because more names follow. Pass filenames positionally
(as `download-weights.sh` does) or use `hf` 2.x.

**Both nodes download at ~100 MB/s combined, even though each has a 2.5 GbE port.** Look for a 1 Gb/s hop between their
switch and the router. In our case a laptop plugged into the router got 145 MB/s while the nodes together were capped
at ~100 MB/s; splitting downloads across the two nodes (`--engram-on-worker`) did not help for that reason. A USB hard
disk was no shortcut either (~33 MB/s write). Moving the nodes to a faster path is the real fix.

**`du` shows a download stuck at the same size.** Hugging Face's Xet transfer writes files out of order; watch the
interface counters (`/proc/net/dev`) instead.

**Do large copies with the GPUs idle.** Multi-GB file I/O beside a GPU job has stalled GB10 nodes (reported by sudoingX
and others). Copy weights and images before starting the server.

## Clients

**Thinking levels.** The server accepts `reasoning_effort` = `none`, `minimal`, `low`, `medium`, `high`, `xhigh`, `max`
(`none` turns thinking off), or `chat_template_kwargs: {"enable_thinking": false}`. From `low` to `max` the reasoning
budget roughly quadruples. See `clients/pi/models.json` for a Pi mapping.

**`pi -p` hangs from a script.** In print mode Pi reads stdin when it is not a terminal; redirect it (`< /dev/null`).

**Copying a Pi config between machines.** On the head itself the base URL is `http://127.0.0.1:8888/v1`; from any other
machine it is `http://<head>:8888/v1`.

## Open questions

- **Prefill is 30–45% below the engine author's published numbers** (≈1,300 vs ≈2,400 tok/s at 32K) while decode
  matches. Not yet explained. Candidates: the 1017 pack's 6 bpw drafter layers, cold Engram reads, or a difference in
  the link setup. bertholomus's newer Zig engine (branch `deepseek-v41-zig`) publishes ~2,450 tok/s prefill and
  ~100 tok/s decode after a 128K prompt, and reuses this engine's weight cache.
- **Thinking-on replies are shorter than DeepSeek's API at the same effort name**, with the same accuracy. The effort
  scale (`high` = 75 here) may differ from the API's.
