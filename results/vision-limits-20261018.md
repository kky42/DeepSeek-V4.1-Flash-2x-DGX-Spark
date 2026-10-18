# Image limits

Measured 2026-10-18 with [`tests/image_limits.py`](../tests/image_limits.py) on the setup in
[RESULTS.md](RESULTS.md), thinking off, `max_tokens: 4`, synthetic PNGs.

| request | body | result | prompt tokens |
|---|---:|---|---:|
| text only | 0.0 MB | 200 | 13 |
| 1 × 32×32 | 0.0 MB | 200 | 198 (+185) |
| 1 × 1600×1200 | 0.2 MB | 200 | 1,015 (+1,002) |
| 8 × 32×32 | 0.0 MB | 200 | 1,493 = 13 + 8 × 185 |
| 64 × 32×32 | 0.3 MB | 200, 26.0 s | 11,853 = 13 + 64 × 185 |
| 65 × 32×32 | 0.3 MB | **400** `image input: a request may carry at most 64 images` | — |
| 64 × 1600×1200 | **78.1 MB** | 200, 275.7 s | 64,141 = 13 + 64 × 1,002 |

Per-image cost adds exactly and does not depend on the other images in the request: image 0 costs 185 tokens alone and
185 inside a 64-image request. Size alone decides it — below `min_pixels` (295,936) an image is scaled up, at or above
`max_image_tokens` (1,024) it is scaled down; both come from the model's `config.json`. One image is capped at 20 MB
(`DsVision.max_bytes`).

With the stock 32 MiB body cap, 64 images may average only 0.5 MB each, and a 78 MB body is dropped before it reaches
the engine; `TF_MAX_BODY_MIB` raises the cap. The 64-image request took 275.7 s end to end from a laptop over WiFi —
about 100 s of that is the 78 MB upload, and the server's time to first token is 174 s: 63 s of prefill plus roughly
110 s in rank 0's vision tower (~1.7 s per image, fp32 attention). Prefix caching still works across image turns
(2048-token blocks).

On 64 synthetic images the reply was "I observed forty images". Counting 64 abstract PNGs is not something to rely on;
the token arithmetic above is the reproducible part.
