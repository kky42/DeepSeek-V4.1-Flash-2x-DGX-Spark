"""Correctness smoke for the DS-V4.1-Flash-1017 server: arithmetic, recall, code, thinking on, a tool call, an image."""
import base64
import json
import struct
import sys
import time
import urllib.request
import zlib

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8888"   # e.g. python3 tests/smoke.py http://HEAD:8888
MODEL = sys.argv[2] if len(sys.argv) > 2 else "deepseek-v4.1-flash"


def ask(body, label):
    body.setdefault("model", MODEL)
    body.setdefault("temperature", 0)
    t = time.time()
    req = urllib.request.Request(f"{BASE}/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    r = json.load(urllib.request.urlopen(req, timeout=600))
    m, u = r["choices"][0]["message"], r.get("usage", {})
    print(f"--- {label}  ({time.time() - t:.1f}s, {u.get('completion_tokens')} tok, "
          f"finish={r['choices'][0]['finish_reason']})")
    if m.get("reasoning_content"):
        print("  [reasoning]", m["reasoning_content"][:300].replace("\n", " "), "...")
    if m.get("content"):
        print("  [content]", m["content"][:600])
    if m.get("tool_calls"):
        print("  [tool_calls]", json.dumps(m["tool_calls"])[:400])


OFF = {"enable_thinking": False}
ask({"messages": [{"role": "user", "content": "What is 19 + 23? Answer with just the number."}],
     "max_tokens": 16, "chat_template_kwargs": OFF}, "arithmetic, thinking off")
ask({"messages": [{"role": "user", "content": "Name the capital of Australia and the year it became the capital. One sentence."}],
     "max_tokens": 64, "chat_template_kwargs": OFF}, "fact, thinking off")
ask({"messages": [{"role": "user", "content": "Write a Python function is_prime(n) that is correct and efficient. Code only."}],
     "max_tokens": 300, "chat_template_kwargs": OFF}, "code, thinking off")
ask({"messages": [{"role": "user", "content": "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. "
                                               "How much is the ball? Give the final answer in cents."}],
     "max_tokens": 2048, "chat_template_kwargs": {"enable_thinking": True}, "reasoning_effort": "low"},
    "reasoning, thinking on (effort low)")
TOOLS = [{"type": "function", "function": {
    "name": "get_weather", "description": "Get the current weather for a city",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"},
                                                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}},
                   "required": ["city"]}}}]
ask({"messages": [{"role": "user", "content": "What is the weather in Paris right now, in celsius?"}],
     "tools": TOOLS, "max_tokens": 256, "chat_template_kwargs": OFF}, "tool call, thinking off")


def two_color_png(size=64):
    """A PNG whose left half is red and right half blue (no image library needed)."""
    row = b"\x00" + b"".join(b"\xff\x00\x00" if x < size // 2 else b"\x00\x00\xff" for x in range(size))

    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xffffffff)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(row * size)) + chunk(b"IEND", b""))


ask({"messages": [{"role": "user", "content": [
        {"type": "text", "text": "What two colors are in this image, and which is on the left? One sentence."},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(two_color_png()).decode()}}]}],
     "max_tokens": 64, "chat_template_kwargs": OFF}, "image (expect red on the left, blue on the right)")
