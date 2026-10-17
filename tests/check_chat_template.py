"""Check a Jinja chat template against DeepSeek-V4.1-Flash-1017's own golden encodings.

The 1017 release has no chat_template.jinja; its prompt format is defined by encoding/encoding.py and five golden
input/output pairs in encoding/tests. This renders each golden input through the template the way TensorFold's server
does (Hugging Face-style sandbox, tool-call arguments parsed from JSON strings first) and compares byte for byte.

  pip install jinja2
  python3 tests/check_chat_template.py                      # the pinned official V4.1 template (expect 5/5)
  python3 tests/check_chat_template.py path/to/template.jinja
Standard library + jinja2; downloads a few small files from Hugging Face.
"""
import json
import sys
import urllib.request

import jinja2.exceptions
import jinja2.ext
from jinja2.sandbox import ImmutableSandboxedEnvironment

BASE = "https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/resolve/1bae0e1088993b9b82de08f59de5be5a34e42170"
TEMPLATE = ("https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/resolve/"
            "314113f8a7dadb388215396ba5fbee2c610a04d6/chat_template.jinja")


def get(url):
    with urllib.request.urlopen(url, timeout=60) as r:
        return r.read().decode()


source = open(sys.argv[1]).read() if len(sys.argv) > 1 else get(TEMPLATE)
cfg = json.loads(get(f"{BASE}/tokenizer_config.json"))
specials = {k: (v.get("content") if isinstance(v, dict) else v)
            for k, v in cfg.items() if k in ("bos_token", "eos_token", "pad_token", "unk_token")}
env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=[jinja2.ext.loopcontrols])
env.filters["tojson"] = lambda x, ensure_ascii=False, indent=None, separators=None, sort_keys=False: json.dumps(
    x, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)
def raise_exception(message):
    raise jinja2.exceptions.TemplateError(message)


env.globals["raise_exception"] = raise_exception
template = env.from_string(source)

passed = 0
for i in range(1, 6):
    case = json.loads(get(f"{BASE}/encoding/tests/test_input_{i}.json"))
    case = case if isinstance(case, dict) else {"messages": case}
    gold = get(f"{BASE}/encoding/tests/test_output_{i}.txt")
    messages = json.loads(json.dumps(case["messages"]))
    for m in messages:                       # as the server does: tool-call arguments as objects, not JSON strings
        for tc in m.get("tool_calls") or []:
            f = tc.get("function", tc)
            if isinstance(f.get("arguments"), str):
                try:
                    f["arguments"] = json.loads(f["arguments"])
                except ValueError:
                    pass
    mode = case.get("thinking_mode", "chat")           # the encoder's test runner defaults to chat mode
    kw = dict(specials, messages=messages, tools=case.get("tools"), thinking_mode=mode, enable_thinking=mode == "thinking")
    if "reasoning_effort" in case:
        kw["reasoning_effort"] = case["reasoning_effort"]
    outs = [template.render(add_generation_prompt=agp, **kw) for agp in (False, True)]
    if gold in outs:
        passed += 1
        print(f"case {i} ({mode}): match")
    else:
        out = outs[0]
        n = next((j for j, (a, b) in enumerate(zip(out, gold)) if a != b), min(len(out), len(gold)))
        print(f"case {i} ({mode}): DIFF at char {n}: got {out[max(0, n - 40):n + 40]!r} want {gold[max(0, n - 40):n + 40]!r}")
print(f"{passed}/5 golden encodings reproduced")

# TensorFold's image path: the server first replaces each image part with DeepSeek's placeholder token and joins the
# parts into one string with blank lines (vision.split), renders that, then grows each placeholder into the image's
# span. The template must accept the placeholder in string content and render the same prompt as for image blocks.
IMG = "<｜deepseek_image｜>"
image_ok = True
for i in range(1, 6):
    case = json.loads(get(f"{BASE}/encoding/tests/test_input_{i}.json"))
    case = case if isinstance(case, dict) else {"messages": case}
    if not any(isinstance(m.get("content"), list) and any(p.get("type") in ("image", "image_url") for p in m["content"])
               for m in case["messages"]):
        continue
    gold = get(f"{BASE}/encoding/tests/test_output_{i}.txt")
    messages = []
    for m in case["messages"]:
        c = m.get("content")
        if isinstance(c, list):
            m = {**m, "content": "\n\n".join(IMG if p.get("type") in ("image", "image_url") else p.get("text", "") for p in c)}
        messages.append(m)
    mode = case.get("thinking_mode", "chat")
    kw = dict(specials, messages=messages, tools=case.get("tools"), thinking_mode=mode, enable_thinking=mode == "thinking")
    if "reasoning_effort" in case:
        kw["reasoning_effort"] = case["reasoning_effort"]
    try:
        outs = [template.render(add_generation_prompt=agp, **kw) for agp in (False, True)]
        result = "match" if gold in outs else "DIFF"
    except Exception as e:  # noqa: BLE001 - the official template raises on placeholders in string content
        result = f"rejected ({type(e).__name__}: {str(e)[:80]})"
    image_ok &= result == "match"
    print(f"case {i} via TensorFold's image placeholders: {result}")
sys.exit(0 if passed == 5 and image_ok else 1)
