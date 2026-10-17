"""Fetch the MMLU "all" test split (cais/mmlu) and the GSM8K test split (openai/gsm8k) and write the JSONL files
quality_eval.py reads, into eval/data/ (git-ignored). The subset the eval draws is fixed by its seed, so these files
must be the full splits (14,042 and 1,319 rows).

  pip install huggingface_hub pandas pyarrow
  python3 eval/prepare-data.py
"""
import json
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download

out = Path(__file__).resolve().parent / "data"
out.mkdir(exist_ok=True)
mmlu = pd.read_parquet(hf_hub_download("cais/mmlu", "all/test-00000-of-00001.parquet", repo_type="dataset"))
gsm = pd.read_parquet(hf_hub_download("openai/gsm8k", "main/test-00000-of-00001.parquet", repo_type="dataset"))
with open(out / "mmlu_all_test.jsonl", "w") as f:
    for r in mmlu.itertuples():
        f.write(json.dumps({"subject": r.subject, "question": r.question, "choices": list(r.choices),
                            "answer": int(r.answer)}) + "\n")
with open(out / "gsm8k_test.jsonl", "w") as f:
    for r in gsm.itertuples():
        f.write(json.dumps({"question": r.question, "answer": r.answer}) + "\n")
assert len(mmlu) == 14042 and len(gsm) == 1319, (len(mmlu), len(gsm))
print(f"wrote {out}/mmlu_all_test.jsonl ({len(mmlu)}) and gsm8k_test.jsonl ({len(gsm)})")
