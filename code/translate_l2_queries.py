# -*- coding: utf-8 -*-
"""
English retrieval queries for L2 trajectory steps.

Same problem as L1: the corpus is English while the case states are Chinese,
so raw Chinese queries retrieve almost nothing. Each step's state + question +
options is translated once, stored on disk, and reused by eval_trajectory's RAG
condition (which fuses the Chinese original and the English query by RRF).

Usage:
  python translate_l2_queries.py --cases data/items/l2_trajectories_v2.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from llm_client import LLM                        # noqa: E402

ROOT = os.path.dirname(os.path.abspath(os.path.join(__file__, "..")))
OUT = os.path.join(ROOT, "data", "rag", "l2_queries_en.jsonl")

SYS = """Translate the following Chinese CAR-T bedside scenario step into ONE
concise English search query using standard haematology terminology (CRS,
ICANS, ICE score, tocilizumab, dexamethasone, product labels, ASTCT grading).
Keep drug and product names. Do not answer the question. Output only the query."""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default=os.path.join(
        ROOT, "data", "items", "l2_trajectories_v2.jsonl"))
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--model", default="deepseek-flash")
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()

    cases = [json.loads(l) for l in open(a.cases, encoding="utf-8") if l.strip()]
    cases = [c for c in cases if c.get("status") in (None, "verified")]
    done = {}
    if os.path.exists(a.out):
        for l in open(a.out, encoding="utf-8"):
            if l.strip():
                r = json.loads(l)
                done[(r["case_id"], r["step"])] = r

    llm = LLM(a.model, temperature=0.0, max_tokens=4096)
    jobs = []
    for c in cases:
        for s in c["steps"]:
            key = (c["case_id"], s["step"])
            if key in done:
                continue
            q = (s["state"] + "\n" + s["question"] + "\n" +
                 " ".join(o["text"] for o in s["options"]))
            jobs.append((c, s, q))
    print(f"{len(jobs)} steps to translate")

    def one(job):
        c, s, q = job
        try:
            tr = llm.chat([{"role": "system", "content": SYS},
                           {"role": "user", "content": q}]).text.strip()
        except Exception as e:  # noqa: BLE001
            print(f"  {c['case_id']}#{s['step']}: {e}")
            return None
        return {"case_id": c["case_id"], "step": s["step"],
                "query_en": tr, "model": llm.model_id,
                "ts": datetime.now(timezone.utc).isoformat(timespec="seconds")}

    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        rows = [r for r in ex.map(one, jobs) if r]
    with open(a.out, "a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} -> {a.out}")


if __name__ == "__main__":
    main()
