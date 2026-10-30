# -*- coding: utf-8 -*-
"""
Retry adversarial verification for items whose verify call died on the wire.

`pending_*.jsonl` holds items that were drafted and structurally fine but whose
verify call failed at the API (empty content / truncation), not a quality
verdict. They keep `_section_text`, so they can be re-verified without paying
for the draft again.

Usage:
  python verify_pending.py --workers 2
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gen_items import verify_one                     # noqa: E402
from llm_client import LLM                           # noqa: E402
from taxonomy import describe_codes, is_safety_critical, severity_of  # noqa: E402

ROOT = os.path.dirname(os.path.abspath(os.path.join(__file__, "..")))
ITEMS = os.path.join(ROOT, "data", "items")


def next_id(task: str) -> int:
    p = os.path.join(ITEMS, f"verified_{task}.jsonl")
    n = 0
    if os.path.exists(p):
        for line in open(p, encoding="utf-8"):
            if line.strip():
                iid = json.loads(line)["item_id"]
                n = max(n, int(iid.rsplit("-", 1)[-1]))
    return n + 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items-dir", default=ITEMS)
    ap.add_argument("--verify-model", default="deepseek-v4-pro")
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()

    verifier = LLM(a.verify_model, temperature=0.0, max_tokens=8000)
    paths = sorted(glob.glob(os.path.join(a.items_dir, "pending_*.jsonl")))
    for p in paths:
        task = os.path.basename(p)[len("pending_"):-len(".jsonl")]
        rows = [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
        rows = [r for r in rows if r.get("_section_text")]
        if not rows:
            continue
        print(f"{task}: retrying {len(rows)}")

        with ThreadPoolExecutor(max_workers=a.workers) as ex:
            verdicts = list(ex.map(lambda it: verify_one(verifier, it), rows))

        kept, pending, rejected = [], [], []
        for it, v in zip(rows, verdicts):
            it["review"] = {"status": "draft", "verifier": v}
            it["provenance"] = {**(it.get("provenance") or {}),
                                "verify_model": a.verify_model,
                                "verified_at": datetime.now(timezone.utc)
                                .isoformat(timespec="seconds")}
            if v.get("pass") is None:
                it["status"] = "needs_verify"
                pending.append(it)
            elif v.get("pass") and not v.get("fatal"):
                it["status"] = "verified"
                codes = it.get("failure_codes") or []
                it["safety_critical"] = bool(
                    is_safety_critical(codes)
                    or (v.get("safety_critical", False)
                        and it.get("action_type") != "not_applicable"))
                it["harm_severity"] = severity_of(codes)
                if it["safety_critical"] and not it.get("safety_rationale"):
                    it["safety_rationale"] = describe_codes(codes) or \
                        "错误答案可能导致临床处置偏差，需专家判定。"
                kept.append(it)
            else:
                it["status"] = "rejected"
                rejected.append({"reason": "verifier", **it})

        start = next_id(task)
        for n, it in enumerate(kept, start):
            it["item_id"] = f"CARB-{task.split('_')[0]}-{n:04d}"
        for it in kept:
            it.pop("_section_text", None)
        for it in rejected:
            it.pop("_section_text", None)

        def append(name: str, rows_: list[dict]):
            with open(os.path.join(a.items_dir, name), "a", encoding="utf-8") as f:
                for r in rows_:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")

        append(f"verified_{task}.jsonl", kept)
        append(f"rejected_{task}.jsonl", rejected)
        # remaining pending are still infra failures
        with open(p, "w", encoding="utf-8") as f:
            for r in pending:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"  verified {len(kept)}, still pending {len(pending)}, "
              f"rejected {len(rejected)}")


if __name__ == "__main__":
    main()
