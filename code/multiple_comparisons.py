# -*- coding: utf-8 -*-
"""
Holm correction across the pre-specified paired tests, by benchmark layer.

Families
--------
L1    : 2 models x 3 scopes (all / S1+S2 / S1) x 2 comparisons (RAG, oracle)
hard  : 2 models x 2 scopes (all / S1+S2) x 2 comparisons
L2    : 2 models x 2 tests (case sign test, trajectory-pass McNemar)
L3    : 2 models x 2 conditions (RAG, oracle) sign tests on paired rubric score

Raw p-values come from results/analysis_full.json, analysis_hard.json,
l2_analysis.json and l3_*_v3.summary.json.

Usage:
  python multiple_comparisons.py
"""
from __future__ import annotations

import json
import os

from scipy.stats import binomtest

ROOT = os.path.dirname(os.path.abspath(os.path.join(__file__, "..")))
RES = os.path.join(ROOT, "results")


def load(name):
    with open(os.path.join(RES, name), encoding="utf-8") as f:
        return json.load(f)


def holm(pvals):
    """Return adjusted p-values in the original order (Holm step-down)."""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        val = min(1.0, (m - rank) * pvals[i])
        running = max(running, val)
        adj[i] = running
    return adj


def collect():
    fams = {}

    a = load("analysis_full.json")
    rows = []
    for scope, key in (("all", "paired"), ("S1+S2", "paired_safety"),
                       ("S1", "paired_s1")):
        for p in a.get(key, []):
            rows.append((f"L1/{p['model']}/{scope}/{p['compare']}",
                         p["mcnemar_p"]))
    fams["L1"] = rows

    h = load("analysis_hard.json")
    rows = []
    for scope, key in (("all", "paired"), ("S1+S2", "paired_safety")):
        for p in h.get(key, []):
            rows.append((f"hard/{p['model']}/{scope}/{p['compare']}",
                         p["mcnemar_p"]))
    fams["hard"] = rows

    l2 = load("l2_analysis.json")["results"]
    rows = []
    for r in l2:
        rows.append((f"L2/{r['model']}/case-sign-test",
                     r["case_sign_test"]["p"]))
        rows.append((f"L2/{r['model']}/trajectory-pass",
                     r["trajectory_pass_mcnemar"]["p"]))
    fams["L2"] = rows

    rows = []
    for m, fn in (("deepseek-flash", "l3_flash_v3.summary.json"),
                  ("deepseek-v4-pro", "l3_v4pro_v3.summary.json")):
        d = load(fn)
        for cond in ("rag", "oracle"):
            st = d["conditions"][cond].get("sign_test") or {}
            pos, neg = st.get("positive", 0), st.get("negative", 0)
            p = binomtest(pos, pos + neg, 0.5).pvalue if (pos + neg) else 1.0
            rows.append((f"L3/{m}/{cond}-sign-test", float(p)))
    fams["L3"] = rows
    return fams


def main():
    fams = collect()
    out = {}
    lines = ["# Holm 多重比较校正", ""]
    for fam, rows in fams.items():
        pvals = [p for _, p in rows]
        adj = holm(pvals)
        out[fam] = [{"test": name, "p_raw": p, "p_holm": a,
                     "significant": a < 0.05}
                    for (name, p), a in zip(rows, adj)]
        lines += [f"## {fam}（{len(rows)} 个检验）", "",
                  "| test | p raw | p Holm | sig |", "|---|---|---|---|"]
        for r in out[fam]:
            lines.append(f"| {r['test']} | {r['p_raw']:.4g} | "
                         f"{r['p_holm']:.4g} | "
                         f"{'yes' if r['significant'] else 'no'} |")
        lines.append("")
    json.dump(out, open(os.path.join(RES, "holm.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    open(os.path.join(RES, "holm.md"), "w", encoding="utf-8").write(
        "\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
