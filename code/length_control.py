# -*- coding: utf-8 -*-
"""
Length-controlled sensitivity analysis.

RAG answers are longer than closed-book answers, so a raw accuracy gain can be
partly a verbosity effect. This script fits a logistic model with condition and
answer length, clustered by item (the same item appears in every condition), and
reports the condition effect before and after adjusting for length.

Also runs an OLS version for the L3 weighted rubric score.

Usage:
  python length_control.py
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

ROOT = os.path.dirname(os.path.abspath(os.path.join(__file__, "..")))
RES = os.path.join(ROOT, "results")


def load_rows(paths):
    rows = []
    for model, path in paths:
        for line in open(path, encoding="utf-8"):
            if not line.strip():
                continue
            r = json.loads(line)
            r["model"] = model
            rows.append(r)
    return pd.DataFrame(rows)


def fit(df, outcome, family):
    df = df.dropna(subset=[outcome, "answer_chars"]).copy()
    df["log_chars"] = np.log1p(df["answer_chars"])
    df["condition"] = pd.Categorical(df["condition"],
                                     categories=["closed", "rag", "oracle"])
    # unadjusted and adjusted models, both clustered by item
    m0 = smf.glm(f"{outcome} ~ C(condition) + C(model)", data=df,
                 family=family).fit(
        cov_type="cluster", cov_kwds={"groups": df["item_id"]})
    m1 = smf.glm(f"{outcome} ~ C(condition) + log_chars + C(model)", data=df,
                 family=family).fit(
        cov_type="cluster", cov_kwds={"groups": df["item_id"]})
    return m0, m1, df


def extract(m, family_name):
    out = {}
    for term in m.params.index:
        if "condition" in term:
            lab = term.split("[T.")[-1].rstrip("]")
            out[lab] = {
                "coef": round(float(m.params[term]), 4),
                "se": round(float(m.bse[term]), 4),
                "p": round(float(m.pvalues[term]), 6),
                "odds_ratio": (round(float(np.exp(m.params[term])), 3)
                               if family_name == "logit" else None),
            }
    if "log_chars" in m.params.index:
        out["log_chars"] = {
            "coef": round(float(m.params["log_chars"]), 4),
            "se": round(float(m.bse["log_chars"]), 4),
            "p": round(float(m.pvalues["log_chars"]), 6),
            "odds_ratio": (round(float(np.exp(m.params["log_chars"])), 3)
                           if family_name == "logit" else None),
        }
    return out


def main():
    paths = [("deepseek-flash", os.path.join(RES, "full_flash.jsonl")),
             ("deepseek-v4-pro", os.path.join(RES, "full_v4pro.jsonl"))]
    df = load_rows(paths)
    df["correct"] = df["correct"].astype(int)
    lengths = (df.groupby(["model", "condition"])["answer_chars"]
               .agg(["mean", "median"]).round(1).reset_index()
               .to_dict(orient="records"))

    m0, m1, d = fit(df, "correct", sm.families.Binomial())
    report = {
        "n_rows": int(len(d)),
        "lengths": lengths,
        "unadjusted": extract(m0, "logit"),
        "adjusted_for_length": extract(m1, "logit"),
    }

    # L3: OLS of weighted score on condition + answer length
    l3 = []
    for model, fn in (("deepseek-flash", "l3_flash_v3.jsonl"),
                      ("deepseek-v4-pro", "l3_v4pro_v3.jsonl")):
        for line in open(os.path.join(RES, fn), encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                r["model"] = model
                r["answer_chars"] = len(r.get("answer") or "")
                l3.append(r)
    l3 = pd.DataFrame(l3)
    if not l3.empty:
        l3["condition"] = pd.Categorical(l3["condition"],
                                         categories=["closed", "rag", "oracle"])
        l3["log_chars"] = np.log1p(l3["answer_chars"])
        try:
            o0 = smf.ols("weighted_score ~ C(condition) + C(model)",
                         data=l3).fit(cov_type="cluster",
                                      cov_kwds={"groups": l3["item_id"]})
            o1 = smf.ols("weighted_score ~ C(condition) + log_chars + C(model)",
                         data=l3).fit(cov_type="cluster",
                                      cov_kwds={"groups": l3["item_id"]})
            report["l3"] = {
                "unadjusted": extract(o0, "ols"),
                "adjusted_for_length": extract(o1, "ols"),
            }
        except Exception as e:  # noqa: BLE001
            report["l3"] = {"error": str(e)}

    json.dump(report, open(os.path.join(RES, "length_control.json"), "w",
                           encoding="utf-8"), ensure_ascii=False, indent=2)

    lines = ["# 长度控制敏感性分析", "",
             "## 平均答案长度（字符）", "",
             "| model | condition | mean | median |", "|---|---|---|---|"]
    for r in lengths:
        lines.append(f"| {r['model']} | {r['condition']} | "
                     f"{r['mean']:.0f} | {r['median']:.0f} |")
    lines += ["", "## 准确率：Logit（按题目聚类稳健 SE）", "",
              "| 模型 | 条件 | 未校正 OR (p) | 校正长度后 OR (p) |",
              "|---|---|---|---|"]
    for cond in ("rag", "oracle"):
        u = report["unadjusted"].get(cond, {})
        a = report["adjusted_for_length"].get(cond, {})
        lines.append(f"| 全部 | {cond} vs closed | "
                     f"{u.get('odds_ratio')} ({u.get('p')}) | "
                     f"{a.get('odds_ratio')} ({a.get('p')}) |")
    lc = report["adjusted_for_length"].get("log_chars", {})
    lines += ["", f"长度系数（每 log 字符）：OR={lc.get('odds_ratio')}, "
                  f"p={lc.get('p')}", ""]
    if "l3" in report and "adjusted_for_length" in report["l3"]:
        lines += ["## L3 加权得分：OLS", "",
                  "| 条件 | 未校正 β (p) | 校正长度后 β (p) |", "|---|---|---|"]
        for cond in ("rag", "oracle"):
            u = report["l3"]["unadjusted"].get(cond, {})
            a = report["l3"]["adjusted_for_length"].get(cond, {})
            lines.append(f"| {cond} vs closed | {u.get('coef')} ({u.get('p')}) | "
                         f"{a.get('coef')} ({a.get('p')}) |")
        lines.append("")
    open(os.path.join(RES, "length_control.md"), "w",
         encoding="utf-8").write("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
