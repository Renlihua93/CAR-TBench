# -*- coding: utf-8 -*-
"""
Export verified items to an expert-review workbook (one xlsx per task).

The workbook is what the 3 specialists (hematologist / CAR-T specialist /
clinical pharmacist) actually fill in. Layout is chosen so one item fits one
screen: stem, options, keyed answer, the verbatim guideline quote that
justifies it, then the reviewer columns.

Usage:
    python export_review.py --items data/items/verified_T3_crs.jsonl \
                            --out data/expert_forms/review_T3_crs.xlsx --lang zh
"""
from __future__ import annotations

import argparse
import json
import os

from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.datavalidation import DataValidation

COLS_ZH = [
    ("题号", 12), ("任务类型", 14), ("患者/题干", 60), ("选项", 60),
    ("参考答案", 8), ("参考答案依据（指南原文）", 55),
    ("安全关键", 9), ("错误答案可能造成的伤害", 30),
    ("答案是否正确？(是/否/有争议)", 16), ("若有误，正确答案应为", 14),
    ("临床合理性 1-5", 12), ("表述清晰度 1-5", 12),
    ("难度 易/中/难", 12), ("是否应保留 (保留/修改/删除)", 18),
    ("修改建议", 40),
]
COLS_EN = [
    ("Item ID", 12), ("Task", 14), ("Stem", 60), ("Options", 60),
    ("Key", 8), ("Evidence quote", 55), ("Safety-critical", 9),
    ("Harm if wrong", 30), ("Answer correct? (yes/no/disputed)", 16),
    ("Correct answer if wrong", 14), ("Clinical validity 1-5", 12),
    ("Clarity 1-5", 12), ("Difficulty easy/med/hard", 12),
    ("Keep? (keep/edit/drop)", 18), ("Suggested edit", 40),
]

INTRO = {
    "zh": [
        ("CAR-TBench 专家评审表", None),
        ("", None),
        ("请您独立完成，不要查阅他人评分。每题约需 40-60 秒。", None),
        ("评分定义：", None),
        ("  · 答案是否正确：仅依据您所在专科公认的指南/共识判断；如指南本身存在争议请选「有争议」并说明。", None),
        ("  · 临床合理性：1=临床上有害或不可接受；3=可接受但不理想；5=完全符合临床实践。", None),
        ("  · 表述清晰度：1=有歧义/无法作答；3=需读两遍；5=一次读清。", None),
        ("  · 安全关键：错误答案若被采信，可能直接导致患者伤害（如错误分级导致延误升压药/托珠单抗）。", None),
        ("  · 是否保留：保留=可直接用；修改=方向对但题干/选项/答案需改；删除=临床不成立或有第二个正确答案。", None),
        ("审阅表返回后，我们会按「多数一致 + 分歧仲裁」处理：任两位专家同意的意见为准，三人全不一致时由第三位专家（或通讯作者）裁定。", None),
    ],
    "en": [
        ("CAR-TBench expert review sheet", None),
        ("", None),
        ("Please rate independently; do not discuss items with the other reviewers.", None),
        ("Definitions", None),
        ("  Answer correct: judge by the guideline your specialty actually follows; use 'disputed' if the guideline is genuinely contested.", None),
        ("  Clinical validity: 1 = harmful/unacceptable; 3 = acceptable but suboptimal; 5 = fully concordant with practice.", None),
        ("  Clarity: 1 = ambiguous/unanswerable; 3 = needs a second read; 5 = unambiguous first pass.", None),
        ("  Safety-critical: a credibly wrong answer could directly harm a patient (e.g. mis-grading delays vasopressors/tocilizumab).", None),
        ("  Keep: keep = usable as is; edit = right idea, needs rewrite; drop = clinically invalid or second defensible answer.", None),
        ("Adjudication: majority of three reviewers; ties escalated to the senior author.", None),
    ],
}


def options_text(it: dict) -> str:
    return "\n".join(f"{o['key']}. {o['text']}" for o in it.get("options", []))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lang", default="zh")
    ap.add_argument("--reviewers", type=int, default=3)
    ap.add_argument("--mode", choices=["review", "baseline"], default="review",
                    help="baseline = blind answering form for the human baseline "
                         "(answer and evidence columns are withheld)")
    a = ap.parse_args()

    items = [json.loads(l) for l in open(a.items, encoding="utf-8") if l.strip()]
    os.makedirs(os.path.dirname(a.out), exist_ok=True)

    wb = Workbook()
    ws = wb.active
    ws.title = "封面 Cover"
    for i, (line, _) in enumerate(INTRO[a.lang], 1):
        c = ws.cell(row=i, column=1, value=line)
        if i == 1:
            c.font = Font(bold=True, size=14)

    cols = COLS_ZH if a.lang == "zh" else COLS_EN
    for r in range(a.reviewers):
        sheet = wb.create_sheet(f"R{r + 1}")
        sheet.freeze_panes = "C2"
        for j, (name, width) in enumerate(cols, 1):
            c = sheet.cell(row=1, column=j, value=name)
            c.font = Font(bold=True, color="FFFFFF")
            c.fill = PatternFill("solid", fgColor="2F5597")
            c.alignment = Alignment(wrap_text=True, vertical="center")
            sheet.column_dimensions[get_column_letter(j)].width = width
        sheet.row_dimensions[1].height = 42

        for i, it in enumerate(items, start=2):
            ev = it.get("evidence", [{}])[0]
            blind = a.mode == "baseline"
            vals = [
                it.get("item_id", ""), it.get("task_type", ""), it.get("stem", ""),
                options_text(it), "" if blind else it.get("answer", ""),
                f'{ev.get("source_id", "")} / {ev.get("section", "")}\n'
                f'{ev.get("quote", "")}',
                ("是" if it.get("safety_critical") else "否") if a.lang == "zh"
                else ("yes" if it.get("safety_critical") else "no"),
                it.get("safety_rationale", ""),
                "", "", "", "", "", "", "",
            ]
            for j, v in enumerate(vals, 1):
                c = sheet.cell(row=i, column=j, value=v)
                c.alignment = Alignment(wrap_text=True, vertical="top")
            sheet.row_dimensions[i].height = 92

        last = len(items) + 1
        dvs = [
            ("I", '"是,否,有争议"' if a.lang == "zh" else '"yes,no,disputed"'),
            ("K", '"1,2,3,4,5"'), ("L", '"1,2,3,4,5"'),
            ("M", '"易,中,难"' if a.lang == "zh" else '"easy,medium,hard"'),
            ("N", '"保留,修改,删除"' if a.lang == "zh" else '"keep,edit,drop"'),
        ]
        for col, formula in dvs:
            dv = DataValidation(type="list", formula1=formula, allow_blank=True)
            sheet.add_data_validation(dv)
            dv.add(f"{col}2:{col}{last}")

    wb.save(a.out)
    print(f"wrote {a.out}: {len(items)} items x {a.reviewers} reviewers")


if __name__ == "__main__":
    main()
