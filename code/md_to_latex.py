# -*- coding: utf-8 -*-
"""
Convert the JBI manuscript Markdown to a standalone LaTeX document.

This is a pragmatic converter for this manuscript (headings, bold, bullet
lists, pipe tables, numbered references). It is not a general Markdown parser.

Usage:
  python md_to_latex.py manuscript/JBI_submission_v1.md
"""
from __future__ import annotations

import argparse
import os
import re

SYMBOLS = {
    "κ": r"$\kappa$", "ρ": r"$\rho$", "α": r"$\alpha$",
    "≥": r"$\geq$", "≤": r"$\leq$", "×": r"$\times$",
    "−": r"$-$",
    "±": r"$\pm$", "–": "--", "—": "---",
    "’": "'", "“": "``", "”": "''", "→": r"$\rightarrow$",
}


def esc(text: str) -> str:
    text = text.replace("\\", r"\textbackslash{}")
    for ch in ("&", "%", "#", "_", "$"):
        text = text.replace(ch, "\\" + ch)
    for k, v in SYMBOLS.items():
        text = text.replace(k, v)
    text = re.sub(r"\*\*(.+?)\*\*", r"\\textbf{\1}", text)
    text = re.sub(r"`([^`]+)`", r"\\texttt{\1}", text)
    return text


def is_row(line: str) -> bool:
    return line.strip().startswith("|") and line.strip().endswith("|")


def row_cells(line: str) -> list[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def table_tex(header, rows, caption=None):
    n = len(header)
    spec = "@{}" + "Y" * n + "@{}"
    out = []
    out.append(r"\begin{table}[htbp]\centering\small")
    if caption:
        # caption is already escaped; strip the bold "Table N." prefix
        caption = re.sub(
            r"^\\textbackslash\{\}textbf\{(?:Supplementary )?Table [^}]+\.\}\s*",
            "", caption)
        caption = re.sub(r"^\\textbf\{(?:Supplementary )?Table [^}]+\.\}\s*",
                         "", caption)
        out.append(r"\caption{" + caption + "}")
    out.append(r"\begin{tabularx}{\textwidth}{" + spec + "}")
    out.append(r"\toprule")
    out.append(" & ".join(r"\textbf{" + esc(h) + "}" for h in header) + r" \\")
    out.append(r"\midrule")
    for row in rows:
        row = row + [""] * (n - len(row))
        out.append(" & ".join(esc(c) for c in row[:n]) + r" \\")
    out.append(r"\bottomrule")
    out.append(r"\end{tabularx}")
    out.append(r"\end{table}")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    lines = open(a.src, encoding="utf-8").read().splitlines()

    title = lines[0].lstrip("# ").strip()
    body_start = 1
    # skip draft quote lines
    while body_start < len(lines) and (not lines[body_start].strip()
                                       or lines[body_start].startswith(">")):
        body_start += 1
    text = lines[body_start:]

    out = [
        r"\documentclass[11pt]{article}",
        r"\usepackage[margin=2.2cm]{geometry}",
        r"\usepackage{booktabs,tabularx,array,graphicx,xcolor,hyperref}",
        r"\graphicspath{{../reports/figures/}}",
        r"\usepackage{enumitem}",
        r"\newcolumntype{Y}{>{\raggedright\arraybackslash}X}",
        r"\hypersetup{colorlinks=true,linkcolor=black,urlcolor=blue!60!black}",
        r"\setlist{nosep,leftmargin=1.4em}",
        r"\title{\bfseries " + esc(title) + "}",
        r"\author{[Authors to be inserted]\\[2pt]\small [Affiliations to be inserted]}",
        r"\date{}",
        r"\begin{document}",
        r"\maketitle",
    ]

    i = 0
    in_abstract = False
    abstract_lines = []
    refs = []
    in_refs = False
    buf = []

    def flush_para():
        nonlocal buf
        if buf:
            out.append(esc(" ".join(buf)).strip())
            out.append("")
            buf = []

    while i < len(text):
        line = text[i].rstrip()
        if not line.strip():
            flush_para()
            i += 1
            continue

        # references section
        if line.startswith("## References"):
            flush_para()
            in_refs = True
            i += 1
            continue
        if in_refs:
            if line.startswith("## "):
                in_refs = False      # fall through and handle the heading
            else:
                m = re.match(r"^\d+\.\s+(.*)$", line.strip())
                if m:
                    refs.append(m.group(1))
                i += 1
                continue

        if line.strip() == "**Highlights**":
            flush_para()
            out.append(r"\section*{Highlights}")
            out.append(r"\begin{itemize}[leftmargin=1.2em]")
            i += 1
            while i < len(text) and not text[i].strip():
                i += 1
            while i < len(text) and text[i].strip().startswith("- "):
                out.append(r"\item " + esc(text[i].strip()[2:]))
                i += 1
            out.append(r"\end{itemize}")
            out.append("")
            continue

        if line.startswith("**Keywords:**"):
            flush_para()
            kw = re.sub(r"^\*\*Keywords:\*\*\s*", "", line)
            out.append(r"\noindent\textbf{Keywords:} " + esc(kw))
            out.append("")
            i += 1
            continue

        if line.startswith("## Abstract"):
            flush_para()
            in_abstract = True
            i += 1
            continue
        if in_abstract:
            if line.startswith("## "):
                in_abstract = False
                out.append(r"\begin{abstract}")
                out.extend(abstract_lines)
                out.append(r"\end{abstract}")
                out.append("")
                continue
            if line.strip() == "**Keywords:**":
                continue
            abstract_lines.append(esc(line.strip()))
            i += 1
            continue

        # tables
        if is_row(line) and i + 1 < len(text) and re.match(
                r"^\|[\s:\-|]+\|?$", text[i + 1].strip()):
            flush_para()
            header = row_cells(line)
            rows = []
            j = i + 2
            while j < len(text) and is_row(text[j]):
                rows.append(row_cells(text[j]))
                j += 1
            caption = None
            idx = max((k for k, x in enumerate(out) if x.strip()), default=-1)
            if idx >= 0 and (
                    out[idx].startswith(r"\textbf{Table")
                    or out[idx].startswith(r"\textbf{Supplementary Table")):
                caption = out.pop(idx)
            out.append(table_tex(header, rows, caption))
            out.append("")
            i = j
            continue

        if line.startswith("### "):
            flush_para()
            head = re.sub(r"^\d+(\.\d+)*\s*", "", line[4:].strip())
            out.append(r"\subsection{" + esc(head) + "}")
            out.append("")
            i += 1
            continue
        if line.startswith("## "):
            flush_para()
            head = line[3:].strip()
            head = re.sub(r"^\d+\.\s*", "", head)
            unnumbered = head in ("Declarations", "Figure legends",
                                  "Supplementary material", "References")
            if unnumbered:
                out.append(r"\section*{" + esc(head) + "}")
            else:
                out.append(r"\section{" + esc(head) + "}")
            out.append("")
            i += 1
            continue
        if line.startswith("# "):
            i += 1
            continue

        if line.strip().startswith("**Table") and line.strip().endswith(".**") is False:
            pass

        if line.strip().startswith("- "):
            flush_para()
            out.append(r"\begin{itemize}[leftmargin=1.2em]")
            while i < len(text) and text[i].strip().startswith("- "):
                out.append(r"\item " + esc(text[i].strip()[2:]))
                i += 1
            out.append(r"\end{itemize}")
            out.append("")
            continue

        buf.append(line.strip())
        i += 1

    flush_para()
    if in_abstract and abstract_lines:
        out.append(r"\begin{abstract}")
        out.extend(abstract_lines)
        out.append(r"\end{abstract}")

    if refs:
        out.append(r"\begin{thebibliography}{99}")
        for r in refs:
            out.append(r"\bibitem{} " + esc(r))
        out.append(r"\end{thebibliography}")

    out.append(r"\end{document}")
    dst = a.out or os.path.splitext(a.src)[0] + ".tex"
    open(dst, "w", encoding="utf-8").write("\n".join(out))
    print("wrote", dst, f"({len(' '.join(out).split())} tokens)")


if __name__ == "__main__":
    main()
