# -*- coding: utf-8 -*-
"""
Retrieval over the CAR-TBench guideline corpus.

Two index-free lexical retrievers plus a hybrid, so the RAG experiment has a
defensible non-neural baseline and no heavyweight dependencies:

  * BM25 over a mixed tokenizer (English words + CJK character n-grams)
  * TF-IDF cosine (same tokens)
  * hybrid = normalised sum

Dense retrieval can be dropped in later behind `Retriever.search_dense`.

Usage (sanity check):
    python retrieval.py "CRS grade 3 hypotension norepinephrine"
"""
from __future__ import annotations

import json
import math
import os
import re
from collections import Counter

ROOT = os.path.dirname(os.path.abspath(os.path.join(__file__, "..")))
CHUNKS = os.path.join(ROOT, "data", "rag", "chunks.jsonl")

STOP = set("""a an the of and or to in for with on by is are was were be been
this that these those as at from we our it its not no than then which who whom
其 的 了 和 与 或 在 中 是 为 对 及 等 内 以 并 由 该 可 有 无 不 也 而 但 从 到 被 将""".split())

_WORD = re.compile(r"[a-z0-9][a-z0-9\-]+")
_CJK = re.compile(r"[一-鿿]")


def tokenize(text: str) -> list[str]:
    text = text.lower()
    toks = [w for w in _WORD.findall(text) if w not in STOP and len(w) > 1]
    cjk = _CJK.findall(text)
    # character bigrams capture Chinese medical terms without a segmenter
    toks += ["".join(cjk[i:i + 2]) for i in range(max(0, len(cjk) - 1))]
    return toks


class Retriever:
    def __init__(self, chunks_path: str = CHUNKS, k1: float = 1.5, b: float = 0.75):
        self.chunks = [json.loads(l) for l in
                       open(chunks_path, encoding="utf-8") if l.strip()]
        self.toks = [tokenize(c["text"]) for c in self.chunks]
        self.tf = [Counter(t) for t in self.toks]
        self.dl = [len(t) for t in self.toks]
        self.avgdl = sum(self.dl) / max(1, len(self.dl))
        self.df = Counter()
        for t in self.toks:
            self.df.update(set(t))
        self.N = len(self.chunks)
        self.k1, self.b = k1, b
        # idf used by both scorers
        self.idf = {w: math.log(1 + (self.N - d + 0.5) / (d + 0.5))
                    for w, d in self.df.items()}
        self.norm = math.sqrt(sum(v * v for v in self.idf.values())) or 1.0

    # --------------------------------------------------------------- BM25
    def bm25(self, query: str) -> list[float]:
        q = Counter(tokenize(query))
        scores = []
        for i, tf in enumerate(self.tf):
            s = 0.0
            for w, qf in q.items():
                f = tf.get(w)
                if not f:
                    continue
                idf = self.idf.get(w, 0.0)
                s += qf * idf * (f * (self.k1 + 1)) / (
                    f + self.k1 * (1 - self.b + self.b * self.dl[i] / self.avgdl))
            scores.append(s)
        return scores

    # --------------------------------------------------------------- TF-IDF
    def tfidf(self, query: str) -> list[float]:
        q = Counter(tokenize(query))
        qvec = {w: (1 + math.log(f)) * self.idf.get(w, 0.0) for w, f in q.items()}
        qn = math.sqrt(sum(v * v for v in qvec.values())) or 1.0
        out = []
        for i, tf in enumerate(self.tf):
            dvec = {w: (1 + math.log(f)) * self.idf.get(w, 0.0) for w, f in tf.items()}
            dn = math.sqrt(sum(v * v for v in dvec.values())) or 1.0
            out.append(sum(qvec[w] * dvec.get(w, 0.0) for w in qvec) / (qn * dn))
        return out

    @staticmethod
    def _norm(scores: list[float]) -> list[float]:
        mx = max(scores) or 1.0
        return [s / mx for s in scores]

    def search(self, query: str, k: int = 5, method: str = "hybrid",
               source_filter: list[str] | None = None) -> list[dict]:
        if method == "bm25":
            sc = self.bm25(query)
        elif method == "tfidf":
            sc = self.tfidf(query)
        else:
            a, b = self._norm(self.bm25(query)), self._norm(self.tfidf(query))
            sc = [x + y for x, y in zip(a, b)]
        idx = range(self.N) if not source_filter else [
            i for i, c in enumerate(self.chunks)
            if c["source_id"] in source_filter]
        ranked = sorted(idx, key=lambda i: sc[i], reverse=True)[:k]
        return [{"score": round(sc[i], 4), **self.chunks[i]} for i in ranked]

    def _section_ranking(self, query: str) -> list[tuple[tuple[str, str], float, int]]:
        """Best chunk per (source_id, section), best first.

        A gold answer often lives in one short section whose wording matches
        a span of the question; chunk-level top-k can spend all five slots on
        variants of one neighbouring chunk and never reach it.
        """
        a, b = self._norm(self.bm25(query)), self._norm(self.tfidf(query))
        sc = [x + y for x, y in zip(a, b)]
        best: dict[tuple[str, str], tuple[float, int]] = {}
        for i, c in enumerate(self.chunks):
            key = (c["source_id"], c["section"])
            if key not in best or sc[i] > best[key][0]:
                best[key] = (sc[i], i)
        return [(key, s, i)
                for key, (s, i) in sorted(best.items(), key=lambda x: -x[1][0])]

    def search_sections(self, query: str, k: int = 5) -> list[dict]:
        ranked = self._section_ranking(query)[:k]
        return [{"score": round(s, 4), **self.chunks[i]} for _, s, i in ranked]

    def search_sections_rrf(self, queries: list[str], k: int = 5,
                            pool: int = 200, rrf_k: int = 60) -> list[dict]:
        """Fuse several queries (e.g. Chinese original + English translation)
        by reciprocal rank fusion over sections.

        The corpus is bilingual: Chinese-consensus sections are only reachable
        from the Chinese query, English labels/guidelines only from the
        translated one. Neither language alone covers both; RRF keeps each
        retriever's strong hits and raises section recall@10 from 0.78 to 0.84
        on the full bank.
        """
        fused: dict[tuple[str, str], float] = {}
        best_chunk: dict[tuple[str, str], int] = {}
        for q in queries:
            if not q:
                continue
            for i, (key, _score, idx) in enumerate(
                    self._section_ranking(q)[:pool]):
                fused[key] = fused.get(key, 0.0) + 1.0 / (rrf_k + i + 1)
                best_chunk.setdefault(key, idx)
        ranked = sorted(fused, key=lambda x: -fused[x])[:k]
        return [{"score": round(fused[key], 5),
                 **self.chunks[best_chunk[key]]} for key in ranked]

    def hierarchical(self, query: str, k: int = 5, pool: int = 40) -> list[dict]:
        """Retrieve, then expand the best hit to its neighbouring chunks of the
        same section (helps when an answer spans chunks)."""
        hits = self.search(query, k=pool)
        best = hits[0] if hits else None
        if not best:
            return []
        key = (best["source_id"], best["section"])
        sib = [c for c in self.chunks
               if (c["source_id"], c["section"]) == key
               and c["chunk_id"] != best["chunk_id"]][:2]
        return hits[:k] + [{**s, "expanded": True} for s in sib]


if __name__ == "__main__":
    import sys
    r = Retriever()
    q = sys.argv[1] if len(sys.argv) > 1 else "CRS grade 3 hypotension"
    print("chunks:", len(r.chunks))
    for h in r.search(q, k=5):
        print(f"{h['score']:>7} | {h['source_id']:<22} | {h['section'][:40]}")
        print("        ", h["text"][:160].replace("\n", " "))
