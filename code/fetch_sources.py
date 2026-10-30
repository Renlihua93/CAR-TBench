# -*- coding: utf-8 -*-
"""
Fetch and stage authoritative CAR-T sources into data/sources/.

Design notes
------------
* Every source is recorded in data/sources/manifest.json with: title, citation,
  URL, access date, license/terms note, local file, sha256.
* `redistributable` marks whether the raw text may go into a PUBLIC benchmark
  release. ASTCT/NCCN/Chinese consensus full text may NOT be redistributed;
  items derived from them are fact-level and are released with citation only.
* Idempotent: skips files that already exist unless --force.

Usage:  python fetch_sources.py [--force]
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request
from datetime import date

ROOT = os.path.dirname(os.path.abspath(os.path.join(__file__, "..")))
SRC = os.path.join(ROOT, "data", "sources")
os.makedirs(SRC, exist_ok=True)
MANIFEST = os.path.join(SRC, "manifest.json")

UA = "CAR-TBench/0.1 (academic benchmark construction; contact: local)"
TODAY = date.today().isoformat()

# --- FDA labels (US government work / SPL: public domain) -------------------
LABELS = [
    ("kymriah", "KYMRIAH", "tisagenlecleucel"),
    ("yescarta", "YESCARTA", "axicabtagene ciloleucel"),
    ("breyanzi", "BREYANZI", "lisocabtagene maraleucel"),
    ("abecma", "ABECMA", "idecabtagene vicleucel"),
    ("carvykti", "CARVYKTI", "ciltacabtagene autoleucel"),
    ("tecartus", "TECARTUS", "brexucabtagene autoleucel"),
]

# --- PMC / Europe PMC documents --------------------------------------------
PMC_DOCS = [
    ("astct2019", "PMC12180426",
     "Lee DW, et al. ASTCT Consensus Grading for Cytokine Release Syndrome and "
     "Neurologic Toxicity Associated with Immune Effector Cells. Biol Blood Marrow "
     "Transplant. 2019;25(4):625-638.", "html"),
    ("ebmt2020", "PMC7012497",
     "Yakoub-Agha I, et al. Management of adults and children undergoing CAR T-cell "
     "therapy: best practice recommendations of the EBMT and JACIE. J Clin Oncol. "
     "2020;38(16):1852-1870.", "xml"),
    ("cn_mm_consensus2022", "PMC9189483",
     "Chinese consensus for the CAR-T cell therapy in multiple myeloma (2022 version).",
     "xml"),
]

# --- Pivotal trial abstracts (facts only, cited) ----------------------------
TRIALS = [
    ("zuma1", "10.1056/NEJMoa1707447", "ZUMA-1 (axi-cel, r/r LBCL)"),
    ("juliet", "10.1056/NEJMoa1804980", "JULIET (tisa-cel, r/r DLBCL)"),
    ("transcend", "10.1016/S0140-6736(20)31366-0", "TRANSCEND NHL-001 (liso-cel)"),
    ("eliana", "10.1056/NEJMoa1709866", "ELIANA (tisa-cel, pediatric ALL)"),
    ("karmma", "10.1056/NEJMoa2024850", "KarMMa (ide-cel, r/r MM)"),
    ("cartitude1", "10.1016/S0140-6736(21)00933-8", "CARTITUDE-1 (cilta-cel, r/r MM)"),
    ("zuma7", "10.1056/NEJMoa2116133", "ZUMA-7 (axi-cel 2L LBCL)"),
    ("transform", "10.1016/S0140-6736(22)01200-4", "TRANSFORM (liso-cel 2L LBCL)"),
]


def http_get(url: str, timeout: int = 60) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def html_to_text(html: str) -> str:
    t = re.sub(r"<script.*?</script>", "", html, flags=re.S | re.I)
    t = re.sub(r"<style.*?</style>", "", t, flags=re.S | re.I)
    t = re.sub(r"<[^>]+>", " ", t)
    t = t.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n\s*\n+", "\n", t)
    return t.strip()


def jats_to_text(xml: str) -> str:
    """Coarse JATS -> text, keeping section titles and paragraph breaks."""
    xml = re.sub(r"<ref-list.*?</ref-list>", " ", xml, flags=re.S)
    xml = re.sub(r"<back>.*?</back>", " ", xml, flags=re.S)
    xml = re.sub(r"<(title|p|th|td|caption)\b[^>]*>", "\n", xml)
    xml = re.sub(r"</(tr)>", "\n", xml)
    xml = re.sub(r"<[^>]+>", " ", xml)
    xml = xml.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    xml = re.sub(r"[ \t]+", " ", xml)
    xml = re.sub(r"\n\s*\n+", "\n", xml)
    return xml.strip()


def load_manifest() -> dict:
    if os.path.exists(MANIFEST):
        with open(MANIFEST, encoding="utf-8") as f:
            return json.load(f)
    return {"sources": []}


def record(man: dict, entry: dict) -> None:
    man["sources"] = [s for s in man["sources"] if s["id"] != entry["id"]] + [entry]
    with open(MANIFEST, "w", encoding="utf-8") as f:
        json.dump(man, f, ensure_ascii=False, indent=2)


def save(force: bool, man: dict, sid: str, fname: str, url: str, text: str,
         **meta) -> None:
    path = os.path.join(SRC, fname)
    if os.path.exists(path) and not force and not text:
        print(f"  skip {fname}")
        return
    if text:
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
    entry = {
        "id": sid,
        "file": fname,
        "url": url,
        "accessed": TODAY,
        "sha256": sha256(path),
        "chars": len(text or ""),
        **meta,
    }
    record(man, entry)
    print(f"  ok   {fname} ({entry['chars']} chars)")


def main(force: bool = False) -> None:
    man = load_manifest()

    print("[1/4] ASTM/EBMT/Chinese consensus full texts")
    for sid, pmc, cite, fmt in PMC_DOCS:
        fname = f"{sid}.txt"
        if os.path.exists(os.path.join(SRC, fname)) and not force:
            print(f"  skip {fname}")
            continue
        text = ""
        if fmt == "xml":
            try:
                raw = http_get(
                    f"https://www.ebi.ac.uk/europepmc/webservices/rest/{pmc}/fullTextXML")
                text = jats_to_text(raw.decode("utf-8", "replace"))
            except Exception as e:  # noqa: BLE001
                print(f"  WARN europepmc failed for {pmc}: {e}")
        if not text:
            raw = http_get(f"https://pmc.ncbi.nlm.nih.gov/articles/{pmc}/")
            text = html_to_text(raw.decode("utf-8", "replace"))
        save(force, man, sid, fname, f"https://pmc.ncbi.nlm.nih.gov/articles/{pmc}/",
             text, title=cite, pmcid=pmc,
             license_note="Full text retained locally for research; NOT redistributable",
             redistributable=False)
        time.sleep(1.0)

    print("[2/4] FDA product labels (public domain)")
    for sid, brand, generic in LABELS:
        fname = f"fda_{sid}.txt"
        if os.path.exists(os.path.join(SRC, fname)) and not force:
            print(f"  skip {fname}")
            continue
        meta = urllib.parse.urlencode({"drug_name": brand, "pagesize": "5"})
        try:
            js = json.loads(http_get(
                f"https://dailymed.nlm.nih.gov/dailymed/services/v2/spls.json?{meta}"))
            items = js.get("data") or []
            if not items:
                print(f"  WARN no label for {brand}")
                continue
            setid = items[0]["setid"]
            title = items[0]["title"]
            raw = http_get(
                f"https://dailymed.nlm.nih.gov/dailymed/services/v2/spls/{setid}.xml")
            text = html_to_text(raw.decode("utf-8", "replace"))
        except Exception as e:  # noqa: BLE001
            print(f"  WARN label {brand}: {e}")
            continue
        save(force, man, sid, fname,
             f"https://dailymed.nlm.nih.gov/dailymed/services/v2/spls/{setid}.xml",
             text, title=title, brand=brand, generic=generic, setid=setid,
             license_note="FDA SPL: public government work", redistributable=True)
        time.sleep(0.5)

    print("[3/4] Pivotal trial records (abstract-level facts)")
    for sid, doi, label in TRIALS:
        fname = f"trial_{sid}.json"
        if os.path.exists(os.path.join(SRC, fname)) and not force:
            print(f"  skip {fname}")
            continue
        q = urllib.parse.quote(f'DOI:"{doi}"')
        try:
            js = json.loads(http_get(
                "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
                f"?query={q}&format=json&resultType=core&pageSize=1"))
            res = js["resultList"]["result"]
            if not res:
                print(f"  WARN no record for {doi}")
                continue
            r = res[0]
            rec = {k: r.get(k) for k in
                   ("pmid", "pmcid", "title", "authorString", "journalTitle",
                    "pubYear", "doi", "abstractText", "isOpenAccess")}
            rec.update({"id": sid, "label": label, "accessed": TODAY,
                        "license_note": "metadata/abstract only",
                        "redistributable": False})
            with open(os.path.join(SRC, fname), "w", encoding="utf-8") as f:
                json.dump(rec, f, ensure_ascii=False, indent=2)
            record(man, {"id": sid, "file": fname,
                         "url": f"https://doi.org/{doi}", "accessed": TODAY,
                         "sha256": sha256(os.path.join(SRC, fname)),
                         "chars": len(rec.get("abstractText") or ""),
                         "title": rec["title"], "accessed_via": "europepmc",
                         "license_note": "metadata/abstract only",
                         "redistributable": False})
            print(f"  ok   {fname}")
        except Exception as e:  # noqa: BLE001
            print(f"  WARN trial {sid}: {e}")
        time.sleep(0.5)

    print("[4/4] done ->", MANIFEST)


if __name__ == "__main__":
    main("--force" in sys.argv)
