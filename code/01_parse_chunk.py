"""Step 1: PDFs -> clean page text -> overlapping chunks with metadata.

    python 01_parse_chunk.py

Reads data/pdfs/*.pdf and data/pdfs/manifest.csv, writes data/chunks.jsonl and
results/corpus_stats.json.
"""
import csv
import json
import os
import re
import time
from collections import Counter

import fitz  # PyMuPDF
from transformers import AutoTokenizer

from common import (CHUNK_OVERLAP, CHUNK_TOKENS, CHUNKS, EMBEDDERS, DEFAULT_EMBEDDER, PDFS, RESULTS,
                    access_for, write_jsonl)


def load_manifest():
    path = os.path.join(PDFS, "manifest.csv")
    rows = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                rows[r["file"]] = r
    return rows


def clean_page(text):
    text = text.replace("­", "")                        # soft hyphens
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)             # words split across lines
    lines = [l.strip() for l in text.splitlines()]
    out = []
    for l in lines:
        if not l or re.fullmatch(r"(page )?\d{1,3}( of \d+)?", l.lower()):
            continue                                         # blank lines and bare page numbers
        out.append(l)
    text = "\n".join(out)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def strip_repeated_lines(pages):
    """Drop running headers/footers: lines that appear on more than 40% of a document's pages."""
    cnt = Counter(l for p in pages for l in set(p.splitlines()) if len(l) < 120)
    common = {l for l, c in cnt.items() if len(pages) >= 5 and c / len(pages) > 0.4}
    return ["\n".join(l for l in p.splitlines() if l not in common) for p in pages]


BOILERPLATE = re.compile(
    r"Creative Commons (licence|Attribution)|Cataloguing data|Disclaimer|Professional independence|"
    r"acknowledge the continuous connection|Ownership of intellectual property|This report was prepared by|"
    r"Acknowledgements|ISSN \d{4}", re.I)


def is_boilerplate(chunk):
    if len(re.findall(r"\.{6,}", chunk)) >= 3:                 # table of contents / list of figures
        return True
    return len(BOILERPLATE.findall(chunk)) >= 1 and not re.search(r"\d+(\.\d+)? (million )?tonnes|per cent", chunk)


def chunk_page(text, tok):
    """Split one page into ~CHUNK_TOKENS token windows, preferring paragraph boundaries."""
    paras = [p for p in re.split(r"\n(?=[A-Z0-9•\-])", text) if p.strip()]
    chunks, cur, cur_len = [], [], 0
    for p in paras:
        n = len(tok.tokenize(p))
        if n > CHUNK_TOKENS:                                  # a very long paragraph: hard-split by tokens
            ids = tok(p, add_special_tokens=False)["input_ids"]
            for i in range(0, len(ids), CHUNK_TOKENS - CHUNK_OVERLAP):
                chunks.append(tok.decode(ids[i:i + CHUNK_TOKENS]))
            continue
        if cur_len + n > CHUNK_TOKENS and cur:
            chunks.append("\n".join(cur))
            # carry the tail of the previous chunk forward as overlap
            tail, tl = [], 0
            for q in reversed(cur):
                tl += len(tok.tokenize(q))
                tail.insert(0, q)
                if tl >= CHUNK_OVERLAP:
                    break
            cur, cur_len = tail, tl
        cur.append(p)
        cur_len += n
    if cur:
        chunks.append("\n".join(cur))
    return [c.strip() for c in chunks if len(c.split()) >= 25 and not is_boilerplate(c)]   # skip fragments (captions, headers)


MONTHS = "January February March April May June July August September October November December".split()


def make_header(title, date, page):
    """Contextual chunk header: tells the embedder and the reranker which report and edition a chunk is from."""
    pub = ""
    if re.fullmatch(r"\d{4}-\d{2}", date or ""):
        pub = f", published {MONTHS[int(date[5:]) - 1]} {date[:4]}"
    return f"{title}{pub}, page {page}"


def main():
    tok = AutoTokenizer.from_pretrained(EMBEDDERS[DEFAULT_EMBEDDER])
    man = load_manifest()
    files = sorted(f for f in os.listdir(PDFS) if f.lower().endswith(".pdf"))
    rows, stats, t0 = [], [], time.time()
    for f in files:
        m = man.get(f, {})
        series = m.get("series") or f.split("_20")[0]
        doc = fitz.open(os.path.join(PDFS, f))
        pages = strip_repeated_lines([clean_page(p.get_text("text")) for p in doc])
        n_before = len(rows)
        for pno, text in enumerate(pages, start=1):
            for j, c in enumerate(chunk_page(text, tok)):
                rows.append({
                    "id": f"{os.path.splitext(f)[0]}_p{pno}_{j}",
                    "file": f,
                    "title": m.get("title") or os.path.splitext(f)[0],
                    "publisher": m.get("publisher", ""),
                    "series": series,
                    "date": m.get("date", ""),
                    "url": m.get("url", ""),
                    "page": pno,
                    "access": access_for(series, f),
                    "header": make_header(m.get("title") or os.path.splitext(f)[0], m.get("date", ""), pno),
                    "text": c,
                })
        stats.append({"file": f, "pages": len(pages), "chunks": len(rows) - n_before,
                      "words": sum(len(p.split()) for p in pages)})
        print(f"{f:60s} {len(pages):4d} pages {len(rows) - n_before:5d} chunks")
    write_jsonl(CHUNKS, rows)
    lens = sorted(len(tok.tokenize(r["text"])) for r in rows)
    summary = {
        "documents": len(files), "pages": sum(s["pages"] for s in stats), "chunks": len(rows),
        "words": sum(s["words"] for s in stats), "median_chunk_tokens": lens[len(lens) // 2],
        "p95_chunk_tokens": lens[int(len(lens) * 0.95)], "seconds": round(time.time() - t0, 1),
        "per_file": stats,
    }
    json.dump(summary, open(os.path.join(RESULTS, "corpus_stats.json"), "w"), indent=1)
    print({k: v for k, v in summary.items() if k != "per_file"})


if __name__ == "__main__":
    main()
