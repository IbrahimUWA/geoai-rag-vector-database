"""Step 3: does the right passage come back? Retrieval evaluation.

    python 03_eval_retrieval.py

For every embedder x retrieval mode, on the answerable questions in data/eval_questions.json:
  hit@k  = an answer-bearing chunk (from the gold report AND containing the answer string) is in the top k
  MRR    = mean reciprocal rank of the first answer-bearing chunk (top 10)
Writes results/retrieval_eval.json and results/fig_retrieval.png.
"""
import gc
import json
import os
import time

import numpy as np
import torch

from charts import SERIES, TEXT2, save, setup
from common import DATA, DEFAULT_RECENCY, EMBEDDERS, RESULTS, USE_HEADERS
from rag import Retriever

OUT = "retrieval_eval.json" if USE_HEADERS else "retrieval_eval_noheader.json"
MODES = ["bm25", "dense", "hybrid", "hybrid+rerank", "hybrid+rerank+recency", "full"]   # full = + supersede


def bearing(hit, item):
    if hit["file"] != item["gold"]:
        return False
    t = hit["text"].replace("–", "-")
    return all(any(alt in t for alt in group) for group in item["must"])


def main():
    qs = [q for q in json.load(open(os.path.join(DATA, "eval_questions.json"))) if not q.get("unanswerable")]
    import sys
    out = {}
    if "--plot" in sys.argv:                    # redraw the figure from saved results
        out = json.load(open(os.path.join(RESULTS, OUT)))
    for emb in ([] if out else EMBEDDERS):
        ret = Retriever(embedder=emb, use_reranker=True)
        for mode in MODES:
            if mode == "bm25" and out.get("bm25"):
                continue
            ranks, t0 = [], time.time()
            for item in qs:
                rec = DEFAULT_RECENCY if mode in ("hybrid+rerank+recency", "full") else 0.0
                base = "hybrid+rerank" if mode == "full" else mode.replace("+recency", "")
                hits = ret.search(item["q"], k=10, mode=base, recency=rec, supersede=(mode == "full"))
                r = next((i + 1 for i, h in enumerate(hits) if bearing(h, item)), None)
                ranks.append(r)
            ms = (time.time() - t0) / len(qs) * 1e3
            res = {"hit@1": np.mean([r == 1 for r in ranks]), "hit@3": np.mean([r is not None and r <= 3 for r in ranks]),
                   "hit@6": np.mean([r is not None and r <= 6 for r in ranks]),
                   "hit@10": np.mean([r is not None for r in ranks]),
                   "mrr": np.mean([1 / r if r else 0 for r in ranks]), "ms_per_query": ms,
                   "ranks": ranks}
            res = {k: (round(float(v), 3) if k != "ranks" else v) for k, v in res.items()}
            key = "bm25" if mode == "bm25" else f"{emb}|{mode}"
            out[key] = res
            print(f"{key:28s} hit@1 {res['hit@1']:.2f} hit@3 {res['hit@3']:.2f} hit@6 {res['hit@6']:.2f} "
                  f"MRR {res['mrr']:.2f}  {ms:.0f} ms", flush=True)
        del ret
        gc.collect(); torch.cuda.empty_cache()
    if "--plot" not in sys.argv:
        json.dump(out, open(os.path.join(RESULTS, OUT), "w"), indent=1)

    if not USE_HEADERS:
        return
    # figure: hit@3 per embedder and mode
    fig, ax = setup()
    labels = {"minilm": "MiniLM (22M)", "bge-base": "BGE base (110M)", "qwen3-emb": "Qwen3 Embedding (0.6B)"}
    x = np.arange(len(EMBEDDERS))
    modes = ["dense", "hybrid", "hybrid+rerank", "hybrid+rerank+recency", "full"]
    names = {"full": "+ supersede old editions"}
    wbar = 0.16
    for j, m in enumerate(modes):
        vals = [out[f"{e}|{m}"]["hit@3"] * 100 for e in EMBEDDERS]
        bars = ax.bar(x + (j - 2) * wbar, vals, wbar - 0.02, color=SERIES[j], label=names.get(m, m.replace("+", " + ")))
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 1.2, f"{v:.0f}", ha="center", fontsize=16, color=TEXT2)
    ax.axhline(out["bm25"]["hit@3"] * 100, color=TEXT2, ls="--", lw=2,
               label=f"BM25 keywords only ({out['bm25']['hit@3'] * 100:.0f})")
    ax.set_xticks(x, [labels[e] for e in EMBEDDERS])
    ax.set_ylim(0, 108)
    ax.set_ylabel("Answer-bearing passage in top 3 (%)")
    ax.set_title(f"Retrieval quality on {len(qs)} real questions: embedder x retrieval method")
    ax.legend(loc="upper left", ncol=3, fontsize=17, columnspacing=1.2)
    ax.set_ylim(0, 118)
    ax.grid(axis="x", visible=False)
    save(fig, os.path.join(RESULTS, "fig_retrieval.png"))


if __name__ == "__main__":
    main()
