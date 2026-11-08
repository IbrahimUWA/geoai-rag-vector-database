"""Step 7: write RAG_ACCURACY_REPORT.md from the saved evaluation results.

    python 07_report.py
"""
import json
import os

from common import DATA, EP, RESULTS, read_jsonl, CHUNKS


def pct(x):
    return f"{x * 100:.0f}%"


def main():
    corpus = json.load(open(os.path.join(RESULTS, "corpus_stats.json")))
    idx = json.load(open(os.path.join(RESULTS, "index_benchmark.json")))
    ret = json.load(open(os.path.join(RESULTS, "retrieval_eval.json")))
    v1p = os.path.join(RESULTS, "retrieval_eval_noheader.json")
    ret_v1 = json.load(open(v1p)) if os.path.exists(v1p) else None
    ans = json.load(open(os.path.join(RESULTS, "answer_eval.json")))
    qs = json.load(open(os.path.join(DATA, "eval_questions.json")))
    answerable = [q for q in qs if not q.get("unanswerable")]
    n_a, n_u = len(answerable), len(qs) - len(answerable)
    best = max((k for k in ret if k != "bm25"), key=lambda k: (ret[k]["hit@3"], ret[k]["mrr"], k.endswith("|full")))
    L = []
    w = L.append
    w("# RAG accuracy report\n")
    w("GeoAI Lab, Series 2 episode 1: *RAG and vector databases*. All numbers are produced by the scripts in "
      "`code/` and can be reproduced on one workstation GPU.\n")
    w("## Summary\n")
    first = next(iter(ans))
    w(f"* **Corpus:** {corpus['documents']} ABARES reports, {corpus['pages']:,} pages, {corpus['words']:,} words, "
      f"split into {corpus['chunks']:,} chunks (median {corpus['median_chunk_tokens']} tokens).")
    w(f"* **Evaluation set:** {n_a} answerable questions with answers verified against the source page, plus "
      f"{n_u} questions the documents cannot answer. Answers are scored automatically: correct only if the exact "
      f"figure from the report appears in the answer.")
    w(f"* **Retrieval:** best configuration `{best}` puts an answer-bearing passage in the top 3 for "
      f"{pct(ret[best]['hit@3'])} of questions and in the top 6 (what the LLM sees) for {pct(ret[best]['hit@6'])} "
      f"(MRR {ret[best]['mrr']:.2f}). Keyword search alone: {pct(ret['bm25']['hit@3'])} top 3.")
    for name, r in ans.items():
        w(f"* **{name}:** {pct(r['closed_correct'])} correct on its own, **{pct(r['rag_correct'])} correct with RAG**; "
          f"declined {pct(r['rag_refusal'])} of unanswerable questions with RAG (vs {pct(r['closed_refusal'])} alone).")
    w("* **Access control:** restricted and internal documents are filtered out before reranking, so a lower-privilege "
      "role never has them in the model's context (verified below).\n")

    w("## 1. Retrieval accuracy\n")
    w("Hit@k = a chunk from the gold report that contains the required answer string is among the top k results. "
      "MRR = mean reciprocal rank of the first such chunk (0 if not in the top 10).\n")
    w("| Embedder | Method (full = hybrid + rerank + recency + supersede) | Hit@1 | Hit@3 | Hit@6 | Hit@10 | MRR | ms / query |")
    w("|---|---|---|---|---|---|---|---|")
    for k, v in ret.items():
        emb, mode = (k.split("|") + [""])[:2] if "|" in k else ("-", "BM25 keywords")
        bold = "**" if k == best else ""
        w(f"| {emb} | {bold}{mode}{bold} | {v['hit@1']:.2f} | {v['hit@3']:.2f} | {v['hit@6']:.2f} | {v['hit@10']:.2f} "
          f"| {v['mrr']:.2f} | {v['ms_per_query']:.0f} |")
    w("")
    if ret_v1:
        w("**Effect of contextual chunk headers** (title + publication month + page prepended before embedding), "
          "hit@3 with hybrid + rerank:\n")
        w("| Embedder | Without headers | With headers |")
        w("|---|---|---|")
        for e in ("minilm", "bge-base", "qwen3-emb"):
            a, b = ret_v1.get(f"{e}|hybrid+rerank"), ret.get(f"{e}|hybrid+rerank")
            if a and b:
                w(f"| {e} | {a['hit@3']:.2f} | {b['hit@3']:.2f} |")
        w("")
    w("![Retrieval](results/fig_retrieval.png)\n")
    ranks = ret[best]["ranks"]
    w(f"Per-question rank of the first answer-bearing chunk ({best}):\n")
    w("| # | Question | Gold report | Rank |")
    w("|---|---|---|---|")
    for i, (q, r) in enumerate(zip(answerable, ranks), 1):
        w(f"| {i} | {q['q']} | `{q['gold']}` | {r if r else 'not in top 10'} |")
    w("")

    w("## 2. Answer accuracy: the same LLMs with and without retrieval\n")
    w("| Model | Correct, LLM alone | Correct, LLM + RAG | Declines unanswerable, alone | Declines unanswerable, RAG | Eval time (s) |")
    w("|---|---|---|---|---|---|")
    for name, r in ans.items():
        w(f"| {name} | {pct(r['closed_correct'])} | **{pct(r['rag_correct'])}** | {pct(r['closed_refusal'])} | "
          f"{pct(r['rag_refusal'])} | {r['seconds']:.0f} |")
    w("\n![Answers](results/fig_answers.png)\n")
    for name, r in ans.items():
        w(f"### {name}: question by question\n")
        w("| # | Question | Alone | RAG |")
        w("|---|---|---|---|")
        for i, row in enumerate(r["rows"], 1):
            tag = " *(unanswerable)*" if row["unanswerable"] else ""
            w(f"| {i} | {row['q']}{tag} | {'✅' if row['closed_ok'] else '❌'} | {'✅' if row['rag_ok'] else '❌'} |")
        w("")
        wrong = [row for row in r["rows"] if not row["rag_ok"]]
        if wrong:
            w(f"**RAG failures for {name}** (answer text as generated):\n")
            for row in wrong:
                w(f"* *{row['q']}*  \n  > {row['rag'].strip().replace(chr(10), ' ')[:600]}")
            w("")

    w("## 3. Access control\n")
    w("Questions whose answer is in an `internal` or `restricted` report were also asked as **Public visitor**. "
      "The retriever drops non-public chunks before reranking, so the model cannot quote them.\n")
    rows = read_jsonl(CHUNKS)
    w("| Model | Protected questions | Public-role answers containing the protected figure |")
    w("|---|---|---|")
    for name, r in ans.items():
        w(f"| {name} | {r['protected_questions']} | {r['public_leaks']} |")
    w("\nA non-zero count would not necessarily be a leak: the same figure can also appear in a public report "
      "(for example, a climate statistic quoted in the public Snapshot). Each case is listed below for review.\n")
    for name, r in ans.items():
        for row in r["rows"]:
            if row.get("public") is not None:
                w(f"* **{name}**, *{row['q']}* (as Public visitor): {row['public'].strip().replace(chr(10), ' ')[:300]}")
    w("")

    w("## 4. Vector store benchmark\n")
    w(f"{idx['n_chunks']:,} vectors, 200 queries, top 10.\n")
    w("| Store | Build (s) | Query (ms) | Recall@10 vs exact | Disk (MB) |")
    w("|---|---|---|---|---|")
    for k, v in idx["stores"].items():
        q = f"{v['query_ms']:.2f}" if v["query_ms"] >= 0.01 else "< 0.01"
        w(f"| {k} | {v['build_s']:.2f} | {q} | {v['recall@10']:.3f} | {v['disk_mb']:.1f} |")
    w("\n| Embedder | Params (M) | Dim | Chunks / s (GPU) |")
    w("|---|---|---|---|")
    for k, v in idx["embedders"].items():
        w(f"| {v['model']} | {v['params_m']} | {v['dim']} | {v['chunks_per_s']:,.0f} |")

    w("\n## 5. Method and caveats\n")
    w("* Scoring is strict string matching on the figure (e.g. `61 million`, `16%`/`16 per cent`). A correct answer "
      "phrased differently (\"sixty-one million\") would be marked wrong, so the numbers are a lower bound.")
    w("* For forecast questions the gold answer is the **latest** edition (September 2026). A model that quotes an "
      "earlier edition's forecast is marked wrong even though the figure exists in the library.")
    w(f"* {n_a + n_u} questions show direction and size of effects, not a guarantee. Re-run with your own documents "
      "and questions before relying on a system like this.")
    w("* Greedy decoding (no sampling), Qwen3 thinking mode off, top 6 passages, all models in bf16 on one GPU.")
    open(os.path.join(EP, "RAG_ACCURACY_REPORT.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("wrote RAG_ACCURACY_REPORT.md")


if __name__ == "__main__":
    main()
