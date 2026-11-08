"""Step 4: does RAG make the answers right? LLM alone vs LLM + retrieval, and access control.

    python 04_eval_answers.py [--llms Qwen3-8B,Qwen3-14B]

For each local LLM:
  closed-book : the question alone (the model's memory)
  RAG         : the question + top-6 passages (hybrid + rerank), as Staff
  correct     = every required number/phrase appears in the answer
  refusal     = for unanswerable questions, the model says it cannot find it
Plus an access-control check: the restricted climate questions asked as "Public visitor" must not
be answered from restricted documents.
Writes results/answer_eval.json, results/answers_examples.md, results/fig_answers.png
"""
import argparse
import gc
import json
import os
import re
import time

import numpy as np
import torch

from charts import BAD, GOOD, SERIES, TEXT2, save, setup
from common import DATA, LLMS, RESULTS
from rag import SYSTEM_PLAIN, SYSTEM_RAG, Generator, Retriever, allowed_for, rag_prompt

REFUSAL = re.compile(r"can't find|cannot find|could not find|couldn't find|not (contain|include|mention|provide)|"
                     r"no information|don't have|do not have|unable to|not available|not specified|not stated", re.I)


def correct(ans, item):
    a = ans.replace("–", "-").replace(" ", " ")
    return all(any(alt.lower() in a.lower() for alt in group) for group in item["must"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llms", default=",".join(LLMS))
    ap.add_argument("--plot", action="store_true", help="only redraw the figure from saved results")
    args = ap.parse_args()
    qs = json.load(open(os.path.join(DATA, "eval_questions.json")))
    if args.plot:
        return plot(json.load(open(os.path.join(RESULTS, "answer_eval.json"))), qs)
    ret = Retriever()
    # retrieve once (as Staff, and as Public visitor for the access test)
    ctx_staff = {i: ret.search(it["q"], k=6, allowed=allowed_for("Staff")) for i, it in enumerate(qs)}
    ctx_exec = {i: ret.search(it["q"], k=6, allowed=allowed_for("Executive")) for i, it in enumerate(qs)}
    ctx_public = {i: ret.search(it["q"], k=6, allowed=allowed_for("Public visitor")) for i, it in enumerate(qs)}
    results, examples = {}, []
    for name in [n.strip() for n in args.llms.split(",") if n.strip()]:
        gen = Generator(name)
        rows = []
        t0 = time.time()
        for i, it in enumerate(qs):
            closed = gen.answer(SYSTEM_PLAIN, it["q"], max_new_tokens=220)
            hits = ctx_exec[i] if it.get("access") == "restricted" else ctx_staff[i]
            rag = gen.answer(SYSTEM_RAG, rag_prompt(it["q"], hits), max_new_tokens=300)
            pub = None
            if it.get("access") in ("internal", "restricted"):
                pub = gen.answer(SYSTEM_RAG, rag_prompt(it["q"], ctx_public[i]), max_new_tokens=200)
            row = {"q": it["q"], "closed": closed, "rag": rag, "public": pub,
                   "unanswerable": bool(it.get("unanswerable")), "access": it.get("access")}
            if it.get("unanswerable"):
                row["closed_ok"] = bool(REFUSAL.search(closed))
                row["rag_ok"] = bool(REFUSAL.search(rag))
            else:
                row["closed_ok"] = correct(closed, it)
                row["rag_ok"] = correct(rag, it)
                if pub is not None:
                    # leak = public user got the protected number
                    row["public_leak"] = correct(pub, it)
                    row["public_refused"] = bool(REFUSAL.search(pub))
            rows.append(row)
            print(f"[{name}] {i:2d} closed={'Y' if row['closed_ok'] else '.'} rag={'Y' if row['rag_ok'] else '.'} "
                  f"{'leak!' if row.get('public_leak') else ''} {it['q'][:70]}", flush=True)
        ans = [r for r in rows if not r["unanswerable"]]
        una = [r for r in rows if r["unanswerable"]]
        prot = [r for r in ans if r["public"] is not None]
        results[name] = {
            "closed_correct": round(float(np.mean([r["closed_ok"] for r in ans])), 3),
            "rag_correct": round(float(np.mean([r["rag_ok"] for r in ans])), 3),
            "closed_refusal": round(float(np.mean([r["closed_ok"] for r in una])), 3),
            "rag_refusal": round(float(np.mean([r["rag_ok"] for r in una])), 3),
            "public_leaks": int(sum(r.get("public_leak", False) for r in prot)), "protected_questions": len(prot),
            "seconds": round(time.time() - t0, 1), "rows": rows,
        }
        print(name, {k: v for k, v in results[name].items() if k != "rows"}, flush=True)
        del gen
        gc.collect(); torch.cuda.empty_cache()
    json.dump(results, open(os.path.join(RESULTS, "answer_eval.json"), "w"), indent=1)

    # examples markdown (first model)
    first = next(iter(results))
    with open(os.path.join(RESULTS, "answers_examples.md"), "w", encoding="utf-8") as f:
        for r in results[first]["rows"]:
            f.write(f"### {r['q']}\n\n**Closed-book ({'correct' if r['closed_ok'] else 'wrong'})**\n\n{r['closed']}\n\n"
                    f"**RAG ({'correct' if r['rag_ok'] else 'wrong'})**\n\n{r['rag']}\n\n")
            if r["public"]:
                f.write(f"**As Public visitor**\n\n{r['public']}\n\n")

    plot(results, qs)


def plot(results, qs):
    n_ans = len([q for q in qs if not q.get("unanswerable")])
    fig, ax = setup()
    names = list(results)
    x = np.arange(len(names))
    w = 0.36
    c = [results[n]["closed_correct"] * 100 for n in names]
    r = [results[n]["rag_correct"] * 100 for n in names]
    b1 = ax.bar(x - w / 2, c, w - 0.03, color=SERIES[1], label="LLM alone (closed book)")
    b2 = ax.bar(x + w / 2, r, w - 0.03, color=SERIES[0], label="LLM + RAG over the reports")
    for bars, vals in ((b1, c), (b2, r)):
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{round(v * n_ans / 100)} / {n_ans}", ha="center",
                    fontsize=24, color=TEXT2)
    ax.set_xticks(x, names)
    ax.set_ylim(0, 128)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.set_ylabel("Questions answered correctly (%)")
    ax.set_title(f"Same open-weight models, same {n_ans} questions: with and without retrieval")
    ax.legend(loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.0))
    ax.grid(axis="x", visible=False)
    save(fig, os.path.join(RESULTS, "fig_answers.png"))


if __name__ == "__main__":
    main()
