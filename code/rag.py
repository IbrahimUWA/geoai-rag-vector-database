"""The RAG engine: hybrid retrieval (dense + BM25) -> rerank -> access filter -> grounded answer.

Used by app.py (the chat interface) and 04_eval_answers.py.
"""
import os
import re
import threading
import time

import numpy as np
import torch

from common import (DEFAULT_EMBEDDER, DEFAULT_RECENCY, EMBEDDERS, INDEX, LLMS, QUERY_PREFIX, RERANKER, ROLE_CAN_READ, CHUNKS,
                    read_jsonl, tokenize_simple, index_text)

SYSTEM_RAG = (
    "You are the document assistant for an agriculture department. Answer ONLY from the numbered "
    "sources provided. Cite every fact with its source number in square brackets, like [2]. Quote "
    "numbers exactly as written in the sources, with their units and the season or date they refer to. "
    "If sources disagree, prefer the most recently published one and say which edition you used. "
    "If the sources do not contain the answer, say: \"I can't find that in the documents I have access to.\" "
    "Do not use outside knowledge. Be concise: a short paragraph or a few bullet points."
)
SYSTEM_PLAIN = (
    "You are a helpful assistant. Answer the question concisely. If you are not sure, say so."
)


MONTH_YEAR = re.compile(r"(?<![a-z])(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*[.]?[ ]+(19|20)[0-9][0-9](?![0-9])|"
                        r"(?<![a-z])(edition|quarter)(?![a-z])", re.I)


def names_edition(query):
    """True when the question asks about a specific publication (e.g. 'In September 2025, ...')."""
    return bool(MONTH_YEAR.search(query))


def is_periodic(series):
    s = series.lower()
    return any(k in s for k in ("crop report", "commodities", "snapshot"))


class Retriever:
    def __init__(self, embedder=DEFAULT_EMBEDDER, use_reranker=True, device=None):
        from rank_bm25 import BM25Okapi
        from sentence_transformers import SentenceTransformer
        import faiss

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.rows = read_jsonl(CHUNKS)
        self.embedder_key = embedder
        self.model = SentenceTransformer(EMBEDDERS[embedder], device=self.device,
                                         model_kwargs={"torch_dtype": torch.float16})
        emb = np.load(os.path.join(INDEX, f"emb_{embedder}.npy"))
        self.index = faiss.IndexFlatIP(emb.shape[1])
        self.index.add(emb)
        self.bm25 = BM25Okapi([tokenize_simple(index_text(r)) for r in self.rows])
        # recency in [0, 1]: 1 = newest publication in the library
        ds = sorted({r["date"] for r in self.rows if r.get("date")})
        self.recency = np.array([ds.index(r["date"]) / max(1, len(ds) - 1) if r.get("date") else 0.0
                                 for r in self.rows], np.float32)
        self.reranker = None
        if use_reranker:
            from sentence_transformers import CrossEncoder
            self.reranker = CrossEncoder(RERANKER, device=self.device, model_kwargs={"torch_dtype": torch.float16})

    def dense(self, query, k):
        q = self.model.encode([QUERY_PREFIX[self.embedder_key] + query], normalize_embeddings=True).astype(np.float32)
        s, i = self.index.search(q, k)
        return list(i[0]), list(s[0])

    def sparse(self, query, k):
        s = self.bm25.get_scores(tokenize_simple(query))
        i = np.argsort(-s)[:k]
        return list(i), list(s[i])

    def search(self, query, k=6, mode="hybrid+rerank", allowed=None, pool=40, recency=DEFAULT_RECENCY,
               supersede=True):
        """mode: dense | bm25 | hybrid | hybrid+rerank. allowed: set of access labels the user may read.
        recency: weight of a 'prefer the newer edition' bonus added to the reranker score (0 = off).
        supersede: drop older editions of a periodic report when a newer one was retrieved."""
        def ok(i):
            return allowed is None or self.rows[i]["access"] in allowed

        if mode == "dense":
            ids, _ = self.dense(query, pool * 3)
            cand = [i for i in ids if ok(i)][:k]
            return [self._hit(i, None) for i in cand]
        if mode == "bm25":
            ids, _ = self.sparse(query, pool * 3)
            cand = [i for i in ids if ok(i)][:k]
            return [self._hit(i, None) for i in cand]

        # reciprocal rank fusion of dense and BM25 lists
        d_ids, _ = self.dense(query, pool * 3)
        s_ids, _ = self.sparse(query, pool * 3)
        fused = {}
        for lst in (d_ids, s_ids):
            for rank, i in enumerate([i for i in lst if ok(i)][:pool]):
                fused[i] = fused.get(i, 0.0) + 1.0 / (60 + rank)
        cand = sorted(fused, key=fused.get, reverse=True)[:pool]
        if mode == "hybrid" or self.reranker is None:
            return [self._hit(i, fused[i]) for i in cand[:k]]
        scores = np.asarray(self.reranker.predict([(query, index_text(self.rows[i])) for i in cand], batch_size=32))
        scores = scores + recency * self.recency[cand]
        order = list(np.argsort(-np.asarray(scores)))
        if supersede and not names_edition(query):
            order = self._supersede([cand[j] for j in order], order)
        return [self._hit(cand[j], float(scores[j])) for j in order[:k]]

    def _supersede(self, ranked_rows, order, look=10):
        """Periodic reports: if a newer edition of the same series is among the top `look` results,
        drop chunks from older editions of that series (a revised forecast replaces the old one)."""
        newest = {}
        for i in ranked_rows[:look]:
            r = self.rows[i]
            if is_periodic(r["series"]):
                newest[r["series"]] = max(newest.get(r["series"], ""), r["date"])
        keep = []
        for i, j in zip(ranked_rows, order):
            r = self.rows[i]
            if is_periodic(r["series"]) and r["series"] in newest and r["date"] < newest[r["series"]]:
                continue
            keep.append(j)
        return keep

    def _hit(self, i, score):
        r = dict(self.rows[i])
        r["score"] = score
        r["row"] = int(i)
        return r


def format_context(hits):
    blocks = []
    for n, h in enumerate(hits, start=1):
        blocks.append(f"[{n}] {h['title']} ({h['date']}), page {h['page']}\n{h['text']}")
    return "\n\n".join(blocks)


class Generator:
    """A local open-weight chat model. Nothing leaves this machine."""

    def __init__(self, name=None):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.name = name or next(iter(LLMS))
        repo = LLMS.get(self.name, self.name)
        self.tok = AutoTokenizer.from_pretrained(repo)
        self.model = AutoModelForCausalLM.from_pretrained(repo, torch_dtype=torch.bfloat16, device_map="cuda")
        self.model.eval()

    def _prompt(self, system, user):
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        kw = {"enable_thinking": False} if "qwen3" in self.name.lower() else {}
        return self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **kw)

    def stream(self, system, user, max_new_tokens=400):
        from transformers import TextIteratorStreamer
        inputs = self.tok(self._prompt(system, user), return_tensors="pt").to(self.model.device)
        streamer = TextIteratorStreamer(self.tok, skip_prompt=True, skip_special_tokens=True)
        th = threading.Thread(target=self.model.generate, kwargs=dict(
            **inputs, streamer=streamer, max_new_tokens=max_new_tokens, do_sample=False, temperature=None,
            top_p=None, top_k=None, pad_token_id=self.tok.eos_token_id))
        th.start()
        for piece in streamer:
            yield piece
        th.join()

    def answer(self, system, user, max_new_tokens=400):
        return "".join(self.stream(system, user, max_new_tokens)).strip()


def rag_prompt(question, hits):
    return f"Sources:\n\n{format_context(hits)}\n\nQuestion: {question}"


def used_citations(answer, n):
    return sorted({int(m) for m in re.findall(r"\[(\d+)\]", answer) if 1 <= int(m) <= n})


def allowed_for(role):
    return ROLE_CAN_READ.get(role, {"public"})
