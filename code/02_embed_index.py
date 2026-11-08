"""Step 2: chunks -> embeddings -> four vector stores (FAISS, Chroma, Qdrant, LanceDB).

    python 02_embed_index.py

Writes data/index/emb_<model>.npy for every embedder, the stores for the default
embedder, and results/index_benchmark.json (build time, disk size, query latency,
recall of approximate indexes against exact search).
"""
import json
import os
import shutil
import time

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from common import index_text, CHUNKS, DEFAULT_EMBEDDER, EMBEDDERS, INDEX, QUERY_PREFIX, RESULTS, read_jsonl


def dir_mb(p):
    if os.path.isfile(p):
        return os.path.getsize(p) / 1e6
    return sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(p) for f in fs) / 1e6


def embed_all(rows):
    out = {}
    for key, name in EMBEDDERS.items():
        path = os.path.join(INDEX, f"emb_{key}.npy")
        model = SentenceTransformer(name, device="cuda" if torch.cuda.is_available() else "cpu",
                                    model_kwargs={"torch_dtype": torch.float16})
        t0 = time.time()
        emb = model.encode([index_text(r) for r in rows], batch_size=64, normalize_embeddings=True,
                           show_progress_bar=True, convert_to_numpy=True).astype(np.float32)
        secs = time.time() - t0
        np.save(path, emb)
        out[key] = {"model": name, "dim": int(emb.shape[1]), "seconds": round(secs, 1),
                    "chunks_per_s": round(len(rows) / secs, 1),
                    "params_m": round(sum(p.numel() for p in model.parameters()) / 1e6)}
        print(key, out[key])
        del model
        torch.cuda.empty_cache()
    return out


def bench_queries(emb, n=200, seed=0):
    """Queries for latency/recall tests: perturbed copies of random chunk vectors."""
    rng = np.random.default_rng(seed)
    q = emb[rng.choice(len(emb), n, replace=False)] + rng.normal(0, 0.02, (n, emb.shape[1])).astype(np.float32)
    return q / np.linalg.norm(q, axis=1, keepdims=True)


def recall_at(found, truth, k=10):
    return float(np.mean([len(set(f[:k]) & set(t[:k])) / k for f, t in zip(found, truth)]))


def build_stores(rows, emb):
    import faiss
    import chromadb
    import lancedb
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, PointStruct, VectorParams

    d, k = emb.shape[1], 10
    qs = bench_queries(emb)
    res = {}

    # ---- FAISS: exact, HNSW graph, IVF-PQ compressed ----
    flat = faiss.IndexFlatIP(d)
    t0 = time.time(); flat.add(emb); tb = time.time() - t0
    t0 = time.time(); _, truth = flat.search(qs, k); tq = (time.time() - t0) / len(qs)
    faiss.write_index(flat, os.path.join(INDEX, "faiss_flat.index"))
    res["FAISS Flat (exact)"] = {"build_s": tb, "query_ms": tq * 1e3, "recall@10": 1.0,
                                 "disk_mb": dir_mb(os.path.join(INDEX, "faiss_flat.index"))}

    hnsw = faiss.IndexHNSWFlat(d, 32, faiss.METRIC_INNER_PRODUCT)
    hnsw.hnsw.efConstruction, hnsw.hnsw.efSearch = 200, 64
    t0 = time.time(); hnsw.add(emb); tb = time.time() - t0
    t0 = time.time(); _, I = hnsw.search(qs, k); tq = (time.time() - t0) / len(qs)
    faiss.write_index(hnsw, os.path.join(INDEX, "faiss_hnsw.index"))
    res["FAISS HNSW"] = {"build_s": tb, "query_ms": tq * 1e3, "recall@10": recall_at(I, truth),
                         "disk_mb": dir_mb(os.path.join(INDEX, "faiss_hnsw.index"))}

    nlist = max(16, int(np.sqrt(len(emb))))
    ivf = faiss.IndexIVFPQ(faiss.IndexFlatIP(d), d, nlist, 48 if d % 48 == 0 else 32, 8, faiss.METRIC_INNER_PRODUCT)
    t0 = time.time(); ivf.train(emb); ivf.add(emb); tb = time.time() - t0
    ivf.nprobe = 8
    t0 = time.time(); _, I = ivf.search(qs, k); tq = (time.time() - t0) / len(qs)
    faiss.write_index(ivf, os.path.join(INDEX, "faiss_ivfpq.index"))
    res["FAISS IVF-PQ"] = {"build_s": tb, "query_ms": tq * 1e3, "recall@10": recall_at(I, truth),
                           "disk_mb": dir_mb(os.path.join(INDEX, "faiss_ivfpq.index"))}

    ids = [r["id"] for r in rows]
    pos = {cid: i for i, cid in enumerate(ids)}
    metas = [{"file": r["file"], "page": r["page"], "series": r["series"], "access": r["access"],
              "date": r["date"]} for r in rows]

    # ---- Chroma (embedded database, HNSW inside, metadata filters) ----
    p = os.path.join(INDEX, "chroma"); shutil.rmtree(p, ignore_errors=True)
    client = chromadb.PersistentClient(path=p)
    col = client.create_collection("reports", metadata={"hnsw:space": "cosine"})
    t0 = time.time()
    for i in range(0, len(rows), 4000):
        col.add(ids=ids[i:i + 4000], embeddings=emb[i:i + 4000].tolist(), metadatas=metas[i:i + 4000],
                documents=[r["text"] for r in rows[i:i + 4000]])
    tb = time.time() - t0
    t0 = time.time()
    found = [[pos[x] for x in col.query(query_embeddings=[q.tolist()], n_results=k)["ids"][0]] for q in qs]
    tq = (time.time() - t0) / len(qs)
    res["Chroma"] = {"build_s": tb, "query_ms": tq * 1e3, "recall@10": recall_at(found, truth), "disk_mb": dir_mb(p)}
    del client

    # ---- Qdrant (local mode; the same API talks to a Qdrant server) ----
    p = os.path.join(INDEX, "qdrant"); shutil.rmtree(p, ignore_errors=True)
    qc = QdrantClient(path=p)
    qc.create_collection("reports", vectors_config=VectorParams(size=d, distance=Distance.COSINE))
    t0 = time.time()
    for i in range(0, len(rows), 1000):
        qc.upsert("reports", points=[PointStruct(id=j, vector=emb[j].tolist(), payload={**metas[j], "chunk_id": ids[j]})
                                     for j in range(i, min(i + 1000, len(rows)))])
    tb = time.time() - t0
    t0 = time.time()
    found = [[h.id for h in qc.query_points("reports", query=q.tolist(), limit=k).points] for q in qs]
    tq = (time.time() - t0) / len(qs)
    res["Qdrant (local)"] = {"build_s": tb, "query_ms": tq * 1e3, "recall@10": recall_at(found, truth), "disk_mb": dir_mb(p)}
    qc.close()

    # ---- LanceDB (columnar files on disk, good for very large corpora) ----
    p = os.path.join(INDEX, "lancedb"); shutil.rmtree(p, ignore_errors=True)
    db = lancedb.connect(p)
    t0 = time.time()
    tbl = db.create_table("reports", data=[{"vector": emb[j].tolist(), "row": j, "chunk_id": ids[j], **metas[j]}
                                           for j in range(len(rows))])
    tb = time.time() - t0
    t0 = time.time()
    found = [tbl.search(q.tolist()).metric("cosine").limit(k).to_pandas()["row"].tolist() for q in qs]
    tq = (time.time() - t0) / len(qs)
    res["LanceDB"] = {"build_s": tb, "query_ms": tq * 1e3, "recall@10": recall_at(found, truth), "disk_mb": dir_mb(p)}

    for v in res.values():
        for kk in v:
            v[kk] = round(float(v[kk]), 3)
    return res


def main():
    import sys
    rows = read_jsonl(CHUNKS)
    print(len(rows), "chunks")
    emb_info = embed_all(rows)
    if "--no-stores" in sys.argv:
        return
    emb = np.load(os.path.join(INDEX, f"emb_{DEFAULT_EMBEDDER}.npy"))
    stores = build_stores(rows, emb)
    for k, v in stores.items():
        print(f"{k:22s} {v}")
    json.dump({"n_chunks": len(rows), "embedders": emb_info, "stores": stores},
              open(os.path.join(RESULTS, "index_benchmark.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
