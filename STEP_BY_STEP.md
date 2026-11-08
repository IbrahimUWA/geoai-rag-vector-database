# Building a vector database and RAG system, step by step

This is the written companion to the GeoAI Lab lecture *RAG and vector databases: a private AI assistant
for your organisation's documents*. Every number below comes from the scripts in `code/`, run on the
32 ABARES reports in `data/pdfs/`.

```
                INDEXING (offline)                                       ANSWERING (per question)
 PDFs -> parse/clean -> chunk -> embed -> vector DB  <---- retrieve <- question + user role
                                                              |
                                                     rerank -> local LLM -> answer with [citations] -> audit log
```

---

## Step 0. Set up

```bash
python -m venv .venv
.venv\Scripts\activate                 # Windows   (source .venv/bin/activate on Linux/macOS)
pip install torch --index-url https://download.pytorch.org/whl/cu128     # or the CPU wheel
pip install -r requirements.txt
```

Hardware used: one NVIDIA GPU with 24 GB+ for the 8B model in bf16 (16 GB). The embedding and retrieval
steps run on a CPU; the LLM is the only part that really wants a GPU. On a smaller GPU use a 4-bit
quantised model (bitsandbytes), or a 4B model.

## Step 1. Collect the documents, and record their licence and access level

`data/pdfs/manifest.csv` lists every file with title, publisher, series, date, URL and licence. All 32 are
ABARES publications under Creative Commons Attribution (31 CC BY 4.0, one CC BY 3.0 AU).

In an organisation this manifest is the most important file in the project. It is where each document gets
its **owner, date and access level**. RAG inherits whatever permissions you attach here, and nothing else.
For the demo, `common.py` assigns three tiers by series: crop and commodity reports and the Snapshot are
`public`, farm surveys are `internal`, and the climate research is `restricted`. These reports are all public
in reality; the labels exist to show the mechanism.

## Step 2. Parse and clean (`01_parse_chunk.py`)

```bash
python code/01_parse_chunk.py
```

* **PyMuPDF** extracts text page by page (it is fast: 1,338 pages in about 5 seconds).
* Remove what would pollute search: soft hyphens, words split across lines, bare page numbers,
  **running headers/footers** (lines that repeat on more than 40% of a document's pages), copyright /
  disclaimer pages, and **tables of contents** (dotted leaders). Without this, a question about
  "disclaimers" matches every report equally, and the table of contents outranks the actual chapter.

## Step 3. Chunk

Language models and embedders work on passages, not 120-page PDFs.

* **Size: about 350 tokens**, measured with the embedder's own tokenizer (median 307, 95th percentile 349).
  Small enough that one chunk is about one idea; big enough to keep a number with its sentence.
* **Overlap: 60 tokens**, so a fact that straddles a boundary survives in one of the two chunks.
* **Split on paragraph boundaries** first, and hard-split by tokens only when a paragraph is too long.
* **Keep metadata with every chunk**: file, title, page, date, series, access level.
* **Contextual header**: prepend `"<report title>, published <Month YYYY>, page N"` to the text that gets
  embedded and reranked. The library holds 11 editions of the same quarterly report with nearly identical
  sentences; the header tells the reranker and the LLM which edition a passage is from. Measured honestly
  (ablation in the accuracy report), it was roughly neutral for search accuracy on its own (top-6 hit 0.95 vs
  0.91, top-3 0.86 vs 0.91, same MRR); the edition problem was fixed by the recency rules in Step 6.

Result: **2,739 chunks** from 469,339 words.

## Step 4. Embed (`02_embed_index.py`)

```bash
python code/02_embed_index.py
```

An embedding model turns each chunk into a vector, so that passages with similar meaning end up close
together (cosine similarity). Three open models were compared, all running locally:

| Model | Params | Dim | Speed on GPU |
|---|---|---|---|
| all-MiniLM-L6-v2 | 23 M | 384 | 2,600 chunks/s |
| BAAI/bge-base-en-v1.5 | 109 M | 768 | 1,400 chunks/s |
| Qwen/Qwen3-Embedding-0.6B | 596 M | 1,024 | 150 chunks/s |

Two details that are easy to miss: **normalise** the vectors (then inner product = cosine similarity), and
use the **query instruction** some models expect (BGE and Qwen3 embed questions with a prefix and
documents without one).

## Step 5. Store the vectors: the vector database

The same 2,739 vectors stored six ways (`results/index_benchmark.json`, Qwen3 vectors, 200 queries, top 10):

| Store | What it is | Query | Recall@10 | Disk |
|---|---|---|---|---|
| FAISS Flat | library, exact brute force | 0.28 ms | 1.000 | 11 MB |
| FAISS HNSW | library, graph index | < 0.01 ms | 1.000 | 12 MB |
| FAISS IVF-PQ | library, clustered + compressed | 0.01 ms | 0.734 | 1.4 MB |
| Chroma | embedded database | 1.1 ms | 0.998 | 61 MB |
| Qdrant (local mode) | database, same API as a Qdrant server | 6.7 ms | 0.998 | 28 MB |
| LanceDB | columnar files on disk | 11.6 ms | 0.998 | 11 MB |

How to choose:

* **Exact search (FAISS Flat)** is fine up to a few hundred thousand chunks. Start here.
* **HNSW** (a navigable graph) is the default approximate index in most vector databases: near-exact recall,
  sub-millisecond queries, more memory.
* **IVF-PQ** compresses vectors about 8x; it loses recall (0.73 here) and needs far more training points than
  this library has. It is for hundreds of millions of vectors.
* A **database** (Chroma, Qdrant, LanceDB, Weaviate, Milvus, or **pgvector** inside PostgreSQL) adds what a
  library does not: persistence, **metadata filters** (essential for permissions), updates and deletes,
  concurrent users, backups. If your organisation already runs PostgreSQL, pgvector is often the right answer.

## Step 6. Retrieve: dense + keyword (hybrid), then rerank (`rag.py`)

Evaluated in `03_eval_retrieval.py` on 22 real questions with answers checked against the reports.
"Hit@3" = a passage from the right report that contains the answer is in the top 3.

| Method (Qwen3 embedder) | Hit@1 | Hit@3 | Hit@6 | MRR |
|---|---|---|---|---|
| BM25 keywords only | 0.36 | 0.41 | 0.55 | 0.42 |
| Dense vectors only | 0.32 | 0.46 | 0.50 | 0.41 |
| Hybrid (dense + BM25, reciprocal rank fusion) | 0.41 | 0.59 | 0.68 | 0.53 |
| Hybrid + cross-encoder rerank | 0.64 | 0.77 | 0.91 | 0.73 |
| **Hybrid + rerank + recency + supersede** | **0.77** | **0.86** | **0.95** | **0.83** |

What each piece does:

1. **Dense search** finds passages that mean the same thing in different words.
2. **BM25 keyword search** finds exact tokens: numbers, years, place names, product codes. Dense models are
   weak at "2026-27" versus "2025-26"; keywords are not.
3. **Reciprocal rank fusion** merges the two ranked lists: `score = sum 1 / (60 + rank)`. No tuning needed.
4. **A cross-encoder reranker** (`BAAI/bge-reranker-v2-m3`) reads the question and each of the 40 candidates
   together and scores relevance properly. It is slower (about 100 ms) but it is the single biggest gain.
5. **Recency rules** for reports revised every quarter: a small bonus (0.15) for newer publications, and
   **supersession**: if a newer edition of the same report series is in the top 10, chunks from older editions
   are dropped, unless the question names an edition ("In September 2025, ..."). Without supersession the model
   quoted the June 2026 forecast (54.5 Mt) even though the September revision (61 Mt) was ranked first.
6. **Access filter**: candidates the user's role may not read are removed *before* reranking, so the model
   never sees them.

## Step 7. Generate the answer with a local open-weight LLM

```python
SYSTEM = ("Answer ONLY from the numbered sources. Cite every fact like [2]. Quote numbers exactly, with units "
          "and season. If sources disagree, prefer the most recent. If the sources do not contain the answer, "
          "say: I can't find that in the documents I have access to.")
prompt = "Sources:\n\n[1] <title>, page <p>\n<chunk text>\n\n[2] ...\n\nQuestion: <question>"
```

* Models: **Qwen3-8B** (default) and Qwen3-14B, open weights under the Apache 2.0 licence,
  loaded with Hugging Face `transformers` in bf16 on the local GPU. Greedy decoding (no sampling) for
  repeatable answers. Qwen3's "thinking" mode is switched off for short factual answers.
* **No training happens anywhere in this pipeline.** The LLM is used exactly as downloaded; your knowledge
  lives in the index, where it can be updated, permissioned, cited and deleted.

Results are in `results/answer_eval.json` and `RAG_ACCURACY_REPORT.md`.

## Step 8. The interface (`app.py`)

```bash
python code/app.py         # http://127.0.0.1:7860
```

A Gradio chat with: role selector (Public visitor / Staff / Executive), model selector, RAG on/off switch to
compare against the bare model, retrieval mode and top-k, and a sources panel where the passages the answer
actually cited are highlighted with report, page and a link to the PDF. Every question is appended to
`data/audit_log.jsonl` with time, role, model, retrieved chunk ids and citations.

## Step 9. Evaluate before you trust it

* Build a question set **from your own documents** with answers you have checked (here:
  `data/eval_questions.json`, 20 answerable + 4 unanswerable).
* Measure **retrieval** separately from **generation**. If the right passage is not retrieved, no LLM can
  answer correctly; if it is retrieved and the answer is still wrong, fix the prompt or the model.
* Include questions the system **must refuse**, and questions asked by a role that should not see the answer.
* Re-run the evaluation every time you change the chunking, the embedder, the model or the prompt.

## Step 10. Making it fit for an organisation

* **Data stays inside**: local embedding model, local vector store, local LLM. No document or question is
  sent to a third party. If you do use a hosted model, use an enterprise agreement with no training on your
  data and a known data region.
* **Permissions at retrieval time**, from the document system's access lists, never "please don't reveal X"
  in the prompt.
* **Citations on every answer**, so people verify instead of trusting.
* **Audit log** of questions, roles and sources.
* **Freshness**: re-index on document change; delete from the index when a document is withdrawn.
* **Prompt injection**: treat retrieved text as data. A document that says "ignore your instructions" must not
  be able to change the system's behaviour; keep tools and actions out of a pure Q&A system.
* **Evaluation set** kept with the code and run in CI.
