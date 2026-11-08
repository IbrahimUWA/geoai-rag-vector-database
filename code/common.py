"""Shared paths and settings for the episode 6 RAG pipeline."""
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
EP = os.path.dirname(HERE)
DATA = os.path.join(EP, "data")
PDFS = os.path.join(DATA, "pdfs")
INDEX = os.path.join(DATA, "index")
RESULTS = os.path.join(EP, "results")
CHUNKS = os.path.join(DATA, "chunks.jsonl")
for _d in (DATA, PDFS, INDEX, RESULTS):
    os.makedirs(_d, exist_ok=True)

# Embedding models compared in the episode (all open weights, run locally).
EMBEDDERS = {
    "minilm": "sentence-transformers/all-MiniLM-L6-v2",   # 22M params, the classic baseline
    "bge-base": "BAAI/bge-base-en-v1.5",                   # 110M params
    "qwen3-emb": "Qwen/Qwen3-Embedding-0.6B",              # 600M params, multilingual
}
# Query instructions some embedders expect (documents are embedded without one).
QUERY_PREFIX = {
    "minilm": "",
    "bge-base": "Represent this sentence for searching relevant passages: ",
    "qwen3-emb": "Instruct: Given a question about Australian agriculture, retrieve passages that answer it\nQuery: ",
}
DEFAULT_EMBEDDER = "qwen3-emb"     # best in 03_eval_retrieval.py
DEFAULT_RECENCY = 0.15             # "prefer the newer edition" bonus on the reranker score
RERANKER = "BAAI/bge-reranker-v2-m3"

# Open-weight chat models (ungated on Hugging Face, so no token is needed).
LLMS = {
    "Qwen3-8B": "Qwen/Qwen3-8B",
    "Qwen3-14B": "Qwen/Qwen3-14B",
}
DEFAULT_LLM = "Qwen3-8B"

# Chunking
CHUNK_TOKENS = 350
CHUNK_OVERLAP = 60

# Demo access control: which roles may read which document series.
# The labels are a teaching device (all these reports are actually public);
# in a real organisation they would come from the document management system.
SERIES_ACCESS = [            # (keyword in series name, access level); first match wins
    ("crop report", "public"),
    ("commodities", "public"),
    ("snapshot", "public"),
    ("farm survey", "internal"),
    ("climate", "restricted"),
]


def access_for(series, file=""):
    s = f"{series} {file}".lower().replace("_", " ")
    for kw, level in SERIES_ACCESS:
        if kw in s:
            return level
    return "internal" if "farm" in s else "public"


ROLE_CAN_READ = {
    "Public visitor": {"public"},
    "Staff": {"public", "internal"},
    "Executive": {"public", "internal", "restricted"},
}


USE_HEADERS = not os.environ.get("RAG_NO_HEADERS")   # prepend "title, published Month YYYY, page N" before embedding
if not USE_HEADERS:                                    # ablation: keep a separate index
    INDEX = os.path.join(DATA, "index_noheader")
    os.makedirs(INDEX, exist_ok=True)


def index_text(r):
    if USE_HEADERS and r.get("header"):
        return r["header"] + "\n" + r["text"]
    return r["text"]


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def tokenize_simple(text):
    """Lower-case word tokens for BM25."""
    return re.findall(r"[a-z0-9]+(?:\.[0-9]+)?", text.lower())
