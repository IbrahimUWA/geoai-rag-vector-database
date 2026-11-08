"""Step 5: explanatory figures for the episode.

    python 05_figures.py

results/fig_architecture.png   the two RAG pipelines (indexing offline, answering online)
results/fig_embedding_map.png  2-D map of every chunk's embedding, coloured by report series, with a query
results/fig_index_bench.png    the vector-store benchmark as a table figure
results/fig_approaches.png     prompt stuffing vs fine-tuning vs RAG
"""
import json
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from charts import GRID, SERIES, SURFACE, TEXT, TEXT2, save, setup
from common import CHUNKS, DEFAULT_EMBEDDER, INDEX, QUERY_PREFIX, EMBEDDERS, RESULTS, read_jsonl

W, H = 1920, 1080
BG = (11, 18, 32)
PANEL = (17, 25, 41)
FG = (236, 239, 244)
MUTED = (148, 163, 184)
ACC = (56, 189, 248)
GRN = (74, 222, 128)
AMB = (251, 191, 36)
F = lambda s, b=False: ImageFont.truetype("C:/Windows/Fonts/" + ("segoeuib.ttf" if b else "segoeui.ttf"), s)


def box(d, x, y, w, h, title, sub, col):
    d.rounded_rectangle((x, y, x + w, y + h), radius=18, fill=PANEL, outline=col, width=3)
    tw = d.textlength(title, font=F(30, True))
    d.text((x + (w - tw) / 2, y + 22), title, font=F(30, True), fill=FG)
    yy = y + 66
    for line in sub.split("\n"):
        lw = d.textlength(line, font=F(22))
        d.text((x + (w - lw) / 2, yy), line, font=F(22), fill=MUTED)
        yy += 30


def arrow(d, x0, y0, x1, y1, col, label=None):
    d.line((x0, y0, x1, y1), fill=col, width=4)
    v = np.array([x1 - x0, y1 - y0], float)
    v /= np.linalg.norm(v)
    n = np.array([-v[1], v[0]])
    p = np.array([x1, y1])
    d.polygon([tuple(p), tuple(p - 18 * v + 9 * n), tuple(p - 18 * v - 9 * n)], fill=col)
    if label:
        d.text(((x0 + x1) / 2 + 8, (y0 + y1) / 2 - 34), label, font=F(20), fill=MUTED)


def architecture():
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    d.text((80, 40), "How a RAG system works", font=F(56, True), fill=FG)
    d.text((80, 120), "Two pipelines that share one vector database", font=F(30), fill=MUTED)

    # indexing (offline)
    d.text((80, 205), "1  INDEXING  -  once, and again whenever documents change", font=F(26, True), fill=AMB)
    y, bw, bh = 255, 300, 170
    xs = [80, 450, 820, 1190]
    items = [("Documents", "32 ABARES PDFs\n1,338 pages", AMB), ("Parse + clean", "PyMuPDF text\nstrip headers, TOC", AMB),
             ("Chunk", "~350 tokens\n60 token overlap", AMB), ("Embed", "Qwen3 embedding model\ntext -> 1,024 numbers", AMB)]
    for x, (t, s, c) in zip(xs, items):
        box(d, x, y, bw, bh, t, s, c)
    for a, b in zip(xs[:-1], xs[1:]):
        arrow(d, a + bw + 6, y + bh / 2, b - 8, y + bh / 2, AMB)
    # vector db
    vx, vy, vw, vh = 1560, 330, 300, 380
    d.rounded_rectangle((vx, vy, vx + vw, vy + vh), radius=26, fill=PANEL, outline=ACC, width=4)
    d.text((vx + 40, vy + 26), "Vector", font=F(40, True), fill=ACC)
    d.text((vx + 40, vy + 76), "database", font=F(40, True), fill=ACC)
    for i, l in enumerate(["vectors + text", "+ metadata", "(report, page,", " access level)", "", "FAISS, Chroma,",
                           "Qdrant, LanceDB"]):
        d.text((vx + 40, vy + 150 + i * 30), l, font=F(23), fill=MUTED)
    arrow(d, xs[-1] + bw + 6, y + bh / 2, vx - 8, vy + 60, AMB)

    # answering (online)
    d.text((80, 520), "2  ANSWERING  -  every question, in about a second", font=F(26, True), fill=GRN)
    y2 = 570
    items2 = [("Question", "plus the user's role\nfrom sign-in", GRN), ("Retrieve", "vector + keyword search\nfiltered by access", GRN),
              ("Rerank", "cross-encoder keeps\nthe best 6 passages", GRN), ("Local LLM", "Qwen3 8B / 14B writes\nan answer from them", GRN)]
    for x, (t, s, c) in zip(xs, items2):
        box(d, x, y2, bw, bh, t, s, c)
    for a, b in zip(xs[:-1], xs[1:]):
        arrow(d, a + bw + 6, y2 + bh / 2, b - 8, y2 + bh / 2, GRN)
    d.line((vx - 4, 475, 735, 475), fill=ACC, width=4)          # elbow: database -> retrieve
    arrow(d, 735, 473, 735, y2 - 8, ACC, None)
    d.text((1010, 437), "nearest chunks come back", font=F(22), fill=ACC)
    box(d, 450, 850, 1040, 130, "Answer with citations  [1] [2]", "every claim points at a report and page; the question and sources are logged", GRN)
    arrow(d, xs[3] + bw / 2, y2 + bh + 6, 1300, 846, GRN)
    d.text((80, 1030), "Nothing in either pipeline leaves the organisation's computer.", font=F(24), fill=MUTED)
    img.save(os.path.join(RESULTS, "fig_architecture.png"))


def approaches():
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    d.text((80, 40), "Three ways to give a model your organisation's knowledge", font=F(52, True), fill=FG)
    cols = [
        ("Paste it into the prompt", AMB, ["Works for one short document", "Context window limits: a 1,338 page library does not fit",
                                           "Every question pays to re-read everything", "No access control: the model sees it all"]),
        ("Fine-tune the model", (248, 113, 113), ["Retrains weights on your text", "Learns style and format well, facts poorly",
                                                  "Out of date the moment a new report lands", "Cannot cite sources, cannot forget a document",
                                                  "Needs GPUs, data prep and repeat runs"]),
        ("Retrieval (RAG)", GRN, ["Model stays as downloaded: no training", "New report? Add it to the index in seconds",
                                  "Answers cite report and page", "Permissions applied before the model sees text",
                                  "Delete a document, it is gone from answers"]),
    ]
    cw = (W - 160 - 80) / 3
    for i, (t, c, bl) in enumerate(cols):
        x = 80 + i * (cw + 40)
        d.rounded_rectangle((x, 170, x + cw, 1000), radius=22, fill=PANEL)
        d.rounded_rectangle((x, 170, x + cw, 180), radius=4, fill=c)
        d.text((x + 34, 205), t, font=F(40, True), fill=c)
        yy = 290
        for b in bl:
            words, line, lines = b.split(), "", []
            for w_ in words:
                if d.textlength(line + " " + w_, font=F(30)) > cw - 90:
                    lines.append(line); line = w_
                else:
                    line = (line + " " + w_).strip()
            lines.append(line)
            d.ellipse((x + 36, yy + 13, x + 50, yy + 27), fill=c)
            for ln in lines:
                d.text((x + 68, yy), ln, font=F(30), fill=FG)
                yy += 42
            yy += 22
    img.save(os.path.join(RESULTS, "fig_approaches.png"))


def embedding_map():
    from sentence_transformers import SentenceTransformer
    from sklearn.manifold import TSNE
    rows = read_jsonl(CHUNKS)
    emb = np.load(os.path.join(INDEX, "emb_bge-base.npy"))
    q = "How much barley will Western Australia produce this season?"
    m = SentenceTransformer(EMBEDDERS["bge-base"])
    qv = m.encode([QUERY_PREFIX["bge-base"] + q], normalize_embeddings=True)
    allv = np.vstack([emb, qv])
    xy = TSNE(n_components=2, perplexity=35, init="pca", random_state=0, metric="cosine").fit_transform(allv)
    groups = {"Australian Crop Report": "crop_report", "Agricultural Commodities": "ag_commodities",
              "Snapshot of Agriculture": "snapshot", "Farm surveys": ("farm_financial", "lending"),
              "Climate & drought": ("climate", "drought")}
    fig, ax = setup()
    for i, (lab, key) in enumerate(groups.items()):
        keys = key if isinstance(key, tuple) else (key,)
        idx = [j for j, r in enumerate(rows) if any(k in r["file"] for k in keys)]
        ax.scatter(xy[idx, 0], xy[idx, 1], s=16, color=SERIES[i], alpha=0.75, label=lab, linewidths=0)
    sims = emb @ qv[0]
    top = np.argsort(-sims)[:6]
    ax.scatter(xy[top, 0], xy[top, 1], s=160, facecolors="none", edgecolors=TEXT, linewidths=2.2, label="6 nearest chunks")
    ax.scatter(xy[-1, 0], xy[-1, 1], s=420, marker="*", color="#facc15", edgecolors=SURFACE, linewidths=1.5, zorder=5,
               label="the question")
    ax.annotate(f'"{q}"', xy[-1], xytext=(30, 40), textcoords="offset points", fontsize=19, color=TEXT,
                arrowprops=dict(arrowstyle="-", color=TEXT2))
    ax.set_xticks([]); ax.set_yticks([])
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(f"{len(rows):,} chunks as points: similar meaning lands close together (t-SNE of 768-d BGE vectors)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.01), markerscale=1.6, ncol=4, fontsize=18,
              columnspacing=1.6, handletextpad=0.4)
    save(fig, os.path.join(RESULTS, "fig_embedding_map.png"))


def index_table():
    b = json.load(open(os.path.join(RESULTS, "index_benchmark.json")))
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    d.text((80, 40), f"Six ways to store the same {b['n_chunks']:,} vectors", font=F(54, True), fill=FG)
    d.text((80, 118), "1,024-dimension Qwen3 embeddings, 200 test queries, top 10, measured on this workstation's CPU",
           font=F(28), fill=MUTED)
    cols = ["Store", "Kind", "Build (s)", "Query (ms)", "Recall@10", "Disk (MB)"]
    kind = {"FAISS Flat (exact)": "library, brute force", "FAISS HNSW": "library, graph index",
            "FAISS IVF-PQ": "library, compressed", "Chroma": "embedded database", "Qdrant (local)": "database (server-ready)",
            "LanceDB": "columnar files on disk"}
    xs = [80, 560, 1000, 1210, 1440, 1680]
    y = 220
    for x, c in zip(xs, cols):
        d.text((x, y), c, font=F(30, True), fill=ACC)
    d.line((80, y + 52, W - 80, y + 52), fill=(30, 41, 59), width=2)
    y += 80
    for name, v in b["stores"].items():
        vals = [name, kind.get(name, ""), f"{v['build_s']:.2f}", (f"{v['query_ms']:.2f}" if v['query_ms'] >= 0.01 else "< 0.01"), f"{v['recall@10']:.3f}",
                f"{v['disk_mb']:.1f}"]
        for i, (x, s) in enumerate(zip(xs, vals)):
            col = FG if i != 1 else MUTED
            if i == 4:
                col = GRN if v["recall@10"] >= 0.95 else AMB
            d.text((x, y), s, font=F(32, i == 0), fill=col)
        y += 92
        d.line((80, y - 20, W - 80, y - 20), fill=(22, 30, 46), width=1)
    d.text((80, 940), "Recall@10 = share of the exact top-10 that the approximate index also returns.",
           font=F(26), fill=MUTED)
    d.text((80, 980), "At a few thousand chunks every option is instant; the choice matters at millions, and for filters, "
                      "persistence and multi-user access.", font=F(26), fill=MUTED)
    img.save(os.path.join(RESULTS, "fig_index_bench.png"))


if __name__ == "__main__":
    architecture()
    approaches()
    if os.path.exists(os.path.join(RESULTS, "index_benchmark.json")):
        index_table()
        embedding_map()
    print("figures written to", RESULTS)
