"""Ask the reports: a private, local RAG chat over Australian agriculture reports.

    python app.py            (opens http://127.0.0.1:7860)

Everything runs on this machine: the embedding model, the vector index, the reranker
and the LLM. Each question is written to an audit log with the user's role and the
sources that were shown.
"""
import datetime as dt
import gc
import html
import json
import os
import time

import gradio as gr
import torch

from common import DATA, DEFAULT_LLM, LLMS, ROLE_CAN_READ
from rag import (SYSTEM_PLAIN, SYSTEM_RAG, Generator, Retriever, allowed_for, rag_prompt, used_citations)

AUDIT = os.path.join(DATA, "audit_log.jsonl")
RET = Retriever()
STATE = {"gen": None}

EXAMPLES = [
    "What is ABARES forecasting for Australia's winter crop production in 2025-26, and how does it compare with the 10-year average?",
    "How much barley is Western Australia expected to produce this season?",
    "Why did South Australia's crop forecast fall in 2024-25?",
    "How has Australia's climate changed since 1910 according to the State of the Climate report?",
    "What was the average farm cash income for broadacre farms?",
    "What is the CEO's salary?",
]


def get_gen(name):
    if STATE["gen"] is None or STATE["gen"].name != name:
        STATE["gen"] = None
        gc.collect(); torch.cuda.empty_cache()
        STATE["gen"] = Generator(name)
    return STATE["gen"]


def sources_html(hits, cited):
    if not hits:
        return "<div class='src-empty'>No sources retrieved.</div>"
    out = []
    for n, h in enumerate(hits, start=1):
        cls = "src cited" if n in cited else "src"
        badge = f"<span class='acc acc-{h['access']}'>{h['access']}</span>"
        link = f"<a href='{html.escape(h['url'])}' target='_blank'>open report</a>" if h.get("url") else ""
        txt = html.escape(h["text"][:900]) + ("..." if len(h["text"]) > 900 else "")
        out.append(f"<div class='{cls}'><div class='src-h'><b>[{n}]</b> {html.escape(h['title'])} "
                   f"&middot; p.{h['page']} {badge} {link}</div><div class='src-t'>{txt}</div></div>")
    return "".join(out)


def respond(message, history, role, llm, use_rag, mode, k):
    history = history + [{"role": "user", "content": message}, {"role": "assistant", "content": ""}]
    t0 = time.time()
    gen = get_gen(llm)
    hits, cited = [], []
    if use_rag:
        hits = RET.search(message, k=int(k), mode=mode, allowed=allowed_for(role))
        stream = gen.stream(SYSTEM_RAG, rag_prompt(message, hits))
    else:
        stream = gen.stream(SYSTEM_PLAIN, message)
    t_ret = time.time() - t0
    src = sources_html(hits, cited)
    for piece in stream:
        history[-1]["content"] += piece
        yield history, src, ""
    cited = used_citations(history[-1]["content"], len(hits))
    meta = (f"{llm} · {'RAG: ' + mode + f', top {int(k)}' if use_rag else 'no retrieval (model memory only)'}"
            f" · retrieval {t_ret:.2f}s · total {time.time() - t0:.1f}s · role: {role}")
    with open(AUDIT, "a", encoding="utf-8") as f:
        f.write(json.dumps({"time": dt.datetime.now().isoformat(timespec="seconds"), "role": role, "llm": llm,
                            "rag": use_rag, "question": message, "sources": [h["id"] for h in hits],
                            "cited": cited}) + "\n")
    yield history, sources_html(hits, cited), meta


CSS = """
.gradio-container {max-width: 1500px !important}
#title h1 {margin-bottom: 0}
.src {border:1px solid var(--border-color-primary); border-radius:10px; padding:10px 12px; margin-bottom:10px; opacity:.75}
.src.cited {opacity:1; border-color:#2e9e5b; box-shadow:0 0 0 1px #2e9e5b inset}
.src-h {font-size:13px; margin-bottom:6px}
.src-t {font-size:12.5px; white-space:pre-wrap; line-height:1.4; max-height:180px; overflow:auto}
.acc {font-size:11px; padding:1px 7px; border-radius:9px; margin-left:4px; color:white}
.acc-public {background:#2d6cdf} .acc-internal {background:#d68910} .acc-restricted {background:#c0392b}
#meta {font-size:12.5px; opacity:.8}
"""

with gr.Blocks(title="Ask the reports - local RAG") as demo:
    gr.Markdown("# Ask the reports\nA private question-answering system over Australian government agriculture "
                "and climate reports. Retrieval, reranking and the language model all run on this computer; "
                "no question or document leaves it.", elem_id="title")
    with gr.Row():
        with gr.Column(scale=1, min_width=250):
            role = gr.Dropdown(list(ROLE_CAN_READ), value="Staff", label="Signed in as (access level)")
            llm = gr.Dropdown(list(LLMS), value=DEFAULT_LLM, label="Language model (open weights, local)")
            use_rag = gr.Checkbox(True, label="Use retrieval (RAG)")
            mode = gr.Radio(["hybrid+rerank", "hybrid", "dense", "bm25"], value="hybrid+rerank", label="Retrieval")
            k = gr.Slider(2, 12, value=6, step=1, label="Sources given to the model")
            gr.Markdown(f"Index: {len(RET.rows):,} chunks from {len({r['file'] for r in RET.rows})} reports")
        with gr.Column(scale=3):
            chat = gr.Chatbot(height=560, buttons=["copy"], label="Answer")
            box = gr.Textbox(placeholder="Ask a question about the reports...", show_label=False, autofocus=True)
            meta = gr.Markdown(elem_id="meta")
            gr.Examples(EXAMPLES, inputs=box)
        with gr.Column(scale=2):
            gr.Markdown("### Sources (green = cited in the answer)")
            src = gr.HTML()
    box.submit(respond, [box, chat, role, llm, use_rag, mode, k], [chat, src, meta]).then(lambda: "", None, box)
    gr.ClearButton([chat, src, meta, box], value="New conversation")

if __name__ == "__main__":
    demo.queue().launch(server_name="127.0.0.1", server_port=int(os.environ.get("PORT", 7860)),
                        css=CSS, theme=gr.themes.Soft())
