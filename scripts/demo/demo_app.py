"""Live demo page. Run it, wait for "Ready", then open http://localhost:8000

    python scripts/demo/demo_app.py

Ask a question over the HotpotQA paragraphs, or paste your own text and ask about that.
"""

import html
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs

import run_demo as D

from trustrag import features as F
from trustrag import rag
from trustrag.detector import Detector, check

COLOURS = {"show": "#d9ead3", "flag": "#fff2cc", "hold": "#f4cccc"}
PAGE = """<!doctype html><meta charset="utf-8"><title>TrustRAG demo</title>
<style>body{font:17px Arial;max-width:900px;margin:30px auto;padding:0 16px}textarea,input{width:100%%;font:16px Arial;
box-sizing:border-box;margin:4px 0 10px;padding:6px}button{font:16px Arial;padding:6px 16px}
.s{padding:8px;margin:6px 0}.m{color:#555;font-size:14px}.x{font-size:14px;padding:4px 10px;margin:0 6px 8px 0}</style>
<h2>TrustRAG demo</h2>
<p class="m">p is the estimated chance that a sentence is unsupported by the passages. SHOW at or below %(t_flag).2f, FLAG in between, HOLD at 0.50 or above. This checks grounding in the passages, not whether the answer is correct.</p>
<div id="examples"></div>
<form method="post" action="/ask#result"><b>Ask a question</b> <span class="m">(searches the HotpotQA paragraphs)</span>
<input name="question" value="%(question)s"><button>Retrieve, answer and check</button></form><hr>
<form method="post" action="/own#result"><b>Ask about your own text</b> <span class="m">(uses only what you paste here)</span>
<textarea name="source" rows="5" placeholder="Paste your source text">%(source)s</textarea>
<input name="source_question" placeholder="Your question about that text" value="%(source_question)s"><button>Answer from my text and check</button></form><hr>
<div id="result">%(result)s</div>
<script>const EX = %(examples)s;
EX.forEach(e => { const b = document.createElement("button"); b.className = "x"; b.type = "button"; b.textContent = e.label;
  b.onclick = () => ["question", "source", "source_question"].forEach(
    k => { document.getElementsByName(k)[0].value = e[k] || ""; });
  document.getElementById("examples").appendChild(b); });</script>"""

OWN_TEXT = ("TrustRAG checks every sentence of an answer against its source passages. It uses DeBERTa-v3-large, a "
            "natural language inference model with 0.4B parameters, and does not fine-tune it. The model's scores are "
            "combined with six simple word-matching features in a logistic regression. Each sentence is then shown, "
            "flagged or held back. On the RAGTruth QA test set it reaches an AUROC of 0.833. The whole check takes about "
            "0.5 seconds per answer on a laptop GPU.")
EXAMPLES = [
    {"label": "1. Ask a question", "question": "whats the famous bavagheet by arun date?"},
    {"label": "2. Your own source text", "source": OWN_TEXT,
     "source_question": "What model does TrustRAG use, and does it fine-tune it?"},
]


def render(claims, ms, retrieved=None, used=None):
    rows = []
    if used:
        rows.append('<p class="m">Used from your text: %s</p>' % html.escape(" | ".join(
            u if len(u) <= 220 else u[:220] + "..." for u in used)))
    if retrieved:
        rows.append('<p class="m">Retrieved: %s</p>' % html.escape(" | ".join(retrieved)))
    for c in claims:
        extra = "" if c["action"] == "show" else '<div class="m">Closest support: %s</div>' % html.escape(c["evidence"])
        rows.append('<div class="s" style="background:%s"><b>%s %.2f</b>&nbsp; %s'
                    '<div class="m">NLI support %.2f (from -1 to 1)</div>%s</div>'
                    % (COLOURS[c["action"]], c["action"].upper(), c["p"], html.escape(c["claim"]), c["support"], extra))
    rows.append('<p class="m">Checked in %.0f ms</p>' % ms)
    return "".join(rows)


print("Loading models...", flush=True)
detector = Detector()
titles, texts = D.corpus()
vectors = D.corpus_vectors(texts, "dev")
generator = rag.Generator()
check("This sentence loads the checking models.", ["This passage loads the checking models."], detector)


class Handler(BaseHTTPRequestHandler):
    def respond(self, **fields):
        values = {"question": "", "source": "", "source_question": "", "result": "",
                  "t_flag": detector.t_flag, "examples": json.dumps(EXAMPLES)}
        values.update({k: html.escape(v) if k != "result" else v for k, v in fields.items()})
        body = (PAGE % values).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        self.respond()

    def do_POST(self):
        size = int(self.headers.get("Content-Length", 0))
        form = {k: v[0].strip() for k, v in parse_qs(self.rfile.read(size).decode("utf-8")).items()}
        if self.path == "/ask" and form.get("question"):
            question = form["question"]
            idx = rag.retrieve([question], vectors)[0]
            passages = [texts[j] for j in idx]
            claims, ms = check(generator(rag.prompt_for(question, passages)), passages, detector)
            self.respond(question=question, result=render(claims, ms, [titles[j] for j in idx]))
        elif self.path == "/own" and form.get("source") and form.get("source_question"):
            # browsers send line breaks as CRLF, which changes how sentences get split
            source, question = form["source"].replace("\r\n", "\n"), form["source_question"]
            parts = rag.chunks_of(source)
            top = rag.retrieve([question], F.embed(parts), k=min(rag.TOP_K, len(parts)))[0]
            passages = [parts[j] for j in top]
            claims, ms = check(generator(rag.prompt_for(question, passages)), passages, detector)
            note = ""
            if len(source) > rag.MAX_SOURCE:
                note = '<p class="m">Your text is %s characters. Only the first %s were read.</p>' % (
                    format(len(source), ","), format(rag.MAX_SOURCE, ","))
            self.respond(source=source, source_question=question, result=note + render(claims, ms, used=passages))
        elif self.path == "/own":
            self.respond(source=form.get("source", ""), source_question=form.get("source_question", ""),
                         result='<p class="m">Please paste some text and type a question.</p>')
        else:
            self.respond()

    def log_message(self, *args):
        pass


if __name__ == "__main__":
    print("Ready: http://localhost:8000", flush=True)
    HTTPServer(("127.0.0.1", 8000), Handler).serve_forever()
