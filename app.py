"""Web UI for docu-ask: upload documents, ask questions, inspect sources."""

import os

from flask import Flask, jsonify, redirect, render_template, request, url_for
from werkzeug.utils import secure_filename

from rag import RAGIndex, load_documents
from answer import answer_question

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOCS_DIR = os.path.join(BASE_DIR, "docs")
ALLOWED_EXTS = {".txt", ".md"}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024  # 5 MB per upload

index = RAGIndex()


def _seed_sample_docs():
    """First run: copy the sample docs into docs/ so the UI works out of
    the box. Skipped as soon as the user has uploaded anything."""
    os.makedirs(DOCS_DIR, exist_ok=True)
    if os.listdir(DOCS_DIR):
        return
    sample_dir = os.path.join(BASE_DIR, "sample_docs")
    if not os.path.isdir(sample_dir):
        return
    for fname in os.listdir(sample_dir):
        if not fname.lower().endswith((".txt", ".md")):
            continue
        dst = os.path.join(DOCS_DIR, fname)
        if os.path.exists(dst):
            continue
        with open(os.path.join(sample_dir, fname), encoding="utf-8") as fsrc, \
             open(dst, "w", encoding="utf-8") as fdst:
            fdst.write(fsrc.read())


def rebuild_index():
    index.build(load_documents(DOCS_DIR))


_seed_sample_docs()
rebuild_index()


@app.route("/", methods=["GET", "POST"])
def ask_page():
    answer = None
    hits = []
    used_llm = False
    question = ""
    if request.method == "POST":
        question = request.form.get("question", "").strip()
        if question:
            hits = index.retrieve(question, k=5)
            answer, used_llm = answer_question(question, hits)
    return render_template("index.html", question=question, answer=answer,
                           hits=hits, used_llm=used_llm)


@app.route("/upload", methods=["GET", "POST"])
def upload():
    msg = None
    if request.method == "POST":
        f = request.files.get("file")
        if not f or not f.filename:
            msg = "Pick a file first."
        else:
            ext = os.path.splitext(f.filename)[1].lower()
            if ext not in ALLOWED_EXTS:
                msg = "Only .txt and .md files."
            else:
                os.makedirs(DOCS_DIR, exist_ok=True)
                f.save(os.path.join(DOCS_DIR, secure_filename(f.filename)))
                rebuild_index()  # new doc -> fresh index
                return redirect(url_for("documents"))
    return render_template("upload.html", msg=msg)


@app.route("/documents")
def documents():
    docs = load_documents(DOCS_DIR)
    info = [{"name": n, "words": len(t.split())} for n, t in docs]
    return render_template("documents.html", docs=info)


@app.route("/api/ask", methods=["POST"])
def api_ask():
    """JSON API: POST {"question": "...", "k": 5} -> answer + sources."""
    data = request.get_json(force=True, silent=True) or {}
    question = str(data.get("question", "")).strip()
    if not question:
        return jsonify({"error": "question is required"}), 400
    hits = index.retrieve(question, k=int(data.get("k", 5)))
    answer, used_llm = answer_question(question, hits)
    return jsonify({
        "answer": answer,
        "used_llm": used_llm,
        "sources": [
            {"doc": h["doc_name"], "chunk": h["chunk_id"],
             "score": round(h["score"], 4)}
            for h in hits
        ],
    })


if __name__ == "__main__":
    app.run(debug=True)
