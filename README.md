# docu-ask

Ask questions over your own documents. Upload `.txt`/`.md` files, type a
question, and get an answer grounded in what you uploaded — with the source
chunks and similarity scores shown so you can verify it.

I built this to learn how retrieval-augmented generation (RAG) actually works
under the hood, without hiding behind a framework. The whole pipeline —
chunking, indexing, retrieval, answering — is plain code you can read.

## How it works

1. **Chunking** — each document is split into overlapping word windows
   (default 150 words, 30-word overlap) so no sentence gets cut off at a
   boundary.
2. **Indexing** — chunks are vectorized with TF-IDF (scikit-learn) and stored
   in a sparse matrix.
3. **Retrieval** — your question is vectorized the same way and ranked by
   cosine similarity. Top-k chunks come back with their scores.
4. **Answering** — two paths:
   - If `google-generativeai` is installed **and** `GEMINI_API_KEY` is set,
     Gemini writes a fluent answer from the retrieved chunks only, with
     source citations.
   - Otherwise (the default), you get an **extractive answer**: the top
     matching passages quoted with their document, chunk id, and score.
     Nothing crashes, nothing requires a key.

## Run it

```bash
pip install -r requirements.txt

# terminal
python ask.py "how do I factory reset the widget?" --docs sample_docs

# web UI
python app.py
# open http://127.0.0.1:5000
```

Uploaded files land in `docs/` and are indexed immediately. On first run the
three sample docs are copied into `docs/` automatically so the web UI works
right away. The files in `sample_docs/` are fictional (a product manual, a
remote-work policy, a travel guide) — just there so you can try it without
uploading anything.

Enable the LLM path:

```bash
pip install google-generativeai   # optional
export GEMINI_API_KEY="your-key"
python ask.py "what is the home office stipend?"
```

There's also a JSON API: `POST /api/ask` with `{"question": "...", "k": 5}`.

## Tests

```bash
pytest
```

Covers chunking math, retrieval picking the right document for obvious
queries, and the no-key fallback path.

## Limitations (honest ones)

- **TF-IDF is keyword matching, not understanding.** It won't connect
  synonyms ("laptop" vs "notebook") or grasp meaning the way embeddings do.
  For a small personal doc set it's fine; for anything serious I'd swap in
  sentence-transformers — the `retrieve()` signature is designed to survive
  that change (there's a TODO in `rag.py`).
- Chunking is a dumb word window — no respect for headings or paragraphs.
- The index rebuilds from scratch on every upload and app start. Fine for
  hundreds of docs; a real deployment would persist the matrix.
- The extractive fallback quotes passages verbatim — useful and truthful,
  but not a synthesized answer.

## Project layout

```
app.py            Flask web UI (ask, upload, documents, JSON API)
rag.py            chunking, TF-IDF index, retrieve()
answer.py         Gemini integration + extractive fallback
ask.py            CLI entry point
templates/        Jinja pages
static/css/       stylesheet
sample_docs/      3 fictional docs to try it with
tests/            pytest suite
```
