"""Chunking + TF-IDF retrieval for docu-ask.

Nothing fancy: word-window chunking and scikit-learn TF-IDF with cosine
similarity. Plenty for a small personal doc collection, and the retrieve()
signature stays the same if I ever swap in embeddings.
"""

import os
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def load_documents(docs_dir):
    """Read every .txt/.md file in docs_dir -> [(name, text), ...]."""
    docs = []
    if not os.path.isdir(docs_dir):
        return docs
    for fname in sorted(os.listdir(docs_dir)):
        if not fname.lower().endswith((".txt", ".md")):
            continue
        path = os.path.join(docs_dir, fname)
        try:
            with open(path, encoding="utf-8") as f:
                docs.append((fname, f.read()))
        except (OSError, UnicodeDecodeError):
            # skip unreadable files instead of blowing up the whole index
            continue
    return docs


def chunk_text(text, doc_name, chunk_words=150, overlap=30):
    """Split text into overlapping word-window chunks.

    Returns a list of dicts: {doc_name, chunk_id, text}.
    """
    words = text.split()
    chunks = []
    if not words:
        return chunks
    step = max(chunk_words - overlap, 1)
    idx = 0
    for start in range(0, len(words), step):
        piece = words[start:start + chunk_words]
        if not piece:
            break
        chunks.append({
            "doc_name": doc_name,
            "chunk_id": idx,
            "text": " ".join(piece),
        })
        idx += 1
        if start + chunk_words >= len(words):
            break
    return chunks


class RAGIndex:
    """TF-IDF index over document chunks with cosine-similarity retrieval."""

    def __init__(self, chunk_words=150, overlap=30):
        self.chunk_words = chunk_words
        self.overlap = overlap
        self.chunks = []
        self.vectorizer = None
        self.matrix = None

    def build(self, documents):
        """Build the index. documents: list of (doc_name, text)."""
        self.chunks = []
        for doc_name, text in documents:
            self.chunks.extend(
                chunk_text(text, doc_name, self.chunk_words, self.overlap)
            )
        if not self.chunks:
            self.vectorizer, self.matrix = None, None
            return
        # max_features keeps the matrix small on bigger doc sets
        self.vectorizer = TfidfVectorizer(stop_words="english", max_features=20000)
        self.matrix = self.vectorizer.fit_transform(c["text"] for c in self.chunks)

    def retrieve(self, query, k=5):
        """Top-k chunks for query, highest cosine score first.

        Each hit: {doc_name, chunk_id, text, score}. Zero-score hits dropped.
        """
        if not self.chunks or self.vectorizer is None:
            return []
        q_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(q_vec, self.matrix)[0]
        order = scores.argsort()[::-1][:k]
        hits = []
        for i in order:
            if scores[i] <= 0:
                continue
            c = self.chunks[int(i)]
            hits.append({
                "doc_name": c["doc_name"],
                "chunk_id": c["chunk_id"],
                "text": c["text"],
                "score": float(scores[i]),
            })
        return hits

    # TODO: embeddings upgrade — replace TfidfVectorizer with sentence-transformers
    # and cosine_similarity with a dot product over normalized vectors. The chunk
    # dicts and retrieve() signature can stay exactly the same.

    def save(self, path):
        """Persist the built index so the web app doesn't rebuild on boot."""
        with open(path, "wb") as f:
            pickle.dump({
                "chunks": self.chunks,
                "vectorizer": self.vectorizer,
                "matrix": self.matrix,
                "chunk_words": self.chunk_words,
                "overlap": self.overlap,
            }, f)

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            data = pickle.load(f)
        idx = cls(data["chunk_words"], data["overlap"])
        idx.chunks = data["chunks"]
        idx.vectorizer = data["vectorizer"]
        idx.matrix = data["matrix"]
        return idx
