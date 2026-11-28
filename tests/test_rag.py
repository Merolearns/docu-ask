"""Tests for docu-ask chunking, retrieval, and the LLM fallback path."""

import os

import pytest
from sklearn.feature_extraction.text import TfidfVectorizer

from rag import RAGIndex, chunk_text, load_documents
from answer import answer_question, extractive_answer

SAMPLE_DIR = os.path.join(os.path.dirname(__file__), "..", "sample_docs")


def test_chunking_counts():
    # 400 words, 100-word windows, 30-word overlap -> step of 70
    # starts: 0, 70, 140, 210, 280, 350 -> 6 chunks
    text = " ".join(f"word{i}" for i in range(400))
    chunks = chunk_text(text, "test.md", chunk_words=100, overlap=30)
    assert len(chunks) == 6
    assert chunks[0]["chunk_id"] == 0
    assert chunks[0]["doc_name"] == "test.md"
    # overlap check: last 30 words of chunk 0 == first 30 of chunk 1
    assert chunks[0]["text"].split()[-30:] == chunks[1]["text"].split()[:30]


def test_chunking_short_text_is_single_chunk():
    chunks = chunk_text("hello world", "a.md")
    assert len(chunks) == 1
    assert chunks[0]["text"] == "hello world"


def test_chunking_empty_text():
    assert chunk_text("", "a.md") == []
    assert chunk_text("   ", "a.md") == []


def test_load_documents_reads_md_and_txt(tmp_path):
    (tmp_path / "a.md").write_text("hello")
    (tmp_path / "b.txt").write_text("world")
    (tmp_path / "c.pdf").write_text("ignored")
    docs = load_documents(str(tmp_path))
    assert sorted(n for n, _ in docs) == ["a.md", "b.txt"]


@pytest.fixture
def index():
    docs = load_documents(SAMPLE_DIR)
    assert len(docs) == 3, "expected the 3 sample docs"
    idx = RAGIndex(chunk_words=120, overlap=25)
    idx.build(docs)
    return idx


def test_retrieval_picks_widget_manual(index):
    hits = index.retrieve("how do I factory reset the widget?", k=3)
    assert hits, "expected at least one hit"
    assert hits[0]["doc_name"] == "acme-widget-manual.md"


def test_retrieval_picks_remote_policy(index):
    hits = index.retrieve("what is the home office stipend amount?", k=3)
    assert hits, "expected at least one hit"
    assert hits[0]["doc_name"] == "northwind-remote-policy.md"


def test_retrieval_picks_travel_guide(index):
    hits = index.retrieve("best time of year to visit Lisbon", k=3)
    assert hits, "expected at least one hit"
    assert hits[0]["doc_name"] == "lisbon-travel-guide.md"


def test_retrieval_scores_descending(index):
    hits = index.retrieve("battery charging warranty", k=5)
    scores = [h["score"] for h in hits]
    assert scores == sorted(scores, reverse=True)
    assert all(s > 0 for s in scores)


def test_retrieve_empty_index():
    idx = RAGIndex()
    idx.build([])
    assert idx.retrieve("anything") == []


def test_build_stop_words_only_doc_is_empty_index_not_crash():
    # a doc of nothing but English stop words used to crash the build
    # with "empty vocabulary", taking the app down on boot and 500ing
    # the upload that added it
    idx = RAGIndex()
    idx.build([("stop.txt", "the and of the")])
    assert idx.retrieve("the") == []


def test_build_stop_words_only_doc_does_not_poison_other_docs():
    # the stop-words doc contributes no vocabulary, the good doc still
    # answers - the empty-vocabulary guard must not swallow real docs
    idx = RAGIndex()
    idx.build([
        ("good.md", "the widget factory reset button is behind the battery cover"),
        ("stop.txt", "the and of the"),
    ])
    hits = idx.retrieve("factory reset widget", k=3)
    assert hits and hits[0]["doc_name"] == "good.md"


def test_retrieve_nonpositive_k_returns_nothing(index):
    # k=0 or negative must not slice backwards into almost-all hits
    assert index.retrieve("factory reset widget", k=0) == []
    assert index.retrieve("factory reset widget", k=-1) == []


def test_extractive_answer_quotes_chunks(index):
    hits = index.retrieve("factory reset widget", k=2)
    text = extractive_answer("how do I reset it?", hits)
    assert "acme-widget-manual.md" in text
    assert "chunk" in text


def test_extractive_answer_no_hits():
    assert "couldn't find" in extractive_answer("q", [])


def test_answer_falls_back_without_key(index, monkeypatch):
    # no GEMINI_API_KEY -> extractive path, never raises
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    hits = index.retrieve("factory reset", k=2)
    text, used_llm = answer_question("how do I reset?", hits, allow_llm=True)
    assert used_llm is False
    assert "acme-widget-manual.md" in text


def test_answer_no_llm_flag(index):
    hits = index.retrieve("factory reset", k=2)
    text, used_llm = answer_question("how do I reset?", hits, allow_llm=False)
    assert used_llm is False
    assert text  # still got an answer


def test_retrieve_during_rebuild_uses_old_index(monkeypatch):
    # an ask landing mid-rebuild must see the complete old index, never
    # the half-built one - build() used to swap in a fresh unfitted
    # vectorizer before fit_transform ran, so retrieve() raised
    # NotFittedError and the ask 500ed
    idx = RAGIndex()
    idx.build([("a.md", "the widget factory reset button is behind the battery cover")])
    assert idx.retrieve("factory reset widget")

    seen = {}
    orig_fit = TfidfVectorizer.fit_transform

    def fit_and_ask(self, raw_documents, y=None):
        # exactly what a concurrent ask sees: the new chunks are in and
        # the new vectorizer exists, but fit hasn't run yet. Record the
        # outcome instead of letting it escape - on the old code the
        # NotFittedError would be swallowed by build()'s ValueError
        # handler (NotFittedError subclasses ValueError) and the build
        # would silently finish as an empty index.
        try:
            seen["hits"] = idx.retrieve("factory reset widget")
        except Exception as e:  # noqa: BLE001 - the error IS the signal
            seen["error"] = e
        return orig_fit(self, raw_documents, y)

    monkeypatch.setattr(TfidfVectorizer, "fit_transform", fit_and_ask)
    idx.build([("b.md", "lisbon travel guide best time to visit the city in spring")])
    assert "error" not in seen, seen["error"]
    assert seen["hits"], "expected the old index's hits mid-rebuild"
    assert seen["hits"][0]["doc_name"] == "a.md"
    # and the new index took over once the build finished
    hits = idx.retrieve("lisbon travel spring", k=3)
    assert hits and hits[0]["doc_name"] == "b.md"


def test_concurrent_rebuild_and_retrieve_never_raises():
    # rebuilds and retrieves racing each other must never surface a
    # half-built index; bigger docs widen the fit window so the race
    # actually gets exercised
    import threading

    idx = RAGIndex()
    big_a = " ".join(["widget factory reset button battery cover"] * 400)
    big_b = " ".join(["lisbon travel guide visit city spring"] * 400)
    docs_a = [("a.md", big_a)]
    docs_b = [("b.md", big_b)]
    idx.build(docs_a)
    errors = []
    stop = threading.Event()

    def retriever():
        while not stop.is_set():
            try:
                idx.retrieve("factory reset widget")
            except Exception as e:  # noqa: BLE001 - any error is a failure
                errors.append(e)

    t = threading.Thread(target=retriever)
    t.start()
    try:
        for i in range(20):
            idx.build(docs_a if i % 2 == 0 else docs_b)
    finally:
        stop.set()
        t.join()
    assert not errors, errors[:3]
