"""Tests for docu-ask chunking, retrieval, and the LLM fallback path."""

import os

import pytest

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
