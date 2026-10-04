#!/usr/bin/env python3
"""Ask questions over your documents from the terminal.

Usage:
    python ask.py "how do I reset the widget?" [--docs sample_docs] [--k 5]
"""

import argparse

from rag import RAGIndex, load_documents
from answer import answer_question


def main():
    p = argparse.ArgumentParser(description="Ask questions over your documents.")
    p.add_argument("question", help="the question to ask")
    p.add_argument("--docs", default="sample_docs",
                   help="folder with .txt/.md docs (default: sample_docs)")
    p.add_argument("--k", type=int, default=5, help="chunks to retrieve")
    p.add_argument("--no-llm", action="store_true",
                   help="skip Gemini even if it's configured")
    args = p.parse_args()

    docs = load_documents(args.docs)
    if not docs:
        print(f"No .txt/.md documents found in {args.docs}")
        return 1

    index = RAGIndex()
    index.build(docs)
    print(f"Indexed {len(docs)} document(s), {len(index.chunks)} chunks.\n")

    hits = index.retrieve(args.question, k=args.k)
    answer, used_llm = answer_question(args.question, hits,
                                       allow_llm=not args.no_llm)

    print(answer)
    print("\n--- sources ---")
    for h in hits:
        print(f'{h["doc_name"]} chunk {h["chunk_id"]}  (score {h["score"]:.3f})')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
