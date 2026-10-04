"""Answer generation: Gemini when it's available, extractive fallback otherwise.

The fallback is the default path — the app never requires an API key and
never crashes when the key or the package is missing.
"""

import os


def _gemini_configured():
    return bool(os.environ.get("GEMINI_API_KEY"))


def extractive_answer(question, chunks, max_chunks=3):
    """Answer by quoting the most relevant chunks. Always works, no key."""
    if not chunks:
        return "I couldn't find anything relevant in your documents."
    lines = ["Based on your documents:\n"]
    for c in chunks[:max_chunks]:
        snippet = c["text"].strip()
        if len(snippet) > 600:
            snippet = snippet[:600].rstrip() + "..."
        lines.append(
            f'[{c["doc_name"]} — chunk {c["chunk_id"]}, score {c["score"]:.2f}]'
        )
        lines.append(f'"{snippet}"\n')
    return "\n".join(lines)


def gemini_answer(question, chunks, max_chunks=4):
    """Ask Gemini for a fluent answer grounded in the retrieved chunks."""
    import google.generativeai as genai  # optional dependency

    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
    context = "\n\n".join(
        f'[Source: {c["doc_name"]} chunk {c["chunk_id"]}]\n{c["text"]}'
        for c in chunks[:max_chunks]
    )
    prompt = (
        "Answer the question using ONLY the context below. "
        "If the context doesn't contain the answer, say so plainly. "
        "Cite your sources like [doc-name chunk N].\n\n"
        f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    )
    model = genai.GenerativeModel("gemini-2.0-flash")
    return model.generate_content(prompt).text.strip()


def answer_question(question, chunks, allow_llm=True):
    """Answer a question. Returns (answer_text, used_llm).

    Falls back to extractive passages whenever the LLM isn't configured
    or errors out — an answer is always returned.
    """
    if allow_llm and _gemini_configured():
        try:
            return gemini_answer(question, chunks), True
        except Exception as e:  # network, quota, bad key — degrade gracefully
            note = (
                f"(LLM unavailable: {type(e).__name__}; "
                "showing extracted passages instead)\n\n"
            )
            return note + extractive_answer(question, chunks), False
    return extractive_answer(question, chunks), False
