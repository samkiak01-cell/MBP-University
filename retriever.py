"""
MBP University — Retriever

Similarity search over the FAISS index and context formatting for Claude.
"""

import numpy as np
from sentence_transformers import SentenceTransformer

from config import TOP_K


def search(
    query: str,
    index,
    metadata: list[dict],
    model: SentenceTransformer,
    top_k: int = TOP_K,
) -> list[dict]:
    """Embed the query and return the top-k most relevant chunks."""
    if index is None or not metadata:
        return []

    query_vec = model.encode([query], normalize_embeddings=True).astype("float32")
    scores, indices = index.search(query_vec, min(top_k, index.ntotal))

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue
        chunk = metadata[idx].copy()
        chunk["score"] = float(score)
        results.append(chunk)

    return results


def format_context(results: list[dict]) -> str:
    """Format retrieved chunks into a context block for Claude."""
    if not results:
        return "(No relevant documents found.)"

    parts = []
    for i, r in enumerate(results, 1):
        header = [f"[Source {i}]"]
        header.append(f"File: {r.get('source', 'Unknown')}")
        if r.get("type") == "faq":
            header.append("Type: FAQ")
            if r.get("topic"):
                header.append(f"Topic: {r['topic']}")
            if r.get("location"):
                header.append(f"Location: {r['location']}")
            if r.get("question"):
                header.append(f"Question: {r['question']}")
            if r.get("resource_link"):
                header.append(f"Resource Link: {r['resource_link']}")
        else:
            header.append("Type: SOP")
            if r.get("section"):
                header.append(f"Section: {r['section']}")

        parts.append(f"{' | '.join(header)}\n{r['text']}")

    return "\n\n---\n\n".join(parts)


def format_sources_for_display(results: list[dict]) -> str:
    """Human-readable source list for the UI expander."""
    if not results:
        return "No sources found."

    lines = []
    seen: set[str] = set()
    for r in results:
        if r.get("type") == "faq":
            key = f"faq-{r.get('question', '')}"
            if key in seen:
                continue
            seen.add(key)
            line = (
                f"• **MBP University FAQ** → Topic: {r.get('topic', 'N/A')}, "
                f"Row: \"{r.get('question', 'N/A')}\""
            )
            lines.append(line)
            if r.get("resource_link"):
                lines.append(f"  🔗 Related Resource: {r['resource_link']}")
        else:
            key = f"sop-{r.get('source', '')}-{r.get('section', '')}"
            if key in seen:
                continue
            seen.add(key)
            line = (
                f"• **{r.get('source', 'Unknown')}** → "
                f"Section: \"{r.get('section', 'N/A')}\""
            )
            lines.append(line)

    return "\n".join(lines)
