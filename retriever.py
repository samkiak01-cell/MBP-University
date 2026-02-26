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
    """
    Clean, grouped source list for the UI expander.
    Groups by document, deduplicates sections, and shows FAQ resources.
    """
    if not results:
        return "No sources found."

    # Group: SOP sources by filename, FAQ sources separately
    sop_sources: dict[str, list[str]] = {}  # filename → list of section names
    faq_sources: list[dict] = []
    seen_faq: set[str] = set()

    for r in results:
        if r.get("type") == "faq":
            q = r.get("question", "")
            if q in seen_faq:
                continue
            seen_faq.add(q)
            faq_sources.append(r)
        else:
            fname = r.get("source", "Unknown")
            section = r.get("section", "N/A")
            if fname not in sop_sources:
                sop_sources[fname] = []
            if section not in sop_sources[fname]:
                sop_sources[fname].append(section)

    lines: list[str] = []

    # SOP sources grouped by document
    for fname, sections in sop_sources.items():
        display_name = fname.replace("_", " ").replace("-", " ").rsplit(".", 1)[0]
        lines.append(f"📄 **{display_name}**")
        for sec in sections:
            lines.append(f"&nbsp;&nbsp;&nbsp;&nbsp;• {sec}")
        lines.append("")  # blank line between groups

    # FAQ sources
    for r in faq_sources:
        topic = r.get("topic", "General")
        question = r.get("question", "N/A")
        lines.append(f"📊 **MBP University FAQ** — {topic}")
        lines.append(f"&nbsp;&nbsp;&nbsp;&nbsp;• {question}")
        if r.get("resource_link"):
            lines.append(f"&nbsp;&nbsp;&nbsp;&nbsp;🔗 {r['resource_link']}")
        lines.append("")

    return "\n".join(lines).strip()
