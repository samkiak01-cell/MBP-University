"""
MBP University — Main Streamlit Application
"""

import logging
import streamlit as st
import anthropic
from sentence_transformers import SentenceTransformer

from config import (
    APP_TITLE,
    APP_DESCRIPTION,
    WELCOME_MESSAGE,
    SYSTEM_PROMPT,
    CLAUDE_MODEL,
    MAX_TOKENS,
    EMBEDDING_MODEL,
    DOCUMENTS_DIR,
)
from ingest import ingest_all, build_vector_store
from retriever import search, format_context, format_sources_for_display

# ── Logging (visible in Streamlit Cloud logs) ───────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mbp_university")

# ── Page Config ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MBP University",
    page_icon="🎓",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ──────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    [data-testid="stSidebar"] { background-color: #0e1a2b; }
    [data-testid="stSidebar"] * { color: #e0e6ed !important; }
    [data-testid="stSidebar"] .stMarkdown a { color: #6db3f2 !important; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ────────────────────────────────────────────────────────────────────────
# Cached resource loaders
# ────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading embedding model …")
def get_embedding_model():
    logger.info("Loading SentenceTransformer model …")
    return SentenceTransformer(EMBEDDING_MODEL)


@st.cache_resource(show_spinner="Indexing documents — this may take a minute on first boot …")
def build_index():
    """
    Ingest every document in the documents/ folder, embed them, and
    return (faiss_index, chunk_metadata_list, list_of_filenames).

    Everything lives in RAM — no disk writes needed.
    """
    logger.info(f"Documents directory: {DOCUMENTS_DIR}")

    # --- Diagnostic: list what's actually in the directory ---------------
    import os
    if os.path.isdir(DOCUMENTS_DIR):
        contents = os.listdir(DOCUMENTS_DIR)
        logger.info(f"Files in documents/: {contents}")
    else:
        logger.error(f"documents/ directory NOT FOUND at {DOCUMENTS_DIR}")
        return None, [], []

    # --- Ingest ----------------------------------------------------------
    chunks, filenames = ingest_all(DOCUMENTS_DIR)
    if not chunks:
        logger.warning("Ingestion returned 0 chunks.")
        return None, [], filenames

    # --- Embed & build FAISS index --------------------------------------
    model = get_embedding_model()
    index = build_vector_store(chunks, model)
    logger.info(f"Index ready: {index.ntotal} vectors, files={filenames}")
    return index, chunks, filenames


def get_anthropic_client():
    """Create Anthropic client from Streamlit secrets."""
    api_key = st.secrets.get("ANTHROPIC_API_KEY")
    if not api_key:
        return None
    return anthropic.Anthropic(api_key=api_key)


# ────────────────────────────────────────────────────────────────────────
# Sidebar
# ────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown(f"# {APP_TITLE}")
    st.markdown(f"*{APP_DESCRIPTION}*")
    st.divider()

    # Build / load the index (cached after first run)
    index, metadata, doc_filenames = build_index()

    if doc_filenames:
        st.markdown("**📚 Loaded Sources:**")
        for fn in doc_filenames:
            st.markdown(f"- `{fn}`")
    else:
        st.warning("No documents found in the documents/ folder.")

    st.divider()

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")
    st.caption("Powered by Claude · Anthropic")


# ────────────────────────────────────────────────────────────────────────
# Validate essentials
# ────────────────────────────────────────────────────────────────────────

client = get_anthropic_client()
if client is None:
    st.error(
        "⚠️ **Anthropic API key not found.** "
        "Add `ANTHROPIC_API_KEY` to your Streamlit Cloud Secrets "
        "(Settings → Secrets)."
    )
    st.stop()

if index is None or not metadata:
    st.error(
        "📭 **No documents indexed.** Make sure your `.docx` and `.xlsx` files "
        "are committed inside the `documents/` folder in your GitHub repo, "
        "then reboot the app."
    )
    st.stop()


# ────────────────────────────────────────────────────────────────────────
# Chat interface
# ────────────────────────────────────────────────────────────────────────

embed_model = get_embedding_model()

if "messages" not in st.session_state:
    st.session_state.messages = []

# Welcome message
if not st.session_state.messages:
    with st.chat_message("assistant", avatar="🎓"):
        st.markdown(WELCOME_MESSAGE)

# Render history
for msg in st.session_state.messages:
    avatar = "🎓" if msg["role"] == "assistant" else None
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("📄 View Sources"):
                st.markdown(msg["sources"])

# Handle new input
if prompt := st.chat_input("Ask MBP University a question …"):
    # Display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Retrieve context
    results = search(prompt, index, metadata, embed_model)
    context_block = format_context(results)
    sources_display = format_sources_for_display(results)

    # Build Claude messages (include recent history for follow-ups)
    claude_messages: list[dict] = []
    for m in st.session_state.messages[-11:-1]:  # last 10 turns before current
        claude_messages.append({"role": m["role"], "content": m["content"]})

    user_content = (
        f"## Retrieved Document Context\n\n{context_block}\n\n---\n\n"
        f"## User Question\n\n{prompt}"
    )
    claude_messages.append({"role": "user", "content": user_content})

    # Stream response
    with st.chat_message("assistant", avatar="🎓"):
        placeholder = st.empty()
        full_response = ""

        try:
            with client.messages.stream(
                model=CLAUDE_MODEL,
                max_tokens=MAX_TOKENS,
                system=SYSTEM_PROMPT,
                messages=claude_messages,
            ) as stream:
                for text in stream.text_stream:
                    full_response += text
                    placeholder.markdown(full_response + "▌")

            placeholder.markdown(full_response)

            with st.expander("📄 View Sources"):
                st.markdown(sources_display)

        except anthropic.APIError as e:
            full_response = f"⚠️ API error: {e.message}"
            placeholder.markdown(full_response)
            sources_display = ""
        except Exception as e:
            full_response = f"⚠️ Unexpected error: {e}"
            placeholder.markdown(full_response)
            sources_display = ""

    st.session_state.messages.append({
        "role": "assistant",
        "content": full_response,
        "sources": sources_display,
    })
