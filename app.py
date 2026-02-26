"""
MBP University — Main Streamlit Application
"""

import os
import logging
import streamlit as st
import anthropic

from config import (
    APP_TITLE,
    APP_DESCRIPTION,
    WELCOME_MESSAGE,
    SYSTEM_PROMPT,
    CLAUDE_MODEL,
    MAX_TOKENS,
    EMBEDDING_MODEL,
    EMBEDDING_DIMENSION,
    DOCUMENTS_DIR,
)
from ingest import ingest_all, build_vector_store
from retriever import search, format_context, format_sources_for_display

# ── Logging (visible in Streamlit Cloud → Manage app → Logs) ────────────
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger("mbp_university")

# ── Page Config ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MBP University",
    page_icon="🎓",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ────────────────────────────────────────────────────────────────────────
# Custom CSS — myBasePay Brand Guidelines
#
#   Primary:   MBP Emerald #006633  |  Prussian Blue #121631
#              Wisteria Blue #7393f9  |  Alabaster Gray #e8e8e8
#   Secondary: Frosted Mint #e7fcdb  |  Icy Blue #b7d4f7
#              Marigold #ffbf00      |  White Smoke #f2f2f2
#   Font:      Roboto  |  Corners: 12px rounded
# ────────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* ── Load Roboto from Google Fonts ──────────────────────────── */
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;500;700;900&display=swap');

    html, body, [class*="css"] {
        font-family: 'Roboto', sans-serif !important;
    }

    /* ── Sidebar — Prussian Blue background ────────────────────── */
    [data-testid="stSidebar"] {
        background: #121631;
    }
    [data-testid="stSidebar"] * {
        color: #e8e8e8 !important;
    }
    [data-testid="stSidebar"] h1 {
        color: #ffffff !important;
        font-family: 'Roboto', sans-serif !important;
        font-weight: 900 !important;
        font-size: 1.5rem !important;
        margin-bottom: 0.25rem !important;
    }
    [data-testid="stSidebar"] .stMarkdown a {
        color: #7393f9 !important;
    }

    /* ── Sidebar source pills — Emerald tint ───────────────────── */
    [data-testid="stSidebar"] .source-item {
        background: rgba(0, 102, 51, 0.1);
        border: 1px solid rgba(0, 102, 51, 0.25);
        border-radius: 12px;
        padding: 8px 12px;
        margin-bottom: 6px;
        font-size: 0.78rem;
        line-height: 1.4;
        color: #b7d4f7 !important;
        word-wrap: break-word;
        overflow-wrap: break-word;
    }
    [data-testid="stSidebar"] .source-item .source-icon {
        margin-right: 6px;
    }

    /* ── Sidebar description ───────────────────────────────────── */
    [data-testid="stSidebar"] .sidebar-desc {
        font-size: 0.85rem;
        color: #afafaf !important;
        line-height: 1.5;
        margin-bottom: 0.5rem;
    }

    /* ── Sidebar divider ───────────────────────────────────────── */
    [data-testid="stSidebar"] hr {
        border-color: rgba(255, 255, 255, 0.08) !important;
    }

    /* ── Clear Chat button — Marigold hover ────────────────────── */
    [data-testid="stSidebar"] button {
        background: rgba(255, 255, 255, 0.06) !important;
        border: 1px solid rgba(255, 255, 255, 0.12) !important;
        color: #e8e8e8 !important;
        border-radius: 12px !important;
        font-family: 'Roboto', sans-serif !important;
        font-weight: 500 !important;
        transition: all 0.2s ease;
    }
    [data-testid="stSidebar"] button:hover {
        background: rgba(255, 191, 0, 0.15) !important;
        border-color: rgba(255, 191, 0, 0.4) !important;
        color: #ffbf00 !important;
    }

    /* ── Main chat input — Emerald focus ring ──────────────────── */
    .stChatInput > div {
        border-color: #e8e8e8 !important;
        border-radius: 12px !important;
    }
    .stChatInput > div:focus-within {
        border-color: #006633 !important;
        box-shadow: 0 0 0 1px #006633 !important;
    }

    /* ── Source expander — Emerald accent ───────────────────────── */
    .streamlit-expanderHeader {
        font-size: 0.85rem !important;
        color: #006633 !important;
        font-family: 'Roboto', sans-serif !important;
    }

    /* ── Global rounded corners on cards/containers ────────────── */
    [data-testid="stExpander"] {
        border-radius: 12px !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ────────────────────────────────────────────────────────────────────────
# Cached resource loaders  (kept FLAT — no nested cache calls)
# ────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading embedding model …")
def get_embedding_model():
    from sentence_transformers import SentenceTransformer
    logger.info("Loading SentenceTransformer model …")
    model = SentenceTransformer(EMBEDDING_MODEL)
    logger.info("Embedding model loaded.")
    return model


@st.cache_resource(show_spinner="Indexing documents — this may take a minute on first boot …")
def build_index(_embed_model):
    """
    Ingest + embed all documents.  Returns (index, metadata, filenames).
    """
    logger.info(f"DOCUMENTS_DIR resolved to: {DOCUMENTS_DIR}")
    logger.info(f"  exists?  {os.path.isdir(DOCUMENTS_DIR)}")

    if os.path.isdir(DOCUMENTS_DIR):
        contents = os.listdir(DOCUMENTS_DIR)
        logger.info(f"  contents: {contents}")
    else:
        logger.error(f"documents/ NOT FOUND at {DOCUMENTS_DIR}")
        parent = os.path.dirname(DOCUMENTS_DIR)
        if os.path.isdir(parent):
            logger.error(f"  parent dir ({parent}) contains: {os.listdir(parent)}")
        return None, [], []

    try:
        chunks, filenames = ingest_all(DOCUMENTS_DIR)
    except Exception as e:
        logger.error(f"Ingestion crashed: {e}", exc_info=True)
        return None, [], []

    if not chunks:
        logger.warning("Ingestion returned 0 chunks.")
        return None, [], filenames

    try:
        index = build_vector_store(chunks, _embed_model)
    except Exception as e:
        logger.error(f"Embedding/FAISS crashed: {e}", exc_info=True)
        return None, [], filenames

    logger.info(f"Index ready: {index.ntotal} vectors from {filenames}")
    return index, chunks, filenames


def get_anthropic_client():
    api_key = st.secrets.get("ANTHROPIC_API_KEY")
    if not api_key:
        return None
    return anthropic.Anthropic(api_key=api_key)


# ────────────────────────────────────────────────────────────────────────
# Helper: pretty filename for sidebar
# ────────────────────────────────────────────────────────────────────────

def _pretty_filename(fn: str) -> tuple[str, str]:
    """Return (icon, display_name) for a document filename."""
    if fn.lower().endswith(".xlsx"):
        return "📊", fn.replace("_", " ").replace("-", " ").rsplit(".", 1)[0]
    else:
        return "📄", fn.replace("_", " ").replace("-", " ").rsplit(".", 1)[0]


# ────────────────────────────────────────────────────────────────────────
# Sidebar
# ────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown(f"# {APP_TITLE}")
    st.markdown(
        f'<p class="sidebar-desc">{APP_DESCRIPTION}</p>',
        unsafe_allow_html=True,
    )
    st.divider()

    # Load model FIRST, then pass into build_index
    embed_model = get_embedding_model()
    index, metadata, doc_filenames = build_index(embed_model)

    if doc_filenames:
        st.markdown("**📚 Loaded Sources**")
        for fn in doc_filenames:
            icon, display = _pretty_filename(fn)
            st.markdown(
                f'<div class="source-item">'
                f'<span class="source-icon">{icon}</span>{display}'
                f'</div>',
                unsafe_allow_html=True,
            )
    else:
        st.warning("No documents found in the documents/ folder.")

    st.divider()

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


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
        "📭 **No documents indexed.**\n\n"
        f"- **Documents dir checked:** `{DOCUMENTS_DIR}`\n"
        f"- **Directory exists:** `{os.path.isdir(DOCUMENTS_DIR)}`\n"
        f"- **Contents:** `{os.listdir(DOCUMENTS_DIR) if os.path.isdir(DOCUMENTS_DIR) else 'N/A'}`\n\n"
        "Make sure `.docx` and `.xlsx` files are **committed** inside the "
        "`documents/` folder in your GitHub repo, then reboot the app."
    )
    st.stop()


# ────────────────────────────────────────────────────────────────────────
# Chat interface
# ────────────────────────────────────────────────────────────────────────

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
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Retrieve context
    results = search(prompt, index, metadata, embed_model)
    context_block = format_context(results)
    sources_display = format_sources_for_display(results)

    # Build Claude messages (include recent history for follow-ups)
    claude_messages: list[dict] = []
    for m in st.session_state.messages[-11:-1]:
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
