"""
MBP University — Configuration Constants
"""

import os

# ── Paths (resolved relative to THIS file so it works on Streamlit Cloud) ─
_APP_DIR = os.path.dirname(os.path.abspath(__file__))
DOCUMENTS_DIR = os.path.join(_APP_DIR, "documents")

# ── Chunking ────────────────────────────────────────────────────────────
CHUNK_MAX_TOKENS = 1000
CHUNK_OVERLAP_TOKENS = 100

# ── Retrieval ───────────────────────────────────────────────────────────
TOP_K = 5

# ── Embedding model (HuggingFace — free, no API key needed) ────────────
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384

# ── LLM ─────────────────────────────────────────────────────────────────
CLAUDE_MODEL = "claude-sonnet-4-20250514"
MAX_TOKENS = 2048

# ── UI ──────────────────────────────────────────────────────────────────
APP_TITLE = "🎓 MBP University"
APP_DESCRIPTION = (
    "Your internal knowledge assistant. Ask me anything about our "
    "SOPs, benefits, payroll, contracts, and more."
)
WELCOME_MESSAGE = (
    "👋 Welcome to MBP University! I'm here to help you find answers "
    "from our company SOPs and FAQ. Ask me anything!"
)
FALLBACK_ANSWER = (
    "I wasn't able to find an answer to that in our current knowledge base. "
    "Please reach out to your manager or the relevant department for assistance."
)

# ── System prompt for Claude ────────────────────────────────────────────
SYSTEM_PROMPT = f"""\
You are **MBP University**, a helpful, professional, and friendly internal \
company knowledge assistant. Your job is to answer employee questions using \
ONLY the document context provided to you.

Rules you MUST follow:
1. Answer STRICTLY from the provided context. Never invent or assume information.
2. If the context does not contain enough information to answer, respond with: \
"{FALLBACK_ANSWER}"
3. Do NOT include any source citations, references, or document names in your \
answer. The application will automatically display sources separately. Just \
answer the question directly.
4. If the user's question relates to a specific location (e.g., US, Hawaii), \
pay close attention to any Location metadata in the FAQ context.
5. Be concise yet thorough. Use bullet points when listing steps or multiple items.
6. Keep your tone professional and friendly.
"""
