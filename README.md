# 🎓 MBP University — Internal Knowledge Base Agent

An AI-powered internal knowledge assistant that lets employees ask questions about company SOPs, benefits, payroll, contracts, and more — and get accurate, sourced answers instantly.

---

## How It Works

1. **Document Ingestion** — On app startup, all `.docx` SOP files and `.xlsx` FAQ files in the `documents/` folder are automatically parsed. SOP section boundaries are detected by looking for bold uppercase text patterns (since our SOPs use bold List Paragraph formatting, not Word heading styles). The FAQ Excel is parsed row by row, preserving topic, location, and resource link metadata.

2. **Chunking & Embedding** — SOP sections are split into overlapping chunks (up to ~1,000 tokens each) to keep search results precise. Each chunk is embedded into a numerical vector using the `all-MiniLM-L6-v2` model (free, runs locally, no API key needed). All vectors are stored in an in-memory FAISS index.

3. **Retrieval** — When a user asks a question, the query is embedded and matched against the FAISS index. The top 5 most relevant chunks are retrieved.

4. **Answer Generation** — The retrieved chunks are sent to Claude (Anthropic) as context. Claude answers strictly from that context and never makes anything up. If the answer isn't in the documents, it says so. Sources are displayed separately in a collapsible panel below each answer.

---

## Project Structure

```
mbp-university/
├── app.py                 # Streamlit UI, chat loop, streaming Claude responses
├── config.py              # All settings: paths, models, system prompt, UI text
├── ingest.py              # Document parsing (.docx + .xlsx), chunking, FAISS indexing
├── retriever.py           # Similarity search + source formatting for the UI
├── requirements.txt       # Python dependencies
├── .gitignore             # Keeps secrets.toml out of the repo
├── README.md
├── .streamlit/
│   └── config.toml        # Streamlit theme (brand colors)
└── documents/
    ├── MBP_University-_My_Professors_Brain.xlsx
    └── *.docx SOP files
```

---

## Adding New Documents

1. Drop any `.docx` SOP or `.xlsx` FAQ file into the `documents/` folder
2. Commit and push to GitHub
3. Reboot the app in Streamlit Cloud (⋮ menu → Reboot app)

The app re-indexes automatically on every reboot. No code changes needed.

---

## Tech Stack

| Component | Technology |
|---|---|
| Frontend & Hosting | Streamlit (Community Cloud) |
| LLM | Claude Sonnet via Anthropic API |
| Embeddings | `all-MiniLM-L6-v2` (free, no API key) |
| Vector Store | FAISS (in-memory) |
| Document Parsing | `python-docx` for SOPs, `openpyxl` for FAQ Excel |

---

## Security

All API keys are managed through Streamlit's built-in secrets system (`st.secrets`). No keys are ever hardcoded in the source code. The `.streamlit/secrets.toml` file is gitignored and never committed.

---

## Troubleshooting

| Issue | Fix |
|---|---|
| "API key not found" | Add `ANTHROPIC_API_KEY` in Streamlit Cloud → Settings → Secrets |
| "No documents indexed" | Make sure files are committed in `documents/`, not gitignored |
| Slow first load | Normal — embedding model downloads once (~90 MB), then cached |
| Answers seem outdated after adding docs | Reboot the app to trigger re-indexing |
