# 🎓 MBP University — Internal Knowledge Base Agent

AI-powered internal knowledge assistant built with **Streamlit**, **Claude (Anthropic)**, and **FAISS**. Employees ask questions about company SOPs, benefits, payroll, contracts — and get accurate, sourced answers.

---

## How It Works

1. **Ingestion:** On app boot, all `.docx` and `.xlsx` files in `documents/` are parsed. SOP sections are detected via bold-text heading patterns. FAQ rows become individual searchable chunks.
2. **Embedding:** Chunks are embedded with `all-MiniLM-L6-v2` (free, no API key) and stored in an in-memory FAISS index.
3. **Retrieval:** User queries are embedded and matched against the index (top 5 results).
4. **Generation:** Claude answers strictly from retrieved context, with source citations.

---

## Deploy to Streamlit Community Cloud

### 1. Prepare your GitHub repo

Your repo should look like this:

```
mbp-university/
├── app.py
├── ingest.py
├── retriever.py
├── config.py
├── requirements.txt
├── .gitignore
├── README.md
├── .streamlit/
│   └── config.toml
└── documents/
    ├── MBP_University-_My_Professors_Brain.xlsx
    └── Standard_Operating_Procedure__SOP__-_....docx
```

> ⚠️ **The files inside `documents/` MUST be committed to the repo.** Streamlit Cloud clones your repo — if the documents aren't committed, they won't exist on the server.

### 2. Deploy

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Click **"New app"** → select your repo, branch `main`, main file **`app.py`**
3. Click **"Advanced settings"** → **"Secrets"** and paste:

```toml
ANTHROPIC_API_KEY = "sk-ant-your-key-here"
```

4. Click **Deploy**

First boot takes 3–5 minutes (downloads the embedding model + indexes documents). After that, reboots are fast thanks to caching.

---

## Adding New Documents

1. Add any `.docx` SOP or `.xlsx` FAQ file to the `documents/` folder
2. Commit and push to GitHub
3. In Streamlit Cloud, click the **⋮ menu** → **Reboot app**

The app will re-index automatically on reboot.

---

## Local Development (Optional)

```bash
pip install -r requirements.txt

# Create secrets file
mkdir -p .streamlit
echo 'ANTHROPIC_API_KEY = "sk-ant-your-key-here"' > .streamlit/secrets.toml

# Run
streamlit run app.py
```

---

## Troubleshooting

| Issue | Fix |
|---|---|
| **"API key not found"** | Add `ANTHROPIC_API_KEY` in Streamlit Cloud → Settings → Secrets |
| **"No documents indexed"** | Make sure `.docx`/`.xlsx` files are **committed** in `documents/` (not gitignored) |
| **App crashes on boot** | Check Streamlit Cloud logs (Manage app → Logs). Common: dependency install failure |
| **Slow first load** | Normal — the embedding model (~90 MB) downloads once, then is cached |
| **Out of memory** | Free tier has 1 GB RAM. If you have many large SOPs, consider the paid tier |

---

## Security

- **No API keys in code.** All secrets use `st.secrets` (Streamlit's built-in secret manager).
- `.streamlit/secrets.toml` is in `.gitignore` and never committed.
- The repo can be public — only the Streamlit Cloud secrets dashboard holds sensitive values.

---

*Internal use — MBP*
