# CyberlawGPT ⚖️

CyberlawGPT is a RAG (Retrieval-Augmented Generation) assistant for querying Pakistani Cyber Laws and Electronic Transaction legislation.

## Quick Setup Instructions

### Streamlit Community Cloud
1. Push repository files (`app.py`, `requirements.txt`, `readme.md`) to GitHub.
2. In Streamlit Cloud, click **Manage App** -> **Re-deploy** to ensure new requirements are installed cleanly.
3. Open **App Settings** -> **Secrets** and add:
   ```toml
   GROQ_API_KEY = "gsk_your_groq_api_key_here"
