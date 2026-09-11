# CyberlawGPT ⚖️

CyberlawGPT is a Retrieval-Augmented Generation (RAG) Streamlit application designed to answer questions regarding cyber laws, digital rights, and e-commerce legislation in Pakistan.

## Features
* **Document Processing:** Automates embeddings generation on startup using HuggingFace sentence transformers and FAISS vector index.
* **Customizable Responses:** Adjust Technical/Legal Depth and Response Length on the UI.
* **Groq API Integration:** Ultra-fast inferencing powered by Llama 3 / Mixtral models.

## Deployment Instructions

### 1. Running on Streamlit Community Cloud
1. Push `app.py`, `requirements.txt`, and `readme.md` to a public GitHub repository.
2. Sign in to [Streamlit Community Cloud](https://share.streamlit.io/).
3. Connect your repository and select `app.py` as the entry point.
4. Add your `GROQ_API_KEY` in Streamlit Secrets or enter it via the application sidebar UI.

### 2. Running on Google Colab
Run the following commands inside a Colab notebook:

```bash
!pip install -q streamlit langchain langchain-community langchain-groq faiss-cpu pypdf sentence-transformers tiktoken pyngrok

# Write app files
# Run streamlit via localtunnel or ngrok
!npx localtunnel --port 8501 & streamlit run app.py
