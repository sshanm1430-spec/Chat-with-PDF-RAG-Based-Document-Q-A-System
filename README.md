# Chat with PDF (RAG)

A lightweight Streamlit application for uploading PDF documents, extracting text, indexing chunks into a local Chroma vector database, and answering natural-language questions with source citations.

## Features

- PDF upload and text extraction
- Local vector database using Chroma
- Retrieval-augmented answers with citations
- Conversation history in the Streamlit session
- Minimal dependency stack using Python, Streamlit, PyPDF2, Chroma, and SentenceTransformers

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app stores uploaded PDFs and the vector database under the workspace `data/` directory.
