import os
import re
import uuid
from pathlib import Path
from io import BytesIO
from typing import List, Dict, Any, Tuple

import streamlit as st
from PyPDF2 import PdfReader
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer


BASE_DIR = Path(__file__).resolve().parent
PDF_DIR = BASE_DIR / "data" / "pdfs"
DB_DIR = BASE_DIR / "data" / "chroma"
PDF_DIR.mkdir(parents=True, exist_ok=True)
DB_DIR.mkdir(parents=True, exist_ok=True)

CHUNK_SIZE = 700
CHUNK_OVERLAP = 120
TOP_K = 4


@st.cache_resource

def load_embedder() -> SentenceTransformer:
    return SentenceTransformer("all-MiniLM-L6-v2")


@st.cache_resource

def get_collection() -> chromadb.Collection:
    client = chromadb.PersistentClient(path=str(DB_DIR))
    collection = client.get_or_create_collection(
        name="pdf_rag_collection",
        metadata={"hnsw:space": "cosine"},
    )
    return collection


st.set_page_config(page_title="Chat with PDF", page_icon="📄", layout="wide")

st.title("📚 Chat with PDF (RAG)")
st.caption("Upload PDFs and ask grounded questions from the uploaded corpus.")


if "messages" not in st.session_state:
    st.session_state.messages = []

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())


with st.sidebar:
    st.header("Upload PDFs")
    pdf_files = st.file_uploader(
        "Choose one or more PDF files",
        type=["pdf"],
        accept_multiple_files=True,
    )

    st.markdown("---")
    st.subheader("RAG Settings")
    top_k = st.slider("Top documents to retrieve", min_value=1, max_value=8, value=TOP_K)
    chunk_size = st.number_input("Chunk size", min_value=150, max_value=1200, value=CHUNK_SIZE)

    if st.button("Clear chat history", use_container_width=True):
        st.session_state.messages = []



def clean_text(text: str) -> str:
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()


def split_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    clean = clean_text(text)
    if len(clean) <= chunk_size:
        return [clean]

    chunks = []
    start = 0
    while start < len(clean):
        end = min(start + chunk_size, len(clean))
        chunk = clean[start:end]
        if len(chunk) == 0:
            break
        chunks.append(chunk)
        start = end - overlap
        if start <= 0:
            start = end
    return chunks


def extract_pdf_text(file) -> Tuple[str, Dict[str, Any]]:
    pdf_bytes = file.getvalue()
    reader = PdfReader(BytesIO(pdf_bytes))
    pages = []
    for page_num, page in enumerate(reader.pages, start=1):
        txt = page.extract_text() or ""
        pages.append({"page": page_num, "text": txt})

    combined_text = "\n\n".join(f"Page {p['page']}\n{p['text']}" for p in pages)
    meta = {
        "filename": file.name,
        "num_pages": len(reader.pages),
        "file_size": len(pdf_bytes),
    }
    return combined_text, meta


def add_docs_to_vector_db(file, collection) -> None:
    text, meta = extract_pdf_text(file)
    file_key = file.name.replace(" ", "_")
    storage_path = PDF_DIR / file_key

    # Save uploaded file locally for reproducibility.
    with open(storage_path, "wb") as f:
        f.write(file.getvalue())

    chunks = split_text(text, chunk_size=chunk_size)
    source_assignments = []

    page_num = 1
    for idx, chunk in enumerate(chunks):
        # assign a page based on the approximate location of the chunk; simple page count anchor
        # we store the filename and page number in metadata, even if page numbering is approximate
        chunk_id = f"{file_key}-{idx}-{uuid.uuid4().hex[:8]}"
        page_info = page_num
        source_assignments.append({
            "filename": file.name,
            "page": page_info,
            "chunk_id": chunk_id,
            "source": f"{file.name} (page {page_info})",
        })

        # create a more accurate page estimate by counting page markers in the chunk
        # page start is approximate as a fallback when pages are not explicitly marked in the raw text
        # actual page numbering is included in the raw page text marker from the PDF extraction
        if "Page " in chunk:
            page_inst = re.findall(r"Page\s+(\d+)", chunk)
            if page_inst:
                page_num = int(page_inst[-1])

        collection.add(
            embeddings=[load_embedder().encode(chunk).tolist()],
            documents=[chunk],
            metadatas=[
                {
                    "filename": file.name,
                    "page": page_info,
                    "source": f"{file.name} (page {page_info})",
                }
            ],
            ids=[chunk_id],
        )


def ingest_uploaded_pdfs(uploaded_files, collection) -> None:
    if not uploaded_files:
        return

    for file in uploaded_files:
        if not file.name.lower().endswith(".pdf"):
            continue
        add_docs_to_vector_db(file, collection)


collection = get_collection()

if pdf_files:
    ingest_uploaded_pdfs(pdf_files, collection)



def retrieve(question: str, collection, top_k: int) -> List[Dict[str, Any]]:
    embedding = load_embedder().encode(question).tolist()
    results = collection.query(
        query_embeddings=[embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    docs = []
    for doc, meta, distance in zip(results["documents"][0], results["metadatas"][0], results["distances"][0]):
        docs.append({
            "text": doc,
            "metadata": meta,
            "distance": distance,
        })
    return docs


def answer_from_sources(question: str, sources: List[Dict[str, Any]]) -> str:
    if not sources:
        return "I could not find relevant content in the uploaded PDFs. Please upload PDF files and ask another question."

    scored = []
    lower_q = question.lower()
    question_terms = set(re.findall(r"[a-zA-Z0-9]+", lower_q))

    for source in sources:
        text = source["text"]
        words = re.findall(r"[a-zA-Z0-9]+", text.lower())
        overlap = sum(1 for term in question_terms if term in text.lower())
        # include token overlap and page/source scoring
        scored.append((overlap, text, source))

    scored.sort(key=lambda x: x[0], reverse=True)
    selected_texts = []
    source_labels = []
    for _, text, source in scored[:min(3, len(scored))]:
        sentence_candidates = re.split(r"(?<=[.!?])\s+", text)
        for sentence in sentence_candidates:
            if any(term in sentence.lower() for term in question_terms) or len(sentence_candidates) == 1:
                selected_texts.append(sentence)
                break

        metadata = source.get("metadata", {})
        file_name = metadata.get("filename", "Uploaded PDF")
        page = metadata.get("page", 1)
        source_labels.append(f"{file_name} (page {page})")

    if not selected_texts:
        # Fall back to a clean top chunk summary.
        selected_texts = [scored[0][1][:500]]

    answer = "Based on the uploaded PDFs, the most relevant content suggests:\n\n"
    answer += " ".join(selected_texts[:2])
    answer += "\n\nSources: " + ", ".join(source_labels)
    return answer


# display current collection summary
with st.expander("Stored PDF retrieval index"):
    collection = get_collection()
    n = collection.count()
    st.write(f"Indexed chunks: {n}")


for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

question = st.chat_input("Ask a question about the uploaded PDF...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    sources = retrieve(question, collection, top_k)
    response = answer_from_sources(question, sources)

    # show sources in a separate panel for better citation visibility.
    source_panel = []
    for source in sources:
        meta = source.get("metadata", {})
        file_name = meta.get("filename", "Uploaded PDF")
        page = meta.get("page", 1)
        source_panel.append(f"{file_name} — page {page}")

    with st.chat_message("assistant"):
        st.markdown(response)
        if source_panel:
            st.markdown("**Sources:** " + " | ".join(source_panel))

    st.session_state.messages.append({"role": "assistant", "content": response})
