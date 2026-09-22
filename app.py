"""
Smart Document Assistant
-------------------------
Upload PDF/TXT documents, ask questions, get grounded answers with sources.
RAG pipeline: extract -> chunk -> embed (sentence-transformers) -> FAISS store
-> retrieve -> LLM (Groq, OpenAI-compatible) -> grounded answer with citations.

Run:
    streamlit run app.py
"""

import os
import io
import time
import numpy as np
import streamlit as st
import faiss
from sentence_transformers import SentenceTransformer
import pdfplumber
import requests

# ----------------------------- Config -----------------------------------

CHUNK_SIZE = 800          # characters per chunk
CHUNK_OVERLAP = 150       # overlap between chunks
TOP_K = 4                 # number of chunks retrieved per question
SIMILARITY_THRESHOLD = 0.30  # below this, we say "I don't know" (cosine sim, 0-1)
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
GROQ_MODEL = "openai/gpt-oss-20b"
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

# --------------------------- Cached resources -----------------------------

@st.cache_resource
def load_embedder():
    return SentenceTransformer(EMBED_MODEL_NAME)


# ------------------------------ Helpers -----------------------------------

def extract_text(file) -> str:
    """Extract raw text from an uploaded PDF or TXT file."""
    name = file.name.lower()
    if name.endswith(".pdf"):
        text_parts = []
        with pdfplumber.open(io.BytesIO(file.read())) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                page_text = page.extract_text() or ""
                if page_text.strip():
                    text_parts.append(f"[Page {page_num}]\n{page_text}")
        return "\n\n".join(text_parts)
    elif name.endswith(".txt"):
        return file.read().decode("utf-8", errors="ignore")
    else:
        return ""


def chunk_text(text: str, source: str):
    """Split text into overlapping chunks, keeping track of source doc."""
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + CHUNK_SIZE, n)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append({"text": chunk, "source": source})
        if end == n:
            break
        start = end - CHUNK_OVERLAP
    return chunks


def build_index(all_chunks, embedder):
    """Embed all chunks and build a FAISS cosine-similarity index."""
    texts = [c["text"] for c in all_chunks]
    embeddings = embedder.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    embeddings = np.array(embeddings, dtype="float32")
    index = faiss.IndexFlatIP(embeddings.shape[1])  # inner product on normalized vecs = cosine sim
    index.add(embeddings)
    return index


def retrieve(question, embedder, index, all_chunks, top_k=TOP_K):
    q_vec = embedder.encode([question], normalize_embeddings=True)
    q_vec = np.array(q_vec, dtype="float32")
    scores, idxs = index.search(q_vec, top_k)
    results = []
    for score, idx in zip(scores[0], idxs[0]):
        if idx == -1:
            continue
        results.append({**all_chunks[idx], "score": float(score)})
    return results


def call_llm(question, context_chunks, chat_history, api_key):
    """Call Groq's OpenAI-compatible chat completion endpoint."""
    context_text = "\n\n---\n\n".join(
        f"Source: {c['source']}\n{c['text']}" for c in context_chunks
    )

    history_text = ""
    if chat_history:
        recent = chat_history[-3:]  # last 3 turns for follow-up context
        history_text = "\n".join(f"Q: {h['q']}\nA: {h['a']}" for h in recent)

    system_prompt = (
        "You are a document assistant. Answer ONLY using the provided context "
        "from the user's uploaded documents. Do not use outside knowledge. "
        "If the context does not contain the answer, respond exactly with: "
        "\"I couldn't find information about that in the uploaded documents.\" "
        "Keep answers concise and mention which source(s) you used."
    )

    user_prompt = f"""Conversation so far (for follow-up context):
{history_text}

Context from documents:
{context_text}

Question: {question}

Answer using only the context above."""

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.2,
        "max_tokens": 500,
    }
    resp = requests.post(GROQ_URL, headers=headers, json=payload, timeout=30)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


# ------------------------------- UI ----------------------------------------

st.set_page_config(page_title="Smart Document Assistant", layout="wide")
st.title("📄 Smart Document Assistant")
st.caption("Upload documents, ask questions, get grounded answers with sources.")

if "chunks" not in st.session_state:
    st.session_state.chunks = []
if "index" not in st.session_state:
    st.session_state.index = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "doc_names" not in st.session_state:
    st.session_state.doc_names = []

with st.sidebar:
    st.header("Setup")
    api_key = st.text_input("Groq API key", type="password", value=os.environ.get("GROQ_API_KEY", ""))
    st.caption("Free key: console.groq.com/keys")
    st.divider()
    st.header("1. Upload documents")
    uploaded_files = st.file_uploader(
        "PDF or TXT files", type=["pdf", "txt"], accept_multiple_files=True
    )
    if st.button("Process documents", disabled=not uploaded_files):
        embedder = load_embedder()
        all_chunks = []
        names = []
        with st.spinner("Extracting and indexing..."):
            for f in uploaded_files:
                text = extract_text(f)
                if text.strip():
                    all_chunks.extend(chunk_text(text, f.name))
                    names.append(f.name)
            if all_chunks:
                st.session_state.index = build_index(all_chunks, embedder)
                st.session_state.chunks = all_chunks
                st.session_state.doc_names = names
                st.session_state.chat_history = []
                st.success(f"Indexed {len(all_chunks)} chunks from {len(names)} document(s).")
            else:
                st.error("No extractable text found in the uploaded files.")

    if st.session_state.doc_names:
        st.divider()
        st.subheader("Uploaded documents")
        for n in st.session_state.doc_names:
            st.write(f"• {n}")

st.header("2. Ask a question")

if not st.session_state.index:
    st.info("Upload and process at least one document to get started.")
else:
    question = st.text_input("Your question")
    ask = st.button("Ask", type="primary", disabled=not question)

    if ask and question:
        if not api_key:
            st.error("Add your Groq API key in the sidebar first.")
        else:
            embedder = load_embedder()
            with st.spinner("Retrieving relevant passages..."):
                results = retrieve(question, embedder, st.session_state.index, st.session_state.chunks)

            top_score = results[0]["score"] if results else 0.0

            if top_score < SIMILARITY_THRESHOLD:
                answer = "I couldn't find information about that in the uploaded documents."
                results = []
            else:
                with st.spinner("Generating answer..."):
                    try:
                        answer = call_llm(question, results, st.session_state.chat_history, api_key)
                    except Exception as e:
                        answer = f"Error calling the LLM: {e}"

            st.session_state.chat_history.append({"q": question, "a": answer})

            st.markdown("### Answer")
            st.write(answer)

            # Confidence indicator (creative feature)
            confidence_pct = int(top_score * 100)
            if confidence_pct >= 60:
                st.success(f"Confidence: {confidence_pct}% (strong match)")
            elif confidence_pct >= 30:
                st.warning(f"Confidence: {confidence_pct}% (partial match)")
            else:
                st.error(f"Confidence: {confidence_pct}% (weak/no match)")

            if results:
                st.markdown("### Sources")
                for r in results:
                    with st.expander(f"{r['source']}  (similarity: {r['score']:.2f})"):
                        st.write(r["text"])

    if st.session_state.chat_history:
        st.divider()
        st.subheader("Conversation history")
        for turn in reversed(st.session_state.chat_history[:-1] if ask else st.session_state.chat_history):
            st.markdown(f"**Q:** {turn['q']}")
            st.markdown(f"**A:** {turn['a']}")
            st.markdown("---")
