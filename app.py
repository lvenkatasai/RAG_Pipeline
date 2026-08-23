#!/usr/bin/env python3
"""
RAG Pipeline - Complete Implementation
"""

import os
import numpy as np
import streamlit as st
from pathlib import Path
from typing import List, Dict, Tuple

from src.pdf_reader import read_pdfs
from src.chunker import create_chunks
from src.embedder import create_embeddings

try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False

DATA_DIR = Path("data")
CHUNK_SIZE = 500
OVERLAP = 100
TOP_K = 3
RELEVANCE_THRESHOLD = 0.3  # Below this score, query is considered unrelated to data


def load_documents() -> List[Dict]:
    if not DATA_DIR.exists():
        return []
    docs = read_pdfs(str(DATA_DIR))
    return docs


def build_faiss_index(embeddings: np.ndarray):
    if not HAS_FAISS:
        raise ImportError("FAISS not installed")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index


def retrieve_chunks(query: str, chunks: List[Dict], embeddings: np.ndarray, index, k: int = TOP_K) -> List[Dict]:
    query_embedding = create_embeddings([{"text": query}])
    distances, indices = index.search(query_embedding.reshape(1, -1), k)
    retrieved = []
    for i, idx in enumerate(indices[0]):
        chunk = chunks[idx].copy()
        chunk["score"] = float(1 / (1 + distances[0][i]))
        retrieved.append(chunk)
    return retrieved


def generate_answer(query: str, retrieved_chunks: List[Dict]) -> str:
    if not retrieved_chunks:
        return "No relevant documents found."
    # Check if the top retrieved chunk is relevant enough
    top_score = retrieved_chunks[0]["score"]
    if top_score < RELEVANCE_THRESHOLD:
        return "⚠️ This question doesn't seem to be related to the provided documents. Please ask a question about the content in the uploaded PDFs (API Reference, Employee Handbook, FAQ Support, Onboarding Guide, Pricing and SLA, Product Manual, or Security Policy)."
    context_parts = []
    for chunk in retrieved_chunks:
        context_parts.append(f"[{chunk['document']} Page {chunk['page']} Score: {chunk['score']:.2f}]\n{chunk['text']}")
    return "Based on retrieved documents:\n\n" + "\n\n".join(context_parts)


def build_rag_pipeline():
    st.set_page_config(page_title="RAG Pipeline", layout="wide")
    st.title("RAG Pipeline - Document Q&A")
    st.markdown("Retrieval-Augmented Generation pipeline")
    st.markdown("Made by L Venkatasai")
    
    with st.spinner("Loading PDFs..."):
        documents = load_documents()
    if not documents:
        st.error("No PDFs found in data/")
        return
    
    st.success(f"Loaded {len(documents)} pages")
    
    with st.spinner("Chunking..."):
        chunks = create_chunks(documents, chunk_size=CHUNK_SIZE, overlap=OVERLAP)
    st.info(f"Chunks: {len(chunks)}")
    
    with st.spinner("Embedding..."):
        embeddings = create_embeddings(chunks)
    st.info(f"Embeddings shape: {embeddings.shape}")
    
    if HAS_FAISS:
        with st.spinner("Building FAISS index..."):
            index = build_faiss_index(embeddings)
        st.success(f"Index: {index.ntotal} vectors")
    else:
        index = None
        st.warning("FAISS not installed")
    
    query = st.text_input("Ask a question:", "What is this document about?")
    if query:
        st.write(f"Query: {query}")
        if index:
            with st.spinner("Retrieving..."):
                retrieved = retrieve_chunks(query, chunks, embeddings, index, TOP_K)
            answer = generate_answer(query, retrieved)
            st.subheader("Answer")
            st.markdown(answer)
            with st.expander("Retrieved chunks"):
                for r in retrieved:
                    st.write(f"**{r['document']}** (Page {r['page']}) Score: {r['score']:.2f}")
                    st.text(r['text'][:300])
        else:
            st.info("Install faiss-cpu for retrieval")


if __name__ == "__main__":
    build_rag_pipeline()
