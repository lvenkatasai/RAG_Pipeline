# RAG Pipeline

A complete Retrieval-Augmented Generation pipeline for PDF document Q&A.

## What It Does
Reads PDFs → chunks text → generates embeddings → stores in FAISS → answers questions via Streamlit UI.

## Technologies Used
- Python 3.14.7
- streamlit (UI)
- pymupdf (PDF reading)
- sentence-transformers (embeddings)
- faiss-cpu (vector search)
- numpy

## Project Structure
```
RAG-Pipeline/
├── app.py              # Main Streamlit app
├── src/
│   ├── pdf_reader.py   # PDF extraction
│   ├── chunker.py      # Text chunking
│   └── embedder.py     # Embedding generation
├── data/               # 7 PDF files
├── venv/               # Python 3.14.7 environment
└── PROJECT_DOCUMENTATION.txt
```

## How to Run
```powershell
.\venv\Scripts\Activate.ps1
$env:STREAMLIT_BROWSER_GATHER_USAGE_STATS="false"
streamlit run app.py --server.headless true
```

## Pipeline Flow
PDF → Chunk → Embed (384-dim) → FAISS Index → Retrieve → Answer

## Test Results
- 20 pages loaded
- 48 chunks created
- Embeddings: (48, 384)
