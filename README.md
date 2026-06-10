# 📚 Research Paper RAG Assistant

A Retrieval-Augmented Generation (RAG) system for research paper analysis built using Streamlit, FAISS, Ollama, and Qwen3.

## Features

- Upload PDF research papers
- Automatic text extraction
- Semantic chunking
- Vector embeddings using nomic-embed-text
- FAISS vector database
- Context-aware retrieval
- Question answering with Qwen3
- Fully local pipeline (no API keys required)

## Architecture

PDF
↓
Chunking
↓
nomic-embed-text Embeddings
↓
FAISS Vector Store
↓
Top-k Retrieval
↓
Qwen3 (Ollama)
↓
Answer Generation

## Tech Stack

- Python
- Streamlit
- Ollama
- Qwen3
- FAISS
- nomic-embed-text
- NumPy
- PyPDF

## Installation

```bash
pip install streamlit
pip install ollama
pip install faiss-cpu
pip install pypdf
pip install numpy
```

Pull models:

```bash
ollama pull qwen3:4b
ollama pull nomic-embed-text
```

Run:

```bash
streamlit run app.py
```

## Applications

- Research Assistance
- Literature Review
- Academic Study
- Semantic Search
- Document Understanding