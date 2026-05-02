# RAG-Based Document Question Answering System with FAISS, OpenAI, and Docker

## Project Overview

This project is a Retrieval-Augmented Generation (RAG)-based Document Question Answering system that allows users to upload a PDF document, ask questions about the document, retrieve relevant source chunks, and generate grounded answers using an LLM.

The system extracts text from uploaded PDFs, splits the text into chunks, generates embeddings using SentenceTransformers, stores and searches those embeddings using FAISS, and sends the retrieved context to OpenAI to generate an answer with page-level source references.

This project demonstrates an end-to-end AI application pipeline combining document processing, semantic search, retrieval augmentation, LLM-based answer generation, and Docker-based deployment.

---

## Features

- Upload PDF documents through a Streamlit interface
- Extract text page-by-page using PyMuPDF
- Split extracted text into overlapping chunks
- Generate semantic embeddings using SentenceTransformers
- Store and search embeddings using FAISS vector search
- Retrieve top relevant document chunks for user questions
- Improve retrieval using:
  - FAISS semantic search
  - keyword scoring
  - section-aware helper for questions like “What is the executive summary?”
- Generate grounded answers using OpenAI API
- Display source file name and page number for retrieved chunks
- Containerized with Docker for reproducible setup

---

## Tech Stack

| Component | Technology |
|---|---|
| Frontend | Streamlit |
| PDF Processing | PyMuPDF |
| Embeddings | SentenceTransformers |
| Vector Search | FAISS |
| LLM | OpenAI API |
| Environment Variables | python-dotenv |
| Containerization | Docker |
| Programming Language | Python |

---

## RAG Pipeline

The system follows this pipeline:

```text
PDF Upload
    ↓
Text Extraction
    ↓
Text Cleaning
    ↓
Chunking with Page Metadata
    ↓
Embedding Generation
    ↓
FAISS Vector Indexing
    ↓
Question Embedding
    ↓
Semantic Retrieval
    ↓
Hybrid Reranking
    ↓
OpenAI Answer Generation
    ↓
Answer with Source References
