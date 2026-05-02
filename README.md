# RAG-Based Document Question Answering System using LLMs, FAISS, and Docker

## Project Overview

This project is a Retrieval-Augmented Generation (RAG)-based Document Question Answering system that allows users to upload a PDF document, ask questions about the document, retrieve relevant source chunks, and generate grounded answers using an LLM.

The system extracts text from uploaded PDFs, splits the text into chunks, generates embeddings using SentenceTransformers, stores and searches those embeddings using FAISS, and sends the retrieved context to OpenAI to generate answers with page-level source references.

This project demonstrates an end-to-end AI application pipeline that combines document processing, semantic search, retrieval augmentation, LLM-based answer generation, source grounding, and Docker-based deployment.

---

## Features

- Upload PDF documents through a Streamlit interface
- Extract text page-by-page using PyMuPDF
- Clean and preprocess extracted document text
- Split documents into overlapping chunks with page-level metadata
- Generate semantic embeddings using SentenceTransformers
- Store and search embeddings using FAISS vector search
- Retrieve relevant document chunks for user questions
- Generate grounded answers using OpenAI API
- Display source file name and page number for retrieved chunks
- Improved retrieval using a hybrid approach:
  - FAISS semantic search
  - keyword overlap scoring
  - optional section-aware retrieval for common report-style headings
- Containerized with Docker for reproducible setup and deployment

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
User Question
    ↓
Question Embedding
    ↓
Semantic Retrieval
    ↓
Hybrid Retrieval Improvement
    ↓
OpenAI LLM Answer Generation
    ↓
Answer with Source References
