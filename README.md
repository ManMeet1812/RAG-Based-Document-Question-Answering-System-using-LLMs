# RAG-Based Document Question Answering System

A Retrieval-Augmented Generation (RAG) based document question-answering system that allows users to upload PDF documents, extract and process document text, store semantic embeddings in a vector database, and ask natural language questions. The system retrieves the most relevant document chunks and generates accurate, context-aware answers using a Large Language Model.

---

## Project Overview

This project is designed to solve the problem of searching and understanding long documents efficiently. Instead of manually reading large PDF files, users can upload documents and ask questions directly. The system uses a RAG pipeline to retrieve relevant information from the document and generate answers grounded in the uploaded content.

The project demonstrates practical skills in natural language processing, vector databases, semantic search, document processing, and LLM-based application development.

---

## Features

- Upload PDF documents
- Extract text from uploaded PDFs
- Split long documents into manageable text chunks
- Generate embeddings for document chunks
- Store embeddings in a vector database
- Retrieve relevant chunks based on user questions
- Generate answers using an LLM
- Display source chunks used for answer generation
- Simple and interactive Streamlit user interface

---

## Tech Stack

| Component | Technology |
|---|---|
| Programming Language | Python |
| User Interface | Streamlit |
| PDF Processing | PyMuPDF |
| Text Chunking | LangChain |
| Embeddings | OpenAI Embeddings |
| Vector Database | ChromaDB |
| LLM | OpenAI GPT Model |
| Environment Variables | python-dotenv |

---

## System Architecture

```text
User Uploads PDF
        |
        v
PDF Text Extraction
        |
        v
Text Chunking
        |
        v
Embedding Generation
        |
        v
Vector Database Storage
        |
        v
User Asks Question
        |
        v
Relevant Chunk Retrieval
        |
        v
LLM Answer Generation
        |
        v
Answer + Source Chunks Displayed
