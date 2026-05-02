import os
import re

import faiss
import fitz  # PyMuPDF
import numpy as np
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer


# -----------------------------
# Load environment variables
# -----------------------------
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None


# -----------------------------
# Streamlit page setup
# -----------------------------
st.set_page_config(page_title="RAG Document QA", page_icon="📄", layout="wide")

st.title("RAG-Based Document Question Answering System")
st.write(
    "Upload a PDF, retrieve accurate source chunks using FAISS semantic search, "
    "and generate grounded answers using OpenAI."
)


# -----------------------------
# Load embedding model
# -----------------------------
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


# -----------------------------
# Extract pages from PDF
# -----------------------------
def extract_pages_from_pdf(file):
    pages = []

    pdf_document = fitz.open(stream=file.read(), filetype="pdf")

    for page_number, page in enumerate(pdf_document, start=1):
        page_text = page.get_text()

        if page_text.strip():
            pages.append(
                {
                    "page": page_number,
                    "text": page_text.strip(),
                }
            )

    return pages


# -----------------------------
# Clean text
# -----------------------------
def clean_text(text):
    text = re.sub(r"\n+", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


# -----------------------------
# Dynamic heading detection
# -----------------------------
def detect_dynamic_headings(text):
    headings = []

    lines = text.split("\n")

    for line in lines:
        clean_line = line.strip()

        if not clean_line:
            continue

        if len(clean_line) > 100:
            continue

        looks_numbered = bool(re.match(r"^\d+(\.\d+)*\.?\s+[A-Z]", clean_line))

        known_heading = clean_line.lower() in {
            "executive summary",
            "introduction",
            "conclusion",
            "references",
            "metrics for success",
            "strategy for change management",
            "organizational context and need for change",
        }

        short_title_like = (
            len(clean_line.split()) <= 8
            and clean_line[0].isupper()
            and not clean_line.endswith(".")
        )

        if looks_numbered or known_heading or short_title_like:
            headings.append(clean_line)

    return headings


# -----------------------------
# Split pages into chunks with metadata
# Uses full-page chunks + sliding chunks
# -----------------------------
def split_pages_into_chunks(pages, source_name, chunk_size=1200, overlap=250):
    chunks = []

    for page in pages:
        page_number = page["page"]
        text = clean_text(page["text"])
        headings = detect_dynamic_headings(text)

        # Full-page chunk helps section-level questions
        if text:
            chunks.append(
                {
                    "text": text,
                    "page": page_number,
                    "source": source_name,
                    "chunk_type": "full_page",
                    "headings": headings,
                }
            )

        # Sliding chunks help detailed semantic retrieval
        start = 0

        while start < len(text):
            end = start + chunk_size
            chunk_text = text[start:end].strip()

            if chunk_text:
                chunks.append(
                    {
                        "text": chunk_text,
                        "page": page_number,
                        "source": source_name,
                        "chunk_type": "semantic_chunk",
                        "headings": headings,
                    }
                )

            start += chunk_size - overlap

    return chunks


# -----------------------------
# Normalize vectors for cosine similarity
# -----------------------------
def normalize_vectors(vectors):
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return vectors / norms


# -----------------------------
# Create FAISS index
# -----------------------------
def create_faiss_index(chunks, embedding_model):
    chunk_texts = [chunk["text"] for chunk in chunks]

    embeddings = embedding_model.encode(chunk_texts)
    embeddings = np.array(embeddings).astype("float32")
    embeddings = normalize_vectors(embeddings)

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    return index, embeddings


# -----------------------------
# Known section-title query detection
# -----------------------------
def detect_known_section_query(question):
    question_lower = question.lower()

    known_sections = [
        "executive summary",
        "introduction",
        "organizational context",
        "need for change",
        "strategy for change management",
        "change blueprint",
        "overcoming opposition",
        "building buy-in",
        "improved interaction",
        "leadership",
        "leadership's role",
        "metrics for success",
        "success metrics",
        "conclusion",
        "references",
    ]

    for section in known_sections:
        if section in question_lower:
            return section

    return None


# -----------------------------
# Keyword overlap score
# -----------------------------
def keyword_overlap_score(question, chunk_text):
    question_words = set(re.findall(r"\b[a-zA-Z]{4,}\b", question.lower()))

    stop_words = {
        "what",
        "where",
        "when",
        "which",
        "about",
        "this",
        "that",
        "does",
        "document",
        "uploaded",
        "tell",
        "give",
        "show",
        "explain",
        "summary",
        "summarize",
        "main",
        "section",
        "answer",
    }

    important_words = question_words - stop_words
    chunk_words = set(re.findall(r"\b[a-zA-Z]{4,}\b", chunk_text.lower()))

    return len(important_words.intersection(chunk_words)) * 0.15


# -----------------------------
# Section helper score
# Searches ALL chunks, not just FAISS top chunks
# -----------------------------
def section_helper_score(question, chunk):
    section = detect_known_section_query(question)

    if not section:
        return 0.0

    section_aliases = {
        "success metrics": "metrics for success",
        "leadership's role": "leadership",
    }

    section_to_find = section_aliases.get(section, section)
    chunk_text_lower = chunk["text"].lower()

    score = 0.0

    if section_to_find in chunk_text_lower:
        score += 2.5

        position = chunk_text_lower.find(section_to_find)

        if position < 200:
            score += 1.0

    for heading in chunk.get("headings", []):
        if section_to_find in heading.lower():
            score += 1.5

    return score


# -----------------------------
# Hybrid retrieval:
# FAISS semantic search + section helper + keyword scoring
# -----------------------------
def retrieve_relevant_chunks(
    question,
    chunks,
    index,
    embedding_model,
    top_k=5,
    use_section_helper=True,
):
    question_embedding = embedding_model.encode([question])
    question_embedding = np.array(question_embedding).astype("float32")
    question_embedding = normalize_vectors(question_embedding)

    search_k = min(30, len(chunks))

    semantic_scores, semantic_indices = index.search(question_embedding, search_k)

    semantic_score_map = {}

    for score, idx in zip(semantic_scores[0], semantic_indices[0]):
        if idx != -1:
            semantic_score_map[idx] = float(score)

    candidates = []

    for idx, chunk in enumerate(chunks):
        semantic_score = semantic_score_map.get(idx, 0.0)
        keyword_score = keyword_overlap_score(question, chunk["text"])

        helper_score = 0.0
        if use_section_helper:
            helper_score = section_helper_score(question, chunk)

        if semantic_score > 0 or keyword_score > 0 or helper_score > 0:
            final_score = semantic_score + keyword_score + helper_score

            candidates.append(
                {
                    "chunk": chunk,
                    "semantic_score": semantic_score,
                    "keyword_score": keyword_score,
                    "helper_score": helper_score,
                    "final_score": final_score,
                    "retrieval_method": (
                        "FAISS semantic search + section helper + keyword scoring"
                        if use_section_helper
                        else "FAISS semantic search + keyword scoring"
                    ),
                }
            )

    candidates = sorted(candidates, key=lambda x: x["final_score"], reverse=True)

    # Remove near duplicate chunks
    results = []
    used_keys = set()

    for candidate in candidates:
        chunk = candidate["chunk"]

        key = (
            chunk["page"],
            chunk["text"][:150].lower(),
        )

        if key in used_keys:
            continue

        used_keys.add(key)
        results.append(candidate)

        if len(results) == top_k:
            break

    return results


# -----------------------------
# Build OpenAI prompt
# -----------------------------
def build_prompt(question, retrieved_results):
    context_parts = []

    for i, result in enumerate(retrieved_results, start=1):
        chunk = result["chunk"]

        context_parts.append(
            f"[Source {i}]\n"
            f"File: {chunk['source']}\n"
            f"Page: {chunk['page']}\n"
            f"Chunk type: {chunk['chunk_type']}\n"
            f"Text:\n{chunk['text']}"
        )

    context = "\n\n---\n\n".join(context_parts)

    prompt = f"""
You are a document question-answering assistant.

Your task:
Answer the user's question using ONLY the provided document context.

Important rules:
1. Do not use outside knowledge.
2. If the answer is not present in the context, say:
   "I could not find this information in the uploaded document."
3. Include source references in the answer using the file name and page number.
4. If the user asks for a section such as "executive summary", summarize that section from the retrieved context.
5. Do not invent missing details.
6. Keep the answer clear, structured, and concise.

User question:
{question}

Document context:
{context}

Final answer:
"""

    return prompt


# -----------------------------
# Generate OpenAI answer
# -----------------------------
def generate_answer(question, retrieved_results):
    if client is None:
        return "OpenAI API key not found. Please add your API key to the .env file."

    prompt = build_prompt(question, retrieved_results)

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
        )

        return response.choices[0].message.content

    except Exception as error:
        return f"OpenAI error: {error}"


# -----------------------------
# Main app
# -----------------------------
embedding_model = load_embedding_model()

uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"])

use_section_helper = st.checkbox(
    "Use section helper for section-title questions",
    value=True,
)

st.caption(
    "FAISS semantic search is the main retrieval method. "
    "The section helper improves questions like 'What is the executive summary?' "
    "by checking section titles across the whole document before sending context to OpenAI."
)

if uploaded_file is not None:
    st.success(f"Uploaded file: {uploaded_file.name}")

    pages = extract_pages_from_pdf(uploaded_file)

    st.subheader("PDF Text Extraction")
    st.info(f"Total pages with text extracted: {len(pages)}")

    with st.expander("View extracted text from first page"):
        if pages:
            st.write(pages[0]["text"][:2500])
        else:
            st.warning("No text was extracted from this PDF.")

    chunks = split_pages_into_chunks(
        pages=pages,
        source_name=uploaded_file.name,
        chunk_size=1200,
        overlap=250,
    )

    st.subheader("Document Chunks")
    st.write(f"Total chunks created: {len(chunks)}")

    with st.expander("View first 5 chunks"):
        for i, chunk in enumerate(chunks[:5], start=1):
            st.markdown(f"### Chunk {i}")
            st.write(
                f"**Source:** {chunk['source']} | "
                f"**Page:** {chunk['page']} | "
                f"**Type:** {chunk['chunk_type']}"
            )

            if chunk["headings"]:
                st.write(f"**Detected headings:** {', '.join(chunk['headings'])}")

            st.write(chunk["text"])

    if len(chunks) > 0:
        st.subheader("Vector Database")

        index, embeddings = create_faiss_index(chunks, embedding_model)

        st.success("Embeddings created and stored in FAISS successfully.")
        st.write(f"Embedding shape: {embeddings.shape}")

        question = st.text_input("Ask a question about the uploaded document")

        if question:
            retrieved_results = retrieve_relevant_chunks(
                question=question,
                chunks=chunks,
                index=index,
                embedding_model=embedding_model,
                top_k=5,
                use_section_helper=use_section_helper,
            )

            st.subheader("Generated Answer")

            with st.spinner("Generating answer using OpenAI..."):
                answer = generate_answer(question, retrieved_results)
                st.write(answer)

            st.subheader("Top Relevant Chunks Retrieved")

            for i, result in enumerate(retrieved_results, start=1):
                chunk = result["chunk"]

                st.markdown(f"## Retrieved Chunk {i}")

                st.write(
                    f"**Source:** {chunk['source']} | "
                    f"**Page:** {chunk['page']} | "
                    f"**Type:** {chunk['chunk_type']}"
                )

                if chunk["headings"]:
                    st.write(f"**Detected headings:** {', '.join(chunk['headings'])}")

                st.write(f"**Retrieval method:** {result['retrieval_method']}")

                st.write(chunk["text"])

                st.caption(
                    f"Semantic score: {result['semantic_score']:.4f} | "
                    f"Helper score: {result['helper_score']:.4f} | "
                    f"Keyword score: {result['keyword_score']:.4f} | "
                    f"Final score: {result['final_score']:.4f}"
                )

                st.divider()