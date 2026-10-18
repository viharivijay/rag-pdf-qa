# RAG PDF Q&A Assistant

A Retrieval-Augmented Generation (RAG) application that lets you upload a PDF and ask natural-language questions about it. Answers are grounded in the document's actual content — not the LLM's memory — and include page-number citations, reducing hallucination.

## Why RAG?

Large language models are powerful but can confidently generate incorrect information when asked about content they weren't trained on (like your own documents). RAG solves this by:

1. Breaking the document into chunks
2. Embedding those chunks into vectors and storing them for similarity search
3. Retrieving only the most relevant chunks for a given question
4. Feeding those chunks to the LLM as context, so it answers from the actual source material

## Architecture

```
 PDF Upload
     |
     v
 Text Extraction (pdfplumber)
     |
     v
 Chunking (overlapping word windows, tracked per page)
     |
     v
 Embedding (sentence-transformers: all-MiniLM-L6-v2)
     |
     v
 Vector Index (FAISS, in-memory)
     |
     v
 User Question --> Embed Question --> Similarity Search (top-k chunks)
     |
     v
 Grounded Prompt (question + retrieved chunks)
     |
     v
 LLM (OpenAI gpt-4o-mini) --> Answer + Source Citations
```

## Tech Stack

| Layer          | Tool                              |
|----------------|------------------------------------|
| Backend        | FastAPI                            |
| Frontend       | Streamlit                          |
| Embeddings     | sentence-transformers (MiniLM)     |
| Vector Search  | FAISS                              |
| LLM            | OpenAI API (gpt-4o-mini)           |
| PDF Parsing    | pdfplumber                         |

## Project Structure

```
rag-pdf-qa/
├── app/
│   ├── main.py        # FastAPI app: /upload and /ask endpoints
│   ├── ingest.py       # PDF text extraction + chunking + embedding
│   ├── retriever.py    # FAISS vector store + similarity search
│   ├── prompt.py        # Grounded prompt template construction
│   └── llm.py            # LLM API call wrapper
├── ui/
│   └── streamlit_app.py  # Chat-style frontend
├── .env.example
├── requirements.txt
└── README.md
```

## Setup

1. Clone the repo and create a virtual environment:
   ```bash
   git clone https://github.com/<your-username>/rag-pdf-qa.git
   cd rag-pdf-qa
   python -m venv venv
   source venv/bin/activate   # Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up your API key:
   ```bash
   cp .env.example .env
   # then edit .env and add your OPENAI_API_KEY
   ```

4. Start the backend:
   ```bash
   uvicorn app.main:app --reload
   ```

5. In a separate terminal, start the frontend:
   ```bash
   streamlit run ui/streamlit_app.py
   ```

6. Open the Streamlit URL shown in your terminal, upload a PDF, and start asking questions.

## Example

> **Uploaded:** a 12-page research paper on transformer architectures
> **Question:** "What dataset was used for evaluation?"
> **Answer:** "The paper evaluates on the WMT 2014 English-German dataset (Page 7)."

## Design Decisions

- **Chunk size (400 words, 50-word overlap):** balances context completeness against retrieval precision. Overlap prevents losing meaning at chunk boundaries.
- **MiniLM embeddings:** small, fast, and free to run locally — no embedding API costs, which matters for a portfolio project people will actually run.
- **FAISS over a hosted vector DB:** keeps the project dependency-free and runnable offline/locally; swapping in Pinecone or Weaviate later is a small change (see Future Work).
- **In-memory single-document store:** kept intentionally simple for a demo. A production version would key the index by session or user.

## Limitations & Future Work

- Currently supports one PDF at a time (no multi-document search)
- No OCR support — scanned/image-only PDFs won't extract text
- Vector index is in-memory and resets on restart (no persistence)
- Could add: streaming LLM responses, multi-document support with metadata filtering, a hosted vector DB for production scale, and an automated retrieval-quality eval set

## License

MIT
