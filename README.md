# RAG PDF Q&A Assistant

A Retrieval-Augmented Generation (RAG) application that lets you upload a PDF and ask natural-language questions about it. Answers are grounded in the document's actual content — not the LLM's memory — and include page-number citations, reducing hallucination.

## Why RAG?

Large language models are powerful but can confidently generate incorrect information when asked about content they weren't trained on (like your own documents). RAG solves this by:

1. Breaking the document into chunks
2. Embedding those chunks into vectors and storing them for similarity search
3. Retrieving only the most relevant chunks for a given question
4. Feeding those chunks to the LLM as context, so it answers from the actual source material


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

## Limitations & Future Work

- Currently supports one PDF at a time (no multi-document search)
- No OCR support — scanned/image-only PDFs won't extract text
- Vector index is in-memory and resets on restart (no persistence)
- Could add: streaming LLM responses, multi-document support with metadata filtering, a hosted vector DB for production scale, and an automated retrieval-quality eval set

## License

MIT
