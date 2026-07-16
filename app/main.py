"""
main.py
FastAPI backend exposing two endpoints:
  POST /upload  -> upload a PDF, chunk + embed it, build the vector index
  POST /ask     -> ask a question against the currently indexed PDF
"""

import os
import shutil
import tempfile

from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel

from app.ingest import chunk_pdf
from app.retriever import VectorStore
from app.prompt import build_prompt
from app.llm import call_llm

app = FastAPI(title="RAG PDF Q&A Assistant")

# NOTE: a single global store is fine for a demo/portfolio project.
# For multi-user production use, key this by session/user id instead.
store = VectorStore()


class AskRequest(BaseModel):
    query: str
    top_k: int = 3


@app.get("/")
def health_check():
    return {"status": "ok", "message": "RAG PDF Q&A Assistant is running"}


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        chunks, metadata = chunk_pdf(tmp_path)
        if not chunks:
            raise HTTPException(
                status_code=400,
                detail="Could not extract any text from this PDF. It may be scanned/image-only.",
            )
        store.build(chunks, metadata)
    finally:
        os.remove(tmp_path)

    return {
        "filename": file.filename,
        "chunks_indexed": len(chunks),
        "message": "Document indexed successfully. You can now ask questions.",
    }


@app.post("/ask")
def ask_question(request: AskRequest):
    if not store.is_ready:
        raise HTTPException(
            status_code=400,
            detail="No document has been indexed yet. Upload a PDF first via /upload.",
        )

    retrieved = store.search(request.query, k=request.top_k)
    prompt = build_prompt(request.query, retrieved)
    answer = call_llm(prompt)

    sources = [
        {"page": meta.get("page"), "excerpt": chunk[:200]}
        for chunk, meta, _ in retrieved
    ]

    return {"answer": answer, "sources": sources}
