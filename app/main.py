import os

from fastapi import FastAPI, HTTPException
import ollama

from app.rag import Retriever
from app.schemas import AskRequest, AskResponse, Source


app = FastAPI(
    title="GenAI RAG Document Assistant",
    version="1.0.0",
    description="A small Retrieval-Augmented Generation API.",
)

retriever = Retriever()

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")

ollama_client = ollama.Client(host=OLLAMA_HOST)


def generate_answer(question: str, retrieved_chunks: list[dict]) -> str:
    context = "\n\n".join(
        f"[Source: {item['document']} | chunk {item['chunk_id']}]\n{item['text']}"
        for item in retrieved_chunks
    )

    system_prompt = (
        "You are a precise document assistant. "
        "Answer using only the supplied context. "
        "If the context does not contain enough information, say so. "
        "Do not invent facts."
    )

    user_prompt = f"""Context:
{context}

Question:
{question}

Give a concise answer based only on the context."""

    response = ollama_client.chat(
        model=OLLAMA_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )

    return response["message"]["content"].strip()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    try:
        retrieved = retriever.search(request.question, request.top_k)
        answer = generate_answer(request.question, retrieved)

        sources = [
            Source(document=item["document"], chunk_id=item["chunk_id"])
            for item in retrieved
        ]

        return AskResponse(
            answer=answer,
            sources=sources,
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"RAG pipeline error: {exc}",
        ) from exc