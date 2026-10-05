from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


BASE_DIR = Path(__file__).resolve().parent.parent
DOCUMENTS_DIR = BASE_DIR / "data" / "documents"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

CHUNK_SIZE = 80
CHUNK_OVERLAP = 20


def load_documents() -> list[dict]:
    documents = []

    for path in sorted(DOCUMENTS_DIR.glob("*.txt")):
        documents.append({
            "document": path.name,
            "text": path.read_text(encoding="utf-8"),
        })

    return documents


def chunk_text(
    text: str,
    chunk_size: int = 500,
    overlap: int = 100,
) -> list[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError(
            "overlap must be >= 0 and smaller than chunk_size"
        )

    paragraphs = [
        paragraph.strip()
        for paragraph in text.split("\n\n")
        if paragraph.strip()
    ]

    chunks = []

    for paragraph in paragraphs:
        words = paragraph.split()

        if len(words) <= chunk_size:
            chunks.append(paragraph)
            continue

        step = chunk_size - overlap

        for start in range(0, len(words), step):
            chunk = " ".join(words[start:start + chunk_size])

            if chunk:
                chunks.append(chunk)

    return chunks


class Retriever:
    def __init__(self):
        self.model = SentenceTransformer(EMBEDDING_MODEL)
        self.index, self.chunks = self._build_index()

    def _build_index(self):
        chunks = []

        for document in load_documents():
            document_chunks = chunk_text(
                document["text"],
                chunk_size=CHUNK_SIZE,
                overlap=CHUNK_OVERLAP,
            )

            for chunk_id, text in enumerate(document_chunks):
                chunks.append({
                    "document": document["document"],
                    "chunk_id": chunk_id,
                    "text": text,
                })

        if not chunks:
            raise RuntimeError(
                "No documents found in data/documents"
            )

        embeddings = self.model.encode(
            [item["text"] for item in chunks],
            normalize_embeddings=True,
        )

        vectors = np.asarray(
            embeddings,
            dtype="float32",
        )

        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)

        return index, chunks

    def search(
        self,
        question: str,
        top_k: int = 3,
    ) -> list[dict]:
        if not question.strip():
            raise ValueError("question cannot be empty")

        embedding = self.model.encode(
            [question],
            normalize_embeddings=True,
        )

        vector = np.asarray(
            embedding,
            dtype="float32",
        )

        scores, indices = self.index.search(
            vector,
            min(top_k, len(self.chunks)),
        )

        results = []

        for score, index in zip(scores[0], indices[0]):
            if index < 0:
                continue

            result = self.chunks[index].copy()
            result["score"] = float(score)
            results.append(result)

        return results