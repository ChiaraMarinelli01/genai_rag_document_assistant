# GenAI RAG Document Assistant

A lightweight GenAI engineering portfolio project demonstrating a complete **Retrieval-Augmented Generation (RAG)** pipeline for question answering over a local document collection.

The application retrieves relevant document chunks using semantic search and uses a locally hosted **Llama 3.2** model through **Ollama** to generate grounded answers with source references.

## Features

* Python application architecture
* REST API with FastAPI
* Document ingestion from local text files
* Semantic paragraph-based chunking
* Text embeddings with Sentence Transformers
* Vector similarity search with FAISS
* Local LLM inference with Ollama / Llama 3.2
* Retrieval-grounded answer generation
* Source references in API responses
* Automated tests with pytest
* Docker containerization
* Docker-to-host Ollama integration

## Architecture

```text
Documents
    ↓
Paragraph-based Chunking
    ↓
Sentence Transformer Embeddings
    ↓
FAISS Vector Index
    ↓
Semantic Retrieval
    ↓
Retrieved Context
    ↓
Ollama / Llama 3.2
    ↓
Grounded Answer + Sources
```

The retrieval layer and the generation layer are intentionally separated:

* **FAISS** is responsible for semantic retrieval.
* **Ollama / Llama 3.2** is responsible for generating the final answer from the retrieved context.
* The prompt instructs the model to avoid unsupported claims and answer only from the provided context.

## Project Structure

```text
genai_rag_document_assistant/
├── app/
│   ├── main.py
│   ├── rag.py
│   ├── schemas.py
│   └── __init__.py
├── data/
│   ├── documents/
│   │   ├── analytics.txt
│   │   ├── data_quality.txt
│   │   ├── databricks_architecture.txt
│   │   ├── pipeline_overview.txt
│   │   └── testing_and_dashboard.txt
│   └── test_documents/
│       └── sample.txt
├── tests/
│   └── test_rag.py
├── .env.example
├── .gitignore
├── Dockerfile
├── pytest.ini
├── README.md
└── requirements.txt
```

## Technology Stack

| Component        | Technology            |
| ---------------- | --------------------- |
| Language         | Python                |
| API              | FastAPI               |
| Embeddings       | Sentence Transformers |
| Vector Search    | FAISS                 |
| LLM              | Llama 3.2             |
| LLM Runtime      | Ollama                |
| Validation       | Pydantic              |
| Testing          | pytest                |
| Containerization | Docker                |

## Local Setup

### 1. Create the virtual environment

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Install and start Ollama

Make sure the `llama3.2` model is available locally:

```bash
ollama run llama3.2
```

The default configuration is:

```text
OLLAMA_MODEL=llama3.2
OLLAMA_HOST=http://localhost:11434
```

These values can be configured through `.env`.

### 4. Start the API

```bash
uvicorn app.main:app --reload
```

The API will be available at:

```text
http://127.0.0.1:8000
```

Interactive API documentation:

```text
http://127.0.0.1:8000/docs
```

## API

### Health Check

```http
GET /health
```

Example response:

```json
{
  "status": "ok"
}
```

### Ask a Question

```http
POST /ask
```

Example request:

```json
{
  "question": "What does the Bronze layer represent?",
  "top_k": 3
}
```

Example response:

```json
{
  "answer": "The Bronze layer represents the raw input data and preserves the source records, providing the initial ingestion layer for the pipeline.",
  "sources": [
    {
      "document": "databricks_architecture.txt",
      "chunk_id": 2
    }
  ]
}
```

The `sources` field identifies the document chunks used as retrieval context for the generated answer.

## Retrieval and Chunking

Documents are split using **paragraph-based semantic chunking** rather than arbitrary fixed-size text fragments.

This approach preserves related information within the same chunk and improves retrieval quality for questions about specific concepts such as:

* Bronze, Silver, and Gold layers
* Data quality validation
* Analytical datasets
* Testing and dashboards
* ETL pipeline architecture

Embeddings are generated with:

```text
all-MiniLM-L6-v2
```

The resulting vectors are indexed using FAISS with inner-product similarity on normalized embeddings.

## Testing

Run the automated test suite:

```bash
pytest -v
```

The tests cover:

* Chunk generation
* Invalid chunk configuration
* Semantic retrieval of relevant documents

Current test suite:

```text
3 passed
```

## Docker

Build the image:

```bash
docker build -t genai-rag-document-assistant .
```

Run the container:

```powershell
docker run --rm -p 8000:8000 `
  -e OLLAMA_HOST=http://host.docker.internal:11434 `
  -e OLLAMA_MODEL=llama3.2 `
  genai-rag-document-assistant
```

On Windows, `host.docker.internal` allows the containerized API to communicate with the Ollama service running on the host machine.

The API can then be accessed at:

```text
http://127.0.0.1:8000
```

## Engineering Validation

The project was validated end-to-end with:

* Local FastAPI execution
* Local Ollama / Llama 3.2 inference
* FAISS semantic retrieval
* Bronze, Silver, and Gold retrieval tests
* Grounded answer generation
* Docker container execution
* Docker-to-host Ollama communication
* Automated pytest tests

For example, a Bronze-layer question retrieves the relevant chunk from `databricks_architecture.txt` as the top semantic result and generates an answer grounded in that context.

## Design Considerations

The project intentionally keeps the architecture lightweight and local:

* No external vector database is required.
* Embeddings are generated locally.
* LLM inference is performed locally through Ollama.
* FAISS provides the vector index.
* FastAPI exposes the RAG pipeline through a REST interface.

This makes the project easy to run locally while demonstrating the core engineering components of a RAG system.

## Future Extensions

Possible improvements include:

* Retrieval evaluation datasets
* Reranking
* LLM-as-a-Judge evaluation
* Observability and tracing
* Conversation memory
* Authentication
* External vector databases
* Cloud deployment
* Agent and tool integration

## Portfolio Context

This project complements a separate **Operating Room Data Pipeline & Analytics** project focused on data engineering, ETL, PostgreSQL, PySpark, Databricks, Docker, testing, and analytics.

Together, the projects demonstrate experience across both **Data Engineering and GenAI Engineering**.

