from app.rag import chunk_text


def test_chunk_text_creates_multiple_chunks():
    text = " ".join(f"word{i}" for i in range(20))
    chunks = chunk_text(text, chunk_size=8, overlap=2)

    assert len(chunks) > 1
    assert all(chunk for chunk in chunks)


def test_chunk_text_rejects_invalid_overlap():
    text = "some example text"

    try:
        chunk_text(text, chunk_size=10, overlap=10)
        assert False, "Expected ValueError"
    except ValueError:
        assert True

def test_retrieval_returns_relevant_document():
    from app.rag import Retriever

    retriever = Retriever()

    results = retriever.search(
        "What data quality checks are performed in the Silver layer?",
        top_k=3,
    )

    assert results
    assert any(result["document"] == "data_quality.txt" for result in results[:3])