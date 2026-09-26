from unittest.mock import MagicMock
from agent.tools.rag_search import embed_text, search
from ingest import chunk_text, ingest_folder

def test_chunk_text_splits_on_size_with_overlap():
    text = "word" * 300
    chunks = chunk_text(text, chunk_size=500, overlap=50)
    assert len(chunks) > 1
    assert all(len(c) <= 500 for c in chunks)

def test_chunk_text_short_text_single_chunk():
    assert chunk_text("short text", chunk_size=500, overlap=50) == ["short text"]

def test_embed_text_calls_ollama_embeddings():
    fake_ollama = MagicMock()
    fake_ollama.embeddings.return_value = {"embedding": [0.1, 0.2, 0.3]}
    result = embed_text("hello", ollama_client=fake_ollama)
    assert result == [0.1, 0.2, 0.3]
    fake_ollama.embeddings.assert_called_once_with(model="nomic-embed-text", prompt="hello")

def test_search_formats_results():
    fake_ollama = MagicMock()
    fake_ollama.embeddings.return_value = {"embedding": [0.1, 0.2]}
    fake_qdrant = MagicMock()
    hit = MagicMock(payload={"text": "Qdrant is a vector database."}, score = 0.9)
    fake_qdrant.search.return_value = [hit]

    result = search("what is qdrant", ollama_client=fake_ollama, qdrant_client=fake_qdrant)
    assert "Qdrant is a vector database." in result

def test_search_returns_error_string_on_qdrant_failure():
    fake_ollama = MagicMock()
    fake_ollama.embeddings.return_value = {"embedding": [0.1, 0.2]}
    fake_qdrant = MagicMock()
    fake_qdrant.search.side_effect = ConnectionError("refused")

    result = search("anything", ollama_client=fake_ollama, qdrant_client=fake_qdrant)
    assert result == "error: vector store unavailable"

def test_ingest_folder_upserts_expected_chunk_count(tmp_path):
    notes = tmp_path / "notes"
    notes.mkdir()
    (notes / "a.md").write_text("shbd" * 300)
    (notes / "b.md").write_text("short note")

    fake_ollama = MagicMock()
    fake_ollama.embeddings.return_value = {"embedding": [0.1, 0.2]}
    fake_qdrant = MagicMock()

    count = ingest_folder(str(notes), ollama_client=fake_ollama, qdrant_client=fake_qdrant)

    assert count >= 2
    fake_qdrant.recreate_collection.assert_called_once()
    assert fake_qdrant.upsert.call_count == 1
