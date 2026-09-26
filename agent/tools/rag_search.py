import ollama as ollama_module
from qdrant_client import QdrantClient

EMBED_MODEL = "nomic-embed-text"

def embed_text(text: str, ollama_client=None) -> list[float]:
    client = ollama_client or ollama_module
    response = client.embeddings(model=EMBED_MODEL, prompt=text)
    return response["embedding"]

def search(
    query:str,
    collection_name: str = "notes",
    limit: int = 3,
    ollama_client=None,
    qdrant_client=None
) -> str:
    vector = embed_text(query, ollama_client=ollama_client)
    qclient = qdrant_client or QdrantClient(host="localhost", port=6333)
    try:
        hits = qclient.search(collection_name=collection_name, query_vector=vector, limit=limit)
    except:
        return "error: vector store unavailable"
    
    if not hits:
        return "no relevant notes found"

    return "\n---\n".join(hit.payload["text"] for hit in hits)
