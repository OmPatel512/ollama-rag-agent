import sys
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance

from agent.tools.rag_search import embed_text

VECTOR_SIZE = 768 

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    if len(text) <= chunk_size:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks

def ingest_folder(
    folder: str,
    collection_name: str = "notes",
    ollama_client = None,
    qdrant_client = None,
) -> int:
    qclient = qdrant_client or QdrantClient(host="localhost", port=6333)
    qclient.recreate_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
    )

    points = []
    point_id = 0
    for path in sorted(Path(folder).glob("*.md")):
        text = path.read_text()
        for chunk in chunk_text(text):
            vector = embed_text(chunk, ollama_client=ollama_client)
            points.append(
                PointStruct(id=point_id, vector=vector, payload={"text": chunk, "source": str(path)})
            )
            point_id += 1

    if points:
        qclient.upsert(collection_name=collection_name, points=points)
    return len(points)

                                                                                                                                                                                  
if __name__ == "__main__":                                                                                                                                                        
    folder = sys.argv[1] if len(sys.argv) > 1 else "notes"                                                                                                                        
    count = ingest_folder(folder)                                                                                                                                                 
    print(f"Ingested {count} chunks from {folder}")    