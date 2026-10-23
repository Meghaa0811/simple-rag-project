
import os
from typing import List
from dotenv import load_dotenv
from pinecone import Pinecone, EmbedQuery

load_dotenv()

pinecone_client = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pinecone_client.Index(os.getenv("PINECONE_INDEX_NAME"))

NAMESPACE = "rag-documents"


def store_in_pinecone(
    chunks: List[str],
    embeddings=None,
    namespace: str = NAMESPACE
):
    """Store text in a Pinecone document-schema index."""

    records = []

    for i, chunk in enumerate(chunks):
        records.append({
            "_id": f"chunk_{i}",
            "title": f"Chunk {i}",
            "chunk_text": chunk,
            "chunk_index": i
        })

    for start in range(0, len(records), 50):
        index.documents.upsert(
            namespace=namespace,
            documents=records[start:start + 50]
        )

    print(f"Successfully stored {len(records)} chunks.")






def search_in_pinecone(
    query: str,
    top_k: int = 3,
    namespace: str = NAMESPACE
):
    """Search document chunks using Pinecone integrated embeddings."""

    results = index.documents.search(
        namespace=namespace,
        score_by=[
            EmbedQuery(
                field="chunk_text",
                query=query
            )
        ],
        top_k=top_k,
        include_fields=["chunk_text", "title"]
    )

    matched_chunks = []

    for hit in results.matches:
        chunk = getattr(hit, "chunk_text", "")

        if chunk:
            matched_chunks.append(chunk)

    return matched_chunks
