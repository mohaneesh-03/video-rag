import os
from typing import List, Dict, Any, Tuple, Optional
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

url = os.getenv("QDRANT_URL")
api_key = os.getenv("QDRANT_API_KEY")

COLLECTION_NAME = "video_rag"
EMBEDDING_SIZE = 384

client = None
_embedding_model = None


def get_embedding_model() -> SentenceTransformer:
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedding_model


import time


def retry_qdrant(fn, max_retries: int = 4, delay: float = 1.0):
    """
    Executes a Qdrant operation with exponential backoff retry
    to handle transient DNS or socket connection hiccups.
    """
    last_err = None
    for attempt in range(max_retries):
        try:
            return fn()
        except Exception as e:
            last_err = e
            if attempt < max_retries - 1:
                time.sleep(delay * (2 ** attempt))
            else:
                raise last_err


def get_client() -> QdrantClient:
    global client
    if client is None:
        client = QdrantClient(url=url, api_key=api_key, check_compatibility=False, timeout=30.0)
    return client


def create_collection(collection_name: str = COLLECTION_NAME):
    client = get_client()

    def _op():
        if client.collection_exists(collection_name):
            print(f"Collection '{collection_name}' already exists. Recreating it for fresh indexing...")
            client.delete_collection(collection_name)

        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=EMBEDDING_SIZE,
                distance=Distance.COSINE
            )
        )
        print(f"Collection created - {collection_name}")
        return client

    return retry_qdrant(_op)


def embed(transcript: str) -> Tuple[Any, List[str], List[str]]:
    """
    Sentence-Window Chunking:
    - Splits text into focused base chunks (~300 chars) for sharp semantic vector representation.
    - Generates an expanded context window (3-chunk window: prev + current + next)
      to eliminate severed thoughts at chunk boundaries.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=320,
        chunk_overlap=30,
        separators=["\n\n", "\n", ". ", "? ", "! ", " ", ""]
    )
    chunks = splitter.split_text(transcript)

    print(f"Split text into {len(chunks)} base chunks for indexing")

    # Build 3-chunk surrounding context window for each chunk
    contexts = []
    num_chunks = len(chunks)
    for i in range(num_chunks):
        start_idx = max(0, i - 1)
        end_idx = min(num_chunks, i + 2)
        window_text = " ".join(chunks[start_idx:end_idx])
        contexts.append(window_text)

    model = get_embedding_model()
    embeddings = model.encode(chunks, show_progress_bar=False)

    return embeddings, chunks, contexts


def upload_vectors(transcript: str, collection_name: str = COLLECTION_NAME):
    """
    Indexes the complete transcript into Qdrant Cloud with sentence-window payload metadata.
    """
    client = create_collection(collection_name=collection_name)
    embeddings, chunks, contexts = embed(transcript)

    points = []
    total = len(chunks)

    for i, embedding in enumerate(embeddings):
        point = PointStruct(
            id=i + 1,
            vector=embedding.tolist() if hasattr(embedding, "tolist") else embedding,
            payload={
                "text": chunks[i],
                "context": contexts[i],
                "chunk_index": i,
                "total_chunks": total,
            }
        )
        points.append(point)

    def _upsert():
        client.upsert(collection_name=collection_name, points=points)

    retry_qdrant(_upsert)
    print(f"Uploaded {len(points)} sentence-window vectors to {collection_name}")


def search(query: str, top_k: int = 3, collection_name: str = COLLECTION_NAME):
    """
    Standard dense semantic search in Qdrant.
    Returns matched PointStructs with payload containing both 'text' and expanded 'context'.
    """
    client = get_client()

    def _check():
        return client.collection_exists(collection_name)

    try:
        exists = retry_qdrant(_check)
        if not exists:
            print(f"Warning: Collection '{collection_name}' does not exist yet. Please index transcript first.")
            return []
    except Exception as e:
        print(f"Warning: Could not connect to Qdrant collection: {e}")
        return []

    model = get_embedding_model()
    query_vector = model.encode(query).tolist()

    def _query():
        return client.query_points(
            collection_name=collection_name,
            query=query_vector,
            limit=top_k,
            with_payload=True,
        ).points

    try:
        return retry_qdrant(_query)
    except Exception as e:
        print(f"Vector search failed: {e}")
        return []


def targeted_search(
    queries: List[str],
    top_k_per_query: int = 4,
    include_milestones: bool = True,
    collection_name: str = COLLECTION_NAME
) -> List[Dict[str, Any]]:
    """
    Multi-query targeted semantic retrieval for RAG-driven synthesis.
    Deduplicates points across queries and orders them chronologically by chunk_index.
    """
    client = get_client()
    try:
        if not client.collection_exists(collection_name):
            return []
    except Exception:
        return []

    model = get_embedding_model()
    seen_ids = set()
    selected_points = []

    # Include milestone anchor: intro chunk (ID 1)
    if include_milestones:
        try:
            scroll_res, _ = retry_qdrant(
                lambda: client.scroll(
                    collection_name=collection_name,
                    limit=1,
                    with_payload=True,
                )
            )
            for pt in scroll_res:
                if pt.id not in seen_ids:
                    seen_ids.add(pt.id)
                    selected_points.append(pt)
        except Exception:
            pass

    for q in queries:
        query_vector = model.encode(q).tolist()
        try:
            results = retry_qdrant(
                lambda: client.query_points(
                    collection_name=collection_name,
                    query=query_vector,
                    limit=top_k_per_query,
                    with_payload=True,
                ).points
            )
            for pt in results:
                if pt.id not in seen_ids:
                    seen_ids.add(pt.id)
                    selected_points.append(pt)
        except Exception as e:
            print(f"Targeted search query '{q}' error: {e}")

    # Sort chronologically to preserve discourse order
    def get_chunk_idx(p):
        if p.payload and "chunk_index" in p.payload:
            return p.payload["chunk_index"]
        return p.id

    selected_points.sort(key=get_chunk_idx)

    # Format return list
    formatted = []
    for pt in selected_points:
        formatted.append({
            "id": pt.id,
            "chunk_index": pt.payload.get("chunk_index", pt.id) if pt.payload else pt.id,
            "text": pt.payload.get("text", "") if pt.payload else "",
            "context": pt.payload.get("context", pt.payload.get("text", "")) if pt.payload else "",
        })
    return formatted