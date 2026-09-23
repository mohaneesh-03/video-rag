from PIL.Image import enum
from groq.resources.audio import transcriptions
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

url = os.getenv("QDRANT_URL")
api_key = os.getenv("QDRANT_API_KEY")

COLLECTION_NAME = "video_rag"
EMBEDDING_SIZE = 384

client = None

def get_client() -> QdrantClient:
    global client
    if client is None:
        client = QdrantClient(url=url, api_key=api_key, check_compatibility=False)
    return client

def create_collection():
    client = get_client()
    if client.collection_exists(COLLECTION_NAME):
        print(f"Collection '{COLLECTION_NAME}' already exists. Deleting it...")
        client.delete_collection(COLLECTION_NAME)

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size = EMBEDDING_SIZE,
            distance=Distance.COSINE
        )
    )

    print(f"Collection created - {COLLECTION_NAME}")
    return client

def embed(transcript: str):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = 500,
        chunk_overlap = 50
    )
    chunks = splitter.split_text(transcript)

    print(f"Split text into {len(chunks)} chunks")

    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(chunks, show_progress_bar=True)

    return embeddings,chunks


def upload_vectors(transcript: str):
    client = create_collection()
    embeddings,chunks = embed(transcript)

    points = []

    for i, embedding in enumerate(embeddings):
        point = PointStruct(
            id= i+1,
            vector = embedding,
            payload = {
                "text": chunks[i]
            }
        )
        points.append(point)

    client.upsert(collection_name=COLLECTION_NAME, points=points)

    print(f"Uploaded {len(points)} vectors to {COLLECTION_NAME}")


def search(query: str, top_k: int = 3):
    client = get_client()
    try:
        if not client.collection_exists(COLLECTION_NAME):
            print(f"Warning: Collection '{COLLECTION_NAME}' does not exist yet. Please index transcript first.")
            return []
    except Exception as e:
        print(f"Warning: Could not connect to Qdrant collection: {e}")
        return []

    model = SentenceTransformer("all-MiniLM-L6-v2")
    query_vector = model.encode(query).tolist()

    try:
        result = client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            limit=top_k,
            with_payload=True,
        ).points
        return result
    except Exception as e:
        print(f"Vector search failed: {e}")
        return []