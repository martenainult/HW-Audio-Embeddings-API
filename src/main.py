from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from contextlib import asynccontextmanager
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import uuid
from .audio_processor import compute_embedding

# Configuration
QDRANT_HOST = "qdrant"
QDRANT_PORT = 6333
COLLECTION_NAME = "audio_embeddings"
VECTOR_SIZE = 1024  # YAMNet output size

# Database Client
qdrant = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Ensure the vector collection exists
    if not qdrant.collection_exists(COLLECTION_NAME):
        qdrant.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )
    yield

app = FastAPI(lifespan=lifespan)

@app.post("/embeddings")
async def create_embeddings(files: list[UploadFile] = File(...)):
    results = {}
    
    for file in files:
        content = await file.read()
        try:
            # 1. Generate Vector
            vector = compute_embedding(content, file.filename)
            
            # 2. Store in Qdrant
            # We use the filename as the payload, and a UUID as the point ID
            point_id = str(uuid.uuid4())
            qdrant.upsert(
                collection_name=COLLECTION_NAME,
                points=[
                    PointStruct(
                        id=point_id,
                        vector=vector,
                        payload={"filename": file.filename}
                    )
                ]
            )
            results[file.filename] = "Stored successfully"
        except Exception as e:
            results[file.filename] = f"Error: {str(e)}"
            
    return results

@app.post("/search")
async def search_audio(
    file: UploadFile = File(...), 
    top_k: int = Form(5)
):
    content = await file.read()
    
    try:
        # 1. Generate Vector for the query audio
        query_vector = compute_embedding(content, file.filename)
        
        # 2. Search Qdrant
        search_result = qdrant.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vector,
            limit=top_k
        )
        
        # 3. Format Response
        response = []
        for hit in search_result:
            response.append({
                "filename": hit.payload["filename"],
                "similarity_score": hit.score  # Cosine similarity (0 to 1)
            })
            
        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
