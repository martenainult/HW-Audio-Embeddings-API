import uuid
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.concurrency import run_in_threadpool
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from .audio_processor import compute_embedding

# --- Configuration ---
QDRANT_HOST = "qdrant"
QDRANT_PORT = 6333
COLLECTION_NAME = "audio_embeddings"
VECTOR_SIZE = 1024

# --- Database Setup ---
qdrant = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Ensure the vector collection exists
    # Fix: We use get_collections() which works on all client versions
    try:
        existing_collections = qdrant.get_collections().collections
        exists = any(c.name == COLLECTION_NAME for c in existing_collections)
        
        if not exists:
            print(f"Creating collection: {COLLECTION_NAME}")
            qdrant.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
            )
        else:
            print(f"Collection {COLLECTION_NAME} already exists.")
            
    except Exception as e:
        print(f"Database initialization warning: {e}")
        
    yield

app = FastAPI(lifespan=lifespan)

@app.post("/embeddings")
async def create_embeddings(files: list[UploadFile] = File(...)):
    results = {}
    
    for file in files:
        # 1. Read file (IO bound - keep async)
        content = await file.read()
        
        try:
            # 2. Process Audio (CPU bound - move to threadpool!)
            # This prevents the server from freezing while calculating vectors
            vector = await run_in_threadpool(compute_embedding, content, file.filename)
            
            # 3. Store in DB
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
        # Calculate vector in a thread
        query_vector = await run_in_threadpool(compute_embedding, content, file.filename)
        
        # Search DB
        search_result = qdrant.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vector,
            limit=top_k
        )
        
        # Format results
        return [
            {
                "filename": hit.payload["filename"],
                "similarity_score": hit.score
            }
            for hit in search_result
        ]

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))