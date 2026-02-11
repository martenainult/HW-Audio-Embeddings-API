import os
import uuid
import hashlib
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.concurrency import run_in_threadpool
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, PointIdsList
from .audio_processor import compute_embedding
from fastapi.middleware.cors import CORSMiddleware

# --- Configuration from Environment Variables ---
QDRANT_HOST = os.getenv("QDRANT_HOST")
QDRANT_PORT = int(os.getenv("QDRANT_PORT"))
COLLECTION_NAME = os.getenv("COLLECTION_NAME")
VECTOR_SIZE = int(os.getenv("VECTOR_SIZE"))

# Database Setup
qdrant = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        existing_collections = qdrant.get_collections().collections
        exists = any(c.name == COLLECTION_NAME for c in existing_collections)
        if not exists:
            qdrant.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
            )
    except Exception as e:
        print(f"Database warning: {e}")
    yield

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[os.getenv("FRONTEND_URL", "http://localhost:5173")],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_file_id_by_name(filename: str) -> str:
    """Generates a deterministic UUID based ONLY on the filename."""
    # uuid5 creates a consistent UUID for the same string input
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, filename))

# @app.get("/healthz")
# async def health_check():
#     try:
#         # Try a lightweight operation on Qdrant
#         qdrant.get_collections()
#         return {"status": "ok", "database": "connected"}
#     except Exception as e:
#         # Returning a non-200 status code tells Docker the service is NOT healthy
#         raise HTTPException(status_code=503, detail="Database connection failed")

@app.get("/embeddings")
async def list_embeddings():
    """Retrieves all stored file metadata from the collection."""
    try:
        # 'scroll' allows us to iterate through the collection
        # We set a limit of 100 for safety, but you can adjust this
        points, _ = qdrant.scroll(
            collection_name=COLLECTION_NAME,
            limit=100,
            with_payload=True,
            with_vectors=False
        )
        
        return [
            {"id": p.id, "filename": p.payload.get("filename", "Unknown")} 
            for p in points
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list records: {str(e)}")

@app.post("/embeddings")
async def create_embeddings(files: list[UploadFile] = File(...)):
    results = {}
    
    for file in files:
        # 1. Generate ID from filename only
        point_id = get_file_id_by_name(file.filename)
        
        # 2. Check if this filename ID already exists in Qdrant
        existing_points = qdrant.retrieve(
            collection_name=COLLECTION_NAME,
            ids=[point_id]
        )
        
        if existing_points:
            results[file.filename] = {
                "status": "skipped", 
                "message": f"File with name '{file.filename}' already exists in database",
                "id": point_id
            }
            continue

        try:
            content = await file.read()
            # 3. Process Audio
            vector = await run_in_threadpool(compute_embedding, content, file.filename)
            
            # 4. Store in DB
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
            results[file.filename] = {"status": "success", "id": point_id}
            
        except Exception as e:
            results[file.filename] = {"status": "error", "message": str(e)}
            
    return results

@app.post("/search")
async def search_audio(file: UploadFile = File(...), top_k: int = Form(5)):
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    # This offloads the heavy YAMNet math to a background thread
    query_vector = await run_in_threadpool(compute_embedding, content, file.filename)
    
    search_result = qdrant.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vector,
        limit=top_k
    )
    
    return [
        {
            "filename": hit.payload["filename"],
            "similarity_score": hit.score, # Qdrant score
            "id": hit.id
        }
        for hit in search_result
    ]
    
@app.post("/search-by-id/{file_id}")
async def search_by_id(file_id: str, top_k: int = 5):
    """Requirement: Search stored embeddings for similar files using an existing ID."""
    try:
        # 1. Retrieve the vector for the existing file from Qdrant
        points = qdrant.retrieve(
            collection_name=COLLECTION_NAME,
            ids=[file_id],
            with_vectors=True
        )
        
        if not points:
            raise HTTPException(status_code=404, detail="Original file vector not found")
            
        query_vector = points[0].vector
        
        # 2. Perform the similarity search
        search_result = qdrant.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vector,
            limit=top_k + 1  # Get one extra to account for the file itself
        )
        
        # 3. Filter out the original file from the results and return
        return [
            {
                "filename": hit.payload["filename"],
                "similarity_score": hit.score,
                "id": hit.id
            }
            for hit in search_result if hit.id != file_id
        ][:top_k]

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/embeddings/{file_id}")
async def delete_embedding(file_id: str):
    """Deletes a specific audio embedding by its deterministic ID."""
    try:
        qdrant.delete(
            collection_name=COLLECTION_NAME,
            points_selector=PointIdsList(points=[file_id])
        )
        return {"status": "success", "message": f"Deleted {file_id}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/embeddings")
async def delete_all_embeddings():
    """Wipes the collection and recreates it."""
    try:
        qdrant.delete_collection(collection_name=COLLECTION_NAME)
        qdrant.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )
        return {"status": "success", "message": "Database cleared"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))