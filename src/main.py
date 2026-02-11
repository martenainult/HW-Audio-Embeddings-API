import uuid
import hashlib
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.concurrency import run_in_threadpool
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, PointIdsList
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
    # Robust startup check (works on all client versions)
    try:
        existing_collections = qdrant.get_collections().collections
        exists = any(c.name == COLLECTION_NAME for c in existing_collections)
        
        if not exists:
            print(f"Creating collection: {COLLECTION_NAME}")
            qdrant.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
            )
    except Exception as e:
        print(f"Database warning: {e}")
    yield

app = FastAPI(lifespan=lifespan)

def get_file_id(content: bytes) -> str:
    """Generates a deterministic UUID based on file content hash."""
    file_hash = hashlib.sha256(content).hexdigest()
    # Create a UUID based on the hash (Namespace DNS is just a standard placeholder)
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, file_hash))


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
        content = await file.read()
        
        # 1. Generate ID from content
        # If the file is identical, this ID will be identical.
        point_id = get_file_id(content)
        
        # 2. Check for Duplicates (Fast Check)
        # We ask Qdrant if this ID exists.
        existing_points = qdrant.retrieve(
            collection_name=COLLECTION_NAME,
            ids=[point_id]
        )
        
        if existing_points:
            # SKIP PROCESSING
            print(f"Skipping {file.filename} (Duplicate)")
            results[file.filename] = {
                "status": "skipped", 
                "message": "Duplicate file detected",
                "id": point_id
            }
            continue

        try:
            # 3. Process Audio (Heavy Lifting)
            print(f"Processing new file: {file.filename}")
            vector = await run_in_threadpool(compute_embedding, content, file.filename)
            
            # 4. Store in DB
            qdrant.upsert(
                collection_name=COLLECTION_NAME,
                points=[
                    PointStruct(
                        id=point_id,  # Use the hash-based ID
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
async def search_audio(
    file: UploadFile = File(...), 
    top_k: int = Form(5)
):
    content = await file.read()
    
    try:
        query_vector = await run_in_threadpool(compute_embedding, content, file.filename)
        
        search_result = qdrant.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vector,
            limit=top_k
        )
        
        return [
            {
                "filename": hit.payload["filename"],
                "similarity_score": hit.score,
                "id": hit.id
            }
            for hit in search_result
        ]

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

@app.delete("/embeddings/{file_id}")
async def delete_embedding(file_id: str):
    """Deletes a specific audio embedding by its ID."""
    try:
        # We use the point ID (the hash-based UUID) to remove the record
        result = qdrant.delete(
            collection_name=COLLECTION_NAME,
            points_selector=PointIdsList(
                points=[file_id]
            )
        )
        return {"status": "success", "message": f"Deleted ID {file_id}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Delete failed: {str(e)}")

@app.delete("/embeddings")
async def delete_all_embeddings():
    """Wipes the entire collection (Dangerous/Internal use)."""
    try:
        # Re-creating the collection is the fastest way to "delete all" in Qdrant
        qdrant.delete_collection(collection_name=COLLECTION_NAME)
        qdrant.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )
        return {"status": "success", "message": "All embeddings deleted and collection reset"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Wipe failed: {str(e)}")