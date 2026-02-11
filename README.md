# HW-Audio-Embeddings-API

This guide documents how to run the Audio Embeddings API service locally using Docker and how to execute the integrated test suite.

The service uses a .env file for configuration

## Setup

The application is containerized using **Docker Compose** to orchestrate the *FastAPI backend* and the *Qdrant vector database*.

Navigate to the project root and run:

```bash
docker-compose up --build
```

(optional) Monitor the vector database at http://localhost:6333/dashboard.

## Tests

An integration test suite is provided in ``src/integration_test`` folder. It automates the verification of file uploads, listing, searching, and deletion.

Run tests by running the ``test_api.py``
PS! Make sure the backend is running via docker for testing.

## API routes

``/embeddings`` ``GET``   Lists all stored file metadata.
/embeddings POST Uploads multiple audio files and generates YAMNet embeddings.

``/search`` ``POST`` Searches for similar audio using an uploaded file and top_k.

``/embeddings/{id}`` ``DELETE`` Deletes a specific embedding by its ID.

``/embeddings`` ``DELETE`` Wipes the entire database collection.

## Extra

If you made this far, then don't forget to test out the **interface** branch ;)