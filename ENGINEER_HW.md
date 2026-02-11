# Audio Embeddings API (Engineer HW)

## Overview

Implement an API service that generates and searches audio embeddings using a pre-trained **YAMNet** model.  

---

## Tech Stack

- Python 3.12
- FastAPI
- Database (choice of implementation)
- Docker & Docker Compose

---

## API Endpoints

### /embeddings

Accepts multiple audio files and generates embeddings using YAMNet.

**Behavior:**
- Computes embeddings for each file
- Stores embeddings with filenames in a database
- Returns a JSON object keyed by filename

---

### /search

Searches the database for similar embeddings.

**Parameters:**
- Audio file
- `top_k` (integer) - maximum number of top similarity embeddings to return

**Behavior:**
- Computes an embedding for the input file
- Searches stored embeddings
- Returns filenames and similarity scores (sorted decendingly by similarity score)

---

## Submission

* Provide an accessible repository (github is preferred).

* Make it runnable locally with minimal setup.

* Document how to run the service in the README.

## Notes

* You are free to make design decisions regarding repo structure, API structure, HTTP method choices, validation, model loading, testing, etc.

* We are interested in seeing what you prioritize and how you structure your solution.

* If you have any questions do not hesitate to reach out.