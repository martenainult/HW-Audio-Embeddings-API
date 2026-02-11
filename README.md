# HW-Audio-Embeddings-API

Welcome to the bonus section. Here we have a nice interface, added tests and separated docker environments for smooth development.

## Backend Setup

* **Navigate to the backend folder.** This service handles audio processing via YAMNet and stores vectors in **Qdrant**.

* **Configure Environment**: Ensure your .env file exists with the correct API_PORT (default 8000).

Run the following command in backend folder

```bash
docker-compose up --build
```

The database can be accessed from ``http://localhost:6333/dashboard#/collections``

## Frontend Setup

* Navigate to the frontend folder
* Configure Environment: Ensure .env has ``VITE_API_URL=http://localhost:8000``.

Run the following command in frontend folder

```bash
docker-compose up --build
```

Access the UI: Open your browser to http://localhost:5173.


## Usage

There are already some test files prepared, but you are welcome to test and break the site :)
