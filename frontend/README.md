# RAG Architect Frontend

A Dockerized interface for interacting with local knowledge bases via RAG, LightRAG, and KAG strategies.

## Prerequisites

- Docker installed.
- Local LLM running (e.g., LM Studio) on port `1234`.

## Running via Docker Compose (Recommended)

From the project root:

```bash
docker-compose up -d frontend
```

The application will be available at `http://localhost:8501`.

## Manual Docker Run

If running the container manually, you must mount the persistent volumes located in the project root.

1. **Build:**

   ```bash
   docker build -t rag-frontend .
   ```

2. **Run:**
   (Execute from project root)

   ```bash
   docker run -d \
     --name rag-frontend \
     -p 8501:8501 \
     --add-host=host.docker.internal:host-gateway \
     -e LOCAL_LLM_API_URL="http://host.docker.internal:1234/v1" \
     -v "$(pwd)/volumes/databases:/app/databases" \
     -v "$(pwd)/volumes/chat_history:/app/chat_history" \
     rag-frontend
   ```

## Configuration

| Environment Variable | Default                    | Description                           |
| :------------------- | :------------------------- | :------------------------------------ |
| `LOCAL_LLM_API_URL`  | `http://localhost:1234/v1` | URL for local LLM inference.          |
| `RAG_DATABASE_DIR`   | `/app/databases`           | Internal container path for indexes.  |
| `CHAT_HISTORY_DIR`   | `/app/chat_history`        | Internal container path for sessions. |
