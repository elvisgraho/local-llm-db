# RAG Database Builder

A utility for ingesting documents and generating RAG knowledge bases.

## Prerequisites

- Docker installed.
- Local LLM running on port `1234` (required for Embeddings).

## Running via Docker Compose (Recommended)

From the project root:

```bash
docker-compose up -d builder
```

The application will be available at `http://localhost:8502`.

## Manual Docker Run

If running manually, map the raw data and output database folders.

1. **Build:**

   ```bash
   docker build -t rag-builder .
   ```

2. **Run:**
   (Execute from project root)

   ```bash
   docker run -d \
     --name rag-builder \
     -p 8502:8501 \
     --add-host=host.docker.internal:host-gateway \
     -e LOCAL_LLM_API_URL="http://host.docker.internal:1234/v1" \
     -v "$(pwd)/volumes/raw_files:/project/data" \
     -v "$(pwd)/volumes/databases:/project/databases" \
     rag-builder
   ```

## Configuration

| Environment Variable | Default                    | Description                             |
| :------------------- | :------------------------- | :-------------------------------------- |
| `LOCAL_LLM_API_URL`  | `http://localhost:1234/v1` | Endpoint for embeddings generation.     |
| `RAW_FILES_DIR`      | `/project/raw_files`       | Internal path for source PDF/TXT files. |
| `RAG_DATABASE_DIR`   | `/project/databases`       | Internal path for output indexes.       |
