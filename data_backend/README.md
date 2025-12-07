# 🏗️ RAG Database Builder

A dedicated utility for ingesting documents, experimenting with chunking strategies, and building high-performance knowledge bases (RAG, LightRAG, KAG) for the RAG Architect system.

## 🚀 Quick Start

### 1. Prerequisites

- **Docker** installed.
- **Local LLM** running (LM Studio, Ollama) on port `1234` (required for Embeddings & Auto-Tagging).
- **Source Documents** (PDFs, Text, Markdown) ready to upload.

### 2. Build Image

```bash
docker build -t rag-builder .
```

### 3. Run Container

You must mount a local folder for **Data** (source files) and **Databases** (generated indexes).

**Linux:**

```bash
docker run -d \
  --name rag-builder \
  --network host \
  -v $(pwd)/data:/project/data \
  -v $(pwd)/databases:/project/databases \
  rag-builder
```

**Windows / Mac:**
_(We map port `8502` here so you can run the Frontend on `8501` simultaneously)_

```powershell
docker run -d `
  --name rag-builder `
  -p 8502:8501 `
  -e LOCAL_LLM_API_URL="http://host.docker.internal:1234/v1" `
  -v "${PWD}/data:/project/data" `
  -v "${PWD}/databases:/project/databases" `
  rag-builder
```

Open your browser at **`http://localhost:8502`**.

---

## 📂 Volume Mounts

| Container Path       | Description                                                                                                  | Access     |
| :------------------- | :----------------------------------------------------------------------------------------------------------- | :--------- |
| `/project/data`      | **Source Files.** Drop your PDFs/MD files here manually, or upload via the UI.                               | Read-Write |
| `/project/databases` | **Output.** The builder generates ChromaDB and Graph files here. **Mount this same folder to the Frontend.** | Read-Write |

## ⚙️ Configuration

| Env Variable           | Default                    | Description                                              |
| :--------------------- | :------------------------- | :------------------------------------------------------- |
| `LOCAL_LLM_API_URL`    | `http://localhost:1234/v1` | Endpoint for Embeddings and metadata generation.         |
| `EMBEDDING_MODEL_NAME` | _Auto-detected_            | Name of the embedding model loaded in your LLM provider. |

## 🛠️ Workflow

1.  **Import Tab:** Drag & drop documents. They are saved to the `/project/data` volume.
2.  **Chunking Lab:** Pick a file and test different chunk sizes/overlaps to see how the text gets split before you build.
3.  **Build Database:**
    - Select Strategy: **Standard RAG**, **LightRAG**, or **KAG**.
    - Name your DB (e.g., `finance_docs`).
    - Click **Launch Population**.
4.  **Connect Frontend:** Once finished, point your **RAG Architect Frontend** to the same `databases` folder to chat with your data.

## ⚠️ Notes for Docker Users

- **LLM Connection:** Ensure your Local LLM (LM Studio/Ollama) is listening on `0.0.0.0` (not just localhost) so the Docker container can reach it.
- **Performance:** Building large databases (especially KAG/Graph) is resource-intensive. Ensure Docker has access to sufficient RAM/CPU.
