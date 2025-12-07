# 🧠 RAG Architect Frontend

A streamlined, Dockerized interface for interacting with large-scale local knowledge bases. Features intelligent query refinement, persistent chat sessions, and multi-strategy retrieval (RAG, LightRAG, KAG).

## 🚀 Quick Start

### 1. Prerequisites

- **Docker** installed.
- **Local LLM** running (LM Studio, Ollama, etc.) on port `1234`.
- **Nvidia Container Toolkit** (Optional: Only if you want the internal Reranker to use GPU on Linux/WSL).

### 2. Build Image

```bash
docker build -t rag-frontend .
```

### 3. Run Container

#### 🐧 Linux / WSL2 (With GPU Support)

_Note: The `--gpus all` flag allows the internal Reranker to use your GPU. The main LLM is accessed via network._

```bash
docker run -d \
  --name rag-frontend \
  --network host \
  --gpus all \
  -v /path/to/your/databases:/app/databases \
  -v $(pwd)/chat_history:/app/chat_history \
  rag-frontend
```

#### 🪟 Windows (Powershell) / Mac

_Note: Connecting GPU to Docker on Windows Desktop is often unnecessary for this app as the heavy lifting is done by the external API._

```powershell
docker run -d `
  --name rag-frontend `
  -p 8501:8501 `
  --add-host=host.docker.internal:host-gateway `
  -e LOCAL_LLM_API_URL="http://host.docker.internal:1234/v1" `
  -v "C:\path\to\databases:/app/databases" `
  -v "${PWD}/chat_history:/app/chat_history" `
  rag-frontend
```

Open your browser at **`http://localhost:8501`**.

---

## 📂 Volume Mounts

| Container Path      | Description                                                        | Access           |
| :------------------ | :----------------------------------------------------------------- | :--------------- |
| `/app/databases`    | **Required.** Folder containing your pre-built RAG/KAG databases.  | Read-Only (Rec.) |
| `/app/chat_history` | **Required.** Stores JSON session files so chats survive restarts. | Read-Write       |

## ⚙️ Configuration

| Env Variable        | Default                    | Description                       |
| :------------------ | :------------------------- | :-------------------------------- |
| `LOCAL_LLM_API_URL` | `http://localhost:1234/v1` | URL for your local LLM inference. |

```

### Summary of "Connect GPU"

Since this is a **RAG Frontend**, the heaviest computational work (The LLM Generation) happens **outside** the container (in LM Studio/Ollama).

*   **You usually do NOT need `--gpus all`** for this specific container.
*   The only component inside this container that uses AI is the **Reranker** (CrossEncoder) and **KeyBERT**.
*   The model used (`ms-marco-MiniLM-L-6-v2`) is extremely small. It runs in ~50ms on a standard CPU.
*   **Recommendation:** Stick to the standard CPU-based `docker run` command. It is more stable and portable. Only use `--gpus all` if you notice the "Retrieving..." step taking 5+ seconds.
```
