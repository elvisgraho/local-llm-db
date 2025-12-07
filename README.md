# 🧬 Local LLM RAG System

A Dockerized system for building Knowledge Bases and Chatting with them locally.

## 📂 Project Structure

This setup uses Docker Volumes (created automatically in the `volumes/` folder) to persist data:

- **`volumes/raw_files`**: Place your PDFs here. The **Builder** sees them.
- **`volumes/databases`**: Generated indexes live here. Shared between Builder and Frontend.
- **`volumes/chat_history`**: Chat logs live here.

## 🚀 Windows Setup Guide

### 1. Prerequisites

1.  **Docker Desktop** installed and running.
2.  **LM Studio** (or Ollama) running.
    - **Crucial:** Ensure the Local LLM Server is running on port `1234`.
    - **LM Studio Users:** Go to the Server tab (double arrow icon) -> Enable "Cross-Origin-Resource-Sharing (CORS)" -> Start Server.

### 2. Start the System

Open PowerShell (or Command Prompt) in this folder:

```powershell
docker-compose up --build -d
```

### 3. Workflow

#### Step A: Build a Database

1.  Open **[http://localhost:8502](http://localhost:8502)** (The Builder).
2.  Go to the **Import Files** tab.
3.  Upload your PDFs (or manually drop them into `local-llm-db/volumes/raw_files`).
4.  Go to **Build Database**, choose a name (e.g., "manuals"), and click **Launch**.

#### Step B: Chat

1.  Open **[http://localhost:8501](http://localhost:8501)** (The Frontend).
2.  In the sidebar settings:
    - **LLM Provider:** Local.
    - **API URL:** Leave as default (`http://host.docker.internal:1234/v1`).
3.  Select your database ("manuals") from the dropdown.
4.  Start chatting.

## 🛠️ Troubleshooting

**"Connection Error" to LLM:**
Inside Docker, `localhost` refers to the container, not your Windows machine.

- We use `host.docker.internal` to reach your Windows host.
- Ensure your firewall allows Docker to communicate with LM Studio.

**Rebuilding:**
If you change code, run:

```powershell
docker-compose up --build -d
```
