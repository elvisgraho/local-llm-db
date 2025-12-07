FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    RAG_DATABASE_DIR=/app/databases \
    CHAT_HISTORY_DIR=/app/chat_history \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    ANONYMIZED_TELEMETRY=False

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download model
RUN python3 -c "from sentence_transformers import CrossEncoder; CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"

# Copy app code
COPY . .

# Explicitly create mount points (good practice)
RUN mkdir -p /app/databases && \
    mkdir -p /app/chat_history

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]