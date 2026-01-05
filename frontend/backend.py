import logging
import uvicorn
import os
from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

# --- Internal Logic Imports ---
# These must exist in your project structure
from query.query_data import query_direct, query_lightrag
from query.database_paths import list_available_dbs, db_exists, DATABASE_DIR
from query.templates import REFINE_QUERY_PROMPT

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("RAG-Architect-Backend")

app = FastAPI(
    title="LightRAG Architect Engine",
    description="Backend API for RAG operations, contextual rewriting, and LLM interaction.",
    version="1.0.0"
)

# --- Pydantic Models (The "Contract") ---

class LLMConfig(BaseModel):
    provider: str = Field("local", description="Provider: 'local' or 'openai'")
    modelName: Optional[str] = Field(None, description="Model identifier (e.g. 'llama3')")
    api_url: Optional[str] = Field(None, description="Endpoint for the LLM API")
    apiKey: Optional[str] = Field("EMPTY", description="API Key if applicable")
    temperature: float = Field(0.7, ge=0.0, le=1.0)
    system_prompt: Optional[str] = Field(None, description="Current system instructions")
    context_window: int = Field(8182, description="Total token limit for the model")

class RAGConfig(BaseModel):
    rag_type: str = Field("direct", description="Mode: 'direct', 'rag', or 'lightrag'")
    db_name: Optional[str] = Field(None, description="Target database folder name")
    top_k: int = Field(5, description="Number of context chunks to retrieve")
    hybrid: bool = Field(False, description="Enable BM25 + Vector hybrid search")
    verify: bool = Field(False, description="Enable the two-pass LLM verification logic")
    history_limit: int = Field(5, description="Number of past messages to include")

class RewriteRequest(BaseModel):
    query: str
    history: List[Dict[str, str]]
    llm_config: LLMConfig

class QueryRequest(BaseModel):
    query_payload: str
    llm_config: LLMConfig
    rag_config: RAGConfig
    history: List[Dict[str, str]]

# --- Discovery & Utility Endpoints ---

@app.get("/health")
def health_check():
    """Service status and path verification."""
    return {
        "status": "online",
        "database_dir": str(DATABASE_DIR),
        "db_initialization": "ready" if DATABASE_DIR.exists() else "missing"
    }

@app.get("/databases")
def get_databases():
    """Dynamically lists all available DBs detected on disk."""
    return {
        "rag": list_available_dbs("rag"),
        "lightrag": list_available_dbs("lightrag")
    }

@app.get("/config-schema")
def get_config_schema():
    """Returns a full example of the expected configuration for the frontend."""
    return {
        "llm_config": LLMConfig().model_dump(),
        "rag_config": RAGConfig().model_dump(),
        "sample_history": [{"role": "user", "content": "example"}]
    }

# --- Core Logic Endpoints ---

@app.post("/rewrite")
def rewrite_query(req: RewriteRequest):
    """
    Uses the LLM to rewrite the user query into a standalone search query.
    Uses REFINE_QUERY_PROMPT as the system_prompt.
    """
    try:
        # Prepare the specific config for rewriting
        rewrite_cfg = req.llm_config.model_dump()
        rewrite_cfg["system_prompt"] = REFINE_QUERY_PROMPT
        rewrite_cfg["temperature"] = 0.2  # Stability for keywords

        logger.info(f"Contextualizing query: '{req.query[:50]}...'")

        response = query_direct(
            query_text=req.query,
            llm_config=rewrite_cfg,
            conversation_history=req.history
        )
        
        rewritten = response.get("text", "").strip()
        return {"rewritten_query": rewritten if rewritten else req.query}

    except Exception as e:
        logger.error(f"Rewrite Pipeline Error: {e}", exc_info=True)
        # Fallback to original query so the chat doesn't break
        return {"rewritten_query": req.query, "error": str(e)}

@app.post("/query")
def execute_query(req: QueryRequest):
    """
    The main execution pipeline. Handles Direct, RAG, and LightRAG.
    Respects system_prompts, verify flags, and database existence.
    """
    try:
        # Convert Pydantic to Dicts for internal functions
        llm_dict = req.llm_config.model_dump()
        rag_dict = req.rag_config.model_dump()
        
        rag_type = rag_dict.get("rag_type", "direct")
        db_name = rag_dict.get("db_name")
        verify_enabled = rag_dict.get("verify", False)

        # 1. Database Validation
        if rag_type != "direct":
            if not db_name:
                raise HTTPException(status_code=400, detail="RAG mode requires a db_name.")
            if not db_exists(rag_type, db_name):
                logger.error(f"Database validation failed for {rag_type}/{db_name}")
                raise HTTPException(status_code=404, detail=f"Database '{db_name}' not found for type '{rag_type}'.")

        # 2. Argument Construction
        # We pass the full llm_dict which includes your custom 'system_prompt'
        query_args = {
            "query_text": req.query_payload,
            "llm_config": llm_dict,
            "conversation_history": req.history,
            "verify": verify_enabled
        }

        # 3. Execution Branching
        logger.info(f"Execution: Mode={rag_type} | DB={db_name} | Verify={verify_enabled}")
        
        if rag_type == 'direct':
            response = query_direct(**query_args)
        else:  # LightRAG
            response = query_lightrag(
                **query_args,
                db_name=db_name,
                top_k=rag_dict.get("top_k", 5),
                hybrid=rag_dict.get("hybrid", False)
            )

        if not response:
            raise ValueError("Query function returned an empty response.")

        return response

    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Main Query Pipeline Error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

# --- Server Lifecycle ---

if __name__ == "__main__":
    # Ensure standard directories exist before starting
    if not DATABASE_DIR.exists():
        logger.warning(f"DATABASE_DIR not found at {DATABASE_DIR}. Creating now...")
        DATABASE_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Starting LightRAG Architect Backend on port 8005...")
    uvicorn.run(
        "backend_server:app", 
        host="0.0.0.0", 
        port=8005, 
        reload=True,
        log_level="info"
    )