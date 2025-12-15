import streamlit as st
import time
import logging
import os

# --- Internal Imports ---
import app_utils
from query.query_data import query_direct, query_rag, query_lightrag
from query.session_manager import session_manager

def process_user_input(session_data, config, state_manager):
    """
    Orchestrates the AI response pipeline:
    1. Prepares context and history.
    2. Calls the appropriate backend function (Direct/RAG/LightRAG).
    3. Updates the UI with status steps.
    4. Saves the result to session history.
    """
    
    # Extract Configs for easier access
    llm_cfg = config["llm_config"]
    rag_cfg = config["rag_config"]
    
    # Get the last user message (the query)
    last_msg = session_data["messages"][-1]
    user_query = last_msg["content"]

    # --- 1. Smart Context Construction ---
    # Combine User Query with any Uploaded File Context
    final_query_text = user_query
    if session_data.get("temp_context"):
        final_query_text = (
            f"Session Context:\n{session_data['temp_context']}\n\n"
            f"Query: {user_query}"
        )

    # Calculate available tokens for history
    # Context Window - (Reserve for Output + Retrieval Chunks)
    estimated_retrieval_tokens = rag_cfg["top_k"] * 200 # approx 200 tokens per chunk
    safe_ctx_limit = llm_cfg["context_window"] - 1500 - estimated_retrieval_tokens
    
    # Get history excluding the very last message (which is the current query)
    history_msgs = [{"role": m["role"], "content": m["content"]} for m in session_data["messages"][:-1]]
    
    # Prune history to fit context
    optimized_history = app_utils.smart_prune_history(history_msgs, safe_ctx_limit)
    
    # Enforce user slider limit
    if rag_cfg["history_limit"] < len(optimized_history):
        optimized_history = optimized_history[-rag_cfg["history_limit"]:]

    # --- 2. Execution Pipeline ---
    with st.chat_message("assistant"):
        start_time = time.time()
        response_placeholder = st.empty()
        
        # Prepare arguments common to all query functions
        # Note: We structure llm_config to match what query_data expects
        query_args = {
            "query_text": final_query_text,
            "llm_config": {
                "provider": llm_cfg["provider"],
                "modelName": llm_cfg["model_name"],
                "apiKey": llm_cfg["api_key"],
                "api_url": llm_cfg["local_url"],
                "temperature": llm_cfg["temperature"],
                "system_prompt": llm_cfg["system_prompt"],
                "context_window": llm_cfg["context_window"]
            },
            "conversation_history": optimized_history
        }

        # Status Indicator
        with st.status("🧠 Orchestrating...", expanded=True) as status:
            try:
                rag_type = rag_cfg["rag_type"]
                db_name = rag_cfg["db_name"]
                response = {}

                # A. Direct Chat
                if rag_type == 'direct':
                    status.write("Direct LLM query (no retrieval)...")
                    response = query_direct(**query_args, verify=rag_cfg["verify"])

                # B. Standard RAG & LightRAG
                elif db_name:
                    status.write(f"🔍 Retrieving from **{db_name}** ({rag_type.upper()})...")
                    
                    # Select Strategy
                    strategy_func = query_lightrag if rag_type == 'lightrag' else query_rag
                    
                    response = strategy_func(
                        **query_args,
                        db_name=db_name,
                        top_k=rag_cfg["top_k"],
                        hybrid=rag_cfg["hybrid"],
                        verify=rag_cfg["verify"]
                    )
                    
                    src_count = len(response.get("sources", []))
                    status.write(f"✅ Found {src_count} relevant documents.")

                else:
                    raise ValueError(f"Please select a database for {rag_type}.")

                status.update(label="Response Generated!", state="complete", expanded=False)

            except Exception as e:
                status.update(label="❌ Pipeline Error", state="error")
                st.error(f"An error occurred: {str(e)}")
                logging.error("Pipeline Failure", exc_info=True)
                return # Stop execution safely

        # --- 3. Output Parsing ---
        raw_text = response.get("text", "")
        sources = response.get("sources", [])
        est_tokens = response.get("estimated_context_tokens", 0)
        
        # Update State with Token Count (for the bar chart)
        state_manager.set_last_retrieval_count(est_tokens)

        # Parse Reasoning (Think Tags)
        clean_text, reasoning = app_utils.parse_reasoning(raw_text)

        # --- 4. Render Final Response ---
        # Show Reasoning if available
        if reasoning:
            with st.expander("💭 Internal Thought Process", expanded=False):
                st.markdown(reasoning)

        # Show Content
        formatted_response = app_utils.format_citations(clean_text)
        response_placeholder.markdown(formatted_response, unsafe_allow_html=True)

        # Show Sources
        if sources:
            with st.expander(f"📚 Cited Sources ({len(sources)})", expanded=False):
                unique_sources = list(set(sources))
                for src in unique_sources:
                    col_ico, col_txt = st.columns([0.05, 0.95])
                    col_ico.text("📄")
                    col_txt.caption(f"{os.path.basename(src)} — `{src}`")

        # Footer Stats
        c_time, c_tok = st.columns([0.2, 0.8])
        c_time.caption(f"⏱️ {time.time()-start_time:.2f}s")
        c_tok.caption(f"🪙 ~{est_tokens} tokens context")

        # --- 5. Persist Data ---
        session_data["messages"].append({
            "role": "assistant",
            "content": clean_text,
            "reasoning": reasoning,
            "sources": sources
        })

        # --- 5. Persist Data with METADATA ---
        # We attach the usage stats directly to the message. 
        # This ensures that when we reload the chat, the bar chart is accurate.
        session_data["messages"].append({
            "role": "assistant",
            "content": clean_text,
            "reasoning": reasoning,
            "sources": sources,
            "usage": {
                "retrieval_tokens": est_tokens,
                "timestamp": time.time()
            }
        })

        # Auto-Rename Logic
        if len(session_data["messages"]) == 2:
            _auto_rename_session(session_data, clean_text, state_manager)

        session_manager.save_session(session_data)
        st.rerun() 

def _auto_rename_session(session_data, response_text, state_manager):
    """Generates a short title based on the first response."""
    # Simple truncation strategy
    # (Ideally, you'd ask the LLM to summarize, but this saves a call)
    clean = response_text.replace("#", "").replace("*", "").strip()
    new_title = (clean[:30] + "...") if len(clean) > 30 else clean
    
    session_data["title"] = new_title
    
    # Update state immediately so Sidebar reflects it
    state_manager.update_session_title(session_data["id"], new_title)