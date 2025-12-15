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
    Orchestrates the AI response pipeline with optimized error handling 
    and duplicate prevention.
    """
    
    # 1. Validation: Prevent processing if the last message is already from AI
    # This prevents "double triggers" on rapid UI refreshes.
    if not session_data.get("messages") or session_data["messages"][-1]["role"] == "assistant":
        return

    # Extract Configs
    llm_cfg = config["llm_config"]
    rag_cfg = config["rag_config"]
    
    # Get the last user message
    last_msg = session_data["messages"][-1]
    user_query = last_msg["content"]

    # --- 2. Smart Context Construction ---
    # Combine User Query with any Uploaded File Context
    final_query_text = user_query
    
    # Optimization: Only inject context if it exists and isn't empty
    if session_data.get("temp_context"):
        final_query_text = (
            f"Session Context:\n{session_data['temp_context']}\n\n"
            f"Query: {user_query}"
        )

    # --- 3. History Pruning (Performance) ---
    # Calculate safe context limits
    estimated_retrieval_tokens = rag_cfg["top_k"] * 200 
    # Ensure limit never goes negative even with high top_k
    safe_ctx_limit = max(1000, llm_cfg["context_window"] - 1500 - estimated_retrieval_tokens)
    
    # Get history excluding the current query
    history_msgs = [{"role": m["role"], "content": m["content"]} for m in session_data["messages"][:-1]]
    
    # Prune
    optimized_history = app_utils.smart_prune_history(history_msgs, safe_ctx_limit)
    
    # Enforce User Slider Limit
    if rag_cfg["history_limit"] < len(optimized_history):
        optimized_history = optimized_history[-rag_cfg["history_limit"]:]

    # --- 4. Execution Pipeline ---
    with st.chat_message("assistant"):
        start_time = time.time()
        response_placeholder = st.empty()
        
        # Args preparation
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
                    status.write("Direct LLM query...")
                    response = query_direct(**query_args, verify=rag_cfg["verify"])

                # B. Standard RAG & LightRAG
                elif db_name:
                    status.write(f"🔍 Retrieving from **{db_name}**...")
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
                return # Exit to prevent saving broken state

        # --- 5. Output Parsing ---
        raw_text = response.get("text", "")
        if not raw_text:
            st.error("Received empty response from LLM.")
            return

        sources = response.get("sources", [])
        est_tokens = response.get("estimated_context_tokens", 0)
        
        # Update State immediately for UI responsiveness
        state_manager.set_last_retrieval_count(est_tokens)

        # Parse Reasoning
        clean_text, reasoning = app_utils.parse_reasoning(raw_text)

        # --- 6. Render Transient UI ---
        # We render this here so the user sees it BEFORE the rerun triggers.
        if reasoning:
            with st.expander("💭 Internal Thought Process", expanded=False):
                st.markdown(reasoning)

        formatted_response = app_utils.format_citations(clean_text)
        response_placeholder.markdown(formatted_response, unsafe_allow_html=True)

        if sources:
            with st.expander(f"📚 Cited Sources ({len(sources)})", expanded=False):
                unique_sources = list(set(sources))
                for src in unique_sources:
                    col_ico, col_txt = st.columns([0.05, 0.95])
                    col_ico.text("📄")
                    col_txt.caption(f"{os.path.basename(src)}")

        # Stats Footer
        st.caption(f"⏱️ {time.time()-start_time:.2f}s | 🪙 ~{est_tokens} tokens")

        # --- 7. Persist Data (FIXED) ---
        # Removed the duplicate .append() call. 
        # Added metadata 'usage' for the Token Estimator.
        
        new_message = {
            "role": "assistant",
            "content": clean_text,
            "reasoning": reasoning,
            "sources": sources,
            "usage": {
                "retrieval_tokens": est_tokens,
                "timestamp": time.time()
            }
        }
        
        # Double-check against duplicate appends (Edge Case Protection)
        last_saved = session_data["messages"][-1]
        if last_saved["role"] != "assistant" or last_saved["content"] != clean_text:
            session_data["messages"].append(new_message)

        # Auto-Rename (Only on first interaction)
        if len(session_data["messages"]) == 2:
            _auto_rename_session(session_data, clean_text, state_manager)

        # Save to Disk
        session_manager.save_session(session_data)
        
        # Force UI Refresh to sync sidebar and chat history cleanly
        st.rerun()

def _auto_rename_session(session_data, response_text, state_manager):
    """Generates a short title based on the first response."""
    # Remove markdown symbols for cleaner titles
    clean = response_text.replace("#", "").replace("*", "").replace("`", "").strip()
    new_title = (clean[:30] + "...") if len(clean) > 30 else clean
    
    session_data["title"] = new_title
    state_manager.update_session_title(session_data["id"], new_title)