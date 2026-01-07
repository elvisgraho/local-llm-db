import streamlit as st
import time
import logging
import os

# --- Internal Imports ---
import app_utils
from query.templates import REFINE_QUERY_PROMPT
from query.query_data import query_direct, query_lightrag
from query.session_manager import session_manager
from query.config import config as global_config

def process_user_input(session_data, config, state_manager, container=None):
    """
    Orchestrates the AI response pipeline.
    
    Args:
        session_data: The active session dictionary.
        config: App configuration.
        state_manager: State manager instance.
        container: (Optional) The Streamlit container to render output into. 
                   Crucial for Sticky Footer layouts to prevent overlapping.
    """
    
    # 0. Define Output Target
    # If a specific container (history) is provided, use it. Otherwise, default to main page.
    target_ui = container if container else st
    
    # 1. Validation
    if not session_data.get("messages") or session_data["messages"][-1]["role"] == "assistant":
        return

    # Extract Configs
    llm_cfg = config["llm_config"]
    rag_cfg = config["rag_config"]
    
    last_msg = session_data["messages"][-1]
    raw_user_query = last_msg["content"]
    
    # --- 2. PREPARE HISTORY ---
    estimated_retrieval_tokens = rag_cfg["top_k"] * 250 
    safe_ctx_limit = max(1000, llm_cfg["context_window"] - 1500 - estimated_retrieval_tokens)
    
    raw_history = [{"role": m["role"], "content": m["content"]} for m in session_data["messages"][:-1]]
    optimized_history = app_utils.smart_prune_history(raw_history, safe_ctx_limit)
    
    if rag_cfg["history_limit"] < len(optimized_history):
        optimized_history = optimized_history[-rag_cfg["history_limit"]:]

    effective_query = raw_user_query

    # --- 3. CONTEXTUAL REWRITE (Rendered inside Container) ---
    if rag_cfg.get("rag_rewrite", False):
        # Use target_ui.status to ensure it stays in the scrollable area
        with target_ui.status("✨ Contextualizing Query...", expanded=False) as status:
            try:
                rewrite_cfg = llm_cfg.copy()
                rewrite_cfg["system_prompt"] = REFINE_QUERY_PROMPT
                rewrite_cfg["temperature"] = 0.3

                rewrite_response = query_direct(
                    query_text=raw_user_query,
                    llm_config=rewrite_cfg,
                    conversation_history=optimized_history
                )
                
                optimized_text = rewrite_response.get("text", "").strip()
                
                if optimized_text and len(optimized_text) > 3:
                    effective_query = optimized_text
                    status.write(f"**Original:** {raw_user_query}")
                    status.write(f"**Contextualized:** {effective_query}")
                    status.update(label="Query Contextualized", state="complete")
                else:
                    status.update(label="Optimization unclear, using original.", state="error")

            except Exception as e:
                logging.error(f"Auto-rewrite failed: {e}")
                status.update(label="Optimization skipped (Error)", state="error")

    # --- 4. CONTEXT INJECTION ---
    final_query_payload = effective_query
    if session_data.get("temp_context"):
        final_query_payload = (
            f"Session Context:\n{session_data['temp_context']}\n\n"
            f"Query: {effective_query}"
        )

    # --- 5. EXECUTION PIPELINE (Rendered inside Container) ---
    # Apply UI chunk expansion settings to global config
    global_config.rag.enable_chunk_expansion = rag_cfg.get("enable_chunk_expansion", True)
    global_config.rag.chunk_expansion_window = rag_cfg.get("chunk_expansion_window", 1)
    global_config.rag.chunk_expansion_for_code_only = rag_cfg.get("chunk_expansion_code_only", True)

    # Use target_ui.chat_message to append to the history container specifically
    with target_ui.chat_message("assistant"):
        start_time = time.time()
        response_placeholder = st.empty()

        query_args = {
            "query_text": final_query_payload,
            "llm_config": llm_cfg,
            "conversation_history": optimized_history
        }

        status_label = "🧠 Analyzing..." if not rag_cfg.get("rag_rewrite") else "🧠 Processing Contextualized Query..."

        with st.status(status_label, expanded=True) as status:
            try:
                rag_type = rag_cfg["rag_type"]
                db_name = rag_cfg["db_name"]
                response = {}

                if rag_type == 'direct':
                    status.write("Direct LLM query...")
                    response = query_direct(**query_args, verify=rag_cfg["verify"])

                elif db_name:
                    status.write(f"🔍 Retrieving from **{db_name}**...")
                    strategy_func = query_lightrag
                    
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
                error_content = f"❌ **Error:** {str(e)}"
                logging.error("Pipeline Failure", exc_info=True)

                # Display the error immediately in the response placeholder
                response_placeholder.markdown(error_content, unsafe_allow_html=True)

                # Save error message to chat history so it persists
                error_message = {
                    "role": "assistant",
                    "content": error_content,
                    "sources": [],
                    "usage": {
                        "retrieval_tokens": 0,
                        "timestamp": time.time()
                    }
                }
                session_data["messages"].append(error_message)

                # Auto-rename if this is the first exchange (even on error)
                if len(session_data["messages"]) == 2:
                    _auto_rename_session(session_data, "", state_manager)

                session_manager.save_session(session_data)
                st.rerun()
                return 

        # --- 6. OUTPUT PARSING ---
        raw_text = response.get("text", "")
        if not raw_text:
            error_content = "❌ **Error:** Received empty response from LLM."

            # Display the error immediately in the response placeholder
            response_placeholder.markdown(error_content, unsafe_allow_html=True)

            # Save error message to chat history so it persists
            error_message = {
                "role": "assistant",
                "content": error_content,
                "sources": [],
                "usage": {
                    "retrieval_tokens": 0,
                    "timestamp": time.time()
                }
            }
            session_data["messages"].append(error_message)

            # Auto-rename if this is the first exchange (even on error)
            if len(session_data["messages"]) == 2:
                _auto_rename_session(session_data, "", state_manager)

            session_manager.save_session(session_data)
            st.rerun()
            return

        sources = response.get("sources", [])
        est_tokens = response.get("estimated_context_tokens", 0)
        
        state_manager.set_last_retrieval_count(est_tokens)

        clean_text, reasoning = app_utils.parse_reasoning(raw_text)

        # --- 7. TRANSIENT UI ---
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

        st.caption(f"⏱️ {time.time()-start_time:.2f}s | 🪙 ~{est_tokens} tokens")

        # --- 8. PERSIST DATA ---
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
        
        last_saved = session_data["messages"][-1]
        if last_saved["role"] != "assistant" or last_saved["content"] != clean_text:
            session_data["messages"].append(new_message)

        if len(session_data["messages"]) == 2:
            _auto_rename_session(session_data, clean_text, state_manager)

        session_manager.save_session(session_data)
        
        # Force refresh to sync UI
        st.rerun()

def _auto_rename_session(session_data, response_text, state_manager):
    """
    Generates a short title based on the first response.
    Falls back to user message if response is empty or invalid.
    Only renames if the current title is generic ("New Chat", "Chat", etc.)
    """
    try:
        current_title = session_data.get("title", "").strip()

        # Skip renaming if the chat already has a meaningful custom title
        # (not a generic one like "New Chat" or auto-generated ones)
        generic_titles = ["New Chat", "Chat", "Untitled Chat", ""]
        is_generic = current_title in generic_titles or current_title.startswith("Chat ")

        if not is_generic:
            # Already has a good title, don't override
            return

        # Try to use the assistant's response first
        clean = response_text.replace("#", "").replace("*", "").replace("`", "").strip()

        # If response is too short or empty, use the user's message instead
        if len(clean) < 3:
            # Get the first user message
            user_msg = next((m["content"] for m in session_data["messages"] if m["role"] == "user"), None)
            if user_msg:
                clean = user_msg.replace("#", "").replace("*", "").replace("`", "").strip()

        # Final fallback if still empty
        if len(clean) < 3:
            new_title = f"Chat {session_data.get('id', '')[:8]}"
        else:
            # Truncate to reasonable length
            new_title = (clean[:35] + "...") if len(clean) > 35 else clean

        session_data["title"] = new_title
        state_manager.update_session_title(session_data["id"], new_title)

    except Exception as e:
        logging.error(f"Failed to auto-rename session: {e}")
        # Use a safe default
        session_data["title"] = f"Chat {session_data.get('id', '')[:8]}"