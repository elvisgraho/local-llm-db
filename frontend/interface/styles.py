import streamlit as st

def apply_custom_styles():
    """
    Injects global CSS to customize the Streamlit interface.
    Handles:
    - Layout maximizers (padding reduction)
    - Chat message rendering
    - Code block formatting (Fira Code)
    - Citation/Source styling
    """
    
    st.markdown("""
    <style>
        /* =============================================
           1. MAIN LAYOUT & CONTAINER
           ============================================= */
        
        /* Reduce top padding to make better use of screen space */
        .main .block-container { 
            max_width: 95%; 
            padding-top: 1.5rem; 
            padding-bottom: 3rem; 
        }

        /* Remove default streamlit menu burger if desired (optional, currently commented out) */
        /* #MainMenu {visibility: hidden;} */
        /* footer {visibility: hidden;} */

        /* =============================================
           2. CHAT ELEMENTS
           ============================================= */
        
        /* Transparent background for chat messages for a cleaner look */
        .stChatMessage { 
            background-color: transparent; 
            border-bottom: 1px solid rgba(128, 128, 128, 0.1); 
            padding-top: 1rem;
            padding-bottom: 1rem;
        }

        /* User message distinct styling (optional) */
        .stChatMessage[data-testid="stChatMessage"]:nth-child(odd) {
            /* background-color: rgba(255, 255, 255, 0.02); */
        }

        /* =============================================
           3. TYPOGRAPHY & CODE
           ============================================= */
        
        /* Force Fira Code or Consolas for all code blocks */
        code { 
            font-family: 'Fira Code', 'Consolas', 'Monaco', monospace; 
            font-size: 0.9em; 
        }

        /* FLATTENED CODE BLOCKS 
           (Used when the LLM puts code inside a Markdown table) 
        */
        .flattened-code {
            font-family: 'Fira Code', 'Consolas', monospace;
            font-size: 0.85em;
            white-space: pre-wrap;       /* Wrap text so table doesn't overflow */
            word-break: break-all;       /* Break long strings/hashes */
            background-color: rgba(128, 128, 128, 0.08);
            border: 1px solid rgba(128, 128, 128, 0.2);
            border-radius: 4px;
            padding: 6px;
            margin: 4px 0;
            display: block;
        }

        /* =============================================
           4. CUSTOM WIDGETS (CITATIONS & TOKEN BARS)
           ============================================= */
        
        /* Source Citation Badges */
        .source-citation {
            display: inline-flex;
            align-items: center;
            background-color: rgba(0, 173, 181, 0.15); 
            border: 1px solid rgba(0, 173, 181, 0.4);
            border-radius: 4px;
            padding: 2px 6px;
            margin: 0 3px;
            font-size: 0.8em;
            color: #00ADB5; /* Teal text for contrast */
            font-family: 'Segoe UI', sans-serif;
            vertical-align: middle;
            cursor: default;
            transition: all 0.2s ease;
        }
        
        .source-citation:hover {
            background-color: rgba(0, 173, 181, 0.25);
            border-color: #00ADB5;
        }
        
        .source-citation:before {
            content: "📄";
            margin-right: 4px;
            font-size: 0.9em;
        }

        /* Token Usage Bar Container */
        .token-bar-container {
            background-color: rgba(128,128,128,0.2); 
            border-radius: 5px; 
            height: 8px; 
            width: 100%; 
            margin-bottom: 5px;
            overflow: hidden;
        }
        
        /* Status Animations */
        @keyframes pulse-blue {
            0% { box-shadow: 0 0 0 0 rgba(0, 173, 181, 0.7); }
            70% { box-shadow: 0 0 0 10px rgba(0, 173, 181, 0); }
            100% { box-shadow: 0 0 0 0 rgba(0, 173, 181, 0); }
        }

    </style>
    """, unsafe_allow_html=True)

def inject_sticky_css():
    st.markdown("""
        <style>
        /* 1. Reserve space at the bottom of the page so messages aren't hidden */
        .main .block-container {
            padding-bottom: 200px !important;
        }

        /* 2. The Sticky Footer Container */
        .sticky-input-container {
            position: fixed;
            bottom: 0;
            left: 0;
            right: 0;
            width: 100%;
            background-color: #0E1117; /* Matches Streamlit Dark Theme */
            border-top: 1px solid #262730;
            padding: 1rem 2rem 1rem 2rem; /* Top/Right/Bottom/Left */
            z-index: 1000; /* Sit above standard content */
            display: flex;
            flex-direction: column;
            gap: 0.5rem;
        }

        /* 3. Make sure it respects the sidebar if possible (Visual adjustment) */
        @media (min-width: 576px) {
            .sticky-input-container {
                /* This is a safe margin to avoid sidebar overlap on standard screens */
                /* If sidebar is collapsed, it just adds nice whitespace */
                padding-left: 20rem; 
                padding-right: 20rem;
            }
        }
        
        /* 4. Hide the standard Streamlit footer to prevent clutter */
        footer {visibility: hidden;}
        </style>
    """, unsafe_allow_html=True)