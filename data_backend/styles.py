"""
Minimalistic, professional styling for LightRAG Architect UI.

Conservative design principles:
- Clean typography with intentional hierarchy
- Subtle borders and spacing
- No flashy effects or gradients
- Professional color palette
"""

import streamlit as st


def apply_custom_styles():
    """Apply minimalistic, professional CSS styling."""
    st.markdown("""
    <style>
    /* ============================================
       MINIMALISTIC DESIGN SYSTEM
       ============================================ */

    :root {
        --primary: #2563eb;
        --text-primary: #e5e7eb;
        --text-secondary: #9ca3af;
        --text-muted: #6b7280;
        --bg-primary: #0f1117;
        --bg-secondary: #1a1d24;
        --bg-tertiary: #262930;
        --border: #333741;
        --border-light: #404754;
        --success: #059669;
        --warning: #d97706;
        --error: #dc2626;
    }

    /* ============================================
       LAYOUT & SPACING
       ============================================ */

    .main {
        padding: 2rem 3rem;
        max-width: 1600px;
        margin: 0 auto;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    /* ============================================
       TYPOGRAPHY
       ============================================ */

    * {
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
    }

    h1, h2, h3, h4, h5, h6 {
        font-weight: 600;
        letter-spacing: -0.01em;
        color: var(--text-primary);
    }

    h1 {
        font-size: 1.875rem;
        margin-bottom: 0.5rem;
    }

    h2 {
        font-size: 1.5rem;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }

    h3 {
        font-size: 1.25rem;
        margin-top: 1.5rem;
        margin-bottom: 0.75rem;
    }

    p {
        line-height: 1.6;
        color: var(--text-secondary);
    }

    /* ============================================
       BUTTONS
       ============================================ */

    .stButton > button {
        background: var(--bg-secondary);
        color: var(--text-primary);
        border: 1px solid var(--border);
        border-radius: 6px;
        padding: 0.5rem 1rem;
        font-weight: 500;
        font-size: 0.875rem;
        transition: all 0.15s ease;
    }

    .stButton > button:hover {
        background: var(--bg-tertiary);
        border-color: var(--border-light);
    }

    .stButton > button[kind="primary"] {
        background: var(--primary);
        border-color: var(--primary);
        color: white;
    }

    .stButton > button[kind="primary"]:hover {
        background: #1d4ed8;
        border-color: #1d4ed8;
    }

    /* ============================================
       INPUTS
       ============================================ */

    .stTextInput > div > div > input,
    .stNumberInput > div > div > input,
    .stTextArea > div > div > textarea {
        background: var(--bg-secondary);
        border: 1px solid var(--border);
        border-radius: 6px;
        color: var(--text-primary);
        font-size: 0.875rem;
        padding: 0.5rem 0.75rem;
    }

    /* Selectbox fix for dropdown visibility */
    .stSelectbox > div > div {
        background: var(--bg-secondary);
        border: 1px solid var(--border);
        border-radius: 6px;
        color: var(--text-primary);
        font-size: 0.875rem;
    }

    .stSelectbox [data-baseweb="select"] > div {
        min-height: 38px;
    }

    /* Dropdown menu */
    [data-baseweb="popover"] {
        background: var(--bg-secondary);
    }

    [role="listbox"] {
        background: var(--bg-secondary) !important;
        border: 1px solid var(--border) !important;
    }

    [role="option"] {
        background: var(--bg-secondary) !important;
        color: var(--text-primary) !important;
        padding: 0.5rem 0.75rem !important;
    }

    [role="option"]:hover {
        background: var(--bg-tertiary) !important;
    }

    .stTextInput > div > div > input:focus,
    .stNumberInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus {
        border-color: var(--primary);
        outline: none;
        box-shadow: 0 0 0 1px var(--primary);
    }

    /* ============================================
       TABS
       ============================================ */

    .stTabs {
        background: transparent;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
        border-bottom: 1px solid var(--border);
    }

    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border: none;
        color: var(--text-secondary);
        padding: 0.75rem 1.5rem;
        font-weight: 500;
        font-size: 0.875rem;
    }

    .stTabs [data-baseweb="tab"]:hover {
        color: var(--text-primary);
    }

    .stTabs [aria-selected="true"] {
        color: var(--primary);
        border-bottom: 2px solid var(--primary);
    }

    /* ============================================
       METRICS
       ============================================ */

    [data-testid="stMetric"] {
        background: var(--bg-secondary);
        border: 1px solid var(--border);
        border-radius: 6px;
        padding: 1rem;
    }

    [data-testid="stMetricLabel"] {
        color: var(--text-secondary);
        font-size: 0.75rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    [data-testid="stMetricValue"] {
        color: var(--text-primary);
        font-size: 1.5rem;
        font-weight: 600;
    }

    /* ============================================
       DATAFRAMES
       ============================================ */

    [data-testid="stDataFrame"] {
        border: 1px solid var(--border);
        border-radius: 6px;
    }

    [data-testid="stDataFrame"] thead tr th {
        background: var(--bg-secondary) !important;
        border-bottom: 1px solid var(--border) !important;
        color: var(--text-secondary) !important;
        font-size: 0.75rem !important;
        font-weight: 500 !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        padding: 0.75rem 1rem !important;
    }

    [data-testid="stDataFrame"] tbody tr td {
        border-bottom: 1px solid var(--border) !important;
        padding: 0.75rem 1rem !important;
        font-size: 0.875rem;
    }

    /* ============================================
       EXPANDERS
       ============================================ */

    .streamlit-expanderHeader {
        background: var(--bg-secondary);
        border: 1px solid var(--border);
        border-radius: 6px;
        font-weight: 500;
        font-size: 0.875rem;
        padding: 0.75rem 1rem;
    }

    .streamlit-expanderHeader:hover {
        border-color: var(--border-light);
    }

    .streamlit-expanderContent {
        border: 1px solid var(--border);
        border-top: none;
        border-radius: 0 0 6px 6px;
        background: var(--bg-secondary);
        padding: 1rem;
    }

    /* ============================================
       ALERTS
       ============================================ */

    .stAlert {
        background: var(--bg-secondary);
        border: 1px solid var(--border);
        border-radius: 6px;
        padding: 0.75rem 1rem;
        font-size: 0.875rem;
    }

    /* ============================================
       PROGRESS
       ============================================ */

    .stProgress > div > div > div {
        background: var(--bg-secondary);
        border-radius: 4px;
    }

    .stProgress > div > div > div > div {
        background: var(--primary);
        border-radius: 4px;
    }

    /* ============================================
       FILE UPLOADER
       ============================================ */

    [data-testid="stFileUploader"] {
        background: var(--bg-secondary);
        border: 1px dashed var(--border);
        border-radius: 6px;
        padding: 2rem;
    }

    [data-testid="stFileUploader"]:hover {
        border-color: var(--border-light);
    }

    /* ============================================
       SIDEBAR
       ============================================ */

    [data-testid="stSidebar"] {
        background: var(--bg-primary);
        border-right: 1px solid var(--border);
    }

    [data-testid="stSidebar"] hr {
        border: none;
        border-top: 1px solid var(--border);
        margin: 1.5rem 0;
    }

    /* ============================================
       CODE
       ============================================ */

    code {
        background: var(--bg-tertiary);
        color: var(--text-primary);
        border: 1px solid var(--border);
        border-radius: 3px;
        padding: 0.125rem 0.375rem;
        font-size: 0.875rem;
    }

    pre {
        background: var(--bg-secondary) !important;
        border: 1px solid var(--border) !important;
        border-radius: 6px !important;
        padding: 1rem !important;
    }

    /* ============================================
       DIVIDERS
       ============================================ */

    hr {
        border: none;
        border-top: 1px solid var(--border);
        margin: 2rem 0;
    }

    /* ============================================
       SCROLLBAR
       ============================================ */

    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }

    ::-webkit-scrollbar-track {
        background: var(--bg-secondary);
    }

    ::-webkit-scrollbar-thumb {
        background: var(--border-light);
        border-radius: 4px;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: #4b5563;
    }

    /* ============================================
       RESPONSIVE
       ============================================ */

    @media (max-width: 768px) {
        .main {
            padding: 1rem;
        }
    }

    </style>
    """, unsafe_allow_html=True)


def metric_card(label: str, value: str) -> str:
    """Simple metric card with clean design."""
    return f"""
    <div style="background: var(--bg-secondary); border: 1px solid var(--border);
                border-radius: 6px; padding: 1rem; text-align: center;">
        <div style="color: var(--text-secondary); font-size: 0.75rem;
                    font-weight: 500; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.5rem;">
            {label}
        </div>
        <div style="color: var(--text-primary); font-size: 1.5rem; font-weight: 600;">
            {value}
        </div>
    </div>
    """
