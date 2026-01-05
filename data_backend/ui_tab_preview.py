"""
Enhanced Chunking Preview Tab with optimization for large documents.

Features:
- Side-by-side chunking comparison
- Memory-efficient document preview
- Interactive chunk explorer
- Visual overlap highlighting
- Chunk statistics and analysis
- Export chunking configuration
"""

import streamlit as st
from pathlib import Path
from typing import List, Dict, Tuple
from langchain_text_splitters import RecursiveCharacterTextSplitter
from common import RAW_FILES_DIR, extract_text_for_preview


# Constants
MAX_PREVIEW_CHARS = 50000  # Limit for preview to avoid memory issues
MAX_CHUNKS_DISPLAY = 10    # Max chunks to display at once


def get_file_list() -> List[Path]:
    """Get list of files available for preview."""
    if not RAW_FILES_DIR.exists():
        return []

    files = [
        f for f in RAW_FILES_DIR.rglob('*')
        if f.is_file() and not f.name.startswith('.')
    ]

    # Sort by modification time (newest first)
    files.sort(key=lambda x: x.stat().st_mtime, reverse=True)

    return files


def create_splitter(chunk_size: int, chunk_overlap: int) -> RecursiveCharacterTextSplitter:
    """Create text splitter with given parameters."""
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
        separators=["\n\n", "\n", ". ", "? ", "! ", " ", ""],
        keep_separator=True
    )


def analyze_chunks(chunks: List[str]) -> Dict:
    """
    Analyze chunk statistics.

    Args:
        chunks: List of text chunks

    Returns:
        Dictionary with statistics
    """
    if not chunks:
        return {
            "count": 0,
            "avg_length": 0,
            "min_length": 0,
            "max_length": 0,
            "total_chars": 0
        }

    lengths = [len(c) for c in chunks]

    return {
        "count": len(chunks),
        "avg_length": sum(lengths) / len(lengths),
        "min_length": min(lengths),
        "max_length": max(lengths),
        "total_chars": sum(lengths),
        "empty_chunks": sum(1 for c in chunks if not c.strip())
    }


def highlight_overlap(chunk1: str, chunk2: str, overlap_size: int) -> Tuple[str, str]:
    """
    Highlight overlapping text between two consecutive chunks.

    Args:
        chunk1: First chunk
        chunk2: Second chunk
        overlap_size: Expected overlap size

    Returns:
        Tuple of (highlighted_chunk1, highlighted_chunk2)
    """
    # Simple overlap detection (last N chars of chunk1 should match first N chars of chunk2)
    if overlap_size > 0 and len(chunk1) >= overlap_size and len(chunk2) >= overlap_size:
        overlap_text = chunk1[-overlap_size:]

        # Check if it actually overlaps
        if chunk2.startswith(overlap_text):
            highlighted_chunk1 = chunk1[:-overlap_size] + f"<mark>{overlap_text}</mark>"
            highlighted_chunk2 = f"<mark>{chunk2[:overlap_size]}</mark>" + chunk2[overlap_size:]
            return highlighted_chunk1, highlighted_chunk2

    return chunk1, chunk2


def render_preview_tab():
    """Render the enhanced chunking preview tab."""

    # Initialize session state
    if "preview_chunk_size" not in st.session_state:
        st.session_state.preview_chunk_size = 512
    if "preview_chunk_overlap" not in st.session_state:
        st.session_state.preview_chunk_overlap = 100
    if "preview_comparison_mode" not in st.session_state:
        st.session_state.preview_comparison_mode = False
    if "preview_show_overlap" not in st.session_state:
        st.session_state.preview_show_overlap = True

    st.markdown("### Chunking Laboratory")
    st.caption("Test and visualize how your documents will be split into chunks")

    # File selection
    all_files = get_file_list()

    if not all_files:
        st.info("📂 No files available. Upload documents in the Import tab first.")
        return

    # File selector
    file_col, options_col = st.columns([2, 1])

    with file_col:
        selected_file = st.selectbox(
            "Select file to analyze",
            all_files,
            format_func=lambda f: f"📄 {f.name} ({f.stat().st_size / 1024:.1f} KB)"
        )

    with options_col:
        st.markdown("####  ")  # Spacing
        comparison_mode = st.checkbox(
            "Comparison Mode",
            value=st.session_state.preview_comparison_mode,
            help="Compare two different chunking strategies side by side"
        )
        st.session_state.preview_comparison_mode = comparison_mode

    st.divider()

    # Load document content
    if selected_file:
        with st.spinner("Loading document..."):
            content = extract_text_for_preview(selected_file, char_limit=MAX_PREVIEW_CHARS)

        if "Error" in content:
            st.error(content)
            return

        # Document info
        doc_length = len(content)
        was_truncated = doc_length >= MAX_PREVIEW_CHARS

        info_col1, info_col2, info_col3 = st.columns(3)

        with info_col1:
            st.metric("Document Length", f"{doc_length:,} chars")
        with info_col2:
            st.metric("File Size", f"{selected_file.stat().st_size / 1024:.1f} KB")
        with info_col3:
            if was_truncated:
                st.metric("Preview", "Truncated", help="Only first 50k characters shown")
            else:
                st.metric("Preview", "Complete")

        st.divider()

        # === COMPARISON MODE ===
        if comparison_mode:
            st.markdown("#### Compare Chunking Strategies")

            config_col1, config_col2 = st.columns(2)

            # Config 1
            with config_col1:
                st.markdown("##### Configuration A")
                chunk_size_a = st.slider(
                    "Chunk Size A",
                    100, 2000, 512,
                    step=50,
                    key="chunk_size_a"
                )
                chunk_overlap_a = st.slider(
                    "Overlap A",
                    0, 500, 100,
                    step=25,
                    key="chunk_overlap_a"
                )

                # Create chunks A
                splitter_a = create_splitter(chunk_size_a, chunk_overlap_a)
                chunks_a = splitter_a.split_text(content)
                stats_a = analyze_chunks(chunks_a)

                st.markdown(f"""
                <div style="background: #1a1d29; padding: 1rem; border-radius: 8px; border: 2px solid #6366f1;">
                    <h4 style="margin: 0 0 0.5rem 0; color: #6366f1;">Strategy A Results</h4>
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 0.5rem;">
                        <div>📊 Chunks: <strong>{stats_a['count']}</strong></div>
                        <div>📏 Avg: <strong>{stats_a['avg_length']:.0f}</strong></div>
                        <div>📉 Min: <strong>{stats_a['min_length']}</strong></div>
                        <div>📈 Max: <strong>{stats_a['max_length']}</strong></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # Config 2
            with config_col2:
                st.markdown("##### Configuration B")
                chunk_size_b = st.slider(
                    "Chunk Size B",
                    100, 2000, 1024,
                    step=50,
                    key="chunk_size_b"
                )
                chunk_overlap_b = st.slider(
                    "Overlap B",
                    0, 500, 200,
                    step=25,
                    key="chunk_overlap_b"
                )

                # Create chunks B
                splitter_b = create_splitter(chunk_size_b, chunk_overlap_b)
                chunks_b = splitter_b.split_text(content)
                stats_b = analyze_chunks(chunks_b)

                st.markdown(f"""
                <div style="background: #1a1d29; padding: 1rem; border-radius: 8px; border: 2px solid #8b5cf6;">
                    <h4 style="margin: 0 0 0.5rem 0; color: #8b5cf6;">Strategy B Results</h4>
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 0.5rem;">
                        <div>📊 Chunks: <strong>{stats_b['count']}</strong></div>
                        <div>📏 Avg: <strong>{stats_b['avg_length']:.0f}</strong></div>
                        <div>📉 Min: <strong>{stats_b['min_length']}</strong></div>
                        <div>📈 Max: <strong>{stats_b['max_length']}</strong></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            st.divider()

            # Side-by-side chunk comparison
            st.markdown("#### Chunk Comparison (First 5)")

            max_compare = min(5, stats_a['count'], stats_b['count'])

            for i in range(max_compare):
                st.markdown(f"**Chunk {i + 1}**")

                compare_col1, compare_col2 = st.columns(2)

                with compare_col1:
                    with st.expander(f"Strategy A ({len(chunks_a[i])} chars)", expanded=(i == 0)):
                        st.code(chunks_a[i][:500] + ("..." if len(chunks_a[i]) > 500 else ""), language="text")

                with compare_col2:
                    with st.expander(f"Strategy B ({len(chunks_b[i])} chars)", expanded=(i == 0)):
                        st.code(chunks_b[i][:500] + ("..." if len(chunks_b[i]) > 500 else ""), language="text")

        # === SINGLE MODE ===
        else:
            st.markdown("#### Chunking Configuration")

            control_col1, control_col2, control_col3 = st.columns([2, 2, 1])

            with control_col1:
                chunk_size = st.slider(
                    "Chunk Size (characters)",
                    100, 2000,
                    st.session_state.preview_chunk_size,
                    step=50,
                    help="Target size for each chunk"
                )
                st.session_state.preview_chunk_size = chunk_size

            with control_col2:
                chunk_overlap = st.slider(
                    "Chunk Overlap (characters)",
                    0, min(500, chunk_size - 10),
                    min(st.session_state.preview_chunk_overlap, chunk_size - 10),
                    step=25,
                    help="Number of overlapping characters between chunks"
                )
                st.session_state.preview_chunk_overlap = chunk_overlap

            with control_col3:
                st.markdown("####  ")  # Spacing
                show_overlap = st.checkbox(
                    "Highlight Overlap",
                    value=st.session_state.preview_show_overlap,
                    help="Visually highlight overlapping text"
                )
                st.session_state.preview_show_overlap = show_overlap

            # Create chunks
            splitter = create_splitter(chunk_size, chunk_overlap)
            chunks = splitter.split_text(content)
            stats = analyze_chunks(chunks)

            st.divider()

            # Statistics
            st.markdown("#### Chunking Statistics")

            stat_col1, stat_col2, stat_col3, stat_col4, stat_col5 = st.columns(5)

            with stat_col1:
                st.metric("Total Chunks", stats['count'])

            with stat_col2:
                st.metric("Avg Length", f"{stats['avg_length']:.0f}")

            with stat_col3:
                st.metric("Min Length", stats['min_length'])

            with stat_col4:
                st.metric("Max Length", stats['max_length'])

            with stat_col5:
                efficiency = (stats['total_chars'] / doc_length) if doc_length > 0 else 0
                st.metric(
                    "Efficiency",
                    f"{efficiency:.1%}",
                    help="Ratio of total chunk characters to original document (>100% due to overlap)"
                )

            # Warnings
            if stats['empty_chunks'] > 0:
                st.warning(f"⚠️ {stats['empty_chunks']} empty chunk(s) detected. Consider adjusting parameters.")

            if stats['max_length'] > chunk_size * 1.5:
                st.warning(f"⚠️ Some chunks exceed target size significantly. Document may contain unsplittable sections.")

            st.divider()

            # Chunk explorer
            st.markdown("#### Chunk Explorer")

            # Pagination for chunks
            chunks_per_page = 5
            total_chunk_pages = (len(chunks) + chunks_per_page - 1) // chunks_per_page

            if "chunk_page" not in st.session_state:
                st.session_state.chunk_page = 0

            chunk_page = st.session_state.chunk_page

            chunk_nav_col1, chunk_nav_col2, chunk_nav_col3 = st.columns([1, 2, 1])

            with chunk_nav_col1:
                if st.button("⬅️ Previous Chunks", disabled=chunk_page == 0, use_container_width=True):
                    st.session_state.chunk_page -= 1
                    st.rerun()

            with chunk_nav_col2:
                st.markdown(f"<div style='text-align: center; padding-top: 0.5rem;'>Page {chunk_page + 1} of {total_chunk_pages}</div>", unsafe_allow_html=True)

            with chunk_nav_col3:
                if st.button("Next Chunks ➡️", disabled=chunk_page >= total_chunk_pages - 1, use_container_width=True):
                    st.session_state.chunk_page += 1
                    st.rerun()

            # Display chunks for current page
            start_idx = chunk_page * chunks_per_page
            end_idx = min(start_idx + chunks_per_page, len(chunks))

            for i in range(start_idx, end_idx):
                chunk_text = chunks[i]

                # Check for overlap with next chunk
                overlap_info = ""
                if show_overlap and i < len(chunks) - 1 and chunk_overlap > 0:
                    next_chunk = chunks[i + 1]
                    # Simple overlap check
                    overlap_candidate = chunk_text[-chunk_overlap:]
                    if next_chunk.startswith(overlap_candidate):
                        overlap_info = f"✓ Overlaps with next chunk ({chunk_overlap} chars)"
                    else:
                        overlap_info = "⚠️ Expected overlap not found"

                # Chunk header
                header_col1, header_col2 = st.columns([3, 1])

                with header_col1:
                    st.markdown(f"##### Chunk {i + 1} of {len(chunks)}")

                with header_col2:
                    st.caption(f"{len(chunk_text)} characters")

                if overlap_info:
                    st.caption(overlap_info)

                # Show chunk content
                with st.expander("View Content", expanded=(i == start_idx)):
                    # Highlight overlap if enabled
                    if show_overlap and i < len(chunks) - 1 and chunk_overlap > 0:
                        overlap_size = min(chunk_overlap, len(chunk_text))
                        if overlap_size > 0:
                            main_part = chunk_text[:-overlap_size]
                            overlap_part = chunk_text[-overlap_size:]

                            st.markdown(
                                f"<div style='font-family: monospace; white-space: pre-wrap; background: #0f1117; padding: 1rem; border-radius: 8px;'>"
                                f"{main_part}<span style='background: rgba(251, 191, 36, 0.3); padding: 2px 4px; border-radius: 4px;'>{overlap_part}</span>"
                                f"</div>",
                                unsafe_allow_html=True
                            )
                        else:
                            st.code(chunk_text, language="text")
                    else:
                        st.code(chunk_text, language="text")

