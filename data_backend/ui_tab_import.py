"""
Enhanced Import Tab with optimized file handling for large datasets.

Features:
- Batch file upload with progress tracking
- Pagination for 10k+ file display
- File filtering and search
- Batch operations (delete, move)
- Smart duplicate detection
- File validation and health checks
"""

import streamlit as st
import os
import shutil
import time
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple
from common import RAW_FILES_DIR, get_file_inventory


# Constants
SUPPORTED_EXTENSIONS = ['.pdf', '.txt', '.md', '.markdown', '.json', '.log', '.py', '.js', '.html', '.css']
MAX_FILE_SIZE_MB = 100
FILES_PER_PAGE = 50


def get_paginated_files(page: int = 0, search: str = "", file_type: str = "all") -> Tuple[List[Dict], int]:
    """
    Get paginated and filtered file list.

    Args:
        page: Page number (0-indexed)
        search: Search query for filename filtering
        file_type: File type filter

    Returns:
        Tuple of (file_list, total_count)
    """
    if not RAW_FILES_DIR.exists():
        return [], 0

    # Get all files
    all_files = [
        f for f in RAW_FILES_DIR.rglob('*')
        if f.is_file() and not f.name.startswith('.')
    ]

    # Apply search filter
    if search:
        all_files = [f for f in all_files if search.lower() in f.name.lower()]

    # Apply file type filter
    if file_type != "all":
        all_files = [f for f in all_files if f.suffix.lower() == f".{file_type}"]

    total_count = len(all_files)

    # Sort by modification time (newest first)
    all_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)

    # Paginate
    start_idx = page * FILES_PER_PAGE
    end_idx = start_idx + FILES_PER_PAGE
    page_files = all_files[start_idx:end_idx]

    # Format for display
    formatted = [{
        "Path": f,
        "Filename": f.name,
        "Size (KB)": f"{f.stat().st_size / 1024:.1f}",
        "Size_bytes": f.stat().st_size,
        "Type": f.suffix.lower(),
        "Modified": datetime.fromtimestamp(f.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S'),
        "Modified_ts": f.stat().st_mtime
    } for f in page_files]

    return formatted, total_count


def validate_file(file_path: Path) -> Tuple[bool, str]:
    """
    Validate uploaded file.

    Args:
        file_path: Path to file

    Returns:
        Tuple of (is_valid, error_message)
    """
    # Check size
    size_mb = file_path.stat().st_size / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        return False, f"File exceeds {MAX_FILE_SIZE_MB}MB limit ({size_mb:.1f}MB)"

    # Check extension
    if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        return False, f"Unsupported file type: {file_path.suffix}"

    # Check if readable
    try:
        with open(file_path, 'rb') as f:
            f.read(1024)  # Try reading first 1KB
    except Exception as e:
        return False, f"File is corrupted or unreadable: {str(e)}"

    return True, ""


def get_storage_stats() -> Dict:
    """Get storage statistics for staging area."""
    if not RAW_FILES_DIR.exists():
        return {"total_files": 0, "total_size_mb": 0, "by_type": {}}

    files = [f for f in RAW_FILES_DIR.rglob('*') if f.is_file() and not f.name.startswith('.')]

    total_size = sum(f.stat().st_size for f in files)
    by_type = {}

    for f in files:
        ext = f.suffix.lower() or 'no_extension'
        if ext not in by_type:
            by_type[ext] = {"count": 0, "size": 0}
        by_type[ext]["count"] += 1
        by_type[ext]["size"] += f.stat().st_size

    return {
        "total_files": len(files),
        "total_size_mb": total_size / (1024 * 1024),
        "by_type": by_type
    }


def render_import_tab():
    """Render the enhanced import tab."""

    # Initialize session state
    if "import_page" not in st.session_state:
        st.session_state.import_page = 0
    if "import_search" not in st.session_state:
        st.session_state.import_search = ""
    if "import_filter" not in st.session_state:
        st.session_state.import_filter = "all"
    if "selected_files" not in st.session_state:
        st.session_state.selected_files = set()

    # Header with stats
    col1, col2, col3, col4 = st.columns([2, 1, 1, 1])

    with col1:
        st.markdown("### Document Staging Area")

    # Get storage stats
    stats = get_storage_stats()

    with col2:
        st.metric("Total Files", f"{stats['total_files']:,}")

    with col3:
        st.metric("Total Size", f"{stats['total_size_mb']:.1f} MB")

    with col4:
        if st.button("🔄 Refresh", use_container_width=True):
            st.rerun()

    st.divider()

    # ===== UPLOAD SECTION =====
    st.markdown("### Upload Documents")

    uploaded_files = st.file_uploader(
        "Drop files here or click to browse",
        accept_multiple_files=True,
        type=[ext[1:] for ext in SUPPORTED_EXTENSIONS],
        help=f"Supported formats: {', '.join(SUPPORTED_EXTENSIONS)}\nMax size: {MAX_FILE_SIZE_MB}MB per file",
        label_visibility="collapsed"
    )

    if uploaded_files:
        st.caption(f"📁 {len(uploaded_files)} file(s) selected")

        # Upload button
        if st.button(f"💾 Upload {len(uploaded_files)} File(s)", type="primary", use_container_width=True):
            success_count = 0
            error_count = 0
            errors = []

            progress_bar = st.progress(0, text="Uploading files...")

            for i, uf in enumerate(uploaded_files):
                try:
                    # Save file
                    file_path = RAW_FILES_DIR / uf.name

                    # Check for duplicates
                    if file_path.exists():
                        # Add timestamp to filename
                        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                        name_parts = uf.name.rsplit('.', 1)
                        if len(name_parts) == 2:
                            new_name = f"{name_parts[0]}_{timestamp}.{name_parts[1]}"
                        else:
                            new_name = f"{uf.name}_{timestamp}"
                        file_path = RAW_FILES_DIR / new_name

                    # Write file
                    with open(file_path, "wb") as f:
                        f.write(uf.getbuffer())

                    # Validate
                    is_valid, error_msg = validate_file(file_path)
                    if is_valid:
                        success_count += 1
                    else:
                        errors.append(f"{uf.name}: {error_msg}")
                        os.remove(file_path)
                        error_count += 1

                except Exception as e:
                    errors.append(f"{uf.name}: {str(e)}")
                    error_count += 1

                # Update progress
                progress_bar.progress((i + 1) / len(uploaded_files))

            progress_bar.empty()

            # Show results
            if success_count > 0:
                st.success(f"✅ Successfully uploaded {success_count} file(s)")

            if error_count > 0:
                st.error(f"❌ Failed to upload {error_count} file(s)")
                with st.expander("View errors"):
                    for error in errors:
                        st.text(error)

            time.sleep(1)
            st.rerun()

    # Storage breakdown
    if stats['total_files'] > 0:
        st.markdown("#### Storage Breakdown")

        for ext, data in sorted(stats['by_type'].items(), key=lambda x: x[1]['count'], reverse=True):
            percentage = (data['count'] / stats['total_files']) * 100
            size_mb = data['size'] / (1024 * 1024)

            st.markdown(f"""
            <div style="margin-bottom: 0.5rem;">
                <div style="display: flex; justify-content: space-between; margin-bottom: 0.25rem;">
                    <span><code>{ext}</code></span>
                    <span style="color: #9ca3af;">{data['count']} files ({size_mb:.1f} MB)</span>
                </div>
                <div style="background: #1a1d29; border-radius: 4px; height: 8px; overflow: hidden;">
                    <div style="background: #2563eb; width: {percentage}%; height: 100%;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.divider()

    # ===== FILE MANAGEMENT SECTION =====
    st.markdown("### File Management")

    # Search and filter controls
    search_col, filter_col, action_col = st.columns([2, 1, 1])

    with search_col:
        search = st.text_input(
            "Search files",
            value=st.session_state.import_search,
            placeholder="Search by filename...",
            label_visibility="collapsed",
            key="search_input"
        )
        if search != st.session_state.import_search:
            st.session_state.import_search = search
            st.session_state.import_page = 0
            st.rerun()

    with filter_col:
        # Get unique file types
        file_types = ["all"] + sorted(list(set(
            f.suffix.lower()[1:] for f in RAW_FILES_DIR.rglob('*')
            if f.is_file() and f.suffix
        ))) if RAW_FILES_DIR.exists() else ["all"]

        file_filter = st.selectbox(
            "Type",
            file_types,
            index=file_types.index(st.session_state.import_filter) if st.session_state.import_filter in file_types else 0,
            label_visibility="collapsed",
            key="filter_select"
        )
        if file_filter != st.session_state.import_filter:
            st.session_state.import_filter = file_filter
            st.session_state.import_page = 0
            st.rerun()

    with action_col:
        if st.button("🗑️ Clear All", use_container_width=True):
            st.session_state.show_clear_confirm = True

    # Get paginated files
    files, total_count = get_paginated_files(
        st.session_state.import_page,
        st.session_state.import_search,
        st.session_state.import_filter
    )

    if total_count == 0:
        st.info("📂 No files found. Upload documents to get started.")
    else:
        # Pagination info
        total_pages = (total_count + FILES_PER_PAGE - 1) // FILES_PER_PAGE
        current_page = st.session_state.import_page + 1

        st.caption(f"Showing {len(files)} of {total_count:,} files (Page {current_page}/{total_pages})")

        # File list with enhanced display
        for idx, file_info in enumerate(files):
            with st.container():
                col_check, col_info, col_actions = st.columns([0.3, 3, 1])

                with col_check:
                    file_key = str(file_info["Path"])
                    is_selected = st.checkbox(
                        "Select",
                        value=file_key in st.session_state.selected_files,
                        key=f"select_{idx}_{current_page}",
                        label_visibility="collapsed"
                    )

                    if is_selected and file_key not in st.session_state.selected_files:
                        st.session_state.selected_files.add(file_key)
                    elif not is_selected and file_key in st.session_state.selected_files:
                        st.session_state.selected_files.remove(file_key)

                with col_info:
                    # File icon based on type
                    icon_map = {
                        '.pdf': '📄',
                        '.txt': '📝',
                        '.md': '📋',
                        '.py': '🐍',
                        '.js': '📜',
                        '.json': '📊'
                    }
                    icon = icon_map.get(file_info["Type"], '📁')

                    st.markdown(f"""
                    <div style="padding: 0.75rem; background: #1a1d29; border-radius: 8px; border: 1px solid #2d3142;">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <div>
                                <span style="font-size: 1.2rem; margin-right: 0.5rem;">{icon}</span>
                                <strong>{file_info['Filename']}</strong>
                            </div>
                            <div style="text-align: right; color: #9ca3af; font-size: 0.875rem;">
                                <div>{file_info['Size (KB)']} KB</div>
                                <div>{file_info['Modified'].split()[1]}</div>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                with col_actions:
                    if st.button("🗑️", key=f"del_{idx}_{current_page}", help="Delete file"):
                        try:
                            os.remove(file_info["Path"])
                            st.success(f"Deleted {file_info['Filename']}")
                            time.sleep(0.5)
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error: {str(e)}")

        # Pagination controls
        st.divider()

        nav_col1, nav_col2, nav_col3 = st.columns([1, 2, 1])

        with nav_col1:
            if st.button("⬅️ Previous", disabled=current_page == 1, use_container_width=True):
                st.session_state.import_page -= 1
                st.rerun()

        with nav_col2:
            # Page selector
            page_options = list(range(1, total_pages + 1))
            selected_page = st.selectbox(
                "Page",
                page_options,
                index=current_page - 1,
                label_visibility="collapsed",
                key="page_select"
            )
            if selected_page != current_page:
                st.session_state.import_page = selected_page - 1
                st.rerun()

        with nav_col3:
            if st.button("Next ➡️", disabled=current_page == total_pages, use_container_width=True):
                st.session_state.import_page += 1
                st.rerun()

        # Batch actions
        if len(st.session_state.selected_files) > 0:
            st.divider()
            st.markdown(f"**{len(st.session_state.selected_files)} file(s) selected**")

            batch_col1, batch_col2 = st.columns(2)

            with batch_col1:
                if st.button("🗑️ Delete Selected", type="primary", use_container_width=True):
                    deleted = 0
                    for file_path_str in list(st.session_state.selected_files):
                        try:
                            os.remove(file_path_str)
                            deleted += 1
                        except Exception as e:
                            st.error(f"Error deleting {Path(file_path_str).name}: {str(e)}")

                    st.session_state.selected_files.clear()
                    st.success(f"Deleted {deleted} file(s)")
                    time.sleep(0.5)
                    st.rerun()

            with batch_col2:
                if st.button("❌ Clear Selection", use_container_width=True):
                    st.session_state.selected_files.clear()
                    st.rerun()

    # Clear all confirmation modal
    if st.session_state.get("show_clear_confirm", False):
        st.divider()
        st.error("⚠️ **WARNING:** This will delete ALL files in the staging area. This action cannot be undone!")

        confirm_col1, confirm_col2 = st.columns(2)

        with confirm_col1:
            if st.button("✅ Yes, Delete Everything", type="primary", use_container_width=True):
                try:
                    deleted_count = 0
                    for item in RAW_FILES_DIR.iterdir():
                        try:
                            if item.is_file():
                                os.remove(item)
                                deleted_count += 1
                            elif item.is_dir():
                                shutil.rmtree(item)
                                deleted_count += 1
                        except Exception as e:
                            st.error(f"Failed to delete {item.name}: {str(e)}")
    
                    st.success(f"Deleted {deleted_count} items from staging area")
                    st.session_state.show_clear_confirm = False
                    st.session_state.selected_files.clear()
                    time.sleep(1)
                    st.rerun()
                except Exception as e:
                    st.error(f"Error clearing staging area: {str(e)}")

        with confirm_col2:
            if st.button("❌ Cancel", use_container_width=True):
                st.session_state.show_clear_confirm = False
                st.rerun()
