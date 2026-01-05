"""
Database Analytics Tab - Insights and health monitoring.

Features:
- Database health metrics
- Storage analysis and optimization suggestions
- Document distribution visualization
- Performance insights
- Export and backup utilities
- Database comparison
"""

import streamlit as st
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
from common import DATABASE_DIR, scan_databases, delete_database_instance


def get_database_details(db_path: Path) -> Dict:
    """
    Get detailed information about a database.

    Args:
        db_path: Path to database directory

    Returns:
        Dictionary with database details
    """
    if not db_path.exists():
        return {}

    details = {
        "name": db_path.name,
        "type": db_path.parent.name,
        "path": str(db_path),
        "size_mb": 0,
        "file_count": 0,
        "chunk_count": 0,
        "config": {},
        "processed_files": [],
        "created": None,
        "last_modified": None
    }

    # Get size
    try:
        total_size = sum(f.stat().st_size for f in db_path.rglob('*') if f.is_file())
        details["size_mb"] = total_size / (1024 * 1024)
    except:
        pass

    # Get config
    config_path = db_path / "db_config.json"
    if config_path.exists():
        try:
            with open(config_path, 'r') as f:
                details["config"] = json.load(f)
        except:
            pass

    # Get processed files
    processed_path = db_path / "processed_files.json"
    if processed_path.exists():
        try:
            with open(processed_path, 'r') as f:
                details["processed_files"] = json.load(f)
                details["file_count"] = len(details["processed_files"])
        except:
            pass

    # Get timestamps
    try:
        details["created"] = datetime.fromtimestamp(db_path.stat().st_ctime)
        details["last_modified"] = datetime.fromtimestamp(db_path.stat().st_mtime)
    except:
        pass

    # Estimate chunk count from ChromaDB if available
    chroma_db = db_path / "chroma_db"
    if chroma_db.exists():
        try:
            # Count files in chroma_db as rough estimate
            chunk_files = list(chroma_db.rglob('*'))
            details["chunk_count"] = len(chunk_files) * 100  # Rough estimate
        except:
            pass

    return details


def calculate_health_score(details: Dict) -> tuple[int, List[str]]:
    """
    Calculate database health score and issues.

    Args:
        details: Database details dictionary

    Returns:
        Tuple of (score 0-100, list of issues)
    """
    score = 100
    issues = []

    # Check if config exists
    if not details.get("config"):
        score -= 20
        issues.append("Missing configuration file")

    # Check if files were processed
    if details.get("file_count", 0) == 0:
        score -= 30
        issues.append("No files processed")

    # Check size
    if details.get("size_mb", 0) < 0.1:
        score -= 20
        issues.append("Database appears empty or corrupted")

    # Check config validity
    config = details.get("config", {})
    chunk_size = config.get("chunk_size", 0)
    chunk_overlap = config.get("chunk_overlap", 0)

    if chunk_size == 0:
        score -= 15
        issues.append("Invalid chunk size configuration")

    if chunk_overlap >= chunk_size:
        score -= 10
        issues.append("Overlap exceeds chunk size")

    # Check last modified (staleness)
    last_modified = details.get("last_modified")
    if last_modified:
        days_old = (datetime.now() - last_modified).days
        if days_old > 90:
            score -= 5
            issues.append(f"Database hasn't been updated in {days_old} days")

    return max(0, score), issues


def render_analytics_tab():
    """Render the analytics tab."""

    st.markdown("### Database Analytics & Management")
    st.caption("Monitor database health, analyze storage, and manage your knowledge bases")

    st.divider()

    # Get all databases
    db_inventory = scan_databases(DATABASE_DIR)

    if not db_inventory:
        st.info("📊 No databases found. Build a database to see analytics here.")
        return

    # Database selector
    st.markdown("#### Select Database")

    select_col1, select_col2 = st.columns([3, 1])

    with select_col1:
        db_options = [f"{db['Type'].upper()} | {db['Name']}" for db in db_inventory]
        selected_key = st.selectbox(
            "Choose a database to analyze",
            db_options,
            label_visibility="collapsed"
        )

    with select_col2:
        if st.button("🔄 Refresh", use_container_width=True):
            st.rerun()

    # Find selected database
    selected_db = None
    for db in db_inventory:
        if f"{db['Type'].upper()} | {db['Name']}" == selected_key:
            selected_db = db
            break

    if not selected_db:
        return

    st.divider()

    # Get detailed info
    db_path = Path(selected_db["Path"])
    details = get_database_details(db_path)
    health_score, issues = calculate_health_score(details)

    # === HEALTH OVERVIEW ===
    st.markdown("#### Health Overview")

    health_col1, health_col2, health_col3, health_col4 = st.columns(4)

    with health_col1:
        # Health score with color
        if health_score >= 80:
            score_color = "#10b981"
            score_status = "Excellent"
        elif health_score >= 60:
            score_color = "#f59e0b"
            score_status = "Good"
        else:
            score_color = "#ef4444"
            score_status = "Needs Attention"

        st.markdown(f"""
        <div style="background: #1a1d29; padding: 1.5rem; border-radius: 12px; border: 2px solid {score_color}; text-align: center;">
            <div style="color: #9ca3af; font-size: 0.875rem; margin-bottom: 0.5rem;">HEALTH SCORE</div>
            <div style="font-size: 3rem; font-weight: 700; color: {score_color}; line-height: 1;">{health_score}</div>
            <div style="color: {score_color}; font-size: 0.875rem; margin-top: 0.5rem;">{score_status}</div>
        </div>
        """, unsafe_allow_html=True)

    with health_col2:
        st.metric("Total Files", f"{details['file_count']:,}")

    with health_col3:
        st.metric("Storage Size", f"{details['size_mb']:.1f} MB")

    with health_col4:
        st.metric("Database Type", details['type'].upper())

    # Issues
    if issues:
        st.divider()
        st.markdown("#### Issues Detected")

        for issue in issues:
            st.warning(f"⚠️ {issue}")

    st.divider()

    # === CONFIGURATION ===
    st.markdown("#### Configuration")

    config_col1, config_col2 = st.columns(2)

    with config_col1:
        config = details.get("config", {})

        if config:
            st.markdown("""
            <div style="background: #1a1d29; padding: 1.5rem; border-radius: 12px; border: 1px solid #2d3142;">
            """, unsafe_allow_html=True)

            st.markdown(f"""
            **Chunking Settings**
            - Chunk Size: `{config.get('chunk_size', 'N/A')}`
            - Chunk Overlap: `{config.get('chunk_overlap', 'N/A')}`
            - Embedding Model: `{config.get('embedding_model', 'N/A')}`
            """)

            if details['type'] == 'lightrag':
                st.markdown(f"""
                **LightRAG Settings**
                - LLM Model: `{config.get('llm_model', 'N/A')}`
                - Graph Enabled: `{config.get('use_graph', False)}`
                """)

            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.warning("Configuration file not found")

    with config_col2:
        st.markdown("""
        <div style="background: #1a1d29; padding: 1.5rem; border-radius: 12px; border: 1px solid #2d3142;">
        """, unsafe_allow_html=True)

        st.markdown("**Metadata**")

        if details.get("created"):
            st.markdown(f"- Created: `{details['created'].strftime('%Y-%m-%d %H:%M')}`")

        if details.get("last_modified"):
            st.markdown(f"- Last Modified: `{details['last_modified'].strftime('%Y-%m-%d %H:%M')}`")

        st.markdown(f"- Path: `{db_path.name}`")

        st.markdown("</div>", unsafe_allow_html=True)

    st.divider()

    # === FILE ANALYSIS ===
    st.markdown("#### File Analysis")

    if details.get("processed_files"):
        files = details["processed_files"]

        # File type distribution
        file_types = {}
        for file_path in files:
            ext = Path(file_path).suffix.lower() or 'no_extension'
            file_types[ext] = file_types.get(ext, 0) + 1

        # Display as chart
        chart_col1, chart_col2 = st.columns([2, 1])

        with chart_col1:
            st.markdown("**File Type Distribution**")

            for ext, count in sorted(file_types.items(), key=lambda x: x[1], reverse=True):
                percentage = (count / len(files)) * 100

                st.markdown(f"""
                <div style="margin-bottom: 0.75rem;">
                    <div style="display: flex; justify-content: space-between; margin-bottom: 0.25rem;">
                        <span><code>{ext}</code></span>
                        <span style="color: #9ca3af;">{count} files ({percentage:.1f}%)</span>
                    </div>
                    <div style="background: #0f1117; border-radius: 4px; height: 10px; overflow: hidden;">
                        <div style="background: linear-gradient(90deg, #6366f1, #8b5cf6); width: {percentage}%; height: 100%;"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        with chart_col2:
            st.markdown("**Quick Stats**")

            st.metric("Total Files", len(files))
            st.metric("File Types", len(file_types))

            avg_per_type = len(files) / len(file_types) if file_types else 0
            st.metric("Avg per Type", f"{avg_per_type:.1f}")

        # File list
        with st.expander("📄 View All Files", expanded=False):
            file_display = sorted(files)[:100]  # Limit to 100 for performance

            for file_path in file_display:
                st.text(f"• {Path(file_path).name}")

            if len(files) > 100:
                st.caption(f"...and {len(files) - 100} more files")

    else:
        st.info("No processed files information available")

    st.divider()

    # === STORAGE OPTIMIZATION ===
    st.markdown("#### Storage Insights")

    # Calculate storage breakdown
    storage_breakdown = {}

    try:
        for item in db_path.iterdir():
            if item.is_dir():
                size = sum(f.stat().st_size for f in item.rglob('*') if f.is_file())
                storage_breakdown[item.name] = size / (1024 * 1024)
            elif item.is_file():
                storage_breakdown[item.name] = item.stat().st_size / (1024 * 1024)
    except:
        pass

    if storage_breakdown:
        total_storage = sum(storage_breakdown.values())

        for name, size_mb in sorted(storage_breakdown.items(), key=lambda x: x[1], reverse=True)[:5]:
            percentage = (size_mb / total_storage * 100) if total_storage > 0 else 0

            st.markdown(f"""
            <div style="margin-bottom: 0.75rem;">
                <div style="display: flex; justify-content: space-between; margin-bottom: 0.25rem;">
                    <span><code>{name}</code></span>
                    <span style="color: #9ca3af;">{size_mb:.1f} MB ({percentage:.1f}%)</span>
                </div>
                <div style="background: #0f1117; border-radius: 4px; height: 8px; overflow: hidden;">
                    <div style="background: linear-gradient(90deg, #6366f1, #8b5cf6); width: {percentage}%; height: 100%;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.divider()

    # === MANAGEMENT ACTIONS ===
    st.markdown("#### Management Actions")

    action_col1, action_col2, action_col3 = st.columns(3)

    with action_col1:
        if st.button("📊 Export Config", use_container_width=True):
            if details.get("config"):
                config_json = json.dumps(details["config"], indent=2)
                st.download_button(
                    "⬇️ Download JSON",
                    config_json,
                    file_name=f"{details['name']}_config.json",
                    mime="application/json",
                    use_container_width=True
                )
            else:
                st.warning("No configuration to export")

    with action_col2:
        if st.button("📋 Copy Path", use_container_width=True):
            st.code(str(db_path), language="bash")
            st.success("Path displayed above")

    with action_col3:
        if st.button("🗑️ Delete Database", type="primary", use_container_width=True):
            st.session_state.confirm_delete_analytics = True

    # Delete confirmation
    if st.session_state.get("confirm_delete_analytics", False):
        st.divider()
        st.error(f"⚠️ **WARNING:** Delete database '{details['name']}'? This cannot be undone!")

        confirm_col1, confirm_col2 = st.columns(2)

        with confirm_col1:
            if st.button("✅ Yes, Delete", type="primary", use_container_width=True):
                if delete_database_instance(str(db_path)):
                    st.success("Database deleted successfully")
                    st.session_state.confirm_delete_analytics = False
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error("Failed to delete database")

        with confirm_col2:
            if st.button("❌ Cancel", use_container_width=True):
                st.session_state.confirm_delete_analytics = False
                st.rerun()

    st.divider()

    # === RECOMMENDATIONS ===
    st.markdown("#### Recommendations")

    recommendations = []

    # Size-based recommendations
    if details['size_mb'] > 1000:
        recommendations.append({
            "type": "info",
            "title": "Large Database",
            "message": "Consider splitting into multiple smaller databases for better performance"
        })

    # Config-based recommendations
    config = details.get("config", {})
    chunk_size = config.get("chunk_size", 512)

    if chunk_size < 256:
        recommendations.append({
            "type": "warning",
            "title": "Small Chunk Size",
            "message": "Very small chunks may reduce context quality. Consider 512-1024 for better results"
        })
    elif chunk_size > 1500:
        recommendations.append({
            "type": "warning",
            "title": "Large Chunk Size",
            "message": "Large chunks may reduce retrieval precision. Consider 512-1024 for balanced performance"
        })

    # Health-based recommendations
    if health_score < 80:
        recommendations.append({
            "type": "error",
            "title": "Health Issues",
            "message": "Database has issues that should be addressed. Consider rebuilding or reviewing logs"
        })

    # Display recommendations
    if recommendations:
        for rec in recommendations:
            if rec["type"] == "error":
                st.error(f"**{rec['title']}**: {rec['message']}")
            elif rec["type"] == "warning":
                st.warning(f"**{rec['title']}**: {rec['message']}")
            else:
                st.info(f"**{rec['title']}**: {rec['message']}")
    else:
        st.success("✅ No issues detected. Database is healthy!")
