import sys
import os
import json
from pathlib import Path

# Add 'frontend' to sys.path
frontend_dir = os.path.join(os.getcwd(), 'frontend')
if frontend_dir not in sys.path:
    sys.path.insert(0, frontend_dir)

try:
    from query.data_service import data_service
    from query.database_paths import DATABASE_DIR
except ImportError as e:
    print(f"❌ CRITICAL IMPORT ERROR: {e}")
    sys.exit(1)

# --- 2. DIAGNOSTIC UTILITIES ---
def scan_available_dbs():
    """Finds all databases in volumes/databases."""
    found = []
    if not DATABASE_DIR.exists():
        return []

    for rag_type in ['rag', 'lightrag', 'kag']:
        type_dir = DATABASE_DIR / rag_type
        if type_dir.exists():
            for item in type_dir.iterdir():
                if item.is_dir() and not item.name.startswith('.'):
                    found.append((rag_type, item.name, item))
    return found

def grade_chunk(text):
    """
    Grades a chunk to see if it suffers from specific issues.
    Returns: (Grade, Message)
    """
    if not text or not text.strip():
        return "🔴", "EMPTY/NULL CONTENT"
    
    clean_text = text.strip()
    length = len(clean_text)

    # CHECK 1: The "Orphan Header" Bug
    # Starts with hash, short length, very few newlines (indicates it didn't capture the paragraph)
    if clean_text.startswith('#') and length < 100 and clean_text.count('\n') < 2:
        return "🟡", f"ORPHAN HEADER (Len: {length})"

    # CHECK 2: The "Noise" Bug
    if length < 50:
        return "🟠", f"TOO SHORT (Len: {length})"

    # CHECK 3: Good Chunk
    # Should ideally have a header AND body text, or just body text
    return "🟢", f"Healthy (Len: {length})"

def analyze_db(rag_type, db_name, db_path):
    print(f"\n{'='*60}")
    print(f"🔎 DIAGNOSING: [{rag_type.upper()}] {db_name}")
    print(f"   Path: {db_path}")

    # 1. Check Config
    config_path = db_path / "db_config.json"
    if config_path.exists():
        try:
            with open(config_path, 'r') as f:
                cfg = json.load(f)
                print(f"   ⚙️  Config: Chunk Size={cfg.get('chunk_size')}, Overlap={cfg.get('chunk_overlap')}")
        except:
            print("   ⚠️  Config file corrupt.")
    else:
        print("   ⚠️  No db_config.json found.")

    # 2. Connect via DataService
    try:
        db = data_service.get_chroma_db(rag_type, db_name)
        if not db:
            print("   ❌ FAILED to load ChromaDB instance.")
            return
        
        count = db._collection.count()
        print(f"   📊 Total Documents: {count}")
        
        if count == 0:
            print("   ⚠️  Database is EMPTY. Population failed or hasn't run.")
            return

        # 3. Analyze Content
        # Fetch first 10 docs to check for "Headeritis"
        # Fetch 5 docs from an offset (if supported) or just first 10
        results = db.get(limit=10, include=['metadatas', 'documents'])
        docs = results.get('documents', [])
        metas = results.get('metadatas', [])
        ids = results.get('ids', [])

        print(f"\n   --- 🧪 SAMPLING FIRST {len(docs)} CHUNKS ---")
        
        issues_found = 0
        
        for i, (doc_id, content, meta) in enumerate(zip(ids, docs, metas)):
            grade, msg = grade_chunk(content)
            
            # Formatting for display
            display_content = content.replace('\n', ' ↩ ') if content else "None"
            if len(display_content) > 85:
                display_content = display_content[:85] + "..."
            
            source_file = Path(meta.get('source', 'unknown')).name
            
            print(f"   {grade} [{i}] {msg}")
            print(f"       Source: {source_file}")
            print(f"       Preview: \"{display_content}\"")
            
            if grade != "🟢":
                issues_found += 1

        print(f"\n   --- 🩺 DIAGNOSIS ---")
        if issues_found == 0:
            print("   ✅ PASSED: Sample chunks look healthy.")
            print("      - Content is present.")
            print("      - No orphan headers detected in sample.")
            print("      - Separator fix appears successful.")
        elif issues_found == len(docs):
            print("   ❌ FAILED: All sampled chunks are problematic.")
            print("      - Verify if 'processing_utils.py' filters are active.")
            print("      - Verify if 'RecursiveCharacterTextSplitter' separators were updated.")
        else:
            print(f"   ⚠️  MIXED: {issues_found}/{len(docs)} chunks have potential issues.")
            print("      - Some noise is expected, but ensure headers are attached to bodies.")

    except Exception as e:
        print(f"   ❌ CRITICAL ERROR during inspection: {e}")

# --- 3. MAIN EXECUTION ---
if __name__ == "__main__":
    print("🏥 STARTING DATABASE DIAGNOSTICS...")
    
    dbs = scan_available_dbs()
    
    if not dbs:
        print("❌ No databases found in 'volumes/databases'.")
    else:
        print(f"Found {len(dbs)} databases.")
        for r_type, name, path in dbs:
            analyze_db(r_type, name, path)
            
    print("\n🏁 DIAGNOSTICS COMPLETE")