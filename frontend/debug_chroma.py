import sys
import os

# 1. SETUP PATHS
# Ensure we are running from root by checking for 'frontend' folder
current_path = os.getcwd()
if not os.path.exists(os.path.join(current_path, 'frontend')):
    print(f"⚠️ Warning: It looks like you aren't in the root directory. Current: {current_path}")
    # Try to find root if we are inside frontend
    if os.path.basename(current_path) == 'frontend':
        os.chdir('..')
        print(f"✅ Moved up to root: {os.getcwd()}")

# Add 'frontend' to sys.path so we can import 'query'
frontend_dir = os.path.join(os.getcwd(), 'frontend')
if frontend_dir not in sys.path:
    sys.path.insert(0, frontend_dir)

try:
    from query.data_service import data_service
except ImportError as e:
    print(f"❌ CRITICAL IMPORT ERROR: {e}")
    print("Run this script from the project root containing 'frontend' and 'volumes'.")
    sys.exit(1)

def inspect_db(rag_type, db_name):
    print(f"\n--------------------------------------------------")
    print(f"🔍 INSPECTING: {rag_type} / {db_name}")
    
    # 2. PHYSICAL PATH CHECK
    # Based on your file tree: volumes/databases/lightrag/...
    expected_path = os.path.join("volumes", "databases", rag_type, db_name)
    abs_path = os.path.abspath(expected_path)
    
    if os.path.exists(abs_path):
        print(f"   📁 Folder Found: {abs_path}")
    else:
        print(f"   ❌ Folder NOT Found: {abs_path}")
        return

    # 3. LOAD DB VIA SERVICE
    try:
        db = data_service.get_chroma_db(rag_type, db_name)
        if not db:
            print("   ❌ data_service returned None (Check logs for path errors)")
            return
            
        # 4. CHECK CONTENTS
        count = db._collection.count()
        print(f"   📊 Document Count: {count}")
        
        if count == 0:
            print("   ⚠️ Database is empty.")
            return

        # Peek at data
        results = db.get(limit=3, include=['metadatas', 'documents'])
        docs = results.get('documents', [])
        metas = results.get('metadatas', [])
        
        for i, (doc, meta) in enumerate(zip(docs, metas)):
            status = "✅ TEXT PRESENT" if doc and doc.strip() else "❌ TEXT MISSING"
            preview = doc[:50].replace('\n', ' ') if doc else "None"
            print(f"   - Doc {i}: {status} | Len: {len(doc) if doc else 0}")
            print(f"     Meta: {meta}")

    except Exception as e:
        print(f"   ❌ Error accessing Chroma: {e}")

if __name__ == "__main__":
    # Test 1: The one you were trying (Underscore)
    inspect_db('lightrag', 'red_team')

    # Test 2: The one visible in your file tree (Hyphen)
    # Your tree shows 'red-team' has the data files (data_level0.bin)
    inspect_db('lightrag', 'red-team')