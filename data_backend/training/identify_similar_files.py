import os
import sys
import hashlib
import multiprocessing
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

def get_signature(file_path, num_hashes=128):
    """Generates a MinHash signature for a file, skipping first 3 lines."""
    try:
        # Resolve Windows long paths and encoding
        p = Path(file_path).resolve()
        with p.open('r', encoding='utf-8', errors='ignore') as f:
            for _ in range(3): f.readline()
            content = f.read().split()
        
        if not content: return None
        
        # 3-word shingles for structural matching
        shingles = set(" ".join(content[i:i+3]) for i in range(len(content)-2))
        if not shingles: return None

        signature = []
        for i in range(num_hashes):
            min_hash = float('inf')
            for s in shingles:
                # MurmurHash-style salt for unique hash functions
                h = int(hashlib.md5(f"{i}{s}".encode()).hexdigest(), 16)
                if h < min_hash: min_hash = h
            signature.append(min_hash)
        return (p.name, signature)
    except Exception:
        return None

def main():
    target_path = sys.argv[1] if len(sys.argv) > 1 else "."
    threshold = float(sys.argv[2]) if len(sys.argv) > 2 else 0.9
    
    # --- CPU Intelligence ---
    logical_cores = os.cpu_count() or 1
    # Leave 1 core free to prevent system-wide lag
    workers = max(1, logical_cores - 1)
    
    target_dir = Path(target_path).resolve()
    files = list(target_dir.glob("*.txt"))
    
    print(f"System: {logical_cores} cores detected. Using {workers} workers.")
    print(f"Directory: {target_dir}")
    print(f"Files found: {len(files)}")

    if len(files) < 2:
        return print("Not enough files to compare.")

    # Phase 1: Parallel Hashing with Progress Bar
    signatures = {}
    print("Hashing Files...")
    with multiprocessing.Pool(processes=workers) as pool:
        for result in tqdm(pool.imap_unordered(get_signature, files), total=len(files), desc="Signatures"):
            if result:
                name, sig = result
                signatures[name] = sig

    # Phase 2: LSH Bucketing (Optimization for 10k files)
    num_hashes = 128
    bands = 16 # Adjusting bands/rows changes sensitivity
    rows = 8   # bands * rows must = num_hashes
    
    buckets = defaultdict(set)
    for name, sig in signatures.items():
        for b in range(bands):
            band = tuple(sig[b*rows : (b+1)*rows])
            buckets[hash((b, band))].add(name)

    # Extract candidates from buckets
    candidates = set()
    for cluster in buckets.values():
        if len(cluster) > 1:
            items = sorted(list(cluster))
            for i in range(len(items)):
                for j in range(i + 1, len(items)):
                    candidates.add((items[i], items[j]))

    # Phase 3: Final Similarity Verification
    print(f"Comparing {len(candidates)} high-probability candidates...")
    results_found = []
    for f1, f2 in tqdm(candidates, desc="Comparing"):
        sig1, sig2 = signatures[f1], signatures[f2]
        sim = sum(1 for a, b in zip(sig1, sig2) if a == b) / num_hashes
        if sim >= threshold:
            results_found.append(f"{f1} and {f2} are {sim:.1%} similar")

    print("\n" + "\n".join(results_found))
    print(f"\nTotal pairs found: {len(results_found)}")

if __name__ == "__main__":
    main()