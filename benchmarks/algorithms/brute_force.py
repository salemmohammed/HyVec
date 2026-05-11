"""
brute_force.py
--------------
Brute force exact search — the recall=1.0 reference point.

Every ANN algorithm is measured against this.
Recall=1.0 means you found ALL true nearest neighbors.
Lower QPS than any ANN method, but always correct.

Usage:
  python3 brute_force.py
"""

import numpy as np
import time
import struct
import csv
import socket
from datetime import datetime


def read_fvecs(filename):
    with open(filename, 'rb') as f:
        dim = struct.unpack('i', f.read(4))[0]
        f.seek(0)
        data = np.fromfile(f, dtype=np.float32)
    return data.reshape(-1, dim + 1)[:, 1:].copy()

def read_ivecs(filename):
    with open(filename, 'rb') as f:
        dim = struct.unpack('i', f.read(4))[0]
        f.seek(0)
        data = np.fromfile(f, dtype=np.int32)
    return data.reshape(-1, dim + 1)[:, 1:].copy()


def main():
    print("=" * 50)
    print("  Brute Force Exact Search (Recall = 1.0)")
    print("=" * 50)

    print("\nLoading SIFT1M...")
    base    = read_fvecs("../data/sift/sift_base.fvecs")
    queries = read_fvecs("../data/sift/sift_query.fvecs")
    gt      = read_ivecs("../data/sift/sift_groundtruth.ivecs")

    n, d = base.shape
    K = 10
    num_queries = min(1000, len(queries))  # limit for speed
    print(f"Base: {n:,} × {d}  |  Queries: {num_queries}")

    print(f"\nSearching {num_queries} queries (exact)...")
    correct = 0
    t0 = time.time()

    for i in range(num_queries):
        # Compute L2 distance to all base vectors
        diffs = base - queries[i]
        dists = (diffs * diffs).sum(axis=1)
        top_k = np.argpartition(dists, K)[:K]

        if gt[i][0] in top_k:
            correct += 1

    search_time = time.time() - t0
    recall = correct / num_queries
    qps = num_queries / search_time

    print(f"\n=== RESULTS ===")
    print(f"Recall@1:    {recall:.4f}  (should be ~1.0)")
    print(f"QPS:         {qps:.1f}")
    print(f"Search time: {search_time:.2f}s")
    print(f"Note: Only ran {num_queries} queries (full 10K would take ~{search_time*10:.0f}s)")

    # Save
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    machine = socket.gethostname()
    with open('../results.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([ts, machine, 1, 'brute_force', recall, qps, 0])
    print("\nResults appended to ../results.csv")


if __name__ == "__main__":
    main()
