"""
faiss_ivf.py
------------
FAISS IVF (Inverted File Index) — comparison algorithm.

This is the partition-based approach that Ada-IVF improves upon.
We compare it against our Clustered HNSW approach.

Usage:
  python3 faiss_ivf.py
  python3 faiss_ivf.py --dataset ../data/sift-128-euclidean.hdf5

Install:
  pip3 install faiss-cpu h5py numpy
"""

import numpy as np
import time
import argparse
import os

try:
    import faiss
except ImportError:
    print("Install faiss: pip3 install faiss-cpu")
    exit(1)

try:
    import h5py
    HDF5_AVAILABLE = True
except ImportError:
    HDF5_AVAILABLE = False


# ─── CONFIG ──────────────────────────────────────────────────────────────────
N_CLUSTERS   = 1000    # number of IVF partitions (same as our K)
N_PROBE_LIST = [1, 5, 10, 20, 50, 100]  # nprobe sweep (like our TOP_CLUSTERS)
K_RESULTS    = 10      # number of nearest neighbors
# ─────────────────────────────────────────────────────────────────────────────


def load_hdf5(path):
    with h5py.File(path, 'r') as f:
        train = f['train'][:]
        test  = f['test'][:]
        gt    = f['neighbors'][:]
    return train.astype(np.float32), test.astype(np.float32), gt


def load_fvecs(base_path, query_path, gt_path):
    """Load SIFT .fvecs format (fallback if no HDF5)"""
    import struct

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

    return read_fvecs(base_path), read_fvecs(query_path), read_ivecs(gt_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default=None, help='Path to HDF5 dataset')
    args = parser.parse_args()

    print("=" * 55)
    print("  FAISS IVF Benchmark")
    print("  Comparison algorithm for Clustered HNSW paper")
    print("=" * 55)

    # Load data
    if args.dataset and HDF5_AVAILABLE and os.path.exists(args.dataset):
        print(f"\nLoading {args.dataset}...")
        train, test, gt = load_hdf5(args.dataset)
    else:
        print("\nLoading SIFT1M from .fvecs...")
        sift_dir = "../../experiments/sift"
        train, test, gt = load_fvecs(
            f"{sift_dir}/sift_base.fvecs",
            f"{sift_dir}/sift_query.fvecs",
            f"{sift_dir}/sift_groundtruth.ivecs"
        )

    n, d = train.shape
    print(f"Train: {n:,} × {d}  |  Test: {len(test):,}")

    # Build IVF index
    print(f"\nBuilding FAISS IVF (n_clusters={N_CLUSTERS})...")
    quantizer = faiss.IndexFlatL2(d)
    index = faiss.IndexIVFFlat(quantizer, d, N_CLUSTERS, faiss.METRIC_L2)

    t0 = time.time()
    index.train(train)
    index.add(train)
    build_time = time.time() - t0
    print(f"Build time: {build_time:.2f}s")

    # Search sweep over nprobe values
    print(f"\n{'nprobe':<10} {'Recall@1':<12} {'QPS':<12} {'Time(s)':<10}")
    print("-" * 44)

    results = []
    for nprobe in N_PROBE_LIST:
        index.nprobe = nprobe
        correct = 0

        t1 = time.time()
        D, I = index.search(test, K_RESULTS)
        search_time = time.time() - t1

        for i in range(len(test)):
            if gt[i][0] in I[i]:
                correct += 1

        recall = correct / len(test)
        qps = len(test) / search_time
        results.append((nprobe, recall, qps, search_time))

        print(f"{nprobe:<10} {recall:<12.4f} {qps:<12.1f} {search_time:<10.3f}")

    print("-" * 44)
    print(f"\nBuild time: {build_time:.2f}s")
    print(f"n_clusters: {N_CLUSTERS}")
    print(f"\nNote: nprobe in FAISS IVF ≈ TOP_CLUSTERS in our Clustered HNSW")
    print(f"Both control how many partitions are searched per query")

    # Save results
    import csv
    with open('../results.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        import socket
        from datetime import datetime
        ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        machine = socket.gethostname()
        for nprobe, recall, qps, _ in results:
            writer.writerow([ts, machine, 1, f'faiss_ivf_nprobe{nprobe}', recall, qps, build_time])
    print("\nResults appended to ../results.csv")


if __name__ == "__main__":
    main()
