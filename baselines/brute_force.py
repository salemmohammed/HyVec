"""
brute_force.py
--------------
Exact nearest neighbor baseline using FAISS IndexFlatL2.

This script loads an ANN-Benchmarks HDF5 dataset, runs exact Euclidean
nearest neighbor search using FAISS, computes Recall@1 and Recall@k against
the provided ground truth, and appends results to ../results.csv.

Usage:
  python3 brute_force.py --dataset ../data/sift-128-euclidean.hdf5 --k 10
"""

import argparse
import csv
import os
import socket
import time
from datetime import datetime

import faiss
import h5py
import numpy as np


def read_hdf5(path):
    """Load train vectors, query vectors, and ground-truth neighbors."""
    with h5py.File(path, "r") as f:
        train = np.asarray(f["train"], dtype=np.float32)
        test = np.asarray(f["test"], dtype=np.float32)
        neighbors = np.asarray(f["neighbors"], dtype=np.int32)

    return train, test, neighbors


def compute_recall_at_k(predicted, ground_truth, k):
    """Compute mean Recall@k."""
    total = 0.0

    for i in range(predicted.shape[0]):
        retrieved = set(predicted[i, :k])
        truth = set(ground_truth[i, :k])
        total += len(retrieved & truth) / k

    return total / predicted.shape[0]


def save_results(row, output_path="../results.csv"):
    """Append benchmark result to CSV."""
    file_exists = os.path.exists(output_path)

    with open(output_path, "a", newline="") as f:
        writer = csv.writer(f)

        if not file_exists:
            writer.writerow([
                "timestamp",
                "machine",
                "algorithm",
                "dataset",
                "distance",
                "k",
                "recall_at_1",
                "recall_at_k",
                "qps",
                "avg_latency_ms",
                "build_time_s",
                "search_time_s",
                "index_size_mb",
                "num_base",
                "num_queries",
                "dimension",
            ])

        writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(
        description="FAISS brute-force exact search baseline using IndexFlatL2"
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Path to ANN-Benchmarks HDF5 dataset",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=10,
        help="Number of nearest neighbors to retrieve",
    )
    args = parser.parse_args()

    print("=" * 72)
    print("  FAISS Brute Force Exact Search — IndexFlatL2")
    print("  Metric: L2 / Euclidean")
    print(f"  Dataset: {args.dataset}")
    print(f"  k: {args.k}")
    print("=" * 72)

    print("\nLoading dataset...")
    base, queries, ground_truth = read_hdf5(args.dataset)

    num_base, dimension = base.shape
    num_queries = queries.shape[0]

    print(f"  Base vectors : {num_base:,} x {dimension}")
    print(f"  Query vectors: {num_queries:,}")
    print(f"  Ground truth : top-{ground_truth.shape[1]} neighbors per query")

    print("\nBuilding FAISS IndexFlatL2...")
    build_start = time.time()

    index = faiss.IndexFlatL2(dimension)
    index.add(base)

    build_time = time.time() - build_start

    print("Running exact search...")
    search_start = time.time()

    _, predicted = index.search(queries, args.k)

    search_time = time.time() - search_start

    recall_at_1 = compute_recall_at_k(predicted, ground_truth, 1)
    recall_at_k = compute_recall_at_k(predicted, ground_truth, args.k)

    qps = num_queries / search_time
    avg_latency_ms = (search_time / num_queries) * 1000.0
    index_size_mb = (num_base * dimension * 4) / (1024 * 1024)

    print("\n" + "=" * 72)
    print("  RESULTS")
    print("=" * 72)
    print("  Algorithm      : faiss_flat_l2")
    print("  Distance       : L2 / Euclidean")
    print(f"  Recall@1       : {recall_at_1:.4f}")
    print(f"  Recall@{args.k:<7}: {recall_at_k:.5f}")
    print(f"  QPS            : {qps:.2f} queries/second")
    print(f"  Avg latency    : {avg_latency_ms:.6f} ms/query")
    print(f"  Build time     : {build_time:.6f} s")
    print(f"  Search time    : {search_time:.6f} s")
    print(f"  Index size     : {index_size_mb:.3f} MB")
    print(f"  Base vectors   : {num_base}")
    print(f"  Query vectors  : {num_queries}")
    print(f"  Dimension      : {dimension}")
    print("=" * 72)

    if recall_at_1 < 0.999:
        print("\nNOTE: Recall@1 is below 1.0 compared with provided ground truth.")
        print("This can occur due to tie handling or ground-truth generation differences.")

    row = [
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        socket.gethostname(),
        "faiss_flat_l2",
        os.path.basename(args.dataset),
        "l2",
        args.k,
        f"{recall_at_1:.6f}",
        f"{recall_at_k:.6f}",
        f"{qps:.6f}",
        f"{avg_latency_ms:.6f}",
        f"{build_time:.6f}",
        f"{search_time:.6f}",
        f"{index_size_mb:.6f}",
        num_base,
        num_queries,
        dimension,
    ]

    save_results(row)

    print("\nResults saved to ../results.csv")


if __name__ == "__main__":
    main()