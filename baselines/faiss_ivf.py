"""
faiss_ivf.py
------------
FAISS IVF (Inverted File Index) — partition-based comparison baseline.

This script benchmarks FAISS IVF against our Clustered HNSW system.
It is the most direct comparison point: both systems partition the dataset
into K clusters using k-means, then route each query to the nearest
clusters. The key difference is what happens inside each partition:

    FAISS IVF:       partition → flat exact L2 scan inside each cell
    Clustered HNSW:  partition → HNSW graph search inside each cluster

By sweeping nprobe (number of clusters searched per query), we produce
the full Recall vs QPS curve for FAISS IVF. This curve appears as one
of the baseline lines in Figure 6 of the paper.

-------------------------------------------------------------------------
Algorithm: Inverted File Index (IVF)

Step 1 — Build (offline):
  1. Run k-means on the full dataset to produce K centroids.
  2. Assign each vector to its nearest centroid (Voronoi cell).
  3. Store each cell as a posting list of vector indices.

Step 2 — Query (online):
  1. Compute distance from query q to all K centroids: O(K × d)
  2. Select the nprobe nearest centroids.
  3. For each selected cell, scan all vectors with exact L2: O(nprobe × N/K × d)
  4. Return the k nearest vectors found across all searched cells.

The nprobe parameter controls the recall/speed tradeoff:
  nprobe = 1    → fastest, lowest recall  (search 1 cluster only)
  nprobe = K    → exact search (equivalent to brute force)
  nprobe = 20   → typical operating point (~0.97 recall on SIFT1M)

This is directly analogous to TOP_CLUSTERS in our Clustered HNSW:
  FAISS nprobe       ↔  our TOP_CLUSTERS
  FAISS nlist (K)    ↔  our K (number of clusters)

-------------------------------------------------------------------------
Complexity:

  Build : O(N × K × d)                   k-means assignment
  Query : O(K × d) + O(nprobe × N/K × d) centroid scan + flat search
  Memory: O(N × d)                        raw float32 vectors

  For SIFT1M: N=1M, K=1000, d=128, nprobe=20
    → centroid scan : 1000 × 128 = 128K ops
    → flat search   : 20 × 1000 × 128 = 2.56M ops per query

-------------------------------------------------------------------------
Relationship to our work:

  FAISS IVF was introduced by Jégou et al. [2] and is implemented in
  the FAISS library by Johnson et al. [1]. It is the standard partition-
  based baseline in the ANN literature and appears in all major ANN
  benchmarks [3, 5]. Our Clustered HNSW replaces the flat posting list
  with a dedicated HNSW graph per cluster [4], reducing within-cluster
  search from O(n) to O(log n) at the cost of higher build time and
  memory per cluster.

-------------------------------------------------------------------------
Install:
  pip3 install faiss-cpu h5py numpy

Usage:
  python3 faiss_ivf.py
  python3 faiss_ivf.py --dataset ../data/sift-128-euclidean.hdf5
  python3 faiss_ivf.py --dataset ../data/sift-128-euclidean.hdf5 --k 10
  python3 faiss_ivf.py --n-clusters 1000 --k 10

-------------------------------------------------------------------------
References:

  [1] J. Johnson, M. Douze, H. Jégou, "Billion-Scale Similarity Search
      with GPUs," IEEE Transactions on Big Data, 7(3): 535-547, 2021.
      https://arxiv.org/abs/1702.08734
      → FAISS library — IndexIVFFlat implementation

  [2] H. Jégou, M. Douze, C. Schmid, "Product Quantization for Nearest
      Neighbor Search," IEEE TPAMI, 33(1): 117-128, 2011.
      https://doi.org/10.1109/TPAMI.2010.57
      → Original IVF algorithm and SIFT1M dataset

  [3] M. Aumüller, E. Bernhardsson, A. Faithfull, "ANN-Benchmarks: A
      Benchmarking Tool for Approximate Nearest Neighbor Algorithms,"
      Information Systems, 87, 2020.
      https://doi.org/10.1016/j.is.2019.02.006
      → Evaluation protocol, Recall@k definition, HDF5 dataset format

  [4] Y. A. Malkov, D. A. Yashunin, "Efficient and Robust Approximate
      Nearest Neighbor Search Using Hierarchical Navigable Small World
      Graphs," IEEE TPAMI, 42(4): 824-836, 2020.
      https://arxiv.org/abs/1603.09320
      → HNSW — the graph index used inside each cluster in our system

  [5] M. Wang, X. Xu, Q. Yue, Y. Wang, "A Comprehensive Survey and
      Experimental Comparison of Graph-Based Approximate Nearest Neighbor
      Search," PVLDB, 14(11): 1964-1978, 2021.
      https://www.vldb.org/pvldb/vol14/p1964-wang.pdf
      → Comprehensive comparison including IVF variants

  [6] S. P. Lloyd, "Least Squares Quantization in PCM," IEEE TIT,
      28(2): 129-137, 1982. https://doi.org/10.1109/TIT.1982.1056489
      → K-means algorithm used for centroid computation in IVF build phase
"""

import argparse
import csv
import os
import socket
import struct
import time
from datetime import datetime

import numpy as np

try:
    import faiss
except ImportError:
    print("ERROR: FAISS not installed. Run: pip3 install faiss-cpu")
    exit(1)

try:
    import h5py
    HDF5_AVAILABLE = True
except ImportError:
    HDF5_AVAILABLE = False
    print("WARNING: h5py not installed. HDF5 datasets unavailable.")
    print("         Run: pip3 install h5py")


# ---------------------------------------------------------------------------
# Configuration
#
# N_CLUSTERS matches our Clustered HNSW K parameter — keeping this
# consistent ensures a fair comparison between the two systems.
#
# N_PROBE_LIST sweeps the nprobe parameter, which is directly analogous
# to TOP_CLUSTERS in our system. Each value produces one point on the
# Recall vs QPS curve in Figure 6.
#
# Reference: Aumüller et al. [3] — parameter sweep methodology
# ---------------------------------------------------------------------------

N_CLUSTERS   = 1000              # IVF partitions — matches our K
N_PROBE_LIST = [1, 5, 10, 20,    # nprobe sweep — matches our TOP_CLUSTERS sweep
                50, 100]          # higher nprobe = higher recall, lower QPS
K_RESULTS    = 10                 # number of nearest neighbors to retrieve


# ---------------------------------------------------------------------------
# Dataset loaders
#
# Two formats are supported:
#   .hdf5  — ANN-Benchmarks format [3]: single file with train/test/neighbors
#   .fvecs — SIFT1M binary format [2]: separate files for base/query/gt
#
# Both return the same three arrays:
#   train  : float32 (N, d) — base vectors to index
#   test   : float32 (Q, d) — query vectors
#   gt     : int32   (Q, K) — ground truth top-K neighbor indices
# ---------------------------------------------------------------------------

def load_hdf5(path):
    """
    Load an ANN-Benchmarks .hdf5 dataset.

    File layout (from Aumüller et al. [3]):
      /train      float32 (N, d) — base vectors
      /test       float32 (Q, d) — query vectors
      /neighbors  int32   (Q, 100) — ground truth top-100 neighbor indices
      /distances  float32 (Q, 100) — ground truth distances (unused here)

    Returns:
        train     : np.float32 (N, d)
        test      : np.float32 (Q, d)
        neighbors : np.int32   (Q, 100)
    """
    with h5py.File(path, 'r') as f:
        train = np.array(f['train'],     dtype=np.float32)
        test  = np.array(f['test'],      dtype=np.float32)
        gt    = np.array(f['neighbors'], dtype=np.int32)
    return train, test, gt


def read_fvecs(filename):
    """
    Read a SIFT1M .fvecs binary file.

    Format (Jégou et al. [2]):
      [dim (int32)] [v_0 v_1 ... v_{dim-1} (float32)] × N vectors

    Returns: np.float32 (N, d)
    """
    with open(filename, 'rb') as f:
        dim = struct.unpack('i', f.read(4))[0]
        f.seek(0)
        data = np.fromfile(f, dtype=np.float32)
    # Each record is (1 + dim) floats: skip the leading dim field
    return data.reshape(-1, dim + 1)[:, 1:].copy()


def read_ivecs(filename):
    """
    Read a SIFT1M .ivecs binary file (ground truth indices).

    Format identical to .fvecs but with int32 values.
    Returns: np.int32 (N, d)
    """
    with open(filename, 'rb') as f:
        dim = struct.unpack('i', f.read(4))[0]
        f.seek(0)
        data = np.fromfile(f, dtype=np.int32)
    return data.reshape(-1, dim + 1)[:, 1:].copy()


def load_dataset(args):
    """
    Auto-detect dataset format and load base, query, and ground truth arrays.

    Priority:
      1. --dataset path ending in .hdf5 → load_hdf5()
      2. Default SIFT1M binary files    → read_fvecs() / read_ivecs()
    """
    if args.dataset and HDF5_AVAILABLE and os.path.exists(args.dataset):
        print(f"Loading HDF5 dataset: {args.dataset}")
        return load_hdf5(args.dataset)
    else:
        print("Loading SIFT1M from .fvecs binary files...")
        sift_dir = "../data/sift"
        train = read_fvecs(f"{sift_dir}/sift_base.fvecs")
        test  = read_fvecs(f"{sift_dir}/sift_query.fvecs")
        gt    = read_ivecs(f"{sift_dir}/sift_groundtruth.ivecs")
        return train, test, gt


# ---------------------------------------------------------------------------
# Recall@k
#
# Recall@k = |R ∩ R̃| / k
#
#   R  = ground truth top-k neighbors (from HDF5 /neighbors or .ivecs)
#   R̃  = top-k neighbors returned by FAISS IVF
#
# We compute Recall@1 specifically because it is the primary metric
# used in Figure 6 and Table 1 of the paper — it measures whether the
# single nearest neighbor was found correctly.
#
# Reference: Aumüller et al. [3], Section 3 — evaluation metrics
# ---------------------------------------------------------------------------

def compute_recall_at_1(predicted, ground_truth):
    """
    Compute mean Recall@1 across all queries.

    Recall@1 = fraction of queries where the true nearest neighbor
               appears anywhere in the returned top-k results.

    Args:
        predicted    : np.int32 (Q, k) — FAISS IVF returned indices
        ground_truth : np.int32 (Q, K) — true neighbor indices (K >= 1)

    Returns:
        float — mean Recall@1 in [0.0, 1.0]

    Reference: Aumüller et al. [3], Section 3
    """
    # ground_truth[:, 0] is the true nearest neighbor for each query
    true_nn = ground_truth[:, 0]

    # Check if the true nearest neighbor appears in any of the k returned results
    # predicted shape: (Q, k) — any column match counts as a hit
    hits = np.any(predicted == true_nn[:, np.newaxis], axis=1)
    return float(np.mean(hits))


def compute_recall_at_k(predicted, ground_truth, k):
    """
    Compute mean Recall@k across all queries.

    Recall@k = |R ∩ R̃| / k
      R  = true top-k neighbors
      R̃  = returned top-k neighbors

    Args:
        predicted    : np.int32 (Q, k)
        ground_truth : np.int32 (Q, K) where K >= k
        k            : int

    Returns:
        float — mean Recall@k in [0.0, 1.0]
    """
    scores = []
    for i in range(len(predicted)):
        true_set = set(ground_truth[i][:k].tolist())
        pred_set = set(predicted[i][:k].tolist())
        scores.append(len(true_set & pred_set) / k)
    return float(np.mean(scores))


# ---------------------------------------------------------------------------
# Build IVF index
#
# FAISS IndexIVFFlat construction [1]:
#
#   Step 1 — Train (k-means):
#     Runs Lloyd's algorithm [6] on a sample of the dataset to find
#     K centroids. Each centroid defines a Voronoi cell.
#     Complexity: O(N × K × d × n_iter)
#     FAISS default: n_iter = 25 iterations
#
#   Step 2 — Add (assignment):
#     Assigns each of the N vectors to its nearest centroid.
#     Stores vectors in per-centroid posting lists.
#     Complexity: O(N × K × d) for centroid distances
#                + O(N × d) for copying vectors into posting lists
#
# FAISS uses IndexFlatL2 as the quantizer — a flat index over the K
# centroids. At query time, the quantizer finds the nprobe nearest
# centroids in O(K × d) by exhaustive scan over the K centroids.
#
# Reference: Johnson et al. [1], Jégou et al. [2]
# ---------------------------------------------------------------------------

def build_ivf_index(train, n_clusters, d):
    """
    Build a FAISS IndexIVFFlat index.

    Architecture:
      quantizer = IndexFlatL2(d)      — exact L2 scan over K centroids
      index     = IndexIVFFlat(...)   — K posting lists of float32 vectors

    The quantizer is itself a brute force index over the K centroids.
    This is fast because K << N (1000 << 1,000,000).

    Args:
        train      : np.float32 (N, d) — base vectors
        n_clusters : int               — number of IVF partitions (K)
        d          : int               — vector dimensionality

    Returns:
        index      : trained and populated faiss.IndexIVFFlat
        build_time : float (seconds)

    Reference: FAISS IndexIVFFlat [1], IVF algorithm [2]
    """
    # Step 1: Create the coarse quantizer
    # IndexFlatL2 performs exhaustive L2 scan over all K centroids
    # at query time. Since K=1000, this costs only 1000 × d = 128K ops.
    quantizer = faiss.IndexFlatL2(d)

    # Step 2: Create the IVF index
    # faiss.METRIC_L2 — use squared Euclidean distance (same as brute_force.py)
    # n_clusters      — number of Voronoi cells (= our K parameter)
    index = faiss.IndexIVFFlat(quantizer, d, n_clusters, faiss.METRIC_L2)

    t0 = time.time()

    # Step 3: Train — runs k-means to find K centroids
    # Requires at least 39 * n_clusters training vectors (FAISS rule of thumb)
    # For K=1000, N=1M: well above the minimum requirement
    assert train.shape[0] >= 39 * n_clusters, (
        f"Need at least {39 * n_clusters} training vectors for K={n_clusters}"
    )
    index.train(train)

    # Step 4: Add — assign all vectors to their nearest centroid
    # After this, index.ntotal == N
    index.add(train)

    build_time = time.time() - t0
    return index, build_time


# ---------------------------------------------------------------------------
# Search sweep
#
# We sweep nprobe over N_PROBE_LIST to produce the full Recall vs QPS
# curve. Each value of nprobe is one point on the curve in Figure 6.
#
# nprobe controls how many Voronoi cells are searched per query:
#   nprobe = 1   → only the nearest cell is searched (fastest, lowest recall)
#   nprobe = 10  → 10 nearest cells are searched
#   nprobe = K   → all cells searched = equivalent to brute force
#
# This is directly analogous to TOP_CLUSTERS in our Clustered HNSW.
# The comparison is fair because both use K=N_CLUSTERS partitions.
#
# Reference: Jégou et al. [2], Section 4 — nprobe parameter analysis
# ---------------------------------------------------------------------------

def sweep_nprobe(index, test, gt, k, nprobe_list):
    """
    Run search for each value of nprobe and record Recall@1 and QPS.

    Args:
        index      : trained faiss.IndexIVFFlat
        test       : np.float32 (Q, d) — query vectors
        gt         : np.int32   (Q, K) — ground truth neighbors
        k          : int — number of results to retrieve
        nprobe_list: list of int — nprobe values to sweep

    Returns:
        list of dicts with keys: nprobe, recall1, recallk, qps, search_time
    """
    results = []

    for nprobe in nprobe_list:
        # Set nprobe — this is the only parameter that changes between runs
        # All other index parameters (K, M, ef) remain fixed
        index.nprobe = nprobe

        # Run search over all Q queries
        # FAISS search() returns (distances, indices), both shape (Q, k)
        t0 = time.time()
        _, predicted = index.search(test, k)
        search_time = time.time() - t0

        qps     = len(test) / search_time
        recall1 = compute_recall_at_1(predicted, gt)
        recallk = compute_recall_at_k(predicted, gt, k)

        results.append({
            'nprobe':      nprobe,
            'recall1':     recall1,
            'recallk':     recallk,
            'qps':         qps,
            'search_time': search_time,
        })

    return results


# ---------------------------------------------------------------------------
# CSV output
#
# Appends results to ../results.csv after each run.
# Schema matches brute_force.py:
#   timestamp, machine, algorithm, recall@1, recall@k,
#   qps, build_time_s, search_time_s, num_queries, dataset, metric
# ---------------------------------------------------------------------------

def save_csv(results, build_time, num_queries, dataset):
    """
    Append one row per nprobe value to ../results.csv.
    Each row corresponds to one point on the Recall vs QPS curve.
    """
    ts      = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    machine = socket.gethostname()

    with open('../results.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        for r in results:
            writer.writerow([
                ts,
                machine,
                f"faiss_ivf_nprobe{r['nprobe']}",  # algorithm name
                f"{r['recall1']:.4f}",              # recall@1
                f"{r['recallk']:.4f}",              # recall@k
                f"{r['qps']:.1f}",                  # QPS
                f"{build_time:.2f}",                # build time (seconds)
                f"{r['search_time']:.3f}",          # search time (seconds)
                num_queries,                        # number of queries
                dataset,                            # dataset name
                'l2',                               # distance metric
            ])


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='FAISS IVF benchmark — partition-based comparison baseline'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default=None,
        help='Path to ANN-Benchmarks .hdf5 dataset (default: SIFT1M .fvecs)'
    )
    parser.add_argument(
        '--n-clusters',
        type=int,
        default=N_CLUSTERS,
        help=f'Number of IVF clusters K (default: {N_CLUSTERS})'
    )
    parser.add_argument(
        '--k',
        type=int,
        default=K_RESULTS,
        help=f'Number of nearest neighbors to retrieve (default: {K_RESULTS})'
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  FAISS IVF — Partition-Based Comparison Baseline")
    print("  Comparison: nprobe (IVF) ↔ TOP_CLUSTERS (Clustered HNSW)")
    print("=" * 60)

    # -----------------------------------------------------------------------
    # Load dataset
    # -----------------------------------------------------------------------

    print("\nLoading dataset...")
    train, test, gt = load_dataset(args)
    n, d = train.shape
    print(f"  Base vectors : {n:,} × {d} dimensions")
    print(f"  Query vectors: {len(test):,}")
    print(f"  Ground truth : top-{gt.shape[1]} neighbors per query")

    # -----------------------------------------------------------------------
    # Build IVF index
    #
    # This is the most expensive step — k-means on 1M × 128-d vectors.
    # FAISS uses optimized BLAS for the centroid distance computations.
    # Reference: Johnson et al. [1], Jégou et al. [2]
    # -----------------------------------------------------------------------

    print(f"\nBuilding FAISS IVF index (K={args.n_clusters} clusters)...")
    print(f"  Step 1: k-means training (Lloyd's algorithm [6])...")
    print(f"  Step 2: assigning {n:,} vectors to nearest centroid...")

    index, build_time = build_ivf_index(train, args.n_clusters, d)

    print(f"  Build time  : {build_time:.2f}s")
    print(f"  Vectors indexed: {index.ntotal:,}")

    # -----------------------------------------------------------------------
    # Sweep nprobe
    #
    # Each value of nprobe produces one (Recall, QPS) point on Figure 6.
    # nprobe directly controls the recall/speed tradeoff — analogous to
    # TOP_CLUSTERS in our Clustered HNSW system.
    # -----------------------------------------------------------------------

    print(f"\nSweeping nprobe = {N_PROBE_LIST}...")
    print(f"  (nprobe ↔ TOP_CLUSTERS: both control partition search depth)")
    print()
    print(f"  {'nprobe':<10} {'Recall@1':<12} {'Recall@'+str(args.k):<12} "
          f"{'QPS':<12} {'Time(s)':<10}")
    print("  " + "-" * 56)

    results = sweep_nprobe(index, test, gt, args.k, N_PROBE_LIST)

    for r in results:
        print(f"  {r['nprobe']:<10} {r['recall1']:<12.4f} "
              f"{r['recallk']:<12.4f} {r['qps']:<12.1f} "
              f"{r['search_time']:<10.3f}")

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------

    print()
    print("=" * 60)
    print(f"  SUMMARY")
    print("=" * 60)
    print(f"  Algorithm    : FAISS IndexIVFFlat [1]")
    print(f"  K (clusters) : {args.n_clusters}  "
          f"(= N_CLUSTERS in Clustered HNSW)")
    print(f"  Build time   : {build_time:.2f}s")
    print(f"  Distance     : L2 Euclidean")
    print(f"  Note: nprobe ↔ TOP_CLUSTERS — same role in both systems")
    print("=" * 60)

    # -----------------------------------------------------------------------
    # Save results to CSV
    # -----------------------------------------------------------------------

    dataset_name = args.dataset if args.dataset else 'sift'
    save_csv(results, build_time, len(test), dataset_name)
    print(f"\n  Results saved to ../results.csv")
    print(f"  Each row = one nprobe value = one point on Figure 6 curve")


if __name__ == "__main__":
    main()