"""
brute_force.py
--------------
Exact nearest neighbor search — the Recall=1.0 reference baseline.

Every ANN algorithm in this benchmark is evaluated against brute force.
Brute force computes the true nearest neighbors by exhaustively computing
the distance between the query vector and every vector in the dataset.
It is always correct (Recall@1 = 1.0) but scales linearly with dataset
size — O(N x d) per query — making it impractical at million scale.

Supported distance metrics:
  --metric l2       L2 (Euclidean) distance     — SIFT1M, GIST1M
  --metric ip       Inner Product (dot product)  — recommendation embeddings
  --metric cosine   Cosine similarity            — text, NLP embeddings
  --metric angular  Angular distance             — GloVe-25, GloVe-100
  --metric l1       Manhattan (L1) distance      — sparse data
  --metric hamming  Hamming distance             — binary codes

Distance formulas:
  L2       : delta(x, q) = sqrt( sum_i (x_i - q_i)^2 )
  IP       : delta(x, q) = - sum_i x_i * q_i     (negated: higher = closer)
  Cosine   : delta(x, q) = 1 - (x.q) / (|x| |q|)
  Angular  : delta(x, q) = arccos( (x.q) / (|x| |q|) )
  L1       : delta(x, q) = sum_i |x_i - q_i|
  Hamming  : delta(x, q) = number of differing bits

Implementation:
  L2, IP, Cosine, Angular -> FAISS IndexFlat* [1]
  L1, Hamming             -> NumPy (FAISS does not support these natively)

References:
  [1] J. Johnson, M. Douze, H. Jegou, "Billion-Scale Similarity Search
      with GPUs," IEEE Big Data, 7(3): 535-547, 2021.
      https://arxiv.org/abs/1702.08734

  [2] H. Jegou, M. Douze, C. Schmid, "Product Quantization for Nearest
      Neighbor Search," IEEE TPAMI, 33(1): 117-128, 2011.
      http://corpus-texmex.irisa.fr/

  [3] M. Aumüller, E. Bernhardsson, A. Faithfull, "ANN-Benchmarks: A
      Benchmarking Tool for Approximate Nearest Neighbor Algorithms,"
      Information Systems, 87, 2020.
      https://doi.org/10.1016/j.is.2019.02.006

  [4] J. Pennington, R. Socher, C. Manning, "GloVe: Global Vectors for
      Word Representation," EMNLP, 2014.
      https://nlp.stanford.edu/projects/glove/

  [5] M. Wang et al., "A Comprehensive Survey and Experimental Comparison
      of Graph-Based Approximate Nearest Neighbor Search,"
      PVLDB, 14(11): 1964-1978, 2021.
      https://www.vldb.org/pvldb/vol14/p1964-wang.pdf

Usage:
  python3 brute_force.py --metric l2
  python3 brute_force.py --metric cosine --dataset ../data/glove-25-angular.hdf5
  python3 brute_force.py --metric angular --dataset ../data/glove-100-angular.hdf5
  python3 brute_force.py --metric ip
  python3 brute_force.py --metric l1
  python3 brute_force.py --metric hamming
"""

import numpy as np
import faiss
import time
import struct
import csv
import socket
import argparse
import os
from datetime import datetime


# ---------------------------------------------------------------------------
# Dataset readers
#
# SIFT1M uses binary .fvecs / .ivecs format [2]:
#   .fvecs: [dim (int32)] [v0 v1 ... v_dim (float32)] repeated N times
#   .ivecs: [dim (int32)] [v0 v1 ... v_dim (int32)]   repeated N times
#
# ANN-Benchmarks uses .hdf5 format [3] — single file contains
# train, test, neighbors, and distances.
# ---------------------------------------------------------------------------

def read_fvecs(filename):
    """Read a .fvecs binary file -> float32 numpy array (N, d)."""
    with open(filename, 'rb') as f:
        dim = struct.unpack('i', f.read(4))[0]
        f.seek(0)
        data = np.fromfile(f, dtype=np.float32)
    return data.reshape(-1, dim + 1)[:, 1:].copy()


def read_ivecs(filename):
    """Read a .ivecs binary file -> int32 numpy array (N, d)."""
    with open(filename, 'rb') as f:
        dim = struct.unpack('i', f.read(4))[0]
        f.seek(0)
        data = np.fromfile(f, dtype=np.int32)
    return data.reshape(-1, dim + 1)[:, 1:].copy()


def read_hdf5(filename):
    """
    Read an ANN-Benchmarks .hdf5 file [3].
    Returns (train, test, neighbors, distances).
    """
    try:
        import h5py
    except ImportError:
        raise ImportError("Install h5py: pip3 install h5py")
    with h5py.File(filename, 'r') as f:
        train     = np.array(f['train'],     dtype=np.float32)
        test      = np.array(f['test'],      dtype=np.float32)
        neighbors = np.array(f['neighbors'], dtype=np.int32)
        distances = np.array(f['distances'], dtype=np.float32)
    return train, test, neighbors, distances


def load_dataset(args):
    """
    Auto-detect file format and load base, queries, and ground truth.
    Supports .fvecs/.ivecs (SIFT1M) and .hdf5 (ANN-Benchmarks).
    """
    path = args.dataset
    ext  = os.path.splitext(path)[1].lower()

    if ext == '.hdf5':
        # ANN-Benchmarks format — one file contains everything [3]
        base, queries, gt, _ = read_hdf5(path)
        return base, queries, gt
    else:
        # SIFT1M binary format [2]
        base    = read_fvecs('../data/sift/sift_base.fvecs')
        queries = read_fvecs('../data/sift/sift_query.fvecs')
        gt      = read_ivecs('../data/sift/sift_groundtruth.ivecs')
        return base, queries, gt


# ---------------------------------------------------------------------------
# Recall@k
#
# Recall@k = |R ∩ R~| / k
#
#   R  = ground truth top-k neighbors (exact)
#   R~ = top-k neighbors returned by the algorithm
#
# For brute force: R~ == R by definition -> Recall@k = 1.0 always.
# Reference: Aumüller et al. [3], Section 3.
# ---------------------------------------------------------------------------

def compute_recall_at_k(predicted, ground_truth, k):
    """
    Compute mean Recall@k across all queries.

    Args:
        predicted    (N, k) int array — returned neighbor indices
        ground_truth (N, K) int array — true neighbor indices (K >= k)
        k            int              — number of neighbors to evaluate

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
# Vector normalization
#
# Required for cosine and angular metrics.
# Normalizing to unit length converts cosine similarity to inner product:
#   cosine(x, q) = x . q  (when |x| = |q| = 1)
# This allows FAISS IndexFlatIP to compute cosine/angular distance.
# ---------------------------------------------------------------------------

def normalize(vectors):
    """Normalize rows of a float32 matrix to unit L2 norm."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)    # avoid division by zero
    return (vectors / norms).astype(np.float32)


# ---------------------------------------------------------------------------
# Search functions — one per metric
#
# L2, IP, Cosine, Angular: FAISS IndexFlat* [1]
#   Optimized BLAS + SIMD — 10-100x faster than numpy loops
#   Mathematically identical to naive exhaustive scan
#
# L1, Hamming: NumPy batch computation
#   FAISS does not support these metrics for float vectors natively
# ---------------------------------------------------------------------------

def search_l2(base, queries, k):
    """
    Exact L2 Euclidean nearest neighbor search.

    Formula    : delta(x, q) = sqrt( sum_i (x_i - q_i)^2 )
    Complexity : O(N x d) per query
    Used for   : SIFT1M [2], GIST1M — image feature vectors
    FAISS index: IndexFlatL2

    Reference: FAISS [1]
    """
    d = base.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(base.astype(np.float32))
    _, indices = index.search(queries.astype(np.float32), k)
    return indices, index


def search_ip(base, queries, k):
    """
    Exact Inner Product (dot product) nearest neighbor search.

    Formula    : score(x, q) = sum_i x_i * q_i  (higher = more similar)
    Complexity : O(N x d) per query
    Used for   : recommendation systems, dense passage embeddings
    FAISS index: IndexFlatIP

    Note: FAISS returns results sorted by descending inner product.
    Reference: FAISS [1]
    """
    d = base.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(base.astype(np.float32))
    _, indices = index.search(queries.astype(np.float32), k)
    return indices, index


def search_cosine(base, queries, k):
    """
    Exact Cosine similarity nearest neighbor search.

    Formula    : cosine(x, q) = (x . q) / (|x| |q|)
    After L2 normalization: cosine(x, q) = x . q  (since |x| = |q| = 1)
    Complexity : O(N x d) per query
    Used for   : text embeddings, sentence transformers, NLP vectors
    FAISS index: IndexFlatIP on normalized vectors

    Reference: FAISS normalization trick [1]
    """
    base_norm    = normalize(base.copy())
    queries_norm = normalize(queries.copy())
    d = base_norm.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(base_norm)
    _, indices = index.search(queries_norm, k)
    return indices, index


def search_angular(base, queries, k):
    """
    Exact Angular distance nearest neighbor search.

    Formula    : delta(x, q) = arccos( (x . q) / (|x| |q|) )
    After normalization, inner product ranking == angular ranking.
    Complexity : O(N x d) per query
    Used for   : GloVe-25, GloVe-100 [4], ANN-Benchmarks angular datasets
    FAISS index: IndexFlatIP on normalized vectors

    Reference: ANN-Benchmarks [3], angular distance metric
    """
    base_norm    = normalize(base.copy())
    queries_norm = normalize(queries.copy())
    d = base_norm.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(base_norm)
    _, indices = index.search(queries_norm, k)
    return indices, index


def search_l1(base, queries, k):
    """
    Exact L1 Manhattan distance nearest neighbor search using NumPy.

    Formula    : delta(x, q) = sum_i |x_i - q_i|
    Complexity : O(N x d) per query
    Used for   : sparse data, certain image descriptors
    Note       : FAISS does not support L1 for float vectors natively.
                 We batch queries to control memory usage.

    Reference: Wang et al. [5], distance metric comparison
    """
    batch_size  = 100
    all_indices = []

    for start in range(0, len(queries), batch_size):
        batch = queries[start:start + batch_size]
        # Shape: (batch_size, N) — L1 distance from each query to all base vectors
        dists = np.sum(
            np.abs(base[np.newaxis, :, :] - batch[:, np.newaxis, :]),
            axis=2
        )
        idx = np.argpartition(dists, k, axis=1)[:, :k]
        sorted_idx = np.array([
            idx[i][np.argsort(dists[i][idx[i]])]
            for i in range(len(batch))
        ])
        all_indices.append(sorted_idx)

    return np.vstack(all_indices), None


def search_hamming(base, queries, k):
    """
    Exact Hamming distance nearest neighbor search using NumPy.

    Formula    : delta(x, q) = number of bit positions where x_i != q_i
    Complexity : O(N x d) per query
    Used for   : binary codes, LSH outputs, locality-sensitive hashing
    Note       : float vectors are binarized by thresholding at 0.
                 For true uint8 binary vectors, use faiss.IndexBinaryFlat.

    Reference: Wang et al. [5], hashing-based ANN methods
    """
    # Binarize: positive values -> 1, non-positive -> 0
    base_bin    = (base > 0).astype(np.uint8)
    queries_bin = (queries > 0).astype(np.uint8)

    batch_size  = 100
    all_indices = []

    for start in range(0, len(queries_bin), batch_size):
        batch = queries_bin[start:start + batch_size]
        # XOR then sum = Hamming distance
        dists = np.sum(
            base_bin[np.newaxis, :, :] ^ batch[:, np.newaxis, :],
            axis=2
        )
        idx = np.argpartition(dists, k, axis=1)[:, :k]
        sorted_idx = np.array([
            idx[i][np.argsort(dists[i][idx[i]])]
            for i in range(len(batch))
        ])
        all_indices.append(sorted_idx)

    return np.vstack(all_indices), None


# ---------------------------------------------------------------------------
# Metric dispatcher
# ---------------------------------------------------------------------------

SEARCH_FUNCTIONS = {
    'l2':      search_l2,
    'ip':      search_ip,
    'cosine':  search_cosine,
    'angular': search_angular,
    'l1':      search_l1,
    'hamming': search_hamming,
}

METRIC_DESCRIPTIONS = {
    'l2':      'L2 Euclidean distance       — FAISS IndexFlatL2',
    'ip':      'Inner Product (dot product) — FAISS IndexFlatIP',
    'cosine':  'Cosine similarity           — FAISS IndexFlatIP + normalize',
    'angular': 'Angular distance            — FAISS IndexFlatIP + normalize',
    'l1':      'Manhattan L1 distance       — NumPy batch',
    'hamming': 'Hamming distance            — NumPy binarized',
}

METRIC_DATASETS = {
    'l2':      'SIFT1M, GIST1M',
    'ip':      'Recommendation, dense embeddings',
    'cosine':  'Text embeddings, sentence transformers',
    'angular': 'GloVe-25, GloVe-100, NYTimes',
    'l1':      'Sparse data',
    'hamming': 'Binary codes, LSH outputs',
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Brute force exact nearest neighbor search — Recall=1.0 baseline'
    )
    parser.add_argument(
        '--metric',
        type=str,
        default='l2',
        choices=SEARCH_FUNCTIONS.keys(),
        help='Distance metric: l2 | ip | cosine | angular | l1 | hamming (default: l2)'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default='sift',
        help='Path to .hdf5 dataset or "sift" for SIFT1M .fvecs files (default: sift)'
    )
    parser.add_argument(
        '--k',
        type=int,
        default=10,
        help='Number of nearest neighbors to retrieve (default: 10)'
    )
    args = parser.parse_args()

    print("=" * 65)
    print("  Brute Force Exact Search — Recall=1.0 Reference Baseline")
    print(f"  Metric  : {METRIC_DESCRIPTIONS[args.metric]}")
    print(f"  Typical : {METRIC_DATASETS[args.metric]}")
    print(f"  Dataset : {args.dataset}")
    print(f"  k       : {args.k}")
    print("=" * 65)

    # -----------------------------------------------------------------------
    # Load dataset
    # -----------------------------------------------------------------------
    print("\nLoading dataset...")
    base, queries, gt = load_dataset(args)
    n, d = base.shape
    num_queries = len(queries)
    print(f"  Base vectors : {n:,} x {d} dimensions")
    print(f"  Query vectors: {num_queries:,}")
    print(f"  Ground truth : top-{gt.shape[1]} neighbors per query")

    # -----------------------------------------------------------------------
    # Run exact search
    # -----------------------------------------------------------------------
    print(f"\nRunning exact {args.metric.upper()} search over {n:,} vectors...")
    search_fn = SEARCH_FUNCTIONS[args.metric]

    t0 = time.time()
    indices, index = search_fn(base, queries, args.k)
    search_time = time.time() - t0

    qps = num_queries / search_time

    # -----------------------------------------------------------------------
    # Compute Recall@1 and Recall@k
    # Both should be 1.0 for brute force — we verify correctness.
    # Reference: Aumüller et al. [3]
    # -----------------------------------------------------------------------
    recall_at_1 = compute_recall_at_k(indices, gt, k=1)
    recall_at_k = compute_recall_at_k(indices, gt, k=args.k)

    # -----------------------------------------------------------------------
    # Report
    # -----------------------------------------------------------------------
    print(f"\n{'=' * 65}")
    print(f"  RESULTS")
    print(f"{'=' * 65}")
    print(f"  Metric      : {args.metric.upper()}")
    print(f"  Recall@1    : {recall_at_1:.4f}  (expected: 1.0000)")
    print(f"  Recall@{args.k:<4}  : {recall_at_k:.4f}  (expected: 1.0000)")
    print(f"  QPS         : {qps:,.1f} queries/second")
    print(f"  Search time : {search_time:.2f}s  ({num_queries:,} queries)")
    print(f"  Build time  : 0s  (no index — exhaustive scan)")
    if index is not None:
        mem_mb = n * d * 4 / 1024**2
        print(f"  Index size  : {mem_mb:.1f} MB  (raw float32 vectors)")
    print(f"{'=' * 65}")

    if recall_at_1 < 0.999:
        print(f"\n  WARNING: Recall@1 = {recall_at_1:.4f} — should be 1.0 for exact search.")
        print(f"  Check ground truth alignment and metric correctness.")

    # -----------------------------------------------------------------------
    # Save to CSV
    # CSV columns: timestamp, machine, algorithm, recall@1, recall@k,
    #              qps, build_time_s, search_time_s, num_queries,
    #              dataset, metric
    # -----------------------------------------------------------------------
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    machine   = socket.gethostname()
    row = [
        timestamp,
        machine,
        f'brute_force_{args.metric}',
        f'{recall_at_1:.4f}',
        f'{recall_at_k:.4f}',
        f'{qps:.1f}',
        '0',
        f'{search_time:.2f}',
        num_queries,
        args.dataset,
        args.metric
    ]

    with open('../results.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(row)

    print(f"\n  Results saved to ../results.csv")


if __name__ == "__main__":
    main()
