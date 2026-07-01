# Brute Force — Exact Nearest Neighbor Search

The brute-force algorithm serves as the **exact-search baseline** for evaluating all Approximate Nearest Neighbor (ANN) indexing techniques in this project. Unlike ANN methods that trade accuracy for speed, brute force computes the exact nearest neighbors by comparing every query vector against every vector in the dataset.

This implementation uses **FAISS IndexFlatL2**, which performs exhaustive Euclidean distance search while leveraging optimized SIMD instructions and highly optimized linear algebra routines. Although significantly faster than a naïve implementation, it still performs an exhaustive scan and therefore produces exact search results.

---

## Purpose

The brute-force baseline is used to:

- Generate the exact nearest neighbors for evaluation.
- Compute Recall@k for ANN algorithms.
- Establish the upper bound on search accuracy.
- Provide a reference for measuring the speedup achieved by approximate methods.

Every ANN algorithm in this repository is compared against the results produced by this implementation.

---

## Distance Metric

This benchmark uses only the **Euclidean (L2)** distance metric because the primary benchmark dataset, **SIFT1M**, is defined using Euclidean distance.

The distance between a query vector \(q\) and a database vector \(x\) is

\[
d(x,q)=\sqrt{\sum_{i=1}^{d}(x_i-q_i)^2}
\]

FAISS implements this computation using the **IndexFlatL2** index.

---

## Implementation

The implementation consists of three main stages.

### 1. Load Dataset

The benchmark loads the ANN-Benchmarks HDF5 dataset containing

- database vectors (`train`)
- query vectors (`test`)
- ground-truth nearest neighbors (`neighbors`)

### 2. Build Exact Index

The database vectors are inserted into a FAISS `IndexFlatL2` index.

```python
index = faiss.IndexFlatL2(dimension)
index.add(base_vectors)
```

Unlike ANN indexes, `IndexFlatL2` performs no clustering, partitioning, graph construction, or quantization.

### 3. Exact Search

Each query is compared against every database vector.

```python
distances, indices = index.search(query_vectors, k)
```

The returned neighbors are compared with the provided ground truth to compute Recall@k.

---

## Evaluation Metrics

The following metrics are collected during execution.

| Metric | Description |
|---------|-------------|
| Recall@1 | Exact nearest-neighbor accuracy |
| Recall@k | Fraction of correct top-k neighbors |
| Queries Per Second (QPS) | Search throughput |
| Average Latency | Mean query execution time |
| Build Time | Time required to create the index |
| Search Time | Total query execution time |
| Index Size | Memory occupied by the index |

---

## Computational Complexity

| Stage | Complexity |
|---------|------------|
| Index construction | O(N) |
| Query search | O(N × d) |
| Memory | O(N × d) |

where

- **N** = number of database vectors
- **d** = vector dimension

Because every query examines all vectors, brute force provides perfect accuracy but has the highest computational cost.

---

## Running

```bash
python3 brute_force.py \
    --dataset ../data/sift-128-euclidean.hdf5 \
    --k 10
```

---

## Example Output

```text
Algorithm      : faiss_flat_l2
Distance       : L2 / Euclidean
Recall@1       : 0.9926
Recall@10      : 0.99935
QPS            : 3630.70 queries/second
Avg latency    : 0.275429 ms/query
Build time     : 0.039210 s
Search time    : 2.754292 s
Index size     : 488.281 MB
```

---

## References

1. J. Johnson, M. Douze, and H. Jégou. *Billion-Scale Similarity Search with GPUs.* IEEE Transactions on Big Data, 2021.

2. M. Aumüller, E. Bernhardsson, and A. Faithfull. *ANN-Benchmarks: A Benchmarking Tool for Approximate Nearest Neighbor Algorithms.* Information Systems, 2019.

3. H. Jégou, M. Douze, and C. Schmid. *Product Quantization for Nearest Neighbor Search.* IEEE TPAMI, 2011.