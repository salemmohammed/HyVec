## Introduction

This document summarizes the Approximate Nearest Neighbor Search (ANNS) algorithms evaluated in this benchmarking framework. It provides a concise overview of the algorithms, their underlying principles, computational characteristics, and their role in the experimental evaluation.

The benchmark includes representative methods from the major ANN indexing families together with the proposed **Hybrid Attribute-Spatial HNSW** index. All algorithms are evaluated under identical datasets, hardware, and evaluation protocols to ensure fair and reproducible comparisons.

---

# Problem Formulation

Given a dataset

\[
\mathcal{X}=\{x_1,x_2,\ldots,x_N\}\subset\mathbb{R}^{d}
\]

and a query vector \(q\in\mathbb{R}^{d}\), the nearest-neighbor search problem is

\[
x^*=\arg\min_{x\in\mathcal{X}}\delta(x,q)
\]

where \(\delta(\cdot,\cdot)\) denotes the Euclidean (L2) distance.

Exact search examines every vector in the dataset and guarantees the correct nearest neighbors. Approximate Nearest Neighbor Search (ANNS) reduces the search cost by exploring only a subset of candidate vectors while maintaining high search accuracy.

---

# Evaluation Metric

Search accuracy is measured using **Recall@k**

\[
\mathrm{Recall@k}=
\frac{\left|\mathcal{R}\cap\tilde{\mathcal{R}}\right|}{k},
\]

where

- \(\mathcal{R}\) is the exact top-\(k\) result set.
- \(\tilde{\mathcal{R}}\) is the approximate result set.

The primary performance trade-off is between **Recall@k** and **Queries Per Second (QPS)**.

---

# Algorithm Taxonomy

Following Wang et al., ANN algorithms are grouped into four major families.

| Family | Representative Algorithms |
|---------|---------------------------|
| Hash-based | LSH, E2LSH |
| Tree-based | KD-Tree, FLANN, Annoy, MRPT |
| Quantization-based | PQ, IVF-PQ, ScaNN |
| Graph-based | NSW, HNSW, NSG, DiskANN |

Graph-based methods currently represent the state of the art for high-dimensional vector search and are widely adopted in modern vector database systems.

---

# Evaluated Algorithms

## 1. Brute Force (Exact Search)

Brute Force performs exhaustive nearest-neighbor search by comparing every query vector against every vector in the dataset.

This implementation uses **FAISS IndexFlatL2**, which produces the exact nearest neighbors and serves as the ground-truth baseline for evaluating approximate indexing algorithms.

### Characteristics

| Property | Value |
|----------|-------|
| Search Type | Exact |
| Distance Metric | Euclidean (L2) |
| Query Complexity | \(O(Nd)\) |
| Build Complexity | \(O(N)\) |
| Memory | \(O(Nd)\) |

Brute Force provides the highest possible search accuracy but has the highest computational cost.

---

## 2. FAISS-IVF (Inverted File Index)

FAISS-IVF partitions the dataset into \(K\) Voronoi cells using k-means clustering. During query processing, the query is compared with all centroids, the nearest **nprobe** clusters are selected, and vectors within those clusters are searched using exact Euclidean distance.

### Main Parameters

| Parameter | Description |
|-----------|-------------|
| nlist | Number of clusters |
| nprobe | Number of clusters searched |

### Complexity

| Property | Complexity |
|----------|------------|
| Query | \(O(Kd+nprobe\cdot N/K)\) |
| Build | Dominated by k-means clustering |
| Memory | \(O(Nd)\) |

FAISS-IVF serves as the primary partition-based baseline.

---

## 3. HNSW

Hierarchical Navigable Small World (HNSW) constructs a hierarchical proximity graph for efficient approximate nearest-neighbor search.

The search begins at the highest graph layer and greedily descends toward lower layers until the nearest neighbors are located.

### Main Parameters

| Parameter | Description |
|-----------|-------------|
| M | Maximum graph degree |
| ef_construction | Construction search width |
| ef_search | Query search width |

### Complexity

| Property | Complexity |
|----------|------------|
| Query | \(O(\log N)\) |
| Build | \(O(N\log N)\) |
| Memory | \(O(NM)\) |

HNSW is the representative graph-based baseline used in this benchmark.

---

## 4. ScaNN

ScaNN combines partitioning with anisotropic vector quantization to accelerate approximate nearest-neighbor search.

It reduces both memory usage and search latency while maintaining high recall and is included as the representative quantization-based baseline.

---

## 5. Hybrid Attribute-Spatial HNSW (Proposed)

The proposed Hybrid Attribute-Spatial HNSW index combines spatial partitioning with graph-based search.

During index construction:

- The dataset is partitioned into \(K\) clusters using k-means.
- Each cluster stores its centroid.
- Each cluster maintains an independent HNSW index.

For a query:

\[
q
\rightarrow
\text{Centroid Search}
\rightarrow
\text{Top-}T
\rightarrow
\text{Local HNSW Search}
\rightarrow
\text{Merge}
\rightarrow
\text{Top-}k
\]

Only the \(T\) nearest clusters are searched.

### Main Parameters

| Parameter | Description |
|-----------|-------------|
| K | Number of clusters |
| T | Number of searched clusters |
| M | HNSW graph degree |
| ef_construction | Construction parameter |
| ef_search | Query parameter |

### Complexity

| Property | Complexity |
|----------|------------|
| Query | \(O(Kd+T\log(N/K))\) |
| Build | K-means + local HNSW construction |
| Memory | \(O(NM)\) |

Compared with a global HNSW index, this approach restricts graph traversal to spatially relevant partitions while preserving HNSW search efficiency.

---

# Evaluation Protocol

All algorithms are evaluated under identical experimental conditions.

## Dataset

- SIFT1M
- Euclidean (L2) distance

## Metrics

- Recall@1
- Recall@k
- Queries Per Second (QPS)
- Average Query Latency
- Index Construction Time
- Search Time
- Memory Consumption
- Index Size

All baseline algorithms are executed on the same hardware platform to ensure fair and reproducible comparisons.

---

# References

1. Wang et al. *A Comprehensive Survey and Experimental Comparison of Graph-Based Approximate Nearest Neighbor Search*. PVLDB, 2021.
2. Malkov and Yashunin. *Efficient and Robust Approximate Nearest Neighbor Search Using Hierarchical Navigable Small World Graphs*. IEEE TPAMI, 2020.
3. Johnson, Douze, and Jégou. *Billion-Scale Similarity Search with GPUs*. IEEE Transactions on Big Data, 2021.
4. Guo et al. *Accelerating Large-Scale Inference with Anisotropic Vector Quantization*. ICML, 2020.
5. Aumüller, Bernhardsson, and Faithfull. *ANN-Benchmarks: A Benchmarking Tool for Approximate Nearest Neighbor Algorithms*. Information Systems, 2020.
