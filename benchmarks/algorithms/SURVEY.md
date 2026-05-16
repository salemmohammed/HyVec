# Algorithms

This folder contains implementations of the approximate nearest neighbor (ANN)
algorithms evaluated in our benchmarks. We follow the taxonomy and evaluation
methodology established by Wang et al. [1] the most comprehensive survey of
graph-based ANNS algorithms to date and extend it with our Clustered HNSW system.

---

## Background: What is ANNS?

### The Problem

Imagine you have 1M images stored as vectors. Each image represented as a list of 128 numbers that describe its visual content. Given a new query image, you want to find the most similar images in your collection. The naive approach is to compare the query against every single vector is called **exact nearest neighbor search**. It always finds the correct answer, but at 1M comparisons per query, it becomes too slow for real-time applications.

### The Solution: Approximate Search

**Approximate Nearest Neighbor Search (ANNS)** accepts a small loss in accuracy in exchange for dramatically faster query times. Instead of searching all N vectors, it builds an index that guides the search toward a small candidate set and visiting only a few thousand vectors instead of millions. The accuracy of an ANNS algorithm is measured by **Recall@k** the fraction of the true k nearest neighbors that were actually returned:

```
Recall@k = |R ∩ R̃| / k

  R  = the exact k nearest neighbors (ground truth, from brute force)
  R̃  = the approximate k nearest neighbors returned by ANNS

  Distance: δ(x, q) = √ Σ (xᵢ - qᵢ)²    ← Euclidean (L2) distance

Example: k = 10, ANNS returns 9 of the true 10 → Recall@10 = 0.9
```

A Recall@k of 1.0 means perfect accuracy. A value of 0.95 means 95% of the true nearest neighbors were found which is acceptable for most applications.

### The Core Tradeoff

Every ANNS algorithm navigates the same fundamental tradeoff:

```
Search fewer vectors → faster QPS,  lower Recall
Search more vectors  → slower QPS,  higher Recall
```

The goal is to find an index structure that sits as high and as far right as possible on the Recall vs QPS curve achieving high accuracy without sacrificing speed.

## Algorithm Taxonomy

Following Wang et al. [1], ANNS algorithms fall into four major families:

```
ANNS Algorithms
│
├── Hashing-based       → LSH, E2LSH
├── Tree-based          → KD-Tree, Annoy, FLANN, MRPT
├── Quantization-based  → PQ, IVFPQ, ScaNN
└── Graph-based         → NSW, HNSW, NSG, DiskANN  ← state of the art
```

Graph-based algorithms have emerged as the dominant paradigm because they evaluate fewer candidate points while achieving higher recall than other families [1]. Our system is a **hybrid** that combines graph-based search (HNSW) with partition-based routing (IVF-style clustering) — routing each query to a small attribute-based cluster, then applying HNSW search within that cluster only.

---

## Graph-Based ANNS: Four Base Graphs

Wang et al. [1] identify four foundational graph structures from which all graph-based ANNS algorithms are derived:

| Base Graph | Key Property | Used By |
|---|---|---|
| **Delaunay Graph (DG)** | Guarantees exact NNS; nearly fully connected in high dimensions | NSW |
| **Relative Neighborhood Graph (RNG)** | Cuts redundant neighbors; distributes neighbors omnidirectionally | HNSW, NSG, FANNG |
| **K-Nearest Neighbor Graph (KNNG)** | Limits neighbors to K; efficient but may lose global connectivity | EFANNA, IEH |
| **Minimum Spanning Tree (MST)** | Fewest edges for global connectivity; may detour during search | HCNNG |

HNSW is an **RNG-based** algorithm — it approximates the RNG by diversifying neighbor distribution across hierarchical layers. This omnidirectional neighbor distribution is what enables greedy graph traversal to reliably converge toward the nearest neighbor, achieving logarithmic search complexity O(log N) — the best among all 13 graph-based algorithms surveyed by Wang et al. [1].

---

## Algorithms

---

### 1. Brute Force — Exact Search (Reference Baseline)

**Core idea:**
Compare the query vector against every vector in the dataset. No index is built. Returns exact nearest neighbors by definition.

**Formal complexity:**
```
Query time:  O(N × d)   — linear scan over N vectors of dimension d
Build time:  O(1)       — no index construction
Memory:      O(N × d)   — store the full dataset
Recall@k:    1.0        — always exact
```

**Role in our evaluation:**
Brute force defines the upper bound of recall (1.0) and the lower bound of QPS. Every ANNS algorithm is measured against it. We use FAISS IndexFlatL2 as our implementation.

**When to use:**
Only practical for small datasets (N < 100K) or when exact results are required regardless of cost.

**Reference:**
- Johnson, Douze & Jégou, "Billion-Scale Similarity Search with GPUs," IEEE Big Data, 2021. [arXiv:1702.08734](https://arxiv.org/abs/1702.08734)

---

### 2. HNSW — Hierarchical Navigable Small World (Baseline)

**Core idea:**
HNSW [2] constructs a hierarchical multi-layer proximity graph. Upper layers contain long-range edges connecting randomly selected nodes for fast navigation. Lower layers contain short range edges for precise local search. At query time, the algorithm enters at the top layer and greedily descends,
narrowing the candidate set at each layer until a termination condition is met.

**Why it outperforms NSW:**
NSW has poly-logarithmic search complexity. HNSW fixes the upper bound of each vertex's neighbor count per layer, reducing search complexity to logarithmic O(log N) [1, 2].

**Key parameters:**
| Parameter | Role | Our Setting |
|---|---|---|
| `M` | Bidirectional links per node — controls graph connectivity | 16 |
| `ef_construction` | Candidate list size during build — controls index quality | 200 |
| `ef_search` | Candidate list size during query — controls recall/speed tradeoff | 50 |

**Formal complexity:**
```
Query time:  O(log N)
Build time:  O(N × log N)
Memory:      O(N × M)
```

**Limitation for our work:**
Standard HNSW has no concept of attributes. It treats all 1M vectors as a single undifferentiated graph. Query routing is determined entirely by graph structure which is making the index opaque and uninterpretable.

---

### 3. Clustered HNSW — Our System

**Core idea:**
We partition the full dataset into K attribute-based clusters at build time.
Each cluster is represented by a centroid vector and indexed with its own
dedicated HNSW graph. At query time, the query is compared against all K
centroids, routed to the nearest cluster, and searched only within that
cluster's HNSW index.

**Two-phase design:**

```
─── Offline Phase (Build) ──────────────────────────────────────

  1M vectors → K-Means (K clusters)
                  │
                  ├── Cluster 0   → centroid_0    +  HNSW_0    (~N/K vectors)
                  ├── Cluster 1   → centroid_1    +  HNSW_1    (~N/K vectors)
                  ├── ...
                  └── Cluster K-1 → centroid_K-1  +  HNSW_K-1  (~N/K vectors)

─── Online Phase (Query) ───────────────────────────────────────

  Query → compare against K centroids (flat scan)  [O(K × d)]
        → route to nearest cluster                  [O(1)]
        → search that cluster's HNSW index          [O(log(N/K))]
        → return Top-K results
```

**Key parameters:**
| Parameter | Role |
|---|---|
| `K` | Number of clusters — determined by attributes or K-Means |
| `TOP_CLUSTERS` | Clusters searched per query — controls recall/speed tradeoff |
| `M`, `ef_construction`, `ef_search` | Per-cluster HNSW parameters |

**Formal complexity:**
```
Query time:  O(K × d) + O(log(N/K))
Build time:  O(N × K) + K × O((N/K) × log(N/K))
Memory:      O(N × M)
```

**Real-time insertion:**
A background thread monitors insertions and triggers re-clustering every N
new vectors. New indexes are built and swapped atomically — zero query
interruption during re-indexing.

**Understandability advantage:**
| Property | Standard HNSW | Clustered HNSW (Ours) |
|---|---|---|
| Query routing | Opaque graph traversal | Traceable — cluster ID logged per query |
| Failure diagnosis | Global recall only | Per-cluster recall measurable |
| Worst-case latency | Unpredictable | Bounded by largest cluster size |

---

### 4. FAISS-IVF — Inverted File Index

**Core idea:**
IVF [3] partitions the dataset into K Voronoi cells using K-Means. Each cell
is a posting list of vectors assigned to that centroid. At query time, the
query is compared against all centroids and the `nprobe` closest cells are
searched using flat exact L2 search within each posting list.

**Relationship to our work:**
FAISS-IVF is the closest existing system to our approach:

```
FAISS-IVF:        partition → flat search inside each cell     (exact within cell)
Clustered HNSW:   partition → HNSW graph inside each cluster   (approximate, faster)
```

Our system replaces the flat posting list with a dedicated HNSW index per
cluster — trading a small amount of within-cluster recall for higher QPS.

**Key parameters:**
| Parameter | Role |
|---|---|
| `nlist` | Number of Voronoi cells |
| `nprobe` | Cells searched per query — controls recall/speed tradeoff |

**Formal complexity:**
```
Query time:  O(K × d) + O(nprobe × N/K)
Build time:  O(N × K)
Memory:      O(N × d)
```
---

### 5. ScaNN — Scalable Nearest Neighbors (Google)

**Core idea:**
ScaNN [4] combines two techniques. SOAR (Spill trees with
Orthogonality-Amplified Residuals) partitions the space using overlapping
trees, where each tree is optimized to cover the failure modes of others.
Anisotropic Vector Quantization (AVQ) compresses vectors by prioritizing
accurate distance estimation for likely nearest neighbors — rather than
minimizing average reconstruction error as standard PQ does.

**Why it matters:**
ScaNN consistently achieves the highest QPS at high recall on ANN-Benchmarks,
making it the toughest performance competitor for any new ANN system.

---

### 6. Annoy — Approximate Nearest Neighbors Oh Yeah (Spotify)

**Core idea:**
Annoy [5] builds a forest of random projection trees. Each tree is
constructed by repeatedly choosing a random hyperplane that splits the
current vector set into two halves, recursing until each leaf contains
fewer than K vectors. At query time, all trees are traversed and candidate
sets are merged and ranked by distance.

**Key parameters:**
| Parameter | Role |
|---|---|
| `n_trees` | Number of trees — more trees = higher recall, larger index |
| `search_k` | Nodes inspected per query — controls recall/speed tradeoff |

**Limitation:**
Annoy does not support incremental insertions — the entire forest must be
rebuilt when new vectors are added. This is a fundamental limitation that
our background re-clustering directly addresses.

---

## Summary Comparison

| Algorithm | Family | Recall@High | Incremental Insert | Attribute-Aware | Build Time |
|---|---|---|---|---|---|
| Brute Force | Exact | 1.00 | ✅ | ❌ | None |
| HNSW (baseline) | Graph (RNG-based) | ~0.97 | ⚠️ Degrades | ❌ | High |
| **Clustered HNSW (ours)** | Graph + Partition | TBD | ✅ re-clustering | ✅ | Low |
| FAISS-IVF | Partition | ~0.95 | ❌ Rebuild | ❌ | Medium |
| ScaNN | Partition + Quant | ~0.99 | ❌ Rebuild | ❌ | Medium |
| Annoy | Tree | ~0.90 | ❌ Rebuild | ❌ | Low |

> Results marked TBD will be updated upon completion of experiments on the NVIDIA DGX Spark.

---

## Evaluation Metrics

Following Wang et al. [1] and Aumüller et al. [6]:

| Metric | Definition | Formula |
|---|---|---|
| **Recall@k** | Fraction of true k-NNs returned | `|R ∩ R̃| / k` |
| **QPS** | Queries processed per second | `num_queries / search_time` |
| **Build time** | Time to construct the index | seconds |
| **Index size** | Memory footprint | MB |

The primary evaluation plot is **Recall vs QPS** — a curve where
higher and further right is strictly better.

---

## References

[1] M. Wang, X. Xu, Q. Yue, Y. Wang, "A Comprehensive Survey and Experimental
Comparison of Graph-Based Approximate Nearest Neighbor Search,"
PVLDB, 14(11): 1964–1978, 2021.
[Link](https://www.vldb.org/pvldb/vol14/p1964-wang.pdf)

[2] Y. A. Malkov, D. A. Yashunin, "Efficient and Robust Approximate Nearest
Neighbor Search Using Hierarchical Navigable Small World Graphs,"
IEEE TPAMI, 42(4): 824–836, 2020.
[arXiv:1603.09320](https://arxiv.org/abs/1603.09320)

[3] J. Johnson, M. Douze, H. Jégou, "Billion-Scale Similarity Search with GPUs,"
IEEE Big Data, 7(3): 535–547, 2021.
[arXiv:1702.08734](https://arxiv.org/abs/1702.08734)

[4] R. Guo et al., "Accelerating Large-Scale Inference with Anisotropic Vector
Quantization," ICML 2020.
[arXiv:1908.10396](https://arxiv.org/abs/1908.10396)

[5] E. Bernhardsson, Annoy: Approximate Nearest Neighbors in C++/Python.
[GitHub](https://github.com/spotify/annoy)

[6] M. Aumüller, E. Bernhardsson, A. Faithfull, "ANN-Benchmarks: A Benchmarking
Tool for Approximate Nearest Neighbor Algorithms,"
Information Systems, 87, 2020.
[DOI:10.1016/j.is.2019.02.006](https://doi.org/10.1016/j.is.2019.02.006)

[7] G. Pandit, M. Röder, A. Ngonga Ngomo, "Evaluating Approximate Nearest
Neighbour Search Systems on Knowledge Graph Embeddings," ESWC 2025.
[Link](https://papers.dice-research.org/2025/ESWC_ANN_Benchmark/public.pdf)
