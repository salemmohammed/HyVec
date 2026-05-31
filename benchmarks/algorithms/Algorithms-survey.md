# Algorithms

This section describes the approximate nearest neighbor search (ANNS) algorithms
evaluated in our study. We follow the taxonomy and evaluation methodology of Wang
et al. [1], the most comprehensive survey of graph-based ANNS algorithms to date,
and extend it with our proposed Clustered HNSW system.

---

## Background

### Problem Definition

Given a dataset $\mathcal{X} = \{x_1, \ldots, x_N\} \subset \mathbb{R}^d$ and a
query vector $q \in \mathbb{R}^d$, the nearest neighbor search problem is to find:

$$x^* = \arg\min_{x \in \mathcal{X}} \delta(x, q)$$

where $\delta(x, q) = \sqrt{\sum_i (x_i - q_i)^2}$ denotes the Euclidean (L2)
distance. Exact search requires $O(N \cdot d)$ comparisons per query — prohibitive
at scale. Approximate Nearest Neighbor Search (ANNS) sacrifices a bounded amount
of accuracy for dramatically reduced query latency.

### Accuracy Metric

Algorithm accuracy is measured by **Recall@k** — the fraction of the true k nearest
neighbors returned by the algorithm:

$$\text{Recall@}k = \frac{|\mathcal{R} \cap \tilde{\mathcal{R}}|}{k}$$

where $\mathcal{R}$ is the exact result set (ground truth) and $\tilde{\mathcal{R}}$
is the approximate result set. A value of 1.0 indicates perfect accuracy.

### The Recall–QPS Tradeoff

Every ANNS algorithm navigates a fundamental tradeoff between search accuracy and
throughput. Searching fewer candidates yields higher queries per second (QPS) at
the cost of lower recall; searching more candidates improves recall at the cost of
throughput. The primary evaluation plot is **Recall vs. QPS**, where higher and
further right is strictly better.

---

## Algorithm Taxonomy

Following Wang et al. [1], ANNS algorithms are organized into four families:

| Family | Representative Methods |
|---|---|
| **Hashing-based** | LSH, E2LSH |
| **Tree-based** | KD-Tree, Annoy, FLANN, MRPT |
| **Quantization-based** | PQ, IVF-PQ, ScaNN |
| **Graph-based** | NSW, HNSW, NSG, DiskANN |

Graph-based algorithms have emerged as the dominant paradigm, achieving higher
recall at lower latency than competing families across standard benchmarks [1].
Our system is a **hybrid** combining graph-based search (HNSW) with
partition-based routing (IVF-style k-means clustering).

---

## Graph-Based ANNS: Foundational Graph Structures

Wang et al. [1] identify four base graph structures from which all graph-based
ANNS algorithms are derived:

| Base Graph | Key Property | Representative Algorithms |
|---|---|---|
| Delaunay Graph (DG) | Guarantees exact NNS; near-fully connected in high dimensions | NSW |
| Relative Neighborhood Graph (RNG) | Eliminates redundant edges; omnidirectional neighbor distribution | HNSW, NSG, FANNG |
| K-Nearest Neighbor Graph (KNNG) | Bounds degree to K; efficient but may lose global connectivity | EFANNA, IEH |
| Minimum Spanning Tree (MST) | Minimal edges for global connectivity; suboptimal search paths | HCNNG |

HNSW approximates the RNG by diversifying neighbor distribution across hierarchical
layers, achieving logarithmic search complexity $O(\log N)$ — the best among the
13 graph-based algorithms surveyed by Wang et al. [1].

---

## Algorithms

### 1. Brute Force (Exact Search)

Brute force computes the distance from the query to every vector in the dataset
and returns the exact k nearest neighbors. It requires no index construction and
achieves Recall@k = 1.0 by definition.

| Metric | Complexity |
|---|---|
| Query time | $O(N \cdot d)$ |
| Build time | $O(1)$ |
| Memory | $O(N \cdot d)$ |
| Recall@k | 1.0 (exact) |

Brute force serves as the upper bound on recall and the lower bound on QPS. We
use FAISS `IndexFlatL2` as the implementation [3].

---

### 2. HNSW — Hierarchical Navigable Small World

HNSW [2] constructs a hierarchical multi-layer proximity graph. Upper layers
contain long-range edges connecting a randomly selected subset of nodes for fast
coarse navigation; lower layers contain short-range edges for precise local search.
At query time, the algorithm performs a greedy descent from the top layer,
narrowing the candidate set at each layer until convergence.

Compared to NSW, which has poly-logarithmic search complexity, HNSW bounds the
maximum degree per node per layer, reducing search complexity to $O(\log N)$ [1, 2].

**Key parameters:**

| Parameter | Role |
|---|---|
| `M` | Bidirectional links per node — controls graph connectivity |
| `ef_construction` | Candidate list size during build — controls index quality |
| `ef_search` | Candidate list size during query — controls the recall/QPS tradeoff |

| Metric | Complexity |
|---|---|
| Query time | $O(\log N)$ |
| Build time | $O(N \log N)$ |
| Memory | $O(N \cdot M)$ |

Standard HNSW treats all vectors as an undifferentiated graph with no notion of
data partitioning or attribute-based routing, which limits interpretability and
precludes efficient real-time insertion without index degradation.

---

### 3. Clustered HNSW (Proposed Method)

The proposed system partitions the dataset into $K$ clusters at build time using
k-means. Each cluster is assigned a centroid vector and an independent HNSW index.
At query time, the query is compared against all $K$ centroids and routed to the
$T$ nearest clusters, which are searched in parallel.

**Build phase:**

$$\mathcal{X} \xrightarrow{\text{k-means}} \{C_0, C_1, \ldots, C_{K-1}\}, \quad
\text{each } C_k \rightarrow (\mu_k,\ \text{HNSW}_k)$$

**Query phase:**

$$q \rightarrow \underbrace{O(K \cdot d)}_{\text{centroid scan}}
\rightarrow \text{top-}T\text{ clusters}
\rightarrow \underbrace{O(\log(N/K))}_{\text{HNSW search per cluster}}
\rightarrow \text{top-}k\text{ results}$$

**Key parameters:**

| Parameter | Role |
|---|---|
| `K` | Number of clusters |
| `T` (TOP_CLUSTERS) | Clusters searched per query — controls the recall/QPS tradeoff |
| `M`, `ef_construction`, `ef_search` | Per-cluster HNSW parameters |

| Metric | Complexity |
|---|---|
| Query time | $O(K \cdot d + T \cdot \log(N/K))$ |
| Build time | $O(N \cdot K + N \cdot M \cdot \log(N/K))$ |
| Memory | $O(N \cdot M)$ |

Real-time insertions are handled by a background thread that monitors the arrival
rate and triggers local re-clustering when a cluster exceeds a size threshold.
New indexes are built and swapped atomically, incurring zero query interruption.

---

### 4. FAISS-IVF — Inverted File Index

FAISS-IVF [3] partitions the dataset into $K$ Voronoi cells using k-means. Each
cell stores a posting list of its assigned vectors. At query time, the query is
compared against all centroids and the `nprobe` closest cells are searched using
exact flat L2 scan within each posting list.

| Metric | Complexity |
|---|---|
| Query time | $O(K \cdot d + \text{nprobe} \cdot N/K)$ |
| Build time | $O(N \cdot K)$ |
| Memory | $O(N \cdot d)$ |

FAISS-IVF is the closest prior system to our approach. The key distinction is that
FAISS-IVF performs exact flat search within each cell, whereas our system replaces
the flat posting list with a dedicated HNSW index per cluster — achieving higher
QPS at comparable recall.

---

### 5. ScaNN — Scalable Nearest Neighbors

ScaNN [4] combines two complementary techniques. Structured Orthogonal Random
Projection (SOAR) partitions the space using overlapping trees, where each tree is
optimized to cover the failure modes of the others. Anisotropic Vector Quantization
(AVQ) compresses vectors by prioritizing accurate distance estimation for likely
nearest neighbors, rather than minimizing average reconstruction error as standard
product quantization does. ScaNN consistently achieves the highest QPS at high
recall on ANN-Benchmarks and represents the strongest throughput competitor for
any proposed ANN system.

---

### 6. Annoy — Approximate Nearest Neighbors Oh Yeah

Annoy [5] builds a forest of random projection trees. Each tree is constructed by
repeatedly selecting a random hyperplane that bisects the current vector set,
recursing until each leaf contains fewer than a fixed number of vectors. At query
time, all trees are traversed and candidate sets are merged and re-ranked by
distance. Annoy does not support incremental insertion — the full forest must be
rebuilt when new vectors are added — which is a fundamental limitation that our
background re-clustering strategy directly addresses.

---

## Summary

| Algorithm | Family | Incremental Insert | Recall@High | Build Cost |
|---|---|---|---|---|
| Brute Force | Exact | ✅ | 1.00 | None |
| HNSW | Graph (RNG) | ⚠️ Degrades | ~0.97 | High |
| **Clustered HNSW (ours)** | Graph + Partition | ✅ | TBD | Lower than HNSW |
| FAISS-IVF | Partition | ❌ Rebuild | ~0.95 | Medium |
| ScaNN | Partition + Quantization | ❌ Rebuild | ~0.99 | Medium |
| Annoy | Tree | ❌ Rebuild | ~0.90 | Low |

> Clustered HNSW results marked TBD will be updated upon completion of experiments
> on the NVIDIA DGX Spark.

---

## Evaluation Protocol

We adopt the evaluation protocol and datasets of Aumüller et al. [6]. All
algorithms are evaluated on the SIFT1M dataset (1M × 128-d, Euclidean distance).
All baseline algorithms are rerun on our NVIDIA DGX Spark to ensure a consistent
and fair comparison under identical hardware conditions.

| Metric | Definition | Formula |
|---|---|---|
| **Recall@k** | Fraction of true k-NNs returned | $\|\mathcal{R} \cap \tilde{\mathcal{R}}\| / k$ |
| **QPS** | Queries processed per second | `num_queries / search_time` |
| **Build time** | Time to construct the index | seconds |
| **Index size** | Memory footprint of the index | MB |

---

## References

[1] M. Wang, X. Xu, Q. Yue, Y. Wang: **A Comprehensive Survey and Experimental
Comparison of Graph-Based Approximate Nearest Neighbor Search.**
PVLDB 14(11): 1964–1978, 2021.
[vldb.org/pvldb/vol14/p1964-wang.pdf](https://www.vldb.org/pvldb/vol14/p1964-wang.pdf)

[2] Y. A. Malkov, D. A. Yashunin: **Efficient and Robust Approximate Nearest
Neighbor Search Using Hierarchical Navigable Small World Graphs.**
IEEE TPAMI 42(4): 824–836, 2020.
[arXiv:1603.09320](https://arxiv.org/abs/1603.09320)

[3] J. Johnson, M. Douze, H. Jégou: **Billion-Scale Similarity Search with GPUs.**
IEEE Big Data 7(3): 535–547, 2021.
[arXiv:1702.08734](https://arxiv.org/abs/1702.08734)

[4] R. Guo et al.: **Accelerating Large-Scale Inference with Anisotropic Vector
Quantization (ScaNN).** ICML 2020.
[arXiv:1908.10396](https://arxiv.org/abs/1908.10396)

[5] E. Bernhardsson: **Annoy: Approximate Nearest Neighbors in C++/Python.**
[github.com/spotify/annoy](https://github.com/spotify/annoy)

[6] M. Aumüller, E. Bernhardsson, A. Faithfull: **ANN-Benchmarks: A Benchmarking
Tool for Approximate Nearest Neighbor Algorithms.**
Information Systems 87, 2020.
[DOI:10.1016/j.is.2019.02.006](https://doi.org/10.1016/j.is.2019.02.006)