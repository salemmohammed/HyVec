# Algorithms

This folder contains implementations of the approximate nearest neighbor (ANN) algorithms evaluated in our benchmarks. Each algorithm represents a different indexing strategy with distinct tradeoffs between search speed, recall, build time, and memory usage.

---

## Algorithm Families

ANN algorithms generally fall into four families:

| Family | Core Idea | Example Systems |
|---|---|---|
| **Graph-based** | Build a proximity graph; traverse it greedily at query time | HNSW, NSG, NGT |
| **Partition-based** | Divide vectors into clusters; search only relevant partitions | FAISS-IVF, our system |
| **Tree-based** | Recursively split the space into subspaces | Annoy, FLANN, MRPT |
| **Quantization-based** | Compress vectors into compact codes; approximate distances | PQ, IVFPQ |

Our system belongs to both the **graph-based** and **partition-based** families — it partitions vectors into attribute-based clusters, then builds a dedicated HNSW index per cluster.

---

## Algorithms

---

### Brute Force (Exact Search)

**What it does:**
Compares the query vector against every vector in the dataset and returns the exact nearest neighbors.

**Why it matters:**
It is the gold standard — recall is always 1.0 by definition. Every ANN algorithm is measured against it.

**Complexity:**
- Query time: `O(N × d)` where N = dataset size, d = dimensions
- Build time: none
- Memory: `O(N)`

**Limitation:**
Exact search is impractical at scale. On SIFT1M (1M × 128-dim), a brute force search processes 128 million multiplications per query.

**References:**
- FAISS `IndexFlatL2` — exact L2 brute force implementation
- [FAISS GitHub](https://github.com/facebookresearch/faiss)

---

### HNSW — Hierarchical Navigable Small World (Baseline)

**What it does:**
Builds a multi-layer proximity graph where upper layers contain long-range connections for fast navigation and lower layers contain short-range connections for precise search. At query time, the algorithm enters at the top layer and greedily descends toward the nearest neighbor.

**Why it matters:**
HNSW is the current state-of-the-art for in-memory ANN search. It achieves near-logarithmic search complexity with high recall and consistently ranks at the Pareto-optimal front on ANN-Benchmarks.

**Key parameters:**
| Parameter | Role |
|---|---|
| `M` | Number of bidirectional links per node — controls graph connectivity |
| `ef_construction` | Size of the candidate list during index build — controls build quality |
| `ef_search` | Size of the candidate list during query — controls recall/speed tradeoff |

**Complexity:**
- Query time: `O(log N)` — logarithmic in dataset size
- Build time: `O(N × log N)`
- Memory: `O(N × M)`

**Our baseline configuration:**
```
M = 16, ef_construction = 200, ef_search = 50
Recall@1 = 0.9686, QPS = 7,443, Build time = 1013s
```

**References:**
- Malkov & Yashunin, "Efficient and Robust Approximate Nearest Neighbor Search Using Hierarchical Navigable Small World Graphs," IEEE TPAMI, 2020. [arXiv:1603.09320](https://arxiv.org/abs/1603.09320)
- [hnswlib GitHub](https://github.com/nmslib/hnswlib)
- [Pinecone — HNSW explained](https://www.pinecone.io/learn/series/faiss/hnsw/)

---

### Clustered HNSW — Our System

**What it does:**
Partitions the full dataset into K attribute-based clusters at build time, representing each cluster by a centroid vector. At query time, the query is compared against all K centroids, routed to the nearest cluster, and searched within that cluster's dedicated HNSW index only.

**Why it matters:**
By restricting search to one cluster (~1,000 vectors) instead of the full dataset (1,000,000 vectors), the system reduces the effective search space while maintaining a dedicated HNSW graph per cluster for high recall within each partition.

**Key parameters:**
| Parameter | Role |
|---|---|
| `K` | Number of clusters — determined by attribute count or K-Means |
| `TOP_CLUSTERS` | Number of clusters searched per query — controls recall/speed tradeoff |
| `M`, `ef` | Per-cluster HNSW parameters |

**Two-phase design:**
```
Offline (build):
  1M vectors → K-Means → K clusters
  Each cluster → 1 centroid + 1 HNSW index

Online (query):
  Query → compare against K centroids → nearest cluster
        → search that cluster's HNSW index only
        → return Top-K results
```

**Real-time support:**
A background thread monitors insertions and triggers re-clustering every N new vectors, rebuilding indexes and performing an atomic swap with zero query interruption.

**References:**
- Jégou et al., "Product Quantization for Nearest Neighbor Search," IEEE TPAMI, 2011. [Link](http://corpus-texmex.irisa.fr/) *(IVF partitioning concept)*
- Malkov & Yashunin, 2020. [arXiv:1603.09320](https://arxiv.org/abs/1603.09320) *(HNSW base algorithm)*
- Aslam et al., "Clustered Hybrid Search," [GitHub](https://github.com/AdeelAslamUnimore/Clustered_Hybrid_Search)

---

### FAISS-IVF — Inverted File Index

**What it does:**
Partitions vectors into K Voronoi cells using K-Means. At query time, the query is compared against all centroids and the `nprobe` closest cells are searched using flat (exact) L2 search within each cell.

**Why it matters:**
FAISS-IVF is the standard partition-based baseline and the closest existing system to our approach. The key difference is that FAISS-IVF uses flat search within each partition while our system uses a dedicated HNSW graph per partition.

**Key parameters:**
| Parameter | Role |
|---|---|
| `nlist` | Number of Voronoi cells (partitions) |
| `nprobe` | Number of cells searched per query — controls recall/speed tradeoff |

**Complexity:**
- Query time: `O(nprobe × N/nlist)` — sub-linear when `nprobe << nlist`
- Build time: `O(N × K)` for K-Means

**References:**
- Johnson, Douze & Jégou, "Billion-Scale Similarity Search with GPUs," IEEE Big Data, 2021. [arXiv:1702.08734](https://arxiv.org/abs/1702.08734)
- [FAISS GitHub](https://github.com/facebookresearch/faiss)
- [FAISS official wiki](https://github.com/facebookresearch/faiss/wiki)
- [Pinecone — FAISS tutorial](https://www.pinecone.io/learn/series/faiss/faiss-tutorial/)

---

### ScaNN — Scalable Nearest Neighbors (Google)

**What it does:**
Uses anisotropic vector quantization (AVQ) to compress vectors in a way that prioritizes accurate distance estimation for likely nearest neighbors rather than minimizing average reconstruction error. Combined with SOAR (Spill trees with Orthogonality-Amplified Residuals) for partitioning.

**Why it matters:**
ScaNN consistently achieves the highest QPS at high recall on ANN-Benchmarks, making it the toughest competitor for any new ANN system.

**References:**
- Guo et al., "Accelerating Large-Scale Inference with Anisotropic Vector Quantization," ICML 2020. [arXiv:1908.10396](https://arxiv.org/abs/1908.10396)
- Sun et al., "SOAR: Improved Indexing for Approximate Nearest Neighbor Search," NeurIPS 2023.
- [ScaNN GitHub](https://github.com/google-research/google-research/tree/master/scann)
- [Google AI Blog — ScaNN](https://ai.googleblog.com/2020/07/announcing-scann-efficient-vector.html)

---

### Annoy — Approximate Nearest Neighbors Oh Yeah (Spotify)

**What it does:**
Builds a forest of random projection trees. Each tree recursively splits the vector space by random hyperplanes until leaf nodes contain fewer than K vectors. At query time, all trees are traversed and candidate sets are merged and ranked.

**Why it matters:**
Annoy was one of the first practical ANN libraries and powers Spotify's music recommendations. It is memory-efficient and supports static file-based indexes shared across processes.

**Key parameters:**
| Parameter | Role |
|---|---|
| `n_trees` | Number of trees — more trees = higher recall, larger index |
| `search_k` | Nodes inspected per query — controls recall/speed tradeoff |

**Limitation:**
Annoy does not support incremental insertions — the index must be rebuilt from scratch when new vectors are added. This is a key weakness that our real-time re-clustering directly addresses.

**References:**
- [Annoy GitHub](https://github.com/spotify/annoy)
- Bernhardsson, "Nearest Neighbors and Vector Models," [Blog](https://erikbern.com/2015/10/01/nearest-neighbors-and-vector-models-part-2-how-to-search-in-high-dimensional-spaces.html)

---

## Summary Comparison

| Algorithm | Type | Incremental Insert | Attribute-Aware | Recall@High | QPS@High Recall |
|---|---|---|---|---|---|
| Brute Force | Exact | ✅ | ❌ | 1.00 | Very low |
| HNSW (baseline) | Graph | ⚠️ Degrades | ❌ | ~0.97 | High |
| **Clustered HNSW (ours)** | Graph + Partition | ✅ w/ re-clustering | ✅ | TBD | TBD |
| FAISS-IVF | Partition | ❌ Rebuild needed | ❌ | ~0.95 | High |
| ScaNN | Partition + Quant | ❌ | ❌ | ~0.99 | Highest |
| Annoy | Tree | ❌ Rebuild needed | ❌ | ~0.90 | Medium |

> Results will be updated upon completion of experiments on the NVIDIA DGX Spark.

---

## Key Metrics

| Metric | Definition |
|---|---|
| **Recall@k** | Fraction of true k-nearest neighbors returned |
| **QPS** | Queries per second — throughput |
| **Build time** | Time to construct the index from scratch |
| **Index size** | Memory footprint of the index |

---

## Further Reading

| Resource | Type |
|---|---|
| [ANN-Benchmarks](https://ann-benchmarks.com) | Live leaderboard — recall vs QPS for all algorithms |
| [Pandit et al., ESWC 2025](https://papers.dice-research.org/2025/ESWC_ANN_Benchmark/public.pdf) | 22 ANN systems compared on Knowledge Graph Embeddings |
| [Pinecone Learning Center](https://www.pinecone.io/learn/) | Practical tutorials on all ANN algorithms |
| [Papers With Code — ANN](https://paperswithcode.com/task/approximate-nearest-neighbor-search) | Latest research ranked by benchmark results |
| [Comprehensive ANN Guide](https://towardsdatascience.com/comprehensive-guide-to-approximate-nearest-neighbors-algorithms-8b94f057d6b6/) | Trabelsi, Towards Data Science — intuition + code |
| [Graph-based ANN Overview](https://medium.com/@juanjuango321/some-popular-approximate-nearest-neighbor-graph-based-search-algorithms-10ca1cd3aa5) | NSW, HNSW, NSG explained with diagrams |
