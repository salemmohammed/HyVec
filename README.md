# Real-Time Hybrid Clustered Vector Search

## Clustered Attributed Vector Search
We partition the base dataset into K spatial clusters [11] using MiniBatchKMeans. Each partion will store one centroid per cluster. At query time, the query vector is compared against all centroids to identify the nearest clusters. This is the routing step, inspired by IVF in FAISS [2]. Rather than scanning the full 1M vectors, only the vectors within the top K clusters are searched. Within each cluster, we build an independent HNSW index [1], a hierarchical graph where each vector is connected to its nearest neighbors across multiple layers. Search traverses this graph greedily, moving layer by layer toward the query vector. This replaces the flat exhaustive scan used in standard IVF, giving higher recall at the same cluster probe budget.

---

## Authors

Salem Alqahtani, Adeel Aslam, Khaled Mahmoud, Badr Asiri, Osama Al-Senani,
Faisal Azib, Abdullatif Hadi, Faisal Awad, and Omar Abdulaziz

## Vector DB pipeline  

[Interactive pipeline →](https://yourname.github.io/my-vector-search)

### Step-by-Step

**1. Query Vector** — A 128-dimensional float vector representing the search query.  
> Example: a SIFT descriptor of an image patch `[0.1, 0.4, ..., 0.9]`

**2. Distance Metric — L2 Euclidean** — Measures similarity between vectors using squared Euclidean distance.  
> Example: `dist(q, x) = √Σ(qᵢ - xᵢ)²`  
> Used by both HNSW [1] and FAISS [2] as the default metric for SIFT vectors.

**3. Coarse Routing — IVF-style K-Means** — The query is compared against K=1000 precomputed centroids. The nearest N clusters are selected, reducing the search space from 1M to ~N×1000 vectors.  
> Example: `TOP_CLUSTERS=20` → search only 20,000 vectors instead of 1,000,000  
> Inspired by the Inverted File Index (IVF) in FAISS [2] and Ada-IVF [3].

**4. Fine Search — HNSW per Cluster** — Within each selected cluster, a dedicated HNSW index [1] performs greedy graph traversal to find the nearest neighbors. Each cluster has its own independent graph built with `M=16, ef_construction=200`.  
> Example: cluster #42 has 1,200 vectors → HNSW traverses its graph layer by layer, returning top-k candidates in ~0.1ms.  
> HNSW outperforms flat IVF scan at small cluster sizes (~1,000 vectors) [4].

**5. Merge & Rank** — Candidates from all probed clusters are collected, sorted by L2 distance, and the global top-k are returned.  
> Example: `TOP_CLUSTERS=20` returns 20×k candidates → sorted → top-k returned.

**6. Top-k Results** — The k nearest neighbors to the query vector across the entire dataset.  
> Example: `k=1` → Recall@1 = 0.9689 at `TOP_CLUSTERS=20`, matching the full-index HNSW baseline.

---

### References

[1] Y. Malkov and D. Yashunin, "Efficient and Robust Approximate Nearest Neighbor Search Using Hierarchical Navigable Small World Graphs," *IEEE TPAMI*, 2020.  
[2] J. Johnson, M. Douze, and H. Jégou, "Billion-scale similarity search with GPUs," *IEEE Trans. Big Data*, 2021. [GitHub](https://github.com/facebookresearch/faiss)  
[3] J. Mohoney et al., "High-Throughput Vector Similarity Search in Knowledge Graphs," *SIGMOD*, 2023.  
[4] J. Zhu et al., "An Experimental Evaluation of Hybrid Querying on Vectors," *PVLDB*, 2025.

## Research Questions

| # | Question | Theme |
|---|---|---|
| RQ1 | Can representing each partition of a million-scale vector dataset by a centroid vector, and routing queries to the nearest partition, improve query throughput while preserving recall@1 ≥ 0.95 compared to searching the full unpartitioned dataset? | **Efficiency** |
| RQ2 | When a vector dataset has complete attributes, partial attributes, no attributes, imbalanced clusters, or spatially overlapping clusters, which partitioning strategy preserves the best recall@1 and QPS, and what are the tradeoffs? | **Partitioning Strategy Under Varying Attribute Conditions** |
| RQ3 | As new vectors are continuously inserted into a live index, how does recall and QPS degrade over time without re-clustering, and what is the degradation rate relative to insertion volume? | **Real-Time Degradation** |
| RQ4 | Can background re-clustering with atomic index swapping fully restore recall and QPS to pre-insertion levels with zero query interruption, and what is the measurable cost of re-clustering itself? | **Real-Time Recovery** |
| RQ5 | Does the clustered approach scale sub-linearly with dataset growth compared to standard HNSW, and at what dataset size does clustering yield the greatest benefit? | **Scalability** |
| RQ6 | Does attribute based clustering make vector search more understandable than monolithic HNSW by enabling three capabilities that standard HNSW cannot provide: (1) tracing exactly which cluster a query was routed to, (2) identifying which individual cluster is causing recall degradation, and (3) predicting worst-case query latency from cluster size alone without running the full system? | **Understandability** |
---


## Core Idea

Standard HNSW traverses a subset of the full graph on every query, but that graph contains all 1,000,000 vectors, and its search space grows as the dataset scales. This project partitions vectors into attribute-based clusters, so each query is routed to one small cluster only making the search space bounded and independent of total dataset size.

```
Standard HNSW:
  Query --> traverses a small subset of the 1M graph
        --> controlled by ef_search parameter
        --> visits O(ef × log N) nodes approximately
        --> search space grows with dataset size N

Clustered HNSW (this project):

  Offline (build time):
    1,000,000 vectors  --> partitioned into K clusters
    Each cluster       --> represented by 1 centroid vector
    Each cluster       --> gets its own small HNSW index (~1,000 vectors)

  Online (query time):
    Query --> compared against K centroids only
          --> routed to nearest cluster
          --> HNSW runs on that cluster only (~1,000 vectors)
          --> search space bounded by cluster size
          --> independent of total dataset size N
```

---

## System Architecture

```mermaid
flowchart TD
    A([Raw Vectors\n1M × 128-dim]) --> B[K-Means Clustering\nK = 1000]
    B --> C{Cluster Assignment\nAttribute ID 0–999}
    C --> D1[Cluster 0\nHNSW Index]
    C --> D2[Cluster 1\nHNSW Index]
    C --> D3[· · ·]
    C --> D4[Cluster 999\nHNSW Index]

    E([Query Vector]) --> F[Find Nearest Centroid]
    F --> G{Route to\nCluster ID}
    G --> D1
    G --> D2
    G --> D4

    D1 --> H([Top-K Results])
    D2 --> H
    D4 --> H

    style A fill:#4C9BE8,color:#fff,stroke:#2563EB
    style E fill:#4C9BE8,color:#fff,stroke:#2563EB
    style H fill:#16A34A,color:#fff,stroke:#15803D
    style B fill:#7C3AED,color:#fff,stroke:#6D28D9
    style F fill:#7C3AED,color:#fff,stroke:#6D28D9
    style C fill:#D97706,color:#fff,stroke:#B45309
    style G fill:#D97706,color:#fff,stroke:#B45309
```

---

## Standard HNSW vs Clustered HNSW

```mermaid
flowchart LR
    subgraph Standard["Standard HNSW"]
        direction TB
        Q1([Query]) --> I1[Traverse subset of 1M graph]
        I1 --> R1([Result])
    end

    subgraph Clustered["Clustered HNSW (This Work)"]
        direction TB
        Q2([Query]) --> C1[Compare against K centroids]
        C1 --> C2[Route to nearest cluster]
        C2 --> C3[Search ~1,000 vectors only]
        C3 --> R2([Result])
    end

    style Standard fill:#FEE2E2,stroke:#EF4444
    style Clustered fill:#DCFCE7,stroke:#16A34A
    style Q1 fill:#3B82F6,color:#fff
    style Q2 fill:#3B82F6,color:#fff
    style R1 fill:#16A34A,color:#fff
    style R2 fill:#16A34A,color:#fff
```
---

## Real-Time Insertion & Re-Clustering

```mermaid
flowchart TD
    A([New Vector\nInserted]) --> B[Assign to Nearest Cluster]
    B --> C{N insertions\nreached?}
    C -- No --> D[Insert into\nexisting cluster index]
    C -- Yes --> E[Background Thread\nRe-runs K-Means]
    E --> F[Build New\nHNSW Indexes]
    F --> G[Atomic Swap\nold ➜ new index]
    G --> H([Search Continues\nUninterrupted])
    D --> H

    style A fill:#4C9BE8,color:#fff,stroke:#2563EB
    style H fill:#16A34A,color:#fff,stroke:#15803D
    style E fill:#7C3AED,color:#fff,stroke:#6D28D9
    style F fill:#7C3AED,color:#fff,stroke:#6D28D9
    style G fill:#D97706,color:#fff,stroke:#B45309
    style C fill:#D97706,color:#fff,stroke:#B45309
```

---

## Algorithm Pipeline

```mermaid
sequenceDiagram
    participant D as Dataset (SIFT1M)
    participant KM as K-Means (K=1000)
    participant IDX as Cluster Indexes
    participant Q as Query Engine

    D->>KM: 1M × 128-dim vectors
    KM->>KM: Assign cluster IDs (attributes)
    KM->>IDX: Build 1 HNSW index per cluster
    Note over IDX: 1000 indexes × ~1000 vectors each

    Q->>KM: Query vector
    KM->>Q: Nearest centroid → Attribute ID
    Q->>IDX: Search only that cluster's index
    IDX->>Q: Top-K approximate neighbors
    Note over Q: 1000× smaller search space
```

---

## What Is a Vector / Embedding?

An embedding converts complex data (images, text) into numbers so that similar things have similar numbers.

```
Cat photo    → [0.2, 0.8, 0.1, 0.9, ...]   ← similar!
Another cat  → [0.2, 0.7, 0.1, 0.8, ...]   ← similar!
Dog photo    → [0.9, 0.1, 0.7, 0.2, ...]   ← different
```

This project uses **SIFT descriptors** — 128-dimensional vectors describing visual patches of images.

---

## Dataset: SIFT1M

| Property | Value |
|---|---|
| Base vectors | 1,000,000 |
| Query vectors | 10,000 |
| Dimensions | 128 |
| Data type | float32 |
| Distance metric | L2 (Euclidean) |
| Size | ~500 MB |

```bash
cd experiments
mkdir sift && cd sift
wget ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz
tar -xzf sift.tar.gz
mv sift/* . && rm -rf sift sift.tar.gz
```

---

## Key Concepts

| Concept | Definition |
|---|---|
| Recall@1 | % of queries where true nearest neighbor was found |
| QPS | Queries Per Second — how fast the system is |
| K (results) | Number of nearest neighbors to return |
| ef | Exploration factor — beam width during search |
| M | Graph connectivity — neighbors per node |
| Cluster | A partition of vectors sharing similar spatial properties |
| Attribute | The cluster ID assigned to each vector |

---

## Libraries

| Library | Role | Source |
|---|---|---|
| hnswlib | Core HNSW algorithm (header-only C++) | [nmslib/hnswlib](https://github.com/nmslib/hnswlib) |
| FAISS | Alternative index (IVF, PQ, HNSW + GPU) | [facebookresearch/faiss](https://github.com/facebookresearch/faiss) |

Both are included as Git submodules pointing to author forks.

---

## Repository Structure

```
my-vector-search/
├── faiss/                         ← FAISS submodule (fork)
├── hnswlib/                       ← hnswlib submodule (fork)
│   └── hnswlib/
│       ├── hnswalg.h              ← core HNSW algorithm
│       ├── hnswlib.h              ← entry point + interfaces
│       ├── space_l2.h             ← L2 distance function
│       └── space_ip.h             ← inner product distance
│
├── src/                           ← core reusable code
│   ├── clustered_index.h          ← ClusteredIndex class definition
│   └── clustered_index.cpp        ← build + search + re-clustering logic
│
├── experiments/                   ← all experiments
│   ├── README.md                  ← detailed experiment guide
│   ├── sift/                      ← dataset files (gitignored)
│   ├── baseline/
│   │   └── sift1m_search.cpp      ← standard HNSW baseline
│   └── approach2_attribute/
│       ├── generate_attributes.py ← assign cluster attributes (K=1000)
│       ├── build_indexes.cpp      ← build one HNSW index per cluster
│       ├── search.cpp             ← attribute-based search
│       └── dynamic_reindex.cpp    ← background re-clustering thread
│
├── docs/
├── .gitignore
├── .gitmodules
└── README.md
```

---

## Approach: Attribute-Based Clustering (K=1000)

```
Step 1: Run K-Means (K=1000) on 1M SIFT vectors
        Each cluster = one attribute
        Attribute value = cluster ID (0 to 999)

Step 2: Build one HNSW index per cluster
        1000 indexes × ~1000 vectors each

Step 3: At query time:
        find nearest centroid → attribute ID
        search only that cluster's index
        return top K results

Step 4 (real-time):
        background thread re-clusters every N insertions
        atomic swap — search never stops
```

---

## Baseline Results (Standard HNSW)

| Metric | Value |
|---|---|
| Dataset | SIFT1M (1M × 128) |
| Build time | 1013.82s |
| Recall@1 | 0.9686 |
| Search time | 1.34s |
| QPS | 7,443 |
| M | 16 |
| ef_construction | 200 |
| ef_search | 50 |
| K results | 10 |

---

## Expected vs Baseline

| Metric | Baseline | Clustered (K=1000) | Goal |
|---|---|---|---|
| Build time | 1013s | TBD | Faster |
| Recall@1 | 0.9686 | TBD | Maintain ≥0.95 |
| QPS | 7,443 | TBD | Higher |
| Search space | 1,000,000 | ~1,000 | 1000× smaller |

---

## Quick Start

```bash
# 1. Clone with submodules
git clone --recurse-submodules https://github.com/salemmohammed/my-vector-search.git
cd my-vector-search

# 2. Install dependencies
brew install cmake
pip3 install numpy scikit-learn

# 3. Download SIFT1M
cd experiments && mkdir sift && cd sift
wget ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz
tar -xzf sift.tar.gz && mv sift/* . && rm -rf sift sift.tar.gz
cd ../..

# 4. See experiments/README.md for full instructions
```

---

## Related Work

This project is positioned within a growing body of research on **filtered and partitioned approximate nearest neighbor (ANN) search**:

- **HNSW** — Malkov & Yashunin (2018) introduced the hierarchical navigable small world graph, which achieves near-logarithmic search behavior with strong recall at millisecond latencies. HNSW consistently ranks at the Pareto-optimal front on standard ANN benchmarks such as ANN-Benchmarks [[1]](#references).

- **FAISS** — Johnson et al. (2017) introduced FAISS, which provides efficient IVF (inverted file) indexing by clustering vectors with K-Means and searching only the nearest cluster centroids at query time [[2]](#references). This is the foundational technique our approach extends.

- **IVF Cluster Routing** — Traditional IVF methods partition vectors into clusters based on distance to centroids, then at query time identify the most promising clusters and perform exact search only within those posting lists [[3]](#references). Our approach applies this partitioning idea at a finer granularity, with per-cluster HNSW indexes instead of flat lists.

- **ACORN** — Patel et al. (2024) proposed ACORN for predicate-agnostic filtered vector search, leveraging HNSW with denser vertex neighborhoods to maintain graph connectivity under attribute filters [[4]](#references).

- **Compass** — Ye et al. (2025) introduced Compass, combining IVF-based cluster routing with a proximity graph, demonstrating that hybrid partition + graph indexing achieves competitive QPS and recall across selectivity levels [[5]](#references).

- **Curator** — Engels et al. (2026) showed that partition-based approaches like per-label HNSW indexes can eliminate query-time filtering overhead but require careful tradeoffs between memory and search performance [[6]](#references).

- **Filtered ANNS Survey** — Recent work on filtered approximate nearest neighbor search demonstrates that HNSW's performance degrades in low-selectivity scenarios, where partition-based methods may be preferable, and that hybrid indexing can be necessary for optimal performance across the full selectivity spectrum [[7]](#references).

- **Insertion Order Effects** — Elliott & Clark (2024) showed that HNSW recall is significantly influenced by data insertion sequence and intrinsic dimensionality, with insertion order informed by known categories shifting recall by up to 12 percentage points [[8]](#references). This directly motivates our cluster-aware insertion strategy.

- **ANN-Benchmarks** — Aumüller et al. (2018) established the standard benchmark for evaluating ANN indexes on datasets including SIFT1M, measuring recall@k, QPS, build time, and memory footprint [[9]](#references).

- **Clustered Hybrid Search (Aslam et al.)** — A directly related project on clustered hybrid search that this work builds upon [[10]](#references).

---

## References

<a name="references"></a>

[1] Y. A. Malkov and D. A. Yashunin, "Efficient and Robust Approximate Nearest Neighbor Search Using Hierarchical Navigable Small World Graphs," *IEEE Transactions on Pattern Analysis and Machine Intelligence*, vol. 42, no. 4, pp. 824–836, 2020. [arXiv:1603.09320](https://arxiv.org/abs/1603.09320)

[2] J. Johnson, M. Douze, and H. Jégou, "Billion-Scale Similarity Search with GPUs," *IEEE Transactions on Big Data*, vol. 7, no. 3, pp. 535–547, 2021. [arXiv:1702.08734](https://arxiv.org/abs/1702.08734)

[3] H. Jégou, M. Douze, and C. Schmid, "Product Quantization for Nearest Neighbor Search," *IEEE Transactions on Pattern Analysis and Machine Intelligence*, vol. 33, no. 1, pp. 117–128, 2011. *(SIFT1M dataset also introduced here.)* [Link](http://corpus-texmex.irisa.fr/)

[4] L. Patel, P. Kraft, C. Guestrin, and M. Zaharia, "ACORN: Performant and Predicate-Agnostic Search Over Vector Embeddings and Structured Data," *Proc. ACM Management of Data*, 2024. [arXiv:2403.04871](https://arxiv.org/abs/2403.04871)

[5] C. Ye et al., "Compass: General Filtered Search across Vector and Structured Data," *arXiv preprint*, 2025. [arXiv:2510.27141](https://arxiv.org/abs/2510.27141)

[6] J. Engels, B. Landrum et al., "Curator: Efficient Vector Search with Low-Selectivity Filters," *arXiv preprint*, 2026. [arXiv:2601.01291](https://arxiv.org/abs/2601.01291)

[7] (Anonymous), "Filtered Approximate Nearest Neighbor Search in Vector Databases," *arXiv preprint*, 2026. [arXiv:2602.11443](https://arxiv.org/abs/2602.11443)

[8] O. P. Elliott and J. Clark, "The Impacts of Data, Ordering, and Intrinsic Dimensionality on Recall in Hierarchical Navigable Small Worlds," *arXiv preprint*, 2024. [arXiv:2405.17813](https://arxiv.org/abs/2405.17813)

[9] M. Aumüller, E. Bernhardsson, and A. Faithfull, "ANN-Benchmarks: A Benchmarking Tool for Approximate Nearest Neighbor Algorithms," *Information Systems*, vol. 87, 2020. [ann-benchmarks.com](https://ann-benchmarks.com/)

[10] A. Aslam et al., "Clustered Hybrid Search," GitHub Repository. [Link](https://github.com/AdeelAslamUnimore/Clustered_Hybrid_Search)

[11] M. Ester, H.-P. Kriegel, J. Sander, and X. Xu,
"A Density-Based Algorithm for Discovering Clusters in Large Spatial Databases with Noise,"
KDD, 1996.
