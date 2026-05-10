# my-vector-search

A research project on **Clustered Attributed Vector Search** using HNSW and FAISS.

Vectors are partitioned into clusters where each cluster represents an attribute.
Search is routed to the relevant cluster only — dramatically reducing search space.

---

## Authors

Salem Alqahtani, Adeel Aslam, Khaled Mahmoud, Badr Asiri, Osama Al-Senani,
Faisal Azib, Abdullatif Hadi, Faisal Awad, and Omar Abdulaziz

---

## Research Questions

1. Can spatial clustering reduce vector search time while maintaining recall?
2. What is the optimal number of clusters (K) for the best recall/QPS tradeoff?
3. Can background re-clustering maintain index quality for real-time insertions?

---

## Core Idea

```
Standard HNSW:
Query → search ALL 1,000,000 vectors → slow

Clustered HNSW (this project):
Query → find nearest cluster (attribute) → search ~1,000 vectors → faster!
```

---

## What Is a Vector / Embedding?

An embedding converts complex data (images, text) into numbers so that
similar things have similar numbers.

```
Cat photo    → [0.2, 0.8, 0.1, 0.9, ...]   ← similar!
Another cat  → [0.2, 0.7, 0.1, 0.8, ...]   ← similar!
Dog photo    → [0.9, 0.1, 0.7, 0.2, ...]   ← different
```

This project uses SIFT descriptors — 128-dimensional vectors describing visual patches of images.

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

Download:
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
| Build time | 1013s | TBD | faster |
| Recall@1 | 0.9686 | TBD | maintain |
| QPS | 7,443 | TBD | higher |
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

## References

- [HNSW Paper](https://arxiv.org/abs/1603.09320) — Malkov & Yashunin, 2016
- [FAISS Paper](https://arxiv.org/abs/1702.08734) — Johnson et al., 2017
- [SIFT1M Dataset](http://corpus-texmex.irisa.fr/) — Jégou et al., 2011
- [ANN Benchmarks](https://ann-benchmarks.com/)
- [Clustered Hybrid Search](https://github.com/AdeelAslamUnimore/Clustered_Hybrid_Search) — Aslam et al.
