# Experiments

Detailed guide for running all experiments in this project.

---

## Overview

```
experiments/
└── approach_attribute/
    ├── generate_attributes.py     ← Step 1: cluster vectors into attributes
    ├── build_indexes.cpp          ← Step 2: build one HNSW index per cluster
    ├── search.cpp                 ← Step 3: sweep TOP_CLUSTERS values
    ├── dynamic_reindex.cpp        ← Step 4: real-time re-clustering
    └── run_all.sh                 ← run everything at once
```

---

## Prerequisites

```bash
brew install cmake
pip3 install numpy scikit-learn
```

---

## Download SIFT1M

```bash
cd experiments/sift
wget ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz
tar -xzf sift.tar.gz
mv sift/* . && rm -rf sift sift.tar.gz
```

Files:
```
sift_base.fvecs        ← 1M database vectors (492MB)
sift_query.fvecs       ← 10K query vectors
sift_groundtruth.ivecs ← true nearest neighbors (answer key)
sift_learn.fvecs       ← 100K training vectors
```

---

## Run Everything

```bash
cd experiments/approach2_attribute
./run_all.sh
```

Or run each step individually below.

---

## Step 1: Generate Attributes

Runs K-Means (K=1000) on 1M vectors.
Each cluster = one attribute (ID 0–999).

```bash
cd experiments/approach2_attribute
python3 generate_attributes.py
```

Output files:
```
sift/attr_labels.npy     → 1M integers, cluster ID per vector
sift/attr_centroids.npy  → 1000 × 128 centroid vectors
```

Expected output (results pending):
```
Clusters:         1000
Min cluster size: xxx
Max cluster size: xxx
Avg cluster size: xxx
Clustering time:  xxx s
```

Config (in `generate_attributes.py`):
```python
K          = 1000   # number of clusters
BATCH_SIZE = 10000  # K-Means batch size
```

---

## Step 2: Build Per-Cluster HNSW Indexes

Builds one HNSW index per cluster — 1000 small indexes in total.

```bash
g++ -O3 -std=c++17 build_indexes.cpp -o build_indexes -I../../hnswlib
./build_indexes
```

Output files:
```
sift/indexes/index_0.bin
sift/indexes/index_1.bin
...
sift/indexes/index_999.bin
```

Expected output (results pending):
```
Total indexes: 1000
Build time:    xxx s
```

Config (in `build_indexes.cpp`):
```cpp
const int K               = 1000;
const int M               = 16;
const int EF_CONSTRUCTION = 200;
```

---

## Step 3: Clustered Search Sweep

Tests different TOP_CLUSTERS values: 1, 3, 5, 10, 20, 50.
Loads all 1000 indexes once, then searches with each setting.

```bash
g++ -O3 -std=c++17 search.cpp -o search -I../../hnswlib
./search
```

Expected output (results pending):
```
TOP_CLUSTERS   Recall@1    QPS         Time(s)
---------------------------------------------------
1              xxx         xxx         xxx
3              xxx         xxx         xxx
5              xxx         xxx         xxx
10             xxx         xxx         xxx
20             xxx         xxx         xxx
50             xxx         xxx         xxx
---------------------------------------------------
Baseline:      xxx         xxx         xxx
```

> Results will be updated upon completion of experiments on the NVIDIA DGX Spark.

---

## Step 4: Dynamic Re-clustering (Real-Time)

Demonstrates real-time insertion with background re-clustering.
Search never stops during re-clustering (atomic swap).

```bash
g++ -O3 -std=c++17 dynamic_reindex.cpp -o dynamic_reindex -I../../hnswlib -lpthread
./dynamic_reindex
```

Expected output (results pending):
```
Insert rate:      xxx vectors/second
Re-cluster every: 10,000 insertions
Re-cluster time:  xxx s
Search downtime:  0 (atomic swap)
```

Config (in `dynamic_reindex.cpp`):
```cpp
const int RECLUSTER_EVERY = 10000;  // re-cluster every N insertions
```

> Results will be updated upon completion of experiments on the NVIDIA DGX Spark.

---

## Understanding the Design

### Why Recall May Drop With Small TOP_CLUSTERS

```
True nearest neighbor of query Q is in cluster 5.
Q's nearest centroid is cluster 3 (slightly off).
→ We search cluster 3, miss cluster 5.
→ Miss the true answer → recall drops.

Fix: increase TOP_CLUSTERS to search more clusters.
```

### Why Build Time Is Expected to Be Faster

```
Baseline:  build 1 index with 1,000,000 vectors
Clustered: build 1000 indexes with ~1,000 vectors each

HNSW build complexity: O(N × log N)
1 × O(1M × log 1M)  >>  1000 × O(1K × log 1K)

The logarithmic term shrinks significantly with smaller N per cluster.
Exact speedup factor to be confirmed by experiments.
```

### Why QPS May Drop With More TOP_CLUSTERS

```
Each cluster search has overhead:
  - distance to 1000 centroids: O(1000 × 128)
  - load index from memory
  - HNSW graph traversal

Searching T small indexes adds overhead that
a single large HNSW index does not incur.
Exact tradeoff to be confirmed by experiments.
```

---

## Tuning Guide

| Parameter | Location | Effect |
|---|---|---|
| K (clusters) | `generate_attributes.py` | more clusters = smaller indexes = faster build |
| TOP_CLUSTERS | `search.cpp` | higher = better recall, lower QPS |
| M | `build_indexes.cpp` | higher = better recall, slower build, more memory |
| EF_CONSTRUCTION | `build_indexes.cpp` | higher = better recall, slower build |
| EF_SEARCH | `search.cpp` | higher = better recall, lower QPS |
| RECLUSTER_EVERY | `dynamic_reindex.cpp` | lower = fresher clusters, more CPU overhead |

---

## File Descriptions


### `approach_attribute/generate_attributes.py`
Runs MiniBatchKMeans. Saves cluster labels and centroids.
Functions: `read_fvecs()`, `main()`.

### `approach_attribute/build_indexes.cpp`
Reads labels. Builds one HierarchicalNSW per cluster. Saves to disk.
Functions: `read_fvecs()`, `read_npy_labels()`, `main()`.

### `approach_attribute/search.cpp`
Loads all indexes. Sweeps TOP_CLUSTERS = {1, 3, 5, 10, 20, 50}.
Reports Recall@1 and QPS per setting.
Functions: `read_fvecs()`, `read_ivecs()`, `read_npy_centroids()`,
`find_nearest_centroids()`, `main()`.

### `approach_attribute/dynamic_reindex.cpp`
Real-time insertion + background re-clustering with atomic swap.
Classes: `DynamicClusteredIndex`.
Methods: `insert()`, `search()`, `recluster_thread()`, `load_initial()`.
