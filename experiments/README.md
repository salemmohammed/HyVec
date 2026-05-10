# Experiments

This folder contains all experiments for the Clustered Attributed Vector Search project.

---

## What We Are Solving

Standard HNSW searches ALL vectors for every query. This is slow at scale.
We partition vectors into clusters (attributes) and search only the relevant cluster.

```
Without clustering:  search 1,000,000 vectors per query
With clustering:     search ~1,000 vectors per query (1000x smaller!)
```

---

## Folder Structure

```
experiments/
├── README.md                      ← this file
├── sift/                          ← dataset (gitignored, download separately)
│   ├── sift_base.fvecs            ← 1M database vectors
│   ├── sift_query.fvecs           ← 10K query vectors
│   ├── sift_groundtruth.ivecs     ← true nearest neighbors
│   ├── sift_learn.fvecs           ← 100K training vectors
│   ├── attr_labels.npy            ← generated: cluster ID per vector
│   └── attr_centroids.npy         ← generated: cluster centroids
│
├── baseline/
│   └── sift1m_search.cpp          ← standard HNSW, no clustering
│
└── approach2_attribute/
    ├── generate_attributes.py     ← Step 1: cluster vectors into attributes
    ├── build_indexes.cpp          ← Step 2: build one HNSW index per cluster
    ├── search.cpp                 ← Step 3: search using cluster attributes
    └── dynamic_reindex.cpp        ← Step 4: real-time re-clustering thread
```

---

## Prerequisites

```bash
brew install cmake
pip3 install numpy scikit-learn
```

---

## Download SIFT1M Dataset

```bash
cd experiments
mkdir sift && cd sift
wget ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz
tar -xzf sift.tar.gz
mv sift/* .
rm -rf sift sift.tar.gz
ls -lh   # should show 4 files, ~500MB total
```

---

## Step 0: Run the Baseline

Standard HNSW with no clustering. This is our comparison point.

```bash
cd experiments/baseline
g++ -O3 -std=c++17 sift1m_search.cpp -o sift1m_search -I../../hnswlib
./sift1m_search
```

Expected output:
```
Base vectors:  1000000 x 128
Query vectors: 10000
Build time:    ~1013s
Recall@1:      0.9686
QPS:           ~7443
```

This is the number to beat.

---

## Step 1: Generate Attributes (K-Means K=1000)

Runs K-Means on 1M vectors. Each cluster = one attribute.
K is automatically determined by the data distribution.

```bash
cd experiments/approach2_attribute
python3 generate_attributes.py
```

What it does:
```
Input:  ../sift/sift_base.fvecs   (1M vectors)
Process: MiniBatchKMeans K=1000
Output: ../sift/attr_labels.npy   (1M integers, each 0-999)
        ../sift/attr_centroids.npy (1000 × 128 centroid vectors)
```

Expected output:
```
Loaded: 1000000 vectors x 128 dimensions
Running MiniBatchKMeans K=1000...
=== ATTRIBUTE RESULTS ===
Number of attributes (clusters): 1000
Min cluster size: ~500
Max cluster size: ~2000
Avg cluster size: ~1000
```

---

## Step 2: Build Per-Cluster HNSW Indexes

Builds one HNSW index per cluster. 1000 small indexes instead of 1 big one.

```bash
cd experiments/approach2_attribute
g++ -O3 -std=c++17 build_indexes.cpp -o build_indexes -I../../hnswlib
./build_indexes
```

What it does:
```
Input:  ../sift/sift_base.fvecs
        ../sift/attr_labels.npy
Process: for each cluster c:
           collect all vectors where label == c
           build HierarchicalNSW index for those vectors
Output: ../sift/indexes/index_0.bin
        ../sift/indexes/index_1.bin
        ...
        ../sift/indexes/index_999.bin
```

Expected output:
```
Building 1000 HNSW indexes...
Built index 0: 1023 vectors
Built index 1: 987 vectors
...
Total build time: ~Xs
```

---

## Step 3: Search Using Cluster Attributes

For each query: find nearest centroid → search only that cluster's index.

```bash
cd experiments/approach2_attribute
g++ -O3 -std=c++17 search.cpp -o search -I../../hnswlib
./search
```

What it does:
```
Input:  ../sift/sift_query.fvecs       (10K queries)
        ../sift/sift_groundtruth.ivecs  (answer key)
        ../sift/attr_centroids.npy      (1000 centroids)
        ../sift/indexes/               (1000 HNSW indexes)
Process: for each query:
           1. compute distance to all 1000 centroids
           2. pick nearest centroid → cluster ID
           3. search only index[cluster_ID]
           4. return top K results
Output: Recall@1, QPS, search time
```

Expected output:
```
=== CLUSTERED SEARCH RESULTS ===
Recall@1:    ~0.85-0.95  (lower than baseline due to cluster boundary effects)
Search time: ~Xs
QPS:         ~higher than baseline
```

---

## Step 4: Dynamic Re-clustering (Real-Time)

Background thread re-clusters every N insertions. Atomic swap keeps search running.

```bash
cd experiments/approach2_attribute
g++ -O3 -std=c++17 dynamic_reindex.cpp -o dynamic_reindex -I../../hnswlib -lpthread
./dynamic_reindex
```

What it does:
```
Main thread:       insert new vectors + serve queries
Background thread: every 10,000 insertions →
                   re-run K-Means →
                   rebuild indexes →
                   atomic swap (search never stops)
```

Expected output:
```
Inserting vectors...
[Background] Re-clustering triggered at 10000 insertions
[Background] New indexes ready, swapping...
[Background] Swap complete. Search continues.
Inserting vectors...
[Background] Re-clustering triggered at 20000 insertions
...
```

---

## Results Comparison

| Metric | Baseline | Clustered K=1000 | Improvement |
|---|---|---|---|
| Build time | 1013s | TBD | TBD |
| Recall@1 | 0.9686 | TBD | TBD |
| QPS | 7,443 | TBD | TBD |
| Search space | 1,000,000 | ~1,000 | ~1000× |

---

## Tuning K (Number of Clusters)

K is the most important parameter. Run search with different K values:

```bash
# Edit generate_attributes.py and change K
K = 100   → avg 10,000 vectors/cluster → higher recall, lower QPS
K = 500   → avg 2,000 vectors/cluster  → balanced
K = 1000  → avg 1,000 vectors/cluster  → lower recall, higher QPS
K = 5000  → avg 200 vectors/cluster    → very low recall, very high QPS
```

Plot recall vs QPS for each K to find the optimal tradeoff.

---

## Key Research Insights

### Why Recall May Drop
```
True nearest neighbor of query Q might be in cluster 5
But Q's nearest centroid is cluster 3
→ We search cluster 3 and miss the true answer
→ This is the cluster boundary problem
```

### How to Improve Recall
```
Search top-2 or top-3 nearest centroids instead of just top-1
→ higher recall at cost of some QPS
```

### Real-Time Insight
```
K-Means centroids are fixed after training
New vectors may not fit existing centroids
→ Background re-clustering fixes this
→ Atomic swap ensures zero downtime
```

---

## File Descriptions

### `baseline/sift1m_search.cpp`
Standard HNSW search on all 1M vectors. No clustering.
Reference point for all comparisons.

### `approach2_attribute/generate_attributes.py`
Runs MiniBatchKMeans on SIFT vectors.
Saves cluster labels and centroids.
K=1000 by default (configurable).

### `approach2_attribute/build_indexes.cpp`
Reads cluster labels.
Builds one HierarchicalNSW index per cluster.
Saves each index to disk as a binary file.

### `approach2_attribute/search.cpp`
Loads all 1000 indexes and centroids.
For each query: finds nearest centroid, searches that index.
Reports Recall@1, QPS, and search time.

### `approach2_attribute/dynamic_reindex.cpp`
Demonstrates real-time insertion with background re-clustering.
Uses std::thread and std::mutex for atomic index swap.
Shows zero-downtime index maintenance.
