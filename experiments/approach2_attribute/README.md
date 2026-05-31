# Approach 2: Clustered Attributed Vector Search

This folder contains the full implementation of our Clustered HNSW system —
the main contribution of the paper. It is organized as a four-step pipeline
that partitions a vector dataset into clusters, builds one HNSW index per
cluster, searches across clusters, and supports real-time insertions with
background re-clustering.

---

## Language Overview

| File | Language | Role |
|---|---|---|
| `generate_attributes.py` | Python | K-Means clustering via scikit-learn |
| `build_indexes.cpp` | C++ | Build one HNSW index per cluster |
| `search.cpp` | C++ | Query sweep across TOP_CLUSTERS values |
| `dynamic_reindex.cpp` | C++ | Real-time insertion + background re-clustering |
| `run_all.sh` | Bash | Run all four steps in order |
| `benchmark.sh` | Bash | Run and record results to CSV |

**Why Python for Step 1?**
scikit-learn's `MiniBatchKMeans` is well-optimized and handles 1M vectors
efficiently. Steps 2–4 are C++ because they involve tight loops over HNSW
graph traversal where latency matters.

---

## Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        BUILD PHASE (offline)                    │
│                                                                 │
│  1M SIFT vectors                                                │
│       │                                                         │
│       ▼                                                         │
│  generate_attributes.py                                         │
│  MiniBatchKMeans (K=1000)                                       │
│       │                                                         │
│       ├── attr_labels.npy      (1M integers, cluster ID)        │
│       └── attr_centroids.npy   (1000 × 128 centroid vectors)    │
│                                                                 │
│       ▼                                                         │
│  build_indexes.cpp                                              │
│  One HierarchicalNSW per cluster                                │
│       │                                                         │
│       ├── indexes/index_0.bin                                   │
│       ├── indexes/index_1.bin                                   │
│       ├── ...                                                   │
│       └── indexes/index_999.bin                                 │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                        QUERY PHASE (online)                     │
│                                                                 │
│  Query vector q                                                 │
│       │                                                         │
│       ▼                                                         │
│  find_nearest_centroids()          O(K × d) = O(1000 × 128)    │
│  Compare q to all 1000 centroids                                │
│  Select TOP_CLUSTERS nearest                                    │
│       │                                                         │
│       ▼                                                         │
│  searchKnn() × TOP_CLUSTERS        O(log(N/K)) per cluster      │
│  HNSW graph traversal in each                                   │
│  selected cluster                                               │
│       │                                                         │
│       ▼                                                         │
│  Merge + sort all candidates                                    │
│  Return top-k results                                           │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                   DYNAMIC PHASE (real-time)                     │
│                                                                 │
│  Main thread          Background thread                         │
│  ───────────          ─────────────────                         │
│  insert(vec, id)  →   monitors score per cluster                │
│  find centroid    →   if score > threshold:                     │
│  add to HNSW      →     rebuild cluster index                   │
│  update metadata  →     atomic swap (zero downtime)             │
│                                                                 │
│  search(q, k)         Score = temperature ×                     │
│  never blocked    ←     (β × imbalance + (1-β) × drift)        │
└─────────────────────────────────────────────────────────────────┘
```

---

## File-by-File Explanation

### Step 1 — `generate_attributes.py` (Python)

**What it does:** Runs MiniBatchKMeans on the 1M SIFT vectors to produce
K=1000 clusters. Each cluster becomes one "attribute" — a partition of the
dataset that groups geometrically similar vectors.

**Key functions:**

```
read_fvecs()     reads SIFT1M binary format → numpy float32 array
main()
  └── MiniBatchKMeans(K=1000)
        ├── fit_predict(base)   → labels[i] = cluster ID for vector i
        └── cluster_centers_    → 1000 × 128 centroid matrix
```

**Why MiniBatchKMeans?**
Full Lloyd's k-means on 1M × 128-d vectors requires O(N × K × d) per
iteration = 1M × 1000 × 128 = 128B ops per iteration. MiniBatchKMeans
processes random subsets (batch_size=10000) per iteration — roughly 100×
faster while converging to a comparable solution.

**Outputs:**
```
attr_labels.npy     shape (1000000,)    int32   — cluster ID per vector
attr_centroids.npy  shape (1000, 128)   float32 — centroid coordinates
```

---

### Step 2 — `build_indexes.cpp` (C++)

**What it does:** Reads the cluster labels, groups vectors by cluster,
and builds one `HierarchicalNSW` index per cluster using hnswlib.

**Key functions:**

```
read_fvecs()         reads SIFT1M .fvecs binary → vector<vector<float>>
read_npy_labels()    reads attr_labels.npy → vector<int>
main()
  ├── cluster_to_ids[label].push_back(i)   group vectors by cluster
  └── for each cluster:
        ├── HierarchicalNSW(space, size, M=16, ef=200)
        ├── addPoint(base[id], id)          add each vector with global ID
        └── saveIndex("indexes/index_k.bin")
```

**Why C++?**
hnswlib is a header-only C++ library. Building 1000 HNSW indexes requires
tight memory management — each index allocates its own graph structure.
C++ gives direct control over allocation and avoids Python GIL overhead
when building many indexes sequentially.

**HNSW parameters:**
```cpp
M               = 16    // bidirectional links per node
                        // higher M → better recall, more memory
EF_CONSTRUCTION = 200   // candidate list size during build
                        // higher ef → better index quality, slower build
```

**Build complexity:**
```
Per cluster:  O(n_k × log(n_k))   where n_k = |cluster k| ≈ 1000
Total:        O(N × log(N/K))     vs O(N × log(N)) for one big HNSW
                                  → log(1000) ≈ 10 vs log(1M) ≈ 20
                                  → ~2× fewer ops per vector
```

---

### Step 3 — `search.cpp` (C++)

**What it does:** Loads all 1000 HNSW indexes into memory, then sweeps
TOP_CLUSTERS ∈ {1, 3, 5, 10, 20, 50} to produce the full Recall vs QPS
curve used in Figure 6.

**Key functions:**

```
read_fvecs()              reads query vectors
read_ivecs()              reads ground truth neighbor indices
read_npy_centroids()      reads 1000 × 128 centroid matrix
find_nearest_centroids()  O(K × d) — sorts all K centroid distances,
                          returns top_n nearest cluster IDs
main()
  ├── load all 1000 indexes into map<int, HierarchicalNSW*>
  └── for each TOP_CLUSTERS in {1,3,5,10,20,50}:
        for each query q:
          ├── find_nearest_centroids(q, centroids, TOP_CLUSTERS)
          ├── searchKnn(q, K_RESULTS) in each selected cluster
          ├── merge all results, sort by distance
          └── check if gt[i][0] in top-K found → Recall@1
        report Recall@1, QPS, Time
```

**The TOP_CLUSTERS tradeoff:**
```
TOP_CLUSTERS = 1  → search 1 cluster only  → fastest,  lowest recall
TOP_CLUSTERS = 20 → search 20 clusters     → slower,   higher recall
TOP_CLUSTERS = K  → search all clusters    → = brute force, Recall=1.0
```

**Query complexity:**
```
Centroid scan:  O(K × d)              = O(1000 × 128) = 128K ops
HNSW search:    O(TOP × log(N/K))     = O(TOP × log(1000)) ≈ TOP × 10
```

**Search parameters:**
```cpp
EF_SEARCH  = 50   // candidate list during query
                  // higher ef → better recall, lower QPS
K_RESULTS  = 10   // number of neighbors to retrieve
```

---

### Step 4 — `dynamic_reindex.cpp` (C++)

**What it does:** Demonstrates real-time vector insertion with a background
thread that monitors cluster health and triggers local re-clustering when
needed. Search is never blocked — new indexes are swapped in atomically.

**Class: `DynamicClusteredIndex`**

```
Member data:
  active_indexes    map<int, HierarchicalNSW*>   — live HNSW indexes
  centroids         vector<vector<float>>         — K centroid vectors
  metadata          vector<ClusterMetadata>       — per-cluster health stats
  all_vectors       vector<vector<float>>         — all inserted vectors
  all_cluster_ids   vector<int>                   — cluster assignment per vector

Key methods:

  insert(vec, id)
    ├── find_nearest_centroid(vec)       O(K × d) centroid scan
    ├── all_vectors.push_back(vec)       store for potential re-clustering
    ├── active_indexes[cluster]          insert into live HNSW index
    │   └── addPoint(vec, id)
    ├── update metadata.current_centroid  running mean update
    ├── compute_score(metadata[cluster])  imbalance + drift score
    └── if score > threshold → cv.notify() trigger background thread

  search(q, k)
    ├── find_nearest_centroid(q)         O(K × d)
    ├── active_indexes[cluster]          always available (never NULL)
    └── searchKnn(q.data(), k)           O(log(N/K))

  recluster_thread()   (runs in background)
    └── loop:
          wait for cv signal
          for each cluster in clusters_to_reindex:
            ├── collect all vectors in that cluster
            ├── build new HierarchicalNSW from scratch
            ├── active_indexes[c] = new_idx   atomic pointer swap
            └── reset metadata score to 0
```

**Re-clustering score:**
```
score = temperature × (β × imbalance + (1-β) × drift)

  imbalance = |size - TARGET_SIZE| / TARGET_SIZE
              how far the cluster size is from the ideal size

  drift     = ||current_centroid - initial_centroid|| / d
              how far the cluster center has moved since last re-index

  temperature = per-cluster query frequency weight
                clusters queried often get higher priority for re-indexing

  β = 0.5   — equal weight to imbalance and drift
```

**Why zero search downtime?**
The re-clustering thread builds a completely new `HierarchicalNSW` object
in the background. When ready, it replaces the old pointer in `active_indexes`
under a mutex. The search thread holds the same mutex only for the duration
of `searchKnn()` — never for the duration of a rebuild.

---

## Language Summary

```
Step 1  generate_attributes.py    Python   scikit-learn MiniBatchKMeans
Step 2  build_indexes.cpp         C++      hnswlib HierarchicalNSW build
Step 3  search.cpp                C++      hnswlib HierarchicalNSW search
Step 4  dynamic_reindex.cpp       C++      hnswlib + std::thread + std::mutex
```

Steps 2–4 are C++ because:
- hnswlib is a C++ header-only library
- HNSW graph traversal is latency-sensitive — no GIL, no Python overhead
- Step 4 uses `std::thread` and `std::mutex` directly for fine-grained
  control over the atomic swap pattern

Step 1 is Python because:
- scikit-learn MiniBatchKMeans is production-quality and well-tested
- K-Means runs once offline — latency is not critical here
- NumPy handles the 1M × 128 float32 matrix efficiently

---

## How to Run

### Full pipeline
```bash
./run_all.sh
```

### Step by step
```bash
# Step 1 — cluster
python3 generate_attributes.py

# Step 2 — build indexes
g++ -O3 -std=c++17 build_indexes.cpp -o build_indexes -I../../hnswlib
./build_indexes

# Step 3 — search sweep
g++ -O3 -std=c++17 search.cpp -o search -I../../hnswlib
./search

# Step 4 — dynamic insertion
g++ -O3 -std=c++17 dynamic_reindex.cpp -o dynamic_reindex \
    -I../../hnswlib -lpthread
./dynamic_reindex
```

### Benchmark (record results to CSV)
```bash
./benchmark.sh              # full run
./benchmark.sh --search-only  # skip build, just search
```

---

## Configuration

| Parameter | File | Default | Effect |
|---|---|---|---|
| `K` | `generate_attributes.py` | 1000 | number of clusters |
| `BATCH_SIZE` | `generate_attributes.py` | 10000 | k-means mini-batch size |
| `M` | `build_indexes.cpp` | 16 | HNSW graph connectivity |
| `EF_CONSTRUCTION` | `build_indexes.cpp` | 200 | HNSW build quality |
| `EF_SEARCH` | `search.cpp` | 50 | HNSW query quality |
| `TOP_CLUSTERS` | `search.cpp` | swept | clusters searched per query |
| `REINDEX_THRESHOLD` | `dynamic_reindex.cpp` | 0.1 | score to trigger re-cluster |
| `BETA` | `dynamic_reindex.cpp` | 0.5 | imbalance vs drift weight |
| `TARGET_SIZE` | `dynamic_reindex.cpp` | 1000 | ideal cluster size |

---

## Expected Results

> All results marked `xxx` will be filled after running experiments
> on the NVIDIA DGX Spark (GB10 Grace Blackwell, 128GB unified memory).

### Step 3 — Search Sweep

| TOP_CLUSTERS | Recall@1 | QPS | Time (s) |
|---|---|---|---|
| 1 | xxx | xxx | xxx |
| 3 | xxx | xxx | xxx |
| 5 | xxx | xxx | xxx |
| 10 | xxx | xxx | xxx |
| 20 | xxx | xxx | xxx |
| 50 | xxx | xxx | xxx |
| Baseline HNSW | xxx | xxx | xxx |

### Step 4 — Dynamic Insertion

| Metric | Value |
|---|---|
| Insert rate (vectors/sec) | xxx |
| Re-cluster events | xxx |
| Clusters rebuilt | xxx |
| Search downtime | 0s (atomic swap) |

---

## References

[1] Y. A. Malkov, D. A. Yashunin: **Efficient and Robust Approximate Nearest
Neighbor Search Using Hierarchical Navigable Small World Graphs.**
IEEE TPAMI, 42(4): 824–836, 2020. arXiv:1603.09320
→ hnswlib — `HierarchicalNSW` used in Steps 2, 3, 4

[2] H. Jégou, M. Douze, C. Schmid: **Product Quantization for Nearest Neighbor
Search.** IEEE TPAMI, 33(1): 117–128, 2011.
→ SIFT1M dataset and .fvecs/.ivecs binary format

[3] M. Aumüller, E. Bernhardsson, A. Faithfull: **ANN-Benchmarks: A Benchmarking
Tool for Approximate Nearest Neighbor Algorithms.**
Information Systems, 87, 2020. DOI: 10.1016/j.is.2019.02.006
→ Evaluation protocol and Recall@k definition

[4] D. Sculley: **Web-Scale K-Means Clustering.** WWW, 2010.
→ MiniBatchKMeans — the clustering algorithm used in Step 1

[5] S. P. Lloyd: **Least Squares Quantization in PCM.** IEEE TIT, 28(2), 1982.
→ Lloyd's algorithm — the theoretical basis of k-means
