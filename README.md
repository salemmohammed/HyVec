# my-vector-search

A research project exploring **attributed vector search using spatial clustering** on top of HNSW (Hierarchical Navigable Small World) and FAISS indexes.

---

## What Is This Project?

Modern AI systems (ChatGPT, Google, Pinterest) convert data into **vectors (embeddings)** and search for similar vectors to answer queries. This project investigates how to make that search **faster** by partitioning vectors into clusters and searching only the relevant cluster instead of the entire dataset.

### The Core Idea

```
Standard HNSW:
Query → search ALL 1,000,000 vectors → slow

Clustered HNSW (this project):
Query → find nearest cluster → search ~10,000 vectors → faster!
```

---

## What Is a Vector / Embedding?

An embedding is a way to represent complex data (images, text, audio) as a list of numbers so that **similar things have similar numbers**.

```
Cat photo      → [0.2, 0.8, 0.1, 0.9, ...]   similar numbers!
Another cat    → [0.2, 0.7, 0.1, 0.8, ...]   ↑
Dog photo      → [0.9, 0.1, 0.7, 0.2, ...]   different numbers
```

This project uses **SIFT descriptors** — 128-dimensional vectors that describe visual patches of images.

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

### Files
```
experiments/sift/
├── sift_base.fvecs          ← 1M database vectors
├── sift_query.fvecs         ← 10K query vectors
├── sift_groundtruth.ivecs   ← true nearest neighbors (answer key)
├── sift_learn.fvecs         ← 100K training vectors
├── cluster_labels.npy       ← cluster ID for each vector (generated)
└── cluster_centroids.npy    ← center of each cluster (generated)
```

### File Format (.fvecs)
```
Each vector stored as:
[dim (4 bytes)][f1 (4 bytes)][f2 (4 bytes)]...[f128 (4 bytes)]
= 4 + (128 × 4) = 516 bytes per vector
```

---

## Key Concepts

### Recall
```
Recall = correct results found / total correct results

Recall = 1.0 → 100% accurate (perfect)
Recall = 0.97 → found true nearest neighbor in 97% of queries
```

### QPS (Queries Per Second)
```
QPS = number of queries / search time
QPS = 7,443 → system answers 7,443 queries per second
Higher QPS = faster system
```

### K (Number of Results)
```
K = how many nearest neighbors to return
K=10 → return the 10 most similar vectors
```

### ef (Exploration Factor)
```
ef = how many candidates HNSW explores during search
Small ef → faster but less accurate
Large ef → slower but more accurate
Rule: ef must be >= K
```

### M (Graph Connectivity)
```
M = number of neighbors per node in the HNSW graph
Low M  → sparse graph, fast build, less memory, lower recall
High M → dense graph, slow build, more memory, higher recall
Default: M=16
```

---

## Libraries Used

### HNSW (hnswlib)
- Graph-based approximate nearest neighbor search
- Builds a layered graph of vectors
- Fast search by navigating the graph layer by layer
- GitHub: [nmslib/hnswlib](https://github.com/nmslib/hnswlib)

### FAISS
- Facebook AI Similarity Search
- Multiple index types (IVF, PQ, HNSW)
- GPU support
- GitHub: [facebookresearch/faiss](https://github.com/facebookresearch/faiss)

---

## How HNSW Works

### Build Phase
```
addPoint(vector)
  │
  ├─ assign random layer level
  ├─ navigate from top layer to find entry point
  ├─ getNeighborsByHeuristic2() → pick best M neighbors
  └─ mutuallyConnectNewElement() → connect bidirectionally
```

### Search Phase
```
searchKnn(query, K)
  │
  ├─ start at top layer entry point
  ├─ greedy descent layer by layer
  └─ beam search at layer 0 with ef candidates
        └─ return top K results
```

### Layered Graph Structure
```
Layer 2 (express):   A ————————————— F
Layer 1 (local):     A —— B —— C —— F —— G
Layer 0 (all nodes): A — B — C — D — E — F — G — H
```

---

## Project Architecture

```
my-vector-search/
├── faiss/                        ← FAISS submodule (your fork)
├── hnswlib/                      ← hnswlib submodule (your fork)
│   └── hnswlib/
│       ├── hnswalg.h             ← core HNSW algorithm
│       ├── hnswlib.h             ← main entry point
│       ├── space_l2.h            ← L2 distance
│       └── space_ip.h            ← inner product distance
│
├── src/                          ← your core code
│   ├── clustered_index.h         ← cluster index class definition
│   └── clustered_index.cpp       ← build one HNSW index per cluster
│
├── experiments/                  ← scripts and results
│   ├── sift/                     ← dataset (not in git)
│   ├── cluster_sift.py           ← Phase 1: cluster the data
│   ├── sift1m_search.cpp         ← baseline: standard HNSW
│   ├── clustered_index.cpp       ← Phase 2: build per-cluster indexes
│   └── clustered_search.cpp      ← Phase 4: clustered search pipeline
│
├── docs/
└── README.md
```

---

## Implementation Plan

### Phase 1: Cluster the Data (`experiments/cluster_sift.py`)
```
Input:  sift_base.fvecs (1M vectors)
Process: K-Means clustering (K=100)
Output: cluster_labels.npy    → which cluster each vector belongs to
        cluster_centroids.npy → center of each cluster
```

### Phase 2: Build Per-Cluster Indexes (`src/clustered_index.cpp`)
```
Input:  sift_base.fvecs + cluster_labels.npy
Process: build one HierarchicalNSW index per cluster
Output: 100 small HNSW indexes
```

### Phase 3: Modify Search (`hnswlib/hnswlib/hnswalg.h`)
```
No changes to hnswalg.h (Plan A)
Routing handled externally in clustered_index.cpp
```

### Phase 4: Search Pipeline (`experiments/clustered_search.cpp`)
```
Input:  query vector + cluster_centroids.npy
Process:
  1. find nearest centroid → cluster ID
  2. search only that cluster's HNSW index
Output: K nearest neighbors + recall + QPS
```

---

## Baseline Results (Standard HNSW)

Run on MacBook (single thread):

| Metric | Value |
|---|---|
| Dataset | SIFT1M (1M vectors, 128 dim) |
| Build time | 1013.82s |
| Recall@1 | 0.9686 |
| Search time | 1.34s |
| QPS | 7,443 |
| M | 16 |
| ef_construction | 200 |
| ef_search | 50 |
| K | 10 |

---

## Clustering Results

| Metric | Value |
|---|---|
| Algorithm | MiniBatchKMeans |
| K (clusters) | 100 |
| Converged at step | 278 / 10,000 |
| Min cluster size | 5,292 vectors |
| Max cluster size | 18,176 vectors |
| Avg cluster size | 10,000 vectors |

---

## Expected Results (Clustered HNSW)

| Metric | Baseline | Clustered | Goal |
|---|---|---|---|
| Build time | 1013s | TBD | faster |
| Recall@1 | 0.9686 | TBD | close to baseline |
| QPS | 7,443 | TBD | higher |

---

## Setup

### Prerequisites
```bash
brew install cmake
pip3 install numpy scikit-learn
```

### Clone with Submodules
```bash
git clone --recurse-submodules https://github.com/salemmohammed/my-vector-search.git
cd my-vector-search
```

### Download SIFT1M Dataset
```bash
cd experiments
mkdir sift && cd sift
wget ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz
tar -xzf sift.tar.gz
mv sift/* . && rm -rf sift sift.tar.gz
```

### Build hnswlib Examples
```bash
cd hnswlib
mkdir build && cd build
cmake ..
make
./example_search
```

---

## Running the Experiments

### Step 1: Baseline
```bash
cd experiments
g++ -O3 -std=c++17 sift1m_search.cpp -o sift1m_search
./sift1m_search
```

### Step 2: Cluster the Data
```bash
python3 cluster_sift.py
```

### Step 3: Build Clustered Index
```bash
g++ -O3 -std=c++17 clustered_index.cpp -o clustered_index
./clustered_index
```

### Step 4: Run Clustered Search
```bash
g++ -O3 -std=c++17 clustered_search.cpp -o clustered_search
./clustered_search
```

---

## Submodule Management

### Add Upstream Remotes
```bash
cd faiss
git remote add upstream https://github.com/facebookresearch/faiss.git
cd ../hnswlib
git remote add upstream https://github.com/nmslib/hnswlib.git
```

### Sync with Upstream
```bash
git fetch upstream
git merge upstream/main
```

### Push Your Changes
```bash
# inside a submodule
git add .
git commit -m "your change"
git push origin main

# back in root repo
cd ..
git add faiss  # or hnswlib
git commit -m "update submodule pointer"
git push
```

---

## Authors
- Salem Alqahtani

---

## References
- [HNSW Paper](https://arxiv.org/abs/1603.09320) — Malkov & Yashunin, 2016
- [FAISS Paper](https://arxiv.org/abs/1702.08734) — Johnson et al., 2017
- [SIFT1M Dataset](http://corpus-texmex.irisa.fr/) — Jégou et al., 2011
- [ANN Benchmarks](https://ann-benchmarks.com/) — comparison of ANN algorithms
