# Benchmarks

This directory contains the benchmarking framework used to evaluate the proposed **Hybrid Attribute-Spatial HNSW** index against representative **Approximate Nearest Neighbor (ANN)** indexing techniques. The benchmark is built upon the **ANN-Benchmarks** framework, providing a standardized and reproducible environment for evaluating search accuracy, throughput, index construction time, query latency, memory consumption, and index size.

The evaluation includes representative ANN methods from the major indexing families:

- **Exact search:** Brute Force (ground-truth reference)
- **Hash-based methods:** Locality-Sensitive Hashing (LSH)
- **Cluster-based methods:** IVF and IVFADC
- **Partition and quantization methods:** ScaNN
- **Graph-based methods:** HNSW
- **Disk-based graph methods:** DiskANN
- **Hybrid methods:** Hybrid Attribute-Spatial HNSW (proposed)

Experiments are conducted using the **SIFT1M** benchmark dataset to evaluate conventional ANN search performance and an **attributed vector dataset** to evaluate hybrid attribute-aware vector search. While SIFT1M provides a standardized benchmark for comparing ANN indexing techniques, the attributed dataset evaluates the effectiveness of combining attribute filtering, spatial clustering, and local HNSW search.

Performance is evaluated using **Recall@k**, **Queries Per Second (QPS)**, **index construction time**, **query latency**, **memory consumption**, and **index size** under identical hardware and software configurations.

---

# Benchmark Goals

The benchmarking framework is designed to answer the following questions:

- How does Hybrid Attribute-Spatial HNSW compare with representative ANN indexing techniques?
- What is the trade-off between search accuracy and throughput?
- How much overhead does attribute-aware partitioning introduce?
- How does local HNSW indexing affect query performance?
- What are the construction and maintenance costs of clustered HNSW indexes?

---

# Hardware Platform

All benchmark experiments are conducted on an **NVIDIA DGX Spark** workstation.

| Component | Specification |
|-----------|---------------|
| **System** | NVIDIA DGX Spark |
| **Processor** | NVIDIA GB10 Grace Blackwell |
| **CPU** | ARM Cortex @ 3.8 GHz |
| **Memory** | 128 GB Unified Memory |
| **Storage** | 4 TB NVMe SSD |
| **Operating System** | NVIDIA DGX OS |

---

# Datasets

Two datasets are used throughout the evaluation.

## Public ANN Benchmark

The **SIFT1M** dataset provided by ANN-Benchmarks is used for evaluating conventional ANN search performance.

| Dataset | Dimension | Database Size | Query Size | Distance |
|----------|----------:|--------------:|-----------:|----------|
| **SIFT1M** | 128 | 1,000,000 | 10,000 | Euclidean (L2) |

## Attributed Dataset

An attributed vector dataset is used to evaluate hybrid vector search with metadata filtering.

Each vector is associated with one or more attributes, allowing evaluation of:

- Attribute-aware filtering
- Spatial clustering
- Local HNSW indexing
- Hybrid vector retrieval

---

## Download SIFT1M

```bash
cd benchmarks/data

curl -L \
  -A "Mozilla/5.0" \
  -o sift-128-euclidean.hdf5 \
  https://ann-benchmarks.com/sift-128-euclidean.hdf5
```

Verify the download:

```bash
ls -lh sift-128-euclidean.hdf5
```

Expected output:

```text
-rw-r--r--  ... 501M sift-128-euclidean.hdf5
```

Directory layout:

```text
benchmarks/
├── algorithms/
├── data/
│   ├── download_datasets.py
│   └── sift-128-euclidean.hdf5
├── results/
└── README.md
```

---

# Supported ANN Algorithms

The benchmarking framework evaluates representative ANN indexing techniques available through ANN-Benchmarks.

## Graph-Based Methods

| Algorithm | Description |
|-----------|-------------|
| **hnswlib** | Reference HNSW implementation |
| **HNSW (NMSLIB)** | HNSW implementation in NMSLIB |
| **HNSW (FAISS)** | HNSW implementation in FAISS |
| **GLASS** | Optimized graph-based ANN search |
| **DiskANN (Vamana)** | Disk-resident graph-based ANN search |
| **N2** | Graph-based ANN implementation |

## Partition-Based Methods

| Algorithm | Description |
|-----------|-------------|
| **FAISS-IVF** | Inverted File index |
| **ScaNN** | Partitioning with anisotropic quantization |
| **PyNNDescent** | Approximate graph construction using NN-Descent |

## Exact Search

| Algorithm | Description |
|-----------|-------------|
| **FAISS IndexFlatL2** | Exact nearest neighbor search used as the ground-truth baseline. |

---

# Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **Recall@k** | Fraction of true nearest neighbors returned |
| **Queries Per Second (QPS)** | Query throughput |
| **Average Query Latency** | Average search time per query |
| **Build Time** | Time required to construct the index |
| **Search Time** | Total query execution time |
| **Memory Consumption** | Memory required by the index |
| **Index Size** | Size of the constructed index |

Recall is computed as

```text
Recall@k = |Retrieved ∩ GroundTruth| / k
```

---

# Experimental Setup

All algorithms are evaluated on the same hardware using identical software configurations.

The evaluation consists of two complementary studies.

## Public Benchmark

- Dataset: **SIFT1M**
- Distance: **Euclidean (L2)**
- Metrics:
  - Recall@k
  - QPS
  - Query latency
  - Build time
  - Search time
  - Memory usage
  - Index size

## Hybrid Search Benchmark

- Dataset: Attributed vector dataset
- Evaluation:
  - Attribute filtering
  - Spatial clustering
  - Local HNSW search
  - Hybrid vector retrieval

---

# Repository Structure

```text
benchmarks/
├── README.md
├── data/
│   ├── download_datasets.py
│   └── sift-128-euclidean.hdf5
│
├── algorithms/
│   ├── brute_force.cpp
│   ├── baseline_hnsw.cpp
│   ├── clustered_hnsw.cpp
│   └── faiss_ivf.py
│
├── results/
├── figures/
└── run_benchmark.sh
```

---

## Running the Algorithms

All commands assume you are inside:

```bash
cd benchmarks/algorithms
```

---

## Brute Force (FAISS IndexFlatL2)

### Build

```bash
g++ -O3 -std=c++17 brute_force.cpp -o brute_force \
$(pkg-config --cflags --libs hdf5) \
-I/opt/homebrew/opt/faiss/include \
-L/opt/homebrew/opt/faiss/lib \
-lfaiss
```

### Run

```bash
./brute_force \
    --dataset ../data/sift-128-euclidean.hdf5 \
    --k 10
```

---

## Baseline HNSW

### Build

```bash
g++ -O3 -std=c++17 baseline_hnsw.cpp -o baseline_hnsw \
-I../../hnswlib \
$(pkg-config --cflags --libs hdf5)
```

### Run

```bash
./baseline_hnsw \
    --dataset ../data/sift-128-euclidean.hdf5 \
    --k 10
```

---

## FAISS-IVF (IndexIVFFlat)

### Build

```bash
g++ -O3 -std=c++17 faiss_ivf.cpp -o faiss_ivf \
$(pkg-config --cflags --libs hdf5) \
-I/opt/homebrew/opt/faiss/include \
-L/opt/homebrew/opt/faiss/lib \
-lfaiss
```

### Run

```bash
./faiss_ivf \
    --dataset ../data/sift-128-euclidean.hdf5 \
    --k 10 \
    --nlist 1000
```

The implementation automatically evaluates multiple **nprobe** values (`1, 5, 10, 20, 50, 100`) and appends the results to `results.csv`.

---

## Hybrid Attribute-Spatial HNSW (Proposed)

### Build

```bash
g++ -O3 -std=c++17 clustered_hnsw.cpp -o clustered_hnsw \
-I../../hnswlib \
$(pkg-config --cflags --libs hdf5)
```

### Run

```bash
./clustered_hnsw \
    --dataset ../data/sift-128-euclidean.hdf5 \
    --k 10
```

---

## Output

Each benchmark prints the following performance metrics and appends them to `../results.csv`:

- Recall@1
- Recall@k
- Queries Per Second (QPS)
- Average query latency
- Index construction time
- Search time
- Index size

---

# Results

Benchmark results are automatically exported to **results.csv**.

The table below summarizes the current benchmark results obtained on the NVIDIA DGX Spark platform.

| Algorithm | Dataset | Recall@1 | Recall@10 | QPS | Avg Latency (ms) | Build Time (s) | Search Time (s) | Index Size (MB) |
|-----------|---------|---------:|----------:|----:|-----------------:|---------------:|----------------:|----------------:|
| **FAISS IndexFlatL2 (Brute Force)** | SIFT1M | 0.9926 | 0.99935 | 1146.03 | 0.8726 | 0.0424 | 8.7257 | 488.28 |
| FAISS IndexFlatL2 (Brute Force, C++) | SIFT1M | 0.9926 | 0.99935 | 1146.03 | 0.8726 | 0.0424 | 8.7257 | 488.28 |
| **FAISS-IVF (nlist=1000, nprobe=20)** | SIFT1M | 0.9642 | 0.95190 | 10123.50 | 0.0988 | 4.3771 | 0.9878 | 488.28 |
| Baseline HNSW | SIFT1M | — | — | — | — | — | — | — |
| Hybrid Attribute-Spatial HNSW | SIFT1M | — | — | — | — | — | — | — |

> **Note:** FAISS `IndexFlatL2` is used as the exact-search baseline. Minor differences from the provided ANN-Benchmarks ground truth may occur due to tie handling or differences in ground-truth generation.

---

# References

1. M. Aumüller, E. Bernhardsson, and A. Faithfull. **ANN-Benchmarks: A Benchmarking Tool for Approximate Nearest Neighbor Algorithms.** *Information Systems*, 2019.

2. Y. Malkov and D. Yashunin. **Efficient and Robust Approximate Nearest Neighbor Search Using Hierarchical Navigable Small World Graphs.** *IEEE TPAMI*, 2020.

3. J. Johnson, M. Douze, and H. Jégou. **Billion-Scale Similarity Search with GPUs.** *IEEE Big Data*, 2019.

4. H. Jégou, M. Douze, and C. Schmid. **Product Quantization for Nearest Neighbor Search.** *IEEE TPAMI*, 2011.