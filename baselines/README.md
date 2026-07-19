# Benchmarks

This repository provides a benchmarking framework for evaluating **baseline Approximate Nearest Neighbor (ANN)** indexing algorithms. It serves as a common experimental platform for comparing representative ANN methods under identical datasets, hardware configurations, and evaluation protocols.

**This repository currently contains only baseline ANN implementations.** Its primary purpose is to establish reference performance for widely used ANN indexing techniques. These baseline results will be used to evaluate our proposed **Hybrid Attribute-Spatial HNSW** index in future work. The proposed method is **not included in this folder**.

All baseline algorithms are evaluated using the same datasets, hardware platform, and experimental settings to ensure fair, reproducible, and meaningful comparisons.

---

# Hardware Platform

All benchmark experiments are conducted on the same hardware platform.

| Component | Specification |
|----------|---------------|
| **System** | NVIDIA DGX Spark |
| **Processor** | NVIDIA GB10 Grace Blackwell |
| **CPU** | ARM Cortex @ 3.8 GHz |
| **Memory** | 128 GB Unified Memory |
| **Storage** | 4 TB NVMe SSD |
| **Operating System** | NVIDIA DGX OS |

---

# Datasets

The benchmarking framework currently supports the following datasets:

- **SIFT1M** — Standard ANN benchmark dataset for evaluating conventional nearest neighbor search.
- **Attributed Dataset** — Used for evaluating hybrid vector search with metadata filtering and attribute-aware retrieval.

## Download SIFT1M

```bash
cd benchmarks/data

curl -L \
    -o sift-128-euclidean.hdf5 \
    https://ann-benchmarks.com/sift-128-euclidean.hdf5
```

---

# Baseline ANN Algorithms

The repository currently includes the following baseline ANN implementations.

| Family | Algorithm |
|---------|-----------|
| **Exact Search** | Brute Force |
| **Cluster-Based** | FAISS-IVF |
| **Graph-Based** | HNSW |

Additional baseline algorithms (e.g., LSH, DiskANN, and ScaNN) may be incorporated as the benchmarking framework evolves.

---

# Evaluation Metrics

All benchmark implementations are evaluated using a common set of performance metrics:

- **Recall@1**
- **Recall@k**
- **Queries Per Second (QPS)**
- **Average Query Latency**
- **Index Construction Time**
- **Search Time**
- **Memory Consumption**
- **Index Size**

---

# Running Benchmarks

All commands assume the current working directory is

```bash
cd baselines
```

## Brute Force

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

## HNSW

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

## FAISS-IVF

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

The implementation automatically evaluates multiple **nprobe** values (`1`, `5`, `10`, `20`, `50`, and `100`) and appends the results to `results.csv`.

---

# Output

Each benchmark records the following performance statistics:

- Recall@1
- Recall@k
- Queries Per Second (QPS)
- Average Query Latency
- Index Construction Time
- Search Time
- Memory Consumption
- Index Size

Benchmark results are automatically appended to

```text
results/results.csv
```

The generated results can be used to produce Recall–QPS, Recall–Latency, Recall–Memory, and other comparative performance plots.
