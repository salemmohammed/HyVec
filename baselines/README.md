# Benchmarks

This repository provides a benchmarking framework for evaluating **Approximate Nearest Neighbor (ANN)** indexing algorithms. It serves as a common experimental platform for comparing representative ANN methods under identical datasets, hardware configurations, and evaluation protocols.

The primary objective of this benchmark is to evaluate the performance of our proposed **Hybrid Attribute-Spatial HNSW** index against established ANN techniques. All algorithms are executed using the same benchmark datasets and experimental settings to ensure fair, reproducible, and meaningful comparisons.

Performance is evaluated using standard ANN metrics, including **Recall@k**, **Queries Per Second (QPS)**, **query latency**, **index construction time**, **search time**, **memory consumption**, and **index size**.

---

# Hardware Platform

All benchmark experiments are conducted on the same hardware platform.

| Component            | Specification               |
| -------------------- | --------------------------- |
| **System**           | NVIDIA DGX Spark            |
| **Processor**        | NVIDIA GB10 Grace Blackwell |
| **CPU**              | ARM Cortex @ 3.8 GHz        |
| **Memory**           | 128 GB Unified Memory       |
| **Storage**          | 4 TB NVMe SSD               |
| **Operating System** | NVIDIA DGX OS               |

---

# Datasets

The benchmarking framework currently supports the following datasets:

* **SIFT1M** — Standard ANN benchmark dataset for evaluating conventional nearest neighbor search.
* **Attributed Dataset** — Used to evaluate hybrid vector search with metadata filtering and attribute-aware retrieval.

## Download SIFT1M

```bash
cd benchmarks/data

curl -L \
    -o sift-128-euclidean.hdf5 \
    https://ann-benchmarks.com/sift-128-euclidean.hdf5
```

---

# Supported ANN Algorithms

The benchmarking framework includes representative algorithms from the major ANN indexing families.

| Family                 | Representative Algorithms                |
| ---------------------- | ---------------------------------------- |
| **Exact Search**       | Brute Force                              |
| **Hash-Based**         | Locality-Sensitive Hashing (LSH)         |
| **Cluster-Based**      | IVF, IVFADC                              |
| **Quantization-Based** | ScaNN                                    |
| **Graph-Based**        | HNSW                                     |
| **Disk-Based Graph**   | DiskANN                                  |
| **Hybrid**             | Hybrid Attribute-Spatial HNSW (proposed) |

Additional algorithms may be incorporated as the benchmarking framework evolves.

---

# Evaluation Metrics

All benchmark implementations are evaluated using a common set of performance metrics:

* **Recall@k**
* **Queries Per Second (QPS)**
* **Average Query Latency**
* **Index Construction Time**
* **Search Time**
* **Memory Consumption**
* **Index Size**

---

# Running Benchmarks

All commands assume the current working directory is

```bash
cd benchmarks/algorithms
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

## Hybrid Attribute-Spatial HNSW

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

# Output

Each benchmark records the following performance statistics:

* Recall@1
* Recall@k
* Queries Per Second (QPS)
* Average Query Latency
* Index Construction Time
* Search Time
* Memory Consumption
* Index Size

Benchmark results are automatically appended to:

```text
results/results.csv
```

The generated results can be used to produce Recall–QPS, Recall–Latency, and other comparative performance plots.
