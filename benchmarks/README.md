# Benchmarks

This directory contains the benchmarking framework used to evaluate the proposed **Hybrid Attribute-Spatial HNSW** index against representative **Approximate Nearest Neighbor (ANN)** indexing techniques. The benchmark is built on the **ANN-Benchmarks** framework, providing a standardized and reproducible environment for comparing search accuracy, throughput, index construction time, query latency, and memory consumption.

The evaluation considers representative ANN methods from the major indexing families:

- **Exact search:** Brute Force (ground-truth reference).
- **Hash-based methods:** Locality-Sensitive Hashing (LSH).
- **Cluster-based methods:** IVF and IVFADC.
- **Partition and quantization methods:** ScaNN.
- **Graph-based methods:** HNSW.
- **Disk-based graph methods:** DiskANN.
- **Hybrid methods:** Hybrid Attribute-Spatial HNSW (proposed).

Experiments are conducted using the **SIFT1M** benchmark dataset to evaluate conventional ANN search performance and an **attributed vector dataset** to evaluate hybrid attribute-aware vector search. While SIFT1M provides a standardized benchmark for comparing ANN indexing techniques, the attributed dataset assesses the effectiveness of combining attribute filtering, spatial clustering, and local HNSW search.

Performance is evaluated using **Recall@k**, **Queries Per Second (QPS)**, **index construction time**, **query latency**, **memory consumption**, and **index size** under identical hardware and software configurations. The benchmarking framework enables reproducible and fair comparisons between the proposed Hybrid Attribute-Spatial HNSW index and representative state-of-the-art ANN indexing techniques.

---

## Hardware Platform

All benchmark experiments are conducted on an **NVIDIA DGX Spark** workstation. The hardware configuration used throughout the evaluation is summarized below.

| Component | Specification |
|-----------|---------------|
| **System** | NVIDIA DGX Spark |
| **Processor** | NVIDIA GB10 Grace Blackwell |
| **CPU** | ARM Cortex @ 3.8 GHz |
| **Memory** | 128 GB Unified Memory |
| **Storage** | 4 TB NVMe SSD |
| **Operating System** | NVIDIA DGX OS |

## Datasets

The benchmark uses two datasets to evaluate different aspects of the proposed Hybrid Attribute-Spatial HNSW index.

### Public ANN Benchmark

The **SIFT1M** dataset, provided by the ANN-Benchmarks framework, serves as the primary benchmark for evaluating conventional approximate nearest neighbor search performance.

| Dataset | Dimension | Database Size | Query Size | Distance Metric |
|----------|----------:|--------------:|-----------:|-----------------|
| **SIFT1M** | 128 | 1,000,000 | 10,000 | Euclidean |

### Attributed Dataset

An attributed vector dataset is used to evaluate hybrid vector search with metadata filtering. Unlike SIFT1M, this dataset associates each vector with one or more attributes, enabling evaluation of:

- Attribute-aware filtering
- Spatial clustering
- Local HNSW search
- Hybrid vector retrieval

The attributed dataset complements SIFT1M by evaluating the proposed architecture under realistic filtered vector search workloads.

### Download All Datasets

```bash
cd benchmarks/data
python3 download_datasets.py
```

---

## Supported ANN Algorithms

The benchmarking framework is designed to evaluate representative **Approximate Nearest Neighbor (ANN)** indexing techniques available through the **ANN-Benchmarks** framework. These algorithms span multiple indexing families and provide representative baselines for evaluating the proposed **Hybrid Attribute-Spatial HNSW** index.

### Graph-Based Methods

| Algorithm | Description | Repository |
|-----------|-------------|------------|
| **hnswlib** | Reference HNSW implementation | https://github.com/nmslib/hnswlib |
| **HNSW (NMSLIB)** | HNSW implementation in NMSLIB | https://github.com/nmslib/nmslib |
| **HNSW (FAISS)** | HNSW implementation in FAISS | https://github.com/facebookresearch/faiss |
| **GLASS** | Optimized graph-based ANN search | https://github.com/hhy3/pyglass |
| **DiskANN (Vamana)** | Disk-resident graph-based ANN search | https://github.com/microsoft/diskann |
| **N2** | Graph-based ANN implementation | https://github.com/kakao/n2 |

### Partition-Based Methods

| Algorithm | Description | Repository |
|-----------|-------------|------------|
| **FAISS-IVF** | Inverted File (IVF) index | https://github.com/facebookresearch/faiss |
| **ScaNN** | Partitioning with anisotropic quantization | https://github.com/google-research/google-research/tree/master/scann |
| **PyNNDescent** | Approximate graph construction using NN-Descent | https://github.com/lmcinnes/pynndescent |

### Exact Search

| Algorithm | Description |
|-----------|-------------|
| **Brute Force** | Exact nearest neighbor search used to generate ground truth and compute recall. |

> **Note:** The benchmarking framework supports these ANN implementations through ANN-Benchmarks. Individual benchmark results will be added as experiments are completed on the NVIDIA DGX Spark platform.

---

## Metrics Explained

| Metric | Definition | How We Measure |
|---|---|---|
| **Recall@k** | Fraction of true k-nearest neighbors found | compare results to ground truth |
| **QPS** | Queries per second | `num_queries / search_time` |
| **Build time** | Time to build the index | measured in seconds |
| **Index size** | Memory used by the index | measured in KB |

### Recall formula
```
Recall@k = |found ∩ ground_truth| / k

Example: k=10, found 9 of the true 10 nearest neighbors → Recall = 0.9
```

---

## Experimental Setup

Benchmark experiments are conducted using the **ANN-Benchmarks** framework on an **NVIDIA DGX Spark** workstation. All evaluated ANN methods are executed under identical hardware and software configurations to ensure fair and reproducible comparisons.

The evaluation consists of two complementary studies:

1. **Public Benchmark Evaluation**
   - Dataset: **SIFT1M**
   - Distance metric: Euclidean
   - Evaluation: Recall@k, Queries Per Second (QPS), index construction time, query latency, memory consumption, and index size.

2. **Hybrid Search Evaluation**
   - Dataset: Attributed vector dataset
   - Evaluation: Hybrid attribute-aware vector search using attribute filtering, spatial clustering, and local HNSW search.

As benchmark experiments are completed, performance results for representative ANN indexing techniques—including HNSW, IVF, ScaNN, DiskANN, and the proposed Hybrid Attribute-Spatial HNSW—will be added to this repository.

---

## Repository Structure

```text
benchmarks/
├── README.md
├── data/
│   ├── download_datasets.py
│   └── sift/
│
├── algorithms/
│   ├── baseline_hnsw.cpp
│   ├── clustered_hnsw.cpp
│   ├── faiss_ivf.py
│   └── brute_force.py
│
├── run_benchmark.sh
├── results/
└── figures/
```

---

## Running the Benchmarks

### Run all benchmarks

```bash
cd benchmarks
./run_benchmark.sh
```

### Run ANN-Benchmarks

```bash
git clone https://github.com/erikbern/ann-benchmarks.git
cd ann-benchmarks

pip install -r requirements.txt
python install.py

python run.py \
    --dataset sift-128-euclidean \
    --algorithm hnswlib

python plot.py \
    --dataset sift-128-euclidean
```

---

## Results

Benchmark results, plots, and performance comparisons will be added as experiments are completed.

---

## References

1. M. Aumüller, E. Bernhardsson, and A. Faithfull. **ANN-Benchmarks: A Benchmarking Tool for Approximate Nearest Neighbor Algorithms.** *Information Systems*, 2019.

2. Y. Malkov and D. Yashunin. **Efficient and Robust Approximate Nearest Neighbor Search Using Hierarchical Navigable Small World Graphs.** *IEEE TPAMI*, 2020.

3. J. Johnson, M. Douze, and H. Jégou. **Billion-Scale Similarity Search with GPUs.** *IEEE Big Data*, 2019.

4. H. Jégou, M. Douze, and C. Schmid. **Product Quantization for Nearest Neighbor Search.** *IEEE TPAMI*, 2011.

---