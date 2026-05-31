# Algorithms

This folder contains implementations of all algorithms we compare against.
Each algorithm can be run on any ANN-Benchmarks dataset.

---

## Our Method

| File | Algorithm | Description |
|---|---|---|
| `clustered_hnsw.cpp` | **Clustered HNSW** | Our contribution — HNSW per cluster |
| `baseline_hnsw.cpp` | Standard HNSW | hnswlib baseline |

---

## Reference Algorithms

| File | Language | Algorithm | Type | Note |
|---|---|---|---|---|
| `brute_force.cpp` | C++ | Brute Force | Exact | Manual implementation — study version |
| `brute_force.py` | Python | Brute Force | Exact | FAISS-backed — same results, less code |
| `faiss_ivf.cpp` | C++ | FAISS IVF | Partition | Manual implementation — study version |
| `faiss_ivf.py` | Python | FAISS IVF | Partition | FAISS-backed — same results, less code |

> The C++ files implement every step manually (k-means, posting lists, flat scan)
> for study purposes. The Python files call FAISS internally and produce identical
> results. Use Python for benchmarking, C++ for understanding the internals.

---

## ANN-Benchmarks Algorithms (external)

These run via the official ANN-Benchmarks framework:

| Algorithm | Type | Best for |
|---|---|---|
| hnswlib | Graph | High recall, fast search |
| glass | Graph | Top QPS performer |
| faiss-ivf | Partition | Large scale |
| scann | Partition | Google production |
| vamana(diskann) | Graph | Disk-based |
| annoy | Tree | Simple, no build step |
| pynndescent | Graph | Python native |
| qdrant | Vector DB | Production systems |

---

## How to Run Each

### Our Clustered HNSW (C++)
```bash
g++ -O3 -std=c++17 -fopenmp clustered_hnsw.cpp -o clustered_hnsw \
    -I../../hnswlib $(pkg-config --cflags --libs hdf5)
./clustered_hnsw --dataset ../data/sift-128-euclidean.hdf5
```

### Brute Force — C++ (manual, for study)
```bash
g++ -O3 -std=c++17 -fopenmp brute_force.cpp -o brute_force \
    $(pkg-config --cflags --libs hdf5)
./brute_force --metric l2 --dataset ../data/sift-128-euclidean.hdf5
```

### Brute Force — Python (FAISS-backed, for benchmarking)
```bash
pip3 install faiss-cpu h5py numpy
python3 brute_force.py --metric l2 --dataset ../data/sift-128-euclidean.hdf5
```

### FAISS IVF — C++ (manual, for study)
```bash
g++ -O3 -std=c++17 -fopenmp faiss_ivf.cpp -o faiss_ivf \
    $(pkg-config --cflags --libs hdf5)
./faiss_ivf --dataset ../data/sift-128-euclidean.hdf5
```

### FAISS IVF — Python (FAISS-backed, for benchmarking)
```bash
pip3 install faiss-cpu h5py numpy
python3 faiss_ivf.py --dataset ../data/sift-128-euclidean.hdf5
```

### Full ANN-Benchmarks suite
```bash
cd /path/to/ann-benchmarks
python run.py --dataset sift-128-euclidean --algorithm hnswlib
python run.py --dataset sift-128-euclidean --algorithm faiss-ivf
python plot.py --dataset sift-128-euclidean
```

---

## References

The following papers provide head-to-head comparisons of ANN algorithms and are
the standard citations used in this area.

### Benchmark Frameworks

1. M. Aumüller, E. Bernhardsson, A. Faithfull: **ANN-Benchmarks: A Benchmarking
   Tool for Approximate Nearest Neighbor Algorithms.** Information Systems, 2019.
   DOI: 10.1016/j.is.2019.02.006 · [ann-benchmarks.com](https://ann-benchmarks.com) · [GitHub](https://github.com/erikbern/ann-benchmarks)

2. H. Simhadri et al.: **Results of the NeurIPS'21 Challenge on Billion-Scale
   Approximate Nearest Neighbor Search.** arXiv:2205.03763, 2022.
   [arxiv.org/abs/2205.03763](https://arxiv.org/abs/2205.03763)

3. M. Douze et al.: **The FAISS Library.** arXiv:2401.08281, 2024.
   [arxiv.org/abs/2401.08281](https://arxiv.org/abs/2401.08281)

### Survey & Comparison Papers

4. W. Li, Y. Zhang, Y. Sun, W. Wang, M. Li, W. Zhang, X. Lin: **Approximate
   Nearest Neighbor Search on High Dimensional Data — Experiments, Analyses,
   and Improvement.** IEEE Transactions on Knowledge and Data Engineering, 2020.
   arXiv:1610.02455 · [arxiv.org/abs/1610.02455](https://arxiv.org/abs/1610.02455)
   *(Most cited head-to-head comparison of LSH, tree, quantization, and graph methods)*

5. M. Iwasaki, D. Miyazaki: **Optimization of Indexing Based on k-Nearest
   Neighbor Graph for Proximity Search in High-dimensional Data.**
   arXiv:1810.07355, 2018.
   [arxiv.org/abs/1810.07355](https://arxiv.org/abs/1810.07355)
