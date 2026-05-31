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

| File | Algorithm | Type | From |
|---|---|---|---|
| `faiss_ivf.py` | FAISS IVF | Partition | Facebook |
| `brute_force.py` | Brute Force | Exact | Reference |

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

### Our clustered HNSW
```bash
g++ -O3 -std=c++17 clustered_hnsw.cpp -o clustered_hnsw -I../../hnswlib
./clustered_hnsw --dataset ../data/sift-128-euclidean.hdf5
```

### FAISS IVF
```bash
pip3 install faiss-cpu h5py
python3 faiss_ivf.py --dataset ../data/sift-128-euclidean.hdf5
```

### Brute force (exact search, recall=1.0)
```bash
python3 brute_force.py --dataset ../data/sift-128-euclidean.hdf5
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