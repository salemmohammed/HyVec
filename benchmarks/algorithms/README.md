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
