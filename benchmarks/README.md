# Benchmarks

This folder contains everything needed to compare our system against
ANN-Benchmarks — the standard evaluation framework for approximate
nearest neighbor search algorithms.

---

## Citation

All benchmark comparisons in this project use the ANN-Benchmarks framework:

> M. Aumüller, E. Bernhardsson, A. Faithfull:
> **ANN-Benchmarks: A Benchmarking Tool for Approximate Nearest Neighbor Algorithms.**
> Information Systems, 2019.
> DOI: [10.1016/j.is.2019.02.006](https://doi.org/10.1016/j.is.2019.02.006)

Website: [https://ann-benchmarks.com](https://ann-benchmarks.com)
GitHub:  [https://github.com/erikbern/ann-benchmarks](https://github.com/erikbern/ann-benchmarks)

---

## What ANN-Benchmarks Is

ANN-Benchmarks is a standardized benchmarking environment for approximate
nearest neighbor (ANN) search algorithms. It provides:

- Pre-computed datasets in HDF5 format with ground truth
- Docker containers for each algorithm
- Standardized evaluation metrics (Recall, QPS, build time, index size)
- A public leaderboard at ann-benchmarks.com

The benchmark enforces single-CPU execution so results are comparable
across hardware. All results on ann-benchmarks.com were run on an
AWS r6i.16xlarge machine with `--parallelism 31` and hyperthreading disabled.

---

## What the Lines Represent

Each line on the ANN-Benchmarks plot = one algorithm.
Each point on a line = one parameter configuration (e.g. ef=10, ef=50, ef=100).

```
                High QPS
                    ↑
                    │    ★ glass
                    │   ╱ hnswlib
                    │  ╱  scann
                    │ ╱   faiss-ivf
                    │╱    annoy
                    └──────────────→ High Recall
                  Low             High
```

A curve that is **higher and to the right** is strictly better —
it achieves more queries per second at any given recall level.

---

## Datasets

All datasets are pre-split into train/test and include ground truth
for the top-100 nearest neighbors. Source: ann-benchmarks.com

| Dataset | Dimensions | Train size | Test size | Distance | Size | Our Use |
|---|---|---|---|---|---|---|
| [SIFT](http://corpus-texmex.irisa.fr/) | 128 | 1,000,000 | 10,000 | Euclidean | 501MB | ✅ Primary |
| [GIST](http://corpus-texmex.irisa.fr/) | 960 | 1,000,000 | 1,000 | Euclidean | 3.6GB | Future |
| [GloVe-25](http://nlp.stanford.edu/projects/glove/) | 25 | 1,183,514 | 10,000 | Angular | 121MB | Future |
| [GloVe-100](http://nlp.stanford.edu/projects/glove/) | 100 | 1,183,514 | 10,000 | Angular | 463MB | Future |
| [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist) | 784 | 60,000 | 10,000 | Euclidean | 217MB | Future |
| [NYTimes](https://archive.ics.uci.edu/ml/datasets/bag+of+words) | 256 | 290,000 | 10,000 | Angular | 301MB | Future |
| [DEEP1B](http://sites.skoltech.ru/compvision/noimi/) | 96 | 9,990,000 | 10,000 | Angular | 3.6GB | Future |
| [Last.fm](https://github.com/erikbern/ann-benchmarks/pull/91) | 65 | 292,385 | 50,000 | Angular | 135MB | Future |

### Download All Datasets

```bash
cd benchmarks/data
python3 download_datasets.py
```

---

## Algorithms Compared

These are the main algorithms on ANN-Benchmarks relevant to our work.
Full list: [ann-benchmarks.com/#algorithms](https://ann-benchmarks.com/index.html#algorithms)

### Graph-Based (same family as our work)
| Algorithm | Description | GitHub |
|---|---|---|
| **hnswlib** | Our base algorithm — HNSW graph search | [nmslib/hnswlib](https://github.com/nmslib/hnswlib) |
| hnsw(nmslib) | HNSW in NMSLIB | [nmslib/nmslib](https://github.com/nmslib/nmslib) |
| hnsw(faiss) | HNSW in FAISS | [facebookresearch/faiss](https://github.com/facebookresearch/faiss) |
| glass | Top performer, HNSW variant | [hhy3/pyglass](https://github.com/hhy3/pyglass) |
| vamana(diskann) | Microsoft DiskANN | [microsoft/diskann](https://github.com/microsoft/diskann) |
| n2 | Kakao HNSW implementation | [kakao/n2](https://github.com/kakao/n2) |

### Partition-Based (related to our clustering approach)
| Algorithm | Description | GitHub |
|---|---|---|
| **faiss-ivf** | IVF index (clusters + flat search) | [facebookresearch/faiss](https://github.com/facebookresearch/faiss) |
| scann | Google ScaNN (partitioned) | [google-research/scann](https://github.com/google-research/google-research/tree/master/scann) |
| annoy | Spotify Annoy (trees) | [spotify/annoy](https://github.com/spotify/annoy) |
| pynndescent | PyNNDescent | [lmcinnes/pynndescent](https://github.com/lmcinnes/pynndescent) |

### Vector Databases
| Algorithm | Description | GitHub |
|---|---|---|
| qdrant | Qdrant vector database | [qdrant/qdrant](https://github.com/qdrant/qdrant) |
| weaviate | Weaviate vector database | [weaviate/weaviate](https://github.com/weaviate/weaviate) |
| pgvector | PostgreSQL vector extension | [pgvector/pgvector](https://github.com/pgvector/pgvector) |
| Milvus(Knowhere) | Milvus vector database | [milvus-io/milvus](https://github.com/milvus-io/milvus) |

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

## Our Results vs ANN-Benchmarks (SIFT-128-Euclidean)

Hardware note: ANN-Benchmarks runs on AWS r6i.16xlarge (single CPU, no hyperthreading).
Our Mac results are lower QPS due to hardware difference. Shape of curve matters more than raw numbers.

### Recall vs QPS

| Method | Recall@1 | QPS (our Mac) | QPS (ANN-benchmarks server) |
|---|---|---|---|
| hnswlib ef=10 | 0.713 | — | 69,662 |
| hnswlib ef=50 | 0.950 | — | 28,021 |
| hnswlib ef=100 | 0.985 | — | 16,108 |
| **Our baseline** ef=50 | **0.9686** | **7,443** | — |
| **Clustered TOP=1** | **0.4359** | **9,785** | — |
| **Clustered TOP=3** | **0.6977** | **5,301** | — |
| **Clustered TOP=5** | **0.8062** | **3,603** | — |
| **Clustered TOP=10** | **0.9117** | **1,998** | — |
| **Clustered TOP=20** | **0.9689** | **1,056** | — |
| **Clustered TOP=50** | **0.9960** | **435** | — |

### Build Time

| Method | Build time | vs Baseline |
|---|---|---|
| Baseline HNSW (1M vectors) | 1013s | 1× |
| **Clustered HNSW (1000 indexes)** | **66s** | **15× faster** |
| ANN-benchmarks hnswlib | ~120s (server) | — |

---

## Folder Structure

```
benchmarks/
├── README.md                  ← this file
├── data/
│   ├── download_datasets.py   ← download all ANN-benchmark datasets
│   └── sift/                  ← SIFT1M data (gitignored)
│
├── algorithms/
│   ├── README.md              ← algorithm descriptions
│   ├── baseline_hnsw.cpp      ← standard hnswlib (our baseline)
│   ├── clustered_hnsw.cpp     ← our method (main contribution)
│   ├── faiss_ivf.py           ← FAISS IVF comparison
│   └── brute_force.py        ← exact search (recall=1.0 reference)
│
├── run_benchmark.sh           ← run all algorithms, save results
├── results.csv                ← all benchmark results (auto-generated)
└── dashboard.html             ← interactive results viewer
```

---

## How to Run

### Quick comparison (our methods only)
```bash
cd benchmarks
./run_benchmark.sh
```

### Full ANN-Benchmarks comparison
```bash
# Install ANN-Benchmarks
git clone https://github.com/erikbern/ann-benchmarks.git
cd ann-benchmarks
pip install -r requirements.txt
python install.py

# Run on SIFT dataset
python run.py --dataset sift-128-euclidean --algorithm hnswlib
python plot.py --dataset sift-128-euclidean
```

---

## Key Findings

1. **Build time:** Our clustered approach builds 15× faster than standard HNSW
   (66s vs 1013s on SIFT1M)

2. **Recall match:** With TOP_CLUSTERS=20, our method matches standard HNSW
   recall (0.9689 vs 0.9686) while searching only 2% of data

3. **QPS:** At TOP_CLUSTERS=1, our method achieves 34% higher QPS than
   standard HNSW at lower recall — useful for latency-critical applications

4. **Real-time:** 5,155 insertions/second with zero search downtime,
   using Ada-IVF inspired local re-clustering

---

## References

1. M. Aumüller, E. Bernhardsson, A. Faithfull: **ANN-Benchmarks: A Benchmarking
   Tool for Approximate Nearest Neighbor Algorithms.** Information Systems, 2019.
   DOI: 10.1016/j.is.2019.02.006

2. Y. Malkov, D. Yashunin: **Efficient and Robust Approximate Nearest Neighbor
   Search Using Hierarchical Navigable Small World Graphs.** IEEE TPAMI, 2020.
   arXiv: 1603.09320

3. J. Johnson, M. Douze, H. Jégou: **Billion-Scale Similarity Search with GPUs.**
   IEEE Big Data, 2021. arXiv: 1702.08734

4. H. Jégou, M. Douze, C. Schmid: **Product Quantization for Nearest Neighbor
   Search.** IEEE TPAMI, 2011. DOI: 10.1109/TPAMI.2010.57

5. J. Mohoney et al.: **Incremental IVF Index Maintenance for Streaming Vector
   Search (Ada-IVF).** arXiv: 2411.00970, 2024.
