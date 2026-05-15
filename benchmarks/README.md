# Benchmarks

This benchmark evaluates our clustered HNSW system against standard ANN algorithms including hnswlib, FAISS-IVF, ScaNN, and Annoy by rerunning the ANN-Benchmarks framework to produce comparable results on the same hardware. We evaluate on two datasets: SIFT1M as a standard public benchmark, and our own attributed dataset to validate hybrid search performance.

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

## ANN-Benchmarks Evaluation Framework

ANN-Benchmarks [1] is a standardized evaluation environment for approximate nearest neighbor (ANN) search algorithms. It provides pre-computed datasets with ground-truth nearest neighbors, isolated execution environments per algorithm, and a unified set of metrics including recall, QPS, build time, and index size. To ensure fair comparison, the framework enforces single CPU execution with hyperthreading disabled. We rerun the framework under identical conditions to directly compare our Clustered HNSW system against existing algorithms.

---

## Datasets

All datasets are pre-split into train/test and include ground truth
for the top-100 nearest neighbors. Source: ann-benchmarks.com

| Dataset | Dimensions | Train size | Test size | Distance | Size | Our Use |
|---|---|---|---|---|---|---|
| [SIFT](http://corpus-texmex.irisa.fr/) | 128 | 1,000,000 | 10,000 | Euclidean | 501MB | Primary |
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

## Experimental Setup
 
- **ANN-Benchmarks baseline:** AWS r6i.16xlarge (single CPU, hyperthreading disabled).
- **Our system:** NVIDIA DGX Spark (128GB unified memory, NVIDIA Blackwell GPU).
- All experiments were rerun on our hardware to ensure a fair and consistent comparison across all evaluated algorithms.

### Recall vs QPS

| Method | Recall@1 | QPS | Build Time (s) |
|---|---|---|---|
| hnswlib | — | — | — |
| FAISS-IVF | — | — | — |
| ScaNN | — | — | — |
| Annoy | — | — | — |
| **Clustered HNSW (Ours)** | — | — | — |

> Results will be updated upon completion of experiments on the NVIDIA DGX Spark.

### Build Time

| Method | Build Time (s) | vs Baseline |
|---|---|---|
| Baseline HNSW (1M vectors) | — | 1× |
| **Clustered HNSW (1000 indexes)** | — | TBD |
| hnswlib (ANN-Benchmarks) | — | — |
| FAISS-IVF (ANN-Benchmarks) | — | — |

> Results will be updated upon completion of experiments on the NVIDIA DGX Spark.

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

## Expected Findings

1. **Build time:** 

2. **Recall match:** 

3. **QPS tradeoff:** 

4. **Real-time insertions:**



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

## Figures

See [figures.html](figures.html) for Table 1, Figure 4, Figure 5, and Figure 6 with full citations.
