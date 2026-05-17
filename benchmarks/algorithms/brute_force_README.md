# Brute Force — Exact Nearest Neighbor Search

Brute force exact search is the **Recall=1.0 reference baseline** for all
ANN benchmarks. It computes the true nearest neighbors by exhaustively
evaluating the distance between the query vector and every vector in the
dataset. Every ANN algorithm in this project is measured against it.

---

## Role in Evaluation

```
Brute Force defines:
  - Upper bound of Recall  → Recall@k = 1.0  (always exact)
  - Lower bound of QPS     → slowest possible search
  - Ground truth           → true nearest neighbors for computing recall
                             of all other algorithms
```

If an ANN algorithm returns Recall@1 = 0.95, it means 95% of the time
it found the same answer that brute force would have found.

---

## Supported Distance Metrics

This implementation supports all six distance metrics used in
ANN-Benchmarks [3] and the broader ANN literature [5]:

| Metric | Formula | Implementation | Typical Dataset |
|---|---|---|---|
| **L2 (Euclidean)** | √ Σ(xᵢ - qᵢ)² | FAISS IndexFlatL2 | SIFT1M, GIST1M |
| **Inner Product** | Σ xᵢ × qᵢ | FAISS IndexFlatIP | Recommendation, embeddings |
| **Cosine** | (x·q) / (\|x\|\|q\|) | FAISS IndexFlatIP + normalize | Text, NLP embeddings |
| **Angular** | arccos(cosine) | FAISS IndexFlatIP + normalize | GloVe-25, GloVe-100 |
| **L1 (Manhattan)** | Σ \|xᵢ - qᵢ\| | NumPy batch | Sparse data |
| **Hamming** | count differing bits | NumPy binarized | Binary codes, LSH |

---

## Why Different Metrics?

Different data types have different notions of similarity:

**L2 (Euclidean)** — measures absolute geometric distance.
Best for image feature vectors like SIFT [2] where magnitude matters.

**Inner Product** — measures alignment between vectors.
Best for recommendation systems where embeddings are trained with dot
product objectives (e.g. matrix factorization).

**Cosine / Angular** — measures the angle between vectors, ignoring magnitude.
Best for text and NLP embeddings [4] where direction matters but scale
does not — two documents can be similar regardless of length.

**L1 (Manhattan)** — sums absolute differences dimension by dimension.
Less sensitive to outlier dimensions than L2. Used for sparse data.

**Hamming** — counts differing bits between binary codes.
Used when vectors have been compressed to binary representations via
locality-sensitive hashing (LSH) or learning-to-hash methods [5].

---

## Implementation

### L2, IP, Cosine, Angular — FAISS IndexFlat

We use FAISS [1] for L2, inner product, cosine, and angular metrics.
FAISS IndexFlat performs the same exhaustive scan as a naive numpy loop
but uses optimized BLAS routines and SIMD instructions — 10-100× faster
while remaining mathematically exact.

```python
# L2 — direct Euclidean distance
index = faiss.IndexFlatL2(d)

# Inner Product — dot product similarity
index = faiss.IndexFlatIP(d)

# Cosine / Angular — normalize first, then inner product
faiss.normalize_L2(vectors)     # |x| = 1 for all vectors
index = faiss.IndexFlatIP(d)    # dot product on unit vectors = cosine
```

### L1, Hamming — NumPy

FAISS does not support L1 or Hamming distance for float vectors natively.
We implement these using batched NumPy operations:

```python
# L1 — sum of absolute differences
dists = np.sum(np.abs(base - query), axis=1)

# Hamming — binarize then XOR and count
base_bin  = (base > 0).astype(np.uint8)
query_bin = (query > 0).astype(np.uint8)
dists     = np.sum(base_bin ^ query_bin, axis=1)
```

---

## Recall Formula

```
Recall@k = |R ∩ R̃| / k

  R  = ground truth top-k neighbors (from brute force or dataset GT)
  R̃  = top-k neighbors returned by the algorithm

Example: k=10, algorithm returns 9 of the true 10 → Recall@10 = 0.9
```

For brute force, R̃ == R by definition → Recall@k = 1.0 always.
We verify this on every run to confirm correctness.

Reference: Aumüller et al. [3], Section 3.

---

## Complexity

| Step | Complexity | Notes |
|---|---|---|
| Build | O(1) | No index constructed |
| Query | O(N × d) | Linear scan — all N vectors, d dimensions |
| Memory | O(N × d) | Raw float32 vectors only |

For SIFT1M: N=1,000,000, d=128 → 128M multiply-adds per query.

---

## Usage

```bash
# Install dependencies
pip3 install faiss-cpu numpy h5py

# L2 distance — SIFT1M (default)
python3 brute_force.py --metric l2

# Inner product — recommendation embeddings
python3 brute_force.py --metric ip

# Cosine similarity — text embeddings
python3 brute_force.py --metric cosine --dataset ../data/glove-25-angular.hdf5

# Angular distance — GloVe datasets
python3 brute_force.py --metric angular --dataset ../data/glove-100-angular.hdf5

# L1 Manhattan — sparse data
python3 brute_force.py --metric l1

# Hamming — binary codes
python3 brute_force.py --metric hamming

# Custom k
python3 brute_force.py --metric l2 --k 100
```

---

## Output

Results are printed to the console and appended to `../results.csv`:

```
================================================================
  Brute Force Exact Search — Recall=1.0 Reference Baseline
  Metric  : L2 Euclidean distance — FAISS IndexFlatL2
  Dataset : sift
  k       : 10
================================================================

  Recall@1    : 1.0000  (expected: 1.0000)
  Recall@10   : 1.0000  (expected: 1.0000)
  QPS         : 142.3 queries/second
  Search time : 70.28s  (10,000 queries)
  Build time  : 0s  (no index — exhaustive scan)
  Index size  : 488.3 MB  (raw float32 vectors)
```

CSV columns: `timestamp, machine, algorithm, recall@1, recall@k, qps,
build_time_s, search_time_s, num_queries, dataset, metric`

---

## References

[1] J. Johnson, M. Douze, H. Jégou, "Billion-Scale Similarity Search
with GPUs," IEEE Transactions on Big Data, 7(3): 535–547, 2021.
[arXiv:1702.08734](https://arxiv.org/abs/1702.08734)
[GitHub](https://github.com/facebookresearch/faiss)

[2] H. Jégou, M. Douze, C. Schmid, "Product Quantization for Nearest
Neighbor Search," IEEE TPAMI, 33(1): 117–128, 2011.
[Dataset](http://corpus-texmex.irisa.fr/)

[3] M. Aumüller, E. Bernhardsson, A. Faithfull, "ANN-Benchmarks: A
Benchmarking Tool for Approximate Nearest Neighbor Algorithms,"
Information Systems, 87, 2020.
[DOI:10.1016/j.is.2019.02.006](https://doi.org/10.1016/j.is.2019.02.006)

[4] J. Pennington, R. Socher, C. Manning, "GloVe: Global Vectors for
Word Representation," EMNLP, 2014.
[Link](https://nlp.stanford.edu/projects/glove/)

[5] M. Wang, X. Xu, Q. Yue, Y. Wang, "A Comprehensive Survey and
Experimental Comparison of Graph-Based Approximate Nearest Neighbor
Search," PVLDB, 14(11): 1964–1978, 2021.
[Link](https://www.vldb.org/pvldb/vol14/p1964-wang.pdf)
