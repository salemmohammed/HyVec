# Nearest-Neighbor Search: Problem Formulation

## 1. Setup

Given a dataset of vectors

$$
\mathcal{X} = \{x_1, x_2, \ldots, x_N\} \subset \mathbb{R}^{d}
$$

and a query vector $q \in \mathbb{R}^{d}$, the **nearest-neighbor search (NNS)** problem asks us to find the point in $\mathcal{X}$ that is closest to $q$:

$$
x^{*} = \arg\min_{x \in \mathcal{X}} \delta(x, q)
$$

where $\delta(\cdot, \cdot)$ denotes the **Euclidean (L2) distance**:

$$
\delta(x, q) = \|x - q\|_2 = \sqrt{\sum_{i=1}^{d} (x_i - q_i)^2}
$$

| Symbol | Meaning |
|---|---|
| $\mathcal{X}$ | Dataset of $N$ vectors (e.g. embeddings) |
| $d$ | Dimensionality of each vector |
| $q$ | Query vector, same dimensionality as $x_i$ |
| $\delta(x,q)$ | Euclidean distance between $x$ and $q$ |
| $x^{*}$ | The true nearest neighbor of $q$ in $\mathcal{X}$ |

The $\arg\min$ notation means: *return the vector $x$ that minimizes the distance*, not the minimum distance value itself.

---

## 2. Exact Search

**Exact search** (brute force) computes $\delta(x_i, q)$ for **every** vector $x_i \in \mathcal{X}$, then selects the minimum.

- ✅ **Guarantees correctness** — always returns the true nearest neighbor.
- ❌ **Cost**: $O(N \cdot d)$ per query.
- ❌ At scale ($N$ in the millions or billions, as with modern embedding datasets), this becomes computationally prohibitive for real-time use.

---

## 3. Approximate Nearest Neighbor Search (ANNS)

**ANNS** relaxes the correctness guarantee in exchange for speed. Instead of scanning the whole dataset, it explores only a **subset of promising candidates**, using specialized index structures such as:

- **Graph-based indexes** — e.g. HNSW
- **Tree-based indexes** — e.g. KD-trees, ball trees
- **Hashing-based indexes** — e.g. LSH
- **Clustering/quantization-based indexes** — e.g. IVF, product quantization

| | Exact Search | ANNS |
|---|---|---|
| Candidates examined | All $N$ vectors | Small subset |
| Correctness | Guaranteed | High probability, not guaranteed |
| Speed | Slow at scale — $O(N \cdot d)$ | Much faster (often orders of magnitude) |
| Typical use case | Small datasets, offline batch jobs | Large-scale, real-time retrieval |

---

## 4. Why This Matters

Systems like semantic search, recommendation engines, and **retrieval-augmented generation (RAG)** pipelines often need to search over millions or billions of embeddings *per query, in real time*. Exact search simply doesn't scale to this setting — which is why ANNS methods are the practical backbone of modern vector search infrastructure, trading a small, usually negligible loss in accuracy for the speed needed to make large-scale retrieval feasible.

---

## 5. Evaluation Metric: Recall@k

Search accuracy is measured using **Recall@k**:

$$
\mathrm{Recall@k} = \frac{|\mathcal{R} \cap \tilde{\mathcal{R}}|}{k}
$$

| Symbol | Meaning |
|---|---|
| $\mathcal{R}$ | Exact top-$k$ result set (ground truth, from brute-force search) |
| $\tilde{\mathcal{R}}$ | Approximate result set (what the ANNS method actually returned) |
| $\lvert \mathcal{R} \cap \tilde{\mathcal{R}} \rvert$ | Number of approximate results that match the true top-$k$ |

- **Recall@k $= 1.0$** → perfect match, the ANNS method found every true nearest neighbor.
- **Recall@k $= 0$** → no overlap at all with the ground truth.

### The Recall–QPS Trade-off

The primary performance trade-off in ANNS is between **Recall@k** and **Queries Per Second (QPS)**:

- Exploring **more candidates** → higher recall, but **lower QPS** (slower).
- Exploring **fewer candidates** → higher QPS (faster), but **lower recall**.

This trade-off is typically visualized as a **Recall–QPS curve**, where different ANNS methods (or different parameter settings of the same method) are plotted against each other. A method is considered better if it achieves:

- higher QPS at the same recall level, **or**
- higher recall at the same QPS.

### The Recall–QPS Trade-off

The primary performance trade-off in ANNS is between **Recall@k** and **Queries Per Second (QPS)**:

- Exploring **more candidates** → higher recall, but **lower QPS** (slower).
- Exploring **fewer candidates** → higher QPS (faster), but **lower recall**.

This trade-off is typically visualized as a **Recall–QPS curve**, where different ANNS methods (or different parameter settings of the same method) are plotted against each other. A method is considered better if it achieves:

- higher QPS at the same recall level, **or**
- higher recall at the same QPS.
