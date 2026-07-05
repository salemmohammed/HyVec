# HyVec: An Online Hybrid Vector Search Engine for Approximate Nearest Neighbor Search

## Clustered Attribute-Based Vector Search

In this project, we employ a hybrid partitioning strategy that combines attribute-based filtering with spatial clustering. The dataset is first partitioned according to its metadata attributes. Since vectors sharing the same attribute may be distributed across different regions of the embedding space, each attribute partition is further divided into multiple spatial clusters using MiniBatch K-Means[16]. Each cluster is represented by a centroid, which is stored for query routing.

An independent HNSW index [1] is then constructed for every final spatial cluster. At query time, the query is first filtered using its attribute, limiting the search to the relevant attribute partition. The query vector is then compared with the centroids of all clusters within that partition, and the nearest nprobe clusters are selected using an IVF-inspired routing strategy[2]. Instead of searching all vectors associated with the attribute, only the HNSW indices of the selected clusters are traversed. Each HNSW graph performs greedy hierarchical search to efficiently identify approximate nearest neighbors. Finally, the candidate results from all searched clusters are merged and ranked by similarity, and the top-k nearest neighbors are returned.

---

## Research Questions

| # | Question | Theme |
|---|---|---|
| RQ1 | **RQ1:** Can centroid-based partitioning improve query throughput while maintaining recall comparable to state-of-the-art vector search indexes over large-scale datasets? | **Efficiency** |
| RQ2 | Can centroid-based partitioning improve query throughput while maintaining high recall across datasets with different levels of attribute availability? | **Impact of Attribute Availability on Hybrid Vector Search** |
| RQ3 | As new vectors are continuously inserted into HyVec, how quickly do Recall@1 and QPS degrade when the original cluster centroids are not updated? | **Online Insertion Without Re-Clustering** |
| RQ4 | Can background re-clustering with atomic index swapping recover Recall@1 and QPS after online insertions while minimizing query interruption, and what is the maintenance overhead? | **Background Re-Clustering and Atomic Index Swapping** |
| RQ5 | Does HyVec’s clustered search scale more efficiently than a standard unpartitioned HNSW index as dataset size increases? | **Scalability with Dataset Growth** |

---

## Partitioning Strategies

Partitioning is a fundamental technique for improving the scalability of vector search by dividing a large dataset into smaller, manageable subsets. Rather than searching every vector, the query is first routed to one or more relevant partitions, and similarity search is performed only within those partitions. By reducing the search space, partitioning significantly improves query throughput (QPS) while maintaining high search accuracy. However, searching too few partitions may reduce Recall@1, creating a tradeoff between efficiency and accuracy [1], [2].

### Attribute-Based Partitioning

Attribute-based partitioning organizes vectors according to metadata such as category, document type, geographic region, user identifier, or tenant identifier. During query processing, metadata filters identify the relevant partitions before vector similarity search is executed. This approach is particularly effective when metadata is complete, accurate, and highly selective, but its performance deteriorates when attributes are missing, inconsistent, or poorly correlated with vector similarity [17].
### Spatial Partitioning

Spatial partitioning organizes vectors according to their positions in the embedding space using clustering algorithms such as K-Means or MiniBatchKMeans. During query processing, the query vector is assigned to the nearest cluster centroid(s), and similarity search is performed only within those clusters. This approach forms the basis of the Inverted File (IVF) index and is one of the most widely adopted partitioning techniques in modern vector databases because it substantially reduces search complexity while preserving high recall [2], [3].

### Hybrid Partitioning

Hybrid partitioning combines metadata-based filtering with spatial clustering. The dataset is first partitioned using attributes and then clustered according to vector similarity within each partition. This strategy leverages both semantic filtering and vector locality to improve search efficiency while maintaining high recall [14].

### Random Partitioning

Random partitioning assigns vectors to partitions without considering either metadata or vector similarity. Although this strategy generally produces balanced partition sizes and serves as a useful experimental baseline, it does not preserve spatial locality. Consequently, nearest neighbors are often distributed across multiple partitions, requiring additional searches and typically resulting in lower recall and reduced search efficiency than similarity-aware partitioning methods [9].

### No Partitioning

Without partitioning, all vectors are stored in a single index, and every query searches the complete dataset. This approach eliminates routing errors and partition-induced recall loss, thereby achieving the highest possible search accuracy. However, its computational cost increases with dataset size, making exhaustive search impractical for large-scale vector databases despite its optimal recall [2].

## Core Idea

Standard HNSW traverses a subset of the full graph on every query, but that graph still contains all 1M vectors, causing the search space to grow as the dataset scales. This project partitions vectors into attribute-based clusters, allowing each query to be routed to a single small cluster before HNSW search is performed. As a result, the search space remains bounded and largely independent of the total dataset size.

A key advantage of this design is support for real-time updates. New vectors can be inserted directly into their corresponding clusters without rebuilding a global index, enabling the system to maintain low-latency search performance while continuously ingesting new data.

```
Standard HNSW:
  Query --> traverses a small subset of the 1M graph
        --> controlled by ef_search parameter
        --> visits O(ef × log N) nodes approximately
        --> search space grows with dataset size N

Clustered HNSW (this project):

  Offline (build time):
    1,000,000 vectors  --> partitioned into K clusters
    Each cluster       --> represented by 1 centroid vector
    Each cluster       --> gets its own small HNSW index (~1,000 vectors)

  Online (query time):
    Query --> compared against K centroids only
          --> routed to nearest cluster
          --> HNSW runs on that cluster only (~1,000 vectors)
          --> search space bounded by cluster size
          --> independent of total dataset size N
```

---

## System Architecture

```mermaid
flowchart TD
    A([Raw Vectors\n1M × 128-dim]) --> B[K-Means Clustering\nK = 1000]
    B --> C{Cluster Assignment\nAttribute ID 0–999}
    C --> D1[Cluster 0\nHNSW Index]
    C --> D2[Cluster 1\nHNSW Index]
    C --> D3[· · ·]
    C --> D4[Cluster 999\nHNSW Index]

    E([Query Vector]) --> F[Find Nearest Centroid]
    F --> G{Route to\nCluster ID}
    G --> D1
    G --> D2
    G --> D4

    D1 --> H([Top-K Results])
    D2 --> H
    D4 --> H

    style A fill:#4C9BE8,color:#fff,stroke:#2563EB
    style E fill:#4C9BE8,color:#fff,stroke:#2563EB
    style H fill:#16A34A,color:#fff,stroke:#15803D
    style B fill:#7C3AED,color:#fff,stroke:#6D28D9
    style F fill:#7C3AED,color:#fff,stroke:#6D28D9
    style C fill:#D97706,color:#fff,stroke:#B45309
    style G fill:#D97706,color:#fff,stroke:#B45309
```

---

## Real-Time Insertion & Re-Clustering

```mermaid
flowchart TD
    A([New Vector\nInserted]) --> B[Assign to Nearest Cluster]
    B --> C{N insertions\nreached?}
    C -- No --> D[Insert into\nexisting cluster index]
    C -- Yes --> E[Background Thread\nRe-runs K-Means]
    E --> F[Build New\nHNSW Indexes]
    F --> G[Atomic Swap\nold ➜ new index]
    G --> H([Search Continues\nUninterrupted])
    D --> H

    style A fill:#4C9BE8,color:#fff,stroke:#2563EB
    style H fill:#16A34A,color:#fff,stroke:#15803D
    style E fill:#7C3AED,color:#fff,stroke:#6D28D9
    style F fill:#7C3AED,color:#fff,stroke:#6D28D9
    style G fill:#D97706,color:#fff,stroke:#B45309
    style C fill:#D97706,color:#fff,stroke:#B45309
```

---

## Algorithm Pipeline

```mermaid
sequenceDiagram
    participant D as Dataset (SIFT1M)
    participant KM as K-Means (K=1000)
    participant IDX as Cluster Indexes
    participant Q as Query Engine

    D->>KM: 1M × 128-dim vectors
    KM->>KM: Assign cluster IDs (attributes)
    KM->>IDX: Build 1 HNSW index per cluster
    Note over IDX: 1000 indexes × ~1000 vectors each

    Q->>KM: Query vector
    KM->>Q: Nearest centroid → Attribute ID
    Q->>IDX: Search only that cluster's index
    IDX->>Q: Top-K approximate neighbors
    Note over Q: 1000× smaller search space
```

---

## Dataset: SIFT1M

| Property | Value |
|---|---|
| Base vectors | 1,000,000 |
| Query vectors | 10,000 |
| Dimensions | 128 |
| Data type | float32 |
| Distance metric | L2 (Euclidean) |
| Size | ~500 MB |

```bash
cd experiments
mkdir sift && cd sift
wget ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz
tar -xzf sift.tar.gz
mv sift/* . && rm -rf sift sift.tar.gz
```

---

## Key Concepts

| Concept | Definition |
|---|---|
| Recall@1 | % of queries where true nearest neighbor was found |
| QPS | Query throughput, measured as the average number of queries processed per second |
| K (results) | Number of nearest neighbors to return |
| ef | Search exploration factor; controls the size of the candidate list maintained during graph traversal. Larger values typically improve recall at the cost of higher query latency. |
| M | Graph connectivity; neighbors per node |
| Cluster | A partition of vectors sharing similar spatial properties |
| Attribute | The cluster ID assigned to each vector |

---

## Libraries

| Library | Role | Source |
|---|---|---|
| hnswlib | Core HNSW algorithm (header-only C++) | [nmslib/hnswlib](https://github.com/nmslib/hnswlib) |
| FAISS | Alternative index (IVF, PQ, HNSW + GPU) | [facebookresearch/faiss](https://github.com/facebookresearch/faiss) |

Both are included as Git submodules pointing to author forks.

---

## Approach: Attribute-Based Clustering (K=1000)

```
Step 1: Run K-Means (K=1000) on 1M SIFT vectors
        Each cluster = one attribute
        Attribute value = cluster ID (0 to 999)

Step 2: Build one HNSW index per cluster
        1000 indexes × ~1000 vectors each

Step 3: At query time:
        find nearest centroid → attribute ID
        search only that cluster's index
        return top K results

Step 4 (real-time):
        background thread re-clusters every N insertions
        atomic swap — search never stops
```

---

## Evaluation

| Metric | Description |
|---|---|
| Build Time | Index construction time |
| Recall@1 | Top-1 nearest-neighbor accuracy |
| QPS | Queries processed per second |
| Search Space | Vectors searched per query |
| Update Cost | Insert and maintenance cost |
| Scalability | Performance as dataset size increases |

## Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/salemmohammed/my-vector-search.git
cd my-vector-search
```

### 2. Install Dependencies

**macOS**

```bash
brew install cmake
pip3 install numpy scikit-learn
```

**Ubuntu**

```bash
sudo apt update
sudo apt install -y build-essential cmake python3 python3-pip wget
pip3 install numpy scikit-learn
```

### 3. Download the SIFT1M Dataset

```bash
cd experiments
mkdir -p sift
cd sift

wget ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz
tar -xzf sift.tar.gz

mv sift/* .
rm -rf sift sift.tar.gz

cd ../..
```

### 4. Run Experiments

See `experiments/README.md` for detailed instructions on running the baseline HNSW and clustered HNSW evaluations.

---

## Related Work

This project is positioned within a growing body of research on **filtered and partitioned approximate nearest neighbor (ANN) search**:

- **HNSW** — Malkov & Yashunin (2018) introduced the hierarchical navigable small world graph, which achieves near-logarithmic search behavior with strong recall at millisecond latencies. HNSW consistently ranks at the Pareto-optimal front on standard ANN benchmarks such as ANN-Benchmarks [[1]](#references).

- **FAISS** — Johnson et al. (2017) introduced FAISS, which provides efficient IVF (inverted file) indexing by clustering vectors with K-Means and searching only the nearest cluster centroids at query time [[2]](#references). This is the foundational technique our approach extends.

- **IVF Cluster Routing** — Traditional IVF methods partition vectors into clusters based on distance to centroids, then at query time identify the most promising clusters and perform exact search only within those posting lists [[3]](#references). Our approach applies this partitioning idea at a finer granularity, with per-cluster HNSW indexes instead of flat lists.

- **ACORN** — Patel et al. (2024) proposed ACORN for predicate-agnostic filtered vector search, leveraging HNSW with denser vertex neighborhoods to maintain graph connectivity under attribute filters [[4]](#references).

- **Compass** — Ye et al. (2025) introduced Compass, combining IVF-based cluster routing with a proximity graph, demonstrating that hybrid partition + graph indexing achieves competitive QPS and recall across selectivity levels [[5]](#references).

- **Curator** — Engels et al. (2026) showed that partition-based approaches like per-label HNSW indexes can eliminate query-time filtering overhead but require careful tradeoffs between memory and search performance [[6]](#references).

- **Filtered ANNS Survey** — Recent work on filtered approximate nearest neighbor search demonstrates that HNSW's performance degrades in low-selectivity scenarios, where partition-based methods may be preferable, and that hybrid indexing can be necessary for optimal performance across the full selectivity spectrum [[7]](#references).

- **Insertion Order Effects** — Elliott & Clark (2024) showed that HNSW recall is significantly influenced by data insertion sequence and intrinsic dimensionality, with insertion order informed by known categories shifting recall by up to 12 percentage points [[8]](#references). This directly motivates our cluster-aware insertion strategy.

- **ANN-Benchmarks** — Aumüller et al. (2018) established the standard benchmark for evaluating ANN indexes on datasets including SIFT1M, measuring recall@k, QPS, build time, and memory footprint [[9]](#references).

- **Clustered Hybrid Search (Aslam et al.)** — A directly related project on clustered hybrid search that this work builds upon [[10]](#references).

---

## References

<a name="references"></a>

[1] Y. A. Malkov and D. A. Yashunin, "Efficient and Robust Approximate Nearest Neighbor Search Using Hierarchical Navigable Small World Graphs," *IEEE Transactions on Pattern Analysis and Machine Intelligence*, vol. 42, no. 4, pp. 824–836, 2020. [arXiv:1603.09320](https://arxiv.org/abs/1603.09320)

[2] J. Johnson, M. Douze, and H. Jégou, "Billion-Scale Similarity Search with GPUs," *IEEE Transactions on Big Data*, vol. 7, no. 3, pp. 535–547, 2021. [arXiv:1702.08734](https://arxiv.org/abs/1702.08734)

[3] H. Jégou, M. Douze, and C. Schmid, "Product Quantization for Nearest Neighbor Search," *IEEE Transactions on Pattern Analysis and Machine Intelligence*, vol. 33, no. 1, pp. 117–128, 2011. *(SIFT1M dataset also introduced here.)* [Link](http://corpus-texmex.irisa.fr/)

[4] L. Patel, P. Kraft, C. Guestrin, and M. Zaharia, "ACORN: Performant and Predicate-Agnostic Search Over Vector Embeddings and Structured Data," *Proc. ACM Management of Data*, 2024. [arXiv:2403.04871](https://arxiv.org/abs/2403.04871)

[5] C. Ye et al., "Compass: General Filtered Search across Vector and Structured Data," *arXiv preprint*, 2025. [arXiv:2510.27141](https://arxiv.org/abs/2510.27141)

[6] J. Engels, B. Landrum et al., "Curator: Efficient Vector Search with Low-Selectivity Filters," *arXiv preprint*, 2026. [arXiv:2601.01291](https://arxiv.org/abs/2601.01291)

[7] (Anonymous), "Filtered Approximate Nearest Neighbor Search in Vector Databases," *arXiv preprint*, 2026. [arXiv:2602.11443](https://arxiv.org/abs/2602.11443)

[8] O. P. Elliott and J. Clark, "The Impacts of Data, Ordering, and Intrinsic Dimensionality on Recall in Hierarchical Navigable Small Worlds," *arXiv preprint*, 2024. [arXiv:2405.17813](https://arxiv.org/abs/2405.17813)

[9] M. Aumüller, E. Bernhardsson, and A. Faithfull, "ANN-Benchmarks: A Benchmarking Tool for Approximate Nearest Neighbor Algorithms," *Information Systems*, vol. 87, 2020. [ann-benchmarks.com](https://ann-benchmarks.com/)

[10] A. Aslam et al., "Clustered Hybrid Search," GitHub Repository. [Link](https://github.com/AdeelAslamUnimore/Clustered_Hybrid_Search)

[11] M. Ester, H.-P. Kriegel, J. Sander, and X. Xu, "A Density-Based Algorithm for Discovering Clusters in Large Spatial Databases with Noise," KDD, 1996.

[12] J. Mohoney et al., "High-Throughput Vector Similarity Search in Knowledge Graphs," *SIGMOD*, 2023.  

[13] J. Zhu et al., "An Experimental Evaluation of Hybrid Querying on Vectors," *PVLDB*, 2025.

[14] Y. Jin et al., "Curator: Efficient Indexing for Multi-Tenant Vector Databases," *arXiv preprint arXiv:2401.07119*, 2024.

[15] J. Sivic and A. Zisserman, "Video Google: A Text Retrieval Approach to Object Matching in Videos," *Proceedings of the IEEE International Conference on Computer Vision (ICCV)*, 2003.

[16] D. Sculley, "Web-Scale K-Means Clustering," in *Proceedings of the 19th International Conference on World Wide Web (WWW)*, Raleigh, NC, USA, 2010, pp. 1177–1178, doi: 10.1145/1772690.1772862.

[17] Y. Lin, K. Zhang, Z. He, Y. Jing, and X. S. Wang, "Survey of Filtered Approximate Nearest Neighbor Search over the Vector-Scalar Hybrid Data," arXiv preprint arXiv:2505.06501, 2025.

---

## Authors

Salem Alqahtani, Adeel Aslam, Khaled Mahmoud, Badr Asiri, Osama Al-Senani,
Faisal Azib, Abdullatif Hadi, Faisal Awad, and Omar Abdulaziz

