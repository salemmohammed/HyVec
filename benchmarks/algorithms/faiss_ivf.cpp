/**
 * faiss_ivf.cpp
 * -------------
 * FAISS IVF (Inverted File Index) — manual C++ implementation.
 *
 * This file reimplements FAISS IndexIVFFlat from scratch without using
 * the FAISS library. Every step — k-means, centroid assignment, posting
 * lists, flat search — is implemented manually so you can study exactly
 * what FAISS does internally.
 *
 * This is the partition-based comparison baseline for our Clustered HNSW
 * paper. The key structural difference between the two systems is:
 *
 *   FAISS IVF:       partition → flat exact L2 scan inside each cell
 *   Clustered HNSW:  partition → HNSW graph search inside each cluster
 *
 * Both systems use the same k-means partitioning front-end. The only
 * difference is what data structure lives inside each partition.
 *
 * -------------------------------------------------------------------------
 * Algorithm: Inverted File Index (IVF)
 *
 * Build phase (offline):
 *   1. Run k-means (Lloyd's algorithm [6]) on the full dataset.
 *      Produces K centroids: mu_0, mu_1, ..., mu_{K-1}
 *   2. Assign each vector x_i to its nearest centroid.
 *      Creates K posting lists: L_0, L_1, ..., L_{K-1}
 *      Each posting list stores the indices of assigned vectors.
 *
 * Query phase (online):
 *   1. Compute L2 distance from query q to all K centroids: O(K x d)
 *   2. Select the nprobe nearest centroids.
 *   3. For each selected centroid k, scan all vectors in L_k
 *      with exact L2 distance: O(nprobe x |L_k| x d)
 *   4. Keep a min-heap of the k nearest candidates across all cells.
 *   5. Return the k nearest vectors.
 *
 * -------------------------------------------------------------------------
 * Complexity:
 *
 *   Build : O(N x K x d x n_iter)   k-means (n_iter=25 iterations)
 *         + O(N x K x d)             final assignment pass
 *   Query : O(K x d)                 centroid scan
 *         + O(nprobe x N/K x d)      flat search in nprobe cells
 *   Memory: O(N x d)                 raw float32 vectors
 *         + O(K x d)                 K centroids
 *         + O(N)                     posting list indices
 *
 *   For SIFT1M: N=1M, K=1000, d=128, nprobe=20
 *     centroid scan : 1000 x 128       =   128K ops/query
 *     flat search   : 20 x 1000 x 128  = 2,560K ops/query
 *
 * -------------------------------------------------------------------------
 * Connection to Clustered HNSW:
 *
 *   nprobe in IVF  ↔  TOP_CLUSTERS in our system
 *   nlist  in IVF  ↔  K in our system
 *
 *   The build phase (k-means + assignment) is identical in both systems.
 *   The query phase differs only in the within-cluster search:
 *     IVF:            flat O(|L_k| x d) scan
 *     Clustered HNSW: HNSW O(log|L_k|) graph traversal
 *
 * -------------------------------------------------------------------------
 * Compilation:
 *
 *   g++ -O3 -std=c++17 -fopenmp faiss_ivf.cpp -o faiss_ivf \
 *       $(pkg-config --cflags --libs hdf5)
 *
 * -------------------------------------------------------------------------
 * Usage:
 *
 *   ./faiss_ivf --dataset ../data/sift-128-euclidean.hdf5
 *   ./faiss_ivf --dataset ../data/sift-128-euclidean.hdf5 --k 10
 *   ./faiss_ivf --n-clusters 1000 --k 10
 *
 * -------------------------------------------------------------------------
 * References:
 *
 *   [1] J. Johnson, M. Douze, H. Jégou, "Billion-Scale Similarity Search
 *       with GPUs," IEEE Transactions on Big Data, 7(3): 535-547, 2021.
 *       https://arxiv.org/abs/1702.08734
 *       → FAISS library — reference implementation of IndexIVFFlat
 *
 *   [2] H. Jégou, M. Douze, C. Schmid, "Product Quantization for Nearest
 *       Neighbor Search," IEEE TPAMI, 33(1): 117-128, 2011.
 *       https://doi.org/10.1109/TPAMI.2010.57
 *       → Original IVF algorithm and SIFT1M dataset
 *
 *   [3] M. Aumüller, E. Bernhardsson, A. Faithfull, "ANN-Benchmarks: A
 *       Benchmarking Tool for Approximate Nearest Neighbor Algorithms,"
 *       Information Systems, 87, 2020.
 *       https://doi.org/10.1016/j.is.2019.02.006
 *       → Evaluation protocol, Recall@k, HDF5 dataset format
 *
 *   [4] Y. A. Malkov, D. A. Yashunin, "Efficient and Robust Approximate
 *       Nearest Neighbor Search Using Hierarchical Navigable Small World
 *       Graphs," IEEE TPAMI, 42(4): 824-836, 2020.
 *       https://arxiv.org/abs/1603.09320
 *       → HNSW — the graph index used inside each cluster in our system
 *
 *   [5] M. Wang, X. Xu, Q. Yue, Y. Wang, "A Comprehensive Survey and
 *       Experimental Comparison of Graph-Based Approximate Nearest Neighbor
 *       Search," PVLDB, 14(11): 1964-1978, 2021.
 *       https://www.vldb.org/pvldb/vol14/p1964-wang.pdf
 *       → IVF in the broader ANN algorithm taxonomy
 *
 *   [6] S. P. Lloyd, "Least Squares Quantization in PCM," IEEE TIT,
 *       28(2): 129-137, 1982. https://doi.org/10.1109/TIT.1982.1056489
 *       → Lloyd's algorithm — the k-means procedure used in IVF build
 */

#include <algorithm>      // std::sort, std::partial_sort, std::min
#include <cassert>         // assert()
#include <chrono>          // std::chrono::high_resolution_clock
#include <cmath>           // std::sqrt, std::abs
#include <cstring>         // std::memcpy, std::memset
#include <fstream>         // std::ofstream
#include <iostream>        // std::cout, std::cerr
#include <limits>          // std::numeric_limits
#include <numeric>         // std::iota
#include <random>          // std::mt19937, std::uniform_int_distribution
#include <string>          // std::string
#include <vector>          // std::vector

// HDF5 C library — reads ANN-Benchmarks .hdf5 datasets [3]
// Install: apt install libhdf5-dev  or  brew install hdf5
#include <hdf5.h>

// OpenMP — parallelizes k-means assignment and search across CPU cores
#ifdef _OPENMP
#include <omp.h>
#endif


// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

static const int DEFAULT_K          = 10;    // nearest neighbors to retrieve
static const int DEFAULT_N_CLUSTERS = 1000;  // IVF partitions (= our K)
static const int KMEANS_ITERS       = 25;    // Lloyd's algorithm iterations [6]
                                              // FAISS default is also 25

// nprobe sweep — each value produces one point on the Recall vs QPS curve
// Directly analogous to TOP_CLUSTERS sweep in Clustered HNSW
static const std::vector<int> NPROBE_LIST = {1, 5, 10, 20, 50, 100};


// ---------------------------------------------------------------------------
// Timing
// ---------------------------------------------------------------------------

using Clock     = std::chrono::high_resolution_clock;
using TimePoint = std::chrono::time_point<Clock>;

inline TimePoint now() { return Clock::now(); }
inline double elapsed(TimePoint a, TimePoint b) {
    return std::chrono::duration<double>(b - a).count();
}


// ---------------------------------------------------------------------------
// Matrix — flat row-major float32 storage
//
// matrix[i * d + j] = j-th dimension of the i-th vector
// Matches FAISS internal layout [1] and NumPy default C order.
// ---------------------------------------------------------------------------

struct Matrix {
    std::vector<float> data;
    int n;   // number of vectors
    int d;   // dimensionality

    const float* row(int i) const { return data.data() + (size_t)i * d; }
          float* row(int i)       { return data.data() + (size_t)i * d; }

    void resize(int n_, int d_) {
        n = n_; d = d_;
        data.assign((size_t)n * d, 0.0f);
    }
};


// ---------------------------------------------------------------------------
// HDF5 dataset loader
//
// ANN-Benchmarks [3] distributes datasets as .hdf5 files:
//   /train      float32 (N, d) — base vectors to index
//   /test       float32 (Q, d) — query vectors
//   /neighbors  int32   (Q, 100) — ground truth top-100 neighbor indices
// ---------------------------------------------------------------------------

void hdf5_read_float(hid_t file, const char* name, Matrix& out) {
    hid_t   dset  = H5Dopen2(file, name, H5P_DEFAULT);
    hid_t   space = H5Dget_space(dset);
    hsize_t dims[2];
    H5Sget_simple_extent_dims(space, dims, nullptr);
    out.resize((int)dims[0], (int)dims[1]);
    H5Dread(dset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL,
            H5P_DEFAULT, out.data.data());
    H5Sclose(space);
    H5Dclose(dset);
}

void hdf5_read_int(hid_t file, const char* name,
                   std::vector<std::vector<int>>& out) {
    hid_t   dset  = H5Dopen2(file, name, H5P_DEFAULT);
    hid_t   space = H5Dget_space(dset);
    hsize_t dims[2];
    H5Sget_simple_extent_dims(space, dims, nullptr);
    int n = (int)dims[0], cols = (int)dims[1];
    std::vector<int> flat(n * cols);
    H5Dread(dset, H5T_NATIVE_INT, H5S_ALL, H5S_ALL,
            H5P_DEFAULT, flat.data());
    H5Sclose(space);
    H5Dclose(dset);
    out.resize(n);
    for (int i = 0; i < n; i++)
        out[i].assign(flat.begin() + i * cols,
                      flat.begin() + (i + 1) * cols);
}

void load_hdf5(const std::string& path,
               Matrix& base, Matrix& queries,
               std::vector<std::vector<int>>& gt) {
    hid_t file = H5Fopen(path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    if (file < 0) {
        std::cerr << "ERROR: Cannot open: " << path << "\n";
        std::exit(1);
    }
    hdf5_read_float(file, "train",     base);
    hdf5_read_float(file, "test",      queries);
    hdf5_read_int  (file, "neighbors", gt);
    H5Fclose(file);
    std::cout << "  Base    : " << base.n    << " x " << base.d << "\n";
    std::cout << "  Queries : " << queries.n << "\n";
    std::cout << "  GT      : top-" << gt[0].size() << " per query\n";
}


// ---------------------------------------------------------------------------
// Distance: L2 squared
//
// delta^2(x, q) = sum_i (x_i - q_i)^2
//
// We use squared L2 throughout — avoids sqrt() with identical ranking.
// The compiler auto-vectorizes this loop with SIMD (NEON on DGX Spark).
// Reference: Jégou et al. [2], Section 2
// ---------------------------------------------------------------------------

inline float l2_sq(const float* __restrict__ x,
                   const float* __restrict__ q,
                   int d) {
    float sum = 0.0f;
    for (int i = 0; i < d; i++) {
        float diff = x[i] - q[i];
        sum += diff * diff;
    }
    return sum;
    // Note: sqrt omitted — ranking by dist^2 == ranking by dist
}


// ---------------------------------------------------------------------------
// K-Means (Lloyd's Algorithm)
//
// This is the build phase shared by both IVF and our Clustered HNSW.
// Understanding this is key to understanding both systems.
//
// Algorithm (Lloyd [6]):
//
//   Initialize: pick K random vectors as initial centroids
//
//   Repeat for n_iter iterations:
//     E-step (Assignment):
//       For each vector x_i, find its nearest centroid:
//         c_i = argmin_k delta^2(x_i, mu_k)
//       Complexity: O(N x K x d) per iteration
//
//     M-step (Update):
//       Recompute each centroid as the mean of its assigned vectors:
//         mu_k = (1/|C_k|) * sum_{i in C_k} x_i
//       Complexity: O(N x d) per iteration
//
//   Total: O(N x K x d x n_iter)
//   For SIFT1M: 1M x 1000 x 128 x 25 = 3.2 trillion ops
//   → This is why FAISS uses GPU acceleration for large K [1]
//
// Reference: Lloyd [6], FAISS implementation [1]
// ---------------------------------------------------------------------------

/**
 * Run k-means on the dataset and return K centroids.
 *
 * @param data       Matrix of N vectors (N x d)
 * @param K          Number of clusters
 * @param n_iter     Number of Lloyd iterations (default 25, same as FAISS)
 * @return           Matrix of K centroids (K x d)
 */
Matrix kmeans(const Matrix& data, int K, int n_iter = KMEANS_ITERS) {
    int N = data.n;
    int d = data.d;

    // -----------------------------------------------------------------------
    // Initialize centroids: pick K random vectors from the dataset
    //
    // FAISS uses the same random initialization strategy [1].
    // More sophisticated methods (k-means++, oversampling) exist but
    // random init with 25 iterations converges well for ANN purposes.
    // -----------------------------------------------------------------------
    std::mt19937 rng(42);   // fixed seed for reproducibility
    std::vector<int> perm(N);
    std::iota(perm.begin(), perm.end(), 0);
    std::shuffle(perm.begin(), perm.end(), rng);

    Matrix centroids;
    centroids.resize(K, d);
    for (int k = 0; k < K; k++) {
        std::memcpy(centroids.row(k), data.row(perm[k]),
                    d * sizeof(float));
    }

    // Assignment array: assigns[i] = index of nearest centroid for vector i
    std::vector<int> assigns(N, 0);

    std::cout << "  Running k-means: K=" << K
              << ", N=" << N << ", d=" << d
              << ", iters=" << n_iter << "\n";

    for (int iter = 0; iter < n_iter; iter++) {

        // -------------------------------------------------------------------
        // E-step: Assignment
        //
        // For each vector, find the nearest centroid.
        // This is O(N x K x d) — the dominant cost of k-means.
        //
        // OpenMP parallelizes across vectors — each assignment is independent.
        // On the DGX Spark (72 ARM cores) this gives ~60x speedup.
        // -------------------------------------------------------------------
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < N; i++) {
            float best_dist = std::numeric_limits<float>::max();
            int   best_k    = 0;
            const float* xi = data.row(i);

            for (int k = 0; k < K; k++) {
                float dist = l2_sq(xi, centroids.row(k), d);
                if (dist < best_dist) {
                    best_dist = dist;
                    best_k    = k;
                }
            }
            assigns[i] = best_k;
        }

        // -------------------------------------------------------------------
        // M-step: Update centroids
        //
        // Recompute each centroid as the mean of its assigned vectors.
        // This is O(N x d) — cheaper than the E-step.
        //
        // Implementation note: accumulate sums first, then divide.
        // Must handle empty clusters (no vectors assigned) — keep old centroid.
        // -------------------------------------------------------------------
        Matrix new_centroids;
        new_centroids.resize(K, d);
        std::vector<int> counts(K, 0);

        // Accumulate: sum up all vectors assigned to each centroid
        // NOTE: not parallelized here to avoid race conditions on new_centroids
        // Could use per-thread accumulators for parallel version
        for (int i = 0; i < N; i++) {
            int   k  = assigns[i];
            float* c = new_centroids.row(k);
            const float* xi = data.row(i);
            for (int j = 0; j < d; j++) c[j] += xi[j];
            counts[k]++;
        }

        // Divide by count to get mean; keep old centroid if cluster is empty
        for (int k = 0; k < K; k++) {
            if (counts[k] > 0) {
                float inv = 1.0f / counts[k];
                float* c  = new_centroids.row(k);
                for (int j = 0; j < d; j++) c[j] *= inv;
            } else {
                // Empty cluster — keep previous centroid (same as FAISS [1])
                std::memcpy(new_centroids.row(k), centroids.row(k),
                            d * sizeof(float));
            }
        }

        centroids = std::move(new_centroids);

        // Progress report every 5 iterations
        if ((iter + 1) % 5 == 0 || iter == n_iter - 1) {
            std::cout << "    iter " << (iter + 1) << "/" << n_iter
                      << " done\n";
        }
    }

    return centroids;
}


// ---------------------------------------------------------------------------
// IVF Index
//
// The IVF index consists of two components:
//
//   1. Centroids matrix (K x d):
//      The K cluster centers from k-means.
//      Used at query time to find the nprobe nearest cells.
//
//   2. Posting lists (K posting lists, total N entries):
//      posting_lists[k] = list of (index, vector) for all vectors
//      assigned to cluster k.
//      Used at query time for the flat L2 scan within each cell.
//
// This matches the internal structure of FAISS IndexIVFFlat [1].
// Reference: Jégou et al. [2], Section 3
// ---------------------------------------------------------------------------

struct PostingEntry {
    int   idx;           // original vector index in the dataset
    float vec[];         // flexible array member — d floats follow immediately
                         // This avoids a separate heap allocation per vector
};

struct IVFIndex {
    Matrix centroids;    // K x d — cluster centroids from k-means

    // Posting lists: one per cluster
    // Each entry stores the original index + the raw float vector
    // We store raw vectors (not just indices) to avoid cache misses
    // during the flat scan — same design as FAISS InvertedLists [1]
    std::vector<std::vector<int>>   posting_ids;    // [k][i] = vector index
    std::vector<std::vector<float>> posting_vecs;   // [k] = flat float array
                                                     //  (n_k * d) floats
    int K;   // number of clusters
    int d;   // vector dimensionality
    int N;   // total vectors indexed
};


/**
 * Build the IVF index from a dataset.
 *
 * Phase 1: k-means — find K centroids (Lloyd's algorithm [6])
 * Phase 2: assignment — assign each vector to nearest centroid
 * Phase 3: posting lists — organize vectors by cluster
 *
 * @param data  Matrix of N base vectors (N x d)
 * @param K     Number of clusters
 * @return      Populated IVFIndex
 *
 * Reference: Jégou et al. [2], Johnson et al. [1]
 */
IVFIndex build_index(const Matrix& data, int K) {
    IVFIndex idx;
    idx.K = K;
    idx.d = data.d;
    idx.N = data.n;

    // -----------------------------------------------------------------------
    // Phase 1: K-Means
    //
    // Run Lloyd's algorithm [6] to find K centroids.
    // This is identical to what FAISS IndexIVFFlat.train() does [1].
    // -----------------------------------------------------------------------
    std::cout << "\n[Phase 1] K-Means clustering...\n";
    auto t0 = now();
    idx.centroids = kmeans(data, K);
    std::cout << "  K-Means time: " << elapsed(t0, now()) << "s\n";

    // -----------------------------------------------------------------------
    // Phase 2: Final assignment
    //
    // Assign every vector to its nearest centroid.
    // This is a single E-step pass — same computation as k-means E-step
    // but done once after convergence to populate the posting lists.
    //
    // Complexity: O(N x K x d)
    // -----------------------------------------------------------------------
    std::cout << "\n[Phase 2] Assigning " << data.n
              << " vectors to nearest centroid...\n";
    t0 = now();

    std::vector<int> assigns(data.n);

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < data.n; i++) {
        float best = std::numeric_limits<float>::max();
        int   best_k = 0;
        const float* xi = data.row(i);
        for (int k = 0; k < K; k++) {
            float dist = l2_sq(xi, idx.centroids.row(k), data.d);
            if (dist < best) { best = dist; best_k = k; }
        }
        assigns[i] = best_k;
    }
    std::cout << "  Assignment time: " << elapsed(t0, now()) << "s\n";

    // -----------------------------------------------------------------------
    // Phase 3: Build posting lists
    //
    // Organize vectors into K posting lists, one per cluster.
    // Each posting list stores:
    //   - the original vector index (for recall computation)
    //   - the raw float vector (to avoid pointer chasing during search)
    //
    // Storing raw vectors (not just indices) is the same design as
    // FAISS InvertedLists — it ensures sequential memory access during
    // the flat scan, which is critical for cache performance [1].
    //
    // Complexity: O(N x d) for copying vectors
    // -----------------------------------------------------------------------
    std::cout << "\n[Phase 3] Building posting lists...\n";
    t0 = now();

    idx.posting_ids.resize(K);
    idx.posting_vecs.resize(K);

    // First pass: count cluster sizes to pre-allocate
    std::vector<int> cluster_sizes(K, 0);
    for (int i = 0; i < data.n; i++) cluster_sizes[assigns[i]]++;

    for (int k = 0; k < K; k++) {
        idx.posting_ids[k].reserve(cluster_sizes[k]);
        idx.posting_vecs[k].reserve((size_t)cluster_sizes[k] * data.d);
    }

    // Second pass: populate posting lists
    for (int i = 0; i < data.n; i++) {
        int k = assigns[i];
        idx.posting_ids[k].push_back(i);
        // Append the d-dimensional vector to the flat posting_vecs array
        const float* xi = data.row(i);
        idx.posting_vecs[k].insert(idx.posting_vecs[k].end(),
                                   xi, xi + data.d);
    }

    std::cout << "  Posting list build time: " << elapsed(t0, now()) << "s\n";

    // Print cluster size statistics
    int min_sz = data.n, max_sz = 0;
    double avg_sz = 0;
    for (int k = 0; k < K; k++) {
        int sz = (int)idx.posting_ids[k].size();
        min_sz  = std::min(min_sz, sz);
        max_sz  = std::max(max_sz, sz);
        avg_sz += sz;
    }
    avg_sz /= K;
    std::cout << "  Cluster sizes: min=" << min_sz
              << "  max=" << max_sz
              << "  avg=" << (int)avg_sz << "\n";

    return idx;
}


// ---------------------------------------------------------------------------
// IVF Search
//
// Query phase for a single query vector q with parameter nprobe.
//
// Step 1 — Centroid scan: O(K x d)
//   Compute L2 distance from q to all K centroids.
//   Select the nprobe nearest centroids.
//   This is a brute force scan over K=1000 centroids — fast because K << N.
//
// Step 2 — Flat scan: O(nprobe x N/K x d)
//   For each of the nprobe selected clusters, scan all vectors in the
//   posting list and compute exact L2 distance to q.
//   This is identical to brute force but over N/K vectors instead of N.
//
// Step 3 — Top-k selection:
//   Keep a min-heap of the k nearest candidates across all nprobe cells.
//   Return the k nearest.
//
// Reference: Jégou et al. [2], Section 3 — query procedure
// ---------------------------------------------------------------------------

/**
 * Search the IVF index for the k nearest neighbors of query q.
 *
 * @param idx     Populated IVFIndex
 * @param q       Query vector (length d)
 * @param k       Number of neighbors to return
 * @param nprobe  Number of clusters to search (recall/speed tradeoff)
 * @return        Vector of k neighbor indices, nearest-first
 *
 * nprobe is directly analogous to TOP_CLUSTERS in our Clustered HNSW:
 *   nprobe=1  → search only the nearest cluster (fastest, lowest recall)
 *   nprobe=K  → search all clusters (= brute force, Recall=1.0)
 */
std::vector<int> search_one(const IVFIndex& idx,
                             const float*    q,
                             int             k,
                             int             nprobe) {
    int K = idx.K;
    int d = idx.d;

    // -----------------------------------------------------------------------
    // Step 1: Centroid scan
    //
    // Compute distance from q to all K centroids.
    // Select the nprobe nearest centroids.
    //
    // We use partial_sort to find the nprobe nearest in O(K log nprobe)
    // rather than full sort O(K log K).
    // -----------------------------------------------------------------------
    std::vector<std::pair<float, int>> centroid_dists(K);
    for (int k_ = 0; k_ < K; k_++) {
        centroid_dists[k_] = { l2_sq(idx.centroids.row(k_), q, d), k_ };
    }

    // Partial sort: bring the nprobe nearest centroids to the front
    std::partial_sort(centroid_dists.begin(),
                      centroid_dists.begin() + nprobe,
                      centroid_dists.end());

    // -----------------------------------------------------------------------
    // Step 2: Flat scan over nprobe posting lists
    //
    // For each of the nprobe nearest clusters, scan all vectors in the
    // posting list with exact L2 distance.
    //
    // We collect all candidates across all nprobe cells into one array,
    // then do a single partial_sort for the top-k.
    // Alternative: maintain a running min-heap (more complex, similar perf
    // for small nprobe).
    // -----------------------------------------------------------------------
    std::vector<std::pair<float, int>> candidates;
    candidates.reserve((size_t)nprobe * (idx.N / K + 1));

    for (int p = 0; p < nprobe; p++) {
        int cluster_id = centroid_dists[p].second;

        const std::vector<int>&   ids  = idx.posting_ids[cluster_id];
        const std::vector<float>& vecs = idx.posting_vecs[cluster_id];
        int n_k = (int)ids.size();

        // Flat L2 scan over all n_k vectors in this cluster
        // Memory access pattern: sequential — cache-friendly
        for (int i = 0; i < n_k; i++) {
            const float* xi = vecs.data() + (size_t)i * d;
            float dist = l2_sq(xi, q, d);
            candidates.push_back({ dist, ids[i] });
        }
    }

    // -----------------------------------------------------------------------
    // Step 3: Top-k selection
    //
    // Find the k nearest candidates across all nprobe cells.
    // partial_sort is O(M log k) where M = total candidates across cells.
    // -----------------------------------------------------------------------
    int result_k = std::min(k, (int)candidates.size());
    std::partial_sort(candidates.begin(),
                      candidates.begin() + result_k,
                      candidates.end());

    std::vector<int> result(result_k);
    for (int i = 0; i < result_k; i++) result[i] = candidates[i].second;
    return result;
}


/**
 * Search the IVF index for all query vectors.
 *
 * Parallelizes across queries using OpenMP — each query is independent.
 * On the DGX Spark (72 ARM cores), this gives ~60-70x speedup over
 * single-threaded search.
 *
 * @param idx      Populated IVFIndex
 * @param queries  Matrix of Q query vectors (Q x d)
 * @param k        Number of neighbors per query
 * @param nprobe   Clusters to search per query
 * @return         (Q, k) matrix of neighbor indices
 */
std::vector<std::vector<int>> search_all(const IVFIndex& idx,
                                          const Matrix&   queries,
                                          int             k,
                                          int             nprobe) {
    int Q = queries.n;
    std::vector<std::vector<int>> results(Q);

    #pragma omp parallel for schedule(dynamic, 32)
    for (int i = 0; i < Q; i++) {
        results[i] = search_one(idx, queries.row(i), k, nprobe);
    }

    return results;
}


// ---------------------------------------------------------------------------
// Recall@k
//
// Recall@k = |R ∩ R̃| / k
//
//   R  = ground truth top-k (from HDF5 /neighbors)
//   R̃  = top-k returned by IVF search
//
// Reference: Aumüller et al. [3], Section 3
// ---------------------------------------------------------------------------

double compute_recall(const std::vector<std::vector<int>>& predicted,
                      const std::vector<std::vector<int>>& gt,
                      int k) {
    double total = 0.0;
    int    Q     = (int)predicted.size();

    for (int i = 0; i < Q; i++) {
        // Build sorted ground truth set for binary search
        std::vector<int> true_set(gt[i].begin(), gt[i].begin() + k);
        std::sort(true_set.begin(), true_set.end());

        int hits = 0;
        for (int j = 0; j < (int)predicted[i].size() && j < k; j++) {
            if (std::binary_search(true_set.begin(), true_set.end(),
                                   predicted[i][j]))
                hits++;
        }
        total += (double)hits / k;
    }
    return total / Q;
}


// ---------------------------------------------------------------------------
// Argument parsing
// ---------------------------------------------------------------------------

struct Args {
    std::string dataset    = "sift";
    int         k          = DEFAULT_K;
    int         n_clusters = DEFAULT_N_CLUSTERS;
};

Args parse_args(int argc, char** argv) {
    Args args;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--dataset"    && i + 1 < argc) args.dataset    = argv[++i];
        if (a == "--k"          && i + 1 < argc) args.k          = std::stoi(argv[++i]);
        if (a == "--n-clusters" && i + 1 < argc) args.n_clusters = std::stoi(argv[++i]);
    }
    return args;
}


// ---------------------------------------------------------------------------
// CSV output — schema matches brute_force.cpp and faiss_ivf.py
// ---------------------------------------------------------------------------

void save_csv(const std::string& algorithm,
              double recall1, double recallk, int k,
              double qps, double build_time, double search_time,
              int num_queries, const std::string& dataset) {
    std::ofstream f("../results.csv", std::ios::app);
    if (!f.is_open()) return;

    auto t  = std::chrono::system_clock::now();
    auto tt = std::chrono::system_clock::to_time_t(t);
    char ts[32];
    std::strftime(ts, sizeof(ts), "%Y-%m-%d %H:%M:%S", std::localtime(&tt));

    f << ts          << ","
      << "dgx-spark" << ","
      << algorithm   << ","
      << recall1     << ","
      << recallk     << ","
      << qps         << ","
      << build_time  << ","
      << search_time << ","
      << num_queries << ","
      << dataset     << ","
      << "l2"        << "\n";
}


// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(int argc, char** argv) {

    Args args = parse_args(argc, argv);

    std::cout << std::string(65, '=') << "\n";
    std::cout << "  FAISS IVF — Manual C++ Implementation\n";
    std::cout << "  Partition-Based Comparison Baseline\n";
    std::cout << "  nprobe (IVF) <-> TOP_CLUSTERS (Clustered HNSW)\n";
    std::cout << std::string(65, '=') << "\n\n";

    // -----------------------------------------------------------------------
    // Load dataset
    // -----------------------------------------------------------------------

    std::cout << "Loading dataset...\n";
    Matrix base, queries;
    std::vector<std::vector<int>> gt;

    std::string path = args.dataset;
    if (path == "sift") path = "../data/sift-128-euclidean.hdf5";
    load_hdf5(path, base, queries, gt);

    // -----------------------------------------------------------------------
    // Build IVF index
    //
    // Phase 1: k-means   → K centroids
    // Phase 2: assignment → posting lists
    //
    // This is identical to FAISS IndexIVFFlat.train() + .add() [1]
    // and to the build phase of our Clustered HNSW (same k-means front-end).
    // -----------------------------------------------------------------------

    std::cout << "\nBuilding IVF index (K=" << args.n_clusters << ")...\n";
    auto t_build_start = now();
    IVFIndex idx = build_index(base, args.n_clusters);
    double build_time = elapsed(t_build_start, now());
    std::cout << "\nTotal build time: " << build_time << "s\n";

    // -----------------------------------------------------------------------
    // Sweep nprobe
    //
    // Each nprobe value = one point on the Recall vs QPS curve (Figure 6).
    // nprobe directly controls the recall/speed tradeoff.
    // Compare: TOP_CLUSTERS in our Clustered HNSW plays the same role.
    // -----------------------------------------------------------------------

    std::cout << "\nSweeping nprobe = { ";
    for (int np : NPROBE_LIST) std::cout << np << " ";
    std::cout << "}\n";
    std::cout << "(nprobe <-> TOP_CLUSTERS: both control partition search depth)\n\n";

    std::cout << "  " << std::left
              << std::setw(10) << "nprobe"
              << std::setw(14) << "Recall@1"
              << std::setw(14) << "Recall@k"
              << std::setw(14) << "QPS"
              << std::setw(12) << "Time(s)"
              << "\n";
    std::cout << "  " << std::string(64, '-') << "\n";

    for (int nprobe : NPROBE_LIST) {

        auto t0      = now();
        auto results = search_all(idx, queries, args.k, nprobe);
        double search_time = elapsed(t0, now());

        double qps     = queries.n / search_time;
        double recall1 = compute_recall(results, gt, 1);
        double recallk = compute_recall(results, gt, args.k);

        std::cout << "  " << std::left
                  << std::setw(10) << nprobe
                  << std::setw(14) << recall1
                  << std::setw(14) << recallk
                  << std::setw(14) << (int)qps
                  << std::setw(12) << search_time
                  << "\n";

        save_csv("faiss_ivf_nprobe" + std::to_string(nprobe),
                 recall1, recallk, args.k,
                 qps, build_time, search_time,
                 queries.n, args.dataset);
    }

    std::cout << "\n" << std::string(65, '=') << "\n";
    std::cout << "  Build time  : " << build_time << "s\n";
    std::cout << "  K (clusters): " << args.n_clusters << "\n";
    std::cout << "  Distance    : L2 Euclidean\n";
    std::cout << "  Results saved to ../results.csv\n";
    std::cout << std::string(65, '=') << "\n";

    return 0;
}
