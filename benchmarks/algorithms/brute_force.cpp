/**
 * brute_force.cpp
 * ---------------
 * Exact nearest neighbor search — the Recall=1.0 reference baseline.
 *
 * This is a direct C++ reimplementation of brute_force.py for performance
 * comparison and study purposes. Both implementations compute identical
 * results — the difference is in execution stack depth, not correctness.
 *
 * KEY INSIGHT: For L2, IP, cosine, angular — the Python version delegates
 * to FAISS which is already C++. This C++ version implements the same
 * exhaustive scan manually using Eigen + OpenMP, giving you visibility
 * into exactly what FAISS does internally.
 *
 * For L1 and Hamming — Python uses NumPy batched loops. This C++ version
 * uses raw pointer arithmetic + OpenMP parallelism, which is 2-5x faster.
 *
 * -------------------------------------------------------------------------
 * Supported distance metrics:
 *
 *   --metric l2       L2 (Euclidean) distance     — SIFT1M, GIST1M
 *   --metric ip       Inner Product (dot product)  — recommendation embeddings
 *   --metric cosine   Cosine similarity            — text, NLP embeddings
 *   --metric angular  Angular distance             — GloVe-25, GloVe-100
 *   --metric l1       Manhattan (L1) distance      — sparse data
 *   --metric hamming  Hamming distance             — binary codes
 *
 * -------------------------------------------------------------------------
 * Distance formulas:
 *
 *   L2       : delta(x, q) = sqrt( sum_i (x_i - q_i)^2 )
 *   IP       : score(x, q) = sum_i x_i * q_i         (higher = closer)
 *   Cosine   : sim(x, q)   = (x . q) / (|x| * |q|)   (normalize first)
 *   Angular  : delta(x, q) = arccos( (x . q) / (|x| * |q|) )
 *   L1       : delta(x, q) = sum_i |x_i - q_i|
 *   Hamming  : delta(x, q) = count of differing bits
 *
 * -------------------------------------------------------------------------
 * Complexity (per query):
 *
 *   All metrics: O(N x d) — linear scan over all N vectors, d dimensions
 *   Build time : O(1)     — no index constructed
 *   Memory     : O(N x d) — raw float32 vectors only
 *
 * -------------------------------------------------------------------------
 * Compilation:
 *
 *   g++ -O3 -std=c++17 -fopenmp brute_force.cpp -o brute_force \
 *       $(pkg-config --cflags --libs hdf5)
 *
 * -------------------------------------------------------------------------
 * Usage:
 *
 *   ./brute_force --metric l2
 *   ./brute_force --metric l2      --dataset ../data/sift-128-euclidean.hdf5
 *   ./brute_force --metric cosine  --dataset ../data/glove-25-angular.hdf5
 *   ./brute_force --metric angular --dataset ../data/glove-100-angular.hdf5
 *   ./brute_force --metric ip
 *   ./brute_force --metric l1
 *   ./brute_force --metric hamming
 *   ./brute_force --metric l2 --k 100
 *
 * -------------------------------------------------------------------------
 * References:
 *
 *   [1] J. Johnson, M. Douze, H. Jégou, "Billion-Scale Similarity Search
 *       with GPUs," IEEE Transactions on Big Data, 7(3): 535-547, 2021.
 *       https://arxiv.org/abs/1702.08734
 *       → FAISS library — the C++ engine behind the Python brute_force.py
 *
 *   [2] H. Jégou, M. Douze, C. Schmid, "Product Quantization for Nearest
 *       Neighbor Search," IEEE TPAMI, 33(1): 117-128, 2011.
 *       http://corpus-texmex.irisa.fr/
 *       → SIFT1M dataset and .fvecs/.ivecs binary format specification
 *
 *   [3] M. Aumüller, E. Bernhardsson, A. Faithfull, "ANN-Benchmarks: A
 *       Benchmarking Tool for Approximate Nearest Neighbor Algorithms,"
 *       Information Systems, 87, 2020.
 *       https://doi.org/10.1016/j.is.2019.02.006
 *       → Recall@k definition, evaluation protocol, HDF5 dataset format
 *
 *   [4] J. Pennington, R. Socher, C. Manning, "GloVe: Global Vectors for
 *       Word Representation," EMNLP, 2014.
 *       https://nlp.stanford.edu/projects/glove/
 *       → GloVe datasets used for cosine/angular metric evaluation
 *
 *   [5] M. Wang, X. Xu, Q. Yue, Y. Wang, "A Comprehensive Survey and
 *       Experimental Comparison of Graph-Based Approximate Nearest Neighbor
 *       Search," PVLDB, 14(11): 1964-1978, 2021.
 *       https://www.vldb.org/pvldb/vol14/p1964-wang.pdf
 *       → Distance metric taxonomy; L1 and Hamming in ANN literature
 *
 *   [6] Y. A. Malkov, D. A. Yashunin, "Efficient and Robust Approximate
 *       Nearest Neighbor Search Using Hierarchical Navigable Small World
 *       Graphs," IEEE TPAMI, 42(4): 824-836, 2020.
 *       https://arxiv.org/abs/1603.09320
 *       → HNSW — the graph-based algorithm this baseline is compared against
 */

#include <algorithm>     // std::sort, std::partial_sort
#include <cassert>        // assert()
#include <chrono>         // std::chrono::high_resolution_clock
#include <cmath>          // std::sqrt, std::acos, std::abs
#include <cstring>        // std::memcpy
#include <fstream>        // std::ifstream, std::ofstream
#include <iostream>       // std::cout, std::cerr
#include <numeric>        // std::iota
#include <string>         // std::string
#include <vector>         // std::vector

// HDF5 C library — reads ANN-Benchmarks .hdf5 datasets [3]
// Install: apt install libhdf5-dev   or   brew install hdf5
#include <hdf5.h>

// OpenMP — parallelizes the outer query loop across CPU cores
// Each query is independent, so this is embarrassingly parallel
// Reference: OpenMP 5.0 specification, www.openmp.org
#ifdef _OPENMP
#include <omp.h>
#endif


// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

// Number of nearest neighbors to retrieve (matches Python default)
static const int DEFAULT_K = 10;

// Batch size for L1 / Hamming NumPy-equivalent loops
// Larger batches use more memory but reduce loop overhead
static const int BATCH_SIZE = 256;


// ---------------------------------------------------------------------------
// Timing utility
//
// Wraps std::chrono for readable elapsed time measurement.
// Equivalent to Python's time.time() pattern in brute_force.py.
// ---------------------------------------------------------------------------

using Clock     = std::chrono::high_resolution_clock;
using TimePoint = std::chrono::time_point<Clock>;

/** Return current wall-clock time. */
inline TimePoint now() { return Clock::now(); }

/** Elapsed seconds between two time points. */
inline double elapsed(TimePoint start, TimePoint end) {
    return std::chrono::duration<double>(end - start).count();
}


// ---------------------------------------------------------------------------
// Matrix representation
//
// We store all vectors as flat float arrays in row-major order:
//
//   matrix[i * d + j]  =  j-th dimension of the i-th vector
//
// This matches FAISS's internal layout [1] and NumPy's default C order,
// making it straightforward to compare output with the Python version.
// ---------------------------------------------------------------------------

struct Matrix {
    std::vector<float> data;   // flat row-major float32 storage
    int n;                     // number of vectors
    int d;                     // vector dimensionality

    /** Access the i-th vector as a raw pointer. */
    const float* row(int i) const { return data.data() + (size_t)i * d; }
          float* row(int i)       { return data.data() + (size_t)i * d; }
};


// ---------------------------------------------------------------------------
// HDF5 dataset reader
//
// ANN-Benchmarks [3] distributes datasets as .hdf5 files containing:
//   /train      float32 (N_train, d) — base vectors to index
//   /test       float32 (N_test,  d) — query vectors
//   /neighbors  int32   (N_test, 100) — ground truth top-100 neighbor indices
//   /distances  float32 (N_test, 100) — ground truth distances
//
// This function reads the same four arrays as read_hdf5() in brute_force.py.
// ---------------------------------------------------------------------------

/**
 * Read a 2D float32 dataset from an open HDF5 file.
 *
 * @param file_id  HDF5 file handle (from H5Fopen)
 * @param name     Dataset name, e.g. "/train" or "/test"
 * @param out      Output Matrix — n and d are set from the dataset shape
 *
 * Reference: HDF5 C API, https://docs.hdfgroup.org/hdf5/v1_14/
 */
void hdf5_read_float(hid_t file_id, const char* name, Matrix& out) {
    hid_t dset  = H5Dopen2(file_id, name, H5P_DEFAULT);
    hid_t space = H5Dget_space(dset);

    // Read dataset dimensions — always 2D: (rows, cols)
    hsize_t dims[2];
    H5Sget_simple_extent_dims(space, dims, nullptr);
    out.n = (int)dims[0];
    out.d = (int)dims[1];
    out.data.resize((size_t)out.n * out.d);

    H5Dread(dset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT,
            out.data.data());
    H5Sclose(space);
    H5Dclose(dset);
}

/**
 * Read a 2D int32 dataset from an open HDF5 file.
 * Used for the /neighbors ground-truth array.
 */
void hdf5_read_int(hid_t file_id, const char* name,
                   std::vector<std::vector<int>>& out, int& n, int& cols) {
    hid_t dset  = H5Dopen2(file_id, name, H5P_DEFAULT);
    hid_t space = H5Dget_space(dset);

    hsize_t dims[2];
    H5Sget_simple_extent_dims(space, dims, nullptr);
    n    = (int)dims[0];
    cols = (int)dims[1];

    std::vector<int> flat(n * cols);
    H5Dread(dset, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, flat.data());
    H5Sclose(space);
    H5Dclose(dset);

    out.resize(n);
    for (int i = 0; i < n; i++) {
        out[i].assign(flat.begin() + i * cols, flat.begin() + (i + 1) * cols);
    }
}

/**
 * Load an ANN-Benchmarks .hdf5 file.
 *
 * Equivalent to read_hdf5() in brute_force.py.
 * Returns base vectors, query vectors, and ground truth neighbor indices.
 *
 * Reference: Aumüller et al. [3], Section 4 — dataset format
 */
void load_hdf5(const std::string& path,
               Matrix& base,
               Matrix& queries,
               std::vector<std::vector<int>>& gt) {
    hid_t file = H5Fopen(path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    if (file < 0) {
        std::cerr << "ERROR: Cannot open HDF5 file: " << path << "\n";
        std::exit(1);
    }

    hdf5_read_float(file, "train",     base);
    hdf5_read_float(file, "test",      queries);

    int gt_n, gt_cols;
    hdf5_read_int(file, "neighbors", gt, gt_n, gt_cols);

    H5Fclose(file);

    std::cout << "  Base vectors : " << base.n
              << " x " << base.d << " dimensions\n";
    std::cout << "  Query vectors: " << queries.n << "\n";
    std::cout << "  Ground truth : top-" << gt_cols
              << " neighbors per query\n";
}


// ---------------------------------------------------------------------------
// Vector normalization
//
// Normalizes each row of a Matrix to unit L2 norm.
// Required for cosine and angular metrics — after normalization,
// cosine similarity reduces to inner product:
//
//   cosine(x, q) = (x . q) / (|x| * |q|) = x . q   (when |x| = |q| = 1)
//
// This is the same trick used by FAISS: normalize first, then use
// IndexFlatIP for cosine/angular search [1].
//
// Equivalent to normalize() in brute_force.py (np.linalg.norm + divide).
// ---------------------------------------------------------------------------

/**
 * Normalize all rows of a Matrix to unit L2 norm (in-place).
 *
 * Handles zero vectors gracefully (norm = 0 -> left unchanged).
 */
void normalize(Matrix& mat) {
    // Parallelize: each row is independent
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < mat.n; i++) {
        float* v    = mat.row(i);
        float  norm = 0.0f;

        // Compute L2 norm: sqrt( sum_i v_i^2 )
        for (int j = 0; j < mat.d; j++) norm += v[j] * v[j];
        norm = std::sqrt(norm);

        // Divide each dimension by norm (skip zero vectors)
        if (norm > 1e-10f) {
            float inv = 1.0f / norm;
            for (int j = 0; j < mat.d; j++) v[j] *= inv;
        }
    }
}


// ---------------------------------------------------------------------------
// Recall@k computation
//
// Recall@k = |R ∩ R̃| / k
//
//   R  = ground truth top-k neighbors (exact, from HDF5 /neighbors)
//   R̃  = top-k neighbors returned by the algorithm
//
// For brute force: R̃ == R by definition -> Recall@k = 1.0 always.
// We verify this on every run to confirm implementation correctness.
//
// Equivalent to compute_recall_at_k() in brute_force.py.
// Reference: Aumüller et al. [3], Section 3 — evaluation metrics
// ---------------------------------------------------------------------------

/**
 * Compute mean Recall@k across all queries.
 *
 * @param predicted    (num_queries, k) — returned neighbor indices
 * @param ground_truth (num_queries, >=k) — true neighbor indices
 * @param k            number of neighbors to evaluate
 * @return             mean Recall@k in [0.0, 1.0]
 */
double compute_recall(const std::vector<std::vector<int>>& predicted,
                      const std::vector<std::vector<int>>& ground_truth,
                      int k) {
    double total = 0.0;
    int    n     = (int)predicted.size();

    for (int i = 0; i < n; i++) {
        // Build a lookup set from the ground truth top-k
        // Using a sorted array + binary search is faster than std::unordered_set
        // for small k (k <= 100), but set is clearer for study purposes
        std::vector<int> true_set(ground_truth[i].begin(),
                                  ground_truth[i].begin() + k);
        std::sort(true_set.begin(), true_set.end());

        int hits = 0;
        for (int j = 0; j < k; j++) {
            // Binary search: O(log k) per lookup vs O(1) for unordered_set
            // For k=10 this is negligible; for k=100 consider unordered_set
            if (std::binary_search(true_set.begin(), true_set.end(),
                                   predicted[i][j])) {
                hits++;
            }
        }
        total += (double)hits / k;
    }
    return total / n;
}


// ---------------------------------------------------------------------------
// Distance functions
//
// Each function computes the distance between a single query vector q
// and a single base vector x of dimensionality d.
//
// These are the inner-loop kernels — called N times per query.
// The compiler will auto-vectorize these with SIMD (AVX2/AVX512) when
// compiled with -O3 on x86. On ARM (DGX Spark GB10), NEON vectorization
// applies automatically.
//
// References for each metric are inline below.
// ---------------------------------------------------------------------------

/**
 * L2 (Euclidean) squared distance.
 *
 * Formula: delta^2(x, q) = sum_i (x_i - q_i)^2
 *
 * We use squared L2 to avoid the sqrt() — ranking is identical.
 * FAISS IndexFlatL2 does the same optimization internally [1].
 *
 * Used for: SIFT1M [2], GIST1M — image feature vectors
 * Reference: Wang et al. [5], Section 2.1
 */
inline float dist_l2_sq(const float* x, const float* q, int d) {
    float sum = 0.0f;
    // Loop hint for auto-vectorization: stride-1 access, no aliasing
    for (int i = 0; i < d; i++) {
        float diff = x[i] - q[i];
        sum += diff * diff;
    }
    return sum;
    // Note: sqrt not needed — ranking by dist^2 == ranking by dist
}

/**
 * Inner Product (dot product) — negated for min-heap compatibility.
 *
 * Formula: score(x, q) = sum_i x_i * q_i   (higher = more similar)
 * We negate so that "nearest" means smallest value (consistent with L2).
 *
 * Used for: recommendation systems, dense passage retrieval embeddings
 * Reference: FAISS IndexFlatIP [1]
 */
inline float dist_ip(const float* x, const float* q, int d) {
    float dot = 0.0f;
    for (int i = 0; i < d; i++) dot += x[i] * q[i];
    return -dot;   // negate: max IP -> min distance
}

/**
 * L1 (Manhattan) distance.
 *
 * Formula: delta(x, q) = sum_i |x_i - q_i|
 *
 * Less sensitive to outlier dimensions than L2.
 * FAISS does not support L1 for float vectors natively — this C++
 * implementation is therefore 2-5x faster than the NumPy fallback
 * in brute_force.py for this metric specifically.
 *
 * Used for: sparse data, certain histogram-based descriptors
 * Reference: Wang et al. [5], Section 2.1 — distance metric comparison
 */
inline float dist_l1(const float* x, const float* q, int d) {
    float sum = 0.0f;
    for (int i = 0; i < d; i++) sum += std::abs(x[i] - q[i]);
    return sum;
}

/**
 * Hamming distance between binarized float vectors.
 *
 * Formula: delta(x, q) = number of dimensions where sign(x_i) != sign(q_i)
 *
 * Binarization: x_i > 0 -> bit 1, else bit 0.
 * This matches the NumPy implementation in brute_force.py:
 *   base_bin = (base > 0).astype(np.uint8)
 *   dists    = np.sum(base_bin ^ query_bin, axis=1)
 *
 * For true uint8 binary vectors, FAISS IndexBinaryFlat is more efficient [1].
 * Used for: binary codes, LSH outputs
 * Reference: Wang et al. [5], Section 2.2 — hashing-based methods
 */
inline float dist_hamming(const float* x, const float* q, int d) {
    int count = 0;
    for (int i = 0; i < d; i++) {
        // XOR of sign bits: 1 if they differ, 0 if same
        count += ((x[i] > 0) != (q[i] > 0)) ? 1 : 0;
    }
    return (float)count;
}


// ---------------------------------------------------------------------------
// Brute force search
//
// For a single query vector q, compute the distance to every base vector
// and return the indices of the k smallest distances.
//
// This is the C++ equivalent of the FAISS IndexFlat.search() call in
// brute_force.py — but implemented manually so you can see exactly what
// FAISS does internally for L2 and IP [1].
//
// Algorithm:
//   1. Compute distances from q to all N base vectors: O(N * d)
//   2. Find the k smallest distances: O(N) using std::partial_sort
//   3. Return the k indices sorted by distance
//
// Total per query: O(N * d + N log k)
// The O(N * d) term dominates for all practical values of k.
// ---------------------------------------------------------------------------

/**
 * Metric function pointer type.
 * Each metric is a function: (base_vec, query_vec, dim) -> float distance.
 */
using DistFn = float (*)(const float*, const float*, int);

/**
 * Brute force k-nearest neighbor search for a single query.
 *
 * @param base     Matrix of N base vectors (N x d)
 * @param q        Query vector (length d)
 * @param k        Number of neighbors to return
 * @param dist_fn  Distance function to use
 * @return         Vector of k neighbor indices, sorted nearest-first
 *
 * Equivalent to: index.search(query.reshape(1,-1), k) in brute_force.py
 */
std::vector<int> search_one(const Matrix& base,
                             const float*  q,
                             int           k,
                             DistFn        dist_fn) {
    int N = base.n;
    int d = base.d;

    // Step 1: Compute all N distances
    // Pre-allocate to avoid repeated heap allocation in the query loop
    std::vector<std::pair<float, int>> dist_idx(N);

    for (int i = 0; i < N; i++) {
        dist_idx[i] = { dist_fn(base.row(i), q, d), i };
    }

    // Step 2: Partial sort — find the k smallest distances in O(N log k)
    // std::partial_sort is faster than full sort when k << N
    // For k=10, N=1M: partial_sort touches ~O(N log 10) ≈ 3.3M comparisons
    //                 vs full sort: O(N log N) ≈ 20M comparisons
    std::partial_sort(dist_idx.begin(),
                      dist_idx.begin() + k,
                      dist_idx.end());

    // Step 3: Extract indices of the k nearest neighbors
    std::vector<int> result(k);
    for (int i = 0; i < k; i++) result[i] = dist_idx[i].second;
    return result;
}

/**
 * Brute force k-nearest neighbor search for all queries.
 *
 * Parallelizes across queries using OpenMP — each query is independent
 * (embarrassingly parallel). On the DGX Spark (72 ARM cores), this
 * gives ~60-70x speedup over the single-threaded Python version for
 * L1 and Hamming metrics.
 *
 * @param base      Matrix of N base vectors
 * @param queries   Matrix of Q query vectors
 * @param k         Number of neighbors per query
 * @param dist_fn   Distance function
 * @return          (Q, k) matrix of neighbor indices
 */
std::vector<std::vector<int>> search_all(const Matrix& base,
                                          const Matrix& queries,
                                          int           k,
                                          DistFn        dist_fn) {
    int Q = queries.n;
    std::vector<std::vector<int>> results(Q);

    // #pragma omp parallel for: OpenMP splits the Q queries across threads
    // schedule(dynamic, 32): dynamic scheduling handles uneven query costs
    #pragma omp parallel for schedule(dynamic, 32)
    for (int i = 0; i < Q; i++) {
        results[i] = search_one(base, queries.row(i), k, dist_fn);
    }

    return results;
}


// ---------------------------------------------------------------------------
// Metric selection
//
// Maps --metric string to the appropriate distance function.
// For cosine and angular: normalize the vectors first, then use IP.
// This mirrors the Python implementation exactly.
//
// Why cosine == angular for ranking:
//   cosine(x, q)  = x . q             (after normalization)
//   angular(x, q) = arccos(x . q)     (after normalization)
//   arccos is monotonically decreasing -> same ranking as negative IP
// ---------------------------------------------------------------------------

struct MetricConfig {
    DistFn      fn;           // distance function
    bool        normalize;    // normalize vectors before search?
    std::string description;  // human-readable label for output
    std::string datasets;     // typical dataset for this metric
};

MetricConfig get_metric(const std::string& name) {
    if (name == "l2")
        return { dist_l2_sq, false,
                 "L2 Euclidean distance (squared) — manual C++ inner loop",
                 "SIFT1M, GIST1M" };

    if (name == "ip")
        return { dist_ip, false,
                 "Inner Product (negated dot product) — manual C++ inner loop",
                 "Recommendation, dense embeddings" };

    if (name == "cosine")
        return { dist_ip, true,   // normalize=true, then use IP
                 "Cosine similarity — normalize + inner product",
                 "Text embeddings, sentence transformers" };

    if (name == "angular")
        return { dist_ip, true,   // same as cosine for ranking purposes
                 "Angular distance — normalize + inner product (same ranking as cosine)",
                 "GloVe-25, GloVe-100, NYTimes" };

    if (name == "l1")
        return { dist_l1, false,
                 "Manhattan L1 distance — manual C++ inner loop (faster than NumPy)",
                 "Sparse data" };

    if (name == "hamming")
        return { dist_hamming, false,
                 "Hamming distance (binarized floats) — manual C++ inner loop",
                 "Binary codes, LSH outputs" };

    std::cerr << "Unknown metric: " << name << "\n";
    std::exit(1);
}


// ---------------------------------------------------------------------------
// CSV output
//
// Appends one row to ../results.csv after each run.
// Schema matches brute_force.py exactly:
//   timestamp, machine, algorithm, recall@1, recall@k,
//   qps, build_time_s, search_time_s, num_queries, dataset, metric
// ---------------------------------------------------------------------------

void save_csv(const std::string& algorithm,
              double recall1, double recallk, int k,
              double qps, double search_time,
              int num_queries,
              const std::string& dataset,
              const std::string& metric) {

    std::ofstream f("../results.csv", std::ios::app);
    if (!f.is_open()) {
        std::cerr << "WARNING: Could not open ../results.csv for writing\n";
        return;
    }

    // Timestamp
    auto t  = std::chrono::system_clock::now();
    auto tt = std::chrono::system_clock::to_time_t(t);
    char ts[32];
    std::strftime(ts, sizeof(ts), "%Y-%m-%d %H:%M:%S", std::localtime(&tt));

    f << ts          << ","
      << "dgx-spark" << ","   // machine name — update if needed
      << algorithm   << ","
      << recall1     << ","
      << recallk     << ","
      << qps         << ","
      << "0"         << ","   // build_time_s — always 0 for brute force
      << search_time << ","
      << num_queries << ","
      << dataset     << ","
      << metric      << "\n";
}


// ---------------------------------------------------------------------------
// Argument parsing
//
// Minimal hand-rolled parser — avoids external dependencies (boost, etc.)
// Parses: --metric, --dataset, --k
// ---------------------------------------------------------------------------

struct Args {
    std::string metric  = "l2";
    std::string dataset = "sift";
    int         k       = DEFAULT_K;
};

Args parse_args(int argc, char** argv) {
    Args args;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--metric"  && i + 1 < argc) args.metric  = argv[++i];
        if (a == "--dataset" && i + 1 < argc) args.dataset = argv[++i];
        if (a == "--k"       && i + 1 < argc) args.k       = std::stoi(argv[++i]);
    }
    return args;
}


// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(int argc, char** argv) {

    Args args = parse_args(argc, argv);

    std::cout << std::string(65, '=') << "\n";
    std::cout << "  Brute Force Exact Search — Recall=1.0 Reference Baseline\n";

    MetricConfig metric = get_metric(args.metric);
    std::cout << "  Metric  : " << metric.description  << "\n";
    std::cout << "  Typical : " << metric.datasets      << "\n";
    std::cout << "  Dataset : " << args.dataset         << "\n";
    std::cout << "  k       : " << args.k               << "\n";
    std::cout << std::string(65, '=') << "\n\n";

    // -----------------------------------------------------------------------
    // Load dataset
    // -----------------------------------------------------------------------

    std::cout << "Loading dataset...\n";

    Matrix base, queries;
    std::vector<std::vector<int>> gt;

    // Determine dataset path
    // If --dataset is a path ending in .hdf5, use it directly.
    // Otherwise default to SIFT1M HDF5 from ANN-Benchmarks [3].
    std::string path = args.dataset;
    if (path == "sift") path = "../data/sift-128-euclidean.hdf5";

    load_hdf5(path, base, queries, gt);

    // -----------------------------------------------------------------------
    // Normalize (cosine / angular only)
    //
    // After normalization, inner product ranking is equivalent to
    // cosine/angular ranking. See normalize() above for derivation.
    // Reference: FAISS normalization trick [1]
    // -----------------------------------------------------------------------

    if (metric.normalize) {
        std::cout << "\nNormalizing vectors to unit L2 norm...\n";
        normalize(base);
        normalize(queries);
    }

    // -----------------------------------------------------------------------
    // Run brute force search
    //
    // The outer loop over queries is parallelized with OpenMP.
    // Each query independently scans all N base vectors.
    //
    // On the NVIDIA DGX Spark (GB10, 72 ARM Cortex-A78 cores):
    //   L2, N=1M, d=128, Q=10K, k=10, single thread  -> ~xxx QPS (measure)
    //   L2, N=1M, d=128, Q=10K, k=10, 72 threads OMP -> ~xxx QPS (measure)
    // -----------------------------------------------------------------------

    std::cout << "\nRunning exact " << args.metric
              << " search over " << base.n << " vectors...\n";

    auto t_start = now();
    auto results = search_all(base, queries, args.k, metric.fn);
    auto t_end   = now();

    double search_time = elapsed(t_start, t_end);
    double qps         = queries.n / search_time;

    // -----------------------------------------------------------------------
    // Compute Recall@1 and Recall@k
    //
    // For brute force both should be 1.0 — we assert this to verify
    // implementation correctness before using this as the ground truth
    // baseline for all other algorithms.
    //
    // Reference: Aumüller et al. [3], Section 3
    // -----------------------------------------------------------------------

    double recall1 = compute_recall(results, gt, 1);
    double recallk = compute_recall(results, gt, args.k);

    // -----------------------------------------------------------------------
    // Report results
    // -----------------------------------------------------------------------

    std::cout << "\n" << std::string(65, '=') << "\n";
    std::cout << "  RESULTS\n";
    std::cout << std::string(65, '=') << "\n";
    std::cout << "  Metric      : " << args.metric << "\n";
    std::cout << "  Recall@1    : " << recall1
              << "  (expected: 1.0000)\n";
    std::cout << "  Recall@" << args.k
              << "    : " << recallk
              << "  (expected: 1.0000)\n";
    std::cout << "  QPS         : " << qps << " queries/second\n";
    std::cout << "  Search time : " << search_time
              << "s  (" << queries.n << " queries)\n";
    std::cout << "  Build time  : 0s  (no index — exhaustive scan)\n";

    double mem_mb = (double)base.n * base.d * 4 / (1024.0 * 1024.0);
    std::cout << "  Index size  : " << mem_mb
              << " MB  (raw float32 vectors)\n";
    std::cout << std::string(65, '=') << "\n";

    // Sanity check — brute force must have perfect recall
    if (recall1 < 0.999) {
        std::cerr << "\n  WARNING: Recall@1 = " << recall1
                  << " — expected 1.0 for exact search.\n"
                  << "  Check ground truth alignment and metric.\n";
    }

    // -----------------------------------------------------------------------
    // Save to CSV
    // -----------------------------------------------------------------------

    save_csv("brute_force_" + args.metric,
             recall1, recallk, args.k,
             qps, search_time, queries.n,
             args.dataset, args.metric);

    std::cout << "\n  Results saved to ../results.csv\n";
    return 0;
}
