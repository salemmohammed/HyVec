/**
 * brute_force.cpp
 * ---------------
 * Exact nearest neighbor search using FAISS IndexFlatL2.
 *
 * This file implements the official brute-force baseline for the benchmark.
 * It uses Euclidean (L2) distance only and performs exact search over all
 * database vectors. The returned neighbors are used as the ground-truth
 * reference for evaluating ANN indexes such as HNSW, FAISS-IVF, ScaNN,
 * DiskANN, and the proposed Hybrid Attribute-Spatial HNSW index.
 *
 * Why FAISS?
 *   - FAISS IndexFlatL2 is exact brute-force search.
 *   - It is optimized, well-tested, and commonly used in ANN benchmarks.
 *   - It avoids maintaining a separate manual OpenMP implementation.
 *
 * Supported metric:
 *   L2 / Euclidean distance only.
 *
 * Complexity:
 *   Build time : O(N * d) to add vectors to the FAISS flat index
 *   Search     : O(Q * N * d) exhaustive scan
 *   Memory     : O(N * d) raw float32 vectors
 *
 * Compilation:
 *
 *   g++ -O3 -std=c++17 brute_force.cpp -o brute_force \
 *       $(pkg-config --cflags --libs hdf5) \
 *       -lfaiss
 *
 * Usage:
 *
 *   ./brute_force
 *   ./brute_force --dataset ../data/sift-128-euclidean.hdf5
 *   ./brute_force --dataset ../data/sift-128-euclidean.hdf5 --k 10
 *
 * Output:
 *   Appends one row to ../results.csv
 */

#include <algorithm>
#include <chrono>
#include <ctime>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <hdf5.h>
#include <faiss/IndexFlat.h>

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

static const int DEFAULT_K = 10;

// ---------------------------------------------------------------------------
// Timing utility
// ---------------------------------------------------------------------------

using Clock = std::chrono::high_resolution_clock;
using TimePoint = std::chrono::time_point<Clock>;

inline TimePoint now() {
    return Clock::now();
}

inline double elapsed(TimePoint start, TimePoint end) {
    return std::chrono::duration<double>(end - start).count();
}

// ---------------------------------------------------------------------------
// Matrix representation
// ---------------------------------------------------------------------------

struct Matrix {
    std::vector<float> data;
    int n = 0;
    int d = 0;

    const float* row(int i) const {
        return data.data() + static_cast<size_t>(i) * d;
    }

    float* row(int i) {
        return data.data() + static_cast<size_t>(i) * d;
    }
};

// ---------------------------------------------------------------------------
// HDF5 helpers
// ---------------------------------------------------------------------------

void hdf5_read_float(hid_t file_id, const char* name, Matrix& out) {
    hid_t dset = H5Dopen2(file_id, name, H5P_DEFAULT);
    if (dset < 0) {
        std::cerr << "ERROR: Cannot open HDF5 dataset: " << name << "\n";
        std::exit(1);
    }

    hid_t space = H5Dget_space(dset);

    hsize_t dims[2];
    H5Sget_simple_extent_dims(space, dims, nullptr);

    out.n = static_cast<int>(dims[0]);
    out.d = static_cast<int>(dims[1]);
    out.data.resize(static_cast<size_t>(out.n) * out.d);

    H5Dread(
        dset,
        H5T_NATIVE_FLOAT,
        H5S_ALL,
        H5S_ALL,
        H5P_DEFAULT,
        out.data.data()
    );

    H5Sclose(space);
    H5Dclose(dset);
}

void hdf5_read_int(
    hid_t file_id,
    const char* name,
    std::vector<std::vector<int>>& out,
    int& n,
    int& cols
) {
    hid_t dset = H5Dopen2(file_id, name, H5P_DEFAULT);
    if (dset < 0) {
        std::cerr << "ERROR: Cannot open HDF5 dataset: " << name << "\n";
        std::exit(1);
    }

    hid_t space = H5Dget_space(dset);

    hsize_t dims[2];
    H5Sget_simple_extent_dims(space, dims, nullptr);

    n = static_cast<int>(dims[0]);
    cols = static_cast<int>(dims[1]);

    std::vector<int> flat(static_cast<size_t>(n) * cols);

    H5Dread(
        dset,
        H5T_NATIVE_INT,
        H5S_ALL,
        H5S_ALL,
        H5P_DEFAULT,
        flat.data()
    );

    H5Sclose(space);
    H5Dclose(dset);

    out.resize(n);
    for (int i = 0; i < n; i++) {
        out[i].assign(
            flat.begin() + static_cast<size_t>(i) * cols,
            flat.begin() + static_cast<size_t>(i + 1) * cols
        );
    }
}

void load_hdf5(
    const std::string& path,
    Matrix& base,
    Matrix& queries,
    std::vector<std::vector<int>>& ground_truth
) {
    hid_t file = H5Fopen(path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    if (file < 0) {
        std::cerr << "ERROR: Cannot open HDF5 file: " << path << "\n";
        std::exit(1);
    }

    hdf5_read_float(file, "train", base);
    hdf5_read_float(file, "test", queries);

    int gt_n = 0;
    int gt_cols = 0;
    hdf5_read_int(file, "neighbors", ground_truth, gt_n, gt_cols);

    H5Fclose(file);

    if (base.d != queries.d) {
        std::cerr << "ERROR: Base and query dimensions do not match.\n";
        std::exit(1);
    }

    std::cout << "  Base vectors : " << base.n << " x " << base.d << "\n";
    std::cout << "  Query vectors: " << queries.n << " x " << queries.d << "\n";
    std::cout << "  Ground truth : top-" << gt_cols << " neighbors per query\n";
}

// ---------------------------------------------------------------------------
// Recall@k
// ---------------------------------------------------------------------------

double compute_recall(
    const std::vector<std::vector<int>>& predicted,
    const std::vector<std::vector<int>>& ground_truth,
    int k
) {
    double total = 0.0;
    int num_queries = static_cast<int>(predicted.size());

    for (int i = 0; i < num_queries; i++) {
        std::vector<int> truth(
            ground_truth[i].begin(),
            ground_truth[i].begin() + k
        );

        std::sort(truth.begin(), truth.end());

        int hits = 0;
        for (int j = 0; j < k; j++) {
            if (std::binary_search(truth.begin(), truth.end(), predicted[i][j])) {
                hits++;
            }
        }

        total += static_cast<double>(hits) / k;
    }

    return total / num_queries;
}

// ---------------------------------------------------------------------------
// CSV output
// ---------------------------------------------------------------------------

void save_csv(
    const std::string& algorithm,
    const std::string& dataset,
    int k,
    double recall1,
    double recallk,
    double qps,
    double avg_latency_ms,
    double build_time_s,
    double search_time_s,
    double index_size_mb,
    int num_base,
    int num_queries,
    int dimension
) {
    std::ofstream f("../results.csv", std::ios::app);
    if (!f.is_open()) {
        std::cerr << "WARNING: Could not open ../results.csv for writing\n";
        return;
    }

    auto t = std::chrono::system_clock::now();
    auto tt = std::chrono::system_clock::to_time_t(t);

    char timestamp[32];
    std::strftime(
        timestamp,
        sizeof(timestamp),
        "%Y-%m-%d %H:%M:%S",
        std::localtime(&tt)
    );

    f << timestamp << ","
      << "dgx-spark" << ","
      << algorithm << ","
      << dataset << ","
      << "l2" << ","
      << k << ","
      << recall1 << ","
      << recallk << ","
      << qps << ","
      << avg_latency_ms << ","
      << build_time_s << ","
      << search_time_s << ","
      << index_size_mb << ","
      << num_base << ","
      << num_queries << ","
      << dimension << "\n";
}

// ---------------------------------------------------------------------------
// Arguments
// ---------------------------------------------------------------------------

struct Args {
    std::string dataset = "../data/sift-128-euclidean.hdf5";
    int k = DEFAULT_K;
};

Args parse_args(int argc, char** argv) {
    Args args;

    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];

        if (a == "--dataset" && i + 1 < argc) {
            args.dataset = argv[++i];
        } else if (a == "--k" && i + 1 < argc) {
            args.k = std::stoi(argv[++i]);
        } else if (a == "--metric") {
            std::cerr << "ERROR: This benchmark supports only L2 / Euclidean distance.\n";
            std::cerr << "Remove --metric. FAISS IndexFlatL2 is used by default.\n";
            std::exit(1);
        } else {
            std::cerr << "ERROR: Unknown argument: " << a << "\n";
            std::exit(1);
        }
    }

    return args;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(int argc, char** argv) {
    Args args = parse_args(argc, argv);

    std::cout << std::string(72, '=') << "\n";
    std::cout << "  FAISS Brute Force Exact Search — IndexFlatL2\n";
    std::cout << "  Metric: L2 / Euclidean\n";
    std::cout << "  Dataset: " << args.dataset << "\n";
    std::cout << "  k: " << args.k << "\n";
    std::cout << std::string(72, '=') << "\n\n";

    Matrix base;
    Matrix queries;
    std::vector<std::vector<int>> ground_truth;

    std::cout << "Loading dataset...\n";
    load_hdf5(args.dataset, base, queries, ground_truth);

    if (args.k <= 0) {
        std::cerr << "ERROR: k must be positive.\n";
        return 1;
    }

    if (!ground_truth.empty() && args.k > static_cast<int>(ground_truth[0].size())) {
        std::cerr << "ERROR: k is larger than available ground-truth neighbors.\n";
        return 1;
    }

    std::cout << "\nBuilding FAISS IndexFlatL2...\n";

    auto build_start = now();

    faiss::IndexFlatL2 index(base.d);
    index.add(base.n, base.data.data());

    auto build_end = now();
    double build_time_s = elapsed(build_start, build_end);

    std::cout << "Running exact search...\n";

    std::vector<float> distances(static_cast<size_t>(queries.n) * args.k);
    std::vector<faiss::idx_t> labels(static_cast<size_t>(queries.n) * args.k);

    auto search_start = now();

    index.search(
        queries.n,
        queries.data.data(),
        args.k,
        distances.data(),
        labels.data()
    );

    auto search_end = now();
    double search_time_s = elapsed(search_start, search_end);

    std::vector<std::vector<int>> predicted(queries.n, std::vector<int>(args.k));

    for (int i = 0; i < queries.n; i++) {
        for (int j = 0; j < args.k; j++) {
            predicted[i][j] = static_cast<int>(labels[static_cast<size_t>(i) * args.k + j]);
        }
    }

    double recall1 = compute_recall(predicted, ground_truth, 1);
    double recallk = compute_recall(predicted, ground_truth, args.k);

    double qps = static_cast<double>(queries.n) / search_time_s;
    double avg_latency_ms = (search_time_s / queries.n) * 1000.0;
    double index_size_mb =
        static_cast<double>(base.n) * base.d * sizeof(float) / (1024.0 * 1024.0);

    std::cout << "\n" << std::string(72, '=') << "\n";
    std::cout << "  RESULTS\n";
    std::cout << std::string(72, '=') << "\n";
    std::cout << "  Algorithm      : faiss_flat_l2\n";
    std::cout << "  Distance       : L2 / Euclidean\n";
    std::cout << "  Recall@1       : " << recall1 << "\n";
    std::cout << "  Recall@" << args.k << "       : " << recallk << "\n";
    std::cout << "  QPS            : " << qps << " queries/second\n";
    std::cout << "  Avg latency    : " << avg_latency_ms << " ms/query\n";
    std::cout << "  Build time     : " << build_time_s << " s\n";
    std::cout << "  Search time    : " << search_time_s << " s\n";
    std::cout << "  Index size     : " << index_size_mb << " MB\n";
    std::cout << "  Base vectors   : " << base.n << "\n";
    std::cout << "  Query vectors  : " << queries.n << "\n";
    std::cout << "  Dimension      : " << base.d << "\n";
    std::cout << std::string(72, '=') << "\n";

    if (recall1 < 0.999) {
        std::cerr << "\nWARNING: Recall@1 is below 1.0.\n";
        std::cerr << "Check that the HDF5 ground truth was generated using L2 distance.\n";
    }

    save_csv(
        "faiss_flat_l2",
        args.dataset,
        args.k,
        recall1,
        recallk,
        qps,
        avg_latency_ms,
        build_time_s,
        search_time_s,
        index_size_mb,
        base.n,
        queries.n,
        base.d
    );

    std::cout << "\nResults saved to ../results.csv\n";
    return 0;
}
