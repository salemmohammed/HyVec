 /*
 * faiss_ivf.cpp
 * -------------
 * Official FAISS IVF baseline using IndexIVFFlat.
 *
 * This implementation is used as the partition-based ANN baseline for
 * evaluating Hybrid Attribute-Spatial HNSW. It uses the official FAISS
 * implementation rather than a manual IVF implementation, ensuring that
 * benchmark results reflect a standard, optimized baseline.
 *
 * Distance metric:
 *   - L2 / Euclidean only
 *
 * Dataset format:
 *   - ANN-Benchmarks HDF5 file
 *   - Required datasets:
 *       /train      float32 matrix of database vectors
 *       /test       float32 matrix of query vectors
 *       /neighbors  int32 matrix of ground-truth neighbors
 *
 * Example:
 *   ./faiss_ivf --dataset ../data/sift-128-euclidean.hdf5 --k 10
 *   ./faiss_ivf --dataset ../data/sift-128-euclidean.hdf5 --k 10 --nlist 1000
 *
 * Compilation on macOS/Homebrew:
 *   g++ -O3 -std=c++17 faiss_ivf.cpp -o faiss_ivf \
 *     $(pkg-config --cflags --libs hdf5) \
 *     -I/opt/homebrew/opt/faiss/include \
 *     -L/opt/homebrew/opt/faiss/lib \
 *     -lfaiss
 *
 * Notes:
 *   - nlist controls the number of IVF partitions.
 *   - nprobe controls how many partitions are searched per query.
 *   - This program sweeps nprobe values and appends results to ../results.csv.
 */

#include <algorithm>
#include <chrono>
#include <ctime>
#include <fstream>
#include <iostream>
#include <set>
#include <string>
#include <vector>

#include <hdf5.h>

#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFFlat.h>

using Clock = std::chrono::high_resolution_clock;

static const int DEFAULT_K = 10;
static const int DEFAULT_NLIST = 1000;

static const std::vector<int> DEFAULT_NPROBE_LIST = {1, 5, 10, 20, 50, 100};

struct Matrix {
    std::vector<float> data;
    int n = 0;
    int d = 0;

    const float* row(int i) const {
        return data.data() + static_cast<size_t>(i) * d;
    }
};

struct Args {
    std::string dataset = "../data/sift-128-euclidean.hdf5";
    int k = DEFAULT_K;
    int nlist = DEFAULT_NLIST;
};

static double seconds_since(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}

static void read_float_matrix(hid_t file, const char* name, Matrix& out) {
    hid_t dset = H5Dopen2(file, name, H5P_DEFAULT);
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

static void read_int_matrix(
    hid_t file,
    const char* name,
    std::vector<std::vector<int>>& out
) {
    hid_t dset = H5Dopen2(file, name, H5P_DEFAULT);
    if (dset < 0) {
        std::cerr << "ERROR: Cannot open HDF5 dataset: " << name << "\n";
        std::exit(1);
    }

    hid_t space = H5Dget_space(dset);
    hsize_t dims[2];
    H5Sget_simple_extent_dims(space, dims, nullptr);

    int n = static_cast<int>(dims[0]);
    int cols = static_cast<int>(dims[1]);

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

static void load_hdf5(
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

    read_float_matrix(file, "train", base);
    read_float_matrix(file, "test", queries);
    read_int_matrix(file, "neighbors", ground_truth);

    H5Fclose(file);

    if (base.d != queries.d) {
        std::cerr << "ERROR: Base/query dimension mismatch.\n";
        std::exit(1);
    }

    std::cout << "  Base vectors : " << base.n << " x " << base.d << "\n";
    std::cout << "  Query vectors: " << queries.n << " x " << queries.d << "\n";
    std::cout << "  Ground truth : top-" << ground_truth[0].size()
              << " neighbors per query\n";
}

static double compute_recall_at_k(
    const std::vector<faiss::idx_t>& predicted,
    const std::vector<std::vector<int>>& ground_truth,
    int query_count,
    int result_k,
    int eval_k
) {
    double total = 0.0;

    for (int i = 0; i < query_count; i++) {
        std::set<int> truth;
        for (int j = 0; j < eval_k; j++) {
            truth.insert(ground_truth[i][j]);
        }

        int hits = 0;
        for (int j = 0; j < eval_k && j < result_k; j++) {
            int id = static_cast<int>(predicted[static_cast<size_t>(i) * result_k + j]);
            if (truth.find(id) != truth.end()) {
                hits++;
            }
        }

        total += static_cast<double>(hits) / eval_k;
    }

    return total / query_count;
}

static std::string timestamp_now() {
    auto now = std::chrono::system_clock::now();
    auto t = std::chrono::system_clock::to_time_t(now);

    char buf[32];
    std::strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", std::localtime(&t));
    return std::string(buf);
}

static void ensure_results_header(const std::string& path) {
    std::ifstream in(path);
    bool exists_and_has_content = in.good() && in.peek() != std::ifstream::traits_type::eof();
    in.close();

    if (!exists_and_has_content) {
        std::ofstream out(path, std::ios::app);
        out << "timestamp,machine,algorithm,dataset,distance,k,recall_at_1,"
            << "recall_at_k,qps,avg_latency_ms,build_time_s,search_time_s,"
            << "index_size_mb,num_base,num_queries,dimension,nlist,nprobe\n";
    }
}

static void save_csv(
    const std::string& algorithm,
    const std::string& dataset,
    int k,
    double recall1,
    double recallk,
    double qps,
    double avg_latency_ms,
    double build_time,
    double search_time,
    double index_size_mb,
    int num_base,
    int num_queries,
    int dimension,
    int nlist,
    int nprobe
) {
    std::string path = "../results.csv";
    ensure_results_header(path);

    std::ofstream out(path, std::ios::app);
    if (!out.is_open()) {
        std::cerr << "WARNING: Could not open " << path << " for writing.\n";
        return;
    }

    out << timestamp_now() << ","
        << "dgx-spark" << ","
        << algorithm << ","
        << dataset << ","
        << "l2" << ","
        << k << ","
        << recall1 << ","
        << recallk << ","
        << qps << ","
        << avg_latency_ms << ","
        << build_time << ","
        << search_time << ","
        << index_size_mb << ","
        << num_base << ","
        << num_queries << ","
        << dimension << ","
        << nlist << ","
        << nprobe << "\n";
}

static Args parse_args(int argc, char** argv) {
    Args args;

    for (int i = 1; i < argc; i++) {
        std::string current = argv[i];

        if (current == "--dataset" && i + 1 < argc) {
            args.dataset = argv[++i];
        } else if (current == "--k" && i + 1 < argc) {
            args.k = std::stoi(argv[++i]);
        } else if (current == "--nlist" && i + 1 < argc) {
            args.nlist = std::stoi(argv[++i]);
        } else if (current == "--help") {
            std::cout << "Usage:\n"
                      << "  ./faiss_ivf --dataset ../data/sift-128-euclidean.hdf5 "
                      << "--k 10 --nlist 1000\n";
            std::exit(0);
        }
    }

    return args;
}

int main(int argc, char** argv) {
    Args args = parse_args(argc, argv);

    std::cout << "========================================================================\n";
    std::cout << "  FAISS IVF Baseline — IndexIVFFlat\n";
    std::cout << "  Distance: L2 / Euclidean\n";
    std::cout << "  Dataset : " << args.dataset << "\n";
    std::cout << "  k       : " << args.k << "\n";
    std::cout << "  nlist   : " << args.nlist << "\n";
    std::cout << "========================================================================\n\n";

    std::cout << "Loading dataset...\n";
    Matrix base;
    Matrix queries;
    std::vector<std::vector<int>> ground_truth;
    load_hdf5(args.dataset, base, queries, ground_truth);

    if (args.nlist <= 0 || args.nlist > base.n) {
        std::cerr << "ERROR: nlist must be in range [1, num_base].\n";
        return 1;
    }

    std::cout << "\nBuilding FAISS IndexIVFFlat...\n";
    auto build_start = Clock::now();

    faiss::IndexFlatL2 quantizer(base.d);
    faiss::IndexIVFFlat index(&quantizer, base.d, args.nlist, faiss::METRIC_L2);

    std::cout << "Training IVF centroids...\n";
    index.train(base.n, base.data.data());

    std::cout << "Adding vectors to IVF index...\n";
    index.add(base.n, base.data.data());

    double build_time = seconds_since(build_start);

    double index_size_mb =
        (static_cast<double>(base.n) * base.d * sizeof(float)) / (1024.0 * 1024.0);

    std::cout << "Build time: " << build_time << " s\n";

    std::cout << "\nSweeping nprobe values...\n";
    std::cout << "  nprobe      Recall@1      Recall@" << args.k
              << "      QPS          Latency(ms)   Search(s)\n";
    std::cout << "  --------------------------------------------------------------------\n";

    for (int nprobe : DEFAULT_NPROBE_LIST) {
        if (nprobe > args.nlist) {
            continue;
        }

        index.nprobe = nprobe;

        std::vector<float> distances(static_cast<size_t>(queries.n) * args.k);
        std::vector<faiss::idx_t> labels(static_cast<size_t>(queries.n) * args.k);

        auto search_start = Clock::now();

        index.search(
            queries.n,
            queries.data.data(),
            args.k,
            distances.data(),
            labels.data()
        );

        double search_time = seconds_since(search_start);
        double qps = queries.n / search_time;
        double avg_latency_ms = (search_time / queries.n) * 1000.0;

        double recall1 = compute_recall_at_k(
            labels,
            ground_truth,
            queries.n,
            args.k,
            1
        );

        double recallk = compute_recall_at_k(
            labels,
            ground_truth,
            queries.n,
            args.k,
            args.k
        );

        std::cout << "  "
                  << nprobe << "           "
                  << recall1 << "       "
                  << recallk << "       "
                  << qps << "       "
                  << avg_latency_ms << "       "
                  << search_time << "\n";

        save_csv(
            "faiss_ivf_nprobe" + std::to_string(nprobe),
            args.dataset,
            args.k,
            recall1,
            recallk,
            qps,
            avg_latency_ms,
            build_time,
            search_time,
            index_size_mb,
            base.n,
            queries.n,
            base.d,
            args.nlist,
            nprobe
        );
    }

    std::cout << "\n========================================================================\n";
    std::cout << "  FAISS IVF benchmark complete\n";
    std::cout << "  Results saved to ../results.csv\n";
    std::cout << "========================================================================\n";

    return 0;
}
