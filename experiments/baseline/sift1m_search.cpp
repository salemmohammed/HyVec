#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include "../hnswlib/hnswlib/hnswlib.h"

// Read .fvecs file
std::vector<std::vector<float>> read_fvecs(const std::string& filename) {
    std::ifstream in(filename, std::ios::binary);
    if (!in) { std::cerr << "Cannot open " << filename << std::endl; exit(1); }

    std::vector<std::vector<float>> data;
    int dim;
    while (in.read((char*)&dim, sizeof(int))) {
        std::vector<float> vec(dim);
        in.read((char*)vec.data(), dim * sizeof(float));
        data.push_back(vec);
    }
    return data;
}

// Read .ivecs file (ground truth)
std::vector<std::vector<int>> read_ivecs(const std::string& filename) {
    std::ifstream in(filename, std::ios::binary);
    std::vector<std::vector<int>> data;
    int dim;
    while (in.read((char*)&dim, sizeof(int))) {
        std::vector<int> vec(dim);
        in.read((char*)vec.data(), dim * sizeof(int));
        data.push_back(vec);
    }
    return data;
}

int main() {
    // 1. Load data
    std::cout << "Loading SIFT1M..." << std::endl;
    auto base    = read_fvecs("sift/sift_base.fvecs");
    auto queries = read_fvecs("sift/sift_query.fvecs");
    auto gt      = read_ivecs("sift/sift_groundtruth.ivecs");

    int dim = base[0].size();
    int num_base    = base.size();
    int num_queries = queries.size();
    int K = 10;

    std::cout << "Base vectors:  " << num_base << " x " << dim << std::endl;
    std::cout << "Query vectors: " << num_queries << std::endl;

    // 2. Build HNSW index
    std::cout << "Building index..." << std::endl;
    hnswlib::L2Space space(dim);
    hnswlib::HierarchicalNSW<float> index(&space, num_base, 16, 200);

    auto t1 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_base; i++) {
        index.addPoint(base[i].data(), i);
        if (i % 100000 == 0) std::cout << "Added " << i << " vectors\n";
    }
    auto t2 = std::chrono::high_resolution_clock::now();
    double build_time = std::chrono::duration<double>(t2 - t1).count();
    std::cout << "Build time: " << build_time << "s" << std::endl;

    // 3. Search
    std::cout << "Searching..." << std::endl;
    index.setEf(50);
    int correct = 0;

    auto t3 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_queries; i++) {
        auto result = index.searchKnn(queries[i].data(), K);

        // Check recall against ground truth
        std::vector<int> found;
        while (!result.empty()) {
            found.push_back(result.top().second);
            result.pop();
        }
        // Check if true nearest neighbor is in results
        if (std::find(found.begin(), found.end(), gt[i][0]) != found.end())
            correct++;
    }
    auto t4 = std::chrono::high_resolution_clock::now();
    double search_time = std::chrono::duration<double>(t4 - t3).count();

    // 4. Print results
    std::cout << "\n=== RESULTS ===" << std::endl;
    std::cout << "Recall@1:    " << (float)correct / num_queries << std::endl;
    std::cout << "Search time: " << search_time << "s" << std::endl;
    std::cout << "QPS:         " << num_queries / search_time << std::endl;

    return 0;
}
