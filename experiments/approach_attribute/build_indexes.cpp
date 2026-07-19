/*
 * build_indexes.cpp
 * -----------------
 * Step 2 of Clustered Attributed Vector Search.
 *
 * Reads cluster labels from generate_attributes.py output.
 * Builds one HierarchicalNSW index per cluster.
 * Saves each index to disk.
 *
 * Compile:
 *   g++ -O3 -std=c++17 build_indexes.cpp -o build_indexes -I../../hnswlib
 *
 * Run:
 *   ./build_indexes
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <map>
#include <chrono>
#include <filesystem>
#include "../../hnswlib/hnswlib/hnswlib.h"

// ─── CONFIG ──────────────────────────────────────────────────────────────────
const int   K               = 1000;    // number of clusters
const int   DIM             = 128;     // vector dimension
const int   M               = 16;     // HNSW M parameter
const int   EF_CONSTRUCTION = 200;    // HNSW ef_construction
const std::string BASE_PATH = "../sift/sift_base.fvecs";
const std::string LABELS_PATH    = "../sift/attr_labels.npy";
const std::string INDEX_DIR      = "../sift/indexes/";
// ─────────────────────────────────────────────────────────────────────────────


// Read .fvecs binary file
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


// Read .npy int array (simple format for 1D int arrays)
std::vector<int> read_npy_labels(const std::string& filename, int expected_size) {
    std::ifstream in(filename, std::ios::binary);
    if (!in) { std::cerr << "Cannot open " << filename << std::endl; exit(1); }

    // Skip numpy header (first 128 bytes for simple 1D int32 arrays)
    char header[128];
    in.read(header, 128);

    std::vector<int> labels(expected_size);
    in.read((char*)labels.data(), expected_size * sizeof(int));
    return labels;
}


int main() {
    std::cout << "================================================" << std::endl;
    std::cout << "  Building Per-Cluster HNSW Indexes" << std::endl;
    std::cout << "================================================" << std::endl;

    // 1. Create output directory
    std::filesystem::create_directories(INDEX_DIR);

    // 2. Load base vectors
    std::cout << "\nLoading base vectors..." << std::endl;
    auto base = read_fvecs(BASE_PATH);
    int num_base = base.size();
    std::cout << "Loaded: " << num_base << " vectors x " << DIM << " dimensions" << std::endl;

    // 3. Load cluster labels
    std::cout << "Loading cluster labels..." << std::endl;
    auto labels = read_npy_labels(LABELS_PATH, num_base);
    std::cout << "Loaded: " << labels.size() << " labels" << std::endl;

    // 4. Group vectors by cluster
    std::cout << "\nGrouping vectors by cluster..." << std::endl;
    std::map<int, std::vector<int>> cluster_to_ids;
    for (int i = 0; i < num_base; i++) {
        cluster_to_ids[labels[i]].push_back(i);
    }
    std::cout << "Number of clusters: " << cluster_to_ids.size() << std::endl;

    // 5. Build one HNSW index per cluster
    std::cout << "\nBuilding " << K << " HNSW indexes..." << std::endl;
    std::cout << "M=" << M << ", ef_construction=" << EF_CONSTRUCTION << std::endl;

    auto t_start = std::chrono::high_resolution_clock::now();

    hnswlib::L2Space space(DIM);
    int built = 0;

    for (auto& [cluster_id, ids] : cluster_to_ids) {
        int cluster_size = ids.size();

        // Build index for this cluster
        hnswlib::HierarchicalNSW<float>* index =
            new hnswlib::HierarchicalNSW<float>(&space, cluster_size, M, EF_CONSTRUCTION);

        // Add all vectors in this cluster
        for (int id : ids) {
            index->addPoint(base[id].data(), id);  // use original global ID as label
        }

        // Save index to disk
        std::string index_path = INDEX_DIR + "index_" + std::to_string(cluster_id) + ".bin";
        index->saveIndex(index_path);

        delete index;
        built++;

        // Progress
        if (built % 100 == 0 || built == 1) {
            std::cout << "Built " << built << "/" << K
                      << " indexes (cluster " << cluster_id
                      << " has " << cluster_size << " vectors)" << std::endl;
        }
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    double build_time = std::chrono::duration<double>(t_end - t_start).count();

    std::cout << "\n=== BUILD RESULTS ===" << std::endl;
    std::cout << "Total indexes built: " << built << std::endl;
    std::cout << "Total build time:    " << build_time << "s" << std::endl;
    std::cout << "Indexes saved to:    " << INDEX_DIR << std::endl;
    std::cout << "\nDone! Ready for Step 3: search" << std::endl;

    return 0;
}
