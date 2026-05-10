/*
 * search.cpp
 * ----------
 * Step 3 of Clustered Attributed Vector Search.
 *
 * For each query:
 *   1. Find nearest centroid → cluster ID (the attribute)
 *   2. Search only that cluster's HNSW index
 *   3. Return top K results
 *
 * Compares recall and QPS against the baseline (sift1m_search).
 *
 * Compile:
 *   g++ -O3 -std=c++17 search.cpp -o search -I../../hnswlib
 *
 * Run:
 *   ./search
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <map>
#include <chrono>
#include <algorithm>
#include <limits>
#include <cmath>
#include "../../hnswlib/hnswlib/hnswlib.h"

// ─── CONFIG ──────────────────────────────────────────────────────────────────
const int   K_RESULTS       = 10;     // number of nearest neighbors to return
const int   K_CLUSTERS      = 1000;   // number of clusters
const int   DIM             = 128;    // vector dimension
const int   EF_SEARCH       = 50;    // HNSW ef for search
const int   TOP_CLUSTERS    = 1;     // search top-N clusters (1=fast, 3=better recall)
const std::string QUERY_PATH      = "../sift/sift_query.fvecs";
const std::string GROUNDTRUTH_PATH = "../sift/sift_groundtruth.ivecs";
const std::string CENTROIDS_PATH  = "../sift/attr_centroids.npy";
const std::string INDEX_DIR       = "../sift/indexes/";
// ─────────────────────────────────────────────────────────────────────────────


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
    if (!in) { std::cerr << "Cannot open " << filename << std::endl; exit(1); }
    std::vector<std::vector<int>> data;
    int dim;
    while (in.read((char*)&dim, sizeof(int))) {
        std::vector<int> vec(dim);
        in.read((char*)vec.data(), dim * sizeof(int));
        data.push_back(vec);
    }
    return data;
}


// Read centroids from .npy file (1000 × 128 float32)
std::vector<std::vector<float>> read_npy_centroids(const std::string& filename, int K, int dim) {
    std::ifstream in(filename, std::ios::binary);
    if (!in) { std::cerr << "Cannot open " << filename << std::endl; exit(1); }

    // Skip numpy header
    char header[128];
    in.read(header, 128);

    std::vector<std::vector<float>> centroids(K, std::vector<float>(dim));
    for (int i = 0; i < K; i++) {
        in.read((char*)centroids[i].data(), dim * sizeof(float));
    }
    return centroids;
}


// L2 distance between two vectors
float l2_distance(const std::vector<float>& a, const std::vector<float>& b) {
    float dist = 0.0f;
    for (int i = 0; i < (int)a.size(); i++) {
        float diff = a[i] - b[i];
        dist += diff * diff;
    }
    return dist;  // squared L2 (no need for sqrt for comparison)
}


// Find top-N nearest centroids to query
std::vector<int> find_nearest_centroids(
    const std::vector<float>& query,
    const std::vector<std::vector<float>>& centroids,
    int top_n)
{
    std::vector<std::pair<float, int>> dists;
    for (int i = 0; i < (int)centroids.size(); i++) {
        float d = l2_distance(query, centroids[i]);
        dists.push_back({d, i});
    }
    std::sort(dists.begin(), dists.end());

    std::vector<int> result;
    for (int i = 0; i < top_n && i < (int)dists.size(); i++) {
        result.push_back(dists[i].second);
    }
    return result;
}


int main() {
    std::cout << "================================================" << std::endl;
    std::cout << "  Clustered Attributed Vector Search" << std::endl;
    std::cout << "================================================" << std::endl;
    std::cout << "K_RESULTS=" << K_RESULTS
              << " K_CLUSTERS=" << K_CLUSTERS
              << " TOP_CLUSTERS=" << TOP_CLUSTERS
              << " EF=" << EF_SEARCH << std::endl;

    // 1. Load queries and ground truth
    std::cout << "\nLoading queries..." << std::endl;
    auto queries = read_fvecs(QUERY_PATH);
    auto gt      = read_ivecs(GROUNDTRUTH_PATH);
    std::cout << "Queries: " << queries.size() << std::endl;

    // 2. Load centroids
    std::cout << "Loading centroids..." << std::endl;
    auto centroids = read_npy_centroids(CENTROIDS_PATH, K_CLUSTERS, DIM);
    std::cout << "Centroids: " << centroids.size() << std::endl;

    // 3. Load all HNSW indexes
    std::cout << "\nLoading " << K_CLUSTERS << " HNSW indexes..." << std::endl;
    hnswlib::L2Space space(DIM);
    std::map<int, hnswlib::HierarchicalNSW<float>*> indexes;

    for (int c = 0; c < K_CLUSTERS; c++) {
        std::string path = INDEX_DIR + "index_" + std::to_string(c) + ".bin";
        std::ifstream f(path);
        if (f.good()) {
            indexes[c] = new hnswlib::HierarchicalNSW<float>(&space, path);
            indexes[c]->setEf(EF_SEARCH);
        }
        if (c % 200 == 0) std::cout << "Loaded " << c << "/" << K_CLUSTERS << " indexes\n";
    }
    std::cout << "Total indexes loaded: " << indexes.size() << std::endl;

    // 4. Search
    std::cout << "\nSearching " << queries.size() << " queries..." << std::endl;
    int correct = 0;
    int num_queries = queries.size();

    auto t_start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < num_queries; i++) {
        // Find nearest cluster(s)
        auto cluster_ids = find_nearest_centroids(queries[i], centroids, TOP_CLUSTERS);

        // Search each selected cluster
        std::vector<std::pair<float, int>> all_results;
        for (int cid : cluster_ids) {
            if (indexes.find(cid) == indexes.end()) continue;
            auto result = indexes[cid]->searchKnn(queries[i].data(), K_RESULTS);
            while (!result.empty()) {
                all_results.push_back({result.top().first, result.top().second});
                result.pop();
            }
        }

        // Sort and take top K_RESULTS
        std::sort(all_results.begin(), all_results.end());
        std::vector<int> found;
        for (int j = 0; j < K_RESULTS && j < (int)all_results.size(); j++) {
            found.push_back(all_results[j].second);
        }

        // Check recall: did we find the true nearest neighbor?
        if (std::find(found.begin(), found.end(), gt[i][0]) != found.end()) {
            correct++;
        }
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    double search_time = std::chrono::duration<double>(t_end - t_start).count();

    // 5. Print results
    std::cout << "\n=== CLUSTERED SEARCH RESULTS ===" << std::endl;
    std::cout << "Recall@1:    " << (float)correct / num_queries << std::endl;
    std::cout << "Search time: " << search_time << "s" << std::endl;
    std::cout << "QPS:         " << num_queries / search_time << std::endl;

    std::cout << "\n=== COMPARISON ===" << std::endl;
    std::cout << "Baseline Recall@1:  0.9686" << std::endl;
    std::cout << "Baseline QPS:       7443" << std::endl;
    std::cout << "Clustered Recall@1: " << (float)correct / num_queries << std::endl;
    std::cout << "Clustered QPS:      " << num_queries / search_time << std::endl;

    // Cleanup
    for (auto& [c, idx] : indexes) delete idx;

    return 0;
}
