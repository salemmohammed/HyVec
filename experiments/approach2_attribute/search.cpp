#include <iostream>
#include <fstream>
#include <vector>
#include <map>
#include <chrono>
#include <algorithm>
#include <limits>
#include <cmath>
#include <iomanip>
#include "../../hnswlib/hnswlib/hnswlib.h"

// ─── CONFIG ──────────────────────────────────────────────────────────────────
const int   K_RESULTS       = 10;
const int   K_CLUSTERS      = 1000;
const int   DIM             = 128;
const int   EF_SEARCH       = 50;
const std::string QUERY_PATH       = "../sift/sift_query.fvecs";
const std::string GROUNDTRUTH_PATH = "../sift/sift_groundtruth.ivecs";
const std::string CENTROIDS_PATH   = "../sift/attr_centroids.npy";
const std::string INDEX_DIR        = "../sift/indexes/";
// ─────────────────────────────────────────────────────────────────────────────

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

std::vector<std::vector<float>> read_npy_centroids(const std::string& filename, int K, int dim) {
    std::ifstream in(filename, std::ios::binary);
    if (!in) { std::cerr << "Cannot open " << filename << std::endl; exit(1); }
    char header[128];
    in.read(header, 128);
    std::vector<std::vector<float>> centroids(K, std::vector<float>(dim));
    for (int i = 0; i < K; i++)
        in.read((char*)centroids[i].data(), dim * sizeof(float));
    return centroids;
}

std::vector<int> find_nearest_centroids(
    const std::vector<float>& query,
    const std::vector<std::vector<float>>& centroids,
    int top_n)
{
    std::vector<std::pair<float, int>> dists;
    for (int i = 0; i < (int)centroids.size(); i++) {
        float dist = 0;
        for (int j = 0; j < DIM; j++) {
            float d = query[j] - centroids[i][j];
            dist += d * d;
        }
        dists.push_back({dist, i});
    }
    std::sort(dists.begin(), dists.end());
    std::vector<int> result;
    for (int i = 0; i < top_n; i++) result.push_back(dists[i].second);
    return result;
}

int main() {
    std::cout << "================================================" << std::endl;
    std::cout << "  Clustered Search — TOP_CLUSTERS Sweep" << std::endl;
    std::cout << "================================================" << std::endl;

    auto queries   = read_fvecs(QUERY_PATH);
    auto gt        = read_ivecs(GROUNDTRUTH_PATH);
    auto centroids = read_npy_centroids(CENTROIDS_PATH, K_CLUSTERS, DIM);

    std::cout << "Loading " << K_CLUSTERS << " indexes..." << std::endl;
    hnswlib::L2Space space(DIM);
    std::map<int, hnswlib::HierarchicalNSW<float>*> indexes;
    for (int c = 0; c < K_CLUSTERS; c++) {
        std::string path = INDEX_DIR + "index_" + std::to_string(c) + ".bin";
        std::ifstream f(path);
        if (f.good()) {
            indexes[c] = new hnswlib::HierarchicalNSW<float>(&space, path);
            indexes[c]->setEf(EF_SEARCH);
        }
    }
    std::cout << "Loaded " << indexes.size() << " indexes.\n" << std::endl;

    std::cout << std::left
              << std::setw(15) << "TOP_CLUSTERS"
              << std::setw(12) << "Recall@1"
              << std::setw(12) << "QPS"
              << std::setw(12) << "Time(s)"
              << std::endl;
    std::cout << std::string(51, '-') << std::endl;

    for (int top : {1, 3, 5, 10, 20, 50}) {
        int correct = 0;
        int num_queries = queries.size();
        auto t_start = std::chrono::high_resolution_clock::now();

        for (int i = 0; i < num_queries; i++) {
            auto cluster_ids = find_nearest_centroids(queries[i], centroids, top);
            std::vector<std::pair<float, int>> all_results;

            for (int cid : cluster_ids) {
                if (indexes.find(cid) == indexes.end()) continue;
                auto result = indexes[cid]->searchKnn(queries[i].data(), K_RESULTS);
                while (!result.empty()) {
                    all_results.push_back({result.top().first, result.top().second});
                    result.pop();
                }
            }

            std::sort(all_results.begin(), all_results.end());
            std::vector<int> found;
            for (int j = 0; j < K_RESULTS && j < (int)all_results.size(); j++)
                found.push_back(all_results[j].second);

            if (std::find(found.begin(), found.end(), gt[i][0]) != found.end())
                correct++;
        }

        auto t_end = std::chrono::high_resolution_clock::now();
        double search_time = std::chrono::duration<double>(t_end - t_start).count();

        std::cout << std::left
                  << std::setw(15) << top
                  << std::setw(12) << std::fixed << std::setprecision(4)
                  << (float)correct / num_queries
                  << std::setw(12) << std::setprecision(1)
                  << num_queries / search_time
                  << std::setw(12) << std::setprecision(3)
                  << search_time
                  << std::endl;
    }

    std::cout << std::string(51, '-') << std::endl;
    

    for (auto& [c, idx] : indexes) delete idx;
    return 0;
}